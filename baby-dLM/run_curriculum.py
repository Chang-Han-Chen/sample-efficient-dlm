#!/usr/bin/env python3
"""
run_curriculum.py — Curriculum runner for block-diffusion scaling experiments (§6.4).

Executes a multi-stage curriculum by launching train.py once per stage,
warm-starting each stage from the previous stage's checkpoint (weights only;
optimizer and LR schedule reset at each transition).

Supports both optimizer families:
  - adamw:   uses the calibrated scalar LR for each (model_family, size) pair
  - normuon: uses the calibrated (adam_mult, matrix_mult) config, with per-group
             LRs realized from model width at runtime

Usage:
    # Run a specific curriculum by name:
    python run_curriculum.py --curriculum c1_geometric --size 50M --budget 1e18

    # Run with normuon optimizer:
    python run_curriculum.py --curriculum c1_geometric --size 50M --budget 1e18 \
        --optimizer normuon

    # Dry-run (print commands without executing):
    python run_curriculum.py --curriculum baseline_ar --size 98M --budget 2e18 --dry-run

    # List all available curriculum names:
    python run_curriculum.py --list

    # Run a curriculum-0 sweep variant:
    python run_curriculum.py --curriculum c0_plain_p20 --size 50M --budget 1e18
"""

import argparse
import json
import os
import pickle
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import experiment_config as ec

# ---------------------------------------------------------------
# Curriculum registry: name -> Curriculum object
# ---------------------------------------------------------------

CURRICULUM_REGISTRY: Dict[str, ec.Curriculum] = {}

for cur in ec.ALL_CURRICULA:
    CURRICULUM_REGISTRY[cur.name] = cur


def list_curricula():
    """Print all registered curriculum names and descriptions."""
    print(f"\n{'='*70}")
    print("Available curricula")
    print(f"{'='*70}")
    print(f"  {'name':<30s}  {'stages':>6s}  description")
    print(f"  {'-'*30:<30s}  {'-'*6:>6s}  {'-'*30}")
    for name, cur in sorted(CURRICULUM_REGISTRY.items()):
        print(f"  {name:<30s}  {len(cur.stages):>6d}  {cur.description}")
    print()


# ---------------------------------------------------------------
# Stdout parsing (same regex as run_lr_sweep.py)
# ---------------------------------------------------------------

# Matches mid-training log lines:
#   step 100 | tok_epoch 0.03 | loss 7.1234 | grad_norm 1.23 | lr 0.001000
_STEP_RE = re.compile(
    r"step\s+(\d+)\s+\|.*?"
    r"loss\s+([\d.]+)\s+\|.*?"
    r"grad_norm\s+([\d.eE+\-]+)\s+\|.*?"
    r"lr\s+([\d.eE+\-]+)"
)

# Matches both mid-training eval lines and the forced final eval line:
#   step 300 | tok_epoch 0.09 | train 6.5400 | val 6.7800 | lr 0.001000
#   step 1000 (final) | train 6.5400 | val 6.7800
_VAL_RE = re.compile(
    r"step\s+(\d+)\s*(?:\(final\))?\s*\|.*?"
    r"val\s+([\d.]+)"
)

# Matches the forced final eval line for train loss:
#   step 1000 (final) | train 6.5400 | val 6.7800
_FINAL_EVAL_RE = re.compile(
    r"step\s+(\d+)\s*\(final\)\s*\|.*?"
    r"train\s+([\d.]+)"
)


@dataclass
class StageResult:
    """Results from a single curriculum stage."""
    stage_index: int
    model_family: str
    block_len: Optional[int]
    flop_frac: float
    max_iters: int
    checkpoint_path: str
    train_losses: List[Tuple[int, float]] = field(default_factory=list)
    val_losses: List[Tuple[int, float]] = field(default_factory=list)
    grad_norms: List[Tuple[int, float]] = field(default_factory=list)
    wall_time_seconds: float = 0.0
    return_code: int = 0


@dataclass
class CurriculumResult:
    """Aggregated results from a full curriculum run."""
    curriculum_name: str
    size: str
    total_budget: float
    optimizer_family: str
    lr_config: dict  # per-family: {"ar": {"lr": 0.01}, "bd3lm": {"lr": 0.003}, ...}
    stages: List[StageResult] = field(default_factory=list)

    @property
    def final_val_loss(self) -> Optional[float]:
        """Validation loss from the last stage's last eval point."""
        if self.stages and self.stages[-1].val_losses:
            return self.stages[-1].val_losses[-1][1]
        return None

    @property
    def final_train_loss(self) -> Optional[float]:
        """Train loss from the last stage's last logged step."""
        if self.stages and self.stages[-1].train_losses:
            return self.stages[-1].train_losses[-1][1]
        return None

    @property
    def total_wall_time(self) -> float:
        return sum(s.wall_time_seconds for s in self.stages)

    def summary_dict(self) -> dict:
        """Serializable summary for JSON output."""
        return {
            "curriculum": self.curriculum_name,
            "size": self.size,
            "budget": self.total_budget,
            "optimizer": self.optimizer_family,
            "lr_config": self.lr_config,
            "final_train_loss": self.final_train_loss,
            "final_val_loss": self.final_val_loss,
            "total_wall_time_s": round(self.total_wall_time, 1),
            "num_stages": len(self.stages),
            "stages": [
                {
                    "index": s.stage_index,
                    "model": s.model_family,
                    "block_len": s.block_len,
                    "flop_frac": s.flop_frac,
                    "max_iters": s.max_iters,
                    "final_train_loss": s.train_losses[-1][1] if s.train_losses else None,
                    "final_val_loss": s.val_losses[-1][1] if s.val_losses else None,
                    "wall_time_s": round(s.wall_time_seconds, 1),
                    "return_code": s.return_code,
                }
                for s in self.stages
            ],
        }


# ---------------------------------------------------------------
# Stage execution
# ---------------------------------------------------------------

def parse_stage_stdout(text: str) -> Tuple[
    List[Tuple[int, float]],   # train_losses: [(step, loss), ...]
    List[Tuple[int, float]],   # val_losses: [(step, loss), ...]
    List[Tuple[int, float]],   # grad_norms: [(step, grad_norm), ...]
]:
    """
    Parse train.py stdout to extract per-step metrics.

    Handles two distinct log formats:
      - Mid-training:   step N | ... | loss X | grad_norm Y | lr Z
      - Final eval:     step N (final) | train X | val Y
    """
    train_losses = []
    grad_norms = []
    val_losses = []

    # Mid-training step logs (contain loss + grad_norm)
    for m in _STEP_RE.finditer(text):
        step = int(m.group(1))
        loss = float(m.group(2))
        gn = float(m.group(3))
        train_losses.append((step, loss))
        grad_norms.append((step, gn))

    # Validation losses from both mid-training evals and the forced final eval.
    # The updated regex accepts the optional "(final)" suffix after the step number.
    for m in _VAL_RE.finditer(text):
        step = int(m.group(1))
        val = float(m.group(2))
        val_losses.append((step, val))

    # The forced final eval uses "train X.XXXX" (not "loss X.XXXX"), so _STEP_RE
    # won't match it.  Pick it up with _FINAL_EVAL_RE to avoid a stale last entry.
    for m in _FINAL_EVAL_RE.finditer(text):
        step = int(m.group(1))
        loss = float(m.group(2))
        train_losses.append((step, loss))

    return train_losses, val_losses, grad_norms


def compute_stage_steps(
    stage: ec.CurriculumStage,
    size: str,
    total_budget: float,
) -> int:
    """Compute the number of training steps for a curriculum stage."""
    tokens_per_step = (ec.CLIMBMIX_TOKENS_PER_STEP
                       if size in ec.CLIMBMIX_MODEL_SIZES
                       else ec.ISOFLOP_TOKENS_PER_STEP)
    stage_budget = total_budget * stage.flop_frac
    N = ec.non_embedding_params(size)
    C = ec.flop_multiplier(stage.model_family)
    stage_tokens = int(stage_budget / (C * N))
    return max(1, stage_tokens // tokens_per_step)


def run_stage(
    stage: ec.CurriculumStage,
    stage_index: int,
    size: str,
    out_dir: str,
    *,
    total_budget: Optional[float] = None,
    total_steps: Optional[int] = None,
    resume_from: Optional[str] = None,
    optimizer: str = "adamw",
    dry_run: bool = False,
    eval_interval: int = 300,
    eval_iters: int = 50,
    save_interval: int = 1000,
) -> Optional[StageResult]:
    """
    Launch train.py for a single curriculum stage.

    Provide exactly one of *total_budget* (FLOP-based step calculation) or
    *total_steps* (direct step count, split by flop_frac).

    Returns a StageResult on success, or None on failure / dry-run.
    """
    if total_steps is not None:
        stage_steps = max(1, round(total_steps * stage.flop_frac))
    else:
        stage_steps = compute_stage_steps(stage, size, total_budget)

    stage_dir = os.path.join(
        out_dir,
        f"stage_{stage_index}_{stage.model_family}"
        + (f"_bl{stage.block_len}" if stage.block_len is not None else ""),
    )
    os.makedirs(stage_dir, exist_ok=True)

    ckpt_path = os.path.join(stage_dir, "ckpt.pt")

    # Build the train.py command.
    # When total_steps is given, use build_command directly (no FLOP accounting).
    # When total_budget is given, use build_stage_command (FLOP → steps).
    try:
        if total_steps is not None:
            cmd = ec.build_command(
                model=stage.model_family,
                size=size,
                out_dir=stage_dir,
                max_iters=stage_steps,
                block_len=stage.block_len,
                warmup_stable=True,
                resume_from=resume_from,
                optimizer=optimizer,
                eval_interval=eval_interval,
                eval_iters=eval_iters,
                save_interval=save_interval,
            )
        else:
            cmd = ec.build_stage_command(
                stage=stage,
                size=size,
                total_budget=total_budget,
                out_dir=stage_dir,
                resume_from=resume_from,
                optimizer=optimizer,
                eval_interval=eval_interval,
                eval_iters=eval_iters,
                save_interval=save_interval,
            )
    except ValueError as e:
        if dry_run:
            # In dry-run, missing calibration is non-fatal — show what we can.
            block_desc = f"block_len={stage.block_len}" if stage.block_len else "—"
            print(f"\n  Stage {stage_index}: {stage.model_family} | {block_desc} | "
                  f"flop_frac={stage.flop_frac:.2f} | steps={stage_steps}")
            print(f"    [dry-run] WARNING: {e}")
            return StageResult(
                stage_index=stage_index,
                model_family=stage.model_family,
                block_len=stage.block_len,
                flop_frac=stage.flop_frac,
                max_iters=stage_steps,
                checkpoint_path=ckpt_path,
            )
        raise

    # Print stage header
    block_desc = f"block_len={stage.block_len}" if stage.block_len else "—"
    print(f"\n  Stage {stage_index}: {stage.model_family} | {block_desc} | "
          f"flop_frac={stage.flop_frac:.2f} | steps={stage_steps}")
    if resume_from:
        print(f"    resume_from: {resume_from}")

    if dry_run:
        print(f"    [dry-run] cmd: {' '.join(cmd)}")
        return StageResult(
            stage_index=stage_index,
            model_family=stage.model_family,
            block_len=stage.block_len,
            flop_frac=stage.flop_frac,
            max_iters=stage_steps,
            checkpoint_path=ckpt_path,
        )

    t0 = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    wall_time = time.time() - t0

    train_losses, val_losses, grad_norms = parse_stage_stdout(result.stdout)

    stage_result = StageResult(
        stage_index=stage_index,
        model_family=stage.model_family,
        block_len=stage.block_len,
        flop_frac=stage.flop_frac,
        max_iters=stage_steps,
        checkpoint_path=ckpt_path,
        train_losses=train_losses,
        val_losses=val_losses,
        grad_norms=grad_norms,
        wall_time_seconds=wall_time,
        return_code=result.returncode,
    )

    if result.returncode != 0:
        print(f"    FAILED (exit {result.returncode}, {wall_time:.0f}s)")
        if result.stderr:
            for line in result.stderr.strip().splitlines()[-5:]:
                print(f"    stderr: {line}")
        return stage_result  # return partial result so we can inspect

    # Report results
    t_loss = f"{train_losses[-1][1]:.4f}" if train_losses else "N/A"
    v_loss = f"{val_losses[-1][1]:.4f}" if val_losses else "N/A"
    print(f"    OK ({wall_time:.0f}s) | train_loss={t_loss} | val_loss={v_loss} | "
          f"{len(train_losses)} logged steps")

    return stage_result


# ---------------------------------------------------------------
# Full curriculum execution
# ---------------------------------------------------------------

def resolve_lr_config(optimizer: str, model: str, size: str) -> dict:
    """
    Look up the calibrated LR config for a (optimizer, model, size) triple.

    For adamw: returns {"lr": float}.
    For normuon: returns {"adam_mult": float, "matrix_mult": float}.

    Raises ValueError if the config has not been calibrated yet.
    """
    if optimizer == "adamw":
        lr = ec.get_optimal_lr(model, size)
        if lr is None:
            raise ValueError(
                f"No calibrated AdamW LR for ({model}, {size}). "
                "Run run_lr_sweep.py first (§6.3)."
            )
        return {"lr": lr}
    elif optimizer == "normuon":
        cfg = ec.get_optimal_normuon(model, size)
        if cfg is None:
            raise ValueError(
                f"No calibrated NorMuon config for ({model}, {size}). "
                "Run run_lr_sweep.py --optimizer normuon first (§6.3)."
            )
        return dict(cfg)  # {"adam_mult": ..., "matrix_mult": ...}
    else:
        raise ValueError(f"Unknown optimizer family: {optimizer!r}")


def _collect_lr_configs(
    curriculum: ec.Curriculum,
    size: str,
    optimizer: str,
) -> Dict[str, dict]:
    """
    Collect the LR config for every distinct model_family used in the
    curriculum.  Returns {model_family: lr_config_dict}.

    For curricula that mix model families (e.g. AR -> BD3-LM), each
    family may have its own calibrated config.
    """
    families_seen = set()
    configs = {}
    for stage in curriculum.stages:
        fam = stage.model_family
        if fam not in families_seen:
            families_seen.add(fam)
            configs[fam] = resolve_lr_config(optimizer, fam, size)
    return configs


def run_curriculum(
    curriculum: ec.Curriculum,
    size: str,
    out_dir: str,
    *,
    total_budget: Optional[float] = None,
    total_steps: Optional[int] = None,
    optimizer: str = "adamw",
    dry_run: bool = False,
    eval_interval: int = 300,
    eval_iters: int = 50,
    save_interval: int = 1000,
) -> Optional[CurriculumResult]:
    """
    Run all stages of a curriculum sequentially.

    Provide exactly one of *total_budget* (FLOP-based step calculation) or
    *total_steps* (direct: each stage gets round(total_steps * flop_frac) steps).

    Each stage warm-starts from the previous stage's checkpoint (weights
    only).  The optimizer and LR schedule are reset at each stage
    transition, as specified in PLAN.md §3.

    Returns a CurriculumResult on success (even partial), None on dry-run.
    """
    if total_budget is None and total_steps is None:
        raise ValueError("Provide either total_budget or total_steps")

    # Validate compatibility
    ec.check_size_curriculum_compat(size, curriculum)

    # Resolve LR configs for all model families in the curriculum.
    # We validate calibration upfront to fail fast before launching any stages.
    # In dry-run mode, missing calibration is non-fatal (we show placeholder commands).
    try:
        lr_configs = _collect_lr_configs(curriculum, size, optimizer)
    except ValueError as e:
        if dry_run:
            print(f"  WARNING: {e}")
            print(f"  (dry-run continues with placeholder LR configs)\n")
            lr_configs = {}
        else:
            raise

    # Store *all* per-family LR configs so mixed-family curricula
    # (e.g. AR → BD3-LM) don't lose the earlier-stage settings.

    scope_str = (f"steps={total_steps}" if total_steps is not None
                 else f"budget={total_budget:.2e}")

    print(f"\n{'='*70}")
    print(f"Curriculum: {curriculum.name}")
    print(f"  {curriculum.description}")
    print(f"  size={size}  {scope_str}  optimizer={optimizer}")
    print(f"  stages={len(curriculum.stages)}  out_dir={out_dir}")
    for fam, cfg in lr_configs.items():
        print(f"  LR config [{fam}]: {cfg}")
    print(f"{'='*70}")

    if dry_run:
        # Walk through stages to show commands; chain checkpoint paths.
        prev_ckpt = None
        for i, stage in enumerate(curriculum.stages):
            sr = run_stage(
                stage, i, size, out_dir,
                total_budget=total_budget,
                total_steps=total_steps,
                resume_from=prev_ckpt,
                optimizer=optimizer,
                dry_run=True,
                eval_interval=eval_interval,
                eval_iters=eval_iters,
                save_interval=save_interval,
            )
            if sr is not None:
                prev_ckpt = sr.checkpoint_path
        return None

    cur_result = CurriculumResult(
        curriculum_name=curriculum.name,
        size=size,
        total_budget=total_budget or 0.0,
        optimizer_family=optimizer,
        lr_config=lr_configs,
    )

    prev_ckpt = None

    for i, stage in enumerate(curriculum.stages):
        stage_result = run_stage(
            stage, i, size, out_dir,
            total_budget=total_budget,
            total_steps=total_steps,
            resume_from=prev_ckpt,
            optimizer=optimizer,
            dry_run=False,
            eval_interval=eval_interval,
            eval_iters=eval_iters,
            save_interval=save_interval,
        )

        if stage_result is None:
            print(f"\n  Stage {i} returned no result — aborting curriculum.")
            break

        cur_result.stages.append(stage_result)

        if stage_result.return_code != 0:
            print(f"\n  Stage {i} failed (exit {stage_result.return_code}) "
                  f"— aborting curriculum.")
            break

        # Chain: next stage warm-starts from this stage's checkpoint.
        prev_ckpt = stage_result.checkpoint_path

    # Final summary
    print(f"\n{'='*70}")
    print(f"Curriculum {curriculum.name} complete")
    print(f"  stages completed: {len(cur_result.stages)}/{len(curriculum.stages)}")
    if cur_result.final_train_loss is not None:
        print(f"  final train loss: {cur_result.final_train_loss:.4f}")
    if cur_result.final_val_loss is not None:
        print(f"  final val loss:   {cur_result.final_val_loss:.4f}")
    print(f"  total wall time:  {cur_result.total_wall_time:.0f}s")
    print(f"{'='*70}")

    # Save results
    _save_results(cur_result, out_dir)

    return cur_result


# ---------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------

def _save_results(result: CurriculumResult, out_dir: str):
    """Save curriculum results as both pickle (full) and JSON (summary)."""
    os.makedirs(out_dir, exist_ok=True)

    # Full result with all metrics (for plotting / analysis)
    pkl_path = os.path.join(out_dir, "curriculum_result.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(result, f)
    print(f"  Full results saved to {pkl_path}")

    # Human-readable summary
    json_path = os.path.join(out_dir, "curriculum_summary.json")
    with open(json_path, "w") as f:
        json.dump(result.summary_dict(), f, indent=2)
    print(f"  Summary saved to {json_path}")


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Curriculum runner for block-diffusion scaling experiments (§6.4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_curriculum.py --list\n"
            "  python run_curriculum.py --curriculum c1_geometric --size 50M --steps 1000\n"
            "  python run_curriculum.py --curriculum c1_geometric --size 50M --budget 1e18\n"
            "  python run_curriculum.py --curriculum baseline_ar --size 98M --budget 2e18 "
            "--optimizer normuon\n"
            "  python run_curriculum.py --curriculum c0_plain_p20 --size 50M --steps 1000 "
            "--dry-run\n"
        ),
    )

    parser.add_argument(
        "--list", action="store_true",
        help="List all available curriculum names and exit",
    )
    parser.add_argument(
        "--curriculum", type=str, default=None,
        help="Curriculum name (use --list to see options)",
    )
    parser.add_argument(
        "--size", type=str, default=None,
        choices=ec.ALL_SIZES,
        help="Model size label (e.g., 50M, 98M, 170M)",
    )
    parser.add_argument(
        "--budget", type=float, default=None,
        help="Total FLOP budget (e.g., 1e18, 2e18). Mutually exclusive with --steps.",
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Total training steps, split across stages by flop_frac "
             "(e.g., --steps 1000). Mutually exclusive with --budget.",
    )
    parser.add_argument(
        "--optimizer", type=str, default="adamw",
        choices=ec.OPTIMIZER_FAMILIES,
        help="Optimizer family (default: adamw)",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Output directory (default: runs/curriculum/<name>_<size>_<budget>_<opt>)",
    )
    parser.add_argument(
        "--eval-interval", type=int, default=300,
        help="Validation evaluation interval in steps (default: 300)",
    )
    parser.add_argument(
        "--eval-iters", type=int, default=50,
        help="Number of eval batches per evaluation (default: 50)",
    )
    parser.add_argument(
        "--save-interval", type=int, default=1000,
        help="Checkpoint save interval in steps (default: 1000)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing",
    )

    args = parser.parse_args()

    # Handle --list
    if args.list:
        list_curricula()
        return

    # Validate required arguments
    if args.curriculum is None:
        parser.error("--curriculum is required (use --list to see options)")
    if args.size is None:
        parser.error("--size is required")
    if args.budget is None and args.steps is None:
        parser.error("--budget or --steps is required")
    if args.budget is not None and args.steps is not None:
        parser.error("--budget and --steps are mutually exclusive")

    # Look up curriculum
    if args.curriculum not in CURRICULUM_REGISTRY:
        parser.error(
            f"Unknown curriculum {args.curriculum!r}. "
            f"Use --list to see available curricula."
        )
    curriculum = CURRICULUM_REGISTRY[args.curriculum]

    # Default output directory
    if args.out_dir is None:
        scope_str = (f"{args.steps}steps"
                     if args.steps is not None
                     else f"{args.budget:.0e}")
        args.out_dir = os.path.join(
            "runs", "curriculum",
            f"{args.curriculum}_{args.size}_{scope_str}_{args.optimizer}",
        )

    # Run
    result = run_curriculum(
        curriculum=curriculum,
        size=args.size,
        out_dir=args.out_dir,
        total_budget=args.budget,
        total_steps=args.steps,
        optimizer=args.optimizer,
        dry_run=args.dry_run,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        save_interval=args.save_interval,
    )

    if result is not None and result.final_val_loss is not None:
        print(f"\nDone. Final val loss = {result.final_val_loss:.4f}")


if __name__ == "__main__":
    main()
