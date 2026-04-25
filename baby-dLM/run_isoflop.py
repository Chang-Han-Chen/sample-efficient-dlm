#!/usr/bin/env python3
"""
run_isoflop.py — IsoFLOP sweep for block-diffusion scaling experiments (§6.5).

For each (budget, curriculum, optimizer_family, model_size) combination, runs
the curriculum via run_curriculum.run_curriculum() and collects the final
validation loss.  The result is a set of IsoFLOP curves: for each
(budget, curriculum, optimizer_family), a mapping from model_size to val_loss.

Phase 1 scope (PLAN §5):
  - Optimizer families: adamw, normuon
  - Curricula: baselines (AR, MDLM, BD3-LM) + curriculum 0 + curriculum 1
  - Budgets: {1e18, 2e18, 4e18, 1e19}
  - Sizes: 50M, 98M, 170M

Usage:
    # Full phase-1 sweep (all budgets × curricula × optimizers × sizes):
    python run_isoflop.py

    # Single optimizer:
    python run_isoflop.py --optimizer adamw

    # Single budget:
    python run_isoflop.py --budget 1e18

    # Include extra curricula beyond phase-1 defaults:
    python run_isoflop.py --curriculum c2_aggressive_jump

    # Run ONLY specific curricula (replace phase-1 defaults):
    python run_isoflop.py --curriculum-only c1_geometric --curriculum-only baseline_ar

    # Restrict to specific sizes:
    python run_isoflop.py --sizes 50M 98M

    # Dry-run (print what would be run):
    python run_isoflop.py --dry-run
"""

import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import experiment_config as ec
from run_curriculum import (
    CURRICULUM_REGISTRY,
    CurriculumResult,
    run_curriculum,
)

# ---------------------------------------------------------------
# Phase-1 curricula (PLAN §5)
# ---------------------------------------------------------------

# Baselines + curriculum 0 (all p_AR variants) + curriculum 1.
PHASE1_CURRICULA = (
    [ec.BASELINE_AR, ec.BASELINE_MDLM, ec.BASELINE_BD3LM]
    + ec.CURRICULUM_0_SWEEPS
    + [ec.CURRICULUM_1]
)

PHASE1_CURRICULUM_NAMES = [c.name for c in PHASE1_CURRICULA]


# ---------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------

@dataclass
class IsoFLOPPoint:
    """One (budget, curriculum, optimizer, size) data point."""
    budget: float
    curriculum: str
    optimizer: str
    size: str
    val_loss: Optional[float]
    train_loss: Optional[float]
    lr_config: dict
    wall_time_seconds: float
    out_dir: str
    num_stages_completed: int
    num_stages_total: int
    skipped: bool = False       # True if the size/budget combo was infeasible
    skip_reason: str = ""

    @property
    def completed(self) -> bool:
        """
        True only when all stages ran to completion and a val_loss is available.

        Points that are skipped, partially completed, from dry-runs, or
        missing a final val_loss are NOT considered completed and must be
        excluded from IsoFLOP curves used for scaling-law fits.
        """
        return (
            not self.skipped
            and self.num_stages_completed == self.num_stages_total
            and self.num_stages_completed > 0
            and self.val_loss is not None
        )


@dataclass
class IsoFLOPSweepResult:
    """Full sweep results, ready for plotting."""
    points: List[IsoFLOPPoint] = field(default_factory=list)

    def curves(self) -> Dict[Tuple[float, str, str], Dict[str, float]]:
        """
        Group results into IsoFLOP curves.

        Returns {(budget, curriculum, optimizer): {size: val_loss}}.
        Each curve maps model_size -> val_loss for a fixed
        (budget, curriculum, optimizer) triple.

        Only fully completed points (all stages done, val_loss present) are
        included.  Skipped, partial, and dry-run points are excluded to
        prevent corrupted scaling-law fits.
        """
        out: Dict[Tuple[float, str, str], Dict[str, float]] = {}
        for p in self.points:
            if not p.completed:
                continue
            key = (p.budget, p.curriculum, p.optimizer)
            if key not in out:
                out[key] = {}
            out[key][p.size] = p.val_loss
        return out

    def summary_table(self) -> str:
        """Human-readable summary table of all data points."""
        lines = []
        lines.append(
            f"{'budget':>10s}  {'curriculum':<30s}  {'optimizer':>8s}  "
            f"{'size':>5s}  {'val_loss':>10s}  {'train_loss':>12s}  "
            f"{'wall_s':>8s}  {'status':<10s}"
        )
        lines.append("-" * 110)

        for p in sorted(self.points,
                        key=lambda x: (x.budget, x.curriculum, x.optimizer, x.size)):
            if p.skipped:
                status = f"SKIP({p.skip_reason})"
                vl = "—"
                tl = "—"
                wt = "—"
            else:
                status = (f"OK({p.num_stages_completed}/{p.num_stages_total})"
                          if p.num_stages_completed == p.num_stages_total
                          else f"PARTIAL({p.num_stages_completed}/{p.num_stages_total})")
                vl = f"{p.val_loss:.4f}" if p.val_loss is not None else "N/A"
                tl = f"{p.train_loss:.4f}" if p.train_loss is not None else "N/A"
                wt = f"{p.wall_time_seconds:.0f}"
            lines.append(
                f"{p.budget:>10.2e}  {p.curriculum:<30s}  {p.optimizer:>8s}  "
                f"{p.size:>5s}  {vl:>10s}  {tl:>12s}  {wt:>8s}  {status:<10s}"
            )
        return "\n".join(lines)

    def summary_dict(self) -> dict:
        """Serializable summary for JSON."""
        return {
            "num_points": len(self.points),
            "num_completed": sum(1 for p in self.points if p.completed),
            "num_skipped": sum(1 for p in self.points if p.skipped),
            "curves": {
                f"{b:.2e}|{c}|{o}": size_map
                for (b, c, o), size_map in self.curves().items()
            },
            "points": [
                {
                    "budget": p.budget,
                    "curriculum": p.curriculum,
                    "optimizer": p.optimizer,
                    "size": p.size,
                    "val_loss": p.val_loss,
                    "train_loss": p.train_loss,
                    "lr_config": p.lr_config,
                    "wall_time_s": round(p.wall_time_seconds, 1),
                    "stages_completed": p.num_stages_completed,
                    "stages_total": p.num_stages_total,
                    "skipped": p.skipped,
                    "skip_reason": p.skip_reason,
                }
                for p in self.points
            ],
        }


# ---------------------------------------------------------------
# Feasibility check
# ---------------------------------------------------------------

def check_feasibility(
    curriculum: ec.Curriculum,
    size: str,
    budget: float,
) -> Optional[str]:
    """
    Check whether every stage of a curriculum is feasible at (size, budget).

    Returns None if feasible, or a reason string if not.

    A stage is infeasible when its step count falls outside
    [ISOFLOP_MIN_STEPS, ISOFLOP_MAX_STEPS].  For multi-family curricula,
    we check each stage individually since different families have
    different FLOP multipliers.
    """
    for i, stage in enumerate(curriculum.stages):
        stage_budget = budget * stage.flop_frac
        steps = ec.compute_isoflop_steps(
            stage_budget,
            stage.model_family,
            size,
        )
        if steps is None:
            # Recompute raw steps for the error message
            tokens_per_step = (ec.CLIMBMIX_TOKENS_PER_STEP
                               if size in ec.CLIMBMIX_MODEL_SIZES
                               else ec.ISOFLOP_TOKENS_PER_STEP)
            N = ec.non_embedding_params(size)
            C = ec.flop_multiplier(stage.model_family)
            raw_steps = int(stage_budget / (C * N * tokens_per_step))
            return (
                f"stage {i} ({stage.model_family}): "
                f"{raw_steps} steps outside "
                f"[{ec.ISOFLOP_MIN_STEPS}, {ec.ISOFLOP_MAX_STEPS}]"
            )
    return None


# ---------------------------------------------------------------
# Single run
# ---------------------------------------------------------------

def run_single_isoflop(
    curriculum: ec.Curriculum,
    size: str,
    budget: float,
    optimizer: str,
    out_root: str,
    *,
    dry_run: bool = False,
    eval_interval: int = 300,
    eval_iters: int = 50,
    save_interval: int = 1000,
) -> IsoFLOPPoint:
    """
    Run one (budget, curriculum, optimizer, size) point.

    Returns an IsoFLOPPoint with results (or skip info).
    """
    budget_str = f"{budget:.0e}"
    run_dir = os.path.join(
        out_root,
        f"{curriculum.name}_{size}_{budget_str}_{optimizer}",
    )

    # Feasibility gate
    reason = check_feasibility(curriculum, size, budget)
    if reason is not None:
        print(f"  SKIP {curriculum.name} / {size} / {budget_str} / {optimizer}: {reason}")
        return IsoFLOPPoint(
            budget=budget,
            curriculum=curriculum.name,
            optimizer=optimizer,
            size=size,
            val_loss=None,
            train_loss=None,
            lr_config={},
            wall_time_seconds=0.0,
            out_dir=run_dir,
            num_stages_completed=0,
            num_stages_total=len(curriculum.stages),
            skipped=True,
            skip_reason=reason,
        )

    print(f"\n{'─'*70}")
    print(f"  IsoFLOP: {curriculum.name} / {size} / {budget_str} / {optimizer}")
    print(f"{'─'*70}")

    if dry_run:
        # Delegate to run_curriculum in dry-run mode
        run_curriculum(
            curriculum=curriculum,
            size=size,
            total_budget=budget,
            out_dir=run_dir,
            optimizer=optimizer,
            dry_run=True,
            eval_interval=eval_interval,
            eval_iters=eval_iters,
            save_interval=save_interval,
        )
        return IsoFLOPPoint(
            budget=budget,
            curriculum=curriculum.name,
            optimizer=optimizer,
            size=size,
            val_loss=None,
            train_loss=None,
            lr_config={},
            wall_time_seconds=0.0,
            out_dir=run_dir,
            num_stages_completed=0,
            num_stages_total=len(curriculum.stages),
            skipped=False,
            skip_reason="dry-run",
        )

    result = run_curriculum(
        curriculum=curriculum,
        size=size,
        total_budget=budget,
        out_dir=run_dir,
        optimizer=optimizer,
        dry_run=False,
        eval_interval=eval_interval,
        eval_iters=eval_iters,
        save_interval=save_interval,
    )

    if result is None:
        # Should not happen outside dry-run, but guard
        return IsoFLOPPoint(
            budget=budget,
            curriculum=curriculum.name,
            optimizer=optimizer,
            size=size,
            val_loss=None,
            train_loss=None,
            lr_config={},
            wall_time_seconds=0.0,
            out_dir=run_dir,
            num_stages_completed=0,
            num_stages_total=len(curriculum.stages),
            skipped=False,
            skip_reason="no result",
        )

    return IsoFLOPPoint(
        budget=budget,
        curriculum=result.curriculum_name,
        optimizer=result.optimizer_family,
        size=result.size,
        val_loss=result.final_val_loss,
        train_loss=result.final_train_loss,
        lr_config=result.lr_config,
        wall_time_seconds=result.total_wall_time,
        out_dir=run_dir,
        num_stages_completed=len(result.stages),
        num_stages_total=len(curriculum.stages),
    )


# ---------------------------------------------------------------
# Full sweep
# ---------------------------------------------------------------

def run_isoflop_sweep(
    budgets: List[float],
    curricula: List[ec.Curriculum],
    optimizers: List[str],
    sizes: List[str],
    out_root: str,
    *,
    dry_run: bool = False,
    eval_interval: int = 300,
    eval_iters: int = 50,
    save_interval: int = 1000,
) -> IsoFLOPSweepResult:
    """
    Run the full IsoFLOP sweep.

    Iterates over all (budget, curriculum, optimizer, size) combinations.
    Skips infeasible points and collects results into an IsoFLOPSweepResult.
    """
    sweep = IsoFLOPSweepResult()

    total = len(budgets) * len(curricula) * len(optimizers) * len(sizes)
    done = 0

    print(f"\n{'='*70}")
    print(f"IsoFLOP Sweep")
    print(f"  budgets:    {[f'{b:.0e}' for b in budgets]}")
    print(f"  curricula:  {[c.name for c in curricula]}")
    print(f"  optimizers: {optimizers}")
    print(f"  sizes:      {sizes}")
    print(f"  total combinations: {total}")
    print(f"  out_root: {out_root}")
    print(f"{'='*70}")

    for budget in budgets:
        for curriculum in curricula:
            for optimizer in optimizers:
                for size in sizes:
                    done += 1
                    print(f"\n[{done}/{total}] ", end="")

                    point = run_single_isoflop(
                        curriculum=curriculum,
                        size=size,
                        budget=budget,
                        optimizer=optimizer,
                        out_root=out_root,
                        dry_run=dry_run,
                        eval_interval=eval_interval,
                        eval_iters=eval_iters,
                        save_interval=save_interval,
                    )
                    sweep.points.append(point)

    return sweep


# ---------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------

def save_sweep_results(sweep: IsoFLOPSweepResult, out_root: str):
    """Save the full sweep result as pickle + JSON."""
    os.makedirs(out_root, exist_ok=True)

    pkl_path = os.path.join(out_root, "isoflop_sweep.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(sweep, f)
    print(f"\nFull sweep results saved to {pkl_path}")

    json_path = os.path.join(out_root, "isoflop_summary.json")
    with open(json_path, "w") as f:
        json.dump(sweep.summary_dict(), f, indent=2)
    print(f"Summary saved to {json_path}")


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="IsoFLOP sweep for block-diffusion scaling experiments (§6.5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_isoflop.py                                    # full phase-1\n"
            "  python run_isoflop.py --optimizer adamw                  # single optimizer\n"
            "  python run_isoflop.py --budget 1e18 --budget 2e18        # subset of budgets\n"
            "  python run_isoflop.py --curriculum c2_aggressive_jump    # phase-1 + c2\n"
            "  python run_isoflop.py --curriculum-only c1_geometric     # only c1\n"
            "  python run_isoflop.py --sizes 50M 98M                    # subset of sizes\n"
            "  python run_isoflop.py --dry-run                          # preview only\n"
        ),
    )

    parser.add_argument(
        "--budget", type=float, action="append", default=None,
        help="FLOP budget (repeatable; default: all phase-1 budgets)",
    )
    parser.add_argument(
        "--curriculum", type=str, action="append", default=None,
        help="Extra curriculum to include beyond the phase-1 defaults "
             "(repeatable). Use 'python run_curriculum.py --list' to see options.",
    )
    parser.add_argument(
        "--curriculum-only", type=str, action="append", default=None,
        help="Run ONLY these curricula, replacing the phase-1 default set "
             "(repeatable). Mutually exclusive with --curriculum.",
    )
    parser.add_argument(
        "--optimizer", type=str, action="append", default=None,
        choices=ec.OPTIMIZER_FAMILIES,
        help="Optimizer family (repeatable; default: all)",
    )
    parser.add_argument(
        "--sizes", type=str, nargs="+", default=None,
        choices=ec.CLIMBMIX_SIZES,
        help="Model sizes to sweep (default: all ClimbMix sizes)",
    )
    parser.add_argument(
        "--out-root", type=str, default="runs/isoflop",
        help="Root output directory (default: runs/isoflop)",
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

    # Resolve defaults
    budgets = args.budget if args.budget else ec.CLIMBMIX_ISOFLOP_BUDGETS
    optimizers = args.optimizer if args.optimizer else ec.OPTIMIZER_FAMILIES
    sizes = args.sizes if args.sizes else ec.CLIMBMIX_SIZES

    # Resolve curricula
    if args.curriculum and args.curriculum_only:
        parser.error("--curriculum and --curriculum-only are mutually exclusive")

    def _resolve_names(names):
        resolved = []
        for name in names:
            if name not in CURRICULUM_REGISTRY:
                parser.error(
                    f"Unknown curriculum {name!r}. "
                    "Use 'python run_curriculum.py --list' to see options."
                )
            resolved.append(CURRICULUM_REGISTRY[name])
        return resolved

    if args.curriculum_only:
        # Replace mode: only the explicitly listed curricula
        curricula = _resolve_names(args.curriculum_only)
    elif args.curriculum:
        # Additive mode: phase-1 defaults + extras (deduplicated, order preserved)
        extras = _resolve_names(args.curriculum)
        seen = {c.name for c in PHASE1_CURRICULA}
        curricula = list(PHASE1_CURRICULA)
        for c in extras:
            if c.name not in seen:
                seen.add(c.name)
                curricula.append(c)
    else:
        curricula = PHASE1_CURRICULA

    # Run
    sweep = run_isoflop_sweep(
        budgets=budgets,
        curricula=curricula,
        optimizers=optimizers,
        sizes=sizes,
        out_root=args.out_root,
        dry_run=args.dry_run,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        save_interval=args.save_interval,
    )

    # Print summary table
    print(f"\n{'='*70}")
    print("IsoFLOP Sweep Summary")
    print(f"{'='*70}")
    print(sweep.summary_table())

    # Save
    if not args.dry_run:
        save_sweep_results(sweep, args.out_root)
    else:
        print(f"\n[dry-run] Would save results to {args.out_root}/")

    # Print IsoFLOP curves
    curves = sweep.curves()
    if curves:
        print(f"\n{'='*70}")
        print("IsoFLOP Curves (budget, curriculum, optimizer → size: val_loss)")
        print(f"{'='*70}")
        for (b, c, o), size_map in sorted(curves.items()):
            print(f"\n  C={b:.2e}  {c}  [{o}]")
            for sz in sorted(size_map.keys(),
                             key=lambda s: ec.non_embedding_params(s)):
                vl = size_map[sz]
                vl_str = f"{vl:.4f}" if vl is not None else "N/A"
                N = ec.non_embedding_params(sz)
                print(f"    {sz:>5s} (N={N:>12,d}):  val_loss = {vl_str}")


if __name__ == "__main__":
    main()
