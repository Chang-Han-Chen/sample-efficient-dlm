#!/usr/bin/env python3
"""Launcher/status helper for the MoE old-bundle BD3 curriculum sweep.

This script intentionally keeps the mechanics boring: it generates a run queue,
prints exact `train_ar.py` commands, launches one run at a time, and records
status/metrics in `status.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATUS_PATH = REPO_ROOT / "status.json"

RUN_ROOT = Path("runs/bd3_curriculum_moe_old_bundle")
LAUNCHER_LOG_DIR = RUN_ROOT / "_launcher_logs"
SKIP_LAUNCH_PATH = RUN_ROOT / "_skip_launches.txt"

P_VALUES = [0.30, 0.50, 0.80, 0.90, 0.95]
P_SLUGS = {
    0.30: "p030",
    0.50: "p050",
    0.80: "p080",
    0.90: "p090",
    0.95: "p095",
}
BD3_BLOCK_LENS = [256, 64, 16, 4]
BD3_BASE_LR_MULT = 2.0
PROBE_LR_MULTS = [BD3_BASE_LR_MULT / 3.0, BD3_BASE_LR_MULT, BD3_BASE_LR_MULT * 3.0]

SOURCE_WARMUP_STEPS = 100
SOURCE_CONSTANT_STEPS = 10_000
TOTAL_STEPS = SOURCE_WARMUP_STEPS + SOURCE_CONSTANT_STEPS
PROBE_POST_SWITCH_STEPS = 1_050

DEFAULT_NUM_DEVICES = 4
DEFAULT_BATCH_SIZE = 512
DEFAULT_GRAD_ACCUM_STEPS = 1
DEFAULT_CONTEXT_LENGTH = 512
DEFAULT_VOCAB_SIZE = 8192
DEFAULT_DIFFUSION_MASK_TOKEN_ID = 8192
DEFAULT_EVAL_BATCHES = 4
DEFAULT_LOG_EVERY = 50

AR_CONFIG = "configs/experiments/ar_moe_old_bundle.yaml"
MDLM_CONFIG = "configs/experiments/mdlm_moe_old_bundle.yaml"
BD3_CONFIG = MDLM_CONFIG

WANDB_GROUP = "bd3_curriculum_moe_old_bundle"

TERMINAL_STATUSES = {"completed", "failed", "stopped", "skipped"}
VALID_STATUSES = TERMINAL_STATUSES | {"pending", "running", "promoted"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def rel(path: Path | str) -> str:
    return str(path)


def abs_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def lr_slug(value: float) -> str:
    text = f"{value:.6g}".replace(".", "p").replace("-", "m")
    return f"lr{text}"


def source_step_count(p_value: float) -> int:
    return SOURCE_WARMUP_STEPS + int(round(p_value * SOURCE_CONSTANT_STEPS))


def output_dir_for(run_id: str) -> Path:
    return RUN_ROOT / run_id


def checkpoint_path_for(run_id: str) -> Path:
    return output_dir_for(run_id) / "checkpoints" / "final"


def log_jsonl_for(run_id: str) -> Path:
    return output_dir_for(run_id) / "train.jsonl"


def source_run_id(source: str, target_step_count: int) -> str:
    return f"source_{source}_to{target_step_count:05d}"


def source_checkpoint_run_id(source: str, p_value: float) -> str:
    return source_run_id(source, source_step_count(p_value))


def common_run_fields(
    *,
    run_id: str,
    phase: str,
    source: str,
    objective: str,
    config: str,
    max_steps: int,
    lr_mult: float,
    dependencies: list[str] | None = None,
    restore_checkpoint: str | None = None,
    p_source: float | None = None,
    source_step: int | None = None,
    block_len: int | None = None,
    post_switch_steps: int | None = None,
) -> dict[str, Any]:
    return {
        "id": run_id,
        "status": "pending",
        "phase": phase,
        "source": source,
        "objective": objective,
        "config": config,
        "max_steps": max_steps,
        "lr_mult": lr_mult,
        "dependencies": dependencies or [],
        "restore_checkpoint": restore_checkpoint,
        "p_source": p_source,
        "source_step_count": source_step,
        "block_len": block_len,
        "post_switch_steps": post_switch_steps,
        "run_name": run_id,
        "output_dir": rel(output_dir_for(run_id)),
        "log_jsonl": rel(log_jsonl_for(run_id)),
        "checkpoint_path": rel(checkpoint_path_for(run_id)),
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "attempts": 0,
        "metrics": {},
        "notes": [],
    }


def make_source_runs(source: str) -> list[dict[str, Any]]:
    if source not in {"ar", "mdlm"}:
        raise ValueError(source)
    config = AR_CONFIG if source == "ar" else MDLM_CONFIG
    objective = source
    lr_mult = BD3_BASE_LR_MULT

    checkpoints = [source_step_count(p) for p in P_VALUES]
    checkpoints.append(TOTAL_STEPS)
    checkpoints = sorted(set(checkpoints))

    runs: list[dict[str, Any]] = []
    previous_run_id: str | None = None
    for target_step_count in checkpoints:
        run_id = source_run_id(source, target_step_count)
        restore_checkpoint = None
        dependencies: list[str] = []
        if previous_run_id is not None:
            dependencies.append(previous_run_id)
            restore_checkpoint = rel(checkpoint_path_for(previous_run_id))
        runs.append(
            common_run_fields(
                run_id=run_id,
                phase="source",
                source=source,
                objective=objective,
                config=config,
                max_steps=target_step_count,
                lr_mult=lr_mult,
                dependencies=dependencies,
                restore_checkpoint=restore_checkpoint,
                source_step=target_step_count,
            )
        )
        previous_run_id = run_id
    return runs


def make_bd3_scratch_runs() -> list[dict[str, Any]]:
    runs = []
    for block_len in BD3_BLOCK_LENS:
        run_id = f"scratch_b{block_len}_{lr_slug(BD3_BASE_LR_MULT)}_full"
        runs.append(
            common_run_fields(
                run_id=run_id,
                phase="scratch_full",
                source="scratch",
                objective="bd3lm",
                config=BD3_CONFIG,
                max_steps=TOTAL_STEPS,
                lr_mult=BD3_BASE_LR_MULT,
                block_len=block_len,
                post_switch_steps=TOTAL_STEPS,
            )
        )
    return runs


def make_switch_run(
    *,
    source: str,
    p_value: float,
    block_len: int,
    lr_mult: float,
    phase: str,
    probe_parent: str | None = None,
) -> dict[str, Any]:
    source_steps = source_step_count(p_value)
    p_slug = P_SLUGS[p_value]
    full_or_probe = "probe" if phase == "switch_probe" else "full"
    run_id = f"{source}_{p_slug}_b{block_len}_{lr_slug(lr_mult)}_{full_or_probe}"
    dep = source_checkpoint_run_id(source, p_value)
    max_steps = (
        source_steps + PROBE_POST_SWITCH_STEPS
        if phase == "switch_probe"
        else TOTAL_STEPS
    )
    post_switch = max_steps - source_steps
    run = common_run_fields(
        run_id=run_id,
        phase=phase,
        source=source,
        objective="bd3lm",
        config=BD3_CONFIG,
        max_steps=max_steps,
        lr_mult=lr_mult,
        dependencies=[dep],
        restore_checkpoint=rel(checkpoint_path_for(dep)),
        p_source=p_value,
        source_step=source_steps,
        block_len=block_len,
        post_switch_steps=post_switch,
    )
    if probe_parent is not None:
        run["probe_parent"] = probe_parent
    return run


def make_switch_probe_runs() -> list[dict[str, Any]]:
    runs = []
    for source in ("ar", "mdlm"):
        for p_value in P_VALUES:
            for block_len in BD3_BLOCK_LENS:
                for lr_mult in PROBE_LR_MULTS:
                    runs.append(
                        make_switch_run(
                            source=source,
                            p_value=p_value,
                            block_len=block_len,
                            lr_mult=lr_mult,
                            phase="switch_probe",
                        )
                    )
    return runs


def build_initial_status() -> dict[str, Any]:
    runs = []
    runs.extend(make_source_runs("ar"))
    runs.extend(make_source_runs("mdlm"))
    runs.extend(make_bd3_scratch_runs())
    runs.extend(make_switch_probe_runs())
    return {
        "schema_version": 1,
        "experiment": "bd3_curriculum_moe_old_bundle",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "repo_root": str(REPO_ROOT),
        "instruction_file": "BD3_CURRICULUM_INSTRUCTIONS.md",
        "launcher": "scripts/bd3_curriculum_launcher.py",
        "controls": {
            "metric": "eval_loss",
            "lower_is_better": True,
            "hardware_target": "4xH100",
            "num_devices": DEFAULT_NUM_DEVICES,
            "effective_batch_size": DEFAULT_BATCH_SIZE * DEFAULT_GRAD_ACCUM_STEPS,
            "context_length": DEFAULT_CONTEXT_LENGTH,
            "base_vocab_size": DEFAULT_VOCAB_SIZE,
            "mask_token_id": DEFAULT_DIFFUSION_MASK_TOKEN_ID,
            "ar_source_model_vocab_size": DEFAULT_VOCAB_SIZE + 1,
            "total_steps": TOTAL_STEPS,
            "source_warmup_steps": SOURCE_WARMUP_STEPS,
            "source_constant_steps": SOURCE_CONSTANT_STEPS,
            "p_values": P_VALUES,
            "bd3_block_lens": BD3_BLOCK_LENS,
            "bd3_base_lr_mult": BD3_BASE_LR_MULT,
            "probe_lr_mults": PROBE_LR_MULTS,
            "probe_post_switch_steps": PROBE_POST_SWITCH_STEPS,
            "no_bd3_block_len_512": True,
            "save_optimizer_state": True,
            "restore_optimizer_state": True,
            "wandb_group": WANDB_GROUP,
        },
        "best_config_overrides": {
            "moe": True,
            "attn_qknorm": True,
            "attn_val_residual": True,
            "attn_gating": "per-head",
            "layernorm_scaling": True,
            "value_embedding": False,
            "moe_layers": "alternating",
            "moe_num_experts": 4,
            "moe_top_k": 1,
            "moe_capacity_factor": 1.25,
            "moe_split_router_input": False,
            "moe_router_z_loss_weight": 0.01,
        },
        "baseline_anchors": {
            "best_ar_moe_old_bundle": {
                "run": "ar_moe_old_bundle_nonsplit_lr2p0_zloss0p01_ckpt1k",
                "best_eval_loss": 2.752304255962372,
                "final_eval_loss": 2.7791925072669983,
            },
            "best_mdlm_moe_old_bundle": {
                "run": "mdlm_moe_old_bundle_arstyle_lr2p0_rz0p01_24x_4a100",
                "best_eval_loss": 3.0771419405937195,
                "final_eval_loss": 3.0771419405937195,
            },
        },
        "runs": runs,
        "decisions": [],
    }


def load_status(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return json.load(fh)


def write_status(path: Path, status: dict[str, Any]) -> None:
    status["updated_at"] = now_iso()
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as fh:
        json.dump(status, fh, indent=2, sort_keys=True)
        fh.write("\n")
    tmp.replace(path)


def load_skip_launch_ids() -> set[str]:
    path = abs_path(SKIP_LAUNCH_PATH)
    if not path.exists():
        return set()
    skip_ids = set()
    for line in path.read_text().splitlines():
        run_id = line.split("#", 1)[0].strip()
        if run_id:
            skip_ids.add(run_id)
    return skip_ids


def runs_by_id(status: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {run["id"]: run for run in status["runs"]}


def find_run(status: dict[str, Any], run_id: str) -> dict[str, Any]:
    for run in status["runs"]:
        if run["id"] == run_id:
            return run
    raise KeyError(f"unknown run id: {run_id}")


def final_checkpoint_exists(run: dict[str, Any]) -> bool:
    return (abs_path(run["checkpoint_path"]) / "metadata.json").exists()


def dependencies_satisfied(status: dict[str, Any], run: dict[str, Any]) -> bool:
    lookup = runs_by_id(status)
    for dep_id in run.get("dependencies", []):
        dep = lookup.get(dep_id)
        if dep is None:
            return False
        if dep.get("status") != "completed" and not final_checkpoint_exists(dep):
            return False
        if not final_checkpoint_exists(dep):
            return False
    restore_checkpoint = run.get("restore_checkpoint")
    if restore_checkpoint and not (abs_path(restore_checkpoint) / "metadata.json").exists():
        return False
    return True


def run_matches(
    run: dict[str, Any],
    *,
    status_filter: str | None = None,
    phase: str | None = None,
    source: str | None = None,
    objective: str | None = None,
) -> bool:
    if status_filter is not None and run.get("status") != status_filter:
        return False
    if phase is not None and run.get("phase") != phase:
        return False
    if source is not None and run.get("source") != source:
        return False
    if objective is not None and run.get("objective") != objective:
        return False
    return True


def select_runs(
    status: dict[str, Any],
    *,
    status_filter: str | None = None,
    phase: str | None = None,
    source: str | None = None,
    objective: str | None = None,
    eligible_only: bool = False,
) -> list[dict[str, Any]]:
    selected = []
    for run in status["runs"]:
        if not run_matches(
            run,
            status_filter=status_filter,
            phase=phase,
            source=source,
            objective=objective,
        ):
            continue
        if eligible_only and not dependencies_satisfied(status, run):
            continue
        selected.append(run)
    return selected


def command_for_run(run: dict[str, Any]) -> list[str]:
    python = os.environ.get("PYTHON", sys.executable or "python")
    cmd = [
        python,
        "train_ar.py",
        "--config",
        run["config"],
        "--objective",
        run["objective"],
        "--seed",
        "42",
        "--run-name",
        run["run_name"],
        "--output-dir",
        run["output_dir"],
        "--log-jsonl",
        run["log_jsonl"],
        "--max-steps",
        str(run["max_steps"]),
        "--warmup-steps",
        str(SOURCE_WARMUP_STEPS),
        "--batch-size",
        str(DEFAULT_BATCH_SIZE),
        "--grad-accum-steps",
        str(DEFAULT_GRAD_ACCUM_STEPS),
        "--data-parallel",
        "--num-devices",
        str(DEFAULT_NUM_DEVICES),
        "--context-length",
        str(DEFAULT_CONTEXT_LENGTH),
        "--attention-impl",
        "cudnn",
        "--attn-val-residual",
        "--no-moe-split-router-input",
        "--moe-router-z-loss-weight",
        "0.01",
        "--lr-mult",
        f"{run['lr_mult']:.12g}",
        "--eval-batches",
        str(DEFAULT_EVAL_BATCHES),
        "--log-every",
        str(DEFAULT_LOG_EVERY),
        "--checkpoint-interval",
        "0",
        "--no-wandb-checkpoints",
        "--wandb-group",
        WANDB_GROUP,
    ]
    restore_checkpoint = run.get("restore_checkpoint")
    if restore_checkpoint:
        cmd.extend(["--restore-checkpoint", restore_checkpoint])

    if run["objective"] == "ar":
        # Diffusion models have an extra mask-token row. Train AR with one extra
        # unused output row so AR checkpoints restore into BD3 with optimizer
        # state intact.
        cmd.extend(["--vocab-size", str(DEFAULT_VOCAB_SIZE + 1)])
    else:
        cmd.extend(
            [
                "--vocab-size",
                str(DEFAULT_VOCAB_SIZE),
                "--mask-token-id",
                str(DEFAULT_DIFFUSION_MASK_TOKEN_ID),
                "--diffusion-steps",
                "100",
                "--t-min",
                "0.45",
                "--t-max",
                "0.95",
                "--noise-schedule",
                "linear",
                "--eval-t-frac",
                "0.6",
            ]
        )
    if run["objective"] == "bd3lm":
        cmd.extend(["--bd3-block-len", str(run["block_len"]), "--bd3-attention", "dense"])

    tags = [
        "bd3_curriculum",
        "moe",
        "old_bundle",
        run["phase"],
        run["source"],
        run["objective"],
    ]
    if run.get("block_len") is not None:
        tags.append(f"b{run['block_len']}")
    if run.get("p_source") is not None:
        tags.append(P_SLUGS[float(run["p_source"])])
    cmd.extend(["--wandb-tags", *tags])
    return cmd


def quote_command(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def parse_jsonl_metrics(path: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if not path.exists():
        return metrics
    last_row = None
    eval_rows = []
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            last_row = row
            if "eval_loss" in row:
                eval_rows.append(row)
    if last_row is not None:
        metrics["last_step"] = last_row.get("step")
        metrics["last_loss"] = last_row.get("loss")
        metrics["last_grad_norm"] = last_row.get("grad_norm")
        metrics["last_moe_dropped_fraction"] = last_row.get("moe_dropped_fraction")
        metrics["last_moe_router_entropy"] = last_row.get("moe_router_entropy")
    if eval_rows:
        best = min(eval_rows, key=lambda row: row["eval_loss"])
        final = eval_rows[-1]
        metrics["best_eval_loss"] = best["eval_loss"]
        metrics["best_eval_step"] = best.get("step")
        metrics["final_eval_loss"] = final["eval_loss"]
        metrics["final_eval_step"] = final.get("step")
        metrics["num_eval_points"] = len(eval_rows)
    return metrics


def parse_checkpoint_metrics(run: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    final_metadata = abs_path(run["checkpoint_path"]) / "metadata.json"
    if final_metadata.exists():
        with final_metadata.open() as fh:
            metadata = json.load(fh)
        metrics["final_checkpoint_step"] = metadata.get("step")
        final_metrics = metadata.get("metrics") or {}
        if "avg_measured_step_sec" in final_metrics:
            metrics["avg_measured_step_sec"] = final_metrics["avg_measured_step_sec"]
        if "tokens_per_sec" in final_metrics:
            metrics["tokens_per_sec"] = final_metrics["tokens_per_sec"]
        if "jax_peak_hbm_gb" in final_metrics:
            metrics["jax_peak_hbm_gb"] = final_metrics["jax_peak_hbm_gb"]
    best_metadata = abs_path(run["output_dir"]) / "checkpoints" / "best" / "metadata.json"
    if best_metadata.exists():
        with best_metadata.open() as fh:
            metadata = json.load(fh)
        best_metrics = metadata.get("metrics") or {}
        if "eval_loss" in best_metrics:
            metrics["best_checkpoint_eval_loss"] = best_metrics["eval_loss"]
            metrics["best_checkpoint_step"] = metadata.get("step")
    return metrics


def update_metrics_for_run(run: dict[str, Any]) -> bool:
    before = json.dumps(run.get("metrics", {}), sort_keys=True)
    metrics = {}
    metrics.update(parse_jsonl_metrics(abs_path(run["log_jsonl"])))
    metrics.update(parse_checkpoint_metrics(run))
    if metrics:
        run["metrics"] = metrics
    if final_checkpoint_exists(run) and run.get("status") in {"pending", "running", "failed"}:
        run["status"] = "completed"
    after = json.dumps(run.get("metrics", {}), sort_keys=True)
    changed = before != after
    if changed:
        run["updated_at"] = now_iso()
    return changed


def cmd_init(args: argparse.Namespace) -> int:
    path = args.status_path
    if path.exists() and not args.force:
        print(f"{path} already exists; pass --force to overwrite", file=sys.stderr)
        return 2
    status = build_initial_status()
    write_status(path, status)
    print(f"initialized {path} with {len(status['runs'])} runs")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    status = load_status(args.status_path)
    runs = select_runs(
        status,
        status_filter=args.status,
        phase=args.phase,
        source=args.source,
        objective=args.objective,
        eligible_only=args.eligible,
    )
    if args.limit is not None:
        runs = runs[: args.limit]
    for run in runs:
        dep_ok = "yes" if dependencies_satisfied(status, run) else "no"
        metric = run.get("metrics", {}).get("final_eval_loss")
        best = run.get("metrics", {}).get("best_eval_loss")
        metric_text = ""
        if best is not None:
            metric_text += f" best={best:.6f}"
        if metric is not None:
            metric_text += f" final={metric:.6f}"
        print(
            f"{run['id']:<52} {run['status']:<10} {run['phase']:<13} "
            f"src={run['source']:<6} obj={run['objective']:<5} "
            f"b={str(run.get('block_len')):<4} p={str(run.get('p_source')):<4} "
            f"lr={run['lr_mult']:<8.4g} deps={dep_ok}{metric_text}"
        )
    print(f"{len(runs)} run(s)")
    return 0


def cmd_command(args: argparse.Namespace) -> int:
    status = load_status(args.status_path)
    run = find_run(status, args.run_id)
    command = command_for_run(run)
    if args.json:
        print(json.dumps(command, indent=2))
    else:
        print(quote_command(command))
    return 0


def pick_next_run(args: argparse.Namespace, status: dict[str, Any]) -> dict[str, Any] | None:
    candidates = select_runs(
        status,
        status_filter="pending",
        phase=args.phase,
        source=args.source,
        objective=args.objective,
        eligible_only=True,
    )
    return candidates[0] if candidates else None


def cmd_next(args: argparse.Namespace) -> int:
    status = load_status(args.status_path)
    run = pick_next_run(args, status)
    if run is None:
        print("no eligible pending run")
        return 1
    print(run["id"])
    if args.command:
        print(quote_command(command_for_run(run)))
    return 0


def launch_run(args: argparse.Namespace, status: dict[str, Any], run: dict[str, Any]) -> int:
    if not args.force and run["id"] in load_skip_launch_ids():
        run["status"] = "skipped"
        run["updated_at"] = now_iso()
        run.setdefault("notes", []).append(
            {"at": now_iso(), "text": f"Skipped by launch policy {SKIP_LAUNCH_PATH}."}
        )
        write_status(args.status_path, status)
        print(f"{run['id']} skipped by launch policy {SKIP_LAUNCH_PATH}")
        return 0

    if run["status"] not in {"pending", "failed", "stopped"} and not args.force:
        print(f"{run['id']} has status {run['status']}; pass --force to launch anyway", file=sys.stderr)
        return 2
    if not dependencies_satisfied(status, run) and not args.force:
        print(f"{run['id']} dependencies are not satisfied", file=sys.stderr)
        return 2

    command = command_for_run(run)
    log_path = abs_path(LAUNCHER_LOG_DIR / f"{run['id']}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    run["status"] = "running"
    run["attempts"] = int(run.get("attempts", 0)) + 1
    run["started_at"] = now_iso()
    run["updated_at"] = now_iso()
    run["launcher_log"] = rel(log_path.relative_to(REPO_ROOT) if log_path.is_relative_to(REPO_ROOT) else log_path)
    run["command"] = command
    write_status(args.status_path, status)

    print(quote_command(command), flush=True)
    started = time.time()
    proc: subprocess.Popen[str] | None = None
    try:
        with log_path.open("a") as log_fh:
            log_fh.write(f"\n# launch {now_iso()} run_id={run['id']}\n")
            log_fh.write(quote_command(command) + "\n")
            log_fh.flush()
            proc = subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                log_fh.write(line)
            return_code = proc.wait()
    except KeyboardInterrupt:
        if proc is not None and proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        run["status"] = "stopped"
        run["exit_code"] = 130
        run["completed_at"] = now_iso()
        run["wall_time_sec"] = time.time() - started
        run["updated_at"] = now_iso()
        update_metrics_for_run(run)
        write_status(args.status_path, status)
        print(f"stopped {run['id']}")
        return 130

    run["exit_code"] = return_code
    run["completed_at"] = now_iso()
    run["wall_time_sec"] = time.time() - started
    run["status"] = "completed" if return_code == 0 else "failed"
    update_metrics_for_run(run)
    run["updated_at"] = now_iso()
    write_status(args.status_path, status)
    print(f"{run['id']} finished with exit_code={return_code}")
    return return_code


def cmd_launch(args: argparse.Namespace) -> int:
    status = load_status(args.status_path)
    run = find_run(status, args.run_id)
    return launch_run(args, status, run)


def cmd_launch_next(args: argparse.Namespace) -> int:
    status = load_status(args.status_path)
    run = pick_next_run(args, status)
    if run is None:
        print("no eligible pending run")
        return 1
    return launch_run(args, status, run)


def cmd_mark(args: argparse.Namespace) -> int:
    if args.new_status not in VALID_STATUSES:
        print(f"invalid status {args.new_status!r}; expected one of {sorted(VALID_STATUSES)}", file=sys.stderr)
        return 2
    status = load_status(args.status_path)
    run = find_run(status, args.run_id)
    run["status"] = args.new_status
    run["updated_at"] = now_iso()
    if args.note:
        run.setdefault("notes", []).append({"at": now_iso(), "text": args.note})
    write_status(args.status_path, status)
    print(f"{run['id']} -> {args.new_status}")
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    status = load_status(args.status_path)
    changed = False
    for run in status["runs"]:
        if update_metrics_for_run(run):
            changed = True
    write_status(args.status_path, status)

    rows = [
        run
        for run in status["runs"]
        if run.get("metrics", {}).get("final_eval_loss") is not None
        or run.get("metrics", {}).get("best_eval_loss") is not None
    ]
    rows.sort(
        key=lambda run: (
            run.get("phase", ""),
            run.get("source", ""),
            999 if run.get("block_len") is None else int(run["block_len"]),
            999.0 if run.get("p_source") is None else float(run["p_source"]),
            run.get("metrics", {}).get("final_eval_loss", float("inf")),
        )
    )
    for run in rows:
        metrics = run["metrics"]
        print(
            f"{run['id']:<52} {run['phase']:<13} src={run['source']:<6} "
            f"b={str(run.get('block_len')):<4} p={str(run.get('p_source')):<4} "
            f"lr={run['lr_mult']:<8.4g} "
            f"best={metrics.get('best_eval_loss')} final={metrics.get('final_eval_loss')}"
        )
    print(f"updated={changed} summarized={len(rows)}")
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    status = load_status(args.status_path)
    probe = find_run(status, args.probe_run_id)
    if probe.get("phase") != "switch_probe":
        print(f"{args.probe_run_id} is not a switch_probe run", file=sys.stderr)
        return 2
    source = probe["source"]
    p_source = float(probe["p_source"])
    block_len = int(probe["block_len"])
    lr_mult = float(probe["lr_mult"])
    full_run = make_switch_run(
        source=source,
        p_value=p_source,
        block_len=block_len,
        lr_mult=lr_mult,
        phase="switch_full",
        probe_parent=probe["id"],
    )
    lookup = runs_by_id(status)
    if full_run["id"] in lookup:
        print(f"{full_run['id']} already exists")
        return 0
    full_run["notes"].append({"at": now_iso(), "text": f"promoted from {probe['id']}"})
    status["runs"].append(full_run)
    status["decisions"].append(
        {
            "at": now_iso(),
            "type": "promote_probe_to_full",
            "probe_run_id": probe["id"],
            "full_run_id": full_run["id"],
            "reason": args.reason,
        }
    )
    probe["status"] = "promoted"
    probe["updated_at"] = now_iso()
    write_status(args.status_path, status)
    print(full_run["id"])
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--status-path", type=Path, default=DEFAULT_STATUS_PATH)
    sub = p.add_subparsers(dest="command_name", required=True)

    init_p = sub.add_parser("init", help="initialize status.json")
    init_p.add_argument("--force", action="store_true")
    init_p.set_defaults(func=cmd_init)

    list_p = sub.add_parser("list", help="list runs")
    list_p.add_argument("--status", choices=sorted(VALID_STATUSES), default=None)
    list_p.add_argument("--phase", default=None)
    list_p.add_argument("--source", choices=("ar", "mdlm", "scratch"), default=None)
    list_p.add_argument("--objective", choices=("ar", "mdlm", "bd3lm"), default=None)
    list_p.add_argument("--eligible", action="store_true")
    list_p.add_argument("--limit", type=int, default=None)
    list_p.set_defaults(func=cmd_list)

    command_p = sub.add_parser("command", help="print the train_ar.py command for a run")
    command_p.add_argument("run_id")
    command_p.add_argument("--json", action="store_true")
    command_p.set_defaults(func=cmd_command)

    next_p = sub.add_parser("next", help="print the next eligible pending run")
    next_p.add_argument("--phase", default=None)
    next_p.add_argument("--source", choices=("ar", "mdlm", "scratch"), default=None)
    next_p.add_argument("--objective", choices=("ar", "mdlm", "bd3lm"), default=None)
    next_p.add_argument("--command", action="store_true")
    next_p.set_defaults(func=cmd_next)

    launch_p = sub.add_parser("launch", help="launch one run in the foreground")
    launch_p.add_argument("run_id")
    launch_p.add_argument("--force", action="store_true")
    launch_p.set_defaults(func=cmd_launch)

    launch_next_p = sub.add_parser("launch-next", help="launch the next eligible pending run")
    launch_next_p.add_argument("--phase", default=None)
    launch_next_p.add_argument("--source", choices=("ar", "mdlm", "scratch"), default=None)
    launch_next_p.add_argument("--objective", choices=("ar", "mdlm", "bd3lm"), default=None)
    launch_next_p.add_argument("--force", action="store_true")
    launch_next_p.set_defaults(func=cmd_launch_next)

    mark_p = sub.add_parser("mark", help="manually set run status")
    mark_p.add_argument("run_id")
    mark_p.add_argument("new_status")
    mark_p.add_argument("--note", default=None)
    mark_p.set_defaults(func=cmd_mark)

    summarize_p = sub.add_parser("summarize", help="parse logs/checkpoints and update metrics")
    summarize_p.set_defaults(func=cmd_summarize)

    promote_p = sub.add_parser("promote", help="add a full continuation from a probe")
    promote_p.add_argument("probe_run_id")
    promote_p.add_argument("--reason", default=None)
    promote_p.set_defaults(func=cmd_promote)
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
