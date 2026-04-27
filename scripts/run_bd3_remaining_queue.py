#!/usr/bin/env python3
"""Run the remaining fixed-lr BD3 curriculum queue.

This wrapper intentionally delegates each individual run to
scripts/bd3_curriculum_launcher.py so status.json, commands, logs, and
checkpoint parsing keep using the existing curriculum machinery.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = REPO_ROOT / "status.json"
LAUNCHER = REPO_ROOT / "scripts" / "bd3_curriculum_launcher.py"

CURRENT_RUN_IDS = ["ar_p030_b4_lr2_full"]
REMAINING_P_SLUGS = ["p050", "p080", "p090", "p095"]
AR_BLOCK_LENS = [256, 4]
TERMINAL_STATUSES = {"completed", "failed", "stopped", "skipped", "promoted"}

METRIC_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([-+0-9.eE]+)")


def load_status(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return json.load(fh)


def default_run_ids() -> list[str]:
    ids = list(CURRENT_RUN_IDS)
    for p_slug in REMAINING_P_SLUGS:
        for block_len in AR_BLOCK_LENS:
            ids.append(f"ar_{p_slug}_b{block_len}_lr2_full")
    return ids


def runs_by_id(status: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {run["id"]: run for run in status["runs"]}


def abs_repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def final_checkpoint_exists(run: dict[str, Any]) -> bool:
    return (abs_repo_path(run["checkpoint_path"]) / "metadata.json").exists()


def dependencies_satisfied(status: dict[str, Any], run: dict[str, Any]) -> bool:
    lookup = runs_by_id(status)
    for dep_id in run.get("dependencies", []):
        dep = lookup.get(dep_id)
        if dep is None or not final_checkpoint_exists(dep):
            return False
    restore_checkpoint = run.get("restore_checkpoint")
    if restore_checkpoint and not (abs_repo_path(restore_checkpoint) / "metadata.json").exists():
        return False
    return True


def parse_jsonl_tail(path: Path, max_rows: int = 200) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(rows) > max_rows:
                rows.pop(0)
    return rows


def latest_health(run: dict[str, Any]) -> dict[str, Any]:
    rows = parse_jsonl_tail(abs_repo_path(run["log_jsonl"]))
    if not rows:
        return {}
    last = rows[-1]
    eval_rows = [row for row in rows if "eval_loss" in row]
    recent = rows[-50:]
    drop_values = [row["moe_dropped_fraction"] for row in recent if "moe_dropped_fraction" in row]
    step_values = [row["step_time_sec"] for row in recent if "step_time_sec" in row]
    health = {
        "step": last.get("step"),
        "loss": last.get("loss"),
        "eval_loss": eval_rows[-1].get("eval_loss") if eval_rows else None,
        "eval_step": eval_rows[-1].get("step") if eval_rows else None,
        "moe_drop": last.get("moe_dropped_fraction"),
        "moe_drop_avg50": sum(drop_values) / len(drop_values) if drop_values else None,
        "moe_entropy": last.get("moe_router_entropy"),
        "step_time_avg50": sum(step_values) / len(step_values) if step_values else None,
    }
    return health


def format_health(health: dict[str, Any]) -> str:
    if not health:
        return "no metrics yet"
    parts = [f"step={health.get('step')}"]
    for key in ("loss", "eval_loss", "moe_drop", "moe_drop_avg50", "moe_entropy", "step_time_avg50"):
        value = health.get(key)
        if value is None:
            continue
        parts.append(f"{key}={value:.4g}")
    return " ".join(parts)


def bad_float(value: Any) -> bool:
    return isinstance(value, (int, float)) and not math.isfinite(float(value))


def health_is_bad(run: dict[str, Any], health: dict[str, Any], args: argparse.Namespace) -> str | None:
    if not health:
        return None
    for key in ("loss", "eval_loss", "moe_drop", "moe_entropy", "step_time_avg50"):
        if bad_float(health.get(key)):
            return f"{key} is not finite"
    loss = health.get("loss")
    eval_loss = health.get("eval_loss")
    if isinstance(loss, (int, float)) and loss > args.max_loss:
        return f"loss {loss:.4g} exceeds {args.max_loss:g}"
    if isinstance(eval_loss, (int, float)) and eval_loss > args.max_eval_loss:
        return f"eval_loss {eval_loss:.4g} exceeds {args.max_eval_loss:g}"

    step = health.get("step")
    source_step = run.get("source_step_count") or 0
    post_switch_steps = step - source_step if isinstance(step, int) else 0
    drop_avg = health.get("moe_drop_avg50")
    if (
        run.get("source") != "scratch"
        and post_switch_steps >= args.moe_drop_grace_steps
        and isinstance(drop_avg, (int, float))
        and drop_avg > args.max_moe_drop_avg50
    ):
        return f"avg50 moe drop {drop_avg:.4g} exceeds {args.max_moe_drop_avg50:g}"
    return None


def process_running_for_run(run_id: str) -> bool:
    proc = subprocess.run(
        ["pgrep", "-af", run_id],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return proc.returncode == 0 and bool(proc.stdout.strip())


def mark_run(run_id: str, status: str, note: str) -> None:
    subprocess.run(
        [sys.executable, str(LAUNCHER), "mark", run_id, status, "--note", note],
        cwd=str(REPO_ROOT),
        check=True,
    )


def parse_metrics_from_line(line: str) -> dict[str, float]:
    metrics = {}
    for key, value in METRIC_RE.findall(line):
        try:
            metrics[key] = float(value)
        except ValueError:
            continue
    return metrics


def stream_launch(run: dict[str, Any], args: argparse.Namespace) -> int:
    cmd = [sys.executable, str(LAUNCHER), "launch", run["id"]]
    if args.dry_run:
        print("DRY RUN:", " ".join(cmd), flush=True)
        return 0

    print(f"\n=== launching {run['id']} ===", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    last_health_print = 0.0
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
            metrics = parse_metrics_from_line(line)
            if not metrics:
                continue
            health = latest_health(run)
            reason = health_is_bad(run, health, args)
            if reason is not None:
                print(f"\nhealth stop for {run['id']}: {reason}", flush=True)
                proc.terminate()
                try:
                    return proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    return proc.wait()
            now = time.time()
            if now - last_health_print >= args.health_print_sec:
                print(f"health {run['id']}: {format_health(health)}", flush=True)
                last_health_print = now
    except KeyboardInterrupt:
        proc.terminate()
        raise
    return proc.wait()


def wait_for_running_target(run: dict[str, Any], args: argparse.Namespace) -> None:
    print(f"waiting for active run {run['id']} to finish", flush=True)
    last_step = None
    last_change = time.time()
    while True:
        status = load_status(args.status_path)
        current = runs_by_id(status)[run["id"]]
        health = latest_health(current)
        print(f"active {run['id']}: {format_health(health)}", flush=True)
        reason = health_is_bad(current, health, args)
        if reason is not None:
            raise RuntimeError(f"active run {run['id']} looks unhealthy: {reason}")
        step = health.get("step")
        if step != last_step:
            last_step = step
            last_change = time.time()
        elif time.time() - last_change > args.stale_sec:
            if args.recover_stale and not process_running_for_run(run["id"]):
                note = f"Queue wrapper recovered stale running status after {args.stale_sec:g}s with no matching process."
                print(note, flush=True)
                mark_run(run["id"], "stopped", note)
                return
            raise RuntimeError(f"active run {run['id']} has not advanced for {args.stale_sec:g}s")
        if current.get("status") in TERMINAL_STATUSES or final_checkpoint_exists(current):
            return
        time.sleep(args.poll_sec)


def choose_next(status: dict[str, Any], target_ids: list[str]) -> dict[str, Any] | None:
    lookup = runs_by_id(status)
    for run_id in target_ids:
        run = lookup.get(run_id)
        if run is None:
            raise KeyError(f"{run_id} is missing from status.json")
        if run.get("status") == "pending" and dependencies_satisfied(status, run):
            return run
    return None


def active_target(status: dict[str, Any], target_ids: list[str]) -> dict[str, Any] | None:
    lookup = runs_by_id(status)
    for run_id in target_ids:
        run = lookup.get(run_id)
        if run and run.get("status") == "running":
            return run
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-path", type=Path, default=STATUS_PATH)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--run-id",
        action="append",
        default=None,
        help="Explicit run id to include in the queue. Repeat to set order.",
    )
    parser.add_argument("--poll-sec", type=float, default=60.0)
    parser.add_argument("--stale-sec", type=float, default=900.0)
    parser.add_argument("--health-print-sec", type=float, default=300.0)
    parser.add_argument("--max-loss", type=float, default=20.0)
    parser.add_argument("--max-eval-loss", type=float, default=20.0)
    parser.add_argument("--max-moe-drop-avg50", type=float, default=0.25)
    parser.add_argument("--moe-drop-grace-steps", type=int, default=100)
    parser.add_argument("--recover-stale", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    targets = args.run_id or default_run_ids()
    print("target queue:", *targets, sep="\n  ", flush=True)

    while True:
        status = load_status(args.status_path)
        running = active_target(status, targets)
        if running is not None:
            wait_for_running_target(running, args)
            continue

        run = choose_next(status, targets)
        if run is None:
            print("all target runs are complete or non-pending", flush=True)
            return 0

        return_code = stream_launch(run, args)
        if return_code != 0:
            print(f"{run['id']} exited with {return_code}; stopping queue", flush=True)
            return return_code


if __name__ == "__main__":
    raise SystemExit(main())
