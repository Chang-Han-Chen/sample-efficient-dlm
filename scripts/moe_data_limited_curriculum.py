#!/usr/bin/env python3
"""Data-limited BD3 curriculum launcher for the MoE old-bundle model.

The intended flow is:

1. Prepare a 10-shard ClimbMix source with ``data/prepare_climbmix.py``.
2. Derive 0.5x, 1x, and 2x shard datasets from that tokenized source.
3. Launch the first U=25M-ish cells for ``p_ar in {0.0, 0.3}`` and
   ``wd in {0.0, 0.1}``, where WD is applied to Muon, Adam, and router Adam.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

SOURCE_DATA_ROOT = REPO_ROOT / "data/climbmix_10x_newtok_8192"
RUN_ROOT = REPO_ROOT / "runs/moe_data_limited_curriculum"
DEFAULT_STATUS_PATH = RUN_ROOT / "status.json"
LAUNCHER_LOG_DIR = RUN_ROOT / "_launcher_logs"

DATASET_SPECS: dict[str, dict[str, Any]] = {
    "u25mish": {
        "name": "climbmix_0p5x_newtok_8192",
        "shard_multiple": 0.5,
        "description": "first half of shard_00000",
    },
    "u50mish": {
        "name": "climbmix_1x_newtok_8192",
        "shard_multiple": 1.0,
        "description": "all of shard_00000",
    },
    "u100mish": {
        "name": "climbmix_2x_newtok_8192",
        "shard_multiple": 2.0,
        "description": "all of shard_00000 and shard_00001",
    },
}

AR_CONFIG = "configs/experiments/ar_moe_old_bundle.yaml"
BD3_CONFIG = "configs/experiments/mdlm_moe_old_bundle.yaml"

DEFAULT_EPOCHS = 32.0
DEFAULT_BATCH_SIZE = 512
DEFAULT_CONTEXT_LENGTH = 512
DEFAULT_GRAD_ACCUM_STEPS = 1
DEFAULT_NUM_DEVICES = 4
DEFAULT_BASE_VOCAB_SIZE = 8192
DEFAULT_MASK_TOKEN_ID = 8192
DEFAULT_BD3_BLOCK_LEN = 4
DEFAULT_EVAL_BATCHES = 4
DEFAULT_LOG_EVERY = 200
DEFAULT_WARMUP_STEPS = 100
DEFAULT_SEED = 42
DEFAULT_P_VALUES = (0.0, 0.3)
DEFAULT_WD_VALUES = (0.0, 0.1)
DEFAULT_AR_LR_MULT = 5.0
DEFAULT_BD3_LR_MULT = 2.0
WANDB_GROUP = "moe_data_limited_curriculum"

TERMINAL_STATUSES = {"completed", "failed", "stopped", "skipped"}
VALID_STATUSES = TERMINAL_STATUSES | {"pending", "running"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def abs_path(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def float_tag(value: float) -> str:
    return f"{value:.12g}".replace("-", "m").replace(".", "p")


def p_tag(value: float) -> str:
    return f"p{int(round(value * 100)):03d}"


def wd_tag(value: float) -> str:
    return f"allwd{float_tag(value)}"


def token_arrays(path: Path) -> list[Path]:
    if path.is_dir():
        arrays = sorted(p for p in path.glob("*.npy") if "offsets_" not in p.name)
    else:
        arrays = [path]
    if not arrays:
        raise SystemExit(f"No .npy token arrays found at {path}")
    return arrays


def count_tokens(path: Path) -> int:
    total = 0
    for npy in token_arrays(path):
        total += int(np.load(npy, mmap_mode="r").shape[0])
    return total


def tokens_per_step(*, batch_size: int, context_length: int, grad_accum_steps: int) -> int:
    return int(batch_size) * int(context_length) * int(grad_accum_steps)


def steps_for_epochs(
    token_count: int,
    *,
    epochs: float,
    batch_size: int,
    context_length: int,
    grad_accum_steps: int,
) -> int:
    return math.ceil(
        float(epochs)
        * float(token_count)
        / float(
            tokens_per_step(
                batch_size=batch_size,
                context_length=context_length,
                grad_accum_steps=grad_accum_steps,
            )
        )
    )


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def ensure_symlink(source: Path, dest: Path, *, overwrite: bool) -> None:
    source = source.resolve()
    if dest.exists() or dest.is_symlink():
        if dest.is_symlink() and dest.resolve() == source:
            return
        if not overwrite:
            raise SystemExit(f"{dest} already exists; pass --overwrite to replace it")
        remove_path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    target = os.path.relpath(source, start=dest.parent.resolve())
    os.symlink(target, dest)


def write_prefix_array(source: Path, dest: Path, token_count: int, *, overwrite: bool) -> dict[str, Any]:
    if dest.exists() and not overwrite:
        arr = np.load(dest, mmap_mode="r")
        if int(arr.shape[0]) == token_count:
            return {
                "path": rel(dest),
                "source": rel(source),
                "used_tokens": token_count,
                "mode": "existing_prefix",
            }
        raise SystemExit(f"{dest} already exists with shape {arr.shape}; pass --overwrite")
    if dest.exists():
        dest.unlink()

    src = np.load(source, mmap_mode="r")
    if token_count > int(src.shape[0]):
        raise SystemExit(f"Requested {token_count} tokens from {source}, only {src.shape[0]} available")

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=src.dtype, shape=(token_count,))
    out[:] = src[:token_count]
    out.flush()
    del out
    tmp.replace(dest)
    return {
        "path": rel(dest),
        "source": rel(source),
        "available_tokens": int(src.shape[0]),
        "used_tokens": token_count,
        "mode": "copied_prefix",
    }


def prepare_one_dataset(
    *,
    label: str,
    source_root: Path,
    output_parent: Path,
    batch_size: int,
    context_length: int,
    grad_accum_steps: int,
    epochs: float,
    overwrite: bool,
) -> dict[str, Any]:
    spec = DATASET_SPECS[label]
    source_train = source_root / "tokens/train"
    source_val = source_root / "tokens/val"
    source_tokenizer = source_root / "tokenizer"
    source_metadata_path = source_root / "metadata.json"
    if not source_train.exists():
        raise SystemExit(f"Missing source train tokens: {source_train}")
    if not source_val.exists():
        raise SystemExit(f"Missing source eval tokens: {source_val}")
    if not source_tokenizer.exists():
        raise SystemExit(f"Missing source tokenizer: {source_tokenizer}")

    arrays = token_arrays(source_train)
    needed_full_shards = math.ceil(float(spec["shard_multiple"]))
    if len(arrays) < needed_full_shards:
        raise SystemExit(
            f"{label} needs {needed_full_shards} train shard(s), but only found {len(arrays)} at {source_train}"
        )

    data_root = output_parent / spec["name"]
    train_dir = data_root / "tokens/train"
    train_dir.mkdir(parents=True, exist_ok=True)

    selected: list[dict[str, Any]] = []
    multiple = float(spec["shard_multiple"])
    if multiple < 1.0:
        source = arrays[0]
        source_tokens = int(np.load(source, mmap_mode="r").shape[0])
        target_tokens = math.floor(source_tokens * multiple)
        selected.append(
            write_prefix_array(source, train_dir / source.name, target_tokens, overwrite=overwrite)
        )
    else:
        full_count = int(multiple)
        if not math.isclose(multiple, float(full_count)):
            raise SystemExit("Only sub-1.0 fractional specs are currently supported")
        for source in arrays[:full_count]:
            dest = train_dir / source.name
            ensure_symlink(source, dest, overwrite=overwrite)
            selected.append(
                {
                    "path": rel(dest),
                    "source": rel(source),
                    "used_tokens": int(np.load(source, mmap_mode="r").shape[0]),
                    "mode": "symlink_full_shard",
                }
            )

    ensure_symlink(source_val, data_root / "tokens/val", overwrite=overwrite)
    ensure_symlink(source_tokenizer, data_root / "tokenizer", overwrite=overwrite)

    train_token_count = sum(int(item["used_tokens"]) for item in selected)
    step_count = steps_for_epochs(
        train_token_count,
        epochs=epochs,
        batch_size=batch_size,
        context_length=context_length,
        grad_accum_steps=grad_accum_steps,
    )
    metadata = {
        "name": data_root.name,
        "data_limited_curriculum_label": label,
        "source_data_root": rel(source_root),
        "source_metadata": (
            json.loads(source_metadata_path.read_text()) if source_metadata_path.exists() else None
        ),
        "selection": {
            "method": "shard_multiple",
            "shard_multiple": multiple,
            "description": spec["description"],
            "source_files": selected,
        },
        "vocab_size": DEFAULT_BASE_VOCAB_SIZE,
        "mask_token_id": DEFAULT_MASK_TOKEN_ID,
        "tokenizer_dir": rel(data_root / "tokenizer"),
        "train_tokens": rel(train_dir),
        "val_tokens": rel(data_root / "tokens/val"),
        "train_token_count": train_token_count,
        "default_batch_size": batch_size,
        "default_context_length": context_length,
        "default_grad_accum_steps": grad_accum_steps,
        "tokens_per_optimizer_step": tokens_per_step(
            batch_size=batch_size,
            context_length=context_length,
            grad_accum_steps=grad_accum_steps,
        ),
        "steps_per_epoch": train_token_count
        / float(
            tokens_per_step(
                batch_size=batch_size,
                context_length=context_length,
                grad_accum_steps=grad_accum_steps,
            )
        ),
        "epochs": epochs,
        "steps_for_epochs": step_count,
    }
    data_root.mkdir(parents=True, exist_ok=True)
    (data_root / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def cmd_prepare_datasets(args: argparse.Namespace) -> int:
    source_root = args.source_data_root.resolve()
    output_parent = args.output_parent.resolve()
    summaries = []
    for label in args.labels:
        metadata = prepare_one_dataset(
            label=label,
            source_root=source_root,
            output_parent=output_parent,
            batch_size=args.batch_size,
            context_length=args.context_length,
            grad_accum_steps=args.grad_accum_steps,
            epochs=args.epochs,
            overwrite=args.overwrite,
        )
        summaries.append(
            {
                "label": label,
                "data_root": rel(output_parent / DATASET_SPECS[label]["name"]),
                "train_token_count": metadata["train_token_count"],
                "steps_for_epochs": metadata["steps_for_epochs"],
                "steps_per_epoch": round(float(metadata["steps_per_epoch"]), 3),
            }
        )
    print(json.dumps(summaries, indent=2))
    return 0


def default_data_root(label: str) -> Path:
    return REPO_ROOT / "data" / DATASET_SPECS[label]["name"]


def load_dataset_metadata(data_root: Path) -> dict[str, Any]:
    metadata_path = data_root / "metadata.json"
    if not metadata_path.exists():
        raise SystemExit(f"Missing dataset metadata: {metadata_path}")
    return json.loads(metadata_path.read_text())


def run_output_dir(run_id: str) -> Path:
    return RUN_ROOT / run_id


def log_jsonl_path(run_id: str) -> Path:
    return run_output_dir(run_id) / "train.jsonl"


def checkpoint_path(run_id: str, kind: str = "final") -> Path:
    return run_output_dir(run_id) / "checkpoints" / kind


def make_run(
    *,
    run_id: str,
    phase: str,
    objective: str,
    data_label: str,
    data_root: Path,
    train_tokens: int,
    max_steps: int,
    wd: float,
    lr_mult: float,
    p_ar: float,
    dependencies: list[str] | None = None,
    restore_checkpoint: Path | None = None,
) -> dict[str, Any]:
    return {
        "id": run_id,
        "status": "pending",
        "phase": phase,
        "objective": objective,
        "data_label": data_label,
        "data_root": rel(data_root),
        "train_path": rel(data_root / "tokens/train"),
        "eval_path": rel(data_root / "tokens/val"),
        "train_tokens": train_tokens,
        "p_ar": p_ar,
        "bd3_block_len": DEFAULT_BD3_BLOCK_LEN if objective == "bd3lm" else None,
        "wd": wd,
        "muon_wd": wd,
        "adam_wd": wd,
        "router_adam_wd": wd,
        "lr_mult": lr_mult,
        "max_steps": max_steps,
        "dependencies": dependencies or [],
        "restore_checkpoint": None if restore_checkpoint is None else rel(restore_checkpoint),
        "run_name": run_id,
        "output_dir": rel(run_output_dir(run_id)),
        "log_jsonl": rel(log_jsonl_path(run_id)),
        "final_checkpoint": rel(checkpoint_path(run_id, "final")),
        "best_checkpoint": rel(checkpoint_path(run_id, "best")),
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "attempts": 0,
        "metrics": {},
        "notes": [],
    }


def build_runs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runs: list[dict[str, Any]] = []
    dataset_summaries = []
    for label in args.data_labels:
        data_root = default_data_root(label) if args.data_root is None else args.data_root.resolve()
        metadata = load_dataset_metadata(data_root)
        train_tokens = int(metadata.get("train_token_count") or count_tokens(data_root / "tokens/train"))
        total_steps = steps_for_epochs(
            train_tokens,
            epochs=args.epochs,
            batch_size=args.batch_size,
            context_length=args.context_length,
            grad_accum_steps=args.grad_accum_steps,
        )
        dataset_summaries.append(
            {
                "label": label,
                "data_root": rel(data_root),
                "train_tokens": train_tokens,
                "total_steps": total_steps,
                "epochs": args.epochs,
            }
        )
        for wd in args.wd_values:
            for p_ar in args.p_values:
                pslug = p_tag(p_ar)
                wslug = wd_tag(wd)
                if math.isclose(p_ar, 0.0):
                    run_id = f"{label}_{pslug}_bd3_b{DEFAULT_BD3_BLOCK_LEN}_{wslug}"
                    runs.append(
                        make_run(
                            run_id=run_id,
                            phase="scratch_bd3",
                            objective="bd3lm",
                            data_label=label,
                            data_root=data_root,
                            train_tokens=train_tokens,
                            max_steps=total_steps,
                            wd=wd,
                            lr_mult=args.bd3_lr_mult,
                            p_ar=p_ar,
                        )
                    )
                else:
                    ar_steps = max(1, int(round(float(p_ar) * total_steps)))
                    ar_run_id = f"{label}_{pslug}_ar_{wslug}"
                    bd3_run_id = f"{label}_{pslug}_bd3_b{DEFAULT_BD3_BLOCK_LEN}_{wslug}"
                    runs.append(
                        make_run(
                            run_id=ar_run_id,
                            phase="ar_prefix",
                            objective="ar",
                            data_label=label,
                            data_root=data_root,
                            train_tokens=train_tokens,
                            max_steps=ar_steps,
                            wd=wd,
                            lr_mult=args.ar_lr_mult,
                            p_ar=p_ar,
                        )
                    )
                    runs.append(
                        make_run(
                            run_id=bd3_run_id,
                            phase="bd3_after_ar",
                            objective="bd3lm",
                            data_label=label,
                            data_root=data_root,
                            train_tokens=train_tokens,
                            max_steps=total_steps,
                            wd=wd,
                            lr_mult=args.bd3_lr_mult,
                            p_ar=p_ar,
                            dependencies=[ar_run_id],
                            restore_checkpoint=checkpoint_path(ar_run_id, "final"),
                        )
                    )
    return runs, dataset_summaries


def write_status(path: Path, status: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    status["updated_at"] = now_iso()
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def load_status(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return json.load(fh)


def cmd_init(args: argparse.Namespace) -> int:
    if args.status_path.exists() and not args.force:
        print(f"{args.status_path} already exists; pass --force to overwrite", file=sys.stderr)
        return 2
    runs, dataset_summaries = build_runs(args)
    status = {
        "schema_version": 1,
        "experiment": "moe_data_limited_curriculum",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "repo_root": str(REPO_ROOT),
        "launcher": "scripts/moe_data_limited_curriculum.py",
        "instruction_file": "moe_data_limited.md",
        "controls": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "grad_accum_steps": args.grad_accum_steps,
            "tokens_per_optimizer_step": tokens_per_step(
                batch_size=args.batch_size,
                context_length=args.context_length,
                grad_accum_steps=args.grad_accum_steps,
            ),
            "num_devices": args.num_devices,
            "p_values": args.p_values,
            "wd_values": args.wd_values,
            "wd_scope": "muon_wd = adam_wd = router_adam_wd",
            "ar_lr_mult": args.ar_lr_mult,
            "bd3_lr_mult": args.bd3_lr_mult,
            "bd3_block_len": DEFAULT_BD3_BLOCK_LEN,
            "eval_batches": args.eval_batches,
            "log_every": args.log_every,
            "save_best_checkpoint": True,
            "save_final_checkpoint": True,
            "wandb_group": WANDB_GROUP,
        },
        "datasets": dataset_summaries,
        "runs": runs,
    }
    write_status(args.status_path, status)
    print(f"initialized {args.status_path} with {len(runs)} process run(s)")
    return 0


def runs_by_id(status: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {run["id"]: run for run in status["runs"]}


def find_run(status: dict[str, Any], run_id: str) -> dict[str, Any]:
    for run in status["runs"]:
        if run["id"] == run_id:
            return run
    raise SystemExit(f"unknown run id: {run_id}")


def final_checkpoint_exists(run: dict[str, Any]) -> bool:
    return (abs_path(run["final_checkpoint"]) / "metadata.json").exists()


def dependencies_satisfied(status: dict[str, Any], run: dict[str, Any]) -> bool:
    lookup = runs_by_id(status)
    for dep_id in run.get("dependencies", []):
        dep = lookup.get(dep_id)
        if dep is None:
            return False
        if not final_checkpoint_exists(dep):
            return False
    restore = run.get("restore_checkpoint")
    if restore and not (abs_path(restore) / "metadata.json").exists():
        return False
    return True


def run_matches(
    run: dict[str, Any],
    *,
    status_filter: str | None = None,
    phase: str | None = None,
    data_label: str | None = None,
    objective: str | None = None,
) -> bool:
    if status_filter is not None and run.get("status") != status_filter:
        return False
    if phase is not None and run.get("phase") != phase:
        return False
    if data_label is not None and run.get("data_label") != data_label:
        return False
    if objective is not None and run.get("objective") != objective:
        return False
    return True


def select_runs(
    status: dict[str, Any],
    *,
    status_filter: str | None = None,
    phase: str | None = None,
    data_label: str | None = None,
    objective: str | None = None,
    eligible_only: bool = False,
) -> list[dict[str, Any]]:
    selected = []
    for run in status["runs"]:
        if not run_matches(
            run,
            status_filter=status_filter,
            phase=phase,
            data_label=data_label,
            objective=objective,
        ):
            continue
        if eligible_only and not dependencies_satisfied(status, run):
            continue
        selected.append(run)
    return selected


def command_for_run(run: dict[str, Any], controls: dict[str, Any]) -> list[str]:
    python = os.environ.get("PYTHON", sys.executable or "python")
    config = AR_CONFIG if run["objective"] == "ar" else BD3_CONFIG
    cmd = [
        python,
        "train_ar.py",
        "--config",
        config,
        "--objective",
        run["objective"],
        "--train-path",
        run["train_path"],
        "--eval-path",
        run["eval_path"],
        "--seed",
        str(DEFAULT_SEED),
        "--run-name",
        run["run_name"],
        "--output-dir",
        run["output_dir"],
        "--log-jsonl",
        run["log_jsonl"],
        "--max-steps",
        str(run["max_steps"]),
        "--warmup-steps",
        str(DEFAULT_WARMUP_STEPS),
        "--batch-size",
        str(controls["batch_size"]),
        "--grad-accum-steps",
        str(controls["grad_accum_steps"]),
        "--data-parallel",
        "--num-devices",
        str(controls["num_devices"]),
        "--context-length",
        str(controls["context_length"]),
        "--attention-impl",
        "cudnn",
        "--attn-val-residual",
        "--no-moe-split-router-input",
        "--moe-router-z-loss-weight",
        "0.01",
        "--lr-mult",
        f"{float(run['lr_mult']):.12g}",
        "--muon-wd",
        f"{float(run['muon_wd']):.12g}",
        "--adam-wd",
        f"{float(run['adam_wd']):.12g}",
        "--router-adam-wd",
        f"{float(run['router_adam_wd']):.12g}",
        "--eval-batches",
        str(controls["eval_batches"]),
        "--log-every",
        str(controls["log_every"]),
        "--checkpoint-interval",
        "0",
        "--no-wandb-checkpoints",
        "--wandb-group",
        controls["wandb_group"],
    ]
    restore = run.get("restore_checkpoint")
    if restore:
        cmd.extend(["--restore-checkpoint", restore])
    if run["objective"] == "ar":
        cmd.extend(["--vocab-size", str(DEFAULT_BASE_VOCAB_SIZE + 1)])
    else:
        cmd.extend(
            [
                "--vocab-size",
                str(DEFAULT_BASE_VOCAB_SIZE),
                "--mask-token-id",
                str(DEFAULT_MASK_TOKEN_ID),
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
                "--bd3-block-len",
                str(DEFAULT_BD3_BLOCK_LEN),
                "--bd3-attention",
                "dense",
            ]
        )
    tags = [
        "moe_data_limited",
        run["data_label"],
        run["phase"],
        run["objective"],
        p_tag(float(run["p_ar"])),
        wd_tag(float(run["wd"])),
        f"b{DEFAULT_BD3_BLOCK_LEN}" if run["objective"] == "bd3lm" else "ar_prefix",
    ]
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
    best_metadata = abs_path(run["best_checkpoint"]) / "metadata.json"
    final_metadata = abs_path(run["final_checkpoint"]) / "metadata.json"
    if best_metadata.exists():
        metadata = json.loads(best_metadata.read_text())
        best_metrics = metadata.get("metrics") or {}
        metrics["best_checkpoint_step"] = metadata.get("step")
        if "eval_loss" in best_metrics:
            metrics["best_checkpoint_eval_loss"] = best_metrics["eval_loss"]
    if final_metadata.exists():
        metadata = json.loads(final_metadata.read_text())
        final_metrics = metadata.get("metrics") or {}
        metrics["final_checkpoint_step"] = metadata.get("step")
        for key in ("avg_measured_step_sec", "tokens_per_sec", "jax_peak_hbm_gb"):
            if key in final_metrics:
                metrics[key] = final_metrics[key]
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


def cmd_list(args: argparse.Namespace) -> int:
    status = load_status(args.status_path)
    runs = select_runs(
        status,
        status_filter=args.status,
        phase=args.phase,
        data_label=args.data_label,
        objective=args.objective,
        eligible_only=args.eligible,
    )
    for run in runs:
        dep_ok = "yes" if dependencies_satisfied(status, run) else "no"
        metrics = run.get("metrics", {})
        best = metrics.get("best_eval_loss")
        final = metrics.get("final_eval_loss")
        metric_text = ""
        if best is not None:
            metric_text += f" best={best:.6f}"
        if final is not None:
            metric_text += f" final={final:.6f}"
        print(
            f"{run['id']:<40} {run['status']:<10} {run['phase']:<12} "
            f"obj={run['objective']:<5} U={run['data_label']:<8} "
            f"p={run['p_ar']:<4} wd={run['wd']:<4} steps={run['max_steps']:<6} "
            f"deps={dep_ok}{metric_text}"
        )
    print(f"{len(runs)} run(s)")
    return 0


def cmd_command(args: argparse.Namespace) -> int:
    status = load_status(args.status_path)
    run = find_run(status, args.run_id)
    cmd = command_for_run(run, status["controls"])
    if args.json:
        print(json.dumps(cmd, indent=2))
    else:
        print(quote_command(cmd))
    return 0


def pick_next_run(args: argparse.Namespace, status: dict[str, Any]) -> dict[str, Any] | None:
    candidates = select_runs(
        status,
        status_filter="pending",
        phase=args.phase,
        data_label=args.data_label,
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
        print(quote_command(command_for_run(run, status["controls"])))
    return 0


def launch_run(args: argparse.Namespace, status: dict[str, Any], run: dict[str, Any]) -> int:
    if run["status"] not in {"pending", "failed", "stopped"} and not args.force:
        print(f"{run['id']} has status {run['status']}; pass --force", file=sys.stderr)
        return 2
    if not dependencies_satisfied(status, run) and not args.force:
        print(f"{run['id']} dependencies are not satisfied", file=sys.stderr)
        return 2

    cmd = command_for_run(run, status["controls"])
    log_path = LAUNCHER_LOG_DIR / f"{run['id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    run["status"] = "running"
    run["attempts"] = int(run.get("attempts", 0)) + 1
    run["started_at"] = now_iso()
    run["updated_at"] = now_iso()
    run["launcher_log"] = rel(log_path)
    run["command"] = cmd
    write_status(args.status_path, status)

    print(quote_command(cmd), flush=True)
    started = time.time()
    proc: subprocess.Popen[str] | None = None
    try:
        with log_path.open("a") as log_fh:
            log_fh.write(f"\n# launch {now_iso()} run_id={run['id']}\n")
            log_fh.write(quote_command(cmd) + "\n")
            log_fh.flush()
            proc = subprocess.Popen(
                cmd,
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
        return_code = 130
        run["status"] = "stopped"
    else:
        run["status"] = "completed" if return_code == 0 else "failed"

    run["exit_code"] = return_code
    run["completed_at"] = now_iso()
    run["wall_time_sec"] = time.time() - started
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
        if run.get("metrics", {}).get("best_eval_loss") is not None
        or run.get("metrics", {}).get("final_eval_loss") is not None
    ]
    for run in rows:
        metrics = run["metrics"]
        print(
            f"{run['id']:<40} {run['phase']:<12} p={run['p_ar']:<4} wd={run['wd']:<4} "
            f"best={metrics.get('best_eval_loss')} final={metrics.get('final_eval_loss')}"
        )
    print(f"updated={changed} summarized={len(rows)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--status-path", type=Path, default=DEFAULT_STATUS_PATH)
    sub = p.add_subparsers(dest="command_name", required=True)

    prep = sub.add_parser("prepare-datasets", help="derive 0.5x/1x/2x datasets from tokenized ClimbMix")
    prep.add_argument("--source-data-root", type=Path, default=SOURCE_DATA_ROOT)
    prep.add_argument("--output-parent", type=Path, default=REPO_ROOT / "data")
    prep.add_argument("--labels", nargs="+", choices=sorted(DATASET_SPECS), default=list(DATASET_SPECS))
    prep.add_argument("--epochs", type=float, default=DEFAULT_EPOCHS)
    prep.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    prep.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    prep.add_argument("--grad-accum-steps", type=int, default=DEFAULT_GRAD_ACCUM_STEPS)
    prep.add_argument("--overwrite", action="store_true")
    prep.set_defaults(func=cmd_prepare_datasets)

    init = sub.add_parser("init", help="initialize a status queue")
    init.add_argument("--force", action="store_true")
    init.add_argument("--data-labels", nargs="+", choices=sorted(DATASET_SPECS), default=["u25mish"])
    init.add_argument("--data-root", type=Path, default=None, help="single explicit data root; mainly for tests")
    init.add_argument("--epochs", type=float, default=DEFAULT_EPOCHS)
    init.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    init.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    init.add_argument("--grad-accum-steps", type=int, default=DEFAULT_GRAD_ACCUM_STEPS)
    init.add_argument("--num-devices", type=int, default=DEFAULT_NUM_DEVICES)
    init.add_argument("--p-values", nargs="+", type=float, default=list(DEFAULT_P_VALUES))
    init.add_argument("--wd-values", nargs="+", type=float, default=list(DEFAULT_WD_VALUES))
    init.add_argument("--ar-lr-mult", type=float, default=DEFAULT_AR_LR_MULT)
    init.add_argument("--bd3-lr-mult", type=float, default=DEFAULT_BD3_LR_MULT)
    init.add_argument("--eval-batches", type=int, default=DEFAULT_EVAL_BATCHES)
    init.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY)
    init.set_defaults(func=cmd_init)

    list_p = sub.add_parser("list", help="list runs")
    list_p.add_argument("--status", choices=sorted(VALID_STATUSES), default=None)
    list_p.add_argument("--phase", default=None)
    list_p.add_argument("--data-label", choices=sorted(DATASET_SPECS), default=None)
    list_p.add_argument("--objective", choices=("ar", "bd3lm"), default=None)
    list_p.add_argument("--eligible", action="store_true")
    list_p.set_defaults(func=cmd_list)

    command_p = sub.add_parser("command", help="print the train_ar.py command for a run")
    command_p.add_argument("run_id")
    command_p.add_argument("--json", action="store_true")
    command_p.set_defaults(func=cmd_command)

    next_p = sub.add_parser("next", help="print the next eligible pending run")
    next_p.add_argument("--phase", default=None)
    next_p.add_argument("--data-label", choices=sorted(DATASET_SPECS), default=None)
    next_p.add_argument("--objective", choices=("ar", "bd3lm"), default=None)
    next_p.add_argument("--command", action="store_true")
    next_p.set_defaults(func=cmd_next)

    launch_p = sub.add_parser("launch", help="launch one run in the foreground")
    launch_p.add_argument("run_id")
    launch_p.add_argument("--force", action="store_true")
    launch_p.set_defaults(func=cmd_launch)

    launch_next_p = sub.add_parser("launch-next", help="launch the next eligible pending run")
    launch_next_p.add_argument("--phase", default=None)
    launch_next_p.add_argument("--data-label", choices=sorted(DATASET_SPECS), default=None)
    launch_next_p.add_argument("--objective", choices=("ar", "bd3lm"), default=None)
    launch_next_p.add_argument("--force", action="store_true")
    launch_next_p.set_defaults(func=cmd_launch_next)

    summarize_p = sub.add_parser("summarize", help="parse logs/checkpoints and update metrics")
    summarize_p.set_defaults(func=cmd_summarize)
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
