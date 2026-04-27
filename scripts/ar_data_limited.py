#!/usr/bin/env python3
"""Small AR data-limited experiment helper.

This script intentionally does not modify the existing trainer or config files.
It prepares a fixed 25M-token training split and launches one AR dense/MoE run
at a time with explicit CLI hyperparameters.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATA_ROOT = REPO_ROOT / "data/climbmix_24x_newtok_8192"
DEFAULT_DATA_ROOT = REPO_ROOT / "data/climbmix_25m_newtok_8192"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs/data_limited_ar_25m"
DEFAULT_TOKENS = 25_000_000
DEFAULT_BATCH_SIZE = 512
DEFAULT_CONTEXT_LENGTH = 512
DEFAULT_MAX_STEPS = 5100
DEFAULT_WD_VALUES = (0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4)


MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "dense": {
        "config": "configs/experiments/ar_old_bundle.yaml",
        "tags": ("dense", "old_bundle"),
        "extra_args": (),
    },
    "moe": {
        "config": "configs/experiments/ar_moe_old_bundle.yaml",
        "tags": ("moe", "old_bundle", "switch", "zloss0p01", "nonsplit_router"),
        "extra_args": (
            "--lr-mult",
            "2.0",
            "--moe-router-z-loss-weight",
            "0.01",
            "--no-moe-split-router-input",
        ),
    },
}


def rel(path: Path) -> str:
    path = Path(os.path.abspath(path))
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def float_tag(value: float) -> str:
    text = f"{value:.12g}"
    return text.replace("-", "m").replace(".", "p")


def require_repo_root() -> None:
    if not (REPO_ROOT / "train_ar.py").exists():
        raise SystemExit(f"Could not find train_ar.py under {REPO_ROOT}")


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


def steps_per_epoch(tokens: int, batch_size: int, context_length: int) -> float:
    return tokens / float(batch_size * context_length)


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def ensure_dir_symlink(source: Path, dest: Path, *, overwrite: bool) -> None:
    if dest.exists() or dest.is_symlink():
        if dest.is_symlink() and dest.resolve() == source.resolve():
            return
        if not overwrite:
            raise SystemExit(f"{dest} already exists; pass --overwrite to replace it")
        remove_path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    target = os.path.relpath(source.resolve(), start=dest.parent.resolve())
    os.symlink(target, dest)


def prepare_fixed_tokens(args: argparse.Namespace) -> None:
    require_repo_root()
    source_root = args.source_data_root.resolve()
    data_root = args.data_root.resolve()
    source_train = source_root / "tokens/train"
    source_val = source_root / "tokens/val"
    source_tokenizer = source_root / "tokenizer"

    if not source_train.exists():
        raise SystemExit(f"Missing source train tokens: {source_train}")
    if not source_val.exists():
        raise SystemExit(f"Missing source val tokens: {source_val}")
    if not source_tokenizer.exists():
        raise SystemExit(f"Missing source tokenizer: {source_tokenizer}")

    train_dir = data_root / "tokens/train"
    val_link = data_root / "tokens/val"
    tokenizer_link = data_root / "tokenizer"
    out_file = train_dir / "shard_00000.npy"
    tmp_file = train_dir / "shard_00000.tmp.npy"

    train_dir.mkdir(parents=True, exist_ok=True)
    target_tokens = int(args.tokens)
    if target_tokens <= args.context_length + 1:
        raise SystemExit("--tokens must exceed context_length + 1")

    existing_ok = False
    if out_file.exists() and not args.overwrite:
        arr = np.load(out_file, mmap_mode="r")
        existing_ok = int(arr.shape[0]) == target_tokens
        if not existing_ok:
            raise SystemExit(
                f"{out_file} already has shape {arr.shape}; pass --overwrite to replace it"
            )

    consumed: list[dict[str, Any]] = []
    if not existing_ok:
        if out_file.exists():
            out_file.unlink()
        if tmp_file.exists():
            tmp_file.unlink()

        arrays = token_arrays(source_train)
        first = np.load(arrays[0], mmap_mode="r")
        out = np.lib.format.open_memmap(
            tmp_file,
            mode="w+",
            dtype=first.dtype,
            shape=(target_tokens,),
        )
        cursor = 0
        for npy in arrays:
            src = np.load(npy, mmap_mode="r")
            take = min(int(src.shape[0]), target_tokens - cursor)
            if take <= 0:
                break
            out[cursor : cursor + take] = src[:take]
            consumed.append(
                {
                    "path": rel(npy),
                    "available_tokens": int(src.shape[0]),
                    "used_tokens": int(take),
                }
            )
            cursor += take
            if cursor == target_tokens:
                break
        out.flush()
        del out
        if cursor != target_tokens:
            tmp_file.unlink(missing_ok=True)
            raise SystemExit(
                f"Source data only provided {cursor} tokens, expected {target_tokens}"
            )
        tmp_file.replace(out_file)
    else:
        consumed.append(
            {
                "path": rel(out_file),
                "available_tokens": target_tokens,
                "used_tokens": target_tokens,
                "already_prepared": True,
            }
        )

    ensure_dir_symlink(source_val, val_link, overwrite=args.overwrite)
    ensure_dir_symlink(source_tokenizer, tokenizer_link, overwrite=args.overwrite)

    source_metadata_path = source_root / "metadata.json"
    source_metadata = None
    if source_metadata_path.exists():
        source_metadata = json.loads(source_metadata_path.read_text())

    metadata = {
        "name": data_root.name,
        "source_data_root": rel(source_root),
        "source_metadata": source_metadata,
        "selection": {
            "method": "prefix",
            "tokens": target_tokens,
            "source_files": consumed,
        },
        "vocab_size": 8192,
        "mask_token_id": 8192,
        "tokenizer_dir": rel(tokenizer_link),
        "train_tokens": rel(train_dir),
        "val_tokens": rel(val_link),
        "train_token_file": rel(out_file),
        "train_token_dtype": str(np.load(out_file, mmap_mode="r").dtype),
        "train_token_count": target_tokens,
        "default_batch_size": args.batch_size,
        "default_context_length": args.context_length,
        "steps_per_epoch": steps_per_epoch(
            target_tokens,
            args.batch_size,
            args.context_length,
        ),
    }
    data_root.mkdir(parents=True, exist_ok=True)
    (data_root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    epoch_steps = metadata["steps_per_epoch"]
    print(
        json.dumps(
            {
                "prepared": rel(data_root),
                "train_tokens": target_tokens,
                "train_path": rel(train_dir),
                "val_path": rel(val_link),
                "steps_per_epoch_at_default_batch": round(epoch_steps, 2),
                "epochs_at_5100_steps": round(DEFAULT_MAX_STEPS / epoch_steps, 2),
                "steps_for_100_epochs": math.ceil(100 * epoch_steps),
            },
            indent=2,
        )
    )


def build_run_command(args: argparse.Namespace) -> tuple[list[str], Path, dict[str, Any]]:
    require_repo_root()
    preset = MODEL_PRESETS[args.model]
    data_root = args.data_root.resolve()
    train_path = data_root / "tokens/train"
    eval_path = data_root / "tokens/val"
    if not train_path.exists():
        raise SystemExit(f"Missing train path {train_path}; run the prepare subcommand first")
    if not eval_path.exists():
        raise SystemExit(f"Missing eval path {eval_path}; run the prepare subcommand first")

    train_tokens = count_tokens(train_path)
    tokens_per_step = int(args.batch_size) * int(args.context_length) * int(args.grad_accum_steps)
    epoch_steps = train_tokens / float(tokens_per_step)
    epochs = args.max_steps / epoch_steps

    wd_label = "defaultwd"
    explicit_wd = args.wd is not None
    if explicit_wd:
        wd_label = f"{args.wd_scope}wd{float_tag(args.wd)}"
    if args.muon_wd is not None:
        wd_label = f"muonwd{float_tag(args.muon_wd)}"
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = f"dl25m_ar_{args.model}_{wd_label}_s{args.max_steps}_seed{args.seed}"
    if "/" in run_name or "\\" in run_name:
        raise SystemExit("--run-name must not contain path separators")

    output_dir = args.output_root.resolve() / run_name
    log_jsonl = output_dir / "train.jsonl"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite_output and not args.dry_run:
        raise SystemExit(
            f"{output_dir} already exists and is non-empty; pass --overwrite-output to reuse it"
        )

    tags_raw = [
        "data_limited",
        "25m",
        "ar",
        args.model,
        *preset["tags"],
        wd_label,
        "no_checkpoints",
    ]
    tags_raw.extend(args.wandb_tags or [])
    tags = list(dict.fromkeys(tags_raw))

    cmd = [
        args.python,
        "train_ar.py",
        "--config",
        preset["config"],
        "--train-path",
        rel(train_path),
        "--eval-path",
        rel(eval_path),
        "--run-name",
        run_name,
        "--seed",
        str(args.seed),
        "--wandb-group",
        args.wandb_group,
        "--wandb-tags",
        *tags,
        "--output-dir",
        rel(output_dir),
        "--log-jsonl",
        rel(log_jsonl),
        "--max-steps",
        str(args.max_steps),
        "--warmup-steps",
        str(args.warmup_steps),
        "--batch-size",
        str(args.batch_size),
        "--grad-accum-steps",
        str(args.grad_accum_steps),
        "--context-length",
        str(args.context_length),
        "--num-devices",
        str(args.num_devices),
        "--log-every",
        str(args.log_every),
        "--eval-batches",
        str(args.eval_batches),
        "--checkpoint-interval",
        "0",
        "--no-save-final-checkpoint",
        "--no-save-best-checkpoint",
        "--no-wandb-checkpoints",
    ]
    if args.no_wandb:
        cmd.append("--no-wandb")

    cmd.extend(preset["extra_args"])
    if args.lr_mult is not None:
        cmd.extend(["--lr-mult", str(args.lr_mult)])

    if args.wd is not None:
        if args.wd_scope == "muon":
            cmd.extend(["--muon-wd", str(args.wd)])
        elif args.wd_scope == "all":
            cmd.extend(["--muon-wd", str(args.wd)])
            cmd.extend(["--adam-wd", str(args.wd)])
            cmd.extend(["--router-adam-wd", str(args.wd)])
        else:
            raise AssertionError(args.wd_scope)
    if args.muon_wd is not None:
        cmd.extend(["--muon-wd", str(args.muon_wd)])
    if args.adam_wd is not None:
        cmd.extend(["--adam-wd", str(args.adam_wd)])
    if args.router_adam_wd is not None:
        cmd.extend(["--router-adam-wd", str(args.router_adam_wd)])
    passthrough = list(args.train_ar_args)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    cmd.extend(passthrough)

    launch_info = {
        "run_name": run_name,
        "model": args.model,
        "command": cmd,
        "train_tokens": train_tokens,
        "tokens_per_step": tokens_per_step,
        "steps_per_epoch": epoch_steps,
        "max_steps": args.max_steps,
        "estimated_epochs": epochs,
        "checkpoints": "disabled",
        "wandb_train_loss": "logged every optimizer step",
        "wandb_eval_loss": f"logged every {args.log_every} steps",
        "output_dir": rel(output_dir),
        "log_jsonl": rel(log_jsonl),
    }
    return cmd, output_dir, launch_info


def run_one(args: argparse.Namespace) -> None:
    cmd, output_dir, launch_info = build_run_command(args)
    print(json.dumps(launch_info, indent=2))
    print("command:", " ".join(cmd))
    if args.dry_run:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "launch.json").write_text(json.dumps(launch_info, indent=2) + "\n")
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    raise SystemExit(result.returncode)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    if args.logs:
        logs = [Path(p).resolve() for p in args.logs]
    else:
        logs = sorted(root.glob("*/train.jsonl"))
    if not logs:
        raise SystemExit(f"No train.jsonl files found under {root}")

    header = (
        "run",
        "last",
        "best_eval",
        "best_step",
        "last_eval",
        "last_loss",
        "avg100_step",
        "moe_drop",
        "moe_ent",
    )
    print("\t".join(header))
    for log in logs:
        rows = read_jsonl(log)
        if not rows:
            continue
        eval_rows = [row for row in rows if "eval_loss" in row]
        best_eval = min(eval_rows, key=lambda row: row["eval_loss"]) if eval_rows else None
        last_eval = eval_rows[-1] if eval_rows else None
        recent = rows[-100:]

        def avg(key: str) -> float | None:
            vals = [float(row[key]) for row in recent if key in row]
            return sum(vals) / len(vals) if vals else None

        values = [
            log.parent.name,
            str(rows[-1].get("step", "")),
            "" if best_eval is None else f"{best_eval['eval_loss']:.6f}",
            "" if best_eval is None else str(best_eval.get("step", "")),
            "" if last_eval is None else f"{last_eval['eval_loss']:.6f}",
            f"{float(rows[-1].get('loss', float('nan'))):.6f}",
            "" if avg("step_time_sec") is None else f"{avg('step_time_sec'):.4f}",
            "" if avg("moe_dropped_fraction") is None else f"{avg('moe_dropped_fraction'):.6f}",
            "" if avg("moe_router_entropy") is None else f"{avg('moe_router_entropy'):.4f}",
        ]
        print("\t".join(values))


def print_wd_values(args: argparse.Namespace) -> None:
    values = DEFAULT_WD_VALUES
    print(" ".join(str(v) for v in values))
    if args.commands:
        for model in args.model:
            for value in values:
                print(
                    "python scripts/ar_data_limited.py run "
                    f"--model {model} --wd {value} --wd-scope {args.wd_scope}"
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="prepare fixed 25M train tokens")
    p_prepare.add_argument("--source-data-root", type=Path, default=SOURCE_DATA_ROOT)
    p_prepare.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p_prepare.add_argument("--tokens", type=int, default=DEFAULT_TOKENS)
    p_prepare.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p_prepare.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    p_prepare.add_argument("--overwrite", action="store_true")
    p_prepare.set_defaults(func=prepare_fixed_tokens)

    p_run = sub.add_parser(
        "run",
        help="launch one explicit AR data-limited run",
        epilog=(
            "Any arguments after '--' are appended directly to train_ar.py. "
            "Example: scripts/ar_data_limited.py run --model dense --wd 0.1 -- --seed 123"
        ),
    )
    p_run.add_argument("--model", choices=sorted(MODEL_PRESETS), required=True)
    p_run.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p_run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p_run.add_argument("--run-name", default=None)
    p_run.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    p_run.add_argument("--warmup-steps", type=int, default=100)
    p_run.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p_run.add_argument("--grad-accum-steps", type=int, default=1)
    p_run.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    p_run.add_argument("--num-devices", type=int, default=4)
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--log-every", type=int, default=10)
    p_run.add_argument("--eval-batches", type=int, default=4)
    p_run.add_argument("--lr-mult", type=float, default=None)
    p_run.add_argument(
        "--wd",
        type=float,
        default=None,
        help=(
            "shorthand weight-decay value; applies to --muon-wd, --adam-wd, "
            "and --router-adam-wd by default"
        ),
    )
    p_run.add_argument("--wd-scope", choices=("muon", "all"), default="all")
    p_run.add_argument("--muon-wd", type=float, default=None)
    p_run.add_argument("--adam-wd", type=float, default=None)
    p_run.add_argument("--router-adam-wd", type=float, default=None)
    p_run.add_argument("--wandb-group", default="data_limited_ar_25m")
    p_run.add_argument("--wandb-tags", nargs="*", default=None)
    p_run.add_argument("--no-wandb", action="store_true")
    p_run.add_argument("--python", default=sys.executable)
    p_run.add_argument("--dry-run", action="store_true")
    p_run.add_argument("--overwrite-output", action="store_true")
    p_run.add_argument("train_ar_args", nargs=argparse.REMAINDER)
    p_run.set_defaults(func=run_one)

    p_summary = sub.add_parser("summary", help="summarize local JSONL logs")
    p_summary.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p_summary.add_argument("logs", nargs="*")
    p_summary.set_defaults(func=summarize)

    p_wd = sub.add_parser("wd-values", help="print the planned WD sweep values")
    p_wd.add_argument("--commands", action="store_true")
    p_wd.add_argument("--model", nargs="+", choices=sorted(MODEL_PRESETS), default=["dense", "moe"])
    p_wd.add_argument("--wd-scope", choices=("muon", "all"), default="all")
    p_wd.set_defaults(func=print_wd_values)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
