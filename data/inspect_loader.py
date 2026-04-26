"""Inspect token batches from a tokenized `.npy` path or shard directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.loader import inspect_batches


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--context-length", type=int, default=128)
    p.add_argument("--batch-sizes", type=str, default="1,4,16")
    p.add_argument("--samples-per-size", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    batch_sizes = tuple(int(x) for x in args.batch_sizes.split(",") if x)
    rows = inspect_batches(
        args.path,
        context_length=args.context_length,
        batch_sizes=batch_sizes,
        samples_per_size=args.samples_per_size,
        seed=args.seed,
    )
    for row in rows:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
