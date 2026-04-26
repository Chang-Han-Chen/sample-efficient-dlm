"""Loader helpers for tokenized ClimbMix shards."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from training.data import MemoryMappedTokenDataset


def inspect_batches(
    path: str | Path,
    *,
    context_length: int,
    batch_sizes: tuple[int, ...] = (1, 4, 16),
    samples_per_size: int = 3,
    seed: int = 42,
) -> list[dict[str, object]]:
    """Sample batches and return simple statistics for sanity checks."""
    rows: list[dict[str, object]] = []
    for batch_size in batch_sizes:
        ds = MemoryMappedTokenDataset(path, context_length, seed=seed)
        for sample_idx in range(samples_per_size):
            x, y = ds.get_batch(batch_size)
            rows.append(
                {
                    "batch_size": batch_size,
                    "sample": sample_idx,
                    "x_shape": tuple(x.shape),
                    "y_shape": tuple(y.shape),
                    "x_min": int(x.min()),
                    "x_max": int(x.max()),
                    "y_min": int(y.min()),
                    "y_max": int(y.max()),
                    "unique_x": int(np.unique(x).size),
                    "shift_ok": bool(np.array_equal(x[:, 1:], y[:, :-1])),
                    "first_row_x": x[0, : min(16, x.shape[1])].tolist(),
                    "first_row_y": y[0, : min(16, y.shape[1])].tolist(),
                }
            )
    return rows
