"""Simple deterministic token loaders for JAX AR training."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Iterator

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class MemoryMappedTokenDataset:
    """Memory-mapped `.npy` token dataset compatible with the PyTorch trainer.

    Random training batches are sampled by start index. Validation iterators use
    deterministic non-overlapping windows.
    """

    def __init__(
        self,
        path: str | Path,
        context_length: int,
        *,
        seed: int = 42,
    ):
        self.path = Path(path)
        self.context_length = int(context_length)
        self.rng = np.random.default_rng(seed)

        if self.path.is_dir():
            self.arrays = [
                np.load(fp, mmap_mode="r")
                for fp in sorted(self.path.glob("*.npy"))
                if "offsets_" not in fp.name
            ]
        else:
            self.arrays = [np.load(self.path, mmap_mode="r")]
        if not self.arrays:
            raise ValueError(f"No .npy token arrays found at {self.path}")

        self.lengths = np.asarray([arr.shape[0] for arr in self.arrays], dtype=np.int64)
        self.offsets = np.concatenate([np.asarray([0], dtype=np.int64), np.cumsum(self.lengths)])
        self.total_length = int(self.lengths.sum())
        self.num_start_positions = self.total_length - self.context_length
        if self.num_start_positions <= 0:
            raise ValueError("Dataset is shorter than context_length + 1")

        span = self.context_length + 1
        self.windows = [
            sliding_window_view(arr, span) if arr.shape[0] >= span else None
            for arr in self.arrays
        ]

    def __len__(self) -> int:
        return self.num_start_positions

    def _gather(self, starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        chunk_ids = np.searchsorted(self.offsets, starts, side="right") - 1
        local = starts - self.offsets[chunk_ids]
        span = self.context_length + 1
        out = np.empty((starts.shape[0], span), dtype=np.int32)

        for chunk_id in np.unique(chunk_ids):
            rows = np.nonzero(chunk_ids == chunk_id)[0]
            arr = self.arrays[chunk_id]
            win = self.windows[chunk_id]
            for row in rows:
                start = int(local[row])
                if start + span <= arr.shape[0]:
                    if win is None:
                        raise RuntimeError("internal data window missing for a long-enough shard")
                    out[row] = win[start].astype(np.int32, copy=False)
                else:
                    needed = span
                    filled = 0
                    c = int(chunk_id)
                    pos = start
                    while needed > 0 and c < len(self.arrays):
                        take = min(self.arrays[c].shape[0] - pos, needed)
                        out[row, filled:filled + take] = self.arrays[c][pos:pos + take]
                        filled += take
                        needed -= take
                        c += 1
                        pos = 0
                    if needed:
                        raise IndexError("Sample crosses past the end of the dataset")

        return out[:, :-1], out[:, 1:]

    def get_batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        starts = self.rng.integers(0, self.num_start_positions, size=batch_size, dtype=np.int64)
        return self._gather(starts)

    def iter_sequential(self, batch_size: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        span = self.context_length + 1
        starts = np.arange(
            0,
            self.total_length - batch_size * span + 1,
            batch_size * span,
        )
        for first in starts:
            batch_starts = first + np.arange(batch_size, dtype=np.int64) * span
            yield self._gather(batch_starts)
