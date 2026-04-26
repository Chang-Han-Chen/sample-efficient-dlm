"""Memory-mapped token dataset edge cases."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np

from training.data import MemoryMappedTokenDataset


def test_dataset_allows_exactly_one_window(tmp_path):
    path = tmp_path / "tokens.npy"
    np.save(path, np.arange(5, dtype=np.int32))

    dataset = MemoryMappedTokenDataset(path, context_length=4, seed=0)
    assert len(dataset) == 1
    inputs, targets = dataset.get_batch(1)
    np.testing.assert_array_equal(inputs, np.asarray([[0, 1, 2, 3]], dtype=np.int32))
    np.testing.assert_array_equal(targets, np.asarray([[1, 2, 3, 4]], dtype=np.int32))


def test_dataset_samples_across_short_shards(tmp_path):
    np.save(tmp_path / "a.npy", np.asarray([0, 1, 2], dtype=np.int32))
    np.save(tmp_path / "b.npy", np.asarray([3, 4, 5, 6], dtype=np.int32))

    dataset = MemoryMappedTokenDataset(tmp_path, context_length=4, seed=0)
    assert len(dataset) == 3
    inputs, targets = dataset._gather(np.asarray([0, 1, 2], dtype=np.int64))
    np.testing.assert_array_equal(
        inputs,
        np.asarray(
            [
                [0, 1, 2, 3],
                [1, 2, 3, 4],
                [2, 3, 4, 5],
            ],
            dtype=np.int32,
        ),
    )
    np.testing.assert_array_equal(
        targets,
        np.asarray(
            [
                [1, 2, 3, 4],
                [2, 3, 4, 5],
                [3, 4, 5, 6],
            ],
            dtype=np.int32,
        ),
    )


def test_sequential_iterator_keeps_last_full_non_overlapping_batch(tmp_path):
    path = tmp_path / "tokens.npy"
    np.save(path, np.arange(20, dtype=np.int32))

    dataset = MemoryMappedTokenDataset(path, context_length=4, seed=0)
    batches = list(dataset.iter_sequential(batch_size=2))
    assert len(batches) == 2
    np.testing.assert_array_equal(
        batches[1][0],
        np.asarray([[10, 11, 12, 13], [15, 16, 17, 18]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        batches[1][1],
        np.asarray([[11, 12, 13, 14], [16, 17, 18, 19]], dtype=np.int32),
    )

