"""Data-parallel parity tests on simulated multi-CPU devices.

These tests exercise the multi-GPU code path on a single host by forcing
JAX to expose multiple CPU devices via XLA_FLAGS. Run with:

    XLA_FLAGS=--xla_force_host_platform_device_count=2 \
        python jax/tests/test_data_parallel_parity.py

The tests are split into AR and MDLM:

* AR: one-step parity between a single-device run on the full global batch
  and a 2-device DP run on the sharded batch. Loss, grads, and post-update
  params must agree.
* MDLM: same parity check, but with intentionally uneven supervise masks
  across the two shards. This is the load-bearing test for the sum/count
  reduction — under the previous mean-of-means code, this test would fail
  on uneven masks.
* `supervised_tokens` must equal the global supervised-token count after
  reduction (psum, not pmean).
* After one DP step, the optimizer state must be identical across devices.
"""

import os
import sys

# Force two simulated CPU devices BEFORE jax is imported.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import copy

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from transformer.transformer import Transformer
from training.diffusion import (
    DiffusionConfig,
    make_model_context,
    prepare_diffusion_training_batch,
)
from training.loss import supervised_lm_loss_sums
from training.optimizer import NormuonAdamWConfig, create_normuon_adamw
from training.step import (
    train_step,
    train_step_data_parallel,
    train_step_supervised,
    train_step_supervised_data_parallel,
)

NUM_DEVICES = 2


def _require_two_devices():
    if jax.local_device_count() < NUM_DEVICES:
        raise SystemExit(
            f"this test needs {NUM_DEVICES} JAX devices; saw "
            f"{jax.local_device_count()}. Run with "
            "XLA_FLAGS=--xla_force_host_platform_device_count=2"
        )


def _small_model(seed: int, vocab_size: int = 32, seq_len: int = 16, is_causal: bool = True):
    return Transformer(
        nnx.Rngs(seed),
        n_layers=2,
        vocab_size=vocab_size,
        d_model=32,
        n_heads=4,
        d_ff=64,
        max_seq_len=seq_len,
        is_causal=is_causal,
        dtype=jnp.float32,
    )


def _small_optimizer(model):
    cfg = NormuonAdamWConfig(
        table_adam_lr=1e-3,
        scalar_adam_lr=1e-3,
        muon_lr=1e-3,
        adam_weight_decay=0.0,
        muon_weight_decay=0.0,
        scheduler="constant",
        warmup_steps=0,
        momentum_warmup_steps=0,
    )
    return nnx.Optimizer(model, create_normuon_adamw(model, cfg), wrt=nnx.Param), cfg


def _flat_param_arrays(model) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for path, variable in nnx.to_flat_state(nnx.state(model, nnx.Param)):
        key = "/".join(str(p) for p in path)
        value = variable[...]
        if hasattr(value, "shape") and len(getattr(value, "shape", ())) > 0:
            # If pmap left a device axis on the array, every device's copy
            # should be identical; just take the first.
            value = np.asarray(value)
        else:
            value = np.asarray(value)
        out[key] = value
    return out


def _all_reduce_devices_equal(model) -> bool:
    """For pmap'd state, every device replica should be bit-exact."""
    for _, variable in nnx.to_flat_state(nnx.state(model, nnx.Param)):
        arr = np.asarray(variable[...])
        if arr.ndim == 0:
            continue
        # If a leading device axis is present, all slices must be equal.
        if arr.shape[0] == NUM_DEVICES:
            head = arr[0]
            for d in range(1, NUM_DEVICES):
                if not np.array_equal(arr[d], head):
                    return False
    return True


def _make_ar_batch(rng, batch_size: int, seq_len: int, vocab_size: int):
    tokens = rng.integers(0, vocab_size, size=(batch_size, seq_len + 1), dtype=np.int32)
    return tokens[:, :-1].copy(), tokens[:, 1:].copy()


def test_ar_single_vs_dp_parity():
    _require_two_devices()
    rng = np.random.default_rng(0)
    seq_len = 16
    vocab_size = 32
    global_batch = 8

    inputs_np, targets_np = _make_ar_batch(rng, global_batch, seq_len, vocab_size)
    inputs = jnp.asarray(inputs_np)
    targets = jnp.asarray(targets_np)
    inputs_sharded = inputs_np.reshape(NUM_DEVICES, global_batch // NUM_DEVICES, seq_len)
    targets_sharded = targets_np.reshape(NUM_DEVICES, global_batch // NUM_DEVICES, seq_len)
    inputs_dp = jnp.asarray(inputs_sharded)
    targets_dp = jnp.asarray(targets_sharded)

    model_single = _small_model(seed=42, vocab_size=vocab_size, seq_len=seq_len)
    opt_single, _ = _small_optimizer(model_single)

    model_dp = _small_model(seed=42, vocab_size=vocab_size, seq_len=seq_len)
    opt_dp, _ = _small_optimizer(model_dp)

    metrics_single = train_step(model_single, opt_single, inputs, targets, 0.0, 1.0, "full", 1024)
    metrics_dp_raw = train_step_data_parallel(
        model_dp, opt_dp, inputs_dp, targets_dp, 0.0, 1.0, "full", 1024
    )
    metrics_dp = jax.tree_util.tree_map(
        lambda v: jnp.mean(v) if getattr(v, "shape", ()) != () else v, metrics_dp_raw
    )

    np.testing.assert_allclose(
        float(metrics_single["loss"]), float(metrics_dp["loss"]), atol=1e-5, rtol=1e-5
    )
    np.testing.assert_allclose(
        float(metrics_single["total_loss"]),
        float(metrics_dp["total_loss"]),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        float(metrics_single["grad_norm"]), float(metrics_dp["grad_norm"]), atol=1e-5, rtol=1e-5
    )

    # Post-update params should agree to single-precision tolerance.
    single_params = _flat_param_arrays(model_single)
    dp_params = _flat_param_arrays(model_dp)
    assert single_params.keys() == dp_params.keys()
    for key in single_params:
        single_arr = single_params[key]
        dp_arr = dp_params[key]
        # If DP state still carries a device axis, take the first replica;
        # the optimizer.update is identical across devices.
        if dp_arr.ndim > single_arr.ndim and dp_arr.shape[0] == NUM_DEVICES:
            dp_arr = dp_arr[0]
        # Loss/grads match at ~1e-5, but Muon's Newton-Schulz iteration
        # amplifies floating-point summation-order differences between a
        # single-device sum and a pmean of per-device sums. atol=1e-3 is a
        # comfortable bound for one optimizer step at this tiny scale.
        np.testing.assert_allclose(
            single_arr,
            dp_arr,
            atol=1e-3,
            rtol=1e-3,
            err_msg=f"AR DP-vs-single param mismatch at {key}",
        )

    assert _all_reduce_devices_equal(model_dp), "AR DP optimizer state diverged across devices"


def _make_mdlm_uneven_masked_batch(rng, vocab_size, seq_len, *, per_device_batch):
    """Construct an MDLM-style supervised batch with deliberately uneven mask
    counts across the two shards: shard 0 ~25% supervised, shard 1 ~75%.
    """
    global_batch = per_device_batch * NUM_DEVICES
    x0 = rng.integers(1, vocab_size - 1, size=(global_batch, seq_len), dtype=np.int32)
    targets = x0.copy()

    mask_token_id = vocab_size - 1
    inputs = x0.copy()
    supervise = np.zeros((global_batch, seq_len), dtype=bool)

    half = per_device_batch
    # Shard 0: ~25% positions supervised. Shard 1: ~75%.
    for b in range(half):
        n_mask = max(1, int(0.25 * seq_len))
        idx = rng.choice(seq_len, size=n_mask, replace=False)
        supervise[b, idx] = True
    for b in range(half, global_batch):
        n_mask = max(1, int(0.75 * seq_len))
        idx = rng.choice(seq_len, size=n_mask, replace=False)
        supervise[b, idx] = True
    inputs[supervise] = mask_token_id
    return inputs, targets, supervise


def test_mdlm_uneven_mask_dp_parity():
    """Load-bearing: with uneven supervised counts across shards, the
    weighted-mean reduction must reproduce the single-device global mean."""
    _require_two_devices()
    rng = np.random.default_rng(7)
    seq_len = 16
    vocab_size = 33  # last id = mask token
    per_device_batch = 4
    global_batch = per_device_batch * NUM_DEVICES

    inputs_np, targets_np, supervise_np = _make_mdlm_uneven_masked_batch(
        rng, vocab_size, seq_len, per_device_batch=per_device_batch
    )
    # Shard 0 has 4 sequences * 4 supervised tokens = 16 positions.
    # Shard 1 has 4 sequences * 12 supervised tokens = 48 positions.
    # These are intentionally uneven; under mean-of-means, shard 0's
    # per-token grad would get 3x the weight of shard 1's.
    shard0_count = int(supervise_np[:per_device_batch].sum())
    shard1_count = int(supervise_np[per_device_batch:].sum())
    assert shard0_count != shard1_count, "test requires uneven mask counts"
    global_count = shard0_count + shard1_count

    inputs = jnp.asarray(inputs_np, dtype=jnp.int32)
    targets = jnp.asarray(targets_np, dtype=jnp.int32)
    supervise = jnp.asarray(supervise_np, dtype=bool)

    inputs_dp = jnp.asarray(
        inputs_np.reshape(NUM_DEVICES, per_device_batch, seq_len), dtype=jnp.int32
    )
    targets_dp = jnp.asarray(
        targets_np.reshape(NUM_DEVICES, per_device_batch, seq_len), dtype=jnp.int32
    )
    supervise_dp = jnp.asarray(
        supervise_np.reshape(NUM_DEVICES, per_device_batch, seq_len), dtype=bool
    )

    model_single = _small_model(seed=11, vocab_size=vocab_size, seq_len=seq_len, is_causal=False)
    opt_single, _ = _small_optimizer(model_single)
    model_dp = _small_model(seed=11, vocab_size=vocab_size, seq_len=seq_len, is_causal=False)
    opt_dp, _ = _small_optimizer(model_dp)

    metrics_single = train_step_supervised(
        model_single, opt_single, inputs, targets, supervise,
        None, None, 0.0, 1.0, False, None, "full", 1024, None,
    )
    metrics_dp_raw = train_step_supervised_data_parallel(
        model_dp, opt_dp, inputs_dp, targets_dp, supervise_dp,
        None, None, 0.0, 1.0, False, None, "full", 1024, None,
    )
    metrics_dp = jax.tree_util.tree_map(
        lambda v: jnp.mean(v) if getattr(v, "shape", ()) != () else v, metrics_dp_raw
    )

    np.testing.assert_allclose(
        float(metrics_single["loss"]), float(metrics_dp["loss"]), atol=1e-5, rtol=1e-5
    )
    np.testing.assert_allclose(
        float(metrics_single["total_loss"]),
        float(metrics_dp["total_loss"]),
        atol=1e-5,
        rtol=1e-5,
    )

    # supervised_tokens must equal the global supervised-token count, not a
    # per-device average. Under the old pmean'd metrics, this would fail.
    np.testing.assert_allclose(
        float(metrics_single["supervised_tokens"]), float(global_count), atol=0.0
    )
    np.testing.assert_allclose(
        float(metrics_dp["supervised_tokens"]), float(global_count), atol=0.0
    )

    # Post-update params should match.
    single_params = _flat_param_arrays(model_single)
    dp_params = _flat_param_arrays(model_dp)
    for key in single_params:
        single_arr = single_params[key]
        dp_arr = dp_params[key]
        if dp_arr.ndim > single_arr.ndim and dp_arr.shape[0] == NUM_DEVICES:
            dp_arr = dp_arr[0]
        np.testing.assert_allclose(
            single_arr,
            dp_arr,
            atol=1e-3,
            rtol=1e-3,
            err_msg=f"MDLM uneven-mask DP param mismatch at {key}",
        )

    assert _all_reduce_devices_equal(model_dp), "MDLM DP optimizer state diverged across devices"


def test_supervised_loss_sums_basic_invariants():
    """supervised_lm_loss_sums must return loss_sum / valid_count consistent
    with the mean-form loss for a non-trivial mask."""
    rng = np.random.default_rng(2)
    seq_len = 8
    vocab_size = 17
    batch = 3
    inputs = jnp.asarray(rng.integers(0, vocab_size, size=(batch, seq_len)).astype(np.int32))
    targets = jnp.asarray(rng.integers(0, vocab_size, size=(batch, seq_len)).astype(np.int32))
    mask = np.zeros((batch, seq_len), dtype=bool)
    mask[0, 0:2] = True
    mask[1, 2:6] = True
    mask[2, 7:8] = True
    expected_count = int(mask.sum())
    supervise = jnp.asarray(mask, dtype=bool)

    model = _small_model(seed=3, vocab_size=vocab_size, seq_len=seq_len, is_causal=False)
    total_sum, metrics = supervised_lm_loss_sums(
        model, inputs, targets, supervise, z_loss_weight=1e-4
    )
    valid_count = float(metrics["valid_count"])
    assert valid_count == float(expected_count)
    # total_sum / count should be close to (loss_mean + z_w * z_mean) computed
    # via the mean-form helper; reconstruct from sums:
    loss_mean = float(metrics["loss_sum"]) / valid_count
    z_mean = float(metrics["z_loss_sum"]) / valid_count
    total_mean_recovered = loss_mean + 1e-4 * z_mean
    np.testing.assert_allclose(
        float(total_sum) / valid_count, total_mean_recovered, atol=1e-6, rtol=1e-6
    )


def main():
    test_supervised_loss_sums_basic_invariants()
    print("ok: supervised_loss_sums invariants")
    test_ar_single_vs_dp_parity()
    print("ok: AR single vs DP parity")
    test_mdlm_uneven_mask_dp_parity()
    print("ok: MDLM uneven-mask DP parity")
    print("all data-parallel parity tests passed")


if __name__ == "__main__":
    main()
