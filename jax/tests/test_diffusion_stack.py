"""MDLM/BD3LM objective and mask tests for the JAX training stack."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from transformer.transformer import Transformer
from training.diffusion import (
    DiffusionConfig,
    eval_t_step_from_frac,
    make_bd3lm_batch,
    make_mdlm_batch,
    make_model_context,
    prepare_diffusion_training_batch,
    survival_prob_np,
)
from training.loss import cross_entropy_with_z_loss, supervised_lm_loss


def _small_diffusion_model(vocab_size=33, seq_len=8):
    return Transformer(
        nnx.Rngs(0),
        n_layers=2,
        vocab_size=vocab_size,
        d_model=32,
        n_heads=4,
        d_ff=64,
        max_seq_len=seq_len,
        is_causal=False,
        dtype=jnp.float32,
    )


def test_default_time_window_clips_survival_probability():
    cfg = DiffusionConfig(num_steps=10, t_min=0.45, t_max=0.95, mask_token_id=32)
    probs = survival_prob_np(np.asarray([1, 5, 10, 999]), cfg)
    np.testing.assert_allclose(probs, np.asarray([0.55, 0.50, 0.05, 0.05]), atol=1e-6)
    assert eval_t_step_from_frac(cfg, 0.6) == 6


def test_mdlm_batch_masks_only_selected_positions():
    cfg = DiffusionConfig(num_steps=10, t_min=0.0, t_max=1.0, mask_token_id=32)
    rng = np.random.default_rng(0)
    x0 = rng.integers(1, 32, size=(64, 32), dtype=np.int32)
    xt, target, mask = make_mdlm_batch(x0, cfg, rng, fixed_t_step=5)
    assert xt.shape == x0.shape
    assert target.shape == x0.shape
    assert mask.shape == x0.shape
    assert mask.dtype == bool
    assert np.array_equal(target, x0)
    assert np.all(xt[mask] == cfg.mask_token_id)
    assert np.array_equal(xt[~mask], x0[~mask])
    assert 0.35 < float(mask.mean()) < 0.65


def test_bd3lm_batch_uses_blockwise_timesteps_and_masks():
    cfg = DiffusionConfig(num_steps=10, t_min=0.0, t_max=1.0, mask_token_id=32, block_len=4)
    rng = np.random.default_rng(1)
    x0 = rng.integers(1, 32, size=(8, 16), dtype=np.int32)
    xt, target, mask = make_bd3lm_batch(x0, cfg, rng, fixed_t_step=10)
    assert np.array_equal(target, x0)
    assert mask.all()
    assert np.all(xt == cfg.mask_token_id)


def test_bd3_model_context_repeats_positions_and_builds_dual_mask():
    cfg = DiffusionConfig(mask_token_id=32, block_len=4)
    ctx = make_model_context("bd3lm", 8, cfg)
    assert ctx.is_causal is False
    assert ctx.output_length == 8
    assert tuple(ctx.attention_mask.shape) == (1, 1, 16, 16)
    np.testing.assert_array_equal(
        np.asarray(ctx.token_positions),
        np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32),
    )


def test_mdlm_supervised_loss_matches_manual_masked_ce():
    model = _small_diffusion_model()
    x0 = jnp.asarray([[5, 6, 7, 8, 9, 10, 11, 12]], dtype=jnp.int32)
    xt = x0.at[:, [2, 5]].set(32)
    mask = jnp.asarray([[False, False, True, False, False, True, False, False]])
    logits = model(xt, is_causal=False)
    expected_loss, expected_z, _ = cross_entropy_with_z_loss(
        logits,
        x0,
        valid_mask=mask,
    )
    total, metrics = supervised_lm_loss(
        model,
        xt,
        x0,
        mask,
        is_causal=False,
        z_loss_weight=0.0,
    )
    np.testing.assert_allclose(np.asarray(metrics["loss"]), np.asarray(expected_loss), atol=1e-6)
    np.testing.assert_allclose(np.asarray(metrics["z_loss"]), np.asarray(expected_z), atol=1e-6)
    np.testing.assert_allclose(np.asarray(total), np.asarray(expected_loss), atol=1e-6)


def test_chunked_supervised_loss_matches_full_for_diffusion_mask():
    model = _small_diffusion_model()
    x0 = jnp.asarray(np.arange(16, dtype=np.int32).reshape(2, 8) % 31 + 1)
    xt = x0.at[:, [1, 3, 6]].set(32)
    mask = xt == 32
    full_total, full_metrics = supervised_lm_loss(
        model,
        xt,
        x0,
        mask,
        is_causal=False,
        z_loss_weight=1e-4,
        loss_impl="full",
    )
    chunk_total, chunk_metrics = supervised_lm_loss(
        model,
        xt,
        x0,
        mask,
        is_causal=False,
        z_loss_weight=1e-4,
        loss_impl="chunked",
        logit_chunk_size=3,
    )
    np.testing.assert_allclose(np.asarray(chunk_total), np.asarray(full_total), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(
        np.asarray(chunk_metrics["loss"]),
        np.asarray(full_metrics["loss"]),
        atol=1e-5,
        rtol=1e-5,
    )


def test_unsupervised_diffusion_targets_do_not_affect_loss_or_gradients():
    model = _small_diffusion_model()
    xt = jnp.asarray([[32, 6, 32, 8, 9, 32, 11, 12]], dtype=jnp.int32)
    targets_a = jnp.asarray([[5, 6, 7, 8, 9, 10, 11, 12]], dtype=jnp.int32)
    # Change only unsupervised target positions. These must be ignored by CE.
    targets_b = jnp.asarray([[5, 20, 7, 21, 22, 10, 23, 24]], dtype=jnp.int32)
    mask = jnp.asarray([[True, False, True, False, False, True, False, False]])

    def loss_fn(m, targets):
        total, _ = supervised_lm_loss(
            m,
            xt,
            targets,
            mask,
            is_causal=False,
            z_loss_weight=0.0,
        )
        return total

    loss_a, grads_a = nnx.value_and_grad(loss_fn)(model, targets_a)
    loss_b, grads_b = nnx.value_and_grad(loss_fn)(model, targets_b)
    np.testing.assert_allclose(np.asarray(loss_a), np.asarray(loss_b), atol=1e-6)

    leaves_a = jax.tree_util.tree_leaves(nnx.as_pure(grads_a))
    leaves_b = jax.tree_util.tree_leaves(nnx.as_pure(grads_b))
    assert len(leaves_a) == len(leaves_b)
    for ga, gb in zip(leaves_a, leaves_b, strict=True):
        np.testing.assert_allclose(np.asarray(ga), np.asarray(gb), atol=1e-6, rtol=1e-6)


def test_bd3_single_block_preparation_skips_clean_stream():
    cfg = DiffusionConfig(mask_token_id=32, block_len=8)
    rng = np.random.default_rng(2)
    x0 = rng.integers(1, 32, size=(2, 8), dtype=np.int32)
    model_inputs, targets, supervise = prepare_diffusion_training_batch(
        "bd3lm",
        x0,
        cfg,
        rng,
        fixed_t_step=5,
    )
    assert model_inputs.shape == x0.shape
    assert targets.shape == x0.shape
    assert supervise.shape == x0.shape
    ctx = make_model_context("bd3lm", 8, cfg)
    assert ctx.attention_mask is None
    assert ctx.token_positions is None
    assert ctx.output_length is None


def test_bd3_dual_stream_mask_blocks_clean_target_leakage():
    model = _small_diffusion_model(seq_len=8)
    cfg = DiffusionConfig(mask_token_id=32, block_len=4)
    ctx = make_model_context("bd3lm", 8, cfg)
    xt = jnp.asarray([[1, 2, 3, 4, 32, 32, 7, 8]], dtype=jnp.int32)
    x0_a = jnp.asarray([[9, 10, 11, 12, 13, 14, 15, 16]], dtype=jnp.int32)
    x0_b = x0_a.at[:, :4].set(jnp.asarray([[20, 21, 22, 23]], dtype=jnp.int32))

    logits_a = model(
        jnp.concatenate([xt, x0_a], axis=1),
        token_positions=ctx.token_positions,
        attention_mask=ctx.attention_mask,
        is_causal=ctx.is_causal,
    )[:, :8]
    logits_b = model(
        jnp.concatenate([xt, x0_b], axis=1),
        token_positions=ctx.token_positions,
        attention_mask=ctx.attention_mask,
        is_causal=ctx.is_causal,
    )[:, :8]

    # Noisy block 0 cannot see any clean x0 tokens.
    np.testing.assert_allclose(
        np.asarray(logits_a[:, :4]),
        np.asarray(logits_b[:, :4]),
        atol=1e-5,
        rtol=1e-5,
    )
    # Noisy block 1 can see clean block 0, so changing clean block 0 should matter.
    diff = np.max(np.abs(np.asarray(logits_a[:, 4:8] - logits_b[:, 4:8])))
    assert diff > 1e-5


def test_bd3_dual_stream_mask_behavior_through_model_outputs():
    """Perturb allowed/disallowed regions and check actual logits."""
    model = _small_diffusion_model(seq_len=4)
    cfg = DiffusionConfig(mask_token_id=32, block_len=2)
    ctx = make_model_context("bd3lm", 4, cfg)
    xt = jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32)
    x0 = jnp.asarray([[10, 11, 12, 13]], dtype=jnp.int32)

    def logits(xt_, x0_):
        return model(
            jnp.concatenate([xt_, x0_], axis=1),
            token_positions=ctx.token_positions,
            attention_mask=ctx.attention_mask,
            is_causal=ctx.is_causal,
        )

    def max_diff(a, b, sl):
        return float(jnp.max(jnp.abs(a[:, sl] - b[:, sl])))

    base = logits(xt, x0)

    x0_clean0 = x0.at[:, :2].set(jnp.asarray([[20, 21]], dtype=jnp.int32))
    out = logits(xt, x0_clean0)
    assert max_diff(base, out, slice(0, 2)) == 0.0
    assert max_diff(base, out, slice(2, 4)) > 1e-5
    assert max_diff(base, out, slice(6, 8)) > 1e-5

    x0_clean1 = x0.at[:, 2:4].set(jnp.asarray([[22, 23]], dtype=jnp.int32))
    out = logits(xt, x0_clean1)
    assert max_diff(base, out, slice(0, 2)) == 0.0
    assert max_diff(base, out, slice(2, 4)) == 0.0
    assert max_diff(base, out, slice(4, 6)) == 0.0
    assert max_diff(base, out, slice(6, 8)) > 1e-5

    xt_noisy1 = xt.at[:, 2:4].set(jnp.asarray([[30, 31]], dtype=jnp.int32))
    out = logits(xt_noisy1, x0)
    assert max_diff(base, out, slice(0, 2)) == 0.0
    assert max_diff(base, out, slice(2, 4)) > 1e-5
    assert max_diff(base, out, slice(4, 8)) == 0.0

    xt_noisy0 = xt.at[:, :2].set(jnp.asarray([[24, 25]], dtype=jnp.int32))
    out = logits(xt_noisy0, x0)
    assert max_diff(base, out, slice(0, 2)) > 1e-5
    assert max_diff(base, out, slice(2, 4)) == 0.0
    assert max_diff(base, out, slice(4, 8)) == 0.0


def test_mdlm_bidirectional_attention_behavior_through_model_outputs():
    model = _small_diffusion_model(seq_len=4)
    x = jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32)
    x_alt = x.at[:, 3].set(20)
    logits = model(x, is_causal=False)
    logits_alt = model(x_alt, is_causal=False)
    diff_earlier = np.max(np.abs(np.asarray(logits[:, :1] - logits_alt[:, :1])))
    assert diff_earlier > 1e-5


if __name__ == "__main__":
    test_default_time_window_clips_survival_probability()
    test_mdlm_batch_masks_only_selected_positions()
    test_bd3lm_batch_uses_blockwise_timesteps_and_masks()
    test_bd3_model_context_repeats_positions_and_builds_dual_mask()
    test_mdlm_supervised_loss_matches_manual_masked_ce()
    test_chunked_supervised_loss_matches_full_for_diffusion_mask()
    test_unsupervised_diffusion_targets_do_not_affect_loss_or_gradients()
    test_bd3_single_block_preparation_skips_clean_stream()
    test_bd3_dual_stream_mask_blocks_clean_target_leakage()
    test_bd3_dual_stream_mask_behavior_through_model_outputs()
    test_mdlm_bidirectional_attention_behavior_through_model_outputs()
    print("DIFFUSION STACK TESTS PASSED")
