"""JAX-only transformer behavior and smoke tests."""

import inspect
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from transformer.attention import MultiHeadSelfAttention
from transformer.masks import make_bd3_train_mask, make_block_causal_mask
from transformer.transformer import Transformer, has_value_embedding_layer


def assert_close(name, actual, expected, *, atol=1e-6, rtol=1e-6):
    try:
        np.testing.assert_allclose(
            np.asarray(actual),
            np.asarray(expected),
            atol=atol,
            rtol=rtol,
        )
    except AssertionError as exc:
        raise AssertionError(f"{name} mismatch") from exc


_UPDATE_TAKES_MODEL = "model" in inspect.signature(nnx.Optimizer.update).parameters


def _apply_update(optimizer, model, grads):
    if _UPDATE_TAKES_MODEL:
        optimizer.update(model, grads)
    else:
        optimizer.update(grads)


def test_jitted_training_step_decreases_loss():
    cfg = dict(n_layers=3, vocab_size=128, d_model=64, n_heads=4, d_ff=256)
    model = Transformer(nnx.Rngs(0), **cfg, num_grad_checkpoint_layers=2)
    optimizer = nnx.Optimizer(model, optax.sgd(0.1), wrt=nnx.Param)

    rng = np.random.default_rng(0)
    ids = jnp.asarray(
        rng.integers(0, cfg["vocab_size"], (4, 16), dtype=np.int32),
        dtype=jnp.int32,
    )

    def loss_fn(model, ids):
        logits = model(ids[:, :-1])
        targets = ids[:, 1:]
        logp = jax.nn.log_softmax(logits, axis=-1)
        token_loss = -jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
        return jnp.mean(token_loss)

    @nnx.jit
    def train_step(model, optimizer, ids):
        loss, grads = nnx.value_and_grad(loss_fn)(model, ids)
        _apply_update(optimizer, model, grads)
        return loss

    losses = [float(loss_fn(model, ids))]
    for _ in range(20):
        losses.append(float(train_step(model, optimizer, ids)))
    assert losses[-1] < losses[0] * 0.8, losses


def test_weight_tying_constructor_shares_state():
    model = Transformer(
        nnx.Rngs(0),
        n_layers=1,
        vocab_size=16,
        d_model=8,
        n_heads=2,
        d_ff=16,
        weight_tying=True,
    )
    assert model.lm_head.weight is model.embedding.weight
    flat_paths = [
        ".".join(str(part) for part in path)
        for path, _ in nnx.to_flat_state(nnx.state(model, nnx.Param))
    ]
    assert "embedding.weight" in flat_paths
    assert "lm_head.weight" not in flat_paths


def test_grad_checkpointing_matches_plain_forward():
    cfg = dict(n_layers=3, vocab_size=40, d_model=32, n_heads=4, d_ff=128)
    plain = Transformer(nnx.Rngs(0), **cfg, num_grad_checkpoint_layers=0)
    remat = Transformer(nnx.Rngs(0), **cfg, num_grad_checkpoint_layers=2)
    ids = jnp.asarray(np.random.default_rng(1).integers(0, cfg["vocab_size"], (2, 5)), dtype=jnp.int32)
    assert_close("remat forward", remat(ids), plain(ids), atol=1e-6, rtol=1e-6)


def test_qknorm_and_gating_modes_forward():
    x = jnp.asarray(np.random.default_rng(2).normal(size=(2, 6, 32)).astype(np.float32))
    qknorm = MultiHeadSelfAttention(nnx.Rngs(0), d_model=32, n_heads=4, qknorm=True)
    y, v = qknorm(x)
    assert y.shape == x.shape
    assert v.shape == (2, 6, 4, 8)
    assert bool(jnp.isfinite(y).all())

    for mode in ["elementwise", "per-head", "per-head-hd"]:
        attn = MultiHeadSelfAttention(nnx.Rngs(1), d_model=32, n_heads=4, gating=mode)
        y, v = attn(x)
        assert y.shape == x.shape
        assert v.shape == (2, 6, 4, 8)
        assert bool(jnp.isfinite(y).all())


def test_old_architecture_bfloat16_dtype_path():
    attn = MultiHeadSelfAttention(
        nnx.Rngs(0),
        d_model=32,
        n_heads=4,
        qknorm=True,
        value_residual=True,
        gating="per-head",
        dtype=jnp.bfloat16,
    )
    x = jnp.asarray(np.random.default_rng(3).normal(size=(2, 6, 32)).astype(np.float32)).astype(jnp.bfloat16)
    y, v = attn(x, is_causal=True)
    assert y.dtype == jnp.bfloat16
    assert v.dtype == jnp.bfloat16


def test_value_embedding_layers():
    assert [has_value_embedding_layer(i, 8) for i in range(8)] == [
        False, True, False, True, False, True, False, True,
    ]
    assert [has_value_embedding_layer(i, 7) for i in range(7)] == [
        True, False, True, False, True, False, True,
    ]
    assert [has_value_embedding_layer(i, 4, "all") for i in range(4)] == [
        True, True, True, True,
    ]
    assert [has_value_embedding_layer(i, 4, [0, 3]) for i in range(4)] == [
        True, False, False, True,
    ]


def _copy_value_residual_attention(base, target):
    target.W_q.weight[...] = base.W_q.weight[...]
    target.W_k.weight[...] = base.W_k.weight[...]
    target.W_v.weight[...] = base.W_v.weight[...]
    target.W_o.weight[...] = base.W_o.weight[...]
    target.alpha1[...] = base.alpha1[...]
    target.alpha2[...] = base.alpha2[...]
    target.scale[...] = base.scale[...]


def test_value_embedding_zero_scale_noop():
    d_model, n_heads, vocab_size = 32, 4, 64
    base = MultiHeadSelfAttention(nnx.Rngs(0), d_model, n_heads, value_residual=True)
    ve = MultiHeadSelfAttention(
        nnx.Rngs(1),
        d_model,
        n_heads,
        vocab_size=vocab_size,
        value_residual=True,
        value_embedding=True,
        value_embedding_scale=0.0,
    )
    _copy_value_residual_attention(base, ve)

    rng = np.random.default_rng(4)
    x = jnp.asarray(rng.normal(size=(2, 6, d_model)).astype(np.float32))
    ids = jnp.asarray(rng.integers(0, vocab_size, (2, 6)), dtype=jnp.int32)
    y_base, v_base = base(x)
    y_ve, v_ve = ve(x, token_ids=ids)
    assert_close("value_embedding scale=0 out", y_ve, y_base)
    assert_close("value_embedding scale=0 raw_v", v_ve, v_base)


def test_value_embedding_gain_zero_noop():
    d_model, n_heads, vocab_size = 32, 4, 64
    base = MultiHeadSelfAttention(nnx.Rngs(0), d_model, n_heads, value_residual=True)
    ve = MultiHeadSelfAttention(
        nnx.Rngs(1),
        d_model,
        n_heads,
        vocab_size=vocab_size,
        value_residual=True,
        value_embedding=True,
        value_embedding_scale=1.0,
    )
    _copy_value_residual_attention(base, ve)

    rng = np.random.default_rng(5)
    x = jnp.asarray(rng.normal(size=(2, 6, d_model)).astype(np.float32))
    ids = jnp.asarray(rng.integers(0, vocab_size, (2, 6)), dtype=jnp.int32)
    y_base, v_base = base(x)
    y_ve, v_ve = ve(x, token_ids=ids)
    assert_close("value_embedding gain=0 out", y_ve, y_base)
    assert_close("value_embedding gain=0 raw_v", v_ve, v_base)


def test_bd3_masks():
    seq_len, block_len = 8, 2
    train = np.asarray(make_bd3_train_mask(seq_len, block_len))[0, 0]
    sample = np.asarray(make_block_causal_mask(seq_len, block_len))[0, 0]
    assert train.shape == (2 * seq_len, 2 * seq_len)
    assert sample.shape == (seq_len, seq_len)
    assert not train[seq_len:, :seq_len].any(), "x0 stream must not see xt stream"
    for start in range(0, seq_len, block_len):
        end = start + block_len
        assert train[start:end, start:end].all()
        assert train[seq_len + start:seq_len + end, seq_len + start:seq_len + end].all()
    assert sample[block_len, 0]
    assert not sample[0, block_len]


def test_transformer_hidden_mask_and_value_embedding_api():
    cfg = dict(n_layers=4, vocab_size=64, d_model=32, n_heads=4, d_ff=128)
    model = Transformer(
        nnx.Rngs(0),
        **cfg,
        is_causal=False,
        value_embedding=True,
        value_embedding_scale=1.0,
    )
    ids = jnp.asarray(np.random.default_rng(6).integers(0, cfg["vocab_size"], (2, 6)), dtype=jnp.int32)
    full_mask = jnp.ones((1, 1, 6, 6), dtype=bool)
    hidden = model(ids, attention_mask=full_mask, return_hidden=True)
    logits_from_hidden = model.project_logits(hidden)
    logits_direct = model(ids, attention_mask=full_mask)
    assert hidden.shape == (2, 6, cfg["d_model"])
    assert logits_direct.shape == (2, 6, cfg["vocab_size"])
    assert_close("return_hidden/project_logits", logits_from_hidden, logits_direct)
    assert not hasattr(model.blocks[0].attn, "value_embedding_table")
    assert hasattr(model.blocks[1].attn, "value_embedding_table")
    assert hasattr(model.blocks[3].attn, "value_embedding_table")
