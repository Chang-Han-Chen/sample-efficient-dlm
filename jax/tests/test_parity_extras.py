"""Extra flag-combinations: weight tying, layernorm-scaling, qknorm, and
nnx.remat (checkpointing) - all should give identical outputs."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from test_parity import (
    parity_env, to_jnp, assert_close, copy_block, copy_mhsa,
)
import torch, numpy as np
import jax.numpy as jnp
from flax import nnx

# Need both module sets available (order matters: JAX first, then PT overrides
# sys.modules to the PT tree).  We keep references via module globals.
jax_mod = parity_env.load_jax()
from sample_efficient_gpt.transformer.transformer import (
    Transformer as JTransformer, has_value_embedding_layer,
)
from sample_efficient_gpt.transformer.attention import MultiHeadSelfAttention as JMHSA
from sample_efficient_gpt.transformer.masks import (
    make_bd3_train_mask as j_make_bd3_train_mask,
    make_block_causal_mask as j_make_block_causal_mask,
)

pt_mod = parity_env.load_pt()
from sample_efficient_gpt.transformer.transformer import (
    Transformer as PTransformer,
)
from sample_efficient_gpt.transformer.attention import MultiHeadSelfAttention as PMHSA

torch.manual_seed(0)


def test_weight_tying():
    """NB: the upstream PT `weight_tying=True` path is a silent no-op — it
    assigns to `self.lm_head.weight` but the actual param is at
    `self.lm_head.linear.weight`. To compare meaningfully we tie on both
    sides explicitly."""
    print("[Transformer weight_tying (manually tied)]")
    cfg = dict(n_layers=2, vocab_size=40, d_model=32, n_heads=4, d_ff=128)
    pt = PTransformer(**cfg, weight_tying=False)
    jx = JTransformer(nnx.Rngs(0), **cfg, weight_tying=False)
    jx.embedding.weight.value = to_jnp(pt.embedding.weight)
    for i in range(cfg["n_layers"]):
        copy_block(pt.blocks[i], jx.blocks[i], cfg["d_model"], cfg["n_heads"], cfg["n_heads"])
    jx.final_norm.gamma.value = to_jnp(pt.final_norm.gain)
    # Explicit tie on both sides.
    pt.lm_head.linear.weight = pt.embedding.weight
    jx.lm_head.weight = jx.embedding.weight
    ids = torch.randint(0, cfg["vocab_size"], (2, 5))
    y_pt, _ = pt(ids)
    y_jx = jx(jnp.asarray(ids.numpy(), dtype=jnp.int32))
    assert_close("weight_tying", y_jx, y_pt.detach(), atol=1e-3, rtol=1e-3)


def test_weight_tying_constructor_shares_state():
    model = JTransformer(
        nnx.Rngs(0),
        n_layers=1,
        vocab_size=16,
        d_model=8,
        n_heads=2,
        d_ff=16,
        weight_tying=True,
    )
    assert model.lm_head.weight is model.embedding.weight
    flat_paths = [".".join(str(part) for part in path) for path, _ in nnx.to_flat_state(nnx.state(model, nnx.Param))]
    assert "embedding.weight" in flat_paths
    assert "lm_head.weight" not in flat_paths


def test_ln_scaling():
    print("[Transformer layernorm_scaling]")
    cfg = dict(n_layers=3, vocab_size=40, d_model=32, n_heads=4, d_ff=128)
    pt = PTransformer(**cfg, layernorm_scaling=True)
    jx = JTransformer(nnx.Rngs(0), **cfg, layernorm_scaling=True)
    jx.embedding.weight.value = to_jnp(pt.embedding.weight)
    for i in range(cfg["n_layers"]):
        copy_block(pt.blocks[i], jx.blocks[i], cfg["d_model"], cfg["n_heads"], cfg["n_heads"])
    jx.final_norm.gamma.value = to_jnp(pt.final_norm.gain)
    jx.lm_head.weight.value = to_jnp(pt.lm_head.linear.weight)
    ids = torch.randint(0, cfg["vocab_size"], (2, 5))
    y_pt, _ = pt(ids); y_jx = jx(jnp.asarray(ids.numpy(), dtype=jnp.int32))
    assert_close("ln scaling", y_jx, y_pt.detach(), atol=1e-3, rtol=1e-3)


def test_mhsa_qknorm():
    print("[MHSA qknorm]")
    d_model, n_heads = 32, 4
    pt = PMHSA(d_model, n_heads, qknorm=True)
    jx = JMHSA(nnx.Rngs(0), d_model, n_heads, qknorm=True)
    copy_mhsa(pt, jx, d_model, n_heads, n_heads)
    # copy the scalar qk-gain
    with torch.no_grad():
        pt.sdpa_qknorm.gain.copy_(torch.tensor([1.3]))
    jx.qk_scale.value = to_jnp(pt.sdpa_qknorm.gain)
    x = torch.randn(2, 6, d_model)
    y_pt, _ = pt(x); y_jx, _ = jx(to_jnp(x))
    assert_close("qknorm", y_jx, y_pt.detach(), atol=5e-4, rtol=5e-4)


def test_gating():
    """Exercise all three gating modes supported by PT."""
    print("[MHSA gating]")
    d_model, n_heads = 32, 4
    for mode in ["elementwise", "per-head", "per-head-hd"]:
        pt = PMHSA(d_model, n_heads, gating=mode)
        jx = JMHSA(nnx.Rngs(0), d_model, n_heads, gating=mode)
        copy_mhsa(pt, jx, d_model, n_heads, n_heads)
        # copy the gating linear's weight; PT's Linear wrapper stores the
        # real weight at `.linear.weight`.
        jx.attn_gate.weight.value = to_jnp(pt.attn_gate.linear.weight)
        x = torch.randn(2, 6, d_model)
        y_pt, _ = pt(x); y_jx, _ = jx(to_jnp(x))
        assert_close(f"gating {mode}", y_jx, y_pt.detach(), atol=5e-4, rtol=5e-4)


def test_old_architecture_bfloat16_dtype_path():
    print("[MHSA old architecture bf16 dtype path]")
    attn = JMHSA(
        nnx.Rngs(0),
        d_model=32,
        n_heads=4,
        qknorm=True,
        value_residual=True,
        gating="per-head",
        dtype=jnp.bfloat16,
    )
    x = jnp.asarray(np.random.randn(2, 6, 32).astype(np.float32)).astype(jnp.bfloat16)
    y, v = attn(x, is_causal=True)
    assert y.dtype == jnp.bfloat16
    assert v.dtype == jnp.bfloat16


def test_grad_checkpointing():
    """Model with num_grad_checkpoint_layers should give identical forward."""
    print("[Transformer w/ nnx.remat]")
    cfg = dict(n_layers=3, vocab_size=40, d_model=32, n_heads=4, d_ff=128)
    pt = PTransformer(**cfg, num_grad_checkpoint_layers=2)
    jx = JTransformer(nnx.Rngs(0), **cfg, num_grad_checkpoint_layers=2)
    jx.embedding.weight.value = to_jnp(pt.embedding.weight)
    for i in range(cfg["n_layers"]):
        copy_block(pt.blocks[i], jx.blocks[i], cfg["d_model"], cfg["n_heads"], cfg["n_heads"])
    jx.final_norm.gamma.value = to_jnp(pt.final_norm.gain)
    jx.lm_head.weight.value = to_jnp(pt.lm_head.linear.weight)
    ids = torch.randint(0, cfg["vocab_size"], (2, 5))
    y_pt, _ = pt(ids); y_jx = jx(jnp.asarray(ids.numpy(), dtype=jnp.int32))
    assert_close("remat fwd", y_jx, y_pt.detach(), atol=1e-3, rtol=1e-3)


def test_value_embedding_layers():
    print("[value embedding layer placement]")
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


def test_value_embedding_zero_scale_noop():
    """When value_embedding_scale=0, VE params exist but the path is disabled."""
    print("[MHSA value_embedding zero-scale no-op]")
    d_model, n_heads, vocab_size = 32, 4, 64
    base = JMHSA(nnx.Rngs(0), d_model, n_heads, value_residual=True)
    ve = JMHSA(
        nnx.Rngs(1),
        d_model,
        n_heads,
        vocab_size=vocab_size,
        value_residual=True,
        value_embedding=True,
        value_embedding_scale=0.0,
    )
    ve.W_q.weight.value = base.W_q.weight.value
    ve.W_k.weight.value = base.W_k.weight.value
    ve.W_v.weight.value = base.W_v.weight.value
    ve.W_o.weight.value = base.W_o.weight.value
    ve.alpha1.value = base.alpha1.value
    ve.alpha2.value = base.alpha2.value
    ve.scale.value = base.scale.value

    x = jnp.asarray(np.random.randn(2, 6, d_model).astype(np.float32))
    ids = jnp.asarray(np.random.randint(0, vocab_size, (2, 6)), dtype=jnp.int32)
    y_base, v_base = base(x)
    y_ve, v_ve = ve(x, token_ids=ids)
    assert_close("value_embedding scale=0 out", y_ve, y_base, atol=1e-6, rtol=1e-6)
    assert_close("value_embedding scale=0 raw_v", v_ve, v_base, atol=1e-6, rtol=1e-6)


def test_bd3_masks():
    print("[JAX BD3 masks]")
    seq_len, block_len = 8, 2
    train = np.asarray(j_make_bd3_train_mask(seq_len, block_len))[0, 0]
    sample = np.asarray(j_make_block_causal_mask(seq_len, block_len))[0, 0]
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
    print("[Transformer hidden/mask/value_embedding API]")
    cfg = dict(n_layers=4, vocab_size=64, d_model=32, n_heads=4, d_ff=128)
    model = JTransformer(
        nnx.Rngs(0),
        **cfg,
        is_causal=False,
        value_embedding=True,
        value_embedding_scale=1.0,
    )
    ids = jnp.asarray(np.random.randint(0, cfg["vocab_size"], (2, 6)), dtype=jnp.int32)
    full_mask = jnp.ones((1, 1, 6, 6), dtype=bool)
    hidden = model(ids, attention_mask=full_mask, return_hidden=True)
    logits_from_hidden = model.project_logits(hidden)
    logits_direct = model(ids, attention_mask=full_mask)
    assert hidden.shape == (2, 6, cfg["d_model"])
    assert logits_direct.shape == (2, 6, cfg["vocab_size"])
    assert_close(
        "return_hidden/project_logits",
        logits_from_hidden,
        logits_direct,
        atol=1e-6,
        rtol=1e-6,
    )
    assert not hasattr(model.blocks[0].attn, "value_embedding_table")
    assert hasattr(model.blocks[1].attn, "value_embedding_table")
    assert hasattr(model.blocks[3].attn, "value_embedding_table")


if __name__ == "__main__":
    test_weight_tying()
    test_weight_tying_constructor_shares_state()
    test_ln_scaling()
    test_mhsa_qknorm()
    test_gating()
    test_old_architecture_bfloat16_dtype_path()
    test_grad_checkpointing()
    test_value_embedding_layers()
    test_value_embedding_zero_scale_noop()
    test_bd3_masks()
    test_transformer_hidden_mask_and_value_embedding_api()
    print("\nEXTRAS PASSED")
