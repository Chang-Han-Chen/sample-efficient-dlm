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
    Transformer as JTransformer,
)
from sample_efficient_gpt.transformer.attention import MultiHeadSelfAttention as JMHSA

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


if __name__ == "__main__":
    test_weight_tying()
    test_ln_scaling()
    test_mhsa_qknorm()
    test_gating()
    test_grad_checkpointing()
    print("\nEXTRAS PASSED")
