"""End-to-end parity test: PyTorch repo <-> JAX port.

Strategy:
    Because the two frameworks have different RNGs, we can't seed-match.
    Instead, we build each module in both frameworks, copy weights
    PyTorch -> JAX tensor-for-tensor, and verify outputs agree to within a
    loose f32 tolerance.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import parity_env

# ---- Load JAX side -----------------------------------------------------
jax_mod = parity_env.load_jax()
from sample_efficient_gpt.transformer.core import (
    Linear as JLinear, Embedding as JEmbedding, RMSNorm as JRMSNorm,
    SwiGLU as JSwiGLU,
)
from sample_efficient_gpt.transformer.rope import RotaryPositionalEmbedding as JRoPE
from sample_efficient_gpt.transformer.attention import MultiHeadSelfAttention as JMHSA
from sample_efficient_gpt.transformer.transformer import (
    Block as JBlock, Transformer as JTransformer,
)
import jax, jax.numpy as jnp
from flax import nnx

# ---- Load PyTorch side -------------------------------------------------
pt_mod = parity_env.load_pt()
from sample_efficient_gpt.transformer.core import (
    Linear as PLinear, Embedding as PEmbedding, RMSNorm as PRMSNorm,
    SwiGLU as PSwiGLU,
)
from sample_efficient_gpt.transformer.rope import (
    RotatyPositionalEmbedding as PRoPE,
)
from sample_efficient_gpt.transformer.attention import (
    MultiHeadSelfAttention as PMHSA,
)
from sample_efficient_gpt.transformer.transformer import (
    Block as PBlock, Transformer as PTransformer,
)
import torch
torch.manual_seed(0)

# ---- helpers -----------------------------------------------------------

def to_jnp(t):
    return jnp.asarray(t.detach().numpy())

def assert_close(name, a, b, atol=1e-4, rtol=1e-4):
    a_np = np.asarray(a); b_np = np.asarray(b)
    if a_np.shape != b_np.shape:
        raise AssertionError(f"{name}: shape {a_np.shape} vs {b_np.shape}")
    diff = np.max(np.abs(a_np - b_np))
    rel = diff / (np.max(np.abs(b_np)) + 1e-12)
    ok = np.allclose(a_np, b_np, atol=atol, rtol=rtol)
    print(f"  {name:<30s} max|Δ|={diff:.2e}  rel={rel:.2e}  {'OK' if ok else 'FAIL'}")
    if not ok:
        raise AssertionError(f"{name} mismatch")

# ---- Linear ------------------------------------------------------------
def test_linear():
    print("[Linear]")
    pt = PLinear(8, 16)
    jx = JLinear(nnx.Rngs(0), 8, 16)
    # Copy the PT weight (shape (16, 8)) into JAX param (also (16, 8)).
    jx.weight.value = to_jnp(pt.linear.weight)
    x_pt = torch.randn(2, 3, 8)
    x_jx = to_jnp(x_pt)
    y_pt = pt(x_pt)
    y_jx = jx(x_jx)
    assert_close("Linear forward", y_jx, y_pt.detach())

# ---- Embedding ---------------------------------------------------------
def test_embedding():
    print("[Embedding]")
    pt = PEmbedding(50, 16)
    jx = JEmbedding(nnx.Rngs(0), 50, 16)
    jx.weight.value = to_jnp(pt.weight)
    ids = torch.randint(0, 50, (2, 4))
    y_pt = pt(ids)
    y_jx = jx(to_jnp(ids).astype(jnp.int32))
    assert_close("Embedding forward", y_jx, y_pt.detach())

# ---- RMSNorm -----------------------------------------------------------
def test_rmsnorm():
    print("[RMSNorm]")
    for pos in [None, 3]:
        pt = PRMSNorm(16, position=pos)
        jx = JRMSNorm(nnx.Rngs(0), 16, depth_position=pos)
        # random gain to exercise gain path
        with torch.no_grad():
            pt.gain.copy_(torch.randn(16))
        jx.gamma.value = to_jnp(pt.gain)
        x_pt = torch.randn(2, 3, 16)
        y_pt = pt(x_pt)
        y_jx = jx(to_jnp(x_pt))
        assert_close(f"RMSNorm pos={pos}", y_jx, y_pt.detach())

# ---- SwiGLU ------------------------------------------------------------
def test_swiglu():
    print("[SwiGLU]")
    d_model, d_ff = 32, 128
    pt = PSwiGLU(d_model, d_ff)
    jx = JSwiGLU(nnx.Rngs(0), d_model, d_ff)
    # PT fuses up+gate into a single Linear (d_model -> 2*d_ff) and chunks.
    # JAX splits into two. Copy halves correctly.
    # In PT, proj = up(x) -> (left, right); silu(left) * right.
    # left  <- first d_ff output rows of up.linear.weight
    # right <- second d_ff output rows
    W = pt.up.linear.weight.detach()                 # (2*d_ff, d_model)
    W_up_pt, W_gate_pt = W[:d_ff], W[d_ff:]
    jx.w_up.weight.value   = to_jnp(W_up_pt)
    jx.w_gate.weight.value = to_jnp(W_gate_pt)
    jx.w_down.weight.value = to_jnp(pt.down.linear.weight)
    x_pt = torch.randn(2, 3, d_model)
    y_pt = pt(x_pt)
    y_jx = jx(to_jnp(x_pt))
    assert_close("SwiGLU forward", y_jx, y_pt.detach(), atol=1e-4, rtol=1e-4)

# ---- RoPE (module-level) ----------------------------------------------
def test_rope():
    print("[RoPE]")
    d_k, T = 32, 16
    pt = PRoPE(10000.0, d_k, T)
    jx = JRoPE(nnx.Rngs(0), 10000.0, d_k, T)
    # PT input layout: (..., T, d_k). JAX's rope uses the same convention
    # internally (attention transposes before calling it).
    x_pt = torch.randn(2, 4, T, d_k)   # (B, H, T, d_k)
    y_pt = pt(x_pt)
    y_jx = jx(to_jnp(x_pt))
    assert_close("RoPE forward", y_jx, y_pt.detach(), atol=1e-5, rtol=1e-5)


def test_rope_token_positions_broadcast():
    """Exercise the B != H case with explicit token_positions.

    Regression guard for a real bug in the first-pass JAX port: it
    prepended leading size-1 axes to cos/sin, which aligned cos's batch
    axis with x's head axis and silently gave wrong rotations when B != H.

    PT's rope expects its input to already be 3D ((h*b), T, d_k) — it
    rearranges before calling rope — so PT isn't a clean oracle for the
    4D case. Instead, we check self-consistency: computing the rotation
    per-head (looping in Python) should match the batched call."""
    print("[RoPE token_positions, B!=H self-consistency]")
    d_k, T = 32, 8
    B, H = 3, 4
    max_T = 32
    jx = JRoPE(nnx.Rngs(0), 10000.0, d_k, max_T)

    x = jnp.asarray(np.random.randn(B, H, T, d_k).astype(np.float32))
    positions_np = np.stack([np.arange(0,  T),
                             np.arange(5,  T + 5),
                             np.arange(10, T + 10)], axis=0)          # (B, T)
    positions = jnp.asarray(positions_np)

    # Batched call (4D + (B, T) positions).
    y_batched = jx(x, positions)                                       # (B, H, T, d_k)

    # Reference: compute each head independently as a 3D tensor, so the
    # head axis never participates in the RoPE broadcast. The rope table
    # for a given row only depends on positions[b], not on H, so this is
    # the ground truth.
    refs = []
    for h in range(H):
        per_head = jx(x[:, h], positions)            # (B, T, d_k)
        refs.append(per_head)
    y_ref = jnp.stack(refs, axis=1)                  # (B, H, T, d_k)

    assert_close("RoPE w/ positions, B!=H", y_batched, y_ref, atol=1e-6, rtol=1e-6)

# ---- MHSA --------------------------------------------------------------
def copy_mhsa(pt, jx, d_model, n_heads, n_kv_heads):
    """Copy PT fused QKV + O into JAX split Q, K, V, O."""
    head_dim = d_model // n_heads
    q_out = n_heads * head_dim
    kv_out = n_kv_heads * head_dim
    W = pt.qkv.linear.weight.detach()         # ((q_out + 2*kv_out), d_model)
    Wq = W[:q_out]
    Wk = W[q_out:q_out + kv_out]
    Wv = W[q_out + kv_out:]
    jx.W_q.weight.value = to_jnp(Wq)
    jx.W_k.weight.value = to_jnp(Wk)
    jx.W_v.weight.value = to_jnp(Wv)
    jx.W_o.weight.value = to_jnp(pt.out.linear.weight)

def test_mhsa():
    print("[MultiHeadSelfAttention]")
    d_model, n_heads = 32, 4
    pt = PMHSA(d_model, n_heads)
    jx = JMHSA(nnx.Rngs(0), d_model, n_heads)
    copy_mhsa(pt, jx, d_model, n_heads, n_heads)
    x_pt = torch.randn(2, 6, d_model)
    # PT's attention internally rearranges (b seq (h d)) -> ((h b) seq d); to
    # reproduce the same rotary application, PT requires no special shape.
    y_pt, v_pt = pt(x_pt)
    y_jx, v_jx = jx(to_jnp(x_pt))
    assert_close("MHSA out", y_jx, y_pt.detach(), atol=3e-4, rtol=3e-4)
    # NB: the v tensors live in different layouts (PT: (h*B, T, Dh) vs
    # JAX: (B, T, H, Dh)), so we don't compare them directly here. The
    # meaningful check is the attention output.

def test_mhsa_gqa():
    print("[MultiHeadSelfAttention - GQA]")
    d_model, n_heads, n_kv_heads = 32, 4, 2
    pt = PMHSA(d_model, n_heads, n_kv_heads=n_kv_heads)
    jx = JMHSA(nnx.Rngs(0), d_model, n_heads, n_kv_heads=n_kv_heads)
    copy_mhsa(pt, jx, d_model, n_heads, n_kv_heads)
    x_pt = torch.randn(2, 6, d_model)
    y_pt, _ = pt(x_pt)
    y_jx, _ = jx(to_jnp(x_pt))
    assert_close("MHSA GQA out", y_jx, y_pt.detach(), atol=3e-4, rtol=3e-4)

# ---- Block -------------------------------------------------------------
def copy_block(pt, jx, d_model, n_heads, n_kv_heads):
    # ln1, ln2
    jx.ln1.gamma.value = to_jnp(pt.ln1.gain)
    jx.ln2.gamma.value = to_jnp(pt.ln2.gain)
    # attn
    copy_mhsa(pt.attn, jx.attn, d_model, n_heads, n_kv_heads)
    # ffn (SwiGLU)
    d_ff = pt.ffn.down.linear.weight.shape[1]        # down: (d_model, d_ff)
    W = pt.ffn.up.linear.weight.detach()
    jx.ffn.w_up.weight.value   = to_jnp(W[:d_ff])
    jx.ffn.w_gate.weight.value = to_jnp(W[d_ff:])
    jx.ffn.w_down.weight.value = to_jnp(pt.ffn.down.linear.weight)

def test_block():
    print("[Block]")
    d_model, n_heads, d_ff = 32, 4, 128
    pt = PBlock(d_model, n_heads, d_ff)
    jx = JBlock(nnx.Rngs(0), d_model, n_heads, d_ff)
    copy_block(pt, jx, d_model, n_heads, n_heads)
    x_pt = torch.randn(2, 6, d_model)
    y_pt, _kurt, _v = pt(x_pt)
    y_jx, _ = jx(to_jnp(x_pt))
    assert_close("Block forward", y_jx, y_pt.detach(), atol=5e-4, rtol=5e-4)

# ---- Transformer -------------------------------------------------------
def test_transformer():
    print("[Transformer end-to-end]")
    cfg = dict(n_layers=3, vocab_size=50, d_model=32, n_heads=4, d_ff=128)
    pt = PTransformer(**cfg)
    jx = JTransformer(nnx.Rngs(0), **cfg)
    # embedding
    jx.embedding.weight.value = to_jnp(pt.embedding.weight)
    # blocks
    for i in range(cfg["n_layers"]):
        copy_block(pt.blocks[i], jx.blocks[i],
                   cfg["d_model"], cfg["n_heads"], cfg["n_heads"])
    # final norm + lm_head
    jx.final_norm.gamma.value = to_jnp(pt.final_norm.gain)
    jx.lm_head.weight.value = to_jnp(pt.lm_head.linear.weight)
    # forward
    ids = torch.randint(0, cfg["vocab_size"], (2, 7))
    y_pt, _ = pt(ids)
    y_jx = jx(jnp.asarray(ids.numpy(), dtype=jnp.int32))
    assert_close("Transformer logits", y_jx, y_pt.detach(), atol=1e-3, rtol=1e-3)

# ---- Value residual ----------------------------------------------------
def test_value_residual():
    print("[MHSA with value_residual]")
    d_model, n_heads = 32, 4
    pt = PMHSA(d_model, n_heads, value_residual=True)
    jx = JMHSA(nnx.Rngs(0), d_model, n_heads, value_residual=True)
    copy_mhsa(pt, jx, d_model, n_heads, n_heads)
    # copy alpha1/alpha2/scale
    with torch.no_grad():
        pt.alpha1.copy_(torch.tensor([0.7]))
        pt.alpha2.copy_(torch.tensor([0.3]))
        pt.scale.copy_(torch.tensor([1.1]))
    jx.alpha1.value = to_jnp(pt.alpha1)
    jx.alpha2.value = to_jnp(pt.alpha2)
    jx.scale.value  = to_jnp(pt.scale)
    x_pt = torch.randn(2, 6, d_model)
    y_pt, _ = pt(x_pt)
    y_jx, _ = jx(to_jnp(x_pt))
    assert_close("MHSA value_residual", y_jx, y_pt.detach(), atol=3e-4, rtol=3e-4)

# ---- run them all ------------------------------------------------------
if __name__ == "__main__":
    test_linear()
    test_embedding()
    test_rmsnorm()
    test_swiglu()
    test_rope()
    test_rope_token_positions_broadcast()
    test_mhsa()
    test_mhsa_gqa()
    test_value_residual()
    test_block()
    test_transformer()
    print()
    print("ALL TESTS PASSED")
