"""Multi-head self-attention (with optional GQA, QK-norm, value residual, gating).

KV caching is intentionally left out for now — this version is for training.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Float, Int

from .core import Linear, ValueEmbedding, rms_normalize_last_dim, _MAX_SEQ_LEN
from .rope import RotaryPositionalEmbedding

Array = jax.Array

AttentionImpl = str | None


class MultiHeadSelfAttention(nnx.Module):
    """Causal MHSA with three independent Q, K, V projections.

    The PyTorch reference uses a single fused QKV projection; keeping them
    split makes weight-copying trivial and is the same math. Shapes inside:

        x                  : (B, T, d_model)
        q                  : (B, T, n_heads,    head_dim)
        k, v               : (B, T, n_kv_heads, head_dim)
        sdpa out           : (B, T, n_heads,    head_dim)   # flax/jax layout
        reshape -> (B, T, d_model) -> W_o

    JAX DPA handles GQA directly, so K/V are deliberately not repeated to
    ``n_heads`` before attention.

    Optional `gating` applies a sigmoid-gated multiplier on the attention
    output (after the per-head concat):
        * ``"elementwise"`` : a d_model -> d_model projection, scales every feature
        * ``"per-head"``    : a d_model -> n_heads projection, one scalar per head
        * ``"per-head-hd"`` : a head_dim -> n_heads projection, applied to the first
                              head_dim features of x (mostly historical)
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        d_model: int,
        n_heads: int,
        vocab_size: int | None = None,
        n_kv_heads: int | None = None,
        theta_base: float = 10_000.0,
        max_seq_len: int = _MAX_SEQ_LEN,
        qknorm: bool = False,
        value_residual: bool = False,
        value_embedding: bool = False,
        value_embedding_scale: float = 1.0,
        value_embedding_gate_channels: int = 32,
        value_embedding_init_std: float | None = None,
        gating: str | bool | None = False,
        is_causal: bool = True,
        attention_impl: AttentionImpl = None,
        fuse_qkv: bool = True,
        dtype: jnp.dtype = jnp.float32,
    ):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"GQA: n_heads={self.n_heads} must be divisible by n_kv_heads={self.n_kv_heads}"
            )
        self.head_dim = d_model // n_heads
        self.theta_base = theta_base
        self.dtype = dtype
        self.qknorm = qknorm
        self.value_residual = value_residual
        self.value_embedding = value_embedding
        self.value_embedding_scale = float(value_embedding_scale)
        self.value_embedding_gate_channels = min(int(value_embedding_gate_channels), d_model)
        self.gating = gating
        self.is_causal = bool(is_causal)
        if attention_impl not in (None, "xla", "cudnn"):
            raise ValueError("attention_impl must be one of None, 'xla', or 'cudnn'")
        self.attention_impl = attention_impl
        self.fuse_qkv = bool(fuse_qkv)

        self.q_out = self.n_heads * self.head_dim
        self.kv_out = self.n_kv_heads * self.head_dim

        # Keep Q/K/V as separate trainable matrices so Muon sees separate
        # matrix parameters. When `fuse_qkv=True`, only the forward matmul is
        # fused by concatenating these weights inside `__call__`.
        self.W_q = Linear(rngs, d_model, self.q_out, dtype=dtype)
        self.W_k = Linear(rngs, d_model, self.kv_out, dtype=dtype)
        self.W_v = Linear(rngs, d_model, self.kv_out, dtype=dtype)
        self.W_o = Linear(rngs, d_model, d_model, dtype=dtype)

        self.rope = RotaryPositionalEmbedding(
            rngs, theta_base, self.head_dim, max_seq_len, dtype=dtype
        )

        if self.qknorm:
            # Scalar gain applied to q after RMS-normalizing q and k.
            self.qk_scale = nnx.Param(jnp.ones((1,), dtype=jnp.float32))

        if self.value_residual:
            # Init: alpha1=1, alpha2=0, scale=1 -> mix is identity.
            self.alpha1 = nnx.Param(jnp.ones((1,),  dtype=jnp.float32))
            self.alpha2 = nnx.Param(jnp.zeros((1,), dtype=jnp.float32))
            self.scale  = nnx.Param(jnp.ones((1,),  dtype=jnp.float32))

        if self.value_embedding:
            if vocab_size is None:
                raise ValueError("vocab_size is required when value_embedding=True")
            self.value_embedding_table = ValueEmbedding(
                rngs,
                vocab_size,
                self.n_kv_heads,
                self.head_dim,
                d_model,
                dtype=dtype,
                init_std=value_embedding_init_std,
            )
            self.value_embedding_gate = Linear(
                rngs,
                self.value_embedding_gate_channels,
                self.n_kv_heads,
                dtype=dtype,
            )
            self.value_embedding_gate.weight.value = jnp.zeros_like(
                self.value_embedding_gate.weight.value
            )

        if self.gating:
            if self.gating == "elementwise":
                self.attn_gate = Linear(rngs, d_model, d_model, dtype=dtype)
            elif self.gating == "per-head":
                self.attn_gate = Linear(rngs, d_model, n_heads, dtype=dtype)
            elif self.gating == "per-head-hd":
                self.attn_gate = Linear(rngs, self.head_dim, n_heads, dtype=dtype)
            else:
                raise ValueError(f"gating={self.gating!r} is not a known mode")

    def __call__(
        self,
        x: Float[Array, "b seq d_model"],
        token_positions: Int[Array, "b seq"] | None = None,
        v1: Float[Array, "b seq kv_h head_dim"] | None = None,
        token_ids: Int[Array, "b seq"] | None = None,
        attention_mask: Array | None = None,
        is_causal: bool | None = None,
    ) -> tuple[Float[Array, "b seq d_model"], Float[Array, "b seq kv_h head_dim"]]:
        B, T, _ = x.shape

        if self.fuse_qkv:
            qkv_weight = jnp.concatenate(
                [
                    self.W_q.weight.value,
                    self.W_k.weight.value,
                    self.W_v.weight.value,
                ],
                axis=0,
            )
            qkv = x @ qkv_weight.T
            q, k, v_raw = jnp.split(
                qkv,
                [self.q_out, self.q_out + self.kv_out],
                axis=-1,
            )
        else:
            q = self.W_q(x)
            k = self.W_k(x)
            v_raw = self.W_v(x)
        q = q.reshape(B, T, self.n_heads, self.head_dim)
        k = k.reshape(B, T, self.n_kv_heads, self.head_dim)
        v_raw = v_raw.reshape(B, T, self.n_kv_heads, self.head_dim)

        # --- RoPE --------------------------------------------------------
        # rope expects seq at axis -2. We're in (B, T, H, Dh) (flax sdpa
        # layout), so transpose to (B, H, T, Dh) for the rotation, then back.
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)
        q = jnp.transpose(q, (0, 2, 1, 3))   # (B, T, n_heads,    Dh)
        k = jnp.transpose(k, (0, 2, 1, 3))   # (B, T, n_kv_heads, Dh)

        # --- Value residual ---------------------------------------------
        if self.value_residual:
            v1 = v_raw if v1 is None else v1
            denom = jnp.sqrt(self.alpha1.value ** 2 + self.alpha2.value ** 2 + 1e-8)
            v = (
                self.scale.value
                * (self.alpha1.value * v_raw + self.alpha2.value * v1)
                / denom
            ).astype(v_raw.dtype)
        else:
            v = v_raw

        # --- Token value embeddings -------------------------------------
        # PLAN Option D: cache the raw first-layer V stream, normalize/mix the
        # value-residual channel, then add token value embeddings afterward.
        if self.value_embedding and self.value_embedding_scale != 0.0:
            if token_ids is None:
                raise ValueError("token_ids must be provided when value_embedding=True")
            value_emb = self.value_embedding_table(token_ids).astype(v.dtype)
            gate_in = x[..., : self.value_embedding_gate_channels]
            gate = 2.0 * jax.nn.sigmoid(self.value_embedding_gate(gate_in))
            gate = gate[..., None].astype(v.dtype)
            v = v + self.value_embedding_scale * gate * value_emb

        # --- QK-norm -----------------------------------------------------
        if self.qknorm:
            q = rms_normalize_last_dim(q) * self.qk_scale.value.astype(q.dtype)
            k = rms_normalize_last_dim(k)

        # --- Scaled dot-product attention (causal) ----------------------
        # nnx.dot_product_attention does NOT accept is_causal; jax.nn does.
        use_causal = self.is_causal if is_causal is None else bool(is_causal)
        attn = jax.nn.dot_product_attention(
            q,
            k,
            v,
            mask=attention_mask,
            is_causal=use_causal,
            implementation=self.attention_impl,
        )   # (B, T, n_heads, head_dim)

        # --- Gating ------------------------------------------------------
        # Gates are applied *after* attention but *before* the output proj.
        # Formula matches the PT reference: gate = 2 * sigmoid(W_g(x)).
        if not self.gating:
            attn_cat = attn.reshape(B, T, self.d_model)      # (B, T, D)
        elif self.gating == "elementwise":
            attn_cat = attn.reshape(B, T, self.d_model)
            gate = 2.0 * jax.nn.sigmoid(self.attn_gate(x))   # (B, T, D)
            attn_cat = gate * attn_cat
        elif self.gating == "per-head":
            gate = 2.0 * jax.nn.sigmoid(self.attn_gate(x))        # (B, T, H)
            gate = gate[..., None]                                 # (B, T, H, 1)
            attn_cat = (gate * attn).reshape(B, T, self.d_model)
        elif self.gating == "per-head-hd":
            gate = 2.0 * jax.nn.sigmoid(
                self.attn_gate(x[..., : self.head_dim])            # (B, T, H)
            )
            gate = gate[..., None]                                 # (B, T, H, 1)
            attn_cat = (gate * attn).reshape(B, T, self.d_model)
        else:
            raise ValueError(f"gating={self.gating!r} is not a known mode")

        return self.W_o(attn_cat), v_raw
