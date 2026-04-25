"""Core building blocks: Linear, Embedding, RMSNorm, SwiGLU, softmax.

Conventions chosen so weights can be copied from the PyTorch repo 1:1:

  * Linear.weight has shape (d_out, d_in) — same as nn.Linear.
    Forward is ``x @ W.T``.
  * Embedding.weight has shape (vocab, d_model) — same as nn.Embedding.
  * RMSNorm parameter is called ``gamma`` (vs PyTorch's ``gain``); we map at
    the weight-copy boundary.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Float, Int

Array = jax.Array

_MAX_SEQ_LEN = 8192


def softmax(x: Array, axis: int = -1, temperature: float = 1.0) -> Array:
    x = x - jnp.max(x, axis=axis, keepdims=True)
    if temperature != 0.0:
        x = x / temperature
    return jax.nn.softmax(x, axis=axis)


def _truncated_normal(
    rngs: nnx.Rngs,
    shape: tuple[int, ...],
    std: float,
    dtype: jnp.dtype = jnp.float32,
) -> Array:
    """truncated N(0, std**2), clipped to ±3*std — mirrors nn.init.trunc_normal_."""
    return jax.random.truncated_normal(
        rngs.params(), lower=-3.0, upper=3.0, shape=shape, dtype=dtype
    ) * std


class Linear(nnx.Module):
    """Bias-less linear layer. Weight stored as (d_out, d_in) for PT parity."""

    def __init__(
        self,
        rngs: nnx.Rngs,
        d_in: int,
        d_out: int,
        dtype: jnp.dtype = jnp.float32,
    ):
        sigma = 1.0 / jnp.sqrt(d_in)
        self.weight = nnx.Param(
            _truncated_normal(rngs, (d_out, d_in), float(sigma), dtype)
        )
        self.d_in = d_in
        self.d_out = d_out
        self.dtype = dtype

    def __call__(self, x: Array) -> Array:
        return x @ self.weight.value.T


class Embedding(nnx.Module):
    """Weight shape (vocab_size, d_model) — shareable with an lm_head Linear."""

    def __init__(
        self,
        rngs: nnx.Rngs,
        vocab_size: int,
        d_model: int,
        dtype: jnp.dtype = jnp.float32,
    ):
        sigma = 1.0 / jnp.sqrt(d_model)
        self.weight = nnx.Param(
            _truncated_normal(rngs, (vocab_size, d_model), float(sigma), dtype)
        )
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.dtype = dtype

    def __call__(self, token_ids: Int[Array, "b seq"]) -> Float[Array, "b seq d"]:
        return self.weight.value[token_ids]


class RMSNorm(nnx.Module):
    """RMSNorm with optional "layernorm scaling" (rsqrt(position)) scalar."""

    def __init__(
        self,
        rngs: nnx.Rngs,
        d_model: int,
        dtype: jnp.dtype = jnp.float32,
        eps: float = 1e-5,
        depth_position: int | None = None,
    ):
        del rngs  # RMSNorm has no randomly-init'd params, but keep signature uniform
        self.eps = eps
        self.gamma = nnx.Param(jnp.ones((d_model,), dtype=dtype))
        self.d_model = d_model
        self.dtype = dtype
        if depth_position is None:
            self.depth_scaling = 1.0
        else:
            self.depth_scaling = float(jax.lax.rsqrt(jnp.float32(depth_position)))

    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "... d"]:
        in_dtype = x.dtype
        x_f = x.astype(jnp.float32)
        rms = jnp.sqrt(jnp.mean(x_f * x_f, axis=-1, keepdims=True) + self.eps)
        # Broadcasts on the trailing d_model axis; no manual None slicing needed.
        y = x_f * (self.gamma.value / rms) * self.depth_scaling
        return y.astype(in_dtype)


class SwiGLU(nnx.Module):
    """SwiGLU FFN: down( silu(up(x)) * gate(x) ).

    PyTorch uses a single fused (d_model -> 2*d_ff) projection that's chunked;
    semantically identical, two projections is just easier to weight-copy.
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        d_model: int,
        d_ff: int,
        dtype: jnp.dtype = jnp.float32,
    ):
        # Match PyTorch's round-to-64 for hardware alignment.
        d_ff = int(d_ff // 64 * 64)
        self.d_model = d_model
        self.d_ff = d_ff
        self.dtype = dtype
        self.w_up = Linear(rngs, d_model, d_ff, dtype)
        self.w_gate = Linear(rngs, d_model, d_ff, dtype)
        self.w_down = Linear(rngs, d_ff, d_model, dtype)

    def __call__(self, x: Array) -> Array:
        return self.w_down(jax.nn.silu(self.w_up(x)) * self.w_gate(x))


def rms_normalize_last_dim(x: Array, eps: float = 1e-6) -> Array:
    """Per-token RMS normalize on the last dim (used by QK-norm attention)."""
    x_f = x.astype(jnp.float32)
    rms = jnp.sqrt(jnp.mean(x_f * x_f, axis=-1, keepdims=True) + eps)
    return (x_f / rms).astype(x.dtype)
