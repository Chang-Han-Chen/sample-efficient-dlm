"""Rotary Positional Embeddings in Flax NNX.

Internally we operate on tensors whose **seq axis is -2** (``(..., T, d_k)``)
— same convention as the PyTorch reference. The attention code will transpose
``(B, T, H, Dh) -> (B, H, T, Dh)`` before calling us and back after.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Float, Int

Array = jax.Array


class RoPEBuffer(nnx.Variable):
    """Non-trainable state, analogous to PyTorch's register_buffer."""
    pass


def get_cos_sin(
    max_seq_len: int,
    theta_base: float,
    d_k: int,
    dtype: jnp.dtype = jnp.float32,
) -> tuple[Float[Array, "T half"], Float[Array, "T half"]]:
    """Precompute cos/sin of shape (max_seq_len, d_k // 2)."""
    assert d_k % 2 == 0, "RoPE requires even d_k."
    half = d_k // 2
    # Keep the frequency computation in float32 regardless of `dtype` so bf16
    # tables still have accurate rotations.
    j = jnp.arange(half, dtype=jnp.float32)
    inv_freqs = jnp.asarray(theta_base, dtype=jnp.float32) ** (-2.0 * j / d_k)
    positions = jnp.arange(max_seq_len, dtype=jnp.float32)
    thetas = jnp.outer(positions, inv_freqs)
    return jnp.cos(thetas).astype(dtype), jnp.sin(thetas).astype(dtype)


class RotaryPositionalEmbedding(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        theta: float,
        d_k: int,
        max_seq_len: int,
        *,
        dtype: jnp.dtype = jnp.float32,
    ):
        del rngs  # no randomly-init'd params
        cos, sin = get_cos_sin(
            max_seq_len=max_seq_len,
            theta_base=theta,
            d_k=d_k,
            dtype=dtype,
        )
        self.cos = RoPEBuffer(cos)
        self.sin = RoPEBuffer(sin)
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

    def __call__(
        self,
        x: Float[Array, "... T d_k"],
        token_positions: Int[Array, "... T"] | None = None,
    ) -> Float[Array, "... T d_k"]:
        """Split-half RoPE, expecting seq axis at position -2.

        Typical attention input: ``(B, H, T, Dh)``. We promote cos/sin to
        x.ndim by inserting size-1 axes **just before the seq axis**, so
        batch stays aligned with batch and heads stays aligned with heads.

        Concretely:
            * no ``token_positions``: cos starts as ``(T, half)`` — becomes
              ``(1, 1, T, half)`` to broadcast against ``(B, H, T, half)``.
            * with ``token_positions`` of shape ``(B, T)``: cos starts as
              ``(B, T, half)`` — becomes ``(B, 1, T, half)``.

        Inserting **leading** axes (what the earlier version did) was wrong
        with token_positions because it aligned the batch dim of cos with
        the head dim of x, silently producing the wrong rotation when
        ``B != H``.
        """
        in_dtype = x.dtype
        seq_len = x.shape[-2]

        if token_positions is None:
            cos = self.cos.value[:seq_len]   # (T, half)
            sin = self.sin.value[:seq_len]
        else:
            cos = self.cos.value[token_positions]  # (..., T, half)
            sin = self.sin.value[token_positions]

        # Insert size-1 axes immediately *before* the seq axis (position -2
        # in cos/sin, which has a trailing `half` axis). Each insertion adds
        # one "head-like" broadcast axis.
        while cos.ndim < x.ndim:
            cos = cos[..., None, :, :]
            sin = sin[..., None, :, :]

        # Do the rotation in f32 for stability, then cast back.
        x_f = x.astype(jnp.float32)
        cos_f = cos.astype(jnp.float32)
        sin_f = sin.astype(jnp.float32)

        half = x_f.shape[-1] // 2
        x1 = x_f[..., :half]
        x2 = x_f[..., half:]

        row1 = x1 * cos_f - x2 * sin_f
        row2 = x1 * sin_f + x2 * cos_f
        return jnp.concatenate([row1, row2], axis=-1).astype(in_dtype)
