"""Attention-mask helpers for AR and block diffusion backbones.

All masks are boolean with shape ``(1, 1, T_q, T_kv)`` because
``jax.nn.dot_product_attention`` broadcasts that layout over batch and heads.
``True`` means the key/value token is visible to the query token.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jax.Array


def validate_block_len(seq_len: int, block_len: int) -> None:
    if block_len <= 0:
        raise ValueError(f"block_len must be positive, got {block_len}")
    if seq_len % block_len != 0:
        raise ValueError(f"block_len={block_len} must divide seq_len={seq_len}")


def make_causal_mask(seq_len: int) -> Array:
    idx = jnp.arange(seq_len)
    return (idx[:, None] >= idx[None, :])[None, None, :, :]


def make_block_causal_mask(seq_len: int, block_len: int) -> Array:
    """One-stream BD3 sampling mask.

    Token i may attend to token j iff block(j) <= block(i), so each current
    block is bidirectional internally while previous blocks remain visible.
    """
    validate_block_len(seq_len, block_len)
    pos = jnp.arange(seq_len)
    mask = (pos[:, None] // block_len) >= (pos[None, :] // block_len)
    return mask[None, None, :, :]


def make_bd3_train_mask(seq_len: int, block_len: int) -> Array:
    """Dense dual-stream BD3 training mask for ``x_t`` concatenated with ``x_0``.

    The four quadrants are:
      - top-left: noisy stream block-diagonal attention
      - top-right: noisy block b sees clean blocks strictly before b
      - bottom-left: clean stream never sees noisy stream
      - bottom-right: clean stream block-causal attention
    """
    validate_block_len(seq_len, block_len)
    idx = jnp.arange(2 * seq_len)
    q_idx = idx[:, None]
    kv_idx = idx[None, :]

    q_is_x0 = q_idx >= seq_len
    kv_is_x0 = kv_idx >= seq_len

    q_block = jnp.where(q_is_x0, (q_idx - seq_len) // block_len, q_idx // block_len)
    kv_block = jnp.where(kv_is_x0, (kv_idx - seq_len) // block_len, kv_idx // block_len)

    block_diagonal = (q_block == kv_block) & (q_is_x0 == kv_is_x0)
    offset_block_causal = (q_block > kv_block) & kv_is_x0 & (~q_is_x0)
    block_causal = (q_block >= kv_block) & kv_is_x0 & q_is_x0
    return (block_diagonal | offset_block_causal | block_causal)[None, None, :, :]
