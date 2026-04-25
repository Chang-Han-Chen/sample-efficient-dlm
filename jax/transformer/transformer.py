"""Transformer blocks + full model in Flax NNX.

Notes on activation checkpointing (grad-checkpointing):
    In Flax NNX, ``jax.checkpoint`` applied directly to an ``nnx.Module``
    is awkward because the module carries state as a pytree of
    ``nnx.Variable`` leaves. The ergonomic primitive is ``nnx.remat``, a
    thin wrapper that splits the module into (graphdef, state), remats the
    functional call, and merges back. See the comment in ``Transformer``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Float, Int

from .core import SwiGLU, RMSNorm, Embedding, Linear, softmax
from .attention import MultiHeadSelfAttention

Array = jax.Array

# flax 0.11+ requires a data-marked container for a *list of submodules*
# assigned to a module attribute; plain lists are rejected with a
# "data on value of type list assigned to static attribute" error. Older
# flax (e.g. 0.10.x) does not have `nnx.List` but also does not enforce
# that check. Prefer `nnx.List` where available.
try:
    from flax.nnx import List as _ModuleList
except ImportError:          # pragma: no cover - fallback for flax < 0.11
    _ModuleList = list


class Block(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        d_model: int,
        n_heads: int,
        d_ff: int,
        attn_qknorm: bool = False,
        attn_val_residual: bool = False,
        attn_gating: str | bool | None = False,
        n_kv_heads: int | None = None,
        theta_base: float = 10_000.0,
        depth_position: int | None = None,
        max_seq_len: int = 8192,
        dtype: jnp.dtype = jnp.float32,
    ):
        self.ln1 = RMSNorm(rngs, d_model, dtype=dtype, depth_position=depth_position)
        self.attn = MultiHeadSelfAttention(
            rngs,
            d_model,
            n_heads,
            n_kv_heads=n_kv_heads,
            theta_base=theta_base,
            max_seq_len=max_seq_len,
            qknorm=attn_qknorm,
            value_residual=attn_val_residual,
            gating=attn_gating,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(rngs, d_model, dtype=dtype, depth_position=depth_position)
        self.ffn = SwiGLU(rngs, d_model, d_ff, dtype=dtype)

    def __call__(
        self,
        x: Float[Array, "b seq d"],
        token_positions: Int[Array, "b seq"] | None = None,
        v1: Float[Array, "b seq h head_dim"] | None = None,
    ) -> tuple[Float[Array, "b seq d"], Float[Array, "b seq h head_dim"]]:
        attn_out, v = self.attn(self.ln1(x), token_positions, v1)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, v


class Transformer(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        n_layers: int,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        attn_qknorm: bool = False,
        attn_val_residual: bool = False,
        attn_gating: str | bool | None = False,
        layernorm_scaling: bool = False,
        theta_base: float = 10_000.0,
        n_kv_heads: int | None = None,
        max_seq_len: int = 8192,
        dtype: jnp.dtype = jnp.float32,
        weight_tying: bool = False,
        num_grad_checkpoint_layers: int = 0,
    ):
        self.n_layers = n_layers
        self.num_grad_checkpoint_layers = num_grad_checkpoint_layers

        self.embedding = Embedding(rngs, vocab_size, d_model, dtype=dtype)
        self.blocks = _ModuleList(
            [
                Block(
                    rngs,
                    d_model,
                    n_heads,
                    d_ff,
                    attn_qknorm=attn_qknorm,
                    attn_val_residual=attn_val_residual,
                    attn_gating=attn_gating,
                    n_kv_heads=n_kv_heads,
                    theta_base=theta_base,
                    depth_position=pos if layernorm_scaling else None,
                    max_seq_len=max_seq_len,
                    dtype=dtype,
                )
                for pos in range(1, n_layers + 1)
            ]
        )
        self.final_norm = RMSNorm(rngs, d_model, dtype=dtype)
        self.lm_head = Linear(rngs, d_model, vocab_size, dtype=dtype)

        if weight_tying:
            # Both weights are stored (vocab, d_model) so this is shape-safe.
            self.lm_head.weight = self.embedding.weight

    def __call__(
        self,
        token_ids: Int[Array, "b seq"],
        token_positions: Int[Array, "b seq"] | None = None,
    ) -> Float[Array, "b seq vocab"]:
        x = self.embedding(token_ids)

        v1 = None
        for i, block in enumerate(self.blocks):
            if i < self.num_grad_checkpoint_layers:
                # nnx.remat wraps the stateful callable so the forward is
                # recomputed during the backward pass to save activations.
                rematted = nnx.remat(block)
                x, v = rematted(x, token_positions, v1)
            else:
                x, v = block(x, token_positions, v1)
            if v1 is None:
                v1 = v

        x = self.final_norm(x)
        return self.lm_head(x)
