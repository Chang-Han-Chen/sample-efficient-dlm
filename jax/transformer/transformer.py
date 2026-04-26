"""Transformer blocks + full model in Flax NNX.

Notes on activation checkpointing (grad-checkpointing):
    In Flax NNX, ``jax.checkpoint`` applied directly to an ``nnx.Module``
    is awkward because the module carries state as a pytree of
    ``nnx.Variable`` leaves. The ergonomic primitive is ``nnx.remat``, a
    thin wrapper that splits the module into (graphdef, state), remats the
    functional call, and merges back. See the comment in ``Transformer``.
"""

from __future__ import annotations

from collections.abc import Sequence

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


def has_value_embedding_layer(
    layer_idx: int,
    n_layers: int,
    placement: str | Sequence[int] | None = "alternating",
) -> bool:
    """Return whether a 0-indexed layer should receive token value embeddings."""
    if placement is None:
        return False
    if isinstance(placement, str):
        placement = placement.lower()
        if placement in ("none", "off", "false"):
            return False
        if placement == "all":
            return True
        if placement == "final":
            return layer_idx == n_layers - 1
        if placement == "alternating":
            return layer_idx % 2 == (n_layers - 1) % 2
        raise ValueError(
            "value_embedding_layers must be one of 'alternating', 'all', "
            "'final', 'none', or a sequence of layer indices"
        )
    return layer_idx in set(int(i) for i in placement)


class Block(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        d_model: int,
        n_heads: int,
        d_ff: int,
        vocab_size: int | None = None,
        attn_qknorm: bool = False,
        attn_val_residual: bool = False,
        attn_gating: str | bool | None = False,
        value_embedding: bool = False,
        value_embedding_scale: float = 1.0,
        value_embedding_gain: bool = True,
        value_embedding_gain_init: float = 0.0,
        value_embedding_gate_channels: int = 32,
        value_embedding_init_std: float | None = None,
        value_embedding_split_token_id: int | None = None,
        n_kv_heads: int | None = None,
        theta_base: float = 10_000.0,
        depth_position: int | None = None,
        max_seq_len: int = 8192,
        is_causal: bool = True,
        attention_impl: str | None = None,
        fuse_qkv: bool = True,
        fuse_swiglu: bool = True,
        linear_init_std: float | None = None,
        dtype: jnp.dtype = jnp.float32,
    ):
        self.ln1 = RMSNorm(rngs, d_model, dtype=dtype, depth_position=depth_position)
        self.attn = MultiHeadSelfAttention(
            rngs,
            d_model,
            n_heads,
            vocab_size=vocab_size,
            n_kv_heads=n_kv_heads,
            theta_base=theta_base,
            max_seq_len=max_seq_len,
            qknorm=attn_qknorm,
            value_residual=attn_val_residual,
            value_embedding=value_embedding,
            value_embedding_scale=value_embedding_scale,
            value_embedding_gain=value_embedding_gain,
            value_embedding_gain_init=value_embedding_gain_init,
            value_embedding_gate_channels=value_embedding_gate_channels,
            value_embedding_init_std=value_embedding_init_std,
            value_embedding_split_token_id=value_embedding_split_token_id,
            gating=attn_gating,
            is_causal=is_causal,
            attention_impl=attention_impl,
            fuse_qkv=fuse_qkv,
            linear_init_std=linear_init_std,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(rngs, d_model, dtype=dtype, depth_position=depth_position)
        self.ffn = SwiGLU(
            rngs,
            d_model,
            d_ff,
            dtype=dtype,
            fuse_up_gate=fuse_swiglu,
            linear_init_std=linear_init_std,
        )

    def __call__(
        self,
        x: Float[Array, "b seq d"],
        token_positions: Int[Array, "b seq"] | None = None,
        v1: Float[Array, "b seq kv_h head_dim"] | None = None,
        token_ids: Int[Array, "b seq"] | None = None,
        attention_mask: Array | None = None,
        is_causal: bool | None = None,
        bd3_block_len: int | None = None,
    ) -> tuple[Float[Array, "b seq d"], Float[Array, "b seq kv_h head_dim"]]:
        attn_out, v = self.attn(
            self.ln1(x),
            token_positions,
            v1,
            token_ids=token_ids,
            attention_mask=attention_mask,
            is_causal=is_causal,
            bd3_block_len=bd3_block_len,
        )
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
        value_embedding: bool = False,
        value_embedding_scale: float = 1.0,
        value_embedding_gain: bool = True,
        value_embedding_gain_init: float = 0.0,
        value_embedding_layers: str | Sequence[int] | None = "alternating",
        value_embedding_gate_channels: int = 32,
        value_embedding_init_std: float | None = None,
        value_embedding_split_token_id: int | None = None,
        layernorm_scaling: bool = False,
        theta_base: float = 10_000.0,
        n_kv_heads: int | None = None,
        max_seq_len: int = 8192,
        is_causal: bool = True,
        attention_impl: str | None = None,
        fuse_qkv: bool = True,
        fuse_swiglu: bool = True,
        dtype: jnp.dtype = jnp.float32,
        weight_tying: bool = False,
        num_grad_checkpoint_layers: int = 0,
    ):
        self.n_layers = n_layers
        self.num_grad_checkpoint_layers = num_grad_checkpoint_layers
        self.value_embedding = value_embedding
        self.value_embedding_layers = value_embedding_layers

        self.embedding = Embedding(rngs, vocab_size, d_model, dtype=dtype)
        self.blocks = _ModuleList(
            [
                Block(
                    rngs,
                    d_model,
                    n_heads,
                    d_ff,
                    vocab_size=vocab_size,
                    attn_qknorm=attn_qknorm,
                    attn_val_residual=attn_val_residual,
                    attn_gating=attn_gating,
                    value_embedding=(
                        value_embedding
                        and has_value_embedding_layer(pos - 1, n_layers, value_embedding_layers)
                    ),
                    value_embedding_scale=value_embedding_scale,
                    value_embedding_gain=value_embedding_gain,
                    value_embedding_gain_init=value_embedding_gain_init,
                    value_embedding_gate_channels=value_embedding_gate_channels,
                    value_embedding_init_std=value_embedding_init_std,
                    value_embedding_split_token_id=value_embedding_split_token_id,
                    n_kv_heads=n_kv_heads,
                    theta_base=theta_base,
                    depth_position=pos if layernorm_scaling else None,
                    max_seq_len=max_seq_len,
                    is_causal=is_causal,
                    attention_impl=attention_impl,
                    fuse_qkv=fuse_qkv,
                    fuse_swiglu=fuse_swiglu,
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

    def encode(
        self,
        token_ids: Int[Array, "b seq"],
        token_positions: Int[Array, "b seq"] | None = None,
        attention_mask: Array | None = None,
        is_causal: bool | None = None,
        bd3_block_len: int | None = None,
    ) -> Float[Array, "b seq d"]:
        """Return final normalized hidden states without materializing logits."""
        x = self.embedding(token_ids)

        v1 = None
        for i, block in enumerate(self.blocks):
            if i < self.num_grad_checkpoint_layers:
                # nnx.remat wraps the stateful callable so the forward is
                # recomputed during the backward pass to save activations.
                rematted = nnx.remat(block)
                x, v = rematted(
                    x,
                    token_positions,
                    v1,
                    token_ids,
                    attention_mask,
                    is_causal,
                    bd3_block_len,
                )
            else:
                x, v = block(
                    x,
                    token_positions,
                    v1,
                    token_ids,
                    attention_mask,
                    is_causal,
                    bd3_block_len,
                )
            if v1 is None:
                v1 = v

        return self.final_norm(x)

    def project_logits(
        self,
        hidden_states: Float[Array, "b seq d"],
    ) -> Float[Array, "b seq vocab"]:
        """Project final hidden states to vocabulary logits."""
        return self.lm_head(hidden_states)

    def __call__(
        self,
        token_ids: Int[Array, "b seq"],
        token_positions: Int[Array, "b seq"] | None = None,
        attention_mask: Array | None = None,
        is_causal: bool | None = None,
        return_hidden: bool = False,
        bd3_block_len: int | None = None,
    ) -> Float[Array, "b seq vocab"] | Float[Array, "b seq d"]:
        hidden = self.encode(
            token_ids,
            token_positions=token_positions,
            attention_mask=attention_mask,
            is_causal=is_causal,
            bd3_block_len=bd3_block_len,
        )
        if return_hidden:
            return hidden
        return self.project_logits(hidden)
