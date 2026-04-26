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
from .moe import add_moe_aux, finalize_moe_aux, zero_moe_aux, SwitchMoE
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


def has_layer(
    layer_idx: int,
    n_layers: int,
    placement: str | Sequence[int] | None,
) -> bool:
    """Return whether a 0-indexed layer is selected by a placement spec."""
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
        if placement in ("alternating", "alternating_late"):
            return layer_idx % 2 == (n_layers - 1) % 2
        if placement == "alternating_early":
            return layer_idx % 2 != (n_layers - 1) % 2
        raise ValueError(
            "placement must be one of 'alternating', 'alternating_late', "
            "'alternating_early', 'all', 'final', 'none', or a sequence of layer indices"
        )
    return layer_idx in set(int(i) for i in placement)


def has_value_embedding_layer(
    layer_idx: int,
    n_layers: int,
    placement: str | Sequence[int] | None = "alternating",
) -> bool:
    """Return whether a 0-indexed layer should receive token value embeddings."""
    return has_layer(layer_idx, n_layers, placement)


def has_moe_layer(
    layer_idx: int,
    n_layers: int,
    placement: str | Sequence[int] | None = "alternating",
) -> bool:
    """Return whether a 0-indexed layer should use an MoE FFN."""
    return has_layer(layer_idx, n_layers, placement)


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
        value_embedding_split_token_zero: bool = False,
        n_kv_heads: int | None = None,
        theta_base: float = 10_000.0,
        depth_position: int | None = None,
        max_seq_len: int = 8192,
        is_causal: bool = True,
        attention_impl: str | None = None,
        fuse_qkv: bool = True,
        fuse_swiglu: bool = True,
        linear_init_std: float | None = None,
        moe: bool = False,
        moe_num_experts: int = 4,
        moe_expert_d_ff: int | None = None,
        moe_capacity_factor: float = 1.25,
        moe_use_router_prob: bool = True,
        moe_split_router_input: bool = True,
        moe_router_dtype: jnp.dtype = jnp.float32,
        moe_drop_tokens: bool = True,
        dtype: jnp.dtype = jnp.float32,
    ):
        self.is_moe = bool(moe)
        self.moe_split_router_input = bool(moe_split_router_input)
        self.moe_expert_input_scale = (
            float(jax.lax.rsqrt(jnp.float32(depth_position)))
            if self.is_moe and self.moe_split_router_input and depth_position is not None
            else 1.0
        )

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
            value_embedding_split_token_zero=value_embedding_split_token_zero,
            gating=attn_gating,
            is_causal=is_causal,
            attention_impl=attention_impl,
            fuse_qkv=fuse_qkv,
            linear_init_std=linear_init_std,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(
            rngs,
            d_model,
            dtype=dtype,
            depth_position=None if self.is_moe and self.moe_split_router_input else depth_position,
        )
        if self.is_moe:
            self.ffn = SwitchMoE(
                rngs,
                d_model,
                d_ff,
                num_experts=moe_num_experts,
                expert_d_ff=moe_expert_d_ff,
                capacity_factor=moe_capacity_factor,
                use_router_prob=moe_use_router_prob,
                router_dtype=moe_router_dtype,
                drop_tokens=moe_drop_tokens,
                fuse_up_gate=fuse_swiglu,
                linear_init_std=linear_init_std,
                dtype=dtype,
            )
        else:
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
        return_aux: bool = False,
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
        h = self.ln2(x)
        if self.is_moe:
            if self.moe_split_router_input:
                expert_h = h * jnp.asarray(self.moe_expert_input_scale, dtype=h.dtype)
                ffn_out, moe_aux = self.ffn(expert_h, router_x=h)
            else:
                ffn_out, moe_aux = self.ffn(h)
        else:
            ffn_out = self.ffn(h)
            moe_aux = zero_moe_aux()
        x = x + ffn_out
        if return_aux:
            return x, v, moe_aux
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
        value_embedding_split_token_zero: bool = False,
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
        moe: bool = False,
        moe_routing: str = "token_choice_switch",
        moe_top_k: int = 1,
        moe_layers: str | Sequence[int] | None = "alternating",
        moe_num_experts: int = 4,
        moe_expert_d_ff: int | None = None,
        moe_capacity_factor: float = 1.25,
        moe_use_router_prob: bool = True,
        moe_split_router_input: bool = True,
        moe_router_dtype: jnp.dtype = jnp.float32,
        moe_drop_tokens: bool = True,
    ):
        self.n_layers = n_layers
        self.num_grad_checkpoint_layers = num_grad_checkpoint_layers
        self.value_embedding = value_embedding
        self.value_embedding_layers = value_embedding_layers
        self.moe = bool(moe)
        self.moe_layers = moe_layers
        if self.moe:
            if moe_routing not in ("token_choice_switch", "token_choice_top1_switch_swiglu"):
                raise ValueError("Only token_choice_switch MoE routing is supported")
            if int(moe_top_k) != 1:
                raise ValueError("Only moe_top_k=1 is supported")

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
                    value_embedding_split_token_zero=value_embedding_split_token_zero,
                    n_kv_heads=n_kv_heads,
                    theta_base=theta_base,
                    depth_position=pos if layernorm_scaling else None,
                    max_seq_len=max_seq_len,
                    is_causal=is_causal,
                    attention_impl=attention_impl,
                    fuse_qkv=fuse_qkv,
                    fuse_swiglu=fuse_swiglu,
                    moe=(
                        moe
                        and has_moe_layer(pos - 1, n_layers, moe_layers)
                    ),
                    moe_num_experts=moe_num_experts,
                    moe_expert_d_ff=moe_expert_d_ff,
                    moe_capacity_factor=moe_capacity_factor,
                    moe_use_router_prob=moe_use_router_prob,
                    moe_split_router_input=moe_split_router_input,
                    moe_router_dtype=moe_router_dtype,
                    moe_drop_tokens=moe_drop_tokens,
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
        return_aux: bool = False,
    ) -> Float[Array, "b seq d"]:
        """Return final normalized hidden states without materializing logits."""
        x = self.embedding(token_ids)

        v1 = None
        moe_aux = zero_moe_aux()
        for i, block in enumerate(self.blocks):
            if i < self.num_grad_checkpoint_layers:
                # nnx.remat wraps the stateful callable so the forward is
                # recomputed during the backward pass to save activations.
                rematted = nnx.remat(block, static_argnums=(7,))
                if return_aux:
                    x, v, block_aux = rematted(
                        x,
                        token_positions,
                        v1,
                        token_ids,
                        attention_mask,
                        is_causal,
                        bd3_block_len,
                        return_aux,
                    )
                    moe_aux = add_moe_aux(moe_aux, block_aux)
                else:
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
                if return_aux:
                    x, v, block_aux = block(
                        x,
                        token_positions,
                        v1,
                        token_ids,
                        attention_mask,
                        is_causal,
                        bd3_block_len,
                        return_aux,
                    )
                    moe_aux = add_moe_aux(moe_aux, block_aux)
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

        hidden = self.final_norm(x)
        if return_aux:
            return hidden, finalize_moe_aux(moe_aux)
        return hidden

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
        return_aux: bool = False,
    ) -> Float[Array, "b seq vocab"] | Float[Array, "b seq d"]:
        hidden = self.encode(
            token_ids,
            token_positions=token_positions,
            attention_mask=attention_mask,
            is_causal=is_causal,
            bd3_block_len=bd3_block_len,
            return_aux=return_aux,
        )
        aux = None
        if return_aux:
            hidden, aux = hidden
        if return_hidden:
            if return_aux:
                return hidden, aux
            return hidden
        logits = self.project_logits(hidden)
        if return_aux:
            return logits, aux
        return logits
