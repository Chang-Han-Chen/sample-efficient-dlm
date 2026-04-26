"""Sparse token-choice top-1 Switch MoE over SwiGLU experts."""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx

from .core import Linear, SwiGLU

Array = jax.Array

try:
    from flax.nnx import List as _ModuleList
except ImportError:  # pragma: no cover - fallback for flax < 0.11
    _ModuleList = list


class MoEAux(NamedTuple):
    load_balance_loss: Array
    router_z_loss: Array
    dropped_fraction: Array
    router_entropy: Array
    expert_fraction_min: Array
    expert_fraction_max: Array
    expert_fraction_std: Array
    router_prob_fraction_min: Array
    router_prob_fraction_max: Array
    router_prob_fraction_std: Array
    num_moe_layers: Array


def zero_moe_aux() -> MoEAux:
    z = jnp.zeros((), dtype=jnp.float32)
    return MoEAux(z, z, z, z, z, z, z, z, z, z, z)


def add_moe_aux(a: MoEAux, b: MoEAux) -> MoEAux:
    return MoEAux(*(x + y for x, y in zip(a, b, strict=True)))


def finalize_moe_aux(aux: MoEAux) -> MoEAux:
    denom = jnp.maximum(aux.num_moe_layers, 1.0)
    return MoEAux(
        aux.load_balance_loss / denom,
        aux.router_z_loss / denom,
        aux.dropped_fraction / denom,
        aux.router_entropy / denom,
        aux.expert_fraction_min / denom,
        aux.expert_fraction_max / denom,
        aux.expert_fraction_std / denom,
        aux.router_prob_fraction_min / denom,
        aux.router_prob_fraction_max / denom,
        aux.router_prob_fraction_std / denom,
        aux.num_moe_layers,
    )


class SwitchMoE(nnx.Module):
    """Token-choice top-1 Switch routing over independent SwiGLU experts."""

    def __init__(
        self,
        rngs: nnx.Rngs,
        d_model: int,
        d_ff: int,
        *,
        num_experts: int,
        expert_d_ff: int | None = None,
        capacity_factor: float = 1.25,
        use_router_prob: bool = True,
        router_dtype: jnp.dtype = jnp.float32,
        drop_tokens: bool = True,
        fuse_up_gate: bool = True,
        linear_init_std: float | None = None,
        dtype: jnp.dtype = jnp.float32,
    ):
        if int(num_experts) < 1:
            raise ValueError("num_experts must be >= 1")
        if float(capacity_factor) <= 0.0:
            raise ValueError("capacity_factor must be > 0")
        if jnp.dtype(router_dtype) != jnp.dtype(jnp.float32):
            raise ValueError("First MoE implementation only supports fp32 router dtype")
        if not bool(drop_tokens):
            raise ValueError("First MoE implementation requires drop_tokens=True")

        expert_d_ff = d_ff if expert_d_ff is None else int(expert_d_ff)
        self.num_experts = int(num_experts)
        self.capacity_factor = float(capacity_factor)
        self.use_router_prob = bool(use_router_prob)
        self.drop_tokens = bool(drop_tokens)
        self.d_model = int(d_model)
        self.d_ff = int(d_ff)
        self.expert_d_ff = int(expert_d_ff)
        self.dtype = dtype
        self.router_dtype = router_dtype

        self.router = Linear(rngs, d_model, self.num_experts, dtype=router_dtype)
        self.experts = _ModuleList(
            [
                SwiGLU(
                    rngs,
                    d_model,
                    self.expert_d_ff,
                    dtype=dtype,
                    fuse_up_gate=fuse_up_gate,
                    linear_init_std=linear_init_std,
                )
                for _ in range(self.num_experts)
            ]
        )

    def __call__(self, x: Array) -> tuple[Array, MoEAux]:
        bsz, seq_len, d_model = x.shape
        n_tokens = bsz * seq_len
        n_experts = self.num_experts
        capacity = max(1, int(math.ceil(self.capacity_factor * n_tokens / n_experts)))

        x_flat = x.reshape((n_tokens, d_model))
        router_logits = self.router(x_flat.astype(jnp.float32)).astype(jnp.float32)
        router_probs = jax.nn.softmax(router_logits, axis=-1)
        expert_id = jnp.argmax(router_probs, axis=-1).astype(jnp.int32)
        selected_prob = jnp.take_along_axis(router_probs, expert_id[:, None], axis=-1)[:, 0]

        expert_one_hot_i = jax.nn.one_hot(expert_id, n_experts, dtype=jnp.int32)
        positions_all = jnp.cumsum(expert_one_hot_i, axis=0) - 1
        slot = jnp.sum(positions_all * expert_one_hot_i, axis=-1)
        valid = slot < capacity
        safe_slot = jnp.where(valid, slot, 0)

        expert_inputs = jnp.zeros((n_experts, capacity, d_model), dtype=x.dtype)
        expert_inputs = expert_inputs.at[expert_id, safe_slot].add(
            x_flat * valid[:, None].astype(x.dtype)
        )
        expert_outputs = jnp.stack(
            [self.experts[e](expert_inputs[e]) for e in range(n_experts)],
            axis=0,
        )

        y_flat = expert_outputs[expert_id, safe_slot]
        y_flat = y_flat * valid[:, None].astype(y_flat.dtype)
        if self.use_router_prob:
            y_flat = y_flat * selected_prob[:, None].astype(y_flat.dtype)
        y = y_flat.reshape((bsz, seq_len, d_model))

        expert_fraction = jnp.mean(
            jax.nn.one_hot(expert_id, n_experts, dtype=jnp.float32),
            axis=0,
        )
        router_prob_fraction = jnp.mean(router_probs, axis=0)
        load_balance_loss = n_experts * jnp.sum(
            jax.lax.stop_gradient(expert_fraction) * router_prob_fraction
        )
        router_z_loss = jnp.mean(jax.nn.logsumexp(router_logits, axis=-1) ** 2)
        router_entropy = -jnp.mean(
            jnp.sum(router_probs * jnp.log(router_probs + 1e-9), axis=-1)
        )
        dropped_fraction = 1.0 - jnp.mean(valid.astype(jnp.float32))

        aux = MoEAux(
            load_balance_loss=load_balance_loss,
            router_z_loss=router_z_loss,
            dropped_fraction=dropped_fraction,
            router_entropy=router_entropy,
            expert_fraction_min=jnp.min(expert_fraction),
            expert_fraction_max=jnp.max(expert_fraction),
            expert_fraction_std=jnp.std(expert_fraction),
            router_prob_fraction_min=jnp.min(router_prob_fraction),
            router_prob_fraction_max=jnp.max(router_prob_fraction),
            router_prob_fraction_std=jnp.std(router_prob_fraction),
            num_moe_layers=jnp.ones((), dtype=jnp.float32),
        )
        return y, aux
