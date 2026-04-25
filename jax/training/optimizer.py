"""NorMuon+AdamW optimizer for Flax NNX models.

The grouping follows the role-based pattern in ``karpathy/train.py`` while
keeping only three LR families for the tied-embedding JAX model. The default
constants stay anchored to the ``pytorch/`` small-GPT optimizer recipe:

* AdamW table params: token embedding / tied lm head.
* AdamW value-embedding table params: same LR as tables, separate betas.
* AdamW scalar/vector params: norms, QK gain, value-residual scalars.
* NorMuon with cautious weight decay for the remaining matrix parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx

Array = jax.Array


@dataclass(frozen=True)
class NormuonAdamWConfig:
    table_adam_lr: float = 7e-3
    scalar_adam_lr: float = 7e-3
    muon_lr: float = 1.5e-2
    adam_weight_decay: float = 0.0
    muon_weight_decay: float = 1e-4
    adam_betas: tuple[float, float] = (0.95, 0.99)
    value_embedding_adam_betas: tuple[float, float] = (0.8, 0.95)
    muon_momentum: float = 0.95
    muon_beta2: float = 0.95
    adam_eps: float = 1e-8
    warmup_steps: int = 100
    momentum_warmup_steps: int = 300
    momentum_warmup_start: float = 0.85
    lr_min_coeff: float = 1.0
    scheduler: str = "warmup_stable"
    cautious_weight_decay: bool = True


@dataclass(frozen=True)
class ParamSpec:
    kind: str
    path: str
    shape: tuple[int, ...]
    lr_mul: float = 1.0
    wd_mul: float = 1.0


class NormuonAdamWState(NamedTuple):
    count: Array
    adam_m: object
    adam_v: object
    muon_momentum: object
    muon_second: object


def _path_to_str(path: tuple[object, ...]) -> str:
    return ".".join(str(part) for part in path)


def _is_table_param(path: str) -> bool:
    return path == "embedding.weight" or path == "lm_head.weight"


def _is_value_embedding_table_param(path: str) -> bool:
    return path.endswith(".value_embedding_table.weight")


def _param_kind(path: str, value: Array) -> str:
    if _is_table_param(path):
        return "adam_table"
    if _is_value_embedding_table_param(path):
        return "adam_ve_table"
    if value.ndim < 2:
        return "adam_scalar"
    return "muon"


def build_param_specs(model) -> nnx.State:
    """Build a static tree that labels each trainable param as AdamW or Muon."""
    flat = []
    for path, variable in nnx.to_flat_state(nnx.state(model, nnx.Param)):
        path_str = _path_to_str(path)
        value = variable.value
        kind = _param_kind(path_str, value)
        flat.append((path, ParamSpec(kind=kind, path=path_str, shape=tuple(value.shape))))
    return nnx.from_flat_state(flat, cls=nnx.State)


def learning_rates(step: Array, cfg: NormuonAdamWConfig) -> tuple[Array, Array, Array]:
    """Return table-AdamW, scalar-AdamW, and Muon LRs for this optimizer step."""
    step_f = step.astype(jnp.float32)
    warmup = max(int(cfg.warmup_steps), 1)
    if cfg.scheduler == "warmup_stable":
        mult = jnp.minimum(step_f / float(warmup), 1.0)
    elif cfg.scheduler == "constant":
        mult = jnp.ones((), dtype=jnp.float32)
    else:
        raise ValueError(f"Unsupported scheduler: {cfg.scheduler!r}")
    return cfg.table_adam_lr * mult, cfg.scalar_adam_lr * mult, cfg.muon_lr * mult


def _muon_momentum(step: Array, cfg: NormuonAdamWConfig) -> Array:
    if cfg.momentum_warmup_steps <= 0:
        return jnp.asarray(cfg.muon_momentum, dtype=jnp.float32)
    step_f = jnp.minimum(jnp.maximum(step.astype(jnp.float32), 1.0), float(cfg.momentum_warmup_steps))
    frac = step_f / float(cfg.momentum_warmup_steps)
    return cfg.momentum_warmup_start + frac * (cfg.muon_momentum - cfg.momentum_warmup_start)


def _second_moment_zeros(param: Array, spec: ParamSpec) -> Array:
    if spec.kind != "muon":
        return jnp.zeros_like(param)
    if param.shape[-2] >= param.shape[-1]:
        return jnp.zeros((*param.shape[:-1], 1), dtype=jnp.float32)
    return jnp.zeros((*param.shape[:-2], 1, param.shape[-1]), dtype=jnp.float32)


_POLAR_EXPRESS_COEFFS = (
    (7.2086, -15.5131, 9.0178),
    (3.9623, -2.5813, 0.4542),
    (3.9466, -2.5765, 0.4544),
    (3.8991, -2.5671, 0.4566),
    (3.7186, -2.5308, 0.4653),
    (3.1390, -2.3073, 0.4733),
    (2.1715, -1.5246, 0.3885),
)


def _orthogonalize_update(g: Array, eps: float = 1e-7) -> Array:
    """Newton-Schulz / Polar Express update used by the PyTorch Muon path."""
    transpose = g.shape[-2] > g.shape[-1]
    x = jnp.swapaxes(g, -1, -2) if transpose else g
    x = x.astype(jnp.bfloat16)
    x = x / (jnp.linalg.norm(x, axis=(-2, -1), keepdims=True) + eps)
    for a, b, c in _POLAR_EXPRESS_COEFFS:
        gram = x @ jnp.swapaxes(x, -1, -2)
        update = b * gram + c * (gram @ gram)
        x = a * x + update @ x
    return jnp.swapaxes(x, -1, -2).astype(jnp.float32) if transpose else x.astype(jnp.float32)


def _normuon_leaf(
    grad: Array,
    param: Array,
    momentum_buffer: Array,
    second_buffer: Array,
    spec: ParamSpec,
    *,
    lr: Array,
    weight_decay: float,
    momentum: Array,
    beta2: float,
    cautious_weight_decay: bool,
) -> tuple[Array, Array, Array]:
    m = momentum * momentum_buffer + (1.0 - momentum) * grad
    g = (1.0 - momentum) * grad + momentum * m
    v = _orthogonalize_update(g)

    red_axis = -1 if param.shape[-2] >= param.shape[-1] else -2
    v_norm = jnp.linalg.norm(v, axis=(-2, -1), keepdims=True)
    v_mean = jnp.mean(v * v, axis=red_axis, keepdims=True)
    second = beta2 * second_buffer + (1.0 - beta2) * v_mean
    step_size = jax.lax.rsqrt(jnp.maximum(second, 1e-10))
    v_scaled = v * step_size
    v_norm_new = jnp.maximum(jnp.linalg.norm(v_scaled, axis=(-2, -1), keepdims=True), 1e-10)
    v_scaled = v_scaled * (v_norm / v_norm_new)

    rows, cols = param.shape[-2], param.shape[-1]
    mup_mult = max(1.0, rows / cols) ** 0.5
    eff_lr = lr * mup_mult * spec.lr_mul
    eff_wd = lr * weight_decay * spec.wd_mul
    if cautious_weight_decay:
        wd_mask = (v_scaled * param.astype(jnp.float32)) >= 0
        wd_update = -eff_wd * param * wd_mask.astype(param.dtype)
    else:
        wd_update = -eff_wd * param
    update = wd_update - eff_lr * v_scaled.astype(param.dtype)
    return update, m, second


def create_normuon_adamw(model, cfg: NormuonAdamWConfig) -> optax.GradientTransformation:
    specs = build_param_specs(model)

    def init_fn(params):
        params = nnx.as_pure(params)
        adam_m = jax.tree_util.tree_map(jnp.zeros_like, params)
        adam_v = jax.tree_util.tree_map(jnp.zeros_like, params)
        muon_m = jax.tree_util.tree_map(jnp.zeros_like, params)
        muon_second = jax.tree_util.tree_map(_second_moment_zeros, params, specs)
        return NormuonAdamWState(
            count=jnp.zeros((), dtype=jnp.int32),
            adam_m=adam_m,
            adam_v=adam_v,
            muon_momentum=muon_m,
            muon_second=muon_second,
        )

    def update_fn(grads, state: NormuonAdamWState, params):
        table_adam_lr, scalar_adam_lr, muon_lr = learning_rates(state.count, cfg)
        muon_momentum = _muon_momentum(state.count, cfg)
        t = state.count.astype(jnp.float32) + 1.0

        def update_leaf(grad, param, spec, adam_m, adam_v, muon_m, muon_second):
            if spec.kind in ("adam_table", "adam_ve_table", "adam_scalar"):
                adam_lr = table_adam_lr if spec.kind in ("adam_table", "adam_ve_table") else scalar_adam_lr
                beta1, beta2 = (
                    cfg.value_embedding_adam_betas
                    if spec.kind == "adam_ve_table"
                    else cfg.adam_betas
                )
                new_m = beta1 * adam_m + (1.0 - beta1) * grad
                new_v = beta2 * adam_v + (1.0 - beta2) * (grad * grad)
                bias1 = 1.0 - beta1**t
                bias2 = 1.0 - beta2**t
                step_size = adam_lr * jnp.sqrt(bias2) / bias1
                denom = jnp.sqrt(new_v) + cfg.adam_eps
                direction = jnp.nan_to_num(
                    new_m / denom,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                update = -adam_lr * cfg.adam_weight_decay * param - step_size * direction
                return update.astype(param.dtype), new_m, new_v, muon_m, muon_second

            update, new_muon_m, new_muon_second = _normuon_leaf(
                grad.astype(jnp.float32),
                param,
                muon_m.astype(jnp.float32),
                muon_second.astype(jnp.float32),
                spec,
                lr=muon_lr,
                weight_decay=cfg.muon_weight_decay,
                momentum=muon_momentum,
                beta2=cfg.muon_beta2,
                cautious_weight_decay=cfg.cautious_weight_decay,
            )
            update = jnp.nan_to_num(update, nan=0.0, posinf=0.0, neginf=0.0)
            return update.astype(param.dtype), adam_m, adam_v, new_muon_m, new_muon_second

        mapped = jax.tree_util.tree_map(
            update_leaf,
            grads,
            params,
            specs,
            state.adam_m,
            state.adam_v,
            state.muon_momentum,
            state.muon_second,
        )
        params_def = jax.tree_util.tree_structure(params)
        tuple_def = jax.tree_util.tree_structure((0, 0, 0, 0, 0))
        updates, adam_m, adam_v, muon_m, muon_second = jax.tree_util.tree_transpose(
            params_def,
            tuple_def,
            mapped,
        )
        new_state = NormuonAdamWState(
            count=state.count + 1,
            adam_m=adam_m,
            adam_v=adam_v,
            muon_momentum=muon_m,
            muon_second=muon_second,
        )
        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


def global_norm(grads) -> Array:
    leaves = jax.tree_util.tree_leaves(nnx.as_pure(grads))
    total = sum(jnp.sum(jnp.square(x.astype(jnp.float32))) for x in leaves)
    return jnp.sqrt(total)


def clip_by_global_norm(grads, max_norm: float, eps: float = 1e-8):
    norm = global_norm(grads)
    scale = jnp.minimum(1.0, max_norm / (norm + eps))
    return jax.tree_util.tree_map(lambda g: g * scale, grads), norm
