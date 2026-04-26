"""Jitted AR train/eval steps."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from flax import nnx

from .loss import (
    ar_loss,
    cross_entropy_with_z_loss,
    supervised_lm_loss,
    supervised_lm_loss_sums,
)
from .optimizer import clip_by_global_norm

Array = jax.Array
DATA_AXIS = "data"
BROADCAST_GRAPH_AXES = nnx.StateAxes({...: None})


def loss_fn(
    model,
    inputs: Array,
    targets: Array,
    z_loss_weight: float,
    loss_impl: str,
    logit_chunk_size: int,
    moe_load_balance_loss_weight: float = 0.0,
    moe_router_z_loss_weight: float = 0.0,
):
    return ar_loss(
        model,
        inputs,
        targets,
        z_loss_weight=z_loss_weight,
        loss_impl=loss_impl,
        logit_chunk_size=logit_chunk_size,
        moe_load_balance_loss_weight=moe_load_balance_loss_weight,
        moe_router_z_loss_weight=moe_router_z_loss_weight,
    )


def supervised_loss_fn(
    model,
    inputs: Array,
    targets: Array,
    supervise_mask: Array,
    token_positions: Array | None,
    attention_mask: Array | None,
    z_loss_weight: float,
    is_causal: bool | None,
    output_length: int | None,
    bd3_block_len: int | None,
    loss_impl: str,
    logit_chunk_size: int,
    loss_denominator: Array | float | None = None,
    moe_average_axis: str | None = None,
    moe_load_balance_loss_weight: float = 0.0,
    moe_router_z_loss_weight: float = 0.0,
):
    """Sum-form supervised loss used inside ``value_and_grad``.

    Returns ``(total_sum, metrics_sums)``. Each train step variant divides
    by the requested denominator after local accumulation and/or device
    ``psum``. AR currently uses the cheaper ``loss_fn`` mean path.
    """
    return supervised_lm_loss_sums(
        model,
        inputs,
        targets,
        supervise_mask,
        token_positions=token_positions,
        attention_mask=attention_mask,
        is_causal=is_causal,
        output_length=output_length,
        bd3_block_len=bd3_block_len,
        z_loss_weight=z_loss_weight,
        loss_impl=loss_impl,
        logit_chunk_size=logit_chunk_size,
        loss_denominator=loss_denominator,
        moe_average_axis=moe_average_axis,
        moe_load_balance_loss_weight=moe_load_balance_loss_weight,
        moe_router_z_loss_weight=moe_router_z_loss_weight,
    )


def _safe_inv(denominator: Array) -> Array:
    return jnp.where(denominator > 0, 1.0 / denominator, 0.0)


def _scale_grads(grads, scale: Array):
    return jax.tree_util.tree_map(lambda g: g * scale, grads)


def _loss_denominator(loss_normalizer: float | Array | None, fallback_count: Array) -> Array:
    if loss_normalizer is None:
        return fallback_count
    return jnp.asarray(loss_normalizer, dtype=jnp.float32)


def _supervised_valid_count(
    targets: Array,
    supervise_mask: Array | None,
    *,
    output_length: int | None = None,
    ignore_index: int | None = None,
) -> Array:
    target_slice = targets[:, :output_length] if output_length is not None else targets
    if supervise_mask is None:
        if ignore_index is None:
            return jnp.asarray(target_slice.size, dtype=jnp.float32)
        return jnp.sum((target_slice != ignore_index).astype(jnp.float32))
    mask = supervise_mask[:, :output_length] if output_length is not None else supervise_mask
    valid = mask.astype(bool)
    if ignore_index is not None:
        valid = valid & (target_slice != ignore_index)
    return jnp.sum(valid.astype(jnp.float32))


def _add_moe_step_metrics(out: dict[str, Array], metrics_sums: dict[str, Array], inv: Array) -> None:
    if "moe_aux_loss_sum" in metrics_sums:
        out["moe_aux_loss"] = metrics_sums["moe_aux_loss_sum"] * inv
        out["moe_load_balance_loss"] = metrics_sums["moe_load_balance_loss_sum"] * inv
        out["moe_router_z_loss"] = metrics_sums["moe_router_z_loss_sum"] * inv
    for key, value in metrics_sums.items():
        if key.startswith("moe_") and key not in {
            "moe_aux_loss_sum",
            "moe_load_balance_loss_sum",
            "moe_router_z_loss_sum",
        } and key not in out:
            out[key] = value


def _init_moe_metric_sums(metrics: dict[str, Array]) -> dict[str, Array]:
    return {key: value for key, value in metrics.items() if key.startswith("moe_")}


def _accumulate_moe_metric_sums(dst: dict[str, Array], src: dict[str, Array]) -> None:
    for key in dst:
        dst[key] = dst[key] + src[key]


def _finalize_accumulated_moe_metrics(metrics: dict[str, Array], accum_steps: int) -> dict[str, Array]:
    out = {}
    inv_steps = 1.0 / float(accum_steps)
    for key, value in metrics.items():
        if key.endswith("_sum"):
            out[key] = value
        else:
            out[key] = value * inv_steps
    return out


@functools.partial(nnx.jit, static_argnums=(6, 7))
def train_step(
    model,
    optimizer,
    inputs: Array,
    targets: Array,
    z_loss_weight: float,
    max_grad_norm: float,
    loss_impl: str = "full",
    logit_chunk_size: int = 1024,
    moe_load_balance_loss_weight: float = 0.0,
    moe_router_z_loss_weight: float = 0.0,
):
    (total_loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
        model,
        inputs,
        targets,
        z_loss_weight,
        loss_impl,
        logit_chunk_size,
        moe_load_balance_loss_weight,
        moe_router_z_loss_weight,
    )
    clipped_grads, grad_norm = clip_by_global_norm(grads, max_grad_norm)
    optimizer.update(model, clipped_grads)
    metrics["total_loss"] = total_loss
    metrics["grad_norm"] = grad_norm
    return metrics


@functools.partial(
    nnx.pmap,
    axis_name=DATA_AXIS,
    in_axes=(BROADCAST_GRAPH_AXES, BROADCAST_GRAPH_AXES, 0, 0, None, None, None, None, None, None),
    static_broadcasted_argnums=(6, 7),
)
def _train_step_data_parallel_pmapped(
    model,
    optimizer,
    inputs: Array,
    targets: Array,
    z_loss_weight: float,
    max_grad_norm: float,
    loss_impl: str = "full",
    logit_chunk_size: int = 1024,
    moe_load_balance_loss_weight: float = 0.0,
    moe_router_z_loss_weight: float = 0.0,
):
    """Single optimizer step over a pmapped, data-parallel microbatch."""
    (total_loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
        model,
        inputs,
        targets,
        z_loss_weight,
        loss_impl,
        logit_chunk_size,
        moe_load_balance_loss_weight,
        moe_router_z_loss_weight,
    )
    mean_grads = jax.lax.pmean(grads, DATA_AXIS)
    clipped_grads, grad_norm = clip_by_global_norm(mean_grads, max_grad_norm)
    optimizer.update(model, clipped_grads)
    metrics = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, DATA_AXIS), metrics)
    metrics["total_loss"] = jax.lax.pmean(total_loss, DATA_AXIS)
    metrics["grad_norm"] = grad_norm
    return metrics


def train_step_data_parallel(
    model,
    optimizer,
    inputs: Array,
    targets: Array,
    z_loss_weight: float,
    max_grad_norm: float,
    loss_impl: str = "full",
    logit_chunk_size: int = 1024,
    moe_load_balance_loss_weight: float = 0.0,
    moe_router_z_loss_weight: float = 0.0,
):
    return _train_step_data_parallel_pmapped(
        model,
        optimizer,
        inputs,
        targets,
        z_loss_weight,
        max_grad_norm,
        loss_impl,
        logit_chunk_size,
        moe_load_balance_loss_weight,
        moe_router_z_loss_weight,
    )


@functools.partial(nnx.jit, static_argnums=(6, 7))
def train_step_accumulated(
    model,
    optimizer,
    inputs: Array,
    targets: Array,
    z_loss_weight: float,
    max_grad_norm: float,
    loss_impl: str = "full",
    logit_chunk_size: int = 1024,
    moe_load_balance_loss_weight: float = 0.0,
    moe_router_z_loss_weight: float = 0.0,
):
    """Update once using mean gradients over leading microbatch axis."""
    accum_steps = inputs.shape[0]
    (total_loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
        model,
        inputs[0],
        targets[0],
        z_loss_weight,
        loss_impl,
        logit_chunk_size,
        moe_load_balance_loss_weight,
        moe_router_z_loss_weight,
    )
    summed_grads = grads
    metric_sums = metrics

    for i in range(1, accum_steps):
        (loss_i, metrics_i), grads_i = nnx.value_and_grad(loss_fn, has_aux=True)(
            model,
            inputs[i],
            targets[i],
            z_loss_weight,
            loss_impl,
            logit_chunk_size,
            moe_load_balance_loss_weight,
            moe_router_z_loss_weight,
        )
        total_loss = total_loss + loss_i
        summed_grads = jax.tree_util.tree_map(lambda a, b: a + b, summed_grads, grads_i)
        metric_sums = {
            k: metric_sums[k] + metrics_i[k]
            for k in metric_sums
        }

    inv = 1.0 / float(accum_steps)
    mean_grads = jax.tree_util.tree_map(lambda g: g * inv, summed_grads)
    clipped_grads, grad_norm = clip_by_global_norm(mean_grads, max_grad_norm)
    optimizer.update(model, clipped_grads)
    metrics = {k: v * inv for k, v in metric_sums.items()}
    metrics["total_loss"] = total_loss * inv
    metrics["grad_norm"] = grad_norm
    return metrics


@functools.partial(
    nnx.pmap,
    axis_name=DATA_AXIS,
    in_axes=(BROADCAST_GRAPH_AXES, BROADCAST_GRAPH_AXES, 0, 0, None, None, None, None, None, None),
    static_broadcasted_argnums=(6, 7),
)
def _train_step_accumulated_data_parallel_pmapped(
    model,
    optimizer,
    inputs: Array,
    targets: Array,
    z_loss_weight: float,
    max_grad_norm: float,
    loss_impl: str = "full",
    logit_chunk_size: int = 1024,
    moe_load_balance_loss_weight: float = 0.0,
    moe_router_z_loss_weight: float = 0.0,
):
    """Data-parallel step with local gradient accumulation on each device."""
    accum_steps = inputs.shape[0]
    (total_loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
        model,
        inputs[0],
        targets[0],
        z_loss_weight,
        loss_impl,
        logit_chunk_size,
        moe_load_balance_loss_weight,
        moe_router_z_loss_weight,
    )
    summed_grads = grads
    metric_sums = metrics

    for i in range(1, accum_steps):
        (loss_i, metrics_i), grads_i = nnx.value_and_grad(loss_fn, has_aux=True)(
            model,
            inputs[i],
            targets[i],
            z_loss_weight,
            loss_impl,
            logit_chunk_size,
            moe_load_balance_loss_weight,
            moe_router_z_loss_weight,
        )
        total_loss = total_loss + loss_i
        summed_grads = jax.tree_util.tree_map(lambda a, b: a + b, summed_grads, grads_i)
        metric_sums = {k: metric_sums[k] + metrics_i[k] for k in metric_sums}

    inv = 1.0 / float(accum_steps)
    local_mean_grads = jax.tree_util.tree_map(lambda g: g * inv, summed_grads)
    mean_grads = jax.lax.pmean(local_mean_grads, DATA_AXIS)
    clipped_grads, grad_norm = clip_by_global_norm(mean_grads, max_grad_norm)
    optimizer.update(model, clipped_grads)
    metrics = {
        k: jax.lax.pmean(v * inv, DATA_AXIS)
        for k, v in metric_sums.items()
    }
    metrics["total_loss"] = jax.lax.pmean(total_loss * inv, DATA_AXIS)
    metrics["grad_norm"] = grad_norm
    return metrics


def train_step_accumulated_data_parallel(
    model,
    optimizer,
    inputs: Array,
    targets: Array,
    z_loss_weight: float,
    max_grad_norm: float,
    loss_impl: str = "full",
    logit_chunk_size: int = 1024,
    moe_load_balance_loss_weight: float = 0.0,
    moe_router_z_loss_weight: float = 0.0,
):
    return _train_step_accumulated_data_parallel_pmapped(
        model,
        optimizer,
        inputs,
        targets,
        z_loss_weight,
        max_grad_norm,
        loss_impl,
        logit_chunk_size,
        moe_load_balance_loss_weight,
        moe_router_z_loss_weight,
    )


@functools.partial(nnx.jit, static_argnums=(9, 10, 11, 12, 13))
def train_step_supervised(
    model,
    optimizer,
    inputs: Array,
    targets: Array,
    supervise_mask: Array,
    token_positions: Array | None,
    attention_mask: Array | None,
    z_loss_weight: float,
    max_grad_norm: float,
    is_causal: bool | None,
    output_length: int | None,
    loss_impl: str = "full",
    logit_chunk_size: int = 1024,
    bd3_block_len: int | None = None,
    loss_normalizer: float | Array | None = None,
    moe_load_balance_loss_weight: float = 0.0,
    moe_router_z_loss_weight: float = 0.0,
):
    valid_count_for_denom = _supervised_valid_count(
        targets,
        supervise_mask,
        output_length=output_length,
    )
    denominator = _loss_denominator(loss_normalizer, valid_count_for_denom)
    (total_sum, metrics_sums), grads = nnx.value_and_grad(supervised_loss_fn, has_aux=True)(
        model,
        inputs,
        targets,
        supervise_mask,
        token_positions,
        attention_mask,
        z_loss_weight,
        is_causal,
        output_length,
        bd3_block_len,
        loss_impl,
        logit_chunk_size,
        denominator,
        None,
        moe_load_balance_loss_weight,
        moe_router_z_loss_weight,
    )
    valid_count = metrics_sums["valid_count"]
    inv = _safe_inv(denominator)
    grads = _scale_grads(grads, inv)
    clipped_grads, grad_norm = clip_by_global_norm(grads, max_grad_norm)
    optimizer.update(model, clipped_grads)
    out = {
        "loss": metrics_sums["loss_sum"] * inv,
        "z_loss": metrics_sums["z_loss_sum"] * inv,
        "total_loss": total_sum * inv,
        "supervised_tokens": valid_count,
        "loss_normalizer": denominator,
        "grad_norm": grad_norm,
    }
    _add_moe_step_metrics(out, metrics_sums, inv)
    return out


@functools.partial(
    nnx.pmap,
    axis_name=DATA_AXIS,
    in_axes=(BROADCAST_GRAPH_AXES, BROADCAST_GRAPH_AXES, 0, 0, 0, None, None, None, None, None, None, None, None, None, None, None, None),
    static_broadcasted_argnums=(9, 10, 11, 12, 13),
)
def _train_step_supervised_data_parallel_pmapped(
    model,
    optimizer,
    inputs: Array,
    targets: Array,
    supervise_mask: Array,
    token_positions: Array | None,
    attention_mask: Array | None,
    z_loss_weight: float,
    max_grad_norm: float,
    is_causal: bool | None,
    output_length: int | None,
    loss_impl: str = "full",
    logit_chunk_size: int = 1024,
    bd3_block_len: int | None = None,
    loss_normalizer: float | Array | None = None,
    moe_load_balance_loss_weight: float = 0.0,
    moe_router_z_loss_weight: float = 0.0,
):
    """Data-parallel supervised step with sum-form loss reduction.

    Aggregates per-shard gradient sums via ``psum`` and divides by the
    requested global denominator. If ``loss_normalizer`` is not supplied,
    this falls back to the global supervised-token count.
    """
    local_count_for_denom = _supervised_valid_count(
        targets,
        supervise_mask,
        output_length=output_length,
    )
    global_count_for_denom = jax.lax.psum(local_count_for_denom, DATA_AXIS)
    denominator = _loss_denominator(loss_normalizer, global_count_for_denom)
    (total_sum, metrics_sums), grads = nnx.value_and_grad(supervised_loss_fn, has_aux=True)(
        model,
        inputs,
        targets,
        supervise_mask,
        token_positions,
        attention_mask,
        z_loss_weight,
        is_causal,
        output_length,
        bd3_block_len,
        loss_impl,
        logit_chunk_size,
        denominator,
        DATA_AXIS,
        moe_load_balance_loss_weight,
        moe_router_z_loss_weight,
    )
    summed_grads = jax.lax.psum(grads, DATA_AXIS)
    global_loss_sum = jax.lax.psum(metrics_sums["loss_sum"], DATA_AXIS)
    global_z_loss_sum = jax.lax.psum(metrics_sums["z_loss_sum"], DATA_AXIS)
    global_count = jax.lax.psum(metrics_sums["valid_count"], DATA_AXIS)
    inv = _safe_inv(denominator)
    moe_aux_loss_sum = metrics_sums["moe_aux_loss_sum"]
    global_total_sum = global_loss_sum + z_loss_weight * global_z_loss_sum + moe_aux_loss_sum
    grads = _scale_grads(summed_grads, inv)
    clipped_grads, grad_norm = clip_by_global_norm(grads, max_grad_norm)
    optimizer.update(model, clipped_grads)
    out = {
        "loss": global_loss_sum * inv,
        "z_loss": global_z_loss_sum * inv,
        "total_loss": global_total_sum * inv,
        "supervised_tokens": global_count,
        "loss_normalizer": denominator,
        "grad_norm": grad_norm,
    }
    moe_metrics = _init_moe_metric_sums(metrics_sums)
    for key, value in list(moe_metrics.items()):
        if key.startswith("moe_") and not key.endswith("_sum"):
            moe_metrics[key] = jax.lax.pmean(value, DATA_AXIS)
    _add_moe_step_metrics(out, moe_metrics, inv)
    return out


def train_step_supervised_data_parallel(
    model,
    optimizer,
    inputs: Array,
    targets: Array,
    supervise_mask: Array,
    token_positions: Array | None,
    attention_mask: Array | None,
    z_loss_weight: float,
    max_grad_norm: float,
    is_causal: bool | None,
    output_length: int | None,
    loss_impl: str = "full",
    logit_chunk_size: int = 1024,
    bd3_block_len: int | None = None,
    loss_normalizer: float | Array | None = None,
    moe_load_balance_loss_weight: float = 0.0,
    moe_router_z_loss_weight: float = 0.0,
):
    return _train_step_supervised_data_parallel_pmapped(
        model,
        optimizer,
        inputs,
        targets,
        supervise_mask,
        token_positions,
        attention_mask,
        z_loss_weight,
        max_grad_norm,
        is_causal,
        output_length,
        loss_impl,
        logit_chunk_size,
        bd3_block_len,
        loss_normalizer,
        moe_load_balance_loss_weight,
        moe_router_z_loss_weight,
    )


@functools.partial(nnx.jit, static_argnums=(9, 10, 11, 12, 13))
def train_step_supervised_accumulated(
    model,
    optimizer,
    inputs: Array,
    targets: Array,
    supervise_mask: Array,
    token_positions: Array | None,
    attention_mask: Array | None,
    z_loss_weight: float,
    max_grad_norm: float,
    is_causal: bool | None,
    output_length: int | None,
    loss_impl: str = "full",
    logit_chunk_size: int = 1024,
    bd3_block_len: int | None = None,
    loss_normalizer: float | Array | None = None,
    moe_load_balance_loss_weight: float = 0.0,
    moe_router_z_loss_weight: float = 0.0,
):
    """Accumulated supervised update with sum-form loss reduction.

    Sums per-microbatch gradient sums and supervised-token counts across
    accum microsteps, then divides by ``loss_normalizer`` or the accumulated
    supervised-token count.
    """
    accum_steps = inputs.shape[0]
    count_for_denom = jnp.zeros((), dtype=jnp.float32)
    for i in range(accum_steps):
        count_for_denom = count_for_denom + _supervised_valid_count(
            targets[i],
            supervise_mask[i],
            output_length=output_length,
        )
    denominator = _loss_denominator(loss_normalizer, count_for_denom)
    micro_aux_denominator = denominator / float(accum_steps)
    (total_sum, metrics_sums), grads = nnx.value_and_grad(supervised_loss_fn, has_aux=True)(
        model,
        inputs[0],
        targets[0],
        supervise_mask[0],
        token_positions,
        attention_mask,
        z_loss_weight,
        is_causal,
        output_length,
        bd3_block_len,
        loss_impl,
        logit_chunk_size,
        micro_aux_denominator,
        None,
        moe_load_balance_loss_weight,
        moe_router_z_loss_weight,
    )
    summed_grads = grads
    summed_loss_sum = metrics_sums["loss_sum"]
    summed_z_loss_sum = metrics_sums["z_loss_sum"]
    summed_count = metrics_sums["valid_count"]
    summed_total_sum = total_sum
    summed_moe_metrics = _init_moe_metric_sums(metrics_sums)

    for i in range(1, accum_steps):
        (total_sum_i, metrics_i), grads_i = nnx.value_and_grad(supervised_loss_fn, has_aux=True)(
            model,
            inputs[i],
            targets[i],
            supervise_mask[i],
            token_positions,
            attention_mask,
            z_loss_weight,
            is_causal,
            output_length,
            bd3_block_len,
            loss_impl,
            logit_chunk_size,
            micro_aux_denominator,
            None,
            moe_load_balance_loss_weight,
            moe_router_z_loss_weight,
        )
        summed_grads = jax.tree_util.tree_map(lambda a, b: a + b, summed_grads, grads_i)
        summed_loss_sum = summed_loss_sum + metrics_i["loss_sum"]
        summed_z_loss_sum = summed_z_loss_sum + metrics_i["z_loss_sum"]
        summed_count = summed_count + metrics_i["valid_count"]
        summed_total_sum = summed_total_sum + total_sum_i
        _accumulate_moe_metric_sums(summed_moe_metrics, metrics_i)

    inv = _safe_inv(denominator)
    grads = _scale_grads(summed_grads, inv)
    clipped_grads, grad_norm = clip_by_global_norm(grads, max_grad_norm)
    optimizer.update(model, clipped_grads)
    out = {
        "loss": summed_loss_sum * inv,
        "z_loss": summed_z_loss_sum * inv,
        "total_loss": summed_total_sum * inv,
        "supervised_tokens": summed_count,
        "loss_normalizer": denominator,
        "grad_norm": grad_norm,
    }
    _add_moe_step_metrics(
        out,
        _finalize_accumulated_moe_metrics(summed_moe_metrics, accum_steps),
        inv,
    )
    return out


@functools.partial(
    nnx.pmap,
    axis_name=DATA_AXIS,
    in_axes=(BROADCAST_GRAPH_AXES, BROADCAST_GRAPH_AXES, 0, 0, 0, None, None, None, None, None, None, None, None, None, None, None, None),
    static_broadcasted_argnums=(9, 10, 11, 12, 13),
)
def _train_step_supervised_accumulated_data_parallel_pmapped(
    model,
    optimizer,
    inputs: Array,
    targets: Array,
    supervise_mask: Array,
    token_positions: Array | None,
    attention_mask: Array | None,
    z_loss_weight: float,
    max_grad_norm: float,
    is_causal: bool | None,
    output_length: int | None,
    loss_impl: str = "full",
    logit_chunk_size: int = 1024,
    bd3_block_len: int | None = None,
    loss_normalizer: float | Array | None = None,
    moe_load_balance_loss_weight: float = 0.0,
    moe_router_z_loss_weight: float = 0.0,
):
    """Data-parallel supervised step with local gradient accumulation.

    Sums grads and supervised-token counts locally over accum microsteps,
    then ``psum``s across devices, then divides by ``loss_normalizer`` or
    the global accumulated supervised-token count.
    """
    accum_steps = inputs.shape[0]
    local_count_for_denom = jnp.zeros((), dtype=jnp.float32)
    for i in range(accum_steps):
        local_count_for_denom = local_count_for_denom + _supervised_valid_count(
            targets[i],
            supervise_mask[i],
            output_length=output_length,
        )
    global_count_for_denom = jax.lax.psum(local_count_for_denom, DATA_AXIS)
    denominator = _loss_denominator(loss_normalizer, global_count_for_denom)
    micro_aux_denominator = denominator / float(accum_steps)
    (total_sum, metrics_sums), grads = nnx.value_and_grad(supervised_loss_fn, has_aux=True)(
        model,
        inputs[0],
        targets[0],
        supervise_mask[0],
        token_positions,
        attention_mask,
        z_loss_weight,
        is_causal,
        output_length,
        bd3_block_len,
        loss_impl,
        logit_chunk_size,
        micro_aux_denominator,
        DATA_AXIS,
        moe_load_balance_loss_weight,
        moe_router_z_loss_weight,
    )
    summed_grads = grads
    summed_loss_sum = metrics_sums["loss_sum"]
    summed_z_loss_sum = metrics_sums["z_loss_sum"]
    summed_count = metrics_sums["valid_count"]
    summed_total_sum = total_sum
    summed_moe_metrics = _init_moe_metric_sums(metrics_sums)

    for i in range(1, accum_steps):
        (total_sum_i, metrics_i), grads_i = nnx.value_and_grad(supervised_loss_fn, has_aux=True)(
            model,
            inputs[i],
            targets[i],
            supervise_mask[i],
            token_positions,
            attention_mask,
            z_loss_weight,
            is_causal,
            output_length,
            bd3_block_len,
            loss_impl,
            logit_chunk_size,
            micro_aux_denominator,
            DATA_AXIS,
            moe_load_balance_loss_weight,
            moe_router_z_loss_weight,
        )
        summed_grads = jax.tree_util.tree_map(lambda a, b: a + b, summed_grads, grads_i)
        summed_loss_sum = summed_loss_sum + metrics_i["loss_sum"]
        summed_z_loss_sum = summed_z_loss_sum + metrics_i["z_loss_sum"]
        summed_count = summed_count + metrics_i["valid_count"]
        summed_total_sum = summed_total_sum + total_sum_i
        _accumulate_moe_metric_sums(summed_moe_metrics, metrics_i)

    global_grads = jax.lax.psum(summed_grads, DATA_AXIS)
    global_loss_sum = jax.lax.psum(summed_loss_sum, DATA_AXIS)
    global_z_loss_sum = jax.lax.psum(summed_z_loss_sum, DATA_AXIS)
    global_count = jax.lax.psum(summed_count, DATA_AXIS)
    inv = _safe_inv(denominator)
    moe_metrics = _finalize_accumulated_moe_metrics(summed_moe_metrics, accum_steps)
    for key, value in list(moe_metrics.items()):
        if key.startswith("moe_") and not key.endswith("_sum"):
            moe_metrics[key] = jax.lax.pmean(value, DATA_AXIS)
    global_total_sum = (
        global_loss_sum
        + z_loss_weight * global_z_loss_sum
        + moe_metrics["moe_aux_loss_sum"]
    )
    grads = _scale_grads(global_grads, inv)
    clipped_grads, grad_norm = clip_by_global_norm(grads, max_grad_norm)
    optimizer.update(model, clipped_grads)
    out = {
        "loss": global_loss_sum * inv,
        "z_loss": global_z_loss_sum * inv,
        "total_loss": global_total_sum * inv,
        "supervised_tokens": global_count,
        "loss_normalizer": denominator,
        "grad_norm": grad_norm,
    }
    _add_moe_step_metrics(out, moe_metrics, inv)
    return out


def train_step_supervised_accumulated_data_parallel(
    model,
    optimizer,
    inputs: Array,
    targets: Array,
    supervise_mask: Array,
    token_positions: Array | None,
    attention_mask: Array | None,
    z_loss_weight: float,
    max_grad_norm: float,
    is_causal: bool | None,
    output_length: int | None,
    loss_impl: str = "full",
    logit_chunk_size: int = 1024,
    bd3_block_len: int | None = None,
    loss_normalizer: float | Array | None = None,
    moe_load_balance_loss_weight: float = 0.0,
    moe_router_z_loss_weight: float = 0.0,
):
    return _train_step_supervised_accumulated_data_parallel_pmapped(
        model,
        optimizer,
        inputs,
        targets,
        supervise_mask,
        token_positions,
        attention_mask,
        z_loss_weight,
        max_grad_norm,
        is_causal,
        output_length,
        loss_impl,
        logit_chunk_size,
        bd3_block_len,
        loss_normalizer,
        moe_load_balance_loss_weight,
        moe_router_z_loss_weight,
    )


@functools.partial(nnx.jit, static_argnums=(3, 4))
def eval_step(model, inputs: Array, targets: Array, loss_impl: str = "full", logit_chunk_size: int = 1024):
    if loss_impl == "full":
        logits = model(inputs)
        loss, z_loss, _ = cross_entropy_with_z_loss(logits, targets)
    elif loss_impl == "chunked":
        _, metrics = ar_loss(
            model,
            inputs,
            targets,
            z_loss_weight=0.0,
            loss_impl=loss_impl,
            logit_chunk_size=logit_chunk_size,
        )
        loss = metrics["loss"]
        z_loss = metrics["z_loss"]
    else:
        raise ValueError("loss_impl must be one of 'full' or 'chunked'")
    return {"loss": loss, "z_loss": z_loss}


@functools.partial(nnx.jit, static_argnums=(6, 7, 8, 9, 10))
def eval_step_supervised(
    model,
    inputs: Array,
    targets: Array,
    supervise_mask: Array,
    token_positions: Array | None,
    attention_mask: Array | None,
    is_causal: bool | None,
    output_length: int | None,
    loss_impl: str = "full",
    logit_chunk_size: int = 1024,
    bd3_block_len: int | None = None,
    loss_normalizer: float | Array | None = None,
):
    _, metrics = supervised_lm_loss_sums(
        model,
        inputs,
        targets,
        supervise_mask,
        token_positions=token_positions,
        attention_mask=attention_mask,
        is_causal=is_causal,
        output_length=output_length,
        bd3_block_len=bd3_block_len,
        z_loss_weight=0.0,
        loss_impl=loss_impl,
        logit_chunk_size=logit_chunk_size,
    )
    denominator = _loss_denominator(loss_normalizer, metrics["valid_count"])
    inv = _safe_inv(denominator)
    return {
        "loss": metrics["loss_sum"] * inv,
        "z_loss": metrics["z_loss_sum"] * inv,
        "supervised_tokens": metrics["valid_count"],
        "loss_normalizer": denominator,
    }
