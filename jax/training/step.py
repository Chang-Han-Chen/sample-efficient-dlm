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
):
    return ar_loss(
        model,
        inputs,
        targets,
        z_loss_weight=z_loss_weight,
        loss_impl=loss_impl,
        logit_chunk_size=logit_chunk_size,
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
):
    """Sum-form supervised loss used inside ``value_and_grad``.

    Returns ``(total_sum, metrics_sums)``. Each train step variant divides
    by the appropriate count (local, ``psum``, or accumulated) to recover
    the global masked-token mean. AR can use this same path because every
    shard has the same valid-token count, but it currently uses the
    cheaper ``loss_fn`` mean path.
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
    )


def _safe_inv(count: Array) -> Array:
    return jnp.where(count > 0, 1.0 / jnp.maximum(count, 1.0), 0.0)


def _scale_grads(grads, scale: Array):
    return jax.tree_util.tree_map(lambda g: g * scale, grads)


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
):
    (total_loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
        model,
        inputs,
        targets,
        z_loss_weight,
        loss_impl,
        logit_chunk_size,
    )
    clipped_grads, grad_norm = clip_by_global_norm(grads, max_grad_norm)
    optimizer.update(model, clipped_grads)
    metrics["total_loss"] = total_loss
    metrics["grad_norm"] = grad_norm
    return metrics


@functools.partial(
    nnx.pmap,
    axis_name=DATA_AXIS,
    in_axes=(BROADCAST_GRAPH_AXES, BROADCAST_GRAPH_AXES, 0, 0, None, None, None, None),
    static_broadcasted_argnums=(6, 7),
)
def train_step_data_parallel(
    model,
    optimizer,
    inputs: Array,
    targets: Array,
    z_loss_weight: float,
    max_grad_norm: float,
    loss_impl: str = "full",
    logit_chunk_size: int = 1024,
):
    """Single optimizer step over a pmapped, data-parallel microbatch."""
    (total_loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
        model,
        inputs,
        targets,
        z_loss_weight,
        loss_impl,
        logit_chunk_size,
    )
    mean_grads = jax.lax.pmean(grads, DATA_AXIS)
    clipped_grads, grad_norm = clip_by_global_norm(mean_grads, max_grad_norm)
    optimizer.update(model, clipped_grads)
    metrics = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, DATA_AXIS), metrics)
    metrics["total_loss"] = jax.lax.pmean(total_loss, DATA_AXIS)
    metrics["grad_norm"] = grad_norm
    return metrics


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
    in_axes=(BROADCAST_GRAPH_AXES, BROADCAST_GRAPH_AXES, 0, 0, None, None, None, None),
    static_broadcasted_argnums=(6, 7),
)
def train_step_accumulated_data_parallel(
    model,
    optimizer,
    inputs: Array,
    targets: Array,
    z_loss_weight: float,
    max_grad_norm: float,
    loss_impl: str = "full",
    logit_chunk_size: int = 1024,
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
):
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
    )
    valid_count = metrics_sums["valid_count"]
    inv = _safe_inv(valid_count)
    grads = _scale_grads(grads, inv)
    clipped_grads, grad_norm = clip_by_global_norm(grads, max_grad_norm)
    optimizer.update(model, clipped_grads)
    return {
        "loss": metrics_sums["loss_sum"] * inv,
        "z_loss": metrics_sums["z_loss_sum"] * inv,
        "total_loss": total_sum * inv,
        "supervised_tokens": valid_count,
        "grad_norm": grad_norm,
    }


@functools.partial(
    nnx.pmap,
    axis_name=DATA_AXIS,
    in_axes=(BROADCAST_GRAPH_AXES, BROADCAST_GRAPH_AXES, 0, 0, 0, None, None, None, None, None, None, None, None, None),
    static_broadcasted_argnums=(9, 10, 11, 12, 13),
)
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
):
    """Data-parallel supervised step with weighted-mean reduction.

    Aggregates per-shard gradient sums via ``psum`` and divides by the
    global supervised-token count. This matches the gradient of a single
    masked-token mean over the whole global batch even when shards have
    different supervised counts (uneven masks across devices).
    """
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
    )
    summed_grads = jax.lax.psum(grads, DATA_AXIS)
    global_loss_sum = jax.lax.psum(metrics_sums["loss_sum"], DATA_AXIS)
    global_z_loss_sum = jax.lax.psum(metrics_sums["z_loss_sum"], DATA_AXIS)
    global_total_sum = jax.lax.psum(total_sum, DATA_AXIS)
    global_count = jax.lax.psum(metrics_sums["valid_count"], DATA_AXIS)
    inv = _safe_inv(global_count)
    grads = _scale_grads(summed_grads, inv)
    clipped_grads, grad_norm = clip_by_global_norm(grads, max_grad_norm)
    optimizer.update(model, clipped_grads)
    return {
        "loss": global_loss_sum * inv,
        "z_loss": global_z_loss_sum * inv,
        "total_loss": global_total_sum * inv,
        "supervised_tokens": global_count,
        "grad_norm": grad_norm,
    }


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
):
    """Accumulated supervised update with weighted-mean reduction.

    Sums per-microbatch gradient sums and supervised-token counts across
    accum microsteps, then divides by the accumulated global count.
    """
    accum_steps = inputs.shape[0]
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
    )
    summed_grads = grads
    summed_loss_sum = metrics_sums["loss_sum"]
    summed_z_loss_sum = metrics_sums["z_loss_sum"]
    summed_count = metrics_sums["valid_count"]
    summed_total_sum = total_sum

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
        )
        summed_grads = jax.tree_util.tree_map(lambda a, b: a + b, summed_grads, grads_i)
        summed_loss_sum = summed_loss_sum + metrics_i["loss_sum"]
        summed_z_loss_sum = summed_z_loss_sum + metrics_i["z_loss_sum"]
        summed_count = summed_count + metrics_i["valid_count"]
        summed_total_sum = summed_total_sum + total_sum_i

    inv = _safe_inv(summed_count)
    grads = _scale_grads(summed_grads, inv)
    clipped_grads, grad_norm = clip_by_global_norm(grads, max_grad_norm)
    optimizer.update(model, clipped_grads)
    return {
        "loss": summed_loss_sum * inv,
        "z_loss": summed_z_loss_sum * inv,
        "total_loss": summed_total_sum * inv,
        "supervised_tokens": summed_count,
        "grad_norm": grad_norm,
    }


@functools.partial(
    nnx.pmap,
    axis_name=DATA_AXIS,
    in_axes=(BROADCAST_GRAPH_AXES, BROADCAST_GRAPH_AXES, 0, 0, 0, None, None, None, None, None, None, None, None, None),
    static_broadcasted_argnums=(9, 10, 11, 12, 13),
)
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
):
    """Data-parallel supervised step with local gradient accumulation.

    Sums grads and supervised-token counts locally over accum microsteps,
    then ``psum``s across devices, then divides by the global accumulated
    count. Equivalent to a single masked-token mean over the full
    effective batch (D * A microbatches), correct under uneven masks.
    """
    accum_steps = inputs.shape[0]
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
    )
    summed_grads = grads
    summed_loss_sum = metrics_sums["loss_sum"]
    summed_z_loss_sum = metrics_sums["z_loss_sum"]
    summed_count = metrics_sums["valid_count"]
    summed_total_sum = total_sum

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
        )
        summed_grads = jax.tree_util.tree_map(lambda a, b: a + b, summed_grads, grads_i)
        summed_loss_sum = summed_loss_sum + metrics_i["loss_sum"]
        summed_z_loss_sum = summed_z_loss_sum + metrics_i["z_loss_sum"]
        summed_count = summed_count + metrics_i["valid_count"]
        summed_total_sum = summed_total_sum + total_sum_i

    global_grads = jax.lax.psum(summed_grads, DATA_AXIS)
    global_loss_sum = jax.lax.psum(summed_loss_sum, DATA_AXIS)
    global_z_loss_sum = jax.lax.psum(summed_z_loss_sum, DATA_AXIS)
    global_count = jax.lax.psum(summed_count, DATA_AXIS)
    global_total_sum = jax.lax.psum(summed_total_sum, DATA_AXIS)
    inv = _safe_inv(global_count)
    grads = _scale_grads(global_grads, inv)
    clipped_grads, grad_norm = clip_by_global_norm(grads, max_grad_norm)
    optimizer.update(model, clipped_grads)
    return {
        "loss": global_loss_sum * inv,
        "z_loss": global_z_loss_sum * inv,
        "total_loss": global_total_sum * inv,
        "supervised_tokens": global_count,
        "grad_norm": grad_norm,
    }


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
):
    _, metrics = supervised_lm_loss(
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
    return {"loss": metrics["loss"], "z_loss": metrics["z_loss"]}
