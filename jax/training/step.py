"""Jitted AR train/eval steps."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from flax import nnx

from .loss import ar_loss, cross_entropy_with_z_loss
from .optimizer import clip_by_global_norm

Array = jax.Array


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
