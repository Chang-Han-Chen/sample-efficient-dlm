"""Autoregressive language-modeling losses in pure JAX."""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jax.Array


def cross_entropy_with_z_loss(
    logits: Array,
    targets: Array,
    *,
    ignore_index: int | None = None,
    per_token_byte_lengths: Array | None = None,
) -> tuple[Array, Array, Array | None]:
    """Mean token CE plus PaLM-style z-loss statistic.

    This mirrors the PyTorch training loss numerically: compute logsumexp in
    fp32, average CE over valid targets, and report ``mean(logsumexp**2)`` as a
    separate z-loss term. ``per_token_byte_lengths`` is optional and only for
    BPB reporting.
    """
    logits_f = logits.astype(jnp.float32)
    targets = targets.astype(jnp.int32)

    if ignore_index is None:
        valid = jnp.ones(targets.shape, dtype=bool)
        gather_targets = targets
    else:
        valid = targets != ignore_index
        gather_targets = jnp.where(valid, targets, 0)

    lse = jax.nn.logsumexp(logits_f, axis=-1)
    target_logits = jnp.take_along_axis(
        logits_f,
        gather_targets[..., None],
        axis=-1,
    )[..., 0]
    per_token_loss = lse - target_logits
    valid_f = valid.astype(jnp.float32)
    denom = jnp.maximum(jnp.sum(valid_f), 1.0)

    loss = jnp.sum(per_token_loss * valid_f) / denom
    z_loss = jnp.sum((lse**2) * valid_f) / denom

    if per_token_byte_lengths is None:
        loss_bpb = None
    else:
        byte_len = per_token_byte_lengths[gather_targets].astype(jnp.float32) * valid_f
        byte_denom = jnp.maximum(jnp.sum(byte_len), 1.0)
        nats_per_byte = jnp.sum(jax.lax.stop_gradient(per_token_loss) * byte_len) / byte_denom
        loss_bpb = nats_per_byte / jnp.log(2.0)

    return loss, z_loss, loss_bpb


def ar_loss(
    model,
    inputs: Array,
    targets: Array,
    *,
    z_loss_weight: float = 1e-4,
    ignore_index: int | None = None,
    per_token_byte_lengths: Array | None = None,
) -> tuple[Array, dict[str, Array]]:
    logits = model(inputs)
    loss, z_loss, loss_bpb = cross_entropy_with_z_loss(
        logits,
        targets,
        ignore_index=ignore_index,
        per_token_byte_lengths=per_token_byte_lengths,
    )
    total = loss + z_loss_weight * z_loss
    metrics = {
        "loss": loss,
        "z_loss": z_loss,
        "total_loss": total,
    }
    if loss_bpb is not None:
        metrics["loss_bpb"] = loss_bpb
    return total, metrics
