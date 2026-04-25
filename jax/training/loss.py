"""Autoregressive language-modeling losses in pure JAX."""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp

Array = jax.Array


def _cross_entropy_sums(
    logits: Array,
    targets: Array,
    *,
    ignore_index: int | None = None,
    valid_mask: Array | None = None,
    per_token_byte_lengths: Array | None = None,
) -> tuple[Array, Array, Array, Array | None, Array | None]:
    logits_f = logits.astype(jnp.float32)
    targets = targets.astype(jnp.int32)

    if valid_mask is None:
        valid = jnp.ones(targets.shape, dtype=bool)
    else:
        valid = valid_mask.astype(bool)
    if ignore_index is not None:
        valid = valid & (targets != ignore_index)
    gather_targets = jnp.where(valid, targets, 0)

    lse = jax.nn.logsumexp(logits_f, axis=-1)
    target_logits = jnp.take_along_axis(
        logits_f,
        gather_targets[..., None],
        axis=-1,
    )[..., 0]
    per_token_loss = lse - target_logits
    valid_f = valid.astype(jnp.float32)

    loss_sum = jnp.sum(per_token_loss * valid_f)
    z_loss_sum = jnp.sum((lse**2) * valid_f)
    valid_count = jnp.sum(valid_f)

    if per_token_byte_lengths is None:
        bpb_numer = None
        byte_denom = None
    else:
        byte_len = per_token_byte_lengths[gather_targets].astype(jnp.float32) * valid_f
        byte_denom = jnp.sum(byte_len)
        bpb_numer = jnp.sum(jax.lax.stop_gradient(per_token_loss) * byte_len)

    return loss_sum, z_loss_sum, valid_count, bpb_numer, byte_denom


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
    loss_sum, z_loss_sum, valid_count, bpb_numer, byte_denom = _cross_entropy_sums(
        logits,
        targets,
        ignore_index=ignore_index,
        per_token_byte_lengths=per_token_byte_lengths,
    )
    denom = jnp.maximum(valid_count, 1.0)
    loss = loss_sum / denom
    z_loss = z_loss_sum / denom

    if per_token_byte_lengths is None:
        loss_bpb = None
    else:
        nats_per_byte = bpb_numer / jnp.maximum(byte_denom, 1.0)
        loss_bpb = nats_per_byte / jnp.log(2.0)
    return loss, z_loss, loss_bpb


def _chunk_inputs(hidden: Array, targets: Array, chunk_size: int) -> tuple[Array, Array, Array]:
    flat_hidden = hidden.reshape((-1, hidden.shape[-1]))
    flat_targets = targets.reshape((-1,))
    n_tokens = flat_targets.shape[0]
    n_chunks = int(math.ceil(n_tokens / int(chunk_size)))
    padded_tokens = n_chunks * int(chunk_size)
    pad_tokens = padded_tokens - n_tokens
    hidden_pad = jnp.pad(flat_hidden, ((0, pad_tokens), (0, 0)))
    targets_pad = jnp.pad(flat_targets, (0, pad_tokens))
    valid_pad = jnp.arange(padded_tokens) < n_tokens
    return (
        hidden_pad.reshape((n_chunks, int(chunk_size), hidden.shape[-1])),
        targets_pad.reshape((n_chunks, int(chunk_size))),
        valid_pad.reshape((n_chunks, int(chunk_size))),
    )


def _chunked_linear_ce_forward_impl(
    hidden: Array,
    weight: Array,
    targets: Array,
    chunk_size: int,
    ignore_index: int | None,
) -> tuple[Array, Array, Array]:
    hidden_chunks, target_chunks, valid_chunks = _chunk_inputs(hidden, targets, chunk_size)

    def body(carry, xs):
        loss_sum, z_loss_sum, valid_count = carry
        h_chunk, t_chunk, valid_chunk = xs
        logits = h_chunk @ weight.T
        chunk_loss, chunk_z, chunk_count, _, _ = _cross_entropy_sums(
            logits,
            t_chunk,
            ignore_index=ignore_index,
            valid_mask=valid_chunk,
        )
        return (
            loss_sum + chunk_loss,
            z_loss_sum + chunk_z,
            valid_count + chunk_count,
        ), None

    init = (
        jnp.zeros((), dtype=jnp.float32),
        jnp.zeros((), dtype=jnp.float32),
        jnp.zeros((), dtype=jnp.float32),
    )
    (loss_sum, z_loss_sum, valid_count), _ = jax.lax.scan(
        body,
        init,
        (hidden_chunks, target_chunks, valid_chunks),
    )
    denom = jnp.maximum(valid_count, 1.0)
    return loss_sum / denom, z_loss_sum / denom, valid_count


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _chunked_linear_ce_with_z_loss(
    hidden: Array,
    weight: Array,
    targets: Array,
    chunk_size: int,
    ignore_index: int | None,
) -> tuple[Array, Array]:
    loss, z_loss, _ = _chunked_linear_ce_forward_impl(
        hidden,
        weight,
        targets,
        chunk_size,
        ignore_index,
    )
    return loss, z_loss


def _chunked_linear_ce_fwd(hidden, weight, targets, chunk_size, ignore_index):
    loss, z_loss, valid_count = _chunked_linear_ce_forward_impl(
        hidden,
        weight,
        targets,
        chunk_size,
        ignore_index,
    )
    return (loss, z_loss), (hidden, weight, targets, valid_count)


def _chunked_linear_ce_bwd(chunk_size, ignore_index, residuals, cotangents):
    hidden, weight, targets, valid_count = residuals
    loss_bar, z_loss_bar = cotangents
    hidden_chunks, target_chunks, valid_chunks = _chunk_inputs(hidden, targets, chunk_size)
    denom = jnp.maximum(valid_count, 1.0)
    vocab_size = weight.shape[0]

    def body(carry, xs):
        grad_weight = carry
        h_chunk, t_chunk, valid_chunk = xs
        logits = (h_chunk @ weight.T).astype(jnp.float32)
        valid = valid_chunk.astype(bool)
        if ignore_index is not None:
            valid = valid & (t_chunk != ignore_index)
        valid_f = valid.astype(jnp.float32)
        gather_targets = jnp.where(valid, t_chunk.astype(jnp.int32), 0)

        lse = jax.nn.logsumexp(logits, axis=-1)
        probs = jax.nn.softmax(logits, axis=-1)
        grad_logits = (
            (loss_bar / denom) * probs
            + (z_loss_bar / denom) * (2.0 * lse[..., None] * probs)
        )
        grad_logits = grad_logits * valid_f[..., None]
        grad_logits = grad_logits.at[jnp.arange(int(chunk_size)), gather_targets].add(
            -(loss_bar / denom) * valid_f
        )

        grad_hidden = grad_logits @ weight.astype(jnp.float32)
        grad_weight = grad_weight + grad_logits.T @ h_chunk.astype(jnp.float32)
        return grad_weight, grad_hidden

    init_grad_weight = jnp.zeros((vocab_size, hidden.shape[-1]), dtype=jnp.float32)
    grad_weight, grad_hidden_chunks = jax.lax.scan(
        body,
        init_grad_weight,
        (hidden_chunks, target_chunks, valid_chunks),
    )
    flat_grad_hidden = grad_hidden_chunks.reshape((-1, hidden.shape[-1]))[: targets.size]
    grad_hidden = flat_grad_hidden.reshape(hidden.shape).astype(hidden.dtype)
    grad_weight = grad_weight.astype(weight.dtype)
    return grad_hidden, grad_weight, None


_chunked_linear_ce_with_z_loss.defvjp(
    _chunked_linear_ce_fwd,
    _chunked_linear_ce_bwd,
)


def linear_cross_entropy_with_z_loss_chunked(
    hidden_states: Array,
    weight: Array,
    targets: Array,
    *,
    chunk_size: int,
    ignore_index: int | None = None,
) -> tuple[Array, Array]:
    """Cross entropy for ``hidden @ weight.T`` without full-logit materialization.

    The custom VJP recomputes each chunk's logits during the backward pass and
    accumulates gradients into hidden states and the LM-head weight.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return _chunked_linear_ce_with_z_loss(
        hidden_states,
        weight,
        targets,
        int(chunk_size),
        ignore_index,
    )


def ar_loss(
    model,
    inputs: Array,
    targets: Array,
    *,
    z_loss_weight: float = 1e-4,
    ignore_index: int | None = None,
    per_token_byte_lengths: Array | None = None,
    loss_impl: str = "full",
    logit_chunk_size: int = 1024,
) -> tuple[Array, dict[str, Array]]:
    if loss_impl == "full":
        logits = model(inputs)
        loss, z_loss, loss_bpb = cross_entropy_with_z_loss(
            logits,
            targets,
            ignore_index=ignore_index,
            per_token_byte_lengths=per_token_byte_lengths,
        )
    elif loss_impl == "chunked":
        if per_token_byte_lengths is not None:
            raise NotImplementedError("BPB reporting is not implemented for chunked loss yet")
        hidden = model(inputs, return_hidden=True)
        loss, z_loss = linear_cross_entropy_with_z_loss_chunked(
            hidden,
            model.lm_head.weight.value,
            targets,
            chunk_size=logit_chunk_size,
            ignore_index=ignore_index,
        )
        loss_bpb = None
    else:
        raise ValueError("loss_impl must be one of 'full' or 'chunked'")
    total = loss + z_loss_weight * z_loss
    metrics = {
        "loss": loss,
        "z_loss": z_loss,
        "total_loss": total,
    }
    if loss_bpb is not None:
        metrics["loss_bpb"] = loss_bpb
    return total, metrics
