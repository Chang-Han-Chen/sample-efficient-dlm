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
    valid_mask: Array | None = None,
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
        valid_mask=valid_mask,
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


def _chunk_inputs(
    hidden: Array,
    targets: Array,
    valid_mask: Array | None,
    chunk_size: int,
) -> tuple[Array, Array, Array]:
    flat_hidden = hidden.reshape((-1, hidden.shape[-1]))
    flat_targets = targets.reshape((-1,))
    if valid_mask is None:
        flat_valid = jnp.ones(flat_targets.shape, dtype=bool)
    else:
        flat_valid = valid_mask.reshape((-1,)).astype(bool)
    n_tokens = flat_targets.shape[0]
    n_chunks = int(math.ceil(n_tokens / int(chunk_size)))
    padded_tokens = n_chunks * int(chunk_size)
    pad_tokens = padded_tokens - n_tokens
    hidden_pad = jnp.pad(flat_hidden, ((0, pad_tokens), (0, 0)))
    targets_pad = jnp.pad(flat_targets, (0, pad_tokens))
    valid_pad = jnp.pad(flat_valid, (0, pad_tokens), constant_values=False)
    return (
        hidden_pad.reshape((n_chunks, int(chunk_size), hidden.shape[-1])),
        targets_pad.reshape((n_chunks, int(chunk_size))),
        valid_pad.reshape((n_chunks, int(chunk_size))),
    )


def _chunked_linear_ce_forward_impl(
    hidden: Array,
    weight: Array,
    targets: Array,
    valid_mask: Array | None,
    chunk_size: int,
    ignore_index: int | None,
) -> tuple[Array, Array, Array]:
    hidden_chunks, target_chunks, valid_chunks = _chunk_inputs(
        hidden,
        targets,
        valid_mask,
        chunk_size,
    )

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


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def _chunked_linear_ce_with_z_loss(
    hidden: Array,
    weight: Array,
    targets: Array,
    valid_mask: Array | None,
    chunk_size: int,
    ignore_index: int | None,
) -> tuple[Array, Array]:
    loss, z_loss, _ = _chunked_linear_ce_forward_impl(
        hidden,
        weight,
        targets,
        valid_mask,
        chunk_size,
        ignore_index,
    )
    return loss, z_loss


def _chunked_linear_ce_fwd(hidden, weight, targets, valid_mask, chunk_size, ignore_index):
    loss, z_loss, valid_count = _chunked_linear_ce_forward_impl(
        hidden,
        weight,
        targets,
        valid_mask,
        chunk_size,
        ignore_index,
    )
    return (loss, z_loss), (hidden, weight, targets, valid_mask, valid_count)


def _chunked_linear_ce_bwd(chunk_size, ignore_index, residuals, cotangents):
    hidden, weight, targets, valid_mask, valid_count = residuals
    loss_bar, z_loss_bar = cotangents
    hidden_chunks, target_chunks, valid_chunks = _chunk_inputs(
        hidden,
        targets,
        valid_mask,
        chunk_size,
    )
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
    return grad_hidden, grad_weight, None, None


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
    valid_mask: Array | None = None,
) -> tuple[Array, Array]:
    """Cross entropy for ``hidden @ weight.T`` without full-logit materialization.

    The custom VJP recomputes each chunk's logits during the backward pass and
    accumulates gradients into hidden states and the LM-head weight.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if valid_mask is None:
        valid_mask = jnp.ones(targets.shape, dtype=bool)
    return _chunked_linear_ce_with_z_loss(
        hidden_states,
        weight,
        targets,
        valid_mask,
        int(chunk_size),
        ignore_index,
    )


def supervised_lm_loss(
    model,
    inputs: Array,
    targets: Array,
    supervise_mask: Array | None = None,
    *,
    token_positions: Array | None = None,
    attention_mask: Array | None = None,
    is_causal: bool | None = None,
    output_length: int | None = None,
    bd3_block_len: int | None = None,
    z_loss_weight: float = 1e-4,
    ignore_index: int | None = None,
    loss_impl: str = "full",
    logit_chunk_size: int = 1024,
) -> tuple[Array, dict[str, Array]]:
    """Generic supervised CE over selected token positions.

    This covers AR, MDLM, and BD3LM:
      * AR passes a full-True supervision mask and causal attention.
      * MDLM passes the random masked-token positions and no causal mask.
      * BD3LM can feed a dual stream and set ``output_length`` to the clean
        sequence length so only noisy-stream outputs are scored.
    """
    if output_length is not None:
        targets = targets[:, :output_length]
        if supervise_mask is not None:
            supervise_mask = supervise_mask[:, :output_length]

    if loss_impl == "full":
        if output_length is not None:
            hidden = model(
                inputs,
                token_positions=token_positions,
                attention_mask=attention_mask,
                is_causal=is_causal,
                return_hidden=True,
                bd3_block_len=bd3_block_len,
            )[:, :output_length]
            logits = model.project_logits(hidden)
        else:
            logits = model(
                inputs,
                token_positions=token_positions,
                attention_mask=attention_mask,
                is_causal=is_causal,
                bd3_block_len=bd3_block_len,
            )
        loss, z_loss, _ = cross_entropy_with_z_loss(
            logits,
            targets,
            ignore_index=ignore_index,
            valid_mask=supervise_mask,
        )
    elif loss_impl == "chunked":
        hidden = model(
            inputs,
            token_positions=token_positions,
            attention_mask=attention_mask,
            is_causal=is_causal,
            return_hidden=True,
            bd3_block_len=bd3_block_len,
        )
        if output_length is not None:
            hidden = hidden[:, :output_length]
        loss, z_loss = linear_cross_entropy_with_z_loss_chunked(
            hidden,
            model.lm_head.weight.value,
            targets,
            chunk_size=logit_chunk_size,
            ignore_index=ignore_index,
            valid_mask=supervise_mask,
        )
    else:
        raise ValueError("loss_impl must be one of 'full' or 'chunked'")

    total = loss + z_loss_weight * z_loss
    metrics = {
        "loss": loss,
        "z_loss": z_loss,
        "total_loss": total,
    }
    if supervise_mask is not None:
        metrics["supervised_tokens"] = jnp.sum(supervise_mask.astype(jnp.float32))
    return total, metrics


def supervised_lm_loss_sums(
    model,
    inputs: Array,
    targets: Array,
    supervise_mask: Array | None = None,
    *,
    token_positions: Array | None = None,
    attention_mask: Array | None = None,
    is_causal: bool | None = None,
    output_length: int | None = None,
    bd3_block_len: int | None = None,
    z_loss_weight: float = 1e-4,
    ignore_index: int | None = None,
    loss_impl: str = "full",
    logit_chunk_size: int = 1024,
) -> tuple[Array, dict[str, Array]]:
    """Sum/count variant of ``supervised_lm_loss``.

    Returns ``(total_sum, metrics_sums)`` where ``total_sum`` is the sum of
    CE + ``z_loss_weight * z_loss`` aggregated over all supervised tokens in
    this microbatch (a sum, not a mean). The metrics dict contains the
    sum-form quantities plus the supervised-token count, so each step type
    can do its own correct reduction:

    * single device: divide grads by ``valid_count``.
    * accumulation: sum sums and counts across microsteps, then divide.
    * DP: ``psum`` sums and counts across devices, then divide.
    * DP + accumulation: sum locally, then ``psum``, then divide.

    Differentiating ``total_sum`` w.r.t. params yields ``valid_count *
    grad(total_mean)`` because ``valid_count`` is mask-derived (no param
    dependency), and the inner mean loss divides by ``max(valid_count, 1)``.
    For ``valid_count > 0`` this is exactly the gradient of the true
    sum-form loss; for ``valid_count == 0`` both quantities are zero.
    """
    if output_length is not None and supervise_mask is not None:
        mask_for_count = supervise_mask[:, :output_length]
    else:
        mask_for_count = supervise_mask
    target_slice = targets[:, :output_length] if output_length is not None else targets
    if mask_for_count is None:
        if ignore_index is None:
            valid_count = jnp.asarray(target_slice.size, dtype=jnp.float32)
        else:
            valid_count = jnp.sum((target_slice != ignore_index).astype(jnp.float32))
    else:
        valid_bool = mask_for_count.astype(bool)
        if ignore_index is not None:
            valid_bool = valid_bool & (target_slice != ignore_index)
        valid_count = jnp.sum(valid_bool.astype(jnp.float32))

    _, mean_metrics = supervised_lm_loss(
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
        ignore_index=ignore_index,
        loss_impl=loss_impl,
        logit_chunk_size=logit_chunk_size,
    )
    # Mean is loss_sum_internal / max(count, 1). Multiplying by count yields
    # loss_sum_internal exactly when count > 0 and 0 when count == 0.
    loss_sum = mean_metrics["loss"] * valid_count
    z_loss_sum = mean_metrics["z_loss"] * valid_count
    total_sum = loss_sum + z_loss_weight * z_loss_sum
    return total_sum, {
        "loss_sum": loss_sum,
        "z_loss_sum": z_loss_sum,
        "valid_count": valid_count,
    }


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
