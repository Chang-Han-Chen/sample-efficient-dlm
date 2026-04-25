"""
model_bd3lm.py — Block Discrete Denoising Diffusion Language Model (BD3-LM).

A self-contained block diffusion model using MDLM-style progressive unmasking.

Training follows the BD3-LM pattern:
  - sample one timestep per block
  - corrupt all blocks independently
  - run a single dual-stream transformer pass on x_t ⊕ x_0

Sampling is block-autoregressive:
  - previous blocks are frozen
  - diffusion (progressive unmasking) happens only inside the current block
"""

import math
from typing import Optional, Tuple

import torch
from torch.nn import functional as F

from backbone import DiffusionBackbone


# ------------------------------------------------------------------
# Block-diffusion helpers (inlined from block_utils.py)
# ------------------------------------------------------------------


def _validate_block_len(block_size: int, block_len: int) -> None:
    if block_len <= 0:
        raise ValueError(f"block_len must be positive, got {block_len}")
    if block_size % block_len != 0:
        raise ValueError(
            f"block_len={block_len} must divide block_size={block_size}"
        )


def _num_blocks(block_size: int, block_len: int) -> int:
    _validate_block_len(block_size, block_len)
    return block_size // block_len


def _expand_block_values(
    values: torch.Tensor, block_len: int, *, seq_len: Optional[int] = None
) -> torch.Tensor:
    """Repeat per-block values to per-token."""
    out = values.repeat_interleave(block_len, dim=1)
    if seq_len is not None:
        out = out[:, :seq_len]
    return out


# ------------------------------------------------------------------
# Batch construction
# ------------------------------------------------------------------


def _sample_data_chunk(split: str, cfg) -> torch.Tensor:
    data_split = cfg["train_data"] if split == "train" else cfg["val_data"]
    B, L = cfg["batch_size"], cfg["block_size"]
    idx = torch.randint(len(data_split) - L, (B,))
    return torch.stack([data_split[i : i + L] for i in idx])


def _sample_block_timesteps(
    batch_size: int,
    block_size: int,
    block_len: int,
    T: int,
    device,
    fixed_t_step: Optional[int] = None,
) -> torch.Tensor:
    nblk = _num_blocks(block_size, block_len)
    if fixed_t_step is None:
        return torch.randint(1, T + 1, (batch_size, nblk), device=device)
    fixed_t_step = int(max(1, min(T, fixed_t_step)))
    return torch.full(
        (batch_size, nblk), fixed_t_step, device=device, dtype=torch.long
    )


def _make_block_noisy_batch(
    x0: torch.Tensor,
    cfg,
    fixed_t_step: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """BD3-style blockwise masking: one timestep per block."""
    B, L = x0.shape
    device = x0.device
    block_len = cfg["block_len"]

    t_blocks = _sample_block_timesteps(
        B, L, block_len, cfg["T"], device=device, fixed_t_step=fixed_t_step,
    )
    a_blocks = cfg["survival_prob_tensor"](t_blocks)
    a_tokens = _expand_block_values(a_blocks, block_len, seq_len=L)

    token_mask = torch.rand(B, L, device=device) > a_tokens
    xt = x0.clone()
    xt[token_mask] = cfg["mask_token_id"]
    return xt, x0, token_mask


def _get_block_batch(split: str, cfg, fixed_t_step: Optional[int] = None):
    x0 = _sample_data_chunk(split, cfg)
    xt, x0, token_mask = _make_block_noisy_batch(
        x0, cfg, fixed_t_step=fixed_t_step
    )
    dev = cfg["device"]
    return xt.to(dev), x0.to(dev), token_mask.to(dev)


# ------------------------------------------------------------------
# Generation helpers
# ------------------------------------------------------------------


def _current_block_range(block_idx: int, block_len: int) -> Tuple[int, int]:
    start = block_idx * block_len
    return start, start + block_len


def _prompt_start_block(prompt_mask: torch.Tensor, block_len: int) -> int:
    """First block not entirely inside the prompt."""
    prompt_len = int(prompt_mask[0].sum().item())
    return prompt_len // block_len


def _build_generation_prompt_mask(
    full_prompt_mask: torch.Tensor,
    block_start: int,
    block_end: int,
) -> torch.Tensor:
    """All blocks before the current one are frozen; current block uses original prompt mask."""
    out = torch.ones(
        full_prompt_mask.size(0),
        block_end,
        dtype=torch.bool,
        device=full_prompt_mask.device,
    )
    out[:, block_start:block_end] = full_prompt_mask[:, block_start:block_end]
    return out


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------


class Model(DiffusionBackbone):
    pass


# ------------------------------------------------------------------
# Training interface
# ------------------------------------------------------------------


def make_batch(x0, cfg):
    """Apply blockwise masking to clean tokens for training."""
    return _make_block_noisy_batch(x0, cfg)


def make_eval_batch(x0, cfg, fixed_t_step=None, **kwargs):
    """Apply blockwise masking with optional fixed timestep."""
    return _make_block_noisy_batch(x0, cfg, fixed_t_step=fixed_t_step)


# Legacy helper kept for older tests/utilities
def get_batch(split, cfg):
    return _get_block_batch(split, cfg)


def get_eval_batch(split, cfg, fixed_t_step):
    return _get_block_batch(split, cfg, fixed_t_step=fixed_t_step)


def compute_loss(model, batch, cfg):
    xt, x0, mask = batch
    _, loss = model.forward_train(xt, x0, targets=x0, supervise_mask=mask)
    return loss


def compute_eval_loss(model, xt, x0, mask):
    _, loss = model.forward_train(xt, x0, targets=x0, supervise_mask=mask)
    return loss


# ------------------------------------------------------------------
# Reverse diffusion sampling (progressive unmasking, block-autoregressive)
# ------------------------------------------------------------------


def _progressive_unmask_step(
    x, prompt_mask, sampled, sampled_conf, t, *, mask_token_id, survival_prob_scalar
):
    """Unmask the highest-confidence masked tokens for the next timestep."""
    current_mask = (x == mask_token_id) & (~prompt_mask)
    if not current_mask.any():
        return x

    if t == 1:
        return torch.where(current_mask, sampled, x)

    total_gen = (~prompt_mask).sum(dim=1)
    current_masked = current_mask.sum(dim=1)
    target_masked_next = torch.floor(
        (1.0 - survival_prob_scalar(t - 1)) * total_gen.float()
    ).long()
    target_masked_next = torch.minimum(target_masked_next, current_masked)
    num_to_unmask = (current_masked - target_masked_next).clamp_min(0)

    for b in range(x.size(0)):
        k = int(num_to_unmask[b].item())
        if k <= 0:
            continue
        conf_b = sampled_conf[b].masked_fill(~current_mask[b], float("-inf"))
        chosen = torch.topk(conf_b, k=k, dim=0, largest=True).indices
        x[b, chosen] = sampled[b, chosen]

    return x


@torch.no_grad()
def generate_from(
    model,
    x,
    prompt_mask,
    *,
    T,
    block_size,
    block_len,
    vocab_size,
    mask_token_id,
    survival_prob_scalar,
    decode,
):
    """
    Block-autoregressive sampler with MDLM-style progressive unmasking.

    Iterates over blocks left-to-right; within each block, runs T reverse
    diffusion steps where already-revealed tokens are carried forward and
    the highest-confidence masked positions are progressively unmasked.
    """
    model.eval()

    num_blk = block_size // block_len
    first_block = _prompt_start_block(prompt_mask, block_len)

    for block_idx in range(first_block, num_blk):
        block_start, block_end = _current_block_range(block_idx, block_len)
        x_view = x[:, :block_end].clone()
        frozen_mask = _build_generation_prompt_mask(
            prompt_mask, block_start, block_end
        )

        for t in reversed(range(1, T + 1)):
            logits, _ = model.forward_sample(x_view)
            probs = F.softmax(logits, dim=-1)
            sampled = torch.multinomial(
                probs.view(-1, vocab_size), 1
            ).view_as(x_view)
            sampled_conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

            x_view = _progressive_unmask_step(
                x_view,
                frozen_mask,
                sampled,
                sampled_conf,
                t,
                mask_token_id=mask_token_id,
                survival_prob_scalar=survival_prob_scalar,
            )

        # Final cleanup for any remaining masks in the current block
        if (x_view[:, block_start:block_end] == mask_token_id).any():
            logits, _ = model.forward_sample(x_view)
            final_tokens = torch.argmax(logits, dim=-1)
            x_view = torch.where(x_view == mask_token_id, final_tokens, x_view)

        x[:, :block_end] = x_view

    return decode(x[0].tolist())
