"""
block_utils.py — BD3-LM attention masks and basic block helpers.

Used by backbone.py for constructing the dual-stream training mask
and the block-causal sampling mask.
"""

from functools import partial

import torch

try:
    from torch.nn.attention.flex_attention import create_block_mask

    FLEX_ATTN_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on local torch build
    create_block_mask = None
    FLEX_ATTN_AVAILABLE = False


# ------------------------------------------------------------------
# Basic helpers
# ------------------------------------------------------------------


def validate_block_len(block_size: int, block_len: int) -> None:
    if block_len <= 0:
        raise ValueError(f"block_len must be positive, got {block_len}")
    if block_size % block_len != 0:
        raise ValueError(
            f"block_len={block_len} must divide block_size={block_size}"
        )


def num_blocks(block_size: int, block_len: int) -> int:
    validate_block_len(block_size, block_len)
    return block_size // block_len


# ------------------------------------------------------------------
# BD3-LM masks
# ------------------------------------------------------------------


def bd3_train_mask_mod(_batch, _head, q_idx, kv_idx, *, seq_len: int, block_len: int):
    """
    Token-level BD3 training mask rule used by both dense SDPA masks and
    sparse FlexAttention block masks.
    """
    q_is_x0 = q_idx >= seq_len
    kv_is_x0 = kv_idx >= seq_len

    q_block = torch.where(q_is_x0, (q_idx - seq_len) // block_len, q_idx // block_len)
    kv_block = torch.where(kv_is_x0, (kv_idx - seq_len) // block_len, kv_idx // block_len)

    block_diagonal = (q_block == kv_block) & (q_is_x0 == kv_is_x0)
    offset_block_causal = (q_block > kv_block) & kv_is_x0 & (~q_is_x0)
    block_causal = (q_block >= kv_block) & kv_is_x0 & q_is_x0
    return block_diagonal | offset_block_causal | block_causal


def block_causal_mask_mod(_batch, _head, q_idx, kv_idx, *, block_len: int):
    """Token-level one-stream block-causal rule used during BD3 sampling."""
    return (q_idx // block_len) >= (kv_idx // block_len)


def make_bd3_train_mask(seq_len: int, block_len: int, device=None) -> torch.Tensor:
    """
    Full 2L x 2L BD3-LM training mask for x_t ⊕ x_0.

    This implements the paper/repo mask

        M_full = [[M_BD, M_OBC],
                  [   0,  M_BC]]

    with
      - M_BD  : block-diagonal attention within noisy blocks x_t
      - M_OBC : attention from noisy block b to clean blocks < b
      - M_BC  : block-causal attention on the clean x_0 stream

    Returned mask is boolean and shaped (1, 1, 2L, 2L), ready for SDPA.
    True means "allowed to attend".
    """
    validate_block_len(seq_len, block_len)
    idx = torch.arange(2 * seq_len, device=device)
    mask = bd3_train_mask_mod(
        None,
        None,
        idx[:, None],
        idx[None, :],
        seq_len=seq_len,
        block_len=block_len,
    )
    return mask[None, None, :, :]


def make_block_causal_mask(seq_len: int, block_len: int, device=None) -> torch.Tensor:
    """
    One-stream block-causal mask used at sampling time.

    Token i may attend to token j iff block(j) <= block(i), i.e.
    previous blocks are fully visible and the current block is
    bidirectional within itself.
    """
    validate_block_len(seq_len, block_len)
    pos = torch.arange(seq_len, device=device)
    mask = block_causal_mask_mod(
        None,
        None,
        pos[:, None],
        pos[None, :],
        block_len=block_len,
    )
    return mask[None, None, :, :]


def _normalize_mask_device(device) -> str:
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(device)


def make_bd3_train_block_mask(seq_len: int, block_len: int, device=None):
    """
    Sparse FlexAttention block mask for dual-stream BD3 training.

    Intentionally uses FlexAttention's default sparse block size instead of
    tying it to the logical BD3 token block_len. This matches the upstream
    BD3LM usage and keeps the sparse-mask metadata much smaller for long
    sequences (e.g. 4096 dual-stream tokens).
    """
    if not FLEX_ATTN_AVAILABLE:
        raise RuntimeError("FlexAttention is not available in this torch build")
    validate_block_len(seq_len, block_len)
    return create_block_mask(
        partial(bd3_train_mask_mod, seq_len=seq_len, block_len=block_len),
        B=None,
        H=None,
        Q_LEN=2 * seq_len,
        KV_LEN=2 * seq_len,
        device=_normalize_mask_device(device),
    )


def make_block_causal_block_mask(seq_len: int, block_len: int, device=None):
    """
    Sparse FlexAttention block mask for one-stream BD3 sampling.
    """
    if not FLEX_ATTN_AVAILABLE:
        raise RuntimeError("FlexAttention is not available in this torch build")
    validate_block_len(seq_len, block_len)
    return create_block_mask(
        partial(block_causal_mask_mod, block_len=block_len),
        B=None,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=_normalize_mask_device(device),
    )


# ------------------------------------------------------------------
# Agreement checks (used by tests)
# ------------------------------------------------------------------


def block_causal_equals_causal_when_block_len_is_one(seq_len: int, device=None) -> bool:
    mask = make_block_causal_mask(seq_len, 1, device=device)[0, 0]
    tri = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    return bool(torch.equal(mask, tri))


def bd3_train_mask_special_cases_ok(seq_len: int, block_len: int, device=None) -> bool:
    mask = make_bd3_train_mask(seq_len, block_len, device=device)[0, 0]
    xt_to_xt = mask[:seq_len, :seq_len]
    xt_to_x0 = mask[:seq_len, seq_len:]
    x0_to_xt = mask[seq_len:, :seq_len]
    x0_to_x0 = mask[seq_len:, seq_len:]

    if block_len == seq_len:
        cond_1 = bool(xt_to_xt.all())
        cond_2 = bool((~xt_to_x0).all())
        cond_3 = bool((~x0_to_xt).all())
        cond_4 = bool(x0_to_x0.all())
        return cond_1 and cond_2 and cond_3 and cond_4

    if block_len == 1:
        eye = torch.eye(seq_len, dtype=torch.bool, device=device)
        tri = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        strict_lower = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=-1)
        return (
            bool(torch.equal(xt_to_xt, eye))
            and bool(torch.equal(xt_to_x0, strict_lower))
            and bool((~x0_to_xt).all())
            and bool(torch.equal(x0_to_x0, tri))
        )

    return True
