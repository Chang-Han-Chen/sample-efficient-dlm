"""MDLM and BD3LM batch construction helpers.

The training objective semantics mirror ``baby-dLM``:

* MDLM samples one timestep per sequence and trains denoising CE on masked
  positions under bidirectional attention.
* BD3LM samples one timestep per block, trains on ``x_t || x_0`` with the BD3
  four-quadrant mask, and scores only the noisy-stream outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from transformer.masks import make_bd3_train_mask, validate_block_len

Array = jax.Array
Objective = Literal["ar", "mdlm", "bd3lm"]


@dataclass(frozen=True)
class DiffusionConfig:
    num_steps: int = 100
    t_min: float = 0.45
    t_max: float = 0.95
    noise_schedule: Literal["linear", "cosine"] = "linear"
    mask_token_id: int = 8192
    block_len: int = 128


class ModelContext(NamedTuple):
    token_positions: Array | None
    attention_mask: Array | None
    is_causal: bool
    output_length: int | None
    bd3_block_len: int | None = None


def _clip_t_steps_np(t_steps: np.ndarray, cfg: DiffusionConfig) -> np.ndarray:
    return np.clip(t_steps.astype(np.int32), 1, int(cfg.num_steps))


def eval_t_step_from_frac(cfg: DiffusionConfig, frac: float) -> int:
    return int(max(1, min(cfg.num_steps, round(float(frac) * cfg.num_steps))))


def survival_prob_np(t_steps: np.ndarray, cfg: DiffusionConfig) -> np.ndarray:
    """Probability a token remains unmasked at integer diffusion timesteps."""
    t_steps = _clip_t_steps_np(t_steps, cfg)
    t_frac = t_steps.astype(np.float32) / float(cfg.num_steps)
    t_frac = np.clip(t_frac, float(cfg.t_min), float(cfg.t_max))
    if cfg.noise_schedule == "linear":
        a_t = 1.0 - t_frac
    elif cfg.noise_schedule == "cosine":
        a_t = np.cos(0.5 * np.pi * t_frac)
    else:
        raise ValueError(f"unknown noise_schedule: {cfg.noise_schedule!r}")
    return np.clip(a_t, 0.0, 1.0).astype(np.float32)


def survival_prob_jnp(t_steps: Array, cfg: DiffusionConfig) -> Array:
    t_steps = jnp.clip(t_steps.astype(jnp.float32), 1.0, float(cfg.num_steps))
    t_frac = jnp.clip(t_steps / float(cfg.num_steps), float(cfg.t_min), float(cfg.t_max))
    if cfg.noise_schedule == "linear":
        a_t = 1.0 - t_frac
    elif cfg.noise_schedule == "cosine":
        a_t = jnp.cos(0.5 * jnp.pi * t_frac)
    else:
        raise ValueError(f"unknown noise_schedule: {cfg.noise_schedule!r}")
    return jnp.clip(a_t, 0.0, 1.0)


def _sample_sequence_timesteps(
    rng: np.random.Generator,
    batch_size: int,
    cfg: DiffusionConfig,
    *,
    fixed_t_step: int | None = None,
) -> np.ndarray:
    if fixed_t_step is not None:
        t = np.full((batch_size,), fixed_t_step, dtype=np.int32)
        return _clip_t_steps_np(t, cfg)
    return rng.integers(1, int(cfg.num_steps) + 1, size=(batch_size,), dtype=np.int32)


def _sample_block_timesteps(
    rng: np.random.Generator,
    batch_size: int,
    seq_len: int,
    cfg: DiffusionConfig,
    *,
    fixed_t_step: int | None = None,
) -> np.ndarray:
    validate_block_len(seq_len, cfg.block_len)
    n_blocks = seq_len // int(cfg.block_len)
    if fixed_t_step is not None:
        t = np.full((batch_size, n_blocks), fixed_t_step, dtype=np.int32)
        return _clip_t_steps_np(t, cfg)
    return rng.integers(
        1,
        int(cfg.num_steps) + 1,
        size=(batch_size, n_blocks),
        dtype=np.int32,
    )


def expand_block_values_np(values: np.ndarray, block_len: int, *, seq_len: int | None = None) -> np.ndarray:
    out = np.repeat(values, int(block_len), axis=1)
    if seq_len is not None:
        out = out[:, :seq_len]
    return out


def make_mdlm_batch(
    x0: np.ndarray,
    cfg: DiffusionConfig,
    rng: np.random.Generator,
    *,
    fixed_t_step: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply MDLM per-sequence random masking to clean token IDs."""
    x0 = np.asarray(x0, dtype=np.int32)
    batch_size, seq_len = x0.shape
    t = _sample_sequence_timesteps(rng, batch_size, cfg, fixed_t_step=fixed_t_step)
    keep_prob = survival_prob_np(t, cfg)[:, None]
    supervise_mask = rng.random((batch_size, seq_len), dtype=np.float32) > keep_prob
    xt = x0.copy()
    xt[supervise_mask] = int(cfg.mask_token_id)
    return xt, x0, supervise_mask.astype(bool)


def make_bd3lm_batch(
    x0: np.ndarray,
    cfg: DiffusionConfig,
    rng: np.random.Generator,
    *,
    fixed_t_step: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply BD3 per-block random masking to clean token IDs."""
    x0 = np.asarray(x0, dtype=np.int32)
    batch_size, seq_len = x0.shape
    validate_block_len(seq_len, cfg.block_len)
    t_blocks = _sample_block_timesteps(
        rng,
        batch_size,
        seq_len,
        cfg,
        fixed_t_step=fixed_t_step,
    )
    keep_blocks = survival_prob_np(t_blocks, cfg)
    keep_tokens = expand_block_values_np(keep_blocks, cfg.block_len, seq_len=seq_len)
    supervise_mask = rng.random((batch_size, seq_len), dtype=np.float32) > keep_tokens
    xt = x0.copy()
    xt[supervise_mask] = int(cfg.mask_token_id)
    return xt, x0, supervise_mask.astype(bool)


def make_diffusion_batch(
    objective: Literal["mdlm", "bd3lm"],
    x0: np.ndarray,
    cfg: DiffusionConfig,
    rng: np.random.Generator,
    *,
    fixed_t_step: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if objective == "mdlm":
        return make_mdlm_batch(x0, cfg, rng, fixed_t_step=fixed_t_step)
    if objective == "bd3lm":
        return make_bd3lm_batch(x0, cfg, rng, fixed_t_step=fixed_t_step)
    raise ValueError(f"unknown diffusion objective: {objective!r}")


def prepare_diffusion_training_batch(
    objective: Literal["mdlm", "bd3lm"],
    x0: np.ndarray,
    cfg: DiffusionConfig,
    rng: np.random.Generator,
    *,
    fixed_t_step: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xt, x0, supervise_mask = make_diffusion_batch(
        objective,
        x0,
        cfg,
        rng,
        fixed_t_step=fixed_t_step,
    )
    if objective == "bd3lm" and int(cfg.block_len) < x0.shape[1]:
        model_inputs = np.concatenate([xt, x0], axis=1)
    else:
        model_inputs = xt
    return (
        model_inputs.astype(np.int32, copy=False),
        x0.astype(np.int32, copy=False),
        supervise_mask.astype(bool, copy=False),
    )


def make_model_context(
    objective: Objective,
    seq_len: int,
    cfg: DiffusionConfig | None = None,
    *,
    bd3_attention: Literal["dense", "blocked"] = "dense",
) -> ModelContext:
    """Static transformer call context for an objective/sequence length."""
    if objective == "ar":
        return ModelContext(None, None, True, None, None)
    if objective == "mdlm":
        return ModelContext(None, None, False, None, None)
    if objective != "bd3lm":
        raise ValueError(f"unknown objective: {objective!r}")
    if cfg is None:
        raise ValueError("cfg is required for bd3lm")
    validate_block_len(seq_len, cfg.block_len)
    if int(cfg.block_len) == int(seq_len):
        return ModelContext(None, None, False, None, None)
    if bd3_attention not in ("dense", "blocked"):
        raise ValueError("bd3_attention must be one of 'dense' or 'blocked'")
    token_positions = jnp.concatenate(
        [jnp.arange(seq_len, dtype=jnp.int32), jnp.arange(seq_len, dtype=jnp.int32)]
    )
    if bd3_attention == "blocked":
        return ModelContext(token_positions, None, False, seq_len, int(cfg.block_len))
    attention_mask = make_bd3_train_mask(seq_len, int(cfg.block_len))
    return ModelContext(token_positions, attention_mask, False, seq_len, None)
