"""Small MDLM/BD3LM inspection runs.

This is intentionally not a rigid test suite. It prints statistics that are
useful before profiling: mask rates, logits shape/range/entropy, top tokens,
position-to-position similarity, and BD3 clean-stream dependency checks.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from transformer.transformer import Transformer
from training.diffusion import (
    DiffusionConfig,
    make_model_context,
    prepare_diffusion_training_batch,
    survival_prob_np,
)
from training.loss import supervised_lm_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=3)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--base-vocab-size", type=int, default=64)
    p.add_argument("--d-model", type=int, default=48)
    p.add_argument("--d-ff", type=int, default=96)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--diffusion-steps", type=int, default=100)
    p.add_argument("--t-min", type=float, default=0.45)
    p.add_argument("--t-max", type=float, default=0.95)
    p.add_argument("--eval-t-frac", type=float, default=0.6)
    p.add_argument("--bd3-block-len", type=int, default=4)
    return p.parse_args()


def entropy_from_logits(logits: jax.Array) -> jax.Array:
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    probs = jnp.exp(log_probs)
    return -jnp.sum(probs * log_probs, axis=-1)


def summarize_logits(name: str, logits: jax.Array, mask_token_id: int) -> None:
    logits_np = np.asarray(logits)
    finite = bool(np.isfinite(logits_np).all())
    ent = np.asarray(entropy_from_logits(logits))
    top_vals, top_ids = jax.lax.top_k(logits[0, 0].astype(jnp.float32), k=5)
    top_vals = np.asarray(top_vals)
    top_ids = np.asarray(top_ids)
    top1 = np.asarray(jnp.argmax(logits, axis=-1))

    flat = logits.reshape((-1, logits.shape[-1])).astype(jnp.float32)
    flat = flat - jnp.mean(flat, axis=-1, keepdims=True)
    flat = flat / jnp.maximum(jnp.linalg.norm(flat, axis=-1, keepdims=True), 1e-8)
    sim = np.asarray(flat @ flat.T)
    offdiag = sim[~np.eye(sim.shape[0], dtype=bool)]

    mask_rank = np.asarray(jnp.sum(logits > logits[..., mask_token_id, None], axis=-1) + 1)
    print(
        f"{name}: shape={tuple(logits.shape)} finite={finite} "
        f"mean={float(logits_np.mean()):.4f} std={float(logits_np.std()):.4f} "
        f"min={float(logits_np.min()):.4f} max={float(logits_np.max()):.4f}"
    )
    print(
        f"{name}: entropy mean={float(ent.mean()):.4f} std={float(ent.std()):.4f} "
        f"pos-cos offdiag mean={float(offdiag.mean()):.4f} max={float(offdiag.max()):.4f}"
    )
    print(
        f"{name}: first-token top5 ids={top_ids.tolist()} vals="
        f"{[round(float(x), 4) for x in top_vals.tolist()]}"
    )
    print(f"{name}: first-row top1 ids={top1[0].tolist()}")
    print(
        f"{name}: mask-token rank mean={float(mask_rank.mean()):.2f} "
        f"min={int(mask_rank.min())} max={int(mask_rank.max())}"
    )


def run_objective(
    objective: str,
    model: Transformer,
    clean: np.ndarray,
    cfg: DiffusionConfig,
    rng: np.random.Generator,
    fixed_t_step: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    inputs_np, targets_np, supervise_np = prepare_diffusion_training_batch(
        objective,
        clean,
        cfg,
        rng,
        fixed_t_step=fixed_t_step,
    )
    ctx = make_model_context(objective, clean.shape[1], cfg)
    inputs = jnp.asarray(inputs_np, dtype=jnp.int32)
    targets = jnp.asarray(targets_np, dtype=jnp.int32)
    supervise = jnp.asarray(supervise_np, dtype=bool)
    logits = model(
        inputs,
        token_positions=ctx.token_positions,
        attention_mask=ctx.attention_mask,
        is_causal=ctx.is_causal,
    )
    if ctx.output_length is not None:
        logits = logits[:, : ctx.output_length]
    _, metrics = supervised_lm_loss(
        model,
        inputs,
        targets,
        supervise,
        token_positions=ctx.token_positions,
        attention_mask=ctx.attention_mask,
        is_causal=ctx.is_causal,
        output_length=ctx.output_length,
        z_loss_weight=0.0,
    )
    print(
        f"{objective}: model_input_shape={tuple(inputs.shape)} "
        f"target_shape={tuple(targets.shape)} mask_rate={float(supervise.mean()):.4f} "
        f"loss={float(metrics['loss']):.4f} supervised={float(metrics['supervised_tokens']):.0f}"
    )
    print(f"{objective}: clean[0]={clean[0].tolist()}")
    print(f"{objective}: input[0]={inputs_np[0].tolist()}")
    print(f"{objective}: supervise[0]={supervise_np[0].astype(int).tolist()}")
    summarize_logits(objective, logits, cfg.mask_token_id)
    return inputs, targets, supervise, logits


def bd3_perturbation_check(model: Transformer, clean: np.ndarray, cfg: DiffusionConfig) -> None:
    seq_len = clean.shape[1]
    if cfg.block_len >= seq_len:
        print("bd3 perturbation: skipped because block_len >= seq_len")
        return
    ctx = make_model_context("bd3lm", seq_len, cfg)
    xt = jnp.asarray(clean[:1].copy(), dtype=jnp.int32)
    xt = xt.at[:, seq_len // 2 :].set(cfg.mask_token_id)
    x0_a = jnp.asarray(clean[:1], dtype=jnp.int32)
    x0_b = x0_a.at[:, : cfg.block_len].set(
        (x0_a[:, : cfg.block_len] + 7) % cfg.mask_token_id
    )
    logits_a = model(
        jnp.concatenate([xt, x0_a], axis=1),
        token_positions=ctx.token_positions,
        attention_mask=ctx.attention_mask,
        is_causal=ctx.is_causal,
    )[:, :seq_len]
    logits_b = model(
        jnp.concatenate([xt, x0_b], axis=1),
        token_positions=ctx.token_positions,
        attention_mask=ctx.attention_mask,
        is_causal=ctx.is_causal,
    )[:, :seq_len]
    diff = np.asarray(jnp.max(jnp.abs(logits_a - logits_b), axis=-1))[0]
    print(
        "bd3 perturbation max-logit-diff by position after changing clean block 0:",
        [round(float(x), 6) for x in diff.tolist()],
    )
    print(
        f"bd3 perturbation: block0_max={float(diff[:cfg.block_len].max()):.6f} "
        f"block1_max={float(diff[cfg.block_len:2*cfg.block_len].max()):.6f}"
    )


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    mask_token_id = args.base_vocab_size
    model_vocab_size = args.base_vocab_size + 1
    fixed_t_step = int(round(args.eval_t_frac * args.diffusion_steps))
    cfg = DiffusionConfig(
        num_steps=args.diffusion_steps,
        t_min=args.t_min,
        t_max=args.t_max,
        mask_token_id=mask_token_id,
        block_len=min(args.bd3_block_len, args.seq_len),
    )

    print("devices:", [str(d) for d in jax.devices()])
    print(
        "schedule:",
        {
            "steps": args.diffusion_steps,
            "t_min": args.t_min,
            "t_max": args.t_max,
            "fixed_t_step": fixed_t_step,
            "survival_fixed": float(survival_prob_np(np.asarray([fixed_t_step]), cfg)[0]),
        },
    )

    clean = rng.integers(
        1,
        args.base_vocab_size,
        size=(args.batch_size, args.seq_len),
        dtype=np.int32,
    )
    model = Transformer(
        nnx.Rngs(args.seed),
        n_layers=args.n_layers,
        vocab_size=model_vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_seq_len=args.seq_len,
        is_causal=False,
        weight_tying=True,
        dtype=jnp.float32,
    )

    run_objective("mdlm", model, clean, cfg, np.random.default_rng(args.seed + 1), fixed_t_step)
    run_objective("bd3lm", model, clean, cfg, np.random.default_rng(args.seed + 2), fixed_t_step)
    bd3_perturbation_check(model, clean, cfg)


if __name__ == "__main__":
    main()
