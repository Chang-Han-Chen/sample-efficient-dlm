"""Minimal JAX AR trainer/benchmark.

Usage examples:

  python jax/train_ar.py --synthetic --max-steps 20
  python jax/train_ar.py --train-path /path/tokens.npy --max-steps 200
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jaxlib
import jax.numpy as jnp
import numpy as np
from flax import nnx

from transformer.transformer import Transformer
from training.data import MemoryMappedTokenDataset
from training.optimizer import NormuonAdamWConfig, create_normuon_adamw, learning_rates
from training.step import eval_step, train_step, train_step_accumulated


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-path", type=str, default=None)
    p.add_argument("--eval-path", type=str, default=None)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--vocab-size", type=int, default=32768)
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--d-ff", type=int, default=2048)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--n-heads", type=int, default=12)
    p.add_argument("--n-kv-heads", type=int, default=None)
    p.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    p.add_argument("--attention-impl", choices=("xla", "cudnn"), default=None)
    p.add_argument("--disable-qkv-fusion", action="store_true")
    p.add_argument("--disable-swiglu-fusion", action="store_true")
    p.add_argument("--attn-qknorm", action="store_true")
    p.add_argument("--attn-val-residual", action="store_true")
    p.add_argument("--attn-gating", default=False)
    p.add_argument("--layernorm-scaling", action="store_true")
    p.add_argument("--value-embedding", action="store_true")
    p.add_argument("--value-embedding-scale", type=float, default=1.0)
    p.add_argument("--weight-tying", dest="weight_tying", action="store_true")
    p.add_argument("--no-weight-tying", dest="weight_tying", action="store_false")
    p.set_defaults(weight_tying=True)
    p.add_argument("--grad-checkpoint-layers", type=int, default=0)
    p.add_argument("--adam-lr", type=float, default=7e-3)
    p.add_argument("--muon-lr", type=float, default=1.5e-2)
    p.add_argument("--adam-wd", type=float, default=0.0)
    p.add_argument("--muon-wd", type=float, default=1e-4)
    p.add_argument("--z-loss-weight", type=float, default=1e-4)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--eval-batches", type=int, default=0)
    p.add_argument("--overfit-batch", action="store_true")
    p.add_argument("--log-jsonl", type=Path, default=None)
    p.add_argument("--measure-start-step", type=int, default=2)
    p.add_argument("--peak-flops", type=float, default=312e12)
    return p.parse_args()


def synthetic_batch(rng: np.random.Generator, batch_size: int, context_length: int, vocab_size: int):
    tokens = rng.integers(0, vocab_size, size=(batch_size, context_length + 1), dtype=np.int32)
    return tokens[:, :-1], tokens[:, 1:]


def to_device_batch(batch: tuple[np.ndarray, np.ndarray]) -> tuple[jax.Array, jax.Array]:
    inputs_np, targets_np = batch
    return jnp.asarray(inputs_np, dtype=jnp.int32), jnp.asarray(targets_np, dtype=jnp.int32)


def mean_eval_loss(model, dataset: MemoryMappedTokenDataset, *, batches: int, batch_size: int) -> dict[str, float]:
    totals = {"loss": 0.0, "z_loss": 0.0}
    for _ in range(batches):
        inputs, targets = to_device_batch(dataset.get_batch(batch_size))
        metrics = eval_step(model, inputs, targets)
        totals["loss"] += float(jax.block_until_ready(metrics["loss"]))
        totals["z_loss"] += float(metrics["z_loss"])
    return {k: v / batches for k, v in totals.items()}


def count_parameters(model) -> int:
    total = 0
    for _, variable in nnx.to_flat_state(nnx.state(model, nnx.Param)):
        total += int(np.prod(variable[...].shape))
    return total


def estimate_compute_parameters(
    trainable_params: int,
    *,
    vocab_size: int,
    d_model: int,
    weight_tying: bool,
) -> int:
    # Tying removes a trainable lm_head parameter, but not the logits matmul.
    return trainable_params + (vocab_size * d_model if weight_tying else 0)


def estimate_training_flops_per_token(
    *,
    compute_params: int,
    n_layers: int,
    n_heads: int,
    d_model: int,
    context_length: int,
) -> float:
    head_dim = d_model // n_heads
    dense_flops = 6.0 * compute_params
    attention_flops = 12.0 * n_layers * n_heads * head_dim * context_length
    return dense_flops + attention_flops


def main():
    args = parse_args()
    if not args.synthetic and args.train_path is None:
        raise ValueError("Provide --train-path or pass --synthetic for a smoke run")

    dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32
    model = Transformer(
        nnx.Rngs(args.seed),
        n_layers=args.n_layers,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_kv_heads=args.n_kv_heads,
        attn_qknorm=args.attn_qknorm,
        attn_val_residual=args.attn_val_residual,
        attn_gating=args.attn_gating,
        layernorm_scaling=args.layernorm_scaling,
        value_embedding=args.value_embedding,
        value_embedding_scale=args.value_embedding_scale,
        weight_tying=args.weight_tying,
        num_grad_checkpoint_layers=args.grad_checkpoint_layers,
        max_seq_len=args.context_length,
        attention_impl=args.attention_impl,
        fuse_qkv=not args.disable_qkv_fusion,
        fuse_swiglu=not args.disable_swiglu_fusion,
        dtype=dtype,
    )
    opt_cfg = NormuonAdamWConfig(
        adam_lr=args.adam_lr,
        muon_lr=args.muon_lr,
        adam_weight_decay=args.adam_wd,
        muon_weight_decay=args.muon_wd,
        warmup_steps=args.warmup_steps,
    )
    optimizer = nnx.Optimizer(model, create_normuon_adamw(model, opt_cfg), wrt=nnx.Param)
    trainable_params = count_parameters(model)
    compute_params = estimate_compute_parameters(
        trainable_params,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        weight_tying=args.weight_tying,
    )
    flops_per_token = estimate_training_flops_per_token(
        compute_params=compute_params,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        context_length=args.context_length,
    )

    rng = np.random.default_rng(args.seed)
    dataset = None if args.synthetic else MemoryMappedTokenDataset(
        args.train_path,
        args.context_length,
        seed=args.seed,
    )
    eval_dataset = None
    if args.eval_path is not None and args.eval_batches > 0:
        eval_dataset = MemoryMappedTokenDataset(
            args.eval_path,
            args.context_length,
            seed=args.seed + 1,
        )

    print(
        "config:",
        {
            "steps": args.max_steps,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "context_length": args.context_length,
            "tokens_per_optimizer_step": args.batch_size * args.context_length * args.grad_accum_steps,
            "vocab_size": args.vocab_size,
            "dtype": args.dtype,
            "attention_impl": args.attention_impl,
            "fuse_qkv": not args.disable_qkv_fusion,
            "fuse_swiglu": not args.disable_swiglu_fusion,
            "attn_qknorm": args.attn_qknorm,
            "attn_val_residual": args.attn_val_residual,
            "attn_gating": args.attn_gating,
            "layernorm_scaling": args.layernorm_scaling,
            "value_embedding": args.value_embedding,
            "value_embedding_scale": args.value_embedding_scale,
            "weight_tying": args.weight_tying,
            "optimizer": "NorMuonCWD+AdamW",
            "trainable_param_count": trainable_params,
            "compute_param_count": compute_params,
            "flops_per_token_estimate": flops_per_token,
            "peak_flops_denominator": args.peak_flops,
            "jax_version": jax.__version__,
            "jaxlib_version": jaxlib.__version__,
            "overfit_batch": args.overfit_batch,
            "eval_batches": args.eval_batches,
            "measure_start_step": args.measure_start_step,
            "devices": [str(d) for d in jax.devices()],
        },
    )

    fixed_batch = None
    if args.overfit_batch:
        if dataset is None:
            fixed_batch = synthetic_batch(
                rng,
                args.batch_size,
                args.context_length,
                args.vocab_size,
            )
        else:
            fixed_batch = dataset.get_batch(args.batch_size)
        x0, y0 = fixed_batch
        print(
            "overfit_batch:",
            {
                "x_shape": tuple(x0.shape),
                "y_shape": tuple(y0.shape),
                "shift_ok": bool(np.array_equal(x0[:, 1:], y0[:, :-1])),
                "x_min": int(x0.min()),
                "x_max": int(x0.max()),
                "first_row_x": x0[0, : min(16, x0.shape[1])].tolist(),
                "first_row_y": y0[0, : min(16, y0.shape[1])].tolist(),
            },
        )

    log_fh = None
    if args.log_jsonl is not None:
        args.log_jsonl.parent.mkdir(parents=True, exist_ok=True)
        log_fh = args.log_jsonl.open("w")

    compile_start = time.perf_counter()
    first_metrics = None
    measured_time = 0.0
    measured_steps = 0
    try:
        for step in range(args.max_steps):
            data_start = time.perf_counter()
            batches = []
            for _ in range(args.grad_accum_steps):
                if fixed_batch is not None:
                    batch = fixed_batch
                elif dataset is None:
                    batch = synthetic_batch(
                        rng,
                        args.batch_size,
                        args.context_length,
                        args.vocab_size,
                    )
                else:
                    batch = dataset.get_batch(args.batch_size)
                batches.append(batch)
            data_time = time.perf_counter() - data_start

            if args.grad_accum_steps == 1:
                inputs, targets = to_device_batch(batches[0])
            else:
                inputs_np = np.stack([b[0] for b in batches], axis=0)
                targets_np = np.stack([b[1] for b in batches], axis=0)
                inputs = jnp.asarray(inputs_np, dtype=jnp.int32)
                targets = jnp.asarray(targets_np, dtype=jnp.int32)

            start = time.perf_counter()
            if args.grad_accum_steps == 1:
                metrics = train_step(
                    model,
                    optimizer,
                    inputs,
                    targets,
                    args.z_loss_weight,
                    args.max_grad_norm,
                )
            else:
                metrics = train_step_accumulated(
                    model,
                    optimizer,
                    inputs,
                    targets,
                    args.z_loss_weight,
                    args.max_grad_norm,
                )
            loss_value = float(jax.block_until_ready(metrics["loss"]))
            elapsed = time.perf_counter() - start
            if first_metrics is None:
                first_metrics = metrics
                compile_time = time.perf_counter() - compile_start
            if step >= args.measure_start_step:
                measured_time += elapsed
                measured_steps += 1
            adam_lr, muon_lr = learning_rates(jnp.asarray(step, dtype=jnp.int32), opt_cfg)
            row = {
                "step": step,
                "loss": loss_value,
                "z_loss": float(metrics["z_loss"]),
                "total_loss": float(metrics["total_loss"]),
                "grad_norm": float(metrics["grad_norm"]),
                "adam_lr": float(adam_lr),
                "muon_lr": float(muon_lr),
                "data_time_sec": data_time,
                "step_time_sec": elapsed,
            }
            if eval_dataset is not None and step % args.log_every == 0:
                eval_metrics = mean_eval_loss(
                    model,
                    eval_dataset,
                    batches=args.eval_batches,
                    batch_size=args.batch_size,
                )
                row["eval_loss"] = eval_metrics["loss"]
                row["eval_z_loss"] = eval_metrics["z_loss"]

            if step % args.log_every == 0:
                eval_text = f" eval_loss={row['eval_loss']:.4f}" if "eval_loss" in row else ""
                print(
                    f"step={step:05d} loss={row['loss']:.4f}{eval_text} "
                    f"z={row['z_loss']:.4f} grad_norm={row['grad_norm']:.3f} "
                    f"adam_lr={row['adam_lr']:.3e} muon_lr={row['muon_lr']:.3e} "
                    f"data_time={data_time:.4f}s step_time={elapsed:.4f}s"
                )
            if log_fh is not None:
                log_fh.write(json.dumps(row) + "\n")
                log_fh.flush()
    finally:
        if log_fh is not None:
            log_fh.close()

    if first_metrics is None:
        return

    if measured_steps == 0:
        measured_time = elapsed
        measured_steps = 1
        measured_from = args.max_steps - 1
    else:
        measured_from = args.measure_start_step
    avg_step = measured_time / measured_steps
    tokens_per_step = args.batch_size * args.context_length * args.grad_accum_steps
    flops_per_step = flops_per_token * tokens_per_step
    achieved_flops = flops_per_step / avg_step
    mfu = achieved_flops / args.peak_flops
    memory_stats = {}
    try:
        memory_stats = jax.devices()[0].memory_stats() or {}
    except Exception:
        memory_stats = {}
    peak_hbm_bytes = int(memory_stats.get("peak_bytes_in_use", 0) or 0)
    peak_reserved_bytes = int(memory_stats.get("peak_bytes_reserved", 0) or 0)
    print(
        f"compile_plus_first_step={compile_time:.3f}s "
        f"avg_measured_step={avg_step:.4f}s "
        f"measured_from_step={measured_from} "
        f"tokens_per_sec={tokens_per_step / avg_step:.0f} "
        f"est_tflops={achieved_flops / 1e12:.1f} "
        f"mfu={mfu * 100:.1f}% "
        f"mfu_peak_tflops={args.peak_flops / 1e12:.0f} "
        f"jax_peak_hbm_gb={peak_hbm_bytes / 1e9:.2f} "
        f"jax_peak_reserved_gb={peak_reserved_bytes / 1e9:.2f}"
    )


if __name__ == "__main__":
    main()
