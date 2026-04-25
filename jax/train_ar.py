"""Minimal JAX AR trainer/benchmark.

Usage examples:

  python jax/train_ar.py --synthetic --max-steps 20
  python jax/train_ar.py --train-path /path/tokens.npy --max-steps 200
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jaxlib
import jax.numpy as jnp
import numpy as np
from flax import nnx, serialization

from transformer.transformer import Transformer
from training.data import MemoryMappedTokenDataset
from training.diffusion import (
    DiffusionConfig,
    eval_t_step_from_frac,
    make_model_context,
    prepare_diffusion_training_batch,
)
from training.optimizer import NormuonAdamWConfig, create_normuon_adamw, learning_rates
from training.step import (
    eval_step,
    eval_step_supervised,
    train_step,
    train_step_accumulated,
    train_step_accumulated_data_parallel,
    train_step_data_parallel,
    train_step_supervised,
    train_step_supervised_accumulated,
    train_step_supervised_accumulated_data_parallel,
    train_step_supervised_data_parallel,
)


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(parents=[pre])
    p.add_argument("--objective", "--model", choices=("ar", "mdlm", "bd3lm"), default="ar")
    p.add_argument("--train-path", type=str, default=None)
    p.add_argument("--eval-path", type=str, default=None)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--checkpoint-dir", type=Path, default=None)
    p.add_argument("--checkpoint-interval", type=int, default=0)
    p.add_argument("--restore-checkpoint", type=Path, default=None)
    p.add_argument("--restore-wandb-artifact", type=str, default=None)
    p.add_argument(
        "--wandb-artifact-dir",
        type=Path,
        default=Path(os.environ.get("WANDB_ARTIFACT_DIR", "/tmp/sample_efficient_gpt_wandb_artifacts")),
    )
    p.add_argument("--no-save-final-checkpoint", dest="save_final_checkpoint", action="store_false")
    p.add_argument("--no-save-best-checkpoint", dest="save_best_checkpoint", action="store_false")
    p.add_argument("--no-wandb-checkpoints", dest="wandb_checkpoints", action="store_false")
    p.set_defaults(save_final_checkpoint=True, save_best_checkpoint=True, wandb_checkpoints=True)
    p.add_argument("--no-wandb", dest="wandb", action="store_false", help=argparse.SUPPRESS)
    p.set_defaults(wandb=True)
    p.add_argument("--wandb-entity", type=str, default="y38283929-uc-berkeley-electrical-engineering-computer-sc")
    p.add_argument("--wandb-project", type=str, default="sample-efficient-dlm")
    p.add_argument("--wandb-group", type=str, default=None)
    p.add_argument("--wandb-tags", nargs="*", default=None)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--data-parallel", action="store_true")
    p.add_argument("--num-devices", type=int, default=None)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--vocab-size", type=int, default=8192)
    p.add_argument("--mask-token-id", type=int, default=None)
    p.add_argument("--diffusion-steps", type=int, default=100)
    p.add_argument("--t-min", type=float, default=0.45)
    p.add_argument("--t-max", type=float, default=0.95)
    p.add_argument("--noise-schedule", choices=("linear", "cosine"), default="linear")
    p.add_argument("--bd3-block-len", "--block-len", dest="bd3_block_len", type=int, default=128)
    p.add_argument("--bd3-attention", choices=("dense", "blocked"), default="dense")
    p.add_argument("--eval-t-frac", type=float, default=0.6)
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
    p.add_argument("--adam-lr", type=float, default=None, help=argparse.SUPPRESS)
    p.add_argument("--table-adam-lr", type=float, default=7e-3)
    p.add_argument("--scalar-adam-lr", type=float, default=7e-3)
    p.add_argument("--muon-lr", type=float, default=1.5e-2)
    p.add_argument("--lr-mult", type=float, default=1.0)
    p.add_argument("--adam-wd", type=float, default=0.0)
    p.add_argument("--muon-wd", type=float, default=1e-4)
    p.add_argument("--adam-beta1", type=float, default=0.95)
    p.add_argument("--adam-beta2", type=float, default=0.99)
    p.add_argument("--value-embedding-adam-beta1", type=float, default=0.8)
    p.add_argument("--value-embedding-adam-beta2", type=float, default=0.95)
    p.add_argument("--adam-eps", type=float, default=1e-8)
    p.add_argument("--muon-beta2", type=float, default=0.95)
    p.add_argument("--muon-momentum", type=float, default=0.95)
    p.add_argument("--momentum-warmup-steps", type=int, default=300)
    p.add_argument("--momentum-warmup-start", type=float, default=0.85)
    p.add_argument("--z-loss-weight", type=float, default=1e-4)
    p.add_argument("--loss-impl", choices=("full", "chunked"), default="full")
    p.add_argument("--logit-chunk-size", type=int, default=1024)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--eval-batches", type=int, default=0)
    p.add_argument("--overfit-batch", action="store_true")
    p.add_argument("--log-jsonl", type=Path, default=None)
    p.add_argument("--measure-start-step", type=int, default=2)
    p.add_argument("--peak-flops", type=float, default=312e12)
    p.add_argument(
        "--compilation-cache-dir",
        type=Path,
        default=Path(os.environ.get("JAX_COMPILATION_CACHE_DIR", "/tmp/sample_efficient_gpt_jax_cache")),
    )
    p.add_argument("--disable-compilation-cache", action="store_true")
    p.add_argument("--explain-cache-misses", action="store_true")

    if pre_args.config is not None:
        import yaml

        with pre_args.config.open() as fh:
            config_doc = yaml.safe_load(fh) or {}
        train_args = config_doc.get("train_args")
        if train_args is None:
            raise ValueError(f"{pre_args.config} must contain a top-level train_args mapping")
        if not isinstance(train_args, dict):
            raise TypeError(f"{pre_args.config}: train_args must be a mapping")
        valid_dests = {action.dest for action in p._actions}
        unknown = sorted(set(train_args) - valid_dests)
        if unknown:
            raise ValueError(f"{pre_args.config}: unknown train_args keys: {unknown}")
        path_keys = {
            "checkpoint_dir",
            "compilation_cache_dir",
            "log_jsonl",
            "output_dir",
            "restore_checkpoint",
            "wandb_artifact_dir",
        }
        defaults = {
            key: Path(value) if key in path_keys and value is not None else value
            for key, value in train_args.items()
        }
        p.set_defaults(**defaults)
    return p.parse_args()


def synthetic_batch(rng: np.random.Generator, batch_size: int, context_length: int, vocab_size: int):
    tokens = rng.integers(0, vocab_size, size=(batch_size, context_length + 1), dtype=np.int32)
    return tokens[:, :-1], tokens[:, 1:]


def to_device_batch(batch: tuple[np.ndarray, np.ndarray]) -> tuple[jax.Array, jax.Array]:
    inputs_np, targets_np = batch
    return jnp.asarray(inputs_np, dtype=jnp.int32), jnp.asarray(targets_np, dtype=jnp.int32)


def shard_batch_for_devices(array: np.ndarray, num_devices: int) -> np.ndarray:
    if array.shape[0] % num_devices != 0:
        raise ValueError(
            f"global batch size {array.shape[0]} must be divisible by num_devices={num_devices}"
        )
    per_device = array.shape[0] // num_devices
    return array.reshape((num_devices, per_device, *array.shape[1:]))


def shard_accumulated_for_devices(array: np.ndarray, num_devices: int) -> np.ndarray:
    if array.shape[1] % num_devices != 0:
        raise ValueError(
            f"global batch size {array.shape[1]} must be divisible by num_devices={num_devices}"
        )
    accum_steps, global_batch = array.shape[:2]
    per_device = global_batch // num_devices
    sharded = array.reshape((accum_steps, num_devices, per_device, *array.shape[2:]))
    return np.swapaxes(sharded, 0, 1)


def scalarize_metrics(metrics: dict[str, jax.Array]) -> dict[str, jax.Array]:
    return {
        key: jnp.mean(value) if getattr(value, "shape", ()) != () else value
        for key, value in metrics.items()
    }


def diffusion_expected_mask_rate(cfg: DiffusionConfig) -> float:
    return 0.5 * (float(cfg.t_min) + float(cfg.t_max))


def diffusion_loss_normalizer(batch_size: int, context_length: int, cfg: DiffusionConfig) -> float:
    return float(batch_size) * float(context_length) * diffusion_expected_mask_rate(cfg)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_bytes_atomic(path: Path, data: bytes) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_bytes(data)
    tmp_path.replace(path)


def write_json_atomic(path: Path, payload: dict) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    tmp_path.replace(path)


def save_training_checkpoint(
    checkpoint_path: Path,
    model,
    optimizer,
    *,
    step: int,
    kind: str,
    metrics: dict[str, float],
    resolved_config: dict,
    rng_state: dict | None = None,
    dataset_rng_state: dict | None = None,
    eval_rng_state: dict | None = None,
) -> dict:
    """Save model params and optimizer state in a simple restore-tested format."""
    if checkpoint_path.exists():
        shutil.rmtree(checkpoint_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    model_path = checkpoint_path / "model.msgpack"
    optimizer_path = checkpoint_path / "optimizer.msgpack"
    config_path = checkpoint_path / "resolved_config.json"
    metadata_path = checkpoint_path / "metadata.json"

    model_state = jax.device_get(nnx.to_pure_dict(nnx.state(model, nnx.Param)))
    optimizer_state = jax.device_get(nnx.to_pure_dict(nnx.state(optimizer)))
    write_bytes_atomic(model_path, serialization.to_bytes(model_state))
    write_bytes_atomic(optimizer_path, serialization.to_bytes(optimizer_state))
    write_json_atomic(config_path, resolved_config)

    metadata = {
        "format_version": 1,
        "kind": kind,
        "step": int(step),
        "metrics": metrics,
        "state_files": {
            "model": {
                "path": model_path.name,
                "sha256": sha256_file(model_path),
            },
            "optimizer": {
                "path": optimizer_path.name,
                "sha256": sha256_file(optimizer_path),
            },
        },
        "rng_state": rng_state,
        "dataset_rng_state": dataset_rng_state,
        "eval_rng_state": eval_rng_state,
    }
    write_json_atomic(metadata_path, metadata)
    return metadata


def restore_training_checkpoint(checkpoint_path: Path, model, optimizer) -> dict:
    metadata_path = checkpoint_path / "metadata.json"
    model_path = checkpoint_path / "model.msgpack"
    optimizer_path = checkpoint_path / "optimizer.msgpack"
    with metadata_path.open() as fh:
        metadata = json.load(fh)

    expected_model_hash = metadata.get("state_files", {}).get("model", {}).get("sha256")
    expected_optimizer_hash = metadata.get("state_files", {}).get("optimizer", {}).get("sha256")
    if expected_model_hash is not None and sha256_file(model_path) != expected_model_hash:
        raise ValueError(f"model checkpoint hash mismatch in {checkpoint_path}")
    if expected_optimizer_hash is not None and sha256_file(optimizer_path) != expected_optimizer_hash:
        raise ValueError(f"optimizer checkpoint hash mismatch in {checkpoint_path}")

    model_target = nnx.to_pure_dict(nnx.state(model, nnx.Param))
    restored_model = serialization.from_bytes(model_target, model_path.read_bytes())
    model_state = nnx.state(model, nnx.Param)
    nnx.replace_by_pure_dict(model_state, restored_model)
    nnx.update(model, model_state)

    optimizer_target = nnx.to_pure_dict(nnx.state(optimizer))
    restored_optimizer = serialization.from_bytes(optimizer_target, optimizer_path.read_bytes())
    optimizer_state = nnx.state(optimizer)
    nnx.replace_by_pure_dict(optimizer_state, restored_optimizer)
    nnx.update(optimizer, optimizer_state)

    return metadata


def sanitized_artifact_name(value: str) -> str:
    sanitized = "".join(c if c.isalnum() or c in "._-" else "-" for c in value)
    return sanitized.strip(".-") or "jax-train-checkpoint"


def upload_checkpoint_artifact(wandb_run, checkpoint_path: Path, *, kind: str) -> None:
    import wandb

    base_name = wandb_run.name or wandb_run.id or "jax-train"
    artifact_name = sanitized_artifact_name(f"{base_name}-checkpoint-{kind}")
    metadata_path = checkpoint_path / "metadata.json"
    with metadata_path.open() as fh:
        metadata = json.load(fh)
    step = int(metadata.get("step", 0))
    artifact = wandb.Artifact(
        artifact_name,
        type="model",
        metadata={
            "kind": kind,
            "step": step,
            "checkpoint_format_version": metadata.get("format_version", 1),
            "loss": metadata.get("metrics", {}).get("loss"),
            "eval_loss": metadata.get("metrics", {}).get("eval_loss"),
        },
    )
    artifact.add_dir(str(checkpoint_path))
    logged = wandb_run.log_artifact(artifact, aliases=[kind, f"step-{step}"])
    logged.wait()


def download_checkpoint_artifact(
    artifact_spec: str,
    *,
    entity: str,
    project: str,
    root: Path,
) -> Path:
    import wandb

    artifact_ref = (
        artifact_spec
        if artifact_spec.split(":", 1)[0].count("/") >= 2
        else f"{entity}/{project}/{artifact_spec}"
    )
    download_root = root / sanitized_artifact_name(artifact_spec.replace(":", "-"))
    artifact = wandb.Api().artifact(artifact_ref, type="model")
    downloaded = Path(artifact.download(root=str(download_root)))
    if not (downloaded / "metadata.json").exists():
        raise FileNotFoundError(
            f"W&B artifact {artifact_ref!r} did not download as a train_ar.py checkpoint"
        )
    return downloaded


def mean_eval_loss(
    model,
    dataset: MemoryMappedTokenDataset,
    *,
    batches: int,
    batch_size: int,
    objective: str,
    diffusion_cfg: DiffusionConfig | None,
    model_context,
    rng: np.random.Generator,
    fixed_t_step: int | None,
    loss_impl: str,
    logit_chunk_size: int,
    loss_normalizer: float | None,
) -> dict[str, float]:
    totals = {"loss": 0.0, "z_loss": 0.0}
    for _ in range(batches):
        raw_batch = dataset.get_batch(batch_size)
        if objective == "ar":
            inputs, targets = to_device_batch(raw_batch)
            metrics = eval_step(model, inputs, targets, loss_impl, logit_chunk_size)
        else:
            if diffusion_cfg is None:
                raise ValueError("diffusion_cfg is required for diffusion eval")
            inputs_np, targets_np, supervise_np = prepare_diffusion_training_batch(
                objective,
                raw_batch[0],
                diffusion_cfg,
                rng,
                fixed_t_step=fixed_t_step,
            )
            inputs = jnp.asarray(inputs_np, dtype=jnp.int32)
            targets = jnp.asarray(targets_np, dtype=jnp.int32)
            supervise_mask = jnp.asarray(supervise_np, dtype=bool)
            metrics = eval_step_supervised(
                model,
                inputs,
                targets,
                supervise_mask,
                model_context.token_positions,
                model_context.attention_mask,
                model_context.is_causal,
                model_context.output_length,
                loss_impl,
                logit_chunk_size,
                model_context.bd3_block_len,
                loss_normalizer,
            )
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
    if args.checkpoint_interval < 0:
        raise ValueError("--checkpoint-interval must be non-negative")
    if args.restore_checkpoint is not None and args.restore_wandb_artifact is not None:
        raise ValueError("Use either --restore-checkpoint or --restore-wandb-artifact, not both")
    if args.disable_compilation_cache:
        jax.config.update("jax_enable_compilation_cache", False)
    else:
        args.compilation_cache_dir.mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_enable_compilation_cache", True)
        jax.config.update("jax_compilation_cache_dir", str(args.compilation_cache_dir))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
        if args.explain_cache_misses:
            jax.config.update("jax_explain_cache_misses", True)
    if not args.synthetic and args.train_path is None:
        raise ValueError("Provide --train-path or pass --synthetic for a smoke run")
    local_devices = jax.local_devices()
    if args.data_parallel:
        num_devices = args.num_devices if args.num_devices is not None else len(local_devices)
        if num_devices < 1:
            raise ValueError("--num-devices must be at least 1")
        if num_devices > len(local_devices):
            raise ValueError(
                f"requested {num_devices} data-parallel devices, but JAX sees {len(local_devices)}"
            )
        if args.batch_size % num_devices != 0:
            raise ValueError(
                f"--batch-size {args.batch_size} must be divisible by --num-devices {num_devices}"
            )
    else:
        num_devices = 1
    per_device_batch_size = args.batch_size // num_devices

    dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32
    if args.objective == "ar":
        model_vocab_size = args.vocab_size
        diffusion_cfg = None
    else:
        mask_token_id = args.mask_token_id if args.mask_token_id is not None else args.vocab_size
        model_vocab_size = max(args.vocab_size + 1, int(mask_token_id) + 1)
        bd3_block_len = min(int(args.bd3_block_len), int(args.context_length))
        diffusion_cfg = DiffusionConfig(
            num_steps=args.diffusion_steps,
            t_min=args.t_min,
            t_max=args.t_max,
            noise_schedule=args.noise_schedule,
            mask_token_id=int(mask_token_id),
            block_len=bd3_block_len,
        )
    model_context = make_model_context(
        args.objective,
        args.context_length,
        diffusion_cfg,
        bd3_attention=args.bd3_attention,
    )
    model = Transformer(
        nnx.Rngs(args.seed),
        n_layers=args.n_layers,
        vocab_size=model_vocab_size,
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
        is_causal=(args.objective == "ar"),
        attention_impl=args.attention_impl,
        fuse_qkv=not args.disable_qkv_fusion,
        fuse_swiglu=not args.disable_swiglu_fusion,
        dtype=dtype,
    )
    table_adam_lr_base = args.table_adam_lr if args.adam_lr is None else args.adam_lr
    scalar_adam_lr_base = args.scalar_adam_lr if args.adam_lr is None else args.adam_lr
    opt_cfg = NormuonAdamWConfig(
        table_adam_lr=table_adam_lr_base * args.lr_mult,
        scalar_adam_lr=scalar_adam_lr_base * args.lr_mult,
        muon_lr=args.muon_lr * args.lr_mult,
        adam_weight_decay=args.adam_wd,
        muon_weight_decay=args.muon_wd,
        adam_betas=(args.adam_beta1, args.adam_beta2),
        value_embedding_adam_betas=(
            args.value_embedding_adam_beta1,
            args.value_embedding_adam_beta2,
        ),
        muon_momentum=args.muon_momentum,
        muon_beta2=args.muon_beta2,
        adam_eps=args.adam_eps,
        warmup_steps=args.warmup_steps,
        momentum_warmup_steps=args.momentum_warmup_steps,
        momentum_warmup_start=args.momentum_warmup_start,
    )
    optimizer = nnx.Optimizer(model, create_normuon_adamw(model, opt_cfg), wrt=nnx.Param)
    restored_metadata = None
    restore_checkpoint_path = args.restore_checkpoint
    if args.restore_wandb_artifact is not None:
        restore_checkpoint_path = download_checkpoint_artifact(
            args.restore_wandb_artifact,
            entity=args.wandb_entity,
            project=args.wandb_project,
            root=args.wandb_artifact_dir,
        )
        print(
            "downloaded_wandb_checkpoint:",
            {
                "artifact": args.restore_wandb_artifact,
                "path": str(restore_checkpoint_path),
            },
            flush=True,
        )
    if restore_checkpoint_path is not None:
        restored_metadata = restore_training_checkpoint(restore_checkpoint_path, model, optimizer)
        print(
            "restored_checkpoint:",
            {
                "path": str(restore_checkpoint_path),
                "step": restored_metadata.get("step"),
                "kind": restored_metadata.get("kind"),
            },
            flush=True,
        )
    trainable_params = count_parameters(model)
    compute_params = estimate_compute_parameters(
        trainable_params,
        vocab_size=model_vocab_size,
        d_model=args.d_model,
        weight_tying=args.weight_tying,
    )
    model_sequence_length = (
        2 * args.context_length
        if args.objective == "bd3lm"
        and diffusion_cfg is not None
        and diffusion_cfg.block_len < args.context_length
        else args.context_length
    )
    flops_per_token = estimate_training_flops_per_token(
        compute_params=compute_params,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        context_length=model_sequence_length,
    )
    if model_sequence_length != args.context_length:
        flops_per_token *= model_sequence_length / args.context_length

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
    if restored_metadata is not None:
        if restored_metadata.get("rng_state") is not None:
            rng.bit_generator.state = restored_metadata["rng_state"]
        if dataset is not None and restored_metadata.get("dataset_rng_state") is not None:
            dataset.rng.bit_generator.state = restored_metadata["dataset_rng_state"]

    eval_rng = np.random.default_rng(args.seed + 1_000_003)
    if restored_metadata is not None and restored_metadata.get("eval_rng_state") is not None:
        eval_rng.bit_generator.state = restored_metadata["eval_rng_state"]
    start_step = int(restored_metadata["step"]) + 1 if restored_metadata is not None else 0
    train_loss_normalizer = (
        None
        if diffusion_cfg is None
        else diffusion_loss_normalizer(
            args.batch_size * args.grad_accum_steps,
            args.context_length,
            diffusion_cfg,
        )
    )
    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None and args.output_dir is not None:
        checkpoint_dir = args.output_dir / "checkpoints"
    checkpointing_enabled = checkpoint_dir is not None and (
        args.checkpoint_interval > 0
        or args.save_final_checkpoint
        or args.save_best_checkpoint
    )
    if checkpointing_enabled:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    resolved_config = {
        "objective": args.objective,
        "config": None if args.config is None else str(args.config),
        "run_name": args.run_name,
        "output_dir": None if args.output_dir is None else str(args.output_dir),
        "checkpoint_dir": None if not checkpointing_enabled else str(checkpoint_dir),
        "checkpoint_interval": args.checkpoint_interval,
        "restore_checkpoint": None if restore_checkpoint_path is None else str(restore_checkpoint_path),
        "restore_wandb_artifact": args.restore_wandb_artifact,
        "wandb_artifact_dir": str(args.wandb_artifact_dir),
        "start_step": start_step,
        "save_final_checkpoint": args.save_final_checkpoint,
        "save_best_checkpoint": args.save_best_checkpoint,
        "wandb_checkpoints": args.wandb_checkpoints,
        "wandb": args.wandb,
        "wandb_entity": args.wandb_entity,
        "wandb_project": args.wandb_project,
        "wandb_group": args.wandb_group,
        "wandb_tags": args.wandb_tags,
        "steps": args.max_steps,
        "batch_size": args.batch_size,
        "per_device_batch_size": per_device_batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "data_parallel": args.data_parallel,
        "data_parallel_devices": num_devices,
        "context_length": args.context_length,
        "model_sequence_length": model_sequence_length,
        "tokens_per_optimizer_step": args.batch_size * args.context_length * args.grad_accum_steps,
        "base_vocab_size": args.vocab_size,
        "model_vocab_size": model_vocab_size,
        "mask_token_id": None if diffusion_cfg is None else diffusion_cfg.mask_token_id,
        "diffusion_loss_expected_mask_rate": (
            None if diffusion_cfg is None else diffusion_expected_mask_rate(diffusion_cfg)
        ),
        "diffusion_loss_normalizer": train_loss_normalizer,
        "diffusion_steps": None if diffusion_cfg is None else diffusion_cfg.num_steps,
        "t_min": None if diffusion_cfg is None else diffusion_cfg.t_min,
        "t_max": None if diffusion_cfg is None else diffusion_cfg.t_max,
        "noise_schedule": None if diffusion_cfg is None else diffusion_cfg.noise_schedule,
        "bd3_block_len": None if diffusion_cfg is None else diffusion_cfg.block_len,
        "bd3_attention": args.bd3_attention,
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
        "loss_impl": args.loss_impl,
        "logit_chunk_size": args.logit_chunk_size,
        "optimizer": "NorMuonCWD+AdamW",
        "lr_mult": args.lr_mult,
        "table_adam_lr_base": table_adam_lr_base,
        "scalar_adam_lr_base": scalar_adam_lr_base,
        "muon_lr_base": args.muon_lr,
        "table_adam_lr_peak": opt_cfg.table_adam_lr,
        "scalar_adam_lr_peak": opt_cfg.scalar_adam_lr,
        "muon_lr_peak": opt_cfg.muon_lr,
        "adam_wd": args.adam_wd,
        "muon_wd": args.muon_wd,
        "adam_betas": [args.adam_beta1, args.adam_beta2],
        "value_embedding_adam_betas": [
            args.value_embedding_adam_beta1,
            args.value_embedding_adam_beta2,
        ],
        "adam_eps": args.adam_eps,
        "muon_momentum": args.muon_momentum,
        "muon_beta2": args.muon_beta2,
        "momentum_warmup_steps": args.momentum_warmup_steps,
        "momentum_warmup_start": args.momentum_warmup_start,
        "trainable_param_count": trainable_params,
        "compute_param_count": compute_params,
        "flops_per_token_estimate": flops_per_token,
        "peak_flops_denominator": args.peak_flops,
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "overfit_batch": args.overfit_batch,
        "eval_batches": args.eval_batches,
        "measure_start_step": args.measure_start_step,
        "compilation_cache_dir": None if args.disable_compilation_cache else str(args.compilation_cache_dir),
        "devices": [str(d) for d in jax.devices()],
        "local_devices": [str(d) for d in local_devices[:num_devices]],
    }
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / "resolved_config.json").open("w") as fh:
            json.dump(resolved_config, fh, indent=2, sort_keys=True)
            fh.write("\n")
    print("config:", resolved_config, flush=True)

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.run_name,
            group=args.wandb_group,
            tags=args.wandb_tags,
            config=resolved_config,
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
            flush=True,
        )
        if args.objective != "ar" and diffusion_cfg is not None:
            preview_rng = np.random.default_rng(args.seed + 10_000)
            model_in, target_preview, supervise_preview = prepare_diffusion_training_batch(
                args.objective,
                x0,
                diffusion_cfg,
                preview_rng,
                fixed_t_step=eval_t_step_from_frac(diffusion_cfg, args.eval_t_frac),
            )
            print(
                "diffusion_overfit_preview:",
                {
                    "model_input_shape": tuple(model_in.shape),
                    "target_shape": tuple(target_preview.shape),
                    "mask_rate": float(supervise_preview.mean()),
                    "mask_token_id": diffusion_cfg.mask_token_id,
                    "first_row_input": model_in[0, : min(24, model_in.shape[1])].tolist(),
                    "first_row_target": target_preview[0, : min(24, target_preview.shape[1])].tolist(),
                    "first_row_supervise": supervise_preview[0, : min(24, supervise_preview.shape[1])].astype(int).tolist(),
                },
                flush=True,
            )

    log_fh = None
    if args.log_jsonl is not None:
        args.log_jsonl.parent.mkdir(parents=True, exist_ok=True)
        log_fh = args.log_jsonl.open("w")

    eval_fixed_t_step = (
        eval_t_step_from_frac(diffusion_cfg, args.eval_t_frac)
        if diffusion_cfg is not None
        else None
    )
    compile_start = time.perf_counter()
    first_metrics = None
    measured_time = 0.0
    measured_steps = 0
    best_eval_loss = None
    checkpoint_paths: dict[str, Path] = {}
    last_row = None
    try:
        for step in range(start_step, args.max_steps):
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

            if args.objective == "ar":
                if args.grad_accum_steps == 1:
                    inputs_np, targets_np = batches[0]
                    if args.data_parallel:
                        inputs_np = shard_batch_for_devices(inputs_np, num_devices)
                        targets_np = shard_batch_for_devices(targets_np, num_devices)
                    inputs = jnp.asarray(inputs_np, dtype=jnp.int32)
                    targets = jnp.asarray(targets_np, dtype=jnp.int32)
                else:
                    inputs_np = np.stack([b[0] for b in batches], axis=0)
                    targets_np = np.stack([b[1] for b in batches], axis=0)
                    if args.data_parallel:
                        inputs_np = shard_accumulated_for_devices(inputs_np, num_devices)
                        targets_np = shard_accumulated_for_devices(targets_np, num_devices)
                    inputs = jnp.asarray(inputs_np, dtype=jnp.int32)
                    targets = jnp.asarray(targets_np, dtype=jnp.int32)
                supervise_mask = None
            else:
                if diffusion_cfg is None:
                    raise ValueError("diffusion_cfg is required for diffusion training")
                prepared = [
                    prepare_diffusion_training_batch(
                        args.objective,
                        b[0],
                        diffusion_cfg,
                        rng,
                    )
                    for b in batches
                ]
                if args.grad_accum_steps == 1:
                    inputs_np, targets_np, supervise_np = prepared[0]
                    if args.data_parallel:
                        inputs_np = shard_batch_for_devices(inputs_np, num_devices)
                        targets_np = shard_batch_for_devices(targets_np, num_devices)
                        supervise_np = shard_batch_for_devices(supervise_np, num_devices)
                else:
                    inputs_np = np.stack([p[0] for p in prepared], axis=0)
                    targets_np = np.stack([p[1] for p in prepared], axis=0)
                    supervise_np = np.stack([p[2] for p in prepared], axis=0)
                    if args.data_parallel:
                        inputs_np = shard_accumulated_for_devices(inputs_np, num_devices)
                        targets_np = shard_accumulated_for_devices(targets_np, num_devices)
                        supervise_np = shard_accumulated_for_devices(supervise_np, num_devices)
                inputs = jnp.asarray(inputs_np, dtype=jnp.int32)
                targets = jnp.asarray(targets_np, dtype=jnp.int32)
                supervise_mask = jnp.asarray(supervise_np, dtype=bool)
            data_time = time.perf_counter() - data_start

            start = time.perf_counter()
            if args.objective == "ar":
                if args.grad_accum_steps == 1:
                    step_fn = train_step_data_parallel if args.data_parallel else train_step
                    metrics = step_fn(
                        model,
                        optimizer,
                        inputs,
                        targets,
                        args.z_loss_weight,
                        args.max_grad_norm,
                        args.loss_impl,
                        args.logit_chunk_size,
                    )
                else:
                    step_fn = (
                        train_step_accumulated_data_parallel
                        if args.data_parallel
                        else train_step_accumulated
                    )
                    metrics = step_fn(
                        model,
                        optimizer,
                        inputs,
                        targets,
                        args.z_loss_weight,
                        args.max_grad_norm,
                        args.loss_impl,
                        args.logit_chunk_size,
                    )
            else:
                if args.grad_accum_steps == 1:
                    step_fn = (
                        train_step_supervised_data_parallel
                        if args.data_parallel
                        else train_step_supervised
                    )
                    metrics = step_fn(
                        model,
                        optimizer,
                        inputs,
                        targets,
                        supervise_mask,
                        model_context.token_positions,
                        model_context.attention_mask,
                        args.z_loss_weight,
                        args.max_grad_norm,
                        model_context.is_causal,
                        model_context.output_length,
                        args.loss_impl,
                        args.logit_chunk_size,
                        model_context.bd3_block_len,
                        train_loss_normalizer,
                    )
                else:
                    step_fn = (
                        train_step_supervised_accumulated_data_parallel
                        if args.data_parallel
                        else train_step_supervised_accumulated
                    )
                    metrics = step_fn(
                        model,
                        optimizer,
                        inputs,
                        targets,
                        supervise_mask,
                        model_context.token_positions,
                        model_context.attention_mask,
                        args.z_loss_weight,
                        args.max_grad_norm,
                        model_context.is_causal,
                        model_context.output_length,
                        args.loss_impl,
                        args.logit_chunk_size,
                        model_context.bd3_block_len,
                        train_loss_normalizer,
                    )
            if args.data_parallel:
                metrics = scalarize_metrics(metrics)
            loss_value = float(jax.block_until_ready(metrics["loss"]))
            elapsed = time.perf_counter() - start
            if first_metrics is None:
                first_metrics = metrics
                compile_time = time.perf_counter() - compile_start
            if step >= args.measure_start_step:
                measured_time += elapsed
                measured_steps += 1
            table_adam_lr, scalar_adam_lr, muon_lr = learning_rates(jnp.asarray(step, dtype=jnp.int32), opt_cfg)
            row = {
                "step": step,
                "loss": loss_value,
                "z_loss": float(metrics["z_loss"]),
                "total_loss": float(metrics["total_loss"]),
                "grad_norm": float(metrics["grad_norm"]),
                "table_adam_lr": float(table_adam_lr),
                "scalar_adam_lr": float(scalar_adam_lr),
                "muon_lr": float(muon_lr),
                "data_time_sec": data_time,
                "step_time_sec": elapsed,
            }
            if "supervised_tokens" in metrics:
                row["supervised_tokens"] = float(metrics["supervised_tokens"])
            if "loss_normalizer" in metrics:
                row["loss_normalizer"] = float(metrics["loss_normalizer"])
            if eval_dataset is not None and step % args.log_every == 0:
                # When data-parallel is on, the global batch is sized to fit only
                # after sharding to ``num_devices``. Eval here runs through a
                # single-device JIT, so use the per-device shard size to avoid
                # OOM in the eval path even though training fits.
                eval_batch_size = per_device_batch_size if args.data_parallel else args.batch_size
                eval_loss_normalizer = (
                    None
                    if diffusion_cfg is None
                    else diffusion_loss_normalizer(
                        eval_batch_size,
                        args.context_length,
                        diffusion_cfg,
                    )
                )
                eval_metrics = mean_eval_loss(
                    model,
                    eval_dataset,
                    batches=args.eval_batches,
                    batch_size=eval_batch_size,
                    objective=args.objective,
                    diffusion_cfg=diffusion_cfg,
                    model_context=model_context,
                    rng=eval_rng,
                    fixed_t_step=eval_fixed_t_step,
                    loss_impl=args.loss_impl,
                    logit_chunk_size=args.logit_chunk_size,
                    loss_normalizer=eval_loss_normalizer,
                )
                row["eval_loss"] = eval_metrics["loss"]
                row["eval_z_loss"] = eval_metrics["z_loss"]
            last_row = row

            if step % args.log_every == 0:
                eval_text = f" eval_loss={row['eval_loss']:.4f}" if "eval_loss" in row else ""
                sup_text = (
                    f" supervised={row['supervised_tokens']:.0f}"
                    if "supervised_tokens" in row
                    else ""
                )
                print(
                    f"step={step:05d} loss={row['loss']:.4f}{eval_text} "
                    f"z={row['z_loss']:.4f}{sup_text} grad_norm={row['grad_norm']:.3f} "
                    f"table_lr={row['table_adam_lr']:.3e} "
                    f"scalar_lr={row['scalar_adam_lr']:.3e} "
                    f"muon_lr={row['muon_lr']:.3e} "
                    f"data_time={data_time:.4f}s step_time={elapsed:.4f}s",
                    flush=True,
                )
            if log_fh is not None:
                log_fh.write(json.dumps(row) + "\n")
                log_fh.flush()
            if wandb_run is not None:
                wandb_run.log(row, step=step)
            if checkpointing_enabled:
                assert checkpoint_dir is not None
                if args.checkpoint_interval > 0 and (step + 1) % args.checkpoint_interval == 0:
                    periodic_path = checkpoint_dir / f"step_{step + 1:08d}"
                    save_training_checkpoint(
                        periodic_path,
                        model,
                        optimizer,
                        step=step,
                        kind="periodic",
                        metrics=row,
                        resolved_config=resolved_config,
                        rng_state=rng.bit_generator.state,
                        dataset_rng_state=None if dataset is None else dataset.rng.bit_generator.state,
                        eval_rng_state=eval_rng.bit_generator.state,
                    )
                    checkpoint_paths[f"step_{step + 1:08d}"] = periodic_path
                    print(f"saved_checkpoint={periodic_path}", flush=True)
                if (
                    args.save_best_checkpoint
                    and "eval_loss" in row
                    and (best_eval_loss is None or row["eval_loss"] < best_eval_loss)
                ):
                    best_eval_loss = row["eval_loss"]
                    best_path = checkpoint_dir / "best"
                    save_training_checkpoint(
                        best_path,
                        model,
                        optimizer,
                        step=step,
                        kind="best",
                        metrics=row,
                        resolved_config=resolved_config,
                        rng_state=rng.bit_generator.state,
                        dataset_rng_state=None if dataset is None else dataset.rng.bit_generator.state,
                        eval_rng_state=eval_rng.bit_generator.state,
                    )
                    checkpoint_paths["best"] = best_path
                    print(f"saved_best_checkpoint={best_path} eval_loss={best_eval_loss:.6f}", flush=True)
    finally:
        if log_fh is not None:
            log_fh.close()
        if wandb_run is not None and first_metrics is None:
            wandb_run.finish()

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
    # Take the max across devices used by this run. With pure data parallel,
    # device profiles are similar but not identical (XLA workspace, mask
    # caches), and reporting only device 0 can hide the actual peak.
    peak_hbm_bytes = 0
    peak_reserved_bytes = 0
    per_device_peaks = []
    try:
        for d in local_devices[:num_devices]:
            stats = d.memory_stats() or {}
            in_use = int(stats.get("peak_bytes_in_use", 0) or 0)
            reserved = int(stats.get("peak_bytes_reserved", 0) or 0)
            per_device_peaks.append({"device": str(d), "peak_bytes_in_use": in_use, "peak_bytes_reserved": reserved})
            peak_hbm_bytes = max(peak_hbm_bytes, in_use)
            peak_reserved_bytes = max(peak_reserved_bytes, reserved)
    except Exception:
        pass
    performance_summary = {
        "compile_plus_first_step_sec": compile_time,
        "avg_measured_step_sec": avg_step,
        "tokens_per_sec": tokens_per_step / avg_step,
        "est_tflops": achieved_flops / 1e12,
        "mfu_percent": mfu * 100,
        "jax_peak_hbm_gb": peak_hbm_bytes / 1e9,
        "jax_peak_reserved_gb": peak_reserved_bytes / 1e9,
        "jax_per_device_peaks": per_device_peaks,
    }
    if checkpointing_enabled and args.save_final_checkpoint:
        assert checkpoint_dir is not None
        final_path = checkpoint_dir / "final"
        final_metrics = dict(last_row or {})
        final_metrics.update(performance_summary)
        save_training_checkpoint(
            final_path,
            model,
            optimizer,
            step=int(final_metrics.get("step", args.max_steps - 1)),
            kind="final",
            metrics=final_metrics,
            resolved_config=resolved_config,
            rng_state=rng.bit_generator.state,
            dataset_rng_state=None if dataset is None else dataset.rng.bit_generator.state,
            eval_rng_state=eval_rng.bit_generator.state,
        )
        checkpoint_paths["final"] = final_path
        print(f"saved_final_checkpoint={final_path}", flush=True)
    print(
        f"compile_plus_first_step={compile_time:.3f}s "
        f"avg_measured_step={avg_step:.4f}s "
        f"measured_from_step={measured_from} "
        f"tokens_per_sec={tokens_per_step / avg_step:.0f} "
        f"est_tflops={achieved_flops / 1e12:.1f} "
        f"mfu={mfu * 100:.1f}% "
        f"mfu_peak_tflops={args.peak_flops / 1e12:.0f} "
        f"jax_peak_hbm_gb={peak_hbm_bytes / 1e9:.2f} "
        f"jax_peak_reserved_gb={peak_reserved_bytes / 1e9:.2f}",
        flush=True,
    )
    if wandb_run is not None:
        summary_update = dict(performance_summary)
        summary_update.update({
            f"{kind}_checkpoint_path": str(path)
            for kind, path in checkpoint_paths.items()
            if kind in ("best", "final")
        })
        wandb_run.summary.update(summary_update)
        if checkpointing_enabled and args.wandb_checkpoints:
            for kind in ("best", "final"):
                path = checkpoint_paths.get(kind)
                if path is not None and path.exists():
                    upload_checkpoint_artifact(
                        wandb_run,
                        path,
                        kind=kind,
                    )
        wandb_run.finish()


if __name__ == "__main__":
    main()
