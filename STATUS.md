# Current Status

The repo is now a root-level JAX training codebase. The old PyTorch,
`baby-dLM`, Karpathy, and package-parity trees have been removed; historical
notes remain in `PROGRESS.md` and `REVIEW.md`, but active code should live in
the directories below.

## Directory Layout

```
transformer/
  core.py                # Linear, Embedding, RMSNorm, SwiGLU, value embeddings
  rope.py                # Split-half RoPE
  attention.py           # GQA, QK-norm, value residuals, value embeddings, BD3 blocked attention
  masks.py               # AR/block-causal/BD3 dense boolean masks
  transformer.py         # Block + Transformer, including nnx.remat
training/
  data.py                # Memory-mapped .npy token batches
  diffusion.py           # MDLM/BD3LM batch construction and model contexts
  loss.py                # AR/supervised CE + z-loss, full and chunked paths
  optimizer.py           # NorMuonCWD+AdamW Optax transform
  step.py                # JIT/pmap train and eval steps
configs/
  data/                  # Dataset metadata used by config validation
  experiments/           # Canonical train_ar.py experiment configs
  model/ optimizer/ schedule/
tests/
  test_configs.py
  test_data.py
  test_transformer_stack.py
  test_training_stack.py
  test_diffusion_stack.py
  test_data_parallel.py
train_ar.py              # Main training entrypoint
inspect_diffusion.py     # Human-readable diffusion inspection CLI
```

## What Is Covered

- Transformer construction, weight tying, remat, QK-norm, gating, value
  embeddings, hidden-state return, and BD3 masks have JAX-only tests.
- Training stack tests cover parameter grouping, NorMuon+AdamW updates,
  fixed-batch loss decrease, chunked CE, gradient accumulation, and checkpoint
  round trips.
- Diffusion tests cover MDLM/BD3LM masking, dense vs blocked BD3 attention,
  supervised loss masking, and clean-stream leakage behavior.
- Data-parallel tests compare single-device and pmapped JAX paths. They skip
  under normal one-device pytest runs unless two CPU/GPU devices are visible.
- Config tests parse every experiment config and verify `base_configs` point to
  existing root-level files.

## Common Commands

```bash
python train_ar.py --synthetic --max-steps 20 --no-wandb
python train_ar.py --config configs/experiments/ar_baseline.yaml
python -m pytest tests/test_configs.py tests/test_data.py -q
python -m pytest tests/test_transformer_stack.py tests/test_training_stack.py tests/test_diffusion_stack.py -q
XLA_FLAGS=--xla_force_host_platform_device_count=2 python -m pytest tests/test_data_parallel.py -q
```

## Notes

- `train_ar.py` reads only the top-level `train_args` mapping from experiment
  YAMLs. `base_configs` is metadata for humans and validation tests; it is not
  merged into the runtime args yet.
- The default persistent compilation cache remains
  `/tmp/sample_efficient_gpt_jax_cache`.
- The tracked test suite no longer depends on PyTorch or the removed parity
  package namespace.
