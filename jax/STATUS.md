# JAX Port — Current Status

Last updated after the value-embedding/mask backbone pass. This file is the
source of truth for "where is the port right now?" — `REVIEW.md` is a
historical snapshot of the first-pass review whose bugs have all been fixed.

## TL;DR

- The port trains. A 3-layer transformer with QK-norm, value-residual,
  `per-head` attention gating, and `num_grad_checkpoint_layers=1` runs
  one jitted train step end-to-end and the loss decreases.
- Every module has a PyTorch parity test. Max `|Δ|` across the suite
  sits at ≤ `~3 × 10⁻⁶` in float32 (roundoff).
- Activation checkpointing goes through `nnx.remat` and is parity-tested.
- The backbone now has the planned value-embedding path, explicit attention
  masks for diffusion, selectable JAX attention implementation, and an
  `encode()` path that returns final hidden states without logits.
- A first AR training stack exists: pure-JAX CE/z-loss, memory-mapped token
  loader, jitted train/eval steps with gradient accumulation, and a custom
  NNX/Optax NorMuonCWD+AdamW transform with PyTorch-style parameter grouping.
- A ClimbMix smoke pipeline exists and has been exercised end-to-end: one
  train shard plus one validation shard, a 32768-token BPE tokenizer, batch
  inspection at several batch sizes, tokenizer sanity checks, and a monitored
  one-batch A100 overfit run.
- `jax/train_ar.py` now defaults to tied token embedding / LM-head weights
  (`--no-weight-tying` is available for explicit untied ablations).

## Directory layout

```
jax/
  transformer/
    __init__.py            # re-exports Transformer
    core.py                # Linear, Embedding, RMSNorm, SwiGLU, softmax, rms_normalize_last_dim
    rope.py                # Split-half RoPE, seq axis at -2
    attention.py           # MultiHeadSelfAttention: GQA, QK-norm, value-residual, value embeddings, masks, 3 gating modes
    masks.py               # AR/block-causal/BD3 dense boolean attention masks
    transformer.py         # Block + Transformer (nnx.remat on first K layers)
  training/
    data.py                # Memory-mapped `.npy` token batches
    loss.py                # AR CE + z-loss
    optimizer.py           # NorMuonCWD+AdamW Optax transform
    step.py                # jitted train/eval/gradient-accumulation steps
  data/
    prepare_climbmix.py    # ClimbMix parquet download, BPE train, tokenization
    loader.py              # batch inspection helpers
    inspect_loader.py      # batch sampling sanity CLI
    inspect_tokenizer.py   # tokenizer/vocab sanity CLI
  train_ar.py              # minimal AR trainer/throughput smoke entrypoint
  tests/
    parity_env.py          # sys.modules trickery so one Python process can host both trees
    test_parity.py         # Linear .. Transformer layer-by-layer parity against PT
    test_parity_extras.py  # weight tying, ln-scaling, qknorm, gating, remat, value embeddings, masks
    test_training.py       # jit + value_and_grad + nnx.Optimizer smoke; loss decreases
    test_training_stack.py # NorMuon+AdamW grouping + fixed-batch loss decrease
  CHECKPOINTING.md         # Pedagogical writeup of jax.checkpoint vs nnx.remat
  REVIEW.md                # HISTORICAL — first-pass code review (bugs fixed, notes kept for pedagogy)
  STATUS.md                # ← this file
```

## Feature coverage vs. the PyTorch repo

| Feature                          | Ported? | Parity-tested?                       |
|----------------------------------|---------|--------------------------------------|
| `Linear`, `Embedding`, `RMSNorm` | yes     | yes                                  |
| `SwiGLU`                         | yes     | yes (fused 2×d_ff split into up/gate)|
| Split-half RoPE                  | yes     | yes, incl. `B != H` + token_positions |
| MHSA, GQA (`n_kv_heads`)         | yes     | yes                                  |
| QK-norm (`attn_qknorm=True`)     | yes     | yes                                  |
| Value residual                   | yes     | yes (alpha1/alpha2/scale learnable)  |
| Attention gating                 | yes     | yes — `elementwise`, `per-head`, `per-head-hd` |
| Weight tying                     | yes     | yes (constructor sharing + PT manual parity) |
| LayerNorm scaling (depth)        | yes     | yes                                  |
| Activation checkpointing         | yes, via `nnx.remat` | forward parity + training smoke |
| KV cache / `generate()`          | **no** — training only                  | n/a                          |
| `value_embedding`                | yes — Option D ordering from PLAN       | zero-scale/no-op + placement |
| Attention mask override          | yes — boolean DPA mask + causal override| BD3 mask helper tests        |
| Hidden-state return before logits| yes — `encode()` / `return_hidden=True` | shape/API test |
| AR CE + z-loss                   | yes — full-logit pure JAX              | training stack smoke |
| NorMuonCWD+AdamW                 | first pass — native JAX/Optax          | grouping + loss decrease |
| AR training loop                 | first pass — `jax/train_ar.py`         | synthetic + ClimbMix smoke |
| Gradient accumulation            | yes — update once over microbatch axis | synthetic smoke |
| ClimbMix data/tokenizer          | first pass — one-shard smoke + metadata| loader/tokenizer inspection |
| PyTorch/JAX short training parity| **no**                                 | n/a |
| A100 profiling                   | first smoke only                       | monitored one-batch overfit |

## How to run the tests

```
python jax/tests/test_parity.py
python jax/tests/test_parity_extras.py
python jax/tests/test_training.py
python jax/tests/test_training_stack.py
python jax/data/inspect_loader.py data/climbmix_smoke/tokens/train --context-length 128 --batch-sizes 1,4,16 --samples-per-size 3
python jax/data/inspect_tokenizer.py data/climbmix_smoke/tokenizer/tokenizer.json --metadata data/climbmix_smoke/tokenizer/metadata.json --token-bytes data/climbmix_smoke/tokenizer/token_bytes.npy
```

The parity tests flip `sys.modules["sample_efficient_gpt"]` between the
PT tree (repo root) and the JAX tree (`jax/`). Because both trees use
the same top-level package name, only one can be live in a given Python
process at a time; `parity_env.py` handles the swap.

## Known wrinkles

1. **`nnx.Optimizer.update` signature changed across flax versions.**
   flax ≥ 0.11 expects `optimizer.update(model, grads)`; flax 0.10 and
   earlier took only `optimizer.update(grads)`. `test_training.py`
   inspects the signature once at import time and dispatches accordingly.
2. **`nnx.List` is the preferred container for submodule lists.** In
   flax ≥ 0.11 a plain Python list assigned to a module attribute is
   rejected. `transformer.py` does `_ModuleList = getattr(nnx, "List", list)`
   so the file works on both old and new flax.
3. **The PT repo's `weight_tying=True` is a silent no-op upstream.** The
   line `self.lm_head.weight = self.embedding.weight` assigns to the
   outer `Linear` wrapper; the actual parameter lives at
   `self.lm_head.linear.weight`. The JAX port does tie (the param is
   directly on the module), so to compare apples-to-apples the parity
   test manually ties both sides. See the footnote in
   [REVIEW.md](./REVIEW.md) for the full story.
4. **Value embeddings follow the PLAN's Option D ordering.** The attention
   layer caches the raw first-layer `V_l` stream before token value embeddings,
   applies the value-residual mixture to the raw streams, and only then adds
   `value_embedding_scale * gate(h_l) * E_l(ids)`.
5. **The current AR trainer is not the final profiling harness.** It is enough
   to train and time real `.npy` token batches and supports gradient
   accumulation, JSONL logs, eval batches, fixed-batch overfit mode, and
   explicit steady-step measurement. It still lacks checkpointing, W&B logging,
   config resolution, and PyTorch/JAX same-data training parity.
6. **JAX CUDA requires running Python with GPU permission in this environment.**
   Inside the default sandbox, JAX reports only `cpu:0` and logs a CUDA init
   error. Escalated Python sees `cuda:0`. `train_ar.py` sets
   `XLA_PYTHON_CLIENT_PREALLOCATE=false` before importing JAX so `nvidia-smi`
   memory readings are not dominated by default preallocation.
7. **Tied embeddings are the JAX training default.** The transformer constructor
   replaces `lm_head.weight` with the exact same `nnx.Param` object as
   `embedding.weight`; NNX state then exposes only `embedding.weight`, so the
   optimizer does not see a duplicate tied parameter.
8. **Smoke-run notes.** The ClimbMix smoke dataset lives under
   `data/climbmix_smoke/` and is ignored by git. A small A100 overfit run
   (`batch_size=8`, `context_length=128`, `d_model=128`, `n_layers=2`) reduced
   fixed-batch loss from `10.87` to `~0.005`; the monitor showed about `1 GB`
   HBM rather than full-device preallocation.
9. **Initial A100 profile notes.** All runs below used tied embeddings,
   `vocab_size=32768`, `d_model=768`, `d_ff=2048`, `n_layers=8`, `n_heads=12`,
   bf16, NorMuonCWD+AdamW, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, and an A100
   SXM BF16 dense-peak MFU denominator of `312e12` FLOP/s. The FLOP estimate
   counts the tied LM projection as compute even though it is not a separate
   trainable parameter.

   - `seq_len=512`, `batch_size=16`, `attention_impl=cudnn`: peak HBM
     `~6.6 GB`, `0.0747 s/step`, `109.6k tok/s`, `74.5 TFLOP/s`, `23.9% MFU`.
   - `seq_len=1024`, `batch_size=16`, `attention_impl=cudnn`: peak HBM
     `~11.7 GB`, `0.1050 s/step`, `156.0k tok/s`, `111.9 TFLOP/s`, `35.9% MFU`.
   - `seq_len=2048`, `batch_size=16`, `attention_impl=cudnn`: peak HBM
     `~23.0 GB`, `0.1771 s/step`, `185.0k tok/s`, `146.7 TFLOP/s`, `47.0% MFU`.
   - With `seq_len=512`, increasing batch size improved utilization:
     `batch_size=64` reached `~41.4 GB` peak HBM, `154.8k tok/s`,
     `105.2 TFLOP/s`, and `33.7% MFU`; `batch_size=96` reached `~61.3 GB`,
     `169.5k tok/s`, `115.2 TFLOP/s`, and `36.9% MFU`.
   - The `512 -> 1024 -> 2048` fixed-batch memory trend is close to linear,
     not quadratic, which is consistent with cuDNN memory-efficient attention.
