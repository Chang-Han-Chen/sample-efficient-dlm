# JAX Port — Current Status

Last updated after three review rounds. This file is the source of truth
for "where is the port right now?" — `REVIEW.md` is a historical snapshot
of the first-pass review whose bugs have all been fixed.

## TL;DR

- The port trains. A 3-layer transformer with QK-norm, value-residual,
  `per-head` attention gating, and `num_grad_checkpoint_layers=1` runs
  one jitted train step end-to-end and the loss decreases.
- Every module has a PyTorch parity test. Max `|Δ|` across the suite
  sits at ≤ `~3 × 10⁻⁶` in float32 (roundoff).
- Activation checkpointing goes through `nnx.remat` and is parity-tested.

## Directory layout

```
jax/
  transformer/
    __init__.py            # re-exports Transformer
    core.py                # Linear, Embedding, RMSNorm, SwiGLU, softmax, rms_normalize_last_dim
    rope.py                # Split-half RoPE, seq axis at -2
    attention.py           # MultiHeadSelfAttention: GQA, QK-norm, value-residual, 3 gating modes
    transformer.py         # Block + Transformer (nnx.remat on first K layers)
  tests/
    parity_env.py          # sys.modules trickery so one Python process can host both trees
    test_parity.py         # Linear .. Transformer layer-by-layer parity against PT
    test_parity_extras.py  # weight tying, ln-scaling, qknorm, gating, remat-forward parity
    test_training.py       # jit + value_and_grad + nnx.Optimizer smoke; loss decreases
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
| Weight tying                     | yes     | yes (tied manually on both sides)    |
| LayerNorm scaling (depth)        | yes     | yes                                  |
| Activation checkpointing         | yes, via `nnx.remat` | forward parity + training smoke |
| KV cache / `generate()`          | **no** — training only                  | n/a                          |
| `value_embedding`                | **no** — raises `NotImplementedError`   | n/a                          |

## How to run the tests

```
python jax/tests/test_parity.py
python jax/tests/test_parity_extras.py
python jax/tests/test_training.py
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
