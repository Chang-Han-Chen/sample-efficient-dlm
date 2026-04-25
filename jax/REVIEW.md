# JAX Port Review — **Historical Snapshot (first pass)**

> **This document is a snapshot of the first-pass review of the JAX port.**
> Every bug listed here has since been fixed. Keep it around for the
> pedagogical explanations — the *why* of each pitfall is still useful —
> but do not treat it as a punch-list of outstanding work. For the
> current state of the port see [STATUS.md](./STATUS.md). The
> checkpointing explanation in [CHECKPOINTING.md](./CHECKPOINTING.md) is
> also still current.

---

High-level take first, then a line-by-line tour of each file with the
PyTorch original alongside. The goal is to explain *why* each difference
matters, not just "change this."

---

## Part 0 — Three cross-cutting issues worth internalizing first

These bugs recur throughout the port, so it is more useful to explain the
idea once than to re-explain it per file.

### A. `rngs` must be threaded through every constructor

In PyTorch, `nn.Module` owns a hidden global PRNG (the CPU / CUDA RNG).
You never pass it around — `nn.init.trunc_normal_(self.weight, ...)`
reaches for it implicitly. Flax NNX deliberately removes that hidden
state: every parameter that needs a random init must receive a
`nnx.Rngs` object explicitly, and constructors must *forward* it to their
submodules.

Your `core.py` already knows this — `Linear.__init__(self, rngs, ...)`,
`Embedding.__init__(self, rngs, ...)`, `SwiGLU.__init__(self, rngs, ...)`
— but the callers *higher up the tree* do not. For example:

```python
# jax/transformer/transformer.py
self.embedding = Embedding(vocab_size, d_model, dtype=dtype)
# -> vocab_size is being passed as `rngs`, d_model as `vocab_size`, …
```

The fix is the standard NNX pattern: every module takes `rngs` as a
keyword or first arg and splits/forwards it.

```python
class Block(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, d_model, n_heads, d_mlp, ...):
        self.ln1  = RMSNorm(rngs, d_model, ...)
        self.attn = MultiHeadSelfAttention(rngs, d_model, n_heads, ...)
        self.ln2  = RMSNorm(rngs, d_model, ...)
        self.ffn  = SwiGLU(rngs, d_model, d_mlp, ...)
```

Since `Rngs` is a stream you can reuse it inside the same model — Flax
forks it internally per call. So you do not need to manually split keys
between submodules; just pass the same `rngs` down.

### B. "Logical" tensor layout in JAX attention is `(B, T, H, D)`, not `(B, H, T, D)`

In the original PyTorch code, `MultiHeadSelfAttention` uses einops to
reshape into `(h*b, seq, head_d)` before RoPE. That layout has **seq as
the second-to-last axis**, which is what PyTorch's `scaled_dot_product_
attention` wants.

JAX/Flax is different. `nnx.dot_product_attention` (and
`jax.nn.dot_product_attention`) expect

```
query/key/value : (batch..., q_length, num_heads, qk_depth_per_head)
```

i.e. **heads is the second-to-last axis, seq is third-from-last**. Your
reshape

```python
q = q.reshape(B, T, self.n_heads, self.head_dim)  # correct for SDPA
```

is right for the SDPA call. But RoPE is now being handed a tensor in a
layout it was not designed for, and that is where the arithmetic silently
breaks (see § RoPE below).

Also — and this will explode at runtime — `nnx.dot_product_attention`
**does not take an `is_causal` kwarg**:

```python
>>> nnx.dot_product_attention(q, q, q, is_causal=True)
TypeError: dot_product_attention() got an unexpected keyword argument 'is_causal'
```

You have two clean options:

1. Use `jax.nn.dot_product_attention(q, k, v, is_causal=True)` directly
   — it supports the flag and Flax's wrapper will delegate to it anyway.
2. Build a boolean mask `mask = jnp.tril(jnp.ones((T, T), bool))` and
   pass it as `mask=mask` to `nnx.dot_product_attention`.

### C. `nnx.Param` weight layout affects weight tying

In PyTorch, `nn.Linear(in, out).weight` has shape `(out, in)` and the
forward is `x @ W.T`. Embedding weight has shape `(vocab, d_model)`.
`self.lm_head.weight = self.embedding.weight` works because both are
`(vocab, d_model)`.

In your JAX `Linear`, the weight is stored `(d_in, d_out)` and forward
is `x @ W`. So `lm_head.weight` has shape `(d_model, vocab)` — the
*transpose* of the embedding. Sharing them naively:

```python
if weight_tying:
    self.lm_head.weight = self.embedding.weight   # shape mismatch!
```

will either crash at the matmul or silently produce garbage. Fix the
lm_head forward to do `x @ self.weight.value.T` (and store the same
`(vocab, d_model)` tensor), or transpose at assignment time. The
cleanest option is the former: make weight tying the *only* path where
`Linear` uses the transposed matmul, or just implement the LM head as
`x @ embedding.weight.T` directly without a separate `Linear`.

---

## Part 1 — `rope.py`

Overall the structure is a faithful port. The math is correct in
isolation. The bug is in how the `cos`/`sin` tables get broadcast
against the attention tensor.

### Bug 1: head-axis insertion is for the wrong layout

```python
def _add_rope_head_axis(table, x):
    while table.ndim < x.ndim:
        table = jnp.expand_dims(table, axis=-3)
    return table
```

This repeatedly inserts a size-1 axis at position `-3`. If `x` has shape
`(B, T, H, D)` and the table starts as `(T, D/2)`, you end up with
`(1, 1, T, D/2)`. That would be right if the tensor layout were
`(B, H, T, D)` (PyTorch style), but the Flax attention layout is
`(B, T, H, D)`, so you need `(1, T, 1, D/2)` to broadcast correctly.
Concretely:

```
x1 :    (2, 8, 4, 16)
cos_b:  (1, 1, 8, 16)   # what your code produces
mul:    ValueError — incompatible broadcast
```

I reproduced this — it raises `mul got incompatible shapes for
broadcasting: (2, 8, 4, 16), (1, 1, 8, 16)`.

**Fix (simplest, most robust).** Stop guessing where to insert the head
axis and instead have the *caller* hand RoPE a tensor whose seq axis is
at `-2`. Example attention-side plumbing:

```python
# pre-rope: put heads before seq
q = jnp.transpose(q, (0, 2, 1, 3))  # (B, T, H, D) -> (B, H, T, D)
k = jnp.transpose(k, (0, 2, 1, 3))
q = self.rope(q, token_positions)
k = self.rope(k, token_positions)
# back to (B, T, H, D) for SDPA
q = jnp.transpose(q, (0, 2, 1, 3))
k = jnp.transpose(k, (0, 2, 1, 3))
```

With that, the RoPE file's existing `_add_rope_head_axis` logic
(`expand_dims(..., axis=-3)`) is correct.

**Alternative fix (keep layout, fix RoPE).** Insert the size-1 axis at
`-2` instead:

```python
def _add_rope_head_axis(table, x):
    while table.ndim < x.ndim:
        table = jnp.expand_dims(table, axis=-2)
    return table
```

Then `cos` shape goes `(T, D/2) -> (T, 1, D/2) -> (1, T, 1, D/2)`,
which broadcasts against `(B, T, H, D/2)`. This is the smaller diff but
makes RoPE file implicitly aware of the "seq-before-head" convention;
document it.

### Minor: `token_positions` indexing and dtype

```python
cos = self.cos[token_positions]
```

This works (I verified — `nnx.Variable` forwards `__getitem__`), but
being explicit via `.value` is a good habit:

```python
cos = self.cos.value[token_positions]
```

It makes intent clearer and avoids the rare case where Flax wraps the
proxy in an unexpected way (e.g. under `nnx.jit`).

### Micro: `jnp.asarray(theta_base, dtype=dtype) ** (-2.0 * j / d_k)`

Functionally identical to the PyTorch line `theta_base ** (-2 * j / d_k)`.
One subtle difference: your exponent `-2.0 * j / d_k` is computed in
`dtype` because `j` is cast to `dtype`. If you ever use `bfloat16` as
the table dtype, you will lose precision on the frequencies. The
PyTorch reference computes frequencies as float tensors because `j` is
`torch.long`. I would pin the cos/sin tables to `float32` and let
`__call__` cast on demand (which you already do for `x_f`). Your
default dtype is float32, so this is fine today.

---

## Part 2 — `core.py`

### Bug 1: `NameError: Array` at import time

```python
def _rms_normalize_last_dim(x: Float[Array, "..."], eps: float = 1e-5) -> ...
```

`Array` is not imported in this file. Either add `Array = jax.Array` near
the top (as in the other files) or just annotate with `jax.Array`.

### Bug 2: `softmax` vs `safe_softmax`

```python
# core.py defines:
def safe_softmax(...):

# but transformer.py imports:
from sample_efficient_gpt.transformer.core import ..., softmax
```

Rename one or the other. Conventionally you would just call it
`softmax` — the "safe" bit (subtracting max for numerical stability) is
standard and not worth distinguishing by name.

### Bug 3: `rngs.params.truncated_normal(...)` does not exist

```python
def _truncated_normal(rngs, shape, std, dtype=jnp.float32):
    return rngs.params.truncated_normal(-3, 3, shape=shape, dtype=dtype) * std
```

`rngs.params` is a `flax.nnx.rnglib.RngStream`. It has `count`, `fork`,
`key`, `tag` — but no `truncated_normal`. I verified this. The
idiomatic JAX version is:

```python
def _truncated_normal(rngs, shape, std, dtype=jnp.float32):
    return jax.random.truncated_normal(
        rngs.params(), lower=-3.0, upper=3.0, shape=shape, dtype=dtype
    ) * std
```

Note `rngs.params()` with parentheses — that call draws and advances a
fresh key. This mirrors the PyTorch init

```python
nn.init.trunc_normal_(self.linear.weight, std=sigma, a=-3*sigma, b=3*sigma)
```

where the truncation bounds are in *standardized units* (so ±3 in JAX,
pre-multiplying by sigma, matches ±3σ in the PyTorch trunc_normal).

### Bug 4: RMSNorm hard-codes a 3D input

```python
y = self.gamma.value[None, None, :] * x_float / rms * self.depth_scaling
```

By slicing `[None, None, :]` you are asserting the input is exactly 3D
`(B, T, d_model)`. The original PyTorch version does

```python
out = x * reverse_rms * self.gain
```

and relies on NumPy-style broadcasting against the last axis. Flax
works the same way, so just write:

```python
y = x_float * (self.gamma.value / rms) * self.depth_scaling
```

It is shape-polymorphic — works for `(B, T, d)`, `(B, T, H, d)`, or any
trailing-`d_model` input.

### Bug 5: `Linear` weight layout → breaks weight tying (see Part 0.C)

Your `Linear` forward is `x @ self.weight.value` with weight stored as
`(d_in, d_out)`. That is fine on its own, but it is the *transpose* of
`nn.Linear` in PyTorch (which stores `(out, in)` and does `x @ W.T`).
Weight-tying the LM head to the embedding (`(vocab, d_model)`) will
disagree. Pick one strategy and be consistent.

### Minor: `SwiGLU` fused-vs-split projection

PyTorch uses a single `up: d_model -> 2*d_ff` Linear and splits with
`chunk`. You use two separate projections (`w_up`, `w_gate`). Both are
mathematically identical after training; the PyTorch choice is a
throughput optimization (one matmul instead of two). For correctness
this does not matter. The PyTorch version also rounds
`d_ff = int(d_ff // 64 * 64)` for alignment; you dropped that rounding.
If you plan to load a checkpoint across the two, the round-down must
match.

### Minor: `LigerRMSNormFunction`, `LigerSiLUMulFunction`, `nvtx_range`

These are PyTorch / Triton / NVTX specific. They are unused in this
file and will at best fail to import cleanly when you actually run the
module. Delete the imports or guard them.

---

## Part 3 — `attention.py`

### Bug 1: `_MAX_SEQ_LEN` is undefined

```python
from sample_efficient_gpt.transformer.core import Linear, softmax
...
max_seq_len: int = _MAX_SEQ_LEN,     # NameError at class construction
```

Define it locally or import from core (where `_MAX_SEQ_LEN = 8192`
lives). Also, `softmax` does not exist in core (see § core.py).

### Bug 2: `Array` is used in type hints but never imported

Same as core.py. Add `Array = jax.Array`.

### Bug 3: `_rms_normalize_last_dim` is not imported

```python
if self.qknorm:
    q = _rms_normalize_last_dim(q) * self.qk_scale.value.astype(q.dtype)
```

It lives in `core.py`. Import it explicitly.

### Bug 4: `nnx.dot_product_attention(..., is_causal=True)` is not a thing

Verified — the Flax wrapper does not accept `is_causal`. Either

```python
attn = jax.nn.dot_product_attention(q, k, v, is_causal=True)
```

or

```python
mask = jnp.tril(jnp.ones((T, T), dtype=bool))  # (T, T)
attn = nnx.dot_product_attention(q, k, v, mask=mask)
```

The `jax.nn` call is usually what you want because it can dispatch to
cuDNN flash attention on GPU.

### Bug 5: SDPA output is 4D; `W_o` expects 3D

```python
attn = nnx.dot_product_attention(q, k, v, is_causal=True)  # (B, T, H, Dh)
return self.W_o(attn), v                                   # expects (B, T, D)
```

You are feeding a 4D tensor into `W_o`, whose weight is
`(d_model, d_model)`. The matmul will fail unless `head_dim == d_model`
(i.e. `n_heads == 1`). The PyTorch version uses einops to flatten
`(h b) seq head_d -> b seq (h head_d)` before `self.out`. Flatten
before the projection:

```python
attn = attn.reshape(B, T, D)   # (B, T, H, Dh) -> (B, T, H*Dh)
return self.W_o(attn), v_for_residual
```

### Bug 6: RoPE is called on `(B, T, H, Dh)` (see Part 0.B and Part 1)

Either transpose to `(B, H, T, Dh)` before/after RoPE (cleanest) or
change RoPE's head-axis insertion axis from `-3` to `-2`.

### Bug 7: `gating`, `qknorm`, `value_embedding` are half-wired

- `Block(..., attn_gating=attn_gating, ...)` in `transformer.py` passes
  the flag, but the JAX `MultiHeadSelfAttention.__init__` does not accept
  a `gating` kwarg at all (you do accept `value_residual` and `qknorm`).
  Passing it will raise `TypeError`.
- Even if you added it, the gating branches (elementwise / per-head /
  per-head-hd) are not ported.
- `attn_qknorm` also is not threaded from `Block` to
  `MultiHeadSelfAttention` — you accept it in `Block.__init__` but never
  pass it through. So `qknorm` is effectively dead code today.
- Same story for `value_embedding` in `Block` / `Transformer`.

### Semantic note: `v` returned before vs. after value-residual mix

```python
if v1 is None:
    v1 = v                      # capture pre-mix v

if self.value_residual:
    ...                          # v is overwritten by mixed value
attn = sdpa(q, k, v, ...)
return self.W_o(attn), v        # returns MIXED v
```

This matches the PyTorch reference: PyTorch also computes
`V = scale * (alpha1*V + alpha2*V1) * rsqrt(...)` and returns the mixed
`V`. The first layer is a no-op at initialization because alpha1=1,
alpha2=0, scale=1 — same as the PyTorch init. Good.

One subtlety: the original PyTorch code runs the "mix" formula
*unconditionally* (when `value_residual=False` it uses buffered
alpha1=1, alpha2=0, scale=1 so it is identity). Your code skips the mix
entirely when `self.value_residual` is False. Result is the same but
the JAX version is cleaner — fine.

### Micro: GQA expansion

```python
k = jnp.repeat(k, repeat, axis=2)
v = jnp.repeat(v, repeat, axis=2)
```

Correct for `(B, T, H_kv, Dh)` layout (axis 2 is the kv-heads axis).
One small perf nit: `jnp.repeat` materializes a new array; for large
contexts you can avoid the copy with `jnp.broadcast_to` after an
explicit `jnp.expand_dims`:

```python
k = jnp.broadcast_to(
    k[:, :, :, None, :],                 # (B, T, H_kv, 1, Dh)
    (B, T, self.n_kv_heads, repeat, Dh),
).reshape(B, T, self.n_heads, Dh)
```

Under `jax.jit` the two usually compile identically, so only worry
about this if a profile says so.

---

## Part 4 — `transformer.py`

### Bug 1: `final_norm` is applied twice

```python
x = self.final_norm(x)
return self.lm_head(self.final_norm(x))
```

Copy-paste error. Should be:

```python
x = self.final_norm(x)
return self.lm_head(x)
```

### Bug 2: checkpointed branch drops `token_positions` and `v1`, and does not return `v`

```python
if i < self.num_grad_checkpoint_layers:
    x = jax.checkpoint(block)(x)         # no v returned
else:
    x, v = block(x, token_positions=token_positions, v1=v1)

if v1 is None:
    v1 = v                               # NameError if first block was checkpointed
```

Two issues:
1. The checkpointed call needs the same inputs and outputs as the
   non-checkpointed call, otherwise `v1` is undefined on the first
   iteration if any layer is checkpointed.
2. `jax.checkpoint` applied to an `nnx.Module` callable does not
   re-materialize correctly because the module carries state as a
   pytree of `nnx.Variable`s. The NNX-friendly primitive is
   `nnx.remat`:

```python
block_fn = nnx.remat(Block, static_argnums=())
```

Or, simpler, split the module into graphdef + state and remat the
functional version. The cleanest pattern in current Flax is:

```python
from flax import nnx

if i < self.num_grad_checkpoint_layers:
    remat_block = nnx.remat(block)             # re-materializing wrapper
    x, v = remat_block(x, token_positions=token_positions, v1=v1)
else:
    x, v = block(x, token_positions=token_positions, v1=v1)
```

### Bug 3: imports that do not exist

- `from ... import ..., softmax` — not defined in your JAX `core.py`
  (only `safe_softmax`).
- `from ...attention import MultiHeadSelfAttention, KVCache` — you
  intentionally skipped KV cache, so `KVCache` does not exist in your
  JAX `attention.py`. The import will fail.
- `from sample_efficient_gpt.utils.profiling import nvtx_range` — this
  is a PyTorch profiling helper; it is fine to keep importing only if
  it is a pure context manager that no-ops on non-CUDA. Otherwise drop
  or stub it for the JAX path.

### Bug 4: `attn_qknorm` and `value_embedding` not forwarded

Same as in attention.py — `Block` accepts them but does not pass them
to `MultiHeadSelfAttention`, so they have no effect.

### Bug 5: top-p implementation has indexing and interface problems

```python
def _apply_top_p(self, logits, top_p=0.4):
    sorted_idx = jnp.argsort(logits, axis=-1)[:, :, ::-1]
    sorted_probs = jnp.take_along_axis(logits, sorted_idx, axis=-1)
    keep_sorted_id = jnp.cumsum(sorted_probs, axis=-1) <= top_p
    keep_sorted_id = keep_sorted_id.at[..., 0].set(True)

    batch_idx = jnp.arange(logits.shape[0])[:, None, None]
    keep = jnp.zeros_like(logits, dtype=bool).at[batch_idx, sorted_idx].set(keep_sorted_id)
    ...
```

Three problems, in increasing severity:

1. The variable is called `logits` but the caller should pass in
   *probs*. The PyTorch reference computes `softmax(logits, temp)` and
   then top-p's the probabilities. Rename for clarity.
2. `jnp.argsort` in JAX has no `descending` kwarg; your
   `[:, :, ::-1]` reverses only the last axis which is correct, but it
   hard-codes a 3D layout. Either call `jnp.argsort(-logits, axis=-1)`
   (negate to reverse) or keep the slice but document the rank.
3. The scatter is under-indexed:

   ```python
   keep = jnp.zeros_like(logits, dtype=bool)        # shape (B, S, V)
        .at[batch_idx, sorted_idx]                   # only 2 indices
        .set(keep_sorted_id)
   ```

   You index with `batch_idx` of shape `(B, 1, 1)` and `sorted_idx` of
   shape `(B, S, V)`. JAX interprets this as "pick the first two axes"
   and leaves the third (vocab) unindexed, so the write does not do
   what `scatter_(dim=-1, ...)` does in PyTorch. You also need a
   seq-axis index:

   ```python
   B, S, V = logits.shape
   b = jnp.arange(B)[:, None, None]
   s = jnp.arange(S)[None, :, None]
   keep = (jnp.zeros_like(logits, dtype=bool)
           .at[b, s, sorted_idx].set(keep_sorted_id))
   ```

   That is the correct JAX analogue of PyTorch's `scatter_(-1, sorted_idx, mask)`.

For training you do not need `generate` at all. If you only need it
working for eval, I would rewrite `_apply_top_p` to operate on a 2D
`(B, V)` slice (matching the PyTorch reference where it is called on
`last_prob`), which sidesteps the seq-axis issue entirely.

### Semantic note: Block returned 3 things in PyTorch

The PyTorch `Block.forward` returns `(y, avg_kurtosis, v)`, and
`Transformer.forward` collects the kurtosis values for logging. Your
JAX `Block` returns `(x, v)` and drops the kurtosis signal. That is
perfectly fine for correctness; it is a diagnostic-only channel, and if
you are not wired up to log it yet you can add it later. Just noting
because it is a silent behavior change.

### Micro: redundant `.value` reads in RMSNorm

Not wrong — but note that `nnx.Param`'s `__call__`-site arithmetic will
usually just work via the proxy; you are being conservative by writing
`.value` explicitly. That is a reasonable default style; keep it
consistent across the codebase.

---

## Part 5 — Suggested order to fix things

If you want a smoke-test to succeed quickly, here is a priority order:

1. Fix imports (`Array`, `softmax`/`safe_softmax`, `_MAX_SEQ_LEN`,
   `_rms_normalize_last_dim`, drop `KVCache`).
2. Fix `_truncated_normal` to use `jax.random.truncated_normal`.
3. Thread `rngs` through `Block`, `Transformer`,
   `MultiHeadSelfAttention` constructors.
4. Fix `nnx.dot_product_attention(..., is_causal=True)` → either use
   `jax.nn.dot_product_attention` or build a mask.
5. Fix RoPE head-axis broadcasting + reshape attention output to 3D
   before `W_o`.
6. Fix `final_norm` double-application in `Transformer.__call__`.
7. Decide on `Linear` layout and (if you want weight tying) fix the
   transpose mismatch against `Embedding`.
8. Optional / correctness-later: `nnx.remat` for checkpointing, top-p
   indexing fix, gating + qknorm + value_embedding wiring.

Once you have that working, writing a tiny `test_parity.py` that loads
the same weights into both the PyTorch and JAX models and compares
logits for a fixed input is a high-leverage safety net.

## Footnote — a latent bug in the original PyTorch `weight_tying`

While writing the parity test I noticed the original PT code does:

```python
self.lm_head.weight = self.embedding.weight
```

But `self.lm_head` is your custom `Linear` wrapper whose actual parameter
lives at `self.lm_head.linear.weight`. The forward uses `self.linear(x)`,
so the outer `.weight` attribute is ignored. Net effect: weight-tying on
the PyTorch side is a silent no-op. If you *want* tying, write
`self.lm_head.linear.weight = self.embedding.weight` on the PT side (and
your JAX port already does the right thing as long as `Linear.weight` is
the true parameter).
