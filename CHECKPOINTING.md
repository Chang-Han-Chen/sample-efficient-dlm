# Activation checkpointing in JAX — high-level intuition and the NNX specifics

## 1. What checkpointing buys you (framework-agnostic)

Training's memory pressure comes from the *activations* that backprop needs
to reuse. In a vanilla forward pass, every tensor you compute is kept
around until its gradient is done with it. For a transformer block the
peak memory is roughly `n_layers * (per-layer activations)`.

**Activation checkpointing (aka rematerialization / "remat"):** don't
store the intermediate activations. Store only the *inputs* to the
checkpointed region. During the backward pass, when those intermediates
are needed, recompute the forward pass of that region from the saved
input. You pay extra FLOPs for saved memory — typically an extra forward
per checkpointed region, which for a transformer block is a ~33%
compute overhead for a large memory win.

You usually turn it on selectively: for example, checkpoint the first
`K` blocks of your `n_layers`-layer transformer, because those are the
ones whose activations cost the most to keep around (activations for
later blocks get freed earlier).

## 2. The PyTorch reference

Your PT code does:

```python
from torch.utils.checkpoint import checkpoint

if i < self.num_grad_checkpoint_layers:
    def layer_forward(x, v1, kv_cache):
        return layer(x, v1=v1, kv_cache=kv_cache)
    x, avg_kurtosis, v = checkpoint(
        layer_forward, x, v1,
        kv_cache[i] if kv_cache is not None else None,
        use_reentrant=False,
    )
else:
    x, avg_kurtosis, v = layer(x, v1=v1, kv_cache=...)
```

PyTorch's `torch.utils.checkpoint` takes a *callable* and its inputs. It
runs the forward normally but does not save the intermediate activations;
at backward time it re-runs the callable with the saved inputs to
reconstruct what autograd needs. The `use_reentrant=False` flag
activates a newer autograd path that plays nicer with things like
`torch.compile` and is generally safer.

## 3. JAX's primitive: `jax.checkpoint` (aka `jax.remat`)

In pure JAX, the equivalent is:

```python
import jax

def layer_fn(x, v1):
    return layer(x, v1=v1)

rematted = jax.checkpoint(layer_fn)        # same as jax.remat(layer_fn)
y, v = rematted(x, v1)
```

`jax.checkpoint` wraps a *pure function* of arrays-in, arrays-out. At
tracing time (under `jit`) it inserts a rematerialization marker: the
forward graph computes and returns `y` as normal, but the backward graph
is rewritten to recompute `layer_fn(x, v1)` from scratch rather than
pull intermediate values out of the forward's saved activations.

There are three important properties:

1. **`jax.checkpoint` requires a pure function**, meaning everything it
   reads or writes must flow through its arguments and return value.
   Hidden state — like module parameters — must be made explicit.
2. **It does nothing outside a differentiation context.** On a plain
   forward it is effectively the identity; the saving / recomputing only
   kicks in when the enclosing function is being transformed with
   `jax.grad`.
3. **`policy`** is a knob that decides which intermediate values are OK
   to keep vs. must be recomputed. Use this to control the
   memory/compute tradeoff precisely; e.g. `jax.checkpoint_policies.dots_with_no_batch_dims_saveable`.

## 4. The NNX wrinkle — why `jax.checkpoint(block)(x)` is awkward

Your model is an `nnx.Module`. Its parameters (`nnx.Param`) and buffers
(`nnx.Variable` subclasses) are *attached to the module*, not passed in
as arguments. When you write

```python
rematted = jax.checkpoint(block)   # block is an nnx.Module instance
y, v = rematted(x)
```

JAX sees only `x` in the input signature. The module's parameters enter
the computation through closure rather than as explicit inputs, which
means:

- `jax.checkpoint` can't see them as inputs to save/re-supply at
  recompute time.
- Under `jit` + `grad`, NNX's graph/state split is what makes this work
  at all — the params get threaded as closure state, but `jax.checkpoint`
  doesn't know that.

What you want is: treat the call to `block` as a pure-function call on
`(state, x)` where `state` is the pytree of all parameters/buffers,
remat that pure function, then plug the state back into the module.

## 5. `nnx.remat` does the split-remat-merge dance for you

```python
from flax import nnx

rematted = nnx.remat(block)                 # same signature as block(...)
x, v = rematted(x, token_positions, v1)     # works like block(...)
```

Internally `nnx.remat` does roughly:

```python
def nnx_remat(mod):
    graphdef, state = nnx.split(mod)        # make it functional
    def pure_fn(state, *args, **kwargs):
        m = nnx.merge(graphdef, state)      # reconstitute a module
        return m(*args, **kwargs)
    checkpointed = jax.checkpoint(pure_fn)  # now the state is an explicit arg
    def wrapped(*args, **kwargs):
        return checkpointed(nnx.state(mod), *args, **kwargs)
    return wrapped
```

(The real implementation is in `flax.nnx.transforms.remat` and handles
state mutation correctly; the sketch above is the mental model.)

That is what's happening in your JAX transformer now:

```python
for i, block in enumerate(self.blocks):
    if i < self.num_grad_checkpoint_layers:
        rematted = nnx.remat(block)                   # remat wrapper
        x, v = rematted(x, token_positions, v1)
    else:
        x, v = block(x, token_positions, v1)
```

Under `nnx.jit` + `nnx.grad` (or `nnx.value_and_grad`), the first
`num_grad_checkpoint_layers` blocks have their forward recomputed during
backward; the rest behave like normal layers.

## 6. Where to put the remat wrapper — two styles

There are two common patterns. Yours is the first.

**(a) Wrap at call time, inside the loop.** Simple, and gives you a
runtime knob (`num_grad_checkpoint_layers`). Cost: `nnx.remat` runs on
every forward call, but it's just a transform over the graphdef — it
does not re-trace the underlying module. Under `jit` the compilation
cost is paid once.

**(b) Wrap at construction time.** Make the block class itself
rematted:

```python
RematBlock = nnx.remat(Block)                        # class-level
# later…
self.blocks = [RematBlock(rngs, d_model, …) if i < K else Block(…)
               for i in range(n_layers)]
```

This is a tiny bit cleaner if you never flip the flag at runtime.

## 7. Controlling what gets saved: `policy`

For a real production model you typically don't want "recompute
*everything* inside the block." Large matmul outputs (the attention
score matrix, the FFN intermediate) are the expensive activations
memory-wise; bias adds and layernorm outputs are cheap.

`nnx.remat` accepts the same `policy` argument that `jax.checkpoint`
takes:

```python
rematted = nnx.remat(block, policy=jax.checkpoint_policies.nothing_saveable)
# or the common "save the big matmul outputs, recompute everything else":
rematted = nnx.remat(
    block,
    policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
)
```

`nothing_saveable` is the most aggressive (recomputes literally
everything inside the block). `dots_with_no_batch_dims_saveable` keeps
the outputs of reduction-like dots around, which is often the right
sweet spot. See the `jax.checkpoint_policies` module for the full
menu — they're all "keep *these* values, drop *those*" predicates.

## 8. A tiny mental model to take away

- `jax.checkpoint` : remat for a pure function. The primitive.
- `nnx.remat`     : remat for a stateful NNX module. Wraps
  `jax.checkpoint` with the state split/merge you'd otherwise write by hand.
- Memory ↔ compute knob: `policy=…` decides which intermediates are
  worth keeping; more aggressive = less memory, more FLOPs during
  backward.

In your code today, `nnx.remat(block)` inside the transformer loop is
the clean, minimal version. The JAX-only tests under `tests/` are designed to
verify this: `test_transformer_stack.py` checks that the forward with
`num_grad_checkpoint_layers=2` matches the uncheckpointed forward to float32
roundoff, and runs a few `nnx.jit` + `nnx.value_and_grad` steps with
checkpointing on to confirm gradients flow through the remat boundary. Rerun
those after any structural changes to the `Transformer`.
