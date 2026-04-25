# Flax NNX and Optax — what I learned porting this project

This document is my synthesis of the concrete bugs I hit in the JAX port,
the current Flax NNX / Optax documentation, and the 0.10 → 0.11 → 0.12
migration notes. It is written to be the thing I wish I had read *before*
starting the port.

Target audience: someone comfortable with PyTorch, reasonably fluent in
JAX (`jit`, `grad`, `vmap`, pytrees), but new to Flax NNX.

Sources are cited at the bottom.

---

## 1. The one-sentence mental model

> In modern Flax NNX, a `Module` **is a pytree**. Everything else — the
> rules about `nnx.data`, `nnx.List`, the `Optimizer` signature change,
> the ban on in-place mutation — falls out of making that one idea
> airtight.

If you remember nothing else, remember: NNX used to be a handwavy "object
with parameters attached" and secretly handed the framework a
functional view when you called `nnx.split`. Since **0.11** modules are
native pytrees; since **0.12** the pytree contract is *strict*. Every
ergonomic change downstream is JAX reasserting itself: if a module is a
pytree, it must behave like a pytree — values not references,
immutable not in-place, static vs. data axes explicit not inferred.

---

## 2. What changed, and why — the version-by-version story

### 2a. NNX 0.10 → 0.11: Modules became pytrees

> "NNX modules are now Pytrees. This means that you can use them with
> JAX transforms like `jax.vmap` and `jax.jit` directly." — *0.10→0.11
> migration guide*

Before 0.11, you'd write `nnx.split(model)` to pull out `(GraphDef,
State)`, apply a JAX transform to a function of the state, then
`nnx.merge` to put it back. Now the `Module` object itself is a
pytree; the transforms know how to traverse it. `nnx.jit` still exists
and provides "automatic state management," but for simple cases `jax.jit`
works directly.

**Practical consequence for me:** the split/merge dance I described in
my CHECKPOINTING.md is still a correct mental model, but **you don't
have to write it by hand** any more. `nnx.remat(block)` internally does
`split → jax.checkpoint(pure_fn) → merge`, but under the hood the
traversal uses the module's own pytree registration.

### 2b. NNX 0.12: the strict-pytree contract

This is the one that bit me hardest (twice).

> "`nnx.Pytree` and therefore `nnx.Module` are now stricter with regards
> to attributes that contain Arrays and changing the status of
> attributes." — *0.12.0 release notes*

The rules, in order of how often they matter:

1. **Every attribute is classified as `data` or `static`** at first
   assignment, and the status sticks.
2. **What's data by default**: Arrays, `ArrayRef`s, `nnx.Variable`s
   (including `nnx.Param`), anything registered via
   `nnx.register_data_type`, and any *pytree that contains* one of
   these.
3. **What's static by default**: everything else (ints, strs, Python
   classes, etc.).
4. **Plain Python containers that contain arrays/modules** are the
   sharp edge: a plain `list` of `Module`s or `Array`s is **not**
   automatically recognized as a data pytree. You have to say so
   explicitly.

> "JAX pytree structures that contain Arrays now have to be marked with
> `nnx.data`. Alternatively, if the container pytree is a list or a
> dict, you can use `nnx.List` or `nnx.Dict`." — *0.12.0 release notes*

**The exact error I hit**: `ValueError: Found data on value of type
list assigned to static attribute blocks`. That phrasing — "data on
value of a list assigned to static attribute" — is NNX 0.12 saying
"you gave me a `list` whose contents look like data, but by default I
classified the attribute as static, and I refuse to guess."

**The fix options, ordered by how much I like them:**

```python
# (a) Preferred: nnx.List, which is explicitly "a list of data".
self.blocks = nnx.List([Block(...) for _ in range(n)])

# (b) Equivalent, more verbose, but works for arbitrary containers:
self.blocks = nnx.data([Block(...) for _ in range(n)])

# (c) Type-annotation form, if you're using dataclass-style fields:
class T(nnx.Module):
    blocks: nnx.Data[list]
    def __init__(self, …):
        self.blocks = [Block(...) for _ in range(n)]
```

Note: `nnx.Data[T]` / `nnx.Static[T]` **type annotations are
deprecated** in 0.12. Use `nnx.dataclass` with `nnx.data` / `nnx.static`
as field descriptors instead.

### 2c. NNX 0.11: Optimizer lost its model reference

The signature change I hit:

```python
# v0.10 (old):
optimizer = nnx.Optimizer(model, optax.adam(1e-3))
optimizer.update(grads)

# v0.11+ (current):
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
optimizer.update(model, grads)
```

From the release notes:

> "The Optimizer abstraction no longer holds a reference to the model to
> avoid reference sharing; instead the model must be provided as the
> first argument to update." — *0.11.0 release notes*

**What "reference sharing" means here, and why it matters.** Once
modules are pytrees, JAX transforms treat them as *values*. But the
old `nnx.Optimizer(model)` stashed a *reference* to the model. Pass
both the optimizer and the model into `jit`, and the framework now has
two independent pytree copies for what was originally one Python
object. Updates done through `optimizer.update(grads)` would mutate the
optimizer's copy; the model you handed to the loss function would drift
out of sync. Making the update a pure function `(model, grads) ->
new_model_state` removes the ambiguity entirely.

**Additional requirement** in 0.11+: `wrt=nnx.Param` (or whichever
`Variable` subclass you're optimizing) is no longer optional. You're
being explicit about which subset of the module's variables should be
treated as trainable.

### 2d. NNX 0.11+: in-place mutation is banned

> "In-place operators will now raise an error. This is done as part of
> the push for Variables to be compatible with Tracer semantics." —
> *0.12.0 release notes*

JAX Tracers can't be mutated. When Variables stand in for Tracers
inside `jit`, allowing `var *= 2` in eager code but not traced code
would be a nasty inconsistency. So it's forbidden unconditionally now.
You write:

```python
self.counter.value = self.counter.value + 1    # OK
self.counter.value += 1                         # error
```

This didn't affect my port directly, but it explains why some older
tutorials' code patterns no longer work.

### 2e. Transforms: `nnx.remat`, `nnx.jit`, …

Because modules are pytrees, `jax.jit(my_nnx_fn)` and
`jax.checkpoint(my_nnx_fn)` *can* work directly when the nnx module is
passed as an explicit argument. But there are two reasons to prefer
the `nnx.*` wrappers:

1. **Automatic state management.** `nnx.jit` detects when variables
   change between calls and handles in-place-looking mutation on your
   behalf (internally it's doing purely-functional updates).
2. **Ergonomics.** You can call `block(x)` instead of threading state
   through an extra argument.

The catch: `nnx.jit` has measurable overhead on tiny models. For a
large transformer the overhead is in the noise.

---

## 3. Optax — the stable half of the stack

Unlike NNX, optax itself has barely changed API-wise. The canonical
loop from the [getting-started
page](https://optax.readthedocs.io/en/latest/getting_started.html) is:

```python
for _ in range(N):
    grads = jax.grad(compute_loss)(params, xs, ys)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
```

Two things worth internalizing:

1. **A `GradientTransformation` is `(init, update)`** — init maps
   `params -> state`, update maps `(grads, state, params?) -> (new_grads,
   new_state)`. The transformation is *not responsible for applying*
   the update to the params. That's what `optax.apply_updates` is for.
   The separation is what makes `optax.chain` clean:

   ```python
   my_optimizer = optax.chain(
       optax.clip_by_global_norm(1.0),
       optax.scale_by_adam(eps=1e-4),
       optax.scale(-learning_rate),
   )
   ```

   Each step is an independent `GradientTransformation` being composed;
   `apply_updates` happens once at the end of the pipeline.

2. **Schedules are just transformations too.** Plug into an optimizer
   via `scale_by_schedule(sched)` or by passing the schedule fn as the
   `learning_rate` argument: `optax.adam(learning_rate=warmup_cosine_decay_schedule(...))`.

**The NNX–optax boundary.** `nnx.Optimizer` is the only piece that
lives at the boundary. optax itself makes no mention of NNX and
doesn't need to — you can use plain optax with any pytree of params.
Which is why the signature change in `nnx.Optimizer.update` is an
NNX-side thing, not an optax-side thing.

---

## 4. Style rules I'd follow going forward

Distilled from everything above plus the hands-on debugging:

1. **Assume NNX 0.12 semantics; target it.** Use `nnx.List` and
   `nnx.data` explicitly. Do not rely on auto-classification even when
   you get away with it on older versions.
2. **Thread `rngs` through every constructor.** Never close over
   random state from outside. Use `rngs.params()` (the call with
   parens) to draw a key when you need one.
3. **Build params in `float32` regardless of compute dtype.** Cast at
   the use site. The PyTorch-style trick of initializing directly in
   bf16 is a correctness footgun and the docs are not your oracle for
   numeric precision.
4. **`optimizer.update(model, grads)` always.** If you need cross-version
   support, do `inspect.signature(nnx.Optimizer.update)` once at
   module-load time and dispatch. Never try/except inside a
   jit-compiled function — that re-traces.
5. **Prefer `nnx.remat` over hand-rolled `jax.checkpoint` for stateful
   modules.** It's a thin wrapper but it gets the state split/merge
   right, which is easy to screw up by hand.
6. **Use relative imports inside your JAX package.** If your JAX port
   lives in a sibling directory of the PyTorch code, and both reuse
   the same top-level package name, absolute imports turn into a
   sys.modules minefield.
7. **Separate dtype for compute, dtype for params, dtype for RoPE
   tables.** Three different axes that want different defaults; a
   unified `dtype=…` arg is a trap when you want mixed precision.
8. **When a flax version smells off, read the release notes first.**
   Both issues that embarrassed me across review rounds were explicitly
   called out as breaking changes in the 0.11 and 0.12 release posts.
   If I'd read those first I'd have saved two review cycles.

---

## 5. Cheatsheet

| You want to… | Old-NNX (0.10) | Current (0.12) |
|---|---|---|
| Declare a list of submodules | `self.blocks = [Block(), …]` | `self.blocks = nnx.List([Block(), …])` |
| Declare a buffer (non-trainable) | `class Buf(nnx.Variable): pass; self.x = Buf(arr)` | same |
| Random init | `rngs.params.truncated_normal(...)` *(never worked)* | `jax.random.truncated_normal(rngs.params(), -3, 3, shape, dtype)` |
| Build optimizer | `opt = nnx.Optimizer(m, optax.adam(lr))` | `opt = nnx.Optimizer(m, optax.adam(lr), wrt=nnx.Param)` |
| Apply update | `opt.update(grads)` | `opt.update(model, grads)` |
| Grad-checkpoint a block | `jax.checkpoint(block)(x)` *(awkward w/ state)* | `nnx.remat(block)(x)` |
| Causal attention | `nnx.dot_product_attention(q,k,v,is_causal=True)` *(fails)* | `jax.nn.dot_product_attention(q,k,v,is_causal=True)` |
| Convert module → pytree manually | `nnx.split(m)` / `nnx.merge(g, s)` | still works; rarely needed now |

---

## Sources

- [Flax NNX basics](https://flax.readthedocs.io/en/latest/nnx_basics.html) — current `nnx.Optimizer.update(model, grads)`, `nnx.jit`, state management
- [Flax Module & Pytree guide](https://flax.readthedocs.io/en/stable/guides/pytree.html) — `nnx.data`, `nnx.List`, `nnx.Dict`, static vs data
- [Flax 0.10 → 0.11 migration](https://flax.readthedocs.io/en/latest/migrating/nnx_010_to_nnx_011.html) — Optimizer signature change, `wrt=` becoming required, modules-as-pytrees
- [Flax 0.12.0 release notes](https://github.com/google/flax/discussions/4984) — strict pytree attributes, in-place-op ban, `nnx.Data`/`nnx.Static` annotation deprecation
- [Optax getting started](https://optax.readthedocs.io/en/latest/getting_started.html) — `GradientTransformation`, `apply_updates`, `chain`, schedules
