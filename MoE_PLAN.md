# Token-Choice Top-1 Switch MoE Plan

This plan adds an honest sparse MoE path to `jax/`: token-choice, top-1
Switch routing over SwiGLU experts. The goal is a credible first DLM ablation:
if performance is bad, the result should not be easy to dismiss as a toy dense
mixture, wrong layer placement, broken router gradients, or wrong loss scaling.

## Target Behavior

First experiment config:

```yaml
moe: true
moe_routing: token_choice_switch
moe_top_k: 1
moe_layers: alternating      # 0-indexed [1, 3, 5, 7] for n_layers=8
moe_num_experts: 4
moe_expert_d_ff: null        # null means use the dense block d_ff
moe_capacity_factor: 1.25
moe_use_router_prob: true
moe_drop_tokens: true
moe_router_dtype: float32

moe_load_balance_loss_weight: 1.0e-2
moe_router_z_loss_weight: 1.0e-3
```

Important placement decision: MoE and value embeddings should be on the same
layers. The current `has_value_embedding_layer(..., "alternating")` semantics
select `layer_idx % 2 == (n_layers - 1) % 2`, so for 8 layers it selects
`[1, 3, 5, 7]`. MoE should reuse that same placement by default. Explicit
`[1, 3, 5, 7]` is also acceptable, but `"alternating"` is the intended default
because it tracks the existing value-embedding convention.

## Non-Goals For First Pass

- No expert-choice routing.
- No top-2 routing.
- No expert parallelism or all-to-all.
- No variable-capacity routing.
- No nondropping fallback for overflow tokens.
- No dense soft mixture that evaluates every expert per token.
- No PyTorch parity requirement for MoE.

The strict first-pass routing name should be boring and precise:
`token_choice_top1_switch_swiglu`.

## Files To Touch

- `jax/transformer/moe.py`: new Switch MoE module and aux helpers.
- `jax/transformer/transformer.py`: layer placement helper, Block/Transformer
  MoE wiring, aux return path.
- `jax/transformer/__init__.py`: export MoE placement helper if tests need it.
- `jax/training/loss.py`: AR and supervised loss integration.
- `jax/training/step.py`: train-step signatures, static argnums, denominator
  handling for supervised sum-form losses.
- `jax/training/optimizer.py`: route router weights to AdamW.
- `jax/train_ar.py`: CLI/config args, model construction, loss weights,
  logging, FLOP/MFU accounting.
- Tests:
  - `jax/tests/test_parity_extras.py`
  - `jax/tests/test_training_stack.py`
  - `jax/tests/test_diffusion_stack.py`
  - `jax/tests/test_data_parallel_parity.py`
  - possibly a new `jax/tests/test_moe.py`

## Step 1: Generalize Layer Placement

Add a shared helper in `jax/transformer/transformer.py`:

```python
def has_layer(layer_idx, n_layers, placement):
    ...
```

Required semantics:

- `None`, `"none"`, `"off"`, `"false"` -> no layers.
- `"all"` -> all layers.
- `"final"` -> last layer only.
- `"alternating"` and `"alternating_late"` -> current value-embedding behavior:
  `layer_idx % 2 == (n_layers - 1) % 2`.
- Optional `"alternating_early"` -> opposite parity, useful later, but not the
  first MoE experiment.
- Sequence of ints -> exact 0-indexed layer set.

Keep backward compatibility:

```python
def has_value_embedding_layer(layer_idx, n_layers, placement="alternating"):
    return has_layer(layer_idx, n_layers, placement)

def has_moe_layer(layer_idx, n_layers, placement="alternating"):
    return has_layer(layer_idx, n_layers, placement)
```

For `n_layers=8`, both `has_value_embedding_layer(..., "alternating")` and
`has_moe_layer(..., "alternating")` must select `[1, 3, 5, 7]`.

## Step 2: Add `jax/transformer/moe.py`

Prefer a new module instead of placing MoE in `core.py`; this avoids expanding
the already-basic core file and avoids importing `_ModuleList` from
`transformer.py`.

In `moe.py`, define a local module-list helper:

```python
try:
    from flax.nnx import List as _ModuleList
except ImportError:
    _ModuleList = list
```

Define `MoEAux` as a `NamedTuple`, not a normal dataclass, so it is naturally
JAX/NNX friendly:

```python
class MoEAux(NamedTuple):
    load_balance_loss: Array
    router_z_loss: Array
    dropped_fraction: Array
    router_entropy: Array
    expert_fraction_min: Array
    expert_fraction_max: Array
    expert_fraction_std: Array
    router_prob_fraction_min: Array
    router_prob_fraction_max: Array
    router_prob_fraction_std: Array
    num_moe_layers: Array
```

Add helpers:

- `zero_moe_aux() -> MoEAux`
- `add_moe_aux(a, b) -> MoEAux`
- `finalize_moe_aux(aux) -> MoEAux`

Aggregation rule:

- Sum raw per-layer losses and diagnostics while traversing blocks.
- `num_moe_layers` is `1.0` for each active MoE layer and `0.0` for dense
  layers.
- Finalize by dividing average-like fields by `max(num_moe_layers, 1.0)`.
- For first pass, average per-layer min/max/std diagnostics. That is simple and
  scalar-only; add full per-expert histograms later if needed.

## Step 3: Implement `SwitchMoE`

Constructor args:

```python
SwitchMoE(
    rngs,
    d_model,
    d_ff,
    *,
    num_experts,
    expert_d_ff=None,
    capacity_factor=1.25,
    use_router_prob=True,
    router_dtype=jnp.float32,
    drop_tokens=True,
    fuse_up_gate=True,
    linear_init_std=None,
    dtype=jnp.float32,
)
```

Strict validation:

- `num_experts >= 1`
- `capacity_factor > 0`
- `router_dtype == jnp.float32` in first pass
- `drop_tokens is True`
- `expert_d_ff = d_ff if expert_d_ff is None else expert_d_ff`

Module structure:

```python
self.router = Linear(rngs, d_model, num_experts, dtype=router_dtype)
self.experts = _ModuleList([
    SwiGLU(
        rngs,
        d_model,
        expert_d_ff,
        dtype=dtype,
        fuse_up_gate=fuse_up_gate,
        linear_init_std=linear_init_std,
    )
    for _ in range(num_experts)
])
```

Router logits/probs must be fp32:

```python
x_flat = x.reshape(N, D)
router_logits = self.router(x_flat.astype(jnp.float32)).astype(jnp.float32)
router_probs = jax.nn.softmax(router_logits, axis=-1)
expert_id = jnp.argmax(router_probs, axis=-1).astype(jnp.int32)
selected_prob = jnp.take_along_axis(router_probs, expert_id[:, None], axis=-1)[:, 0]
```

Static capacity:

```python
B, T, D = x.shape
N = B * T
E = self.num_experts
C = max(1, int(math.ceil(self.capacity_factor * N / E)))
```

Use scatter/gather, not a dense `[tokens, experts, capacity]` dispatch mask:

```python
expert_one_hot = jax.nn.one_hot(expert_id, E, dtype=jnp.int32)
positions_all = jnp.cumsum(expert_one_hot, axis=0) - 1
slot = jnp.sum(positions_all * expert_one_hot, axis=-1)

valid = slot < C
safe_slot = jnp.where(valid, slot, 0)

expert_inputs = jnp.zeros((E, C, D), dtype=x.dtype)
expert_inputs = expert_inputs.at[expert_id, safe_slot].add(
    x_flat * valid[:, None].astype(x.dtype)
)
```

Run only selected expert batches:

```python
expert_outputs = jnp.stack(
    [self.experts[e](expert_inputs[e]) for e in range(E)],
    axis=0,
)
y_flat = expert_outputs[expert_id, safe_slot]
y_flat = y_flat * valid[:, None].astype(y_flat.dtype)
if self.use_router_prob:
    y_flat = y_flat * selected_prob[:, None].astype(y_flat.dtype)
y = y_flat.reshape(B, T, D)
```

Return `(y, aux)`.

## Step 4: Compute Raw MoE Aux Losses

Inside `SwitchMoE`, compute unweighted raw losses and scalar diagnostics.

Switch load-balancing loss:

```python
expert_fraction = jnp.mean(
    jax.nn.one_hot(expert_id, E, dtype=jnp.float32),
    axis=0,
)
router_prob_fraction = jnp.mean(router_probs, axis=0)
load_balance_loss = E * jnp.sum(
    jax.lax.stop_gradient(expert_fraction) * router_prob_fraction
)
```

Router z-loss:

```python
router_z_loss = jnp.mean(jax.nn.logsumexp(router_logits, axis=-1) ** 2)
```

Router entropy:

```python
router_entropy = -jnp.mean(
    jnp.sum(router_probs * jnp.log(router_probs + 1e-9), axis=-1)
)
```

Dropped fraction:

```python
dropped_fraction = 1.0 - jnp.mean(valid.astype(jnp.float32))
```

Diagnostics should be scalar-only in the first pass:

- `expert_fraction_min/max/std`
- `router_prob_fraction_min/max/std`

Do not multiply by `moe_load_balance_loss_weight` or
`moe_router_z_loss_weight` inside the model. Coefficients belong to the loss
functions.

## Step 5: Wire MoE Into `Block`

Add Block constructor args:

```python
moe: bool = False
moe_num_experts: int = 4
moe_expert_d_ff: int | None = None
moe_capacity_factor: float = 1.25
moe_use_router_prob: bool = True
moe_router_dtype: jnp.dtype = jnp.float32
moe_drop_tokens: bool = True
```

Set:

```python
self.is_moe = bool(moe)
self.ffn = SwitchMoE(...) if self.is_moe else SwiGLU(...)
```

Update `Block.__call__`:

```python
def __call__(..., return_aux: bool = False):
    attn_out, v = self.attn(...)
    x = x + attn_out
    h = self.ln2(x)

    if self.is_moe:
        ffn_out, moe_aux = self.ffn(h)
    else:
        ffn_out = self.ffn(h)
        moe_aux = zero_moe_aux()

    x = x + ffn_out

    if return_aux:
        return x, v, moe_aux
    return x, v
```

Preserve old `(x, v)` behavior when `return_aux=False`.

## Step 6: Wire MoE Into `Transformer`

Add model architecture/routing args only:

```python
moe: bool = False
moe_routing: str = "token_choice_switch"
moe_top_k: int = 1
moe_layers: str | Sequence[int] | None = "alternating"
moe_num_experts: int = 4
moe_expert_d_ff: int | None = None
moe_capacity_factor: float = 1.25
moe_use_router_prob: bool = True
moe_router_dtype: jnp.dtype = jnp.float32
moe_drop_tokens: bool = True
```

Validate in `Transformer.__init__`:

- If `moe=False`, ignore MoE layer construction and keep dense FFNs.
- If `moe=True`, only support `moe_routing == "token_choice_switch"` and
  `moe_top_k == 1`.

For each block:

```python
layer_idx = pos - 1
block_moe = moe and has_moe_layer(layer_idx, n_layers, moe_layers)
```

Because both value embedding and MoE default to `"alternating"`, they are
co-located on `[1, 3, 5, 7]` for the 8-layer default.

Add `return_aux` to `encode` and `__call__`.

Return table:

- `return_hidden=False, return_aux=False` -> `logits`
- `return_hidden=True, return_aux=False` -> `hidden`
- `return_hidden=False, return_aux=True` -> `(logits, aux)`
- `return_hidden=True, return_aux=True` -> `(hidden, aux)`

Carefully update both branches of the existing `nnx.remat(block)` loop. When
`return_aux=True`, both rematted and non-rematted branches must unpack three
values; otherwise both unpack two values.

Add `self.moe = bool(moe)` so loss code can cheaply decide whether to request
aux, or add a small helper/property such as `has_moe = bool(moe)`.

## Step 7: Integrate AR Loss

Update `ar_loss` signature:

```python
moe_load_balance_loss_weight: float = 0.0
moe_router_z_loss_weight: float = 0.0
```

If `model.moe` is true, call model with `return_aux=True`.

Full logits path:

```python
logits, aux = model(inputs, return_aux=True)
loss, z_loss, loss_bpb = cross_entropy_with_z_loss(...)
```

Chunked path:

```python
hidden, aux = model(inputs, return_hidden=True, return_aux=True)
loss, z_loss = linear_cross_entropy_with_z_loss_chunked(...)
```

If MoE is disabled, use `zero_moe_aux()` or an equivalent zero metrics helper.

Compute:

```python
moe_aux_loss = (
    moe_load_balance_loss_weight * aux.load_balance_loss
    + moe_router_z_loss_weight * aux.router_z_loss
)
total = loss + z_loss_weight * z_loss + moe_aux_loss
```

Add metrics:

- `moe_load_balance_loss`
- `moe_router_z_loss`
- `moe_aux_loss`
- `moe_dropped_fraction`
- `moe_router_entropy`
- `moe_expert_fraction_min/max/std`
- `moe_router_prob_fraction_min/max/std`
- `moe_num_layers`

Keep existing `z_loss` as LM-vocabulary z-loss only.

## Step 8: Integrate Supervised MDLM/BD3LM Losses

Update both:

- `supervised_lm_loss`
- `supervised_lm_loss_sums`

Add the same MoE loss-weight args.

Mean-form `supervised_lm_loss` is analogous to AR:

```python
total = loss + z_loss_weight * z_loss + moe_aux_loss
```

For `output_length` slicing, request hidden plus aux once:

```python
hidden, aux = model(..., return_hidden=True, return_aux=True)
hidden = hidden[:, :output_length]
```

Important: MoE aux is computed over the model input sequence, including BD3 dual
stream if present. That is intended: routing regularizes the actual conditional
compute used by the model, not only the supervised output slice.

## Step 9: Fix Supervised Sum-Form Normalization

The supervised training steps differentiate a sum-form loss, then divide grads
and metrics by a denominator. For MDLM/BD3LM this denominator can be the
expected-mask `loss_normalizer`, not the realized supervised-token count.

To preserve exact requested MoE coefficients, pass the denominator into
`supervised_lm_loss_sums` rather than multiplying aux loss by `valid_count`.

Add optional arg:

```python
loss_denominator: Array | float | None = None
```

Inside `supervised_lm_loss_sums`:

```python
denom_for_aux = valid_count if loss_denominator is None else jnp.asarray(loss_denominator, jnp.float32)
raw_moe_aux_loss = (
    moe_load_balance_loss_weight * mean_metrics["moe_load_balance_loss"]
    + moe_router_z_loss_weight * mean_metrics["moe_router_z_loss"]
)
moe_aux_loss_sum = denom_for_aux * raw_moe_aux_loss
total_sum = loss_sum + z_loss_weight * z_loss_sum + moe_aux_loss_sum
```

Return sum-form metrics for MoE:

- `moe_aux_loss_sum`
- `moe_load_balance_loss_sum`
- `moe_router_z_loss_sum`

The train-step wrappers then divide these by the same denominator they already
use for `loss` and `z_loss`.

## Step 10: Update Train-Step Wrappers

Update `loss_fn` and `supervised_loss_fn` signatures with:

```python
moe_load_balance_loss_weight
moe_router_z_loss_weight
```

Update all train steps:

- `train_step`
- `train_step_data_parallel`
- `train_step_accumulated`
- `train_step_accumulated_data_parallel`
- `train_step_supervised`
- `train_step_supervised_data_parallel`
- `train_step_supervised_accumulated`
- `train_step_supervised_accumulated_data_parallel`

Static argnums/in_axes must be adjusted carefully after adding positional args.
The MoE weights are numeric traced args, not static args. `loss_impl`,
`logit_chunk_size`, `is_causal`, `output_length`, and `bd3_block_len` remain
static as they are today.

For supervised steps, compute the denominator before `value_and_grad` and pass
it into `supervised_loss_fn`, so the aux sum uses the same denominator that will
scale gradients:

```python
denominator = _loss_denominator(loss_normalizer, precomputed_or_local_count)
```

For the single-device non-accumulated path, the count can be computed inside
`supervised_lm_loss_sums`; if avoiding a duplicate count calculation is awkward,
it is acceptable to compute the same mask-derived count in a helper before the
grad call. Prefer clear exactness over cleverness here.

For data-parallel supervised paths, denominator is global. The clean design is:

- Per shard computes CE/z-loss sums and raw MoE aux.
- `psum` CE/z-loss sums and valid counts.
- Average or `pmean` raw MoE aux across shards/layers for logging and the loss
  coefficient.
- Apply aux as `global_denominator * global_raw_moe_aux_loss` before gradient
  scaling.

If this becomes too invasive, document the first implementation's exact
semantics and add a parity test. The intended semantics are the exact global
coefficient semantics above.

Eval helpers should keep clean CE/z-loss validation loss. They may return MoE
diagnostics, but should not include MoE regularizers in eval `loss`.

## Step 11: Update Optimizer Grouping

Router weights have shape `(num_experts, d_model)` because `Linear.weight` is
stored as `(d_out, d_in)`. Without special handling, the current optimizer will
send router weights to Muon.

Add:

```python
def _is_router_param(path: str) -> bool:
    return ".router.weight" in path or path.endswith("router.weight")
```

Preferred config additions:

```python
router_adam_lr: float = 1e-3
router_adam_weight_decay: float = 0.0
router_adam_betas: tuple[float, float] = (0.9, 0.999)  # or existing Adam betas
```

Add param kind:

```python
if _is_router_param(path):
    return "adam_router"
```

Update `learning_rates` to return router LR, and update the Adam branch to
handle `adam_router`. If this is too much for the first patch, route router
weights to `adam_scalar`; that is less clean but still better than Muon.

Expert matrices remain Muon because experts are ordinary `SwiGLU` modules with
ordinary 2D `Linear` weights.

## Step 12: Update `train_ar.py`

Add CLI/config keys before YAML validation:

```text
--moe / --no-moe
--moe-routing
--moe-top-k
--moe-layers
--moe-num-experts
--moe-expert-d-ff
--moe-capacity-factor
--moe-use-router-prob / --no-moe-use-router-prob
--moe-drop-tokens / --no-moe-drop-tokens
--moe-router-dtype
--moe-load-balance-loss-weight
--moe-router-z-loss-weight
--router-adam-lr
--router-adam-wd
```

`moe_layers` parsing must accept both strings and integer lists from YAML:

- `"alternating"`
- `"all"`
- `"final"`
- `"none"`
- `"1,3,5,7"`
- `[1, 3, 5, 7]`

First-pass router dtype should validate hard:

```python
if args.moe_router_dtype != "float32":
    raise ValueError("First MoE implementation only supports fp32 router dtype.")
```

Pass architecture args into `Transformer`.

Pass loss weights into AR and supervised train steps.

Add resolved config fields and per-step row fields for all scalar MoE metrics.
Print a compact subset:

```text
moe_aux=... moe_drop=... moe_ent=...
```

## Step 13: Fix FLOP/MFU Accounting

The current `estimate_compute_parameters(trainable_params, ...)` treats all
trainable parameters as active compute. That overcounts MoE, because inactive
expert parameters are not used per token.

For MoE runs, report:

- `trainable_param_count`
- `dense_compute_param_count`
- `active_compute_param_count_estimate`
- `flops_per_token_estimate`
- whether the estimate is MoE-aware

Rough active estimate:

- Dense non-MoE layers count normally.
- Each MoE layer counts one active SwiGLU expert per token, scaled by capacity
  factor for padded expert batches:
  `active_moe_ffn_params ~= capacity_factor * dense_swiglu_params`.
- Add router params for MoE layers.
- Do not count `num_experts * dense_swiglu_params` as active compute.

The estimate does not need to be perfect, but it must stop making MoE look
artificially inefficient by counting inactive experts as active FLOPs.

## Step 14: Tests

Placement:

1. `has_value_embedding_layer("alternating")` gives `[1, 3, 5, 7]` for
   `n_layers=8`.
2. `has_moe_layer("alternating")` gives `[1, 3, 5, 7]` for `n_layers=8`.
3. `has_moe_layer([1, 3, 5, 7])` gives `[1, 3, 5, 7]`.
4. Optional: `has_moe_layer("alternating_early")` gives `[0, 2, 4, 6]`.

MoE module:

5. `SwitchMoE` forward output has shape/dtype `(B, T, D)`.
6. Aux values are finite.
7. Tiny capacity creates `dropped_fraction > 0`.
8. Router gradient norm is nonzero.
9. Expert gradient norm is nonzero.
10. With `moe_use_router_prob=True`, selected router probabilities affect the
    forward output and give the router a supervised gradient path.

Transformer API:

11. `Transformer(..., moe=False)` old call path still returns logits.
12. `Transformer(..., moe=True, return_aux=True)` returns `(logits, aux)`.
13. `Transformer(..., moe=True, return_hidden=True, return_aux=True)` returns
    `(hidden, aux)`.
14. `Transformer(..., moe=True, num_grad_checkpoint_layers=1)` works with
    `return_aux=True`.
15. MoE and VE are colocated under default `"alternating"` placement.

Loss and training:

16. `ar_loss` with MoE is finite in full and chunked modes.
17. `supervised_lm_loss` with MoE is finite in full and chunked modes.
18. `supervised_lm_loss_sums` preserves exact aux coefficient when
    `loss_normalizer != valid_count`.
19. Non-accumulated supervised train step works.
20. Accumulated supervised train step works.
21. Data-parallel supervised parity continues to hold, especially with uneven
    MDLM masks.

Optimizer:

22. `blocks.1.ffn.router.weight` is AdamW/router Adam.
23. `blocks.1.ffn.experts.0.w_up_gate.weight` is Muon.
24. Value-embedding table/mask grouping remains unchanged.

Trainer/config:

25. YAML config accepts `moe_layers: alternating`.
26. YAML config accepts `moe_layers: [1, 3, 5, 7]`.
27. Resolved config logs MoE architecture, loss weights, router LR, and
    MoE-aware compute estimates.

## Acceptance Criteria

- Default dense path remains unchanged and current tests pass.
- MoE path is sparse: each token runs at most one expert.
- MoE and value embeddings are colocated by default on existing alternating
  layers.
- Router logits/probs are fp32.
- Router gets supervised gradients via selected gate probability and auxiliary
  losses.
- Load-balancing and router z-loss are raw model aux values, weighted only in
  loss code.
- AR, MDLM, and BD3LM train paths include MoE regularizers during training.
- Eval loss remains clean CE/z-loss and does not include MoE regularizers.
- Router weights do not silently go to Muon.
- FLOP/MFU logs distinguish trainable parameters from active compute estimates.
