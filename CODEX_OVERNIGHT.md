# CODEX_OVERNIGHT

## 2026-04-26 10:57:25 UTC

- Read `MoE_PLAN.md` and inspected the existing MoE-related worktree changes.
- Found that the sparse Switch MoE implementation, transformer wiring, loss integration, optimizer grouping, trainer args, and focused MoE tests were already present in the worktree.
- Ran `pytest -q tests/test_moe.py`; the implementation behaved correctly, but one test expected one MoE layer for `n_layers=3`. Under the plan's `alternating` placement rule, layers `[0, 2]` are selected, so the correct count is two.
- Updated `tests/test_moe.py` to match the plan's placement semantics.
- Added six MoE experiment configs under `configs/experiments/` for the requested AR/MDLM baseline, old-bundle, and value-embedding variants.

## 2026-04-26 11:06:29 UTC

- Fixed data-parallel train-step compatibility by making MoE loss weights keyword-only for pmapped train steps and restoring pmap `in_axes` to the original positional arity.
- Updated `train_ar.py` to pass MoE loss weights by keyword, preserving existing dense and test call sites.
- Verification:
  - `python -m py_compile transformer/moe.py transformer/transformer.py training/loss.py training/step.py training/optimizer.py train_ar.py` passed.
  - `pytest -q tests/test_moe.py tests/test_configs.py` passed: 10 passed, 1 pytest config warning.
  - `pytest -q tests/test_data_parallel.py` passed: 5 passed, 1 pytest config warning.
  - AR synthetic MoE trainer smoke passed for 2 steps.
  - MDLM synthetic MoE + value embedding + split mask token + no mask vector smoke passed for 2 steps.

## 2026-04-26 later update

- Reworked data-parallel step wrappers again after the first pmap fix still failed with keyword args at runtime. The final shape is fixed-arity private pmapped implementations plus public Python wrappers that provide default MoE loss weights for older call sites.
- Updated `train_ar.py` to pass `moe_load_balance_loss_weight` and `moe_router_z_loss_weight` positionally into data-parallel train steps.
- Verification after this fix:
  - `python -m py_compile training/step.py train_ar.py` passed.
  - `pytest -q tests/test_moe.py tests/test_data_parallel.py` passed: 13 passed, 1 pytest config warning.
- Ran a real-data AR MoE smoke from `configs/experiments/ar_moe_baseline.yaml` for 2 steps with W&B disabled. It completed successfully; losses were about `9.53`, `moe_drop` was about `0.09`, and router entropy was about `1.12`.

## AR MoE baseline run

- Launched and completed the full AR MoE baseline W&B run:
  - Config: `configs/experiments/ar_moe_baseline.yaml`
  - W&B display name: `ar_moe_baseline_lr0p8`
  - W&B run id: `n0dh3ffk`
  - Output dir: `runs/moe_matrix/ar_moe_baseline_lr0p8`
- Important config values:
  - AR next-token CE, `loss_impl=full`
  - batch size `512`, context `512`, data parallel over 2 local GPUs
  - `n_layers=8`, `d_model=768`, dense `d_ff=2048`
  - MoE layers `[1, 3, 5, 7]`
  - 4 SwiGLU experts per MoE layer
  - expert capacity factor `1.25`
  - load-balance loss weight `0.01`
  - router z-loss weight `0.001`
  - router Adam peak LR `8e-4` after `lr_mult=0.8`
  - Muon peak LR `0.032` after `lr_mult=0.8`
- Final training state:
  - step `5099`: train loss `2.8122`, total loss `2.8234`
  - last eval at step `5050`: eval loss `2.8380`
  - best observed eval: step `4800`, eval loss `2.8062`
  - final MoE drop fraction `0.004754`
  - final router entropy `0.8130`
  - final expert fraction min/max `0.2126` / `0.2904`
  - final router z metric `0.1388`
- Timing:
  - measured average step time: `0.2370s`
  - measured throughput: about `1.105M` tokens/sec
  - printed MFU is not trustworthy for this run because the compute estimate and peak-FLOP accounting are approximate for MoE.

## Dense AR comparison

- Compared against the existing W&B dense run `ar_baseline_lr0p8`:
  - W&B run id: `mbbnqirk`
  - summary eval loss: `2.8841`
  - summary train loss: `2.8879`
  - W&B average step time: `0.4401s`
- The loss comparison is fair in the important sense: both eval losses are clean AR next-token CE with the same batch/context/vocab setup and no MoE auxiliary terms in eval.
- The old W&B wall-time comparison is not fair because that dense run was on A100 while the MoE run was on the current H100 setup.
- Active-compute comparison:
  - dense trainable params: about `62.9M`
  - MoE trainable params: about `119.6M`
  - dense active compute estimate: about `69.2M` params
  - MoE active compute estimate: about `73.95M` params
  - dense estimated FLOPs/token: about `453.1M`
  - MoE estimated FLOPs/token: about `481.4M`
  - MoE is about `1.9x` trainable params but only about `1.06x` estimated active FLOPs/token.

## Dense AR H100 timing run

- Ran a fresh dense AR baseline timing run on the current H100 setup with W&B disabled:
  - Config: `configs/experiments/ar_baseline.yaml`
  - Run name: `ar_baseline_h100_timing_100`
  - Command used `--max-steps 100 --lr-mult 0.8 --eval-batches 0 --log-every 10 --no-wandb --no-save-final-checkpoint --no-save-best-checkpoint`
  - Log dir: `runs/ar_matrix/ar_baseline_h100_timing_100`
- Dense H100 timing after warmup:
  - post-step-5 mean: `0.1782s/step`
  - median: `0.1773s/step`
  - min/max: `0.1708s` / `0.1892s`
  - throughput: about `1.47M` tokens/sec
- Same-machine timing conclusion:
  - MoE improves eval loss versus the dense anchor, but it is slower per step on the same H100 setup.
  - Dense H100 timing is about `0.178s/step`; MoE is about `0.237s/step` over the full run.
  - That makes MoE roughly `1.3x` slower wall-clock per step despite only about `1.06x` estimated active FLOPs/token, so routing/scatter overhead is material.

## Git/data notes

- The local MoE implementation is committed on `main` at `6ebb584` and `origin/main` pointed to the same commit in the last observed status.
- `git ls-tree` checks showed no `data/climbmix_24x_newtok_8192/` data and no notebook checkpoints in `HEAD`.
- `.gitignore` intentionally ignores the local raw/tokenized dataset directory:
  - `data/climbmix_24x_newtok_8192/`
- Run logs are intentionally not ignored by that rule; the user wants logs preserved.

## MoE LR/router sweep notes

- Global LR probes above `lr_mult=0.8` were stable but worse:
  - `lr_mult=1.0`, router base LR `0.001`: last matched eval around step `950` was `3.2349` vs baseline `3.1526`.
  - `lr_mult=1.25`, router base LR `0.001`: stable to `600`, last eval `3.4597`.
  - `lr_mult=1.5`, router base LR `0.001`: stable until stopped, but clearly worse by step `475`.
- At `lr_mult=1.0`, changing router LR affected routing but did not fix the loss gap:
  - router base LR `0.0005`: best/last eval `3.3564` at step `575`.
  - router base LR `0.002`: best/last eval `3.3570` at step `575`.
- At `lr_mult=0.8`, short router-LR probes through `600` steps were close:
  - router base LR `0.0005`, effective peak `0.0004`: best/last eval `3.3071` at step `575`.
  - router base LR `0.002`, effective peak `0.0016`: best/last eval `3.3006` at step `575`.
  - router base LR `0.005`, effective peak `0.004`: best/last eval `3.2964` at step `575`.
- Longer validation of router base LR `0.005`:
  - Run: `ar_moe_baseline_lr0p8_router_halftable_probe_1200`
  - W&B run id: `fx9schci`
  - Completed `1200` train steps, last eval at step `1150`.
  - Step `1150`: eval `3.1005` vs original `lr0p8` eval `3.1020`, delta `-0.0014`.
  - Mean matched eval delta after step `600`: about `-0.0010`, too small to treat as a clear win.
  - Tail-100 step time: `0.2389s`, essentially the same as the original MoE run.
  - Tail-100 drop mean `0.0160` vs original run tail-100 drop mean `0.0061`.
  - Tail-100 router entropy `0.7140` vs original run tail-100 entropy `0.8118`.
  - Conclusion: router base LR `0.005` is not obviously unstable, but it stresses routing more and only produces noise-level eval improvement. Do not promote it to a full run unless the goal is explicitly to study router dynamics.

## AR MoE old-bundle sweep notes

- Moved from plain AR MoE to old-bundle-only AR MoE, still no value embeddings.
- Old-bundle means:
  - `attn_qknorm: true`
  - `attn_val_residual: true`
  - `attn_gating: per-head`
  - `layernorm_scaling: true`
  - `value_embedding: false`
- The inherited config default `lr_mult=5.0` was not used as the first run. Initial probes used 1K train steps, 100 warmup steps, no checkpoints, W&B on, and compared against the plain AR MoE baseline at matched steps.
- Layernorm-scaling-on old-bundle probes:
  - `lr_mult=0.6`: last/best eval `3.2961` at step `950`, `+0.1435` worse than plain AR MoE at the same step. Had real routing shocks: token drop briefly reached `35.6%`, and pre-clip grad norm reached about `7.2e7`.
  - `lr_mult=0.8`: last/best eval `3.3672` at step `950`, `+0.2146` worse than plain AR MoE. Router entropy collapsed to about `0.53` by the end, with drop about `5.4%`.
  - `lr_mult=1.0`: last/best eval `3.1894` at step `950`, only `+0.0368` worse than plain AR MoE. Better than `0.6/0.8`, but still not a win.
  - `lr_mult=1.25`: stopped at step `704`; last/best eval `3.5534` at step `700`, much worse.
  - `lr_mult=2.0`: completed 1K and is the best completed old-bundle probe so far. Last/best eval `3.0599` at step `950`, `-0.0926` better than plain AR MoE at the same step. Routing was clean: drop `0.0009`, entropy `0.998`, pre-clip grad norm `0.0406`.
  - `lr_mult=3.0`: unstable. It was merely behind through step `650`, then collapsed: step `700` eval `4.9065`, drop `28.1%`, grad norm about `1.2e6`; step `750` eval `10.5643`, drop `55.9%`, entropy `0.263`; stopped before the queued `5.0` probe.
  - `lr_mult=3.0` with router peak held near the successful `2.0` run by setting base `router_adam_lr=0.0006666667`: stopped at step `668`; last/best eval `3.3044` at step `650`, not competitive. Lowering router LR alone did not rescue `3.0`.
- Added `configs/experiments/ar_moe_old_bundle_no_lnscale.yaml` to test old-bundle without layernorm scaling while keeping QK-norm, value residual, per-head attention gating, and MoE.
- No-layernorm-scaling probes:
  - `lr_mult=0.8`: completed 1K. Best eval `3.1547` at step `800`, last eval `3.2438` at step `950`. This was much cleaner than layernorm-scaling-on `0.8`, but not better than layernorm-scaling-on `2.0`.
  - `lr_mult=1.0`: stopped at step `698`; last/best eval `3.4189` at step `650`, not competitive.
  - `lr_mult=2.0`: currently running as `ar_moe_old_bundle_no_lnscale_lr2p0_probe_1k` / W&B run id `p18cdqm3`. Latest recorded local state when this note was written: rows `475`, last train step `474`, last eval step `450`, eval `3.2942`, `-0.0775` versus plain AR MoE at the same step, drop `0.0000`, entropy `1.008`, grad norm `0.0705`.
- Current conclusion:
  - The old-bundle MoE recipe wants a much hotter global LR than plain baseline MoE.
  - The best completed short-run candidate is old-bundle with layernorm scaling enabled and `lr_mult=2.0`.
  - `lr_mult=3.0` crosses an instability boundary even when router LR is reduced.
  - Turning off layernorm scaling helps low-LR stability, but has not yet beaten the completed `lr_mult=2.0` layernorm-scaling-on run. Finish the active no-layernorm-scaling `2.0` probe before deciding whether to promote `lr_mult=2.0` layernorm-scaling-on to a full 5.1K run.

## Remaining notes

- The focused MoE/config/data-parallel tests pass. A broader stack run seen earlier had three non-MoE numerical strictness failures: chunked CE weight-gradient differed by about `9.1e-5`; BD3 blocked attention differed from dense-mask attention by about `7.4e-4`; BD3 blocked model logits differed by about `3.0e-3`.
- The next useful experiment is probably not another basic MoE run. If tuning, inspect whether the step-time overhead comes from token dispatch/scatter, expert matmuls, or pmap communication, then try capacity factor/router regularization changes only if the loss curve or expert/drop metrics justify it.
