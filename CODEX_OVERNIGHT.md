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

## Remaining notes

- The focused MoE/config/data-parallel tests pass. A broader stack run seen earlier had three non-MoE numerical strictness failures: chunked CE weight-gradient differed by about `9.1e-5`; BD3 blocked attention differed from dense-mask attention by about `7.4e-4`; BD3 blocked model logits differed by about `3.0e-3`.
- The next useful experiment is probably not another basic MoE run. If tuning, inspect whether the step-time overhead comes from token dispatch/scatter, expert matmuls, or pmap communication, then try capacity factor/router regularization changes only if the loss curve or expert/drop metrics justify it.
