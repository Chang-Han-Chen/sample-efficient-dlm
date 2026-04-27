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
  - `lr_mult=2.0`: completed 1K as `ar_moe_old_bundle_no_lnscale_lr2p0_probe_1k` / W&B run id `p18cdqm3`. Last/best eval `3.0621` at step `950`, `-0.0905` better than plain AR MoE at the same step. This essentially tied the best layernorm-scaling-on `2.0` run (`3.0599`) while keeping routing very clean: drop `0.0000`, entropy `1.019`, pre-clip grad norm `0.0392`.
- Current conclusion:
  - The old-bundle MoE recipe wants a much hotter global LR than plain baseline MoE.
  - The two best completed short-run candidates are old-bundle `lr_mult=2.0` with layernorm scaling enabled (`3.0599` at step `950`) and without layernorm scaling (`3.0621` at step `950`). The enabled version is ahead by only about `0.0022`, so this difference is noise-level at 1K.
  - `lr_mult=3.0` crosses an instability boundary even when router LR is reduced.
  - Turning off layernorm scaling helps low-LR stability, but at `lr_mult=2.0` both variants are similarly strong. Use `lr_mult=2.0` as the full-run starting point; choose whether to keep layernorm scaling based on whether matching the dense old-bundle intervention bundle is more important than the slightly cleaner routing in the no-layernorm-scaling variant.

## AR MoE old-bundle full-run result

- Launched the layernorm-scaling-on best candidate as `ar_moe_old_bundle_lr2p0_full_nockpt` / W&B run id `tpesv0nb`, output dir `runs/moe_matrix/ar_moe_old_bundle_lr2p0_full_nockpt`.
- Command shape: `configs/experiments/ar_moe_old_bundle.yaml`, `--max-steps 5100`, `--warmup-steps 100`, `--lr-mult 2.0`, checkpoint saving disabled to match the successful 1K probes.
- Effective LR peaks for this run: table Adam `0.020`, value-embedding/mask Adam group `0.020` even though value embeddings are off, scalar Adam `0.010`, router Adam `0.002`, Muon `0.080`.
- Completed `5100` train steps: JSONL rows `5100`, last train step `5099`, last eval step `5050`.
- Best eval was `2.7564` at step `4800`; last eval was `2.8000` at step `5050`.
- This beats the completed plain AR MoE baseline `ar_moe_baseline_lr0p8`:
  - Best-vs-best at step `4800`: old-bundle `2.7564` vs plain `2.8062`, delta `-0.0498`.
  - Last eval at step `5050`: old-bundle `2.8000` vs plain `2.8380`, delta `-0.0380`.
- Wall time per step: mean `0.2649s`, median `0.2558s` after the first few steps. This is close to the 1K old-bundle probe timing and slower than the plain AR MoE baseline's `0.2370s` mean / `0.2313s` median.
- Routing stayed clean through the finish. In the tail from step `5000` onward, mean drop was `0.0020`, mean router entropy was `0.7815`, and mean pre-clip grad norm was `0.0767`. Final train row had loss `2.7788`, drop `0.0000`, entropy `0.7630`.
- There was one isolated pre-clip grad-norm spike at step `1900` (`5.23e9`) without an accompanying routing collapse or eval regression. Subsequent grad norms returned to the normal `0.03-0.06` range.
- Earlier failed full launch `ar_moe_old_bundle_lr2p0` / W&B run id `jugxbwpq` was stopped at step `708`; best eval was only `3.4017` at step `700`, with worse routing/drop behavior than the probe. Code inspection did not find a deterministic checkpoint mutation path: checkpoint save serializes copied state, and the LR schedule depends on `step`/`warmup_steps`, not `max_steps`. Current working hypothesis is MoE top-1 route sensitivity and nondeterminism, possibly amplified by sync/timing differences, not a direct checkpointing math bug.
- Current conclusion: old-bundle with layernorm scaling and `lr_mult=2.0` is the best completed AR MoE run so far. It improves eval over the plain AR MoE baseline by about `0.05` at best eval, at roughly `10-12%` higher step time.

## Split-router LayerNorm-scaling MoE probe

- Implemented the proposed MoE split:
  - Attention remains unchanged: `attn(RMSNorm(x) * depth_scale)`.
  - Dense FFN remains unchanged: `ffn(RMSNorm(x) * depth_scale)`.
  - MoE now routes on unscaled `RMSNorm(x)` but computes experts on `depth_scale * RMSNorm(x)`.
- Code shape:
  - `SwitchMoE.__call__(expert_x, *, router_x=None)` uses `router_x` for router logits/probs/top-1/aux losses and `expert_x` for dispatch into SwiGLU experts.
  - MoE `Block.ln2` is unscaled when `layernorm_scaling` is enabled; the depth scale is stored separately as `moe_expert_input_scale` and applied only to the expert input.
  - Router-prob gating is unchanged; no mean-normalized gate was added yet.
- Tests:
  - `pytest -q tests/test_moe.py`: `10 passed`, one existing pytest config warning.
- First probe used the same trusted old-bundle config as the completed full run, but only for 1K steps: `ar_moe_old_bundle_splitrouter_lr2p0_probe_1k` / W&B run id `wvwbwqix`, output dir `runs/moe_matrix/ar_moe_old_bundle_splitrouter_lr2p0_probe_1k`.
  - Config: `configs/experiments/ar_moe_old_bundle.yaml`, `--max-steps 1000`, `--warmup-steps 100`, `--lr-mult 2.0`, checkpoint saving disabled.
  - Effective LR peaks: table Adam `0.020`, scalar Adam `0.010`, router Adam `0.002`, Muon `0.080`.
  - Best/last eval: `3.0396` at step `950`.
  - Final matched-step routing metrics at step `950`: drop `0.0000`, router entropy `0.960`, expert fraction max `0.281`, router-prob fraction max `0.266`, pre-clip grad norm `0.0445`.
  - Wall time after initial compile: mean `0.2645s`, median `0.2568s`.
- Matched-step comparison at step `950`:
  - Split-router old-bundle `lr_mult=2.0`: `3.0396`
  - Previous layernorm-scaling old-bundle short probe: `3.0599`
  - Previous full no-checkpoint branch at the same step: `3.1177`
  - Plain AR MoE baseline: `3.1526`
- Started a full `lr_mult=2.0` split-router run, then stopped it early at step `168` to bracket LR first. Its early rows were healthy: step `150` eval `3.7844`, drop `0.0000`, entropy `0.734`.
- Bracketed split-router LR around the `2.0` anchor:
  - `lr_mult=1.5`: stopped at step `668`. Best eval was `3.2709` at step `550`, then regressed to `3.3497` at step `650` with drop `0.0288`, entropy `0.683`, expert fraction max `0.331`, and grad norm `0.536`.
  - `lr_mult=2.5`: stopped at step `553`. Best eval was `3.3268` at step `550`, worse than `2.0` at the same step (`3.2236`). It had a transient drop spike at step `400` (`0.0426`) and lower router entropy around `0.6-0.68`.
- Full split-router `lr_mult=2.0` rerun result:
  - Run: `ar_moe_old_bundle_splitrouter_lr2p0_full_nockpt_rerun`, output dir `runs/moe_matrix/ar_moe_old_bundle_splitrouter_lr2p0_full_nockpt_rerun`.
  - Stopped after collapse at step `3500`.
  - Best eval before collapse: `2.8282` at step `3200`.
  - The run was competitive before collapse but noticeably less stable than the 1K smoke. Examples:
    - step `2100`: eval `3.0120`, drop `0.0682`, entropy `0.694`, expert fraction max `0.371`
    - step `2750`: eval `2.9339`, drop `0.0501`, entropy `0.657`, expert fraction max `0.332`
    - step `3200`: best eval `2.8282`, drop `0.0000`, entropy `0.669`
  - Hard collapse began immediately after that:
    - step `3300`: eval `4.7616`, grad norm `153`, drop `0.3459`, entropy `0.385`, expert fraction max `0.655`
    - step `3500`: eval `6.5181`, grad norm `541`, drop `0.5388`, entropy `0.190`, expert fraction max `0.835`
  - Wall time before stop: mean `0.2637s`, median `0.2550s`.
  - Revised interpretation:
  - The split-router architecture improved the 1K smoke (`3.0396 @950`) but did not produce a stable long run with the same optimizer settings.
  - Principled explanation: before the split, deeper MoE routers saw `depth_scale * RMSNorm(x)`, which accidentally damped router logits and kept softmax probabilities higher entropy. After the split, routers see unscaled `RMSNorm(x)`, which removes depth-dependent router temperature coupling but also removes that accidental logit damping. With the same router LR/z-loss/gate, the router can sharpen over training, hit the top-1 capacity cliff, and collapse.
  - Next principled variants should keep the split but add an explicit router control knob: lower router LR, explicit router temperature > 1, stronger router z-loss, or mean-normalized selected-prob gate. Do not promote the current split-router `lr_mult=2.0` recipe to a full baseline.
- Lower-router-LR split probe:
  - Run: `ar_moe_old_bundle_splitrouter_lr2p0_routerhalf_full_nockpt` / W&B run id `wv47sqt1`, output dir `runs/moe_matrix/ar_moe_old_bundle_splitrouter_lr2p0_routerhalf_full_nockpt`.
  - Config was the same split-router old-bundle recipe with `lr_mult=2.0`, but `--router-adam-lr 0.0005`, giving router peak `0.001` instead of `0.002`. Table/scalar/Muon peaks were unchanged.
  - Stopped at step `2425` as noncompetitive.
  - Early router entropy was healthier than the full-router-LR split run, but the loss curve was slower and drops returned anyway:
    - step `550`: eval `3.2538`, drop `0.0000`, entropy `0.925`
    - step `950`: eval `3.2181`, drop `0.0349`, entropy `0.827`
    - step `1600`: best-so-far `3.0254`, drop `0.0001`, entropy `0.691`
    - step `1900`: eval `3.1294`, drop `0.0394`, entropy `0.616`
    - step `2400`: best eval `2.9983`, drop `0.0041`, entropy `0.602`
  - Wall time: mean `0.2612s`, median `0.2548s`.
  - Conclusion: lowering router LR alone does not fix split-router stability. It preserves entropy early but gives a weaker optimization trajectory, then entropy still drifts down and capacity/drop events recur. Prefer explicit router temperature, stronger z-loss, or gate normalization next.
- Stronger-router-z-loss split run:
  - Run: `ar_moe_old_bundle_splitrouter_lr2p0_zloss0p01_full_nockpt` / W&B run id `13fqp3bd`, output dir `runs/moe_matrix/ar_moe_old_bundle_splitrouter_lr2p0_zloss0p01_full_nockpt`.
  - Config was the same split-router old-bundle recipe with `lr_mult=2.0`, router peak LR `0.002`, Muon peak `0.080`, but `--moe-router-z-loss-weight 0.01` instead of `0.001`.
  - Completed `5100` train steps: JSONL rows `5100`, last train step `5099`, last eval step `5050`.
  - Best eval was `2.7621` at step `4450`; last eval was `2.7946` at step `5050`.
  - This essentially ties the old completed non-split old-bundle full run:
    - Best-vs-best: z-loss split `2.7621` vs old non-split `2.7564`, delta `+0.0057`.
    - Last eval: z-loss split `2.7946` vs old non-split `2.8000`, delta `-0.0054`.
  - Routing was much healthier than the unstable split run and cleaner than the old full run:
    - Tail from step `5000` onward: mean drop `0.00236`, mean router entropy `0.8773`, mean pre-clip grad norm `0.1082`.
    - Recent eval rows from `4300` through `5050` had drop `0.0000`; entropy stayed around `0.875-0.887`; router z stayed around `0.046-0.049`.
    - Best eval row at step `4450`: eval `2.7621`, drop `0.0000`, entropy `0.880`, router z `0.048`, expert fraction max `0.273`.
  - Wall time per step: mean `0.2650s`, median `0.2561s`, essentially the same as the other old-bundle MoE full runs.
  - Conclusion: stronger router z-loss is the first split-router variant that stays stable through 5.1K. It does not clearly beat the old non-split full run on best eval, but it matches it within noise and gives a much cleaner router. This is the best current split-router recipe.

## Non-split old-bundle + stronger z-loss checkpointed run

- Launched the old, non-split router-input behavior again with stronger router z-loss and periodic local checkpoints:
  - Run: `ar_moe_old_bundle_nonsplit_lr2p0_zloss0p01_ckpt1k`
  - W&B run id: `9a1ik1kw`
  - Output dir: `runs/moe_matrix/ar_moe_old_bundle_nonsplit_lr2p0_zloss0p01_ckpt1k`
  - Local log: `runs/moe_matrix/ar_moe_old_bundle_nonsplit_lr2p0_zloss0p01_ckpt1k/train.jsonl`
  - Config: `configs/experiments/ar_moe_old_bundle.yaml`, `--max-steps 5100`, `--warmup-steps 100`, `--lr-mult 2.0`, `--no-moe-split-router-input`, `--moe-router-z-loss-weight 0.01`.
  - Checkpointing: `--checkpoint-interval 1000`, `--no-save-best-checkpoint`, `--no-wandb-checkpoints`; local checkpoint dirs were written for `step_00001000`, `step_00002000`, `step_00003000`, `step_00004000`, `step_00005000`, and `final`.
- Completed `5100` train steps: JSONL rows `5100`, last train step `5099`, last eval step `5050`.
- Best eval was `2.7523` at step `4450`; last eval was `2.7792` at step `5050`.
- Comparison against completed full runs:
  - Plain AR MoE baseline `ar_moe_baseline_lr0p8`: best `2.8062 @4800`, last `2.8380 @5050`.
  - Old non-split z-loss `0.001` run `ar_moe_old_bundle_lr2p0_full_nockpt`: best `2.7564 @4800`, last `2.8000 @5050`.
  - Split-router z-loss `0.01` run `ar_moe_old_bundle_splitrouter_lr2p0_zloss0p01_full_nockpt`: best `2.7621 @4450`, last `2.7946 @5050`.
  - This checkpointed non-split z-loss `0.01` run is currently the best by best eval and last eval: `-0.0041` better than old non-split z-loss `0.001` best, `-0.0098` better than split z-loss `0.01` best, and `-0.0539` better than plain MoE best.
- Routing/optimizer diagnostics:
  - Tail from step `5000` onward: mean drop `0.00027`, mean router entropy `0.9776`, mean pre-clip grad norm `0.0541`.
  - Tail from step `4500` onward: mean drop `0.0039`, mean router entropy `0.975`.
  - Recent eval rows show an isolated routing/drop event around step `4700` (`eval 2.8530`, drop `0.0219`, entropy `0.959`, z `0.092`), but it recovered by step `4800` and ended cleanly.
- Wall time per step: mean `0.2664s`, median `0.2574s`. This is essentially the same as the other old-bundle MoE full runs; checkpoint saves are included in the mean, so the small slowdown versus no-checkpoint old-bundle runs is expected.
- Current verdict:
  - Push forward the non-split old-bundle recipe with `lr_mult=2.0` and router z-loss weight `0.01`.
  - The split-router idea is still principled and stable with stronger z-loss, but in the completed 5.1K comparisons it did not beat the simpler non-split stronger-z-loss run.
  - The stronger z-loss appears useful even without the split: it keeps router entropy higher and z metrics lower while slightly improving loss.

## AR MoE value-embedding probes

- Moved to the final AR candidate: MoE + value embeddings.
- Meaning of the candidate:
  - Config: `configs/experiments/ar_moe_value_embedding.yaml`.
  - AR objective, QK-norm, per-head attention gating, layernorm scaling.
  - `attn_val_residual: false`.
  - `value_embedding: true`, `value_embedding_layers: alternating`, `value_embedding_scale: 1.0`, `value_embedding_gain: false`.
  - MoE and value embeddings are colocated on alternating layers `[1, 3, 5, 7]`.
  - The MoE recipe was updated to inherit the best old-bundle MoE settings so far: `moe_split_router_input: false`, `moe_router_z_loss_weight: 0.01`, and local checkpoints every `1000` steps.
- Checkpointing decision:
  - Briefly tried W&B checkpoint artifacts for 1K periodic checkpoints in `ar_moe_value_embedding_no_vr_nogain_lr2p0_nonsplit_zloss0p01_wandbckpt1k`.
  - The first checkpoint directory was about `1.5G`; artifact upload blocked the training loop after step `999`.
  - Patched `train_ar.py` so that if `wandb_checkpoints: true` is used later, periodic checkpoints upload immediately when saved, but the current config default is back to local-only: `checkpoint_interval: 1000`, `save_best_checkpoint: false`, `wandb_checkpoints: false`.
- Run lookup index:
  - W&B project: `y38283929-uc-berkeley-electrical-engineering-computer-sc/sample-efficient-dlm`.
  - Plain MoE baseline: run `ar_moe_baseline_lr0p8`, W&B id `n0dh3ffk`, log `runs/moe_matrix/ar_moe_baseline_lr0p8/train.jsonl`.
  - Old-bundle non-split z-loss `0.001`: run `ar_moe_old_bundle_lr2p0_full_nockpt`, W&B id `tpesv0nb`, log `runs/moe_matrix/ar_moe_old_bundle_lr2p0_full_nockpt/train.jsonl`.
  - Split-router z-loss `0.01`: run `ar_moe_old_bundle_splitrouter_lr2p0_zloss0p01_full_nockpt`, W&B id `13fqp3bd`, log `runs/moe_matrix/ar_moe_old_bundle_splitrouter_lr2p0_zloss0p01_full_nockpt/train.jsonl`.
  - Best old-bundle non-split z-loss `0.01`: run `ar_moe_old_bundle_nonsplit_lr2p0_zloss0p01_ckpt1k`, W&B id `9a1ik1kw`, log `runs/moe_matrix/ar_moe_old_bundle_nonsplit_lr2p0_zloss0p01_ckpt1k/train.jsonl`, local checkpoints under `runs/moe_matrix/ar_moe_old_bundle_nonsplit_lr2p0_zloss0p01_ckpt1k/checkpoints/`.
  - Aborted VE local-only first launch: run `ar_moe_value_embedding_no_vr_nogain_lr2p0_nonsplit_zloss0p01_ckpt1k`, W&B id `6445buov`, log `runs/moe_matrix/ar_moe_value_embedding_no_vr_nogain_lr2p0_nonsplit_zloss0p01_ckpt1k/train.jsonl`; stopped at step `341` and not used for the verdict.
  - VE `lr_mult=2.0` W&B-checkpoint attempt: run `ar_moe_value_embedding_no_vr_nogain_lr2p0_nonsplit_zloss0p01_wandbckpt1k`, W&B id `9xerzca4`, log `runs/moe_matrix/ar_moe_value_embedding_no_vr_nogain_lr2p0_nonsplit_zloss0p01_wandbckpt1k/train.jsonl`.
  - VE `lr_mult=1.0` probe: run `ar_moe_ve_nonsplit_z0p01_lr1p0_probe_1k`, W&B id `yrt9w3du`, log `runs/moe_matrix/ar_moe_ve_nonsplit_z0p01_lr1p0_probe_1k/train.jsonl`.
  - VE `lr_mult=0.8` probe: run `ar_moe_ve_nonsplit_z0p01_lr0p8_probe_1k`, W&B id `vyo9podz`, log `runs/moe_matrix/ar_moe_ve_nonsplit_z0p01_lr0p8_probe_1k/train.jsonl`.
  - VE table-LR probe: run `ar_moe_ve_nonsplit_z0p01_lr1p0_vetbl0p005_probe_1k`, W&B id `3pt0qxai`, log `runs/moe_matrix/ar_moe_ve_nonsplit_z0p01_lr1p0_vetbl0p005_probe_1k/train.jsonl`.
  - VE capacity-factor `2.0` diagnostic: run `ar_moe_ve_nonsplit_z0p01_lr1p0_vetbl0p005_cap2p0_probe_1k`, W&B id `df98quc7`, log `runs/moe_matrix/ar_moe_ve_nonsplit_z0p01_lr1p0_vetbl0p005_cap2p0_probe_1k/train.jsonl`.
  - VE gain-ramp diagnostic: run `ar_moe_ve_nonsplit_z0p01_lr1p0_vetbl0p005_gain0_probe_1k`, W&B id `vgi6avym`, log `runs/moe_matrix/ar_moe_ve_nonsplit_z0p01_lr1p0_vetbl0p005_gain0_probe_1k/train.jsonl`.
- `lr_mult=2.0` VE probe:
  - Run: `ar_moe_value_embedding_no_vr_nogain_lr2p0_nonsplit_zloss0p01_wandbckpt1k` / W&B run id `9xerzca4`.
  - Stopped after the 1K checkpoint/upload issue; JSONL still contains a complete 1K probe through step `999`.
  - Best eval `3.3318 @550`; last eval `3.3452 @950`.
  - Step time mean `0.3420s`, median `0.3368s`.
  - Routing was stressed late: tail-100 drop `0.0744`, tail router entropy `0.978`.
  - Matched step `950`: VE `3.3452` vs plain MoE `3.1526` vs current best old-bundle MoE `3.1111`.
- `lr_mult=1.0` VE probe:
  - Run: `ar_moe_ve_nonsplit_z0p01_lr1p0_probe_1k` / W&B run id `yrt9w3du`.
  - Completed `1000` steps with no checkpoints.
  - Best/last eval `3.2618 @950`.
  - Step time mean `0.3464s`, median `0.3427s`.
  - Tail-100 drop `0.0315`, tail router entropy `0.887`.
  - Lowering global LR from `2.0` helped loss and routing substantially, but it still did not beat plain MoE at the same step (`3.1526 @950`).
- `lr_mult=0.8` VE probe:
  - Run: `ar_moe_ve_nonsplit_z0p01_lr0p8_probe_1k` / W&B run id `vyo9podz`.
  - Completed `1000` steps with no checkpoints.
  - Best/last eval `3.4200 @950`.
  - Step time mean `0.3450s`, median `0.3386s`.
  - Tail-100 drop `0.0244`, tail router entropy `0.857`.
  - Routing was cleaner in the tail than `lr_mult=1.0`, but the loss curve was slower and had a bad mid-run event: step `650` eval `4.2930`, drop `0.1914`, grad norm `237`.
  - Conclusion: lowering all LRs below `1.0` is too slow and does not make VE competitive.
- VE table-LR wiring and probe:
  - Fixed the existing `--value-embedding-table-adam-lr` flag so it actually controls the normal AR value-embedding table optimizer group. Before this, `adam_ve_table` still used `table_adam_lr`; the flag only affected the split mask-token fallback path.
  - `training/optimizer.py` now has `value_embedding_table_adam_lr`; `adam_ve_table` uses `ve_table_adam_lr`; `train_ar.py` logs both `value_embedding_table_adam_lr` and `value_embedding_mask_adam_lr`.
  - Verification: `python -m py_compile train_ar.py training/optimizer.py` passed.
  - Probe: `ar_moe_ve_nonsplit_z0p01_lr1p0_vetbl0p005_probe_1k` / W&B run id `3pt0qxai`.
  - Config delta from the `lr_mult=1.0` VE probe: token table LR stayed `0.010`, router LR stayed `0.001`, Muon stayed `0.040`, but VE table LR was reduced to `0.005`.
  - Completed `1000` steps with no checkpoints.
  - Best eval `3.2184 @900`; last eval `3.8186 @950`.
  - Step time mean `0.3359s`, median `0.3309s`.
  - Early/mid-run improved over previous VE `lr_mult=1.0`: at step `450`, `3.4227` vs previous `3.6901`, with drop `0.0130` vs `0.0710`.
  - The run then had a late routing shock: step `950` eval `3.8186`, drop `0.1948`, router entropy `0.739`, router z `0.391`; tail-100 drop was `0.1197`.
  - Conclusion: reducing VE table LR helps the loss trajectory, but the MoE/router still needs additional stabilization before VE can be promoted.
- Capacity-factor `2.0` diagnostic:
  - Run: `ar_moe_ve_nonsplit_z0p01_lr1p0_vetbl0p005_cap2p0_probe_1k` / W&B run id `df98quc7`.
  - Config delta from the VE table-LR probe: `moe_capacity_factor: 2.0` instead of `1.25`; no checkpoints.
  - Terminated after step `989`; it had already logged eval through step `950`.
  - Best/last eval `3.3456 @950`.
  - Step time mean `0.3741s`, median `0.3682s`.
  - The larger capacity did what it was supposed to diagnostically: step `950` drop was `0.0000`, and late tail drop was much lower than the capacity `1.25` VE table-LR run.
  - It did not solve loss or efficiency: step `950` eval `3.3456` was worse than VE table-LR capacity `1.25` best `3.2184 @900` and worse than plain MoE `3.1526 @950`, while wall time was much slower.
  - Conclusion: the capacity cliff explains the late drop shock, but simply increasing capacity is not a competitive recipe.
- VE gain-ramp diagnostic:
  - Run: `ar_moe_ve_nonsplit_z0p01_lr1p0_vetbl0p005_gain0_probe_1k` / W&B run id `vgi6avym`.
  - Config delta from the VE table-LR probe: `--value-embedding-gain --value-embedding-gain-init 0.0`, keeping `lr_mult=1.0`, VE table peak LR `0.005`, capacity `1.25`, and z-loss `0.01`; no checkpoints.
  - Terminated after user request at about step `873`; evals through step `850` are available.
  - Best eval `3.4675 @850`; earlier local best was `3.6023 @350`, then the run degraded around steps `550-750` before partially recovering.
  - Routing was cleaner than the sharp capacity-`1.25` table-LR shock but still not clean enough: step `550` eval `3.8362`, drop `0.0639`, entropy `0.798`; step `750` drop `0.0903`; step `850` drop `0.0211`, entropy `0.798`.
  - Step time mean `0.3531s`, median `0.3430s`, tail mean `0.3607s`.
  - Conclusion: ramping VE from zero helps early routing compared with full-strength VE, but it does not make VE+MoE competitive with plain MoE or old-bundle MoE.
- Dense AR comparison after VE probes:
  - Best dense overall found in W&B is still dense old-bundle without value embeddings: run `ar_old_bundle`, W&B id `8fjqg6b9`.
    - Config: QK-norm, value residual, per-head attention gating, layernorm scaling, no value embeddings, no MoE, `lr_mult=5.0`.
    - Best eval `2.7686 @4450`; last eval `2.8000 @5050`.
  - Best dense VE/no-value-residual run is `ar_value_embedding_no_vr_nogain_lr2p0_newtok_5k1_from500`, W&B id `dsezmr2u`, local log `runs/ar_matrix/ar_value_embedding_no_vr_nogain_lr2p0_5k1_from500/train.jsonl`.
    - Config: QK-norm, per-head attention gating, layernorm scaling, value embeddings on alternating layers, no value residual, no trainable VE gain, `lr_mult=2.0`.
    - Best eval `2.8480 @5000`; last eval `2.8895 @5050`.
  - Best MoE old-bundle remains `ar_moe_old_bundle_nonsplit_lr2p0_zloss0p01_ckpt1k`, W&B id `9a1ik1kw`.
    - Best eval `2.7523 @4450`; last eval `2.7792 @5050`.
  - Against dense old-bundle, best MoE is a narrow but real win: best delta `-0.0163`, last-eval delta `-0.0208`.
  - Against dense VE/no-VR, best MoE is a clear win: best delta `-0.0957`, last-eval delta `-0.1103`.
- Current VE verdict:
  - VE adds significant wall-time cost: about `0.34s/step`, versus `0.237s/step` for plain MoE and `0.266s/step` for the best old-bundle MoE checkpointed run.
  - Global LR-only tuning did not make VE beat plain MoE. Decoupling VE table LR improved the best observed VE eval to `3.2184 @900`, but the run was not stable through `950`.
  - Capacity `2.0` prevents the late capacity-drop shock but is slower and does not improve loss enough.
  - The gain-ramp diagnostic did not rescue VE+MoE.
  - Mainline recommendation: stop spending mainline time on VE+MoE. The serious AR path is old-bundle MoE without value embeddings, especially `ar_moe_old_bundle_nonsplit_lr2p0_zloss0p01_ckpt1k`.
  - If VE is revisited, treat it as a mechanism/debugging project rather than the next candidate baseline. The most relevant implementation hypothesis is that `value_embedding_gate.weight` is still a matrix parameter trained by Muon; moving that gate to a low-LR Adam group may be worth testing only if we explicitly return to VE.

## Remaining notes

- The focused MoE/config/data-parallel tests pass. A broader stack run seen earlier had three non-MoE numerical strictness failures: chunked CE weight-gradient differed by about `9.1e-5`; BD3 blocked attention differed from dense-mask attention by about `7.4e-4`; BD3 blocked model logits differed by about `3.0e-3`.
- The best completed AR MoE recipe remains `ar_moe_old_bundle_nonsplit_lr2p0_zloss0p01_ckpt1k`.
- For VE+MoE, do not launch a full 5.1K run unless the goal is explicitly to study VE mechanics; current results say it is not worth pushing as the main AR MoE candidate.
