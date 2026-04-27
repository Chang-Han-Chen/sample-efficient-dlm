# BD3 Curriculum Findings

Date: 2026-04-27

This note summarizes the **MDLM-init** side of the BD3 curriculum sweep on the
`climbmix_24x_newtok_8192` data. Per `BD3_CURRICULUM_INSTRUCTIONS.md`, this VM
was responsible for the MDLM source checkpoints and MDLM-initialized BD3 switch
runs. The AR-init track is running on another VM and is not included here.

The MDLM-init full grid is complete. The scratch `b4` BD3LM baseline is still
running, and scratch `b16` was intentionally skipped after the plan was revised.
Therefore this is a completed MDLM-init report, but not yet the final
AR-vs-MDLM-vs-scratch curriculum comparison.

## Configuration

Shared settings for the full MDLM-to-BD3 switch sweep:

- Config: `configs/experiments/mdlm_moe_old_bundle.yaml`
- Objective after switch: `bd3lm`
- Context length: `512`
- Batch size: `512`
- Devices: `4`
- Attention: dense BD3 attention, cuDNN attention implementation
- Validation residual: enabled via `--attn-val-residual`
- Router input split: disabled via `--no-moe-split-router-input`
- MoE router z-loss weight: `0.01`
- Vocabulary size: `8192`
- Mask token id: `8192`
- Diffusion steps: `100`
- Switch full-run max step: `10100`
- MDLM-to-BD3 switch LR multiplier: `0.666666666667`
- Scratch BD3LM baseline LR multiplier: `2.0`

## Data

Prepared token data:

- Train: `data/climbmix_24x_newtok_8192/tokens/train`
- Validation: `data/climbmix_24x_newtok_8192/tokens/val`
- Train shards: `24`
- Validation shards: `1`
- Train tokens: `1,542,544,386`
- Validation tokens: `64,423,581`
- Token dtype: `uint16`
- Token ids observed: `0..8191`
- Tokenizer vocab size: `8192`
- Diffusion vocab size: `8193`
- BOS id: `0`
- Mask id: `8192`

## Run Status

Current status at last inspection:

- MDLM-init full runs: `20 / 20` complete
- Scratch BD3LM baseline: `scratch_b4_lr2_full` active, around step `1660 / 10100`
  with best eval loss `3.055470` at step `1650`
- Scratch `b16`: skipped by request
- AR-init runs: handled separately on another VM

## MDLM-Init Full Sweep

Runs are sorted by best validation loss. MoE diagnostics are from the most
recent validation record for each run.

| Run | Block | p_mdlm | Best eval loss | Best step | Last eval loss | Recent MoE drop | Router entropy | Expert std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mdlm_p030_b4_lr0p666667_full` | 4 | 0.30 | **2.821899** | 9750 | 2.891958 | 0.00004 | 1.191 | 0.0194 |
| `mdlm_p050_b4_lr0p666667_full` | 4 | 0.50 | 2.898758 | 9250 | 2.908865 | 0.00387 | 1.201 | 0.0278 |
| `mdlm_p080_b4_lr0p666667_full` | 4 | 0.80 | 2.918467 | 9750 | 2.928295 | 0.01397 | 1.202 | 0.0355 |
| `mdlm_p080_b256_lr0p666667_full` | 256 | 0.80 | 2.935842 | 9000 | 2.955644 | 0.00000 | 1.222 | 0.0186 |
| `mdlm_p030_b64_lr0p666667_full` | 64 | 0.30 | 2.943600 | 9750 | 3.014566 | 0.00000 | 1.219 | 0.0168 |
| `mdlm_p090_b4_lr0p666667_full` | 4 | 0.90 | 2.948743 | 10000 | 2.972770 | 0.01788 | 1.188 | 0.0374 |
| `mdlm_p030_b16_lr0p666667_full` | 16 | 0.30 | 2.949728 | 9750 | 3.002683 | 0.00000 | 1.196 | 0.0185 |
| `mdlm_p090_b256_lr0p666667_full` | 256 | 0.90 | 2.950467 | 9950 | 2.969985 | 0.00000 | 1.216 | 0.0179 |
| `mdlm_p080_b64_lr0p666667_full` | 64 | 0.80 | 2.951223 | 9750 | 2.970217 | 0.00266 | 1.215 | 0.0244 |
| `mdlm_p030_b256_lr0p666667_full` | 256 | 0.30 | 2.952807 | 9750 | 3.031669 | 0.00508 | 1.218 | 0.0416 |
| `mdlm_p095_b256_lr0p666667_full` | 256 | 0.95 | 2.957018 | 9750 | 2.990182 | 0.00017 | 1.210 | 0.0242 |
| `mdlm_p050_b16_lr0p666667_full` | 16 | 0.50 | 2.961007 | 9200 | 2.980553 | 0.00211 | 1.210 | 0.0300 |
| `mdlm_p050_b256_lr0p666667_full` | 256 | 0.50 | 2.963033 | 9200 | 2.992326 | 0.00000 | 1.226 | 0.0173 |
| `mdlm_p080_b16_lr0p666667_full` | 16 | 0.80 | 2.964520 | 9750 | 2.978522 | 0.01100 | 1.206 | 0.0309 |
| `mdlm_p090_b64_lr0p666667_full` | 64 | 0.90 | 2.966734 | 10000 | 2.988187 | 0.00959 | 1.209 | 0.0415 |
| `mdlm_p050_b64_lr0p666667_full` | 64 | 0.50 | 2.967869 | 9200 | 2.994018 | 0.00000 | 1.218 | 0.0234 |
| `mdlm_p090_b16_lr0p666667_full` | 16 | 0.90 | 2.981070 | 10000 | 3.000462 | 0.01236 | 1.198 | 0.0316 |
| `mdlm_p095_b64_lr0p666667_full` | 64 | 0.95 | 2.993819 | 9750 | 3.014657 | 0.01139 | 1.198 | 0.0307 |
| `mdlm_p095_b4_lr0p666667_full` | 4 | 0.95 | 3.011452 | 10050 | 3.011452 | 0.02201 | 1.174 | 0.0367 |
| `mdlm_p095_b16_lr0p666667_full` | 16 | 0.95 | 3.029247 | 10050 | 3.029247 | 0.02182 | 1.186 | 0.0400 |

## Best By Slice

Best MDLM-init run at each block length:

| Block length | Best p_mdlm | Best eval loss | Run |
| ---: | ---: | ---: | --- |
| 4 | 0.30 | **2.821899** | `mdlm_p030_b4_lr0p666667_full` |
| 16 | 0.30 | 2.949728 | `mdlm_p030_b16_lr0p666667_full` |
| 64 | 0.30 | 2.943600 | `mdlm_p030_b64_lr0p666667_full` |
| 256 | 0.80 | 2.935842 | `mdlm_p080_b256_lr0p666667_full` |

Best MDLM-init run at each switch point:

| p_mdlm | Best block length | Best eval loss | Run |
| ---: | ---: | ---: | --- |
| 0.30 | 4 | **2.821899** | `mdlm_p030_b4_lr0p666667_full` |
| 0.50 | 4 | 2.898758 | `mdlm_p050_b4_lr0p666667_full` |
| 0.80 | 4 | 2.918467 | `mdlm_p080_b4_lr0p666667_full` |
| 0.90 | 4 | 2.948743 | `mdlm_p090_b4_lr0p666667_full` |
| 0.95 | 256 | 2.957018 | `mdlm_p095_b256_lr0p666667_full` |

## MDLM Source Chain

| Source run | Final step | Best eval loss | Best step |
| --- | ---: | ---: | ---: |
| `source_mdlm_to03100` | 3099 | 3.229854 | 3050 |
| `source_mdlm_to05100` | 5099 | 3.148435 | 4850 |
| `source_mdlm_to08100` | 8099 | 3.034931 | 7500 |
| `source_mdlm_to09100` | 9099 | 2.986118 | 9000 |
| `source_mdlm_to09600` | 9599 | 3.021582 | 9250 |
| `source_mdlm_to10100` | 10099 | 2.970368 | 9750 |

The source MDLM chain improves substantially through about 9k-10k steps, but
the BD3 switch results do not support "train MDLM as long as possible" under a
fixed total budget to 10.1k steps.

## LR Sweep Findings

The LR probes favored `lr_mult=0.666666666667` for MDLM-to-BD3 switches.

### p=0.30, block length 256 short probe

| LR multiplier | Best eval loss | Notes |
| ---: | ---: | --- |
| 0.6667 | **3.095994** | Best |
| 2.0 | 3.170001 | Worse |
| 6.0 | 3.220722 | Too hot |

### p=0.30, block length 64, 1500-step sweep

| LR multiplier | Best eval loss | Notes |
| ---: | ---: | --- |
| 0.6667 | **3.097255** | Best |
| 2.0 | 3.131764 | Viable but worse |
| 0.22 | 3.171580 | Too slow/low, terminated |
| 6.0 | 3.337286 | Bad, terminated |

Conclusion: `lr_mult=0.6667` is the right choice for MDLM-to-BD3 switch runs.
`lr_mult=2.0` is kept for the scratch BD3LM baseline.

## Main Findings

### Best MDLM-init curriculum

The best observed MDLM-init curriculum is:

`MDLM to p=0.30 -> BD3LM block_len=4, lr_mult=0.6667`

This run has the lowest validation loss by a large margin:

- Best eval loss: `2.821899`
- Last eval loss: `2.891958`
- Recent MoE dropped fraction: `0.00004`
- Recent router entropy: `1.191`
- Recent expert fraction std: `0.0194`

It is both the lowest-loss run and one of the cleanest MoE runs. There is no
sign of expert collapse or severe token dropping.

### Block length matters

At fixed `p_mdlm`, smaller block lengths are usually better, especially
`block_len=4`. The best run for `p=0.30`, `p=0.50`, `p=0.80`, and `p=0.90` is
always `block_len=4`.

The exception is `p=0.95`, where `block_len=256` is best. That late switch has
very little BD3 adaptation time left, and the small-block runs show weaker MoE
health. This suggests that small-block BD3 is a strong adaptation objective,
but it needs enough post-switch training budget.

### More MDLM pretraining can improve adaptation speed, but not final loss here

At fixed block size, especially for larger blocks and even for `block_len=4`,
later MDLM checkpoints often show a steeper post-switch loss decay. That means
MDLM pretraining is learning useful denoising representations that transfer to
BD3LM.

However, under the fixed total budget of 10.1k steps, later switch points do
not win overall. The best final losses come from earlier switches, especially
`p=0.30, block_len=4`. The likely explanation is budget allocation: late MDLM
switches start from a stronger representation and adapt quickly, but they leave
too little BD3 time to exploit the small-block objective.

### Diffusion-vs-AR interpretation

The results should not be described as a simple contradiction between "more
diffusion is better" and "more AR is better." These are different axes:

- Higher `p_mdlm` means a stronger MDLM denoising initialization.
- Smaller BD3 block length means more clean left context after the switch, so
  the BD3 objective becomes more AR-like.

The current evidence says that MDLM pretraining helps build transferable
denoising features, but the model benefits from adapting those features with a
small-block, more AR-conditioned BD3 objective. The best fixed-budget recipe so
far is therefore early MDLM init plus enough `block_len=4` BD3 training time.

### MoE health

MoE health degrades as the switch gets very late, particularly for small block
lengths:

- `p=0.30, b4`: drop `0.00004`, entropy `1.191`, std `0.0194`
- `p=0.80, b4`: drop `0.01397`, entropy `1.202`, std `0.0355`
- `p=0.90, b4`: drop `0.01788`, entropy `1.188`, std `0.0374`
- `p=0.95, b4`: drop `0.02201`, entropy `1.174`, std `0.0367`

This reinforces the loss result: late small-block switches are not just worse
on eval loss; they also look less healthy from a routing/drop perspective.

## Recommendation

For the MDLM-init track:

1. Treat `mdlm_p030_b4_lr0p666667_full` as the current winner.
2. Treat `block_len=4` as the leading BD3LM adaptation setting when enough
   post-switch budget is available.
3. Do not use the latest MDLM checkpoint by default under a fixed total budget;
   later MDLM can adapt faster, but it did not produce the best final loss.
4. Keep `lr_mult=0.6667` for MDLM-to-BD3 switch runs.
5. Compare this MDLM-init result against the AR-init VM results and the scratch
   `b4` baseline before making the final curriculum claim requested in
   `BD3_CURRICULUM_INSTRUCTIONS.md`.

Current best MDLM-init candidate:

`mdlm_p030_b4_lr0p666667_full`

## AR-Init Addendum

This section summarizes the AR-init side of the same BD3 curriculum sweep from
the second VM. The MDLM-init findings above are unchanged. AR-init runs use the
same data, model config, BD3 objective, context length, batch size, dense BD3
attention, cuDNN attention implementation, validation residual, disabled split
router input, and MoE z-loss settings described above. The AR-to-BD3 switch
runs use `lr_mult=2.0`.

Current AR-init status at last inspection:

- Completed AR-init BD3 runs: `p_ar=0.30` for block lengths `256`, `64`, `16`,
  and `4`; `p_ar=0.50` for block lengths `256` and `4`; `p_ar=0.80` for block
  length `256`
- Active AR-init BD3 run: `ar_p080_b4_lr2_full`
- Pending AR-init BD3 runs: `p_ar=0.90` and `p_ar=0.95` for block lengths `256`
  and `4`
- Scratch BD3LM baselines completed on this VM: block lengths `256` and `64`

### AR Source Chain

| Source run | Final step | Best eval loss | Best step | Last eval loss |
| --- | ---: | ---: | ---: | ---: |
| `source_ar_to03100` | 3099 | 2.827348 | 3050 | 2.827348 |
| `source_ar_to05100` | 5099 | 2.741890 | 5050 | 2.741890 |
| `source_ar_to08100` | 8099 | 2.679161 | 7100 | 2.708264 |
| `source_ar_to09100` | 9099 | 2.661022 | 8950 | 2.702767 |
| `source_ar_to09600` | 9599 | 2.676333 | 9400 | 2.712166 |
| `source_ar_to10100` | 10099 | 2.671805 | 9900 | 2.707195 |

The AR source chain is much stronger than the MDLM source chain in raw
validation loss. However, the AR-to-BD3 switch results so far do not support
"train AR as long as possible before switching" under the fixed 10.1k-step
budget.

### AR-Init Completed And Active Runs

Runs are sorted by curriculum family and block length. The active `p_ar=0.80,
block_len=4` row is provisional and should not be used as a final comparison.

| Run | Block | p_ar | Status | Best eval loss | Best step | Last eval loss |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `scratch_b256_lr2_full` | 256 | 0.00 | complete | 2.940822 | 8650 | 3.006591 |
| `scratch_b64_lr2_full` | 64 | 0.00 | complete | 2.946705 | 8950 | 2.974041 |
| `ar_p030_b256_lr2_full` | 256 | 0.30 | complete | 2.962922 | 9750 | 2.994042 |
| `ar_p030_b64_lr2_full` | 64 | 0.30 | complete | 2.917921 | 9500 | 2.966949 |
| `ar_p030_b16_lr2_full` | 16 | 0.30 | complete | 2.876053 | 9500 | 2.924899 |
| `ar_p030_b4_lr2_full` | 4 | 0.30 | complete | **2.727532** | 9750 | 2.780340 |
| `ar_p050_b256_lr2_full` | 256 | 0.50 | complete | 3.012413 | 9600 | 3.031457 |
| `ar_p050_b4_lr2_full` | 4 | 0.50 | complete | 2.771696 | 9200 | 2.787228 |
| `ar_p080_b256_lr2_full` | 256 | 0.80 | complete | 3.151225 | 9700 | 3.162386 |
| `ar_p080_b4_lr2_full` | 4 | 0.80 | running | 3.676480 | 8150 | 3.676480 |

### AR-Init: Fixed Block Length

Treating scratch as `p_ar=0`, the clearest completed fixed-block comparison is
for `block_len=256`:

| Block length | p_ar | Best eval loss | Run |
| ---: | ---: | ---: | --- |
| 256 | 0.00 | **2.940822** | `scratch_b256_lr2_full` |
| 256 | 0.30 | 2.962922 | `ar_p030_b256_lr2_full` |
| 256 | 0.50 | 3.012413 | `ar_p050_b256_lr2_full` |
| 256 | 0.80 | 3.151225 | `ar_p080_b256_lr2_full` |

For `block_len=256`, AR init is not helping. The scratch run is the best
completed result, and increasing `p_ar` monotonically worsens best validation
loss among completed runs.

For `block_len=64`, only scratch and `p_ar=0.30` are available:

| Block length | p_ar | Best eval loss | Run |
| ---: | ---: | ---: | --- |
| 64 | 0.00 | 2.946705 | `scratch_b64_lr2_full` |
| 64 | 0.30 | **2.917921** | `ar_p030_b64_lr2_full` |

Here early AR init does help. This is the first block length where AR
pretraining beats the scratch BD3 baseline.

For `block_len=4`, the completed AR-init comparison is:

| Block length | p_ar | Best eval loss | Run |
| ---: | ---: | ---: | --- |
| 4 | 0.30 | **2.727532** | `ar_p030_b4_lr2_full` |
| 4 | 0.50 | 2.771696 | `ar_p050_b4_lr2_full` |
| 4 | 0.80 | 3.676480 | `ar_p080_b4_lr2_full` running |

The active `p_ar=0.80` run is still too early for a final claim, but the
completed `p_ar=0.30` and `p_ar=0.50` runs already show that earlier switching
is better at `block_len=4`.

### AR-Init: Fixed p_ar

At fixed `p_ar=0.30`, smaller BD3 block length is strongly and monotonically
better:

| p_ar | Block length | Best eval loss | Run |
| ---: | ---: | ---: | --- |
| 0.30 | 256 | 2.962922 | `ar_p030_b256_lr2_full` |
| 0.30 | 64 | 2.917921 | `ar_p030_b64_lr2_full` |
| 0.30 | 16 | 2.876053 | `ar_p030_b16_lr2_full` |
| 0.30 | 4 | **2.727532** | `ar_p030_b4_lr2_full` |

At fixed `p_ar=0.50`, the same small-block preference is visible:

| p_ar | Block length | Best eval loss | Run |
| ---: | ---: | ---: | --- |
| 0.50 | 256 | 3.012413 | `ar_p050_b256_lr2_full` |
| 0.50 | 4 | **2.771696** | `ar_p050_b4_lr2_full` |

At `p_ar=0.80`, `block_len=4` is still running, so the fixed-`p_ar` comparison
is not complete:

| p_ar | Block length | Best eval loss | Run |
| ---: | ---: | ---: | --- |
| 0.80 | 256 | 3.151225 | `ar_p080_b256_lr2_full` |
| 0.80 | 4 | 3.676480 | `ar_p080_b4_lr2_full` running |

For scratch BD3LM, treated as `p_ar=0`, only block lengths `256` and `64` are
complete on this VM:

| p_ar | Block length | Best eval loss | Run |
| ---: | ---: | ---: | --- |
| 0.00 | 256 | **2.940822** | `scratch_b256_lr2_full` |
| 0.00 | 64 | 2.946705 | `scratch_b64_lr2_full` |

### AR-Init Main Findings

The best completed AR-init curriculum so far is:

`AR to p=0.30 -> BD3LM block_len=4, lr_mult=2.0`

This run is also the best BD3 curriculum result observed across the completed
AR-init and MDLM-init runs currently recorded in this file:

- Best AR-init BD3 loss: `2.727532` from `ar_p030_b4_lr2_full`
- Best MDLM-init BD3 loss: `2.821899` from `mdlm_p030_b4_lr0p666667_full`
- Best scratch BD3 loss available here: `2.940822` from `scratch_b256_lr2_full`

At fixed `p_ar`, smaller BD3 block lengths are better. The `p_ar=0.30` sweep is
the cleanest evidence: `b256 -> b64 -> b16 -> b4` improves from `2.962922` to
`2.727532`.

At fixed block length, increasing `p_ar` does not help so far. For
`block_len=256`, scratch is best and later AR switches get progressively worse:
`2.940822` at `p_ar=0`, `2.962922` at `p_ar=0.30`, `3.012413` at `p_ar=0.50`,
and `3.151225` at `p_ar=0.80`. For `block_len=4`, the completed runs also favor
the earlier switch: `2.727532` at `p_ar=0.30` versus `2.771696` at `p_ar=0.50`.

The current interpretation is that AR features transfer well to BD3 only when
the post-switch objective is sufficiently AR-like, meaning small BD3 block
lengths. Large-block BD3 does not benefit from AR initialization under this
budget, and late AR switching leaves too little BD3 adaptation time.
