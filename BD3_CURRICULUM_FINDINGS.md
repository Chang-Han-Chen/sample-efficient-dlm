# BD3 Curriculum Findings

Date: 2026-04-27

This summarizes the MDLM-init side of the BD3 curriculum sweep on the
`climbmix_24x_newtok_8192` data. Per `BD3_CURRICULUM_INSTRUCTIONS.md`, this VM
is responsible for MDLM source training and MDLM-initialized BD3 switch runs;
the AR-init side is handled separately on another VM.

The main full sweep is still in progress, so the conclusions below should be
treated as interim findings for the MDLM-init track only. They are not yet the
final AR-vs-MDLM-vs-scratch curriculum comparison requested by the instruction
file.

## Configuration

Shared settings for the current full switch sweep:

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

## Completed And Active Full Runs

Current full queue status at last inspection:

- Completed full runs: `7 / 22`
- Active run: `mdlm_p050_b4_lr0p666667_full`
- Active run progress: about `8634 / 10100`

These are MDLM-initialized BD3LM runs. AR-initialized runs are intentionally
not included here.

| Run | Status | Best eval loss | Best step | Recent MoE drop | Recent router entropy | Recent expert std | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `mdlm_p030_b4_lr0p666667_full` | done | **2.821899** | 9750 | 0.00033 | 1.191 | 0.0205 | Best overall so far |
| `mdlm_p030_b64_lr0p666667_full` | done | 2.943600 | 9750 | 0.00000 | 1.217 | 0.0146 | Healthy |
| `mdlm_p030_b16_lr0p666667_full` | done | 2.949728 | 9750 | 0.00198 | 1.194 | 0.0248 | Healthy |
| `mdlm_p030_b256_lr0p666667_full` | done | 2.952807 | 9750 | 0.00180 | 1.223 | 0.0177 | Healthy |
| `mdlm_p050_b4_lr0p666667_full` | running | **2.915056** | 8350 | 0.00466 | 1.200 | 0.0289 | Strong, still running |
| `mdlm_p050_b16_lr0p666667_full` | done | 2.961007 | 9200 | 0.00134 | 1.210 | 0.0258 | Healthy |
| `mdlm_p050_b256_lr0p666667_full` | done | 2.963033 | 9200 | 0.00007 | 1.225 | 0.0187 | Healthy |
| `mdlm_p050_b64_lr0p666667_full` | done | 2.967869 | 9200 | 0.00014 | 1.218 | 0.0218 | Healthy |

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
the full switch results so far do not show that later MDLM switch points are
automatically better for BD3LM.

## LR Sweep Findings

The early LR probes favored `lr_mult=0.666666666667` for MDLM-to-BD3 switches.

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

Conclusion: `lr_mult=0.6667` is the right choice for the full MDLM-to-BD3
switch sweep based on the probes. `lr_mult=2.0` remains reserved for the
scratch BD3LM baselines.

## Main Findings

These findings are for the MDLM-init track. The official sweep metric is
validation loss; MoE health is included here as a diagnostic to avoid selecting
a low-loss run that is showing router collapse or severe dropping.

### Best curriculum so far

The best observed MDLM-init curriculum so far is:

`MDLM to p=0.30 -> BD3LM block_len=4, lr_mult=0.6667`

This run has the best loss by a large margin:

- Best eval loss: `2.821899`
- Recent MoE dropped fraction: about `0.00033`
- Recent router entropy: about `1.191`
- Recent expert fraction std: about `0.0205`

It is not just the lowest-loss run; its MoE health also looks clean. There is
no sign of expert collapse or severe token dropping.

### Block length matters

Block length appears to matter strongly. Shorter block length, especially
`block_len=4`, is consistently better in the runs observed so far.

For `p=0.30`:

| Block length | Best eval loss |
| ---: | ---: |
| 4 | **2.821899** |
| 64 | 2.943600 |
| 16 | 2.949728 |
| 256 | 2.952807 |

For `p=0.50`:

| Block length | Best eval loss |
| ---: | ---: |
| 4 | **2.915056** so far, still running |
| 16 | 2.961007 |
| 256 | 2.963033 |
| 64 | 2.967869 |

The block length 4 result is best for both switch points observed so far.

### Longer MDLM pretraining has not clearly helped yet

The completed `p=0.50` switch runs at block lengths 16, 64, and 256 are
slightly worse than the corresponding `p=0.30` runs.

Current evidence:

- `p=0.30, b16`: `2.949728`
- `p=0.50, b16`: `2.961007`
- `p=0.30, b64`: `2.943600`
- `p=0.50, b64`: `2.967869`
- `p=0.30, b256`: `2.952807`
- `p=0.50, b256`: `2.963033`

The active `p=0.50, b4` run is strong, but it has not beaten `p=0.30, b4`.

## Recommendation

Interim recommendation:

1. Treat `block_len=4` as the leading BD3LM setting.
2. For MDLM init, treat `p_mdlm=0.30` as the best switch point unless later full
   runs overturn it.
3. Continue the remaining full sweep before making a final claim about p=0.80,
   p=0.90, and p=0.95.
4. Keep using `lr_mult=0.6667` for MDLM-to-BD3 switch runs.
5. Compare against the scratch BD3LM `b4` and `b16` baselines once they finish.
6. Merge this MDLM-init summary with the AR-init VM results before making the
   final curriculum call requested in `BD3_CURRICULUM_INSTRUCTIONS.md`.

Current best candidate:

`mdlm_p030_b4_lr0p666667_full`
