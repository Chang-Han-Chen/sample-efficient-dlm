# PLAN: Efficient JAX Backbone and Optimizer for Language Diffusion Models

This document is written for a future AI agent working in this repository. Keep working independently until the `jax/` implementation is mathematically faithful to the PyTorch reference where intended, supports the planned diffusion experiments, and is efficient enough to make the experiment matrix finish quickly on a single A100 VM.

## Goal

Find an efficient transformer backbone plus optimizer recipe for language diffusion models (LDMs).

Part 1 is about the backbone, optimizer, data pipeline, and controlled ablations. Do not spend time on new LDM training algorithms beyond implementing MDLM and BD3LM faithfully enough (use `baby-dLM/` as the reference) to compare the same transformer interventions under diffusion training.

The core scientific question:

> Do architecture and optimization interventions that help autoregressive language modeling also help diffusion language modeling?

The engineering question:

> Can the JAX implementation run the required experiments faster, with better memory behavior, and with scalability hooks that make multi-device work straightforward later?

## Repository Map

- `pytorch/`: original `sample_efficient_gpt` PyTorch implementation. Treat this as the reference for the current autoregressive model and optimizer behavior.
- `jax/`: active JAX port. At the time of writing, `jax/transformer/` contains `core.py`, `attention.py`, `rope.py`, and `transformer.py`.
- `baby-dLM/`: PyTorch implementations and experiments for MDLM, and BD3LM. Use this for diffusion objective semantics and BD3 masking details. Notice that the current code is not perfect though! I struggled with FlexAttention and triton kernels. Refer to the original BD3LM repo [https://github.com/kuleshov-group/bd3lms.git] for more accurate details, especially code in `bd3lms/models/`.
- `karpathy/`: reference single-GPU training and data preparation code. Use `prepare.py` as a reference for downloading Nvidia ClimbMix and training a BPE tokenizer, but adapt it to this project.

## Non-Negotiables

1. Do not blindly port `pytorch/sample_efficient_gpt/transformer/ops/` to JAX.
2. Profile seriously before adding custom kernels.
3. Prefer native JAX/XLA/cuDNN first. Write Pallas or other manual kernels only for a measured bottleneck.
4. Keep the PyTorch and JAX models mathematically aligned unless a planned ablation intentionally differs.
5. Optimize for experiment runtime after profiling and implementation, not just single-step elegance. So something like runtime for 50 steps might be a good proxy.
6. Use NorMuon+AdamW as the default optimizer path. Do not spend time on pure AdamW except as a named optimizer ablation if explicitly requested later.

## Current Position on `ops/`

Do not recreate `jax/transformer/ops/` as a copied Triton folder.

The likely strategy is:

- RMSNorm, SwiGLU, residual add, QK norm, and ordinary cross entropy: implement in pure JAX and rely on XLA fusion under `jit`.
- Attention: use `jax.nn.dot_product_attention`; make the implementation selectable. Try `"cudnn"` on supported NVIDIA GPU, default/fallback to `"xla"` elsewhere. Avoid explicit GQA key/value repetition if JAX DPA can handle grouped query attention directly.
- Fused linear cross entropy: keep this as the one serious candidate for a custom memory optimization. Start with a pure JAX chunked loss over the `(B*T)` axis using `lax.scan` or `lax.fori_loop` and a `custom_vjp`. Only write Pallas after profiling proves the pure JAX chunked version is too slow or too memory-hungry.

Important nuance: ordinary CE fusion does not solve peak memory if the model has already materialized `(B*T, vocab)` logits. If memory is the limiter, refactor the model/loss API so the model can return final hidden states and the loss owns the `hidden @ lm_head.weight.T` projection chunking.

## Equivalence Protocol

Do not rely on "same random seed" alone to establish equivalence. Different frameworks and initializers can diverge enough to make loss comparisons noisy.

Required checks:

1. Unit-level parity: copy PyTorch weights into JAX modules and compare outputs for `Linear`, `Embedding`, `RMSNorm`, RoPE, attention variants, block, and full transformer.
2. Loss parity: with copied weights and identical token batches, compare logits/loss for the AR objective.
3. Training parity: train the baseline PyTorch and JAX models on the same data order for a short run, initially 50 to 200 optimizer steps after warmup/compilation. Compare loss curves, final loss, throughput, and memory.
4. Optimizer parity: verify NorMuon+AdamW parameter grouping, schedules, width scaling, momentum warmup, cautious weight decay, and scalar/embedding/unembedding groups against the PyTorch implementation.

The expected final losses should be nearly identical only when initial parameters, data order, loss formula, optimizer state, precision policy, and schedule are matched. If any of those differ, compare distributions/trends and document the reason.

## Profiling Protocol

Run performance work on a VM with one A100.

Always report:

- Model config, sequence length, vocab size, dtype, optimizer, and JAX/PyTorch versions.
- Effective global tokens per optimizer step.
- Device microbatch size and gradient accumulation steps.
- Tokens/sec.
- Step time split if possible: data loading, forward, backward, optimizer, compile.
- MFU or model FLOP utilization. State the exact FLOP estimate and peak-FLOP denominator used.
- Peak HBM usage and memory headroom.
- JAX compilation time separately from steady-state step time.
- Whether loss includes full logits, chunked logits, z-loss, softcap, BPB reporting, or diffusion reweighting.

Benchmarking rules:

- Exclude one-time compilation from steady-state throughput, but log it.
- Use enough steps for stable numbers. For early work, 50 measured steps is acceptable after warmup; for final comparisons, use longer windows.
- Sweep sequence length and batch shape for runtime, but keep experiments scientifically meaningful. Sequence length 512 or 768 is acceptable for intervention validation. Batch size 1 is not acceptable as an experiment proxy.
- Search for the largest useful microbatch that fits with good MFU. If memory bound, prefer changes that increase effective batch per GPU without changing the experimental conclusion.

## Data Plan

Use one simplified stage-1 data pipeline based on Nvidia ClimbMix.

Reference: `karpathy/prepare.py` downloads from `karpathy/climbmix-400b-shuffle` and trains a BPE tokenizer. Reuse the ideas, not the exact assumptions.

Requirements:

- Train/use a project tokenizer with base vocab size `32768`.
- For diffusion runs, add one extra diffusion mask token, so diffusion `vocab_size = 32768 + 1 = 32769`.
- Produce deterministic train/validation splits.
- Store artifacts in a predictable cache or repo-configured data directory.
- Integrate with JAX training without forcing the PyTorch trainer abstractions into JAX.
- Preserve enough metadata to reproduce a run: tokenizer config, shard list, validation shard/list, BPE pattern, special tokens, and seed.
- Support fast streaming batches for one A100 without making data loading the bottleneck.

Suggested implementation shape:

- `jax/data/prepare_climbmix.py`: download shards, train tokenizer, write metadata.
- `jax/data/loader.py`: deterministic token stream and batch iterator for AR, MDLM, and BD3LM.
- `jax/configs/`: small typed configs or YAMLs for model, data, optimizer, and experiment sweeps.

## Config Tracking

All experiment configs must be written before launching the sweep. Do not rely on ad hoc CLI flags or notebook state as the source of truth.

Use a structured config tree, for example:

```text
jax/configs/
  data/climbmix_32768.yaml
  model/gpt_small_70m.yaml
  optimizer/normuon_adamw.yaml
  schedule/ws_100_5k.yaml
  experiments/
    ar_baseline.yaml
    ar_old_bundle.yaml
    ar_value_embedding.yaml
    ar_non_mup_init.yaml
    mdlm_*.yaml
    bd3lm_*.yaml
```

Each run config should fully resolve to:

- model architecture and intervention flags
- tokenizer/data paths and vocab size
- objective (`ar`, `mdlm`, `bd3lm`)
- optimizer group LR base values and LR sweep multiplier
- schedule
- sequence length, effective batch size, microbatch size, and grad accumulation
- dtype, attention implementation, checkpointing/remat settings, and loss implementation
- W&B project/entity/group/tags
- output/checkpoint directory
- seed

At launch time, save the resolved config next to local checkpoints and log it to W&B.

## Backbone Plan

The JAX transformer should support:

- Current sample-efficient GPT features:
  - RMSNorm
  - RoPE
  - SwiGLU
  - fixed attention head layout for the chosen architecture
  - QK norm
  - value residual
  - attention gating modes
  - optional weight tying
  - depth/layernorm scaling if retained from the PyTorch code
- New ablation feature:
  - value embeddings, based on the `karpathy/train.py` implementation unless clarified otherwise.
- Initialization ablation:
  - current sample-efficient/muP-ish initialization
  - non-muP or "variance inversely proportional to hidden dimension" initialization variant

Value embeddings need special care. The intended ablation is now the clean combined version: compute the normalized value-residual mixture first, using the raw first-layer value stream as the residual anchor, and only then add token value embeddings into the attention value stream. Do not cache or reuse a first-layer value stream that already includes value embeddings. This keeps the value-residual channel and token-value-embedding channel disentangled.

### Candidate Old Architecture Interventions

The old architecture interventions visible in the PyTorch configs/code are:

1. QK-norm: RMS-normalize query and key vectors before attention, with a learnable scalar gain on Q.
2. Value residual: reuse the first layer's value stream in later attention layers through learnable scalar mixing.
3. Attention gating: apply a learned gate to the attention output before the output projection. The final configs use `per-head`.
4. LayerNorm/depth scaling: scale RMSNorm outputs by `rsqrt(layer_position)` when `layernorm_scaling=True`.
5. Weight tying: share token embedding and LM head weights. Note that the original PyTorch implementation appears to have a wrapper bug here, so JAX should implement the intended behavior and parity tests should handle the PyTorch no-op carefully.

NorMuon is not an architecture ablation for this project. It is the default optimizer.

GQA is not an ablation axis. Once the architecture is fixed, the head layout is fixed too. For the GPT sanity runs, use the small GPT architecture from `pytorch/sample_efficient_gpt/configs/gpt_small_faster.py`, approximately 70M params in the original setup:

```text
d_model = 768
d_ff = 2048
n_layers = 8
n_heads = 12
n_kv_heads = None unless explicitly fixed otherwise
```

That original config's old intervention bundle is QK-norm + value residual + layernorm/depth scaling. Attention gating and weight tying exist in larger/final configs, so include them only if the user confirms they belong in the small GPT sanity bundle.

Please confirm whether this is the complete "old intervention" set before running the final matrix.

### Value Embedding Options

Use these formulas to decide what "value embeddings" should mean before implementation.

Let `h_l` be the normalized layer input, `ids` the token ids, and:

```text
Q_l = h_l W_Q_l
K_l = h_l W_K_l
V_l = h_l W_V_l
```

Option A: no value embedding, current ordinary attention value path.

```text
V_attn_l = V_l
Y_l = Attention(Q_l, K_l, V_attn_l)
```

Option B: existing sample-efficient GPT value residual.

```text
V_first = raw value stream returned by the first layer, before any token value-embedding addition
V_attn_l = s_l * (a_l * V_l + b_l * V_first) / sqrt(a_l^2 + b_l^2 + eps)
```

Initialization in the PyTorch/JAX port is `a_l=1`, `b_l=0`, `s_l=1`, so this starts as a no-op.

Option C: Karpathy-style token value embeddings.

For selected layers, usually alternating layers with the last layer included:

```text
E_l(ids) in R[B, T, n_kv_heads, head_dim]
g_l(h_l) = 2 * sigmoid(h_l[..., :gate_channels] W_gate_l)
V_attn_l = V_l + g_l(h_l)[..., None] * E_l(ids)
```

This adds a token-conditioned value vector directly into the attention value stream. Gate weights are initialized to zero, so `g_l = 1` at init; the path is controlled mostly by the value-embedding initialization scale.

Option D: chosen combined value-residual plus token value-embedding path.

Normalize the value-residual mixture before adding token value embeddings:

```text
V_first = raw first-layer value stream, before any token value-embedding addition
V_res_l = s_l * (a_l * V_l + b_l * V_first) / sqrt(a_l^2 + b_l^2 + eps)
V_attn_l = V_res_l + gamma_ve_l * g_l(h_l)[..., None] * E_l(ids)
Y_l = Attention(Q_l, K_l, V_attn_l)
```

This is the default value-embedding ablation for this plan. It isolates the question "does adding token value embeddings help on top of the existing old-intervention recipe?" while preserving the existing value-residual normalization behavior.

Implementation details:

- Cache `V_first` from the raw first-layer `V_l = h_l W_V_l`, before adding any token value embedding.
- Apply the `s_l * (...) / sqrt(a_l^2 + b_l^2 + eps)` normalization only to the local/first-layer value-residual mixture.
- Add token value embeddings after that normalization. Do not include `E_1(ids)` inside the value-residual anchor unless running a separate explicit ablation.
- Initialize value residual as before: `a_l=1`, `b_l=0`, `s_l=1`.
- Expose `gamma_ve_l` or an equivalent global `value_embedding_scale` in config. For exact old-bundle initialization, set it to `0`; for a more literal Karpathy-style run, set it to `1` and control the path mostly through the value-embedding initialization scale.
- Use the Karpathy-style layer placement unless profiling or ablations say otherwise: alternating layers with the final layer always included.

## Non-muP Initialization Research Note

The user cited a claim from "Pre-training Under Infinite Compute" saying that, when scaling models, Marin changed initialization to have variance inversely proportional to hidden dimension and that this outperformed muP in their framework.

Initial research:

- Marin issue 621 is titled "MuP for scaling laws" and was closed as completed. The final public comment says the conclusion was "not worth it" because their heuristic LR performed well.
- A Marin issue comment summarizes a CerebrasGPT-style muP recipe involving scaled linear weights, scaled logits, embedding output scaling, `1/d_head` attention scaling, and extra scaling for output projections and FFN down projections.
- The paper quote should be treated as a hypothesis to test in this codebase, not as settled evidence.

Action for the agent:

1. Read the relevant Marin issue, linked WandB report if accessible, CerebrasGPT recipe, and any code changes in the linked Levanter/Haliax PRs.
2. Determine what "non-muP init" should mean concretely here.
3. Compare against the current PyTorch/JAX initialization. Note that the current `Linear` implementation already uses `std = 1 / sqrt(fan_in)`, which gives variance proportional to `1/fan_in`; this may or may not match the intended non-muP variant depending on output projections, embeddings, logits, and residual-path scaling.
4. Add the initialization mode as a named config option, not a hard-coded replacement.

Useful links:

- https://github.com/marin-community/marin/issues/621
- https://openreview.net/pdf/e28fd04869f97f7a613d3a543a11d0649727c510.pdf
- https://arxiv.org/pdf/2304.03208

## Optimizer Plan

Default optimizer is NorMuon+AdamW.

Port or reimplement:

- Muon matrix groups for 2D transformer matrix parameters.
- AdamW groups for token embeddings, value embeddings, unembedding/lm head, scalar/vector params, and other non-matrix params.
- Width-based LR scaling from the PyTorch/Karpathy paths where intended.
- Momentum warmup.
- Cautious weight decay for Muon.
- Schedule support used in the ablations.

For Part 1 experiments, use a warmup-stable schedule:

```text
steps 0..99:    linear warmup from 0 to peak LR
steps 100..5099: constant peak LR
total:          5100 optimizer steps
```

Sweep peak learning rate. Because NorMuon+AdamW has multiple parameter groups, it might get tricky here. Read `karpathy/train.py` to understand how he scales the lr and the knobs we can tune. I expect AdamW has 1 lr, and NorMuon has 2. Start around the small GPT reference recipe or Karpathy's, and then sweep, for example:

```text
lr_mult in [0.25, 0.5, 1.0, 2.0, 4.0]
```

I suggest you to monitor the lr-sweep runs closely, because often times it is clear if a lr is good very early on. You can terminate the runs as soon as the result is obvious, e.g., grad norm too high, loss decay too slowly, etc. This would save a lot of time.

Track at minimum train loss, eval loss, and global grad norm for every run.

Keep optimizer ablations separate from the main port. The immediate baseline should not be pure AdamW.

## Diffusion Plan

Add MDLM and BD3LM training paths that share the same JAX transformer backbone.

Use `baby-dLM/` as the semantic reference (but use [https://github.com/kuleshov-group/bd3lms.git] whenever confused; in fact, you might find it helpful to keep it on the side):

- `model_MDLM.py`: MDLM objective and masking/noising behavior.
- `model_bd3lm.py`, `backbone.py`, and `block_utils.py`: BD3 objective, block-diffusion masks, dual-stream training, and block-causal sampling.
- Tests in `baby-dLM/tests/`: use as behavioral clues for edge cases.

Architectural interventions must be toggled in the shared backbone, not duplicated separately for AR/MDLM/BD3LM.

Very importantly, you should be careful about the implementation of attention mechanism!!
MDLM uses no masking attention, while BD3LM uses a sparse 4-quadrant masking. In PyTorch, both could be efficient via FlashAttention and FlexAttention. But, our code is in Jax, and I am less certain about how it would be implemented. In particular, for BD3LM, because the input length is doubled at fixed sequence length (if you don't know why, read `baby-dLM/` first), the attention memory is very costly. In my experience, the batch size per step (resp. grad accumulations) need to be much smaller (resp. larger), so BD3LM trains very slowly. You should try very hard to optimize the wall-clock efficiency of BD3LM runs (which i think the key is to reduce peak memory so that grad accumulation can be small).
We know the shape of all matrices because we fix the architechture config. This means that we could write hacky kernels that work at this particular model size (~70M as mentioned above). I was annoyed by FlexAttention because autocasting takes so long, but in our case, we can precompute the best setting beforehand.

## Experiment Plan

Phase A: JAX AR parity and speed

1. Finish the JAX training stack for AR.
2. Verify parity with PyTorch on copied weights and identical batches.
3. Benchmark PyTorch vs JAX on one A100.
4. Profile memory and throughput.
5. Optimize only measured bottlenecks.

Before ablations, profile sequence length and batch shape:

1. Compare `seq_len=512` and `seq_len=768` on the A100. Based on the findings, feel free to try others or move on.
2. Keep total effective batch fixed at 512 sequences for the profiling target, using gradient accumulation as needed. This means `512 * seq_len` tokens per optimizer step.
3. Find the largest per-device microbatch that fits with healthy memory headroom.
4. Pick the sequence length/microbatch/grad-accum setting that gives the shortest stable optimizer-step time without making the experiment scientifically meaningless.

Phase B: GPT/AR sanity matrix

Run exactly four GPT/AR runs unless profiling reveals a serious flaw:

1. Baseline: old architecture interventions off, NorMuon+AdamW on.
2. Old intervention bundle for the fixed small GPT architecture. For `gpt_small_faster`, this is at least QK-norm + value residual + layernorm/depth scaling.
3. Add value embeddings: start from the old intervention bundle and use Option D above, i.e. normalize the value-residual mixture first, then add token value embeddings. Keep value embeddings only if they improve.
4. Try non-muP init: start from the current best of runs 2-3 and switch only the initialization mode.

Each of these four configurations needs an LR sweep under the 100-warmup + 5k-constant schedule. Compare each config by its best LR run, while keeping all LR sweep results logged.

Do not rerun old GPT ablations one by one. They were already done before; this is only a sanity check that the bundled old interventions still beat the baseline in the JAX setup, plus a test of the two new ideas.

Phase C: MDLM and BD3LM

1. Add MDLM and BD3LM objectives on top of the same JAX backbone.
2. Validate small deterministic cases against `baby-dLM/`.
3. Benchmark and optimize enough that the diffusion runs are not dominated by avoidable overhead.

Phase D: Diffusion ablations

Run serious one-by-one diffusion ablations under MDLM and BD3LM.

Use a greedy sequential design, not a `2^N` factorial design:

1. Start with the diffusion baseline.
2. Add one candidate intervention.
3. If it improves the chosen logged metrics, keep it for subsequent runs.
4. If it does not improve, discard it for subsequent runs.
5. Continue through the intervention list in a documented order.

This assumes positive interventions mostly stay positive with or without the others. If results look order-sensitive or contradictory, pause and document the issue rather than exploding into a full factorial matrix.

The goal is to decide whether the AR-winning architecture plus optimizer is also the right LDM backbone recipe.

## Suggested Metrics for Scientific Runs

Record at least:

- Train loss curve.
- Validation loss or bits-per-byte on a fixed validation set.
- Global grad norm.
- Tokens/sec and total wall time.
- Peak memory.
- Effective batch size and sequence length.
- Run seed and data shard/window order.
- Any compile time or first-step anomalies.

Use Weights & Biases for run tracking if available:

- Log all train/eval losses as scalar histories.
- Log the full config, git status/commit when available, profiler settings, and hardware metadata.
- Log throughput, MFU, peak memory, compile time, grad accumulation, global grad norm, LR multiplier, and tokens per optimizer step.
- Store checkpoints locally and upload at least final and best checkpoints as W&B artifacts. Uploading every checkpoint is useful only if artifact storage and bandwidth are acceptable; prefer sparse periodic checkpoints for large sweeps.
- Use consistent tags/groups: `ar`, `mdlm`, `bd3lm`, `baseline`, `old_bundle`, `value_embedding`, `non_mup_init`, `seq512`, `seq768`, etc.

For diffusion:

- Objective-specific loss.
- If available, a comparable validation metric across MDLM/BD3LM.
- Mask/noise schedule config.
- Block length for BD3LM.

## Done Criteria for Part 1

Part 1 is done when:

1. JAX AR baseline is parity-checked against PyTorch and benchmarked on one A100.
2. JAX AR baseline is at least as efficient as PyTorch for the chosen experimental regime, or any remaining gap is profiled and justified.
3. Data preparation and loading for ClimbMix with vocab size 32768 are reproducible.
4. NorMuon+AdamW is the default optimizer in JAX.
5. All planned run configs are written before the first real sweep.
6. Value embeddings and initialization mode are config-controlled.
7. MDLM and BD3LM train on the shared JAX backbone.
8. The selected AR and diffusion ablation matrix has been run or is runnable with clear commands.
9. Results are logged in a way that supports choosing the final architecture plus optimizer recipe for LDMs.

## Questions for the User

1. Where should large data artifacts live on the A100 VM: default user cache, project-local path, or a mounted volume?
2. What W&B entity/project should be used?
