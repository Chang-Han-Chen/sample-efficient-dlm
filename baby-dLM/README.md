# dLM

Scaling laws for language diffusion models. This repo compares autoregressive, masked diffusion (MDLM), and block diffusion (BD3-LM) language models, with support for AR-to-diffusion curriculum training.

Implementation note: the BD3 attention masks and dual-stream training path are aligned with the upstream [kuleshov-group/bd3lms](https://github.com/kuleshov-group/bd3lms) formulation, but this repo intentionally uses a simplified training/sampling stack. In particular, MDLM/BD3-LM training is implemented as masked-token cross-entropy, and generation uses confidence-based progressive unmasking rather than the upstream semi-autoregressive first-hitting/DDPM-style reverse update.

## Three models

Each variant shares the same transformer backbone (RoPE, QK-norm, ReGLU MLP, RMSNorm); they differ only in training objective and generation strategy.

- **AR** (`model_AR.py`): Standard next-token prediction GPT. Causal attention, left-to-right generation.
- **MDLM** (`model_MDLM.py`): Masked Diffusion LM. Trains on cross-entropy over masked positions. At generation, tokens are progressively unmasked from high to low confidence — once committed, a token is final.
- **BD3-LM** (`model_bd3lm.py`): Block Discrete Denoising Diffusion LM. Splits the sequence into blocks and denoises them left-to-right with cross-block causal attention. Uses MDLM-style progressive unmasking within each block.

## Data

All training uses ClimbMix (BPE tokenized Hugging Face parquet shards). `prepare.py` downloads shards and trains the tokenizer into repo-local `data_cache/`.

```bash
python prepare.py                  # download 10 shards + train tokenizer
python prepare.py --num-shards 8   # download only 8 shards (for testing)
```

## Repo structure

```
train.py                 Training loop (all 3 models)
backbone.py              Shared transformer backbone (standard + dual-stream)
block_utils.py           BD3-LM attention masks
model_AR.py              Autoregressive model
model_MDLM.py            Masked diffusion model
model_bd3lm.py           Block diffusion model (BD3-LM)
experiment_config.py     Shared config (sizes, LRs, FLOP multipliers)
prepare.py               ClimbMix data download + BPE tokenizer training
reproduce.ipynb          Generates data and figures
tests/                   Test suite
```
