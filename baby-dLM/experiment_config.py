"""
experiment_config.py — Shared configuration for all experiment runners.

Centralizes MODEL_SIZES, OPTIMAL_LR, model metadata, and command building
so that sweep.py, run_scaling.py, run_isoflop.py, and generate_samples.py
all import from one place.
"""

import json
import os
import sys

# ---------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------

MODEL_MODULE_MAP = {
    "mdlm":  "model_MDLM",
    "ar":    "model_AR",
    "bd3lm": "model_bd3lm",
}

DIFFUSION_MODELS = ["mdlm"]
BLOCK_MODELS = ["bd3lm"]
ALL_MODELS = DIFFUSION_MODELS + ["ar"] + BLOCK_MODELS

BLOCK_MODEL_SET = set(BLOCK_MODELS)


def is_block_model(model):
    return model in BLOCK_MODEL_SET


# ---------------------------------------------------------------
# Model sizes: label -> (n_embd, n_layer, n_head, N_params)
#
# N_params counts only non-embedding transformer parameters:
#   N = 12 * n_layer * n_embd^2
# This includes attention (QKV + output projection) and MLP
# (two linear layers with 4x expansion) but excludes the token
# embedding table and the LM head.
#
# Head dim is always 64 for the ClimbMix sizes (50M, 98M, 170M).
# Smaller legacy sizes keep n_head=4 (variable head dim).
# ---------------------------------------------------------------

# --- Smaller legacy sizes (kept for backward compat) ---
LEGACY_MODEL_SIZES = {
    "0.1M":  (64,   2,  4,  12 * 2  * 64**2),    #  ~0.1M
    "0.3M":  (96,   3,  4,  12 * 3  * 96**2),     #  ~0.3M
    "0.5M":  (120,  3,  4,  12 * 3  * 120**2),    #  ~0.5M
    "1M":    (128,  5,  4,  12 * 5  * 128**2),     #  ~1.0M
    "2M":    (200,  4,  4,  12 * 4  * 200**2),     #  ~1.9M
    "3M":    (256,  4,  4,  12 * 4  * 256**2),     #  ~3.1M
}

# --- ClimbMix sizes (head_dim = 64 throughout) ---
CLIMBMIX_MODEL_SIZES = {
    "50M":  (768,   7, 12,  12 * 7  * 768**2),    #  49.5M
    "98M":  (896,  10, 14,  12 * 10 * 896**2),    #  96.3M
    "170M": (1024, 14, 16,  12 * 14 * 1024**2),   # 176.2M
}

# Combined registry — scripts can use ALL_SIZES or the subset they need.
MODEL_SIZES = {**LEGACY_MODEL_SIZES, **CLIMBMIX_MODEL_SIZES}

ALL_SIZES = list(MODEL_SIZES.keys())
CLIMBMIX_SIZES = list(CLIMBMIX_MODEL_SIZES.keys())

DEFAULT_BLOCK_LENS = [4, 16, 64]

# Default block_len for IsoFLOP / scaling experiments with block models.
ISOFLOP_BLOCK_LEN = 16  # plan §3 uses block_len=16 as the standard target


# ---------------------------------------------------------------
# FLOP accounting
# ---------------------------------------------------------------

# Smaller legacy-size defaults
LEGACY_ISOFLOP_BATCH_SIZE = 128
LEGACY_ISOFLOP_BLOCK_SIZE = 256

# ClimbMix regime — seq_len = 2048
# Micro-batch (what fits in GPU VRAM for a single fwd/bwd pass)
CLIMBMIX_BATCH_SIZE = 128
CLIMBMIX_BLOCK_SIZE = 2048   # MAX_SEQ_LEN from prepare.py
# Gradient accumulation: 2 micro-batches → effective batch = 256
CLIMBMIX_GRAD_ACCUM = 2
# Tokens per optimizer step (effective):
CLIMBMIX_TOKENS_PER_STEP = CLIMBMIX_BATCH_SIZE * CLIMBMIX_BLOCK_SIZE * CLIMBMIX_GRAD_ACCUM  # 524 288

# Backward-compatible aliases (used by legacy sweep scripts)
ISOFLOP_BATCH_SIZE = LEGACY_ISOFLOP_BATCH_SIZE
ISOFLOP_BLOCK_SIZE = LEGACY_ISOFLOP_BLOCK_SIZE
ISOFLOP_TOKENS_PER_STEP = ISOFLOP_BATCH_SIZE * ISOFLOP_BLOCK_SIZE  # 32 768

ISOFLOP_MIN_STEPS = 150
ISOFLOP_MAX_STEPS = 100_000   # raised for larger ClimbMix budgets

# IsoFLOP compute budgets (§5 of PLAN.md)
CLIMBMIX_ISOFLOP_BUDGETS = [1e18, 2e18, 4e18, 1e19]


def flop_multiplier(model):
    """
    Approximate C in FLOPs = C * N * tokens_processed.

    Standard diffusion / AR: C = 6 (one fwd+bwd).
    Block models (bd3lm) during training use a 2L-length input (dual-stream),
    so attention cost roughly doubles: C ≈ 12.

    N counts only non-embedding transformer parameters:
        N = 12 * n_layer * n_embd^2
    """
    return 12 if is_block_model(model) else 6


def non_embedding_params(size):
    """Return N (non-embedding transformer params) for a model size label."""
    _, _, _, N = MODEL_SIZES[size]
    return N


def tokens_for_budget(budget, model, size):
    """How many tokens to process to spend *budget* FLOPs with (model, size)."""
    N = non_embedding_params(size)
    C = flop_multiplier(model)
    return int(budget / (C * N))


def compute_isoflop_steps(budget, model, size, *, tokens_per_step=None):
    """
    Steps needed for (budget, model, size).

    *tokens_per_step* defaults to CLIMBMIX_TOKENS_PER_STEP for ClimbMix sizes
    and ISOFLOP_TOKENS_PER_STEP for legacy sizes.
    Returns None if the step count is outside [ISOFLOP_MIN_STEPS, ISOFLOP_MAX_STEPS].
    """
    if tokens_per_step is None:
        tokens_per_step = (CLIMBMIX_TOKENS_PER_STEP
                           if size in CLIMBMIX_MODEL_SIZES
                           else ISOFLOP_TOKENS_PER_STEP)
    N = non_embedding_params(size)
    C = flop_multiplier(model)
    steps = int(budget / (C * N * tokens_per_step))
    if ISOFLOP_MIN_STEPS <= steps <= ISOFLOP_MAX_STEPS:
        return steps
    return None


def dropout_for_model(model, default_dropout=0.1):
    """AR uses 0.2 by default; diffusion models use 0.1."""
    return 0.2 if model == "ar" else default_dropout


# ---------------------------------------------------------------
# Curriculum definitions  (§3 of PLAN.md)
#
# Each curriculum is a list of CurriculumStage tuples.
# A stage specifies:
#   model_family : "ar" | "mdlm" | "bd3lm"
#   block_len    : int or None (None for AR / MDLM)
#   flop_frac    : fraction of total FLOP budget spent in this stage
#
# Stage transitions warm-start from the previous stage's checkpoint
# (weights only; optimizer and LR schedule reset).
# ---------------------------------------------------------------

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class CurriculumStage:
    model_family: str
    block_len: Optional[int]   # None for AR and MDLM
    flop_frac: float           # fraction of total compute budget


@dataclass(frozen=True)
class Curriculum:
    name: str
    stages: List[CurriculumStage]
    description: str = ""

    def validate(self):
        total = sum(s.flop_frac for s in self.stages)
        assert abs(total - 1.0) < 1e-6, (
            f"Curriculum {self.name!r}: flop fractions sum to {total}, expected 1.0"
        )
        for s in self.stages:
            if s.model_family == "bd3lm":
                assert s.block_len is not None, "bd3lm stages need a block_len"
            else:
                assert s.block_len is None, (
                    f"{s.model_family} stages should not set block_len"
                )
        return self


# ---- Curriculum 0: Plain (AR → BD3-LM block_len=16) -----------
def make_curriculum_0(p_ar: float) -> Curriculum:
    """Plain two-stage: p_ar fraction AR, then (1-p_ar) BD3-LM(16)."""
    return Curriculum(
        name=f"c0_plain_p{int(p_ar*100)}",
        stages=[
            CurriculumStage("ar",    None, p_ar),
            CurriculumStage("bd3lm", 16,   1.0 - p_ar),
        ],
        description=f"Plain AR({int(p_ar*100)}%) → BD3(16)",
    ).validate()


CURRICULUM_0_SWEEPS = [make_curriculum_0(p) for p in [0.2, 0.3, 0.5, 0.8]]


# ---- Curriculum 1: Geometric doubling (SDAR / NBDiff style) ---
CURRICULUM_1 = Curriculum(
    name="c1_geometric",
    stages=[
        CurriculumStage("ar",    None, 0.2),
        CurriculumStage("bd3lm", 2,    0.2),
        CurriculumStage("bd3lm", 4,    0.2),
        CurriculumStage("bd3lm", 8,    0.2),
        CurriculumStage("bd3lm", 16,   0.2),
    ],
    description="AR → BD3(2) → BD3(4) → BD3(8) → BD3(16), 20% each",
).validate()


# ---- Curriculum 2: Aggressive early jump (LLaDA 2 style) ------
CURRICULUM_2 = Curriculum(
    name="c2_aggressive_jump",
    stages=[
        CurriculumStage("ar",    None, 0.2),
        CurriculumStage("bd3lm", 8,    0.2),
        CurriculumStage("bd3lm", 64,   0.2),
        CurriculumStage("bd3lm", 512,  0.2),
        CurriculumStage("bd3lm", 16,   0.2),
    ],
    description="AR → BD3(8) → BD3(64) → BD3(512) → BD3(16), 20% each",
).validate()


# ---- Curriculum 3: AR-heavy warmup, two diffusion stages ------
def make_curriculum_3(p_ar: float) -> Curriculum:
    """AR(p_ar) → BD3(4, 0.8-p_ar) → BD3(16, 0.2)."""
    return Curriculum(
        name=f"c3_ar_heavy_p{int(p_ar*100)}",
        stages=[
            CurriculumStage("ar",    None, p_ar),
            CurriculumStage("bd3lm", 4,    0.8 - p_ar),
            CurriculumStage("bd3lm", 16,   0.2),
        ],
        description=f"AR({int(p_ar*100)}%) → BD3(4,{int((0.8-p_ar)*100)}%) → BD3(16,20%)",
    ).validate()


CURRICULUM_3_SWEEPS = [make_curriculum_3(p) for p in [0.2, 0.3, 0.5]]


# ---- Baselines (single-family, no stage transitions) ----------
BASELINE_AR = Curriculum(
    name="baseline_ar",
    stages=[CurriculumStage("ar", None, 1.0)],
    description="Pure AR for the full budget",
).validate()

BASELINE_MDLM = Curriculum(
    name="baseline_mdlm",
    stages=[CurriculumStage("mdlm", None, 1.0)],
    description="Pure MDLM for the full budget",
).validate()

BASELINE_BD3LM = Curriculum(
    name="baseline_bd3lm_16",
    stages=[CurriculumStage("bd3lm", 16, 1.0)],
    description="Pure BD3-LM (block_len=16) for the full budget",
).validate()


# Master list for iteration
ALL_CURRICULA = (
    CURRICULUM_0_SWEEPS
    + [CURRICULUM_1, CURRICULUM_2]
    + CURRICULUM_3_SWEEPS
    + [BASELINE_AR, BASELINE_MDLM, BASELINE_BD3LM]
)


# ---------------------------------------------------------------
# Optimal LR per (model, size)
# ---------------------------------------------------------------

# Legacy LRs — from sweep at dropout=0.1 (diffusion) / 0.2 (AR).
# New sizes (0.5M, 2M) interpolated from neighbors.
# Block models initially inherit from their non-block counterparts.

_LEGACY_LR = {
    # mdlm
    ("mdlm",   "0.1M"): 1e-2,
    ("mdlm",   "0.3M"): 3e-3,
    ("mdlm",   "0.5M"): 3e-3,
    ("mdlm",   "1M"):   3e-3,
    ("mdlm",   "2M"):   3e-3,
    ("mdlm",   "3M"):   3e-3,
    # ar
    ("ar",     "0.1M"): 1e-2,
    ("ar",     "0.3M"): 1e-2,
    ("ar",     "0.5M"): 1e-2,
    ("ar",     "1M"):   1e-2,
    ("ar",     "2M"):   3e-3,
    ("ar",     "3M"):   3e-3,
    # bd3lm
    ("bd3lm",  "0.1M"): 1e-2,
    ("bd3lm",  "0.3M"): 1e-2,
    ("bd3lm",  "0.5M"): 1e-2,
    ("bd3lm",  "1M"):   1e-2,
    ("bd3lm",  "2M"):   3e-3,
    ("bd3lm",  "3M"):   3e-3,
}

# ClimbMix LRs — populated by run_lr_sweep.py (§6.3).
# Until the sweep is run, values are None (sentinel).
# After the sweep, call set_calibrated_lr() to persist results.
_CLIMBMIX_LR = {
    ("ar",    "50M"):  None,
    ("ar",    "98M"):  None,
    ("ar",    "170M"): None,
    ("mdlm",  "50M"):  None,
    ("mdlm",  "98M"):  None,
    ("mdlm",  "170M"): None,
    ("bd3lm", "50M"):  None,
    ("bd3lm", "98M"):  None,
    ("bd3lm", "170M"): None,
}

# Combined lookup — scripts always go through OPTIMAL_LR / get_optimal_lr().
OPTIMAL_LR = {**_LEGACY_LR, **_CLIMBMIX_LR}


# ---------------------------------------------------------------
# NorMuon optimizer calibration  (§4 & §6 of PLAN.md)
#
# For the normuon optimizer family, we store (adam_mult, matrix_mult)
# rather than a single scalar LR.  The realized per-group LRs are
# computed at runtime from model width via normuon.compute_normuon_lrs().
# ---------------------------------------------------------------

# Sentinel: None means not yet calibrated.
_NORMUON_CLIMBMIX = {
    ("ar",    "50M"):  None,
    ("ar",    "98M"):  None,
    ("ar",    "170M"): None,
    ("mdlm",  "50M"):  None,
    ("mdlm",  "98M"):  None,
    ("mdlm",  "170M"): None,
    ("bd3lm", "50M"):  None,
    ("bd3lm", "98M"):  None,
    ("bd3lm", "170M"): None,
}

# Master lookup: (model, size) -> {"adam_mult": float, "matrix_mult": float}
OPTIMAL_NORMUON = dict(_NORMUON_CLIMBMIX)

# ---- JSON persistence for calibrated LRs ----
# run_lr_sweep.py writes results here; later scripts load on import.
# The file lives next to this module so it travels with the repo.
_LR_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "calibrated_lrs.json")


_NORMUON_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "calibrated_normuon.json")


def _load_calibrated_lrs():
    """Load previously saved LRs from calibrated_lrs.json into OPTIMAL_LR."""
    if not os.path.exists(_LR_CACHE_PATH):
        return
    with open(_LR_CACHE_PATH) as f:
        saved = json.load(f)
    for key_str, lr in saved.items():
        # JSON keys are "model|size" strings
        model, size = key_str.split("|")
        OPTIMAL_LR[(model, size)] = lr


def _load_calibrated_normuon():
    """Load previously saved NorMuon configs from calibrated_normuon.json."""
    if not os.path.exists(_NORMUON_CACHE_PATH):
        return
    with open(_NORMUON_CACHE_PATH) as f:
        saved = json.load(f)
    for key_str, config in saved.items():
        model, size = key_str.split("|")
        OPTIMAL_NORMUON[(model, size)] = config


# Auto-load on import so downstream scripts always see persisted configs.
_load_calibrated_lrs()
_load_calibrated_normuon()


def get_optimal_lr(model, size):
    """Return calibrated LR, or None if not yet determined."""
    return OPTIMAL_LR.get((model, size))


def set_calibrated_lr(model, size, lr):
    """
    Store a newly calibrated LR and persist to calibrated_lrs.json.

    Safe to call repeatedly — the JSON file is overwritten atomically
    with the full set of calibrated ClimbMix LRs.
    """
    key = (model, size)
    if key not in OPTIMAL_LR:
        raise KeyError(f"Unknown (model, size) pair: {key}")
    OPTIMAL_LR[key] = lr
    _CLIMBMIX_LR[key] = lr
    _save_calibrated_lrs()


def _save_calibrated_lrs():
    """Persist all non-None ClimbMix LRs to disk."""
    to_save = {}
    for (model, size), lr in OPTIMAL_LR.items():
        if lr is not None and size in CLIMBMIX_MODEL_SIZES:
            to_save[f"{model}|{size}"] = lr
    tmp = _LR_CACHE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(to_save, f, indent=2)
    os.replace(tmp, _LR_CACHE_PATH)   # atomic on POSIX


def _save_calibrated_normuon():
    """Persist all non-None NorMuon configs to disk."""
    to_save = {}
    for (model, size), config in OPTIMAL_NORMUON.items():
        if config is not None and size in CLIMBMIX_MODEL_SIZES:
            to_save[f"{model}|{size}"] = config
    tmp = _NORMUON_CACHE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(to_save, f, indent=2)
    os.replace(tmp, _NORMUON_CACHE_PATH)


def get_optimal_normuon(model, size):
    """Return calibrated NorMuon config dict, or None if not yet determined."""
    return OPTIMAL_NORMUON.get((model, size))


def set_calibrated_normuon(model, size, adam_mult, matrix_mult):
    """
    Store a newly calibrated NorMuon config and persist to disk.

    The config is a dict with keys "adam_mult" and "matrix_mult".
    """
    key = (model, size)
    config = {"adam_mult": adam_mult, "matrix_mult": matrix_mult}
    OPTIMAL_NORMUON[key] = config
    _NORMUON_CLIMBMIX[key] = config
    _save_calibrated_normuon()


# ---------------------------------------------------------------
# LR sweep configuration  (§4 & §6.3 of PLAN.md)
#
# Candidate grids for the 200-step calibration runs.
# For each (model_family, size), sweep these LRs and pick the
# largest one whose grad norm stays stable.
# ---------------------------------------------------------------

# General-purpose grid (log-spaced, covers 1e-4 to 3e-2)
LR_SWEEP_GRID_DEFAULT = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]

# Per-family overrides (optional — use if prior work suggests a
# narrower search range for a specific family).
LR_SWEEP_GRIDS = {
    "ar":    [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    "mdlm":  LR_SWEEP_GRID_DEFAULT,
    "bd3lm": LR_SWEEP_GRID_DEFAULT,
}

LR_SWEEP_STEPS = 150
LR_SWEEP_WARMUP_FRAC = 0.05   # 5% linear warmup, then constant

# NorMuon 2D sweep grid (§4 of PLAN.md)
# adam_mult rescales embedding_lr, unembedding_lr, and scalar_lr.
# matrix_mult rescales matrix_lr.
NORMUON_ADAM_MULT_GRID = [0.3, 1.0, 3.0]
NORMUON_MATRIX_MULT_GRID = [0.3, 1.0, 3.0]

# Optimizer families
OPTIMIZER_FAMILIES = ["adamw", "normuon"]


# ---------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------

def build_command(
    model,
    size,
    out_dir,
    *,
    max_iters=4000,
    batch_size=None,
    block_size=None,
    block_len=None,
    grad_accum_steps=None,
    dropout=None,
    lr=None,
    eval_interval=300,
    eval_iters=50,
    warmup_iters=None,
    warmup_stable=False,
    skip_final_eval=False,
    skip_final_checkpoint=False,
    data=None,
    resume_from=None,
    gpt2_eval_interval=0,
    gpt2_eval_samples=0,
    save_interval=1000,
    num_final_samples=5,
    sample_interval=0,
    train_log_interval=100,
    optimizer="adamw",
    adam_mult=None,
    matrix_mult=None,
    normuon_weight_decay=0.2,
    use_compile=True,
):
    """
    Build a train.py command line.

    Always trains on ClimbMix data. ClimbMix defaults are used for the
    canonical 50M/98M/170M sizes; smaller legacy sizes keep their prior
    batch/accum defaults unless explicitly overridden.

    For optimizer="normuon", adam_mult and matrix_mult default to the
    calibrated values from get_optimal_normuon() if available, falling
    back to 1.0 (Karpathy reference) otherwise.
    """
    n_embd, n_layer, n_head, _ = MODEL_SIZES[size]

    if data is None:
        data = "climbmix"
    if batch_size is None:
        batch_size = (
            CLIMBMIX_BATCH_SIZE
            if size in CLIMBMIX_MODEL_SIZES
            else LEGACY_ISOFLOP_BATCH_SIZE
        )
    if block_size is None:
        block_size = CLIMBMIX_BLOCK_SIZE
    if grad_accum_steps is None:
        grad_accum_steps = CLIMBMIX_GRAD_ACCUM if size in CLIMBMIX_MODEL_SIZES else 1

    # Warmup: default to 5% of max_iters (plan §4/§5 policy), min 1.
    # Legacy default of 100 is preserved when warmup_iters is passed explicitly.
    if warmup_iters is None:
        warmup_iters = max(1, int(max_iters * LR_SWEEP_WARMUP_FRAC))

    if optimizer == "normuon":
        # Load calibrated NorMuon config if multipliers not explicitly given.
        if adam_mult is None or matrix_mult is None:
            calibrated = get_optimal_normuon(model, size)
            if calibrated is not None:
                if adam_mult is None:
                    adam_mult = calibrated["adam_mult"]
                if matrix_mult is None:
                    matrix_mult = calibrated["matrix_mult"]
            else:
                # Fall back to Karpathy reference (1.0, 1.0)
                if adam_mult is None:
                    adam_mult = 1.0
                if matrix_mult is None:
                    matrix_mult = 1.0
        # For normuon, the per-group LRs are computed from adam_mult/matrix_mult
        # at runtime.  We still need a dummy scalar LR for the schedule.
        if lr is None:
            lr = 1.0  # schedule multiplier base; actual LRs come from normuon
        min_lr = lr / 10.0
    else:
        # adamw path — multipliers are unused
        if adam_mult is None:
            adam_mult = 1.0
        if matrix_mult is None:
            matrix_mult = 1.0
        if lr is None:
            lr = get_optimal_lr(model, size)
        if lr is None:
            raise ValueError(f"No calibrated LR for ({model}, {size}). "
                             "Run the LR sweep first (§6.3).")
        min_lr = lr / 10.0
    if dropout is None:
        dropout = dropout_for_model(model)

    loss_path = os.path.join(out_dir, "loss.pkl")
    ckpt_path = os.path.join(out_dir, "ckpt.pt")

    cmd = [
        sys.executable, "-u", "train.py",
        "--data", data,
        "--model", model,
        "--n_embd", str(n_embd),
        "--n_layer", str(n_layer),
        "--n_head", str(n_head),
        "--dropout", str(dropout),
        "--learning_rate", str(lr),
        "--min_lr", str(min_lr),
        "--batch_size", str(batch_size),
        "--block_size", str(block_size),
        "--max_iters", str(max_iters),
        "--eval_interval", str(eval_interval),
        "--train_log_interval", str(train_log_interval),
        "--eval_iters", str(eval_iters),
        "--warmup_iters", str(warmup_iters),
        "--gpt2_eval_interval", str(gpt2_eval_interval),
        "--gpt2_eval_samples", str(gpt2_eval_samples),
        "--sample_interval", str(sample_interval),
        "--save_interval", str(save_interval) if save_interval > 0 else str(max_iters + 1),
        "--num_final_samples", str(num_final_samples),
        "--loss_log_path", loss_path,
        "--checkpoint_path", ckpt_path,
        "--grad_accum_steps", str(grad_accum_steps),
    ]

    # train.py defines these with type=str2bool, so pass explicit values.
    cmd.extend(["--warmup_stable", "true" if warmup_stable else "false"])
    cmd.extend(["--skip_final_eval", "true" if skip_final_eval else "false"])
    cmd.extend(["--skip_final_checkpoint", "true" if skip_final_checkpoint else "false"])
    cmd.extend(["--use_compile", "true" if use_compile else "false"])

    cmd.extend(["--optimizer", optimizer])
    if optimizer == "normuon":
        cmd.extend(["--adam_mult", str(adam_mult)])
        cmd.extend(["--matrix_mult", str(matrix_mult)])
        cmd.extend(["--normuon_weight_decay", str(normuon_weight_decay)])

    if block_len is not None:
        cmd.extend(["--block_len", str(block_len)])

    if resume_from is not None:
        cmd.extend(["--resume_from", str(resume_from)])

    return cmd


# ---------------------------------------------------------------
# Size / curriculum compatibility
# ---------------------------------------------------------------

# Block lengths that only make sense at ClimbMix seq_len (2048).
# Legacy sizes use block_size=256, so block_len must be <= 256.
_LEGACY_MAX_BLOCK_LEN = 256


def check_size_curriculum_compat(size: str, curriculum: Curriculum):
    """
    Raise ValueError if *curriculum* uses block_lens that are
    incompatible with *size*'s sequence length.

    For example, a legacy 0.1M model (block_size=256) cannot run a
    stage with block_len=512.
    """
    if size in CLIMBMIX_MODEL_SIZES:
        return  # ClimbMix block_size=2048 supports all planned block_lens
    for stage in curriculum.stages:
        if stage.block_len is not None and stage.block_len > _LEGACY_MAX_BLOCK_LEN:
            raise ValueError(
                f"Curriculum {curriculum.name!r} has a stage with "
                f"block_len={stage.block_len}, which exceeds the legacy "
                f"block_size={_LEGACY_MAX_BLOCK_LEN} for size {size!r}. "
                f"Use a ClimbMix size ({', '.join(CLIMBMIX_SIZES)}) instead."
            )


# ---------------------------------------------------------------
# Convenience: build a command for a single curriculum stage
# ---------------------------------------------------------------

def build_stage_command(
    stage: CurriculumStage,
    size: str,
    total_budget: float,
    out_dir: str,
    *,
    resume_from: str = None,
    **kwargs,
):
    """
    Build a train.py command for one curriculum stage.

    Computes max_iters from the stage's flop_frac and the total budget.
    Raises ValueError if the stage's block_len is incompatible with *size*.
    """
    # Guard: reject block_lens that exceed the size's sequence length
    if stage.block_len is not None and size not in CLIMBMIX_MODEL_SIZES:
        if stage.block_len > _LEGACY_MAX_BLOCK_LEN:
            raise ValueError(
                f"block_len={stage.block_len} exceeds legacy block_size="
                f"{_LEGACY_MAX_BLOCK_LEN} for size {size!r}"
            )

    tokens_per_step = (CLIMBMIX_TOKENS_PER_STEP
                       if size in CLIMBMIX_MODEL_SIZES
                       else ISOFLOP_TOKENS_PER_STEP)
    stage_budget = total_budget * stage.flop_frac
    N = non_embedding_params(size)
    C = flop_multiplier(stage.model_family)
    stage_tokens = int(stage_budget / (C * N))
    stage_steps = max(1, stage_tokens // tokens_per_step)

    return build_command(
        model=stage.model_family,
        size=size,
        out_dir=out_dir,
        max_iters=stage_steps,
        block_len=stage.block_len,
        warmup_stable=True,   # plan §5: warmup-stable for all IsoFLOP runs
        resume_from=resume_from,
        **kwargs,
    )
