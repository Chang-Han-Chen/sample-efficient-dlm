"""
Tests for run_curriculum.py — the curriculum runner (§6.4).
"""

import json
import os
import pickle
import sys
import pytest

# Ensure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import experiment_config as ec
from run_curriculum import (
    CURRICULUM_REGISTRY,
    CurriculumResult,
    StageResult,
    compute_stage_steps,
    list_curricula,
    parse_stage_stdout,
    resolve_lr_config,
    _collect_lr_configs,
    _save_results,
)


# ---------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------

class TestCurriculumRegistry:
    def test_all_curricula_registered(self):
        """Every curriculum in ec.ALL_CURRICULA should be in the registry."""
        for cur in ec.ALL_CURRICULA:
            assert cur.name in CURRICULUM_REGISTRY, (
                f"Curriculum {cur.name!r} not found in registry"
            )

    def test_registry_size(self):
        """Registry should have at least the baselines + c0 sweeps + c1 + c2 + c3 sweeps."""
        # 4 c0 sweeps + 1 c1 + 1 c2 + 3 c3 sweeps + 3 baselines = 12
        assert len(CURRICULUM_REGISTRY) >= 12

    def test_baselines_present(self):
        assert "baseline_ar" in CURRICULUM_REGISTRY
        assert "baseline_mdlm" in CURRICULUM_REGISTRY
        assert "baseline_bd3lm_16" in CURRICULUM_REGISTRY


# ---------------------------------------------------------------
# Stdout parsing
# ---------------------------------------------------------------

class TestParseStageStdout:
    SAMPLE_OUTPUT = (
        "step 100 | tok_epoch 0.03 | loss 7.1234 | grad_norm 1.23 | lr 0.001000\n"
        "step 200 | tok_epoch 0.06 | loss 6.8912 | grad_norm 0.98 | lr 0.001000\n"
        "step 200 | tok_epoch 0.06 | train 6.8912 | val 7.0010 | lr 0.001000\n"
        "step 300 | tok_epoch 0.09 | loss 6.5400 | grad_norm 0.85 | lr 0.001000\n"
        "step 300 | tok_epoch 0.09 | train 6.5400 | val 6.7800 | lr 0.001000\n"
    )

    def test_parse_train_losses(self):
        train, val, gn = parse_stage_stdout(self.SAMPLE_OUTPUT)
        assert len(train) == 3
        assert train[0] == (100, 7.1234)
        assert train[-1] == (300, 6.5400)

    def test_parse_val_losses(self):
        train, val, gn = parse_stage_stdout(self.SAMPLE_OUTPUT)
        assert len(val) == 2
        assert val[0] == (200, 7.0010)
        assert val[-1] == (300, 6.7800)

    def test_parse_grad_norms(self):
        train, val, gn = parse_stage_stdout(self.SAMPLE_OUTPUT)
        assert len(gn) == 3
        assert gn[0] == (100, 1.23)

    def test_empty_output(self):
        train, val, gn = parse_stage_stdout("")
        assert train == [] and val == [] and gn == []

    # --- P1 regression tests: forced final eval with "(final)" suffix ---

    FINAL_EVAL_OUTPUT = (
        "step 100 | tok_epoch 0.03 | loss 7.1234 | grad_norm 1.23 | lr 0.001000\n"
        "step 100 | tok_epoch 0.03 | train 7.0000 | val 7.2000 | lr 0.001000\n"
        "step 200 | tok_epoch 0.06 | loss 6.8912 | grad_norm 0.98 | lr 0.001000\n"
        "step 250 (final) | train 6.5000 | val 6.7000\n"
    )

    def test_final_eval_val_loss_parsed(self):
        """The (final) eval line's val loss should be captured."""
        train, val, gn = parse_stage_stdout(self.FINAL_EVAL_OUTPUT)
        # val should include both the mid-training eval (step 100) and the final (step 250)
        assert len(val) == 2
        assert val[-1] == (250, 6.7000)

    def test_final_eval_train_loss_parsed(self):
        """The (final) eval line's train loss should be captured."""
        train, val, gn = parse_stage_stdout(self.FINAL_EVAL_OUTPUT)
        # train should include step 100 (from _STEP_RE), step 200 (from _STEP_RE),
        # and step 250 (from _FINAL_EVAL_RE)
        assert len(train) == 3
        assert train[-1] == (250, 6.5000)

    def test_final_eval_is_last_entry(self):
        """final_val_loss on CurriculumResult should reflect the (final) eval."""
        train, val, gn = parse_stage_stdout(self.FINAL_EVAL_OUTPUT)
        result = CurriculumResult(
            curriculum_name="test",
            size="50M",
            total_budget=1e18,
            optimizer_family="adamw",
            lr_config={"ar": {"lr": 3e-3}},
            stages=[
                StageResult(
                    stage_index=0,
                    model_family="ar",
                    block_len=None,
                    flop_frac=1.0,
                    max_iters=250,
                    checkpoint_path="/tmp/ckpt.pt",
                    train_losses=train,
                    val_losses=val,
                    grad_norms=gn,
                ),
            ],
        )
        assert result.final_val_loss == 6.7000
        assert result.final_train_loss == 6.5000

    def test_final_eval_with_gpt2(self):
        """Final eval line may include an extra gpt2_ce field; still parses."""
        output = "step 500 (final) | train 6.1234 | val 6.3456 | gpt2_ce 0.9876\n"
        train, val, gn = parse_stage_stdout(output)
        assert val == [(500, 6.3456)]
        assert train == [(500, 6.1234)]

    def test_no_final_eval_still_works(self):
        """If skip_final_eval is set, there's no (final) line — should degrade gracefully."""
        output = (
            "step 100 | tok_epoch 0.03 | loss 7.1234 | grad_norm 1.23 | lr 0.001000\n"
            "step 100 | tok_epoch 0.03 | train 7.0000 | val 7.2000 | lr 0.001000\n"
        )
        train, val, gn = parse_stage_stdout(output)
        assert val == [(100, 7.2000)]
        assert train == [(100, 7.1234)]

    def test_short_stage_only_final(self):
        """A very short stage might only emit a final eval and no mid-training logs."""
        output = "step 5 (final) | train 8.9000 | val 9.1000\n"
        train, val, gn = parse_stage_stdout(output)
        assert train == [(5, 8.9000)]
        assert val == [(5, 9.1000)]
        assert gn == []


# ---------------------------------------------------------------
# Stage step computation
# ---------------------------------------------------------------

class TestComputeStageSteps:
    def test_ar_stage_50m(self):
        """AR stage: C=6, N=non_embedding_params(50M). Steps follow the FLOP budget formula."""
        stage = ec.CurriculumStage("ar", None, 0.2)
        steps = compute_stage_steps(stage, "50M", 1e18)
        N = ec.non_embedding_params("50M")
        expected = int(1e18 * 0.2 / (6 * N)) // ec.CLIMBMIX_TOKENS_PER_STEP
        assert steps == expected
        assert steps > 0

    def test_bd3lm_stage_50m(self):
        """BD3-LM stage: C=12 (dual-stream), so fewer steps at same budget."""
        stage = ec.CurriculumStage("bd3lm", 16, 0.2)
        steps = compute_stage_steps(stage, "50M", 1e18)
        N = ec.non_embedding_params("50M")
        expected = int(1e18 * 0.2 / (12 * N)) // ec.CLIMBMIX_TOKENS_PER_STEP
        assert steps == expected

    def test_full_budget_single_stage(self):
        """A baseline (1 stage, frac=1.0) should use the full budget."""
        stage = ec.CurriculumStage("ar", None, 1.0)
        steps = compute_stage_steps(stage, "50M", 1e18)
        N = ec.non_embedding_params("50M")
        expected = int(1e18 / (6 * N)) // ec.CLIMBMIX_TOKENS_PER_STEP
        assert steps == expected

    def test_minimum_one_step(self):
        """Even a tiny budget should produce at least 1 step."""
        stage = ec.CurriculumStage("ar", None, 0.01)
        steps = compute_stage_steps(stage, "170M", 1e10)  # very small
        assert steps >= 1


# ---------------------------------------------------------------
# LR config resolution
# ---------------------------------------------------------------

class TestResolveLrConfig:
    def test_adamw_legacy_model(self):
        """Legacy models have pre-set LRs."""
        cfg = resolve_lr_config("adamw", "ar", "1M")
        assert "lr" in cfg
        assert cfg["lr"] == ec.get_optimal_lr("ar", "1M")

    def test_adamw_uncalibrated_raises(self):
        """Uncalibrated ClimbMix sizes should raise ValueError."""
        # Only raises if the calibrated_lrs.json doesn't have it
        lr = ec.get_optimal_lr("ar", "50M")
        if lr is None:
            with pytest.raises(ValueError, match="No calibrated"):
                resolve_lr_config("adamw", "ar", "50M")

    def test_normuon_uncalibrated_raises(self):
        """Uncalibrated NorMuon should raise ValueError."""
        cfg = ec.get_optimal_normuon("ar", "50M")
        if cfg is None:
            with pytest.raises(ValueError, match="No calibrated"):
                resolve_lr_config("normuon", "ar", "50M")

    def test_unknown_optimizer_raises(self):
        with pytest.raises(ValueError, match="Unknown optimizer"):
            resolve_lr_config("sgd", "ar", "1M")


class TestCollectLrConfigs:
    def test_single_family_curriculum(self):
        """Baseline AR uses only the 'ar' family."""
        # Use legacy sizes that have pre-calibrated LRs
        configs = _collect_lr_configs(ec.BASELINE_AR, "1M", "adamw")
        assert "ar" in configs
        assert len(configs) == 1

    def test_multi_family_curriculum(self):
        """Curriculum 1 uses both 'ar' and 'bd3lm' families."""
        # This only works if both are calibrated; test logic only
        families = {s.model_family for s in ec.CURRICULUM_1.stages}
        assert "ar" in families
        assert "bd3lm" in families


# ---------------------------------------------------------------
# CurriculumResult
# ---------------------------------------------------------------

class TestCurriculumResult:
    def _make_result(self):
        return CurriculumResult(
            curriculum_name="test_cur",
            size="50M",
            total_budget=1e18,
            optimizer_family="adamw",
            lr_config={"ar": {"lr": 1e-2}, "bd3lm": {"lr": 3e-3}},
            stages=[
                StageResult(
                    stage_index=0,
                    model_family="ar",
                    block_len=None,
                    flop_frac=0.5,
                    max_iters=1000,
                    checkpoint_path="/tmp/s0/ckpt.pt",
                    train_losses=[(500, 6.5), (1000, 6.0)],
                    val_losses=[(500, 6.8), (1000, 6.3)],
                    grad_norms=[(500, 1.0), (1000, 0.9)],
                    wall_time_seconds=120.0,
                    return_code=0,
                ),
                StageResult(
                    stage_index=1,
                    model_family="bd3lm",
                    block_len=16,
                    flop_frac=0.5,
                    max_iters=500,
                    checkpoint_path="/tmp/s1/ckpt.pt",
                    train_losses=[(250, 5.8), (500, 5.5)],
                    val_losses=[(250, 6.0), (500, 5.7)],
                    grad_norms=[(250, 0.8), (500, 0.7)],
                    wall_time_seconds=90.0,
                    return_code=0,
                ),
            ],
        )

    def test_final_val_loss(self):
        result = self._make_result()
        assert result.final_val_loss == 5.7

    def test_final_train_loss(self):
        result = self._make_result()
        assert result.final_train_loss == 5.5

    def test_total_wall_time(self):
        result = self._make_result()
        assert result.total_wall_time == 210.0

    def test_summary_dict(self):
        result = self._make_result()
        d = result.summary_dict()
        assert d["curriculum"] == "test_cur"
        assert d["final_val_loss"] == 5.7
        assert len(d["stages"]) == 2
        assert d["stages"][1]["block_len"] == 16

    def test_empty_stages(self):
        result = CurriculumResult(
            curriculum_name="empty",
            size="50M",
            total_budget=1e18,
            optimizer_family="adamw",
            lr_config={},
        )
        assert result.final_val_loss is None
        assert result.final_train_loss is None
        assert result.total_wall_time == 0.0

    # --- P2 regression tests: per-family LR configs preserved ---

    def test_lr_config_preserves_all_families(self):
        """lr_config should retain configs for every model family in the curriculum."""
        result = self._make_result()
        assert "ar" in result.lr_config
        assert "bd3lm" in result.lr_config
        assert result.lr_config["ar"] == {"lr": 1e-2}
        assert result.lr_config["bd3lm"] == {"lr": 3e-3}

    def test_summary_dict_includes_all_families(self):
        """The JSON summary should include LR settings for all families."""
        result = self._make_result()
        d = result.summary_dict()
        assert "ar" in d["lr_config"]
        assert "bd3lm" in d["lr_config"]
        assert d["lr_config"]["ar"]["lr"] == 1e-2
        assert d["lr_config"]["bd3lm"]["lr"] == 3e-3

    def test_normuon_multi_family_lr_config(self):
        """NorMuon configs should store per-family (adam_mult, matrix_mult)."""
        result = CurriculumResult(
            curriculum_name="c0_normuon",
            size="50M",
            total_budget=1e18,
            optimizer_family="normuon",
            lr_config={
                "ar": {"adam_mult": 1.0, "matrix_mult": 0.3},
                "bd3lm": {"adam_mult": 3.0, "matrix_mult": 1.0},
            },
        )
        assert result.lr_config["ar"]["adam_mult"] == 1.0
        assert result.lr_config["bd3lm"]["matrix_mult"] == 1.0
        d = result.summary_dict()
        assert d["lr_config"]["ar"]["matrix_mult"] == 0.3
        assert d["lr_config"]["bd3lm"]["adam_mult"] == 3.0


# ---------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------

class TestSaveResults:
    def test_save_creates_files(self, tmp_path):
        result = CurriculumResult(
            curriculum_name="test",
            size="50M",
            total_budget=1e18,
            optimizer_family="adamw",
            lr_config={"ar": {"lr": 3e-3}},
            stages=[
                StageResult(
                    stage_index=0,
                    model_family="ar",
                    block_len=None,
                    flop_frac=1.0,
                    max_iters=100,
                    checkpoint_path=str(tmp_path / "ckpt.pt"),
                    train_losses=[(100, 6.0)],
                    val_losses=[(100, 6.2)],
                ),
            ],
        )
        out_dir = str(tmp_path / "output")
        _save_results(result, out_dir)

        # Check pickle
        pkl_path = os.path.join(out_dir, "curriculum_result.pkl")
        assert os.path.exists(pkl_path)
        with open(pkl_path, "rb") as f:
            loaded = pickle.load(f)
        assert loaded.curriculum_name == "test"
        assert loaded.final_val_loss == 6.2

        # Check JSON
        json_path = os.path.join(out_dir, "curriculum_summary.json")
        assert os.path.exists(json_path)
        with open(json_path) as f:
            summary = json.load(f)
        assert summary["final_val_loss"] == 6.2
        assert summary["num_stages"] == 1

    def test_save_preserves_all_lr_configs(self, tmp_path):
        """Saved pickle and JSON should retain all per-family LR configs."""
        result = CurriculumResult(
            curriculum_name="mixed",
            size="50M",
            total_budget=1e18,
            optimizer_family="adamw",
            lr_config={"ar": {"lr": 1e-2}, "bd3lm": {"lr": 3e-3}},
            stages=[
                StageResult(
                    stage_index=0, model_family="ar", block_len=None,
                    flop_frac=0.5, max_iters=100,
                    checkpoint_path=str(tmp_path / "s0.pt"),
                    train_losses=[(100, 6.0)], val_losses=[(100, 6.2)],
                ),
                StageResult(
                    stage_index=1, model_family="bd3lm", block_len=16,
                    flop_frac=0.5, max_iters=50,
                    checkpoint_path=str(tmp_path / "s1.pt"),
                    train_losses=[(50, 5.5)], val_losses=[(50, 5.7)],
                ),
            ],
        )
        out_dir = str(tmp_path / "output")
        _save_results(result, out_dir)

        # Check pickle round-trip
        with open(os.path.join(out_dir, "curriculum_result.pkl"), "rb") as f:
            loaded = pickle.load(f)
        assert loaded.lr_config == {"ar": {"lr": 1e-2}, "bd3lm": {"lr": 3e-3}}

        # Check JSON
        with open(os.path.join(out_dir, "curriculum_summary.json")) as f:
            summary = json.load(f)
        assert summary["lr_config"]["ar"]["lr"] == 1e-2
        assert summary["lr_config"]["bd3lm"]["lr"] == 3e-3
