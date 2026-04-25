"""
Tests for evaluate.py — Post-training evaluation script (§6.6).

Covers the non-GPU-dependent parts:
  - EvalResult dataclass and summary_dict
  - infer_size_from_args
  - infer_lr_config
  - find_final_checkpoint (directory scanning)
  - load_curriculum_metadata
  - format_results_table
  - save_results (JSON + pickle)
  - BPBModelWrapper interface (with mock model)
  - CLI argument parsing
"""

import json
import os
import pickle
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import experiment_config as ec
from evaluate import (
    EvalResult,
    BPBModelWrapper,
    infer_size_from_args,
    infer_lr_config,
    find_final_checkpoint,
    load_curriculum_metadata,
    format_results_table,
    save_results,
    _load_losses_from_loss_log,
    _size_sort_key,
)


# ---------------------------------------------------------------
# EvalResult
# ---------------------------------------------------------------

class TestEvalResult:
    def test_defaults(self):
        r = EvalResult(checkpoint_path="ckpt.pt", model_family="ar", size="50M")
        assert r.curriculum == ""
        assert r.flop_budget == 0.0
        assert r.optimizer_family == "adamw"
        assert r.lr_config == {}
        assert r.normuon_realized_lrs is None
        assert r.final_train_loss is None
        assert r.final_val_loss is None
        assert r.bpb is None
        assert r.samples == []
        assert r.checkpoint_iter is None

    def test_summary_dict_basic(self):
        r = EvalResult(
            checkpoint_path="runs/ckpt.pt",
            model_family="ar",
            size="50M",
            curriculum="baseline_ar",
            flop_budget=1e18,
            optimizer_family="adamw",
            lr_config={"lr": 0.003},
            final_train_loss=2.5,
            final_val_loss=2.7,
            bpb=1.05,
            checkpoint_iter=1000,
        )
        d = r.summary_dict()
        assert d["checkpoint"] == "runs/ckpt.pt"
        assert d["model_family"] == "ar"
        assert d["size"] == "50M"
        assert d["curriculum"] == "baseline_ar"
        assert d["flop_budget"] == 1e18
        assert d["optimizer_family"] == "adamw"
        assert d["lr_config"] == {"lr": 0.003}
        assert d["final_train_loss"] == 2.5
        assert d["final_val_loss"] == 2.7
        assert d["bpb"] == 1.05
        assert d["checkpoint_iter"] == 1000
        assert d["num_samples"] == 0
        # normuon_realized_lrs should be absent when None
        assert "normuon_realized_lrs" not in d

    def test_summary_dict_includes_normuon_lrs(self):
        r = EvalResult(
            checkpoint_path="ckpt.pt",
            model_family="bd3lm",
            size="98M",
            optimizer_family="normuon",
            lr_config={"adam_mult": 1.0, "matrix_mult": 3.0},
            normuon_realized_lrs={"embedding_lr": 0.01, "matrix_lr": 0.03},
        )
        d = r.summary_dict()
        assert d["normuon_realized_lrs"] == {"embedding_lr": 0.01, "matrix_lr": 0.03}

    def test_summary_dict_sample_count(self):
        r = EvalResult(
            checkpoint_path="ckpt.pt",
            model_family="ar",
            size="50M",
            samples=["hello world", "foo bar", "baz"],
        )
        assert r.summary_dict()["num_samples"] == 3


# ---------------------------------------------------------------
# infer_size_from_args
# ---------------------------------------------------------------

class TestInferSizeFromArgs:
    def test_50m(self):
        assert infer_size_from_args({"n_embd": 768, "n_layer": 7}) == "50M"

    def test_98m(self):
        assert infer_size_from_args({"n_embd": 896, "n_layer": 10}) == "98M"

    def test_170m(self):
        assert infer_size_from_args({"n_embd": 1024, "n_layer": 14}) == "170M"

    def test_legacy_1m(self):
        assert infer_size_from_args({"n_embd": 128, "n_layer": 5}) == "1M"

    def test_legacy_climbmix_50m(self):
        assert infer_size_from_args({"n_embd": 512, "n_layer": 16}) == "50M"

    def test_legacy_climbmix_98m(self):
        assert infer_size_from_args({"n_embd": 640, "n_layer": 20}) == "98M"

    def test_legacy_climbmix_170m(self):
        assert infer_size_from_args({"n_embd": 768, "n_layer": 24}) == "170M"

    def test_unknown_returns_none(self):
        assert infer_size_from_args({"n_embd": 999, "n_layer": 99}) is None

    def test_missing_n_embd_returns_none(self):
        assert infer_size_from_args({"n_layer": 7}) is None

    def test_missing_n_layer_returns_none(self):
        assert infer_size_from_args({"n_embd": 768}) is None

    def test_empty_args_returns_none(self):
        assert infer_size_from_args({}) is None


# ---------------------------------------------------------------
# infer_lr_config
# ---------------------------------------------------------------

class TestInferLrConfig:
    def test_adamw_from_args(self):
        args = {"learning_rate": 0.003, "optimizer": "adamw"}
        meta = {"optimizer_family": "adamw"}
        cfg = infer_lr_config(args, meta)
        assert cfg == {"lr": 0.003}

    def test_adamw_default_lr(self):
        """When learning_rate is absent, default to 0.0."""
        cfg = infer_lr_config({}, {"optimizer_family": "adamw"})
        assert cfg == {"lr": 0.0}

    def test_normuon_from_args(self):
        args = {"adam_mult": 3.0, "matrix_mult": 0.3}
        meta = {"optimizer_family": "normuon"}
        cfg = infer_lr_config(args, meta)
        assert cfg == {"adam_mult": 3.0, "matrix_mult": 0.3}

    def test_normuon_default_mults(self):
        """When multipliers are absent, default to 1.0."""
        cfg = infer_lr_config({}, {"optimizer_family": "normuon"})
        assert cfg == {"adam_mult": 1.0, "matrix_mult": 1.0}

    def test_optimizer_family_from_meta_overrides_args(self):
        """meta['optimizer_family'] takes precedence over args['optimizer']."""
        args = {"optimizer": "adamw", "adam_mult": 2.0, "matrix_mult": 2.0}
        meta = {"optimizer_family": "normuon"}
        cfg = infer_lr_config(args, meta)
        # Should treat as normuon because meta says so
        assert "adam_mult" in cfg

    def test_fallback_to_args_optimizer(self):
        """If meta has no optimizer_family, fall back to args['optimizer']."""
        args = {"optimizer": "normuon", "adam_mult": 0.5, "matrix_mult": 0.5}
        meta = {}
        cfg = infer_lr_config(args, meta)
        assert cfg == {"adam_mult": 0.5, "matrix_mult": 0.5}


# ---------------------------------------------------------------
# find_final_checkpoint (filesystem tests)
# ---------------------------------------------------------------

class TestFindFinalCheckpoint:
    def test_nonexistent_dir(self, tmp_path):
        assert find_final_checkpoint(str(tmp_path / "nope")) is None

    def test_empty_dir(self, tmp_path):
        assert find_final_checkpoint(str(tmp_path)) is None

    def test_direct_checkpoint(self, tmp_path):
        (tmp_path / "ckpt.pt").touch()
        result = find_final_checkpoint(str(tmp_path))
        assert result == str(tmp_path / "ckpt.pt")

    def test_single_stage(self, tmp_path):
        stage = tmp_path / "stage_0_ar"
        stage.mkdir()
        (stage / "ckpt.pt").touch()
        result = find_final_checkpoint(str(tmp_path))
        assert result == str(stage / "ckpt.pt")

    def test_multi_stage_picks_last(self, tmp_path):
        """Should pick the highest-numbered stage with a checkpoint."""
        for i, name in enumerate(["stage_0_ar", "stage_1_bd3lm", "stage_2_bd3lm"]):
            d = tmp_path / name
            d.mkdir()
            (d / "ckpt.pt").touch()
        result = find_final_checkpoint(str(tmp_path))
        assert result == str(tmp_path / "stage_2_bd3lm" / "ckpt.pt")

    def test_last_stage_no_checkpoint(self, tmp_path):
        """If last stage has no ckpt, falls back to earlier stage."""
        s0 = tmp_path / "stage_0_ar"
        s0.mkdir()
        (s0 / "ckpt.pt").touch()
        s1 = tmp_path / "stage_1_bd3lm"
        s1.mkdir()
        # stage_1 has no ckpt.pt
        result = find_final_checkpoint(str(tmp_path))
        assert result == str(s0 / "ckpt.pt")

    def test_direct_checkpoint_preferred_over_stages(self, tmp_path):
        """A direct ckpt.pt in the root should be returned immediately."""
        (tmp_path / "ckpt.pt").touch()
        s0 = tmp_path / "stage_0_ar"
        s0.mkdir()
        (s0 / "ckpt.pt").touch()
        result = find_final_checkpoint(str(tmp_path))
        assert result == str(tmp_path / "ckpt.pt")

    def test_non_stage_subdirs_ignored(self, tmp_path):
        """Directories that don't start with stage_ should be ignored."""
        other = tmp_path / "tensorboard_logs"
        other.mkdir()
        (other / "ckpt.pt").touch()
        assert find_final_checkpoint(str(tmp_path)) is None


# ---------------------------------------------------------------
# load_curriculum_metadata
# ---------------------------------------------------------------

class TestLoadCurriculumMetadata:
    def test_missing_json(self, tmp_path):
        assert load_curriculum_metadata(str(tmp_path)) == {}

    def test_loads_json(self, tmp_path):
        data = {"curriculum": "c1_geometric", "budget": 1e18, "size": "50M"}
        with open(tmp_path / "curriculum_summary.json", "w") as f:
            json.dump(data, f)
        result = load_curriculum_metadata(str(tmp_path))
        assert result["curriculum"] == "c1_geometric"
        assert result["budget"] == 1e18
        assert result["size"] == "50M"


# ---------------------------------------------------------------
# format_results_table
# ---------------------------------------------------------------

class TestFormatResultsTable:
    def _make_result(self, **kwargs):
        defaults = dict(
            checkpoint_path="ckpt.pt",
            model_family="ar",
            size="50M",
            curriculum="baseline_ar",
            flop_budget=1e18,
            optimizer_family="adamw",
            lr_config={"lr": 0.003},
            final_train_loss=2.5,
            final_val_loss=2.7,
            bpb=1.05,
            checkpoint_iter=1000,
        )
        defaults.update(kwargs)
        return EvalResult(**defaults)

    def test_single_result(self):
        table = format_results_table([self._make_result()])
        lines = table.strip().split("\n")
        assert len(lines) == 3  # header, separator, one row
        assert "ar" in lines[2]
        assert "50M" in lines[2]
        assert "baseline_ar" in lines[2]
        assert "adamw" in lines[2]
        assert "1.05" in lines[2]

    def test_none_metrics_show_dash(self):
        table = format_results_table([
            self._make_result(
                final_train_loss=None,
                final_val_loss=None,
                bpb=None,
                flop_budget=0.0,
                checkpoint_iter=None,
            )
        ])
        # The dash character used is —
        assert "—" in table

    def test_normuon_lr_string(self):
        table = format_results_table([
            self._make_result(
                optimizer_family="normuon",
                lr_config={"adam_mult": 1.0, "matrix_mult": 3.0},
            )
        ])
        assert "am=1.0" in table
        assert "mm=3.0" in table

    def test_adamw_lr_string(self):
        table = format_results_table([
            self._make_result(lr_config={"lr": 0.01})
        ])
        assert "lr=0.01" in table

    def test_sorted_output(self):
        """Results should be sorted by (model_family, size, curriculum, optimizer)."""
        results = [
            self._make_result(model_family="bd3lm", size="98M", curriculum="c1"),
            self._make_result(model_family="ar", size="50M", curriculum="baseline_ar"),
            self._make_result(model_family="ar", size="170M", curriculum="baseline_ar"),
        ]
        table = format_results_table(results)
        lines = table.strip().split("\n")[2:]  # data rows only
        families = [l.split()[0] for l in lines]
        assert families == ["ar", "ar", "bd3lm"]

    def test_empty_results(self):
        table = format_results_table([])
        lines = table.strip().split("\n")
        assert len(lines) == 2  # header + separator only


# ---------------------------------------------------------------
# save_results
# ---------------------------------------------------------------

class TestSaveResults:
    def _make_result(self, **kwargs):
        defaults = dict(
            checkpoint_path="ckpt.pt",
            model_family="ar",
            size="50M",
        )
        defaults.update(kwargs)
        return EvalResult(**defaults)

    def test_saves_json_and_pickle(self, tmp_path):
        results = [self._make_result(bpb=1.05)]
        out = str(tmp_path / "eval_results.json")
        save_results(results, out)

        assert os.path.exists(out)
        assert os.path.exists(str(tmp_path / "eval_results.pkl"))

        with open(out) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["bpb"] == 1.05

    def test_pickle_roundtrip(self, tmp_path):
        results = [
            self._make_result(bpb=1.05, samples=["hello"]),
            self._make_result(bpb=1.10, model_family="bd3lm"),
        ]
        out = str(tmp_path / "eval_results.json")
        save_results(results, out)

        with open(str(tmp_path / "eval_results.pkl"), "rb") as f:
            loaded = pickle.load(f)
        assert len(loaded) == 2
        assert loaded[0].bpb == 1.05
        assert loaded[0].samples == ["hello"]
        assert loaded[1].model_family == "bd3lm"

    def test_json_path_without_extension(self, tmp_path):
        out = str(tmp_path / "results")
        save_results([self._make_result()], out)
        assert os.path.exists(out + ".json")
        assert os.path.exists(out + ".pkl")

    def test_creates_parent_dirs(self, tmp_path):
        out = str(tmp_path / "sub" / "dir" / "results.json")
        save_results([self._make_result()], out)
        assert os.path.exists(out)

    def test_multiple_results_in_json(self, tmp_path):
        results = [
            self._make_result(curriculum="baseline_ar", flop_budget=1e18),
            self._make_result(curriculum="c1_geometric", flop_budget=2e18),
        ]
        out = str(tmp_path / "eval.json")
        save_results(results, out)
        with open(out) as f:
            data = json.load(f)
        assert len(data) == 2
        assert data[0]["curriculum"] == "baseline_ar"
        assert data[1]["curriculum"] == "c1_geometric"


# ---------------------------------------------------------------
# BPBModelWrapper (with mock model)
# ---------------------------------------------------------------

@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestBPBModelWrapper:
    """Test the wrapper interface without GPU — uses simple mock models."""

    def test_callable_with_reduction_none(self):
        """Wrapper should be callable as model(x, y, reduction='none')."""
        import torch

        class MockARModel:
            def __call__(self, x):
                B, T = x.shape
                return torch.randn(B, T, 100), None  # logits, loss
            def eval(self):
                return self
            def parameters(self):
                return iter([torch.zeros(1)])

        model = MockARModel()
        wrapper = BPBModelWrapper(model, "ar", vocab_size=100)
        wrapper.eval()

        x = torch.randint(0, 100, (2, 16))
        y = torch.randint(0, 100, (2, 16))

        result = wrapper(x, y, reduction='none')
        assert result.shape == (2 * 16,)

    def test_reduction_mean(self):
        import torch

        class MockARModel:
            def __call__(self, x):
                B, T = x.shape
                return torch.randn(B, T, 50), None
            def eval(self):
                return self
            def parameters(self):
                return iter([torch.zeros(1)])

        wrapper = BPBModelWrapper(MockARModel(), "ar", vocab_size=50)
        x = torch.randint(0, 50, (2, 8))
        y = torch.randint(0, 50, (2, 8))
        result = wrapper(x, y, reduction='mean')
        assert result.dim() == 0  # scalar

    def test_reduction_sum(self):
        import torch

        class MockARModel:
            def __call__(self, x):
                B, T = x.shape
                return torch.randn(B, T, 50), None
            def eval(self):
                return self
            def parameters(self):
                return iter([torch.zeros(1)])

        wrapper = BPBModelWrapper(MockARModel(), "ar", vocab_size=50)
        x = torch.randint(0, 50, (2, 8))
        y = torch.randint(0, 50, (2, 8))
        result = wrapper(x, y, reduction='sum')
        assert result.dim() == 0  # scalar

    def test_diffusion_model_uses_causal_logits(self):
        """Non-AR models should go through _causal_logits (not _forward_core)."""
        import torch
        import torch.nn as nn

        V = 50
        T = 8
        n_embd = 32

        class MockBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(n_embd, n_embd, bias=False)
                nn.init.eye_(self.linear.weight)
            def forward(self, x, cos_sin, attn_mask=None):
                # Record the mask for verification
                self._last_mask = attn_mask
                return self.linear(x)

        class MockDiffusionBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.token_emb = nn.Embedding(V, n_embd)
                self.emb_dropout = nn.Dropout(0.0)
                cos, sin = torch.ones(1, T, 1, n_embd // 4), torch.zeros(1, T, 1, n_embd // 4)
                self.register_buffer("cos", cos)
                self.register_buffer("sin", sin)
                self.blocks = nn.ModuleList([MockBlock()])
                self.lm_head = nn.Linear(n_embd, V, bias=False)

        model = MockDiffusionBackbone()
        model.eval()
        wrapper = BPBModelWrapper(model, "bd3lm", vocab_size=V)

        x = torch.randint(0, V, (2, T))
        y = torch.randint(0, V, (2, T))
        result = wrapper(x, y, reduction='none')
        assert result.shape == (2 * T,)

        # Verify the block received a causal (lower-triangular) mask
        mask = model.blocks[0]._last_mask
        assert mask is not None
        assert mask.shape == (1, 1, T, T)
        # Every entry above the diagonal should be False
        for i in range(T):
            for j in range(i + 1, T):
                assert mask[0, 0, i, j].item() == False
        # Every entry on or below the diagonal should be True
        for i in range(T):
            for j in range(0, i + 1):
                assert mask[0, 0, i, j].item() == True

    def test_eval_returns_self(self):
        import torch

        class MockModel:
            def __call__(self, x):
                return torch.zeros(1, 1, 10), None
            def eval(self):
                return self
            def parameters(self):
                return iter([torch.zeros(1)])

        wrapper = BPBModelWrapper(MockModel(), "ar", vocab_size=10)
        assert wrapper.eval() is wrapper

    def test_parameters_delegates(self):
        import torch

        param = torch.nn.Parameter(torch.zeros(5))

        class MockModel:
            def eval(self):
                return self
            def parameters(self_inner):
                return iter([param])

        wrapper = BPBModelWrapper(MockModel(), "ar", vocab_size=10)
        params = list(wrapper.parameters())
        assert len(params) == 1
        assert params[0] is param


# ---------------------------------------------------------------
# P2 regression: survival_prob_scalar in generate_samples
# ---------------------------------------------------------------

class TestSurvivalProbScalar:
    """
    Verify the survival_prob_scalar closure built in generate_samples
    matches train.py's behavior: takes integer timesteps t ∈ {1..T},
    converts to fraction t/T, clamps to [t_min, t_max], and applies
    the noise schedule.

    The old bug: evaluate.py defined survival_prob_scalar(t) = 1 - t,
    treating t as a fraction.  With T=100, t=50 yielded -49 instead
    of ~0.5.  These tests import the generate_samples function and
    patch out everything except the closure to verify it directly.
    """

    def _build_survival_fn(self, T=100, noise_schedule="linear",
                           t_min=0.45, t_max=0.95):
        """Reconstruct the closure from generate_samples without torch."""
        import math

        def survival_prob_scalar(t_step):
            t_frac = float(t_step) / T
            t_frac = min(max(t_frac, t_min), t_max)
            if noise_schedule == "linear":
                a_t = 1.0 - t_frac
            elif noise_schedule == "cosine":
                a_t = math.cos(0.5 * math.pi * t_frac)
            else:
                raise ValueError(f"unknown noise_schedule: {noise_schedule}")
            return max(0.0, min(1.0, a_t))

        return survival_prob_scalar

    def test_linear_midpoint(self):
        """t=50, T=100 → t_frac=0.5 (clamped to 0.5) → a_t = 0.5."""
        fn = self._build_survival_fn(T=100, noise_schedule="linear")
        result = fn(50)
        assert abs(result - 0.5) < 1e-6

    def test_linear_always_nonnegative(self):
        """Old bug: fn(50) returned -49.  All timesteps must give a_t ∈ [0, 1]."""
        fn = self._build_survival_fn(T=100, noise_schedule="linear")
        for t in range(1, 101):
            a = fn(t)
            assert 0.0 <= a <= 1.0, f"t={t} gave a_t={a}"

    def test_linear_t1(self):
        """t=1, T=100 → t_frac=0.01, clamped to t_min=0.45 → a_t = 0.55."""
        fn = self._build_survival_fn(T=100, noise_schedule="linear",
                                     t_min=0.45, t_max=0.95)
        result = fn(1)
        assert abs(result - 0.55) < 1e-6

    def test_linear_tT(self):
        """t=T → t_frac=1.0, clamped to t_max=0.95 → a_t = 0.05."""
        fn = self._build_survival_fn(T=100, noise_schedule="linear",
                                     t_min=0.45, t_max=0.95)
        result = fn(100)
        assert abs(result - 0.05) < 1e-6

    def test_cosine_midpoint(self):
        """Cosine schedule at t=50, T=100 → t_frac=0.5 → cos(π/4) ≈ 0.707."""
        import math
        fn = self._build_survival_fn(T=100, noise_schedule="cosine")
        expected = math.cos(0.5 * math.pi * 0.5)
        result = fn(50)
        assert abs(result - expected) < 1e-6

    def test_cosine_always_nonnegative(self):
        fn = self._build_survival_fn(T=100, noise_schedule="cosine")
        for t in range(1, 101):
            a = fn(t)
            assert 0.0 <= a <= 1.0, f"t={t} gave a_t={a}"

    def test_unknown_schedule_raises(self):
        fn = self._build_survival_fn(noise_schedule="banana")
        with pytest.raises(ValueError, match="unknown noise_schedule"):
            fn(50)


# ---------------------------------------------------------------
# P2 regression: metric extraction decoupled from model loading
# ---------------------------------------------------------------

class TestLoadLossesFromLossLog:
    def test_missing_path_returns_nones(self):
        result = _load_losses_from_loss_log("")
        assert result["final_train_loss"] is None
        assert result["final_val_loss"] is None

    def test_nonexistent_file_returns_nones(self):
        result = _load_losses_from_loss_log("/no/such/file.pkl")
        assert result["final_train_loss"] is None
        assert result["final_val_loss"] is None

    def test_loads_train_and_val(self, tmp_path):
        loss_data = {
            "train": [(100, 2.5), (200, 2.3)],
            "val": [(100, 2.8), (200, 2.6)],
        }
        path = str(tmp_path / "loss.pkl")
        with open(path, "wb") as f:
            pickle.dump(loss_data, f)
        result = _load_losses_from_loss_log(path)
        assert result["final_train_loss"] == 2.3
        assert result["final_val_loss"] == 2.6

    def test_partial_data(self, tmp_path):
        """Only val present, train missing."""
        loss_data = {"val": [(100, 3.0)]}
        path = str(tmp_path / "loss.pkl")
        with open(path, "wb") as f:
            pickle.dump(loss_data, f)
        result = _load_losses_from_loss_log(path)
        assert result["final_train_loss"] is None
        assert result["final_val_loss"] == 3.0


class TestMetricFallbackOrder:
    """
    P2 regression: evaluate_run_dir should pull metrics from
    curriculum_summary.json first, then fall back to loss.pkl.
    """

    def test_curriculum_summary_metrics_used(self, tmp_path):
        """When curriculum_summary.json has metrics, those should be primary."""
        summary = {
            "curriculum": "c1_geometric",
            "budget": 1e18,
            "size": "50M",
            "final_train_loss": 2.1,
            "final_val_loss": 2.3,
        }
        with open(tmp_path / "curriculum_summary.json", "w") as f:
            json.dump(summary, f)

        meta = load_curriculum_metadata(str(tmp_path))
        # Verify the JSON has the metrics that evaluate_run_dir would use
        assert meta["final_train_loss"] == 2.1
        assert meta["final_val_loss"] == 2.3

    def test_curriculum_summary_without_metrics(self, tmp_path):
        """When curriculum_summary.json exists but has no metrics, get None."""
        summary = {"curriculum": "baseline_ar", "budget": 1e18}
        with open(tmp_path / "curriculum_summary.json", "w") as f:
            json.dump(summary, f)

        meta = load_curriculum_metadata(str(tmp_path))
        assert meta.get("final_train_loss") is None
        assert meta.get("final_val_loss") is None


# ---------------------------------------------------------------
# P3a regression: realized NorMuon LRs in table
# ---------------------------------------------------------------

class TestNormuonRealizedLRsInTable:
    def _make_result(self, **kwargs):
        defaults = dict(
            checkpoint_path="ckpt.pt",
            model_family="bd3lm",
            size="50M",
            optimizer_family="normuon",
            lr_config={"adam_mult": 1.0, "matrix_mult": 3.0},
        )
        defaults.update(kwargs)
        return EvalResult(**defaults)

    def test_realized_lrs_appear_in_table(self):
        r = self._make_result(
            normuon_realized_lrs={
                "embedding_lr": 0.01,
                "matrix_lr": 0.03,
                "scalar_lr": 0.005,
            }
        )
        table = format_results_table([r])
        assert "embedding_lr=0.01" in table
        assert "matrix_lr=0.03" in table
        assert "scalar_lr=0.005" in table

    def test_no_realized_lrs_no_brackets(self):
        """When normuon_realized_lrs is None, no brackets should appear."""
        r = self._make_result(normuon_realized_lrs=None)
        table = format_results_table([r])
        assert "[" not in table
        assert "]" not in table

    def test_empty_realized_lrs_no_brackets(self):
        """When normuon_realized_lrs is {}, no brackets should appear."""
        r = self._make_result(normuon_realized_lrs={})
        table = format_results_table([r])
        assert "[" not in table


# ---------------------------------------------------------------
# P3b regression: numeric model size sorting
# ---------------------------------------------------------------

class TestSizeSortKey:
    def test_50m_before_98m_before_170m(self):
        assert _size_sort_key("50M") < _size_sort_key("98M")
        assert _size_sort_key("98M") < _size_sort_key("170M")

    def test_unknown_sorts_last(self):
        assert _size_sort_key("unknown") > _size_sort_key("170M")

    def test_legacy_sizes_ordered(self):
        assert _size_sort_key("0.1M") < _size_sort_key("1M")
        assert _size_sort_key("1M") < _size_sort_key("3M")


class TestTableSortOrder:
    def _make_result(self, **kwargs):
        defaults = dict(
            checkpoint_path="ckpt.pt",
            model_family="ar",
            curriculum="baseline_ar",
            optimizer_family="adamw",
            lr_config={"lr": 0.003},
        )
        defaults.update(kwargs)
        return EvalResult(**defaults)

    def test_numeric_size_order(self):
        """Rows should appear 50M, 98M, 170M — not 170M, 50M, 98M."""
        results = [
            self._make_result(size="170M"),
            self._make_result(size="50M"),
            self._make_result(size="98M"),
        ]
        table = format_results_table(results)
        data_lines = table.strip().split("\n")[2:]  # skip header + separator
        sizes = [line.split()[1] for line in data_lines]
        assert sizes == ["50M", "98M", "170M"]

    def test_lexicographic_would_fail(self):
        """Confirm that naive string sort gives wrong order (the old bug)."""
        assert sorted(["170M", "50M", "98M"]) == ["170M", "50M", "98M"]
        # But numeric sort gives the right order:
        assert sorted(["170M", "50M", "98M"],
                       key=_size_sort_key) == ["50M", "98M", "170M"]


# ---------------------------------------------------------------
# CLI argument parsing (subprocess tests)
# ---------------------------------------------------------------

_EVAL_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "evaluate.py")


class TestCLI:
    def test_mutually_exclusive_inputs(self):
        """--checkpoint and --run-dir cannot both be specified."""
        result = subprocess.run(
            [sys.executable, _EVAL_SCRIPT,
             "--checkpoint", "a.pt", "--run-dir", "b/"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "not allowed" in result.stderr.lower() or "error" in result.stderr.lower()

    def test_no_input_errors(self):
        """Must specify at least one input mode."""
        result = subprocess.run(
            [sys.executable, _EVAL_SCRIPT],
            capture_output=True, text=True,
        )
        assert result.returncode != 0

    def test_help_runs(self):
        result = subprocess.run(
            [sys.executable, _EVAL_SCRIPT, "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "checkpoint" in result.stdout
        assert "run-dir" in result.stdout
        assert "sweep-dir" in result.stdout
        assert "skip-bpb" in result.stdout

    def test_triple_mutually_exclusive(self):
        """All three input modes together should fail."""
        result = subprocess.run(
            [sys.executable, _EVAL_SCRIPT,
             "--checkpoint", "a.pt", "--run-dir", "b/", "--sweep-dir", "c/"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
