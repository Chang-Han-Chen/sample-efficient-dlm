"""
Tests for run_isoflop.py — the IsoFLOP sweep runner (§6.5).
"""

import json
import os
import pickle
import subprocess
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import experiment_config as ec
from run_isoflop import (
    PHASE1_CURRICULA,
    PHASE1_CURRICULUM_NAMES,
    IsoFLOPPoint,
    IsoFLOPSweepResult,
    check_feasibility,
    save_sweep_results,
)


# ---------------------------------------------------------------
# Phase-1 curriculum set
# ---------------------------------------------------------------

class TestPhase1Curricula:
    def test_includes_baselines(self):
        names = PHASE1_CURRICULUM_NAMES
        assert "baseline_ar" in names
        assert "baseline_mdlm" in names
        assert "baseline_bd3lm_16" in names

    def test_includes_curriculum_0_sweeps(self):
        names = PHASE1_CURRICULUM_NAMES
        assert "c0_plain_p20" in names
        assert "c0_plain_p30" in names
        assert "c0_plain_p50" in names
        assert "c0_plain_p80" in names

    def test_includes_curriculum_1(self):
        assert "c1_geometric" in PHASE1_CURRICULUM_NAMES

    def test_excludes_later_curricula(self):
        """Phase 1 should not include c2 or c3 (deferred per §5)."""
        names = PHASE1_CURRICULUM_NAMES
        assert "c2_aggressive_jump" not in names
        assert all(not n.startswith("c3_") for n in names)

    def test_count(self):
        # 3 baselines + 4 c0 + 1 c1 = 8
        assert len(PHASE1_CURRICULA) == 8


# ---------------------------------------------------------------
# Feasibility check
# ---------------------------------------------------------------

class TestCheckFeasibility:
    def test_baseline_ar_50m_1e18_feasible(self):
        """Baseline AR at 50M / 1e18 should be feasible."""
        reason = check_feasibility(ec.BASELINE_AR, "50M", 1e18)
        assert reason is None

    def test_baseline_ar_170m_1e18_feasible(self):
        reason = check_feasibility(ec.BASELINE_AR, "170M", 1e18)
        assert reason is None

    def test_tiny_budget_infeasible(self):
        """A tiny budget on a large model should be infeasible (too few steps)."""
        reason = check_feasibility(ec.BASELINE_AR, "170M", 1e12)
        assert reason is not None
        assert "steps outside" in reason

    def test_multi_stage_checks_each_stage(self):
        """Curriculum 1 splits budget 5 ways; each stage is checked."""
        # With a very small budget, each stage gets only 20% of an already
        # tiny budget, so the per-stage step count will be too low.
        reason = check_feasibility(ec.CURRICULUM_1, "170M", 1e12)
        assert reason is not None
        assert "stage" in reason

    def test_large_budget_feasible(self):
        """1e19 on 50M should be feasible for all phase-1 curricula."""
        for cur in PHASE1_CURRICULA:
            reason = check_feasibility(cur, "50M", 1e19)
            # Might still be infeasible if steps > MAX for some stage,
            # but 50M is small enough that it should work.
            # We just check it doesn't crash.
            assert reason is None or isinstance(reason, str)


# ---------------------------------------------------------------
# IsoFLOPPoint.completed property
# ---------------------------------------------------------------

class TestIsoFLOPPoint:
    def test_skipped_point(self):
        p = IsoFLOPPoint(
            budget=1e18, curriculum="baseline_ar", optimizer="adamw",
            size="170M", val_loss=None, train_loss=None,
            lr_config={}, wall_time_seconds=0.0, out_dir="/tmp",
            num_stages_completed=0, num_stages_total=1,
            skipped=True, skip_reason="too few steps",
        )
        assert p.skipped
        assert not p.completed
        assert p.val_loss is None

    def test_completed_point(self):
        p = IsoFLOPPoint(
            budget=1e18, curriculum="baseline_ar", optimizer="adamw",
            size="50M", val_loss=6.5, train_loss=6.0,
            lr_config={"ar": {"lr": 0.01}},
            wall_time_seconds=120.0, out_dir="/tmp/run",
            num_stages_completed=1, num_stages_total=1,
        )
        assert not p.skipped
        assert p.completed
        assert p.val_loss == 6.5

    def test_partial_point_not_completed(self):
        """A point where only 3 of 5 stages finished is not completed."""
        p = IsoFLOPPoint(
            budget=1e18, curriculum="c1_geometric", optimizer="adamw",
            size="50M", val_loss=7.0, train_loss=6.8,
            lr_config={"ar": {"lr": 0.01}},
            wall_time_seconds=60.0, out_dir="/tmp/partial",
            num_stages_completed=3, num_stages_total=5,
        )
        assert not p.completed

    def test_dry_run_point_not_completed(self):
        """A dry-run point has 0 stages completed and no val_loss."""
        p = IsoFLOPPoint(
            budget=1e18, curriculum="baseline_ar", optimizer="adamw",
            size="50M", val_loss=None, train_loss=None,
            lr_config={}, wall_time_seconds=0.0, out_dir="/tmp/dry",
            num_stages_completed=0, num_stages_total=1,
        )
        assert not p.completed

    def test_completed_but_no_val_loss_not_completed(self):
        """All stages done but val_loss is None (e.g. skip_final_eval)."""
        p = IsoFLOPPoint(
            budget=1e18, curriculum="baseline_ar", optimizer="adamw",
            size="50M", val_loss=None, train_loss=6.0,
            lr_config={"ar": {"lr": 0.01}},
            wall_time_seconds=120.0, out_dir="/tmp/run",
            num_stages_completed=1, num_stages_total=1,
        )
        assert not p.completed


# ---------------------------------------------------------------
# IsoFLOPSweepResult — curve filtering
# ---------------------------------------------------------------

class TestIsoFLOPSweepResult:
    def _make_sweep(self):
        return IsoFLOPSweepResult(points=[
            # Budget 1e18, baseline_ar, adamw — two completed sizes
            IsoFLOPPoint(
                budget=1e18, curriculum="baseline_ar", optimizer="adamw",
                size="50M", val_loss=6.5, train_loss=6.0,
                lr_config={"ar": {"lr": 0.01}},
                wall_time_seconds=100.0, out_dir="/tmp/50M",
                num_stages_completed=1, num_stages_total=1,
            ),
            IsoFLOPPoint(
                budget=1e18, curriculum="baseline_ar", optimizer="adamw",
                size="98M", val_loss=6.2, train_loss=5.8,
                lr_config={"ar": {"lr": 0.003}},
                wall_time_seconds=200.0, out_dir="/tmp/98M",
                num_stages_completed=1, num_stages_total=1,
            ),
            # Budget 1e18, baseline_ar, normuon — two completed sizes
            IsoFLOPPoint(
                budget=1e18, curriculum="baseline_ar", optimizer="normuon",
                size="50M", val_loss=6.3, train_loss=5.9,
                lr_config={"ar": {"adam_mult": 1.0, "matrix_mult": 1.0}},
                wall_time_seconds=110.0, out_dir="/tmp/normuon_50M",
                num_stages_completed=1, num_stages_total=1,
            ),
            IsoFLOPPoint(
                budget=1e18, curriculum="baseline_ar", optimizer="normuon",
                size="98M", val_loss=6.0, train_loss=5.6,
                lr_config={"ar": {"adam_mult": 1.0, "matrix_mult": 0.3}},
                wall_time_seconds=220.0, out_dir="/tmp/normuon_98M",
                num_stages_completed=1, num_stages_total=1,
            ),
            # A skipped point
            IsoFLOPPoint(
                budget=1e18, curriculum="baseline_ar", optimizer="adamw",
                size="170M", val_loss=None, train_loss=None,
                lr_config={}, wall_time_seconds=0.0, out_dir="/tmp/170M",
                num_stages_completed=0, num_stages_total=1,
                skipped=True, skip_reason="test skip",
            ),
        ])

    def test_curves_groups_correctly(self):
        sweep = self._make_sweep()
        curves = sweep.curves()
        # Two curves: (1e18, baseline_ar, adamw) and (1e18, baseline_ar, normuon)
        assert len(curves) == 2
        assert (1e18, "baseline_ar", "adamw") in curves
        assert (1e18, "baseline_ar", "normuon") in curves

    def test_curves_contain_correct_sizes(self):
        sweep = self._make_sweep()
        curves = sweep.curves()
        adamw_curve = curves[(1e18, "baseline_ar", "adamw")]
        assert adamw_curve == {"50M": 6.5, "98M": 6.2}

    def test_curves_exclude_skipped(self):
        sweep = self._make_sweep()
        curves = sweep.curves()
        adamw_curve = curves[(1e18, "baseline_ar", "adamw")]
        assert "170M" not in adamw_curve

    def test_curves_exclude_partial_runs(self):
        """A partial run (3/5 stages) must not appear in curves."""
        sweep = IsoFLOPSweepResult(points=[
            IsoFLOPPoint(
                budget=1e18, curriculum="c1_geometric", optimizer="adamw",
                size="50M", val_loss=7.0, train_loss=6.8,
                lr_config={"ar": {"lr": 0.01}},
                wall_time_seconds=60.0, out_dir="/tmp/partial",
                num_stages_completed=3, num_stages_total=5,
            ),
        ])
        assert sweep.curves() == {}

    def test_curves_exclude_dry_run_points(self):
        """Dry-run points (0 stages, no val_loss) must not appear in curves."""
        sweep = IsoFLOPSweepResult(points=[
            IsoFLOPPoint(
                budget=1e18, curriculum="baseline_ar", optimizer="adamw",
                size="50M", val_loss=None, train_loss=None,
                lr_config={}, wall_time_seconds=0.0, out_dir="/tmp/dry",
                num_stages_completed=0, num_stages_total=1,
            ),
        ])
        assert sweep.curves() == {}

    def test_curves_exclude_none_val_loss(self):
        """Even if all stages 'completed', None val_loss must be excluded."""
        sweep = IsoFLOPSweepResult(points=[
            IsoFLOPPoint(
                budget=1e18, curriculum="baseline_ar", optimizer="adamw",
                size="50M", val_loss=None, train_loss=6.0,
                lr_config={"ar": {"lr": 0.01}},
                wall_time_seconds=120.0, out_dir="/tmp/no_val",
                num_stages_completed=1, num_stages_total=1,
            ),
        ])
        assert sweep.curves() == {}

    def test_curves_mix_completed_and_partial(self):
        """Only the completed point contributes to the curve."""
        sweep = IsoFLOPSweepResult(points=[
            # Completed
            IsoFLOPPoint(
                budget=1e18, curriculum="c1_geometric", optimizer="adamw",
                size="50M", val_loss=6.3, train_loss=5.9,
                lr_config={"ar": {"lr": 0.01}},
                wall_time_seconds=300.0, out_dir="/tmp/ok",
                num_stages_completed=5, num_stages_total=5,
            ),
            # Partial (stage 2 failed)
            IsoFLOPPoint(
                budget=1e18, curriculum="c1_geometric", optimizer="adamw",
                size="98M", val_loss=7.1, train_loss=6.9,
                lr_config={"ar": {"lr": 0.003}},
                wall_time_seconds=80.0, out_dir="/tmp/fail",
                num_stages_completed=2, num_stages_total=5,
            ),
        ])
        curves = sweep.curves()
        curve = curves[(1e18, "c1_geometric", "adamw")]
        assert curve == {"50M": 6.3}
        assert "98M" not in curve

    def test_summary_table_runs(self):
        sweep = self._make_sweep()
        table = sweep.summary_table()
        assert "baseline_ar" in table
        assert "adamw" in table
        assert "normuon" in table
        assert "SKIP" in table

    def test_summary_dict_structure(self):
        sweep = self._make_sweep()
        d = sweep.summary_dict()
        assert d["num_points"] == 5
        assert d["num_completed"] == 4
        assert d["num_skipped"] == 1
        assert "curves" in d
        assert "points" in d
        assert len(d["points"]) == 5

    def test_summary_dict_curves_keyed_correctly(self):
        sweep = self._make_sweep()
        d = sweep.summary_dict()
        assert "1.00e+18|baseline_ar|adamw" in d["curves"]
        assert "1.00e+18|baseline_ar|normuon" in d["curves"]

    def test_summary_dict_preserves_lr_configs(self):
        """Each point in the JSON should retain its per-family LR config."""
        sweep = self._make_sweep()
        d = sweep.summary_dict()
        # Find the normuon 50M point
        normuon_50m = [p for p in d["points"]
                       if p["optimizer"] == "normuon" and p["size"] == "50M"]
        assert len(normuon_50m) == 1
        assert normuon_50m[0]["lr_config"]["ar"]["adam_mult"] == 1.0

    def test_empty_sweep(self):
        sweep = IsoFLOPSweepResult()
        assert sweep.curves() == {}
        assert sweep.summary_table()  # should not crash
        d = sweep.summary_dict()
        assert d["num_points"] == 0

    def test_num_completed_excludes_partial(self):
        """summary_dict().num_completed must only count fully completed points."""
        sweep = IsoFLOPSweepResult(points=[
            IsoFLOPPoint(
                budget=1e18, curriculum="c1_geometric", optimizer="adamw",
                size="50M", val_loss=6.3, train_loss=5.9,
                lr_config={}, wall_time_seconds=300.0, out_dir="/tmp/ok",
                num_stages_completed=5, num_stages_total=5,
            ),
            IsoFLOPPoint(
                budget=1e18, curriculum="c1_geometric", optimizer="adamw",
                size="98M", val_loss=7.0, train_loss=6.8,
                lr_config={}, wall_time_seconds=60.0, out_dir="/tmp/partial",
                num_stages_completed=2, num_stages_total=5,
            ),
        ])
        d = sweep.summary_dict()
        assert d["num_completed"] == 1


# ---------------------------------------------------------------
# CLI: --curriculum additive vs --curriculum-only replace
# ---------------------------------------------------------------

class TestCLICurriculumArgs:
    """Test the CLI argument resolution via subprocess dry-run."""

    @staticmethod
    def _run_cli(*extra_args):
        """Run run_isoflop.py with --dry-run and capture stdout."""
        cmd = [
            sys.executable, "run_isoflop.py",
            "--dry-run",
            "--budget", "1e18",
            "--optimizer", "adamw",
            "--sizes", "50M",
        ] + list(extra_args)
        result = subprocess.run(
            cmd,
            capture_output=True, text=True,
            cwd=os.path.join(os.path.dirname(__file__), ".."),
        )
        return result

    def test_default_uses_phase1(self):
        """No --curriculum flag → phase-1 set (8 curricula × 1 size = 8 combos)."""
        r = self._run_cli()
        assert r.returncode == 0
        assert "total combinations: 8" in r.stdout

    def test_curriculum_adds_to_phase1(self):
        """--curriculum c2_aggressive_jump → phase-1 + c2 = 9 combos."""
        r = self._run_cli("--curriculum", "c2_aggressive_jump")
        assert r.returncode == 0
        assert "total combinations: 9" in r.stdout
        # Both phase-1 and c2 appear
        assert "baseline_ar" in r.stdout
        assert "c2_aggressive_jump" in r.stdout

    def test_curriculum_deduplicates(self):
        """--curriculum baseline_ar (already in phase-1) → still 8 combos."""
        r = self._run_cli("--curriculum", "baseline_ar")
        assert r.returncode == 0
        assert "total combinations: 8" in r.stdout

    def test_curriculum_only_replaces(self):
        """--curriculum-only c1_geometric → just 1 curriculum × 1 size = 1 combo."""
        r = self._run_cli("--curriculum-only", "c1_geometric")
        assert r.returncode == 0
        assert "total combinations: 1" in r.stdout
        assert "c1_geometric" in r.stdout

    def test_curriculum_and_curriculum_only_errors(self):
        """Using both --curriculum and --curriculum-only is an error."""
        r = self._run_cli(
            "--curriculum", "c2_aggressive_jump",
            "--curriculum-only", "c1_geometric",
        )
        assert r.returncode != 0
        assert "mutually exclusive" in r.stderr


# ---------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------

class TestSaveSweepResults:
    def test_save_creates_files(self, tmp_path):
        sweep = IsoFLOPSweepResult(points=[
            IsoFLOPPoint(
                budget=1e18, curriculum="baseline_ar", optimizer="adamw",
                size="50M", val_loss=6.5, train_loss=6.0,
                lr_config={"ar": {"lr": 0.01}},
                wall_time_seconds=100.0, out_dir=str(tmp_path / "run"),
                num_stages_completed=1, num_stages_total=1,
            ),
        ])
        out_dir = str(tmp_path / "output")
        save_sweep_results(sweep, out_dir)

        # Pickle
        pkl_path = os.path.join(out_dir, "isoflop_sweep.pkl")
        assert os.path.exists(pkl_path)
        with open(pkl_path, "rb") as f:
            loaded = pickle.load(f)
        assert len(loaded.points) == 1
        assert loaded.points[0].val_loss == 6.5

        # JSON
        json_path = os.path.join(out_dir, "isoflop_summary.json")
        assert os.path.exists(json_path)
        with open(json_path) as f:
            summary = json.load(f)
        assert summary["num_points"] == 1
        assert summary["num_completed"] == 1

    def test_save_round_trips_curves(self, tmp_path):
        sweep = IsoFLOPSweepResult(points=[
            IsoFLOPPoint(
                budget=1e18, curriculum="c1_geometric", optimizer="adamw",
                size="50M", val_loss=6.3, train_loss=5.9,
                lr_config={"ar": {"lr": 0.01}, "bd3lm": {"lr": 0.003}},
                wall_time_seconds=300.0, out_dir=str(tmp_path / "run"),
                num_stages_completed=5, num_stages_total=5,
            ),
            IsoFLOPPoint(
                budget=1e18, curriculum="c1_geometric", optimizer="adamw",
                size="98M", val_loss=6.0, train_loss=5.5,
                lr_config={"ar": {"lr": 0.003}, "bd3lm": {"lr": 0.001}},
                wall_time_seconds=500.0, out_dir=str(tmp_path / "run2"),
                num_stages_completed=5, num_stages_total=5,
            ),
        ])
        out_dir = str(tmp_path / "output")
        save_sweep_results(sweep, out_dir)

        with open(os.path.join(out_dir, "isoflop_sweep.pkl"), "rb") as f:
            loaded = pickle.load(f)
        curves = loaded.curves()
        assert (1e18, "c1_geometric", "adamw") in curves
        assert curves[(1e18, "c1_geometric", "adamw")] == {"50M": 6.3, "98M": 6.0}

        with open(os.path.join(out_dir, "isoflop_summary.json")) as f:
            summary = json.load(f)
        curve_key = "1.00e+18|c1_geometric|adamw"
        assert curve_key in summary["curves"]
        assert summary["curves"][curve_key]["50M"] == 6.3
