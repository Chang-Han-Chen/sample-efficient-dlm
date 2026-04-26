"""Config-path validation for the root-level JAX layout."""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import yaml

from train_ar import parse_args


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_CONFIGS = sorted((REPO_ROOT / "configs" / "experiments").glob("*.yaml"))
REMOVED_PREFIXES = ("jax/", "pytorch/", "baby-dLM/", "karpathy/")


def _load_yaml(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def test_experiment_configs_parse_and_do_not_reference_removed_code_roots(monkeypatch):
    assert EXPERIMENT_CONFIGS
    for config_path in EXPERIMENT_CONFIGS:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train_ar.py",
                "--config",
                str(config_path),
                "--synthetic",
                "--max-steps",
                "0",
                "--eval-batches",
                "0",
                "--no-wandb",
                "--disable-compilation-cache",
            ],
        )
        args = parse_args()
        assert args.config == config_path

        doc = _load_yaml(config_path)
        for group_path in doc.get("base_configs", {}).values():
            assert not group_path.startswith(REMOVED_PREFIXES), (config_path, group_path)
            assert (REPO_ROOT / group_path).exists(), (config_path, group_path)

        for key in ("train_path", "eval_path"):
            value = doc["train_args"].get(key)
            assert value is None or not value.startswith(REMOVED_PREFIXES), (config_path, key, value)


def test_data_base_configs_match_experiment_train_args():
    for config_path in EXPERIMENT_CONFIGS:
        doc = _load_yaml(config_path)
        data_config_path = REPO_ROOT / doc["base_configs"]["data"]
        data_doc = _load_yaml(data_config_path)
        for key in ("train_path", "eval_path", "vocab_size"):
            assert doc["train_args"][key] == data_doc[key], (config_path, key)
        if "mask_token_id" in doc["train_args"]:
            assert doc["train_args"]["mask_token_id"] == data_doc["mask_token_id"], config_path

