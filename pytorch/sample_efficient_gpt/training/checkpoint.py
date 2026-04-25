from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Optimizer

from sample_efficient_gpt.config_schema import Config
from sample_efficient_gpt.utils.config_tools import save_config


def _to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, (list, tuple)):
        t = [_to_device(xx, device) for xx in x]
        return type(x)(t)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    return x  # ints/floats/etc.


def save_checkpoint(
    fpath: Path,
    cfg: Config | None,
    model: nn.Module,
    optimizers: list[Optimizer],
    iteration: int = 0,
    run_id: str | None = None,
):
    model_state_dict = model.state_dict()
    optimizers_state_dict = [optimizer.state_dict() for optimizer in optimizers]

    torch.save(
        {
            "config": save_config(cfg) if cfg is not None else None,
            "model": model_state_dict,
            "optimizer": optimizers_state_dict,
            "iteration": iteration,
            "run_id": run_id,
        },
        fpath,
    )


def load_model(fpath: Path, cfg: Config, model: nn.Module) -> int:
    checkpoint = torch.load(fpath, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    return checkpoint["iteration"], checkpoint.get("run_id")


def load_optimizer(fpath: Path, cfg: Config, model, optimizers: list[Optimizer] | None) -> None:
    checkpoint = torch.load(fpath, map_location="cpu", weights_only=False)
    if optimizers is not None and optimizers[0] is not None:
        if checkpoint.get("optimizer") is None or (
            isinstance(checkpoint["optimizer"], list) and checkpoint["optimizer"][0] is None
        ):
            print("no optimizers detected in the checkpoint, skipping...")
            return
        # for muon we load 2 optimizers
        for opt_idx, (opt, opt_sd) in enumerate(zip(optimizers, checkpoint["optimizer"])):
            # Check if param_groups have compatible sizes
            current_groups = opt.state_dict()["param_groups"]
            saved_groups = opt_sd["param_groups"]
            if len(current_groups) != len(saved_groups):
                print(
                    f"optimizer {opt_idx}: param_groups count mismatch ({len(current_groups)} vs {len(saved_groups)}), skipping optimizer load..."
                )
                continue
            groups_compatible = True
            for g_idx, (cg, sg) in enumerate(zip(current_groups, saved_groups)):
                if len(cg["params"]) != len(sg["params"]):
                    print(
                        f"optimizer {opt_idx} group {g_idx}: params count mismatch ({len(cg['params'])} vs {len(sg['params'])}), skipping optimizer load..."
                    )
                    groups_compatible = False
                    break
            if not groups_compatible:
                continue
            # Validate state shapes
            current_state = opt.state_dict()["state"]
            saved_state = opt_sd["state"]
            for (k1, v1), (k2, v2) in zip(current_state.items(), saved_state.items()):
                assert k1 == k2 and v1.shape == v2.shape, f"{k1} != {k2} or {v1.shape=}!={v2.shape=}"
            opt.load_state_dict(opt_sd)
            for p, state in opt.state.items():  # keys are parameter tensors
                p_dev = p.device
                for k, v in list(state.items()):
                    state[k] = _to_device(v, p_dev)
