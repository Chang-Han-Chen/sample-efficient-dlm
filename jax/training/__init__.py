"""JAX training utilities for the sample-efficient transformer."""

from .loss import ar_loss, cross_entropy_with_z_loss
from .optimizer import (
    NormuonAdamWConfig,
    build_param_specs,
    create_normuon_adamw,
    global_norm,
    clip_by_global_norm,
    learning_rates,
)

__all__ = [
    "NormuonAdamWConfig",
    "ar_loss",
    "build_param_specs",
    "clip_by_global_norm",
    "create_normuon_adamw",
    "cross_entropy_with_z_loss",
    "global_norm",
    "learning_rates",
]
