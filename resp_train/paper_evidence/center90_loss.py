from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from resp_train.losses.task import RespirationTaskLoss
from resp_train.protocols.respiration import canonicalize_numpy, canonicalize_torch


PI90_SAMPLES = 9000


def pi90_torch(signal: torch.Tensor) -> torch.Tensor:
    if signal.shape[-1] != PI90_SAMPLES:
        raise ValueError(f"Pi_90 期望 {PI90_SAMPLES} 点，实际 {signal.shape[-1]}")
    _, canonical = canonicalize_torch(
        signal, fs=100.0, low_hz=0.05, high_hz=0.70, scale_eps=1e-8
    )
    return canonical


def pi90_numpy(signal: np.ndarray) -> np.ndarray:
    values = np.asarray(signal)
    if values.shape[-1] != PI90_SAMPLES:
        raise ValueError(f"Pi_90 期望 {PI90_SAMPLES} 点，实际 {values.shape[-1]}")
    _, canonical = canonicalize_numpy(
        values, fs=100.0, low_hz=0.05, high_hz=0.70, scale_eps=1e-8
    )
    return canonical


class Center90ContextLoss(RespirationTaskLoss):
    def __init__(self, cfg: DictConfig) -> None:
        derived = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        derived.window.duration_samples = PI90_SAMPLES
        super().__init__(derived)
        if (
            self.fs != 100.0
            or self.band_low_hz != 0.05
            or self.band_high_hz != 0.70
            or self.max_lag_samples != 30
            or self.envelope_window != 1000
            or self.envelope_step != 500
            or self.sync_weight != 1.0
            or self.effort_weight != 0.25
        ):
            raise ValueError("center90 loss 合同漂移")

    def forward(self, prediction: torch.Tensor | Mapping[str, Any], target: torch.Tensor):
        return super().forward(prediction, target)


__all__ = ["Center90ContextLoss", "PI90_SAMPLES", "pi90_numpy", "pi90_torch"]
