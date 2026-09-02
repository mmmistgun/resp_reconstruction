from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from resp_train.losses.task import RespirationTaskLoss
from resp_train.protocols.respiration import canonicalize_numpy, canonicalize_torch


PI30_SAMPLES = 3000


def pi30_torch(signal: torch.Tensor) -> torch.Tensor:
    if signal.shape[-1] != PI30_SAMPLES:
        raise ValueError(f"Pi_30 期望 {PI30_SAMPLES} 点，实际 {signal.shape[-1]}")
    _, canonical = canonicalize_torch(signal, fs=100.0, low_hz=0.05, high_hz=0.70, scale_eps=1e-8)
    return canonical


def pi30_numpy(signal: np.ndarray) -> np.ndarray:
    values = np.asarray(signal)
    if values.shape[-1] != PI30_SAMPLES:
        raise ValueError(f"Pi_30 期望 {PI30_SAMPLES} 点，实际 {values.shape[-1]}")
    _, canonical = canonicalize_numpy(values, fs=100.0, low_hz=0.05, high_hz=0.70, scale_eps=1e-8)
    return canonical


class Center30ContextLoss(RespirationTaskLoss):
    def __init__(self, cfg: DictConfig) -> None:
        derived = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        derived.window.duration_samples = PI30_SAMPLES
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
            raise ValueError("center30 loss 合同漂移")

    def forward(self, prediction: torch.Tensor | Mapping[str, Any], target: torch.Tensor):
        return super().forward(prediction, target)


__all__ = ["Center30ContextLoss", "PI30_SAMPLES", "pi30_numpy", "pi30_torch"]
