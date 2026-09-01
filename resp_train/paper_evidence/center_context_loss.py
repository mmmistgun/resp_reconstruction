from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from resp_train.losses.task import RespirationTaskLoss
from resp_train.protocols.respiration import canonicalize_numpy, canonicalize_torch


PI60_SAMPLES = 6000


def pi60_torch(signal: torch.Tensor) -> torch.Tensor:
    if signal.shape[-1] != PI60_SAMPLES:
        raise ValueError(f"Pi_60 期望 {PI60_SAMPLES} 点，实际 {signal.shape[-1]}")
    _, canonical = canonicalize_torch(
        signal,
        fs=100.0,
        low_hz=0.05,
        high_hz=0.70,
        scale_eps=1e-8,
    )
    return canonical


def pi60_numpy(signal: np.ndarray) -> np.ndarray:
    values = np.asarray(signal)
    if values.shape[-1] != PI60_SAMPLES:
        raise ValueError(f"Pi_60 期望 {PI60_SAMPLES} 点，实际 {values.shape[-1]}")
    _, canonical = canonicalize_numpy(
        values,
        fs=100.0,
        low_hz=0.05,
        high_hz=0.70,
        scale_eps=1e-8,
    )
    return canonical


class CenterContextLoss(RespirationTaskLoss):
    """冻结中心目标上的 ``L_sync_60 + 0.25 L_effort_60``。"""

    def __init__(self, cfg: DictConfig) -> None:
        derived = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        derived.window.duration_samples = PI60_SAMPLES
        super().__init__(derived)
        if self.fs != 100.0 or self.band_low_hz != 0.05 or self.band_high_hz != 0.70:
            raise ValueError("中心 loss 的 fs/频带必须固定为 100 Hz / 0.05–0.70 Hz")
        if self.sync_weight != 1.0 or self.effort_weight != 0.25:
            raise ValueError("中心 loss 权重必须固定为 1.0 / 0.25")
        if self.max_lag_samples != 30 or self.envelope_window != 1000 or self.envelope_step != 500:
            raise ValueError("中心 loss lag/envelope 合同已漂移")

    def forward(self, prediction: torch.Tensor | Mapping[str, Any], target: torch.Tensor):
        return super().forward(prediction, target)


__all__ = ["CenterContextLoss", "PI60_SAMPLES", "pi60_numpy", "pi60_torch"]
