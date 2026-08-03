from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn

from resp_train.protocols.respiration import (
    as_batch_waveform_torch,
    canonicalize_torch,
    centered_energy_torch,
    lag_priority,
)


class RespirationTaskLoss(nn.Module):
    """消融后冻结的同步与相对努力趋势损失。"""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        loss_cfg = cfg.loss
        self.fs = float(cfg.window.target_fs)
        self.length = int(cfg.window.duration_samples)
        self.band_low_hz = float(loss_cfg.band_low_hz)
        self.band_high_hz = float(loss_cfg.band_high_hz)
        self.scale_eps = float(loss_cfg.scale_eps)
        self.dynamic_eps = float(loss_cfg.dynamic_eps)
        self.corr_eps = float(loss_cfg.corr_eps)
        self.envelope_eps = float(loss_cfg.envelope_eps)
        self.max_lag_samples = int(round(float(loss_cfg.max_lag_sec) * self.fs))
        self.envelope_window = int(round(float(loss_cfg.envelope_window_sec) * self.fs))
        self.envelope_step = int(round(float(loss_cfg.envelope_step_sec) * self.fs))
        self.sync_weight = float(loss_cfg.sync_weight)
        self.effort_weight = float(loss_cfg.effort_weight)
        removed_fields = [name for name in ("rhythm_weight", "pol_start_weight") if name in loss_cfg]
        if removed_fields:
            raise ValueError(
                "当前最终 loss 已删除 rhythm 与 polarity，训练配置不能再包含: "
                + ", ".join(removed_fields)
            )
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        if self.max_lag_samples < 0 or self.length <= 2 * self.max_lag_samples:
            raise ValueError("max lag 与样本长度不兼容")
        if self.envelope_window <= 0 or self.envelope_step <= 0:
            raise ValueError("envelope window/step 必须为正")

    def forward(self, prediction: torch.Tensor | Mapping[str, Any], target: torch.Tensor):
        pred = as_batch_waveform_torch(prediction)
        ref = as_batch_waveform_torch(target)
        if pred.shape != ref.shape:
            raise ValueError(f"prediction/target shape 不一致: {tuple(pred.shape)} vs {tuple(ref.shape)}")
        if pred.shape[-1] != self.length:
            raise ValueError(f"期望 {self.length} 点，实际 {pred.shape[-1]}")
        if not torch.isfinite(pred).all() or not torch.isfinite(ref).all():
            raise FloatingPointError("prediction/target 包含 NaN/Inf")

        pred_band, pred_x = canonicalize_torch(
            pred,
            fs=self.fs,
            low_hz=self.band_low_hz,
            high_hz=self.band_high_hz,
            scale_eps=self.scale_eps,
        )
        target_band, target_x = canonicalize_torch(
            ref,
            fs=self.fs,
            low_hz=self.band_low_hz,
            high_hz=self.band_high_hz,
            scale_eps=self.scale_eps,
        )

        sync_values, sync_eligible, aligned_pred, aligned_target = self._sync_terms(
            pred_x,
            target_x,
            target_band,
        )
        loss_sync = _eligible_mean(sync_values, sync_eligible, pred_x)

        effort_values, effort_eligible = self._effort_terms(
            aligned_pred,
            aligned_target,
            sync_eligible,
        )
        loss_effort = _eligible_mean(effort_values, effort_eligible, pred_x)

        total = self.sync_weight * loss_sync + self.effort_weight * loss_effort
        parts: dict[str, torch.Tensor | float] = {
            "loss_sync": loss_sync,
            "loss_effort": loss_effort,
            **_aggregation_parts("loss_sync", sync_values, sync_eligible),
            **_aggregation_parts("loss_effort", effort_values, effort_eligible),
        }
        return total, parts

    def _sync_terms(
        self,
        pred_x: torch.Tensor,
        target_x: torch.Tensor,
        target_band: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        start = self.max_lag_samples
        stop = self.length - self.max_lag_samples
        target_common = target_x[:, start:stop]
        target_band_common = target_band[:, start:stop]
        eligible = centered_energy_torch(target_band_common) > self.dynamic_eps

        correlations: list[torch.Tensor] = []
        lags = lag_priority(self.max_lag_samples)
        for lag in lags:
            pred_common = pred_x[:, start + lag : stop + lag]
            correlations.append(_stable_corr(pred_common, target_common, eps=self.corr_eps))
        corr_grid = torch.stack(correlations, dim=1)
        best_priority_index = torch.argmax(corr_grid, dim=1)
        best_corr = corr_grid.gather(1, best_priority_index[:, None]).squeeze(1)
        selected_lags = torch.as_tensor(lags, device=pred_x.device, dtype=torch.long)[best_priority_index]

        base = torch.arange(start, stop, device=pred_x.device, dtype=torch.long)
        gather_index = base[None, :] + selected_lags[:, None]
        aligned_pred = pred_x.gather(1, gather_index)
        return 1.0 - best_corr, eligible, aligned_pred, target_common

    def _effort_terms(
        self,
        aligned_pred: torch.Tensor,
        aligned_target: torch.Tensor,
        sync_eligible: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred_frames = aligned_pred.unfold(-1, self.envelope_window, self.envelope_step)
        target_frames = aligned_target.unfold(-1, self.envelope_window, self.envelope_step)
        pred_log_rms = 0.5 * torch.log(pred_frames.square().mean(dim=-1) + self.envelope_eps)
        target_log_rms = 0.5 * torch.log(target_frames.square().mean(dim=-1) + self.envelope_eps)
        target_valid = torch.isfinite(target_log_rms).all(dim=1) & (
            centered_energy_torch(target_log_rms) > self.dynamic_eps
        )
        eligible = sync_eligible & target_valid
        return 1.0 - _stable_corr(pred_log_rms, target_log_rms, eps=self.corr_eps), eligible

def _stable_corr(left: torch.Tensor, right: torch.Tensor, *, eps: float) -> torch.Tensor:
    left_centered = left - left.mean(dim=-1, keepdim=True)
    right_centered = right - right.mean(dim=-1, keepdim=True)
    numerator = (left_centered * right_centered).sum(dim=-1)
    denominator = torch.sqrt(left_centered.square().sum(dim=-1) + float(eps)) * torch.sqrt(
        right_centered.square().sum(dim=-1) + float(eps)
    )
    return torch.clamp(numerator / denominator, -1.0, 1.0)


def _eligible_mean(values: torch.Tensor, eligible: torch.Tensor, graph_source: torch.Tensor) -> torch.Tensor:
    if bool(eligible.any()):
        return values[eligible].mean()
    return graph_source.sum() * 0.0


def _aggregation_parts(name: str, values: torch.Tensor, eligible: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        f"__sum_{name}": (values.detach() * eligible).sum(),
        f"__count_{name}": eligible.detach().sum(),
    }
