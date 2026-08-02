from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from resp_train.protocols.respiration import (
    as_batch_waveform_torch,
    canonicalize_torch,
    centered_energy_torch,
    lag_priority,
    symmetric_hann_torch,
)


class RespirationTaskLoss(nn.Module):
    """新 THO 协议的同步、节律、努力趋势与早期极性损失。"""

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
        self.power_eps = float(loss_cfg.power_eps)
        self.envelope_eps = float(loss_cfg.envelope_eps)
        self.max_lag_samples = int(round(float(loss_cfg.max_lag_sec) * self.fs))
        self.global_rhythm_window = int(round(float(loss_cfg.global_rhythm_window_sec) * self.fs))
        self.local_rhythm_window = int(round(float(loss_cfg.local_rhythm_window_sec) * self.fs))
        self.local_rhythm_hop = int(round(float(loss_cfg.local_rhythm_hop_sec) * self.fs))
        self.envelope_window = int(round(float(loss_cfg.envelope_window_sec) * self.fs))
        self.envelope_step = int(round(float(loss_cfg.envelope_step_sec) * self.fs))
        self.sync_weight = float(loss_cfg.sync_weight)
        self.rhythm_weight = float(loss_cfg.rhythm_weight)
        self.effort_weight = float(loss_cfg.effort_weight)
        self.pol_start_weight = float(loss_cfg.pol_start_weight)
        self.pol_fraction = float(loss_cfg.pol_fraction)
        self.smooth_l1_beta = float(loss_cfg.smooth_l1_beta)
        self._optimizer_step = 0
        self._total_optimizer_steps: int | None = None
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        if self.length != self.global_rhythm_window:
            raise ValueError("global rhythm window 必须等于完整样本长度")
        if self.max_lag_samples < 0 or self.length <= 2 * self.max_lag_samples:
            raise ValueError("max lag 与样本长度不兼容")
        if self.local_rhythm_window <= 0 or self.local_rhythm_hop <= 0:
            raise ValueError("local rhythm window/hop 必须为正")
        if self.envelope_window <= 0 or self.envelope_step <= 0:
            raise ValueError("envelope window/step 必须为正")
        if not (0.0 < self.pol_fraction <= 1.0):
            raise ValueError("pol_fraction 必须位于 (0,1]")

    def set_total_optimizer_steps(self, total_steps: int) -> None:
        if int(total_steps) <= 0:
            raise ValueError("total optimizer steps 必须为正")
        self._total_optimizer_steps = int(total_steps)

    @property
    def polarity_weight(self) -> float:
        if self._total_optimizer_steps is None:
            raise RuntimeError("训练前必须调用 set_total_optimizer_steps")
        progress = float(self._optimizer_step) / float(self._total_optimizer_steps)
        return self.pol_start_weight * max(0.0, 1.0 - progress / self.pol_fraction)

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

        global_values, global_eligible = self._spectral_scale(
            pred_x.unsqueeze(1),
            target_x.unsqueeze(1),
            target_band.unsqueeze(1),
        )
        local_values, local_eligible = self._spectral_scale(
            pred_x.unfold(-1, self.local_rhythm_window, self.local_rhythm_hop),
            target_x.unfold(-1, self.local_rhythm_window, self.local_rhythm_hop),
            target_band.unfold(-1, self.local_rhythm_window, self.local_rhythm_hop),
        )
        loss_rhythm_global = _eligible_mean(global_values, global_eligible, pred_x)
        loss_rhythm_local = _eligible_mean(local_values, local_eligible, pred_x)
        loss_rhythm = 0.5 * (loss_rhythm_global + loss_rhythm_local)

        effort_values, effort_eligible = self._effort_terms(
            aligned_pred,
            aligned_target,
            sync_eligible,
        )
        loss_effort = _eligible_mean(effort_values, effort_eligible, pred_x)

        pol_values = self._polarity_terms(aligned_pred, aligned_target)
        loss_pol = _eligible_mean(pol_values, sync_eligible, pred_x)
        pol_weight = self.polarity_weight if self.training else 0.0

        total = (
            self.sync_weight * loss_sync
            + self.rhythm_weight * loss_rhythm
            + self.effort_weight * loss_effort
            + pol_weight * loss_pol
        )
        parts: dict[str, torch.Tensor | float] = {
            "loss_sync": loss_sync,
            "loss_rhythm": loss_rhythm,
            "loss_effort": loss_effort,
            "loss_pol": loss_pol,
            **_aggregation_parts("loss_sync", sync_values, sync_eligible),
            **_aggregation_parts("loss_rhythm_global", global_values, global_eligible),
            **_aggregation_parts("loss_rhythm_local", local_values, local_eligible),
            **_aggregation_parts("loss_effort", effort_values, effort_eligible),
            **_aggregation_parts("loss_pol", pol_values, sync_eligible),
        }
        if self.training:
            self._optimizer_step += 1
        return total, parts

    def aggregate_epoch_summary(self, summary: dict[str, float]) -> dict[str, float]:
        result = dict(summary)
        global_value = float(result.pop("loss_rhythm_global", 0.0))
        local_value = float(result.pop("loss_rhythm_local", 0.0))
        result["loss_rhythm"] = 0.5 * (global_value + local_value)
        return result

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

    def _spectral_scale(
        self,
        pred_frames: torch.Tensor,
        target_frames: torch.Tensor,
        target_band_frames: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frame_length = int(pred_frames.shape[-1])
        window = symmetric_hann_torch(frame_length, device=pred_frames.device, dtype=pred_frames.dtype)

        def power(frames: torch.Tensor) -> torch.Tensor:
            centered = frames - frames.mean(dim=-1, keepdim=True)
            spectrum = torch.fft.rfft(centered * window, n=frame_length, dim=-1, norm="backward")
            frequencies = torch.fft.rfftfreq(frame_length, d=1.0 / self.fs, device=frames.device)
            mask = (frequencies >= self.band_low_hz) & (frequencies <= self.band_high_hz)
            return spectrum.abs().square()[..., mask]

        pred_power = power(pred_frames)
        target_power = power(target_frames)
        eligibility_power = power(target_band_frames)
        target_dynamic = centered_energy_torch(target_band_frames) > self.dynamic_eps
        frame_eligible = target_dynamic & (eligibility_power.sum(dim=-1) > self.power_eps)

        pred_distribution = (pred_power + self.power_eps) / (
            pred_power + self.power_eps
        ).sum(dim=-1, keepdim=True)
        target_distribution = (target_power + self.power_eps) / (
            target_power + self.power_eps
        ).sum(dim=-1, keepdim=True)
        frame_distance = 0.5 * torch.abs(pred_distribution - target_distribution).sum(dim=-1)
        eligible_count = frame_eligible.sum(dim=1)
        sample_eligible = eligible_count > 0
        sample_value = torch.where(
            sample_eligible,
            (frame_distance * frame_eligible).sum(dim=1) / eligible_count.clamp_min(1),
            torch.zeros_like(frame_distance[:, 0]),
        )
        return sample_value, sample_eligible

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

    def _polarity_terms(self, aligned_pred: torch.Tensor, aligned_target: torch.Tensor) -> torch.Tensor:
        pred_z = _window_standardize(aligned_pred, eps=self.scale_eps)
        target_z = _window_standardize(aligned_target, eps=self.scale_eps)
        pointwise = F.smooth_l1_loss(pred_z, target_z, reduction="none", beta=self.smooth_l1_beta)
        return pointwise.mean(dim=-1)


def _stable_corr(left: torch.Tensor, right: torch.Tensor, *, eps: float) -> torch.Tensor:
    left_centered = left - left.mean(dim=-1, keepdim=True)
    right_centered = right - right.mean(dim=-1, keepdim=True)
    numerator = (left_centered * right_centered).sum(dim=-1)
    denominator = torch.sqrt(left_centered.square().sum(dim=-1) + float(eps)) * torch.sqrt(
        right_centered.square().sum(dim=-1) + float(eps)
    )
    return torch.clamp(numerator / denominator, -1.0, 1.0)


def _window_standardize(signal: torch.Tensor, *, eps: float) -> torch.Tensor:
    centered = signal - signal.mean(dim=-1, keepdim=True)
    return centered / torch.sqrt(centered.square().mean(dim=-1, keepdim=True) + float(eps))


def _eligible_mean(values: torch.Tensor, eligible: torch.Tensor, graph_source: torch.Tensor) -> torch.Tensor:
    if bool(eligible.any()):
        return values[eligible].mean()
    return graph_source.sum() * 0.0


def _aggregation_parts(name: str, values: torch.Tensor, eligible: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        f"__sum_{name}": (values.detach() * eligible).sum(),
        f"__count_{name}": eligible.detach().sum(),
    }
