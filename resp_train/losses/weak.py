from __future__ import annotations

import torch
import torch.nn.functional as F
from omegaconf import DictConfig


class WeakSyncLoss(torch.nn.Module):
    """用于波形弱同步训练的组合损失。"""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.fs = float(cfg.window.target_fs)
        self.envelope_window = max(1, int(float(cfg.loss.envelope_window_sec) * self.fs))
        self.env_weight = float(cfg.loss.envelope_weight)
        self.spec_weight = float(cfg.loss.spectrum_weight)
        self.smooth_weight = float(cfg.loss.smooth_weight)
        self.low_hz = float(cfg.loss.spectrum_low_hz)
        self.high_hz = float(cfg.loss.spectrum_high_hz)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._validate_inputs(pred, target)
        env = self._envelope_loss(pred, target)
        spec = self._spectrum_loss(pred, target)
        smooth = torch.mean(torch.abs(pred[..., 1:] - pred[..., :-1]))
        total = self.env_weight * env + self.spec_weight * spec + self.smooth_weight * smooth
        return total, {"envelope": env.detach(), "spectrum": spec.detach(), "smooth": smooth.detach()}

    @staticmethod
    def _validate_inputs(pred: torch.Tensor, target: torch.Tensor) -> None:
        if pred.shape != target.shape:
            raise ValueError(f"pred 和 target shape 必须一致: pred={tuple(pred.shape)} target={tuple(target.shape)}")
        if pred.ndim != 3 or pred.shape[1] != 1:
            raise ValueError(f"pred 和 target 必须为 [B, 1, T]，当前 shape={tuple(pred.shape)}")

    def _envelope_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_env = self._zscore(self._rms_envelope(pred))
        target_env = self._zscore(self._rms_envelope(target))
        corr = torch.mean(pred_env * target_env, dim=-1)
        return torch.mean(1.0 - corr)

    def _rms_envelope(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.envelope_window // 2
        smoothed = F.avg_pool1d(x.square(), kernel_size=self.envelope_window, stride=1, padding=pad)
        if smoothed.shape[-1] > x.shape[-1]:
            smoothed = smoothed[..., : x.shape[-1]]
        return torch.sqrt(torch.clamp(smoothed, min=1e-8))

    def _spectrum_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_dist = self._band_distribution(pred)
        target_dist = self._band_distribution(target)
        return torch.mean(torch.sum(torch.abs(pred_dist - target_dist), dim=-1))

    def _band_distribution(self, x: torch.Tensor) -> torch.Tensor:
        centered = x - x.mean(dim=-1, keepdim=True)
        power = torch.fft.rfft(centered, dim=-1).abs().square().squeeze(1)
        freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / self.fs).to(x.device)
        mask = (freqs >= self.low_hz) & (freqs <= self.high_hz)
        if not bool(mask.any()):
            raise ValueError(
                f"频谱损失频带为空: low_hz={self.low_hz} high_hz={self.high_hz} "
                f"fs={self.fs} n={x.shape[-1]}"
            )
        band = power[:, mask]
        return band / torch.clamp(band.sum(dim=-1, keepdim=True), min=1e-8)

    @staticmethod
    def _zscore(x: torch.Tensor) -> torch.Tensor:
        std = x.std(dim=-1, keepdim=True)
        return (x - x.mean(dim=-1, keepdim=True)) / torch.clamp(std, min=1e-6)
