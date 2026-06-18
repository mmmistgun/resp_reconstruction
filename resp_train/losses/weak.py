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
        self.high_freq_weight = float(cfg.loss.get("high_freq_weight", 0.0))
        self.relative_env_weight = float(cfg.loss.get("relative_envelope_weight", 0.0))
        self.phase_alignment_weight = float(cfg.loss.get("phase_alignment_weight", 0.0))
        self.band_waveform_weight = float(cfg.loss.get("band_waveform_weight", 0.0))
        self.curvature_weight = float(cfg.loss.get("curvature_weight", 0.0))
        self.phase_alignment_zero_weight = float(cfg.loss.get("phase_alignment_zero_weight", 0.5))
        self.phase_alignment_lag_weight = float(cfg.loss.get("phase_alignment_lag_weight", 0.5))
        self.phase_alignment_lag_penalty_weight = float(cfg.loss.get("phase_alignment_lag_penalty_weight", 0.1))
        self.phase_alignment_max_sec = float(cfg.loss.get("phase_alignment_max_sec", 0.5))
        self.phase_alignment_step_sec = float(cfg.loss.get("phase_alignment_step_sec", 0.1))
        self.phase_alignment_temperature = float(cfg.loss.get("phase_alignment_temperature", 0.05))
        for name, value in (
            ("phase_alignment_weight", self.phase_alignment_weight),
            ("phase_alignment_zero_weight", self.phase_alignment_zero_weight),
            ("phase_alignment_lag_weight", self.phase_alignment_lag_weight),
            ("phase_alignment_lag_penalty_weight", self.phase_alignment_lag_penalty_weight),
            ("band_waveform_weight", self.band_waveform_weight),
            ("curvature_weight", self.curvature_weight),
        ):
            if value < 0:
                raise ValueError(f"{name} 必须非负，当前={value}")
        if self.phase_alignment_max_sec < 0:
            raise ValueError(f"phase_alignment_max_sec 必须非负，当前={self.phase_alignment_max_sec}")
        if self.phase_alignment_step_sec <= 0:
            raise ValueError(f"phase_alignment_step_sec 必须为正数，当前={self.phase_alignment_step_sec}")
        if self.phase_alignment_temperature <= 0:
            raise ValueError(f"phase_alignment_temperature 必须为正数，当前={self.phase_alignment_temperature}")
        self.relative_env_trend_window = max(self.envelope_window, int(round(self.fs * 20.0)))
        self.low_hz = float(cfg.loss.spectrum_low_hz)
        self.high_hz = float(cfg.loss.spectrum_high_hz)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._validate_inputs(pred, target)
        # AMP 下模型输出可能是 float16；cuFFT 对非 2 次幂长度的 half FFT
        # 有限制，因此损失内部统一用 float32 做频域和包络计算。
        pred_loss = pred.float()
        target_loss = target.float()
        env = self._envelope_loss(pred_loss, target_loss)
        spec = self._spectrum_loss(pred_loss, target_loss)
        smooth = torch.mean(torch.abs(pred_loss[..., 1:] - pred_loss[..., :-1]))
        high_freq = self._high_frequency_energy(pred_loss)
        relative_env = self._relative_envelope_loss(pred_loss, target_loss)
        band_waveform = self._band_waveform_loss(pred_loss, target_loss)
        curvature = self._curvature_loss(pred_loss)
        if self.phase_alignment_weight > 0:
            phase_alignment = self._phase_alignment_loss(pred_loss, target_loss)
        else:
            phase_alignment = pred_loss.new_tensor(0.0)
        total = (
            self.env_weight * env
            + self.spec_weight * spec
            + self.smooth_weight * smooth
            + self.high_freq_weight * high_freq
            + self.relative_env_weight * relative_env
            + self.phase_alignment_weight * phase_alignment
            + self.band_waveform_weight * band_waveform
            + self.curvature_weight * curvature
        )
        return total, {
            "envelope": env.detach(),
            "spectrum": spec.detach(),
            "smooth": smooth.detach(),
            "high_freq": high_freq.detach(),
            "relative_envelope": relative_env.detach(),
            "phase_alignment": phase_alignment.detach(),
            "band_waveform": band_waveform.detach(),
            "curvature": curvature.detach(),
        }

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
        centered = self._center(x)
        power = torch.fft.rfft(centered, dim=-1).abs().square().squeeze(1)
        mask = self._band_mask(x.shape[-1], x.device)
        band = power[:, mask]
        return band / torch.clamp(band.sum(dim=-1, keepdim=True), min=1e-8)

    def _band_limited_waveform(self, x: torch.Tensor) -> torch.Tensor:
        centered = self._center(x)
        spectrum = torch.fft.rfft(centered, dim=-1)
        mask = self._band_mask(x.shape[-1], x.device).view(1, 1, -1)
        filtered = torch.where(mask, spectrum, torch.zeros_like(spectrum))
        return torch.fft.irfft(filtered, n=x.shape[-1], dim=-1)

    def _lag_samples(self, n_samples: int, max_sec: float, step_sec: float) -> list[int]:
        max_lag = int(round(max_sec * self.fs))
        step = max(1, int(round(step_sec * self.fs)))
        max_lag = min(max_lag, max(0, n_samples - 2))
        non_negative_lags = list(range(0, max_lag + 1, step))
        if max_lag not in non_negative_lags:
            non_negative_lags.append(max_lag)
        lags = non_negative_lags + [-lag for lag in non_negative_lags]
        return sorted(set(lags))

    def _lagged_corr(self, pred_band: torch.Tensor, target_band: torch.Tensor, lag_samples: int) -> torch.Tensor:
        n = pred_band.shape[-1]
        if lag_samples > 0:
            pred_slice = pred_band[..., lag_samples:]
            target_slice = target_band[..., : n - lag_samples]
        elif lag_samples < 0:
            lead = -lag_samples
            pred_slice = pred_band[..., : n - lead]
            target_slice = target_band[..., lead:]
        else:
            pred_slice = pred_band
            target_slice = target_band
        pred_norm = self._zscore(pred_slice)
        target_norm = self._zscore(target_slice)
        return torch.mean(pred_norm * target_norm, dim=-1).squeeze(1)

    def _phase_alignment_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_band = self._band_limited_waveform(pred)
        target_band = self._band_limited_waveform(target)
        zero_corr = self._lagged_corr(pred_band, target_band, 0)
        lags = self._lag_samples(pred.shape[-1], self.phase_alignment_max_sec, self.phase_alignment_step_sec)
        lag_corrs = [self._lagged_corr(pred_band, target_band, lag) for lag in lags]
        corr = torch.stack(lag_corrs, dim=-1)
        weights = torch.softmax(corr / self.phase_alignment_temperature, dim=-1)
        soft_best_corr = torch.sum(weights * corr, dim=-1)
        lag_sec = pred.new_tensor(lags, dtype=pred.dtype) / self.fs
        soft_abs_lag_sec = torch.sum(weights * lag_sec.abs(), dim=-1)
        max_lag_sec = max(self.phase_alignment_max_sec, 1.0 / self.fs)
        lag_penalty = soft_abs_lag_sec / max_lag_sec
        loss = (
            self.phase_alignment_zero_weight * (1.0 - zero_corr)
            + self.phase_alignment_lag_weight * (1.0 - soft_best_corr)
            + self.phase_alignment_lag_penalty_weight * lag_penalty
        )
        return torch.mean(loss)

    def _band_waveform_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_band = self._zscore(self._band_limited_waveform(pred))
        target_band = self._zscore(self._band_limited_waveform(target))
        return F.smooth_l1_loss(pred_band, target_band)

    def _curvature_loss(self, pred: torch.Tensor) -> torch.Tensor:
        pred_norm = self._zscore(pred)
        if pred_norm.shape[-1] < 3:
            return pred_norm.new_tensor(0.0)
        second_diff = pred_norm[..., 2:] - 2.0 * pred_norm[..., 1:-1] + pred_norm[..., :-2]
        return torch.mean(torch.abs(second_diff))

    def _band_mask(self, n_samples: int, device: torch.device) -> torch.Tensor:
        freqs = torch.fft.rfftfreq(n_samples, d=1.0 / self.fs).to(device)
        mask = (freqs >= self.low_hz) & (freqs <= self.high_hz)
        if not bool(mask.any()):
            raise ValueError(
                f"频谱损失频带为空: low_hz={self.low_hz} high_hz={self.high_hz} "
                f"fs={self.fs} n={n_samples}"
            )
        return mask

    def _high_frequency_energy(self, pred: torch.Tensor) -> torch.Tensor:
        """惩罚预测中高于呼吸频带上限的相对频谱能量。"""
        centered = pred - pred.mean(dim=-1, keepdim=True)
        power = torch.fft.rfft(centered, dim=-1).abs().square().squeeze(1)
        freqs = torch.fft.rfftfreq(pred.shape[-1], d=1.0 / self.fs).to(pred.device)
        mask = freqs > self.high_hz
        if not bool(mask.any()):
            return power.new_tensor(0.0)
        total = torch.clamp(power.sum(dim=-1), min=1e-8)
        return torch.mean(power[:, mask].sum(dim=-1) / total)

    def _relative_envelope_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_rel = self._relative_envelope_trace(pred)
        target_rel = self._relative_envelope_trace(target)
        return torch.mean(torch.abs(pred_rel - target_rel))

    def _relative_envelope_trace(self, x: torch.Tensor) -> torch.Tensor:
        env = self._rms_envelope(x)
        trend_window = min(self.relative_env_trend_window, env.shape[-1])
        # avg_pool1d 要求 padding 不超过半个 kernel；偶数窗口短序列时使用奇数窗口更稳。
        if trend_window > 1 and trend_window % 2 == 0:
            trend_window -= 1
        pad = trend_window // 2
        trend = F.avg_pool1d(env, kernel_size=trend_window, stride=1, padding=pad)
        if trend.shape[-1] > env.shape[-1]:
            trend = trend[..., : env.shape[-1]]
        rel = torch.log(torch.clamp(env, min=1e-8)) - torch.log(torch.clamp(trend, min=1e-8))
        return rel - rel.mean(dim=-1, keepdim=True)

    @staticmethod
    def _center(x: torch.Tensor) -> torch.Tensor:
        return x - x.mean(dim=-1, keepdim=True)

    @staticmethod
    def _zscore(x: torch.Tensor) -> torch.Tensor:
        std = x.std(dim=-1, keepdim=True)
        return (x - x.mean(dim=-1, keepdim=True)) / torch.clamp(std, min=1e-6)
