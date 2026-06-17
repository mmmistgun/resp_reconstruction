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
        self.band_waveform_weight = float(cfg.loss.get("band_waveform_weight", 0.0))
        self.phase_lag_weight = float(cfg.loss.get("phase_lag_weight", 0.0))
        self.phase_lag_max_sec = float(cfg.loss.get("phase_lag_max_sec", 1.0))
        self.phase_lag_step_sec = float(cfg.loss.get("phase_lag_step_sec", 0.1))
        self.phase_lag_temperature = float(cfg.loss.get("phase_lag_temperature", 0.05))
        self.phase_alignment_weight = float(cfg.loss.get("phase_alignment_weight", 0.0))
        self.phase_alignment_zero_weight = float(cfg.loss.get("phase_alignment_zero_weight", 0.5))
        self.phase_alignment_lag_weight = float(cfg.loss.get("phase_alignment_lag_weight", 0.5))
        self.phase_alignment_lag_penalty_weight = float(cfg.loss.get("phase_alignment_lag_penalty_weight", 0.1))
        self.phase_alignment_max_sec = float(cfg.loss.get("phase_alignment_max_sec", 0.5))
        self.phase_alignment_step_sec = float(cfg.loss.get("phase_alignment_step_sec", 0.1))
        self.phase_alignment_temperature = float(cfg.loss.get("phase_alignment_temperature", 0.05))
        self.stft_mag_weight = float(cfg.loss.get("stft_mag_weight", 0.0))
        self.stft_phase_weight = float(cfg.loss.get("stft_phase_weight", 0.0))
        self.stft_complex_weight = float(cfg.loss.get("stft_complex_weight", 0.0))
        self.stft_window_sec = float(cfg.loss.get("stft_window_sec", 32.0))
        self.stft_hop_sec = float(cfg.loss.get("stft_hop_sec", 4.0))
        self.stft_n_fft_sec = float(cfg.loss.get("stft_n_fft_sec", 64.0))
        self.stft_low_hz = float(cfg.loss.get("stft_low_hz", cfg.loss.spectrum_low_hz))
        self.stft_high_hz = float(cfg.loss.get("stft_high_hz", cfg.loss.spectrum_high_hz))
        self.stft_log_magnitude = bool(cfg.loss.get("stft_log_magnitude", True))
        self.stft_eps = float(cfg.loss.get("stft_eps", 1e-6))
        if self.phase_lag_weight < 0:
            raise ValueError(f"phase_lag_weight 必须非负，当前={self.phase_lag_weight}")
        if self.phase_lag_max_sec < 0:
            raise ValueError(f"phase_lag_max_sec 必须非负，当前={self.phase_lag_max_sec}")
        if self.phase_lag_step_sec <= 0:
            raise ValueError(f"phase_lag_step_sec 必须为正数，当前={self.phase_lag_step_sec}")
        if self.phase_lag_temperature <= 0:
            raise ValueError(f"phase_lag_temperature 必须为正数，当前={self.phase_lag_temperature}")
        for name, value in (
            ("phase_alignment_weight", self.phase_alignment_weight),
            ("phase_alignment_zero_weight", self.phase_alignment_zero_weight),
            ("phase_alignment_lag_weight", self.phase_alignment_lag_weight),
            ("phase_alignment_lag_penalty_weight", self.phase_alignment_lag_penalty_weight),
        ):
            if value < 0:
                raise ValueError(f"{name} 必须非负，当前={value}")
        if self.phase_alignment_max_sec < 0:
            raise ValueError(f"phase_alignment_max_sec 必须非负，当前={self.phase_alignment_max_sec}")
        if self.phase_alignment_step_sec <= 0:
            raise ValueError(f"phase_alignment_step_sec 必须为正数，当前={self.phase_alignment_step_sec}")
        if self.phase_alignment_temperature <= 0:
            raise ValueError(f"phase_alignment_temperature 必须为正数，当前={self.phase_alignment_temperature}")
        for name, value in (
            ("stft_mag_weight", self.stft_mag_weight),
            ("stft_phase_weight", self.stft_phase_weight),
            ("stft_complex_weight", self.stft_complex_weight),
        ):
            if value < 0:
                raise ValueError(f"{name} 必须非负，当前={value}")
        if self.stft_window_sec <= 0:
            raise ValueError(f"stft_window_sec 必须为正数，当前={self.stft_window_sec}")
        if self.stft_hop_sec <= 0:
            raise ValueError(f"stft_hop_sec 必须为正数，当前={self.stft_hop_sec}")
        if self.stft_n_fft_sec <= 0:
            raise ValueError(f"stft_n_fft_sec 必须为正数，当前={self.stft_n_fft_sec}")
        if not 0 < self.stft_low_hz < self.stft_high_hz < self.fs / 2:
            raise ValueError(f"STFT 频带非法: low={self.stft_low_hz} high={self.stft_high_hz} fs={self.fs}")
        if self.stft_eps <= 0:
            raise ValueError(f"stft_eps 必须为正数，当前={self.stft_eps}")
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
        if self.phase_lag_weight > 0:
            phase_lag = self._phase_lag_loss(pred_loss, target_loss)
        else:
            phase_lag = pred_loss.new_tensor(0.0)
        if self.phase_alignment_weight > 0:
            phase_alignment = self._phase_alignment_loss(pred_loss, target_loss)
        else:
            phase_alignment = pred_loss.new_tensor(0.0)
        if self.stft_mag_weight > 0 or self.stft_phase_weight > 0 or self.stft_complex_weight > 0:
            pred_stft, target_stft = self._low_frequency_stft_pair(pred_loss, target_loss)
            stft_magnitude = self._stft_magnitude_loss(pred_stft, target_stft)
            stft_phase = self._stft_phase_loss(pred_stft, target_stft)
            stft_complex = self._stft_complex_loss(pred_stft, target_stft)
        else:
            stft_magnitude = pred_loss.new_tensor(0.0)
            stft_phase = pred_loss.new_tensor(0.0)
            stft_complex = pred_loss.new_tensor(0.0)
        total = (
            self.env_weight * env
            + self.spec_weight * spec
            + self.smooth_weight * smooth
            + self.high_freq_weight * high_freq
            + self.relative_env_weight * relative_env
            + self.band_waveform_weight * band_waveform
            + self.phase_lag_weight * phase_lag
            + self.phase_alignment_weight * phase_alignment
            + self.stft_mag_weight * stft_magnitude
            + self.stft_phase_weight * stft_phase
            + self.stft_complex_weight * stft_complex
        )
        return total, {
            "envelope": env.detach(),
            "spectrum": spec.detach(),
            "smooth": smooth.detach(),
            "high_freq": high_freq.detach(),
            "relative_envelope": relative_env.detach(),
            "band_waveform": band_waveform.detach(),
            "phase_lag": phase_lag.detach(),
            "phase_alignment": phase_alignment.detach(),
            "stft_magnitude": stft_magnitude.detach(),
            "stft_phase": stft_phase.detach(),
            "stft_complex": stft_complex.detach(),
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

    def _band_waveform_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_band = self._zscore(self._band_limited_waveform(pred))
        target_band = self._zscore(self._band_limited_waveform(target))
        return torch.mean(torch.abs(pred_band - target_band))

    def _band_limited_waveform(self, x: torch.Tensor) -> torch.Tensor:
        centered = self._center(x)
        spectrum = torch.fft.rfft(centered, dim=-1)
        mask = self._band_mask(x.shape[-1], x.device).view(1, 1, -1)
        filtered = torch.where(mask, spectrum, torch.zeros_like(spectrum))
        return torch.fft.irfft(filtered, n=x.shape[-1], dim=-1)

    def _phase_lag_samples(self, n_samples: int) -> list[int]:
        return self._lag_samples(n_samples, self.phase_lag_max_sec, self.phase_lag_step_sec)

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

    def _phase_lag_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_band = self._band_limited_waveform(pred)
        target_band = self._band_limited_waveform(target)
        lag_corrs = [
            self._lagged_corr(pred_band, target_band, lag)
            for lag in self._phase_lag_samples(pred.shape[-1])
        ]
        corr = torch.stack(lag_corrs, dim=-1)
        # 用 softmax 近似最佳 lag，保留相位容忍损失的可微性。
        weights = torch.softmax(corr / self.phase_lag_temperature, dim=-1)
        soft_best_corr = torch.sum(weights * corr, dim=-1)
        return torch.mean(1.0 - soft_best_corr)

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

    def _low_frequency_stft_pair(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pred_stft = self._low_frequency_stft(pred)
        target_stft = self._low_frequency_stft(target)
        return pred_stft, target_stft

    def _low_frequency_stft(self, x: torch.Tensor) -> torch.Tensor:
        centered = self._center(x).squeeze(1)
        n_samples = centered.shape[-1]
        n_fft = min(n_samples, max(16, int(round(self.stft_n_fft_sec * self.fs))))
        win_length = min(n_fft, max(8, int(round(self.stft_window_sec * self.fs))))
        hop_length = max(1, int(round(self.stft_hop_sec * self.fs)))
        window = torch.hann_window(win_length, device=x.device, dtype=centered.dtype)
        spectrum = torch.stft(
            centered,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            return_complex=True,
        )
        freqs = torch.fft.rfftfreq(n_fft, d=1.0 / self.fs).to(x.device)
        mask = (freqs >= self.stft_low_hz) & (freqs <= self.stft_high_hz)
        if not bool(mask.any()):
            raise ValueError(
                f"STFT 频带为空: low_hz={self.stft_low_hz} high_hz={self.stft_high_hz} "
                f"fs={self.fs} n_fft={n_fft}"
            )
        return spectrum[:, mask, :]

    def _stft_magnitude_loss(self, pred_stft: torch.Tensor, target_stft: torch.Tensor) -> torch.Tensor:
        pred_mag = pred_stft.abs()
        target_mag = target_stft.abs()
        if self.stft_log_magnitude:
            pred_mag = torch.log1p(pred_mag)
            target_mag = torch.log1p(target_mag)
        return torch.mean(torch.abs(pred_mag - target_mag))

    def _stft_phase_loss(self, pred_stft: torch.Tensor, target_stft: torch.Tensor) -> torch.Tensor:
        target_mag = target_stft.abs()
        weights = target_mag / torch.clamp(target_mag.sum(dim=(1, 2), keepdim=True), min=self.stft_eps)
        phase_diff = torch.angle(pred_stft * target_stft.conj())
        return torch.mean(torch.sum(weights * (1.0 - torch.cos(phase_diff)), dim=(1, 2)))

    def _stft_complex_loss(self, pred_stft: torch.Tensor, target_stft: torch.Tensor) -> torch.Tensor:
        pred_scale = torch.clamp(pred_stft.abs().mean(dim=(1, 2), keepdim=True), min=self.stft_eps)
        target_scale = torch.clamp(target_stft.abs().mean(dim=(1, 2), keepdim=True), min=self.stft_eps)
        pred_norm = pred_stft / pred_scale
        target_norm = target_stft / target_scale
        return torch.mean(torch.abs(pred_norm - target_norm))

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
