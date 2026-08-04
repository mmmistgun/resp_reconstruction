from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from resp_train.models.timeseries import PatchMixer1D


class RespiratoryBandStftFeatures(nn.Module):
    """从单通道 BCG 生成固定呼吸带的局部谱形与相对带内能量。"""

    def __init__(
        self,
        *,
        sample_rate: float,
        window_samples: int,
        hop_samples: int,
        low_hz: float,
        high_hz: float,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.sample_rate = float(sample_rate)
        self.window_samples = int(window_samples)
        self.hop_samples = int(hop_samples)
        self.low_hz = float(low_hz)
        self.high_hz = float(high_hz)
        self.eps = float(eps)

        if self.sample_rate <= 0:
            raise ValueError("sample_rate 必须大于 0")
        if self.window_samples <= 1 or self.hop_samples <= 0:
            raise ValueError("STFT window_samples 必须大于 1，hop_samples 必须大于 0")
        if not (0.0 <= self.low_hz < self.high_hz <= self.sample_rate / 2.0):
            raise ValueError("STFT 频带必须满足 0 <= low_hz < high_hz <= Nyquist")
        if self.eps <= 0:
            raise ValueError("STFT feature eps 必须大于 0")

        # 频带端点按离散 rFFT bin 包含；30 秒窗、100 Hz、0.05–0.70 Hz 对应 k=2…21。
        min_bin = math.ceil(self.low_hz * self.window_samples / self.sample_rate - 1e-12)
        max_bin = math.floor(self.high_hz * self.window_samples / self.sample_rate + 1e-12)
        nyquist_bin = self.window_samples // 2
        min_bin = max(min_bin, 0)
        max_bin = min(max_bin, nyquist_bin)
        if max_bin < min_bin:
            raise ValueError("STFT 呼吸频带内没有可用 rFFT bin")

        band_indices = torch.arange(min_bin, max_bin + 1, dtype=torch.long)
        self.register_buffer("band_indices", band_indices, persistent=False)
        self.register_buffer(
            "window",
            torch.hann_window(self.window_samples, periodic=False, dtype=torch.float32),
            persistent=False,
        )

    @property
    def band_bin_count(self) -> int:
        return int(self.band_indices.numel())

    @property
    def feature_channels(self) -> int:
        return self.band_bin_count + 1

    def frame_count(self, signal_length: int) -> int:
        signal_length = int(signal_length)
        if signal_length < self.window_samples:
            raise ValueError(
                f"输入长度 {signal_length} 小于 STFT 窗长 {self.window_samples}"
            )
        return (signal_length - self.window_samples) // self.hop_samples + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(1) != 1:
            raise ValueError(f"STFT 特征期望输入 (B, 1, L)，实际为 {tuple(x.shape)}")
        self.frame_count(x.size(-1))

        # 谱计算固定为 float32，避免 AMP 改变低能量功率和归一化语义。
        waveform = x[:, 0].to(dtype=torch.float32)
        frames = waveform.unfold(-1, self.window_samples, self.hop_samples)
        frames = frames - frames.mean(dim=-1, keepdim=True)
        window = self.window.to(device=frames.device)
        spectrum = torch.fft.rfft(frames * window, n=self.window_samples, dim=-1, norm="backward")
        power = spectrum.real.square() + spectrum.imag.square()
        power = power.index_select(-1, self.band_indices.to(device=power.device))

        log_power = torch.log1p(power)
        shape_mean = log_power.mean(dim=-1, keepdim=True)
        shape_var = (log_power - shape_mean).square().mean(dim=-1, keepdim=True)
        spectral_shape = (log_power - shape_mean) / torch.sqrt(shape_var + self.eps)

        # 公共输出会消除整窗绝对尺度，因此这里只保留窗内随时间变化的相对 effort。
        log_band_energy = torch.log1p(power.mean(dim=-1))
        energy_mean = log_band_energy.mean(dim=-1, keepdim=True)
        energy_var = (log_band_energy - energy_mean).square().mean(dim=-1, keepdim=True)
        relative_energy = (log_band_energy - energy_mean) / torch.sqrt(energy_var + self.eps)

        return torch.cat(
            [spectral_shape.transpose(1, 2), relative_energy.unsqueeze(1)],
            dim=1,
        )


class TimeStftFusion1D(nn.Module):
    """PatchMixer 时间分支加一个固定契约的呼吸带 STFT 残差分支。"""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        patch_len: int = 256,
        patch_stride: int = 128,
        mixer_layers: int = 2,
        overlap_window: str = "hann",
        output_smoothing_kernel: int = 1,
        sample_rate: float = 100.0,
        stft_window_samples: int = 3000,
        stft_hop_samples: int = 1000,
        stft_low_hz: float = 0.05,
        stft_high_hz: float = 0.70,
        stft_feature_eps: float = 1e-8,
        stft_channels: int = 16,
    ) -> None:
        super().__init__()
        if int(in_channels) != 1:
            raise ValueError("T1 固定要求单通道 BCG 输入")
        if int(stft_channels) <= 0:
            raise ValueError("stft_channels 必须大于 0")

        # 先构建时间分支；在相同 seed 下，它与 B0 PatchMixer 具有相同初始化。
        self.time_backbone = PatchMixer1D(
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            base_channels=int(base_channels),
            patch_len=int(patch_len),
            patch_stride=int(patch_stride),
            mixer_layers=int(mixer_layers),
            overlap_window=str(overlap_window),
            output_smoothing_kernel=int(output_smoothing_kernel),
        )
        self.stft_features = RespiratoryBandStftFeatures(
            sample_rate=float(sample_rate),
            window_samples=int(stft_window_samples),
            hop_samples=int(stft_hop_samples),
            low_hz=float(stft_low_hz),
            high_hz=float(stft_high_hz),
            eps=float(stft_feature_eps),
        )
        feature_channels = self.stft_features.feature_channels
        self.stft_encoder = nn.Sequential(
            nn.Conv1d(feature_channels, int(stft_channels), kernel_size=3, padding=1),
            nn.GroupNorm(1, int(stft_channels)),
            nn.SiLU(),
            nn.Conv1d(int(stft_channels), int(stft_channels), kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.stft_projection = nn.Conv1d(int(stft_channels), int(base_channels), kernel_size=1)
        # T1 在初始化时严格退化为 B0；一次 optimizer step 后 projection 即可学习非零注入。
        nn.init.zeros_(self.stft_projection.weight)
        nn.init.zeros_(self.stft_projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(1) != 1:
            raise ValueError(f"T1 期望输入 (B, 1, L)，实际为 {tuple(x.shape)}")

        frame_features = self.stft_features(x)
        encoder_dtype = self.stft_encoder[0].weight.dtype
        encoded = self.stft_encoder(frame_features.to(dtype=encoder_dtype))
        token_count = self.time_backbone.token_count_for_length(x.size(-1))
        aligned = F.interpolate(encoded, size=token_count, mode="linear", align_corners=False)
        injection = self.stft_projection(aligned)
        return self.time_backbone.forward_with_token_injection(
            x,
            token_injection=injection,
            inject_position="post_mixer",
        )
