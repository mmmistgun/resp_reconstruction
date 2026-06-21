from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def _match_length(x: torch.Tensor, length: int) -> torch.Tensor:
    diff = int(length) - x.size(-1)
    if diff > 0:
        return F.pad(x, (0, diff))
    if diff < 0:
        return x[..., :length]
    return x


def _odd_kernel(value: int, *, minimum: int = 3) -> int:
    value = max(int(value), minimum)
    return value if value % 2 == 1 else value + 1


class ChannelNorm1D(nn.Module):
    """按窗口做可逆标准化，减少不同片段幅值尺度影响。"""

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(self.eps)
        return (x - mean) / std, mean, std


class DepthwiseMovingAverage1D(nn.Module):
    """每个通道独立移动平均，用于低频趋势提取。"""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        kernel_size = _odd_kernel(kernel_size)
        self.padding = kernel_size // 2
        weight = torch.full((int(channels), 1, kernel_size), 1.0 / kernel_size)
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padded = F.pad(x, (self.padding, self.padding), mode="reflect")
        return F.conv1d(padded, self.weight, groups=x.size(1))


class BasisDecoder1D(nn.Module):
    """用少量低频基函数重建整段呼吸轨迹，显式限制输出自由度。"""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        basis_count: int = 96,
        encoder_stride: int = 20,
        duration_samples: int = 18000,
    ) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        self.basis_count = max(int(basis_count), 8)
        self.duration_samples = int(duration_samples)
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=9, stride=int(encoder_stride), padding=4),
            nn.GroupNorm(1, base_channels),
            nn.SiLU(),
            nn.Conv1d(base_channels, base_channels, kernel_size=9, padding=4),
            nn.GroupNorm(1, base_channels),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.coeff_head = nn.Linear(base_channels, out_channels * self.basis_count)
        basis = self._build_basis(self.duration_samples, self.basis_count)
        self.register_buffer("basis", basis, persistent=False)

    @staticmethod
    def _build_basis(length: int, basis_count: int) -> torch.Tensor:
        t = torch.linspace(0.0, 1.0, int(length), dtype=torch.float32)
        cols = [torch.ones_like(t)]
        max_harmonics = max(math.ceil((int(basis_count) - 1) / 2), 1)
        for k in range(1, max_harmonics + 1):
            cols.append(torch.sin(2.0 * math.pi * k * t))
            if len(cols) >= basis_count:
                break
            cols.append(torch.cos(2.0 * math.pi * k * t))
            if len(cols) >= basis_count:
                break
        basis = torch.stack(cols[:basis_count], dim=0)
        return basis / basis.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(-1)
        features = self.pool(self.encoder(x)).squeeze(-1)
        coeff = self.coeff_head(features).view(x.size(0), self.out_channels, self.basis_count)
        basis = self.basis.to(device=x.device, dtype=x.dtype)
        if basis.size(-1) != length:
            basis = F.interpolate(basis.unsqueeze(0), size=length, mode="linear", align_corners=False).squeeze(0)
        return torch.einsum("bck,kl->bcl", coeff, basis)


class MultiScaleDecompMixer1D(nn.Module):
    """TimeMixer/MICN-lite：多尺度低频分解后融合输出。"""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        downsample_factors: list[int] | tuple[int, ...] = (1, 4, 16),
        mixer_layers: int = 2,
    ) -> None:
        super().__init__()
        self.factors = tuple(int(v) for v in downsample_factors)
        self.branches = nn.ModuleList()
        for factor in self.factors:
            layers: list[nn.Module] = [
                nn.Conv1d(in_channels, base_channels, kernel_size=9, padding=4),
                nn.GroupNorm(1, base_channels),
                nn.SiLU(),
            ]
            for _ in range(max(int(mixer_layers), 1)):
                layers.extend(
                    [
                        nn.Conv1d(base_channels, base_channels, kernel_size=7, padding=3, groups=base_channels),
                        nn.Conv1d(base_channels, base_channels, kernel_size=1),
                        nn.GroupNorm(1, base_channels),
                        nn.SiLU(),
                    ]
                )
            self.branches.append(nn.Sequential(*layers))
        self.fuse = nn.Sequential(
            nn.Conv1d(base_channels * len(self.factors), base_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(base_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor | tuple[torch.Tensor, int]:
        length = x.size(-1)
        outputs = []
        for factor, branch in zip(self.factors, self.branches):
            if factor > 1:
                pooled = F.avg_pool1d(x, kernel_size=factor, stride=factor, ceil_mode=True)
            else:
                pooled = x
            y = branch(pooled)
            if y.size(-1) != length:
                y = F.interpolate(y, size=length, mode="linear", align_corners=False)
            outputs.append(y)
        fused_input = torch.cat(outputs, dim=1)
        if return_features:
            return fused_input, length
        return self.fuse(fused_input)


class TimesNetLite1D(nn.Module):
    """在低频表示上做周期重排和 2D 卷积，避免直接让 BCG 高频主导周期选择。"""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        period_top_k: int = 3,
        period_min_sec: float = 2.0,
        period_max_sec: float = 20.0,
        sample_rate: float = 100.0,
        lowpass_kernel: int = 401,
    ) -> None:
        super().__init__()
        self.period_top_k = int(period_top_k)
        self.min_period = max(int(period_min_sec * sample_rate), 1)
        self.max_period = max(int(period_max_sec * sample_rate), self.min_period)
        self.lowpass = DepthwiseMovingAverage1D(in_channels, lowpass_kernel)
        self.embed = nn.Conv1d(in_channels, base_channels, kernel_size=1)
        self.period_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=(3, 5), padding=(1, 2), groups=base_channels),
            nn.Conv2d(base_channels, base_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=(3, 5), padding=(1, 2), groups=base_channels),
            nn.Conv2d(base_channels, base_channels, kernel_size=1),
        )
        self.out = nn.Conv1d(base_channels, out_channels, kernel_size=1)

    def _periods(self, x: torch.Tensor) -> list[int]:
        spectrum = torch.fft.rfft(x.float(), dim=-1).abs().mean(dim=(0, 1))
        spectrum[0] = 0
        length = x.size(-1)
        min_bin = max(length // self.max_period, 1)
        max_bin = min(length // self.min_period, spectrum.numel() - 1)
        if max_bin < min_bin:
            return [min(self.min_period, length)]

        masked = torch.zeros_like(spectrum)
        masked[min_bin : max_bin + 1] = spectrum[min_bin : max_bin + 1]
        top = torch.topk(masked, k=min(self.period_top_k, max_bin - min_bin + 1)).indices
        periods = [max(length // int(idx.item()), 1) for idx in top if int(idx.item()) > 0]
        return periods or [min(self.min_period, length)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(-1)
        low = self.lowpass(x)
        z = self.embed(low)
        outputs = []
        for period in self._periods(low):
            padded_len = math.ceil(length / period) * period
            z_pad = _match_length(z, padded_len)
            grid = z_pad.view(z.size(0), z.size(1), padded_len // period, period)
            y = self.period_conv(grid).reshape(z.size(0), z.size(1), padded_len)[..., :length]
            outputs.append(y)
        mixed = torch.stack(outputs, dim=0).mean(dim=0)
        return self.out(mixed + z)


class FrequencyBottleneck1D(nn.Module):
    """只预测低频 spectrum bins，再 irfft 回到时域，限制高频输出自由度。"""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        freq_bins: int = 128,
        max_freq_hz: float = 0.7,
        sample_rate: float = 100.0,
        duration_samples: int = 18000,
    ) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        self.duration_samples = int(duration_samples)
        max_bins = int(max_freq_hz * self.duration_samples / sample_rate) + 1
        self.freq_bins = max(2, min(int(freq_bins), max_bins))
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=17, stride=20, padding=8),
            nn.GroupNorm(1, base_channels),
            nn.SiLU(),
            nn.Conv1d(base_channels, base_channels, kernel_size=9, padding=4),
            nn.GroupNorm(1, base_channels),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.real_head = nn.Linear(base_channels, out_channels * self.freq_bins)
        self.imag_head = nn.Linear(base_channels, out_channels * self.freq_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(-1)
        features = self.encoder(x).squeeze(-1)
        real = self.real_head(features).view(x.size(0), self.out_channels, self.freq_bins)
        imag = self.imag_head(features).view(x.size(0), self.out_channels, self.freq_bins)
        imag[..., 0] = 0.0
        spectrum = torch.zeros(
            x.size(0),
            self.out_channels,
            length // 2 + 1,
            dtype=torch.complex64,
            device=x.device,
        )
        low = torch.complex(real.float(), imag.float())
        active_bins = min(self.freq_bins, spectrum.size(-1))
        spectrum[..., :active_bins] = low[..., :active_bins]
        y = torch.fft.irfft(spectrum, n=length, dim=-1)
        return y.to(dtype=x.dtype)


class GatedStateBlock1D(nn.Module):
    """轻量 SSM-like 门控状态块，用卷积门控近似长程状态更新。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.filter = nn.Conv1d(channels, channels, kernel_size=9, padding=4, groups=channels)
        self.gate = nn.Conv1d(channels, channels, kernel_size=1)
        self.mix = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        update = torch.tanh(self.filter(z))
        gate = torch.sigmoid(self.gate(z))
        return x + self.mix(update * gate)


class DownsampledSSM1D(nn.Module):
    """先降采样到低频 latent，再用门控状态块建模，最后上采样输出。"""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        latent_stride: int = 20,
        state_layers: int = 2,
    ) -> None:
        super().__init__()
        self.latent_stride = max(int(latent_stride), 1)
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=9, padding=4),
            nn.GroupNorm(1, base_channels),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(*[GatedStateBlock1D(base_channels) for _ in range(max(int(state_layers), 1))])
        self.output_proj = nn.Sequential(
            nn.Conv1d(base_channels, base_channels, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(base_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(-1)
        if self.latent_stride > 1:
            z = F.avg_pool1d(x, kernel_size=self.latent_stride, stride=self.latent_stride, ceil_mode=True)
        else:
            z = x
        z = self.blocks(self.input_proj(z))
        z = F.interpolate(z, size=length, mode="linear", align_corners=False)
        return self.output_proj(z)
