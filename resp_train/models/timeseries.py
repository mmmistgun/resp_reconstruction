from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from resp_train.models.unet1d import ConvBlock, UNet1DTiny


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


class MovingAverage1D(nn.Module):
    """按通道做移动平均，用作 DLinear/TimeMixer 风格的趋势分解。"""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        kernel_size = _odd_kernel(kernel_size)
        self.padding = kernel_size // 2
        weight = torch.full((channels, 1, kernel_size), 1.0 / kernel_size)
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padded = F.pad(x, (self.padding, self.padding), mode="reflect")
        return F.conv1d(padded, self.weight, groups=x.size(1))


class PeriodicUNet1DTiny(nn.Module):
    """TimesNet 周期建模启发的轻量 U-Net。

    这里不直接照搬 TimesNet 的 2D 周期卷积；对 18000 点窗口来说，先用多尺度
    深度卷积提取呼吸相关低频，再交给现有 U-Net，是更稳的最小改动。
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        lowpass_kernel: int = 101,
    ) -> None:
        super().__init__()
        self.smoother = MovingAverage1D(in_channels, lowpass_kernel)
        self.periodic_frontend = nn.Sequential(
            nn.Conv1d(in_channels * 2, base_channels, kernel_size=9, padding=4),
            nn.GroupNorm(1, base_channels),
            nn.SiLU(),
            nn.Conv1d(base_channels, base_channels, kernel_size=15, padding=7, groups=base_channels),
            nn.Conv1d(base_channels, in_channels, kernel_size=1),
        )
        self.backbone = UNet1DTiny(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        smooth = self.smoother(x)
        residual = self.periodic_frontend(torch.cat([x, smooth], dim=1))
        return self.backbone(smooth + residual)


class PatchMixerBlock(nn.Module):
    """PatchTST/TSMixer 风格的 patch 内与 patch 间混合。"""

    def __init__(self, channels: int, patch_count: int, hidden_channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(1, channels)
        self.patch_mixer = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.GELU(),
        )
        self.norm2 = nn.GroupNorm(1, channels)
        self.channel_mixer = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.patch_mixer(self.norm1(x))
        return x + self.channel_mixer(self.norm2(x))


class PatchMixer1D(nn.Module):
    """将长窗口切成 patch 后混合，再折回原长度。"""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        patch_len: int = 256,
        patch_stride: int = 128,
        mixer_layers: int = 2,
    ) -> None:
        super().__init__()
        self.patch_len = max(int(patch_len), 8)
        self.patch_stride = max(int(patch_stride), 1)
        self.patch_embed = nn.Linear(in_channels * self.patch_len, base_channels)
        self.blocks = nn.ModuleList(
            [PatchMixerBlock(base_channels, patch_count=1, hidden_channels=base_channels * 2) for _ in range(mixer_layers)]
        )
        self.patch_head = nn.Linear(base_channels, out_channels * self.patch_len)
        self.out_channels = int(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, length = x.shape
        padded_length = self._padded_length(length)
        x_padded = _match_length(x, padded_length)
        patches = x_padded.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        patch_count = patches.size(2)
        tokens = patches.permute(0, 2, 1, 3).reshape(batch, patch_count, -1)
        tokens = self.patch_embed(tokens).transpose(1, 2)
        for block in self.blocks:
            tokens = block(tokens)
        patch_values = self.patch_head(tokens.transpose(1, 2))
        patch_values = patch_values.view(batch, patch_count, self.out_channels, self.patch_len)
        return self._overlap_add(patch_values, length, padded_length)

    def _padded_length(self, length: int) -> int:
        if length <= self.patch_len:
            return self.patch_len
        steps = math.ceil((length - self.patch_len) / self.patch_stride)
        return steps * self.patch_stride + self.patch_len

    def _overlap_add(self, patches: torch.Tensor, length: int, padded_length: int) -> torch.Tensor:
        batch, patch_count, channels, _ = patches.shape
        out = patches.new_zeros(batch, channels, padded_length)
        weight = patches.new_zeros(batch, channels, padded_length)
        for idx in range(patch_count):
            start = idx * self.patch_stride
            end = start + self.patch_len
            out[..., start:end] = out[..., start:end] + patches[:, idx]
            weight[..., start:end] = weight[..., start:end] + 1
        out = out / weight.clamp_min(1)
        return out[..., :length]


class DLinearWaveform(nn.Module):
    """DLinear 风格的波形到波形基线。

    对每个时刻使用局部 1D 卷积分别映射 trend/seasonal，再相加输出；它不是
    预测未来窗口，而是作为同长度 waveform 重建的强基线。
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        moving_avg: int = 101,
    ) -> None:
        super().__init__()
        self.decomp = MovingAverage1D(in_channels, moving_avg)
        self.trend = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(base_channels, out_channels, kernel_size=1),
        )
        self.seasonal = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=9, padding=4),
            nn.GELU(),
            nn.Conv1d(base_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = self.decomp(x)
        seasonal = x - trend
        return self.trend(trend) + self.seasonal(seasonal)
