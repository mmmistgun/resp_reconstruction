from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """一维卷积基础块，保持时间长度不变。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.SiLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet1DTiny(nn.Module):
    """轻量一维 U-Net，只输出重建后的单通道 waveform。"""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 16) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.down1 = nn.Conv1d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.enc2 = ConvBlock(base_channels * 2, base_channels * 2)
        self.down2 = nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)

        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 4)

        self.up2 = nn.ConvTranspose1d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        self.out = nn.Conv1d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        z = self.bottleneck(self.down2(e2))

        d2 = self.up2(z)
        d2 = self.dec2(torch.cat([self._match_length(d2, e2), e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([self._match_length(d1, e1), e1], dim=1))
        return self.out(d1)

    @staticmethod
    def _match_length(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """对齐跳连张量长度，避免奇数采样点导致拼接失败。"""
        diff = reference.size(-1) - x.size(-1)
        if diff > 0:
            return F.pad(x, (0, diff))
        if diff < 0:
            return x[..., : reference.size(-1)]
        return x


class UNet1DTinyNoSkip1(nn.Module):
    """去掉最后浅层跳连的一维 U-Net，减少输入高频细节直通输出。"""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 16) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.down1 = nn.Conv1d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.enc2 = ConvBlock(base_channels * 2, base_channels * 2)
        self.down2 = nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)

        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 4)

        self.up2 = nn.ConvTranspose1d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.dec1 = ConvBlock(base_channels, base_channels)
        self.out = nn.Conv1d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        z = self.bottleneck(self.down2(e2))

        d2 = self.up2(z)
        d2 = self.dec2(torch.cat([UNet1DTiny._match_length(d2, e2), e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(UNet1DTiny._match_length(d1, e1))
        return self.out(d1)
