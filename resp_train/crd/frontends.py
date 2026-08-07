from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from resp_train.crd.blocks import ResidualDWBlock
from resp_train.crd.spectral_ops import GaussianAnalyticFilterbank
from resp_train.models.timeseries import PatchMixer1D


def _initialize_conv_norm(conv: nn.Conv1d, norm: nn.GroupNorm) -> None:
    nn.init.kaiming_normal_(conv.weight, mode="fan_in", nonlinearity="relu")
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)
    nn.init.ones_(norm.weight)
    nn.init.zeros_(norm.bias)


class PatchTokenFrontend(nn.Module):
    """B0 的 post-mixer tokens 到 CRD 10-Hz latent 的桥接前端。"""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = LegacyPatchTokenEncoder()
        self.adapter = nn.Conv1d(16, 96, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(12, 96, eps=1e-5, affine=True)
        self.activation = nn.SiLU()
        _initialize_conv_norm(self.adapter, self.norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(x)
        if x.shape[-1] == 18000 and tokens.shape[-1] != 140:
            raise RuntimeError(f"CRD patch bridge 期望 140 tokens，实际 {tokens.shape[-1]}")
        tokens = F.interpolate(tokens, size=1800, mode="linear", align_corners=False)
        return self.activation(self.norm(self.adapter(tokens)))


class LegacyPatchTokenEncoder(nn.Module):
    """只保留 B0 的 patch embedding 与两个 mixer，不注册未使用的 patch head。"""

    def __init__(self) -> None:
        super().__init__()
        # 先按原类完整构造，再只接管共享 encoder modules；这样初始化逐 tensor 对齐 B0。
        reference = PatchMixer1D(
            in_channels=1,
            out_channels=1,
            base_channels=16,
            patch_len=256,
            patch_stride=128,
            mixer_layers=2,
            overlap_window="hann",
            output_smoothing_kernel=1,
        )
        self.patch_len = reference.patch_len
        self.patch_stride = reference.patch_stride
        self.patch_embed = reference.patch_embed
        self.blocks = reference.blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, length = x.shape
        if length <= self.patch_len:
            padded_length = self.patch_len
        else:
            steps = math.ceil((length - self.patch_len) / self.patch_stride)
            padded_length = steps * self.patch_stride + self.patch_len
        if padded_length > length:
            x = F.pad(x, (0, padded_length - length))
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        tokens = patches.permute(0, 2, 1, 3).reshape(batch, patches.shape[2], -1)
        tokens = self.patch_embed(tokens).transpose(1, 2)
        for block in self.blocks:
            tokens = block(tokens)
        return tokens


class DirectAnalyticFrontend(nn.Module):
    """六带 analytic filterbank 与固定 Direct encoder。"""

    def __init__(self) -> None:
        super().__init__()
        self.filterbank = GaussianAnalyticFilterbank(sample_rate=100.0, length=18000)
        self.conv_in = nn.Conv1d(12, 48, kernel_size=7, padding=3, bias=False)
        self.norm_in = nn.GroupNorm(6, 48, eps=1e-5, affine=True)
        self.conv_out = nn.Conv1d(48, 96, kernel_size=1, bias=False)
        self.norm_out = nn.GroupNorm(12, 96, eps=1e-5, affine=True)
        self.activation = nn.SiLU()
        self.residual = nn.Sequential(
            ResidualDWBlock(96, dilation=1),
            ResidualDWBlock(96, dilation=2),
        )
        _initialize_conv_norm(self.conv_in, self.norm_in)
        _initialize_conv_norm(self.conv_out, self.norm_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        analytic_10hz = self.filterbank(x)[..., ::10]
        latent = self.activation(self.norm_in(self.conv_in(analytic_10hz)))
        latent = self.activation(self.norm_out(self.conv_out(latent)))
        return self.residual(latent)
