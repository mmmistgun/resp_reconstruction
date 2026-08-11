from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import nn


def _group_count(channels: int) -> int:
    groups = {48: 6, 64: 8, 96: 12, 128: 16}.get(int(channels))
    if groups is None:
        raise ValueError(f"CRD-v1 未定义 channels={channels} 的 GroupNorm groups")
    return groups


def _kaiming(module: nn.Conv1d) -> None:
    nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class ResidualDWBlock(nn.Module):
    """CRD-v1.1 唯一的一维 depthwise residual block。"""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        channels = int(channels)
        dilation = int(dilation)
        self.norm = nn.GroupNorm(_group_count(channels), channels, eps=1e-5, affine=True)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=5,
            padding=2 * dilation,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.expand = nn.Conv1d(channels, 2 * channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(0.10)
        self.project = nn.Conv1d(2 * channels, channels, kernel_size=1, bias=True)
        self.activation = nn.SiLU()
        _kaiming(self.depthwise)
        _kaiming(self.expand)
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.norm(x)
        residual = self.activation(self.depthwise(residual))
        residual = self.activation(self.expand(residual))
        residual = self.dropout(residual)
        return x + self.project(residual)


class LocalTCNBlock(nn.Module):
    """C1 参数匹配 full-context TCN 的唯一 residual block。"""

    def __init__(self, *, channels: int = 96, hidden_channels: int = 488, dilation: int) -> None:
        super().__init__()
        channels = int(channels)
        hidden_channels = int(hidden_channels)
        dilation = int(dilation)
        if channels != 96 or hidden_channels != 488 or dilation <= 0:
            raise ValueError("C1 LocalTCNBlock 固定 C=96、H=488 且 dilation>0")
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.dilation = dilation
        self.norm = nn.GroupNorm(12, channels, eps=1e-5, affine=True)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=5,
            padding=2 * dilation,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.expand = nn.Conv1d(channels, hidden_channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(0.10)
        self.project = nn.Conv1d(hidden_channels, channels, kernel_size=1, bias=True)
        self.activation = nn.SiLU()
        _kaiming(self.depthwise)
        _kaiming(self.expand)
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.channels:
            raise ValueError(f"C1 LocalTCNBlock 期望 (B,{self.channels},T)，实际 {tuple(x.shape)}")
        residual = self.norm(x)
        residual = self.activation(self.depthwise(residual))
        residual = self.activation(self.expand(residual))
        residual = self.dropout(residual)
        return x + self.project(residual)


class CustomRMSNorm(nn.Module):
    """不依赖 PyTorch RMSNorm 版本语义的外部 RMSNorm。"""

    def __init__(self, dimension: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(int(dimension), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps)
        normalized = x * scale.to(dtype=x.dtype)
        return normalized * self.weight.to(dtype=x.dtype)


class BidirectionalMamba2Block(nn.Module):
    """独立正反向 Mamba2，经 concat+linear 合并后做残差。"""

    def __init__(
        self,
        dimension: int,
        *,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 32,
        ngroups: int = 1,
        chunk_size: int = 256,
        dropout: float = 0.10,
        mamba_factory: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if mamba_factory is None:
            try:
                from mamba_ssm import Mamba2
            except ImportError as exc:
                raise RuntimeError(
                    "CRD-v1 Mamba 变体要求 mamba-ssm==2.3.2.post1；不允许自动 fallback"
                ) from exc
            mamba_factory = Mamba2

        dimension = int(dimension)
        kwargs = {
            "d_model": dimension,
            "d_state": int(d_state),
            "d_conv": int(d_conv),
            "expand": int(expand),
            "headdim": int(headdim),
            "ngroups": int(ngroups),
            "chunk_size": int(chunk_size),
            "rmsnorm": True,
            "bias": False,
            "conv_bias": True,
            "use_mem_eff_path": True,
        }
        self.norm = CustomRMSNorm(dimension, eps=1e-5)
        self.forward_mamba = mamba_factory(**kwargs)
        self.backward_mamba = mamba_factory(**kwargs)
        self.merge = nn.Linear(2 * dimension, dimension, bias=True)
        nn.init.kaiming_uniform_(self.merge.weight, a=math.sqrt(5.0))
        fan_in = self.merge.weight.shape[1]
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(self.merge.bias, -bound, bound)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"BiMamba2 期望 (B,L,D)，实际 {tuple(x.shape)}")
        normalized = self.norm(x)
        forward = self.forward_mamba(normalized)
        backward = torch.flip(
            self.backward_mamba(torch.flip(normalized, dims=(1,))),
            dims=(1,),
        )
        merged = self.merge(torch.cat((forward, backward), dim=-1))
        return x + self.dropout(merged)


class CoarseWaveformHead(nn.Module):
    """固定的 10-Hz coarse waveform head。"""

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(12, 96, eps=1e-5, affine=True)
        self.conv = nn.Conv1d(96, 64, kernel_size=5, padding=2, bias=False)
        self.depthwise = nn.Conv1d(64, 64, kernel_size=5, padding=2, groups=64, bias=False)
        self.reduce = nn.Conv1d(64, 32, kernel_size=1, bias=False)
        self.output = nn.Conv1d(32, 1, kernel_size=1, bias=True)
        self.activation = nn.SiLU()
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        for layer in (self.conv, self.depthwise, self.reduce):
            _kaiming(layer)
        nn.init.normal_(self.output.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.activation(self.conv(x))
        x = self.activation(self.depthwise(x))
        x = self.activation(self.reduce(x))
        return self.output(x)
