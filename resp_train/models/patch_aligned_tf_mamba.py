"""物理时间对齐的 H-CWT 条件化 patch 波形重建。"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from resp_train.crd.blocks import BidirectionalMamba2Block
from resp_train.crd.initialization import module_seed


@dataclass(frozen=True)
class PatchTFConfig:
    dimension: int = 96
    waveform_channels: int = 64
    condition_channels: int = 32
    heads: int = 4
    layers: int = 6
    dropout: float = 0.1
    window_epsilon: float = 1e-3

    def __post_init__(self):
        for name in ("dimension", "waveform_channels", "condition_channels", "heads", "layers"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} 必须为正整数")
        if self.dimension % self.heads or (2 * self.dimension) % 32:
            raise ValueError("D 必须整除注意力头数，且 2D 必须为 Mamba headdim=32 的整数倍")
        if not 0 <= self.dropout < 1 or not 0 < self.window_epsilon <= 1:
            raise ValueError("dropout/window_epsilon 超出范围")


def _finite(value: torch.Tensor, label: str):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{label} 包含 NaN/Inf")


def _frequency_tensor(frequencies: Sequence[float]) -> torch.Tensor:
    value = torch.as_tensor(frequencies, dtype=torch.float64).detach().clone()
    if value.shape != (41,):
        raise ValueError("H-CWT 需要 41 个实际映射频率")
    _finite(value, "H-CWT 频率")
    if not bool(((value > .8) & (value <= 8)).all()) or not bool((value[1:] >= value[:-1]).all()):
        raise ValueError("H-CWT 频率须在 (0.8,8] Hz 内并按特征行顺序升序排列")
    return value


def h_cwt_frequencies() -> np.ndarray:
    """沿用原生 Morlet 网格的实际映射；导入模型时不构造 CWT。"""
    from resp_train.crd.tf_v1_features import morlet_scales_and_frequencies

    _, actual = morlet_scales_and_frequencies(12, 97)
    selected = np.sort(actual[(actual > .8) & (actual <= 8)], kind="stable").copy()
    _frequency_tensor(selected)
    return selected


def h_cwt_features(waveform: np.ndarray | torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """返回 log1p 幅值的连续 50 点均值与相应的实际频率。"""
    from resp_train.crd.tf_v1_features import cwt_magnitude_features

    values, actual = cwt_magnitude_features(waveform)
    selected = (actual > .8) & (actual <= 8)
    frequencies = np.ascontiguousarray(actual[selected])
    _frequency_tensor(frequencies)
    result = np.ascontiguousarray(values[selected])
    if result.shape != (41, 360) or not np.isfinite(result).all():
        raise ValueError("H-CWT 特征 shape/finite 检查失败")
    if not np.array_equal(frequencies, h_cwt_frequencies()):
        raise ValueError("CWT 返回频率与原生映射不一致")
    return result, frequencies


class ChannelNorm(nn.Module):
    """逐时间/时频位置归一化通道，保持局部特征的时间感受野。"""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, value):
        return self.norm(value.movedim(1, -1)).movedim(-1, 1)


class LocalResidual(nn.Module):
    def __init__(self, channels: int, kernel: int):
        super().__init__()
        self.net = nn.Sequential(
            ChannelNorm(channels),
            nn.Conv1d(channels, channels, kernel, padding=kernel // 2, groups=channels),
            nn.GELU(), nn.Conv1d(channels, channels, 1),
        )

    def forward(self, value):
        return value + self.net(value)


class ContinuousPatchEncoder(nn.Module):
    def __init__(self, channels: int, dimension: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.GELU(), LocalResidual(32, 5),
            nn.Conv1d(32, channels, 5, padding=2), nn.GELU(), LocalResidual(channels, 3),
        )
        self.position = nn.Sequential(nn.Linear(1, channels), nn.GELU(), nn.Linear(channels, channels))
        self.score = nn.Linear(channels, 1)
        self.project = nn.Linear(channels, dimension)
        self.register_buffer("relative_seconds", (torch.arange(200).float() - 99.5) / 100)

    def forward(self, waveform):
        features = self.stem(waveform)
        # 同一时间位置共享连续卷积结果；仅池化窗口发生重叠。
        windows = features.unfold(-1, 200, 50).permute(0, 2, 3, 1)
        positioned = windows + self.position(self.relative_seconds[:, None]).to(windows.dtype)
        weights = self.score(positioned).float().softmax(dim=2).to(positioned.dtype)
        return self.project((weights * positioned).sum(dim=2))


class LocalTFAttention(nn.Module):
    def __init__(self, config: PatchTFConfig, frequencies: Sequence[float]):
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.dimension // config.heads
        c = config.condition_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1), ChannelNorm(c), nn.GELU(),
            nn.Conv2d(c, c, 3, padding=1, groups=c), nn.GELU(), nn.Conv2d(c, c, 1),
        )
        self.query = nn.Linear(config.dimension, config.dimension)
        self.key = nn.Linear(c, config.dimension)
        self.value = nn.Linear(c, config.dimension)
        self.output = nn.Linear(config.dimension, config.dimension)
        self.frequency_bias = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, self.heads))
        self.time_bias = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, self.heads))
        self.register_buffer("frequencies_hz", _frequency_tensor(frequencies))
        self.register_buffer("relative_seconds", torch.tensor([-.75, -.25, .25, .75]))

    @staticmethod
    def local_cells(features):
        # B,C,F,T → B,patch,F,4,C；只建立实际需要的局部读取。
        return features.unfold(-1, 4, 1).permute(0, 3, 2, 4, 1)

    def forward(self, tokens, condition):
        features = self.encoder(condition[:, None])
        # 在完整特征图上只投影一次，随后对 K/V 做局部视图展开。
        cells = features.movedim(1, -1)
        key = self.local_cells(self.key(cells).movedim(-1, 1))
        value = self.local_cells(self.value(cells).movedim(-1, 1))
        b, n, d = tokens.shape
        key = key.reshape(b, n, 164, self.heads, self.head_dim)
        value = value.reshape(b, n, 164, self.heads, self.head_dim)
        query = self.query(tokens).reshape(b, n, self.heads, self.head_dim)
        # logits 与 softmax 采用 float32，避免混合精度过早舍入。
        logits = torch.einsum("bnhd,bnkhd->bnhk", query.float(), key.float()) / math.sqrt(self.head_dim)
        frequency_bias = self.frequency_bias(self.frequencies_hz.log().float()[:, None])
        time_bias = self.time_bias(self.relative_seconds[:, None])
        bias = (frequency_bias[:, None, :] + time_bias[None, :, :]).reshape(164, self.heads).T
        attention = (logits + bias.float()[None, None]).softmax(dim=-1)
        context = torch.einsum("bnhk,bnkhd->bnhd", attention, value.float()).reshape(b, n, d)
        return self.output(context.to(tokens.dtype))


class PositiveOverlapAdd(nn.Module):
    def __init__(self, epsilon: float):
        super().__init__()
        weight = torch.hann_window(200, periodic=False).clamp_min(epsilon)
        self.register_buffer("weight", weight)
        denominator = self.fold(weight[None, :, None].expand(1, 200, 357))
        if not bool((denominator > 0).all()):
            raise RuntimeError("合成权重未完整覆盖输出")
        self.register_buffer("denominator", denominator)

    @staticmethod
    def fold(value):
        return F.fold(value, (1, 18000), (1, 200), stride=(1, 50)).flatten(2)

    def forward(self, patches):
        if patches.ndim != 3 or patches.shape[1:] != (357, 200):
            raise ValueError("局部输出必须为 (B,357,200)")
        _finite(patches, "局部波形")
        return self.fold(patches.float().transpose(1, 2) * self.weight[None, :, None]) / self.denominator


class PatchAlignedTFMamba(nn.Module):
    """100-Hz/180-s BCG 与 (41,360) H-CWT → 100-Hz 呼吸波形。"""

    def __init__(self, frequencies_hz: Sequence[float], *, config: PatchTFConfig | None = None,
                 initialization_seed: int = 20261004,
                 mamba_factory: Callable[..., nn.Module] | None = None):
        super().__init__()
        self.config = config or PatchTFConfig()
        cfg = self.config
        _frequency_tensor(frequencies_hz)
        with module_seed(initialization_seed, "patch_aligned_tf_mamba_v1"):
            self.waveform_encoder = ContinuousPatchEncoder(cfg.waveform_channels, cfg.dimension)
            self.condition = LocalTFAttention(cfg, frequencies_hz)
            self.film = nn.Sequential(nn.Linear(cfg.dimension, cfg.dimension), nn.GELU(),
                                      nn.Linear(cfg.dimension, 2 * cfg.dimension))
            nn.init.zeros_(self.film[-1].weight)
            nn.init.zeros_(self.film[-1].bias)
            self.trunk = nn.Sequential(*[
                BidirectionalMamba2Block(cfg.dimension, dropout=cfg.dropout, mamba_factory=mamba_factory)
                for _ in range(cfg.layers)
            ])
            self.decoder = nn.Sequential(nn.Linear(cfg.dimension, cfg.dimension), nn.GELU(),
                                         nn.Linear(cfg.dimension, 200))
            self.overlap = PositiveOverlapAdd(cfg.window_epsilon)

    def forward(self, x: torch.Tensor, *, tf: Mapping[str, torch.Tensor] | None = None,
                **_: Any) -> dict[str, torch.Tensor]:
        if x.ndim != 3 or x.shape[1:] != (1, 18000) or x.shape[0] < 1:
            raise ValueError("波形输入必须为 (B,1,18000)")
        if tf is None or set(tf) != {"w"} or tf["w"].shape != (x.shape[0], 41, 360):
            raise ValueError("条件输入必须为同 batch 的 tf['w']: (B,41,360)")
        if not x.is_floating_point() or not tf["w"].is_floating_point() or x.device != tf["w"].device:
            raise ValueError("波形和条件须为同设备浮点张量")
        _finite(x, "BCG 输入")
        _finite(tf["w"], "H-CWT 输入")
        z = self.waveform_encoder(x)
        context = self.condition(z, tf["w"])
        gamma, beta = self.film(context).chunk(2, dim=-1)
        # tanh 会掩盖 Inf，因此在有界调制前显式检查。
        _finite(gamma, "FiLM gamma")
        _finite(beta, "FiLM beta")
        u = z * (1 + .5 * gamma.tanh()) + .5 * beta.tanh()
        waveform = self.overlap(self.decoder(self.trunk(u)))
        _finite(waveform, "重建输出")
        return {"waveform": waveform}


def build_patch_aligned_tf_mamba(cfg: Any, *, mamba_factory=None):
    """独立模型配置入口；实际频率通过已有原生映射解析。"""
    if cfg.model.name != "patch_aligned_tf_mamba":
        raise ValueError("模型名称不匹配")
    options = dict(cfg.model.get("patch_aligned_tf_mamba", {}))
    return PatchAlignedTFMamba(h_cwt_frequencies(), config=PatchTFConfig(**options),
                               initialization_seed=int(cfg.model.initialization_seed),
                               mamba_factory=mamba_factory)
