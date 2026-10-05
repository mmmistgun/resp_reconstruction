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
from torch.utils.checkpoint import checkpoint

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
    sample_rate: int = 100
    window_samples: int = 18000
    patch_samples: int = 200
    patch_hop_samples: int = 50
    cwt_pool_samples: int = 50
    stem_dilations: tuple[int, ...] = (1, 2, 4, 8, 16)
    patch_chunk_size: int = 16
    checkpoint_local: bool = True

    def __post_init__(self):
        for name in ("dimension", "waveform_channels", "condition_channels", "heads", "layers",
                     "sample_rate", "window_samples", "patch_samples", "patch_hop_samples",
                     "cwt_pool_samples", "patch_chunk_size"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} 必须为正整数")
        if self.dimension % self.heads or (2 * self.dimension) % 32:
            raise ValueError("D 必须是注意力头数的整数倍，且 2D 必须为 Mamba headdim=32 的整数倍")
        if not 0 <= self.dropout < 1 or not 0 < self.window_epsilon <= 1:
            raise ValueError("dropout/window_epsilon 超出范围")
        if (self.sample_rate, self.window_samples, self.cwt_pool_samples) != (100, 18000, 50):
            raise ValueError("冻结 H-CWT 表示要求 100 Hz、18000 点和 50 点池化")
        if (self.patch_hop_samples != self.cwt_pool_samples
                or self.patch_samples % self.cwt_pool_samples
                or self.patch_samples > self.window_samples
                or (self.window_samples - self.patch_samples) % self.patch_hop_samples):
            raise ValueError("patch 几何须与 CWT 时间格整除对齐并完整覆盖输入")
        if not self.stem_dilations or any(type(d) is not int or d < 1 for d in self.stem_dilations):
            raise ValueError("stem_dilations 必须为正整数序列")
        object.__setattr__(self, "stem_dilations", tuple(self.stem_dilations))
        if type(self.checkpoint_local) is not bool:
            raise ValueError("checkpoint_local 必须为 bool")

    @property
    def patch_count(self):
        return (self.window_samples - self.patch_samples) // self.patch_hop_samples + 1

    @property
    def cwt_bins_per_patch(self):
        return self.patch_samples // self.cwt_pool_samples

    @property
    def cwt_frames(self):
        return self.window_samples // self.cwt_pool_samples

    @property
    def stem_receptive_field(self):
        return 17 + 2 * sum(self.stem_dilations)


def _local_call(function, enabled, *values):
    # checkpoint 仅保留连续 map 的共享视图，反传按块重算局部激活。
    if enabled and torch.is_grad_enabled():
        return checkpoint(function, *values, use_reentrant=False, preserve_rng_state=False)
    return function(*values)


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
    def __init__(self, channels: int, kernel: int, dilation: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            ChannelNorm(channels),
            nn.Conv1d(channels, channels, kernel, padding=(kernel // 2) * dilation,
                      dilation=dilation, groups=channels, padding_mode="reflect"),
            nn.GELU(), nn.Conv1d(channels, channels, 1),
        )

    def forward(self, value):
        return value + self.net(value)


class ContinuousPatchEncoder(nn.Module):
    def __init__(self, config: PatchTFConfig):
        super().__init__()
        self.config = config
        channels, dimension = config.waveform_channels, config.dimension
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3, padding_mode="reflect"), nn.GELU(), LocalResidual(32, 5),
            nn.Conv1d(32, channels, 5, padding=2, padding_mode="reflect"), nn.GELU(), LocalResidual(channels, 3),
            *[LocalResidual(channels, 3, d) for d in config.stem_dilations],
        )
        self.position = nn.Sequential(nn.Linear(1, channels), nn.GELU(), nn.Linear(channels, channels))
        self.score = nn.Linear(channels, 1)
        self.project = nn.Linear(channels, dimension)
        self.register_buffer("relative_seconds",
            (torch.arange(config.patch_samples).float() - (config.patch_samples - 1) / 2) / config.sample_rate)

    def _pool_chunk(self, features, scores, position, position_scores):
        p, s = self.config.patch_samples, self.config.patch_hop_samples
        windows = features.unfold(-1, p, s)  # B,C,N,P，共享 storage。
        logits = scores.unfold(-1, p, s).float() + position_scores.float()
        weights = logits.softmax(-1)
        # 两项分开聚合，避免构造 B,N,P,C 的 positioned tensor。
        # 分块同时限制 einsum 后端可能产生的临时连续副本。
        pooled = torch.einsum("bnp,bcnp->bnc", weights.to(features.dtype), windows)
        pooled = pooled + torch.einsum("bnp,pc->bnc", weights.to(position.dtype), position)
        return pooled

    def forward(self, waveform):
        features = self.stem(waveform)
        position = self.position(self.relative_seconds[:, None]).to(features.dtype)
        scores = self.score(features.transpose(1, 2)).squeeze(-1)
        position_scores = F.linear(position, self.score.weight).squeeze(-1)
        cfg = self.config
        chunks = []
        for start in range(0, cfg.patch_count, cfg.patch_chunk_size):
            stop = min(start + cfg.patch_chunk_size, cfg.patch_count)
            lo, hi = start * cfg.patch_hop_samples, (stop - 1) * cfg.patch_hop_samples + cfg.patch_samples
            chunks.append(_local_call(self._pool_chunk, cfg.checkpoint_local,
                features[..., lo:hi], scores[..., lo:hi], position, position_scores))
        return self.project(torch.cat(chunks, dim=1))


class ScaleTimePad(nn.Module):
    """时间反射、频率端点延拓；分别表达两个轴的边界语义。"""

    def forward(self, value):
        return F.pad(F.pad(value, (1, 1, 0, 0), mode="reflect"), (0, 0, 1, 1), mode="replicate")


class LocalTFAttention(nn.Module):
    def __init__(self, config: PatchTFConfig, frequencies: Sequence[float]):
        super().__init__()
        self.config = config
        self.heads = config.heads
        self.head_dim = config.dimension // config.heads
        c = config.condition_channels
        self.encoder = nn.Sequential(
            ScaleTimePad(), nn.Conv2d(1, c, 3), nn.GroupNorm(math.gcd(8, c), c), nn.GELU(),
            ScaleTimePad(), nn.Conv2d(c, c, 3, groups=c), nn.GELU(), nn.Conv2d(c, c, 1),
        )
        self.query = nn.Linear(config.dimension, config.dimension)
        self.key = nn.Linear(c, config.dimension)
        self.value = nn.Linear(c, config.dimension)
        self.output = nn.Linear(config.dimension, config.dimension)
        self.frequency_bias = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, self.heads))
        self.time_bias = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, self.heads))
        self.register_buffer("frequencies_hz", _frequency_tensor(frequencies))
        self.register_buffer("relative_seconds",
            (torch.arange(config.cwt_bins_per_patch).float() * config.cwt_pool_samples
             + (config.cwt_pool_samples - 1) / 2 - (config.patch_samples - 1) / 2) / config.sample_rate)

    @staticmethod
    def local_cells(features, bins=4):
        # 供几何检查使用的共享视图，主注意力逐时间偏移读取。
        return features.unfold(-1, bins, 1).permute(0, 3, 2, 4, 1)

    def _attend_chunk(self, query, key, value, bias):
        n = query.shape[1]
        # 禁用内部 autocast，保证点积与概率累计使用 float32。
        with torch.autocast(device_type=query.device.type, enabled=False):
            q = query.float()
            logits = torch.stack([
                torch.einsum("bnhd,bfnhd->bnhf", q, key[:, :, r:r + n].float())
                for r in range(self.config.cwt_bins_per_patch)
            ], dim=-1) / math.sqrt(self.head_dim)
            logits = logits + bias.float()[None, None]
            attention = logits.flatten(-2).softmax(-1).view_as(logits)
            context = sum(torch.einsum("bnhf,bfnhd->bnhd", attention[..., r],
                          value[:, :, r:r + n].float())
                          for r in range(self.config.cwt_bins_per_patch))
        return context.flatten(-2)

    def forward(self, tokens, condition):
        features = self.encoder(condition[:, None])
        # 在完整特征图上只投影一次，随后对 K/V 做局部视图展开。
        cells = features.movedim(1, -1)
        b, n, _ = tokens.shape
        key = self.key(cells).unflatten(-1, (self.heads, self.head_dim))
        value = self.value(cells).unflatten(-1, (self.heads, self.head_dim))
        query = self.query(tokens).reshape(b, n, self.heads, self.head_dim)
        frequency_bias = self.frequency_bias(self.frequencies_hz.log().float()[:, None])
        time_bias = self.time_bias(self.relative_seconds[:, None])
        bias = (frequency_bias[:, None, :] + time_bias[None, :, :]).permute(2, 0, 1)
        contexts = []
        for start in range(0, n, self.config.patch_chunk_size):
            stop = min(start + self.config.patch_chunk_size, n)
            end = stop + self.config.cwt_bins_per_patch - 1
            contexts.append(_local_call(self._attend_chunk, self.config.checkpoint_local,
                query[:, start:stop], key[:, :, start:end], value[:, :, start:end], bias))
        context = torch.cat(contexts, dim=1)
        return self.output(context.to(tokens.dtype))


class PositiveOverlapAdd(nn.Module):
    def __init__(self, config: PatchTFConfig):
        super().__init__()
        self.config = config
        weight = torch.hann_window(config.patch_samples, periodic=False).clamp_min(config.window_epsilon)
        self.register_buffer("weight", weight)
        denominator = self.fold(weight[None, :, None].expand(1, config.patch_samples, config.patch_count))
        if not bool((denominator > 0).all()):
            raise RuntimeError("合成权重未完整覆盖输出")
        self.register_buffer("denominator", denominator)

    def fold(self, value):
        cfg = self.config
        return F.fold(value, (1, cfg.window_samples), (1, cfg.patch_samples),
                      stride=(1, cfg.patch_hop_samples)).flatten(2)

    def forward(self, patches):
        if patches.ndim != 3 or patches.shape[1:] != (self.config.patch_count, self.config.patch_samples):
            raise ValueError("局部输出与 patch 几何不符")
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
            self.waveform_encoder = ContinuousPatchEncoder(cfg)
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
                                         nn.Linear(cfg.dimension, cfg.patch_samples))
            self.overlap = PositiveOverlapAdd(cfg)

    def forward(self, x: torch.Tensor, *, tf: Mapping[str, torch.Tensor] | None = None,
                **_: Any) -> dict[str, torch.Tensor]:
        if x.ndim != 3 or x.shape[1:] != (1, self.config.window_samples) or x.shape[0] < 1:
            raise ValueError("波形输入必须与配置窗口一致")
        if tf is None or set(tf) != {"w"} or tf["w"].shape != (
                x.shape[0], len(self.condition.frequencies_hz), self.config.cwt_frames):
            raise ValueError("条件输入必须为同 batch 的冻结 H-CWT 网格")
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
    """训练入口从冻结缓存元数据构造坐标，不执行 CWT 变换。"""
    from resp_train.data.patch_aligned_tf import frozen_h_metadata

    if cfg.model.name != "patch_aligned_tf_mamba":
        raise ValueError("模型名称不匹配")
    options = dict(cfg.model.get("patch_aligned_tf_mamba", {}))
    if list(cfg.model.get("tf_representations", [])) != ["w"]:
        raise ValueError("模型配置要求 tf_representations=[w]")
    frequencies, _ = frozen_h_metadata(cfg.data.tf_cache_path)
    return PatchAlignedTFMamba(frequencies, config=PatchTFConfig(**options),
                               initialization_seed=int(cfg.model.initialization_seed),
                               mamba_factory=mamba_factory)
