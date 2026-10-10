"""P1 组件消融：固定 1 秒几何，按独立模块种子保持共有权重一致。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from resp_train.crd.blocks import BidirectionalMamba2Block
from resp_train.crd.initialization import module_seed
from resp_train.data.patch_aligned_tf import frozen_h_metadata
from resp_train.models.patch_aligned_tf_mamba import (
    ContinuousPatchEncoder, LocalTFAttention, PatchTFConfig, PositiveOverlapAdd, ScaleTimePad, _finite,
)


@dataclass(frozen=True)
class Arm:
    name: str
    band: str = "W-full"
    readout: str = "attention"
    fusion: str = "film"
    placement: str = "pre"
    mamba: bool = True
    pooling: str = "attention"
    group: str = "core"


ARMS = {
    "M0": Arm("P1-Full"),
    "M1": Arm("P1-NoTF", band="none", readout="none", fusion="none"),
    "M2": Arm("P1-H", band="H"),
    "M3": Arm("P1-Full-MeanTF", readout="mean"),
    "M4": Arm("P1-Full-Add", fusion="add"),
    "M5": Arm("P1-Full-PostFiLM", placement="post"),
    "M6": Arm("P1-Full-NoMamba", mamba=False),
    "S1": Arm("P1-L", band="L", group="supplementary"),
    "S2": Arm("P1-H-MeanTF", band="H", readout="mean", group="supplementary"),
    "S3": Arm("P1-Full-UniformPatchPool", pooling="uniform", group="supplementary"),
}


def architecture():
    return PatchTFConfig(patch_samples=100)


def band_indices(frequencies, band):
    values = np.asarray(frequencies, dtype=np.float64)
    if (values.shape != (97,) or not np.isfinite(values).all() or np.any(values <= 0)
            or np.any(np.diff(values) < 0)):
        raise ValueError("W-full 需要按实际频率升序排列的 97-scale 元数据")
    if band == "W-full":
        indices = np.arange(97)
    elif band == "H":
        indices = np.flatnonzero((values > .8) & (values <= 8))
    elif band == "L":
        indices = np.flatnonzero(values <= .8)
    else:
        raise ValueError("未知条件频带")
    if len(indices) != {"W-full": 97, "H": 41, "L": 56}[band]:
        raise ValueError("冻结 W 频带的尺度数量不符")
    return indices


def frequency_metadata(cache_root):
    # 复用冻结 manifest/频率文件 SHA 验证，再读取完整频率向量。
    high, indices = frozen_h_metadata(cache_root)
    values = np.load(Path(cache_root) / "w_frequencies_hz.npy", allow_pickle=False)
    band_indices(values, "W-full")
    if not np.array_equal(values[indices], high):
        raise ValueError("频率文件在校验后发生变化")
    return values.copy()


def condition_encoder(cfg, seed):
    with module_seed(seed, "p1_components.condition_encoder"):
        c = cfg.condition_channels
        return nn.Sequential(
            ScaleTimePad(), nn.Conv2d(1, c, 3), nn.GroupNorm(math.gcd(8, c), c), nn.GELU(),
            ScaleTimePad(), nn.Conv2d(c, c, 3, groups=c), nn.GELU(), nn.Conv2d(c, c, 1),
        )


class BandAttention(LocalTFAttention):
    """复用分块注意力数学实现；尺度坐标来自所选表示的真实频率。"""

    def __init__(self, cfg, frequencies, seed):
        nn.Module.__init__(self)
        self.config, self.heads = cfg, cfg.heads
        self.head_dim = cfg.dimension // cfg.heads
        self.encoder = condition_encoder(cfg, seed)
        with module_seed(seed, "p1_components.attention"):
            self.query = nn.Linear(cfg.dimension, cfg.dimension)
            self.key = nn.Linear(cfg.condition_channels, cfg.dimension)
            self.value = nn.Linear(cfg.condition_channels, cfg.dimension)
            self.output = nn.Linear(cfg.dimension, cfg.dimension)
            self.frequency_bias = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, cfg.heads))
            self.time_bias = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, cfg.heads))
        self.register_buffer("frequencies_hz", torch.as_tensor(frequencies, dtype=torch.float64).clone())
        self.register_buffer("relative_seconds", torch.tensor([-.25, .25]))


class MeanTF(nn.Module):
    """对每个物理对齐区域的 F×2 特征均匀聚合，再作线性映射。"""

    def __init__(self, cfg, frequencies, seed):
        super().__init__()
        self.config = cfg
        self.encoder = condition_encoder(cfg, seed)
        self.register_buffer("frequencies_hz", torch.as_tensor(frequencies, dtype=torch.float64).clone())
        with module_seed(seed, "p1_components.mean_projection"):
            self.projection = nn.Linear(cfg.condition_channels, cfg.dimension)

    def forward(self, tokens, condition):
        features = self.encoder(condition[:, None])
        pooled = features.mean(dim=2).unfold(-1, self.config.cwt_bins_per_patch, 1).mean(dim=-1)
        return self.projection(pooled.transpose(1, 2))


class UniformPatchEncoder(ContinuousPatchEncoder):
    """保留连续特征及相对位置表示，以均匀权重形成 patch token。"""

    def __init__(self, cfg):
        super().__init__(cfg)
        del self.score

    def forward(self, waveform):
        features = self.stem(waveform)
        cfg = self.config
        pooled = features.unfold(-1, cfg.patch_samples, cfg.patch_hop_samples).mean(-1).transpose(1, 2)
        positions = self.position(self.relative_seconds[:, None]).to(features.dtype)
        return self.project(pooled + positions.mean(0))


class BoundedFiLM(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(dimension, dimension), nn.GELU(), nn.Linear(dimension, 2 * dimension))
        nn.init.zeros_(self.projection[-1].weight)
        nn.init.zeros_(self.projection[-1].bias)

    def forward(self, tokens, context):
        gamma, beta = self.projection(context).chunk(2, dim=-1)
        _finite(gamma, "FiLM gamma")
        _finite(beta, "FiLM beta")
        return tokens * (1 + .5 * gamma.tanh()) + .5 * beta.tanh()


class ResidualAdd(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.projection = nn.Linear(dimension, dimension)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, tokens, context):
        return tokens + self.projection(context)


class P1ComponentsModel(nn.Module):
    def __init__(self, arm, seed, frequencies_hz=None, *, cfg=None, mamba_factory=None):
        super().__init__()
        if arm not in ARMS:
            raise ValueError("未知 P1 实验臂")
        self.arm, self.spec = arm, ARMS[arm]
        self.config = cfg or architecture()
        cfg = self.config
        if cfg.patch_samples != 100:
            raise ValueError("P1 组件消融固定 1 秒 patch")
        with module_seed(seed, "p1_components.waveform_encoder"):
            self.waveform_encoder = (UniformPatchEncoder(cfg) if self.spec.pooling == "uniform"
                                     else ContinuousPatchEncoder(cfg))
        self.condition, self.fusion = None, None
        if self.spec.band != "none":
            indices = band_indices(frequencies_hz, self.spec.band)
            frequencies = np.asarray(frequencies_hz)[indices]
            self.condition = (BandAttention(cfg, frequencies, seed) if self.spec.readout == "attention"
                              else MeanTF(cfg, frequencies, seed))
            with module_seed(seed, "p1_components.fusion"):
                self.fusion = ResidualAdd(cfg.dimension) if self.spec.fusion == "add" else BoundedFiLM(cfg.dimension)
        elif frequencies_hz is not None:
            raise ValueError("NoTF 构造不接收 CWT 频率元数据")
        with module_seed(seed, "p1_components.trunk"):
            self.trunk = (nn.Sequential(*[BidirectionalMamba2Block(cfg.dimension,
                dropout=cfg.dropout, mamba_factory=mamba_factory) for _ in range(cfg.layers)])
                if self.spec.mamba else nn.Identity())
        with module_seed(seed, "p1_components.decoder"):
            self.decoder = nn.Sequential(nn.Linear(cfg.dimension, cfg.dimension), nn.GELU(), nn.Linear(cfg.dimension, cfg.patch_samples))
        self.overlap = PositiveOverlapAdd(cfg)

    def forward(self, x, *, tf=None, **kwargs):
        cfg = self.config
        if x.ndim != 3 or x.shape[1:] != (1, cfg.window_samples) or x.shape[0] < 1 or not x.is_floating_point():
            raise ValueError("P1 输入要求浮点 (B,1,18000)")
        _finite(x, "BCG")
        if self.condition is None:
            if tf:
                raise ValueError("NoTF 不接收时频条件")
        else:
            if tf is None or set(tf) != {"w"} or tf["w"].shape != (len(x), len(self.condition.frequencies_hz), cfg.cwt_frames):
                raise ValueError("时频条件与实验臂的频带不一致")
            if tf["w"].device != x.device or not tf["w"].is_floating_point():
                raise ValueError("时频条件须为同设备浮点张量")
            _finite(tf["w"], "CWT")
        z = self.waveform_encoder(x)
        if self.condition is None:
            latent = self.trunk(z)
        else:
            # Pre/Post 均由原始 z 读取条件，位置消融只移动融合算子。
            context = self.condition(z, tf["w"])
            latent = (self.fusion(self.trunk(z), context) if self.spec.placement == "post"
                      else self.trunk(self.fusion(z, context)))
        output = self.overlap(self.decoder(latent))
        _finite(output, "P1 重建输出")
        return {"waveform": output}


def contract(arm):
    return {"arm": arm, **asdict(ARMS[arm])}


def build_model(cfg, *, mamba_factory=None):
    value = dict(cfg.model.p1_components)
    arm = value["arm"]
    if value != contract(arm) or cfg.model.name != "p1_components_v1":
        raise ValueError("P1 模型合同不符")
    frequencies = None if ARMS[arm].band == "none" else frequency_metadata(cfg.data.tf_cache_path)
    return P1ComponentsModel(arm, int(cfg.model.initialization_seed), frequencies,
        cfg=PatchTFConfig(**dict(cfg.model.architecture)), mamba_factory=mamba_factory)
