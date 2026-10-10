"""原始 P1-M4 的组件消融；保持 P1 双投影及共有模块初始化合同。"""
from dataclasses import asdict

import numpy as np
import torch
from torch import nn

from resp_train.crd.blocks import BidirectionalMamba2Block
from resp_train.crd.initialization import module_seed
from resp_train.models.patch_aligned_tf_mamba import ContinuousPatchEncoder, PatchTFConfig, PositiveOverlapAdd
from scripts.p1_components_model import (
    Arm, BandAttention, MeanTF, P1ComponentsModel, ResidualAdd, UniformPatchEncoder,
    architecture, band_indices, frequency_metadata,
)


ARMS = {
    "B0": Arm("M4-Full", fusion="add"),
    "B1": Arm("M4-NoTF", band="none", readout="none", fusion="none"),
    "B2": Arm("M4-MeanTF-Add", readout="mean", fusion="add"),
    "B3": Arm("M4-ContentAttention-Add", readout="content", fusion="add"),
    "B4": Arm("M4-H-Add", band="H", fusion="add"),
    "B5": Arm("M4-L-Add", band="L", fusion="add"),
    "B6": Arm("M4-NoMamba-Add", fusion="add", mamba=False),
    "B7": Arm("M4-UniformPatchPool-Add", fusion="add", pooling="uniform"),
}
REFERENCES = {"B0": "M4", "B1": "M1"}
TRAIN_ARMS = tuple(a for a in ARMS if a not in REFERENCES)


class ZeroCoordinateBias(nn.Module):
    """移除坐标网络的参数，复用原注意力的分块及 float32 累计实现。"""
    def __init__(self, heads):
        super().__init__()
        self.heads = heads

    def forward(self, coordinates):
        return coordinates.new_zeros((*coordinates.shape[:-1], self.heads))


class ContentAttention(BandAttention):
    def __init__(self, cfg, frequencies, seed):
        # 先按原 P1 顺序初始化，保证 Q/K/V/output 逐张量相同。
        super().__init__(cfg, frequencies, seed)
        self.frequency_bias = ZeroCoordinateBias(cfg.heads)
        self.time_bias = ZeroCoordinateBias(cfg.heads)


class M4ComponentsModel(P1ComponentsModel):
    """复用冻结 P1 的前向检查、融合位置、解码和 overlap-add。"""
    def __init__(self, arm, seed, frequencies_hz=None, *, cfg=None, mamba_factory=None):
        nn.Module.__init__(self)
        if arm not in ARMS:
            raise ValueError("未知 M4 组件实验臂")
        self.arm, self.spec = arm, ARMS[arm]
        self.config = cfg or architecture()
        cfg = self.config
        if cfg.patch_samples != 100:
            raise ValueError("M4 组件消融固定 1 秒 patch")
        with module_seed(seed, "p1_components.waveform_encoder"):
            self.waveform_encoder = (UniformPatchEncoder(cfg) if self.spec.pooling == "uniform"
                                     else ContinuousPatchEncoder(cfg))
        self.condition, self.fusion = None, None
        if self.spec.band != "none":
            indices = band_indices(frequencies_hz, self.spec.band)
            frequencies = np.asarray(frequencies_hz)[indices]
            reader = {"attention": BandAttention, "content": ContentAttention, "mean": MeanTF}[self.spec.readout]
            self.condition = reader(cfg, frequencies, seed)
            with module_seed(seed, "p1_components.fusion"):
                self.fusion = ResidualAdd(cfg.dimension)
        elif frequencies_hz is not None:
            raise ValueError("NoTF 构造不接收 CWT 频率元数据")
        with module_seed(seed, "p1_components.trunk"):
            self.trunk = (nn.Sequential(*[BidirectionalMamba2Block(cfg.dimension,
                dropout=cfg.dropout, mamba_factory=mamba_factory) for _ in range(cfg.layers)])
                if self.spec.mamba else nn.Identity())
        with module_seed(seed, "p1_components.decoder"):
            self.decoder = nn.Sequential(nn.Linear(cfg.dimension, cfg.dimension), nn.GELU(), nn.Linear(cfg.dimension, cfg.patch_samples))
        self.overlap = PositiveOverlapAdd(cfg)


def contract(arm):
    return {"arm": arm, **asdict(ARMS[arm])}


def build_model(cfg, *, mamba_factory=None):
    value = dict(cfg.model.m4_components)
    arm = value["arm"]
    if value != contract(arm) or cfg.model.name != "m4_components_v1":
        raise ValueError("M4 组件模型合同不符")
    frequencies = None if ARMS[arm].band == "none" else frequency_metadata(cfg.data.tf_cache_path)
    return M4ComponentsModel(arm, int(cfg.model.initialization_seed), frequencies,
        cfg=PatchTFConfig(**dict(cfg.model.architecture)), mamba_factory=mamba_factory)
