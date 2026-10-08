"""M4-v2：物理对齐 cross-attention 的单投影残差与组件对照。"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from resp_train.crd.blocks import BidirectionalMamba2Block, CustomRMSNorm
from resp_train.crd.initialization import module_seed, named_subseed
from resp_train.models.patch_aligned_tf_mamba import (
    ContinuousPatchEncoder, LocalTFAttention, PatchTFConfig, PositiveOverlapAdd, _finite, _local_call,
)
from scripts.p1_components_model import band_indices, condition_encoder, frequency_metadata


@dataclass(frozen=True)
class Arm:
    name: str
    readout: str = "attention"
    mamba: bool = True
    query_norm: str = "identity"
    coordinate_bias: bool = True
    group: str = "core"

    @property
    def band(self):
        return "none" if self.readout == "none" else "W-full"


ARMS = {
    "A0": Arm("M4-v2"),
    "A1": Arm("M4-v2-NoTF", readout="none", coordinate_bias=False),
    "A2": Arm("M4-v2-MeanTF-Residual", readout="mean", coordinate_bias=False),
    "A3": Arm("M4-v2-NoMamba", mamba=False),
    "S1": Arm("M4-v2-QueryRMSNorm", query_norm="rmsnorm", group="supplementary"),
    "S2": Arm("M4-v2-ContentAttention", coordinate_bias=False, group="supplementary"),
}


def architecture():
    return PatchTFConfig(patch_samples=100)


def encoder(cfg, seed):
    # 独立 namespace 的派生 seed 复用既有 scale-time CNN 构造及 GroupNorm 语义。
    # condition_encoder 内部自行管理 RNG，共有 encoder 在所有条件臂中逐张量一致。
    return condition_encoder(cfg, named_subseed(seed, "m4_residual.condition_encoder"))


class ResidualAttention(LocalTFAttention):
    """输出 delta；唯一 D→D 输出投影零初始化，Q 的归一化不触及 residual z。"""

    def __init__(self, cfg, frequencies, seed, spec):
        nn.Module.__init__(self)
        self.config, self.heads = cfg, cfg.heads
        self.head_dim = cfg.dimension // cfg.heads
        self.encoder = encoder(cfg, seed)
        with module_seed(seed, "m4_residual.query_key_value"):
            self.query = nn.Linear(cfg.dimension, cfg.dimension)
            self.key = nn.Linear(cfg.condition_channels, cfg.dimension)
            self.value = nn.Linear(cfg.condition_channels, cfg.dimension)
        self.query_norm = CustomRMSNorm(cfg.dimension, eps=1e-5) if spec.query_norm == "rmsnorm" else nn.Identity()
        with module_seed(seed, "m4_residual.output"):
            self.output = nn.Linear(cfg.dimension, cfg.dimension)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.frequency_bias, self.time_bias = None, None
        if spec.coordinate_bias:
            with module_seed(seed, "m4_residual.coordinates"):
                self.frequency_bias = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, cfg.heads))
                self.time_bias = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, cfg.heads))
        self.register_buffer("frequencies_hz", torch.as_tensor(frequencies, dtype=torch.float64).clone())
        self.register_buffer("relative_seconds", torch.tensor([-.25, .25]))

    def forward(self, tokens, condition):
        features = self.encoder(condition[:, None])
        cells = features.movedim(1, -1)
        b, n, _ = tokens.shape
        key = self.key(cells).unflatten(-1, (self.heads, self.head_dim))
        value = self.value(cells).unflatten(-1, (self.heads, self.head_dim))
        query = self.query(self.query_norm(tokens)).reshape(b, n, self.heads, self.head_dim)
        if self.frequency_bias is None:
            bias = tokens.new_zeros((self.heads, len(self.frequencies_hz), self.config.cwt_bins_per_patch))
        else:
            bf = self.frequency_bias(self.frequencies_hz.log().float()[:, None])
            bt = self.time_bias(self.relative_seconds[:, None])
            bias = (bf[:, None, :] + bt[None, :, :]).permute(2, 0, 1)
        contexts = []
        for start in range(0, n, self.config.patch_chunk_size):
            stop = min(start + self.config.patch_chunk_size, n)
            end = stop + self.config.cwt_bins_per_patch - 1
            contexts.append(_local_call(self._attend_chunk, self.config.checkpoint_local,
                query[:, start:stop], key[:, :, start:end], value[:, :, start:end], bias))
        return self.output(torch.cat(contexts, dim=1).to(tokens.dtype))


class MeanResidual(nn.Module):
    """同一 F×2 编码区域均值，经一个零初始化 C→D 投影生成 delta。"""

    def __init__(self, cfg, frequencies, seed):
        super().__init__()
        self.config = cfg
        self.encoder = encoder(cfg, seed)
        self.register_buffer("frequencies_hz", torch.as_tensor(frequencies, dtype=torch.float64).clone())
        with module_seed(seed, "m4_residual.mean_output"):
            self.output = nn.Linear(cfg.condition_channels, cfg.dimension)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, tokens, condition):
        features = self.encoder(condition[:, None])
        pooled = features.mean(2).unfold(-1, self.config.cwt_bins_per_patch, 1).mean(-1)
        return self.output(pooled.transpose(1, 2).to(tokens.dtype))


class M4ResidualModel(nn.Module):
    def __init__(self, arm, seed, frequencies_hz=None, *, cfg=None, mamba_factory=None):
        super().__init__()
        if arm not in ARMS:
            raise ValueError("未知 M4-v2 实验臂")
        self.arm, self.spec = arm, ARMS[arm]
        self.config = cfg or architecture()
        cfg = self.config
        if cfg.patch_samples != 100:
            raise ValueError("M4-v2 固定 1 秒 patch")
        with module_seed(seed, "m4_residual.waveform_encoder"):
            self.waveform_encoder = ContinuousPatchEncoder(cfg)
        self.condition = None
        if self.spec.band != "none":
            band_indices(frequencies_hz, "W-full")
            frequencies = np.asarray(frequencies_hz, dtype=np.float64)
            self.condition = (MeanResidual(cfg, frequencies, seed) if self.spec.readout == "mean"
                              else ResidualAttention(cfg, frequencies, seed, self.spec))
        elif frequencies_hz is not None:
            raise ValueError("NoTF 构造不接收 CWT 频率元数据")
        with module_seed(seed, "m4_residual.trunk"):
            self.trunk = (nn.Sequential(*[BidirectionalMamba2Block(cfg.dimension,
                dropout=cfg.dropout, mamba_factory=mamba_factory) for _ in range(cfg.layers)])
                if self.spec.mamba else nn.Identity())
        with module_seed(seed, "m4_residual.decoder"):
            self.decoder = nn.Sequential(nn.Linear(cfg.dimension, cfg.dimension), nn.GELU(), nn.Linear(cfg.dimension, cfg.patch_samples))
        self.overlap = PositiveOverlapAdd(cfg)

    def forward(self, x, *, tf=None, **kwargs):
        cfg = self.config
        if x.ndim != 3 or x.shape[1:] != (1, cfg.window_samples) or x.shape[0] < 1 or not x.is_floating_point():
            raise ValueError("M4-v2 输入要求浮点 (B,1,18000)")
        _finite(x, "BCG")
        if self.condition is None:
            if tf:
                raise ValueError("NoTF 不接收时频条件")
        else:
            if tf is None or set(tf) != {"w"} or tf["w"].shape != (len(x), 97, cfg.cwt_frames):
                raise ValueError("M4-v2 条件要求完整 W-full 网格")
            if tf["w"].device != x.device or not tf["w"].is_floating_point():
                raise ValueError("时频条件须为同设备浮点张量")
            _finite(tf["w"], "CWT")
        z = self.waveform_encoder(x)
        u = z if self.condition is None else z + self.condition(z, tf["w"])
        output = self.overlap(self.decoder(self.trunk(u)))
        _finite(output, "M4-v2 重建输出")
        return {"waveform": output}


def contract(arm):
    return {"arm": arm, **asdict(ARMS[arm]), "band": ARMS[arm].band,
        "residual_output": "zero_initialized_single_affine", "query_rmsnorm_eps": 1e-5}


def build_model(cfg, *, mamba_factory=None):
    value = dict(cfg.model.m4_residual)
    arm = value["arm"]
    if value != contract(arm) or cfg.model.name != "m4_residual_v2":
        raise ValueError("M4-v2 模型合同不符")
    frequencies = None if ARMS[arm].band == "none" else frequency_metadata(cfg.data.tf_cache_path)
    return M4ResidualModel(arm, int(cfg.model.initialization_seed), frequencies,
        cfg=PatchTFConfig(**dict(cfg.model.architecture)), mamba_factory=mamba_factory)
