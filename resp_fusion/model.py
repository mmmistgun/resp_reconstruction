from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from resp_train.aligned_dual_view.model import AlignedDualViewV1, ModelConfig as ADVConfig
from resp_train.aligned_dual_view.features import CWTBatch
from resp_train.aligned_dual_view.signal import require_finite
from resp_train.crd.initialization import module_seed
from resp_train.crd.spectral_ops import fourier_interpolate

ARMS = {
    "A": ("concat", "pre"), "B": ("concat", "post"),
    "C": ("film", "pre"), "D": ("film", "post"),
    "E": ("attention", "pre"), "F": ("attention", "post"),
}


@dataclass(frozen=True)
class ModelConfig:
    arm: str = "A"
    dimension: int = 64
    depth: int = 6
    waveform_kernel: int = 51
    initialization_seed: int = 20260811

    def __post_init__(self):
        if self.arm not in ARMS:
            raise ValueError("融合 arm 必须为 A..F")
        if (self.dimension, self.depth, self.waveform_kernel) != (64, 6, 51):
            raise ValueError("本轮固定 D64、六层主干、kernel51")
        if type(self.initialization_seed) is not int or self.initialization_seed < 0:
            raise ValueError("initialization_seed 必须为非负整数")


class Fusion(nn.Module):
    """输入为 (B,L,64)；注意力只沿同一时刻的两个视图 token 归一化。"""

    def __init__(self, method):
        super().__init__()
        self.method = method
        if method == "concat":
            self.projection = nn.Linear(128, 64)
            with torch.no_grad():
                self.projection.weight.zero_()
                self.projection.weight[:, :64].copy_(torch.eye(64))
                self.projection.bias.zero_()
        elif method == "film":
            self.gamma = nn.Linear(64, 64)
            self.beta = nn.Linear(64, 64)
            for layer in (self.gamma, self.beta):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
        elif method == "attention":
            # 两个 key 共享的 bias 会在 softmax 中抵消；Q/K/V 统一不带 bias。
            self.query = nn.Linear(64, 64, bias=False)
            self.key = nn.Linear(64, 64, bias=False)
            self.value = nn.Linear(64, 64, bias=False)
            self.output = nn.Linear(64, 64)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)
        else:
            raise ValueError("未知融合方式")

    def attention(self, h, v):
        b, length, _ = h.shape
        tokens = torch.stack((h, v), dim=2)
        q = self.query(h).reshape(b, length, 4, 16)
        k = self.key(tokens).reshape(b, length, 2, 4, 16).transpose(2, 3)
        values = self.value(tokens).reshape(b, length, 2, 4, 16).transpose(2, 3)
        # float32 softmax 稳定化；时间 L 从未并入 key 轴。
        logits = (q.unsqueeze(-2).float() * k.float()).sum(-1) / math.sqrt(16)
        weights = logits.softmax(dim=-1)
        merged = (weights.to(values.dtype).unsqueeze(-1) * values).sum(-2).reshape(b, length, 64)
        return self.output(merged), weights

    def forward(self, h, v):
        if h.ndim != 3 or h.shape != v.shape or h.shape[-1] != 64 or h.numel() == 0:
            raise ValueError("融合要求相同的非空 (B,L,64) 输入")
        if h.device != v.device or not h.is_floating_point() or not v.is_floating_point():
            raise ValueError("融合输入必须为同设备浮点 tensor")
        require_finite("融合波形", h)
        require_finite("融合 CWT", v)
        if self.method == "concat":
            output = self.projection(torch.cat((h, v), dim=-1))
        elif self.method == "film":
            output = h * (1 + 0.5 * torch.tanh(self.gamma(v))) + 0.5 * torch.tanh(self.beta(v))
        else:
            residual, _ = self.attention(h, v)
            output = h + residual
        require_finite("融合输出", output)
        return output


class FusionModel(AlignedDualViewV1):
    def __init__(self, config=None, *, mamba_factory=None):
        cfg = config or ModelConfig()
        # 继承已验证的共同模块构造；各自命名子 seed 保证六组逐 tensor 一致。
        super().__init__(ADVConfig(initialization_seed=cfg.initialization_seed), mamba_factory=mamba_factory)
        self.config = cfg
        self.method, self.position = ARMS[cfg.arm]
        with module_seed(cfg.initialization_seed, "fusion-v1.wave_adapter"):
            self.wave_adapter = nn.Linear(32, 64)
        with module_seed(cfg.initialization_seed, "fusion-v1.cwt_adapter"):
            self.cwt_adapter = nn.Linear(32, 64)
        with module_seed(cfg.initialization_seed, f"fusion-v1.{self.method}"):
            self.fusion = Fusion(self.method)

    def forward(self, x, *, tf=None, cwt=None):
        if tf is not None:
            if cwt is not None or not isinstance(tf, dict) or set(tf) != {"adv_cwt"}:
                raise ValueError("仅允许单一 adv_cwt 输入")
            cwt = CWTBatch(tf["adv_cwt"], self.representation_id)
        if x.ndim != 3 or x.shape[1:] != (1, 18000) or not len(x) or not x.is_floating_point():
            raise ValueError("融合模型输入期望非空浮点 (B,1,18000)")
        require_finite("输入", x)
        if not isinstance(cwt, CWTBatch) or cwt.representation_id != self.representation_id:
            raise ValueError("CWT 表示身份不匹配")
        v = cwt.values
        if v.shape != (len(x), 97, 1800) or v.dtype != torch.float32 or v.device != x.device:
            raise ValueError("CWT 必须为同设备 float32 (B,97,1800)")
        require_finite("CWT", v)
        u = self.wave_adapter(self.downsample(self.waveform_encoder(x)).transpose(1, 2))
        v = self.cwt_adapter(self.scale_projection(v).transpose(1, 2))
        h = self.fusion(u, v) if self.position == "pre" else u
        for block in self.blocks:
            h = block(h.contiguous())
        if self.position == "post":
            h = self.fusion(h, v)
        low = self.readout(h.transpose(1, 2))
        high = fourier_interpolate(low, target_length=18000)
        require_finite("10 Hz prediction", low)
        require_finite("100 Hz prediction", high)
        return {"waveform": high, "waveform_10hz": low}


def module_report(model):
    counts = {name: sum(p.numel() for p in module.parameters()) for name, module in model.named_children()}
    # 一次乘加为 1 MAC；不包含 bias/激活/softmax/FiLM 逐元素计算。
    mac_per_time = {"concat": 128 * 64, "film": 2 * 64 * 64,
                    "attention": 6 * 64 * 64 + 4 * 64}[model.method]
    return {"arm": model.config.arm, "method": model.method, "position": model.position,
            "modules": counts, "total_parameters": sum(counts.values()),
            "fusion_macs_per_window": mac_per_time * 1800,
            "fusion_counted_flops_per_window": 2 * mac_per_time * 1800,
            "attention_weights_per_window": 1800 * 4 * 2 if model.method == "attention" else 0}
