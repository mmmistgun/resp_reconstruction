"""原始 P1-M4 的配对参数化对照，保留历史模块初始化流。"""
from dataclasses import asdict

import torch
from torch import nn

from resp_train.models.patch_aligned_tf_mamba import PatchTFConfig
from scripts.p1_components_model import (
    Arm, P1ComponentsModel, architecture, band_indices, frequency_metadata,
)

ARMS = {
    "M4": Arm("P1-Full-Add", fusion="add"),
    "SP": Arm("P1-M4-SingleProjection", fusion="single_projection"),
}


class DirectAdd(nn.Module):
    """delta 已经过唯一输出投影；此处只加回未经变换的原始 z。"""

    def forward(self, tokens, delta):
        return tokens + delta


class M4SingleProjectionModel(P1ComponentsModel):
    def __init__(self, arm, seed, frequencies_hz, *, cfg=None, mamba_factory=None):
        if arm not in ARMS:
            raise ValueError("未知单投影对照实验臂")
        # 保留历史 attention RNG 消耗顺序，包括 output 的构造位置。
        # 这样 Q/K/V、坐标网络以及其余独立 namespace 的共有权重均严格匹配 M4。
        super().__init__("M4", seed, frequencies_hz, cfg=cfg, mamba_factory=mamba_factory)
        self.arm, self.spec = arm, ARMS[arm]
        if arm == "SP":
            nn.init.zeros_(self.condition.output.weight)
            nn.init.zeros_(self.condition.output.bias)
            self.fusion = DirectAdd()


def contract(arm):
    return {"arm": arm, **asdict(ARMS[arm]), "initialization": "original_p1_module_namespaces"}


def build_model(cfg, *, mamba_factory=None):
    value = dict(cfg.model.m4_single_projection)
    arm = value["arm"]
    if value != contract(arm) or cfg.model.name != "m4_single_projection_v1":
        raise ValueError("M4 单投影模型合同不符")
    if PatchTFConfig(**dict(cfg.model.architecture)) != architecture():
        raise ValueError("正式单投影实验必须保留原始 M4 几何与宽度")
    return M4SingleProjectionModel(arm, int(cfg.model.initialization_seed),
        frequency_metadata(cfg.data.tf_cache_path), mamba_factory=mamba_factory)


def verify_shared_weights(reference, candidate):
    """检查全部共有 parameter/buffer，输出投影参数化是唯一允许的差异。"""
    left, right = reference.state_dict(), candidate.state_dict()
    removed = {"fusion.projection.weight", "fusion.projection.bias"}
    changed = {"condition.output.weight", "condition.output.bias"}
    if set(left) - set(right) != removed or set(right) - set(left):
        raise AssertionError("单投影模型的模块集合发生非预设变化")
    common = sorted(set(right) - changed)
    for name in common:
        if not torch.equal(left[name], right[name]):
            raise AssertionError(f"共有权重或 buffer 不一致: {name}")
    return common
