"""固定APOR上的三个独立对照：普通隐藏层GELU、H64与Direct。"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from omegaconf import OmegaConf

from resp_train.crd.initialization import module_seed
from resp_train.crd.tf_v1_model import _ActiveParameterFill, _require_feature
from resp_train.paper_evidence.patch_apor_v1_model import PatchAporModel, model_contract as original_contract

PROTOCOL = "apor-gelu-refiner-v1-20260930"
SEEDS = (20260811, 20260812, 20260813)


@dataclass(frozen=True)
class ArmSpec:
    source_arm: str = "A0"
    gelu_hidden: bool = False
    refiner: str = "h65"
    condition: bool = True


ARM_SPECS = {"B0": ArmSpec(), "GELU": ArmSpec(gelu_hidden=True),
             "H64": ArmSpec(refiner="h64"), "DIRECT": ArmSpec(refiner="direct")}
ARMS = tuple(ARM_SPECS)
REFERENCE_ARMS = ("B0",)
TRAIN_ARMS = ("GELU", "H64", "DIRECT")
PARAMETERS = {"B0": 1077640, "GELU": 1077640, "H64": 1077448, "DIRECT": 1065064}


def model_contract(arm):
    spec = ARM_SPECS[arm]
    return {"protocol": PROTOCOL, "arm": arm, "source": original_contract("A0"),
            "parameters": PARAMETERS[arm], "selected_structure": "N0",
            "ordinary_hidden_activation": "gelu" if spec.gelu_hidden else "mixed",
            "changed_activation_count": 5 if spec.gelu_hidden else 0,
            "condition_refiner": spec.refiner, "h64_initialization": "first_64_h65_units" if arm == "H64" else None,
            "condition_grid": "physical_coordinates_140_patches_x5", "local_projection": [480, 96],
            "normalization_unchanged": True, "mamba_unchanged": True}


class GeluResidual(nn.Module):
    """接管原权重与名称；只替换条件残差中的普通隐藏层激活。"""

    def __init__(self, source):
        super().__init__()
        self.expand, self.project = source.expand, source.project

    def forward(self, value):
        return value + self.project(F.gelu(self.expand(value)))


class GeluPatchCondition(nn.Module):
    """沿用五点物理采样，覆盖原分支中函数形式的两处SiLU。"""

    def __init__(self, source):
        super().__init__()
        for name in ("conv_in", "norm", "depthwise", "conv_out", "sample", "local_projection", "final_projection"):
            setattr(self, name, getattr(source, name))
        self.parameter_fill = GeluResidual(source.parameter_fill)

    def forward(self, tf):
        value = _require_feature(tf, "w", (97, 360)).float()[:, None]
        value = F.gelu(self.norm(self.conv_in(value)))
        value = F.gelu(self.conv_out(self.depthwise(value))).mean(dim=2)
        sampled = self.sample(value)
        value = sampled.permute(0, 1, 3, 2).reshape(value.shape[0], -1, 140)
        value = self.local_projection(value)
        return self.final_projection(self.parameter_fill(value)).chunk(2, dim=1)


class GeluRefinerModel(PatchAporModel):
    def __init__(self, arm, seed):
        spec = ARM_SPECS[arm]
        super().__init__("A0", seed)
        self.study_arm = arm
        if spec.gelu_hidden:
            self.base.frontend.activation = nn.GELU()
            self.branches["w"] = GeluPatchCondition(self.branches["w"])
            self.patch_head[1] = nn.GELU()
        elif spec.refiner == "h64":
            source = self.branches["w"].parameter_fill
            with module_seed(seed, "apor_gelu_refiner_h64"):
                target = _ActiveParameterFill(64)
            # 嵌套初始化：保留前64个单元的全部权重，只去掉第65个隐藏单元。
            # 不缩放保留权重，公共模块及随机数流不因该构造改变。
            with torch.no_grad():
                target.expand.weight.copy_(source.expand.weight[:64])
                target.project.weight.copy_(source.project.weight[:, :64])
                target.project.bias.copy_(source.project.bias)
            self.branches["w"].parameter_fill = target
        elif spec.refiner == "direct":
            self.branches["w"].parameter_fill = nn.Identity()
        if sum(p.numel() for p in self.parameters()) != PARAMETERS[arm]:
            raise RuntimeError("GELU/refiner参数合同漂移")


def build_model(cfg):
    contract = OmegaConf.to_container(cfg.model.apor_gelu_refiner_v1, resolve=True)
    arm = contract["arm"]
    if cfg.protocol.name != PROTOCOL or contract != model_contract(arm):
        raise ValueError("GELU/refiner模型合同不符")
    return GeluRefinerModel(arm, int(cfg.model.initialization_seed))
