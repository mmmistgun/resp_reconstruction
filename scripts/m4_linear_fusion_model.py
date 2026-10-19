"""M4 简化基线的双投影与单投影 residual fusion 对照。"""
from dataclasses import asdict, dataclass

from torch import nn

from resp_train.models.patch_aligned_tf_mamba import PatchTFConfig
from scripts.m4_cross_time_bias_model import CrossTimeBiasModel, architecture, band_indices, frequency_metadata


@dataclass(frozen=True)
class FusionArm:
    name: str
    fusion_projection: bool
    time_bias: bool = False
    frequency_bias: bool = True
    band: str = "W-full"
    mamba: bool = True


ARMS = {"L0": FusionArm("M4-DualProjection", True), "L1": FusionArm("M4-SingleProjection", False)}
REFERENCES = {"L0": "A1"}
TRAIN_ARMS = ("L1",)


class DirectResidualAdd(nn.Module):
    def forward(self, tokens, context):
        return tokens + context


class LinearFusionModel(CrossTimeBiasModel):
    def __init__(self, arm, seed, frequencies_hz, *, cfg=None, mamba_factory=None):
        if arm not in ARMS:
            raise ValueError("未知 linear fusion 实验臂")
        # 完整复用简化基线的构造顺序；零初始化和参数移除不增加随机数消耗。
        super().__init__("A1", seed, frequencies_hz, cfg=cfg, mamba_factory=mamba_factory)
        self.arm = arm
        if not ARMS[arm].fusion_projection:
            # 两个 affine 的初始合成映射为零；单投影也从零残差开始。
            nn.init.zeros_(self.condition.output.weight)
            nn.init.zeros_(self.condition.output.bias)
            self.fusion = DirectResidualAdd()


def contract(arm):
    return {"arm": arm, **asdict(ARMS[arm]),
        "residual_initialization": "zero_fusion_projection" if ARMS[arm].fusion_projection else "zero_attention_output"}


def build_model(cfg, *, mamba_factory=None):
    value = dict(cfg.model.m4_linear_fusion)
    arm = value["arm"]
    if arm not in TRAIN_ARMS or value != contract(arm) or cfg.model.name != "m4_linear_fusion_v1":
        raise ValueError("linear fusion 训练合同不符")
    return LinearFusionModel(arm, int(cfg.model.initialization_seed), frequency_metadata(cfg.data.tf_cache_path),
        cfg=PatchTFConfig(**dict(cfg.model.architecture)), mamba_factory=mamba_factory)
