"""原始 M4 交叉注意力 Δt 消融：仅移除时间 bias，保持频率 bias。"""
from dataclasses import asdict, dataclass

from resp_train.models.patch_aligned_tf_mamba import PatchTFConfig
from scripts.m4_components_model import ZeroCoordinateBias
from scripts.p1_components_model import P1ComponentsModel, architecture, band_indices, frequency_metadata


@dataclass(frozen=True)
class TimeBiasArm:
    name: str
    time_bias: bool
    frequency_bias: bool = True
    band: str = "W-full"
    mamba: bool = True


ARMS = {"A0": TimeBiasArm("M4-Full", True), "A1": TimeBiasArm("M4-NoCrossTimeBias", False)}
REFERENCES = {"A0": "M4"}
TRAIN_ARMS = ("A1",)


class CrossTimeBiasModel(P1ComponentsModel):
    def __init__(self, arm, seed, frequencies_hz, *, cfg=None, mamba_factory=None):
        if arm not in ARMS:
            raise ValueError("未知交叉注意力时间 bias 实验臂")
        # 先完成原始 M4 的完整构造，保持共有参数及 CPU/CUDA RNG 的原构造序列。
        super().__init__("M4", seed, frequencies_hz, cfg=cfg, mamba_factory=mamba_factory)
        self.arm = arm
        if not ARMS[arm].time_bias:
            self.condition.time_bias = ZeroCoordinateBias(self.config.heads)


def contract(arm):
    return {"arm": arm, **asdict(ARMS[arm])}


def build_model(cfg, *, mamba_factory=None):
    value = dict(cfg.model.m4_cross_time_bias)
    arm = value["arm"]
    if arm not in TRAIN_ARMS or value != contract(arm) or cfg.model.name != "m4_cross_time_bias_v1":
        raise ValueError("交叉注意力时间 bias 训练合同不符")
    return CrossTimeBiasModel(arm, int(cfg.model.initialization_seed), frequency_metadata(cfg.data.tf_cache_path),
        cfg=PatchTFConfig(**dict(cfg.model.architecture)), mamba_factory=mamba_factory)
