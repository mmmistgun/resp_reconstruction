"""W0 三因素结构对照的八个原生模型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from resp_train.crd.initialization import module_seed
from resp_train.crd.tf_v1_model import CRDTfV1Model, trainable_parameter_count
from resp_train.temporal.blocks import ChannelLayerNorm1D, TemporalStem


PROTOCOL = "w0-structural-factorial-v1-es30p15-20260923"
FRONTEND_SUBSEED = "sfv1_frontend_conv20"
W0_MODEL_PARAMETERS = 1_219_850
PATCH_FRONTEND_PARAMETERS = 8_784
CONV20_FRONTEND_PARAMETERS = 24_672
TM3_PARAMETERS = 112_896
REF2_PARAMETERS = 75_264


@dataclass(frozen=True)
class ArmSpec:
    arm: str
    frontend: str
    cwt_temporal: bool
    refinement: bool
    trainable_parameters: int
    factor_covered_macs: int

    @property
    def factors(self) -> tuple[int, int, int]:
        return (
            int(self.frontend == "conv20"),
            int(self.cwt_temporal),
            int(self.refinement),
        )


def _arm(
    name: str,
    *,
    frontend: str,
    cwt_temporal: bool,
    refinement: bool,
    parameters: int,
    macs: int,
) -> ArmSpec:
    return ArmSpec(
        arm=name,
        frontend=frontend,
        cwt_temporal=bool(cwt_temporal),
        refinement=bool(refinement),
        trainable_parameters=int(parameters),
        factor_covered_macs=int(macs),
    )


ARM_SPECS: dict[str, ArmSpec] = {
    spec.arm: spec
    for spec in (
        _arm(
            "sfv1_patch_tm3_ref2",
            frontend="patch",
            cwt_temporal=True,
            refinement=True,
            parameters=1_219_850,
            macs=339_806_080,
        ),
        _arm(
            "sfv1_patch_tm3_ref0",
            frontend="patch",
            cwt_temporal=True,
            refinement=False,
            parameters=1_144_586,
            macs=205_367_680,
        ),
        _arm(
            "sfv1_patch_tm0_ref2",
            frontend="patch",
            cwt_temporal=False,
            refinement=True,
            parameters=1_106_954,
            macs=138_148_480,
        ),
        _arm(
            "sfv1_patch_tm0_ref0",
            frontend="patch",
            cwt_temporal=False,
            refinement=False,
            parameters=1_031_690,
            macs=3_710_080,
        ),
        _arm(
            "sfv1_conv20_tm3_ref2",
            frontend="conv20",
            cwt_temporal=True,
            refinement=True,
            parameters=1_235_738,
            macs=446_396_400,
        ),
        _arm(
            "sfv1_conv20_tm3_ref0",
            frontend="conv20",
            cwt_temporal=True,
            refinement=False,
            parameters=1_160_474,
            macs=311_958_000,
        ),
        _arm(
            "sfv1_conv20_tm0_ref2",
            frontend="conv20",
            cwt_temporal=False,
            refinement=True,
            parameters=1_122_842,
            macs=244_738_800,
        ),
        _arm(
            "sfv1_conv20_tm0_ref0",
            frontend="conv20",
            cwt_temporal=False,
            refinement=False,
            parameters=1_047_578,
            macs=110_300_400,
        ),
    )
}
ARMS = tuple(ARM_SPECS)
REFERENCE_ARM = "sfv1_patch_tm3_ref2"


CONV20_CONTRACT: dict[str, Any] = {
    "implementation": "resp_train.temporal.blocks.TemporalStem",
    "input_fs_hz": 100.0,
    "intermediate_fs_hz": 20.0,
    "output_fs_hz": 10.0,
    "output_channels": 96,
    "output_length": 1800,
    "stage_100_to_20": {
        "boundary": "line",
        "window": "kaiser",
        "beta": 8.6,
        "down": 5,
        "numtaps": 255,
        "cutoff_hz": 9.0,
    },
    "learned_20hz": {
        "carrier_filter": [1, 48, 21],
        "depthwise": [48, 48, 5],
        "projection": [48, 96, 5],
        "normalization": "channel_only_layer_norm_per_time_step",
        "activation": "silu_after_each_learned_stage",
    },
    "stage_20_to_10": {
        "boundary": "line",
        "window": "kaiser",
        "beta": 8.6,
        "down": 2,
        "numtaps": 127,
        "cutoff_hz": 4.5,
    },
    "trainable_parameters": CONV20_FRONTEND_PARAMETERS,
    "fixed_fir_values": 382,
}


class Conv20Frontend(TemporalStem):
    """冻结的 20-Hz 学习编码与 10-Hz 输出前端。"""

    def __init__(self) -> None:
        super().__init__(out_channels=96, mid_channels=48)
        norms = [module for module in self.modules() if isinstance(module, ChannelLayerNorm1D)]
        if (
            trainable_parameter_count(self) != CONV20_FRONTEND_PARAMETERS
            or tuple(self.decimate_100_to_20.taps.shape) != (255,)
            or tuple(self.decimate_20_to_10.taps.shape) != (127,)
            or self.depthwise_20.groups != 48
            or len(norms) != 3
            or any(parameter.requires_grad for parameter in self.decimate_100_to_20.parameters())
            or any(parameter.requires_grad for parameter in self.decimate_20_to_10.parameters())
        ):
            raise RuntimeError("CONV20 frontend 结构合同错误")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not bool(torch.isfinite(x).all()):
            raise FloatingPointError("CONV20 frontend 输入包含 NaN/Inf")
        output = super().forward(x)
        if not bool(torch.isfinite(output).all()):
            raise FloatingPointError("CONV20 frontend 输出包含 NaN/Inf")
        return output


class W0StructuralFactorialModel(CRDTfV1Model):
    """完整构造 W0 后，仅按冻结因素替换前端、TM3 和 REF2。"""

    def __init__(self, arm: str, initialization_seed: int) -> None:
        if arm not in ARM_SPECS:
            raise ValueError(f"未知结构因子 arm={arm!r}")
        self.arm_spec = ARM_SPECS[arm]
        super().__init__("crd_tf102_w", int(initialization_seed))
        if self.arm_spec.frontend == "conv20":
            with module_seed(int(initialization_seed), FRONTEND_SUBSEED):
                self.base.frontend = Conv20Frontend()
        if not self.arm_spec.cwt_temporal:
            self.branches["w"].temporal = nn.Identity()
        if not self.arm_spec.refinement:
            self.base.refinement = nn.Identity()
        self.experiment_arm = arm
        self._validate_factorial_contract()

    def _validate_factorial_contract(self) -> None:
        spec = self.arm_spec
        if (
            self.tf_variant != "crd_tf102_w"
            or self.representations != ("w",)
            or set(self.branches) != {"w"}
            or self.base.local_block_count != 6
            or len(self.base.local_blocks) != 6
            or self.gamma_coefficient != 0.5
            or self.beta_coefficient != 0.5
            or (spec.frontend == "conv20") != isinstance(self.base.frontend, Conv20Frontend)
            or spec.cwt_temporal == isinstance(self.branches["w"].temporal, nn.Identity)
            or spec.refinement == isinstance(self.base.refinement, nn.Identity)
            or trainable_parameter_count(self) != spec.trainable_parameters
        ):
            raise RuntimeError(f"结构因子模型合同错误: {spec.arm}")

    @property
    def factor_state_prefixes(self) -> dict[str, tuple[str, ...]]:
        """供实现锁和测试核对相同因素的逐 tensor 初始化。"""

        return {
            "frontend": ("base.frontend.",),
            "cwt_temporal": ("branches.w.temporal.",),
            "refinement": ("base.refinement.",),
        }


def model_contract(arm: str) -> dict[str, Any]:
    if arm not in ARM_SPECS:
        raise ValueError(f"未知结构因子 arm={arm!r}")
    spec = ARM_SPECS[arm]
    return {
        "protocol": PROTOCOL,
        "arm": arm,
        "frontend": spec.frontend,
        "cwt_temporal": "tm3" if spec.cwt_temporal else "tm0",
        "refinement": "ref2" if spec.refinement else "ref0",
        "factors": list(spec.factors),
        "trainable_parameters": spec.trainable_parameters,
        "factor_covered_macs": spec.factor_covered_macs,
        "conv20_contract": CONV20_CONTRACT,
    }


def build_w0_structural_factorial_model(cfg: Any) -> W0StructuralFactorialModel:
    from omegaconf import OmegaConf

    contract = cfg.model.get("w0_structural_factorial_v1")
    if not OmegaConf.is_config(contract):
        raise ValueError("缺少结构因子模型合同")
    arm = str(contract.arm)
    if (
        str(cfg.protocol.name) != PROTOCOL
        or str(cfg.model.name) != "crd_v1"
        or str(cfg.model.variant) != "crd_tf102_w"
        or OmegaConf.to_container(contract, resolve=True) != model_contract(arm)
    ):
        raise ValueError("结构因子模型只允许独立科学配置入口")
    return W0StructuralFactorialModel(arm, int(cfg.model.initialization_seed))
