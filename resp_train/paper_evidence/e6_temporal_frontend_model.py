"""E6：在 20 Hz 完成可学习解调后再形成 10-Hz W0 latent。"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from resp_train.crd.initialization import module_seed
from resp_train.crd.tf_v1_model import CRDTfV1Model, trainable_parameter_count
from resp_train.temporal.blocks import ChannelLayerNorm1D, TemporalStem


ARM = "e6_tfe201_aa20_demod10_w0"
CONTROL_ARM = "e6_tfe000_patch_w0"
PROTOCOL = "e6-temporal-frontend-v1-20260923"
FRONTEND_SUBSEED = "e6_temporal_frontend_aa20_demod10"
FRONTEND_PARAMETERS = 24_672
MODEL_PARAMETERS = 1_235_738
W0_MODEL_PARAMETERS = 1_219_850
W0_FRONTEND_PARAMETERS = 8_784

FRONTEND_CONTRACT: dict[str, Any] = {
    "name": ARM,
    "input_fs_hz": 100.0,
    "intermediate_fs_hz": 20.0,
    "output_fs_hz": 10.0,
    "output_channels": 96,
    "output_length": 1800,
    "stage_100_to_20": {
        "implementation": "fixed_polyphase_line",
        "window": "kaiser",
        "beta": 8.6,
        "down": 5,
        "numtaps": 255,
        "cutoff_hz": 9.0,
    },
    "learned_20hz": {
        "carrier_filter": {
            "in_channels": 1,
            "out_channels": 48,
            "kernel_size": 21,
            "padding": 10,
            "bias": False,
        },
        "depthwise": {
            "channels": 48,
            "kernel_size": 5,
            "padding": 2,
            "bias": False,
        },
        "projection": {
            "in_channels": 48,
            "out_channels": 96,
            "kernel_size": 5,
            "padding": 2,
            "bias": False,
        },
        "normalization": "channel_only_layer_norm_per_time_step",
        "activation": "silu_after_each_learned_stage",
        "initialization": "kaiming_normal_fan_in_relu",
    },
    "stage_20_to_10": {
        "implementation": "fixed_polyphase_line",
        "window": "kaiser",
        "beta": 8.6,
        "down": 2,
        "numtaps": 127,
        "cutoff_hz": 4.5,
    },
    "implementation_source": "resp_train.temporal.blocks.TemporalStem",
    "trainable_parameters": FRONTEND_PARAMETERS,
    "model_parameters": MODEL_PARAMETERS,
}


class E6TemporalFrontend(TemporalStem):
    """复用 RTM 冻结 substrate，并补充 W0 实验所需的严格有限值合同。"""

    def __init__(self) -> None:
        super().__init__(out_channels=96, mid_channels=48)
        self._validate_structure()

    def _validate_structure(self) -> None:
        norms = [module for module in self.modules() if isinstance(module, ChannelLayerNorm1D)]
        if (
            trainable_parameter_count(self) != FRONTEND_PARAMETERS
            or tuple(self.decimate_100_to_20.taps.shape) != (255,)
            or tuple(self.decimate_20_to_10.taps.shape) != (127,)
            or any(parameter.requires_grad for parameter in self.decimate_100_to_20.parameters())
            or any(parameter.requires_grad for parameter in self.decimate_20_to_10.parameters())
            or self.depthwise_20.groups != 48
            or len(norms) != 3
        ):
            raise RuntimeError("E6 frontend 结构或 fixed-filter 合同错误")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not bool(torch.isfinite(x).all()):
            raise FloatingPointError("E6 frontend 输入包含 NaN/Inf")
        output = super().forward(x)
        if not bool(torch.isfinite(output).all()):
            raise FloatingPointError("E6 frontend 输出包含 NaN/Inf")
        return output


class E6TemporalFrontendModel(CRDTfV1Model):
    """完整 W0 只替换 base.frontend，其余模块保持原命名和同 seed 初始化。"""

    def __init__(self, initialization_seed: int) -> None:
        super().__init__("crd_tf102_w", int(initialization_seed))
        with module_seed(int(initialization_seed), FRONTEND_SUBSEED):
            self.base.frontend = E6TemporalFrontend()
        self.experiment_arm = ARM
        self._validate_e6_contract()

    def _validate_e6_contract(self) -> None:
        if (
            self.tf_variant != "crd_tf102_w"
            or self.representations != ("w",)
            or set(self.branches) != {"w"}
            or self.branch_parameter_counts() != {"w": 150_048}
            or self.base.local_block_count != 6
            or len(self.base.local_blocks) != 6
            or not isinstance(self.base.frontend, E6TemporalFrontend)
            or trainable_parameter_count(self.base.frontend) != FRONTEND_PARAMETERS
            or trainable_parameter_count(self) != MODEL_PARAMETERS
            or self.gamma_coefficient != 0.5
            or self.beta_coefficient != 0.5
        ):
            raise RuntimeError("E6 完整模型合同错误")


def build_e6_model(cfg: Any) -> E6TemporalFrontendModel:
    from omegaconf import OmegaConf

    contract = cfg.model.get("e6_temporal_frontend")
    if (
        str(cfg.protocol.name) != PROTOCOL
        or str(cfg.model.name) != "crd_v1"
        or str(cfg.model.variant) != "crd_tf102_w"
        or not OmegaConf.is_config(contract)
        or OmegaConf.to_container(contract, resolve=True) != FRONTEND_CONTRACT
    ):
        raise ValueError("E6 模型只允许独立科学配置入口")
    return E6TemporalFrontendModel(int(cfg.model.initialization_seed))
