"""E5 唯一候选：W0 专用的显式抗混叠 10-Hz 局部残差前端。"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from resp_train.crd.initialization import module_seed
from resp_train.crd.tf_v1_model import CRDTfV1Model, trainable_parameter_count
from resp_train.temporal.blocks import FixedPolyphaseDecimator1D


ARM = "e5_tfe101_aa10_res_w0"
CONTROL_ARM = "e5_tfe000_patch_w0"
PROTOCOL = "e5-temporal-frontend-v1-20260922"
FRONTEND_SUBSEED = "e5_temporal_frontend_aa10_residual"
FRONTEND_PARAMETERS = 1_536
MODEL_PARAMETERS = 1_212_602
W0_MODEL_PARAMETERS = 1_219_850
W0_FRONTEND_PARAMETERS = 8_784
FIXED_TAP_COUNT = 511

FRONTEND_CONTRACT: dict[str, Any] = {
    "name": ARM,
    "input_fs_hz": 100.0,
    "output_fs_hz": 10.0,
    "output_channels": 96,
    "output_length": 1800,
    "anti_alias": {
        "implementation": "fixed_polyphase_line",
        "window": "kaiser",
        "beta": 8.6,
        "down": 10,
        "numtaps": FIXED_TAP_COUNT,
        "cutoff_hz": 4.5,
    },
    "local_embedding": {
        "in_channels": 1,
        "out_channels": 96,
        "kernel_size": 11,
        "padding": 5,
        "bias": False,
        "initialization": "kaiming_normal_fan_in_relu",
    },
    "local_residual": {
        "type": "silu_then_depthwise_conv1d",
        "channels": 96,
        "kernel_size": 5,
        "padding": 2,
        "bias": False,
        "initialization": "zeros",
    },
    "normalization": "none",
    "pointwise_mixer": False,
    "trainable_parameters": FRONTEND_PARAMETERS,
    "model_parameters": MODEL_PARAMETERS,
}


def _kaiming_embedding(module: nn.Conv1d) -> None:
    nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class E5TemporalFrontend(nn.Module):
    """先建立严格 10-Hz 网格，再编码有符号局部波形与有限非线性形态。"""

    def __init__(self) -> None:
        super().__init__()
        self.decimate_100_to_10 = FixedPolyphaseDecimator1D(
            input_fs=100.0,
            down=10,
            numtaps=FIXED_TAP_COUNT,
            cutoff_hz=4.5,
        )
        self.local_embedding = nn.Conv1d(
            1,
            96,
            kernel_size=11,
            padding=5,
            bias=False,
        )
        self.local_residual = nn.Conv1d(
            96,
            96,
            kernel_size=5,
            padding=2,
            groups=96,
            bias=False,
        )
        _kaiming_embedding(self.local_embedding)
        # 初始函数严格退化为线性局部 embedding；残差首轮仍有直接梯度。
        nn.init.zeros_(self.local_residual.weight)
        self._validate_structure()

    def _validate_structure(self) -> None:
        if any(isinstance(module, (nn.GroupNorm, nn.LayerNorm, nn.BatchNorm1d)) for module in self.modules()):
            raise RuntimeError("E5 frontend 禁止 normalization")
        if (
            trainable_parameter_count(self) != FRONTEND_PARAMETERS
            or tuple(self.decimate_100_to_10.taps.shape) != (FIXED_TAP_COUNT,)
            or any(parameter.requires_grad for parameter in self.decimate_100_to_10.parameters())
            or self.local_residual.groups != 96
        ):
            raise RuntimeError("E5 frontend 参数或 fixed-filter 合同错误")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (1, 18000):
            raise ValueError(f"E5 frontend 期望 (B,1,18000)，实际 {tuple(x.shape)}")
        if not bool(torch.isfinite(x).all()):
            raise FloatingPointError("E5 frontend 输入包含 NaN/Inf")
        sampled = self.decimate_100_to_10(x)
        if sampled.shape[1:] != (1, 1800):
            raise RuntimeError(f"E5 fixed decimator 输出错误: {tuple(sampled.shape)}")
        embedded = self.local_embedding(sampled)
        output = embedded + self.local_residual(F.silu(embedded))
        if output.shape[1:] != (96, 1800):
            raise RuntimeError(f"E5 frontend 输出契约错误: {tuple(output.shape)}")
        if not bool(torch.isfinite(output).all()):
            raise FloatingPointError("E5 frontend 输出包含 NaN/Inf")
        return output


class E5TemporalFrontendModel(CRDTfV1Model):
    """完整 W0，只替换 base.frontend；其余公共模块保持原命名和初始化。"""

    def __init__(self, initialization_seed: int) -> None:
        super().__init__("crd_tf102_w", int(initialization_seed))
        with module_seed(int(initialization_seed), FRONTEND_SUBSEED):
            self.base.frontend = E5TemporalFrontend()
        self.experiment_arm = ARM
        self._validate_e5_contract()

    def _validate_e5_contract(self) -> None:
        if (
            self.tf_variant != "crd_tf102_w"
            or self.representations != ("w",)
            or set(self.branches) != {"w"}
            or self.branch_parameter_counts() != {"w": 150_048}
            or self.base.local_block_count != 6
            or len(self.base.local_blocks) != 6
            or not isinstance(self.base.frontend, E5TemporalFrontend)
            or trainable_parameter_count(self.base.frontend) != FRONTEND_PARAMETERS
            or trainable_parameter_count(self) != MODEL_PARAMETERS
            or self.gamma_coefficient != 0.5
            or self.beta_coefficient != 0.5
        ):
            raise RuntimeError("E5 完整模型合同错误")


def build_e5_model(cfg: Any) -> E5TemporalFrontendModel:
    from omegaconf import OmegaConf

    contract = cfg.model.get("e5_temporal_frontend")
    if (
        str(cfg.protocol.name) != PROTOCOL
        or str(cfg.model.name) != "crd_v1"
        or str(cfg.model.variant) != "crd_tf102_w"
        or not OmegaConf.is_config(contract)
        or OmegaConf.to_container(contract, resolve=True) != FRONTEND_CONTRACT
    ):
        raise ValueError("E5 模型只允许独立科学配置入口")
    return E5TemporalFrontendModel(int(cfg.model.initialization_seed))
