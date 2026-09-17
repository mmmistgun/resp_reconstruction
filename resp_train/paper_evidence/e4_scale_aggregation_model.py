"""E4 唯一候选：冻结 W0 公共模块与四区域残差尺度聚合。"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from resp_train.crd.initialization import module_seed
from resp_train.crd.tf_v1_model import CwtBranch, CRDTfV1Model, _require_feature, trainable_parameter_count

ARM = "w0_mr4_residual"
REGIONS = ((0, 25), (25, 49), (49, 73), (73, 97))
FREQUENCY_CONTENT_SHA = "9fb164e7b09d42b31f7ee57a7d3e966af7adabc6d8c6a0e1eff3090b00b73c0c"
FREQUENCY_FILE_SHA = "15cc722c38e5572b3284c92137cfdeec7cbc4153b4588c2c1e086f26ae3238b3"
ADDED_PARAMETERS = 36_864
BRANCH_PARAMETERS = 186_912
MODEL_PARAMETERS = 1_256_714
AGGREGATION_CONTRACT = {
    "name": ARM, "regions": [list(pair) for pair in REGIONS],
    "partition": "equal_log_actual_frequency", "channels": 96,
    "pool": "per_region_arithmetic_mean", "concatenation": "region_then_channel",
    "fusion": "global_mean_plus_zero_init_conv1d_384_to_96_no_bias",
    "fill_hidden_channels": 65, "frequency_file_sha256": FREQUENCY_FILE_SHA,
    "branch_parameters": BRANCH_PARAMETERS, "model_parameters": MODEL_PARAMETERS,
}


def validate_frequency_grid(values: np.ndarray) -> dict:
    """校验原始 dtype/字节，再验证几何边界；推理只使用已锁的整数切片。"""
    if values.dtype != np.dtype("float64") or values.shape != (97,):
        raise ValueError("E4 frequency dtype/shape 漂移")
    if not np.isfinite(values).all() or np.any(values <= 0) or np.any(np.diff(values) < 0):
        raise ValueError("E4 frequency 必须有限、正值、非降序")
    if hashlib.sha256(values.tobytes(order="C")).hexdigest() != FREQUENCY_CONTENT_SHA:
        raise ValueError("E4 frequency content identity 漂移")
    edges = np.geomspace(values[0], values[-1], 5)
    membership = np.searchsorted(edges[1:-1], values, side="right")
    expected = np.concatenate([np.full(stop - start, k) for k, (start, stop) in enumerate(REGIONS)])
    if not np.array_equal(membership, expected):
        raise ValueError("E4 frequency partition 漂移")
    return {"values_hz": values.tolist(), "edges_hz": edges.tolist(),
            "regions": [list(pair) for pair in REGIONS], "membership": membership.tolist()}


class FourRegionAggregation(nn.Module):
    def __init__(self, initialization_seed: int) -> None:
        super().__init__()
        # Conv 构造器会消耗随机数，即使随后置零也必须独立隔离。
        with module_seed(initialization_seed, "e4_mr4_aggregation"):
            self.projection = nn.Conv1d(384, 96, kernel_size=1, bias=False)
            nn.init.zeros_(self.projection.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1:] != (96, 97, 360):
            raise ValueError("E4 aggregation 期望 (B,96,97,360)")
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError("E4 aggregation input 包含 NaN/Inf")
        global_mean = value.mean(dim=2)
        regional = torch.cat([value[:, :, start:stop].mean(dim=2) for start, stop in REGIONS], dim=1)
        output = global_mean + self.projection(regional)
        if not bool(torch.isfinite(output).all()):
            raise FloatingPointError("E4 aggregation output 包含 NaN/Inf")
        return output


class E4CwtBranch(CwtBranch):
    def __init__(self, initialization_seed: int) -> None:
        # 必须先完整执行原 W0 初始化和 fill=65，随后追加聚合器。
        super().__init__(scale_count=97)
        self.aggregation = FourRegionAggregation(initialization_seed)
        if self.parameter_fill.expand.out_channels != 65 or trainable_parameter_count(self) != BRANCH_PARAMETERS:
            raise RuntimeError("E4 branch/fill 参数合同错误")

    def forward(self, tf: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        value = _require_feature(tf, "w", (97, 360)).float()[:, None]
        value = F.silu(self.norm(self.conv_in(value)))
        value = F.silu(self.conv_out(self.depthwise(value)))
        value = self.aggregation(value)
        value = F.interpolate(value, size=1800, mode="linear", align_corners=False)
        return self.project_condition(self.temporal(value))


class E4ScaleAggregationModel(CRDTfV1Model):
    def __init__(self, initialization_seed: int) -> None:
        super().__init__("crd_tf102_w", initialization_seed)
        # 重建单个 branch 使用原子 seed，公共参数键/次序和外部 RNG 均保持。
        with module_seed(initialization_seed, "tf_branch_w"):
            self.branches["w"] = E4CwtBranch(initialization_seed)
        self.experiment_arm = ARM
        self._validate_parameter_contract()

    def _validate_parameter_contract(self) -> None:
        # 父构造的第一阶段校验原 W0；替换完成后使用独立的精确 E4 合同。
        if not hasattr(self, "experiment_arm"):
            return super()._validate_parameter_contract()
        if (set(self.branches) != {"w"} or not isinstance(self.branches["w"], E4CwtBranch)
                or trainable_parameter_count(self.branches["w"]) != BRANCH_PARAMETERS
                or trainable_parameter_count(self) != MODEL_PARAMETERS or len(self.base.local_blocks) != 6):
            raise RuntimeError("E4 模型参数/深度合同错误")


def build_e4_model(cfg) -> E4ScaleAggregationModel:
    from omegaconf import OmegaConf

    aggregation = cfg.model.get("e4_aggregation")
    if (str(cfg.model.variant) != "crd_tf102_w" or str(cfg.protocol.name) != "e4-w0-scale-aggregation-v1-20260917"
            or not OmegaConf.is_config(aggregation)
            or OmegaConf.to_container(aggregation, resolve=True) != AGGREGATION_CONTRACT):
        raise ValueError("E4 模型仅允许独立科学配置入口")
    return E4ScaleAggregationModel(int(cfg.model.initialization_seed))
