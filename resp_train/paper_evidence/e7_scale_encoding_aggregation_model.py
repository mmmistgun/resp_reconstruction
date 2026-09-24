"""E7 聚合前尺度编码 × 尺度聚合六臂模型。"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from resp_train.crd.initialization import module_seed, named_subseed
from resp_train.crd.tf_v1_model import (
    CwtBranch,
    CRDTfV1Model,
    _initialize_conv,
    _require_feature,
    trainable_parameter_count,
)
from resp_train.paper_evidence.e4_scale_aggregation_model import validate_frequency_grid


PROTOCOL = "e7-scale-encoding-aggregation-factorial-v1-20260924"
ENCODERS = ("s0_shallow", "s1_deep_local", "s2_axis_spanning")
AGGREGATIONS = ("mean", "frequency_attention")
ARMS = tuple(f"{encoder}__{aggregation}" for encoder in ENCODERS for aggregation in AGGREGATIONS)
DILATIONS = {
    "s0_shallow": (),
    "s1_deep_local": (1, 1, 1, 1),
    "s2_axis_spanning": (1, 2, 4, 8),
}
THEORETICAL_RECEPTIVE_FIELDS = {
    "s0_shallow": 7,
    "s1_deep_local": 39,
    "s2_axis_spanning": 127,
}
W0_MODEL_PARAMETERS = 1_219_850
W0_BRANCH_PARAMETERS = 150_048
BLOCK_PARAMETERS = 10_368
ENCODER_PARAMETERS = {name: len(DILATIONS[name]) * BLOCK_PARAMETERS for name in ENCODERS}
AGGREGATION_PARAMETERS = {"mean": 0, "frequency_attention": 792}


@contextmanager
def cpu_module_seed(seed: int, name: str) -> Iterator[None]:
    """隔离新增 CPU 参数初始化，同时保持 CUDA generator 状态。"""

    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(named_subseed(int(seed), name))
        yield


def split_arm(arm: str) -> tuple[str, str]:
    if arm not in ARMS:
        raise ValueError(f"E7 未知 arm={arm!r}")
    encoder, aggregation = arm.split("__", maxsplit=1)
    return encoder, aggregation


def arm_contract(arm: str) -> dict[str, Any]:
    encoder, aggregation = split_arm(arm)
    added = ENCODER_PARAMETERS[encoder] + AGGREGATION_PARAMETERS[aggregation]
    return {
        "arm": arm,
        "encoder": encoder,
        "aggregation": aggregation,
        "channels": 96,
        "scale_count": 97,
        "context_length": 360,
        "encoder_dilations": list(DILATIONS[encoder]),
        "encoder_kernel": [9, 1],
        "encoder_normalization": "channel_only_layer_norm_per_scale_time",
        "encoder_activation": "silu",
        "encoder_initialization": "kaiming_normal_fan_in_relu",
        "theoretical_scale_receptive_field": THEORETICAL_RECEPTIVE_FIELDS[encoder],
        "attention_hidden_channels": 8,
        "frequency_coordinate": "log_frequency_minmax_minus1_plus1",
        "attention_normalization": "fp32_stable_softmax_over_scale",
        "fill_hidden_channels": 65,
        "added_parameters": added,
        "branch_parameters": W0_BRANCH_PARAMETERS + added,
        "model_parameters": W0_MODEL_PARAMETERS + added,
    }


class ChannelLayerNorm2D(nn.Module):
    """对 NCHW 的每个空间位置独立执行 channel-only LayerNorm。"""

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.channels = int(channels)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.channels))
        self.bias = nn.Parameter(torch.zeros(self.channels))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1] != self.channels:
            raise ValueError(
                f"channel-only LayerNorm 期望 (B,{self.channels},S,T)，实际 {tuple(value.shape)}"
            )
        channels_last = value.movedim(1, -1)
        normalized = F.layer_norm(
            channels_last,
            (self.channels,),
            self.weight,
            self.bias,
            self.eps,
        )
        return normalized.movedim(-1, 1)


class ScaleResidualBlock(nn.Module):
    def __init__(self, dilation: int) -> None:
        super().__init__()
        if int(dilation) <= 0:
            raise ValueError("E7 dilation 必须为正整数")
        self.dilation = int(dilation)
        self.norm = ChannelLayerNorm2D(96)
        self.depthwise = nn.Conv2d(
            96,
            96,
            kernel_size=(9, 1),
            dilation=(self.dilation, 1),
            padding=(4 * self.dilation, 0),
            groups=96,
            bias=False,
        )
        self.projection = nn.Conv2d(96, 96, kernel_size=1, bias=True)
        _initialize_conv(self.depthwise)
        _initialize_conv(self.projection)
        if trainable_parameter_count(self) != BLOCK_PARAMETERS:
            raise RuntimeError("E7 scale residual block 参数数错误")

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.projection(self.depthwise(F.silu(self.norm(value))))
        output = value + residual
        if output.shape != value.shape or not bool(torch.isfinite(output).all()):
            raise FloatingPointError("E7 scale residual block 输出异常")
        return output


class ScaleEncoder(nn.Module):
    def __init__(self, encoder: str, initialization_seed: int) -> None:
        super().__init__()
        if encoder not in ENCODERS:
            raise ValueError(f"E7 未知 encoder={encoder!r}")
        self.encoder = encoder
        with cpu_module_seed(int(initialization_seed), "e7_scale_encoder"):
            self.blocks = nn.ModuleList(
                [ScaleResidualBlock(dilation) for dilation in DILATIONS[encoder]]
            )
        if trainable_parameter_count(self) != ENCODER_PARAMETERS[encoder]:
            raise RuntimeError("E7 scale encoder 参数数错误")

    def forward(
        self,
        value: torch.Tensor,
        *,
        enabled: bool = True,
        return_blocks: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        records: list[tuple[torch.Tensor, torch.Tensor]] = []
        output = value
        if enabled:
            for block in self.blocks:
                block_input = output
                output = block(output)
                if return_blocks:
                    records.append((block_input, output))
        if return_blocks:
            return output, records
        return output


class MeanAggregation(nn.Module):
    def forward(
        self,
        value: torch.Tensor,
        *,
        uniform: bool = False,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        del uniform
        mean = value.mean(dim=2)
        if return_details:
            return mean, {"mean": mean, "correction": torch.zeros_like(mean), "weights": None}
        return mean


class FrequencyAttentionAggregation(nn.Module):
    def __init__(self, initialization_seed: int, frequencies: np.ndarray) -> None:
        super().__init__()
        validate_frequency_grid(frequencies)
        coordinate = np.log(frequencies)
        coordinate = 2 * (coordinate - coordinate[0]) / (coordinate[-1] - coordinate[0]) - 1
        self.register_buffer(
            "frequency_coordinate",
            torch.tensor(coordinate, dtype=torch.float32).reshape(1, 1, 97, 1),
        )
        with cpu_module_seed(int(initialization_seed), "e7_frequency_attention"):
            self.content = nn.Conv2d(96, 8, kernel_size=1, bias=True)
            self.score = nn.Conv2d(8, 1, kernel_size=1, bias=False)
            self.frequency_weight = nn.Parameter(torch.empty(1, 8, 1, 1))
            _initialize_conv(self.content)
            _initialize_conv(self.score)
            nn.init.normal_(self.frequency_weight, std=96**-0.5)
        if trainable_parameter_count(self) != AGGREGATION_PARAMETERS["frequency_attention"]:
            raise RuntimeError("E7 frequency attention 参数数错误")

    def forward(
        self,
        value: torch.Tensor,
        *,
        uniform: bool = False,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if value.ndim != 4 or value.shape[1:] != (96, 97, 360):
            raise ValueError("E7 attention 期望 (B,96,97,360)")
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError("E7 attention 输入非有限")
        mean = value.mean(dim=2)
        if uniform:
            weights = torch.full(
                (value.shape[0], 1, 97, value.shape[3]),
                1.0 / 97.0,
                dtype=torch.float32,
                device=value.device,
            )
            output = mean
        else:
            hidden = self.content(value)
            positional = self.frequency_weight * self.frequency_coordinate
            hidden = hidden + positional.to(device=hidden.device, dtype=hidden.dtype)
            logits = self.score(F.silu(hidden)).float()
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError("E7 attention logits 非有限")
            weights = torch.softmax(logits, dim=2)
            output = (value.float() * weights).sum(dim=2).to(dtype=mean.dtype)
        correction = output - mean
        if not bool(torch.isfinite(output).all()):
            raise FloatingPointError("E7 attention 输出非有限")
        if return_details:
            return output, {"mean": mean, "correction": correction, "weights": weights}
        return output


class E7CwtBranch(CwtBranch):
    def __init__(
        self,
        arm: str,
        initialization_seed: int,
        frequencies: np.ndarray,
    ) -> None:
        super().__init__(scale_count=97)
        encoder, aggregation = split_arm(arm)
        self.arm = arm
        self.encoder_name = encoder
        self.aggregation_name = aggregation
        self.scale_encoder = ScaleEncoder(encoder, initialization_seed)
        if aggregation == "mean":
            self.aggregation = MeanAggregation()
        else:
            self.aggregation = FrequencyAttentionAggregation(initialization_seed, frequencies)
        self._residual_enabled = True
        self._uniform_attention = False
        if self.parameter_fill.expand.out_channels != 65:
            raise RuntimeError("E7 active fill hidden channels 必须为 65")
        if trainable_parameter_count(self) != arm_contract(arm)["branch_parameters"]:
            raise RuntimeError("E7 branch 参数数错误")

    @contextmanager
    def intervention(
        self,
        *,
        residual_enabled: bool = True,
        uniform_attention: bool = False,
    ) -> Iterator[None]:
        previous = (self._residual_enabled, self._uniform_attention)
        self._residual_enabled = bool(residual_enabled)
        self._uniform_attention = bool(uniform_attention)
        try:
            yield
        finally:
            self._residual_enabled, self._uniform_attention = previous

    def encode(self, tf: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        value = _require_feature(tf, "w", (97, 360)).float()[:, None]
        value = F.silu(self.norm(self.conv_in(value)))
        x0 = F.silu(self.conv_out(self.depthwise(value)))
        xe = self.scale_encoder(x0, enabled=self._residual_enabled)
        return x0, xe

    def forward_features(
        self,
        tf: Mapping[str, torch.Tensor],
    ) -> dict[str, Any]:
        value = _require_feature(tf, "w", (97, 360)).float()[:, None]
        value = F.silu(self.norm(self.conv_in(value)))
        x0 = F.silu(self.conv_out(self.depthwise(value)))
        xe, blocks = self.scale_encoder(
            x0,
            enabled=self._residual_enabled,
            return_blocks=True,
        )
        aggregated, details = self.aggregation(
            xe,
            uniform=self._uniform_attention,
            return_details=True,
        )
        return {
            "x0": x0,
            "xe": xe,
            "block_records": blocks,
            "aggregated": aggregated,
            **details,
        }

    def forward(self, tf: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        _, xe = self.encode(tf)
        value = self.aggregation(xe, uniform=self._uniform_attention)
        value = F.interpolate(value, size=1800, mode="linear", align_corners=False)
        return self.project_condition(self.temporal(value))


class E7ScaleFactorialModel(CRDTfV1Model):
    def __init__(self, arm: str, initialization_seed: int, frequencies: np.ndarray) -> None:
        super().__init__("crd_tf102_w", int(initialization_seed))
        with module_seed(int(initialization_seed), "tf_branch_w"):
            self.branches["w"] = E7CwtBranch(arm, initialization_seed, frequencies)
        self.experiment_arm = arm
        self._validate_e7_contract()

    @property
    def e7_branch(self) -> E7CwtBranch:
        branch = self.branches["w"]
        if not isinstance(branch, E7CwtBranch):
            raise RuntimeError("E7 W branch 类型错误")
        return branch

    @contextmanager
    def intervention(
        self,
        *,
        residual_enabled: bool = True,
        uniform_attention: bool = False,
    ) -> Iterator[None]:
        with self.e7_branch.intervention(
            residual_enabled=residual_enabled,
            uniform_attention=uniform_attention,
        ):
            yield

    def _validate_e7_contract(self) -> None:
        contract = arm_contract(self.experiment_arm)
        if (
            self.tf_variant != "crd_tf102_w"
            or self.representations != ("w",)
            or set(self.branches) != {"w"}
            or self.base.local_block_count != 6
            or len(self.base.local_blocks) != 6
            or trainable_parameter_count(self.e7_branch) != contract["branch_parameters"]
            or trainable_parameter_count(self) != contract["model_parameters"]
            or self.e7_branch.parameter_fill.expand.out_channels != 65
        ):
            raise RuntimeError("E7 完整模型合同错误")


def build_e7_model(cfg: Any) -> E7ScaleFactorialModel:
    from omegaconf import OmegaConf

    settings = cfg.model.get("e7_factorial")
    if (
        str(cfg.protocol.name) != PROTOCOL
        or str(cfg.model.variant) != "crd_tf102_w"
        or not OmegaConf.is_config(settings)
    ):
        raise ValueError("E7 模型只允许独立科学配置入口")
    arm = str(settings.arm)
    if OmegaConf.to_container(settings, resolve=True) != arm_contract(arm):
        raise ValueError("E7 arm 合同漂移")
    frequencies = np.asarray(
        OmegaConf.to_container(cfg.model.aggregation_frequencies_hz, resolve=True),
        dtype=np.float64,
    )
    validate_frequency_grid(frequencies)
    return E7ScaleFactorialModel(arm, int(cfg.model.initialization_seed), frequencies)
