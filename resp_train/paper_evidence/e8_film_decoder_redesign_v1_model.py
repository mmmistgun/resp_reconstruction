"""E8 FiLM 条件末端 × 波形解码端的十二个原生模型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from resp_train.crd.blocks import DecoderResidual
from resp_train.crd.initialization import module_seed
from resp_train.crd.tf_v1_model import CRDTfV1Model, trainable_parameter_count


PROTOCOL = "e8-film-decoder-redesign-factorial-v1-20260925"
W0_MODEL_PARAMETERS = 1_219_850
W0_CONDITION_REFINER_PARAMETERS = 12_576
W0_DECODER_RESIDUAL_PARAMETERS = 1_057

CONDITION_REFINERS = ("fill65", "direct", "res96", "res192")
DECODERS = ("pointwise", "single", "temporal")

CONDITION_PARAMETERS = {
    "fill65": 12_576,
    "direct": 0,
    "res96": 18_528,
    "res192": 36_960,
}
DECODER_PARAMETERS = {
    "pointwise": 1_057,
    "single": 0,
    "temporal": 1_217,
}

# 这里只覆盖发生变化的条件重组与 residual readout，不宣称是完整模型 FLOPs。
CONDITION_COVERED_MACS = {
    "fill65": 22_464_000,
    "direct": 0,
    "res96": 33_177_600,
    "res192": 66_355_200,
}
DECODER_COVERED_MACS = {
    "pointwise": 1_900_800,
    "single": 0,
    "temporal": 2_188_800,
}


def _kaiming(module: nn.Conv1d) -> None:
    nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class ChannelResidualRefiner(nn.Module):
    """保持时间索引不变的逐点通道残差重组。"""

    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        hidden_channels = int(hidden_channels)
        if hidden_channels not in {96, 192}:
            raise ValueError("E8 condition refiner hidden_channels 只允许 96 或 192")
        self.hidden_channels = hidden_channels
        self.expand = nn.Conv1d(96, hidden_channels, kernel_size=1, bias=False)
        self.project = nn.Conv1d(hidden_channels, 96, kernel_size=1, bias=True)
        _kaiming(self.expand)
        _kaiming(self.project)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 3 or context.shape[1:] != (96, 1800):
            raise ValueError(
                f"E8 condition refiner 期望 (B,96,1800)，实际 {tuple(context.shape)}"
            )
        return context + self.project(F.silu(self.expand(context)))


class SingleReadoutResidual(nn.Module):
    """单一读出 arm 的显式零修正；不包含可学习参数。"""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[1] != 32:
            raise ValueError(
                f"E8 single readout 期望 (B,32,T)，实际 {tuple(features.shape)}"
            )
        return torch.zeros_like(features[:, :1])


class TemporalDecoderResidual(nn.Module):
    """10 Hz 局部时间残差：DW k5/d2 后接两层逐点投影。"""

    def __init__(self) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            32,
            32,
            kernel_size=5,
            dilation=2,
            padding=4,
            groups=32,
            bias=False,
        )
        self.mix = nn.Conv1d(32, 32, kernel_size=1, bias=False)
        self.output = nn.Conv1d(32, 1, kernel_size=1, bias=True)
        self.activation = nn.SiLU()
        _kaiming(self.depthwise)
        _kaiming(self.mix)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[1] != 32:
            raise ValueError(
                f"E8 temporal decoder 期望 (B,32,T)，实际 {tuple(features.shape)}"
            )
        residual = self.activation(self.depthwise(features))
        residual = self.activation(self.mix(residual))
        return self.output(residual)


@dataclass(frozen=True)
class ArmSpec:
    arm: str
    condition_refiner: str
    decoder: str
    trainable_parameters: int
    factor_covered_macs: int

    @property
    def factors(self) -> tuple[str, str]:
        return self.condition_refiner, self.decoder


def _parameter_count(condition_refiner: str, decoder: str) -> int:
    return (
        W0_MODEL_PARAMETERS
        - W0_CONDITION_REFINER_PARAMETERS
        - W0_DECODER_RESIDUAL_PARAMETERS
        + CONDITION_PARAMETERS[condition_refiner]
        + DECODER_PARAMETERS[decoder]
    )


def _covered_macs(condition_refiner: str, decoder: str) -> int:
    return CONDITION_COVERED_MACS[condition_refiner] + DECODER_COVERED_MACS[decoder]


ARM_SPECS: dict[str, ArmSpec] = {}
for _condition_refiner in CONDITION_REFINERS:
    for _decoder in DECODERS:
        _arm = f"e8_{_condition_refiner}_{_decoder}"
        ARM_SPECS[_arm] = ArmSpec(
            arm=_arm,
            condition_refiner=_condition_refiner,
            decoder=_decoder,
            trainable_parameters=_parameter_count(_condition_refiner, _decoder),
            factor_covered_macs=_covered_macs(_condition_refiner, _decoder),
        )

ARMS = tuple(ARM_SPECS)
REFERENCE_ARM = "e8_fill65_pointwise"
RECOMMENDED_ARM = "e8_direct_temporal"


class E8FilmDecoderRedesignModel(CRDTfV1Model):
    """完整构造 W0 后，仅替换预注册的条件末端与解码残差。"""

    def __init__(self, arm: str, initialization_seed: int) -> None:
        if arm not in ARM_SPECS:
            raise ValueError(f"未知 E8 arm={arm!r}")
        self.arm_spec = ARM_SPECS[arm]
        super().__init__("crd_tf102_w", int(initialization_seed))

        branch = self.branches["w"]
        condition_refiner = self.arm_spec.condition_refiner
        if condition_refiner == "direct":
            branch.parameter_fill = nn.Identity()
        elif condition_refiner in {"res96", "res192"}:
            hidden_channels = 96 if condition_refiner == "res96" else 192
            with module_seed(int(initialization_seed), f"e8_condition_{condition_refiner}"):
                branch.parameter_fill = ChannelResidualRefiner(hidden_channels)

        decoder = self.arm_spec.decoder
        if decoder == "single":
            self.base.decoder_residual = SingleReadoutResidual()
        elif decoder == "temporal":
            with module_seed(int(initialization_seed), "e8_decoder_temporal"):
                self.base.decoder_residual = TemporalDecoderResidual()

        self.experiment_arm = arm
        self._validate_e8_contract()

    def _validate_e8_contract(self) -> None:
        spec = self.arm_spec
        branch = self.branches["w"]
        condition_matches = {
            "fill65": branch.parameter_fill.__class__.__name__ == "_ActiveParameterFill",
            "direct": isinstance(branch.parameter_fill, nn.Identity),
            "res96": isinstance(branch.parameter_fill, ChannelResidualRefiner)
            and branch.parameter_fill.hidden_channels == 96,
            "res192": isinstance(branch.parameter_fill, ChannelResidualRefiner)
            and branch.parameter_fill.hidden_channels == 192,
        }
        decoder_matches = {
            "pointwise": isinstance(self.base.decoder_residual, DecoderResidual),
            "single": isinstance(self.base.decoder_residual, SingleReadoutResidual),
            "temporal": isinstance(self.base.decoder_residual, TemporalDecoderResidual),
        }
        final_projection = branch.final_projection
        if (
            self.tf_variant != "crd_tf102_w"
            or self.representations != ("w",)
            or self.gamma_coefficient != 0.5
            or self.beta_coefficient != 0.5
            or not condition_matches[spec.condition_refiner]
            or not decoder_matches[spec.decoder]
            or final_projection.in_channels != 96
            or final_projection.out_channels != 192
            or not bool(final_projection.weight.detach().eq(0).all())
            or not bool(final_projection.bias.detach().eq(0).all())
            or trainable_parameter_count(self) != spec.trainable_parameters
        ):
            raise RuntimeError(f"E8 模型合同错误: {spec.arm}")
        if spec.decoder in {"pointwise", "temporal"}:
            decoder_output = self.base.decoder_residual.output
            if not bool(decoder_output.weight.detach().eq(0).all()) or not bool(
                decoder_output.bias.detach().eq(0).all()
            ):
                raise RuntimeError(f"E8 decoder 末层未零初始化: {spec.arm}")

    @property
    def factor_state_prefixes(self) -> dict[str, tuple[str, ...]]:
        return {
            "condition_refiner": ("branches.w.parameter_fill.",),
            "decoder": ("base.decoder_residual.",),
        }


def model_contract(arm: str) -> dict[str, Any]:
    if arm not in ARM_SPECS:
        raise ValueError(f"未知 E8 arm={arm!r}")
    spec = ARM_SPECS[arm]
    return {
        "protocol": PROTOCOL,
        "arm": arm,
        "condition_refiner": spec.condition_refiner,
        "decoder": spec.decoder,
        "condition_width": {
            "fill65": 65,
            "direct": None,
            "res96": 96,
            "res192": 192,
        }[spec.condition_refiner],
        "decoder_residual_receptive_field_tokens": {
            "pointwise": 1,
            "single": 0,
            "temporal": 9,
        }[spec.decoder],
        "scale_aggregation": "mean",
        "film_projection": "conv1d_96_to_192_zero_init",
        "film_bounds": {"gamma": 0.5, "beta": 0.5, "activation": "tanh"},
        "waveform_generation_hz": 10,
        "upsampling": "fourier_10_to_100_hz",
        "trainable_parameters": spec.trainable_parameters,
        "factor_covered_macs": spec.factor_covered_macs,
    }


def build_e8_film_decoder_redesign_model(cfg: Any) -> E8FilmDecoderRedesignModel:
    from omegaconf import OmegaConf

    contract = cfg.model.get("e8_film_decoder_redesign_v1")
    if not OmegaConf.is_config(contract):
        raise ValueError("缺少 E8 模型合同")
    arm = str(contract.arm)
    if (
        str(cfg.protocol.name) != PROTOCOL
        or str(cfg.model.name) != "crd_v1"
        or str(cfg.model.variant) != "crd_tf102_w"
        or OmegaConf.to_container(contract, resolve=True) != model_contract(arm)
    ):
        raise ValueError("E8 模型只允许独立科学配置入口")
    return E8FilmDecoderRedesignModel(arm, int(cfg.model.initialization_seed))
