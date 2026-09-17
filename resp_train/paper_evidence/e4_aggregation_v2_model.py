"""四个尺度聚合候选，共用冻结 W0 的初始化、分支与主干。"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from resp_train.crd.initialization import named_subseed
from resp_train.crd.tf_v1_model import CwtBranch, CRDTfV1Model, _require_feature, trainable_parameter_count
from resp_train.paper_evidence.e4_scale_aggregation_model import (
    REGIONS, FREQUENCY_FILE_SHA, validate_frequency_grid,
)

PROTOCOL = "e4-scale-aggregation-v2-20260917"
ARMS = ("static_scale", "scale_attention", "frequency_attention", "channel_region")
ADDED_PARAMETERS = {"static_scale": 97, "scale_attention": 784, "frequency_attention": 792, "channel_region": 384}
W0_PARAMETERS = 1_219_850
W0_BRANCH_PARAMETERS = 150_048


@contextmanager
def cpu_module_seed(seed: int, name: str):
    """新增 CPU 参数构造只设 CPU generator，保持原 W0 构造后的 CUDA RNG。

    torch.manual_seed 会同时设置 CUDA seed，而 fork_rng(devices=[]) 只恢复 CPU；
    此处直接设置 CPU generator，不触碰 CUDA，也不会为初始化额外打开 GPU context。
    """
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(named_subseed(seed, name))
        yield


def aggregation_contract(arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError(f"E4-v2 未知 arm={arm!r}")
    return {
        "arm": arm, "scale_count": 97, "channels": 96, "context_length": 360,
        "regions": [list(pair) for pair in REGIONS], "region_counts": [25, 24, 24, 24],
        "hidden_channels": 8, "score_activation": "silu", "score_output_bias": False,
        "frequency_coordinate": "log_frequency_minmax_minus1_plus1",
        "frequency_file_sha256": FREQUENCY_FILE_SHA,
        "normalization": "positive_prior_times_exp_stable_logits",
        "output": "global_mean_plus_prior_centered_weight_correction",
        "logit_or_score_output_initialization": "zero", "fill_hidden_channels": 65,
        "added_parameters": ADDED_PARAMETERS[arm],
        "branch_parameters": W0_BRANCH_PARAMETERS + ADDED_PARAMETERS[arm],
        "model_parameters": W0_PARAMETERS + ADDED_PARAMETERS[arm],
    }


def centered_weights(logits: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """等价于 softmax(logits+log(counts)) 减先验；零 logits 时分子精确为零。

    counts 为原始尺度数，保持整数可精确表示，避免先验概率舍入破坏 W0 初值。
    归一化在 FP32 进行，输出修正由调用者转回原 mean 的 dtype。
    """
    scores = logits.float()
    if not bool(torch.isfinite(scores).all()):
        raise FloatingPointError("E4-v2 attention logits 非有限")
    prior = counts.float()
    mass = (scores - scores.amax(dim=2, keepdim=True)).exp() * prior
    total = mass.sum(dim=2, keepdim=True)
    n = prior.sum(dim=2, keepdim=True)
    return (n * mass - prior * total) / (n * total)


class ScaleAggregator(nn.Module):
    def __init__(self, arm: str, initialization_seed: int, frequencies: np.ndarray) -> None:
        super().__init__()
        aggregation_contract(arm)
        validate_frequency_grid(frequencies)
        self.arm = arm
        counts = [stop - start for start, stop in REGIONS] if arm == "channel_region" else [1] * 97
        self.register_buffer("prior_counts", torch.tensor(counts, dtype=torch.float32).reshape(1, 1, -1, 1))
        if arm == "static_scale":
            self.logits = nn.Parameter(torch.zeros(1, 1, 97, 1))
        elif arm == "channel_region":
            self.logits = nn.Parameter(torch.zeros(1, 96, 4, 1))
        else:
            # 两种注意力的公共打分层同 seed 逐 tensor 相同；频率支路不消耗它们的 RNG。
            with cpu_module_seed(initialization_seed, "e4_v2_attention_content"):
                self.content = nn.Conv2d(96, 8, kernel_size=1, bias=True)
                self.score = nn.Conv2d(8, 1, kernel_size=1, bias=False)
                nn.init.zeros_(self.score.weight)
            if arm == "frequency_attention":
                coordinate = np.log(frequencies)
                coordinate = 2 * (coordinate - coordinate[0]) / (coordinate[-1] - coordinate[0]) - 1
                self.register_buffer("frequency_coordinate", torch.tensor(coordinate, dtype=torch.float32).reshape(1, 1, 97, 1))
                with cpu_module_seed(initialization_seed, "e4_v2_frequency_coefficient"):
                    self.frequency_weight = nn.Parameter(torch.empty(1, 8, 1, 1))
                    nn.init.normal_(self.frequency_weight, std=96 ** -0.5)
        if trainable_parameter_count(self) != ADDED_PARAMETERS[arm]:
            raise RuntimeError("E4-v2 聚合器参数数错误")

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1:] != (96, 97, 360):
            raise ValueError("E4-v2 aggregation 期望 (B,96,97,360)")
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError("E4-v2 aggregation input 非有限")
        mean = value.mean(dim=2)
        if self.arm == "channel_region":
            regional = torch.stack([value[:, :, start:stop].mean(dim=2) for start, stop in REGIONS], dim=2)
            correction = (regional.float() * centered_weights(self.logits, self.prior_counts)).sum(dim=2)
        else:
            if self.arm == "static_scale":
                logits = self.logits
            else:
                hidden = self.content(value)
                if self.arm == "frequency_attention":
                    # 坐标在非线性前与内容交互，最后的 bias 会被尺度归一化抵消，故不设置。
                    positional = self.frequency_weight * self.frequency_coordinate
                    hidden = hidden + positional.to(dtype=hidden.dtype)
                logits = self.score(F.silu(hidden))
            correction = (value.float() * centered_weights(logits, self.prior_counts)).sum(dim=2)
        output = mean + correction.to(dtype=mean.dtype)
        if not bool(torch.isfinite(output).all()):
            raise FloatingPointError("E4-v2 aggregation output 非有限")
        return output


class AggregationCwtBranch(CwtBranch):
    def __init__(self, arm: str, initialization_seed: int, frequencies: np.ndarray) -> None:
        super().__init__(scale_count=97)
        self.aggregation = ScaleAggregator(arm, initialization_seed, frequencies)
        if self.parameter_fill.expand.out_channels != 65:
            raise RuntimeError("E4-v2 fill hidden 必须为 65")

    def forward(self, tf) -> tuple[torch.Tensor, torch.Tensor]:
        value = _require_feature(tf, "w", (97, 360)).float()[:, None]
        value = F.silu(self.norm(self.conv_in(value)))
        value = self.aggregation(F.silu(self.conv_out(self.depthwise(value))))
        value = F.interpolate(value, size=1800, mode="linear", align_corners=False)
        return self.project_condition(self.temporal(value))


class AggregationModel(CRDTfV1Model):
    def __init__(self, arm: str, initialization_seed: int, frequencies: np.ndarray) -> None:
        super().__init__("crd_tf102_w", initialization_seed)
        with cpu_module_seed(initialization_seed, "tf_branch_w"):
            self.branches["w"] = AggregationCwtBranch(arm, initialization_seed, frequencies)
        self.experiment_arm = arm
        self._validate_parameter_contract()

    def _validate_parameter_contract(self) -> None:
        if not hasattr(self, "experiment_arm"):
            return super()._validate_parameter_contract()
        contract = aggregation_contract(self.experiment_arm)
        if (trainable_parameter_count(self) != contract["model_parameters"]
                or trainable_parameter_count(self.branches["w"]) != contract["branch_parameters"]
                or len(self.base.local_blocks) != 6 or self.branches["w"].parameter_fill.expand.out_channels != 65):
            raise RuntimeError("E4-v2 模型参数/深度合同错误")


def build_model(cfg) -> AggregationModel:
    from omegaconf import OmegaConf
    if cfg.protocol.name != PROTOCOL or cfg.model.variant != "crd_tf102_w":
        raise ValueError("E4-v2 要求独立科学配置")
    settings = OmegaConf.to_container(cfg.model.aggregation_v2, resolve=True)
    arm = settings["arm"]
    if settings != aggregation_contract(arm):
        raise ValueError("E4-v2 聚合配置漂移")
    frequencies = np.asarray(OmegaConf.to_container(cfg.model.aggregation_frequencies_hz), dtype=np.float64)
    return AggregationModel(arm, int(cfg.model.initialization_seed), frequencies)
