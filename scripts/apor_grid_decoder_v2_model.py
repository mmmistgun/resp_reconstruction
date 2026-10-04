"""APOR v2：公共Patch网格上的调制后时间细化，保留v1受管源码。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from omegaconf import OmegaConf

from resp_train.crd.initialization import module_seed
from resp_train.crd.tf_v1_model import _initialize_conv, _checkpointed_mapping_branch
from resp_train.paper_evidence.patch_apor_v1_model import PatchAporModel, model_contract as original_contract

PROTOCOL = "apor-grid-decoder-v2-20260930"
SEEDS = (20260811, 20260812, 20260813)


@dataclass(frozen=True)
class ArmSpec:
    source_arm: str
    temporal: bool
    condition: bool = True


ARM_SPECS = {
    "N0": ArmSpec("A0", False), "N1": ArmSpec("A0", True),
    "D0": ArmSpec("A3", False), "D1": ArmSpec("A3", True),
}
ARMS = tuple(ARM_SPECS)
TRAIN_ARMS = ("N1", "D1")
REFERENCE_ARMS = ("N0", "D0")


def model_contract(arm):
    spec = ARM_SPECS[arm]
    return {
        "protocol": PROTOCOL, "arm": arm, "source": original_contract(spec.source_arm),
        "parameters": 1115080 if spec.temporal else 1077640,
        "postfilm": {
            "enabled": spec.temporal, "grid": "patch_centers_140", "groups": 12,
            "channels": 96, "hidden": 192, "kernel": 3, "dilation": 1,
            "dropout": 0.0, "activation": "silu", "zero_output": True,
        },
    }


class PostFilmTemporal(nn.Module):
    """在公共140点网格上处理调制后的特征；零末端保证初始恒等。"""

    def __init__(self):
        super().__init__()
        self.norm = nn.GroupNorm(12, 96, eps=1e-5, affine=True)
        self.depthwise = nn.Conv1d(96, 96, 3, padding=1, groups=96, bias=False)
        self.expand = nn.Conv1d(96, 192, 1, bias=False)
        self.project = nn.Conv1d(192, 96, 1, bias=True)
        self.activation = nn.SiLU()
        for layer in (self.depthwise, self.expand):
            _initialize_conv(layer)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, value):
        if value.ndim != 3 or value.shape[1:] != (96, 140):
            raise ValueError("调制后时间细化要求(B,96,140)")
        residual = self.activation(self.depthwise(self.norm(value)))
        return value + self.project(self.activation(self.expand(residual)))


class GridDecoderModel(PatchAporModel):
    def __init__(self, arm, seed):
        spec = ARM_SPECS[arm]
        super().__init__(spec.source_arm, seed)
        self.study_arm = arm
        if spec.temporal:
            with module_seed(seed, "apor_grid_decoder_v2_postfilm_temporal"):
                self.postfilm = PostFilmTemporal()
        else:
            self.postfilm = nn.Identity()
        if sum(p.numel() for p in self.parameters()) != model_contract(arm)["parameters"]:
            raise RuntimeError("v2参数数量与合同不符")

    def forward(self, x, *, tf=None, **kwargs):
        if self.study_arm in REFERENCE_ARMS:
            return super().forward(x, tf=tf, **kwargs)
        if x.ndim != 3 or x.shape[1:] != (1, 18000) or x.shape[0] < 1:
            raise ValueError("输入要求(B,1,18000)")
        if not bool(torch.isfinite(x).all()):
            raise FloatingPointError("输入包含NaN/Inf")
        if tf is None or set(tf) != {"w"} or tf["w"].shape[0] != x.shape[0]:
            raise ValueError("条件输入要求同batch的w")
        frontend = self.base.frontend
        value = frontend.encoder(x)
        if self.spec.dense_trunk:
            value = self.to_dense(value)
        latent = frontend.activation(frontend.norm(frontend.adapter(value)))
        tokens = latent.transpose(1, 2)
        for block in self.base.local_blocks:
            tokens = block(tokens)
        latent = tokens.transpose(1, 2)
        if self.spec.dense_trunk:
            latent = self.to_patch(latent)
        gamma, beta = _checkpointed_mapping_branch(self.branches["w"], tf)
        latent = latent * (1 + 0.5 * torch.tanh(gamma)) + 0.5 * torch.tanh(beta)
        output = self.overlap(self.patch_head(self.postfilm(latent).transpose(1, 2)))
        if not bool(torch.isfinite(output).all()):
            raise FloatingPointError("模型输出包含NaN/Inf")
        return {"waveform": output}


def build_model(cfg):
    contract = OmegaConf.to_container(cfg.model.apor_grid_decoder_v2, resolve=True)
    arm = contract["arm"]
    if cfg.protocol.name != PROTOCOL or contract != model_contract(arm):
        raise ValueError("v2模型合同不符")
    return GridDecoderModel(arm, int(cfg.model.initialization_seed))
