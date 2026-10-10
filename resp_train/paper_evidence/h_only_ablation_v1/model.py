"""从同 seed 的 H65 初始模型独立派生各消融，保留公共 tensor。"""
from dataclasses import replace
import torch
from torch import nn
from torch.nn import functional as F
from resp_train.crd.initialization import module_seed
from resp_train.crd.tf_v1_model import _ActiveParameterFill, _require_feature, _checkpointed_mapping_branch
from resp_train.paper_evidence.patch_apor_v1_model import PatchAporModel, PatchCondition, CoordinateSample, OverlapAdd
from .spec import ARMS


def gain(raw, mode):
    if mode == "bounded":
        return 1 + .5 * torch.tanh(raw)
    if mode == "positive":
        return 1 + torch.tanh(raw / 2)
    if mode == "signed":
        return 1 + 2 * torch.tanh(raw / 4)
    if mode == "unbounded":
        return 1 + .5 * raw
    raise ValueError("未知增益参数化")


class AblatedMixer(nn.Module):
    """删除整个残差子层，包括其归一化；保留子层沿用原对象和名称。"""
    def __init__(self, source, spec):
        super().__init__()
        if spec.patch_mixing:
            self.norm1, self.patch_mixer = source.norm1, source.patch_mixer
        if spec.channel_mixing:
            self.norm2, self.channel_mixer = source.norm2, source.channel_mixer

    def forward(self, x):
        if hasattr(self, "patch_mixer"):
            x = x + self.patch_mixer(self.norm1(x))
        if hasattr(self, "channel_mixer"):
            x = x + self.channel_mixer(self.norm2(x))
        return x


class HCondition(PatchCondition):
    def __init__(self, source, spec, seed):
        # 先复用已有 PatchCondition 对单路投影及尺度卷积的实现，再恢复原五槽投影。
        local = source.local_projection
        super().__init__(source, replace(source.spec, film=spec.film,
                         scale_neighborhood=spec.scale_neighborhood), seed)
        self.local_projection = local
        if spec.hidden == 64:
            original = self.parameter_fill
            with module_seed(seed, "ha_h64_nested"):
                self.parameter_fill = _ActiveParameterFill(64)
            with torch.no_grad():
                self.parameter_fill.expand.weight.copy_(original.expand.weight[:64])
                self.parameter_fill.project.weight.copy_(original.project.weight[:, :64])
                self.parameter_fill.project.bias.copy_(original.project.bias)
        elif spec.hidden == 0:
            self.parameter_fill = nn.Identity()
        offsets = torch.full((5,), 127.5) if spec.center_only else torch.linspace(0, 255, 5)
        self.sample = CoordinateSample(360, 24.5, 50, torch.arange(140)[:, None]*128 + offsets)

    def forward(self, tf):
        value = _require_feature(tf, "w", (41, 360)).float()[:, None]
        value = F.silu(self.norm(self.conv_in(value)))
        value = F.silu(self.conv_out(self.depthwise(value))).mean(dim=2)
        sampled = self.sample(value)
        value = sampled.permute(0, 1, 3, 2).reshape(value.shape[0], 480, 140)
        projected = self.final_projection(self.parameter_fill(self.local_projection(value)))
        if self.spec.film == "add":
            return torch.zeros_like(projected), projected
        if self.spec.film == "scale":
            return projected, torch.zeros_like(projected)
        return projected.chunk(2, dim=1)


class HOnlyModel(PatchAporModel):
    def __init__(self, arm, seed):
        if arm not in ARMS:
            raise ValueError("未知消融臂")
        super().__init__("A0", seed)
        self.arm, self.ablation = arm, ARMS[arm]
        spec = self.ablation
        if spec.condition:
            self.branches["w"] = HCondition(self.branches["w"], spec, seed)
        else:
            self.branches = nn.ModuleDict()
            self.representations = ()
        if not spec.patch_mixing or not spec.channel_mixing:
            self.base.frontend.encoder.blocks = nn.ModuleList(
                [AblatedMixer(b, spec) for b in self.base.frontend.encoder.blocks])
        if not spec.mamba:
            self.base.local_blocks = nn.ModuleList()
            self.base.local_block_count = 0
        if spec.decoder == "linear":
            with module_seed(seed, "apor_decoder_linear"):
                self.patch_head = nn.Linear(96, 256)
        if spec.overlap == "uniform":
            self.overlap = OverlapAdd("uniform")
        self.gain_observer = None
        if arm in ("HA0", "HA12", "HA13", "HA14", "HA15") and self.parameter_count != 1077448:
            raise RuntimeError("H64 参数量漂移")

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x, *, tf=None, **_):
        if x.ndim != 3 or x.shape[1:] != (1, 18000) or len(x) < 1:
            raise ValueError("输入要求 (B,1,18000)")
        if not torch.isfinite(x).all():
            raise FloatingPointError("输入包含非有限值")
        frontend = self.base.frontend
        latent = frontend.activation(frontend.norm(frontend.adapter(frontend.encoder(x))))
        tokens = latent.transpose(1, 2)
        for block in self.base.local_blocks:
            tokens = block(tokens)
        latent = tokens.transpose(1, 2)
        if self.ablation.condition:
            if tf is None or set(tf) != {"w"} or len(tf["w"]) != len(x):
                raise ValueError("条件要求同 batch 的 w")
            a, b = _checkpointed_mapping_branch(self.branches["w"], tf)
            s = gain(a, self.ablation.gain)
            if not torch.isfinite(s).all():
                raise FloatingPointError("FiLM 增益非有限")
            if self.gain_observer is not None:
                self.gain_observer(s.detach(), self.training)
            latent = latent * s + .5 * torch.tanh(b)
        elif tf:
            raise ValueError("HA1 不接收 CWT 条件")
        waveform = self.overlap(self.patch_head(latent.transpose(1, 2)))
        if not torch.isfinite(waveform).all():
            raise FloatingPointError("模型输出非有限")
        return {"waveform": waveform}


def build_model(arm, seed):
    return HOnlyModel(arm, seed)
