"""APOR/A0原生CWT条件：五位置物理坐标随池化窗口同步变化。"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from resp_train.crd.tf_v1_model import _require_feature
from resp_train.paper_evidence.patch_apor_v1_model import PatchAporModel, CoordinateSample

GN_NAMES = ("norm",)
PARAMETERS = 1077640


class NativePatchCondition(nn.Module):
    def __init__(self, source, scale_count, frames, pool_samples):
        super().__init__()
        if scale_count < 1 or pool_samples not in (25, 50, 100) or frames * pool_samples != 18000:
            raise ValueError("APOR条件尺度/时间几何错误")
        for name, module in source.named_children():
            self.add_module(name, module)
        self.spec = source.spec
        self.scale_count, self.frames, self.pool_samples = scale_count, frames, pool_samples
        positions = torch.arange(140)[:, None] * 128 + torch.linspace(0, 255, 5)[None, :]
        # 原始100-Hz采样索引：bin中心=(pool-1)/2+j*pool，不能只改输入shape。
        self.sample = CoordinateSample(frames, (pool_samples-1)/2, pool_samples, positions)

    def forward(self, tf):
        value = _require_feature(tf, "w", (self.scale_count, self.frames)).float()[:, None]
        value = F.silu(self.norm(self.conv_in(value)))
        value = F.silu(self.conv_out(self.depthwise(value))).mean(dim=2)
        sampled = self.sample(value)
        value = sampled.permute(0, 1, 3, 2).reshape(value.shape[0], 480, 140)
        value = self.local_projection(value)
        return self.final_projection(self.parameter_fill(value)).chunk(2, dim=1)


def build_model(seed, rep):
    model = PatchAporModel("A0", seed)
    before = dict(model.branches["w"].named_parameters())
    branch = NativePatchCondition(model.branches["w"], *rep["shape"], rep["arm"]["pool_samples"])
    for name, parameter in branch.named_parameters():
        if parameter is not before[name]:
            raise RuntimeError("条件几何扩展意外重新初始化参数")
    model.branches["w"] = branch
    if sum(p.numel() for p in model.parameters()) != PARAMETERS:
        raise RuntimeError("APOR/A0参数量漂移")
    return model


@dataclass(frozen=True)
class CapturedFilmBatch:
    z: torch.Tensor
    gamma_raw: torch.Tensor
    beta_raw: torch.Tensor
    z_prime: torch.Tensor


def forward_with_capture(model, x, *, tf):
    """捕获原生A0的主干末端和patch decoder入口，不重写FiLM计算图。"""
    if model.training or torch.is_grad_enabled():
        raise ValueError("APOR捕获仅允许eval/no_grad")
    if model.spec.family != "apor" or model.spec.dense_trunk or model.spec.condition_points != 5:
        raise ValueError("捕获只支持本轮A0图")
    if len(model.base.local_blocks) != 6:
        raise ValueError("APOR主干必须为六层")
    tensors, handles = {}, []
    def latent(_, inputs, output):
        tensors["z"] = output.transpose(1, 2).detach()
    def condition(_, inputs, output):
        tensors["gamma_raw"], tensors["beta_raw"] = (v.detach() for v in output)
    def conditioned(_, inputs):
        tensors["z_prime"] = inputs[0].transpose(1, 2).detach()
    try:
        handles.append(model.base.local_blocks[-1].register_forward_hook(latent))
        handles.append(model.branches["w"].register_forward_hook(condition))
        handles.append(model.patch_head.register_forward_pre_hook(conditioned))
        prediction = model(x, tf=tf)
    finally:
        for handle in handles:
            handle.remove()
    if set(tensors) != {"z", "gamma_raw", "beta_raw", "z_prime"}:
        raise RuntimeError("APOR FiLM捕获不完整")
    if any(v.shape != (len(x), 96, 140) or not torch.isfinite(v).all() for v in tensors.values()):
        raise ValueError("APOR FiLM shape/finite错误")
    return prediction, CapturedFilmBatch(**tensors)
