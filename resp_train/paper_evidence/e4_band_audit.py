"""E4 频带诊断：保持特征的聚合干预，以及卷积前的入口干预。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from resp_train.paper_evidence.e4_aggregation_v2_model import centered_weights
from resp_train.paper_evidence.e4_scale_aggregation_model import REGIONS
from resp_train.paper_evidence.e1_scale_topology import PRIMARY, PCC

CONDITIONS = ("FULL", "UNIFORM", *(f"RESET_R{k}" for k in range(4)),
              *(f"INPUT_{reference}_R{k}" for reference in ("MEAN", "ZERO") for k in range(4)))


def training_reference(values):
    if values.ndim != 3 or values.shape[1:] != (97,360) or not len(values):
        raise ValueError("训练参考必须是非空 N×97×360")
    total = np.zeros(97,dtype=np.float64)
    for start in range(0,len(values),64):
        block=np.asarray(values[start:start+64],dtype=np.float64)
        if not np.isfinite(block).all():
            raise FloatingPointError("训练参考来源包含非有限值")
        total+=block.sum(axis=(0,2))
    return (total/(len(values)*360)).astype(np.float32)


def finite(value):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError("E4 band audit 非有限 tensor")


def scale_weights(branch, x):
    """返回实际槽位概率及数值稳定的先验修正；区域概率保留逐通道差别。"""
    finite(x)
    if x.ndim != 4 or tuple(x.shape[1:]) != (96, 97, 360):
        raise ValueError("聚合特征必须为 B×96×97×360")
    module = getattr(branch, "aggregation", None)
    if module is None:
        alpha = x.new_full((1, 1, 97, 1), 1/97, dtype=torch.float32)
        return alpha, torch.zeros_like(alpha)
    if module.arm in ("static_scale", "channel_region"):
        logits = module.logits
    else:
        hidden = module.content(x)
        if module.arm == "frequency_attention":
            hidden = hidden + (module.frequency_weight * module.frequency_coordinate).to(hidden.dtype)
        logits = module.score(F.silu(hidden))
    finite(logits)
    delta = centered_weights(logits, module.prior_counts)
    alpha = torch.softmax(logits.float() + module.prior_counts.log(), dim=2)
    if module.arm == "channel_region":
        # 区域质量分摊回槽位；不平均通道，也不把四区误作等权。
        alpha = torch.cat([alpha[:, :, k:k+1].expand(-1, -1, stop-start, -1)/(stop-start)
                           for k, (start, stop) in enumerate(REGIONS)], dim=2)
        delta = torch.cat([delta[:, :, k:k+1].expand(-1, -1, stop-start, -1)/(stop-start)
                           for k, (start, stop) in enumerate(REGIONS)], dim=2)
    return alpha, delta


def reset_band(alpha, region):
    """区内每个槽位恢复 1/97；区外保持比例，按样本/通道/时间分别守恒。"""
    if region not in range(4):
        raise ValueError("region 必须为 0..3")
    finite(alpha)
    if alpha.shape[2] != 97 or bool((alpha < 0).any()):
        raise ValueError("尺度权重非法")
    torch.testing.assert_close(alpha.sum(2), torch.ones_like(alpha.sum(2)), atol=2e-6, rtol=0)
    start, stop = REGIONS[region]
    mask = torch.ones(97, dtype=torch.bool, device=alpha.device)
    mask[start:stop] = False
    remaining = alpha[:, :, mask].sum(2, keepdim=True)
    if bool((remaining <= 0).any()):
        raise FloatingPointError("区外概率为零，无法保持原比例；禁止 epsilon 或均匀替代")
    changed = alpha.clone()
    changed[:, :, mask] *= (1-(stop-start)/97)/remaining
    changed[:, :, start:stop] = 1/97
    finite(changed)
    torch.testing.assert_close(changed.sum(2), torch.ones_like(changed.sum(2)), atol=2e-6, rtol=0)
    return changed


def aggregate(branch, x, condition):
    mean = x.mean(2)
    if condition == "FULL":
        return branch.aggregation(x) if hasattr(branch, "aggregation") else mean
    if condition == "UNIFORM" or (condition.startswith("RESET_R") and not hasattr(branch, "aggregation")):
        return mean
    if condition not in CONDITIONS or not condition.startswith("RESET_R"):
        raise ValueError("不是聚合干预条件")
    alpha, _ = scale_weights(branch, x)
    changed = reset_band(alpha, int(condition[-1]))
    if branch.aggregation.arm == "channel_region":
        regional = torch.stack([x[:,:,start:stop].mean(2) for start,stop in REGIONS],dim=2)
        masses = torch.cat([changed[:,:,start:stop].sum(2,keepdim=True) for start,stop in REGIONS],dim=2)
        prior = branch.aggregation.prior_counts/97
        return mean + ((masses-prior)*regional.float()).sum(2).to(mean.dtype)
    return mean + ((changed-1/97)*x.float()).sum(2).to(mean.dtype)


def replace_input(w, condition, reference):
    if condition not in CONDITIONS or not condition.startswith("INPUT_"):
        raise ValueError("不是入口干预条件")
    finite(w); finite(reference)
    if w.ndim != 3 or tuple(w.shape[1:]) != (97, 360) or reference.shape != (97,):
        raise ValueError("入口 shape 不符")
    start, stop = REGIONS[int(condition[-1])]
    changed = w.clone()
    replacement = reference[start:stop] if condition.startswith("INPUT_MEAN") else torch.zeros_like(reference[start:stop])
    changed[:, start:stop] = replacement[None, :, None].to(w)
    return changed


class AuditModel(nn.Module):
    """仅在一次 forward 内替换 W 分支计算；其余 temporal/FiLM/主干原样调用。"""
    def __init__(self, model, condition, *, reference=None, fixed_features=None, observe=None):
        super().__init__()
        if condition not in CONDITIONS:
            raise ValueError("未知条件")
        self.model, self.condition = model, condition
        self.reference, self.fixed_features, self.observe = reference, fixed_features, observe
        self.offset = 0

    def forward(self, value, *, tf, **kwargs):
        if self.training or self.model.training or torch.is_grad_enabled():
            raise RuntimeError("诊断只允许 eval + no_grad")
        branch = self.model.branches["w"]
        original = branch.forward
        captured = {}
        count = len(value)
        if self.condition in CONDITIONS[1:6] and self.fixed_features is None:
            raise ValueError("聚合干预必须重用 FULL 的固定 X")
        features = tf
        if self.condition.startswith("INPUT_"):
            features = dict(tf, w=replace_input(tf["w"], self.condition, self.reference.to(tf["w"].device)))

        def forward_w(mapping):
            if self.fixed_features is not None:
                x = self.fixed_features(self.offset, count).to(value.device)
            else:
                w = mapping["w"].float()[:, None]
                x = F.silu(branch.conv_out(branch.depthwise(F.silu(branch.norm(branch.conv_in(w))))))
            finite(x)
            local_condition = "FULL" if self.condition.startswith("INPUT_") else self.condition
            z = aggregate(branch, x, local_condition)
            if self.observe is not None:
                alpha, delta = scale_weights(branch, x)
                captured.update(x=x.detach(), z=z.detach(), alpha=alpha.detach(), delta=delta.detach())
            return branch.project_condition(branch.temporal(F.interpolate(z, size=1800, mode="linear", align_corners=False)))

        branch.forward = forward_w
        try:
            result = self.model(value, tf=features, **kwargs)
        finally:
            branch.forward = original
        for tensor in result.values():
            finite(tensor)
        if self.observe is not None:
            self.observe(self.offset, captured)
        self.offset += count
        return result


def paired_changes(full, changed, arm, seed, condition):
    """先配对窗口，再分别报告 pooled 与逐 samp_id；保留 target-only 分母。"""
    if len(full) != len(changed):
        raise ValueError("配对行数不一致")
    for key in ("dataset_row_id", "samp_id", "split", "whole_rr_target_eligible", "local_rr_target_eligible",
                "local_rr_target_eligible_windows", "joint_target_eligible", "envelope_spearman_target_eligible"):
        if not np.array_equal(full[key], changed[key]):
            raise ValueError(f"干预改变 identity/target eligibility: {key}")
    records = []
    groups = [("pooled", "ALL", np.ones(len(full), dtype=bool))]
    groups += [("samp_id", str(s), full.samp_id.eq(s).to_numpy()) for s in sorted(full.samp_id.unique())]
    for scope, group, mask in groups:
        for metric in PRIMARY:
            baseline = full.loc[mask, metric].to_numpy(dtype=float)
            perturbed = changed.loc[mask, metric].to_numpy(dtype=float)
            valid = np.isfinite(baseline)
            if not np.array_equal(valid, np.isfinite(perturbed)):
                raise ValueError("干预改变指标有效样本集合")
            n = int(valid.sum())
            b, p = (float(v[valid].mean()) if n else None for v in (baseline, perturbed))
            delta = (b-p if metric == PCC else p-b) if n else None
            records.append(dict(arm=arm, seed=seed, condition=condition, scope=scope, group=group,
                metric=metric, n=n, full=b, intervened=p, degradation=delta,
                relative_degradation_pct=100*delta/b if n and b != 0 and metric != PCC else None))
    return pd.DataFrame(records)
