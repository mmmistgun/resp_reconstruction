"""M4 高频时间结构干预；仅用于冻结模型的 eval/no_grad 推理。"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

import numpy as np
import torch
from torch import nn
from scipy.signal import periodogram

from resp_train.paper_evidence.e4_r3_norm import (
    native_stats, frozen_norm, replay_norm, check_formula,
)
from scripts.p1_components_model import band_indices, ResidualAdd

SEEDS = (20260811, 20260812, 20260813)
TRANSFORMS = ("FULL", "H_SHIFT_1", "H_SHIFT_2", "H_SHIFT_3", "H_MEAN")
CONDITIONS = tuple(f"{t}__{m}" for t in TRANSFORMS for m in ("NAT", "FIXED"))
SHIFT_SEED = 20261009


def finite(value):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError("高频干预出现非有限张量")


def row_shifts(row_ids):
    """偏移由 row identity 决定，跨 seed、batch、执行顺序保持一致。"""
    result = []
    for row in row_ids:
        if int(row) < 0:
            raise ValueError("非法 dataset_row_id")
        digest = hashlib.sha256(f"m4-hf-v1:{SHIFT_SEED}:{int(row)}".encode()).digest()
        rng = np.random.Generator(np.random.PCG64(int.from_bytes(digest[:16], "little")))
        result.append(rng.choice(np.arange(60, 301), size=3, replace=False))
    return np.asarray(result, dtype=np.int64).reshape(-1, 3)


def transform(w, frequencies, kind, shifts):
    if kind not in TRANSFORMS or w.ndim != 3 or tuple(w.shape[1:]) != (97, 360):
        raise ValueError("非法干预或 W shape")
    finite(w)
    indices = torch.as_tensor(band_indices(frequencies, "H"), device=w.device)
    if kind == "FULL":
        return w
    changed = w.clone()
    high = w.index_select(1, indices)
    if kind == "H_MEAN":
        changed[:, indices] = high.mean(-1, keepdim=True)
    else:
        shifts = torch.as_tensor(shifts, device=w.device)
        if shifts.shape != (len(w), 3) or shifts.dtype != torch.int64:
            raise ValueError("偏移要求 int64 B×3")
        if bool(((shifts < 60) | (shifts > 300)).any()):
            raise ValueError("偏移必须为 30–150 秒")
        chosen = shifts[:, int(kind[-1]) - 1]
        index = (torch.arange(360, device=w.device)[None] - chosen[:, None]) % 360
        changed[:, indices] = high.gather(2, index[:, None].expand(-1, len(indices), -1))
    finite(changed)
    return changed


@dataclass
class Trace:
    stats: dict = field(default_factory=dict)
    used: dict = field(default_factory=dict)
    points: dict = field(default_factory=dict)
    formula_error: dict = field(default_factory=dict)


def traced_forward(model, x, w, *, baseline=None):
    """临时 hook 原始模块，固定条件 GN，捕获 z/context/u；不改 state dict。"""
    if model.training or torch.is_grad_enabled():
        raise RuntimeError("干预只允许 eval/no_grad")
    if not isinstance(model.fusion, ResidualAdd) or model.spec.band != "W-full":
        raise ValueError("要求原始 Full CWT residual-add 模型")
    norms = {name: module for name, module in model.condition.named_modules()
             if isinstance(module, nn.GroupNorm)}
    if tuple(norms) != ("encoder.2",):
        raise ValueError("M4 条件 GroupNorm 拓扑变化")
    if baseline is not None and set(baseline.stats) != set(norms):
        raise ValueError("缺少同窗口 FULL GN 统计")
    trace, handles = Trace(), []

    def norm_hook(name, module):
        def hook(_, args, output):
            natural = native_stats(module, args[0], output)
            selected = natural if baseline is None else baseline.stats[name]
            trace.stats[name], trace.used[name] = natural, selected
            check_formula(output, frozen_norm(module, args[0], natural, output.dtype))
            result = output if baseline is None else replay_norm(module, args[0], selected, natural, output)
            expected = frozen_norm(module, args[0], selected, output.dtype)
            check_formula(result, expected)
            trace.formula_error[name] = float((result.float() - expected.float()).abs().max())
            return result
        return hook

    def capture(name):
        def hook(_, args, output):
            finite(output)
            trace.points[name] = output.detach()
        return hook

    try:
        for name, module in norms.items():
            handles.append(module.register_forward_hook(norm_hook(name, module)))
        for name, module in (("z", model.waveform_encoder), ("context", model.condition), ("u", model.fusion)):
            handles.append(module.register_forward_hook(capture(name)))
        output = model(x, tf={"w": w})["waveform"]
        finite(output)
    finally:
        for handle in handles:
            handle.remove()
    if set(trace.points) != {"z", "context", "u"} or set(trace.stats) != set(norms):
        raise RuntimeError("机制路径捕获不完整")
    trace.points["residual"] = trace.points["u"].float() - trace.points["z"].float()
    return output, trace


def paired_batch(model, x, w, frequencies, shifts):
    """每个 batch 先建立原生 FULL，逐条件产出；调用方及时落盘，限制内存。"""
    baseline_output, baseline = traced_forward(model, x, w)
    # 验证 hook 未改变原生预测；每批都核验，尾批同样覆盖。
    plain = model(x, tf={"w": w})["waveform"]
    torch.testing.assert_close(baseline_output, plain, rtol=0, atol=0)
    for condition in CONDITIONS:
        kind, mode = condition.split("__")
        changed = transform(w, frequencies, kind, shifts)
        if condition == "FULL__NAT":
            output, trace = baseline_output, baseline
        else:
            output, trace = traced_forward(model, x, changed, baseline=baseline if mode == "FIXED" else None)
        torch.testing.assert_close(trace.points["z"], baseline.points["z"], rtol=0, atol=0)
        if kind == "FULL":
            torch.testing.assert_close(output, baseline_output, rtol=0, atol=0)
            for name in trace.points:
                torch.testing.assert_close(trace.points[name], baseline.points[name], rtol=0, atol=0)
        if mode == "FIXED":
            for name in baseline.stats:
                torch.testing.assert_close(trace.used[name].mean, baseline.stats[name].mean, rtol=0, atol=0)
                torch.testing.assert_close(trace.used[name].rstd, baseline.stats[name].rstd, rtol=0, atol=0)
        yield condition, output, trace, baseline, changed


def modulation_psd(w):
    """载频逐尺度去 DC 的 Hann 周期图；常量尺度显式标无效，不赋伪零谱。"""
    w = np.asarray(w, dtype=np.float64)
    if w.ndim != 2 or w.shape != (97, 360) or not np.isfinite(w).all():
        raise ValueError("调制谱要求有限 97×360 W")
    frequency, psd = periodogram(w, fs=2., window="hann", detrend="constant", axis=-1)
    if not np.isfinite(psd).all():
        raise FloatingPointError("调制谱计算出现非有限值")
    psd[:, 0] = 0.
    total = psd.sum(-1)
    valid = total > 1e-20
    normalized = np.full_like(psd, np.nan)
    normalized[valid] = psd[valid] / total[valid, None]
    if not np.isfinite(normalized[valid]).all():
        raise FloatingPointError("有效尺度的归一化调制谱非有限")
    return frequency, normalized, valid


def select_cases(rows):
    """按冻结 row 顺序选每人的首/中/末窗，不读取预测或干预效果。"""
    if rows.dataset_row_id.duplicated().any():
        raise ValueError("案例源含重复 row")
    selected = []
    for _, group in rows.groupby("samp_id", sort=True):
        positions = sorted({0, len(group) // 2, len(group) - 1})
        selected.extend(group.iloc[positions].dataset_row_id.astype(int))
    return selected
