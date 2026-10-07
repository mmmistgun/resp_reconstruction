"""预先固定的 FP32 容差和逐 trajectory/step 的加速路径数值验收。"""

import numpy as np
import torch

from .sampling import ensemble_prefixes, sample_trajectory
from .sampling_accelerated import (
    AccelerationSpec, accelerated_prefixes, encode_condition, grouped_noise, sample_group,
)

# 验证前固定；低 SNR 的 ε→x0 放大使 reverse 不能沿用 ε 的绝对容差。
TOLERANCES = {
    "epsilon": {"atol": 1e-4, "rtol": 1e-3},
    "reverse": {"atol": .02, "rtol": 1e-3},
    "trajectory": {"atol": .02, "rtol": 1e-3},
    "prefix": {"atol": .02, "rtol": 1e-3},
    "raw_waveform": {"atol": .02, "rtol": 1e-3},
    "postfiltered_waveform": {"atol": .02, "rtol": 1e-3},
    "initial_noise": {"atol": 0., "rtol": 0.},
    "reverse_noise": {"atol": 0., "rtol": 0.},
}


def error_record(actual, reference, stage, **identity):
    if isinstance(actual, torch.Tensor):
        actual = actual.detach().cpu().numpy()
    if isinstance(reference, torch.Tensor):
        reference = reference.detach().cpu().numpy()
    actual, reference = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if actual.shape != reference.shape:
        raise ValueError(f"{stage} shape 不匹配")
    finite = bool(np.isfinite(actual).all() and np.isfinite(reference).all())
    if not finite:
        raise FloatingPointError(f"{stage} 非有限输出")
    difference = abs(actual - reference)
    tolerance = TOLERANCES[stage]
    bound = tolerance["atol"] + tolerance["rtol"] * abs(reference)
    ratio = np.divide(difference, bound, out=np.zeros_like(difference), where=bound != 0)
    ratio[(bound == 0) & (difference != 0)] = np.inf
    passed = bool(np.all(difference <= bound))
    record = {**identity, "stage": stage, "finite": finite, "passed": passed,
        "max_abs_error": float(difference.max()), "rms_error": float(np.sqrt((difference ** 2).mean())),
        "max_tolerance_fraction": float(ratio.max()) if np.isfinite(ratio).all() else None,
        "n_outside_tolerance": int((difference > bound).sum()), **tolerance}
    return record


def reference_trace(model, condition, keys, spec, trajectory):
    """用原 sampler 的 module hook 记录 ε 和实际下一步输入，而非重写参考更新。"""
    records = []
    def capture(module, inputs, output):
        records.append({"timestep": int(inputs[2][0]),
            "current": inputs[1].detach().cpu().clone(), "epsilon": output.detach().cpu().clone()})
    handle = model.diffusion_model.register_forward_hook(capture)
    try:
        final = sample_trajectory(model, condition, keys, spec, trajectory).cpu()
    finally:
        handle.remove()
    if [record["timestep"] for record in records] != list(spec.timesteps):
        raise ValueError("参考 sampler 的实际 step 序列不符")
    for index, record in enumerate(records):
        record["reverse"] = records[index + 1]["current"] if index + 1 < len(records) else final
    return records, final


@torch.inference_mode()
def verify_grouped(model, condition, keys, spec, acceleration, *, counts=None):
    """参考 trace 仅驻留当前 G 条轨迹；验收 ε/reverse/每条终态和每个 prefix。"""
    counts = tuple(counts or (spec.n_trajectories,))
    references = {count: value.cpu() for count, value in ensemble_prefixes(model, condition, keys, spec,
                                                                         checkpoints=counts)}
    records = []
    cache = encode_condition(model.diffusion_model, condition)
    total = torch.zeros_like(condition, device="cpu", dtype=torch.float64)
    size = acceleration.trajectory_group_size
    for first in range(0, spec.n_trajectories, size):
        ids = tuple(range(first, min(first + size, spec.n_trajectories)))
        traces = {trajectory: reference_trace(model, condition, keys, spec, trajectory) for trajectory in ids}
        def observe(event):
            step_index = spec.timesteps.index(event["timestep"])
            identity = {"G": size, "timestep": event["timestep"]}
            for offset, trajectory in enumerate(ids):
                reference = traces[trajectory][0][step_index]
                for stage, key in (("epsilon", "epsilon"), ("reverse", "reverse")):
                    records.append(error_record(event[key][offset], reference[key], stage,
                                                trajectory_id=trajectory, **identity))
                if step_index == 0:
                    records.append(error_record(event["current"][offset], reference["current"], "initial_noise",
                                                trajectory_id=trajectory, **identity))
            if event["reverse_noise"] is not None:
                if event["timestep"]:
                    expected = grouped_noise(keys, ids, spec, "cpu", reverse_step=event["timestep"])
                else:
                    expected = torch.zeros(event["reverse_noise"].shape, dtype=torch.float32)
                records.append(error_record(event["reverse_noise"], expected, "reverse_noise", **identity))
        final = sample_group(model, condition, keys, spec, ids, cache, observer=observe)
        for offset, trajectory in enumerate(ids):
            value = final[offset].cpu()
            records.append(error_record(value, traces[trajectory][1], "trajectory", G=size, trajectory_id=trajectory))
            total.add_(value)
            count = trajectory + 1
            if count in counts:
                mean = (total / count).float()
                records.append(error_record(mean, references[count], "prefix", G=size, N=count))
        del final, traces
    return records
