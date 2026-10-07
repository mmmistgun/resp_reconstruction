"""当前原始 batch 的 condition 缓存与独立 trajectory 分组；参考 sampler 保留。"""

from dataclasses import dataclass
from collections.abc import Callable

import torch
from torch.nn import functional as F

from resp_train.respdiff.model import finite, waveform
from .sampling import batch_trajectory_noise


@dataclass(frozen=True)
class AccelerationSpec:
    trajectory_group_size: int = 1

    def __post_init__(self):
        if type(self.trajectory_group_size) is not int or self.trajectory_group_size not in (1, 2, 4, 8):
            raise ValueError("trajectory_group_size 只支持 G=1/2/4/8")


@dataclass(frozen=True)
class ConditionFeatures:
    """短生命周期对象，不注册为模型参数/buffer，不跨原始 condition batch 复用。"""
    owner: object
    condition: torch.Tensor
    version: int | None
    snapshot: torch.Tensor | None
    fine: torch.Tensor
    weighted_coarse: torch.Tensor


def _version(tensor):
    return None if tensor.is_inference() else tensor._version


@torch.inference_mode()
def encode_condition(denoiser, condition):
    waveform("condition", condition)
    if denoiser.training or any(module.training for module in denoiser.modules()):
        raise ValueError("condition 缓存仅用于完整 eval mode 的推理")
    # 合批不应改变任何依赖当前 batch 统计的非 condition 分支。
    for branch in (denoiser.noise_encoder, denoiser.diff_model, denoiser.de, denoiser.diffusion_embedding):
        if any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in branch.modules()):
            raise ValueError("非 condition 分支存在 BatchNorm，不能使用 trajectory 合批")
    fine = denoiser.ppg_encoder2(condition)
    weighted_coarse = denoiser.weight * denoiser.ppg_encoder1(condition)
    finite("cached fine condition", fine)
    finite("cached weighted coarse condition", weighted_coarse)
    version = _version(condition)
    snapshot = condition.clone() if version is None else None
    return ConditionFeatures(denoiser, condition, version, snapshot, fine, weighted_coarse)


@torch.inference_mode()
def forward_cached(denoiser, condition, cache, noisy_target, step):
    """condition 按 B 编码；仅 noisy encoder/RNN/decoder 对 G*B 合批。"""
    if cache.owner is not denoiser or cache.condition is not condition or cache.version != _version(condition):
        raise ValueError("condition cache 不属于当前 batch，或 condition 已被原地改变")
    if cache.snapshot is not None and not torch.equal(condition, cache.snapshot):
        raise ValueError("inference tensor condition 已被原地改变，缓存无效")
    if denoiser.training:
        raise ValueError("cached forward 要求 eval mode")
    waveform("noisy_target", noisy_target)
    batch = len(condition)
    if (noisy_target.device != condition.device or noisy_target.shape[1:] != condition.shape[1:]
            or len(noisy_target) % batch):
        raise ValueError("cached forward 输入须为当前 batch 的 G 条 trajectory")
    group = len(noisy_target) // batch
    if group not in (1, 2, 3, 4, 5, 6, 7, 8):
        raise ValueError("当前 group 必须包含 1…8 条 trajectory")
    if (step.dtype != torch.long or step.device != condition.device or step.shape != (len(noisy_target),)
            or torch.any((step < 0) | (step >= 50))):
        raise ValueError("cached step 必须为同 device 的 [G*B] int64，范围 0…49")
    steps = step.reshape(group, batch)
    if not torch.equal(steps, steps[0].expand_as(steps)):
        raise ValueError("同组 trajectory 必须同步 reverse timestep")
    # embedding 也按原始 B 计算，避免无必要的 batch 形状数值变化。
    embedding = denoiser.diffusion_embedding(steps[0]).unsqueeze(2)
    if group == 1:
        f1 = (cache.fine + embedding) + cache.weighted_coarse
        expanded_embedding = embedding
    else:
        f1 = ((cache.fine.unsqueeze(0) + embedding.unsqueeze(0))
              + cache.weighted_coarse.unsqueeze(0)).expand(group, -1, -1, -1)
        f1 = f1.reshape(group * batch, *f1.shape[2:])
        expanded_embedding = embedding.unsqueeze(0).expand(group, -1, -1, -1).reshape(group * batch, -1, 1)
    f2 = denoiser.noise_encoder(noisy_target)
    if denoiser.spec.profile == "source_plain":
        f2 = f2 + expanded_embedding
    features = torch.cat([f1, f2], dim=1).permute(0, 2, 1)
    output = denoiser.de(F.relu(denoiser.diff_model(features))).permute(0, 2, 1)
    finite("cached predicted epsilon", output)
    return output


def grouped_noise(keys, trajectory_ids, spec, device, *, reverse_step=None):
    """trajectory-major 排列；逐条调用历史噪声函数，G 不进入噪声身份。"""
    return torch.stack([batch_trajectory_noise(keys, trajectory_id=trajectory,
        noise_seed=spec.noise_seed, device=device, reverse_step=reverse_step) for trajectory in trajectory_ids])


@torch.inference_mode()
def sample_group(model, condition, keys, spec, trajectory_ids, cache, *, observer: Callable | None = None):
    """仅保存当前组状态；observer 用于短时数值验收，不在性能测量中启用。"""
    ids = tuple(trajectory_ids)
    if (not ids or len(ids) > 8 or list(ids) != list(range(ids[0], ids[0] + len(ids)))
            or ids[0] < 0 or ids[-1] >= spec.n_trajectories):
        raise ValueError("trajectory group 须为预算内连续且有序的 id")
    if model.training or len(keys) != len(condition):
        raise ValueError("sample_group 要求 eval mode 和匹配的 keys")
    current = grouped_noise(keys, ids, spec, condition.device)
    group, batch = current.shape[:2]
    for index, timestep in enumerate(spec.timesteps):
        steps = torch.full((group * batch,), timestep, dtype=torch.long, device=condition.device)
        flat = current.reshape(group * batch, *condition.shape[1:])
        predicted = forward_cached(model.diffusion_model, condition, cache, flat, steps)
        if spec.sampler == "ddim":
            alpha = model.alpha_torch[timestep]
            following = (model.alpha_torch[spec.timesteps[index + 1]]
                         if index + 1 < spec.nfe else torch.ones_like(alpha))
            x0 = (flat - (1 - alpha).sqrt() * predicted) / alpha.sqrt()
            result = following.sqrt() * x0 + (1 - following).sqrt() * predicted
            reverse_noise = None
        else:
            reverse_noise = (grouped_noise(keys, ids, spec, condition.device, reverse_step=timestep)
                             if timestep else torch.zeros_like(current))
            result = model.reverse_step(flat, predicted, step=timestep,
                                        noise=reverse_noise.reshape_as(flat))
        finite(f"group reverse t={timestep}", result)
        if observer is not None:
            observer({"trajectory_ids": ids, "timestep": timestep, "current": current,
                "epsilon": predicted.reshape_as(current), "reverse": result.reshape_as(current),
                "reverse_noise": reverse_noise})
        current = result.reshape(group, batch, *condition.shape[1:])
    return current


def accelerated_groups(model, condition, keys, spec, acceleration=AccelerationSpec(), *, observer=None):
    waveform("condition", condition)
    # 缓存由该 iterator 私有创建与持有，生命周期严格限于一次原始 batch。
    # 数值计算函数各自使用 inference_mode；避免 generator context wrapper
    # 持有上次 yield 的整组 Tensor，导致下一组计算时旧组仍驻留。
    cache = encode_condition(model.diffusion_model, condition)
    size = acceleration.trajectory_group_size
    for first in range(0, spec.n_trajectories, size):
        ids = tuple(range(first, min(first + size, spec.n_trajectories)))
        current = sample_group(model, condition, keys, spec, ids, cache, observer=observer)
        yield ids, current
        del current


@torch.inference_mode()
def accelerated_prefixes(model, condition, keys, spec, acceleration=AccelerationSpec(), *, checkpoints=None,
                         observer=None, completion_observer=None):
    checkpoints = tuple(checkpoints or (spec.n_trajectories,))
    if (tuple(sorted(set(checkpoints))) != checkpoints or not checkpoints
            or checkpoints[-1] != spec.n_trajectories or checkpoints[0] < 1):
        raise ValueError("ensemble checkpoints 须严格递增并包含最大 N")
    total = torch.zeros_like(condition, dtype=torch.float64)
    for ids, current in accelerated_groups(model, condition, keys, spec, acceleration, observer=observer):
        # 组内 trajectory 完成同一轮计算，不能把相同 group 墙钟当成独立 N 耗时。
        if completion_observer is not None:
            completion_observer(ids)
        for offset, trajectory in enumerate(ids):
            total.add_(current[offset])
            count = trajectory + 1
            if count in checkpoints:
                mean = (total / count).to(condition.dtype)
                finite("accelerated ensemble mean", mean)
                yield count, mean
        del current


def execution_budget(spec, acceleration, *, n_batches=1):
    groups = (spec.n_trajectories + acceleration.trajectory_group_size - 1) // acceleration.trajectory_group_size
    return {"logical_denoiser_calls_per_chunk": spec.nfe * spec.n_trajectories,
            "grouped_denoiser_forward_calls_per_batch": spec.nfe * groups,
            "actual_denoiser_forward_calls": spec.nfe * groups * n_batches,
            "condition_encoding_calls_per_batch": 1, "trajectory_group_size": acceleration.trajectory_group_size}
