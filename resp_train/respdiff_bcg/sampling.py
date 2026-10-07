"""保幅 ε sampler：nested parent 噪声、DDIM6/DDPM50 与在线 ensemble。"""

from dataclasses import dataclass
import hashlib
import json

import torch

from resp_train.respdiff.model import finite, waveform
from .baseband import lowpass_parent
from .model import TIMESTEPS
from .signal import STARTS, parent_noise, restore_parent


@dataclass(frozen=True)
class SamplerSpec:
    sampler: str = "ddim"
    nfe: int = 6
    n_trajectories: int = 16
    noise_seed: int = 20261003
    postfilter: bool = True
    eta: float = 0.0

    def __post_init__(self):
        expected = {"ddim": (6, 16), "ddpm": (50, 4)}
        if self.sampler not in expected or self.nfe != expected[self.sampler][0]:
            raise ValueError("只支持 DDIM6 或 DDPM50")
        if type(self.n_trajectories) is not int or not 1 <= self.n_trajectories <= expected[self.sampler][1]:
            raise ValueError("trajectory 数量超过本轮 sampler 预算")
        if type(self.noise_seed) is not int or self.noise_seed < 0 or type(self.postfilter) is not bool:
            raise ValueError("noise_seed/postfilter 非法")
        if self.eta != 0:
            raise ValueError("本轮 DDIM 固定 eta=0")

    @property
    def timesteps(self):
        return TIMESTEPS if self.sampler == "ddim" else tuple(range(49, -1, -1))


def trajectory_parent_noise(*, split, row_id, trajectory_id, noise_seed, reverse_step=None):
    """4200 点共享 Gaussian 场；初始 trajectory=0 精确保留历史噪声身份。"""
    if type(trajectory_id) is not int or trajectory_id < 0:
        raise ValueError("trajectory_id 须为非负整数")
    if type(noise_seed) is not int or noise_seed < 0 or split not in {"train", "val", "synthetic"}:
        raise ValueError("非法 split/noise_seed")
    if reverse_step is not None and (type(reverse_step) is not int or not 1 <= reverse_step <= 49):
        raise ValueError("DDPM reverse noise 只在 step 1…49 生成")
    if trajectory_id == 0 and reverse_step is None:
        return parent_noise(split=split, row_id=row_id, seed=noise_seed)
    identity = json.dumps(["respdiff-bcg-trajectory-noise-v2", split, int(row_id),
                           trajectory_id, noise_seed, "initial" if reverse_step is None else reverse_step])
    seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little") % (2**63)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    field = torch.randn(4200, generator=generator, dtype=torch.float32)
    return torch.stack([field[start:start + 600] for start in STARTS])[:, None]


def batch_trajectory_noise(keys, *, trajectory_id, noise_seed, device, reverse_step=None):
    """keys 为 (split,parent_row_id,chunk_index)；逐 parent 缓存本次单个噪声场。"""
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("batch noise keys 必须非空且无重复")
    fields, chunks = {}, []
    for split, row_id, chunk in keys:
        if type(chunk) is not int or not 0 <= chunk < 13:
            raise ValueError("chunk_index 须为 0…12")
        parent = (split, row_id)
        if parent not in fields:
            fields[parent] = trajectory_parent_noise(split=split, row_id=row_id,
                trajectory_id=trajectory_id, noise_seed=noise_seed, reverse_step=reverse_step)
        chunks.append(fields[parent][chunk])
    return torch.stack(chunks).to(device)


@torch.inference_mode()
def sample_trajectory(model, condition, keys, spec, trajectory_id):
    waveform("condition", condition)
    if model.training or len(keys) != len(condition):
        raise ValueError("采样要求 eval mode 及匹配的 keys")
    if not 0 <= trajectory_id < spec.n_trajectories:
        raise ValueError("trajectory_id 超出当前预算")
    current = batch_trajectory_noise(keys, trajectory_id=trajectory_id,
                                     noise_seed=spec.noise_seed, device=condition.device)
    for index, step in enumerate(spec.timesteps):
        steps = torch.full((len(condition),), step, dtype=torch.long, device=condition.device)
        predicted = model.diffusion_model(condition, current, steps)
        finite("predicted epsilon", predicted)
        if spec.sampler == "ddim":
            alpha = model.alpha_torch[step]
            following = (model.alpha_torch[spec.timesteps[index + 1]]
                         if index + 1 < spec.nfe else torch.ones_like(alpha))
            x0 = (current - (1 - alpha).sqrt() * predicted) / alpha.sqrt()
            current = following.sqrt() * x0 + (1 - following).sqrt() * predicted
        else:
            noise = (batch_trajectory_noise(keys, trajectory_id=trajectory_id, noise_seed=spec.noise_seed,
                                           device=condition.device, reverse_step=step)
                     if step else torch.zeros_like(current))
            current = model.reverse_step(current, predicted, step=step, noise=noise)
        finite(f"{spec.sampler} trajectory={trajectory_id} t={step}", current)
    return current


@torch.inference_mode()
def ensemble_prefixes(model, condition, keys, spec, *, checkpoints=None):
    """逐 trajectory 累积 FP64 sum，仅产出所需 N 的均值，不保存 trajectory 维。"""
    checkpoints = tuple(checkpoints or (spec.n_trajectories,))
    if (tuple(sorted(set(checkpoints))) != checkpoints or not checkpoints
            or checkpoints[-1] != spec.n_trajectories or checkpoints[0] < 1):
        raise ValueError("ensemble checkpoints 须严格递增并包含最大 N")
    total = torch.zeros_like(condition, dtype=torch.float64)
    for trajectory in range(spec.n_trajectories):
        current = sample_trajectory(model, condition, keys, spec, trajectory)
        total.add_(current)
        del current
        count = trajectory + 1
        if count in checkpoints:
            mean = (total / count).to(condition.dtype)
            finite("ensemble mean", mean)
            yield count, mean


def reconstruct_ensemble_parent(chunks, *, postfilter=True):
    """chunk ensemble mean → OLA → 100 Hz interpolation → parent LPF。"""
    low, raw = restore_parent(chunks, chunk_indices=range(13))
    filtered = lowpass_parent(raw) if postfilter else raw.copy()
    return low, raw, filtered
