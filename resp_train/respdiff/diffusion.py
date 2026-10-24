"""来源loss与50步DDPM；训练RNG和按样本身份固定的采样RNG分离。"""

import hashlib
import json
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from .model import RespDiffDenoiser, RespDiffSpec, finite, waveform


def keyed_noise(shape, *, keys: Sequence[str], seed: int, trajectory: int, step: int, device):
    """逐样本CPU生成标准噪声，再转device；噪声身份不依赖batch划分。"""
    if len(keys) != shape[0] or len(set(keys)) != len(keys) or not all(isinstance(k, str) and k for k in keys):
        raise ValueError("采样keys必须与batch等长、非空且唯一")
    rows = []
    for key in keys:
        identity = json.dumps(["respdiff-noise-v1", seed, key, trajectory, step], ensure_ascii=False)
        value = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little") % (2**63)
        generator = torch.Generator(device="cpu").manual_seed(value)
        rows.append(torch.randn(shape[1:], generator=generator, dtype=torch.float32))
    return torch.stack(rows).to(device)


class RespDiff(nn.Module):
    def __init__(self, spec: RespDiffSpec = RespDiffSpec()):
        super().__init__()
        self.spec = spec
        self.diffusion_model = RespDiffDenoiser(spec)
        # 来源训练使用float32累积alpha，采样系数来自NumPy float64调度。
        beta = np.linspace(0.0001, 0.5, 50)
        alpha_hat = 1 - beta
        alpha = np.cumprod(alpha_hat)
        self.register_buffer("alpha_torch", torch.tensor(alpha).float()[:, None, None])
        self.register_buffer("beta", torch.from_numpy(beta), persistent=False)
        self.register_buffer("alpha_hat", torch.from_numpy(alpha_hat), persistent=False)
        self.register_buffer("alpha", torch.from_numpy(alpha), persistent=False)

    def training_loss(self, condition, target, *, generator=None, step=None, noise=None):
        waveform("condition", condition)
        waveform("target", target)
        if condition.shape != target.shape or condition.device != target.device:
            raise ValueError("训练condition/target的shape与device须相同")
        if step is None:
            step = torch.randint(50, (len(condition),), device=condition.device, generator=generator)
        if noise is None:
            noise = torch.randn(target.shape, dtype=target.dtype, device=target.device, generator=generator)
        waveform("training_noise", noise)
        if noise.shape != target.shape or noise.device != target.device:
            raise ValueError("训练噪声shape/device错误")
        if (step.dtype != torch.long or step.shape != (len(condition),)
                or step.device != condition.device or torch.any((step < 0) | (step >= 50))):
            raise ValueError("训练step非法")
        alpha = self.alpha_torch[step]
        noisy_target = alpha.sqrt() * target + (1 - alpha).sqrt() * noise
        predicted = self.diffusion_model(condition, noisy_target, step)
        noise_loss = (noise - predicted).square().sum() / condition.shape[-1]
        reconstructed = (noisy_target - (1 - alpha).sqrt() * predicted) / alpha.sqrt()
        finite("training_x0", reconstructed)
        fft_loss = (torch.fft.fft(reconstructed, norm="ortho").abs()
                    - torch.fft.fft(target, norm="ortho").abs()).square().mean()
        total = noise_loss + 0.01 * fft_loss if self.spec.profile == "source_fft" else noise_loss
        for name, value in (("loss_noise", noise_loss), ("loss_fft", fft_loss), ("loss", total)):
            finite(name, value)
        return {"loss": total, "loss_noise": noise_loss, "loss_fft": fft_loss}

    def reverse_step(self, current, predicted_noise, *, step: int, noise):
        if type(step) is not int or not 0 <= step < 50:
            raise ValueError("DDPM step要求0…49整数")
        for name, value in (("sample", current), ("predicted_noise", predicted_noise), ("reverse_noise", noise)):
            waveform(name, value)
            if value.shape != current.shape or value.device != current.device:
                raise ValueError("DDPM tensor shape/device不一致")
        coeff1 = 1 / self.alpha_hat[step].sqrt()
        coeff2 = (1 - self.alpha_hat[step]) / (1 - self.alpha[step]).sqrt()
        result = coeff1 * (current - coeff2 * predicted_noise)
        if step > 0:
            sigma = ((1 - self.alpha[step - 1]) / (1 - self.alpha[step]) * self.beta[step]).sqrt()
            result = result + sigma * noise
        finite(f"DDPM step={step}", result)
        return result

    @torch.inference_mode()
    def sample_mean(self, condition, *, keys: Sequence[str], seed: int, n_samples: int):
        """固定50步逐轨迹采样；要求调用方显式eval，避免隐式改变模式。"""
        waveform("condition", condition)
        if self.training:
            raise ValueError("sample_mean要求先model.eval()")
        if type(n_samples) is not int or n_samples < 1 or type(seed) is not int or seed < 0:
            raise ValueError("n_samples须为正整数，seed须为非负整数")
        total = torch.zeros_like(condition)
        for trajectory in range(n_samples):
            current = keyed_noise(condition.shape, keys=keys, seed=seed,
                                  trajectory=trajectory, step=50, device=condition.device)
            for step in range(49, -1, -1):
                steps = torch.full((len(condition),), step, dtype=torch.long, device=condition.device)
                predicted = self.diffusion_model(condition, current, steps)
                noise = (keyed_noise(condition.shape, keys=keys, seed=seed, trajectory=trajectory,
                                     step=step, device=condition.device) if step else torch.zeros_like(current))
                current = self.reverse_step(current, predicted, step=step, noise=noise)
            # 在线求均值省去N维存储，测试覆盖与stack.mean的舍入差异。
            total.add_(current / n_samples)
            finite("sample_mean", total)
        return total
