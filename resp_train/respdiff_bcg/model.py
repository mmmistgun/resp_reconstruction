"""平均噪声损失与保幅六步 DDIM。"""

import torch

from resp_train.respdiff.diffusion import RespDiff
from resp_train.respdiff.model import finite, waveform

TIMESTEPS = (49, 39, 29, 19, 9, 0)


class RespDiffBCG(RespDiff):
    def training_loss(self, condition, target, **kwargs):
        if self.spec.profile != "source_fft":
            raise ValueError("BCG 合同要求 source_fft 网络")
        # 来源噪声项为 sum/L；这里明确改为对 B、K、L 全部取均值。
        losses = super().training_loss(condition, target, **kwargs)
        noise = losses["loss_noise"] / condition.shape[0]
        loss = noise + 0.01 * losses["loss_fft"]
        finite("BCG loss", loss)
        return {"loss": loss, "loss_noise": noise, "loss_fft": losses["loss_fft"]}

    @torch.inference_mode()
    def sample_chunks(self, condition, initial_noise):
        if self.training:
            raise ValueError("采样前必须 model.eval()")
        waveform("condition", condition)
        waveform("initial_noise", initial_noise)
        if condition.shape != initial_noise.shape or condition.device != initial_noise.device:
            raise ValueError("初始噪声 shape/device 不匹配")
        current = initial_noise.clone()
        for index, step in enumerate(TIMESTEPS):
            steps = torch.full((len(condition),), step, dtype=torch.long, device=condition.device)
            predicted = self.diffusion_model(condition, current, steps)
            alpha = self.alpha_torch[step]
            # 最后一步到干净 x0，其 alpha_bar=1，不使用 Python 的 -1 下标。
            following = (self.alpha_torch[TIMESTEPS[index + 1]]
                         if index + 1 < len(TIMESTEPS) else torch.ones_like(alpha))
            x0 = (current - (1 - alpha).sqrt() * predicted) / alpha.sqrt()
            current = following.sqrt() * x0 + (1 - following).sqrt() * predicted
            finite(f"DDIM t={step}", current)
        return current
