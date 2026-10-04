"""独立 loss 修复候选：逐样本 SNR 加权 FFT 与 epsilon-only 对照。"""

from omegaconf import OmegaConf
import torch

from resp_train.respdiff.model import RespDiffSpec, finite, waveform
from .config import ROOT
from .model import RespDiffBCG

OBJECTIVES = ("snr_weighted_fft", "epsilon_only")
REPAIR_CONFIG = ROOT / "configs/respdiff_bcg_loss_repair_v1/snr_weighted_fft.yaml"
SCHEMA = "respdiff-bcg-loss-repair-v1"


def snr_fft_per_sample(noise_error, target, alpha):
    """等价于 SNR(t)*FFTmagMSE(x0_hat,target)，避免先除极小 sqrt(alpha)。

    sqrt(SNR)*x0_hat = sqrt(SNR)*target + (epsilon-epsilon_hat)。
    在频域直接使用该恒等式；权重在每个样本归约前应用。
    """
    if alpha.shape != (len(target), 1, 1) or torch.any((alpha <= 0) | (alpha >= 1)):
        raise ValueError("alpha 必须是各样本的 (0,1) 累积 alpha")
    finite("alpha", alpha)
    snr = alpha / (1 - alpha)
    reference = snr.sqrt() * torch.fft.fft(target, norm="ortho")
    residual = torch.fft.fft(noise_error, norm="ortho")
    per_sample = ((reference + residual).abs() - reference.abs()).square().mean((1, 2))
    finite("snr FFT per sample", per_sample)
    return per_sample


class RespDiffBCGLossRepair(RespDiffBCG):
    def __init__(self, spec=RespDiffSpec(), *, objective="snr_weighted_fft"):
        if objective not in OBJECTIVES or spec.profile != "source_fft":
            raise ValueError("非法修复 objective 或网络 profile")
        super().__init__(spec)
        self.objective = objective

    def training_loss(self, condition, target, *, generator=None, step=None, noise=None):
        waveform("condition", condition)
        waveform("target", target)
        if condition.shape != target.shape or condition.device != target.device:
            raise ValueError("condition/target shape/device 不一致")
        if step is None:
            step = torch.randint(50, (len(condition),), device=condition.device, generator=generator)
        if noise is None:
            noise = torch.randn(target.shape, dtype=target.dtype, device=target.device, generator=generator)
        waveform("training_noise", noise)
        if noise.shape != target.shape or noise.device != target.device:
            raise ValueError("noise shape/device 不一致")
        if (step.dtype != torch.long or step.shape != (len(condition),)
                or step.device != condition.device or torch.any((step < 0) | (step >= 50))):
            raise ValueError("step 必须为同 device 的 [B] int64，取值 0…49")
        alpha = self.alpha_torch[step]
        noisy = alpha.sqrt() * target + (1 - alpha).sqrt() * noise
        predicted = self.diffusion_model(condition, noisy, step)
        error = noise - predicted
        noise_loss = error.square().mean()
        if self.objective == "snr_weighted_fft":
            spectral_rows = snr_fft_per_sample(error, target, alpha)
            spectral_loss = spectral_rows.mean()
            weighted = .01 * spectral_loss
            # 仅诊断记录原始 x0 频谱误差，不将它加回修复后的目标。
            raw_fft = (spectral_rows.detach() / (alpha / (1 - alpha)).flatten()).mean()
            losses = {"loss": noise_loss + weighted, "loss_noise": noise_loss,
                      "loss_fft_snr": spectral_loss, "loss_fft_weighted": weighted,
                      "diagnostic_fft_raw": raw_fft}
        else:
            losses = {"loss": noise_loss, "loss_noise": noise_loss}
        for name, value in losses.items():
            finite(name, value)
        return losses


def load_repair_config(path=REPAIR_CONFIG):
    cfg = OmegaConf.load(path)
    actual = OmegaConf.to_container(cfg, resolve=True)
    objective = actual.get("objective", {}).get("name")
    if objective not in OBJECTIVES:
        raise ValueError("修复 objective 必须为 snr_weighted_fft 或 epsilon_only")
    reference = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/respdiff_bcg_v1/gpu_b64_b64.yaml"), resolve=True)
    reference["protocol"] = {"name": "respdiff-bcg-loss-repair-v1-20261004", "parent": "respdiff-bcg-v1-20261003"}
    reference["objective"] = {"name": objective, "fft_weight": .01 if objective == "snr_weighted_fft" else 0.0}
    for section, key in (("training", "seed"), ("training", "device")):
        value = actual.get(section, {}).get(key)
        if key == "seed" and (type(value) is not int or value < 0):
            raise ValueError("training.seed 须为非负整数")
        if key == "device" and (not isinstance(value, str) or not value.startswith("cuda:")):
            raise ValueError("device 要求显式 cuda:N")
        reference[section][key] = value
    if actual != reference:
        raise ValueError("配置偏离 loss-repair-v1 合同；仅 objective、seed、device 可按定义变化")
    return cfg


def make_repair_model(cfg, *, tiny=False):
    spec = RespDiffSpec(8, 1, 4) if tiny else RespDiffSpec(**OmegaConf.to_container(cfg.model))
    return RespDiffBCGLossRepair(spec, objective=str(cfg.objective.name))
