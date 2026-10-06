"""双 1 Hz 父窗口低通及三个固定呼吸基带 objective。"""

import hashlib

import numpy as np
from omegaconf import OmegaConf
from scipy.signal import butter, sosfiltfilt
import torch
from torch.utils.data import Dataset

from resp_train.respdiff.data import array, float32
from resp_train.respdiff.model import RespDiffSpec, finite, waveform
from .config import ROOT
from .data import ChunkDataset
from .model import RespDiffBCG
from .repair import snr_fft_per_sample

OBJECTIVES = ("epsilon_only", "source_equivalent", "snr_resp_spectral")
SCHEMA = "respdiff-bcg-baseband-v1"
DEFAULT_CONFIG = ROOT / "configs/respdiff_bcg_baseband_v1/snr_resp_spectral.yaml"
SOS = butter(8, 1.0, fs=100.0, btype="lowpass", output="sos")
LPF_CONTRACT = {"family": "Butterworth", "order": 8, "cutoff_hz": 1.0,
                "fs": 100, "operator": "sosfiltfilt", "padtype": "odd", "padlen": 27,
                "stage": "parent180s_before_downsample", "channels": ["x", "target"],
                "sos_sha256": hashlib.sha256(SOS.astype("<f8").tobytes()).hexdigest()}


def lowpass_parent(values):
    """在完整 180 s 上双向滤波；SOS 避免低截止频率高阶 BA 的数值病态。"""
    values = array("LPF input", values, (18000,)).astype(np.float64)
    filtered = sosfiltfilt(SOS, values, padtype="odd", padlen=27)
    return float32("LPF output", array("LPF output", filtered, (18000,)))


class LowpassParents(Dataset):
    def __init__(self, parents):
        self.raw = parents
        if hasattr(parents, "rows"):
            self.rows = parents.rows
        self.index_csv_path = getattr(parents, "index_csv_path", "/")

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, index):
        item = dict(self.raw[index])
        for key in ("x", "target"):
            tensor = item[key]
            filtered = lowpass_parent(tensor.detach().cpu().numpy().reshape(-1))
            item[key] = torch.from_numpy(filtered.copy()).reshape(tensor.shape)
        return item


def baseband_dataset(parents, rows):
    return ChunkDataset(LowpassParents(parents), rows)


def absolute_stats(values):
    finite("diagnostic waveform", values)
    flat = values.detach().double().abs().flatten()
    quantiles = torch.quantile(flat, flat.new_tensor([.99, .999]))
    return {"rms": float(flat.square().mean().sqrt()), "p99": float(quantiles[0]),
            "p999": float(quantiles[1]), "max": float(flat.max())}


def resp_spectral_per_sample(error, target, alpha):
    """逐样本 SNR 加权 Hann-rFFT 幅度 MSE；只包含 bin 2…21（20 个）。"""
    if target.shape != error.shape or target.shape[-1] != 600:
        raise ValueError("spectral 输入必须匹配且长度为 600")
    finite("spectral alpha", alpha)
    if alpha.shape != (len(target), 1, 1) or torch.any((alpha <= 0) | (alpha >= 1)):
        raise ValueError("alpha 应为 [B,1,1] 且在 (0,1)")
    window = torch.hann_window(600, periodic=True, dtype=target.dtype, device=target.device)
    reference = (alpha / (1 - alpha)).sqrt() * torch.fft.rfft(target * window, norm="ortho")
    residual = torch.fft.rfft(error * window, norm="ortho")
    # 整数 bin 固定 inclusive 0.70 Hz 边界，避免 float32 rfftfreq 的端点舍入。
    difference = (reference + residual).abs()[..., 2:22] - reference.abs()[..., 2:22]
    result = difference.square().mean((1, 2))
    finite("resp spectral per sample", result)
    return result


class RespDiffBCGBaseband(RespDiffBCG):
    def __init__(self, spec=RespDiffSpec(), *, objective="snr_resp_spectral"):
        if objective not in OBJECTIVES or spec.profile != "source_fft":
            raise ValueError("非法 baseband objective/profile")
        super().__init__(spec)
        self.objective = objective
        self.last_diagnostics = {}

    def training_loss(self, condition, target, *, generator=None, step=None, noise=None):
        waveform("condition", condition)
        waveform("target", target)
        if condition.shape != target.shape or condition.device != target.device:
            raise ValueError("condition/target shape/device 不一致")
        if step is None:
            step = torch.randint(50, (len(target),), device=target.device, generator=generator)
        if noise is None:
            noise = torch.randn(target.shape, dtype=target.dtype, device=target.device, generator=generator)
        waveform("noise", noise)
        if noise.shape != target.shape or noise.device != target.device:
            raise ValueError("noise shape/device 不一致")
        if (step.dtype != torch.long or step.shape != (len(target),) or step.device != target.device
                or torch.any((step < 0) | (step >= 50))):
            raise ValueError("step 必须是同 device 的 [B] int64，范围 0…49")
        alpha = self.alpha_torch[step]
        noisy = alpha.sqrt() * target + (1 - alpha).sqrt() * noise
        error = noise - self.diffusion_model(condition, noisy, step)
        noise_loss = error.square().mean()
        if self.objective == "snr_resp_spectral":
            spectral = resp_spectral_per_sample(error, target, alpha).mean()
            weighted = .01 * spectral
        elif self.objective == "source_equivalent":
            # 保留作者全频、无 analysis window 的 FFT；常数 128 不随实际 batch 变化。
            spectral = (snr_fft_per_sample(error, target, alpha)
                        / (alpha / (1 - alpha)).flatten()).mean()
            weighted = (.01 / 128) * spectral
        else:
            spectral = noise_loss.new_zeros(())
            weighted = spectral
        losses = {"loss": noise_loss + weighted, "loss_noise": noise_loss,
                  "loss_spec": spectral, "loss_spec_weighted": weighted}
        for key, value in losses.items():
            finite(key, value)
        self.last_diagnostics = {"timestep": step.tolist(),
            **{f"epsilon_residual_{key}": value for key, value in absolute_stats(error).items()}}
        return losses


def config_contract(objective):
    if objective not in OBJECTIVES:
        raise ValueError("未知 objective")
    cfg = OmegaConf.load(ROOT / "configs/respdiff_bcg_v1/gpu_b64_b64.yaml")
    cfg.protocol = {"name": "respdiff-bcg-baseband-v1-20261006"}
    cfg.preprocessing = {key: value for key, value in LPF_CONTRACT.items() if key != "sos_sha256"}
    cfg.objective = {"name": objective, "spectral_weight": {
        "epsilon_only": 0., "source_equivalent": .01 / 128, "snr_resp_spectral": .01}[objective]}
    cfg.diagnostics = {"probe_updates": [0, 1600, 3200, 4800, 6400],
                       "timesteps": [0, 9, 19, 29, 39, 49], "noise_seed": 20261006,
                       "probe_chunks": 64}
    return cfg


def load_baseband_config(path=DEFAULT_CONFIG):
    cfg = OmegaConf.load(path)
    reference = config_contract(str(cfg.objective.name))
    # 当前三臂固定同一 seed；device 仅用于选择执行设备。
    device = cfg.training.device
    if not isinstance(device, str) or not device.startswith("cuda:") or not device[5:].isdigit():
        raise ValueError("device 必须为显式 cuda:N")
    reference.training.device = device
    if OmegaConf.to_container(cfg, resolve=True) != OmegaConf.to_container(reference, resolve=True):
        raise ValueError("配置偏离 baseband-v1 固定合同")
    return cfg


def make_baseband_model(cfg, *, tiny=False):
    spec = RespDiffSpec(8, 1, 4) if tiny else RespDiffSpec(**OmegaConf.to_container(cfg.model))
    return RespDiffBCGBaseband(spec, objective=str(cfg.objective.name))
