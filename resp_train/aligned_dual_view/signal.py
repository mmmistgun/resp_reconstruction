from __future__ import annotations

import numpy as np
import torch
from scipy.signal import firwin
from torch import nn
from torch.nn import functional as F

SAMPLE_RATE = 100.0
WINDOW_SAMPLES = 18000
DECIMATION = 10
LATENT_LENGTH = WINDOW_SAMPLES // DECIMATION
FIR_TAPS = 501
FIR_CUTOFF_HZ = 4.0
FIR_BETA = 8.6


def fir_coefficients() -> np.ndarray:
    """用成熟的 FIR 设计实现生成系数；float64 设计，float32 执行。"""
    return firwin(
        FIR_TAPS, FIR_CUTOFF_HZ, window=("kaiser", FIR_BETA),
        pass_zero="lowpass", scale=True, fs=SAMPLE_RATE,
    )


def require_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"ADV-v1 {name} 包含 NaN/Inf")


class CenteredFIRDecimate10(nn.Module):
    """两路共用的中心滤波抽取：输出 n 对应原始位置 10*n。"""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "kernel", torch.from_numpy(fir_coefficients().astype(np.float32)).view(1, 1, -1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 3 or min(value.shape) <= 0 or value.shape[-1] <= FIR_TAPS // 2:
            raise ValueError("D10 期望非空 (B,C,T)，且 T > 250")
        if not value.is_floating_point():
            raise TypeError("D10 输入必须是浮点 tensor")
        if self.kernel.dtype != torch.float32:
            raise TypeError("D10 固定系数必须保持 float32；混合精度请使用 autocast")
        if value.device.type == "cuda" and torch.backends.cudnn.conv.fp32_precision != "ieee":
            raise RuntimeError("D10 要求 torch.backends.cudnn.conv.fp32_precision='ieee'，避免 TF32 量化滤波核")
        require_finite("D10 输入", value)
        # AMP 下仍用 float32，避免 bf16 系数量化破坏阻带。
        # 对称核的互相关等于卷积；reflect250 + valid stride10 已补偿群延迟。
        # 只计算保留点，等价于 stride1 后 [:, :, ::10]。
        with torch.amp.autocast(value.device.type, enabled=False):
            batch, channels, length = value.shape
            work = value.float().reshape(batch * channels, 1, length)
            padded = F.pad(work, (FIR_TAPS // 2, FIR_TAPS // 2), mode="reflect")
            result = F.conv1d(padded, self.kernel.float(), stride=DECIMATION)
            result = result.reshape(batch, channels, -1)
        require_finite("D10 输出", result)
        return result
