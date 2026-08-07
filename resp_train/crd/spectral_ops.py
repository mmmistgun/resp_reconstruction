from __future__ import annotations

import torch
from torch import nn


DIRECT_CENTER_INTERVALS = (
    (0.05, 0.09),
    (0.09, 0.14),
    (0.14, 0.22),
    (0.22, 0.34),
    (0.34, 0.50),
    (0.50, 0.69),
)
DIRECT_CENTER_INITIAL = (0.07, 0.11, 0.18, 0.28, 0.43, 0.62)
DIRECT_BANDWIDTH_INTERVALS = (
    (0.008, 0.030),
    (0.010, 0.040),
    (0.015, 0.060),
    (0.020, 0.080),
    (0.030, 0.120),
    (0.040, 0.150),
)
DIRECT_BANDWIDTH_INITIAL = (0.015, 0.020, 0.030, 0.040, 0.060, 0.080)


def _inverse_logit(value: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(value.dtype).eps
    value = value.clamp(eps, 1.0 - eps)
    return torch.log(value) - torch.log1p(-value)


class GaussianAnalyticFilterbank(nn.Module):
    """CRD Direct 分支的六带可学习 Gaussian analytic filterbank。"""

    def __init__(self, *, sample_rate: float = 100.0, length: int = 18000) -> None:
        super().__init__()
        self.sample_rate = float(sample_rate)
        self.length = int(length)
        if self.sample_rate != 100.0 or self.length != 18000:
            raise ValueError("CRD-v1 Direct filterbank 固定为 100 Hz、18000 点")

        center_bounds = torch.tensor(DIRECT_CENTER_INTERVALS, dtype=torch.float32)
        bandwidth_bounds = torch.tensor(DIRECT_BANDWIDTH_INTERVALS, dtype=torch.float32)
        center_initial = torch.tensor(DIRECT_CENTER_INITIAL, dtype=torch.float32)
        bandwidth_initial = torch.tensor(DIRECT_BANDWIDTH_INITIAL, dtype=torch.float32)
        center_fraction = (center_initial - center_bounds[:, 0]) / (center_bounds[:, 1] - center_bounds[:, 0])
        bandwidth_fraction = (bandwidth_initial - bandwidth_bounds[:, 0]) / (
            bandwidth_bounds[:, 1] - bandwidth_bounds[:, 0]
        )
        self.center_logits = nn.Parameter(_inverse_logit(center_fraction))
        self.bandwidth_logits = nn.Parameter(_inverse_logit(bandwidth_fraction))
        self.register_buffer("center_bounds", center_bounds, persistent=True)
        self.register_buffer("bandwidth_bounds", bandwidth_bounds, persistent=True)

    def centers(self) -> torch.Tensor:
        return self.center_bounds[:, 0] + torch.sigmoid(self.center_logits) * (
            self.center_bounds[:, 1] - self.center_bounds[:, 0]
        )

    def bandwidths(self) -> torch.Tensor:
        return self.bandwidth_bounds[:, 0] + torch.sigmoid(self.bandwidth_logits) * (
            self.bandwidth_bounds[:, 1] - self.bandwidth_bounds[:, 0]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 1 or x.shape[-1] != self.length:
            raise ValueError(f"Direct filterbank 期望 (B,1,{self.length})，实际 {tuple(x.shape)}")
        if not torch.isfinite(x).all():
            raise FloatingPointError("Direct filterbank 输入包含 NaN/Inf")

        # 频谱与 Gaussian mask 始终在 float32/complex64 中计算，不继承 bf16 autocast。
        with torch.amp.autocast(x.device.type, enabled=False):
            waveform = x[:, 0].float()
            spectrum = torch.fft.fft(waveform, n=self.length, dim=-1)
            frequencies = torch.fft.fftfreq(
                self.length,
                d=1.0 / self.sample_rate,
                device=waveform.device,
            )
            centers = self.centers().float()
            bandwidths = self.bandwidths().float()
            delta = (frequencies[None, :] - centers[:, None]) / bandwidths[:, None]
            masks = torch.exp(-0.5 * delta.square())
            positive_support = (frequencies > 0.0) & (frequencies >= 0.03) & (frequencies <= 0.70)
            masks = masks * positive_support.to(dtype=masks.dtype)[None, :]
            analytic_spectrum = 2.0 * spectrum[:, None, :] * masks[None, :, :]
            analytic = torch.fft.ifft(analytic_spectrum, n=self.length, dim=-1)
            channels = torch.stack((analytic.real, analytic.imag), dim=2)
            return channels.reshape(x.shape[0], 12, self.length).float()


def fft_hard_lowpass(signal: torch.Tensor, *, sample_rate: float, cutoff_hz: float) -> torch.Tensor:
    """整窗 hard LPF；保留 DC，并固定用 float32。"""

    if signal.ndim < 2 or signal.shape[-1] <= 0:
        raise ValueError("signal 必须包含非空时间维")
    if not (0.0 <= float(cutoff_hz) <= float(sample_rate) / 2.0):
        raise ValueError("cutoff_hz 必须位于 [0, Nyquist]")
    with torch.amp.autocast(signal.device.type, enabled=False):
        work = signal.float()
        length = int(work.shape[-1])
        spectrum = torch.fft.rfft(work, n=length, dim=-1)
        frequencies = torch.fft.rfftfreq(length, d=1.0 / float(sample_rate), device=work.device)
        mask = frequencies <= float(cutoff_hz)
        return torch.fft.irfft(spectrum * mask.to(spectrum.dtype), n=length, dim=-1).float()


def fourier_interpolate(signal: torch.Tensor, *, target_length: int) -> torch.Tensor:
    """按 CRD 契约用 ``norm=forward`` 的频谱零填充做周期插值。"""

    if signal.ndim < 2:
        raise ValueError("signal 至少需要 batch 与时间维")
    source_length = int(signal.shape[-1])
    target_length = int(target_length)
    if source_length <= 0 or target_length < source_length:
        raise ValueError("Fourier interpolation 只支持非空序列上采样")
    if target_length == source_length:
        return signal.float()

    with torch.amp.autocast(signal.device.type, enabled=False):
        work = signal.float()
        source = torch.fft.rfft(work, n=source_length, dim=-1, norm="forward")
        # 偶数长度源序列的 Nyquist bin 没有可直接复制的正/负频率对，协议规定丢弃。
        copied_bins = source.shape[-1] - (1 if source_length % 2 == 0 else 0)
        target_bins = target_length // 2 + 1
        if copied_bins > target_bins:
            raise ValueError("目标频谱无法容纳源频谱")
        target = torch.zeros(
            *source.shape[:-1],
            target_bins,
            device=source.device,
            dtype=source.dtype,
        )
        target[..., :copied_bins] = source[..., :copied_bins]
        return torch.fft.irfft(target, n=target_length, dim=-1, norm="forward").float()
