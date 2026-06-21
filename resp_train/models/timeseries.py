from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from resp_train.models.unet1d import ConvBlock, UNet1DTiny


def _match_length(x: torch.Tensor, length: int) -> torch.Tensor:
    diff = int(length) - x.size(-1)
    if diff > 0:
        return F.pad(x, (0, diff))
    if diff < 0:
        return x[..., :length]
    return x


def _odd_kernel(value: int, *, minimum: int = 3) -> int:
    value = max(int(value), minimum)
    return value if value % 2 == 1 else value + 1


class MovingAverage1D(nn.Module):
    """按通道做移动平均，用作 DLinear/TimeMixer 风格的趋势分解。"""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        kernel_size = _odd_kernel(kernel_size)
        self.padding = kernel_size // 2
        weight = torch.full((channels, 1, kernel_size), 1.0 / kernel_size)
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padded = F.pad(x, (self.padding, self.padding), mode="reflect")
        return F.conv1d(padded, self.weight, groups=x.size(1))


def _sinc_lowpass(cutoff_hz: float, kernel_size: int, sample_rate: float) -> torch.Tensor:
    kernel_size = _odd_kernel(kernel_size)
    half = kernel_size // 2
    t = torch.arange(-half, half + 1, dtype=torch.float32)
    cutoff = float(cutoff_hz) / float(sample_rate)
    kernel = 2.0 * cutoff * torch.sinc(2.0 * cutoff * t)
    window = torch.hamming_window(kernel_size, periodic=False, dtype=torch.float32)
    kernel = kernel * window
    return kernel / kernel.sum().clamp_min(1e-8)


class FIRBandpass1D(nn.Module):
    """初始化为呼吸频带的 depthwise FIR 前端。"""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 401,
        low_hz: float = 0.05,
        high_hz: float = 0.7,
        sample_rate: float = 100.0,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        kernel_size = _odd_kernel(kernel_size)
        high = _sinc_lowpass(high_hz, kernel_size, sample_rate)
        low = _sinc_lowpass(low_hz, kernel_size, sample_rate)
        band = high - low
        weight = band.view(1, 1, kernel_size).repeat(int(channels), 1, 1)
        self.weight = nn.Parameter(weight, requires_grad=bool(trainable))
        self.padding = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padded = F.pad(x, (self.padding, self.padding), mode="reflect")
        return F.conv1d(padded, self.weight, groups=x.size(1))


class PeriodicUNet1DTiny(nn.Module):
    """TimesNet 周期建模启发的轻量 U-Net。

    这里不直接照搬 TimesNet 的 2D 周期卷积；对 18000 点窗口来说，先用多尺度
    深度卷积提取呼吸相关低频，再交给现有 U-Net，是更稳的最小改动。
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        lowpass_kernel: int = 101,
        output_smoothing_kernel: int = 1,
    ) -> None:
        super().__init__()
        self.smoother = MovingAverage1D(in_channels, lowpass_kernel)
        self.periodic_frontend = nn.Sequential(
            nn.Conv1d(in_channels * 2, base_channels, kernel_size=9, padding=4),
            nn.GroupNorm(1, base_channels),
            nn.SiLU(),
            nn.Conv1d(base_channels, base_channels, kernel_size=15, padding=7, groups=base_channels),
            nn.Conv1d(base_channels, in_channels, kernel_size=1),
        )
        self.backbone = UNet1DTiny(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels)
        self.output_smoother: nn.Module
        if int(output_smoothing_kernel) > 1:
            self.output_smoother = MovingAverage1D(out_channels, int(output_smoothing_kernel))
        else:
            self.output_smoother = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        smooth = self.smoother(x)
        residual = self.periodic_frontend(torch.cat([x, smooth], dim=1))
        return self.output_smoother(self.backbone(smooth + residual))


class PatchMixerBlock(nn.Module):
    """PatchTST/TSMixer 风格的 patch 内与 patch 间混合。"""

    def __init__(self, channels: int, patch_count: int, hidden_channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(1, channels)
        self.patch_mixer = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.GELU(),
        )
        self.norm2 = nn.GroupNorm(1, channels)
        self.channel_mixer = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.patch_mixer(self.norm1(x))
        return x + self.channel_mixer(self.norm2(x))


class PatchMixer1D(nn.Module):
    """将长窗口切成 patch 后混合，再折回原长度。"""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        patch_len: int = 256,
        patch_stride: int = 128,
        mixer_layers: int = 2,
        overlap_window: str = "uniform",
        output_smoothing_kernel: int = 1,
    ) -> None:
        super().__init__()
        self.patch_len = max(int(patch_len), 8)
        self.patch_stride = max(int(patch_stride), 1)
        self.overlap_window = str(overlap_window)
        self.register_buffer(
            "overlap_weight",
            self._build_overlap_weight(self.patch_len, self.overlap_window),
            persistent=False,
        )
        self.patch_embed = nn.Linear(in_channels * self.patch_len, base_channels)
        self.blocks = nn.ModuleList(
            [PatchMixerBlock(base_channels, patch_count=1, hidden_channels=base_channels * 2) for _ in range(mixer_layers)]
        )
        self.patch_head = nn.Linear(base_channels, out_channels * self.patch_len)
        self.out_channels = int(out_channels)
        if int(output_smoothing_kernel) > 1:
            self.output_smoother = MovingAverage1D(out_channels, int(output_smoothing_kernel))
        else:
            self.output_smoother = nn.Identity()

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor | tuple[torch.Tensor, int]:
        batch, _, length = x.shape
        padded_length = self._padded_length(length)
        x_padded = _match_length(x, padded_length)
        patches = x_padded.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        patch_count = patches.size(2)
        tokens = patches.permute(0, 2, 1, 3).reshape(batch, patch_count, -1)
        tokens = self.patch_embed(tokens).transpose(1, 2)
        for block in self.blocks:
            tokens = block(tokens)
        if return_features:
            return tokens, length
        patch_values = self.patch_head(tokens.transpose(1, 2))
        patch_values = patch_values.view(batch, patch_count, self.out_channels, self.patch_len)
        return self.output_smoother(self._overlap_add(patch_values, length, padded_length))

    def _padded_length(self, length: int) -> int:
        if length <= self.patch_len:
            return self.patch_len
        steps = math.ceil((length - self.patch_len) / self.patch_stride)
        return steps * self.patch_stride + self.patch_len

    def _overlap_add(self, patches: torch.Tensor, length: int, padded_length: int) -> torch.Tensor:
        batch, patch_count, channels, _ = patches.shape
        out = patches.new_zeros(batch, channels, padded_length)
        weight = patches.new_zeros(batch, channels, padded_length)
        patch_weight = self.overlap_weight.to(device=patches.device, dtype=patches.dtype).view(1, 1, -1)
        for idx in range(patch_count):
            start = idx * self.patch_stride
            end = start + self.patch_len
            out[..., start:end] = out[..., start:end] + patches[:, idx] * patch_weight
            weight[..., start:end] = weight[..., start:end] + patch_weight
        out = out / weight.clamp_min(1e-8)
        return out[..., :length]

    @staticmethod
    def _build_overlap_weight(patch_len: int, overlap_window: str) -> torch.Tensor:
        name = str(overlap_window).lower()
        if name in {"uniform", "none", "flat"}:
            return torch.ones(patch_len, dtype=torch.float32)
        if name == "hann":
            window = torch.hann_window(patch_len, periodic=False, dtype=torch.float32)
        elif name in {"triangular", "triangle"}:
            pos = torch.arange(patch_len, dtype=torch.float32)
            center = (patch_len - 1) / 2.0
            window = 1.0 - torch.abs((pos - center) / max(center, 1e-6))
        else:
            raise ValueError(f"未知 overlap_window={overlap_window!r}，可选: uniform, hann, triangular")
        # 首尾保留极小权重，避免窗口边缘无人覆盖时被置零。
        return window.clamp_min(1e-3)


class _PatchHannTokenEncoder1D(nn.Module):
    """Patch-Hann token encoder，共享给低自由度输出头。"""

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        patch_len: int,
        patch_stride: int,
        mixer_layers: int,
    ) -> None:
        super().__init__()
        self.patch_len = max(int(patch_len), 8)
        self.patch_stride = max(int(patch_stride), 1)
        self.patch_embed = nn.Linear(int(in_channels) * self.patch_len, int(base_channels))
        self.blocks = nn.ModuleList(
            [
                PatchMixerBlock(int(base_channels), patch_count=1, hidden_channels=int(base_channels) * 2)
                for _ in range(max(int(mixer_layers), 0))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, length = x.shape
        padded_length = self._padded_length(length)
        x_padded = _match_length(x, padded_length)
        patches = x_padded.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        patch_count = patches.size(2)
        tokens = patches.permute(0, 2, 1, 3).reshape(batch, patch_count, -1)
        tokens = self.patch_embed(tokens).transpose(1, 2)
        for block in self.blocks:
            tokens = block(tokens)
        return tokens

    def _padded_length(self, length: int) -> int:
        if length <= self.patch_len:
            return self.patch_len
        steps = math.ceil((length - self.patch_len) / self.patch_stride)
        return steps * self.patch_stride + self.patch_len


def _hard_lowpass_output(
    y: torch.Tensor,
    *,
    max_freq_hz: float,
    sample_rate: float,
) -> torch.Tensor:
    """对最终输出做硬低通投影，限制模型只表达呼吸低频结构。"""

    length = y.size(-1)
    spectrum = torch.fft.rfft(y.float(), dim=-1)
    max_bin = int(math.floor(float(max_freq_hz) * length / max(float(sample_rate), 1e-8)))
    max_bin = max(0, min(max_bin, spectrum.size(-1) - 1))
    if max_bin + 1 < spectrum.size(-1):
        spectrum = spectrum.clone()
        spectrum[..., max_bin + 1 :] = 0
    filtered = torch.fft.irfft(spectrum, n=length, dim=-1)
    return filtered.to(dtype=y.dtype)


class _WeightedWaveformFusion1D(nn.Module):
    """用可学习权重融合多个同长度 waveform 分支。"""

    def __init__(self, branch_count: int) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(max(int(branch_count), 1)))

    def forward(self, outputs: list[torch.Tensor]) -> torch.Tensor:
        if not outputs:
            raise ValueError("至少需要一个 waveform 分支输出")
        weights = torch.softmax(self.logits[: len(outputs)], dim=0)
        stacked = torch.stack(outputs, dim=0)
        return torch.sum(weights.view(-1, 1, 1, 1) * stacked, dim=0)


class PatchHannControlPointDecoder1D(nn.Module):
    """Patch-Hann encoder + 低采样控制点输出头。

    decoder 只预测少量控制点，再插值回原窗口长度，直接限制输出时间自由度。
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        patch_len: int = 256,
        patch_stride: int = 128,
        mixer_layers: int = 2,
        control_points: int = 360,
    ) -> None:
        super().__init__()
        self.control_points = max(int(control_points), 2)
        self.encoder = _PatchHannTokenEncoder1D(
            in_channels=in_channels,
            base_channels=base_channels,
            patch_len=patch_len,
            patch_stride=patch_stride,
            mixer_layers=mixer_layers,
        )
        self.control_head = nn.Conv1d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(-1)
        controls = self.control_head(self.encoder(x))
        if controls.size(-1) != self.control_points:
            controls = F.interpolate(controls, size=self.control_points, mode="linear", align_corners=False)
        return F.interpolate(controls, size=length, mode="linear", align_corners=False)


class PatchHannBasisResidualDecoder1D(nn.Module):
    """Patch-Hann encoder + 低频基函数主输出 + 小 residual 输出头。"""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        patch_len: int = 256,
        patch_stride: int = 128,
        mixer_layers: int = 2,
        basis_count: int = 96,
        residual_points: int = 180,
        residual_scale: float = 0.05,
        duration_samples: int = 18000,
    ) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        self.basis_count = max(int(basis_count), 8)
        self.residual_points = max(int(residual_points), 2)
        self.residual_scale = float(residual_scale)
        self.encoder = _PatchHannTokenEncoder1D(
            in_channels=in_channels,
            base_channels=base_channels,
            patch_len=patch_len,
            patch_stride=patch_stride,
            mixer_layers=mixer_layers,
        )
        self.coeff_head = nn.Linear(base_channels, self.out_channels * self.basis_count)
        self.residual_head = nn.Conv1d(base_channels, self.out_channels, kernel_size=1)
        basis = self._build_basis(int(duration_samples), self.basis_count)
        self.register_buffer("basis", basis, persistent=False)

    @staticmethod
    def _build_basis(length: int, basis_count: int) -> torch.Tensor:
        t = torch.linspace(0.0, 1.0, int(length), dtype=torch.float32)
        cols = [torch.ones_like(t)]
        max_harmonics = max(math.ceil((int(basis_count) - 1) / 2), 1)
        for k in range(1, max_harmonics + 1):
            cols.append(torch.sin(2.0 * math.pi * k * t))
            if len(cols) >= basis_count:
                break
            cols.append(torch.cos(2.0 * math.pi * k * t))
            if len(cols) >= basis_count:
                break
        basis = torch.stack(cols[:basis_count], dim=0)
        return basis / basis.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(-1)
        tokens = self.encoder(x)
        pooled = tokens.mean(dim=-1)
        coeff = self.coeff_head(pooled).view(x.size(0), self.out_channels, self.basis_count)
        basis = self.basis.to(device=x.device, dtype=x.dtype)
        if basis.size(-1) != length:
            basis = F.interpolate(basis.unsqueeze(0), size=length, mode="linear", align_corners=False).squeeze(0)
        main = torch.einsum("bck,kl->bcl", coeff, basis)
        residual = self.residual_head(tokens)
        if residual.size(-1) != self.residual_points:
            residual = F.interpolate(residual, size=self.residual_points, mode="linear", align_corners=False)
        residual = F.interpolate(residual, size=length, mode="linear", align_corners=False)
        return main + self.residual_scale * residual


class PatchHannBandlimitedOutput1D(nn.Module):
    """Patch-Hann baseline 后接硬低频投影，验证最终输出带限是否足够。"""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        patch_len: int = 256,
        patch_stride: int = 128,
        mixer_layers: int = 2,
        max_freq_hz: float = 0.7,
        sample_rate: float = 100.0,
    ) -> None:
        super().__init__()
        self.max_freq_hz = float(max_freq_hz)
        self.sample_rate = float(sample_rate)
        self.backbone = PatchMixer1D(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            patch_len=patch_len,
            patch_stride=patch_stride,
            mixer_layers=mixer_layers,
            overlap_window="hann",
            output_smoothing_kernel=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.backbone(x)
        return _hard_lowpass_output(y, max_freq_hz=self.max_freq_hz, sample_rate=self.sample_rate)


class MultiScalePatchHannBandlimited1D(nn.Module):
    """SEMixer 启发的多尺度 Patch-Hann 带限输出模型。

    每个分支使用不同 patch 长度覆盖不同呼吸时间尺度，融合后仍做硬低通投影。
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        patch_lengths: list[int] | tuple[int, ...] = (256, 512, 1024, 2048),
        patch_stride_ratio: float = 0.5,
        mixer_layers: int = 2,
        max_freq_hz: float = 0.7,
        sample_rate: float = 100.0,
    ) -> None:
        super().__init__()
        lengths = [max(int(v), 8) for v in patch_lengths]
        if not lengths:
            raise ValueError("patch_lengths 不能为空")
        self.max_freq_hz = float(max_freq_hz)
        self.sample_rate = float(sample_rate)
        self.branches = nn.ModuleList(
            [
                PatchMixer1D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    base_channels=base_channels,
                    patch_len=patch_len,
                    patch_stride=max(1, int(round(patch_len * float(patch_stride_ratio)))),
                    mixer_layers=mixer_layers,
                    overlap_window="hann",
                    output_smoothing_kernel=1,
                )
                for patch_len in lengths
            ]
        )
        self.fusion = _WeightedWaveformFusion1D(len(self.branches))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fusion([branch(x) for branch in self.branches])
        return _hard_lowpass_output(y, max_freq_hz=self.max_freq_hz, sample_rate=self.sample_rate)


class PeriodAwarePatchHannBandlimited1D(MultiScalePatchHannBandlimited1D):
    """GFMixer 启发的周期候选 Patch-Hann 带限输出模型。"""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        period_secs: list[float] | tuple[float, ...] = (2.5, 4.0, 6.0, 10.0),
        patch_stride_ratio: float = 0.5,
        mixer_layers: int = 2,
        max_freq_hz: float = 0.7,
        sample_rate: float = 100.0,
    ) -> None:
        patch_lengths = [max(int(round(float(period_sec) * float(sample_rate))), 8) for period_sec in period_secs]
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            patch_lengths=patch_lengths,
            patch_stride_ratio=patch_stride_ratio,
            mixer_layers=mixer_layers,
            max_freq_hz=max_freq_hz,
            sample_rate=sample_rate,
        )


class PolyphasePatchHannBandlimited1D(nn.Module):
    """Time-TK 启发的 polyphase 子序列 Patch-Hann 带限输出模型。

    不改变原始输入来源，只从多个采样相位/步长视角读取同一窗口，降低局部采样相位敏感性。
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        offsets: list[int] | tuple[int, ...] = (1, 2, 4),
        patch_len: int = 256,
        patch_stride: int = 128,
        mixer_layers: int = 2,
        max_freq_hz: float = 0.7,
        sample_rate: float = 100.0,
    ) -> None:
        super().__init__()
        self.offsets = [max(int(v), 1) for v in offsets]
        if not self.offsets:
            raise ValueError("offsets 不能为空")
        self.max_freq_hz = float(max_freq_hz)
        self.sample_rate = float(sample_rate)
        self.branches = nn.ModuleList(
            [
                PatchMixer1D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    base_channels=base_channels,
                    patch_len=max(8, int(round(int(patch_len) / offset))),
                    patch_stride=max(1, int(round(int(patch_stride) / offset))),
                    mixer_layers=mixer_layers,
                    overlap_window="hann",
                    output_smoothing_kernel=1,
                )
                for offset in self.offsets
            ]
        )
        self.fusion = _WeightedWaveformFusion1D(len(self.branches))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(-1)
        outputs = []
        for offset, branch in zip(self.offsets, self.branches):
            if offset == 1:
                y = branch(x)
            else:
                y = branch(x[..., ::offset])
                y = F.interpolate(y, size=length, mode="linear", align_corners=False)
            outputs.append(y)
        y = self.fusion(outputs)
        return _hard_lowpass_output(y, max_freq_hz=self.max_freq_hz, sample_rate=self.sample_rate)


class FIRFrontendPatchMixer1D(nn.Module):
    """先用呼吸频带 FIR 前端滤波，再交给 PatchMixer。"""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        patch_len: int = 256,
        patch_stride: int = 128,
        mixer_layers: int = 2,
        overlap_window: str = "hann",
        output_smoothing_kernel: int = 1,
        fir_kernel_size: int = 401,
        fir_low_hz: float = 0.05,
        fir_high_hz: float = 0.7,
        fir_sample_rate: float = 100.0,
        fir_trainable: bool = True,
    ) -> None:
        super().__init__()
        self.fir = FIRBandpass1D(
            channels=in_channels,
            kernel_size=fir_kernel_size,
            low_hz=fir_low_hz,
            high_hz=fir_high_hz,
            sample_rate=fir_sample_rate,
            trainable=fir_trainable,
        )
        self.backbone = PatchMixer1D(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            patch_len=patch_len,
            patch_stride=patch_stride,
            mixer_layers=mixer_layers,
            overlap_window=overlap_window,
            output_smoothing_kernel=output_smoothing_kernel,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.fir(x))


class DLinearWaveform(nn.Module):
    """DLinear 风格的波形到波形基线。

    对每个时刻使用局部 1D 卷积分别映射 trend/seasonal，再相加输出；它不是
    预测未来窗口，而是作为同长度 waveform 重建的强基线。
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        moving_avg: int = 101,
    ) -> None:
        super().__init__()
        self.decomp = MovingAverage1D(in_channels, moving_avg)
        self.trend = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(base_channels, out_channels, kernel_size=1),
        )
        self.seasonal = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=9, padding=4),
            nn.GELU(),
            nn.Conv1d(base_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = self.decomp(x)
        seasonal = x - trend
        return self.trend(trend) + self.seasonal(seasonal)
