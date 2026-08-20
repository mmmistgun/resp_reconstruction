from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F
from scipy.signal import firwin
from torch import nn

from resp_train.crd.blocks import BidirectionalMamba2Block, DecoderResidual
from resp_train.crd.spectral_ops import fourier_interpolate


LATENT_CHANNELS = 96
LATENT_LENGTH = 1800
OUTPUT_LENGTH = 18000
KAISER_BETA = 8.6


def _kaiming_conv(module: nn.Conv1d) -> None:
    nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def _kaiser_lowpass_taps(*, sample_rate_hz: float, cutoff_hz: float, numtaps: int) -> torch.Tensor:
    if int(numtaps) <= 1 or int(numtaps) % 2 == 0:
        raise ValueError("固定抗混叠 FIR 必须使用大于 1 的奇数 numtaps")
    if not 0.0 < float(cutoff_hz) < float(sample_rate_hz) / 2.0:
        raise ValueError("固定抗混叠 FIR cutoff 必须位于 (0, Nyquist)")
    taps = firwin(
        int(numtaps),
        float(cutoff_hz),
        fs=float(sample_rate_hz),
        window=("kaiser", KAISER_BETA),
        pass_zero="lowpass",
        scale=True,
    )
    return torch.as_tensor(taps, dtype=torch.float64)


def _filter_work_dtype(x: torch.Tensor) -> torch.dtype:
    return torch.float64 if x.dtype == torch.float64 else torch.float32


class FixedPolyphaseDecimator1D(nn.Module):
    """可微的固定 FIR decimator，精确复刻 audit 的 ``resample_poly(..., padtype='line')``。

    RTM-v1 的两个共同降采样均为 ``up=1``、奇数对称 FIR。在此条件下，
    SciPy 的 polyphase 对齐等价于：按端点定义的全局直线延拓、零相位 FIR、
    再从原始第 0 个样本位置开始每 ``down`` 点取样。
    """

    def __init__(self, *, input_fs: float, down: int, numtaps: int, cutoff_hz: float) -> None:
        super().__init__()
        if int(down) <= 1:
            raise ValueError("FixedPolyphaseDecimator1D 要求 down>1")
        self.input_fs = float(input_fs)
        self.output_fs = self.input_fs / int(down)
        self.down = int(down)
        self.numtaps = int(numtaps)
        self.cutoff_hz = float(cutoff_hz)
        self.radius = self.numtaps // 2
        self.register_buffer(
            "taps",
            _kaiser_lowpass_taps(
                sample_rate_hz=self.input_fs,
                cutoff_hz=self.cutoff_hz,
                numtaps=self.numtaps,
            ),
            persistent=True,
        )

    def _line_pad(self, x: torch.Tensor) -> torch.Tensor:
        length = int(x.shape[-1])
        if length < 2:
            raise ValueError("line boundary extension 至少需要 2 个时间点")
        slope = (x[..., -1:] - x[..., :1]) / float(length - 1)
        left_offsets = torch.arange(-self.radius, 0, device=x.device, dtype=x.dtype)
        right_offsets = torch.arange(1, self.radius + 1, device=x.device, dtype=x.dtype)
        left = x[..., :1] + slope * left_offsets
        right = x[..., -1:] + slope * right_offsets
        return torch.cat((left, x, right), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or int(x.shape[-1]) < 2:
            raise ValueError(f"固定 polyphase decimator 期望 (B,C,T>=2)，实际 {tuple(x.shape)}")
        work_dtype = _filter_work_dtype(x)
        with torch.amp.autocast(x.device.type, enabled=False):
            work = self._line_pad(x.to(dtype=work_dtype))
            channels = int(work.shape[1])
            kernel = self.taps.to(device=work.device, dtype=work.dtype).view(1, 1, -1)
            kernel = kernel.expand(channels, 1, -1)
            output = F.conv1d(work, kernel, stride=self.down, groups=channels)
        expected = (int(x.shape[-1]) + self.down - 1) // self.down
        if int(output.shape[-1]) != expected:
            raise RuntimeError(f"固定 polyphase 输出长度异常: {output.shape[-1]} != {expected}")
        return output


class FixedBlockCenterDecimator1D(nn.Module):
    """复刻 signal audit 的 reflect-FIR + block-center decimation。"""

    def __init__(self, *, input_fs: float, down: int, numtaps: int, cutoff_hz: float) -> None:
        super().__init__()
        if int(down) <= 1:
            raise ValueError("FixedBlockCenterDecimator1D 要求 down>1")
        self.input_fs = float(input_fs)
        self.output_fs = self.input_fs / int(down)
        self.down = int(down)
        self.numtaps = int(numtaps)
        self.cutoff_hz = float(cutoff_hz)
        self.radius = self.numtaps // 2
        self.register_buffer(
            "taps",
            _kaiser_lowpass_taps(
                sample_rate_hz=self.input_fs,
                cutoff_hz=self.cutoff_hz,
                numtaps=self.numtaps,
            ),
            persistent=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"固定 block-center decimator 期望 (B,C,T)，实际 {tuple(x.shape)}")
        length = int(x.shape[-1])
        if length <= self.radius or length % self.down != 0:
            raise ValueError("block-center decimation 要求长度大于 FIR radius 且能被 down 整除")
        work_dtype = _filter_work_dtype(x)
        with torch.amp.autocast(x.device.type, enabled=False):
            work = x.to(dtype=work_dtype)
            work = F.pad(work, (self.radius, self.radius), mode="reflect")
            channels = int(work.shape[1])
            kernel = self.taps.to(device=work.device, dtype=work.dtype).view(1, 1, -1)
            filtered = F.conv1d(work, kernel.expand(channels, 1, -1), groups=channels)
            if self.down % 2:
                output = filtered[..., (self.down - 1) // 2 :: self.down]
            else:
                left = filtered[..., self.down // 2 - 1 :: self.down]
                right = filtered[..., self.down // 2 :: self.down]
                output = 0.5 * (left + right)
        expected = length // self.down
        if int(output.shape[-1]) != expected:
            raise RuntimeError(f"固定 block-center 输出长度异常: {output.shape[-1]} != {expected}")
        return output


class ChannelLayerNorm1D(nn.Module):
    """只沿 channel 归一化，避免引入跨时间的全局统计通路。"""

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.channels = int(channels)
        self.norm = nn.LayerNorm(self.channels, eps=float(eps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.channels:
            raise ValueError(f"ChannelLayerNorm1D 期望 (B,{self.channels},T)，实际 {tuple(x.shape)}")
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class TemporalStem(nn.Module):
    """锁定的共同 substrate：显式 100→20、20-Hz 学习解调、显式 20→10。"""

    def __init__(self, out_channels: int = LATENT_CHANNELS, mid_channels: int = 48) -> None:
        super().__init__()
        if int(out_channels) != LATENT_CHANNELS or int(mid_channels) != 48:
            raise ValueError("RTM-v1 stem 固定 mid=48、out=96")
        self.decimate_100_to_20 = FixedPolyphaseDecimator1D(
            input_fs=100.0,
            down=5,
            numtaps=255,
            cutoff_hz=9.0,
        )
        self.carrier_filter_20 = nn.Conv1d(1, 48, kernel_size=21, padding=10, bias=False)
        self.norm_20_a = ChannelLayerNorm1D(48)
        self.depthwise_20 = nn.Conv1d(48, 48, kernel_size=5, padding=2, groups=48, bias=False)
        self.norm_20_b = ChannelLayerNorm1D(48)
        self.project_20 = nn.Conv1d(48, 96, kernel_size=5, padding=2, bias=False)
        self.norm_20_c = ChannelLayerNorm1D(96)
        self.activation = nn.SiLU()
        self.decimate_20_to_10 = FixedPolyphaseDecimator1D(
            input_fs=20.0,
            down=2,
            numtaps=127,
            cutoff_hz=4.5,
        )
        for layer in (self.carrier_filter_20, self.depthwise_20, self.project_20):
            _kaiming_conv(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (1, OUTPUT_LENGTH):
            raise ValueError(f"RTM-v1 stem 期望 (B,1,{OUTPUT_LENGTH})，实际 {tuple(x.shape)}")
        latent = self.decimate_100_to_20(x)
        latent = self.activation(self.norm_20_a(self.carrier_filter_20(latent)))
        latent = self.activation(self.norm_20_b(self.depthwise_20(latent)))
        latent = self.activation(self.norm_20_c(self.project_20(latent)))
        latent = self.decimate_20_to_10(latent)
        if latent.shape[1:] != (LATENT_CHANNELS, LATENT_LENGTH):
            raise RuntimeError(f"RTM-v1 stem 输出契约错误: {tuple(latent.shape)}")
        return latent


class TemporalCoarseHead(nn.Module):
    """共同 10-Hz waveform head；规范化只沿 channel。"""

    def __init__(self) -> None:
        super().__init__()
        self.norm = ChannelLayerNorm1D(96)
        self.conv = nn.Conv1d(96, 64, kernel_size=5, padding=2, bias=False)
        self.depthwise = nn.Conv1d(64, 64, kernel_size=5, padding=2, groups=64, bias=False)
        self.reduce = nn.Conv1d(64, 32, kernel_size=1, bias=False)
        self.output = nn.Conv1d(32, 1, kernel_size=1, bias=True)
        self.activation = nn.SiLU()
        for layer in (self.conv, self.depthwise, self.reduce):
            _kaiming_conv(layer)
        nn.init.normal_(self.output.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output.bias)

    def features(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 3 or latent.shape[1:] != (LATENT_CHANNELS, LATENT_LENGTH):
            raise ValueError(f"RTM-v1 head 期望 (B,96,1800)，实际 {tuple(latent.shape)}")
        features = self.activation(self.conv(self.norm(latent)))
        features = self.activation(self.depthwise(features))
        return self.activation(self.reduce(features))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.output(self.features(latent))


class TemporalDecoder(nn.Module):
    """共享 C201-style 10-Hz nonlinear capacity 与 Fourier 上采样。"""

    def __init__(self) -> None:
        super().__init__()
        self.head = TemporalCoarseHead()
        self.residual = DecoderResidual()

    def forward(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.head.features(latent)
        waveform_10hz = self.head.output(features) + self.residual(features)
        waveform = fourier_interpolate(waveform_10hz.float(), target_length=OUTPUT_LENGTH)
        return {"waveform": waveform, "waveform_10hz": waveform_10hz}


class DilatedTCNBlock(nn.Module):
    """RTM-v1 noncausal dilated TCN residual block。"""

    def __init__(self, *, channels: int = 96, hidden_channels: int, dilation: int) -> None:
        super().__init__()
        if int(channels) != 96 or int(hidden_channels) <= 0 or int(dilation) <= 0:
            raise ValueError("RTM-v1 TCN 要求 C=96、H>0、dilation>0")
        self.channels = int(channels)
        self.dilation = int(dilation)
        self.norm = ChannelLayerNorm1D(channels)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=5,
            dilation=dilation,
            padding=2 * dilation,
            groups=channels,
            bias=False,
        )
        self.expand = nn.Conv1d(channels, int(hidden_channels), kernel_size=1, bias=False)
        self.project = nn.Conv1d(int(hidden_channels), channels, kernel_size=1, bias=True)
        self.dropout = nn.Dropout(0.10)
        self.activation = nn.SiLU()
        _kaiming_conv(self.depthwise)
        _kaiming_conv(self.expand)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.channels:
            raise ValueError(f"DilatedTCNBlock 输入契约错误: {tuple(x.shape)}")
        residual = self.activation(self.depthwise(self.norm(x)))
        residual = self.dropout(self.activation(self.expand(residual)))
        return x + self.project(residual)


class BidirectionalLSTMTrunk(nn.Module):
    """锁定的两层、每方向 H=96 离线双向 LSTM。"""

    def __init__(self, *, channels: int = 96, layers: int = 2) -> None:
        super().__init__()
        if int(channels) != 96 or int(layers) != 2:
            raise ValueError("RTM-v1 BiLSTM 固定 C=H=96、layers=2")
        self.channels = int(channels)
        self.layers = int(layers)
        self.norm = nn.LayerNorm(channels)
        self.lstm = nn.LSTM(
            input_size=channels,
            hidden_size=channels,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.10,
        )
        self.project = nn.Linear(2 * channels, channels, bias=True)
        self.dropout = nn.Dropout(0.10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.channels, LATENT_LENGTH):
            raise ValueError(f"BidirectionalLSTMTrunk 输入契约错误: {tuple(x.shape)}")
        sequence = x.transpose(1, 2)
        encoded, _ = self.lstm(self.norm(sequence))
        return (sequence + self.dropout(self.project(encoded))).transpose(1, 2)


class BidirectionalMambaTrunk(nn.Module):
    """锁定的六层 D=96 BiMamba2 trunk；不允许 fallback。"""

    def __init__(
        self,
        *,
        channels: int = 96,
        layers: int = 6,
        mamba_factory: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if int(channels) != 96 or int(layers) != 6:
            raise ValueError("RTM-v1 BiMamba2 固定 D=96、layers=6")
        self.channels = int(channels)
        self.layers = int(layers)
        self.blocks = nn.ModuleList(
            [BidirectionalMamba2Block(channels, mamba_factory=mamba_factory) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.channels, LATENT_LENGTH):
            raise ValueError(f"BidirectionalMambaTrunk 输入契约错误: {tuple(x.shape)}")
        sequence = x.transpose(1, 2)
        for block in self.blocks:
            sequence = block(sequence)
        return sequence.transpose(1, 2)


class MultiscaleTemporalPyramid(nn.Module):
    """锁定的 10/2/1-Hz feature pyramid；只有融合后的共同 waveform head。"""

    DILATIONS: tuple[tuple[int, ...], ...] = (
        (1, 2, 4, 8),
        (1, 2, 4, 8),
        (1, 2, 4, 8, 16, 32),
    )
    FACTORS: tuple[int, ...] = (1, 5, 10)

    def __init__(self, *, channels: int = 96, hidden_channels: int = 384) -> None:
        super().__init__()
        if int(channels) != 96 or int(hidden_channels) != 384:
            raise ValueError("RTM-v1 multiscale 固定 C=96、H=384")
        self.channels = int(channels)
        self.decimate_10_to_2 = FixedBlockCenterDecimator1D(
            input_fs=10.0,
            down=5,
            numtaps=255,
            cutoff_hz=0.85,
        )
        self.decimate_10_to_1 = FixedBlockCenterDecimator1D(
            input_fs=10.0,
            down=10,
            numtaps=255,
            cutoff_hz=0.45,
        )
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        DilatedTCNBlock(
                            channels=channels,
                            hidden_channels=hidden_channels,
                            dilation=dilation,
                        )
                        for dilation in dilations
                    ]
                )
                for dilations in self.DILATIONS
            ]
        )
        self.fusion = nn.Conv1d(3 * channels, channels, kernel_size=1, bias=True)
        _kaiming_conv(self.fusion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.channels, LATENT_LENGTH):
            raise ValueError(f"MultiscaleTemporalPyramid 输入契约错误: {tuple(x.shape)}")
        scaled_inputs = (x, self.decimate_10_to_2(x), self.decimate_10_to_1(x))
        outputs: list[torch.Tensor] = []
        for factor, branch, scaled in zip(self.FACTORS, self.branches, scaled_inputs):
            encoded = branch(scaled)
            if factor != 1:
                encoded = F.interpolate(encoded, size=LATENT_LENGTH, mode="linear", align_corners=False)
            outputs.append(encoded)
        return self.fusion(torch.cat(outputs, dim=1))


def tcn_receptive_field_tokens(dilations: Sequence[int], *, kernel_size: int = 5) -> int:
    """串联 same-padding dilated convolution 的理论感受野。"""

    if int(kernel_size) <= 0 or any(int(value) <= 0 for value in dilations):
        raise ValueError("kernel_size 与 dilations 必须为正")
    return 1 + (int(kernel_size) - 1) * sum(int(value) for value in dilations)
