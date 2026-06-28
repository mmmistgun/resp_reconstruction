from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _band_indices(stft_win: int, sample_rate: float, low_hz: float, high_hz: float) -> tuple[int, int]:
    """计算 rFFT 频点中指定频带的左闭右开切片范围。"""

    freqs = torch.fft.rfftfreq(int(stft_win), d=1.0 / float(sample_rate))
    bin_width = float(sample_rate) / float(stft_win)
    tol = bin_width * 1e-4
    start = int(torch.searchsorted(freqs, torch.tensor(float(low_hz) - tol), right=False).item())
    end = int(torch.searchsorted(freqs, torch.tensor(float(high_hz) + tol), right=True).item())
    if start >= end:
        raise ValueError("STFT 频带内没有可用 rFFT 频点，请增大 stft_win 或调整 low_hz/high_hz")
    return start, end


def _stft_frame_count(length: int, win_length: int, hop_length: int, n_fft: int, center: bool) -> int:
    """返回 torch.stft 对应的时间帧数，用于约束 auxiliary head 输出形状。"""

    length = int(length)
    win_length = int(win_length)
    hop_length = int(hop_length)
    n_fft = int(n_fft)
    if center:
        effective_length = length + 2 * (n_fft // 2)
    else:
        effective_length = length
    if effective_length < win_length:
        raise ValueError("STFT 窗长不能大于有效输入长度")
    return (effective_length - win_length) // hop_length + 1


def _validate_stft_config(stft_win: int, stft_hop: int, sample_rate: float, low_hz: float, high_hz: float) -> None:
    """提前拒绝无效频带，避免后续索引计算静默裁剪配置错误。"""

    if int(stft_win) <= 0:
        raise ValueError("stft_win 必须大于 0")
    if int(stft_hop) <= 0:
        raise ValueError("stft_hop 必须大于 0")
    if not math.isfinite(float(sample_rate)):
        raise ValueError("sample_rate 必须是有限数值")
    if not math.isfinite(float(low_hz)):
        raise ValueError("low_hz 必须是有限数值")
    if not math.isfinite(float(high_hz)):
        raise ValueError("high_hz 必须是有限数值")
    if float(sample_rate) <= 0.0:
        raise ValueError("sample_rate 必须大于 0")
    if not (0.0 <= float(low_hz) < float(high_hz)):
        raise ValueError("low_hz 和 high_hz 必须满足 0 <= low_hz < high_hz")
    nyquist = float(sample_rate) / 2.0
    if float(high_hz) > nyquist:
        raise ValueError("high_hz 必须小于等于 sample_rate / 2")


def align_to_time(feats: torch.Tensor, target_len: int) -> torch.Tensor:
    """将时间序列特征重采样到目标长度，长度一致时保持原张量。"""

    target_len = int(target_len)
    if feats.size(-1) == target_len:
        return feats
    return F.interpolate(feats, size=target_len, mode="linear", align_corners=False)


class FusionHead(nn.Module):
    """将融合后的时间序列特征解码为目标长度的单通道波形。"""

    def __init__(self, in_channels: int, out_length: int, hidden: int = 16, decoder_style: str = "deep") -> None:
        super().__init__()
        self.out_length = int(out_length)
        hidden = int(hidden)
        self.decoder_style = str(decoder_style).lower()
        if self.decoder_style == "lite":
            self.decoder = nn.Sequential(
                nn.Conv1d(int(in_channels), hidden, kernel_size=1),
                nn.SiLU(),
                nn.Conv1d(hidden, 1, kernel_size=1),
            )
        elif self.decoder_style == "k1_norm":
            self.decoder = nn.Sequential(
                nn.Conv1d(int(in_channels), hidden, kernel_size=1),
                nn.GroupNorm(1, hidden),
                nn.SiLU(),
                nn.Conv1d(hidden, 1, kernel_size=1),
            )
        elif self.decoder_style == "k3_no_norm":
            self.decoder = nn.Sequential(
                nn.Conv1d(int(in_channels), hidden, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv1d(hidden, 1, kernel_size=1),
            )
        elif self.decoder_style == "deep":
            self.decoder = nn.Sequential(
                nn.Conv1d(int(in_channels), hidden, kernel_size=3, padding=1),
                nn.GroupNorm(1, hidden),
                nn.SiLU(),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv1d(hidden, 1, kernel_size=1),
            )
        else:
            raise ValueError(f"decoder_style 必须是 deep、lite、k1_norm 或 k3_no_norm，当前为: {decoder_style}")

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        decoded = self.decoder(fused)
        if decoded.size(-1) == self.out_length:
            return decoded
        return F.interpolate(decoded, size=self.out_length, mode="linear", align_corners=False)


class TargetStftAuxHead(nn.Module):
    """从共享 token 预测目标胸带 STFT log-magnitude，用于 F-B auxiliary supervision。"""

    def __init__(
        self,
        in_channels: int,
        out_length: int,
        sample_rate: float = 100.0,
        win_length: int = 3000,
        hop_length: int = 500,
        n_fft: int = 3000,
        center: bool = False,
        low_hz: float = 0.033,
        high_hz: float = 3.0,
        hidden_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_length = int(out_length)
        self.sample_rate = float(sample_rate)
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        self.n_fft = int(n_fft)
        self.center = bool(center)
        self.low_hz = float(low_hz)
        self.high_hz = float(high_hz)
        if self.n_fft < self.win_length:
            raise ValueError("fb_aux_stft_n_fft 必须大于等于 fb_aux_stft_win_length")
        _validate_stft_config(self.n_fft, self.hop_length, self.sample_rate, self.low_hz, self.high_hz)
        self.band_start, self.band_end = _band_indices(self.n_fft, self.sample_rate, self.low_hz, self.high_hz)
        self.frame_count = _stft_frame_count(
            self.out_length,
            self.win_length,
            self.hop_length,
            self.n_fft,
            self.center,
        )
        hidden = int(hidden_channels or self.in_channels)
        self.decoder = nn.Sequential(
            nn.Conv1d(self.in_channels, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(1, hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, self.band_bin_count(), kernel_size=1),
        )

    def band_bin_count(self) -> int:
        return int(self.band_end - self.band_start)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError(f"TargetStftAuxHead 期望 token 形状为 (B, C, T)，实际为 {tuple(tokens.shape)}")
        aligned = align_to_time(tokens, self.frame_count)
        return self.decoder(aligned)


class BandAwareTargetStftAuxHead(nn.Module):
    """按少量生理频带拆分输出 head 的 target STFT auxiliary head。"""

    def __init__(
        self,
        in_channels: int,
        out_length: int,
        sample_rate: float = 100.0,
        win_length: int = 3000,
        hop_length: int = 500,
        n_fft: int = 3000,
        center: bool = False,
        low_hz: float = 0.033,
        high_hz: float = 3.0,
        hidden_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_length = int(out_length)
        self.sample_rate = float(sample_rate)
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        self.n_fft = int(n_fft)
        self.center = bool(center)
        self.low_hz = float(low_hz)
        self.high_hz = float(high_hz)
        if self.n_fft < self.win_length:
            raise ValueError("fb_aux_stft_n_fft 必须大于等于 fb_aux_stft_win_length")
        _validate_stft_config(self.n_fft, self.hop_length, self.sample_rate, self.low_hz, self.high_hz)
        self.band_start, self.band_end = _band_indices(self.n_fft, self.sample_rate, self.low_hz, self.high_hz)
        self.frame_count = _stft_frame_count(
            self.out_length,
            self.win_length,
            self.hop_length,
            self.n_fft,
            self.center,
        )
        hidden = int(hidden_channels or self.in_channels)
        self.shared = nn.Sequential(
            nn.Conv1d(self.in_channels, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(1, hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.band_slices = self._build_band_slices()
        self.band_heads = nn.ModuleList(
            nn.Conv1d(hidden, rel_end - rel_start, kernel_size=1)
            for rel_start, rel_end in self.band_slices
        )

    def band_bin_count(self) -> int:
        return int(self.band_end - self.band_start)

    def band_count(self) -> int:
        return len(self.band_slices)

    def _build_band_slices(self) -> list[tuple[int, int]]:
        # 低频诊断区、常规呼吸主频段、偏快/谐波区分头；输出仍按频率顺序拼回原 bin 栅格。
        edges = [self.low_hz, 0.067, 0.3, 0.7, 1.2, self.high_hz]
        clipped = sorted({float(min(max(edge, self.low_hz), self.high_hz)) for edge in edges})
        if clipped[0] > self.low_hz:
            clipped.insert(0, self.low_hz)
        if clipped[-1] < self.high_hz:
            clipped.append(self.high_hz)

        slices: list[tuple[int, int]] = []
        cursor = 0
        total = self.band_bin_count()
        for lo, hi in zip(clipped[:-1], clipped[1:]):
            if hi <= lo:
                continue
            abs_start, abs_end = _band_indices(self.n_fft, self.sample_rate, lo, hi)
            rel_start = max(cursor, abs_start - self.band_start)
            rel_end = min(total, abs_end - self.band_start)
            if rel_end > rel_start:
                slices.append((rel_start, rel_end))
                cursor = rel_end
        if cursor < total:
            slices.append((cursor, total))
        if not slices:
            raise ValueError("enc2_band_aware_aux 未生成有效频带 head")
        return slices

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError(f"BandAwareTargetStftAuxHead 期望 token 形状为 (B, C, T)，实际为 {tuple(tokens.shape)}")
        aligned = align_to_time(tokens, self.frame_count)
        shared = self.shared(aligned)
        return torch.cat([head(shared) for head in self.band_heads], dim=1)


class LowComplexResidualHead(nn.Module):
    """从共享 token 预测低频复数 STFT 残差，再 iSTFT 成小幅 waveform residual。"""

    def __init__(
        self,
        in_channels: int,
        out_length: int,
        sample_rate: float = 100.0,
        win_length: int = 3000,
        hop_length: int = 500,
        n_fft: int = 3000,
        center: bool = True,
        low_hz: float = 0.067,
        high_hz: float = 1.2,
        hidden_channels: int | None = None,
        residual_scale: float = 0.03,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_length = int(out_length)
        self.sample_rate = float(sample_rate)
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        self.n_fft = int(n_fft)
        self.center = bool(center)
        self.low_hz = float(low_hz)
        self.high_hz = float(high_hz)
        self.residual_scale = float(residual_scale)
        if self.n_fft < self.win_length:
            raise ValueError("fb_residual_stft_n_fft 必须大于等于 fb_residual_stft_win_length")
        if self.residual_scale < 0:
            raise ValueError("fb_residual_scale 必须非负")
        _validate_stft_config(self.n_fft, self.hop_length, self.sample_rate, self.low_hz, self.high_hz)
        self.band_start, self.band_end = _band_indices(self.n_fft, self.sample_rate, self.low_hz, self.high_hz)
        self.frame_count = _stft_frame_count(
            self.out_length,
            self.win_length,
            self.hop_length,
            self.n_fft,
            self.center,
        )
        hidden = int(hidden_channels or self.in_channels)
        self.decoder = nn.Sequential(
            nn.Conv1d(self.in_channels, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(1, hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, 2 * self.band_bin_count(), kernel_size=1),
        )
        last = self.decoder[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.register_buffer("istft_window", torch.hann_window(self.win_length), persistent=False)

    def band_bin_count(self) -> int:
        return int(self.band_end - self.band_start)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError(f"LowComplexResidualHead 期望 token 形状为 (B, C, T)，实际为 {tuple(tokens.shape)}")
        aligned = align_to_time(tokens, self.frame_count)
        decoded = self.decoder(aligned)
        original_dtype = decoded.dtype
        work_dtype = torch.float64 if decoded.dtype == torch.float64 else torch.float32
        decoded = torch.tanh(decoded.to(dtype=work_dtype)) * float(self.residual_scale)
        real, imag = decoded.chunk(2, dim=1)
        complex_dtype = torch.complex128 if work_dtype == torch.float64 else torch.complex64
        spectrum = torch.zeros(
            real.size(0),
            self.n_fft // 2 + 1,
            self.frame_count,
            device=real.device,
            dtype=complex_dtype,
        )
        spectrum[:, self.band_start : self.band_end, :] = torch.complex(real, imag)
        window = self.istft_window.to(device=real.device, dtype=work_dtype)
        residual = torch.istft(
            spectrum,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
            length=self.out_length,
        )
        return residual.unsqueeze(1).to(dtype=original_dtype)


class LowBandComplexStftOutputHead(nn.Module):
    """从共享 token 直接预测低频复数 STFT，并 iSTFT 成主 waveform 输出。"""

    def __init__(
        self,
        in_channels: int,
        out_length: int,
        sample_rate: float = 100.0,
        win_length: int = 3000,
        hop_length: int = 500,
        n_fft: int = 3000,
        center: bool = True,
        low_hz: float = 0.0,
        high_hz: float = 3.0,
        hidden_channels: int | None = None,
        output_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_length = int(out_length)
        self.sample_rate = float(sample_rate)
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        self.n_fft = int(n_fft)
        self.center = bool(center)
        self.low_hz = float(low_hz)
        self.high_hz = float(high_hz)
        self.output_scale = float(output_scale)
        if self.n_fft < self.win_length:
            raise ValueError("output_stft_n_fft 必须大于等于 output_stft_win_length")
        if self.output_scale <= 0:
            raise ValueError("output_stft_scale 必须大于 0")
        _validate_stft_config(self.n_fft, self.hop_length, self.sample_rate, self.low_hz, self.high_hz)
        self.band_start, self.band_end = _band_indices(self.n_fft, self.sample_rate, self.low_hz, self.high_hz)
        self.frame_count = _stft_frame_count(
            self.out_length,
            self.win_length,
            self.hop_length,
            self.n_fft,
            self.center,
        )
        hidden = int(hidden_channels or self.in_channels)
        self.decoder = nn.Sequential(
            nn.Conv1d(self.in_channels, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(1, hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, 2 * self.band_bin_count(), kernel_size=1),
        )
        self.register_buffer("istft_window", torch.hann_window(self.win_length), persistent=False)

    def band_bin_count(self) -> int:
        return int(self.band_end - self.band_start)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.dim() != 3:
            raise ValueError(f"LowBandComplexStftOutputHead 期望 token 形状为 (B, C, T)，实际为 {tuple(tokens.shape)}")
        aligned = align_to_time(tokens, self.frame_count)
        decoded = self.decoder(aligned)
        original_dtype = decoded.dtype
        work_dtype = torch.float64 if decoded.dtype == torch.float64 else torch.float32
        decoded = decoded.to(dtype=work_dtype) * float(self.output_scale)
        real, imag = decoded.chunk(2, dim=1)
        realimag = torch.stack(
            [
                real.reshape(real.size(0), self.band_bin_count(), self.frame_count),
                imag.reshape(imag.size(0), self.band_bin_count(), self.frame_count),
            ],
            dim=1,
        )
        complex_dtype = torch.complex128 if work_dtype == torch.float64 else torch.complex64
        spectrum = torch.zeros(
            real.size(0),
            self.n_fft // 2 + 1,
            self.frame_count,
            device=real.device,
            dtype=complex_dtype,
        )
        spectrum[:, self.band_start : self.band_end, :] = torch.complex(
            realimag[:, 0],
            realimag[:, 1],
        )
        window = self.istft_window.to(device=real.device, dtype=work_dtype)
        waveform = torch.istft(
            spectrum,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
            length=self.out_length,
        )
        return waveform.unsqueeze(1).to(dtype=original_dtype), realimag.to(dtype=original_dtype)


class TfGridLiteResidualHead(nn.Module):
    """TF-Grid-lite 复数 STFT residual head，用小型时频卷积约束 residual 搜索空间。"""

    def __init__(
        self,
        in_channels: int,
        out_length: int,
        sample_rate: float = 100.0,
        win_length: int = 3000,
        hop_length: int = 500,
        n_fft: int = 3000,
        center: bool = True,
        low_hz: float = 0.067,
        high_hz: float = 1.2,
        hidden_channels: int | None = None,
        residual_scale: float = 0.03,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_length = int(out_length)
        self.sample_rate = float(sample_rate)
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        self.n_fft = int(n_fft)
        self.center = bool(center)
        self.low_hz = float(low_hz)
        self.high_hz = float(high_hz)
        self.residual_scale = float(residual_scale)
        if self.n_fft < self.win_length:
            raise ValueError("fb_residual_stft_n_fft 必须大于等于 fb_residual_stft_win_length")
        if self.residual_scale < 0:
            raise ValueError("fb_residual_scale 必须非负")
        _validate_stft_config(self.n_fft, self.hop_length, self.sample_rate, self.low_hz, self.high_hz)
        self.band_start, self.band_end = _band_indices(self.n_fft, self.sample_rate, self.low_hz, self.high_hz)
        self.frame_count = _stft_frame_count(
            self.out_length,
            self.win_length,
            self.hop_length,
            self.n_fft,
            self.center,
        )
        hidden = int(hidden_channels or self.in_channels)
        bins = self.band_bin_count()
        self.input_proj = nn.Conv1d(self.in_channels, hidden, kernel_size=1)
        self.freq_embedding = nn.Parameter(torch.zeros(1, hidden, bins, 1))
        self.freq_conv = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=(3, 1), padding=(1, 0)),
            nn.GroupNorm(1, hidden),
            nn.SiLU(),
        )
        self.time_conv = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=(1, 3), padding=(0, 1)),
            nn.GroupNorm(1, hidden),
            nn.SiLU(),
        )
        self.band_gate = nn.Conv2d(hidden, hidden, kernel_size=1)
        self.output_conv = nn.Conv2d(hidden, 2, kernel_size=1)
        nn.init.zeros_(self.band_gate.weight)
        nn.init.constant_(self.band_gate.bias, 2.0)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)
        self.register_buffer("istft_window", torch.hann_window(self.win_length), persistent=False)

    def band_bin_count(self) -> int:
        return int(self.band_end - self.band_start)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError(f"TfGridLiteResidualHead 期望 token 形状为 (B, C, T)，实际为 {tuple(tokens.shape)}")
        aligned = align_to_time(tokens, self.frame_count)
        hidden = self.input_proj(aligned)
        grid = hidden.unsqueeze(2).expand(-1, -1, self.band_bin_count(), -1) + self.freq_embedding
        grid = self.freq_conv(grid)
        grid = self.time_conv(grid)
        band_context = grid.mean(dim=3, keepdim=True)
        grid = grid * torch.sigmoid(self.band_gate(band_context))
        decoded = self.output_conv(grid)

        original_dtype = decoded.dtype
        work_dtype = torch.float64 if decoded.dtype == torch.float64 else torch.float32
        decoded = torch.tanh(decoded.to(dtype=work_dtype)) * float(self.residual_scale)
        real = decoded[:, 0, :, :]
        imag = decoded[:, 1, :, :]
        complex_dtype = torch.complex128 if work_dtype == torch.float64 else torch.complex64
        spectrum = torch.zeros(
            real.size(0),
            self.n_fft // 2 + 1,
            self.frame_count,
            device=real.device,
            dtype=complex_dtype,
        )
        spectrum[:, self.band_start : self.band_end, :] = torch.complex(real, imag)
        window = self.istft_window.to(device=real.device, dtype=work_dtype)
        residual = torch.istft(
            spectrum,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
            length=self.out_length,
        )
        return residual.unsqueeze(1).to(dtype=original_dtype)


class STFTEncoder(nn.Module):
    """固定 STFT 前端加轻量卷积编码器，输出按 STFT 帧对齐的时间序列特征。"""

    def __init__(
        self,
        sample_rate: float = 100.0,
        stft_win: int = 3000,
        stft_hop: int = 500,
        low_hz: float = 0.05,
        high_hz: float = 8.0,
        out_channels: int = 16,
        norm: str = "n0",
        band_scale_path: str | None = None,
        encoder_type: str = "conv1d",
        energy_bands: list[tuple[float, float]] | None = None,
        band_group_f: int = 32,
    ) -> None:
        super().__init__()
        self.sample_rate = float(sample_rate)
        self.stft_win = int(stft_win)
        self.stft_hop = int(stft_hop)
        self.low_hz = float(low_hz)
        self.high_hz = float(high_hz)
        self.out_channels = int(out_channels)
        self.norm = str(norm).lower()
        self.encoder_type = str(encoder_type).lower()
        self.band_group_f = int(band_group_f)

        if self.norm not in {"n0", "n1"}:
            raise ValueError(f"未知 norm={norm!r}，可选: n0, n1")

        _validate_stft_config(self.stft_win, self.stft_hop, self.sample_rate, self.low_hz, self.high_hz)
        self.band_start, self.band_end = _band_indices(self.stft_win, self.sample_rate, self.low_hz, self.high_hz)
        freq_bins = self.band_bin_count()

        self.register_buffer("stft_window", torch.hann_window(self.stft_win), persistent=False)
        if self.norm == "n1" and band_scale_path:
            scale = np.asarray(np.load(band_scale_path), dtype=np.float32)
            if scale.shape != (freq_bins,):
                raise ValueError(
                    f"band_scale 长度必须等于频带 bin 数 {freq_bins}，实际 shape={tuple(scale.shape)}"
                )
            if not np.isfinite(scale).all() or (scale <= 0.0).any():
                raise ValueError("band_scale 必须全部为有限正数")
            band_scale = torch.from_numpy(scale)
        else:
            band_scale = torch.ones(freq_bins, dtype=torch.float32)
        self.register_buffer("band_scale", band_scale, persistent=True)

        self.freq_mlp: nn.Module | None = None
        self.soft_band_logits: nn.Parameter | None = None

        # bandenergy/bandgroup/soft_band 默认 5 个重叠频带（路线图 time_frequency_input_fusion_plan.md）。
        # bandenergy 带内求均；bandgroup 保留带内频点；soft_band 用同一频带初始化可学习权重。
        # 三者共用同一份频带 bin 切片（相对裁剪后特征坐标）。
        self.energy_slices: list[tuple[int, int]] = []
        if self.encoder_type in {"bandenergy", "bandgroup", "soft_band"}:
            if energy_bands is None:
                default_bands = [(0.05, 0.3), (0.1, 0.7), (0.3, 1.2), (0.7, 3.0), (3.0, 8.0)]
                # 默认频带随当前 STFT 频率窗口裁剪；显式 energy_bands 仍保持严格校验。
                bands = []
                for lo, hi in default_bands:
                    clipped_lo = max(float(lo), self.low_hz)
                    clipped_hi = min(float(hi), self.high_hz)
                    if clipped_lo < clipped_hi:
                        bands.append((clipped_lo, clipped_hi))
            else:
                bands = energy_bands
            for lo, hi in bands:
                if not (self.low_hz <= float(lo) < float(hi) <= self.high_hz):
                    raise ValueError(
                        f"energy_band ({lo},{hi}) 必须落在 [{self.low_hz},{self.high_hz}] 内且 lo<hi"
                    )
                abs_start, abs_end = _band_indices(self.stft_win, self.sample_rate, float(lo), float(hi))
                # 转成相对裁剪后特征（band_start 起点）的索引，与 forward 里 features 同坐标。
                rel_start = abs_start - self.band_start
                rel_end = abs_end - self.band_start
                if rel_end <= rel_start:
                    raise ValueError(f"energy_band ({lo},{hi}) 在当前 STFT 分辨率下 bin 切片为空")
                self.energy_slices.append((rel_start, rel_end))
            if not self.energy_slices:
                raise ValueError("当前 STFT 频率窗口内没有可用 energy_band")

        if self.encoder_type == "conv1d":
            self.encoder = nn.Sequential(
                nn.Conv1d(freq_bins, self.out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(1, self.out_channels),
                nn.SiLU(),
                nn.Conv1d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
                nn.SiLU(),
            )
        elif self.encoder_type == "conv2d":
            self.encoder = nn.Sequential(
                nn.Conv2d(1, self.out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(1, self.out_channels),
                nn.SiLU(),
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
                nn.SiLU(),
            )
        elif self.encoder_type == "freq_mlp":
            # 每个 STFT 帧独立沿频率维做 MLP mixing，再交给 Conv1d 处理时间上下文。
            # 末层零初始化 + 残差连接，使初始状态接近 conv1d fullband。
            self.freq_mlp = nn.Sequential(
                nn.Linear(freq_bins, freq_bins),
                nn.SiLU(),
                nn.Linear(freq_bins, freq_bins),
            )
            last = self.freq_mlp[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
            self.encoder = nn.Sequential(
                nn.Conv1d(freq_bins, self.out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(1, self.out_channels),
                nn.SiLU(),
                nn.Conv1d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
                nn.SiLU(),
            )
        elif self.encoder_type == "bandenergy":
            # 输入通道 = 频带数；结构与 conv1d 一致，仅首层输入维度改为带数。
            self.encoder = nn.Sequential(
                nn.Conv1d(len(self.energy_slices), self.out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(1, self.out_channels),
                nn.SiLU(),
                nn.Conv1d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
                nn.SiLU(),
            )
        elif self.encoder_type == "bandgroup":
            # 5 带堆成 (B, n_bands, F, T) → 共享 conv2d → 沿 F 求均 → (B, out_channels, T)。
            # 仅引入「频带分组」单一变量：权重共享、参数≈conv2d，不丢带内频率结构。
            self.encoder = nn.Sequential(
                nn.Conv2d(len(self.energy_slices), self.out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(1, self.out_channels),
                nn.SiLU(),
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
                nn.SiLU(),
            )
        elif self.encoder_type == "soft_band":
            # 每个 band 是覆盖全频段的 softmax 权重；初始 logits 让权重集中在原硬频带内。
            # 训练可学习频带边界和形状，同时输出契约保持 (B, out_channels, T)。
            init_logits = torch.full((len(self.energy_slices), freq_bins), -8.0, dtype=torch.float32)
            for idx, (rel_start, rel_end) in enumerate(self.energy_slices):
                init_logits[idx, rel_start:rel_end] = 0.0
            self.soft_band_logits = nn.Parameter(init_logits)
            self.encoder = nn.Sequential(
                nn.Conv1d(len(self.energy_slices), self.out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(1, self.out_channels),
                nn.SiLU(),
                nn.Conv1d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
                nn.SiLU(),
            )
        else:
            raise ValueError(
                "未知 encoder_type="
                f"{encoder_type!r}，可选: conv1d, conv2d, bandenergy, bandgroup, freq_mlp, soft_band"
            )

    def band_bin_count(self) -> int:
        return int(self.band_end - self.band_start)

    def energy_band_count(self) -> int:
        return len(self.energy_slices)

    def band_group_count(self) -> int:
        return len(self.energy_slices)

    def freq_mlp_bin_count(self) -> int:
        return self.band_bin_count() if self.freq_mlp is not None else 0

    def soft_band_count(self) -> int:
        return 0 if self.soft_band_logits is None else int(self.soft_band_logits.shape[0])

    def soft_band_weights(self) -> torch.Tensor:
        if self.soft_band_logits is None:
            raise RuntimeError("soft_band_weights 仅在 encoder_type=soft_band 时可用")
        return torch.softmax(self.soft_band_logits, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.size(1) != 1:
            raise ValueError(f"STFTEncoder 期望输入形状为 (B, 1, L)，实际为 {tuple(x.shape)}")

        waveform = x[:, 0]
        window = self.stft_window.to(device=waveform.device, dtype=waveform.dtype)
        spectrum = torch.stft(
            waveform,
            n_fft=self.stft_win,
            hop_length=self.stft_hop,
            win_length=self.stft_win,
            window=window,
            center=True,
            return_complex=True,
        )
        features = torch.log1p(spectrum.abs())
        features = features[:, self.band_start : self.band_end, :]
        encoder_param = next(self.encoder.parameters(), None)
        if encoder_param is not None:
            features = features.to(dtype=encoder_param.dtype)
        if self.norm == "n1":
            scale = self.band_scale.to(device=features.device, dtype=features.dtype).view(1, -1, 1).clamp_min(1e-6)
            features = features / scale

        if self.encoder_type == "conv1d":
            return self.encoder(features)

        if self.encoder_type == "freq_mlp":
            assert self.freq_mlp is not None
            mixed = features.transpose(1, 2)
            mixed = mixed + self.freq_mlp(mixed)
            return self.encoder(mixed.transpose(1, 2))

        if self.encoder_type == "bandenergy":
            # 对每个重叠频带在 bin 维求均，得到 (B, n_bands, T) 能量序列再编码。
            energies = [features[:, s:e, :].mean(dim=1, keepdim=True) for (s, e) in self.energy_slices]
            return self.encoder(torch.cat(energies, dim=1))

        if self.encoder_type == "bandgroup":
            # 每带切片保留带内频点 → 沿频率轴插值到公共 F → 堆成 (B, n_bands, F, T)。
            groups = []
            for (s, e) in self.energy_slices:
                band = features[:, s:e, :]  # (B, bins_i, T)
                # 把频率放到最后一维做 1D 插值到 F，再换回 (B, F, T)。
                band = F.interpolate(
                    band.transpose(1, 2), size=self.band_group_f, mode="linear", align_corners=False
                ).transpose(1, 2)  # (B, F, T)
                groups.append(band.unsqueeze(1))  # (B, 1, F, T)
            stacked = torch.cat(groups, dim=1)  # (B, n_bands, F, T)
            encoded = self.encoder(stacked)  # (B, out_channels, F, T)
            return encoded.mean(dim=2)  # 沿 F 求均 → (B, out_channels, T)

        if self.encoder_type == "soft_band":
            weights = self.soft_band_weights().to(device=features.device, dtype=features.dtype)
            bands = torch.einsum("bft,nf->bnt", features, weights)
            return self.encoder(bands)

        encoded = self.encoder(features.unsqueeze(1))
        return encoded.mean(dim=2)


def _build_time_backbone(name: str, kwargs: dict) -> nn.Module:
    """按名称构建时间域骨干，避免在 STFT 模块顶层引入训练注册表。"""

    # 局部导入用于隔离依赖，避免仅使用 STFT 分支时加载全部时间域模型。
    from .lowfreq import MultiScaleDecompMixer1D
    from .timeseries import PatchMixer1D

    builders = {
        "patch_mixer1d": PatchMixer1D,
        "multiscale_decomp_mixer1d": MultiScaleDecompMixer1D,
    }
    key = str(name).lower()
    if key not in builders:
        raise KeyError(f"未知时间域骨干 {name!r}，可选: {', '.join(sorted(builders))}")
    return builders[key](**dict(kwargs))


class SSTCachedEncoder(nn.Module):
    """读离线预计算的 SST 幅度谱缓存 (B, in_freq, T) → 轻量 conv2d 编码 → (B, out_channels, T)。

    与 STFTEncoder 的 conv2d 分支同结构，但输入是缓存好的 log1p(|SST|)，不从波形现场算
    （SST 单窗 ~357ms，必须离线）。输出契约与 conv1d/conv2d/bandgroup 一致。
    """

    def __init__(self, in_freq: int, out_channels: int = 16) -> None:
        super().__init__()
        self.in_freq = int(in_freq)
        self.out_channels = int(out_channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, self.out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, self.out_channels),
            nn.SiLU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )

    def forward(self, sst: torch.Tensor) -> torch.Tensor:
        if sst.dim() != 3:
            raise ValueError(f"SSTCachedEncoder 期望输入 (B, freq, T)，实际为 {tuple(sst.shape)}")
        encoder_param = next(self.encoder.parameters(), None)
        if encoder_param is not None:
            sst = sst.to(dtype=encoder_param.dtype)
        encoded = self.encoder(sst.unsqueeze(1))  # (B, out_ch, freq, T)
        return encoded.mean(dim=2)  # (B, out_ch, T)


class _ResidualTcnBlock(nn.Module):
    """轻量残差 TCN block，用于 F-D 缓存特征编码器。"""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        channels = int(channels)
        dilation = int(dilation)
        if channels <= 0 or dilation <= 0:
            raise ValueError("_ResidualTcnBlock 的 channels/dilation 必须大于 0")
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.GroupNorm(1, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.GroupNorm(1, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x + self.net(x))


class CachedTfTcnEncoder(nn.Module):
    """F-D Enc1：缓存 high-CWT/SST map (B, F, T) → CWT-CNN + TCN → (B, C, T)。"""

    def __init__(
        self,
        in_freq: int,
        out_channels: int = 16,
        hidden_channels: int | None = None,
        pooled_freq: int = 6,
    ) -> None:
        super().__init__()
        self.in_freq = int(in_freq)
        self.out_channels = int(out_channels)
        hidden = int(hidden_channels or max(32, self.out_channels))
        self.pooled_freq = int(pooled_freq)
        if self.in_freq <= 0 or self.out_channels <= 0 or hidden <= 0 or self.pooled_freq <= 0:
            raise ValueError("CachedTfTcnEncoder 的 in_freq/out_channels/hidden_channels/pooled_freq 必须大于 0")
        self.map_encoder = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=(3, 7), padding=(1, 3)),
            nn.GroupNorm(1, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=(3, 5), padding=(1, 2)),
            nn.GroupNorm(1, hidden),
            nn.SiLU(),
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden * self.pooled_freq, hidden, kernel_size=1),
            nn.GroupNorm(1, hidden),
            nn.SiLU(),
            _ResidualTcnBlock(hidden, dilation=1),
            _ResidualTcnBlock(hidden, dilation=2),
            _ResidualTcnBlock(hidden, dilation=4),
            nn.Conv1d(hidden, self.out_channels, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 3:
            raise ValueError(f"CachedTfTcnEncoder 期望输入 (B, F, T)，实际为 {tuple(features.shape)}")
        if features.size(1) != self.in_freq:
            raise ValueError(f"CachedTfTcnEncoder 输入频点必须为 {self.in_freq}，实际为 {features.size(1)}")
        encoder_param = next(self.map_encoder.parameters(), None)
        if encoder_param is not None:
            features = features.to(dtype=encoder_param.dtype)
        encoded = self.map_encoder(features.unsqueeze(1))
        encoded = F.adaptive_avg_pool2d(encoded, output_size=(self.pooled_freq, encoded.size(-1)))
        encoded = encoded.flatten(1, 2)
        return self.temporal(encoded)


class CachedSequenceEncoder(nn.Module):
    """预计算序列特征 (B, K, T) 的轻量 TCN 编码器，用于 F-D modulation features。"""

    def __init__(self, in_channels: int, out_channels: int = 16, hidden_channels: int | None = None) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        hidden = int(hidden_channels or max(self.out_channels, self.in_channels))
        if self.in_channels <= 0 or self.out_channels <= 0 or hidden <= 0:
            raise ValueError("CachedSequenceEncoder 的 in_channels/out_channels/hidden_channels 必须大于 0")
        layers: list[nn.Module] = [
            nn.Conv1d(self.in_channels, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(1, hidden),
            nn.SiLU(),
        ]
        for dilation in (1, 2, 4):
            layers.extend(
                [
                    nn.Conv1d(hidden, hidden, kernel_size=3, padding=dilation, dilation=dilation),
                    nn.GroupNorm(1, hidden),
                    nn.SiLU(),
                ]
            )
        layers.append(nn.Conv1d(hidden, self.out_channels, kernel_size=1))
        self.encoder = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 3:
            raise ValueError(f"CachedSequenceEncoder 期望输入 (B, K, T)，实际为 {tuple(features.shape)}")
        if features.size(1) != self.in_channels:
            raise ValueError(
                f"CachedSequenceEncoder 输入通道必须为 {self.in_channels}，实际为 {features.size(1)}"
            )
        encoder_param = next(self.encoder.parameters(), None)
        if encoder_param is not None:
            features = features.to(dtype=encoder_param.dtype)
        return self.encoder(features)


class ResidualCachedSequenceEncoder(nn.Module):
    """F-D Enc2b：缓存序列特征 (B, K, T) 的残差 TCN 编码器。"""

    def __init__(self, in_channels: int, out_channels: int = 16, hidden_channels: int | None = None) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        hidden = int(hidden_channels or max(32, self.out_channels, self.in_channels))
        if self.in_channels <= 0 or self.out_channels <= 0 or hidden <= 0:
            raise ValueError("ResidualCachedSequenceEncoder 的 in_channels/out_channels/hidden_channels 必须大于 0")
        self.encoder = nn.Sequential(
            nn.GroupNorm(1, self.in_channels),
            nn.Conv1d(self.in_channels, hidden, kernel_size=1),
            nn.GroupNorm(1, hidden),
            nn.SiLU(),
            _ResidualTcnBlock(hidden, dilation=1),
            _ResidualTcnBlock(hidden, dilation=2),
            _ResidualTcnBlock(hidden, dilation=4),
            nn.Conv1d(hidden, self.out_channels, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 3:
            raise ValueError(f"ResidualCachedSequenceEncoder 期望输入 (B, K, T)，实际为 {tuple(features.shape)}")
        if features.size(1) != self.in_channels:
            raise ValueError(
                f"ResidualCachedSequenceEncoder 输入通道必须为 {self.in_channels}，实际为 {features.size(1)}"
            )
        encoder_param = next(self.encoder.parameters(), None)
        if encoder_param is not None:
            features = features.to(dtype=encoder_param.dtype)
        return self.encoder(features)


class TokenCrossAttentionAdapter(nn.Module):
    """time token 查询 STFT token 的轻量 cross-attention 适配器。"""

    def __init__(
        self,
        time_channels: int,
        stft_channels: int,
        num_heads: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        time_channels = int(time_channels)
        stft_channels = int(stft_channels)
        num_heads = int(num_heads)
        if time_channels <= 0 or stft_channels <= 0:
            raise ValueError("time_channels 和 stft_channels 必须大于 0")
        if num_heads <= 0:
            raise ValueError("cross_attention_heads 必须大于 0")
        if time_channels % num_heads != 0:
            raise ValueError(
                f"cross_attention_heads={num_heads} 必须整除 time_feat_channels={time_channels}"
            )
        self.kv_proj = nn.Conv1d(stft_channels, time_channels, kernel_size=1)
        self.attn = nn.MultiheadAttention(
            embed_dim=time_channels,
            num_heads=num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.out_proj = nn.Conv1d(time_channels, time_channels, kernel_size=1)
        # ReZero 式输出投影：初始不扰动 time-only 解，训练首步先打开投影。
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, time_tokens: torch.Tensor, stft_feats: torch.Tensor) -> torch.Tensor:
        if time_tokens.dim() != 3 or stft_feats.dim() != 3:
            raise ValueError("time_tokens 和 stft_feats 都必须是 (B, C, T)")
        if time_tokens.size(0) != stft_feats.size(0) or time_tokens.size(-1) != stft_feats.size(-1):
            raise ValueError(
                "time_tokens 与 stft_feats 的 batch/token 长度必须一致，"
                f"time={tuple(time_tokens.shape)} stft={tuple(stft_feats.shape)}"
            )
        query = time_tokens.transpose(1, 2)
        key_value = self.kv_proj(stft_feats).transpose(1, 2)
        attended, _ = self.attn(query, key_value, key_value, need_weights=False)
        return self.out_proj(attended.transpose(1, 2))


class TimeStftDual1D(nn.Module):
    """时间域骨干与 STFT 编码器的轻量双分支包装器。"""

    def __init__(
        self,
        time_backbone_name: str,
        time_backbone_kwargs: dict,
        time_feat_channels: int,
        branch_mode: str,
        out_length: int,
        fuse_len: int,
        stft_kwargs: dict,
        fusion_hidden: int = 16,
        fusion_decoder: str = "deep",
        fusion_mode: str = "concat_generic",
        stft_inject_position: str = "post_mixer",
        cross_attention_heads: int = 1,
        cross_attention_dropout: float = 0.0,
        fb_aux_head: str = "none",
        fb_aux_hidden_channels: int | None = None,
        fb_aux_stft_win_length: int = 3000,
        fb_aux_stft_hop_length: int = 500,
        fb_aux_stft_n_fft: int = 3000,
        fb_aux_stft_center: bool = False,
        fb_aux_stft_low_hz: float = 0.033,
        fb_aux_stft_high_hz: float = 3.0,
        fb_residual_head: str = "none",
        fb_residual_hidden_channels: int | None = None,
        fb_residual_scale: float = 0.03,
        fb_residual_stft_win_length: int = 3000,
        fb_residual_stft_hop_length: int = 500,
        fb_residual_stft_n_fft: int = 3000,
        fb_residual_stft_center: bool = True,
        fb_residual_stft_low_hz: float = 0.067,
        fb_residual_stft_high_hz: float = 1.2,
        fb_residual_energy_cap: float = 0.0,
    ) -> None:
        super().__init__()
        mode = str(branch_mode).lower()
        if mode not in {"time_only", "stft_only", "dual"}:
            raise ValueError("branch_mode 必须是 time_only、stft_only 或 dual")

        self.branch_mode = mode
        self.fuse_len = int(fuse_len)
        self.fusion_mode = str(fusion_mode).lower()
        self.stft_inject_position = str(stft_inject_position).lower()
        native_modes = {"native_inject", "token_context_inject", "gated_native_inject", "cross_attention_inject"}
        if self.fusion_mode not in {"concat_generic", *native_modes}:
            raise ValueError(
                "fusion_mode 必须是 concat_generic、native_inject、token_context_inject "
                "、gated_native_inject 或 cross_attention_inject"
            )
        use_time = mode in {"time_only", "dual"}
        use_stft = mode in {"stft_only", "dual"}
        self.fb_aux_head_name = str(fb_aux_head or "none").lower()
        if self.fb_aux_head_name not in {"none", "enc1_min_aux", "enc2_band_aware_aux"}:
            raise ValueError("fb_aux_head 必须是 none、enc1_min_aux 或 enc2_band_aware_aux")
        self.fb_residual_head_name = str(fb_residual_head or "none").lower()
        if self.fb_residual_head_name not in {"none", "low_complex_residual", "enc3_tfgrid_residual"}:
            raise ValueError("fb_residual_head 必须是 none、low_complex_residual 或 enc3_tfgrid_residual")
        self.fb_residual_energy_cap = float(fb_residual_energy_cap)
        if self.fb_residual_energy_cap < 0:
            raise ValueError("fb_residual_energy_cap 必须非负")

        # cached_*：分支改读离线时频/序列缓存，不从波形现场算；训练引擎仍通过 batch["sst"] 传入。
        self.stft_encoder_type = str(dict(stft_kwargs).get("encoder_type", "conv1d")).lower()
        self.sst_cached = self.stft_encoder_type == "sst_cached"
        self.cached_context = self.stft_encoder_type in {
            "sst_cached",
            "cached_tf",
            "cached_sequence",
            "cached_tf_tcn",
            "cached_sequence_res_tcn",
        }
        if self.cached_context and self.fusion_mode not in {"native_inject", "gated_native_inject", "cross_attention_inject"}:
            raise ValueError(
                "cached context 当前仅支持 fusion_mode=native_inject、gated_native_inject 或 cross_attention_inject"
            )

        self.time_backbone = _build_time_backbone(time_backbone_name, time_backbone_kwargs) if use_time else None
        if not use_stft:
            self.stft_encoder = None
        elif self.stft_encoder_type in {"sst_cached", "cached_tf"}:
            self.stft_encoder = SSTCachedEncoder(
                in_freq=int(stft_kwargs["in_freq"]),
                out_channels=int(stft_kwargs.get("out_channels", 16)),
            )
        elif self.stft_encoder_type == "cached_tf_tcn":
            self.stft_encoder = CachedTfTcnEncoder(
                in_freq=int(stft_kwargs["in_freq"]),
                out_channels=int(stft_kwargs.get("out_channels", 16)),
                hidden_channels=(
                    int(stft_kwargs["hidden_channels"]) if stft_kwargs.get("hidden_channels") is not None else None
                ),
                pooled_freq=int(stft_kwargs.get("pooled_freq", 6)),
            )
        elif self.stft_encoder_type == "cached_sequence":
            self.stft_encoder = CachedSequenceEncoder(
                in_channels=int(stft_kwargs["in_freq"]),
                out_channels=int(stft_kwargs.get("out_channels", 16)),
            )
        elif self.stft_encoder_type == "cached_sequence_res_tcn":
            self.stft_encoder = ResidualCachedSequenceEncoder(
                in_channels=int(stft_kwargs["in_freq"]),
                out_channels=int(stft_kwargs.get("out_channels", 16)),
                hidden_channels=(
                    int(stft_kwargs["hidden_channels"]) if stft_kwargs.get("hidden_channels") is not None else None
                ),
            )
        else:
            self.stft_encoder = STFTEncoder(**dict(stft_kwargs))

        self.stft_adapter: nn.Module | None = None
        self.stft_gate: nn.Module | None = None
        self.cross_attention_adapter: nn.Module | None = None
        self.fb_aux_head: TargetStftAuxHead | None = None
        self.fb_residual_head: LowComplexResidualHead | TfGridLiteResidualHead | None = None
        if self.fb_aux_head_name != "none":
            if not use_time:
                raise ValueError("fb_aux_head 需要启用 time_backbone")
            aux_head_cls = (
                BandAwareTargetStftAuxHead
                if self.fb_aux_head_name == "enc2_band_aware_aux"
                else TargetStftAuxHead
            )
            self.fb_aux_head = aux_head_cls(
                in_channels=int(time_feat_channels),
                out_length=int(out_length),
                sample_rate=float(stft_kwargs.get("sample_rate", 100.0)),
                win_length=int(fb_aux_stft_win_length),
                hop_length=int(fb_aux_stft_hop_length),
                n_fft=int(fb_aux_stft_n_fft),
                center=bool(fb_aux_stft_center),
                low_hz=float(fb_aux_stft_low_hz),
                high_hz=float(fb_aux_stft_high_hz),
                hidden_channels=fb_aux_hidden_channels,
            )
        if self.fb_residual_head_name != "none":
            if not use_time:
                raise ValueError("fb_residual_head 需要启用 time_backbone")
            residual_head_cls = (
                TfGridLiteResidualHead
                if self.fb_residual_head_name == "enc3_tfgrid_residual"
                else LowComplexResidualHead
            )
            self.fb_residual_head = residual_head_cls(
                in_channels=int(time_feat_channels),
                out_length=int(out_length),
                sample_rate=float(stft_kwargs.get("sample_rate", 100.0)),
                win_length=int(fb_residual_stft_win_length),
                hop_length=int(fb_residual_stft_hop_length),
                n_fft=int(fb_residual_stft_n_fft),
                center=bool(fb_residual_stft_center),
                low_hz=float(fb_residual_stft_low_hz),
                high_hz=float(fb_residual_stft_high_hz),
                hidden_channels=fb_residual_hidden_channels,
                residual_scale=float(fb_residual_scale),
            )
        if self.fusion_mode in native_modes:
            # 原生解码融合：在 token 栅格把 STFT 加性注入主干特征，再走主干原生解码契约。
            if not (use_time and hasattr(self.time_backbone, "decode_from_features")):
                raise ValueError(
                    f"{self.fusion_mode} 仅支持暴露 decode_from_features 的主干（当前仅 patch_mixer1d）"
                )
            self._validate_stft_inject_position()
            self.fusion_head = None
            if use_stft:
                if self.fusion_mode in {"native_inject", "gated_native_inject"}:
                    # 末层零初始化（ReZero 式暖启）：dual 在 init 时注入项为 0，输出等价于原生解码。
                    self.stft_proj = nn.Conv1d(
                        int(self.stft_encoder.out_channels), int(time_feat_channels), kernel_size=1
                    )
                    nn.init.zeros_(self.stft_proj.weight)
                    nn.init.zeros_(self.stft_proj.bias)
                    if self.fusion_mode == "gated_native_inject":
                        # E5-A0 轻量门控：只改变 STFT token delta 的使用强度，不改变注入位置。
                        # gate bias 初始化为 sigmoid(2)≈0.88，接近 C1B ungated，同时允许训练中下调。
                        self.stft_gate = nn.Conv1d(
                            int(self.stft_encoder.out_channels), int(time_feat_channels), kernel_size=1
                        )
                        nn.init.zeros_(self.stft_gate.weight)
                        nn.init.constant_(self.stft_gate.bias, 2.0)
                elif self.fusion_mode == "cross_attention_inject":
                    self.stft_proj = None
                    self.cross_attention_adapter = TokenCrossAttentionAdapter(
                        time_channels=int(time_feat_channels),
                        stft_channels=int(self.stft_encoder.out_channels),
                        num_heads=int(cross_attention_heads),
                        dropout=float(cross_attention_dropout),
                    )
                else:
                    # A0.1 对齐探针：给 STFT 一条 token 栅格上的局部上下文通路，再加性注入。
                    # 最后一层零初始化，确保初始输出等价 time-only，训练中再学习是否使用 STFT。
                    hidden = int(time_feat_channels)
                    self.stft_proj = None
                    self.stft_adapter = nn.Sequential(
                        nn.Conv1d(int(self.stft_encoder.out_channels), hidden, kernel_size=3, padding=1),
                        nn.GroupNorm(1, hidden),
                        nn.SiLU(),
                        nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
                    )
                    last = self.stft_adapter[-1]
                    nn.init.zeros_(last.weight)
                    nn.init.zeros_(last.bias)
            else:
                self.stft_proj = None
            return

        fused_channels = 0
        if use_time:
            fused_channels += int(time_feat_channels)
        if use_stft:
            fused_channels += int(self.stft_encoder.out_channels)
        self.fusion_head = FusionHead(
            fused_channels,
            out_length=out_length,
            hidden=fusion_hidden,
            decoder_style=fusion_decoder,
        )
        self.stft_proj = None
        self.stft_gate = None
        self.cross_attention_adapter = None

    def forward(self, x: torch.Tensor, sst: torch.Tensor | None = None) -> torch.Tensor:
        if self.fusion_mode in {"native_inject", "token_context_inject", "gated_native_inject", "cross_attention_inject"}:
            tokens, length = self._native_tokens(x, sst)
            waveform = self.time_backbone.decode_from_features(tokens, length)
            return self._maybe_attach_aux(waveform, tokens)

        features: list[torch.Tensor] = []
        aux_tokens: torch.Tensor | None = None
        if self.time_backbone is not None:
            time_feats, _ = self.time_backbone(x, return_features=True)
            aux_tokens = time_feats
            features.append(align_to_time(time_feats, self.fuse_len))
        if self.stft_encoder is not None:
            features.append(align_to_time(self.stft_encoder(x), self.fuse_len))
        waveform = self.fusion_head(torch.cat(features, dim=1))
        return self._maybe_attach_aux(waveform, aux_tokens)

    def _native_tokens(self, x: torch.Tensor, sst: torch.Tensor | None = None) -> tuple[torch.Tensor, int]:
        self._validate_stft_inject_position()
        if self.stft_encoder is not None:
            if self.fusion_mode == "cross_attention_inject":
                time_tokens, length = self.time_backbone.tokenize_input(x)
                stft_feats = self._encode_stft_features(x, sst, target_len=time_tokens.size(-1))
                token_delta = self.cross_attention_adapter(time_tokens, stft_feats)
                tokens = self.time_backbone._apply_token_injection(
                    time_tokens,
                    token_delta,
                    self.stft_inject_position,
                )
                return tokens, length
            target_tokens = self.time_backbone.token_count_for_length(x.size(-1))
            stft_feats = self._encode_stft_features(x, sst, target_len=target_tokens)
            token_delta = self._project_stft_features(stft_feats)
            return self.time_backbone.encode_tokens(
                x,
                token_injection=token_delta,
                inject_position=self.stft_inject_position,
            )
        return self.time_backbone(x, return_features=True)

    def _validate_stft_inject_position(self) -> None:
        if self.stft_inject_position not in {"pre_mixer", "mid_mixer", "post_mixer"}:
            raise ValueError(
                "stft_inject_position 必须是 pre_mixer、mid_mixer 或 post_mixer，"
                f"当前为: {self.stft_inject_position}"
            )

    def _encode_stft_features(
        self,
        x: torch.Tensor,
        sst: torch.Tensor | None,
        target_len: int,
    ) -> torch.Tensor:
        if self.cached_context:
            if sst is None:
                raise ValueError("cached context 模式需要传入预计算缓存张量 (B, C, T)")
            branch_feats = self.stft_encoder(sst)
        else:
            branch_feats = self.stft_encoder(x)
        return align_to_time(branch_feats, int(target_len))

    def _project_stft_features(self, stft_feats: torch.Tensor) -> torch.Tensor:
        if self.fusion_mode == "native_inject":
            # stft_proj 零初始化（标准 ReZero）：dual 在 init 时注入项为 0、输出等价原生解码。
            # stft_proj.weight 首步即获非零梯度并离开零点，分支从第二步起正常学习。
            return self.stft_proj(stft_feats)
        if self.fusion_mode == "gated_native_inject":
            # gate 与 stft_proj 同 token 栅格逐通道相乘。由于 stft_proj 零初始化，init 输出仍等价
            # time-only；第一步先让投影离开零点，随后 gate 开始学习选择性抑制或放大 STFT delta。
            return self.stft_proj(stft_feats) * torch.sigmoid(self.stft_gate(stft_feats))
        if self.fusion_mode == "cross_attention_inject":
            raise RuntimeError("cross_attention_inject 应在 forward 中用 time token 直接计算")
        return self.stft_adapter(stft_feats)

    def _cap_residual_energy(self, residual: torch.Tensor, base_waveform: torch.Tensor) -> torch.Tensor:
        if self.fb_residual_energy_cap <= 0:
            return residual
        residual_power = residual.float().square().mean(dim=-1, keepdim=True).clamp_min(1e-16)
        residual_rms = residual_power.sqrt()
        base_rms = base_waveform.detach().float().square().mean(dim=-1, keepdim=True).sqrt()
        max_rms = float(self.fb_residual_energy_cap) * base_rms
        scale = torch.minimum(torch.ones_like(residual_rms), max_rms / residual_rms)
        return residual * scale.to(device=residual.device, dtype=residual.dtype)

    def _maybe_attach_aux(self, waveform: torch.Tensor, tokens: torch.Tensor | None):
        if self.fb_aux_head is None and self.fb_residual_head is None:
            return waveform
        if tokens is None:
            raise RuntimeError("F-B auxiliary/residual head 需要 time tokens，但当前 forward 未产生 tokens")

        outputs: dict[str, torch.Tensor] = {}
        final_waveform = waveform
        if self.fb_residual_head is not None:
            residual = self.fb_residual_head(tokens).to(dtype=waveform.dtype)
            residual = self._cap_residual_energy(residual, waveform)
            final_waveform = waveform + residual
            outputs["base_waveform"] = waveform
            outputs["residual_waveform"] = residual
        if self.fb_aux_head is not None:
            outputs["aux_target_stft_logmag"] = self.fb_aux_head(tokens)
        return {"waveform": final_waveform, **outputs}


class TimeStftLowComplexOutput1D(TimeStftDual1D):
    """F-C wrapper：用 low-band complex STFT 作为主输出空间，再 iSTFT 回 waveform。"""

    def __init__(
        self,
        time_backbone_name: str,
        time_backbone_kwargs: dict,
        time_feat_channels: int,
        branch_mode: str,
        out_length: int,
        fuse_len: int,
        stft_kwargs: dict,
        fusion_hidden: int = 16,
        fusion_decoder: str = "deep",
        fusion_mode: str = "native_inject",
        stft_inject_position: str = "pre_mixer",
        cross_attention_heads: int = 1,
        cross_attention_dropout: float = 0.0,
        output_stft_hidden_channels: int | None = None,
        output_stft_win_length: int = 3000,
        output_stft_hop_length: int = 500,
        output_stft_n_fft: int = 3000,
        output_stft_center: bool = True,
        output_stft_low_hz: float = 0.0,
        output_stft_high_hz: float = 3.0,
        output_stft_scale: float = 1.0,
    ) -> None:
        mode = str(branch_mode).lower()
        native_modes = {"native_inject", "token_context_inject", "gated_native_inject", "cross_attention_inject"}
        if mode == "stft_only":
            raise ValueError("time_stft_low_complex_output1d 需要 time_backbone token，branch_mode 不能为 stft_only")
        if str(fusion_mode).lower() not in native_modes:
            raise ValueError("time_stft_low_complex_output1d 仅支持 native/token 注入类 fusion_mode")
        super().__init__(
            time_backbone_name=time_backbone_name,
            time_backbone_kwargs=time_backbone_kwargs,
            time_feat_channels=time_feat_channels,
            branch_mode=branch_mode,
            out_length=out_length,
            fuse_len=fuse_len,
            stft_kwargs=stft_kwargs,
            fusion_hidden=fusion_hidden,
            fusion_decoder=fusion_decoder,
            fusion_mode=fusion_mode,
            stft_inject_position=stft_inject_position,
            cross_attention_heads=cross_attention_heads,
            cross_attention_dropout=cross_attention_dropout,
            fb_aux_head="none",
            fb_residual_head="none",
        )
        self.output_stft_head = LowBandComplexStftOutputHead(
            in_channels=int(time_feat_channels),
            out_length=int(out_length),
            sample_rate=float(stft_kwargs.get("sample_rate", 100.0)),
            win_length=int(output_stft_win_length),
            hop_length=int(output_stft_hop_length),
            n_fft=int(output_stft_n_fft),
            center=bool(output_stft_center),
            low_hz=float(output_stft_low_hz),
            high_hz=float(output_stft_high_hz),
            hidden_channels=output_stft_hidden_channels,
            output_scale=float(output_stft_scale),
        )

    def forward(self, x: torch.Tensor, sst: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        tokens, _ = self._native_tokens(x, sst)
        waveform, realimag = self.output_stft_head(tokens)
        return {
            "waveform": waveform,
            "output_stft_realimag": realimag,
        }
