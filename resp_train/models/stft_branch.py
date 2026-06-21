from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _band_indices(stft_win: int, sample_rate: float, low_hz: float, high_hz: float) -> tuple[int, int]:
    """计算 rFFT 频点中指定频带的左闭右开切片范围。"""

    freqs = torch.fft.rfftfreq(int(stft_win), d=1.0 / float(sample_rate))
    start = int(torch.searchsorted(freqs, torch.tensor(float(low_hz)), right=False).item())
    end = int(torch.searchsorted(freqs, torch.tensor(float(high_hz)), right=True).item())
    if start >= end:
        raise ValueError("STFT 频带内没有可用 rFFT 频点，请增大 stft_win 或调整 low_hz/high_hz")
    return start, end


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

    def __init__(self, in_channels: int, out_length: int, hidden: int = 16) -> None:
        super().__init__()
        self.out_length = int(out_length)
        hidden = int(hidden)
        self.decoder = nn.Sequential(
            nn.Conv1d(int(in_channels), hidden, kernel_size=3, padding=1),
            nn.GroupNorm(1, hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, 1, kernel_size=1),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        decoded = self.decoder(fused)
        if decoded.size(-1) == self.out_length:
            return decoded
        return F.interpolate(decoded, size=self.out_length, mode="linear", align_corners=False)


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
        encoder_type: str = "conv1d",
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

        if self.norm != "n0":
            raise ValueError(f"未知 norm={norm!r}，当前 STFTEncoder 仅支持 norm='n0'")

        _validate_stft_config(self.stft_win, self.stft_hop, self.sample_rate, self.low_hz, self.high_hz)
        self.band_start, self.band_end = _band_indices(self.stft_win, self.sample_rate, self.low_hz, self.high_hz)
        freq_bins = self.band_bin_count()

        self.register_buffer("stft_window", torch.hann_window(self.stft_win), persistent=False)
        self.register_buffer("band_scale", torch.ones(freq_bins, dtype=torch.float32), persistent=True)

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
        else:
            raise ValueError(f"未知 encoder_type={encoder_type!r}，可选: conv1d, conv2d")

    def band_bin_count(self) -> int:
        return int(self.band_end - self.band_start)

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
        encoder_dtype = next(self.encoder.parameters()).dtype
        features = features.to(dtype=encoder_dtype)

        if self.encoder_type == "conv1d":
            return self.encoder(features)

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
    ) -> None:
        super().__init__()
        mode = str(branch_mode).lower()
        if mode not in {"time_only", "stft_only", "dual"}:
            raise ValueError("branch_mode 必须是 time_only、stft_only 或 dual")

        self.branch_mode = mode
        self.fuse_len = int(fuse_len)
        use_time = mode in {"time_only", "dual"}
        use_stft = mode in {"stft_only", "dual"}

        self.time_backbone = _build_time_backbone(time_backbone_name, time_backbone_kwargs) if use_time else None
        self.stft_encoder = STFTEncoder(**dict(stft_kwargs)) if use_stft else None

        fused_channels = 0
        if use_time:
            fused_channels += int(time_feat_channels)
        if use_stft:
            fused_channels += int(self.stft_encoder.out_channels)
        self.fusion_head = FusionHead(fused_channels, out_length=out_length, hidden=fusion_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features: list[torch.Tensor] = []
        if self.time_backbone is not None:
            time_feats, _ = self.time_backbone(x, return_features=True)
            features.append(align_to_time(time_feats, self.fuse_len))
        if self.stft_encoder is not None:
            features.append(align_to_time(self.stft_encoder(x), self.fuse_len))
        return self.fusion_head(torch.cat(features, dim=1))
