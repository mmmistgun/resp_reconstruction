from __future__ import annotations

import math

import numpy as np
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

        # bandenergy/bandgroup 默认 5 个重叠频带（路线图 time_frequency_input_fusion_plan.md），
        # bandenergy 带内求均→(B,n_bands,T)；bandgroup 带内保留频点、插值到公共 F→(B,n_bands,F,T)。
        # 两者共用同一份频带 bin 切片（相对裁剪后特征坐标）。
        self.energy_slices: list[tuple[int, int]] = []
        if self.encoder_type in {"bandenergy", "bandgroup"}:
            bands = energy_bands or [(0.05, 0.3), (0.1, 0.7), (0.3, 1.2), (0.7, 3.0), (3.0, 8.0)]
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
        else:
            raise ValueError(
                f"未知 encoder_type={encoder_type!r}，可选: conv1d, conv2d, bandenergy, bandgroup"
            )

    def band_bin_count(self) -> int:
        return int(self.band_end - self.band_start)

    def energy_band_count(self) -> int:
        return len(self.energy_slices)

    def band_group_count(self) -> int:
        return len(self.energy_slices)

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
    ) -> None:
        super().__init__()
        mode = str(branch_mode).lower()
        if mode not in {"time_only", "stft_only", "dual"}:
            raise ValueError("branch_mode 必须是 time_only、stft_only 或 dual")

        self.branch_mode = mode
        self.fuse_len = int(fuse_len)
        self.fusion_mode = str(fusion_mode).lower()
        if self.fusion_mode not in {"concat_generic", "native_inject"}:
            raise ValueError("fusion_mode 必须是 concat_generic 或 native_inject")
        use_time = mode in {"time_only", "dual"}
        use_stft = mode in {"stft_only", "dual"}

        # encoder_type=sst_cached：STFT 分支改读离线 SST 缓存，不从波形现场算。
        self.sst_cached = str(dict(stft_kwargs).get("encoder_type", "conv1d")).lower() == "sst_cached"
        if self.sst_cached and self.fusion_mode != "native_inject":
            raise ValueError("sst_cached 当前仅支持 fusion_mode=native_inject")

        self.time_backbone = _build_time_backbone(time_backbone_name, time_backbone_kwargs) if use_time else None
        if not use_stft:
            self.stft_encoder = None
        elif self.sst_cached:
            self.stft_encoder = SSTCachedEncoder(
                in_freq=int(stft_kwargs["in_freq"]),
                out_channels=int(stft_kwargs.get("out_channels", 16)),
            )
        else:
            self.stft_encoder = STFTEncoder(**dict(stft_kwargs))

        if self.fusion_mode == "native_inject":
            # 原生解码融合：在 token 栅格把 STFT 加性注入主干特征，再走主干原生解码契约。
            if not (use_time and hasattr(self.time_backbone, "decode_from_features")):
                raise ValueError(
                    "native_inject 仅支持暴露 decode_from_features 的主干（当前仅 patch_mixer1d）"
                )
            self.fusion_head = None
            if use_stft:
                # 末层零初始化（ReZero 式暖启）：dual 在 init 时注入项为 0，输出等价于原生解码。
                self.stft_proj = nn.Conv1d(int(self.stft_encoder.out_channels), int(time_feat_channels), kernel_size=1)
                nn.init.zeros_(self.stft_proj.weight)
                nn.init.zeros_(self.stft_proj.bias)
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

    def forward(self, x: torch.Tensor, sst: torch.Tensor | None = None) -> torch.Tensor:
        if self.fusion_mode == "native_inject":
            time_feats, length = self.time_backbone(x, return_features=True)
            if self.stft_encoder is not None:
                if self.sst_cached:
                    if sst is None:
                        raise ValueError("sst_cached 模式需要传入预计算 sst 张量 (B, freq, T)")
                    branch_feats = self.stft_encoder(sst)
                else:
                    branch_feats = self.stft_encoder(x)
                stft_feats = align_to_time(branch_feats, time_feats.size(-1))
                # stft_proj 末层零初始化（标准 ReZero）：dual 在 init 时注入项为 0、输出等价原生解码。
                # stft_proj.weight 首步即获非零梯度并离开零点，分支从第二步起正常学习。
                time_feats = time_feats + self.stft_proj(stft_feats)
            return self.time_backbone.decode_from_features(time_feats, length)

        features: list[torch.Tensor] = []
        if self.time_backbone is not None:
            time_feats, _ = self.time_backbone(x, return_features=True)
            features.append(align_to_time(time_feats, self.fuse_len))
        if self.stft_encoder is not None:
            features.append(align_to_time(self.stft_encoder(x), self.fuse_len))
        return self.fusion_head(torch.cat(features, dim=1))
