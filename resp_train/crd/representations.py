from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from resp_train.crd.blocks import ResidualDWBlock
from resp_train.crd.initialization import module_seed
from resp_train.crd.spectral_ops import AmplitudeModulationFilterbank, fft_hard_bandpass


def _validate_representation_input(x: torch.Tensor) -> None:
    if x.ndim != 3 or x.shape[1:] != (1, 18000):
        raise ValueError(f"S2 representation 期望输入 (B,1,18000)，实际 {tuple(x.shape)}")
    if not torch.isfinite(x).all():
        raise FloatingPointError("S2 representation 输入包含 NaN/Inf")


def _initialize_conv(conv: nn.Conv1d) -> None:
    nn.init.kaiming_normal_(conv.weight, mode="fan_in", nonlinearity="relu")
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


def _initialize_norm(norm: nn.GroupNorm | nn.LayerNorm) -> None:
    nn.init.ones_(norm.weight)
    nn.init.zeros_(norm.bias)


def _zero_projection(projection: nn.Conv1d) -> None:
    nn.init.zeros_(projection.weight)
    if projection.bias is not None:
        nn.init.zeros_(projection.bias)


class LegacyEnergyRepresentation(nn.Module):
    """S2A Legacy E：固定五带 STFT energy 经轻量 encoder 静态注入。"""

    bands = (
        (0.05, 0.30),
        (0.10, 0.70),
        (0.30, 1.20),
        (0.70, 3.00),
        (3.00, 8.00),
    )

    def __init__(self, initialization_seed: int) -> None:
        super().__init__()
        self.sample_rate = 100.0
        self.n_fft = 2000
        self.hop_length = 250
        self.register_buffer(
            "window",
            torch.hann_window(self.n_fft, periodic=True, dtype=torch.float32),
            persistent=False,
        )
        with module_seed(initialization_seed, "legacy_energy_encoder"):
            self.conv_in = nn.Conv1d(5, 16, kernel_size=3, padding=1, bias=True)
            self.norm = nn.GroupNorm(1, 16, eps=1e-5, affine=True)
            self.conv_out = nn.Conv1d(16, 16, kernel_size=3, padding=1, bias=True)
            _initialize_conv(self.conv_in)
            _initialize_norm(self.norm)
            _initialize_conv(self.conv_out)
        with module_seed(initialization_seed, "legacy_energy_projection"):
            self.projection = nn.Conv1d(16, 96, kernel_size=1, bias=True)
            _zero_projection(self.projection)
        self.activation = nn.SiLU()

    def band_energy(self, x: torch.Tensor) -> torch.Tensor:
        _validate_representation_input(x)
        with torch.amp.autocast(x.device.type, enabled=False):
            waveform = x[:, 0].float()
            spectrum = torch.stft(
                waveform,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=self.window,
                center=True,
                pad_mode="reflect",
                normalized=False,
                onesided=True,
                return_complex=True,
            )
            magnitude = torch.log1p(spectrum.abs())
            frequencies = torch.fft.rfftfreq(
                self.n_fft,
                d=1.0 / self.sample_rate,
                device=waveform.device,
            )
            features = []
            for low, high in self.bands:
                bins = (frequencies >= low) & (frequencies <= high)
                if not bool(bins.any()):
                    raise RuntimeError(f"Legacy E 频带 [{low},{high}] 没有 STFT bin")
                features.append(magnitude[:, bins, :].mean(dim=1))
            energy = torch.stack(features, dim=1).float()
            if energy.shape[1:] != (5, 73):
                raise RuntimeError(f"Legacy E 期望 (B,5,73)，实际 {tuple(energy.shape)}")
            return energy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.band_energy(x)
        latent = self.activation(self.norm(self.conv_in(latent)))
        latent = self.activation(self.conv_out(latent))
        latent = F.interpolate(latent, size=1800, mode="linear", align_corners=False)
        return self.projection(latent)


class AnalyticAMRepresentation(nn.Module):
    """S2A Analytic A：高频 analytic magnitude 的低频幅度调制表征。"""

    def __init__(self, initialization_seed: int) -> None:
        super().__init__()
        with module_seed(initialization_seed, "analytic_am_filterbank"):
            self.filterbank = AmplitudeModulationFilterbank(sample_rate=100.0, length=18000)
        with module_seed(initialization_seed, "analytic_am_encoder"):
            self.conv_in = nn.Conv1d(8, 48, kernel_size=5, padding=2, bias=False)
            self.norm_in = nn.GroupNorm(6, 48, eps=1e-5, affine=True)
            self.conv_out = nn.Conv1d(48, 96, kernel_size=1, bias=False)
            self.norm_out = nn.GroupNorm(12, 96, eps=1e-5, affine=True)
            self.residual = nn.Sequential(
                ResidualDWBlock(96, dilation=1),
                ResidualDWBlock(96, dilation=2),
                ResidualDWBlock(96, dilation=4),
            )
            _initialize_conv(self.conv_in)
            _initialize_norm(self.norm_in)
            _initialize_conv(self.conv_out)
            _initialize_norm(self.norm_out)
        with module_seed(initialization_seed, "analytic_am_projection"):
            self.projection = nn.Conv1d(96, 96, kernel_size=1, bias=True)
            _zero_projection(self.projection)
        self.activation = nn.SiLU()

    def amplitude_modulation(self, x: torch.Tensor) -> torch.Tensor:
        _validate_representation_input(x)
        analytic = self.filterbank(x)
        with torch.amp.autocast(x.device.type, enabled=False):
            magnitude = torch.log1p(analytic.abs()).float()
            envelope = fft_hard_bandpass(
                magnitude,
                sample_rate=100.0,
                low_hz=0.03,
                high_hz=0.80,
                include_low=True,
            )[..., ::10]
            if envelope.shape[1:] != (8, 1800):
                raise RuntimeError(f"Analytic A 期望 (B,8,1800)，实际 {tuple(envelope.shape)}")
            return envelope.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.amplitude_modulation(x)
        latent = self.activation(self.norm_in(self.conv_in(latent)))
        latent = self.activation(self.norm_out(self.conv_out(latent)))
        latent = self.residual(latent)
        return self.projection(latent)


class _MorphologyWindowEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_in = nn.Conv1d(1, 32, kernel_size=9, bias=False)
        self.norm_in = nn.GroupNorm(4, 32, eps=1e-5, affine=True)
        self.conv_stride = nn.Conv1d(32, 32, kernel_size=7, stride=2, bias=False)
        self.norm_stride = nn.GroupNorm(4, 32, eps=1e-5, affine=True)
        self.depthwise = nn.Conv1d(32, 32, kernel_size=5, padding=2, groups=32, bias=False)
        self.conv_out = nn.Conv1d(32, 64, kernel_size=1, bias=False)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.embedding = nn.Linear(64, 96, bias=True)
        self.embedding_norm = nn.LayerNorm(96, eps=1e-5, elementwise_affine=True)
        self.activation = nn.SiLU()

        for conv in (self.conv_in, self.conv_stride, self.depthwise, self.conv_out):
            _initialize_conv(conv)
        for norm in (self.norm_in, self.norm_stride, self.embedding_norm):
            _initialize_norm(norm)
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5.0))
        nn.init.zeros_(self.embedding.bias)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        features = self.activation(self.norm_in(self.conv_in(windows)))
        features = self.activation(self.norm_stride(self.conv_stride(features)))
        features = self.activation(self.depthwise(features))
        features = self.activation(self.conv_out(features))
        features = self.pool(features).squeeze(-1)
        return self.embedding_norm(self.embedding(features))


class MorphologyRepresentation(nn.Module):
    """S2A Morphology M：幅度归一化局部形态与正交 prototype package。"""

    token_chunk_size = 128
    prototype_count = 16
    prototype_temperature = 0.10

    def __init__(self, initialization_seed: int) -> None:
        super().__init__()
        with module_seed(initialization_seed, "morphology_encoder"):
            self.encoder = _MorphologyWindowEncoder()
            self.fusion = nn.Linear(112, 96, bias=True)
            self.fusion_norm = nn.LayerNorm(96, eps=1e-5, elementwise_affine=True)
            nn.init.kaiming_uniform_(self.fusion.weight, a=math.sqrt(5.0))
            nn.init.zeros_(self.fusion.bias)
            _initialize_norm(self.fusion_norm)
        with module_seed(initialization_seed, "morphology_prototypes"):
            # 参数名 P 与 CRD AdamW 的显式 prototype no-decay 规则一致。
            self.P = nn.Parameter(torch.empty(self.prototype_count, 96, dtype=torch.float32))
            nn.init.orthogonal_(self.P)
        with module_seed(initialization_seed, "morphology_projection"):
            self.projection = nn.Conv1d(96, 96, kernel_size=1, bias=True)
            _zero_projection(self.projection)

    def normalized_windows(self, x: torch.Tensor) -> torch.Tensor:
        _validate_representation_input(x)
        with torch.amp.autocast(x.device.type, enabled=False):
            filtered = fft_hard_bandpass(
                x,
                sample_rate=100.0,
                low_hz=0.70,
                high_hz=8.0,
                include_low=False,
            )
            padded = F.pad(filtered.float(), (75, 75), mode="reflect")
            windows = padded.unfold(dimension=-1, size=151, step=10)
            windows = windows - windows.mean(dim=-1, keepdim=True)
            windows = windows * torch.rsqrt(windows.square().mean(dim=-1, keepdim=True) + 1e-6)
            if windows.shape[1:] != (1, 1800, 151):
                raise RuntimeError(f"Morphology M 期望 (B,1,1800,151)，实际 {tuple(windows.shape)}")
            return windows.float()

    def _prototype_scores(self, embeddings: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(embeddings.device.type, enabled=False):
            normalized_embeddings = F.normalize(embeddings.float(), p=2.0, dim=-1, eps=1e-8)
            normalized_prototypes = F.normalize(self.P.float(), p=2.0, dim=-1, eps=1e-8)
            logits = torch.matmul(normalized_embeddings, normalized_prototypes.transpose(0, 1))
            return torch.softmax(logits / self.prototype_temperature, dim=-1).float()

    def prototype_loss(self) -> torch.Tensor:
        with torch.amp.autocast(self.P.device.type, enabled=False):
            normalized = F.normalize(self.P.float(), p=2.0, dim=-1, eps=1e-8)
            gram = normalized @ normalized.transpose(0, 1)
            off_diagonal = gram - torch.diag_embed(torch.diagonal(gram))
            denominator = self.prototype_count * (self.prototype_count - 1)
            return off_diagonal.square().sum() / float(denominator)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        windows = self.normalized_windows(x)
        batch_size = int(windows.shape[0])
        token_embeddings = []
        for start in range(0, 1800, self.token_chunk_size):
            stop = min(start + self.token_chunk_size, 1800)
            chunk = windows[:, :, start:stop, :]
            chunk = chunk.permute(0, 2, 1, 3).reshape(batch_size * (stop - start), 1, 151)
            embedding = self.encoder(chunk).reshape(batch_size, stop - start, 96)
            scores = self._prototype_scores(embedding)
            fused = self.fusion_norm(self.fusion(torch.cat((embedding, scores), dim=-1)))
            token_embeddings.append(fused)
        latent = torch.cat(token_embeddings, dim=1).transpose(1, 2)
        if latent.shape[1:] != (96, 1800):
            raise RuntimeError(f"Morphology M latent 契约错误: {tuple(latent.shape)}")
        return self.projection(latent)


__all__ = [
    "AnalyticAMRepresentation",
    "LegacyEnergyRepresentation",
    "MorphologyRepresentation",
]
