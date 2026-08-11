from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import torch
from torch import nn


PROTOCOL = "crd-tf-v1-research-informed-20260812"
SAMPLE_RATE = 100.0
WINDOW_SAMPLES = 18000
POOL_SAMPLES = 50
CONTEXT_LENGTH = 360
MORLET_MU = 13.4
W_VOICES_PER_OCTAVE = 12
W_SCALE_COUNT = 97
S_VOICES_PER_OCTAVE = 48
S_SCALE_COUNT = 387
RIDGE_EPS = 1e-8
RIDGE_CONFIDENCE_MIN = 1e-4


@dataclass(frozen=True)
class RidgeParameters:
    smoothness_penalty: float
    suppression_radius_bins: int

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.smoothness_penalty)) or float(self.smoothness_penalty) < 0.0:
            raise ValueError("ridge smoothness_penalty 必须是有限非负数")
        if int(self.suppression_radius_bins) < 0:
            raise ValueError("ridge suppression_radius_bins 必须非负")


def multires_stft_features(waveform: np.ndarray | torch.Tensor) -> dict[str, np.ndarray]:
    """计算协议固定的 M slow/fast log-magnitude，不执行可学习编码。"""

    values = _waveform_tensor(waveform)
    slow = _stft_feature_branch(values, n_fft=3000, hop_length=150, low_hz=0.03, high_hz=1.20)
    fast = _stft_feature_branch(values, n_fft=600, hop_length=50, low_hz=0.70, high_hz=8.00)
    if slow.shape != (36, 101) or fast.shape != (44, 349):
        raise RuntimeError(f"M feature shape 错误: slow={slow.shape}, fast={fast.shape}")
    return {"m_slow": slow, "m_fast": fast}


def _stft_feature_branch(
    waveform: torch.Tensor,
    *,
    n_fft: int,
    hop_length: int,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    frames = waveform.unfold(0, int(n_fft), int(hop_length)).float()
    frames = frames - frames.mean(dim=-1, keepdim=True)
    window = torch.hann_window(int(n_fft), periodic=False, dtype=torch.float32, device=frames.device)
    spectrum = torch.fft.rfft(frames * window, n=int(n_fft), norm="backward", dim=-1)
    frequencies = torch.fft.rfftfreq(int(n_fft), d=1.0 / SAMPLE_RATE, device=frames.device)
    selected = (frequencies >= float(low_hz)) & (frequencies <= float(high_hz))
    feature = torch.log1p(spectrum.abs())[:, selected].transpose(0, 1).contiguous()
    output = feature.cpu().numpy().astype(np.float32, copy=False)
    _require_finite("M STFT", output)
    return output


def morlet_target_frequencies(*, voices_per_octave: int, count: int) -> np.ndarray:
    frequencies = 8.0 * np.power(2.0, -np.arange(int(count), dtype=np.float64) / int(voices_per_octave))
    return np.sort(frequencies).astype(np.float64, copy=False)


@lru_cache(maxsize=4)
def morlet_scales_and_frequencies(
    voices_per_octave: int,
    count: int,
    length: int = WINDOW_SAMPLES,
    sample_rate: float = SAMPLE_RATE,
) -> tuple[np.ndarray, np.ndarray]:
    """把协议目标频率映射到 ssqueezepy scales，并返回实际离散中心频率。"""

    _require_ssqueezepy_version()
    from ssqueezepy import Wavelet
    from ssqueezepy.experimental import freq_to_scale, scale_to_freq

    target = morlet_target_frequencies(voices_per_octave=int(voices_per_octave), count=int(count))
    wavelet = Wavelet(("morlet", {"mu": MORLET_MU}), N=int(length), dtype="float32")
    search_count = max(4096, 20 * int(count))
    # freq_to_scale 返回大尺度到小尺度；cwt 要求 scales 单调递增。
    scales = freq_to_scale(
        target,
        wavelet,
        int(length),
        fs=float(sample_rate),
        n_search_scales=search_count,
        kind="peak",
        base=2,
    )[::-1]
    actual = np.asarray(
        scale_to_freq(scales, wavelet, int(length), fs=float(sample_rate), padtype="reflect"),
        dtype=np.float64,
    )
    if scales.shape != (int(count),) or actual.shape != (int(count),):
        raise RuntimeError("Morlet scale/frequency 数量错误")
    if not np.all(np.diff(scales) > 0.0):
        raise RuntimeError("Morlet scales 必须严格递增")
    _require_finite("Morlet scales", scales)
    _require_finite("Morlet actual frequencies", actual)
    scales = np.asarray(scales, dtype=np.float64)
    scales.setflags(write=False)
    actual.setflags(write=False)
    return scales, actual


def cwt_magnitude_features(waveform: np.ndarray | torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """计算固定 W CWT log-magnitude，并按实际中心频率升序输出。"""

    _require_ssqueezepy_version()
    from ssqueezepy import Wavelet, cwt
    from ssqueezepy.experimental import scale_to_freq

    values = _waveform_numpy(waveform)
    scales, _ = morlet_scales_and_frequencies(W_VOICES_PER_OCTAVE, W_SCALE_COUNT)
    wavelet = Wavelet(("morlet", {"mu": MORLET_MU}), N=WINDOW_SAMPLES, dtype="float32")
    coefficients, returned_scales = cwt(
        values,
        wavelet=wavelet,
        scales=scales,
        fs=SAMPLE_RATE,
        padtype="reflect",
        rpadded=False,
        vectorized=True,
        astensor=False,
        nan_checks=False,
    )
    actual = np.asarray(
        scale_to_freq(returned_scales, wavelet, WINDOW_SAMPLES, fs=SAMPLE_RATE, padtype="reflect"),
        dtype=np.float64,
    )
    order = np.argsort(actual, kind="stable")
    magnitude = np.log1p(np.abs(np.asarray(coefficients)[order])).astype(np.float32)
    pooled = magnitude.reshape(W_SCALE_COUNT, CONTEXT_LENGTH, POOL_SAMPLES).mean(axis=-1, dtype=np.float32)
    frequencies = actual[order]
    if pooled.shape != (W_SCALE_COUNT, CONTEXT_LENGTH):
        raise RuntimeError(f"W feature shape 错误: {pooled.shape}")
    _require_finite("W CWT", pooled)
    return np.ascontiguousarray(pooled), np.ascontiguousarray(frequencies)


def wsst_energy_map(waveform: np.ndarray | torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """计算固定 S WSST magnitude map，并池化到 2 Hz；ridge 提取在其后独立执行。"""

    _require_ssqueezepy_version()
    from ssqueezepy import Wavelet, ssq_cwt

    values = _waveform_numpy(waveform)
    scales, _ = morlet_scales_and_frequencies(S_VOICES_PER_OCTAVE, S_SCALE_COUNT)
    wavelet = Wavelet(("morlet", {"mu": MORLET_MU}), N=WINDOW_SAMPLES, dtype="float32")
    squeezed, _, frequencies, _ = ssq_cwt(
        values,
        wavelet=wavelet,
        scales=scales,
        nv=S_VOICES_PER_OCTAVE,
        fs=SAMPLE_RATE,
        ssq_freqs=None,
        padtype="reflect",
        squeezing="sum",
        maprange="peak",
        difftype="trig",
        vectorized=True,
        preserve_transform=None,
        astensor=False,
        flipud=True,
        nan_checks=False,
    )
    frequencies = np.asarray(frequencies, dtype=np.float64)
    order = np.argsort(frequencies, kind="stable")
    magnitude = np.abs(np.asarray(squeezed)[order]).astype(np.float32)
    pooled = magnitude.reshape(S_SCALE_COUNT, CONTEXT_LENGTH, POOL_SAMPLES).mean(axis=-1, dtype=np.float32)
    frequencies = frequencies[order]
    if pooled.shape != (S_SCALE_COUNT, CONTEXT_LENGTH):
        raise RuntimeError(f"S WSST shape 错误: {pooled.shape}")
    _require_finite("S WSST", pooled)
    return np.ascontiguousarray(pooled), np.ascontiguousarray(frequencies)


def wsst_ridge_features(
    energy: np.ndarray,
    frequencies: np.ndarray,
    parameters: RidgeParameters,
) -> np.ndarray:
    """从已池化 WSST magnitude 中提取两个频带各两条 deterministic ridge。"""

    values = np.asarray(energy, dtype=np.float32)
    freqs = np.asarray(frequencies, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != freqs.size or values.shape[1] != CONTEXT_LENGTH:
        raise ValueError(f"WSST ridge 输入 shape 错误: energy={values.shape}, frequencies={freqs.shape}")
    if not np.all(np.diff(freqs) >= 0.0) or np.any(freqs <= 0.0):
        raise ValueError("WSST frequencies 必须为正且单调非降")
    _require_finite("WSST ridge energy", values)
    _require_finite("WSST ridge frequencies", freqs)

    respiratory = (freqs >= 0.03) & (freqs <= 0.80)
    carrier = (freqs > 0.80) & (freqs <= 8.00)
    if not respiratory.any() or not carrier.any():
        raise RuntimeError("WSST ridge 频带没有可用 bins")
    parts = [
        _band_ridge_features(values[respiratory], freqs[respiratory], parameters),
        _band_ridge_features(values[carrier], freqs[carrier], parameters),
    ]
    output = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
    if output.shape != (12, CONTEXT_LENGTH):
        raise RuntimeError(f"S ridge feature shape 错误: {output.shape}")
    _require_finite("S ridge feature", output)
    return np.ascontiguousarray(output)


def _band_ridge_features(
    energy: np.ndarray,
    frequencies: np.ndarray,
    parameters: RidgeParameters,
) -> np.ndarray:
    original = np.asarray(energy, dtype=np.float64)
    work = original.copy()
    band_total = original.sum(axis=0)
    features: list[np.ndarray] = []
    for _ in range(2):
        path = _ridge_path(work, frequencies, float(parameters.smoothness_penalty))
        times = np.arange(work.shape[1])
        ridge_energy = original[path, times]
        confidence = ridge_energy / (band_total + RIDGE_EPS)
        valid = (band_total > RIDGE_EPS) & (confidence >= RIDGE_CONFIDENCE_MIN)
        ridge_frequency = np.where(valid, frequencies[path], 0.0)
        ridge_amplitude = np.where(valid, np.log1p(ridge_energy), 0.0)
        ridge_confidence = np.where(valid, np.clip(confidence, 0.0, 1.0), 0.0)
        features.extend((ridge_frequency, ridge_amplitude, ridge_confidence))

        radius = int(parameters.suppression_radius_bins)
        for time_index, center in enumerate(path.tolist()):
            low = max(0, int(center) - radius)
            high = min(work.shape[0], int(center) + radius + 1)
            work[low:high, time_index] = 0.0
    return np.stack(features, axis=0).astype(np.float32)


def _ridge_path(energy: np.ndarray, frequencies: np.ndarray, penalty: float) -> np.ndarray:
    """用 O(F*T) max-plus L1 distance transform 求 log-frequency 平滑 ridge。"""

    values = np.asarray(energy, dtype=np.float64)
    freqs = np.asarray(frequencies, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != freqs.size or values.shape[1] <= 0:
        raise ValueError("ridge path 输入 shape 错误")
    coordinates = np.log2(freqs)
    emission = np.log(values + RIDGE_EPS)
    frequency_count, time_count = values.shape
    back = np.zeros((time_count, frequency_count), dtype=np.int32)
    score = emission[:, 0].copy()
    for time_index in range(1, time_count):
        best_value, predecessor = _max_plus_l1(score, coordinates, float(penalty))
        score = emission[:, time_index] + best_value
        back[time_index] = predecessor
    final_candidates = np.flatnonzero(score == np.max(score))
    current = int(final_candidates[0])
    path = np.empty(time_count, dtype=np.int32)
    path[-1] = current
    for time_index in range(time_count - 1, 0, -1):
        current = int(back[time_index, current])
        path[time_index - 1] = current
    return path


def _max_plus_l1(values: np.ndarray, coordinates: np.ndarray, penalty: float) -> tuple[np.ndarray, np.ndarray]:
    count = int(values.size)
    left_value = np.empty(count, dtype=np.float64)
    left_index = np.empty(count, dtype=np.int32)
    running_value = -math.inf
    running_index = 0
    for index in range(count):
        candidate = float(values[index] + penalty * coordinates[index])
        if candidate > running_value or (candidate == running_value and index < running_index):
            running_value = candidate
            running_index = index
        left_value[index] = running_value - penalty * coordinates[index]
        left_index[index] = running_index

    right_value = np.empty(count, dtype=np.float64)
    right_index = np.empty(count, dtype=np.int32)
    running_value = -math.inf
    running_index = count - 1
    for index in range(count - 1, -1, -1):
        candidate = float(values[index] - penalty * coordinates[index])
        if candidate > running_value or (candidate == running_value and index < running_index):
            running_value = candidate
            running_index = index
        right_value[index] = running_value + penalty * coordinates[index]
        right_index[index] = running_index

    choose_left = (left_value > right_value) | (
        (left_value == right_value) & (left_index <= right_index)
    )
    best = np.where(choose_left, left_value, right_value)
    predecessor = np.where(choose_left, left_index, right_index).astype(np.int32)
    return best, predecessor


class LearnableCarrierModulation(nn.Module):
    """P1 的 24-band learnable analytic carrier-modulation 前端；尚未接入 CRD 模型。"""

    filter_count = 24

    def __init__(self, *, chunk_filters: int = 6) -> None:
        super().__init__()
        if int(chunk_filters) <= 0:
            raise ValueError("chunk_filters 必须为正")
        self.chunk_filters = int(chunk_filters)
        centers = torch.logspace(math.log10(0.7), math.log10(8.0), self.filter_count, dtype=torch.float32)
        midpoints = torch.sqrt(centers[:-1] * centers[1:])
        lower = torch.cat((torch.tensor([0.7], dtype=torch.float32), midpoints))
        upper = torch.cat((midpoints, torch.tensor([8.0], dtype=torch.float32)))
        fraction = (centers - lower) / (upper - lower)
        rho_fraction = torch.full((self.filter_count,), (0.15 - 0.08) / (0.30 - 0.08), dtype=torch.float32)
        self.center_logits = nn.Parameter(_inverse_logit(fraction))
        self.bandwidth_logits = nn.Parameter(_inverse_logit(rho_fraction))
        self.register_buffer("center_lower", lower, persistent=True)
        self.register_buffer("center_upper", upper, persistent=True)

    def centers(self) -> torch.Tensor:
        return self.center_lower + torch.sigmoid(self.center_logits) * (self.center_upper - self.center_lower)

    def rhos(self) -> torch.Tensor:
        return 0.08 + torch.sigmoid(self.bandwidth_logits) * (0.30 - 0.08)

    def forward(
        self,
        waveform: torch.Tensor | None = None,
        *,
        one_sided_spectrum: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (waveform is None) == (one_sided_spectrum is None):
            raise ValueError("waveform 与 one_sided_spectrum 必须且只能提供一个")
        if waveform is not None:
            if waveform.ndim == 2:
                waveform = waveform[:, None, :]
            if waveform.ndim != 3 or waveform.shape[1:] != (1, WINDOW_SAMPLES):
                raise ValueError(f"L waveform 期望 (B,1,{WINDOW_SAMPLES})，实际 {tuple(waveform.shape)}")
            if not bool(torch.isfinite(waveform).all()):
                raise FloatingPointError("L waveform 包含 NaN/Inf")
            spectrum = torch.fft.rfft(waveform[:, 0].float(), n=WINDOW_SAMPLES, norm="backward")
        else:
            assert one_sided_spectrum is not None
            spectrum = one_sided_spectrum
            if spectrum.ndim == 1:
                spectrum = spectrum[None, :]
            if spectrum.ndim != 2 or spectrum.shape[1] != WINDOW_SAMPLES // 2 + 1:
                raise ValueError(f"L cached spectrum 期望 (B,9001)，实际 {tuple(spectrum.shape)}")
            if not torch.is_complex(spectrum):
                raise ValueError("L cached spectrum 必须是 complex tensor")
            if not bool(torch.isfinite(spectrum.real).all() and torch.isfinite(spectrum.imag).all()):
                raise FloatingPointError("L cached spectrum 包含 NaN/Inf")
            spectrum = spectrum.to(dtype=torch.complex64)

        frequencies = torch.fft.rfftfreq(
            WINDOW_SAMPLES,
            d=1.0 / SAMPLE_RATE,
            device=spectrum.device,
        )
        centers = self.centers().to(device=spectrum.device, dtype=torch.float32)
        bandwidths = (centers * self.rhos().to(device=spectrum.device, dtype=torch.float32)).clamp_min(1e-6)
        outputs = []
        for start in range(0, self.filter_count, self.chunk_filters):
            stop = min(start + self.chunk_filters, self.filter_count)
            delta = (frequencies[None, :] - centers[start:stop, None]) / bandwidths[start:stop, None]
            masks = torch.exp(-0.5 * delta.square())
            support = ((frequencies >= 0.70) & (frequencies <= 8.00)).to(masks.dtype)
            positive = 2.0 * spectrum[:, None, :] * (masks * support[None, :])[None, :, :]
            analytic_spectrum = torch.zeros(
                spectrum.shape[0],
                stop - start,
                WINDOW_SAMPLES,
                device=spectrum.device,
                dtype=torch.complex64,
            )
            analytic_spectrum[..., : WINDOW_SAMPLES // 2 + 1] = positive
            analytic = torch.fft.ifft(analytic_spectrum, n=WINDOW_SAMPLES, norm="backward", dim=-1)
            outputs.append(torch.log1p(analytic.abs().square()))
        envelope = torch.cat(outputs, dim=1)
        envelope_spectrum = torch.fft.rfft(envelope.float(), n=WINDOW_SAMPLES, norm="backward", dim=-1)
        modulation_frequencies = torch.fft.rfftfreq(
            WINDOW_SAMPLES,
            d=1.0 / SAMPLE_RATE,
            device=envelope.device,
        )
        band = (modulation_frequencies >= 0.03) & (modulation_frequencies <= 0.80)
        filtered = torch.fft.irfft(envelope_spectrum * band, n=WINDOW_SAMPLES, norm="backward", dim=-1)
        output = filtered[..., ::POOL_SAMPLES]
        if output.shape[1:] != (self.filter_count, CONTEXT_LENGTH):
            raise RuntimeError(f"L output shape 错误: {tuple(output.shape)}")
        if not bool(torch.isfinite(output).all()):
            raise FloatingPointError("L output 包含 NaN/Inf")
        return output.float()


def one_sided_input_spectrum(waveform: np.ndarray | torch.Tensor) -> np.ndarray:
    values = _waveform_tensor(waveform)
    spectrum = torch.fft.rfft(values, n=WINDOW_SAMPLES, norm="backward")
    output = spectrum.cpu().numpy().astype(np.complex64, copy=False)
    if output.shape != (WINDOW_SAMPLES // 2 + 1,):
        raise RuntimeError(f"L input spectrum shape 错误: {output.shape}")
    _require_complex_finite("L input spectrum", output)
    return np.ascontiguousarray(output)


def _inverse_logit(value: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(value.dtype).eps
    value = value.clamp(eps, 1.0 - eps)
    return torch.log(value) - torch.log1p(-value)


def _waveform_tensor(waveform: np.ndarray | torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(waveform, dtype=torch.float32)
    if values.ndim == 2 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 1 or values.numel() != WINDOW_SAMPLES:
        raise ValueError(f"TF-v1 waveform 期望 ({WINDOW_SAMPLES},)，实际 {tuple(values.shape)}")
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("TF-v1 waveform 包含 NaN/Inf")
    return values.contiguous()


def _waveform_numpy(waveform: np.ndarray | torch.Tensor) -> np.ndarray:
    return _waveform_tensor(waveform).cpu().numpy().astype(np.float32, copy=False)


def _require_ssqueezepy_version() -> None:
    import importlib.metadata

    actual = importlib.metadata.version("ssqueezepy")
    if actual != "0.6.6":
        raise RuntimeError(f"CRD-TF v1 要求 ssqueezepy==0.6.6，当前为 {actual}")


def _require_finite(label: str, values: np.ndarray) -> None:
    if not np.isfinite(np.asarray(values)).all():
        raise FloatingPointError(f"{label} 包含 NaN/Inf")


def _require_complex_finite(label: str, values: np.ndarray) -> None:
    array = np.asarray(values)
    if not np.isfinite(array.real).all() or not np.isfinite(array.imag).all():
        raise FloatingPointError(f"{label} 包含 NaN/Inf")


def fixed_transform_spec() -> dict[str, Any]:
    """返回参与 cache identity 的纯 JSON 固定数学规格。"""

    return {
        "protocol": PROTOCOL,
        "sample_rate": SAMPLE_RATE,
        "window_samples": WINDOW_SAMPLES,
        "pool_samples": POOL_SAMPLES,
        "m": {
            "window": "symmetric_hann",
            "center": False,
            "padding": False,
            "feature": "log1p_abs",
            "slow": {"n_fft": 3000, "hop": 150, "low_hz": 0.03, "high_hz": 1.20, "shape": [36, 101]},
            "fast": {"n_fft": 600, "hop": 50, "low_hz": 0.70, "high_hz": 8.00, "shape": [44, 349]},
        },
        "w": {
            "library": "ssqueezepy==0.6.6",
            "wavelet": "morlet",
            "mu": MORLET_MU,
            "voices_per_octave": W_VOICES_PER_OCTAVE,
            "scale_count": W_SCALE_COUNT,
            "padtype": "reflect",
            "feature": "mean50_log1p_abs",
            "shape": [97, 360],
        },
        "s": {
            "library": "ssqueezepy==0.6.6",
            "wavelet": "morlet",
            "mu": MORLET_MU,
            "voices_per_octave": S_VOICES_PER_OCTAVE,
            "scale_count": S_SCALE_COUNT,
            "padtype": "reflect",
            "pool": "mean50_abs",
            "shape": [12, 360],
            "confidence_min": RIDGE_CONFIDENCE_MIN,
        },
        "l_spectrum": {
            "fft": "rfft_norm_backward",
            "dtype": "complex64",
            "shape": [9001],
        },
    }
