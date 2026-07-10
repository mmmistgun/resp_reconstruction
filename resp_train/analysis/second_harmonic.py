from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as scipy_signal

from resp_train.metrics.signal import estimate_robust_peak_rate_bpm


ELIGIBLE = "eligible"
THO_REFERENCE_UNSTABLE = "tho_reference_unstable"
SECOND_HARMONIC_OUT_OF_BAND = "second_harmonic_out_of_band"


@dataclass(frozen=True)
class HarmonicFeatureConfig:
    fs: float = 100.0
    low_hz: float = 0.05
    high_hz: float = 0.7
    filter_order: int = 4
    welch_nperseg: int = 4096
    neighborhood_hz: float = 0.025
    energy_floor: float = 1e-12
    tho_rr_agreement_bpm: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.fs) or self.fs <= 0:
            raise ValueError(f"fs 必须为正数，当前={self.fs}")
        if not 0 < self.low_hz < self.high_hz < self.fs / 2:
            raise ValueError(
                f"频带必须满足 0 < low_hz < high_hz < Nyquist，当前={self.low_hz}, {self.high_hz}"
            )
        if int(self.filter_order) <= 0:
            raise ValueError(f"filter_order 必须为正整数，当前={self.filter_order}")
        if int(self.welch_nperseg) < 2:
            raise ValueError(f"welch_nperseg 必须 >= 2，当前={self.welch_nperseg}")
        if not np.isfinite(self.neighborhood_hz) or self.neighborhood_hz <= 0:
            raise ValueError(f"neighborhood_hz 必须为正数，当前={self.neighborhood_hz}")
        if not np.isfinite(self.energy_floor) or self.energy_floor <= 0:
            raise ValueError(f"energy_floor 必须为正数，当前={self.energy_floor}")
        if not np.isfinite(self.tho_rr_agreement_bpm) or self.tho_rr_agreement_bpm < 0:
            raise ValueError(
                f"tho_rr_agreement_bpm 必须为非负数，当前={self.tho_rr_agreement_bpm}"
            )


@dataclass(frozen=True)
class HarmonicThresholds:
    version: str
    tho_rr_agreement_bpm: float
    peak_relative_tolerance: float
    harmonic_to_fundamental_min: float
    harmonic_band_fraction_min: float
    correction_ratio_drop_min: float

    def __post_init__(self) -> None:
        if not str(self.version).strip():
            raise ValueError("version 不能为空")
        if not np.isfinite(self.tho_rr_agreement_bpm) or self.tho_rr_agreement_bpm < 0:
            raise ValueError("tho_rr_agreement_bpm 必须为非负数")
        if not np.isfinite(self.peak_relative_tolerance) or not 0 <= self.peak_relative_tolerance < 1:
            raise ValueError("peak_relative_tolerance 必须位于 [0, 1)")
        if not np.isfinite(self.harmonic_to_fundamental_min) or self.harmonic_to_fundamental_min < 0:
            raise ValueError("harmonic_to_fundamental_min 必须为非负数")
        if not np.isfinite(self.harmonic_band_fraction_min) or not 0 <= self.harmonic_band_fraction_min <= 1:
            raise ValueError("harmonic_band_fraction_min 必须位于 [0, 1]")
        if not np.isfinite(self.correction_ratio_drop_min) or not 0 <= self.correction_ratio_drop_min <= 1:
            raise ValueError("correction_ratio_drop_min 必须位于 [0, 1]")


@dataclass(frozen=True)
class HarmonicFeatures:
    status: str
    tho_reference_hz: float
    tho_robust_rr_bpm: float
    tho_spectral_rr_bpm: float
    bcg_peak_hz: float
    peak_to_tho_ratio: float
    peak_second_harmonic_relative_error: float
    fundamental_energy: float
    second_harmonic_energy: float
    band_energy: float
    harmonic_to_fundamental_ratio: float
    harmonic_band_fraction: float


def resolve_eligibility_status(
    tho_robust_rr_bpm: float,
    tho_spectral_rr_bpm: float,
    *,
    cfg: HarmonicFeatureConfig,
) -> str:
    robust = float(tho_robust_rr_bpm)
    spectral = float(tho_spectral_rr_bpm)
    if not np.isfinite(robust) or not np.isfinite(spectral) or robust <= 0 or spectral <= 0:
        return THO_REFERENCE_UNSTABLE
    if abs(robust - spectral) > cfg.tho_rr_agreement_bpm:
        return THO_REFERENCE_UNSTABLE
    tho_reference_hz = robust / 60.0
    if 2.0 * tho_reference_hz > cfg.high_hz:
        return SECOND_HARMONIC_OUT_OF_BAND
    return ELIGIBLE


def extract_harmonic_features(
    bcg: np.ndarray,
    tho: np.ndarray,
    *,
    cfg: HarmonicFeatureConfig,
) -> HarmonicFeatures:
    bcg_x = _as_finite_1d(bcg, name="BCG")
    tho_x = _as_finite_1d(tho, name="THO")
    if bcg_x.shape != tho_x.shape:
        raise ValueError(f"BCG 和 THO 长度必须一致，当前 {bcg_x.shape} != {tho_x.shape}")

    bcg_band = _bandpass_resp(bcg_x, cfg)
    tho_band = _bandpass_resp(tho_x, cfg)
    tho_freqs, tho_power = _welch_band(tho_band, cfg)
    tho_spectral_rr_bpm = _spectral_rate_bpm(tho_freqs, tho_power)
    tho_robust_rr_bpm = estimate_robust_peak_rate_bpm(
        tho_band,
        fs=cfg.fs,
        low_hz=cfg.low_hz,
        high_hz=cfg.high_hz,
    )
    status = resolve_eligibility_status(tho_robust_rr_bpm, tho_spectral_rr_bpm, cfg=cfg)
    tho_reference_hz = float(tho_robust_rr_bpm / 60.0) if np.isfinite(tho_robust_rr_bpm) else float("nan")

    bcg_freqs, bcg_power = _welch_band(bcg_band, cfg)
    bcg_peak_hz = _peak_frequency(bcg_freqs, bcg_power)
    if np.isfinite(tho_reference_hz) and tho_reference_hz > 0:
        peak_to_tho_ratio = float(bcg_peak_hz / tho_reference_hz)
        second_harmonic_hz = 2.0 * tho_reference_hz
        peak_second_harmonic_relative_error = float(
            abs(bcg_peak_hz - second_harmonic_hz) / second_harmonic_hz
        )
        fundamental_energy = _neighborhood_energy(
            bcg_freqs,
            bcg_power,
            center_hz=tho_reference_hz,
            half_width_hz=cfg.neighborhood_hz,
        )
        second_harmonic_energy = _neighborhood_energy(
            bcg_freqs,
            bcg_power,
            center_hz=second_harmonic_hz,
            half_width_hz=cfg.neighborhood_hz,
        )
    else:
        peak_to_tho_ratio = float("nan")
        peak_second_harmonic_relative_error = float("nan")
        fundamental_energy = 0.0
        second_harmonic_energy = 0.0

    band_energy = _band_energy(bcg_freqs, bcg_power)
    harmonic_to_fundamental_ratio = float(
        second_harmonic_energy / max(fundamental_energy, cfg.energy_floor)
    )
    harmonic_band_fraction = float(second_harmonic_energy / max(band_energy, cfg.energy_floor))
    return HarmonicFeatures(
        status=status,
        tho_reference_hz=tho_reference_hz,
        tho_robust_rr_bpm=float(tho_robust_rr_bpm),
        tho_spectral_rr_bpm=float(tho_spectral_rr_bpm),
        bcg_peak_hz=bcg_peak_hz,
        peak_to_tho_ratio=peak_to_tho_ratio,
        peak_second_harmonic_relative_error=peak_second_harmonic_relative_error,
        fundamental_energy=fundamental_energy,
        second_harmonic_energy=second_harmonic_energy,
        band_energy=band_energy,
        harmonic_to_fundamental_ratio=harmonic_to_fundamental_ratio,
        harmonic_band_fraction=harmonic_band_fraction,
    )


def classify_harmonic_window(features: HarmonicFeatures, thresholds: HarmonicThresholds) -> str:
    if features.status != ELIGIBLE:
        return features.status
    peak_doubling = (
        np.isfinite(features.peak_second_harmonic_relative_error)
        and features.peak_second_harmonic_relative_error <= thresholds.peak_relative_tolerance
    )
    harmonic_prominent = (
        np.isfinite(features.harmonic_to_fundamental_ratio)
        and np.isfinite(features.harmonic_band_fraction)
        and features.harmonic_to_fundamental_ratio >= thresholds.harmonic_to_fundamental_min
        and features.harmonic_band_fraction >= thresholds.harmonic_band_fraction_min
    )
    if peak_doubling and harmonic_prominent:
        return "strong_harmonic"
    if peak_doubling:
        return "peak_doubling"
    if harmonic_prominent:
        return "harmonic_prominent"
    return "harmonic_negative"


def classify_model_correction(
    input_features: HarmonicFeatures,
    output_features: HarmonicFeatures,
    thresholds: HarmonicThresholds,
) -> str:
    positive_labels = {"strong_harmonic", "peak_doubling", "harmonic_prominent"}
    if classify_harmonic_window(input_features, thresholds) not in positive_labels:
        return "not_corrected"
    if output_features.status != ELIGIBLE:
        return "not_corrected"
    input_ratio = float(input_features.harmonic_to_fundamental_ratio)
    output_ratio = float(output_features.harmonic_to_fundamental_ratio)
    if not np.isfinite(input_ratio) or not np.isfinite(output_ratio) or input_ratio <= 0:
        return "not_corrected"
    relative_drop = (input_ratio - output_ratio) / max(input_ratio, np.finfo(np.float64).eps)
    harmonic_dropped = relative_drop >= thresholds.correction_ratio_drop_min
    fundamental_peak = (
        np.isfinite(output_features.peak_to_tho_ratio)
        and abs(output_features.peak_to_tho_ratio - 1.0) <= thresholds.peak_relative_tolerance
    )
    if harmonic_dropped and fundamental_peak:
        return "corrected"
    if harmonic_dropped:
        return "partially_corrected"
    return "not_corrected"


def _as_finite_1d(signal: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(signal, dtype=np.float64).squeeze()
    if array.ndim != 1 or array.size < 2:
        raise ValueError(f"{name} 必须是一维且至少包含 2 个样本，当前 shape={array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} 包含非有限值")
    return array


def _bandpass_resp(signal: np.ndarray, cfg: HarmonicFeatureConfig) -> np.ndarray:
    sos = scipy_signal.butter(
        int(cfg.filter_order),
        [cfg.low_hz, cfg.high_hz],
        btype="bandpass",
        fs=cfg.fs,
        output="sos",
    )
    return np.asarray(scipy_signal.sosfiltfilt(sos, signal), dtype=np.float64)


def _welch_band(signal: np.ndarray, cfg: HarmonicFeatureConfig) -> tuple[np.ndarray, np.ndarray]:
    nperseg = min(int(cfg.welch_nperseg), int(signal.size))
    freqs, power = scipy_signal.welch(signal, fs=cfg.fs, nperseg=nperseg)
    mask = (freqs >= cfg.low_hz) & (freqs <= cfg.high_hz)
    if not mask.any():
        raise ValueError("Welch 频率网格在目标呼吸频带内没有频点")
    return np.asarray(freqs[mask], dtype=np.float64), np.asarray(power[mask], dtype=np.float64)


def _spectral_rate_bpm(freqs: np.ndarray, power: np.ndarray) -> float:
    if not np.isfinite(power).all() or power.size == 0 or float(np.max(power)) <= 0:
        return float("nan")
    return float(freqs[int(np.argmax(power))] * 60.0)


def _peak_frequency(freqs: np.ndarray, power: np.ndarray) -> float:
    if not np.isfinite(power).all() or power.size == 0 or float(np.max(power)) <= 0:
        return float("nan")
    return float(freqs[int(np.argmax(power))])


def _neighborhood_energy(
    freqs: np.ndarray,
    power: np.ndarray,
    *,
    center_hz: float,
    half_width_hz: float,
) -> float:
    mask = np.abs(freqs - float(center_hz)) <= float(half_width_hz)
    if not mask.any():
        return 0.0
    return float(max(np.sum(power[mask]) * _frequency_bin_width(freqs), 0.0))


def _band_energy(freqs: np.ndarray, power: np.ndarray) -> float:
    if power.size == 0:
        return 0.0
    return float(max(np.sum(power) * _frequency_bin_width(freqs), 0.0))


def _frequency_bin_width(freqs: np.ndarray) -> float:
    if freqs.size < 2:
        raise ValueError("频谱能量计算至少需要两个频率点")
    differences = np.diff(freqs)
    if not np.isfinite(differences).all() or np.any(differences <= 0):
        raise ValueError("频率网格必须有限且严格递增")
    return float(np.median(differences))
