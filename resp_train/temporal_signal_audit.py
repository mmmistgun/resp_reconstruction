from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Any, Iterable, Mapping
from uuid import uuid4

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from scipy.signal import fftconvolve, firwin, hilbert, resample, resample_poly

from resp_train.data.index import filter_index
from resp_train.data.research_v2 import ResearchV2WindowDataset, adapt_research_v2_index
from resp_train.metrics.task import (
    TaskMetricConfig,
    _envelope_metrics,
    _envelope_modulation,
    _envelope_stratum,
    _local_rr,
    _log_rms_envelope,
    _periodogram,
    _respiratory_coherence,
    _rr_from_power,
    _whole_rr,
)
from resp_train.protocols.respiration import (
    canonicalize_numpy,
    centered_energy_numpy,
    lag_priority,
    symmetric_hann_numpy,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs/resp_temporal_v1/signal_audit_train_v1.yaml"
CONFIG_SCHEMA_VERSION = "rtm-v1-signal-audit-config-v1"
RECEIPT_SCHEMA_VERSION = "rtm-v1-signal-audit-receipt-v1"
PROTOCOL_ID = "resp-temporal-v1-train-signal-audit-20260820"
FROZEN_CONFIG_SHA256 = "20591d70947d44d5b2f0e49ffec2d75eb29af1b20d9fa964d59442b3ced5ffb5"

TARGET_WINDOW_FILENAME = "target_timescale_window_metrics.csv"
TARGET_SEQUENCE_FILENAME = "target_timescale_sequence_metrics.csv"
TARGET_SAMPLE_FILENAME = "target_timescale_sample_metrics.csv"
PROXY_WINDOW_FILENAME = "proxy_observability_window_metrics.csv"
PROXY_SAMPLE_FILENAME = "proxy_observability_sample_metrics.csv"
SAMPLING_WINDOW_FILENAME = "sampling_order_window_metrics.csv"
SAMPLING_SAMPLE_FILENAME = "sampling_order_sample_metrics.csv"
GRID_WINDOW_FILENAME = "multiscale_grid_window_metrics.csv"
GRID_SAMPLE_FILENAME = "multiscale_grid_sample_metrics.csv"
SUMMARY_FILENAME = "sample_direct_summary.csv"
RESOLVED_CONFIG_FILENAME = "resolved_config.json"
MANIFEST_FILENAME = "artifact_manifest.json"
RECEIPT_FILENAME = "access_receipt.json"
RECEIPT_HASH_FILENAME = "access_receipt.sha256"


@dataclass(frozen=True)
class BandSpec:
    low_hz: float
    high_hz: float
    low_inclusive: bool
    high_inclusive: bool


@dataclass(frozen=True)
class ResampleSpec:
    up: int
    down: int
    numtaps: int
    cutoff_hz: float


@dataclass(frozen=True)
class SignalAuditConfig:
    path: Path
    raw: dict[str, Any]
    sha256: str

    @property
    def source_config_path(self) -> Path:
        return (REPO_ROOT / str(self.raw["source_config_path"])).resolve()

    @property
    def output_dir(self) -> Path:
        return (REPO_ROOT / str(self.raw["output"]["directory"])).resolve()

    def band(self, name: str) -> BandSpec:
        value = self.raw["bands"][name]
        return BandSpec(
            low_hz=float(value["low_hz"]),
            high_hz=float(value["high_hz"]),
            low_inclusive=bool(value["low_inclusive"]),
            high_inclusive=bool(value["high_inclusive"]),
        )

    def resample(self, name: str) -> ResampleSpec:
        value = self.raw["operators"]["resampling"]["maps"][name]
        return ResampleSpec(
            up=int(value["up"]),
            down=int(value["down"]),
            numtaps=int(value["numtaps"]),
            cutoff_hz=float(value["cutoff_hz"]),
        )


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_signal_audit_config(path: str | Path = DEFAULT_CONFIG_PATH) -> SignalAuditConfig:
    """加载不可调参的审计配置；任何字节级修改都需要先修订协议与代码。"""

    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"RTM-v1 signal-audit 配置不存在: {config_path}")
    digest = sha256_file(config_path)
    if digest != FROZEN_CONFIG_SHA256:
        raise ValueError(
            "RTM-v1 signal-audit 配置哈希不匹配；禁止用覆盖项、调带或修改算子参数旁路协议: "
            f"expected={FROZEN_CONFIG_SHA256}, actual={digest}"
        )
    raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("RTM-v1 signal-audit 配置必须是 mapping")
    _validate_config_semantics(raw)
    return SignalAuditConfig(path=config_path, raw=raw, sha256=digest)


def _validate_config_semantics(raw: Mapping[str, Any]) -> None:
    _require_exact_keys(
        raw,
        {
            "schema_version",
            "protocol_id",
            "source_config_path",
            "source_config_sha256",
            "access",
            "signal",
            "bands",
            "operators",
            "metrics",
            "output",
        },
        "config",
    )
    if raw["schema_version"] != CONFIG_SCHEMA_VERSION or raw["protocol_id"] != PROTOCOL_ID:
        raise ValueError("RTM-v1 signal-audit schema/protocol ID 不匹配")
    access = raw["access"]
    _require_exact_keys(
        access,
        {
            "split",
            "require_full_eligible_split",
            "expected_windows",
            "expected_samp_ids",
            "expected_dataset_row_ids_sha256",
            "validation_allowed",
            "research_test_allowed",
            "checkpoint_allowed",
            "model_training_allowed",
            "model_inference_allowed",
        },
        "config.access",
    )
    if access["split"] != "train" or not bool(access["require_full_eligible_split"]):
        raise ValueError("审计只允许完整 eligible train split")
    forbidden = (
        "validation_allowed",
        "research_test_allowed",
        "checkpoint_allowed",
        "model_training_allowed",
        "model_inference_allowed",
    )
    if any(bool(access[name]) for name in forbidden):
        raise ValueError("审计配置不得开放 validation/test/checkpoint/model access")
    signal = raw["signal"]
    _require_exact_keys(
        signal,
        {
            "sample_rate_hz",
            "duration_seconds",
            "duration_samples",
            "input_key",
            "target_key",
            "scale_eps",
            "dynamic_eps",
            "corr_eps",
            "envelope_eps",
        },
        "config.signal",
    )
    if (
        float(signal["sample_rate_hz"]) != 100.0
        or float(signal["duration_seconds"]) != 180.0
        or int(signal["duration_samples"]) != 18000
        or signal["input_key"] != "bcg_rawish_segment_soft_z_key"
        or signal["target_key"] != "target_waveform_segment_soft_z_key"
    ):
        raise ValueError("审计 input/target/采样率/窗口长度已冻结")
    expected_bands = {
        "displacement": (0.05, 0.80, True, True),
        "carrier_low": (0.80, 3.00, False, True),
        "carrier_high": (3.00, 8.00, False, True),
        "target": (0.05, 0.70, True, True),
    }
    _require_exact_keys(raw["bands"], set(expected_bands), "config.bands")
    for name, expected in expected_bands.items():
        band = raw["bands"][name]
        _require_exact_keys(band, {"low_hz", "high_hz", "low_inclusive", "high_inclusive"}, f"config.bands.{name}")
        actual = (float(band["low_hz"]), float(band["high_hz"]), bool(band["low_inclusive"]), bool(band["high_inclusive"]))
        if actual != expected:
            raise ValueError(f"频带 {name} 必须固定为 {expected}，实际 {actual}")
    output = raw["output"]
    _require_exact_keys(output, {"directory", "allow_overwrite", "require_clean_git", "csv_float_format"}, "config.output")
    if bool(output["allow_overwrite"]) or not bool(output["require_clean_git"]):
        raise ValueError("审计必须禁止覆盖并要求干净工作树")


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} 必须是 mapping")
    actual = set(value)
    if actual != expected:
        raise ValueError(f"{label} 字段不严格匹配: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")


def fft_band_extract(signal: np.ndarray, *, fs: float, band: BandSpec) -> np.ndarray:
    """float64 whole-window RFFT 矩形投影，显式落实四个频带的端点语义。"""

    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise FloatingPointError("band extraction 输入必须为非空有限波形")
    if not (0.0 <= band.low_hz < band.high_hz <= float(fs) / 2.0):
        raise ValueError(f"频带超出 Nyquist: {band}")
    centered = values - float(np.mean(values))
    spectrum = np.fft.rfft(centered, norm="backward")
    frequencies = np.fft.rfftfreq(values.size, d=1.0 / float(fs))
    lower = frequencies >= band.low_hz if band.low_inclusive else frequencies > band.low_hz
    upper = frequencies <= band.high_hz if band.high_inclusive else frequencies < band.high_hz
    return np.fft.irfft(spectrum * (lower & upper), n=values.size, norm="backward")


def carrier_log_envelope(
    carrier: np.ndarray,
    *,
    fs: float,
    target_band: BandSpec,
    eps: float,
) -> np.ndarray:
    """Hilbert analytic magnitude → log → 去均值 → 固定 target-band projection。"""

    values = np.asarray(carrier, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise FloatingPointError("carrier envelope 输入必须为非空有限波形")
    magnitude = np.abs(hilbert(values))
    log_envelope = np.log(magnitude + float(eps))
    log_envelope -= float(np.mean(log_envelope))
    result = fft_band_extract(log_envelope, fs=fs, band=target_band)
    if not np.isfinite(result).all():
        raise FloatingPointError("carrier envelope 输出包含 NaN/Inf")
    return result


def fixed_resample(
    signal: np.ndarray,
    *,
    input_fs: float,
    spec: ResampleSpec,
    beta: float = 8.6,
    padtype: str = "line",
) -> np.ndarray:
    """使用显式 Kaiser FIR taps 的抗混叠 polyphase 重采样。"""

    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise FloatingPointError("resample 输入必须为非空有限波形")
    if spec.up <= 0 or spec.down <= 0 or spec.numtaps <= 1 or spec.numtaps % 2 == 0:
        raise ValueError(f"resample spec 无效: {spec}")
    nyquist = float(input_fs) * float(spec.up) / 2.0
    if not 0.0 < spec.cutoff_hz < nyquist:
        raise ValueError(f"resample cutoff 超出上采样后 Nyquist: {spec}")
    taps = firwin(
        spec.numtaps,
        spec.cutoff_hz,
        fs=float(input_fs) * float(spec.up),
        window=("kaiser", float(beta)),
        pass_zero="lowpass",
        scale=True,
    )
    output = resample_poly(values, spec.up, spec.down, window=taps, padtype=padtype)
    expected = math.ceil(values.size * spec.up / spec.down)
    if output.size != expected or not np.isfinite(output).all():
        raise FloatingPointError(f"resample 输出长度/finite 异常: expected={expected}, actual={output.size}")
    return np.asarray(output, dtype=np.float64)


def build_carrier_sampling_paths(
    raw_bcg: np.ndarray,
    *,
    carrier_band: BandSpec,
    config: SignalAuditConfig,
) -> dict[str, np.ndarray]:
    """固定实现 100→demod→10、100→20→demod→10、100→10→demod。"""

    target_band = config.band("target")
    eps = float(config.raw["operators"]["log_envelope_eps"])
    resampling = config.raw["operators"]["resampling"]
    beta = float(resampling["beta"])
    padtype = str(resampling["padtype"])
    carrier_100 = fft_band_extract(raw_bcg, fs=100.0, band=carrier_band)

    envelope_100 = carrier_log_envelope(carrier_100, fs=100.0, target_band=target_band, eps=eps)
    path_1 = fixed_resample(
        envelope_100,
        input_fs=100.0,
        spec=config.resample("100_to_10"),
        beta=beta,
        padtype=padtype,
    )

    carrier_20 = fixed_resample(
        carrier_100,
        input_fs=100.0,
        spec=config.resample("100_to_20"),
        beta=beta,
        padtype=padtype,
    )
    envelope_20 = carrier_log_envelope(carrier_20, fs=20.0, target_band=target_band, eps=eps)
    path_2 = fixed_resample(
        envelope_20,
        input_fs=20.0,
        spec=config.resample("20_to_10"),
        beta=beta,
        padtype=padtype,
    )

    carrier_10 = fixed_resample(
        carrier_100,
        input_fs=100.0,
        spec=config.resample("100_to_10"),
        beta=beta,
        padtype=padtype,
    )
    path_3 = carrier_log_envelope(carrier_10, fs=10.0, target_band=target_band, eps=eps)
    paths = {
        "p1_demod_100_then_10": path_1,
        "p2_100_to_20_demod_then_10": path_2,
        "p3_100_to_10_then_demod": path_3,
    }
    if any(values.shape != (1800,) or not np.isfinite(values).all() for values in paths.values()):
        raise RuntimeError("carrier sampling path 输出必须全部为 finite [1800]")
    return paths


def fixed_proxy_combination(proxies: Iterable[np.ndarray], *, dynamic_eps: float) -> np.ndarray:
    """固定 equal-RMS sum；不根据 target 拟合符号、频带或权重。"""

    normalized: list[np.ndarray] = []
    for proxy in proxies:
        values = np.asarray(proxy, dtype=np.float64).reshape(-1)
        centered = values - float(np.mean(values))
        energy = float(np.mean(np.square(centered)))
        if not np.isfinite(energy):
            raise FloatingPointError("proxy combination component energy 非有限")
        if energy <= float(dynamic_eps):
            normalized.append(np.zeros_like(centered))
        else:
            normalized.append(centered / math.sqrt(energy))
    if not normalized or len({values.size for values in normalized}) != 1:
        raise ValueError("proxy combination 需要等长非空 components")
    result = np.sum(np.stack(normalized, axis=0), axis=0) / math.sqrt(float(len(normalized)))
    if not np.isfinite(result).all():
        raise FloatingPointError("proxy combination 输出非有限")
    return result


def metric_config_at_fs(base: TaskMetricConfig, *, fs: float, length: int) -> TaskMetricConfig:
    ratio = float(fs) / float(base.fs)
    if ratio <= 0.0:
        raise ValueError("metric fs ratio 必须为正")
    return replace(
        base,
        fs=float(fs),
        length=int(length),
        max_lag_samples=int(round(base.max_lag_samples * ratio)),
        local_rr_window=int(round(base.local_rr_window * ratio)),
        local_rr_step=int(round(base.local_rr_step * ratio)),
        envelope_window=int(round(base.envelope_window * ratio)),
        envelope_step=int(round(base.envelope_step * ratio)),
        ibi_peak_distance=max(1, int(round(base.ibi_peak_distance * ratio))),
        ibi_match_tolerance=max(1, int(round(base.ibi_match_tolerance * ratio))),
        ndtw_downsample=1,
        ndtw_radius=int(round(0.3 * float(fs))),
    )


def target_timescale_metrics(
    target: np.ndarray,
    *,
    metric_cfg: TaskMetricConfig,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """审计 A：严格复用冻结 Pi/RR/effort 语义，补充预注册的时间尺度描述。"""

    target_band, target_x = canonicalize_numpy(
        np.asarray(target, dtype=np.float64),
        fs=metric_cfg.fs,
        low_hz=metric_cfg.band_low_hz,
        high_hz=metric_cfg.band_high_hz,
        scale_eps=metric_cfg.scale_eps,
    )
    target_band = np.asarray(target_band, dtype=np.float64).reshape(-1)
    target_x = np.asarray(target_x, dtype=np.float64).reshape(-1)
    if target_x.size != metric_cfg.length or not np.isfinite(target_x).all():
        raise FloatingPointError("canonical target 长度或 finite 异常")

    local_values: list[float] = []
    sequence_rows: list[dict[str, Any]] = []
    for start in range(0, metric_cfg.length - metric_cfg.local_rr_window + 1, metric_cfg.local_rr_step):
        stop = start + metric_cfg.local_rr_window
        eligible = centered_energy_numpy(target_band[start:stop]) > metric_cfg.dynamic_eps
        value = (
            _rr_from_power(_periodogram(target_x[start:stop], metric_cfg), metric_cfg.local_rr_window, metric_cfg)
            if eligible
            else np.nan
        )
        if eligible:
            local_values.append(float(value))
        sequence_rows.append(
            {
                "sequence": "local_rr_bpm",
                "point_index": int(len(sequence_rows)),
                "start_sec": float(start / metric_cfg.fs),
                "center_sec": float((start + metric_cfg.local_rr_window / 2.0) / metric_cfg.fs),
                "value": float(value),
                "eligible": bool(eligible),
            }
        )
    local = np.asarray(local_values, dtype=np.float64)
    effort = _log_rms_envelope(target_x, metric_cfg)
    for point_index, value in enumerate(effort):
        start = point_index * metric_cfg.envelope_step
        sequence_rows.append(
            {
                "sequence": "log_effort",
                "point_index": int(point_index),
                "start_sec": float(start / metric_cfg.fs),
                "center_sec": float((start + metric_cfg.envelope_window / 2.0) / metric_cfg.fs),
                "value": float(value),
                "eligible": True,
            }
        )
    effort_diff = np.abs(np.diff(effort))
    effort_centered = effort - float(np.mean(effort))
    effort_variance = float(np.mean(np.square(effort_centered)))
    decay_seconds = np.nan
    decay_censored = False
    effort_autocorr_eligible = effort_variance > metric_cfg.dynamic_eps
    if effort_autocorr_eligible:
        threshold = math.exp(-1.0)
        correlations = np.asarray(
            [
                float(np.dot(effort_centered[:-lag], effort_centered[lag:]))
                / float(np.dot(effort_centered, effort_centered))
                for lag in range(1, effort.size)
            ],
            dtype=np.float64,
        )
        crossings = np.flatnonzero(correlations <= threshold)
        if crossings.size:
            decay_seconds = float((int(crossings[0]) + 1) * metric_cfg.envelope_step / metric_cfg.fs)
        else:
            decay_censored = True

    window = symmetric_hann_numpy(metric_cfg.length)
    spectrum = np.fft.rfft((target_x - float(np.mean(target_x))) * window, norm="backward")
    power = np.square(np.abs(spectrum))
    frequencies = np.fft.rfftfreq(metric_cfg.length, d=1.0 / metric_cfg.fs)
    band_mask = (frequencies >= metric_cfg.band_low_hz) & (frequencies <= metric_cfg.band_high_hz)
    band_power = power[band_mask]
    band_freq = frequencies[band_mask]
    total_power = float(np.sum(band_power))
    low_end = (frequencies >= 0.05) & (frequencies <= 0.10)
    if not np.isfinite(total_power):
        raise FloatingPointError("target 频带功率非有限")
    target_dynamic = centered_energy_numpy(target_band) > metric_cfg.dynamic_eps
    if target_dynamic and total_power > 0.0:
        centroid = float(np.sum(band_freq * band_power) / total_power)
        bandwidth = float(np.sqrt(np.sum(np.square(band_freq - centroid) * band_power) / total_power))
        spectral_peak = float(band_freq[int(np.argmax(band_power))])
        low_end_fraction = float(np.sum(power[low_end]) / total_power)
        whole_rr = float(_whole_rr(target_x, metric_cfg))
    else:
        centroid = np.nan
        bandwidth = np.nan
        spectral_peak = np.nan
        low_end_fraction = np.nan
        whole_rr = np.nan
    modulation = _envelope_modulation(effort, metric_cfg)
    record = {
        "target_waveform_dynamic": bool(target_dynamic),
        "target_whole_rr_bpm": whole_rr,
        "target_local_rr_n": int(local.size),
        "target_local_rr_mean_bpm": float(np.mean(local)) if local.size else np.nan,
        "target_local_rr_range_bpm": float(np.ptp(local)) if local.size else np.nan,
        "target_local_rr_iqr_bpm": float(np.quantile(local, 0.75) - np.quantile(local, 0.25)) if local.size else np.nan,
        "target_local_rr_adjacent_abs_median_bpm": float(np.median(np.abs(np.diff(local)))) if local.size > 1 else np.nan,
        "target_local_rr_adjacent_abs_max_bpm": float(np.max(np.abs(np.diff(local)))) if local.size > 1 else np.nan,
        "target_effort_n": int(effort.size),
        "target_effort_range": float(np.ptp(effort)),
        "target_effort_iqr": float(np.quantile(effort, 0.75) - np.quantile(effort, 0.25)),
        "target_effort_adjacent_abs_median": float(np.median(effort_diff)) if effort_diff.size else np.nan,
        "target_effort_adjacent_abs_max": float(np.max(effort_diff)) if effort_diff.size else np.nan,
        "target_effort_autocorr_eligible": bool(effort_autocorr_eligible),
        "target_effort_autocorr_decay_sec": decay_seconds,
        "target_effort_autocorr_decay_censored": bool(decay_censored),
        "target_spectral_centroid_hz": centroid,
        "target_spectral_peak_hz": spectral_peak,
        "target_spectral_bandwidth_hz": bandwidth,
        "target_low_0p05_0p10_energy_fraction": low_end_fraction,
        "target_envelope_modulation": float(modulation),
        "target_stratum": _envelope_stratum(modulation, metric_cfg),
    }
    if len(sequence_rows) != 44:
        raise RuntimeError(f"target sequence 点数必须为 9+35=44，实际 {len(sequence_rows)}")
    return record, target_band, target_x, sequence_rows


def _absolute_lag_alignment(
    proxy: np.ndarray,
    target: np.ndarray,
    *,
    metric_cfg: TaskMetricConfig,
) -> tuple[float, int, bool]:
    start = metric_cfg.max_lag_samples
    stop = metric_cfg.length - metric_cfg.max_lag_samples
    proxy_common = np.asarray(proxy, dtype=np.float64)
    target_common = np.asarray(target, dtype=np.float64)[start:stop]
    if centered_energy_numpy(proxy_common[start:stop]) <= metric_cfg.dynamic_eps:
        return np.nan, 0, False
    best_abs = -math.inf
    best_corr = np.nan
    best_lag = 0
    for lag in lag_priority(metric_cfg.max_lag_samples):
        left = proxy_common[start + lag : stop + lag]
        a = left - float(np.mean(left))
        b = target_common - float(np.mean(target_common))
        denominator = math.sqrt(float(np.sum(np.square(a))) + metric_cfg.corr_eps) * math.sqrt(
            float(np.sum(np.square(b))) + metric_cfg.corr_eps
        )
        corr = float(np.clip(np.sum(a * b) / denominator, -1.0, 1.0))
        if abs(corr) > best_abs:
            best_abs = abs(corr)
            best_corr = corr
            best_lag = lag
    return float(best_corr), int(best_lag), True


def proxy_observability_metrics(
    proxy: np.ndarray,
    *,
    target_band: np.ndarray,
    target_x: np.ndarray,
    metric_cfg: TaskMetricConfig,
) -> dict[str, Any]:
    """审计 B/C 公共指标；prediction-degenerate 使用冻结 39 bpm/-1 语义，不静默丢弃。"""

    proxy_band, proxy_x = canonicalize_numpy(
        np.asarray(proxy, dtype=np.float64),
        fs=metric_cfg.fs,
        low_hz=metric_cfg.band_low_hz,
        high_hz=metric_cfg.band_high_hz,
        scale_eps=metric_cfg.scale_eps,
    )
    proxy_band = np.asarray(proxy_band, dtype=np.float64).reshape(-1)
    proxy_x = np.asarray(proxy_x, dtype=np.float64).reshape(-1)
    target_band = np.asarray(target_band, dtype=np.float64).reshape(-1)
    target_x = np.asarray(target_x, dtype=np.float64).reshape(-1)
    if not (proxy_x.size == target_x.size == metric_cfg.length):
        raise ValueError("proxy/target 长度与 metric contract 不一致")
    target_eligible = centered_energy_numpy(target_band) > metric_cfg.dynamic_eps
    proxy_dynamic = centered_energy_numpy(proxy_band) > metric_cfg.dynamic_eps
    if target_eligible:
        target_rr = _whole_rr(target_x, metric_cfg)
        whole_error = abs(_whole_rr(proxy_x, metric_cfg) - target_rr) if proxy_dynamic else 39.0
        coherence = _respiratory_coherence(proxy_x, target_x, metric_cfg) if proxy_dynamic else 0.0
    else:
        whole_error = np.nan
        coherence = np.nan
    local_error, local_valid_fraction, local_eligible_n = _local_rr(
        proxy_x,
        target_x,
        target_band,
        metric_cfg,
    )
    envelope = _envelope_metrics(proxy_x, target_x, metric_cfg)
    lag_corr, lag_samples, lag_defined = _absolute_lag_alignment(proxy_x, target_x, metric_cfg=metric_cfg)
    proxy_energy = centered_energy_numpy(proxy_band)
    target_energy = centered_energy_numpy(target_band)
    return {
        "whole_rr_abs_error_bpm": float(whole_error),
        "whole_rr_target_eligible": bool(target_eligible),
        "whole_rr_proxy_dynamic": bool(proxy_dynamic),
        "local_rr_mae_bpm": float(local_error),
        "local_rr_proxy_valid_fraction": float(local_valid_fraction),
        "local_rr_target_eligible_windows": int(local_eligible_n),
        "effort_spearman": float(envelope["target_stratified_envelope_spearman"]),
        "effort_target_eligible": bool(envelope["envelope_spearman_target_eligible"]),
        "effort_proxy_degenerate": bool(envelope["envelope_spearman_prediction_degenerate"]),
        "respiratory_band_coherence": float(coherence),
        "best_abs_lag_corr_signed": float(lag_corr),
        "best_abs_lag_samples": int(lag_samples),
        "best_abs_lag_sec": float(lag_samples / metric_cfg.fs),
        "best_abs_lag_defined": bool(lag_defined),
        "proxy_target_band_energy": float(proxy_energy),
        "target_band_energy": float(target_energy),
        "proxy_to_target_band_energy_ratio": float(proxy_energy / target_energy) if target_energy > 0.0 else np.nan,
    }


def _zero_lag_corr(left: np.ndarray, right: np.ndarray, *, eps: float) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.size != b.size:
        raise ValueError("zero-lag correlation 输入长度不一致")
    a -= float(np.mean(a))
    b -= float(np.mean(b))
    denominator = math.sqrt(float(np.sum(np.square(a))) + eps) * math.sqrt(float(np.sum(np.square(b))) + eps)
    return float(np.clip(np.sum(a * b) / denominator, -1.0, 1.0))


def _normalized_rmse(candidate: np.ndarray, reference: np.ndarray, *, eps: float) -> float:
    candidate_values = np.asarray(candidate, dtype=np.float64).reshape(-1)
    reference_values = np.asarray(reference, dtype=np.float64).reshape(-1)
    if candidate_values.size != reference_values.size:
        raise ValueError("NRMSE 输入长度不一致")
    numerator = math.sqrt(float(np.mean(np.square(candidate_values - reference_values))))
    denominator = math.sqrt(float(np.mean(np.square(reference_values - float(np.mean(reference_values))))))
    return float(numerator / max(denominator, eps))


def boxcar_decimate(signal: np.ndarray, *, factor: int) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    if factor <= 0 or values.size % int(factor) != 0:
        raise ValueError("boxcar decimation 要求正整数 factor 且整除长度")
    output = values.reshape(-1, int(factor)).mean(axis=1)
    if not np.isfinite(output).all():
        raise FloatingPointError("boxcar decimation 输出非有限")
    return output


def lowpass_block_center_decimate(
    signal: np.ndarray,
    *,
    input_fs: float,
    spec: ResampleSpec,
    beta: float = 8.6,
) -> np.ndarray:
    """零相位 FIR 后在与无 padding average-pool 相同的 block center 取样。

    这样 boxcar 与显式低通的差异不会被半个 pooling block 的时间偏移污染。
    """

    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    if spec.up != 1 or spec.down <= 0 or values.size % spec.down != 0:
        raise ValueError("block-center decimation 要求 up=1 且 down 整除输入长度")
    if spec.numtaps <= 1 or spec.numtaps % 2 == 0:
        raise ValueError("block-center FIR 必须是奇数长度")
    taps = firwin(
        spec.numtaps,
        spec.cutoff_hz,
        fs=float(input_fs),
        window=("kaiser", float(beta)),
        pass_zero="lowpass",
        scale=True,
    )
    radius = spec.numtaps // 2
    padded = np.pad(values, (radius, radius), mode="reflect")
    filtered = fftconvolve(padded, taps, mode="valid")
    if filtered.size != values.size:
        raise RuntimeError("block-center FIR 输出长度异常")
    positions = (np.arange(values.size // spec.down, dtype=np.float64) + 0.5) * spec.down - 0.5
    output = np.interp(positions, np.arange(values.size, dtype=np.float64), filtered)
    if not np.isfinite(output).all():
        raise FloatingPointError("block-center decimation 输出非有限")
    return output


def reconstruct_block_center_grid(
    signal: np.ndarray,
    *,
    factor: int,
    target_length: int,
    target_fs: float,
) -> np.ndarray:
    """Fourier resample 后补偿无 padding pooling 的 block-center 时间偏移。"""

    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    if factor <= 0 or target_length != values.size * int(factor):
        raise ValueError("block-center reconstruction 的 factor/length 不闭合")
    upsampled = np.asarray(resample(values, target_length), dtype=np.float64)
    offset_seconds = (float(factor) - 1.0) / (2.0 * float(target_fs))
    spectrum = np.fft.rfft(upsampled, norm="backward")
    frequencies = np.fft.rfftfreq(target_length, d=1.0 / float(target_fs))
    aligned = np.fft.irfft(
        spectrum * np.exp(-2j * np.pi * frequencies * offset_seconds),
        n=target_length,
        norm="backward",
    )
    if not np.isfinite(aligned).all():
        raise FloatingPointError("block-center reconstruction 输出非有限")
    return aligned


def above_nyquist_power_fraction(signal: np.ndarray, *, fs: float, destination_fs: float) -> float:
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    spectrum = np.fft.rfft((values - float(np.mean(values))) * symmetric_hann_numpy(values.size), norm="backward")
    power = np.square(np.abs(spectrum))
    frequencies = np.fft.rfftfreq(values.size, d=1.0 / float(fs))
    total = float(np.sum(power))
    return float(np.sum(power[frequencies > destination_fs / 2.0]) / total) if total > 0.0 else 0.0


def grid_audit_rows(
    source: np.ndarray,
    *,
    source_name: str,
    target_band: np.ndarray,
    target_x: np.ndarray,
    metric_cfg_10: TaskMetricConfig,
    config: SignalAuditConfig,
) -> list[dict[str, Any]]:
    """审计 D：只检查 2/1/0.5-Hz 离散 grid 与 boxcar/显式 LP，不搜索 cutoff。"""

    values = np.asarray(source, dtype=np.float64).reshape(-1)
    if values.shape != (1800,) or not np.isfinite(values).all():
        raise ValueError("grid audit source 必须是 finite 10-Hz [1800]")
    maps = {2.0: "10_to_2", 1.0: "10_to_1", 0.5: "10_to_0p5"}
    factors = {2.0: 5, 1.0: 10, 0.5: 20}
    beta = float(config.raw["operators"]["resampling"]["beta"])
    rows: list[dict[str, Any]] = []
    for grid_hz in (2.0, 1.0, 0.5):
        lowpass = lowpass_block_center_decimate(
            values,
            input_fs=10.0,
            spec=config.resample(maps[grid_hz]),
            beta=beta,
        )
        boxcar = boxcar_decimate(values, factor=factors[grid_hz])
        if lowpass.size != boxcar.size:
            raise RuntimeError("grid lowpass/boxcar 长度不一致")
        comparison_nrmse = _normalized_rmse(boxcar, lowpass, eps=metric_cfg_10.scale_eps)
        above_fraction = above_nyquist_power_fraction(values, fs=10.0, destination_fs=grid_hz)
        for method, downsampled in (("explicit_lowpass_decimation", lowpass), ("boxcar_average_pooling", boxcar)):
            reconstructed = reconstruct_block_center_grid(
                downsampled,
                factor=factors[grid_hz],
                target_length=values.size,
                target_fs=10.0,
            )
            envelope = _envelope_metrics(reconstructed, values, metric_cfg_10)
            full_waveform_eligible = grid_hz >= 2.0
            rr_metrics: dict[str, Any]
            if source_name == "target" and full_waveform_eligible:
                rr_metrics = proxy_observability_metrics(
                    reconstructed,
                    target_band=target_band,
                    target_x=target_x,
                    metric_cfg=metric_cfg_10,
                )
                whole_rr_error = rr_metrics["whole_rr_abs_error_bpm"]
                local_rr_error = rr_metrics["local_rr_mae_bpm"]
                coherence = rr_metrics["respiratory_band_coherence"]
            else:
                whole_rr_error = np.nan
                local_rr_error = np.nan
                coherence = np.nan
            rows.append(
                {
                    "source": source_name,
                    "grid_hz": float(grid_hz),
                    "grid_length": int(downsampled.size),
                    "method": method,
                    "intended_role": "full_waveform_and_context" if full_waveform_eligible else "context_effort_only",
                    "full_waveform_eligible": bool(full_waveform_eligible),
                    "fastest_target_cycle_points": float(grid_hz / metric_cfg_10.band_high_hz),
                    "effort_window_points": float(grid_hz * 10.0),
                    "local_rr_window_points": float(grid_hz * 60.0),
                    "global_window_points": float(grid_hz * 180.0),
                    "source_above_grid_nyquist_power_fraction": float(above_fraction),
                    "boxcar_vs_lowpass_nrmse": float(comparison_nrmse),
                    "roundtrip_nrmse": _normalized_rmse(reconstructed, values, eps=metric_cfg_10.scale_eps),
                    "roundtrip_zero_lag_corr": _zero_lag_corr(reconstructed, values, eps=metric_cfg_10.corr_eps),
                    "context_effort_spearman": float(envelope["target_stratified_envelope_spearman"]),
                    "context_effort_target_eligible": bool(envelope["envelope_spearman_target_eligible"]),
                    "context_effort_degenerate": bool(envelope["envelope_spearman_prediction_degenerate"]),
                    "whole_rr_roundtrip_abs_error_bpm": float(whole_rr_error),
                    "local_rr_roundtrip_mae_bpm": float(local_rr_error),
                    "respiratory_band_roundtrip_coherence": float(coherence),
                }
            )
    return rows


def _validate_source_config(source_cfg: DictConfig, audit_cfg: SignalAuditConfig) -> None:
    signal = audit_cfg.raw["signal"]
    required = {
        "data.format": "research_v2",
        "data.train_split": "train",
        "data.bcg_input_key": signal["input_key"],
        "data.target_key": signal["target_key"],
        "data.filter_unusable": True,
        "data.drop_nonfinite_windows": False,
        "data.max_train_windows": None,
        "window.target_fs": signal["sample_rate_hz"],
        "window.duration_samples": signal["duration_samples"],
        "window.duration_sec": signal["duration_seconds"],
        "loss.band_low_hz": audit_cfg.raw["bands"]["target"]["low_hz"],
        "loss.band_high_hz": audit_cfg.raw["bands"]["target"]["high_hz"],
        "loss.scale_eps": signal["scale_eps"],
        "loss.dynamic_eps": signal["dynamic_eps"],
        "loss.corr_eps": signal["corr_eps"],
        "loss.envelope_eps": signal["envelope_eps"],
        "loss.max_lag_sec": audit_cfg.raw["metrics"]["max_lag_seconds"],
        "loss.envelope_window_sec": audit_cfg.raw["metrics"]["effort_window_seconds"],
        "loss.envelope_step_sec": audit_cfg.raw["metrics"]["effort_step_seconds"],
        "evaluation.local_rr_window_sec": audit_cfg.raw["metrics"]["local_rr_window_seconds"],
        "evaluation.local_rr_step_sec": audit_cfg.raw["metrics"]["local_rr_step_seconds"],
        "evaluation.envelope_strata_low": audit_cfg.raw["metrics"]["envelope_strata_low"],
        "evaluation.envelope_strata_high": audit_cfg.raw["metrics"]["envelope_strata_high"],
        "evaluation.envelope_quantile_method": audit_cfg.raw["metrics"]["envelope_quantile_method"],
    }
    for key, expected in required.items():
        actual = OmegaConf.select(source_cfg, key)
        if isinstance(expected, float):
            matches = float(actual) == float(expected)
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(f"source config 与 RTM-v1 audit contract 不一致: {key} expected={expected!r}, actual={actual!r}")


def _metadata_record(item: Mapping[str, Any]) -> dict[str, Any]:
    meta = item["meta"]
    if str(meta["split"]) != "train":
        raise RuntimeError(f"signal audit 试图读取非 train item: {meta['split']!r}")
    return {
        "dataset_row_id": int(meta["dataset_row_id"]),
        "samp_id": int(meta["samp_id"]),
        "coupling_state_id": int(meta["coupling_state_id"]),
        "split": "train",
    }


def audit_signal_window(
    raw_bcg: np.ndarray,
    raw_target: np.ndarray,
    *,
    identity: Mapping[str, Any],
    source_metric_cfg: TaskMetricConfig,
    audit_cfg: SignalAuditConfig,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """对单个 train window 生成 A/B/C/D 四类确定性审计记录。"""

    x = np.asarray(raw_bcg, dtype=np.float64).reshape(-1)
    y = np.asarray(raw_target, dtype=np.float64).reshape(-1)
    expected = int(audit_cfg.raw["signal"]["duration_samples"])
    if x.shape != (expected,) or y.shape != (expected,):
        raise ValueError(f"signal audit window shape 异常: x={x.shape}, y={y.shape}")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise FloatingPointError(f"admitted train window 包含 NaN/Inf: row={identity['dataset_row_id']}")

    target_record, target_band_100, target_x_100, target_sequences = target_timescale_metrics(
        y,
        metric_cfg=source_metric_cfg,
    )
    target_record = {**identity, **target_record}
    target_stratum = str(target_record["target_stratum"])
    target_sequence_rows = [
        {**identity, "target_stratum": target_stratum, **sequence}
        for sequence in target_sequences
    ]

    displacement_100 = fft_band_extract(x, fs=100.0, band=audit_cfg.band("displacement"))
    carrier_low_100 = fft_band_extract(x, fs=100.0, band=audit_cfg.band("carrier_low"))
    carrier_high_100 = fft_band_extract(x, fs=100.0, band=audit_cfg.band("carrier_high"))
    envelope_low_100 = carrier_log_envelope(
        carrier_low_100,
        fs=100.0,
        target_band=audit_cfg.band("target"),
        eps=float(audit_cfg.raw["operators"]["log_envelope_eps"]),
    )
    envelope_high_100 = carrier_log_envelope(
        carrier_high_100,
        fs=100.0,
        target_band=audit_cfg.band("target"),
        eps=float(audit_cfg.raw["operators"]["log_envelope_eps"]),
    )
    combination = fixed_proxy_combination(
        (displacement_100, envelope_low_100, envelope_high_100),
        dynamic_eps=source_metric_cfg.dynamic_eps,
    )
    combination = fft_band_extract(combination, fs=100.0, band=audit_cfg.band("target"))
    proxies_100 = {
        "displacement": displacement_100,
        "carrier_low_envelope": envelope_low_100,
        "carrier_high_envelope": envelope_high_100,
        "fixed_equal_rms_combination": combination,
    }
    proxy_rows: list[dict[str, Any]] = []
    for name, proxy in proxies_100.items():
        proxy_rows.append(
            {
                **identity,
                "target_stratum": target_stratum,
                "proxy": name,
                **proxy_observability_metrics(
                    proxy,
                    target_band=target_band_100,
                    target_x=target_x_100,
                    metric_cfg=source_metric_cfg,
                ),
            }
        )

    beta = float(audit_cfg.raw["operators"]["resampling"]["beta"])
    padtype = str(audit_cfg.raw["operators"]["resampling"]["padtype"])
    target_10_raw = fixed_resample(
        target_x_100,
        input_fs=100.0,
        spec=audit_cfg.resample("100_to_10"),
        beta=beta,
        padtype=padtype,
    )
    metric_cfg_10 = metric_config_at_fs(source_metric_cfg, fs=10.0, length=1800)
    target_band_10, target_x_10 = canonicalize_numpy(
        target_10_raw,
        fs=10.0,
        low_hz=metric_cfg_10.band_low_hz,
        high_hz=metric_cfg_10.band_high_hz,
        scale_eps=metric_cfg_10.scale_eps,
    )
    target_band_10 = np.asarray(target_band_10, dtype=np.float64).reshape(-1)
    target_x_10 = np.asarray(target_x_10, dtype=np.float64).reshape(-1)

    carrier_paths = {
        "carrier_low": build_carrier_sampling_paths(
            x,
            carrier_band=audit_cfg.band("carrier_low"),
            config=audit_cfg,
        ),
        "carrier_high": build_carrier_sampling_paths(
            x,
            carrier_band=audit_cfg.band("carrier_high"),
            config=audit_cfg,
        ),
    }
    sampling_rows: list[dict[str, Any]] = []
    for band_name, paths in carrier_paths.items():
        reference = paths["p1_demod_100_then_10"]
        reference_energy = centered_energy_numpy(reference)
        for path_name, proxy in paths.items():
            energy = centered_energy_numpy(proxy)
            sampling_rows.append(
                {
                    **identity,
                    "target_stratum": target_stratum,
                    "carrier_band": band_name,
                    "path": path_name,
                    "agreement_with_p1_nrmse": _normalized_rmse(
                        proxy,
                        reference,
                        eps=metric_cfg_10.scale_eps,
                    ),
                    "agreement_with_p1_zero_lag_corr": _zero_lag_corr(
                        proxy,
                        reference,
                        eps=metric_cfg_10.corr_eps,
                    ),
                    "target_band_energy_ratio_to_p1": float(energy / reference_energy)
                    if reference_energy > 0.0
                    else np.nan,
                    **proxy_observability_metrics(
                        proxy,
                        target_band=target_band_10,
                        target_x=target_x_10,
                        metric_cfg=metric_cfg_10,
                    ),
                }
            )

    displacement_target_100 = fft_band_extract(displacement_100, fs=100.0, band=audit_cfg.band("target"))
    displacement_10 = fixed_resample(
        displacement_target_100,
        input_fs=100.0,
        spec=audit_cfg.resample("100_to_10"),
        beta=beta,
        padtype=padtype,
    )
    grid_sources = {
        "target": target_x_10,
        "displacement": displacement_10,
        "carrier_low_envelope": carrier_paths["carrier_low"]["p1_demod_100_then_10"],
        "carrier_high_envelope": carrier_paths["carrier_high"]["p1_demod_100_then_10"],
    }
    grid_rows: list[dict[str, Any]] = []
    for source_name, source in grid_sources.items():
        for row in grid_audit_rows(
            source,
            source_name=source_name,
            target_band=target_band_10,
            target_x=target_x_10,
            metric_cfg_10=metric_cfg_10,
            config=audit_cfg,
        ):
            grid_rows.append({**identity, "target_stratum": target_stratum, **row})
    return target_record, target_sequence_rows, proxy_rows, sampling_rows, grid_rows


_IDENTITY_NUMERIC_COLUMNS = {"dataset_row_id", "samp_id", "coupling_state_id"}


def sample_direct_mean(frame: pd.DataFrame, *, dimensions: list[str]) -> pd.DataFrame:
    """先 window→samp_id，再由 summary 对 samp_id direct mean；同时保留 all/三分层。"""

    required = {"samp_id", "target_stratum", *dimensions}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError(f"sample-direct 输入为空或缺列: {sorted(required - set(frame.columns))}")
    strata = set(frame["target_stratum"].astype(str))
    if not strata.issubset({"low", "medium", "high"}):
        raise ValueError(f"未知 target stratum: {sorted(strata)}")
    all_rows = frame.copy()
    all_rows["target_stratum"] = "all"
    expanded = pd.concat([frame, all_rows], ignore_index=True)
    numeric = [
        column
        for column in expanded.columns
        if column not in _IDENTITY_NUMERIC_COLUMNS and pd.api.types.is_numeric_dtype(expanded[column])
    ]
    keys = ["samp_id", "target_stratum", *dimensions]
    grouped = expanded.groupby(keys, dropna=False, sort=True)
    means = grouped[numeric].mean().reset_index()
    means["n_windows"] = grouped.size().to_numpy(dtype=np.int64)
    return means


def build_sample_direct_summary(
    tables: Mapping[str, tuple[pd.DataFrame, list[str]]],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    dimension_union = ["target_stratum", "proxy", "carrier_band", "path", "source", "grid_hz", "method"]
    for table_name, (frame, dimensions) in tables.items():
        group_columns = ["target_stratum", *dimensions]
        excluded = {"samp_id", "n_windows"}
        metric_columns = [
            column
            for column in frame.columns
            if column not in excluded and column not in group_columns and pd.api.types.is_numeric_dtype(frame[column])
        ]
        for keys, group in frame.groupby(group_columns, dropna=False, sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            dimensions_record = dict(zip(group_columns, keys, strict=True))
            for metric in metric_columns:
                values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=np.float64)
                finite = values[np.isfinite(values)]
                record: dict[str, Any] = {
                    "table": table_name,
                    **{name: dimensions_record.get(name, "") for name in dimension_union},
                    "metric": metric,
                    "n_samp_ids_total": int(group["samp_id"].nunique()),
                    "n_samp_ids_finite": int(finite.size),
                    "mean": float(np.mean(finite)) if finite.size else np.nan,
                    "sample_sd": float(np.std(finite, ddof=1)) if finite.size > 1 else np.nan,
                    "median": float(np.median(finite)) if finite.size else np.nan,
                    "q10": float(np.quantile(finite, 0.10)) if finite.size else np.nan,
                    "q90": float(np.quantile(finite, 0.90)) if finite.size else np.nan,
                }
                records.append(record)
    return pd.DataFrame.from_records(records)


def finite_frame_audit(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        raise ValueError("输出 frame 不能为空")
    columns: dict[str, Any] = {}
    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
            infinite = int(np.isinf(values).sum())
            if infinite:
                raise FloatingPointError(f"输出列 {column} 包含 {infinite} 个 Inf")
            columns[column] = {
                "n_total": int(values.size),
                "n_finite": int(np.isfinite(values).sum()),
                "n_null": int(np.isnan(values).sum()),
                "n_infinite": infinite,
            }
    return {"n_rows": int(len(frame)), "numeric_columns": columns}


def _exclusion_audit(train_adapted: pd.DataFrame) -> dict[str, int]:
    usable = train_adapted["usable"].astype(bool)
    allowed = train_adapted["allowed_losses"].fillna("").astype(str)
    reasons = train_adapted.get("reason", pd.Series([""] * len(train_adapted))).fillna("").astype(str)
    return {
        "n_train_index_rows": int(len(train_adapted)),
        "n_eligible": int(usable.sum()),
        "n_excluded_unique": int((~usable).sum()),
        "excluded_upstream_reason_nonempty": int((~reasons.eq("")).sum()),
        "excluded_missing_waveform_loss": int((~allowed.str.split(";").apply(lambda values: "waveform" in set(values))).sum()),
        "excluded_hard_valid_ratio_below_threshold": int(
            (train_adapted["hard_valid_ratio"].astype(float) < 0.80).sum()
        ),
        "excluded_alignment_valid_ratio_below_threshold": int(
            (train_adapted["state_alignment_valid_ratio"].astype(float) < 0.80).sum()
        ),
    }


def _git_identity() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if commit.returncode != 0 or status.returncode != 0:
        raise RuntimeError("无法读取 Git commit/dirty state")
    return commit.stdout.strip(), bool(status.stdout.strip())


def _dependency_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("numpy", "pandas", "scipy", "torch", "omegaconf", "tqdm"):
        result[package] = importlib.metadata.version(package)
    return result


def _json_write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _artifact_record(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    return {
        "filename": path.name,
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
        "rows": None if rows is None else int(rows),
    }


def validate_access_receipt(receipt: Mapping[str, Any]) -> None:
    """严格 receipt schema；用于写前自检与用户回传后的 fail-closed 解析。"""

    _require_exact_keys(
        receipt,
        {
            "schema_version",
            "protocol_id",
            "created_utc",
            "execution",
            "data",
            "access",
            "counts",
            "finite_audit",
            "artifacts",
            "manifest_sha256",
        },
        "receipt",
    )
    if receipt["schema_version"] != RECEIPT_SCHEMA_VERSION or receipt["protocol_id"] != PROTOCOL_ID:
        raise ValueError("receipt schema/protocol ID 不匹配")
    execution = receipt["execution"]
    _require_exact_keys(
        execution,
        {
            "command",
            "cwd",
            "git_commit",
            "git_dirty",
            "python_version",
            "platform",
            "dependencies",
            "audit_config_path",
            "audit_config_sha256",
            "source_config_path",
            "source_config_sha256",
        },
        "receipt.execution",
    )
    if bool(execution["git_dirty"]):
        raise ValueError("正式 signal-audit receipt 不允许 git_dirty=true")
    data = receipt["data"]
    _require_exact_keys(
        data,
        {
            "dataset_root",
            "shared_index_path",
            "shared_index_sha256",
            "selected_dataset_row_ids_sha256",
            "selected_input_content_sha256",
            "selected_target_content_sha256",
            "split",
            "input_key",
            "target_key",
        },
        "receipt.data",
    )
    if data["split"] != "train":
        raise ValueError("receipt.data.split 必须为 train")
    access = receipt["access"]
    _require_exact_keys(
        access,
        {
            "shared_index_metadata_read",
            "signal_splits_accessed",
            "validation_accessed",
            "validation_target_accessed",
            "validation_prediction_accessed",
            "research_test_accessed",
            "checkpoint_accessed",
            "model_training_used",
            "model_inference_used",
            "gpu_used",
        },
        "receipt.access",
    )
    if access["signal_splits_accessed"] != ["train"]:
        raise ValueError("receipt 只能登记 train signal access")
    forbidden_true = {
        "validation_accessed",
        "validation_target_accessed",
        "validation_prediction_accessed",
        "research_test_accessed",
        "checkpoint_accessed",
        "model_training_used",
        "model_inference_used",
        "gpu_used",
    }
    if any(bool(access[key]) for key in forbidden_true):
        raise ValueError("receipt access flags 显示越界访问")
    _require_exact_keys(
        receipt["counts"],
        {
            "expected_windows",
            "actual_windows",
            "expected_samp_ids",
            "actual_samp_ids",
            "exclusions",
            "output_rows",
        },
        "receipt.counts",
    )
    if int(receipt["counts"]["actual_windows"]) != int(receipt["counts"]["expected_windows"]):
        raise ValueError("receipt train window count 不完整")
    if int(receipt["counts"]["actual_samp_ids"]) != int(receipt["counts"]["expected_samp_ids"]):
        raise ValueError("receipt train samp_id count 不完整")
    if not isinstance(receipt["finite_audit"], Mapping) or not receipt["finite_audit"]:
        raise ValueError("receipt finite_audit 不能为空")
    for filename, audit in receipt["finite_audit"].items():
        _require_exact_keys(audit, {"n_rows", "numeric_columns"}, f"receipt.finite_audit.{filename}")
        for column, counts in audit["numeric_columns"].items():
            _require_exact_keys(
                counts,
                {"n_total", "n_finite", "n_null", "n_infinite"},
                f"receipt.finite_audit.{filename}.{column}",
            )
            if int(counts["n_infinite"]) != 0 or int(counts["n_finite"]) + int(counts["n_null"]) != int(counts["n_total"]):
                raise ValueError(f"receipt finite counts 不闭合: {filename}:{column}")
    if not isinstance(receipt["artifacts"], list) or not receipt["artifacts"]:
        raise ValueError("receipt artifacts 不能为空")
    for index, artifact in enumerate(receipt["artifacts"]):
        _require_exact_keys(artifact, {"filename", "sha256", "size_bytes", "rows"}, f"receipt.artifacts[{index}]")


def _write_output_bundle(
    *,
    audit_cfg: SignalAuditConfig,
    source_cfg: DictConfig,
    source_config_sha256: str,
    index_path: Path,
    index_sha256: str,
    row_ids_sha256: str,
    input_content_sha256: str,
    target_content_sha256: str,
    command: str,
    git_commit: str,
    exclusions: dict[str, int],
    frames: Mapping[str, pd.DataFrame],
) -> Path:
    output_dir = audit_cfg.output_dir
    if output_dir.exists():
        raise FileExistsError(f"RTM-v1 signal audit 输出禁止覆盖: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.parent / f".{output_dir.name}.{uuid4().hex}.tmp"
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        resolved = {
            "protocol_id": PROTOCOL_ID,
            "audit_config": audit_cfg.raw,
            "audit_config_sha256": audit_cfg.sha256,
            "source_config": OmegaConf.to_container(source_cfg, resolve=True),
            "source_config_sha256": source_config_sha256,
            "runtime_only_overrides": {"data.preload_windows": False},
        }
        _json_write(temporary / RESOLVED_CONFIG_FILENAME, resolved)
        float_format = str(audit_cfg.raw["output"]["csv_float_format"])
        for filename, frame in frames.items():
            frame.to_csv(temporary / filename, index=False, float_format=float_format)

        artifacts = [_artifact_record(temporary / RESOLVED_CONFIG_FILENAME)]
        artifacts.extend(_artifact_record(temporary / filename, rows=len(frame)) for filename, frame in frames.items())
        manifest = {
            "schema_version": "rtm-v1-signal-audit-manifest-v1",
            "protocol_id": PROTOCOL_ID,
            "artifacts": artifacts,
        }
        _json_write(temporary / MANIFEST_FILENAME, manifest)
        manifest_record = _artifact_record(temporary / MANIFEST_FILENAME)
        artifacts_with_manifest = [*artifacts, manifest_record]

        actual_windows = int(frames[TARGET_WINDOW_FILENAME]["dataset_row_id"].nunique())
        actual_samp_ids = int(frames[TARGET_WINDOW_FILENAME]["samp_id"].nunique())
        finite_audit = {filename: finite_frame_audit(frame) for filename, frame in frames.items()}
        receipt = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "execution": {
                "command": command,
                "cwd": str(REPO_ROOT),
                "git_commit": git_commit,
                "git_dirty": False,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "dependencies": _dependency_versions(),
                "audit_config_path": str(audit_cfg.path),
                "audit_config_sha256": audit_cfg.sha256,
                "source_config_path": str(audit_cfg.source_config_path),
                "source_config_sha256": source_config_sha256,
            },
            "data": {
                "dataset_root": str(source_cfg.data.dataset_root),
                "shared_index_path": str(index_path),
                "shared_index_sha256": index_sha256,
                "selected_dataset_row_ids_sha256": row_ids_sha256,
                "selected_input_content_sha256": input_content_sha256,
                "selected_target_content_sha256": target_content_sha256,
                "split": "train",
                "input_key": str(source_cfg.data.bcg_input_key),
                "target_key": str(source_cfg.data.target_key),
            },
            "access": {
                "shared_index_metadata_read": True,
                "signal_splits_accessed": ["train"],
                "validation_accessed": False,
                "validation_target_accessed": False,
                "validation_prediction_accessed": False,
                "research_test_accessed": False,
                "checkpoint_accessed": False,
                "model_training_used": False,
                "model_inference_used": False,
                "gpu_used": False,
            },
            "counts": {
                "expected_windows": int(audit_cfg.raw["access"]["expected_windows"]),
                "actual_windows": actual_windows,
                "expected_samp_ids": int(audit_cfg.raw["access"]["expected_samp_ids"]),
                "actual_samp_ids": actual_samp_ids,
                "exclusions": exclusions,
                "output_rows": {filename: int(len(frame)) for filename, frame in frames.items()},
            },
            "finite_audit": finite_audit,
            "artifacts": artifacts_with_manifest,
            "manifest_sha256": manifest_record["sha256"],
        }
        validate_access_receipt(receipt)
        _json_write(temporary / RECEIPT_FILENAME, receipt)
        receipt_hash = sha256_file(temporary / RECEIPT_FILENAME)
        (temporary / RECEIPT_HASH_FILENAME).write_text(
            f"{receipt_hash}  {RECEIPT_FILENAME}\n",
            encoding="utf-8",
        )
        if output_dir.exists():
            raise FileExistsError(f"RTM-v1 signal audit 输出在写入期间出现，拒绝覆盖: {output_dir}")
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output_dir / RECEIPT_FILENAME


def run_train_signal_audit(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    command: str,
) -> Path:
    """只读完整 eligible train split；调用者必须显式运行，本函数不被训练入口调用。"""

    audit_cfg = load_signal_audit_config(config_path)
    if audit_cfg.output_dir.exists():
        raise FileExistsError(f"RTM-v1 signal audit 输出禁止覆盖: {audit_cfg.output_dir}")
    git_commit, git_dirty = _git_identity()
    if git_dirty:
        raise RuntimeError("RTM-v1 正式 train-only signal audit 要求干净 Git 工作树")

    source_config_sha256 = sha256_file(audit_cfg.source_config_path)
    expected_source_config_sha256 = str(audit_cfg.raw["source_config_sha256"])
    if source_config_sha256 != expected_source_config_sha256:
        raise RuntimeError(
            "RTM-v1 source config SHA-256 不匹配；数据/指标口径变化必须先修订协议: "
            f"expected={expected_source_config_sha256}, actual={source_config_sha256}"
        )
    source_cfg = OmegaConf.load(audit_cfg.source_config_path)
    _validate_source_config(source_cfg, audit_cfg)
    source_cfg = OmegaConf.create(OmegaConf.to_container(source_cfg, resolve=True))
    source_cfg.data.preload_windows = False
    index_path = (Path(str(source_cfg.data.dataset_root)) / str(source_cfg.data.index_csv)).resolve()
    if not index_path.is_file():
        raise FileNotFoundError(f"research-v2 shared index 不存在: {index_path}")
    index_sha256 = sha256_file(index_path)

    # shared index 是多 split 元数据容器；先按 split 过滤，只有 train 行可进入适配器和 NPZ reader。
    raw_index = pd.read_csv(index_path)
    if "split" not in raw_index.columns:
        raise ValueError("shared index 缺少 split 列")
    train_raw = raw_index.loc[raw_index["split"].astype(str).eq("train")].copy()
    del raw_index
    train_adapted = adapt_research_v2_index(train_raw, source_cfg)
    exclusions = _exclusion_audit(train_adapted)
    rows = filter_index(
        train_adapted,
        source_cfg,
        split="train",
        max_windows=None,
        sample_strategy=str(source_cfg.data.train_sample_strategy),
        sample_seed=int(source_cfg.data.train_sample_seed),
    )
    expected_windows = int(audit_cfg.raw["access"]["expected_windows"])
    expected_samp_ids = int(audit_cfg.raw["access"]["expected_samp_ids"])
    if len(rows) != expected_windows or int(rows["samp_id"].nunique()) != expected_samp_ids:
        raise RuntimeError(
            "完整 eligible train identity 不匹配: "
            f"windows={len(rows)}/{expected_windows}, samp_ids={rows['samp_id'].nunique()}/{expected_samp_ids}"
        )
    row_ids = rows["dataset_row_id"].to_numpy(dtype=np.int64, copy=True)
    row_ids_sha256 = sha256(row_ids.tobytes()).hexdigest()
    expected_row_hash = str(audit_cfg.raw["access"]["expected_dataset_row_ids_sha256"])
    if row_ids_sha256 != expected_row_hash:
        raise RuntimeError(
            f"eligible train dataset_row_id SHA-256 不匹配: expected={expected_row_hash}, actual={row_ids_sha256}"
        )
    if not rows["split"].astype(str).eq("train").all():
        raise RuntimeError("train-only rows 中混入非 train split")

    dataset = ResearchV2WindowDataset(index_path, rows, source_cfg, preload_windows=False)
    if len(dataset) != expected_windows or not dataset.rows["split"].astype(str).eq("train").all():
        raise RuntimeError("train-only dataset identity 在 reader 内发生变化")
    source_metric_cfg = TaskMetricConfig.from_config(source_cfg)
    target_records: list[dict[str, Any]] = []
    target_sequence_records: list[dict[str, Any]] = []
    proxy_records: list[dict[str, Any]] = []
    sampling_records: list[dict[str, Any]] = []
    grid_records: list[dict[str, Any]] = []
    input_hasher = sha256()
    target_hasher = sha256()

    from tqdm import tqdm

    for index in tqdm(range(len(dataset)), desc="RTM-v1 train-only signal audit", unit="window"):
        item = dataset[index]
        identity = _metadata_record(item)
        x_float32 = np.asarray(item["x"].numpy(), dtype="<f4").reshape(-1)
        y_float32 = np.asarray(item["target"].numpy(), dtype="<f4").reshape(-1)
        row_bytes = np.asarray([identity["dataset_row_id"]], dtype="<i8").tobytes()
        input_hasher.update(row_bytes)
        input_hasher.update(x_float32.tobytes(order="C"))
        target_hasher.update(row_bytes)
        target_hasher.update(y_float32.tobytes(order="C"))
        target_row, target_sequence_rows, proxy_rows, sampling_rows, grid_rows = audit_signal_window(
            x_float32,
            y_float32,
            identity=identity,
            source_metric_cfg=source_metric_cfg,
            audit_cfg=audit_cfg,
        )
        target_records.append(target_row)
        target_sequence_records.extend(target_sequence_rows)
        proxy_records.extend(proxy_rows)
        sampling_records.extend(sampling_rows)
        grid_records.extend(grid_rows)

    target_window = pd.DataFrame.from_records(target_records)
    target_sequence = pd.DataFrame.from_records(target_sequence_records)
    proxy_window = pd.DataFrame.from_records(proxy_records)
    sampling_window = pd.DataFrame.from_records(sampling_records)
    grid_window = pd.DataFrame.from_records(grid_records)
    expected_row_counts = {
        TARGET_WINDOW_FILENAME: expected_windows,
        TARGET_SEQUENCE_FILENAME: expected_windows * 44,
        PROXY_WINDOW_FILENAME: expected_windows * 4,
        SAMPLING_WINDOW_FILENAME: expected_windows * 2 * 3,
        GRID_WINDOW_FILENAME: expected_windows * 4 * 3 * 2,
    }
    actual_window_frames = {
        TARGET_WINDOW_FILENAME: target_window,
        TARGET_SEQUENCE_FILENAME: target_sequence,
        PROXY_WINDOW_FILENAME: proxy_window,
        SAMPLING_WINDOW_FILENAME: sampling_window,
        GRID_WINDOW_FILENAME: grid_window,
    }
    for filename, expected_count in expected_row_counts.items():
        frame = actual_window_frames[filename]
        if len(frame) != expected_count or not frame["split"].astype(str).eq("train").all():
            raise RuntimeError(f"{filename} 行数/split 不闭合: expected={expected_count}, actual={len(frame)}")

    target_sample = sample_direct_mean(target_window, dimensions=[])
    proxy_sample = sample_direct_mean(proxy_window, dimensions=["proxy"])
    sampling_sample = sample_direct_mean(sampling_window, dimensions=["carrier_band", "path"])
    grid_sample = sample_direct_mean(grid_window, dimensions=["source", "grid_hz", "method"])
    summary = build_sample_direct_summary(
        {
            "target_timescale": (target_sample, []),
            "proxy_observability": (proxy_sample, ["proxy"]),
            "sampling_order": (sampling_sample, ["carrier_band", "path"]),
            "multiscale_grid": (grid_sample, ["source", "grid_hz", "method"]),
        }
    )
    frames = {
        TARGET_WINDOW_FILENAME: target_window,
        TARGET_SEQUENCE_FILENAME: target_sequence,
        TARGET_SAMPLE_FILENAME: target_sample,
        PROXY_WINDOW_FILENAME: proxy_window,
        PROXY_SAMPLE_FILENAME: proxy_sample,
        SAMPLING_WINDOW_FILENAME: sampling_window,
        SAMPLING_SAMPLE_FILENAME: sampling_sample,
        GRID_WINDOW_FILENAME: grid_window,
        GRID_SAMPLE_FILENAME: grid_sample,
        SUMMARY_FILENAME: summary,
    }
    for frame in frames.values():
        finite_frame_audit(frame)
    return _write_output_bundle(
        audit_cfg=audit_cfg,
        source_cfg=source_cfg,
        source_config_sha256=source_config_sha256,
        index_path=index_path,
        index_sha256=index_sha256,
        row_ids_sha256=row_ids_sha256,
        input_content_sha256=input_hasher.hexdigest(),
        target_content_sha256=target_hasher.hexdigest(),
        command=command,
        git_commit=git_commit,
        exclusions=exclusions,
        frames=frames,
    )
