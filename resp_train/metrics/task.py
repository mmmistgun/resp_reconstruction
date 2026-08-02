from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.signal import find_peaks
from scipy.stats import rankdata

from resp_train.protocols.respiration import (
    as_batch_waveform_numpy,
    canonicalize_numpy,
    centered_energy_numpy,
    lag_priority,
    symmetric_hann_numpy,
)

_EVALUATION_CHUNK_SIZE = 64


@dataclass(frozen=True)
class TaskMetricConfig:
    fs: float
    length: int
    band_low_hz: float
    band_high_hz: float
    scale_eps: float
    dynamic_eps: float
    corr_eps: float
    envelope_eps: float
    max_lag_samples: int
    local_rr_window: int
    local_rr_step: int
    envelope_window: int
    envelope_step: int
    ibi_peak_distance: int
    ibi_match_tolerance: int
    ibi_coverage_threshold: float
    ndtw_downsample: int
    ndtw_radius: int

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "TaskMetricConfig":
        loss = cfg.loss
        evaluation = cfg.evaluation
        fs = float(cfg.window.target_fs)
        return cls(
            fs=fs,
            length=int(cfg.window.duration_samples),
            band_low_hz=float(loss.band_low_hz),
            band_high_hz=float(loss.band_high_hz),
            scale_eps=float(loss.scale_eps),
            dynamic_eps=float(loss.dynamic_eps),
            corr_eps=float(loss.corr_eps),
            envelope_eps=float(loss.envelope_eps),
            max_lag_samples=int(round(float(loss.max_lag_sec) * fs)),
            local_rr_window=int(round(float(evaluation.local_rr_window_sec) * fs)),
            local_rr_step=int(round(float(evaluation.local_rr_step_sec) * fs)),
            envelope_window=int(round(float(loss.envelope_window_sec) * fs)),
            envelope_step=int(round(float(loss.envelope_step_sec) * fs)),
            ibi_peak_distance=int(evaluation.ibi_peak_distance_samples),
            ibi_match_tolerance=int(round(float(evaluation.ibi_match_tolerance_sec) * fs)),
            ibi_coverage_threshold=float(evaluation.ibi_coverage_threshold),
            ndtw_downsample=int(round(fs / float(evaluation.ndtw_fs))),
            ndtw_radius=int(round(float(evaluation.ndtw_radius_sec) * float(evaluation.ndtw_fs))),
        )


def evaluate_task_predictions(
    predictions: dict[str, np.ndarray],
    cfg: DictConfig,
    *,
    include_test_only: bool = False,
    method: str | None = None,
) -> pd.DataFrame:
    """按冻结协议生成逐 180 秒 sample 指标。"""

    protocol = TaskMetricConfig.from_config(cfg)
    pred = as_batch_waveform_numpy(predictions["r_tho_hat"])
    target = as_batch_waveform_numpy(predictions["tho_ref"])
    if pred.shape != target.shape:
        raise ValueError(f"prediction/target shape 不一致: {pred.shape} vs {target.shape}")
    if pred.shape[1] != protocol.length:
        raise ValueError(f"期望 {protocol.length} 点，实际 {pred.shape[1]}")
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise FloatingPointError("prediction/target 包含 NaN/Inf")

    records: list[dict[str, Any]] = []
    for chunk_start in range(0, pred.shape[0], _EVALUATION_CHUNK_SIZE):
        chunk_stop = min(pred.shape[0], chunk_start + _EVALUATION_CHUNK_SIZE)
        pred_band, pred_x = canonicalize_numpy(
            pred[chunk_start:chunk_stop],
            fs=protocol.fs,
            low_hz=protocol.band_low_hz,
            high_hz=protocol.band_high_hz,
            scale_eps=protocol.scale_eps,
        )
        target_band, target_x = canonicalize_numpy(
            target[chunk_start:chunk_stop],
            fs=protocol.fs,
            low_hz=protocol.band_low_hz,
            high_hz=protocol.band_high_hz,
            scale_eps=protocol.scale_eps,
        )
        for offset in range(chunk_stop - chunk_start):
            index = chunk_start + offset
            record = _metadata_record(predictions, index=index, method=method)
            record.update(
                _evaluate_sample(
                    pred_x[offset],
                    target_x[offset],
                    pred_band[offset],
                    target_band[offset],
                    protocol,
                    include_test_only=include_test_only,
                )
            )
            records.append(record)
    return pd.DataFrame.from_records(records)


def validation_local_rr_mean(predictions: dict[str, np.ndarray], cfg: DictConfig) -> float:
    """每 epoch 唯一 selector；只计算 Local RR，避免生成完整指标集合。"""

    protocol = TaskMetricConfig.from_config(cfg)
    pred = as_batch_waveform_numpy(predictions["r_tho_hat"])
    target = as_batch_waveform_numpy(predictions["tho_ref"])
    if pred.shape != target.shape or pred.shape[1] != protocol.length:
        raise ValueError("Local RR selector 的 prediction/target shape 异常")
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise FloatingPointError("validation prediction/target 包含 NaN/Inf")
    values: list[float] = []
    for chunk_start in range(0, pred.shape[0], _EVALUATION_CHUNK_SIZE):
        chunk_stop = min(pred.shape[0], chunk_start + _EVALUATION_CHUNK_SIZE)
        _, pred_x = canonicalize_numpy(
            pred[chunk_start:chunk_stop],
            fs=protocol.fs,
            low_hz=protocol.band_low_hz,
            high_hz=protocol.band_high_hz,
            scale_eps=protocol.scale_eps,
        )
        target_band, target_x = canonicalize_numpy(
            target[chunk_start:chunk_stop],
            fs=protocol.fs,
            low_hz=protocol.band_low_hz,
            high_hz=protocol.band_high_hz,
            scale_eps=protocol.scale_eps,
        )
        values.extend(
            _local_rr(pred_x[offset], target_x[offset], target_band[offset], protocol)[0]
            for offset in range(chunk_stop - chunk_start)
        )
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("完整 validation 没有 target-eligible Local RR sample")
    return float(np.mean(finite))


def summarize_task_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """逐 sample direct mean；不做 subject balancing 或比例 pooling。"""

    columns = [
        "whole_rr_abs_error_bpm",
        "local_rr_mae_bpm",
        "local_rr_prediction_valid_fraction",
        "global_effort_spearman",
        "local_effort_spearman",
        "lag_aware_signed_pcc",
        "ibi_medae_sec",
        "ibi_coverage",
        "respiratory_band_coherence",
        "constrained_ndtw",
    ]
    row: dict[str, float | int] = {"n_samples": int(len(metrics))}
    for column in columns:
        if column not in metrics:
            continue
        values = pd.to_numeric(metrics[column], errors="coerce")
        row[f"{column}_mean"] = float(values.mean())
        row[f"{column}_n"] = int(values.notna().sum())
    if {"ibi_interpretable", "ibi_target_eligible"}.issubset(metrics.columns):
        ibi_eligible = metrics["ibi_target_eligible"].astype(bool).to_numpy()
        interpretable = metrics.loc[ibi_eligible, "ibi_interpretable"].astype(bool).to_numpy()
        row["ibi_interpretable_fraction"] = float(np.mean(interpretable)) if interpretable.size else np.nan
        row["ibi_interpretable_n"] = int(interpretable.size)
    if {"best_lag_sec", "joint_target_eligible", "joint_prediction_degenerate"}.issubset(metrics.columns):
        joint_eligible = metrics["joint_target_eligible"].astype(bool).to_numpy()
        prediction_degenerate = metrics["joint_prediction_degenerate"].astype(bool).to_numpy()
        lag_defined = joint_eligible & ~prediction_degenerate
        lag = np.abs(
            pd.to_numeric(metrics.loc[lag_defined, "best_lag_sec"], errors="coerce").to_numpy(dtype=np.float64)
        )
        lag = lag[np.isfinite(lag)]
        row["best_abs_lag_median_sec"] = float(np.median(lag)) if lag.size else np.nan
        row["best_abs_lag_p95_sec"] = float(np.quantile(lag, 0.95)) if lag.size else np.nan
        row["best_lag_boundary_fraction"] = (
            float(np.mean(np.isclose(lag, 0.30, atol=1e-12))) if lag.size else np.nan
        )
        row["joint_target_eligible_fraction"] = float(np.mean(joint_eligible)) if joint_eligible.size else np.nan
        row["joint_prediction_degenerate_fraction"] = (
            float(np.mean(prediction_degenerate[joint_eligible])) if np.any(joint_eligible) else np.nan
        )
    return pd.DataFrame([row])


def _evaluate_sample(
    pred_x: np.ndarray,
    target_x: np.ndarray,
    pred_band: np.ndarray,
    target_band: np.ndarray,
    cfg: TaskMetricConfig,
    *,
    include_test_only: bool,
) -> dict[str, Any]:
    del pred_band
    whole_eligible = centered_energy_numpy(target_band) > cfg.dynamic_eps
    if whole_eligible:
        target_rr = _whole_rr(target_x, cfg)
        pred_rr = _whole_rr(pred_x, cfg) if centered_energy_numpy(pred_x) > cfg.dynamic_eps else None
        whole_error = abs(float(pred_rr) - float(target_rr)) if pred_rr is not None else 39.0
    else:
        whole_error = np.nan

    local_rr, local_valid_fraction, local_eligible_windows = _local_rr(pred_x, target_x, target_band, cfg)
    global_effort, global_effort_eligible = _global_effort(pred_x, target_x, target_band, cfg)
    local_effort, local_effort_eligible = _local_effort(pred_x, target_x, target_band, cfg)
    joint_pcc, best_lag, joint_eligible, prediction_dynamic = _lag_aware_pcc(pred_x, target_x, target_band, cfg)
    ibi_medae, ibi_coverage, ibi_interpretable, ibi_eligible = _ibi_metrics(
        pred_x,
        target_x,
        target_band,
        best_lag,
        joint_eligible,
        cfg,
    )

    result: dict[str, Any] = {
        "whole_rr_abs_error_bpm": whole_error,
        "whole_rr_target_eligible": bool(whole_eligible),
        "local_rr_mae_bpm": local_rr,
        "local_rr_prediction_valid_fraction": local_valid_fraction,
        "local_rr_target_eligible": bool(local_eligible_windows > 0),
        "local_rr_target_eligible_windows": int(local_eligible_windows),
        "global_effort_spearman": global_effort,
        "global_effort_target_eligible": bool(global_effort_eligible),
        "local_effort_spearman": local_effort,
        "local_effort_target_eligible": bool(local_effort_eligible),
        "lag_aware_signed_pcc": joint_pcc,
        "best_lag_samples": int(best_lag),
        "best_lag_sec": float(best_lag / cfg.fs),
        "joint_target_eligible": bool(joint_eligible),
        "joint_prediction_degenerate": bool(joint_eligible and not prediction_dynamic),
        "ibi_medae_sec": ibi_medae,
        "ibi_coverage": ibi_coverage,
        "ibi_interpretable": bool(ibi_interpretable),
        "ibi_target_eligible": bool(ibi_eligible),
    }
    if include_test_only:
        if joint_eligible:
            result["respiratory_band_coherence"] = _respiratory_coherence(pred_x, target_x, cfg)
            result["constrained_ndtw"] = _constrained_ndtw(pred_x, target_x, cfg)
        else:
            result["respiratory_band_coherence"] = np.nan
            result["constrained_ndtw"] = np.nan
    return result


def _whole_rr(signal: np.ndarray, cfg: TaskMetricConfig) -> float:
    segment = int(round(60.0 * cfg.fs))
    step = segment // 2
    powers = [_periodogram(signal[start : start + segment], cfg) for start in range(0, cfg.length - segment + 1, step)]
    return _rr_from_power(np.median(np.stack(powers, axis=0), axis=0), segment, cfg)


def _local_rr(
    pred: np.ndarray,
    target: np.ndarray,
    target_band: np.ndarray,
    cfg: TaskMetricConfig,
) -> tuple[float, float, int]:
    errors: list[float] = []
    valid_predictions = 0
    for start in range(0, cfg.length - cfg.local_rr_window + 1, cfg.local_rr_step):
        stop = start + cfg.local_rr_window
        if centered_energy_numpy(target_band[start:stop]) <= cfg.dynamic_eps:
            continue
        target_rr = _rr_from_power(_periodogram(target[start:stop], cfg), cfg.local_rr_window, cfg)
        if centered_energy_numpy(pred[start:stop]) > cfg.dynamic_eps:
            pred_rr = _rr_from_power(_periodogram(pred[start:stop], cfg), cfg.local_rr_window, cfg)
            if np.isfinite(pred_rr):
                errors.append(abs(pred_rr - target_rr))
                valid_predictions += 1
                continue
        errors.append(39.0)
    if not errors:
        return np.nan, np.nan, 0
    return float(np.mean(errors)), float(valid_predictions / len(errors)), len(errors)


def _periodogram(signal: np.ndarray, cfg: TaskMetricConfig) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64)
    centered = values - np.mean(values)
    spectrum = np.fft.rfft(centered * symmetric_hann_numpy(values.size), n=values.size, norm="backward")
    return np.square(np.abs(spectrum))


def _rr_from_power(power: np.ndarray, n_fft: int, cfg: TaskMetricConfig) -> float:
    frequencies = np.fft.rfftfreq(int(n_fft), d=1.0 / cfg.fs)
    indices = np.flatnonzero((frequencies >= cfg.band_low_hz) & (frequencies <= cfg.band_high_hz))
    if indices.size == 0:
        raise ValueError("RR 频带没有 FFT bin")
    peak_index = int(indices[int(np.argmax(power[indices]))])
    delta = 0.0
    if peak_index != int(indices[0]) and peak_index != int(indices[-1]):
        logs = np.log(np.asarray(power[peak_index - 1 : peak_index + 2], dtype=np.float64) + 1e-12)
        denominator = logs[0] - 2.0 * logs[1] + logs[2]
        if denominator != 0.0:
            candidate = 0.5 * (logs[0] - logs[2]) / denominator
            if np.isfinite(candidate):
                delta = float(np.clip(candidate, -0.5, 0.5))
    return float(60.0 * (peak_index + delta) * cfg.fs / float(n_fft))


def _rms_envelope(signal: np.ndarray, cfg: TaskMetricConfig) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64)
    starts = range(0, values.size - cfg.envelope_window + 1, cfg.envelope_step)
    return np.asarray(
        [np.sqrt(np.mean(np.square(values[start : start + cfg.envelope_window])) + cfg.envelope_eps) for start in starts],
        dtype=np.float64,
    )


def _spearman_or_worst(pred: np.ndarray, target: np.ndarray, cfg: TaskMetricConfig) -> float:
    target_ranks = rankdata(target, method="average")
    pred_ranks = rankdata(pred, method="average")
    if centered_energy_numpy(pred_ranks) <= cfg.dynamic_eps:
        return -1.0
    return _stable_corr_numpy(pred_ranks, target_ranks, eps=cfg.corr_eps)


def _global_effort(
    pred: np.ndarray,
    target: np.ndarray,
    target_band: np.ndarray,
    cfg: TaskMetricConfig,
) -> tuple[float, bool]:
    target_env = _rms_envelope(target, cfg)
    eligible = centered_energy_numpy(target_band) > cfg.dynamic_eps and centered_energy_numpy(target_env) > cfg.dynamic_eps
    if not eligible:
        return np.nan, False
    return _spearman_or_worst(_rms_envelope(pred, cfg), target_env, cfg), True


def _local_effort(
    pred: np.ndarray,
    target: np.ndarray,
    target_band: np.ndarray,
    cfg: TaskMetricConfig,
) -> tuple[float, bool]:
    values: list[float] = []
    for start in range(0, cfg.length - cfg.local_rr_window + 1, cfg.local_rr_step):
        stop = start + cfg.local_rr_window
        target_env = _rms_envelope(target[start:stop], cfg)
        if centered_energy_numpy(target_band[start:stop]) <= cfg.dynamic_eps:
            continue
        if centered_energy_numpy(target_env) <= cfg.dynamic_eps:
            continue
        values.append(_spearman_or_worst(_rms_envelope(pred[start:stop], cfg), target_env, cfg))
    if not values:
        return np.nan, False
    return float(np.median(values)), True


def _lag_aware_pcc(
    pred: np.ndarray,
    target: np.ndarray,
    target_band: np.ndarray,
    cfg: TaskMetricConfig,
) -> tuple[float, int, bool, bool]:
    start = cfg.max_lag_samples
    stop = cfg.length - cfg.max_lag_samples
    eligible = centered_energy_numpy(target_band[start:stop]) > cfg.dynamic_eps
    if not eligible:
        return np.nan, 0, False, False
    if centered_energy_numpy(pred[start:stop]) <= cfg.dynamic_eps:
        return -1.0, 0, True, False
    target_common = target[start:stop]
    best_corr = -np.inf
    best_lag = 0
    for lag in lag_priority(cfg.max_lag_samples):
        corr = _stable_corr_numpy(pred[start + lag : stop + lag], target_common, eps=cfg.corr_eps)
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    return float(best_corr), int(best_lag), True, True


def _stable_corr_numpy(left: np.ndarray, right: np.ndarray, *, eps: float) -> float:
    a = np.asarray(left, dtype=np.float64) - float(np.mean(left))
    b = np.asarray(right, dtype=np.float64) - float(np.mean(right))
    denominator = np.sqrt(np.sum(np.square(a)) + float(eps)) * np.sqrt(np.sum(np.square(b)) + float(eps))
    return float(np.clip(np.sum(a * b) / denominator, -1.0, 1.0))


def _ibi_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    target_band: np.ndarray,
    best_lag: int,
    joint_eligible: bool,
    cfg: TaskMetricConfig,
) -> tuple[float, float, bool, bool]:
    if not joint_eligible:
        return np.nan, np.nan, False, False
    start = cfg.max_lag_samples
    stop = cfg.length - cfg.max_lag_samples
    if centered_energy_numpy(target_band[start:stop]) <= cfg.dynamic_eps:
        return np.nan, np.nan, False, False
    pred_aligned = pred[start + best_lag : stop + best_lag]
    target_common = target[start:stop]
    target_peaks = _detect_peaks(target_common, cfg)
    if target_peaks.size < 2:
        return np.nan, np.nan, False, False
    pred_peaks = _detect_peaks(pred_aligned, cfg)
    if pred_peaks.size < 2:
        return np.nan, 0.0, False, True
    pairs = _ordered_event_match(target_peaks, pred_peaks, tolerance=cfg.ibi_match_tolerance)
    errors: list[float] = []
    for (target_i, pred_j), (target_next, pred_next) in zip(pairs[:-1], pairs[1:], strict=False):
        if target_next != target_i + 1 or pred_next != pred_j + 1:
            continue
        target_ibi = int(target_peaks[target_next]) - int(target_peaks[target_i])
        pred_ibi = int(pred_peaks[pred_next]) - int(pred_peaks[pred_j])
        errors.append(abs(pred_ibi - target_ibi) / cfg.fs)
    coverage = float(len(errors) / (target_peaks.size - 1))
    interpretable = coverage >= cfg.ibi_coverage_threshold and bool(errors)
    medae = float(np.median(errors)) if interpretable else np.nan
    return medae, coverage, interpretable, True


def _detect_peaks(signal: np.ndarray, cfg: TaskMetricConfig) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64)
    prominence = max(0.2 * float(np.std(values, ddof=0)), 0.08 * float(np.percentile(values, 95) - np.percentile(values, 5)))
    peaks, _ = find_peaks(values, distance=cfg.ibi_peak_distance, prominence=prominence)
    return np.asarray(peaks, dtype=np.int64)


def _ordered_event_match(target: np.ndarray, pred: np.ndarray, *, tolerance: int) -> tuple[tuple[int, int], ...]:
    @lru_cache(maxsize=None)
    def solve(i: int, j: int) -> tuple[int, int, tuple[tuple[int, int], ...]]:
        if i >= target.size or j >= pred.size:
            return 0, 0, ()
        candidates = [solve(i + 1, j), solve(i, j + 1)]
        if abs(int(target[i]) - int(pred[j])) <= int(tolerance):
            count, cost, pairs = solve(i + 1, j + 1)
            candidates.append((count + 1, cost + abs(int(target[i]) - int(pred[j])), ((i, j), *pairs)))
        return min(candidates, key=lambda item: (-item[0], item[1], item[2]))

    return solve(0, 0)[2]


def _respiratory_coherence(pred: np.ndarray, target: np.ndarray, cfg: TaskMetricConfig) -> float:
    segment = int(round(60.0 * cfg.fs))
    step = segment // 2
    window = symmetric_hann_numpy(segment)
    pred_spectra: list[np.ndarray] = []
    target_spectra: list[np.ndarray] = []
    for start in range(0, cfg.length - segment + 1, step):
        pred_values = pred[start : start + segment]
        target_values = target[start : start + segment]
        pred_spectra.append(np.fft.rfft((pred_values - np.mean(pred_values)) * window, n=segment, norm="backward"))
        target_spectra.append(np.fft.rfft((target_values - np.mean(target_values)) * window, n=segment, norm="backward"))
    pred_fft = np.stack(pred_spectra)
    target_fft = np.stack(target_spectra)
    s_pp = np.mean(np.square(np.abs(pred_fft)), axis=0)
    s_tt = np.mean(np.square(np.abs(target_fft)), axis=0)
    s_pt = np.mean(pred_fft * np.conj(target_fft), axis=0)
    denominator = s_pp * s_tt
    coherence = np.zeros_like(denominator, dtype=np.float64)
    valid = denominator > 0.0
    coherence[valid] = np.square(np.abs(s_pt[valid])) / denominator[valid]
    coherence = np.clip(coherence, 0.0, 1.0)
    frequencies = np.fft.rfftfreq(segment, d=1.0 / cfg.fs)
    mask = (frequencies >= cfg.band_low_hz) & (frequencies <= cfg.band_high_hz)
    return float(np.mean(coherence[mask]))


def _constrained_ndtw(pred: np.ndarray, target: np.ndarray, cfg: TaskMetricConfig) -> float:
    u = np.asarray(pred[:: cfg.ndtw_downsample], dtype=np.float64)
    v = np.asarray(target[:: cfg.ndtw_downsample], dtype=np.float64)
    if u.size != v.size:
        raise ValueError("nDTW 下采样后长度不一致")
    previous: dict[int, tuple[float, int]] = {}
    for i in range(u.size):
        current: dict[int, tuple[float, int]] = {}
        for j in range(max(0, i - cfg.ndtw_radius), min(v.size, i + cfg.ndtw_radius + 1)):
            point_cost = abs(float(u[i]) - float(v[j]))
            if i == 0 and j == 0:
                current[j] = (point_cost, 1)
                continue
            predecessors: list[tuple[float, int, int]] = []
            if j - 1 in previous:
                cost, length = previous[j - 1]
                predecessors.append((cost, length, 0))  # diagonal
            if j in previous:
                cost, length = previous[j]
                predecessors.append((cost, length, 1))  # vertical
            if j - 1 in current:
                cost, length = current[j - 1]
                predecessors.append((cost, length, 2))  # horizontal
            if not predecessors:
                continue
            cost, length, _ = min(predecessors, key=lambda item: (item[0], item[1], item[2]))
            current[j] = (cost + point_cost, length + 1)
        previous = current
    if (u.size - 1) not in previous:
        raise RuntimeError("nDTW 约束下不存在完整路径")
    total_cost, path_length = previous[u.size - 1]
    return float(total_cost / path_length)


def _metadata_record(predictions: dict[str, np.ndarray], *, index: int, method: str | None) -> dict[str, Any]:
    record: dict[str, Any] = {}
    if method is not None:
        record["method"] = str(method)
    for key in ("dataset_row_id", "split", "input_set", "samp_id", "coupling_state_id"):
        if key not in predictions:
            continue
        value = np.asarray(predictions[key])[index]
        record[key] = value.item() if np.asarray(value).ndim == 0 else value
    return record
