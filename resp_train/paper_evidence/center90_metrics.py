from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from resp_train.metrics.task import (
    TaskMetricConfig,
    _ibi_metrics,
    _lag_aware_pcc,
    _log_rms_envelope,
    _periodogram,
    _rr_from_power,
)
from resp_train.protocols.respiration import as_batch_waveform_numpy, canonicalize_numpy, centered_energy_numpy


CENTER90_PRIMARY_METRICS = (
    "center90_rr_mae_bpm",
    "center90_ibi_medae_sec",
    "center90_envelope_trajectory_mae",
    "center90_global_envelope_modulation_error",
    "center90_lag_aware_signed_pcc",
)


@dataclass(frozen=True)
class Center90MetricProtocol:
    task: TaskMetricConfig

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "Center90MetricProtocol":
        fs = float(cfg.window.target_fs)
        task = TaskMetricConfig(
            fs=fs,
            length=9000,
            band_low_hz=float(cfg.loss.band_low_hz),
            band_high_hz=float(cfg.loss.band_high_hz),
            scale_eps=float(cfg.loss.scale_eps),
            dynamic_eps=float(cfg.loss.dynamic_eps),
            corr_eps=float(cfg.loss.corr_eps),
            envelope_eps=float(cfg.loss.envelope_eps),
            max_lag_samples=int(round(float(cfg.loss.max_lag_sec) * fs)),
            local_rr_window=9000,
            local_rr_step=9000,
            envelope_window=int(round(float(cfg.loss.envelope_window_sec) * fs)),
            envelope_step=int(round(float(cfg.loss.envelope_step_sec) * fs)),
            ibi_peak_distance=int(cfg.evaluation.ibi_peak_distance_samples),
            ibi_match_tolerance=int(round(float(cfg.evaluation.ibi_match_tolerance_sec) * fs)),
            ibi_coverage_threshold=float(cfg.evaluation.ibi_coverage_threshold),
            ndtw_downsample=10,
            ndtw_radius=3,
            envelope_strata_low=0.0,
            envelope_strata_high=1.0,
            envelope_quantile_method=str(cfg.evaluation.envelope_quantile_method),
        )
        if (
            task.fs != 100.0
            or task.length != 9000
            or task.max_lag_samples != 30
            or task.envelope_window != 1000
            or task.envelope_step != 500
            or task.envelope_quantile_method != "linear"
        ):
            raise ValueError("center90 metric 合同漂移")
        return cls(task=task)


def evaluate_center90_predictions(
    predictions: dict[str, np.ndarray],
    cfg: DictConfig,
    *,
    method: str | None = None,
) -> pd.DataFrame:
    protocol = Center90MetricProtocol.from_config(cfg).task
    pred = as_batch_waveform_numpy(predictions["r_tho_hat"])
    target = as_batch_waveform_numpy(predictions["tho_ref"])
    if pred.shape != target.shape or pred.ndim != 2 or pred.shape[1] != 9000:
        raise ValueError(f"center90 prediction/target 期望 [B,9000]，实际 {pred.shape}/{target.shape}")
    if not np.isfinite(pred).all():
        raise FloatingPointError("center90 prediction 含 NaN/Inf，整个 checkpoint 评价失败")
    if not np.isfinite(target).all():
        raise FloatingPointError("center90 target 含 NaN/Inf")
    pred_band, pred_x = canonicalize_numpy(
        pred,
        fs=protocol.fs,
        low_hz=protocol.band_low_hz,
        high_hz=protocol.band_high_hz,
        scale_eps=protocol.scale_eps,
    )
    target_band, target_x = canonicalize_numpy(
        target,
        fs=protocol.fs,
        low_hz=protocol.band_low_hz,
        high_hz=protocol.band_high_hz,
        scale_eps=protocol.scale_eps,
    )
    records = []
    for index in range(pred.shape[0]):
        record = _metadata_record(predictions, index=index, method=method)
        record.update(
            _evaluate_center90_sample(
                pred_x[index],
                target_x[index],
                pred_band[index],
                target_band[index],
                protocol,
            )
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def validation_center90_rr_mean(predictions: dict[str, np.ndarray], cfg: DictConfig) -> float:
    protocol = Center90MetricProtocol.from_config(cfg).task
    pred = as_batch_waveform_numpy(predictions["r_tho_hat"])
    target = as_batch_waveform_numpy(predictions["tho_ref"])
    if pred.shape != target.shape or pred.ndim != 2 or pred.shape[1] != 9000:
        raise ValueError("center90 RR selector prediction/target shape 错误")
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise FloatingPointError("center90 RR selector prediction/target 含 NaN/Inf")
    pred_band, pred_x = canonicalize_numpy(
        pred,
        fs=protocol.fs,
        low_hz=protocol.band_low_hz,
        high_hz=protocol.band_high_hz,
        scale_eps=protocol.scale_eps,
    )
    target_band, target_x = canonicalize_numpy(
        target,
        fs=protocol.fs,
        low_hz=protocol.band_low_hz,
        high_hz=protocol.band_high_hz,
        scale_eps=protocol.scale_eps,
    )
    values = []
    for index in range(pred.shape[0]):
        error, eligible, _ = _center90_rr(
            pred_x[index], target_x[index], pred_band[index], target_band[index], protocol
        )
        if eligible:
            values.append(error)
    if not values:
        raise ValueError("完整 validation 没有 target-eligible center90 RR sample")
    result = float(np.mean(values))
    if not np.isfinite(result):
        raise FloatingPointError("center90 RR selector 非有限")
    return result


def center90_rr_strictly_improved(value: float, best: float) -> bool:
    if not np.isfinite(value):
        raise FloatingPointError("center90 RR selector value 必须有限")
    return float(value) < float(best)


def summarize_center90_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    row: dict[str, Any] = {"n_samples": int(len(metrics))}
    for metric in CENTER90_PRIMARY_METRICS:
        if metric not in metrics:
            raise ValueError(f"center90 metrics 缺少 {metric}")
        values = pd.to_numeric(metrics[metric], errors="coerce").to_numpy(dtype=np.float64)
        if np.isinf(values).any():
            raise FloatingPointError(f"{metric} 包含 Inf")
        finite = values[np.isfinite(values)]
        row[f"{metric}_mean"] = float(np.mean(finite)) if finite.size else np.nan
        row[f"{metric}_n"] = int(finite.size)
    eligible = metrics["center90_ibi_target_eligible"].astype(bool).to_numpy()
    interpretable = metrics["center90_ibi_interpretable"].astype(bool).to_numpy()
    coverage = pd.to_numeric(metrics["center90_ibi_coverage"], errors="coerce").to_numpy(dtype=np.float64)
    eligible_coverage = coverage[eligible]
    row["center90_ibi_coverage_mean"] = float(np.mean(eligible_coverage)) if eligible_coverage.size else np.nan
    row["center90_ibi_coverage_n"] = int(eligible_coverage.size)
    row["center90_ibi_interpretable_fraction"] = (
        float(np.mean(interpretable[eligible])) if np.any(eligible) else np.nan
    )
    row["center90_ibi_target_eligible_n"] = int(np.sum(eligible))
    return pd.DataFrame([row])


def _evaluate_center90_sample(
    pred_x: np.ndarray,
    target_x: np.ndarray,
    pred_band: np.ndarray,
    target_band: np.ndarray,
    cfg: TaskMetricConfig,
) -> dict[str, Any]:
    rr_error, rr_eligible, rr_degenerate = _center90_rr(
        pred_x, target_x, pred_band, target_band, cfg
    )
    pcc, best_lag, joint_eligible, prediction_dynamic = _lag_aware_pcc(
        pred_x, target_x, target_band, cfg
    )
    ibi_medae, ibi_coverage, ibi_interpretable, ibi_eligible = _ibi_metrics(
        pred_x, target_x, target_band, best_lag, joint_eligible, cfg
    )
    pred_env = _log_rms_envelope(pred_x, cfg)
    target_env = _log_rms_envelope(target_x, cfg)
    if pred_env.size != 17 or target_env.size != 17:
        raise RuntimeError(f"center90 log-RMS envelope 必须恰为 17 点，实际 {pred_env.size}/{target_env.size}")
    if not np.isfinite(pred_env).all() or not np.isfinite(target_env).all():
        raise FloatingPointError("center90 log-RMS envelope 包含 NaN/Inf")
    pred_centered = pred_env - float(np.median(pred_env))
    target_centered = target_env - float(np.median(target_env))
    trajectory = float(np.mean(np.abs(pred_centered - target_centered)))
    pred_range = float(
        np.quantile(pred_env, 0.90, method="linear") - np.quantile(pred_env, 0.10, method="linear")
    )
    target_range = float(
        np.quantile(target_env, 0.90, method="linear")
        - np.quantile(target_env, 0.10, method="linear")
    )
    return {
        "center90_rr_mae_bpm": rr_error,
        "center90_rr_target_eligible": bool(rr_eligible),
        "center90_rr_prediction_degenerate": bool(rr_degenerate),
        "center90_ibi_medae_sec": ibi_medae,
        "center90_ibi_coverage": ibi_coverage,
        "center90_ibi_interpretable": bool(ibi_interpretable),
        "center90_ibi_target_eligible": bool(ibi_eligible),
        "center90_envelope_trajectory_mae": trajectory,
        "center90_global_envelope_modulation_error": abs(pred_range - target_range),
        "center90_lag_aware_signed_pcc": pcc,
        "center90_best_lag_samples": int(best_lag),
        "center90_best_lag_sec": float(best_lag / cfg.fs),
        "center90_joint_target_eligible": bool(joint_eligible),
        "center90_joint_prediction_degenerate": bool(joint_eligible and not prediction_dynamic),
    }


def _center90_rr(
    pred_x: np.ndarray,
    target_x: np.ndarray,
    pred_band: np.ndarray,
    target_band: np.ndarray,
    cfg: TaskMetricConfig,
) -> tuple[float, bool, bool]:
    del pred_band
    eligible = centered_energy_numpy(target_band) > cfg.dynamic_eps
    if not eligible:
        return np.nan, False, False
    target_rr = _rr_from_power(_periodogram(target_x, cfg), 9000, cfg)
    prediction_dynamic = centered_energy_numpy(pred_x) > cfg.dynamic_eps
    if not prediction_dynamic:
        return 39.0, True, True
    pred_rr = _rr_from_power(_periodogram(pred_x, cfg), 9000, cfg)
    if not np.isfinite(pred_rr):
        return 39.0, True, True
    return abs(float(pred_rr) - float(target_rr)), True, False


def _metadata_record(
    predictions: dict[str, np.ndarray], *, index: int, method: str | None
) -> dict[str, Any]:
    record: dict[str, Any] = {}
    if method is not None:
        record["method"] = str(method)
    for key in ("dataset_row_id", "split", "input_set", "samp_id", "coupling_state_id"):
        if key in predictions:
            value = np.asarray(predictions[key])[index]
            record[key] = value.item() if np.asarray(value).ndim == 0 else value
    return record


__all__ = [
    "CENTER90_PRIMARY_METRICS",
    "Center90MetricProtocol",
    "center90_rr_strictly_improved",
    "evaluate_center90_predictions",
    "summarize_center90_metrics",
    "validation_center90_rr_mean",
]
