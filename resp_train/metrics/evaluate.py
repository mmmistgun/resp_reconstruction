from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from resp_train.metrics.signal import (
    band_limited_corr,
    best_lag_correlation,
    estimate_bandpassed_peak_rate_bpm,
    estimate_bandpassed_zero_crossing_breath_count,
    estimate_peak_rate_bpm,
    estimate_spectral_rate_bpm,
    relative_envelope_metrics,
    rms_envelope,
    spectrum_similarity,
)


def evaluate_prediction_dict(predictions: dict[str, np.ndarray], cfg: DictConfig, *, method: str) -> pd.DataFrame:
    """将模型预测字典转换为逐窗口评价指标表。"""
    _validate_predictions(predictions)
    fs = float(cfg.window.target_fs)
    low_hz = float(cfg.loss.spectrum_low_hz)
    high_hz = float(cfg.loss.spectrum_high_hz)
    env_window = max(1, int(round(fs * float(cfg.loss.envelope_window_sec))))
    evaluation_cfg = cfg.get("evaluation", {})
    max_lag_sec = float(evaluation_cfg.get("max_lag_sec", 1.0))
    lag_bandpass_order = int(evaluation_cfg.get("lag_bandpass_order", 4))

    preds = np.asarray(predictions["r_tho_hat"])
    targets = np.asarray(predictions["tho_ref"])
    records: list[dict[str, Any]] = []
    for idx in range(preds.shape[0]):
        pred = np.asarray(preds[idx], dtype=np.float64).reshape(-1)
        target = np.asarray(targets[idx], dtype=np.float64).reshape(-1)
        pred_env = rms_envelope(pred, env_window)
        target_env = rms_envelope(target, env_window)
        pred_rr_spec = estimate_spectral_rate_bpm(pred, fs=fs, low_hz=low_hz, high_hz=high_hz)
        target_rr_spec = estimate_spectral_rate_bpm(target, fs=fs, low_hz=low_hz, high_hz=high_hz)
        pred_rr_peak = estimate_peak_rate_bpm(pred, fs=fs, distance_sec=2.0, low_hz=low_hz, high_hz=high_hz)
        target_rr_peak = estimate_peak_rate_bpm(target, fs=fs, distance_sec=2.0, low_hz=low_hz, high_hz=high_hz)
        pred_rr_peak_band = estimate_bandpassed_peak_rate_bpm(
            pred,
            fs=fs,
            distance_sec=2.0,
            low_hz=low_hz,
            high_hz=high_hz,
            order=lag_bandpass_order,
        )
        target_rr_peak_band = estimate_bandpassed_peak_rate_bpm(
            target,
            fs=fs,
            distance_sec=2.0,
            low_hz=low_hz,
            high_hz=high_hz,
            order=lag_bandpass_order,
        )
        pred_breath_count_zero_cross = estimate_bandpassed_zero_crossing_breath_count(
            pred,
            fs=fs,
            low_hz=low_hz,
            high_hz=high_hz,
            order=lag_bandpass_order,
        )
        target_breath_count_zero_cross = estimate_bandpassed_zero_crossing_breath_count(
            target,
            fs=fs,
            low_hz=low_hz,
            high_hz=high_hz,
            order=lag_bandpass_order,
        )
        rel_env = relative_envelope_metrics(
            pred,
            target,
            fs=fs,
            envelope_window_sec=float(cfg.loss.envelope_window_sec),
        )
        lag_metrics = best_lag_correlation(
            pred,
            target,
            fs=fs,
            max_lag_sec=max_lag_sec,
            low_hz=low_hz,
            high_hz=high_hz,
            order=lag_bandpass_order,
        )

        records.append(
            {
                "method": str(method),
                "dataset_row_id": int(_meta_value(predictions, "dataset_row_id", idx, default=-1)),
                "split": str(_meta_value(predictions, "split", idx, default="")),
                "input_set": str(_meta_value(predictions, "input_set", idx, default="")),
                "residual_quality_class": str(_meta_value(predictions, "residual_quality_class", idx, default="")),
                "pred_rr_spec_bpm": pred_rr_spec,
                "target_rr_spec_bpm": target_rr_spec,
                "rr_spec_abs_error": _abs_error_or_nan(pred_rr_spec, target_rr_spec),
                "pred_rr_peak_bpm": pred_rr_peak,
                "target_rr_peak_bpm": target_rr_peak,
                "rr_peak_abs_error": _abs_error_or_nan(pred_rr_peak, target_rr_peak),
                "pred_rr_peak_band_bpm": pred_rr_peak_band,
                "target_rr_peak_band_bpm": target_rr_peak_band,
                "rr_peak_band_abs_error": _abs_error_or_nan(pred_rr_peak_band, target_rr_peak_band),
                "pred_breath_count_zero_cross": pred_breath_count_zero_cross,
                "target_breath_count_zero_cross": target_breath_count_zero_cross,
                "breath_count_zero_cross_abs_error": abs(
                    pred_breath_count_zero_cross - target_breath_count_zero_cross
                ),
                "envelope_corr": _corrcoef_or_nan(pred_env, target_env),
                "relative_envelope_corr": rel_env["relative_envelope_corr"],
                "relative_envelope_mae": rel_env["relative_envelope_mae"],
                "spectrum_similarity": spectrum_similarity(pred, target, fs=fs, low_hz=low_hz, high_hz=high_hz),
                "band_limited_corr": band_limited_corr(
                    pred,
                    target,
                    fs=fs,
                    low_hz=low_hz,
                    high_hz=high_hz,
                    order=lag_bandpass_order,
                ),
                "best_lag_corr": lag_metrics["best_lag_corr"],
                "best_lag_sec": lag_metrics["best_lag_sec"],
            }
        )
    return pd.DataFrame.from_records(records)


def _validate_predictions(predictions: dict[str, np.ndarray]) -> None:
    missing = [key for key in ("r_tho_hat", "tho_ref") if key not in predictions]
    if missing:
        raise KeyError(f"预测缺少必需字段: {missing}")
    preds = np.asarray(predictions["r_tho_hat"])
    targets = np.asarray(predictions["tho_ref"])
    if preds.shape[0] == 0:
        raise ValueError("预测不能为空")
    if preds.shape != targets.shape:
        raise ValueError(f"预测和目标 shape 必须一致: pred={preds.shape} target={targets.shape}")
    if preds.ndim < 2:
        raise ValueError(f"预测必须至少包含 batch 和时间维，当前 shape={preds.shape}")


def _meta_value(predictions: dict[str, np.ndarray], key: str, idx: int, *, default: Any) -> Any:
    values = predictions.get(key)
    if values is None:
        return default
    return np.asarray(values)[idx]


def _corrcoef_or_nan(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) <= 0 or np.std(b) <= 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _abs_error_or_nan(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    return float(abs(a - b))
