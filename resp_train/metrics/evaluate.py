from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import sys
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm.auto import tqdm

from resp_train.metrics.signal import (
    band_distribution,
    band_limited_corr_from_filtered,
    bandpass_filter,
    best_lag_correlation_from_filtered,
    estimate_peak_rate_bpm,
    estimate_robust_peak_rate_bpm,
    estimate_spectral_rate_bpm_from_distribution,
    lag_aligned_overlap,
    local_rr_metrics,
    local_rr_metrics_from_rate_traces,
    local_rr_rate_trace,
    local_rr_v2_metrics,
    local_rr_v2_rate_trace,
    local_rr_v3_metrics,
    local_rr_v3_rate_trace,
    relative_envelope_metrics,
    rms_envelope,
    spectrum_similarity_from_distributions,
    zero_crossing_counts,
)


def evaluate_prediction_dict(
    predictions: dict[str, np.ndarray],
    cfg: DictConfig,
    *,
    method: str,
    show_progress: bool | None = None,
    target_features: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """将模型预测字典转换为逐窗口评价指标表。"""
    _validate_predictions(predictions)
    if target_features is not None:
        _validate_target_feature_cache(predictions, target_features)
    fs = float(cfg.window.target_fs)
    low_hz = float(cfg.loss.spectrum_low_hz)
    high_hz = float(cfg.loss.spectrum_high_hz)
    env_window = max(1, int(round(fs * float(cfg.loss.envelope_window_sec))))
    evaluation_cfg = cfg.get("evaluation", {})
    max_lag_sec = float(evaluation_cfg.get("max_lag_sec", 1.0))
    lag_bandpass_order = int(evaluation_cfg.get("lag_bandpass_order", 4))
    raw_peak_min_good_segment_sec = float(evaluation_cfg.get("raw_peak_min_good_segment_sec", 20.0))
    local_rr_window_sec = float(evaluation_cfg.get("local_rr_window_sec", 20.0))
    local_rr_step_sec = float(evaluation_cfg.get("local_rr_step_sec", 5.0))
    local_rr_v2_window_sec = float(evaluation_cfg.get("local_rr_v2_window_sec", 40.0))
    local_rr_v2_step_sec = float(evaluation_cfg.get("local_rr_v2_step_sec", 10.0))
    local_rr_v3_window_sec = float(evaluation_cfg.get("local_rr_v3_window_sec", 40.0))
    local_rr_v3_step_sec = float(evaluation_cfg.get("local_rr_v3_step_sec", 10.0))
    metric_workers = _metric_worker_count(evaluation_cfg)

    preds = np.asarray(predictions["r_tho_hat"])
    targets = np.asarray(predictions["tho_ref"])
    indices = range(preds.shape[0])

    def evaluate_one(idx: int) -> dict[str, Any]:
        return _evaluate_one_window(
            idx,
            predictions=predictions,
            preds=preds,
            targets=targets,
            method=method,
            fs=fs,
            low_hz=low_hz,
            high_hz=high_hz,
            env_window=env_window,
            envelope_window_sec=float(cfg.loss.envelope_window_sec),
            max_lag_sec=max_lag_sec,
            lag_bandpass_order=lag_bandpass_order,
            raw_peak_min_good_segment_sec=raw_peak_min_good_segment_sec,
            local_rr_window_sec=local_rr_window_sec,
            local_rr_step_sec=local_rr_step_sec,
            local_rr_v2_window_sec=local_rr_v2_window_sec,
            local_rr_v2_step_sec=local_rr_v2_step_sec,
            local_rr_v3_window_sec=local_rr_v3_window_sec,
            local_rr_v3_step_sec=local_rr_v3_step_sec,
            target_feature=_target_feature_at(target_features, idx) if target_features is not None else None,
        )

    iterable = indices
    if metric_workers > 1:
        with ThreadPoolExecutor(max_workers=metric_workers) as pool:
            progress = tqdm(
                pool.map(evaluate_one, indices),
                desc="compute val metrics",
                leave=False,
                disable=not _should_show_eval_progress(show_progress),
            )
            return pd.DataFrame.from_records(list(progress))

    progress = tqdm(
        iterable,
        desc="compute val metrics",
        leave=False,
        disable=not _should_show_eval_progress(show_progress),
    )
    records = [evaluate_one(idx) for idx in progress]
    return pd.DataFrame.from_records(records)


def _metric_worker_count(evaluation_cfg: Any) -> int:
    workers = int(evaluation_cfg.get("metric_workers", 1))
    return max(1, workers)


def build_target_feature_cache(
    predictions: dict[str, np.ndarray],
    cfg: DictConfig,
    *,
    show_progress: bool | None = None,
) -> dict[str, np.ndarray]:
    """预计算只依赖 target/mask/eval 配置的逐窗口特征，用于多模型评价复用。"""

    _validate_predictions(predictions)
    context = target_feature_context(cfg)
    targets = np.asarray(predictions["tho_ref"])
    records: list[dict[str, Any]] = []
    indices = range(targets.shape[0])
    progress = tqdm(
        indices,
        desc="compute target features",
        leave=False,
        disable=not _should_show_eval_progress(show_progress),
    )
    for idx in progress:
        target = np.asarray(targets[idx], dtype=np.float64).reshape(-1)
        rr_peak_valid_mask = _rr_peak_valid_mask(predictions, idx, expected_size=target.size)
        records.append(build_target_feature_record(target, rr_peak_valid_mask, context))
    return stack_target_feature_records(records)


def target_feature_context(cfg: DictConfig) -> dict[str, float | int]:
    """提取 target-side 特征计算所需的标量配置，便于进程间传递。"""

    fs = float(cfg.window.target_fs)
    evaluation_cfg = cfg.get("evaluation", {})
    return {
        "fs": fs,
        "low_hz": float(cfg.loss.spectrum_low_hz),
        "high_hz": float(cfg.loss.spectrum_high_hz),
        "env_window": max(1, int(round(fs * float(cfg.loss.envelope_window_sec)))),
        "lag_bandpass_order": int(evaluation_cfg.get("lag_bandpass_order", 4)),
        "raw_peak_min_good_segment_sec": float(evaluation_cfg.get("raw_peak_min_good_segment_sec", 20.0)),
        "local_rr_window_sec": float(evaluation_cfg.get("local_rr_window_sec", 20.0)),
        "local_rr_step_sec": float(evaluation_cfg.get("local_rr_step_sec", 5.0)),
        "local_rr_v2_window_sec": float(evaluation_cfg.get("local_rr_v2_window_sec", 40.0)),
        "local_rr_v2_step_sec": float(evaluation_cfg.get("local_rr_v2_step_sec", 10.0)),
        "local_rr_v3_window_sec": float(evaluation_cfg.get("local_rr_v3_window_sec", 40.0)),
        "local_rr_v3_step_sec": float(evaluation_cfg.get("local_rr_v3_step_sec", 10.0)),
    }


def build_target_feature_record(
    target: np.ndarray,
    rr_peak_valid_mask: np.ndarray,
    context: dict[str, float | int],
) -> dict[str, Any]:
    """计算单个窗口的 target-only 特征；串行和 chunk 并行共用。"""

    fs = float(context["fs"])
    low_hz = float(context["low_hz"])
    high_hz = float(context["high_hz"])
    target_env = rms_envelope(target, int(context["env_window"]))
    target_band_distribution = band_distribution(target, fs=fs, low_hz=low_hz, high_hz=high_hz)
    target_rr_peak, target_rr_peak_segment_count = _estimate_masked_peak_rate_bpm(
        target,
        rr_peak_valid_mask,
        fs=fs,
        distance_sec=2.0,
        low_hz=low_hz,
        high_hz=high_hz,
        min_good_segment_sec=float(context["raw_peak_min_good_segment_sec"]),
    )
    target_filtered = bandpass_filter(
        target,
        fs=fs,
        low_hz=low_hz,
        high_hz=high_hz,
        order=int(context["lag_bandpass_order"]),
    )
    target_rr_peak_band = estimate_peak_rate_bpm(
        target_filtered,
        fs=fs,
        distance_sec=2.0,
        low_hz=low_hz,
        high_hz=high_hz,
    )
    target_rr_peak_band_robust = estimate_robust_peak_rate_bpm(
        target_filtered,
        fs=fs,
        low_hz=low_hz,
        high_hz=high_hz,
    )
    target_breath_count_zero_cross_counts = zero_crossing_counts(target_filtered)
    return {
        "target_env": target_env,
        "target_band_freqs": target_band_distribution["freqs"],
        "target_band_power": target_band_distribution["power"],
        "target_rr_spec_bpm": estimate_spectral_rate_bpm_from_distribution(target_band_distribution),
        "target_rr_peak_bpm": target_rr_peak,
        "target_rr_peak_segment_count": target_rr_peak_segment_count,
        "target_rr_peak_unmasked_bpm": estimate_peak_rate_bpm(
            target,
            fs=fs,
            distance_sec=2.0,
            low_hz=low_hz,
            high_hz=high_hz,
        ),
        "target_filtered": target_filtered,
        "target_rr_peak_band_bpm": target_rr_peak_band,
        "target_rr_peak_band_robust_bpm": target_rr_peak_band_robust,
        "target_breath_count_zero_cross": target_breath_count_zero_cross_counts["cycle"],
        "target_breath_count_zero_cross_up": target_breath_count_zero_cross_counts["up"],
        "target_breath_count_zero_cross_down": target_breath_count_zero_cross_counts["down"],
        "target_local_rr_rates": local_rr_rate_trace(
            target_filtered,
            fs=fs,
            window_sec=float(context["local_rr_window_sec"]),
            step_sec=float(context["local_rr_step_sec"]),
            low_hz=low_hz,
            high_hz=high_hz,
        ),
        "target_local_rr_v2_rates": local_rr_v2_rate_trace(
            target_filtered,
            fs=fs,
            window_sec=float(context["local_rr_v2_window_sec"]),
            step_sec=float(context["local_rr_v2_step_sec"]),
            low_hz=low_hz,
            high_hz=high_hz,
        ),
        "target_local_rr_v3_rates": local_rr_v3_rate_trace(
            target_filtered,
            fs=fs,
            window_sec=float(context["local_rr_v3_window_sec"]),
            step_sec=float(context["local_rr_v3_step_sec"]),
            low_hz=low_hz,
            high_hz=high_hz,
        ),
    }


def stack_target_feature_records(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    if not records:
        raise ValueError("target feature cache 不能为空")
    return {key: np.stack([np.asarray(record[key]) for record in records], axis=0) for key in records[0]}


def _validate_target_feature_cache(predictions: dict[str, np.ndarray], target_features: dict[str, np.ndarray]) -> None:
    required = {
        "target_env",
        "target_band_freqs",
        "target_band_power",
        "target_rr_spec_bpm",
        "target_rr_peak_bpm",
        "target_rr_peak_unmasked_bpm",
        "target_filtered",
        "target_rr_peak_band_bpm",
        "target_rr_peak_band_robust_bpm",
        "target_breath_count_zero_cross",
        "target_breath_count_zero_cross_up",
        "target_breath_count_zero_cross_down",
        "target_local_rr_rates",
        "target_local_rr_v2_rates",
        "target_local_rr_v3_rates",
    }
    missing = sorted(required - set(target_features))
    if missing:
        raise KeyError(f"target feature cache 缺少字段: {missing}")
    n_windows = int(np.asarray(predictions["tho_ref"]).shape[0])
    for key in required:
        value = np.asarray(target_features[key])
        if value.shape[:1] != (n_windows,):
            raise ValueError(f"target feature cache 字段 {key} 第一维必须等于窗口数: {value.shape} vs {n_windows}")


def _target_feature_at(target_features: dict[str, np.ndarray] | None, idx: int) -> dict[str, np.ndarray] | None:
    if target_features is None:
        return None
    return {key: np.asarray(value)[idx] for key, value in target_features.items()}


def _evaluate_one_window(
    idx: int,
    *,
    predictions: dict[str, np.ndarray],
    preds: np.ndarray,
    targets: np.ndarray,
    method: str,
    fs: float,
    low_hz: float,
    high_hz: float,
    env_window: int,
    envelope_window_sec: float,
    max_lag_sec: float,
    lag_bandpass_order: int,
    raw_peak_min_good_segment_sec: float,
    local_rr_window_sec: float,
    local_rr_step_sec: float,
    local_rr_v2_window_sec: float,
    local_rr_v2_step_sec: float,
    local_rr_v3_window_sec: float,
    local_rr_v3_step_sec: float,
    target_feature: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    pred = np.asarray(preds[idx], dtype=np.float64).reshape(-1)
    target = np.asarray(targets[idx], dtype=np.float64).reshape(-1)
    rr_peak_valid_mask = _rr_peak_valid_mask(predictions, idx, expected_size=pred.size)
    pred_env = rms_envelope(pred, env_window)
    target_env = (
        np.asarray(target_feature["target_env"], dtype=np.float64).reshape(-1)
        if target_feature is not None
        else rms_envelope(target, env_window)
    )
    pred_band_distribution = band_distribution(pred, fs=fs, low_hz=low_hz, high_hz=high_hz)
    target_band_distribution = (
        {
            "freqs": np.asarray(target_feature["target_band_freqs"], dtype=np.float64).reshape(-1),
            "power": np.asarray(target_feature["target_band_power"], dtype=np.float64).reshape(-1),
        }
        if target_feature is not None
        else band_distribution(target, fs=fs, low_hz=low_hz, high_hz=high_hz)
    )
    pred_rr_spec = estimate_spectral_rate_bpm_from_distribution(pred_band_distribution)
    target_rr_spec = (
        float(target_feature["target_rr_spec_bpm"])
        if target_feature is not None
        else estimate_spectral_rate_bpm_from_distribution(target_band_distribution)
    )
    pred_rr_peak_unmasked = estimate_peak_rate_bpm(
        pred,
        fs=fs,
        distance_sec=2.0,
        low_hz=low_hz,
        high_hz=high_hz,
    )
    target_rr_peak_unmasked = (
        float(target_feature["target_rr_peak_unmasked_bpm"])
        if target_feature is not None
        else estimate_peak_rate_bpm(
            target,
            fs=fs,
            distance_sec=2.0,
            low_hz=low_hz,
            high_hz=high_hz,
        )
    )
    pred_rr_peak, rr_peak_segment_count = _estimate_masked_peak_rate_bpm(
        pred,
        rr_peak_valid_mask,
        fs=fs,
        distance_sec=2.0,
        low_hz=low_hz,
        high_hz=high_hz,
        min_good_segment_sec=raw_peak_min_good_segment_sec,
    )
    target_rr_peak = (
        float(target_feature["target_rr_peak_bpm"])
        if target_feature is not None
        else _estimate_masked_peak_rate_bpm(
            target,
            rr_peak_valid_mask,
            fs=fs,
            distance_sec=2.0,
            low_hz=low_hz,
            high_hz=high_hz,
            min_good_segment_sec=raw_peak_min_good_segment_sec,
        )[0]
    )
    pred_filtered = bandpass_filter(
        pred,
        fs=fs,
        low_hz=low_hz,
        high_hz=high_hz,
        order=lag_bandpass_order,
    )
    target_filtered = (
        np.asarray(target_feature["target_filtered"], dtype=np.float64).reshape(-1)
        if target_feature is not None
        else bandpass_filter(
            target,
            fs=fs,
            low_hz=low_hz,
            high_hz=high_hz,
            order=lag_bandpass_order,
        )
    )
    pred_rr_peak_band = estimate_peak_rate_bpm(
        pred_filtered,
        fs=fs,
        distance_sec=2.0,
        low_hz=low_hz,
        high_hz=high_hz,
    )
    target_rr_peak_band = (
        float(target_feature["target_rr_peak_band_bpm"])
        if target_feature is not None
        else estimate_peak_rate_bpm(
            target_filtered,
            fs=fs,
            distance_sec=2.0,
            low_hz=low_hz,
            high_hz=high_hz,
        )
    )
    pred_rr_peak_band_robust = estimate_robust_peak_rate_bpm(
        pred_filtered,
        fs=fs,
        low_hz=low_hz,
        high_hz=high_hz,
    )
    target_rr_peak_band_robust = (
        float(target_feature["target_rr_peak_band_robust_bpm"])
        if target_feature is not None
        else estimate_robust_peak_rate_bpm(
            target_filtered,
            fs=fs,
            low_hz=low_hz,
            high_hz=high_hz,
        )
    )
    pred_breath_count_zero_cross_counts = zero_crossing_counts(pred_filtered)
    target_breath_count_zero_cross_counts = (
        {
            "cycle": int(target_feature["target_breath_count_zero_cross"]),
            "up": int(target_feature["target_breath_count_zero_cross_up"]),
            "down": int(target_feature["target_breath_count_zero_cross_down"]),
        }
        if target_feature is not None
        else zero_crossing_counts(target_filtered)
    )
    pred_breath_count_zero_cross = pred_breath_count_zero_cross_counts["cycle"]
    target_breath_count_zero_cross = target_breath_count_zero_cross_counts["cycle"]
    rel_env = relative_envelope_metrics(
        pred,
        target,
        fs=fs,
        envelope_window_sec=envelope_window_sec,
    )
    lag_metrics = best_lag_correlation_from_filtered(
        pred_filtered,
        target_filtered,
        fs=fs,
        max_lag_sec=max_lag_sec,
        low_hz=low_hz,
    )
    lag4_metrics = best_lag_correlation_from_filtered(
        pred_filtered,
        target_filtered,
        fs=fs,
        max_lag_sec=4.0,
        low_hz=low_hz,
    )
    rel_env_lag4 = _lag_aligned_relative_envelope_metrics(
        pred,
        target,
        lag_sec=lag4_metrics["best_lag_sec"],
        fs=fs,
        envelope_window_sec=envelope_window_sec,
    )
    local_rr = (
        local_rr_metrics_from_rate_traces(
            local_rr_rate_trace(
                pred_filtered,
                fs=fs,
                window_sec=local_rr_window_sec,
                step_sec=local_rr_step_sec,
                low_hz=low_hz,
                high_hz=high_hz,
            ),
            np.asarray(target_feature["target_local_rr_rates"], dtype=np.float64).reshape(-1),
        )
        if target_feature is not None
        else local_rr_metrics(
            pred_filtered,
            target_filtered,
            fs=fs,
            window_sec=local_rr_window_sec,
            step_sec=local_rr_step_sec,
            low_hz=low_hz,
            high_hz=high_hz,
        )
    )
    local_rr_v2 = (
        local_rr_metrics_from_rate_traces(
            local_rr_v2_rate_trace(
                pred_filtered,
                fs=fs,
                window_sec=local_rr_v2_window_sec,
                step_sec=local_rr_v2_step_sec,
                low_hz=low_hz,
                high_hz=high_hz,
            ),
            np.asarray(target_feature["target_local_rr_v2_rates"], dtype=np.float64).reshape(-1),
        )
        if target_feature is not None
        else local_rr_v2_metrics(
            pred_filtered,
            target_filtered,
            fs=fs,
            window_sec=local_rr_v2_window_sec,
            step_sec=local_rr_v2_step_sec,
            low_hz=low_hz,
            high_hz=high_hz,
        )
    )
    local_rr_v3 = (
        local_rr_metrics_from_rate_traces(
            local_rr_v3_rate_trace(
                pred_filtered,
                fs=fs,
                window_sec=local_rr_v3_window_sec,
                step_sec=local_rr_v3_step_sec,
                low_hz=low_hz,
                high_hz=high_hz,
            ),
            np.asarray(target_feature["target_local_rr_v3_rates"], dtype=np.float64).reshape(-1),
        )
        if target_feature is not None
        else local_rr_v3_metrics(
            pred_filtered,
            target_filtered,
            fs=fs,
            window_sec=local_rr_v3_window_sec,
            step_sec=local_rr_v3_step_sec,
            low_hz=low_hz,
            high_hz=high_hz,
        )
    )

    return {
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
        "pred_rr_peak_unmasked_bpm": pred_rr_peak_unmasked,
        "target_rr_peak_unmasked_bpm": target_rr_peak_unmasked,
        "rr_peak_unmasked_abs_error": _abs_error_or_nan(pred_rr_peak_unmasked, target_rr_peak_unmasked),
        "rr_peak_valid_ratio": float(np.mean(rr_peak_valid_mask)),
        "rr_peak_valid_segment_count": int(rr_peak_segment_count),
        "pred_rr_peak_band_bpm": pred_rr_peak_band,
        "target_rr_peak_band_bpm": target_rr_peak_band,
        "rr_peak_band_abs_error": _abs_error_or_nan(pred_rr_peak_band, target_rr_peak_band),
        "pred_rr_peak_band_robust_bpm": pred_rr_peak_band_robust,
        "target_rr_peak_band_robust_bpm": target_rr_peak_band_robust,
        "rr_peak_band_robust_abs_error": _abs_error_or_nan(
            pred_rr_peak_band_robust,
            target_rr_peak_band_robust,
        ),
        "pred_breath_count_zero_cross": pred_breath_count_zero_cross,
        "target_breath_count_zero_cross": target_breath_count_zero_cross,
        "pred_breath_count_zero_cross_up": pred_breath_count_zero_cross_counts["up"],
        "target_breath_count_zero_cross_up": target_breath_count_zero_cross_counts["up"],
        "pred_breath_count_zero_cross_down": pred_breath_count_zero_cross_counts["down"],
        "target_breath_count_zero_cross_down": target_breath_count_zero_cross_counts["down"],
        "breath_count_zero_cross_abs_error": abs(pred_breath_count_zero_cross - target_breath_count_zero_cross),
        "envelope_corr": _corrcoef_or_nan(pred_env, target_env),
        "relative_envelope_corr": rel_env["relative_envelope_corr"],
        "relative_envelope_mae": rel_env["relative_envelope_mae"],
        "relative_envelope_corr_lag4s": rel_env_lag4["relative_envelope_corr"],
        "relative_envelope_mae_lag4s": rel_env_lag4["relative_envelope_mae"],
        "spectrum_similarity": spectrum_similarity_from_distributions(
            pred_band_distribution,
            target_band_distribution,
        ),
        "band_limited_corr": band_limited_corr_from_filtered(pred_filtered, target_filtered),
        "best_lag_corr": lag_metrics["best_lag_corr"],
        "best_lag_sec": lag_metrics["best_lag_sec"],
        "best_lag_corr_4s": lag4_metrics["best_lag_corr"],
        "best_lag_sec_4s": lag4_metrics["best_lag_sec"],
        "local_rr_mae": local_rr["local_rr_mae"],
        "local_rr_corr": local_rr["local_rr_corr"],
        "local_rr_valid_frac": local_rr["local_rr_valid_frac"],
        "local_rr_v2_mae": _local_rr_value(local_rr_v2, prefix="local_rr_v2", suffix="mae"),
        "local_rr_v2_corr": _local_rr_value(local_rr_v2, prefix="local_rr_v2", suffix="corr"),
        "local_rr_v2_valid_frac": _local_rr_value(local_rr_v2, prefix="local_rr_v2", suffix="valid_frac"),
        "local_rr_v3_mae": _local_rr_value(local_rr_v3, prefix="local_rr_v3", suffix="mae"),
        "local_rr_v3_corr": _local_rr_value(local_rr_v3, prefix="local_rr_v3", suffix="corr"),
        "local_rr_v3_valid_frac": _local_rr_value(local_rr_v3, prefix="local_rr_v3", suffix="valid_frac"),
    }


def _local_rr_value(metrics: dict[str, float], *, prefix: str, suffix: str) -> float:
    prefixed = f"{prefix}_{suffix}"
    if prefixed in metrics:
        return metrics[prefixed]
    return metrics[f"local_rr_{suffix}"]


def _should_show_eval_progress(show_progress: bool | None) -> bool:
    """评价指标计算进度；None 时仅交互终端显示。"""
    if show_progress is not None:
        return bool(show_progress)
    return sys.stderr.isatty()


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


def _lag_aligned_relative_envelope_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    lag_sec: float,
    fs: float,
    envelope_window_sec: float,
) -> dict[str, float]:
    if not np.isfinite(lag_sec):
        return {"relative_envelope_corr": float("nan"), "relative_envelope_mae": float("nan")}
    lag_samples = int(round(float(lag_sec) * float(fs)))
    pred_overlap, target_overlap = lag_aligned_overlap(pred, target, lag_samples=lag_samples)
    if pred_overlap.size < 2 or target_overlap.size < 2:
        return {"relative_envelope_corr": float("nan"), "relative_envelope_mae": float("nan")}
    return relative_envelope_metrics(
        pred_overlap,
        target_overlap,
        fs=fs,
        envelope_window_sec=envelope_window_sec,
    )


def _abs_error_or_nan(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    return float(abs(a - b))


def _rr_peak_valid_mask(predictions: dict[str, np.ndarray], idx: int, *, expected_size: int) -> np.ndarray:
    values = predictions.get("rr_peak_valid_mask")
    if values is None:
        return np.ones(expected_size, dtype=np.bool_)
    mask = np.asarray(values[idx]).reshape(-1).astype(np.bool_, copy=False)
    if mask.size != expected_size:
        raise ValueError(f"rr_peak_valid_mask 长度必须等于窗口长度: mask={mask.size} expected={expected_size}")
    return mask


def _estimate_masked_peak_rate_bpm(
    signal: np.ndarray,
    valid_mask: np.ndarray,
    *,
    fs: float,
    distance_sec: float,
    low_hz: float,
    high_hz: float,
    min_good_segment_sec: float,
) -> tuple[float, int]:
    """按连续共同好段估计 raw peak RR，避免拼接好段扭曲峰间距。"""
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    mask = np.asarray(valid_mask, dtype=np.bool_).reshape(-1)
    if x.size != mask.size:
        raise ValueError(f"signal 和 valid_mask 长度不一致: signal={x.size} mask={mask.size}")

    min_samples = max(1, int(round(float(min_good_segment_sec) * float(fs))))
    rates: list[float] = []
    weights: list[int] = []
    segment_count = 0
    for start, end in _true_spans(mask):
        length = end - start
        if length < min_samples:
            continue
        segment_count += 1
        rate = estimate_peak_rate_bpm(
            x[start:end],
            fs=fs,
            distance_sec=distance_sec,
            low_hz=low_hz,
            high_hz=high_hz,
        )
        if np.isfinite(rate):
            rates.append(float(rate))
            weights.append(length)

    if not rates:
        return float("nan"), segment_count
    return float(np.average(np.asarray(rates, dtype=np.float64), weights=np.asarray(weights, dtype=np.float64))), segment_count


def _true_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask):
        if bool(value) and start is None:
            start = idx
        elif not bool(value) and start is not None:
            spans.append((start, idx))
            start = None
    if start is not None:
        spans.append((start, int(mask.size)))
    return spans
