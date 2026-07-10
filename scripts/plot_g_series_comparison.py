from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal

from scripts.export_g_series_comparison_cache import ComparisonSpec, REQUIRED_MODELS
from resp_train.metrics.signal import bandpass_filter, local_rr_rate_trace


CANONICAL_METRIC_COLUMNS = (
    "dataset_row_id",
    "split",
    "pred_rr_peak_band_robust_bpm",
    "target_rr_peak_band_robust_bpm",
    "rr_peak_band_robust_abs_error",
    "breath_count_zero_cross_abs_error",
    "best_lag_corr_4s",
    "best_lag_sec_4s",
    "relative_envelope_corr_lag4s",
    "local_rr_mae",
    "local_rr_corr",
    "local_rr_valid_frac",
)


def _validate_cache_ids(values: Iterable[int]) -> pd.Index:
    row_ids = pd.Index(np.asarray(list(values), dtype=np.int64), name="dataset_row_id")
    if row_ids.empty:
        raise ValueError("cache dataset_row_id 不能为空")
    if row_ids.has_duplicates:
        raise ValueError("cache dataset_row_id 存在重复")
    return row_ids


def _validate_metrics_frame(label: str, frame: pd.DataFrame, cache_ids: pd.Index) -> pd.DataFrame:
    missing_columns = sorted(set(CANONICAL_METRIC_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"{label} metrics 缺少 canonical 列: {missing_columns}")
    metrics = frame.loc[:, CANONICAL_METRIC_COLUMNS].copy()
    metrics["dataset_row_id"] = pd.to_numeric(metrics["dataset_row_id"], errors="raise").astype(np.int64)
    if metrics["dataset_row_id"].duplicated().any():
        raise ValueError(f"{label} metrics 的 dataset_row_id 存在重复")
    if not metrics["split"].astype(str).eq("test").all():
        raise ValueError(f"{label} metrics 必须全部来自 test split")
    metric_ids = pd.Index(metrics["dataset_row_id"], name="dataset_row_id")
    missing_ids = cache_ids.difference(metric_ids)
    if not missing_ids.empty:
        raise ValueError(f"{label} metrics 缺少 cache row: {missing_ids[:12].tolist()}")
    extra_ids = metric_ids.difference(cache_ids)
    if not extra_ids.empty:
        raise ValueError(f"{label} metrics 含 cache 外 row: {extra_ids[:12].tolist()}")
    return metrics.set_index("dataset_row_id").reindex(cache_ids)


def align_canonical_metrics(
    cache_row_ids: Iterable[int],
    metrics_by_label: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    cache_ids = _validate_cache_ids(cache_row_ids)
    labels = tuple(metrics_by_label)
    if labels != REQUIRED_MODELS:
        raise ValueError(f"metrics label 必须依次为: {list(REQUIRED_MODELS)}")
    frames: list[pd.DataFrame] = []
    for label in REQUIRED_MODELS:
        frame = _validate_metrics_frame(label, metrics_by_label[label], cache_ids)
        frame["breath_count_zero_cross_bpm_error"] = (
            pd.to_numeric(frame["breath_count_zero_cross_abs_error"], errors="coerce") / 3.0
        )
        frame = frame.drop(columns=["split"])
        frame.columns = [f"{label}__{column}" for column in frame.columns]
        frames.append(frame)
    return pd.concat(frames, axis=1).reindex(cache_ids)


def load_canonical_metrics(
    metrics_dir: str | Path,
    specs: Iterable[ComparisonSpec],
    *,
    cache_row_ids: Iterable[int],
) -> pd.DataFrame:
    resolved_specs = tuple(specs)
    if tuple(spec.label for spec in resolved_specs) != REQUIRED_MODELS:
        raise ValueError("canonical metrics 必须使用固定的四个模型")
    root = Path(metrics_dir)
    frames: dict[str, pd.DataFrame] = {}
    for spec in resolved_specs:
        path = root / f"{spec.label}_{spec.seed}_test_metrics.csv"
        if not path.exists():
            raise FileNotFoundError(f"缺少 canonical test metrics: {path}")
        frames[spec.label] = pd.read_csv(path)
    return align_canonical_metrics(cache_row_ids, frames)


def select_plot_rows(
    features: pd.DataFrame,
    *,
    filter_mode: str,
    stable_fraction: float,
) -> pd.DataFrame:
    required = {
        "dataset_row_id",
        "spectral_peak_fraction",
        "local_rr_valid_frac",
        "local_rr_iqr_bpm",
    }
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"输入稳定度特征缺少列: {missing}")
    if not 0.0 < float(stable_fraction) < 1.0:
        raise ValueError("stable_fraction 必须在 (0, 1) 内")
    result = features.copy()
    result["dataset_row_id"] = pd.to_numeric(result["dataset_row_id"], errors="raise").astype(np.int64)
    if result["dataset_row_id"].duplicated().any():
        raise ValueError("输入稳定度特征的 dataset_row_id 存在重复")
    for column in ("spectral_peak_fraction", "local_rr_valid_frac", "local_rr_iqr_bpm"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["input_stability_score"] = (
        0.50 * result["spectral_peak_fraction"].rank(method="average", pct=True)
        + 0.30 * result["local_rr_valid_frac"].rank(method="average", pct=True)
        + 0.20 * (1.0 - result["local_rr_iqr_bpm"].rank(method="average", pct=True))
    )
    if filter_mode == "all":
        result["plot_status"] = "retained"
        return result.sort_values("dataset_row_id").reset_index(drop=True)
    if filter_mode != "exclude-input-stable":
        raise ValueError("filter_mode 只能是 all 或 exclude-input-stable")
    n_excluded = int(math.ceil(float(stable_fraction) * len(result)))
    stable_ids = set(
        result.sort_values(
            ["input_stability_score", "dataset_row_id"], ascending=[False, True]
        ).head(n_excluded)["dataset_row_id"].tolist()
    )
    result["plot_status"] = np.where(
        result["dataset_row_id"].isin(stable_ids),
        "input_stable_excluded",
        "retained",
    )
    return result.sort_values("dataset_row_id").reset_index(drop=True)


def input_stability_features(
    bcg: np.ndarray,
    *,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int,
) -> dict[str, float]:
    values = np.asarray(bcg, dtype=np.float64).reshape(-1)
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError("BCG 输入必须是一维有限信号")
    filtered = bandpass_filter(values, fs=fs, low_hz=low_hz, high_hz=high_hz, order=order)
    freqs, power = scipy_signal.welch(filtered, fs=fs, nperseg=min(4096, filtered.size))
    in_band = (freqs >= float(low_hz)) & (freqs <= float(high_hz))
    if not np.any(in_band):
        raise ValueError("Welch 频率网格未覆盖呼吸带")
    band_power = np.asarray(power[in_band], dtype=np.float64)
    peak_fraction = float(band_power.max() / max(float(band_power.sum()), np.finfo(np.float64).eps))
    rates = local_rr_rate_trace(
        filtered,
        fs=fs,
        window_sec=40.0,
        step_sec=10.0,
        low_hz=low_hz,
        high_hz=high_hz,
    )
    valid = rates[np.isfinite(rates)]
    iqr = float(np.subtract(*np.percentile(valid, [75, 25]))) if valid.size >= 2 else float("inf")
    return {
        "spectral_peak_fraction": peak_fraction,
        "local_rr_valid_frac": float(valid.size / rates.size),
        "local_rr_iqr_bpm": iqr,
    }
