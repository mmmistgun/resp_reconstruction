from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from resp_train.data.factory import build_window_data
from resp_train.metrics.evaluate import _estimate_masked_peak_rate_bpm
from resp_train.metrics.signal import (
    bandpass_filter,
    estimate_bandpassed_peak_rate_bpm,
    estimate_peak_rate_bpm,
    estimate_spectral_rate_bpm,
    rms_envelope,
    spectrum_similarity,
)


def evaluate_baseline_dataset(dataset: Any, cfg: DictConfig) -> pd.DataFrame:
    fs = float(cfg.window.get("target_fs", 100.0))
    low_hz = float(cfg.baseline.get("bandpass_low_hz", cfg.loss.get("spectrum_low_hz", 0.05)))
    high_hz = float(cfg.baseline.get("bandpass_high_hz", cfg.loss.get("spectrum_high_hz", 0.7)))
    order = int(cfg.baseline.get("filter_order", 4))
    evaluation_cfg = cfg.get("evaluation", {})
    raw_peak_min_good_segment_sec = float(evaluation_cfg.get("raw_peak_min_good_segment_sec", 20.0))
    envelope_window = max(1, int(round(float(cfg.loss.get("envelope_window_sec", 2.0)) * fs)))
    records: list[dict[str, Any]] = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        records.append(
            _evaluate_baseline_sample(
                sample,
                fs=fs,
                low_hz=low_hz,
                high_hz=high_hz,
                order=order,
                envelope_window=envelope_window,
                raw_peak_min_good_segment_sec=raw_peak_min_good_segment_sec,
            )
        )
    return pd.DataFrame.from_records(records)


def run_baseline(cfg: DictConfig, output: str | Path) -> pd.DataFrame:
    """在验证集上运行 BCG 平凡基线并写出逐窗口指标。"""
    bundle = build_window_data(
        cfg,
        split=str(cfg.data.get("val_split", "val")),
        max_windows=cfg.data.get("max_val_windows"),
        sample_strategy=str(cfg.data.get("val_sample_strategy", "stratified_random")),
        sample_seed=int(cfg.data.get("val_sample_seed", cfg.data.get("train_sample_seed", 0) + 1)),
        shuffle=False,
    )
    df = evaluate_baseline_dataset(bundle.dataset, cfg)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def _evaluate_baseline_sample(
    sample: dict[str, Any],
    *,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int,
    envelope_window: int,
    raw_peak_min_good_segment_sec: float,
) -> dict[str, Any]:
    meta = sample["meta"]
    x = sample["x"].detach().cpu().numpy().squeeze()
    target = sample["target"].detach().cpu().numpy().squeeze()

    pred = rms_envelope(
        bandpass_filter(x, fs=fs, low_hz=low_hz, high_hz=high_hz, order=order),
        window_samples=envelope_window,
    )
    target_filtered = bandpass_filter(target, fs=fs, low_hz=low_hz, high_hz=high_hz, order=order)
    target_env = rms_envelope(target_filtered, window_samples=envelope_window)

    pred_rr_spec = estimate_spectral_rate_bpm(pred, fs=fs, low_hz=low_hz, high_hz=high_hz)
    target_rr_spec = estimate_spectral_rate_bpm(target_env, fs=fs, low_hz=low_hz, high_hz=high_hz)
    rr_peak_valid_mask = _sample_rr_peak_valid_mask(meta, expected_size=pred.size)
    pred_rr_peak_unmasked = estimate_peak_rate_bpm(pred, fs=fs, distance_sec=2.0, low_hz=low_hz, high_hz=high_hz)
    target_rr_peak_unmasked = estimate_peak_rate_bpm(
        target_env,
        fs=fs,
        distance_sec=2.0,
        low_hz=low_hz,
        high_hz=high_hz,
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
    target_rr_peak, _ = _estimate_masked_peak_rate_bpm(
        target_env,
        rr_peak_valid_mask,
        fs=fs,
        distance_sec=2.0,
        low_hz=low_hz,
        high_hz=high_hz,
        min_good_segment_sec=raw_peak_min_good_segment_sec,
    )
    pred_rr_peak_band = estimate_bandpassed_peak_rate_bpm(
        pred,
        fs=fs,
        distance_sec=2.0,
        low_hz=low_hz,
        high_hz=high_hz,
        order=order,
    )
    target_rr_peak_band = estimate_bandpassed_peak_rate_bpm(
        target_env,
        fs=fs,
        distance_sec=2.0,
        low_hz=low_hz,
        high_hz=high_hz,
        order=order,
    )

    return {
        **meta,
        "method": "bcg_bandpass_rms",
        "pred_rr_spec_bpm": pred_rr_spec,
        "target_rr_spec_bpm": target_rr_spec,
        "rr_spec_abs_error": abs(pred_rr_spec - target_rr_spec),
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
        "envelope_corr": _corrcoef_or_nan(pred, target_env),
        "spectrum_similarity": spectrum_similarity(pred, target_env, fs=fs, low_hz=low_hz, high_hz=high_hz),
    }


def _corrcoef_or_nan(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) <= 0 or np.std(b) <= 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _abs_error_or_nan(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    return float(abs(a - b))


def _sample_rr_peak_valid_mask(meta: dict[str, Any], *, expected_size: int) -> np.ndarray:
    value = meta.get("rr_peak_valid_mask")
    if value is None:
        return np.ones(expected_size, dtype=np.bool_)
    if hasattr(value, "detach"):
        mask = value.detach().cpu().numpy()
    else:
        mask = np.asarray(value)
    mask = np.asarray(mask).reshape(-1).astype(np.bool_, copy=False)
    if mask.size != expected_size:
        raise ValueError(f"rr_peak_valid_mask 长度必须等于窗口长度: mask={mask.size} expected={expected_size}")
    return mask
