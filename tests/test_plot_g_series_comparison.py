from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.export_g_series_comparison_cache import ComparisonSpec, REQUIRED_MODELS
from scripts.plot_g_series_comparison import (
    CANONICAL_METRIC_COLUMNS,
    align_canonical_metrics,
    input_stability_features,
    load_canonical_metrics,
    select_plot_rows,
)


def _metric_frame(row_ids: list[int]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "dataset_row_id": row_ids,
            "split": "test",
            "pred_rr_peak_band_robust_bpm": 12.0,
            "target_rr_peak_band_robust_bpm": 12.5,
            "rr_peak_band_robust_abs_error": 0.5,
            "breath_count_zero_cross_abs_error": 1.0,
            "best_lag_corr_4s": 0.8,
            "best_lag_sec_4s": 0.1,
            "relative_envelope_corr_lag4s": 0.7,
            "local_rr_mae": 0.4,
            "local_rr_corr": 0.6,
            "local_rr_valid_frac": 1.0,
        }
    )
    assert set(CANONICAL_METRIC_COLUMNS) <= set(frame.columns)
    return frame


def test_align_canonical_metrics_requires_one_row_per_cache_id() -> None:
    cache_ids = [10, 20, 30]
    frames = {label: _metric_frame(cache_ids) for label in REQUIRED_MODELS}

    aligned = align_canonical_metrics(cache_ids, frames)

    assert aligned.index.tolist() == cache_ids
    assert aligned.loc[10, "g0_time_only__breath_count_zero_cross_bpm_error"] == pytest.approx(1 / 3)

    frames["g3_c_bandenergy"] = _metric_frame([10, 20])
    with pytest.raises(ValueError, match="缺少 cache row"):
        align_canonical_metrics(cache_ids, frames)


def test_align_canonical_metrics_rejects_duplicate_ids_and_non_test_split() -> None:
    frames = {label: _metric_frame([10, 20]) for label in REQUIRED_MODELS}
    frames["g0_time_only"] = pd.concat([frames["g0_time_only"], frames["g0_time_only"].iloc[[0]]])
    with pytest.raises(ValueError, match="重复"):
        align_canonical_metrics([10, 20], frames)

    frames = {label: _metric_frame([10, 20]) for label in REQUIRED_MODELS}
    frames["g0_time_only"].loc[0, "split"] = "val"
    with pytest.raises(ValueError, match="test split"):
        align_canonical_metrics([10, 20], frames)


def test_select_plot_rows_excludes_exact_top_twenty_percent_input_stable() -> None:
    features = pd.DataFrame(
        {
            "dataset_row_id": [1, 2, 3, 4, 5],
            "spectral_peak_fraction": [0.9, 0.7, 0.5, 0.3, 0.1],
            "local_rr_valid_frac": [1.0, 1.0, 1.0, 0.8, 0.2],
            "local_rr_iqr_bpm": [0.1, 0.2, 0.5, 1.0, 3.0],
        }
    )

    selected = select_plot_rows(features, filter_mode="exclude-input-stable", stable_fraction=0.2)

    assert selected.loc[selected.dataset_row_id == 1, "plot_status"].item() == "input_stable_excluded"
    assert selected.plot_status.eq("input_stable_excluded").sum() == 1
    assert select_plot_rows(features, filter_mode="all", stable_fraction=0.2).plot_status.eq("retained").all()


def test_input_stability_features_uses_only_bcg_waveform() -> None:
    fs = 100.0
    time = np.arange(18_000) / fs
    bcg = np.sin(2.0 * np.pi * 0.2 * time)

    features = input_stability_features(bcg, fs=fs, low_hz=0.05, high_hz=0.7, order=4)

    assert set(features) == {"spectral_peak_fraction", "local_rr_valid_frac", "local_rr_iqr_bpm"}
    assert features["spectral_peak_fraction"] > 0.5
    assert features["local_rr_valid_frac"] == pytest.approx(1.0)


def test_load_canonical_metrics_uses_frozen_label_seed_filenames(tmp_path) -> None:
    specs = []
    for index, label in enumerate(REQUIRED_MODELS):
        seed = 100 + index
        _metric_frame([1, 2]).to_csv(tmp_path / f"{label}_{seed}_test_metrics.csv", index=False)
        specs.append(
            ComparisonSpec(
                label=label,
                seed=seed,
                checkpoint=tmp_path / f"{label}.pt",
                selection_source="validation_topk_legacy_task_selection",
            )
        )

    aligned = load_canonical_metrics(tmp_path, specs, cache_row_ids=[1, 2])

    assert aligned.shape[0] == 2
    assert "g3_c_bandenergy__local_rr_valid_frac" in aligned.columns
