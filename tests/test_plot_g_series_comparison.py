from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from scripts.export_g_series_comparison_cache import ComparisonSpec, REQUIRED_MODELS, spec_fingerprint
from scripts.plot_g_series_comparison import (
    CANONICAL_METRIC_COLUMNS,
    _build_window_index,
    align_canonical_metrics,
    build_render_tasks,
    input_stability_features,
    initialize_render_worker,
    load_cache,
    load_canonical_metrics,
    RenderTask,
    render_one_window,
    resolve_workers,
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


def _synthetic_specs(tmp_path: Path) -> tuple[ComparisonSpec, ...]:
    return tuple(
        ComparisonSpec(
            label=label,
            seed=100 + index,
            checkpoint=tmp_path / f"{label}.pt",
            selection_source="validation_topk_legacy_task_selection",
        )
        for index, label in enumerate(REQUIRED_MODELS)
    )


def _write_synthetic_cache(tmp_path: Path) -> tuple[Path, tuple[ComparisonSpec, ...]]:
    root = tmp_path / "cache"
    predictions_root = root / "predictions"
    root.mkdir()
    specs = _synthetic_specs(tmp_path)
    time = np.arange(1_000) / 100.0
    bcg = np.sin(2 * np.pi * 0.2 * time)[None, :].astype(np.float32)
    tho = np.sin(2 * np.pi * 0.2 * time + 0.1)[None, :].astype(np.float32)
    np.save(root / "dataset_row_id.npy", np.asarray([101], dtype=np.int64))
    np.save(root / "bcg_input.npy", bcg)
    np.save(root / "tho_ref.npy", tho)
    arrays = {
        "dataset_row_id": {"path": str(root / "dataset_row_id.npy")},
        "bcg_input": {"path": str(root / "bcg_input.npy")},
        "tho_ref": {"path": str(root / "tho_ref.npy")},
    }
    for index, spec in enumerate(specs):
        prediction_path = predictions_root / spec.label / "r_tho_hat.npy"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(prediction_path, tho + (index * 0.01))
        arrays[f"prediction:{spec.label}"] = {"path": str(prediction_path)}
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "spec_fingerprint": spec_fingerprint(specs),
                "arrays": arrays,
            }
        ),
        encoding="utf-8",
    )
    return root, specs


def _render_metrics() -> dict[str, dict[str, float]]:
    return {
        label: {
            "pred_rr_peak_band_robust_bpm": 12.0,
            "target_rr_peak_band_robust_bpm": 12.5,
            "rr_peak_band_robust_abs_error": 0.5,
            "breath_count_zero_cross_bpm_error": 0.3,
            "best_lag_corr_4s": 0.8,
            "best_lag_sec_4s": 0.1,
            "relative_envelope_corr_lag4s": 0.7,
            "local_rr_mae": 0.4,
            "local_rr_corr": 0.6,
            "local_rr_valid_frac": 1.0,
        }
        for label in REQUIRED_MODELS
    }


def test_load_cache_and_render_one_window_writes_png(tmp_path: Path) -> None:
    cache_dir, specs = _write_synthetic_cache(tmp_path)

    cache = load_cache(cache_dir, specs)
    initialize_render_worker(cache_dir, specs, fs=100.0, low_hz=0.05, high_hz=0.7, order=4)
    result = render_one_window(
        RenderTask(
            row_index=0,
            dataset_row_id=101,
            output_dir=tmp_path / "figures",
            metrics_by_label=_render_metrics(),
        )
    )

    assert cache.bcg_input.shape == (1, 1_000)
    assert result.status == "written"
    assert result.figure_path.exists()
    assert result.figure_path.suffix == ".png"


def test_resolve_workers_uses_48_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scripts.plot_g_series_comparison.os.cpu_count", lambda: 102)

    assert resolve_workers("auto", n_tasks=2_310) == 48
    assert resolve_workers("auto", n_tasks=7) == 7
    assert resolve_workers(12, n_tasks=100) == 12


def test_build_render_tasks_keeps_only_retained_rows(tmp_path: Path) -> None:
    cache_dir, specs = _write_synthetic_cache(tmp_path)
    cache = load_cache(cache_dir, specs)
    aligned = align_canonical_metrics(
        [101],
        {label: _metric_frame([101]) for label in REQUIRED_MODELS},
    )
    selected = pd.DataFrame(
        {"dataset_row_id": [101], "plot_status": ["retained"]}
    )

    tasks = build_render_tasks(cache, aligned, selected, output_dir=tmp_path / "figures")

    assert len(tasks) == 1
    assert tasks[0].dataset_row_id == 101
    assert set(tasks[0].metrics_by_label) == set(REQUIRED_MODELS)


def test_window_index_sorts_results_and_keeps_failure_record(tmp_path: Path) -> None:
    selected = pd.DataFrame(
        {
            "dataset_row_id": [20, 10, 30],
            "plot_status": ["retained", "retained", "input_stable_excluded"],
        }
    )
    tasks = (
        RenderTask(0, 20, tmp_path, _render_metrics()),
        RenderTask(1, 10, tmp_path, _render_metrics()),
    )
    written = [
        type("Result", (), {"dataset_row_id": 20, "figure_path": tmp_path / "row_20.png"})(),
    ]
    failures = [{"dataset_row_id": 10, "message": "synthetic render failure"}]

    index = _build_window_index(selected, tasks, written, failures)

    assert index.dataset_row_id.tolist() == [10, 20, 30]
    assert index.render_status.tolist() == ["failed", "written", "input_stable_excluded"]
    assert index.loc[index.dataset_row_id == 10, "error"].item() == "synthetic render failure"
