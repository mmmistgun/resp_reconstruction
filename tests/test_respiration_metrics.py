from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from resp_train.config import load_config
from resp_train.metrics.task import (
    evaluate_task_predictions,
    summarize_task_metrics,
    validation_local_rr_mean,
)


def _config():
    return load_config("configs/tho_research_v2.yaml")


def _waveform() -> np.ndarray:
    time = np.arange(18000, dtype=np.float64) / 100.0
    amplitude = 1.0 + 0.3 * np.sin(2.0 * np.pi * 0.01 * time)
    return amplitude * np.sin(2.0 * np.pi * (0.18 * time + 0.00005 * time**2))


def _prediction_dict(prediction: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "r_tho_hat": prediction[None, None, :].astype(np.float32),
        "tho_ref": target[None, None, :].astype(np.float32),
        "dataset_row_id": np.asarray([7], dtype=np.int64),
    }


def test_identity_has_ideal_primary_ibi_and_test_only_metrics() -> None:
    target = _waveform()
    metrics = evaluate_task_predictions(
        _prediction_dict(target, target),
        _config(),
        include_test_only=True,
    )
    row = metrics.iloc[0]
    assert row["whole_rr_abs_error_bpm"] == pytest.approx(0.0)
    assert row["local_rr_mae_bpm"] == pytest.approx(0.0)
    assert row["global_effort_spearman"] == pytest.approx(1.0)
    assert row["local_effort_spearman"] == pytest.approx(1.0)
    assert row["lag_aware_signed_pcc"] == pytest.approx(1.0)
    assert row["best_lag_samples"] == 0
    assert row["ibi_medae_sec"] == pytest.approx(0.0)
    assert row["ibi_coverage"] == pytest.approx(1.0)
    assert bool(row["ibi_interpretable"])
    assert row["respiratory_band_coherence"] == pytest.approx(1.0)
    assert row["constrained_ndtw"] == pytest.approx(0.0)
    summary = summarize_task_metrics(metrics).iloc[0]
    assert summary["n_samples"] == 1
    assert summary["lag_aware_signed_pcc_mean"] == pytest.approx(1.0)


def test_constant_prediction_cannot_escape_primary_metrics() -> None:
    target = _waveform()
    prediction = np.zeros_like(target)
    metrics = evaluate_task_predictions(
        _prediction_dict(prediction, target),
        _config(),
        include_test_only=True,
    )
    row = metrics.iloc[0]
    assert row["whole_rr_abs_error_bpm"] == pytest.approx(39.0)
    assert row["local_rr_mae_bpm"] == pytest.approx(39.0)
    assert row["local_rr_prediction_valid_fraction"] == pytest.approx(0.0)
    assert row["global_effort_spearman"] == pytest.approx(-1.0)
    assert row["local_effort_spearman"] == pytest.approx(-1.0)
    assert row["lag_aware_signed_pcc"] == pytest.approx(-1.0)
    assert row["ibi_coverage"] == pytest.approx(0.0)
    assert np.isnan(row["ibi_medae_sec"])
    assert row["respiratory_band_coherence"] == pytest.approx(0.0)
    assert np.isfinite(row["constrained_ndtw"])


def test_validation_selector_only_returns_local_rr_mean() -> None:
    target = _waveform()
    assert validation_local_rr_mean(_prediction_dict(target, target), _config()) == pytest.approx(0.0)


def test_positive_lag_means_prediction_is_delayed() -> None:
    target = _waveform()
    prediction = np.zeros_like(target)
    prediction[20:] = target[:-20]
    row = evaluate_task_predictions(_prediction_dict(prediction, target), _config()).iloc[0]
    assert row["best_lag_samples"] == 20
    assert row["lag_aware_signed_pcc"] > 0.999


def test_metrics_reject_nonfinite_checkpoint_output() -> None:
    target = _waveform()
    prediction = target.copy()
    prediction[0] = np.inf
    with pytest.raises(FloatingPointError, match="NaN/Inf"):
        evaluate_task_predictions(_prediction_dict(prediction, target), _config())


def test_signed_pcc_recovers_positive_prediction_delay() -> None:
    target = _waveform()
    prediction = np.roll(target, 20)
    metrics = evaluate_task_predictions(_prediction_dict(prediction, target), _config())
    row = metrics.iloc[0]
    assert row["best_lag_samples"] == 20
    assert row["lag_aware_signed_pcc"] == pytest.approx(1.0, abs=1e-6)


def test_test_only_columns_are_absent_during_validation() -> None:
    target = _waveform()
    metrics = evaluate_task_predictions(_prediction_dict(target, target), _config())
    assert "respiratory_band_coherence" not in metrics
    assert "constrained_ndtw" not in metrics


def test_summary_uses_direct_sample_means_and_eligible_diagnostic_denominators() -> None:
    metrics = pd.DataFrame(
        {
            "ibi_coverage": [0.0, 1.0, np.nan],
            "ibi_interpretable": [False, True, False],
            "ibi_target_eligible": [True, True, False],
            "best_lag_sec": [0.30, 0.10, 0.0],
            "joint_target_eligible": [True, True, False],
            "joint_prediction_degenerate": [False, True, False],
        }
    )
    summary = summarize_task_metrics(metrics).iloc[0]
    assert summary["ibi_coverage_mean"] == pytest.approx(0.5)
    assert summary["ibi_interpretable_fraction"] == pytest.approx(0.5)
    assert summary["best_abs_lag_median_sec"] == pytest.approx(0.30)
    assert summary["joint_target_eligible_fraction"] == pytest.approx(2.0 / 3.0)
    assert summary["joint_prediction_degenerate_fraction"] == pytest.approx(0.5)
