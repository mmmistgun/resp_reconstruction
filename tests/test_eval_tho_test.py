from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

import scripts.eval_tho_test as eval_test
from resp_train.experiments.tho import ThoExperiment


def _prepare_dataset(root: Path) -> None:
    training_dir = root / "training"
    npz_dir = root / "whole_night" / "mixed_bcg_to_tho" / "1"
    training_dir.mkdir(parents=True)
    npz_dir.mkdir(parents=True)
    t = np.linspace(0, 4 * np.pi, 128, dtype=np.float32)
    np.savez(npz_dir / "sample.npz", bcg=np.sin(t).astype(np.float32), tho=np.cos(t).astype(np.float32))
    records = []
    row_id = 1
    for split in ("train", "val", "test"):
        for start in (0, 16, 32, 48):
            records.append(
                {
                    "dataset_row_id": row_id,
                    "input_set": "mixed_zscore",
                    "split": split,
                    "samp_id": 1,
                    "segment_id": 1,
                    "window_id_in_segment": row_id,
                    "source_npz": "../whole_night/mixed_bcg_to_tho/1/sample.npz",
                    "bcg_signal_key": "bcg",
                    "target_signal_key": "tho",
                    "valid_sec_key": "valid",
                    "segment_decision": "include_candidate",
                    "window_start_sample": start,
                    "window_end_sample": start + 32,
                    "window_duration_samples": 32,
                    "target_fs": 100,
                    "valid_ratio": 1.0,
                    "input_finite_ratio": 1.0,
                    "target_finite_ratio": 1.0,
                    "residual_quality_class": (
                        "near_zero_residual" if row_id % 2 else "stable_nonzero_residual"
                    ),
                    "base_alignment_method": "keep_original",
                    "apply_decision": "approved",
                    "reason": "ok",
                }
            )
            row_id += 1
    pd.DataFrame.from_records(records).to_csv(training_dir / "dataset_index.csv", index=False)


def _cfg(tmp_path: Path):
    root = tmp_path / "dataset"
    _prepare_dataset(root)
    return OmegaConf.create(
        {
            "data": {
                "dataset_root": str(root),
                "index_csv": "training/dataset_index.csv",
                "input_set": "mixed_zscore",
                "train_split": "train",
                "val_split": "val",
                "test_split": "test",
                "max_train_windows": 4,
                "max_val_windows": 4,
                "max_test_windows": 3,
                "filter_unusable": True,
                "valid_ratio_min": 0.99,
                "input_finite_ratio_min": 0.99,
                "target_finite_ratio_min": 0.99,
                "unusable_residual_classes": [],
                "preload_windows": True,
                "train_sample_strategy": "stratified_random",
                "val_sample_strategy": "stratified_random",
                "test_sample_strategy": "head",
                "train_sample_seed": 1,
                "val_sample_seed": 2,
                "test_sample_seed": 3,
                "stratify_column": "residual_quality_class",
            },
            "window": {"target_fs": 100, "duration_samples": 32, "duration_sec": 0.32},
            "model": {"name": "unet1d_tiny", "in_channels": 1, "out_channels": 1, "base_channels": 4},
            "loss": {
                "envelope_weight": 1.0,
                "spectrum_weight": 0.2,
                "smooth_weight": 0.001,
                "envelope_window_sec": 0.08,
                "spectrum_low_hz": 0.05,
                "spectrum_high_hz": 5.0,
            },
            "training": {
                "epochs": 1,
                "batch_size": 2,
                "learning_rate": 0.001,
                "num_workers": 0,
                "seed": 1,
                "device": "cpu",
                "patience": 2,
                "min_delta": 0.0,
                "lr_scheduler": "none",
                "grad_clip_norm": None,
                "use_amp": False,
            },
            "baseline": {"bandpass_low_hz": 0.05, "bandpass_high_hz": 5.0, "filter_order": 2},
            "outputs": {"run_root": str(tmp_path / "runs"), "max_prediction_windows": 2},
        }
    )


def test_evaluate_tho_test_checkpoint_writes_test_metrics_summary_and_manifest_without_predictions(tmp_path: Path):
    cfg = _cfg(tmp_path)
    run_dir = ThoExperiment(cfg).train()

    outputs = eval_test.evaluate_tho_test_checkpoint(
        checkpoint_path=run_dir / "checkpoint.pt",
        config_path=run_dir / "config.yaml",
        metrics_output_path=None,
        summary_output_path=None,
        manifest_output_path=None,
    )

    assert outputs.metrics == run_dir / "test_metrics.csv"
    assert outputs.summary == run_dir / "test_summary.csv"
    assert outputs.manifest == run_dir / "test_eval_manifest.csv"
    assert not (run_dir / "test_predictions.npz").exists()
    metrics = pd.read_csv(outputs.metrics)
    assert len(metrics) == cfg.data.max_test_windows
    assert metrics["split"].unique().tolist() == ["test"]
    assert "rr_peak_band_robust_abs_error" in metrics.columns
    summary = pd.read_csv(outputs.summary)
    assert summary["split"].tolist() == ["test"]
    assert "rr_peak_band_robust_abs_error_mean" in summary.columns
    manifest = pd.read_csv(outputs.manifest)
    assert manifest.loc[0, "split"] == "test"
    assert manifest.loc[0, "n_windows"] == cfg.data.max_test_windows
    assert manifest.loc[0, "metrics_output"] == str(outputs.metrics)


def test_summarize_test_metrics_includes_detail_metrics():
    metrics = pd.DataFrame(
        {
            "rr_peak_band_robust_abs_error": [0.2, 0.6],
            "breath_count_zero_cross_abs_error": [1.0, 3.0],
            "relative_envelope_mae_lag4s": [0.1, 0.3],
            "relative_envelope_corr_lag4s": [0.7, 0.9],
            "band_limited_corr": [0.4, 0.8],
            "best_lag_corr_4s": [0.6, 0.9],
            "best_lag_sec_4s": [-2.0, 1.0],
            "local_rr_mae": [0.4, 0.8],
            "local_rr_corr": [0.5, 0.7],
            "local_rr_valid_frac": [1.0, 0.5],
        }
    )

    summary = eval_test.summarize_test_metrics(metrics, split="test", method="model")

    assert summary.loc[0, "relative_envelope_mae_lag4s_mean"] == 0.2
    assert summary.loc[0, "relative_envelope_corr_lag4s_median"] == 0.8
    assert summary.loc[0, "best_lag_corr_4s_mean"] == 0.75
    assert summary.loc[0, "best_lag_sec_4s_median"] == -0.5
    assert summary.loc[0, "local_rr_mae_mean"] == pytest.approx(0.6)
    assert summary.loc[0, "local_rr_corr_median"] == pytest.approx(0.6)
    assert summary.loc[0, "local_rr_valid_frac_mean"] == 0.75


def test_evaluate_predictions_chunked_matches_direct_metrics():
    fs = 100
    t = np.arange(0, 80, 1 / fs)
    target = np.sin(2 * np.pi * 0.25 * t).astype(np.float32)
    pred_a = target.copy()
    pred_b = np.roll(target, 15).astype(np.float32)
    preds = {
        "r_tho_hat": np.stack([pred_a, pred_b, pred_a]).reshape(3, 1, -1),
        "tho_ref": np.stack([target, target, target]).reshape(3, 1, -1),
        "dataset_row_id": np.asarray([1, 2, 3]),
    }
    cfg = OmegaConf.create(
        {
            "window": {"target_fs": fs},
            "loss": {
                "envelope_window_sec": 2.0,
                "spectrum_low_hz": 0.05,
                "spectrum_high_hz": 0.7,
            },
            "evaluation": {"metric_workers": 1},
        }
    )

    direct = eval_test.evaluate_prediction_dict(preds, cfg, method="model")
    target_features = eval_test.build_target_feature_cache(preds, cfg)
    chunked = eval_test.evaluate_predictions_chunked(
        preds,
        cfg,
        method="model",
        metrics_workers=2,
        metrics_chunk_size=1,
        target_features=target_features,
        show_progress=False,
    )

    pd.testing.assert_frame_equal(direct, chunked, check_exact=False, rtol=1e-10, atol=1e-10)


def test_target_feature_cache_round_trips_through_file(tmp_path: Path, monkeypatch):
    fs = 100
    t = np.arange(0, 80, 1 / fs)
    target = np.sin(2 * np.pi * 0.25 * t).astype(np.float32)
    preds = {
        "r_tho_hat": np.stack([target]).reshape(1, 1, -1),
        "tho_ref": np.stack([target]).reshape(1, 1, -1),
        "dataset_row_id": np.asarray([1]),
    }
    cfg = OmegaConf.create(
        {
            "window": {"target_fs": fs},
            "loss": {
                "envelope_window_sec": 2.0,
                "spectrum_low_hz": 0.05,
                "spectrum_high_hz": 0.7,
            },
            "evaluation": {"metric_workers": 1},
        }
    )
    cache_dir = tmp_path / "target_cache"
    first = eval_test.load_or_build_target_feature_cache(preds, cfg, cache_dir=cache_dir, show_progress=False)

    def fail_build(*args, **kwargs):
        raise AssertionError("cache hit should not rebuild target features")

    monkeypatch.setattr(eval_test, "build_target_feature_cache", fail_build)
    second = eval_test.load_or_build_target_feature_cache(preds, cfg, cache_dir=cache_dir, show_progress=False)

    assert list(cache_dir.glob("*.npz"))
    for key, value in first.items():
        np.testing.assert_array_equal(value, second[key])
