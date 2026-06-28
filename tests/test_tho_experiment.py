from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from resp_train.experiments.tho import ThoExperiment, evaluate_tho_checkpoint


def _prepare_dataset(root: Path):
    training_dir = root / "training"
    npz_dir = root / "whole_night" / "mixed_bcg_to_tho" / "1"
    training_dir.mkdir(parents=True)
    npz_dir.mkdir(parents=True)
    t = np.linspace(0, 4 * np.pi, 128, dtype=np.float32)
    np.savez(npz_dir / "sample.npz", bcg=np.sin(t).astype(np.float32), tho=np.cos(t).astype(np.float32))
    records = []
    row_id = 1
    for split in ("train", "val"):
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
                "max_train_windows": 4,
                "max_val_windows": 4,
                "filter_unusable": True,
                "valid_ratio_min": 0.99,
                "input_finite_ratio_min": 0.99,
                "target_finite_ratio_min": 0.99,
                "unusable_residual_classes": [],
                "preload_windows": True,
                "train_sample_strategy": "stratified_random",
                "val_sample_strategy": "stratified_random",
                "train_sample_seed": 1,
                "val_sample_seed": 2,
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


def test_tho_experiment_smoke_writes_run_outputs(tmp_path: Path):
    run_dir = ThoExperiment(_cfg(tmp_path)).train()

    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "audit.csv").exists()
    assert (run_dir / "baseline_metrics.csv").exists()
    assert (run_dir / "checkpoint.pt").exists()
    assert (run_dir / "train_history.csv").exists()
    assert (run_dir / "metrics.csv").exists()
    assert not (run_dir / "predictions.npz").exists()


def test_eval_checkpoint_with_metrics_writes_only_metrics(tmp_path: Path):
    cfg = _cfg(tmp_path)
    run_dir = ThoExperiment(cfg).train()
    metrics_output = tmp_path / "eval_metrics.csv"

    evaluate_tho_checkpoint(
        checkpoint_path=run_dir / "checkpoint.pt",
        config_path=run_dir / "config.yaml",
        metrics_output_path=metrics_output,
    )

    metrics = pd.read_csv(metrics_output)
    assert len(metrics) == cfg.data.max_val_windows
    assert not (tmp_path / "eval_predictions.npz").exists()


def test_eval_checkpoint_with_metrics_does_not_build_train_bundle(monkeypatch, tmp_path: Path):
    cfg = _cfg(tmp_path)
    run_dir = ThoExperiment(cfg).train()
    metrics_output = tmp_path / "eval_metrics.csv"

    def fail_build_tho_data(*args, **kwargs):
        raise AssertionError("checkpoint eval 不应构建 train+val 全量数据")

    monkeypatch.setattr("resp_train.experiments.tho.build_tho_data", fail_build_tho_data)

    evaluate_tho_checkpoint(
        checkpoint_path=run_dir / "checkpoint.pt",
        config_path=run_dir / "config.yaml",
        metrics_output_path=metrics_output,
    )

    metrics = pd.read_csv(metrics_output)
    assert len(metrics) == cfg.data.max_val_windows


def test_tho_experiment_logs_final_metric_summary(monkeypatch, tmp_path: Path):
    cfg = _cfg(tmp_path)
    cfg.training.show_progress = True
    seen_show_progress = []

    def fake_evaluate_prediction_dict(preds, cfg_arg, *, method, show_progress=None):
        seen_show_progress.append(show_progress)
        return pd.DataFrame(
            {
                "rr_peak_band_abs_error": [0.2, 1.2, 2.5],
                "envelope_corr": [0.5, 0.6, 0.7],
            }
        )

    monkeypatch.setattr("resp_train.experiments.tho.evaluate_prediction_dict", fake_evaluate_prediction_dict)

    run_dir = ThoExperiment(cfg).train()
    log_text = (run_dir / "train.log").read_text(encoding="utf-8")

    assert seen_show_progress == [True]
    assert "metrics: n=3 rr_peak_band_abs_error mean=1.300000 median=1.200000 p95=2.370000 frac_gt_1=0.666667" in log_text


def test_run_baseline_skips_when_disabled(monkeypatch, tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cfg = _cfg(tmp_path)
    cfg.baseline.enabled = False
    data = ThoExperiment(cfg).build_data()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("baseline.enabled=false 时不应计算 baseline")

    monkeypatch.setattr("resp_train.experiments.tho.evaluate_baseline_dataset", fail_if_called)

    ThoExperiment(cfg).run_baseline(data, run_dir)

    assert not (run_dir / "baseline_metrics.csv").exists()
