from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from scripts.plot_tho_predictions import _load_input_lookup, _load_npz, plot_run_predictions
from scripts.summarize_tho_runs import summarize_runs


def _patch_plot_inference(monkeypatch, rows: dict[int, dict] | None = None) -> None:
    def fake_infer(run_path, cfg, row_ids):
        del run_path, cfg
        t = np.linspace(0, 2 * np.pi, 128, dtype=np.float32)
        defaults = {
            int(row_id): {
                "pred": np.sin(t),
                "target": np.cos(t),
                "x": None,
                "meta": {"split": "val", "input_set": "mixed_zscore", "residual_quality_class": "near_zero_residual"},
            }
            for row_id in row_ids
        }
        if rows:
            defaults.update(rows)
        return defaults

    monkeypatch.setattr("scripts.plot_tho_predictions._infer_prediction_lookup", fake_infer)


def _write_minimal_run(run_dir: Path, *, val_loss: float = 1.0) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, 2 * np.pi, 128, dtype=np.float32)
    pred = np.sin(t).reshape(1, 1, -1)
    target = np.cos(t).reshape(1, 1, -1)
    np.savez(
        run_dir / "predictions.npz",
        r_tho_hat=pred,
        tho_ref=target,
        dataset_row_id=np.asarray([7]),
        split=np.asarray(["val"]),
        input_set=np.asarray(["mixed_zscore"]),
        residual_quality_class=np.asarray(["near_zero_residual"]),
    )
    pd.DataFrame(
        [
            {
                "method": "unet1d_tiny",
                "dataset_row_id": 7,
                "split": "val",
                "input_set": "mixed_zscore",
                "residual_quality_class": "near_zero_residual",
                "rr_spec_abs_error": 1.5,
                "rr_peak_abs_error": 2.5,
                "envelope_corr": 0.25,
                "relative_envelope_corr": 0.42,
                "relative_envelope_mae": 0.18,
                "spectrum_similarity": 0.8,
                "pred_rr_spec_bpm": 18.0,
                "target_rr_spec_bpm": 16.0,
                "pred_rr_peak_bpm": 19.0,
                "target_rr_peak_bpm": 17.0,
            }
        ]
    ).to_csv(run_dir / "metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "dataset_row_id": 7,
                "rr_spec_abs_error": 2.0,
                "rr_peak_abs_error": 1.0,
                "envelope_corr": 0.5,
                "spectrum_similarity": 0.9,
            }
        ]
    ).to_csv(run_dir / "baseline_metrics.csv", index=False)
    pd.DataFrame([{"epoch": 1, "train_loss": 1.2, "val_loss": val_loss}]).to_csv(
        run_dir / "train_history.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "split": "val",
                "input_set": "mixed_zscore",
                "residual_quality_class": "near_zero_residual",
                "n_windows": 1,
                "n_usable": 1,
                "usable_ratio": 1.0,
            }
        ]
    ).to_csv(run_dir / "audit.csv", index=False)
    OmegaConf.save(
        OmegaConf.create({"window": {"target_fs": 100}, "data": {"dataset_root": "", "index_csv": ""}}),
        run_dir / "config.yaml",
    )


def _write_research_v2_run(run_dir: Path, dataset_root: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    training_dir = dataset_root / "training"
    align_dir = dataset_root / "whole_night" / "alignment" / "88"
    bank_dir = dataset_root / "whole_night" / "signal_bank" / "88"
    training_dir.mkdir(parents=True, exist_ok=True)
    align_dir.mkdir(parents=True, exist_ok=True)
    bank_dir.mkdir(parents=True, exist_ok=True)

    base = np.arange(80, dtype=np.float32)
    np.savez(
        align_dir / "research_v2_alignment.npz",
        bcg_resp_band_state_aligned=base + 100,
    )
    np.savez(
        bank_dir / "research_v2_signal_bank.npz",
        tho_waveform_ref=base * 2,
    )
    pd.DataFrame(
        [
            {
                "dataset_row_id": 1,
                "split": "val",
                "samp_id": 88,
                "coupling_state_id": 1,
                "window_start_s": 0.1,
                "window_end_s": 0.5,
                "source_npz": "../whole_night/alignment/88/research_v2_alignment.npz",
                "target_source_npz": "../whole_night/signal_bank/88/research_v2_signal_bank.npz",
                "bcg_input_key": "bcg_resp_band_state_aligned",
                "bcg_input_aligned_key": "bcg_resp_band_state_aligned",
                "target_waveform_key": "tho_waveform_ref",
                "hard_valid_ratio": 1.0,
                "state_alignment_valid_ratio": 1.0,
                "allowed_losses": "waveform",
                "supervision_confidence_level": "high",
                "state_alignment_method": "constant_shift",
                "reason": "",
            }
        ]
    ).to_csv(training_dir / "dataset_index.csv", index=False)
    np.savez(
        run_dir / "predictions.npz",
        r_tho_hat=np.zeros((1, 1, 40), dtype=np.float32),
        tho_ref=np.zeros((1, 1, 40), dtype=np.float32),
        dataset_row_id=np.asarray([1]),
        split=np.asarray(["val"]),
        input_set=np.asarray(["research_v2_waveform"]),
        residual_quality_class=np.asarray(["waveform"]),
    )
    OmegaConf.save(
        OmegaConf.create(
            {
                "data": {
                    "format": "research_v2",
                    "dataset_root": str(dataset_root),
                    "index_csv": "training/dataset_index.csv",
                    "input_set": "research_v2_waveform",
                    "target_task": "waveform",
                    "bcg_input_key": "bcg_input_aligned_key",
                    "target_key": "target_waveform_key",
                    "filter_unusable": True,
                    "preload_windows": False,
                    "min_hard_valid_ratio": 0.8,
                    "min_state_alignment_valid_ratio": 0.8,
                },
                "window": {"target_fs": 100, "duration_samples": 40},
            }
        ),
        run_dir / "config.yaml",
    )


def test_plot_run_predictions_writes_png(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    _write_minimal_run(run_dir)
    _patch_plot_inference(monkeypatch)

    written = plot_run_predictions(run_dir, max_plots=1)

    assert len(written) == 1
    assert written[0].suffix == ".png"
    assert written[0].exists()
    assert written[0].stat().st_size > 0


def test_plot_run_predictions_infers_metric_selected_rows(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    _write_minimal_run(run_dir)
    metrics = pd.read_csv(run_dir / "metrics.csv")
    metrics.loc[len(metrics)] = {
        **metrics.iloc[0].to_dict(),
        "dataset_row_id": 9,
        "relative_envelope_mae": 0.99,
    }
    metrics.to_csv(run_dir / "metrics.csv", index=False)

    def fake_infer(run_path, cfg, row_ids):
        assert run_path == run_dir
        assert row_ids == [9]
        t = np.linspace(0, 2 * np.pi, 128, dtype=np.float32)
        return {
            9: {
                "pred": np.sin(t),
                "target": np.cos(t),
                "meta": {"split": "val", "input_set": "mixed_zscore", "residual_quality_class": "near_zero_residual"},
            }
        }

    monkeypatch.setattr("scripts.plot_tho_predictions._infer_prediction_lookup", fake_infer)

    written = plot_run_predictions(run_dir, max_plots=1, sort_by="relative_envelope_mae")

    assert written[0].name.endswith("_row_9.png")


def test_plot_run_predictions_requires_checkpoint(tmp_path):
    run_dir = tmp_path / "run"
    _write_minimal_run(run_dir)

    with pytest.raises(FileNotFoundError, match="checkpoint.pt"):
        plot_run_predictions(run_dir, max_plots=1)


def test_plot_run_predictions_writes_diagnostic_four_panel_png(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    _write_minimal_run(run_dir)
    _patch_plot_inference(monkeypatch)

    written = plot_run_predictions(run_dir, max_plots=1)

    from PIL import Image

    width, height = Image.open(written[0]).size
    assert width >= 1800
    assert height >= 1400


def test_plot_title_includes_rr_metric_values(tmp_path):
    run_dir = tmp_path / "run"
    _write_minimal_run(run_dir)
    predictions = _load_npz(run_dir / "predictions.npz")
    metrics = pd.read_csv(run_dir / "metrics.csv")

    from scripts.plot_tho_predictions import _metric_text

    text = _metric_text(_metric_row=metrics.iloc[0].to_dict(), predictions=predictions, pred_idx=0)

    assert "rr_spec=18.0/16.0 bpm" in text
    assert "rr_peak=19.0/17.0 bpm" in text


def test_metric_text_includes_relative_envelope_metrics():
    from scripts.plot_tho_predictions import _metric_text

    text = _metric_text(
        _metric_row={
            "relative_envelope_corr": 0.42,
            "relative_envelope_mae": 0.18,
        },
        predictions={},
        pred_idx=0,
    )

    assert "relative_envelope_corr=0.420" in text
    assert "relative_envelope_mae=0.180" in text


def test_plot_input_lookup_supports_research_v2_format(tmp_path):
    run_dir = tmp_path / "run"
    dataset_root = tmp_path / "research_v2_dataset"
    _write_research_v2_run(run_dir, dataset_root)
    predictions = _load_npz(run_dir / "predictions.npz")
    cfg = OmegaConf.load(run_dir / "config.yaml")

    lookup = _load_input_lookup(run_dir, predictions, cfg)

    assert lookup[1].tolist() == np.arange(110, 150, dtype=np.float32).tolist()


def test_summarize_runs_writes_one_row_per_run(tmp_path):
    root = tmp_path / "runs"
    _write_minimal_run(root / "run_a", val_loss=1.0)
    _write_minimal_run(root / "run_b", val_loss=0.8)
    output = tmp_path / "summary.csv"

    frame = summarize_runs(root, output)

    assert output.exists()
    assert frame["run_id"].tolist() == ["run_a", "run_b"]
    assert frame["val_loss"].tolist() == [1.0, 0.8]
    assert frame["model_rr_spec_abs_error_mean"].tolist() == [1.5, 1.5]
    assert frame["baseline_spectrum_similarity_mean"].tolist() == [0.9, 0.9]


def test_summarize_runs_includes_relative_envelope_metrics(tmp_path):
    root = tmp_path / "runs"
    _write_minimal_run(root / "run_a")
    output = tmp_path / "summary.csv"

    frame = summarize_runs(root, output)

    assert frame["model_relative_envelope_corr_mean"].tolist() == [0.42]
    assert frame["model_relative_envelope_mae_median"].tolist() == [0.18]


def test_train_script_delegates_to_tho_experiment():
    source = Path("scripts/train_tho_small.py").read_text(encoding="utf-8")

    assert "ThoExperiment" in source
    assert "train_one_epoch" not in source
    assert "run_baseline" not in source


def test_eval_script_delegates_to_tho_checkpoint_evaluator():
    source = Path("scripts/eval_tho_small.py").read_text(encoding="utf-8")

    assert "evaluate_tho_checkpoint" in source
    assert "RespWindowDataset" not in source
    assert "filter_index" not in source


def test_audit_and_baseline_scripts_use_data_factory():
    audit_source = Path("scripts/audit_tho_dataset.py").read_text(encoding="utf-8")
    baseline_source = Path("scripts/baseline_tho_hilbert.py").read_text(encoding="utf-8")

    assert "build_tho_data" in audit_source
    assert "build_window_data" in baseline_source or "build_tho_data" in baseline_source
