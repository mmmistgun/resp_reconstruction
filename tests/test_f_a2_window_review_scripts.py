from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from scripts.plot_paired_f_a2_windows import plot_paired_f_a2_windows
from scripts.summarize_f_a2_stft_ratios import summarize_f_a2_stft_ratios


def _window_list(path: Path, baseline_run_dir: Path, candidate_run_dir: Path) -> Path:
    frame = pd.DataFrame(
        [
            {
                "label": "F-A2b_dist_bandE_w005",
                "seed": 20260700,
                "dataset_row_id": 7,
                "baseline_run_dir": str(baseline_run_dir),
                "candidate_run_dir": str(candidate_run_dir),
                "baseline_rr_peak_band_abs_error": 0.2,
                "candidate_rr_peak_band_abs_error": 0.9,
                "delta_rr_peak_band_abs_error": 0.7,
                "baseline_pred_rr_peak_band_bpm": 14.0,
                "candidate_pred_rr_peak_band_bpm": 11.0,
                "target_rr_peak_band_bpm": 14.0,
                "dirty_easy_lowspec": True,
            }
        ]
    )
    frame.to_csv(path, index=False)
    return path


def test_plot_paired_f_a2_windows_writes_one_png(tmp_path, monkeypatch):
    baseline_run_dir = tmp_path / "f0"
    candidate_run_dir = tmp_path / "fa2b"
    baseline_run_dir.mkdir()
    candidate_run_dir.mkdir()
    window_list = _window_list(tmp_path / "windows.csv", baseline_run_dir, candidate_run_dir)

    def fake_load_config(run_path):
        del run_path
        return OmegaConf.create({"window": {"target_fs": 8}, "training": {"device": "cpu"}})

    def fake_infer(run_path, cfg, row_ids):
        del cfg
        t = np.arange(128, dtype=np.float32) / 8.0
        target = np.sin(2 * np.pi * 1.0 * t)
        pred = np.sin(2 * np.pi * (0.7 if Path(run_path).name == "fa2b" else 1.0) * t)
        return {
            int(row_ids[0]): {
                "pred": pred,
                "target": target,
                "x": None,
                "meta": {"split": "val", "input_set": "test", "residual_quality_class": "clean"},
            }
        }

    monkeypatch.setattr("scripts.plot_paired_f_a2_windows._load_config", fake_load_config)
    monkeypatch.setattr("scripts.plot_paired_f_a2_windows._infer_prediction_lookup", fake_infer)

    written = plot_paired_f_a2_windows(window_list, output_dir=tmp_path / "plots", max_rows=1)

    assert len(written) == 1
    assert written[0].suffix == ".png"
    assert written[0].exists()
    assert written[0].stat().st_size > 0


def test_summarize_f_a2_stft_ratios_reports_harmonic_ratio_delta(tmp_path, monkeypatch):
    baseline_run_dir = tmp_path / "f0"
    candidate_run_dir = tmp_path / "fa2b"
    baseline_run_dir.mkdir()
    candidate_run_dir.mkdir()
    window_list = _window_list(tmp_path / "windows.csv", baseline_run_dir, candidate_run_dir)

    def fake_pair_predictions(row, **kwargs):
        del row, kwargs
        fs = 8.0
        t = np.arange(128, dtype=np.float32) / fs
        target = np.sin(2 * np.pi * 1.0 * t)
        baseline = np.sin(2 * np.pi * 1.0 * t)
        candidate = np.sin(2 * np.pi * 2.0 * t)
        return {"target": target, "baseline": baseline, "candidate": candidate}

    monkeypatch.setattr("scripts.summarize_f_a2_stft_ratios.infer_pair_predictions", fake_pair_predictions)

    ratios = summarize_f_a2_stft_ratios(
        window_list,
        fs=8.0,
        win_length=128,
        hop_length=64,
        n_fft=128,
        bands={"low": (0.5, 1.5), "harm": (1.5, 2.5), "high": (2.5, 3.5)},
        use_cache=False,
    )

    row = ratios.iloc[0]
    assert row["dataset_row_id"] == 7
    assert row["candidate_energy_harm"] > row["baseline_energy_harm"]
    assert row["delta_log_harm_over_low"] > 1.0
