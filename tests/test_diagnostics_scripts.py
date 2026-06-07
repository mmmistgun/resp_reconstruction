from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from scripts.plot_tho_predictions import plot_run_predictions
from scripts.summarize_tho_runs import summarize_runs


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
                "spectrum_similarity": 0.8,
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


def test_plot_run_predictions_writes_png(tmp_path):
    run_dir = tmp_path / "run"
    _write_minimal_run(run_dir)

    written = plot_run_predictions(run_dir, max_plots=1)

    assert len(written) == 1
    assert written[0].suffix == ".png"
    assert written[0].exists()
    assert written[0].stat().st_size > 0


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
