from pathlib import Path
import subprocess
import sys

import pandas as pd

from scripts.diagnose_f_a2_guard_windows import diagnose_f_a2_guard_windows


def _write_run(root: Path, name: str, rows: list[dict], seed: int) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(run_dir / "metrics.csv", index=False)
    (run_dir / "config.yaml").write_text(f"training:\n  seed: {seed}\n", encoding="utf-8")
    return run_dir


def _metric_row(
    row_id: int,
    *,
    peak_error: float,
    pred_rr: float,
    target_rr: float,
    spectrum: float,
    count_error: float,
    lag_sec: float = 0.0,
    spec_error: float = 0.0,
) -> dict:
    return {
        "dataset_row_id": row_id,
        "rr_peak_band_abs_error": peak_error,
        "rr_spec_abs_error": spec_error,
        "breath_count_zero_cross_abs_error": count_error,
        "relative_envelope_mae": 0.2,
        "relative_envelope_corr": 0.5,
        "spectrum_similarity": spectrum,
        "band_limited_corr": 0.7,
        "best_lag_corr": 0.8,
        "best_lag_sec": lag_sec,
        "pred_rr_peak_band_bpm": pred_rr,
        "target_rr_peak_band_bpm": target_rr,
    }


def _manifest(tmp_path: Path, f0_root: Path, fa_root: Path) -> Path:
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "tag": "f0_dual_20260700",
                "label": "F0_native_stft_pre_mixer",
                "branch_mode": "dual",
                "seed": 20260700,
                "paired_f0_label": "F0_native_stft_pre_mixer",
                "paired_time_only_label": "F0_native_time_only",
                "overrides": f"outputs.run_root={f0_root}",
            },
            {
                "tag": "f_a2b_dual_20260700",
                "label": "F-A2b_dist_bandE_w005",
                "branch_mode": "dual",
                "seed": 20260700,
                "paired_f0_label": "F0_native_stft_pre_mixer",
                "paired_time_only_label": "F0_native_time_only",
                "overrides": f"outputs.run_root={fa_root}",
            },
        ]
    ).to_csv(manifest, index=False)
    return manifest


def test_diagnose_f_a2_guard_windows_outputs_window_delta_and_buckets(tmp_path):
    f0_root = tmp_path / "f0"
    fa_root = tmp_path / "fa"
    _write_run(
        f0_root,
        "20260627_000000",
        [
            _metric_row(1, peak_error=0.1, pred_rr=14.1, target_rr=14.0, spectrum=0.99, count_error=0),
            _metric_row(2, peak_error=0.2, pred_rr=13.8, target_rr=14.0, spectrum=0.40, count_error=2),
            _metric_row(3, peak_error=1.6, pred_rr=18.0, target_rr=20.0, spectrum=0.60, count_error=3),
            _metric_row(4, peak_error=0.8, pred_rr=21.0, target_rr=21.5, spectrum=0.80, count_error=1),
        ],
        seed=20260700,
    )
    _write_run(
        fa_root,
        "20260627_000001",
        [
            _metric_row(1, peak_error=0.12, pred_rr=14.2, target_rr=14.0, spectrum=0.98, count_error=0),
            _metric_row(2, peak_error=0.9, pred_rr=13.0, target_rr=14.0, spectrum=0.41, count_error=1),
            _metric_row(3, peak_error=0.7, pred_rr=19.3, target_rr=20.0, spectrum=0.61, count_error=1),
            _metric_row(4, peak_error=0.6, pred_rr=21.2, target_rr=21.5, spectrum=0.82, count_error=1),
        ],
        seed=20260700,
    )

    window_delta, bucket_summary, top_degraded_easy, top_improved_hard = diagnose_f_a2_guard_windows(
        _manifest(tmp_path, f0_root, fa_root),
        candidate_labels=["F-A2b_dist_bandE_w005"],
        top_n=2,
    )

    by_row = window_delta.set_index("dataset_row_id")
    assert by_row.loc[1, "clean_easy_highspec"]
    assert by_row.loc[2, "dirty_easy_lowspec"]
    assert by_row.loc[2, "delta_rr_peak_band_abs_error"] == 0.7
    assert by_row.loc[3, "baseline_hard"]
    assert by_row.loc[3, "delta_rr_peak_band_abs_error"] == -0.9
    assert by_row.loc[4, "fast_rr"]

    dirty = bucket_summary[
        (bucket_summary["bucket_type"] == "diagnosis_stratum")
        & (bucket_summary["bucket"] == "dirty_easy_lowspec")
    ].iloc[0]
    assert dirty["n_windows"] == 1
    assert dirty["mean_delta_rr_peak_band_abs_error"] == 0.7

    assert top_degraded_easy.iloc[0]["dataset_row_id"] == 2
    assert top_improved_hard.iloc[0]["dataset_row_id"] == 3


def test_diagnose_f_a2_guard_windows_writes_all_outputs(tmp_path):
    f0_root = tmp_path / "f0"
    fa_root = tmp_path / "fa"
    _write_run(
        f0_root,
        "20260627_000000",
        [_metric_row(1, peak_error=0.2, pred_rr=14.2, target_rr=14.0, spectrum=0.9, count_error=0)],
        seed=20260700,
    )
    _write_run(
        fa_root,
        "20260627_000001",
        [_metric_row(1, peak_error=0.5, pred_rr=14.5, target_rr=14.0, spectrum=0.9, count_error=0)],
        seed=20260700,
    )
    output_dir = tmp_path / "diag"

    diagnose_f_a2_guard_windows(
        _manifest(tmp_path, f0_root, fa_root),
        candidate_labels=["F-A2b_dist_bandE_w005"],
        output_dir=output_dir,
    )

    assert (output_dir / "window_delta.csv").exists()
    assert (output_dir / "bucket_summary.csv").exists()
    assert (output_dir / "top_degraded_easy.csv").exists()
    assert (output_dir / "top_improved_hard.csv").exists()


def test_diagnose_f_a2_guard_windows_script_runs_as_file(tmp_path):
    f0_root = tmp_path / "f0"
    fa_root = tmp_path / "fa"
    _write_run(
        f0_root,
        "20260627_000000",
        [_metric_row(1, peak_error=0.2, pred_rr=14.2, target_rr=14.0, spectrum=0.9, count_error=0)],
        seed=20260700,
    )
    _write_run(
        fa_root,
        "20260627_000001",
        [_metric_row(1, peak_error=0.5, pred_rr=14.5, target_rr=14.0, spectrum=0.9, count_error=0)],
        seed=20260700,
    )
    output_dir = tmp_path / "diag"
    repo_root = Path(__file__).resolve().parents[1]

    subprocess.run(
        [
            sys.executable,
            "scripts/diagnose_f_a2_guard_windows.py",
            "--manifest",
            str(_manifest(tmp_path, f0_root, fa_root)),
            "--candidate-label",
            "F-A2b_dist_bandE_w005",
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
        check=True,
    )

    assert (output_dir / "window_delta.csv").exists()
