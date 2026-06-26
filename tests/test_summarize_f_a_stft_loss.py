from pathlib import Path

import pandas as pd

from scripts.summarize_f_a_stft_loss import summarize_f_a_runs


def _write_run(
    root: Path,
    name: str,
    rows: list[dict],
    train_loss: float = 1.0,
    seed: int | None = None,
) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(run_dir / "metrics.csv", index=False)
    pd.DataFrame({"epoch": [1], "train_loss": [train_loss], "val_loss": [train_loss + 0.1]}).to_csv(
        run_dir / "train_history.csv",
        index=False,
    )
    if seed is not None:
        (run_dir / "config.yaml").write_text(f"training:\n  seed: {seed}\n", encoding="utf-8")
    return run_dir


def _metrics(errors: list[float], spectrum: list[float], target_rr: list[float]) -> list[dict]:
    rows = []
    for idx, error in enumerate(errors):
        rows.append(
            {
                "dataset_row_id": idx,
                "rr_peak_band_abs_error": error,
                "rr_spec_abs_error": 0.2 + 0.1 * idx,
                "breath_count_zero_cross_abs_error": idx % 2,
                "relative_envelope_mae": 0.2,
                "relative_envelope_corr": 0.4,
                "spectrum_similarity": spectrum[idx],
                "band_limited_corr": 0.7,
                "best_lag_corr": 0.8,
                "best_lag_sec": 0.1,
                "target_rr_peak_band_bpm": target_rr[idx],
            }
        )
    return rows


def test_summarize_f_a_runs_outputs_detail_pair_delta_and_strata(tmp_path):
    f0_root = tmp_path / "f0"
    fa_root = tmp_path / "fa"
    _write_run(f0_root, "20260626_000000", _metrics([0.1, 1.4, 2.2, 0.3], [0.99, 0.7, 0.6, 0.95], [14, 16, 20, 22]))
    _write_run(fa_root, "20260626_000001", _metrics([0.1, 0.9, 1.6, 0.4], [0.99, 0.7, 0.6, 0.95], [14, 16, 20, 22]))
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "tag": "f0_native_stft_pre_mixer_dual_20260700",
                "label": "F0_native_stft_pre_mixer",
                "branch_mode": "dual",
                "seed": 20260700,
                "paired_f0_label": "F0_native_stft_pre_mixer",
                "paired_time_only_label": "F0_native_time_only",
                "overrides": f"outputs.run_root={f0_root}",
            },
            {
                "tag": "f_a0_dist_dual_20260700",
                "label": "F-A0_dist",
                "branch_mode": "dual",
                "seed": 20260700,
                "paired_f0_label": "F0_native_stft_pre_mixer",
                "paired_time_only_label": "F0_native_time_only",
                "overrides": f"outputs.run_root={fa_root}",
            },
        ]
    ).to_csv(manifest, index=False)

    detail, paired, strata = summarize_f_a_runs(manifest)

    assert set(detail["label"]) == {"F0_native_stft_pre_mixer", "F-A0_dist"}
    delta_row = paired.iloc[0]
    assert delta_row["label"] == "F-A0_dist"
    assert delta_row["delta_rr_peak_band_abs_error_mean"] < 0
    assert delta_row["delta_frac_gt_1"] < 0
    assert {"baseline_hard", "baseline_easy", "low_spectrum", "fast_rr"} <= set(strata["stratum"])


def test_summarize_f_a_runs_matches_run_dir_by_training_seed(tmp_path):
    f0_root = tmp_path / "f0"
    fa_root = tmp_path / "fa"
    _write_run(
        f0_root,
        "20260626_000000",
        _metrics([1.0], [0.9], [14]),
        train_loss=1.0,
        seed=20260700,
    )
    _write_run(
        f0_root,
        "20260626_000001",
        _metrics([2.0], [0.9], [14]),
        train_loss=2.0,
        seed=20260837,
    )
    _write_run(
        fa_root,
        "20260626_000002",
        _metrics([0.5], [0.9], [14]),
        train_loss=1.0,
        seed=20260700,
    )
    _write_run(
        fa_root,
        "20260626_000003",
        _metrics([3.0], [0.9], [14]),
        train_loss=2.0,
        seed=20260837,
    )
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "tag": "f0_native_stft_pre_mixer_dual_20260700",
                "label": "F0_native_stft_pre_mixer",
                "branch_mode": "dual",
                "seed": 20260700,
                "paired_f0_label": "F0_native_stft_pre_mixer",
                "paired_time_only_label": "F0_native_time_only",
                "overrides": f"outputs.run_root={f0_root}",
            },
            {
                "tag": "f0_native_stft_pre_mixer_dual_20260837",
                "label": "F0_native_stft_pre_mixer",
                "branch_mode": "dual",
                "seed": 20260837,
                "paired_f0_label": "F0_native_stft_pre_mixer",
                "paired_time_only_label": "F0_native_time_only",
                "overrides": f"outputs.run_root={f0_root}",
            },
            {
                "tag": "f_a0_dist_dual_20260700",
                "label": "F-A0_dist",
                "branch_mode": "dual",
                "seed": 20260700,
                "paired_f0_label": "F0_native_stft_pre_mixer",
                "paired_time_only_label": "F0_native_time_only",
                "overrides": f"outputs.run_root={fa_root}",
            },
            {
                "tag": "f_a0_dist_dual_20260837",
                "label": "F-A0_dist",
                "branch_mode": "dual",
                "seed": 20260837,
                "paired_f0_label": "F0_native_stft_pre_mixer",
                "paired_time_only_label": "F0_native_time_only",
                "overrides": f"outputs.run_root={fa_root}",
            },
        ]
    ).to_csv(manifest, index=False)

    detail, paired, _ = summarize_f_a_runs(manifest)

    by_tag = detail.set_index("tag")
    assert by_tag.loc["f0_native_stft_pre_mixer_dual_20260700", "rr_peak_band_abs_error_mean"] == 1.0
    assert by_tag.loc["f0_native_stft_pre_mixer_dual_20260837", "rr_peak_band_abs_error_mean"] == 2.0
    paired_by_seed = paired.set_index("seed")
    assert paired_by_seed.loc[20260700, "delta_rr_peak_band_abs_error_mean"] == -0.5
    assert paired_by_seed.loc[20260837, "delta_rr_peak_band_abs_error_mean"] == 1.0
