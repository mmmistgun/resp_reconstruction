import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.stratified_eval_analysis import (
    Comparison,
    StratifiedAnalysisSpec,
    run_analysis,
    run_external_strata_analysis,
)


def _metrics_rows(*, robust_shift: float = 0.0, count_shift: float = 0.0) -> list[dict]:
    base_rows = [
        {
            "dataset_row_id": 1,
            "rr_peak_band_robust_abs_error": 0.10,
            "breath_count_zero_cross_abs_error": 0.0,
            "spectrum_similarity": 0.90,
            "target_rr_peak_band_robust_bpm": 8.0,
        },
        {
            "dataset_row_id": 2,
            "rr_peak_band_robust_abs_error": 1.50,
            "breath_count_zero_cross_abs_error": 1.0,
            "spectrum_similarity": 0.40,
            "target_rr_peak_band_robust_bpm": 12.0,
        },
        {
            "dataset_row_id": 3,
            "rr_peak_band_robust_abs_error": 0.70,
            "breath_count_zero_cross_abs_error": 2.0,
            "spectrum_similarity": 0.20,
            "target_rr_peak_band_robust_bpm": 16.0,
        },
        {
            "dataset_row_id": 4,
            "rr_peak_band_robust_abs_error": 0.30,
            "breath_count_zero_cross_abs_error": 0.0,
            "spectrum_similarity": 0.80,
            "target_rr_peak_band_robust_bpm": 20.0,
        },
        {
            "dataset_row_id": 5,
            "rr_peak_band_robust_abs_error": 2.00,
            "breath_count_zero_cross_abs_error": 3.0,
            "spectrum_similarity": 0.30,
            "target_rr_peak_band_robust_bpm": 28.0,
        },
    ]
    rows = []
    for row in base_rows:
        updated = dict(row)
        updated["rr_peak_band_robust_abs_error"] = max(
            0.0,
            updated["rr_peak_band_robust_abs_error"] + robust_shift,
        )
        updated["breath_count_zero_cross_abs_error"] = max(
            0.0,
            updated["breath_count_zero_cross_abs_error"] + count_shift,
        )
        updated.update(
            {
                "best_lag_corr_4s": 0.80 - 0.01 * updated["dataset_row_id"],
                "relative_envelope_corr_lag4s": 0.50 - 0.01 * updated["dataset_row_id"],
                "local_rr_mae": 1.00 + 0.10 * updated["dataset_row_id"],
                "local_rr_corr": 0.40 + 0.02 * updated["dataset_row_id"],
                "rr_peak_band_abs_error": updated["rr_peak_band_robust_abs_error"] + 0.05,
                "rr_spec_abs_error": updated["rr_peak_band_robust_abs_error"] + 0.10,
            }
        )
        rows.append(updated)
    return rows


def _write_eval_metrics(root: Path, label: str, seed: int, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(root / f"{label}_{seed}_test_metrics.csv", index=False)


def _prepare_eval_root(tmp_path: Path) -> tuple[Path, Path]:
    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    for seed in (101, 102):
        _write_eval_metrics(eval_root, "time", seed, _metrics_rows())
        wide = _metrics_rows()
        wide[0]["rr_peak_band_robust_abs_error"] += 0.05
        wide[1]["rr_peak_band_robust_abs_error"] -= 0.60
        wide[2]["rr_peak_band_robust_abs_error"] -= 0.20
        wide[4]["rr_peak_band_robust_abs_error"] -= 0.50
        wide[1]["breath_count_zero_cross_abs_error"] -= 1.0
        wide[2]["breath_count_zero_cross_abs_error"] -= 1.0
        wide[4]["breath_count_zero_cross_abs_error"] -= 2.0
        _write_eval_metrics(eval_root, "wide", seed, wide)

        bandenergy = [dict(row) for row in wide]
        bandenergy[0]["rr_peak_band_robust_abs_error"] += 0.08
        bandenergy[1]["rr_peak_band_robust_abs_error"] -= 0.25
        bandenergy[4]["rr_peak_band_robust_abs_error"] -= 0.20
        bandenergy[2]["breath_count_zero_cross_abs_error"] -= 1.0
        bandenergy[4]["breath_count_zero_cross_abs_error"] -= 1.0
        _write_eval_metrics(eval_root, "bandenergy", seed, bandenergy)

    dataset_index = tmp_path / "dataset_index.csv"
    pd.DataFrame(
        {
            "dataset_row_id": [1, 2, 3, 4, 5],
            "samp_id": [220, 220, 671, 671, 1006],
            "window_start_s": [0, 30, 60, 90, 120],
            "allowed_losses": [
                "rate;waveform",
                "rate;phase;event;waveform",
                "rate;waveform",
                "rate;phase;event;waveform",
                "rate;waveform",
            ],
        }
    ).to_csv(dataset_index, index=False)
    return eval_root, dataset_index


def test_run_analysis_writes_reusable_strata_tail_rr_and_subject_outputs(tmp_path):
    eval_root, dataset_index = _prepare_eval_root(tmp_path)
    output_dir = tmp_path / "analysis"

    outputs = run_analysis(
        StratifiedAnalysisSpec(
            eval_root=eval_root,
            output_dir=output_dir,
            labels=["time", "wide", "bandenergy"],
            seeds=[101, 102],
            comparisons=[
                Comparison(name="wide_vs_time", target="wide", baseline="time"),
                Comparison(name="bandenergy_vs_wide", target="bandenergy", baseline="wide"),
            ],
            dataset_index=dataset_index,
        )
    )

    expected_files = {
        "strata_seed_summary.csv",
        "strata_summary.csv",
        "tail_seed_summary.csv",
        "tail_summary.csv",
        "rr_bin_seed_summary.csv",
        "rr_bin_summary.csv",
        "subject_seed_summary.csv",
        "subject_summary.csv",
        "analysis_manifest.json",
    }
    assert expected_files <= {path.name for path in outputs.values()}

    strata = pd.read_csv(output_dir / "strata_summary.csv")
    assert {
        "baseline_hard_robust_gt1",
        "baseline_easy_robust_le025",
        "low_spectrum_le_median",
        "baseline_count_error_gt0",
        "baseline_count_error_eq0",
    } <= set(strata["stratum"])
    hard = strata[
        (strata["comparison"] == "bandenergy_vs_wide")
        & (strata["stratum"] == "baseline_hard_robust_gt1")
    ].iloc[0]
    assert hard["delta_rr_peak_band_robust_abs_error_mean"] < 0
    count_error = strata[
        (strata["comparison"] == "wide_vs_time")
        & (strata["stratum"] == "baseline_count_error_gt0")
    ].iloc[0]
    assert "baseline_breath_count_zero_cross_abs_error_mean" not in strata.columns
    assert count_error["baseline_breath_count_zero_cross_bpm_error_mean"] == pytest.approx(
        2.0 / 3.0
    )
    assert count_error["target_breath_count_zero_cross_bpm_error_mean"] == pytest.approx(
        2.0 / 9.0
    )

    rr_bins = pd.read_csv(output_dir / "rr_bin_summary.csv")
    assert {"rr_lt10", "rr_10_14", "rr_14_18", "rr_18_24", "rr_ge24"} <= set(
        rr_bins["rr_bin"]
    )

    subject = pd.read_csv(output_dir / "subject_summary.csv")
    assert {220, 671, 1006} <= set(subject["samp_id"])

    tail = pd.read_csv(output_dir / "tail_summary.csv")
    assert {"mean", "median", "p90", "p95", "frac_gt_1", "frac_gt_2"} <= set(
        tail["stat"]
    )
    assert "breath_count_zero_cross_abs_error" not in set(tail["metric"])
    assert "breath_count_zero_cross_bpm_error" in set(tail["metric"])

    manifest = json.loads((output_dir / "analysis_manifest.json").read_text(encoding="utf-8"))
    assert manifest["window_seconds"] == 180.0
    assert "breath_count_zero_cross_bpm_error" in manifest["metrics"]
    assert manifest["comparisons"][1] == {
        "name": "bandenergy_vs_wide",
        "target": "bandenergy",
        "baseline": "wide",
    }


def test_external_strata_analysis_reuses_fixed_rows_for_models_and_paired_delta(tmp_path):
    eval_root, _ = _prepare_eval_root(tmp_path)
    strata = pd.DataFrame(
        {
            "dataset_row_id": [1, 2, 3, 4, 5],
            "samp_id": [220, 220, 671, 671, 1006],
            "stratum": [
                "harmonic_negative",
                "strong_harmonic",
                "harmonic_prominent",
                "peak_doubling",
                "strong_harmonic",
            ],
            "harmonic_positive": [False, True, True, True, True],
        }
    )
    spec = StratifiedAnalysisSpec(
        eval_root=eval_root,
        output_dir=tmp_path / "unused",
        labels=["time", "wide", "bandenergy"],
        seeds=[101, 102],
        comparisons=[
            Comparison(name="wide_vs_time", target="wide", baseline="time"),
            Comparison(name="bandenergy_vs_time", target="bandenergy", baseline="time"),
        ],
    )

    outputs = run_external_strata_analysis(spec, strata)

    model_seed = outputs["model_seed"]
    assert {"all_windows", "eligible_total", "harmonic_positive_union"} <= set(
        model_seed["stratum"]
    )
    positive = model_seed[
        (model_seed["label"] == "wide")
        & (model_seed["seed"] == 101)
        & (model_seed["stratum"] == "harmonic_positive_union")
    ].iloc[0]
    assert positive["n_windows"] == 4
    assert positive["n_subjects"] == 3
    assert "rr_peak_band_robust_abs_error_mean" in model_seed.columns
    assert "breath_count_zero_cross_bpm_error_p95" in model_seed.columns

    paired_seed = outputs["paired_seed"]
    wide_positive = paired_seed[
        (paired_seed["comparison"] == "wide_vs_time")
        & (paired_seed["seed"] == 101)
        & (paired_seed["stratum"] == "harmonic_positive_union")
    ].iloc[0]
    assert wide_positive["n_windows"] == 4
    assert wide_positive["delta_rr_peak_band_robust_abs_error_mean"] < 0
    assert {"value_mean", "value_std"} <= set(outputs["model_summary"].columns)


def test_external_strata_analysis_rejects_metrics_missing_fixed_rows(tmp_path):
    eval_root, _ = _prepare_eval_root(tmp_path)
    path = eval_root / "wide_101_test_metrics.csv"
    pd.read_csv(path).iloc[:-1].to_csv(path, index=False)
    strata = pd.DataFrame(
        {
            "dataset_row_id": [1, 2, 3, 4, 5],
            "samp_id": [220, 220, 671, 671, 1006],
            "stratum": "harmonic_negative",
            "harmonic_positive": False,
        }
    )
    spec = StratifiedAnalysisSpec(
        eval_root=eval_root,
        output_dir=tmp_path / "unused",
        labels=["time", "wide"],
        seeds=[101, 102],
        comparisons=[Comparison(name="wide_vs_time", target="wide", baseline="time")],
    )

    with pytest.raises(ValueError, match="固定分层 row"):
        run_external_strata_analysis(spec, strata)
