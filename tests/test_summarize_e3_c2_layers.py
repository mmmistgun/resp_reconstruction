import pandas as pd

import scripts.summarize_e3_c2_layers as c2


def _metrics(rows):
    return pd.DataFrame(rows)


def test_c2_strata_use_fixed_baseline_bins_and_min_window_gate():
    baseline = _metrics(
        [
            {
                "dataset_row_id": 1,
                "rr_peak_band_abs_error": 0.4,
                "rr_peak_valid_ratio": 1.0,
                "band_limited_corr": 0.5,
                "spectrum_similarity": 0.9,
                "target_rr_peak_band_bpm": 9.0,
            },
            {
                "dataset_row_id": 2,
                "rr_peak_band_abs_error": 1.4,
                "rr_peak_valid_ratio": 0.7,
                "band_limited_corr": -0.3,
                "spectrum_similarity": 0.4,
                "target_rr_peak_band_bpm": 16.0,
            },
            {
                "dataset_row_id": 3,
                "rr_peak_band_abs_error": 2.0,
                "rr_peak_valid_ratio": 0.3,
                "band_limited_corr": 0.0,
                "spectrum_similarity": 0.2,
                "target_rr_peak_band_bpm": 23.0,
            },
        ]
    )

    strata = c2.build_strata_frame(baseline, success_threshold=1.0)

    assert strata["baseline_peak_band_bin"].tolist() == ["success", "failure", "failure"]
    assert strata["rr_peak_valid_ratio_bin"].tolist() == ["high", "mid", "low"]
    assert strata["band_limited_corr_bin"].tolist() == ["positive", "negative", "low_corr"]
    assert strata["target_rr_bin"].tolist() == ["slow", "normal", "fast"]


def test_c2_summaries_keep_training_delta_and_ablation_delta_separate():
    baseline = _metrics(
        [
            {
                "dataset_row_id": 1,
                "rr_peak_band_abs_error": 0.5,
                "relative_envelope_corr": 0.2,
                "rr_peak_valid_ratio": 1.0,
                "band_limited_corr": 0.4,
                "spectrum_similarity": 0.8,
                "target_rr_peak_band_bpm": 12.0,
            },
            {
                "dataset_row_id": 2,
                "rr_peak_band_abs_error": 1.5,
                "relative_envelope_corr": 0.1,
                "rr_peak_valid_ratio": 1.0,
                "band_limited_corr": 0.5,
                "spectrum_similarity": 0.7,
                "target_rr_peak_band_bpm": 14.0,
            },
        ]
    )
    dual = _metrics(
        [
            {"dataset_row_id": 1, "rr_peak_band_abs_error": 0.4, "relative_envelope_corr": 0.3},
            {"dataset_row_id": 2, "rr_peak_band_abs_error": 1.0, "relative_envelope_corr": 0.2},
        ]
    )
    stft_zero = _metrics(
        [
            {"dataset_row_id": 1, "rr_peak_band_abs_error": 0.8, "relative_envelope_corr": 0.1},
            {"dataset_row_id": 2, "rr_peak_band_abs_error": 1.2, "relative_envelope_corr": 0.05},
        ]
    )
    strata = c2.build_strata_frame(baseline, success_threshold=1.0)

    training = c2.summarize_metric_delta(
        candidate=dual,
        reference=baseline,
        strata=strata,
        metrics=["rr_peak_band_abs_error", "relative_envelope_corr"],
        delta_kind="dual_minus_time_only",
        comparison="C1B_vs_C1T",
        min_windows=1,
    )
    ablation = c2.summarize_metric_delta(
        candidate=dual,
        reference=stft_zero,
        strata=strata,
        metrics=["rr_peak_band_abs_error", "relative_envelope_corr"],
        delta_kind="normal_minus_stft_zero",
        comparison="C1B_normal_vs_stft_zero",
        min_windows=1,
    )

    combined = pd.concat([training, ablation], ignore_index=True)
    global_rows = combined[(combined["stratum_name"] == "all") & (combined["metric"] == "rr_peak_band_abs_error")]

    assert set(global_rows["delta_kind"]) == {"dual_minus_time_only", "normal_minus_stft_zero"}
    assert global_rows.set_index("delta_kind").loc["dual_minus_time_only", "mean_delta"] == -0.3
    assert global_rows.set_index("delta_kind").loc["normal_minus_stft_zero", "mean_delta"] == -0.3
    assert {"candidate_mean", "reference_mean", "n_windows"} <= set(combined.columns)
