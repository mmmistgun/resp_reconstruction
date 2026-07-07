import math

import scripts.profile_epoch_val_metrics as profiler


def test_build_summary_reports_first_and_steady_epoch_costs():
    summary = profiler.build_summary(
        checkpoint="runs/example/checkpoint.pt",
        config="runs/example/config.yaml",
        n_windows=2675,
        metrics_workers=4,
        metrics_chunk_size=128,
        target_workers=4,
        target_chunk_size=128,
        collect_predictions_sec=12.5,
        target_features_sec=18.0,
        metrics_secs=[41.0, 39.0, 40.0],
    )

    assert summary["n_windows"] == 2675
    assert summary["target_workers"] == 4
    assert summary["collect_predictions_sec"] == 12.5
    assert summary["target_features_sec"] == 18.0
    assert summary["metrics_sec_median"] == 40.0
    assert summary["checkpoint_eval_like_sec"] == 70.5
    assert summary["first_epoch_extra_after_valid_prediction_sec"] == 58.0
    assert summary["steady_epoch_extra_after_valid_prediction_sec"] == 40.0
    assert math.isclose(summary["metrics_windows_per_sec"], 66.875)
