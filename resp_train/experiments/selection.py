from __future__ import annotations

TASK_SELECTION_COLUMNS = (
    "rr_peak_band_abs_error_mean",
    "frac_gt_1",
    "frac_gt_2",
    "rr_spec_abs_error_mean",
    "breath_count_zero_cross_abs_error_mean",
)

EPOCH_TASK_SELECTION_COLUMNS = tuple(f"val_{column}" for column in TASK_SELECTION_COLUMNS)

