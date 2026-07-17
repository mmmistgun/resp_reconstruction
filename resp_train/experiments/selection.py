from __future__ import annotations

TASK_GUARD_COLUMNS = (
    "rr_peak_band_robust_abs_error_mean",
    "breath_count_zero_cross_abs_error_mean",
)

# 当前任务先按两个主护栏择优；旧 peak-band/spec 只在主护栏完全持平时打破平局。
TASK_SELECTION_COLUMNS = (
    *TASK_GUARD_COLUMNS,
    "rr_peak_band_abs_error_mean",
    "rr_spec_abs_error_mean",
)

EPOCH_TASK_SELECTION_COLUMNS = tuple(f"val_{column}" for column in TASK_SELECTION_COLUMNS)
