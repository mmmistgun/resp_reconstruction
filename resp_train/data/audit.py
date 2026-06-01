from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig


def add_usable_flag(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    audited = df.copy()
    unusable_classes = set(cfg.data.get("unusable_residual_classes", []))
    audited["usable"] = (
        (audited["valid_ratio"] >= float(cfg.data.valid_ratio_min))
        & (audited["input_finite_ratio"] >= float(cfg.data.input_finite_ratio_min))
        & (audited["target_finite_ratio"] >= float(cfg.data.target_finite_ratio_min))
        & (audited["segment_decision"] == "include_candidate")
        & (~audited["residual_quality_class"].isin(unusable_classes))
    )
    audited["usable"] = audited["usable"].astype(bool)
    return audited


def summarize_audit(audited: pd.DataFrame) -> pd.DataFrame:
    grouped = audited.groupby(["split", "input_set", "residual_quality_class"], dropna=False)
    summary = grouped.agg(
        n_windows=("dataset_row_id", "count"),
        n_usable=("usable", "sum"),
        valid_ratio_mean=("valid_ratio", "mean"),
        valid_ratio_min=("valid_ratio", "min"),
        input_finite_ratio_mean=("input_finite_ratio", "mean"),
        target_finite_ratio_mean=("target_finite_ratio", "mean"),
        base_alignment_method_main=("base_alignment_method", _mode_or_empty),
        apply_decision_main=("apply_decision", _mode_or_empty),
    ).reset_index()
    summary["usable_ratio"] = summary["n_usable"] / summary["n_windows"].clip(lower=1)
    return summary[
        [
            "split",
            "input_set",
            "residual_quality_class",
            "n_windows",
            "n_usable",
            "usable_ratio",
            "valid_ratio_mean",
            "valid_ratio_min",
            "input_finite_ratio_mean",
            "target_finite_ratio_mean",
            "base_alignment_method_main",
            "apply_decision_main",
        ]
    ]


def _mode_or_empty(series: pd.Series) -> str:
    mode = series.mode(dropna=True)
    if mode.empty:
        return ""
    return str(mode.iloc[0])
