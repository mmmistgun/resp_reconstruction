from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import DictConfig


REQUIRED_INDEX_COLUMNS = [
    "dataset_row_id",
    "input_set",
    "split",
    "samp_id",
    "segment_id",
    "window_id_in_segment",
    "source_npz",
    "bcg_signal_key",
    "target_signal_key",
    "valid_sec_key",
    "segment_decision",
    "window_start_sample",
    "window_end_sample",
    "window_duration_samples",
    "target_fs",
    "valid_ratio",
    "input_finite_ratio",
    "target_finite_ratio",
    "residual_quality_class",
    "base_alignment_method",
    "apply_decision",
    "reason",
]


def read_index(dataset_root: str | Path, index_csv: str | Path) -> pd.DataFrame:
    index_path = Path(dataset_root) / index_csv
    if not index_path.exists():
        raise FileNotFoundError(f"索引文件不存在: {index_path}")
    df = pd.read_csv(index_path)
    validate_index_columns(df)
    return df


def validate_index_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_INDEX_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"索引缺少必需列: {missing}")


def filter_index(
    df: pd.DataFrame,
    cfg: DictConfig,
    *,
    split: str,
    max_windows: int | None,
) -> pd.DataFrame:
    filtered = df[(df["input_set"] == cfg.data.input_set) & (df["split"] == split)].copy()
    if bool(cfg.data.get("filter_unusable", True)) and "usable" in filtered.columns:
        filtered = filtered[filtered["usable"]].copy()
    filtered = filtered.sort_values("dataset_row_id").reset_index(drop=True)
    if max_windows is not None:
        filtered = filtered.head(int(max_windows)).reset_index(drop=True)
    return filtered
