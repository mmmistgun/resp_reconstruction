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

SAMPLE_STRATEGIES = {"head", "random", "stratified_random"}


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
    sample_strategy: str | None = None,
    sample_seed: int | None = None,
) -> pd.DataFrame:
    filtered = df[(df["input_set"] == cfg.data.input_set) & (df["split"] == split)].copy()
    if bool(cfg.data.get("filter_unusable", True)) and "usable" in filtered.columns:
        filtered = filtered[filtered["usable"]].copy()
    strategy = _resolve_sample_strategy(cfg, split, sample_strategy)
    seed = _resolve_sample_seed(cfg, split, sample_seed)
    return _sample_rows(filtered, cfg, max_windows=max_windows, strategy=strategy, seed=seed)


def _resolve_sample_strategy(cfg: DictConfig, split: str, explicit: str | None) -> str:
    if explicit is not None:
        strategy = explicit
    elif split == str(cfg.data.get("train_split", "train")):
        strategy = cfg.data.get("train_sample_strategy", "stratified_random")
    elif split == str(cfg.data.get("val_split", "val")):
        strategy = cfg.data.get("val_sample_strategy", "stratified_random")
    else:
        strategy = "stratified_random"

    strategy = str(strategy)
    if explicit is None and strategy == "head":
        strategy = "stratified_random"
    if strategy not in SAMPLE_STRATEGIES:
        raise ValueError(f"sample_strategy 必须是 {sorted(SAMPLE_STRATEGIES)} 之一，当前为: {strategy}")
    return strategy


def _resolve_sample_seed(cfg: DictConfig, split: str, explicit: int | None) -> int:
    if explicit is not None:
        return int(explicit)

    training_cfg = cfg.get("training", {})
    fallback_seed = int(training_cfg.get("seed", 0)) if hasattr(training_cfg, "get") else 0
    if split == str(cfg.data.get("train_split", "train")):
        return int(cfg.data.get("train_sample_seed", fallback_seed))
    if split == str(cfg.data.get("val_split", "val")):
        return int(cfg.data.get("val_sample_seed", fallback_seed))
    return fallback_seed


def _sample_rows(
    df: pd.DataFrame,
    cfg: DictConfig,
    *,
    max_windows: int | None,
    strategy: str,
    seed: int,
) -> pd.DataFrame:
    sorted_rows = df.sort_values("dataset_row_id").reset_index(drop=True)
    if max_windows is None or len(sorted_rows) <= int(max_windows):
        return sorted_rows

    n = int(max_windows)
    if n < 0:
        raise ValueError("max_windows 必须是非负整数或 None")
    if n == 0:
        return sorted_rows.iloc[0:0].copy()

    if strategy == "head":
        sampled = sorted_rows.head(n)
    elif strategy == "random":
        sampled = sorted_rows.sample(n=n, random_state=seed)
    else:
        sampled = _stratified_random_sample(sorted_rows, cfg, n=n, seed=seed)
    return sampled.sort_values("dataset_row_id").reset_index(drop=True)


def _stratified_random_sample(df: pd.DataFrame, cfg: DictConfig, *, n: int, seed: int) -> pd.DataFrame:
    column = str(cfg.data.get("stratify_column", "residual_quality_class"))
    if column not in df.columns:
        raise ValueError(f"分层抽样列不存在: {column}")

    counts = df[column].value_counts(dropna=False, sort=False)
    total = len(df)
    buckets = []
    for order, (value, count) in enumerate(counts.items()):
        raw_quota = float(count) * n / total
        quota = int(raw_quota)
        buckets.append(
            {
                "value": value,
                "count": int(count),
                "quota": quota,
                "fraction": raw_quota - quota,
                "order": order,
            }
        )

    remaining = n - sum(int(bucket["quota"]) for bucket in buckets)
    for bucket in sorted(buckets, key=lambda item: (-float(item["fraction"]), int(item["order"]))):
        if remaining <= 0:
            break
        if int(bucket["quota"]) < int(bucket["count"]):
            bucket["quota"] = int(bucket["quota"]) + 1
            remaining -= 1

    # 极端情况下若某层配额超过可用量，继续分配给仍有余量的层。
    for bucket in buckets:
        if int(bucket["quota"]) > int(bucket["count"]):
            remaining += int(bucket["quota"]) - int(bucket["count"])
            bucket["quota"] = int(bucket["count"])
    while remaining > 0:
        available = [bucket for bucket in buckets if int(bucket["quota"]) < int(bucket["count"])]
        if not available:
            break
        for bucket in sorted(available, key=lambda item: (-float(item["fraction"]), int(item["order"]))):
            if remaining <= 0:
                break
            bucket["quota"] = int(bucket["quota"]) + 1
            remaining -= 1

    samples = []
    for offset, bucket in enumerate(buckets):
        quota = int(bucket["quota"])
        if quota <= 0:
            continue
        value = bucket["value"]
        if pd.isna(value):
            layer = df[df[column].isna()]
        else:
            layer = df[df[column] == value]
        samples.append(layer.sample(n=quota, random_state=seed + offset))

    if not samples:
        return df.iloc[0:0].copy()
    return pd.concat(samples)
