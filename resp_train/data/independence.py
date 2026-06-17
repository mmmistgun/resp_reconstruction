from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral, Real

import pandas as pd


DEFAULT_CATEGORICAL_COLUMNS = ("allowed_losses", "residual_quality_class", "input_set")
DEFAULT_NUMERIC_COLUMNS = ("valid_ratio", "input_finite_ratio", "target_finite_ratio")
CATEGORICAL_DISTRIBUTION_COLUMNS = ["column", "split", "value", "count", "ratio"]


def audit_split_independence(
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    *,
    categorical_columns: Iterable[str] = DEFAULT_CATEGORICAL_COLUMNS,
    numeric_columns: Iterable[str] = DEFAULT_NUMERIC_COLUMNS,
) -> dict[str, pd.DataFrame]:
    """审计 train/val 是否在个体和片段层面独立，并给出基础分布对照。"""
    train = train_rows.copy()
    val = val_rows.copy()
    _require_columns(train, ("samp_id", "segment_id"))
    _require_columns(val, ("samp_id", "segment_id"))

    train_samp = set(_normalized_values(train["samp_id"]))
    val_samp = set(_normalized_values(val["samp_id"]))
    train_segments = set(_segment_keys(train))
    val_segments = set(_segment_keys(val))
    overlap_samp = sorted(train_samp & val_samp)
    overlap_segments = sorted(train_segments & val_segments)

    summary = pd.DataFrame(
        [
            {
                "train_windows": int(len(train)),
                "val_windows": int(len(val)),
                "train_samp_id_count": int(len(train_samp)),
                "val_samp_id_count": int(len(val_samp)),
                "overlap_samp_id_count": int(len(overlap_samp)),
                "overlap_segment_count": int(len(overlap_segments)),
                "has_samp_id_leakage": bool(overlap_samp),
                "has_segment_leakage": bool(overlap_segments),
                "max_train_windows_per_samp_id": _max_group_size(train, "samp_id"),
                "max_val_windows_per_samp_id": _max_group_size(val, "samp_id"),
            }
        ]
    )

    return {
        "summary": summary,
        "overlap_samp_id": pd.DataFrame({"samp_id": overlap_samp}),
        "overlap_segment": pd.DataFrame({"segment_key": overlap_segments}),
        "categorical_distribution": _categorical_distribution(train, val, categorical_columns),
        "numeric_distribution": _numeric_distribution(train, val, numeric_columns),
        "per_samp_id": _per_samp_summary(train, val),
    }


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"独立性审计缺少必需列: {missing}")


def _segment_keys(frame: pd.DataFrame) -> list[str]:
    keys: list[str] = []
    for samp_id, segment_id in zip(frame["samp_id"], frame["segment_id"], strict=False):
        samp_key = _normalize_id_key(samp_id)
        segment_key = _normalize_id_key(segment_id)
        if samp_key is None or segment_key is None:
            continue
        keys.append(f"{samp_key}::{segment_key}")
    return keys


def _normalized_values(series: pd.Series) -> list[str]:
    return [key for value in series if (key := _normalize_id_key(value)) is not None]


def _normalize_id_key(value: object) -> str | None:
    """把数值等价的 ID 归一成同一 key，同时保留字符串 ID 的原始写法。"""
    if pd.isna(value):
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real):
        numeric_value = float(value)
        if numeric_value.is_integer():
            return str(int(numeric_value))
        return format(numeric_value, ".15g")
    return str(value)


def _max_group_size(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame.columns:
        return 0
    return int(frame.groupby(column, dropna=False).size().max())


def _categorical_distribution(
    train: pd.DataFrame,
    val: pd.DataFrame,
    columns: Iterable[str],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for column in columns:
        if column not in train.columns or column not in val.columns:
            continue
        for split_name, frame in (("train", train), ("val", val)):
            counts = frame[column].fillna("__MISSING__").astype(str).value_counts(dropna=False)
            total = max(int(counts.sum()), 1)
            for value, count in counts.items():
                records.append(
                    {
                        "column": str(column),
                        "split": split_name,
                        "value": str(value),
                        "count": int(count),
                        "ratio": float(count) / float(total),
                    }
                )
    return pd.DataFrame.from_records(records, columns=CATEGORICAL_DISTRIBUTION_COLUMNS)


def _numeric_distribution(
    train: pd.DataFrame,
    val: pd.DataFrame,
    columns: Iterable[str],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for split_name, frame in (("train", train), ("val", val)):
        record: dict[str, object] = {"split": split_name}
        for column in columns:
            if column not in frame.columns:
                continue
            values = pd.to_numeric(frame[column], errors="coerce")
            record[f"{column}_mean"] = _stable_float(values.mean())
            record[f"{column}_median"] = _stable_float(values.median())
            record[f"{column}_p10"] = _stable_float(values.quantile(0.10))
            record[f"{column}_p90"] = _stable_float(values.quantile(0.90))
        records.append(record)
    return pd.DataFrame.from_records(records)


def _stable_float(value: object) -> float:
    """规整浮点尾差，保证审计 CSV 和测试输出稳定。"""
    return round(float(value), 12)


def _per_samp_summary(train: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    records = []
    for split_name, frame in (("train", train), ("val", val)):
        grouped = frame.groupby("samp_id", dropna=False).size().reset_index(name="window_count")
        grouped["split"] = split_name
        records.append(grouped[["split", "samp_id", "window_count"]])
    if not records:
        return pd.DataFrame(columns=["split", "samp_id", "window_count"])
    return pd.concat(records, ignore_index=True)
