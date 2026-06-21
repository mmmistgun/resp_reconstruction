from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from resp_train.data.cache import WholeNightCache

_REQUIRED_COLUMNS = [
    "dataset_row_id",
    "split",
    "samp_id",
    "coupling_state_id",
    "window_start_s",
    "window_end_s",
    "source_npz",
    "target_source_npz",
    "hard_valid_ratio",
    "state_alignment_valid_ratio",
    "allowed_losses",
]


class ResearchV2WindowDataset(Dataset):
    """Research v2 整晚 NPZ + 窗口索引读取器。

    当前训练栈仍是单通道波形监督，因此这里默认只切 waveform target；多任务
    的 rate/phase/event 信息保留在 meta 和索引中，后续可以继续扩展 loss。
    """

    def __init__(
        self,
        index_csv_path: str | Path,
        rows: pd.DataFrame,
        cfg: DictConfig,
        *,
        preload_windows: bool = False,
    ) -> None:
        self.index_csv_path = Path(index_csv_path)
        self.cfg = cfg
        self.rows = rows.copy().reset_index(drop=True)
        if bool(cfg.data.get("filter_unusable", True)) and "usable" in self.rows.columns:
            self.rows = self.rows[self.rows["usable"]].reset_index(drop=True)
        self.source_cache = WholeNightCache(self.index_csv_path)
        self.target_cache = WholeNightCache(self.index_csv_path)
        if bool(cfg.data.get("drop_nonfinite_windows", True)):
            self.rows = self._drop_nonfinite_rows(self.rows)
        self._preloaded: list[dict[str, Any]] | None = None
        if preload_windows:
            self._preloaded = [self._load_item(i) for i in range(len(self.rows))]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._preloaded is not None:
            return self._preloaded[idx]
        return self._load_item(idx)

    def _load_item(self, idx: int) -> dict[str, Any]:
        row = self.rows.iloc[idx]
        start = int(row["window_start_sample"])
        end = int(row["window_end_sample"])
        expected = int(self.cfg.window.duration_samples)
        if end <= start or (end - start) != expected:
            raise ValueError(f"窗口长度异常 row={row['dataset_row_id']} start={start} end={end}")

        source_key = str(row["bcg_signal_key"])
        target_key = str(row["target_signal_key"])
        source_arrays = self.source_cache.get_arrays(str(row["source_npz"]), [source_key])
        target_arrays = self.target_cache.get_arrays(str(row["target_source_npz"]), [target_key])
        x = source_arrays[source_key][start:end].astype(np.float32, copy=False)
        y = target_arrays[target_key][start:end].astype(np.float32, copy=False)
        if x.shape[0] != expected or y.shape[0] != expected:
            raise ValueError(
                f"实际切片长度异常 row={row['dataset_row_id']} "
                f"x_len={x.shape[0]} target_len={y.shape[0]} expected={expected}"
            )
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"窗口包含非有限值 row={row['dataset_row_id']}")

        return {
            "x": torch.from_numpy(x.copy()).view(1, -1),
            "target": torch.from_numpy(y.copy()).view(1, -1),
            "meta": {
                "dataset_row_id": int(row["dataset_row_id"]),
                "split": str(row.get("split", "")),
                "input_set": str(row.get("input_set", "")),
                "samp_id": int(row["samp_id"]),
                "coupling_state_id": int(row["coupling_state_id"]),
                "allowed_losses": str(row.get("allowed_losses", "")),
                "supervision_confidence_level": str(row.get("supervision_confidence_level", "")),
                "state_alignment_method": str(row.get("state_alignment_method", "")),
            },
        }

    def _drop_nonfinite_rows(self, rows: pd.DataFrame) -> pd.DataFrame:
        if rows.empty:
            return rows
        keep: list[bool] = []
        expected = int(self.cfg.window.duration_samples)
        for _, row in rows.iterrows():
            start = int(row["window_start_sample"])
            end = int(row["window_end_sample"])
            if end <= start or (end - start) != expected:
                keep.append(False)
                continue
            source_key = str(row["bcg_signal_key"])
            target_key = str(row["target_signal_key"])
            try:
                source_arrays = self.source_cache.get_arrays(str(row["source_npz"]), [source_key])
                target_arrays = self.target_cache.get_arrays(str(row["target_source_npz"]), [target_key])
            except (FileNotFoundError, KeyError):
                keep.append(False)
                continue
            x = source_arrays[source_key][start:end]
            y = target_arrays[target_key][start:end]
            keep.append(
                x.shape[0] == expected
                and y.shape[0] == expected
                and bool(np.isfinite(x).all())
                and bool(np.isfinite(y).all())
            )
        return rows.loc[keep].reset_index(drop=True)


def read_research_v2_index(dataset_root: str | Path, index_csv: str | Path, cfg: DictConfig) -> pd.DataFrame:
    index_path = Path(dataset_root) / index_csv
    if not index_path.exists():
        raise FileNotFoundError(f"索引文件不存在: {index_path}")
    raw = pd.read_csv(index_path)
    return adapt_research_v2_index(raw, cfg)


def adapt_research_v2_index(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    missing = [column for column in _REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Research v2 索引缺少必需列: {missing}")

    target_task = str(cfg.data.get("target_task", "waveform"))
    if target_task != "waveform":
        raise ValueError(f"当前训练适配仅支持 data.target_task=waveform，当前为: {target_task}")

    fs = float(cfg.window.target_fs)
    expected = int(cfg.window.duration_samples)
    adapted = df.copy()
    adapted["input_set"] = str(cfg.data.get("input_set", "research_v2_waveform"))
    adapted["segment_id"] = adapted["coupling_state_id"].astype(int)
    adapted["window_id_in_segment"] = adapted.groupby(["samp_id", "coupling_state_id"]).cumcount() + 1
    adapted["window_start_sample"] = np.rint(adapted["window_start_s"].astype(float) * fs).astype(int)
    adapted["window_end_sample"] = adapted["window_start_sample"] + expected
    adapted["window_duration_samples"] = expected
    adapted["target_fs"] = fs
    adapted["bcg_signal_key"] = _resolve_key_column(adapted, cfg, cfg.data.get("bcg_input_key", "bcg_input_aligned_key"))
    adapted["target_signal_key"] = _resolve_key_column(adapted, cfg, cfg.data.get("target_key", "target_waveform_key"))
    adapted["valid_sec_key"] = "state_alignment_valid_sec"
    adapted["valid_ratio"] = adapted["state_alignment_valid_ratio"].astype(float)
    adapted["input_finite_ratio"] = 1.0
    adapted["target_finite_ratio"] = 1.0
    adapted["residual_quality_class"] = adapted["allowed_losses"].fillna("").astype(str)
    adapted["base_alignment_method"] = adapted.get("state_alignment_method", "").fillna("")
    adapted["apply_decision"] = np.where(adapted.get("reason", "").fillna("").eq(""), "included", "excluded")
    adapted["usable"] = _usable_mask(adapted, cfg)
    return adapted


def summarize_research_v2_audit(audited: pd.DataFrame) -> pd.DataFrame:
    grouped = audited.groupby(["split", "input_set", "residual_quality_class"], dropna=False)
    summary = grouped.agg(
        n_windows=("dataset_row_id", "count"),
        n_usable=("usable", "sum"),
        valid_ratio_mean=("valid_ratio", "mean"),
        valid_ratio_min=("valid_ratio", "min"),
        hard_valid_ratio_mean=("hard_valid_ratio", "mean"),
        supervision_confidence_main=("supervision_confidence_level", _mode_or_empty),
        alignment_method_main=("state_alignment_method", _mode_or_empty),
    ).reset_index()
    summary["usable_ratio"] = summary["n_usable"] / summary["n_windows"].clip(lower=1)
    return summary


def _resolve_key_column(adapted: pd.DataFrame, cfg: DictConfig, raw: Any) -> pd.Series:
    value = str(raw)
    if value in adapted.columns:
        return adapted[value].astype(str)
    return pd.Series([value] * len(adapted), index=adapted.index)


def _usable_mask(adapted: pd.DataFrame, cfg: DictConfig) -> pd.Series:
    allowed = adapted["allowed_losses"].fillna("").astype(str)
    has_waveform = allowed.str.split(";").apply(lambda items: "waveform" in set(items))
    reason_ok = adapted.get("reason", "").fillna("").eq("")
    hard_ok = adapted["hard_valid_ratio"].astype(float) >= float(cfg.data.get("min_hard_valid_ratio", 0.8))
    align_ok = adapted["state_alignment_valid_ratio"].astype(float) >= float(
        cfg.data.get("min_state_alignment_valid_ratio", 0.8)
    )
    return (has_waveform & reason_ok & hard_ok & align_ok).astype(bool)


def _mode_or_empty(series: pd.Series) -> str:
    mode = series.mode(dropna=True)
    if mode.empty:
        return ""
    return str(mode.iloc[0])
