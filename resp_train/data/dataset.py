from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from resp_train.data.cache import WholeNightCache


class RespWindowDataset(Dataset):
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
        self.cache = WholeNightCache(self.index_csv_path)
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

        arrays = self.cache.get_arrays(
            str(row["source_npz"]),
            [str(row["bcg_signal_key"]), str(row["target_signal_key"])],
        )
        x = arrays[str(row["bcg_signal_key"])][start:end].astype(np.float32, copy=False)
        y = arrays[str(row["target_signal_key"])][start:end].astype(np.float32, copy=False)
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
                "segment_id": int(row["segment_id"]),
                "window_id_in_segment": int(row["window_id_in_segment"]),
                "residual_quality_class": str(row.get("residual_quality_class", "")),
            },
        }
