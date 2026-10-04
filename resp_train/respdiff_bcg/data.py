"""复用已有父窗口与 soft-z 数值，片段不重新归一化。"""

from collections import OrderedDict
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from resp_train.respdiff.data import REQUIRED, validate_rows
from .signal import chunks_from_parent, prepare_parent


class ChunkDataset(Dataset):
    def __init__(self, parents, rows):
        def canonical(frame):
            frame = frame.reset_index(drop=True).copy()
            base = Path(getattr(parents, "index_csv_path", "/")).parent
            for key in ("source_npz", "target_source_npz"):
                frame[key] = frame[key].map(lambda p: str((base / str(p)).resolve()))
            return frame

        rows = canonical(rows)
        validate_rows(rows)
        if len(parents) != len(rows):
            raise ValueError("父窗口数与 rows 不匹配")
        self.parents, self.rows = parents, rows.reset_index(drop=True).copy()
        if hasattr(parents, "rows") and not canonical(parents.rows)[list(REQUIRED)].equals(
                self.rows[list(REQUIRED)]):
            raise ValueError("父 dataset 身份不匹配")
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.rows) * 13

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        parent, chunk = divmod(index, 13)
        if parent not in self.cache:
            item = self.parents[parent]
            if int(item["meta"]["dataset_row_id"]) != int(self.rows.iloc[parent].dataset_row_id):
                raise ValueError("父 dataset 返回错误 row")
            self.cache[parent] = tuple(chunks_from_parent(prepare_parent(
                item[key].detach().cpu().numpy().reshape(-1))) for key in ("x", "target"))
            if len(self.cache) > 64:
                self.cache.popitem(last=False)
        self.cache.move_to_end(parent)
        x, target = self.cache[parent]
        return {"x": torch.from_numpy(x[chunk].copy()),
                "target": torch.from_numpy(target[chunk].copy()), "index": index}

    def manifest(self):
        return pd.DataFrame([{**{key: row[key] for key in REQUIRED},
                              "parent_index": parent, "chunk_index": chunk,
                              "relative_start_20hz": chunk * 300 - 300,
                              "relative_end_20hz": chunk * 300 + 300}
                             for parent, row in enumerate(self.rows.to_dict("records"))
                             for chunk in range(13)])
