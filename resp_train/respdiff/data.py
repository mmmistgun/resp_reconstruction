"""180秒父窗口与5秒片段的显式映射；预测路径不接收target统计。"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt, resample
from torch.utils.data import Dataset

PROFILES = {"segment_soft_z", "source_minmax"}
REQUIRED = ("dataset_row_id", "split", "samp_id", "source_npz", "target_source_npz",
            "window_start_sample", "window_end_sample")


def array(name, value, shape):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name} 要求有限数组{shape}，实际{value.shape}")
    return value


def float32(name, value):
    with np.errstate(over="raise", invalid="raise"):
        result = value.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} 转FP32产生非有限值")
    return result


def lowpass(values):
    # 与来源同为30 Hz下八阶1 Hz低通，显式固定SciPy的默认pad合同。
    b, a = butter(8, 1 / 15, btype="low", analog=False)
    result = filtfilt(b, a, values, padtype="odd", padlen=3 * max(len(a), len(b)))
    if not np.isfinite(result).all():
        raise ValueError("低通输出非有限")
    return result


def minmax(chunks, *, condition):
    width = np.ptp(chunks, axis=-1, keepdims=True)
    if np.any(width <= 0):
        raise ValueError("源码min-max遇到常量片段，无法定义归一化")
    result = (chunks - chunks.min(axis=-1, keepdims=True)) / width
    return 2 * result - 1 if condition else result


def prepare_condition(values, *, profile="segment_soft_z"):
    if profile not in PROFILES:
        raise ValueError(f"未知归一化profile: {profile}")
    values = array("BCG父窗口", values, (18000,))
    chunks = resample(values, 5400).reshape(36, 1, 150)
    if profile == "source_minmax":
        chunks = minmax(chunks, condition=True)
    return float32("condition chunks", array("condition chunks", chunks, (36, 1, 150)))


def prepare_target(values, *, profile="segment_soft_z"):
    if profile not in PROFILES:
        raise ValueError(f"未知归一化profile: {profile}")
    values = array("THO父窗口", values, (18000,))
    chunks = lowpass(resample(values, 5400)).reshape(36, 1, 150)
    if profile == "source_minmax":
        chunks = minmax(chunks, condition=False)
    return float32("target chunks", array("target chunks", chunks, (36, 1, 150)))


def restore_parent(predictions, *, chunk_indices):
    if list(chunk_indices) != list(range(36)):
        raise ValueError("父窗口必须按0…35完整顺序重组，存在缺失/重复/错序")
    chunks = array("prediction chunks", predictions, (36, 1, 150))
    result = resample(lowpass(chunks.reshape(5400)), 18000)
    return float32("重组prediction", array("重组prediction", result, (18000,)))


def validate_rows(rows):
    if rows.empty or any(key not in rows or rows[key].isna().any() for key in REQUIRED):
        raise ValueError("父窗口缺少完整身份")
    if not set(rows.split).issubset({"train", "val"}):
        raise ValueError("RespDiff开发入口仅允许train/val，test未开放")
    if rows.dataset_row_id.duplicated().any():
        raise ValueError("父窗口row ID重复")
    for column in ("dataset_row_id", "samp_id", "window_start_sample", "window_end_sample"):
        values = rows[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values != np.floor(values)):
            raise ValueError(f"父窗口{column}须为非负整数")
    if np.any(rows.window_end_sample - rows.window_start_sample != 18000):
        raise ValueError("父窗口必须是100 Hz、180秒")
    for column in ("source_npz", "target_source_npz"):
        if not all(isinstance(value, str) and value for value in rows[column]):
            raise ValueError(f"父窗口{column}须为非空路径")
        if any(not Path(value).is_absolute() or str(Path(value).resolve()) != value for value in rows[column]):
            raise ValueError(f"父窗口{column}须为规范化绝对路径")


def assert_split_independence(train_rows, val_rows):
    validate_rows(train_rows)
    validate_rows(val_rows)
    if set(train_rows.split) != {"train"} or set(val_rows.split) != {"val"}:
        raise ValueError("train/val角色不匹配")
    if set(train_rows.samp_id) & set(val_rows.samp_id):
        raise ValueError("train/val主体重叠")
    if set(train_rows.dataset_row_id) & set(val_rows.dataset_row_id):
        raise ValueError("train/val row ID重叠")
    for column in ("source_npz", "target_source_npz"):
        if set(train_rows[column]) & set(val_rows[column]):
            raise ValueError(f"train/val记录重叠: {column}")


def chunk_manifest(rows):
    validate_rows(rows)
    records = []
    for row in rows.to_dict("records"):
        for index in range(36):
            start = int(row["window_start_sample"]) + 500 * index
            records.append({**{key: row[key] for key in REQUIRED}, "chunk_index": index,
                            "chunk_start_sample": start, "chunk_end_sample": start + 500,
                            "sampling_key": json.dumps([row["split"], int(row["dataset_row_id"]), index])})
    return pd.DataFrame(records)


def interval_counts(manifest):
    counts = Counter(zip(manifest.samp_id, manifest.source_npz, manifest.chunk_start_sample,
                         manifest.chunk_end_sample))
    return {"chunks": len(manifest), "unique_source_intervals": len(counts),
            "max_multiplicity": max(counts.values(), default=0)}


class ChunkDataset(Dataset):
    """包装已经准入的ResearchV2父数据集；只缓存最近父窗口，避免36倍存储。"""
    def __init__(self, parents, rows, *, profile="segment_soft_z"):
        validate_rows(rows)
        if profile not in PROFILES or len(parents) != len(rows):
            raise ValueError("profile或父dataset长度错误")
        if hasattr(parents, "rows"):
            actual = parents.rows[list(REQUIRED)].reset_index(drop=True)
            if not actual.equals(rows[list(REQUIRED)].reset_index(drop=True)):
                raise ValueError("父dataset实际rows与声明身份不一致")
        self.parents, self.rows, self.profile = parents, rows.reset_index(drop=True).copy(), profile
        self.manifest = chunk_manifest(self.rows)
        self._cached = None

    def __len__(self):
        return len(self.rows) * 36

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        parent_index, chunk_index = divmod(index, 36)
        if self._cached is None or self._cached[0] != parent_index:
            item = self.parents[parent_index]
            expected = int(self.rows.iloc[parent_index].dataset_row_id)
            if int(item["meta"]["dataset_row_id"]) != expected:
                raise ValueError("父dataset返回了错误row")
            condition = prepare_condition(item["x"].detach().cpu().numpy().reshape(-1), profile=self.profile)
            target = prepare_target(item["target"].detach().cpu().numpy().reshape(-1), profile=self.profile)
            self._cached = (parent_index, condition, target)
        return {"x": torch.from_numpy(self._cached[1][chunk_index].copy()),
                "target": torch.from_numpy(self._cached[2][chunk_index].copy()),
                "meta": self.manifest.iloc[index].to_dict()}
