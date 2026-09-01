from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from resp_train.data.index import filter_index
from resp_train.data.research_v2 import ResearchV2WindowDataset, read_research_v2_index
from resp_train.paper_evidence.center_context_config import CENTER_CONTEXT_INPUT_SAMPLES


PARENT_SAMPLES = 18000
CENTER_OUTPUT_SAMPLES = 6000
INPUT_SLICES = {
    6000: (6000, 12000),
    9000: (4500, 13500),
    18000: (0, 18000),
}
TARGET_SLICE = (6000, 12000)
LATENT_CENTER_SLICES = {
    6000: (0, 600),
    9000: (150, 750),
    18000: (600, 1200),
}


def center_input_bounds(input_samples: int) -> tuple[int, int]:
    try:
        return INPUT_SLICES[int(input_samples)]
    except KeyError as exc:
        raise ValueError(f"input_samples 只允许 {list(CENTER_CONTEXT_INPUT_SAMPLES)}") from exc


def latent_center_bounds(input_samples: int) -> tuple[int, int]:
    try:
        return LATENT_CENTER_SLICES[int(input_samples)]
    except KeyError as exc:
        raise ValueError(f"input_samples 只允许 {list(CENTER_CONTEXT_INPUT_SAMPLES)}") from exc


def crop_center_context_tensors(
    parent_input: torch.Tensor,
    parent_target: torch.Tensor,
    *,
    input_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if parent_input.shape[-1] != PARENT_SAMPLES or parent_target.shape[-1] != PARENT_SAMPLES:
        raise ValueError("中心上下文视图要求 18000 点父 input/target")
    if not bool(torch.isfinite(parent_input).all() and torch.isfinite(parent_target).all()):
        raise FloatingPointError("父 input/target 包含 NaN/Inf")
    start, stop = center_input_bounds(input_samples)
    target_start, target_stop = TARGET_SLICE
    return parent_input[..., start:stop], parent_target[..., target_start:target_stop]


class CenterContextDataset(Dataset):
    """把冻结 180 s admitted parent row 映射为单一嵌套输入与共同中心 target。"""

    def __init__(
        self,
        parent: Dataset,
        *,
        input_samples: int,
        w_cache: Any | None = None,
    ) -> None:
        self.parent = parent
        self.input_samples = int(input_samples)
        center_input_bounds(self.input_samples)
        self.w_cache = w_cache
        rows = getattr(parent, "rows", None)
        if isinstance(rows, pd.DataFrame):
            splits = set(rows["split"].astype(str)) if "split" in rows else set()
            if not splits or not splits.issubset({"train", "val"}):
                raise ValueError(f"中心上下文 dataset 只允许 train/val，实际 {sorted(splits)}")
            if self.w_cache is not None:
                self.w_cache.verify_rows(rows["dataset_row_id"].astype(int).tolist())

    def __len__(self) -> int:
        return len(self.parent)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.parent[index]
        if not isinstance(item, Mapping) or "x" not in item or "target" not in item or "meta" not in item:
            raise TypeError("父 dataset item 必须包含 x、target、meta")
        x, target = crop_center_context_tensors(
            item["x"],
            item["target"],
            input_samples=self.input_samples,
        )
        meta = dict(item["meta"])
        split = str(meta.get("split", ""))
        if split not in {"train", "val"}:
            raise ValueError(f"中心上下文 item 拒绝 split={split!r}")
        meta.update(
            {
                "parent_samples": PARENT_SAMPLES,
                "input_samples": self.input_samples,
                "input_slice_start": center_input_bounds(self.input_samples)[0],
                "input_slice_stop": center_input_bounds(self.input_samples)[1],
                "target_slice_start": TARGET_SLICE[0],
                "target_slice_stop": TARGET_SLICE[1],
            }
        )
        output: dict[str, Any] = {"x": x.contiguous(), "target": target.contiguous(), "meta": meta}
        if self.w_cache is not None:
            output["tf"] = {"w": self.w_cache.get(int(meta["dataset_row_id"]))}
        return output


@dataclass(frozen=True)
class CenterContextDataBundle:
    index_path: Path
    train_rows: pd.DataFrame
    val_rows: pd.DataFrame
    train_dataset: CenterContextDataset
    val_dataset: CenterContextDataset
    train_loader: DataLoader
    val_loader: DataLoader
    identity_audit: dict[str, Any]


def build_center_context_data(cfg: DictConfig) -> CenterContextDataBundle:
    """只构建 train/validation；函数没有 split 参数，避免意外路由到 test。"""

    index_path = (Path(str(cfg.data.dataset_root)) / str(cfg.data.index_csv)).resolve()
    audited = read_research_v2_index(cfg.data.dataset_root, cfg.data.index_csv, cfg)
    train_rows = filter_index(
        audited,
        cfg,
        split="train",
        max_windows=cfg.data.get("max_train_windows"),
        sample_strategy=str(cfg.data.train_sample_strategy),
        sample_seed=int(cfg.data.train_sample_seed),
    )
    val_rows = filter_index(
        audited,
        cfg,
        split="val",
        max_windows=cfg.data.get("max_val_windows"),
        sample_strategy=str(cfg.data.val_sample_strategy),
        sample_seed=int(cfg.data.val_sample_seed),
    )
    if str(cfg.protocol.run_role) == "formal":
        counts = {"train": len(train_rows), "val": len(val_rows)}
        if counts != {"train": 10141, "val": 2675}:
            raise RuntimeError(f"中心 formal parent row count 漂移: {counts}")
    identity = audit_parent_row_identity(train_rows, val_rows)
    parent_train = ResearchV2WindowDataset(
        index_path,
        train_rows,
        cfg,
        preload_windows=bool(cfg.data.get("preload_windows", False)),
    )
    parent_val = ResearchV2WindowDataset(
        index_path,
        val_rows,
        cfg,
        preload_windows=bool(cfg.data.get("preload_windows", False)),
    )
    w_train = w_val = None
    cache_path = cfg.data.get("center_w_cache_path")
    if cache_path:
        from resp_train.paper_evidence.center_context_cache import CenterContextWCacheReader

        require_complete = str(cfg.protocol.run_role) == "formal"
        w_train = CenterContextWCacheReader(
            cache_path,
            split="train",
            input_samples=int(cfg.window.input_samples),
            require_complete=require_complete,
        )
        w_val = CenterContextWCacheReader(
            cache_path,
            split="val",
            input_samples=int(cfg.window.input_samples),
            require_complete=require_complete,
        )
    train_dataset = CenterContextDataset(parent_train, input_samples=int(cfg.window.input_samples), w_cache=w_train)
    val_dataset = CenterContextDataset(parent_val, input_samples=int(cfg.window.input_samples), w_cache=w_val)
    generator = torch.Generator()
    generator.manual_seed(int(cfg.training.seed))
    loader_kwargs = {
        "batch_size": int(cfg.training.batch_size),
        "num_workers": int(cfg.training.num_workers),
        "drop_last": False,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return CenterContextDataBundle(
        index_path=index_path,
        train_rows=train_rows,
        val_rows=val_rows,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        identity_audit=identity,
    )


def audit_parent_row_identity(train_rows: pd.DataFrame, val_rows: pd.DataFrame) -> dict[str, Any]:
    required = {"dataset_row_id", "split", "samp_id", "coupling_state_id"}
    for name, rows in (("train", train_rows), ("val", val_rows)):
        missing = sorted(required - set(rows.columns))
        if missing:
            raise ValueError(f"{name} rows 缺少身份列: {missing}")
        if rows.empty or set(rows["split"].astype(str)) != {name}:
            raise ValueError(f"{name} rows 为空或 split 不一致")
        if rows["dataset_row_id"].duplicated().any():
            raise ValueError(f"{name} dataset_row_id 重复")
    train_ids = set(train_rows["dataset_row_id"].astype(int))
    val_ids = set(val_rows["dataset_row_id"].astype(int))
    train_samp = set(train_rows["samp_id"].astype(int))
    val_samp = set(val_rows["samp_id"].astype(int))
    train_session = set(zip(train_rows["samp_id"].astype(int), train_rows["coupling_state_id"].astype(int)))
    val_session = set(zip(val_rows["samp_id"].astype(int), val_rows["coupling_state_id"].astype(int)))
    if train_ids & val_ids:
        raise RuntimeError("train/validation dataset_row_id 发生重叠")
    if train_samp & val_samp:
        raise RuntimeError("train/validation samp_id 发生泄漏")
    if train_session & val_session:
        raise RuntimeError("train/validation subject/session 发生泄漏")
    return {
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_row_ids_sha256": _row_id_hash(train_rows["dataset_row_id"]),
        "val_row_ids_sha256": _row_id_hash(val_rows["dataset_row_id"]),
        "row_overlap_count": 0,
        "samp_id_overlap_count": 0,
        "subject_session_overlap_count": 0,
        "views_share_parent_rows": True,
    }


def audit_nested_view_identity(parent_items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not parent_items:
        raise ValueError("嵌套视图审计至少需要一个父 item")
    checked = 0
    for item in parent_items:
        targets = []
        row_id = int(item["meta"]["dataset_row_id"])
        for input_samples in CENTER_CONTEXT_INPUT_SAMPLES:
            x, target = crop_center_context_tensors(item["x"], item["target"], input_samples=input_samples)
            if x.shape[-1] != input_samples or target.shape[-1] != CENTER_OUTPUT_SAMPLES:
                raise RuntimeError(f"row={row_id} 中心视图 shape 错误")
            targets.append(target)
        if not all(torch.equal(targets[0], value) for value in targets[1:]):
            raise RuntimeError(f"row={row_id} 三种视图中心 target 不同")
        checked += 1
    return {"checked_parent_rows": checked, "center_targets_pointwise_identical": True}


def _row_id_hash(values: pd.Series) -> str:
    ids = np.sort(values.to_numpy(dtype=np.int64))
    return hashlib.sha256(ids.tobytes(order="C")).hexdigest()


__all__ = [
    "CENTER_OUTPUT_SAMPLES",
    "CenterContextDataBundle",
    "CenterContextDataset",
    "INPUT_SLICES",
    "LATENT_CENTER_SLICES",
    "PARENT_SAMPLES",
    "TARGET_SLICE",
    "audit_nested_view_identity",
    "audit_parent_row_identity",
    "build_center_context_data",
    "center_input_bounds",
    "crop_center_context_tensors",
    "latent_center_bounds",
]
