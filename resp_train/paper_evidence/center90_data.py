from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from resp_train.data.index import filter_index
from resp_train.data.research_v2 import ResearchV2WindowDataset, read_research_v2_index
from resp_train.paper_evidence.center90_config import CENTER90_INPUT_SAMPLES
from resp_train.paper_evidence.center_context_data import audit_parent_row_identity


PARENT_SAMPLES = 18000
INPUT_SLICES = {
    9000: (4500, 13500),
    13500: (2250, 15750),
    18000: (0, 18000),
}
TARGET_SLICE = (4500, 13500)
LATENT_CENTER_SLICES = {
    9000: (0, 900),
    13500: (225, 1125),
    18000: (450, 1350),
}


def center90_input_bounds(input_samples: int) -> tuple[int, int]:
    try:
        return INPUT_SLICES[int(input_samples)]
    except KeyError as exc:
        raise ValueError(f"center90 input_samples 只允许 {list(CENTER90_INPUT_SAMPLES)}") from exc


def center90_latent_bounds(input_samples: int) -> tuple[int, int]:
    try:
        return LATENT_CENTER_SLICES[int(input_samples)]
    except KeyError as exc:
        raise ValueError(f"center90 input_samples 只允许 {list(CENTER90_INPUT_SAMPLES)}") from exc


def crop_center90_tensors(
    parent_input: torch.Tensor,
    parent_target: torch.Tensor,
    *,
    input_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if parent_input.shape[-1] != PARENT_SAMPLES or parent_target.shape[-1] != PARENT_SAMPLES:
        raise ValueError("center90 视图要求 18000 点父 input/target")
    if not bool(torch.isfinite(parent_input).all() and torch.isfinite(parent_target).all()):
        raise FloatingPointError("center90 父 input/target 包含 NaN/Inf")
    start, stop = center90_input_bounds(input_samples)
    return parent_input[..., start:stop], parent_target[..., TARGET_SLICE[0] : TARGET_SLICE[1]]


class Center90ContextDataset(Dataset):
    def __init__(self, parent: Dataset, *, input_samples: int, w_cache: Any | None = None) -> None:
        self.parent = parent
        self.input_samples = int(input_samples)
        center90_input_bounds(self.input_samples)
        self.w_cache = w_cache
        rows = getattr(parent, "rows", None)
        if isinstance(rows, pd.DataFrame):
            splits = set(rows["split"].astype(str)) if "split" in rows else set()
            if not splits or not splits.issubset({"train", "val"}):
                raise ValueError(f"center90 dataset 只允许 train/val，实际 {sorted(splits)}")
            if self.w_cache is not None:
                self.w_cache.verify_rows(rows["dataset_row_id"].astype(int).tolist())

    def __len__(self) -> int:
        return len(self.parent)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.parent[index]
        if not isinstance(item, Mapping) or not {"x", "target", "meta"}.issubset(item):
            raise TypeError("center90 父 dataset item 必须包含 x、target、meta")
        x, target = crop_center90_tensors(item["x"], item["target"], input_samples=self.input_samples)
        meta = dict(item["meta"])
        if str(meta.get("split", "")) not in {"train", "val"}:
            raise ValueError(f"center90 item 拒绝 split={meta.get('split')!r}")
        input_start, input_stop = center90_input_bounds(self.input_samples)
        meta.update(
            {
                "parent_samples": PARENT_SAMPLES,
                "input_samples": self.input_samples,
                "input_slice_start": input_start,
                "input_slice_stop": input_stop,
                "target_slice_start": TARGET_SLICE[0],
                "target_slice_stop": TARGET_SLICE[1],
            }
        )
        output: dict[str, Any] = {"x": x.contiguous(), "target": target.contiguous(), "meta": meta}
        if self.w_cache is not None:
            output["tf"] = {"w": self.w_cache.get(int(meta["dataset_row_id"]))}
        return output


@dataclass(frozen=True)
class Center90DataBundle:
    index_path: Path
    train_rows: pd.DataFrame
    val_rows: pd.DataFrame
    train_dataset: Center90ContextDataset
    val_dataset: Center90ContextDataset
    train_loader: DataLoader
    val_loader: DataLoader
    identity_audit: dict[str, Any]


def build_center90_data(cfg: DictConfig) -> Center90DataBundle:
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
    if str(cfg.protocol.run_role) == "formal" and {"train": len(train_rows), "val": len(val_rows)} != {
        "train": 10141,
        "val": 2675,
    }:
        raise RuntimeError("center90 formal parent row count 漂移")
    identity = audit_parent_row_identity(train_rows, val_rows)
    parent_train = ResearchV2WindowDataset(index_path, train_rows, cfg, preload_windows=False)
    parent_val = ResearchV2WindowDataset(index_path, val_rows, cfg, preload_windows=False)
    w_train = w_val = None
    cache_path = cfg.data.get("center_w_cache_path")
    if cache_path:
        from resp_train.paper_evidence.center90_cache import open_center90_w_cache

        resolved = Path(str(cache_path)).resolve()
        expected_hash = str(cfg.data.center_w_cache_manifest_sha256)
        w_train = open_center90_w_cache(
            resolved,
            split="train",
            input_samples=int(cfg.window.input_samples),
            expected_manifest_sha256=expected_hash,
        )
        w_val = open_center90_w_cache(
            resolved,
            split="val",
            input_samples=int(cfg.window.input_samples),
            expected_manifest_sha256=expected_hash,
        )
        identity.update({"center_w_cache_path": str(resolved), "center_w_cache_manifest_sha256": expected_hash})
    train_dataset = Center90ContextDataset(
        parent_train, input_samples=int(cfg.window.input_samples), w_cache=w_train
    )
    val_dataset = Center90ContextDataset(parent_val, input_samples=int(cfg.window.input_samples), w_cache=w_val)
    generator = torch.Generator().manual_seed(int(cfg.training.seed))
    loader_kwargs = {
        "batch_size": int(cfg.training.batch_size),
        "num_workers": int(cfg.training.num_workers),
        "drop_last": False,
    }
    return Center90DataBundle(
        index_path=index_path,
        train_rows=train_rows,
        val_rows=val_rows,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_loader=DataLoader(train_dataset, shuffle=True, generator=generator, **loader_kwargs),
        val_loader=DataLoader(val_dataset, shuffle=False, **loader_kwargs),
        identity_audit=identity,
    )


def audit_center90_nested_views(parent_items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not parent_items:
        raise ValueError("center90 nested-view audit 至少需要一个 parent item")
    for item in parent_items:
        targets = [
            crop_center90_tensors(item["x"], item["target"], input_samples=length)[1]
            for length in CENTER90_INPUT_SAMPLES
        ]
        if not all(torch.equal(targets[0], value) for value in targets[1:]):
            raise RuntimeError("center90 三种输入视图的 target 不一致")
    return {"checked_parent_rows": len(parent_items), "center90_targets_pointwise_identical": True}


__all__ = [
    "Center90ContextDataset",
    "Center90DataBundle",
    "INPUT_SLICES",
    "LATENT_CENTER_SLICES",
    "PARENT_SAMPLES",
    "TARGET_SLICE",
    "audit_center90_nested_views",
    "build_center90_data",
    "center90_input_bounds",
    "center90_latent_bounds",
    "crop_center90_tensors",
]
