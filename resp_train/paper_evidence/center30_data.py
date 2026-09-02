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
from resp_train.paper_evidence.center30_config import CENTER30_INPUT_SAMPLES, CENTER30_OUTPUT_SAMPLES
from resp_train.paper_evidence.center_context_data import audit_parent_row_identity


PARENT_SAMPLES = 18000
INPUT_SLICES = {
    3000: (7500, 10500),
    4500: (6750, 11250),
    6000: (6000, 12000),
    9000: (4500, 13500),
}
TARGET_SLICE = (7500, 10500)
LATENT_CENTER_SLICES = {
    3000: (0, 300),
    4500: (75, 375),
    6000: (150, 450),
    9000: (300, 600),
}


def center30_input_bounds(input_samples: int) -> tuple[int, int]:
    try:
        return INPUT_SLICES[int(input_samples)]
    except KeyError as exc:
        raise ValueError(f"center30 input_samples 只允许 {list(CENTER30_INPUT_SAMPLES)}") from exc


def center30_latent_bounds(input_samples: int) -> tuple[int, int]:
    try:
        return LATENT_CENTER_SLICES[int(input_samples)]
    except KeyError as exc:
        raise ValueError(f"center30 input_samples 只允许 {list(CENTER30_INPUT_SAMPLES)}") from exc


def crop_center30_tensors(
    parent_input: torch.Tensor,
    parent_target: torch.Tensor,
    *,
    input_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if parent_input.shape[-1] != PARENT_SAMPLES or parent_target.shape[-1] != PARENT_SAMPLES:
        raise ValueError("center30 视图要求 18000 点父 input/target")
    if not bool(torch.isfinite(parent_input).all() and torch.isfinite(parent_target).all()):
        raise FloatingPointError("center30 父 input/target 包含 NaN/Inf")
    start, stop = center30_input_bounds(input_samples)
    return parent_input[..., start:stop], parent_target[..., TARGET_SLICE[0] : TARGET_SLICE[1]]


class Center30ContextDataset(Dataset):
    def __init__(self, parent: Dataset, *, input_samples: int, w_cache: Any | None = None) -> None:
        self.parent = parent
        self.input_samples = int(input_samples)
        center30_input_bounds(self.input_samples)
        self.w_cache = w_cache
        rows = getattr(parent, "rows", None)
        if isinstance(rows, pd.DataFrame):
            splits = set(rows["split"].astype(str)) if "split" in rows else set()
            if not splits or not splits.issubset({"train", "val"}):
                raise ValueError(f"center30 dataset 只允许 train/val，实际 {sorted(splits)}")
            if self.w_cache is not None:
                self.w_cache.verify_rows(rows["dataset_row_id"].astype(int).tolist())

    def __len__(self) -> int:
        return len(self.parent)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.parent[index]
        if not isinstance(item, Mapping) or not {"x", "target", "meta"}.issubset(item):
            raise TypeError("center30 父 dataset item 必须包含 x、target、meta")
        x, target = crop_center30_tensors(item["x"], item["target"], input_samples=self.input_samples)
        meta = dict(item["meta"])
        if str(meta.get("split", "")) not in {"train", "val"}:
            raise ValueError(f"center30 item 拒绝 split={meta.get('split')!r}")
        input_start, input_stop = center30_input_bounds(self.input_samples)
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
class Center30DataBundle:
    index_path: Path
    train_rows: pd.DataFrame
    val_rows: pd.DataFrame
    train_dataset: Center30ContextDataset
    val_dataset: Center30ContextDataset
    train_loader: DataLoader
    val_loader: DataLoader
    identity_audit: dict[str, Any]


def build_center30_data(cfg: DictConfig) -> Center30DataBundle:
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
        raise RuntimeError("center30 formal parent row count 漂移")
    identity = audit_parent_row_identity(train_rows, val_rows)
    parent_train = ResearchV2WindowDataset(index_path, train_rows, cfg, preload_windows=False)
    parent_val = ResearchV2WindowDataset(index_path, val_rows, cfg, preload_windows=False)
    w_train = w_val = None
    cache_path = cfg.data.get("center_w_cache_path")
    if cache_path:
        from resp_train.paper_evidence.center30_cache import Center30WCacheReader, sha256_file

        resolved = Path(str(cache_path)).resolve()
        manifest_sha = sha256_file(resolved / "cache_manifest.json")
        if manifest_sha != str(cfg.data.center_w_cache_manifest_sha256):
            raise RuntimeError("center30 W cache manifest SHA-256 漂移")
        identity.update({"center_w_cache_path": str(resolved), "center_w_cache_manifest_sha256": manifest_sha})
        w_train = Center30WCacheReader(resolved, split="train", input_samples=int(cfg.window.input_samples))
        w_val = Center30WCacheReader(resolved, split="val", input_samples=int(cfg.window.input_samples))
    train_dataset = Center30ContextDataset(parent_train, input_samples=int(cfg.window.input_samples), w_cache=w_train)
    val_dataset = Center30ContextDataset(parent_val, input_samples=int(cfg.window.input_samples), w_cache=w_val)
    generator = torch.Generator().manual_seed(int(cfg.training.seed))
    loader_kwargs = {"batch_size": int(cfg.training.batch_size), "num_workers": int(cfg.training.num_workers), "drop_last": False}
    return Center30DataBundle(
        index_path=index_path,
        train_rows=train_rows,
        val_rows=val_rows,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_loader=DataLoader(train_dataset, shuffle=True, generator=generator, **loader_kwargs),
        val_loader=DataLoader(val_dataset, shuffle=False, **loader_kwargs),
        identity_audit=identity,
    )


def audit_center30_nested_views(parent_items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not parent_items:
        raise ValueError("center30 nested-view audit 至少需要一个 parent item")
    for item in parent_items:
        targets = [
            crop_center30_tensors(item["x"], item["target"], input_samples=length)[1]
            for length in CENTER30_INPUT_SAMPLES
        ]
        if not all(torch.equal(targets[0], value) for value in targets[1:]):
            raise RuntimeError("center30 四种输入视图的 target 不一致")
    return {"checked_parent_rows": len(parent_items), "center30_targets_pointwise_identical": True}


__all__ = [
    "Center30ContextDataset",
    "Center30DataBundle",
    "INPUT_SLICES",
    "LATENT_CENTER_SLICES",
    "PARENT_SAMPLES",
    "TARGET_SLICE",
    "audit_center30_nested_views",
    "build_center30_data",
    "center30_input_bounds",
    "center30_latent_bounds",
    "crop_center30_tensors",
]
