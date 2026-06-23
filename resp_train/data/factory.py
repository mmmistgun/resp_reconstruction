from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader

from resp_train.data.audit import add_usable_flag, summarize_audit
from resp_train.data.dataset import RespWindowDataset
from resp_train.data.index import filter_index, read_index
from resp_train.data.research_v2 import (
    ResearchV2WindowDataset,
    read_research_v2_index,
    summarize_research_v2_audit,
)


@dataclass(frozen=True)
class WindowDataBundle:
    index_path: Path
    rows: pd.DataFrame
    dataset: RespWindowDataset | ResearchV2WindowDataset
    loader: DataLoader
    audited: pd.DataFrame
    audit_summary: pd.DataFrame


@dataclass(frozen=True)
class ThoDataBundle:
    train: WindowDataBundle
    val: WindowDataBundle
    audited: pd.DataFrame
    audit_summary: pd.DataFrame


def build_window_data(
    cfg: DictConfig,
    *,
    split: str,
    max_windows: int | None,
    sample_strategy: str | None = None,
    sample_seed: int | None = None,
    shuffle: bool,
    audited: pd.DataFrame | None = None,
) -> WindowDataBundle:
    index_path = _index_path(cfg)
    audited_frame = _read_audited_index(cfg) if audited is None else audited
    audit_summary = _summarize_audit(audited_frame, cfg)
    return _build_window_bundle(
        cfg,
        index_path=index_path,
        audited=audited_frame,
        audit_summary=audit_summary,
        split=split,
        max_windows=max_windows,
        sample_strategy=sample_strategy,
        sample_seed=sample_seed,
        shuffle=shuffle,
    )


def build_tho_data(cfg: DictConfig) -> ThoDataBundle:
    index_path = _index_path(cfg)
    audited = _read_audited_index(cfg)
    audit_summary = _summarize_audit(audited, cfg)
    train = _build_window_bundle(
        cfg,
        index_path=index_path,
        audited=audited,
        audit_summary=audit_summary,
        split=str(cfg.data.train_split),
        max_windows=cfg.data.get("max_train_windows"),
        sample_strategy=str(cfg.data.train_sample_strategy),
        sample_seed=int(cfg.data.train_sample_seed),
        shuffle=True,
        empty_error_label="train",
    )
    val = _build_window_bundle(
        cfg,
        index_path=index_path,
        audited=audited,
        audit_summary=audit_summary,
        split=str(cfg.data.val_split),
        max_windows=cfg.data.get("max_val_windows"),
        sample_strategy=str(cfg.data.val_sample_strategy),
        sample_seed=int(cfg.data.val_sample_seed),
        shuffle=False,
        empty_error_label="val",
    )
    return ThoDataBundle(train=train, val=val, audited=audited, audit_summary=audit_summary)


def _build_window_bundle(
    cfg: DictConfig,
    *,
    index_path: Path,
    audited: pd.DataFrame,
    audit_summary: pd.DataFrame,
    split: str,
    max_windows: int | None,
    sample_strategy: str | None,
    sample_seed: int | None,
    shuffle: bool,
    empty_error_label: str | None = None,
) -> WindowDataBundle:
    rows = filter_index(
        audited,
        cfg,
        split=split,
        max_windows=max_windows,
        sample_strategy=sample_strategy,
        sample_seed=sample_seed,
    )
    dataset_cls = ResearchV2WindowDataset if _is_research_v2(cfg) else RespWindowDataset
    preload_progress_enabled = _preload_progress_enabled(cfg)
    dataset = dataset_cls(
        index_path,
        rows,
        cfg,
        preload_windows=bool(cfg.data.get("preload_windows", False)),
        preload_progress_desc=f"preload {split} windows" if preload_progress_enabled else None,
        preload_show_progress=True if preload_progress_enabled else False,
    )
    if empty_error_label is not None and len(dataset) == 0:
        raise RuntimeError(f"{empty_error_label} 数据为空，请检查 input_set、split、可用性过滤和抽样配置。")
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=bool(shuffle),
        pin_memory=_should_pin_memory(cfg),
        **_dataloader_worker_options(cfg),
    )
    return WindowDataBundle(
        index_path=index_path,
        rows=rows,
        dataset=dataset,
        loader=loader,
        audited=audited,
        audit_summary=audit_summary,
    )


def _index_path(cfg: DictConfig) -> Path:
    return Path(str(cfg.data.dataset_root)) / str(cfg.data.index_csv)


def _read_audited_index(cfg: DictConfig) -> pd.DataFrame:
    if _is_research_v2(cfg):
        return read_research_v2_index(cfg.data.dataset_root, cfg.data.index_csv, cfg)
    return add_usable_flag(read_index(cfg.data.dataset_root, cfg.data.index_csv), cfg)


def _summarize_audit(audited: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    if _is_research_v2(cfg):
        return summarize_research_v2_audit(audited)
    return summarize_audit(audited)


def _is_research_v2(cfg: DictConfig) -> bool:
    return str(cfg.data.get("format", "stage2_1")) == "research_v2"


def _should_pin_memory(cfg: DictConfig) -> bool:
    """CUDA 训练时启用 pinned memory，加速 CPU 到 GPU 的异步拷贝。"""
    device = str(cfg.training.get("device", "auto"))
    return device.startswith("cuda") or (device == "auto" and torch.cuda.is_available())


def _dataloader_worker_options(cfg: DictConfig) -> dict[str, int | bool]:
    """仅在多进程 DataLoader 下启用 worker 持久化和预取参数。"""
    num_workers = int(cfg.training.get("num_workers", 0))
    options: dict[str, int | bool] = {"num_workers": num_workers}
    if num_workers <= 0:
        return options

    options["persistent_workers"] = bool(cfg.training.get("persistent_workers", False))
    prefetch_factor = cfg.training.get("prefetch_factor", None)
    if prefetch_factor is not None:
        options["prefetch_factor"] = int(prefetch_factor)
    return options


def _preload_progress_enabled(cfg: DictConfig) -> bool:
    """预加载进度跟随 training.show_progress；false 时保持批量实验安静。"""
    value = cfg.training.get("show_progress", None)
    if value in (None, "auto"):
        return sys.stderr.isatty()
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
        raise ValueError(f"training.show_progress 只能是 true/false/auto，当前为: {value}")
    return bool(value)
