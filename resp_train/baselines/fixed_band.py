from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from resp_train.metrics.task import evaluate_task_predictions, summarize_task_metrics


FIXED_BAND_METHOD = "F0_fixed_band_bcg"
FIXED_BAND_SOURCE_COLUMN = "bcg_input_segment_soft_z_key"
FIXED_BAND_EXPECTED_SIGNAL_KEY = "bcg_resp_band_state_aligned_segment_soft_z"


def prepare_fixed_band_config(cfg: DictConfig) -> DictConfig:
    """复制配置并切换到数据集中已有的固定呼吸带 soft-z 信号。"""

    if str(cfg.data.get("format", "")) != "research_v2":
        raise ValueError("固定呼吸带基线只支持当前 research_v2 数据集")
    prepared = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    prepared.data.bcg_input_key = FIXED_BAND_SOURCE_COLUMN
    # 基线按 batch 流式评价，避免把完整 validation 波形重复常驻内存。
    prepared.data.preload_windows = False
    prepared.training.device = "cpu"
    return prepared


def evaluate_fixed_band_loader(
    loader: Iterable[Mapping[str, Any]],
    cfg: DictConfig,
    *,
    include_test_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把固定呼吸带 BCG 直接作为 prediction，复用冻结任务指标。"""

    frames: list[pd.DataFrame] = []
    for batch in loader:
        predictions = {
            "r_tho_hat": _as_numpy(batch["x"]),
            "tho_ref": _as_numpy(batch["target"]),
            **_metric_metadata(batch.get("meta", {})),
        }
        frames.append(
            evaluate_task_predictions(
                predictions,
                cfg,
                include_test_only=include_test_only,
                method=FIXED_BAND_METHOD,
            )
        )
    if not frames:
        raise ValueError("固定呼吸带基线没有可评价 sample")
    metrics = pd.concat(frames, ignore_index=True)
    summary = summarize_task_metrics(metrics)
    summary.insert(0, "method", FIXED_BAND_METHOD)
    return metrics, summary


def attach_alignment_metadata(metrics: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """把评价所用索引元数据一对一附到逐 sample 结果，便于完整追溯。"""

    columns = [
        "dataset_row_id",
        "bcg_signal_key",
        "state_alignment_method",
        "state_alignment_is_reference_assisted",
        "state_alignment_lag_s",
    ]
    available = [column for column in columns if column in rows.columns]
    if "dataset_row_id" not in available:
        raise ValueError("数据索引缺少 dataset_row_id")
    metadata = rows[available].drop_duplicates("dataset_row_id")
    if len(metadata) != len(rows):
        raise ValueError("数据索引包含重复 dataset_row_id")
    merged = metrics.merge(metadata, on="dataset_row_id", how="left", validate="one_to_one")
    if merged["dataset_row_id"].isna().any() or len(merged) != len(metrics):
        raise ValueError("逐 sample 指标与数据索引无法一对一对应")
    return merged


def add_baseline_summary_metadata(summary: pd.DataFrame, metrics: pd.DataFrame, *, split: str) -> pd.DataFrame:
    enriched = summary.copy()
    enriched.insert(1, "split", str(split))
    if "bcg_signal_key" in metrics:
        keys = sorted(set(metrics["bcg_signal_key"].astype(str)))
        if keys != [FIXED_BAND_EXPECTED_SIGNAL_KEY]:
            raise ValueError(
                "固定呼吸带基线读取了意外信号 key: "
                f"expected={FIXED_BAND_EXPECTED_SIGNAL_KEY} actual={keys}"
            )
        enriched["source_signal_key"] = keys[0]
    return enriched


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _metric_metadata(meta: Mapping[str, Any]) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for key in ("dataset_row_id", "split", "input_set", "samp_id", "coupling_state_id"):
        if key not in meta:
            continue
        output[key] = _as_numpy(meta[key])
    return output
