from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from resp_train.crd.candidate_lock import sha256_file
from resp_train.crd.config import CRD_S2A_PROTOCOL_VERSION, load_crd_config
from resp_train.crd.experiment import _validate_checkpoint_config
from resp_train.crd.model import build_crd_model
from resp_train.crd.representations import MorphologyRepresentation
from resp_train.data.factory import build_window_data
from resp_train.utils.run import resolve_device, save_execution_manifest, set_seed


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOTYPE_OUTPUT_ROOT = REPO_ROOT / "runs/crd_v1/crd_s2a_prototype_diagnostics"
WINDOW_FILENAME = "prototype_window_metrics.csv"
SAMP_FILENAME = "prototype_samp_summary.csv"
SUMMARY_FILENAME = "prototype_summary.json"
MANIFEST_FILENAME = "prototype_manifest.json"


@torch.no_grad()
def evaluate_morphology_prototypes(
    *,
    checkpoint_path: str | Path,
    device: str = "cuda:0",
    output_root: str | Path = DEFAULT_PROTOTYPE_OUTPUT_ROOT,
) -> Path:
    """只读完整 validation，输出 CRD_204 prototype usage/entropy 描述性审计。"""

    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint_path}")
    config_path = checkpoint_path.parent / "config.yaml"
    cfg = load_crd_config(
        config_path,
        overrides=[f"training.device={device}", "training.show_progress=false"],
    )
    if str(cfg.protocol.name) != CRD_S2A_PROTOCOL_VERSION or str(cfg.protocol.run_role) != "formal":
        raise ValueError("prototype 诊断只允许 S2A formal checkpoint")
    if str(cfg.model.variant) != "crd_204_base_morphology":
        raise ValueError("prototype 诊断只允许 crd_204_base_morphology")
    if cfg.data.get("max_val_windows") is not None:
        raise ValueError("prototype 诊断必须读取完整 validation")

    seed = int(cfg.training.seed)
    final_dir = Path(output_root) / str(cfg.model.variant) / f"seed_{seed}"
    if final_dir.exists():
        raise FileExistsError(f"prototype 诊断产物禁止覆盖: {final_dir}")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = final_dir.parent / f".{final_dir.name}.{uuid4().hex}.tmp"
    temporary_dir.mkdir(parents=False, exist_ok=False)

    try:
        resolved_device = resolve_device(device)
        set_seed(seed)
        model = build_crd_model(cfg).to(resolved_device).eval()
        checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
        _validate_checkpoint_config(checkpoint.get("config"), cfg)
        model.load_state_dict(checkpoint["model_state_dict"])
        representation = getattr(model, "representation", None)
        if not isinstance(representation, MorphologyRepresentation):
            raise TypeError("CRD_204 checkpoint 未构造 MorphologyRepresentation")

        validation = build_window_data(
            cfg,
            split=str(cfg.data.val_split),
            max_windows=None,
            sample_strategy=str(cfg.data.val_sample_strategy),
            sample_seed=int(cfg.data.val_sample_seed),
            shuffle=False,
        )
        rows: list[dict[str, Any]] = []
        amp_enabled = resolved_device.type == "cuda" and bool(cfg.training.use_amp)
        for batch in validation.loader:
            if "meta" not in batch:
                raise KeyError("validation batch 缺少 meta")
            sensor = batch["x"].to(resolved_device, non_blocking=resolved_device.type == "cuda")
            with torch.amp.autocast(
                resolved_device.type,
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                scores = representation.prototype_scores(sensor)
            scores = scores.float()
            if not bool(torch.isfinite(scores).all()):
                raise FloatingPointError("prototype scores 包含 NaN/Inf")
            if not torch.allclose(scores.sum(dim=-1), torch.ones_like(scores[..., 0]), atol=2e-5, rtol=0.0):
                raise RuntimeError("prototype scores 未归一化为概率")

            hard = F.one_hot(scores.argmax(dim=-1), num_classes=representation.prototype_count).float()
            hard_usage = hard.mean(dim=1).cpu().numpy()
            soft_usage = scores.mean(dim=1).cpu().numpy()
            token_entropy = _normalized_entropy(scores).mean(dim=1).cpu().numpy()
            hard_entropy = _normalized_entropy(torch.from_numpy(hard_usage)).numpy()
            soft_entropy = _normalized_entropy(torch.from_numpy(soft_usage)).numpy()
            for index in range(scores.shape[0]):
                row: dict[str, Any] = {
                    "dataset_row_id": int(_meta_value(batch["meta"], "dataset_row_id", index)),
                    "samp_id": int(_meta_value(batch["meta"], "samp_id", index)),
                    "coupling_state_id": int(_meta_value(batch["meta"], "coupling_state_id", index)),
                    "prototype_token_entropy_normalized": float(token_entropy[index]),
                    "prototype_hard_usage_entropy_normalized": float(hard_entropy[index]),
                    "prototype_soft_usage_entropy_normalized": float(soft_entropy[index]),
                    "prototype_dominant_id": int(hard_usage[index].argmax()),
                    "prototype_dominant_fraction": float(hard_usage[index].max()),
                }
                for prototype in range(representation.prototype_count):
                    row[f"prototype_hard_usage_{prototype:02d}"] = float(hard_usage[index, prototype])
                    row[f"prototype_soft_usage_{prototype:02d}"] = float(soft_usage[index, prototype])
                rows.append(row)

        frame = pd.DataFrame(rows)
        _validate_window_identity(frame, checkpoint_path.parent / "metrics.csv")
        summary, samp_summary = summarize_prototype_frame(
            frame,
            prototype_count=representation.prototype_count,
        )
        summary.update(
            {
                "protocol": CRD_S2A_PROTOCOL_VERSION,
                "variant": str(cfg.model.variant),
                "seed": seed,
                "selected_epoch": int(checkpoint["epoch"]),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "prototype_temperature": representation.prototype_temperature,
                "prototype_loss": float(representation.prototype_loss().cpu()),
            }
        )
        frame.to_csv(temporary_dir / WINDOW_FILENAME, index=False)
        samp_summary.to_csv(temporary_dir / SAMP_FILENAME, index=False)
        (temporary_dir / SUMMARY_FILENAME).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        save_execution_manifest(
            temporary_dir / MANIFEST_FILENAME,
            task="crd_v1_s2a_morphology_prototype_diagnostics",
            phase="validation_prototype_diagnostics",
            protocol=CRD_S2A_PROTOCOL_VERSION,
            evaluation_split="validation",
            variant=str(cfg.model.variant),
            seed=seed,
            selected_epoch=int(checkpoint["epoch"]),
            checkpoint=str(checkpoint_path),
            checkpoint_sha256=sha256_file(checkpoint_path),
            observed_validation_windows=len(frame),
            observed_samp_ids=int(frame["samp_id"].nunique()),
        )
        os.replace(temporary_dir, final_dir)
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
    return final_dir


def summarize_prototype_frame(
    frame: pd.DataFrame,
    *,
    prototype_count: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    hard_columns = [f"prototype_hard_usage_{index:02d}" for index in range(prototype_count)]
    soft_columns = [f"prototype_soft_usage_{index:02d}" for index in range(prototype_count)]
    required = {
        "dataset_row_id",
        "samp_id",
        "prototype_token_entropy_normalized",
        *hard_columns,
        *soft_columns,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"prototype window frame 缺少字段: {missing}")
    numeric = frame[list(required - {"dataset_row_id", "samp_id"})].to_numpy(dtype=np.float64)
    if len(frame) == 0 or not np.isfinite(numeric).all():
        raise ValueError("prototype window frame 必须非空且数值有限")

    hard_usage = frame[hard_columns].mean(axis=0).to_numpy(dtype=np.float64)
    soft_usage = frame[soft_columns].mean(axis=0).to_numpy(dtype=np.float64)
    summary = {
        "n_windows": int(len(frame)),
        "n_samp_ids": int(frame["samp_id"].nunique()),
        "prototype_count": int(prototype_count),
        "hard_usage": hard_usage.tolist(),
        "soft_usage": soft_usage.tolist(),
        "hard_usage_entropy_normalized": _numpy_normalized_entropy(hard_usage),
        "soft_usage_entropy_normalized": _numpy_normalized_entropy(soft_usage),
        "mean_token_entropy_normalized": float(frame["prototype_token_entropy_normalized"].mean()),
        "dominant_hard_prototype": int(hard_usage.argmax()),
        "dominant_hard_fraction": float(hard_usage.max()),
        "dominant_soft_prototype": int(soft_usage.argmax()),
        "dominant_soft_fraction": float(soft_usage.max()),
    }

    samp_rows: list[dict[str, Any]] = []
    for samp_id, group in frame.groupby("samp_id", sort=True):
        samp_hard = group[hard_columns].mean(axis=0).to_numpy(dtype=np.float64)
        samp_soft = group[soft_columns].mean(axis=0).to_numpy(dtype=np.float64)
        row: dict[str, Any] = {
            "samp_id": int(samp_id),
            "n_windows": int(len(group)),
            "prototype_token_entropy_normalized_mean": float(
                group["prototype_token_entropy_normalized"].mean()
            ),
            "prototype_hard_usage_entropy_normalized": _numpy_normalized_entropy(samp_hard),
            "prototype_soft_usage_entropy_normalized": _numpy_normalized_entropy(samp_soft),
            "prototype_dominant_id": int(samp_hard.argmax()),
            "prototype_dominant_fraction": float(samp_hard.max()),
        }
        for prototype in range(prototype_count):
            row[hard_columns[prototype]] = float(samp_hard[prototype])
            row[soft_columns[prototype]] = float(samp_soft[prototype])
        samp_rows.append(row)
    return summary, pd.DataFrame(samp_rows)


def _normalized_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    count = int(probabilities.shape[-1])
    if count <= 1:
        raise ValueError("entropy 至少要求两个 prototype")
    work = probabilities.float().clamp_min(torch.finfo(torch.float32).tiny)
    return -(work * work.log()).sum(dim=-1) / math.log(count)


def _numpy_normalized_entropy(probabilities: np.ndarray) -> float:
    work = np.clip(np.asarray(probabilities, dtype=np.float64), np.finfo(np.float64).tiny, None)
    return float(-(work * np.log(work)).sum() / math.log(len(work)))


def _meta_value(meta: Any, key: str, index: int) -> Any:
    if key not in meta:
        raise KeyError(f"validation meta 缺少 {key}")
    value = meta[key]
    if torch.is_tensor(value):
        selected = value[index] if value.ndim > 0 else value
        return selected.item() if selected.ndim == 0 else selected.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        selected = value[index] if value.ndim > 0 else value
        return selected.item() if np.asarray(selected).ndim == 0 else selected
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


def _validate_window_identity(frame: pd.DataFrame, metrics_path: Path) -> None:
    if not metrics_path.exists():
        raise FileNotFoundError(f"checkpoint run 缺少 metrics.csv: {metrics_path}")
    metrics = pd.read_csv(metrics_path)
    identity = ["dataset_row_id", "samp_id", "coupling_state_id"]
    if len(frame) != len(metrics):
        raise RuntimeError(f"prototype validation 行数不一致: {len(frame)} != {len(metrics)}")
    for column in identity:
        if not np.array_equal(frame[column].to_numpy(), metrics[column].to_numpy()):
            raise RuntimeError(f"prototype validation {column} 顺序与正式 metrics 不一致")


__all__ = [
    "DEFAULT_PROTOTYPE_OUTPUT_ROOT",
    "evaluate_morphology_prototypes",
    "summarize_prototype_frame",
]
