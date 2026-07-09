from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from resp_train.metrics.evaluate import (
    _rr_peak_valid_mask,
    _validate_predictions,
    build_target_feature_cache,
    build_target_feature_record,
    evaluate_prediction_dict,
    stack_target_feature_records,
    target_feature_context,
)


@dataclass(frozen=True)
class MetricChunkTask:
    start: int
    end: int
    method: str


@dataclass(frozen=True)
class TargetFeatureChunkTask:
    start: int
    end: int


_METRICS_PREDICTIONS: dict[str, np.ndarray] | None = None
_METRICS_TARGET_FEATURES: dict[str, np.ndarray] | None = None
_METRICS_CFG: Any | None = None
_TARGET_PREDICTIONS: dict[str, np.ndarray] | None = None
_TARGET_CONTEXT: dict[str, float | int] | None = None


def evaluate_predictions_chunked(
    predictions: dict[str, np.ndarray],
    cfg: DictConfig,
    *,
    method: str,
    metrics_workers: int,
    metrics_chunk_size: int,
    target_features: dict[str, np.ndarray] | None,
    show_progress: bool | None,
) -> pd.DataFrame:
    """按窗口 chunk 用进程池计算 metrics；worker 内禁用线程嵌套。"""

    n_windows = int(np.asarray(predictions["r_tho_hat"]).shape[0])
    worker_count = resolve_metrics_workers(metrics_workers, task_count=max(1, _chunk_count(n_windows, metrics_chunk_size)))
    if worker_count == 1:
        return evaluate_prediction_dict(
            predictions,
            _cfg_without_metric_threads(cfg),
            method=method,
            show_progress=show_progress,
            target_features=target_features,
        )
    if "fork" not in mp.get_all_start_methods():
        raise RuntimeError("chunk-parallel metrics 需要 fork start method；当前平台不支持")

    tasks = build_metric_chunk_tasks(n_windows=n_windows, chunk_size=int(metrics_chunk_size), method=str(method))
    global _METRICS_PREDICTIONS, _METRICS_TARGET_FEATURES, _METRICS_CFG
    _METRICS_PREDICTIONS = predictions
    _METRICS_TARGET_FEATURES = target_features
    _METRICS_CFG = _cfg_without_metric_threads(cfg)
    frames: list[tuple[int, pd.DataFrame]] = []
    ctx = mp.get_context("fork")
    try:
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=ctx) as pool:
            futures = [pool.submit(_compute_metrics_chunk_worker, task, bool(show_progress)) for task in tasks]
            done_count = 0
            total = len(futures)
            for future in as_completed(futures):
                start, frame = future.result()
                frames.append((start, frame))
                done_count += 1
                if done_count == 1 or done_count % max(1, total // 10) == 0 or done_count == total:
                    print(f"  metrics chunks done {done_count}/{total}", flush=True)
    finally:
        _METRICS_PREDICTIONS = None
        _METRICS_TARGET_FEATURES = None
        _METRICS_CFG = None
    return pd.concat([frame for _, frame in sorted(frames, key=lambda item: item[0])], ignore_index=True)


def build_metric_chunk_tasks(*, n_windows: int, chunk_size: int, method: str) -> list[MetricChunkTask]:
    if int(chunk_size) <= 0:
        raise ValueError("metrics_chunk_size 必须大于 0")
    return [
        MetricChunkTask(start=start, end=min(start + int(chunk_size), int(n_windows)), method=str(method))
        for start in range(0, int(n_windows), int(chunk_size))
    ]


def build_target_feature_chunk_tasks(*, n_windows: int, chunk_size: int) -> list[TargetFeatureChunkTask]:
    if int(chunk_size) <= 0:
        raise ValueError("target_chunk_size 必须大于 0")
    return [
        TargetFeatureChunkTask(start=start, end=min(start + int(chunk_size), int(n_windows)))
        for start in range(0, int(n_windows), int(chunk_size))
    ]


def resolve_metrics_workers(value: int, *, task_count: int) -> int:
    return min(max(1, int(value)), max(1, int(task_count)))


def load_or_build_target_feature_cache(
    predictions: dict[str, np.ndarray],
    cfg: DictConfig,
    *,
    cache_dir: str | Path | None,
    show_progress: bool | None,
    target_workers: int = 1,
    target_chunk_size: int = 128,
) -> dict[str, np.ndarray]:
    """读取或生成 target-side feature cache；cache_dir 为空时只做进程内缓存。"""

    if cache_dir is None:
        return build_target_feature_cache_chunked(
            predictions,
            cfg,
            target_workers=target_workers,
            target_chunk_size=target_chunk_size,
            show_progress=show_progress,
        )
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_key = target_feature_cache_key(predictions, cfg)
    cache_path = cache_root / f"{cache_key}.npz"
    if cache_path.exists():
        return load_target_feature_cache(cache_path, expected_key=cache_key)
    features = build_target_feature_cache_chunked(
        predictions,
        cfg,
        target_workers=target_workers,
        target_chunk_size=target_chunk_size,
        show_progress=show_progress,
    )
    save_target_feature_cache(cache_path, features, cache_key=cache_key)
    return features


def build_target_feature_cache_chunked(
    predictions: dict[str, np.ndarray],
    cfg: DictConfig,
    *,
    target_workers: int,
    target_chunk_size: int,
    show_progress: bool | None,
) -> dict[str, np.ndarray]:
    """按窗口 chunk 并行计算 target-only 特征；worker 内不再嵌套线程。"""

    _validate_predictions(predictions)
    n_windows = int(np.asarray(predictions["tho_ref"]).shape[0])
    worker_count = resolve_metrics_workers(target_workers, task_count=max(1, _chunk_count(n_windows, target_chunk_size)))
    if worker_count == 1:
        return build_target_feature_cache(predictions, cfg, show_progress=show_progress)
    if "fork" not in mp.get_all_start_methods():
        raise RuntimeError("chunk-parallel target features 需要 fork start method；当前平台不支持")

    tasks = build_target_feature_chunk_tasks(n_windows=n_windows, chunk_size=int(target_chunk_size))
    global _TARGET_PREDICTIONS, _TARGET_CONTEXT
    _TARGET_PREDICTIONS = predictions
    _TARGET_CONTEXT = target_feature_context(cfg)
    frames: list[tuple[int, list[dict[str, Any]]]] = []
    ctx = mp.get_context("fork")
    try:
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=ctx) as pool:
            futures = [pool.submit(_compute_target_feature_chunk_worker, task) for task in tasks]
            done_count = 0
            total = len(futures)
            for future in as_completed(futures):
                start, records = future.result()
                frames.append((start, records))
                done_count += 1
                if done_count == 1 or done_count % max(1, total // 10) == 0 or done_count == total:
                    print(f"  target feature chunks done {done_count}/{total}", flush=True)
    finally:
        _TARGET_PREDICTIONS = None
        _TARGET_CONTEXT = None
    records = [
        record
        for _, chunk_records in sorted(frames, key=lambda item: item[0])
        for record in chunk_records
    ]
    return stack_target_feature_records(records)


def save_target_feature_cache(path: Path, features: dict[str, np.ndarray], *, cache_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp.npz")
    payload = {key: np.asarray(value) for key, value in features.items()}
    payload["cache_key"] = np.asarray(cache_key)
    try:
        np.savez(tmp_path, **payload)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def load_target_feature_cache(path: Path, *, expected_key: str | None = None) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        cache_key = str(np.asarray(data["cache_key"]).item()) if "cache_key" in data.files else ""
        if expected_key is not None and cache_key != str(expected_key):
            raise ValueError(f"target feature cache key 不匹配: path={path}")
        return {key: np.asarray(data[key]) for key in data.files if key != "cache_key"}


def target_feature_cache_key(predictions: dict[str, np.ndarray], cfg: DictConfig) -> str:
    h = hashlib.sha256()
    h.update(b"tho_target_feature_cache_v3")
    relevant_cfg = {
        "target_fs": float(cfg.window.target_fs),
        "envelope_window_sec": float(cfg.loss.envelope_window_sec),
        "spectrum_low_hz": float(cfg.loss.spectrum_low_hz),
        "spectrum_high_hz": float(cfg.loss.spectrum_high_hz),
        "evaluation": {
            "lag_bandpass_order": int(cfg.get("evaluation", {}).get("lag_bandpass_order", 4)),
            "raw_peak_min_good_segment_sec": float(
                cfg.get("evaluation", {}).get("raw_peak_min_good_segment_sec", 20.0)
            ),
            "local_rr_window_sec": float(cfg.get("evaluation", {}).get("local_rr_window_sec", 40.0)),
            "local_rr_step_sec": float(cfg.get("evaluation", {}).get("local_rr_step_sec", 10.0)),
        },
    }
    h.update(repr(relevant_cfg).encode("utf-8"))
    for key in ("dataset_row_id", "tho_ref", "rr_peak_valid_mask"):
        if key not in predictions:
            h.update(f"{key}:none".encode("utf-8"))
            continue
        arr = np.ascontiguousarray(np.asarray(predictions[key]))
        h.update(str(key).encode("utf-8"))
        h.update(str(arr.shape).encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(arr.view(np.uint8))
    return h.hexdigest()[:24]


def _chunk_count(n_windows: int, chunk_size: int) -> int:
    if int(chunk_size) <= 0:
        raise ValueError("metrics_chunk_size 必须大于 0")
    return max(1, (int(n_windows) + int(chunk_size) - 1) // int(chunk_size))


def _compute_metrics_chunk_worker(task: MetricChunkTask, show_progress: bool) -> tuple[int, pd.DataFrame]:
    if _METRICS_PREDICTIONS is None or _METRICS_CFG is None:
        raise RuntimeError("metrics worker 未继承预测数组，请确认使用 fork start method")
    predictions = _slice_prediction_dict(_METRICS_PREDICTIONS, task.start, task.end)
    target_features = (
        _slice_feature_dict(_METRICS_TARGET_FEATURES, task.start, task.end)
        if _METRICS_TARGET_FEATURES is not None
        else None
    )
    frame = evaluate_prediction_dict(
        predictions,
        _METRICS_CFG,
        method=task.method,
        show_progress=show_progress,
        target_features=target_features,
    )
    return task.start, frame


def _compute_target_feature_chunk_worker(task: TargetFeatureChunkTask) -> tuple[int, list[dict[str, Any]]]:
    if _TARGET_PREDICTIONS is None or _TARGET_CONTEXT is None:
        raise RuntimeError("target feature worker 未继承预测数组，请确认使用 fork start method")
    targets = np.asarray(_TARGET_PREDICTIONS["tho_ref"])
    records: list[dict[str, Any]] = []
    for idx in range(task.start, task.end):
        target = np.asarray(targets[idx], dtype=np.float64).reshape(-1)
        rr_peak_valid_mask = _rr_peak_valid_mask(_TARGET_PREDICTIONS, idx, expected_size=target.size)
        records.append(build_target_feature_record(target, rr_peak_valid_mask, _TARGET_CONTEXT))
    return task.start, records


def _slice_prediction_dict(predictions: dict[str, np.ndarray], start: int, end: int) -> dict[str, np.ndarray]:
    n_windows = int(np.asarray(predictions["r_tho_hat"]).shape[0])
    sliced: dict[str, np.ndarray] = {}
    for key, value in predictions.items():
        arr = np.asarray(value)
        sliced[key] = arr[start:end] if arr.shape[:1] == (n_windows,) else arr
    return sliced


def _slice_feature_dict(features: dict[str, np.ndarray], start: int, end: int) -> dict[str, np.ndarray]:
    n_windows = int(np.asarray(next(iter(features.values()))).shape[0])
    sliced: dict[str, np.ndarray] = {}
    for key, value in features.items():
        arr = np.asarray(value)
        sliced[key] = arr[start:end] if arr.shape[:1] == (n_windows,) else arr
    return sliced


def _cfg_without_metric_threads(cfg: DictConfig):
    updated = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    if "evaluation" not in updated or updated.evaluation is None:
        updated.evaluation = {}
    updated.evaluation.metric_workers = 1
    return updated
