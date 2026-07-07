from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import multiprocessing as mp
import os
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import numpy as np
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.config import load_config
from resp_train.data.factory import build_window_data
from resp_train.engine import collect_predictions
from resp_train.experiments.tho import _resolve_config_path, _validate_checkpoint_config
from resp_train.metrics.evaluate import build_target_feature_cache, evaluate_prediction_dict
from resp_train.models.registry import build_model
from resp_train.utils.run import resolve_device


@dataclass(frozen=True)
class TestEvalOutputs:
    metrics: Path
    summary: Path
    manifest: Path


@dataclass(frozen=True)
class MetricChunkTask:
    start: int
    end: int
    method: str


_METRICS_PREDICTIONS: dict[str, np.ndarray] | None = None
_METRICS_TARGET_FEATURES: dict[str, np.ndarray] | None = None
_METRICS_CFG: Any | None = None


def evaluate_tho_test_checkpoint(
    *,
    checkpoint_path: str | Path,
    config_path: str | Path | None,
    metrics_output_path: str | Path | None,
    summary_output_path: str | Path | None,
    manifest_output_path: str | Path | None,
    overrides: list[str] | None = None,
    metrics_workers: int = 1,
    metrics_chunk_size: int = 128,
    target_cache_dir: str | Path | None = None,
) -> TestEvalOutputs:
    """固定 checkpoint，在 held-out test split 上评价并保存指标与追踪信息。"""

    resolved_checkpoint = Path(checkpoint_path)
    resolved_config = _resolve_config_path(config_path, resolved_checkpoint)
    cfg = load_config(resolved_config, overrides=overrides)
    output_paths = _resolve_output_paths(
        resolved_checkpoint,
        metrics_output_path=metrics_output_path,
        summary_output_path=summary_output_path,
        manifest_output_path=manifest_output_path,
    )

    device = resolve_device(str(cfg.training.device))
    model = build_model(cfg).to(device)
    checkpoint = torch.load(resolved_checkpoint, map_location=device)
    _validate_checkpoint_config(checkpoint.get("config"), cfg)
    model.load_state_dict(checkpoint["model_state_dict"])

    split = str(cfg.data.get("test_split", "test"))
    max_windows = cfg.data.get("max_test_windows", None)
    sample_strategy = str(cfg.data.get("test_sample_strategy", "stratified_random"))
    sample_seed = int(cfg.data.get("test_sample_seed", cfg.training.get("seed", 0)))
    test_data = build_window_data(
        cfg,
        split=split,
        max_windows=max_windows,
        sample_strategy=sample_strategy,
        sample_seed=sample_seed,
        shuffle=False,
    )
    if len(test_data.dataset) == 0:
        raise RuntimeError(f"test 数据为空，请检查 split={split!r}、input_set 和过滤配置。")

    predictions = collect_predictions(
        model,
        test_data.loader,
        device=device,
        max_windows=len(test_data.dataset),
    )
    target_features = load_or_build_target_feature_cache(
        predictions,
        cfg,
        cache_dir=target_cache_dir,
        show_progress=_show_progress(cfg),
    )
    metrics = evaluate_predictions_chunked(
        predictions,
        cfg,
        method=str(cfg.model.name),
        metrics_workers=int(metrics_workers),
        metrics_chunk_size=int(metrics_chunk_size),
        target_features=target_features,
        show_progress=_show_progress(cfg),
    )

    _write_csv(metrics, output_paths.metrics)
    summary = summarize_test_metrics(metrics, split=split, method=str(cfg.model.name))
    _write_csv(summary, output_paths.summary)
    manifest = build_manifest(
        cfg,
        checkpoint_path=resolved_checkpoint,
        config_path=resolved_config,
        outputs=output_paths,
        split=split,
        max_windows=max_windows,
        sample_strategy=sample_strategy,
        sample_seed=sample_seed,
        n_windows=len(metrics),
        device=str(device),
        metrics_workers=int(metrics_workers),
        metrics_chunk_size=int(metrics_chunk_size),
        target_cache_dir="" if target_cache_dir is None else str(target_cache_dir),
    )
    _write_csv(manifest, output_paths.manifest)
    return output_paths


def evaluate_predictions_chunked(
    predictions: dict[str, np.ndarray],
    cfg,
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
                    print(f"  test metrics chunks done {done_count}/{total}", flush=True)
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


def resolve_metrics_workers(value: int, *, task_count: int) -> int:
    return min(max(1, int(value)), max(1, int(task_count)))


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


def _cfg_without_metric_threads(cfg):
    updated = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    if "evaluation" not in updated or updated.evaluation is None:
        updated.evaluation = {}
    updated.evaluation.metric_workers = 1
    return updated


def load_or_build_target_feature_cache(
    predictions: dict[str, np.ndarray],
    cfg,
    *,
    cache_dir: str | Path | None,
    show_progress: bool | None,
) -> dict[str, np.ndarray]:
    """读取或生成 target-side feature cache；cache_dir 为空时只做进程内缓存。"""

    if cache_dir is None:
        return build_target_feature_cache(predictions, cfg, show_progress=show_progress)
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_key = target_feature_cache_key(predictions, cfg)
    cache_path = cache_root / f"{cache_key}.npz"
    if cache_path.exists():
        return load_target_feature_cache(cache_path, expected_key=cache_key)
    features = build_target_feature_cache(predictions, cfg, show_progress=show_progress)
    save_target_feature_cache(cache_path, features, cache_key=cache_key)
    return features


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


def target_feature_cache_key(predictions: dict[str, np.ndarray], cfg) -> str:
    h = hashlib.sha256()
    h.update(b"tho_target_feature_cache_v1")
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
            "local_rr_window_sec": float(cfg.get("evaluation", {}).get("local_rr_window_sec", 20.0)),
            "local_rr_step_sec": float(cfg.get("evaluation", {}).get("local_rr_step_sec", 5.0)),
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


def summarize_test_metrics(metrics: pd.DataFrame, *, split: str, method: str) -> pd.DataFrame:
    record: dict[str, Any] = {
        "split": split,
        "method": method,
        "n_windows": int(len(metrics)),
    }
    for column in (
        "rr_peak_band_abs_error",
        "rr_peak_band_robust_abs_error",
        "rr_spec_abs_error",
        "breath_count_zero_cross_abs_error",
        "relative_envelope_mae",
        "relative_envelope_corr",
        "relative_envelope_mae_lag4s",
        "relative_envelope_corr_lag4s",
        "spectrum_similarity",
        "band_limited_corr",
        "best_lag_corr",
        "best_lag_sec",
        "best_lag_corr_4s",
        "best_lag_sec_4s",
        "local_rr_mae",
        "local_rr_corr",
        "local_rr_valid_frac",
    ):
        if column not in metrics:
            continue
        values = pd.to_numeric(metrics[column], errors="coerce").dropna()
        if values.empty:
            record[f"{column}_mean"] = float("nan")
            record[f"{column}_median"] = float("nan")
            continue
        record[f"{column}_mean"] = float(values.mean())
        record[f"{column}_median"] = float(values.median())
        if column.endswith("_abs_error"):
            record[f"{column}_p95"] = float(values.quantile(0.95))
            record[f"{column}_frac_gt_1"] = float((values.to_numpy() > 1.0).mean())
    return pd.DataFrame([record])


def build_manifest(
    cfg,
    *,
    checkpoint_path: Path,
    config_path: Path,
    outputs: TestEvalOutputs,
    split: str,
    max_windows: Any,
    sample_strategy: str,
    sample_seed: int,
    n_windows: int,
    device: str,
    metrics_workers: int = 1,
    metrics_chunk_size: int = 128,
    target_cache_dir: str = "",
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "checkpoint": str(checkpoint_path),
                "config": str(config_path),
                "dataset_root": str(cfg.data.dataset_root),
                "index_csv": str(cfg.data.index_csv),
                "input_set": str(cfg.data.input_set),
                "split": split,
                "max_windows": "" if max_windows is None else int(max_windows),
                "sample_strategy": sample_strategy,
                "sample_seed": int(sample_seed),
                "n_windows": int(n_windows),
                "model": str(cfg.model.name),
                "device": device,
                "metrics_workers": int(metrics_workers),
                "metrics_chunk_size": int(metrics_chunk_size),
                "target_cache_dir": str(target_cache_dir),
                "metrics_output": str(outputs.metrics),
                "summary_output": str(outputs.summary),
            }
        ]
    )


def _resolve_output_paths(
    checkpoint_path: Path,
    *,
    metrics_output_path: str | Path | None,
    summary_output_path: str | Path | None,
    manifest_output_path: str | Path | None,
) -> TestEvalOutputs:
    run_dir = checkpoint_path.parent
    return TestEvalOutputs(
        metrics=Path(metrics_output_path) if metrics_output_path else run_dir / "test_metrics.csv",
        summary=Path(summary_output_path) if summary_output_path else run_dir / "test_summary.csv",
        manifest=Path(manifest_output_path) if manifest_output_path else run_dir / "test_eval_manifest.csv",
    )


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _show_progress(cfg) -> bool | None:
    value = cfg.training.get("show_progress", None)
    if value in (None, "auto"):
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
        raise ValueError(f"training.show_progress 只能是 true/false/auto，当前为: {value}")
    return bool(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="固定 checkpoint，在 THO test split 上生成最终评价指标")
    parser.add_argument("--config", default="", help="配置文件路径；为空时优先使用 checkpoint 同目录 config.yaml")
    parser.add_argument("--checkpoint", required=True, help="训练产生的 checkpoint.pt 或 checkpoint_topN.pt")
    parser.add_argument("--metrics-output", default="", help="逐窗口指标 CSV；默认写入 checkpoint 同目录 test_metrics.csv")
    parser.add_argument("--summary-output", default="", help="汇总指标 CSV；默认写入 checkpoint 同目录 test_summary.csv")
    parser.add_argument("--manifest-output", default="", help="评价 manifest CSV；默认写入 checkpoint 同目录 test_eval_manifest.csv")
    parser.add_argument("--metrics-workers", type=int, default=1, help="metrics chunk 进程数；1 表示当前进程串行计算")
    parser.add_argument("--metrics-chunk-size", type=int, default=128, help="每个 metrics 进程任务处理的窗口数")
    parser.add_argument("--target-cache-dir", default="", help="target-side feature cache 目录；为空时只使用本进程内缓存")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="OmegaConf dotlist 覆盖，可重复传入")
    args = parser.parse_args()
    if args.metrics_workers < 1:
        raise SystemExit("--metrics-workers 必须 >= 1")
    if args.metrics_chunk_size < 1:
        raise SystemExit("--metrics-chunk-size 必须 >= 1")

    outputs = evaluate_tho_test_checkpoint(
        checkpoint_path=args.checkpoint,
        config_path=args.config or None,
        metrics_output_path=args.metrics_output or None,
        summary_output_path=args.summary_output or None,
        manifest_output_path=args.manifest_output or None,
        overrides=args.overrides,
        metrics_workers=int(args.metrics_workers),
        metrics_chunk_size=int(args.metrics_chunk_size),
        target_cache_dir=args.target_cache_dir or None,
    )
    print(f"写出 test metrics: {outputs.metrics}")
    print(f"写出 test summary: {outputs.summary}")
    print(f"写出 test manifest: {outputs.manifest}")


if __name__ == "__main__":
    main()
