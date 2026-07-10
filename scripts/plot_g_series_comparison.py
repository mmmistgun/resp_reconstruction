from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import traceback
from typing import Iterable
import uuid

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

for _thread_env_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_env_name, "1")

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal

from scripts.export_g_series_comparison_cache import (
    ComparisonSpec,
    REQUIRED_MODELS,
    load_comparison_specs,
    sha256_file,
    spec_fingerprint,
)
from resp_train.metrics.signal import bandpass_filter, local_rr_rate_trace


CANONICAL_METRIC_COLUMNS = (
    "dataset_row_id",
    "split",
    "pred_rr_peak_band_robust_bpm",
    "target_rr_peak_band_robust_bpm",
    "rr_peak_band_robust_abs_error",
    "breath_count_zero_cross_abs_error",
    "best_lag_corr_4s",
    "best_lag_sec_4s",
    "relative_envelope_corr_lag4s",
    "local_rr_mae",
    "local_rr_corr",
    "local_rr_valid_frac",
)

RENDER_METRIC_COLUMNS = (
    "pred_rr_peak_band_robust_bpm",
    "target_rr_peak_band_robust_bpm",
    "rr_peak_band_robust_abs_error",
    "breath_count_zero_cross_bpm_error",
    "best_lag_corr_4s",
    "best_lag_sec_4s",
    "relative_envelope_corr_lag4s",
    "local_rr_mae",
    "local_rr_corr",
    "local_rr_valid_frac",
)

THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


@dataclass(frozen=True)
class CacheArrays:
    root: Path
    dataset_row_id: np.ndarray
    bcg_input: np.ndarray
    tho_ref: np.ndarray
    predictions: dict[str, np.ndarray]


@dataclass(frozen=True)
class RenderTask:
    row_index: int
    dataset_row_id: int
    output_dir: Path
    metrics_by_label: dict[str, dict[str, float]]


@dataclass(frozen=True)
class RenderResult:
    dataset_row_id: int
    status: str
    figure_path: Path


_RENDER_CACHE: CacheArrays | None = None
_RENDER_PARAMS: dict[str, float | int] | None = None


def _validate_cache_ids(values: Iterable[int]) -> pd.Index:
    row_ids = pd.Index(np.asarray(list(values), dtype=np.int64), name="dataset_row_id")
    if row_ids.empty:
        raise ValueError("cache dataset_row_id 不能为空")
    if row_ids.has_duplicates:
        raise ValueError("cache dataset_row_id 存在重复")
    return row_ids


def _validate_metrics_frame(label: str, frame: pd.DataFrame, cache_ids: pd.Index) -> pd.DataFrame:
    missing_columns = sorted(set(CANONICAL_METRIC_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"{label} metrics 缺少 canonical 列: {missing_columns}")
    metrics = frame.loc[:, CANONICAL_METRIC_COLUMNS].copy()
    metrics["dataset_row_id"] = pd.to_numeric(metrics["dataset_row_id"], errors="raise").astype(np.int64)
    if metrics["dataset_row_id"].duplicated().any():
        raise ValueError(f"{label} metrics 的 dataset_row_id 存在重复")
    if not metrics["split"].astype(str).eq("test").all():
        raise ValueError(f"{label} metrics 必须全部来自 test split")
    metric_ids = pd.Index(metrics["dataset_row_id"], name="dataset_row_id")
    missing_ids = cache_ids.difference(metric_ids)
    if not missing_ids.empty:
        raise ValueError(f"{label} metrics 缺少 cache row: {missing_ids[:12].tolist()}")
    extra_ids = metric_ids.difference(cache_ids)
    if not extra_ids.empty:
        raise ValueError(f"{label} metrics 含 cache 外 row: {extra_ids[:12].tolist()}")
    return metrics.set_index("dataset_row_id").reindex(cache_ids)


def align_canonical_metrics(
    cache_row_ids: Iterable[int],
    metrics_by_label: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    cache_ids = _validate_cache_ids(cache_row_ids)
    labels = tuple(metrics_by_label)
    if labels != REQUIRED_MODELS:
        raise ValueError(f"metrics label 必须依次为: {list(REQUIRED_MODELS)}")
    frames: list[pd.DataFrame] = []
    for label in REQUIRED_MODELS:
        frame = _validate_metrics_frame(label, metrics_by_label[label], cache_ids)
        frame["breath_count_zero_cross_bpm_error"] = (
            pd.to_numeric(frame["breath_count_zero_cross_abs_error"], errors="coerce") / 3.0
        )
        frame = frame.drop(columns=["split"])
        frame.columns = [f"{label}__{column}" for column in frame.columns]
        frames.append(frame)
    return pd.concat(frames, axis=1).reindex(cache_ids)


def load_canonical_metrics(
    metrics_dir: str | Path,
    specs: Iterable[ComparisonSpec],
    *,
    cache_row_ids: Iterable[int],
) -> pd.DataFrame:
    resolved_specs = tuple(specs)
    if tuple(spec.label for spec in resolved_specs) != REQUIRED_MODELS:
        raise ValueError("canonical metrics 必须使用固定的四个模型")
    root = Path(metrics_dir)
    frames: dict[str, pd.DataFrame] = {}
    for spec in resolved_specs:
        path = root / f"{spec.label}_{spec.seed}_test_metrics.csv"
        if not path.exists():
            raise FileNotFoundError(f"缺少 canonical test metrics: {path}")
        frames[spec.label] = pd.read_csv(path)
    return align_canonical_metrics(cache_row_ids, frames)


def build_render_tasks(
    cache: CacheArrays,
    aligned_metrics: pd.DataFrame,
    selected_rows: pd.DataFrame,
    *,
    output_dir: str | Path,
) -> tuple[RenderTask, ...]:
    """将经过输入侧过滤的行严格映射为只读缓存的渲染任务。"""
    cache_ids = _validate_cache_ids(cache.dataset_row_id)
    if not aligned_metrics.index.equals(cache_ids):
        raise ValueError("aligned metrics 必须按 cache dataset_row_id 完整对齐")
    required_selected_columns = {"dataset_row_id", "plot_status"}
    missing_selected_columns = sorted(required_selected_columns - set(selected_rows.columns))
    if missing_selected_columns:
        raise ValueError(f"过滤结果缺少列: {missing_selected_columns}")
    selected = selected_rows.loc[:, ["dataset_row_id", "plot_status"]].copy()
    selected["dataset_row_id"] = pd.to_numeric(selected["dataset_row_id"], errors="raise").astype(np.int64)
    if selected["dataset_row_id"].duplicated().any():
        raise ValueError("过滤结果 dataset_row_id 存在重复")
    selected_ids = pd.Index(selected["dataset_row_id"], name="dataset_row_id")
    if not cache_ids.difference(selected_ids).empty or not selected_ids.difference(cache_ids).empty:
        raise ValueError("过滤结果必须覆盖且仅覆盖 cache dataset_row_id")
    invalid_status = sorted(set(selected["plot_status"].astype(str)) - {"retained", "input_stable_excluded"})
    if invalid_status:
        raise ValueError(f"未知 plot_status: {invalid_status}")
    required_metric_columns = {
        f"{label}__{metric}"
        for label in REQUIRED_MODELS
        for metric in RENDER_METRIC_COLUMNS
    }
    missing_metric_columns = sorted(required_metric_columns - set(aligned_metrics.columns))
    if missing_metric_columns:
        raise ValueError(f"aligned metrics 缺少渲染指标列: {missing_metric_columns}")
    selected_by_id = selected.set_index("dataset_row_id")["plot_status"]
    cache_position_by_id = {int(dataset_row_id): position for position, dataset_row_id in enumerate(cache_ids)}
    tasks: list[RenderTask] = []
    for dataset_row_id in sorted(int(value) for value in cache_ids):
        if selected_by_id.loc[dataset_row_id] != "retained":
            continue
        row = aligned_metrics.loc[dataset_row_id]
        metrics_by_label = {
            label: {
                metric: float(row[f"{label}__{metric}"])
                for metric in RENDER_METRIC_COLUMNS
            }
            for label in REQUIRED_MODELS
        }
        tasks.append(
            RenderTask(
                row_index=cache_position_by_id[dataset_row_id],
                dataset_row_id=int(dataset_row_id),
                output_dir=Path(output_dir),
                metrics_by_label=metrics_by_label,
            )
        )
    return tuple(tasks)


def _cache_array_path(root: Path, label: str | None = None) -> Path:
    if label is None:
        raise ValueError("cache array label 不能为空")
    return root / "predictions" / label / "r_tho_hat.npy"


def load_cache(
    cache_dir: str | Path,
    specs: Iterable[ComparisonSpec],
    *,
    verify_values: bool = True,
) -> CacheArrays:
    root = Path(cache_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"缓存缺少 manifest.json: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"缓存 manifest 必须为 complete: {manifest_path}")
    resolved_specs = tuple(specs)
    if tuple(spec.label for spec in resolved_specs) != REQUIRED_MODELS:
        raise ValueError("缓存加载必须使用固定的四个模型")
    if manifest.get("spec_fingerprint") != spec_fingerprint(resolved_specs):
        raise ValueError("缓存 manifest 的 spec_fingerprint 与当前 spec 不一致")
    manifest_specs = manifest.get("specs")
    if manifest_specs is not None:
        actual_specs = tuple(
            (
                str(item.get("label", "")),
                int(item.get("seed")),
                str(item.get("checkpoint", "")),
            )
            for item in manifest_specs
        )
        expected_specs = tuple(
            (spec.label, int(spec.seed), str(spec.checkpoint))
            for spec in resolved_specs
        )
        if actual_specs != expected_specs:
            raise ValueError("缓存 manifest 的 label/seed/checkpoint 与当前 spec 不一致")
    required_arrays = {"dataset_row_id", "bcg_input", "tho_ref", *(f"prediction:{label}" for label in REQUIRED_MODELS)}
    arrays_manifest = manifest.get("arrays", {})
    missing_arrays = sorted(required_arrays - set(arrays_manifest))
    if missing_arrays:
        raise ValueError(f"缓存 manifest 缺少数组声明: {missing_arrays}")
    row_ids = np.load(root / "dataset_row_id.npy", mmap_mode="r")
    bcg = np.load(root / "bcg_input.npy", mmap_mode="r")
    target = np.load(root / "tho_ref.npy", mmap_mode="r")
    ids = _validate_cache_ids(row_ids)
    bcg_values = np.asarray(bcg)
    target_values = np.asarray(target)
    if bcg_values.ndim != 2 or target_values.ndim != 2 or bcg_values.shape != target_values.shape:
        raise ValueError("缓存 BCG/THO 必须为相同 shape 的二维数组")
    if bcg_values.shape[0] != ids.size:
        raise ValueError("缓存 BCG 行数与 dataset_row_id 不一致")
    if verify_values and (not np.isfinite(bcg_values).all() or not np.isfinite(target_values).all()):
        raise ValueError("缓存 BCG/THO 包含非有限值")
    predictions: dict[str, np.ndarray] = {}
    for label in REQUIRED_MODELS:
        path = _cache_array_path(root, label)
        if not path.exists():
            raise FileNotFoundError(f"缓存缺少预测数组: {path}")
        values = np.load(path, mmap_mode="r")
        if values.shape != target_values.shape:
            raise ValueError(f"{label} 预测 shape 与 THO 不一致")
        if verify_values and not np.isfinite(np.asarray(values)).all():
            raise ValueError(f"{label} 预测包含非有限值")
        predictions[label] = values
    return CacheArrays(root=root, dataset_row_id=row_ids, bcg_input=bcg, tho_ref=target, predictions=predictions)


def initialize_render_worker(
    cache_dir: str | Path,
    specs: Iterable[ComparisonSpec],
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int,
) -> None:
    for name in THREAD_ENV:
        os.environ[name] = "1"
    global _RENDER_CACHE, _RENDER_PARAMS
    _RENDER_CACHE = load_cache(cache_dir, specs, verify_values=False)
    _RENDER_PARAMS = {
        "fs": float(fs),
        "low_hz": float(low_hz),
        "high_hz": float(high_hz),
        "order": int(order),
    }


def resolve_workers(value: str | int, *, n_tasks: int) -> int:
    if n_tasks <= 0:
        raise ValueError("待绘制任务数必须大于 0")
    if str(value).lower() == "auto":
        return min(48, max(1, (os.cpu_count() or 1) - 2), n_tasks)
    workers = int(value)
    if workers < 1:
        raise ValueError("--workers 必须 >= 1")
    return min(workers, n_tasks)


def _panel_limits(*signals: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([np.asarray(signal, dtype=np.float64).reshape(-1) for signal in signals])
    low, high = np.percentile(values, [0.5, 99.5])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        center = float(np.mean(values))
        return center - 1.0, center + 1.0
    margin = 0.05 * (high - low)
    return float(low - margin), float(high + margin)


def _normalized_band_psd(values: np.ndarray, *, fs: float, low_hz: float, high_hz: float) -> tuple[np.ndarray, np.ndarray]:
    freqs, power = scipy_signal.welch(np.asarray(values, dtype=np.float64), fs=fs, nperseg=min(4096, len(values)))
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    band_power = np.asarray(power[mask], dtype=np.float64)
    return freqs[mask], band_power / max(float(band_power.sum()), np.finfo(np.float64).eps)


def _format_metric(value: float | int | None, *, digits: int = 3) -> str:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "nan"
    return f"{number:.{digits}f}" if np.isfinite(number) else "nan"


def render_one_window(task: RenderTask) -> RenderResult:
    if _RENDER_CACHE is None or _RENDER_PARAMS is None:
        raise RuntimeError("绘图 worker 尚未初始化")
    if task.row_index < 0 or task.row_index >= len(_RENDER_CACHE.dataset_row_id):
        raise IndexError(f"row_index 越界: {task.row_index}")
    if int(_RENDER_CACHE.dataset_row_id[task.row_index]) != int(task.dataset_row_id):
        raise ValueError("RenderTask dataset_row_id 与缓存顺序不一致")
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    from matplotlib.gridspec import GridSpec

    fs = float(_RENDER_PARAMS["fs"])
    low_hz = float(_RENDER_PARAMS["low_hz"])
    high_hz = float(_RENDER_PARAMS["high_hz"])
    order = int(_RENDER_PARAMS["order"])
    bcg = np.asarray(_RENDER_CACHE.bcg_input[task.row_index], dtype=np.float64)
    tho = np.asarray(_RENDER_CACHE.tho_ref[task.row_index], dtype=np.float64)
    predictions = {label: np.asarray(values[task.row_index], dtype=np.float64) for label, values in _RENDER_CACHE.predictions.items()}
    bcg_band = bandpass_filter(bcg, fs=fs, low_hz=low_hz, high_hz=high_hz, order=order)
    time = np.arange(bcg.size, dtype=np.float64) / fs
    task.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = task.output_dir / f"row_{int(task.dataset_row_id)}.png"
    if output_path.exists():
        raise FileExistsError(f"PNG 已存在，拒绝覆盖: {output_path}")
    figure = plt.figure(figsize=(18, 17))
    grid = GridSpec(4, 1, figure=figure, height_ratios=(1.0, 1.2, 1.0, 1.35))
    input_ax = figure.add_subplot(grid[0])
    output_ax = figure.add_subplot(grid[1], sharex=input_ax)
    spectrum_ax = figure.add_subplot(grid[2])
    table_ax = figure.add_subplot(grid[3])
    input_ax.plot(time, bcg, color="#7f7f7f", linewidth=0.55, label="BCG soft-z input")
    input_ax.plot(time, bcg_band, color="#1f77b4", linewidth=0.9, label="BCG resp-band (0.05–0.7 Hz)")
    input_ax.set_ylim(*_panel_limits(bcg, bcg_band))
    input_ax.set_ylabel("soft-z")
    input_ax.legend(loc="upper right")
    input_ax.grid(alpha=0.2)
    output_ax.plot(time, tho, color="#111111", linewidth=1.35, label="THO target")
    colors = {"g0_time_only": "#7f7f7f", "g0_f0_native_stft_pre_mixer": "#ff7f0e", "g3_c_wide_8p0": "#1f77b4", "g3_c_bandenergy": "#2ca02c"}
    for label in REQUIRED_MODELS:
        output_ax.plot(time, predictions[label], color=colors[label], linewidth=0.75, alpha=0.85, label=label)
    output_ax.set_ylim(*_panel_limits(tho, *predictions.values()))
    output_ax.set_ylabel("soft-z")
    output_ax.set_xlabel("time (s)")
    output_ax.legend(loc="upper right", ncol=2)
    output_ax.grid(alpha=0.2)
    for label, values, color in (("BCG resp-band", bcg_band, "#1f77b4"), ("THO target", tho, "#111111"), *((label, predictions[label], colors[label]) for label in REQUIRED_MODELS)):
        freqs, power = _normalized_band_psd(values, fs=fs, low_hz=low_hz, high_hz=high_hz)
        spectrum_ax.plot(freqs, power, color=color, linewidth=1.0, label=label)
    spectrum_ax.set_xlim(low_hz, high_hz)
    spectrum_ax.set_ylabel("normalized band power")
    spectrum_ax.set_xlabel("frequency (Hz)")
    spectrum_ax.legend(loc="upper right", ncol=2)
    spectrum_ax.grid(alpha=0.2)
    table_ax.axis("off")
    row_labels = ["robust RR pred/target/error", "count bpm error", "lag4 corr / sec", "relative env corr", "local RR MAE/corr/valid"]
    table_values = []
    for row_index in range(len(row_labels)):
        row: list[str] = []
        for label in REQUIRED_MODELS:
            metrics = task.metrics_by_label[label]
            if row_index == 0:
                row.append(" / ".join(_format_metric(metrics.get(key), digits=2) for key in ("pred_rr_peak_band_robust_bpm", "target_rr_peak_band_robust_bpm", "rr_peak_band_robust_abs_error")))
            elif row_index == 1:
                row.append(_format_metric(metrics.get("breath_count_zero_cross_bpm_error")))
            elif row_index == 2:
                row.append(" / ".join(_format_metric(metrics.get(key)) for key in ("best_lag_corr_4s", "best_lag_sec_4s")))
            elif row_index == 3:
                row.append(_format_metric(metrics.get("relative_envelope_corr_lag4s")))
            else:
                row.append(" / ".join(_format_metric(metrics.get(key)) for key in ("local_rr_mae", "local_rr_corr", "local_rr_valid_frac")))
        table_values.append(row)
    table_ax.table(cellText=table_values, rowLabels=row_labels, colLabels=list(REQUIRED_MODELS), loc="center", cellLoc="center")
    figure.suptitle(f"G-series comparison | test row={int(task.dataset_row_id)}", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    temporary_path = output_path.with_name(f".{output_path.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp.png")
    try:
        figure.savefig(temporary_path, dpi=140)
        os.replace(temporary_path, output_path)
    finally:
        plt.close(figure)
        if temporary_path.exists():
            temporary_path.unlink()
    return RenderResult(dataset_row_id=int(task.dataset_row_id), status="written", figure_path=output_path)


def compute_input_stability_frame(
    cache: CacheArrays,
    *,
    specs: Iterable[ComparisonSpec],
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int,
    workers: int,
) -> pd.DataFrame:
    """从完整共享 BCG 缓存构建输入侧稳定度表，不读取目标、预测或指标。"""
    if workers < 1:
        raise ValueError("输入稳定度 workers 必须 >= 1")
    resolved_specs = tuple(specs)
    row_indices = tuple(range(len(cache.dataset_row_id)))
    if workers == 1:
        rows = [
            {
                "dataset_row_id": int(cache.dataset_row_id[row_index]),
                **input_stability_features(
                    cache.bcg_input[row_index],
                    fs=fs,
                    low_hz=low_hz,
                    high_hz=high_hz,
                    order=order,
                ),
            }
            for row_index in row_indices
        ]
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(int(workers), len(row_indices)),
            mp_context=context,
            initializer=initialize_render_worker,
            initargs=(str(cache.root), resolved_specs, fs, low_hz, high_hz, order),
        ) as executor:
            rows = list(executor.map(_compute_input_stability_row, row_indices))
    return pd.DataFrame(rows)


def _compute_input_stability_row(row_index: int) -> dict[str, float | int]:
    if _RENDER_CACHE is None or _RENDER_PARAMS is None:
        raise RuntimeError("输入稳定度 worker 尚未初始化")
    if row_index < 0 or row_index >= len(_RENDER_CACHE.dataset_row_id):
        raise IndexError(f"row_index 越界: {row_index}")
    return {
        "dataset_row_id": int(_RENDER_CACHE.dataset_row_id[row_index]),
        **input_stability_features(
            _RENDER_CACHE.bcg_input[row_index],
            fs=float(_RENDER_PARAMS["fs"]),
            low_hz=float(_RENDER_PARAMS["low_hz"]),
            high_hz=float(_RENDER_PARAMS["high_hz"]),
            order=int(_RENDER_PARAMS["order"]),
        ),
    }


def _render_parameters(spec: ComparisonSpec) -> dict[str, float | int]:
    from resp_train.config import load_config

    config_path = spec.checkpoint.parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"checkpoint 同目录缺少 config.yaml: {config_path}")
    cfg = load_config(config_path)
    return {
        "fs": float(cfg.window.target_fs),
        "low_hz": float(cfg.loss.spectrum_low_hz),
        "high_hz": float(cfg.loss.spectrum_high_hz),
        "order": int(cfg.get("evaluation", {}).get("lag_bandpass_order", 4)),
    }


def _atomic_write_text(path: str | Path, content: str) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary_path.write_text(content, encoding="utf-8")
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return output_path


def _atomic_write_csv(path: str | Path, frame: pd.DataFrame) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp.csv")
    try:
        frame.to_csv(temporary_path, index=False)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return output_path


def _run_render_tasks(
    tasks: Iterable[RenderTask],
    *,
    cache_dir: str | Path,
    specs: Iterable[ComparisonSpec],
    workers: int,
    render_params: dict[str, float | int],
) -> tuple[list[RenderResult], list[dict[str, object]]]:
    requested_tasks = tuple(tasks)
    if not requested_tasks:
        return [], []
    context = mp.get_context("spawn")
    results: list[RenderResult] = []
    failures: list[dict[str, object]] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=initialize_render_worker,
        initargs=(
            str(cache_dir),
            tuple(specs),
            float(render_params["fs"]),
            float(render_params["low_hz"]),
            float(render_params["high_hz"]),
            int(render_params["order"]),
        ),
    ) as executor:
        futures = {executor.submit(render_one_window, task): task for task in requested_tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:  # pragma: no cover - 依赖多进程远端异常的具体包装方式。
                failures.append(
                    {
                        "dataset_row_id": int(task.dataset_row_id),
                        "row_index": int(task.row_index),
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": "".join(traceback.format_exception(exc)),
                    }
                )
    return (
        sorted(results, key=lambda result: result.dataset_row_id),
        sorted(failures, key=lambda failure: int(failure["dataset_row_id"])),
    )


def _build_window_index(
    selected_rows: pd.DataFrame,
    requested_tasks: Iterable[RenderTask],
    results: Iterable[RenderResult],
    failures: Iterable[dict[str, object]],
) -> pd.DataFrame:
    index = selected_rows.copy().sort_values("dataset_row_id").reset_index(drop=True)
    requested_ids = {int(task.dataset_row_id) for task in requested_tasks}
    result_by_id = {int(result.dataset_row_id): result for result in results}
    failure_by_id = {int(failure["dataset_row_id"]): failure for failure in failures}
    statuses: list[str] = []
    figure_paths: list[str] = []
    errors: list[str] = []
    for row in index.itertuples(index=False):
        dataset_row_id = int(row.dataset_row_id)
        if dataset_row_id in result_by_id:
            statuses.append("written")
            figure_paths.append(str(result_by_id[dataset_row_id].figure_path))
            errors.append("")
        elif dataset_row_id in failure_by_id:
            statuses.append("failed")
            figure_paths.append("")
            errors.append(str(failure_by_id[dataset_row_id]["message"]))
        elif getattr(row, "plot_status") != "retained":
            statuses.append("input_stable_excluded")
            figure_paths.append("")
            errors.append("")
        elif dataset_row_id not in requested_ids:
            statuses.append("not_requested_by_max_plots")
            figure_paths.append("")
            errors.append("")
        else:
            statuses.append("missing_result")
            figure_paths.append("")
            errors.append("任务没有返回成功或失败记录")
    index["render_status"] = statuses
    index["figure_path"] = figure_paths
    index["error"] = errors
    return index


def _plot_summary(
    window_index: pd.DataFrame,
    *,
    filter_mode: str,
    stable_fraction: float,
    max_plots: int | None,
    worker_count: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "filter": filter_mode,
                "stable_fraction": float(stable_fraction),
                "max_plots": max_plots,
                "worker_count": int(worker_count),
                "n_total": int(len(window_index)),
                "n_retained": int(window_index["plot_status"].eq("retained").sum()),
                "n_input_stable_excluded": int(window_index["plot_status"].eq("input_stable_excluded").sum()),
                "n_requested": int(window_index["render_status"].isin(["written", "failed", "missing_result"]).sum()),
                "n_written": int(window_index["render_status"].eq("written").sum()),
                "n_failed": int(window_index["render_status"].eq("failed").sum()),
            }
        ]
    )


def _metric_hashes(metrics_dir: str | Path, specs: Iterable[ComparisonSpec]) -> dict[str, str]:
    root = Path(metrics_dir)
    hashes = {}
    for spec in specs:
        path = root / f"{spec.label}_{spec.seed}_test_metrics.csv"
        if not path.exists():
            raise FileNotFoundError(f"缺少 canonical test metrics: {path}")
        hashes[spec.label] = sha256_file(path)
    return hashes


def run_plot_comparison(
    *,
    cache_dir: str | Path,
    metrics_dir: str | Path,
    output_dir: str | Path,
    specs: Iterable[ComparisonSpec],
    filter_mode: str,
    stable_fraction: float,
    workers: str | int,
    max_plots: int | None,
) -> Path:
    """执行完整 CPU 绘图阶段；出现任一失败时保留失败清单但不发布 complete manifest。"""
    resolved_specs = tuple(specs)
    output_path = Path(output_dir)
    if output_path.exists():
        raise FileExistsError(f"绘图输出目录已存在，拒绝覆盖: {output_path}")
    if max_plots is not None and max_plots < 1:
        raise ValueError("--max-plots 必须 >= 1")
    cache = load_cache(cache_dir, resolved_specs)
    aligned_metrics = load_canonical_metrics(metrics_dir, resolved_specs, cache_row_ids=cache.dataset_row_id)
    render_params = _render_parameters(resolved_specs[0])
    feature_worker_count = resolve_workers(workers, n_tasks=len(cache.dataset_row_id))
    feature_frame = compute_input_stability_frame(
        cache,
        specs=resolved_specs,
        workers=feature_worker_count,
        **render_params,
    )
    selected_rows = select_plot_rows(
        feature_frame,
        filter_mode=filter_mode,
        stable_fraction=stable_fraction,
    )
    all_tasks = build_render_tasks(cache, aligned_metrics, selected_rows, output_dir=output_path / "figures")
    requested_tasks = all_tasks if max_plots is None else all_tasks[:max_plots]
    worker_count = resolve_workers(workers, n_tasks=len(requested_tasks)) if requested_tasks else 0
    output_path.mkdir(parents=True, exist_ok=False)
    results, failures = _run_render_tasks(
        requested_tasks,
        cache_dir=cache.root,
        specs=resolved_specs,
        workers=worker_count,
        render_params=render_params,
    )
    window_index = _build_window_index(selected_rows, requested_tasks, results, failures)
    _atomic_write_csv(output_path / "window_index.csv", window_index)
    _atomic_write_csv(
        output_path / "filter_summary.csv",
        _plot_summary(
            window_index,
            filter_mode=filter_mode,
            stable_fraction=stable_fraction,
            max_plots=max_plots,
            worker_count=worker_count,
        ),
    )
    cache_manifest_path = cache.root / "manifest.json"
    payload: dict[str, object] = {
        "schema_version": 1,
        "cache_dir": str(cache.root),
        "cache_manifest_sha256": sha256_file(cache_manifest_path),
        "spec_fingerprint": spec_fingerprint(resolved_specs),
        "metrics_sha256": _metric_hashes(metrics_dir, resolved_specs),
        "filter": filter_mode,
        "stable_fraction": float(stable_fraction),
        "max_plots": max_plots,
        "feature_worker_count": feature_worker_count,
        "worker_count": worker_count,
        "render_parameters": render_params,
        "n_total": int(len(window_index)),
        "n_requested": int(len(requested_tasks)),
        "n_written": int(len(results)),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if failures:
        payload["status"] = "failed"
        payload["failures"] = failures
        _atomic_write_text(
            output_path / "plot_failure_manifest.json",
            json.dumps(payload, ensure_ascii=False, indent=2),
        )
        raise RuntimeError(f"{len(failures)} 个绘图任务失败；详见 {output_path / 'plot_failure_manifest.json'}")
    payload["status"] = "complete"
    _atomic_write_text(
        output_path / "plot_manifest.json",
        json.dumps(payload, ensure_ascii=False, indent=2),
    )
    return output_path


def select_plot_rows(
    features: pd.DataFrame,
    *,
    filter_mode: str,
    stable_fraction: float,
) -> pd.DataFrame:
    required = {
        "dataset_row_id",
        "spectral_peak_fraction",
        "local_rr_valid_frac",
        "local_rr_iqr_bpm",
    }
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"输入稳定度特征缺少列: {missing}")
    if not 0.0 < float(stable_fraction) < 1.0:
        raise ValueError("stable_fraction 必须在 (0, 1) 内")
    result = features.copy()
    result["dataset_row_id"] = pd.to_numeric(result["dataset_row_id"], errors="raise").astype(np.int64)
    if result["dataset_row_id"].duplicated().any():
        raise ValueError("输入稳定度特征的 dataset_row_id 存在重复")
    for column in ("spectral_peak_fraction", "local_rr_valid_frac", "local_rr_iqr_bpm"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["input_stability_score"] = (
        0.50 * result["spectral_peak_fraction"].rank(method="average", pct=True)
        + 0.30 * result["local_rr_valid_frac"].rank(method="average", pct=True)
        + 0.20 * (1.0 - result["local_rr_iqr_bpm"].rank(method="average", pct=True))
    )
    if filter_mode == "all":
        result["plot_status"] = "retained"
        return result.sort_values("dataset_row_id").reset_index(drop=True)
    if filter_mode != "exclude-input-stable":
        raise ValueError("filter_mode 只能是 all 或 exclude-input-stable")
    n_excluded = int(math.ceil(float(stable_fraction) * len(result)))
    stable_ids = set(
        result.sort_values(
            ["input_stability_score", "dataset_row_id"], ascending=[False, True]
        ).head(n_excluded)["dataset_row_id"].tolist()
    )
    result["plot_status"] = np.where(
        result["dataset_row_id"].isin(stable_ids),
        "input_stable_excluded",
        "retained",
    )
    return result.sort_values("dataset_row_id").reset_index(drop=True)


def input_stability_features(
    bcg: np.ndarray,
    *,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int,
) -> dict[str, float]:
    values = np.asarray(bcg, dtype=np.float64).reshape(-1)
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError("BCG 输入必须是一维有限信号")
    filtered = bandpass_filter(values, fs=fs, low_hz=low_hz, high_hz=high_hz, order=order)
    freqs, power = scipy_signal.welch(filtered, fs=fs, nperseg=min(4096, filtered.size))
    in_band = (freqs >= float(low_hz)) & (freqs <= float(high_hz))
    if not np.any(in_band):
        raise ValueError("Welch 频率网格未覆盖呼吸带")
    band_power = np.asarray(power[in_band], dtype=np.float64)
    peak_fraction = float(band_power.max() / max(float(band_power.sum()), np.finfo(np.float64).eps))
    rates = local_rr_rate_trace(
        filtered,
        fs=fs,
        window_sec=40.0,
        step_sec=10.0,
        low_hz=low_hz,
        high_hz=high_hz,
    )
    valid = rates[np.isfinite(rates)]
    iqr = float(np.subtract(*np.percentile(valid, [75, 25]))) if valid.size >= 2 else float("inf")
    return {
        "spectral_peak_fraction": peak_fraction,
        "local_rr_valid_frac": float(valid.size / rates.size),
        "local_rr_iqr_bpm": iqr,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="并行绘制 G 系列四模型测试窗口对比图")
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("configs/eval_specs/g_series_four_model_visualization.csv"),
        help="冻结的四模型 checkpoint 清单",
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--filter",
        choices=("all", "exclude-input-stable"),
        default="exclude-input-stable",
        help="仅由 BCG 输入决定的选图过滤规则",
    )
    parser.add_argument("--stable-fraction", type=float, default=0.20)
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--max-plots", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = load_comparison_specs(args.spec)
    output_path = run_plot_comparison(
        cache_dir=args.cache_dir,
        metrics_dir=args.metrics_dir,
        output_dir=args.output_dir,
        specs=specs,
        filter_mode=str(args.filter),
        stable_fraction=float(args.stable_fraction),
        workers=args.workers,
        max_plots=args.max_plots,
    )
    print(f"complete plot manifest: {output_path / 'plot_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
