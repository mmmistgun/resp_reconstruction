from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.config import load_config
from resp_train.data.factory import build_window_data
from resp_train.engine.train import _batch_sst, _extract_meta
from resp_train.experiments.tho import _validate_checkpoint_config
from resp_train.metrics.evaluate import evaluate_prediction_dict
from resp_train.models.registry import build_model
from resp_train.models.stft_branch import align_to_time
from resp_train.utils.run import resolve_device

DEFAULT_MODES = ("normal", "stft_zero", "stft_shuffle_batch", "stft_shuffle_time")
SUPPORTED_MODES = (*DEFAULT_MODES, "time_zero")
_METRICS_PREDICTIONS_BY_MODE: dict[str, dict[str, np.ndarray]] | None = None


@dataclass(frozen=True)
class RunSpec:
    run_dir: Path
    checkpoint_path: Path
    config_path: Path
    mode: str
    metrics_output: Path

    @property
    def tag(self) -> str:
        return f"{self.run_dir.parents[1].name}_{self.run_dir.parent.name}_{self.run_dir.name}_{self.mode}"


@dataclass(frozen=True)
class MetricChunkTask:
    mode: str
    start: int
    end: int
    method: str


def _checkpoint_suffix(checkpoint_name: str) -> str:
    stem = Path(checkpoint_name).stem
    if stem.startswith("checkpoint_"):
        return stem.removeprefix("checkpoint_")
    if stem == "checkpoint":
        return "best"
    return stem


def discover_run_specs(
    runs_root: Path,
    *,
    arm: str,
    branch: str,
    checkpoint_name: str,
    modes: Sequence[str],
    force: bool,
) -> list[RunSpec]:
    """发现待做 E3-C0 消融的 run/mode 任务。"""

    branch_root = runs_root / arm / branch
    if not branch_root.exists():
        raise FileNotFoundError(f"run 分支目录不存在: {branch_root}")
    suffix = _checkpoint_suffix(checkpoint_name)
    specs: list[RunSpec] = []
    for run_dir in sorted(path for path in branch_root.iterdir() if path.is_dir()):
        config_path = run_dir / "config.yaml"
        checkpoint_path = run_dir / checkpoint_name
        if not config_path.exists() or not checkpoint_path.exists():
            continue
        for mode in modes:
            output = run_dir / f"metrics_e3c_{mode}_{suffix}.csv"
            if output.exists() and not force:
                continue
            specs.append(
                RunSpec(
                    run_dir=run_dir,
                    checkpoint_path=checkpoint_path,
                    config_path=config_path,
                    mode=mode,
                    metrics_output=output,
                )
            )
    return specs


def _validate_modes(modes: Sequence[str]) -> list[str]:
    normalized = [str(mode).strip().lower() for mode in modes]
    if not normalized:
        raise ValueError("至少需要一个消融 mode")
    unknown = sorted(set(normalized) - set(SUPPORTED_MODES))
    if unknown:
        raise ValueError(f"未知消融 mode: {unknown}，可选: {', '.join(SUPPORTED_MODES)}")
    return normalized


def _validate_concat_dual_model(model: torch.nn.Module) -> None:
    if getattr(model, "branch_mode", None) != "dual":
        raise ValueError("E3-C 消融当前只支持 branch_mode=dual 的 checkpoint")
    if getattr(model, "fusion_mode", None) != "concat_generic":
        raise ValueError("当前模型不是 concat_generic，不能走 concat 消融路径")
    if getattr(model, "time_backbone", None) is None:
        raise ValueError("模型缺少 time_backbone，无法做 time/STFT 贡献拆分")
    if getattr(model, "stft_encoder", None) is None:
        raise ValueError("模型缺少 stft_encoder，无法做 STFT 消融")
    if getattr(model, "fusion_head", None) is None:
        raise ValueError("模型缺少 fusion_head，无法复用 concat 融合路径")


def _validate_native_dual_model(model: torch.nn.Module) -> None:
    if getattr(model, "branch_mode", None) != "dual":
        raise ValueError("E3-C 消融当前只支持 branch_mode=dual 的 checkpoint")
    if getattr(model, "fusion_mode", None) not in {"native_inject", "token_context_inject"}:
        raise ValueError("当前模型不是 native/token 注入模式，不能走 token 消融路径")
    if getattr(model, "time_backbone", None) is None:
        raise ValueError("模型缺少 time_backbone，无法做 token 注入消融")
    if getattr(model, "stft_encoder", None) is None:
        raise ValueError("模型缺少 stft_encoder，无法做 STFT 消融")
    if not hasattr(model.time_backbone, "forward_with_token_injection"):
        raise ValueError("time_backbone 缺少 forward_with_token_injection，无法复用原生注入路径")
    if not hasattr(model.time_backbone, "token_count_for_length"):
        raise ValueError("time_backbone 缺少 token_count_for_length，无法对齐 STFT token")
    if not hasattr(model, "_encode_stft_features") or not hasattr(model, "_project_stft_features"):
        raise ValueError("模型缺少 STFT token 编码/投影方法，无法构造 token_delta 消融")


@torch.no_grad()
def collect_ablation_predictions(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, Any]],
    *,
    device: torch.device | str,
    max_windows: int,
    modes: Sequence[str],
    shuffle_seed: int,
    progress_every: int | None = None,
    progress_label: str = "",
) -> dict[str, dict[str, np.ndarray]]:
    """按融合模式分发消融收集逻辑。

    C0 的 concat 路径消融的是 fusion_head 输入特征；C2 的 native 路径消融的是已经
    投影到 patch token 栅格的 token_delta。两者语义不同，不能混用同一内部路径。
    """

    fusion_mode = str(getattr(model, "fusion_mode", "")).lower()
    if fusion_mode == "concat_generic":
        return collect_concat_ablation_predictions(
            model,
            loader,
            device=device,
            max_windows=max_windows,
            modes=modes,
            shuffle_seed=shuffle_seed,
            progress_every=progress_every,
            progress_label=progress_label,
        )
    if fusion_mode in {"native_inject", "token_context_inject"}:
        return collect_native_ablation_predictions(
            model,
            loader,
            device=device,
            max_windows=max_windows,
            modes=modes,
            shuffle_seed=shuffle_seed,
            progress_every=progress_every,
            progress_label=progress_label,
        )
    raise ValueError(f"不支持的 fusion_mode: {fusion_mode}")


@torch.no_grad()
def collect_concat_ablation_predictions(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, Any]],
    *,
    device: torch.device | str,
    max_windows: int,
    modes: Sequence[str],
    shuffle_seed: int,
    progress_every: int | None = None,
    progress_label: str = "",
) -> dict[str, dict[str, np.ndarray]]:
    """单次遍历 dataloader，同时收集多个 STFT 消融模式的预测。

    这里直接复用 concat_generic 的内部数据流：每个 batch 只计算一次 time branch 和
    STFT branch 特征，再在内存中构造消融特征，避免为每个 mode 重建 dataloader。
    """

    if int(max_windows) <= 0:
        raise ValueError("max_windows 必须大于 0")
    selected_modes = _validate_modes(modes)
    _validate_concat_dual_model(model)

    resolved_device = torch.device(device)
    model.to(resolved_device)
    model.eval()
    non_blocking = resolved_device.type == "cuda"
    generator = torch.Generator(device="cpu").manual_seed(int(shuffle_seed))

    pred_parts: dict[str, list[np.ndarray]] = {mode: [] for mode in selected_modes}
    targets: list[np.ndarray] = []
    meta_records: list[dict[str, Any]] = []
    collected = 0
    batch_idx = 0
    total_data_wait_sec = 0.0
    total_compute_sec = 0.0

    loader_iter = iter(loader)
    while len(meta_records) < int(max_windows):
        data_start = time.perf_counter()
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        data_wait_sec = time.perf_counter() - data_start
        compute_start = time.perf_counter()
        batch_idx += 1
        if "meta" not in batch:
            raise KeyError("batch 必须包含 meta")
        x = batch["x"].to(resolved_device, non_blocking=non_blocking)
        sst = _batch_sst(batch, resolved_device, non_blocking=non_blocking)
        time_feats, _ = model.time_backbone(x, return_features=True)
        branch_feats = model.stft_encoder(sst) if getattr(model, "sst_cached", False) else model.stft_encoder(x)
        time_feats = align_to_time(time_feats, int(model.fuse_len))
        stft_feats = align_to_time(branch_feats, int(model.fuse_len))

        for mode in selected_modes:
            pred = _predict_with_ablation(model, time_feats, stft_feats, mode=mode, generator=generator)
            pred_parts[mode].append(pred.detach().cpu().numpy())

        targets.append(batch["target"].detach().cpu().numpy())
        batch_size = x.size(0)
        for idx in range(batch_size):
            meta_records.append(_extract_meta(batch["meta"], idx))
        collected = min(len(meta_records), int(max_windows))
        compute_sec = time.perf_counter() - compute_start
        total_data_wait_sec += data_wait_sec
        total_compute_sec += compute_sec
        if progress_every and (batch_idx == 1 or batch_idx % int(progress_every) == 0 or collected >= int(max_windows)):
            label = f" {progress_label}" if progress_label else ""
            print(
                f"  collect{label} batch={batch_idx} windows={collected}/{int(max_windows)} "
                f"data_wait={data_wait_sec:.3f}s compute={compute_sec:.3f}s "
                f"total_data_wait={total_data_wait_sec:.1f}s total_compute={total_compute_sec:.1f}s",
                flush=True,
            )

    if not meta_records:
        raise RuntimeError("没有可收集的预测窗口")

    limit = int(max_windows)
    target_arr = np.concatenate(targets, axis=0)[:limit]
    meta_records = meta_records[:limit]
    outputs: dict[str, dict[str, np.ndarray]] = {}
    for mode, parts in pred_parts.items():
        pred_arr = np.concatenate(parts, axis=0)[:limit]
        outputs[mode] = _prediction_dict(pred_arr, target_arr, meta_records)
    return outputs


@torch.no_grad()
def collect_native_ablation_predictions(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, Any]],
    *,
    device: torch.device | str,
    max_windows: int,
    modes: Sequence[str],
    shuffle_seed: int,
    progress_every: int | None = None,
    progress_label: str = "",
) -> dict[str, dict[str, np.ndarray]]:
    """单次遍历 dataloader，收集 native/token 注入路径的 STFT token_delta 消融预测。

    native_inject 不存在 concat 的 fusion_head 输入，因此消融点必须放在 STFT 分支投影后的
    token_delta 上，并继续调用主干的 forward_with_token_injection 保持注入位置语义不变。
    """

    if int(max_windows) <= 0:
        raise ValueError("max_windows 必须大于 0")
    selected_modes = _validate_modes(modes)
    _validate_native_dual_model(model)

    resolved_device = torch.device(device)
    model.to(resolved_device)
    model.eval()
    non_blocking = resolved_device.type == "cuda"
    generator = torch.Generator(device="cpu").manual_seed(int(shuffle_seed))

    pred_parts: dict[str, list[np.ndarray]] = {mode: [] for mode in selected_modes}
    targets: list[np.ndarray] = []
    meta_records: list[dict[str, Any]] = []
    batch_idx = 0
    total_data_wait_sec = 0.0
    total_compute_sec = 0.0

    loader_iter = iter(loader)
    while len(meta_records) < int(max_windows):
        data_start = time.perf_counter()
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        data_wait_sec = time.perf_counter() - data_start
        compute_start = time.perf_counter()
        batch_idx += 1
        if "meta" not in batch:
            raise KeyError("batch 必须包含 meta")
        x = batch["x"].to(resolved_device, non_blocking=non_blocking)
        sst = _batch_sst(batch, resolved_device, non_blocking=non_blocking)
        target_tokens = int(model.time_backbone.token_count_for_length(x.size(-1)))
        stft_feats = model._encode_stft_features(x, sst, target_len=target_tokens)
        token_delta = model._project_stft_features(stft_feats)

        for mode in selected_modes:
            pred = _predict_native_with_ablation(model, x, token_delta, mode=mode, generator=generator)
            pred_parts[mode].append(pred.detach().cpu().numpy())

        targets.append(batch["target"].detach().cpu().numpy())
        batch_size = x.size(0)
        for idx in range(batch_size):
            meta_records.append(_extract_meta(batch["meta"], idx))
        collected = min(len(meta_records), int(max_windows))
        compute_sec = time.perf_counter() - compute_start
        total_data_wait_sec += data_wait_sec
        total_compute_sec += compute_sec
        if progress_every and (batch_idx == 1 or batch_idx % int(progress_every) == 0 or collected >= int(max_windows)):
            label = f" {progress_label}" if progress_label else ""
            print(
                f"  collect{label} batch={batch_idx} windows={collected}/{int(max_windows)} "
                f"data_wait={data_wait_sec:.3f}s compute={compute_sec:.3f}s "
                f"total_data_wait={total_data_wait_sec:.1f}s total_compute={total_compute_sec:.1f}s",
                flush=True,
            )

    if not meta_records:
        raise RuntimeError("没有可收集的预测窗口")

    limit = int(max_windows)
    target_arr = np.concatenate(targets, axis=0)[:limit]
    meta_records = meta_records[:limit]
    outputs: dict[str, dict[str, np.ndarray]] = {}
    for mode, parts in pred_parts.items():
        pred_arr = np.concatenate(parts, axis=0)[:limit]
        outputs[mode] = _prediction_dict(pred_arr, target_arr, meta_records)
    return outputs


def _predict_with_ablation(
    model: torch.nn.Module,
    time_feats: torch.Tensor,
    stft_feats: torch.Tensor,
    *,
    mode: str,
    generator: torch.Generator,
) -> torch.Tensor:
    if mode == "normal":
        used_time = time_feats
        used_stft = stft_feats
    elif mode == "stft_zero":
        used_time = time_feats
        used_stft = torch.zeros_like(stft_feats)
    elif mode == "time_zero":
        used_time = torch.zeros_like(time_feats)
        used_stft = stft_feats
    elif mode == "stft_shuffle_batch":
        used_time = time_feats
        if stft_feats.size(0) <= 1:
            used_stft = stft_feats.roll(shifts=1, dims=-1)
        else:
            perm = torch.randperm(stft_feats.size(0), generator=generator, device=torch.device("cpu"))
            used_stft = stft_feats[perm.to(stft_feats.device)]
    elif mode == "stft_shuffle_time":
        used_time = time_feats
        shifts = max(1, stft_feats.size(-1) // 3)
        used_stft = stft_feats.roll(shifts=shifts, dims=-1)
    else:
        raise ValueError(f"未知消融 mode: {mode}")
    return model.fusion_head(torch.cat([used_time, used_stft], dim=1))


def _predict_native_with_ablation(
    model: torch.nn.Module,
    x: torch.Tensor,
    token_delta: torch.Tensor,
    *,
    mode: str,
    generator: torch.Generator,
) -> torch.Tensor:
    if mode == "normal":
        used_x = x
        used_delta = token_delta
    elif mode == "stft_zero":
        used_x = x
        used_delta = torch.zeros_like(token_delta)
    elif mode == "time_zero":
        used_x = torch.zeros_like(x)
        used_delta = token_delta
    elif mode == "stft_shuffle_batch":
        used_x = x
        if token_delta.size(0) <= 1:
            used_delta = token_delta.roll(shifts=1, dims=-1)
        else:
            perm = torch.randperm(token_delta.size(0), generator=generator, device=torch.device("cpu"))
            used_delta = token_delta[perm.to(token_delta.device)]
    elif mode == "stft_shuffle_time":
        used_x = x
        shifts = max(1, token_delta.size(-1) // 3)
        used_delta = token_delta.roll(shifts=shifts, dims=-1)
    else:
        raise ValueError(f"未知消融 mode: {mode}")
    return model.time_backbone.forward_with_token_injection(
        used_x,
        used_delta,
        inject_position=getattr(model, "stft_inject_position", "post_mixer"),
    )


def _prediction_dict(
    pred_arr: np.ndarray,
    target_arr: np.ndarray,
    meta_records: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    output = {
        "r_tho_hat": pred_arr,
        "tho_ref": target_arr,
        "dataset_row_id": np.asarray([int(m.get("dataset_row_id", -1)) for m in meta_records], dtype=np.int64),
        "split": np.asarray([str(m.get("split", "")) for m in meta_records]),
        "input_set": np.asarray([str(m.get("input_set", "")) for m in meta_records]),
        "residual_quality_class": np.asarray([str(m.get("residual_quality_class", "")) for m in meta_records]),
    }
    if meta_records and "rr_peak_valid_mask" in meta_records[0]:
        output["rr_peak_valid_mask"] = np.stack(
            [np.asarray(m["rr_peak_valid_mask"], dtype=np.bool_).reshape(-1) for m in meta_records],
            axis=0,
        )
    return output


def _evaluate_run_once(
    run_dir: Path,
    checkpoint_path: Path,
    config_path: Path,
    modes: Sequence[str],
    outputs: Mapping[str, Path],
    *,
    device: torch.device,
    max_windows: int | None,
    shuffle_seed: int,
    show_progress: bool | None,
    progress_every: int | None,
    metrics_workers: int,
    metrics_chunk_size: int,
) -> None:
    print(f"  build data/model: {run_dir}", flush=True)
    cfg = load_config(config_path, overrides=[f"training.device={device}"])
    model = build_model(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    _validate_checkpoint_config(checkpoint.get("config"), cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_bundle = build_window_data(
        cfg,
        split=str(cfg.data.val_split),
        max_windows=cfg.data.get("max_val_windows"),
        sample_strategy=str(cfg.data.val_sample_strategy),
        sample_seed=int(cfg.data.val_sample_seed),
        shuffle=False,
    )
    max_eval_windows = len(val_bundle.dataset) if max_windows is None else int(max_windows)
    print(f"  collect predictions: {run_dir} n={max_eval_windows} modes={','.join(modes)}", flush=True)
    predictions_by_mode = collect_ablation_predictions(
        model,
        val_bundle.loader,
        device=device,
        max_windows=max_eval_windows,
        modes=modes,
        shuffle_seed=shuffle_seed,
        progress_every=progress_every,
        progress_label=run_dir.name,
    )
    n_windows = int(next(iter(predictions_by_mode.values()))["r_tho_hat"].shape[0])
    chunk_tasks = build_metric_chunk_tasks(
        outputs,
        n_windows=n_windows,
        chunk_size=int(metrics_chunk_size),
        method_prefix=str(cfg.model.name),
    )
    worker_count = resolve_metrics_workers(metrics_workers, task_count=len(chunk_tasks))
    if worker_count == 1:
        for mode, metrics_output in outputs.items():
            _compute_metrics_worker(
                predictions_by_mode[mode],
                str(config_path),
                f"{cfg.model.name}_{mode}",
                mode,
                str(metrics_output),
                bool(show_progress),
            )
        return

    print(
        f"  compute metrics chunk-parallel: {run_dir} workers={worker_count} "
        f"chunks={len(chunk_tasks)} chunk_size={int(metrics_chunk_size)} modes={','.join(outputs)}",
        flush=True,
    )
    _compute_metrics_chunks_parallel(
        predictions_by_mode,
        chunk_tasks,
        outputs,
        str(config_path),
        worker_count=worker_count,
        show_progress=bool(show_progress),
    )


def _compute_metrics_chunks_parallel(
    predictions_by_mode: dict[str, dict[str, np.ndarray]],
    tasks: Sequence[MetricChunkTask],
    outputs: Mapping[str, Path],
    config_path: str,
    *,
    worker_count: int,
    show_progress: bool,
) -> None:
    if "fork" not in mp.get_all_start_methods():
        raise RuntimeError("chunk-parallel metrics 需要 fork start method；当前平台不支持")

    global _METRICS_PREDICTIONS_BY_MODE
    _METRICS_PREDICTIONS_BY_MODE = predictions_by_mode
    frames_by_mode: dict[str, list[tuple[int, pd.DataFrame]]] = {mode: [] for mode in outputs}
    ctx = mp.get_context("fork")
    try:
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=ctx) as pool:
            futures = [
                pool.submit(_compute_metrics_chunk_worker, task, config_path, bool(show_progress))
                for task in tasks
            ]
            done_count = 0
            total = len(futures)
            for future in as_completed(futures):
                mode, start, frame = future.result()
                frames_by_mode[mode].append((start, frame))
                done_count += 1
                if done_count == 1 or done_count % max(1, total // 10) == 0 or done_count == total:
                    print(f"  metrics chunks done {done_count}/{total}", flush=True)
    finally:
        _METRICS_PREDICTIONS_BY_MODE = None

    for mode, metrics_output in outputs.items():
        print(f"  write metrics: {metrics_output.parent} mode={mode}", flush=True)
        metrics_output.parent.mkdir(parents=True, exist_ok=True)
        frames = [frame for _, frame in sorted(frames_by_mode[mode], key=lambda item: item[0])]
        pd.concat(frames, ignore_index=True).to_csv(metrics_output, index=False)
        print(f"  wrote {metrics_output}", flush=True)


def _compute_metrics_chunk_worker(
    task: MetricChunkTask,
    config_path: str,
    show_progress: bool,
) -> tuple[str, int, pd.DataFrame]:
    if _METRICS_PREDICTIONS_BY_MODE is None:
        raise RuntimeError("metrics worker 未继承预测数组，请确认使用 fork start method")
    predictions = _slice_prediction_dict(_METRICS_PREDICTIONS_BY_MODE[task.mode], task.start, task.end)
    cfg = load_config(config_path, overrides=[])
    metrics = evaluate_prediction_dict(
        predictions,
        cfg,
        method=task.method,
        show_progress=show_progress,
    )
    return task.mode, task.start, metrics


def _slice_prediction_dict(predictions: dict[str, np.ndarray], start: int, end: int) -> dict[str, np.ndarray]:
    n_windows = int(np.asarray(predictions["r_tho_hat"]).shape[0])
    sliced: dict[str, np.ndarray] = {}
    for key, value in predictions.items():
        arr = np.asarray(value)
        sliced[key] = arr[start:end] if arr.shape[:1] == (n_windows,) else arr
    return sliced


def build_metric_chunk_tasks(
    outputs: Mapping[str, Path],
    *,
    n_windows: int,
    chunk_size: int,
    method_prefix: str = "time_stft_dual1d",
) -> list[MetricChunkTask]:
    if int(chunk_size) <= 0:
        raise ValueError("metrics_chunk_size 必须大于 0")
    tasks: list[MetricChunkTask] = []
    for mode in outputs:
        for start in range(0, int(n_windows), int(chunk_size)):
            end = min(start + int(chunk_size), int(n_windows))
            tasks.append(MetricChunkTask(mode=mode, start=start, end=end, method=f"{method_prefix}_{mode}"))
    return tasks


def _compute_metrics_worker(
    predictions: dict[str, np.ndarray],
    config_path: str,
    method: str,
    mode: str,
    metrics_output: str,
    show_progress: bool,
) -> str:
    """单个 mode 的指标计算 worker；输入已是 CPU numpy，适合进程级并行。"""

    output = Path(metrics_output)
    print(f"  compute metrics: {output.parent} mode={mode}", flush=True)
    cfg = load_config(config_path, overrides=[])
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_prediction_dict(
        predictions,
        cfg,
        method=method,
        show_progress=show_progress,
    )
    metrics.to_csv(output, index=False)
    print(f"  wrote {output}", flush=True)
    return str(output)


def _group_specs(specs: Sequence[RunSpec]) -> dict[tuple[Path, Path, Path], dict[str, Path]]:
    grouped: dict[tuple[Path, Path, Path], dict[str, Path]] = {}
    for spec in specs:
        key = (spec.run_dir, spec.checkpoint_path, spec.config_path)
        grouped.setdefault(key, {})[spec.mode] = spec.metrics_output
    return grouped


def resolve_devices(devices: Sequence[str] | None) -> list[str | None]:
    """显式传入多张卡时按 run 轮转；未传时由各 run config 决定设备。"""

    return [str(device) for device in devices] if devices else [None]


def resolve_metrics_workers(value: int, *, task_count: int) -> int:
    """每个 run 内的 metrics worker 数；上限为 chunk 任务数，至少为 1。"""

    return max(1, min(int(value), int(task_count)))


def assign_run_groups(
    groups: Mapping[tuple[Path, Path, Path], Mapping[str, Path]],
    devices: Sequence[str | None],
) -> list[tuple[tuple[Path, Path, Path], Mapping[str, Path], str | None]]:
    """按 run 分组轮转分配设备，确保同一 run 的多 mode 共用一次 dataloader。"""

    if not devices:
        raise ValueError("devices 不能为空")
    return [
        (group_key, outputs, devices[idx % len(devices)])
        for idx, (group_key, outputs) in enumerate(groups.items())
    ]


def run_assignment_worker(
    group_key: tuple[Path, Path, Path],
    outputs: Mapping[str, Path],
    device_name: str | None,
    *,
    max_windows: int | None,
    shuffle_seed: int,
    show_progress: bool,
    progress_every: int | None,
    metrics_workers: int,
    metrics_chunk_size: int,
) -> str:
    """进程级 run worker；每个进程只占一张卡，metrics 阶段避开 GIL 串行瓶颈。"""

    run_dir, checkpoint_path, config_path = group_key
    cfg = OmegaConf.load(config_path)
    cfg_device = OmegaConf.select(cfg, "training.device", default=None)
    selected_device = resolve_device(device_name if device_name is not None else cfg_device)
    print(
        f"start {run_dir} checkpoint={checkpoint_path.name} device={selected_device} modes={','.join(outputs)}",
        flush=True,
    )
    _evaluate_run_once(
        run_dir,
        checkpoint_path,
        config_path,
        list(outputs),
        outputs,
        device=torch.device(selected_device),
        max_windows=max_windows,
        shuffle_seed=int(shuffle_seed),
        show_progress=bool(show_progress),
        progress_every=progress_every,
        metrics_workers=int(metrics_workers),
        metrics_chunk_size=int(metrics_chunk_size),
    )
    print(f"done {run_dir}", flush=True)
    return str(run_dir)


def _write_manifest(path: Path, specs: Sequence[RunSpec]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["tag", "run_dir", "checkpoint", "config", "mode", "metrics_output"],
        )
        writer.writeheader()
        for spec in specs:
            writer.writerow(
                {
                    "tag": spec.tag,
                    "run_dir": str(spec.run_dir),
                    "checkpoint": str(spec.checkpoint_path),
                    "config": str(spec.config_path),
                    "mode": spec.mode,
                    "metrics_output": str(spec.metrics_output),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="E3-C：单次 dataloader 遍历的 STFT 贡献消融评价")
    parser.add_argument("--runs-root", default="runs/e3_b", help="E3-B runs 根目录")
    parser.add_argument("--arm", default="e3_b0_concat_fullband_ref", help="要评价的 arm slug")
    parser.add_argument("--branch", default="dual", help="要评价的 branch，默认 dual")
    parser.add_argument("--checkpoint-name", default="checkpoint.pt", help="run 内 checkpoint 文件名")
    parser.add_argument("--mode", action="append", default=None, help="消融模式，可重复；默认 normal/stft_zero/shuffle")
    parser.add_argument("--force", action="store_true", help="覆盖已存在的 metrics_e3c_*.csv")
    parser.add_argument("--dry-run", action="store_true", help="只写 manifest 并打印计划")
    parser.add_argument("--device", action="append", default=None, help="评价设备，可重复传入；默认读取各 run 配置")
    parser.add_argument("--max-parallel", type=int, default=1, help="并发 run 数；多卡时通常设为设备数")
    parser.add_argument("--max-windows", type=int, default=None, help="最多评价多少个 val window，默认全量")
    parser.add_argument("--shuffle-seed", type=int, default=20260624, help="batch shuffle 消融随机种子")
    parser.add_argument("--progress-every", type=int, default=10, help="collect 阶段每 N 个 batch 打印一次计时；0 表示关闭")
    parser.add_argument("--metrics-workers", type=int, default=1, help="每个 run 内并行计算 metrics 的进程数")
    parser.add_argument("--metrics-chunk-size", type=int, default=128, help="metrics 并行时每个 CPU 任务处理的窗口数")
    parser.add_argument("--show-progress", action="store_true", help="显示指标计算进度条")
    parser.add_argument("--manifest", default="runs/e3_c_ablation_manifest.csv", help="消融评价 manifest 输出路径")
    args = parser.parse_args()

    if args.max_parallel < 1:
        raise SystemExit("--max-parallel 必须 >= 1")

    modes = _validate_modes(args.mode or DEFAULT_MODES)
    specs = discover_run_specs(
        Path(args.runs_root),
        arm=str(args.arm),
        branch=str(args.branch),
        checkpoint_name=str(args.checkpoint_name),
        modes=modes,
        force=bool(args.force),
    )
    _write_manifest(Path(args.manifest), specs)
    if not specs:
        print("no pending E3-C0 ablation tasks", flush=True)
        return
    if args.dry_run:
        for spec in specs:
            print(f"plan {spec.tag} -> {spec.metrics_output}", flush=True)
        return

    grouped = _group_specs(specs)
    assignments = assign_run_groups(grouped, resolve_devices(args.device))
    print(f"queued run_groups={len(assignments)} mode_tasks={len(specs)}", flush=True)

    workers = min(int(args.max_parallel), len(assignments))
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futures = [
            pool.submit(
                run_assignment_worker,
                group_key,
                outputs,
                device,
                max_windows=args.max_windows,
                shuffle_seed=int(args.shuffle_seed),
                show_progress=bool(args.show_progress),
                progress_every=None if int(args.progress_every) <= 0 else int(args.progress_every),
                metrics_workers=max(1, int(args.metrics_workers)),
                metrics_chunk_size=max(1, int(args.metrics_chunk_size)),
            )
            for group_key, outputs, device in assignments
        ]
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
