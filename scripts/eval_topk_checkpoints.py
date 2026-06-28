from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf


METRICS = [
    "rr_peak_band_abs_error",
    "rr_spec_abs_error",
    "breath_count_zero_cross_abs_error",
    "relative_envelope_mae",
    "relative_envelope_corr",
    "spectrum_similarity",
    "band_limited_corr",
    "best_lag_corr",
    "best_lag_sec",
]

SELECTION_COLUMNS = [
    "rr_peak_band_abs_error_mean",
    "frac_gt_1",
    "frac_gt_2",
    "rr_spec_abs_error_mean",
    "breath_count_zero_cross_abs_error_mean",
]


@dataclass(frozen=True)
class EvalSpec:
    run_dir: Path
    arm_slug: str
    branch_mode: str
    run_id: str
    rank: int
    checkpoint_path: Path
    config_path: Path
    metrics_output: Path

    @property
    def tag(self) -> str:
        return f"{self.arm_slug}_{self.branch_mode}_{self.run_id}_top{self.rank}"


@dataclass(frozen=True)
class OutputPaths:
    manifest: Path
    all_metrics: Path
    best_by_rr: Path


def discover_eval_specs(runs_root: Path, *, top_k: int = 3, force: bool = False) -> list[EvalSpec]:
    """发现通用 run 目录中的 checkpoint_topN 评价任务。"""

    if top_k < 1:
        raise ValueError("top_k 必须 >= 1")
    if not runs_root.exists():
        raise FileNotFoundError(f"runs 根目录不存在: {runs_root}")

    specs: list[EvalSpec] = []
    for run_dir in _iter_run_dirs(runs_root):
        config_path = run_dir / "config.yaml"
        if not config_path.exists():
            continue
        arm_slug = run_dir.parents[1].name
        branch_mode = run_dir.parent.name
        run_id = run_dir.name
        for rank in range(1, top_k + 1):
            checkpoint_path = run_dir / f"checkpoint_top{rank}.pt"
            metrics_output = run_dir / f"metrics_top{rank}.csv"
            if not checkpoint_path.exists():
                continue
            if metrics_output.exists() and not force:
                continue
            specs.append(
                EvalSpec(
                    run_dir=run_dir,
                    arm_slug=arm_slug,
                    branch_mode=branch_mode,
                    run_id=run_id,
                    rank=rank,
                    checkpoint_path=checkpoint_path,
                    config_path=config_path,
                    metrics_output=metrics_output,
                )
            )
    return specs


def resolve_devices(devices: list[str] | None) -> list[str]:
    """显式传入设备时不混入默认 cuda:0，避免多卡调度偏置。"""

    return devices or ["cuda:0"]


def assign_devices(specs: list[EvalSpec], devices: list[str]) -> list[tuple[EvalSpec, str]]:
    """按任务顺序轮转分配设备；并发数由 --max-parallel 独立控制。"""

    return [(spec, devices[idx % len(devices)]) for idx, spec in enumerate(specs)]


def build_launch_plan(
    assignments: list[tuple[EvalSpec, str]],
    *,
    max_parallel: int,
    start_stagger_sec: float,
) -> list[tuple[EvalSpec, str, float]]:
    """按并发槽位错开启动时间，降低 checkpoint/config 同时读取峰值。"""

    if not assignments:
        return []
    workers = min(max(1, int(max_parallel)), len(assignments))
    stagger = max(0.0, float(start_stagger_sec))
    return [(spec, device, float(idx % workers) * stagger) for idx, (spec, device) in enumerate(assignments)]


def command_for_spec(
    spec: EvalSpec,
    device: str,
    *,
    python: str = sys.executable,
    metric_workers: int | None = None,
) -> list[str]:
    """生成单个 checkpoint 评价命令。"""

    command = [
        python,
        "scripts/eval_tho_small.py",
        "--checkpoint",
        str(spec.checkpoint_path),
        "--config",
        str(spec.config_path),
        "--metrics-output",
        str(spec.metrics_output),
        "--set",
        f"training.device={device}",
    ]
    if metric_workers is not None and int(metric_workers) > 1:
        command.extend(["--set", f"evaluation.metric_workers={int(metric_workers)}"])
    return command


def manifest_row(spec: EvalSpec, device: str, launch_delay_sec: float = 0.0) -> dict[str, str | int | float]:
    return {
        "tag": spec.tag,
        "arm_slug": spec.arm_slug,
        "branch_mode": spec.branch_mode,
        "run_id": spec.run_id,
        "rank": spec.rank,
        "device": device,
        "launch_delay_sec": float(launch_delay_sec),
        "checkpoint": str(spec.checkpoint_path),
        "config": str(spec.config_path),
        "metrics_output": str(spec.metrics_output),
    }


def output_paths(runs_root: Path, *, output_prefix: str | Path | None) -> OutputPaths:
    prefix = Path(output_prefix) if output_prefix else runs_root.with_name(f"{runs_root.name}_topk")
    return OutputPaths(
        manifest=Path(f"{prefix}_eval_manifest.csv"),
        all_metrics=Path(f"{prefix}_all_metrics.csv"),
        best_by_rr=Path(f"{prefix}_best_by_rr.csv"),
    )


def summarize_topk_results(runs_root: Path, *, top_k: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """汇总 metrics_topN.csv，并按任务主指标为每个 run 选择一个 rank。"""

    records: list[dict[str, Any]] = []
    for run_dir in _iter_run_dirs(runs_root):
        topk_path = run_dir / "checkpoint_topk.csv"
        if not topk_path.exists():
            continue
        topk = pd.read_csv(topk_path)
        seed = _training_seed(run_dir)
        arm_slug = run_dir.parents[1].name
        branch_mode = run_dir.parent.name
        run_id = run_dir.name
        for row in topk.to_dict("records"):
            rank = int(row["rank"])
            if rank > top_k:
                continue
            metrics_path = run_dir / f"metrics_top{rank}.csv"
            if not metrics_path.exists():
                continue
            metrics = pd.read_csv(metrics_path)
            record: dict[str, Any] = {
                "run_dir": str(run_dir),
                "label": arm_slug,
                "branch_mode": branch_mode,
                "run_id": run_id,
                "seed": seed if seed is not None else np.nan,
                "rank": rank,
                "checkpoint": str(row.get("checkpoint", f"checkpoint_top{rank}.pt")),
                "epoch": int(row["epoch"]),
                "val_loss": float(row["val_loss"]),
                "metrics_output": str(metrics_path),
            }
            record.update(_metric_summary(metrics))
            records.append(record)

    all_frame = pd.DataFrame.from_records(records)
    if all_frame.empty:
        return all_frame, all_frame.copy()

    best = (
        all_frame.sort_values(["run_dir", *SELECTION_COLUMNS], na_position="last")
        .groupby("run_dir", as_index=False, sort=False)
        .head(1)
        .reset_index(drop=True)
    )
    return all_frame, best


def write_topk_summary(all_frame: pd.DataFrame, best_frame: pd.DataFrame, paths: OutputPaths) -> None:
    paths.all_metrics.parent.mkdir(parents=True, exist_ok=True)
    paths.best_by_rr.parent.mkdir(parents=True, exist_ok=True)
    all_frame.to_csv(paths.all_metrics, index=False)
    best_frame.to_csv(paths.best_by_rr, index=False)


def _metric_summary(metrics: pd.DataFrame) -> dict[str, Any]:
    record: dict[str, Any] = {"n_windows": int(len(metrics))}
    for metric in METRICS:
        values = pd.to_numeric(metrics.get(metric, pd.Series(dtype=float)), errors="coerce")
        record[f"{metric}_mean"] = float(values.mean()) if not values.empty else np.nan
        record[f"{metric}_median"] = float(values.median()) if not values.empty else np.nan
    peak = pd.to_numeric(metrics.get("rr_peak_band_abs_error", pd.Series(dtype=float)), errors="coerce").dropna()
    for threshold in (1.0, 2.0):
        record[f"frac_gt_{threshold:g}"] = float((peak > threshold).mean()) if not peak.empty else np.nan
    return record


def _training_seed(run_dir: Path) -> int | None:
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return None
    try:
        config = OmegaConf.load(config_path)
        seed = OmegaConf.select(config, "training.seed")
    except Exception:
        return None
    return int(seed) if seed is not None else None


def _iter_run_dirs(runs_root: Path) -> list[Path]:
    return sorted(path for path in runs_root.glob("*/*/*") if path.is_dir())


def _run_one(spec: EvalSpec, device: str, *, metric_workers: int, launch_delay_sec: float = 0.0) -> str:
    if float(launch_delay_sec) > 0.0:
        print(f"delay {spec.tag} device={device} sleep={float(launch_delay_sec):.1f}s", flush=True)
        time.sleep(float(launch_delay_sec))
    print(f"start {spec.tag} device={device}", flush=True)
    subprocess.run(command_for_spec(spec, device, metric_workers=metric_workers), check=True)
    print(f"done {spec.tag}", flush=True)
    return spec.tag


def _write_manifest(path: Path, launch_plan: list[tuple[EvalSpec, str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "tag",
                "arm_slug",
                "branch_mode",
                "run_id",
                "rank",
                "device",
                "launch_delay_sec",
                "checkpoint",
                "config",
                "metrics_output",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_row(spec, device, delay) for spec, device, delay in launch_plan)


def _run_eval(launch_plan: list[tuple[EvalSpec, str, float]], *, max_parallel: int, metric_workers: int) -> None:
    if not launch_plan:
        print("no pending topK eval tasks", flush=True)
        return
    workers = min(max_parallel, len(launch_plan))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_run_one, spec, device, metric_workers=metric_workers, launch_delay_sec=delay)
            for spec, device, delay in launch_plan
        ]
        for future in as_completed(futures):
            future.result()


def main() -> None:
    parser = argparse.ArgumentParser(description="重评通用 run 目录中的 checkpoint_topN，并按任务指标择优")
    parser.add_argument("--runs-root", required=True, help="runs 根目录，形如 runs/f_d_highfreq")
    parser.add_argument("--top-k", type=int, default=3, help="重评 checkpoint_topN 的 N 上限，默认 3")
    parser.add_argument("--force", action="store_true", help="覆盖已存在的 metrics_topN.csv")
    parser.add_argument("--dry-run", action="store_true", help="只打印评价计划，不执行评价或择优")
    parser.add_argument("--eval-only", action="store_true", help="只重评 checkpoint_topN，不输出择优表")
    parser.add_argument("--select-only", action="store_true", help="跳过重评，只读取已有 metrics_topN.csv 并择优")
    parser.add_argument("--device", action="append", default=None, help="评价设备，可重复传入；默认 cuda:0")
    parser.add_argument("--max-parallel", type=int, default=1, help="并发评价进程数；默认 1")
    parser.add_argument("--metric-workers", type=int, default=1, help="每个评价进程的指标计算线程数；默认 1")
    parser.add_argument("--start-stagger-sec", type=float, default=0.0, help="按并发槽位错开启动秒数；默认 0")
    parser.add_argument("--output-prefix", default="", help="输出前缀；默认为 <runs-root>_topk")
    parser.add_argument("--manifest", default="", help="评价任务 manifest 输出路径；默认由 output-prefix 推导")
    args = parser.parse_args()

    if args.top_k < 1:
        raise SystemExit("--top-k 必须 >= 1")
    if args.max_parallel < 1:
        raise SystemExit("--max-parallel 必须 >= 1")
    if args.metric_workers < 1:
        raise SystemExit("--metric-workers 必须 >= 1")
    if args.start_stagger_sec < 0:
        raise SystemExit("--start-stagger-sec 必须 >= 0")
    if args.eval_only and args.select_only:
        raise SystemExit("--eval-only 和 --select-only 不能同时使用")

    runs_root = Path(args.runs_root)
    paths = output_paths(runs_root, output_prefix=args.output_prefix or None)
    manifest_path = Path(args.manifest) if args.manifest else paths.manifest

    if not args.select_only:
        specs = discover_eval_specs(runs_root, top_k=args.top_k, force=args.force)
        assignments = assign_devices(specs, resolve_devices(args.device))
        launch_plan = build_launch_plan(
            assignments,
            max_parallel=int(args.max_parallel),
            start_stagger_sec=float(args.start_stagger_sec),
        )
        _write_manifest(manifest_path, launch_plan)
        if args.dry_run:
            for spec, device, delay in launch_plan:
                print(f"plan {spec.tag} device={device} delay={delay:.1f}s", flush=True)
            return
        _run_eval(launch_plan, max_parallel=int(args.max_parallel), metric_workers=int(args.metric_workers))

    if args.eval_only:
        return

    all_frame, best_frame = summarize_topk_results(runs_root, top_k=args.top_k)
    write_topk_summary(all_frame, best_frame, paths)
    print(f"all: {paths.all_metrics} rows={len(all_frame)}", flush=True)
    print(f"best: {paths.best_by_rr} rows={len(best_frame)}", flush=True)
    if not best_frame.empty:
        display_columns = [
            "label",
            "seed",
            "rank",
            "epoch",
            "rr_peak_band_abs_error_mean",
            "frac_gt_1",
            "frac_gt_2",
            "rr_spec_abs_error_mean",
        ]
        print(best_frame[display_columns].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
