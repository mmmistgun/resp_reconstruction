from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.config import check_required_packages, load_config
from resp_train.engine import collect_predictions
from resp_train.experiments.tho import ThoExperiment, _resolve_config_path, _validate_checkpoint_config
from resp_train.metrics.parallel import evaluate_predictions_chunked, load_or_build_target_feature_cache
from resp_train.utils.run import resolve_device


def build_summary(
    *,
    checkpoint: str,
    config: str,
    n_windows: int,
    metrics_workers: int,
    metrics_chunk_size: int,
    target_workers: int,
    target_chunk_size: int,
    collect_predictions_sec: float,
    target_features_sec: float,
    metrics_secs: list[float],
) -> dict[str, Any]:
    """汇总一次 profiling；区分 checkpoint 复评耗时和训练内复用 valid 预测的增量耗时。"""

    metrics_median = float(statistics.median(metrics_secs))
    return {
        "checkpoint": str(checkpoint),
        "config": str(config),
        "n_windows": int(n_windows),
        "metrics_workers": int(metrics_workers),
        "metrics_chunk_size": int(metrics_chunk_size),
        "target_workers": int(target_workers),
        "target_chunk_size": int(target_chunk_size),
        "collect_predictions_sec": float(collect_predictions_sec),
        "target_features_sec": float(target_features_sec),
        "metrics_sec_min": float(min(metrics_secs)),
        "metrics_sec_median": metrics_median,
        "metrics_sec_max": float(max(metrics_secs)),
        "metrics_repeats": int(len(metrics_secs)),
        "metrics_windows_per_sec": float(int(n_windows) / metrics_median) if metrics_median > 0 else float("inf"),
        "checkpoint_eval_like_sec": float(collect_predictions_sec + target_features_sec + metrics_median),
        "first_epoch_extra_after_valid_prediction_sec": float(target_features_sec + metrics_median),
        "steady_epoch_extra_after_valid_prediction_sec": metrics_median,
    }


def profile_checkpoint(
    *,
    checkpoint_path: Path,
    config_path: Path | None,
    output_dir: Path,
    overrides: list[str],
    metrics_workers: int,
    metrics_chunk_size: int,
    target_workers: int,
    target_chunk_size: int,
    repeats: int,
    target_cache_dir: Path | None,
    metrics_output: bool,
) -> dict[str, Any]:
    resolved_config = _resolve_config_path(config_path, checkpoint_path)
    cfg = load_config(resolved_config, overrides=overrides)
    device = resolve_device(str(cfg.training.device))
    experiment = ThoExperiment(cfg)
    model = experiment.build_model().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    _validate_checkpoint_config(checkpoint.get("config"), cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_data = experiment._build_checkpoint_eval_val_data()

    collect_start = time.perf_counter()
    predictions = collect_predictions(
        model,
        val_data.loader,
        device=device,
        max_windows=len(val_data.dataset),
    )
    collect_sec = time.perf_counter() - collect_start
    n_windows = int(predictions["r_tho_hat"].shape[0])

    cache_dir = target_cache_dir if target_cache_dir is not None else output_dir / "target_feature_cache"
    target_start = time.perf_counter()
    target_features = load_or_build_target_feature_cache(
        predictions,
        cfg,
        cache_dir=cache_dir,
        show_progress=False,
        target_workers=int(target_workers),
        target_chunk_size=int(target_chunk_size),
    )
    target_sec = time.perf_counter() - target_start

    metrics_secs: list[float] = []
    first_metrics: pd.DataFrame | None = None
    for repeat_idx in range(1, int(repeats) + 1):
        metrics_start = time.perf_counter()
        metrics = evaluate_predictions_chunked(
            predictions,
            cfg,
            method=str(cfg.model.name),
            metrics_workers=int(metrics_workers),
            metrics_chunk_size=int(metrics_chunk_size),
            target_features=target_features,
            show_progress=False,
        )
        metrics_secs.append(time.perf_counter() - metrics_start)
        if repeat_idx == 1:
            first_metrics = metrics

    output_dir.mkdir(parents=True, exist_ok=True)
    if metrics_output and first_metrics is not None:
        first_metrics.to_csv(output_dir / "profile_metrics_repeat1.csv", index=False)

    summary = build_summary(
        checkpoint=str(checkpoint_path),
        config=str(resolved_config),
        n_windows=n_windows,
        metrics_workers=int(metrics_workers),
        metrics_chunk_size=int(metrics_chunk_size),
        target_workers=int(target_workers),
        target_chunk_size=int(target_chunk_size),
        collect_predictions_sec=collect_sec,
        target_features_sec=target_sec,
        metrics_secs=metrics_secs,
    )
    timing_rows = [
        {"stage": "collect_predictions", "repeat": 1, "seconds": collect_sec},
        {"stage": "target_features", "repeat": 1, "seconds": target_sec},
        *[
            {"stage": "metrics", "repeat": idx, "seconds": seconds}
            for idx, seconds in enumerate(metrics_secs, start=1)
        ],
    ]
    pd.DataFrame(timing_rows).to_csv(output_dir / "timings.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="profile 每个 epoch 全量 val 指标计算的时间成本")
    parser.add_argument("--checkpoint", required=True, help="待 profile 的 checkpoint.pt 或 checkpoint_topN.pt")
    parser.add_argument("--config", default="", help="配置文件；为空时使用 checkpoint 同目录 config.yaml")
    parser.add_argument("--output-dir", required=True, help="写出 timings.csv / summary.json 的目录")
    parser.add_argument("--metrics-workers", type=int, default=4, help="metrics chunk 进程数")
    parser.add_argument("--metrics-chunk-size", type=int, default=128, help="每个 metrics chunk 的窗口数")
    parser.add_argument("--target-workers", type=int, default=0, help="target feature chunk 进程数；0 表示跟随 metrics-workers")
    parser.add_argument("--target-chunk-size", type=int, default=128, help="每个 target feature chunk 的窗口数")
    parser.add_argument("--repeats", type=int, default=2, help="重复 metrics 阶段次数，用于估计稳态中位数")
    parser.add_argument("--target-cache-dir", default="", help="target feature cache 目录；默认 output-dir/target_feature_cache")
    parser.add_argument("--write-metrics", action="store_true", help="额外写出第一轮逐窗口 metrics，便于复核")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="OmegaConf dotlist 覆盖，可重复传入")
    args = parser.parse_args()

    if args.metrics_workers < 1:
        raise SystemExit("--metrics-workers 必须 >= 1")
    if args.metrics_chunk_size < 1:
        raise SystemExit("--metrics-chunk-size 必须 >= 1")
    if args.target_workers < 0:
        raise SystemExit("--target-workers 必须 >= 0")
    if args.target_chunk_size < 1:
        raise SystemExit("--target-chunk-size 必须 >= 1")
    if args.repeats < 1:
        raise SystemExit("--repeats 必须 >= 1")
    missing = check_required_packages()
    if missing:
        raise SystemExit(f"缺少依赖: {missing}; 请先确认是否安装。")

    summary = profile_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        config_path=Path(args.config) if args.config else None,
        output_dir=Path(args.output_dir),
        overrides=args.overrides,
        metrics_workers=int(args.metrics_workers),
        metrics_chunk_size=int(args.metrics_chunk_size),
        target_workers=int(args.target_workers or args.metrics_workers),
        target_chunk_size=int(args.target_chunk_size),
        repeats=int(args.repeats),
        target_cache_dir=Path(args.target_cache_dir) if args.target_cache_dir else None,
        metrics_output=bool(args.write_metrics),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
