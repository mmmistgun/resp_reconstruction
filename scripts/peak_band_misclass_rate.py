from __future__ import annotations

# peak-band 谐波误拣率后处理分析。
#
# 背景：multiscale 双分支 wrapper 与原生 E1a 的 peak-band MAE「均值」差异，经逐窗
# 分布核查后确认 **不是波形重建退化**（各配置 median 几乎一致，0.18~0.21），而是
# **长尾谐波误拣率**不同——少数窗口把基频认成 2× 谐波，绝对误差很大，单 seed 的尾巴
# 就能把均值从 0.18 拉到 1.0+。因此 peak-band「均值」是被 10~20% 误拣窗口主导的脆弱口径。
#
# 本脚本读取每个 run 的 metrics.csv 逐窗 `rr_peak_band_abs_error`，给出：
#   - median / 截尾均值（去掉尾部 trim_frac）：反映典型窗口与去尾后的稳健中心
#   - mean：现行主指标，被尾部主导
#   - 误拣率 frac>thr（默认 1.0 次/分）：把谐波误拣量化成一个稳健占比指标
#   - p90 / p95 / max / 计数：尾部形态
# 并按 config.yaml 自动标注 backbone / branch_mode / fuse_len / fusion_decoder / seed，
# 便于把「裸 backbone vs wrapper」「deep vs lite」等对照横向铺开。
#
# 用法：
#   python scripts/peak_band_misclass_rate.py --runs-root runs/<某批>  [--output xxx.csv]
#   python scripts/peak_band_misclass_rate.py --runs-root A --runs-root B --threshold 1.0 1.5
#   # 多个根目录可叠加，对照不同批次（如 native E1a 与 wrapper probe）。

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:  # 可选依赖：仅用于读取 config.yaml 标注；缺失时降级为不标注。
    import yaml
except ImportError:  # pragma: no cover - 环境兜底
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

METRIC_COLUMN = "rr_peak_band_abs_error"

# 从 config.yaml 提取、用于横向对照的配置字段（缺失记为空）。
CONFIG_FIELDS = [
    "name",
    "time_backbone",
    "branch_mode",
    "fusion_mode",
    "fuse_len",
    "fusion_decoder",
    "stft_high_hz",
    "stft_norm",
    "stft_encoder_type",
]


def _load_config_labels(run_dir: Path) -> dict[str, Any]:
    """读取 run 的 config.yaml，提取对照所需的模型/训练字段。"""
    labels: dict[str, Any] = {field: "" for field in CONFIG_FIELDS}
    labels["seed"] = ""
    config_path = run_dir / "config.yaml"
    if yaml is None or not config_path.exists():
        return labels
    config = yaml.safe_load(config_path.read_text()) or {}
    model = config.get("model", {}) or {}
    training = config.get("training", {}) or {}
    for field in CONFIG_FIELDS:
        value = model.get(field)
        if value is not None:
            labels[field] = value
    # 裸 backbone（如原生 E1a）没有 time_backbone 字段，用 name 兜底，统一口径。
    if not labels["time_backbone"] and labels["name"]:
        labels["time_backbone"] = labels["name"]
    seed = training.get("seed")
    if seed is not None:
        labels["seed"] = seed
    return labels


def _trimmed_mean(sorted_values: np.ndarray, trim_frac: float) -> float:
    """去掉尾部 trim_frac 后的均值，反映剔除误拣尾巴后的稳健中心。"""
    if sorted_values.size == 0:
        return float("nan")
    keep = max(1, int(round((1.0 - trim_frac) * sorted_values.size)))
    return float(sorted_values[:keep].mean())


def analyze_run(run_dir: Path, thresholds: list[float], trim_frac: float) -> dict[str, Any] | None:
    """对单个 run 的逐窗 peak-band 误差做分布与误拣率统计。"""
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return None
    frame = pd.read_csv(metrics_path)
    if METRIC_COLUMN not in frame.columns:
        return None
    values = pd.to_numeric(frame[METRIC_COLUMN], errors="coerce").dropna().to_numpy()
    if values.size == 0:
        return None
    sorted_values = np.sort(values)
    n = sorted_values.size

    record: dict[str, Any] = {"run_id": run_dir.name, "run_dir": str(run_dir)}
    record.update(_load_config_labels(run_dir))
    record["n_windows"] = int(n)
    record["mean"] = float(sorted_values.mean())
    record["median"] = float(np.median(sorted_values))
    record[f"trimmed{int(round((1 - trim_frac) * 100))}_mean"] = _trimmed_mean(sorted_values, trim_frac)
    record["p90"] = float(np.quantile(sorted_values, 0.90))
    record["p95"] = float(np.quantile(sorted_values, 0.95))
    record["max"] = float(sorted_values[-1])
    # 误拣率：超过阈值（次/分）的窗口占比，作为稳健的谐波误拣量化指标。
    for thr in thresholds:
        count = int((sorted_values > thr).sum())
        record[f"frac_gt_{thr:g}"] = count / n
        record[f"count_gt_{thr:g}"] = count
    return record


def collect(runs_roots: list[Path], thresholds: list[float], trim_frac: float) -> pd.DataFrame:
    """遍历多个 runs 根目录，逐 run 统计并汇总成对照表。"""
    records: list[dict[str, Any]] = []
    for root in runs_roots:
        if not root.exists():
            raise FileNotFoundError(f"runs 根目录不存在: {root}")
        for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            record = analyze_run(run_dir, thresholds, trim_frac)
            if record is not None:
                record["runs_root"] = str(root)
                records.append(record)
    if not records:
        raise RuntimeError("未在给定根目录下找到任何含 metrics.csv 且包含 peak-band 列的 run")
    frame = pd.DataFrame.from_records(records)
    sort_keys = [key for key in ["time_backbone", "branch_mode", "fuse_len", "fusion_decoder", "seed"] if key in frame.columns]
    if sort_keys:
        frame = frame.sort_values(sort_keys, kind="stable").reset_index(drop=True)
    return frame


def _config_group_keys(frame: pd.DataFrame) -> list[str]:
    """选出用于聚合的配置维度（去掉 seed，按配置对多 seed 求统计）。"""
    return [
        key
        for key in [
            "time_backbone",
            "branch_mode",
            "fusion_mode",
            "fuse_len",
            "fusion_decoder",
            "stft_encoder_type",
            "stft_high_hz",
            "stft_norm",
        ]
        if key in frame.columns
    ]


def grouped_summary(frame: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    """按配置（跨 seed）聚合：均值靠尾巴、median 看典型、误拣率看稳健差异。"""
    group_keys = _config_group_keys(frame)
    if not group_keys:
        return pd.DataFrame()
    agg: dict[str, Any] = {
        "run_id": "count",
        "mean": "mean",
        "median": "mean",
        "p95": "mean",
    }
    for thr in thresholds:
        agg[f"frac_gt_{thr:g}"] = "mean"
    summary = frame.groupby(group_keys, dropna=False).agg(agg).reset_index()
    summary = summary.rename(columns={"run_id": "n_seeds"})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="peak-band 谐波误拣率后处理分析")
    parser.add_argument(
        "--runs-root",
        action="append",
        required=True,
        help="runs 根目录（可重复传入多个，叠加对照不同批次）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        nargs="+",
        default=[1.0, 2.0],
        help="误拣阈值（次/分），逐窗误差超过即记为误拣，可传多个",
    )
    parser.add_argument(
        "--trim-frac",
        type=float,
        default=0.05,
        help="截尾均值去掉的尾部比例，默认 0.05（即 95%% 截尾均值）",
    )
    parser.add_argument(
        "--output",
        default="",
        help="逐 run 明细 CSV 输出路径；为空则只打印不落盘",
    )
    parser.add_argument(
        "--grouped-output",
        default="",
        help="按配置跨 seed 聚合的 CSV 输出路径；为空则只打印不落盘",
    )
    args = parser.parse_args()

    runs_roots = [Path(root) for root in args.runs_root]
    thresholds = sorted(set(args.threshold))
    frame = collect(runs_roots, thresholds, args.trim_frac)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    detail_cols = [
        col
        for col in [
            "time_backbone",
            "branch_mode",
            "fusion_mode",
            "fuse_len",
            "fusion_decoder",
            "stft_encoder_type",
            "seed",
            "n_windows",
            "mean",
            "median",
            "p95",
            "max",
        ]
        + [f"frac_gt_{thr:g}" for thr in thresholds]
        if col in frame.columns
    ]
    print("=== 逐 run 明细 ===")
    print(frame[detail_cols].to_string(index=False))

    summary = grouped_summary(frame, thresholds)
    if not summary.empty:
        print("\n=== 按配置跨 seed 聚合 ===")
        print(summary.to_string(index=False))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
        print(f"\n写出逐 run 明细: {output_path} rows={len(frame)}")
    if args.grouped_output and not summary.empty:
        grouped_path = Path(args.grouped_output)
        grouped_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(grouped_path, index=False)
        print(f"写出聚合表: {grouped_path} rows={len(summary)}")


if __name__ == "__main__":
    main()
