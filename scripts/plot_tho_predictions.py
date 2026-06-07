from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/resp_reconstruction_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def plot_run_predictions(
    run_dir: str | Path,
    *,
    max_plots: int = 8,
    output_dir: str | Path | None = None,
    sort_by: str = "rr_peak_abs_error",
) -> list[Path]:
    """为一个 run 的诊断预测生成 PNG 图。"""
    run_path = Path(run_dir)
    predictions_path = run_path / "predictions.npz"
    metrics_path = run_path / "metrics.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"预测文件不存在: {predictions_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"指标文件不存在: {metrics_path}")

    predictions = _load_npz(predictions_path)
    metrics = pd.read_csv(metrics_path)
    if int(max_plots) <= 0:
        raise ValueError("max_plots 必须大于 0")

    out_dir = Path(output_dir) if output_dir is not None else run_path / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _load_config(run_path)
    fs = float(OmegaConf.select(cfg, "window.target_fs", default=100.0))
    input_lookup = _load_input_lookup(run_path, predictions, cfg)

    selected = _select_prediction_indices(predictions, metrics, max_plots=int(max_plots), sort_by=sort_by)
    written: list[Path] = []
    for pred_idx in selected:
        row_id = int(np.asarray(predictions["dataset_row_id"])[pred_idx])
        metric_row = _metric_row(metrics, row_id)
        output = out_dir / f"window_{pred_idx:03d}_row_{row_id}.png"
        _plot_one_window(
            output,
            pred=np.asarray(predictions["r_tho_hat"][pred_idx]).reshape(-1),
            target=np.asarray(predictions["tho_ref"][pred_idx]).reshape(-1),
            x=input_lookup.get(row_id),
            fs=fs,
            meta={key: _prediction_meta(predictions, key, pred_idx) for key in ("split", "input_set", "residual_quality_class")},
            metrics=metric_row,
            row_id=row_id,
        )
        written.append(output)
    return written


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _load_config(run_path: Path):
    config_path = run_path / "config.yaml"
    if not config_path.exists():
        return OmegaConf.create({"window": {"target_fs": 100.0}})
    return OmegaConf.load(config_path)


def _select_prediction_indices(
    predictions: dict[str, np.ndarray],
    metrics: pd.DataFrame,
    *,
    max_plots: int,
    sort_by: str,
) -> list[int]:
    n = int(np.asarray(predictions["r_tho_hat"]).shape[0])
    row_ids = np.asarray(predictions.get("dataset_row_id", np.arange(n))).astype(int)
    if sort_by in metrics.columns and "dataset_row_id" in metrics.columns:
        ranked = metrics[["dataset_row_id", sort_by]].copy()
        ranked = ranked[ranked["dataset_row_id"].isin(row_ids)]
        ranked = ranked.sort_values(sort_by, ascending=False, na_position="last")
        order = [int(np.where(row_ids == int(row_id))[0][0]) for row_id in ranked["dataset_row_id"].tolist()]
        order.extend([idx for idx in range(n) if idx not in order])
    else:
        order = list(range(n))
    return order[:max_plots]


def _metric_row(metrics: pd.DataFrame, dataset_row_id: int) -> dict[str, Any]:
    if "dataset_row_id" not in metrics.columns:
        return {}
    matched = metrics[metrics["dataset_row_id"] == dataset_row_id]
    if matched.empty:
        return {}
    return matched.iloc[0].to_dict()


def _prediction_meta(predictions: dict[str, np.ndarray], key: str, idx: int) -> str:
    values = predictions.get(key)
    if values is None:
        return ""
    return str(np.asarray(values)[idx])


def _load_input_lookup(run_path: Path, predictions: dict[str, np.ndarray], cfg) -> dict[int, np.ndarray]:
    """尽量从训练配置和索引还原输入 BCG；失败时返回空字典。"""
    try:
        from resp_train.data.audit import add_usable_flag
        from resp_train.data.dataset import RespWindowDataset
        from resp_train.data.index import read_index

        dataset_root = OmegaConf.select(cfg, "data.dataset_root")
        index_csv = OmegaConf.select(cfg, "data.index_csv")
        if not dataset_root or not index_csv:
            return {}
        df = add_usable_flag(read_index(dataset_root, index_csv), cfg)
        row_ids = set(np.asarray(predictions.get("dataset_row_id", []), dtype=np.int64).tolist())
        rows = df[df["dataset_row_id"].isin(row_ids)].copy()
        if rows.empty:
            return {}
        dataset = RespWindowDataset(Path(str(dataset_root)) / str(index_csv), rows, cfg, preload_windows=False)
        lookup: dict[int, np.ndarray] = {}
        for idx in range(len(dataset)):
            sample = dataset[idx]
            lookup[int(sample["meta"]["dataset_row_id"])] = sample["x"].detach().cpu().numpy().reshape(-1)
        return lookup
    except Exception:
        return {}


def _plot_one_window(
    output: Path,
    *,
    pred: np.ndarray,
    target: np.ndarray,
    x: np.ndarray | None,
    fs: float,
    meta: dict[str, str],
    metrics: dict[str, Any],
    row_id: int,
) -> None:
    time = np.arange(pred.size) / fs
    n_axes = 2 if x is not None else 1
    fig, axes = plt.subplots(n_axes, 1, figsize=(14, 5 + 2 * (n_axes - 1)), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    ax = axes[0]
    if x is not None:
        axes[0].plot(time[: x.size], _zscore(x), color="#666666", linewidth=0.8, label="bcg input z")
        axes[0].set_ylabel("input")
        axes[0].legend(loc="upper right")
        ax = axes[1]
    ax.plot(time, _zscore(target), color="#1f77b4", linewidth=1.0, label="tho_ref z")
    ax.plot(time, _zscore(pred), color="#d62728", linewidth=1.0, alpha=0.85, label="r_tho_hat z")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("waveform")
    ax.legend(loc="upper right")
    title = _title(row_id=row_id, meta=meta, metrics=metrics)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def _zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    std = float(np.std(arr))
    if std <= 0:
        return arr - float(np.mean(arr))
    return (arr - float(np.mean(arr))) / std


def _title(*, row_id: int, meta: dict[str, str], metrics: dict[str, Any]) -> str:
    parts = [
        f"row={row_id}",
        meta.get("split", ""),
        meta.get("input_set", ""),
        meta.get("residual_quality_class", ""),
    ]
    for key in ("rr_spec_abs_error", "rr_peak_abs_error", "envelope_corr", "spectrum_similarity"):
        if key in metrics:
            parts.append(f"{key}={_format_metric(metrics[key])}")
    return " | ".join(str(part) for part in parts if str(part))


def _format_metric(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value):
        return "nan"
    return f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制 THO 预测诊断图")
    parser.add_argument("--run-dir", required=True, help="训练 run 目录")
    parser.add_argument("--output-dir", default="", help="PNG 输出目录，默认 <run-dir>/plots")
    parser.add_argument("--max-plots", type=int, default=8, help="最多绘制窗口数")
    parser.add_argument("--sort-by", default="rr_peak_abs_error", help="按哪个 metrics 列优先绘制")
    args = parser.parse_args()

    written = plot_run_predictions(
        args.run_dir,
        max_plots=args.max_plots,
        output_dir=args.output_dir or None,
        sort_by=args.sort_by,
    )
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
