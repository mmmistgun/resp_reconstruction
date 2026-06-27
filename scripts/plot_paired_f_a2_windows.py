from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

os.environ.setdefault("MPLCONFIGDIR", "/tmp/resp_reconstruction_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plot_tho_predictions import (  # noqa: E402
    _infer_prediction_lookup,
    _load_config,
    _normalized_power_spectrum,
    _scale_for_display,
)


def plot_paired_f_a2_windows(
    window_list: str | Path,
    *,
    output_dir: str | Path,
    candidate_labels: Iterable[str] | None = None,
    filter_column: str | None = None,
    sort_by: str | None = None,
    sort_ascending: bool = False,
    max_rows: int = 30,
    scale_mode: str = "robust",
    device: str | None = None,
) -> list[Path]:
    """按窗口清单绘制 F0 与 F-A2 候选的 paired 波形诊断图。"""

    rows = load_window_rows(
        window_list,
        candidate_labels=candidate_labels,
        filter_column=filter_column,
        sort_by=sort_by,
        sort_ascending=sort_ascending,
        max_rows=max_rows,
    )
    if rows.empty:
        raise ValueError("窗口清单筛选后为空")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = build_prediction_cache(rows, device=device)
    written: list[Path] = []
    for _, row in rows.iterrows():
        pair = infer_pair_predictions(row, cache=cache, device=device)
        output = out_dir / _plot_filename(row)
        _plot_pair(output, row=row.to_dict(), pair=pair, scale_mode=scale_mode)
        written.append(output)
    return written


def load_window_rows(
    window_list: str | Path,
    *,
    candidate_labels: Iterable[str] | None = None,
    filter_column: str | None = None,
    sort_by: str | None = None,
    sort_ascending: bool = False,
    max_rows: int | None = None,
) -> pd.DataFrame:
    frame = pd.read_csv(window_list)
    required = {"baseline_run_dir", "candidate_run_dir", "dataset_row_id", "label"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"窗口清单缺少列: {missing}")
    if candidate_labels is not None:
        wanted = {str(label) for label in candidate_labels}
        frame = frame[frame["label"].astype(str).isin(wanted)]
    if filter_column:
        if filter_column not in frame.columns:
            raise ValueError(f"窗口清单缺少 filter_column: {filter_column}")
        frame = frame[_truthy_series(frame[filter_column])]
    if sort_by:
        if sort_by not in frame.columns:
            raise ValueError(f"窗口清单缺少 sort_by: {sort_by}")
        frame = frame.sort_values(sort_by, ascending=bool(sort_ascending), na_position="last")
    if max_rows is not None and int(max_rows) > 0:
        frame = frame.head(int(max_rows))
    return frame.reset_index(drop=True)


def build_prediction_cache(rows: pd.DataFrame, *, device: str | None = None) -> dict[str, dict[str, Any]]:
    requests: dict[str, set[int]] = {}
    for _, row in rows.iterrows():
        row_id = int(row["dataset_row_id"])
        requests.setdefault(str(row["baseline_run_dir"]), set()).add(row_id)
        requests.setdefault(str(row["candidate_run_dir"]), set()).add(row_id)
    cache: dict[str, dict[str, Any]] = {}
    for run_dir, row_ids in requests.items():
        run_path = Path(run_dir)
        cfg = _config_with_device(run_path, device=device)
        sorted_row_ids = sorted(int(row_id) for row_id in row_ids)
        cache[str(run_path)] = {
            "cfg": cfg,
            "lookup": _infer_prediction_lookup(run_path, cfg, sorted_row_ids),
            "fs": float(OmegaConf.select(cfg, "window.target_fs", default=100.0)),
        }
    return cache


def infer_pair_predictions(
    row: pd.Series | dict,
    *,
    cache: dict[str, dict[str, Any]] | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    row_id = int(row["dataset_row_id"])
    baseline_dir = str(row["baseline_run_dir"])
    candidate_dir = str(row["candidate_run_dir"])
    if cache is None:
        cache = build_prediction_cache(pd.DataFrame([row]), device=device)
    baseline = cache[baseline_dir]["lookup"][row_id]
    candidate = cache[candidate_dir]["lookup"][row_id]
    return {
        "target": np.asarray(baseline["target"], dtype=np.float64).reshape(-1),
        "baseline": np.asarray(baseline["pred"], dtype=np.float64).reshape(-1),
        "candidate": np.asarray(candidate["pred"], dtype=np.float64).reshape(-1),
        "x": baseline.get("x"),
        "meta": baseline.get("meta", {}),
        "fs": float(cache[baseline_dir]["fs"]),
    }


def _config_with_device(run_path: Path, *, device: str | None):
    cfg = _load_config(run_path)
    if device:
        OmegaConf.update(cfg, "training.device", device, force_add=True)
    return cfg


def _truthy_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"1", "true", "yes", "y"})


def _plot_pair(output: Path, *, row: dict[str, Any], pair: dict[str, Any], scale_mode: str) -> None:
    target = np.asarray(pair["target"]).reshape(-1)
    baseline = np.asarray(pair["baseline"]).reshape(-1)
    candidate = np.asarray(pair["candidate"]).reshape(-1)
    fs = float(pair.get("fs", 100.0))
    time = np.arange(target.size) / fs
    target_z = _scale_for_display(target, mode=scale_mode)
    baseline_z = _scale_for_display(baseline, mode=scale_mode)
    candidate_z = _scale_for_display(candidate, mode=scale_mode)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    axes[0].plot(time, target_z, color="#1f77b4", linewidth=1.1, label="target")
    axes[0].plot(time, baseline_z, color="#555555", linewidth=0.95, alpha=0.9, label="F0")
    axes[0].plot(time, candidate_z, color="#d62728", linewidth=0.95, alpha=0.85, label=str(row.get("label", "candidate")))
    axes[0].set_ylabel(scale_mode)
    axes[0].grid(True, alpha=0.18)
    axes[0].legend(loc="upper right")

    axes[1].plot(time, baseline_z - target_z, color="#555555", linewidth=0.9, label="F0 - target")
    axes[1].plot(time, candidate_z - target_z, color="#d62728", linewidth=0.9, alpha=0.85, label="candidate - target")
    axes[1].axhline(0.0, color="#888888", linewidth=0.7)
    axes[1].set_ylabel("residual")
    axes[1].set_xlabel("time (s)")
    axes[1].grid(True, alpha=0.18)
    axes[1].legend(loc="upper right")

    _plot_pair_spectrum(axes[2], target=target, baseline=baseline, candidate=candidate, fs=fs)
    axes[2].set_xlabel("frequency (Hz)")

    fig.suptitle(_title(row), fontsize=12)
    fig.text(
        0.01,
        0.955,
        _metric_text(row),
        ha="left",
        va="top",
        fontsize=9,
        family="monospace",
        bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.92, "boxstyle": "round,pad=0.35"},
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output, dpi=140)
    plt.close(fig)


def _plot_pair_spectrum(ax, *, target: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, fs: float) -> None:
    for values, color, label in (
        (target, "#1f77b4", "target"),
        (baseline, "#555555", "F0"),
        (candidate, "#d62728", "candidate"),
    ):
        freqs, power = _normalized_power_spectrum(values, fs=fs)
        mask = freqs <= min(3.0, fs / 2.0)
        ax.plot(freqs[mask], power[mask], color=color, linewidth=1.0, alpha=0.9, label=label)
    for x, name in ((0.7, "0.7"), (1.2, "1.2")):
        if x < fs / 2.0:
            ax.axvline(x, color="#333333", linewidth=0.75, linestyle=":", alpha=0.65)
    ax.set_xlim(0.0, min(3.0, fs / 2.0))
    ax.set_ylabel("norm power")
    ax.grid(True, alpha=0.18)
    ax.legend(loc="upper right")


def _title(row: dict[str, Any]) -> str:
    return (
        f"row={int(row['dataset_row_id'])} | {row.get('label', '')} | seed={row.get('seed', '')} | "
        f"delta_peak={_fmt(row.get('delta_rr_peak_band_abs_error'))}"
    )


def _metric_text(row: dict[str, Any]) -> str:
    lines = [
        [
            "baseline_rr_peak_band_abs_error",
            "candidate_rr_peak_band_abs_error",
            "delta_rr_peak_band_abs_error",
        ],
        [
            "baseline_pred_rr_peak_band_bpm",
            "candidate_pred_rr_peak_band_bpm",
            "target_rr_peak_band_bpm",
        ],
        [
            "baseline_spectrum_similarity",
            "baseline_breath_count_zero_cross_abs_error",
            "baseline_best_lag_sec",
            "candidate_best_lag_sec",
        ],
    ]
    return "\n".join(
        " | ".join(f"{key}={_fmt(row.get(key))}" for key in keys if key in row)
        for keys in lines
    )


def _fmt(value: Any) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(val):
        return "nan"
    return f"{val:.3f}"


def _plot_filename(row: pd.Series) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row.get("label", "candidate"))).strip("_").lower()
    return f"row_{int(row['dataset_row_id'])}_seed_{row.get('seed', '')}_{label}.png"


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制 F-A2 护栏窗口 paired 波形图")
    parser.add_argument("--window-list", required=True, help="window_delta/top_degraded/top_improved CSV")
    parser.add_argument("--output-dir", required=True, help="PNG 输出目录")
    parser.add_argument("--candidate-label", action="append", default=None, help="可重复传入；默认不过滤")
    parser.add_argument("--filter-column", default="", help="只绘制该布尔列为 true 的窗口，例如 dirty_easy_lowspec")
    parser.add_argument("--sort-by", default="", help="筛选后按该列排序，例如 delta_rr_peak_band_abs_error")
    parser.add_argument("--sort-ascending", action="store_true", help="按升序排序；默认降序")
    parser.add_argument("--max-rows", type=int, default=30)
    parser.add_argument("--scale-mode", choices=("zscore", "robust"), default="robust")
    parser.add_argument("--device", default="", help="覆盖 run config 中的 training.device，例如 cuda:0")
    args = parser.parse_args()

    written = plot_paired_f_a2_windows(
        args.window_list,
        output_dir=args.output_dir,
        candidate_labels=args.candidate_label,
        filter_column=args.filter_column or None,
        sort_by=args.sort_by or None,
        sort_ascending=args.sort_ascending,
        max_rows=args.max_rows,
        scale_mode=args.scale_mode,
        device=args.device or None,
    )
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
