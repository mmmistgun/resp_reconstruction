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
from scipy import signal as scipy_signal
from torch.utils.data import DataLoader

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
    """为一个 run 的诊断预测生成 PNG 图，按需从 checkpoint 重新推理选中窗口。"""
    run_path = Path(run_dir)
    metrics_path = run_path / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"指标文件不存在: {metrics_path}")

    metrics = pd.read_csv(metrics_path)
    if int(max_plots) <= 0:
        raise ValueError("max_plots 必须大于 0")

    out_dir = Path(output_dir) if output_dir is not None else run_path / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _load_config(run_path)
    fs = float(OmegaConf.select(cfg, "window.target_fs", default=100.0))
    spectrum_high_hz = float(OmegaConf.select(cfg, "loss.spectrum_high_hz", default=0.7))

    selected_row_ids = _select_metric_row_ids(metrics, max_plots=int(max_plots), sort_by=sort_by)
    prediction_lookup = _infer_prediction_lookup(run_path, cfg, selected_row_ids)
    written: list[Path] = []
    for plot_idx, row_id in enumerate(selected_row_ids):
        metric_row = _metric_row(metrics, row_id)
        prediction = prediction_lookup[row_id]
        output = out_dir / f"window_{plot_idx:03d}_row_{row_id}.png"
        _plot_one_window(
            output,
            pred=np.asarray(prediction["pred"]).reshape(-1),
            target=np.asarray(prediction["target"]).reshape(-1),
            x=None if prediction.get("x") is None else np.asarray(prediction["x"]).reshape(-1),
            fs=fs,
            spectrum_high_hz=spectrum_high_hz,
            meta={key: str(prediction.get("meta", {}).get(key, "")) for key in ("split", "input_set", "residual_quality_class")},
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


def _select_metric_row_ids(metrics: pd.DataFrame, *, max_plots: int, sort_by: str) -> list[int]:
    """从 metrics 中选择需要绘制的 dataset_row_id。"""
    if "dataset_row_id" not in metrics.columns:
        raise KeyError("metrics.csv 必须包含 dataset_row_id")
    ranked = metrics.copy()
    if sort_by in ranked.columns:
        ranked = ranked.sort_values(sort_by, ascending=False, na_position="last")
    return [int(row_id) for row_id in ranked["dataset_row_id"].head(max_plots).tolist()]


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


def _infer_prediction_lookup(run_path: Path, cfg, row_ids: list[int]) -> dict[int, dict[str, Any]]:
    """只对选中的窗口重新推理，避免保存全量 predictions.npz。"""
    checkpoint_path = run_path / "checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint.pt 不存在，无法重新推理诊断图: {checkpoint_path}")
    dataset = _load_dataset_for_rows(cfg, row_ids)
    if len(dataset) == 0:
        raise ValueError(f"未在数据索引中找到待绘制 row: {row_ids}")

    import torch

    from resp_train.engine import collect_predictions
    from resp_train.models.registry import build_model

    device = torch.device("cpu")
    model = build_model(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    loader = DataLoader(dataset, batch_size=min(len(dataset), 16), shuffle=False, num_workers=0)
    predictions = collect_predictions(model, loader, device=device, max_windows=len(dataset))
    input_lookup = _load_input_lookup(run_path, predictions, cfg)
    lookup: dict[int, dict[str, Any]] = {}
    for idx, row_id in enumerate(np.asarray(predictions["dataset_row_id"], dtype=np.int64).tolist()):
        row_key = int(row_id)
        lookup[row_key] = {
            "pred": np.asarray(predictions["r_tho_hat"][idx]).reshape(-1),
            "target": np.asarray(predictions["tho_ref"][idx]).reshape(-1),
            "x": input_lookup.get(row_key),
            "meta": {key: _prediction_meta(predictions, key, idx) for key in ("split", "input_set", "residual_quality_class")},
        }
    missing = [row_id for row_id in row_ids if row_id not in lookup]
    if missing:
        raise ValueError(f"重新推理后仍缺少待绘制 row: {missing}")
    return lookup


def _load_dataset_for_rows(cfg, row_ids: list[int]):
    from resp_train.data.audit import add_usable_flag
    from resp_train.data.dataset import RespWindowDataset
    from resp_train.data.index import read_index
    from resp_train.data.research_v2 import ResearchV2WindowDataset, read_research_v2_index

    dataset_root = OmegaConf.select(cfg, "data.dataset_root")
    index_csv = OmegaConf.select(cfg, "data.index_csv")
    if not dataset_root or not index_csv:
        raise ValueError("配置缺少 data.dataset_root 或 data.index_csv，无法重新推理诊断图")
    if str(OmegaConf.select(cfg, "data.format", default="stage2_1")) == "research_v2":
        df = read_research_v2_index(dataset_root, index_csv, cfg)
        dataset_cls = ResearchV2WindowDataset
    else:
        df = add_usable_flag(read_index(dataset_root, index_csv), cfg)
        dataset_cls = RespWindowDataset
    selected = df[df["dataset_row_id"].isin(set(row_ids))].copy()
    order = {int(row_id): idx for idx, row_id in enumerate(row_ids)}
    selected["_plot_order"] = selected["dataset_row_id"].map(lambda value: order[int(value)])
    selected = selected.sort_values("_plot_order").drop(columns=["_plot_order"])
    return dataset_cls(Path(str(dataset_root)) / str(index_csv), selected, cfg, preload_windows=False)


def _load_input_lookup(run_path: Path, predictions: dict[str, np.ndarray], cfg) -> dict[int, np.ndarray]:
    """尽量从训练配置和索引还原输入 BCG；失败时返回空字典。"""
    try:
        from resp_train.data.audit import add_usable_flag
        from resp_train.data.dataset import RespWindowDataset
        from resp_train.data.index import read_index
        from resp_train.data.research_v2 import ResearchV2WindowDataset, read_research_v2_index

        dataset_root = OmegaConf.select(cfg, "data.dataset_root")
        index_csv = OmegaConf.select(cfg, "data.index_csv")
        if not dataset_root or not index_csv:
            return {}
        if str(OmegaConf.select(cfg, "data.format", default="stage2_1")) == "research_v2":
            df = read_research_v2_index(dataset_root, index_csv, cfg)
            dataset_cls = ResearchV2WindowDataset
        else:
            df = add_usable_flag(read_index(dataset_root, index_csv), cfg)
            dataset_cls = RespWindowDataset
        row_ids = set(np.asarray(predictions.get("dataset_row_id", []), dtype=np.int64).tolist())
        rows = df[df["dataset_row_id"].isin(row_ids)].copy()
        if rows.empty:
            return {}
        dataset = dataset_cls(Path(str(dataset_root)) / str(index_csv), rows, cfg, preload_windows=False)
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
    spectrum_high_hz: float,
    meta: dict[str, str],
    metrics: dict[str, Any],
    row_id: int,
) -> None:
    time = np.arange(pred.size) / fs
    lowpass_high_hz = float(np.clip(spectrum_high_hz, 0.01, fs * 0.45))
    pred_z = _zscore(pred)
    target_z = _zscore(target)
    pred_low_z = _zscore(_lowpass(pred, fs=fs, high_hz=lowpass_high_hz))
    residual_z = pred_z - target_z

    n_axes = 4 if x is not None else 3
    fig, axes = plt.subplots(n_axes, 1, figsize=(14, 3.5 * n_axes), sharex=False)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    axis_idx = 0
    if x is not None:
        x_time = np.arange(x.size) / fs
        axes[axis_idx].plot(x_time, _clip_for_display(_zscore(x)), color="#666666", linewidth=0.7, label="bcg input z")
        axes[axis_idx].set_ylabel("input")
        axes[axis_idx].set_ylim(-4, 4)
        axes[axis_idx].grid(True, alpha=0.18)
        axes[axis_idx].legend(loc="upper right")
        axis_idx += 1

    wave_ax = axes[axis_idx]
    wave_ax.plot(time, target_z, color="#1f77b4", linewidth=1.0, label="tho_ref z")
    wave_ax.plot(time, pred_z, color="#d62728", linewidth=1.0, alpha=0.82, label="r_tho_hat z")
    wave_ax.plot(time, pred_low_z, color="#2ca02c", linewidth=1.1, linestyle="--", alpha=0.9, label=f"pred lowpass <= {lowpass_high_hz:.2f}Hz")
    wave_ax.set_ylabel("waveform")
    wave_ax.set_ylim(-3.2, 3.2)
    wave_ax.grid(True, alpha=0.18)
    wave_ax.legend(loc="upper right")
    axis_idx += 1

    residual_ax = axes[axis_idx]
    residual_ax.plot(time, residual_z, color="#9467bd", linewidth=0.8, label="pred - target z")
    residual_ax.axhline(0.0, color="#888888", linewidth=0.7, alpha=0.8)
    residual_ax.set_ylabel("residual")
    residual_ax.set_ylim(-3.2, 3.2)
    residual_ax.grid(True, alpha=0.18)
    residual_ax.legend(loc="upper right")
    axis_idx += 1

    spec_ax = axes[axis_idx]
    _plot_spectrum(spec_ax, pred=pred, target=target, fs=fs, cutoff_hz=lowpass_high_hz)
    spec_ax.set_xlabel("frequency (Hz)")

    axes[-2].set_xlabel("time (s)")
    fig.suptitle(_short_title(row_id=row_id, meta=meta), fontsize=13, y=0.995)
    fig.text(
        0.01,
        0.965,
        _metric_text(_metric_row=metrics, predictions={}, pred_idx=0),
        ha="left",
        va="top",
        fontsize=9,
        family="monospace",
        bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.9, "boxstyle": "round,pad=0.35"},
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, dpi=140)
    plt.close(fig)


def _plot_spectrum(ax, *, pred: np.ndarray, target: np.ndarray, fs: float, cutoff_hz: float) -> None:
    pred_freqs, pred_power = _normalized_power_spectrum(pred, fs=fs)
    target_freqs, target_power = _normalized_power_spectrum(target, fs=fs)
    max_hz = min(2.0, fs / 2.0)
    pred_mask = pred_freqs <= max_hz
    target_mask = target_freqs <= max_hz
    ax.plot(target_freqs[target_mask], target_power[target_mask], color="#1f77b4", linewidth=1.0, label="target spectrum")
    ax.plot(pred_freqs[pred_mask], pred_power[pred_mask], color="#d62728", linewidth=1.0, alpha=0.85, label="pred spectrum")
    ax.axvline(cutoff_hz, color="#333333", linewidth=0.9, linestyle=":", label=f"{cutoff_hz:.2f}Hz cutoff")
    ax.set_xlim(0, max_hz)
    ax.set_ylabel("norm power")
    ax.grid(True, alpha=0.18)
    ax.legend(loc="upper right")


def _normalized_power_spectrum(values: np.ndarray, *, fs: float) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    centered = arr - float(np.mean(arr))
    power = np.abs(np.fft.rfft(centered)) ** 2
    freqs = np.fft.rfftfreq(centered.size, d=1.0 / fs)
    return freqs, power / max(float(power.max()), 1e-12)


def _lowpass(values: np.ndarray, *, fs: float, high_hz: float, order: int = 4) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size < max(16, order * 6) or high_hz >= fs / 2.0:
        return arr.copy()
    sos = scipy_signal.butter(order, high_hz, btype="lowpass", fs=fs, output="sos")
    return scipy_signal.sosfiltfilt(sos, arr)


def _zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    std = float(np.std(arr))
    if std <= 0:
        return arr - float(np.mean(arr))
    return (arr - float(np.mean(arr))) / std


def _clip_for_display(values: np.ndarray, limit: float = 4.0) -> np.ndarray:
    return np.clip(values, -limit, limit)


def _short_title(*, row_id: int, meta: dict[str, str]) -> str:
    parts = [
        f"row={row_id}",
        meta.get("split", ""),
        meta.get("input_set", ""),
        meta.get("residual_quality_class", ""),
    ]
    return " | ".join(str(part) for part in parts if str(part))


def _metric_text(*, _metric_row: dict[str, Any], predictions: dict[str, np.ndarray], pred_idx: int) -> str:
    del predictions, pred_idx
    parts: list[str] = []
    if "pred_rr_spec_bpm" in _metric_row and "target_rr_spec_bpm" in _metric_row:
        parts.append(
            f"rr_spec={_format_bpm(_metric_row['pred_rr_spec_bpm'])}/{_format_bpm(_metric_row['target_rr_spec_bpm'])} bpm"
        )
    if "pred_rr_peak_bpm" in _metric_row and "target_rr_peak_bpm" in _metric_row:
        parts.append(
            f"rr_peak={_format_bpm(_metric_row['pred_rr_peak_bpm'])}/{_format_bpm(_metric_row['target_rr_peak_bpm'])} bpm"
        )
    for key in (
        "rr_spec_abs_error",
        "rr_peak_abs_error",
        "envelope_corr",
        "relative_envelope_corr",
        "relative_envelope_mae",
        "spectrum_similarity",
    ):
        if key in _metric_row:
            parts.append(f"{key}={_format_metric(_metric_row[key])}")
    return " | ".join(parts)


def _format_metric(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value):
        return "nan"
    return f"{value:.3f}"


def _format_bpm(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value):
        return "nan"
    return f"{value:.1f}"


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
