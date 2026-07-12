"""为 THO 组会讨论生成五类当前指标的逐步计算示例图。"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import textwrap
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from omegaconf import OmegaConf
import pandas as pd

from resp_train.metrics.signal import (
    _relative_envelope_trace,
    bandpass_filter,
    best_lag_correlation_from_filtered,
    estimate_robust_peak_rate_bpm,
    lag_aligned_overlap,
    lag_correlation_trace_from_filtered,
    local_rr_metrics_from_rate_traces,
    local_rr_rate_trace,
    relative_envelope_metrics,
    zero_crossing_counts,
)
from scripts.tho_group_meeting_ppt.evidence import EvidenceCatalog, build_evidence_catalog


_ASSET_KEYS = (
    "metric_robust_rr",
    "metric_cycle_count",
    "metric_lag_corr",
    "metric_relative_envelope",
    "metric_local_rr",
)
_BLUE = "#005A9C"
_ORANGE = "#D55E00"
_GREEN = "#009E73"
_PURPLE = "#7B3294"
_GRAY = "#53606B"
_FS = 100.0
_LOW_HZ = 0.05
_HIGH_HZ = 0.7
_ORDER = 4
_ENVELOPE_WINDOW_SEC = 2.0
_MAX_LAG_SEC = 4.0
_LOCAL_RR_WINDOW_SEC = 40.0
_LOCAL_RR_STEP_SEC = 10.0
_WINDOW_SEC = 180.0


def _plot_style() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    candidates = ("Noto Sans CJK SC", "Noto Sans CJK JP", "Source Han Sans CN", "DejaVu Sans")
    plt.rcParams["font.sans-serif"] = [name for name in candidates if name in available] or ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["figure.facecolor"] = "white"


def _repo_display(repo_root: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repo_root))
    except ValueError:
        # worktree 用 docs/figure 与 runs 符号链接只读复用主仓证据；manifest 记录
        # worktree 内的逻辑相对路径，避免把某台机器的绝对 checkout 路径固化进去。
        for anchor in ("docs", "runs"):
            indices = [idx for idx, part in enumerate(resolved.parts) if part == anchor]
            for idx in reversed(indices):
                relative = Path(*resolved.parts[idx:])
                candidate = repo_root / relative
                if candidate.exists() and candidate.resolve() == resolved:
                    return str(relative)
        return str(resolved)


def _file_record(repo_root: Path, path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    payload = resolved.read_bytes()
    return {
        "path": _repo_display(repo_root, resolved),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _load_demo_module(repo_root: Path):
    path = repo_root / "docs/figure/rr_peak_band_metric/plot_rr_peak_band_metric_demo.py"
    spec = importlib.util.spec_from_file_location("tho_metric_robust_rr_demo", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 robust RR demo: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _source_paths(repo_root: Path, catalog: EvidenceCatalog) -> dict[str, Path]:
    metadata_path = catalog.general_signal_npz.with_name("f0_visual_sample_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_config = (repo_root / str(metadata["source_run_config"])).resolve()
    source_checkpoint = (repo_root / str(metadata["source_checkpoint"])).resolve()
    source_metrics = source_config.parent / "metrics.csv"
    signal_assets_manifest = (
        repo_root / "docs/stage_reports/20260708/generated_assets/discussion/signal_assets_manifest.json"
    )
    for description, path in (
        ("通用信号 metadata", metadata_path),
        ("F0 源 config", source_config),
        ("F0 源 checkpoint", source_checkpoint),
        ("F0 源 metrics CSV", source_metrics),
        ("任务3 signal assets manifest", signal_assets_manifest),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{description}不存在: {path}")
    return {
        "signal_metadata_json": metadata_path,
        "source_run_config": source_config,
        "source_checkpoint": source_checkpoint,
        "source_metrics_csv": source_metrics,
        "signal_assets_manifest": signal_assets_manifest,
    }


def _read_source_metrics_row(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    matches = frame[frame["dataset_row_id"].astype(int) == 8025]
    if len(matches) != 1:
        raise ValueError(f"{path} 中 dataset_row_id=8025 匹配到 {len(matches)} 行")
    row = matches.iloc[0]
    if str(row["split"]) != "val":
        raise ValueError(f"通用 F0 资产应来自 val，CSV 实际为 {row['split']!r}: {path}")
    return row


def validate_metric_asset_identity(
    metadata: Mapping[str, Any],
    *,
    signal_npz: Path,
    source_config: Path,
    source_checkpoint: Path,
    source_metrics_row: pd.Series,
) -> dict[str, Any]:
    """交叉校验通用信号、run 与 CSV 身份，防止把别的模型资产误标为 F0。"""
    expected = {
        "dataset_row_id": 8025,
        "split": "val",
        "model_label": "F0_native_stft_pre_mixer",
    }
    for field, expected_value in expected.items():
        actual = metadata.get(field)
        if actual != expected_value:
            raise ValueError(f"通用信号 metadata {field}={actual!r}，期望 {expected_value!r}")
    files = metadata.get("files")
    if not isinstance(files, Mapping) or files.get("signal_arrays_npz") != signal_npz.name:
        raise ValueError(
            "通用信号 metadata files.signal_arrays_npz "
            f"未指向 {signal_npz.name!r}"
        )
    if int(source_metrics_row["dataset_row_id"]) != int(expected["dataset_row_id"]):
        raise ValueError("source metrics CSV dataset_row_id 与通用信号 metadata 不一致")
    if str(source_metrics_row["split"]) != str(expected["split"]):
        raise ValueError("source metrics CSV split 与通用信号 metadata 不一致")
    return dict(expected)


def _resolved_evidence_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _require_frozen_record(
    repo_root: Path,
    manifest_path: Path,
    records: Mapping[str, Any],
    key: str,
    actual_path: Path,
) -> None:
    record = records.get(key)
    if not isinstance(record, Mapping):
        raise ValueError(f"{key} record 缺失: {manifest_path}")
    recorded_path = _resolved_evidence_path(repo_root, str(record.get("path", "")))
    if recorded_path != actual_path.resolve():
        raise ValueError(f"{key}.path 与实际来源不一致: {manifest_path}")
    actual_sha = hashlib.sha256(actual_path.read_bytes()).hexdigest()
    if record.get("sha256") != actual_sha:
        raise ValueError(f"{key}.sha256 与实际文件不一致: {manifest_path}")


def validate_metric_source_evidence(
    *,
    repo_root: str | Path,
    signal_npz: str | Path,
    signal_metadata_json: str | Path,
    source_config: str | Path,
    source_checkpoint: str | Path,
    source_metrics_csv: str | Path,
    signal_assets_manifest: str | Path,
) -> dict[str, Any]:
    """绑定任务3冻结文件并交叉校验 F0 资产的配置、metadata 与 CSV 身份。"""
    root = Path(repo_root).resolve()
    signal_path = Path(signal_npz).resolve()
    metadata_path = Path(signal_metadata_json).resolve()
    config_path = Path(source_config).resolve()
    checkpoint_path = Path(source_checkpoint).resolve()
    csv_path = Path(source_metrics_csv).resolve()
    upstream_path = Path(signal_assets_manifest).resolve()
    upstream = json.loads(upstream_path.read_text(encoding="utf-8"))
    evidence = upstream.get("evidence", {})
    for key, actual in (
        ("signal_metadata_json", metadata_path),
        ("signal_npz", signal_path),
        ("source_checkpoint", checkpoint_path),
        ("source_run_config", config_path),
    ):
        _require_frozen_record(root, upstream_path, evidence, key, actual)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata_config = _resolved_evidence_path(root, str(metadata.get("source_run_config", "")))
    metadata_checkpoint = _resolved_evidence_path(root, str(metadata.get("source_checkpoint", "")))
    if metadata_config != config_path:
        raise ValueError(f"source_run_config 与实际路径不一致: {metadata_path}")
    if metadata_checkpoint != checkpoint_path:
        raise ValueError(f"source_checkpoint 与实际路径不一致: {metadata_path}")

    cfg = OmegaConf.load(config_path)
    frame = pd.read_csv(csv_path)
    matches = frame[frame["dataset_row_id"].astype(int) == int(metadata.get("dataset_row_id", -1))]
    if len(matches) != 1:
        raise ValueError(f"dataset_row_id 在 CSV 中匹配到 {len(matches)} 行: {csv_path}")
    row = matches.iloc[0]
    identity = validate_metric_asset_identity(
        metadata,
        signal_npz=signal_path,
        source_config=config_path,
        source_checkpoint=checkpoint_path,
        source_metrics_row=row,
    )

    expected_scalars = {
        "model.name": (OmegaConf.select(cfg, "model.name"), "time_stft_dual1d"),
        "data.input_set": (OmegaConf.select(cfg, "data.input_set"), "research_v2_waveform"),
        "window.target_fs": (OmegaConf.select(cfg, "window.target_fs"), metadata.get("sampling_rate_hz")),
        "window.duration_sec": (OmegaConf.select(cfg, "window.duration_sec"), metadata.get("window_duration_sec")),
        "model.stft_win": (OmegaConf.select(cfg, "model.stft_win"), metadata.get("stft_win_samples")),
        "model.stft_hop": (OmegaConf.select(cfg, "model.stft_hop"), metadata.get("stft_hop_samples")),
        "model.stft_low_hz": (OmegaConf.select(cfg, "model.stft_low_hz"), metadata.get("stft_low_hz")),
        "model.stft_high_hz": (OmegaConf.select(cfg, "model.stft_high_hz"), metadata.get("stft_high_hz")),
        "model.stft_inject_position": (OmegaConf.select(cfg, "model.stft_inject_position"), "pre_mixer"),
        "model.stft_encoder_type": (OmegaConf.select(cfg, "model.stft_encoder_type"), "conv2d"),
        "model.fusion_mode": (OmegaConf.select(cfg, "model.fusion_mode"), "native_inject"),
    }
    for field, (actual, expected) in expected_scalars.items():
        equal = (
            np.isclose(float(actual), float(expected), rtol=0.0, atol=1e-12)
            if isinstance(expected, (int, float)) and not isinstance(expected, bool)
            else actual == expected
        )
        if not equal:
            raise ValueError(f"{field}={actual!r} 与冻结证据 {expected!r} 不一致: {config_path}")

    f0_record = upstream.get("resolved_configs", {}).get("f0", {})
    for config_field, manifest_field in (
        ("model.stft_win", "stft_win"),
        ("model.stft_hop", "stft_hop"),
        ("model.stft_low_hz", "stft_low_hz"),
        ("model.stft_high_hz", "stft_high_hz"),
        ("model.stft_encoder_type", "stft_encoder_type"),
        ("model.stft_inject_position", "stft_inject_position"),
    ):
        actual = OmegaConf.select(cfg, config_field)
        expected = f0_record.get(manifest_field)
        if actual != expected:
            raise ValueError(f"{config_field} 与 signal manifest {manifest_field} 不一致: {config_path}")

    csv_expected = {
        "method": OmegaConf.select(cfg, "model.name"),
        "input_set": OmegaConf.select(cfg, "data.input_set"),
        "split": metadata["split"],
        "dataset_row_id": metadata["dataset_row_id"],
    }
    for field, expected in csv_expected.items():
        actual = row[field]
        if str(actual) != str(expected):
            raise ValueError(f"{field}={actual!r} 与 config/metadata {expected!r} 不一致: {csv_path}")

    with np.load(signal_path, allow_pickle=False) as arrays:
        required = ("f0_prediction_full", "target_respiration_full")
        for field in required:
            if field not in arrays.files:
                raise ValueError(f"NPZ 缺少 {field}: {signal_path}")
        expected_samples = int(round(float(metadata["sampling_rate_hz"]) * float(metadata["window_duration_sec"])))
        for field in required:
            if np.asarray(arrays[field]).size != expected_samples:
                raise ValueError(f"{field}.size 与 metadata fs/window 不一致: {signal_path}")
    return identity


def _major_text(artist):
    artist.set_gid("layout-key")
    return artist


def _new_metric_figure(title: str, subtitle: str):
    fig = plt.figure(figsize=(16, 9), dpi=120)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.72, 1.0),
        height_ratios=(3.25, 1.0),
        left=0.055,
        right=0.965,
        bottom=0.075,
        top=0.855,
        wspace=0.18,
        hspace=0.25,
    )
    main = fig.add_subplot(grid[0, 0])
    steps = fig.add_subplot(grid[0, 1])
    failure = fig.add_subplot(grid[1, :])
    steps.set_axis_off()
    failure.set_axis_off()
    heading = fig.suptitle(title, x=0.055, y=0.955, ha="left", fontsize=25, fontweight="bold", color="#17212B")
    _major_text(heading)
    sub = fig.text(0.055, 0.895, subtitle, ha="left", va="center", fontsize=12.5, color=_GRAY)
    _major_text(sub)
    return fig, main, steps, failure


def _step_panel(axis, lines: list[str], final_line: str) -> None:
    # 右栏宽度固定；显式换行比依赖 Matplotlib 的 renderer 自动 wrap 更可审计。
    wrapped = [textwrap.fill(line, width=28, break_long_words=True, break_on_hyphens=False) for line in lines]
    body = "\n\n".join(f"{idx + 1}. {line}" for idx, line in enumerate(wrapped))
    text = axis.text(
        0.03,
        0.95,
        body,
        ha="left",
        va="top",
        fontsize=13.0,
        linespacing=1.22,
        transform=axis.transAxes,
        color="#17212B",
    )
    _major_text(text)
    result = axis.text(
        0.03,
        0.08,
        final_line,
        ha="left",
        va="bottom",
        fontsize=15.0,
        fontweight="bold",
        linespacing=1.2,
        transform=axis.transAxes,
        color=_BLUE,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#E8F2FA", "edgecolor": "#8CBAD8"},
    )
    _major_text(result)


def _failure_panel(axis, scenario: str, *, evidence_note: str) -> None:
    text = axis.text(
        0.01,
        0.58,
        f"真实失效场景｜{scenario}\n证据边界｜{evidence_note}",
        ha="left",
        va="center",
        fontsize=12.2,
        linespacing=1.45,
        transform=axis.transAxes,
        color="#17212B",
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "#FFF5E8", "edgecolor": "#E5B66B"},
    )
    _major_text(text)


def _layout_report(fig) -> dict[str, Any]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_box = fig.bbox
    tagged = [artist for artist in fig.findobj() if getattr(artist, "get_gid", lambda: None)() == "layout-key"]
    boxes = [artist.get_window_extent(renderer) for artist in tagged]
    inside = all(
        box.x0 >= figure_box.x0 and box.y0 >= figure_box.y0 and box.x1 <= figure_box.x1 and box.y1 <= figure_box.y1
        for box in boxes
    )
    overlap_count = sum(int(left.overlaps(right)) for i, left in enumerate(boxes) for right in boxes[i + 1 :])
    return {
        "all_key_text_inside": bool(inside),
        "text_overlap_count": int(overlap_count),
        "checked_text_count": len(boxes),
    }


def _save_metric_figure(fig, path: Path) -> dict[str, Any]:
    report = _layout_report(fig)
    if not report["all_key_text_inside"] or report["text_overlap_count"]:
        plt.close(fig)
        raise RuntimeError(f"指标图文字布局失败: {path.name}: {report}")
    fig.savefig(path, dpi=120, bbox_inches=None, metadata={"Software": "matplotlib"})
    plt.close(fig)
    return report


def _robust_rr_figure(pred_details, target_details, path: Path) -> dict[str, Any]:
    error = abs(float(pred_details.rr_bpm) - float(target_details.rr_bpm))
    fig, axis, steps, failure = _new_metric_figure(
        "稳健整窗 RR：从呼吸带波形到 bpm 绝对误差",
        "row 8025｜真实 F0 validation 预测与对应 THO｜0.05–0.7 Hz，4 阶零相位 Butterworth",
    )
    time = np.arange(pred_details.filtered.size) / _FS
    axis.plot(time, target_details.filtered, color=_BLUE, lw=1.35, label="THO 带通")
    axis.plot(time, pred_details.filtered, color=_ORANGE, lw=1.05, alpha=0.86, label="F0 预测带通")
    axis.scatter(target_details.peaks / _FS, target_details.filtered[target_details.peaks], s=17, color=_BLUE)
    axis.scatter(pred_details.peaks / _FS, pred_details.filtered[pred_details.peaks], s=16, color=_ORANGE)
    axis.set(title="输入与中间量：带通波形、稳健峰与峰间距", xlabel="时间（s）", ylabel="soft-z 幅值")
    axis.legend(loc="upper right", frameon=False, ncol=2)
    axis.grid(alpha=0.18)
    _step_panel(
        steps,
        [
            "输入：预测与 THO 波形分别做呼吸带滤波。",
            "中间量：Welch 主频约束最小峰距；突出度取 max(0.2·std, 0.08·(P95−P5))。",
            "分别计算 RR = 60 / median(相邻峰间隔)，单位 bpm。",
            "聚合：取预测 RR 与目标 RR 的绝对差；数值越低表示整窗主要节律越接近。",
        ],
        f"预测 {pred_details.rr_bpm:.6f} bpm\nTHO {target_details.rr_bpm:.6f} bpm\n绝对误差 {error:.6f} bpm",
    )
    _failure_panel(
        failure,
        "弱局部伪峰、尖峰或同周期双峰仍可能改变峰集合；因此需要同时检查周期计数和波形。",
        evidence_note="本页数值由当前函数在通用 F0 资产上即时计算；row 8025 不在 canonical test CSV。",
    )
    return _save_metric_figure(fig, path)


def _crossing_indices(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    up = np.flatnonzero((signal[:-1] <= 0.0) & (signal[1:] > 0.0)) + 1
    down = np.flatnonzero((signal[:-1] >= 0.0) & (signal[1:] < 0.0)) + 1
    return up, down


def _cycle_count_figure(pred_filtered, target_filtered, pred_counts, target_counts, path: Path) -> dict[str, Any]:
    count_error = abs(int(pred_counts["cycle"]) - int(target_counts["cycle"]))
    bpm_error = count_error / (_WINDOW_SEC / 60.0)
    fig, axis, steps, failure = _new_metric_figure(
        "周期数量：上升/下降过零如何合成呼吸周期数",
        "row 8025｜周期数误差单位为 cycle；180 秒窗口可除以 3 换算为平均周期率误差（bpm）",
    )
    time = np.arange(pred_filtered.size) / _FS
    pred_up, _ = _crossing_indices(pred_filtered)
    target_up, _ = _crossing_indices(target_filtered)
    axis.plot(time, target_filtered, color=_BLUE, lw=1.2, label="THO 带通")
    axis.plot(time, pred_filtered, color=_ORANGE, lw=1.0, alpha=0.82, label="F0 预测带通")
    axis.scatter(target_up / _FS, np.zeros(target_up.size), marker="|", s=85, color=_BLUE, label="THO 上升过零")
    axis.scatter(pred_up / _FS, np.zeros(pred_up.size), marker="|", s=65, color=_ORANGE, label="预测上升过零")
    axis.axhline(0.0, color="#333333", lw=0.7)
    axis.set(title="输入与中间量：带通波形及上升过零位置", xlabel="时间（s）", ylabel="soft-z 幅值")
    axis.legend(loc="upper right", frameon=False, ncol=2)
    axis.grid(alpha=0.18)
    _step_panel(
        steps,
        [
            "输入：对两侧波形做同一呼吸带滤波。",
            "中间量：分别统计上升过零 N_up 与下降过零 N_down。",
            "周期数 N_cycle = floor((N_up + N_down)/2 + 0.5)。",
            "聚合：|N_pred − N_target|；再除以 180/60=3 得到 bpm 展示量。",
        ],
        f"预测 up/down={pred_counts['up']}/{pred_counts['down']} → {pred_counts['cycle']} cycles\n"
        f"THO up/down={target_counts['up']}/{target_counts['down']} → {target_counts['cycle']} cycles\n"
        f"误差 {count_error} cycle = {bpm_error:.6f} bpm",
    )
    _failure_panel(
        failure,
        "低幅抖动在零附近反复穿越时会增加过零数；它不评价每次呼吸事件的时间位置。",
        evidence_note="CSV 中同源 row 8025 也记录 51/51 cycles；本页仍按当前函数从 NPZ 重算。",
    )
    return _save_metric_figure(fig, path)


def _lag_figure(pred_filtered, target_filtered, lag_metrics, path: Path) -> dict[str, Any]:
    best_lag = float(lag_metrics["best_lag_sec"])
    best_corr = float(lag_metrics["best_lag_corr"])
    trace = lag_correlation_trace_from_filtered(
        pred_filtered,
        target_filtered,
        fs=_FS,
        max_lag_sec=_MAX_LAG_SEC,
        low_hz=_LOW_HZ,
    )
    lag_sec = trace["lag_sec"]
    corr = trace["correlation"]
    fig, axis, steps, failure = _new_metric_figure(
        "4 秒时延校正相关：在允许范围内搜索低频形态对齐",
        "row 8025｜相关系数无量纲；正 lag 表示预测相对 THO 滞后｜数值高表示对齐后形态更相似",
    )
    axis.plot(lag_sec, corr, color=_PURPLE, lw=2.0)
    axis.scatter([best_lag], [best_corr], color=_ORANGE, s=75, zorder=3, label="当前函数选中的 lag")
    axis.axvline(0.0, color="#333333", lw=0.8, ls="--")
    axis.set(title="中间量：−4 s 到 +4 s 的重叠片段相关曲线", xlabel="lag（s）", ylabel="Pearson r（无量纲）", ylim=(-1.02, 1.02))
    axis.legend(loc="lower right", frameon=False)
    axis.grid(alpha=0.2)
    _step_panel(
        steps,
        [
            "输入：预测与 THO 先做同一呼吸带滤波。",
            "中间量：对每个采样点级 lag，只取两侧重叠片段。",
            "公式：ρ(k)=corr(pred_shifted(k), target)，搜索 |k|≤4·fs。",
            "聚合：逐一检查 −400…400 的 801 个整数 lag；相关近似并列时先选 |lag| 更小者，再选负 lag。",
        ],
        f"best lag = {best_lag:+.2f} s\nbest-lag corr = {best_corr:.6f}\n相关系数：无量纲",
    )
    _failure_panel(
        failure,
        "周期信号可能在错误的整周期偏移处也得到高相关；因此必须把相关值与最佳 lag 一起报告。",
        evidence_note="本页用当前 4 秒搜索函数计算；评价 mask 只用于评价口径，不代表 masked training loss。",
    )
    return _save_metric_figure(fig, path)


def _relative_envelope_figure(pred, target, lag_metrics, rel_metrics, path: Path) -> dict[str, Any]:
    lag_samples = int(round(float(lag_metrics["best_lag_sec"]) * _FS))
    pred_overlap, target_overlap = lag_aligned_overlap(pred, target, lag_samples=lag_samples)
    env_window = int(round(_FS * _ENVELOPE_WINDOW_SEC))
    trend_window = int(round(_FS * 20.0))
    pred_rel = _relative_envelope_trace(pred_overlap, env_window, trend_window)
    target_rel = _relative_envelope_trace(target_overlap, env_window, trend_window)
    time = np.arange(pred_rel.size) / _FS
    fig, axis, steps, failure = _new_metric_figure(
        "相对包络：比较呼吸增强与减弱，而不是绝对幅值",
        "row 8025｜先按 best lag 对齐｜相对轨迹为 log(RMS包络 / 约40秒趋势)，无量纲",
    )
    axis.plot(time, target_rel, color=_BLUE, lw=1.5, label="THO 相对包络")
    axis.plot(time, pred_rel, color=_GREEN, lw=1.25, label="F0 预测相对包络")
    axis.axhline(0.0, color="#333333", lw=0.7)
    axis.fill_between(time, pred_rel, target_rel, color="#B8D8C6", alpha=0.28, label="逐点绝对差区域")
    axis.set(title="中间量：lag 对齐后的 log-ratio 相对包络轨迹", xlabel="重叠片段时间（s）", ylabel="log-ratio（无量纲）")
    axis.legend(loc="upper right", frameon=False, ncol=2)
    axis.grid(alpha=0.18)
    _step_panel(
        steps,
        [
            "输入：按 best_lag_sec_4s 截取预测与 THO 的重叠片段。",
            "中间量：计算 2 秒 RMS 包络，再以 20 秒参数的 2 倍窗口形成约 40 秒趋势。",
            "相对轨迹 r(t)=log(max(env, ε)/max(trend, ε))，消除整体幅度缩放。",
            "聚合：corr(r_pred,r_target) 与 mean(|r_pred−r_target|)。",
        ],
        f"relative-envelope corr = {rel_metrics['relative_envelope_corr']:.6f}\n"
        f"relative-envelope MAE = {rel_metrics['relative_envelope_mae']:.6f}\n"
        "corr 与 log-ratio MAE：无量纲",
    )
    _failure_panel(
        failure,
        "趋势窗口可能吸收持续时间较长的真实增强；低包络区的比值也会对微小扰动敏感。",
        evidence_note="当前 schema 以 lag-aware 列解释强弱；本页不是 canonical 汇总行。",
    )
    return _save_metric_figure(fig, path)


def _local_rr_figure(pred_rates, target_rates, local_metrics, path: Path) -> dict[str, Any]:
    centers = np.linspace(_LOCAL_RR_WINDOW_SEC / 2.0, _WINDOW_SEC - _LOCAL_RR_WINDOW_SEC / 2.0, pred_rates.size)
    valid = np.isfinite(pred_rates) & np.isfinite(target_rates)
    fig, axis, steps, failure = _new_metric_figure(
        "局部 RR：40 秒窗口、10 秒步长追踪三分钟内节律",
        "row 8025｜canonical local RR v2：spectral-guided robust peaks｜legacy 20秒/5秒与过零 v3 不进入本页",
    )
    axis.plot(centers, target_rates, "o-", color=_BLUE, lw=1.8, ms=5, label="THO local RR")
    axis.plot(centers, pred_rates, "o-", color=_ORANGE, lw=1.6, ms=5, label="F0 预测 local RR")
    axis.fill_between(centers, pred_rates, target_rates, where=valid, color="#E8C2AE", alpha=0.30)
    axis.set(title="中间量：每个局部窗口的稳健 RR 曲线", xlabel="局部窗口中心时间（s）", ylabel="RR（bpm）")
    axis.legend(loc="upper right", frameon=False)
    axis.grid(alpha=0.2)
    _step_panel(
        steps,
        [
            "输入：对预测与 THO 的 180 秒波形做呼吸带滤波。",
            "中间量：40 秒窗、10 秒步长；每窗调用 spectral-guided 稳健峰间距估计。",
            "有效集合 V：只保留两侧 local RR 都是有限值的窗口。",
            "聚合：MAE=mean_V(|RR_pred−RR_target|)，同时报告 corr 与 |V|/N。",
        ],
        f"local RR MAE = {local_metrics['local_rr_mae']:.6f} bpm\n"
        f"local RR corr = {local_metrics['local_rr_corr']:.6f}\n"
        f"valid fraction = {local_metrics['local_rr_valid_frac']:.6f}",
    )
    _failure_panel(
        failure,
        "既有复核中 target 侧 80/2310 个窗口出现窗内 RR 范围跳变；谐波抢峰与谱峰硬切换仍会制造局部长尾。",
        evidence_note="当前 40秒/10秒来自 signal.py/evaluate.py 默认值与 2026-07-09 schema；本页数值来自 row 8025。",
    )
    return _save_metric_figure(fig, path)


def _write_manifest(
    repo_root: Path,
    output_dir: Path,
    catalog: EvidenceCatalog,
    source_paths: Mapping[str, Path],
    assets: Mapping[str, Path],
    values: Mapping[str, Any],
) -> None:
    sources = {name: _file_record(repo_root, path) for name, path in source_paths.items()}
    sources.update(
        {
            "signal_npz": _file_record(repo_root, catalog.general_signal_npz),
            "metric_code": _file_record(repo_root, repo_root / "resp_train/metrics/signal.py"),
            "metric_evaluate_code": _file_record(repo_root, repo_root / "resp_train/metrics/evaluate.py"),
            "metric_figure_code": {
                **_file_record(repo_root, Path(__file__)),
                "status": "present",
            },
            "robust_rr_demo_code": _file_record(
                repo_root, repo_root / "docs/figure/rr_peak_band_metric/plot_rr_peak_band_metric_demo.py"
            ),
            "metric_schema": _file_record(repo_root, repo_root / "docs/experiments/metric_schema.md"),
        }
    )
    manifest = {
        "schema_version": 1,
        "dataset_row_id": 8025,
        "evidence_scope": values["evidence_scope"],
        "canonical_csv_alignment": False,
        "evidence_gap": values["evidence_gap"],
        "sources": sources,
        "parameters": values["parameters"],
        "values": {key: value for key, value in values.items() if key not in {"layout", "parameters"}},
        "layout": values["layout"],
        "assets": {key: _file_record(repo_root, path) for key, path in assets.items()},
    }
    (output_dir / "metric_assets_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def build_metric_assets(
    repo_root: str | Path,
    output_dir: str | Path,
) -> tuple[dict[str, Path], dict[str, Any]]:
    """生成五张指标计算图；数值全部来自同一 row 8025 F0 validation 资产。"""
    root = Path(repo_root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    catalog = build_evidence_catalog(root)
    if catalog.general_sample_row_id != 8025:
        raise ValueError(f"通用信号资产 row_id 应为 8025，实际 {catalog.general_sample_row_id}")
    source_paths = _source_paths(root, catalog)
    source_row = _read_source_metrics_row(source_paths["source_metrics_csv"])
    identity = validate_metric_source_evidence(
        repo_root=root,
        signal_npz=catalog.general_signal_npz,
        signal_metadata_json=source_paths["signal_metadata_json"],
        source_config=source_paths["source_run_config"],
        source_checkpoint=source_paths["source_checkpoint"],
        source_metrics_csv=source_paths["source_metrics_csv"],
        signal_assets_manifest=source_paths["signal_assets_manifest"],
    )
    with np.load(catalog.general_signal_npz, allow_pickle=False) as data:
        pred = np.asarray(data["f0_prediction_full"], dtype=np.float64).reshape(-1)
        target = np.asarray(data["target_respiration_full"], dtype=np.float64).reshape(-1)
    if pred.shape != target.shape or pred.size != int(_FS * _WINDOW_SEC):
        raise ValueError(f"row 8025 预测/THO shape 非法: pred={pred.shape}, target={target.shape}")

    pred_filtered = bandpass_filter(pred, fs=_FS, low_hz=_LOW_HZ, high_hz=_HIGH_HZ, order=_ORDER)
    target_filtered = bandpass_filter(target, fs=_FS, low_hz=_LOW_HZ, high_hz=_HIGH_HZ, order=_ORDER)
    demo = _load_demo_module(root)
    pred_details = demo._bandpassed_peak_details(pred, fs=_FS, low_hz=_LOW_HZ, high_hz=_HIGH_HZ, order=_ORDER)
    target_details = demo._bandpassed_peak_details(target, fs=_FS, low_hz=_LOW_HZ, high_hz=_HIGH_HZ, order=_ORDER)
    pred_rr = estimate_robust_peak_rate_bpm(pred_filtered, fs=_FS, low_hz=_LOW_HZ, high_hz=_HIGH_HZ)
    target_rr = estimate_robust_peak_rate_bpm(target_filtered, fs=_FS, low_hz=_LOW_HZ, high_hz=_HIGH_HZ)
    if not np.isclose(pred_details.rr_bpm, pred_rr, atol=1e-12) or not np.isclose(target_details.rr_bpm, target_rr, atol=1e-12):
        raise RuntimeError("robust RR demo 中间量与当前仓库函数不一致")

    pred_counts = zero_crossing_counts(pred_filtered)
    target_counts = zero_crossing_counts(target_filtered)
    if int(source_row["pred_breath_count_zero_cross"]) != pred_counts["cycle"] or int(
        source_row["target_breath_count_zero_cross"]
    ) != target_counts["cycle"]:
        raise RuntimeError("同源 CSV 的周期计数与通用 F0 NPZ 不一致")
    lag_metrics = best_lag_correlation_from_filtered(
        pred_filtered, target_filtered, fs=_FS, max_lag_sec=_MAX_LAG_SEC, low_hz=_LOW_HZ
    )
    lag_samples = int(round(float(lag_metrics["best_lag_sec"]) * _FS))
    pred_overlap, target_overlap = lag_aligned_overlap(pred, target, lag_samples=lag_samples)
    rel_metrics = relative_envelope_metrics(
        pred_overlap, target_overlap, fs=_FS, envelope_window_sec=_ENVELOPE_WINDOW_SEC
    )
    pred_rates = local_rr_rate_trace(
        pred_filtered,
        fs=_FS,
        window_sec=_LOCAL_RR_WINDOW_SEC,
        step_sec=_LOCAL_RR_STEP_SEC,
        low_hz=_LOW_HZ,
        high_hz=_HIGH_HZ,
    )
    target_rates = local_rr_rate_trace(
        target_filtered,
        fs=_FS,
        window_sec=_LOCAL_RR_WINDOW_SEC,
        step_sec=_LOCAL_RR_STEP_SEC,
        low_hz=_LOW_HZ,
        high_hz=_HIGH_HZ,
    )
    local_metrics = local_rr_metrics_from_rate_traces(pred_rates, target_rates)

    assets = {key: output / f"{key}.png" for key in _ASSET_KEYS}
    _plot_style()
    layout = {
        "metric_robust_rr": _robust_rr_figure(pred_details, target_details, assets["metric_robust_rr"]),
        "metric_cycle_count": _cycle_count_figure(
            pred_filtered, target_filtered, pred_counts, target_counts, assets["metric_cycle_count"]
        ),
        "metric_lag_corr": _lag_figure(pred_filtered, target_filtered, lag_metrics, assets["metric_lag_corr"]),
        "metric_relative_envelope": _relative_envelope_figure(
            pred, target, lag_metrics, rel_metrics, assets["metric_relative_envelope"]
        ),
        "metric_local_rr": _local_rr_figure(pred_rates, target_rates, local_metrics, assets["metric_local_rr"]),
    }
    count_error = abs(pred_counts["cycle"] - target_counts["cycle"])
    values: dict[str, Any] = {
        "dataset_row_id": int(identity["dataset_row_id"]),
        "sample_split": str(identity["split"]),
        "sample_model_label": str(identity["model_label"]),
        "evidence_scope": "同一通用 F0 validation 预测资产上的逐步计算示例，不是 canonical test 汇总行",
        "canonical_csv_alignment": False,
        "evidence_gap": (
            "row 8025 不在 canonical G-series test CSV；同源旧 metrics.csv 仅含当时已有列，"
            "不含当前 robust RR、4s lag-aware relative envelope 与 canonical local RR。"
        ),
        "pred_robust_rr_bpm": float(pred_rr),
        "target_robust_rr_bpm": float(target_rr),
        "robust_rr_abs_error": float(abs(pred_rr - target_rr)),
        "pred_cycle_count": int(pred_counts["cycle"]),
        "target_cycle_count": int(target_counts["cycle"]),
        "cycle_count_abs_error": int(count_error),
        "cycle_count_bpm_error": float(count_error / (_WINDOW_SEC / 60.0)),
        "best_lag_corr_4s": float(lag_metrics["best_lag_corr"]),
        "best_lag_sec_4s": float(lag_metrics["best_lag_sec"]),
        "relative_envelope_corr_lag4s": float(rel_metrics["relative_envelope_corr"]),
        "relative_envelope_mae_lag4s": float(rel_metrics["relative_envelope_mae"]),
        "local_rr_mae": float(local_metrics["local_rr_mae"]),
        "local_rr_corr": float(local_metrics["local_rr_corr"]),
        "local_rr_valid_frac": float(local_metrics["local_rr_valid_frac"]),
        "parameters": {
            "sample_rate_hz": _FS,
            "window_sec": _WINDOW_SEC,
            "band_hz": [_LOW_HZ, _HIGH_HZ],
            "bandpass_order": _ORDER,
            "max_lag_sec": _MAX_LAG_SEC,
            "lag_trace_points": 801,
            "lag_trace_integer_samples": [-400, 400],
            "envelope_window_sec": _ENVELOPE_WINDOW_SEC,
            "relative_envelope_trend_parameter_sec": 20.0,
            "relative_envelope_effective_smoothing_sec": 40.0,
            "local_rr_window_sec": _LOCAL_RR_WINDOW_SEC,
            "local_rr_step_sec": _LOCAL_RR_STEP_SEC,
            "figure_size_pixels": [1920, 1080],
        },
        "layout": layout,
    }
    _write_manifest(root, output, catalog, source_paths, assets, values)
    return assets, values


if __name__ == "__main__":
    repository = Path(__file__).resolve().parents[2]
    destination = repository / "docs/stage_reports/20260708/generated_assets/discussion"
    generated, details = build_metric_assets(repository, destination)
    print(json.dumps({"assets": {key: str(path) for key, path in generated.items()}, "values": details}, ensure_ascii=False, indent=2))
