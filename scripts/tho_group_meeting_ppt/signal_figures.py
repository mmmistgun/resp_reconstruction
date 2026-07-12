"""为 THO 组会讨论生成可追溯的信号、预处理与时频表示图。"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
import numpy as np
from omegaconf import OmegaConf
import torch

from scripts.tho_group_meeting_ppt.evidence import EvidenceCatalog, build_evidence_catalog


BANDENERGY_BANDS_HZ: tuple[tuple[float, float], ...] = (
    (0.05, 0.3),
    (0.1, 0.7),
    (0.3, 1.2),
    (0.7, 3.0),
    (3.0, 8.0),
)
_ASSET_KEYS = (
    "signal_overview",
    "preprocessing_comparison",
    "softz_mapping",
    "stft_resolution_comparison",
    "bandenergy_response",
)
_COLORS = ("#005A9C", "#D55E00", "#009E73", "#7B3294", "#CC79A7")


@dataclass(frozen=True)
class StftLogMagnitude:
    log_magnitude: np.ndarray
    frequencies_hz: np.ndarray
    times_sec: np.ndarray
    frequency_resolution_hz: float
    center: bool
    win_samples: int
    hop_samples: int


@dataclass(frozen=True)
class PreprocessingWindow:
    row_id: int
    start_sec: float
    end_sec: float
    sample_rate_hz: float
    bcg_raw: np.ndarray
    bcg_robust: np.ndarray
    tho_raw: np.ndarray
    tho_robust: np.ndarray
    tho_soft: np.ndarray
    bcg_center_median: float | None
    bcg_scale_median: float | None
    softz_params: Mapping[str, Any]


def _band_mask(frequencies_hz: np.ndarray, low_hz: float, high_hz: float) -> np.ndarray:
    # 与 stft_branch._band_indices 一致：上下边界频点都纳入切片。
    tolerance = np.finfo(np.float32).eps * max(1.0, abs(low_hz), abs(high_hz)) * 8
    return (frequencies_hz >= low_hz - tolerance) & (frequencies_hz <= high_hz + tolerance)


def compute_stft_logmag(
    signal: np.ndarray,
    *,
    sample_rate_hz: float,
    win_samples: int,
    hop_samples: int,
    low_hz: float,
    high_hz: float,
    center: bool = True,
) -> StftLogMagnitude:
    """按 ``STFTEncoder.forward`` 的 Hann-window/center/log1p 语义计算 STFT。"""
    waveform = torch.as_tensor(np.asarray(signal, dtype=np.float32).reshape(-1))
    window = torch.hann_window(int(win_samples), dtype=waveform.dtype)
    spectrum = torch.stft(
        waveform,
        n_fft=int(win_samples),
        hop_length=int(hop_samples),
        win_length=int(win_samples),
        window=window,
        center=bool(center),
        return_complex=True,
    )
    frequencies = torch.fft.rfftfreq(int(win_samples), d=1.0 / float(sample_rate_hz)).numpy()
    keep = _band_mask(frequencies, float(low_hz), float(high_hz))
    log_magnitude = torch.log1p(spectrum.abs()).numpy()[keep]
    frame_count = int(log_magnitude.shape[1])
    if center:
        times = np.arange(frame_count, dtype=np.float32) * float(hop_samples) / float(sample_rate_hz)
    else:
        times = (
            np.arange(frame_count, dtype=np.float32) * float(hop_samples) + float(win_samples) / 2
        ) / float(sample_rate_hz)
    return StftLogMagnitude(
        log_magnitude=log_magnitude,
        frequencies_hz=frequencies[keep],
        times_sec=times,
        frequency_resolution_hz=float(sample_rate_hz) / int(win_samples),
        center=bool(center),
        win_samples=int(win_samples),
        hop_samples=int(hop_samples),
    )


def bandenergy_from_logmag(
    log_magnitude: np.ndarray,
    frequencies_hz: np.ndarray,
    bands_hz: Sequence[tuple[float, float]] = BANDENERGY_BANDS_HZ,
) -> np.ndarray:
    """复制 STFTEncoder bandenergy 的重叠频带内 log-magnitude 均值。"""
    values = np.asarray(log_magnitude, dtype=np.float32)
    frequencies = np.asarray(frequencies_hz, dtype=np.float32).reshape(-1)
    if values.ndim != 2 or values.shape[0] != frequencies.size:
        raise ValueError("log_magnitude 必须为 (frequency, time)，且频率轴长度一致")
    energies: list[np.ndarray] = []
    for low, high in bands_hz:
        mask = _band_mask(frequencies, float(low), float(high))
        if not mask.any():
            raise ValueError(f"频带 ({low}, {high}) 没有可用频点")
        energies.append(values[mask].mean(axis=0))
    return np.stack(energies, axis=0)


def _catalog_row(catalog: EvidenceCatalog, row_id: int = 8025) -> dict[str, str]:
    with catalog.dataset_index.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        row = next((item for item in reader if int(item.get("dataset_row_id", -1)) == row_id), None)
    if row is None:
        raise ValueError(f"dataset index 中没有 dataset_row_id={row_id}: {catalog.dataset_index}")
    return row


def _softz_parameters(dataset_root: Path) -> dict[str, Any]:
    provenance = dataset_root / "provenance" / "config_default.yaml"
    if not provenance.is_file():
        return {"provenance_path": str(provenance), "status": "missing"}
    cfg = OmegaConf.load(provenance)
    base = "stage_e.channels.tho.extreme_motion"
    fields = {
        "method": "soft_method",
        "scale_abs": "soft_scale_abs",
        "knee_abs": "knee_abs",
        "limit_abs": "limit_abs",
        "ramp_s": "ramp_s",
    }
    values: dict[str, Any] = {"provenance_path": str(provenance), "status": "resolved"}
    for output_name, config_name in fields.items():
        value = OmegaConf.select(cfg, f"{base}.{config_name}")
        if value is None:
            return {"provenance_path": str(provenance), "status": "missing", "missing": config_name}
        values[output_name] = value
    return values


def _write_preprocessing_gap(
    output_dir: Path,
    *,
    missing_fields: Sequence[str],
    checked: Sequence[Mapping[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "row_id": 8025,
        "missing_fields": sorted(set(missing_fields)),
        "checked_npz_and_keys": list(checked),
        "suggested_generation_entrypoint": (
            "在 resp_prepare 中从 scripts/build_stage_e.py 和 "
            "scripts/build_research_v2_alignment.py 重建对应整晚 NPZ，再运行 "
            "scripts/build_research_v2_dataset.py 导出索引"
        ),
    }
    (output_dir / "evidence_gap_preprocessing.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_preprocessing_window(
    catalog: EvidenceCatalog,
    output_dir: str | Path,
) -> PreprocessingWindow | None:
    """从 row 8025 的整晚源 NPZ 只读真实窗口；缺证据时写结构化缺口。"""
    output_path = Path(output_dir)
    row = _catalog_row(catalog)
    source = (catalog.dataset_index.parent / row["source_npz"]).resolve()
    target = (catalog.dataset_index.parent / row["target_source_npz"]).resolve()
    requests = (
        (source, row.get("bcg_rawish_observed_key", ""), "bcg_raw"),
        (source, row.get("bcg_rawish_segment_robust_z_key", ""), "bcg_robust"),
        (target, row.get("target_waveform_observed_key", ""), "tho_raw"),
        (target, row.get("target_waveform_segment_robust_z_key", ""), "tho_robust"),
        (target, row.get("target_waveform_segment_soft_z_key", ""), "tho_soft"),
    )
    missing: list[str] = []
    checked: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    opened: dict[Path, dict[str, np.ndarray]] = {}
    for npz_path, key, logical_name in requests:
        available: list[str] = []
        if npz_path.is_file():
            if npz_path not in opened:
                with np.load(npz_path, allow_pickle=False) as data:
                    opened[npz_path] = {name: np.asarray(data[name]) for name in data.files}
            available = sorted(opened[npz_path])
        checked.append({"npz": str(npz_path), "requested_key": key, "available_keys": available})
        if not key or key not in opened.get(npz_path, {}):
            missing.append(key or logical_name)
        else:
            arrays[logical_name] = opened[npz_path][key]
    softz = _softz_parameters(catalog.dataset_index.parent.parent)
    if softz.get("status") != "resolved":
        missing.append(f"soft-z parameters: {softz.get('missing', softz['provenance_path'])}")
    if missing:
        _write_preprocessing_gap(output_path, missing_fields=missing, checked=checked)
        return None

    sample_rate = 100.0
    start_sec = float(row["window_start_s"])
    end_sec = float(row["window_end_s"])
    start = int(round(start_sec * sample_rate))
    end = int(round(end_sec * sample_rate))
    for name, values in tuple(arrays.items()):
        arrays[name] = np.asarray(values[start:end], dtype=np.float32)
        if arrays[name].size != 18_000 or not np.isfinite(arrays[name]).all():
            missing.append(f"{name}: expected 18000 finite samples, got {arrays[name].size}")
    if missing:
        _write_preprocessing_gap(output_path, missing_fields=missing, checked=checked)
        return None

    center_median: float | None = None
    scale_median: float | None = None
    source_arrays = opened.get(source, {})
    sec_start, sec_end = int(round(start_sec)), int(round(end_sec))
    if "bcg_rawish_norm_center_sec" in source_arrays:
        center_median = float(np.nanmedian(source_arrays["bcg_rawish_norm_center_sec"][sec_start:sec_end]))
    if "bcg_rawish_norm_scale_sec" in source_arrays:
        scale_median = float(np.nanmedian(source_arrays["bcg_rawish_norm_scale_sec"][sec_start:sec_end]))
    return PreprocessingWindow(
        row_id=8025,
        start_sec=start_sec,
        end_sec=end_sec,
        sample_rate_hz=sample_rate,
        bcg_raw=arrays["bcg_raw"],
        bcg_robust=arrays["bcg_robust"],
        tho_raw=arrays["tho_raw"],
        tho_robust=arrays["tho_robust"],
        tho_soft=arrays["tho_soft"],
        bcg_center_median=center_median,
        bcg_scale_median=scale_median,
        softz_params=softz,
    )


def _plot_style() -> None:
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.is_file():
        font_manager.fontManager.addfont(font_path)
        chinese_font = font_manager.FontProperties(fname=font_path).get_name()
    else:
        chinese_font = "Droid Sans Fallback"
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [chinese_font, "DejaVu Sans"],
            "font.size": 13,
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.titlecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "legend.labelcolor": "black",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    footer = getattr(fig, "_signal_footer_artist", None)
    panel_axes = getattr(fig, "_signal_panel_axes", fig.axes)
    report = validate_figure_text_layout(fig, footer, panel_axes)
    if not report["all_key_text_inside"] or report["footer_overlaps_xlabels"]:
        raise ValueError(f"图中文字布局越界或重叠: {path.name}: {report}")
    fig.savefig(path, dpi=120, facecolor="white")
    plt.close(fig)


def _new_figure(*, rows: int = 1, columns: int = 1, **kwargs: Any):
    return plt.subplots(rows, columns, figsize=(18, 10), constrained_layout=True, **kwargs)


def create_figure_with_footer(
    *,
    rows: int,
    columns: int,
    footer_text: str,
    panel_gridspec_kw: Mapping[str, Any] | None = None,
    **subplot_kwargs: Any,
) -> tuple[plt.Figure, Any, Text]:
    """用独立 GridSpec 行承载底注，避免底注与 x-label 争用画布空间。"""
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    outer = fig.add_gridspec(2, 1, height_ratios=(1.0, 0.075), hspace=0.025)
    panels = outer[0].subgridspec(rows, columns, **dict(panel_gridspec_kw or {}))
    axes = panels.subplots(**subplot_kwargs)
    footer_axis = fig.add_subplot(outer[1])
    footer_axis.axis("off")
    footer = footer_axis.text(
        0.01,
        0.5,
        footer_text,
        transform=footer_axis.transAxes,
        ha="left",
        va="center",
        fontsize=11,
        color="black",
        wrap=True,
    )
    panel_axes = np.atleast_1d(axes).reshape(-1).tolist()
    fig._signal_footer_artist = footer  # type: ignore[attr-defined]
    fig._signal_panel_axes = panel_axes  # type: ignore[attr-defined]
    return fig, axes, footer


def validate_figure_text_layout(
    fig: plt.Figure,
    footer: Text | None,
    panel_axes: Sequence[Any],
) -> dict[str, Any]:
    """用渲染后的 bbox 检查标题/轴标/图例/底注是否越界或互相覆盖。"""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_box = fig.bbox
    key_text: list[Text] = []
    if fig._suptitle is not None:
        key_text.append(fig._suptitle)
    if footer is not None:
        key_text.append(footer)
    for axis in fig.axes:
        key_text.extend((axis.title, axis.xaxis.label, axis.yaxis.label))
        key_text.extend(axis.texts)
        legend = axis.get_legend()
        if legend is not None:
            key_text.extend(legend.get_texts())
    key_text = [artist for artist in key_text if artist.get_visible() and artist.get_text().strip()]
    outside: list[str] = []
    for artist in key_text:
        box = artist.get_window_extent(renderer)
        if (
            box.x0 < figure_box.x0 - 1
            or box.y0 < figure_box.y0 - 1
            or box.x1 > figure_box.x1 + 1
            or box.y1 > figure_box.y1 + 1
        ):
            outside.append(artist.get_text())
    overlaps = 0
    if footer is not None:
        footer_box = footer.get_window_extent(renderer)
        for axis in panel_axes:
            label = axis.xaxis.label
            if label.get_visible() and label.get_text().strip():
                overlaps += int(footer_box.overlaps(label.get_window_extent(renderer)))
    return {
        "all_key_text_inside": not outside,
        "outside_text": outside,
        "footer_overlaps_xlabels": overlaps,
    }


def _plot_signal_overview(signal: Mapping[str, np.ndarray], metadata: Mapping[str, Any], path: Path) -> None:
    fig, axes, _ = create_figure_with_footer(
        rows=3,
        columns=1,
        sharex=True,
        footer_text="dataset_row_id=8025；黄色区域仅用于放大观察，不改变模型的 180 s 输入。",
    )
    time = signal["time_sec_full"]
    series = (
        ("BCG 模型输入（soft-z）", signal["bcg_input_full"], _COLORS[0]),
        ("THO 参考", signal["target_respiration_full"], _COLORS[1]),
        ("F0 预测", signal["f0_prediction_full"], _COLORS[2]),
    )
    crop_start = float(metadata["crop_start_sec"])
    crop_end = crop_start + float(metadata["crop_duration_sec"])
    for axis, (label, values, color) in zip(axes, series):
        axis.plot(time, values, color=color, linewidth=1.0, label=label)
        axis.axvspan(crop_start, crop_end, color="#F0E442", alpha=0.22, label="建议细看片段 30–90 s")
        axis.set_ylabel("幅值（z 空间）")
        axis.legend(loc="upper right", frameon=False)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("整窗内时间（s）")
    fig.suptitle("同一真实 180 s 验证窗：从 BCG 输入到 THO 参考与 F0 输出", fontsize=22)
    _save(fig, path)


def _gap_panel(path: Path, title: str) -> None:
    fig, axis = _new_figure()
    axis.axis("off")
    axis.text(0.5, 0.62, title, ha="center", va="center", fontsize=24, color="black", weight="bold")
    axis.text(
        0.5,
        0.42,
        "证据缺口：所需真实 NPZ 字段或可追溯参数不完整\n未绘制替代曲线；详见 evidence_gap_preprocessing.json",
        ha="center",
        va="center",
        fontsize=18,
        color="black",
    )
    _save(fig, path)


def _plot_preprocessing(window: PreprocessingWindow, path: Path) -> None:
    params = window.softz_params
    bcg_stats = (
        f"BCG 本窗 center 中位数={window.bcg_center_median:.3g}, scale 中位数={window.bcg_scale_median:.3g}"
        if window.bcg_center_median is not None and window.bcg_scale_median is not None
        else "BCG center/scale 数组未提供"
    )
    footer = (
        f"{bcg_stats}；THO soft-z：{params['method']}, knee={params['knee_abs']}, "
        f"scale={params['scale_abs']}, ramp={params['ramp_s']} s（参数来自 dataset provenance）。"
    )
    fig, axes, _ = create_figure_with_footer(
        rows=2,
        columns=2,
        sharex=True,
        footer_text=footer,
    )
    time = np.arange(window.bcg_raw.size) / window.sample_rate_hz
    panels = (
        (axes[0, 0], window.bcg_raw, "BCG 原始对齐 rawish", "原幅值（a.u.）", _COLORS[0]),
        (axes[0, 1], window.bcg_robust, "BCG 分段 robust-z", "robust-z", _COLORS[2]),
        (axes[1, 0], window.tho_raw, "THO 原始观测", "原幅值（a.u.）", _COLORS[1]),
        (axes[1, 1], window.tho_soft, "THO 分段 robust-z 后 soft-z", "soft-z", _COLORS[3]),
    )
    for axis, values, title, ylabel, color in panels:
        axis.plot(time, values, color=color, linewidth=0.8)
        axis.set_title(title, fontsize=16)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    axes[1, 0].set_xlabel("窗口内时间（s）")
    axes[1, 1].set_xlabel("窗口内时间（s）")
    fig.suptitle("真实 row 8025 预处理前后对照（切片 90–270 s）", fontsize=22)
    _save(fig, path)


def _soft_compression(values: np.ndarray, *, knee: float, scale: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    magnitude = np.abs(values)
    output = magnitude.copy()
    over = magnitude > knee
    output[over] = knee + scale * np.log1p((magnitude[over] - knee) / scale)
    return (np.sign(values) * output).astype(np.float32)


def softz_explanation(
    *,
    knee: float,
    scale: float,
    ramp_s: float,
    max_abs_before: float,
    changed_samples: int,
) -> tuple[str, str]:
    """生成不超出单点算子证据边界的 soft-z 标题和说明。"""
    title = "soft-z：full-weight 单点压缩算子与本窗真实配对结果"
    trigger = (
        f"本窗 max|robust-z|={max_abs_before:.6f}<knee={knee:g}，未触发压缩 / "
        f"{changed_samples} samples changed。"
    )
    note = (
        f"左图仅示意 w=1 时的 full-weight 单点算子：|z|>{knee:g} 后按 log1p_tail(scale={scale:g}) 压缩。"
        f"真实算法为 normalized = raw + w(t)·(compressed-raw)，w(t) 由 extreme-motion mask 与 {ramp_s:g} s ramp 决定；"
        f"不能由单点映射推出全局排序。{trigger}"
    )
    return title, note


def _plot_softz(window: PreprocessingWindow, path: Path) -> None:
    params = window.softz_params
    knee, scale = float(params["knee_abs"]), float(params["scale_abs"])
    delta = window.tho_soft.astype(np.float64) - window.tho_robust.astype(np.float64)
    changed_samples = int(np.count_nonzero(delta != 0.0))
    max_abs_before = float(np.max(np.abs(window.tho_robust)))
    title, note = softz_explanation(
        knee=knee,
        scale=scale,
        ramp_s=float(params["ramp_s"]),
        max_abs_before=max_abs_before,
        changed_samples=changed_samples,
    )
    max_abs = max(knee + 2, float(np.nanpercentile(np.abs(window.tho_robust), 99.9)))
    x = np.linspace(-max_abs, max_abs, 1200)
    y = _soft_compression(x, knee=knee, scale=scale)
    fig, axes, _ = create_figure_with_footer(
        rows=1,
        columns=2,
        footer_text=note,
    )
    axes[0].plot(x, x, linestyle="--", color="black", linewidth=1.2, label="不压缩 y=x")
    axes[0].plot(x, y, color=_COLORS[3], linewidth=2.5, label="log1p_tail 映射")
    axes[0].axvline(-knee, color=_COLORS[1], linestyle=":")
    axes[0].axvline(knee, color=_COLORS[1], linestyle=":", label=f"knee=±{knee:g}")
    axes[0].set(xlabel="输入 robust-z", ylabel="full-weight 压缩后 z", title="单点算子示意（w=1）")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False)
    time = np.arange(delta.size, dtype=np.float64) / window.sample_rate_hz
    axes[1].plot(time, delta, color=_COLORS[3], linewidth=1.2)
    if changed_samples == 0:
        axes[1].set_ylim(-0.05, 0.05)
    axes[1].text(
        0.5,
        0.88,
        f"未触发压缩 / {changed_samples} samples changed\nmax|robust-z|={max_abs_before:.6f} < knee={knee:g}",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        color="black",
        fontsize=15,
    )
    axes[1].set(
        xlabel="窗口内时间（s）",
        ylabel="soft-z − robust-z",
        title="row 8025 真实逐样本配对差值",
    )
    axes[1].grid(alpha=0.2)
    fig.suptitle(title, fontsize=22)
    _save(fig, path)


def _plot_stft_comparison(f0: StftLogMagnitude, wide: StftLogMagnitude, path: Path) -> None:
    fig, axes, _ = create_figure_with_footer(
        rows=1,
        columns=2,
        sharey=True,
        footer_text="两图均为 torch.stft + Hann window + center=True；没有使用 specgram 默认参数。",
    )
    for axis, result, name in ((axes[0], f0, "F0 正式 STFT"), (axes[1], wide, "G3-C wide STFT")):
        mesh = axis.pcolormesh(result.times_sec, result.frequencies_hz, result.log_magnitude, shading="auto", cmap="magma")
        axis.set_title(
            f"{name}\nwin={result.win_samples} ({result.win_samples/100:.1f}s), "
            f"hop={result.hop_samples} ({result.hop_samples/100:.1f}s)\n"
            f"Δf={result.frequency_resolution_hz:.3f} Hz, bins={result.log_magnitude.shape[0]}, frames={result.log_magnitude.shape[1]}",
            fontsize=16,
        )
        axis.set_xlabel("窗口内时间（s）")
        axis.set_ylabel("频率（Hz）")
        colorbar = fig.colorbar(mesh, ax=axis, pad=0.01)
        colorbar.set_label("log1p(|STFT|)", color="black")
    fig.suptitle("同一真实 BCG 输入：频率分辨率与时间帧密度的取舍", fontsize=22)
    _save(fig, path)


def _plot_bandenergy(wide: StftLogMagnitude, path: Path) -> None:
    energies = bandenergy_from_logmag(wide.log_magnitude, wide.frequencies_hz)
    fig, (heat_axis, track_axis), _ = create_figure_with_footer(
        rows=2,
        columns=1,
        panel_gridspec_kw={"height_ratios": [1.3, 1]},
        footer_text="频带边界按 stft_branch.py 纳入端点；重叠意味着边界附近信息可进入相邻两带。",
    )
    mesh = heat_axis.pcolormesh(wide.times_sec, wide.frequencies_hz, wide.log_magnitude, shading="auto", cmap="magma")
    for index, (low, high) in enumerate(BANDENERGY_BANDS_HZ):
        heat_axis.axhspan(low, high, facecolor=_COLORS[index], edgecolor=_COLORS[index], alpha=0.10, linewidth=1.2)
        heat_axis.text(181, (low + high) / 2, f"B{index+1}", color="black", va="center", fontsize=11)
    heat_axis.set(xlabel="窗口内时间（s）", ylabel="频率（Hz）", title="wide log-magnitude 与 5 个重叠频带")
    fig.colorbar(mesh, ax=heat_axis, pad=0.01, label="log1p(|STFT|)")
    for index, ((low, high), values) in enumerate(zip(BANDENERGY_BANDS_HZ, energies)):
        track_axis.plot(wide.times_sec, values, color=_COLORS[index], linewidth=1.7, label=f"B{index+1}: {low:g}–{high:g} Hz")
    track_axis.set(xlabel="窗口内时间（s）", ylabel="带内 log-magnitude 均值", title="每个频带压缩为一条随时间变化的轨迹")
    track_axis.grid(alpha=0.2)
    track_axis.legend(ncol=3, frameon=False, loc="upper right")
    fig.suptitle("bandenergy：把 160 个频率 bin 压缩成 5 条重叠频带轨迹", fontsize=22)
    _save(fig, path)


def build_signal_assets(
    repo_root: str | Path,
    output_dir: str | Path,
) -> tuple[dict[str, Path], dict[str, Any]]:
    """生成五张信号讲解资产，并返回稳定 key 与可审计 metadata。"""
    root = Path(repo_root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    catalog = build_evidence_catalog(root)
    metadata_path = catalog.general_signal_npz.with_name("f0_visual_sample_metadata.json")
    signal_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(catalog.general_signal_npz, allow_pickle=False) as data:
        signal = {name: np.asarray(data[name]) for name in data.files}
    required = {"time_sec_full", "bcg_input_full", "target_respiration_full", "f0_prediction_full"}
    missing = sorted(required - signal.keys())
    if missing:
        raise ValueError(f"通用信号 NPZ 缺少数组 {missing}: {catalog.general_signal_npz}")
    signal_length = int(signal["bcg_input_full"].size)
    if signal_length != 18_000:
        raise ValueError(f"通用信号长度应为 18000，实际 {signal_length}")

    f0_config = catalog.run_configs["g0_f0_native_stft_pre_mixer"]
    wide_config = catalog.run_configs["g3_c_wide_8p0"]
    f0 = compute_stft_logmag(
        signal["bcg_input_full"], sample_rate_hz=100.0,
        win_samples=f0_config.stft_win, hop_samples=f0_config.stft_hop,
        low_hz=f0_config.stft_low_hz, high_hz=f0_config.stft_high_hz, center=True,
    )
    wide = compute_stft_logmag(
        signal["bcg_input_full"], sample_rate_hz=100.0,
        win_samples=wide_config.stft_win, hop_samples=wide_config.stft_hop,
        low_hz=wide_config.stft_low_hz, high_hz=wide_config.stft_high_hz, center=True,
    )
    preprocessing = load_preprocessing_window(catalog, output)
    assets = {key: output / f"{key}.png" for key in _ASSET_KEYS}
    _plot_style()
    _plot_signal_overview(signal, signal_metadata, assets["signal_overview"])
    if preprocessing is None:
        _gap_panel(assets["preprocessing_comparison"], "预处理前后对照")
        _gap_panel(assets["softz_mapping"], "soft-z 映射与真实分布")
    else:
        _plot_preprocessing(preprocessing, assets["preprocessing_comparison"])
        _plot_softz(preprocessing, assets["softz_mapping"])
    _plot_stft_comparison(f0, wide, assets["stft_resolution_comparison"])
    _plot_bandenergy(wide, assets["bandenergy_response"])
    if preprocessing is None:
        softz_changed_samples: int | None = None
        softz_max_abs_before: float | None = None
        softz_max_abs_after: float | None = None
    else:
        softz_changed_samples = int(np.count_nonzero(preprocessing.tho_soft != preprocessing.tho_robust))
        softz_max_abs_before = float(np.max(np.abs(preprocessing.tho_robust)))
        softz_max_abs_after = float(np.max(np.abs(preprocessing.tho_soft)))
    metadata = {
        "dataset_row_id": int(catalog.general_sample_row_id),
        "signal_length": signal_length,
        "f0_frames": int(f0.log_magnitude.shape[1]),
        "f0_frequency_bins": int(f0.log_magnitude.shape[0]),
        "wide_frames": int(wide.log_magnitude.shape[1]),
        "wide_frequency_bins": int(wide.log_magnitude.shape[0]),
        "wide_frequency_resolution_hz": float(wide.frequency_resolution_hz),
        "stft_center": True,
        "preprocessing_evidence_gap": preprocessing is None,
        "softz_changed_samples": softz_changed_samples,
        "softz_max_abs_before": softz_max_abs_before,
        "softz_max_abs_after": softz_max_abs_after,
    }
    return assets, metadata


if __name__ == "__main__":
    repository = Path(__file__).resolve().parents[2]
    destination = repository / "docs/stage_reports/20260708/generated_assets/discussion"
    generated, details = build_signal_assets(repository, destination)
    print(json.dumps({"assets": {key: str(path) for key, path in generated.items()}, "metadata": details}, ensure_ascii=False, indent=2))
