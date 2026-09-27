"""W0 定性导出的纯数值与离线绘图工具；不打开原始数据或 checkpoint。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from resp_train.metrics.task import (
    TaskMetricConfig, _periodogram, _rr_from_power, compute_log_rms_envelopes,
)
from resp_train.protocols.respiration import canonicalize_numpy, centered_energy_numpy

SEED = 20260812
EPOCH = 15
PROTOCOL = "w0-test-qualitative-export-v1-20260927"
PRIMARY = (
    "whole_rr_abs_error_bpm", "local_rr_mae_bpm", "envelope_trajectory_mae",
    "global_envelope_modulation_error", "lag_aware_signed_pcc",
)
METHODS = ("F0", "IEWT", "W0")
# 沿用既有波形回放级别的严格误差检查，失败后保留产物用于诊断。
ANCHOR_ATOL = 1e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def save_arrays(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """独占创建；退化 RR 的缺失只允许出现在带有效性标记的轨迹中。"""
    for key, value in arrays.items():
        value = np.asarray(value)
        if value.dtype.kind == "O":
            raise TypeError(f"禁止 pickle/object 数组: {key}")
        if value.dtype.kind in "fc":
            if np.isinf(value).any() or (np.isnan(value).any() and key != "local_rr_bpm"):
                raise FloatingPointError(f"非有限导出数组: {key}")
    if "local_rr_bpm" in arrays:
        if not np.array_equal(np.isfinite(arrays["local_rr_bpm"]), arrays["local_rr_valid"]):
            raise ValueError("RR 缺失与有效性标记不一致")
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def waveform_details(waves: np.ndarray, cfg: Any) -> dict[str, np.ndarray]:
    """按冻结指标的 canonicalization 与频谱估计导出轨迹，顺序为 reference/F0/IEWT/W0。"""
    p = TaskMetricConfig.from_config(cfg)
    waves = np.asarray(waves)
    if waves.ndim != 2 or waves.shape[1] != p.length or not np.isfinite(waves).all():
        raise ValueError("waveform shape/finite 不合格")
    band, canonical = canonicalize_numpy(
        waves, fs=p.fs, low_hz=p.band_low_hz, high_hz=p.band_high_hz, scale_eps=p.scale_eps,
    )
    starts = np.arange(0, p.length - p.local_rr_window + 1, p.local_rr_step)
    rr = np.full((len(waves), len(starts)), np.nan)
    eligible = np.empty(len(starts), dtype=bool)
    for j, start in enumerate(starts):
        stop = start + p.local_rr_window
        eligible[j] = centered_energy_numpy(band[0, start:stop]) > p.dynamic_eps
        for i, wave in enumerate(canonical):
            # 与 _local_rr 相同：target eligibility 使用 band，预测退化检查使用 canonical。
            valid = eligible[j] if i == 0 else centered_energy_numpy(wave[start:stop]) > p.dynamic_eps
            if valid:
                rr[i, j] = _rr_from_power(_periodogram(wave[start:stop], p), p.local_rr_window, p)
                if not np.isfinite(rr[i, j]):
                    raise FloatingPointError("有效窗口的 RR 估计非有限")
    env = compute_log_rms_envelopes(waves, cfg)
    return {
        "canonical_waveforms": canonical,
        "local_rr_bpm": rr,
        "local_rr_valid": np.isfinite(rr),
        "local_rr_target_eligible": eligible,
        "local_rr_time_s": (starts + p.local_rr_window / 2) / p.fs,
        "log_rms_envelopes": env,
        "centered_log_rms_envelopes": env - np.median(env, axis=1, keepdims=True),
        "envelope_time_s": (np.arange(env.shape[1]) * p.envelope_step + p.envelope_window / 2) / p.fs,
    }


def anchor_deltas(observed: pd.DataFrame, frozen: pd.DataFrame) -> pd.DataFrame:
    """逐窗口五主指标配对；不允许缺行、重复行或以 NaN 绕过比较。"""
    for frame in (observed, frozen):
        if frame["dataset_row_id"].duplicated().any():
            raise ValueError("anchor row 重复")
        if not np.isfinite(frame[list(PRIMARY)].to_numpy(float)).all():
            raise FloatingPointError("anchor 主指标非有限")
    if set(observed.dataset_row_id) != set(frozen.dataset_row_id):
        raise ValueError("anchor row 集合不一致")
    a = observed.set_index("dataset_row_id").sort_index()
    b = frozen.set_index("dataset_row_id").loc[a.index]
    for column in ("whole_rr_target_eligible", "local_rr_target_eligible", "local_rr_target_eligible_windows",
                   "joint_target_eligible", "envelope_spearman_target_eligible", "envelope_target_stratum"):
        if not np.array_equal(a[column], b[column]):
            raise ValueError(f"anchor 资格不一致: {column}")
    records = []
    for metric in PRIMARY:
        for row_id, actual, expected in zip(a.index, a[metric], b[metric], strict=True):
            delta = abs(float(actual) - float(expected))
            records.append({"dataset_row_id": int(row_id), "metric": metric,
                            "observed": float(actual), "expected": float(expected),
                            "abs_delta": delta, "atol": ANCHOR_ATOL, "within_atol": delta <= ANCHOR_ATOL})
    return pd.DataFrame(records)


def metric_caption(row: Mapping[str, Any]) -> str:
    return (f"lag-aware PCC={row['lag_aware_signed_pcc']:.3f}  |  "
            f"RR error={row['whole_rr_abs_error_bpm']:.2f} bpm  |  "
            f"Local RR MAE={row['local_rr_mae_bpm']:.2f} bpm\n"
            f"Envelope trajectory MAE={row['envelope_trajectory_mae']:.3f}  |  "
            f"Global modulation error={row['global_envelope_modulation_error']:.3f}  |  "
            f"metric lag={row['best_lag_sec']:+.2f} s")


def render_window(
    arrays: Mapping[str, np.ndarray], metrics: pd.DataFrame, destination: Path,
    *, row_id: int, subject: int, start_s: float,
    zoom: tuple[float, float] | None = None,
) -> list[Path]:
    """四联图与两张诊断附图；只改变显示范围，保持波形时序与原始指标。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if set(metrics.method) != set(METHODS) or len(metrics) != 3:
        raise ValueError("绘图必须包含恰好三个方法")
    if set(metrics.dataset_row_id.astype(int)) != {row_id} or int(arrays["dataset_row_id"]) != row_id:
        raise ValueError("绘图波形/指标 row 不匹配")
    fs = float(arrays["fs_hz"])
    waves = arrays["canonical_waveforms"]
    if waves.shape != (4, len(arrays["bcg"])):
        raise ValueError("四联图波形 shape 不一致")
    duration = waves.shape[1] / fs
    if zoom is not None and not (0 <= zoom[0] < zoom[1] <= duration):
        raise ValueError("zoom 必须在完整窗口内")
    bounds = zoom or (0.0, duration)
    suffix = "full" if zoom is None else f"zoom_{zoom[0]:g}_{zoom[1]:g}"
    paths = [destination / f"row_{row_id}_{kind}_{suffix}.png"
             for kind in ("waveforms", "conditioning", "trajectories")]
    if any(path.exists() for path in paths):
        raise FileExistsError("绘图目标已存在，拒绝覆盖")
    destination.mkdir(parents=True, exist_ok=True)
    title = f"W0 seed {SEED} | row {row_id} | subject {subject} | start {start_s:g} s"
    colors = ("#777777", "#D55E00", "#009E73", "#0072B2")
    written: list[Path] = []

    def save(fig: Any, kind: str) -> None:
        try:
            path = destination / f"row_{row_id}_{kind}_{suffix}.png"
            # exclusive-open 防止两个离线绘图进程互相覆盖。
            with path.open("xb") as handle:
                fig.savefig(handle, format="png", dpi=160, bbox_inches="tight")
            written.append(path)
        finally:
            plt.close(fig)

    time = np.arange(waves.shape[1]) / fs
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True, layout="constrained")
    axes[0].plot(time, arrays["bcg"], color=colors[0], lw=0.55)
    axes[0].set_title("Wideband BCG (state-aligned, segment soft-z)", loc="left")
    axes[0].set_ylabel("Input amplitude")
    by_method = metrics.set_index("method")
    for i, method in enumerate(METHODS, 1):
        axes[i].plot(time, waves[0], color="black", lw=1, label="THO reference")
        axes[i].plot(time, waves[i], color=colors[i], lw=0.9, alpha=0.9,
                     label="F0 fixed respiratory band" if method == "F0" else method)
        axes[i].set_title(metric_caption(by_method.loc[method]), loc="left", fontsize=9)
        axes[i].set_ylabel("Canonical amplitude")
        axes[i].legend(loc="upper right", fontsize=8)
    for ax in axes:
        ax.set_xlim(*bounds)
        ax.grid(alpha=0.15)
    axes[-1].set_xlabel("Time within window (s)")
    fig.suptitle(title + f"\nMetrics: full {duration:g} s; overlays: frozen 0.05–0.7 Hz canonicalization")
    save(fig, "waveforms")

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True, layout="constrained")
    # 实际频率可能重复；按尺度索引绘制，使用真实频率标注，避免伪造严格递增网格。
    frequencies = arrays["cwt_frequency_hz"]
    im = axes[0].imshow(arrays["cwt_w"], origin="lower", aspect="auto", extent=(0, duration, -.5, len(frequencies)-.5))
    ticks = np.unique(np.linspace(0, len(frequencies)-1, 9).astype(int))
    axes[0].set_yticks(ticks, [f"{frequencies[k]:.3g}" for k in ticks])
    axes[0].set_ylabel("Mapped frequency (Hz)")
    axes[0].set_title("CWT network input: mean50(log1p(abs(CWT))); rows = ordered scales", fontsize=9)
    fig.colorbar(im, ax=axes[0], label="Feature value")
    for ax, key in zip(axes[1:3], ("g", "b"), strict=True):
        im = ax.imshow(arrays[key], origin="lower", aspect="auto", cmap="RdBu_r", vmin=-.5, vmax=.5,
                       extent=(0, duration, -.5, arrays[key].shape[0]-.5))
        ax.set_ylabel("Latent channel")
        ax.set_title(f"Effective FiLM {key}", fontsize=9)
        fig.colorbar(im, ax=ax)
    for key, label in (("r_scale_time", "scale"), ("r_shift_time", "shift"), ("r_total_time", "total")):
        axes[3].plot(arrays["latent_time_s"], arrays[key], label=label, lw=.8)
    axes[3].set_ylabel("Relative L2 strength")
    axes[3].legend()
    axes[3].set_xlabel("Time within window (s)")
    for ax in axes:
        ax.set_xlim(*bounds)
    fig.suptitle(title + "\nConditioning activity (descriptive)")
    save(fig, "conditioning")

    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True, layout="constrained")
    for i, label in enumerate(("THO reference", *METHODS)):
        color = "black" if i == 0 else colors[i]
        axes[0].plot(arrays["local_rr_time_s"], arrays["local_rr_bpm"][i], color=color, label=label, marker=".")
        axes[1].plot(arrays["envelope_time_s"], arrays["centered_log_rms_envelopes"][i], color=color, label=label)
    axes[0].set_ylabel("Local RR (bpm)")
    axes[0].set_title("60 s windows / 15 s step; missing estimates retain validity flags", fontsize=9)
    axes[1].set_ylabel("Centered log-RMS")
    axes[1].set_title("Canonical envelope, median-centered; 10 s windows / 5 s step", fontsize=9)
    axes[1].set_xlabel("Window center time (s)")
    for ax in axes:
        ax.set_xlim(*bounds)
        ax.legend(ncol=4)
        ax.grid(alpha=.2)
    fig.suptitle(title)
    save(fig, "trajectories")
    return written
