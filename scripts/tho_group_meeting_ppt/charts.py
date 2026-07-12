from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-tho-ppt")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = (
    "g0_time_only",
    "g0_f0_native_stft_pre_mixer",
    "g3_c_wide_8p0",
    "g3_c_bandenergy",
)
MODEL_NAMES = {
    "g0_time_only": "纯时序",
    "g0_f0_native_stft_pre_mixer": "上一版 STFT",
    "g3_c_wide_8p0": "宽频 STFT",
    "g3_c_bandenergy": "频带能量",
}
MODEL_COLORS = {
    "g0_time_only": "#6B7280",
    "g0_f0_native_stft_pre_mixer": "#67A9CF",
    "g3_c_wide_8p0": "#003F73",
    "g3_c_bandenergy": "#D97824",
}


@dataclass(frozen=True)
class ChartData:
    overall: pd.DataFrame
    tail: pd.DataFrame
    strata: pd.DataFrame
    rr_bins: pd.DataFrame
    harmonic: pd.DataFrame
    harmonic_coverage: dict[str, float]


def _read_csv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"缺少当前口径结果文件：{path}")
    frame = pd.read_csv(path)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"结果文件缺少列 {sorted(missing)}：{path}")
    return frame


def load_chart_data(repo_root: Path) -> ChartData:
    eval_root = repo_root / "runs/test_eval_g_series_20260709_local_rr_canonical"
    strata_root = eval_root / "stratified_analysis_20260709"
    harmonic_root = repo_root / "runs/bcg_second_harmonic_20260710"

    overall_raw = _read_csv(
        eval_root / "g_series_local_rr_canonical_label_summary.csv",
        {
            "label",
            "rr_peak_band_robust_abs_error_mean_mean",
            "breath_count_zero_cross_abs_error_mean_mean",
            "best_lag_corr_4s_mean_mean",
            "local_rr_mae_mean_mean",
            "local_rr_corr_mean_mean",
        },
    ).set_index("label")
    overall = pd.DataFrame(index=overall_raw.index)
    overall["robust_rr"] = overall_raw["rr_peak_band_robust_abs_error_mean_mean"]
    overall["count_bpm"] = overall_raw["breath_count_zero_cross_abs_error_mean_mean"] / 3.0
    overall["lag4_corr"] = overall_raw["best_lag_corr_4s_mean_mean"]
    overall["local_rr_mae"] = overall_raw["local_rr_mae_mean_mean"]
    overall["local_rr_corr"] = overall_raw["local_rr_corr_mean_mean"]
    overall = overall.loc[list(MODEL_ORDER)]

    tail = _read_csv(
        strata_root / "tail_summary.csv",
        {"label", "metric", "stat", "value_mean"},
    )
    strata = _read_csv(
        strata_root / "strata_summary.csv",
        {"comparison", "target", "stratum", "n_windows"},
    )
    rr_bins = _read_csv(
        strata_root / "rr_bin_summary.csv",
        {"comparison", "target", "rr_bin", "n_windows"},
    )

    coverage = _read_csv(
        harmonic_root / "test_v2/coverage_summary.csv",
        {"status", "n_windows", "fraction_of_all", "n_subjects"},
    ).set_index("status")
    coverage_values = {
        "all_windows": int(coverage.loc["all_windows", "n_windows"]),
        "eligible_windows": int(coverage.loc["eligible_total", "n_windows"]),
        "strong_windows": int(coverage.loc["strong_harmonic", "n_windows"]),
        "prominent_windows": int(coverage.loc["harmonic_prominent", "n_windows"]),
        "positive_union_windows": int(coverage.loc["harmonic_positive_union", "n_windows"]),
        "positive_union_fraction": float(coverage.loc["harmonic_positive_union", "fraction_of_all"]),
        "positive_union_subjects": int(coverage.loc["harmonic_positive_union", "n_subjects"]),
    }

    harmonic_long = _read_csv(
        harmonic_root / "model_metrics/model_stratified_metrics_summary.csv",
        {"label", "stratum", "metric_stat", "value_mean"},
    )
    wanted_stats = {
        "rr_peak_band_robust_abs_error_mean": "robust_rr",
        "breath_count_zero_cross_bpm_error_mean": "count_bpm",
        "best_lag_corr_4s_mean": "lag4_corr",
        "relative_envelope_corr_lag4s_mean": "relative_envelope_corr",
        "local_rr_mae_mean": "local_rr_mae",
    }
    harmonic = (
        harmonic_long[
            harmonic_long["stratum"].eq("harmonic_positive_union")
            & harmonic_long["metric_stat"].isin(wanted_stats)
        ]
        .pivot(index="label", columns="metric_stat", values="value_mean")
        .rename(columns=wanted_stats)
    )
    correction = _read_csv(
        harmonic_root / "corrections/model_harmonic_correction_summary.csv",
        {"label", "input_stratum", "corrected_fraction"},
    )
    correction = correction[correction["input_stratum"].eq("harmonic_positive_union")].set_index("label")
    harmonic["correction_rate"] = correction["corrected_fraction"]
    harmonic = harmonic.loc[list(MODEL_ORDER)]

    return ChartData(
        overall=overall,
        tail=tail,
        strata=strata,
        rr_bins=rr_bins,
        harmonic=harmonic,
        harmonic_coverage=coverage_values,
    )


def _setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.edgecolor": "#333333",
            "text.color": "#000000",
            "axes.labelcolor": "#000000",
            "xtick.color": "#000000",
            "ytick.color": "#000000",
        }
    )


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_overall_metrics(overall: pd.DataFrame, path: Path) -> Path:
    _setup_matplotlib()
    metrics = ["robust_rr", "count_bpm", "local_rr_mae"]
    labels = ["稳健 RR 误差", "周期计数 bpm 误差", "局部 RR MAE"]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.2))
    for axis, metric, label in zip(axes, metrics, labels, strict=True):
        values = overall.loc[list(MODEL_ORDER), metric]
        bars = axis.bar(
            np.arange(len(values)),
            values,
            color=[MODEL_COLORS[model] for model in MODEL_ORDER],
            width=0.70,
        )
        axis.set_title(label, fontsize=13)
        axis.set_xticks(np.arange(len(values)), [MODEL_NAMES[model] for model in MODEL_ORDER], rotation=20, ha="right", fontsize=9)
        axis.grid(axis="y", alpha=0.20)
        axis.spines[["top", "right"]].set_visible(False)
        axis.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    fig.suptitle("独立测试集整体结果（3 次独立训练均值）", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def _tail_value(tail: pd.DataFrame, label: str, metric: str, stat: str) -> float:
    rows = tail[(tail["label"] == label) & (tail["metric"] == metric) & (tail["stat"] == stat)]
    if len(rows) != 1:
        raise ValueError(f"长尾统计不是唯一值：label={label}, metric={metric}, stat={stat}")
    return float(rows.iloc[0]["value_mean"])


def plot_tail_metrics(tail: pd.DataFrame, path: Path) -> Path:
    _setup_matplotlib()
    series = {
        "稳健 RR p95": ("rr_peak_band_robust_abs_error", "p95", 1.0),
        "稳健 RR > 2 bpm": ("rr_peak_band_robust_abs_error", "frac_gt_2", 100.0),
        "周期计数 p95（bpm）": ("breath_count_zero_cross_abs_error", "p95", 1.0 / 3.0),
        "局部 RR MAE p95": ("local_rr_mae", "p95", 1.0),
    }
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.0))
    for axis, (title, (metric, stat, scale)) in zip(axes.flat, series.items(), strict=True):
        values = [_tail_value(tail, model, metric, stat) * scale for model in MODEL_ORDER]
        bars = axis.bar(range(4), values, color=[MODEL_COLORS[model] for model in MODEL_ORDER])
        axis.set_title(title, fontsize=12)
        axis.set_xticks(range(4), [MODEL_NAMES[model] for model in MODEL_ORDER], rotation=18, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.20)
        axis.spines[["top", "right"]].set_visible(False)
        axis.bar_label(bars, fmt="%.2f", fontsize=8)
    fig.suptitle("长尾与灾难性误差", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def _comparison_rows(frame: pd.DataFrame, stratum_column: str, strata: list[str]) -> pd.DataFrame:
    rows = frame[frame[stratum_column].isin(strata) & frame["comparison"].isin(["wide_vs_time", "bandenergy_vs_time"])]
    if rows.empty:
        raise ValueError(f"分层结果缺少比较：{strata}")
    return rows


def plot_stratified_roles(strata: pd.DataFrame, rr_bins: pd.DataFrame, path: Path) -> Path:
    _setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.8))
    selected = _comparison_rows(strata, "stratum", ["baseline_hard_peak_gt1", "low_spectrum_le_median"])
    pivot = selected.pivot(index="stratum", columns="target", values="target_rr_peak_band_robust_abs_error_mean")
    pivot.plot(kind="bar", ax=axes[0], color=[MODEL_COLORS.get(column, "#666666") for column in pivot.columns])
    axes[0].set_title("困难窗口：稳健 RR 误差", fontsize=12)
    axes[0].set_xlabel("")
    axes[0].set_xticklabels(["baseline hard", "low spectrum"], rotation=0)
    axes[0].legend([MODEL_NAMES.get(column, column) for column in pivot.columns], fontsize=8)

    rr = _comparison_rows(rr_bins, "rr_bin", ["rr_lt10", "rr_10_14", "rr_14_18", "rr_18_24"])
    rr_pivot = rr.pivot(index="rr_bin", columns="target", values="target_rr_peak_band_robust_abs_error_mean")
    rr_pivot = rr_pivot.reindex(["rr_lt10", "rr_10_14", "rr_14_18", "rr_18_24"])
    rr_pivot.plot(kind="bar", ax=axes[1], color=[MODEL_COLORS.get(column, "#666666") for column in rr_pivot.columns])
    axes[1].set_title("RR 分层：稳健 RR 误差", fontsize=12)
    axes[1].set_xlabel("")
    axes[1].set_xticklabels(["<10", "10–14", "14–18", "18–24"], rotation=0)
    axes[1].legend([MODEL_NAMES.get(column, column) for column in rr_pivot.columns], fontsize=8)
    for axis in axes:
        axis.set_ylabel("bpm")
        axis.grid(axis="y", alpha=0.20)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("wide 与 bandenergy 的收益集中在不同窗口", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def plot_harmonic_coverage(coverage: dict[str, float], path: Path) -> Path:
    _setup_matplotlib()
    values = [coverage["strong_windows"], coverage["prominent_windows"], coverage["positive_union_windows"]]
    labels = ["strong", "harmonic prominent", "阳性并集"]
    fig, axis = plt.subplots(figsize=(7.0, 3.5))
    bars = axis.barh(labels, values, color=["#67A9CF", "#D97824", "#003F73"])
    axis.invert_yaxis()
    axis.set_xlabel("窗口数")
    axis.set_title(
        f"阳性并集：{coverage['positive_union_windows']} / {coverage['all_windows']}（{coverage['positive_union_fraction']:.2%}）",
        fontsize=14,
        fontweight="bold",
    )
    axis.bar_label(bars, fontsize=10, padding=4)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.grid(axis="x", alpha=0.20)
    fig.tight_layout()
    return _save(fig, path)


def plot_harmonic_results(harmonic: pd.DataFrame, path: Path) -> Path:
    _setup_matplotlib()
    metrics = ["robust_rr", "count_bpm", "lag4_corr", "local_rr_mae", "correction_rate"]
    labels = ["稳健 RR 误差", "计数 bpm 误差", "4 秒时延相关", "局部 RR MAE", "谐波纠正率"]
    fig, axes = plt.subplots(1, 5, figsize=(14.5, 3.2))
    for axis, metric, label in zip(axes, metrics, labels, strict=True):
        values = harmonic.loc[list(MODEL_ORDER), metric]
        bars = axis.bar(range(4), values, color=[MODEL_COLORS[model] for model in MODEL_ORDER])
        axis.set_title(label, fontsize=10)
        axis.set_xticks(range(4), [MODEL_NAMES[model] for model in MODEL_ORDER], rotation=25, ha="right", fontsize=7)
        axis.grid(axis="y", alpha=0.20)
        axis.spines[["top", "right"]].set_visible(False)
        fmt = "%.1f%%" if metric == "correction_rate" else "%.3f"
        label_values = values * 100 if metric == "correction_rate" else values
        for bar, value in zip(bars, label_values, strict=True):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt % value, ha="center", va="bottom", fontsize=7)
    fig.suptitle("二次谐波阳性窗口上的四模型结果", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def build_all_charts(repo_root: Path, output_dir: Path) -> dict[str, Path]:
    data = load_chart_data(repo_root)
    return {
        "overall_metrics": plot_overall_metrics(data.overall, output_dir / "overall_metrics.png"),
        "tail_metrics": plot_tail_metrics(data.tail, output_dir / "tail_metrics.png"),
        "stratified_roles": plot_stratified_roles(data.strata, data.rr_bins, output_dir / "stratified_roles.png"),
        "harmonic_coverage": plot_harmonic_coverage(data.harmonic_coverage, output_dir / "harmonic_coverage.png"),
        "harmonic_model_results": plot_harmonic_results(data.harmonic, output_dir / "harmonic_model_results.png"),
    }
