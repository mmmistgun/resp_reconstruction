from __future__ import annotations

import argparse
import csv
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_ARMS = ("E3-C1B_token_pre_mixer", "E3-C1C_token_mid_mixer")
CONTINUITY_ARM = "E3-C1A_concat_post_fusion"
DEFAULT_ABLATION_MODES = ("stft_zero", "stft_shuffle_time", "stft_shuffle_batch")
DEFAULT_METRICS = (
    "rr_peak_band_abs_error",
    "rr_spec_abs_error",
    "relative_envelope_corr",
    "relative_envelope_mae",
    "band_limited_corr",
    "best_lag_corr",
    "spectrum_similarity",
)


@dataclass(frozen=True)
class ManifestSpec:
    label: str
    branch_mode: str
    seed: int
    paired_time_only_label: str
    run_root: Path


def build_strata_frame(baseline: pd.DataFrame, *, success_threshold: float = 1.0) -> pd.DataFrame:
    """按 paired time-only baseline 固定 C2 分层边界。"""

    _require_columns(baseline, ["dataset_row_id"])
    strata = baseline[["dataset_row_id"]].copy()
    strata["all"] = "all"
    strata["baseline_peak_band_bin"] = _success_bins(
        baseline,
        column="rr_peak_band_abs_error",
        threshold=float(success_threshold),
    )
    strata["rr_peak_valid_ratio_bin"] = _valid_ratio_bins(baseline.get("rr_peak_valid_ratio"), baseline.index)
    strata["band_limited_corr_bin"] = _corr_bins(baseline.get("band_limited_corr"), baseline.index)
    strata["spectrum_similarity_bin"] = _tertile_bins(baseline.get("spectrum_similarity"), baseline.index)
    strata["target_rr_bin"] = _target_rr_bins(_target_rr_series(baseline), baseline.index)
    return strata


def summarize_metric_delta(
    *,
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    strata: pd.DataFrame,
    metrics: Sequence[str],
    delta_kind: str,
    comparison: str,
    min_windows: int,
    metadata: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """同窗口计算 candidate-reference 的逐指标 delta，并按固定 strata 聚合。

    这里不解释 delta 正负好坏：error 类指标越低越好，corr 类指标越高越好。输出保留
    candidate/reference 均值，后续记录结论时按指标语义解释。
    """

    selected_metrics = [str(metric) for metric in metrics]
    _require_columns(candidate, ["dataset_row_id", *selected_metrics])
    _require_columns(reference, ["dataset_row_id", *selected_metrics])
    _require_columns(strata, ["dataset_row_id"])
    _reject_duplicate_ids(candidate, "candidate")
    _reject_duplicate_ids(reference, "reference")
    _reject_duplicate_ids(strata, "strata")

    merged = candidate[["dataset_row_id", *selected_metrics]].merge(
        reference[["dataset_row_id", *selected_metrics]],
        on="dataset_row_id",
        how="inner",
        suffixes=("_candidate", "_reference"),
    )
    merged = merged.merge(strata, on="dataset_row_id", how="inner")
    if merged.empty:
        raise ValueError(f"{comparison} 没有可 join 的 dataset_row_id")

    strata_columns = [col for col in strata.columns if col != "dataset_row_id"]
    rows: list[dict[str, Any]] = []
    for metric in selected_metrics:
        cand_col = f"{metric}_candidate"
        ref_col = f"{metric}_reference"
        merged[f"{metric}_delta"] = merged[cand_col] - merged[ref_col]
        for stratum_name in strata_columns:
            for stratum_value, group in merged.groupby(stratum_name, dropna=False):
                rows.append(
                    _summary_row(
                        group,
                        metric=metric,
                        delta_col=f"{metric}_delta",
                        candidate_col=cand_col,
                        reference_col=ref_col,
                        delta_kind=str(delta_kind),
                        comparison=str(comparison),
                        stratum_name=str(stratum_name),
                        stratum_value=str(stratum_value),
                        min_windows=int(min_windows),
                        metadata=metadata,
                    )
                )
    return pd.DataFrame(rows)


def load_manifest(path: Path) -> list[ManifestSpec]:
    with path.open(newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    specs: list[ManifestSpec] = []
    for row in rows:
        run_root = _extract_run_root(str(row.get("overrides", "")))
        specs.append(
            ManifestSpec(
                label=str(row["label"]),
                branch_mode=str(row["branch_mode"]),
                seed=int(row["seed"]),
                paired_time_only_label=str(row["paired_time_only_label"]),
                run_root=Path(run_root),
            )
        )
    return specs


def build_c2_summaries(
    specs: Sequence[ManifestSpec],
    *,
    arms: Sequence[str],
    ablation_modes: Sequence[str],
    ablation_suffix: str,
    metrics: Sequence[str],
    success_threshold: float,
    min_windows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_key = {(spec.label, spec.branch_mode, spec.seed): spec for spec in specs}
    training_frames: list[pd.DataFrame] = []
    ablation_frames: list[pd.DataFrame] = []
    for label in arms:
        dual_specs = [spec for spec in specs if spec.label == label and spec.branch_mode == "dual"]
        if not dual_specs:
            raise ValueError(f"manifest 中未找到 dual arm: {label}")
        for dual_spec in sorted(dual_specs, key=lambda item: item.seed):
            paired_key = (dual_spec.paired_time_only_label, "time_only", dual_spec.seed)
            if paired_key not in by_key:
                raise ValueError(f"缺少配对 time-only substrate: {paired_key}")

            dual_run = find_seed_run_dir(dual_spec.run_root, dual_spec.seed)
            time_spec = by_key[paired_key]
            time_run = find_seed_run_dir(time_spec.run_root, time_spec.seed)
            dual_metrics = read_metrics(dual_run / "metrics.csv")
            time_metrics = read_metrics(time_run / "metrics.csv")
            strata = build_strata_frame(time_metrics, success_threshold=success_threshold)
            base_meta = {
                "label": dual_spec.label,
                "seed": dual_spec.seed,
                "paired_time_only_label": dual_spec.paired_time_only_label,
            }
            training_frames.append(
                summarize_metric_delta(
                    candidate=dual_metrics,
                    reference=time_metrics,
                    strata=strata,
                    metrics=metrics,
                    delta_kind="dual_minus_time_only",
                    comparison=f"{dual_spec.label}_dual_vs_{dual_spec.paired_time_only_label}",
                    min_windows=min_windows,
                    metadata=base_meta,
                )
            )
            normal_metrics = read_ablation_metrics(dual_run, mode="normal", suffix=ablation_suffix)
            for mode in ablation_modes:
                ablated = read_ablation_metrics(dual_run, mode=mode, suffix=ablation_suffix)
                ablation_frames.append(
                    summarize_metric_delta(
                        candidate=normal_metrics,
                        reference=ablated,
                        strata=strata,
                        metrics=metrics,
                        delta_kind=f"normal_minus_{mode}",
                        comparison=f"{dual_spec.label}_normal_vs_{mode}",
                        min_windows=min_windows,
                        metadata=base_meta,
                    )
                )
    training = pd.concat(training_frames, ignore_index=True) if training_frames else pd.DataFrame()
    ablation = pd.concat(ablation_frames, ignore_index=True) if ablation_frames else pd.DataFrame()
    return training, ablation


def find_seed_run_dir(run_root: Path, seed: int) -> Path:
    if not run_root.exists():
        raise FileNotFoundError(f"run_root 不存在: {run_root}")
    matches: list[Path] = []
    for run_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        config_path = run_dir / "config.yaml"
        if not config_path.exists():
            continue
        if _config_seed(config_path) == int(seed):
            matches.append(run_dir)
    if len(matches) != 1:
        raise ValueError(f"{run_root} seed={seed} 匹配到 {len(matches)} 个 run: {[str(item) for item in matches]}")
    return matches[0]


def read_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"metrics 文件不存在: {path}")
    return pd.read_csv(path)


def read_ablation_metrics(run_dir: Path, *, mode: str, suffix: str) -> pd.DataFrame:
    path = run_dir / f"metrics_e3c_{mode}_{suffix}.csv"
    if path.exists():
        return read_metrics(path)
    if str(mode) == "normal" and str(suffix) == "best":
        return read_metrics(run_dir / "metrics.csv")
    raise FileNotFoundError(f"缺少 C2 消融 metrics: {path}")


def _summary_row(
    group: pd.DataFrame,
    *,
    metric: str,
    delta_col: str,
    candidate_col: str,
    reference_col: str,
    delta_kind: str,
    comparison: str,
    stratum_name: str,
    stratum_value: str,
    min_windows: int,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    n_windows = int(len(group))
    enough = n_windows >= int(min_windows)
    delta = group[delta_col]
    row: dict[str, Any] = dict(metadata or {})
    row.update(
        {
            "delta_kind": delta_kind,
            "comparison": comparison,
            "metric": metric,
            "stratum_name": stratum_name,
            "stratum_value": stratum_value,
            "n_windows": n_windows,
            "passes_min_windows": bool(enough),
            "mean_delta": _rounded(delta.mean()) if enough else np.nan,
            "median_delta": _rounded(delta.median()) if enough else np.nan,
            "p25_delta": _rounded(delta.quantile(0.25)) if enough else np.nan,
            "p75_delta": _rounded(delta.quantile(0.75)) if enough else np.nan,
            "candidate_mean": _rounded(group[candidate_col].mean()) if enough else np.nan,
            "reference_mean": _rounded(group[reference_col].mean()) if enough else np.nan,
        }
    )
    return row


def _success_bins(frame: pd.DataFrame, *, column: str, threshold: float) -> pd.Series:
    if column not in frame:
        return pd.Series(["unknown"] * len(frame), index=frame.index)
    return pd.Series(np.where(frame[column] <= float(threshold), "success", "failure"), index=frame.index)


def _valid_ratio_bins(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return _unknown_series(index)
    values = pd.to_numeric(series, errors="coerce")
    bins = np.select(
        [values >= 0.9, (values >= 0.6) & (values < 0.9), values < 0.6],
        ["high", "mid", "low"],
        default="unknown",
    )
    return pd.Series(bins, index=series.index)


def _corr_bins(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return _unknown_series(index)
    values = pd.to_numeric(series, errors="coerce")
    bins = np.select(
        [values > 0.2, values < -0.2, values.between(-0.2, 0.2, inclusive="both")],
        ["positive", "negative", "low_corr"],
        default="unknown",
    )
    return pd.Series(bins, index=series.index)


def _tertile_bins(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return _unknown_series(index)
    values = pd.to_numeric(series, errors="coerce")
    q33 = values.quantile(1 / 3)
    q66 = values.quantile(2 / 3)
    if not np.isfinite(q33) or not np.isfinite(q66) or q33 >= q66:
        return pd.Series(["flat"] * len(values), index=values.index)
    bins = np.select(
        [values <= q33, values >= q66],
        ["low", "high"],
        default="mid",
    )
    return pd.Series(bins, index=series.index)


def _target_rr_series(frame: pd.DataFrame) -> pd.Series | None:
    for column in ("target_rr_peak_band_bpm", "target_rr_peak_bpm", "target_rr_spec_bpm"):
        if column in frame:
            return frame[column]
    return None


def _target_rr_bins(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return _unknown_series(index)
    values = pd.to_numeric(series, errors="coerce")
    bins = np.select(
        [values < 10.0, (values >= 10.0) & (values <= 20.0), values > 20.0],
        ["slow", "normal", "fast"],
        default="unknown",
    )
    return pd.Series(bins, index=series.index)


def _reject_duplicate_ids(frame: pd.DataFrame, name: str) -> None:
    duplicated = frame["dataset_row_id"].duplicated()
    if bool(duplicated.any()):
        ids = frame.loc[duplicated, "dataset_row_id"].head(5).tolist()
        raise ValueError(f"{name} 存在重复 dataset_row_id: {ids}")


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"缺少列: {missing}")


def _extract_run_root(overrides: str) -> str:
    match = re.search(r"(?:^|\s)outputs\.run_root=([^\s]+)", overrides)
    if not match:
        raise ValueError(f"manifest overrides 缺少 outputs.run_root: {overrides}")
    return match.group(1)


def _config_seed(config_path: Path) -> int | None:
    match = re.search(r"(?m)^\s*seed:\s*(\d+)\s*$", config_path.read_text(encoding="utf-8"))
    return int(match.group(1)) if match else None


def _rounded(value: float) -> float:
    return float(round(float(value), 12))


def _unknown_series(index: pd.Index) -> pd.Series:
    return pd.Series(["unknown"] * len(index), index=index)


def main() -> None:
    parser = argparse.ArgumentParser(description="E3-C2：分层汇总 dual-time 与 normal-ablated 两类 delta")
    parser.add_argument("--manifest", default="runs/e3_c1_manifest.csv", help="E3-C1 manifest 路径")
    parser.add_argument("--output-dir", default="runs/e3_c2", help="C2 汇总输出目录")
    parser.add_argument("--arm", action="append", default=None, help="要纳入的 dual arm label，可重复")
    parser.add_argument("--include-continuity", action="store_true", help=f"额外纳入 {CONTINUITY_ARM}")
    parser.add_argument("--ablation-mode", action="append", default=None, help="消融 mode，可重复")
    parser.add_argument("--ablation-suffix", default="best", help="metrics_e3c_<mode>_<suffix>.csv 的 suffix")
    parser.add_argument("--metric", action="append", default=None, help="要汇总的指标列，可重复")
    parser.add_argument("--success-threshold", type=float, default=1.0, help="baseline peak-band 成功阈值")
    parser.add_argument("--min-windows", type=int, default=50, help="分层定量结论的最小窗口数")
    args = parser.parse_args()

    arms = list(args.arm or DEFAULT_ARMS)
    if args.include_continuity and CONTINUITY_ARM not in arms:
        arms.append(CONTINUITY_ARM)
    ablation_modes = list(args.ablation_mode or DEFAULT_ABLATION_MODES)
    metrics = list(args.metric or DEFAULT_METRICS)

    training, ablation = build_c2_summaries(
        load_manifest(Path(args.manifest)),
        arms=arms,
        ablation_modes=ablation_modes,
        ablation_suffix=str(args.ablation_suffix),
        metrics=metrics,
        success_threshold=float(args.success_threshold),
        min_windows=int(args.min_windows),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_path = output_dir / "dual_minus_time_only_by_layer.csv"
    ablation_path = output_dir / "normal_minus_ablation_by_layer.csv"
    training.to_csv(training_path, index=False)
    ablation.to_csv(ablation_path, index=False)
    print(f"写出训练差异分层表: {training_path} rows={len(training)}")
    print(f"写出消融依赖分层表: {ablation_path} rows={len(ablation)}")


if __name__ == "__main__":
    main()
