from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


COUNT_ERROR_METRIC = "breath_count_zero_cross_abs_error"
COUNT_BPM_ERROR_METRIC = "breath_count_zero_cross_bpm_error"

DEFAULT_METRICS = [
    "rr_peak_band_robust_abs_error",
    COUNT_BPM_ERROR_METRIC,
    "best_lag_corr_4s",
    "relative_envelope_corr_lag4s",
    "relative_envelope_mae_lag4s",
    "local_rr_mae",
    "local_rr_corr",
    "rr_peak_band_abs_error",
    "rr_spec_abs_error",
]

TAIL_METRICS = [
    "rr_peak_band_robust_abs_error",
    COUNT_BPM_ERROR_METRIC,
    "local_rr_mae",
]

RR_COLUMNS = [
    "target_rr_peak_band_robust_bpm",
    "target_rr_peak_band_bpm",
    "target_rr_spec_bpm",
]

DEFAULT_RR_BINS = [0.0, 10.0, 14.0, 18.0, 24.0, np.inf]
DEFAULT_RR_BIN_LABELS = ["rr_lt10", "rr_10_14", "rr_14_18", "rr_18_24", "rr_ge24"]


@dataclass(frozen=True)
class Comparison:
    name: str
    target: str
    baseline: str


@dataclass(frozen=True)
class StratifiedAnalysisSpec:
    eval_root: Path
    output_dir: Path
    comparisons: list[Comparison]
    labels: list[str] = field(default_factory=list)
    seeds: list[int] = field(default_factory=list)
    dataset_index: Path | None = None
    file_pattern: str = "{label}_{seed}_test_metrics.csv"
    metrics: list[str] = field(default_factory=lambda: list(DEFAULT_METRICS))
    tail_metrics: list[str] = field(default_factory=lambda: list(TAIL_METRICS))
    rr_bins: list[float] = field(default_factory=lambda: list(DEFAULT_RR_BINS))
    rr_bin_labels: list[str] = field(default_factory=lambda: list(DEFAULT_RR_BIN_LABELS))
    hard_threshold: float = 1.0
    easy_threshold: float = 0.25
    window_seconds: float = 180.0


def run_analysis(spec: StratifiedAnalysisSpec) -> dict[str, Path]:
    labels = _labels_for_spec(spec)
    seeds = spec.seeds or _discover_common_seeds(spec.eval_root, labels, spec.file_pattern)
    metrics_by_run = {
        (label, seed): _with_derived_metrics(
            _read_metrics(spec.eval_root, spec.file_pattern, label, seed),
            spec,
        )
        for label in labels
        for seed in seeds
    }
    dataset_index = _read_dataset_index(spec.dataset_index)
    spec.output_dir.mkdir(parents=True, exist_ok=True)

    strata_seed = _strata_seed_summary(spec, seeds, metrics_by_run)
    strata_summary = _aggregate_seed_rows(strata_seed, ["comparison", "target", "baseline", "stratum"])
    rr_seed = _rr_bin_seed_summary(spec, seeds, metrics_by_run)
    rr_summary = _aggregate_seed_rows(rr_seed, ["comparison", "target", "baseline", "rr_bin"])
    subject_seed = _subject_seed_summary(spec, seeds, metrics_by_run, dataset_index)
    subject_summary = _aggregate_seed_rows(subject_seed, ["comparison", "target", "baseline", "samp_id"])
    tail_seed = _tail_seed_summary(spec, labels, seeds, metrics_by_run)
    tail_summary = _aggregate_tail_rows(tail_seed)

    outputs = {
        "strata_seed": spec.output_dir / "strata_seed_summary.csv",
        "strata": spec.output_dir / "strata_summary.csv",
        "tail_seed": spec.output_dir / "tail_seed_summary.csv",
        "tail": spec.output_dir / "tail_summary.csv",
        "rr_bin_seed": spec.output_dir / "rr_bin_seed_summary.csv",
        "rr_bin": spec.output_dir / "rr_bin_summary.csv",
        "subject_seed": spec.output_dir / "subject_seed_summary.csv",
        "subject": spec.output_dir / "subject_summary.csv",
        "manifest": spec.output_dir / "analysis_manifest.json",
    }
    _write_csv(strata_seed, outputs["strata_seed"])
    _write_csv(strata_summary, outputs["strata"])
    _write_csv(tail_seed, outputs["tail_seed"])
    _write_csv(tail_summary, outputs["tail"])
    _write_csv(rr_seed, outputs["rr_bin_seed"])
    _write_csv(rr_summary, outputs["rr_bin"])
    _write_csv(subject_seed, outputs["subject_seed"])
    _write_csv(subject_summary, outputs["subject"])
    outputs["manifest"].write_text(
        json.dumps(_manifest(spec, labels, seeds, dataset_index), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return outputs


def run_external_strata_analysis(
    spec: StratifiedAnalysisSpec,
    strata: pd.DataFrame,
    *,
    stratum_column: str = "stratum",
) -> dict[str, pd.DataFrame]:
    """按输入/目标预先定义的固定窗口标签汇总模型指标，禁止 outcome-defined 分层。"""

    required = {"dataset_row_id", "samp_id", stratum_column, "harmonic_positive"}
    missing = required - set(strata.columns)
    if missing:
        raise ValueError(f"外部分层标签缺少列: {sorted(missing)}")
    if strata["dataset_row_id"].duplicated().any():
        raise ValueError("外部分层标签存在重复 dataset_row_id")
    labels = _labels_for_spec(spec)
    seeds = spec.seeds or _discover_common_seeds(spec.eval_root, labels, spec.file_pattern)
    fixed = strata.copy().set_index("dataset_row_id", drop=False)
    fixed_ids = set(fixed.index.astype(int).tolist())

    metrics_by_run: dict[tuple[str, int], pd.DataFrame] = {}
    for label in labels:
        for seed in seeds:
            metrics = _with_derived_metrics(
                _read_metrics(spec.eval_root, spec.file_pattern, label, seed),
                spec,
            )
            if metrics["dataset_row_id"].duplicated().any():
                raise ValueError(f"label={label} seed={seed} metrics 存在重复 dataset_row_id")
            actual_ids = set(metrics["dataset_row_id"].astype(int).tolist())
            if actual_ids != fixed_ids:
                missing_ids = sorted(fixed_ids - actual_ids)
                extra_ids = sorted(actual_ids - fixed_ids)
                raise ValueError(
                    f"label={label} seed={seed} 与固定分层 row 不一致: "
                    f"missing={missing_ids[:8]} extra={extra_ids[:8]}"
                )
            metrics_by_run[(label, seed)] = metrics

    model_records: list[dict] = []
    masks = _external_strata_masks(fixed, stratum_column=stratum_column)
    for label in labels:
        for seed in seeds:
            metrics = metrics_by_run[(label, seed)].set_index("dataset_row_id").reindex(fixed.index)
            for stratum_name, mask in masks.items():
                model_records.append(
                    _external_model_record(
                        label=label,
                        seed=seed,
                        stratum=stratum_name,
                        metrics=metrics,
                        strata=fixed,
                        mask=mask,
                        metric_names=spec.metrics,
                    )
                )
    model_seed = pd.DataFrame.from_records(model_records)
    model_summary = _long_seed_summary(model_seed, keys=["label", "stratum"])

    paired_records: list[dict] = []
    for comparison in spec.comparisons:
        for seed in seeds:
            joined = _joined_metrics(metrics_by_run, comparison, seed).reindex(fixed.index)
            for stratum_name, mask in masks.items():
                record = _paired_record(
                    comparison=comparison,
                    seed=seed,
                    group_name="stratum",
                    group_value=stratum_name,
                    joined=joined,
                    mask=mask,
                    metrics=spec.metrics,
                )
                record["n_subjects"] = int(fixed.loc[mask, "samp_id"].nunique())
                paired_records.append(record)
    paired_seed = pd.DataFrame.from_records(paired_records)
    paired_summary = _aggregate_seed_rows(
        paired_seed,
        ["comparison", "target", "baseline", "stratum"],
    )
    return {
        "model_seed": model_seed,
        "model_summary": model_summary,
        "paired_seed": paired_seed,
        "paired_summary": paired_summary,
    }


def _external_strata_masks(
    strata: pd.DataFrame,
    *,
    stratum_column: str,
) -> dict[str, pd.Series]:
    status = strata[stratum_column].astype(str)
    excluded = status.isin({"tho_reference_unstable", "second_harmonic_out_of_band"})
    masks: dict[str, pd.Series] = {
        "all_windows": pd.Series(True, index=strata.index),
        "eligible_total": ~excluded,
        "harmonic_positive_union": strata["harmonic_positive"].astype(bool),
    }
    for value in sorted(status.unique().tolist()):
        masks[value] = status.eq(value)
    return masks


def _external_model_record(
    *,
    label: str,
    seed: int,
    stratum: str,
    metrics: pd.DataFrame,
    strata: pd.DataFrame,
    mask: pd.Series,
    metric_names: list[str],
) -> dict:
    mask = mask.reindex(metrics.index).fillna(False)
    record: dict[str, float | int | str] = {
        "label": label,
        "seed": int(seed),
        "stratum": stratum,
        "n_windows": int(mask.sum()),
        "n_subjects": int(strata.loc[mask, "samp_id"].nunique()),
    }
    for metric in metric_names:
        if metric not in metrics.columns:
            continue
        values = pd.to_numeric(metrics.loc[mask, metric], errors="coerce").dropna()
        if values.empty:
            continue
        record[f"{metric}_mean"] = float(values.mean())
        record[f"{metric}_median"] = float(values.median())
        record[f"{metric}_p95"] = float(values.quantile(0.95))
        if metric in {
            "rr_peak_band_robust_abs_error",
            COUNT_BPM_ERROR_METRIC,
            "local_rr_mae",
        }:
            record[f"{metric}_frac_gt_1"] = float((values > 1.0).mean())
            record[f"{metric}_frac_gt_2"] = float((values > 2.0).mean())
    return record


def _long_seed_summary(frame: pd.DataFrame, *, keys: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    value_columns = [
        column
        for column in frame.columns
        if column not in {*keys, "seed", "n_windows", "n_subjects"}
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    melted = frame.melt(
        id_vars=[*keys, "seed", "n_windows", "n_subjects"],
        value_vars=value_columns,
        var_name="metric_stat",
        value_name="value",
    )
    return (
        melted.groupby([*keys, "metric_stat"], dropna=False)
        .agg(
            value_mean=("value", "mean"),
            value_std=("value", "std"),
            n_windows_mean=("n_windows", "mean"),
            n_subjects_mean=("n_subjects", "mean"),
        )
        .reset_index()
    )


def _labels_for_spec(spec: StratifiedAnalysisSpec) -> list[str]:
    if spec.labels:
        return list(dict.fromkeys(spec.labels))
    labels: list[str] = []
    for comparison in spec.comparisons:
        labels.extend([comparison.target, comparison.baseline])
    return list(dict.fromkeys(labels))


def _read_metrics(eval_root: Path, file_pattern: str, label: str, seed: int) -> pd.DataFrame:
    path = eval_root / file_pattern.format(label=label, seed=seed)
    if not path.exists():
        raise FileNotFoundError(f"找不到 metrics 文件: {path}")
    frame = pd.read_csv(path)
    if "dataset_row_id" not in frame.columns:
        raise ValueError(f"{path} 缺少 dataset_row_id，无法做 paired 分层")
    return frame


def _with_derived_metrics(frame: pd.DataFrame, spec: StratifiedAnalysisSpec) -> pd.DataFrame:
    if spec.window_seconds <= 0:
        raise ValueError("window_seconds 必须为正数")
    if COUNT_ERROR_METRIC not in frame.columns:
        return frame
    result = frame.copy()
    window_minutes = spec.window_seconds / 60.0
    result[COUNT_BPM_ERROR_METRIC] = (
        pd.to_numeric(result[COUNT_ERROR_METRIC], errors="coerce") / window_minutes
    )
    return result


def _read_dataset_index(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    frame = pd.read_csv(path)
    if "dataset_row_id" not in frame.columns:
        raise ValueError(f"{path} 缺少 dataset_row_id，无法回连 subject-level 信息")
    return frame.drop_duplicates("dataset_row_id").set_index("dataset_row_id")


def _discover_common_seeds(eval_root: Path, labels: Iterable[str], file_pattern: str) -> list[int]:
    if file_pattern != "{label}_{seed}_test_metrics.csv":
        raise ValueError("自定义 file_pattern 时必须显式传入 --seeds")
    discovered: list[set[int]] = []
    for label in labels:
        seeds = set()
        prefix = f"{label}_"
        suffix = "_test_metrics.csv"
        for path in eval_root.glob(f"{label}_*_test_metrics.csv"):
            name = path.name
            if not name.startswith(prefix) or not name.endswith(suffix):
                continue
            seed_text = name[len(prefix) : -len(suffix)]
            if seed_text.isdigit():
                seeds.add(int(seed_text))
        if not seeds:
            raise ValueError(f"无法为 label={label!r} 自动发现 seed")
        discovered.append(seeds)
    common = set.intersection(*discovered) if discovered else set()
    if not common:
        raise ValueError("各 label 没有共同 seed，请显式传入 --seeds")
    return sorted(common)


def _strata_seed_summary(
    spec: StratifiedAnalysisSpec,
    seeds: list[int],
    metrics_by_run: dict[tuple[str, int], pd.DataFrame],
) -> pd.DataFrame:
    records = []
    for comparison in spec.comparisons:
        for seed in seeds:
            joined = _joined_metrics(metrics_by_run, comparison, seed)
            base = _baseline_view(joined)
            for stratum, mask in _strata_masks(base, spec).items():
                records.append(
                    _paired_record(
                        comparison=comparison,
                        seed=seed,
                        group_name="stratum",
                        group_value=stratum,
                        joined=joined,
                        mask=mask,
                        metrics=spec.metrics,
                    )
                )
    return pd.DataFrame.from_records(records)


def _rr_bin_seed_summary(
    spec: StratifiedAnalysisSpec,
    seeds: list[int],
    metrics_by_run: dict[tuple[str, int], pd.DataFrame],
) -> pd.DataFrame:
    records = []
    for comparison in spec.comparisons:
        for seed in seeds:
            joined = _joined_metrics(metrics_by_run, comparison, seed)
            base = _baseline_view(joined)
            for rr_bin, mask in _rr_bin_masks(base, spec).items():
                if mask.any():
                    records.append(
                        _paired_record(
                            comparison=comparison,
                            seed=seed,
                            group_name="rr_bin",
                            group_value=rr_bin,
                            joined=joined,
                            mask=mask,
                            metrics=spec.metrics,
                        )
                    )
    return pd.DataFrame.from_records(records)


def _subject_seed_summary(
    spec: StratifiedAnalysisSpec,
    seeds: list[int],
    metrics_by_run: dict[tuple[str, int], pd.DataFrame],
    dataset_index: pd.DataFrame | None,
) -> pd.DataFrame:
    if dataset_index is None or "samp_id" not in dataset_index.columns:
        return pd.DataFrame(
            columns=["comparison", "target", "baseline", "seed", "samp_id", "n_windows"]
        )
    records = []
    for comparison in spec.comparisons:
        for seed in seeds:
            joined = _joined_metrics(metrics_by_run, comparison, seed)
            joined = joined.join(dataset_index[["samp_id"]], how="left")
            for samp_id, frame in joined.groupby("samp_id", dropna=True):
                mask = joined.index.isin(frame.index)
                records.append(
                    _paired_record(
                        comparison=comparison,
                        seed=seed,
                        group_name="samp_id",
                        group_value=int(samp_id),
                        joined=joined,
                        mask=pd.Series(mask, index=joined.index),
                        metrics=spec.metrics,
                    )
                )
    return pd.DataFrame.from_records(records)


def _tail_seed_summary(
    spec: StratifiedAnalysisSpec,
    labels: list[str],
    seeds: list[int],
    metrics_by_run: dict[tuple[str, int], pd.DataFrame],
) -> pd.DataFrame:
    records = []
    for label in labels:
        for seed in seeds:
            frame = metrics_by_run[(label, seed)]
            for metric in spec.tail_metrics:
                if metric not in frame.columns:
                    continue
                values = pd.to_numeric(frame[metric], errors="coerce").dropna()
                if values.empty:
                    continue
                for stat, value in _tail_stats(values).items():
                    records.append(
                        {
                            "label": label,
                            "seed": seed,
                            "metric": metric,
                            "stat": stat,
                            "value": value,
                            "n_windows": int(len(values)),
                        }
                    )
    return pd.DataFrame.from_records(records)


def _tail_stats(values: pd.Series) -> dict[str, float]:
    stats = {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p90": float(values.quantile(0.90)),
        "p95": float(values.quantile(0.95)),
    }
    if values.name in {"rr_peak_band_robust_abs_error", "rr_peak_band_abs_error", None}:
        stats["frac_gt_1"] = float((values > 1.0).mean())
        stats["frac_gt_2"] = float((values > 2.0).mean())
    else:
        stats["frac_gt_1"] = float((values > 1.0).mean())
        stats["frac_gt_2"] = float((values > 2.0).mean())
    return stats


def _joined_metrics(
    metrics_by_run: dict[tuple[str, int], pd.DataFrame],
    comparison: Comparison,
    seed: int,
) -> pd.DataFrame:
    baseline = metrics_by_run[(comparison.baseline, seed)].set_index("dataset_row_id")
    target = metrics_by_run[(comparison.target, seed)].set_index("dataset_row_id")
    joined = baseline.join(target, how="inner", lsuffix="_baseline", rsuffix="_target")
    if joined.empty:
        raise ValueError(f"{comparison.name} seed={seed} 没有可配对 dataset_row_id")
    return joined


def _baseline_view(joined: pd.DataFrame) -> pd.DataFrame:
    baseline = joined.filter(regex="_baseline$")
    baseline = baseline.rename(columns={column: column[: -len("_baseline")] for column in baseline.columns})
    return baseline


def _strata_masks(base: pd.DataFrame, spec: StratifiedAnalysisSpec) -> dict[str, pd.Series]:
    index = base.index
    masks: dict[str, pd.Series] = {"overall": pd.Series(True, index=index)}
    robust = _numeric(base, "rr_peak_band_robust_abs_error")
    if robust is not None:
        masks["baseline_hard_robust_gt1"] = robust > spec.hard_threshold
        masks["baseline_easy_robust_le025"] = robust <= spec.easy_threshold
    old_peak = _numeric(base, "rr_peak_band_abs_error")
    if old_peak is not None:
        masks["baseline_hard_peak_gt1"] = old_peak > spec.hard_threshold
        masks["baseline_easy_peak_le025"] = old_peak <= spec.easy_threshold
    spectrum = _numeric(base, "spectrum_similarity")
    if spectrum is not None and spectrum.notna().any():
        masks["low_spectrum_le_median"] = spectrum <= float(spectrum.median())
    count = _numeric(base, COUNT_ERROR_METRIC)
    if count is not None:
        masks["baseline_count_error_gt0"] = count > 0
        masks["baseline_count_error_eq0"] = count == 0
    return masks


def _rr_bin_masks(base: pd.DataFrame, spec: StratifiedAnalysisSpec) -> dict[str, pd.Series]:
    rr = None
    for column in RR_COLUMNS:
        rr = _numeric(base, column)
        if rr is not None:
            break
    if rr is None:
        return {}
    if len(spec.rr_bins) != len(spec.rr_bin_labels) + 1:
        raise ValueError("rr_bins 数量必须比 rr_bin_labels 多 1")
    masks = {}
    for low, high, label in zip(spec.rr_bins[:-1], spec.rr_bins[1:], spec.rr_bin_labels):
        if np.isinf(high):
            masks[label] = rr >= low
        else:
            masks[label] = (rr >= low) & (rr < high)
    return masks


def _paired_record(
    *,
    comparison: Comparison,
    seed: int,
    group_name: str,
    group_value: str | int,
    joined: pd.DataFrame,
    mask: pd.Series,
    metrics: list[str],
) -> dict:
    mask = mask.reindex(joined.index).fillna(False)
    record = {
        "comparison": comparison.name,
        "target": comparison.target,
        "baseline": comparison.baseline,
        "seed": seed,
        group_name: group_value,
        "n_windows": int(mask.sum()),
    }
    if not mask.any():
        return record
    for metric in metrics:
        base_col = f"{metric}_baseline"
        target_col = f"{metric}_target"
        if base_col not in joined.columns or target_col not in joined.columns:
            continue
        base_values = pd.to_numeric(joined.loc[mask, base_col], errors="coerce")
        target_values = pd.to_numeric(joined.loc[mask, target_col], errors="coerce")
        record[f"baseline_{metric}_mean"] = float(base_values.mean())
        record[f"target_{metric}_mean"] = float(target_values.mean())
        record[f"delta_{metric}_mean"] = float((target_values - base_values).mean())
    return record


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series | None:
    if column not in frame.columns:
        return None
    return pd.to_numeric(frame[column], errors="coerce")


def _aggregate_seed_rows(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    value_columns = [
        column
        for column in frame.columns
        if column not in {*keys, "seed"} and pd.api.types.is_numeric_dtype(frame[column])
    ]
    grouped = frame.groupby(keys, dropna=False)
    mean_frame = grouped[value_columns].mean().reset_index()
    std_frame = grouped[value_columns].std(ddof=1).reset_index()
    std_frame = std_frame.rename(columns={column: f"{column}_std" for column in value_columns})
    return mean_frame.merge(std_frame, on=keys, how="left")


def _aggregate_tail_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    grouped = frame.groupby(["label", "metric", "stat"], dropna=False)
    summary = grouped.agg(
        value_mean=("value", "mean"),
        value_std=("value", "std"),
        n_windows_mean=("n_windows", "mean"),
    )
    return summary.reset_index()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _manifest(
    spec: StratifiedAnalysisSpec,
    labels: list[str],
    seeds: list[int],
    dataset_index: pd.DataFrame | None,
) -> dict:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "eval_root": str(spec.eval_root),
        "output_dir": str(spec.output_dir),
        "file_pattern": spec.file_pattern,
        "labels": labels,
        "seeds": seeds,
        "comparisons": [comparison.__dict__ for comparison in spec.comparisons],
        "dataset_index": str(spec.dataset_index) if spec.dataset_index else None,
        "dataset_index_has_subject": bool(dataset_index is not None and "samp_id" in dataset_index.columns),
        "window_seconds": spec.window_seconds,
        "metrics": spec.metrics,
        "tail_metrics": spec.tail_metrics,
        "strata": {
            "baseline_hard_robust_gt1": f"baseline rr_peak_band_robust_abs_error > {spec.hard_threshold}",
            "baseline_easy_robust_le025": f"baseline rr_peak_band_robust_abs_error <= {spec.easy_threshold}",
            "low_spectrum_le_median": "baseline spectrum_similarity <= seed 内 baseline median",
            "baseline_count_error_gt0": f"baseline {COUNT_ERROR_METRIC} > 0",
            "baseline_count_error_eq0": f"baseline {COUNT_ERROR_METRIC} == 0",
        },
        "rr_bins": [
            {"label": label, "low": low, "high": None if np.isinf(high) else high}
            for label, low, high in zip(spec.rr_bin_labels, spec.rr_bins[:-1], spec.rr_bins[1:])
        ],
        "notes": [
            "本脚本只读取已有逐窗口 metrics，不重新推理、不重评 checkpoint、不覆盖历史 run 文件。",
            "所有 paired delta 均按 dataset_row_id 在同 seed 的 target/baseline 之间配对。",
            f"{COUNT_BPM_ERROR_METRIC} = {COUNT_ERROR_METRIC} / (window_seconds / 60)，表示整窗平均周期率误差。",
        ],
    }


def _parse_comparison(text: str) -> Comparison:
    if "=" in text:
        name, rest = text.split("=", 1)
    else:
        parts = text.split(":", 1)
        if len(parts) != 2:
            raise argparse.ArgumentTypeError("comparison 格式应为 name=target:baseline")
        name, rest = parts
    pieces = rest.split(":")
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("comparison 格式应为 name=target:baseline")
    target, baseline = pieces
    return Comparison(name=name, target=target, baseline=baseline)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线构建 THO eval 逐窗口 metrics 的可复用分层分析表")
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--comparison", action="append", type=_parse_comparison, required=True)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--dataset-index", type=Path, default=None)
    parser.add_argument("--file-pattern", default="{label}_{seed}_test_metrics.csv")
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=180.0,
        help="用于把周期数量误差换算成平均 bpm 误差的窗口长度，默认 180 秒",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    spec = StratifiedAnalysisSpec(
        eval_root=args.eval_root,
        output_dir=args.output_dir,
        labels=args.labels or [],
        seeds=args.seeds or [],
        comparisons=args.comparison,
        dataset_index=args.dataset_index,
        file_pattern=args.file_pattern,
        window_seconds=args.window_seconds,
    )
    outputs = run_analysis(spec)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
