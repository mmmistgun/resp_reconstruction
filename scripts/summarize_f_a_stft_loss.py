from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf


METRICS = [
    "rr_peak_band_abs_error",
    "rr_spec_abs_error",
    "breath_count_zero_cross_abs_error",
    "relative_envelope_mae",
    "relative_envelope_corr",
    "spectrum_similarity",
    "band_limited_corr",
    "best_lag_corr",
    "best_lag_sec",
]


def summarize_f_a_runs(manifest_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """读取 F-A manifest，输出逐 run 摘要、paired delta 和分层 paired delta。"""

    manifest = pd.read_csv(manifest_path)
    detail = _detail_frame(manifest)
    paired = _paired_delta_frame(detail)
    strata = _stratified_delta_frame(detail)
    return detail, paired, strata


def _detail_frame(manifest: pd.DataFrame) -> pd.DataFrame:
    records = []
    for row in manifest.to_dict("records"):
        run_root = _run_root_from_overrides(str(row.get("overrides", "")))
        seed = int(row.get("seed"))
        run_dir = _complete_run_for_seed(run_root, seed)
        record = {
            "tag": row.get("tag", ""),
            "label": row.get("label", ""),
            "branch_mode": row.get("branch_mode", ""),
            "seed": seed,
            "paired_f0_label": row.get("paired_f0_label", ""),
            "paired_time_only_label": row.get("paired_time_only_label", ""),
            "run_root": str(run_root),
            "run_dir": str(run_dir) if run_dir is not None else "",
            "status": "complete" if run_dir is not None else "missing",
        }
        if run_dir is not None:
            metrics = pd.read_csv(run_dir / "metrics.csv")
            record.update(_metric_summary(metrics))
            record.update(_history_summary(run_dir / "train_history.csv"))
        records.append(record)
    return pd.DataFrame.from_records(records)


def _metric_summary(metrics: pd.DataFrame) -> dict[str, Any]:
    record: dict[str, Any] = {"n_windows": int(len(metrics))}
    for metric in METRICS:
        values = pd.to_numeric(metrics.get(metric, pd.Series(dtype=float)), errors="coerce")
        record[f"{metric}_mean"] = float(values.mean()) if not values.empty else np.nan
        record[f"{metric}_median"] = float(values.median()) if not values.empty else np.nan
    peak = pd.to_numeric(metrics.get("rr_peak_band_abs_error", pd.Series(dtype=float)), errors="coerce").dropna()
    for threshold in (1.0, 2.0):
        record[f"frac_gt_{threshold:g}"] = float((peak > threshold).mean()) if not peak.empty else np.nan
    return record


def _history_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"epochs": 0, "best_val_loss": np.nan}
    frame = pd.read_csv(path)
    if frame.empty:
        return {"epochs": 0, "best_val_loss": np.nan}
    return {
        "epochs": int(frame["epoch"].iloc[-1]) if "epoch" in frame.columns else len(frame),
        "best_val_loss": float(frame["val_loss"].min()) if "val_loss" in frame.columns else np.nan,
    }


def _paired_delta_frame(detail: pd.DataFrame) -> pd.DataFrame:
    records = []
    candidates = detail[(detail["status"] == "complete") & _candidate_label_mask(detail["label"])]
    baselines = detail[detail["status"] == "complete"]
    for candidate in candidates.to_dict("records"):
        baseline = _matching_row(
            baselines,
            label=str(candidate["paired_f0_label"]),
            seed=int(candidate["seed"]),
        )
        if baseline is None:
            continue
        record = {
            "tag": candidate["tag"],
            "label": candidate["label"],
            "seed": candidate["seed"],
            "baseline_label": baseline["label"],
            "candidate_run_dir": candidate["run_dir"],
            "baseline_run_dir": baseline["run_dir"],
        }
        for column in [f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "median")]:
            record[f"delta_{column}"] = _delta(candidate, baseline, column)
        for column in ("frac_gt_1", "frac_gt_2"):
            record[f"delta_{column}"] = _delta(candidate, baseline, column)
        records.append(record)
    return pd.DataFrame.from_records(records)


def _stratified_delta_frame(detail: pd.DataFrame) -> pd.DataFrame:
    records = []
    candidates = detail[(detail["status"] == "complete") & _candidate_label_mask(detail["label"])]
    baselines = detail[detail["status"] == "complete"]
    for candidate in candidates.to_dict("records"):
        baseline = _matching_row(
            baselines,
            label=str(candidate["paired_f0_label"]),
            seed=int(candidate["seed"]),
        )
        if baseline is None:
            continue
        candidate_metrics = pd.read_csv(Path(candidate["run_dir"]) / "metrics.csv")
        baseline_metrics = pd.read_csv(Path(baseline["run_dir"]) / "metrics.csv")
        joined = baseline_metrics.merge(
            candidate_metrics,
            on="dataset_row_id",
            suffixes=("_baseline", "_candidate"),
        )
        if joined.empty:
            continue
        for stratum, frame in _strata(joined).items():
            if frame.empty:
                continue
            record = {
                "tag": candidate["tag"],
                "label": candidate["label"],
                "seed": candidate["seed"],
                "baseline_label": baseline["label"],
                "stratum": stratum,
                "n_windows": int(len(frame)),
            }
            for metric in METRICS:
                base = pd.to_numeric(frame[f"{metric}_baseline"], errors="coerce")
                cand = pd.to_numeric(frame[f"{metric}_candidate"], errors="coerce")
                record[f"delta_{metric}_mean"] = float((cand - base).mean())
            records.append(record)
    return pd.DataFrame.from_records(records)


def _strata(joined: pd.DataFrame) -> dict[str, pd.DataFrame]:
    baseline_peak = pd.to_numeric(joined["rr_peak_band_abs_error_baseline"], errors="coerce")
    baseline_similarity = pd.to_numeric(joined["spectrum_similarity_baseline"], errors="coerce")
    target_rr = pd.to_numeric(joined["target_rr_peak_band_bpm_baseline"], errors="coerce")
    low_spectrum_threshold = float(baseline_similarity.median()) if baseline_similarity.notna().any() else np.nan
    return {
        "overall": joined,
        "baseline_hard": joined[baseline_peak > 1.0],
        "baseline_easy": joined[baseline_peak <= 0.25],
        "low_spectrum": joined[baseline_similarity <= low_spectrum_threshold],
        "fast_rr": joined[target_rr >= 18.0],
    }


def _matching_row(frame: pd.DataFrame, *, label: str, seed: int) -> dict[str, Any] | None:
    matched = frame[(frame["label"] == label) & (frame["seed"].astype(int) == int(seed))]
    if matched.empty:
        return None
    return matched.iloc[0].to_dict()


def _candidate_label_mask(labels: pd.Series) -> pd.Series:
    text = labels.astype(str)
    return text.str.startswith("F-A") | text.str.startswith("F-B")


def _delta(candidate: dict[str, Any], baseline: dict[str, Any], column: str) -> float:
    return float(candidate.get(column, np.nan)) - float(baseline.get(column, np.nan))


def _run_root_from_overrides(overrides: str) -> Path:
    for item in str(overrides).split():
        if item.startswith("outputs.run_root="):
            return Path(item.split("=", 1)[1])
    raise ValueError(f"manifest overrides 缺少 outputs.run_root: {overrides}")


def _complete_run_for_seed(run_root: Path, seed: int) -> Path | None:
    if not run_root.exists():
        return None
    candidates = sorted(path for path in run_root.iterdir() if path.is_dir() and (path / "metrics.csv").exists())
    matched = [path for path in candidates if _training_seed(path) == int(seed)]
    if matched:
        return matched[-1]
    if len(candidates) == 1 and _training_seed(candidates[0]) is None:
        return candidates[0]
    return None


def _training_seed(run_dir: Path) -> int | None:
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return None
    try:
        config = OmegaConf.load(config_path)
        seed = OmegaConf.select(config, "training.seed")
    except Exception:
        return None
    return int(seed) if seed is not None else None


def _write_optional(frame: pd.DataFrame, path: str | Path | None) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"写出 {output} rows={len(frame)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总 F-A target STFT loss 实验")
    parser.add_argument("--manifest", default="runs/f_a_stft_loss_manifest.csv")
    parser.add_argument("--output", default="runs/f_a_stft_loss_summary.csv")
    parser.add_argument("--paired-output", default="runs/f_a_stft_loss_paired_delta.csv")
    parser.add_argument("--strata-output", default="runs/f_a_stft_loss_strata_delta.csv")
    args = parser.parse_args()

    detail, paired, strata = summarize_f_a_runs(args.manifest)
    _write_optional(detail, args.output)
    _write_optional(paired, args.paired_output)
    _write_optional(strata, args.strata_output)


if __name__ == "__main__":
    main()
