from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.summarize_f_a_stft_loss import _complete_run_for_seed, _run_root_from_overrides


WINDOW_METRICS = [
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

PREDICTION_COLUMNS = [
    "pred_rr_peak_band_bpm",
    "target_rr_peak_band_bpm",
]


def diagnose_f_a2_guard_windows(
    manifest_path: str | Path,
    *,
    candidate_labels: Iterable[str] | None = None,
    output_dir: str | Path | None = None,
    top_n: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """生成 F-A2 护栏窗口级诊断表。"""

    manifest = pd.read_csv(manifest_path)
    selected = _candidate_rows(manifest, candidate_labels)
    if selected.empty:
        raise ValueError("manifest 中没有匹配的 F-A 候选")

    frames: list[pd.DataFrame] = []
    for candidate in selected.to_dict("records"):
        seed = int(candidate["seed"])
        baseline = _matching_baseline(manifest, label=str(candidate["paired_f0_label"]), seed=seed)
        if baseline is None:
            continue
        candidate_dir = _complete_run_for_seed(_run_root_from_overrides(str(candidate["overrides"])), seed)
        baseline_dir = _complete_run_for_seed(_run_root_from_overrides(str(baseline["overrides"])), seed)
        if candidate_dir is None or baseline_dir is None:
            continue
        frames.append(
            _window_delta_for_pair(
                candidate,
                baseline,
                candidate_dir=candidate_dir,
                baseline_dir=baseline_dir,
            )
        )

    if not frames:
        raise ValueError("没有可诊断的 complete paired run")

    window_delta = pd.concat(frames, ignore_index=True)
    bucket_summary = _bucket_summary(window_delta)
    top_degraded_easy = (
        window_delta[window_delta["baseline_easy"]]
        .sort_values("delta_rr_peak_band_abs_error", ascending=False)
        .head(int(top_n))
        .reset_index(drop=True)
    )
    top_improved_hard = (
        window_delta[window_delta["baseline_hard"]]
        .sort_values("delta_rr_peak_band_abs_error", ascending=True)
        .head(int(top_n))
        .reset_index(drop=True)
    )

    if output_dir is not None:
        _write_outputs(
            Path(output_dir),
            window_delta=window_delta,
            bucket_summary=bucket_summary,
            top_degraded_easy=top_degraded_easy,
            top_improved_hard=top_improved_hard,
        )

    return window_delta, bucket_summary, top_degraded_easy, top_improved_hard


def _candidate_rows(manifest: pd.DataFrame, labels: Iterable[str] | None) -> pd.DataFrame:
    if labels is not None:
        wanted = {str(label) for label in labels}
        return manifest[manifest["label"].astype(str).isin(wanted)].copy()
    return manifest[manifest["label"].astype(str).str.startswith("F-A")].copy()


def _matching_baseline(manifest: pd.DataFrame, *, label: str, seed: int) -> dict | None:
    matched = manifest[(manifest["label"].astype(str) == label) & (manifest["seed"].astype(int) == int(seed))]
    if matched.empty:
        return None
    return matched.iloc[0].to_dict()


def _window_delta_for_pair(
    candidate: dict,
    baseline: dict,
    *,
    candidate_dir: Path,
    baseline_dir: Path,
) -> pd.DataFrame:
    candidate_metrics = pd.read_csv(candidate_dir / "metrics.csv")
    baseline_metrics = pd.read_csv(baseline_dir / "metrics.csv")
    joined = baseline_metrics.merge(candidate_metrics, on="dataset_row_id", suffixes=("_baseline", "_candidate"))
    records = pd.DataFrame(
        {
            "tag": candidate.get("tag", ""),
            "label": candidate.get("label", ""),
            "seed": int(candidate.get("seed")),
            "baseline_label": baseline.get("label", ""),
            "dataset_row_id": joined["dataset_row_id"].astype(int),
            "baseline_run_dir": str(baseline_dir),
            "candidate_run_dir": str(candidate_dir),
        }
    )

    for column in WINDOW_METRICS + PREDICTION_COLUMNS:
        base_col = f"{column}_baseline"
        cand_col = f"{column}_candidate"
        if base_col in joined:
            records[f"baseline_{column}"] = pd.to_numeric(joined[base_col], errors="coerce")
        if cand_col in joined:
            records[f"candidate_{column}"] = pd.to_numeric(joined[cand_col], errors="coerce")

    for metric in WINDOW_METRICS:
        records[f"delta_{metric}"] = _rounded_delta(
            records.get(f"candidate_{metric}"),
            records.get(f"baseline_{metric}"),
        )
    records["target_rr_peak_band_bpm"] = records["baseline_target_rr_peak_band_bpm"]
    records["delta_pred_rr_peak_band_bpm"] = _rounded_delta(
        records.get("candidate_pred_rr_peak_band_bpm"),
        records.get("baseline_pred_rr_peak_band_bpm"),
    )
    records["delta_abs_best_lag_sec"] = _rounded_delta(
        records["candidate_best_lag_sec"].abs(),
        records["baseline_best_lag_sec"].abs(),
    )

    _add_diagnostic_bins(records)
    return records


def _rounded_delta(candidate: pd.Series | None, baseline: pd.Series | None) -> pd.Series:
    if candidate is None or baseline is None:
        return pd.Series(dtype=float)
    return (candidate - baseline).round(12)


def _add_diagnostic_bins(frame: pd.DataFrame) -> None:
    baseline_peak = pd.to_numeric(frame["baseline_rr_peak_band_abs_error"], errors="coerce")
    baseline_spectrum = pd.to_numeric(frame["baseline_spectrum_similarity"], errors="coerce")
    baseline_count = pd.to_numeric(frame["baseline_breath_count_zero_cross_abs_error"], errors="coerce")
    baseline_lag_abs = pd.to_numeric(frame["baseline_best_lag_sec"], errors="coerce").abs()
    target_rr = pd.to_numeric(frame["target_rr_peak_band_bpm"], errors="coerce")

    frame["baseline_easy"] = baseline_peak <= 0.25
    frame["baseline_hard"] = baseline_peak > 1.0
    spectrum_median = float(baseline_spectrum.median()) if baseline_spectrum.notna().any() else np.nan
    frame["low_spectrum"] = baseline_spectrum <= spectrum_median
    frame["fast_rr"] = target_rr >= 18.0
    frame["baseline_spectrum_tertile"] = _tertile_labels(baseline_spectrum)
    frame["target_rr_bin"] = pd.cut(
        target_rr,
        bins=[0.0, 10.0, 14.0, 18.0, 22.0, np.inf],
        labels=["<10", "10-14", "14-18", "18-22", ">=22"],
        right=False,
    ).astype("string")
    frame["baseline_count_err_bin"] = pd.cut(
        baseline_count,
        bins=[-0.1, 0.0, 1.0, 2.0, np.inf],
        labels=["0", "1", "2", ">2"],
    ).astype("string")
    frame["baseline_lag_abs_bin"] = pd.cut(
        baseline_lag_abs,
        bins=[-0.001, 0.5, 1.5, 3.0, np.inf],
        labels=["<=0.5", "0.5-1.5", "1.5-3", ">3"],
    ).astype("string")
    frame["pred_rr_shift"] = np.select(
        [
            frame["delta_pred_rr_peak_band_bpm"] <= -0.5,
            frame["delta_pred_rr_peak_band_bpm"] >= 0.5,
        ],
        ["down", "up"],
        default="same",
    )
    frame["count_clean"] = baseline_count == 0
    frame["clean_easy_highspec"] = (
        frame["baseline_easy"] & frame["count_clean"] & (frame["baseline_spectrum_tertile"] == "high")
    )
    frame["dirty_easy_lowspec"] = (
        frame["baseline_easy"] & (~frame["count_clean"]) & (frame["baseline_spectrum_tertile"] == "low")
    )
    frame["spectrum_count_cross"] = frame["baseline_spectrum_tertile"].astype(str) + "|" + np.where(
        frame["count_clean"], "count_clean", "count_dirty"
    )


def _tertile_labels(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    labels = pd.Series(["mid"] * len(numeric), index=numeric.index, dtype="string")
    valid = numeric.dropna()
    if len(valid) < 3:
        labels.loc[valid.index] = "high"
        return labels
    ranked = valid.rank(method="first")
    labels.loc[valid.index] = pd.qcut(ranked, 3, labels=["low", "mid", "high"]).astype("string")
    return labels


def _bucket_summary(window_delta: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for label, label_frame in window_delta.groupby("label", dropna=False):
        records.extend(_diagnosis_strata_records(str(label), label_frame))
        for bucket_type, column in (
            ("target_rr_bin", "target_rr_bin"),
            ("baseline_spectrum_tertile", "baseline_spectrum_tertile"),
            ("baseline_count_err_bin", "baseline_count_err_bin"),
            ("baseline_lag_abs_bin", "baseline_lag_abs_bin"),
            ("pred_rr_shift", "pred_rr_shift"),
            ("spectrum_count_cross", "spectrum_count_cross"),
        ):
            for bucket, group in label_frame.groupby(column, dropna=False):
                if group.empty:
                    continue
                records.append(_summary_record(str(label), bucket_type, str(bucket), group))
    return pd.DataFrame.from_records(records)


def _diagnosis_strata_records(label: str, frame: pd.DataFrame) -> list[dict]:
    strata = {
        "overall": frame,
        "baseline_easy": frame[frame["baseline_easy"]],
        "clean_easy_highspec": frame[frame["clean_easy_highspec"]],
        "dirty_easy_lowspec": frame[frame["dirty_easy_lowspec"]],
        "baseline_hard": frame[frame["baseline_hard"]],
        "low_spectrum": frame[frame["low_spectrum"]],
        "fast_rr": frame[frame["fast_rr"]],
        "easy_regressed_gt_0_25": frame[frame["baseline_easy"] & (frame["delta_rr_peak_band_abs_error"] > 0.25)],
        "easy_regressed_gt_0_5": frame[frame["baseline_easy"] & (frame["delta_rr_peak_band_abs_error"] > 0.5)],
        "easy_regressed_gt_1": frame[frame["baseline_easy"] & (frame["delta_rr_peak_band_abs_error"] > 1.0)],
    }
    return [
        _summary_record(label, "diagnosis_stratum", name, stratum)
        for name, stratum in strata.items()
        if not stratum.empty
    ]


def _summary_record(label: str, bucket_type: str, bucket: str, frame: pd.DataFrame) -> dict:
    delta_peak = pd.to_numeric(frame["delta_rr_peak_band_abs_error"], errors="coerce")
    return {
        "label": label,
        "bucket_type": bucket_type,
        "bucket": bucket,
        "n_windows": int(len(frame)),
        "mean_delta_rr_peak_band_abs_error": float(delta_peak.mean()),
        "median_delta_rr_peak_band_abs_error": float(delta_peak.median()),
        "regressed_rate": float((delta_peak > 0).mean()),
        "mean_delta_breath_count_zero_cross_abs_error": float(
            pd.to_numeric(frame["delta_breath_count_zero_cross_abs_error"], errors="coerce").mean()
        ),
        "mean_delta_rr_spec_abs_error": float(
            pd.to_numeric(frame["delta_rr_spec_abs_error"], errors="coerce").mean()
        ),
        "mean_delta_abs_best_lag_sec": float(
            pd.to_numeric(frame["delta_abs_best_lag_sec"], errors="coerce").mean()
        ),
        "mean_delta_pred_rr_peak_band_bpm": float(
            pd.to_numeric(frame["delta_pred_rr_peak_band_bpm"], errors="coerce").mean()
        ),
    }


def _write_outputs(
    output_dir: Path,
    *,
    window_delta: pd.DataFrame,
    bucket_summary: pd.DataFrame,
    top_degraded_easy: pd.DataFrame,
    top_improved_hard: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    window_delta.to_csv(output_dir / "window_delta.csv", index=False)
    bucket_summary.to_csv(output_dir / "bucket_summary.csv", index=False)
    top_degraded_easy.to_csv(output_dir / "top_degraded_easy.csv", index=False)
    top_improved_hard.to_csv(output_dir / "top_improved_hard.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 F-A2 护栏窗口级诊断表")
    parser.add_argument("--manifest", default="runs/f_a2_confidence_guard_manifest.csv")
    parser.add_argument("--candidate-label", action="append", default=None, help="可重复传入；默认诊断 manifest 中所有 F-A 候选")
    parser.add_argument("--output-dir", default="runs/diagnostics/f_a2_guard_windows")
    parser.add_argument("--top-n", type=int, default=50)
    args = parser.parse_args()

    window_delta, bucket_summary, top_degraded_easy, top_improved_hard = diagnose_f_a2_guard_windows(
        args.manifest,
        candidate_labels=args.candidate_label,
        output_dir=args.output_dir,
        top_n=args.top_n,
    )
    print(f"写出 {args.output_dir}/window_delta.csv rows={len(window_delta)}")
    print(f"写出 {args.output_dir}/bucket_summary.csv rows={len(bucket_summary)}")
    print(f"top_degraded_easy rows={len(top_degraded_easy)}")
    print(f"top_improved_hard rows={len(top_improved_hard)}")


if __name__ == "__main__":
    main()
