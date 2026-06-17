from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


METRIC_COLUMNS = [
    "rr_spec_abs_error",
    "rr_peak_abs_error",
    "envelope_corr",
    "relative_envelope_corr",
    "relative_envelope_mae",
    "spectrum_similarity",
    "band_limited_corr",
    "best_lag_corr",
    "best_lag_sec",
]


def summarize_runs(runs_root: str | Path, output: str | Path) -> pd.DataFrame:
    """汇总 `runs/tho_small/*` 下每个 run 的核心训练和评价指标。"""
    root = Path(runs_root)
    if not root.exists():
        raise FileNotFoundError(f"runs 目录不存在: {root}")
    records = [_summarize_one_run(run_dir) for run_dir in sorted(path for path in root.iterdir() if path.is_dir())]
    frame = pd.DataFrame.from_records(records)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def _summarize_one_run(run_dir: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"run_id": run_dir.name, "run_dir": str(run_dir)}
    _add_history(record, run_dir / "train_history.csv")
    _add_metrics(record, run_dir / "metrics.csv", prefix="model")
    _add_metrics(record, run_dir / "baseline_metrics.csv", prefix="baseline")
    _add_audit(record, run_dir / "audit.csv")
    return record


def _add_history(record: dict[str, Any], path: Path) -> None:
    if not path.exists():
        record.update({"epochs": 0, "train_loss": np.nan, "val_loss": np.nan})
        return
    frame = pd.read_csv(path)
    if frame.empty:
        record.update({"epochs": 0, "train_loss": np.nan, "val_loss": np.nan})
        return
    last = frame.iloc[-1]
    record["epochs"] = int(last.get("epoch", len(frame)))
    record["train_loss"] = float(last.get("train_loss", np.nan))
    record["val_loss"] = float(last.get("val_loss", np.nan))
    record["best_val_loss"] = float(frame["val_loss"].min()) if "val_loss" in frame.columns else np.nan


def _add_metrics(record: dict[str, Any], path: Path, *, prefix: str) -> None:
    if not path.exists():
        record[f"{prefix}_n_windows"] = 0
        for column in METRIC_COLUMNS:
            record[f"{prefix}_{column}_mean"] = np.nan
            record[f"{prefix}_{column}_median"] = np.nan
        return
    frame = pd.read_csv(path)
    record[f"{prefix}_n_windows"] = int(len(frame))
    if "residual_quality_class" in frame.columns:
        record[f"{prefix}_quality_counts"] = _quality_counts(frame)
    for column in METRIC_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce") if column in frame.columns else pd.Series(dtype=float)
        record[f"{prefix}_{column}_mean"] = float(values.mean()) if not values.empty else np.nan
        record[f"{prefix}_{column}_median"] = float(values.median()) if not values.empty else np.nan


def _add_audit(record: dict[str, Any], path: Path) -> None:
    if not path.exists():
        record["audit_n_windows"] = 0
        record["audit_n_usable"] = 0
        return
    frame = pd.read_csv(path)
    record["audit_n_windows"] = int(frame["n_windows"].sum()) if "n_windows" in frame.columns else 0
    record["audit_n_usable"] = int(frame["n_usable"].sum()) if "n_usable" in frame.columns else 0


def _quality_counts(frame: pd.DataFrame) -> str:
    counts = frame["residual_quality_class"].fillna("").astype(str).value_counts().sort_index()
    return ";".join(f"{key}:{int(value)}" for key, value in counts.items())


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总 THO 小规模训练 runs")
    parser.add_argument("--runs-root", default="runs/tho_small", help="runs 根目录")
    parser.add_argument("--output", default="runs/tho_small/summary.csv", help="汇总 CSV 输出路径")
    args = parser.parse_args()

    frame = summarize_runs(args.runs_root, args.output)
    print(f"写出 run 汇总: {args.output} rows={len(frame)}")


if __name__ == "__main__":
    main()
