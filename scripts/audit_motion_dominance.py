from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/resp_reconstruction_matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.metrics.signal import motion_dominance_metrics
from scripts.plot_tho_predictions import _load_config, _load_dataset_for_rows


def audit_run_motion_dominance(
    run_dir: str | Path,
    *,
    output: str | Path | None = None,
    row_ids: list[int] | None = None,
    max_rows: int | None = None,
    sort_by: str = "rr_peak_band_abs_error",
) -> pd.DataFrame:
    """按 run 配置读取窗口，并审计 input/target 是否被短时极端事件支配。"""
    run_path = Path(run_dir)
    cfg = _load_config(run_path)
    metrics = _load_metrics_if_exists(run_path)
    selected_row_ids = _select_row_ids(metrics, row_ids=row_ids, max_rows=max_rows, sort_by=sort_by)
    dataset = _load_dataset_for_rows(cfg, selected_row_ids)
    records: list[dict[str, Any]] = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        meta = sample.get("meta", {})
        dataset_row_id = int(meta.get("dataset_row_id", selected_row_ids[idx]))
        x = sample["x"].detach().cpu().numpy().reshape(-1)
        target = sample["target"].detach().cpu().numpy().reshape(-1)
        input_motion = motion_dominance_metrics(x)
        target_motion = motion_dominance_metrics(target)
        records.append(
            {
                "dataset_row_id": dataset_row_id,
                "split": str(meta.get("split", "")),
                "samp_id": meta.get("samp_id", ""),
                "coupling_state_id": meta.get("coupling_state_id", ""),
                "recommended_scope": _recommended_scope(
                    input_dominated=bool(input_motion["motion_dominated"]),
                    target_dominated=bool(target_motion["motion_dominated"]),
                ),
                **_prefix_metrics("input", input_motion),
                **_prefix_metrics("target", target_motion),
            }
        )
    frame = pd.DataFrame.from_records(records)
    if metrics is not None and not metrics.empty:
        metric_cols = [
            col
            for col in (
                "dataset_row_id",
                "rr_peak_band_abs_error",
                "rr_spec_abs_error",
                "rr_peak_abs_error",
                "band_limited_corr",
                "best_lag_corr",
                "relative_envelope_corr",
                "relative_envelope_mae",
            )
            if col in metrics.columns
        ]
        frame = frame.merge(metrics[metric_cols], on="dataset_row_id", how="left")
    output_path = Path(output) if output is not None else run_path / "motion_dominance.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def _load_metrics_if_exists(run_path: Path) -> pd.DataFrame | None:
    metrics_path = run_path / "metrics.csv"
    if not metrics_path.exists():
        return None
    return pd.read_csv(metrics_path)


def _select_row_ids(
    metrics: pd.DataFrame | None,
    *,
    row_ids: list[int] | None,
    max_rows: int | None,
    sort_by: str,
) -> list[int]:
    if row_ids:
        selected = [int(row_id) for row_id in row_ids]
    elif metrics is not None and "dataset_row_id" in metrics.columns:
        ranked = metrics.copy()
        if sort_by in ranked.columns:
            ranked = ranked.sort_values(sort_by, ascending=False, na_position="last")
        selected = [int(row_id) for row_id in ranked["dataset_row_id"].tolist()]
    else:
        raise ValueError("未提供 row_id，且 run_dir 下没有可用 metrics.csv")
    if max_rows is not None:
        selected = selected[: int(max_rows)]
    if not selected:
        raise ValueError("没有可审计的 dataset_row_id")
    return selected


def _prefix_metrics(prefix: str, metrics: dict[str, float | bool]) -> dict[str, float | bool]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _recommended_scope(*, input_dominated: bool, target_dominated: bool) -> str:
    if input_dominated and target_dominated:
        return "dataset_quality_flag"
    if target_dominated:
        return "dataset_label_review"
    if input_dominated:
        return "task_input_robustness"
    return "task_regular"


def _print_summary(frame: pd.DataFrame, output: Path) -> None:
    print(f"wrote: {output}")
    print(f"rows: {len(frame)}")
    if "recommended_scope" in frame.columns:
        print("recommended_scope:")
        print(frame["recommended_scope"].value_counts(dropna=False).to_string())
    for column in ("input_motion_dominated", "target_motion_dominated"):
        if column in frame.columns:
            print(f"{column}: {int(frame[column].sum())}/{len(frame)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="审计 THO 窗口是否被短时极端体动支配")
    parser.add_argument("--run-dir", required=True, help="训练 run 目录，读取其中 config.yaml 和 metrics.csv")
    parser.add_argument("--output", default="", help="输出 CSV，默认 <run-dir>/motion_dominance.csv")
    parser.add_argument("--row-id", action="append", type=int, default=[], help="只审计指定 dataset_row_id，可重复传入")
    parser.add_argument("--max-rows", type=int, default=0, help="最多审计多少行；0 表示不限制")
    parser.add_argument("--sort-by", default="rr_peak_band_abs_error", help="未指定 row-id 时按 metrics.csv 哪列排序")
    args = parser.parse_args()

    output = Path(args.output) if args.output else Path(args.run_dir) / "motion_dominance.csv"
    frame = audit_run_motion_dominance(
        args.run_dir,
        output=output,
        row_ids=args.row_id or None,
        max_rows=args.max_rows or None,
        sort_by=args.sort_by,
    )
    _print_summary(frame, output)


if __name__ == "__main__":
    main()
