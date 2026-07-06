from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.config import load_config
from resp_train.data.factory import build_window_data
from resp_train.engine import collect_predictions
from resp_train.experiments.tho import _resolve_config_path, _validate_checkpoint_config
from resp_train.metrics.evaluate import evaluate_prediction_dict
from resp_train.models.registry import build_model
from resp_train.utils.run import resolve_device


@dataclass(frozen=True)
class TestEvalOutputs:
    metrics: Path
    summary: Path
    manifest: Path


def evaluate_tho_test_checkpoint(
    *,
    checkpoint_path: str | Path,
    config_path: str | Path | None,
    metrics_output_path: str | Path | None,
    summary_output_path: str | Path | None,
    manifest_output_path: str | Path | None,
    overrides: list[str] | None = None,
) -> TestEvalOutputs:
    """固定 checkpoint，在 held-out test split 上评价并保存指标与追踪信息。"""

    resolved_checkpoint = Path(checkpoint_path)
    resolved_config = _resolve_config_path(config_path, resolved_checkpoint)
    cfg = load_config(resolved_config, overrides=overrides)
    output_paths = _resolve_output_paths(
        resolved_checkpoint,
        metrics_output_path=metrics_output_path,
        summary_output_path=summary_output_path,
        manifest_output_path=manifest_output_path,
    )

    device = resolve_device(str(cfg.training.device))
    model = build_model(cfg).to(device)
    checkpoint = torch.load(resolved_checkpoint, map_location=device)
    _validate_checkpoint_config(checkpoint.get("config"), cfg)
    model.load_state_dict(checkpoint["model_state_dict"])

    split = str(cfg.data.get("test_split", "test"))
    max_windows = cfg.data.get("max_test_windows", None)
    sample_strategy = str(cfg.data.get("test_sample_strategy", "stratified_random"))
    sample_seed = int(cfg.data.get("test_sample_seed", cfg.training.get("seed", 0)))
    test_data = build_window_data(
        cfg,
        split=split,
        max_windows=max_windows,
        sample_strategy=sample_strategy,
        sample_seed=sample_seed,
        shuffle=False,
    )
    if len(test_data.dataset) == 0:
        raise RuntimeError(f"test 数据为空，请检查 split={split!r}、input_set 和过滤配置。")

    predictions = collect_predictions(
        model,
        test_data.loader,
        device=device,
        max_windows=len(test_data.dataset),
    )
    metrics = evaluate_prediction_dict(
        predictions,
        cfg,
        method=str(cfg.model.name),
        show_progress=_show_progress(cfg),
    )

    _write_csv(metrics, output_paths.metrics)
    summary = summarize_test_metrics(metrics, split=split, method=str(cfg.model.name))
    _write_csv(summary, output_paths.summary)
    manifest = build_manifest(
        cfg,
        checkpoint_path=resolved_checkpoint,
        config_path=resolved_config,
        outputs=output_paths,
        split=split,
        max_windows=max_windows,
        sample_strategy=sample_strategy,
        sample_seed=sample_seed,
        n_windows=len(metrics),
        device=str(device),
    )
    _write_csv(manifest, output_paths.manifest)
    return output_paths


def summarize_test_metrics(metrics: pd.DataFrame, *, split: str, method: str) -> pd.DataFrame:
    record: dict[str, Any] = {
        "split": split,
        "method": method,
        "n_windows": int(len(metrics)),
    }
    for column in (
        "rr_peak_band_abs_error",
        "rr_peak_band_robust_abs_error",
        "rr_spec_abs_error",
        "breath_count_zero_cross_abs_error",
        "relative_envelope_mae",
        "relative_envelope_corr",
        "spectrum_similarity",
        "band_limited_corr",
        "best_lag_corr",
        "best_lag_sec",
    ):
        if column not in metrics:
            continue
        values = pd.to_numeric(metrics[column], errors="coerce").dropna()
        if values.empty:
            record[f"{column}_mean"] = float("nan")
            record[f"{column}_median"] = float("nan")
            continue
        record[f"{column}_mean"] = float(values.mean())
        record[f"{column}_median"] = float(values.median())
        if column.endswith("_abs_error"):
            record[f"{column}_p95"] = float(values.quantile(0.95))
            record[f"{column}_frac_gt_1"] = float((values.to_numpy() > 1.0).mean())
    return pd.DataFrame([record])


def build_manifest(
    cfg,
    *,
    checkpoint_path: Path,
    config_path: Path,
    outputs: TestEvalOutputs,
    split: str,
    max_windows: Any,
    sample_strategy: str,
    sample_seed: int,
    n_windows: int,
    device: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "checkpoint": str(checkpoint_path),
                "config": str(config_path),
                "dataset_root": str(cfg.data.dataset_root),
                "index_csv": str(cfg.data.index_csv),
                "input_set": str(cfg.data.input_set),
                "split": split,
                "max_windows": "" if max_windows is None else int(max_windows),
                "sample_strategy": sample_strategy,
                "sample_seed": int(sample_seed),
                "n_windows": int(n_windows),
                "model": str(cfg.model.name),
                "device": device,
                "metrics_output": str(outputs.metrics),
                "summary_output": str(outputs.summary),
            }
        ]
    )


def _resolve_output_paths(
    checkpoint_path: Path,
    *,
    metrics_output_path: str | Path | None,
    summary_output_path: str | Path | None,
    manifest_output_path: str | Path | None,
) -> TestEvalOutputs:
    run_dir = checkpoint_path.parent
    return TestEvalOutputs(
        metrics=Path(metrics_output_path) if metrics_output_path else run_dir / "test_metrics.csv",
        summary=Path(summary_output_path) if summary_output_path else run_dir / "test_summary.csv",
        manifest=Path(manifest_output_path) if manifest_output_path else run_dir / "test_eval_manifest.csv",
    )


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _show_progress(cfg) -> bool | None:
    value = cfg.training.get("show_progress", None)
    if value in (None, "auto"):
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
        raise ValueError(f"training.show_progress 只能是 true/false/auto，当前为: {value}")
    return bool(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="固定 checkpoint，在 THO test split 上生成最终评价指标")
    parser.add_argument("--config", default="", help="配置文件路径；为空时优先使用 checkpoint 同目录 config.yaml")
    parser.add_argument("--checkpoint", required=True, help="训练产生的 checkpoint.pt 或 checkpoint_topN.pt")
    parser.add_argument("--metrics-output", default="", help="逐窗口指标 CSV；默认写入 checkpoint 同目录 test_metrics.csv")
    parser.add_argument("--summary-output", default="", help="汇总指标 CSV；默认写入 checkpoint 同目录 test_summary.csv")
    parser.add_argument("--manifest-output", default="", help="评价 manifest CSV；默认写入 checkpoint 同目录 test_eval_manifest.csv")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="OmegaConf dotlist 覆盖，可重复传入")
    args = parser.parse_args()

    outputs = evaluate_tho_test_checkpoint(
        checkpoint_path=args.checkpoint,
        config_path=args.config or None,
        metrics_output_path=args.metrics_output or None,
        summary_output_path=args.summary_output or None,
        manifest_output_path=args.manifest_output or None,
        overrides=args.overrides,
    )
    print(f"写出 test metrics: {outputs.metrics}")
    print(f"写出 test summary: {outputs.summary}")
    print(f"写出 test manifest: {outputs.manifest}")


if __name__ == "__main__":
    main()
