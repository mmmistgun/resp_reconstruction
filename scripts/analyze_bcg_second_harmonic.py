from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid
from typing import Any

import numpy as np
import pandas as pd
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.analysis.second_harmonic import (
    HarmonicFeatureConfig,
    HarmonicFeatures,
    HarmonicThresholds,
    classify_model_correction,
    extract_harmonic_features,
)
from resp_train.config import load_config
from resp_train.data.factory import build_window_data
from scripts.stratified_eval_analysis import (
    Comparison,
    StratifiedAnalysisSpec,
    run_external_strata_analysis,
)


PROPOSAL_AGREEMENT_BPM = (0.5, 1.0, 2.0)
PROPOSAL_PEAK_TOLERANCES = (0.05, 0.10, 0.15)
PROPOSAL_QUANTILES = (0.75, 0.85, 0.90)
PROPOSAL_CORRECTION_DROPS = (0.10, 0.20, 0.30)
G_SERIES_LABELS = (
    "g0_time_only",
    "g0_f0_native_stft_pre_mixer",
    "g3_c_wide_8p0",
    "g3_c_bandenergy",
)
G_SERIES_SEEDS = (20260700, 20260837, 20260901)
G_SERIES_COMPARISONS = (
    ("f0_vs_time", "g0_f0_native_stft_pre_mixer", "g0_time_only"),
    ("wide_vs_time", "g3_c_wide_8p0", "g0_time_only"),
    ("bandenergy_vs_time", "g3_c_bandenergy", "g0_time_only"),
)


def build_feature_frame(
    dataset: Any,
    *,
    split: str,
    feature_cfg: HarmonicFeatureConfig,
) -> pd.DataFrame:
    rows = dataset.rows.copy().reset_index(drop=True)
    required = {"dataset_row_id", "samp_id", "split", "window_start_s", "source_npz"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"分析数据索引缺少列: {sorted(missing)}")
    if len(rows) != len(dataset):
        raise ValueError(f"dataset.rows 与 dataset 长度不一致: {len(rows)} != {len(dataset)}")
    if rows["dataset_row_id"].duplicated().any():
        duplicates = rows.loc[rows["dataset_row_id"].duplicated(), "dataset_row_id"].tolist()
        raise ValueError(f"分析数据存在重复 dataset_row_id: {duplicates[:8]}")
    actual_splits = set(rows["split"].astype(str))
    if actual_splits != {str(split)}:
        raise ValueError(f"分析数据 split 不一致，预期={split!r}，实际={sorted(actual_splits)}")

    records: list[dict[str, Any]] = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        input_signal = _to_numpy_1d(sample["x"])
        target_signal = _to_numpy_1d(sample["target"])
        features = extract_harmonic_features(input_signal, target_signal, cfg=feature_cfg)
        row = rows.iloc[idx]
        records.append(
            {
                "dataset_row_id": int(row["dataset_row_id"]),
                "samp_id": int(row["samp_id"]),
                "split": str(row["split"]),
                "window_start_s": float(row["window_start_s"]),
                "source_npz": str(row["source_npz"]),
                **asdict(features),
            }
        )
    return pd.DataFrame.from_records(records)


def build_threshold_proposal(
    features: pd.DataFrame,
    *,
    split: str,
    features_path: Path,
) -> dict[str, Any]:
    if str(split) != "val":
        raise ValueError("阈值 proposal 只能由验证集生成")
    if "split" not in features.columns or set(features["split"].astype(str)) != {"val"}:
        raise ValueError("阈值 proposal 的特征必须全部来自验证集")
    required = {
        "tho_robust_rr_bpm",
        "tho_spectral_rr_bpm",
        "tho_reference_hz",
        "harmonic_to_fundamental_ratio",
        "harmonic_band_fraction",
    }
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"验证特征缺少 proposal 所需列: {sorted(missing)}")

    candidates: list[dict[str, Any]] = []
    candidate_index = 0
    robust = pd.to_numeric(features["tho_robust_rr_bpm"], errors="coerce")
    spectral = pd.to_numeric(features["tho_spectral_rr_bpm"], errors="coerce")
    reference_hz = pd.to_numeric(features["tho_reference_hz"], errors="coerce")
    for agreement_bpm in PROPOSAL_AGREEMENT_BPM:
        eligible = (
            robust.notna()
            & spectral.notna()
            & ((robust - spectral).abs() <= agreement_bpm)
            & ((2.0 * reference_hz) <= 0.7)
        )
        eligible_frame = features.loc[eligible]
        if eligible_frame.empty:
            raise ValueError(f"THO 一致性容差 {agreement_bpm} bpm 下没有可判定验证窗口")
        for peak_tolerance in PROPOSAL_PEAK_TOLERANCES:
            for quantile in PROPOSAL_QUANTILES:
                ratio_min = float(
                    pd.to_numeric(
                        eligible_frame["harmonic_to_fundamental_ratio"], errors="coerce"
                    ).quantile(quantile)
                )
                fraction_min = float(
                    pd.to_numeric(
                        eligible_frame["harmonic_band_fraction"], errors="coerce"
                    ).quantile(quantile)
                )
                if not np.isfinite(ratio_min) or not np.isfinite(fraction_min):
                    raise ValueError("验证特征无法产生有限的谐波能量候选阈值")
                for correction_drop in PROPOSAL_CORRECTION_DROPS:
                    candidates.append(
                        {
                            "candidate_id": f"candidate_{candidate_index:03d}",
                            "quantile": quantile,
                            "n_eligible_validation_windows": int(eligible.sum()),
                            "thresholds": {
                                "tho_rr_agreement_bpm": agreement_bpm,
                                "peak_relative_tolerance": peak_tolerance,
                                "harmonic_to_fundamental_min": ratio_min,
                                "harmonic_band_fraction_min": fraction_min,
                                "correction_ratio_drop_min": correction_drop,
                            },
                        }
                    )
                    candidate_index += 1
    return {
        "status": "proposal",
        "schema_version": "bcg-second-harmonic-threshold-proposal-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "val",
        "features_path": str(features_path),
        "n_validation_windows": int(len(features)),
        "candidates": candidates,
    }


def freeze_threshold_candidate(
    proposal_path: Path,
    *,
    candidate_id: str,
    output_path: Path,
    review_note: str,
    allow_identical_existing: bool = False,
) -> Path:
    proposal_path = Path(proposal_path)
    output_path = Path(output_path)
    proposal_bytes = proposal_path.read_bytes()
    proposal = json.loads(proposal_bytes.decode("utf-8"))
    if proposal.get("status") != "proposal" or proposal.get("split") != "val":
        raise ValueError("只能从验证集 proposal 冻结阈值")
    note = str(review_note).strip()
    if not note:
        raise ValueError("freeze 必须记录非空的人工复核备注")
    matched = [item for item in proposal.get("candidates", []) if item.get("candidate_id") == candidate_id]
    if len(matched) != 1:
        raise ValueError(f"proposal 中找不到唯一 candidate_id={candidate_id!r}")
    proposal_sha256 = hashlib.sha256(proposal_bytes).hexdigest()
    payload = {
        "status": "frozen",
        "schema_version": "bcg-second-harmonic-thresholds-v1",
        "version": f"bcg-second-harmonic-v1-{proposal_sha256[:12]}-{candidate_id}",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "proposal_path": str(proposal_path),
        "proposal_sha256": proposal_sha256,
        "candidate_id": candidate_id,
        "review_note": note,
        "thresholds": matched[0]["thresholds"],
    }
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if allow_identical_existing and _same_frozen_selection(existing, payload):
            return output_path
        raise FileExistsError(f"冻结阈值输出已存在，拒绝覆盖: {output_path}")
    _write_json_exclusive(payload, output_path)
    return output_path


def load_frozen_thresholds(path: Path) -> HarmonicThresholds:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("status") != "frozen":
        raise ValueError(f"阈值文件必须处于 frozen 状态: {path}")
    values = payload.get("thresholds")
    if not isinstance(values, dict):
        raise ValueError(f"冻结阈值文件缺少 thresholds: {path}")
    return HarmonicThresholds(version=str(payload.get("version", "")), **values)


def apply_frozen_thresholds(
    features: pd.DataFrame,
    thresholds: HarmonicThresholds,
) -> pd.DataFrame:
    required = {
        "dataset_row_id",
        "samp_id",
        "split",
        "status",
        "peak_second_harmonic_relative_error",
        "harmonic_to_fundamental_ratio",
        "harmonic_band_fraction",
    }
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"测试特征缺少标签列: {sorted(missing)}")
    if features["dataset_row_id"].duplicated().any():
        raise ValueError("测试特征存在重复 dataset_row_id")

    result = features.copy()
    status = result["status"].astype(str)
    eligible = status.eq("eligible")
    peak_error = pd.to_numeric(result["peak_second_harmonic_relative_error"], errors="coerce")
    ratio = pd.to_numeric(result["harmonic_to_fundamental_ratio"], errors="coerce")
    fraction = pd.to_numeric(result["harmonic_band_fraction"], errors="coerce")
    peak = eligible & peak_error.notna() & (peak_error <= thresholds.peak_relative_tolerance)
    prominent = (
        eligible
        & ratio.notna()
        & fraction.notna()
        & (ratio >= thresholds.harmonic_to_fundamental_min)
        & (fraction >= thresholds.harmonic_band_fraction_min)
    )
    result["stratum"] = status
    result.loc[eligible, "stratum"] = "harmonic_negative"
    result.loc[peak & ~prominent, "stratum"] = "peak_doubling"
    result.loc[prominent & ~peak, "stratum"] = "harmonic_prominent"
    result.loc[peak & prominent, "stratum"] = "strong_harmonic"
    positive = {"strong_harmonic", "peak_doubling", "harmonic_prominent"}
    result["harmonic_positive"] = result["stratum"].isin(positive)
    result["threshold_version"] = thresholds.version
    return result


def summarize_coverage(labels: pd.DataFrame) -> pd.DataFrame:
    required = {"dataset_row_id", "samp_id", "status", "stratum", "harmonic_positive"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"测试标签缺少 coverage 列: {sorted(missing)}")
    if labels["dataset_row_id"].duplicated().any():
        raise ValueError("coverage 输入存在重复 dataset_row_id")
    n_all = int(len(labels))
    if n_all == 0:
        raise ValueError("coverage 输入不能为空")
    eligible_mask = labels["status"].eq("eligible")
    n_eligible = int(eligible_mask.sum())
    masks: list[tuple[str, pd.Series]] = [
        ("all_windows", pd.Series(True, index=labels.index)),
        ("tho_reference_unstable", labels["status"].eq("tho_reference_unstable")),
        (
            "second_harmonic_out_of_band",
            labels["status"].eq("second_harmonic_out_of_band"),
        ),
        ("eligible_total", eligible_mask),
        ("strong_harmonic", labels["stratum"].eq("strong_harmonic")),
        ("peak_doubling", labels["stratum"].eq("peak_doubling")),
        ("harmonic_prominent", labels["stratum"].eq("harmonic_prominent")),
        ("harmonic_negative", labels["stratum"].eq("harmonic_negative")),
        ("harmonic_positive_union", labels["harmonic_positive"].astype(bool)),
    ]
    records = []
    eligible_fraction_rows = {
        "eligible_total",
        "strong_harmonic",
        "peak_doubling",
        "harmonic_prominent",
        "harmonic_negative",
        "harmonic_positive_union",
    }
    for name, mask in masks:
        count = int(mask.sum())
        records.append(
            {
                "status": name,
                "n_windows": count,
                "fraction_of_all": count / n_all,
                "fraction_of_eligible": (
                    count / n_eligible
                    if n_eligible > 0 and name in eligible_fraction_rows
                    else float("nan")
                ),
                "n_subjects": int(labels.loc[mask, "samp_id"].nunique()),
            }
        )
    return pd.DataFrame.from_records(records)


def apply_to_split(
    *,
    config_path: Path,
    split: str,
    thresholds_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    if str(split) != "test":
        raise ValueError("冻结阈值 apply 当前只允许 held-out test split")
    thresholds_path = Path(thresholds_path)
    threshold_bytes_before = thresholds_path.read_bytes()
    thresholds_sha256 = hashlib.sha256(threshold_bytes_before).hexdigest()
    thresholds = load_frozen_thresholds(thresholds_path)
    cfg = load_config(config_path)
    test_split = str(cfg.data.get("test_split", "test"))
    data = _build_split_data(cfg, split=test_split)
    feature_cfg = _feature_config_from_cfg(
        cfg,
        tho_rr_agreement_bpm=thresholds.tho_rr_agreement_bpm,
    )
    features = build_feature_frame(data.dataset, split=test_split, feature_cfg=feature_cfg)
    labels = apply_frozen_thresholds(features, thresholds)
    coverage = summarize_coverage(labels)
    output_dir = Path(output_dir)
    labels_path = output_dir / "test_harmonic_labels.csv"
    coverage_path = output_dir / "coverage_summary.csv"
    manifest_path = output_dir / "analysis_manifest.json"
    _write_csv_exclusive(labels, labels_path)
    _write_csv_exclusive(coverage, coverage_path)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "operation": "apply",
        "config": str(config_path),
        "dataset_root": str(cfg.data.dataset_root),
        "index_csv": str(cfg.data.index_csv),
        "split": test_split,
        "n_windows": int(len(labels)),
        "n_subjects": int(labels["samp_id"].nunique()),
        "feature_config": asdict(feature_cfg),
        "thresholds_path": str(thresholds_path),
        "thresholds_sha256": thresholds_sha256,
        "threshold_version": thresholds.version,
        "outputs": {"labels": str(labels_path), "coverage": str(coverage_path)},
    }
    _write_json_exclusive(manifest, manifest_path)
    if thresholds_path.read_bytes() != threshold_bytes_before:
        raise RuntimeError("apply 期间冻结阈值文件发生变化")
    return {"labels": labels_path, "coverage": coverage_path, "manifest": manifest_path}


def summarize_existing_metrics(
    *,
    labels_path: Path,
    eval_root: Path,
    output_dir: Path,
    model_labels: list[str] | tuple[str, ...] = G_SERIES_LABELS,
    seeds: list[int] | tuple[int, ...] = G_SERIES_SEEDS,
    comparisons: list[tuple[str, str, str]] | tuple[tuple[str, str, str], ...] = G_SERIES_COMPARISONS,
    dataset_index: Path | None = None,
) -> dict[str, Path]:
    labels_path = Path(labels_path)
    labels = pd.read_csv(labels_path)
    comparison_specs = [
        Comparison(name=name, target=target, baseline=baseline)
        for name, target, baseline in comparisons
    ]
    spec = StratifiedAnalysisSpec(
        eval_root=Path(eval_root),
        output_dir=Path(output_dir),
        labels=[str(value) for value in model_labels],
        seeds=[int(value) for value in seeds],
        comparisons=comparison_specs,
        dataset_index=dataset_index,
        window_seconds=180.0,
    )
    frames = run_external_strata_analysis(spec, labels)
    output_dir = Path(output_dir)
    outputs = {
        "model_seed": output_dir / "model_stratified_metrics_seed.csv",
        "model_summary": output_dir / "model_stratified_metrics_summary.csv",
        "paired_seed": output_dir / "paired_delta_vs_time_seed.csv",
        "paired_summary": output_dir / "paired_delta_vs_time_summary.csv",
        "manifest": output_dir / "analysis_manifest.json",
    }
    for name in ("model_seed", "model_summary", "paired_seed", "paired_summary"):
        _write_csv_exclusive(frames[name], outputs[name])
    labels_sha256 = hashlib.sha256(labels_path.read_bytes()).hexdigest()
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "operation": "summarize-metrics",
        "labels_path": str(labels_path),
        "labels_sha256": labels_sha256,
        "eval_root": str(eval_root),
        "dataset_index": None if dataset_index is None else str(dataset_index),
        "model_labels": [str(value) for value in model_labels],
        "seeds": [int(value) for value in seeds],
        "comparisons": [
            {"name": item.name, "target": item.target, "baseline": item.baseline}
            for item in comparison_specs
        ],
        "n_fixed_windows": int(len(labels)),
        "outputs": {name: str(path) for name, path in outputs.items() if name != "manifest"},
    }
    _write_json_exclusive(manifest, outputs["manifest"])
    return outputs


def summarize_prediction_corrections(
    *,
    labels_path: Path,
    thresholds: HarmonicThresholds,
    feature_cfg: HarmonicFeatureConfig,
    predictions_dir: Path,
    output_dir: Path,
    expected_runs: list[tuple[str, int]] | tuple[tuple[str, int], ...] | None = None,
) -> dict[str, Path]:
    labels_path = Path(labels_path)
    labels_bytes = labels_path.read_bytes()
    labels_sha256 = hashlib.sha256(labels_bytes).hexdigest()
    labels = pd.read_csv(labels_path)
    required = {
        "dataset_row_id",
        "samp_id",
        "stratum",
        "harmonic_positive",
        "status",
        "tho_reference_hz",
        "tho_robust_rr_bpm",
        "tho_spectral_rr_bpm",
        "bcg_peak_hz",
        "peak_to_tho_ratio",
        "peak_second_harmonic_relative_error",
        "fundamental_energy",
        "second_harmonic_energy",
        "band_energy",
        "harmonic_to_fundamental_ratio",
        "harmonic_band_fraction",
    }
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"测试标签缺少纠正率分析列: {sorted(missing)}")
    if labels["dataset_row_id"].duplicated().any():
        raise ValueError("测试标签存在重复 dataset_row_id")
    positive = labels[labels["harmonic_positive"].astype(bool)].copy()
    if positive.empty:
        raise ValueError("测试标签没有谐波阳性窗口")
    positive_ids = positive["dataset_row_id"].astype(int).tolist()
    positive_lookup = positive.set_index("dataset_row_id", drop=False)

    prediction_dir = Path(predictions_dir)
    manifest_paths = sorted(prediction_dir.glob("*_harmonic_predictions_manifest.json"))
    if not manifest_paths:
        raise FileNotFoundError(f"预测目录没有 harmonic prediction manifest: {prediction_dir}")
    runs: list[tuple[str, int, Path, dict[str, Any]]] = []
    seen_runs: set[tuple[str, int]] = set()
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        label = str(manifest.get("label", ""))
        seed = int(manifest.get("seed"))
        run_key = (label, seed)
        if run_key in seen_runs:
            raise ValueError(f"预测 manifest 存在重复 label/seed: {run_key}")
        seen_runs.add(run_key)
        if manifest.get("labels_sha256") != labels_sha256:
            raise ValueError(f"预测 manifest 标签哈希不一致: {manifest_path}")
        prediction_path = Path(str(manifest.get("output_path", "")))
        if not prediction_path.is_absolute() and not prediction_path.exists():
            prediction_path = prediction_dir / prediction_path.name
        if not prediction_path.exists():
            raise FileNotFoundError(f"预测 NPZ 不存在: {prediction_path}")
        runs.append((label, seed, prediction_path, manifest))
    if expected_runs is None:
        expected_runs = [(label, seed) for label in G_SERIES_LABELS for seed in G_SERIES_SEEDS]
    expected_set = {(str(label), int(seed)) for label, seed in expected_runs}
    if seen_runs != expected_set:
        raise ValueError(
            f"预测 run 集合不完整: missing={sorted(expected_set - seen_runs)} "
            f"extra={sorted(seen_runs - expected_set)}"
        )

    detail_records: list[dict[str, Any]] = []
    for label, seed, prediction_path, _ in sorted(runs, key=lambda item: (item[0], item[1])):
        with np.load(prediction_path, allow_pickle=False) as blob:
            row_ids = np.asarray(blob["dataset_row_id"], dtype=np.int64).reshape(-1)
            predictions = np.asarray(blob["r_tho_hat"], dtype=np.float64)
            targets = np.asarray(blob["tho_ref"], dtype=np.float64)
        if row_ids.tolist() != positive_ids:
            raise ValueError(f"预测 row 顺序与固定阳性标签不一致: label={label} seed={seed}")
        predictions = _prediction_matrix(predictions, name="r_tho_hat")
        targets = _prediction_matrix(targets, name="tho_ref")
        if predictions.shape != targets.shape or predictions.shape[0] != len(row_ids):
            raise ValueError(f"预测/目标/row shape 不一致: label={label} seed={seed}")
        for index, row_id in enumerate(row_ids.tolist()):
            input_row = positive_lookup.loc[int(row_id)]
            input_features = _features_from_row(input_row)
            output_features = extract_harmonic_features(
                predictions[index],
                targets[index],
                cfg=feature_cfg,
            )
            correction_status = classify_model_correction(
                input_features,
                output_features,
                thresholds,
            )
            input_ratio = float(input_features.harmonic_to_fundamental_ratio)
            output_ratio = float(output_features.harmonic_to_fundamental_ratio)
            ratio_drop = (
                (input_ratio - output_ratio) / max(input_ratio, np.finfo(np.float64).eps)
                if np.isfinite(input_ratio) and np.isfinite(output_ratio) and input_ratio > 0
                else float("nan")
            )
            detail_records.append(
                {
                    "label": label,
                    "seed": seed,
                    "dataset_row_id": int(row_id),
                    "samp_id": int(input_row["samp_id"]),
                    "input_stratum": str(input_row["stratum"]),
                    "correction_status": correction_status,
                    "input_peak_to_tho_ratio": input_features.peak_to_tho_ratio,
                    "output_peak_to_tho_ratio": output_features.peak_to_tho_ratio,
                    "input_harmonic_to_fundamental_ratio": input_ratio,
                    "output_harmonic_to_fundamental_ratio": output_ratio,
                    "harmonic_ratio_relative_drop": ratio_drop,
                    "input_harmonic_band_fraction": input_features.harmonic_band_fraction,
                    "output_harmonic_band_fraction": output_features.harmonic_band_fraction,
                }
            )
    detail = pd.DataFrame.from_records(detail_records)
    seed_summary = _correction_seed_summary(detail)
    summary = _aggregate_correction_seed_summary(seed_summary)
    output_dir = Path(output_dir)
    outputs = {
        "detail": output_dir / "model_harmonic_correction.csv",
        "seed_summary": output_dir / "model_harmonic_correction_seed_summary.csv",
        "summary": output_dir / "model_harmonic_correction_summary.csv",
        "manifest": output_dir / "analysis_manifest.json",
    }
    _write_csv_exclusive(detail, outputs["detail"])
    _write_csv_exclusive(seed_summary, outputs["seed_summary"])
    _write_csv_exclusive(summary, outputs["summary"])
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "operation": "summarize-corrections",
        "labels_path": str(labels_path),
        "labels_sha256": labels_sha256,
        "predictions_dir": str(prediction_dir),
        "threshold_version": thresholds.version,
        "feature_config": asdict(feature_cfg),
        "runs": [{"label": label, "seed": seed} for label, seed in sorted(seen_runs)],
        "n_positive_windows": int(len(positive_ids)),
        "outputs": {name: str(path) for name, path in outputs.items() if name != "manifest"},
    }
    _write_json_exclusive(manifest, outputs["manifest"])
    return outputs


def _features_from_row(row: pd.Series) -> HarmonicFeatures:
    return HarmonicFeatures(
        status=str(row["status"]),
        tho_reference_hz=float(row["tho_reference_hz"]),
        tho_robust_rr_bpm=float(row["tho_robust_rr_bpm"]),
        tho_spectral_rr_bpm=float(row["tho_spectral_rr_bpm"]),
        bcg_peak_hz=float(row["bcg_peak_hz"]),
        peak_to_tho_ratio=float(row["peak_to_tho_ratio"]),
        peak_second_harmonic_relative_error=float(row["peak_second_harmonic_relative_error"]),
        fundamental_energy=float(row["fundamental_energy"]),
        second_harmonic_energy=float(row["second_harmonic_energy"]),
        band_energy=float(row["band_energy"]),
        harmonic_to_fundamental_ratio=float(row["harmonic_to_fundamental_ratio"]),
        harmonic_band_fraction=float(row["harmonic_band_fraction"]),
    )


def _prediction_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 3 and array.shape[1] == 1:
        array = array[:, 0, :]
    if array.ndim != 2:
        raise ValueError(f"{name} 必须为 [N,T] 或 [N,1,T]，当前={array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} 包含非有限值")
    return array


def _correction_seed_summary(detail: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (label, seed), run_frame in detail.groupby(["label", "seed"], sort=True):
        strata = ["harmonic_positive_union", *sorted(run_frame["input_stratum"].unique().tolist())]
        for input_stratum in strata:
            frame = (
                run_frame
                if input_stratum == "harmonic_positive_union"
                else run_frame[run_frame["input_stratum"].eq(input_stratum)]
            )
            n_windows = int(len(frame))
            if n_windows == 0:
                continue
            counts = frame["correction_status"].value_counts()
            records.append(
                {
                    "label": label,
                    "seed": int(seed),
                    "input_stratum": input_stratum,
                    "n_windows": n_windows,
                    "n_subjects": int(frame["samp_id"].nunique()),
                    "corrected_fraction": float(counts.get("corrected", 0) / n_windows),
                    "partially_corrected_fraction": float(
                        counts.get("partially_corrected", 0) / n_windows
                    ),
                    "not_corrected_fraction": float(counts.get("not_corrected", 0) / n_windows),
                    "harmonic_ratio_relative_drop_mean": float(
                        pd.to_numeric(frame["harmonic_ratio_relative_drop"], errors="coerce").mean()
                    ),
                    "output_peak_to_tho_ratio_mean": float(
                        pd.to_numeric(frame["output_peak_to_tho_ratio"], errors="coerce").mean()
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def _aggregate_correction_seed_summary(seed_summary: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        column
        for column in seed_summary.columns
        if column not in {"label", "seed", "input_stratum"}
        and pd.api.types.is_numeric_dtype(seed_summary[column])
    ]
    grouped = seed_summary.groupby(["label", "input_stratum"], dropna=False)
    mean = grouped[numeric].mean().reset_index()
    std = grouped[numeric].std(ddof=1).reset_index().rename(
        columns={column: f"{column}_std" for column in numeric}
    )
    return mean.merge(std, on=["label", "input_stratum"], how="left")


def _feature_config_from_labels_manifest(labels_path: Path) -> HarmonicFeatureConfig:
    manifest_path = Path(labels_path).parent / "analysis_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"标签同目录缺少 analysis_manifest.json: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    values = manifest.get("feature_config")
    if not isinstance(values, dict):
        raise ValueError(f"标签 manifest 缺少 feature_config: {manifest_path}")
    return HarmonicFeatureConfig(**values)


def discover(
    *,
    config_path: Path,
    split: str,
    output_dir: Path,
) -> dict[str, Path]:
    if str(split) != "val":
        raise ValueError("discover 只用于验证集阈值开发；测试集请使用 apply")
    cfg = load_config(config_path)
    data = _build_split_data(cfg, split="val")
    feature_cfg = _feature_config_from_cfg(cfg, tho_rr_agreement_bpm=max(PROPOSAL_AGREEMENT_BPM))
    features = build_feature_frame(data.dataset, split="val", feature_cfg=feature_cfg)
    output_dir = Path(output_dir)
    features_path = output_dir / "validation_harmonic_features.csv"
    distribution_path = output_dir / "validation_distribution_summary.csv"
    proposal_path = output_dir / "proposed_harmonic_thresholds.json"
    manifest_path = output_dir / "analysis_manifest.json"
    _write_csv_exclusive(features, features_path)
    distribution = _distribution_summary(features)
    _write_csv_exclusive(distribution, distribution_path)
    proposal = build_threshold_proposal(features, split="val", features_path=features_path)
    _write_json_exclusive(proposal, proposal_path)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "operation": "discover",
        "config": str(config_path),
        "dataset_root": str(cfg.data.dataset_root),
        "index_csv": str(cfg.data.index_csv),
        "split": "val",
        "n_windows": int(len(features)),
        "feature_config": asdict(feature_cfg),
        "outputs": {
            "features": str(features_path),
            "distribution": str(distribution_path),
            "proposal": str(proposal_path),
        },
    }
    _write_json_exclusive(manifest, manifest_path)
    return {
        "features": features_path,
        "distribution": distribution_path,
        "proposal": proposal_path,
        "manifest": manifest_path,
    }


def _build_split_data(cfg, *, split: str):
    prefix = "val" if split == str(cfg.data.get("val_split", "val")) else "test"
    return build_window_data(
        cfg,
        split=split,
        # 阈值开发和正式分层必须覆盖完整 split，不能继承训练 smoke 的窗口上限。
        max_windows=None,
        sample_strategy=str(cfg.data.get(f"{prefix}_sample_strategy", "stratified_random")),
        sample_seed=int(cfg.data.get(f"{prefix}_sample_seed", 0)),
        shuffle=False,
    )


def _feature_config_from_cfg(cfg, *, tho_rr_agreement_bpm: float) -> HarmonicFeatureConfig:
    evaluation = cfg.get("evaluation", {})
    return HarmonicFeatureConfig(
        fs=float(cfg.window.target_fs),
        low_hz=float(cfg.loss.spectrum_low_hz),
        high_hz=float(cfg.loss.spectrum_high_hz),
        filter_order=int(evaluation.get("lag_bandpass_order", 4)),
        welch_nperseg=int(evaluation.get("harmonic_welch_nperseg", 4096)),
        neighborhood_hz=float(evaluation.get("harmonic_neighborhood_hz", 0.025)),
        energy_floor=float(evaluation.get("harmonic_energy_floor", 1e-12)),
        tho_rr_agreement_bpm=float(tho_rr_agreement_bpm),
    )


def _distribution_summary(features: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "tho_robust_rr_bpm",
        "tho_spectral_rr_bpm",
        "peak_to_tho_ratio",
        "peak_second_harmonic_relative_error",
        "harmonic_to_fundamental_ratio",
        "harmonic_band_fraction",
    ]
    records = []
    for column in columns:
        values = pd.to_numeric(features[column], errors="coerce").dropna()
        records.append(
            {
                "metric": column,
                "n": int(len(values)),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "p75": float(values.quantile(0.75)),
                "p85": float(values.quantile(0.85)),
                "p90": float(values.quantile(0.90)),
                "p95": float(values.quantile(0.95)),
            }
        )
    return pd.DataFrame.from_records(records)


def _to_numpy_1d(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _same_frozen_selection(first: dict[str, Any], second: dict[str, Any]) -> bool:
    keys = ("status", "proposal_sha256", "candidate_id", "review_note", "thresholds")
    return all(first.get(key) == second.get(key) for key in keys)


def _write_csv_exclusive(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖: {path}")
    frame.to_csv(path, index=False)


def _write_json_exclusive(payload: dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖: {path}")
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BCG 呼吸带二次谐波显著窗口离线分析")
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover_parser = subparsers.add_parser("discover", help="在验证集生成特征和阈值候选")
    discover_parser.add_argument("--config", type=Path, required=True)
    discover_parser.add_argument("--split", choices=["val"], default="val")
    discover_parser.add_argument("--output-dir", type=Path, required=True)

    freeze_parser = subparsers.add_parser("freeze", help="人工复核后冻结一个候选阈值")
    freeze_parser.add_argument("--proposal", type=Path, required=True)
    freeze_parser.add_argument("--candidate-id", required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)
    freeze_parser.add_argument("--review-note", required=True)
    freeze_parser.add_argument("--allow-identical-existing", action="store_true")

    apply_parser = subparsers.add_parser("apply", help="把冻结阈值应用到 held-out test")
    apply_parser.add_argument("--config", type=Path, required=True)
    apply_parser.add_argument("--split", choices=["test"], default="test")
    apply_parser.add_argument("--thresholds", type=Path, required=True)
    apply_parser.add_argument("--output-dir", type=Path, required=True)

    summary_parser = subparsers.add_parser(
        "summarize-metrics",
        help="按冻结测试标签汇总现有四模型逐窗口 metrics",
    )
    summary_parser.add_argument("--labels", type=Path, required=True)
    summary_parser.add_argument("--eval-root", type=Path, required=True)
    summary_parser.add_argument("--dataset-index", type=Path, default=None)
    summary_parser.add_argument("--output-dir", type=Path, required=True)

    corrections_parser = subparsers.add_parser(
        "summarize-corrections",
        help="计算模型输出对二次谐波的窗口级纠正状态",
    )
    corrections_parser.add_argument("--labels", type=Path, required=True)
    corrections_parser.add_argument("--thresholds", type=Path, required=True)
    corrections_parser.add_argument("--predictions-dir", type=Path, required=True)
    corrections_parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "discover":
        outputs = discover(config_path=args.config, split=args.split, output_dir=args.output_dir)
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return
    if args.command == "freeze":
        output = freeze_threshold_candidate(
            args.proposal,
            candidate_id=args.candidate_id,
            output_path=args.output,
            review_note=args.review_note,
            allow_identical_existing=bool(args.allow_identical_existing),
        )
        print(f"frozen thresholds: {output}")
        return
    if args.command == "apply":
        outputs = apply_to_split(
            config_path=args.config,
            split=args.split,
            thresholds_path=args.thresholds,
            output_dir=args.output_dir,
        )
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return
    if args.command == "summarize-metrics":
        outputs = summarize_existing_metrics(
            labels_path=args.labels,
            eval_root=args.eval_root,
            output_dir=args.output_dir,
            dataset_index=args.dataset_index,
        )
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return
    if args.command == "summarize-corrections":
        thresholds = load_frozen_thresholds(args.thresholds)
        feature_cfg = _feature_config_from_labels_manifest(args.labels)
        outputs = summarize_prediction_corrections(
            labels_path=args.labels,
            thresholds=thresholds,
            feature_cfg=feature_cfg,
            predictions_dir=args.predictions_dir,
            output_dir=args.output_dir,
        )
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return
    raise RuntimeError(f"未知命令: {args.command}")


if __name__ == "__main__":
    main()
