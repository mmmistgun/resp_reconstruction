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
    HarmonicThresholds,
    extract_harmonic_features,
)
from resp_train.config import load_config
from resp_train.data.factory import build_window_data


PROPOSAL_AGREEMENT_BPM = (0.5, 1.0, 2.0)
PROPOSAL_PEAK_TOLERANCES = (0.05, 0.10, 0.15)
PROPOSAL_QUANTILES = (0.75, 0.85, 0.90)
PROPOSAL_CORRECTION_DROPS = (0.10, 0.20, 0.30)


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
    raise RuntimeError(f"未知命令: {args.command}")


if __name__ == "__main__":
    main()
