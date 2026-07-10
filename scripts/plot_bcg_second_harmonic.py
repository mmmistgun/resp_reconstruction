from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/resp_reconstruction_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.config import load_config
from resp_train.metrics.signal import bandpass_filter
from resp_train.analysis.second_harmonic import HarmonicThresholds
from scripts.analyze_bcg_second_harmonic import (
    G_SERIES_LABELS,
    _build_split_data,
    load_frozen_thresholds,
)


def select_review_cases(
    features: pd.DataFrame,
    proposal: dict[str, Any],
    *,
    candidate_id: str,
    cases_per_group: int,
) -> pd.DataFrame:
    count = int(cases_per_group)
    if count <= 0:
        raise ValueError("cases_per_group 必须为正数")
    candidate = _proposal_candidate(proposal, candidate_id)
    thresholds = candidate["thresholds"]
    required = {
        "dataset_row_id",
        "status",
        "tho_robust_rr_bpm",
        "tho_spectral_rr_bpm",
        "tho_reference_hz",
        "peak_second_harmonic_relative_error",
        "harmonic_to_fundamental_ratio",
        "harmonic_band_fraction",
    }
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"验证特征缺少复核抽样列: {sorted(missing)}")
    if features["dataset_row_id"].duplicated().any():
        raise ValueError("验证特征存在重复 dataset_row_id")

    frame = features.copy()
    robust = pd.to_numeric(frame["tho_robust_rr_bpm"], errors="coerce")
    spectral = pd.to_numeric(frame["tho_spectral_rr_bpm"], errors="coerce")
    reference_hz = pd.to_numeric(frame["tho_reference_hz"], errors="coerce")
    eligible = (
        frame["status"].eq("eligible")
        & ((robust - spectral).abs() <= float(thresholds["tho_rr_agreement_bpm"]))
        & ((2.0 * reference_hz) <= 0.7)
    )
    frame = frame.loc[eligible].copy()
    if len(frame) < 3 * count:
        raise ValueError(f"可复核窗口不足：需要至少 {3 * count}，实际 {len(frame)}")

    ratio_scale = max(float(thresholds["harmonic_to_fundamental_min"]), np.finfo(float).eps)
    fraction_scale = max(float(thresholds["harmonic_band_fraction_min"]), np.finfo(float).eps)
    peak_scale = max(float(thresholds["peak_relative_tolerance"]), np.finfo(float).eps)
    frame["_harmonic_score"] = 0.5 * (
        pd.to_numeric(frame["harmonic_to_fundamental_ratio"], errors="coerce") / ratio_scale
        + pd.to_numeric(frame["harmonic_band_fraction"], errors="coerce") / fraction_scale
    )
    frame["_boundary_distance"] = (
        (
            pd.to_numeric(frame["harmonic_to_fundamental_ratio"], errors="coerce") / ratio_scale
            - 1.0
        ).abs()
        + (
            pd.to_numeric(frame["harmonic_band_fraction"], errors="coerce") / fraction_scale
            - 1.0
        ).abs()
        + (
            pd.to_numeric(frame["peak_second_harmonic_relative_error"], errors="coerce")
            / peak_scale
            - 1.0
        ).abs()
    )

    selected_parts: list[pd.DataFrame] = []
    used: set[int] = set()
    group_orders = (
        ("harmonic_high", "_harmonic_score", False),
        ("threshold_boundary", "_boundary_distance", True),
        ("harmonic_low", "_harmonic_score", True),
    )
    for group, column, ascending in group_orders:
        available = frame.loc[~frame["dataset_row_id"].astype(int).isin(used)]
        chosen = available.sort_values(
            [column, "dataset_row_id"],
            ascending=[ascending, True],
            na_position="last",
        ).head(count).copy()
        if len(chosen) != count:
            raise ValueError(f"复核组 {group} 无法选满 {count} 个不重复窗口")
        chosen["review_group"] = group
        selected_parts.append(chosen)
        used.update(chosen["dataset_row_id"].astype(int).tolist())
    return pd.concat(selected_parts, ignore_index=True).drop(
        columns=["_harmonic_score", "_boundary_distance"]
    )


def plot_validation_review(
    features: pd.DataFrame,
    proposal: dict[str, Any],
    *,
    signal_lookup: dict[int, dict[str, np.ndarray]],
    output_dir: Path,
    candidate_id: str,
    cases_per_group: int,
    fs: float,
    low_hz: float,
    high_hz: float,
    filter_order: int,
) -> pd.DataFrame:
    selected = select_review_cases(
        features,
        proposal,
        candidate_id=candidate_id,
        cases_per_group=cases_per_group,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        row_id = int(row["dataset_row_id"])
        if row_id not in signal_lookup:
            raise ValueError(f"signal_lookup 缺少 dataset_row_id={row_id}")
        signals = signal_lookup[row_id]
        figure_path = output_dir / f"{row['review_group']}_row_{row_id}.png"
        if figure_path.exists():
            raise FileExistsError(f"复核图已存在，拒绝覆盖: {figure_path}")
        _plot_validation_case(
            figure_path,
            bcg=np.asarray(signals["bcg"], dtype=np.float64).reshape(-1),
            tho=np.asarray(signals["tho"], dtype=np.float64).reshape(-1),
            feature_row=row,
            fs=float(fs),
            low_hz=float(low_hz),
            high_hz=float(high_hz),
            filter_order=int(filter_order),
        )
        records.append(
            {
                "dataset_row_id": row_id,
                "samp_id": int(row["samp_id"]),
                "review_group": str(row["review_group"]),
                "candidate_id": candidate_id,
                "figure_path": str(figure_path),
            }
        )
    manifest = pd.DataFrame.from_records(records)
    manifest_path = output_dir / "review_case_manifest.csv"
    if manifest_path.exists():
        raise FileExistsError(f"复核 manifest 已存在，拒绝覆盖: {manifest_path}")
    manifest.to_csv(manifest_path, index=False)
    return manifest


def build_signal_lookup(dataset: Any, row_ids: list[int]) -> dict[int, dict[str, np.ndarray]]:
    requested = [int(value) for value in row_ids]
    if len(requested) != len(set(requested)):
        raise ValueError("待加载 row_ids 存在重复")
    positions = {
        int(row_id): idx for idx, row_id in enumerate(dataset.rows["dataset_row_id"].astype(int).tolist())
    }
    missing = [row_id for row_id in requested if row_id not in positions]
    if missing:
        raise ValueError(f"验证 dataset 缺少待绘制 row: {missing}")
    lookup: dict[int, dict[str, np.ndarray]] = {}
    for row_id in requested:
        sample = dataset[positions[row_id]]
        lookup[row_id] = {
            "bcg": _to_numpy_1d(sample["x"]),
            "tho": _to_numpy_1d(sample["target"]),
        }
    return lookup


def select_model_case_rows(
    corrections: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    thresholds: HarmonicThresholds,
    seed: int,
    model_labels: list[str] | tuple[str, ...],
    max_cases: int,
) -> pd.DataFrame:
    limit = int(max_cases)
    if limit <= 0:
        raise ValueError("max_cases 必须为正数")
    required_corrections = {"label", "seed", "dataset_row_id", "correction_status"}
    missing = required_corrections - set(corrections.columns)
    if missing:
        raise ValueError(f"纠正率明细缺少案例选择列: {sorted(missing)}")
    required_labels = {
        "dataset_row_id",
        "samp_id",
        "peak_second_harmonic_relative_error",
        "harmonic_to_fundamental_ratio",
        "harmonic_band_fraction",
    }
    missing = required_labels - set(labels.columns)
    if missing:
        raise ValueError(f"谐波标签缺少案例选择列: {sorted(missing)}")
    if labels["dataset_row_id"].duplicated().any():
        raise ValueError("谐波标签存在重复 dataset_row_id")

    model_labels = [str(value) for value in model_labels]
    frame = corrections[
        corrections["seed"].astype(int).eq(int(seed))
        & corrections["label"].astype(str).isin(model_labels)
    ].copy()
    if frame.empty:
        raise ValueError(f"纠正率明细没有 seed={seed} 的目标模型")
    if frame[["dataset_row_id", "label"]].duplicated().any():
        raise ValueError("纠正率明细存在重复 dataset_row_id/label")
    pivot = frame.pivot(index="dataset_row_id", columns="label", values="correction_status")
    missing_models = [label for label in model_labels if label not in pivot.columns]
    if missing_models:
        raise ValueError(f"纠正率明细缺少模型: {missing_models}")
    pivot = pivot[model_labels].dropna()
    candidates = pivot.reset_index().merge(labels, on="dataset_row_id", how="left", validate="one_to_one")
    if candidates["samp_id"].isna().any():
        raise ValueError("纠正率案例无法回连谐波标签")
    candidates["seed"] = int(seed)
    candidates["_all_corrected"] = pivot.eq("corrected").all(axis=1).to_numpy()
    candidates["_all_not_corrected"] = pivot.eq("not_corrected").all(axis=1).to_numpy()
    candidates["_status_nunique"] = pivot.nunique(axis=1).to_numpy()
    candidates["_boundary_distance"] = (
        (
            pd.to_numeric(candidates["peak_second_harmonic_relative_error"], errors="coerce")
            / max(thresholds.peak_relative_tolerance, np.finfo(float).eps)
            - 1.0
        ).abs()
        + (
            pd.to_numeric(candidates["harmonic_to_fundamental_ratio"], errors="coerce")
            / max(thresholds.harmonic_to_fundamental_min, np.finfo(float).eps)
            - 1.0
        ).abs()
        + (
            pd.to_numeric(candidates["harmonic_band_fraction"], errors="coerce")
            / max(thresholds.harmonic_band_fraction_min, np.finfo(float).eps)
            - 1.0
        ).abs()
    )

    selected: list[pd.Series] = []
    used: set[int] = set()

    def choose(category: str, mask: pd.Series, sort_columns: list[str], ascending: list[bool]) -> None:
        if len(selected) >= limit:
            return
        available = candidates.loc[mask & ~candidates["dataset_row_id"].astype(int).isin(used)]
        if available.empty:
            return
        row = available.sort_values(sort_columns, ascending=ascending).iloc[0].copy()
        row["case_category"] = category
        selected.append(row)
        used.add(int(row["dataset_row_id"]))

    choose("all_corrected", candidates["_all_corrected"], ["dataset_row_id"], [True])
    choose("all_not_corrected", candidates["_all_not_corrected"], ["dataset_row_id"], [True])
    choose(
        "model_disagreement",
        candidates["_status_nunique"] > 1,
        ["_status_nunique", "dataset_row_id"],
        [False, True],
    )
    choose(
        "threshold_boundary",
        pd.Series(True, index=candidates.index),
        ["_boundary_distance", "dataset_row_id"],
        [True, True],
    )
    remaining = candidates.loc[~candidates["dataset_row_id"].astype(int).isin(used)].sort_values(
        ["_boundary_distance", "dataset_row_id"]
    )
    for _, row in remaining.iterrows():
        if len(selected) >= limit:
            break
        row = row.copy()
        row["case_category"] = "additional_boundary"
        selected.append(row)
    if not selected:
        raise ValueError("没有可绘制的模型案例")
    return pd.DataFrame(selected).drop(
        columns=["_all_corrected", "_all_not_corrected", "_status_nunique", "_boundary_distance"]
    )


def plot_model_cases(
    selected: pd.DataFrame,
    *,
    input_lookup: dict[int, dict[str, np.ndarray]],
    prediction_lookup: dict[tuple[str, int, int], np.ndarray],
    output_dir: Path,
    model_labels: list[str] | tuple[str, ...],
    fs: float,
    low_hz: float,
    high_hz: float,
    filter_order: int,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        row_id = int(row["dataset_row_id"])
        seed = int(row["seed"])
        if row_id not in input_lookup:
            raise ValueError(f"输入 signal lookup 缺少 row={row_id}")
        predictions: dict[str, np.ndarray] = {}
        for label in model_labels:
            key = (str(label), seed, row_id)
            if key not in prediction_lookup:
                raise ValueError(f"预测 lookup 缺少 {key}")
            predictions[str(label)] = np.asarray(prediction_lookup[key], dtype=np.float64).reshape(-1)
        figure_path = output_dir / f"{row['case_category']}_seed_{seed}_row_{row_id}.png"
        if figure_path.exists():
            raise FileExistsError(f"模型案例图已存在，拒绝覆盖: {figure_path}")
        signals = input_lookup[row_id]
        _plot_model_case(
            figure_path,
            bcg=np.asarray(signals["bcg"], dtype=np.float64).reshape(-1),
            tho=np.asarray(signals["tho"], dtype=np.float64).reshape(-1),
            predictions=predictions,
            fs=float(fs),
            low_hz=float(low_hz),
            high_hz=float(high_hz),
            filter_order=int(filter_order),
            title=f"{row['case_category']} seed={seed} row={row_id} subject={int(row['samp_id'])}",
        )
        records.append(
            {
                "dataset_row_id": row_id,
                "samp_id": int(row["samp_id"]),
                "seed": seed,
                "case_category": str(row["case_category"]),
                "figure_path": str(figure_path),
            }
        )
    manifest = pd.DataFrame.from_records(records)
    manifest_path = output_dir / "model_case_manifest.csv"
    if manifest_path.exists():
        raise FileExistsError(f"模型案例 manifest 已存在，拒绝覆盖: {manifest_path}")
    manifest.to_csv(manifest_path, index=False)
    return manifest


def load_model_prediction_lookup(
    predictions_dir: Path,
    *,
    model_labels: list[str] | tuple[str, ...],
    seed: int,
    row_ids: list[int],
) -> dict[tuple[str, int, int], np.ndarray]:
    requested = [int(value) for value in row_ids]
    lookup: dict[tuple[str, int, int], np.ndarray] = {}
    for label in model_labels:
        path = Path(predictions_dir) / f"{label}_{seed}_harmonic_predictions.npz"
        if not path.exists():
            raise FileNotFoundError(f"模型案例缺少预测文件: {path}")
        with np.load(path, allow_pickle=False) as blob:
            ids = np.asarray(blob["dataset_row_id"], dtype=np.int64).reshape(-1)
            predictions = np.asarray(blob["r_tho_hat"], dtype=np.float64)
        if predictions.ndim == 3 and predictions.shape[1] == 1:
            predictions = predictions[:, 0, :]
        positions = {int(row_id): index for index, row_id in enumerate(ids.tolist())}
        missing = [row_id for row_id in requested if row_id not in positions]
        if missing:
            raise ValueError(f"预测文件 {path} 缺少案例 row: {missing}")
        for row_id in requested:
            lookup[(str(label), int(seed), row_id)] = predictions[positions[row_id]].reshape(-1)
    return lookup


def _plot_validation_case(
    output: Path,
    *,
    bcg: np.ndarray,
    tho: np.ndarray,
    feature_row: pd.Series,
    fs: float,
    low_hz: float,
    high_hz: float,
    filter_order: int,
) -> None:
    if bcg.shape != tho.shape:
        raise ValueError(f"BCG/THO 绘图长度不一致: {bcg.shape} != {tho.shape}")
    bcg_band = bandpass_filter(
        bcg,
        fs=fs,
        low_hz=low_hz,
        high_hz=high_hz,
        order=filter_order,
    )
    tho_band = bandpass_filter(
        tho,
        fs=fs,
        low_hz=low_hz,
        high_hz=high_hz,
        order=filter_order,
    )
    time = np.arange(bcg.size) / fs
    freqs, bcg_power = scipy_signal.welch(bcg_band, fs=fs, nperseg=min(4096, bcg.size))
    _, tho_power = scipy_signal.welch(tho_band, fs=fs, nperseg=min(4096, tho.size))
    band_mask = (freqs >= low_hz) & (freqs <= high_hz)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot(time, _display_scale(tho_band), label="THO resp-band", linewidth=1.0)
    axes[0].plot(time, _display_scale(bcg_band), label="BCG resp-band", linewidth=0.8, alpha=0.8)
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("robust display scale")
    axes[0].grid(alpha=0.2)
    axes[0].legend(loc="upper right")

    axes[1].plot(freqs[band_mask], tho_power[band_mask], label="THO PSD", linewidth=1.0)
    axes[1].plot(freqs[band_mask], bcg_power[band_mask], label="BCG PSD", linewidth=1.0)
    f0 = float(feature_row["tho_reference_hz"])
    axes[1].axvline(f0, color="#2ca02c", linestyle="--", label="THO f0")
    axes[1].axvline(2.0 * f0, color="#d62728", linestyle="--", label="THO 2f0")
    axes[1].set_xlim(low_hz, high_hz)
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_ylabel("PSD")
    axes[1].grid(alpha=0.2)
    axes[1].legend(loc="upper right")

    fig.suptitle(
        " ".join(
            [
                f"row={int(feature_row['dataset_row_id'])}",
                f"subject={int(feature_row['samp_id'])}",
                f"THO robust/spec={float(feature_row['tho_robust_rr_bpm']):.2f}/"
                f"{float(feature_row['tho_spectral_rr_bpm']):.2f} bpm",
                f"peak ratio={float(feature_row['peak_to_tho_ratio']):.3f}",
                f"E2/E1={float(feature_row['harmonic_to_fundamental_ratio']):.3f}",
                f"E2/Eband={float(feature_row['harmonic_band_fraction']):.3f}",
            ]
        ),
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_model_case(
    output: Path,
    *,
    bcg: np.ndarray,
    tho: np.ndarray,
    predictions: dict[str, np.ndarray],
    fs: float,
    low_hz: float,
    high_hz: float,
    filter_order: int,
    title: str,
) -> None:
    lengths = {bcg.size, tho.size, *(value.size for value in predictions.values())}
    if len(lengths) != 1:
        raise ValueError(f"模型案例信号长度不一致: {sorted(lengths)}")
    filtered = {
        "BCG input": bandpass_filter(
            bcg, fs=fs, low_hz=low_hz, high_hz=high_hz, order=filter_order
        ),
        "THO target": bandpass_filter(
            tho, fs=fs, low_hz=low_hz, high_hz=high_hz, order=filter_order
        ),
    }
    for label, values in predictions.items():
        filtered[label] = bandpass_filter(
            values, fs=fs, low_hz=low_hz, high_hz=high_hz, order=filter_order
        )
    time = np.arange(bcg.size) / fs
    fig, axes = plt.subplots(2, 1, figsize=(15, 9))
    for label, values in filtered.items():
        linewidth = 1.2 if label == "THO target" else 0.8
        alpha = 1.0 if label == "THO target" else 0.8
        axes[0].plot(time, _display_scale(values), label=label, linewidth=linewidth, alpha=alpha)
        freqs, power = scipy_signal.welch(values, fs=fs, nperseg=min(4096, values.size))
        mask = (freqs >= low_hz) & (freqs <= high_hz)
        normalized = power[mask] / max(float(np.max(power[mask])), np.finfo(float).eps)
        axes[1].plot(freqs[mask], normalized, label=label, linewidth=linewidth, alpha=alpha)
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("robust display scale")
    axes[0].grid(alpha=0.2)
    axes[0].legend(loc="upper right", ncol=2)
    axes[1].set_xlim(low_hz, high_hz)
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_ylabel("normalized PSD")
    axes[1].grid(alpha=0.2)
    axes[1].legend(loc="upper right", ncol=2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _proposal_candidate(proposal: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    if proposal.get("status") != "proposal":
        raise ValueError("复核图必须使用 proposal 阈值文件")
    matched = [item for item in proposal.get("candidates", []) if item.get("candidate_id") == candidate_id]
    if len(matched) != 1:
        raise ValueError(f"proposal 中找不到唯一 candidate_id={candidate_id!r}")
    return matched[0]


def _display_scale(signal: np.ndarray) -> np.ndarray:
    center = float(np.median(signal))
    scale = float(np.percentile(np.abs(signal - center), 95.0))
    return (signal - center) / max(scale, np.finfo(float).eps)


def _to_numpy_1d(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制 BCG 二次谐波阈值复核图")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validation = subparsers.add_parser("validation-review")
    validation.add_argument("--config", type=Path, required=True)
    validation.add_argument("--features", type=Path, required=True)
    validation.add_argument("--proposal", type=Path, required=True)
    validation.add_argument("--output-dir", type=Path, required=True)
    validation.add_argument("--candidate-id", default="candidate_040")
    validation.add_argument("--cases-per-group", type=int, default=6)

    model_cases = subparsers.add_parser("model-cases")
    model_cases.add_argument("--config", type=Path, required=True)
    model_cases.add_argument("--labels", type=Path, required=True)
    model_cases.add_argument("--thresholds", type=Path, required=True)
    model_cases.add_argument("--corrections", type=Path, required=True)
    model_cases.add_argument("--predictions-dir", type=Path, required=True)
    model_cases.add_argument("--output-dir", type=Path, required=True)
    model_cases.add_argument("--seed", type=int, default=20260837)
    model_cases.add_argument("--max-cases", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    evaluation = cfg.get("evaluation", {})
    if args.command == "validation-review":
        features = pd.read_csv(args.features)
        proposal = json.loads(args.proposal.read_text(encoding="utf-8"))
        selected = select_review_cases(
            features,
            proposal,
            candidate_id=args.candidate_id,
            cases_per_group=args.cases_per_group,
        )
        data = _build_split_data(cfg, split="val")
        signal_lookup = build_signal_lookup(
            data.dataset, selected["dataset_row_id"].astype(int).tolist()
        )
        manifest = plot_validation_review(
            features,
            proposal,
            signal_lookup=signal_lookup,
            output_dir=args.output_dir,
            candidate_id=args.candidate_id,
            cases_per_group=args.cases_per_group,
            fs=float(cfg.window.target_fs),
            low_hz=float(cfg.loss.spectrum_low_hz),
            high_hz=float(cfg.loss.spectrum_high_hz),
            filter_order=int(evaluation.get("lag_bandpass_order", 4)),
        )
        print(f"写出 validation review figures: {len(manifest)} -> {args.output_dir}")
        return
    if args.command == "model-cases":
        labels = pd.read_csv(args.labels)
        corrections = pd.read_csv(args.corrections)
        thresholds = load_frozen_thresholds(args.thresholds)
        selected = select_model_case_rows(
            corrections,
            labels,
            thresholds=thresholds,
            seed=args.seed,
            model_labels=G_SERIES_LABELS,
            max_cases=args.max_cases,
        )
        data = _build_split_data(cfg, split=str(cfg.data.get("test_split", "test")))
        row_ids = selected["dataset_row_id"].astype(int).tolist()
        input_lookup = build_signal_lookup(data.dataset, row_ids)
        prediction_lookup = load_model_prediction_lookup(
            args.predictions_dir,
            model_labels=G_SERIES_LABELS,
            seed=args.seed,
            row_ids=row_ids,
        )
        manifest = plot_model_cases(
            selected,
            input_lookup=input_lookup,
            prediction_lookup=prediction_lookup,
            output_dir=args.output_dir,
            model_labels=G_SERIES_LABELS,
            fs=float(cfg.window.target_fs),
            low_hz=float(cfg.loss.spectrum_low_hz),
            high_hz=float(cfg.loss.spectrum_high_hz),
            filter_order=int(evaluation.get("lag_bandpass_order", 4)),
        )
        print(f"写出 model case figures: {len(manifest)} -> {args.output_dir}")
        return
    raise RuntimeError(f"未知命令: {args.command}")


if __name__ == "__main__":
    main()
