from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


PROTOCOL_ID = "paper-p5-w0-functional-evidence-v1-20260905"
REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "runs/crd_tf_w_v2/p_minus_1_validation_audit"
CORRECTION_ROOT = REPO_ROOT / "runs/crd_tf_w_v2/p_minus_1_film_statistics_correction"
OUTPUT_DIR = Path("runs/paper_evidence_v1/p5_w0_functional_evidence")
SOURCE_MANIFEST_SHA256 = "249c761799b1f8020a77ed51875985776718d9cf1e9701900ce5dae783492f3f"
SOURCE_SUMMARY_SHA256 = "e93881e56441759ebbf4b8c78c0fb6e91921bf2e7333ae66464d3c9feefd8d38"
SOURCE_DECISION_SHA256 = "6aae72dd32b09f0ac5c5a5151f78510ec292d7b2c8942feddd0b751bc0787f4a"
CORRECTION_MANIFEST_SHA256 = "1c6f7218c280b2a1579b2169a2b4752d7c10416d89de67ed5e9a1cc58d104463"
CORRECTED_FILM_SHA256 = "2b7a4edef8e9828356a336c3c1ec7880235914207b00d164bf016ce1cd7a5203"
SEEDS = (20260811, 20260812, 20260813)
INTERVENTIONS = (
    "FULL",
    "BETA_ONLY",
    "GAMMA_ONLY",
    "CONDITION_OFF",
    "RESP",
    "CARRIER",
    "CARRIER_L",
    "CARRIER_H",
    "TIME_MEAN",
    "TIME_SHIFT_30S",
)
PAPER_INTERVENTIONS = (
    "CONDITION_OFF",
    "RESP",
    "CARRIER",
    "CARRIER_L",
    "CARRIER_H",
    "TIME_MEAN",
    "TIME_SHIFT_30S",
)
ERROR_METRICS = (
    "whole_rr_abs_error_bpm_mean",
    "local_rr_mae_bpm_mean",
    "envelope_trajectory_mae_mean",
    "global_envelope_modulation_error_mean",
)
PCC_METRIC = "lag_aware_signed_pcc_mean"
METRICS = (*ERROR_METRICS, PCC_METRIC)
METRIC_LABELS = {
    "whole_rr_abs_error_bpm_mean": "Whole RR",
    "local_rr_mae_bpm_mean": "Local RR",
    "envelope_trajectory_mae_mean": "Envelope trajectory",
    "global_envelope_modulation_error_mean": "Global modulation",
    PCC_METRIC: "Signed PCC",
}
FILM_COLUMNS = (
    "mean_abs_gamma",
    "median_abs_gamma",
    "mean_abs_beta",
    "median_abs_beta",
    "gamma_time_mean_abs_difference",
    "beta_time_mean_abs_difference",
    "gamma_saturation_fraction",
    "beta_saturation_fraction",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_sources() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    paths = {
        "source_manifest": (SOURCE_ROOT / "p_minus_1_manifest.json", SOURCE_MANIFEST_SHA256),
        "source_summary": (SOURCE_ROOT / "intervention_summary.csv", SOURCE_SUMMARY_SHA256),
        "source_decision": (SOURCE_ROOT / "p_minus_1_decision.json", SOURCE_DECISION_SHA256),
        "correction_manifest": (
            CORRECTION_ROOT / "p_minus_1_film_statistics_correction_manifest.json",
            CORRECTION_MANIFEST_SHA256,
        ),
        "corrected_film": (CORRECTION_ROOT / "film_statistics_corrected.csv", CORRECTED_FILM_SHA256),
    }
    records = []
    payloads: dict[str, bytes] = {}
    for role, (path, expected_hash) in paths.items():
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected_hash:
            raise RuntimeError(f"P5 冻结来源 SHA-256 漂移: {role}")
        payloads[role] = data
        records.append(
            {"role": role, "path": str(path), "sha256": digest, "size_bytes": len(data)}
        )
    source_manifest = json.loads(payloads["source_manifest"])
    decision = json.loads(payloads["source_decision"])
    correction_manifest = json.loads(payloads["correction_manifest"])
    if (
        source_manifest.get("status") != "passed"
        or source_manifest.get("complete") is not True
        or source_manifest.get("interventions") != list(INTERVENTIONS)
        or source_manifest.get("evaluation_count") != 30
        or source_manifest.get("per_sample_metric_rows") != 80250
        or source_manifest.get("decision") != "retain_film_no_p2_training"
        or source_manifest.get("research_test_used") is not False
        or source_manifest.get("model_training_used") is not False
        or source_manifest.get("files", {}).get("intervention_summary.csv", {}).get("sha256")
        != SOURCE_SUMMARY_SHA256
        or source_manifest.get("files", {}).get("p_minus_1_decision.json", {}).get("sha256")
        != SOURCE_DECISION_SHA256
    ):
        raise RuntimeError("P5 source audit manifest identity 漂移")
    if decision.get("decision") != "retain_film_no_p2_training" or decision.get("research_test_used") is not False:
        raise RuntimeError("P5 source decision identity 漂移")
    if (
        correction_manifest.get("status") != "passed"
        or correction_manifest.get("complete") is not True
        or correction_manifest.get("source_audit_manifest_sha256") != SOURCE_MANIFEST_SHA256
        or correction_manifest.get("source_decision_sha256") != SOURCE_DECISION_SHA256
        or correction_manifest.get("source_decision_unchanged") is not True
        or correction_manifest.get("correction_scope")
        != "gamma/beta adjacent-frame mean absolute difference only"
        or correction_manifest.get("files", {}).get("film_statistics_corrected.csv", {}).get("sha256")
        != CORRECTED_FILM_SHA256
        or correction_manifest.get("research_test_used") is not False
        or correction_manifest.get("model_training_used") is not False
    ):
        raise RuntimeError("P5 correction manifest identity 漂移")
    summary = pd.read_csv(io.BytesIO(payloads["source_summary"]))
    film = pd.read_csv(io.BytesIO(payloads["corrected_film"]))
    _validate_intervention_summary(summary)
    _validate_corrected_film(film)
    for record in records:
        if sha256_file(record["path"]) != record["sha256"]:
            raise RuntimeError("P5 来源在读取期间发生变化")
    return summary, film, correction_manifest, records


def build_seed_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    seed_rows = summary.loc[summary["summary_level"].eq("seed")].copy()
    records: list[dict[str, Any]] = []
    for seed in SEEDS:
        group = seed_rows.loc[seed_rows["seed"].eq(seed)].set_index("intervention")
        for intervention in INTERVENTIONS[1:]:
            for metric in METRICS:
                full = float(group.loc["FULL", metric])
                value = float(group.loc[intervention, metric])
                if metric in ERROR_METRICS:
                    degradation = (value - full) / full
                    unit = "relative"
                else:
                    degradation = full - value
                    unit = "absolute_pcc_decline"
                records.append(
                    {
                        "seed": seed,
                        "intervention": intervention,
                        "metric": metric,
                        "metric_label": METRIC_LABELS[metric],
                        "unit": unit,
                        "full_value": full,
                        "intervention_value": value,
                        "raw_delta_intervention_minus_full": value - full,
                        "oriented_degradation": degradation,
                        "worsened": degradation > 0.0,
                        "improved": degradation < 0.0,
                    }
                )
    return pd.DataFrame.from_records(records)


def build_aggregate_deltas(summary: pd.DataFrame, seed_deltas: pd.DataFrame) -> pd.DataFrame:
    aggregate = summary.loc[summary["summary_level"].eq("three_seed_mean")].set_index("intervention")
    records: list[dict[str, Any]] = []
    for intervention in INTERVENTIONS[1:]:
        for metric in METRICS:
            group = seed_deltas.loc[
                seed_deltas["intervention"].eq(intervention) & seed_deltas["metric"].eq(metric)
            ].sort_values("seed")
            if tuple(group["seed"].astype(int)) != SEEDS:
                raise RuntimeError("P5 seed delta identity 不完整")
            full = float(aggregate.loc["FULL", metric])
            value = float(aggregate.loc[intervention, metric])
            aggregate_degradation = (value - full) / full if metric in ERROR_METRICS else full - value
            values = group["oriented_degradation"].to_numpy(dtype=np.float64)
            records.append(
                {
                    "intervention": intervention,
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "unit": str(group["unit"].iloc[0]),
                    "full_three_seed_mean": full,
                    "intervention_three_seed_mean": value,
                    "raw_delta_intervention_minus_full": value - full,
                    "aggregate_mean_oriented_degradation": aggregate_degradation,
                    "paired_seed_oriented_degradation_mean": float(np.mean(values)),
                    "paired_seed_oriented_degradation_sample_sd": float(np.std(values, ddof=1)),
                    "worsened_seed_count": int(np.sum(values > 0.0)),
                    "improved_seed_count": int(np.sum(values < 0.0)),
                    "equal_seed_count": int(np.sum(values == 0.0)),
                }
            )
    return pd.DataFrame.from_records(records)


def build_paper_table(aggregate: pd.DataFrame) -> pd.DataFrame:
    rows = []
    indexed = aggregate.set_index(["intervention", "metric"])
    for intervention in PAPER_INTERVENTIONS:
        row: dict[str, Any] = {"intervention": intervention}
        for metric in METRICS:
            source = indexed.loc[(intervention, metric)]
            short = {
                "whole_rr_abs_error_bpm_mean": "whole_rr",
                "local_rr_mae_bpm_mean": "local_rr",
                "envelope_trajectory_mae_mean": "trajectory",
                "global_envelope_modulation_error_mean": "global",
                PCC_METRIC: "pcc",
            }[metric]
            row[f"{short}_aggregate_degradation"] = float(
                source["aggregate_mean_oriented_degradation"]
            )
            row[f"{short}_paired_seed_degradation_mean"] = float(
                source["paired_seed_oriented_degradation_mean"]
            )
            row[f"{short}_paired_seed_degradation_sample_sd"] = float(
                source["paired_seed_oriented_degradation_sample_sd"]
            )
            row[f"{short}_worsened_seeds"] = int(source["worsened_seed_count"])
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def build_film_summary(film: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_rows = (
        film.groupby("seed", sort=True)[list(FILM_COLUMNS)]
        .mean()
        .reset_index()
    )
    records = []
    for column in FILM_COLUMNS:
        values = seed_rows[column].to_numpy(dtype=np.float64)
        records.append(
            {
                "statistic": column,
                "three_seed_mean": float(np.mean(values)),
                "three_seed_sample_sd": float(np.std(values, ddof=1)),
                "seed_count": 3,
                "sample_rows_per_seed": 2675,
            }
        )
    return seed_rows, pd.DataFrame.from_records(records)


def render_functional_panel(aggregate: pd.DataFrame, path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    interventions = list(INTERVENTIONS[1:])
    indexed = aggregate.set_index(["intervention", "metric"])
    error_values = np.asarray(
        [
            [100.0 * float(indexed.loc[(intervention, metric), "aggregate_mean_oriented_degradation"]) for metric in ERROR_METRICS]
            for intervention in interventions
        ]
    )
    pcc_values = np.asarray(
        [[float(indexed.loc[(intervention, PCC_METRIC), "aggregate_mean_oriented_degradation"])] for intervention in interventions]
    )
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 6.2), gridspec_kw={"width_ratios": [4.2, 1.0]})
    error_limit = max(1.0, float(np.max(np.abs(error_values))))
    pcc_limit = max(0.001, float(np.max(np.abs(pcc_values))))
    left = axes[0].imshow(error_values, cmap="RdBu_r", vmin=-error_limit, vmax=error_limit, aspect="auto")
    right = axes[1].imshow(pcc_values, cmap="RdBu_r", vmin=-pcc_limit, vmax=pcc_limit, aspect="auto")
    axes[0].set_xticks(range(4), [METRIC_LABELS[m] for m in ERROR_METRICS], rotation=25, ha="right")
    axes[0].set_yticks(range(len(interventions)), interventions)
    axes[1].set_xticks([0], ["PCC decline"])
    axes[1].set_yticks(range(len(interventions)), [""] * len(interventions))
    for row in range(len(interventions)):
        for column in range(4):
            axes[0].text(column, row, f"{error_values[row, column]:+.1f}%", ha="center", va="center", fontsize=8)
        axes[1].text(0, row, f"{pcc_values[row, 0]:+.4f}", ha="center", va="center", fontsize=8)
    axes[0].set_title("Error degradation relative to FULL")
    axes[1].set_title("Absolute PCC decline")
    figure.colorbar(left, ax=axes[0], shrink=0.75, label="Relative degradation (%)")
    figure.colorbar(right, ax=axes[1], shrink=0.75, label="FULL − intervention")
    figure.suptitle("Frozen W0 validation functional interventions")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight", metadata={"Software": "matplotlib"})
    plt.close(figure)


def run_p5_summary(*, repo_root: str | Path, command: str) -> Path:
    root = Path(repo_root).resolve()
    output = root / OUTPUT_DIR
    if output.exists():
        raise FileExistsError(f"P5 输出目录禁止覆盖: {output}")
    commit = _require_clean_git(root)
    summary, film, correction_manifest, source_records = load_frozen_sources()
    seed_deltas = build_seed_deltas(summary)
    aggregate = build_aggregate_deltas(summary, seed_deltas)
    paper_table = build_paper_table(aggregate)
    film_seed, film_summary = build_film_summary(film)
    for frame in (seed_deltas, aggregate, paper_table, film_seed, film_summary):
        numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise FloatingPointError("P5 输出包含 NaN/Inf")
    output.mkdir(parents=True, exist_ok=False)
    seed_deltas.to_csv(output / "intervention_seed_deltas.csv", index=False)
    aggregate.to_csv(output / "intervention_primary_deltas.csv", index=False)
    paper_table.to_csv(output / "paper_functional_table.csv", index=False)
    film_seed.to_csv(output / "film_corrected_seed_summary.csv", index=False)
    film_summary.to_csv(output / "film_corrected_three_seed_summary.csv", index=False)
    render_functional_panel(aggregate, output / "functional_degradation_panel.png")
    for record in source_records:
        if sha256_file(record["path"]) != record["sha256"]:
            raise RuntimeError("P5 来源在写出期间发生变化")
    receipt = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "paper-p5-w0-functional-evidence-summary-v1",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution": {
            "command": command,
            "cwd": str(root),
            "git_commit": commit,
            "git_dirty": False,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "inputs": source_records,
        "source_hashes_unchanged": True,
        "counts": {
            "interventions": 10,
            "comparison_interventions": 9,
            "primary_metrics": 5,
            "seed_delta_rows": len(seed_deltas),
            "aggregate_delta_rows": len(aggregate),
            "paper_table_rows": len(paper_table),
            "corrected_film_rows": len(film),
            "corrected_film_seed_rows": len(film_seed),
            "corrected_film_statistics": len(film_summary),
        },
        "aggregation": {
            "source": "frozen_validation_summary",
            "error_delta": "(intervention-FULL)/FULL",
            "pcc_delta": "FULL-intervention",
            "paired_seed_mean": "arithmetic_mean",
            "paired_seed_sd": "sample_sd_ddof1",
            "film_within_seed": "sample_direct_mean",
            "film_across_seed": "arithmetic_mean_and_sample_sd_ddof1",
        },
        "access": {
            "frozen_validation_summary_read": True,
            "corrected_film_statistics_read": True,
            "per_sample_prediction_metrics_read": False,
            "checkpoint_content_read": False,
            "dataset_or_index_read": False,
            "signal_or_target_array_read": False,
            "research_test_read": False,
            "model_inference_used": False,
            "training_used": False,
            "gpu_used": False,
            "cache_used": False,
        },
        "source_decision": "retain_film_no_p2_training",
        "correction_scope": correction_manifest["correction_scope"],
        "artifacts": _artifact_records(output),
    }
    _write_json_exclusive(output / "summary_receipt.json", receipt)
    _write_json_exclusive(
        output / "artifact_manifest.json",
        {
            "protocol_id": PROTOCOL_ID,
            "status": "complete",
            "files": _artifact_records(output),
        },
    )
    return output


def _validate_intervention_summary(frame: pd.DataFrame) -> None:
    required = {"summary_level", "seed", "intervention", "selected_epoch", "checkpoint_sha256", "n_samples", *METRICS}
    if len(frame) != 40 or not required.issubset(frame.columns):
        raise RuntimeError("P5 intervention summary schema/count 漂移")
    seed_rows = frame.loc[frame["summary_level"].eq("seed")].copy()
    aggregate = frame.loc[frame["summary_level"].eq("three_seed_mean")].copy()
    if len(seed_rows) != 30 or len(aggregate) != 10:
        raise RuntimeError("P5 intervention summary level count 漂移")
    seed_rows["seed"] = seed_rows["seed"].astype(int)
    expected = {(seed, intervention) for seed in SEEDS for intervention in INTERVENTIONS}
    if set(zip(seed_rows["seed"], seed_rows["intervention"])) != expected or tuple(aggregate["intervention"]) != INTERVENTIONS:
        raise RuntimeError("P5 intervention/seed matrix 漂移")
    if not seed_rows["n_samples"].eq(2675).all():
        raise RuntimeError("P5 validation sample count 漂移")
    for seed, group in seed_rows.groupby("seed"):
        if group["selected_epoch"].nunique() != 1 or group["checkpoint_sha256"].nunique() != 1:
            raise RuntimeError(f"P5 seed={seed} checkpoint identity 漂移")
    if not np.isfinite(seed_rows[list(METRICS)].to_numpy(dtype=np.float64)).all():
        raise FloatingPointError("P5 primary summary 含 NaN/Inf")
    indexed = aggregate.set_index("intervention")
    for intervention, group in seed_rows.groupby("intervention"):
        for metric in METRICS:
            values = group.sort_values("seed")[metric].to_numpy(dtype=np.float64)
            if not np.isclose(float(indexed.loc[intervention, metric]), float(np.mean(values)), rtol=0.0, atol=1e-12):
                raise RuntimeError(f"P5 aggregate mean 重算不一致: {intervention}/{metric}")
            sd_column = f"{metric}_seed_sd"
            if sd_column not in aggregate or not np.isclose(float(indexed.loc[intervention, sd_column]), float(np.std(values, ddof=1)), rtol=0.0, atol=1e-12):
                raise RuntimeError(f"P5 aggregate SD 重算不一致: {intervention}/{metric}")


def _validate_corrected_film(frame: pd.DataFrame) -> None:
    required = {"seed", "intervention", "dataset_row_id", *FILM_COLUMNS}
    if len(frame) != 8025 or not required.issubset(frame.columns) or set(frame["intervention"].astype(str)) != {"FULL"}:
        raise RuntimeError("P5 corrected FiLM schema/count/intervention 漂移")
    frame["seed"] = frame["seed"].astype(int)
    if set(frame["seed"]) != set(SEEDS):
        raise RuntimeError("P5 corrected FiLM seeds 漂移")
    reference: np.ndarray | None = None
    for seed, group in frame.groupby("seed"):
        ids = group["dataset_row_id"].to_numpy(dtype=np.int64)
        if len(ids) != 2675 or len(np.unique(ids)) != 2675:
            raise RuntimeError(f"P5 seed={seed} corrected FiLM row identity 漂移")
        ids = np.sort(ids)
        if reference is None:
            reference = ids
        elif not np.array_equal(reference, ids):
            raise RuntimeError("P5 corrected FiLM row set 跨 seed 漂移")
    values = frame[list(FILM_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise FloatingPointError("P5 corrected FiLM 值必须 finite/nonnegative")


def _artifact_records(output: Path) -> list[dict[str, Any]]:
    return [
        {"filename": path.name, "sha256": sha256_file(path), "size_bytes": int(path.stat().st_size)}
        for path in sorted(output.iterdir())
        if path.is_file()
    ]


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _require_clean_git(root: Path) -> str:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, capture_output=True, text=True, check=False)
    if commit.returncode != 0 or status.returncode != 0 or status.stdout.strip():
        raise RuntimeError("P5 正式汇总要求干净 Git commit")
    return commit.stdout.strip()


__all__ = [
    "INTERVENTIONS",
    "METRICS",
    "OUTPUT_DIR",
    "PAPER_INTERVENTIONS",
    "PROTOCOL_ID",
    "build_aggregate_deltas",
    "build_film_summary",
    "build_paper_table",
    "build_seed_deltas",
    "load_frozen_sources",
    "render_functional_panel",
    "run_p5_summary",
]
