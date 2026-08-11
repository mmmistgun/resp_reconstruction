from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from resp_train.crd.candidate_lock import DEFAULT_CANDIDATE_LOCK, REPO_ROOT, sha256_file, verify_candidate_lock
from resp_train.crd.config import (
    CRD_102_FAILURE_DIAGNOSTIC_PROTOCOL_VERSION,
    CRD_102_FAILURE_METADATA_PROTOCOL_VERSION,
    FORMAL_SEEDS,
)
from resp_train.crd.failure_diagnostics import (
    DEFAULT_OUTPUT_ROOT as DEFAULT_FAILURE_OUTPUT_ROOT,
    MANIFEST_FILENAME as FAILURE_MANIFEST_FILENAME,
    WINDOW_FILENAME as FAILURE_WINDOW_FILENAME,
)
from resp_train.crd.s2_selection import BASE_VARIANT
from resp_train.utils.run import save_execution_manifest


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs/crd_v1/crd_102_failure_metadata_diagnostic"
WINDOW_FILENAME = "crd102_failure_metadata_windows.csv"
CONTINUOUS_ASSOCIATION_FILENAME = "crd102_metadata_associations.csv"
FAILURE_CONTRAST_FILENAME = "crd102_metadata_failure_contrasts.csv"
CATEGORICAL_FILENAME = "crd102_metadata_categorical_summary.csv"
EPISODE_FILENAME = "crd102_failure_episodes.csv"
OVERVIEW_FILENAME = "crd102_failure_metadata_overview.json"
MANIFEST_FILENAME = "crd102_failure_metadata_manifest.json"

CONTINUOUS_METADATA_COLUMNS = (
    "hard_valid_ratio",
    "state_alignment_valid_ratio",
    "transient_motion_ratio",
    "posture_transition_ratio",
    "amplitude_reliable_ratio",
    "normalization_reliable_ratio",
    "rate_confidence_score",
    "phase_confidence_score",
    "event_confidence_score",
    "waveform_confidence_score",
    "alignment_confidence_score",
    "supervision_confidence_score",
    "state_alignment_lag_s",
    "state_alignment_drift_s_per_hour",
    "training_finite_ratio",
)
CATEGORICAL_METADATA_COLUMNS = (
    "rate_confidence_level",
    "phase_confidence_level",
    "event_confidence_level",
    "waveform_confidence_level",
    "alignment_confidence_level",
    "supervision_confidence_level",
    "state_alignment_method",
    "state_alignment_is_reference_assisted",
    "allowed_losses",
)
OUTCOME_COLUMNS = {
    "target_envelope_modulation": "target_envelope_modulation",
    "local_rr_error": "local_rr_mae_bpm_seed_mean",
    "trajectory_error": "envelope_trajectory_mae_seed_mean",
    "global_modulation_error": "global_envelope_modulation_error_seed_mean",
    "pcc_error": "lag_aware_signed_pcc_seed_mean",
    "ibi_coverage_error": "ibi_coverage_seed_mean",
    "local_rr_seed_sd": "local_rr_mae_bpm_seed_sd",
    "persistent_core_failure_count": "persistent_core_failure_count",
}
EPISODE_STEP_SEC = 30.0


def summarize_crd102_failure_metadata(
    *,
    candidate_lock_path: str | Path = DEFAULT_CANDIDATE_LOCK,
    failure_root: str | Path = DEFAULT_FAILURE_OUTPUT_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    """把第一层 failure consensus 与冻结 dataset index 一对一配对。"""

    _assert_clean_repository()
    verification = verify_candidate_lock(candidate_lock_path)
    records = sorted(
        (record for record in verification.records if record["variant"] == BASE_VARIANT),
        key=lambda item: int(item["seed"]),
    )
    if len(records) != len(FORMAL_SEEDS):
        raise RuntimeError("candidate lock 中 CRD_102 checkpoint 不完整")

    failure_root = Path(failure_root)
    failure_manifest_path = failure_root / FAILURE_MANIFEST_FILENAME
    consensus_path = failure_root / FAILURE_WINDOW_FILENAME
    if not failure_manifest_path.exists() or not consensus_path.exists():
        raise FileNotFoundError("第一层 CRD_102 failure diagnostic 产物不完整")
    failure_manifest = json.loads(failure_manifest_path.read_text(encoding="utf-8"))
    if (
        failure_manifest.get("protocol") != CRD_102_FAILURE_DIAGNOSTIC_PROTOCOL_VERSION
        or failure_manifest.get("git_dirty") is not False
        or int(failure_manifest.get("checkpoint_count", -1)) != len(FORMAL_SEEDS)
        or failure_manifest.get("candidate_lock_sha256") != verification.lock_sha256
        or failure_manifest.get("research_test_used") is not False
    ):
        raise RuntimeError("第一层 CRD_102 failure diagnostic manifest 不满足冻结要求")

    index_paths: set[Path] = set()
    for record in records:
        config_path = Path(record["checkpoint_path"]).parent / "config.yaml"
        cfg = OmegaConf.load(config_path)
        index_paths.add((Path(str(cfg.data.dataset_root)) / str(cfg.data.index_csv)).resolve())
    if len(index_paths) != 1:
        raise RuntimeError(f"CRD_102 三 seed dataset index 不统一: {sorted(map(str, index_paths))}")
    index_path = next(iter(index_paths))

    consensus = pd.read_csv(consensus_path)
    index = pd.read_csv(index_path)
    outputs = build_metadata_diagnostics(consensus, index)
    overview = _build_overview(outputs)

    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"CRD_102 failure metadata diagnostic 禁止覆盖: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = output_root.parent / f".{output_root.name}.{uuid4().hex}.tmp"
    temporary_dir.mkdir(parents=False, exist_ok=False)
    try:
        outputs["windows"].to_csv(temporary_dir / WINDOW_FILENAME, index=False)
        outputs["associations"].to_csv(temporary_dir / CONTINUOUS_ASSOCIATION_FILENAME, index=False)
        outputs["contrasts"].to_csv(temporary_dir / FAILURE_CONTRAST_FILENAME, index=False)
        outputs["categorical"].to_csv(temporary_dir / CATEGORICAL_FILENAME, index=False)
        outputs["episodes"].to_csv(temporary_dir / EPISODE_FILENAME, index=False)
        (temporary_dir / OVERVIEW_FILENAME).write_text(
            json.dumps(
                {
                    "protocol": CRD_102_FAILURE_METADATA_PROTOCOL_VERSION,
                    "candidate_lock_sha256": verification.lock_sha256,
                    "source_failure_protocol": CRD_102_FAILURE_DIAGNOSTIC_PROTOCOL_VERSION,
                    "source_failure_manifest_sha256": sha256_file(failure_manifest_path),
                    "source_failure_consensus_sha256": sha256_file(consensus_path),
                    "dataset_index": str(index_path),
                    "dataset_index_sha256": sha256_file(index_path),
                    "research_test_used": False,
                    "checkpoint_reselection_allowed": False,
                    "confirmatory_p_values_used": False,
                    "result_informed_exploratory_followup": True,
                    **overview,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        save_execution_manifest(
            temporary_dir / MANIFEST_FILENAME,
            task="crd_102_validation_failure_metadata_diagnostic",
            phase="crd102_result_informed_metadata_followup",
            protocol=CRD_102_FAILURE_METADATA_PROTOCOL_VERSION,
            candidate_lock_sha256=verification.lock_sha256,
            source_failure_manifest_sha256=sha256_file(failure_manifest_path),
            source_failure_consensus_sha256=sha256_file(consensus_path),
            dataset_index=str(index_path),
            dataset_index_sha256=sha256_file(index_path),
            validation_window_count=int(len(outputs["windows"])),
            research_test_used=False,
        )
        os.replace(temporary_dir, output_root)
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
    return output_root / OVERVIEW_FILENAME


def build_metadata_diagnostics(consensus: pd.DataFrame, index: pd.DataFrame) -> dict[str, pd.DataFrame]:
    required_consensus = {
        "dataset_row_id",
        "samp_id",
        "coupling_state_id",
        "envelope_target_stratum",
        "target_envelope_modulation",
        "local_rr_mae_bpm_seed_mean",
        "local_rr_mae_bpm_seed_sd",
        "envelope_trajectory_mae_seed_mean",
        "global_envelope_modulation_error_seed_mean",
        "lag_aware_signed_pcc_seed_mean",
        "ibi_coverage_seed_mean",
        "local_rr_mae_bpm_persistent_failure",
        "persistent_core_failure_count",
    }
    required_index = {
        "dataset_row_id",
        "split",
        "samp_id",
        "coupling_state_id",
        "window_start_s",
        "window_end_s",
        "window_duration_s",
        *CONTINUOUS_METADATA_COLUMNS,
        *CATEGORICAL_METADATA_COLUMNS,
    }
    missing_consensus = sorted(required_consensus - set(consensus.columns))
    missing_index = sorted(required_index - set(index.columns))
    if missing_consensus or missing_index:
        raise ValueError(f"metadata diagnostic 缺少字段: consensus={missing_consensus}, index={missing_index}")
    if consensus.empty or consensus["dataset_row_id"].duplicated().any() or index["dataset_row_id"].duplicated().any():
        raise ValueError("metadata diagnostic 要求非空且 dataset_row_id 唯一")

    index_columns = list(required_index)
    merged = consensus.merge(
        index[index_columns],
        on="dataset_row_id",
        how="left",
        suffixes=("", "_index"),
        validate="one_to_one",
    )
    if merged["split"].isna().any() or set(merged["split"].astype(str)) != {"val"}:
        raise RuntimeError("metadata diagnostic index 配对缺失或包含非 validation split")
    for column in ("samp_id", "coupling_state_id"):
        if not np.array_equal(
            pd.to_numeric(merged[column], errors="coerce").to_numpy(),
            pd.to_numeric(merged[f"{column}_index"], errors="coerce").to_numpy(),
        ):
            raise RuntimeError(f"metadata diagnostic identity 不一致: {column}")
        merged = merged.drop(columns=f"{column}_index")
    if not np.allclose(merged["window_duration_s"].to_numpy(dtype=np.float64), 180.0, rtol=0.0, atol=1e-12):
        raise RuntimeError("metadata diagnostic window duration 不再是冻结的 180 s")

    merged["multimetric_core_failure"] = merged["persistent_core_failure_count"].ge(2)
    associations = _metadata_associations(merged)
    contrasts = _failure_contrasts(merged)
    categorical = _categorical_summary(merged)
    episodes = pd.concat(
        [
            _failure_episodes(merged, failure_column="local_rr_mae_bpm_persistent_failure", failure_type="local_rr"),
            _failure_episodes(merged, failure_column="multimetric_core_failure", failure_type="multimetric_core"),
        ],
        ignore_index=True,
    )
    return {
        "windows": merged.sort_values("dataset_row_id").reset_index(drop=True),
        "associations": associations,
        "contrasts": contrasts,
        "categorical": categorical,
        "episodes": episodes,
    }


def _metadata_associations(frame: pd.DataFrame) -> pd.DataFrame:
    outcomes = {
        name: (-pd.to_numeric(frame[column], errors="coerce") if name in {"pcc_error", "ibi_coverage_error"} else pd.to_numeric(frame[column], errors="coerce"))
        for name, column in OUTCOME_COLUMNS.items()
    }
    rows: list[dict[str, Any]] = []
    for metadata in CONTINUOUS_METADATA_COLUMNS:
        left = pd.to_numeric(frame[metadata], errors="coerce")
        for outcome, right in outcomes.items():
            pair = pd.DataFrame({"metadata": left, "outcome": right}).dropna()
            rho = float(pair["metadata"].corr(pair["outcome"], method="spearman")) if len(pair) >= 2 else math.nan
            rows.append(
                {
                    "metadata": metadata,
                    "outcome": outcome,
                    "n": int(len(pair)),
                    "spearman_rho": rho,
                }
            )
    return pd.DataFrame(rows)


def _failure_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    definitions = {
        "persistent_local_rr": frame["local_rr_mae_bpm_persistent_failure"].astype(bool),
        "multimetric_core": frame["multimetric_core_failure"].astype(bool),
    }
    for failure_name, failure in definitions.items():
        for metadata in CONTINUOUS_METADATA_COLUMNS:
            values = pd.to_numeric(frame[metadata], errors="coerce")
            bad = values[failure].dropna()
            other = values[~failure].dropna()
            rows.append(
                {
                    "failure_definition": failure_name,
                    "metadata": metadata,
                    "failure_n": int(len(bad)),
                    "reference_n": int(len(other)),
                    "failure_mean": float(bad.mean()),
                    "reference_mean": float(other.mean()),
                    "mean_difference": float(bad.mean() - other.mean()),
                    "failure_median": float(bad.median()),
                    "reference_median": float(other.median()),
                    "median_difference": float(bad.median() - other.median()),
                }
            )
    return pd.DataFrame(rows)


def _categorical_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in CATEGORICAL_METADATA_COLUMNS:
        for value, group in frame.groupby(column, sort=True, dropna=False):
            rows.append(
                {
                    "metadata": column,
                    "value": str(value),
                    "window_n": int(len(group)),
                    "window_fraction": float(len(group) / len(frame)),
                    "samp_id_n": int(group["samp_id"].nunique()),
                    "target_envelope_modulation_mean": float(group["target_envelope_modulation"].mean()),
                    "local_rr_mae_bpm_mean": float(group["local_rr_mae_bpm_seed_mean"].mean()),
                    "persistent_local_rr_failure_fraction": float(group["local_rr_mae_bpm_persistent_failure"].mean()),
                    "multimetric_core_failure_fraction": float(group["multimetric_core_failure"].mean()),
                    "trajectory_error_mean": float(group["envelope_trajectory_mae_seed_mean"].mean()),
                    "signed_pcc_mean": float(group["lag_aware_signed_pcc_seed_mean"].mean()),
                    "ibi_coverage_mean": float(group["ibi_coverage_seed_mean"].mean()),
                }
            )
    return pd.DataFrame(rows)


def _failure_episodes(frame: pd.DataFrame, *, failure_column: str, failure_type: str) -> pd.DataFrame:
    failures = frame.loc[frame[failure_column].astype(bool)].copy()
    if failures.empty:
        return pd.DataFrame()
    failures = failures.sort_values(["samp_id", "window_start_s", "dataset_row_id"]).reset_index(drop=True)
    new_episode = failures["samp_id"].ne(failures["samp_id"].shift()) | failures["window_start_s"].sub(
        failures["window_start_s"].shift()
    ).gt(EPISODE_STEP_SEC + 1e-12)
    failures["episode_id"] = new_episode.cumsum().astype(int)
    rows: list[dict[str, Any]] = []
    for _, group in failures.groupby("episode_id", sort=True):
        starts = pd.to_numeric(group["window_start_s"], errors="coerce")
        ends = pd.to_numeric(group["window_end_s"], errors="coerce")
        rows.append(
            {
                "failure_type": failure_type,
                "episode_id": int(len(rows) + 1),
                "samp_id": int(group["samp_id"].iloc[0]),
                "first_dataset_row_id": int(group["dataset_row_id"].iloc[0]),
                "last_dataset_row_id": int(group["dataset_row_id"].iloc[-1]),
                "start_s": float(starts.min()),
                "end_s": float(ends.max()),
                "span_sec": float(ends.max() - starts.min()),
                "failure_window_n": int(len(group)),
                "coupling_state_n": int(group["coupling_state_id"].nunique()),
                "coupling_state_ids": ";".join(map(str, sorted(group["coupling_state_id"].unique()))),
                "target_envelope_modulation_mean": float(group["target_envelope_modulation"].mean()),
                "local_rr_mae_bpm_mean": float(group["local_rr_mae_bpm_seed_mean"].mean()),
                "persistent_core_failure_count_mean": float(group["persistent_core_failure_count"].mean()),
                "transient_motion_ratio_mean": float(group["transient_motion_ratio"].mean()),
                "rate_confidence_score_mean": float(group["rate_confidence_score"].mean()),
                "waveform_confidence_score_mean": float(group["waveform_confidence_score"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _build_overview(outputs: dict[str, pd.DataFrame]) -> dict[str, Any]:
    windows = outputs["windows"]
    associations = outputs["associations"]
    episodes = outputs["episodes"]
    local_associations = associations.loc[associations["outcome"].eq("local_rr_error")].copy()
    local_associations["abs_rho"] = local_associations["spearman_rho"].abs()
    strongest = local_associations.sort_values(["abs_rho", "metadata"], ascending=[False, True]).iloc[0]
    episode_summary: dict[str, Any] = {}
    for failure_type in ("local_rr", "multimetric_core"):
        group = episodes.loc[episodes["failure_type"].eq(failure_type)]
        episode_summary[failure_type] = {
            "failure_window_count": int(group["failure_window_n"].sum()),
            "episode_count": int(len(group)),
            "singleton_episode_fraction": float(group["failure_window_n"].eq(1).mean()),
            "longest_episode_window_count": int(group["failure_window_n"].max()),
            "longest_episode_span_sec": float(group["span_sec"].max()),
        }
    return {
        "validation_window_count": int(len(windows)),
        "strongest_metadata_association_with_local_rr": str(strongest["metadata"]),
        "strongest_metadata_local_rr_spearman_rho": float(strongest["spearman_rho"]),
        "episodes": episode_summary,
    }


def _assert_clean_repository() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode != 0:
        raise RuntimeError(f"无法读取 git 状态: {status.stderr.strip()}")
    if status.stdout.strip():
        raise RuntimeError("CRD_102 failure metadata diagnostic 必须从干净 git commit 生成")
