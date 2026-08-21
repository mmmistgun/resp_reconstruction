from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import resp_train.temporal.formal_summary as summary
from resp_train.temporal.formal import EXPECTED_CANDIDATES, FORMAL_SEEDS
from resp_train.temporal.formal_summary import (
    FROZEN_CONFIG_SHA256,
    OUTPUT_FILENAMES,
    PROTOCOL_ID,
    RECEIPT_SCHEMA_VERSION,
    SummaryConfig,
    _decision_payload,
    _validate_metrics,
    build_candidate_summary,
    build_paired_seed_directions,
    build_pareto_outputs,
    dimension_comparison,
    load_summary_config,
    run_validation_summary,
    validate_summary_config,
    validate_summary_receipt,
)


def test_frozen_summary_config_is_exact_and_closes_15_runs() -> None:
    config = load_summary_config()
    assert config.sha256 == FROZEN_CONFIG_SHA256
    assert len(config.raw["formal_runs"]) == 15
    identities = {(row["candidate_id"], row["seed"]) for row in config.raw["formal_runs"]}
    assert identities == {(candidate, seed) for candidate in EXPECTED_CANDIDATES for seed in FORMAL_SEEDS}
    groups = pd.DataFrame(config.raw["formal_runs"]).groupby(
        ["execution_group", "cuda_visible_devices"]
    ).size().to_dict()
    assert groups == {("gpu_0", "0"): 8, ("gpu_1", "1"): 7}
    for prefix in (
        "summary_protocol",
        "formal_dual_gpu_plan",
        "formal_dual_gpu_correction",
        "formal_dual_gpu_implementation_receipt",
    ):
        path = summary.REPO_ROOT / config.raw["provenance"][f"{prefix}_path"]
        assert summary.sha256_file(path) == config.raw["provenance"][f"{prefix}_sha256"]


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("authorization", "research_test_evaluation", True),
        ("aggregation", "candidate_seed_sd", "population_ddof_0"),
        ("access", "checkpoint_content_allowed", True),
        ("output", "allow_overwrite", True),
    ],
)
def test_summary_config_rejects_contract_drift(section: str, key: str, value: object) -> None:
    raw = copy.deepcopy(load_summary_config().raw)
    raw[section][key] = value
    with pytest.raises(ValueError):
        validate_summary_config(raw)


def test_summary_config_rejects_metric_tolerance_and_run_identity_drift() -> None:
    raw = copy.deepcopy(load_summary_config().raw)
    raw["primary_metrics"][0]["tolerance"] = 0.01
    with pytest.raises(ValueError, match="Pareto"):
        validate_summary_config(raw)
    raw = copy.deepcopy(load_summary_config().raw)
    raw["formal_runs"][0]["seed"] = 1
    with pytest.raises(ValueError, match="formal"):
        validate_summary_config(raw)


def test_dimension_comparison_respects_relative_absolute_and_exact_boundaries() -> None:
    relative_min = {"direction": "minimize", "tolerance_kind": "relative", "tolerance": 0.005}
    assert dimension_comparison(1.005, 1.0, relative_min) == (True, False)
    assert dimension_comparison(0.995, 1.0, relative_min) == (True, False)
    assert dimension_comparison(0.994, 1.0, relative_min) == (True, True)
    absolute_max = {"direction": "maximize", "tolerance_kind": "absolute", "tolerance": 0.002}
    assert dimension_comparison(0.798, 0.8, absolute_max) == (True, False)
    assert dimension_comparison(0.803, 0.8, absolute_max) == (True, True)
    exact_min = {"direction": "minimize", "tolerance_kind": "exact", "tolerance": 0.0}
    assert dimension_comparison(10, 10, exact_min) == (True, False)
    assert dimension_comparison(9, 10, exact_min) == (True, True)
    with pytest.raises(ValueError, match="正值"):
        dimension_comparison(0.0, 1.0, relative_min)


def _synthetic_seed_summary(config: SummaryConfig) -> pd.DataFrame:
    rows = []
    primary = [item["name"] for item in config.raw["primary_metrics"]]
    supplementary = config.raw["supplementary_metrics"]["seed_aggregated"]
    for candidate_index, candidate in enumerate(config.raw["candidates"]):
        for seed_index, seed in enumerate(FORMAL_SEEDS):
            base = 1.0 + candidate_index * 0.1 + seed_index * 0.01
            row = {
                "candidate_id": candidate["candidate_id"],
                "family": candidate["family"],
                "role": candidate["role"],
                "seed": seed,
                "execution_group": "gpu_0" if seed_index % 2 == 0 else "gpu_1",
                "cuda_visible_devices": "0" if seed_index % 2 == 0 else "1",
                "selected_epoch": 10 + seed_index,
                "selected_validation_local_rr_mae": base,
                "trainable_parameters": candidate["trainable_parameters"],
                "receipt_sha256": "a" * 64,
                "metrics_sha256": "b" * 64,
                "stored_summary_sha256": "c" * 64,
                "n_samples": 2675,
            }
            for metric in primary:
                row[metric] = 0.8 - candidate_index * 0.01 + seed_index * 0.001 if "pcc" in metric else base
            for metric in supplementary:
                row[metric] = 0.1 + candidate_index * 0.01 + seed_index * 0.001
            rows.append(row)
    return pd.DataFrame(rows)


def _synthetic_engineering(config: SummaryConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_id": candidate["candidate_id"],
                "update_throughput_windows_per_second": 1000.0 - index * 100.0,
                "update_peak_allocated_mib": 100.0 + index * 20.0,
                "batch1_inference_p50_ms": 1.0 + index,
                "batch1_inference_p90_ms": 1.1 + index,
                "update_peak_reserved_mib": 120.0 + index * 20.0,
            }
            for index, candidate in enumerate(config.raw["candidates"])
        ]
    )


def test_seed_aggregation_uses_arithmetic_mean_and_sample_sd() -> None:
    config = load_summary_config()
    seed = _synthetic_seed_summary(config)
    candidate = build_candidate_summary(seed, config.raw["candidates"], FORMAL_SEEDS, _synthetic_engineering(config))
    assert len(candidate) == 5
    first = candidate.iloc[0]
    values = seed.loc[seed["candidate_id"].eq(EXPECTED_CANDIDATES[0]), "local_rr_mae_bpm_mean"]
    assert first["local_rr_mae_bpm_mean_seed_mean"] == pytest.approx(values.mean())
    assert first["local_rr_mae_bpm_mean_seed_sd"] == pytest.approx(values.std(ddof=1))
    assert first["local_rr_mae_bpm_mean_seed_n"] == 3


def test_paired_seed_and_pareto_outputs_have_frozen_shapes_and_roles() -> None:
    config = load_summary_config()
    seed = _synthetic_seed_summary(config)
    candidate = build_candidate_summary(seed, config.raw["candidates"], FORMAL_SEEDS, _synthetic_engineering(config))
    paired = build_paired_seed_directions(
        seed, EXPECTED_CANDIDATES, config.raw["primary_metrics"], FORMAL_SEEDS
    )
    assert len(paired) == 50
    assert (paired["left_raw_better_count"] + paired["right_raw_better_count"] + paired["raw_equal_count"]).eq(3).all()
    assert (
        paired["left_materially_better_count"]
        + paired["right_materially_better_count"]
        + paired["within_tolerance_count"]
    ).eq(3).all()
    dominance, quality, efficiency = build_pareto_outputs(
        candidate, config.raw["primary_metrics"], config.raw["quality_efficiency_metrics"]
    )
    assert len(dominance) == 40
    assert len(quality) == len(efficiency) == 5
    assert set(quality["candidate_id"]) == set(EXPECTED_CANDIDATES)
    decision = _decision_payload(config, candidate, quality, efficiency)
    assert decision["t0_role"] == "trunk_attribution_control"
    assert decision["research_test_opened"] is False
    assert decision["weighted_score_used"] is False


def _synthetic_metrics() -> pd.DataFrame:
    size = summary.EXPECTED_VALIDATION_ROWS
    rows: dict[str, object] = {}
    for column in summary.EXPECTED_METRICS_COLUMNS:
        rows[column] = np.ones(size, dtype=np.float64)
    rows.update(
        {
            "evaluation_split": ["validation"] * size,
            "method": [EXPECTED_CANDIDATES[0]] * size,
            "dataset_row_id": np.arange(size, dtype=np.int64),
            "split": ["val"] * size,
            "input_set": ["research_v2_waveform"] * size,
            "samp_id": np.arange(size, dtype=np.int64) % summary.EXPECTED_VALIDATION_SAMP_IDS,
            "coupling_state_id": np.arange(size, dtype=np.int64) % 17,
            "envelope_target_stratum": ["medium"] * size,
            "whole_rr_target_eligible": [True] * size,
            "local_rr_target_eligible": [True] * size,
            "envelope_spearman_target_eligible": [True] * size,
            "envelope_spearman_prediction_degenerate": [False] * size,
            "joint_target_eligible": [True] * size,
            "joint_prediction_degenerate": [False] * size,
            "ibi_interpretable": [True] * size,
            "ibi_target_eligible": [True] * size,
        }
    )
    return pd.DataFrame(rows, columns=summary.EXPECTED_METRICS_COLUMNS)


def test_validation_metrics_schema_identity_and_primary_finite_are_fail_closed(monkeypatch) -> None:
    metrics = _synthetic_metrics()
    monkeypatch.setattr(summary, "EXPECTED_VALIDATION_ROW_HASH", summary._row_ids_sha256(metrics))
    static = _validate_metrics(metrics, candidate_id=EXPECTED_CANDIDATES[0], reference_static=None)
    assert len(static) == summary.EXPECTED_VALIDATION_ROWS
    bad = metrics.copy()
    bad.loc[0, "method"] = "wrong"
    with pytest.raises(RuntimeError, match="identity"):
        _validate_metrics(bad, candidate_id=EXPECTED_CANDIDATES[0], reference_static=None)
    bad = metrics.copy()
    bad.loc[0, "local_rr_mae_bpm"] = np.inf
    with pytest.raises(FloatingPointError, match="primary"):
        _validate_metrics(bad, candidate_id=EXPECTED_CANDIDATES[0], reference_static=None)


def _finite_record(null: int = 0) -> dict[str, int]:
    return {"numeric_total": 10, "numeric_finite": 10 - null, "numeric_null": null, "numeric_nonfinite": 0}


def _valid_summary_receipt() -> dict:
    component_names = {
        "seed_summary",
        "candidate_summary",
        "paired_seed_directions",
        "dominance_audit",
        "quality_pareto",
        "quality_efficiency_pareto",
        "supplementary_metrics",
    }
    components = {name: _finite_record(1 if name == "supplementary_metrics" else 0) for name in component_names}
    aggregate = {
        key: sum(record[key] for record in components.values())
        for key in ("numeric_total", "numeric_finite", "numeric_null", "numeric_nonfinite")
    }
    inputs = []
    for candidate in EXPECTED_CANDIDATES:
        for seed in FORMAL_SEEDS:
            inputs.append(
                {
                    "candidate_id": candidate,
                    "seed": seed,
                    "execution_group": "gpu_0",
                    "cuda_visible_devices": "0",
                    "formal_receipt_path": "runs/formal_receipt.json",
                    "formal_receipt_sha256": "a" * 64,
                    "lifecycle_status": "complete",
                    "artifact_manifest_sha256": "b" * 64,
                    "validation_metrics_sha256": "c" * 64,
                    "validation_summary_sha256": "d" * 64,
                    "validation_rows": 2675,
                    "validation_samp_ids": 7,
                    "checkpoint_content_read": False,
                }
            )
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "created_utc": "2026-08-21T00:00:00+00:00",
        "status": "complete",
        "evidence_label": "validation-development evidence",
        "execution": {
            "command": "manual command",
            "cwd": str(summary.REPO_ROOT),
            "git_commit": "a" * 40,
            "git_dirty": False,
            "python_version": "3.12",
            "platform": "linux",
            "dependencies": {},
            "config_path": "configs/resp_temporal_v1/validation_summary_v1.yaml",
            "config_sha256": FROZEN_CONFIG_SHA256,
            "provenance": {},
        },
        "inputs": inputs,
        "counts": {
            "expected_formal_runs": 15,
            "actual_formal_runs": 15,
            "validation_rows_per_run": 2675,
            "total_validation_metric_rows_read": 40125,
            "seed_summary_rows": 15,
            "candidate_summary_rows": 5,
            "paired_seed_direction_rows": 50,
            "dominance_audit_rows": 40,
            "quality_pareto_rows": 5,
            "quality_efficiency_pareto_rows": 5,
            "supplementary_rows": 5,
        },
        "finite_audit": {"components": components, "aggregate": aggregate},
        "access": {
            "formal_receipts_accessed": True,
            "formal_lifecycle_and_manifest_accessed": True,
            "validation_metrics_accessed": True,
            "validation_stored_summary_accessed": True,
            "gpu_engineering_artifacts_accessed": True,
            "checkpoint_metadata_records_read": True,
            "checkpoint_content_read": False,
            "dataset_or_index_accessed": False,
            "signal_or_target_array_accessed": False,
            "model_training_used": False,
            "model_inference_used": False,
            "gpu_used": False,
            "research_test_accessed": False,
        },
        "decision": {
            "quality_pareto_set": [EXPECTED_CANDIDATES[0]],
            "quality_efficiency_pareto_set": [EXPECTED_CANDIDATES[0]],
            "stop_line": "at_least_one_trunk_material_primary_improvement",
            "secondary_used_for_quality_selection": False,
            "research_test_opened": False,
        },
        "artifacts": [
            {"filename": name, "sha256": "e" * 64, "size_bytes": 1, "rows": None}
            for name in OUTPUT_FILENAMES
        ],
        "manifest_sha256": "e" * 64,
    }


def test_summary_receipt_rejects_access_count_finite_and_artifact_drift() -> None:
    validate_summary_receipt(_valid_summary_receipt())
    bad = _valid_summary_receipt()
    bad["access"]["research_test_accessed"] = True
    with pytest.raises(ValueError, match="access"):
        validate_summary_receipt(bad)
    bad = _valid_summary_receipt()
    bad["counts"]["actual_formal_runs"] = 14
    with pytest.raises(ValueError, match="count"):
        validate_summary_receipt(bad)
    bad = _valid_summary_receipt()
    bad["finite_audit"]["components"]["seed_summary"]["numeric_nonfinite"] = 1
    with pytest.raises(ValueError, match="finite"):
        validate_summary_receipt(bad)


def test_existing_summary_output_is_rejected_before_provenance_or_inputs(monkeypatch, tmp_path: Path) -> None:
    frozen = load_summary_config()
    output = tmp_path / "existing"
    output.mkdir()
    config = SummaryConfig(path=frozen.path, sha256=frozen.sha256, raw=frozen.raw, output_dir=output)
    monkeypatch.setattr(summary, "load_summary_config", lambda _: config)
    monkeypatch.setattr(
        summary,
        "_verify_provenance",
        lambda *_: (_ for _ in ()).throw(AssertionError("must not verify provenance")),
    )
    with pytest.raises(FileExistsError, match="禁止覆盖"):
        run_validation_summary(config_path=frozen.path, command="unused")


def test_summary_cli_help_has_no_result_or_access_overrides() -> None:
    result = subprocess.run(
        [sys.executable, str(summary.REPO_ROOT / "scripts/summarize_resp_temporal_v1.py"), "--help"],
        cwd=summary.REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--config" in result.stdout
    for forbidden in ("--output", "--candidate", "--seed", "--checkpoint", "--split", "--test"):
        assert forbidden not in result.stdout
