from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

import resp_train.temporal.research_test as research_test
from resp_train.temporal.formal import EXPECTED_CANDIDATES, FORMAL_SEEDS
from resp_train.temporal.formal_summary import EXPECTED_METRICS_COLUMNS, PRIMARY_SAMPLE_COLUMNS


def _sha() -> str:
    return "0" * 64


def _checkpoint_inputs() -> list[dict[str, object]]:
    return [
        {
            "candidate_id": candidate,
            "seed": seed,
            "formal_receipt_path": f"runs/formal/{candidate}/seed_{seed}/formal_receipt.json",
            "formal_receipt_sha256": _sha(),
            "formal_artifact_manifest_sha256": _sha(),
            "config_path": f"runs/formal/{candidate}/seed_{seed}/config.yaml",
            "config_sha256": _sha(),
            "checkpoint_path": f"runs/formal/{candidate}/seed_{seed}/checkpoint_best_local_rr.pt",
            "checkpoint_sha256": _sha(),
            "selected_epoch": 1,
            "selector": "validation_local_rr_mae_full_split_strict_less_than",
        }
        for candidate in EXPECTED_CANDIDATES
        for seed in FORMAL_SEEDS
    ]


def _receipt() -> dict[str, object]:
    artifacts = [
        {"filename": filename, "sha256": _sha(), "size_bytes": 1, "rows": None}
        for filename in research_test.OUTPUT_ARTIFACTS
    ]
    components = {
        "research_test_metrics": {
            "numeric_total": 10,
            "numeric_finite": 9,
            "numeric_null": 1,
            "numeric_nonfinite": 0,
        },
        "research_test_seed_summary": {
            "numeric_total": 10,
            "numeric_finite": 10,
            "numeric_null": 0,
            "numeric_nonfinite": 0,
        },
        "research_test_primary_mean_sd": {
            "numeric_total": 10,
            "numeric_finite": 10,
            "numeric_null": 0,
            "numeric_nonfinite": 0,
        },
    }
    return {
        "schema_version": research_test.RECEIPT_SCHEMA_VERSION,
        "protocol_id": research_test.PROTOCOL_ID,
        "created_utc": "2026-08-21T00:00:00+00:00",
        "status": "complete",
        "evidence_label": research_test.EVIDENCE_LABEL,
        "execution": {
            "command": "python scripts/eval_resp_temporal_v1_research_test.py --config x",
            "cwd": str(research_test.REPO_ROOT),
            "git_commit": "0" * 40,
            "git_dirty": False,
            "config_path": "configs/resp_temporal_v1/research_test_v1.yaml",
            "config_sha256": research_test.FROZEN_CONFIG_SHA256,
            "python_version": "3.12.0",
            "platform": "test",
            "dependencies": {},
            "device": {},
            "provenance": {},
        },
        "inputs": _checkpoint_inputs(),
        "data": {
            "dataset_index_path": (
                "/mnt/disk_code/marques/resp_prepare/dataset/"
                "20260620_research_v2_resp_reconstruction_stage2_1_segrobustz_bcgstagee_log1psoftz_robustconf/"
                "training/dataset_index.csv"
            ),
            "dataset_index_sha256": "f65ae6524632187c7c2795bd7b5e25434afb649b5a04fe39084b70cc59f9b04f",
            "split": "test",
            "test_windows": 2310,
            "test_batches": 19,
            "test_samp_ids": 8,
            "test_row_ids_sha256": research_test.EXPECTED_TEST_ROW_HASH,
            "sample_strategy": "stratified_random",
            "sample_seed": 20260612,
            "row_id_overlap_with_non_test": 0,
            "signal_splits_accessed": ["test"],
        },
        "access": {
            "shared_index_metadata_read": True,
            "train_signal_target_accessed": False,
            "validation_signal_target_prediction_accessed": False,
            "research_test_signal_target_accessed": True,
            "research_test_prediction_generated": True,
            "formal_receipts_accessed": True,
            "validation_summary_receipt_accessed": True,
            "checkpoint_best_content_accessed": True,
            "checkpoint_final_accessed": False,
            "model_training_used": False,
            "model_inference_used": True,
            "gpu_used": True,
        },
        "counts": {
            "expected_checkpoints": 15,
            "completed_checkpoints": 15,
            "test_windows_per_checkpoint": 2310,
            "research_test_metrics_rows": 34650,
            "seed_summary_rows": 15,
            "candidate_summary_rows": 5,
            "artifact_records": len(research_test.OUTPUT_ARTIFACTS),
        },
        "finite_audit": {
            "components": components,
            "aggregate": {
                "numeric_total": 30,
                "numeric_finite": 29,
                "numeric_null": 1,
                "numeric_nonfinite": 0,
            },
        },
        "runtime": {
            "elapsed_seconds": 1.0,
            "device": "cuda:0",
            "device_name": "NVIDIA GeForce RTX 4070 Ti SUPER",
            "total_memory_mib": 16376.0,
            "peak_allocated_mib": 1.0,
            "peak_reserved_mib": 2.0,
        },
        "artifacts": artifacts,
        "manifest_sha256": _sha(),
    }


def _synthetic_primary_metrics() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate_index, candidate in enumerate(EXPECTED_CANDIDATES):
        for seed_index, seed in enumerate(FORMAL_SEEDS):
            for sample_index in range(2):
                row: dict[str, object] = {"candidate_id": candidate, "seed": seed}
                for metric_index, metric in enumerate(PRIMARY_SAMPLE_COLUMNS):
                    row[metric] = float(candidate_index + seed_index + sample_index + metric_index / 10)
                rows.append(row)
    return pd.DataFrame(rows)


def test_frozen_research_test_config_loads() -> None:
    config = research_test.load_research_test_config()
    assert config.sha256 == research_test.FROZEN_CONFIG_SHA256
    assert config.raw["candidates"] == list(EXPECTED_CANDIDATES)
    assert config.raw["evaluation"]["include_test_only_metrics"] is False


def test_research_test_config_rejects_extra_field() -> None:
    raw = deepcopy(research_test.load_research_test_config().raw)
    raw["unexpected"] = True
    with pytest.raises(ValueError, match="字段必须严格"):
        research_test.validate_research_test_config(raw)


def test_research_test_config_rejects_candidate_reorder() -> None:
    raw = deepcopy(research_test.load_research_test_config().raw)
    raw["candidates"] = list(reversed(raw["candidates"]))
    with pytest.raises(ValueError, match="candidate/seed"):
        research_test.validate_research_test_config(raw)


def test_primary_tables_use_direct_mean_and_sample_sd() -> None:
    seed_summary, candidate_summary = research_test.build_primary_tables(_synthetic_primary_metrics())
    assert len(seed_summary) == 15
    assert len(candidate_summary) == 5
    first_seed = seed_summary.iloc[0]
    assert first_seed[PRIMARY_SAMPLE_COLUMNS[0]] == pytest.approx(0.5)
    first_candidate = candidate_summary.iloc[0]
    assert first_candidate[f"{PRIMARY_SAMPLE_COLUMNS[0]}_mean"] == pytest.approx(1.5)
    assert first_candidate[f"{PRIMARY_SAMPLE_COLUMNS[0]}_sample_sd"] == pytest.approx(1.0)


def test_primary_tables_reject_missing_metric() -> None:
    metrics = _synthetic_primary_metrics().drop(columns=[PRIMARY_SAMPLE_COLUMNS[-1]])
    with pytest.raises(ValueError, match="缺少主指标"):
        research_test.build_primary_tables(metrics)


def test_primary_tables_reject_missing_seed() -> None:
    metrics = _synthetic_primary_metrics()
    metrics = metrics.loc[
        ~(metrics["candidate_id"].eq(EXPECTED_CANDIDATES[0]) & metrics["seed"].eq(FORMAL_SEEDS[0]))
    ]
    with pytest.raises(ValueError, match="identity不完整"):
        research_test.build_primary_tables(metrics)


def test_finite_audit_allows_nan_but_rejects_inf() -> None:
    frames = {
        "research_test_metrics": pd.DataFrame({"x": [1.0, np.nan]}),
        "research_test_seed_summary": pd.DataFrame({"x": [1.0]}),
        "research_test_primary_mean_sd": pd.DataFrame({"x": [1.0]}),
    }
    audit = research_test._finite_audit(frames)
    assert audit["aggregate"]["numeric_null"] == 1
    frames["research_test_metrics"] = pd.DataFrame({"x": [np.inf]})
    with pytest.raises(FloatingPointError, match="Inf"):
        research_test._finite_audit(frames)


def test_research_test_receipt_schema_accepts_complete_fixture() -> None:
    research_test.validate_research_test_receipt(_receipt())


def test_research_test_receipt_rejects_training_access() -> None:
    receipt = _receipt()
    receipt["access"]["model_training_used"] = True
    with pytest.raises(ValueError, match="access"):
        research_test.validate_research_test_receipt(receipt)


def test_research_test_receipt_rejects_partial_checkpoint_count() -> None:
    receipt = _receipt()
    receipt["counts"]["completed_checkpoints"] = 14
    with pytest.raises(ValueError, match="count"):
        research_test.validate_research_test_receipt(receipt)


def test_research_test_receipt_rejects_nonfinite_summary() -> None:
    receipt = _receipt()
    component = receipt["finite_audit"]["components"]["research_test_primary_mean_sd"]
    component.update(numeric_finite=9, numeric_nonfinite=1)
    receipt["finite_audit"]["aggregate"].update(numeric_finite=28, numeric_nonfinite=1)
    with pytest.raises(ValueError, match="finite"):
        research_test.validate_research_test_receipt(receipt)


def test_run_metrics_requires_test_only_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    row_ids = np.arange(2310, dtype=np.int64)
    monkeypatch.setattr(
        research_test,
        "EXPECTED_TEST_ROW_HASH",
        __import__("hashlib").sha256(row_ids.tobytes()).hexdigest(),
    )
    values: dict[str, object] = {}
    for column in EXPECTED_METRICS_COLUMNS:
        if column == "evaluation_split":
            values[column] = ["research_test"] * 2310
        elif column == "method":
            values[column] = [EXPECTED_CANDIDATES[0]] * 2310
        elif column == "dataset_row_id":
            values[column] = row_ids
        elif column == "split":
            values[column] = ["test"] * 2310
        elif column in {"input_set", "coupling_state_id", "envelope_target_stratum"}:
            values[column] = ["x"] * 2310
        elif column == "samp_id":
            values[column] = np.zeros(2310, dtype=np.int64)
        else:
            values[column] = np.ones(2310, dtype=np.float64)
    metrics = pd.DataFrame(values)
    metrics.insert(0, "seed", FORMAL_SEEDS[0])
    metrics.insert(0, "candidate_id", EXPECTED_CANDIDATES[0])
    research_test._validate_run_metrics(metrics, candidate=EXPECTED_CANDIDATES[0], seed=FORMAL_SEEDS[0])
    metrics.loc[0, "split"] = "val"
    with pytest.raises(RuntimeError, match="identity"):
        research_test._validate_run_metrics(metrics, candidate=EXPECTED_CANDIDATES[0], seed=FORMAL_SEEDS[0])
