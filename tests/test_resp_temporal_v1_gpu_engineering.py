from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

import resp_train.temporal.gpu_engineering as engineering
from resp_train.temporal.gpu_engineering import (
    EXPECTED_CANDIDATES,
    FROZEN_CONFIG_SHA256,
    GPUConfig,
    PROTOCOL_ID,
    RECEIPT_SCHEMA_VERSION,
    _finalize_output,
    _is_cuda_oom,
    _summary_from_detail,
    load_gpu_engineering_config,
    make_synthetic_batch,
    run_gpu_engineering,
    sha256_file,
    summarize_latencies,
    validate_access_receipt,
    validate_gpu_engineering_config,
)


def test_frozen_gpu_engineering_config_is_exact_and_non_scientific() -> None:
    config = load_gpu_engineering_config()
    assert config.sha256 == FROZEN_CONFIG_SHA256
    assert tuple(item["candidate_id"] for item in config.raw["candidates"]) == EXPECTED_CANDIDATES
    assert config.raw["scientific_evidence"] is False
    assert config.raw["authorization"] == {
        "gpu_engineering": True,
        "formal_training": False,
        "validation_evaluation": False,
        "research_test_evaluation": False,
    }
    assert config.raw["benchmark"]["physical_batch_schemes"] == [
        {"physical_batch_size": 128, "accumulation_steps": 1},
        {"physical_batch_size": 64, "accumulation_steps": 2},
        {"physical_batch_size": 32, "accumulation_steps": 4},
    ]
    for key in ("protocol", "implementation_receipt"):
        relative = config.raw["provenance"][f"{key}_path"]
        assert sha256_file(engineering.REPO_ROOT / relative) == config.raw["provenance"][f"{key}_sha256"]
    for candidate in config.raw["candidates"]:
        assert sha256_file(engineering.REPO_ROOT / candidate["config_path"]) == candidate["config_sha256"]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("authorization", "formal_training"), True),
        (("device", "allow_tf32"), True),
        (("benchmark", "effective_batch_size"), 64),
        (("benchmark", "planning_wall_time_multiplier"), 1.0),
        (("output", "allow_overwrite"), True),
    ],
)
def test_gpu_engineering_config_rejects_contract_drift(path: tuple[str, str], value) -> None:
    raw = copy.deepcopy(load_gpu_engineering_config().raw)
    raw[path[0]][path[1]] = value
    with pytest.raises(ValueError):
        validate_gpu_engineering_config(raw)


def test_gpu_engineering_config_rejects_candidate_reorder_and_unknown_fields() -> None:
    raw = copy.deepcopy(load_gpu_engineering_config().raw)
    raw["candidates"][0], raw["candidates"][1] = raw["candidates"][1], raw["candidates"][0]
    with pytest.raises(ValueError, match="candidate"):
        validate_gpu_engineering_config(raw)
    raw = copy.deepcopy(load_gpu_engineering_config().raw)
    raw["device"]["extra"] = 1
    with pytest.raises(ValueError, match="字段必须严格"):
        validate_gpu_engineering_config(raw)


def test_synthetic_contract_is_deterministic_finite_and_has_expected_shape() -> None:
    left_input, left_target = make_synthetic_batch(3, device=torch.device("cpu"))
    right_input, right_target = make_synthetic_batch(3, device=torch.device("cpu"))
    assert left_input.shape == left_target.shape == (3, 1, 18000)
    torch.testing.assert_close(left_input, right_input, rtol=0.0, atol=0.0)
    torch.testing.assert_close(left_target, right_target, rtol=0.0, atol=0.0)
    assert torch.isfinite(left_input).all()
    assert torch.isfinite(left_target).all()
    assert not torch.equal(left_input, left_target)


def test_latency_summary_and_oom_classification_are_fail_closed() -> None:
    summary = summarize_latencies([1.0, 2.0, 3.0, 4.0])
    assert summary["p50_ms"] == 2.5
    assert summary["p90_ms"] == pytest.approx(3.7)
    assert summary["repeat_count"] == 4
    with pytest.raises(ValueError):
        summarize_latencies([1.0, float("inf")])
    assert _is_cuda_oom(RuntimeError("CUDA out of memory")) is True
    assert _is_cuda_oom(RuntimeError("unrelated runtime failure")) is False


def test_candidate_summary_keeps_engineering_role_and_wall_time_fields() -> None:
    detail = {
        "candidate_id": EXPECTED_CANDIDATES[0],
        "family": "no_temporal_trunk",
        "trainable_parameters": 59042,
        "status": "passed",
        "failure_stage": None,
        "oom_attempts": [],
        "standard": {
            "batch1_inference": {"p50_ms": 1.0, "p90_ms": 1.2},
            "fixed_forward": {"p50_ms": 2.0, "p90_ms": 2.2, "peak_allocated_mib": 10.0, "peak_reserved_mib": 12.0},
            "fixed_forward_backward": {
                "p50_ms": 4.0,
                "p90_ms": 4.5,
                "peak_allocated_mib": 20.0,
                "peak_reserved_mib": 24.0,
            },
            "fixed_batch_size": 8,
        },
        "selected_update_scheme": {
            "physical_batch_size": 64,
            "accumulation_steps": 2,
            "effective_batch_size": 128,
            "timing": {"p50_ms": 1000.0, "p90_ms": 1200.0, "peak_allocated_mib": 100.0, "peak_reserved_mib": 120.0},
            "throughput_windows_per_second": 128.0,
            "compute_only_6400_updates_hours": 6400 / 3600,
            "planning_6400_updates_hours": 6400 / 3600 * 1.25,
        },
    }
    summary = _summary_from_detail(detail)
    assert summary["status"] == "passed"
    assert summary["physical_batch_size"] == 64
    assert summary["update_p50_seconds"] == 1.0
    assert summary["fixed_peak_reserved_mib"] == 24.0


def _valid_receipt() -> dict:
    candidate_summary = []
    for candidate_id in EXPECTED_CANDIDATES:
        row = {field: None for field in engineering.SUMMARY_FIELDS}
        row.update(
            {
                "candidate_id": candidate_id,
                "family": "family",
                "trainable_parameters": 1,
                "status": "passed",
                "fixed_batch_size": 8,
                "effective_batch_size": 128,
                "oom_attempt_count": 0,
            }
        )
        candidate_summary.append(row)
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "created_utc": "2026-08-20T00:00:00+00:00",
        "status": "complete",
        "scientific_evidence": False,
        "execution": {
            "command": "python benchmark.py",
            "cwd": "/repo",
            "git_commit": "a" * 40,
            "git_dirty": False,
            "python_version": "3.12",
            "platform": "linux",
            "dependencies": {},
            "config_path": "configs/resp_temporal_v1/gpu_engineering_v1.yaml",
            "config_sha256": FROZEN_CONFIG_SHA256,
            "provenance": {},
        },
        "device": {"name": "NVIDIA GeForce RTX 4070 Ti SUPER"},
        "benchmark_contract": {},
        "access": {
            "dataset_accessed": False,
            "index_accessed": False,
            "train_split_accessed": False,
            "validation_accessed": False,
            "validation_target_accessed": False,
            "validation_prediction_accessed": False,
            "research_test_accessed": False,
            "checkpoint_accessed": False,
            "formal_training_used": False,
            "validation_evaluation_used": False,
            "research_test_evaluation_used": False,
            "synthetic_tensor_used": True,
            "synthetic_training_like_update_used": True,
            "gpu_used": True,
        },
        "counts": {
            "expected_candidates": 5,
            "actual_candidates": 5,
            "passed_candidates": 5,
            "unavailable_candidates": 0,
        },
        "candidate_summary": candidate_summary,
        "finite_audit": {
            "numeric_total": 100,
            "numeric_finite": 100,
            "numeric_null": 0,
            "numeric_nonfinite": 0,
        },
        "artifacts": [
            {"filename": "candidate_engineering.csv", "sha256": "b" * 64, "size_bytes": 10, "rows": 5}
        ],
        "manifest_sha256": "c" * 64,
        "failure": None,
    }


def test_receipt_schema_rejects_access_nonfinite_and_incomplete_counts() -> None:
    validate_access_receipt(_valid_receipt())
    bad = _valid_receipt()
    bad["access"]["validation_accessed"] = True
    with pytest.raises(ValueError, match="越界"):
        validate_access_receipt(bad)
    bad = _valid_receipt()
    bad["finite_audit"]["numeric_nonfinite"] = 1
    with pytest.raises(ValueError, match="finite"):
        validate_access_receipt(bad)
    bad = _valid_receipt()
    bad["counts"]["actual_candidates"] = 4
    with pytest.raises(ValueError, match="五项"):
        validate_access_receipt(bad)


def test_existing_output_is_rejected_before_provenance_or_cuda(monkeypatch, tmp_path: Path) -> None:
    output = tmp_path / "already_exists"
    output.mkdir()
    frozen = load_gpu_engineering_config()
    config = GPUConfig(path=frozen.path, sha256=frozen.sha256, raw=frozen.raw, output_dir=output)
    monkeypatch.setattr(engineering, "load_gpu_engineering_config", lambda _: config)
    with pytest.raises(FileExistsError, match="禁止覆盖"):
        run_gpu_engineering(config_path=frozen.path, command="unused")


def test_output_bundle_is_atomic_hashed_and_strict_without_cuda(tmp_path: Path) -> None:
    frozen = load_gpu_engineering_config()
    output = tmp_path / "published"
    config = GPUConfig(path=frozen.path, sha256=frozen.sha256, raw=frozen.raw, output_dir=output)
    temporary = tmp_path / ".temporary"
    temporary.mkdir()
    (temporary / "resolved_config.json").write_text("{}\n", encoding="utf-8")
    details = []
    for index, candidate_id in enumerate(EXPECTED_CANDIDATES):
        detail = {
            "candidate_id": candidate_id,
            "family": f"family_{index}",
            "trainable_parameters": index + 1,
            "status": "passed",
            "failure_stage": None,
            "oom_attempts": [],
            "standard": {
                "batch1_inference": {"p50_ms": 1.0, "p90_ms": 1.1},
                "fixed_forward": {
                    "p50_ms": 2.0,
                    "p90_ms": 2.1,
                    "peak_allocated_mib": 10.0,
                    "peak_reserved_mib": 12.0,
                },
                "fixed_forward_backward": {
                    "p50_ms": 3.0,
                    "p90_ms": 3.1,
                    "peak_allocated_mib": 20.0,
                    "peak_reserved_mib": 22.0,
                },
                "fixed_batch_size": 8,
            },
            "selected_update_scheme": {
                "physical_batch_size": 128,
                "accumulation_steps": 1,
                "effective_batch_size": 128,
                "timing": {
                    "p50_ms": 1000.0,
                    "p90_ms": 1100.0,
                    "peak_allocated_mib": 100.0,
                    "peak_reserved_mib": 120.0,
                },
                "throughput_windows_per_second": 128.0,
                "compute_only_6400_updates_hours": 1.0,
                "planning_6400_updates_hours": 1.25,
            },
        }
        detail["summary"] = _summary_from_detail(detail)
        details.append(detail)
    receipt_path = _finalize_output(
        temporary,
        config=config,
        command="manual command",
        provenance={"git_commit": "a" * 40, "git_dirty": False},
        device_info={"name": "NVIDIA GeForce RTX 4070 Ti SUPER"},
        details=details,
        failure=None,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    validate_access_receipt(receipt)
    assert receipt["status"] == "complete"
    assert receipt["counts"]["actual_candidates"] == 5
    assert (output / "access_receipt.sha256").is_file()
    assert (output / "artifact_manifest.json").is_file()
