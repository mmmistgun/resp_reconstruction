from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import resp_train.temporal.formal as formal
from resp_train.data.factory import ThoDataBundle, WindowDataBundle
from resp_train.temporal.formal import (
    EXPECTED_CANDIDATES,
    FORMAL_PROTOCOL_ID,
    FORMAL_RECEIPT_SCHEMA_VERSION,
    FROZEN_PLAN_SHA256,
    FormalPlan,
    FormalRunSpec,
    _checkpoint_finite_audit,
    _combined_finite_audit,
    _dataframe_finite_audit,
    _validate_data_contract,
    derive_formal_config,
    load_formal_plan,
    preflight_formal_run,
    run_formal_training,
    strict_selector_improved,
    validate_formal_plan,
    validate_formal_receipt,
)


def test_frozen_formal_plan_is_exact_and_has_15_run_matrix() -> None:
    plan = load_formal_plan()
    assert plan.sha256 == FROZEN_PLAN_SHA256
    assert tuple(row["candidate_id"] for row in plan.raw["candidates"]) == EXPECTED_CANDIDATES
    assert plan.raw["seeds"] == [20260811, 20260812, 20260813]
    assert len(plan.raw["candidates"]) * len(plan.raw["seeds"]) == 15
    assert [group["group_id"] for group in plan.raw["execution_groups"]] == ["gpu_0", "gpu_1"]
    assert [len(group["runs"]) for group in plan.raw["execution_groups"]] == [8, 7]
    assert [str(group["cuda_visible_devices"]) for group in plan.raw["execution_groups"]] == ["0", "1"]
    assert plan.raw["data"]["expected_train_windows"] == 10141
    assert plan.raw["data"]["expected_validation_windows"] == 2675
    assert plan.raw["authorization"]["research_test_evaluation"] is False
    protocol_path = formal.REPO_ROOT / plan.raw["provenance"]["formal_protocol_path"]
    assert formal.sha256_file(protocol_path) == plan.raw["provenance"]["formal_protocol_sha256"]


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("authorization", "research_test_evaluation", True),
        ("data", "expected_validation_windows", 2674),
        ("training", "epochs", 79),
        ("selector", "comparison", "less_than_or_equal"),
        ("output", "allow_overwrite", True),
    ],
)
def test_formal_plan_rejects_contract_drift(section: str, key: str, value: object) -> None:
    raw = copy.deepcopy(load_formal_plan().raw)
    raw[section][key] = value
    with pytest.raises(ValueError):
        validate_formal_plan(raw)


def test_formal_plan_rejects_dual_gpu_group_drift_or_duplicate_identity() -> None:
    raw = copy.deepcopy(load_formal_plan().raw)
    raw["execution_groups"][0]["cuda_visible_devices"] = "1"
    with pytest.raises(ValueError, match="dual-GPU"):
        validate_formal_plan(raw)
    raw = copy.deepcopy(load_formal_plan().raw)
    raw["execution_groups"][1]["runs"][0] = copy.deepcopy(raw["execution_groups"][0]["runs"][0])
    with pytest.raises(ValueError, match="dual-GPU|15-run"):
        validate_formal_plan(raw)


def test_derive_formal_config_closes_all_15_identities_without_data_or_gpu() -> None:
    plan = load_formal_plan()
    observed = []
    for candidate_id in EXPECTED_CANDIDATES:
        for seed in plan.raw["seeds"]:
            cfg, record = derive_formal_config(plan, candidate_id, seed)
            observed.append((candidate_id, seed))
            assert record["candidate_id"] == candidate_id
            assert cfg.model.variant == candidate_id
            assert cfg.model.initialization_seed == cfg.training.seed == seed
            assert cfg.training.batch_size == cfg.training.effective_batch_size == 128
            assert cfg.training.gradient_accumulation_steps == 1
            assert cfg.training.epochs == 80
            assert cfg.training.optimizer_updates == 6400
            assert cfg.window.target_fs == 100
            assert cfg.protocol.formal_training_enabled is True
            assert cfg.protocol.validation_evaluation_enabled is True
            assert cfg.protocol.research_test_enabled is False
            group = formal.formal_execution_group(plan, candidate_id, seed)
            assert cfg.formal.execution_group.group_id == group["group_id"]
            assert str(cfg.formal.execution_group.cuda_visible_devices) == group["cuda_visible_devices"]
            assert str(cfg.outputs.run_root).endswith(f"{candidate_id}/seed_{seed}")
    assert len(observed) == 15
    with pytest.raises(ValueError, match="allowlist"):
        derive_formal_config(plan, "unknown", 20260811)
    with pytest.raises(ValueError, match="allowlist"):
        derive_formal_config(plan, EXPECTED_CANDIDATES[0], 1)


def test_strict_selector_uses_strict_less_than_and_rejects_nonfinite() -> None:
    assert strict_selector_improved(1.0, float("inf")) is True
    assert strict_selector_improved(0.9, 1.0) is True
    assert strict_selector_improved(1.0, 1.0) is False
    assert strict_selector_improved(1.1, 1.0) is False
    with pytest.raises(ValueError, match="finite"):
        strict_selector_improved(float("nan"), 1.0)


class _Sized:
    def __init__(self, size: int):
        self.size = size

    def __len__(self) -> int:
        return self.size


def _window_bundle(rows: pd.DataFrame, *, batches: int) -> WindowDataBundle:
    return WindowDataBundle(
        index_path=Path("/unused/index.csv"),
        rows=rows,
        dataset=_Sized(len(rows)),  # type: ignore[arg-type]
        loader=_Sized(batches),  # type: ignore[arg-type]
        audited=rows,
        audit_summary=pd.DataFrame({"n": [len(rows)]}),
    )


def test_data_contract_checks_counts_hashes_and_split_isolation() -> None:
    train_rows = pd.DataFrame({"dataset_row_id": [10, 20], "samp_id": [1, 2], "split": ["train", "train"]})
    val_rows = pd.DataFrame({"dataset_row_id": [30], "samp_id": [3], "split": ["val"]})
    audited = pd.concat([train_rows, val_rows], ignore_index=True)
    bundle = ThoDataBundle(
        train=_window_bundle(train_rows, batches=1),
        val=_window_bundle(val_rows, batches=1),
        audited=audited,
        audit_summary=pd.DataFrame({"split": ["train", "val"]}),
    )
    frozen = load_formal_plan()
    raw = copy.deepcopy(frozen.raw)
    raw["data"].update(
        {
            "expected_train_windows": 2,
            "expected_validation_windows": 1,
            "expected_train_samp_ids": 2,
            "expected_validation_samp_ids": 1,
            "expected_train_batches": 1,
            "expected_validation_batches": 1,
            "expected_train_row_ids_sha256": formal._row_id_sha256(train_rows),
            "expected_validation_row_ids_sha256": formal._row_id_sha256(val_rows),
        }
    )
    plan = FormalPlan(path=frozen.path, sha256=frozen.sha256, raw=raw)
    receipt = _validate_data_contract(bundle, plan)
    assert receipt["signal_splits_accessed"] == ["train", "val"]
    assert receipt["row_id_overlap_count"] == 0

    overlapping = val_rows.copy()
    overlapping.loc[0, "dataset_row_id"] = 20
    bad = ThoDataBundle(
        train=bundle.train,
        val=_window_bundle(overlapping, batches=1),
        audited=pd.concat([train_rows, overlapping], ignore_index=True),
        audit_summary=bundle.audit_summary,
    )
    raw_bad = copy.deepcopy(raw)
    raw_bad["data"]["expected_validation_row_ids_sha256"] = formal._row_id_sha256(overlapping)
    with pytest.raises(RuntimeError, match="重叠"):
        _validate_data_contract(bad, FormalPlan(path=frozen.path, sha256=frozen.sha256, raw=raw_bad))


def test_existing_output_is_rejected_before_provenance_or_cuda(monkeypatch, tmp_path: Path) -> None:
    output = tmp_path / "already_exists"
    output.mkdir()
    monkeypatch.setattr(formal, "formal_run_dir", lambda *_: output)
    monkeypatch.setattr(
        formal,
        "_verify_formal_provenance",
        lambda *_: (_ for _ in ()).throw(AssertionError("must not reach provenance")),
    )
    with pytest.raises(FileExistsError, match="禁止覆盖"):
        preflight_formal_run(
            plan_path=formal.DEFAULT_PLAN_PATH,
            candidate_id=EXPECTED_CANDIDATES[0],
            seed=20260811,
        )


def test_dual_gpu_group_rejects_missing_or_wrong_visibility_before_provenance(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(formal, "formal_run_dir", lambda *_: tmp_path / "new_run")
    monkeypatch.setattr(
        formal,
        "_verify_formal_provenance",
        lambda *_: (_ for _ in ()).throw(AssertionError("must not reach provenance")),
    )
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES"):
        preflight_formal_run(
            plan_path=formal.DEFAULT_PLAN_PATH,
            candidate_id=EXPECTED_CANDIDATES[0],
            seed=20260811,
        )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(RuntimeError, match="expected='0'.*actual='1'"):
        preflight_formal_run(
            plan_path=formal.DEFAULT_PLAN_PATH,
            candidate_id=EXPECTED_CANDIDATES[0],
            seed=20260811,
        )


def test_failure_keeps_fail_closed_lifecycle_without_data_or_gpu(monkeypatch, tmp_path: Path) -> None:
    plan = load_formal_plan()
    cfg, record = derive_formal_config(plan, EXPECTED_CANDIDATES[0], 20260811)
    run_dir = tmp_path / "formal_failure"
    spec = FormalRunSpec(
        plan=plan,
        candidate_record=record,
        seed=20260811,
        cfg=cfg,
        run_dir=run_dir,
        git_commit="a" * 40,
        gpu_device_info={
            "device": "cuda:0",
            "execution_group_id": "gpu_0",
            "cuda_visible_devices": "0",
            "logical_device_after_visibility_filter": "cuda:0",
        },
        execution_group={"group_id": "gpu_0", "cuda_visible_devices": "0", "logical_device": "cuda:0"},
    )
    monkeypatch.setattr(formal, "preflight_formal_run", lambda **_: spec)
    monkeypatch.setattr(formal, "save_config", lambda *_: (_ for _ in ()).throw(RuntimeError("stop")))
    monkeypatch.setattr(
        formal,
        "_build_formal_data",
        lambda *_: (_ for _ in ()).throw(AssertionError("data must not be read")),
    )
    with pytest.raises(RuntimeError, match="stop"):
        run_formal_training(
            plan_path=formal.DEFAULT_PLAN_PATH,
            candidate_id=EXPECTED_CANDIDATES[0],
            seed=20260811,
            command="manual command",
        )
    lifecycle = json.loads((run_dir / "lifecycle.json").read_text(encoding="utf-8"))
    assert lifecycle["status"] == "failed"
    assert lifecycle["last_completed_epoch"] == 0
    assert lifecycle["failure"]["error_type"] == "RuntimeError"
    assert not (run_dir / "formal_receipt.json").exists()


def _finite_record(*, null: int = 0) -> dict[str, int]:
    return {"numeric_total": 10, "numeric_finite": 10 - null, "numeric_null": null, "numeric_nonfinite": 0}


def _valid_receipt() -> dict:
    artifact_names = [
        "config.yaml",
        "formal_plan.json",
        "run_manifest.json",
        "data_receipt.json",
        "audit.csv",
        "optimizer_parameter_groups.json",
        "train_history.csv",
        "checkpoint_best_local_rr.pt",
        "checkpoint_final.pt",
        "validation_metrics.csv",
        "validation_metrics_summary.csv",
        "runtime_summary.json",
        "artifact_manifest.json",
    ]
    components = {
        "train_history": _finite_record(),
        "validation_metrics": _finite_record(null=2),
        "validation_metrics_summary": _finite_record(null=1),
        "checkpoint_best_local_rr": _finite_record(),
        "checkpoint_final": _finite_record(),
    }
    aggregate = {
        key: sum(record[key] for record in components.values())
        for key in ("numeric_total", "numeric_finite", "numeric_null", "numeric_nonfinite")
    }
    return {
        "schema_version": FORMAL_RECEIPT_SCHEMA_VERSION,
        "protocol_id": FORMAL_PROTOCOL_ID,
        "created_utc": "2026-08-20T00:00:00+00:00",
        "status": "complete",
        "evidence_label": "validation-development evidence",
        "candidate_id": EXPECTED_CANDIDATES[0],
        "seed": 20260811,
        "execution": {
            "command": "manual command",
            "cwd": str(formal.REPO_ROOT),
            "git_commit": "a" * 40,
            "git_dirty": False,
            "plan_path": "configs/resp_temporal_v1/formal_dual_gpu_v2.yaml",
            "plan_sha256": FROZEN_PLAN_SHA256,
            "execution_group_id": "gpu_0",
            "cuda_visible_devices": "0",
            "dependencies": {},
            "device": {
                "device": "cuda:0",
                "execution_group_id": "gpu_0",
                "cuda_visible_devices": "0",
                "logical_device_after_visibility_filter": "cuda:0",
            },
        },
        "data": {
            "train_windows": 10141,
            "validation_windows": 2675,
            "train_batches_per_epoch": 80,
            "validation_batches_per_epoch": 21,
            "train_samp_ids": 32,
            "validation_samp_ids": 7,
            "train_row_ids_sha256": "f290e569140a2ff7745cf1a5cfa6a4da943644d76498c9b85517d3ae0702c45e",
            "validation_row_ids_sha256": "b68a51b101bb80033c4de18c9c21cdfbb8924d1bfe134330f047617cf3b0915a",
            "row_id_overlap_count": 0,
            "shared_index_metadata_read": True,
            "signal_splits_accessed": ["train", "val"],
            "research_test_accessed": False,
        },
        "training": {
            "planned_epochs": 80,
            "completed_epochs": 80,
            "planned_optimizer_updates": 6400,
            "completed_optimizer_updates": 6400,
            "physical_batch_size": 128,
            "accumulation_steps": 1,
            "effective_batch_size": 128,
            "early_stopping": False,
            "resume": False,
        },
        "selector": {
            "metric": "local_rr_mae_full_split_sample_direct_mean",
            "comparison": "strict_less_than",
            "best_epoch": 10,
            "best_validation_local_rr_mae": 1.0,
            "reloaded_best_validation_local_rr_mae": 1.0,
            "validation_eligible_samples": 2675,
        },
        "access": {
            "shared_index_metadata_read": True,
            "train_signal_target_accessed": True,
            "validation_signal_target_accessed": True,
            "validation_prediction_generated": True,
            "research_test_accessed": False,
            "checkpoint_input_accessed": False,
            "own_best_checkpoint_reloaded": True,
            "gpu_used": True,
        },
        "counts": {
            "expected_epochs": 80,
            "completed_epochs": 80,
            "expected_optimizer_updates": 6400,
            "completed_optimizer_updates": 6400,
            "validation_metrics_rows": 2675,
            "validation_summary_rows": 1,
            "artifact_records": 13,
        },
        "finite_audit": {"components": components, "aggregate": aggregate},
        "runtime": {
            "elapsed_seconds": 1.0,
            "device": "cuda:0",
            "device_name": "NVIDIA GeForce RTX 4070 Ti SUPER",
            "total_memory_mib": 16384.0,
            "peak_allocated_mib": 100.0,
            "peak_reserved_mib": 120.0,
        },
        "artifacts": [
            {"filename": name, "sha256": "b" * 64, "size_bytes": 1, "rows": None}
            for name in artifact_names
        ],
        "manifest_sha256": "c" * 64,
    }


def test_formal_receipt_schema_closes_access_counts_finite_and_hashes() -> None:
    validate_formal_receipt(_valid_receipt())
    bad = _valid_receipt()
    bad["access"]["research_test_accessed"] = True
    with pytest.raises(ValueError, match="access"):
        validate_formal_receipt(bad)
    bad = _valid_receipt()
    bad["finite_audit"]["components"]["checkpoint_final"]["numeric_nonfinite"] = 1
    with pytest.raises(ValueError, match="finite"):
        validate_formal_receipt(bad)
    bad = _valid_receipt()
    bad["counts"]["validation_metrics_rows"] = 2674
    with pytest.raises(ValueError, match="count"):
        validate_formal_receipt(bad)
    bad = _valid_receipt()
    bad["execution"]["cuda_visible_devices"] = "1"
    with pytest.raises(ValueError, match="dual-GPU"):
        validate_formal_receipt(bad)


def test_finite_helpers_distinguish_null_from_nonfinite_and_audit_checkpoint(tmp_path: Path) -> None:
    frame = pd.DataFrame({"finite": [1.0, 2.0], "nullable": [np.nan, 3.0]})
    frame_record = _dataframe_finite_audit(frame)
    assert frame_record == {
        "numeric_total": 4,
        "numeric_finite": 3,
        "numeric_null": 1,
        "numeric_nonfinite": 0,
    }
    checkpoint = tmp_path / "synthetic.pt"
    torch.save({"model_state_dict": {"weight": torch.ones(2)}}, checkpoint)
    checkpoint_record = _checkpoint_finite_audit(checkpoint)
    assert checkpoint_record["numeric_total"] == checkpoint_record["numeric_finite"] == 2
    with pytest.raises(FloatingPointError, match="Inf"):
        _combined_finite_audit({"bad": _dataframe_finite_audit(pd.DataFrame({"x": [float("inf")]}))})


def test_formal_cli_help_exposes_only_plan_candidate_and_seed() -> None:
    result = subprocess.run(
        [sys.executable, str(formal.REPO_ROOT / "scripts/train_resp_temporal_v1.py"), "--help"],
        cwd=formal.REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--plan" in result.stdout
    assert "--candidate-id" in result.stdout
    assert "--seed" in result.stdout
    for forbidden in ("--device", "--batch-size", "--epochs", "--output", "--resume", "--all"):
        assert forbidden not in result.stdout
