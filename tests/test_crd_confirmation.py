from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

from resp_train.crd import candidate_lock
from resp_train.crd.candidate_lock import CandidateLockError, CandidateLockVerification, sha256_file
from resp_train.crd.confirmation import (
    ACCESS_RECEIPT_FILENAME,
    MANIFEST_FILENAME,
    METRICS_FILENAME,
    S1C_CANDIDATE_LOCK_SHA256,
    SUMMARY_FILENAME,
    evaluate_crd_s1c_checkpoint,
)
from resp_train.crd.selection import apply_s1c_selection


def _write_candidate_lock(root: Path) -> tuple[Path, Path]:
    variants = []
    first_checkpoint: Path | None = None
    for variant_index in range(4):
        checkpoints = []
        for seed_offset in range(3):
            run_dir = root / "runs" / f"variant_{variant_index}" / f"seed_{seed_offset}"
            run_dir.mkdir(parents=True)
            checkpoint = run_dir / "checkpoint_best_local_rr.pt"
            if first_checkpoint is None:
                first_checkpoint = checkpoint
            artifacts = {
                "checkpoint_sha256": checkpoint,
                "resolved_config_sha256": run_dir / "config.yaml",
                "run_manifest_sha256": run_dir / "run_manifest.json",
                "validation_metrics_summary_sha256": run_dir / "metrics_summary.csv",
            }
            for name, path in artifacts.items():
                path.write_text(f"{variant_index}-{seed_offset}-{name}\n", encoding="utf-8")
            checkpoints.append(
                {
                    "seed": 20260811 + seed_offset,
                    "selected_epoch": seed_offset + 1,
                    "checkpoint_path": str(checkpoint.relative_to(root)),
                    "checkpoint_size_bytes": checkpoint.stat().st_size,
                    **{name: sha256_file(path) for name, path in artifacts.items()},
                }
            )
        variants.append(
            {
                "role": "reference" if variant_index == 3 else "candidate",
                "variant": f"variant_{variant_index}",
                "training_commit": "deadbeef",
                "training_protocol": "test",
                "checkpoints": checkpoints,
            }
        )
    lock_path = root / "candidate_lock.json"
    lock_path.write_text(json.dumps({"schema_version": 1, "variants": variants}), encoding="utf-8")
    assert first_checkpoint is not None
    return lock_path, first_checkpoint


def test_candidate_lock_verifies_all_artifacts_and_rejects_tampering(monkeypatch, tmp_path) -> None:
    lock_path, checkpoint = _write_candidate_lock(tmp_path)
    monkeypatch.setattr(candidate_lock, "REPO_ROOT", tmp_path)

    verification = candidate_lock.verify_candidate_lock(lock_path, checkpoint_path=checkpoint)
    assert len(verification.records) == 12
    assert verification.matched_record is not None

    unlocked = tmp_path / "runs" / "unlocked.pt"
    unlocked.write_text("unlocked", encoding="utf-8")
    with pytest.raises(CandidateLockError, match="不在 candidate lock"):
        candidate_lock.verify_candidate_lock(lock_path, checkpoint_path=unlocked)

    checkpoint.write_text("tampered", encoding="utf-8")
    with pytest.raises(CandidateLockError, match="SHA-256 不一致"):
        candidate_lock.verify_candidate_lock(lock_path)


def test_s1c_requires_explicit_research_test_confirmation() -> None:
    with pytest.raises(ValueError, match="显式传入"):
        evaluate_crd_s1c_checkpoint(
            checkpoint_path="missing.pt",
            device="cpu",
            confirm_research_test=False,
        )


def test_s1c_writes_isolated_outputs_and_refuses_repeat(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "run" / "checkpoint_best_local_rr.pt"
    checkpoint.parent.mkdir()
    record = {
        "role": "candidate",
        "variant": "crd_102_b0_local_mamba",
        "seed": 20260811,
        "selected_epoch": 13,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": "checkpoint-sha",
        "training_commit": "training-commit",
        "training_protocol": "training-protocol",
    }
    verification = CandidateLockVerification(
        lock_path=tmp_path / "lock.json",
        lock_sha256=S1C_CANDIDATE_LOCK_SHA256,
        payload={
            "lock_id": "crd-v1.1-candidate-lock-20260809",
            "selection_rule": {"frozen": True},
        },
        records=(record,),
        matched_record=record,
    )
    cfg = OmegaConf.create(
        {
            "protocol": {"run_role": "formal"},
            "model": {"variant": record["variant"]},
            "training": {"seed": record["seed"]},
            "data": {"test_split": "test", "max_test_windows": None},
        }
    )
    metrics = pd.DataFrame(
        [
            {
                "evaluation_split": "test",
                "method": record["variant"],
                "dataset_row_id": 1,
                "split": "test",
                "samp_id": 7,
                "whole_rr_abs_error_bpm": 0.1,
                "whole_rr_target_eligible": True,
                "local_rr_mae_bpm": 0.2,
                "local_rr_target_eligible": True,
                "envelope_trajectory_mae": 0.3,
                "global_envelope_modulation_error": 0.4,
                "lag_aware_signed_pcc": 0.9,
                "joint_target_eligible": True,
                "respiratory_band_coherence": 0.8,
                "constrained_ndtw": 0.2,
            }
        ]
    )
    calls = {"evaluate": 0}

    class FakeExperiment:
        def __init__(self, experiment_cfg) -> None:
            assert experiment_cfg is cfg

        def evaluate_checkpoint(self, path, *, split, metrics_output):
            assert path == checkpoint
            assert split == "test"
            assert metrics_output is None
            calls["evaluate"] += 1
            return metrics

    def fake_manifest(path, **context):
        assert context["evaluation_split"] == "test"
        Path(path).write_text("{}\n", encoding="utf-8")
        return Path(path)

    monkeypatch.setattr("resp_train.crd.confirmation.verify_candidate_lock", lambda *args, **kwargs: verification)
    monkeypatch.setattr("resp_train.crd.confirmation.resolve_repo_path", lambda _path: checkpoint)
    monkeypatch.setattr("resp_train.crd.confirmation.load_crd_config", lambda *args, **kwargs: cfg)
    monkeypatch.setattr("resp_train.crd.confirmation._assert_clean_repository", lambda: None)
    monkeypatch.setattr("resp_train.crd.confirmation.CRDExperiment", FakeExperiment)
    monkeypatch.setattr("resp_train.crd.confirmation.save_execution_manifest", fake_manifest)

    output = evaluate_crd_s1c_checkpoint(
        checkpoint_path=checkpoint,
        device="cpu",
        confirm_research_test=True,
        output_root=tmp_path / "outputs",
        expected_test_windows=1,
        expected_test_samp_ids=1,
    )
    output_dir = output.parent
    assert output.name == METRICS_FILENAME
    assert calls["evaluate"] == 1
    for filename in (ACCESS_RECEIPT_FILENAME, METRICS_FILENAME, SUMMARY_FILENAME, MANIFEST_FILENAME):
        assert (output_dir / filename).exists()

    with pytest.raises(FileExistsError, match="禁止覆盖或重复评价"):
        evaluate_crd_s1c_checkpoint(
            checkpoint_path=checkpoint,
            device="cpu",
            confirm_research_test=True,
            output_root=tmp_path / "outputs",
            expected_test_windows=1,
            expected_test_samp_ids=1,
        )
    assert calls["evaluate"] == 1


def test_s1c_retains_access_receipt_when_evaluation_fails(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "run" / "checkpoint_best_local_rr.pt"
    checkpoint.parent.mkdir()
    record = {
        "role": "reference",
        "variant": "crd_001_b0_retrain",
        "seed": 20260811,
        "selected_epoch": 17,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": "checkpoint-sha",
        "training_commit": "training-commit",
        "training_protocol": "training-protocol",
    }
    verification = CandidateLockVerification(
        lock_path=tmp_path / "lock.json",
        lock_sha256=S1C_CANDIDATE_LOCK_SHA256,
        payload={
            "lock_id": "crd-v1.1-candidate-lock-20260809",
            "selection_rule": {},
        },
        records=(record,),
        matched_record=record,
    )
    cfg = OmegaConf.create(
        {
            "protocol": {"run_role": "formal"},
            "model": {"variant": record["variant"]},
            "training": {"seed": record["seed"]},
            "data": {"test_split": "test", "max_test_windows": None},
        }
    )

    class FailingExperiment:
        def __init__(self, _cfg) -> None:
            pass

        def evaluate_checkpoint(self, *args, **kwargs):
            raise RuntimeError("evaluation failed")

    monkeypatch.setattr("resp_train.crd.confirmation.verify_candidate_lock", lambda *args, **kwargs: verification)
    monkeypatch.setattr("resp_train.crd.confirmation.resolve_repo_path", lambda _path: checkpoint)
    monkeypatch.setattr("resp_train.crd.confirmation.load_crd_config", lambda *args, **kwargs: cfg)
    monkeypatch.setattr("resp_train.crd.confirmation._assert_clean_repository", lambda: None)
    monkeypatch.setattr("resp_train.crd.confirmation.CRDExperiment", FailingExperiment)

    with pytest.raises(RuntimeError, match="evaluation failed"):
        evaluate_crd_s1c_checkpoint(
            checkpoint_path=checkpoint,
            device="cpu",
            confirm_research_test=True,
            output_root=tmp_path / "outputs",
        )
    output_dir = tmp_path / "outputs" / str(record["variant"]) / f"seed_{record['seed']}"
    assert (output_dir / ACCESS_RECEIPT_FILENAME).exists()
    assert not (output_dir / METRICS_FILENAME).exists()


def test_s1c_selection_applies_frozen_eligibility_then_pareto() -> None:
    variants = [
        ("candidate", "crd_102_b0_local_mamba"),
        ("candidate", "crd_104_direct_hier_mamba"),
        ("candidate", "crd_105_direct_coarse"),
        ("reference", "crd_001_b0_retrain"),
    ]
    fixed_seeds = [20260811, 20260812, 20260813]
    metrics = {
        "crd_102_b0_local_mamba": (8.0, 9.9, 0.99, 1.8, 0.81),
        "crd_104_direct_hier_mamba": (7.0, 9.0, 1.03, 1.7, 0.82),
        "crd_105_direct_coarse": (9.0, 10.0, 1.00, 2.0, 0.80),
        "crd_001_b0_retrain": (10.0, 11.0, 1.10, 2.1, 0.79),
    }
    rows = []
    for role, variant in variants:
        whole, local, trajectory, global_error, pcc = metrics[variant]
        for seed in fixed_seeds:
            rows.append(
                {
                    "role": role,
                    "variant": variant,
                    "seed": seed,
                    "selected_epoch": 1,
                    "checkpoint_path": f"{variant}/{seed}.pt",
                    "n_samples": 2310,
                    "whole_rr_abs_error_bpm_mean": whole,
                    "local_rr_mae_bpm_mean": local,
                    "envelope_trajectory_mae_mean": trajectory,
                    "global_envelope_modulation_error_mean": global_error,
                    "lag_aware_signed_pcc_mean": pcc,
                }
            )
    payload = {
        "fixed_seeds": fixed_seeds,
        "anchor_variant": "crd_105_direct_coarse",
        "variants": [{"role": role, "variant": variant} for role, variant in variants],
        "selection_rule": {
            "candidate_eligibility": {
                "local_rr_relative_improvement_min": 0.005,
                "local_rr_paired_seed_improvement_min_count": 2,
                "signed_pcc_absolute_drop_max": 0.005,
                "trajectory_relative_worsening_max": 0.015,
            },
            "primary_metrics": [
                {"name": "whole_rr_abs_error_bpm_mean", "direction": "minimize"},
                {"name": "local_rr_mae_bpm_mean", "direction": "minimize"},
                {"name": "envelope_trajectory_mae_mean", "direction": "minimize"},
                {"name": "global_envelope_modulation_error_mean", "direction": "minimize"},
                {"name": "lag_aware_signed_pcc_mean", "direction": "maximize"},
            ],
        },
    }

    variant_summary, decision = apply_s1c_selection(pd.DataFrame(rows), payload)

    indexed = variant_summary.set_index("variant")
    assert bool(indexed.loc["crd_102_b0_local_mamba", "eligible"]) is True
    assert bool(indexed.loc["crd_104_direct_hier_mamba", "eligible"]) is False
    assert bool(indexed.loc["crd_104_direct_hier_mamba", "trajectory_worsening_pass"]) is False
    assert bool(indexed.loc["crd_105_direct_coarse", "eligible"]) is True
    assert bool(indexed.loc["crd_001_b0_retrain", "eligible"]) is False
    assert decision["eligible_variants"] == ["crd_102_b0_local_mamba", "crd_105_direct_coarse"]
    assert decision["pareto_set"] == ["crd_102_b0_local_mamba"]
    assert decision["secondary_metrics_used_for_selection"] is False
