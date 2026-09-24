from __future__ import annotations

import json
from pathlib import Path

import pytest

from resp_train.paper_evidence import e7_scale_encoding_aggregation_formal as formal


def test_p4_contract_and_upstream_identities():
    contract = formal.p4_contract()
    assert contract["cells"] == 18
    assert contract["max_epochs"] == 80
    assert contract["planned_updates"] == 6400
    assert contract["early_stopping"] == {
        "enabled": True,
        "min_epoch": 30,
        "patience": 15,
        "min_delta": 0.0,
        "monitor": "validation_local_rr_mae_full_split",
    }
    _p1, p1_digest = formal.e7.load_implementation_lock()
    _p3, p3_digest = formal.p3.load_p3_lock()
    assert p1_digest == formal.P1_LOCK_SHA256
    assert p3_digest == formal.P3_LOCK_SHA256
    assert all((formal.e7.ROOT / path).is_file() for path in formal.p4_critical_paths())


def test_p3_closeout_and_single_gpu_receipt_are_strict():
    closeout = formal._load_closeout()
    assert closeout["status"] == "completed"
    assert closeout["gpu_acceptance"]["passed_records"] == 24
    assert closeout["benchmark"]["measurements"] == 36
    lock = {
        "gpu_receipt_allowlist": {
            "path": closeout["gpu_acceptance"]["path"],
            "manifest": closeout["gpu_acceptance"]["manifest"],
            "result": formal.e7.identity(
                formal.e7.ROOT
                / closeout["gpu_acceptance"]["path"]
                / "gpu_acceptance.json"
            ),
            "environment": formal.e7.identity(
                formal.e7.ROOT
                / closeout["gpu_acceptance"]["path"]
                / "environment.json"
            ),
        }
    }
    receipt = formal.e7.ROOT / closeout["gpu_acceptance"]["path"]
    result = formal.verify_gpu_receipt(receipt, lock)
    assert result["passed"] is True and len(result["records"]) == 24
    with pytest.raises(ValueError, match="allowlist"):
        formal.verify_gpu_receipt(receipt.parent, lock)


def test_formal_rejects_cells_outside_matrix(tmp_path):
    receipt = tmp_path / "receipt"
    with pytest.raises(ValueError, match="18-cell"):
        formal.run_formal("other", formal.e7.SEEDS[0], gpu_receipt=receipt, device="cpu")
    with pytest.raises(ValueError, match="18-cell"):
        formal.run_formal(formal.ARMS[0], 1, gpu_receipt=receipt, device="cpu")


def test_attempt_preserves_failure_and_completed_identity(tmp_path):
    digest = "a" * 64
    parent = tmp_path / "formal"
    with pytest.raises(RuntimeError, match="fixture"):
        with formal.attempt(parent, digest, formal.ARMS[0], formal.e7.SEEDS[0]) as output:
            (output / "partial.txt").write_text("kept", encoding="utf-8")
            raise RuntimeError("fixture failure")
    assert (output / "partial.txt").read_text() == "kept"
    assert (output / "lifecycle_failed.json").is_file()
    with pytest.raises(ValueError, match="失败"):
        formal.verify_formal_attempt(output, lock_hash=digest)


def test_execution_lock_preparation_requires_clean_tree(monkeypatch, tmp_path):
    monkeypatch.setattr(formal.e7, "git_state", lambda root: {"commit": "x", "status_porcelain": " M x"})
    with pytest.raises(RuntimeError, match="干净工作树"):
        formal.prepare_p4_lock(tmp_path)
