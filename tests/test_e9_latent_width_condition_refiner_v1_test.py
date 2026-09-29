from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from resp_train.paper_evidence import e9_latent_width_condition_refiner_v1 as e9
from resp_train.paper_evidence import e9_latent_width_condition_refiner_v1_test as test_module
from resp_train.paper_evidence.e9_latent_width_condition_refiner_v1_model import ARMS


def test_research_test_contract_fixes_complete_18_checkpoint_matrix():
    assert len(ARMS) == 6
    assert len(e9.SEEDS) == 3
    assert test_module.COUNT == 2_310
    assert test_module.SUBJECTS == 8
    assert test_module.TEST_SAMPLE_SEED == 20260612
    assert len(test_module.ROW_ORDER_SHA256) == 64
    assert all((e9.ROOT / path).is_file() for path in test_module._code_paths())


def test_evidence_lifecycle_is_frozen_and_duplicate_safe(tmp_path):
    evidence_hash = "a" * 64
    parent = tmp_path / "allowlist"
    with test_module.evidence_attempt(
        parent,
        phase="allowlist",
        evidence_hash=evidence_hash,
        reject_completed=True,
    ) as output:
        for filename in (
            "allowlist.json",
            "allowlist_receipt.json",
            "source_manifest.json",
            "access_receipt.json",
            "source_code.json",
        ):
            (output / filename).write_text("{}", encoding="utf-8")
    completed = next(parent.glob("*/freeze_receipt.json")).parent
    manifest = test_module.verify_attempt(
        completed,
        phase="allowlist",
        evidence_hash=evidence_hash,
    )
    assert manifest["status"] == "completed"
    with pytest.raises(FileExistsError, match="已完成"):
        with test_module.evidence_attempt(
            parent,
            phase="allowlist",
            evidence_hash=evidence_hash,
            reject_completed=True,
        ):
            pass


def test_entry_lookup_rejects_matrix_drift():
    lock = {
        "entries": [
            {"arm": arm, "seed": seed, "entry_id": f"{arm}_{seed}"}
            for seed in e9.SEEDS
            for arm in ARMS
        ]
    }
    assert test_module._entry(lock, ARMS[0], e9.SEEDS[0])["entry_id"] == (
        f"{ARMS[0]}_{e9.SEEDS[0]}"
    )
    with pytest.raises(ValueError, match="固定矩阵"):
        test_module._entry(lock, "unknown", e9.SEEDS[0])


def test_secondary_across_seed_preserves_finite_coverage():
    rows = []
    for arm in ARMS:
        for seed in e9.SEEDS:
            rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "split": "test",
                    "selected_epoch": 1,
                    "quality_acceptance_passed": True,
                    "method": arm,
                    "respiratory_band_coherence_mean": 0.5,
                    "constrained_ndtw_mean": np.nan if seed == e9.SEEDS[0] else 0.2,
                }
            )
    result = test_module._secondary_across_seed(pd.DataFrame(rows))
    assert len(result) == len(ARMS)
    assert set(result.respiratory_band_coherence_mean_finite_seeds) == {3}
    assert set(result.constrained_ndtw_mean_finite_seeds) == {2}
    assert result.constrained_ndtw_mean_seed_mean.isna().all()


@pytest.mark.parametrize("phase", ["evaluate", "summary"])
def test_cli_requires_explicit_research_test_confirmation(tmp_path, phase):
    script = e9.ROOT / test_module.SCRIPT_PATH
    command = [sys.executable, str(script), phase, "--allowlist", str(tmp_path)]
    if phase == "evaluate":
        command.extend(["--arm", ARMS[0], "--seed", str(e9.SEEDS[0])])
    result = subprocess.run(
        command,
        cwd=e9.ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode != 0
    assert "--confirm-research-test" in result.stderr


def test_protocol_keeps_test_arrays_closed_before_explicit_confirmation():
    text = (e9.ROOT / test_module.PROTOCOL_PATH).read_text(encoding="utf-8")
    assert "test_array_read=false" in text
    assert "--confirm-research-test" in text
    assert "18" in text
