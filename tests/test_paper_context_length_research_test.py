from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

import resp_train.paper_evidence.context_length_research_test as research_test
from resp_train.paper_evidence.context_length_research_test import (
    ContextLengthResearchTestDataset,
    EvaluationSpec,
    expected_research_test_evaluations,
)


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_allowlist_is_exactly_42_validation_selected_checkpoints() -> None:
    allowlist = expected_research_test_evaluations()
    assert len(allowlist) == 42
    assert sum(spec.task == "center30" for spec in allowlist.values()) == 24
    assert sum(spec.task == "center60" for spec in allowlist.values()) == 18
    assert len({spec.identity for spec in allowlist.values()}) == 42
    assert all(Path(spec.checkpoint_path).name.startswith("checkpoint_best_center") for spec in allowlist.values())
    assert all(Path(spec.checkpoint_path).is_file() for spec in allowlist.values())
    assert all("checkpoint_final" not in spec.checkpoint_path for spec in allowlist.values())


class _Parent:
    def __init__(self, split: str = "test") -> None:
        self.rows = pd.DataFrame({"dataset_row_id": [7], "split": [split]})
        self.item = {
            "x": torch.arange(18000, dtype=torch.float32).view(1, -1),
            "target": torch.arange(18000, dtype=torch.float32).view(1, -1),
            "meta": {
                "dataset_row_id": 7,
                "split": split,
                "input_set": "research_v2_waveform",
                "samp_id": 1,
                "coupling_state_id": 2,
                "rr_peak_valid_mask": torch.ones(18000, dtype=torch.bool),
            },
        }

    def __len__(self) -> int:
        return 1

    def __getitem__(self, _index: int):
        return self.item


class _Cache:
    def __init__(self) -> None:
        self.verified = None

    def verify_rows(self, values) -> None:
        self.verified = list(values)

    def get(self, dataset_row_id: int) -> torch.Tensor:
        assert dataset_row_id == 7
        return torch.zeros(49, 90)


def test_test_dataset_crops_center30_and_injects_w_by_row_id() -> None:
    cache = _Cache()
    dataset = ContextLengthResearchTestDataset(
        _Parent(), task="center30", input_samples=4500, w_cache=cache
    )
    item = dataset[0]
    assert cache.verified == [7]
    assert item["x"].shape == (1, 4500)
    assert item["target"].shape == (1, 3000)
    assert item["meta"]["rr_peak_valid_mask"].shape == (3000,)
    assert item["tf"]["w"].shape == (49, 90)
    assert item["x"][0, 0].item() == 6750
    assert item["target"][0, 0].item() == 7500


def test_test_dataset_crops_center60_without_w_cache() -> None:
    dataset = ContextLengthResearchTestDataset(
        _Parent(), task="center60", input_samples=18000, w_cache=None
    )
    item = dataset[0]
    assert item["x"].shape == (1, 18000)
    assert item["target"].shape == (1, 6000)
    assert item["meta"]["rr_peak_valid_mask"].shape == (6000,)
    assert "tf" not in item
    assert item["target"][0, 0].item() == 6000
    with pytest.raises(ValueError, match="test rows"):
        ContextLengthResearchTestDataset(
            _Parent("val"), task="center60", input_samples=6000, w_cache=None
        )


def test_checkpoint_payload_must_match_selected_task_identity() -> None:
    spec = EvaluationSpec(
        task="center30",
        model_key="c201",
        variant="c201_center30",
        experiment_id="C30V1_C201_30",
        input_sec=30,
        output_sec=30,
        seed=20260811,
        run_dir="run",
        run_commit="commit",
        artifact_manifest_sha256="manifest",
        resolved_config_sha256="config",
        checkpoint_path="checkpoint_best_center30_rr.pt",
        checkpoint_sha256="checkpoint",
        checkpoint_size_bytes=1,
        selected_epoch=17,
        parameter_count=1069802,
    )
    payload = {
        "epoch": 17,
        "model_state_dict": {},
        "config": {
            "model": {"variant": "c201_center30"},
            "window": {"input_sec": 30, "output_sec": 30},
            "training": {"seed": 20260811},
        },
    }
    research_test._validate_checkpoint_payload(payload, spec)
    payload["epoch"] = 18
    with pytest.raises(RuntimeError, match="selected epoch"):
        research_test._validate_checkpoint_payload(payload, spec)


def test_evaluator_cli_requires_confirmation_before_test_access() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/eval_paper_context_length_research_test_v1.py"),
            "--task",
            "center30",
            "--model",
            "c201",
            "--input-sec",
            "30",
            "--seed",
            "20260811",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "confirm-research-test" in result.stderr
