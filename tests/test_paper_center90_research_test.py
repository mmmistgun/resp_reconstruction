from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

import resp_train.paper_evidence.center90_research_test as research_test
from resp_train.paper_evidence.center90_research_test import (
    Center90ResearchTestDataset,
    EvaluationSpec,
    expected_research_test_evaluations,
)


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_allowlist_is_exactly_18_validation_selected_checkpoints() -> None:
    allowlist = expected_research_test_evaluations()
    assert len(allowlist) == 18
    assert sum(spec.model_key == "c201" for spec in allowlist.values()) == 9
    assert sum(spec.model_key == "wr" for spec in allowlist.values()) == 9
    assert len({spec.identity for spec in allowlist.values()}) == 18
    assert all(Path(spec.checkpoint_path).name == "checkpoint_best_center90_rr.pt" for spec in allowlist.values())
    assert all(Path(spec.checkpoint_path).is_file() for spec in allowlist.values())


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
    def __init__(self, width: int) -> None:
        self.width = width
        self.verified = None

    def verify_rows(self, values) -> None:
        self.verified = list(values)

    def get(self, dataset_row_id: int) -> torch.Tensor:
        assert dataset_row_id == 7
        return torch.zeros(49, self.width)


@pytest.mark.parametrize(
    ("input_samples", "input_start", "width"),
    ((9000, 4500, 180), (13500, 2250, 270), (18000, 0, 360)),
)
def test_dataset_crops_center90_and_injects_w_by_row_id(input_samples: int, input_start: int, width: int) -> None:
    cache = _Cache(width)
    dataset = Center90ResearchTestDataset(_Parent(), input_samples=input_samples, w_cache=cache)
    item = dataset[0]
    assert cache.verified == [7]
    assert item["x"].shape == (1, input_samples)
    assert item["target"].shape == (1, 9000)
    assert item["meta"]["rr_peak_valid_mask"].shape == (9000,)
    assert item["tf"]["w"].shape == (49, width)
    assert item["x"][0, 0].item() == input_start
    assert item["target"][0, 0].item() == 4500


def test_dataset_requires_test_rows_and_c201_has_no_w() -> None:
    item = Center90ResearchTestDataset(_Parent(), input_samples=9000, w_cache=None)[0]
    assert "tf" not in item
    with pytest.raises(ValueError, match="test rows"):
        Center90ResearchTestDataset(_Parent("val"), input_samples=9000, w_cache=None)


def test_checkpoint_payload_must_match_selected_identity() -> None:
    spec = EvaluationSpec(
        model_key="wr",
        variant="w_reduced_center90",
        experiment_id="C90V1_WR_135",
        input_sec=135,
        output_sec=90,
        seed=20260811,
        run_dir="run",
        run_commit="commit",
        artifact_manifest_sha256="manifest",
        resolved_config_sha256="config",
        checkpoint_path="checkpoint_best_center90_rr.pt",
        checkpoint_sha256="checkpoint",
        checkpoint_size_bytes=1,
        selected_epoch=17,
        parameter_count=1219850,
    )
    payload = {
        "epoch": 17,
        "model_state_dict": {},
        "config": {
            "model": {"variant": "w_reduced_center90"},
            "window": {"input_sec": 135, "output_sec": 90},
            "training": {"seed": 20260811},
        },
    }
    research_test._validate_checkpoint_payload(payload, spec)
    payload["epoch"] = 18
    with pytest.raises(RuntimeError, match="selected epoch"):
        research_test._validate_checkpoint_payload(payload, spec)


def test_w_cache_routing_uses_supplement_only_for_135() -> None:
    assert research_test._cache_root(90) == research_test.REUSED_CACHE_ROOT
    assert research_test._cache_root(180) == research_test.REUSED_CACHE_ROOT
    assert research_test._cache_root(135) == research_test.SUPPLEMENT_CACHE_ROOT
    assert research_test._cache_hash(135) == research_test.SUPPLEMENT_CACHE_MANIFEST_SHA256


def test_evaluator_cli_requires_confirmation_before_test_access() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/eval_paper_center90_research_test_v1.py"),
            "--model",
            "c201",
            "--input-sec",
            "90",
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
