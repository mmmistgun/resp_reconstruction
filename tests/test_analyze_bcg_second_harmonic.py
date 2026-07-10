from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from resp_train.analysis.second_harmonic import HarmonicFeatureConfig
from scripts.analyze_bcg_second_harmonic import (
    build_feature_frame,
    build_threshold_proposal,
    freeze_threshold_candidate,
    load_frozen_thresholds,
)


class _FakeDataset:
    def __init__(self) -> None:
        fs = 100.0
        time = np.arange(18000, dtype=np.float64) / fs
        tho = np.sin(2.0 * np.pi * 0.20 * time).astype(np.float32)
        harmonic = (0.3 * tho + np.sin(2.0 * np.pi * 0.40 * time)).astype(np.float32)
        fundamental = tho.copy()
        self.rows = pd.DataFrame(
            {
                "dataset_row_id": [11, 12],
                "samp_id": [220, 221],
                "split": ["val", "val"],
                "window_start_s": [0.0, 30.0],
                "source_npz": ["a.npz", "b.npz"],
            }
        )
        self._samples = [
            {"x": torch.from_numpy(harmonic).view(1, -1), "target": torch.from_numpy(tho).view(1, -1)},
            {"x": torch.from_numpy(fundamental).view(1, -1), "target": torch.from_numpy(tho).view(1, -1)},
        ]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        return self._samples[idx]


def _feature_config() -> HarmonicFeatureConfig:
    return HarmonicFeatureConfig(fs=100.0, welch_nperseg=4096, tho_rr_agreement_bpm=2.0)


def test_build_feature_frame_preserves_dataset_row_and_source_metadata() -> None:
    frame = build_feature_frame(_FakeDataset(), split="val", feature_cfg=_feature_config())

    assert frame["dataset_row_id"].tolist() == [11, 12]
    assert frame["samp_id"].tolist() == [220, 221]
    assert frame["split"].unique().tolist() == ["val"]
    assert frame["window_start_s"].tolist() == [0.0, 30.0]
    assert frame.loc[0, "harmonic_to_fundamental_ratio"] > 1.0
    assert frame.loc[1, "harmonic_to_fundamental_ratio"] < 0.1


def test_build_feature_frame_rejects_duplicate_row_ids_and_wrong_split() -> None:
    duplicate = _FakeDataset()
    duplicate.rows.loc[1, "dataset_row_id"] = 11
    with pytest.raises(ValueError, match="重复 dataset_row_id"):
        build_feature_frame(duplicate, split="val", feature_cfg=_feature_config())

    wrong_split = _FakeDataset()
    wrong_split.rows.loc[1, "split"] = "test"
    with pytest.raises(ValueError, match="split"):
        build_feature_frame(wrong_split, split="val", feature_cfg=_feature_config())


def test_build_threshold_proposal_is_validation_only_and_deterministic() -> None:
    features = build_feature_frame(_FakeDataset(), split="val", feature_cfg=_feature_config())

    proposal = build_threshold_proposal(features, split="val", features_path=Path("validation.csv"))

    assert proposal["status"] == "proposal"
    assert proposal["split"] == "val"
    assert proposal["features_path"] == "validation.csv"
    assert proposal["candidates"][0]["candidate_id"] == "candidate_000"
    assert len(proposal["candidates"]) == 81
    assert {
        "tho_rr_agreement_bpm",
        "peak_relative_tolerance",
        "harmonic_to_fundamental_min",
        "harmonic_band_fraction_min",
        "correction_ratio_drop_min",
    } <= set(proposal["candidates"][0]["thresholds"])

    with pytest.raises(ValueError, match="验证集"):
        build_threshold_proposal(features.assign(split="test"), split="test", features_path=Path("test.csv"))


def test_freeze_candidate_writes_immutable_hash_tracked_thresholds(tmp_path: Path) -> None:
    features = build_feature_frame(_FakeDataset(), split="val", feature_cfg=_feature_config())
    proposal_path = tmp_path / "proposal.json"
    proposal_path.write_text(
        json.dumps(
            build_threshold_proposal(features, split="val", features_path=tmp_path / "features.csv"),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "harmonic_thresholds.json"

    freeze_threshold_candidate(
        proposal_path,
        candidate_id="candidate_000",
        output_path=output,
        review_note="高值、边界和低值案例已人工复核",
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    thresholds = load_frozen_thresholds(output)
    assert payload["status"] == "frozen"
    assert payload["proposal_sha256"]
    assert payload["review_note"] == "高值、边界和低值案例已人工复核"
    assert thresholds.version == payload["version"]

    freeze_threshold_candidate(
        proposal_path,
        candidate_id="candidate_000",
        output_path=output,
        review_note="高值、边界和低值案例已人工复核",
        allow_identical_existing=True,
    )
    with pytest.raises(FileExistsError, match="已存在"):
        freeze_threshold_candidate(
            proposal_path,
            candidate_id="candidate_001",
            output_path=output,
            review_note="尝试改变阈值",
        )


def test_load_frozen_thresholds_rejects_proposal(tmp_path: Path) -> None:
    proposal = tmp_path / "proposal.json"
    proposal.write_text('{"status":"proposal"}', encoding="utf-8")

    with pytest.raises(ValueError, match="frozen"):
        load_frozen_thresholds(proposal)
