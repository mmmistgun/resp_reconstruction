from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import Dataset

from resp_train.paper_evidence.center_context_cache import (
    REDUCED_SCALE_COUNT,
    CenterContextWCacheReader,
    build_center_context_w_cache_from_rows,
    center_context_cwt_features,
    fixed_center_w_transform_spec,
    reduced_target_frequencies,
    write_center_context_w_cache,
)
from resp_train.paper_evidence.center_context_data import (
    CenterContextDataset,
    audit_nested_view_identity,
    audit_parent_row_identity,
    crop_center_context_tensors,
    latent_center_bounds,
)


class _Parent(Dataset):
    def __init__(self, split: str = "train") -> None:
        self.rows = pd.DataFrame({"dataset_row_id": [7], "split": [split]})
        self.item = {
            "x": torch.arange(18000, dtype=torch.float32).view(1, -1),
            "target": (torch.arange(18000, dtype=torch.float32) * 2).view(1, -1),
            "meta": {
                "dataset_row_id": 7,
                "split": split,
                "samp_id": 1,
                "coupling_state_id": 2,
            },
        }

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        return self.item


def test_three_nested_inputs_share_exact_center_target_and_latent_slices() -> None:
    parent = _Parent().item
    targets = []
    for length, expected_bounds in ((6000, (0, 600)), (9000, (150, 750)), (18000, (600, 1200))):
        x, target = crop_center_context_tensors(parent["x"], parent["target"], input_samples=length)
        assert x.shape == (1, length)
        assert target.shape == (1, 6000)
        assert latent_center_bounds(length) == expected_bounds
        targets.append(target)
    assert all(torch.equal(targets[0], value) for value in targets[1:])
    assert audit_nested_view_identity([parent])["center_targets_pointwise_identical"] is True


def test_center_dataset_rejects_test_and_preserves_parent_identity() -> None:
    dataset = CenterContextDataset(_Parent("train"), input_samples=9000)
    item = dataset[0]
    assert item["x"].shape == (1, 9000)
    assert item["target"].shape == (1, 6000)
    assert item["meta"]["dataset_row_id"] == 7
    with pytest.raises(ValueError, match="train/val"):
        CenterContextDataset(_Parent("test"), input_samples=6000)


def test_parent_split_identity_rejects_subject_leakage() -> None:
    train = pd.DataFrame(
        {"dataset_row_id": [1], "split": ["train"], "samp_id": [10], "coupling_state_id": [1]}
    )
    val = pd.DataFrame(
        {"dataset_row_id": [2], "split": ["val"], "samp_id": [11], "coupling_state_id": [1]}
    )
    audit = audit_parent_row_identity(train, val)
    assert audit["row_overlap_count"] == 0
    leaked = val.copy()
    leaked["samp_id"] = 10
    with pytest.raises(RuntimeError, match="samp_id"):
        audit_parent_row_identity(train, leaked)


def test_length_specific_w_cache_shape_identity_access_and_nonoverwrite(tmp_path: Path) -> None:
    length = 9000
    frequencies = reduced_target_frequencies()
    split_features = {
        "train": (np.asarray([1, 2], dtype=np.int64), np.zeros((2, 49, 180), dtype=np.float32)),
        "val": (np.asarray([3], dtype=np.int64), np.ones((1, 49, 180), dtype=np.float32)),
    }
    root = tmp_path / "cache_90s"
    write_center_context_w_cache(
        output_dir=root,
        input_samples=length,
        split_features=split_features,
        frequencies_hz=frequencies,
        dataset_index_sha256="fixture",
    )
    reader = CenterContextWCacheReader(root, split="val", input_samples=length)
    assert reader.get(3).shape == (49, 180)
    reader.verify_rows([3])
    with pytest.raises(ValueError, match="train/val"):
        CenterContextWCacheReader(root, split="test", input_samples=length)
    with pytest.raises(FileExistsError, match="禁止覆盖"):
        write_center_context_w_cache(
            output_dir=root,
            input_samples=length,
            split_features=split_features,
            frequencies_hz=frequencies,
            dataset_index_sha256="fixture",
        )
    assert fixed_center_w_transform_spec(6000)["shape"] == [49, 120]
    assert fixed_center_w_transform_spec(9000)["shape"] == [49, 180]
    assert fixed_center_w_transform_spec(18000)["shape"] == [49, 360]
    assert fixed_center_w_transform_spec(18000)["effective_voices_per_octave"] == 6


@pytest.mark.parametrize("length,context", [(6000, 120), (9000, 180), (18000, 360)])
def test_actual_length_specific_cwt_shape_frequency_identity_and_finite(length: int, context: int) -> None:
    time = np.arange(length, dtype=np.float32) / 100.0
    waveform = np.sin(2.0 * np.pi * 0.2 * time).astype(np.float32)
    feature, frequencies = center_context_cwt_features(waveform)
    assert feature.shape == (REDUCED_SCALE_COUNT, context)
    assert frequencies.shape == (REDUCED_SCALE_COUNT,)
    assert np.isfinite(feature).all()
    assert np.isfinite(frequencies).all()
    assert np.all(np.diff(frequencies) >= 0.0)


def _cache_rows(split: str, row_id: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset_row_id": [row_id],
            "split": [split],
            "source_npz": [f"{split}.npz"],
            "bcg_signal_key": ["bcg"],
            "window_start_sample": [0],
            "window_end_sample": [18000],
        }
    )


def test_streaming_cache_builder_preserves_failure_lifecycle(monkeypatch, tmp_path: Path) -> None:
    class FakeWholeNightCache:
        def __init__(self, _: Path) -> None:
            pass

        def get_arrays(self, _: str, keys: list[str]):
            return {keys[0]: np.zeros(18000, dtype=np.float32)}

    monkeypatch.setattr("resp_train.data.cache.WholeNightCache", FakeWholeNightCache)

    def fail(_: np.ndarray):
        raise RuntimeError("fixture failure")

    output = tmp_path / "failed_cache"
    with pytest.raises(RuntimeError, match="fixture failure"):
        build_center_context_w_cache_from_rows(
            output_dir=output,
            index_path=tmp_path / "index.csv",
            rows_by_split={"train": _cache_rows("train", 1), "val": _cache_rows("val", 2)},
            input_samples=6000,
            dataset_index_sha256="fixture",
            feature_extractor=fail,
            complete=False,
        )
    lifecycle = json.loads((output / "lifecycle.json").read_text(encoding="utf-8"))
    assert lifecycle["status"] == "failed"
    assert lifecycle["error_type"] == "RuntimeError"
    with pytest.raises(FileExistsError):
        build_center_context_w_cache_from_rows(
            output_dir=output,
            index_path=tmp_path / "index.csv",
            rows_by_split={"train": _cache_rows("train", 1), "val": _cache_rows("val", 2)},
            input_samples=6000,
            dataset_index_sha256="fixture",
            feature_extractor=fail,
            complete=False,
        )


def test_streaming_cache_builder_writes_hashed_input_only_artifacts(monkeypatch, tmp_path: Path) -> None:
    class FakeWholeNightCache:
        def __init__(self, _: Path) -> None:
            pass

        def get_arrays(self, _: str, keys: list[str]):
            return {keys[0]: np.zeros(18000, dtype=np.float32)}

    monkeypatch.setattr("resp_train.data.cache.WholeNightCache", FakeWholeNightCache)

    def extract(view: np.ndarray):
        assert view.shape == (9000,)
        return np.zeros((49, 180), dtype=np.float32), reduced_target_frequencies()

    output = tmp_path / "complete_cache"
    manifest_path = build_center_context_w_cache_from_rows(
        output_dir=output,
        index_path=tmp_path / "index.csv",
        rows_by_split={"train": _cache_rows("train", 1), "val": _cache_rows("val", 2)},
        input_samples=9000,
        dataset_index_sha256="fixture",
        feature_extractor=extract,
        complete=True,
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["input_only"] is True
    assert payload["target_read"] is False
    assert payload["test_read"] is False
    assert payload["frequency_count"] == 49
    assert payload["files"]["train_w.npy"]["sha256"]
    assert json.loads((output / "lifecycle.json").read_text())["status"] == "complete"
