from __future__ import annotations

from pathlib import Path

import pytest
import torch

from resp_train.crd.tf_v1_data import (
    FROZEN_CACHE_TRANSFORM_SHA256,
    TfV1CacheReader,
    batch_tf_to_device,
)


FROZEN_CACHE_ROOT = Path("runs/crd_tf_v1/cache") / FROZEN_CACHE_TRANSFORM_SHA256


def test_batch_tf_to_device_preserves_nested_tensor_mapping() -> None:
    batch = {"tf": {"m_slow": torch.ones(2, 36, 101), "l_spectrum": torch.ones(2, 9001, dtype=torch.complex64)}}
    moved = batch_tf_to_device(batch, torch.device("cpu"), non_blocking=False)

    assert moved is not None
    assert set(moved) == {"m_slow", "l_spectrum"}
    assert moved["m_slow"].device.type == "cpu"
    assert moved["l_spectrum"].dtype == torch.complex64
    assert batch_tf_to_device({}, torch.device("cpu"), non_blocking=False) is None


@pytest.mark.skipif(not FROZEN_CACHE_ROOT.exists(), reason="冻结 CRD-TF cache 不在当前机器")
def test_frozen_cache_reader_uses_row_id_and_only_opens_requested_representations() -> None:
    reader = TfV1CacheReader(FROZEN_CACHE_ROOT, split="val", representations=("m", "l"))
    row_id = int(reader.row_ids[0])

    features = reader.get(row_id)

    assert set(reader.arrays) == {"m_slow", "m_fast", "l_spectrum"}
    assert features["m_slow"].shape == (36, 101)
    assert features["m_fast"].shape == (44, 349)
    assert features["l_spectrum"].shape == (9001,)
    assert features["l_spectrum"].dtype == torch.complex64
    reader.verify_rows([row_id])
    with pytest.raises(KeyError, match="dataset_row_id"):
        reader.get(-1)


def test_cache_reader_rejects_test_split_before_touching_files() -> None:
    with pytest.raises(ValueError, match="只允许 train/val"):
        TfV1CacheReader(FROZEN_CACHE_ROOT, split="test", representations=("m",))

