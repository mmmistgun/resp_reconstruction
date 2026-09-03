from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from resp_train.paper_evidence.center30_config import CENTER30_INPUT_SAMPLES
from resp_train.paper_evidence.center30_research_test_cache import (
    EXPECTED_DATASET_INDEX_SHA256,
    EXPECTED_TEST_ROW_IDS_SHA256,
    Center30ResearchTestWCacheReader,
    build_center30_research_test_w_cache_from_rows,
    fixed_research_test_cache_identity,
    research_test_cache_identity_sha256,
)


ROOT = Path(__file__).resolve().parents[1]


def _rows(ids: tuple[int, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset_row_id": ids,
            "split": ["test"] * len(ids),
            "samp_id": list(range(len(ids))),
            "source_npz": ["source.npz"] * len(ids),
            "bcg_signal_key": ["bcg"] * len(ids),
            "window_start_sample": [0] * len(ids),
            "window_end_sample": [18000] * len(ids),
        }
    )


def _row_hash(rows: pd.DataFrame) -> str:
    values = rows["dataset_row_id"].to_numpy(dtype=np.int64)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _features(waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    length = len(waveform)
    assert length in CENTER30_INPUT_SAMPLES
    value = float(np.mean(waveform))
    return (
        np.full((49, length // 50), value, dtype=np.float32),
        np.linspace(0.01, 1.0, 49, dtype=np.float64),
    )


def test_research_test_cache_identity_freezes_four_lengths_and_test_row_hash() -> None:
    identity = fixed_research_test_cache_identity()
    assert identity["split"] == "test"
    assert identity["row_ids_sha256"] == EXPECTED_TEST_ROW_IDS_SHA256
    assert identity["input_samples"] == list(CENTER30_INPUT_SAMPLES)
    assert set(identity["transform_specs"]) == {str(value) for value in CENTER30_INPUT_SAMPLES}
    assert research_test_cache_identity_sha256() == research_test_cache_identity_sha256()


def test_synthetic_cache_is_input_only_complete_and_readable(tmp_path: Path) -> None:
    index_path = tmp_path / "dataset_index.csv"
    index_path.write_text("placeholder\n", encoding="utf-8")
    np.savez(tmp_path / "source.npz", bcg=np.linspace(-1.0, 1.0, 18000, dtype=np.float32))
    rows = _rows((101, 202))
    manifest_path = build_center30_research_test_w_cache_from_rows(
        output_dir=tmp_path / "cache",
        index_path=index_path,
        rows=rows,
        dataset_index_sha256=EXPECTED_DATASET_INDEX_SHA256,
        expected_count=2,
        expected_samp_ids=2,
        expected_row_ids_sha256=_row_hash(rows),
        feature_extractor=_features,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert manifest["test_input_read"] is True
    assert manifest["test_target_array_read"] is False
    assert manifest["target_read"] is False
    assert manifest["model_inference_used"] is False
    assert json.loads((manifest_path.parent / "lifecycle.json").read_text(encoding="utf-8")) == {
        "status": "complete"
    }
    for length in CENTER30_INPUT_SAMPLES:
        reader = Center30ResearchTestWCacheReader(manifest_path.parent, input_samples=length)
        reader.verify_rows([101, 202])
        assert reader.get(101).shape == (49, length // 50)


def test_cache_rejects_non_test_or_duplicate_rows(tmp_path: Path) -> None:
    index_path = tmp_path / "dataset_index.csv"
    index_path.write_text("placeholder\n", encoding="utf-8")
    rows = _rows((1, 2))
    rows.loc[0, "split"] = "val"
    with pytest.raises(ValueError, match="split"):
        build_center30_research_test_w_cache_from_rows(
            output_dir=tmp_path / "bad_split",
            index_path=index_path,
            rows=rows,
            dataset_index_sha256=EXPECTED_DATASET_INDEX_SHA256,
            expected_count=2,
            expected_samp_ids=2,
            expected_row_ids_sha256=_row_hash(rows),
            feature_extractor=_features,
        )
    duplicate = _rows((1, 1))
    with pytest.raises(ValueError, match="无重复"):
        build_center30_research_test_w_cache_from_rows(
            output_dir=tmp_path / "duplicate",
            index_path=index_path,
            rows=duplicate,
            dataset_index_sha256=EXPECTED_DATASET_INDEX_SHA256,
            expected_count=2,
            expected_samp_ids=2,
            expected_row_ids_sha256=_row_hash(duplicate),
            feature_extractor=_features,
        )


def test_cache_output_is_non_overwriting(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    rows = _rows((1,))
    with pytest.raises(FileExistsError):
        build_center30_research_test_w_cache_from_rows(
            output_dir=output,
            index_path=tmp_path / "unused.csv",
            rows=rows,
            dataset_index_sha256=EXPECTED_DATASET_INDEX_SHA256,
            expected_count=1,
            expected_samp_ids=1,
            expected_row_ids_sha256=_row_hash(rows),
            feature_extractor=_features,
        )


def test_cache_cli_requires_confirmation_without_accessing_test() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/build_paper_center30_context_research_test_w_cache_v1.py")],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "confirm-research-test-cache-build" in result.stderr
