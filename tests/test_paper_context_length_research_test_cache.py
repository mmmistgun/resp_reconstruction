from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import resp_train.paper_evidence.context_length_research_test_cache as cache
from resp_train.paper_evidence.center_context_cache import reduced_scales_and_frequencies
from resp_train.paper_evidence.context_length_research_test_cache import (
    ContextLengthResearchTestWCacheReader,
    INPUT_SAMPLES,
    build_context_length_research_test_w_cache_from_rows,
    fixed_input_bounds,
    fixed_research_test_cache_identity,
    fixed_transform_spec,
)


ROOT = Path(__file__).resolve().parents[1]


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset_row_id": [7, 8],
            "split": ["test", "test"],
            "samp_id": [1, 2],
            "source_npz": ["a.npz", "b.npz"],
            "bcg_signal_key": ["bcg", "bcg"],
            "window_start_sample": [0, 0],
            "window_end_sample": [18000, 18000],
        }
    )


def _row_hash(rows: pd.DataFrame) -> str:
    values = rows["dataset_row_id"].to_numpy(dtype=np.int64)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def test_joint_identity_freezes_two_tasks_five_lengths_and_shared_views() -> None:
    identity = fixed_research_test_cache_identity()
    assert tuple(identity["input_samples"]) == INPUT_SAMPLES
    assert identity["tasks"] == {
        "center30": {"input_samples": [3000, 4500, 6000, 9000], "output_samples": 3000},
        "center60": {"input_samples": [6000, 9000, 18000], "output_samples": 6000},
    }
    assert fixed_input_bounds(6000) == (6000, 12000)
    assert fixed_input_bounds(9000) == (4500, 13500)
    assert fixed_transform_spec(6000)["shape"] == [49, 120]
    assert fixed_transform_spec(9000)["shape"] == [49, 180]
    assert reduced_scales_and_frequencies.cache_parameters()["maxsize"] == 5


def test_joint_cache_builder_loads_each_parent_once_and_never_requests_target(monkeypatch, tmp_path: Path) -> None:
    rows = _rows()
    source_calls: list[tuple[str, tuple[str, ...]]] = []
    feature_lengths: list[int] = []

    class _SourceCache:
        def __init__(self, path: Path) -> None:
            assert path == (tmp_path / "index.csv").resolve()

        def get_arrays(self, source: str, keys: list[str]) -> dict[str, np.ndarray]:
            source_calls.append((source, tuple(keys)))
            assert keys == ["bcg"]
            return {"bcg": np.linspace(-1.0, 1.0, 18000, dtype=np.float32)}

    def _extract(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        length = int(values.size)
        feature_lengths.append(length)
        features = np.full((49, length // 50), length / 18000.0, dtype=np.float32)
        frequencies = np.linspace(0.05, 2.0, 49, dtype=np.float64)
        return features, frequencies

    monkeypatch.setattr(cache, "WholeNightCache", _SourceCache)
    output = tmp_path / "joint_cache"
    manifest_path = build_context_length_research_test_w_cache_from_rows(
        output_dir=output,
        index_path=tmp_path / "index.csv",
        rows=rows,
        dataset_index_sha256=cache.EXPECTED_DATASET_INDEX_SHA256,
        expected_count=2,
        expected_samp_ids=2,
        expected_row_ids_sha256=_row_hash(rows),
        non_test_row_overlap_count=0,
        feature_extractor=_extract,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert manifest["test_target_array_read"] is False
    assert manifest["target_read"] is False
    assert manifest["model_inference_used"] is False
    assert manifest["non_test_row_overlap_count"] == 0
    assert len(source_calls) == 2
    assert all(keys == ("bcg",) for _, keys in source_calls)
    assert feature_lengths == list(INPUT_SAMPLES) * 2
    assert set(manifest["features"]) == {str(length) for length in INPUT_SAMPLES}
    reader = ContextLengthResearchTestWCacheReader(output, input_samples=18000)
    assert reader.features.shape == (2, 49, 360)
    assert reader.get(8).shape == (49, 360)
    reader.verify_rows([7, 8])


def test_joint_cache_rejects_row_identity_overlap_and_overwrite(monkeypatch, tmp_path: Path) -> None:
    rows = _rows()
    output = tmp_path / "overlap"
    with pytest.raises(RuntimeError, match="零重叠"):
        build_context_length_research_test_w_cache_from_rows(
            output_dir=output,
            index_path=tmp_path / "index.csv",
            rows=rows,
            dataset_index_sha256=cache.EXPECTED_DATASET_INDEX_SHA256,
            expected_count=2,
            expected_samp_ids=2,
            expected_row_ids_sha256=_row_hash(rows),
            non_test_row_overlap_count=1,
            feature_extractor=lambda _: (_ for _ in ()).throw(AssertionError("not called")),
        )
    output.mkdir()
    with pytest.raises(FileExistsError):
        build_context_length_research_test_w_cache_from_rows(
            output_dir=output,
            index_path=tmp_path / "index.csv",
            rows=rows,
            dataset_index_sha256=cache.EXPECTED_DATASET_INDEX_SHA256,
            expected_count=2,
            expected_samp_ids=2,
            expected_row_ids_sha256=_row_hash(rows),
            non_test_row_overlap_count=0,
            feature_extractor=lambda _: (_ for _ in ()).throw(AssertionError("not called")),
        )


def test_joint_cache_failure_preserves_failed_lifecycle(monkeypatch, tmp_path: Path) -> None:
    rows = _rows().iloc[:1].copy()

    class _SourceCache:
        def __init__(self, _path: Path) -> None:
            pass

        def get_arrays(self, _source: str, _keys: list[str]) -> dict[str, np.ndarray]:
            return {"bcg": np.zeros(18000, dtype=np.float32)}

    monkeypatch.setattr(cache, "WholeNightCache", _SourceCache)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="fixture failure"):
        build_context_length_research_test_w_cache_from_rows(
            output_dir=output,
            index_path=tmp_path / "index.csv",
            rows=rows,
            dataset_index_sha256=cache.EXPECTED_DATASET_INDEX_SHA256,
            expected_count=1,
            expected_samp_ids=1,
            expected_row_ids_sha256=_row_hash(rows),
            non_test_row_overlap_count=0,
            feature_extractor=lambda _: (_ for _ in ()).throw(RuntimeError("fixture failure")),
        )
    lifecycle = json.loads((output / "lifecycle.json").read_text(encoding="utf-8"))
    assert lifecycle["status"] == "failed"
    assert lifecycle["error_type"] == "RuntimeError"


def test_joint_cache_cli_requires_explicit_confirmation_without_accessing_test() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/build_paper_context_length_research_test_w_cache_v1.py")],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "confirm-research-test-cache-build" in result.stderr


def test_joint_cache_row_contract_rejects_non_test_split() -> None:
    rows = _rows()
    rows.loc[0, "split"] = "val"
    with pytest.raises(ValueError, match="schema/count/split"):
        cache._validate_test_rows(
            rows,
            expected_count=2,
            expected_samp_ids=2,
            expected_row_ids_sha256=_row_hash(rows),
        )
