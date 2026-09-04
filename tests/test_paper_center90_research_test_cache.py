from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import resp_train.paper_evidence.center90_research_test_cache as cache
from resp_train.paper_evidence.center90_cache import fixed_center90_w_transform_spec
from resp_train.paper_evidence.center_context_cache import fixed_center_w_transform_spec
from resp_train.paper_evidence.center90_research_test_cache import (
    Center90ResearchTestWCacheReader,
    build_center90_research_test_w_cache_from_rows,
    fixed_cache_identity,
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
    return hashlib.sha256(rows["dataset_row_id"].to_numpy(dtype=np.int64).tobytes(order="C")).hexdigest()


def test_identity_generates_only_135_and_reuses_frozen_90_180() -> None:
    identity = fixed_cache_identity()
    assert identity["task"] == {"input_samples": [9000, 13500, 18000], "output_samples": 9000}
    assert identity["generated_input_samples"] == [13500]
    assert identity["reused_input_samples"] == [9000, 18000]
    assert identity["transform_spec"]["shape"] == [49, 270]
    assert fixed_center90_w_transform_spec(9000) == fixed_center_w_transform_spec(9000)
    assert fixed_center90_w_transform_spec(18000) == fixed_center_w_transform_spec(18000)


def test_fixture_builder_reads_input_once_per_row_and_writes_135_cache(monkeypatch, tmp_path: Path) -> None:
    rows = _rows()
    source_calls: list[tuple[str, tuple[str, ...]]] = []
    feature_lengths: list[int] = []

    class _SourceCache:
        def __init__(self, path: Path) -> None:
            assert path == (tmp_path / "index.csv").resolve()

        def get_arrays(self, source: str, keys: list[str]) -> dict[str, np.ndarray]:
            source_calls.append((source, tuple(keys)))
            return {"bcg": np.linspace(-1.0, 1.0, 18000, dtype=np.float32)}

    def _extract(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        feature_lengths.append(int(values.size))
        return np.ones((49, 270), dtype=np.float32), np.linspace(0.05, 2.0, 49)

    monkeypatch.setattr(cache, "WholeNightCache", _SourceCache)
    output = tmp_path / "cache"
    manifest_path = build_center90_research_test_w_cache_from_rows(
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
    assert manifest["test_input_read"] is True
    assert manifest["test_target_array_read"] is False
    assert manifest["feature"]["shape"] == [2, 49, 270]
    assert len(source_calls) == 2
    assert feature_lengths == [13500, 13500]
    reader = Center90ResearchTestWCacheReader(output)
    assert reader.get(8).shape == (49, 270)
    reader.verify_rows([7, 8])


def test_builder_rejects_overlap_and_preserves_failed_lifecycle(monkeypatch, tmp_path: Path) -> None:
    rows = _rows().iloc[:1].copy()
    with pytest.raises(RuntimeError, match="零重叠"):
        build_center90_research_test_w_cache_from_rows(
            output_dir=tmp_path / "overlap",
            index_path=tmp_path / "index.csv",
            rows=rows,
            dataset_index_sha256=cache.EXPECTED_DATASET_INDEX_SHA256,
            expected_count=1,
            expected_samp_ids=1,
            expected_row_ids_sha256=_row_hash(rows),
            non_test_row_overlap_count=1,
            feature_extractor=lambda _: (_ for _ in ()).throw(AssertionError("not called")),
        )

    class _SourceCache:
        def __init__(self, _path: Path) -> None:
            pass

        def get_arrays(self, _source: str, _keys: list[str]) -> dict[str, np.ndarray]:
            return {"bcg": np.zeros(18000, dtype=np.float32)}

    monkeypatch.setattr(cache, "WholeNightCache", _SourceCache)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="fixture failure"):
        build_center90_research_test_w_cache_from_rows(
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


def test_existing_output_is_not_overwritten(tmp_path: Path) -> None:
    rows = _rows().iloc[:1].copy()
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        build_center90_research_test_w_cache_from_rows(
            output_dir=output,
            index_path=tmp_path / "index.csv",
            rows=rows,
            dataset_index_sha256=cache.EXPECTED_DATASET_INDEX_SHA256,
            expected_count=1,
            expected_samp_ids=1,
            expected_row_ids_sha256=_row_hash(rows),
            non_test_row_overlap_count=0,
            feature_extractor=lambda _: (_ for _ in ()).throw(AssertionError("not called")),
        )


def test_cli_requires_explicit_confirmation_before_test_access() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/build_paper_center90_research_test_w_cache_v1.py")],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "confirm-research-test-cache-build" in result.stderr
