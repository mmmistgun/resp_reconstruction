from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from resp_train.data.cache import WholeNightCache
from resp_train.data.index import filter_index
from resp_train.data.research_v2 import read_research_v2_index
from resp_train.paper_evidence.center30_cache import (
    center30_cwt_features,
    fixed_center30_w_transform_spec,
    sha256_file,
)
from resp_train.paper_evidence.center30_config import CENTER30_INPUT_SAMPLES
from resp_train.paper_evidence.center30_data import center30_input_bounds


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_ID = "paper-center30-context-research-test-v1-20260904"
CACHE_SCHEMA_VERSION = "paper-center30-context-research-test-w-cache-v1"
DEFAULT_SOURCE_CONFIG = REPO_ROOT / "configs/paper_evidence_v1/center30_context_v1.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs/paper_evidence_v1/center30_context_research_test_w_cache"
VALIDATION_SUMMARY_ROOT = REPO_ROOT / "runs/paper_evidence_v1/center30_context/p4s3_validation_summary"
VALIDATION_SUMMARY_RECEIPT_SHA256 = "2c533117c735e5bedb65f31ca77fa9dd22663433c0a284b4ee7bd9dc0beee2ef"
VALIDATION_SUMMARY_MANIFEST_SHA256 = "b18bbed8ab826b8a6e415866c4ad1b9d5171fd4eebc0d5c4090544428fc19bba"
EXPECTED_DATASET_INDEX_SHA256 = "f65ae6524632187c7c2795bd7b5e25434afb649b5a04fe39084b70cc59f9b04f"
EXPECTED_TEST_WINDOWS = 2310
EXPECTED_TEST_SAMP_IDS = 8
EXPECTED_TEST_ROW_IDS_SHA256 = "184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8"
TEST_SAMPLE_STRATEGY = "stratified_random"
TEST_SAMPLE_SEED = 20260612
INPUT_KEY = "bcg_rawish_segment_soft_z_key"
TARGET_KEY = "target_waveform_segment_soft_z_key"
FeatureExtractor = Callable[[np.ndarray | torch.Tensor], tuple[np.ndarray, np.ndarray]]


def fixed_research_test_cache_identity(*, row_ids_sha256: str = EXPECTED_TEST_ROW_IDS_SHA256) -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "schema_version": CACHE_SCHEMA_VERSION,
        "role": "research_test_input_only_w_cache",
        "dataset_index_sha256": EXPECTED_DATASET_INDEX_SHA256,
        "split": "test",
        "row_ids_sha256": str(row_ids_sha256),
        "sample_strategy": TEST_SAMPLE_STRATEGY,
        "sample_seed": TEST_SAMPLE_SEED,
        "input_key": INPUT_KEY,
        "input_samples": list(CENTER30_INPUT_SAMPLES),
        "transform_specs": {
            str(length): fixed_center30_w_transform_spec(length) for length in CENTER30_INPUT_SAMPLES
        },
        "validation_summary_receipt_sha256": VALIDATION_SUMMARY_RECEIPT_SHA256,
        "validation_summary_manifest_sha256": VALIDATION_SUMMARY_MANIFEST_SHA256,
    }


def research_test_cache_identity_sha256(*, row_ids_sha256: str = EXPECTED_TEST_ROW_IDS_SHA256) -> str:
    return _sha256_json(fixed_research_test_cache_identity(row_ids_sha256=row_ids_sha256))


def build_center30_research_test_w_cache(*, show_progress: bool = True) -> Path:
    _assert_clean_repository()
    _verify_validation_summary()
    cfg = OmegaConf.load(DEFAULT_SOURCE_CONFIG)
    _validate_source_config(cfg)
    index_path = (Path(str(cfg.data.dataset_root)) / str(cfg.data.index_csv)).resolve()
    if not index_path.is_file() or sha256_file(index_path) != EXPECTED_DATASET_INDEX_SHA256:
        raise RuntimeError("center30 research-test dataset index 缺失或 SHA-256 漂移")
    audited = read_research_v2_index(cfg.data.dataset_root, cfg.data.index_csv, cfg)
    rows = filter_index(
        audited,
        cfg,
        split="test",
        max_windows=None,
        sample_strategy=TEST_SAMPLE_STRATEGY,
        sample_seed=TEST_SAMPLE_SEED,
    )
    _validate_test_rows(
        rows,
        expected_count=EXPECTED_TEST_WINDOWS,
        expected_samp_ids=EXPECTED_TEST_SAMP_IDS,
        expected_row_ids_sha256=EXPECTED_TEST_ROW_IDS_SHA256,
    )
    output = DEFAULT_OUTPUT_ROOT / research_test_cache_identity_sha256()
    return build_center30_research_test_w_cache_from_rows(
        output_dir=output,
        index_path=index_path,
        rows=rows,
        dataset_index_sha256=EXPECTED_DATASET_INDEX_SHA256,
        expected_count=EXPECTED_TEST_WINDOWS,
        expected_samp_ids=EXPECTED_TEST_SAMP_IDS,
        expected_row_ids_sha256=EXPECTED_TEST_ROW_IDS_SHA256,
        feature_extractor=center30_cwt_features,
        show_progress=show_progress,
    )


def build_center30_research_test_w_cache_from_rows(
    *,
    output_dir: str | Path,
    index_path: str | Path,
    rows: pd.DataFrame,
    dataset_index_sha256: str,
    expected_count: int,
    expected_samp_ids: int,
    expected_row_ids_sha256: str,
    feature_extractor: FeatureExtractor,
    show_progress: bool = False,
) -> Path:
    rows = rows.copy().reset_index(drop=True)
    _validate_test_rows(
        rows,
        expected_count=expected_count,
        expected_samp_ids=expected_samp_ids,
        expected_row_ids_sha256=expected_row_ids_sha256,
    )
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    lifecycle_path = output / "lifecycle.json"
    _write_json(lifecycle_path, {"status": "running"})
    try:
        source_cache = WholeNightCache(Path(index_path).resolve())
        row_ids = rows["dataset_row_id"].to_numpy(dtype=np.int64)
        row_ids_array = np.lib.format.open_memmap(
            output / "test_row_ids.npy", mode="w+", dtype=np.int64, shape=(len(rows),)
        )
        feature_arrays = {
            length: np.lib.format.open_memmap(
                output / f"test_w_{length}.npy",
                mode="w+",
                dtype=np.float32,
                shape=(len(rows), 49, length // 50),
            )
            for length in CENTER30_INPUT_SAMPLES
        }
        frequency_identity: dict[int, np.ndarray] = {}
        source_keys: set[str] = set()
        iterator = tqdm(
            rows.itertuples(index=False),
            total=len(rows),
            desc="center30 research-test W cache",
            disable=not show_progress,
        )
        for position, row in enumerate(iterator):
            parent_start = int(row.window_start_sample)
            parent_stop = int(row.window_end_sample)
            if parent_stop - parent_start != 18000:
                raise RuntimeError(f"row={row.dataset_row_id} 不是冻结 180 s parent")
            source_key = str(row.bcg_signal_key)
            source_keys.add(source_key)
            source = source_cache.get_arrays(str(row.source_npz), [source_key])[source_key]
            parent = np.asarray(source[parent_start:parent_stop], dtype=np.float32)
            if parent.shape != (18000,) or not np.isfinite(parent).all():
                raise FloatingPointError(f"row={row.dataset_row_id} test input shape/finite 错误")
            row_ids_array[position] = int(row.dataset_row_id)
            for length in CENTER30_INPUT_SAMPLES:
                start, stop = center30_input_bounds(length)
                feature, frequencies = feature_extractor(parent[start:stop])
                feature = np.asarray(feature, dtype=np.float32)
                frequencies = np.asarray(frequencies, dtype=np.float64)
                if feature.shape != (49, length // 50) or not np.isfinite(feature).all():
                    raise FloatingPointError(f"row={row.dataset_row_id}/length={length} W feature 不合格")
                _validate_frequencies(frequencies)
                if length not in frequency_identity:
                    frequency_identity[length] = frequencies.copy()
                elif not np.array_equal(frequency_identity[length], frequencies):
                    raise RuntimeError(f"length={length} mapped frequencies 在 test sample 间漂移")
                feature_arrays[length][position] = feature
        row_ids_array.flush()
        for array in feature_arrays.values():
            array.flush()
        for length, values in frequency_identity.items():
            np.save(output / f"w_frequencies_{length}.npy", values, allow_pickle=False)
        identity = fixed_research_test_cache_identity(row_ids_sha256=expected_row_ids_sha256)
        manifest = {
            **identity,
            "cache_identity_sha256": _sha256_json(identity),
            "complete": True,
            "input_only": True,
            "test_read": True,
            "test_input_read": True,
            "test_target_array_read": False,
            "target_read": False,
            "train_signal_or_target_read": False,
            "validation_signal_or_target_read": False,
            "test_cache_created": True,
            "model_training_used": False,
            "model_inference_used": False,
            "dataset_index": str(Path(index_path).resolve()),
            "dataset_index_sha256": str(dataset_index_sha256),
            "test_windows": len(rows),
            "test_samp_ids": int(rows["samp_id"].nunique()),
            "source_keys": sorted(source_keys),
            "features": {
                str(length): {
                    "shape": [len(rows), 49, length // 50],
                    "dtype": "float32",
                    "finite": True,
                    "frequencies_non_decreasing": True,
                    "duplicate_center_count": int(np.sum(np.diff(frequency_identity[length]) == 0.0)),
                }
                for length in CENTER30_INPUT_SAMPLES
            },
            "dependencies": _dependency_versions(),
            "git": _git_identity(),
        }
        manifest["files"] = _inventory_files(output)
        _write_json(output / "cache_manifest.json", manifest)
    except BaseException as exc:
        _write_json(
            lifecycle_path,
            {"status": "failed", "error_type": type(exc).__name__, "error_message": str(exc)},
        )
        raise
    _write_json(lifecycle_path, {"status": "complete"})
    return output / "cache_manifest.json"


class Center30ResearchTestWCacheReader:
    def __init__(
        self,
        root: str | Path,
        *,
        input_samples: int,
        expected_manifest_sha256: str | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.input_samples = int(input_samples)
        if self.input_samples not in CENTER30_INPUT_SAMPLES:
            raise ValueError(f"research-test W input_samples 只允许 {list(CENTER30_INPUT_SAMPLES)}")
        manifest_path = self.root / "cache_manifest.json"
        if expected_manifest_sha256 is not None and sha256_file(manifest_path) != expected_manifest_sha256:
            raise RuntimeError("research-test W cache manifest SHA-256 漂移")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_identity = fixed_research_test_cache_identity(
            row_ids_sha256=str(self.manifest.get("row_ids_sha256", ""))
        )
        if (
            any(self.manifest.get(key) != value for key, value in expected_identity.items())
            or self.manifest.get("cache_identity_sha256") != _sha256_json(expected_identity)
            or self.manifest.get("complete") is not True
            or self.manifest.get("input_only") is not True
            or self.manifest.get("test_read") is not True
            or self.manifest.get("test_target_array_read") is not False
            or self.manifest.get("target_read") is not False
            or self.manifest.get("model_training_used") is not False
            or self.manifest.get("model_inference_used") is not False
        ):
            raise RuntimeError("research-test W cache manifest identity/访问合同不合格")
        self.row_ids = self._open("test_row_ids.npy")
        self.features = self._open(f"test_w_{self.input_samples}.npy")
        self.frequencies_hz = self._open(f"w_frequencies_{self.input_samples}.npy")
        if (
            self.row_ids.shape != (int(self.manifest["test_windows"]),)
            or self.row_ids.dtype != np.int64
            or self.features.shape != (len(self.row_ids), 49, self.input_samples // 50)
            or self.features.dtype != np.float32
            or hashlib.sha256(np.asarray(self.row_ids).tobytes(order="C")).hexdigest()
            != self.manifest["row_ids_sha256"]
            or np.unique(self.row_ids).size != self.row_ids.size
            or not np.all(np.diff(self.row_ids) > 0)
        ):
            raise RuntimeError("research-test W cache array identity 不合格")
        _validate_frequencies(self.frequencies_hz)

    def _open(self, filename: str) -> np.ndarray:
        path = self.root / filename
        record = self.manifest.get("files", {}).get(filename)
        if (
            not isinstance(record, Mapping)
            or not path.is_file()
            or path.stat().st_size != int(record.get("size_bytes", -1))
            or sha256_file(path) != record.get("sha256")
        ):
            raise RuntimeError(f"research-test W cache 文件 identity 错误: {filename}")
        return np.load(path, mmap_mode="r", allow_pickle=False)

    def get(self, dataset_row_id: int) -> torch.Tensor:
        position = int(np.searchsorted(self.row_ids, int(dataset_row_id)))
        if position >= len(self.row_ids) or int(self.row_ids[position]) != int(dataset_row_id):
            raise KeyError(f"research-test W cache 缺少 dataset_row_id={dataset_row_id}")
        value = np.array(self.features[position], copy=True)
        if not np.isfinite(value).all():
            raise FloatingPointError("research-test W cache 读取到 NaN/Inf")
        return torch.from_numpy(value)

    def verify_rows(self, dataset_row_ids: Iterable[int]) -> None:
        requested = np.asarray([int(value) for value in dataset_row_ids], dtype=np.int64)
        positions = np.searchsorted(self.row_ids, requested)
        valid = positions < len(self.row_ids)
        matched = np.zeros(requested.shape, dtype=np.bool_)
        matched[valid] = np.asarray(self.row_ids)[positions[valid]] == requested[valid]
        if requested.size == 0 or not bool(matched.all()):
            raise KeyError("research-test W cache rows 不完整")


def _validate_test_rows(
    rows: pd.DataFrame,
    *,
    expected_count: int,
    expected_samp_ids: int,
    expected_row_ids_sha256: str,
) -> None:
    required = {
        "dataset_row_id",
        "split",
        "samp_id",
        "source_npz",
        "bcg_signal_key",
        "window_start_sample",
        "window_end_sample",
    }
    missing = sorted(required - set(rows.columns))
    if missing or len(rows) != expected_count or set(rows.get("split", pd.Series(dtype=str)).astype(str)) != {"test"}:
        raise ValueError(f"research-test rows schema/count/split 不合格: missing={missing}, rows={len(rows)}")
    row_ids = rows["dataset_row_id"].to_numpy(dtype=np.int64)
    if np.unique(row_ids).size != row_ids.size or not np.all(np.diff(row_ids) > 0):
        raise ValueError("research-test row_ids 必须严格递增且无重复")
    actual_hash = hashlib.sha256(row_ids.tobytes(order="C")).hexdigest()
    if actual_hash != expected_row_ids_sha256 or int(rows["samp_id"].nunique()) != expected_samp_ids:
        raise RuntimeError("research-test row/samp_id identity 漂移")


def _validate_source_config(cfg: Any) -> None:
    if (
        str(cfg.data.dataset_root)
        != "/mnt/disk_code/marques/resp_prepare/dataset/20260620_research_v2_resp_reconstruction_stage2_1_segrobustz_bcgstagee_log1psoftz_robustconf"
        or str(cfg.data.index_csv) != "training/dataset_index.csv"
        or str(cfg.data.test_split) != "test"
        or str(cfg.data.bcg_input_key) != INPUT_KEY
        or str(cfg.data.target_key) != TARGET_KEY
        or int(cfg.window.parent_samples) != 18000
        or int(cfg.window.target_fs) != 100
    ):
        raise RuntimeError("center30 research-test source config identity 漂移")


def _verify_validation_summary() -> None:
    receipt_path = VALIDATION_SUMMARY_ROOT / "summary_receipt.json"
    manifest_path = VALIDATION_SUMMARY_ROOT / "artifact_manifest.json"
    if (
        sha256_file(receipt_path) != VALIDATION_SUMMARY_RECEIPT_SHA256
        or sha256_file(manifest_path) != VALIDATION_SUMMARY_MANIFEST_SHA256
    ):
        raise RuntimeError("center30 validation summary provenance SHA-256 漂移")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "complete"
        or receipt.get("counts", {}).get("actual_runs") != 24
        or receipt.get("access", {}).get("research_test_accessed") is not False
        or receipt.get("decision", {}).get("model_or_length_selection_performed") is not False
    ):
        raise RuntimeError("center30 validation summary 未闭合 test 前置条件")


def _validate_frequencies(values: np.ndarray) -> None:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (49,) or not np.isfinite(array).all() or not np.all(np.diff(array) >= 0.0):
        raise RuntimeError("research-test mapped frequencies 必须为 49 个有限非降值")


def _inventory_files(directory: Path) -> dict[str, dict[str, Any]]:
    return {
        path.name: {"size_bytes": int(path.stat().st_size), "sha256": sha256_file(path)}
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name not in {"cache_manifest.json", "lifecycle.json"}
    }


def _dependency_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for distribution in ("numpy", "pandas", "torch", "ssqueezepy"):
        try:
            result[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[distribution] = None
    return result


def _git_identity() -> dict[str, Any]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "error": None if commit.returncode == 0 and status.returncode == 0 else "git identity failed",
    }


def _assert_clean_repository() -> None:
    identity = _git_identity()
    if identity["error"] is not None or identity["dirty"] is not False:
        raise RuntimeError("完整 center30 research-test W cache 要求干净 Git 工作树")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "Center30ResearchTestWCacheReader",
    "EXPECTED_TEST_ROW_IDS_SHA256",
    "EXPECTED_TEST_SAMP_IDS",
    "EXPECTED_TEST_WINDOWS",
    "PROTOCOL_ID",
    "build_center30_research_test_w_cache",
    "build_center30_research_test_w_cache_from_rows",
    "fixed_research_test_cache_identity",
    "research_test_cache_identity_sha256",
]
