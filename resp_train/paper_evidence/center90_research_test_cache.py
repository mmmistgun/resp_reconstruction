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
from resp_train.paper_evidence.center90_cache import (
    center90_cwt_features,
    fixed_center90_w_transform_spec,
    sha256_file,
)
from resp_train.paper_evidence.center90_data import center90_input_bounds
from resp_train.paper_evidence.center_context_cache import reduced_target_frequencies
from resp_train.paper_evidence.context_length_research_test_cache import (
    EXPECTED_DATASET_INDEX_SHA256,
    EXPECTED_TEST_ROW_IDS_SHA256,
    EXPECTED_TEST_SAMP_IDS,
    EXPECTED_TEST_WINDOWS,
    INPUT_KEY,
    TARGET_KEY,
    TEST_SAMPLE_SEED,
    TEST_SAMPLE_STRATEGY,
    _validate_test_rows,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_ID = "paper-center90-context-research-test-v1-20260904"
CACHE_SCHEMA_VERSION = "paper-center90-context-research-test-w-cache-v1"
INPUT_SAMPLES = 13500
DEFAULT_CONFIG = REPO_ROOT / "configs/paper_evidence_v1/center90_context_v1.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs/paper_evidence_v1/center90_context_research_test_w_cache"
VALIDATION_SUMMARY_ROOT = REPO_ROOT / "runs/paper_evidence_v1/center90_context/validation_summary"
VALIDATION_SUMMARY_RECEIPT_SHA256 = "b8e3428ec4f43094ec7508b4897f9d2b07a54c702dc07749e3fd5fbb9f959886"
VALIDATION_SUMMARY_MANIFEST_SHA256 = "cc532fe3930861dd27ef4eb0f8efbb55381ffd26ad5cb88b7cf5f3c0839d47bc"
REUSED_CACHE_ROOT = REPO_ROOT / (
    "runs/paper_evidence_v1/context_length_research_test_w_cache/"
    "cf89c6e678bb243801c0ca577ec14d69d724603e51b3335c4e1f474e7f999d5c"
)
REUSED_CACHE_MANIFEST_SHA256 = "9b475926258d129851fb7b9c10d2b342ac2e833b121842b18437585059bf1b98"
FeatureExtractor = Callable[[np.ndarray | torch.Tensor], tuple[np.ndarray, np.ndarray]]


def fixed_cache_identity(*, row_ids_sha256: str = EXPECTED_TEST_ROW_IDS_SHA256) -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "schema_version": CACHE_SCHEMA_VERSION,
        "role": "center90_research_test_135s_input_w_cache",
        "task": {"input_samples": [9000, 13500, 18000], "output_samples": 9000},
        "generated_input_samples": [INPUT_SAMPLES],
        "reused_input_samples": [9000, 18000],
        "reused_cache_manifest_sha256": REUSED_CACHE_MANIFEST_SHA256,
        "dataset_index_sha256": EXPECTED_DATASET_INDEX_SHA256,
        "split": "test",
        "row_ids_sha256": str(row_ids_sha256),
        "sample_strategy": TEST_SAMPLE_STRATEGY,
        "sample_seed": TEST_SAMPLE_SEED,
        "input_key": INPUT_KEY,
        "input_slice": list(center90_input_bounds(INPUT_SAMPLES)),
        "transform_spec": fixed_center90_w_transform_spec(INPUT_SAMPLES),
        "validation_summary": {
            "receipt_sha256": VALIDATION_SUMMARY_RECEIPT_SHA256,
            "manifest_sha256": VALIDATION_SUMMARY_MANIFEST_SHA256,
            "runs": 18,
        },
    }


def cache_identity_sha256(*, row_ids_sha256: str = EXPECTED_TEST_ROW_IDS_SHA256) -> str:
    return _sha256_json(fixed_cache_identity(row_ids_sha256=row_ids_sha256))


def build_center90_research_test_w_cache(*, show_progress: bool = True) -> Path:
    _assert_clean_repository()
    _verify_frozen_inputs()
    cfg = OmegaConf.load(DEFAULT_CONFIG)
    _validate_source_config(cfg)
    index_path = (Path(str(cfg.data.dataset_root)) / str(cfg.data.index_csv)).resolve()
    if not index_path.is_file() or sha256_file(index_path) != EXPECTED_DATASET_INDEX_SHA256:
        raise RuntimeError("center90 research-test dataset index identity 漂移")
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
    non_test_ids = audited.loc[audited["split"].astype(str).isin({"train", "val"}), "dataset_row_id"]
    overlap = int(np.intersect1d(rows["dataset_row_id"].to_numpy(), non_test_ids.to_numpy()).size)
    return build_center90_research_test_w_cache_from_rows(
        output_dir=DEFAULT_OUTPUT_ROOT / cache_identity_sha256(),
        index_path=index_path,
        rows=rows,
        dataset_index_sha256=EXPECTED_DATASET_INDEX_SHA256,
        expected_count=EXPECTED_TEST_WINDOWS,
        expected_samp_ids=EXPECTED_TEST_SAMP_IDS,
        expected_row_ids_sha256=EXPECTED_TEST_ROW_IDS_SHA256,
        non_test_row_overlap_count=overlap,
        feature_extractor=center90_cwt_features,
        show_progress=show_progress,
    )


def build_center90_research_test_w_cache_from_rows(
    *,
    output_dir: str | Path,
    index_path: str | Path,
    rows: pd.DataFrame,
    dataset_index_sha256: str,
    expected_count: int,
    expected_samp_ids: int,
    expected_row_ids_sha256: str,
    non_test_row_overlap_count: int,
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
    if int(non_test_row_overlap_count) != 0:
        raise RuntimeError("center90 research-test rows 与非 test rows 必须零重叠")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    lifecycle_path = output / "lifecycle.json"
    _write_json(lifecycle_path, {"status": "running"})
    try:
        source_cache = WholeNightCache(Path(index_path).resolve())
        row_ids = rows["dataset_row_id"].to_numpy(dtype=np.int64)
        row_array = np.lib.format.open_memmap(
            output / "test_row_ids.npy", mode="w+", dtype=np.int64, shape=(len(rows),)
        )
        features = np.lib.format.open_memmap(
            output / "test_w_13500.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(rows), 49, 270),
        )
        frequencies_identity: np.ndarray | None = None
        source_keys: set[str] = set()
        start, stop = center90_input_bounds(INPUT_SAMPLES)
        iterator = tqdm(
            rows.itertuples(index=False),
            total=len(rows),
            desc="center90 research-test 135s W cache",
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
            feature, frequencies = feature_extractor(parent[start:stop])
            feature = np.asarray(feature, dtype=np.float32)
            frequencies = np.asarray(frequencies, dtype=np.float64)
            if feature.shape != (49, 270) or not np.isfinite(feature).all():
                raise FloatingPointError(f"row={row.dataset_row_id} 135 s W feature 不合格")
            _validate_frequencies(frequencies)
            if frequencies_identity is None:
                frequencies_identity = frequencies.copy()
            elif not np.array_equal(frequencies_identity, frequencies):
                raise RuntimeError("135 s mapped frequencies 在 test samples 间漂移")
            row_array[position] = int(row.dataset_row_id)
            features[position] = feature
        row_array.flush()
        features.flush()
        if frequencies_identity is None:
            raise RuntimeError("center90 research-test cache 缺少 frequency identity")
        np.save(output / "w_frequencies_13500.npy", frequencies_identity, allow_pickle=False)
        identity = fixed_cache_identity(row_ids_sha256=expected_row_ids_sha256)
        nominal = reduced_target_frequencies()
        manifest = {
            **identity,
            "cache_identity_sha256": _sha256_json(identity),
            "complete": True,
            "input_only": True,
            "test_input_read": True,
            "test_target_array_read": False,
            "train_signal_or_target_read": False,
            "validation_signal_or_target_read": False,
            "non_test_row_overlap_count": 0,
            "model_training_used": False,
            "model_inference_used": False,
            "dataset_index": str(Path(index_path).resolve()),
            "dataset_index_sha256": str(dataset_index_sha256),
            "test_windows": len(rows),
            "test_samp_ids": int(rows["samp_id"].nunique()),
            "source_keys": sorted(source_keys),
            "feature": {
                "shape": [len(rows), 49, 270],
                "dtype": "float32",
                "finite": True,
                "frequencies_non_decreasing": True,
                "duplicate_center_count": int(np.sum(np.diff(frequencies_identity) == 0.0)),
                "max_nominal_frequency_error_hz": float(np.max(np.abs(frequencies_identity - nominal))),
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


class Center90ResearchTestWCacheReader:
    def __init__(self, root: str | Path, *, expected_manifest_sha256: str | None = None) -> None:
        self.root = Path(root).resolve()
        manifest_path = self.root / "cache_manifest.json"
        if expected_manifest_sha256 is not None and sha256_file(manifest_path) != expected_manifest_sha256:
            raise RuntimeError("center90 research-test W cache manifest SHA-256 漂移")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_identity = fixed_cache_identity(
            row_ids_sha256=str(self.manifest.get("row_ids_sha256", ""))
        )
        if (
            any(self.manifest.get(key) != value for key, value in expected_identity.items())
            or self.manifest.get("cache_identity_sha256") != _sha256_json(expected_identity)
            or self.manifest.get("complete") is not True
            or self.manifest.get("input_only") is not True
            or self.manifest.get("test_target_array_read") is not False
            or self.manifest.get("non_test_row_overlap_count") != 0
            or self.manifest.get("model_training_used") is not False
            or self.manifest.get("model_inference_used") is not False
        ):
            raise RuntimeError("center90 research-test W cache manifest identity 漂移")
        self.row_ids = self._open("test_row_ids.npy")
        self.features = self._open("test_w_13500.npy")
        self.frequencies_hz = self._open("w_frequencies_13500.npy")
        if (
            self.row_ids.shape != (int(self.manifest["test_windows"]),)
            or self.row_ids.dtype != np.int64
            or self.features.shape != tuple(self.manifest["feature"]["shape"])
            or self.features.dtype != np.float32
            or hashlib.sha256(np.asarray(self.row_ids).tobytes(order="C")).hexdigest()
            != self.manifest["row_ids_sha256"]
            or np.unique(self.row_ids).size != self.row_ids.size
            or not np.all(np.diff(self.row_ids) > 0)
        ):
            raise RuntimeError("center90 research-test W cache array identity 漂移")
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
            raise RuntimeError(f"center90 research-test W cache 文件 identity 错误: {filename}")
        return np.load(path, mmap_mode="r", allow_pickle=False)

    def get(self, dataset_row_id: int) -> torch.Tensor:
        position = int(np.searchsorted(self.row_ids, int(dataset_row_id)))
        if position >= len(self.row_ids) or int(self.row_ids[position]) != int(dataset_row_id):
            raise KeyError(f"center90 research-test cache 缺少 dataset_row_id={dataset_row_id}")
        value = np.array(self.features[position], copy=True)
        if not np.isfinite(value).all():
            raise FloatingPointError("center90 research-test W feature 包含 NaN/Inf")
        return torch.from_numpy(value)

    def verify_rows(self, dataset_row_ids: Iterable[int]) -> None:
        requested = np.asarray([int(value) for value in dataset_row_ids], dtype=np.int64)
        positions = np.searchsorted(self.row_ids, requested)
        valid = positions < len(self.row_ids)
        matched = np.zeros(requested.shape, dtype=np.bool_)
        matched[valid] = np.asarray(self.row_ids)[positions[valid]] == requested[valid]
        if requested.size == 0 or not bool(matched.all()):
            raise KeyError("center90 research-test W cache rows 不完整")


def _verify_frozen_inputs() -> None:
    receipt = VALIDATION_SUMMARY_ROOT / "summary_receipt.json"
    manifest = VALIDATION_SUMMARY_ROOT / "artifact_manifest.json"
    reused = REUSED_CACHE_ROOT / "cache_manifest.json"
    if (
        sha256_file(receipt) != VALIDATION_SUMMARY_RECEIPT_SHA256
        or sha256_file(manifest) != VALIDATION_SUMMARY_MANIFEST_SHA256
        or sha256_file(reused) != REUSED_CACHE_MANIFEST_SHA256
    ):
        raise RuntimeError("center90 research-test 冻结输入 provenance 漂移")
    summary = json.loads(receipt.read_text(encoding="utf-8"))
    if (
        summary.get("status") != "complete"
        or summary.get("counts", {}).get("actual_runs") != 18
        or summary.get("access", {}).get("research_test_accessed") is not False
        or summary.get("decision", {}).get("model_or_length_selection_performed") is not False
    ):
        raise RuntimeError("center90 validation summary 未闭合 test 前置条件")


def _validate_source_config(cfg: Any) -> None:
    expected = {
        "dataset_root": "/mnt/disk_code/marques/resp_prepare/dataset/20260620_research_v2_resp_reconstruction_stage2_1_segrobustz_bcgstagee_log1psoftz_robustconf",
        "index_csv": "training/dataset_index.csv",
        "test_split": "test",
        "bcg_input_key": INPUT_KEY,
        "target_key": TARGET_KEY,
    }
    for key, value in expected.items():
        if str(cfg.data[key]) != value:
            raise RuntimeError(f"center90 research-test source config {key} 漂移")
    if int(cfg.window.parent_samples) != 18000 or int(cfg.window.target_fs) != 100:
        raise RuntimeError("center90 research-test parent/fs 漂移")


def _validate_frequencies(values: np.ndarray) -> None:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (49,) or not np.isfinite(array).all() or not np.all(np.diff(array) >= 0.0):
        raise RuntimeError("center90 research-test frequencies 必须为 49 个有限非降值")


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
        raise RuntimeError("center90 research-test W cache 要求干净 Git 工作树")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "Center90ResearchTestWCacheReader",
    "INPUT_SAMPLES",
    "PROTOCOL_ID",
    "build_center90_research_test_w_cache",
    "build_center90_research_test_w_cache_from_rows",
    "cache_identity_sha256",
    "fixed_cache_identity",
]
