from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from resp_train.crd.tf_v1_features import MORLET_MU, SAMPLE_RATE
from resp_train.paper_evidence.center90_config import CENTER90_PROTOCOL_ID
from resp_train.paper_evidence.center90_data import center90_input_bounds
from resp_train.paper_evidence.center_context_cache import (
    POOL_SAMPLES,
    REDUCED_SCALE_COUNT,
    REDUCED_SCALE_INDICES,
    CenterContextWCacheReader,
    fixed_center_w_transform_spec,
    reduced_scales_and_frequencies,
    reduced_target_frequencies,
)


CENTER90_W_CACHE_SCHEMA = "paper-center90-context-w-cache-v1"
CENTER90_NEW_CACHE_SAMPLES = 13500


def fixed_center90_w_transform_spec(input_samples: int) -> dict[str, Any]:
    length = int(input_samples)
    center90_input_bounds(length)
    if length in {9000, 18000}:
        return fixed_center_w_transform_spec(length)
    return {
        "sample_rate_hz": 100.0,
        "input_samples": length,
        "input_slice": list(center90_input_bounds(length)),
        "wavelet": "morlet",
        "mu": 13.4,
        "source_voices_per_octave": 12,
        "effective_voices_per_octave": 6,
        "full_grid_scale_count": 97,
        "selected_full_grid_indices": list(REDUCED_SCALE_INDICES),
        "nominal_scale_count": REDUCED_SCALE_COUNT,
        "length_specific_scale_mapping": True,
        "padtype": "reflect",
        "feature": "log1p_abs_then_mean50",
        "shape": [REDUCED_SCALE_COUNT, length // POOL_SAMPLES],
        "source": "length_specific_cropped_input_only",
    }


def center90_cwt_features(waveform: np.ndarray | torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(torch.as_tensor(waveform, dtype=torch.float32).cpu()).reshape(-1)
    if values.size != CENTER90_NEW_CACHE_SAMPLES:
        raise ValueError(f"center90 新 W cache 输入固定为 {CENTER90_NEW_CACHE_SAMPLES} 点")
    if not np.isfinite(values).all():
        raise FloatingPointError("center90 W 输入包含 NaN/Inf")
    from ssqueezepy import Wavelet, cwt
    from ssqueezepy.experimental import scale_to_freq

    length = int(values.size)
    scales, _ = reduced_scales_and_frequencies(length)
    wavelet = Wavelet(("morlet", {"mu": MORLET_MU}), N=length, dtype="float32")
    coefficients, returned_scales = cwt(
        values,
        wavelet=wavelet,
        scales=scales,
        fs=SAMPLE_RATE,
        padtype="reflect",
        rpadded=False,
        vectorized=True,
        astensor=False,
        nan_checks=False,
    )
    actual = np.asarray(
        scale_to_freq(returned_scales, wavelet, length, fs=SAMPLE_RATE, padtype="reflect"),
        dtype=np.float64,
    )
    order = np.argsort(actual, kind="stable")
    magnitude = np.log1p(np.abs(np.asarray(coefficients)[order])).astype(np.float32)
    pooled = magnitude.reshape(REDUCED_SCALE_COUNT, length // POOL_SAMPLES, POOL_SAMPLES).mean(
        axis=-1, dtype=np.float32
    )
    frequencies = np.ascontiguousarray(actual[order])
    _validate_frequency_identity(frequencies)
    if pooled.shape != (REDUCED_SCALE_COUNT, length // POOL_SAMPLES) or not np.isfinite(pooled).all():
        raise FloatingPointError("center90 W feature shape/finite 错误")
    return np.ascontiguousarray(pooled), frequencies


def build_center90_w_cache_from_rows(
    *,
    output_dir: str | Path,
    index_path: str | Path,
    rows_by_split: dict[str, pd.DataFrame],
    input_samples: int,
    dataset_index_sha256: str,
    feature_extractor: Any = center90_cwt_features,
    complete: bool = True,
) -> Path:
    from resp_train.data.cache import WholeNightCache

    length = int(input_samples)
    if length != CENTER90_NEW_CACHE_SAMPLES:
        raise ValueError("center90 cache builder 只创建 135 s cache")
    if set(rows_by_split) != {"train", "val"}:
        raise ValueError("center90 W cache builder 只允许 train/val")
    input_start, input_stop = center90_input_bounds(length)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    lifecycle = output / "lifecycle.json"
    _write_json(lifecycle, {"status": "running"})
    source_cache = WholeNightCache(Path(index_path).resolve())
    try:
        frequency_identity: np.ndarray | None = None
        split_manifest: dict[str, Any] = {}
        for split in ("train", "val"):
            rows = rows_by_split[split].copy().reset_index(drop=True)
            required = {
                "dataset_row_id",
                "split",
                "source_npz",
                "bcg_signal_key",
                "window_start_sample",
                "window_end_sample",
            }
            missing = sorted(required - set(rows.columns))
            if missing or rows.empty or set(rows["split"].astype(str)) != {split}:
                raise ValueError(f"center90 W {split} rows 身份不完整: {missing}")
            ids = rows["dataset_row_id"].to_numpy(dtype=np.int64)
            if np.unique(ids).size != ids.size or not np.all(np.diff(ids) > 0):
                raise ValueError(f"center90 W {split} row_ids 必须严格递增且无重复")
            id_array = np.lib.format.open_memmap(
                output / f"{split}_row_ids.npy", mode="w+", dtype=np.int64, shape=(len(rows),)
            )
            feature_array = np.lib.format.open_memmap(
                output / f"{split}_w.npy",
                mode="w+",
                dtype=np.float32,
                shape=(len(rows), REDUCED_SCALE_COUNT, length // POOL_SAMPLES),
            )
            source_keys: set[str] = set()
            for position, row in enumerate(rows.itertuples(index=False)):
                parent_start = int(row.window_start_sample)
                parent_stop = int(row.window_end_sample)
                if parent_stop - parent_start != 18000:
                    raise RuntimeError(f"row={row.dataset_row_id} parent 长度不是 18000")
                source_key = str(row.bcg_signal_key)
                source_keys.add(source_key)
                whole = source_cache.get_arrays(str(row.source_npz), [source_key])[source_key]
                parent = np.asarray(whole[parent_start:parent_stop], dtype=np.float32)
                if parent.shape != (18000,) or not np.isfinite(parent).all():
                    raise FloatingPointError(f"row={row.dataset_row_id} parent input shape/finite 错误")
                feature, frequencies = feature_extractor(parent[input_start:input_stop])
                feature = np.asarray(feature, dtype=np.float32)
                frequencies = np.asarray(frequencies, dtype=np.float64)
                if feature.shape != (REDUCED_SCALE_COUNT, length // POOL_SAMPLES) or not np.isfinite(feature).all():
                    raise FloatingPointError(f"row={row.dataset_row_id} W feature shape/finite 错误")
                _validate_frequency_identity(frequencies)
                if frequency_identity is None:
                    frequency_identity = frequencies.copy()
                elif not np.array_equal(frequency_identity, frequencies):
                    raise RuntimeError("center90 mapped frequency identity 在 sample 间漂移")
                id_array[position] = int(row.dataset_row_id)
                feature_array[position] = feature
            id_array.flush()
            feature_array.flush()
            split_manifest[split] = {
                "count": int(len(rows)),
                "row_ids_sha256": hashlib.sha256(ids.tobytes(order="C")).hexdigest(),
                "shape": [len(rows), REDUCED_SCALE_COUNT, length // POOL_SAMPLES],
                "dtype": "float32",
                "finite": True,
                "source_keys": sorted(source_keys),
            }
        if np.intersect1d(
            rows_by_split["train"]["dataset_row_id"].to_numpy(dtype=np.int64),
            rows_by_split["val"]["dataset_row_id"].to_numpy(dtype=np.int64),
        ).size:
            raise RuntimeError("center90 W cache train/validation row_ids 重叠")
        if frequency_identity is None:
            raise RuntimeError("center90 W cache 没有 frequency identity")
        np.save(output / "w_frequencies_hz.npy", frequency_identity, allow_pickle=False)
        spec = fixed_center90_w_transform_spec(length)
        manifest = {
            "protocol_id": CENTER90_PROTOCOL_ID,
            "schema_version": CENTER90_W_CACHE_SCHEMA,
            "input_samples": length,
            "input_slice": [input_start, input_stop],
            "input_only": True,
            "target_read": False,
            "test_read": False,
            "test_cache_created": False,
            "allowed_splits": ["train", "val"],
            "dataset_index_sha256": str(dataset_index_sha256),
            "complete": bool(complete),
            "transform_spec": spec,
            "transform_sha256": _sha256_json(spec),
            "frequency_count": REDUCED_SCALE_COUNT,
            "frequency_monotonic_non_decreasing": True,
            "duplicate_center_count": int(np.sum(np.diff(frequency_identity) == 0.0)),
            "max_nominal_frequency_error_hz": float(
                np.max(np.abs(frequency_identity - reduced_target_frequencies()))
            ),
            "splits": split_manifest,
        }
        manifest["files"] = {
            path.name: {"size_bytes": int(path.stat().st_size), "sha256": sha256_file(path)}
            for path in sorted(output.iterdir())
            if path.is_file() and path.name not in {"cache_manifest.json", "lifecycle.json"}
        }
        _write_json(output / "cache_manifest.json", manifest)
    except BaseException as exc:
        _write_json(
            lifecycle,
            {"status": "failed", "error_type": type(exc).__name__, "error_message": str(exc)},
        )
        raise
    _write_json(lifecycle, {"status": "complete"})
    return output / "cache_manifest.json"


class Center90WCacheReader:
    def __init__(self, root: str | Path, *, split: str, input_samples: int) -> None:
        self.root = Path(root).resolve()
        self.split = str(split)
        self.input_samples = int(input_samples)
        if self.split not in {"train", "val"} or self.input_samples != CENTER90_NEW_CACHE_SAMPLES:
            raise ValueError("center90 native W cache reader 固定为 135 s train/val")
        self.manifest = json.loads((self.root / "cache_manifest.json").read_text(encoding="utf-8"))
        expected_spec = fixed_center90_w_transform_spec(self.input_samples)
        if (
            self.manifest.get("protocol_id") != CENTER90_PROTOCOL_ID
            or self.manifest.get("schema_version") != CENTER90_W_CACHE_SCHEMA
            or self.manifest.get("input_samples") != self.input_samples
            or self.manifest.get("complete") is not True
            or self.manifest.get("input_only") is not True
            or self.manifest.get("transform_spec") != expected_spec
            or self.manifest.get("transform_sha256") != _sha256_json(expected_spec)
            or self.manifest.get("target_read") is not False
            or self.manifest.get("test_read") is not False
            or self.manifest.get("test_cache_created") is not False
            or self.manifest.get("allowed_splits") != ["train", "val"]
        ):
            raise RuntimeError("center90 W cache manifest identity 漂移")
        self.row_ids = self._open(f"{self.split}_row_ids.npy")
        self.features = self._open(f"{self.split}_w.npy")
        self.frequencies_hz = self._open("w_frequencies_hz.npy")
        meta = self.manifest["splits"][self.split]
        row_hash = hashlib.sha256(np.asarray(self.row_ids).tobytes(order="C")).hexdigest()
        if (
            self.row_ids.shape != (int(meta["count"]),)
            or self.row_ids.dtype != np.int64
            or row_hash != str(meta["row_ids_sha256"])
            or np.unique(self.row_ids).size != self.row_ids.size
            or not np.all(np.diff(self.row_ids) > 0)
        ):
            raise RuntimeError("center90 W row identity 错误")
        if self.features.shape != tuple(meta["shape"]) or self.features.dtype != np.float32:
            raise RuntimeError("center90 W feature shape/dtype 错误")
        _validate_frequency_identity(self.frequencies_hz)
        duplicate_count = int(np.sum(np.diff(np.asarray(self.frequencies_hz)) == 0.0))
        nominal_error = float(
            np.max(
                np.abs(
                    np.asarray(self.frequencies_hz, dtype=np.float64)
                    - reduced_target_frequencies()
                )
            )
        )
        if (
            duplicate_count != int(self.manifest.get("duplicate_center_count", -1))
            or not np.isclose(
                nominal_error,
                float(self.manifest.get("max_nominal_frequency_error_hz", np.nan)),
                rtol=0.0,
                atol=1e-15,
            )
        ):
            raise RuntimeError("center90 W frequency audit metadata 漂移")

    def _open(self, filename: str) -> np.ndarray:
        path = self.root / filename
        record = self.manifest["files"].get(filename)
        if (
            record is None
            or not path.is_file()
            or path.stat().st_size != int(record["size_bytes"])
            or sha256_file(path) != record["sha256"]
        ):
            raise RuntimeError(f"center90 W cache 文件 identity 错误: {filename}")
        return np.load(path, mmap_mode="r", allow_pickle=False)

    def get(self, dataset_row_id: int) -> torch.Tensor:
        position = int(np.searchsorted(self.row_ids, int(dataset_row_id)))
        if position >= len(self.row_ids) or int(self.row_ids[position]) != int(dataset_row_id):
            raise KeyError(f"center90 W cache 缺少 dataset_row_id={dataset_row_id}")
        value = np.array(self.features[position], copy=True)
        if not np.isfinite(value).all():
            raise FloatingPointError("center90 W cache feature 包含 NaN/Inf")
        return torch.from_numpy(value)

    def verify_rows(self, dataset_row_ids: Iterable[int]) -> None:
        requested = np.asarray([int(value) for value in dataset_row_ids], dtype=np.int64)
        positions = np.searchsorted(self.row_ids, requested)
        valid = positions < len(self.row_ids)
        matched = np.zeros(requested.shape, dtype=np.bool_)
        matched[valid] = np.asarray(self.row_ids)[positions[valid]] == requested[valid]
        if requested.size == 0 or not bool(matched.all()):
            raise KeyError(f"center90 W cache rows 不完整: {requested[~matched][:10].tolist()}")


def open_center90_w_cache(
    root: str | Path,
    *,
    split: str,
    input_samples: int,
    expected_manifest_sha256: str,
) -> CenterContextWCacheReader | Center90WCacheReader:
    resolved = Path(root).resolve()
    if sha256_file(resolved / "cache_manifest.json") != str(expected_manifest_sha256):
        raise RuntimeError("center90 W cache manifest SHA-256 漂移")
    if int(input_samples) in {9000, 18000}:
        return CenterContextWCacheReader(
            resolved,
            split=split,
            input_samples=int(input_samples),
            require_complete=True,
        )
    return Center90WCacheReader(resolved, split=split, input_samples=int(input_samples))


def _validate_frequency_identity(frequencies: np.ndarray) -> None:
    values = np.asarray(frequencies, dtype=np.float64)
    if (
        values.shape != (REDUCED_SCALE_COUNT,)
        or not np.isfinite(values).all()
        or not np.all(np.diff(values) >= 0.0)
    ):
        raise RuntimeError("center90 135 s mapped centers 必须为 49 个有限非降值")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


__all__ = [
    "CENTER90_NEW_CACHE_SAMPLES",
    "CENTER90_W_CACHE_SCHEMA",
    "Center90WCacheReader",
    "build_center90_w_cache_from_rows",
    "center90_cwt_features",
    "fixed_center90_w_transform_spec",
    "open_center90_w_cache",
    "sha256_file",
]
