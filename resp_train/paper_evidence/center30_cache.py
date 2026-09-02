from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from resp_train.crd.tf_v1_features import MORLET_MU, SAMPLE_RATE
from resp_train.paper_evidence.center30_config import CENTER30_INPUT_SAMPLES, CENTER30_PROTOCOL_ID
from resp_train.paper_evidence.center30_data import center30_input_bounds
from resp_train.paper_evidence.center_context_cache import (
    CENTER_W_CACHE_SCHEMA as LEGACY_CACHE_SCHEMA,
    POOL_SAMPLES,
    REDUCED_SCALE_COUNT,
    REDUCED_SCALE_INDICES,
    reduced_scales_and_frequencies,
    reduced_target_frequencies,
)
from resp_train.paper_evidence.center_context_config import CENTER_CONTEXT_PROTOCOL_ID


CENTER30_W_CACHE_SCHEMA = "paper-center30-context-w-cache-v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def center30_cwt_features(waveform: np.ndarray | torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(torch.as_tensor(waveform, dtype=torch.float32).cpu()).reshape(-1)
    length = int(values.size)
    if length not in CENTER30_INPUT_SAMPLES:
        raise ValueError(f"center30 W 输入长度只允许 {list(CENTER30_INPUT_SAMPLES)}")
    if not np.isfinite(values).all():
        raise FloatingPointError("center30 W 输入包含 NaN/Inf")
    _require_ssqueezepy_066()
    from ssqueezepy import Wavelet, cwt
    from ssqueezepy.experimental import scale_to_freq

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
    context_length = length // POOL_SAMPLES
    magnitude = np.log1p(np.abs(np.asarray(coefficients)[order])).astype(np.float32)
    pooled = magnitude.reshape(REDUCED_SCALE_COUNT, context_length, POOL_SAMPLES).mean(
        axis=-1, dtype=np.float32
    )
    frequencies = np.ascontiguousarray(actual[order])
    _validate_frequency_identity(frequencies)
    if pooled.shape != (REDUCED_SCALE_COUNT, context_length) or not np.isfinite(pooled).all():
        raise FloatingPointError("center30 W feature shape/finite 错误")
    return np.ascontiguousarray(pooled), frequencies


def fixed_center30_w_transform_spec(input_samples: int) -> dict[str, Any]:
    length = int(input_samples)
    return {
        "sample_rate_hz": 100.0,
        "input_samples": length,
        "input_slice": list(center30_input_bounds(length)),
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


def build_center30_w_cache_from_rows(
    *,
    output_dir: str | Path,
    index_path: str | Path,
    rows_by_split: dict[str, pd.DataFrame],
    input_samples: int,
    dataset_index_sha256: str,
    feature_extractor: Any = center30_cwt_features,
    complete: bool = True,
) -> Path:
    from resp_train.data.cache import WholeNightCache

    length = int(input_samples)
    input_start, input_stop = center30_input_bounds(length)
    if set(rows_by_split) != {"train", "val"}:
        raise ValueError("center30 W cache builder 只允许 train/val")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    lifecycle = output / "lifecycle.json"
    _write_lifecycle(lifecycle, status="running")
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
                raise ValueError(f"center30 W {split} rows 缺失或 identity 错误: {missing}")
            ids = rows["dataset_row_id"].to_numpy(dtype=np.int64)
            if np.unique(ids).size != ids.size or not np.all(np.diff(ids) > 0):
                raise ValueError(f"center30 W {split} row_ids 必须严格递增且无重复")
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
                parent_start, parent_stop = int(row.window_start_sample), int(row.window_end_sample)
                if parent_stop - parent_start != 18000:
                    raise RuntimeError(f"row={row.dataset_row_id} 不是冻结 180 s parent")
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
                    raise RuntimeError("center30 length-specific mapped centers 在 sample 间漂移")
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
        train_ids = rows_by_split["train"]["dataset_row_id"].to_numpy(dtype=np.int64)
        val_ids = rows_by_split["val"]["dataset_row_id"].to_numpy(dtype=np.int64)
        if np.intersect1d(train_ids, val_ids).size or frequency_identity is None:
            raise RuntimeError("center30 W cache split 或 frequency identity 未闭合")
        np.save(output / "w_frequencies_hz.npy", frequency_identity, allow_pickle=False)
        spec = fixed_center30_w_transform_spec(length)
        manifest = {
            "protocol_id": CENTER30_PROTOCOL_ID,
            "schema_version": CENTER30_W_CACHE_SCHEMA,
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
        (output / "cache_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except BaseException as exc:
        _write_lifecycle(lifecycle, status="failed", error_type=type(exc).__name__, error_message=str(exc))
        raise
    _write_lifecycle(lifecycle, status="complete")
    return output / "cache_manifest.json"


class Center30WCacheReader:
    def __init__(self, root: str | Path, *, split: str, input_samples: int) -> None:
        self.root = Path(root).resolve()
        self.split = str(split)
        self.input_samples = int(input_samples)
        if self.split not in {"train", "val"}:
            raise ValueError("center30 W cache reader 只允许 train/val")
        self.manifest = json.loads((self.root / "cache_manifest.json").read_text(encoding="utf-8"))
        protocol = self.manifest.get("protocol_id")
        schema = self.manifest.get("schema_version")
        legacy_allowed = self.input_samples in {6000, 9000}
        identity_ok = (protocol, schema) == (CENTER30_PROTOCOL_ID, CENTER30_W_CACHE_SCHEMA) or (
            legacy_allowed and (protocol, schema) == (CENTER_CONTEXT_PROTOCOL_ID, LEGACY_CACHE_SCHEMA)
        )
        expected_spec = fixed_center30_w_transform_spec(self.input_samples)
        if (
            not identity_ok
            or self.manifest.get("input_samples") != self.input_samples
            or self.manifest.get("transform_spec") != expected_spec
            or self.manifest.get("transform_sha256") != _sha256_json(expected_spec)
            or self.manifest.get("complete") is not True
            or self.manifest.get("target_read") is not False
            or self.manifest.get("test_read") is not False
            or self.manifest.get("test_cache_created") is not False
        ):
            raise RuntimeError("center30 W cache manifest identity/访问边界不合格")
        self.row_ids = self._open(f"{self.split}_row_ids.npy")
        self.features = self._open(f"{self.split}_w.npy")
        self.frequencies_hz = self._open("w_frequencies_hz.npy")
        split_meta = self.manifest["splits"][self.split]
        if (
            self.row_ids.shape != (int(split_meta["count"]),)
            or self.row_ids.dtype != np.int64
            or self.features.shape != tuple(split_meta["shape"])
            or self.features.dtype != np.float32
            or hashlib.sha256(np.asarray(self.row_ids).tobytes(order="C")).hexdigest()
            != str(split_meta["row_ids_sha256"])
            or np.unique(self.row_ids).size != self.row_ids.size
            or not np.all(np.diff(self.row_ids) > 0)
        ):
            raise RuntimeError("center30 W cache array identity 错误")
        _validate_frequency_identity(self.frequencies_hz)
        duplicate_count = int(np.sum(np.diff(np.asarray(self.frequencies_hz)) == 0.0))
        nominal_error = float(
            np.max(np.abs(np.asarray(self.frequencies_hz, dtype=np.float64) - reduced_target_frequencies()))
        )
        if duplicate_count != int(self.manifest.get("duplicate_center_count", -1)) or not np.isclose(
            nominal_error,
            float(self.manifest.get("max_nominal_frequency_error_hz", np.nan)),
            rtol=0.0,
            atol=1e-15,
        ):
            raise RuntimeError("center30 W cache frequency audit metadata 漂移")

    def _open(self, filename: str) -> np.ndarray:
        path = self.root / filename
        record = self.manifest["files"].get(filename)
        if record is None or path.stat().st_size != int(record["size_bytes"]) or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"center30 W cache 文件 identity 错误: {filename}")
        return np.load(path, mmap_mode="r", allow_pickle=False)

    def get(self, dataset_row_id: int) -> torch.Tensor:
        position = int(np.searchsorted(self.row_ids, int(dataset_row_id)))
        if position >= len(self.row_ids) or int(self.row_ids[position]) != int(dataset_row_id):
            raise KeyError(f"center30 W cache 缺少 dataset_row_id={dataset_row_id}")
        value = np.array(self.features[position], copy=True)
        if not np.isfinite(value).all():
            raise FloatingPointError("center30 W cache 读取到 NaN/Inf")
        return torch.from_numpy(value)

    def verify_rows(self, dataset_row_ids: Iterable[int]) -> None:
        requested = np.asarray([int(value) for value in dataset_row_ids], dtype=np.int64)
        positions = np.searchsorted(self.row_ids, requested)
        valid = positions < len(self.row_ids)
        matched = np.zeros(requested.shape, dtype=np.bool_)
        matched[valid] = np.asarray(self.row_ids)[positions[valid]] == requested[valid]
        if requested.size == 0 or not bool(matched.all()):
            raise KeyError("center30 W cache rows 不完整")


def _validate_frequency_identity(frequencies: np.ndarray) -> None:
    values = np.asarray(frequencies, dtype=np.float64)
    if values.shape != (REDUCED_SCALE_COUNT,) or not np.isfinite(values).all() or not np.all(np.diff(values) >= 0.0):
        raise RuntimeError("center30 mapped centers 必须是 49 个有限非降值")


def _require_ssqueezepy_066() -> None:
    import importlib.metadata

    if importlib.metadata.version("ssqueezepy") != "0.6.6":
        raise RuntimeError("center30 W cache 要求 ssqueezepy==0.6.6")


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_lifecycle(path: Path, *, status: str, **extra: Any) -> None:
    path.write_text(json.dumps({"status": status, **extra}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


__all__ = [
    "CENTER30_W_CACHE_SCHEMA",
    "Center30WCacheReader",
    "build_center30_w_cache_from_rows",
    "center30_cwt_features",
    "fixed_center30_w_transform_spec",
    "sha256_file",
]
