from __future__ import annotations

import hashlib
import json
import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import numpy as np
import pandas as pd
import torch

from resp_train.crd.tf_v1_features import (
    MORLET_MU,
    SAMPLE_RATE,
    W_SCALE_COUNT,
    W_VOICES_PER_OCTAVE,
    morlet_target_frequencies,
)
from resp_train.paper_evidence.center_context_config import CENTER_CONTEXT_INPUT_SAMPLES, CENTER_CONTEXT_PROTOCOL_ID
from resp_train.paper_evidence.center_context_data import center_input_bounds


CENTER_W_CACHE_SCHEMA = "paper-center-context-w-cache-v1"
POOL_SAMPLES = 50
REDUCED_SCALE_INDICES = tuple(range(0, W_SCALE_COUNT, 2))
REDUCED_SCALE_COUNT = len(REDUCED_SCALE_INDICES)


def reduced_target_frequencies() -> np.ndarray:
    """复用完整 12-voice 网格的偶数索引，形成全频程 49-scale/约 6 voices-per-octave 网格。"""

    full = morlet_target_frequencies(voices_per_octave=W_VOICES_PER_OCTAVE, count=W_SCALE_COUNT)
    reduced = np.ascontiguousarray(full[np.asarray(REDUCED_SCALE_INDICES, dtype=np.int64)])
    reduced.setflags(write=False)
    return reduced


@lru_cache(maxsize=3)
def reduced_scales_and_frequencies(length: int) -> tuple[np.ndarray, np.ndarray]:
    _require_ssqueezepy_066()
    from ssqueezepy import Wavelet
    from ssqueezepy.experimental import freq_to_scale, scale_to_freq

    target = reduced_target_frequencies()
    wavelet = Wavelet(("morlet", {"mu": MORLET_MU}), N=int(length), dtype="float32")
    scales = freq_to_scale(
        target,
        wavelet,
        int(length),
        fs=SAMPLE_RATE,
        n_search_scales=max(4096, 20 * REDUCED_SCALE_COUNT),
        kind="peak",
        base=2,
    )[::-1]
    actual = np.asarray(
        scale_to_freq(scales, wavelet, int(length), fs=SAMPLE_RATE, padtype="reflect"),
        dtype=np.float64,
    )
    if scales.shape != (REDUCED_SCALE_COUNT,) or not np.all(np.diff(scales) > 0.0):
        raise RuntimeError("W-reduced scales 必须是 49 个有限严格递增值")
    scales = np.asarray(scales, dtype=np.float64)
    scales.setflags(write=False)
    actual.setflags(write=False)
    return scales, actual


def center_context_cwt_features(waveform: np.ndarray | torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """仅接收已裁剪的单一输入视图，因而无法读取父窗口范围外样本。"""

    values = np.asarray(torch.as_tensor(waveform, dtype=torch.float32).cpu()).reshape(-1)
    length = int(values.size)
    if length not in CENTER_CONTEXT_INPUT_SAMPLES:
        raise ValueError(f"中心 W 输入长度只允许 {list(CENTER_CONTEXT_INPUT_SAMPLES)}")
    if not np.isfinite(values).all():
        raise FloatingPointError("中心 W 输入包含 NaN/Inf")
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
    pooled = magnitude.reshape(REDUCED_SCALE_COUNT, context_length, POOL_SAMPLES).mean(axis=-1, dtype=np.float32)
    frequencies = np.ascontiguousarray(actual[order])
    if pooled.shape != (REDUCED_SCALE_COUNT, context_length):
        raise RuntimeError(f"中心 W feature shape 错误: {pooled.shape}")
    _validate_frequency_identity(frequencies, length=length)
    if not np.isfinite(pooled).all():
        raise FloatingPointError("中心 W feature 包含 NaN/Inf")
    return np.ascontiguousarray(pooled), frequencies


def write_center_context_w_cache(
    *,
    output_dir: str | Path,
    input_samples: int,
    split_features: dict[str, tuple[np.ndarray, np.ndarray]],
    frequencies_hz: np.ndarray,
    dataset_index_sha256: str,
    complete: bool = True,
) -> Path:
    """原子写入一个长度的 train/validation input-only cache；现有目录一律拒绝。"""

    length = int(input_samples)
    center_input_bounds(length)
    if set(split_features) != {"train", "val"}:
        raise ValueError("中心 W cache 必须且只能同时包含 train/val")
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    _validate_frequency_identity(frequencies, length=length)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"中心 W cache 禁止覆盖: {output}")
    temporary = output.parent / f".{output.name}.{uuid4().hex}.tmp"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        split_manifest: dict[str, Any] = {}
        for split in ("train", "val"):
            row_ids, features = split_features[split]
            ids = np.asarray(row_ids, dtype=np.int64)
            values = np.asarray(features, dtype=np.float32)
            expected_shape = (ids.size, REDUCED_SCALE_COUNT, length // POOL_SAMPLES)
            if values.shape != expected_shape:
                raise ValueError(f"{split} W cache shape {values.shape} != {expected_shape}")
            if ids.ndim != 1 or ids.size == 0 or np.unique(ids).size != ids.size or not np.all(np.diff(ids) > 0):
                raise ValueError(f"{split} row_ids 必须严格递增、非空且无重复")
            if not np.isfinite(values).all():
                raise FloatingPointError(f"{split} W cache 包含 NaN/Inf")
            np.save(temporary / f"{split}_row_ids.npy", ids, allow_pickle=False)
            np.save(temporary / f"{split}_w.npy", values, allow_pickle=False)
            split_manifest[split] = {
                "count": int(ids.size),
                "row_ids_sha256": hashlib.sha256(ids.tobytes(order="C")).hexdigest(),
                "shape": list(values.shape),
                "dtype": str(values.dtype),
                "finite": True,
            }
        train_ids = split_features["train"][0]
        val_ids = split_features["val"][0]
        if np.intersect1d(np.asarray(train_ids, dtype=np.int64), np.asarray(val_ids, dtype=np.int64)).size:
            raise RuntimeError("中心 W cache train/validation row_ids 重叠")
        np.save(temporary / "w_frequencies_hz.npy", frequencies, allow_pickle=False)
        specification = fixed_center_w_transform_spec(length)
        manifest = {
            "protocol_id": CENTER_CONTEXT_PROTOCOL_ID,
            "schema_version": CENTER_W_CACHE_SCHEMA,
            "input_samples": length,
            "input_slice": list(center_input_bounds(length)),
            "input_only": True,
            "target_read": False,
            "test_read": False,
            "test_cache_created": False,
            "allowed_splits": ["train", "val"],
            "dataset_index_sha256": str(dataset_index_sha256),
            "complete": bool(complete),
            "transform_spec": specification,
            "transform_sha256": _sha256_json(specification),
            "frequency_count": int(frequencies.size),
            "frequency_monotonic_non_decreasing": True,
            "duplicate_center_count": int(np.sum(np.diff(frequencies) == 0.0)),
            "max_nominal_frequency_error_hz": float(
                np.max(np.abs(frequencies - reduced_target_frequencies()))
            ),
            "splits": split_manifest,
        }
        manifest["files"] = {
            path.name: {"size_bytes": int(path.stat().st_size), "sha256": _sha256_file(path)}
            for path in sorted(temporary.iterdir())
            if path.is_file()
        }
        (temporary / "cache_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output / "cache_manifest.json"


def build_center_context_w_cache_from_rows(
    *,
    output_dir: str | Path,
    index_path: str | Path,
    rows_by_split: dict[str, pd.DataFrame],
    input_samples: int,
    dataset_index_sha256: str,
    feature_extractor: Any = center_context_cwt_features,
    complete: bool = True,
) -> Path:
    """从冻结 parent rows 流式构建单一长度 cache；失败目录与 lifecycle 原样保留。"""

    from resp_train.data.cache import WholeNightCache

    length = int(input_samples)
    input_start, input_stop = center_input_bounds(length)
    if set(rows_by_split) != {"train", "val"}:
        raise ValueError("中心 W cache builder 只允许 train/val")
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
            if missing:
                raise ValueError(f"中心 W {split} rows 缺列: {missing}")
            if rows.empty or set(rows["split"].astype(str)) != {split}:
                raise ValueError(f"中心 W {split} rows 为空或 split 错误")
            ids = rows["dataset_row_id"].to_numpy(dtype=np.int64)
            if np.unique(ids).size != ids.size or not np.all(np.diff(ids) > 0):
                raise ValueError(f"中心 W {split} row_ids 必须严格递增且无重复")
            id_array = np.lib.format.open_memmap(
                output / f"{split}_row_ids.npy", mode="w+", dtype=np.int64, shape=(len(rows),)
            )
            feature_array = np.lib.format.open_memmap(
                output / f"{split}_w.npy",
                mode="w+",
                dtype=np.float32,
                shape=(len(rows), REDUCED_SCALE_COUNT, length // 50),
            )
            source_keys: set[str] = set()
            for position, row in enumerate(rows.itertuples(index=False)):
                parent_start = int(row.window_start_sample)
                parent_stop = int(row.window_end_sample)
                if parent_stop - parent_start != 18000:
                    raise RuntimeError(f"row={row.dataset_row_id} 不是冻结 180 s parent")
                source_key = str(row.bcg_signal_key)
                source_keys.add(source_key)
                whole = source_cache.get_arrays(str(row.source_npz), [source_key])[source_key]
                parent = np.asarray(whole[parent_start:parent_stop], dtype=np.float32)
                if parent.shape != (18000,) or not np.isfinite(parent).all():
                    raise FloatingPointError(f"row={row.dataset_row_id} parent input shape/finite 错误")
                # 先裁 input view，再调用 CWT；feature extractor 不接收 parent 或 target。
                feature, frequencies = feature_extractor(parent[input_start:input_stop])
                feature = np.asarray(feature, dtype=np.float32)
                frequencies = np.asarray(frequencies, dtype=np.float64)
                if feature.shape != (REDUCED_SCALE_COUNT, length // 50) or not np.isfinite(feature).all():
                    raise FloatingPointError(f"row={row.dataset_row_id} W feature shape/finite 错误")
                _validate_frequency_identity(frequencies, length=length)
                if frequency_identity is None:
                    frequency_identity = frequencies.copy()
                elif not np.array_equal(frequency_identity, frequencies):
                    raise RuntimeError("length-specific mapped centers 在 sample 间漂移")
                id_array[position] = int(row.dataset_row_id)
                feature_array[position] = feature
            id_array.flush()
            feature_array.flush()
            split_manifest[split] = {
                "count": int(len(rows)),
                "row_ids_sha256": hashlib.sha256(ids.tobytes(order="C")).hexdigest(),
                "shape": [len(rows), REDUCED_SCALE_COUNT, length // 50],
                "dtype": "float32",
                "finite": True,
                "source_keys": sorted(source_keys),
            }
        if np.intersect1d(
            rows_by_split["train"]["dataset_row_id"].to_numpy(dtype=np.int64),
            rows_by_split["val"]["dataset_row_id"].to_numpy(dtype=np.int64),
        ).size:
            raise RuntimeError("中心 W cache train/validation row_ids 重叠")
        if frequency_identity is None:
            raise RuntimeError("中心 W cache 没有生成 frequency identity")
        np.save(output / "w_frequencies_hz.npy", frequency_identity, allow_pickle=False)
        spec = fixed_center_w_transform_spec(length)
        manifest = {
            "protocol_id": CENTER_CONTEXT_PROTOCOL_ID,
            "schema_version": CENTER_W_CACHE_SCHEMA,
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
            path.name: {"size_bytes": int(path.stat().st_size), "sha256": _sha256_file(path)}
            for path in sorted(output.iterdir())
            if path.is_file() and path.name not in {"cache_manifest.json", "lifecycle.json"}
        }
        (output / "cache_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except BaseException as exc:
        _write_lifecycle(
            lifecycle,
            status="failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        raise
    _write_lifecycle(lifecycle, status="complete")
    return output / "cache_manifest.json"


class CenterContextWCacheReader:
    def __init__(
        self,
        root: str | Path,
        *,
        split: str,
        input_samples: int,
        require_complete: bool = True,
    ) -> None:
        self.root = Path(root).resolve()
        self.split = str(split)
        self.input_samples = int(input_samples)
        if self.split not in {"train", "val"}:
            raise ValueError("中心 W cache reader 只允许 train/val")
        manifest_path = self.root / "cache_manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_spec = fixed_center_w_transform_spec(self.input_samples)
        if (
            self.manifest.get("protocol_id") != CENTER_CONTEXT_PROTOCOL_ID
            or self.manifest.get("schema_version") != CENTER_W_CACHE_SCHEMA
            or self.manifest.get("input_samples") != self.input_samples
            or self.manifest.get("transform_spec") != expected_spec
            or self.manifest.get("transform_sha256") != _sha256_json(expected_spec)
            or self.manifest.get("target_read") is not False
            or self.manifest.get("test_read") is not False
            or self.manifest.get("test_cache_created") is not False
            or (bool(require_complete) and self.manifest.get("complete") is not True)
        ):
            raise RuntimeError("中心 W cache manifest identity/访问边界不合格")
        self.row_ids = self._open(f"{self.split}_row_ids.npy")
        self.features = self._open(f"{self.split}_w.npy")
        self.frequencies_hz = self._open("w_frequencies_hz.npy")
        split_meta = self.manifest["splits"][self.split]
        if self.row_ids.shape != (int(split_meta["count"]),) or self.row_ids.dtype != np.int64:
            raise RuntimeError("中心 W row_ids shape/dtype 错误")
        row_hash = hashlib.sha256(np.asarray(self.row_ids).tobytes(order="C")).hexdigest()
        if (
            row_hash != str(split_meta["row_ids_sha256"])
            or np.unique(self.row_ids).size != self.row_ids.size
            or not np.all(np.diff(self.row_ids) > 0)
        ):
            raise RuntimeError("中心 W row_ids content/hash 错误")
        if self.features.shape != tuple(split_meta["shape"]) or self.features.dtype != np.float32:
            raise RuntimeError("中心 W features shape/dtype 错误")
        _validate_frequency_identity(self.frequencies_hz, length=self.input_samples)
        duplicate_count = int(np.sum(np.diff(np.asarray(self.frequencies_hz)) == 0.0))
        nominal_error = float(
            np.max(np.abs(np.asarray(self.frequencies_hz, dtype=np.float64) - reduced_target_frequencies()))
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
            raise RuntimeError("中心 W frequency audit metadata 与文件不一致")

    def _open(self, filename: str) -> np.ndarray:
        path = self.root / filename
        record = self.manifest["files"].get(filename)
        if record is None or path.stat().st_size != int(record["size_bytes"]) or _sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"中心 W cache 文件 identity 错误: {filename}")
        return np.load(path, mmap_mode="r", allow_pickle=False)

    def get(self, dataset_row_id: int) -> torch.Tensor:
        position = int(np.searchsorted(self.row_ids, int(dataset_row_id)))
        if position >= len(self.row_ids) or int(self.row_ids[position]) != int(dataset_row_id):
            raise KeyError(f"中心 W cache 缺少 dataset_row_id={dataset_row_id}")
        value = np.array(self.features[position], copy=True)
        if not np.isfinite(value).all():
            raise FloatingPointError("中心 W cache 读取到 NaN/Inf")
        return torch.from_numpy(value)

    def verify_rows(self, dataset_row_ids: Iterable[int]) -> None:
        requested = np.asarray([int(value) for value in dataset_row_ids], dtype=np.int64)
        positions = np.searchsorted(self.row_ids, requested)
        valid = positions < len(self.row_ids)
        matched = np.zeros(requested.shape, dtype=np.bool_)
        matched[valid] = np.asarray(self.row_ids)[positions[valid]] == requested[valid]
        if requested.size == 0 or not bool(matched.all()):
            raise KeyError(f"中心 W cache rows 不完整: {requested[~matched][:10].tolist()}")


def fixed_center_w_transform_spec(input_samples: int) -> dict[str, Any]:
    length = int(input_samples)
    center_input_bounds(length)
    return {
        "sample_rate_hz": 100.0,
        "input_samples": length,
        "input_slice": list(center_input_bounds(length)),
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
        "shape": [REDUCED_SCALE_COUNT, length // 50],
        "source": "length_specific_cropped_input_only",
    }


def _validate_frequency_identity(frequencies: np.ndarray, *, length: int) -> None:
    values = np.asarray(frequencies, dtype=np.float64)
    if (
        values.shape != (REDUCED_SCALE_COUNT,)
        or not np.isfinite(values).all()
        or not np.all(np.diff(values) >= 0.0)
    ):
        raise RuntimeError(
            f"length={length} 的实际 mapped centers 必须是 {REDUCED_SCALE_COUNT} 个有限非降值"
        )


def _require_ssqueezepy_066() -> None:
    import importlib.metadata

    actual = importlib.metadata.version("ssqueezepy")
    if actual != "0.6.6":
        raise RuntimeError(f"中心 W cache 要求 ssqueezepy==0.6.6，当前为 {actual}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_lifecycle(path: Path, *, status: str, **extra: Any) -> None:
    path.write_text(
        json.dumps({"status": status, **extra}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "CENTER_W_CACHE_SCHEMA",
    "REDUCED_SCALE_COUNT",
    "REDUCED_SCALE_INDICES",
    "CenterContextWCacheReader",
    "build_center_context_w_cache_from_rows",
    "center_context_cwt_features",
    "fixed_center_w_transform_spec",
    "reduced_scales_and_frequencies",
    "reduced_target_frequencies",
    "write_center_context_w_cache",
]
