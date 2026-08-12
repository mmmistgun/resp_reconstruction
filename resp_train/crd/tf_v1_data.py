from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from resp_train.crd.tf_v1_features import PROTOCOL


FROZEN_CACHE_TRANSFORM_SHA256 = "bd6cea7348f6b51ed768b89cf9b3425530b6358a82ba78277844517a1c27fea0"
FROZEN_CACHE_MANIFEST_SHA256 = "6fb44aad2689d9426ad78dc1f054db5aaac698792af5818bc01a54563cb9f0b8"
REPRESENTATIONS = frozenset({"m", "w", "l", "s"})


class TfV1CacheReader:
    """按 dataset_row_id 只读冻结的 CRD-TF v1 memory-map cache。"""

    def __init__(self, cache_root: str | Path, *, split: str, representations: Iterable[str]) -> None:
        self.root = Path(cache_root).resolve()
        self.split = str(split)
        self.representations = tuple(sorted({str(value).lower() for value in representations}))
        if self.split not in {"train", "val"}:
            raise ValueError("CRD-TF cache reader 只允许 train/val")
        unknown = set(self.representations) - REPRESENTATIONS
        if unknown:
            raise ValueError(f"未知 TF representations: {sorted(unknown)}")
        manifest_path = self.root / "cache_manifest.json"
        if _sha256_file(manifest_path) != FROZEN_CACHE_MANIFEST_SHA256:
            raise RuntimeError("CRD-TF cache manifest SHA-256 不一致")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            self.manifest.get("protocol") != PROTOCOL
            or self.manifest.get("transform_sha256") != FROZEN_CACHE_TRANSFORM_SHA256
            or self.manifest.get("complete") is not True
            or self.manifest.get("research_test_used") is not False
            or self.manifest.get("test_cache_created") is not False
            or self.manifest.get("target_read") is not False
        ):
            raise RuntimeError("CRD-TF cache manifest identity/边界不合格")
        split_meta = self.manifest["splits"].get(self.split)
        if split_meta is None or split_meta.get("split") != self.split:
            raise RuntimeError(f"CRD-TF cache 缺少 split={self.split}")

        self.row_ids = self._open(f"{self.split}_row_ids.npy")
        expected_count = int(split_meta["count"])
        if self.row_ids.shape != (expected_count,) or self.row_ids.dtype != np.int64:
            raise RuntimeError("CRD-TF row_ids shape/dtype 错误")
        row_hash = hashlib.sha256(np.asarray(self.row_ids).tobytes(order="C")).hexdigest()
        if row_hash != str(split_meta["row_ids_sha256"]):
            raise RuntimeError("CRD-TF row_ids content SHA-256 不一致")
        if np.unique(self.row_ids).size != expected_count or not np.all(np.diff(self.row_ids) > 0):
            raise RuntimeError("CRD-TF row_ids 必须严格递增且无重复")

        self.arrays: dict[str, np.ndarray] = {}
        if "m" in self.representations:
            self.arrays["m_slow"] = self._open(f"{self.split}_m_slow.npy")
            self.arrays["m_fast"] = self._open(f"{self.split}_m_fast.npy")
        if "w" in self.representations:
            self.arrays["w"] = self._open(f"{self.split}_w.npy")
        if "s" in self.representations:
            self.arrays["s"] = self._open(f"{self.split}_s.npy")
        if "l" in self.representations:
            self.arrays["l_spectrum"] = self._open(f"{self.split}_l_spectrum.npy")
        for name, array in self.arrays.items():
            if array.shape[0] != expected_count:
                raise RuntimeError(f"CRD-TF {name} 第一维与 row_ids 不一致")

    def _open(self, filename: str) -> np.ndarray:
        metadata = self.manifest["files"].get(filename)
        if metadata is None:
            raise RuntimeError(f"CRD-TF manifest 缺少 {filename}")
        path = self.root / filename
        if path.stat().st_size != int(metadata["size_bytes"]):
            raise RuntimeError(f"CRD-TF cache 文件大小不一致: {filename}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != metadata["shape"] or str(array.dtype) != metadata["dtype"]:
            raise RuntimeError(f"CRD-TF cache shape/dtype 不一致: {filename}")
        if metadata.get("finite") is not True:
            raise RuntimeError(f"CRD-TF cache manifest 未确认 finite: {filename}")
        return array

    def position(self, dataset_row_id: int) -> int:
        row_id = int(dataset_row_id)
        position = int(np.searchsorted(self.row_ids, row_id))
        if position >= len(self.row_ids) or int(self.row_ids[position]) != row_id:
            raise KeyError(f"CRD-TF cache 缺少 dataset_row_id={row_id}")
        return position

    def get(self, dataset_row_id: int) -> dict[str, torch.Tensor]:
        position = self.position(dataset_row_id)
        output: dict[str, torch.Tensor] = {}
        for name, array in self.arrays.items():
            # memmap 是只读的；显式 copy 避免 torch.from_numpy 接收不可写数组。
            value = np.array(array[position], copy=True)
            output[name] = torch.from_numpy(value)
        return output

    def verify_rows(self, dataset_row_ids: Iterable[int]) -> None:
        requested = np.asarray([int(value) for value in dataset_row_ids], dtype=np.int64)
        if requested.ndim != 1 or requested.size == 0:
            raise ValueError("CRD-TF dataset rows 不得为空")
        positions = np.searchsorted(self.row_ids, requested)
        valid = positions < len(self.row_ids)
        matched = np.zeros(requested.shape, dtype=np.bool_)
        matched[valid] = np.asarray(self.row_ids)[positions[valid]] == requested[valid]
        if not bool(matched.all()):
            missing = requested[~matched][:10].tolist()
            raise KeyError(f"CRD-TF cache 缺少 dataset rows: {missing}")


def batch_tf_to_device(
    batch: dict[str, Any] | Any,
    device: torch.device,
    *,
    non_blocking: bool,
) -> dict[str, torch.Tensor] | None:
    raw = batch.get("tf") if hasattr(batch, "get") else None
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TypeError("batch['tf'] 必须是 tensor mapping")
    output = {}
    for name, value in raw.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"batch['tf'][{name!r}] 必须是 tensor")
        output[str(name)] = value.to(device, non_blocking=non_blocking)
    return output


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
