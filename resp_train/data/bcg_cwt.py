"""完整 97 行 CWT 的只读数据接入与频率身份检查。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from resp_train.crd import tf_v1_data as frozen
from resp_train.crd.tf_v1_features import PROTOCOL, fixed_transform_spec


def frozen_w_metadata(cache_root):
    root = Path(cache_root)
    path = root / "cache_manifest.json"
    if frozen._sha256_file(path) != frozen.FROZEN_CACHE_MANIFEST_SHA256:
        raise RuntimeError("W cache manifest SHA-256 不一致")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("protocol") != PROTOCOL
            or manifest.get("fixed_transform_spec") != fixed_transform_spec()
            or manifest.get("transform_sha256") != frozen.FROZEN_CACHE_TRANSFORM_SHA256
            or manifest.get("complete") is not True
            or any(manifest.get(k) is not False for k in ("research_test_used", "test_cache_created", "target_read"))):
        raise RuntimeError("W cache 表示或访问边界不符")
    path = root / "w_frequencies_hz.npy"
    metadata = manifest["files"][path.name]
    if path.stat().st_size != metadata["size_bytes"] or frozen._sha256_file(path) != metadata["sha256"]:
        raise RuntimeError("W cache 频率文件身份不符")
    frequencies = np.load(path, allow_pickle=False)
    if (frequencies.shape != (97,) or frequencies.dtype != np.float64
            or not np.isfinite(frequencies).all() or np.any(frequencies <= 0) or np.any(np.diff(frequencies) < 0)):
        raise ValueError("W cache 频率 shape/dtype/order/finite 不符")
    return frequencies.copy()


class FrozenWCWTReader:
    def __init__(self, cache_root, *, split):
        if split not in {"train", "val"}:
            raise ValueError("BCG-CWT 数据入口仅开放 train/val")
        self.frequencies_hz = frozen_w_metadata(cache_root)
        self.source = frozen.TfV1CacheReader(cache_root, split=split, representations=("w",))
        values = self.source.arrays["w"]
        if values.shape[1:] != (97, 360) or values.dtype != np.float32:
            raise ValueError("W cache 必须为 float32 (N,97,360)")

    def verify_rows(self, rows):
        self.source.verify_rows(rows)

    def get(self, dataset_row_id):
        result = self.source.get(dataset_row_id)
        if not np.isfinite(result["w"].numpy()).all():
            raise FloatingPointError(f"W cache row={dataset_row_id} 包含 NaN/Inf")
        return result
