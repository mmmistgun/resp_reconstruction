"""从冻结的 97-scale W cache 只读选择 H 频带，供标准 dataset 使用。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from resp_train.crd import tf_v1_data as frozen
from resp_train.crd.tf_v1_features import fixed_transform_spec


def frozen_h_metadata(cache_root):
    """模型与 reader 共用缓存中的实际频率，严格校验冻结身份。"""
    if not cache_root:
        raise ValueError("必须配置 data.tf_cache_path 指向冻结 W cache")
    root = Path(cache_root)
    path = root / "cache_manifest.json"
    if frozen._sha256_file(path) != frozen.FROZEN_CACHE_MANIFEST_SHA256:
        raise RuntimeError("冻结 W cache manifest SHA-256 不一致")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("fixed_transform_spec") != fixed_transform_spec()
            or manifest.get("transform_sha256") != frozen.FROZEN_CACHE_TRANSFORM_SHA256
            or manifest.get("complete") is not True
            or manifest.get("research_test_used") is not False
            or manifest.get("test_cache_created") is not False
            or manifest.get("target_read") is not False):
        raise RuntimeError("冻结 W cache 表示或访问边界不符")
    filename = "w_frequencies_hz.npy"
    meta = manifest["files"][filename]
    frequency_path = root / filename
    if (frequency_path.stat().st_size != meta["size_bytes"]
            or frozen._sha256_file(frequency_path) != meta["sha256"]):
        raise RuntimeError("W cache 频率元数据身份不符")
    frequencies = np.load(frequency_path, allow_pickle=False)
    if (frequencies.shape != (97,) or frequencies.dtype != np.float64
            or not np.isfinite(frequencies).all() or np.any(frequencies <= 0)
            or np.any(np.diff(frequencies) < 0)):
        raise ValueError("W cache 频率 shape/dtype/order/finite 不符")
    indices = np.flatnonzero((frequencies > .8) & (frequencies <= 8))
    if len(indices) != 41:
        raise ValueError("冻结 H 频带必须包含 41 个尺度")
    return frequencies[indices].copy(), indices


class FrozenHCWTReader:
    """复用原 reader 的行身份验证；样本进入模型前完成 97→41 选择。"""

    def __init__(self, cache_root, *, split):
        if split not in {"train", "val"}:
            raise ValueError("Patch-aligned TF cache 仅开放 train/val")
        self.frequencies_hz, self.indices = frozen_h_metadata(cache_root)
        self.source = frozen.TfV1CacheReader(cache_root, split=split, representations=("w",))
        values = self.source.arrays["w"]
        if values.shape[1:] != (97, 360) or values.dtype != np.float32:
            raise ValueError("冻结 W cache 必须为 float32 (N,97,360)")

    def verify_rows(self, rows):
        self.source.verify_rows(rows)

    def get(self, dataset_row_id):
        position = self.source.position(dataset_row_id)
        value = self.source.arrays["w"][position]
        # 对整条源记录检查，避免频带选择掩盖已损坏的 cache 样本。
        if not np.isfinite(value).all():
            raise FloatingPointError("冻结 W cache 样本包含非有限值")
        return {"w": torch.from_numpy(np.array(value[self.indices], copy=True))}
