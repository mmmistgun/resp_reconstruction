from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from resp_train.crd.config import CRD_TF_RESEARCH_TEST_PROTOCOL_VERSION


FROZEN_RESEARCH_TEST_CACHE_ROOT = Path(
    "/mnt/disk_code/marques/resp_reconstruction/runs/crd_tf_v1/research_test_cache/"
    "40a24df424b2ff9182cfcc6f5b7c12d287578b0df1ed1e25b56b7af1c7f73839"
)
FROZEN_RESEARCH_TEST_CACHE_MANIFEST_SHA256 = "5d43ecf34596d5a6dd7cbaba75d91f9b7cbbb00214ae7594a4755e2afe510745"
FROZEN_RESEARCH_TEST_CACHE_IDENTITY = "40a24df424b2ff9182cfcc6f5b7c12d287578b0df1ed1e25b56b7af1c7f73839"
FROZEN_RESEARCH_TEST_CACHE_COMMIT = "dfd931379c01609fbd154549117b16950f16ec3f"
EXPECTED_TEST_COUNT = 2310
REPRESENTATIONS = frozenset({"m", "w", "s"})


class TfV1ResearchTestCacheReader:
    """严格校验并按 dataset_row_id 读取冻结的 test input-only M/W/S cache。"""

    def __init__(self, cache_root: str | Path, *, split: str, representations: Iterable[str]) -> None:
        self.root = Path(cache_root).resolve()
        self.split = str(split)
        self.representations = tuple(sorted({str(value).lower() for value in representations}))
        if self.root != FROZEN_RESEARCH_TEST_CACHE_ROOT.resolve():
            raise RuntimeError(f"research-test cache 路径未冻结: {self.root}")
        if self.split != "test":
            raise ValueError("research-test cache reader 只允许 test")
        unknown = set(self.representations) - REPRESENTATIONS
        if unknown:
            raise ValueError(f"research-test cache 不支持 representations: {sorted(unknown)}")
        manifest_path = self.root / "cache_manifest.json"
        if _sha256_file(manifest_path) != FROZEN_RESEARCH_TEST_CACHE_MANIFEST_SHA256:
            raise RuntimeError("research-test cache manifest SHA-256 不一致")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            self.manifest.get("protocol") != CRD_TF_RESEARCH_TEST_PROTOCOL_VERSION
            or self.manifest.get("cache_identity_sha256") != FROZEN_RESEARCH_TEST_CACHE_IDENTITY
            or self.manifest.get("complete") is not True
            or self.manifest.get("representations") != ["m", "w", "s"]
            or self.manifest.get("research_test_used") is not True
            or self.manifest.get("research_test_input_used") is not True
            or self.manifest.get("test_target_array_read") is not False
            or self.manifest.get("target_read") is not False
            or self.manifest.get("test_cache_created") is not True
            or self.manifest.get("model_inference_used") is not False
            or self.manifest.get("git", {}).get("commit") != FROZEN_RESEARCH_TEST_CACHE_COMMIT
            or self.manifest.get("git", {}).get("dirty") is not False
        ):
            raise RuntimeError("research-test cache manifest identity/边界不合格")
        split_meta = self.manifest.get("splits", {}).get("test", {})
        if split_meta.get("split") != "test" or int(split_meta.get("count", -1)) != EXPECTED_TEST_COUNT:
            raise RuntimeError("research-test cache test split identity 不合格")

        self.row_ids = self._open("test_row_ids.npy")
        if self.row_ids.shape != (EXPECTED_TEST_COUNT,) or self.row_ids.dtype != np.int64:
            raise RuntimeError("research-test row_ids shape/dtype 错误")
        row_hash = hashlib.sha256(np.asarray(self.row_ids).tobytes(order="C")).hexdigest()
        if row_hash != str(split_meta["row_ids_sha256"]):
            raise RuntimeError("research-test row_ids content SHA-256 不一致")
        if np.unique(self.row_ids).size != len(self.row_ids) or not np.all(np.diff(self.row_ids) > 0):
            raise RuntimeError("research-test row_ids 必须严格递增且无重复")
        self.arrays: dict[str, np.ndarray] = {}
        if "m" in self.representations:
            self.arrays["m_slow"] = self._open("test_m_slow.npy")
            self.arrays["m_fast"] = self._open("test_m_fast.npy")
        if "w" in self.representations:
            self.arrays["w"] = self._open("test_w.npy")
        if "s" in self.representations:
            self.arrays["s"] = self._open("test_s.npy")
        for name, array in self.arrays.items():
            if array.shape[0] != EXPECTED_TEST_COUNT:
                raise RuntimeError(f"research-test {name} 第一维与 row_ids 不一致")

    def _open(self, filename: str) -> np.ndarray:
        metadata = self.manifest.get("files", {}).get(filename)
        if metadata is None:
            raise RuntimeError(f"research-test manifest 缺少 {filename}")
        path = self.root / filename
        if path.stat().st_size != int(metadata["size_bytes"]) or _sha256_file(path) != metadata["sha256"]:
            raise RuntimeError(f"research-test cache 文件 identity 不一致: {filename}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != metadata["shape"] or str(array.dtype) != metadata["dtype"]:
            raise RuntimeError(f"research-test cache shape/dtype 不一致: {filename}")
        if metadata.get("finite") is not True:
            raise RuntimeError(f"research-test manifest 未确认 finite: {filename}")
        return array

    def position(self, dataset_row_id: int) -> int:
        row_id = int(dataset_row_id)
        position = int(np.searchsorted(self.row_ids, row_id))
        if position >= len(self.row_ids) or int(self.row_ids[position]) != row_id:
            raise KeyError(f"research-test cache 缺少 dataset_row_id={row_id}")
        return position

    def get(self, dataset_row_id: int) -> dict[str, torch.Tensor]:
        position = self.position(dataset_row_id)
        return {
            name: torch.from_numpy(np.array(array[position], copy=True))
            for name, array in self.arrays.items()
        }

    def verify_rows(self, dataset_row_ids: Iterable[int]) -> None:
        requested = np.asarray([int(value) for value in dataset_row_ids], dtype=np.int64)
        if requested.ndim != 1 or requested.size == 0:
            raise ValueError("research-test dataset rows 不得为空")
        positions = np.searchsorted(self.row_ids, requested)
        valid = positions < len(self.row_ids)
        matched = np.zeros(requested.shape, dtype=np.bool_)
        matched[valid] = np.asarray(self.row_ids)[positions[valid]] == requested[valid]
        if not bool(matched.all()):
            raise KeyError(f"research-test cache 缺少 dataset rows: {requested[~matched][:10].tolist()}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

