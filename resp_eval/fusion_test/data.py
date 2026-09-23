from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from resp_train.aligned_dual_view.artifacts import array_digest, read_json, sha256_file, write_json
from resp_train.aligned_dual_view.data import sample_identity, transform_identity
from resp_train.aligned_dual_view.features import extract_cwt_10hz, spec_digest
from resp_train.data.cache import WholeNightCache
from resp_train.data.index import filter_index
from resp_train.data.research_v2 import read_research_v2_index

from .artifacts import common_manifest, output_directory, verified_artifact
from .contract import LOCK_PATH, load_lock, require_confirmation, source_config


def select_test_rows(cfg, lock, *, confirmed):
    require_confirmation(confirmed)
    path = (Path(cfg.data.dataset_root) / cfg.data.index_csv).resolve()
    if sha256_file(path) != lock["dataset_index_sha256"]:
        raise ValueError("dataset index 偏离冻结版本")
    audited = read_research_v2_index(cfg.data.dataset_root, cfg.data.index_csv, cfg)
    ids = audited.dataset_row_id.to_numpy()
    if (audited.dataset_row_id.isna().any() or audited.dataset_row_id.duplicated().any()
            or not np.array_equal(ids, ids.astype(np.int64))):
        raise ValueError("dataset row ID 必须为唯一整数")
    frames = {split: filter_index(audited, cfg, split=split, max_windows=None,
                                  sample_strategy="stratified_random", sample_seed=lock["test_sample_seed"])
              for split in ("train", "val", "test")}
    for column in ("samp_id", "source_npz", "target_source_npz"):
        pools = {}
        for split, frame in frames.items():
            pools[split] = set(frame[column]) if column == "samp_id" else {
                str((path.parent / str(value)).resolve()) for value in frame[column]
            }
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            if pools[left] & pools[right]:
                raise ValueError(f"split 隔离失败: {left}/{right}/{column}")
    rows = frames["test"]
    row_ids = rows.dataset_row_id.to_numpy(dtype="<i8")
    if (len(rows) != lock["test_count"] or rows.samp_id.nunique() != lock["test_subject_count"]
            or not np.all(np.diff(row_ids) > 0)
            or hashlib.sha256(row_ids.tobytes()).hexdigest() != lock["test_row_ids_sha256"]):
        raise ValueError("test rows 数量、顺序、subject 或 row-ID 哈希不匹配")
    identities = [sample_identity(row) for _, row in rows.iterrows()]
    if any(row["split"] != "test" or row["start"] < 0 or row["end"] - row["start"] != 18000
           for row in identities):
        raise ValueError("test 窗口身份或长度错误")
    return path, rows, identities


def read_input(source, identity):
    key = identity["bcg_signal_key"]
    raw = source.get_arrays(identity["source_npz"], [key])[key]
    waveform = np.array(raw[identity["start"]:identity["end"]], dtype=np.float32, copy=True)
    if waveform.shape != (18000,) or not np.isfinite(waveform).all():
        raise FloatingPointError(f"test 输入非有限或长度错误: {identity['dataset_row_id']}")
    return waveform


def build_test_cache(output, *, lock_path=LOCK_PATH, confirm_research_test=False):
    require_confirmation(confirm_research_test)
    lock = load_lock(lock_path)
    cfg = source_config(lock, next(iter(lock["entries"])))
    with output_directory(output, kind="test_cache", cfg=cfg, lock=lock) as (root, started):
        path, rows, identities = select_test_rows(cfg, lock, confirmed=True)
        rows.to_csv(root / "test_selection.csv", index=False)
        source = WholeNightCache(path)
        feature_path = root / "test_cwt.npy"
        values = np.lib.format.open_memmap(feature_path, mode="w+", dtype=np.float32,
                                          shape=(len(rows), 97, 1800))
        records = []
        try:
            for position, identity in enumerate(tqdm(identities, desc="ADV research-test CWT",
                                                      disable=not bool(cfg.training.show_progress))):
                waveform = read_input(source, identity)
                feature = extract_cwt_10hz(waveform)
                if feature.shape != (97, 1800) or feature.dtype != np.float32:
                    raise ValueError("test CWT shape/dtype 错误")
                records.append({"identity": identity, "input_sha256": array_digest(waveform),
                                "feature_sha256": array_digest(feature)})
                values[position] = feature
        finally:
            values.flush()
            del values
        write_json(root / "test_rows.json", {"rows": records})
        identity = {
            "lock_id": lock["lock_id"], "representation": lock["representation"],
            "transform": transform_identity(), "rows_sha256": sha256_file(root / "test_rows.json"),
            "features_sha256": sha256_file(feature_path), "count": len(rows),
        }
        write_json(root / "manifest.json", {
            **common_manifest("test_cache", lock, started), **identity, "cache_id": spec_digest(identity),
            "features_bytes": feature_path.stat().st_size,
            "selection_sha256": sha256_file(root / "test_selection.csv"),
            "research_test_input_used": True, "target_read": False, "model_inference_used": False,
        })
    return Path(output).resolve()


class TestCache:
    def __init__(self, root, lock, identities):
        self.root = Path(root).resolve()
        manifest = self.manifest = verified_artifact(root, kind="test_cache", lock=lock)
        keys = ("lock_id", "representation", "transform", "rows_sha256", "features_sha256", "count")
        if manifest["cache_id"] != spec_digest({key: manifest[key] for key in keys}):
            raise ValueError("test cache identity 错误")
        if (manifest["representation"] != lock["representation"] or manifest["transform"] != lock["transform"]
                or manifest.get("target_read") is not False or manifest.get("model_inference_used") is not False
                or manifest.get("research_test_input_used") is not True):
            raise ValueError("test cache 表示或访问边界错误")
        for filename, field in (("test_rows.json", "rows_sha256"), ("test_cwt.npy", "features_sha256"),
                                ("test_selection.csv", "selection_sha256")):
            if sha256_file(self.root / filename) != manifest[field]:
                raise ValueError(f"test cache 哈希错误: {filename}")
        self.records = read_json(self.root / "test_rows.json")["rows"]
        if [record["identity"] for record in self.records] != identities or len(identities) != manifest["count"]:
            raise ValueError("test cache 样本集合或行顺序错误")
        feature_path = self.root / "test_cwt.npy"
        if feature_path.stat().st_size != manifest["features_bytes"]:
            raise ValueError("test cache 文件大小错误")
        self.values = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        if self.values.shape != (len(identities), 97, 1800) or self.values.dtype != np.float32:
            raise ValueError("test cache shape/dtype 错误")

    def get(self, position, waveform, *, use_cwt):
        record = self.records[position]
        if array_digest(waveform) != record["input_sha256"]:
            raise ValueError("test 波形内容与 cache 不一致")
        if not use_cwt:
            return None
        feature = np.array(self.values[position], copy=True)
        if array_digest(feature) != record["feature_sha256"]:
            raise ValueError("test CWT 内容损坏")
        return feature
