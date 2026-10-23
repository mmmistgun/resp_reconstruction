"""独立CWT缓存和只读历史缓存引用；数据入口固定为train/validation。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from resp_train import patch_aligned_tf_training as legacy
from resp_train.crd.training import optimizer_updates_per_epoch
from resp_train.data.cache import WholeNightCache
from resp_train.data.factory import build_tho_data
from resp_train.paper_evidence.cwt_time_frequency_v1.features import transform
from scripts import m4_s1_cwt_parameters_artifacts as io
from scripts import m4_s1_cwt_parameters_spec as spec

ROW_COLUMNS = ("dataset_row_id", "samp_id", "split", "source_npz", "target_source_npz",
               "window_start_sample", "window_end_sample", "bcg_signal_key", "target_signal_key", "allowed_losses")
REVIEWED_DATA_SOURCE = {
    "path": "resp_train/data/research_v2.py",
    "old_sha256": "37f013106ecd3270a79a87305c05f5105e0d046fc3a941c46f084a777eb024fb",
    "current_sha256": "3ff5abab6594f9d3a228b8925b5cb6ccf6193ba91c7d5fd002e68f7cad33ea77",
    "reason": "新增patch_aligned_tf_mamba专用冻结H-cache分支；原始波形读取与索引适配保持相同。",
}


def array_hash(array):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def require_confirmation(confirmed):
    if confirmed is not True:
        raise PermissionError("真实数据操作要求显式--confirm-data")


def raw_config(output, device="cpu"):
    cfg = spec.config("B", spec.SEEDS[0], output, device)
    if (cfg.data.train_split != "train" or cfg.data.val_split != "val"
            or cfg.data.drop_nonfinite_windows or cfg.data.tf_cache_path is not None):
        raise ValueError("数据split/非有限处理合同不符")
    return cfg


def audited_data(cfg):
    """仅按冻结index定义建立train/val；目标和波形按需加载。"""
    path = spec.ROOT / legacy.P2_LOCK_PATH
    if io.sha(path) != legacy.P2_LOCK_SHA256:
        raise ValueError("冻结数据来源记录变化")
    source = io.read_json(path)
    index = (Path(cfg.data.dataset_root) / cfg.data.index_csv).resolve()
    if index != Path(source["dataset_index"]["path"]).resolve() or io.sha(index) != source["dataset_index"]["sha256"]:
        raise ValueError("dataset index身份变化")
    data = build_tho_data(cfg)
    for split in spec.COUNTS:
        rows = getattr(data, split).rows
        ids = rows.dataset_row_id.to_numpy(np.int64)
        if (len(rows) != spec.COUNTS[split] or rows.samp_id.nunique() != spec.SUBJECTS[split]
                or rows.dataset_row_id.duplicated().any() or set(rows.split) != {split}
                or array_hash(np.sort(ids)) != source["cache_lock"]["row_identity"][f"{split}_row_content_sha256"]):
            raise ValueError(f"{split}样本集合/资格/分母变化")
    if set(data.train.rows.samp_id) & set(data.val.rows.samp_id):
        raise ValueError("train/validation受试者重叠")
    if optimizer_updates_per_epoch(len(data.train.loader), int(cfg.training.gradient_accumulation_steps)) != 80:
        raise ValueError("每epoch更新数不符")
    return data


def source_files(data, index):
    resolver = WholeNightCache(index)
    names = set()
    for split in spec.COUNTS:
        for col in ("source_npz", "target_source_npz"):
            names.update(resolver.resolve(str(v)).resolve() for v in getattr(data, split).rows[col].unique())
    return {str(p): io.identity(p) for p in sorted(names)}


def verify_sources(metadata):
    io.verify(metadata["index"]["path"], {k: metadata["index"][k] for k in ("sha256", "size_bytes")})
    for path, expected in metadata["source_files"].items():
        io.verify(path, expected)


def prepare_data(session, *, confirmed=False, retry_failed=False):
    require_confirmation(confirmed)
    io.require_data_scope(session)
    session = Path(session)
    binding = io.binding(session, "data")
    prior = io.completed(session / "data", binding)
    if prior:
        verify_sources(io.read_json(prior / "data.json"))
        return prior
    with io.attempt(session / "data", binding, retry_failed=retry_failed) as output:
        io.write_json(output / "access.json", {"splits": ["train", "val"], "research_test": False})
        cfg = raw_config(output / "unused")
        data = audited_data(cfg)
        index = (Path(cfg.data.dataset_root) / cfg.data.index_csv).resolve()
        sources = source_files(data, index)
        for split in spec.COUNTS:
            getattr(data, split).rows.to_csv(output / f"{split}_rows.csv", index=False)
        reference = np.lib.format.open_memmap(output / "validation_reference.npy", mode="w+",
            dtype=np.float32, shape=(spec.COUNTS["val"], 1, 18000))
        try:
            for i in range(len(data.val.dataset)):
                item = data.val.dataset[i]
                target = item["target"].numpy()
                if (target.shape != (1, 18000) or not np.isfinite(target).all()
                        or item["meta"]["dataset_row_id"] != int(data.val.rows.iloc[i].dataset_row_id)):
                    raise ValueError("公共validation参考或row身份非法")
                reference[i] = target
            reference.flush()
        finally:
            del reference
        metadata = {"index": {"path": str(index), **io.identity(index)}, "source_files": sources,
            "counts": spec.COUNTS, "subjects": spec.SUBJECTS, "source_lock_sha256": legacy.P2_LOCK_SHA256,
            "row_order": {s: array_hash(getattr(data, s).rows.dataset_row_id.to_numpy(np.int64)) for s in spec.COUNTS},
            "source_key": str(cfg.data.bcg_input_key), "target_key": str(cfg.data.target_key)}
        verify_sources(metadata)
        io.write_json(output / "data.json", metadata)
        io.load_session(session)
    return output


def data_receipt(session, *, verify_inputs=True):
    io.require_data_scope(session)
    path = io.completed(Path(session) / "data", io.binding(session, "data"))
    if path is None:
        raise RuntimeError("须先完成prepare-data")
    metadata = io.read_json(path / "data.json")
    if metadata["counts"] != spec.COUNTS or metadata["subjects"] != spec.SUBJECTS:
        raise ValueError("数据准备分母变化")
    if verify_inputs:
        verify_sources(metadata)
    return path, metadata


def cache_binding(session, representation, data_path):
    if representation not in spec.CWT_SETTINGS:
        raise ValueError("cache只接受20个唯一表示ID")
    return io.binding(session, "cache", representation=representation, data_sha256=io.sha(data_path / "manifest.json"))


def build_cache(session, representation, *, confirmed=False, retry_failed=False):
    require_confirmation(confirmed)
    frozen = io.require_data_scope(session)
    data_path, metadata = data_receipt(session)
    key = cache_binding(session, representation, data_path)
    parent = Path(session) / "cache" / representation
    prior = io.completed(parent, key)
    if prior:
        cache_receipt(session, representation)
        return prior
    rep = frozen["representations"][representation]
    with io.attempt(parent, key, retry_failed=retry_failed) as output:
        io.write_json(output / "access.json", {"splits": ["train", "val"], "research_test": False, "target_read": False})
        source = WholeNightCache(metadata["index"]["path"])
        for split, count in spec.COUNTS.items():
            rows = pd.read_csv(data_path / f"{split}_rows.csv")
            array = np.lib.format.open_memmap(output / f"{split}_w.npy", mode="w+", dtype=np.float32,
                                             shape=(count, *rep["shape"]))
            np.save(output / f"{split}_row_ids.npy", rows.dataset_row_id.to_numpy(np.int64), allow_pickle=False)
            try:
                for i, row in enumerate(rows.itertuples(index=False)):
                    signal = source.get_arrays(str(row.source_npz), [str(row.bcg_signal_key)])[str(row.bcg_signal_key)]
                    array[i] = transform(signal[int(row.window_start_sample):int(row.window_end_sample)], rep)
                array.flush()
            finally:
                del array
        io.write_json(output / "cache.json", {"representation": rep, "splits": ["train", "val"],
            "source_key": metadata["source_key"], "row_order": metadata["row_order"],
            "finite": True, "target_read": False, "research_test": False})
        io.write_json(output / "cache_record.json", {"kind": "local"})
        verify_sources(metadata)
        io.load_session(session)
    return output


def verify_preprocessing_sources(source_session, files, cfg):
    """仅接受已审阅的确切字节对；专用PatchTF分支在本轮raw dataset中不可达。"""
    required = {"resp_train/paper_evidence/cwt_time_frequency_v1/features.py", "resp_train/crd/tf_v1_features.py"}
    if not required <= files.keys():
        raise ValueError("历史cache缺少变换源码身份")
    reviewed = []
    for relative, expected in files.items():
        if relative not in required and not relative.startswith("resp_train/data/"):
            continue
        current = io.artifact_path(spec.ROOT, relative)
        if io.identity(current) == expected:
            continue
        allowed = REVIEWED_DATA_SOURCE
        if (relative != allowed["path"] or expected["sha256"] != allowed["old_sha256"]
                or io.sha(current) != allowed["current_sha256"]):
            io.verify(current, expected)
        if (str(cfg.model.name) == "patch_aligned_tf_mamba" or cfg.data.tf_cache_path is not None
                or list(cfg.model.tf_representations)):
            raise ValueError("审阅过的源码差异只适用于原始波形数据路径")
        io.verify(io.artifact_path(Path(source_session) / "source_snapshot", relative), expected)
        reviewed.append(dict(allowed))
    return reviewed


def verify_legacy_cache(source_session, representation, expected_rep, data_metadata, data_path):
    """只接受前序W0时频实验的已完成train/val缓存，完整核验数据和变换来源。"""
    from resp_train.paper_evidence.cwt_time_frequency_v1 import artifacts as old
    source_session = Path(source_session).resolve()
    io.verify(source_session / "session.json", old.read_json(source_session / "session_receipt.json"))
    source = old.read_json(source_session / "session.json")
    if (source["protocol"] != old.PROTOCOL or source["research_test_open"] is not False
            or source["provenance"]["dependencies"] != io.dependencies()
            or source["representations"][representation] != expected_rep):
        raise ValueError("历史cache的协议/依赖/表示不匹配")
    verify_preprocessing_sources(source_session, source["provenance"]["files"], raw_config(data_path / "unused"))
    old_data = old.completed(source_session / "data", old.binding(source_session, "data"))
    if old_data is None:
        raise ValueError("历史数据阶段未完成")
    previous = old.read_json(old_data / "data.json")
    for field in ("index", "source_files", "counts", "subjects", "source_lock_sha256", "source_key", "target_key", "row_order"):
        if previous[field] != data_metadata[field]:
            raise ValueError(f"历史cache数据身份不符: {field}")
    for split in spec.COUNTS:
        left = pd.read_csv(old_data / f"{split}_rows.csv")
        right = pd.read_csv(data_path / f"{split}_rows.csv")
        if not left[list(ROW_COLUMNS)].astype(str).equals(right[list(ROW_COLUMNS)].astype(str)):
            raise ValueError("历史cache样本身份或顺序不符")
    cache = old.completed(source_session / "cache" / representation,
        old.binding(source_session, "cache", arm=representation, data=old.identity(old_data / "manifest.json")["sha256"]))
    if cache is None:
        raise ValueError("历史cache阶段未完成")
    cm = old.read_json(cache / "cache.json")
    if cm["representation"] != expected_rep or cm["row_order"] != data_metadata["row_order"] or cm["splits"] != ["train", "val"]:
        raise ValueError("历史cache元数据不匹配")
    return cache


def import_cache(session, representation, source_session, *, confirmed=False, retry_failed=False):
    require_confirmation(confirmed)
    frozen = io.require_data_scope(session)
    data_path, metadata = data_receipt(session)
    key = cache_binding(session, representation, data_path)
    parent = Path(session) / "cache" / representation
    prior = io.completed(parent, key)
    if prior:
        cache_receipt(session, representation)
        return prior
    with io.attempt(parent, key, retry_failed=retry_failed) as output:
        source = Path(source_session).resolve()
        cache = verify_legacy_cache(source, representation, frozen["representations"][representation], metadata, data_path)
        reviewed = verify_preprocessing_sources(source, io.read_json(source / "session.json")["provenance"]["files"],
                                                raw_config(data_path / "unused"))
        io.write_json(output / "cache_record.json", {"kind": "legacy_reference",
            "source_session": str(source), "source_session_identity": io.identity(source / "session.json"),
            "cache": str(cache), "cache_manifest": io.identity(cache / "manifest.json"),
            "reviewed_source_differences": reviewed})
        io.write_json(output / "access.json", {"splits": ["train", "val"], "research_test": False,
                                              "read_only_reference": True})
        io.load_session(session)
    return output


def cache_receipt(session, representation):
    frozen = io.require_data_scope(session)
    data_path, metadata = data_receipt(session, verify_inputs=False)
    path = io.completed(Path(session) / "cache" / representation, cache_binding(session, representation, data_path))
    if path is None:
        raise RuntimeError(f"未完成CWT缓存: {representation}")
    record = io.read_json(path / "cache_record.json")
    if record["kind"] == "local":
        cache = path
    elif record["kind"] == "legacy_reference":
        source = Path(record["source_session"])
        io.verify(source / "session.json", record["source_session_identity"])
        cache = verify_legacy_cache(source, representation, frozen["representations"][representation], metadata, data_path)
        if cache.resolve() != Path(record["cache"]).resolve():
            raise ValueError("历史cache位置变化")
        io.verify(cache / "manifest.json", record["cache_manifest"])
    else:
        raise ValueError("未知cache来源类型")
    return path, cache


class CacheReader:
    def __init__(self, path, split, representation, expected_ids):
        if split not in ("train", "val"):
            raise ValueError("CWT读取入口只允许train/validation")
        self.path, self.split = Path(path), split
        metadata = io.read_json(self.path / "cache.json")
        if (metadata["representation"] != representation or metadata["splits"] != ["train", "val"]
                or metadata["finite"] is not True):
            raise ValueError("缓存表示或split非法")
        self.ids = np.load(self.path / f"{split}_row_ids.npy", allow_pickle=False)
        if (self.ids.dtype != np.int64 or not np.array_equal(self.ids, expected_ids)
                or len(np.unique(self.ids)) != len(self.ids)):
            raise ValueError("cache row身份或顺序变化")
        self.values = np.load(self.path / f"{split}_w.npy", mmap_mode="r", allow_pickle=False)
        if self.values.dtype != np.float32 or self.values.shape != (len(self.ids), *representation["shape"]):
            raise ValueError("cache shape/dtype不符")

    def get(self, index, row_id):
        if int(self.ids[index]) != int(row_id):
            raise ValueError("cache与batch row不一致")
        array = np.array(self.values[index], copy=True)
        if not np.isfinite(array).all():
            raise FloatingPointError("cache样本包含非有限值")
        return {"w": torch.from_numpy(array)}


class ConditionDataset(Dataset):
    def __init__(self, base, reader):
        self.base, self.reader = base, reader

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = dict(self.base[index])
        if item.get("tf"):
            raise ValueError("原始dataset已注入其他时频表示")
        item["tf"] = self.reader.get(index, item["meta"]["dataset_row_id"])
        return item


def training_data(session, arm, cfg, output):
    frozen = io.require_data_scope(session)
    data_path, metadata = data_receipt(session)
    io.write_json(Path(output) / "access_started.json", {"splits": ["train", "val"], "research_test": False,
                                                       "data_manifest": io.identity(data_path / "manifest.json")})
    data = audited_data(cfg)
    representation = spec.ARMS[arm].representation
    cache = cache_receipt(session, representation) if representation else None
    bundles = {}
    for split in spec.COUNTS:
        bundle = getattr(data, split)
        previous = pd.read_csv(data_path / f"{split}_rows.csv")
        if not bundle.rows[list(ROW_COLUMNS)].astype(str).equals(previous[list(ROW_COLUMNS)].astype(str)):
            raise ValueError("训练样本身份/顺序与数据准备不同")
        bundle.rows.to_csv(Path(output) / f"{split}_rows.csv", index=False)
        if representation:
            reader = CacheReader(cache[1], split, frozen["representations"][representation],
                                 bundle.rows.dataset_row_id.to_numpy(np.int64))
            dataset = ConditionDataset(bundle.dataset, reader)
            loader = DataLoader(dataset, batch_size=int(cfg.training.batch_size), shuffle=split == "train",
                num_workers=0, pin_memory=str(cfg.training.device).startswith("cuda"), drop_last=False)
            bundle = replace(bundle, dataset=dataset, loader=loader)
        bundles[split] = bundle
    io.write_json(Path(output) / "data_receipt.json", {"data": str(data_path),
        "data_manifest": io.identity(data_path / "manifest.json"),
        "cache": str(cache[0]) if cache else None,
        "cache_manifest": io.identity(cache[0] / "manifest.json") if cache else None,
        "row_order": metadata["row_order"]})
    return replace(data, **bundles)
