"""独立 W 缓存与数据适配；保留原 CRD 的样本资格和数据加载。"""
from __future__ import annotations
import hashlib
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from resp_train.data.cache import WholeNightCache
from resp_train.data.factory import build_window_data, ThoDataBundle
from resp_train.data.index import filter_index
from resp_train.data.research_v2 import read_research_v2_index
from . import artifacts as io
from .features import transform
from .spec import ARMS, COUNTS, SUBJECTS, SOURCE_LOCK, SOURCE_LOCK_SHA, config, SEEDS


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def raw_config(cfg):
    value = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    value.model.tf_representations = []
    value.data.tf_cache_path = None
    value.data.preload_windows = False
    return value


def audit_rows(cfg):
    if io.identity(SOURCE_LOCK)["sha256"] != SOURCE_LOCK_SHA:
        raise ValueError("历史数据来源锁漂移")
    source = io.read_json(SOURCE_LOCK)
    index = Path(cfg.data.dataset_root) / cfg.data.index_csv
    if index.resolve() != Path(source["dataset_index"]["path"]).resolve():
        raise ValueError("dataset index 路径漂移")
    io.verify(index, source["dataset_index"])
    audited = read_research_v2_index(cfg.data.dataset_root, cfg.data.index_csv, cfg)
    rows = {}
    for split in COUNTS:
        frame = filter_index(audited, cfg, split=split, max_windows=None,
                             sample_strategy=str(cfg.data[f"{split}_sample_strategy"]),
                             sample_seed=int(cfg.data[f"{split}_sample_seed"]))
        expected = source["cache_lock"]["row_identity"][f"{split}_row_content_sha256"]
        if (len(frame) != COUNTS[split] or frame.samp_id.nunique() != SUBJECTS[split]
                or frame.dataset_row_id.duplicated().any() or set(frame.split) != {split}
                or array_hash(np.sort(frame.dataset_row_id.to_numpy(np.int64))) != expected):
            raise ValueError(f"{split}样本合同漂移")
        rows[split] = frame.reset_index(drop=True)
    if set(rows["train"].samp_id) & set(rows["val"].samp_id):
        raise ValueError("train/val 受试者隔离失败")
    return audited, rows


def source_identities(index, rows):
    resolver = WholeNightCache(index)
    names = set()
    for frame in rows.values():
        for column in ("source_npz", "target_source_npz"):
            names.update(resolver.resolve(str(v)) for v in frame[column].unique())
    return {str(p): io.identity(p) for p in sorted(names)}


def verify_sources(payload):
    io.verify(payload["index"]["path"], payload["index"])
    for path, expected in payload["source_files"].items():
        io.verify(path, expected)


def data_receipt(session):
    io.require_data_scope(session)
    path = io.completed(Path(session) / "data", io.binding(session, "data"))
    if path is None:
        raise RuntimeError("先执行 prepare-data")
    return path, io.read_json(path / "data.json")


def prepare_data(session, retry=False):
    frozen = io.require_data_scope(session)
    key = io.binding(session, "data")
    prior = io.completed(session / "data", key)
    if prior:
        return prior
    cfg = raw_config(config("B", SEEDS[0], session / "unused"))
    with io.attempt(session / "data", key, retry) as output:
        io.write_json(output / "access.json", {"splits": ["train", "val"], "research_test": False})
        audited, rows = audit_rows(cfg)
        index = Path(cfg.data.dataset_root) / cfg.data.index_csv
        sources = source_identities(index, rows)
        for split, frame in rows.items():
            frame.to_csv(output / f"{split}_rows.csv", index=False)
        bundle = build_window_data(cfg, split="val", max_windows=None, sample_strategy=cfg.data.val_sample_strategy,
                                   sample_seed=cfg.data.val_sample_seed, shuffle=False, audited=audited)
        if not np.array_equal(bundle.rows.dataset_row_id, rows["val"].dataset_row_id):
            raise ValueError("公共参考 row 顺序不一致")
        reference = np.lib.format.open_memmap(output / "validation_reference.npy", mode="w+", dtype=np.float32,
                                              shape=(COUNTS["val"], 1, 18000))
        for i in range(len(bundle.dataset)):
            value = bundle.dataset[i]["target"].numpy()
            if value.shape != (1, 18000) or not np.isfinite(value).all():
                raise FloatingPointError("公共参考非法")
            reference[i] = value
        reference.flush()
        del reference
        cases = []
        for _, group in rows["val"].groupby("samp_id", sort=True):
            cases.extend(group.iloc[sorted({0, len(group)//2, len(group)-1})].dataset_row_id.astype(int).tolist())
        rng = np.random.Generator(np.random.PCG64(frozen["spec"]["shift_seed"]))
        shifts = np.stack([rng.choice(np.arange(60, 301), 3, replace=False) for _ in range(COUNTS["val"])])
        np.save(output / "val_shift_frames.npy", shifts, allow_pickle=False)
        io.write_json(output / "cases.json", {"rule": frozen["spec"]["case_rule"], "dataset_row_ids": cases})
        payload = {"index": {"path": str(index), **io.identity(index)}, "source_files": sources,
                   "counts": COUNTS, "subjects": SUBJECTS, "source_lock_sha256": SOURCE_LOCK_SHA,
                   "row_order": {s: array_hash(f.dataset_row_id.to_numpy(np.int64)) for s, f in rows.items()},
                   "source_key": str(cfg.data.bcg_input_key), "target_key": str(cfg.data.target_key)}
        verify_sources(payload)
        io.write_json(output / "data.json", payload)
        io.verify_provenance(frozen["provenance"], session)
    return output


def build_cache(session, arm, retry=False):
    frozen = io.load_session(session)
    if arm not in ARMS:
        raise ValueError("未知 arm")
    data_path, data = data_receipt(session)
    key = io.binding(session, "cache", arm=arm, data=io.identity(data_path / "manifest.json")["sha256"])
    parent = session / "cache" / arm
    prior = io.completed(parent, key)
    if prior:
        return prior
    rep = frozen["representations"][arm]
    with io.attempt(parent, key, retry) as output:
        verify_sources(data)
        io.write_json(output / "access.json", {"splits": ["train", "val"], "target_read": False, "test": False})
        source = WholeNightCache(data["index"]["path"])
        for split, count in COUNTS.items():
            rows = pd.read_csv(data_path / f"{split}_rows.csv")
            array = np.lib.format.open_memmap(output / f"{split}_w.npy", mode="w+", dtype=np.float32,
                                             shape=(count, *rep["shape"]))
            ids = rows.dataset_row_id.to_numpy(np.int64)
            np.save(output / f"{split}_row_ids.npy", ids, allow_pickle=False)
            for i, row in enumerate(rows.itertuples(index=False)):
                signal = source.get_arrays(str(row.source_npz), [str(row.bcg_signal_key)])[str(row.bcg_signal_key)]
                array[i] = transform(signal[int(row.window_start_sample):int(row.window_end_sample)], rep)
            array.flush()
            del array
        io.write_json(output / "cache.json", {"representation": rep, "data_manifest": io.identity(data_path / "manifest.json"),
                      "source_key": data["source_key"], "row_order": data["row_order"], "splits": ["train", "val"],
                      "target_read": False, "research_test": False, "finite": True})
        verify_sources(data)
        io.verify_provenance(frozen["provenance"], session)
    return output


class CacheReader:
    def __init__(self, path, split, rep, expected_ids):
        self.path = Path(path)
        metadata = io.read_json(self.path / "cache.json")
        if metadata["representation"] != rep or split not in metadata["splits"] or metadata["finite"] is not True:
            raise ValueError("缓存表示/split/finite 合同漂移")
        self.ids = np.load(self.path / f"{split}_row_ids.npy", allow_pickle=False)
        if self.ids.dtype != np.int64 or not np.array_equal(self.ids, expected_ids) or len(np.unique(self.ids)) != len(self.ids):
            raise ValueError("缓存 row 身份/顺序不一致")
        self.values = np.load(self.path / f"{split}_w.npy", mmap_mode="r", allow_pickle=False)
        if self.values.dtype != np.float32 or self.values.shape != (len(self.ids), *rep["shape"]):
            raise ValueError("缓存 shape/dtype 错误")

    def get(self, index, row_id):
        if int(self.ids[index]) != int(row_id):
            raise ValueError("缓存与真实 batch 的 row 不一致")
        value = np.array(self.values[index], copy=True)
        if not np.isfinite(value).all():
            raise FloatingPointError("缓存样本非有限")
        return {"w": torch.from_numpy(value)}


class ConditionDataset(Dataset):
    def __init__(self, base, reader):
        self.base, self.reader = base, reader

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = dict(self.base[index])
        item["tf"] = self.reader.get(index, item["meta"]["dataset_row_id"])
        return item


def condition_bundle(bundle, cfg, cache, split, rep):
    reader = CacheReader(cache, split, rep, bundle.rows.dataset_row_id.to_numpy(np.int64))
    dataset = ConditionDataset(bundle.dataset, reader)
    loader = DataLoader(dataset, batch_size=int(cfg.training.batch_size), shuffle=split == "train",
                        num_workers=0, pin_memory=str(cfg.training.device).startswith("cuda"), drop_last=False)
    return replace(bundle, dataset=dataset, loader=loader)


def training_data(session, arm, cfg):
    frozen = io.load_session(session)
    data_path, data = data_receipt(session)
    verify_sources(data)
    audited, rows = audit_rows(raw_config(cfg))
    key = io.binding(session, "cache", arm=arm, data=io.identity(data_path / "manifest.json")["sha256"])
    cache = io.completed(session / "cache" / arm, key)
    if cache is None:
        raise RuntimeError(f"未完成 {arm} 的 train/val 缓存")
    bundles = {}
    raw = raw_config(cfg)
    raw.data.preload_windows = bool(cfg.data.preload_windows)
    for split in COUNTS:
        if array_hash(rows[split].dataset_row_id.to_numpy(np.int64)) != data["row_order"][split]:
            raise ValueError("样本顺序漂移")
        bundle = build_window_data(raw, split=split, max_windows=None, audited=audited,
                                  sample_strategy=cfg.data[f"{split}_sample_strategy"],
                                  sample_seed=cfg.data[f"{split}_sample_seed"], shuffle=split == "train")
        bundles[split] = condition_bundle(bundle, cfg, cache, split, frozen["representations"][arm])
    if len(bundles["train"].loader) != 80 or bundles["train"].loader.batch_size != 128:
        raise ValueError("训练 batch/update 预算漂移")
    return ThoDataBundle(bundles["train"], bundles["val"], audited, bundles["train"].audit_summary)
