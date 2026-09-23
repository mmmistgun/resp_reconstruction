from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from resp_train.data.cache import WholeNightCache
from resp_train.data.index import filter_index
from resp_train.data.research_v2 import ResearchV2WindowDataset, read_research_v2_index

from .artifacts import array_digest, artifact_directory, read_json, sha256_file, verified_manifest, write_json
from .config import FORMAL_COUNTS, INDEX_SHA256, PROTOCOL, data_contract, validate_experiment_config
from resp_train.aligned_dual_view.features import extract_cwt_10hz, representation_spec, spec_digest


@dataclass
class Selection:
    index_path: Path
    rows: dict[str, pd.DataFrame]
    identity: dict


def transform_identity() -> dict:
    """数学规格相同但实现或数值依赖不同，也不得静默复用旧特征。"""
    directory = Path(__file__).resolve().parents[1] / "resp_train/aligned_dual_view"
    return {
        "implementation": {name: sha256_file(directory / name)
                           for name in ("features.py", "signal.py", "w0_grid.json")},
        "dependencies": {name: importlib.metadata.version(name)
                         for name in ("ssqueezepy", "numpy", "scipy", "torch")},
    }


def sample_identity(row) -> dict:
    return {
        "dataset_row_id": int(row["dataset_row_id"]), "split": str(row["split"]),
        "samp_id": int(row["samp_id"]), "coupling_state_id": int(row["coupling_state_id"]),
        "start": int(row["window_start_sample"]), "end": int(row["window_end_sample"]),
        "source_npz": str(row["source_npz"]), "bcg_signal_key": str(row["bcg_signal_key"]),
        "target_source_npz": str(row["target_source_npz"]), "target_signal_key": str(row["target_signal_key"]),
    }


def select_rows(cfg) -> Selection:
    validate_experiment_config(cfg)
    path = (Path(cfg.data.dataset_root) / cfg.data.index_csv).resolve()
    index_sha = sha256_file(path)
    if cfg.protocol.run_role == "formal" and index_sha != INDEX_SHA256:
        raise ValueError("formal dataset index 哈希偏离冻结数据集")
    audited = read_research_v2_index(cfg.data.dataset_root, cfg.data.index_csv, cfg)
    if audited.dataset_row_id.isna().any() or audited.dataset_row_id.duplicated().any():
        raise ValueError("dataset_row_id 必须全局非空且唯一")
    if not np.array_equal(audited.dataset_row_id.to_numpy(), audited.dataset_row_id.to_numpy(dtype=np.int64)):
        raise ValueError("dataset_row_id 必须为整数")
    # 索引审计不打开任何 test 波形；实际读入的窗口严格限于这两个 split。
    admitted = {split: filter_index(audited, cfg, split=split, max_windows=None)
                for split in ("train", "val")}
    for column in ("samp_id", "source_npz", "target_source_npz"):
        values = {}
        for split, frame in admitted.items():
            values[split] = set(frame[column]) if column == "samp_id" else {
                str((path.parent / str(value)).resolve()) for value in frame[column]
            }
        if values["train"] & values["val"]:
            raise ValueError(f"train/val 隔离失败: {column}")
    rows = {
        split: filter_index(audited, cfg, split=split,
                            max_windows=cfg.data[f"max_{split}_windows"],
                            sample_strategy=cfg.data[f"{split}_sample_strategy"],
                            sample_seed=cfg.data[f"{split}_sample_seed"])
        for split in ("train", "val")
    }
    identities = {}
    for split, frame in rows.items():
        if not len(frame) or not np.all(np.diff(frame.dataset_row_id.to_numpy()) > 0):
            raise ValueError(f"{split} rows 必须非空且 row ID 严格递增")
        if cfg.protocol.run_role == "formal" and len(frame) != FORMAL_COUNTS[split]:
            raise ValueError(f"formal {split} admission 数量漂移")
        records = [sample_identity(row) for _, row in frame.iterrows()]
        for row in records:
            if row["end"] - row["start"] != 18000 or row["start"] < 0:
                raise ValueError("窗口长度或起点无效")
        identities[split] = {"count": len(records), "rows_sha256": spec_digest(records)}
    return Selection(path, rows, {
        "index_path": str(path), "index_sha256": index_sha,
        "contract": data_contract(cfg), "splits": identities,
    })


def build_cache(cfg, output: str | Path) -> Path:
    validate_experiment_config(cfg)
    with artifact_directory(output, kind="cache", cfg=cfg) as (root, started):
        selection = select_rows(cfg)
        source_cache = WholeNightCache(selection.index_path)
        splits = {}
        for split, frame in selection.rows.items():
            # 保存实际选择的索引行；缓存阶段仅按输入 key 打开 NPZ 数组。
            frame.to_csv(root / f"{split}_selection.csv", index=False)
            feature_path = root / f"{split}_cwt.npy"
            values = np.lib.format.open_memmap(
                feature_path, mode="w+", dtype=np.float32, shape=(len(frame), 97, 1800),
            )
            records = []
            try:
                for position, (_, row) in enumerate(frame.iterrows()):
                    identity = sample_identity(row)
                    key = identity["bcg_signal_key"]
                    source = source_cache.get_arrays(identity["source_npz"], [key])[key]
                    waveform = np.asarray(source[identity["start"]:identity["end"]], dtype=np.float32)
                    if waveform.shape != (18000,) or not np.isfinite(waveform).all():
                        raise FloatingPointError(f"输入 shape/finite 错误: {identity['dataset_row_id']}")
                    feature = extract_cwt_10hz(waveform)
                    if feature.shape != (97, 1800) or feature.dtype != np.float32:
                        raise ValueError("CWT extractor shape/dtype 错误")
                    records.append({"identity": identity, "input_sha256": array_digest(waveform),
                                    "feature_sha256": array_digest(feature)})
                    values[position] = feature
            finally:
                values.flush()
                del values
            rows_path = root / f"{split}_rows.json"
            write_json(rows_path, {"rows": records})
            splits[split] = {
                "count": len(frame), "rows_file": rows_path.name, "rows_sha256": sha256_file(rows_path),
                "features_file": feature_path.name, "features_sha256": sha256_file(feature_path),
                "features_bytes": feature_path.stat().st_size,
                "selection_sha256": sha256_file(root / f"{split}_selection.csv"),
            }
        identity = {"selection": selection.identity, "representation": representation_spec(),
                    "transform": transform_identity(), "splits": splits}
        write_json(root / "manifest.json", {
            "kind": "cache", "protocol": PROTOCOL, "cache_id": spec_digest(identity), **identity,
            "code_sha256": started["code"]["sha256"], "target_read": False, "test_waveform_read": False,
        })
    return Path(output).resolve()


def verified_cache(root):
    """旧缓存只开放已核对的正式 ADV 来源，新缓存属于本轮独立协议。"""
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    if manifest.get("protocol") == PROTOCOL:
        return verified_manifest(root, "cache")
    from resp_train.aligned_dual_view.artifacts import verified_manifest as verify_adv
    if sha256_file(root / "manifest.json") != "ecce7f6c67bd4bc585e98e20f8c441ad19c328cde6c73c83e6e84ba6b6c1f424":
        raise ValueError("旧缓存不在冻结的 ADV formal 来源中")
    manifest = verify_adv(root, "cache")
    if manifest["code_sha256"] != "19b92747cb2a26a5cf45d6954aeebaae1ce70b3de5a01a266677de4761284040":
        raise ValueError("旧缓存源码身份不匹配")
    return manifest


class CacheReader:
    """只读 memmap；逐窗口同时核验索引身份、原始输入和特征内容。"""

    def __init__(self, root: str | Path, selection: Selection, *, split: str):
        if split not in {"train", "val"}:
            raise ValueError("ADV cache 只允许 train/val")
        self.root, self.split = Path(root).resolve(), split
        self.manifest = verified_cache(self.root)
        identity = {key: self.manifest[key] for key in ("selection", "representation", "transform", "splits")}
        if self.manifest.get("cache_id") != spec_digest(identity):
            raise ValueError("cache identity 哈希不匹配")
        if self.manifest["selection"] != selection.identity or self.manifest["representation"] != representation_spec():
            raise ValueError("cache 数据选择或表示身份不匹配")
        if self.manifest["transform"] != transform_identity():
            raise ValueError("cache 前处理实现或依赖版本不匹配")
        if self.manifest.get("target_read") is not False or self.manifest.get("test_waveform_read") is not False:
            raise ValueError("cache 访问边界不合格")
        meta = self.manifest["splits"][split]
        # 文件名固定，避免 manifest 将读路径重定向到其他产物。
        if meta["rows_file"] != f"{split}_rows.json" or meta["features_file"] != f"{split}_cwt.npy":
            raise ValueError("cache 文件名不合格")
        rows_path = self.root / meta["rows_file"]
        if sha256_file(self.root / f"{split}_selection.csv") != meta["selection_sha256"]:
            raise ValueError("cache selection 哈希不匹配")
        if sha256_file(rows_path) != meta["rows_sha256"]:
            raise ValueError("cache rows 哈希不匹配")
        self.records = read_json(rows_path)["rows"]
        expected = [sample_identity(row) for _, row in selection.rows[split].iterrows()]
        if [row["identity"] for row in self.records] != expected or meta["count"] != len(expected):
            raise ValueError("cache row 顺序或 sample 身份不匹配")
        path = self.root / meta["features_file"]
        if path.stat().st_size != meta["features_bytes"]:
            raise ValueError("cache 特征文件大小错误")
        if sha256_file(path) != meta["features_sha256"]:
            raise ValueError("cache 特征文件哈希不匹配")
        self.values = np.load(path, mmap_mode="r", allow_pickle=False)
        if self.values.shape != (len(expected), 97, 1800) or self.values.dtype != np.float32:
            raise ValueError("cache 特征 shape/dtype 错误")

    def get(self, position: int, *, identity: dict, waveform: torch.Tensor, include_features: bool):
        record = self.records[position]
        if identity != record["identity"]:
            raise ValueError("cache sample identity 错配")
        if array_digest(waveform) != record["input_sha256"]:
            raise ValueError(f"cache 输入内容错配: row={identity['dataset_row_id']}")
        if not include_features:
            return None
        feature = np.array(self.values[position], copy=True)
        if array_digest(feature) != record["feature_sha256"]:
            raise ValueError(f"cache 特征内容损坏: row={identity['dataset_row_id']}")
        return torch.from_numpy(feature)


class AlignedDataset(Dataset):
    def __init__(self, cfg, selection: Selection, reader: CacheReader):
        self.rows = selection.rows[reader.split]
        self.reader = reader
        self.uses_cwt = True
        self.base = ResearchV2WindowDataset(selection.index_path, self.rows, cfg, preload_windows=False)
        if not self.base.rows.dataset_row_id.equals(self.rows.dataset_row_id):
            raise ValueError("dataset 构造后样本集合发生变化")
        self.provenance = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, position):
        row = self.rows.iloc[position]
        item = self.base[position]
        identity = sample_identity(row)
        feature = self.reader.get(position, identity=identity, waveform=item["x"], include_features=self.uses_cwt)
        evidence = {
            "identity": identity, "input_sha256": array_digest(item["x"]),
            "target_sha256": array_digest(item["target"]),
            "rr_peak_mask_sha256": array_digest(item["meta"]["rr_peak_valid_mask"]),
        }
        previous = self.provenance.setdefault(position, evidence)
        if previous != evidence:
            raise ValueError("运行中 input/target/mask 内容发生变化")
        if feature is not None:
            item["tf"] = {"adv_cwt": feature}
        return item


def build_loaders(cfg, cache_root: str | Path, *, selection: Selection | None = None,
                  splits=("train", "val")):
    if not splits or len(set(splits)) != len(splits) or set(splits) - {"train", "val"}:
        raise ValueError("数据加载只允许唯一的 train/val splits")
    selection = selection or select_rows(cfg)
    readers = {split: CacheReader(cache_root, selection, split=split) for split in splits}
    loaders = {}
    for split, reader in readers.items():
        dataset = AlignedDataset(cfg, selection, reader)
        # 在优化前扫描全部已选窗口，先发现缓存错配/非有限目标，再进入昂贵训练。
        for index in range(len(dataset)):
            dataset[index]
        loaders[split] = DataLoader(
            dataset, batch_size=int(cfg.training.batch_size), shuffle=split == "train",
            num_workers=0, drop_last=False, pin_memory=str(cfg.training.device).startswith("cuda"),
            generator=torch.Generator().manual_seed(int(cfg.training.seed)),
        )
    return selection, loaders
