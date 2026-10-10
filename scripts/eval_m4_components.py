"""原始 M4 固定 checkpoint research-test：18 项新评价与六项冻结参照。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import traceback
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from scripts import m4_components_runtime as training
from scripts import eval_patch_aligned_tf_mamba as shared
from scripts import eval_p1_components as p1test
from scripts.m4_components_model import ARMS, REFERENCES, TRAIN_ARMS, M4ComponentsModel, band_indices

PROTOCOL = "m4-components-research-test-v1-20261009"
PROTOCOL_PATH = ROOT / "docs/experiments/m4_components_research_test_v1_20261009.md"
TRAINING_SESSION_SHA256 = "9f5f43120f2664f4b2eb214e73890e431c18320944c0f88aa70ad4c10b64f833"
VALIDATION_RECEIPT_SHA256 = "7f0dc351f5d33c60e90d8d07f53442597ab2584c9658bd731677ae0bb994f926"
REFERENCE_TEST_ROOT = ROOT / "runs/p1_components_v1/research_test_v1_20261007"
REFERENCE_ALLOWLIST_SHA256 = "8ac8b2305f66cd4f54f6a317f7c1dad9336bd88d393206fa84b5f300ad68fa47"
REFERENCE_SUMMARY_SHA256 = "bae9f68d1834483dc9cd1b734f01d98ea5826186b9f1c41d8c423af7d962d29e"
COUNT, SUBJECTS, SAMPLE_SEED = shared.COUNT, shared.SUBJECTS, shared.SAMPLE_SEED
EVIDENCE_ROLE = shared.EVIDENCE_ROLE
BATCH_SIZE = 32
sha, read_json, write_json = training.sha, training.read_json, training.write_json
contained, require_confirmation, selected_epoch = shared.contained, shared.require_confirmation, shared.selected_epoch
completed = shared.completed


def test_contract():
    return {**shared.test_contract(), "batch_size": BATCH_SIZE,
        "comparisons": training.COMPARISONS, "normalization": "groupnorm_representation_variant",
        "increments": {name: list(pair) for name, pair in training.INCREMENTS.items()},
        "bands": {a: arm.band for a, arm in ARMS.items()}, "references": REFERENCES,
        "reference_allowlist_sha256": REFERENCE_ALLOWLIST_SHA256,
        "reference_summary_sha256": REFERENCE_SUMMARY_SHA256}


def code_identity():
    sources = training.code_identity()
    sources.update(p1test.code_identity())
    for path in (Path(__file__), Path(shared.__file__), PROTOCOL_PATH,
                 ROOT / "scripts/run_m4_components_test_shard.sh", ROOT / "tests/test_m4_components_research_test.py"):
        sources[str(path.relative_to(ROOT))] = sha(path)
    return sources


def reference_test_sources():
    """只读核验历史 test 的来源记录与哈希；不解析指标值或访问 test 数据源。"""
    root = REFERENCE_TEST_ROOT.resolve()
    if sha(root / "allowlist.json") != REFERENCE_ALLOWLIST_SHA256:
        raise ValueError("P1 test allowlist 身份变化")
    lock = p1test.load_allowlist(root)
    expected_contract = {**shared.test_contract(), "batch_size": BATCH_SIZE}
    if any(lock["contract"].get(k) != v for k, v in expected_contract.items()):
        raise ValueError("历史 test 推理/数据合同与本轮不一致")
    summary = completed(root, root / "summary")
    if summary is None or sha(summary / "receipt.json") != REFERENCE_SUMMARY_SHA256:
        raise ValueError("P1 test 汇总身份不符")
    sources = read_json(summary / "sources.json")
    cells = {}
    for arm, old_arm in REFERENCES.items():
        for seed in training.SEEDS:
            cell_name = f"{old_arm}_seed{seed}"
            entries = [e for e in lock["entries"] if e["cell"] == cell_name]
            if len(entries) != 1:
                raise ValueError("历史 test 缺少唯一参照 cell")
            entry = entries[0]
            result = completed(root, root / "evaluation" / cell_name)
            if result is None or sha(result / "receipt.json") != sources[cell_name]:
                raise ValueError("历史 test cell 完成记录不符")
            receipt = read_json(result / "receipt.json")
            required = {"metrics.csv", "metrics_summary.csv", "evaluation.json", "test_rows.csv",
                "evaluation_config.yaml", "environment.json", "access_started.json", "started.json"}
            if not required.issubset(receipt["artifacts"]):
                raise ValueError("历史 test 缺少必要产物身份")
            evaluation = read_json(result / "evaluation.json")
            expected = {"cell": cell_name, "arm": old_arm, "seed": seed,
                "selected_epoch": entry["selected_epoch"], "checkpoint_sha256": entry["checkpoint"]["sha256"],
                "rows": COUNT, "subjects": SUBJECTS, "evidence_role": EVIDENCE_ROLE}
            if any(evaluation.get(k) != v for k, v in expected.items()):
                raise ValueError("历史 test checkpoint/选点/数据身份不符")
            for key in ("checkpoint", "config", "completion"):
                if sha(entry[key]["path"]) != entry[key]["sha256"]:
                    raise ValueError(f"历史 test 来源变化: {key}")
            environment = read_json(result / "environment.json")
            if any(environment[k] != entry["training_environment"][k] for k in ("torch", "cuda", "packages", "gpu")):
                raise ValueError("历史 test 环境与训练不符")
            cells[f"{arm}_seed{seed}"] = {"path": str(result), "source_arm": old_arm,
                "source_cell": cell_name, "receipt_sha256": sha(result / "receipt.json"),
                "selected_epoch": entry["selected_epoch"], "parameters": entry["parameters"],
                **{key: entry[key] for key in ("checkpoint", "config", "completion")}}
    return {"root": str(root), "allowlist_sha256": REFERENCE_ALLOWLIST_SHA256,
        "summary_receipt_sha256": REFERENCE_SUMMARY_SHA256,
        "development_subjects": lock["development_subjects"], "cells": cells}


def checkpoint_frequencies(state, arm, full=None):
    """NoTF 不接收频率；其余臂与来自 W-full checkpoint 的完整坐标逐值核对。"""
    key = "condition.frequencies_hz"
    if ARMS[arm].band == "none":
        if any(k.startswith(("condition.", "fusion.")) for k in state):
            raise ValueError("NoTF checkpoint 含条件模块")
        return None
    frequencies = state[key].cpu().numpy()
    if frequencies.dtype != np.float64 or not np.isfinite(frequencies).all():
        raise ValueError("checkpoint 频率 dtype/finite 不符")
    if full is None:
        if ARMS[arm].band != "W-full":
            raise ValueError("完整频率来源须为 W-full checkpoint")
        band_indices(frequencies, "W-full")
    elif not np.array_equal(frequencies, np.asarray(full)[band_indices(full, ARMS[arm].band)]):
        raise ValueError("checkpoint 频率坐标与所选频带不符")
    return frequencies.copy()


def prepare(training_session, output):
    """固定完整 24-cell 选点和历史参照；不读取 test index、manifest 或数组。"""
    session, root = Path(training_session).resolve(), Path(output).resolve()
    training.load_session(session)
    if sha(session / "session.json") != TRAINING_SESSION_SHA256:
        raise ValueError("训练 session 不属于本轮已完成矩阵")
    if (root / "allowlist_receipt.json").exists():
        lock = load_allowlist(root)
        if lock["training_session"] != str(session):
            raise ValueError("allowlist 对应另一训练 session")
        return root
    root.mkdir(parents=True, exist_ok=True)
    with training.mutex(root / ".prepare.lock"):
        if set(p.name for p in root.iterdir()) - {".prepare.lock"}:
            raise FileExistsError("输出目录已有产物，请使用独立路径")
        summaries = [p for p in (session / "summary").glob("*/receipt.json") if sha(p) == VALIDATION_RECEIPT_SHA256]
        if len(summaries) != 1:
            raise ValueError("缺少冻结的完整 validation 汇总")
        summary = read_json(summaries[0])
        if set(summary["sources"]) != {c["cell"] for c in training.plan(include_references=True)}:
            raise ValueError("validation 汇总矩阵不完整")
        for name, digest in summary["artifacts"].items():
            if sha(contained(summaries[0].parent, name)) != digest:
                raise ValueError("validation 汇总产物身份变化")
        if sha(ROOT / training.legacy.P2_LOCK_PATH) != training.legacy.P2_LOCK_SHA256:
            raise ValueError("development 来源记录身份变化")
        source = read_json(ROOT / training.legacy.P2_LOCK_PATH)
        references = reference_test_sources()
        entries, development, row_identities, full = [], set(), None, None
        for cell in training.plan(include_references=True):
            # B0/B1 的 completion/config/checkpoint 保持原 P1 身份，只有新 allowlist 使用 B 标签。
            source_cell = summary["sources"][cell["cell"]]
            expected_arm = REFERENCES.get(cell["arm"], cell["arm"])
            directory = (training.REFERENCE_SESSION / "cells" / f"{expected_arm}_seed{cell['seed']}"
                         if cell["arm"] in REFERENCES else session / "cells" / cell["cell"])
            if (Path(source_cell["path"]).resolve() != directory.resolve()
                    or source_cell["source_arm"] != expected_arm or source_cell["role"] != cell["role"]):
                raise ValueError("validation cell 参照路径或身份不符")
            completion_path = directory / "completed.json"
            completion = read_json(completion_path)
            if sha(completion_path) != source_cell["completed_sha256"]:
                raise ValueError("cell 完成来源与 validation 汇总不一致")
            if (completion["arm"], completion["seed"]) != (expected_arm, cell["seed"]):
                raise ValueError("完成 cell 标签不符")
            for name, digest in completion["artifacts"].items():
                if sha(contained(directory, name)) != digest:
                    raise ValueError(f"冻结产物身份变化: {cell['cell']}/{name}")
            cfg_path = directory / "config.yaml"
            cfg = OmegaConf.load(cfg_path)
            builder = training.p1.config if cell["arm"] in REFERENCES else training.config
            expected = builder(expected_arm, cell["seed"], directory, cfg.training.device)
            identity = training.legacy.config_identity(cfg)
            if identity != training.legacy.config_identity(expected):
                raise ValueError("训练配置与矩阵不一致")
            final = torch.load(contained(directory, completion["final_checkpoint"]), map_location="cpu", weights_only=False)
            epoch = selected_epoch(final["history"], completion)
            if final["epoch"] != completion["completed_epochs"] or final["config"] != identity:
                raise ValueError("final checkpoint epoch/config 不符")
            del final
            checkpoint_path = contained(directory, completion["best_checkpoint"])
            best = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            training.legacy.finite_tree(best["model_state_dict"])
            if best["epoch"] != epoch or best["config"] != identity:
                raise ValueError("best checkpoint epoch/config 不符")
            frequencies = checkpoint_frequencies(best["model_state_dict"], cell["arm"], full)
            if full is None:
                full = frequencies
            attempt_dir = contained(directory, completion["metrics"]).parent
            current, subjects = {}, {}
            for split, count, n_subjects in (("train", 10141, 32), ("val", 2675, 7)):
                rows = pd.read_csv(attempt_dir / f"{split}_rows.csv")
                ids = rows.dataset_row_id.to_numpy(np.int64)
                digest = hashlib.sha256(np.sort(ids).tobytes()).hexdigest()
                if (len(rows) != count or rows.samp_id.nunique() != n_subjects or set(rows.split) != {split}
                        or len(np.unique(ids)) != len(ids)
                        or digest != source["cache_lock"]["row_identity"][f"{split}_row_content_sha256"]):
                    raise ValueError("development 行身份不符")
                subjects[split] = set(rows.samp_id.astype(int))
                development.update(subjects[split])
                current[split] = sha(attempt_dir / f"{split}_rows.csv")
            if subjects["train"] & subjects["val"]:
                raise ValueError("train/validation 受试者重叠")
            if row_identities is None:
                row_identities = current
            elif row_identities != current:
                raise ValueError("跨 cell development 样本不一致")
            entry = {**cell, "evaluation_role": "reference" if cell["arm"] in REFERENCES else "evaluate",
                "selected_epoch": epoch, "parameters": completion["parameters"],
                "checkpoint": {"path": str(checkpoint_path), "sha256": sha(checkpoint_path)},
                "config": {"path": str(cfg_path), "sha256": sha(cfg_path)},
                "training_config": OmegaConf.to_container(cfg, resolve=True),
                "completion": {"path": str(completion_path), "sha256": sha(completion_path)},
                "training_environment": read_json(directory / "environment.json")}
            if cell["arm"] in REFERENCES:
                reference = references["cells"][cell["cell"]]
                for key in ("checkpoint", "config", "completion", "selected_epoch", "parameters"):
                    if reference[key] != entry[key]:
                        raise ValueError(f"本轮参照与历史 test 不一致: {key}")
                entry["reference_test"] = reference
            entries.append(entry)
            del best
        if sorted(development) != references["development_subjects"]:
            raise ValueError("新旧 development 受试者身份不一致")
        sources = code_identity()
        with tarfile.open(root / "source_snapshot.tar.gz", "x:gz") as archive:
            for name in sources:
                archive.add(ROOT / name, arcname=name)
        if sources != code_identity():
            raise RuntimeError("allowlist 准备期间源码变化")
        write_json(root / "allowlist.json", {"protocol": PROTOCOL, "training_session": str(session),
            "training_session_sha256": TRAINING_SESSION_SHA256, "validation_receipt_sha256": VALIDATION_RECEIPT_SHA256,
            "contract": test_contract(), "entries": entries, "reference_test_sources": references,
            "development_subjects": sorted(development),
            "development_row_files": row_identities, "dataset_index": source["dataset_index"],
            "frequencies_hz": full.tolist(), "code_sha256": sources,
            "snapshot_sha256": sha(root / "source_snapshot.tar.gz"),
            "test_array_read": False, "test_manifest_read": False, "test_metric_values_read": False,
            "historical_test_provenance_read": True, "model_inference_used": False})
        write_json(root / "allowlist_receipt.json", {"sha256": sha(root / "allowlist.json")})
    return root


def load_allowlist(root):
    root = Path(root)
    if sha(root / "allowlist.json") != read_json(root / "allowlist_receipt.json")["sha256"]:
        raise ValueError("allowlist SHA-256 不符")
    lock = read_json(root / "allowlist.json")
    if (lock["protocol"] != PROTOCOL or lock["contract"] != test_contract()
            or lock["training_session_sha256"] != TRAINING_SESSION_SHA256
            or lock["validation_receipt_sha256"] != VALIDATION_RECEIPT_SHA256
            or [{k: e[k] for k in ("arm", "name", "seed", "cell", "role")} for e in lock["entries"]] != training.plan(include_references=True)
            or lock["reference_test_sources"] != reference_test_sources()
            or lock["code_sha256"] != code_identity()
            or sha(root / "source_snapshot.tar.gz") != lock["snapshot_sha256"]):
        raise ValueError("allowlist 矩阵、合同或源码身份变化")
    for entry in lock["entries"]:
        reference = lock["reference_test_sources"]["cells"].get(entry["cell"])
        if entry["evaluation_role"] != ("reference" if entry["arm"] in REFERENCES else "evaluate"):
            raise ValueError("allowlist 评价/参照角色不符")
        if entry["arm"] in REFERENCES:
            if entry.get("reference_test") != reference or reference is None:
                raise ValueError("allowlist 历史 test 参照变化")
            for key in ("checkpoint", "config", "completion", "selected_epoch", "parameters"):
                if entry[key] != reference[key]:
                    raise ValueError("allowlist 参照 checkpoint 身份变化")
    return lock


class BandTestReader:
    def __init__(self, full_frequencies, arm, *, confirmed=False):
        require_confirmation(confirmed)
        if ARMS[arm].band == "none":
            raise ValueError("NoTF 不构造 CWT reader")
        self.source = shared.test_cache.TfV1ResearchTestCacheReader(
            shared.test_cache.FROZEN_RESEARCH_TEST_CACHE_ROOT, split="test", representations=("w",))
        if self.source.manifest["fixed_transform_spec"] != shared.fixed_transform_spec():
            raise ValueError("test CWT 表示规格不符")
        actual = self.source._open("w_frequencies_hz.npy")
        self.indices = band_indices(actual, ARMS[arm].band)
        if actual.dtype != np.float64 or not np.array_equal(actual, full_frequencies):
            raise ValueError("test 与 checkpoint 的 W-CWT 频率不一致")
        if self.source.arrays["w"].shape != (COUNT, 97, 360) or self.source.arrays["w"].dtype != np.float32:
            raise ValueError("test W cache shape/dtype 不符")

    def get(self, row_id):
        value = self.source.get(row_id)["w"]
        if not torch.isfinite(value).all():
            raise FloatingPointError("test W cache 包含非有限值")
        return {"w": value[self.indices].contiguous()}


def guarded_batches(loader, rows, arm):
    offset = 0
    scales = {"none": 0, "W-full": 97, "H": 41, "L": 56}[ARMS[arm].band]
    for batch in loader:
        count = len(batch["x"])
        expected = rows.iloc[offset:offset + count]
        if count <= 0 or len(expected) != count or set(batch.get("tf", {})) != ({"w"} if scales else set()):
            raise ValueError("test batch 数量/条件 keys 不符")
        for key in shared.IDENTITY_COLUMNS:
            value = batch["meta"][key]
            value = value.cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
            if not np.array_equal(value, expected[key].to_numpy()):
                raise ValueError(f"test batch 身份/顺序不符: {key}")
        tensors = [(batch["x"], (count, 1, 18000)), (batch["target"], (count, 1, 18000))]
        if scales:
            tensors.append((batch["tf"]["w"], (count, scales, 360)))
        for value, shape in tensors:
            if value.shape != shape or not value.is_floating_point():
                raise ValueError("test batch shape/dtype 不符")
            if not torch.isfinite(value).all():
                raise FloatingPointError("test batch 包含非有限值")
        offset += count
        yield batch
    if offset != len(rows):
        raise ValueError("test loader 未完整覆盖样本")


def infer_metrics(model, loader, rows, cfg, device):
    predictions = shared.collect_predictions(model, guarded_batches(loader, rows, cfg.model.m4_components.arm),
        device=device, max_windows=len(rows), use_amp=True)
    for key in ("r_tho_hat", "tho_ref"):
        if predictions[key].shape != (len(rows), 1, 18000) or not np.isfinite(predictions[key]).all():
            raise FloatingPointError("test 预测/参考 shape 或 finite 不符")
    metrics = shared.evaluate_task_predictions(predictions, cfg, include_test_only=True, method=cfg.model.variant)
    return metrics, shared.check_test_metrics(metrics, rows)


@contextmanager
def attempt(root, parent):
    path = Path(parent) / f"attempt_{uuid4().hex}"
    path.mkdir(parents=True)
    write_json(path / "started.json", {"protocol": PROTOCOL, "command": sys.argv,
        "allowlist_sha256": sha(Path(root) / "allowlist.json")})
    try:
        yield path
        files = {str(p.relative_to(path)): sha(p) for p in path.iterdir() if p.is_file()}
        write_json(path / "receipt.json", {"allowlist_sha256": sha(Path(root) / "allowlist.json"), "artifacts": files})
    except BaseException:
        write_json(path / "failed.json", {"traceback": traceback.format_exc()})
        raise


def evaluate(root, cell_name, device, *, confirmed=False, retry=False):
    require_confirmation(confirmed)
    root = Path(root).resolve()
    lock = load_allowlist(root)
    entries = [e for e in lock["entries"] if e["cell"] == cell_name]
    if len(entries) != 1:
        raise ValueError("cell 不属于固定 allowlist")
    entry = entries[0]
    if entry["arm"] not in TRAIN_ARMS or entry["evaluation_role"] != "evaluate":
        raise ValueError("B0/B1 只读复用历史 test，不允许重新评价")
    parent = root / "evaluation" / cell_name
    parent.mkdir(parents=True, exist_ok=True)
    with training.mutex(parent / ".run.lock"):
        previous = completed(root, parent, retry=retry)
        if previous:
            return previous
        with attempt(root, parent) as output:
            if torch.device(device).type != "cuda":
                raise ValueError("正式 test 入口要求 CUDA")
            torch.cuda.set_device(device)
            environment = training.legacy.runtime(device)
            if any(environment[k] != entry["training_environment"][k] for k in ("torch", "cuda", "packages", "gpu")):
                raise ValueError("test 环境与训练环境不一致")
            write_json(output / "environment.json", environment)
            for key in ("checkpoint", "config", "completion"):
                if sha(entry[key]["path"]) != entry[key]["sha256"]:
                    raise ValueError(f"test 来源变化: {key}")
            cfg = OmegaConf.load(entry["config"]["path"])
            if OmegaConf.to_container(cfg, resolve=True) != entry["training_config"]:
                raise ValueError("test 配置与 allowlist 不一致")
            if (cfg.data.test_split != "test" or cfg.data.max_test_windows is not None
                    or cfg.data.test_sample_seed != SAMPLE_SEED or cfg.training.batch_size != BATCH_SIZE
                    or cfg.training.amp_dtype != "bfloat16" or not cfg.training.use_amp):
                raise ValueError("test 推理合同不符")
            checkpoint = torch.load(entry["checkpoint"]["path"], map_location="cpu", weights_only=False)
            if checkpoint["epoch"] != entry["selected_epoch"] or checkpoint["config"] != training.legacy.config_identity(cfg):
                raise ValueError("checkpoint 配置或选点不符")
            training.legacy.finite_tree(checkpoint["model_state_dict"])
            checkpoint_frequencies(checkpoint["model_state_dict"], entry["arm"], lock["frequencies_hz"])
            has_tf = ARMS[entry["arm"]].band != "none"
            model = M4ComponentsModel(entry["arm"], entry["seed"], lock["frequencies_hz"] if has_tf else None,
                cfg=shared.PatchTFConfig(**dict(cfg.model.architecture)))
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            if sum(p.numel() for p in model.parameters()) != entry["parameters"]:
                raise ValueError("模型参数量与完成记录不符")
            del checkpoint
            cfg.training.device, cfg.training.show_progress = device, False
            write_json(output / "access_started.json", {"split": "test", "count": COUNT,
                "cache_read": has_tf, "evidence_role": EVIDENCE_ROLE, "checkpoint": entry["checkpoint"]})
            if sha(lock["dataset_index"]["path"]) != lock["dataset_index"]["sha256"]:
                raise ValueError("test dataset index 身份变化")
            audited = shared.read_research_v2_index(cfg.data.dataset_root, cfg.data.index_csv, cfg)
            rows = shared.filter_index(audited, cfg, split="test", max_windows=None,
                sample_strategy=cfg.data.test_sample_strategy, sample_seed=SAMPLE_SEED)
            shared.check_test_rows(rows)
            if set(rows.samp_id.astype(int)) & set(lock["development_subjects"]):
                raise ValueError("test 与 development 受试者重叠")
            rows.to_csv(output / "test_rows.csv", index=False)
            raw_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            raw_cfg.model.name, raw_cfg.model.tf_representations = "crd_v1", []
            raw_cfg.data.tf_cache_path, raw_cfg.data.preload_windows = None, False
            bundle = shared.build_window_data(raw_cfg, split="test", max_windows=None, shuffle=False, audited=audited,
                sample_strategy=cfg.data.test_sample_strategy, sample_seed=SAMPLE_SEED)
            shared.check_test_rows(bundle.rows, rows)
            # NoTF 直接使用波形 dataset；只有条件臂打开冻结 test W cache。
            reader, dataset = None, bundle.dataset
            if has_tf:
                reader = BandTestReader(lock["frequencies_hz"], entry["arm"], confirmed=confirmed)
                reader.source.verify_rows(rows.dataset_row_id)
                dataset = shared.HTestDataset(dataset, reader)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
            OmegaConf.save(cfg, output / "evaluation_config.yaml")
            print(f"开始 research-test {cell_name}, epoch={entry['selected_epoch']}, {device}", flush=True)
            metrics, summary = infer_metrics(model, loader, rows, cfg, device)
            metrics.insert(0, "arm", entry["arm"])
            metrics.insert(1, "seed", entry["seed"])
            metrics.to_csv(output / "metrics.csv", index=False)
            summary.to_csv(output / "metrics_summary.csv", index=False)
            write_json(output / "evaluation.json", {"cell": cell_name, "arm": entry["arm"], "seed": entry["seed"],
                "selected_epoch": entry["selected_epoch"], "checkpoint_sha256": entry["checkpoint"]["sha256"],
                "rows": COUNT, "subjects": SUBJECTS, "evidence_role": EVIDENCE_ROLE, **shared.quality_flags(metrics)})
            load_allowlist(root)
    del model, bundle, reader, dataset, loader, metrics
    gc.collect()
    torch.cuda.empty_cache()
    return output


def summary_tables(frame):
    # 先校验原生 test 辅助指标与资格，再复用冻结的主候选配对及递增比较。
    native, reference = [], None
    for (arm, seed), group in frame.groupby(["arm", "seed"], sort=False):
        group = group.reset_index(drop=True)
        summary = shared.check_test_metrics(group, group[list(shared.IDENTITY_COLUMNS)])
        identity = group[[*shared.IDENTITY_COLUMNS, *shared.TARGET_COLUMNS]]
        if reference is None:
            reference = identity
        elif not identity.equals(reference):
            raise ValueError("跨 cell 样本或 target 资格变化")
        summary.insert(0, "arm", arm)
        summary.insert(1, "seed", seed)
        native.append(summary)
    return {**training.summary_tables(frame), "native_summary": pd.concat(native, ignore_index=True)}


def summarize(root, *, confirmed=False, retry=False):
    require_confirmation(confirmed)
    root = Path(root)
    lock = load_allowlist(root)
    parent = root / "summary"
    parent.mkdir(exist_ok=True)
    with training.mutex(parent / ".run.lock"):
        previous = completed(root, parent, retry=retry)
        if previous:
            return previous
        frames, sources = [], {}
        for entry in lock["entries"]:
            reference = entry.get("reference_test")
            result = Path(reference["path"]) if reference else completed(root, root / "evaluation" / entry["cell"])
            if result is None:
                raise RuntimeError("18 组 test 尚未全部完成")
            if reference and sha(result / "receipt.json") != reference["receipt_sha256"]:
                raise ValueError("历史 test 参照产物变化")
            receipt = read_json(result / "evaluation.json")
            source_arm = reference["source_arm"] if reference else entry["arm"]
            source_cell = reference["source_cell"] if reference else entry["cell"]
            if (receipt["cell"] != source_cell or receipt["checkpoint_sha256"] != entry["checkpoint"]["sha256"]
                    or receipt["arm"] != source_arm or receipt["seed"] != entry["seed"]
                    or receipt["selected_epoch"] != entry["selected_epoch"]
                    or receipt["rows"] != COUNT or receipt["subjects"] != SUBJECTS
                    or receipt["evidence_role"] != EVIDENCE_ROLE):
                raise ValueError("test cell 来源不符")
            frame = pd.read_csv(result / "metrics.csv")
            if not frame.seed.eq(entry["seed"]).all() or not frame.arm.eq(source_arm).all():
                raise ValueError("test metrics cell 标签不符")
            frame["arm"] = entry["arm"]
            frames.append(frame)
            sources[entry["cell"]] = {"path": str(result), "source_arm": source_arm,
                "source_cell": source_cell, "role": entry["evaluation_role"],
                "receipt_sha256": sha(result / "receipt.json")}
        tables = summary_tables(pd.concat(frames, ignore_index=True))
        tables["parameters"] = pd.DataFrame([{k: e[k] for k in ("arm", "seed", "parameters")} for e in lock["entries"]])
        tables["sources"] = pd.DataFrame([{"cell": cell, **source} for cell, source in sources.items()])
        with attempt(root, parent) as output:
            for name, frame in tables.items():
                frame.to_csv(output / f"{name}.csv", index=False)
            write_json(output / "sources.json", sources)
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--training-session", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    for command in ("evaluate", "status", "summarize"):
        p = sub.add_parser(command)
        p.add_argument("--allowlist", type=Path, required=True)
        if command != "status":
            p.add_argument("--confirm-research-test", action="store_true")
            p.add_argument("--retry-failed", action="store_true")
        if command == "evaluate":
            p.add_argument("--device", default="cuda:0")
            p.add_argument("--shard-index", type=int, default=0)
            p.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.command == "prepare":
        result = str(prepare(args.training_session, args.output))
    elif args.command == "status":
        lock = load_allowlist(args.allowlist)
        result = [{"cell": e["cell"], "status": "reference_verified" if e["evaluation_role"] == "reference" else "completed" if completed(args.allowlist,
            args.allowlist / "evaluation" / e["cell"], retry=True) else "pending_or_incomplete"} for e in lock["entries"]]
    elif args.command == "summarize":
        result = str(summarize(args.allowlist, confirmed=args.confirm_research_test, retry=args.retry_failed))
    else:
        require_confirmation(args.confirm_research_test)
        result = [str(evaluate(args.allowlist, c["cell"], args.device, confirmed=True, retry=args.retry_failed))
                  for c in training.plan(args.shard_index, args.shard_count)]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
