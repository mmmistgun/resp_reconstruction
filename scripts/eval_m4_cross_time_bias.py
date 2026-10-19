"""M4 交叉注意力 Δt 消融：三个固定 checkpoint test 评价及原始 M4 对照。"""
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

from scripts import m4_cross_time_bias_runtime as training
from scripts import eval_patch_aligned_tf_mamba as shared
from scripts import eval_p1_components as p1test
from scripts.m4_cross_time_bias_model import ARMS, REFERENCES, TRAIN_ARMS, CrossTimeBiasModel, band_indices

PROTOCOL = "m4-cross-time-bias-research-test-v1-20261019"
PROTOCOL_PATH = ROOT / "docs/experiments/m4_cross_time_bias_research_test_v1_20261019.md"
TRAINING_SESSION_SHA256 = "96975f061d9e1e0e51cbd75d8f0154f383a07389535d6423c0b9853d68c5ee43"
VALIDATION_RECEIPT_SHA256 = "e3f6206ef63157aaf9ebde106d6c9ffb10f5e5d2c350107e047866c44e1f0458"
REFERENCE_TEST_ROOT = training.PRESERVED_ROOT / "runs/p1_components_v1/research_test_v1_20261007"
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
        "mean_delta_threshold": training.spec()["mean_delta_threshold"],
        "threshold_rule": training.spec()["threshold_rule"],
        "bands": {a: arm.band for a, arm in ARMS.items()}, "references": REFERENCES,
        "reference_allowlist_sha256": REFERENCE_ALLOWLIST_SHA256,
        "reference_summary_sha256": REFERENCE_SUMMARY_SHA256}


def code_identity():
    sources = training.code_identity()
    sources.update(p1test.code_identity())
    for path in (Path(__file__), Path(shared.__file__), PROTOCOL_PATH,
                 ROOT / "scripts/run_m4_cross_time_bias_test.sh", ROOT / "tests/test_m4_cross_time_bias_research_test.py"):
        sources[str(path.relative_to(ROOT))] = sha(path)
    return sources


def reference_test_sources():
    """只读核验历史 test 的来源记录与哈希；不解析指标值或访问 test 数据源。"""
    root = REFERENCE_TEST_ROOT.resolve()
    if sha(root / "allowlist.json") != REFERENCE_ALLOWLIST_SHA256:
        raise ValueError("P1 test allowlist 身份变化")
    # 历史清单先由固定 allowlist SHA 锁定，逐项核验旧源码与快照；新增独立文件由本轮快照覆盖。
    lock = read_json(root / "allowlist.json")
    if sha(root / "source_snapshot.tar.gz") != lock["snapshot_sha256"]:
        raise ValueError("P1 test 源码快照身份变化")
    if any(sha(ROOT / name) != digest for name, digest in lock["code_sha256"].items()):
        raise ValueError("P1 test 历史源码身份变化")
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
                **{key: {"path": str(Path(entry[key]["path"]).resolve()), "sha256": entry[key]["sha256"]}
                   for key in ("checkpoint", "config", "completion")}}
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
    """固定六个 checkpoint 身份和三项历史参照；不读取 test index、manifest 或数组。"""
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
            # A0 的历史 completion/config/checkpoint 保持原 P1-M4 标签与字节。
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
            if Path(cfg.outputs.run_root).resolve() != directory.resolve():
                raise ValueError("训练输出路径未解析到对应 cell")
            expected.outputs.run_root = cfg.outputs.run_root
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
            "validation_summary": {"path": str(summaries[0].parent), "receipt_sha256": VALIDATION_RECEIPT_SHA256},
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
    validation = lock["validation_summary"]
    validation_path = Path(validation["path"]).resolve()
    if (validation["receipt_sha256"] != VALIDATION_RECEIPT_SHA256
            or not validation_path.is_relative_to(Path(lock["training_session"]).resolve() / "summary")
            or sha(validation_path / "receipt.json") != VALIDATION_RECEIPT_SHA256):
        raise ValueError("validation 对照来源变化")
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
    predictions = shared.collect_predictions(model, guarded_batches(loader, rows, cfg.model.m4_cross_time_bias.arm),
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
        raise ValueError("A0 只读复用历史 test，不允许重新评价")
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
            model = CrossTimeBiasModel(entry["arm"], entry["seed"], lock["frequencies_hz"] if has_tf else None,
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
    # 先校验原生 test 辅助指标与资格，再复用冻结的两臂配对与 0.01 判据。
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


def validation_tables(lock):
    """读取冻结 validation 汇总，用作跨 split 对照；不重新汇总或选择 checkpoint。"""
    directory = Path(lock["validation_summary"]["path"])
    if sha(directory / "receipt.json") != VALIDATION_RECEIPT_SHA256:
        raise ValueError("validation summary 身份变化")
    receipt = read_json(directory / "receipt.json")
    for name, digest in receipt["artifacts"].items():
        if sha(contained(directory, name)) != digest:
            raise ValueError("validation summary 产物身份变化")
    names = ("across_seed", "paired_summary", "threshold_summary", "subject_macro_across_seed",
             "subject_macro_paired_summary", "subject_macro_threshold_summary")
    return {name: pd.read_csv(directory / f"{name}.csv") for name in names}


def compare_splits(validation, test):
    records = []
    for prefix, aggregation in (("", "window_weighted"), ("subject_macro_", "subject_macro")):
        left = validation[prefix + "threshold_summary"].set_index("metric")
        right = test[prefix + "threshold_summary"].set_index("metric")
        if set(left.index) != set(training.sf.PRIMARY) or set(right.index) != set(training.sf.PRIMARY):
            raise ValueError("跨 split 比较缺少五项主指标")
        for metric in training.sf.PRIMARY:
            row = {"aggregation": aggregation, "metric": metric}
            for split, table in (("validation", left), ("test", right)):
                item = table.loc[metric]
                if item.threshold != training.spec()["mean_delta_threshold"] or not np.isfinite(item.delta_mean):
                    raise ValueError("跨 split 判据或指标无效")
                row.update({f"{split}_{key}": item[key] for key in ("delta_mean", "mean_classification",
                    "improved_at_threshold_seeds", "worsened_at_threshold_seeds", "below_threshold_seeds")})
            records.append(row)
    return pd.DataFrame(records)


def removal_decision(comparison):
    """五指标两种聚合跨 split 存在达阈值退步时，标记需保留取舍而非直接替换。"""
    adverse = comparison.loc[comparison.validation_mean_classification.eq("worsened") |
                             comparison.test_mean_classification.eq("worsened")]
    return {"threshold": training.spec()["mean_delta_threshold"],
        "scope": "five_primary_metrics_both_splits_and_aggregations_mean_deltas",
        "recommendation": "tradeoff_exceeds_threshold" if len(adverse) else "supports_removal_at_mean_threshold",
        "adverse_comparisons": adverse.to_dict("records"), "statistical_equivalence_test": False,
        "evidence_role": EVIDENCE_ROLE}


def write_report(output, validation, test, comparison):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {"local_rr_mae_bpm": "Local RR MAE (bpm; lower)",
        "whole_rr_abs_error_bpm": "Whole RR AE (bpm; lower)",
        "envelope_trajectory_mae": "Envelope trajectory MAE (lower)",
        "global_envelope_modulation_error": "Global envelope modulation error (lower)",
        "lag_aware_signed_pcc": "Lag-aware signed PCC (higher)"}
    translate = {"no_obvious_mean_difference": "均值无明显差异", "improved": "移除后改善", "worsened": "移除后变差"}
    decision = removal_decision(comparison)
    conclusion = ("两个 split、两种聚合的五项主指标均未出现达到 0.01 的均值退步，按本次判据支持去掉时间 bias。"
        if decision["recommendation"] == "supports_removal_at_mean_threshold" else
        "存在达到 0.01 的均值退步，按本次判据暂不支持直接去掉时间 bias，需结合具体指标取舍。")
    lines = ["# M4 交叉注意力 Δt 时间 bias：test 与 validation 对照", "", conclusion, "",
        "A0 为三个冻结原始 M4 test 参照，A1 为三个固定 NoCrossTimeBias checkpoint；最佳 epoch 均为 25/5/8。",
        "每个 test cell 为 2310 窗口/8 人；validation 为 2675 窗口/7 人，并用于选点。",
        "delta=A1−A0，误差负值/PCC 正值为改善；绝对均值差<0.01 记为均值无明显差异。",
        "三 seed 共用各自 split 人群，SD 为 seed 间样本 SD；本比较为 reused research/development evidence。", ""]
    for prefix, aggregation, title in (("", "window_weighted", "窗口加权"),
                                       ("subject_macro_", "subject_macro", "受试者宏平均")):
        actual = test[prefix + "across_seed"].set_index(["arm", "metric"])
        paired = test[prefix + "paired_summary"].set_index("metric")
        rows = comparison[comparison.aggregation.eq(aggregation)].set_index("metric")
        lines += [f"## {title}", "", "| 指标 | 原始 M4 test mean±SD | 去掉时间 bias test mean±SD | test delta mean±SD |",
            "|---|---:|---:|---:|"]
        for metric in labels:
            values = [f"{actual.loc[(a, metric), 'seed_mean']:.6f} ± {actual.loc[(a, metric), 'seed_sd']:.6f}" for a in ARMS]
            r = paired.loc[metric]
            lines.append(f"| {metric} | {values[0]} | {values[1]} | {r.delta_mean:+.6f} ± {r.delta_sd:.6f} |")
        lines += ["", "| 指标 | validation delta | validation 判据 | test delta | test 判据 | test 达阈值改善/退步/阈值内 seed 数 |",
            "|---|---:|---|---:|---|---|"]
        for metric in labels:
            r = rows.loc[metric]
            lines.append(f"| {metric} | {r.validation_delta_mean:+.6f} | {translate[r.validation_mean_classification]} | "
                f"{r.test_delta_mean:+.6f} | {translate[r.test_mean_classification]} | "
                f"{int(r.test_improved_at_threshold_seeds)}/{int(r.test_worsened_at_threshold_seeds)}/{int(r.test_below_threshold_seeds)} |")
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        for axis, (metric, label) in zip(axes.flat, labels.items()):
            for split, tables, color in (("Validation", validation, "tab:blue"), ("Research-test", test, "tab:orange")):
                values = tables[prefix + "across_seed"].set_index(["arm", "metric"])
                axis.errorbar([0, 1], [values.loc[(a, metric), "seed_mean"] for a in ARMS],
                    yerr=[values.loc[(a, metric), "seed_sd"] for a in ARMS], marker="o", capsize=4, color=color, label=split)
            axis.set_xticks([0, 1], ["Original M4", "No time bias"])
            axis.set_title(label, fontsize=10)
            axis.grid(axis="y", alpha=.2)
        axes.flat[0].legend(fontsize=9)
        axes.flat[5].axis("off")
        axes.flat[5].text(.05, .95, "Three-seed mean +/- sample SD\nValidation: 2675 windows / 7 subjects\nResearch-test: 2310 windows / 8 subjects\n\nFixed validation-selected checkpoints", va="top")
        fig.savefig(output / f"{prefix}validation_test.png", dpi=160)
        plt.close(fig)
        lines.append("")
    lines += ["0.01 是描述性均值判据，不是统计等效检验；逐 seed 和逐受试者变化保留在配对 CSV 中。",
        "该结论仅涉及交叉注意力时间 bias；频率 bias、物理对齐和时域注意力池化保持原结构。"]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(output / "decision.json", decision)


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
        results = {e["cell"]: completed(root, root / "evaluation" / e["cell"])
                   for e in lock["entries"] if e["evaluation_role"] == "evaluate"}
        if any(path is None for path in results.values()):
            raise RuntimeError("三组新 test 尚未全部完成")
        frames, sources = [], {}
        for entry in lock["entries"]:
            reference = entry.get("reference_test")
            result = Path(reference["path"]) if reference else results[entry["cell"]]
            if result is None:
                raise RuntimeError("三组新 test 尚未全部完成")
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
        validation = validation_tables(lock)
        comparison = compare_splits(validation, tables)
        tables["validation_test_comparison"] = comparison
        with attempt(root, parent) as output:
            for name, frame in tables.items():
                frame.to_csv(output / f"{name}.csv", index=False)
            write_json(output / "sources.json", sources)
            write_report(output, validation, tables, comparison)
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
