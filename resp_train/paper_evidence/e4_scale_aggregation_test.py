"""E4 test 附件：锁定 validation-selected checkpoint，复用 W0 test 对照。"""

from __future__ import annotations

import json
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from resp_train.crd.experiment import _validate_checkpoint_config
from resp_train.paper_evidence.e4_scale_aggregation_model import build_e4_model, AGGREGATION_CONTRACT, FREQUENCY_FILE_SHA
from resp_train.crd.tf_v1_research_test_data import (
    FROZEN_RESEARCH_TEST_CACHE_ROOT, FROZEN_RESEARCH_TEST_CACHE_MANIFEST_SHA256,
    FROZEN_RESEARCH_TEST_CACHE_IDENTITY,
)
from resp_train.data.factory import build_window_data
from resp_train.data.index import filter_index
from resp_train.data.research_v2 import read_research_v2_index
from resp_train.engine import collect_predictions
from resp_train.metrics.task import evaluate_task_predictions, summarize_task_metrics
from resp_train.paper_evidence import e4_scale_aggregation as training
from resp_train.paper_evidence.e1_scale_topology import PRIMARY, SEEDS, array_hash
from resp_train.paper_evidence.e1_scale_topology_runtime import identity, sha256_file, write_json, git_state

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "e4-w0-scale-aggregation-test-v1-20260917"
ARM = training.ARM
COUNT, SUBJECTS = 2310, 8
SELECTED_EPOCHS = (10, 5, 17)
TRAIN_LOCK_SHA = "464e073dbd5707a30575d606dec2a84dcd89a161945e537c463b214a15c2b493"
SOURCE_SUMMARY = Path("runs/e4_w0_scale_aggregation_v1/summary/summary_464e073dbd57_20260917T110009Z_b816da172253")
SOURCE_MANIFEST_SHA = "eccc4b2b0a725d19900073b6fbcadf5e00b3fc1a93f13cf5b79a6b7e77ba869b"
W0_AUDIT = Path("runs/crd_tf_v1/research_test_summary/access_audit.csv")
W0_AUDIT_SHA = "eb75cce11e1827b983f6540719e772c1e0288e8ffda06b7f3ae917bc2920d377"
LOCK_PATH = Path("docs/experiments/e4_w0_scale_aggregation_test_lock_20260917.json")
PROTOCOL_PATH = Path("docs/experiments/e4_w0_scale_aggregation_test_protocol_20260917.md")
SCRIPT_PATH = Path("scripts/eval_e4_w0_scale_aggregation_test.py")
TEST_PATH = Path("tests/test_e4_scale_aggregation_test.py")
OUTPUT = Path("runs/e4_w0_scale_aggregation_test_v1")
IDENTITY_COLUMNS = ("dataset_row_id", "samp_id", "split")
TARGET_COLUMNS = ("whole_rr_target_eligible", "local_rr_target_eligible", "local_rr_target_eligible_windows",
                  "joint_target_eligible", "envelope_spearman_target_eligible", "ibi_target_eligible")


def check_rows(rows: pd.DataFrame, reference: pd.DataFrame | None = None, *, row_hash: str | None = None) -> None:
    if (len(rows) != COUNT or rows.dataset_row_id.isna().any() or rows.dataset_row_id.dtype.kind not in "iu"
            or rows.dataset_row_id.duplicated().any() or rows.samp_id.isna().any()
            or rows.samp_id.nunique() != SUBJECTS or set(rows.split.astype(str)) != {"test"}):
        raise ValueError("E4 test rows 数量、split、samp_id 或唯一身份错误")
    if row_hash is not None and array_hash(rows.dataset_row_id.to_numpy()) != row_hash:
        raise ValueError("E4 test row 顺序哈希漂移")
    if reference is not None:
        for key in IDENTITY_COLUMNS:
            if not np.array_equal(rows[key].to_numpy(), reference[key].to_numpy()):
                raise ValueError(f"E4 test 与 W0 身份/顺序不一致: {key}")


def check_metrics(metrics: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    check_rows(metrics, reference)
    training.validate_metrics(metrics, reference)
    for key in TARGET_COLUMNS:
        if not np.array_equal(metrics[key].to_numpy(), reference[key].to_numpy()):
            raise ValueError(f"E4 test target eligibility 漂移: {key}")
    summary = summarize_task_metrics(metrics)
    for key in PRIMARY:
        if not np.isfinite(float(summary.iloc[0][key + "_mean"])) or int(summary.iloc[0][key + "_n"]) != COUNT:
            raise FloatingPointError(f"E4 test 五主指标必须完整有限: {key}")
    return summary


def quality_flags(metrics: pd.DataFrame) -> dict[str, Any]:
    counts = {key: int(metrics[key].astype(bool).sum()) for key in (
        "joint_prediction_degenerate", "envelope_spearman_prediction_degenerate")}
    return {"prediction_degeneracy": counts, "quality_acceptance_passed": not any(counts.values())}


def prepare_lock(root: Path = ROOT) -> Path:
    """核验已冻结的训练与结果文件；test cache 仅读取 manifest，数组留到正式评价校验。"""
    if (root / LOCK_PATH).exists():
        raise FileExistsError("E4 test lock 已存在")
    if sha256_file(root / SOURCE_SUMMARY / "manifest.json") != SOURCE_MANIFEST_SHA:
        raise ValueError("E4 validation summary manifest identity 漂移")
    training.verify_attempt(root / SOURCE_SUMMARY, phase="summary", lock_hash=TRAIN_LOCK_SHA)
    train_lock, train_hash = training.load_lock(root)
    if train_hash != TRAIN_LOCK_SHA:
        raise ValueError("E4 training implementation identity 漂移")
    receipt = json.loads((root / SOURCE_SUMMARY / "summary_receipt.json").read_text())
    if tuple(receipt["seeds"]) != SEEDS or set(receipt["source_runs"]) != {str(s) for s in SEEDS}:
        raise ValueError("E4 validation-selected 三 seed 来源不完整")
    if sha256_file(root / W0_AUDIT) != W0_AUDIT_SHA:
        raise ValueError("冻结 W0 test audit identity 漂移")
    audit = pd.read_csv(root / W0_AUDIT)
    audit = audit[audit.variant.eq("crd_tf102_w")].sort_values("seed")
    if tuple(audit.seed) != SEEDS:
        raise ValueError("冻结 W0 test audit seed 不完整")
    cache_root = FROZEN_RESEARCH_TEST_CACHE_ROOT
    cache_manifest_path = cache_root / "cache_manifest.json"
    if sha256_file(cache_manifest_path) != FROZEN_RESEARCH_TEST_CACHE_MANIFEST_SHA256:
        raise ValueError("E4 test cache manifest identity 漂移")
    cache_manifest = json.loads(cache_manifest_path.read_text())
    metadata = cache_manifest["splits"]["test"]
    if metadata["count"] != COUNT or metadata["samp_id_count"] != SUBJECTS:
        raise ValueError("E4 test cache split contract 漂移")
    files = {}

    def source(path: Path, expected: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if expected is not None:
            training.verify(path, expected)
        entry = identity(path)
        files[str(path.resolve())] = entry
        return {"path": str(path.resolve()), **entry}

    source(root / training.LOCK_PATH)
    source(root / W0_AUDIT)
    source(cache_manifest_path)
    for path in sorted((root / SOURCE_SUMMARY).iterdir()):
        if path.is_file():
            source(path)
    entries, reference_rows = [], None
    for seed, epoch in zip(SEEDS, SELECTED_EPOCHS, strict=True):
        item = receipt["source_runs"][str(seed)]
        formal_root = Path(item["path"])
        manifest = training.verify_attempt(formal_root, phase="formal", lock_hash=TRAIN_LOCK_SHA)
        source(formal_root / "manifest.json", item["manifest"])
        formal = json.loads((formal_root / "formal_receipt.json").read_text())
        if (int(formal["seed"]) != seed or manifest["seed"] != seed or formal["epochs"] != 80
                or formal["updates"] != 6400 or formal["selected_epoch"] != epoch or item["selected_epoch"] != epoch):
            raise ValueError("E4 checkpoint 必须由完整 validation 预选")
        run_dir = (formal_root / formal["run_dir"]).resolve()
        if not run_dir.is_relative_to(formal_root.resolve()):
            raise ValueError("E4 training run_dir 越界")
        cfg = OmegaConf.load(run_dir / "config.yaml")
        training.validate_config(cfg, OmegaConf.create(train_lock["baselines"][str(seed)]),
                                 output_root=formal_root / "training", device=str(cfg.training.device))
        history = pd.read_csv(run_dir / "train_history.csv")
        if training.validate_history(history, cfg) != epoch:
            raise ValueError("E4 checkpoint 不是完整 validation 最早最小 Local RR epoch")
        for name in ("formal_receipt.json", "freeze_receipt.json", "train_rows.csv", "val_rows.csv"):
            source(formal_root / name)
        development = set(pd.read_csv(formal_root / "train_rows.csv").samp_id) | set(pd.read_csv(formal_root / "val_rows.csv").samp_id)
        candidate = {name: source(run_dir / name, manifest["files"][str((run_dir / name).relative_to(formal_root))])
                     for name in ("config.yaml", "train_history.csv", "checkpoint_best_local_rr.pt")}
        old = next(e for e in train_lock["w0_entries"] if int(e["seed"]) == seed)
        row = audit.loc[audit.seed.eq(seed)].iloc[0]
        if row.checkpoint_sha256 != old["checkpoint"]["sha256"] or int(row.validation_selected_epoch) != old["selected_epoch"]:
            raise ValueError("E4 W0 test 对照与冻结 W0 身份不一致")
        old_root = root / old["run_dir"]
        full = {}
        for filename, key in (("research_test_metrics.csv", "metrics_sha256"),
                               ("research_test_metrics_summary.csv", "metrics_summary_sha256"),
                               ("research_test_metrics_manifest.json", "evaluation_manifest_sha256")):
            path = old_root / filename
            if sha256_file(path) != row[key]:
                raise ValueError(f"E4 W0 test source identity 漂移: {filename}")
            full[filename] = source(path)
        metrics = pd.read_csv(old_root / "research_test_metrics.csv")
        check_rows(metrics, reference_rows, row_hash=metadata["row_ids_sha256"])
        check_metrics(metrics, metrics)
        if development & set(metrics.samp_id):
            raise ValueError("E4 test 与 train/validation samp_id 交叉")
        reference_rows = metrics[list(IDENTITY_COLUMNS)].copy()
        entries.append({"seed": seed, "selected_epoch": epoch, "w0_selected_epoch": old["selected_epoch"],
                        "training_attempt": str(formal_root),
                        "training_config": OmegaConf.to_container(cfg, resolve=True), "candidate": candidate,
                        "w0_test": full, "development_samp_ids": sorted(map(int, development))})
    cache_files = {name: cache_manifest["files"][name]
                   for name in ("test_w.npy", "test_row_ids.npy", "w_frequencies_hz.npy")}
    if cache_files["w_frequencies_hz.npy"]["sha256"] != FREQUENCY_FILE_SHA:
        raise ValueError("E4 test 频率网格与训练聚合器不一致")
    code_paths = sorted((root / "resp_train").rglob("*.py")) + [root / p for p in (SCRIPT_PATH, TEST_PATH, PROTOCOL_PATH)]
    lock = {"protocol": PROTOCOL, "arm": ARM, "aggregation_contract": AGGREGATION_CONTRACT,
            "seeds": list(SEEDS), "selected_epochs": list(SELECTED_EPOCHS),
            "split": "test", "count": COUNT, "samp_id_count": SUBJECTS, "batch_size": 128,
            "amp_dtype": "bfloat16", "include_test_only": False,
            "training_lock_sha256": TRAIN_LOCK_SHA, "validation_summary_manifest_sha256": SOURCE_MANIFEST_SHA,
            "entries": entries, "source_files": files, "cache_root": str(cache_root), "cache_files": cache_files,
            "cache_manifest_sha256": FROZEN_RESEARCH_TEST_CACHE_MANIFEST_SHA256,
            "dataset_index": {"path": cache_manifest["dataset_index"], "sha256": cache_manifest["dataset_index_sha256"]},
            "row_order_sha256": metadata["row_ids_sha256"], "preparation_git": git_state(root),
            "prepared_at": datetime.now(timezone.utc).isoformat(),
            "preparation_access": "frozen configs, manifests and metric records; test signal/cache arrays not loaded",
            "code_files": {str(p.relative_to(root)): identity(p) for p in code_paths}}
    write_json(root / LOCK_PATH, lock)
    return root / LOCK_PATH


def load_lock(root: Path = ROOT) -> tuple[dict[str, Any], str]:
    path = root / LOCK_PATH
    lock = json.loads(path.read_text())
    if (lock["protocol"] != PROTOCOL or lock["arm"] != ARM or lock["aggregation_contract"] != AGGREGATION_CONTRACT
            or tuple(lock["seeds"]) != SEEDS
            or tuple(lock["selected_epochs"]) != SELECTED_EPOCHS or lock["split"] != "test"
            or lock["count"] != COUNT or lock["samp_id_count"] != SUBJECTS
            or lock["batch_size"] != 128 or lock["amp_dtype"] != "bfloat16"
            or lock["include_test_only"] is not False or lock["training_lock_sha256"] != TRAIN_LOCK_SHA
            or lock["validation_summary_manifest_sha256"] != SOURCE_MANIFEST_SHA
            or Path(lock["cache_root"]) != FROZEN_RESEARCH_TEST_CACHE_ROOT
            or lock["cache_manifest_sha256"] != FROZEN_RESEARCH_TEST_CACHE_MANIFEST_SHA256):
        raise ValueError("E4 test lock 固定矩阵或来源身份漂移")
    if [(e["seed"], e["selected_epoch"]) for e in lock["entries"]] != list(zip(SEEDS, SELECTED_EPOCHS)):
        raise ValueError("E4 test checkpoint allowlist 不完整")
    if lock["cache_files"]["w_frequencies_hz.npy"]["sha256"] != FREQUENCY_FILE_SHA:
        raise ValueError("E4 test 频率文件 identity 漂移")
    for relative, expected in lock["code_files"].items():
        training.verify(root / relative, expected)
    return lock, sha256_file(path)


@contextmanager
def attempt(parent: Path, phase: str, lock_hash: str, seed: int | None = None) -> Iterator[Path]:
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = parent / f"{phase}_{lock_hash[:12]}_{stamp}_{uuid4().hex[:12]}"
    output.mkdir(exist_ok=False)
    context = {"protocol": PROTOCOL, "phase": phase, "seed": seed, "split": "test", "arm": ARM,
               "implementation_lock_sha256": lock_hash, "command": sys.argv,
               "started_at": datetime.now(timezone.utc).isoformat()}
    write_json(output / "lifecycle_started.json", {**context, "status": "running"})
    print(f"E4 test attempt: {output}", flush=True)
    try:
        yield output
        write_json(output / "lifecycle_completed.json", {**context, "status": "completed",
                   "ended_at": datetime.now(timezone.utc).isoformat()})
        write_json(output / "manifest.json", {**context, "status": "completed", "files": {
            str(p.relative_to(output)): identity(p) for p in sorted(output.rglob("*")) if p.is_file()}})
        write_json(output / "freeze_receipt.json", {"protocol": PROTOCOL, "manifest": identity(output / "manifest.json")})
    except BaseException as exc:
        write_json(output / "lifecycle_failed.json", {**context, "status": "failed", "error": str(exc),
                   "error_type": type(exc).__name__, "traceback": traceback.format_exc()})
        raise


def verify_attempt(output: Path, lock_hash: str, phase: str) -> dict[str, Any]:
    output = output.resolve()
    if (output / "lifecycle_failed.json").exists():
        raise ValueError("E4 test attempt 已失败")
    freeze = json.loads((output / "freeze_receipt.json").read_text())
    training.verify(output / "manifest.json", freeze["manifest"])
    manifest = json.loads((output / "manifest.json").read_text())
    if (manifest["protocol"] != PROTOCOL or manifest["phase"] != phase or manifest["split"] != "test" or manifest["arm"] != ARM
            or manifest["status"] != "completed" or manifest["implementation_lock_sha256"] != lock_hash):
        raise ValueError("E4 test attempt identity 不匹配")
    required = {"lifecycle_completed.json", "summary_receipt.json", "seed_metrics.csv",
                "paired_seed_delta.csv", "three_seed_comparison.csv"} if phase == "summary" else {
        "lifecycle_completed.json", "evaluation_receipt.json", "metrics.csv", "metrics_summary.csv", "test_rows.csv",
        "access_started.json", "access_receipt.json", "environment.json", "resolved_config.yaml", "implementation_lock.json"}
    if not required.issubset(manifest["files"]):
        raise ValueError("E4 test manifest 缺少必需产物")
    for relative, expected in manifest["files"].items():
        path = (output / relative).resolve()
        if not path.is_relative_to(output):
            raise ValueError("E4 test manifest 路径越界")
        training.verify(path, expected)
    return manifest


def evaluation_config(entry: Mapping[str, Any], device: str, cache_root: str) -> tuple[DictConfig, DictConfig]:
    cfg = OmegaConf.load(entry["candidate"]["config.yaml"]["path"])
    if OmegaConf.to_container(cfg, resolve=True) != entry["training_config"]:
        raise ValueError("E4 test 训练配置与 allowlist 不一致")
    if (cfg.protocol.name != training.PROTOCOL or cfg.model.variant != "crd_tf102_w"
            or int(cfg.training.seed) != entry["seed"] or float(cfg.loss.effort_weight) != 0.25
            or float(cfg.loss.sync_weight) != 1.0 or int(cfg.model.initialization_seed) != entry["seed"]
            or OmegaConf.to_container(cfg.model.e4_aggregation, resolve=True) != AGGREGATION_CONTRACT
            or cfg.data.test_split != "test" or cfg.data.max_test_windows is not None
            or int(cfg.data.test_sample_seed) != 20260612 or int(cfg.training.batch_size) != 128
            or not cfg.training.use_amp or str(cfg.training.amp_dtype) != "bfloat16"):
        raise ValueError("E4 test config 必须使用完整冻结 test 和原生推理配置")
    cfg.training.device = device
    cfg.training.show_progress = False
    data_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    data_cfg.data.tf_research_test_cache_path = str(cache_root)
    return cfg, data_cfg


def guarded_batches(loader: Any, rows: pd.DataFrame):
    offset = 0
    for batch in loader:
        count = len(batch["x"])
        expected = rows.iloc[offset:offset + count]
        if count <= 0 or len(expected) != count or set(batch.get("tf", {})) != {"w"}:
            raise ValueError("E4 test batch 数量或 W keys 不合格")
        for key in IDENTITY_COLUMNS:
            value = batch["meta"][key]
            actual = value.cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
            if not np.array_equal(actual, expected[key].to_numpy()):
                raise ValueError(f"E4 test batch identity/order 错误: {key}")
        if tuple(batch["tf"]["w"].shape) != (count, 97, 360):
            raise ValueError("E4 test W shape 错误")
        for tensor in (batch["x"], batch["target"], batch["tf"]["w"]):
            if not bool(torch.isfinite(tensor).all()):
                raise FloatingPointError("E4 test input/target 非有限")
        offset += count
        yield batch
    if offset != len(rows):
        raise ValueError("E4 test loader 样本不完整")


def run_evaluation(seed: int, device: str = "cuda:0") -> Path:
    if seed not in SEEDS:
        raise ValueError("E4 test seed 不在冻结矩阵")
    lock, lock_hash = load_lock()
    entry = next(e for e in lock["entries"] if e["seed"] == seed)
    parent = ROOT / OUTPUT / "evaluation" / f"seed_{seed}"
    with training.phase_guard(parent, lock_hash):
        return _run_evaluation(seed, device, lock, lock_hash, entry, parent)


def _run_evaluation(seed: int, device: str, lock: dict, lock_hash: str, entry: dict, parent: Path) -> Path:
    for completed in parent.glob("*/freeze_receipt.json"):
        manifest = json.loads((completed.parent / "manifest.json").read_text())
        if manifest["implementation_lock_sha256"] == lock_hash:
            verify_attempt(completed.parent, lock_hash, "evaluation")
            raise FileExistsError(f"E4 test 同身份 seed 已完成: {completed.parent}")
    with attempt(parent, "evaluation", lock_hash, seed) as output:
        write_json(output / "environment.json", training.runtime_preflight(device))
        write_json(output / "implementation_lock.json", lock)
        write_json(output / "access_started.json", {"split": "test", "seed": seed, "count": COUNT,
                   "checkpoint": entry["candidate"]["checkpoint_best_local_rr.pt"], "cache_root": lock["cache_root"],
                   "dataset_index": lock["dataset_index"], "purpose": "E4 fixed-checkpoint test evaluation"})
        for path, expected in lock["source_files"].items():
            training.verify(Path(path), expected)
        cache_root = Path(lock["cache_root"])
        if sha256_file(cache_root / "cache_manifest.json") != lock["cache_manifest_sha256"]:
            raise ValueError("E4 test cache manifest 漂移")
        for name, expected in lock["cache_files"].items():
            training.verify(cache_root / name, expected)
        index = lock["dataset_index"]
        if sha256_file(Path(index["path"])) != index["sha256"]:
            raise ValueError("E4 test dataset index 漂移")
        cfg, data_cfg = evaluation_config(entry, device, lock["cache_root"])
        OmegaConf.save(data_cfg, output / "resolved_config.yaml")
        # 先确认被评价的模型身份，再开放真实 test 波形加载。
        checkpoint_path = Path(entry["candidate"]["checkpoint_best_local_rr.pt"]["path"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("config") != entry["training_config"]:
            raise ValueError("E4 test checkpoint 完整训练配置 identity 漂移")
        _validate_checkpoint_config(checkpoint.get("config"), cfg)
        if checkpoint["epoch"] != entry["selected_epoch"]:
            raise ValueError("E4 test checkpoint epoch 漂移")
        training.finite_tree(checkpoint["model_state_dict"])
        model = build_e4_model(cfg)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        reference = pd.read_csv(entry["w0_test"]["research_test_metrics.csv"]["path"])
        audited = read_research_v2_index(cfg.data.dataset_root, cfg.data.index_csv, cfg)
        rows = filter_index(audited, cfg, split="test", max_windows=None,
                            sample_strategy=str(cfg.data.test_sample_strategy), sample_seed=int(cfg.data.test_sample_seed))
        check_rows(rows, reference, row_hash=lock["row_order_sha256"])
        if set(rows.samp_id) & set(entry["development_samp_ids"]):
            raise ValueError("E4 test 与开发 split 的 samp_id 交叉")
        rows.to_csv(output / "test_rows.csv", index=False)
        data = build_window_data(data_cfg, split="test", max_windows=None, sample_strategy=str(cfg.data.test_sample_strategy),
                                 sample_seed=int(cfg.data.test_sample_seed), shuffle=False, audited=audited)
        check_rows(data.rows, rows, row_hash=lock["row_order_sha256"])
        if len(data.dataset) != COUNT:
            raise ValueError("E4 test dataset 缩小了样本集合")
        predictions = collect_predictions(model, guarded_batches(data.loader, rows), device=device, max_windows=COUNT, use_amp=True)
        metrics = evaluate_task_predictions(predictions, cfg, include_test_only=False, method=ARM)
        summary = check_metrics(metrics, reference)
        quality = quality_flags(metrics)
        metrics.insert(0, "seed", seed)
        metrics.insert(0, "arm", ARM)
        summary.insert(0, "seed", seed)
        summary.insert(0, "arm", ARM)
        summary.insert(0, "split", "test")
        summary.insert(0, "quality_acceptance_passed", quality["quality_acceptance_passed"])
        summary.insert(0, "selected_epoch", entry["selected_epoch"])
        metrics.to_csv(output / "metrics.csv", index=False)
        summary.to_csv(output / "metrics_summary.csv", index=False)
        write_json(output / "access_receipt.json", {"split": "test", "seed": seed, "rows": COUNT, "samp_ids": SUBJECTS,
                   "row_order_sha256": array_hash(rows.dataset_row_id.to_numpy()), "cache_identity": FROZEN_RESEARCH_TEST_CACHE_IDENTITY,
                   "cache_files_verified": lock["cache_files"], "checkpoint": entry["candidate"]["checkpoint_best_local_rr.pt"],
                   "test_sample_seed": int(cfg.data.test_sample_seed), "evaluation_commit": git_state()["commit"],
                   "evidence_role": "subject-separated existing test split; reused research evidence"})
        write_json(output / "evaluation_receipt.json", {"protocol": PROTOCOL, "seed": seed, "split": "test", "arm": ARM,
                   "selected_epoch": entry["selected_epoch"], "checkpoint_sha256": entry["candidate"]["checkpoint_best_local_rr.pt"]["sha256"],
                   "rows": COUNT, "samp_ids": SUBJECTS, "primary_finite": True, **quality,
                   "target_eligibility_matches_w0": True, "row_order_sha256": lock["row_order_sha256"]})
    return output


def summarize(runs: list[Path]) -> Path:
    lock, lock_hash = load_lock()
    with training.phase_guard(ROOT / OUTPUT / "summary", lock_hash):
        return _summarize(runs, lock, lock_hash)


def _summarize(runs: list[Path], lock: dict, lock_hash: str) -> Path:
    if len(runs) != 3 or len({p.resolve() for p in runs}) != 3:
        raise ValueError("E4 test 汇总要求三个不同的完成 attempt")
    frames, sources, seeds = [], {}, set()
    for output in runs:
        manifest = verify_attempt(output, lock_hash, "evaluation")
        seed = int(manifest["seed"])
        if seed not in SEEDS or seed in seeds:
            raise ValueError("E4 test 汇总 seed 重复或越界")
        seeds.add(seed)
        entry = next(e for e in lock["entries"] if e["seed"] == seed)
        receipt = json.loads((output / "evaluation_receipt.json").read_text())
        if (receipt["seed"] != seed or receipt["selected_epoch"] != entry["selected_epoch"]
                or receipt["checkpoint_sha256"] != entry["candidate"]["checkpoint_best_local_rr.pt"]["sha256"]
                or receipt["split"] != "test" or receipt["rows"] != COUNT or receipt["arm"] != ARM):
            raise ValueError("E4 test 完成回执与 checkpoint allowlist 不一致")
        for item in entry["w0_test"].values():
            training.verify(Path(item["path"]), item)
        reference = pd.read_csv(entry["w0_test"]["research_test_metrics.csv"]["path"])
        metrics = pd.read_csv(output / "metrics.csv")
        if not metrics.seed.eq(seed).all() or not metrics.arm.eq(ARM).all():
            raise ValueError("E4 test metrics seed/arm 不一致")
        check_rows(metrics, pd.read_csv(output / "test_rows.csv"), row_hash=lock["row_order_sha256"])
        computed = check_metrics(metrics, reference)
        quality = quality_flags(metrics)
        if any(receipt.get(key) != value for key, value in quality.items()):
            raise ValueError("E4 test 质量回执与逐窗口结果不一致")
        candidate = pd.read_csv(output / "metrics_summary.csv")
        full = pd.read_csv(entry["w0_test"]["research_test_metrics_summary.csv"]["path"])
        full_computed = check_metrics(reference, reference)
        for frame, expected in ((candidate, computed), (full, full_computed)):
            if len(frame) != 1:
                raise ValueError("E4 test 每 seed summary 必须一行")
            for metric in PRIMARY:
                if (not np.isclose(frame.iloc[0][metric + "_mean"], expected.iloc[0][metric + "_mean"], atol=1e-12, rtol=0)
                        or int(frame.iloc[0][metric + "_n"]) != COUNT):
                    raise ValueError("E4 test summary 数值或分母不一致")
        if (candidate.iloc[0]["seed"] != seed or candidate.iloc[0]["arm"] != ARM or candidate.iloc[0]["split"] != "test"):
            raise ValueError("E4 test summary seed/arm/split 不一致")
        if (candidate.iloc[0]["selected_epoch"] != entry["selected_epoch"]
                or bool(candidate.iloc[0]["quality_acceptance_passed"]) != quality["quality_acceptance_passed"]):
            raise ValueError("E4 test summary checkpoint/quality identity 不一致")
        full.insert(0, "seed", seed)
        full.insert(0, "arm", "W0_FULL")
        full.insert(0, "split", "test")
        full.insert(0, "selected_epoch", entry["w0_selected_epoch"])
        full.insert(0, "quality_acceptance_passed", quality_flags(reference)["quality_acceptance_passed"])
        full.insert(0, "source_sha256", entry["w0_test"]["research_test_metrics.csv"]["sha256"])
        candidate.insert(0, "source_sha256", sha256_file(output / "metrics.csv"))
        frames.extend((full, candidate))
        sources[str(seed)] = {"path": str(output.resolve()), "manifest": identity(output / "manifest.json"), "selected_epoch": entry["selected_epoch"]}
    combined = pd.concat(frames, ignore_index=True)
    paired, aggregate = training.paired_tables(combined)
    # E4 的配对公式与 validation 相同；本附件的输出必须明确标记 test。
    paired["split"], aggregate["split"] = "test", "test"
    provenance = combined.set_index(["arm", "seed"])
    for side, arm in (("full", "W0_FULL"), ("candidate", ARM)):
        for field in ("selected_epoch", "source_sha256"):
            paired[f"{side}_{field}"] = [provenance.loc[(arm, int(seed)), field] for seed in paired.seed]
        paired[f"{side}_n"] = COUNT
    aggregate["n_windows_per_seed"] = COUNT
    parent = ROOT / OUTPUT / "summary"
    for previous in parent.glob("*/freeze_receipt.json"):
        m = json.loads((previous.parent / "manifest.json").read_text())
        if m["implementation_lock_sha256"] == lock_hash:
            verify_attempt(previous.parent, lock_hash, "summary")
            raise FileExistsError(f"E4 test 同身份汇总已完成: {previous.parent}")
    with attempt(parent, "summary", lock_hash) as output:
        combined.to_csv(output / "seed_metrics.csv", index=False)
        paired.to_csv(output / "paired_seed_delta.csv", index=False, na_rep="NA")
        aggregate.to_csv(output / "three_seed_comparison.csv", index=False, na_rep="NA")
        write_json(output / "summary_receipt.json", {"protocol": PROTOCOL, "split": "test", "seeds": list(SEEDS),
                   "rows_per_seed": COUNT, "new_metric_rows": 3 * COUNT, "source_runs": sources,
                   "w0_test_sources": {str(e["seed"]): e["w0_test"] for e in lock["entries"]},
                   "delta_definition": "positive means W0_FULL better; error=(candidate-full)/full*100; PCC=full-candidate",
                   "evidence_role": "fixed validation-selected checkpoints; subject-separated existing test; reused research evidence"})
    return output
