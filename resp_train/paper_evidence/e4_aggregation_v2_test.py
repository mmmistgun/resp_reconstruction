"""四候选完整 validation 矩阵锁定后的 test 评价与配对汇总。"""

from __future__ import annotations

import json
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from resp_train.crd.experiment import _validate_checkpoint_config
from resp_train.data.factory import build_window_data
from resp_train.data.index import filter_index
from resp_train.data.research_v2 import read_research_v2_index
from resp_train.engine import collect_predictions
from resp_train.metrics.task import evaluate_task_predictions
from resp_train.paper_evidence import e4_aggregation_v2 as training
from resp_train.paper_evidence import e4_scale_aggregation_test as reference_tools
from resp_train.paper_evidence.e4_aggregation_v2_model import ARMS, aggregation_contract, build_model, FREQUENCY_FILE_SHA
from resp_train.paper_evidence.e1_scale_topology import PRIMARY, SEEDS, array_hash
from resp_train.paper_evidence.e1_scale_topology_runtime import identity, sha256_file, write_json, git_state

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "e4-scale-aggregation-v2-test-v1-20260917"
OUTPUT = Path("runs/e4_scale_aggregation_v2_test")
LOCK_PATH = Path("docs/experiments/e4_scale_aggregation_v2_test_lock_20260917.json")
PROTOCOL_PATH = Path("docs/experiments/e4_scale_aggregation_v2_protocol_20260917.md")
SCRIPT_PATH = Path("scripts/eval_e4_aggregation_v2_test.py")
TEST_PATH = Path("tests/test_e4_aggregation_v2_test.py")
COUNT, SUBJECTS = 2310, 8
W0_AUDIT, W0_AUDIT_SHA = reference_tools.W0_AUDIT, reference_tools.W0_AUDIT_SHA
CACHE_ROOT = reference_tools.FROZEN_RESEARCH_TEST_CACHE_ROOT
CACHE_MANIFEST_SHA = reference_tools.FROZEN_RESEARCH_TEST_CACHE_MANIFEST_SHA256
CACHE_IDENTITY = reference_tools.FROZEN_RESEARCH_TEST_CACHE_IDENTITY
check_rows, check_metrics = reference_tools.check_rows, reference_tools.check_metrics
quality_flags, guarded_batches = reference_tools.quality_flags, reference_tools.guarded_batches
IDENTITY_COLUMNS = reference_tools.IDENTITY_COLUMNS


def prepare_lock(validation_summary: Path, root: Path = ROOT):
    """仅从完整12项冻结 validation 汇总生成 checkpoint allowlist。"""
    destination = root / LOCK_PATH
    if destination.exists():
        raise FileExistsError("E4-v2 test lock 已存在")
    lock, train_digest = training.load_lock(root)
    validation_summary = validation_summary.resolve()
    if not validation_summary.is_relative_to((root / training.OUTPUT / "summary").resolve()):
        raise ValueError("E4-v2 test validation summary 来源路径越界")
    training.verify_attempt(validation_summary, phase="summary", lock_hash=train_digest)
    receipt = json.loads((validation_summary / "summary_receipt.json").read_text())
    slots = {f"{arm}/{seed}" for arm in ARMS for seed in SEEDS}
    if (receipt["arms"] != list(ARMS) or receipt["seeds"] != list(SEEDS) or receipt["split"] != "val"
            or set(receipt["source_runs"]) != slots or receipt["implementation_lock_sha256"] != train_digest):
        raise ValueError("E4-v2 test 必须在完整四候选三 seed validation 矩阵后准备")
    if sha256_file(root / W0_AUDIT) != W0_AUDIT_SHA or sha256_file(CACHE_ROOT / "cache_manifest.json") != CACHE_MANIFEST_SHA:
        raise ValueError("E4-v2 W0 test/cache manifest 来源漂移")
    audit = pd.read_csv(root / W0_AUDIT)
    audit = audit[audit.variant.eq("crd_tf102_w")].sort_values("seed")
    if tuple(audit.seed) != SEEDS:
        raise ValueError("E4-v2 W0 test seed 来源不完整")
    cache = json.loads((CACHE_ROOT / "cache_manifest.json").read_text())
    metadata = cache["splits"]["test"]
    if metadata["count"] != COUNT or metadata["samp_id_count"] != SUBJECTS:
        raise ValueError("E4-v2 test cache 样本合同漂移")
    cache_files = {name: cache["files"][name] for name in ("test_w.npy", "test_row_ids.npy", "w_frequencies_hz.npy")}
    if cache_files["w_frequencies_hz.npy"]["sha256"] != FREQUENCY_FILE_SHA:
        raise ValueError("E4-v2 test 与训练频率网格不一致")
    files = {}
    def source(path, expected=None):
        if expected is not None:
            training.verify(path, expected)
        value = identity(path); files[str(path.resolve())] = value
        return {"path": str(path.resolve()), **value}
    source(root / training.LOCK_PATH); source(root / W0_AUDIT); source(CACHE_ROOT / "cache_manifest.json")
    for path in validation_summary.iterdir():
        if path.is_file():
            source(path)
    w0_sources, reference_metrics = {}, {}
    for old in lock["w0_entries"]:
        seed = old["seed"]
        row = audit.loc[audit.seed.eq(seed)].iloc[0]
        if row.checkpoint_sha256 != old["checkpoint"]["sha256"] or row.validation_selected_epoch != old["selected_epoch"]:
            raise ValueError("E4-v2 W0 test checkpoint 与训练锚点不一致")
        full = {}
        for filename, key in (("research_test_metrics.csv", "metrics_sha256"),
                               ("research_test_metrics_summary.csv", "metrics_summary_sha256"),
                               ("research_test_metrics_manifest.json", "evaluation_manifest_sha256")):
            path = root / old["run_dir"] / filename
            if sha256_file(path) != row[key]:
                raise ValueError("E4-v2 W0 test 文件身份漂移")
            full[filename] = source(path)
        frame = pd.read_csv(full["research_test_metrics.csv"]["path"])
        check_rows(frame, row_hash=metadata["row_ids_sha256"])
        check_metrics(frame, frame)
        reference_metrics[seed] = frame
        w0_sources[str(seed)] = {"selected_epoch": old["selected_epoch"], "files": full}
    entries = []
    for arm in ARMS:
        for seed in SEEDS:
            selected = receipt["source_runs"][f"{arm}/{seed}"]
            output = Path(selected["path"])
            manifest = training.verify_attempt(output, phase="formal", lock_hash=train_digest)
            source(output / "manifest.json", selected["manifest"])
            formal = json.loads((output / "formal_receipt.json").read_text())
            epoch = formal["selected_epoch"]
            if (manifest["arm"] != arm or manifest["seed"] != seed or formal["arm"] != arm or formal["seed"] != seed
                    or formal["epochs"] != 80 or formal["updates"] != 6400 or selected["selected_epoch"] != epoch
                    or selected["arm"] != arm or selected["seed"] != seed):
                raise ValueError("E4-v2 完整训练来源/selector identity 不一致")
            run = (output / formal["run_dir"]).resolve()
            if not run.is_relative_to(output.resolve()):
                raise ValueError("E4-v2 training run_dir 越界")
            cfg = OmegaConf.load(run / "config.yaml")
            training.validate_config(cfg, OmegaConf.create(lock["baselines"][str(seed)]), arm, lock["frequency"]["values_hz"],
                                     output_root=output / "training", device=str(cfg.training.device))
            if training.validate_history(pd.read_csv(run / "train_history.csv"), cfg) != epoch:
                raise ValueError("E4-v2 checkpoint 不是 validation 最早最小 Local RR")
            for filename in ("formal_receipt.json", "freeze_receipt.json", "train_rows.csv", "val_rows.csv"):
                source(output / filename)
            development = set(pd.read_csv(output / "train_rows.csv").samp_id) | set(pd.read_csv(output / "val_rows.csv").samp_id)
            if development & set(reference_metrics[seed].samp_id):
                raise ValueError("E4-v2 test 与 train/validation samp_id 交叉")
            candidate = {name: source(run / name, manifest["files"][str((run / name).relative_to(output))])
                         for name in ("config.yaml", "train_history.csv", "checkpoint_best_local_rr.pt")}
            entries.append({"arm": arm, "seed": seed, "selected_epoch": int(epoch), "training_attempt": str(output),
                "training_config": OmegaConf.to_container(cfg, resolve=True), "candidate": candidate,
                "development_samp_ids": sorted(map(int, development))})
    code_paths = sorted((root / "resp_train").rglob("*.py")) + [root / p for p in (SCRIPT_PATH, TEST_PATH, PROTOCOL_PATH)]
    result = {"protocol": PROTOCOL, "arms": list(ARMS), "seeds": list(SEEDS), "count": COUNT, "samp_id_count": SUBJECTS,
        "split": "test", "batch_size": 128, "amp_dtype": "bfloat16", "include_test_only": False,
        "contracts": {arm: aggregation_contract(arm) for arm in ARMS}, "training_lock_sha256": train_digest,
        "validation_summary": str(validation_summary), "validation_summary_manifest": identity(validation_summary / "manifest.json"),
        "entries": entries, "w0_sources": w0_sources, "source_files": files, "cache_root": str(CACHE_ROOT),
        "cache_manifest_sha256": CACHE_MANIFEST_SHA, "cache_files": cache_files,
        "dataset_index": {"path": cache["dataset_index"], "sha256": cache["dataset_index_sha256"]},
        "row_order_sha256": metadata["row_ids_sha256"], "preparation_git": git_state(root),
        "prepared_at": datetime.now(timezone.utc).isoformat(), "preparation_access": "frozen artifacts and cache manifest metadata",
        "code_files": {str(p.relative_to(root)): identity(p) for p in code_paths}}
    write_json(destination, result)
    return destination


def load_lock(root: Path = ROOT):
    path = root / LOCK_PATH
    lock = json.loads(path.read_text())
    if (lock["protocol"] != PROTOCOL or lock["arms"] != list(ARMS) or lock["seeds"] != list(SEEDS)
            or lock["split"] != "test" or lock["count"] != COUNT or lock["samp_id_count"] != SUBJECTS
            or lock["batch_size"] != 128 or lock["amp_dtype"] != "bfloat16" or lock["include_test_only"] is not False
            or lock["contracts"] != {arm: aggregation_contract(arm) for arm in ARMS}
            or lock["cache_root"] != str(CACHE_ROOT) or lock["cache_manifest_sha256"] != CACHE_MANIFEST_SHA
            or lock["cache_files"]["w_frequencies_hz.npy"]["sha256"] != FREQUENCY_FILE_SHA):
        raise ValueError("E4-v2 test lock 矩阵/科学合同漂移")
    expected = [(arm, seed) for arm in ARMS for seed in SEEDS]
    if [(entry["arm"], entry["seed"]) for entry in lock["entries"]] != expected:
        raise ValueError("E4-v2 test checkpoint allowlist 不完整")
    for entry in lock["entries"]:
        if not isinstance(entry["selected_epoch"], int) or not 1 <= entry["selected_epoch"] <= 80:
            raise ValueError("E4-v2 test selected epoch 非法")
    for relative, expected_identity in lock["code_files"].items():
        training.verify(root / relative, expected_identity)
    training.verify(Path(lock["validation_summary"]) / "manifest.json", lock["validation_summary_manifest"])
    if sha256_file(root / training.LOCK_PATH) != lock["training_lock_sha256"]:
        raise ValueError("E4-v2 test training lock identity 漂移")
    return lock, sha256_file(path)


@contextmanager
def attempt(parent, phase, digest, arm=None, seed=None):
    parent.mkdir(parents=True, exist_ok=True)
    output = parent / f"{phase}_{digest[:12]}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:12]}"
    output.mkdir(exist_ok=False)
    context = {"protocol": PROTOCOL, "phase": phase, "split": "test", "arm": arm, "seed": seed,
               "implementation_lock_sha256": digest, "command": sys.argv, "started_at": datetime.now(timezone.utc).isoformat()}
    write_json(output / "lifecycle_started.json", {**context, "status": "running"})
    print(f"E4-v2 test attempt: {output}", flush=True)
    try:
        yield output
        write_json(output / "lifecycle_completed.json", {**context, "status": "completed", "ended_at": datetime.now(timezone.utc).isoformat()})
        write_json(output / "manifest.json", {**context, "status": "completed", "files": {
            str(p.relative_to(output)): identity(p) for p in sorted(output.rglob("*")) if p.is_file()}})
        write_json(output / "freeze_receipt.json", {"protocol": PROTOCOL, "manifest": identity(output / "manifest.json")})
    except BaseException as exc:
        write_json(output / "lifecycle_failed.json", {**context, "status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
        raise


def verify_attempt(output, digest, phase):
    output = output.resolve()
    if (output / "lifecycle_failed.json").exists():
        raise ValueError("E4-v2 test attempt 已失败")
    freeze = json.loads((output / "freeze_receipt.json").read_text())
    training.verify(output / "manifest.json", freeze["manifest"])
    manifest = json.loads((output / "manifest.json").read_text())
    if (manifest["protocol"] != PROTOCOL or manifest["phase"] != phase or manifest["split"] != "test"
            or manifest["status"] != "completed" or manifest["implementation_lock_sha256"] != digest):
        raise ValueError("E4-v2 test attempt identity 不一致")
    required = {"lifecycle_completed.json", "summary_receipt.json", "seed_metrics.csv", "paired_seed_delta.csv", "four_arm_comparison.csv"} if phase == "summary" else {
        "lifecycle_completed.json", "evaluation_receipt.json", "metrics.csv", "metrics_summary.csv", "test_rows.csv", "resolved_config.yaml",
        "access_started.json", "access_receipt.json", "environment.json", "implementation_lock.json"}
    if not required.issubset(manifest["files"]):
        raise ValueError("E4-v2 test manifest 缺少产物")
    for relative, expected in manifest["files"].items():
        path = (output / relative).resolve()
        if not path.is_relative_to(output):
            raise ValueError("E4-v2 test manifest 路径越界")
        training.verify(path, expected)
    return manifest


def reject_completed(parent, digest, phase):
    for receipt in parent.glob("*/freeze_receipt.json"):
        manifest = json.loads((receipt.parent / "manifest.json").read_text())
        if manifest["implementation_lock_sha256"] == digest:
            verify_attempt(receipt.parent, digest, phase)
            raise FileExistsError(f"E4-v2 test 相同身份已完成: {receipt.parent}")


def evaluation_config(entry, device, cache_root):
    cfg = OmegaConf.load(entry["candidate"]["config.yaml"]["path"])
    if OmegaConf.to_container(cfg, resolve=True) != entry["training_config"]:
        raise ValueError("E4-v2 test 训练配置与 allowlist 不一致")
    if (cfg.protocol.name != training.PROTOCOL or cfg.model.variant != "crd_tf102_w"
            or OmegaConf.to_container(cfg.model.aggregation_v2, resolve=True) != aggregation_contract(entry["arm"])
            or cfg.training.seed != entry["seed"] or cfg.model.initialization_seed != entry["seed"]
            or cfg.loss.sync_weight != 1 or cfg.loss.effort_weight != .25 or cfg.data.test_split != "test"
            or cfg.data.max_test_windows is not None or cfg.data.test_sample_seed != 20260612
            or cfg.training.batch_size != 128 or not cfg.training.use_amp or cfg.training.amp_dtype != "bfloat16"):
        raise ValueError("E4-v2 test 科学配置错误")
    cfg.training.device, cfg.training.show_progress = device, False
    data_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    data_cfg.data.tf_research_test_cache_path = str(cache_root)
    return cfg, data_cfg


def run_evaluation(arm: str, seed: int, device="cuda:0"):
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError("E4-v2 test arm/seed 不在冻结矩阵")
    lock, digest = load_lock()
    entry = next(e for e in lock["entries"] if (e["arm"], e["seed"]) == (arm, seed))
    parent = ROOT / OUTPUT / "evaluation" / arm / f"seed_{seed}"
    with training.phase_guard(parent, digest):
        reject_completed(parent, digest, "evaluation")
        with attempt(parent, "evaluation", digest, arm, seed) as output:
            write_json(output / "environment.json", training.runtime_preflight(device))
            write_json(output / "implementation_lock.json", lock)
            write_json(output / "access_started.json", {"split": "test", "arm": arm, "seed": seed, "count": COUNT,
                "checkpoint": entry["candidate"]["checkpoint_best_local_rr.pt"], "cache_root": lock["cache_root"], "dataset_index": lock["dataset_index"]})
            for path, expected in lock["source_files"].items():
                training.verify(Path(path), expected)
            cache_root = Path(lock["cache_root"])
            if sha256_file(cache_root / "cache_manifest.json") != lock["cache_manifest_sha256"]:
                raise ValueError("E4-v2 test cache manifest 漂移")
            for filename, expected in lock["cache_files"].items():
                training.verify(cache_root / filename, expected)
            if sha256_file(Path(lock["dataset_index"]["path"])) != lock["dataset_index"]["sha256"]:
                raise ValueError("E4-v2 test dataset index 漂移")
            cfg, data_cfg = evaluation_config(entry, device, lock["cache_root"])
            OmegaConf.save(data_cfg, output / "resolved_config.yaml")
            checkpoint = torch.load(entry["candidate"]["checkpoint_best_local_rr.pt"]["path"], map_location="cpu", weights_only=False)
            if checkpoint.get("config") != entry["training_config"] or checkpoint["epoch"] != entry["selected_epoch"]:
                raise ValueError("E4-v2 test checkpoint config/epoch 漂移")
            _validate_checkpoint_config(checkpoint["config"], cfg)
            training.finite_tree(checkpoint["model_state_dict"])
            model = build_model(cfg); model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            reference = pd.read_csv(lock["w0_sources"][str(seed)]["files"]["research_test_metrics.csv"]["path"])
            audited = read_research_v2_index(cfg.data.dataset_root, cfg.data.index_csv, cfg)
            rows = filter_index(audited, cfg, split="test", max_windows=None, sample_strategy=str(cfg.data.test_sample_strategy), sample_seed=20260612)
            check_rows(rows, reference, row_hash=lock["row_order_sha256"])
            if set(rows.samp_id) & set(entry["development_samp_ids"]):
                raise ValueError("E4-v2 test 与开发 samp_id 交叉")
            rows.to_csv(output / "test_rows.csv", index=False)
            data = build_window_data(data_cfg, split="test", max_windows=None, sample_strategy=str(cfg.data.test_sample_strategy), sample_seed=20260612, shuffle=False, audited=audited)
            check_rows(data.rows, rows, row_hash=lock["row_order_sha256"])
            if len(data.dataset) != COUNT:
                raise ValueError("E4-v2 test dataset 样本数不完整")
            predictions = collect_predictions(model, guarded_batches(data.loader, rows), device=device, max_windows=COUNT, use_amp=True)
            metrics = evaluate_task_predictions(predictions, cfg, include_test_only=False, method=arm)
            summary = check_metrics(metrics, reference)
            quality = quality_flags(metrics)
            metrics.insert(0, "arm", arm); metrics.insert(0, "seed", seed)
            for name, value in {"arm": arm, "seed": seed, "split": "test", "selected_epoch": entry["selected_epoch"], "quality_acceptance_passed": quality["quality_acceptance_passed"]}.items():
                summary.insert(0, name, value)
            metrics.to_csv(output / "metrics.csv", index=False); summary.to_csv(output / "metrics_summary.csv", index=False)
            write_json(output / "access_receipt.json", {"arm": arm, "seed": seed, "split": "test", "rows": COUNT, "samp_ids": SUBJECTS,
                "row_order_sha256": array_hash(rows.dataset_row_id.to_numpy()), "cache_identity": CACHE_IDENTITY,
                "cache_files_verified": lock["cache_files"], "checkpoint": entry["candidate"]["checkpoint_best_local_rr.pt"], "test_sample_seed": 20260612})
            write_json(output / "evaluation_receipt.json", {"protocol": PROTOCOL, "arm": arm, "seed": seed, "split": "test",
                "selected_epoch": entry["selected_epoch"], "checkpoint_sha256": entry["candidate"]["checkpoint_best_local_rr.pt"]["sha256"],
                "rows": COUNT, "samp_ids": SUBJECTS, "primary_finite": True, **quality,
                "target_eligibility_matches_w0": True, "row_order_sha256": lock["row_order_sha256"]})
    return output


def summarize(runs: list[Path]):
    lock, digest = load_lock()
    if len(runs) != 12 or len({p.resolve() for p in runs}) != 12:
        raise ValueError("E4-v2 test 汇总要求12个不同完成 attempt")
    parent = ROOT / OUTPUT / "summary"
    with training.phase_guard(parent, digest):
        reject_completed(parent, digest, "summary")
        frames, sources = [], {}
        for output in runs:
            manifest = verify_attempt(output, digest, "evaluation")
            arm, seed = manifest["arm"], manifest["seed"]
            key = f"{arm}/{seed}"
            if arm not in ARMS or seed not in SEEDS or key in sources:
                raise ValueError("E4-v2 test arm/seed 重复或越界")
            entry = next(e for e in lock["entries"] if (e["arm"], e["seed"]) == (arm, seed))
            receipt = json.loads((output / "evaluation_receipt.json").read_text())
            if (receipt["arm"] != arm or receipt["seed"] != seed or receipt["split"] != "test" or receipt["rows"] != COUNT
                    or receipt["selected_epoch"] != entry["selected_epoch"]
                    or receipt["checkpoint_sha256"] != entry["candidate"]["checkpoint_best_local_rr.pt"]["sha256"]):
                raise ValueError("E4-v2 test 回执与 checkpoint allowlist 不一致")
            full = lock["w0_sources"][str(seed)]["files"]
            for expected in full.values():
                training.verify(Path(expected["path"]), expected)
            reference = pd.read_csv(full["research_test_metrics.csv"]["path"])
            metrics = pd.read_csv(output / "metrics.csv")
            if not metrics.arm.eq(arm).all() or not metrics.seed.eq(seed).all():
                raise ValueError("E4-v2 test metrics arm/seed 错误")
            check_rows(metrics, pd.read_csv(output / "test_rows.csv"), row_hash=lock["row_order_sha256"])
            expected = check_metrics(metrics, reference)
            quality = quality_flags(metrics)
            if any(receipt.get(k) != v for k, v in quality.items()):
                raise ValueError("E4-v2 test 质量回执不一致")
            saved = pd.read_csv(output / "metrics_summary.csv")
            validate_summary(saved, expected)
            if (saved.iloc[0].arm != arm or saved.iloc[0].seed != seed or saved.iloc[0].split != "test"
                    or saved.iloc[0].selected_epoch != entry["selected_epoch"]
                    or bool(saved.iloc[0].quality_acceptance_passed) != quality["quality_acceptance_passed"]):
                raise ValueError("E4-v2 test summary identity 错误")
            saved.insert(0, "source_sha256", sha256_file(output / "metrics.csv")); frames.append(saved)
            sources[key] = {"arm": arm, "seed": seed, "path": str(output.resolve()), "manifest": identity(output / "manifest.json"), "selected_epoch": entry["selected_epoch"]}
        for seed in SEEDS:
            source = lock["w0_sources"][str(seed)]
            reference = pd.read_csv(source["files"]["research_test_metrics.csv"]["path"])
            expected = check_metrics(reference, reference)
            full = pd.read_csv(source["files"]["research_test_metrics_summary.csv"]["path"])
            validate_summary(full, expected)
            for key, value in {"arm": "W0_FULL", "seed": seed, "split": "test", "selected_epoch": source["selected_epoch"],
                "source_sha256": source["files"]["research_test_metrics.csv"]["sha256"], "quality_acceptance_passed": quality_flags(reference)["quality_acceptance_passed"]}.items():
                full.insert(0, key, value)
            frames.append(full)
        combined = pd.concat(frames, ignore_index=True)
        paired, comparison = training.paired_tables(combined, split="test")
        with attempt(parent, "summary", digest) as output:
            combined.to_csv(output / "seed_metrics.csv", index=False)
            paired.to_csv(output / "paired_seed_delta.csv", index=False, na_rep="NA")
            comparison.to_csv(output / "four_arm_comparison.csv", index=False, na_rep="NA")
            write_json(output / "summary_receipt.json", {"protocol": PROTOCOL, "split": "test", "arms": list(ARMS), "seeds": list(SEEDS),
                "rows_per_run": COUNT, "new_metric_rows": 12 * COUNT, "source_runs": sources, "w0_sources": lock["w0_sources"],
                "delta_definition": "error=(candidate-W0)/W0*100; PCC=W0-candidate; positive=worse"})
    return output


def validate_summary(saved, expected):
    if len(saved) != 1:
        raise ValueError("E4-v2 test summary 必须一行")
    for key in PRIMARY:
        if not np.isclose(saved.iloc[0][key + "_mean"], expected.iloc[0][key + "_mean"], rtol=0, atol=1e-12) or saved.iloc[0][key + "_n"] != COUNT:
            raise ValueError("E4-v2 test summary 数值/分母不一致")


def completed_runs():
    _, digest = load_lock()
    result = []
    for arm in ARMS:
        for seed in SEEDS:
            parent = ROOT / OUTPUT / "evaluation" / arm / f"seed_{seed}"
            matches = []
            for receipt in parent.glob("*/freeze_receipt.json"):
                manifest = json.loads((receipt.parent / "manifest.json").read_text())
                if manifest.get("implementation_lock_sha256") == digest:
                    if manifest.get("arm") != arm or manifest.get("seed") != seed or manifest.get("phase") != "evaluation":
                        raise ValueError("E4-v2 test 完成目录与 manifest cell 不一致")
                    matches.append(receipt.parent.resolve())
            if len(matches) != 1:
                raise ValueError(f"E4-v2 test {arm}/{seed} 需要唯一成功 attempt，实际 {len(matches)}")
            result.extend(matches)
    return result
