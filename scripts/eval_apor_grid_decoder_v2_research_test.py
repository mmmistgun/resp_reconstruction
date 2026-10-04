"""APOR v2固定十二单元research-test：复用六个参照，新增六次评价。"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from resp_train.crd.experiment import _validate_checkpoint_config
from resp_train.crd.tf_v1_research_test_data import (
    FROZEN_RESEARCH_TEST_CACHE_IDENTITY,
    FROZEN_RESEARCH_TEST_CACHE_MANIFEST_SHA256,
    FROZEN_RESEARCH_TEST_CACHE_ROOT,
)
from resp_train.data.factory import build_window_data
from resp_train.data.index import filter_index
from resp_train.data.research_v2 import read_research_v2_index
from resp_train.engine import collect_predictions
from resp_train.metrics.task import evaluate_task_predictions, summarize_task_metrics
from scripts import apor_grid_decoder_v2_runtime as training
from scripts import eval_patch_apor_v1_research_test as old_test
from resp_train.paper_evidence import w0_structural_factorial_v1 as sf
from scripts.apor_grid_decoder_v2_model import ARMS, TRAIN_ARMS, REFERENCE_ARMS, ARM_SPECS, SEEDS, build_model, model_contract
from resp_train.paper_evidence.w0_structural_factorial_v1_engineering import finite_tree
from resp_train.paper_evidence.w0_structural_factorial_v1_test import (
    IDENTITY_COLUMNS, TARGET_COLUMNS, check_test_metrics, check_test_rows, quality_flags,
)

PROTOCOL = "apor-grid-decoder-v2-research-test-20260930"
COUNT, SUBJECTS, SAMPLE_SEED = 2310, 8, 20260612
ROW_HASH = "184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8"
SESSION = ROOT / "runs/apor_grid_decoder_v2/session_20260930T093614Z_511f2ff06f0f"
SESSION_SHA = "0c54b200006030b5a4c692f5a3b386d8e9caf551a1b3bf0415768ed334340386"
OLD_TEST_ROOT = ROOT / "runs/patch_apor_v1/research_test/allowlist_20260930T071732Z_9b85989bdc0b"
VIEW_COUNTS = {"full": (2310, 8), "exclude670": (2231, 7), "subject670": (79, 1)}
OUTPUT = ROOT / "runs/apor_grid_decoder_v2/research_test"
PROTOCOL_PATH = ROOT / "docs/experiments/apor_grid_decoder_v2_research_test_protocol_20260930.md"
TEST_PATH = ROOT / "tests/test_apor_grid_decoder_v2_research_test.py"
read_json, write_json = training.read_json, training.write_json


def code_files():
    # 实现放在独立入口中，保持训练session受管文件集合及其字节身份不变。
    return sorted(set(training.source_files() + old_test.code_files() + [Path(__file__).resolve(), PROTOCOL_PATH, TEST_PATH]))


def frozen_test_references(frozen):
    original = old_test.load_allowlist(OLD_TEST_ROOT)
    records = []
    for arm in REFERENCE_ARMS:
        for seed in SEEDS:
            source = frozen["references"][f"{arm}/{seed}"]
            entry = old_test.entry_for(original, source["source_arm"], seed)
            if entry["checkpoint"] != source["checkpoint"] or entry["selected_epoch"] != source["selected_epoch"]:
                raise ValueError("复用test与v2 validation参照的checkpoint不一致")
            attempt = old_test.completed(OLD_TEST_ROOT, OLD_TEST_ROOT / "evaluation" / source["source_arm"] / f"seed_{seed}",
                                         "evaluation", arm=source["source_arm"], seed=seed)
            if attempt is None:
                raise ValueError("参照test未完成，不能自动重新推理")
            receipt = read_json(attempt / "evaluation_receipt.json")
            if receipt["checkpoint_sha256"] != source["checkpoint"]["sha256"] or receipt["row_order_sha256"] != ROW_HASH:
                raise ValueError("参照test checkpoint/样本身份不一致")
            records.append({"arm": arm, "seed": seed, "source_arm": source["source_arm"],
                            "path": str(attempt), "manifest": sf.identity(attempt / "manifest.json"),
                            "checkpoint": source["checkpoint"], "selected_epoch": source["selected_epoch"]})
    return records


def verify_reference(ref):
    path = Path(ref["path"])
    sf.verify_identity(path / "manifest.json", ref["manifest"])
    sf.verify_identity(path / "manifest.json", read_json(path / "freeze_receipt.json"))
    manifest = read_json(path / "manifest.json")
    if (manifest.get("status") != "completed" or manifest.get("arm") != ref["source_arm"]
            or manifest.get("seed") != ref["seed"]):
        raise ValueError("参照test manifest身份错误")
    for relative, identity in manifest["files"].items():
        file = (path / relative).resolve()
        if not file.is_relative_to(path.resolve()):
            raise ValueError("参照路径越界")
        sf.verify_identity(file, identity)
    return path


def selected_epoch(history: pd.DataFrame, receipt: dict) -> int:
    values = history.val_local_rr_mae.to_numpy(float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("validation selector缺少有限完整轨迹")
    epoch = int(history.iloc[int(np.argmin(values))].epoch)
    if epoch != receipt["best_epoch"] or len(history) != receipt["completed_epochs"]:
        raise ValueError("checkpoint选择与完整validation history不一致")
    return epoch


def prepare_allowlist(session: Path = SESSION) -> Path:
    session = session.resolve()
    frozen = training.load_session(session)
    session_hash = sf.sha256_file(session / "session.json")
    if session_hash != SESSION_SHA:
        raise ValueError("v2 test只允许已完成的固定训练session")
    for prior in sorted(OUTPUT.glob("allowlist_*/allowlist_receipt.json")):
        payload = read_json(prior.parent / "allowlist.json")
        if payload.get("training_session_sha256") == session_hash and payload.get("protocol") == PROTOCOL:
            load_allowlist(prior.parent)
            return prior.parent
    summary = training.completed(session / "summary", session, "summary")
    if summary is None:
        raise RuntimeError("必须先完成12-cell validation汇总")
    decision = read_json(summary / "validation_decision.json")
    if decision["candidate"] != "N0" or decision["test_used"] is not False:
        raise ValueError("validation冻结选择与预期不一致")
    references = frozen_test_references(frozen)
    cache_manifest_path = FROZEN_RESEARCH_TEST_CACHE_ROOT / "cache_manifest.json"
    if sf.sha256_file(cache_manifest_path) != FROZEN_RESEARCH_TEST_CACHE_MANIFEST_SHA256:
        raise ValueError("test cache manifest身份漂移")
    cache_manifest = read_json(cache_manifest_path)
    if cache_manifest["cache_identity_sha256"] != FROZEN_RESEARCH_TEST_CACHE_IDENTITY:
        raise ValueError("test cache identity错误")
    entries = []
    row_file_identities = None
    development_ids = None
    for cell in training.plan():
        arm, seed = cell["arm"], cell["seed"]
        attempt_path = training.completed(session / "formal" / arm / f"seed_{seed}", session, "formal", arm=arm, seed=seed)
        if attempt_path is None:
            raise RuntimeError(f"formal未完成: {arm}/{seed}")
        receipt = read_json(attempt_path / "result.json")
        run = (attempt_path / receipt["run_dir"]).resolve()
        if not run.is_relative_to(attempt_path):
            raise ValueError("formal run路径越界")
        cfg = OmegaConf.load(run / "config.yaml")
        expected = training.config(arm, seed, attempt_path / "training", str(cfg.training.device))
        if OmegaConf.to_container(cfg, resolve=True) != OmegaConf.to_container(expected, resolve=True):
            raise ValueError("来源训练配置漂移")
        history = pd.read_csv(run / "train_history.csv")
        epoch = selected_epoch(history, receipt)
        checkpoint_path = run / "checkpoint_best_local_rr.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["epoch"] != epoch or checkpoint["config"] != OmegaConf.to_container(cfg, resolve=True):
            raise ValueError("best checkpoint epoch/config漂移")
        finite_tree(checkpoint["model_state_dict"], label="allowlist_checkpoint")
        current_rows = {split: sf.identity(attempt_path / f"{split}_rows.csv") for split in ("train", "val")}
        if row_file_identities is None:
            row_file_identities = current_rows
            development_ids = sorted(set().union(*[
                set(pd.read_csv(attempt_path / f"{split}_rows.csv", usecols=["samp_id"]).samp_id.astype(int))
                for split in ("train", "val")]))
        elif row_file_identities != current_rows:
            raise ValueError("不同cell的development样本身份不一致")
        entries.append({"arm": arm, "seed": seed, "selected_epoch": epoch,
                        "formal_attempt": str(attempt_path), "formal_manifest": sf.identity(attempt_path / "manifest.json"),
                        "checkpoint": {"path": str(checkpoint_path), **sf.identity(checkpoint_path)},
                        "config": {"path": str(run / "config.yaml"), **sf.identity(run / "config.yaml")},
                        "training_config": OmegaConf.to_container(cfg, resolve=True),
                        "training_environment": read_json(attempt_path / "environment.json")})
        del checkpoint
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = OUTPUT / f"allowlist_{stamp}_{uuid4().hex[:12]}"
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for path in code_files():
        relative = path.relative_to(ROOT)
        dest = output / "source_snapshot" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        files[str(relative)] = sf.identity(dest)
    source = read_json(ROOT / training.P2_LOCK_PATH)
    payload = {"schema_version": 1, "protocol": PROTOCOL, "training_session": str(session),
               "training_session_sha256": session_hash, "training_dependencies": frozen["dependencies"],
               "validation_summary": {"path": str(summary), **sf.identity(summary / "manifest.json")},
               "split": "test", "count": COUNT, "samp_ids": SUBJECTS, "row_order_sha256": ROW_HASH,
               "sample_seed": SAMPLE_SEED, "batch_size": 128, "amp_dtype": "bfloat16", "include_test_only": True,
               "evidence_role": "reused research/development evidence", "entries": entries,
               "references": references, "views": {k: list(v) for k,v in VIEW_COUNTS.items()},
               "validation_decision": {"path": str(summary / "validation_decision.json"), **sf.identity(summary / "validation_decision.json")},
               "development_samp_ids": development_ids, "dataset_index": source["dataset_index"],
               "cache": {"root": str(FROZEN_RESEARCH_TEST_CACHE_ROOT), "identity": FROZEN_RESEARCH_TEST_CACHE_IDENTITY,
                         "manifest": sf.identity(cache_manifest_path),
                         "files": {name: cache_manifest["files"][name] for name in ("test_row_ids.npy", "test_w.npy")}},
               "code_files": files, "git": sf.git_state(ROOT), "prepared_at": datetime.now(timezone.utc).isoformat(),
               "test_array_read": False, "model_inference_used": False}
    write_json(output / "allowlist.json", payload)
    write_json(output / "allowlist_receipt.json", sf.identity(output / "allowlist.json"))
    return output


def validate_contract(lock):
    expected = [(cell["arm"], cell["seed"]) for cell in training.plan()]
    observed = [(entry["arm"], entry["seed"]) for entry in lock["entries"]]
    if (lock["protocol"] != PROTOCOL or observed != expected or lock["split"] != "test"
            or lock["count"] != COUNT or lock["samp_ids"] != SUBJECTS or lock["row_order_sha256"] != ROW_HASH
            or lock["sample_seed"] != SAMPLE_SEED or lock["batch_size"] != 128 or lock["amp_dtype"] != "bfloat16"
            or lock["include_test_only"] is not True or lock["test_array_read"] is not False
            or lock.get("views") != {k: list(v) for k,v in VIEW_COUNTS.items()}
            or len(lock.get("references", [])) != 6
            or {(r["arm"],r["seed"]) for r in lock["references"]} != {(a,s) for a in REFERENCE_ARMS for s in SEEDS}):
        raise ValueError("固定六个新checkpoint与六个参照的test合同漂移")


def load_allowlist(path: Path):
    sf.verify_identity(path / "allowlist.json", read_json(path / "allowlist_receipt.json"))
    lock = read_json(path / "allowlist.json")
    validate_contract(lock)
    if lock["training_dependencies"] != training.dependencies():
        raise ValueError("test依赖环境与训练不同")
    session = Path(lock["training_session"])
    if sf.sha256_file(session / "session.json") != lock["training_session_sha256"]:
        raise ValueError("训练session来源漂移")
    sf.verify_identity(Path(lock["validation_summary"]["path"]) / "manifest.json", lock["validation_summary"])
    sf.verify_identity(Path(lock["validation_decision"]["path"]), lock["validation_decision"])
    for ref in lock["references"]:
        sf.verify_identity(Path(ref["path"]) / "manifest.json", ref["manifest"])
    for relative, expected in lock["code_files"].items():
        sf.verify_identity(ROOT / relative, expected)
        sf.verify_identity(path / "source_snapshot" / relative, expected)
    return lock


@contextmanager
def evidence_attempt(root: Path, parent: Path, phase: str, **fields):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = parent / f"attempt_{stamp}_{uuid4().hex[:12]}"
    output.mkdir(parents=True, exist_ok=False)
    context = {"protocol": PROTOCOL, "phase": phase, "allowlist_sha256": sf.sha256_file(root / "allowlist.json"), **fields}
    write_json(output / "started.json", context)
    try:
        yield output
        files = {str(p.relative_to(output)): sf.identity(p) for p in sorted(output.rglob("*")) if p.is_file()}
        write_json(output / "manifest.json", {**context, "status": "completed", "files": files})
        write_json(output / "freeze_receipt.json", sf.identity(output / "manifest.json"))
    except BaseException:
        write_json(output / "failed.json", {**context, "traceback": traceback.format_exc()})
        raise


def completed(root: Path, parent: Path, phase: str, retry_failed=False, **fields):
    paths = sorted(parent.glob("attempt_*"))
    done = [p for p in paths if (p / "freeze_receipt.json").is_file()]
    if len(done) > 1:
        raise ValueError("同一test cell有多个成功产物")
    if done:
        path = done[0]
        sf.verify_identity(path / "manifest.json", read_json(path / "freeze_receipt.json"))
        manifest = read_json(path / "manifest.json")
        expected = {"protocol": PROTOCOL, "phase": phase, "allowlist_sha256": sf.sha256_file(root / "allowlist.json"),
                    "status": "completed", **fields}
        if any(manifest.get(k) != v for k, v in expected.items()):
            raise ValueError("research-test产物身份错误")
        for relative, identity in manifest["files"].items():
            file = (path / relative).resolve()
            if not file.is_relative_to(path.resolve()):
                raise ValueError("test产物路径越界")
            sf.verify_identity(file, identity)
        return path
    if paths and not retry_failed:
        raise RuntimeError(f"已有失败/中断attempt，检查后用--retry-failed重试: {parent}")
    return None


def entry_for(lock, arm, seed):
    if arm not in TRAIN_ARMS or seed not in SEEDS:
        raise ValueError("只允许新增N1/D1评价，N0/D0复用冻结test")
    return next(entry for entry in lock["entries"] if entry["arm"] == arm and entry["seed"] == seed)


def evaluation_config(entry, device):
    sf.verify_identity(Path(entry["config"]["path"]), entry["config"])
    cfg = OmegaConf.load(entry["config"]["path"])
    if OmegaConf.to_container(cfg, resolve=True) != entry["training_config"]:
        raise ValueError("训练配置与allowlist不一致")
    if (cfg.model.apor_grid_decoder_v2.arm != entry["arm"] or int(cfg.training.seed) != entry["seed"]
            or cfg.data.test_split != "test" or cfg.data.max_test_windows is not None
            or cfg.data.test_sample_seed != SAMPLE_SEED or cfg.training.batch_size != 128
            or not cfg.training.use_amp or cfg.training.amp_dtype != "bfloat16"):
        raise ValueError("test推理合同不符")
    cfg.training.device = str(device)
    cfg.training.show_progress = False
    data_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    if ARM_SPECS[entry["arm"]].condition:
        data_cfg.data.tf_research_test_cache_path = str(FROZEN_RESEARCH_TEST_CACHE_ROOT)
    return cfg, data_cfg


def guarded_batches(loader, rows, *, condition):
    offset = 0
    for batch in loader:
        count = len(batch["x"])
        expected = rows.iloc[offset:offset + count]
        features = batch.get("tf") or {}
        if count <= 0 or len(expected) != count or set(features) != ({"w"} if condition else set()):
            raise ValueError("test batch数量或条件keys错误")
        for key in IDENTITY_COLUMNS:
            raw = batch["meta"][key]
            actual = raw.cpu().numpy() if torch.is_tensor(raw) else np.asarray(raw)
            if not np.array_equal(actual, expected[key].to_numpy()):
                raise ValueError(f"test batch样本顺序错误: {key}")
        tensors = [batch["x"], batch["target"]]
        if any(tuple(value.shape) != (count, 1, 18000) for value in tensors):
            raise ValueError("test input/target shape错误")
        if condition:
            if tuple(features["w"].shape) != (count, 97, 360):
                raise ValueError("test W shape错误")
            tensors.append(features["w"])
        if any(not bool(torch.isfinite(value).all()) for value in tensors):
            raise FloatingPointError("test input/target/W含NaN/Inf")
        offset += count
        yield batch
    if offset != len(rows):
        raise ValueError("test loader未覆盖完整样本")


def evaluate(root: Path, arm: str, seed: int, device: str, retry_failed=False):
    lock = load_allowlist(root)
    entry = entry_for(lock, arm, seed)
    parent = root / "evaluation" / arm / f"seed_{seed}"
    with training.file_mutex(root / f".cell_{arm}_{seed}.lock"):
        prior = completed(root, parent, "evaluation", retry_failed, arm=arm, seed=seed)
        if prior:
            print(f"复用test {arm}/{seed}: {prior}", flush=True)
            return prior
        with evidence_attempt(root, parent, "evaluation", arm=arm, seed=seed) as output:
            runtime = training.cuda_environment(device)
            if runtime["device_name"] != entry["training_environment"]["device_name"]:
                raise ValueError("test GPU型号与训练不同")
            write_json(output / "environment.json", runtime)
            write_json(output / "access_started.json", {"split": "test", "rows": COUNT,
                       "checkpoint": entry["checkpoint"], "cache_read": ARM_SPECS[arm].condition,
                       "evidence_role": lock["evidence_role"]})
            sf.verify_identity(Path(entry["checkpoint"]["path"]), entry["checkpoint"])
            if sf.sha256_file(Path(lock["dataset_index"]["path"])) != lock["dataset_index"]["sha256"]:
                raise ValueError("test dataset index身份漂移")
            cfg, data_cfg = evaluation_config(entry, device)
            OmegaConf.save(data_cfg, output / "resolved_config.yaml")
            checkpoint = torch.load(entry["checkpoint"]["path"], map_location="cpu", weights_only=False)
            if checkpoint["epoch"] != entry["selected_epoch"] or checkpoint["config"] != entry["training_config"]:
                raise ValueError("test checkpoint来源错误")
            _validate_checkpoint_config(checkpoint["config"], cfg)
            finite_tree(checkpoint["model_state_dict"], label="test_model_state")
            model = build_model(cfg)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            del checkpoint
            audited = read_research_v2_index(cfg.data.dataset_root, cfg.data.index_csv, cfg)
            rows = filter_index(audited, cfg, split="test", max_windows=None,
                                sample_strategy=cfg.data.test_sample_strategy, sample_seed=SAMPLE_SEED)
            check_test_rows(rows)
            if set(rows.samp_id.astype(int)) & set(lock["development_samp_ids"]):
                raise ValueError("test与development受试者交叉")
            rows.to_csv(output / "test_rows.csv", index=False)
            data = build_window_data(data_cfg, split="test", max_windows=None,
                                     sample_strategy=cfg.data.test_sample_strategy, sample_seed=SAMPLE_SEED,
                                     shuffle=False, audited=audited)
            check_test_rows(data.rows, rows)
            if len(data.dataset) != COUNT:
                raise ValueError("test dataset窗口数漂移")
            print(f"开始test {arm}/{seed} epoch={entry['selected_epoch']} {device}", flush=True)
            predictions = collect_predictions(model, guarded_batches(data.loader, rows, condition=ARM_SPECS[arm].condition),
                                               device=device, max_windows=COUNT, use_amp=True)
            for key in ("r_tho_hat", "tho_ref"):
                if predictions[key].shape[0] != COUNT or not np.isfinite(predictions[key]).all():
                    raise FloatingPointError("test prediction/target不完整或非有限")
            metrics = evaluate_task_predictions(predictions, cfg, include_test_only=True, method=arm)
            summary = check_test_metrics(metrics, rows)
            metrics.insert(0, "arm", arm)
            metrics.insert(1, "seed", seed)
            metrics.to_csv(output / "metrics.csv", index=False)
            summary.to_csv(output / "metrics_summary.csv", index=False)
            write_json(output / "evaluation_receipt.json", {"arm": arm, "seed": seed, "selected_epoch": entry["selected_epoch"],
                       "checkpoint_sha256": entry["checkpoint"]["sha256"], "rows": COUNT, "samp_ids": SUBJECTS,
                       "row_order_sha256": ROW_HASH, "evidence_role": lock["evidence_role"], **quality_flags(metrics)})
        print(f"完成test {arm}/{seed}: {output}", flush=True)
    del model, data, predictions, metrics
    gc.collect()
    torch.cuda.empty_cache()
    return output


def summary_tables(frame):
    pairs = set(frame[["arm", "seed"]].drop_duplicates().itertuples(index=False, name=None))
    if pairs != {(a, s) for a in ARMS for s in SEEDS}:
        raise ValueError("test汇总要求完整12-cell")
    first = None
    for (arm, seed), group in frame.groupby(["arm", "seed"], sort=False):
        group = group.reset_index(drop=True)
        check_test_metrics(group, group[list(IDENTITY_COLUMNS)])
        target_identity = group[[*IDENTITY_COLUMNS, *TARGET_COLUMNS]]
        if first is None:
            first = target_identity
        elif not first.equals(target_identity):
            raise ValueError("test跨cell样本/target资格不一致")
    outputs = {}
    for view, (count, subjects_count) in VIEW_COUNTS.items():
        subset = frame if view == "full" else frame.loc[frame.samp_id.ne(670) if view == "exclude670" else frame.samp_id.eq(670)]
        records, secondary = [], []
        for (arm, seed), group in subset.groupby(["arm", "seed"], sort=False):
            if len(group) != count or group.samp_id.nunique() != subjects_count:
                raise ValueError(f"{view}分母与固定方案不一致")
            for metric in sf.PRIMARY:
                values = group.loc[sf._metric_mask(group, metric), metric].to_numpy(float)
                if len(values) != count or not np.isfinite(values).all():
                    raise FloatingPointError("test子集主指标不完整")
                records.append({"arm": arm, "seed": seed, "metric": metric, "mean": float(values.mean()), "n": len(values)})
            native = summarize_task_metrics(group)
            native.insert(0, "seed", seed)
            native.insert(0, "arm", arm)
            secondary.append(native)
        if len(records) != len(ARMS)*len(SEEDS)*len(sf.PRIMARY):
            raise ValueError("test子集缺少科学单元")
        per_seed = pd.DataFrame(records)
        across = per_seed.groupby(["arm", "metric"], sort=False)["mean"].agg(seed_mean="mean", seed_sd="std").reset_index()
        indexed = per_seed.set_index(["arm", "seed", "metric"])["mean"]
        deltas = []
        for arm, reference in training.COMPARISONS:
            for seed in SEEDS:
                for metric in sf.PRIMARY:
                    delta = indexed[arm, seed, metric] - indexed[reference, seed, metric]
                    deltas.append({"arm": arm, "reference": reference, "seed": seed, "metric": metric,
                                   "delta": delta, "improvement": delta if metric == sf.PCC else -delta})
        paired = pd.DataFrame(deltas)
        paired_summary = paired.groupby(["arm", "reference", "metric"], sort=False)["improvement"].agg(
            mean="mean", seed_sd="std", improved_seeds=lambda x: int((x > 0).sum())).reset_index()
        subjects = sf.subject_stratified_metrics(subset)
        macro = subjects.groupby(["arm", "seed", "metric"], sort=False)["mean"].mean().reset_index()
        interactions = []
        for scope, table in (("window", per_seed), ("subject_macro", macro)):
            grid = table.set_index(["arm", "seed", "metric"])["mean"]
            for seed in SEEDS:
                for metric in sf.PRIMARY:
                    get = lambda arm: float(grid.loc[(arm, seed, metric)])
                    interactions.append({"scope": scope, "seed": seed, "metric": metric,
                                         "interaction": (get("D1")-get("D0"))-(get("N1")-get("N0"))})
        inter = pd.DataFrame(interactions)
        tables = {"per_seed": per_seed, "across_seed": across, "paired_delta": paired, "paired_summary": paired_summary,
                  "per_subject": subjects, "subject_macro": macro, "local_rr_tail": sf.local_rr_tail_summary(subset),
                  "denominators": sf.metric_denominators(subset), "native_metrics_per_seed": pd.concat(secondary, ignore_index=True),
                  "interaction_by_seed": inter,
                  "interaction_summary": inter.groupby(["scope", "metric"]).interaction.agg(mean="mean", seed_sd="std").reset_index()}
        for name, table in tables.items():
            table.insert(0, "view", view)
            outputs.setdefault(name, []).append(table)
    return {name: pd.concat(tables, ignore_index=True) for name, tables in outputs.items()}


def summarize(root: Path, retry_failed=False):
    lock = load_allowlist(root)
    prior = completed(root, root / "summary", "summary", retry_failed)
    if prior:
        return prior
    frames, sources = [], []
    for ref in lock["references"]:
        p = verify_reference(ref)
        frame = pd.read_csv(p / "metrics.csv")
        if not frame.arm.eq(ref["source_arm"]).all() or not frame.seed.eq(ref["seed"]).all():
            raise ValueError("参照test标签不一致")
        frame["arm"] = ref["arm"]
        frame["method"] = ref["arm"]
        frames.append(frame)
        sources.append({"arm": ref["arm"], "seed": ref["seed"], "reused": True, "path": str(p), **ref["manifest"]})
    for entry in lock["entries"]:
        arm, seed = entry["arm"], entry["seed"]
        p = completed(root, root / "evaluation" / arm / f"seed_{seed}", "evaluation", arm=arm, seed=seed)
        if p is None:
            raise RuntimeError("test矩阵尚未全部完成")
        receipt = read_json(p / "evaluation_receipt.json")
        if receipt["checkpoint_sha256"] != entry["checkpoint"]["sha256"] or receipt["selected_epoch"] != entry["selected_epoch"]:
            raise ValueError("test结果与allowlist checkpoint不一致")
        frame = pd.read_csv(p / "metrics.csv")
        if not frame.arm.eq(arm).all() or not frame.seed.eq(seed).all():
            raise ValueError("test cell标签不符")
        frames.append(frame)
        sources.append({"path": str(p), **sf.identity(p / "manifest.json")})
    tables = summary_tables(pd.concat(frames, ignore_index=True))
    with evidence_attempt(root, root / "summary", "summary") as output:
        write_json(output / "sources.json", sources)
        for name, table in tables.items():
            table.to_csv(output / f"{name}.csv", index=False)
    return output


def status(root):
    lock = load_allowlist(root)
    cells = []
    for entry in lock["entries"]:
        parent = root / "evaluation" / entry["arm"] / f"seed_{entry['seed']}"
        paths = list(parent.glob("attempt_*"))
        success = [p for p in paths if (p / "freeze_receipt.json").is_file()]
        state = "completed" if success else "failed" if paths and all((p / "failed.json").is_file() for p in paths) else "running_or_interrupted" if paths else "pending"
        cells.append({"arm": entry["arm"], "seed": entry["seed"], "status": state})
    return {"cells": cells, "reused_reference_cells": 6,
            "counts": {s: sum(c["status"] == s for c in cells) for s in ("completed", "failed", "running_or_interrupted", "pending")}}


def run_shard(root, device, index, count, retry_failed=False):
    load_allowlist(root)
    assignment = read_json(root / "parallel_plan.json")
    layout = {**training.parallel_layout([worker["device"] for worker in assignment["workers"]]), "protocol": PROTOCOL}
    if assignment != {**layout, "allowlist_sha256": sf.sha256_file(root / "allowlist.json")} or count != layout["shard_count"] or not 0 <= index < count or str(torch.device(device)) != layout["workers"][index]["device"]:
        raise ValueError("test分片合同不一致")
    with training.session_mutex(root, shared=True), training.file_mutex(root / f".shard_{index}.lock"):
        for cell in training.shard_plan(index, count):
            evaluate(root, cell["arm"], cell["seed"], device, retry_failed)
    return {"shard": index, "completed": len(training.shard_plan(index, count))}


def parallel(root, devices, retry_failed=False):
    load_allowlist(root)
    layout = {**training.parallel_layout(devices), "protocol": PROTOCOL}
    with training.file_mutex(root / ".controller.lock"):
        with training.session_mutex(root):
            environments = [training.cuda_environment(w["device"]) for w in layout["workers"]]
            if len({e["device_uuid"] for e in environments}) != len(environments):
                raise ValueError("test设备映射重复")
            expected = {**layout, "allowlist_sha256": sf.sha256_file(root / "allowlist.json")}
            if (root / "parallel_plan.json").exists():
                if read_json(root / "parallel_plan.json") != expected:
                    raise ValueError("test调度表不可修改")
            else:
                write_json(root / "parallel_plan.json", expected)
        with evidence_attempt(root, root / "dispatch", "dispatch") as output, ExitStack() as stack:
            processes = []
            try:
                for worker in layout["workers"]:
                    cmd = [sys.executable, str(Path(__file__).resolve()), "run-shard", "--allowlist", str(root),
                           "--device", worker["device"], "--shard-index", str(worker["index"]),
                           "--shard-count", str(layout["shard_count"]), "--confirm-research-test"]
                    if retry_failed:
                        cmd.append("--retry-failed")
                    log_path = output / f"worker_{worker['index']}.log"
                    log = stack.enter_context(log_path.open("x"))
                    env = os.environ.copy()
                    env.update({key: "4" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
                    env["PYTHONUNBUFFERED"] = "1"
                    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    processes.append(proc)
                    write_json(output / f"worker_{worker['index']}.json", {**worker, "pid": proc.pid, "command": cmd, "log": str(log_path)})
                    print(f"TEST {worker['device']} {len(worker['cells'])} cells PID={proc.pid} LOG={log_path}", flush=True)
                while any(p.poll() is None for p in processes):
                    if any(p.poll() not in (None, 0) for p in processes):
                        raise RuntimeError(f"test worker失败: {output}")
                    time.sleep(5)
                if any(p.returncode != 0 for p in processes):
                    raise RuntimeError(f"test worker失败: {output}")
            except BaseException:
                training._stop_workers(processes)
                raise
            with training.session_mutex(root):
                result = summarize(root, retry_failed)
            write_json(output / "result.json", {"summary": str(result), "worker_exit_codes": [p.returncode for p in processes]})
    return result


def main():
    parser = argparse.ArgumentParser(description="APOR v2：六个新增checkpoint test与十二单元三视图汇总")
    commands = parser.add_subparsers(dest="phase", required=True)
    prepare = commands.add_parser("prepare-allowlist")
    prepare.add_argument("--session", type=Path, default=SESSION)
    for phase in ("status", "summary", "evaluate", "parallel", "run-shard"):
        sub = commands.add_parser(phase)
        sub.add_argument("--allowlist", required=True, type=Path)
        if phase != "status":
            sub.add_argument("--confirm-research-test", action="store_true", required=True)
            sub.add_argument("--retry-failed", action="store_true")
        if phase == "parallel":
            sub.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
        if phase in {"evaluate", "run-shard"}:
            sub.add_argument("--device", default="cuda:0")
        if phase == "evaluate":
            sub.add_argument("--arm", choices=TRAIN_ARMS, required=True)
            sub.add_argument("--seed", choices=SEEDS, type=int, required=True)
        if phase == "run-shard":
            sub.add_argument("--shard-index", type=int, required=True)
            sub.add_argument("--shard-count", type=int, required=True)
    args = parser.parse_args()
    if args.phase == "prepare-allowlist":
        result = prepare_allowlist(args.session)
    else:
        root = args.allowlist.resolve()
        if args.phase == "status":
            result = status(root)
        elif args.phase == "parallel":
            result = parallel(root, args.devices, args.retry_failed)
        elif args.phase == "run-shard":
            result = run_shard(root, args.device, args.shard_index, args.shard_count, args.retry_failed)
        else:
            with training.session_mutex(root):
                result = summarize(root, args.retry_failed) if args.phase == "summary" else evaluate(root, args.arm, args.seed, args.device, args.retry_failed)
    print(str(result) if isinstance(result, Path) else json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
