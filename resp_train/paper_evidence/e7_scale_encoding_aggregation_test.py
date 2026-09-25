"""E7 六臂×三 seed 固定 checkpoint 的 research-test 评价与析因汇总。"""

from __future__ import annotations

import fcntl
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
from resp_train.paper_evidence import e7_scale_encoding_aggregation as e7
from resp_train.paper_evidence import e7_scale_encoding_aggregation_formal as p4
from resp_train.paper_evidence import e4_scale_aggregation_test as reference_tools
from resp_train.paper_evidence.e1_scale_topology import array_hash
from resp_train.paper_evidence.e7_scale_encoding_aggregation_model import (
    ARMS,
    arm_contract,
    build_e7_model,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "e7-scale-encoding-aggregation-research-test-v1-20260926"
OUTPUT = Path("runs/e7_scale_encoding_aggregation/research_test")
LOCK_PATH = Path("docs/experiments/e7_scale_encoding_aggregation_test_lock_20260926.json")
PROTOCOL_PATH = Path("docs/experiments/e7_scale_encoding_aggregation_test_protocol_20260926.md")
MODULE_PATH = Path("resp_train/paper_evidence/e7_scale_encoding_aggregation_test.py")
SCRIPT_PATH = Path("scripts/eval_e7_scale_encoding_aggregation_test.py")
TEST_PATH = Path("tests/test_e7_scale_encoding_aggregation_test.py")
P4_LOCK_SHA256 = "068ba8ec6c5866ebf1b17d560b4fb5f5448e7e921e0dc8b15dda66c2f907c193"
P5_CLOSEOUT = Path("docs/experiments/e7_scale_encoding_aggregation_p5_closeout_20260926.json")
P5_CLOSEOUT_SHA256 = "7e9bc50516020c2193cacae9c3295dd28669906d4a74f80512d6f35e3fe91eee"
P5_FINAL_MANIFEST_SHA256 = "aae2d23fecd116db73b239ea028b4e394e72300b09e7077d54bc8e4235ad9837"
COUNT, SUBJECTS = 2310, 8
SAMPLE_SEED = 20260612
W0_AUDIT = reference_tools.W0_AUDIT
W0_AUDIT_SHA256 = reference_tools.W0_AUDIT_SHA
IDENTITY_COLUMNS = reference_tools.IDENTITY_COLUMNS
check_rows = reference_tools.check_rows
check_metrics = reference_tools.check_metrics
quality_flags = reference_tools.quality_flags
guarded_batches = reference_tools.guarded_batches


def test_contract() -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "p4_execution_lock_sha256": P4_LOCK_SHA256,
        "p5_closeout_sha256": P5_CLOSEOUT_SHA256,
        "arms": list(ARMS),
        "seeds": list(e7.SEEDS),
        "cells": len(ARMS) * len(e7.SEEDS),
        "split": "test",
        "count": COUNT,
        "samp_id_count": SUBJECTS,
        "sample_seed": SAMPLE_SEED,
        "batch_size": 128,
        "amp_dtype": "bfloat16",
        "include_test_only": False,
        "checkpoint_selector": "validation_local_rr_strict_minimum_earliest_tie",
        "validation_decision_frozen": True,
        "research_test_interpretation": "reused_developmental_description",
    }


def critical_paths(root: Path = ROOT) -> tuple[Path, ...]:
    paths = (MODULE_PATH, SCRIPT_PATH, TEST_PATH, PROTOCOL_PATH)
    missing = [str(path) for path in paths if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"E7 test 缺少关键文件: {missing}")
    return paths


def _source(
    files: dict[str, dict[str, Any]],
    path: Path,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = path.resolve()
    if expected is not None:
        e7.verify(path, expected)
    value = e7.identity(path)
    files[str(path)] = value
    return {"path": str(path), **value}


def _load_p5_closeout(root: Path = ROOT) -> tuple[dict[str, Any], Path]:
    path = root / P5_CLOSEOUT
    if e7.sha256_file(path) != P5_CLOSEOUT_SHA256:
        raise ValueError("E7 P5 closeout identity 漂移")
    closeout = json.loads(path.read_text(encoding="utf-8"))
    if (
        closeout.get("status") != "completed"
        or closeout.get("p4_execution_lock_sha256") != P4_LOCK_SHA256
        or closeout.get("research_test_accessed") is not False
        or closeout.get("tables", {}).get("seed_metrics") != 18
        or closeout.get("diagnostics", {}).get("cells") != 18
    ):
        raise ValueError("E7 P5 closeout 合同漂移")
    final = (root / closeout["final"]["path"]).resolve()
    e7.verify(final / "manifest.json", closeout["final"]["manifest"])
    if e7.sha256_file(final / "manifest.json") != P5_FINAL_MANIFEST_SHA256:
        raise ValueError("E7 P5 final manifest identity 漂移")
    receipt = json.loads((final / "final_receipt.json").read_text(encoding="utf-8"))
    if (
        receipt.get("p4_execution_lock_sha256") != P4_LOCK_SHA256
        or receipt.get("cells") != 18
        or receipt.get("research_test_accessed") is not False
    ):
        raise ValueError("E7 P5 final receipt 合同漂移")
    return closeout, final


def prepare_lock(root: Path = ROOT) -> Path:
    """生成 18-checkpoint allowlist；此阶段只读 test cache manifest 元数据。"""

    destination = root / LOCK_PATH
    if destination.exists():
        raise FileExistsError(f"E7 test lock 已存在: {destination}")
    state = e7.git_state(root)
    if state.get("status_porcelain"):
        raise RuntimeError("E7 test lock 要求干净工作树")
    _p4_lock, p4_digest = p4.load_p4_lock(root)
    if p4_digest != P4_LOCK_SHA256:
        raise ValueError("E7 test P4 execution lock identity 漂移")
    closeout, final = _load_p5_closeout(root)

    cache_root = FROZEN_RESEARCH_TEST_CACHE_ROOT
    cache_manifest_path = cache_root / "cache_manifest.json"
    if e7.sha256_file(cache_manifest_path) != FROZEN_RESEARCH_TEST_CACHE_MANIFEST_SHA256:
        raise ValueError("E7 test cache manifest identity 漂移")
    cache = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    metadata = cache["splits"]["test"]
    if metadata["count"] != COUNT or metadata["samp_id_count"] != SUBJECTS:
        raise ValueError("E7 test cache 样本合同漂移")
    cache_files = {
        name: cache["files"][name]
        for name in ("test_w.npy", "test_row_ids.npy", "w_frequencies_hz.npy")
    }

    p1_lock, _p1_digest = e7.load_implementation_lock(root)
    frequency_sha = p1_lock["cache_lock"]["frequency_file"]["file_sha256"]
    if cache_files["w_frequencies_hz.npy"]["sha256"] != frequency_sha:
        raise ValueError("E7 test 与训练频率网格 identity 不一致")
    if e7.sha256_file(root / W0_AUDIT) != W0_AUDIT_SHA256:
        raise ValueError("E7 W0 test audit identity 漂移")
    audit = pd.read_csv(root / W0_AUDIT)
    audit = audit.loc[audit.variant.eq("crd_tf102_w")].sort_values("seed")
    if tuple(audit.seed) != e7.SEEDS:
        raise ValueError("E7 W0 test audit seed 不完整")

    files: dict[str, dict[str, Any]] = {}
    _source(files, root / p4.P4_LOCK_PATH)
    _source(files, root / P5_CLOSEOUT)
    _source(files, final / "manifest.json", closeout["final"]["manifest"])
    _source(files, final / "final_receipt.json")
    _source(files, root / W0_AUDIT)
    _source(files, cache_manifest_path)

    w0_sources: dict[str, Any] = {}
    reference_ids: pd.DataFrame | None = None
    for old in p1_lock["w0_entries"]:
        seed = int(old["seed"])
        row = audit.loc[audit.seed.eq(seed)].iloc[0]
        if (
            row.checkpoint_sha256 != old["checkpoint"]["sha256"]
            or int(row.validation_selected_epoch) != int(old["selected_epoch"])
        ):
            raise ValueError("E7 W0 test checkpoint 与 P1 anchor 不一致")
        full = {}
        for filename, column in (
            ("research_test_metrics.csv", "metrics_sha256"),
            ("research_test_metrics_summary.csv", "metrics_summary_sha256"),
            ("research_test_metrics_manifest.json", "evaluation_manifest_sha256"),
        ):
            path = root / old["run_dir"] / filename
            if e7.sha256_file(path) != row[column]:
                raise ValueError(f"E7 W0 test 来源漂移: {filename}")
            full[filename] = _source(files, path)
        metrics = pd.read_csv(full["research_test_metrics.csv"]["path"])
        check_rows(metrics, reference_ids, row_hash=metadata["row_ids_sha256"])
        check_metrics(metrics, metrics)
        reference_ids = metrics[list(IDENTITY_COLUMNS)].copy()
        w0_sources[str(seed)] = {
            "selected_epoch": int(old["selected_epoch"]),
            "files": full,
        }

    entries = []
    completed = p4.completed_runs()
    by_cell = {}
    for attempt in completed:
        receipt = json.loads((attempt / "formal_receipt.json").read_text(encoding="utf-8"))
        by_cell[(receipt["arm"], int(receipt["seed"]))] = (attempt, receipt)
    for arm in ARMS:
        for seed in e7.SEEDS:
            attempt, receipt = by_cell[(arm, seed)]
            manifest = p4.verify_formal_attempt(attempt, lock_hash=P4_LOCK_SHA256)
            run = (attempt / receipt["run_dir"]).resolve()
            if not run.is_relative_to(attempt.resolve()):
                raise ValueError("E7 formal run_dir 越界")
            cfg = OmegaConf.load(run / "config.yaml")
            baseline = OmegaConf.create(p1_lock["baselines"][str(seed)])
            e7.validate_config(
                cfg,
                baseline,
                arm,
                p1_lock["frequency"]["values_hz"],
                output_root=attempt / "training",
                device=str(cfg.training.device),
            )
            epoch = e7.validate_history(pd.read_csv(run / "train_history.csv"), cfg)
            if (
                receipt["arm"] != arm
                or int(receipt["seed"]) != seed
                or int(receipt["planned_epochs"]) != e7.EPOCHS
                or int(receipt["planned_updates"]) != e7.EPOCHS * e7.UPDATES_PER_EPOCH
                or int(receipt["selected_epoch"]) != epoch
            ):
                raise ValueError("E7 test formal selector identity 漂移")
            _source(files, attempt / "manifest.json", e7.identity(attempt / "manifest.json"))
            for filename in ("formal_receipt.json", "freeze_receipt.json", "train_rows.csv", "val_rows.csv"):
                _source(files, attempt / filename)
            development = set(pd.read_csv(attempt / "train_rows.csv").samp_id) | set(
                pd.read_csv(attempt / "val_rows.csv").samp_id
            )
            if development & set(reference_ids.samp_id):
                raise ValueError("E7 test 与 train/validation samp_id 交叉")
            candidate = {
                name: _source(
                    files,
                    run / name,
                    manifest["files"][str((run / name).relative_to(attempt))],
                )
                for name in ("config.yaml", "train_history.csv", "checkpoint_best_local_rr.pt")
            }
            if candidate["checkpoint_best_local_rr.pt"]["sha256"] != receipt["selected_checkpoint"]["sha256"]:
                raise ValueError("E7 selected checkpoint identity 漂移")
            entries.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "selected_epoch": epoch,
                    "completed_epoch": int(receipt["completed_epochs"]),
                    "training_attempt": str(attempt.resolve()),
                    "training_manifest": e7.identity(attempt / "manifest.json"),
                    "training_config": OmegaConf.to_container(cfg, resolve=True),
                    "candidate": candidate,
                    "development_samp_ids": sorted(map(int, development)),
                }
            )

    lock = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "research_test_entry_locked_not_run",
        "contract": test_contract(),
        "contracts": {arm: arm_contract(arm) for arm in ARMS},
        "entries": entries,
        "w0_sources": w0_sources,
        "source_files": files,
        "cache_root": str(cache_root),
        "cache_identity": FROZEN_RESEARCH_TEST_CACHE_IDENTITY,
        "cache_manifest_sha256": FROZEN_RESEARCH_TEST_CACHE_MANIFEST_SHA256,
        "cache_files": cache_files,
        "dataset_index": {"path": cache["dataset_index"], "sha256": cache["dataset_index_sha256"]},
        "row_order_sha256": metadata["row_ids_sha256"],
        "p5_final": {"path": str(final), "manifest": closeout["final"]["manifest"]},
        "preparation_git": state,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "preparation_access": "frozen metadata and existing metric records",
        "code_files": {str(path): e7.identity(root / path) for path in critical_paths(root)},
    }
    e7.write_json(destination, lock)
    return destination


def load_lock(root: Path = ROOT) -> tuple[dict[str, Any], str]:
    path = root / LOCK_PATH
    lock = json.loads(path.read_text(encoding="utf-8"))
    if (
        lock.get("schema_version") != 1
        or lock.get("protocol") != PROTOCOL
        or lock.get("status") != "research_test_entry_locked_not_run"
        or lock.get("contract") != test_contract()
        or lock.get("contracts") != {arm: arm_contract(arm) for arm in ARMS}
        or lock.get("cache_root") != str(FROZEN_RESEARCH_TEST_CACHE_ROOT)
        or lock.get("cache_manifest_sha256") != FROZEN_RESEARCH_TEST_CACHE_MANIFEST_SHA256
    ):
        raise ValueError("E7 test lock 科学合同漂移")
    expected = [(arm, seed) for arm in ARMS for seed in e7.SEEDS]
    if [(item["arm"], item["seed"]) for item in lock["entries"]] != expected:
        raise ValueError("E7 test checkpoint allowlist 不完整")
    if set(lock["w0_sources"]) != {str(seed) for seed in e7.SEEDS}:
        raise ValueError("E7 test W0 对照不完整")
    for relative, expected_identity in lock["code_files"].items():
        e7.verify(root / relative, expected_identity)
    e7.verify(root / p4.P4_LOCK_PATH, lock["source_files"][str((root / p4.P4_LOCK_PATH).resolve())])
    if e7.sha256_file(root / p4.P4_LOCK_PATH) != P4_LOCK_SHA256:
        raise ValueError("E7 test P4 lock identity 漂移")
    _load_p5_closeout(root)
    return lock, e7.sha256_file(path)


@contextmanager
def phase_guard(parent: Path, lock_hash: str) -> Iterator[None]:
    parent.mkdir(parents=True, exist_ok=True)
    with (parent / f".execution_{lock_hash}.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("E7 test 相同阶段正在运行") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def attempt(
    parent: Path,
    phase: str,
    lock_hash: str,
    *,
    arm: str | None = None,
    seed: int | None = None,
) -> Iterator[Path]:
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = parent / f"{phase}_{lock_hash[:12]}_{stamp}_{uuid4().hex[:12]}"
    output.mkdir(exist_ok=False)
    context = {
        "protocol": PROTOCOL,
        "phase": phase,
        "split": "test",
        "arm": arm,
        "seed": seed,
        "test_lock_sha256": lock_hash,
        "command": sys.argv,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    e7.write_json(output / "lifecycle_started.json", {**context, "status": "running"})
    print(f"E7 test attempt: {output}", flush=True)
    try:
        yield output
        e7.write_json(
            output / "lifecycle_completed.json",
            {**context, "status": "completed", "ended_at": datetime.now(timezone.utc).isoformat()},
        )
        e7.write_json(
            output / "manifest.json",
            {
                **context,
                "status": "completed",
                "files": {
                    str(path.relative_to(output)): e7.identity(path)
                    for path in sorted(output.rglob("*"))
                    if path.is_file()
                },
            },
        )
        e7.write_json(output / "freeze_receipt.json", {"protocol": PROTOCOL, "manifest": e7.identity(output / "manifest.json")})
    except BaseException as exc:
        e7.write_json(
            output / "lifecycle_failed.json",
            {
                **context,
                "status": "failed",
                "error": str(exc),
                "error_type": type(exc).__name__,
                "traceback": traceback.format_exc(),
            },
        )
        raise


def verify_attempt(output: Path, lock_hash: str, phase: str) -> dict[str, Any]:
    output = output.resolve()
    if (output / "lifecycle_failed.json").exists():
        raise ValueError(f"E7 test attempt 已失败: {output}")
    freeze = json.loads((output / "freeze_receipt.json").read_text(encoding="utf-8"))
    e7.verify(output / "manifest.json", freeze["manifest"])
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("protocol") != PROTOCOL
        or manifest.get("phase") != phase
        or manifest.get("status") != "completed"
        or manifest.get("split") != "test"
        or manifest.get("test_lock_sha256") != lock_hash
    ):
        raise ValueError("E7 test attempt identity 漂移")
    required = (
        {
            "lifecycle_completed.json",
            "evaluation_receipt.json",
            "metrics.csv",
            "metrics_summary.csv",
            "test_rows.csv",
            "resolved_config.yaml",
            "access_started.json",
            "access_receipt.json",
            "environment.json",
            "test_lock.json",
        }
        if phase == "evaluation"
        else {
            "lifecycle_completed.json",
            "summary_receipt.json",
            "per_seed.csv",
            "across_seed.csv",
            "simple_effects_per_seed.csv",
            "simple_effects_across_seed.csv",
            "factorial_contrasts_per_seed.csv",
            "factorial_contrasts_across_seed.csv",
            "materiality_per_seed.csv",
            "materiality_across_seed.csv",
            "subject_macro.csv",
            "w0_paired_per_seed.csv",
            "w0_paired_across_seed.csv",
        }
    )
    if not required.issubset(manifest["files"]):
        raise ValueError("E7 test attempt 缺少必需产物")
    for relative, expected in manifest["files"].items():
        path = (output / relative).resolve()
        if not path.is_relative_to(output):
            raise ValueError("E7 test manifest 路径越界")
        e7.verify(path, expected)
    return manifest


def reject_completed(parent: Path, lock_hash: str, phase: str) -> None:
    for freeze in parent.glob("*/freeze_receipt.json"):
        manifest = json.loads((freeze.parent / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("test_lock_sha256") == lock_hash:
            verify_attempt(freeze.parent, lock_hash, phase)
            raise FileExistsError(f"E7 test 相同身份已完成: {freeze.parent}")


def evaluation_config(entry: Mapping[str, Any], device: str, cache_root: str):
    cfg = OmegaConf.load(entry["candidate"]["config.yaml"]["path"])
    if OmegaConf.to_container(cfg, resolve=True) != entry["training_config"]:
        raise ValueError("E7 test 训练配置与 allowlist 不一致")
    if (
        str(cfg.protocol.name) != e7.PROTOCOL
        or OmegaConf.to_container(cfg.model.e7_factorial, resolve=True) != arm_contract(entry["arm"])
        or int(cfg.training.seed) != entry["seed"]
        or int(cfg.model.initialization_seed) != entry["seed"]
        or float(cfg.loss.sync_weight) != 1.0
        or float(cfg.loss.effort_weight) != 0.25
        or str(cfg.data.test_split) != "test"
        or cfg.data.max_test_windows is not None
        or int(cfg.data.test_sample_seed) != SAMPLE_SEED
        or int(cfg.training.batch_size) != 128
        or not bool(cfg.training.use_amp)
        or str(cfg.training.amp_dtype) != "bfloat16"
    ):
        raise ValueError("E7 test 科学配置错误")
    cfg.training.device = device
    cfg.training.show_progress = False
    data_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    data_cfg.data.tf_research_test_cache_path = cache_root
    return cfg, data_cfg


def run_evaluation(arm: str, seed: int, *, device: str = "cuda:0") -> Path:
    if arm not in ARMS or seed not in e7.SEEDS:
        raise ValueError("E7 test arm/seed 不在冻结矩阵")
    lock, lock_hash = load_lock()
    entry = next(item for item in lock["entries"] if (item["arm"], item["seed"]) == (arm, seed))
    parent = ROOT / OUTPUT / "evaluation" / arm / f"seed_{seed}"
    with phase_guard(parent, lock_hash):
        reject_completed(parent, lock_hash, "evaluation")
        with attempt(parent, "evaluation", lock_hash, arm=arm, seed=seed) as output:
            environment = p4.runtime_preflight(device)
            e7.write_json(output / "environment.json", environment)
            e7.write_json(output / "test_lock.json", lock)
            e7.write_json(
                output / "access_started.json",
                {
                    "split": "test",
                    "arm": arm,
                    "seed": seed,
                    "count": COUNT,
                    "checkpoint": entry["candidate"]["checkpoint_best_local_rr.pt"],
                    "cache_root": lock["cache_root"],
                    "dataset_index": lock["dataset_index"],
                },
            )
            for path, expected in lock["source_files"].items():
                e7.verify(Path(path), expected)
            cache_root = Path(lock["cache_root"])
            if e7.sha256_file(cache_root / "cache_manifest.json") != lock["cache_manifest_sha256"]:
                raise ValueError("E7 test cache manifest 漂移")
            for filename, expected in lock["cache_files"].items():
                e7.verify(cache_root / filename, expected)
            if e7.sha256_file(Path(lock["dataset_index"]["path"])) != lock["dataset_index"]["sha256"]:
                raise ValueError("E7 test dataset index 漂移")

            cfg, data_cfg = evaluation_config(entry, device, lock["cache_root"])
            OmegaConf.save(data_cfg, output / "resolved_config.yaml")
            checkpoint = torch.load(
                entry["candidate"]["checkpoint_best_local_rr.pt"]["path"],
                map_location="cpu",
                weights_only=False,
            )
            if checkpoint.get("config") != entry["training_config"] or int(checkpoint["epoch"]) != entry["selected_epoch"]:
                raise ValueError("E7 test checkpoint config/epoch 漂移")
            _validate_checkpoint_config(checkpoint["config"], cfg)
            p4.finite_tree(checkpoint["model_state_dict"])
            model = build_e7_model(cfg)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)

            reference = pd.read_csv(lock["w0_sources"][str(seed)]["files"]["research_test_metrics.csv"]["path"])
            audited = read_research_v2_index(cfg.data.dataset_root, cfg.data.index_csv, cfg)
            rows = filter_index(
                audited,
                cfg,
                split="test",
                max_windows=None,
                sample_strategy=str(cfg.data.test_sample_strategy),
                sample_seed=SAMPLE_SEED,
            )
            check_rows(rows, reference, row_hash=lock["row_order_sha256"])
            if set(rows.samp_id) & set(entry["development_samp_ids"]):
                raise ValueError("E7 test 与开发 samp_id 交叉")
            rows.to_csv(output / "test_rows.csv", index=False)
            data = build_window_data(
                data_cfg,
                split="test",
                max_windows=None,
                sample_strategy=str(cfg.data.test_sample_strategy),
                sample_seed=SAMPLE_SEED,
                shuffle=False,
                audited=audited,
            )
            check_rows(data.rows, rows, row_hash=lock["row_order_sha256"])
            if len(data.dataset) != COUNT:
                raise ValueError("E7 test dataset 样本数不完整")
            predictions = collect_predictions(
                model,
                guarded_batches(data.loader, rows),
                device=device,
                max_windows=COUNT,
                use_amp=True,
            )
            metrics = evaluate_task_predictions(predictions, cfg, include_test_only=False, method=arm)
            summary = check_metrics(metrics, reference)
            quality = quality_flags(metrics)
            metrics.insert(0, "arm", arm)
            metrics.insert(0, "seed", seed)
            for name, value in {
                "arm": arm,
                "seed": seed,
                "split": "test",
                "selected_epoch": entry["selected_epoch"],
                "quality_acceptance_passed": quality["quality_acceptance_passed"],
            }.items():
                summary.insert(0, name, value)
            metrics.to_csv(output / "metrics.csv", index=False)
            summary.to_csv(output / "metrics_summary.csv", index=False)
            e7.write_json(
                output / "access_receipt.json",
                {
                    "arm": arm,
                    "seed": seed,
                    "split": "test",
                    "rows": COUNT,
                    "samp_ids": SUBJECTS,
                    "row_order_sha256": array_hash(rows.dataset_row_id.to_numpy()),
                    "cache_identity": lock["cache_identity"],
                    "cache_files_verified": lock["cache_files"],
                    "checkpoint": entry["candidate"]["checkpoint_best_local_rr.pt"],
                    "test_sample_seed": SAMPLE_SEED,
                },
            )
            e7.write_json(
                output / "evaluation_receipt.json",
                {
                    "protocol": PROTOCOL,
                    "arm": arm,
                    "seed": seed,
                    "split": "test",
                    "selected_epoch": entry["selected_epoch"],
                    "checkpoint_sha256": entry["candidate"]["checkpoint_best_local_rr.pt"]["sha256"],
                    "rows": COUNT,
                    "samp_ids": SUBJECTS,
                    "primary_finite": True,
                    **quality,
                    "target_eligibility_matches_w0": True,
                    "row_order_sha256": lock["row_order_sha256"],
                },
            )
    return output


def _validate_summary(saved: pd.DataFrame, expected: pd.DataFrame) -> None:
    if len(saved) != 1:
        raise ValueError("E7 test metrics summary 必须一行")
    for metric in e7.PRIMARY:
        if (
            not np.isclose(saved.iloc[0][metric + "_mean"], expected.iloc[0][metric + "_mean"], atol=1e-12, rtol=0)
            or int(saved.iloc[0][metric + "_n"]) != COUNT
        ):
            raise ValueError(f"E7 test summary 数值/分母漂移: {metric}")


def materiality_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    e7._matrix_index(frame, split="test")
    indexed = frame.set_index(["arm", "seed"])
    records = []
    for contrast, coefficients in e7.SIMPLE_CONTRASTS.items():
        candidate_arm = next(arm for arm, value in coefficients.items() if value == 1)
        baseline_arm = next(arm for arm, value in coefficients.items() if value == -1)
        for seed in e7.SEEDS:
            for metric in e7.PRIMARY:
                candidate = float(indexed.loc[(candidate_arm, seed), metric + "_mean"])
                baseline = float(indexed.loc[(baseline_arm, seed), metric + "_mean"])
                if metric == e7.PCC:
                    delta, tolerance, unit, defined = candidate - baseline, e7.PCC_TOLERANCE, "absolute_utility", True
                else:
                    defined = baseline != 0
                    delta = 100 * (baseline - candidate) / baseline if defined else np.nan
                    tolerance, unit = e7.ERROR_TOLERANCE_PERCENT, "relative_percent_utility"
                classification = (
                    "undefined"
                    if not defined
                    else "improved"
                    if delta > tolerance
                    else "degraded"
                    if delta < -tolerance
                    else "within_tolerance"
                )
                records.append(
                    {
                        "contrast": contrast,
                        "candidate_arm": candidate_arm,
                        "baseline_arm": baseline_arm,
                        "seed": seed,
                        "metric": metric,
                        "candidate": candidate,
                        "baseline": baseline,
                        "utility_delta": delta,
                        "unit": unit,
                        "tolerance": tolerance,
                        "classification": classification,
                    }
                )
    per_seed = pd.DataFrame(records)
    across = []
    for (contrast, metric), group in per_seed.groupby(["contrast", "metric"], sort=False):
        finite = group.utility_delta.to_numpy(dtype=float)
        finite = finite[np.isfinite(finite)]
        across.append(
            {
                "contrast": contrast,
                "metric": metric,
                "utility_delta_mean": finite.mean() if len(finite) else np.nan,
                "utility_delta_sample_sd": finite.std(ddof=1) if len(finite) > 1 else np.nan,
                "improved_seeds": int(group.classification.eq("improved").sum()),
                "within_tolerance_seeds": int(group.classification.eq("within_tolerance").sum()),
                "degraded_seeds": int(group.classification.eq("degraded").sum()),
                "undefined_seeds": int(group.classification.eq("undefined").sum()),
            }
        )
    return per_seed, pd.DataFrame(across)


def w0_paired_tables(candidate: pd.DataFrame, w0: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    indexed = w0.set_index("seed")
    for record in candidate.itertuples(index=False):
        for metric in e7.PRIMARY:
            value = float(getattr(record, metric + "_mean"))
            baseline = float(indexed.loc[record.seed, metric + "_mean"])
            if metric == e7.PCC:
                delta, unit = value - baseline, "absolute_utility"
            else:
                delta = 100 * (baseline - value) / baseline if baseline != 0 else np.nan
                unit = "relative_percent_utility"
            rows.append(
                {
                    "arm": record.arm,
                    "seed": int(record.seed),
                    "metric": metric,
                    "candidate": value,
                    "w0": baseline,
                    "utility_delta": delta,
                    "unit": unit,
                    "positive_means": "candidate_improves",
                }
            )
    per_seed = pd.DataFrame(rows)
    across = []
    for (arm, metric), group in per_seed.groupby(["arm", "metric"], sort=False):
        values = group.utility_delta.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        across.append(
            {
                "arm": arm,
                "metric": metric,
                "utility_delta_mean": finite.mean() if len(finite) else np.nan,
                "utility_delta_sample_sd": finite.std(ddof=1) if len(finite) > 1 else np.nan,
                "positive_seeds": int((values > 0).sum()),
                "zero_seeds": int((values == 0).sum()),
                "negative_seeds": int((values < 0).sum()),
            }
        )
    return per_seed, pd.DataFrame(across)


def summarize(runs: list[Path]) -> Path:
    lock, lock_hash = load_lock()
    expected_cells = len(ARMS) * len(e7.SEEDS)
    if len(runs) != expected_cells or len({path.resolve() for path in runs}) != expected_cells:
        raise ValueError(f"E7 test 汇总要求 {expected_cells} 个不同完成 attempt")
    parent = ROOT / OUTPUT / "summary"
    with phase_guard(parent, lock_hash):
        reject_completed(parent, lock_hash, "summary")
        summaries, metrics_frames, sources = [], [], {}
        for output in runs:
            manifest = verify_attempt(output, lock_hash, "evaluation")
            arm, seed = manifest["arm"], int(manifest["seed"])
            key = f"{arm}/{seed}"
            if arm not in ARMS or seed not in e7.SEEDS or key in sources:
                raise ValueError("E7 test 汇总 cell 重复或越界")
            entry = next(item for item in lock["entries"] if (item["arm"], item["seed"]) == (arm, seed))
            receipt = json.loads((output / "evaluation_receipt.json").read_text(encoding="utf-8"))
            if (
                receipt.get("arm") != arm
                or int(receipt.get("seed", -1)) != seed
                or receipt.get("split") != "test"
                or int(receipt.get("rows", -1)) != COUNT
                or int(receipt.get("selected_epoch", -1)) != entry["selected_epoch"]
                or receipt.get("checkpoint_sha256") != entry["candidate"]["checkpoint_best_local_rr.pt"]["sha256"]
            ):
                raise ValueError("E7 test evaluation receipt 与 allowlist 不一致")
            reference = pd.read_csv(lock["w0_sources"][str(seed)]["files"]["research_test_metrics.csv"]["path"])
            metrics = pd.read_csv(output / "metrics.csv")
            if not metrics.arm.eq(arm).all() or not metrics.seed.eq(seed).all():
                raise ValueError("E7 test metrics arm/seed identity 错误")
            check_rows(metrics, pd.read_csv(output / "test_rows.csv"), row_hash=lock["row_order_sha256"])
            expected_summary = check_metrics(metrics, reference)
            saved = pd.read_csv(output / "metrics_summary.csv")
            _validate_summary(saved, expected_summary)
            if saved.iloc[0].arm != arm or int(saved.iloc[0].seed) != seed or saved.iloc[0].split != "test":
                raise ValueError("E7 test metrics summary identity 错误")
            summaries.append(saved)
            metrics_frames.append(metrics)
            sources[key] = {
                "path": str(output.resolve()),
                "manifest": e7.identity(output / "manifest.json"),
                "selected_epoch": entry["selected_epoch"],
            }
        per_seed = pd.concat(summaries, ignore_index=True).sort_values(["arm", "seed"]).reset_index(drop=True)
        e7._matrix_index(per_seed, split="test")
        across = e7.across_seed_table(per_seed, split="test")
        simple_seed, simple_across, factorial_seed, factorial_across = e7.contrast_tables(per_seed, split="test")
        material_seed, material_across = materiality_tables(per_seed)
        subject_macro = e7.subject_macro_table(pd.concat(metrics_frames, ignore_index=True), split="test")

        w0_rows = []
        for seed in e7.SEEDS:
            reference = pd.read_csv(lock["w0_sources"][str(seed)]["files"]["research_test_metrics.csv"]["path"])
            summary = summarize_task_metrics(reference).iloc[0].to_dict()
            w0_rows.append({"seed": seed, **summary})
        w0_seed = pd.DataFrame(w0_rows)
        w0_paired_seed, w0_paired_across = w0_paired_tables(per_seed, w0_seed)
        tables = {
            "per_seed.csv": per_seed,
            "across_seed.csv": across,
            "simple_effects_per_seed.csv": simple_seed,
            "simple_effects_across_seed.csv": simple_across,
            "factorial_contrasts_per_seed.csv": factorial_seed,
            "factorial_contrasts_across_seed.csv": factorial_across,
            "materiality_per_seed.csv": material_seed,
            "materiality_across_seed.csv": material_across,
            "subject_macro.csv": subject_macro,
            "w0_paired_per_seed.csv": w0_paired_seed,
            "w0_paired_across_seed.csv": w0_paired_across,
        }
        with attempt(parent, "summary", lock_hash) as output:
            for filename, frame in tables.items():
                frame.to_csv(output / filename, index=False, na_rep="NA")
            e7.write_json(
                output / "summary_receipt.json",
                {
                    "protocol": PROTOCOL,
                    "split": "test",
                    "arms": list(ARMS),
                    "seeds": list(e7.SEEDS),
                    "cells": expected_cells,
                    "rows_per_cell": COUNT,
                    "new_metric_rows": expected_cells * COUNT,
                    "source_runs": sources,
                    "w0_sources": lock["w0_sources"],
                    "tables": {name: len(frame) for name, frame in tables.items()},
                    "validation_decision_changed": False,
                    "interpretation": "reused research-test developmental description",
                },
            )
    return output


def completed_runs() -> list[Path]:
    _lock, lock_hash = load_lock()
    result = []
    for arm in ARMS:
        for seed in e7.SEEDS:
            parent = ROOT / OUTPUT / "evaluation" / arm / f"seed_{seed}"
            matches = []
            for freeze in parent.glob("*/freeze_receipt.json"):
                manifest = json.loads((freeze.parent / "manifest.json").read_text(encoding="utf-8"))
                if manifest.get("test_lock_sha256") == lock_hash:
                    if manifest.get("arm") != arm or int(manifest.get("seed", -1)) != seed:
                        raise ValueError("E7 test 完成目录与 manifest cell 不一致")
                    verify_attempt(freeze.parent, lock_hash, "evaluation")
                    matches.append(freeze.parent.resolve())
            if len(matches) != 1:
                raise ValueError(f"E7 test {arm}/{seed} 需要唯一成功 attempt，实际 {len(matches)}")
            result.extend(matches)
    return result
