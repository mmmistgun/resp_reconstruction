"""E7 P4 formal train/validation 执行锁、生命周期与产物审计。"""

from __future__ import annotations

import fcntl
import json
import sys
import traceback
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from resp_train.crd.config import check_crd_dependencies
from resp_train.crd.training import build_crd_optimizer
from resp_train.data.index import filter_index
from resp_train.data.research_v2 import read_research_v2_index
from resp_train.metrics.task import summarize_task_metrics
from resp_train.paper_evidence import e7_scale_encoding_aggregation as e7
from resp_train.paper_evidence import e7_scale_encoding_aggregation_engineering as p3
from resp_train.paper_evidence.e1_scale_topology import array_hash
from resp_train.paper_evidence.e1_scale_topology_runtime import environment
from resp_train.paper_evidence.e7_scale_encoding_aggregation_model import ARMS, build_e7_model


P4_PROTOCOL = "e7-scale-encoding-aggregation-p4-formal-v1-20260924"
P1_LOCK_SHA256 = p3.P1_LOCK_SHA256
P3_LOCK_SHA256 = "773f5e78c5e8d7b08cadaac00797ede8dcd9f3f7a77d0775fc5679650a29d32d"
P3_CLOSEOUT = Path("docs/experiments/e7_scale_encoding_aggregation_p3_closeout_20260924.json")
P3_CLOSEOUT_SHA256 = "29334d33e4a3fb6666230bb5cf6915d16c5cf34e8a2cf49a07bd58d499433182"
P4_PROTOCOL_PATH = Path(
    "docs/experiments/e7_scale_encoding_aggregation_p4_formal_protocol_20260924.md"
)
P4_LOCK_PATH = Path(
    "docs/experiments/e7_scale_encoding_aggregation_p4_execution_lock_20260924.json"
)
FORMAL_PATH = Path(
    "resp_train/paper_evidence/e7_scale_encoding_aggregation_formal.py"
)
P4_SCRIPT_PATH = Path("scripts/run_e7_scale_encoding_aggregation_p4.py")
P4_TEST_PATH = Path("tests/test_e7_scale_encoding_aggregation_formal.py")


def p4_contract() -> dict[str, Any]:
    return {
        "protocol": P4_PROTOCOL,
        "p1_implementation_lock_sha256": P1_LOCK_SHA256,
        "p3_engineering_lock_sha256": P3_LOCK_SHA256,
        "p3_closeout_sha256": P3_CLOSEOUT_SHA256,
        "arms": list(ARMS),
        "seeds": list(e7.SEEDS),
        "cells": len(ARMS) * len(e7.SEEDS),
        "counts": e7.COUNTS,
        "samp_ids": e7.SAMP_IDS,
        "max_epochs": e7.EPOCHS,
        "updates_per_epoch": e7.UPDATES_PER_EPOCH,
        "planned_updates": e7.EPOCHS * e7.UPDATES_PER_EPOCH,
        "early_stopping": e7.expected_spec()["matrix"]["early_stopping"],
        "checkpoint_selector": "full_validation_local_rr_strict_minimum_earliest_tie",
        "split_scope": ["train", "val"],
        "research_test_access": False,
    }


def p4_critical_paths() -> tuple[Path, ...]:
    paths = (FORMAL_PATH, P4_SCRIPT_PATH, P4_TEST_PATH, P4_PROTOCOL_PATH)
    missing = [str(path) for path in paths if not (e7.ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"E7 P4 缺少关键文件: {missing}")
    return paths


def _load_closeout(root: Path = e7.ROOT) -> dict[str, Any]:
    path = root / P3_CLOSEOUT
    if e7.sha256_file(path) != P3_CLOSEOUT_SHA256:
        raise ValueError("E7 P3 closeout identity 漂移")
    closeout = json.loads(path.read_text(encoding="utf-8"))
    if (
        closeout.get("schema_version") != 1
        or closeout.get("protocol") != p3.P3_PROTOCOL
        or closeout.get("status") != "completed"
        or closeout.get("p1_implementation_lock_sha256") != P1_LOCK_SHA256
        or closeout.get("p3_engineering_lock_sha256") != P3_LOCK_SHA256
        or closeout.get("gpu_acceptance", {}).get("records") != 24
        or closeout.get("gpu_acceptance", {}).get("passed_records") != 24
        or closeout.get("benchmark", {}).get("measurements") != 36
        or closeout.get("gpu_acceptance", {}).get("real_data_read") is not False
        or closeout.get("gpu_acceptance", {}).get("research_test_read") is not False
    ):
        raise ValueError("E7 P3 closeout 合同漂移")
    gpu_path = root / closeout["gpu_acceptance"]["path"]
    benchmark_path = root / closeout["benchmark"]["path"]
    e7.verify(gpu_path / "manifest.json", closeout["gpu_acceptance"]["manifest"])
    e7.verify(benchmark_path / "manifest.json", closeout["benchmark"]["manifest"])
    if (
        e7.sha256_file(gpu_path / "gpu_acceptance.json")
        != closeout["gpu_acceptance"]["result_sha256"]
        or e7.sha256_file(benchmark_path / "benchmark.json")
        != closeout["benchmark"]["result_sha256"]
    ):
        raise ValueError("E7 P3 closeout result identity 漂移")
    p3.verify_attempt(gpu_path, phase="gpu_acceptance", lock_hash=P3_LOCK_SHA256)
    p3.verify_attempt(benchmark_path, phase="benchmark", lock_hash=P3_LOCK_SHA256)
    gpu_environment = json.loads((gpu_path / "environment.json").read_text())
    benchmark_environment = json.loads((benchmark_path / "environment.json").read_text())
    keys = (
        "packages",
        "gpu_name",
        "cuda",
        "cudnn",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "matmul_allow_tf32",
        "cudnn_allow_tf32",
        "deterministic_algorithms",
    )
    if any(gpu_environment.get(key) != benchmark_environment.get(key) for key in keys):
        raise ValueError("E7 P3 acceptance 与 benchmark 环境漂移")
    return closeout


def prepare_p4_lock(root: Path = e7.ROOT) -> Path:
    destination = root / P4_LOCK_PATH
    if destination.exists():
        raise FileExistsError(f"E7 P4 execution lock 已存在: {destination}")
    state = e7.git_state(root)
    if state.get("status_porcelain"):
        raise RuntimeError("E7 P4 execution lock 要求干净工作树")
    _p1_lock, p1_digest = e7.load_implementation_lock(root)
    _p3_lock, p3_digest = p3.load_p3_lock(root)
    if p1_digest != P1_LOCK_SHA256 or p3_digest != P3_LOCK_SHA256:
        raise ValueError("E7 P4 上游 lock identity 漂移")
    closeout = _load_closeout(root)
    gpu_path = root / closeout["gpu_acceptance"]["path"]
    lock = {
        "schema_version": 1,
        "protocol": P4_PROTOCOL,
        "status": "formal_entry_locked_not_run",
        "contract": p4_contract(),
        "p1_implementation_lock": {
            "path": str(e7.LOCK_PATH),
            **e7.identity(root / e7.LOCK_PATH),
        },
        "p3_engineering_lock": {
            "path": str(p3.P3_LOCK_PATH),
            **e7.identity(root / p3.P3_LOCK_PATH),
        },
        "p3_closeout": {"path": str(P3_CLOSEOUT), **e7.identity(root / P3_CLOSEOUT)},
        "gpu_receipt_allowlist": {
            "path": closeout["gpu_acceptance"]["path"],
            "manifest": closeout["gpu_acceptance"]["manifest"],
            "result": e7.identity(gpu_path / "gpu_acceptance.json"),
            "environment": e7.identity(gpu_path / "environment.json"),
        },
        "code_files": {
            str(path): e7.identity(root / path)
            for path in p4_critical_paths()
        },
        "preparation_git": state,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }
    e7.write_json(destination, lock)
    return destination


def load_p4_lock(root: Path = e7.ROOT) -> tuple[dict[str, Any], str]:
    path = root / P4_LOCK_PATH
    lock = json.loads(path.read_text(encoding="utf-8"))
    if (
        lock.get("schema_version") != 1
        or lock.get("protocol") != P4_PROTOCOL
        or lock.get("status") != "formal_entry_locked_not_run"
        or lock.get("contract") != p4_contract()
    ):
        raise ValueError("E7 P4 execution lock 合同漂移")
    e7.verify(root / e7.LOCK_PATH, lock["p1_implementation_lock"])
    e7.verify(root / p3.P3_LOCK_PATH, lock["p3_engineering_lock"])
    e7.verify(root / P3_CLOSEOUT, lock["p3_closeout"])
    _p1_lock, p1_digest = e7.load_implementation_lock(root)
    _p3_lock, p3_digest = p3.load_p3_lock(root)
    if p1_digest != P1_LOCK_SHA256 or p3_digest != P3_LOCK_SHA256:
        raise ValueError("E7 P4 上游 lock 回载 identity 漂移")
    _load_closeout(root)
    for relative, expected in lock["code_files"].items():
        e7.verify(root / relative, expected)
    return lock, e7.sha256_file(path)


@contextmanager
def phase_guard(parent: Path, lock_hash: str) -> Iterator[None]:
    parent.mkdir(parents=True, exist_ok=True)
    with (parent / f".execution_{lock_hash}.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("E7 P4 相同 cell 正在运行") from exc
        try:
            for receipt in parent.glob("*/freeze_receipt.json"):
                manifest = json.loads((receipt.parent / "manifest.json").read_text())
                if manifest.get("p4_execution_lock_sha256") == lock_hash:
                    verify_formal_attempt(receipt.parent, lock_hash=lock_hash)
                    raise FileExistsError(f"E7 P4 相同 cell 已完成: {receipt.parent}")
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def attempt(
    parent: Path,
    lock_hash: str,
    arm: str,
    seed: int,
) -> Iterator[Path]:
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = parent / f"formal_{lock_hash[:12]}_{stamp}_{uuid4().hex[:12]}"
    output.mkdir(exist_ok=False)
    context = {
        "protocol": P4_PROTOCOL,
        "phase": "formal",
        "arm": arm,
        "seed": int(seed),
        "p4_execution_lock_sha256": lock_hash,
        "p1_implementation_lock_sha256": P1_LOCK_SHA256,
        "p3_engineering_lock_sha256": P3_LOCK_SHA256,
        "command": sys.argv,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    e7.write_json(output / "lifecycle_started.json", {**context, "status": "running"})
    try:
        yield output
        e7.write_json(
            output / "lifecycle_completed.json",
            {**context, "status": "completed", "ended_at": datetime.now(timezone.utc).isoformat()},
        )
        manifest = {
            **context,
            "status": "completed",
            "files": {
                str(file.relative_to(output)): e7.identity(file)
                for file in sorted(output.rglob("*"))
                if file.is_file()
            },
        }
        e7.write_json(output / "manifest.json", manifest)
        e7.write_json(
            output / "freeze_receipt.json",
            {"protocol": P4_PROTOCOL, "manifest": e7.identity(output / "manifest.json")},
        )
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


def verify_formal_attempt(path: Path, *, lock_hash: str) -> dict[str, Any]:
    path = path.resolve()
    if (path / "lifecycle_failed.json").exists():
        raise ValueError(f"E7 P4 formal attempt 失败: {path}")
    freeze = json.loads((path / "freeze_receipt.json").read_text())
    e7.verify(path / "manifest.json", freeze["manifest"])
    manifest = json.loads((path / "manifest.json").read_text())
    if (
        manifest.get("protocol") != P4_PROTOCOL
        or manifest.get("phase") != "formal"
        or manifest.get("status") != "completed"
        or manifest.get("p4_execution_lock_sha256") != lock_hash
    ):
        raise ValueError("E7 P4 formal attempt identity 漂移")
    required = {
        "lifecycle_completed.json",
        "environment.json",
        "access_receipt.json",
        "formal_receipt.json",
        "p4_execution_lock.json",
        "gpu_acceptance_source.json",
    }
    if not required.issubset(manifest["files"]):
        raise ValueError("E7 P4 formal attempt 缺少必需产物")
    for relative, expected in manifest["files"].items():
        file = (path / relative).resolve()
        if not file.is_relative_to(path):
            raise ValueError("E7 P4 formal manifest 路径越界")
        e7.verify(file, expected)
    return manifest


def runtime_preflight(device: str) -> dict[str, Any]:
    state = e7.git_state(e7.ROOT)
    if state.get("status_porcelain"):
        raise RuntimeError("E7 P4 formal 要求干净提交")
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("E7 P4 formal 需要 CUDA")
    problems = check_crd_dependencies()
    if problems:
        raise RuntimeError("E7 P4 原生依赖检查失败: " + "; ".join(problems))
    return {**environment(device), "git": state}


def verify_gpu_receipt(
    receipt: Path,
    lock: Mapping[str, Any],
    *,
    current_environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    expected = lock["gpu_receipt_allowlist"]
    actual = receipt.resolve()
    allowed = (e7.ROOT / expected["path"]).resolve()
    if actual != allowed:
        raise ValueError("E7 P4 gpu receipt 不在 execution lock allowlist")
    p3.verify_attempt(actual, phase="gpu_acceptance", lock_hash=P3_LOCK_SHA256)
    e7.verify(actual / "manifest.json", expected["manifest"])
    e7.verify(actual / "gpu_acceptance.json", expected["result"])
    e7.verify(actual / "environment.json", expected["environment"])
    result = json.loads((actual / "gpu_acceptance.json").read_text())
    expected_cells = {
        (arm, seed, batch)
        for arm in ARMS
        for seed, batch in [*((seed, 1) for seed in e7.SEEDS), (e7.SEEDS[0], 128)]
    }
    observed_cells = {
        (record["arm"], int(record["seed"]), int(record["batch_size"]))
        for record in result.get("records", [])
    }
    if (
        result.get("passed") is not True
        or result.get("arms") != list(ARMS)
        or result.get("seeds") != list(e7.SEEDS)
        or result.get("physical_batch") != 128
        or observed_cells != expected_cells
        or len(result.get("records", [])) != 24
        or any(record.get("status") != "passed" for record in result["records"])
    ):
        raise ValueError("E7 P4 gpu receipt acceptance 矩阵不完整")
    accepted_environment = json.loads((actual / "environment.json").read_text())
    if current_environment is not None:
        keys = (
            "packages",
            "gpu_name",
            "cuda",
            "cudnn",
            "cudnn_benchmark",
            "cudnn_deterministic",
            "matmul_allow_tf32",
            "cudnn_allow_tf32",
            "deterministic_algorithms",
        )
        if any(
            accepted_environment.get(key) != current_environment.get(key)
            for key in keys
        ):
            raise ValueError("E7 P4 formal 与 GPU acceptance 环境不一致")
    return result


def audit_sources(
    p1_lock: Mapping[str, Any],
    cfg,
    output: Path,
) -> dict[str, pd.DataFrame]:
    e7.write_json(
        output / "access_started.json",
        {
            "splits": ["train", "val"],
            "purpose": "E7 P4 formal train and validation",
            "source_files": list(p1_lock["source_files"]),
            "dataset_index": p1_lock["dataset_index"],
            "research_test_access": False,
        },
    )
    for relative, expected in p1_lock["source_files"].items():
        e7.verify(e7.ROOT / relative, expected)
    dataset = p1_lock["dataset_index"]
    dataset_path = Path(dataset["path"])
    if e7.sha256_file(dataset_path) != dataset["sha256"]:
        raise ValueError("E7 P4 dataset index identity 漂移")
    audited = read_research_v2_index(cfg.data.dataset_root, cfg.data.index_csv, cfg)
    rows: dict[str, pd.DataFrame] = {}
    for split, count in e7.COUNTS.items():
        frame = filter_index(
            audited,
            cfg,
            split=split,
            max_windows=None,
            sample_strategy=str(cfg.data[f"{split}_sample_strategy"]),
            sample_seed=int(cfg.data[f"{split}_sample_seed"]),
        )
        if (
            len(frame) != count
            or frame.dataset_row_id.duplicated().any()
            or set(frame.split.astype(str)) != {split}
            or frame.samp_id.nunique() != e7.SAMP_IDS[split]
            or array_hash(np.sort(frame.dataset_row_id.to_numpy()))
            != p1_lock["cache_lock"]["row_identity"][f"{split}_row_content_sha256"]
        ):
            raise ValueError(f"E7 P4 {split} row identity 漂移")
        frame.to_csv(output / f"{split}_rows.csv", index=False)
        rows[split] = frame
    if set(rows["train"].samp_id) & set(rows["val"].samp_id):
        raise ValueError("E7 P4 train/validation samp_id 交叉")
    e7.write_json(
        output / "access_receipt.json",
        {
            "counts": e7.COUNTS,
            "samp_ids": e7.SAMP_IDS,
            "row_order_sha256": {
                split: array_hash(frame.dataset_row_id.to_numpy())
                for split, frame in rows.items()
            },
            "sample_seeds": {
                split: int(cfg.data[f"{split}_sample_seed"])
                for split in e7.COUNTS
            },
            "train_accessed": True,
            "validation_accessed": True,
            "research_test_accessed": False,
        },
    )
    return rows


def finite_tree(value: Any) -> None:
    if torch.is_tensor(value):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError("E7 P4 checkpoint/optimizer tensor 非有限")
    elif isinstance(value, Mapping):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            finite_tree(item)
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise FloatingPointError("E7 P4 checkpoint scalar 非有限")


def validate_run(run_dir: Path, cfg, rows: pd.DataFrame) -> dict[str, Any]:
    saved = OmegaConf.load(run_dir / "config.yaml")
    if OmegaConf.to_container(saved, resolve=True) != OmegaConf.to_container(cfg, resolve=True):
        raise ValueError("E7 P4 保存配置漂移")
    history = pd.read_csv(run_dir / "train_history.csv")
    selected_epoch = e7.validate_history(history, cfg)
    final_epoch = int(history.iloc[-1].epoch)
    model = build_e7_model(cfg)
    optimizer, partition = build_crd_optimizer(model, cfg)
    saved_groups = json.loads((run_dir / "optimizer_parameter_groups.json").read_text())
    if saved_groups != {
        "weight_decay": float(cfg.training.weight_decay),
        "decay": list(partition.decay_names),
        "no_decay": list(partition.no_decay_names),
    }:
        raise ValueError("E7 P4 optimizer 参数分组漂移")
    for filename, expected_epoch in (
        ("checkpoint_best_local_rr.pt", selected_epoch),
        ("checkpoint_final.pt", final_epoch),
    ):
        checkpoint = torch.load(
            run_dir / filename, map_location="cpu", weights_only=False
        )
        finite_tree(checkpoint)
        if (
            int(checkpoint["epoch"]) != expected_epoch
            or checkpoint["config"] != OmegaConf.to_container(cfg, resolve=True)
        ):
            raise ValueError("E7 P4 checkpoint epoch/config identity 漂移")
        history_row = history.loc[history.epoch.eq(expected_epoch)].iloc[0]
        for key in history.columns:
            if key not in checkpoint["metrics"] or not np.isclose(
                float(checkpoint["metrics"][key]),
                float(history_row[key]),
                atol=1e-12,
                rtol=1e-12,
            ):
                raise ValueError(f"E7 P4 checkpoint 与 history 不一致: {key}")
        extra = checkpoint["extra_state"]
        if (
            extra.get("protocol") != e7.PROTOCOL
            or int(extra.get("update_index", -1))
            != expected_epoch * e7.UPDATES_PER_EPOCH
            or int(extra.get("total_updates", -1))
            != e7.EPOCHS * e7.UPDATES_PER_EPOCH
        ):
            raise ValueError("E7 P4 checkpoint update/protocol 漂移")
        early = extra.get("early_stopping")
        if (
            not isinstance(early, Mapping)
            or early.get("enabled") is not True
            or early.get("monitor") != "validation_local_rr_mae_full_split"
            or int(early.get("min_epoch", -1)) != e7.EARLY_STOP_MIN_EPOCH
            or int(early.get("patience", -1)) != e7.EARLY_STOP_PATIENCE
            or float(early.get("min_delta", np.nan)) != e7.EARLY_STOP_MIN_DELTA
            or int(early.get("planned_epochs", -1)) != e7.EPOCHS
            or int(early.get("epochs_without_improvement", -1))
            != int(history_row.early_stopping_wait)
            or bool(early.get("triggered"))
            != bool(history_row.early_stopping_triggered)
        ):
            raise ValueError("E7 P4 checkpoint early stopping 合同漂移")
        if filename == "checkpoint_final.pt" and int(
            early.get("completed_epochs", -1)
        ) != final_epoch:
            raise ValueError("E7 P4 final checkpoint completed_epochs 漂移")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                state = optimizer.state.get(parameter, {})
                if (
                    not {"step", "exp_avg", "exp_avg_sq"}.issubset(state)
                    or float(state["step"])
                    != expected_epoch * e7.UPDATES_PER_EPOCH
                    or state["exp_avg"].shape != parameter.shape
                    or state["exp_avg_sq"].shape != parameter.shape
                ):
                    raise ValueError("E7 P4 optimizer state/step 不完整")
    metrics = pd.read_csv(run_dir / "metrics.csv")
    degeneracy = e7.validate_metrics(metrics, rows)
    arm = str(cfg.model.e7_factorial.arm)
    seed = int(cfg.training.seed)
    if not metrics.arm.eq(arm).all() or not metrics.seed.eq(seed).all():
        raise ValueError("E7 P4 metrics arm/seed identity 漂移")
    summary = pd.read_csv(run_dir / "metrics_summary.csv")
    expected_summary = summarize_task_metrics(metrics)
    if len(summary) != 1:
        raise ValueError("E7 P4 metrics summary 必须恰有一行")
    for metric in e7.PRIMARY:
        if (
            not np.isfinite(summary.iloc[0][metric + "_mean"])
            or not np.isclose(
                summary.iloc[0][metric + "_mean"],
                expected_summary.iloc[0][metric + "_mean"],
                atol=1e-12,
                rtol=0,
            )
            or int(summary.iloc[0][metric + "_n"])
            != int(expected_summary.iloc[0][metric + "_n"])
        ):
            raise ValueError(f"E7 P4 summary 数值/分母漂移: {metric}")
    return {
        "protocol": P4_PROTOCOL,
        "model_protocol": e7.PROTOCOL,
        "arm": arm,
        "seed": seed,
        "planned_epochs": e7.EPOCHS,
        "completed_epochs": final_epoch,
        "planned_updates": e7.EPOCHS * e7.UPDATES_PER_EPOCH,
        "completed_updates": final_epoch * e7.UPDATES_PER_EPOCH,
        "early_stopping_triggered": bool(history.iloc[-1].early_stopping_triggered),
        "selected_epoch": selected_epoch,
        "validation_rows": len(metrics),
        "prediction_degeneracy": degeneracy,
        "quality_acceptance_passed": not any(degeneracy.values()),
        "validation_row_order_sha256": array_hash(rows.dataset_row_id.to_numpy()),
        "selected_checkpoint": e7.identity(run_dir / "checkpoint_best_local_rr.pt"),
        "final_checkpoint": e7.identity(run_dir / "checkpoint_final.pt"),
        "metrics": e7.identity(run_dir / "metrics.csv"),
    }


def validate_anchor_rows(
    metrics: pd.DataFrame,
    p1_lock: Mapping[str, Any],
    seed: int,
) -> None:
    entry = next(item for item in p1_lock["w0_entries"] if int(item["seed"]) == seed)
    path = e7.ROOT / entry["run_dir"] / "metrics.csv"
    e7.verify(path, p1_lock["source_files"][str(path.relative_to(e7.ROOT))])
    reference = pd.read_csv(path).sort_values("dataset_row_id")
    candidate = metrics.sort_values("dataset_row_id")
    if reference.dataset_row_id.duplicated().any() or len(reference) != e7.COUNTS["val"]:
        raise ValueError("E7 P4 W0 validation 来源不完整")
    for key in (
        "dataset_row_id",
        "samp_id",
        "split",
        "whole_rr_target_eligible",
        "local_rr_target_eligible",
        "joint_target_eligible",
    ):
        if not np.array_equal(reference[key].to_numpy(), candidate[key].to_numpy()):
            raise ValueError(f"E7 P4 与 W0 validation identity/eligibility 不一致: {key}")


def run_formal(
    arm: str,
    seed: int,
    *,
    gpu_receipt: Path,
    device: str = "cuda:0",
) -> Path:
    if arm not in ARMS or seed not in e7.SEEDS:
        raise ValueError("E7 P4 arm/seed 不属于固定 18-cell 矩阵")
    p4_lock, p4_digest = load_p4_lock()
    p1_lock, _p1_digest = e7.load_implementation_lock()
    parent = e7.ROOT / e7.OUTPUT / "formal" / arm / f"seed_{seed}"
    with phase_guard(parent, p4_digest):
        with attempt(parent, p4_digest, arm, seed) as output:
            current_environment = runtime_preflight(device)
            e7.write_json(output / "environment.json", current_environment)
            verify_gpu_receipt(
                gpu_receipt,
                p4_lock,
                current_environment=current_environment,
            )
            e7.write_json(
                output / "gpu_acceptance_source.json",
                {
                    "path": str(gpu_receipt.resolve()),
                    "manifest": e7.identity(gpu_receipt / "manifest.json"),
                },
            )
            e7.write_json(output / "p4_execution_lock.json", p4_lock)
            baseline = OmegaConf.create(p1_lock["baselines"][str(seed)])
            frequencies = np.asarray(
                p1_lock["frequency"]["values_hz"], dtype=np.float64
            )
            cfg = e7.derived_config(
                baseline,
                arm,
                frequencies,
                output_root=output / "training",
                device=device,
            )
            e7.validate_config(
                cfg,
                baseline,
                arm,
                frequencies,
                output_root=output / "training",
                device=device,
            )
            rows = audit_sources(p1_lock, cfg, output)
            run_dir = e7.ScaleFactorialExperiment(cfg, rows["val"]).train()
            receipt = validate_run(run_dir, cfg, rows["val"])
            validate_anchor_rows(pd.read_csv(run_dir / "metrics.csv"), p1_lock, seed)
            receipt["run_dir"] = str(run_dir.relative_to(output))
            receipt["p4_execution_lock_sha256"] = p4_digest
            receipt["gpu_acceptance_manifest"] = e7.identity(
                gpu_receipt / "manifest.json"
            )
            e7.write_json(output / "formal_receipt.json", receipt)
    return output


def completed_runs() -> list[Path]:
    _lock, digest = load_p4_lock()
    runs: list[Path] = []
    for arm in ARMS:
        for seed in e7.SEEDS:
            parent = e7.ROOT / e7.OUTPUT / "formal" / arm / f"seed_{seed}"
            matches = []
            for freeze in parent.glob("*/freeze_receipt.json"):
                manifest = json.loads((freeze.parent / "manifest.json").read_text())
                if manifest.get("p4_execution_lock_sha256") == digest:
                    if manifest.get("arm") != arm or int(manifest.get("seed", -1)) != seed:
                        raise ValueError("E7 P4 完成目录 cell identity 漂移")
                    verify_formal_attempt(freeze.parent, lock_hash=digest)
                    matches.append(freeze.parent.resolve())
            if len(matches) != 1:
                raise ValueError(
                    f"E7 P4 {arm}/{seed} 需要唯一成功 attempt，实际 {len(matches)}"
                )
            runs.extend(matches)
    return runs
