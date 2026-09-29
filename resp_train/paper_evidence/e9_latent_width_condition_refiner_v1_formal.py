"""E9 formal train/validation 的实现锁、执行与矩阵状态。"""

from __future__ import annotations

import fcntl
import json
import os
import platform
import sys
import time
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
from omegaconf import DictConfig, OmegaConf

from resp_train.crd.config import check_crd_dependencies, crd_dependency_versions
from resp_train.crd.training import build_crd_optimizer
from resp_train.data.index import filter_index
from resp_train.data.research_v2 import read_research_v2_index
from resp_train.metrics.task import summarize_task_metrics
from resp_train.paper_evidence import e9_latent_width_condition_refiner_v1 as e9
from resp_train.paper_evidence import e9_latent_width_condition_refiner_v1_engineering as engineering
from resp_train.paper_evidence import w0_structural_factorial_v1 as sf
from resp_train.paper_evidence import w0_structural_factorial_v1_formal as sf_formal
from resp_train.paper_evidence.e9_latent_width_condition_refiner_v1_model import (
    ARMS,
    ARM_SPECS,
    build_e9_latent_width_condition_refiner_model,
)


FORMAL_PATH = Path(
    "resp_train/paper_evidence/e9_latent_width_condition_refiner_v1_formal.py"
)
FORMAL_AMENDMENT_PATH = Path(
    "docs/experiments/e9_latent_width_condition_refiner_v1_formal_runtime_amendment_20260929.json"
)
AMENDED_CODE_PATHS = (
    "docs/experiments/e9_latent_width_condition_refiner_v1_protocol_20260929.md",
    "resp_train/paper_evidence/e9_latent_width_condition_refiner_v1_formal.py",
    "scripts/run_e9_latent_width_condition_refiner_v1.py",
    "tests/test_e9_latent_width_condition_refiner_v1.py",
)
P2_ACCEPTANCE = Path(
    "runs/e9_latent_width_condition_refiner_v1/gpu_acceptance/"
    "gpu_acceptance_dfd2efe9fa1b_20260929T052057Z_bd9c52bd4c27"
)
P2_ENGINEERING_IDENTITY = (
    "dfd2efe9fa1be8834c607b157f960f48e779af7348b57ce6bb45697212a7fb48"
)
P2_MANIFEST_SHA256 = (
    "37bef5f86c28fe239bf0ea0d65eabea544c05afd008ba1e768016b0141bdbeaa"
)
P2_FILE_COUNT = 24
W0_SOURCE_LOCK_PATH = sf_formal.P2_LOCK_PATH
W0_SOURCE_LOCK_SHA256 = sf_formal.P2_LOCK_SHA256
COUNTS = {"train": 10_141, "val": 2_675}
SAMP_IDS = {"train": 32, "val": 7}


def verify_p2_evidence() -> dict[str, Any]:
    path = e9.SOURCE_ROOT / P2_ACCEPTANCE
    freeze = json.loads((path / "freeze_receipt.json").read_text(encoding="utf-8"))
    manifest_path = path / "manifest.json"
    if engineering._identity(manifest_path) != freeze.get("manifest"):
        raise RuntimeError("E9 P2 freeze receipt 身份错误")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("protocol") != e9.PROTOCOL
        or manifest.get("phase") != "gpu_acceptance"
        or manifest.get("status") != "completed"
        or manifest.get("engineering_identity_sha256") != P2_ENGINEERING_IDENTITY
        or engineering._sha256(manifest_path) != P2_MANIFEST_SHA256
        or len(manifest.get("files", {})) != P2_FILE_COUNT
    ):
        raise ValueError("E9 P2 manifest 合同漂移")
    for relative, expected in manifest["files"].items():
        target = (path / relative).resolve()
        if not target.is_relative_to(path.resolve()) or engineering._identity(target) != expected:
            raise RuntimeError(f"E9 P2 文件身份漂移: {relative}")
    acceptance = json.loads((path / "gpu_acceptance.json").read_text(encoding="utf-8"))
    batch1 = acceptance.get("batch1", [])
    expected_cells = {(arm, seed) for arm in ARMS for seed in e9.SEEDS}
    observed_cells = {(item.get("arm"), item.get("seed")) for item in batch1}
    physical = acceptance.get("max_resource_batch128", {})
    if (
        acceptance.get("passed") is not True
        or observed_cells != expected_cells
        or len(batch1) != len(expected_cells)
        or any(item.get("batch_size") != 1 or item.get("updates") != 3 for item in batch1)
        or any(
            any(value <= 0 for value in item.get("gradient_steps", [{}])[-1].values())
            for item in batch1
        )
        or any(
            any(value <= 0 for value in item.get("changed_by_factor", {}).values())
            for item in batch1
        )
        or physical.get("arm") != engineering.MAX_RESOURCE_ARM
        or physical.get("batch_size") != engineering.PHYSICAL_BATCH_SIZE
        or physical.get("updates") != engineering.PHYSICAL_BATCH_UPDATES
        or float(physical.get("peak_reserved_fraction", 1.0))
        > engineering.MEMORY_LIMIT_FRACTION
        or any(value <= 0 for value in physical.get("gradient_steps", [{}])[-1].values())
    ):
        raise ValueError("E9 P2 acceptance 合同不完整")
    environment = json.loads((path / "environment.json").read_text(encoding="utf-8"))
    if (
        environment.get("engineering_identity", {}).get("identity_sha256")
        != P2_ENGINEERING_IDENTITY
    ):
        raise ValueError("E9 P2 environment identity 漂移")
    return {
        "path": str(P2_ACCEPTANCE),
        "manifest": engineering._identity(manifest_path),
        "file_count": len(manifest["files"]),
        "engineering_identity_sha256": P2_ENGINEERING_IDENTITY,
        "environment": environment,
        "max_resource": physical,
    }


def _formal_code_paths() -> tuple[Path, ...]:
    return (
        e9.SPEC_PATH,
        e9.PROTOCOL_PATH,
        Path("resp_train/paper_evidence/e9_latent_width_condition_refiner_v1_model.py"),
        Path("resp_train/paper_evidence/e9_latent_width_condition_refiner_v1.py"),
        Path("resp_train/paper_evidence/e9_latent_width_condition_refiner_v1_engineering.py"),
        FORMAL_PATH,
        Path("scripts/run_e9_latent_width_condition_refiner_v1.py"),
        Path("resp_train/crd/model.py"),
        Path("resp_train/crd/tf_v1_model.py"),
        Path("resp_train/crd/frontends.py"),
        Path("resp_train/crd/blocks.py"),
        Path("resp_train/crd/experiment.py"),
        Path("resp_train/crd/training.py"),
        Path("resp_train/losses/task.py"),
        Path("resp_train/metrics/task.py"),
        Path("resp_train/paper_evidence/w0_structural_factorial_v1.py"),
        Path("resp_train/paper_evidence/w0_structural_factorial_v1_formal.py"),
        Path("tests/test_e9_latent_width_condition_refiner_v1.py"),
    )


def prepare_formal_lock() -> Path:
    destination = e9.ROOT / e9.FORMAL_LOCK_PATH
    if destination.exists():
        raise FileExistsError(f"E9 implementation lock 已存在: {destination}")
    state = engineering._git_state()
    if state["status_porcelain"]:
        raise RuntimeError("E9 implementation lock 要求干净 Git 工作树")
    e9.load_experiment_spec()
    p2 = verify_p2_evidence()
    source_lock, source_hash = sf_formal.load_formal_contract()
    if source_hash != W0_SOURCE_LOCK_SHA256:
        raise ValueError("E9 W0 train/validation 来源锁漂移")
    baselines = {
        str(seed): OmegaConf.to_container(e9.load_w0_baseline(seed), resolve=True)
        for seed in e9.SEEDS
    }
    lock = {
        "schema_version": 1,
        "protocol": e9.PROTOCOL,
        "status": "formal_execution_locked",
        "arms": list(ARMS),
        "arm_contracts": e9._arm_payload(),
        "seeds": list(e9.SEEDS),
        "counts": COUNTS,
        "samp_ids": SAMP_IDS,
        "epochs": e9.EPOCHS,
        "updates_per_epoch": e9.UPDATES_PER_EPOCH,
        "planned_updates": e9.PLANNED_UPDATES,
        "early_stopping": {
            "enabled": True,
            "min_epoch": e9.EARLY_STOP_MIN_EPOCH,
            "patience": e9.EARLY_STOP_PATIENCE,
            "min_delta": e9.EARLY_STOP_MIN_DELTA,
        },
        "spec": engineering._identity(e9.ROOT / e9.SPEC_PATH),
        "p2_evidence": p2,
        "w0_source_lock": {
            "path": str(W0_SOURCE_LOCK_PATH),
            "sha256": source_hash,
            "source_file_count": len(source_lock["source_files"]),
        },
        "baselines": baselines,
        "code_files": {
            str(path): engineering._identity(e9.ROOT / path)
            for path in _formal_code_paths()
        },
        "source_repository_root": str(e9.SOURCE_ROOT),
        "artifact_root": str(e9.SOURCE_ROOT / e9.OUTPUT_ROOT),
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "preparation_git": state,
    }
    engineering._write_json(destination, lock)
    return destination


def _base_formal_lock() -> tuple[dict[str, Any], str]:
    path = e9.ROOT / e9.FORMAL_LOCK_PATH
    if not path.is_file():
        raise FileNotFoundError(f"E9 implementation lock 尚未建立: {path}")
    lock_hash = engineering._sha256(path)
    lock = json.loads(path.read_text(encoding="utf-8"))
    if (
        lock.get("schema_version") != 1
        or lock.get("protocol") != e9.PROTOCOL
        or lock.get("status") != "formal_execution_locked"
        or tuple(lock.get("arms", ())) != ARMS
        or tuple(lock.get("seeds", ())) != e9.SEEDS
        or lock.get("arm_contracts") != e9._arm_payload()
        or lock.get("counts") != COUNTS
        or lock.get("samp_ids") != SAMP_IDS
        or lock.get("epochs") != e9.EPOCHS
        or lock.get("updates_per_epoch") != e9.UPDATES_PER_EPOCH
        or lock.get("planned_updates") != e9.PLANNED_UPDATES
        or lock.get("source_repository_root") != str(e9.SOURCE_ROOT)
        or lock.get("artifact_root") != str(e9.SOURCE_ROOT / e9.OUTPUT_ROOT)
    ):
        raise ValueError("E9 implementation lock 科学合同漂移")
    return lock, lock_hash


def _pre_amendment_matrix_snapshot(lock_hash: str) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for cell in formal_plan():
        parent = Path(cell["output_parent"])
        completed: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        running: list[str] = []
        if parent.is_dir():
            for attempt in sorted(path for path in parent.iterdir() if path.is_dir()):
                started_path = attempt / "lifecycle_started.json"
                if not started_path.is_file():
                    continue
                context = json.loads(started_path.read_text(encoding="utf-8"))
                if context.get("implementation_lock_sha256") != lock_hash:
                    continue
                if (attempt / "freeze_receipt.json").is_file():
                    completed.append(
                        {
                            "path": str(attempt),
                            "manifest": engineering._identity(attempt / "manifest.json"),
                        }
                    )
                elif (attempt / "lifecycle_failed.json").is_file():
                    failed.append(
                        {
                            "path": str(attempt),
                            "failure": engineering._identity(
                                attempt / "lifecycle_failed.json"
                            ),
                        }
                    )
                else:
                    running.append(str(attempt))
        if running:
            raise RuntimeError(
                f"E9 amendment 前仍有运行 cell: {cell['arm']}/{cell['seed']}"
            )
        if len(completed) > 1:
            raise RuntimeError(
                f"E9 amendment 前存在重复成功 cell: {cell['arm']}/{cell['seed']}"
            )
        status = "completed" if completed else "failed" if failed else "pending"
        cells.append(
            {
                "arm": cell["arm"],
                "seed": cell["seed"],
                "status": status,
                "completed": completed,
                "failed": failed,
            }
        )
    counts = {
        status: sum(cell["status"] == status for cell in cells)
        for status in ("pending", "failed", "completed")
    }
    return {"counts": counts, "cells": cells}


def prepare_formal_amendment() -> Path:
    """冻结只影响 post-training validation schema 的增量修订。"""

    destination = e9.ROOT / FORMAL_AMENDMENT_PATH
    if destination.exists():
        raise FileExistsError(f"E9 formal amendment 已存在: {destination}")
    state = engineering._git_state()
    if state["status_porcelain"]:
        raise RuntimeError("E9 formal amendment 要求干净 Git 工作树")
    lock, lock_hash = _base_formal_lock()
    amended = set(AMENDED_CODE_PATHS)
    for relative, expected in lock["code_files"].items():
        if relative not in amended and engineering._identity(e9.ROOT / relative) != expected:
            raise ValueError(f"E9 amendment 范围外代码漂移: {relative}")
    p2 = verify_p2_evidence()
    if p2 != lock["p2_evidence"]:
        raise ValueError("E9 amendment P2 evidence 漂移")
    source_lock, source_hash = sf_formal.load_formal_contract()
    if source_hash != lock["w0_source_lock"]["sha256"]:
        raise ValueError("E9 amendment W0 source lock 漂移")
    for seed in e9.SEEDS:
        current = OmegaConf.to_container(e9.load_w0_baseline(seed), resolve=True)
        if current != lock["baselines"][str(seed)]:
            raise ValueError(f"E9 amendment baseline 漂移: {seed}")
    snapshot = _pre_amendment_matrix_snapshot(lock_hash)
    amendment = {
        "schema_version": 1,
        "protocol": e9.PROTOCOL,
        "status": "formal_post_training_validation_amendment_locked",
        "base_formal_lock": {
            "path": str(e9.FORMAL_LOCK_PATH),
            "sha256": lock_hash,
        },
        "scope": "post_training_validation_history_runtime_fields_only",
        "scientific_contract_changed": False,
        "training_or_model_code_changed": False,
        "validation_policy": {
            "runtime_summary_required": True,
            "history_runtime_columns": "validate_if_present",
            "required_history_columns": [
                "epoch",
                "optimizer_update",
                "train_loss_total",
                "train_loss_sync",
                "train_loss_effort",
                "first_learning_rate",
                "last_learning_rate",
                "val_core_loss",
                "val_local_rr_mae",
            ],
        },
        "pre_amendment_matrix": snapshot,
        "amended_code_files": {
            relative: {
                "base": lock["code_files"][relative],
                "revised": engineering._identity(e9.ROOT / relative),
            }
            for relative in AMENDED_CODE_PATHS
        },
        "p2_evidence": p2,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "preparation_git": state,
    }
    engineering._write_json(destination, amendment)
    return destination


def load_formal_amendment(
    lock: Mapping[str, Any], lock_hash: str
) -> tuple[dict[str, Any], str]:
    path = e9.ROOT / FORMAL_AMENDMENT_PATH
    if not path.is_file():
        raise FileNotFoundError(f"E9 formal amendment 尚未建立: {path}")
    amendment = json.loads(path.read_text(encoding="utf-8"))
    amendment_hash = engineering._sha256(path)
    if (
        amendment.get("schema_version") != 1
        or amendment.get("protocol") != e9.PROTOCOL
        or amendment.get("status")
        != "formal_post_training_validation_amendment_locked"
        or amendment.get("base_formal_lock", {}).get("sha256") != lock_hash
        or amendment.get("scope")
        != "post_training_validation_history_runtime_fields_only"
        or amendment.get("scientific_contract_changed") is not False
        or amendment.get("training_or_model_code_changed") is not False
        or amendment.get("p2_evidence") != lock["p2_evidence"]
    ):
        raise ValueError("E9 formal amendment 合同漂移")
    for relative in AMENDED_CODE_PATHS:
        entry = amendment.get("amended_code_files", {}).get(relative, {})
        if (
            entry.get("base") != lock["code_files"][relative]
            or entry.get("revised") != engineering._identity(e9.ROOT / relative)
        ):
            raise ValueError(f"E9 formal amendment 代码身份漂移: {relative}")
    for cell in amendment["pre_amendment_matrix"]["cells"]:
        for failed in cell["failed"]:
            if engineering._identity(
                Path(failed["path"]) / "lifecycle_failed.json"
            ) != failed["failure"]:
                raise ValueError(f"E9 amendment 前失败产物漂移: {failed['path']}")
    return amendment, amendment_hash


def load_formal_lock() -> tuple[dict[str, Any], str, dict[str, Any]]:
    lock, lock_hash = _base_formal_lock()
    if engineering._identity(e9.ROOT / e9.SPEC_PATH) != lock["spec"]:
        raise ValueError("E9 formal spec 身份漂移")
    amendment, amendment_hash = load_formal_amendment(lock, lock_hash)
    amended = set(AMENDED_CODE_PATHS)
    for relative, expected in lock["code_files"].items():
        if relative not in amended and engineering._identity(e9.ROOT / relative) != expected:
            raise ValueError(f"E9 formal 代码身份漂移: {relative}")
    if verify_p2_evidence() != lock["p2_evidence"]:
        raise ValueError("E9 formal P2 evidence 漂移")
    source_lock, source_hash = sf_formal.load_formal_contract()
    if source_hash != lock["w0_source_lock"]["sha256"]:
        raise ValueError("E9 formal W0 source lock 漂移")
    for seed in e9.SEEDS:
        current = OmegaConf.to_container(e9.load_w0_baseline(seed), resolve=True)
        if current != lock["baselines"][str(seed)]:
            raise ValueError(f"E9 formal baseline 漂移: {seed}")
    lock = dict(lock)
    lock["_runtime_amendment"] = amendment
    lock["_runtime_amendment_sha256"] = amendment_hash
    return lock, lock_hash, source_lock


def runtime_preflight(device: str) -> dict[str, Any]:
    state = engineering._git_state()
    if state["status_porcelain"]:
        raise RuntimeError("E9 formal 要求干净 Git 工作树")
    resolved = torch.device(device)
    if resolved.type != "cuda" or resolved.index is None or not torch.cuda.is_available():
        raise RuntimeError("E9 formal 要求显式可用的 cuda:<index>")
    problems = check_crd_dependencies()
    if problems:
        raise RuntimeError("E9 formal 原生依赖检查失败: " + "; ".join(problems))
    torch.cuda.set_device(resolved)
    properties = torch.cuda.get_device_properties(resolved)
    return {
        "git": state,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "dependencies": crd_dependency_versions(),
        "device": str(resolved),
        "device_name": properties.name,
        "device_total_bytes": int(properties.total_memory),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "amp_dtype": "bfloat16",
    }


def _runtime_compatibility(
    runtime: Mapping[str, Any], accepted: Mapping[str, Any]
) -> dict[str, Any]:
    matched = (
        "python",
        "torch",
        "cuda_runtime",
        "cudnn",
        "dependencies",
        "device_name",
        "amp_dtype",
    )
    for key in matched:
        if runtime.get(key) != accepted.get(key):
            raise ValueError(f"E9 formal runtime 与 P2 不一致: {key}")
    current_total = int(runtime["device_total_bytes"])
    accepted_total = int(accepted["device_total_bytes"])
    if current_total < accepted_total:
        raise ValueError("E9 formal GPU 总显存低于 P2 验收设备")
    return {
        "matched_fields": list(matched),
        "capacity_policy": "current_device_total_bytes_gte_p2",
        "p2_device_total_bytes": accepted_total,
        "current_device_total_bytes": current_total,
    }


def audit_sources(
    source_lock: Mapping[str, Any], cfg: DictConfig, output: Path
) -> dict[str, pd.DataFrame]:
    engineering._write_json(
        output / "access_started.json",
        {
            "purpose": "E9 formal train/validation",
            "splits": ["train", "val"],
            "research_test_evaluation": False,
            "dataset_index": source_lock["dataset_index"],
            "source_files": list(source_lock["source_files"]),
        },
    )
    for relative, expected in source_lock["source_files"].items():
        sf.verify_identity(e9.SOURCE_ROOT / relative, expected)
    index = source_lock["dataset_index"]
    if sf.sha256_file(Path(index["path"])) != index["sha256"]:
        raise ValueError("E9 formal dataset index 身份漂移")
    audited = read_research_v2_index(cfg.data.dataset_root, cfg.data.index_csv, cfg)
    rows: dict[str, pd.DataFrame] = {}
    for split, count in COUNTS.items():
        frame = filter_index(
            audited,
            cfg,
            split=split,
            max_windows=None,
            sample_strategy=str(cfg.data[f"{split}_sample_strategy"]),
            sample_seed=int(cfg.data[f"{split}_sample_seed"]),
        )
        expected_hash = source_lock["cache_lock"]["row_identity"][
            f"{split}_row_content_sha256"
        ]
        if (
            len(frame) != count
            or frame.dataset_row_id.duplicated().any()
            or set(frame.split.astype(str)) != {split}
            or frame.samp_id.nunique() != SAMP_IDS[split]
            or sf_formal.array_hash(np.sort(frame.dataset_row_id.to_numpy())) != expected_hash
        ):
            raise ValueError(f"E9 formal {split} row 合同漂移")
        frame.to_csv(output / f"{split}_rows.csv", index=False)
        rows[split] = frame
    if set(rows["train"].samp_id) & set(rows["val"].samp_id):
        raise ValueError("E9 formal train/validation subject 隔离失败")
    engineering._write_json(
        output / "access_receipt.json",
        {
            "counts": COUNTS,
            "samp_ids": SAMP_IDS,
            "row_order_sha256": {
                split: sf_formal.array_hash(frame.dataset_row_id.to_numpy())
                for split, frame in rows.items()
            },
            "sample_seeds": {
                split: int(cfg.data[f"{split}_sample_seed"]) for split in COUNTS
            },
            "dataset_index": index,
            "source_files_verified": source_lock["source_files"],
            "evaluation_splits": ["train", "val"],
            "research_test_evaluation": False,
        },
    )
    return rows


class FormalE9Experiment(e9.E9LatentWidthConditionRefinerExperiment):
    def __init__(
        self,
        cfg: DictConfig,
        validation_rows: pd.DataFrame,
        arm: str,
        initialization_path: Path,
    ) -> None:
        self.validation_rows = validation_rows
        self.arm = arm
        self.initialization_path = initialization_path
        self.task_name = arm
        super().__init__(cfg)

    def _build_model(self):
        model = build_e9_latent_width_condition_refiner_model(self.cfg)
        identity = sf_formal.state_dict_identity(model.state_dict())
        identity.update(
            {
                "arm": self.arm,
                "seed": int(self.cfg.training.seed),
                "trainable_parameters": ARM_SPECS[self.arm].trainable_parameters,
            }
        )
        engineering._write_json(self.initialization_path, identity)
        return model

    def _evaluate_model(self, model, loader, **kwargs):
        frame = super()._evaluate_model(model, loader, **kwargs)
        sf.validate_metrics(frame, self.validation_rows)
        frame.insert(0, "arm", self.arm)
        frame.insert(0, "seed", int(self.cfg.training.seed))
        return frame


def _checkpoint_contract(
    checkpoint: Mapping[str, Any],
    *,
    expected_epoch: int,
    history_row: pd.Series,
    cfg: DictConfig,
) -> None:
    engineering.finite_tree(checkpoint, label="checkpoint")
    if (
        checkpoint["epoch"] != expected_epoch
        or checkpoint["config"] != OmegaConf.to_container(cfg, resolve=True)
    ):
        raise ValueError("E9 formal checkpoint epoch/config 漂移")
    for key in history_row.index:
        if key not in checkpoint["metrics"] or not np.isclose(
            float(checkpoint["metrics"][key]),
            float(history_row[key]),
            atol=1e-12,
            rtol=1e-12,
        ):
            raise ValueError(f"E9 formal checkpoint/history 不一致: {key}")
    extra = checkpoint["extra_state"]
    if (
        extra.get("protocol") != e9.PROTOCOL
        or int(extra.get("update_index", -1)) != expected_epoch * e9.UPDATES_PER_EPOCH
        or int(extra.get("total_updates", -1)) != e9.PLANNED_UPDATES
    ):
        raise ValueError("E9 formal checkpoint update/protocol 漂移")
    early = extra.get("early_stopping")
    if (
        not isinstance(early, Mapping)
        or early.get("enabled") is not True
        or int(early.get("min_epoch", -1)) != e9.EARLY_STOP_MIN_EPOCH
        or int(early.get("patience", -1)) != e9.EARLY_STOP_PATIENCE
        or float(early.get("min_delta", np.nan)) != e9.EARLY_STOP_MIN_DELTA
        or int(early.get("planned_epochs", -1)) != e9.EPOCHS
    ):
        raise ValueError("E9 formal checkpoint early-stopping 合同漂移")


def validate_formal_run(
    run_dir: Path,
    cfg: DictConfig,
    rows: pd.DataFrame,
    source_lock: Mapping[str, Any],
    *,
    arm: str,
    initialization_path: Path,
) -> dict[str, Any]:
    saved = OmegaConf.load(run_dir / "config.yaml")
    if OmegaConf.to_container(saved, resolve=True) != OmegaConf.to_container(cfg, resolve=True):
        raise ValueError("E9 formal 保存配置漂移")
    history = pd.read_csv(run_dir / "train_history.csv")
    best_epoch = sf.validate_history(history, cfg)
    final_epoch = int(history.iloc[-1].epoch)
    _validate_optional_history_runtime(history)
    model = build_e9_latent_width_condition_refiner_model(cfg)
    expected_initialization = sf_formal.state_dict_identity(model.state_dict())
    saved_initialization = json.loads(initialization_path.read_text(encoding="utf-8"))
    if saved_initialization.get("state_sha256") != expected_initialization["state_sha256"]:
        raise ValueError("E9 formal initialization identity 漂移")
    optimizer, partition = build_crd_optimizer(model, cfg)
    groups = json.loads((run_dir / "optimizer_parameter_groups.json").read_text(encoding="utf-8"))
    if groups != {
        "weight_decay": float(cfg.training.weight_decay),
        "decay": list(partition.decay_names),
        "no_decay": list(partition.no_decay_names),
    }:
        raise ValueError("E9 formal optimizer 参数分组漂移")
    for filename, epoch in (
        ("checkpoint_best_local_rr.pt", best_epoch),
        ("checkpoint_final.pt", final_epoch),
    ):
        checkpoint = torch.load(run_dir / filename, map_location="cpu", weights_only=False)
        history_row = history.loc[history.epoch.eq(epoch)].iloc[0]
        _checkpoint_contract(
            checkpoint,
            expected_epoch=epoch,
            history_row=history_row,
            cfg=cfg,
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                state = optimizer.state.get(parameter, {})
                if (
                    not {"step", "exp_avg", "exp_avg_sq"}.issubset(state)
                    or float(state["step"]) != epoch * e9.UPDATES_PER_EPOCH
                    or state["exp_avg"].shape != parameter.shape
                    or state["exp_avg_sq"].shape != parameter.shape
                ):
                    raise ValueError("E9 formal optimizer state/step 不完整")
    metrics = pd.read_csv(run_dir / "metrics.csv")
    degeneracy = sf.validate_metrics(metrics, rows)
    if not metrics.arm.eq(arm).all() or not metrics.seed.eq(int(cfg.training.seed)).all():
        raise ValueError("E9 formal metrics arm/seed identity 漂移")
    sf_formal.validate_anchor_rows(metrics, source_lock, int(cfg.training.seed))
    summary = pd.read_csv(run_dir / "metrics_summary.csv")
    expected_summary = summarize_task_metrics(metrics)
    if len(summary) != 1:
        raise ValueError("E9 formal metrics summary 必须恰有一行")
    for metric in sf.PRIMARY:
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
            raise ValueError(f"E9 formal summary 数值/分母漂移: {metric}")
    runtime = json.loads((run_dir / "runtime_summary.json").read_text(encoding="utf-8"))
    if runtime.get("peak_reserved_fraction") is None or not np.isfinite(
        runtime["peak_reserved_fraction"]
    ):
        raise FloatingPointError("E9 formal runtime summary 不完整")
    return {
        "protocol": e9.PROTOCOL,
        "arm": arm,
        "seed": int(cfg.training.seed),
        "planned_epochs": e9.EPOCHS,
        "completed_epochs": final_epoch,
        "planned_updates": e9.PLANNED_UPDATES,
        "completed_updates": final_epoch * e9.UPDATES_PER_EPOCH,
        "selected_epoch": best_epoch,
        "validation_rows": len(metrics),
        "prediction_degeneracy": degeneracy,
        "validation_row_order_sha256": sf_formal.array_hash(rows.dataset_row_id.to_numpy()),
        "initialization_state_sha256": saved_initialization["state_sha256"],
        "runtime": runtime,
    }


def _validate_optional_history_runtime(history: pd.DataFrame) -> None:
    """E9 stage 的 runtime 位于 runtime_summary；兼容可选的逐 epoch runtime 列。"""

    columns = {"train_elapsed_seconds", "train_samples_per_second"}
    present = columns.intersection(history.columns)
    if present and present != columns:
        raise ValueError("E9 formal history runtime 列必须同时存在")
    if present and not np.isfinite(history[sorted(columns)].to_numpy()).all():
        raise FloatingPointError("E9 formal history runtime 非有限")


@contextmanager
def _formal_attempt(
    parent: Path,
    *,
    lock_hash: str,
    amendment_hash: str,
    arm: str,
    seed: int,
) -> Iterator[Path]:
    parent.mkdir(parents=True, exist_ok=True)
    lock_path = parent / f".execution_{lock_hash}.lock"
    with lock_path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("相同 E9 formal cell 正在运行") from exc
        for receipt_path in parent.glob("*/freeze_receipt.json"):
            manifest_path = receipt_path.parent / "manifest.json"
            freeze = json.loads(receipt_path.read_text(encoding="utf-8"))
            if engineering._identity(manifest_path) != freeze.get("manifest"):
                raise RuntimeError("E9 formal 已完成 attempt 身份错误")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("implementation_lock_sha256") == lock_hash
                and manifest.get("arm") == arm
                and int(manifest.get("seed", -1)) == seed
                and manifest.get("status") == "completed"
            ):
                raise FileExistsError(f"E9 formal cell 已完成: {arm}/{seed}")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = parent / f"formal_{lock_hash[:12]}_{stamp}_{uuid4().hex[:12]}"
        output.mkdir(exist_ok=False)
        context = {
            "protocol": e9.PROTOCOL,
            "phase": "formal",
            "arm": arm,
            "seed": seed,
            "implementation_lock_sha256": lock_hash,
            "formal_runtime_amendment_sha256": amendment_hash,
            "command": sys.argv,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        engineering._write_json(output / "lifecycle_started.json", {**context, "status": "running"})
        try:
            yield output
            engineering._write_json(
                output / "lifecycle_completed.json",
                {
                    **context,
                    "status": "completed",
                    "ended_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            manifest = {
                **context,
                "status": "completed",
                "files": {
                    str(file.relative_to(output)): engineering._identity(file)
                    for file in sorted(output.rglob("*"))
                    if file.is_file()
                },
            }
            engineering._write_json(output / "manifest.json", manifest)
            engineering._write_json(
                output / "freeze_receipt.json",
                {"protocol": e9.PROTOCOL, "manifest": engineering._identity(output / "manifest.json")},
            )
        except BaseException as exc:
            engineering._write_json(
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
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run_formal(arm: str, seed: int, *, device: str = "cuda:0") -> Path:
    if arm not in ARMS or int(seed) not in e9.SEEDS:
        raise ValueError("E9 formal arm/seed 不属于冻结矩阵")
    seed = int(seed)
    lock, lock_hash, source_lock = load_formal_lock()
    amendment = lock["_runtime_amendment"]
    amendment_hash = str(lock["_runtime_amendment_sha256"])
    parent = e9.SOURCE_ROOT / e9.OUTPUT_ROOT / "formal" / arm / f"seed_{seed}"
    with _formal_attempt(
        parent,
        lock_hash=lock_hash,
        amendment_hash=amendment_hash,
        arm=arm,
        seed=seed,
    ) as output:
        runtime = runtime_preflight(device)
        compatibility = _runtime_compatibility(
            runtime, lock["p2_evidence"]["environment"]
        )
        engineering._write_json(output / "environment.json", runtime)
        engineering._write_json(
            output / "p2_source.json",
            {"evidence": lock["p2_evidence"], "runtime_compatibility": compatibility},
        )
        engineering._write_json(
            output / "implementation_lock.json",
            {key: value for key, value in lock.items() if not key.startswith("_")},
        )
        engineering._write_json(output / "formal_runtime_amendment.json", amendment)
        baseline = OmegaConf.create(lock["baselines"][str(seed)])
        cfg = e9.derived_config(
            baseline,
            arm=arm,
            output_root=output / "training",
            device=device,
        )
        e9.validate_config(
            cfg,
            baseline,
            arm=arm,
            output_root=output / "training",
            device=device,
        )
        rows = audit_sources(source_lock, cfg, output)
        initialization_path = output / "initialization.json"
        experiment = FormalE9Experiment(cfg, rows["val"], arm, initialization_path)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        run_dir = experiment.train()
        torch.cuda.synchronize(device)
        wall_seconds = time.perf_counter() - started
        if wall_seconds <= 0 or not np.isfinite(wall_seconds):
            raise RuntimeError("E9 formal wall time 非法")
        receipt = validate_formal_run(
            run_dir,
            cfg,
            rows["val"],
            source_lock,
            arm=arm,
            initialization_path=initialization_path,
        )
        engineering._write_json(
            output / "formal_receipt.json",
            {
                **receipt,
                "implementation_lock_sha256": lock_hash,
                "formal_runtime_amendment_sha256": amendment_hash,
                "run_dir": str(run_dir.relative_to(output)),
                "formal_wall_seconds": wall_seconds,
            },
        )
    return output


def formal_plan() -> list[dict[str, Any]]:
    return [
        {
            "arm": arm,
            "seed": seed,
            "output_parent": str(
                e9.SOURCE_ROOT / e9.OUTPUT_ROOT / "formal" / arm / f"seed_{seed}"
            ),
        }
        for seed in e9.SEEDS
        for arm in ARMS
    ]


def verify_formal_attempt(
    path: Path,
    *,
    lock_hash: str,
    amendment_hash: str,
    arm: str,
    seed: int,
) -> dict[str, Any]:
    freeze = json.loads((path / "freeze_receipt.json").read_text(encoding="utf-8"))
    manifest_path = path / "manifest.json"
    if engineering._identity(manifest_path) != freeze.get("manifest"):
        raise RuntimeError("E9 formal freeze receipt 身份错误")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("protocol") != e9.PROTOCOL
        or manifest.get("phase") != "formal"
        or manifest.get("status") != "completed"
        or manifest.get("implementation_lock_sha256") != lock_hash
        or manifest.get("formal_runtime_amendment_sha256") != amendment_hash
        or manifest.get("arm") != arm
        or int(manifest.get("seed", -1)) != int(seed)
    ):
        raise ValueError("E9 formal manifest 合同漂移")
    for relative, expected in manifest["files"].items():
        target = (path / relative).resolve()
        if not target.is_relative_to(path.resolve()) or engineering._identity(target) != expected:
            raise RuntimeError(f"E9 formal 文件身份漂移: {relative}")
    receipt = json.loads((path / "formal_receipt.json").read_text(encoding="utf-8"))
    if (
        receipt.get("implementation_lock_sha256") != lock_hash
        or receipt.get("formal_runtime_amendment_sha256") != amendment_hash
        or receipt.get("arm") != arm
        or int(receipt.get("seed", -1)) != int(seed)
        or int(receipt.get("validation_rows", -1)) != COUNTS["val"]
    ):
        raise ValueError("E9 formal receipt 合同漂移")
    return manifest


def matrix_status() -> dict[str, Any]:
    lock, lock_hash, _source = load_formal_lock()
    amendment_hash = str(lock["_runtime_amendment_sha256"])
    cells: list[dict[str, Any]] = []
    for cell in formal_plan():
        parent = Path(cell["output_parent"])
        completed: list[str] = []
        failed: list[str] = []
        running: list[str] = []
        if parent.is_dir():
            for attempt in sorted(path for path in parent.iterdir() if path.is_dir()):
                started = attempt / "lifecycle_started.json"
                if not started.is_file():
                    continue
                context = json.loads(started.read_text(encoding="utf-8"))
                if context.get("implementation_lock_sha256") != lock_hash:
                    continue
                if (attempt / "freeze_receipt.json").is_file():
                    verify_formal_attempt(
                        attempt,
                        lock_hash=lock_hash,
                        amendment_hash=amendment_hash,
                        arm=cell["arm"],
                        seed=cell["seed"],
                    )
                    completed.append(str(attempt))
                elif (attempt / "lifecycle_failed.json").is_file():
                    failed.append(str(attempt))
                else:
                    running.append(str(attempt))
        if len(completed) > 1:
            raise RuntimeError(f"E9 formal cell 多个成功 attempt: {cell['arm']}/{cell['seed']}")
        status = "completed" if completed else "running" if running else "failed" if failed else "pending"
        cells.append(
            {
                **cell,
                "status": status,
                "completed": completed,
                "failed": failed,
                "running": running,
            }
        )
    counts = {
        status: sum(cell["status"] == status for cell in cells)
        for status in ("pending", "running", "failed", "completed")
    }
    return {
        "protocol": e9.PROTOCOL,
        "implementation_lock_sha256": lock_hash,
        "formal_runtime_amendment_sha256": amendment_hash,
        "counts": counts,
        "cells": cells,
    }
