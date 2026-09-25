"""E8 formal train/validation 的锁定、执行与矩阵状态。"""

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
from resp_train.paper_evidence import e8_film_decoder_redesign_v1 as e8
from resp_train.paper_evidence import e8_film_decoder_redesign_v1_engineering as engineering
from resp_train.paper_evidence import w0_structural_factorial_v1 as sf
from resp_train.paper_evidence import w0_structural_factorial_v1_formal as sf_formal
from resp_train.paper_evidence.e8_film_decoder_redesign_v1_model import (
    ARMS,
    ARM_SPECS,
    build_e8_film_decoder_redesign_model,
)


FORMAL_PATH = Path("resp_train/paper_evidence/e8_film_decoder_redesign_v1_formal.py")
FORMAL_LOCK_PATH = Path(
    "docs/experiments/e8_film_decoder_redesign_v1_formal_execution_lock_20260925.json"
)
FORMAL_LOCK_SHA256 = "40b30fc6ecdcc9b40750c393be8fd2faf4e223da0566432df739aacb513713c5"
FORMAL_AMENDMENT_PATH = Path(
    "docs/experiments/e8_film_decoder_redesign_v1_formal_runtime_amendment_20260926.json"
)
AMENDED_CODE_PATHS = (
    str(FORMAL_PATH),
    "tests/test_e8_film_decoder_redesign_v1.py",
)
MIN_FORMAL_DEVICE_TOTAL_BYTES = 16_710_500_352
P2_MAX_PEAK_RESERVED_BYTES = 10_643_046_400
P2_CLOSEOUT_PATH = Path(
    "docs/experiments/e8_film_decoder_redesign_v1_p2_closeout_20260925.md"
)
P2_CLOSEOUT_SHA256 = "7e83f2045a586a0b6f7aabfcf4a3b6a71a3662ce9c397c600394dd99b3402010"
W0_SOURCE_LOCK_PATH = sf_formal.P2_LOCK_PATH
W0_SOURCE_LOCK_SHA256 = sf_formal.P2_LOCK_SHA256

P2_ACCEPTANCE = Path(
    "runs/e8_film_decoder_redesign_v1/gpu_acceptance/"
    "gpu_acceptance_aa80977f0c73_20260925T115916Z_e20ea0dd8f28"
)
P2_BENCHMARK = Path(
    "runs/e8_film_decoder_redesign_v1/benchmark/"
    "benchmark_aa80977f0c73_20260925T120044Z_4ddffe1e62b0"
)
P2_MANIFEST_SHA256 = {
    "gpu_acceptance": "a169d42fca40585a55eeb58de9dbb5b9bd3decd21429f9eeef9458b3238c041b",
    "benchmark": "23f4d285d586df25227ddc33c5d5751089de03ee7b7bc278523cacabd9cce9dc",
}
P2_FILE_COUNTS = {"gpu_acceptance": 54, "benchmark": 53}

COUNTS = {"train": 10_141, "val": 2_675}
SAMP_IDS = {"train": 32, "val": 7}


def _verify_frozen_attempt(path: Path, *, phase: str) -> dict[str, Any]:
    path = path.resolve()
    freeze = json.loads((path / "freeze_receipt.json").read_text(encoding="utf-8"))
    manifest_path = path / "manifest.json"
    if engineering._identity(manifest_path) != freeze.get("manifest"):
        raise RuntimeError(f"E8 P2 {phase} freeze receipt 身份错误")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("protocol") != e8.PROTOCOL
        or manifest.get("phase") != phase
        or manifest.get("status") != "completed"
        or manifest.get("engineering_identity_sha256")
        != "aa80977f0c737906bdf011bd9b39367d1c77b39130404a39db9c1f6b2670f3d4"
        or engineering._sha256(manifest_path) != P2_MANIFEST_SHA256[phase]
        or len(manifest.get("files", {})) != P2_FILE_COUNTS[phase]
    ):
        raise ValueError(f"E8 P2 {phase} manifest 合同漂移")
    for relative, expected in manifest["files"].items():
        target = (path / relative).resolve()
        if not target.is_relative_to(path) or engineering._identity(target) != expected:
            raise RuntimeError(f"E8 P2 {phase} 文件身份漂移: {relative}")
    return manifest


def verify_p2_evidence() -> dict[str, Any]:
    closeout = e8.ROOT / P2_CLOSEOUT_PATH
    if engineering._sha256(closeout) != P2_CLOSEOUT_SHA256:
        raise ValueError("E8 P2 closeout 身份漂移")
    acceptance_path = e8.SOURCE_ROOT / P2_ACCEPTANCE
    benchmark_path = e8.SOURCE_ROOT / P2_BENCHMARK
    acceptance_manifest = _verify_frozen_attempt(
        acceptance_path,
        phase="gpu_acceptance",
    )
    benchmark_manifest = _verify_frozen_attempt(benchmark_path, phase="benchmark")
    acceptance = json.loads((acceptance_path / "gpu_acceptance.json").read_text(encoding="utf-8"))
    batch1 = acceptance.get("batch1", [])
    expected_cells = {(arm, seed) for arm in ARMS for seed in e8.SEEDS}
    observed_cells = {(item.get("arm"), item.get("seed")) for item in batch1}
    physical = acceptance.get("max_resource_batch128", {})
    lifecycle = acceptance.get("native_lifecycle", {})
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
        or physical.get("batch_size") != 128
        or physical.get("updates") != 3
        or float(physical.get("peak_reserved_fraction", 1.0)) > 0.8
        or lifecycle.get("train_windows") != 128
        or lifecycle.get("validation_windows") != 32
        or lifecycle.get("validation_metric_rows") != 32
    ):
        raise ValueError("E8 P2 GPU acceptance 合同不完整")
    benchmark = json.loads((benchmark_path / "benchmark.json").read_text(encoding="utf-8"))
    measurements = benchmark.get("measurements", [])
    expected_measurements = {(arm, mode) for arm in ARMS for mode in ("eval", "train")}
    observed_measurements = {
        (item.get("arm"), item.get("mode")) for item in measurements
    }
    if (
        observed_measurements != expected_measurements
        or len(measurements) != len(expected_measurements)
        or any(item.get("warmup") != 5 or item.get("repeats") != 20 for item in measurements)
        or any(float(item.get("peak_reserved_fraction", 1.0)) > 0.8 for item in measurements)
    ):
        raise ValueError("E8 P2 benchmark 合同不完整")
    acceptance_environment = json.loads(
        (acceptance_path / "environment.json").read_text(encoding="utf-8")
    )
    benchmark_environment = json.loads(
        (benchmark_path / "environment.json").read_text(encoding="utf-8")
    )
    for key in (
        "python",
        "torch",
        "cuda_runtime",
        "cudnn",
        "dependencies",
        "device_name",
        "device_total_bytes",
        "amp_dtype",
    ):
        if acceptance_environment.get(key) != benchmark_environment.get(key):
            raise ValueError(f"E8 P2 acceptance/benchmark 环境不一致: {key}")
    return {
        "closeout": {"path": str(P2_CLOSEOUT_PATH), **engineering._identity(closeout)},
        "gpu_acceptance": {
            "path": str(P2_ACCEPTANCE),
            "manifest": engineering._identity(acceptance_path / "manifest.json"),
            "file_count": len(acceptance_manifest["files"]),
        },
        "benchmark": {
            "path": str(P2_BENCHMARK),
            "manifest": engineering._identity(benchmark_path / "manifest.json"),
            "file_count": len(benchmark_manifest["files"]),
        },
        "environment": acceptance_environment,
        "max_acceptance_reserved_fraction": float(physical["peak_reserved_fraction"]),
        "max_benchmark_reserved_fraction": float(
            max(item["peak_reserved_fraction"] for item in measurements)
        ),
    }


def _formal_code_paths() -> tuple[Path, ...]:
    return (
        e8.SPEC_PATH,
        Path("resp_train/paper_evidence/e8_film_decoder_redesign_v1_model.py"),
        Path("resp_train/paper_evidence/e8_film_decoder_redesign_v1.py"),
        FORMAL_PATH,
        Path("scripts/run_e8_film_decoder_redesign_v1.py"),
        Path("resp_train/crd/model.py"),
        Path("resp_train/crd/tf_v1_model.py"),
        Path("resp_train/crd/blocks.py"),
        Path("resp_train/crd/experiment.py"),
        Path("resp_train/crd/training.py"),
        Path("resp_train/losses/task.py"),
        Path("resp_train/metrics/task.py"),
        Path("resp_train/paper_evidence/w0_structural_factorial_v1.py"),
        Path("resp_train/paper_evidence/w0_structural_factorial_v1_formal.py"),
        Path("tests/test_e8_film_decoder_redesign_v1.py"),
    )


def prepare_formal_lock() -> Path:
    destination = e8.ROOT / FORMAL_LOCK_PATH
    if destination.exists():
        raise FileExistsError(f"E8 formal lock 已存在: {destination}")
    state = engineering._git_state()
    if state["status_porcelain"]:
        raise RuntimeError("E8 formal lock 要求干净 Git 工作树")
    e8.load_experiment_spec()
    p2 = verify_p2_evidence()
    source_lock, source_hash = sf_formal.load_formal_contract()
    if source_hash != W0_SOURCE_LOCK_SHA256:
        raise ValueError("E8 W0 train/validation 来源锁漂移")
    baselines = {
        str(seed): OmegaConf.to_container(e8.load_w0_baseline(seed), resolve=True)
        for seed in e8.SEEDS
    }
    lock = {
        "schema_version": 1,
        "protocol": e8.PROTOCOL,
        "status": "formal_execution_locked",
        "arms": list(ARMS),
        "arm_contracts": e8._arm_payload(),
        "seeds": list(e8.SEEDS),
        "counts": COUNTS,
        "samp_ids": SAMP_IDS,
        "epochs": e8.EPOCHS,
        "updates_per_epoch": e8.UPDATES_PER_EPOCH,
        "planned_updates": e8.PLANNED_UPDATES,
        "early_stopping": {
            "enabled": True,
            "min_epoch": e8.EARLY_STOP_MIN_EPOCH,
            "patience": e8.EARLY_STOP_PATIENCE,
            "min_delta": e8.EARLY_STOP_MIN_DELTA,
        },
        "spec": engineering._identity(e8.ROOT / e8.SPEC_PATH),
        "p2_evidence": p2,
        "w0_source_lock": {
            "path": str(W0_SOURCE_LOCK_PATH),
            "sha256": source_hash,
            "source_file_count": len(source_lock["source_files"]),
        },
        "baselines": baselines,
        "code_files": {
            str(path): engineering._identity(e8.ROOT / path)
            for path in _formal_code_paths()
        },
        "source_repository_root": str(e8.SOURCE_ROOT),
        "artifact_root": str(e8.SOURCE_ROOT / e8.OUTPUT_ROOT),
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "preparation_git": state,
    }
    engineering._write_json(destination, lock)
    return destination


def _base_formal_lock() -> tuple[dict[str, Any], str]:
    path = e8.ROOT / FORMAL_LOCK_PATH
    if not path.is_file():
        raise FileNotFoundError(f"E8 formal lock 尚未建立: {path}")
    lock_hash = engineering._sha256(path)
    if lock_hash != FORMAL_LOCK_SHA256:
        raise ValueError("E8 base formal lock 身份漂移")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if (
        lock.get("schema_version") != 1
        or lock.get("protocol") != e8.PROTOCOL
        or lock.get("status") != "formal_execution_locked"
        or tuple(lock.get("arms", ())) != ARMS
        or tuple(lock.get("seeds", ())) != e8.SEEDS
        or lock.get("arm_contracts") != e8._arm_payload()
        or lock.get("counts") != COUNTS
        or lock.get("samp_ids") != SAMP_IDS
        or lock.get("epochs") != e8.EPOCHS
        or lock.get("updates_per_epoch") != e8.UPDATES_PER_EPOCH
        or lock.get("planned_updates") != e8.PLANNED_UPDATES
        or lock.get("source_repository_root") != str(e8.SOURCE_ROOT)
        or lock.get("artifact_root") != str(e8.SOURCE_ROOT / e8.OUTPUT_ROOT)
    ):
        raise ValueError("E8 formal lock 科学合同漂移")
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
                    manifest = verify_formal_attempt(
                        attempt,
                        lock_hash=lock_hash,
                        arm=cell["arm"],
                        seed=cell["seed"],
                    )
                    completed.append(
                        {
                            "path": str(attempt),
                            "manifest": engineering._identity(attempt / "manifest.json"),
                            "file_count": len(manifest["files"]),
                        }
                    )
                elif (attempt / "lifecycle_failed.json").is_file():
                    failed.append(
                        {
                            "path": str(attempt),
                            "failure": engineering._identity(attempt / "lifecycle_failed.json"),
                        }
                    )
                else:
                    running.append(str(attempt))
        if running:
            raise RuntimeError(f"E8 amendment 前仍有运行 cell: {cell['arm']}/{cell['seed']}")
        if len(completed) > 1:
            raise RuntimeError(f"E8 amendment 前存在重复成功 cell: {cell['arm']}/{cell['seed']}")
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
    if counts != {"pending": 24, "failed": 1, "completed": 11}:
        raise ValueError(f"E8 amendment 前矩阵状态不符合冻结预期: {counts}")
    return {"counts": counts, "cells": cells}


def prepare_formal_amendment() -> Path:
    """冻结只影响 GPU 容量兼容门控的增量修订。"""

    destination = e8.ROOT / FORMAL_AMENDMENT_PATH
    if destination.exists():
        raise FileExistsError(f"E8 formal amendment 已存在: {destination}")
    state = engineering._git_state()
    if state["status_porcelain"]:
        raise RuntimeError("E8 formal amendment 要求工作树干净")
    lock, lock_hash = _base_formal_lock()
    amended = set(AMENDED_CODE_PATHS)
    for relative, expected in lock["code_files"].items():
        if relative not in amended and engineering._identity(e8.ROOT / relative) != expected:
            raise ValueError(f"E8 amendment 范围外代码漂移: {relative}")
    p2 = verify_p2_evidence()
    if p2 != lock["p2_evidence"]:
        raise ValueError("E8 amendment P2 evidence 漂移")
    snapshot = _pre_amendment_matrix_snapshot(lock_hash)
    amendment = {
        "schema_version": 1,
        "protocol": e8.PROTOCOL,
        "status": "formal_runtime_compatibility_amendment_locked",
        "base_formal_lock": {
            "path": str(FORMAL_LOCK_PATH),
            "sha256": lock_hash,
        },
        "scope": "runtime_gpu_capacity_compatibility_only",
        "scientific_contract_changed": False,
        "runtime_policy": {
            "matched_fields": [
                "python",
                "torch",
                "cuda_runtime",
                "cudnn",
                "dependencies",
                "device_name",
                "amp_dtype",
            ],
            "minimum_device_total_bytes": MIN_FORMAL_DEVICE_TOTAL_BYTES,
            "p2_max_peak_reserved_bytes": P2_MAX_PEAK_RESERVED_BYTES,
            "maximum_peak_reserved_fraction": 0.8,
            "observed_gpu0_total_bytes": 16_710_500_352,
            "p2_gpu1_total_bytes": 16_717_840_384,
            "absolute_difference_bytes": 7_340_032,
            "relative_difference": 7_340_032 / 16_717_840_384,
        },
        "pre_amendment_matrix": snapshot,
        "amended_code_files": {
            relative: {
                "base": lock["code_files"][relative],
                "revised": engineering._identity(e8.ROOT / relative),
            }
            for relative in AMENDED_CODE_PATHS
        },
        "p2_evidence": p2,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "preparation_git": state,
    }
    engineering._write_json(destination, amendment)
    return destination


def load_formal_amendment(lock: Mapping[str, Any], lock_hash: str) -> tuple[dict[str, Any], str]:
    path = e8.ROOT / FORMAL_AMENDMENT_PATH
    if not path.is_file():
        raise FileNotFoundError(f"E8 formal amendment 尚未建立: {path}")
    amendment = json.loads(path.read_text(encoding="utf-8"))
    amendment_hash = engineering._sha256(path)
    policy = amendment.get("runtime_policy", {})
    if (
        amendment.get("schema_version") != 1
        or amendment.get("protocol") != e8.PROTOCOL
        or amendment.get("status") != "formal_runtime_compatibility_amendment_locked"
        or amendment.get("base_formal_lock", {}).get("sha256") != lock_hash
        or amendment.get("scope") != "runtime_gpu_capacity_compatibility_only"
        or amendment.get("scientific_contract_changed") is not False
        or int(policy.get("minimum_device_total_bytes", -1)) != MIN_FORMAL_DEVICE_TOTAL_BYTES
        or int(policy.get("p2_max_peak_reserved_bytes", -1)) != P2_MAX_PEAK_RESERVED_BYTES
        or float(policy.get("maximum_peak_reserved_fraction", -1)) != 0.8
        or amendment.get("p2_evidence") != lock["p2_evidence"]
    ):
        raise ValueError("E8 formal amendment 合同漂移")
    for relative in AMENDED_CODE_PATHS:
        entry = amendment.get("amended_code_files", {}).get(relative, {})
        if (
            entry.get("base") != lock["code_files"][relative]
            or entry.get("revised") != engineering._identity(e8.ROOT / relative)
        ):
            raise ValueError(f"E8 formal amendment 代码身份漂移: {relative}")
    for cell in amendment["pre_amendment_matrix"]["cells"]:
        for completed in cell["completed"]:
            path_entry = Path(completed["path"])
            if engineering._identity(path_entry / "manifest.json") != completed["manifest"]:
                raise ValueError(f"E8 amendment 前成功产物漂移: {path_entry}")
    return amendment, amendment_hash


def load_formal_lock() -> tuple[dict[str, Any], str, dict[str, Any]]:
    lock, lock_hash = _base_formal_lock()
    if engineering._identity(e8.ROOT / e8.SPEC_PATH) != lock["spec"]:
        raise ValueError("E8 formal spec 身份漂移")
    amendment, amendment_hash = load_formal_amendment(lock, lock_hash)
    amended = set(AMENDED_CODE_PATHS)
    for relative, expected in lock["code_files"].items():
        if relative not in amended and engineering._identity(e8.ROOT / relative) != expected:
            raise ValueError(f"E8 formal 代码身份漂移: {relative}")
    p2 = verify_p2_evidence()
    if p2 != lock["p2_evidence"]:
        raise ValueError("E8 formal P2 evidence 漂移")
    source_lock, source_hash = sf_formal.load_formal_contract()
    if (
        source_hash != lock["w0_source_lock"]["sha256"]
        or source_hash != W0_SOURCE_LOCK_SHA256
    ):
        raise ValueError("E8 formal W0 source lock 漂移")
    for seed in e8.SEEDS:
        current = OmegaConf.to_container(e8.load_w0_baseline(seed), resolve=True)
        if current != lock["baselines"][str(seed)]:
            raise ValueError(f"E8 formal baseline 漂移: {seed}")
    lock = dict(lock)
    lock["_runtime_amendment"] = amendment
    lock["_runtime_amendment_sha256"] = amendment_hash
    return lock, lock_hash, source_lock


def runtime_preflight(device: str) -> dict[str, Any]:
    state = engineering._git_state()
    if state["status_porcelain"]:
        raise RuntimeError("E8 formal 要求干净 Git 工作树")
    resolved = torch.device(device)
    if resolved.type != "cuda" or resolved.index is None or not torch.cuda.is_available():
        raise RuntimeError("E8 formal 要求显式可用的 cuda:<index>")
    problems = check_crd_dependencies()
    if problems:
        raise RuntimeError("E8 formal 原生依赖检查失败: " + "; ".join(problems))
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
    runtime: Mapping[str, Any],
    accepted: Mapping[str, Any],
    amendment: Mapping[str, Any] | None = None,
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
            raise ValueError(f"E8 formal runtime 与 P2 不一致: {key}")
    current_total = int(runtime["device_total_bytes"])
    accepted_total = int(accepted["device_total_bytes"])
    if amendment is None:
        if current_total < accepted_total:
            raise ValueError("E8 formal GPU 总显存低于 P2 验收设备")
        return {
            "matched_fields": list(matched),
            "capacity_policy": "current_device_total_bytes_gte_p2",
            "p2_device_total_bytes": accepted_total,
            "current_device_total_bytes": current_total,
        }
    policy = amendment["runtime_policy"]
    minimum = int(policy["minimum_device_total_bytes"])
    peak_reserved = int(policy["p2_max_peak_reserved_bytes"])
    limit = float(policy["maximum_peak_reserved_fraction"])
    if current_total < minimum or peak_reserved / current_total > limit:
        raise ValueError("E8 formal GPU 容量不满足 runtime amendment 安全线")
    return {
        "matched_fields": list(matched),
        "capacity_policy": "same_stack_exact_model_minimum_total_and_p2_peak_fraction",
        "p2_device_total_bytes": accepted_total,
        "current_device_total_bytes": current_total,
        "minimum_device_total_bytes": minimum,
        "p2_peak_reserved_bytes": peak_reserved,
        "p2_peak_reserved_fraction_on_current_device": peak_reserved / current_total,
        "limit_fraction": limit,
    }


def audit_sources(
    source_lock: Mapping[str, Any],
    cfg: DictConfig,
    output: Path,
) -> dict[str, pd.DataFrame]:
    engineering._write_json(
        output / "access_started.json",
        {
            "purpose": "E8 formal train/validation",
            "splits": ["train", "val"],
            "research_test_evaluation": False,
            "dataset_index": source_lock["dataset_index"],
            "source_files": list(source_lock["source_files"]),
        },
    )
    for relative, expected in source_lock["source_files"].items():
        sf.verify_identity(e8.SOURCE_ROOT / relative, expected)
    index = source_lock["dataset_index"]
    if sf.sha256_file(Path(index["path"])) != index["sha256"]:
        raise ValueError("E8 formal dataset index 身份漂移")
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
            raise ValueError(f"E8 formal {split} row 合同漂移")
        frame.to_csv(output / f"{split}_rows.csv", index=False)
        rows[split] = frame
    if set(rows["train"].samp_id) & set(rows["val"].samp_id):
        raise ValueError("E8 formal train/validation subject 隔离失败")
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


class FormalE8Experiment(e8.E8FilmDecoderExperiment):
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
        model = build_e8_film_decoder_redesign_model(self.cfg)
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
        raise ValueError("E8 formal checkpoint epoch/config 漂移")
    for key in history_row.index:
        if key not in checkpoint["metrics"] or not np.isclose(
            float(checkpoint["metrics"][key]),
            float(history_row[key]),
            atol=1e-12,
            rtol=1e-12,
        ):
            raise ValueError(f"E8 formal checkpoint/history 不一致: {key}")
    extra = checkpoint["extra_state"]
    if (
        extra.get("protocol") != e8.PROTOCOL
        or int(extra.get("update_index", -1)) != expected_epoch * e8.UPDATES_PER_EPOCH
        or int(extra.get("total_updates", -1)) != e8.PLANNED_UPDATES
    ):
        raise ValueError("E8 formal checkpoint update/protocol 漂移")
    early = extra.get("early_stopping")
    if (
        not isinstance(early, Mapping)
        or early.get("enabled") is not True
        or int(early.get("min_epoch", -1)) != e8.EARLY_STOP_MIN_EPOCH
        or int(early.get("patience", -1)) != e8.EARLY_STOP_PATIENCE
        or float(early.get("min_delta", np.nan)) != e8.EARLY_STOP_MIN_DELTA
        or int(early.get("planned_epochs", -1)) != e8.EPOCHS
    ):
        raise ValueError("E8 formal checkpoint early-stopping 合同漂移")


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
        raise ValueError("E8 formal 保存配置漂移")
    history = pd.read_csv(run_dir / "train_history.csv")
    best_epoch = sf.validate_history(history, cfg)
    final_epoch = int(history.iloc[-1].epoch)
    if not np.isfinite(
        history[["train_elapsed_seconds", "train_samples_per_second"]].to_numpy()
    ).all():
        raise FloatingPointError("E8 formal history runtime 非有限")
    model = build_e8_film_decoder_redesign_model(cfg)
    expected_initialization = sf_formal.state_dict_identity(model.state_dict())
    saved_initialization = json.loads(initialization_path.read_text(encoding="utf-8"))
    if saved_initialization.get("state_sha256") != expected_initialization["state_sha256"]:
        raise ValueError("E8 formal initialization identity 漂移")
    optimizer, partition = build_crd_optimizer(model, cfg)
    groups = json.loads((run_dir / "optimizer_parameter_groups.json").read_text(encoding="utf-8"))
    if groups != {
        "weight_decay": float(cfg.training.weight_decay),
        "decay": list(partition.decay_names),
        "no_decay": list(partition.no_decay_names),
    }:
        raise ValueError("E8 formal optimizer 参数分组漂移")
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
        if filename == "checkpoint_final.pt" and int(
            checkpoint["extra_state"]["early_stopping"].get("completed_epochs", -1)
        ) != final_epoch:
            raise ValueError("E8 formal final checkpoint completed_epochs 漂移")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                state = optimizer.state.get(parameter, {})
                if (
                    not {"step", "exp_avg", "exp_avg_sq"}.issubset(state)
                    or float(state["step"]) != epoch * e8.UPDATES_PER_EPOCH
                    or state["exp_avg"].shape != parameter.shape
                    or state["exp_avg_sq"].shape != parameter.shape
                ):
                    raise ValueError("E8 formal optimizer state/step 不完整")
    metrics = pd.read_csv(run_dir / "metrics.csv")
    degeneracy = sf.validate_metrics(metrics, rows)
    if not metrics.arm.eq(arm).all() or not metrics.seed.eq(int(cfg.training.seed)).all():
        raise ValueError("E8 formal metrics arm/seed identity 漂移")
    sf_formal.validate_anchor_rows(metrics, source_lock, int(cfg.training.seed))
    summary = pd.read_csv(run_dir / "metrics_summary.csv")
    expected_summary = summarize_task_metrics(metrics)
    if len(summary) != 1:
        raise ValueError("E8 formal metrics summary 必须恰有一行")
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
            raise ValueError(f"E8 formal summary 数值/分母漂移: {metric}")
    runtime = json.loads((run_dir / "runtime_summary.json").read_text(encoding="utf-8"))
    if runtime.get("peak_reserved_fraction") is None or not np.isfinite(
        runtime["peak_reserved_fraction"]
    ):
        raise FloatingPointError("E8 formal runtime summary 不完整")
    return {
        "protocol": e8.PROTOCOL,
        "arm": arm,
        "seed": int(cfg.training.seed),
        "planned_epochs": e8.EPOCHS,
        "completed_epochs": final_epoch,
        "planned_updates": e8.PLANNED_UPDATES,
        "completed_updates": final_epoch * e8.UPDATES_PER_EPOCH,
        "selected_epoch": best_epoch,
        "validation_rows": len(metrics),
        "prediction_degeneracy": degeneracy,
        "validation_row_order_sha256": sf_formal.array_hash(rows.dataset_row_id.to_numpy()),
        "initialization_state_sha256": saved_initialization["state_sha256"],
        "runtime": runtime,
    }


@contextmanager
def _formal_attempt(
    parent: Path,
    *,
    lock_hash: str,
    amendment_hash: str | None = None,
    arm: str,
    seed: int,
) -> Iterator[Path]:
    parent.mkdir(parents=True, exist_ok=True)
    lock_path = parent / f".execution_{lock_hash}.lock"
    with lock_path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("相同 E8 formal cell 正在运行") from exc
        for receipt_path in parent.glob("*/freeze_receipt.json"):
            manifest_path = receipt_path.parent / "manifest.json"
            freeze = json.loads(receipt_path.read_text(encoding="utf-8"))
            if engineering._identity(manifest_path) != freeze.get("manifest"):
                raise RuntimeError("E8 formal 已完成 attempt 身份错误")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("implementation_lock_sha256") == lock_hash
                and manifest.get("arm") == arm
                and int(manifest.get("seed", -1)) == seed
                and manifest.get("status") == "completed"
            ):
                raise FileExistsError(f"E8 formal cell 已完成: {arm}/{seed}")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = parent / f"formal_{lock_hash[:12]}_{stamp}_{uuid4().hex[:12]}"
        output.mkdir(exist_ok=False)
        context = {
            "protocol": e8.PROTOCOL,
            "phase": "formal",
            "arm": arm,
            "seed": seed,
            "implementation_lock_sha256": lock_hash,
            "formal_runtime_amendment_sha256": amendment_hash,
            "command": sys.argv,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        engineering._write_json(
            output / "lifecycle_started.json",
            {**context, "status": "running"},
        )
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
                {"protocol": e8.PROTOCOL, "manifest": engineering._identity(output / "manifest.json")},
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
    if arm not in ARMS or int(seed) not in e8.SEEDS:
        raise ValueError("E8 formal arm/seed 不属于冻结矩阵")
    seed = int(seed)
    lock, lock_hash, source_lock = load_formal_lock()
    amendment = lock["_runtime_amendment"]
    amendment_hash = str(lock["_runtime_amendment_sha256"])
    parent = e8.SOURCE_ROOT / e8.OUTPUT_ROOT / "formal" / arm / f"seed_{seed}"
    with _formal_attempt(
        parent,
        lock_hash=lock_hash,
        amendment_hash=amendment_hash,
        arm=arm,
        seed=seed,
    ) as output:
        runtime = runtime_preflight(device)
        compatibility = _runtime_compatibility(
            runtime,
            lock["p2_evidence"]["environment"],
            amendment,
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
        cfg = e8.derived_config(
            baseline,
            arm=arm,
            output_root=output / "training",
            device=device,
        )
        e8.validate_config(
            cfg,
            baseline,
            arm=arm,
            output_root=output / "training",
            device=device,
        )
        rows = audit_sources(source_lock, cfg, output)
        initialization_path = output / "initialization.json"
        experiment = FormalE8Experiment(cfg, rows["val"], arm, initialization_path)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        run_dir = experiment.train()
        torch.cuda.synchronize(device)
        wall_seconds = time.perf_counter() - started
        if wall_seconds <= 0 or not np.isfinite(wall_seconds):
            raise RuntimeError("E8 formal wall time 非法")
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
                e8.SOURCE_ROOT / e8.OUTPUT_ROOT / "formal" / arm / f"seed_{seed}"
            ),
        }
        for seed in e8.SEEDS
        for arm in ARMS
    ]


def verify_formal_attempt(
    path: Path,
    *,
    lock_hash: str,
    arm: str,
    seed: int,
    amendment: Mapping[str, Any] | None = None,
    amendment_hash: str | None = None,
) -> dict[str, Any]:
    freeze = json.loads((path / "freeze_receipt.json").read_text(encoding="utf-8"))
    manifest_path = path / "manifest.json"
    if engineering._identity(manifest_path) != freeze.get("manifest"):
        raise RuntimeError("E8 formal freeze receipt 身份错误")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("protocol") != e8.PROTOCOL
        or manifest.get("phase") != "formal"
        or manifest.get("status") != "completed"
        or manifest.get("implementation_lock_sha256") != lock_hash
        or manifest.get("arm") != arm
        or int(manifest.get("seed", -1)) != int(seed)
    ):
        raise ValueError("E8 formal manifest 合同漂移")
    preserved: dict[str, Mapping[str, Any]] = {}
    if amendment is not None:
        preserved = {
            item["path"]: item
            for cell in amendment["pre_amendment_matrix"]["cells"]
            for item in cell["completed"]
        }
    preserved_entry = preserved.get(str(path))
    if preserved_entry is not None:
        if (
            manifest.get("formal_runtime_amendment_sha256") is not None
            or engineering._identity(manifest_path) != preserved_entry["manifest"]
        ):
            raise ValueError("E8 amendment 前成功 attempt 身份漂移")
    elif amendment is not None and manifest.get("formal_runtime_amendment_sha256") != amendment_hash:
        raise ValueError("E8 amendment 后 formal attempt 缺少修订身份")
    for relative, expected in manifest["files"].items():
        target = (path / relative).resolve()
        if not target.is_relative_to(path.resolve()) or engineering._identity(target) != expected:
            raise RuntimeError(f"E8 formal 文件身份漂移: {relative}")
    receipt = json.loads((path / "formal_receipt.json").read_text(encoding="utf-8"))
    if (
        receipt.get("implementation_lock_sha256") != lock_hash
        or receipt.get("arm") != arm
        or int(receipt.get("seed", -1)) != int(seed)
        or int(receipt.get("validation_rows", -1)) != COUNTS["val"]
    ):
        raise ValueError("E8 formal receipt 合同漂移")
    if preserved_entry is not None:
        if receipt.get("formal_runtime_amendment_sha256") is not None:
            raise ValueError("E8 amendment 前 receipt 不应包含修订身份")
    elif amendment is not None and receipt.get("formal_runtime_amendment_sha256") != amendment_hash:
        raise ValueError("E8 amendment 后 receipt 修订身份漂移")
    return manifest


def matrix_status() -> dict[str, Any]:
    lock, lock_hash, _source = load_formal_lock()
    amendment = lock["_runtime_amendment"]
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
                        arm=cell["arm"],
                        seed=cell["seed"],
                        amendment=amendment,
                        amendment_hash=amendment_hash,
                    )
                    completed.append(str(attempt))
                elif (attempt / "lifecycle_failed.json").is_file():
                    failed.append(str(attempt))
                else:
                    running.append(str(attempt))
        if len(completed) > 1:
            raise RuntimeError(f"E8 formal cell 多个成功 attempt: {cell['arm']}/{cell['seed']}")
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
        "protocol": e8.PROTOCOL,
        "implementation_lock_sha256": lock_hash,
        "formal_runtime_amendment_sha256": amendment_hash,
        "counts": counts,
        "cells": cells,
    }
