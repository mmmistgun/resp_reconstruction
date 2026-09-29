"""E9：固定 18 个 validation-selected checkpoint 的 research-test 评价。"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
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
from omegaconf import DictConfig, OmegaConf

from resp_train.crd.config import check_crd_dependencies, crd_dependency_versions
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
from resp_train.paper_evidence import e9_latent_width_condition_refiner_v1 as e9
from resp_train.paper_evidence import e9_latent_width_condition_refiner_v1_engineering as engineering
from resp_train.paper_evidence import e9_latent_width_condition_refiner_v1_formal as formal
from resp_train.paper_evidence import e9_latent_width_condition_refiner_v1_summary as p5
from resp_train.paper_evidence import w0_structural_factorial_v1 as sf
from resp_train.paper_evidence.w0_structural_factorial_v1_test import (
    IDENTITY_COLUMNS,
    TARGET_BOOLEAN_COLUMNS,
    TARGET_COLUMNS,
    check_test_metrics,
    check_test_rows,
    guarded_batches,
    quality_flags,
)
from resp_train.paper_evidence.e9_latent_width_condition_refiner_v1_model import (
    ARMS,
    build_e9_latent_width_condition_refiner_model,
)


PROTOCOL = "e9-latent-width-condition-refiner-v1-research-test-20260929"
COUNT = 2_310
SUBJECTS = 8
TEST_SAMPLE_SEED = 20260612
ROW_ORDER_SHA256 = "184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8"
P5_SUMMARY = Path(
    "runs/e9_latent_width_condition_refiner_v1/validation_summary/"
    "summary_8bb3b1992f64_20260929T144128Z_f1242dd035e9"
)
P5_MANIFEST_SHA256 = "93ae65b2a19c3b76bb3fb6656ce4190d26e994abb0028e810a9c1c2e1a29c81f"
P5_DECISION_SHA256 = "ed7159d43894acea366beac74bdc1c056636bde02a3131f861fa353e85b690a6"
PROTOCOL_PATH = Path(
    "docs/experiments/e9_latent_width_condition_refiner_v1_research_test_protocol_20260929.md"
)
MODULE_PATH = Path("resp_train/paper_evidence/e9_latent_width_condition_refiner_v1_test.py")
SCRIPT_PATH = Path("scripts/run_e9_latent_width_condition_refiner_v1_test.py")
TEST_PATH = Path("tests/test_e9_latent_width_condition_refiner_v1_test.py")


def _research_test_root() -> Path:
    return e9.SOURCE_ROOT / e9.OUTPUT_ROOT / "research_test"


def _test_git_state() -> dict[str, Any]:
    """要求所有tracked文件干净；允许不参与执行的未跟踪实验Markdown。"""

    state = engineering._git_state()
    allowed: list[str] = []
    unsafe: list[str] = []
    for line in state["status_porcelain"].splitlines():
        if line.startswith("?? docs/experiments/") and line.endswith(".md"):
            allowed.append(line[3:])
        elif line:
            unsafe.append(line)
    if unsafe:
        raise RuntimeError(f"E9 research-test存在未冻结tracked/code改动: {unsafe}")
    return {
        **state,
        "status_porcelain": "",
        "allowed_untracked_experiment_docs": sorted(allowed),
    }


def _runtime_preflight(device: str) -> dict[str, Any]:
    state = _test_git_state()
    resolved = torch.device(device)
    if resolved.type != "cuda" or resolved.index is None or not torch.cuda.is_available():
        raise RuntimeError("E9 research-test要求显式可用的cuda:<index>")
    problems = check_crd_dependencies()
    if problems:
        raise RuntimeError("E9 research-test原生依赖检查失败: " + "; ".join(problems))
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


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _code_paths() -> tuple[Path, ...]:
    paths = (
        MODULE_PATH,
        SCRIPT_PATH,
        TEST_PATH,
        PROTOCOL_PATH,
        e9.SPEC_PATH,
        Path("resp_train/paper_evidence/e9_latent_width_condition_refiner_v1_model.py"),
        Path("resp_train/paper_evidence/e9_latent_width_condition_refiner_v1.py"),
        formal.FORMAL_PATH,
        p5.SUMMARY_PATH,
        Path("resp_train/crd/experiment.py"),
        Path("resp_train/crd/tf_v1_research_test_data.py"),
        Path("resp_train/data/research_v2.py"),
        Path("resp_train/engine/train.py"),
        Path("resp_train/metrics/task.py"),
    )
    missing = [str(path) for path in paths if not (e9.ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"E9 research-test 实现缺少文件: {missing}")
    return paths


@contextmanager
def evidence_attempt(
    parent: Path,
    *,
    phase: str,
    evidence_hash: str,
    arm: str | None = None,
    seed: int | None = None,
    reject_completed: bool = False,
) -> Iterator[Path]:
    parent.mkdir(parents=True, exist_ok=True)
    lock_path = parent / f".execution_{evidence_hash}.lock"
    with lock_path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("相同 E9 research-test identity 正在运行") from exc
        if reject_completed:
            for receipt_path in parent.glob("*/freeze_receipt.json"):
                manifest_path = receipt_path.parent / "manifest.json"
                freeze = json.loads(receipt_path.read_text(encoding="utf-8"))
                if engineering._identity(manifest_path) != freeze.get("manifest"):
                    raise RuntimeError("E9 research-test 已完成 attempt 身份错误")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if (
                    manifest.get("evidence_identity_sha256") == evidence_hash
                    and manifest.get("phase") == phase
                    and manifest.get("arm") == arm
                    and manifest.get("seed") == seed
                    and manifest.get("status") == "completed"
                ):
                    raise FileExistsError(
                        f"相同 E9 research-test cell 已完成: {receipt_path.parent}"
                    )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = parent / f"{phase}_{evidence_hash[:12]}_{stamp}_{uuid4().hex[:12]}"
        output.mkdir(exist_ok=False)
        context = {
            "protocol": PROTOCOL,
            "phase": phase,
            "split": "test" if phase != "allowlist" else None,
            "arm": arm,
            "seed": seed,
            "evidence_identity_sha256": evidence_hash,
            "command": sys.argv,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        engineering._write_json(
            output / "lifecycle_started.json", {**context, "status": "running"}
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
                    str(path.relative_to(output)): engineering._identity(path)
                    for path in sorted(output.rglob("*"))
                    if path.is_file()
                },
            }
            engineering._write_json(output / "manifest.json", manifest)
            engineering._write_json(
                output / "freeze_receipt.json",
                {"protocol": PROTOCOL, "manifest": engineering._identity(output / "manifest.json")},
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


def verify_attempt(
    output: Path,
    *,
    phase: str,
    evidence_hash: str,
    arm: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    output = output.resolve()
    if (output / "lifecycle_failed.json").exists():
        raise ValueError(f"E9 research-test attempt 失败: {output}")
    freeze = json.loads((output / "freeze_receipt.json").read_text(encoding="utf-8"))
    manifest_path = output / "manifest.json"
    if engineering._identity(manifest_path) != freeze.get("manifest"):
        raise RuntimeError("E9 research-test freeze receipt 身份错误")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("protocol") != PROTOCOL
        or manifest.get("phase") != phase
        or manifest.get("status") != "completed"
        or manifest.get("evidence_identity_sha256") != evidence_hash
        or manifest.get("arm") != arm
        or manifest.get("seed") != seed
    ):
        raise ValueError("E9 research-test attempt identity 漂移")
    required = {
        "allowlist": {
            "lifecycle_completed.json",
            "allowlist.json",
            "allowlist_receipt.json",
            "source_manifest.json",
            "access_receipt.json",
            "source_code.json",
        },
        "evaluation": {
            "lifecycle_completed.json",
            "evaluation_receipt.json",
            "metrics.csv",
            "metrics_summary.csv",
            "test_rows.csv",
            "access_started.json",
            "access_receipt.json",
            "environment.json",
            "resolved_config.yaml",
            "allowlist_source.json",
        },
        "summary": {
            "lifecycle_completed.json",
            "summary_receipt.json",
            "decision.json",
            "seed_primary_metrics.csv",
            "arm_primary_summary.csv",
            "planned_contrasts_across_seed.csv",
            "subject_stratified_metrics.csv",
            "metric_denominators.csv",
            "validation_test_comparison.csv",
            "source_manifest.json",
            "access_receipt.json",
        },
    }[phase]
    if not required.issubset(manifest["files"]):
        raise ValueError("E9 research-test attempt 缺少必需产物")
    for relative, expected in manifest["files"].items():
        path = (output / relative).resolve()
        if not path.is_relative_to(output) or engineering._identity(path) != expected:
            raise RuntimeError(f"E9 research-test 文件身份漂移: {relative}")
    return manifest


def _p5_summary_path() -> Path:
    return e9.SOURCE_ROOT / P5_SUMMARY


def _checkpoint_entry(
    *,
    arm: str,
    seed: int,
    formal_attempt: Path,
    source: Mapping[str, Any],
    lock: Mapping[str, Any],
    source_lock: Mapping[str, Any],
) -> dict[str, Any]:
    amendment_hash = str(lock["_runtime_amendment_sha256"])
    lock_hash = engineering._sha256(e9.ROOT / e9.FORMAL_LOCK_PATH)
    formal.verify_formal_attempt(
        formal_attempt,
        lock_hash=lock_hash,
        amendment_hash=amendment_hash,
        arm=arm,
        seed=seed,
    )
    receipt = json.loads((formal_attempt / "formal_receipt.json").read_text(encoding="utf-8"))
    run_dir = (formal_attempt / receipt["run_dir"]).resolve()
    if not run_dir.is_relative_to(formal_attempt):
        raise ValueError("E9 allowlist training run_dir 越界")
    cfg = OmegaConf.load(run_dir / "config.yaml")
    baseline = OmegaConf.create(lock["baselines"][str(seed)])
    e9.validate_config(
        cfg,
        baseline,
        arm=arm,
        output_root=formal_attempt / "training",
        device=str(cfg.training.device),
    )
    history = pd.read_csv(run_dir / "train_history.csv")
    selected_epoch = sf.validate_history(history, cfg)
    if (
        selected_epoch != int(receipt["selected_epoch"])
        or selected_epoch != int(source["selected_epoch"])
        or len(history) != int(receipt["completed_epochs"])
    ):
        raise ValueError("E9 allowlist validation selector 漂移")
    formal_manifest = json.loads((formal_attempt / "manifest.json").read_text(encoding="utf-8"))
    checkpoint_path = run_dir / "checkpoint_best_local_rr.pt"
    checkpoint_relative = str(checkpoint_path.relative_to(formal_attempt))
    if engineering._identity(checkpoint_path) != formal_manifest["files"][checkpoint_relative]:
        raise ValueError("E9 allowlist checkpoint manifest 身份漂移")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        int(checkpoint["epoch"]) != selected_epoch
        or checkpoint["config"] != OmegaConf.to_container(cfg, resolve=True)
    ):
        raise ValueError("E9 allowlist checkpoint epoch/config 漂移")
    engineering.finite_tree(checkpoint["model_state_dict"], label="allowlist_checkpoint")
    formal.validate_formal_run(
        run_dir,
        cfg,
        pd.read_csv(formal_attempt / "val_rows.csv"),
        source_lock,
        arm=arm,
        initialization_path=formal_attempt / "initialization.json",
    )
    development = sorted(
        set(pd.read_csv(formal_attempt / "train_rows.csv").samp_id.astype(int))
        | set(pd.read_csv(formal_attempt / "val_rows.csv").samp_id.astype(int))
    )
    files = {
        name: {"path": str(path), **engineering._identity(path)}
        for name, path in (
            ("config", run_dir / "config.yaml"),
            ("history", run_dir / "train_history.csv"),
            ("checkpoint", checkpoint_path),
            ("formal_manifest", formal_attempt / "manifest.json"),
            ("formal_receipt", formal_attempt / "formal_receipt.json"),
        )
    }
    return {
        "entry_id": f"{arm}_seed{seed}",
        "arm": arm,
        "seed": seed,
        "selected_epoch": selected_epoch,
        "completed_epochs": int(receipt["completed_epochs"]),
        "formal_attempt": str(formal_attempt),
        "training_config": OmegaConf.to_container(cfg, resolve=True),
        "development_samp_ids": development,
        "files": files,
    }


def prepare_allowlist() -> Path:
    state = _test_git_state()
    summary = _p5_summary_path()
    if engineering._sha256(summary / "manifest.json") != P5_MANIFEST_SHA256:
        raise ValueError("E9 P5 summary manifest identity 漂移")
    if engineering._sha256(summary / "decision.json") != P5_DECISION_SHA256:
        raise ValueError("E9 P5 decision identity 漂移")
    p5.verify_summary_attempt(summary)
    source_manifest = json.loads((summary / "source_manifest.json").read_text(encoding="utf-8"))
    formal_runs = source_manifest["formal_runs"]
    lock, lock_hash, source_lock = formal.load_formal_lock()
    entries: list[dict[str, Any]] = []
    for seed in e9.SEEDS:
        for arm in ARMS:
            source = formal_runs[f"{arm}/{seed}"]
            attempt = Path(source["attempt"]).resolve()
            if engineering._identity(attempt / "manifest.json") != source["manifest"]:
                raise ValueError("E9 P5 formal source manifest 漂移")
            entries.append(
                _checkpoint_entry(
                    arm=arm,
                    seed=seed,
                    formal_attempt=attempt,
                    source=source,
                    lock=lock,
                    source_lock=source_lock,
                )
            )
    if len(entries) != 18 or len({item["entry_id"] for item in entries}) != 18:
        raise ValueError("E9 research-test checkpoint allowlist 不完整")

    cache_manifest_path = FROZEN_RESEARCH_TEST_CACHE_ROOT / "cache_manifest.json"
    if engineering._sha256(cache_manifest_path) != FROZEN_RESEARCH_TEST_CACHE_MANIFEST_SHA256:
        raise ValueError("E9 research-test cache manifest identity 漂移")
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    split = cache_manifest["splits"]["test"]
    if (
        int(split["count"]) != COUNT
        or int(split["samp_id_count"]) != SUBJECTS
        or split["row_ids_sha256"] != ROW_ORDER_SHA256
    ):
        raise ValueError("E9 research-test cache split contract 漂移")
    cache_files = {
        name: cache_manifest["files"][name]
        for name in ("test_w.npy", "test_row_ids.npy", "w_frequencies_hz.npy")
    }
    code_files = {
        str(path): engineering._identity(e9.ROOT / path) for path in _code_paths()
    }
    preparation_identity = _canonical_sha256(
        {
            "protocol": PROTOCOL,
            "validation_summary_manifest_sha256": P5_MANIFEST_SHA256,
            "validation_decision_sha256": P5_DECISION_SHA256,
            "training_implementation_lock_sha256": lock_hash,
            "formal_runtime_amendment_sha256": lock["_runtime_amendment_sha256"],
            "test_cache_manifest_sha256": FROZEN_RESEARCH_TEST_CACHE_MANIFEST_SHA256,
            "git_commit": state["commit"],
            "code_files": code_files,
        }
    )
    allowlist = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "allowlist_preparation_identity_sha256": preparation_identity,
        "evidence_role": "reused research/development evidence",
        "training_protocol": e9.PROTOCOL,
        "training_implementation_lock_sha256": lock_hash,
        "formal_runtime_amendment_sha256": lock["_runtime_amendment_sha256"],
        "validation_summary": {
            "path": str(summary),
            "manifest": engineering._identity(summary / "manifest.json"),
            "decision": engineering._identity(summary / "decision.json"),
        },
        "arms": list(ARMS),
        "seeds": list(e9.SEEDS),
        "entries": entries,
        "split": "test",
        "count": COUNT,
        "samp_id_count": SUBJECTS,
        "test_sample_seed": TEST_SAMPLE_SEED,
        "row_order_sha256": ROW_ORDER_SHA256,
        "batch_size": 128,
        "amp_dtype": "bfloat16",
        "include_test_only": True,
        "cache": {
            "root": str(FROZEN_RESEARCH_TEST_CACHE_ROOT),
            "identity": FROZEN_RESEARCH_TEST_CACHE_IDENTITY,
            "manifest": {
                "path": str(cache_manifest_path),
                **engineering._identity(cache_manifest_path),
            },
            "files": cache_files,
        },
        "dataset_index": {
            "path": cache_manifest["dataset_index"],
            "sha256": cache_manifest["dataset_index_sha256"],
        },
        "preparation_git": state,
        "code_files": code_files,
    }
    parent = _research_test_root() / "allowlist"
    with evidence_attempt(
        parent,
        phase="allowlist",
        evidence_hash=preparation_identity,
        reject_completed=True,
    ) as output:
        engineering._write_json(output / "allowlist.json", allowlist)
        engineering._write_json(
            output / "allowlist_receipt.json",
            {
                "protocol": PROTOCOL,
                "entries": 18,
                "allowlist": engineering._identity(output / "allowlist.json"),
                "allowlist_preparation_identity_sha256": preparation_identity,
                "validation_summary_manifest_sha256": P5_MANIFEST_SHA256,
            },
        )
        engineering._write_json(
            output / "source_manifest.json",
            {
                "validation_summary": allowlist["validation_summary"],
                "formal_runs": formal_runs,
                "training_implementation_lock_sha256": lock_hash,
                "formal_runtime_amendment_sha256": lock["_runtime_amendment_sha256"],
            },
        )
        engineering._write_json(output / "source_code.json", {"git": state, "files": code_files})
        engineering._write_json(
            output / "access_receipt.json",
            {
                "validation_artifacts_read": True,
                "checkpoint_bytes_read": True,
                "test_cache_manifest_read": True,
                "test_array_read": False,
                "dataset_index_read": False,
                "model_inference_used": False,
            },
        )
    return output


def load_allowlist(path: Path) -> tuple[dict[str, Any], str]:
    path = path.resolve()
    started = json.loads((path / "lifecycle_started.json").read_text(encoding="utf-8"))
    preparation_identity = str(started.get("evidence_identity_sha256", ""))
    if len(preparation_identity) != 64 or any(
        character not in "0123456789abcdef" for character in preparation_identity
    ):
        raise ValueError("E9 research-test allowlist preparation identity 非法")
    verify_attempt(path, phase="allowlist", evidence_hash=preparation_identity)
    allowlist_path = path / "allowlist.json"
    allowlist = json.loads(allowlist_path.read_text(encoding="utf-8"))
    _formal_lock, current_formal_lock_hash, _source_lock = formal.load_formal_lock()
    if (
        allowlist.get("schema_version") != 1
        or allowlist.get("protocol") != PROTOCOL
        or allowlist.get("allowlist_preparation_identity_sha256") != preparation_identity
        or tuple(allowlist.get("arms", ())) != ARMS
        or tuple(allowlist.get("seeds", ())) != e9.SEEDS
        or allowlist.get("split") != "test"
        or allowlist.get("count") != COUNT
        or allowlist.get("samp_id_count") != SUBJECTS
        or allowlist.get("test_sample_seed") != TEST_SAMPLE_SEED
        or allowlist.get("row_order_sha256") != ROW_ORDER_SHA256
        or allowlist.get("batch_size") != 128
        or allowlist.get("amp_dtype") != "bfloat16"
        or allowlist.get("include_test_only") is not True
        or allowlist.get("training_implementation_lock_sha256")
        != current_formal_lock_hash
        or allowlist.get("cache", {}).get("identity") != FROZEN_RESEARCH_TEST_CACHE_IDENTITY
    ):
        raise ValueError("E9 research-test allowlist 合同漂移")
    expected = [(arm, seed) for seed in e9.SEEDS for arm in ARMS]
    observed = [(item["arm"], int(item["seed"])) for item in allowlist["entries"]]
    if observed != expected:
        raise ValueError("E9 research-test allowlist 18-cell 顺序或身份漂移")
    for relative, expected_identity in allowlist["code_files"].items():
        if engineering._identity(e9.ROOT / relative) != expected_identity:
            raise ValueError(f"E9 research-test 代码身份漂移: {relative}")
    return allowlist, engineering._sha256(allowlist_path)


def _entry(lock: Mapping[str, Any], arm: str, seed: int) -> Mapping[str, Any]:
    if arm not in ARMS or int(seed) not in e9.SEEDS:
        raise ValueError("E9 research-test arm/seed 不属于固定矩阵")
    return next(
        item
        for item in lock["entries"]
        if item["arm"] == arm and int(item["seed"]) == int(seed)
    )


def _verify_evaluation_sources(lock: Mapping[str, Any], entry: Mapping[str, Any]) -> None:
    for item in entry["files"].values():
        if engineering._identity(Path(item["path"])) != {
            key: item[key] for key in ("size_bytes", "sha256")
        }:
            raise ValueError("E9 research-test checkpoint 来源漂移")
    cache_root = Path(lock["cache"]["root"])
    manifest_entry = lock["cache"]["manifest"]
    if engineering._identity(Path(manifest_entry["path"])) != {
        key: manifest_entry[key] for key in ("size_bytes", "sha256")
    }:
        raise ValueError("E9 research-test cache manifest 漂移")
    for filename, expected in lock["cache"]["files"].items():
        if engineering._identity(cache_root / filename) != {
            key: expected[key] for key in ("size_bytes", "sha256")
        }:
            raise ValueError(f"E9 research-test cache 文件漂移: {filename}")
    index = lock["dataset_index"]
    if engineering._sha256(Path(index["path"])) != index["sha256"]:
        raise ValueError("E9 research-test dataset index identity 漂移")


def evaluation_config(
    entry: Mapping[str, Any],
    lock: Mapping[str, Any],
    *,
    device: str,
) -> tuple[DictConfig, DictConfig]:
    config_path = Path(entry["files"]["config"]["path"])
    expected = entry["files"]["config"]
    if engineering._identity(config_path) != {
        key: expected[key] for key in ("size_bytes", "sha256")
    }:
        raise ValueError("E9 research-test config 来源漂移")
    cfg = OmegaConf.load(config_path)
    if OmegaConf.to_container(cfg, resolve=True) != entry["training_config"]:
        raise ValueError("E9 research-test config 与 allowlist 不一致")
    formal_attempt = Path(entry["formal_attempt"])
    formal_lock = formal.load_formal_lock()[0]
    baseline = OmegaConf.create(formal_lock["baselines"][str(entry["seed"])])
    e9.validate_config(
        cfg,
        baseline,
        arm=entry["arm"],
        output_root=formal_attempt / "training",
        device=str(cfg.training.device),
    )
    if (
        str(cfg.data.test_split) != "test"
        or cfg.data.max_test_windows is not None
        or int(cfg.data.test_sample_seed) != TEST_SAMPLE_SEED
        or int(cfg.training.batch_size) != 128
        or int(cfg.training.gradient_accumulation_steps) != 1
        or not bool(cfg.training.use_amp)
        or str(cfg.training.amp_dtype) != "bfloat16"
    ):
        raise ValueError("E9 research-test 原生推理配置漂移")
    cfg.training.device = device
    cfg.training.show_progress = False
    data_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    data_cfg.data.tf_research_test_cache_path = lock["cache"]["root"]
    return cfg, data_cfg


def run_evaluation(
    allowlist_path: Path,
    *,
    arm: str,
    seed: int,
    device: str = "cuda:0",
) -> Path:
    lock, allowlist_hash = load_allowlist(allowlist_path)
    entry = _entry(lock, arm, seed)
    parent = _research_test_root() / "evaluation" / arm / f"seed_{seed}"
    with evidence_attempt(
        parent,
        phase="evaluation",
        evidence_hash=allowlist_hash,
        arm=arm,
        seed=int(seed),
        reject_completed=True,
    ) as output:
        runtime = _runtime_preflight(device)
        training_environment = json.loads(
            (Path(entry["formal_attempt"]) / "environment.json").read_text(encoding="utf-8")
        )
        compatibility = formal._runtime_compatibility(runtime, training_environment)
        engineering._write_json(output / "environment.json", runtime)
        engineering._write_json(
            output / "allowlist_source.json",
            {
                "path": str(allowlist_path.resolve()),
                "allowlist": engineering._identity(allowlist_path / "allowlist.json"),
                "runtime_compatibility": compatibility,
            },
        )
        engineering._write_json(
            output / "access_started.json",
            {
                "split": "test",
                "arm": arm,
                "seed": int(seed),
                "rows": COUNT,
                "checkpoint": entry["files"]["checkpoint"],
                "cache": lock["cache"],
                "evidence_role": lock["evidence_role"],
            },
        )
        _verify_evaluation_sources(lock, entry)
        cfg, data_cfg = evaluation_config(entry, lock, device=device)
        OmegaConf.save(data_cfg, output / "resolved_config.yaml")
        checkpoint = torch.load(
            Path(entry["files"]["checkpoint"]["path"]),
            map_location="cpu",
            weights_only=False,
        )
        if (
            int(checkpoint["epoch"]) != int(entry["selected_epoch"])
            or checkpoint["config"] != entry["training_config"]
        ):
            raise ValueError("E9 research-test checkpoint identity 漂移")
        _validate_checkpoint_config(checkpoint.get("config"), cfg)
        engineering.finite_tree(checkpoint["model_state_dict"], label="research_test_checkpoint")
        model = build_e9_latent_width_condition_refiner_model(cfg)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)

        audited = read_research_v2_index(cfg.data.dataset_root, cfg.data.index_csv, cfg)
        rows = filter_index(
            audited,
            cfg,
            split="test",
            max_windows=None,
            sample_strategy=str(cfg.data.test_sample_strategy),
            sample_seed=int(cfg.data.test_sample_seed),
        )
        check_test_rows(rows)
        if set(rows.samp_id.astype(int)) & set(entry["development_samp_ids"]):
            raise ValueError("E9 research-test 与 development subject 交叉")
        rows.to_csv(output / "test_rows.csv", index=False)
        data = build_window_data(
            data_cfg,
            split="test",
            max_windows=None,
            sample_strategy=str(cfg.data.test_sample_strategy),
            sample_seed=int(cfg.data.test_sample_seed),
            shuffle=False,
            audited=audited,
        )
        check_test_rows(data.rows, rows)
        if len(data.dataset) != COUNT:
            raise ValueError("E9 research-test dataset 样本集合不完整")
        predictions = collect_predictions(
            model,
            guarded_batches(data.loader, rows),
            device=device,
            max_windows=COUNT,
            use_amp=True,
        )
        metrics = evaluate_task_predictions(
            predictions,
            cfg,
            include_test_only=True,
            method=arm,
        )
        summary = check_test_metrics(metrics, rows)
        quality = quality_flags(metrics)
        metrics.insert(0, "arm", arm)
        metrics.insert(1, "seed", int(seed))
        for key, value in reversed(
            (
                ("arm", arm),
                ("seed", int(seed)),
                ("split", "test"),
                ("selected_epoch", int(entry["selected_epoch"])),
                ("quality_acceptance_passed", quality["quality_acceptance_passed"]),
            )
        ):
            summary.insert(0, key, value)
        metrics.to_csv(output / "metrics.csv", index=False)
        summary.to_csv(output / "metrics_summary.csv", index=False)
        engineering._write_json(
            output / "access_receipt.json",
            {
                "split": "test",
                "arm": arm,
                "seed": int(seed),
                "rows": COUNT,
                "samp_ids": SUBJECTS,
                "row_order_sha256": formal.sf_formal.array_hash(rows.dataset_row_id.to_numpy()),
                "checkpoint": entry["files"]["checkpoint"],
                "cache_identity": FROZEN_RESEARCH_TEST_CACHE_IDENTITY,
                "test_sample_seed": TEST_SAMPLE_SEED,
                "evidence_role": lock["evidence_role"],
            },
        )
        engineering._write_json(
            output / "evaluation_receipt.json",
            {
                "protocol": PROTOCOL,
                "allowlist_sha256": allowlist_hash,
                "arm": arm,
                "seed": int(seed),
                "selected_epoch": int(entry["selected_epoch"]),
                "checkpoint_sha256": entry["files"]["checkpoint"]["sha256"],
                "rows": COUNT,
                "samp_ids": SUBJECTS,
                "row_order_sha256": ROW_ORDER_SHA256,
                "primary_finite": True,
                "secondary_finite": True,
                **quality,
                "evidence_role": lock["evidence_role"],
            },
        )
    return output


def evaluation_status(allowlist_path: Path) -> dict[str, Any]:
    lock, allowlist_hash = load_allowlist(allowlist_path)
    cells: list[dict[str, Any]] = []
    for seed in e9.SEEDS:
        for arm in ARMS:
            parent = _research_test_root() / "evaluation" / arm / f"seed_{seed}"
            completed: list[str] = []
            failed: list[str] = []
            running: list[str] = []
            if parent.is_dir():
                for attempt in sorted(path for path in parent.iterdir() if path.is_dir()):
                    started = attempt / "lifecycle_started.json"
                    if not started.is_file():
                        continue
                    context = json.loads(started.read_text(encoding="utf-8"))
                    if context.get("evidence_identity_sha256") != allowlist_hash:
                        continue
                    if (attempt / "freeze_receipt.json").is_file():
                        verify_attempt(
                            attempt,
                            phase="evaluation",
                            evidence_hash=allowlist_hash,
                            arm=arm,
                            seed=seed,
                        )
                        completed.append(str(attempt))
                    elif (attempt / "lifecycle_failed.json").is_file():
                        failed.append(str(attempt))
                    else:
                        running.append(str(attempt))
            if len(completed) > 1:
                raise ValueError(f"E9 research-test cell 多个完成 attempt: {arm}/{seed}")
            status = "completed" if completed else "running" if running else "failed" if failed else "pending"
            cells.append(
                {
                    "arm": arm,
                    "seed": seed,
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
        "protocol": PROTOCOL,
        "allowlist_sha256": allowlist_hash,
        "counts": counts,
        "cells": cells,
        "entries": len(lock["entries"]),
    }


def _evaluation_attempts(allowlist_path: Path) -> tuple[list[Path], str]:
    status = evaluation_status(allowlist_path)
    if status["counts"] != {"pending": 0, "running": 0, "failed": 0, "completed": 18}:
        raise RuntimeError(f"E9 research-test summary 要求完整矩阵，当前={status['counts']}")
    return [Path(cell["completed"][0]) for cell in status["cells"]], status["allowlist_sha256"]


def _secondary_across_seed(seed_summaries: pd.DataFrame) -> pd.DataFrame:
    identity = {
        "arm",
        "seed",
        "split",
        "selected_epoch",
        "quality_acceptance_passed",
        "method",
    }
    value_columns = [
        column
        for column in seed_summaries.columns
        if column not in identity and column.endswith("_mean")
    ]
    rows: list[dict[str, Any]] = []
    for arm, group in seed_summaries.groupby("arm", sort=False):
        group = group.set_index("seed").loc[list(e9.SEEDS)]
        row: dict[str, Any] = {"arm": arm, "seed_count": len(e9.SEEDS)}
        for column in value_columns:
            values = group[column].to_numpy(float)
            row[column + "_seed_mean"] = float(values.mean()) if np.isfinite(values).all() else np.nan
            row[column + "_seed_sample_sd"] = (
                float(values.std(ddof=1)) if np.isfinite(values).all() else np.nan
            )
            row[column + "_finite_seeds"] = int(np.isfinite(values).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _validation_test_comparison(test_seed: pd.DataFrame) -> pd.DataFrame:
    validation = pd.read_csv(_p5_summary_path() / "seed_primary_metrics.csv").set_index(
        ["arm", "seed"]
    )
    test = test_seed.set_index(["arm", "seed"])
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for seed in e9.SEEDS:
            for metric in p5.PRIMARY:
                before = float(validation.loc[(arm, seed), metric])
                after = float(test.loc[(arm, seed), metric])
                rows.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "metric": metric,
                        "validation": before,
                        "test": after,
                        "test_minus_validation": after - before,
                    }
                )
    return pd.DataFrame(rows)


def summarize(allowlist_path: Path) -> Path:
    lock, allowlist_hash = load_allowlist(allowlist_path)
    state = _test_git_state()
    attempts, observed_hash = _evaluation_attempts(allowlist_path)
    if observed_hash != allowlist_hash:
        raise ValueError("E9 research-test status allowlist identity 漂移")
    frames: list[pd.DataFrame] = []
    seed_summaries: list[pd.DataFrame] = []
    sources: dict[str, Any] = {}
    reference: pd.DataFrame | None = None
    for attempt in attempts:
        receipt = json.loads((attempt / "evaluation_receipt.json").read_text(encoding="utf-8"))
        arm, seed = str(receipt["arm"]), int(receipt["seed"])
        verify_attempt(
            attempt,
            phase="evaluation",
            evidence_hash=allowlist_hash,
            arm=arm,
            seed=seed,
        )
        entry = _entry(lock, arm, seed)
        if (
            receipt["allowlist_sha256"] != allowlist_hash
            or receipt["selected_epoch"] != entry["selected_epoch"]
            or receipt["checkpoint_sha256"] != entry["files"]["checkpoint"]["sha256"]
            or receipt["rows"] != COUNT
        ):
            raise ValueError("E9 research-test evaluation receipt 与 allowlist 不一致")
        rows = pd.read_csv(attempt / "test_rows.csv")
        metrics = pd.read_csv(attempt / "metrics.csv")
        check_test_rows(rows)
        check_test_rows(metrics, rows)
        computed = check_test_metrics(metrics, rows)
        quality = quality_flags(metrics)
        if any(receipt.get(key) != value for key, value in quality.items()):
            raise ValueError("E9 research-test quality receipt 漂移")
        if reference is None:
            reference = metrics[[*IDENTITY_COLUMNS, *TARGET_COLUMNS]].copy()
        else:
            for key in (*IDENTITY_COLUMNS, *TARGET_COLUMNS):
                if not np.array_equal(metrics[key].to_numpy(), reference[key].to_numpy()):
                    raise ValueError(f"E9 research-test 全矩阵 target identity 漂移: {key}")
        saved = pd.read_csv(attempt / "metrics_summary.csv")
        for metric in p5.PRIMARY:
            if (
                not np.isclose(
                    saved.iloc[0][metric + "_mean"],
                    computed.iloc[0][metric + "_mean"],
                    atol=1e-12,
                    rtol=0,
                )
                or int(saved.iloc[0][metric + "_n"])
                != int(computed.iloc[0][metric + "_n"])
            ):
                raise ValueError("E9 research-test saved summary 漂移")
        frames.append(metrics)
        seed_summaries.append(saved)
        sources[f"{arm}/{seed}"] = {
            "attempt": str(attempt),
            "manifest": engineering._identity(attempt / "manifest.json"),
            "metrics": engineering._identity(attempt / "metrics.csv"),
            "selected_epoch": entry["selected_epoch"],
            "checkpoint_sha256": entry["files"]["checkpoint"]["sha256"],
        }
    combined = pd.concat(frames, ignore_index=True)
    if len(combined) != 18 * COUNT:
        raise ValueError("E9 research-test metrics 总行数错误")
    seed_primary = p5.seed_primary_metrics(combined, expected_rows=COUNT)
    arm_summary = p5.arm_primary_summary(seed_primary)
    arm_comparison = p5.arm_reference_comparison(seed_primary)
    contrasts_seed = p5.planned_contrasts_by_seed(seed_primary)
    contrasts_across = p5.planned_contrasts_across_seed(contrasts_seed)
    tails = sf.local_rr_tail_summary(combined)
    tail_across = p5.local_rr_tail_across_seed(tails)
    subject_metrics = sf.subject_stratified_metrics(combined)
    macro_seed, macro_across = _subject_macro_tables(subject_metrics)
    denominators = sf.metric_denominators(combined)
    secondary_seed = pd.concat(seed_summaries, ignore_index=True)
    secondary_across = _secondary_across_seed(secondary_seed)
    validation_test = _validation_test_comparison(seed_primary)
    test_decision = p5.build_decision(seed_primary, contrasts_across)
    validation_decision_path = _p5_summary_path() / "decision.json"
    decision = {
        "protocol": PROTOCOL,
        "evidence_role": lock["evidence_role"],
        "checkpoint_selection": "fixed by validation before test access",
        "validation_decision": {
            "path": str(validation_decision_path),
            **engineering._identity(validation_decision_path),
        },
        "test_decision_descriptive": test_decision,
        "model_or_checkpoint_reselection_performed": False,
        "independent_confirmation_claimed": False,
    }
    parent = _research_test_root() / "summary"
    with evidence_attempt(
        parent,
        phase="summary",
        evidence_hash=allowlist_hash,
        reject_completed=True,
    ) as output:
        outputs = {
            "seed_primary_metrics.csv": seed_primary,
            "arm_primary_summary.csv": arm_summary,
            "arm_reference_comparison.csv": arm_comparison,
            "planned_contrasts_by_seed.csv": contrasts_seed,
            "planned_contrasts_across_seed.csv": contrasts_across,
            "local_rr_tail_summary.csv": tails,
            "local_rr_tail_across_seed.csv": tail_across,
            "subject_stratified_metrics.csv": subject_metrics,
            "subject_macro_by_seed.csv": macro_seed,
            "subject_macro_across_seed.csv": macro_across,
            "metric_denominators.csv": denominators,
            "secondary_seed_metrics.csv": secondary_seed,
            "secondary_across_seed.csv": secondary_across,
            "validation_test_comparison.csv": validation_test,
        }
        for filename, frame in outputs.items():
            frame.to_csv(output / filename, index=False, na_rep="NA")
        engineering._write_json(output / "decision.json", decision)
        engineering._write_json(
            output / "source_manifest.json",
            {
                "allowlist": {
                    "path": str(allowlist_path.resolve()),
                    "sha256": allowlist_hash,
                },
                "evaluation_runs": sources,
                "validation_summary": lock["validation_summary"],
                "summary_git": state,
            },
        )
        engineering._write_json(
            output / "access_receipt.json",
            {
                "evaluation_attempts_read": 18,
                "test_metric_rows_read": len(combined),
                "test_arrays_read": False,
                "checkpoint_content_read": False,
                "gpu_used": False,
                "model_inference_used": False,
                "evidence_role": lock["evidence_role"],
            },
        )
        engineering._write_json(
            output / "summary_receipt.json",
            {
                "protocol": PROTOCOL,
                "status": "complete",
                "allowlist_sha256": allowlist_hash,
                "arms": list(ARMS),
                "seeds": list(e9.SEEDS),
                "evaluation_attempts": 18,
                "rows_per_attempt": COUNT,
                "test_metric_rows": len(combined),
                "evidence_role": lock["evidence_role"],
                "aggregation": {
                    "window_primary": "eligible sample direct mean within seed",
                    "seed": "arithmetic mean and sample SD ddof=1",
                    "subject_macro": "equal weight across eight test samp_id",
                    "inference": "descriptive; no p-values",
                },
                "decision": decision,
            },
        )
    return output


def _subject_macro_tables(
    subject_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for (arm, seed, metric), group in subject_metrics.groupby(
        ["arm", "seed", "metric"], sort=False
    ):
        if len(group) != SUBJECTS or group.samp_id.nunique() != SUBJECTS:
            raise ValueError(f"E9 test subject macro 不完整: {arm}/{seed}/{metric}")
        values = group["mean"].to_numpy(float)
        rows.append(
            {
                "arm": arm,
                "seed": int(seed),
                "metric": metric,
                "subject_count": len(values),
                "subject_macro_mean": float(values.mean()),
                "subject_sample_sd": float(values.std(ddof=1)),
                "subject_min": float(values.min()),
                "subject_max": float(values.max()),
            }
        )
    by_seed = pd.DataFrame(rows)
    across: list[dict[str, Any]] = []
    for (arm, metric), group in by_seed.groupby(["arm", "metric"], sort=False):
        values = group.set_index("seed").loc[list(e9.SEEDS)].subject_macro_mean.to_numpy(float)
        across.append(
            {
                "arm": arm,
                "metric": metric,
                "seed_count": len(values),
                "subject_macro_seed_mean": float(values.mean()),
                "subject_macro_seed_sample_sd": float(values.std(ddof=1)),
            }
        )
    return by_seed, pd.DataFrame(across)
