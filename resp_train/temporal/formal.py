from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from resp_train.crd.config import check_crd_dependencies
from resp_train.crd.training import build_crd_optimizer, optimizer_updates_per_epoch, train_crd_one_epoch
from resp_train.data.factory import ThoDataBundle, build_window_data
from resp_train.data.research_v2 import adapt_research_v2_index, summarize_research_v2_audit
from resp_train.engine import collect_predictions, save_checkpoint, validate
from resp_train.losses.task import RespirationTaskLoss
from resp_train.metrics.task import evaluate_task_predictions, summarize_task_metrics, validation_local_rr_mean
from resp_train.temporal.config import FORMAL_SEEDS, load_resp_temporal_config
from resp_train.temporal.gpu_engineering import (
    _cuda_identity,
    load_gpu_engineering_config,
    sha256_file,
    validate_access_receipt as validate_gpu_receipt,
)
from resp_train.temporal.model import build_resp_temporal_model, trainable_parameter_count
from resp_train.utils.run import save_config, set_seed


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN_PATH = REPO_ROOT / "configs/resp_temporal_v1/formal_v1.yaml"
PLAN_SCHEMA_VERSION = "rtm-v1-formal-plan-v1"
FORMAL_PROTOCOL_ID = "resp-temporal-v1-formal-validation-20260820"
FORMAL_RECEIPT_SCHEMA_VERSION = "rtm-v1-formal-run-receipt-v1"
FROZEN_PLAN_SHA256 = "fb1d4652c3e65bbcb1eb4b4b770e96315dbef04011cdae8e66391a33265d0697"
EXPECTED_CANDIDATES = (
    "rtm_v1_t0_locked_stem_head",
    "rtm_v1_tcn_d9_h384",
    "rtm_v1_bimamba2_d96_l6",
    "rtm_v1_bilstm_h96_l2",
    "rtm_v1_multiscale_10_2_1_h384",
)


@dataclass(frozen=True)
class FormalPlan:
    path: Path
    sha256: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class FormalRunSpec:
    plan: FormalPlan
    candidate_record: dict[str, Any]
    seed: int
    cfg: DictConfig
    run_dir: Path
    git_commit: str
    gpu_device_info: dict[str, Any]


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(f"{context} 字段必须严格为 {sorted(expected)}，实际为 {sorted(actual)}")


def _repo_path(value: Any, *, context: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{context} 必须是仓库内相对路径")
    return REPO_ROOT / relative


def _is_lower_hex(value: Any, *, length: int) -> bool:
    rendered = str(value)
    return len(rendered) == length and all(character in "0123456789abcdef" for character in rendered)


def validate_formal_plan(raw: Mapping[str, Any]) -> None:
    _require_exact_keys(
        raw,
        {
            "schema_version",
            "protocol_id",
            "role",
            "evidence_label",
            "authorization",
            "provenance",
            "candidates",
            "seeds",
            "data",
            "training",
            "selector",
            "access",
            "output",
        },
        "formal plan",
    )
    if raw["schema_version"] != PLAN_SCHEMA_VERSION or raw["protocol_id"] != FORMAL_PROTOCOL_ID:
        raise ValueError("formal plan schema/protocol 不匹配")
    if raw["role"] != "formal_training" or raw["evidence_label"] != "validation-development evidence":
        raise ValueError("formal plan evidence role 漂移")
    if raw["authorization"] != {
        "formal_training_implementation": True,
        "user_manual_formal_execution": True,
        "codex_formal_execution": False,
        "validation_within_training": True,
        "standalone_validation_evaluation": False,
        "research_test_evaluation": False,
    }:
        raise ValueError("formal authorization 越界")
    provenance = raw["provenance"]
    _require_exact_keys(
        provenance,
        {
            "formal_protocol_path",
            "formal_protocol_sha256",
            "signal_substrate_lock_path",
            "signal_substrate_lock_sha256",
            "candidate_lock_path",
            "candidate_lock_sha256",
            "cpu_implementation_receipt_path",
            "cpu_implementation_receipt_sha256",
            "gpu_engineering_config_path",
            "gpu_engineering_config_sha256",
            "gpu_engineering_lock_path",
            "gpu_engineering_lock_sha256",
            "gpu_execution_receipt_path",
            "gpu_execution_receipt_sha256",
            "gpu_execution_manifest_path",
            "gpu_execution_manifest_sha256",
            "require_clean_git",
        },
        "formal provenance",
    )
    if provenance != {
        "formal_protocol_path": "docs/experiments/resp_temporal_v1_formal_protocol_20260820.md",
        "formal_protocol_sha256": "208b8e0f80c4a2bd426e215567702e41ee78f768e513fdb65266be3d8757c05d",
        "signal_substrate_lock_path": "docs/experiments/resp_temporal_v1_signal_substrate_lock_20260820.json",
        "signal_substrate_lock_sha256": "11bfcad00f4532d4bdfe1413a375b5f06f46eb8ac67dfcd475701872322fee69",
        "candidate_lock_path": "docs/experiments/resp_temporal_v1_candidate_lock_20260820.json",
        "candidate_lock_sha256": "b4a2c83310fa2ce9519e3ca25814aea0b179458ab52d6380a932545c99c25f9b",
        "cpu_implementation_receipt_path": "docs/experiments/resp_temporal_v1_cpu_implementation_receipt_20260820.json",
        "cpu_implementation_receipt_sha256": "6ef3ca0d48e1ac819ff55bab4247d542c8bae0adfddb6951c8a026e81fea7872",
        "gpu_engineering_config_path": "configs/resp_temporal_v1/gpu_engineering_v2.yaml",
        "gpu_engineering_config_sha256": "081fbb0e350997e97838faa5a54478d2a81041dfc1769336573a3a0d267eb8f0",
        "gpu_engineering_lock_path": "docs/experiments/resp_temporal_v1_gpu_engineering_lock_20260820.json",
        "gpu_engineering_lock_sha256": "5632b0e404f943e61c646a8ddcf1761c2391884412915ef1d79aa342f8bc0183",
        "gpu_execution_receipt_path": "runs/resp_temporal_v1/gpu_engineering/rtm_v1_gpu_engineering_v2/access_receipt.json",
        "gpu_execution_receipt_sha256": "c8d34d5f1a9f944af58945f74110b0c9ff74e15696c7a7f240465ce4c83de38a",
        "gpu_execution_manifest_path": "runs/resp_temporal_v1/gpu_engineering/rtm_v1_gpu_engineering_v2/artifact_manifest.json",
        "gpu_execution_manifest_sha256": "e90df905b2708853a761328995c4fd0c6db1e2564b1bb16eb9328ed709f25812",
        "require_clean_git": True,
    }:
        raise ValueError("formal provenance 漂移")
    expected_candidates = [
        (
            "rtm_v1_t0_locked_stem_head",
            "configs/resp_temporal_v1/rtm_v1_t0_locked_stem_head.yaml",
            "eb7cab872305459f62dac7eb0eb15e699690ea03e5ff7d33b581666ed7a31be6",
        ),
        (
            "rtm_v1_tcn_d9_h384",
            "configs/resp_temporal_v1/rtm_v1_tcn_d9_h384.yaml",
            "b838396563580bfe19d211bc9cfd86ad09228a1f72a28084c25e003b67077519",
        ),
        (
            "rtm_v1_bimamba2_d96_l6",
            "configs/resp_temporal_v1/rtm_v1_bimamba2_d96_l6.yaml",
            "18c98c28240fc5f983feef301e174b13b3be5829f7cb723acd032e412e7917f5",
        ),
        (
            "rtm_v1_bilstm_h96_l2",
            "configs/resp_temporal_v1/rtm_v1_bilstm_h96_l2.yaml",
            "05dde416538ea0920f9f4f01b6462baf8e95575551ad1a71c2089315d17b9bb7",
        ),
        (
            "rtm_v1_multiscale_10_2_1_h384",
            "configs/resp_temporal_v1/rtm_v1_multiscale_10_2_1_h384.yaml",
            "cdef3bd3ac3dd7a0e259d8182bfe7d4aa91944cc5391b945835bfe29bf5a7968",
        ),
    ]
    observed_candidates = [
        (record["candidate_id"], record["config_path"], record["config_sha256"])
        for record in raw["candidates"]
    ]
    if observed_candidates != expected_candidates:
        raise ValueError("formal candidate identity/order/hash 漂移")
    if list(raw["seeds"]) != list(FORMAL_SEEDS):
        raise ValueError("formal seeds 漂移")
    if raw["data"] != {
        "shared_index_metadata_read": True,
        "signal_splits": ["train", "val"],
        "expected_train_windows": 10141,
        "expected_validation_windows": 2675,
        "expected_train_samp_ids": 32,
        "expected_validation_samp_ids": 7,
        "expected_train_batches": 80,
        "expected_validation_batches": 21,
        "expected_train_row_ids_sha256": "f290e569140a2ff7745cf1a5cfa6a4da943644d76498c9b85517d3ae0702c45e",
        "expected_validation_row_ids_sha256": "b68a51b101bb80033c4de18c9c21cdfbb8924d1bfe134330f047617cf3b0915a",
    }:
        raise ValueError("formal data identity/count/split contract 漂移")
    if raw["training"] != {
        "epochs": 80,
        "expected_updates_per_epoch": 80,
        "expected_total_updates": 6400,
        "physical_batch_size": 128,
        "accumulation_steps": 1,
        "effective_batch_size": 128,
        "device": "cuda:0",
        "optimizer": "adamw",
        "max_learning_rate": 3e-4,
        "min_learning_rate": 3e-5,
        "warmup_fraction": 0.05,
        "lr_schedule": "step_exact_warmup_cosine",
        "adam_betas": [0.9, 0.999],
        "adam_eps": 1e-8,
        "weight_decay": 1e-4,
        "grad_clip_norm": 1.0,
        "use_amp": True,
        "amp_dtype": "bfloat16",
        "allow_tf32": False,
        "cudnn_benchmark": False,
        "early_stopping": False,
        "resume": False,
        "drop_last": False,
    }:
        raise ValueError("formal training contract 漂移")
    if raw["selector"] != {
        "split": "validation",
        "metric": "local_rr_mae_full_split_sample_direct_mean",
        "comparison": "strict_less_than",
        "tie_updates_checkpoint": False,
        "checkpoint_filename": "checkpoint_best_local_rr.pt",
    }:
        raise ValueError("formal selector 漂移")
    if raw["access"] != {
        "allowed_splits": ["train", "val"],
        "research_test_allowed": False,
        "checkpoint_input_allowed": False,
        "own_best_checkpoint_reload_after_training": True,
    }:
        raise ValueError("formal access contract 漂移")
    if raw["output"] != {
        "root": "runs/resp_temporal_v1/formal",
        "allow_overwrite": False,
        "allow_resume": False,
    }:
        raise ValueError("formal output contract 漂移")


def load_formal_plan(path: str | Path = DEFAULT_PLAN_PATH) -> FormalPlan:
    plan_path = Path(path).resolve()
    if plan_path != DEFAULT_PLAN_PATH.resolve():
        raise ValueError(f"RTM-v1 formal只允许冻结plan: {DEFAULT_PLAN_PATH}")
    if not plan_path.is_file():
        raise FileNotFoundError(f"formal plan不存在: {plan_path}")
    actual_sha = sha256_file(plan_path)
    if actual_sha != FROZEN_PLAN_SHA256:
        raise ValueError(f"formal plan SHA-256漂移: {actual_sha}")
    raw = OmegaConf.to_container(OmegaConf.load(plan_path), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("formal plan顶层必须是mapping")
    validate_formal_plan(raw)
    return FormalPlan(path=plan_path, sha256=actual_sha, raw=raw)


def formal_run_dir(plan: FormalPlan, candidate_id: str, seed: int) -> Path:
    if candidate_id not in EXPECTED_CANDIDATES or int(seed) not in FORMAL_SEEDS:
        raise ValueError("candidate/seed不属于RTM-v1 formal 15-run allowlist")
    return _repo_path(plan.raw["output"]["root"], context="formal output root") / candidate_id / f"seed_{int(seed)}"


def derive_formal_config(plan: FormalPlan, candidate_id: str, seed: int) -> tuple[DictConfig, dict[str, Any]]:
    if candidate_id not in EXPECTED_CANDIDATES or int(seed) not in FORMAL_SEEDS:
        raise ValueError("candidate/seed不属于RTM-v1 formal 15-run allowlist")
    record = next(item for item in plan.raw["candidates"] if item["candidate_id"] == candidate_id)
    candidate_path = _repo_path(record["config_path"], context="formal candidate config")
    if sha256_file(candidate_path) != record["config_sha256"]:
        raise RuntimeError("formal candidate config hash漂移")
    base = load_resp_temporal_config(candidate_path)
    cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
    cfg.protocol.stage = "formal"
    cfg.protocol.run_role = "formal_training"
    cfg.protocol.gpu_engineering_enabled = True
    cfg.protocol.formal_training_enabled = True
    cfg.protocol.validation_evaluation_enabled = True
    cfg.protocol.research_test_enabled = False
    cfg.window.target_fs = 100
    cfg.model.initialization_seed = int(seed)
    cfg.training.seed = int(seed)
    cfg.training.device = str(plan.raw["training"]["device"])
    cfg.training.batch_size = int(plan.raw["training"]["physical_batch_size"])
    cfg.training.gradient_accumulation_steps = int(plan.raw["training"]["accumulation_steps"])
    cfg.training.effective_batch_size = int(plan.raw["training"]["effective_batch_size"])
    cfg.training.epochs = int(plan.raw["training"]["epochs"])
    cfg.training.optimizer_updates = int(plan.raw["training"]["expected_total_updates"])
    cfg.outputs.run_root = str(formal_run_dir(plan, candidate_id, seed))
    cfg.formal = OmegaConf.create(
        {
            "plan_path": str(plan.path.relative_to(REPO_ROOT)),
            "plan_sha256": plan.sha256,
            "candidate_config_path": str(candidate_path.relative_to(REPO_ROOT)),
            "candidate_config_sha256": record["config_sha256"],
            "gpu_engineering_lock_sha256": plan.raw["provenance"]["gpu_engineering_lock_sha256"],
            "selector": plan.raw["selector"],
            "allowed_splits": ["train", "val"],
            "research_test_allowed": False,
        }
    )
    return cfg, dict(record)


def strict_selector_improved(value: float, best: float) -> bool:
    if not math.isfinite(float(value)) or math.isnan(float(best)):
        raise ValueError("selector输入必须finite且best不能为NaN")
    return float(value) < float(best)


def _git_identity() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=False, capture_output=True, text=True
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, check=False, capture_output=True, text=True
    )
    if commit.returncode != 0 or status.returncode != 0:
        raise RuntimeError("无法读取formal Git identity")
    return commit.stdout.strip(), bool(status.stdout.strip())


def _verify_formal_provenance(plan: FormalPlan) -> tuple[str, dict[str, Any]]:
    provenance = plan.raw["provenance"]
    verified: dict[str, str] = {}
    for prefix in (
        "formal_protocol",
        "signal_substrate_lock",
        "candidate_lock",
        "cpu_implementation_receipt",
        "gpu_engineering_config",
        "gpu_engineering_lock",
        "gpu_execution_receipt",
        "gpu_execution_manifest",
    ):
        path = _repo_path(provenance[f"{prefix}_path"], context=prefix)
        actual = sha256_file(path)
        expected = provenance[f"{prefix}_sha256"]
        if actual != expected:
            raise RuntimeError(f"formal provenance hash漂移: {path}: {actual} != {expected}")
        verified[f"{prefix}_sha256"] = actual
    gpu_lock_path = _repo_path(provenance["gpu_engineering_lock_path"], context="gpu lock")
    gpu_lock = json.loads(gpu_lock_path.read_text(encoding="utf-8"))
    if (
        gpu_lock.get("decision") != "all_five_hardware_feasible_at_128x1"
        or not bool(gpu_lock.get("user_confirmation", {}).get("confirmed"))
        or bool(gpu_lock.get("authorization", {}).get("formal_training_execution_allowed"))
    ):
        raise RuntimeError("GPU engineering lock identity/authorization漂移")
    gpu_receipt_path = _repo_path(provenance["gpu_execution_receipt_path"], context="gpu receipt")
    gpu_receipt = json.loads(gpu_receipt_path.read_text(encoding="utf-8"))
    validate_gpu_receipt(gpu_receipt)
    if (
        gpu_receipt["status"] != "complete"
        or gpu_receipt["counts"] != {
            "actual_candidates": 5,
            "expected_candidates": 5,
            "passed_candidates": 5,
            "unavailable_candidates": 0,
        }
        or any(row["physical_batch_size"] != 128 or row["accumulation_steps"] != 1 for row in gpu_receipt["candidate_summary"])
    ):
        raise RuntimeError("GPU execution receipt不支持formal 128x1合同")
    commit, dirty = _git_identity()
    if dirty:
        raise RuntimeError("RTM-v1 formal要求干净Git工作树")
    return commit, verified


def preflight_formal_run(
    *, plan_path: str | Path, candidate_id: str, seed: int
) -> FormalRunSpec:
    plan = load_formal_plan(plan_path)
    cfg, record = derive_formal_config(plan, candidate_id, seed)
    run_dir = formal_run_dir(plan, candidate_id, seed)
    if run_dir.exists():
        raise FileExistsError(f"formal run目录禁止覆盖/resume: {run_dir}")
    commit, _ = _verify_formal_provenance(plan)
    problems = check_crd_dependencies()
    if problems:
        raise RuntimeError("; ".join(problems))
    gpu_cfg = load_gpu_engineering_config()
    _, device_info = _cuda_identity(gpu_cfg.raw)
    return FormalRunSpec(
        plan=plan,
        candidate_record=record,
        seed=int(seed),
        cfg=cfg,
        run_dir=run_dir,
        git_commit=commit,
        gpu_device_info=device_info,
    )


def _json_write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _json_write_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    _json_write(temporary, value)
    os.replace(temporary, path)


def _csv_write_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _row_id_sha256(rows: pd.DataFrame) -> str:
    if "dataset_row_id" not in rows:
        raise KeyError("formal data rows缺少dataset_row_id")
    values = rows["dataset_row_id"].to_numpy(dtype=np.int64, copy=True)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _build_formal_data(cfg: DictConfig) -> ThoDataBundle:
    """读取共享索引元数据，但仅让train/validation行进入适配器和signal reader。"""

    index_path = Path(str(cfg.data.dataset_root)) / str(cfg.data.index_csv)
    if not index_path.is_file():
        raise FileNotFoundError(f"formal shared index不存在: {index_path}")
    raw_index = pd.read_csv(index_path)
    if "split" not in raw_index:
        raise ValueError("formal shared index缺少split列")
    allowed = raw_index.loc[raw_index["split"].astype(str).isin({"train", "val"})].copy()
    del raw_index
    if set(allowed["split"].astype(str).unique()) != {"train", "val"}:
        raise RuntimeError("formal shared index未同时提供train/validation")
    audited = adapt_research_v2_index(allowed, cfg)
    audit_summary = summarize_research_v2_audit(audited)
    train = build_window_data(
        cfg,
        split="train",
        max_windows=None,
        sample_strategy=str(cfg.data.train_sample_strategy),
        sample_seed=int(cfg.data.train_sample_seed),
        shuffle=True,
        audited=audited,
    )
    val = build_window_data(
        cfg,
        split="val",
        max_windows=None,
        sample_strategy=str(cfg.data.val_sample_strategy),
        sample_seed=int(cfg.data.val_sample_seed),
        shuffle=False,
        audited=audited,
    )
    return ThoDataBundle(train=train, val=val, audited=audited, audit_summary=audit_summary)


def _validate_data_contract(data: ThoDataBundle, plan: FormalPlan) -> dict[str, Any]:
    train_count = len(data.train.dataset)
    val_count = len(data.val.dataset)
    train_batches = len(data.train.loader)
    val_batches = len(data.val.loader)
    contract = plan.raw["data"]
    expected_train = int(contract["expected_train_windows"])
    expected_val = int(contract["expected_validation_windows"])
    expected_train_batches = int(contract["expected_train_batches"])
    expected_val_batches = int(contract["expected_validation_batches"])
    if (
        train_count != expected_train
        or val_count != expected_val
        or train_batches != expected_train_batches
        or val_batches != expected_val_batches
    ):
        raise RuntimeError(
            "formal train/validation count或batch漂移: "
            f"train={train_count}/{train_batches}, val={val_count}/{val_batches}"
        )
    if not data.audited["split"].astype(str).isin({"train", "val"}).all():
        raise RuntimeError("formal audited metadata混入train/validation之外split")
    train_ids = set(int(value) for value in data.train.rows["dataset_row_id"].tolist())
    val_ids = set(int(value) for value in data.val.rows["dataset_row_id"].tolist())
    if train_ids & val_ids:
        raise RuntimeError("formal train/validation dataset_row_id重叠")
    train_samp_ids = int(data.train.rows["samp_id"].nunique())
    val_samp_ids = int(data.val.rows["samp_id"].nunique())
    train_row_hash = _row_id_sha256(data.train.rows)
    val_row_hash = _row_id_sha256(data.val.rows)
    if (
        train_samp_ids != int(contract["expected_train_samp_ids"])
        or val_samp_ids != int(contract["expected_validation_samp_ids"])
        or train_row_hash != contract["expected_train_row_ids_sha256"]
        or val_row_hash != contract["expected_validation_row_ids_sha256"]
    ):
        raise RuntimeError("formal train/validation samp或row identity漂移")
    return {
        "train_windows": train_count,
        "validation_windows": val_count,
        "train_batches_per_epoch": train_batches,
        "validation_batches_per_epoch": val_batches,
        "train_samp_ids": train_samp_ids,
        "validation_samp_ids": val_samp_ids,
        "train_row_ids_sha256": train_row_hash,
        "validation_row_ids_sha256": val_row_hash,
        "row_id_overlap_count": 0,
        "shared_index_metadata_read": True,
        "signal_splits_accessed": ["train", "val"],
        "research_test_accessed": False,
    }


def _artifact_record(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    return {
        "filename": path.name,
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
        "rows": rows,
    }


def _dataframe_finite_audit(frame: pd.DataFrame) -> dict[str, int]:
    numeric = frame.select_dtypes(include=[np.number, "bool"])
    values = numeric.to_numpy(dtype=np.float64, copy=True)
    null = np.isnan(values)
    nonfinite = ~np.isfinite(values) & ~null
    return {
        "numeric_total": int(values.size),
        "numeric_finite": int(np.isfinite(values).sum()),
        "numeric_null": int(null.sum()),
        "numeric_nonfinite": int(nonfinite.sum()),
    }


def _checkpoint_finite_audit(path: Path) -> dict[str, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    counts = {"numeric_total": 0, "numeric_finite": 0, "numeric_null": 0, "numeric_nonfinite": 0}

    def visit(value: Any) -> None:
        if torch.is_tensor(value):
            tensor = value.detach()
            counts["numeric_total"] += int(tensor.numel())
            if tensor.is_floating_point() or tensor.is_complex():
                null_count = int(torch.isnan(tensor).sum().item())
                nonfinite_count = int((~torch.isfinite(tensor) & ~torch.isnan(tensor)).sum().item())
                counts["numeric_null"] += null_count
                counts["numeric_nonfinite"] += nonfinite_count
                counts["numeric_finite"] += int(tensor.numel()) - null_count - nonfinite_count
            else:
                counts["numeric_finite"] += int(tensor.numel())
        elif isinstance(value, Mapping):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                visit(nested)

    visit(payload)
    if counts["numeric_total"] <= 0:
        raise RuntimeError(f"formal checkpoint没有tensor: {path}")
    return counts


def _combined_finite_audit(records: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    aggregate = {
        key: sum(int(record[key]) for record in records.values())
        for key in ("numeric_total", "numeric_finite", "numeric_null", "numeric_nonfinite")
    }
    if aggregate["numeric_nonfinite"] != 0:
        raise FloatingPointError("formal artifacts包含Inf/-Inf")
    if aggregate["numeric_total"] != aggregate["numeric_finite"] + aggregate["numeric_null"]:
        raise RuntimeError("formal finite/null计数不闭合")
    return {"components": {name: dict(record) for name, record in records.items()}, "aggregate": aggregate}


def _runtime_summary(device: torch.device, elapsed_seconds: float) -> dict[str, Any]:
    torch.cuda.synchronize(device)
    properties = torch.cuda.get_device_properties(device)
    return {
        "elapsed_seconds": float(elapsed_seconds),
        "device": str(device),
        "device_name": properties.name,
        "total_memory_mib": float(properties.total_memory / (1024**2)),
        "peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / (1024**2)),
        "peak_reserved_mib": float(torch.cuda.max_memory_reserved(device) / (1024**2)),
    }


def _dependency_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in ("numpy", "pandas", "scipy", "torch", "omegaconf", "mamba-ssm"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    result["torch_cuda"] = torch.version.cuda
    result["cudnn"] = str(torch.backends.cudnn.version())
    return result


def validate_formal_receipt(receipt: Mapping[str, Any]) -> None:
    _require_exact_keys(
        receipt,
        {
            "schema_version",
            "protocol_id",
            "created_utc",
            "status",
            "evidence_label",
            "candidate_id",
            "seed",
            "execution",
            "data",
            "training",
            "selector",
            "access",
            "counts",
            "finite_audit",
            "runtime",
            "artifacts",
            "manifest_sha256",
        },
        "formal receipt",
    )
    if (
        receipt["schema_version"] != FORMAL_RECEIPT_SCHEMA_VERSION
        or receipt["protocol_id"] != FORMAL_PROTOCOL_ID
        or receipt["status"] != "complete"
        or receipt["evidence_label"] != "validation-development evidence"
    ):
        raise ValueError("formal receipt identity/status/evidence label无效")
    if receipt["candidate_id"] not in EXPECTED_CANDIDATES or int(receipt["seed"]) not in FORMAL_SEEDS:
        raise ValueError("formal receipt candidate/seed越界")
    execution = receipt["execution"]
    _require_exact_keys(
        execution,
        {
            "command",
            "cwd",
            "git_commit",
            "git_dirty",
            "plan_path",
            "plan_sha256",
            "dependencies",
            "device",
        },
        "formal receipt.execution",
    )
    if (
        not str(execution["command"]).strip()
        or execution["cwd"] != str(REPO_ROOT)
        or not _is_lower_hex(execution["git_commit"], length=40)
        or bool(execution["git_dirty"])
        or execution["plan_path"] != "configs/resp_temporal_v1/formal_v1.yaml"
        or execution["plan_sha256"] != FROZEN_PLAN_SHA256
    ):
        raise ValueError("formal receipt Git/plan provenance无效")
    data = receipt["data"]
    _require_exact_keys(
        data,
        {
            "train_windows",
            "validation_windows",
            "train_batches_per_epoch",
            "validation_batches_per_epoch",
            "train_samp_ids",
            "validation_samp_ids",
            "train_row_ids_sha256",
            "validation_row_ids_sha256",
            "row_id_overlap_count",
            "shared_index_metadata_read",
            "signal_splits_accessed",
            "research_test_accessed",
        },
        "formal receipt.data",
    )
    if (
        int(data.get("train_windows", -1)) != 10141
        or int(data.get("validation_windows", -1)) != 2675
        or int(data.get("train_batches_per_epoch", -1)) != 80
        or int(data.get("validation_batches_per_epoch", -1)) != 21
        or int(data.get("train_samp_ids", -1)) != 32
        or int(data.get("validation_samp_ids", -1)) != 7
        or data.get("train_row_ids_sha256") != "f290e569140a2ff7745cf1a5cfa6a4da943644d76498c9b85517d3ae0702c45e"
        or data.get("validation_row_ids_sha256") != "b68a51b101bb80033c4de18c9c21cdfbb8924d1bfe134330f047617cf3b0915a"
        or int(data.get("row_id_overlap_count", -1)) != 0
        or not bool(data.get("shared_index_metadata_read"))
        or data.get("signal_splits_accessed") != ["train", "val"]
        or bool(data.get("research_test_accessed"))
    ):
        raise ValueError("formal receipt data count/split/isolation无效")
    training = receipt["training"]
    if training != {
        "planned_epochs": 80,
        "completed_epochs": 80,
        "planned_optimizer_updates": 6400,
        "completed_optimizer_updates": 6400,
        "physical_batch_size": 128,
        "accumulation_steps": 1,
        "effective_batch_size": 128,
        "early_stopping": False,
        "resume": False,
    }:
        raise ValueError("formal receipt training completion合同无效")
    selector = receipt["selector"]
    _require_exact_keys(
        selector,
        {
            "metric",
            "comparison",
            "best_epoch",
            "best_validation_local_rr_mae",
            "reloaded_best_validation_local_rr_mae",
            "validation_eligible_samples",
        },
        "formal receipt.selector",
    )
    if (
        selector.get("metric") != "local_rr_mae_full_split_sample_direct_mean"
        or selector.get("comparison") != "strict_less_than"
        or not 1 <= int(selector.get("best_epoch", 0)) <= 80
        or not math.isfinite(float(selector.get("best_validation_local_rr_mae", float("nan"))))
        or not math.isfinite(float(selector.get("reloaded_best_validation_local_rr_mae", float("nan"))))
        or not math.isclose(
            float(selector["best_validation_local_rr_mae"]),
            float(selector["reloaded_best_validation_local_rr_mae"]),
            rel_tol=1e-6,
            abs_tol=1e-8,
        )
        or not 1 <= int(selector.get("validation_eligible_samples", 0)) <= 2675
    ):
        raise ValueError("formal receipt selector无效")
    access = receipt["access"]
    if access != {
        "shared_index_metadata_read": True,
        "train_signal_target_accessed": True,
        "validation_signal_target_accessed": True,
        "validation_prediction_generated": True,
        "research_test_accessed": False,
        "checkpoint_input_accessed": False,
        "own_best_checkpoint_reloaded": True,
        "gpu_used": True,
    }:
        raise ValueError("formal receipt access越界")
    if receipt["counts"] != {
        "expected_epochs": 80,
        "completed_epochs": 80,
        "expected_optimizer_updates": 6400,
        "completed_optimizer_updates": 6400,
        "validation_metrics_rows": 2675,
        "validation_summary_rows": 1,
        "artifact_records": 13,
    }:
        raise ValueError("formal receipt count closure无效")
    finite_audit = receipt["finite_audit"]
    _require_exact_keys(finite_audit, {"components", "aggregate"}, "formal receipt.finite_audit")
    if set(finite_audit["components"]) != {
        "train_history",
        "validation_metrics",
        "validation_metrics_summary",
        "checkpoint_best_local_rr",
        "checkpoint_final",
    }:
        raise ValueError("formal receipt finite components漂移")
    for label, record in {**finite_audit["components"], "aggregate": finite_audit["aggregate"]}.items():
        _require_exact_keys(
            record,
            {"numeric_total", "numeric_finite", "numeric_null", "numeric_nonfinite"},
            f"formal finite {label}",
        )
        if (
            any(int(record[key]) < 0 for key in record)
            or int(record["numeric_total"]) <= 0
            or int(record["numeric_nonfinite"]) != 0
            or int(record["numeric_total"])
            != int(record["numeric_finite"]) + int(record["numeric_null"])
        ):
            raise ValueError(f"formal finite count不闭合: {label}")
    if finite_audit["components"]["train_history"]["numeric_null"] != 0:
        raise ValueError("formal train history存在null")
    for label in ("checkpoint_best_local_rr", "checkpoint_final"):
        if finite_audit["components"][label]["numeric_null"] != 0:
            raise ValueError(f"formal checkpoint存在NaN: {label}")
    runtime = receipt["runtime"]
    _require_exact_keys(
        runtime,
        {
            "elapsed_seconds",
            "device",
            "device_name",
            "total_memory_mib",
            "peak_allocated_mib",
            "peak_reserved_mib",
        },
        "formal receipt.runtime",
    )
    if runtime["device"] != "cuda:0" or any(
        not math.isfinite(float(runtime[key])) or float(runtime[key]) < 0.0
        for key in ("elapsed_seconds", "total_memory_mib", "peak_allocated_mib", "peak_reserved_mib")
    ):
        raise ValueError("formal runtime无效")
    if not isinstance(receipt["artifacts"], list) or not receipt["artifacts"]:
        raise ValueError("formal receipt artifacts为空")
    expected_artifacts = [
        "config.yaml",
        "formal_plan.json",
        "run_manifest.json",
        "data_receipt.json",
        "audit.csv",
        "optimizer_parameter_groups.json",
        "train_history.csv",
        "checkpoint_best_local_rr.pt",
        "checkpoint_final.pt",
        "validation_metrics.csv",
        "validation_metrics_summary.csv",
        "runtime_summary.json",
        "artifact_manifest.json",
    ]
    if [artifact.get("filename") for artifact in receipt["artifacts"]] != expected_artifacts:
        raise ValueError("formal artifact identity/order漂移")
    for artifact in receipt["artifacts"]:
        _require_exact_keys(artifact, {"filename", "sha256", "size_bytes", "rows"}, "formal artifact")
        if not _is_lower_hex(artifact["sha256"], length=64) or int(artifact["size_bytes"]) <= 0:
            raise ValueError("formal artifact hash无效")
    if not _is_lower_hex(receipt["manifest_sha256"], length=64):
        raise ValueError("formal manifest hash无效")


def _write_complete_receipt(
    *,
    spec: FormalRunSpec,
    data_receipt: Mapping[str, Any],
    history: pd.DataFrame,
    best_epoch: int,
    best_local_rr: float,
    reloaded_best_local_rr: float,
    validation_eligible_samples: int,
    validation_metrics_rows: int,
    validation_summary_rows: int,
    audit_rows: int,
    finite_audit: Mapping[str, Any],
    runtime: Mapping[str, Any],
    command: str,
) -> Path:
    artifact_specs = [
        ("config.yaml", None),
        ("formal_plan.json", None),
        ("run_manifest.json", None),
        ("data_receipt.json", None),
        ("audit.csv", audit_rows),
        ("optimizer_parameter_groups.json", None),
        ("train_history.csv", len(history)),
        ("checkpoint_best_local_rr.pt", None),
        ("checkpoint_final.pt", None),
        ("validation_metrics.csv", validation_metrics_rows),
        ("validation_metrics_summary.csv", validation_summary_rows),
        ("runtime_summary.json", None),
    ]
    artifacts = [_artifact_record(spec.run_dir / name, rows=rows) for name, rows in artifact_specs]
    manifest = {
        "schema_version": "rtm-v1-formal-artifact-manifest-v1",
        "protocol_id": FORMAL_PROTOCOL_ID,
        "candidate_id": str(spec.cfg.model.variant),
        "seed": spec.seed,
        "artifacts": artifacts,
    }
    _json_write(spec.run_dir / "artifact_manifest.json", manifest)
    manifest_record = _artifact_record(spec.run_dir / "artifact_manifest.json")
    artifacts.append(manifest_record)
    receipt = {
        "schema_version": FORMAL_RECEIPT_SCHEMA_VERSION,
        "protocol_id": FORMAL_PROTOCOL_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "evidence_label": "validation-development evidence",
        "candidate_id": str(spec.cfg.model.variant),
        "seed": spec.seed,
        "execution": {
            "command": command,
            "cwd": str(REPO_ROOT),
            "git_commit": spec.git_commit,
            "git_dirty": False,
            "plan_path": str(spec.plan.path.relative_to(REPO_ROOT)),
            "plan_sha256": spec.plan.sha256,
            "dependencies": _dependency_versions(),
            "device": spec.gpu_device_info,
        },
        "data": dict(data_receipt),
        "training": {
            "planned_epochs": 80,
            "completed_epochs": int(history.iloc[-1]["epoch"]),
            "planned_optimizer_updates": 6400,
            "completed_optimizer_updates": int(history.iloc[-1]["optimizer_update"]),
            "physical_batch_size": 128,
            "accumulation_steps": 1,
            "effective_batch_size": 128,
            "early_stopping": False,
            "resume": False,
        },
        "selector": {
            "metric": "local_rr_mae_full_split_sample_direct_mean",
            "comparison": "strict_less_than",
            "best_epoch": int(best_epoch),
            "best_validation_local_rr_mae": float(best_local_rr),
            "reloaded_best_validation_local_rr_mae": float(reloaded_best_local_rr),
            "validation_eligible_samples": int(validation_eligible_samples),
        },
        "access": {
            "shared_index_metadata_read": True,
            "train_signal_target_accessed": True,
            "validation_signal_target_accessed": True,
            "validation_prediction_generated": True,
            "research_test_accessed": False,
            "checkpoint_input_accessed": False,
            "own_best_checkpoint_reloaded": True,
            "gpu_used": True,
        },
        "counts": {
            "expected_epochs": 80,
            "completed_epochs": int(history.iloc[-1]["epoch"]),
            "expected_optimizer_updates": 6400,
            "completed_optimizer_updates": int(history.iloc[-1]["optimizer_update"]),
            "validation_metrics_rows": int(validation_metrics_rows),
            "validation_summary_rows": int(validation_summary_rows),
            "artifact_records": len(artifacts),
        },
        "finite_audit": dict(finite_audit),
        "runtime": dict(runtime),
        "artifacts": artifacts,
        "manifest_sha256": manifest_record["sha256"],
    }
    validate_formal_receipt(receipt)
    _json_write_atomic(spec.run_dir / "formal_receipt.json", receipt)
    receipt_hash = sha256_file(spec.run_dir / "formal_receipt.json")
    (spec.run_dir / "formal_receipt.sha256").write_text(
        f"{receipt_hash}  formal_receipt.json\n", encoding="utf-8"
    )
    return spec.run_dir / "formal_receipt.json"


def run_formal_training(
    *, plan_path: str | Path, candidate_id: str, seed: int, command: str
) -> Path:
    """执行一个冻结candidate/seed formal run；仅供用户手动调用。"""

    spec = preflight_formal_run(plan_path=plan_path, candidate_id=candidate_id, seed=seed)
    spec.run_dir.parent.mkdir(parents=True, exist_ok=True)
    spec.run_dir.mkdir(parents=False, exist_ok=False)
    lifecycle_path = spec.run_dir / "lifecycle.json"
    lifecycle = {
        "schema_version": "rtm-v1-formal-lifecycle-v1",
        "protocol_id": FORMAL_PROTOCOL_ID,
        "candidate_id": candidate_id,
        "seed": int(seed),
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "completed_utc": None,
        "last_completed_epoch": 0,
        "last_optimizer_update": 0,
        "failure": None,
    }
    _json_write_atomic(lifecycle_path, lifecycle)
    started = time.perf_counter()
    try:
        save_config(spec.cfg, spec.run_dir)
        _json_write(spec.run_dir / "formal_plan.json", spec.plan.raw)
        _json_write(
            spec.run_dir / "run_manifest.json",
            {
                "schema_version": "rtm-v1-formal-run-manifest-v1",
                "protocol_id": FORMAL_PROTOCOL_ID,
                "candidate_id": candidate_id,
                "seed": int(seed),
                "command": command,
                "git_commit": spec.git_commit,
                "git_dirty": False,
                "candidate_config_sha256": spec.candidate_record["config_sha256"],
                "plan_sha256": spec.plan.sha256,
                "gpu_engineering_lock_sha256": spec.plan.raw["provenance"]["gpu_engineering_lock_sha256"],
            },
        )
        set_seed(int(seed))
        device = torch.device(str(spec.cfg.training.device))
        data = _build_formal_data(spec.cfg)
        data_receipt = _validate_data_contract(data, spec.plan)
        _json_write(spec.run_dir / "data_receipt.json", data_receipt)
        data.audit_summary.to_csv(spec.run_dir / "audit.csv", index=False)

        torch.cuda.reset_peak_memory_stats(device)
        model = build_resp_temporal_model(spec.cfg).to(device)
        observed_parameters = trainable_parameter_count(model)
        if observed_parameters != int(spec.cfg.model.expected_trainable_parameters):
            raise RuntimeError("formal model parameter count漂移")
        loss_fn = RespirationTaskLoss(spec.cfg).to(device)
        optimizer, partition = build_crd_optimizer(model, spec.cfg)
        _json_write(
            spec.run_dir / "optimizer_parameter_groups.json",
            {
                "weight_decay": float(spec.cfg.training.weight_decay),
                "decay": list(partition.decay_names),
                "no_decay": list(partition.no_decay_names),
            },
        )
        updates_per_epoch = optimizer_updates_per_epoch(
            len(data.train.loader), int(spec.cfg.training.gradient_accumulation_steps)
        )
        total_updates = updates_per_epoch * int(spec.cfg.training.epochs)
        if updates_per_epoch != 80 or total_updates != 6400:
            raise RuntimeError("formal optimizer update合同漂移")

        history: list[dict[str, float | int]] = []
        best_local_rr = float("inf")
        best_epoch: int | None = None
        update_index = 0
        for epoch in range(1, 81):
            torch.cuda.synchronize(device)
            epoch_started = time.perf_counter()
            train_summary, update_index = train_crd_one_epoch(
                model,
                data.train.loader,
                loss_fn,
                optimizer,
                device=device,
                accumulation_steps=1,
                update_index=update_index,
                total_updates=6400,
                max_learning_rate=float(spec.cfg.training.max_learning_rate),
                min_learning_rate=float(spec.cfg.training.min_learning_rate),
                warmup_fraction=float(spec.cfg.training.warmup_fraction),
                grad_clip_norm=float(spec.cfg.training.grad_clip_norm),
                use_amp=True,
                show_progress=None,
                epoch=epoch,
                total_epochs=80,
            )
            torch.cuda.synchronize(device)
            train_elapsed = time.perf_counter() - epoch_started
            val_summary, val_predictions = validate(
                model,
                data.val.loader,
                loss_fn,
                device=device,
                show_progress=None,
                epoch=epoch,
                total_epochs=80,
                return_predictions=True,
                use_amp=True,
            )
            val_core_loss = float(spec.cfg.loss.sync_weight) * float(val_summary["loss_sync"]) + float(
                spec.cfg.loss.effort_weight
            ) * float(val_summary["loss_effort"])
            val_local_rr = float(validation_local_rr_mean(val_predictions, spec.cfg))
            finite_epoch_values = (
                val_core_loss,
                val_local_rr,
                train_elapsed,
                float(train_summary["loss"]),
                float(train_summary["loss_sync"]),
                float(train_summary["loss_effort"]),
                float(train_summary["first_learning_rate"]),
                float(train_summary["last_learning_rate"]),
            )
            if not all(math.isfinite(value) for value in finite_epoch_values):
                raise FloatingPointError("formal epoch summary包含NaN/Inf")
            improved = strict_selector_improved(val_local_rr, best_local_rr)
            record = {
                "epoch": epoch,
                "optimizer_update": update_index,
                "train_loss_total": float(train_summary["loss"]),
                "train_loss_sync": float(train_summary["loss_sync"]),
                "train_loss_effort": float(train_summary["loss_effort"]),
                "first_learning_rate": float(train_summary["first_learning_rate"]),
                "last_learning_rate": float(train_summary["last_learning_rate"]),
                "val_core_loss": val_core_loss,
                "val_local_rr_mae": val_local_rr,
                "selector_improved": int(improved),
                "train_elapsed_seconds": float(train_elapsed),
                "train_samples_per_second": float(len(data.train.dataset) / train_elapsed),
            }
            history.append(record)
            _csv_write_atomic(pd.DataFrame(history), spec.run_dir / "train_history.csv")
            if improved:
                best_local_rr = val_local_rr
                best_epoch = epoch
                save_checkpoint(
                    spec.run_dir / "checkpoint_best_local_rr.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=record,
                    cfg=spec.cfg,
                    extra_state={
                        "protocol": FORMAL_PROTOCOL_ID,
                        "selector": "validation_local_rr_mae_full_split_strict_less_than",
                        "update_index": update_index,
                        "total_updates": 6400,
                        "resume_supported": False,
                    },
                )
            lifecycle.update(
                {
                    "last_completed_epoch": epoch,
                    "last_optimizer_update": update_index,
                    "best_epoch": best_epoch,
                    "best_validation_local_rr_mae": best_local_rr,
                }
            )
            _json_write_atomic(lifecycle_path, lifecycle)
            print(
                f"RTM-v1 formal {candidate_id} seed={seed} epoch={epoch}/80 "
                f"update={update_index}/6400 val_local_rr={val_local_rr:.6f}",
                flush=True,
            )
        if update_index != 6400 or best_epoch is None:
            raise RuntimeError("formal训练未完成6400 updates或未产生best checkpoint")
        del val_predictions, val_summary
        save_checkpoint(
            spec.run_dir / "checkpoint_final.pt",
            model=model,
            optimizer=optimizer,
            epoch=80,
            metrics=history[-1],
            cfg=spec.cfg,
            extra_state={
                "protocol": FORMAL_PROTOCOL_ID,
                "update_index": 6400,
                "total_updates": 6400,
                "resume_supported": False,
            },
        )
        checkpoint = torch.load(
            spec.run_dir / "checkpoint_best_local_rr.pt",
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        del checkpoint
        predictions = collect_predictions(
            model,
            data.val.loader,
            device=device,
            max_windows=len(data.val.dataset),
            use_amp=True,
        )
        metrics = evaluate_task_predictions(
            predictions,
            spec.cfg,
            include_test_only=False,
            method=candidate_id,
        )
        metrics.insert(0, "evaluation_split", "validation")
        _csv_write_atomic(metrics, spec.run_dir / "validation_metrics.csv")
        metrics_summary = summarize_task_metrics(metrics)
        _csv_write_atomic(metrics_summary, spec.run_dir / "validation_metrics_summary.csv")
        reloaded_best_local_rr = float(pd.to_numeric(metrics["local_rr_mae_bpm"], errors="coerce").mean())
        validation_eligible_samples = int(
            pd.to_numeric(metrics["local_rr_mae_bpm"], errors="coerce").notna().sum()
        )
        if (
            not math.isfinite(reloaded_best_local_rr)
            or validation_eligible_samples <= 0
            or not math.isclose(reloaded_best_local_rr, best_local_rr, rel_tol=1e-6, abs_tol=1e-8)
        ):
            raise RuntimeError("formal best checkpoint reload未复现selector")
        finite_audit = _combined_finite_audit(
            {
                "train_history": _dataframe_finite_audit(pd.DataFrame(history)),
                "validation_metrics": _dataframe_finite_audit(metrics),
                "validation_metrics_summary": _dataframe_finite_audit(metrics_summary),
                "checkpoint_best_local_rr": _checkpoint_finite_audit(
                    spec.run_dir / "checkpoint_best_local_rr.pt"
                ),
                "checkpoint_final": _checkpoint_finite_audit(spec.run_dir / "checkpoint_final.pt"),
            }
        )
        runtime = _runtime_summary(device, time.perf_counter() - started)
        _json_write(spec.run_dir / "runtime_summary.json", runtime)
        receipt_path = _write_complete_receipt(
            spec=spec,
            data_receipt=data_receipt,
            history=pd.DataFrame(history),
            best_epoch=best_epoch,
            best_local_rr=best_local_rr,
            reloaded_best_local_rr=reloaded_best_local_rr,
            validation_eligible_samples=validation_eligible_samples,
            validation_metrics_rows=len(metrics),
            validation_summary_rows=len(metrics_summary),
            audit_rows=len(data.audit_summary),
            finite_audit=finite_audit,
            runtime=runtime,
            command=command,
        )
        lifecycle.update(
            {
                "status": "complete",
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "formal_receipt_sha256": sha256_file(receipt_path),
            }
        )
        _json_write_atomic(lifecycle_path, lifecycle)
        return receipt_path
    except BaseException as exc:
        lifecycle.update(
            {
                "status": "failed",
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "failure": {"error_type": type(exc).__name__, "message": str(exc)[:2000]},
            }
        )
        _json_write_atomic(lifecycle_path, lifecycle)
        raise
