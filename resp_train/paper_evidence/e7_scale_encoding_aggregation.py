"""E7 六臂训练配置、early-stop 审计、析因汇总与表征诊断。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from resp_train.crd.config import load_crd_config
from resp_train.crd.experiment import (
    CRDExperiment,
    _early_stopping_should_stop,
    _early_stopping_step,
)
from resp_train.crd.training import crd_learning_rate
from resp_train.paper_evidence.e1_scale_topology import ERRORS, PCC, PRIMARY, SEEDS
from resp_train.paper_evidence.e1_scale_topology_runtime import (
    git_state,
    identity,
    sha256_file,
    write_json,
)
from resp_train.paper_evidence.e4_scale_aggregation_model import validate_frequency_grid
from resp_train.paper_evidence.e7_scale_encoding_aggregation_model import (
    AGGREGATIONS,
    ARMS,
    DILATIONS,
    ENCODERS,
    PROTOCOL,
    THEORETICAL_RECEPTIVE_FIELDS,
    arm_contract,
    build_e7_model,
)


ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("configs/e7_scale_encoding_aggregation/experiment.yaml")
PROTOCOL_PATH = Path(
    "docs/experiments/e7_scale_encoding_aggregation_factorial_protocol_20260924.md"
)
MODEL_PATH = Path("resp_train/paper_evidence/e7_scale_encoding_aggregation_model.py")
CONTROL_PATH = Path("resp_train/paper_evidence/e7_scale_encoding_aggregation.py")
SCRIPT_PATH = Path("scripts/run_e7_scale_encoding_aggregation.py")
TEST_PATH = Path("tests/test_e7_scale_encoding_aggregation.py")
SOURCE_LOCK = Path("docs/experiments/e4_w0_scale_aggregation_implementation_lock_20260917.json")
SOURCE_LOCK_SHA256 = "464e073dbd5707a30575d606dec2a84dcd89a161945e537c463b214a15c2b493"
FREQUENCY_AUDIT = Path("docs/experiments/e4_w0_scale_aggregation_source_audit_20260917.json")
FREQUENCY_AUDIT_SHA256 = "198cbb396b0503d7a3e889be5e90ae401371ee537c7ab559b14f842f6ad38090"
E4_CLOSEOUT = Path("docs/experiments/e4_closeout_20260922.md")
E4_CLOSEOUT_SHA256 = "5f4b33ebbfae131a7fd504a3783b297011fdbe1b99da92130149681793366d16"
SOURCE_AUDIT = Path("docs/experiments/e7_scale_encoding_aggregation_source_audit_20260924.json")
LOCK_PATH = Path("docs/experiments/e7_scale_encoding_aggregation_p1_implementation_lock_20260924.json")
MAINLINE_COMPATIBILITY_PATH = Path(
    "docs/experiments/e7_scale_encoding_aggregation_mainline_compatibility_20260926.json"
)
OUTPUT = Path("runs/e7_scale_encoding_aggregation")
COUNTS = {"train": 10_141, "val": 2_675}
SAMP_IDS = {"train": 32, "val": 7}
EPOCHS = 80
UPDATES_PER_EPOCH = 80
EARLY_STOP_MIN_EPOCH = 30
EARLY_STOP_PATIENCE = 15
EARLY_STOP_MIN_DELTA = 0.0
ERROR_TOLERANCE_PERCENT = 0.5
PCC_TOLERANCE = 0.002


def verify(path: Path, expected: Mapping[str, Any]) -> None:
    required = {key: expected[key] for key in ("size_bytes", "sha256")}
    if not path.is_file() or identity(path) != required:
        raise RuntimeError(f"E7 文件身份漂移: {path}")


def critical_paths() -> tuple[Path, ...]:
    paths = (
        SPEC_PATH,
        PROTOCOL_PATH,
        MODEL_PATH,
        CONTROL_PATH,
        SCRIPT_PATH,
        TEST_PATH,
        Path("resp_train/crd/tf_v1_model.py"),
        Path("resp_train/crd/initialization.py"),
        Path("resp_train/crd/experiment.py"),
        Path("resp_train/crd/training.py"),
        Path("resp_train/crd/model.py"),
        Path("resp_train/crd/blocks.py"),
        Path("resp_train/losses/task.py"),
        Path("resp_train/metrics/task.py"),
        Path("resp_train/paper_evidence/e4_scale_aggregation_model.py"),
    )
    missing = [str(path) for path in paths if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"E7 P1 缺少关键文件: {missing}")
    return paths


def _load_parent_lock(root: Path = ROOT) -> dict[str, Any]:
    path = root / SOURCE_LOCK
    if sha256_file(path) != SOURCE_LOCK_SHA256:
        raise ValueError("E7 W0 来源锁漂移")
    parent = json.loads(path.read_text(encoding="utf-8"))
    if (
        parent.get("protocol") != "e4-w0-scale-aggregation-v1-20260917"
        or parent.get("status") != "implementation_locked_gpu_and_training_pending"
        or parent.get("seeds") != list(SEEDS)
        or parent.get("counts") != COUNTS
        or int(parent.get("epochs", -1)) != EPOCHS
        or int(parent.get("updates_per_epoch", -1)) != UPDATES_PER_EPOCH
        or len(parent.get("w0_entries", [])) != len(SEEDS)
    ):
        raise ValueError("E7 W0 来源锁合同漂移")
    values = np.asarray(parent["frequency"]["values_hz"], dtype=np.float64)
    if validate_frequency_grid(values) != parent["frequency"]:
        raise ValueError("E7 W0 来源频率合同漂移")
    if set(parent.get("baselines", {})) != {str(seed) for seed in SEEDS}:
        raise ValueError("E7 W0 三 seed baseline 不完整")
    return parent


def _verify_parent_files(parent: Mapping[str, Any], root: Path = ROOT) -> None:
    for relative, expected in parent["source_files"].items():
        verify(root / relative, expected)
    dataset = parent["dataset_index"]
    dataset_path = Path(dataset["path"])
    if not dataset_path.is_absolute():
        dataset_path = root / dataset_path
    if sha256_file(dataset_path) != dataset["sha256"]:
        raise ValueError("E7 dataset index identity 漂移")


def prepare_source_audit(root: Path = ROOT) -> Path:
    destination = root / SOURCE_AUDIT
    if destination.exists():
        raise FileExistsError(f"E7 source audit 已存在: {destination}")
    load_experiment_spec(root / SPEC_PATH)
    parent = _load_parent_lock(root)
    _verify_parent_files(parent, root)
    for path, digest in (
        (root / FREQUENCY_AUDIT, FREQUENCY_AUDIT_SHA256),
        (root / E4_CLOSEOUT, E4_CLOSEOUT_SHA256),
    ):
        if sha256_file(path) != digest:
            raise ValueError(f"E7 冻结背景来源漂移: {path}")
    audit = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "scope": "train_validation_source_identity_only",
        "question": "aggregation_pre_scale_encoding_depth_span_by_aggregation_factorial",
        "matrix": {
            "arms": list(ARMS),
            "seeds": list(SEEDS),
            "cells": len(ARMS) * len(SEEDS),
            "counts": COUNTS,
            "samp_ids": SAMP_IDS,
        },
        "parent_sources": {
            str(SOURCE_LOCK): identity(root / SOURCE_LOCK),
            str(FREQUENCY_AUDIT): identity(root / FREQUENCY_AUDIT),
            str(E4_CLOSEOUT): identity(root / E4_CLOSEOUT),
        },
        "dataset_index": parent["dataset_index"],
        "cache_lock": parent["cache_lock"],
        "frequency": parent["frequency"],
        "w0_entries": parent["w0_entries"],
        "fixed_controls": {
            "model_variant": "crd_tf102_w",
            "loss": "L_sync + 0.25 L_effort",
            "batch_size": 128,
            "amp_dtype": "bfloat16",
            "max_epochs": EPOCHS,
            "planned_updates": EPOCHS * UPDATES_PER_EPOCH,
            "early_stopping": {
                "min_epoch": EARLY_STOP_MIN_EPOCH,
                "patience": EARLY_STOP_PATIENCE,
                "min_delta": EARLY_STOP_MIN_DELTA,
            },
        },
        "access": {
            "decoded_arrays": False,
            "test_accessed": False,
            "verified_parent_file_count": len(parent["source_files"]),
        },
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "preparation_git": git_state(root),
    }
    write_json(destination, audit)
    return destination


def validate_source_audit(audit: Mapping[str, Any], root: Path = ROOT) -> None:
    if (
        audit.get("schema_version") != 1
        or audit.get("protocol") != PROTOCOL
        or audit.get("scope") != "train_validation_source_identity_only"
        or audit.get("matrix", {}).get("arms") != list(ARMS)
        or audit.get("matrix", {}).get("seeds") != list(SEEDS)
        or audit.get("matrix", {}).get("cells") != len(ARMS) * len(SEEDS)
        or audit.get("access", {}).get("decoded_arrays") is not False
        or audit.get("access", {}).get("test_accessed") is not False
    ):
        raise ValueError("E7 source audit 合同漂移")
    expected_sources = {
        str(SOURCE_LOCK): {"size_bytes": (root / SOURCE_LOCK).stat().st_size, "sha256": SOURCE_LOCK_SHA256},
        str(FREQUENCY_AUDIT): {"size_bytes": (root / FREQUENCY_AUDIT).stat().st_size, "sha256": FREQUENCY_AUDIT_SHA256},
        str(E4_CLOSEOUT): {"size_bytes": (root / E4_CLOSEOUT).stat().st_size, "sha256": E4_CLOSEOUT_SHA256},
    }
    if audit.get("parent_sources") != expected_sources:
        raise ValueError("E7 source audit 父来源漂移")
    validate_frequency_grid(np.asarray(audit["frequency"]["values_hz"], dtype=np.float64))


def prepare_implementation_lock(root: Path = ROOT) -> Path:
    destination = root / LOCK_PATH
    if destination.exists():
        raise FileExistsError(f"E7 P1 implementation lock 已存在: {destination}")
    state = git_state(root)
    if state.get("status_porcelain"):
        raise RuntimeError("E7 implementation lock 要求干净工作树")
    load_experiment_spec(root / SPEC_PATH)
    parent = _load_parent_lock(root)
    _verify_parent_files(parent, root)
    audit_path = root / SOURCE_AUDIT
    if not audit_path.is_file():
        raise FileNotFoundError("E7 implementation lock 缺少已提交 source audit")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    validate_source_audit(audit, root)
    frequencies = np.asarray(parent["frequency"]["values_hz"], dtype=np.float64)
    templates: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        templates[arm] = {}
        for seed in SEEDS:
            baseline = OmegaConf.create(parent["baselines"][str(seed)])
            output_root = root / OUTPUT / "formal" / arm / f"seed_{seed}"
            cfg = derived_config(
                baseline,
                arm,
                frequencies,
                output_root=output_root,
                device="cuda:0",
            )
            validate_config(
                cfg,
                baseline,
                arm,
                frequencies,
                output_root=output_root,
                device="cuda:0",
            )
            templates[arm][str(seed)] = OmegaConf.to_container(cfg, resolve=True)
    source_files = dict(parent["source_files"])
    source_files.update(
        {
            str(SOURCE_LOCK): identity(root / SOURCE_LOCK),
            str(FREQUENCY_AUDIT): identity(root / FREQUENCY_AUDIT),
            str(E4_CLOSEOUT): identity(root / E4_CLOSEOUT),
            str(SOURCE_AUDIT): identity(audit_path),
        }
    )
    lock = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "p1_implemented_not_run",
        "arms": list(ARMS),
        "encoders": list(ENCODERS),
        "aggregations": list(AGGREGATIONS),
        "seeds": list(SEEDS),
        "counts": COUNTS,
        "samp_ids": SAMP_IDS,
        "epochs": EPOCHS,
        "updates_per_epoch": UPDATES_PER_EPOCH,
        "early_stopping": expected_spec()["matrix"]["early_stopping"],
        "tolerances": expected_spec()["tolerances"],
        "contracts": {arm: arm_contract(arm) for arm in ARMS},
        "baselines": parent["baselines"],
        "resolved_templates": templates,
        "w0_entries": parent["w0_entries"],
        "frequency": parent["frequency"],
        "cache_lock": parent["cache_lock"],
        "dataset_index": parent["dataset_index"],
        "source_audit": identity(audit_path),
        "source_files": source_files,
        "code_files": {
            str(path): identity(root / path)
            for path in critical_paths()
        },
        "parameter_compute_report": parameter_compute_report(),
        "preparation_git": state,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(destination, lock)
    return destination


def load_implementation_lock(root: Path = ROOT) -> tuple[dict[str, Any], str]:
    path = root / LOCK_PATH
    lock = json.loads(path.read_text(encoding="utf-8"))
    if (
        lock.get("schema_version") != 1
        or lock.get("protocol") != PROTOCOL
        or lock.get("status") != "p1_implemented_not_run"
        or lock.get("arms") != list(ARMS)
        or lock.get("seeds") != list(SEEDS)
        or lock.get("counts") != COUNTS
        or lock.get("samp_ids") != SAMP_IDS
        or int(lock.get("epochs", -1)) != EPOCHS
        or int(lock.get("updates_per_epoch", -1)) != UPDATES_PER_EPOCH
        or lock.get("contracts") != {arm: arm_contract(arm) for arm in ARMS}
        or lock.get("early_stopping") != expected_spec()["matrix"]["early_stopping"]
        or lock.get("tolerances") != expected_spec()["tolerances"]
    ):
        raise ValueError("E7 P1 implementation lock 科学合同漂移")
    compatibility_path = root / MAINLINE_COMPATIBILITY_PATH
    amended_paths: set[str] = set()
    if compatibility_path.is_file():
        compatibility = json.loads(compatibility_path.read_text(encoding="utf-8"))
        amended = compatibility.get("amended_code_files", {})
        amended_paths = set(amended)
        if (
            compatibility.get("schema_version") != 1
            or compatibility.get("protocol") != PROTOCOL
            or compatibility.get("status") != "mainline_compatibility_locked"
            or compatibility.get("scope") != "closed_experiment_mainline_coexistence"
            or compatibility.get("scientific_contract_changed") is not False
            or compatibility.get("experiment_rerun_authorized") is not False
            or compatibility.get("base_implementation_lock", {}).get("path") != str(LOCK_PATH)
            or compatibility.get("base_implementation_lock", {}).get("sha256")
            != sha256_file(path)
            or amended_paths
            != {
                str(CONTROL_PATH),
                "resp_train/crd/experiment.py",
            }
        ):
            raise ValueError("E7 mainline compatibility 合同漂移")
        for relative, entry in amended.items():
            if entry.get("base") != lock["code_files"].get(relative):
                raise ValueError(f"E7 mainline compatibility 基线漂移: {relative}")
            verify(root / relative, entry["revised"])
    for relative, expected in lock["code_files"].items():
        if relative not in amended_paths:
            verify(root / relative, expected)
    audit_path = root / SOURCE_AUDIT
    verify(audit_path, lock["source_audit"])
    validate_source_audit(json.loads(audit_path.read_text(encoding="utf-8")), root)
    frequencies = np.asarray(lock["frequency"]["values_hz"], dtype=np.float64)
    validate_frequency_grid(frequencies)
    for arm in ARMS:
        for seed in SEEDS:
            output_root = root / OUTPUT / "formal" / arm / f"seed_{seed}"
            validate_config(
                OmegaConf.create(lock["resolved_templates"][arm][str(seed)]),
                OmegaConf.create(lock["baselines"][str(seed)]),
                arm,
                frequencies,
                output_root=output_root,
                device="cuda:0",
            )
    return lock, sha256_file(path)


def expected_spec() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "source": {
            "w0_implementation_lock": str(SOURCE_LOCK),
            "w0_implementation_lock_sha256": SOURCE_LOCK_SHA256,
            "frequency_audit": str(FREQUENCY_AUDIT),
        },
        "encoders": {
            encoder: {
                "dilations": list(DILATIONS[encoder]),
                "theoretical_scale_receptive_field": THEORETICAL_RECEPTIVE_FIELDS[encoder],
            }
            for encoder in ENCODERS
        },
        "aggregations": {
            "mean": {"hidden_channels": 0},
            "frequency_attention": {"hidden_channels": 8},
        },
        "matrix": {
            "arms": list(ARMS),
            "seeds": list(SEEDS),
            "train_windows": COUNTS["train"],
            "validation_windows": COUNTS["val"],
            "train_samp_ids": SAMP_IDS["train"],
            "validation_samp_ids": SAMP_IDS["val"],
            "max_epochs": EPOCHS,
            "planned_optimizer_updates": EPOCHS * UPDATES_PER_EPOCH,
            "early_stopping": {
                "enabled": True,
                "min_epoch": EARLY_STOP_MIN_EPOCH,
                "patience": EARLY_STOP_PATIENCE,
                "min_delta": EARLY_STOP_MIN_DELTA,
                "monitor": "validation_local_rr_mae_full_split",
            },
            "physical_batch": 128,
            "gradient_accumulation_steps": 1,
            "amp_dtype": "bfloat16",
            "checkpoint_selector": "full_validation_local_rr_strict_minimum_earliest_tie",
            "test_enabled": False,
        },
        "tolerances": {
            "error_relative_percent": ERROR_TOLERANCE_PERCENT,
            "pcc_absolute": PCC_TOLERANCE,
        },
    }


def load_experiment_spec(path: Path | None = None) -> DictConfig:
    path = ROOT / SPEC_PATH if path is None else Path(path)
    cfg = OmegaConf.load(path)
    OmegaConf.resolve(cfg)
    if OmegaConf.to_container(cfg, resolve=True) != expected_spec():
        raise ValueError("E7 spec 科学合同漂移")
    return cfg


def load_frequencies(root: Path = ROOT) -> np.ndarray:
    audit = json.loads((root / FREQUENCY_AUDIT).read_text(encoding="utf-8"))
    values = np.asarray(audit["frequency"]["values_hz"], dtype=np.float64)
    validate_frequency_grid(values)
    return values


def baseline_config(seed: int) -> DictConfig:
    if int(seed) not in SEEDS:
        raise ValueError("E7 seed 不属于固定矩阵")
    return load_crd_config(
        ROOT / "configs/crd_tf_v1/crd_tf102_w_formal.yaml",
        overrides=[f"training.seed={int(seed)}", f"model.initialization_seed={int(seed)}"],
    )


def derived_config(
    baseline: DictConfig,
    arm: str,
    frequencies: np.ndarray | list[float],
    *,
    output_root: Path,
    device: str,
) -> DictConfig:
    contract = arm_contract(arm)
    validate_frequency_grid(np.asarray(frequencies, dtype=np.float64))
    cfg = OmegaConf.create(OmegaConf.to_container(baseline, resolve=True))
    cfg.protocol.name = PROTOCOL
    cfg.protocol.execution_gate = "e7_formal"
    cfg.model.e7_factorial = contract
    cfg.model.aggregation_frequencies_hz = list(map(float, frequencies))
    cfg.training.early_stopping_enabled = True
    cfg.training.early_stopping_min_epoch = EARLY_STOP_MIN_EPOCH
    cfg.training.early_stopping_patience = EARLY_STOP_PATIENCE
    cfg.training.early_stopping_min_delta = EARLY_STOP_MIN_DELTA
    cfg.training.device = str(device)
    cfg.training.show_progress = False
    cfg.outputs.run_root = str(output_root)
    return cfg


def validate_config(
    cfg: DictConfig,
    baseline: DictConfig,
    arm: str,
    frequencies: np.ndarray | list[float],
    *,
    output_root: Path,
    device: str,
) -> None:
    expected = derived_config(
        baseline,
        arm,
        frequencies,
        output_root=output_root,
        device=device,
    )
    if OmegaConf.to_container(cfg, resolve=True) != OmegaConf.to_container(expected, resolve=True):
        raise ValueError("E7 配置必须与同 seed W0 合同一致，仅开放析因结构和运行身份")
    if (
        str(baseline.protocol.name) != "crd-tf-v1-research-informed-20260812"
        or str(baseline.model.variant) != "crd_tf102_w"
        or list(baseline.model.tf_representations) != ["w"]
        or int(baseline.training.seed) not in SEEDS
        or int(baseline.model.initialization_seed) != int(baseline.training.seed)
        or int(baseline.training.epochs) != EPOCHS
        or bool(baseline.training.early_stopping_enabled)
        or int(baseline.training.batch_size) != 128
        or int(baseline.training.gradient_accumulation_steps) != 1
        or not bool(baseline.training.use_amp)
        or str(baseline.training.amp_dtype) != "bfloat16"
        or baseline.data.max_train_windows is not None
        or baseline.data.max_val_windows is not None
        or float(baseline.loss.sync_weight) != 1.0
        or float(baseline.loss.effort_weight) != 0.25
    ):
        raise ValueError("E7 W0 来源配置不符合固定矩阵")
    if (
        not bool(cfg.training.early_stopping_enabled)
        or int(cfg.training.early_stopping_min_epoch) != EARLY_STOP_MIN_EPOCH
        or int(cfg.training.early_stopping_patience) != EARLY_STOP_PATIENCE
        or float(cfg.training.early_stopping_min_delta) != EARLY_STOP_MIN_DELTA
    ):
        raise ValueError("E7 early stopping 合同漂移")


class ScaleFactorialExperiment(CRDExperiment):
    task_name = "e7_scale_encoding_aggregation"

    def __init__(self, cfg: DictConfig, validation_rows: pd.DataFrame | None = None):
        super().__init__(cfg)
        self.validation_rows = validation_rows

    def _build_model(self):
        return build_e7_model(self.cfg)

    def _evaluate_model(self, model, loader, **kwargs):
        frame = super()._evaluate_model(model, loader, **kwargs)
        if self.validation_rows is not None:
            validate_metrics(frame, self.validation_rows)
        frame.insert(0, "arm", str(self.cfg.model.e7_factorial.arm))
        frame.insert(0, "seed", int(self.cfg.training.seed))
        return frame


def validate_metrics(metrics: pd.DataFrame, rows: pd.DataFrame) -> dict[str, int]:
    if (
        len(metrics) != len(rows)
        or metrics.dataset_row_id.duplicated().any()
        or rows.dataset_row_id.duplicated().any()
    ):
        raise ValueError("E7 validation metrics row identity 不完整")
    for key in ("dataset_row_id", "samp_id", "split"):
        if not np.array_equal(metrics[key].to_numpy(), rows[key].to_numpy()):
            raise ValueError(f"E7 validation identity/order 漂移: {key}")
    eligibility = {
        ERRORS[0]: "whole_rr_target_eligible",
        ERRORS[1]: "local_rr_target_eligible",
        PCC: "joint_target_eligible",
    }
    for metric in PRIMARY:
        expected = np.ones(len(metrics), dtype=bool)
        if metric in eligibility:
            flags = metrics[eligibility[metric]]
            if flags.isna().any() or not flags.isin([True, False]).all():
                raise ValueError(f"E7 eligibility 非法: {metric}")
            expected = flags.to_numpy(dtype=bool)
        values = metrics[metric].to_numpy(dtype=float)
        if np.isinf(values).any() or not np.array_equal(np.isfinite(values), expected):
            raise FloatingPointError(f"E7 metric finite/eligibility 不一致: {metric}")
    degeneracy: dict[str, int] = {}
    for key in ("joint_prediction_degenerate", "envelope_spearman_prediction_degenerate"):
        flags = metrics[key]
        if flags.isna().any() or not flags.isin([True, False]).all():
            raise ValueError(f"E7 prediction degeneracy flag 非法: {key}")
        degeneracy[key] = int(flags.astype(bool).sum())
    return degeneracy


def validate_history(history: pd.DataFrame, cfg: DictConfig) -> int:
    completed = len(history)
    if (
        completed < EARLY_STOP_MIN_EPOCH
        or completed > EPOCHS
        or not np.array_equal(history.epoch.to_numpy(), np.arange(1, completed + 1))
        or not np.array_equal(
            history.optimizer_update.to_numpy(),
            np.arange(1, completed + 1) * UPDATES_PER_EPOCH,
        )
    ):
        raise ValueError("E7 history epoch/update 轨迹不完整")
    numeric = history.select_dtypes(include=np.number).to_numpy()
    if not np.isfinite(numeric).all():
        raise FloatingPointError("E7 history 非有限")
    if not np.allclose(
        history.train_loss_total,
        history.train_loss_sync + 0.25 * history.train_loss_effort,
        atol=1e-12,
        rtol=0,
    ):
        raise ValueError("E7 total loss 与完整目标不一致")
    total_updates = EPOCHS * UPDATES_PER_EPOCH
    for row in history.itertuples():
        for column, update in (
            ("first_learning_rate", (int(row.epoch) - 1) * UPDATES_PER_EPOCH),
            ("last_learning_rate", int(row.epoch) * UPDATES_PER_EPOCH - 1),
        ):
            expected = crd_learning_rate(
                update,
                total_updates=total_updates,
                max_learning_rate=float(cfg.training.max_learning_rate),
                min_learning_rate=float(cfg.training.min_learning_rate),
                warmup_fraction=float(cfg.training.warmup_fraction),
            )
            if not np.isclose(float(getattr(row, column)), expected, atol=1e-15, rtol=0):
                raise ValueError("E7 learning-rate schedule 漂移")
    required = {
        "early_stopping_improved",
        "early_stopping_wait",
        "early_stopping_triggered",
        "early_stopping_min_epoch",
    }
    if not required.issubset(history.columns):
        raise ValueError("E7 history 缺少 early stopping 字段")
    best = float("inf")
    wait = 0
    triggers: list[int] = []
    for row in history.itertuples():
        improved, wait = _early_stopping_step(
            value=float(row.val_local_rr_mae),
            best=best,
            epochs_without_improvement=wait,
            min_delta=EARLY_STOP_MIN_DELTA,
        )
        if improved:
            best = float(row.val_local_rr_mae)
        triggered = _early_stopping_should_stop(
            epoch=int(row.epoch),
            min_epoch=EARLY_STOP_MIN_EPOCH,
            epochs_without_improvement=wait,
            patience=EARLY_STOP_PATIENCE,
        )
        if (
            int(row.early_stopping_improved) != int(improved)
            or int(row.early_stopping_wait) != wait
            or int(row.early_stopping_triggered) != int(triggered)
            or int(row.early_stopping_min_epoch) != EARLY_STOP_MIN_EPOCH
        ):
            raise ValueError("E7 history early stopping 轨迹漂移")
        triggers.append(int(triggered))
    if any(triggers[:-1]):
        raise ValueError("E7 history 在停止点后继续")
    if completed < EPOCHS and triggers[-1] != 1:
        raise ValueError("E7 history 提前结束但没有合法停止点")
    return int(history.iloc[int(np.argmin(history.val_local_rr_mae.to_numpy()))].epoch)


def _matrix_index(frame: pd.DataFrame, *, split: str) -> pd.DataFrame:
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    observed = set(frame[["arm", "seed"]].itertuples(index=False, name=None))
    if (
        frame[["arm", "seed"]].duplicated().any()
        or observed != expected
        or set(frame["split"]) != {split}
    ):
        raise ValueError("E7 汇总要求六臂×三 seed 的唯一完整矩阵")
    metric_columns = [metric + "_mean" for metric in PRIMARY]
    if not np.isfinite(frame[metric_columns].to_numpy(dtype=float)).all():
        raise FloatingPointError("E7 汇总主指标非有限")
    if (frame[[metric + "_mean" for metric in ERRORS]].to_numpy(dtype=float) < 0).any():
        raise ValueError("E7 error 指标不得为负")
    return frame.set_index(["arm", "seed"]).sort_index()


def _arm(encoder: str, aggregation: str) -> str:
    return f"{encoder}__{aggregation}"


SIMPLE_CONTRASTS: dict[str, dict[str, float]] = {
    "mean_local": {_arm("s1_deep_local", "mean"): 1, _arm("s0_shallow", "mean"): -1},
    "mean_span": {_arm("s2_axis_spanning", "mean"): 1, _arm("s1_deep_local", "mean"): -1},
    "attention_local": {
        _arm("s1_deep_local", "frequency_attention"): 1,
        _arm("s0_shallow", "frequency_attention"): -1,
    },
    "attention_span": {
        _arm("s2_axis_spanning", "frequency_attention"): 1,
        _arm("s1_deep_local", "frequency_attention"): -1,
    },
    **{
        f"{encoder}_aggregation": {
            _arm(encoder, "frequency_attention"): 1,
            _arm(encoder, "mean"): -1,
        }
        for encoder in ENCODERS
    },
}

FACTORIAL_CONTRASTS: dict[str, dict[str, float]] = {
    "c_local": {
        _arm("s1_deep_local", "mean"): 0.5,
        _arm("s0_shallow", "mean"): -0.5,
        _arm("s1_deep_local", "frequency_attention"): 0.5,
        _arm("s0_shallow", "frequency_attention"): -0.5,
    },
    "c_span": {
        _arm("s2_axis_spanning", "mean"): 0.5,
        _arm("s1_deep_local", "mean"): -0.5,
        _arm("s2_axis_spanning", "frequency_attention"): 0.5,
        _arm("s1_deep_local", "frequency_attention"): -0.5,
    },
    "i_local": {
        _arm("s1_deep_local", "frequency_attention"): 1,
        _arm("s1_deep_local", "mean"): -1,
        _arm("s0_shallow", "frequency_attention"): -1,
        _arm("s0_shallow", "mean"): 1,
    },
    "i_span": {
        _arm("s2_axis_spanning", "frequency_attention"): 1,
        _arm("s2_axis_spanning", "mean"): -1,
        _arm("s1_deep_local", "frequency_attention"): -1,
        _arm("s1_deep_local", "mean"): 1,
    },
    "i_endpoint": {
        _arm("s2_axis_spanning", "frequency_attention"): 1,
        _arm("s2_axis_spanning", "mean"): -1,
        _arm("s0_shallow", "frequency_attention"): -1,
        _arm("s0_shallow", "mean"): 1,
    },
}


def _utility(metric: str, value: float) -> float:
    return float(value if metric == PCC else -value)


def contrast_tables(
    frame: pd.DataFrame,
    *,
    split: str = "val",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    indexed = _matrix_index(frame, split=split)

    def calculate(definitions: Mapping[str, Mapping[str, float]]) -> tuple[pd.DataFrame, pd.DataFrame]:
        records: list[dict[str, Any]] = []
        for name, coefficients in definitions.items():
            if not np.isclose(sum(coefficients.values()), 0.0, atol=0, rtol=0):
                raise ValueError(f"E7 contrast {name} 系数和必须为零")
            for seed in SEEDS:
                for metric in PRIMARY:
                    value = sum(
                        coefficient
                        * _utility(metric, indexed.loc[(arm, seed), metric + "_mean"])
                        for arm, coefficient in coefficients.items()
                    )
                    if not np.isfinite(value):
                        raise FloatingPointError("E7 contrast 非有限")
                    records.append(
                        {
                            "contrast": name,
                            "seed": seed,
                            "split": split,
                            "metric": metric,
                            "utility_delta": value,
                            "positive_means": "improvement",
                            "coefficients": json.dumps(coefficients, sort_keys=True),
                        }
                    )
        per_seed = pd.DataFrame(records)
        grouped: list[dict[str, Any]] = []
        for (name, metric), group in per_seed.groupby(["contrast", "metric"], sort=False):
            values = group.utility_delta.to_numpy(dtype=float)
            tolerance = PCC_TOLERANCE if metric == PCC else np.nan
            grouped.append(
                {
                    "contrast": name,
                    "split": split,
                    "metric": metric,
                    "utility_delta_mean": values.mean(),
                    "utility_delta_sample_sd": values.std(ddof=1),
                    "positive_seeds": int((values > 0).sum()),
                    "zero_seeds": int((values == 0).sum()),
                    "negative_seeds": int((values < 0).sum()),
                    "absolute_tolerance": tolerance,
                }
            )
        return per_seed, pd.DataFrame(grouped)

    simple_seed, simple_across = calculate(SIMPLE_CONTRASTS)
    factorial_seed, factorial_across = calculate(FACTORIAL_CONTRASTS)
    return simple_seed, simple_across, factorial_seed, factorial_across


def across_seed_table(frame: pd.DataFrame, *, split: str = "val") -> pd.DataFrame:
    indexed = _matrix_index(frame, split=split)
    del indexed
    records: list[dict[str, Any]] = []
    for arm in ARMS:
        arm_frame = frame.loc[frame.arm.eq(arm)].set_index("seed").loc[list(SEEDS)]
        for metric in PRIMARY:
            values = arm_frame[metric + "_mean"].to_numpy(dtype=float)
            records.append(
                {
                    "arm": arm,
                    "split": split,
                    "metric": metric,
                    "mean": values.mean(),
                    "sample_sd": values.std(ddof=1),
                    "seed_count": len(values),
                }
            )
    return pd.DataFrame(records)


def write_validation_summary(frame: pd.DataFrame, output: Path) -> Path:
    """把完整 18-cell seed metrics 写成不可覆盖的析因汇总目录。"""

    _matrix_index(frame, split="val")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    simple_seed, simple_across, factorial_seed, factorial_across = contrast_tables(frame)
    tables = {
        "per_seed.csv": frame.sort_values(["arm", "seed"]).reset_index(drop=True),
        "across_seed.csv": across_seed_table(frame),
        "simple_effects_per_seed.csv": simple_seed,
        "simple_effects_across_seed.csv": simple_across,
        "factorial_contrasts_per_seed.csv": factorial_seed,
        "factorial_contrasts_across_seed.csv": factorial_across,
    }
    for filename, table in tables.items():
        table.to_csv(output / filename, index=False, na_rep="NA")
    (output / "parameter_compute_report.json").write_text(
        json.dumps(parameter_compute_report(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    receipt = {
        "protocol": PROTOCOL,
        "split": "val",
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "seed_cells": len(frame),
        "tables": {name: len(table) for name, table in tables.items()},
        "positive_contrast_means": "improvement",
    }
    (output / "summary_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def subject_macro_table(metrics: pd.DataFrame, *, split: str = "val") -> pd.DataFrame:
    required = {"arm", "seed", "split", "samp_id", *PRIMARY}
    if not required.issubset(metrics.columns):
        raise ValueError("E7 subject-macro 输入缺少字段")
    if set(metrics["split"]) != {split} or not set(metrics["arm"]).issubset(ARMS):
        raise ValueError("E7 subject-macro split/arm 身份错误")
    if not set(metrics["seed"]).issubset(SEEDS):
        raise ValueError("E7 subject-macro seed 身份错误")
    records: list[dict[str, Any]] = []
    for (arm, seed), group in metrics.groupby(["arm", "seed"], sort=False):
        subjects = sorted(group.samp_id.unique().tolist())
        for metric in PRIMARY:
            subject_values: list[float] = []
            for samp_id in subjects:
                values = group.loc[group.samp_id.eq(samp_id), metric].to_numpy(dtype=float)
                finite = values[np.isfinite(values)]
                if len(finite) == 0:
                    raise ValueError(f"E7 subject {samp_id} 的 {metric} 没有 eligible 值")
                value = float(finite.mean())
                subject_values.append(value)
                records.append(
                    {
                        "arm": arm,
                        "seed": int(seed),
                        "split": split,
                        "metric": metric,
                        "samp_id": samp_id,
                        "subject_mean": value,
                        "subject_macro_mean": np.nan,
                        "subject_count": len(subjects),
                        "row_type": "subject",
                    }
                )
            records.append(
                {
                    "arm": arm,
                    "seed": int(seed),
                    "split": split,
                    "metric": metric,
                    "samp_id": "__macro__",
                    "subject_mean": np.nan,
                    "subject_macro_mean": float(np.mean(subject_values)),
                    "subject_count": len(subjects),
                    "row_type": "macro",
                }
            )
    return pd.DataFrame(records)


def scale_representation_statistics(value: torch.Tensor) -> tuple[dict[str, float], np.ndarray]:
    if value.ndim != 4 or value.shape[2] != 97:
        raise ValueError("E7 representation diagnostics 期望 (B,C,97,T)")
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError("E7 representation diagnostics 输入非有限")
    observations = value.detach().double().permute(0, 1, 3, 2).reshape(-1, 97)
    scale_variance = observations.var(dim=1, unbiased=False).mean()
    centered = observations - observations.mean(dim=0, keepdim=True)
    column_ss = centered.square().sum(dim=0)
    if bool((column_ss <= 0).any()):
        raise ValueError("E7 scale correlation 存在零方差尺度")
    correlation = centered.T @ centered
    correlation = correlation / torch.sqrt(column_ss[:, None] * column_ss[None, :])
    correlation = (correlation + correlation.T) / 2
    eigenvalues = torch.linalg.eigvalsh(correlation).clamp_min(0)
    mass = eigenvalues.sum()
    if not bool(torch.isfinite(mass)) or float(mass) <= 0:
        raise FloatingPointError("E7 effective-rank eigenvalue 异常")
    probabilities = eigenvalues / mass
    nonzero = probabilities > 0
    effective_rank = torch.exp(-(probabilities[nonzero] * probabilities[nonzero].log()).sum())

    def lag_mean(start: int, stop: int) -> torch.Tensor:
        values = [torch.diagonal(correlation, offset=lag).mean() for lag in range(start, stop + 1)]
        return torch.stack(values).mean()

    summary = {
        "cross_scale_variance": float(scale_variance),
        "entropy_effective_rank": float(effective_rank),
        "correlation_lag_1": float(lag_mean(1, 1)),
        "correlation_lag_8_16": float(lag_mean(8, 16)),
        "correlation_lag_48_96": float(lag_mean(48, 96)),
    }
    if not np.isfinite(list(summary.values())).all():
        raise FloatingPointError("E7 representation summary 非有限")
    return summary, correlation.cpu().numpy()


def rms_ratio(delta: torch.Tensor, reference: torch.Tensor) -> float:
    if delta.shape != reference.shape:
        raise ValueError("E7 RMS ratio shape 不一致")
    numerator = delta.detach().float().square().mean().sqrt()
    denominator = reference.detach().float().square().mean().sqrt()
    if not bool(torch.isfinite(numerator)) or not bool(torch.isfinite(denominator)):
        raise FloatingPointError("E7 RMS ratio 非有限")
    if float(denominator) == 0:
        raise ZeroDivisionError("E7 RMS ratio reference 为零")
    return float(numerator / denominator)


def attention_statistics(weights: torch.Tensor) -> dict[str, float]:
    if weights.ndim != 4 or weights.shape[1:3] != (1, 97):
        raise ValueError("E7 attention diagnostics 期望 (B,1,97,T)")
    values = weights.detach().float()
    if not bool(torch.isfinite(values).all()) or not torch.allclose(
        values.sum(dim=2), torch.ones_like(values.sum(dim=2)), atol=1e-6, rtol=0
    ):
        raise FloatingPointError("E7 attention weights 非有限或未归一化")
    safe = values.clamp_min(torch.finfo(values.dtype).tiny)
    entropy = -(safe * safe.log()).sum(dim=2).mean()
    total_variation = (values[:, :, 1:] - values[:, :, :-1]).abs().sum(dim=2).mean()
    masses = [values[:, :, start:stop].sum(dim=2).mean() for start, stop in ((0, 25), (25, 49), (49, 73), (73, 97))]
    result = {
        "attention_entropy": float(entropy),
        "attention_total_variation": float(total_variation),
        **{f"region_{index}_mass": float(mass) for index, mass in enumerate(masses)},
    }
    if not np.isfinite(list(result.values())).all():
        raise FloatingPointError("E7 attention diagnostics 非有限")
    return result


def parameter_compute_report() -> dict[str, Any]:
    spatial = 97 * 360
    residual_macs = 4 * (96 * 9 + 96 * 96) * spatial
    attention_macs = (96 * 8 + 8) * spatial
    return {
        "unit": "per_180s_window_forward",
        "arms": {
            arm: {
                **arm_contract(arm),
                "covered_residual_encoder_macs": residual_macs
                if arm_contract(arm)["encoder"] != "s0_shallow"
                else 0,
                "covered_attention_scorer_macs": attention_macs
                if arm_contract(arm)["aggregation"] == "frequency_attention"
                else 0,
            }
            for arm in ARMS
        },
        "whole_model_flops": None,
        "excluded": [
            "normalization",
            "activation",
            "softmax_and_reductions",
            "interpolation",
            "Mamba_and_decoder",
            "backward_and_checkpoint_recomputation",
        ],
    }


def check_p1() -> dict[str, Any]:
    spec = load_experiment_spec()
    frequencies = load_frequencies()
    configs = {}
    for arm in ARMS:
        configs[arm] = {}
        for seed in SEEDS:
            baseline = baseline_config(seed)
            output = ROOT / OUTPUT / "formal" / arm / f"seed_{seed}"
            cfg = derived_config(
                baseline,
                arm,
                frequencies,
                output_root=output,
                device="cuda:0",
            )
            validate_config(
                cfg,
                baseline,
                arm,
                frequencies,
                output_root=output,
                device="cuda:0",
            )
            configs[arm][str(seed)] = OmegaConf.to_container(cfg, resolve=True)
    return {
        "protocol": str(spec.protocol),
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "resolved_config_count": sum(len(items) for items in configs.values()),
        "parameter_compute_report": parameter_compute_report(),
    }
