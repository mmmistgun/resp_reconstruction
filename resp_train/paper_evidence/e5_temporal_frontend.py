"""E5 时域前端：严格配置、来源锁、训练生命周期与 validation 汇总。"""

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
from omegaconf import DictConfig, OmegaConf

from resp_train.crd.config import check_crd_dependencies
from resp_train.crd.experiment import (
    CRDExperiment,
    _early_stopping_should_stop,
    _early_stopping_step,
)
from resp_train.crd.training import build_crd_optimizer, crd_learning_rate
from resp_train.data.index import filter_index
from resp_train.data.research_v2 import read_research_v2_index
from resp_train.metrics.task import summarize_task_metrics
from resp_train.paper_evidence.e1_scale_topology import ERRORS, PCC, PRIMARY, SEEDS, array_hash
from resp_train.paper_evidence.e1_scale_topology_runtime import (
    environment,
    git_state,
    identity,
    sha256_file,
    write_json,
)
from resp_train.paper_evidence.e5_temporal_frontend_model import (
    ARM,
    CONTROL_ARM,
    FRONTEND_CONTRACT,
    FRONTEND_PARAMETERS,
    MODEL_PARAMETERS,
    PROTOCOL,
    W0_FRONTEND_PARAMETERS,
    W0_MODEL_PARAMETERS,
    build_e5_model,
)


ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("configs/e5_temporal_frontend/e5_tfe101_aa10_res_w0.yaml")
SOURCE_LOCK = Path("docs/experiments/e4_w0_scale_aggregation_implementation_lock_20260917.json")
SOURCE_LOCK_SHA256 = "464e073dbd5707a30575d606dec2a84dcd89a161945e537c463b214a15c2b493"
SOURCE_AUDIT = Path("docs/experiments/e5_temporal_frontend_source_audit_20260922.json")
PROTOCOL_PATH = Path("docs/experiments/e5_temporal_frontend_protocol_20260922.md")
LOCK_PATH = Path("docs/experiments/e5_temporal_frontend_implementation_lock_20260923.json")
SCRIPT_PATH = Path("scripts/run_e5_temporal_frontend.py")
TEST_PATH = Path("tests/test_e5_temporal_frontend.py")
MODEL_PATH = Path("resp_train/paper_evidence/e5_temporal_frontend_model.py")
CONTROL_PATH = Path("resp_train/paper_evidence/e5_temporal_frontend.py")
ENGINEERING_PATH = Path("resp_train/paper_evidence/e5_temporal_frontend_engineering.py")
OUTPUT = Path("runs/e5_temporal_frontend")
COUNTS = {"train": 10_141, "val": 2_675}
SAMP_IDS = {"train": 32, "val": 7}
EPOCHS = 80
UPDATES_PER_EPOCH = 80
EARLY_STOP_MIN_EPOCH = 30
EARLY_STOP_PATIENCE = 15
EARLY_STOP_MIN_DELTA = 0.0

_SPEC_TOP_KEYS = {
    "schema_version",
    "protocol",
    "arm",
    "control_arm",
    "source",
    "frontend",
    "matrix",
}
_MATRIX = {
    "seeds": list(SEEDS),
    "train_windows": COUNTS["train"],
    "validation_windows": COUNTS["val"],
    "train_samp_ids": SAMP_IDS["train"],
    "validation_samp_ids": SAMP_IDS["val"],
    "epochs": EPOCHS,
    "optimizer_updates": EPOCHS * UPDATES_PER_EPOCH,
    "early_stopping_enabled": True,
    "early_stopping_min_epoch": EARLY_STOP_MIN_EPOCH,
    "early_stopping_patience": EARLY_STOP_PATIENCE,
    "early_stopping_min_delta": EARLY_STOP_MIN_DELTA,
    "physical_batch": 128,
    "gradient_accumulation_steps": 1,
    "amp_dtype": "bfloat16",
    "checkpoint_selector": "full_validation_local_rr_strict_minimum_earliest_tie",
    "test_enabled": False,
}


def verify(path: Path, expected: Mapping[str, Any]) -> None:
    if not path.is_file() or identity(path) != {key: expected[key] for key in ("size_bytes", "sha256")}:
        raise RuntimeError(f"E5 文件身份漂移: {path}")


def load_experiment_spec(path: Path | None = None) -> DictConfig:
    path = ROOT / SPEC_PATH if path is None else Path(path)
    cfg = OmegaConf.load(path)
    OmegaConf.resolve(cfg)
    if set(cfg.keys()) != _SPEC_TOP_KEYS:
        raise ValueError("E5 spec 顶层字段漂移")
    if (
        int(cfg.schema_version) != 1
        or str(cfg.protocol) != PROTOCOL
        or str(cfg.arm) != ARM
        or str(cfg.control_arm) != CONTROL_ARM
        or set(cfg.source.keys())
        != {"w0_implementation_lock", "w0_implementation_lock_sha256", "source_audit"}
        or str(cfg.source.w0_implementation_lock) != str(SOURCE_LOCK)
        or str(cfg.source.w0_implementation_lock_sha256) != SOURCE_LOCK_SHA256
        or str(cfg.source.source_audit) != str(SOURCE_AUDIT)
        or OmegaConf.to_container(cfg.frontend, resolve=True) != FRONTEND_CONTRACT
        or OmegaConf.to_container(cfg.matrix, resolve=True) != _MATRIX
    ):
        raise ValueError("E5 spec 科学合同漂移")
    return cfg


def derived_config(baseline: DictConfig, *, output_root: Path, device: str) -> DictConfig:
    """保留同 seed W0 全部训练字段，只增加独立协议和前端身份。"""
    cfg = OmegaConf.create(OmegaConf.to_container(baseline, resolve=True))
    cfg.protocol.name = PROTOCOL
    cfg.protocol.execution_gate = "e5_formal"
    cfg.model.e5_temporal_frontend = FRONTEND_CONTRACT
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
    *,
    output_root: Path,
    device: str,
) -> None:
    expected = derived_config(baseline, output_root=output_root, device=device)
    if OmegaConf.to_container(cfg, resolve=True) != OmegaConf.to_container(expected, resolve=True):
        raise ValueError("E5 配置必须与同 seed W0 合同一致，仅开放前端和实验/运行身份")
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
        raise ValueError("E5 W0 来源配置不符合冻结矩阵")
    if (
        not bool(cfg.training.early_stopping_enabled)
        or int(cfg.training.early_stopping_min_epoch) != EARLY_STOP_MIN_EPOCH
        or int(cfg.training.early_stopping_patience) != EARLY_STOP_PATIENCE
        or float(cfg.training.early_stopping_min_delta) != EARLY_STOP_MIN_DELTA
    ):
        raise ValueError("E5 early stopping 合同漂移")


def _critical_paths() -> tuple[Path, ...]:
    paths = (
        MODEL_PATH,
        CONTROL_PATH,
        ENGINEERING_PATH,
        SCRIPT_PATH,
        TEST_PATH,
        SPEC_PATH,
        PROTOCOL_PATH,
        SOURCE_AUDIT,
        Path("resp_train/crd/frontends.py"),
        Path("resp_train/crd/model.py"),
        Path("resp_train/crd/tf_v1_model.py"),
        Path("resp_train/crd/blocks.py"),
        Path("resp_train/crd/initialization.py"),
        Path("resp_train/crd/experiment.py"),
        Path("resp_train/crd/training.py"),
        Path("resp_train/losses/task.py"),
        Path("resp_train/metrics/task.py"),
        Path("resp_train/temporal/blocks.py"),
    )
    if any(not (ROOT / path).is_file() for path in paths):
        missing = [str(path) for path in paths if not (ROOT / path).is_file()]
        raise FileNotFoundError(f"E5 implementation lock 缺少源码: {missing}")
    return paths


def prepare_lock(root: Path = ROOT) -> Path:
    """重核 W0/cache 字节身份后，一次性建立不可覆盖的 E5 实现锁。"""
    load_experiment_spec(root / SPEC_PATH)
    destination = root / LOCK_PATH
    if destination.exists():
        raise FileExistsError(f"E5 implementation lock 已存在: {destination}")
    if sha256_file(root / SOURCE_LOCK) != SOURCE_LOCK_SHA256:
        raise ValueError("E5 W0 来源锁漂移")
    source = json.loads((root / SOURCE_LOCK).read_text(encoding="utf-8"))
    if tuple(int(seed) for seed in source["seeds"]) != SEEDS:
        raise ValueError("E5 W0 来源 seed 漂移")
    files: dict[str, dict[str, Any]] = {
        str(SOURCE_LOCK): identity(root / SOURCE_LOCK),
        str(SOURCE_AUDIT): identity(root / SOURCE_AUDIT),
    }
    baselines: dict[str, Any] = {}
    templates: dict[str, Any] = {}
    for entry in source["w0_entries"]:
        seed = int(entry["seed"])
        for name in (
            "checkpoint_best_local_rr.pt",
            "config.yaml",
            "run_manifest.json",
            "metrics_summary.csv",
            "metrics.csv",
        ):
            relative = str(Path(entry["run_dir"]) / name)
            expected = source["source_files"].get(relative)
            if expected is None:
                raise ValueError(f"E5 W0 来源锁缺少 {relative}")
            verify(root / relative, expected)
            files[relative] = expected
        baseline = OmegaConf.create(source["baselines"][str(seed)])
        output_root = root / OUTPUT / "formal" / f"seed_{seed}"
        candidate = derived_config(baseline, output_root=output_root, device="cuda:0")
        validate_config(candidate, baseline, output_root=output_root, device="cuda:0")
        baselines[str(seed)] = OmegaConf.to_container(baseline, resolve=True)
        templates[str(seed)] = OmegaConf.to_container(candidate, resolve=True)
    cache = source["cache_lock"]
    for key in ("manifest", "train_w", "val_w", "frequency_file"):
        entry = cache[key]
        expected = {
            "size_bytes": entry["size_bytes"],
            "sha256": entry.get("sha256", entry.get("file_sha256")),
        }
        verify(root / entry["path"], expected)
        files[entry["path"]] = expected
    for split in COUNTS:
        relative = str(Path(cache["root"]) / f"{split}_row_ids.npy")
        expected = source["source_files"].get(relative)
        if expected is None:
            expected = identity(root / relative)
        verify(root / relative, expected)
        files[relative] = expected
    cache_manifest = json.loads((root / cache["manifest"]["path"]).read_text(encoding="utf-8"))
    state = git_state(root)
    if state["status_porcelain"]:
        raise RuntimeError("E5 implementation lock 要求先提交实现并保持工作树干净")
    lock = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "arm": ARM,
        "control_arm": CONTROL_ARM,
        "seeds": list(SEEDS),
        "counts": COUNTS,
        "samp_ids": SAMP_IDS,
        "epochs": EPOCHS,
        "updates_per_epoch": UPDATES_PER_EPOCH,
        "early_stopping": {
            "enabled": True,
            "min_epoch": EARLY_STOP_MIN_EPOCH,
            "patience": EARLY_STOP_PATIENCE,
            "min_delta": EARLY_STOP_MIN_DELTA,
            "monitor": "full_validation_local_rr_strict_minimum_earliest_tie",
        },
        "frontend_contract": FRONTEND_CONTRACT,
        "w0_entries": source["w0_entries"],
        "baselines": baselines,
        "resolved_templates": templates,
        "cache_lock": cache,
        "dataset_index": {
            "path": cache_manifest["dataset_index"],
            "sha256": cache_manifest["dataset_index_sha256"],
        },
        "source_files": files,
        "code_files": {
            str(path): identity(root / path) for path in _critical_paths()
        },
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "preparation_git": state,
        "status": "implementation_locked_gpu_and_training_pending",
    }
    write_json(destination, lock)
    return destination


def load_lock(root: Path = ROOT) -> tuple[dict[str, Any], str]:
    path = root / LOCK_PATH
    if not path.is_file():
        raise FileNotFoundError(f"E5 implementation lock 尚未建立: {path}")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if (
        lock.get("schema_version") != 1
        or lock.get("protocol") != PROTOCOL
        or lock.get("arm") != ARM
        or lock.get("control_arm") != CONTROL_ARM
        or tuple(lock.get("seeds", ())) != SEEDS
        or lock.get("counts") != COUNTS
        or lock.get("samp_ids") != SAMP_IDS
        or lock.get("epochs") != EPOCHS
        or lock.get("updates_per_epoch") != UPDATES_PER_EPOCH
        or lock.get("early_stopping")
        != {
            "enabled": True,
            "min_epoch": EARLY_STOP_MIN_EPOCH,
            "patience": EARLY_STOP_PATIENCE,
            "min_delta": EARLY_STOP_MIN_DELTA,
            "monitor": "full_validation_local_rr_strict_minimum_earliest_tie",
        }
        or lock.get("frontend_contract") != FRONTEND_CONTRACT
    ):
        raise ValueError("E5 implementation lock 科学合同漂移")
    for relative, expected in lock["code_files"].items():
        verify(root / relative, expected)
    for seed in SEEDS:
        baseline = OmegaConf.create(lock["baselines"][str(seed)])
        template = OmegaConf.create(lock["resolved_templates"][str(seed)])
        validate_config(
            template,
            baseline,
            output_root=root / OUTPUT / "formal" / f"seed_{seed}",
            device="cuda:0",
        )
    return lock, sha256_file(path)


@contextmanager
def phase_guard(
    parent: Path,
    lock_hash: str,
    *,
    completed_phase: str | None = None,
) -> Iterator[None]:
    parent.mkdir(parents=True, exist_ok=True)
    with (parent / f".execution_{lock_hash}.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("E5 相同身份正在运行") from exc
        try:
            if completed_phase is not None:
                for receipt in parent.glob("*/freeze_receipt.json"):
                    manifest = json.loads((receipt.parent / "manifest.json").read_text())
                    if manifest["implementation_lock_sha256"] == lock_hash:
                        verify_attempt(receipt.parent, phase=completed_phase, lock_hash=lock_hash)
                        raise FileExistsError(f"E5 相同身份阶段已完成: {receipt.parent}")
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def attempt(
    parent: Path,
    lock_hash: str,
    phase: str,
    seed: int | None = None,
) -> Iterator[Path]:
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = parent / f"{phase}_{lock_hash[:12]}_{stamp}_{uuid4().hex[:12]}"
    path.mkdir(exist_ok=False)
    context = {
        "protocol": PROTOCOL,
        "arm": ARM,
        "seed": seed,
        "phase": phase,
        "implementation_lock_sha256": lock_hash,
        "command": sys.argv,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(path / "lifecycle_started.json", {**context, "status": "running"})
    try:
        yield path
        write_json(
            path / "lifecycle_completed.json",
            {**context, "status": "completed", "ended_at": datetime.now(timezone.utc).isoformat()},
        )
        manifest = {
            **context,
            "status": "completed",
            "files": {
                str(file.relative_to(path)): identity(file)
                for file in sorted(path.rglob("*"))
                if file.is_file()
            },
        }
        write_json(path / "manifest.json", manifest)
        write_json(
            path / "freeze_receipt.json",
            {"protocol": PROTOCOL, "manifest": identity(path / "manifest.json")},
        )
    except BaseException as exc:
        write_json(
            path / "lifecycle_failed.json",
            {
                **context,
                "status": "failed",
                "error": str(exc),
                "error_type": type(exc).__name__,
                "traceback": traceback.format_exc(),
            },
        )
        raise


def verify_attempt(path: Path, *, phase: str, lock_hash: str) -> dict[str, Any]:
    path = path.resolve()
    if (path / "lifecycle_failed.json").exists():
        raise ValueError(f"E5 attempt 失败: {path}")
    freeze = json.loads((path / "freeze_receipt.json").read_text())
    verify(path / "manifest.json", freeze["manifest"])
    manifest = json.loads((path / "manifest.json").read_text())
    if (
        manifest["protocol"] != PROTOCOL
        or manifest["phase"] != phase
        or manifest["status"] != "completed"
        or manifest["implementation_lock_sha256"] != lock_hash
    ):
        raise ValueError("E5 attempt 协议/实现身份不一致")
    required = {"lifecycle_completed.json"}
    required.update(
        {
            "gpu_acceptance": {"environment.json", "gpu_acceptance.json", "access_receipt.json"},
            "benchmark": {"environment.json", "benchmark.json", "access_receipt.json"},
            "formal": {"environment.json", "formal_receipt.json", "implementation_lock.json", "access_receipt.json"},
            "summary": {
                "summary_receipt.json",
                "seed_metrics.csv",
                "paired_seed_delta.csv",
                "three_seed_comparison.csv",
                "paired_seed_subject_delta.csv",
                "subject_macro_by_seed.csv",
            },
        }[phase]
    )
    if not required.issubset(manifest["files"]):
        raise ValueError("E5 attempt 缺少必需回执")
    for relative, expected in manifest["files"].items():
        file = (path / relative).resolve()
        if not file.is_relative_to(path):
            raise ValueError("E5 manifest 路径越界")
        verify(file, expected)
    return manifest


def runtime_preflight(device: str) -> dict[str, Any]:
    state = git_state()
    if state["status_porcelain"]:
        raise RuntimeError("E5 GPU/正式执行要求干净提交")
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise ValueError("E5 GPU 阶段需要 CUDA；CPU 验证使用 synthetic 定向测试")
    problems = check_crd_dependencies()
    if problems:
        raise RuntimeError("E5 原生依赖检查失败: " + "; ".join(problems))
    return {**environment(device), "git": state}


def audit_sources(
    lock: Mapping[str, Any], cfg: DictConfig, output: Path
) -> dict[str, pd.DataFrame]:
    write_json(
        output / "access_started.json",
        {
            "splits": ["train", "val"],
            "sources": list(lock["source_files"]),
            "dataset_index": lock["dataset_index"],
            "purpose": "E5 training and validation",
            "test_access": False,
        },
    )
    for relative, expected in lock["source_files"].items():
        verify(ROOT / relative, expected)
    index = lock["dataset_index"]
    if sha256_file(Path(index["path"])) != index["sha256"]:
        raise ValueError("E5 dataset index 漂移")
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
        if (
            len(frame) != count
            or frame.dataset_row_id.duplicated().any()
            or set(frame.split.astype(str)) != {split}
            or frame.samp_id.nunique() != SAMP_IDS[split]
        ):
            raise ValueError(f"E5 {split} rows 不符合冻结合同")
        expected_hash = lock["cache_lock"]["row_identity"][f"{split}_row_content_sha256"]
        if array_hash(np.sort(frame.dataset_row_id.to_numpy())) != expected_hash:
            raise ValueError(f"E5 {split} row 集合漂移")
        frame.to_csv(output / f"{split}_rows.csv", index=False)
        rows[split] = frame
    if set(rows["train"].samp_id) & set(rows["val"].samp_id):
        raise ValueError("E5 train/validation samp_id 交叉")
    write_json(
        output / "access_receipt.json",
        {
            "counts": COUNTS,
            "samp_ids": SAMP_IDS,
            "row_order_sha256": {
                split: array_hash(frame.dataset_row_id.to_numpy())
                for split, frame in rows.items()
            },
            "source_files_verified": lock["source_files"],
            "dataset_index": index,
            "sample_seeds": {
                split: int(cfg.data[f"{split}_sample_seed"]) for split in COUNTS
            },
            "train_accessed": True,
            "validation_accessed": True,
            "research_test_accessed": False,
        },
    )
    return rows


def validate_metrics(metrics: pd.DataFrame, rows: pd.DataFrame) -> dict[str, int]:
    if (
        len(metrics) != len(rows)
        or rows.dataset_row_id.duplicated().any()
        or metrics.dataset_row_id.duplicated().any()
    ):
        raise ValueError("E5 metrics row 数量/重复 identity 不一致")
    for key in ("dataset_row_id", "samp_id", "split"):
        if not np.array_equal(metrics[key].to_numpy(), rows[key].to_numpy()):
            raise ValueError(f"E5 metrics identity/order 不一致: {key}")
    flags = {
        ERRORS[0]: "whole_rr_target_eligible",
        ERRORS[1]: "local_rr_target_eligible",
        PCC: "joint_target_eligible",
    }
    for metric in PRIMARY:
        expected = np.ones(len(rows), dtype=bool)
        if metric in flags:
            value = metrics[flags[metric]]
            if value.isna().any() or not value.isin([True, False]).all():
                raise ValueError("E5 target eligibility 非法")
            expected = value.to_numpy(dtype=bool)
        values = metrics[metric].to_numpy(dtype=float)
        if np.isinf(values).any() or not np.array_equal(np.isfinite(values), expected):
            raise FloatingPointError(f"E5 finite/eligibility 不一致: {metric}")
    degeneracy: dict[str, int] = {}
    for key in ("joint_prediction_degenerate", "envelope_spearman_prediction_degenerate"):
        value = metrics[key]
        if value.isna().any() or not value.isin([True, False]).all():
            raise ValueError(f"E5 prediction degeneracy flags 非法: {key}")
        degeneracy[key] = int(value.astype(bool).sum())
    return degeneracy


class TemporalFrontendExperiment(CRDExperiment):
    task_name = ARM

    def __init__(self, cfg: DictConfig, validation_rows: pd.DataFrame):
        super().__init__(cfg)
        self.validation_rows = validation_rows

    def _build_model(self):
        return build_e5_model(self.cfg)

    def _evaluate_model(self, model, loader, **kwargs):
        frame = super()._evaluate_model(model, loader, **kwargs)
        validate_metrics(frame, self.validation_rows)
        frame.insert(0, "arm", ARM)
        frame.insert(0, "seed", int(self.cfg.training.seed))
        return frame


def finite_tree(value: Any) -> None:
    if torch.is_tensor(value):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError("E5 tensor 非有限")
    elif isinstance(value, Mapping):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            finite_tree(item)
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise FloatingPointError("E5 scalar 非有限")


def validate_history(history: pd.DataFrame, cfg: DictConfig) -> int:
    completed_epochs = len(history)
    if (
        completed_epochs < EARLY_STOP_MIN_EPOCH
        or completed_epochs > EPOCHS
        or not np.array_equal(history.epoch, np.arange(1, completed_epochs + 1))
    ):
        raise ValueError("E5 history 必须连续且覆盖 min_epoch 至最多 80 epochs")
    if not np.isfinite(history.select_dtypes(include=np.number).to_numpy()).all():
        raise FloatingPointError("E5 history 非有限")
    if not np.array_equal(
        history.optimizer_update,
        np.arange(1, completed_epochs + 1) * UPDATES_PER_EPOCH,
    ):
        raise ValueError("E5 optimizer update 数不完整")
    if not np.allclose(
        history.train_loss_total,
        history.train_loss_sync + 0.25 * history.train_loss_effort,
        atol=1e-12,
        rtol=0,
    ):
        raise ValueError("E5 total loss 与完整目标不一致")
    total = EPOCHS * UPDATES_PER_EPOCH
    for row in history.itertuples():
        for column, update in (
            ("first_learning_rate", (row.epoch - 1) * UPDATES_PER_EPOCH),
            ("last_learning_rate", row.epoch * UPDATES_PER_EPOCH - 1),
        ):
            expected = crd_learning_rate(
                update,
                total_updates=total,
                max_learning_rate=float(cfg.training.max_learning_rate),
                min_learning_rate=float(cfg.training.min_learning_rate),
                warmup_fraction=float(cfg.training.warmup_fraction),
            )
            if not np.isclose(getattr(row, column), expected, atol=1e-15, rtol=0):
                raise ValueError("E5 learning-rate schedule 漂移")
    required = {
        "early_stopping_improved",
        "early_stopping_wait",
        "early_stopping_triggered",
        "early_stopping_min_epoch",
    }
    if not required.issubset(history.columns):
        raise ValueError("E5 history 缺少 early stopping 字段")
    best = float("inf")
    wait = 0
    trigger_values: list[int] = []
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
            raise ValueError("E5 history early stopping 轨迹漂移")
        trigger_values.append(int(triggered))
    if any(trigger_values[:-1]):
        raise ValueError("E5 history 在触发 early stopping 后仍继续")
    if completed_epochs < EPOCHS and trigger_values[-1] != 1:
        raise ValueError("E5 history 提前结束但未触发 early stopping")
    # np.argmin 首次出现即最早严格最小，符合冻结 selector。
    return int(history.iloc[int(np.argmin(history.val_local_rr_mae.to_numpy()))].epoch)


def validate_run(run_dir: Path, cfg: DictConfig, rows: pd.DataFrame) -> dict[str, Any]:
    saved = OmegaConf.load(run_dir / "config.yaml")
    if OmegaConf.to_container(saved, resolve=True) != OmegaConf.to_container(cfg, resolve=True):
        raise ValueError("E5 保存配置漂移")
    history = pd.read_csv(run_dir / "train_history.csv")
    best_epoch = validate_history(history, cfg)
    final_epoch = int(history.iloc[-1].epoch)
    model = build_e5_model(cfg)
    optimizer, partition = build_crd_optimizer(model, cfg)
    saved_groups = json.loads((run_dir / "optimizer_parameter_groups.json").read_text())
    if saved_groups != {
        "weight_decay": float(cfg.training.weight_decay),
        "decay": list(partition.decay_names),
        "no_decay": list(partition.no_decay_names),
    }:
        raise ValueError("E5 optimizer 参数分组漂移")
    for name, expected_epoch in (
        ("checkpoint_best_local_rr.pt", best_epoch),
        ("checkpoint_final.pt", final_epoch),
    ):
        checkpoint = torch.load(run_dir / name, map_location="cpu", weights_only=False)
        finite_tree(checkpoint)
        if (
            checkpoint["epoch"] != expected_epoch
            or checkpoint["config"] != OmegaConf.to_container(cfg, resolve=True)
        ):
            raise ValueError("E5 checkpoint epoch/config identity 漂移")
        history_row = history.loc[history.epoch.eq(expected_epoch)].iloc[0]
        for key in history.columns:
            if key not in checkpoint["metrics"] or not np.isclose(
                float(checkpoint["metrics"][key]),
                float(history_row[key]),
                atol=1e-12,
                rtol=1e-12,
            ):
                raise ValueError(f"E5 checkpoint 与 history 不一致: {key}")
        extra = checkpoint["extra_state"]
        if (
            extra["protocol"] != PROTOCOL
            or extra["update_index"] != expected_epoch * UPDATES_PER_EPOCH
            or extra["total_updates"] != EPOCHS * UPDATES_PER_EPOCH
        ):
            raise ValueError("E5 checkpoint update/protocol 不一致")
        early = extra.get("early_stopping")
        if (
            not isinstance(early, Mapping)
            or early.get("enabled") is not True
            or early.get("monitor") != "validation_local_rr_mae_full_split"
            or int(early.get("min_epoch", -1)) != EARLY_STOP_MIN_EPOCH
            or int(early.get("patience", -1)) != EARLY_STOP_PATIENCE
            or float(early.get("min_delta", np.nan)) != EARLY_STOP_MIN_DELTA
            or int(early.get("epochs_without_improvement", -1))
            != int(history_row.early_stopping_wait)
            or bool(early.get("triggered"))
            != bool(history_row.early_stopping_triggered)
            or int(early.get("planned_epochs", -1)) != EPOCHS
        ):
            raise ValueError("E5 checkpoint early stopping 合同漂移")
        if name == "checkpoint_final.pt" and int(early.get("completed_epochs", -1)) != final_epoch:
            raise ValueError("E5 final checkpoint completed_epochs 漂移")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                state = optimizer.state.get(parameter, {})
                if (
                    not {"step", "exp_avg", "exp_avg_sq"}.issubset(state)
                    or float(state["step"]) != expected_epoch * UPDATES_PER_EPOCH
                    or state["exp_avg"].shape != parameter.shape
                    or state["exp_avg_sq"].shape != parameter.shape
                ):
                    raise ValueError("E5 optimizer state/step 不完整")
    metrics = pd.read_csv(run_dir / "metrics.csv")
    degeneracy = validate_metrics(metrics, rows)
    if not metrics.arm.eq(ARM).all() or not metrics.seed.eq(int(cfg.training.seed)).all():
        raise ValueError("E5 metrics arm/seed identity 不一致")
    summary = pd.read_csv(run_dir / "metrics_summary.csv")
    if len(summary) != 1:
        raise ValueError("E5 summary 必须恰有一行")
    expected = summarize_task_metrics(metrics)
    for key in PRIMARY:
        if (
            not np.isfinite(summary.iloc[0][key + "_mean"])
            or not np.isclose(
                summary.iloc[0][key + "_mean"],
                expected.iloc[0][key + "_mean"],
                atol=1e-12,
                rtol=0,
            )
            or int(summary.iloc[0][key + "_n"])
            != int(expected.iloc[0][key + "_n"])
        ):
            raise ValueError(f"E5 summary 或分母不一致: {key}")
    return {
        "protocol": PROTOCOL,
        "arm": ARM,
        "seed": int(cfg.training.seed),
        "planned_epochs": EPOCHS,
        "completed_epochs": final_epoch,
        "planned_updates": EPOCHS * UPDATES_PER_EPOCH,
        "completed_updates": final_epoch * UPDATES_PER_EPOCH,
        "early_stopping_triggered": bool(final_epoch < EPOCHS or history.iloc[-1].early_stopping_triggered),
        "selected_epoch": best_epoch,
        "validation_rows": len(metrics),
        "checkpoint_and_history_finite": True,
        "prediction_degeneracy": degeneracy,
        "quality_acceptance_passed": not any(degeneracy.values()),
        "validation_row_order_sha256": array_hash(rows.dataset_row_id.to_numpy()),
    }


def validate_anchor_rows(
    metrics: pd.DataFrame, lock: Mapping[str, Any], seed: int
) -> None:
    entry = next(item for item in lock["w0_entries"] if int(item["seed"]) == seed)
    path = ROOT / entry["run_dir"] / "metrics.csv"
    verify(path, lock["source_files"][str(path.relative_to(ROOT))])
    full = pd.read_csv(path).sort_values("dataset_row_id")
    candidate = metrics.sort_values("dataset_row_id")
    if full.dataset_row_id.duplicated().any() or len(full) != COUNTS["val"]:
        raise ValueError("E5 W0 validation 来源身份不完整")
    for key in (
        "dataset_row_id",
        "samp_id",
        "split",
        "whole_rr_target_eligible",
        "local_rr_target_eligible",
        "joint_target_eligible",
    ):
        if not np.array_equal(full[key].to_numpy(), candidate[key].to_numpy()):
            raise ValueError(f"E5 与 W0 validation identity/eligibility 不一致: {key}")


def run_formal(seed: int, *, gpu_receipt: Path, device: str = "cuda:0") -> Path:
    if seed not in SEEDS:
        raise ValueError("E5 seed 必须属于冻结三 seed 矩阵")
    lock, lock_hash = load_lock()
    parent = ROOT / OUTPUT / "formal" / f"seed_{seed}"
    with phase_guard(parent, lock_hash):
        for receipt in parent.glob("*/freeze_receipt.json"):
            manifest = json.loads((receipt.parent / "manifest.json").read_text())
            if manifest["implementation_lock_sha256"] == lock_hash:
                verify_attempt(receipt.parent, phase="formal", lock_hash=lock_hash)
                raise FileExistsError(f"E5 同一身份 seed 已完成: {receipt.parent}")
        with attempt(parent, lock_hash, "formal", seed) as output:
            write_json(output / "environment.json", runtime_preflight(device))
            verify_attempt(gpu_receipt, phase="gpu_acceptance", lock_hash=lock_hash)
            accepted = json.loads((gpu_receipt / "gpu_acceptance.json").read_text())
            if (
                accepted.get("passed") is not True
                or accepted.get("frontend_contract") != FRONTEND_CONTRACT
                or accepted.get("seeds") != list(SEEDS)
                or accepted.get("physical_batch") != 128
            ):
                raise ValueError("E5 synthetic GPU 验收未通过")
            accepted_env = json.loads((gpu_receipt / "environment.json").read_text())
            current_env = json.loads((output / "environment.json").read_text())
            if accepted_env["git"]["commit"] != current_env["git"]["commit"]:
                raise ValueError("E5 GPU 验收与正式训练 commit 不一致")
            for key in (
                "packages",
                "gpu_name",
                "cuda",
                "cudnn",
                "cudnn_benchmark",
                "cudnn_deterministic",
                "matmul_allow_tf32",
                "cudnn_allow_tf32",
                "deterministic_algorithms",
            ):
                if accepted_env.get(key) != current_env.get(key):
                    raise ValueError(f"E5 GPU 验收与正式训练环境不一致: {key}")
            write_json(
                output / "gpu_acceptance_source.json",
                {
                    "path": str(gpu_receipt.resolve()),
                    "manifest": identity(gpu_receipt / "manifest.json"),
                },
            )
            write_json(output / "implementation_lock.json", lock)
            baseline = OmegaConf.create(lock["baselines"][str(seed)])
            cfg = derived_config(
                baseline,
                output_root=output / "training",
                device=device,
            )
            validate_config(
                cfg,
                baseline,
                output_root=output / "training",
                device=device,
            )
            rows = audit_sources(lock, cfg, output)
            run_dir = TemporalFrontendExperiment(cfg, rows["val"]).train()
            receipt = validate_run(run_dir, cfg, rows["val"])
            validate_anchor_rows(pd.read_csv(run_dir / "metrics.csv"), lock, seed)
            receipt["run_dir"] = str(run_dir.relative_to(output))
            write_json(output / "formal_receipt.json", receipt)
        return output


def paired_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected = {
        (arm, seed) for arm in ("W0_FULL", ARM) for seed in SEEDS
    }
    observed = set(frame[["arm", "seed"]].itertuples(index=False, name=None))
    if frame[["arm", "seed"]].duplicated().any() or observed != expected:
        raise ValueError("E5 配对汇总要求两个 arm 的完整三个 seed")
    columns = [key + "_mean" for key in PRIMARY]
    if not np.isfinite(frame[columns].to_numpy()).all():
        raise FloatingPointError("E5 汇总指标非有限")
    full = frame[frame.arm.eq("W0_FULL")].set_index("seed").loc[list(SEEDS)]
    candidate = frame[frame.arm.eq(ARM)].set_index("seed").loc[list(SEEDS)]
    if (frame[[key + "_mean" for key in ERRORS]].to_numpy() < 0).any():
        raise ValueError("E5 error 不得为负")
    paired: list[dict[str, Any]] = []
    aggregate: list[dict[str, Any]] = []
    for metric in PRIMARY:
        full_values = full[metric + "_mean"].to_numpy(dtype=float)
        candidate_values = candidate[metric + "_mean"].to_numpy(dtype=float)
        raw = full_values - candidate_values if metric == PCC else candidate_values - full_values
        delta = (
            raw
            if metric == PCC
            else np.divide(
                100 * raw,
                full_values,
                out=np.full_like(full_values, np.nan),
                where=full_values != 0,
            )
        )
        mean_delta = (
            float(raw.mean())
            if metric == PCC
            else (
                float(100 * raw.mean() / full_values.mean())
                if full_values.mean() != 0
                else np.nan
            )
        )
        for seed, before, after, relative, absolute in zip(
            SEEDS,
            full_values,
            candidate_values,
            delta,
            raw,
            strict=True,
        ):
            paired.append(
                {
                    "seed": seed,
                    "metric": metric,
                    "W0_FULL": float(before),
                    ARM: float(after),
                    "delta": float(relative),
                    "raw_delta": float(absolute),
                    "relative_delta_defined": bool(metric == PCC or before != 0),
                    "split": "val",
                    "unit": "absolute_drop" if metric == PCC else "relative_percent",
                }
            )
        aggregate.append(
            {
                "metric": metric,
                "full_mean": float(full_values.mean()),
                "full_sample_sd": float(full_values.std(ddof=1)),
                "candidate_mean": float(candidate_values.mean()),
                "candidate_sample_sd": float(candidate_values.std(ddof=1)),
                # 任一 seed 的相对变化未定义时，聚合相对变化保持 NA；不静默缩小 seed 集合。
                "paired_delta_mean": float(delta.mean()),
                "paired_delta_sample_sd": float(delta.std(ddof=1)),
                "paired_raw_delta_mean": float(raw.mean()),
                "paired_raw_delta_sample_sd": float(raw.std(ddof=1)),
                "relative_delta_defined_seeds": int(np.isfinite(delta).sum()),
                "delta_of_seed_means": mean_delta,
                "full_better_seeds": int((raw > 0).sum()),
                "candidate_better_seeds": int((raw < 0).sum()),
                "equal_seeds": int((raw == 0).sum()),
                "split": "val",
            }
        )
    return pd.DataFrame(paired), pd.DataFrame(aggregate)


def subject_tables(
    sources: Mapping[int, tuple[pd.DataFrame, pd.DataFrame]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    flags = {
        ERRORS[0]: "whole_rr_target_eligible",
        ERRORS[1]: "local_rr_target_eligible",
        PCC: "joint_target_eligible",
    }
    for seed in SEEDS:
        if seed not in sources:
            raise ValueError("E5 subject 汇总 seed 不完整")
        full, candidate = sources[seed]
        full = full.sort_values("dataset_row_id").reset_index(drop=True)
        candidate = candidate.sort_values("dataset_row_id").reset_index(drop=True)
        if not np.array_equal(full.dataset_row_id, candidate.dataset_row_id):
            raise ValueError("E5 subject 汇总 row identity 漂移")
        for samp_id in sorted(full.samp_id.unique()):
            mask = full.samp_id.eq(samp_id).to_numpy()
            for metric in PRIMARY:
                eligible = mask.copy()
                if metric in flags:
                    a = full[flags[metric]].to_numpy(dtype=bool)
                    b = candidate[flags[metric]].to_numpy(dtype=bool)
                    if not np.array_equal(a, b):
                        raise ValueError("E5 subject 汇总 eligibility 漂移")
                    eligible &= a
                before = float(full.loc[eligible, metric].mean())
                after = float(candidate.loc[eligible, metric].mean())
                if not np.isfinite(before) or not np.isfinite(after):
                    raise FloatingPointError("E5 subject 指标非有限")
                raw = before - after if metric == PCC else after - before
                relative = raw if metric == PCC else (100 * raw / before if before != 0 else np.nan)
                rows.append(
                    {
                        "seed": seed,
                        "samp_id": samp_id,
                        "metric": metric,
                        "n_windows": int(eligible.sum()),
                        "W0_FULL": before,
                        ARM: after,
                        "raw_delta": raw,
                        "delta": relative,
                        "unit": "absolute_drop" if metric == PCC else "relative_percent",
                    }
                )
    paired = pd.DataFrame(rows)
    macro_rows: list[dict[str, Any]] = []
    for (seed, metric), group in paired.groupby(["seed", "metric"], sort=False):
        macro_rows.append(
            {
                "seed": seed,
                "metric": metric,
                "subject_count": int(len(group)),
                "macro_raw_delta": float(group.raw_delta.mean()),
                "macro_delta": float(group.delta.mean()),
                "full_better_subjects": int(group.raw_delta.gt(0).sum()),
                "candidate_better_subjects": int(group.raw_delta.lt(0).sum()),
                "equal_subjects": int(group.raw_delta.eq(0).sum()),
                "unit": group.unit.iloc[0],
            }
        )
    return paired, pd.DataFrame(macro_rows)


def summarize(runs: list[Path]) -> Path:
    lock, lock_hash = load_lock()
    parent = ROOT / OUTPUT / "summary"
    with phase_guard(parent, lock_hash):
        if len(runs) != 3 or len({path.resolve() for path in runs}) != 3:
            raise ValueError("E5 summary 需要三个不同的完成 attempt")
        source_runs: dict[int, Any] = {}
        candidates: list[pd.DataFrame] = []
        subject_sources: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
        for output in runs:
            manifest = verify_attempt(output, phase="formal", lock_hash=lock_hash)
            receipt = json.loads((output / "formal_receipt.json").read_text())
            seed = int(receipt["seed"])
            if seed not in SEEDS or seed in source_runs or manifest["seed"] != seed:
                raise ValueError("E5 summary seed 重复或越界")
            run_dir = (output / receipt["run_dir"]).resolve()
            if not run_dir.is_relative_to(output.resolve()):
                raise ValueError("E5 training run_dir 越界")
            cfg = OmegaConf.load(run_dir / "config.yaml")
            baseline = OmegaConf.create(lock["baselines"][str(seed)])
            validate_config(
                cfg,
                baseline,
                output_root=output.resolve() / "training",
                device=str(cfg.training.device),
            )
            checked = validate_run(run_dir, cfg, pd.read_csv(output / "val_rows.csv"))
            if any(receipt.get(key) != value for key, value in checked.items()):
                raise ValueError("E5 formal receipt 与重新核对结果不一致")
            candidate_metrics = pd.read_csv(run_dir / "metrics.csv")
            validate_anchor_rows(candidate_metrics, lock, seed)
            summary = pd.read_csv(run_dir / "metrics_summary.csv")
            summary.insert(0, "quality_acceptance_passed", checked["quality_acceptance_passed"])
            summary.insert(0, "selected_epoch", receipt["selected_epoch"])
            summary.insert(0, "source_sha256", sha256_file(run_dir / "metrics.csv"))
            summary.insert(0, "split", "val")
            summary.insert(0, "seed", seed)
            summary.insert(0, "arm", ARM)
            candidates.append(summary)
            entry = next(item for item in lock["w0_entries"] if int(item["seed"]) == seed)
            full_metrics = pd.read_csv(ROOT / entry["run_dir"] / "metrics.csv")
            subject_sources[seed] = (full_metrics, candidate_metrics)
            source_runs[seed] = {
                "path": str(output.resolve()),
                "manifest": identity(output / "manifest.json"),
                "selected_epoch": receipt["selected_epoch"],
            }
        if set(source_runs) != set(SEEDS):
            raise ValueError("E5 summary seed 矩阵不完整")
        full_rows: list[pd.DataFrame] = []
        for entry in lock["w0_entries"]:
            path = ROOT / entry["run_dir"] / "metrics_summary.csv"
            verify(path, lock["source_files"][str(path.relative_to(ROOT))])
            frame = pd.read_csv(path)
            frame.insert(0, "quality_acceptance_passed", True)
            frame.insert(0, "selected_epoch", entry["selected_epoch"])
            frame.insert(0, "source_sha256", sha256_file(path))
            frame.insert(0, "split", "val")
            frame.insert(0, "seed", entry["seed"])
            frame.insert(0, "arm", "W0_FULL")
            full_rows.append(frame)
        combined = pd.concat([*full_rows, *candidates], ignore_index=True)
        for metric in PRIMARY:
            for seed in SEEDS:
                full_n = int(
                    combined.loc[
                        combined.arm.eq("W0_FULL") & combined.seed.eq(seed),
                        metric + "_n",
                    ].iloc[0]
                )
                candidate_n = int(
                    combined.loc[
                        combined.arm.eq(ARM) & combined.seed.eq(seed),
                        metric + "_n",
                    ].iloc[0]
                )
                if full_n <= 0 or candidate_n != full_n:
                    raise ValueError(f"E5 主指标分母与同 seed W0 不一致: {seed}/{metric}")
        paired, aggregate = paired_tables(combined)
        subject_paired, subject_macro = subject_tables(subject_sources)
        provenance = combined.set_index(["arm", "seed"])
        for side, arm in (("full", "W0_FULL"), ("candidate", ARM)):
            for field in ("selected_epoch", "source_sha256"):
                paired[f"{side}_{field}"] = [
                    provenance.loc[(arm, int(seed)), field] for seed in paired.seed
                ]
            paired[f"{side}_n"] = [
                int(provenance.loc[(arm, int(row.seed)), row.metric + "_n"])
                for row in paired.itertuples()
            ]
        aggregate["n_windows_per_seed"] = COUNTS["val"]
        for receipt in parent.glob("*/freeze_receipt.json"):
            manifest = json.loads((receipt.parent / "manifest.json").read_text())
            if manifest["implementation_lock_sha256"] == lock_hash:
                raise FileExistsError(f"E5 该实现身份已有完整汇总: {receipt.parent}")
        with attempt(parent, lock_hash, "summary") as output:
            combined.to_csv(output / "seed_metrics.csv", index=False)
            paired.to_csv(output / "paired_seed_delta.csv", index=False, na_rep="NA")
            aggregate.to_csv(output / "three_seed_comparison.csv", index=False, na_rep="NA")
            subject_paired.to_csv(output / "paired_seed_subject_delta.csv", index=False, na_rep="NA")
            subject_macro.to_csv(output / "subject_macro_by_seed.csv", index=False, na_rep="NA")
            write_json(output / "parameter_compute_report.json", parameter_compute_report())
            write_json(
                output / "source_receipt.json",
                {
                    "source_runs": source_runs,
                    "w0_sources": lock["w0_entries"],
                    "implementation_lock_sha256": lock_hash,
                },
            )
            write_json(
                output / "summary_receipt.json",
                {
                    "protocol": PROTOCOL,
                    "seeds": list(SEEDS),
                    "source_runs": source_runs,
                    "w0_sources": lock["w0_entries"],
                    "delta_definition": "positive means W0_FULL better; error=(candidate-full)/full*100; PCC=full-candidate",
                    "subject_macro_role": "descriptive robustness; does not replace frozen sample-direct mean",
                },
            )
        return output


def parameter_compute_report() -> dict[str, Any]:
    return {
        "frontend_parameters_w0": W0_FRONTEND_PARAMETERS,
        "frontend_parameters_e5": FRONTEND_PARAMETERS,
        "model_parameters_w0": W0_MODEL_PARAMETERS,
        "model_parameters_e5": MODEL_PARAMETERS,
        "model_parameter_delta": MODEL_PARAMETERS - W0_MODEL_PARAMETERS,
        "fixed_taps": 511,
        "frontend_covered_macs_w0": 3_710_080,
        "frontend_covered_macs_e5": 3_684_600,
        "e5_macs": {
            "fixed_fir_100_to_10": 919_800,
            "conv_1_to_96_k11": 1_900_800,
            "depthwise_96_k5": 864_000,
        },
        "coverage": "listed FIR/Conv1d kernel products; one MAC per output kernel product",
        "excluded": [
            "SiLU",
            "memory movement",
            "W branch",
            "Mamba/selective scan",
            "FFT/task projection",
            "backward",
        ],
        "whole_model_flops": None,
        "whole_model_flops_status": "not reported: incomplete operator coverage",
    }
