"""W0 三因素结构对照的配置、生命周期与冻结分析合同。"""

from __future__ import annotations

import fcntl
import hashlib
import json
import subprocess
import sys
import traceback
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

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
from resp_train.paper_evidence.w0_structural_factorial_v1_model import (
    ARMS,
    ARM_SPECS,
    CONV20_CONTRACT,
    PROTOCOL,
    REFERENCE_ARM,
    build_w0_structural_factorial_model,
    model_contract,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path("/mnt/disk_code/marques/resp_reconstruction")
SPEC_PATH = Path("configs/w0_structural_factorial_v1/experiment.yaml")
PROTOCOL_PATH = Path("docs/experiments/w0_structural_factorial_v1_protocol_20260923.md")
MODEL_PATH = Path("resp_train/paper_evidence/w0_structural_factorial_v1_model.py")
CONTROL_PATH = Path("resp_train/paper_evidence/w0_structural_factorial_v1.py")
SCRIPT_PATH = Path("scripts/run_w0_structural_factorial_v1.py")
TEST_PATH = Path("tests/test_w0_structural_factorial_v1.py")
P1_RECEIPT_PATH = Path("docs/experiments/w0_structural_factorial_v1_p1_implementation_receipt_20260924.json")
PREVIOUS_LOCK_PATH = Path("docs/experiments/w0_structural_factorial_v1_implementation_lock_20260923.json")
PREVIOUS_LOCK_SHA256 = "9e5d4a53499cec23cad9d6b4cc6d6eaca0b590cef4de5111c0d5b2d460b98b31"
LOCK_PATH = Path("docs/experiments/w0_structural_factorial_v1_implementation_lock_r2_20260924.json")
ENGINEERING_PATH = Path("resp_train/paper_evidence/w0_structural_factorial_v1_engineering.py")
W0_CONFIG_PATH = Path("configs/crd_tf_v1/crd_tf102_w_formal.yaml")
OUTPUT_ROOT = Path("runs/w0_structural_factorial_v1_es30p15")
W0_SOURCE_LOCK = Path("docs/experiments/e4_w0_scale_aggregation_implementation_lock_20260917.json")
W0_SOURCE_LOCK_SHA256 = "464e073dbd5707a30575d606dec2a84dcd89a161945e537c463b214a15c2b493"

SEEDS = (20260811, 20260812, 20260813)
COUNTS = {"train": 10_141, "val": 2_675}
SAMP_IDS = {"train": 32, "val": 7}
EPOCHS = 80
UPDATES_PER_EPOCH = 80
PLANNED_UPDATES = 6_400
EARLY_STOP_MIN_EPOCH = 30
EARLY_STOP_PATIENCE = 15
EARLY_STOP_MIN_DELTA = 0.0

ERRORS = (
    "whole_rr_abs_error_bpm",
    "local_rr_mae_bpm",
    "envelope_trajectory_mae",
    "global_envelope_modulation_error",
)
PCC = "lag_aware_signed_pcc"
PRIMARY = (*ERRORS, PCC)
ELIGIBILITY = {
    ERRORS[0]: "whole_rr_target_eligible",
    ERRORS[1]: "local_rr_target_eligible",
    PCC: "joint_target_eligible",
}
ERROR_RELATIVE_TOLERANCE = 0.005
PCC_ABSOLUTE_TOLERANCE = 0.002

_TOP_KEYS = {"schema_version", "protocol", "output_root", "source", "arms", "matrix", "analysis"}
_SOURCE_KEYS = {"w0_implementation_lock", "w0_implementation_lock_sha256", "w0_config", "temporal_stem"}
_MATRIX = {
    "seeds": list(SEEDS),
    "train_windows": COUNTS["train"],
    "validation_windows": COUNTS["val"],
    "train_samp_ids": SAMP_IDS["train"],
    "validation_samp_ids": SAMP_IDS["val"],
    "epochs": EPOCHS,
    "updates_per_epoch": UPDATES_PER_EPOCH,
    "planned_updates": PLANNED_UPDATES,
    "early_stopping": {
        "enabled": True,
        "min_epoch": EARLY_STOP_MIN_EPOCH,
        "patience": EARLY_STOP_PATIENCE,
        "min_delta": EARLY_STOP_MIN_DELTA,
    },
    "physical_batch": 128,
    "gradient_accumulation_steps": 1,
    "amp_dtype": "bfloat16",
    "checkpoint_selector": "full_validation_local_rr_strict_minimum_earliest_tie",
    "split": "train_validation",
}
_ANALYSIS = {
    "primary_metrics": list(PRIMARY),
    "error_relative_tolerance": ERROR_RELATIVE_TOLERANCE,
    "pcc_absolute_tolerance": PCC_ABSOLUTE_TOLERANCE,
    "local_rr_thresholds_bpm": [2.0, 5.0],
    "seed_sd_ddof": 1,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    return {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_identity(path: Path, expected: Mapping[str, Any]) -> None:
    normalized = {key: expected[key] for key in ("size_bytes", "sha256")}
    if not path.is_file() or identity(path) != normalized:
        raise RuntimeError(f"文件身份漂移: {path}")


def git_state(root: Path = ROOT) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_porcelain": run("status", "--porcelain"),
    }


def _arm_spec_payload() -> dict[str, Any]:
    return {
        arm: {
            "frontend": spec.frontend,
            "cwt_temporal": "tm3" if spec.cwt_temporal else "tm0",
            "refinement": "ref2" if spec.refinement else "ref0",
            "factors": list(spec.factors),
            "trainable_parameters": spec.trainable_parameters,
            "factor_covered_macs": spec.factor_covered_macs,
        }
        for arm, spec in ARM_SPECS.items()
    }


def load_experiment_spec(path: Path | None = None) -> DictConfig:
    path = ROOT / SPEC_PATH if path is None else Path(path)
    cfg = OmegaConf.load(path)
    OmegaConf.resolve(cfg)
    source = OmegaConf.to_container(cfg.source, resolve=True)
    if (
        set(cfg.keys()) != _TOP_KEYS
        or int(cfg.schema_version) != 1
        or str(cfg.protocol) != PROTOCOL
        or str(cfg.output_root) != str(OUTPUT_ROOT)
        or set(source) != _SOURCE_KEYS
        or str(cfg.source.w0_implementation_lock) != str(W0_SOURCE_LOCK)
        or str(cfg.source.w0_implementation_lock_sha256) != W0_SOURCE_LOCK_SHA256
        or str(cfg.source.w0_config) != str(W0_CONFIG_PATH)
        or str(cfg.source.temporal_stem) != "resp_train/temporal/blocks.py"
        or OmegaConf.to_container(cfg.arms, resolve=True) != _arm_spec_payload()
        or OmegaConf.to_container(cfg.matrix, resolve=True) != _MATRIX
        or OmegaConf.to_container(cfg.analysis, resolve=True) != _ANALYSIS
    ):
        raise ValueError("W0 结构因子 spec 科学合同漂移")
    return cfg


def load_w0_baseline(seed: int, root: Path = ROOT) -> DictConfig:
    if int(seed) not in SEEDS:
        raise ValueError("seed 必须属于冻结三 seed 矩阵")
    return load_crd_config(
        root / W0_CONFIG_PATH,
        overrides=[f"training.seed={int(seed)}", f"model.initialization_seed={int(seed)}"],
    )


def derived_config(
    baseline: DictConfig,
    *,
    arm: str,
    output_root: Path,
    device: str,
) -> DictConfig:
    if arm not in ARM_SPECS:
        raise ValueError(f"未知结构因子 arm={arm!r}")
    cfg = OmegaConf.create(OmegaConf.to_container(baseline, resolve=True))
    cfg.protocol.name = PROTOCOL
    cfg.protocol.stage = "w0_structural_factorial_v1"
    cfg.protocol.execution_gate = "sfv1_formal"
    cfg.model.w0_structural_factorial_v1 = model_contract(arm)
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
    arm: str,
    output_root: Path,
    device: str,
) -> None:
    expected = derived_config(baseline, arm=arm, output_root=output_root, device=device)
    if OmegaConf.to_container(cfg, resolve=True) != OmegaConf.to_container(expected, resolve=True):
        raise ValueError("结构因子配置与同 seed W0 合同不一致")
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
        or baseline.data.max_test_windows is not None
        or float(baseline.loss.sync_weight) != 1.0
        or float(baseline.loss.effort_weight) != 0.25
    ):
        raise ValueError("W0 来源配置不符合冻结矩阵")


def critical_paths() -> tuple[Path, ...]:
    paths = (
        SPEC_PATH,
        PROTOCOL_PATH,
        MODEL_PATH,
        CONTROL_PATH,
        ENGINEERING_PATH,
        SCRIPT_PATH,
        TEST_PATH,
        P1_RECEIPT_PATH,
        W0_CONFIG_PATH,
        Path("resp_train/crd/frontends.py"),
        Path("resp_train/crd/model.py"),
        Path("resp_train/crd/tf_v1_model.py"),
        Path("resp_train/crd/blocks.py"),
        Path("resp_train/crd/initialization.py"),
        Path("resp_train/crd/experiment.py"),
        Path("resp_train/crd/training.py"),
        Path("resp_train/temporal/blocks.py"),
        Path("resp_train/losses/task.py"),
        Path("resp_train/metrics/task.py"),
    )
    missing = [str(path) for path in paths if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"结构因子实现缺少文件: {missing}")
    return paths


def prepare_lock(root: Path = ROOT) -> Path:
    """在干净提交上复核全部来源身份并排他生成实现锁。"""

    spec = load_experiment_spec(root / SPEC_PATH)
    destination = root / LOCK_PATH
    if destination.exists():
        raise FileExistsError(f"implementation lock 已存在: {destination}")
    state = git_state(root)
    if state["status_porcelain"]:
        raise RuntimeError("implementation lock 要求工作树干净")
    source_path = root / W0_SOURCE_LOCK
    if sha256_file(source_path) != W0_SOURCE_LOCK_SHA256:
        raise ValueError("W0 来源锁身份漂移")
    previous_lock_path = root / PREVIOUS_LOCK_PATH
    if sha256_file(previous_lock_path) != PREVIOUS_LOCK_SHA256:
        raise ValueError("P1 implementation lock 身份漂移")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if (
        tuple(int(value) for value in source.get("seeds", ())) != SEEDS
        or source.get("counts") != COUNTS
        or set(source.get("baselines", {})) != {str(seed) for seed in SEEDS}
    ):
        raise ValueError("W0 来源锁 seed/count/baseline 合同漂移")

    source_files: dict[str, dict[str, Any]] = {str(W0_SOURCE_LOCK): identity(source_path)}
    for entry in source["w0_entries"]:
        seed = int(entry["seed"])
        if seed not in SEEDS:
            raise ValueError("W0 来源锁包含矩阵外 seed")
        for filename in (
            "checkpoint_best_local_rr.pt",
            "config.yaml",
            "run_manifest.json",
            "metrics.csv",
            "metrics_summary.csv",
        ):
            relative = str(Path(entry["run_dir"]) / filename)
            expected = source["source_files"].get(relative)
            if expected is None:
                raise ValueError(f"W0 来源锁缺少文件身份: {relative}")
            verify_identity(SOURCE_ROOT / relative, expected)
            source_files[relative] = {key: expected[key] for key in ("size_bytes", "sha256")}

    cache = source["cache_lock"]
    cache_entries = (cache["manifest"], cache["train_w"], cache["val_w"], cache["frequency_file"])
    for entry in cache_entries:
        expected = {
            "size_bytes": int(entry["size_bytes"]),
            "sha256": str(entry.get("sha256", entry.get("file_sha256"))),
        }
        verify_identity(SOURCE_ROOT / entry["path"], expected)
        source_files[entry["path"]] = expected
    for split in COUNTS:
        relative = str(Path(cache["root"]) / f"{split}_row_ids.npy")
        expected = source["source_files"].get(relative)
        if expected is None:
            raise ValueError(f"W0 来源锁缺少 row identity: {relative}")
        verify_identity(SOURCE_ROOT / relative, expected)
        source_files[relative] = {key: expected[key] for key in ("size_bytes", "sha256")}

    cache_manifest = json.loads((SOURCE_ROOT / cache["manifest"]["path"]).read_text(encoding="utf-8"))
    baselines: dict[str, Any] = {}
    templates: dict[str, Any] = {}
    for seed in SEEDS:
        baseline = OmegaConf.create(source["baselines"][str(seed)])
        baselines[str(seed)] = OmegaConf.to_container(baseline, resolve=True)
        templates[str(seed)] = {}
        for arm in ARMS:
            output = SOURCE_ROOT / OUTPUT_ROOT / "formal" / arm / f"seed_{seed}"
            candidate = derived_config(baseline, arm=arm, output_root=output, device="cuda:0")
            validate_config(candidate, baseline, arm=arm, output_root=output, device="cuda:0")
            templates[str(seed)][arm] = OmegaConf.to_container(candidate, resolve=True)

    lock = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "spec": identity(root / SPEC_PATH),
        "arms": list(ARMS),
        "arm_contracts": _arm_spec_payload(),
        "seeds": list(SEEDS),
        "counts": COUNTS,
        "samp_ids": SAMP_IDS,
        "epochs": EPOCHS,
        "updates_per_epoch": UPDATES_PER_EPOCH,
        "planned_updates": PLANNED_UPDATES,
        "early_stopping": _MATRIX["early_stopping"],
        "analysis": _ANALYSIS,
        "w0_entries": source["w0_entries"],
        "baselines": baselines,
        "resolved_templates": templates,
        "cache_lock": cache,
        "dataset_index": {
            "path": cache_manifest["dataset_index"],
            "sha256": cache_manifest["dataset_index_sha256"],
        },
        "source_repository_root": str(SOURCE_ROOT),
        "artifact_root": str(SOURCE_ROOT / OUTPUT_ROOT),
        "source_files": source_files,
        "code_files": {str(path): identity(root / path) for path in critical_paths()},
        "previous_implementation_lock": {
            "path": str(PREVIOUS_LOCK_PATH),
            **identity(previous_lock_path),
        },
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "preparation_git": state,
        "status": "implementation_locked_p2_authorized",
    }
    write_json(destination, lock)
    return destination


def load_lock(root: Path = ROOT) -> tuple[dict[str, Any], str]:
    path = root / LOCK_PATH
    if not path.is_file():
        raise FileNotFoundError(f"implementation lock 尚未建立: {path}")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if (
        lock.get("schema_version") != 1
        or lock.get("protocol") != PROTOCOL
        or tuple(lock.get("arms", ())) != ARMS
        or tuple(lock.get("seeds", ())) != SEEDS
        or lock.get("counts") != COUNTS
        or lock.get("samp_ids") != SAMP_IDS
        or lock.get("epochs") != EPOCHS
        or lock.get("updates_per_epoch") != UPDATES_PER_EPOCH
        or lock.get("planned_updates") != PLANNED_UPDATES
        or lock.get("early_stopping") != _MATRIX["early_stopping"]
        or lock.get("analysis") != _ANALYSIS
        or lock.get("arm_contracts") != _arm_spec_payload()
        or lock.get("status") != "implementation_locked_p2_authorized"
        or lock.get("source_repository_root") != str(SOURCE_ROOT)
        or lock.get("artifact_root") != str(SOURCE_ROOT / OUTPUT_ROOT)
    ):
        raise ValueError("implementation lock 科学合同漂移")
    verify_identity(root / SPEC_PATH, lock["spec"])
    previous = lock.get("previous_implementation_lock", {})
    if previous.get("path") != str(PREVIOUS_LOCK_PATH):
        raise ValueError("P1 implementation lock 链接漂移")
    verify_identity(root / PREVIOUS_LOCK_PATH, previous)
    for relative, expected in lock["code_files"].items():
        verify_identity(root / relative, expected)
    for seed in SEEDS:
        baseline = OmegaConf.create(lock["baselines"][str(seed)])
        for arm in ARMS:
            template = OmegaConf.create(lock["resolved_templates"][str(seed)][arm])
            validate_config(
                template,
                baseline,
                arm=arm,
                output_root=SOURCE_ROOT / OUTPUT_ROOT / "formal" / arm / f"seed_{seed}",
                device="cuda:0",
            )
    return lock, sha256_file(path)


class StructuralFactorialExperiment(CRDExperiment):
    """复用原生 CRD trainer，仅替换模型工厂并加强 validation 身份检查。"""

    def __init__(self, cfg: DictConfig, validation_rows: pd.DataFrame, arm: str):
        self.arm = arm
        self.task_name = arm
        self.validation_rows = validation_rows
        super().__init__(cfg)

    def _build_model(self):
        return build_w0_structural_factorial_model(self.cfg)

    def _evaluate_model(self, model, loader, **kwargs):
        frame = super()._evaluate_model(model, loader, **kwargs)
        validate_metrics(frame, self.validation_rows)
        frame.insert(0, "arm", self.arm)
        frame.insert(0, "seed", int(self.cfg.training.seed))
        return frame


def validate_metrics(metrics: pd.DataFrame, rows: pd.DataFrame) -> dict[str, int]:
    if len(metrics) != len(rows) or metrics.dataset_row_id.duplicated().any() or rows.dataset_row_id.duplicated().any():
        raise ValueError("validation metrics row identity 不完整")
    for key in ("dataset_row_id", "samp_id", "split"):
        if not np.array_equal(metrics[key].to_numpy(), rows[key].to_numpy()):
            raise ValueError(f"validation metrics identity/order 不一致: {key}")
    for metric in PRIMARY:
        eligible = np.ones(len(metrics), dtype=bool)
        if metric in ELIGIBILITY:
            flag = metrics[ELIGIBILITY[metric]]
            if flag.isna().any() or not flag.isin([True, False]).all():
                raise ValueError(f"eligibility 非法: {ELIGIBILITY[metric]}")
            eligible = flag.to_numpy(dtype=bool)
        values = pd.to_numeric(metrics[metric], errors="coerce").to_numpy(dtype=float)
        if np.isinf(values).any() or not np.array_equal(np.isfinite(values), eligible):
            raise FloatingPointError(f"指标 finite/eligibility 不一致: {metric}")
    degeneracy: dict[str, int] = {}
    for key in ("joint_prediction_degenerate", "envelope_spearman_prediction_degenerate"):
        values = metrics[key]
        if values.isna().any() or not values.isin([True, False]).all():
            raise ValueError(f"prediction degeneracy flag 非法: {key}")
        degeneracy[key] = int(values.astype(bool).sum())
    return degeneracy


def validate_history(history: pd.DataFrame, cfg: DictConfig) -> int:
    completed_epochs = len(history)
    if (
        completed_epochs < EARLY_STOP_MIN_EPOCH
        or completed_epochs > EPOCHS
        or not np.array_equal(history.epoch, np.arange(1, completed_epochs + 1))
        or not np.isfinite(history.select_dtypes(include=np.number).to_numpy()).all()
    ):
        raise ValueError("history epoch/finite 合同错误")
    expected_updates = np.arange(1, completed_epochs + 1) * UPDATES_PER_EPOCH
    if not np.array_equal(history.optimizer_update, expected_updates):
        raise ValueError("history optimizer update 数错误")
    if not np.allclose(
        history.train_loss_total,
        history.train_loss_sync + 0.25 * history.train_loss_effort,
        atol=1e-12,
        rtol=0,
    ):
        raise ValueError("history total loss 与完整目标不一致")
    for row in history.itertuples():
        for column, update in (
            ("first_learning_rate", (row.epoch - 1) * UPDATES_PER_EPOCH),
            ("last_learning_rate", row.epoch * UPDATES_PER_EPOCH - 1),
        ):
            expected = crd_learning_rate(
                update,
                total_updates=PLANNED_UPDATES,
                max_learning_rate=float(cfg.training.max_learning_rate),
                min_learning_rate=float(cfg.training.min_learning_rate),
                warmup_fraction=float(cfg.training.warmup_fraction),
            )
            if not np.isclose(getattr(row, column), expected, atol=1e-15, rtol=0):
                raise ValueError("learning-rate schedule 漂移")
    required = {
        "early_stopping_improved",
        "early_stopping_wait",
        "early_stopping_triggered",
        "early_stopping_min_epoch",
    }
    if not required.issubset(history.columns):
        raise ValueError("history 缺少 early stopping 字段")
    best = float("inf")
    wait = 0
    triggered: list[int] = []
    for row in history.itertuples():
        improved, wait = _early_stopping_step(
            value=float(row.val_local_rr_mae),
            best=best,
            epochs_without_improvement=wait,
            min_delta=EARLY_STOP_MIN_DELTA,
        )
        if improved:
            best = float(row.val_local_rr_mae)
        stop = _early_stopping_should_stop(
            epoch=int(row.epoch),
            min_epoch=EARLY_STOP_MIN_EPOCH,
            epochs_without_improvement=wait,
            patience=EARLY_STOP_PATIENCE,
        )
        if (
            int(row.early_stopping_improved) != int(improved)
            or int(row.early_stopping_wait) != wait
            or int(row.early_stopping_triggered) != int(stop)
            or int(row.early_stopping_min_epoch) != EARLY_STOP_MIN_EPOCH
        ):
            raise ValueError("early stopping 轨迹漂移")
        triggered.append(int(stop))
    if any(triggered[:-1]) or (completed_epochs < EPOCHS and triggered[-1] != 1):
        raise ValueError("history 停止位置不符合冻结合同")
    return int(history.iloc[int(np.argmin(history.val_local_rr_mae.to_numpy()))].epoch)


def _require_complete_seed_frame(frame: pd.DataFrame, group_columns: Sequence[str] = ("seed",)) -> None:
    required = {*group_columns, "arm", *PRIMARY}
    if not required.issubset(frame.columns):
        raise ValueError(f"seed frame 缺少字段: {sorted(required - set(frame.columns))}")
    expected_seeds = set(SEEDS)
    if set(pd.to_numeric(frame.seed, errors="raise").astype(int)) != expected_seeds:
        raise ValueError("seed frame 必须包含冻结三 seed")
    for _, group in frame.groupby(list(group_columns), sort=False):
        if len(group) != len(ARMS) or set(group.arm) != set(ARMS) or group.arm.duplicated().any():
            raise ValueError("每个分析分组必须包含八个唯一 arm")
        if not np.isfinite(group[list(PRIMARY)].to_numpy(dtype=float)).all():
            raise FloatingPointError("seed frame 主指标非有限")


def arm_primary_summary(seed_frame: pd.DataFrame) -> pd.DataFrame:
    _require_complete_seed_frame(seed_frame)
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        group = seed_frame.loc[seed_frame.arm.eq(arm)].set_index("seed").loc[list(SEEDS)]
        for metric in PRIMARY:
            values = group[metric].to_numpy(dtype=float)
            rows.append(
                {
                    "arm": arm,
                    "metric": metric,
                    "seed_mean": float(values.mean()),
                    "seed_sample_sd": float(values.std(ddof=1)),
                    "seed_count": len(values),
                    "direction": "maximize" if metric == PCC else "minimize",
                }
            )
    return pd.DataFrame(rows)


def _values_by_factor(group: pd.DataFrame, metric: str) -> dict[tuple[int, int, int], float]:
    indexed = group.set_index("arm")
    return {
        spec.factors: float(indexed.loc[arm, metric])
        for arm, spec in ARM_SPECS.items()
    }


def conditional_effects_by_seed(
    seed_frame: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("seed",),
) -> pd.DataFrame:
    _require_complete_seed_frame(seed_frame, group_columns)
    rows: list[dict[str, Any]] = []
    labels = ("A", "B", "C")
    for keys, group in seed_frame.groupby(list(group_columns), sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        identity_fields = dict(zip(group_columns, keys, strict=True))
        for metric in PRIMARY:
            values = _values_by_factor(group, metric)
            reference = values[(0, 1, 1)]
            for factor, label in enumerate(labels):
                others = [index for index in range(3) if index != factor]
                for first in (0, 1):
                    for second in (0, 1):
                        low = [0, 0, 0]
                        high = [0, 0, 0]
                        low[others[0]] = high[others[0]] = first
                        low[others[1]] = high[others[1]] = second
                        high[factor] = 1
                        raw = values[tuple(high)] - values[tuple(low)]
                        rows.append(
                            {
                                **identity_fields,
                                "metric": metric,
                                "factor": label,
                                f"{labels[others[0]]}_level": first,
                                f"{labels[others[1]]}_level": second,
                                "level1_minus_level0": raw,
                                "oriented_benefit": raw if metric == PCC else -raw,
                                "reference_normalized_benefit": (
                                    raw if metric == PCC else (-raw / reference if reference != 0 else np.nan)
                                ),
                            }
                        )
    return pd.DataFrame(rows)


def factorial_effects_by_seed(
    seed_frame: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("seed",),
) -> pd.DataFrame:
    _require_complete_seed_frame(seed_frame, group_columns)
    rows: list[dict[str, Any]] = []
    for keys, group in seed_frame.groupby(list(group_columns), sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        identity_fields = dict(zip(group_columns, keys, strict=True))
        for metric in PRIMARY:
            y = _values_by_factor(group, metric)
            effects = {
                "A": np.mean([y[(1, b, c)] - y[(0, b, c)] for b in (0, 1) for c in (0, 1)]),
                "B": np.mean([y[(a, 1, c)] - y[(a, 0, c)] for a in (0, 1) for c in (0, 1)]),
                "C": np.mean([y[(a, b, 1)] - y[(a, b, 0)] for a in (0, 1) for b in (0, 1)]),
                "AB": np.mean(
                    [(y[(1, 1, c)] - y[(1, 0, c)]) - (y[(0, 1, c)] - y[(0, 0, c)]) for c in (0, 1)]
                ),
                "AC": np.mean(
                    [(y[(1, b, 1)] - y[(1, b, 0)]) - (y[(0, b, 1)] - y[(0, b, 0)]) for b in (0, 1)]
                ),
                "BC": np.mean(
                    [(y[(a, 1, 1)] - y[(a, 1, 0)]) - (y[(a, 0, 1)] - y[(a, 0, 0)]) for a in (0, 1)]
                ),
                "ABC": (
                    y[(1, 1, 1)]
                    - y[(1, 1, 0)]
                    - y[(1, 0, 1)]
                    - y[(0, 1, 1)]
                    + y[(1, 0, 0)]
                    + y[(0, 1, 0)]
                    + y[(0, 0, 1)]
                    - y[(0, 0, 0)]
                ),
            }
            reference = y[(0, 1, 1)]
            for effect, raw in effects.items():
                rows.append(
                    {
                        **identity_fields,
                        "metric": metric,
                        "effect": effect,
                        "raw_effect": float(raw),
                        "oriented_benefit": float(raw if metric == PCC else -raw),
                        "reference_normalized_benefit": float(
                            raw if metric == PCC else (-raw / reference if reference != 0 else np.nan)
                        ),
                    }
                )
    return pd.DataFrame(rows)


def factorial_effects_across_seed(effects: pd.DataFrame) -> pd.DataFrame:
    required = {"seed", "metric", "effect", "raw_effect", "oriented_benefit", "reference_normalized_benefit"}
    if not required.issubset(effects.columns) or set(effects.seed) != set(SEEDS):
        raise ValueError("factorial effect seed 矩阵不完整")
    rows: list[dict[str, Any]] = []
    for (metric, effect), group in effects.groupby(["metric", "effect"], sort=False):
        group = group.set_index("seed").loc[list(SEEDS)]
        row: dict[str, Any] = {"metric": metric, "effect": effect, "seed_count": len(SEEDS)}
        for column in ("raw_effect", "oriented_benefit", "reference_normalized_benefit"):
            values = group[column].to_numpy(dtype=float)
            row[column + "_mean"] = float(values.mean())
            row[column + "_sample_sd"] = float(values.std(ddof=1))
            row[column + "_positive_seeds"] = int((values > 0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _metric_mask(frame: pd.DataFrame, metric: str) -> np.ndarray:
    mask = np.ones(len(frame), dtype=bool)
    if metric in ELIGIBILITY:
        mask &= frame[ELIGIBILITY[metric]].to_numpy(dtype=bool)
    return mask


def seed_primary_metrics(window_metrics: pd.DataFrame, *, expected_rows: int = COUNTS["val"]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    expected_pairs = {(arm, seed) for arm in ARMS for seed in SEEDS}
    observed_pairs = set(window_metrics[["arm", "seed"]].drop_duplicates().itertuples(index=False, name=None))
    if observed_pairs != expected_pairs:
        raise ValueError("window metrics arm/seed 矩阵不完整")
    for (arm, seed), group in window_metrics.groupby(["arm", "seed"], sort=False):
        if len(group) != expected_rows or group.dataset_row_id.duplicated().any():
            raise ValueError("window metrics row identity 不完整")
        row: dict[str, Any] = {"arm": arm, "seed": int(seed)}
        for metric in PRIMARY:
            mask = _metric_mask(group, metric)
            values = group.loc[mask, metric].to_numpy(dtype=float)
            if not len(values) or not np.isfinite(values).all():
                raise FloatingPointError(f"主指标无有效有限值: {arm}/{seed}/{metric}")
            row[metric] = float(values.mean())
            row[metric + "_n"] = int(len(values))
        rows.append(row)
    result = pd.DataFrame(rows)
    _require_complete_seed_frame(result)
    return result


def local_rr_tail_summary(window_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (arm, seed), group in window_metrics.groupby(["arm", "seed"], sort=False):
        values = group.loc[group.local_rr_target_eligible.astype(bool), ERRORS[1]].to_numpy(dtype=float)
        if not len(values) or not np.isfinite(values).all():
            raise FloatingPointError("Local RR tail 缺少有效有限值")
        rows.append(
            {
                "arm": arm,
                "seed": int(seed),
                "eligible_n": len(values),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "p90": float(np.quantile(values, 0.90)),
                "p95": float(np.quantile(values, 0.95)),
                "max": float(values.max()),
                "gt_2_bpm_n": int((values > 2.0).sum()),
                "gt_2_bpm_fraction": float((values > 2.0).mean()),
                "gt_5_bpm_n": int((values > 5.0).sum()),
                "gt_5_bpm_fraction": float((values > 5.0).mean()),
            }
        )
    return pd.DataFrame(rows)


def metric_denominators(window_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    optional_flags = (
        "ibi_target_eligible",
        "ibi_interpretable",
        "envelope_spearman_target_eligible",
    )
    for (arm, seed), group in window_metrics.groupby(["arm", "seed"], sort=False):
        row: dict[str, Any] = {
            "arm": arm,
            "seed": int(seed),
            "rows": len(group),
            "unique_row_ids": int(group.dataset_row_id.nunique()),
            "unique_samp_ids": int(group.samp_id.nunique()),
        }
        for flag in ("whole_rr_target_eligible", "local_rr_target_eligible", "joint_target_eligible", *optional_flags):
            if flag in group:
                row[flag + "_n"] = int(group[flag].astype(bool).sum())
        for flag in ("joint_prediction_degenerate", "envelope_spearman_prediction_degenerate"):
            if flag in group:
                row[flag + "_n"] = int(group[flag].astype(bool).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def subject_stratified_metrics(window_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (arm, seed, samp_id), group in window_metrics.groupby(["arm", "seed", "samp_id"], sort=False):
        for metric in PRIMARY:
            mask = _metric_mask(group, metric)
            values = group.loc[mask, metric].to_numpy(dtype=float)
            if not len(values) or not np.isfinite(values).all():
                raise FloatingPointError("subject metric 缺少有效有限值")
            row: dict[str, Any] = {
                "arm": arm,
                "seed": int(seed),
                "samp_id": samp_id,
                "metric": metric,
                "eligible_n": len(values),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
            }
            if metric == ERRORS[1]:
                row.update(
                    {
                        "p90": float(np.quantile(values, 0.90)),
                        "p95": float(np.quantile(values, 0.95)),
                        "gt_2_bpm_fraction": float((values > 2.0).mean()),
                        "gt_5_bpm_fraction": float((values > 5.0).mean()),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def subject_factorial_effects(subject_metrics: pd.DataFrame) -> pd.DataFrame:
    required = {"arm", "seed", "samp_id", "metric", "mean"}
    if not required.issubset(subject_metrics.columns):
        raise ValueError("subject metrics 字段不完整")
    wide = subject_metrics.pivot(index=["arm", "seed", "samp_id"], columns="metric", values="mean").reset_index()
    return factorial_effects_by_seed(wide, group_columns=("seed", "samp_id"))


def quality_preserving_simplifications(seed_frame: pd.DataFrame) -> pd.DataFrame:
    _require_complete_seed_frame(seed_frame)
    indexed = seed_frame.set_index(["arm", "seed"])
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        if arm == REFERENCE_ARM or ARM_SPECS[arm].trainable_parameters >= ARM_SPECS[REFERENCE_ARM].trainable_parameters:
            continue
        checks: dict[str, bool] = {}
        for metric in PRIMARY:
            reference = np.asarray([indexed.loc[(REFERENCE_ARM, seed), metric] for seed in SEEDS], dtype=float)
            candidate = np.asarray([indexed.loc[(arm, seed), metric] for seed in SEEDS], dtype=float)
            if metric == PCC:
                delta = reference - candidate
                checks[metric] = bool(delta.mean() <= PCC_ABSOLUTE_TOLERANCE)
            else:
                relative = (candidate - reference) / reference
                checks[metric] = bool(relative.mean() <= ERROR_RELATIVE_TOLERANCE)
            if metric in {ERRORS[1], ERRORS[2], PCC}:
                within = (
                    (reference - candidate) <= PCC_ABSOLUTE_TOLERANCE
                    if metric == PCC
                    else ((candidate - reference) / reference) <= ERROR_RELATIVE_TOLERANCE
                )
                checks[metric + "_paired"] = bool(int(within.sum()) >= 2)
        rows.append(
            {
                "arm": arm,
                "reference_arm": REFERENCE_ARM,
                "parameter_reduction": ARM_SPECS[REFERENCE_ARM].trainable_parameters - ARM_SPECS[arm].trainable_parameters,
                "quality_preserving": bool(all(checks.values())),
                **checks,
            }
        )
    return pd.DataFrame(rows)


def parameter_compute_report() -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "arms": {
            arm: {
                "factors": list(spec.factors),
                "trainable_parameters": spec.trainable_parameters,
                "factor_covered_macs": spec.factor_covered_macs,
            }
            for arm, spec in ARM_SPECS.items()
        },
        "factor_parameters": {"conv20_minus_patch": 15_888, "tm3": 112_896, "ref2": 75_264},
        "covered_macs": {
            "patch_frontend": 3_710_080,
            "conv20_frontend": 110_300_400,
            "residual_dw_block": 67_219_200,
            "tm3": 201_657_600,
            "ref2": 134_438_400,
        },
        "conv20_contract": CONV20_CONTRACT,
        "whole_model_flops": None,
    }


@contextmanager
def exclusive_attempt(
    parent: Path,
    *,
    phase: str,
    lock_hash: str,
    arm: str | None = None,
    seed: int | None = None,
) -> Iterator[Path]:
    """排他创建 attempt；成功冻结，失败保留现场。"""

    parent.mkdir(parents=True, exist_ok=True)
    lock_path = parent / f".execution_{lock_hash}.lock"
    with lock_path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("相同结构因子身份正在运行") from exc
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = parent / f"{phase}_{lock_hash[:12]}_{stamp}_{uuid4().hex[:12]}"
        path.mkdir(exist_ok=False)
        context = {
            "protocol": PROTOCOL,
            "phase": phase,
            "arm": arm,
            "seed": seed,
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
            write_json(path / "freeze_receipt.json", {"protocol": PROTOCOL, "manifest": identity(path / "manifest.json")})
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
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
