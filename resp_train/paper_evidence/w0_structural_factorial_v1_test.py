"""W0 三因素结构对照：固定 validation-selected checkpoint 的 research-test 评价。"""

from __future__ import annotations

import fcntl
import hashlib
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
from resp_train.paper_evidence import w0_structural_factorial_v1 as sf
from resp_train.paper_evidence import w0_structural_factorial_v1_formal as formal
from resp_train.paper_evidence import w0_structural_factorial_v1_summary as p4
from resp_train.paper_evidence.w0_structural_factorial_v1_engineering import finite_tree
from resp_train.paper_evidence.w0_structural_factorial_v1_model import (
    ARMS,
    build_w0_structural_factorial_model,
)


PROTOCOL = "w0-structural-factorial-v1-research-test-20260925"
COUNT = 2_310
SUBJECTS = 8
TEST_SAMPLE_SEED = 20260612
ROW_ORDER_SHA256 = "184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8"
P4_SUMMARY = Path(
    "runs/w0_structural_factorial_v1_es30p15/summary/"
    "summary_32141eab672e_20260925T072509Z_dbcda95df316"
)
P4_MANIFEST_SHA256 = "4ffbcece5b07756bf99342b764e438f5ef67bef16cf76cf76e3f6ceb7a880065"
P4_DECISION_SHA256 = "d829f65eaf464a3bfde9ff9364acd5afb936ecd78395278d335f6648c6c6f1f3"
PROTOCOL_PATH = Path(
    "docs/experiments/w0_structural_factorial_v1_research_test_protocol_20260925.md"
)
MODULE_PATH = Path("resp_train/paper_evidence/w0_structural_factorial_v1_test.py")
SCRIPT_PATH = Path("scripts/run_w0_structural_factorial_v1_test.py")
TEST_PATH = Path("tests/test_w0_structural_factorial_v1_test.py")
IDENTITY_COLUMNS = ("dataset_row_id", "samp_id", "split")
TARGET_COLUMNS = (
    "whole_rr_target_eligible",
    "local_rr_target_eligible",
    "local_rr_target_eligible_windows",
    "joint_target_eligible",
    "envelope_spearman_target_eligible",
    "ibi_target_eligible",
)
TARGET_BOOLEAN_COLUMNS = tuple(
    column for column in TARGET_COLUMNS if column != "local_rr_target_eligible_windows"
)


def _research_test_root() -> Path:
    return sf.SOURCE_ROOT / sf.OUTPUT_ROOT / "research_test"


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
        sf.SPEC_PATH,
        sf.MODEL_PATH,
        sf.CONTROL_PATH,
        formal.FORMAL_SOURCE_PATH,
        p4.SUMMARY_SOURCE_PATH,
        Path("resp_train/crd/experiment.py"),
        Path("resp_train/crd/tf_v1_research_test_data.py"),
        Path("resp_train/data/research_v2.py"),
        Path("resp_train/engine/train.py"),
        Path("resp_train/metrics/task.py"),
    )
    missing = [str(path) for path in paths if not (sf.ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"research-test 实现缺少文件: {missing}")
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
            raise RuntimeError("相同 research-test identity 正在运行") from exc
        if reject_completed:
            for receipt_path in parent.glob("*/freeze_receipt.json"):
                manifest_path = receipt_path.parent / "manifest.json"
                freeze = json.loads(receipt_path.read_text(encoding="utf-8"))
                sf.verify_identity(manifest_path, freeze["manifest"])
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if (
                    manifest.get("evidence_identity_sha256") == evidence_hash
                    and manifest.get("phase") == phase
                    and manifest.get("arm") == arm
                    and manifest.get("seed") == seed
                ):
                    raise FileExistsError(f"相同 research-test cell 已完成: {receipt_path.parent}")
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
        sf.write_json(output / "lifecycle_started.json", {**context, "status": "running"})
        try:
            yield output
            sf.write_json(
                output / "lifecycle_completed.json",
                {**context, "status": "completed", "ended_at": datetime.now(timezone.utc).isoformat()},
            )
            manifest = {
                **context,
                "status": "completed",
                "files": {
                    str(path.relative_to(output)): sf.identity(path)
                    for path in sorted(output.rglob("*"))
                    if path.is_file()
                },
            }
            sf.write_json(output / "manifest.json", manifest)
            sf.write_json(
                output / "freeze_receipt.json",
                {"protocol": PROTOCOL, "manifest": sf.identity(output / "manifest.json")},
            )
        except BaseException as exc:
            sf.write_json(
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
        raise ValueError(f"research-test attempt 失败: {output}")
    freeze = json.loads((output / "freeze_receipt.json").read_text(encoding="utf-8"))
    sf.verify_identity(output / "manifest.json", freeze["manifest"])
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("protocol") != PROTOCOL
        or manifest.get("phase") != phase
        or manifest.get("status") != "completed"
        or manifest.get("evidence_identity_sha256") != evidence_hash
        or manifest.get("arm") != arm
        or manifest.get("seed") != seed
    ):
        raise ValueError("research-test attempt identity 漂移")
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
            "factorial_effects_across_seed.csv",
            "subject_stratified_metrics.csv",
            "metric_denominators.csv",
            "validation_test_comparison.csv",
            "source_manifest.json",
            "access_receipt.json",
        },
    }[phase]
    if not required.issubset(manifest["files"]):
        raise ValueError("research-test attempt 缺少必需产物")
    for relative, expected in manifest["files"].items():
        path = (output / relative).resolve()
        if not path.is_relative_to(output):
            raise ValueError("research-test manifest 路径越界")
        sf.verify_identity(path, expected)
    return manifest


def _p4_summary_path() -> Path:
    return sf.SOURCE_ROOT / P4_SUMMARY


def _checkpoint_entry(
    *,
    arm: str,
    seed: int,
    formal_attempt: Path,
    source: Mapping[str, Any],
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    formal.verify_formal_attempt(
        formal_attempt,
        lock_hash=formal.P2_LOCK_SHA256,
        arm=arm,
        seed=seed,
    )
    receipt = json.loads((formal_attempt / "formal_receipt.json").read_text(encoding="utf-8"))
    run_dir = (formal_attempt / receipt["run_dir"]).resolve()
    if not run_dir.is_relative_to(formal_attempt):
        raise ValueError("allowlist training run_dir 越界")
    cfg = OmegaConf.load(run_dir / "config.yaml")
    baseline = OmegaConf.create(lock["baselines"][str(seed)])
    sf.validate_config(
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
        raise ValueError("allowlist validation selector 漂移")
    formal_manifest = json.loads((formal_attempt / "manifest.json").read_text(encoding="utf-8"))
    checkpoint_path = run_dir / "checkpoint_best_local_rr.pt"
    checkpoint_relative = str(checkpoint_path.relative_to(formal_attempt))
    sf.verify_identity(checkpoint_path, formal_manifest["files"][checkpoint_relative])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        int(checkpoint["epoch"]) != selected_epoch
        or checkpoint["config"] != OmegaConf.to_container(cfg, resolve=True)
    ):
        raise ValueError("allowlist checkpoint epoch/config 漂移")
    finite_tree(checkpoint["model_state_dict"], label="allowlist_checkpoint")
    development = sorted(
        set(pd.read_csv(formal_attempt / "train_rows.csv").samp_id.astype(int))
        | set(pd.read_csv(formal_attempt / "val_rows.csv").samp_id.astype(int))
    )
    files = {}
    for name, path in (
        ("config", run_dir / "config.yaml"),
        ("history", run_dir / "train_history.csv"),
        ("checkpoint", checkpoint_path),
        ("formal_manifest", formal_attempt / "manifest.json"),
        ("formal_receipt", formal_attempt / "formal_receipt.json"),
    ):
        files[name] = {"path": str(path), **sf.identity(path)}
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
    state = sf.git_state()
    if state["status_porcelain"]:
        raise RuntimeError("research-test allowlist 要求干净 Git 工作树")
    summary = _p4_summary_path()
    if sf.sha256_file(summary / "manifest.json") != P4_MANIFEST_SHA256:
        raise ValueError("P4 summary manifest identity 漂移")
    if sf.sha256_file(summary / "decision.json") != P4_DECISION_SHA256:
        raise ValueError("P4 decision identity 漂移")
    p4.verify_summary_attempt(summary)
    source_manifest = json.loads((summary / "source_manifest.json").read_text(encoding="utf-8"))
    formal_runs = source_manifest["formal_runs"]
    lock, lock_hash = formal.load_formal_contract()
    entries: list[dict[str, Any]] = []
    for arm in ARMS:
        for seed in sf.SEEDS:
            key = f"{arm}/{seed}"
            source = formal_runs[key]
            attempt = Path(source["attempt"]).resolve()
            sf.verify_identity(attempt / "manifest.json", source["manifest"])
            entries.append(
                _checkpoint_entry(
                    arm=arm,
                    seed=seed,
                    formal_attempt=attempt,
                    source=source,
                    lock=lock,
                )
            )
    if len(entries) != 24 or len({item["entry_id"] for item in entries}) != 24:
        raise ValueError("research-test checkpoint allowlist 不完整")

    cache_manifest_path = FROZEN_RESEARCH_TEST_CACHE_ROOT / "cache_manifest.json"
    if sf.sha256_file(cache_manifest_path) != FROZEN_RESEARCH_TEST_CACHE_MANIFEST_SHA256:
        raise ValueError("research-test cache manifest identity 漂移")
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    split = cache_manifest["splits"]["test"]
    if (
        int(split["count"]) != COUNT
        or int(split["samp_id_count"]) != SUBJECTS
        or split["row_ids_sha256"] != ROW_ORDER_SHA256
    ):
        raise ValueError("research-test cache split contract 漂移")
    cache_files = {
        name: cache_manifest["files"][name]
        for name in ("test_w.npy", "test_row_ids.npy", "w_frequencies_hz.npy")
    }
    code_files = {str(path): sf.identity(sf.ROOT / path) for path in _code_paths()}
    preparation_identity = _canonical_sha256(
        {
            "protocol": PROTOCOL,
            "validation_summary_manifest_sha256": P4_MANIFEST_SHA256,
            "validation_decision_sha256": P4_DECISION_SHA256,
            "training_implementation_lock_sha256": lock_hash,
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
        "training_protocol": sf.PROTOCOL,
        "training_implementation_lock_sha256": lock_hash,
        "validation_summary": {
            "path": str(summary),
            "manifest": sf.identity(summary / "manifest.json"),
            "decision": sf.identity(summary / "decision.json"),
        },
        "arms": list(ARMS),
        "seeds": list(sf.SEEDS),
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
            "manifest": {"path": str(cache_manifest_path), **sf.identity(cache_manifest_path)},
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
        sf.write_json(output / "allowlist.json", allowlist)
        sf.write_json(
            output / "allowlist_receipt.json",
            {
                "protocol": PROTOCOL,
                "entries": 24,
                "allowlist": sf.identity(output / "allowlist.json"),
                "allowlist_preparation_identity_sha256": preparation_identity,
                "validation_summary_manifest_sha256": P4_MANIFEST_SHA256,
            },
        )
        sf.write_json(
            output / "source_manifest.json",
            {
                "validation_summary": allowlist["validation_summary"],
                "formal_runs": formal_runs,
                "training_implementation_lock_sha256": lock_hash,
            },
        )
        sf.write_json(
            output / "source_code.json",
            {"git": state, "files": allowlist["code_files"]},
        )
        sf.write_json(
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
        raise ValueError("research-test allowlist preparation identity 非法")
    verify_attempt(path, phase="allowlist", evidence_hash=preparation_identity)
    allowlist_path = path / "allowlist.json"
    allowlist = json.loads(allowlist_path.read_text(encoding="utf-8"))
    if (
        allowlist.get("schema_version") != 1
        or allowlist.get("protocol") != PROTOCOL
        or allowlist.get("allowlist_preparation_identity_sha256") != preparation_identity
        or tuple(allowlist.get("arms", ())) != ARMS
        or tuple(allowlist.get("seeds", ())) != sf.SEEDS
        or allowlist.get("split") != "test"
        or allowlist.get("count") != COUNT
        or allowlist.get("samp_id_count") != SUBJECTS
        or allowlist.get("test_sample_seed") != TEST_SAMPLE_SEED
        or allowlist.get("row_order_sha256") != ROW_ORDER_SHA256
        or allowlist.get("batch_size") != 128
        or allowlist.get("amp_dtype") != "bfloat16"
        or allowlist.get("include_test_only") is not True
        or allowlist.get("training_implementation_lock_sha256") != formal.P2_LOCK_SHA256
        or allowlist.get("cache", {}).get("identity") != FROZEN_RESEARCH_TEST_CACHE_IDENTITY
    ):
        raise ValueError("research-test allowlist 合同漂移")
    expected = [(arm, seed) for arm in ARMS for seed in sf.SEEDS]
    observed = [(item["arm"], int(item["seed"])) for item in allowlist["entries"]]
    if observed != expected:
        raise ValueError("research-test allowlist 24-cell 顺序或身份漂移")
    for relative, expected_identity in allowlist["code_files"].items():
        sf.verify_identity(sf.ROOT / relative, expected_identity)
    return allowlist, sf.sha256_file(allowlist_path)


def check_test_rows(
    rows: pd.DataFrame,
    reference: pd.DataFrame | None = None,
    *,
    count: int | None = None,
    subjects: int | None = None,
    row_hash: str | None = None,
) -> None:
    count = COUNT if count is None else count
    subjects = SUBJECTS if subjects is None else subjects
    row_hash = ROW_ORDER_SHA256 if row_hash is None else row_hash
    if (
        len(rows) != count
        or rows.dataset_row_id.isna().any()
        or rows.dataset_row_id.dtype.kind not in "iu"
        or rows.dataset_row_id.duplicated().any()
        or rows.samp_id.isna().any()
        or rows.samp_id.nunique() != subjects
        or set(rows.split.astype(str)) != {"test"}
        or formal.array_hash(rows.dataset_row_id.to_numpy()) != row_hash
    ):
        raise ValueError("research-test rows 数量、subject、顺序或 identity 错误")
    if reference is not None:
        for key in IDENTITY_COLUMNS:
            if not np.array_equal(rows[key].to_numpy(), reference[key].to_numpy()):
                raise ValueError(f"research-test row identity/order 不一致: {key}")


def check_test_metrics(metrics: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    check_test_rows(metrics, rows)
    sf.validate_metrics(metrics, rows)
    for key in TARGET_BOOLEAN_COLUMNS:
        values = metrics[key]
        if values.isna().any() or not values.isin([True, False]).all():
            raise ValueError(f"research-test target eligibility 漂移: {key}")
        if key in rows and not np.array_equal(values.to_numpy(), rows[key].to_numpy()):
            raise ValueError(f"research-test target eligibility 漂移: {key}")
    eligible_windows = pd.to_numeric(
        metrics["local_rr_target_eligible_windows"], errors="coerce"
    ).to_numpy(dtype=float)
    if (
        not np.isfinite(eligible_windows).all()
        or (eligible_windows < 0).any()
        or not np.equal(eligible_windows, np.floor(eligible_windows)).all()
        or not np.array_equal(
            eligible_windows > 0,
            metrics["local_rr_target_eligible"].to_numpy(dtype=bool),
        )
    ):
        raise ValueError("research-test Local-RR target eligibility 窗口数非法")
    if "local_rr_target_eligible_windows" in rows and not np.array_equal(
        metrics["local_rr_target_eligible_windows"].to_numpy(),
        rows["local_rr_target_eligible_windows"].to_numpy(),
    ):
        raise ValueError("research-test target eligibility 漂移: local_rr_target_eligible_windows")
    for key in ("respiratory_band_coherence", "constrained_ndtw"):
        values = pd.to_numeric(metrics[key], errors="coerce").to_numpy(dtype=float)
        eligible = metrics.joint_target_eligible.to_numpy(dtype=bool)
        if np.isinf(values).any() or not np.array_equal(np.isfinite(values), eligible):
            raise FloatingPointError(f"research-test secondary finite/eligibility 不一致: {key}")
    summary = summarize_task_metrics(metrics)
    for metric in sf.PRIMARY:
        value = float(summary.iloc[0][metric + "_mean"])
        count = int(summary.iloc[0][metric + "_n"])
        if not np.isfinite(value) or count <= 0:
            raise FloatingPointError(f"research-test primary 汇总不完整: {metric}")
    return summary


def quality_flags(metrics: pd.DataFrame) -> dict[str, Any]:
    counts = {
        key: int(metrics[key].astype(bool).sum())
        for key in (
            "joint_prediction_degenerate",
            "envelope_spearman_prediction_degenerate",
        )
    }
    return {
        "prediction_degeneracy": counts,
        "quality_acceptance_passed": not any(counts.values()),
    }


def evaluation_config(
    entry: Mapping[str, Any],
    lock: Mapping[str, Any],
    *,
    device: str,
) -> tuple[DictConfig, DictConfig]:
    config_path = Path(entry["files"]["config"]["path"])
    sf.verify_identity(config_path, entry["files"]["config"])
    cfg = OmegaConf.load(config_path)
    if OmegaConf.to_container(cfg, resolve=True) != entry["training_config"]:
        raise ValueError("research-test config 与 allowlist 不一致")
    formal_attempt = Path(entry["formal_attempt"])
    baseline = formal.load_formal_contract()[0]["baselines"][str(entry["seed"])]
    sf.validate_config(
        cfg,
        OmegaConf.create(baseline),
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
        raise ValueError("research-test 原生推理配置漂移")
    cfg.training.device = device
    cfg.training.show_progress = False
    data_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    data_cfg.data.tf_research_test_cache_path = lock["cache"]["root"]
    return cfg, data_cfg


def guarded_batches(loader: Any, rows: pd.DataFrame):
    offset = 0
    for batch in loader:
        count = len(batch["x"])
        expected = rows.iloc[offset : offset + count]
        if count <= 0 or len(expected) != count or set(batch.get("tf", {})) != {"w"}:
            raise ValueError("research-test batch 数量或 W keys 错误")
        for key in IDENTITY_COLUMNS:
            raw = batch["meta"][key]
            actual = raw.cpu().numpy() if torch.is_tensor(raw) else np.asarray(raw)
            if not np.array_equal(actual, expected[key].to_numpy()):
                raise ValueError(f"research-test batch identity/order 错误: {key}")
        if tuple(batch["tf"]["w"].shape) != (count, 97, 360):
            raise ValueError("research-test W shape 错误")
        for tensor in (batch["x"], batch["target"], batch["tf"]["w"]):
            if not bool(torch.isfinite(tensor).all()):
                raise FloatingPointError("research-test input/target/W 非有限")
        offset += count
        yield batch
    if offset != len(rows):
        raise ValueError("research-test loader 样本不完整")


def _entry(lock: Mapping[str, Any], arm: str, seed: int) -> Mapping[str, Any]:
    if arm not in ARMS or int(seed) not in sf.SEEDS:
        raise ValueError("research-test arm/seed 不属于固定矩阵")
    return next(item for item in lock["entries"] if item["arm"] == arm and int(item["seed"]) == int(seed))


def _verify_evaluation_sources(lock: Mapping[str, Any], entry: Mapping[str, Any]) -> None:
    for item in entry["files"].values():
        sf.verify_identity(Path(item["path"]), item)
    cache_root = Path(lock["cache"]["root"])
    sf.verify_identity(Path(lock["cache"]["manifest"]["path"]), lock["cache"]["manifest"])
    for filename, expected in lock["cache"]["files"].items():
        sf.verify_identity(cache_root / filename, expected)
    index = lock["dataset_index"]
    if sf.sha256_file(Path(index["path"])) != index["sha256"]:
        raise ValueError("research-test dataset index identity 漂移")


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
        runtime = formal.runtime_preflight(device)
        training_environment = json.loads(
            (Path(entry["formal_attempt"]) / "environment.json").read_text(encoding="utf-8")
        )
        runtime_compatibility = formal._runtime_compatibility(runtime, training_environment)
        sf.write_json(output / "environment.json", runtime)
        sf.write_json(
            output / "allowlist_source.json",
            {
                "path": str(allowlist_path.resolve()),
                "allowlist": sf.identity(allowlist_path / "allowlist.json"),
                "runtime_compatibility": runtime_compatibility,
            },
        )
        sf.write_json(
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
        checkpoint_path = Path(entry["files"]["checkpoint"]["path"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (
            int(checkpoint["epoch"]) != int(entry["selected_epoch"])
            or checkpoint["config"] != entry["training_config"]
        ):
            raise ValueError("research-test checkpoint identity 漂移")
        _validate_checkpoint_config(checkpoint.get("config"), cfg)
        finite_tree(checkpoint["model_state_dict"], label="research_test_checkpoint")
        model = build_w0_structural_factorial_model(cfg)
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
            raise ValueError("research-test 与 development subject 交叉")
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
            raise ValueError("research-test dataset 样本集合不完整")
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
        sf.write_json(
            output / "access_receipt.json",
            {
                "split": "test",
                "arm": arm,
                "seed": int(seed),
                "rows": COUNT,
                "samp_ids": SUBJECTS,
                "row_order_sha256": formal.array_hash(rows.dataset_row_id.to_numpy()),
                "checkpoint": entry["files"]["checkpoint"],
                "cache_identity": FROZEN_RESEARCH_TEST_CACHE_IDENTITY,
                "test_sample_seed": TEST_SAMPLE_SEED,
                "evidence_role": lock["evidence_role"],
            },
        )
        sf.write_json(
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
    for arm in ARMS:
        for seed in sf.SEEDS:
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
                raise ValueError(f"research-test cell 多个完成 attempt: {arm}/{seed}")
            status = "completed" if completed else ("running" if running else ("failed" if failed else "pending"))
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
    if status["counts"] != {"pending": 0, "running": 0, "failed": 0, "completed": 24}:
        raise RuntimeError(f"research-test summary 要求完整矩阵，当前={status['counts']}")
    attempts = [Path(cell["completed"][0]) for cell in status["cells"]]
    return attempts, status["allowlist_sha256"]


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
        group = group.set_index("seed").loc[list(sf.SEEDS)]
        row: dict[str, Any] = {"arm": arm, "seed_count": len(sf.SEEDS)}
        for column in value_columns:
            values = group[column].to_numpy(dtype=float)
            row[column + "_seed_mean"] = float(values.mean()) if np.isfinite(values).all() else np.nan
            row[column + "_seed_sample_sd"] = (
                float(values.std(ddof=1)) if np.isfinite(values).all() else np.nan
            )
            row[column + "_finite_seeds"] = int(np.isfinite(values).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _validation_test_comparison(test_seed: pd.DataFrame) -> pd.DataFrame:
    validation = pd.read_csv(_p4_summary_path() / "seed_primary_metrics.csv")
    rows: list[dict[str, Any]] = []
    validation = validation.set_index(["arm", "seed"])
    test = test_seed.set_index(["arm", "seed"])
    for arm in ARMS:
        for seed in sf.SEEDS:
            for metric in sf.PRIMARY:
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
    state = sf.git_state()
    if state["status_porcelain"]:
        raise RuntimeError("research-test summary 要求干净 Git 工作树")
    attempts, observed_hash = _evaluation_attempts(allowlist_path)
    if observed_hash != allowlist_hash:
        raise ValueError("research-test status allowlist identity 漂移")
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
            raise ValueError("research-test evaluation receipt 与 allowlist 不一致")
        rows = pd.read_csv(attempt / "test_rows.csv")
        metrics = pd.read_csv(attempt / "metrics.csv")
        check_test_rows(rows)
        check_test_rows(metrics, rows)
        computed = check_test_metrics(metrics, rows)
        quality = quality_flags(metrics)
        if any(receipt.get(key) != value for key, value in quality.items()):
            raise ValueError("research-test quality receipt 漂移")
        if reference is None:
            reference = metrics[[*IDENTITY_COLUMNS, *TARGET_COLUMNS]].copy()
        else:
            for key in (*IDENTITY_COLUMNS, *TARGET_COLUMNS):
                if not np.array_equal(metrics[key].to_numpy(), reference[key].to_numpy()):
                    raise ValueError(f"research-test 全矩阵 target identity 漂移: {key}")
        saved = pd.read_csv(attempt / "metrics_summary.csv")
        if len(saved) != 1:
            raise ValueError("research-test metrics summary 必须一行")
        for metric in sf.PRIMARY:
            if (
                not np.isclose(
                    saved.iloc[0][metric + "_mean"],
                    computed.iloc[0][metric + "_mean"],
                    atol=1e-12,
                    rtol=0,
                )
                or int(saved.iloc[0][metric + "_n"]) != int(computed.iloc[0][metric + "_n"])
            ):
                raise ValueError("research-test saved summary 漂移")
        frames.append(metrics)
        seed_summaries.append(saved)
        sources[f"{arm}/{seed}"] = {
            "attempt": str(attempt),
            "manifest": sf.identity(attempt / "manifest.json"),
            "metrics": sf.identity(attempt / "metrics.csv"),
            "selected_epoch": entry["selected_epoch"],
            "checkpoint_sha256": entry["files"]["checkpoint"]["sha256"],
        }
    combined = pd.concat(frames, ignore_index=True)
    if len(combined) != 24 * COUNT:
        raise ValueError("research-test metrics 总行数错误")
    seed_primary = sf.seed_primary_metrics(combined, expected_rows=COUNT)
    arm_summary = sf.arm_primary_summary(seed_primary)
    conditional = sf.conditional_effects_by_seed(seed_primary)
    effects = sf.factorial_effects_by_seed(seed_primary)
    effects_across = sf.factorial_effects_across_seed(effects)
    tails = sf.local_rr_tail_summary(combined)
    tail_across = p4.local_rr_tail_across_seed(tails)
    subject_metrics = sf.subject_stratified_metrics(combined)
    macro_seed, macro_across = p4.subject_macro_tables(
        subject_metrics, expected_subjects=SUBJECTS
    )
    subject_wide = subject_metrics.pivot(
        index=["arm", "seed", "samp_id"], columns="metric", values="mean"
    ).reset_index()
    subject_conditional = sf.conditional_effects_by_seed(
        subject_wide, group_columns=("seed", "samp_id")
    )
    subject_factorial = sf.subject_factorial_effects(subject_metrics)
    denominators = sf.metric_denominators(combined)
    secondary_seed = pd.concat(seed_summaries, ignore_index=True)
    secondary_across = _secondary_across_seed(secondary_seed)
    validation_test = _validation_test_comparison(seed_primary)
    factor_decisions = p4.factor_assessments(conditional, effects, effects_across)
    validation_decision_path = _p4_summary_path() / "decision.json"
    validation_decision = json.loads(validation_decision_path.read_text(encoding="utf-8"))
    decision = {
        "protocol": PROTOCOL,
        "evidence_role": lock["evidence_role"],
        "checkpoint_selection": "fixed by validation before test access",
        "validation_decision": {
            "path": str(validation_decision_path),
            **sf.identity(validation_decision_path),
            "factor_classifications": {
                item["factor"]: item["classification"]
                for item in validation_decision["factor_assessments"]
            },
        },
        "test_factor_assessments_descriptive": factor_decisions,
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
            "conditional_effects_by_seed.csv": conditional,
            "factorial_effects_by_seed.csv": effects,
            "factorial_effects_across_seed.csv": effects_across,
            "local_rr_tail_summary.csv": tails,
            "local_rr_tail_across_seed.csv": tail_across,
            "subject_stratified_metrics.csv": subject_metrics,
            "subject_macro_by_seed.csv": macro_seed,
            "subject_macro_across_seed.csv": macro_across,
            "subject_conditional_effects.csv": subject_conditional,
            "subject_factorial_effects.csv": subject_factorial,
            "metric_denominators.csv": denominators,
            "secondary_seed_metrics.csv": secondary_seed,
            "secondary_across_seed.csv": secondary_across,
            "validation_test_comparison.csv": validation_test,
        }
        for filename, frame in outputs.items():
            frame.to_csv(output / filename, index=False, na_rep="NA")
        sf.write_json(output / "decision.json", decision)
        sf.write_json(
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
        sf.write_json(
            output / "access_receipt.json",
            {
                "evaluation_attempts_read": 24,
                "test_metric_rows_read": len(combined),
                "test_arrays_read": False,
                "checkpoint_content_read": False,
                "gpu_used": False,
                "model_inference_used": False,
                "evidence_role": lock["evidence_role"],
            },
        )
        sf.write_json(
            output / "summary_receipt.json",
            {
                "protocol": PROTOCOL,
                "status": "complete",
                "allowlist_sha256": allowlist_hash,
                "arms": list(ARMS),
                "seeds": list(sf.SEEDS),
                "evaluation_attempts": 24,
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


__all__ = [
    "COUNT",
    "IDENTITY_COLUMNS",
    "PROTOCOL",
    "ROW_ORDER_SHA256",
    "SUBJECTS",
    "TARGET_BOOLEAN_COLUMNS",
    "TARGET_COLUMNS",
    "check_test_metrics",
    "check_test_rows",
    "evaluation_config",
    "evaluation_status",
    "evidence_attempt",
    "guarded_batches",
    "load_allowlist",
    "prepare_allowlist",
    "quality_flags",
    "run_evaluation",
    "summarize",
    "verify_attempt",
]
