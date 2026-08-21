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
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from resp_train.crd.config import check_crd_dependencies
from resp_train.data.factory import WindowDataBundle, build_window_data
from resp_train.data.research_v2 import adapt_research_v2_index
from resp_train.engine import collect_predictions
from resp_train.metrics.task import evaluate_task_predictions
from resp_train.temporal.formal import EXPECTED_CANDIDATES, FORMAL_SEEDS, validate_formal_receipt
from resp_train.temporal.formal_summary import (
    EXPECTED_METRICS_COLUMNS,
    PRIMARY_SAMPLE_COLUMNS,
    STATIC_VALIDATION_COLUMNS,
    validate_summary_receipt,
)
from resp_train.temporal.gpu_engineering import _cuda_identity, load_gpu_engineering_config, sha256_file
from resp_train.temporal.model import build_resp_temporal_model, trainable_parameter_count
from resp_train.utils.run import set_seed


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs/resp_temporal_v1/research_test_v1.yaml"
CONFIG_SCHEMA_VERSION = "rtm-v1-research-test-config-v1"
PROTOCOL_ID = "resp-temporal-v1-reused-research-test-20260821"
RECEIPT_SCHEMA_VERSION = "rtm-v1-research-test-receipt-v1"
FROZEN_CONFIG_SHA256 = "3d8551989fbfa07e8ef9454fbb348f2908151f35c681e15a6191b61a0c60a406"
EVIDENCE_LABEL = (
    "reused research-test evidence; candidate and checkpoint selection frozen before RTM-v1 test access; "
    "not untouched held-out evidence"
)
EXPECTED_TEST_WINDOWS = 2310
EXPECTED_TEST_SAMP_IDS = 8
EXPECTED_TEST_ROW_HASH = "184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8"
EXPECTED_TOTAL_METRIC_ROWS = len(EXPECTED_CANDIDATES) * len(FORMAL_SEEDS) * EXPECTED_TEST_WINDOWS
OUTPUT_ARTIFACTS = (
    "resolved_config.json",
    "checkpoint_inputs.json",
    "test_data_receipt.json",
    "research_test_metrics.csv",
    "research_test_seed_summary.csv",
    "research_test_primary_mean_sd.csv",
    "runtime_summary.json",
    "artifact_manifest.json",
)


@dataclass(frozen=True)
class ResearchTestConfig:
    path: Path
    sha256: str
    raw: dict[str, Any]
    output_dir: Path


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(f"{context}字段必须严格为{sorted(expected)}，实际为{sorted(actual)}")


def _repo_path(value: Any, *, context: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{context}必须是仓库内相对路径")
    return REPO_ROOT / relative


def _is_sha256(value: Any) -> bool:
    rendered = str(value)
    return len(rendered) == 64 and all(character in "0123456789abcdef" for character in rendered)


def validate_research_test_config(raw: Mapping[str, Any]) -> None:
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
            "evaluation",
            "access",
            "output",
        },
        "research-test config",
    )
    if (
        raw["schema_version"] != CONFIG_SCHEMA_VERSION
        or raw["protocol_id"] != PROTOCOL_ID
        or raw["role"] != "reused_research_test_evaluation"
        or raw["evidence_label"] != EVIDENCE_LABEL
    ):
        raise ValueError("research-test config identity/evidence漂移")
    if raw["authorization"] != {
        "user_authorized_research_test": True,
        "user_manual_execution": True,
        "codex_execution": False,
        "model_training": False,
        "checkpoint_reselection": False,
        "candidate_reselection": False,
    }:
        raise ValueError("research-test authorization漂移")
    provenance = raw["provenance"]
    _require_exact_keys(
        provenance,
        {
            "research_test_protocol_path",
            "research_test_protocol_sha256",
            "primary_metrics_table_path",
            "primary_metrics_table_sha256",
            "validation_lock_path",
            "validation_lock_sha256",
            "validation_summary_receipt_path",
            "validation_summary_receipt_sha256",
            "validation_summary_manifest_path",
            "validation_summary_manifest_sha256",
            "formal_root",
            "formal_execution_git_commit",
            "require_clean_git",
        },
        "research-test provenance",
    )
    expected_provenance = {
        "research_test_protocol_path": "docs/experiments/resp_temporal_v1_research_test_protocol_20260821.md",
        "research_test_protocol_sha256": "1da00281ad435a14cc0b1e9a26b84554cae35ec02784ea7f2f369ec539be8184",
        "primary_metrics_table_path": "docs/experiments/resp_temporal_v1_primary_metrics_table_20260821.md",
        "primary_metrics_table_sha256": "417fe73b491d459fe649a365fdad590d00c1d0aa992df0e150b144dde4ba8f21",
        "validation_lock_path": "docs/experiments/resp_temporal_v1_validation_lock_20260821.json",
        "validation_lock_sha256": "989ef0a3a5941ead3e80aba25606878f88d315bd23a1cf5ca4260317f3cffce6",
        "validation_summary_receipt_path": "runs/resp_temporal_v1/formal_validation_summary_v1/access_receipt.json",
        "validation_summary_receipt_sha256": "6f9f1e873b8910b22241bc0e9f2c510909835b0bbc2a9edc12f0e1788fa59aad",
        "validation_summary_manifest_path": "runs/resp_temporal_v1/formal_validation_summary_v1/artifact_manifest.json",
        "validation_summary_manifest_sha256": "aab22094d6efd11927c952e9f12bcbab24e30cedc9a2a5f282f61056f1f193dc",
        "formal_root": "runs/resp_temporal_v1/formal",
        "formal_execution_git_commit": "24c54a88ea8b17ad183b48976bd1f690e3867706",
        "require_clean_git": True,
    }
    if provenance != expected_provenance:
        raise ValueError("research-test provenance漂移")
    if tuple(raw["candidates"]) != EXPECTED_CANDIDATES or tuple(int(seed) for seed in raw["seeds"]) != FORMAL_SEEDS:
        raise ValueError("research-test candidate/seed矩阵漂移")
    if raw["data"] != {
        "dataset_root": (
            "/mnt/disk_code/marques/resp_prepare/dataset/"
            "20260620_research_v2_resp_reconstruction_stage2_1_segrobustz_bcgstagee_log1psoftz_robustconf"
        ),
        "index_csv": "training/dataset_index.csv",
        "dataset_index_sha256": "f65ae6524632187c7c2795bd7b5e25434afb649b5a04fe39084b70cc59f9b04f",
        "split": "test",
        "expected_windows": EXPECTED_TEST_WINDOWS,
        "expected_samp_ids": EXPECTED_TEST_SAMP_IDS,
        "expected_row_ids_sha256": EXPECTED_TEST_ROW_HASH,
        "sample_strategy": "stratified_random",
        "sample_seed": 20260612,
        "input_key": "bcg_rawish_segment_soft_z_key",
        "target_key": "target_waveform_segment_soft_z_key",
    }:
        raise ValueError("research-test data identity漂移")
    if raw["evaluation"] != {
        "checkpoint_filename": "checkpoint_best_local_rr.pt",
        "device": "cuda:0",
        "batch_size": 128,
        "use_amp": True,
        "amp_dtype": "bfloat16",
        "include_test_only_metrics": False,
        "primary_metrics": list(PRIMARY_SAMPLE_COLUMNS),
        "sample_aggregation": "direct_mean",
        "candidate_aggregation": "arithmetic_mean_and_sample_sd_ddof_1",
        "pareto": False,
        "ranking": False,
        "weighted_score": False,
        "paired_seed_direction": False,
        "confirmatory_p_value": False,
    }:
        raise ValueError("research-test evaluation口径漂移")
    if raw["access"] != {
        "train_signal_target_allowed": False,
        "validation_signal_target_prediction_allowed": False,
        "research_test_signal_target_allowed": True,
        "formal_checkpoint_allowed": True,
        "checkpoint_final_allowed": False,
        "model_training_allowed": False,
        "model_inference_allowed": True,
        "gpu_required": True,
    }:
        raise ValueError("research-test access合同漂移")
    if raw["output"] != {
        "root": "runs/resp_temporal_v1/research_test/rtm_v1_research_test_v1",
        "overwrite": False,
        "resume": False,
        "retain_failed_lifecycle": True,
        "files": [*OUTPUT_ARTIFACTS[:-1], "artifact_manifest.json", "access_receipt.json", "access_receipt.sha256"],
    }:
        raise ValueError("research-test output合同漂移")


def load_research_test_config(path: str | Path = DEFAULT_CONFIG_PATH) -> ResearchTestConfig:
    config_path = Path(path).resolve()
    if config_path != DEFAULT_CONFIG_PATH.resolve():
        raise ValueError(f"RTM-v1 research-test只允许冻结config: {DEFAULT_CONFIG_PATH}")
    if not config_path.is_file():
        raise FileNotFoundError(f"research-test config不存在: {config_path}")
    actual_sha = sha256_file(config_path)
    if actual_sha != FROZEN_CONFIG_SHA256:
        raise ValueError(f"research-test config SHA-256漂移: {actual_sha}")
    raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("research-test config顶层必须是mapping")
    validate_research_test_config(raw)
    return ResearchTestConfig(
        path=config_path,
        sha256=actual_sha,
        raw=raw,
        output_dir=_repo_path(raw["output"]["root"], context="research-test output root"),
    )


def _row_ids_sha256(frame: pd.DataFrame) -> str:
    if "dataset_row_id" not in frame:
        raise KeyError("research-test rows缺少dataset_row_id")
    values = frame["dataset_row_id"].to_numpy(dtype=np.int64, copy=True)
    return hashlib.sha256(values.tobytes()).hexdigest()


def build_primary_tables(
    metrics: pd.DataFrame,
    *,
    candidates: Sequence[str] = EXPECTED_CANDIDATES,
    seeds: Sequence[int] = FORMAL_SEEDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"candidate_id", "seed", *PRIMARY_SAMPLE_COLUMNS}
    missing = required - set(metrics)
    if missing:
        raise ValueError(f"research-test metrics缺少主指标字段: {sorted(missing)}")
    identities = [(str(candidate), int(seed)) for candidate in candidates for seed in seeds]
    observed = set(zip(metrics["candidate_id"].astype(str), metrics["seed"].astype(int), strict=True))
    if observed != set(identities):
        raise ValueError("research-test metrics candidate/seed identity不完整")
    rows: list[dict[str, Any]] = []
    for candidate, seed in identities:
        subset = metrics.loc[
            metrics["candidate_id"].astype(str).eq(candidate) & metrics["seed"].astype(int).eq(seed)
        ]
        if subset.empty:
            raise ValueError("research-test seed metrics为空")
        row: dict[str, Any] = {"candidate_id": candidate, "seed": seed, "sample_rows": len(subset)}
        for metric in PRIMARY_SAMPLE_COLUMNS:
            values = pd.to_numeric(subset[metric], errors="coerce")
            finite = values[np.isfinite(values.to_numpy(dtype=np.float64, na_value=np.nan))]
            if finite.empty:
                raise ValueError(f"research-test seed主指标没有finite eligible sample: {candidate}/{seed}/{metric}")
            row[metric] = float(finite.mean())
            row[f"{metric}_eligible_count"] = int(finite.size)
        rows.append(row)
    seed_summary = pd.DataFrame(rows)
    candidate_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        subset = seed_summary.loc[seed_summary["candidate_id"].eq(candidate)]
        if tuple(subset["seed"].astype(int)) != tuple(int(seed) for seed in seeds):
            raise ValueError(f"research-test candidate seed顺序/数量漂移: {candidate}")
        row = {"candidate_id": str(candidate), "seed_count": len(subset)}
        for metric in PRIMARY_SAMPLE_COLUMNS:
            values = subset[metric].to_numpy(dtype=np.float64)
            if values.size != 3 or not np.isfinite(values).all():
                raise ValueError(f"research-test candidate主指标不满足3-seed finite合同: {candidate}/{metric}")
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_sample_sd"] = float(np.std(values, ddof=1))
        candidate_rows.append(row)
    candidate_summary = pd.DataFrame(candidate_rows)
    return seed_summary, candidate_summary


def _git_identity() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=False, capture_output=True, text=True
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, check=False, capture_output=True, text=True
    )
    if commit.returncode != 0 or status.returncode != 0:
        raise RuntimeError("无法读取research-test Git identity")
    return commit.stdout.strip(), bool(status.stdout.strip())


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON顶层必须是mapping: {path}")
    return value


def _verify_provenance(config: ResearchTestConfig) -> tuple[dict[str, Any], dict[str, str]]:
    provenance = config.raw["provenance"]
    verified: dict[str, str] = {}
    for prefix in (
        "research_test_protocol",
        "primary_metrics_table",
        "validation_lock",
        "validation_summary_receipt",
        "validation_summary_manifest",
    ):
        path = _repo_path(provenance[f"{prefix}_path"], context=prefix)
        actual = sha256_file(path)
        expected = provenance[f"{prefix}_sha256"]
        if actual != expected:
            raise RuntimeError(f"research-test provenance hash漂移: {path}: {actual} != {expected}")
        verified[f"{prefix}_sha256"] = actual
    lock = _read_json(_repo_path(provenance["validation_lock_path"], context="validation lock"))
    if (
        lock.get("status") != "confirmed_and_closed"
        or not bool(lock.get("user_confirmation", {}).get("confirmed"))
        or lock.get("quality_lock", {}).get("pareto_set") != ["rtm_v1_multiscale_10_2_1_h384"]
        or not bool(lock.get("stop_and_closure", {}).get("protocol_closed"))
        or bool(lock.get("stop_and_closure", {}).get("research_test_opened"))
    ):
        raise RuntimeError("validation lock identity/closure漂移")
    summary_receipt = _read_json(
        _repo_path(provenance["validation_summary_receipt_path"], context="validation summary receipt")
    )
    validate_summary_receipt(summary_receipt)
    if (
        summary_receipt.get("status") != "complete"
        or summary_receipt.get("counts", {}).get("actual_formal_runs") != 15
        or summary_receipt.get("decision", {}).get("quality_pareto_set")
        != ["rtm_v1_multiscale_10_2_1_h384"]
        or bool(summary_receipt.get("decision", {}).get("research_test_opened"))
    ):
        raise RuntimeError("validation summary receipt不支持冻结research-test输入")
    return summary_receipt, verified


def _artifact_by_name(receipt: Mapping[str, Any], filename: str) -> Mapping[str, Any]:
    matches = [record for record in receipt["artifacts"] if record.get("filename") == filename]
    if len(matches) != 1:
        raise RuntimeError(f"formal receipt artifact未唯一锁定: {filename}")
    return matches[0]


def _collect_checkpoint_inputs(
    config: ResearchTestConfig, summary_receipt: Mapping[str, Any]
) -> list[dict[str, Any]]:
    formal_root = _repo_path(config.raw["provenance"]["formal_root"], context="formal root")
    summary_inputs = summary_receipt["inputs"]
    records: list[dict[str, Any]] = []
    for candidate in EXPECTED_CANDIDATES:
        for seed in FORMAL_SEEDS:
            run_dir = formal_root / candidate / f"seed_{seed}"
            receipt_path = run_dir / "formal_receipt.json"
            expected_relative = str(receipt_path.relative_to(REPO_ROOT))
            matches = [
                item
                for item in summary_inputs
                if item["candidate_id"] == candidate and int(item["seed"]) == int(seed)
            ]
            if len(matches) != 1 or matches[0]["formal_receipt_path"] != expected_relative:
                raise RuntimeError("validation summary未唯一锁定formal candidate/seed receipt")
            summary_input = matches[0]
            receipt_sha = sha256_file(receipt_path)
            if receipt_sha != summary_input["formal_receipt_sha256"]:
                raise RuntimeError(f"formal receipt SHA-256漂移: {candidate}/{seed}")
            formal_receipt = _read_json(receipt_path)
            validate_formal_receipt(formal_receipt)
            if (
                formal_receipt["candidate_id"] != candidate
                or int(formal_receipt["seed"]) != int(seed)
                or formal_receipt["execution"]["git_commit"]
                != config.raw["provenance"]["formal_execution_git_commit"]
            ):
                raise RuntimeError("formal receipt candidate/seed/commit漂移")
            paths_and_records: dict[str, tuple[Path, Mapping[str, Any]]] = {}
            for filename in ("config.yaml", "checkpoint_best_local_rr.pt", "artifact_manifest.json"):
                path = run_dir / filename
                artifact = _artifact_by_name(formal_receipt, filename)
                if sha256_file(path) != artifact["sha256"]:
                    raise RuntimeError(f"formal artifact SHA-256漂移: {candidate}/{seed}/{filename}")
                paths_and_records[filename] = (path, artifact)
            if paths_and_records["artifact_manifest.json"][1]["sha256"] != summary_input["artifact_manifest_sha256"]:
                raise RuntimeError("summary/formal artifact manifest SHA-256链不闭合")
            lifecycle = _read_json(run_dir / "lifecycle.json")
            if lifecycle.get("status") != "complete" or lifecycle.get("formal_receipt_sha256") != receipt_sha:
                raise RuntimeError("formal lifecycle/receipt不闭合")
            records.append(
                {
                    "candidate_id": candidate,
                    "seed": int(seed),
                    "formal_receipt_path": expected_relative,
                    "formal_receipt_sha256": receipt_sha,
                    "formal_artifact_manifest_sha256": summary_input["artifact_manifest_sha256"],
                    "config_path": str(paths_and_records["config.yaml"][0].relative_to(REPO_ROOT)),
                    "config_sha256": paths_and_records["config.yaml"][1]["sha256"],
                    "checkpoint_path": str(
                        paths_and_records["checkpoint_best_local_rr.pt"][0].relative_to(REPO_ROOT)
                    ),
                    "checkpoint_sha256": paths_and_records["checkpoint_best_local_rr.pt"][1]["sha256"],
                    "selected_epoch": int(formal_receipt["selector"]["best_epoch"]),
                    "selector": "validation_local_rr_mae_full_split_strict_less_than",
                }
            )
    if len(records) != 15:
        raise RuntimeError("research-test checkpoint input count不等于15")
    return records


def _derive_evaluation_config(record: Mapping[str, Any], config: ResearchTestConfig) -> DictConfig:
    config_path = _repo_path(record["config_path"], context="formal resolved config")
    if sha256_file(config_path) != record["config_sha256"]:
        raise RuntimeError("formal resolved config在评价前发生漂移")
    cfg = OmegaConf.load(config_path)
    if (
        str(cfg.model.variant) != record["candidate_id"]
        or int(cfg.training.seed) != int(record["seed"])
        or str(cfg.data.dataset_root) != config.raw["data"]["dataset_root"]
        or str(cfg.data.index_csv) != config.raw["data"]["index_csv"]
        or str(cfg.data.bcg_input_key) != config.raw["data"]["input_key"]
        or str(cfg.data.target_key) != config.raw["data"]["target_key"]
    ):
        raise RuntimeError("formal resolved config与research-test identity不一致")
    cfg.protocol.stage = "research_test"
    cfg.protocol.run_role = "reused_research_test_evaluation"
    cfg.protocol.formal_training_enabled = False
    cfg.protocol.validation_evaluation_enabled = False
    cfg.protocol.research_test_enabled = True
    cfg.training.device = str(config.raw["evaluation"]["device"])
    cfg.training.batch_size = int(config.raw["evaluation"]["batch_size"])
    cfg.training.use_amp = bool(config.raw["evaluation"]["use_amp"])
    cfg.training.num_workers = 0
    cfg.training.persistent_workers = False
    cfg.training.prefetch_factor = None
    cfg.formal.research_test_allowed = True
    cfg.research_test = OmegaConf.create(
        {
            "protocol_id": PROTOCOL_ID,
            "config_sha256": config.sha256,
            "split": "test",
            "checkpoint_path": record["checkpoint_path"],
            "checkpoint_sha256": record["checkpoint_sha256"],
            "model_training_allowed": False,
            "checkpoint_reselection_allowed": False,
        }
    )
    return cfg


def _data_signature(cfg: DictConfig) -> str:
    payload = {
        "data": OmegaConf.to_container(cfg.data, resolve=True),
        "window": OmegaConf.to_container(cfg.window, resolve=True),
        "loss": OmegaConf.to_container(cfg.loss, resolve=True),
        "evaluation": OmegaConf.to_container(cfg.evaluation, resolve=True),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _build_test_data(cfg: DictConfig, config: ResearchTestConfig) -> tuple[WindowDataBundle, dict[str, Any]]:
    contract = config.raw["data"]
    index_path = Path(contract["dataset_root"]) / contract["index_csv"]
    if not index_path.is_file() or sha256_file(index_path) != contract["dataset_index_sha256"]:
        raise RuntimeError("research-test dataset index不存在或SHA-256漂移")
    raw_index = pd.read_csv(index_path)
    if "split" not in raw_index or "dataset_row_id" not in raw_index:
        raise ValueError("research-test shared index缺少split/dataset_row_id")
    split_rows = raw_index.loc[raw_index["split"].astype(str).eq("test")].copy()
    non_test_ids = set(raw_index.loc[~raw_index["split"].astype(str).eq("test"), "dataset_row_id"].astype(int))
    if set(split_rows["dataset_row_id"].astype(int)) & non_test_ids:
        raise RuntimeError("research-test dataset_row_id与其他split重叠")
    del raw_index
    audited = adapt_research_v2_index(split_rows, cfg)
    data = build_window_data(
        cfg,
        split="test",
        max_windows=None,
        sample_strategy=str(contract["sample_strategy"]),
        sample_seed=int(contract["sample_seed"]),
        shuffle=False,
        audited=audited,
    )
    row_hash = _row_ids_sha256(data.rows)
    receipt = {
        "dataset_index_path": str(index_path),
        "dataset_index_sha256": sha256_file(index_path),
        "split": "test",
        "test_windows": len(data.dataset),
        "test_batches": len(data.loader),
        "test_samp_ids": int(data.rows["samp_id"].nunique()),
        "test_row_ids_sha256": row_hash,
        "sample_strategy": str(contract["sample_strategy"]),
        "sample_seed": int(contract["sample_seed"]),
        "row_id_overlap_with_non_test": 0,
        "signal_splits_accessed": ["test"],
    }
    if receipt != {
        "dataset_index_path": str(index_path),
        "dataset_index_sha256": contract["dataset_index_sha256"],
        "split": "test",
        "test_windows": EXPECTED_TEST_WINDOWS,
        "test_batches": math.ceil(EXPECTED_TEST_WINDOWS / int(config.raw["evaluation"]["batch_size"])),
        "test_samp_ids": EXPECTED_TEST_SAMP_IDS,
        "test_row_ids_sha256": EXPECTED_TEST_ROW_HASH,
        "sample_strategy": "stratified_random",
        "sample_seed": 20260612,
        "row_id_overlap_with_non_test": 0,
        "signal_splits_accessed": ["test"],
    }:
        raise RuntimeError(f"research-test data count/hash/split漂移: {receipt}")
    return data, receipt


def _validate_run_metrics(metrics: pd.DataFrame, *, candidate: str, seed: int) -> None:
    expected_columns = ("candidate_id", "seed", *EXPECTED_METRICS_COLUMNS)
    if tuple(metrics.columns) != expected_columns:
        raise RuntimeError("research-test metrics schema漂移")
    if (
        len(metrics) != EXPECTED_TEST_WINDOWS
        or metrics["candidate_id"].astype(str).ne(candidate).any()
        or metrics["seed"].astype(int).ne(int(seed)).any()
        or metrics["evaluation_split"].astype(str).ne("research_test").any()
        or metrics["split"].astype(str).ne("test").any()
        or _row_ids_sha256(metrics) != EXPECTED_TEST_ROW_HASH
    ):
        raise RuntimeError("research-test per-checkpoint metrics identity漂移")
    numeric = metrics.select_dtypes(include=[np.number, "bool"]).to_numpy(dtype=np.float64, copy=True)
    if np.isinf(numeric).any():
        raise FloatingPointError("research-test metrics包含Inf/-Inf")


def _frame_finite_counts(frame: pd.DataFrame) -> dict[str, int]:
    numeric = frame.select_dtypes(include=[np.number, "bool"]).to_numpy(dtype=np.float64, copy=True)
    null = np.isnan(numeric)
    nonfinite = ~np.isfinite(numeric) & ~null
    return {
        "numeric_total": int(numeric.size),
        "numeric_finite": int(np.isfinite(numeric).sum()),
        "numeric_null": int(null.sum()),
        "numeric_nonfinite": int(nonfinite.sum()),
    }


def _finite_audit(frames: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    components = {name: _frame_finite_counts(frame) for name, frame in frames.items()}
    aggregate = {
        key: sum(record[key] for record in components.values())
        for key in ("numeric_total", "numeric_finite", "numeric_null", "numeric_nonfinite")
    }
    if aggregate["numeric_nonfinite"] != 0:
        raise FloatingPointError("research-test输出包含Inf/-Inf")
    if aggregate["numeric_total"] != aggregate["numeric_finite"] + aggregate["numeric_null"]:
        raise RuntimeError("research-test finite/null计数不闭合")
    for name in ("research_test_seed_summary", "research_test_primary_mean_sd"):
        if components[name]["numeric_null"] != 0:
            raise RuntimeError(f"research-test汇总存在null: {name}")
    return {"components": components, "aggregate": aggregate}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    _write_json(temporary, value)
    os.replace(temporary, path)


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _artifact_record(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    return {
        "filename": path.name,
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
        "rows": rows,
    }


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("numpy", "omegaconf", "pandas", "scipy", "torch", "mamba-ssm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    versions["torch_cuda"] = torch.version.cuda
    versions["cudnn"] = str(torch.backends.cudnn.version())
    return versions


def validate_research_test_receipt(receipt: Mapping[str, Any]) -> None:
    _require_exact_keys(
        receipt,
        {
            "schema_version",
            "protocol_id",
            "created_utc",
            "status",
            "evidence_label",
            "execution",
            "inputs",
            "data",
            "access",
            "counts",
            "finite_audit",
            "runtime",
            "artifacts",
            "manifest_sha256",
        },
        "research-test receipt",
    )
    if (
        receipt["schema_version"] != RECEIPT_SCHEMA_VERSION
        or receipt["protocol_id"] != PROTOCOL_ID
        or receipt["status"] != "complete"
        or receipt["evidence_label"] != EVIDENCE_LABEL
    ):
        raise ValueError("research-test receipt identity/status/evidence无效")
    execution = receipt["execution"]
    _require_exact_keys(
        execution,
        {
            "command",
            "cwd",
            "git_commit",
            "git_dirty",
            "config_path",
            "config_sha256",
            "python_version",
            "platform",
            "dependencies",
            "device",
            "provenance",
        },
        "research-test receipt.execution",
    )
    if (
        not str(execution["command"]).strip()
        or execution["cwd"] != str(REPO_ROOT)
        or len(str(execution["git_commit"])) != 40
        or bool(execution["git_dirty"])
        or execution["config_path"] != "configs/resp_temporal_v1/research_test_v1.yaml"
        or execution["config_sha256"] != FROZEN_CONFIG_SHA256
    ):
        raise ValueError("research-test execution provenance无效")
    inputs = receipt["inputs"]
    identities = [(item.get("candidate_id"), int(item.get("seed", -1))) for item in inputs]
    if identities != [(candidate, seed) for candidate in EXPECTED_CANDIDATES for seed in FORMAL_SEEDS]:
        raise ValueError("research-test checkpoint inputs identity/order漂移")
    for item in inputs:
        _require_exact_keys(
            item,
            {
                "candidate_id",
                "seed",
                "formal_receipt_path",
                "formal_receipt_sha256",
                "formal_artifact_manifest_sha256",
                "config_path",
                "config_sha256",
                "checkpoint_path",
                "checkpoint_sha256",
                "selected_epoch",
                "selector",
            },
            "research-test checkpoint input",
        )
        for key in (
            "formal_receipt_sha256",
            "formal_artifact_manifest_sha256",
            "config_sha256",
            "checkpoint_sha256",
        ):
            if not _is_sha256(item[key]):
                raise ValueError("research-test checkpoint input SHA-256无效")
        if item["selector"] != "validation_local_rr_mae_full_split_strict_less_than":
            raise ValueError("research-test checkpoint selector漂移")
    expected_data = {
        "dataset_index_path": (
            "/mnt/disk_code/marques/resp_prepare/dataset/"
            "20260620_research_v2_resp_reconstruction_stage2_1_segrobustz_bcgstagee_log1psoftz_robustconf/"
            "training/dataset_index.csv"
        ),
        "dataset_index_sha256": "f65ae6524632187c7c2795bd7b5e25434afb649b5a04fe39084b70cc59f9b04f",
        "split": "test",
        "test_windows": EXPECTED_TEST_WINDOWS,
        "test_batches": 19,
        "test_samp_ids": EXPECTED_TEST_SAMP_IDS,
        "test_row_ids_sha256": EXPECTED_TEST_ROW_HASH,
        "sample_strategy": "stratified_random",
        "sample_seed": 20260612,
        "row_id_overlap_with_non_test": 0,
        "signal_splits_accessed": ["test"],
    }
    if receipt["data"] != expected_data:
        raise ValueError("research-test receipt data identity漂移")
    if receipt["access"] != {
        "shared_index_metadata_read": True,
        "train_signal_target_accessed": False,
        "validation_signal_target_prediction_accessed": False,
        "research_test_signal_target_accessed": True,
        "research_test_prediction_generated": True,
        "formal_receipts_accessed": True,
        "validation_summary_receipt_accessed": True,
        "checkpoint_best_content_accessed": True,
        "checkpoint_final_accessed": False,
        "model_training_used": False,
        "model_inference_used": True,
        "gpu_used": True,
    }:
        raise ValueError("research-test receipt access越界")
    if receipt["counts"] != {
        "expected_checkpoints": 15,
        "completed_checkpoints": 15,
        "test_windows_per_checkpoint": EXPECTED_TEST_WINDOWS,
        "research_test_metrics_rows": EXPECTED_TOTAL_METRIC_ROWS,
        "seed_summary_rows": 15,
        "candidate_summary_rows": 5,
        "artifact_records": len(OUTPUT_ARTIFACTS),
    }:
        raise ValueError("research-test receipt count closure无效")
    finite = receipt["finite_audit"]
    _require_exact_keys(finite, {"components", "aggregate"}, "research-test finite audit")
    if set(finite["components"]) != {
        "research_test_metrics",
        "research_test_seed_summary",
        "research_test_primary_mean_sd",
    }:
        raise ValueError("research-test finite components漂移")
    for label, record in {**finite["components"], "aggregate": finite["aggregate"]}.items():
        _require_exact_keys(
            record,
            {"numeric_total", "numeric_finite", "numeric_null", "numeric_nonfinite"},
            f"research-test finite {label}",
        )
        if (
            int(record["numeric_total"]) <= 0
            or int(record["numeric_nonfinite"]) != 0
            or int(record["numeric_total"])
            != int(record["numeric_finite"]) + int(record["numeric_null"])
        ):
            raise ValueError(f"research-test finite count不闭合: {label}")
    for label in ("research_test_seed_summary", "research_test_primary_mean_sd"):
        if int(finite["components"][label]["numeric_null"]) != 0:
            raise ValueError(f"research-test summary存在null: {label}")
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
        "research-test runtime",
    )
    if runtime["device"] != "cuda:0" or any(
        not math.isfinite(float(runtime[key])) or float(runtime[key]) < 0.0
        for key in ("elapsed_seconds", "total_memory_mib", "peak_allocated_mib", "peak_reserved_mib")
    ):
        raise ValueError("research-test runtime无效")
    artifacts = receipt["artifacts"]
    if [record.get("filename") for record in artifacts] != list(OUTPUT_ARTIFACTS):
        raise ValueError("research-test artifact identity/order漂移")
    for record in artifacts:
        _require_exact_keys(record, {"filename", "sha256", "size_bytes", "rows"}, "research-test artifact")
        if not _is_sha256(record["sha256"]) or int(record["size_bytes"]) <= 0:
            raise ValueError("research-test artifact SHA-256/size无效")
    if receipt["manifest_sha256"] != artifacts[-1]["sha256"]:
        raise ValueError("research-test manifest SHA-256不闭合")


def run_research_test(*, config_path: str | Path = DEFAULT_CONFIG_PATH, command: str) -> Path:
    """一次性评价冻结15个checkpoint；长时间GPU执行只由用户手动调用。"""

    config = load_research_test_config(config_path)
    if config.output_dir.exists():
        raise FileExistsError(f"research-test输出目录禁止覆盖/resume: {config.output_dir}")
    summary_receipt, verified_provenance = _verify_provenance(config)
    commit, dirty = _git_identity()
    if dirty:
        raise RuntimeError("RTM-v1 research-test要求干净Git工作树")
    dependency_problems = check_crd_dependencies()
    if dependency_problems:
        raise RuntimeError("; ".join(dependency_problems))
    gpu_config = load_gpu_engineering_config()
    device, device_info = _cuda_identity(gpu_config.raw)
    checkpoint_inputs = _collect_checkpoint_inputs(config, summary_receipt)

    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=False, exist_ok=False)
    lifecycle_path = config.output_dir / "lifecycle.json"
    lifecycle = {
        "schema_version": "rtm-v1-research-test-lifecycle-v1",
        "protocol_id": PROTOCOL_ID,
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "completed_utc": None,
        "completed_checkpoints": 0,
        "failure": None,
    }
    _write_json_atomic(lifecycle_path, lifecycle)
    started = time.perf_counter()
    try:
        _write_json(config.output_dir / "resolved_config.json", config.raw)
        _write_json(config.output_dir / "checkpoint_inputs.json", checkpoint_inputs)
        base_cfg = _derive_evaluation_config(checkpoint_inputs[0], config)
        data, data_receipt = _build_test_data(base_cfg, config)
        _write_json(config.output_dir / "test_data_receipt.json", data_receipt)
        reference_signature = _data_signature(base_cfg)
        reference_static: pd.DataFrame | None = None
        metric_frames: list[pd.DataFrame] = []
        torch.cuda.reset_peak_memory_stats(device)
        for index, record in enumerate(checkpoint_inputs, start=1):
            candidate = str(record["candidate_id"])
            seed = int(record["seed"])
            cfg = _derive_evaluation_config(record, config)
            if _data_signature(cfg) != reference_signature:
                raise RuntimeError("15项formal config的数据/指标口径不一致")
            set_seed(seed)
            model = build_resp_temporal_model(cfg).to(device)
            if trainable_parameter_count(model) != int(cfg.model.expected_trainable_parameters):
                raise RuntimeError("research-test model parameter count漂移")
            checkpoint_path = _repo_path(record["checkpoint_path"], context="research-test checkpoint")
            if sha256_file(checkpoint_path) != record["checkpoint_sha256"]:
                raise RuntimeError("research-test checkpoint在反序列化前SHA-256漂移")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if int(checkpoint.get("epoch", -1)) != int(record["selected_epoch"]):
                raise RuntimeError("research-test checkpoint epoch不等于冻结validation selector epoch")
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            del checkpoint
            predictions = collect_predictions(
                model,
                data.loader,
                device=device,
                max_windows=len(data.dataset),
                use_amp=bool(config.raw["evaluation"]["use_amp"]),
            )
            metrics = evaluate_task_predictions(
                predictions,
                cfg,
                include_test_only=False,
                method=candidate,
            )
            metrics.insert(0, "evaluation_split", "research_test")
            metrics.insert(0, "seed", seed)
            metrics.insert(0, "candidate_id", candidate)
            _validate_run_metrics(metrics, candidate=candidate, seed=seed)
            static = metrics.loc[:, list(STATIC_VALIDATION_COLUMNS)].reset_index(drop=True)
            if reference_static is None:
                reference_static = static
            elif not static.equals(reference_static):
                raise RuntimeError("research-test target/static eligibility字段跨checkpoint不一致")
            metric_frames.append(metrics)
            del predictions, metrics, model
            torch.cuda.empty_cache()
            lifecycle["completed_checkpoints"] = index
            _write_json_atomic(lifecycle_path, lifecycle)
            print(f"RTM-v1 research-test {index}/15: {candidate} seed={seed}", flush=True)

        all_metrics = pd.concat(metric_frames, ignore_index=True)
        if len(all_metrics) != EXPECTED_TOTAL_METRIC_ROWS:
            raise RuntimeError("research-test总逐sample rows不等于34650")
        seed_summary, candidate_summary = build_primary_tables(all_metrics)
        _write_csv_atomic(all_metrics, config.output_dir / "research_test_metrics.csv")
        _write_csv_atomic(seed_summary, config.output_dir / "research_test_seed_summary.csv")
        _write_csv_atomic(candidate_summary, config.output_dir / "research_test_primary_mean_sd.csv")
        finite_audit = _finite_audit(
            {
                "research_test_metrics": all_metrics,
                "research_test_seed_summary": seed_summary,
                "research_test_primary_mean_sd": candidate_summary,
            }
        )
        torch.cuda.synchronize(device)
        properties = torch.cuda.get_device_properties(device)
        runtime = {
            "elapsed_seconds": float(time.perf_counter() - started),
            "device": str(device),
            "device_name": properties.name,
            "total_memory_mib": float(properties.total_memory / (1024**2)),
            "peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / (1024**2)),
            "peak_reserved_mib": float(torch.cuda.max_memory_reserved(device) / (1024**2)),
        }
        _write_json(config.output_dir / "runtime_summary.json", runtime)
        artifact_specs = (
            ("resolved_config.json", None),
            ("checkpoint_inputs.json", 15),
            ("test_data_receipt.json", None),
            ("research_test_metrics.csv", len(all_metrics)),
            ("research_test_seed_summary.csv", len(seed_summary)),
            ("research_test_primary_mean_sd.csv", len(candidate_summary)),
            ("runtime_summary.json", None),
        )
        artifacts = [_artifact_record(config.output_dir / name, rows=rows) for name, rows in artifact_specs]
        manifest = {
            "schema_version": "rtm-v1-research-test-artifact-manifest-v1",
            "protocol_id": PROTOCOL_ID,
            "artifacts": artifacts,
        }
        _write_json(config.output_dir / "artifact_manifest.json", manifest)
        manifest_record = _artifact_record(config.output_dir / "artifact_manifest.json")
        artifacts.append(manifest_record)
        receipt = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "evidence_label": EVIDENCE_LABEL,
            "execution": {
                "command": command,
                "cwd": str(REPO_ROOT),
                "git_commit": commit,
                "git_dirty": False,
                "config_path": str(config.path.relative_to(REPO_ROOT)),
                "config_sha256": config.sha256,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "dependencies": _dependency_versions(),
                "device": device_info,
                "provenance": verified_provenance,
            },
            "inputs": checkpoint_inputs,
            "data": data_receipt,
            "access": {
                "shared_index_metadata_read": True,
                "train_signal_target_accessed": False,
                "validation_signal_target_prediction_accessed": False,
                "research_test_signal_target_accessed": True,
                "research_test_prediction_generated": True,
                "formal_receipts_accessed": True,
                "validation_summary_receipt_accessed": True,
                "checkpoint_best_content_accessed": True,
                "checkpoint_final_accessed": False,
                "model_training_used": False,
                "model_inference_used": True,
                "gpu_used": True,
            },
            "counts": {
                "expected_checkpoints": 15,
                "completed_checkpoints": 15,
                "test_windows_per_checkpoint": EXPECTED_TEST_WINDOWS,
                "research_test_metrics_rows": len(all_metrics),
                "seed_summary_rows": len(seed_summary),
                "candidate_summary_rows": len(candidate_summary),
                "artifact_records": len(artifacts),
            },
            "finite_audit": finite_audit,
            "runtime": runtime,
            "artifacts": artifacts,
            "manifest_sha256": manifest_record["sha256"],
        }
        validate_research_test_receipt(receipt)
        receipt_path = config.output_dir / "access_receipt.json"
        _write_json_atomic(receipt_path, receipt)
        receipt_hash = sha256_file(receipt_path)
        (config.output_dir / "access_receipt.sha256").write_text(
            f"{receipt_hash}  access_receipt.json\n", encoding="utf-8"
        )
        lifecycle.update(
            {
                "status": "complete",
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "access_receipt_sha256": receipt_hash,
            }
        )
        _write_json_atomic(lifecycle_path, lifecycle)
        return receipt_path
    except BaseException as exc:
        lifecycle.update(
            {
                "status": "failed",
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "failure": {"error_type": type(exc).__name__, "message": str(exc)[:2000]},
            }
        )
        _write_json_atomic(lifecycle_path, lifecycle)
        raise
