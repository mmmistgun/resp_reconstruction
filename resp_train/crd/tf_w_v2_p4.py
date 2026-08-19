from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from resp_train.crd.config import FORMAL_SEEDS, load_crd_config
from resp_train.crd.tf_w_v2 import (
    CANDIDATE_LOCK,
    CANDIDATE_LOCK_SHA256,
    P1_VARIANTS,
    P3_VARIANT,
    SOURCE_CACHE_MANIFEST_SHA256,
    SOURCE_CACHE_ROOT,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FORMAL_ROOT = REPO_ROOT / "runs/crd_tf_w_v2/formal"
OUTPUT_ROOT = REPO_ROOT / "runs/crd_tf_w_v2/p4_validation_summary"
P1_IMPLEMENTATION_LOCK = REPO_ROOT / "docs/experiments/crd_tf_w_v2_p1_implementation_lock_20260817.json"
P1_IMPLEMENTATION_LOCK_SHA256 = "fb822ca7f8607e45443e07a94d91150bcc108217fb25c47f0b4504fbdf644f58"
P3_IMPLEMENTATION_LOCK = REPO_ROOT / "docs/experiments/crd_tf_w_v2_p3_implementation_lock_20260818.json"
P3_IMPLEMENTATION_LOCK_SHA256 = "0aa2f520a52a667640ea6550a6d26c48b0f65dd088de6b4616938705a7e40da7"
P1_FORMAL_COMMIT = "1d1b22edc6f1b9b96459c1b05eddf39208041162"
P3_FORMAL_COMMIT = "6c4f6229eda6eb72c82e4cd17571bdc73bd97d54"
PROTOCOL = "crd-tf-w-v2-research-informed-20260817"
EXPECTED_VALIDATION_SAMPLES = 2675
EXPECTED_UPDATES = 6400

W0 = "W0_FULL_12V_FILM_D6"
W1 = "W1_RESP_12V_FILM_D6"
W2 = "W2_CARRIER_12V_FILM_D6"
W3 = "W3_FULL_6V_FILM_D6"
D4 = "D4_W0"
CANDIDATE_ORDER = (W0, W1, W2, W3, D4)
NEW_CANDIDATES = (W1, W2, W3, D4)
MODEL_VARIANTS = {
    W0: "crd_tf102_w",
    W1: P1_VARIANTS[0],
    W2: P1_VARIANTS[1],
    W3: P1_VARIANTS[2],
    D4: P3_VARIANT,
}
SCIENTIFIC_ROLES = {
    W0: "anchor",
    W1: "mechanism_resp",
    W2: "mechanism_carrier",
    W3: "quality_and_scale_efficiency_candidate",
    D4: "depth_efficiency_candidate",
}
TRAINABLE_PARAMETERS = {W0: 1_219_850, W1: 1_219_850, W2: 1_219_850, W3: 1_219_850, D4: 902_722}
MODEL_INPUT_ELEMENTS = {W0: 34_920, W1: 34_920, W2: 34_920, W3: 17_640, D4: 34_920}
EXPECTED_W_VIEW = {W1: "resp", W2: "carrier", W3: "full_6v", D4: "full_12v"}
EXPECTED_ACTIVE_SCALES = {W1: 56, W2: 41, W3: 49, D4: 97}
EXPECTED_MODEL_INPUT_SHAPES = {W1: [97, 360], W2: [97, 360], W3: [49, 360], D4: [97, 360]}

ERROR_METRICS = (
    "whole_rr_abs_error_bpm_mean",
    "local_rr_mae_bpm_mean",
    "envelope_trajectory_mae_mean",
    "global_envelope_modulation_error_mean",
)
PCC_METRIC = "lag_aware_signed_pcc_mean"
PRIMARY_METRICS = (*ERROR_METRICS, PCC_METRIC)
SAMPLE_COLUMNS = {
    "whole_rr_abs_error_bpm_mean": "whole_rr_abs_error_bpm",
    "local_rr_mae_bpm_mean": "local_rr_mae_bpm",
    "envelope_trajectory_mae_mean": "envelope_trajectory_mae",
    "global_envelope_modulation_error_mean": "global_envelope_modulation_error",
    "lag_aware_signed_pcc_mean": "lag_aware_signed_pcc",
}
REQUIRED_RUN_FILES = (
    "audit.csv",
    "checkpoint_best_local_rr.pt",
    "checkpoint_final.pt",
    "config.yaml",
    "metrics.csv",
    "metrics_summary.csv",
    "optimizer_parameter_groups.json",
    "run_manifest.json",
    "runtime_summary.json",
    "train_history.csv",
)
REQUIRED_OUTPUT_FILES = (
    "formal_run_audit.csv",
    "seed_metrics.csv",
    "candidate_eligibility.csv",
    "pareto.json",
    "p4_summary.json",
    "p4_summary_manifest.json",
)


def generate_p4_summary(candidate_lock: str | Path = CANDIDATE_LOCK) -> Path:
    """一次性审计 W0/W1/W2/W3/D4，并冻结严格候选池与描述性 trade-off Pareto。"""

    commit = _require_clean_git()
    candidate_lock_path = Path(candidate_lock).resolve()
    if candidate_lock_path != CANDIDATE_LOCK.resolve():
        raise ValueError(f"P4 只接受冻结 candidate lock: {CANDIDATE_LOCK}")
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"P4 summary 已存在，拒绝覆盖: {OUTPUT_ROOT}")
    lock = _verify_source_locks()
    formal_audit, seed_metrics = audit_formal_matrix(lock)
    eligibility = build_candidate_eligibility(seed_metrics, lock["summary_schema"])
    pareto = build_pareto_summary(eligibility)
    summary = _build_summary(commit, formal_audit, eligibility, pareto)

    OUTPUT_ROOT.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".p4_validation_summary_", dir=OUTPUT_ROOT.parent))
    try:
        formal_audit.to_csv(temporary / "formal_run_audit.csv", index=False)
        seed_metrics.to_csv(temporary / "seed_metrics.csv", index=False)
        eligibility.to_csv(temporary / "candidate_eligibility.csv", index=False)
        (temporary / "pareto.json").write_text(
            json.dumps(pareto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (temporary / "p4_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        generated = sorted(path for path in temporary.iterdir() if path.is_file())
        manifest = {
            "protocol": PROTOCOL,
            "phase": "p4_validation_summary_manifest",
            "status": "passed",
            "complete": True,
            "git_commit": commit,
            "git_dirty": False,
            "research_test_used": False,
            "samp_id_analysis_used": False,
            "candidate_lock_sha256": CANDIDATE_LOCK_SHA256,
            "source_artifact_set_sha256": _artifact_set_sha256(formal_audit),
            "files": {
                path.name: {"sha256": _sha256_file(path), "size_bytes": path.stat().st_size}
                for path in generated
            },
        }
        (temporary / "p4_summary_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if tuple(sorted(path.name for path in temporary.iterdir() if path.is_file())) != tuple(
            sorted(REQUIRED_OUTPUT_FILES)
        ):
            raise RuntimeError("P4 输出文件集合不符合冻结 schema")
        os.replace(temporary, OUTPUT_ROOT)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return OUTPUT_ROOT / "p4_summary.json"


def audit_formal_matrix(lock: Mapping[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """只读复核 15 个 seed-level artifacts；不创建 P4 输出。"""

    source_lock = dict(lock) if lock is not None else _verify_source_locks()
    rows: list[dict[str, Any]] = []
    w0_anchor = next(item for item in source_lock["anchors"] if item["role"] == "w0_anchor")
    if w0_anchor["variant"] != MODEL_VARIANTS[W0]:
        raise RuntimeError("P4 W0 anchor variant 漂移")
    for entry in w0_anchor["checkpoints"]:
        run_dir = REPO_ROOT / str(entry["run_dir"])
        locked_files = {
            "checkpoint_best_local_rr.pt": entry["checkpoint"],
            "config.yaml": entry["config"],
            "run_manifest.json": entry["manifest"],
            "metrics_summary.csv": entry["validation_summary"],
        }
        for filename, metadata in locked_files.items():
            path = run_dir / filename
            if path.stat().st_size != int(metadata["size_bytes"]) or _sha256_file(path) != metadata["sha256"]:
                raise RuntimeError(f"P4 W0 candidate-lock artifact 漂移: {path}")
        rows.append(
            _audit_run(
                candidate_id=W0,
                run_dir=run_dir,
                seed=int(entry["seed"]),
                expected_commit=str(w0_anchor["training_commit"]),
                expected_selected_epoch=int(entry["selected_epoch"]),
                expected_gate="p4_formal",
                contract_key=None,
            )
        )

    for candidate_id in (W1, W2, W3):
        rows.extend(
            _audit_new_candidate(
                candidate_id=candidate_id,
                expected_commit=P1_FORMAL_COMMIT,
                expected_gate="p1_formal",
                contract_key="tf_w_v2_p1_contract",
            )
        )
    rows.extend(
        _audit_new_candidate(
            candidate_id=D4,
            expected_commit=P3_FORMAL_COMMIT,
            expected_gate="p3_formal",
            contract_key="tf_w_v2_p3_contract",
        )
    )
    formal_audit = pd.DataFrame(rows).sort_values(["candidate_order", "seed"]).reset_index(drop=True)
    if len(formal_audit) != len(CANDIDATE_ORDER) * len(FORMAL_SEEDS):
        raise RuntimeError("P4 formal matrix 行数不完整")
    for candidate_id in CANDIDATE_ORDER:
        observed = formal_audit.loc[formal_audit["candidate_id"] == candidate_id, "seed"].astype(int).tolist()
        if observed != list(FORMAL_SEEDS):
            raise RuntimeError(f"P4 {candidate_id} seeds 不完整: {observed}")
    seed_columns = [
        "candidate_order",
        "candidate_id",
        "model_variant",
        "scientific_role",
        "seed",
        "run_dir",
        "selected_epoch",
        *PRIMARY_METRICS,
        "median_train_samples_per_second",
        "peak_allocated_mib",
        "peak_reserved_mib",
        "peak_reserved_fraction",
        "trainable_parameters",
        "model_input_elements_per_sample",
        "checkpoint_sha256",
    ]
    return formal_audit, formal_audit.loc[:, seed_columns].copy()


def _audit_new_candidate(
    *, candidate_id: str, expected_commit: str, expected_gate: str, contract_key: str
) -> list[dict[str, Any]]:
    model_variant = MODEL_VARIANTS[candidate_id]
    rows = []
    for seed in FORMAL_SEEDS:
        parent = FORMAL_ROOT / model_variant / f"seed_{seed}"
        run_dirs = sorted(path for path in parent.glob("20*") if path.is_dir()) if parent.is_dir() else []
        if len(run_dirs) != 1:
            raise RuntimeError(f"P4 要求 {candidate_id} seed={seed} 唯一 formal run，实际={len(run_dirs)}")
        rows.append(
            _audit_run(
                candidate_id=candidate_id,
                run_dir=run_dirs[0],
                seed=int(seed),
                expected_commit=expected_commit,
                expected_selected_epoch=None,
                expected_gate=expected_gate,
                contract_key=contract_key,
            )
        )
    return rows


def _audit_run(
    *,
    candidate_id: str,
    run_dir: Path,
    seed: int,
    expected_commit: str,
    expected_selected_epoch: int | None,
    expected_gate: str,
    contract_key: str | None,
) -> dict[str, Any]:
    missing = [name for name in REQUIRED_RUN_FILES if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"P4 run lifecycle 不完整: {run_dir}, missing={missing}")
    cfg = load_crd_config(run_dir / "config.yaml")
    model_variant = MODEL_VARIANTS[candidate_id]
    if (
        str(cfg.model.variant) != model_variant
        or str(cfg.protocol.run_role) != "formal"
        or str(cfg.protocol.execution_gate) != expected_gate
        or int(cfg.training.seed) != seed
        or int(cfg.model.initialization_seed) != seed
        or (int(cfg.training.epochs), int(cfg.training.batch_size), int(cfg.training.gradient_accumulation_steps))
        != (80, 128, 1)
        or cfg.training.early_stopping_enabled is not False
        or any(cfg.data.get(name) is not None for name in ("max_train_windows", "max_val_windows", "max_test_windows"))
    ):
        raise RuntimeError(f"P4 formal config identity 不合格: {run_dir}")
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("git_commit") != expected_commit
        or manifest.get("git_dirty") is not False
        or manifest.get("run_role") != "formal"
        or manifest.get("phase") != "train"
    ):
        raise RuntimeError(f"P4 formal manifest identity 不合格: {run_dir}")
    contract: Mapping[str, Any] = {}
    if contract_key is not None:
        contract = manifest.get(contract_key, {})
        if (
            manifest.get("protocol") != PROTOCOL
            or manifest.get("stage") != "tf_w_v2"
            or contract.get("candidate_lock_sha256") != CANDIDATE_LOCK_SHA256
            or contract.get("source_cache_manifest_sha256") != SOURCE_CACHE_MANIFEST_SHA256
            or contract.get("source_cache_modified") is not False
            or contract.get("target_read") is not False
            or contract.get("research_test_used") is not False
            or contract.get("variant") != model_variant
            or contract.get("w_view") != EXPECTED_W_VIEW[candidate_id]
            or int(contract.get("active_scale_count", -1)) != EXPECTED_ACTIVE_SCALES[candidate_id]
            or contract.get("model_input_shape_per_sample") != EXPECTED_MODEL_INPUT_SHAPES[candidate_id]
            or int(contract.get("trainable_parameters", TRAINABLE_PARAMETERS[candidate_id]))
            != TRAINABLE_PARAMETERS[candidate_id]
        ):
            raise RuntimeError(f"P4 tf_w_v2 contract identity 不合格: {run_dir}")
        if candidate_id == D4 and (
            contract.get("w_view") != "full_12v" or int(contract.get("local_bimamba2_blocks", -1)) != 4
        ):
            raise RuntimeError(f"P4 D4 depth/full-12V identity 不合格: {run_dir}")

    history = pd.read_csv(run_dir / "train_history.csv")
    numeric_history = history.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    if (
        len(history) != 80
        or history["epoch"].astype(int).tolist() != list(range(1, 81))
        or int(history.iloc[-1]["optimizer_update"]) != EXPECTED_UPDATES
        or not np.isfinite(numeric_history).all()
    ):
        raise RuntimeError(f"P4 formal history 不完整或非有限: {run_dir}")
    selected_epoch = int(history.loc[history["val_local_rr_mae"].idxmin(), "epoch"])
    if expected_selected_epoch is not None and selected_epoch != expected_selected_epoch:
        raise RuntimeError(f"P4 locked selected epoch 漂移: {run_dir}")
    checkpoint_hashes: dict[str, str] = {}
    for filename, epoch, update in (
        ("checkpoint_best_local_rr.pt", selected_epoch, selected_epoch * 80),
        ("checkpoint_final.pt", 80, EXPECTED_UPDATES),
    ):
        checkpoint = torch.load(run_dir / filename, map_location="cpu", weights_only=False)
        extra = checkpoint.get("extra_state", {})
        tensors = tuple(_iter_tensors(checkpoint.get("model_state_dict"))) + tuple(
            _iter_tensors(checkpoint.get("optimizer_state_dict"))
        )
        if (
            int(checkpoint.get("epoch", -1)) != epoch
            or int(extra.get("update_index", -1)) != update
            or int(extra.get("total_updates", -1)) != EXPECTED_UPDATES
            or not tensors
            or not all(bool(torch.isfinite(tensor).all()) for tensor in tensors)
        ):
            raise RuntimeError(f"P4 checkpoint 不合格: {run_dir / filename}")
        checkpoint_hashes[filename] = _sha256_file(run_dir / filename)

    metrics_frame = pd.read_csv(run_dir / "metrics.csv")
    summary = pd.read_csv(run_dir / "metrics_summary.csv").iloc[0]
    if len(metrics_frame) != EXPECTED_VALIDATION_SAMPLES or int(summary["n_samples"]) != EXPECTED_VALIDATION_SAMPLES:
        raise RuntimeError(f"P4 validation sample count 错误: {run_dir}")
    metrics = {name: float(summary[name]) for name in PRIMARY_METRICS}
    if not np.isfinite(list(metrics.values())).all() or float(summary["joint_prediction_degenerate_fraction"]) != 0.0:
        raise RuntimeError(f"P4 primary 非有限或 degeneracy 非零: {run_dir}")
    if bool(metrics_frame["joint_prediction_degenerate"].astype(bool).any()):
        raise RuntimeError(f"P4 per-sample degeneracy 非零: {run_dir}")
    for metric, sample_column in SAMPLE_COLUMNS.items():
        values = metrics_frame[sample_column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or not np.isclose(
            float(np.mean(values)), metrics[metric], rtol=0.0, atol=1e-12
        ):
            raise RuntimeError(f"P4 per-sample direct mean 不一致: {run_dir}, metric={metric}")

    runtime = json.loads((run_dir / "runtime_summary.json").read_text(encoding="utf-8"))
    peak_allocated = float(runtime["peak_allocated_mib"])
    peak_reserved = float(runtime["peak_reserved_mib"])
    total_memory = float(runtime["total_device_memory_mib"])
    peak_fraction = float(runtime["peak_reserved_fraction"])
    if (
        not np.isfinite([peak_allocated, peak_reserved, total_memory, peak_fraction]).all()
        or min(peak_allocated, peak_reserved, total_memory, peak_fraction) <= 0.0
        or not np.isclose(peak_reserved / total_memory, peak_fraction, rtol=0.0, atol=1e-12)
    ):
        raise RuntimeError(f"P4 runtime summary 不合格: {run_dir}")
    median_sps = float(history.loc[history["epoch"].astype(int) >= 2, "train_samples_per_second"].median())
    artifact_hashes = {name: _sha256_file(run_dir / name) for name in REQUIRED_RUN_FILES}
    return {
        "candidate_order": CANDIDATE_ORDER.index(candidate_id),
        "candidate_id": candidate_id,
        "model_variant": model_variant,
        "scientific_role": SCIENTIFIC_ROLES[candidate_id],
        "seed": seed,
        "run_dir": str(run_dir.resolve()),
        "training_commit": expected_commit,
        "selected_epoch": selected_epoch,
        "lifecycle_complete": True,
        "all_finite": True,
        "prediction_degenerate_fraction": 0.0,
        "research_test_used": False,
        "samp_id_analysis_used": False,
        "trainable_parameters": TRAINABLE_PARAMETERS[candidate_id],
        "model_input_elements_per_sample": MODEL_INPUT_ELEMENTS[candidate_id],
        "median_train_samples_per_second": median_sps,
        "peak_allocated_mib": peak_allocated,
        "peak_reserved_mib": peak_reserved,
        "peak_reserved_fraction": peak_fraction,
        "checkpoint_sha256": checkpoint_hashes["checkpoint_best_local_rr.pt"],
        "final_checkpoint_sha256": checkpoint_hashes["checkpoint_final.pt"],
        "artifact_set_sha256": _hash_mapping(artifact_hashes),
        **{f"{name}_sha256": digest for name, digest in artifact_hashes.items()},
        **metrics,
    }


def build_candidate_eligibility(seed_metrics: pd.DataFrame, summary_schema: Mapping[str, Any]) -> pd.DataFrame:
    """应用冻结 hard gates；同时标记非灾难性的工程 trade-off，二者绝不混写。"""

    thresholds = summary_schema["quality_thresholds"]
    efficiency = summary_schema["efficiency_thresholds"]
    catastrophic = summary_schema["catastrophic_thresholds"]
    groups = {
        candidate_id: seed_metrics.loc[seed_metrics["candidate_id"] == candidate_id].sort_values("seed")
        for candidate_id in CANDIDATE_ORDER
    }
    for candidate_id, group in groups.items():
        if group["seed"].astype(int).tolist() != list(FORMAL_SEEDS):
            raise RuntimeError(f"P4 eligibility seed identity 不完整: {candidate_id}")
    anchor = groups[W0]
    anchor_means = anchor.loc[:, PRIMARY_METRICS].mean()
    anchor_sps = float(anchor["median_train_samples_per_second"].mean())
    anchor_allocated = float(anchor["peak_allocated_mib"].mean())
    anchor_parameters = float(anchor["trainable_parameters"].iloc[0])
    anchor_elements = float(anchor["model_input_elements_per_sample"].iloc[0])
    rows: list[dict[str, Any]] = []
    for candidate_id in CANDIDATE_ORDER:
        group = groups[candidate_id]
        means = group.loc[:, PRIMARY_METRICS].mean()
        sample_sd = group.loc[:, PRIMARY_METRICS].std(ddof=1)
        deltas = {metric: float(means[metric] / anchor_means[metric] - 1.0) for metric in ERROR_METRICS}
        pcc_delta = float(means[PCC_METRIC] - anchor_means[PCC_METRIC])
        better_counts = {
            metric: int((group[metric].to_numpy(dtype=np.float64) < anchor[metric].to_numpy(dtype=np.float64)).sum())
            for metric in ERROR_METRICS
        }
        better_counts[PCC_METRIC] = int(
            (group[PCC_METRIC].to_numpy(dtype=np.float64) > anchor[PCC_METRIC].to_numpy(dtype=np.float64)).sum()
        )
        material = [
            metric
            for metric in ERROR_METRICS
            if deltas[metric] <= -float(thresholds["material_error_relative_improvement_min"])
            and better_counts[metric] >= int(summary_schema["paired_seed_direction_required_for_claimed_quality_improvement"])
        ]
        if (
            pcc_delta >= float(thresholds["material_pcc_absolute_improvement_min"])
            and better_counts[PCC_METRIC]
            >= int(summary_schema["paired_seed_direction_required_for_claimed_quality_improvement"])
        ):
            material.append(PCC_METRIC)
        quality_guardrails = bool(
            deltas["local_rr_mae_bpm_mean"] <= float(thresholds["local_rr_relative_worsening_max"])
            and deltas["whole_rr_abs_error_bpm_mean"] <= float(thresholds["other_error_relative_worsening_max"])
            and deltas["envelope_trajectory_mae_mean"] <= float(thresholds["other_error_relative_worsening_max"])
            and deltas["global_envelope_modulation_error_mean"]
            <= float(thresholds["other_error_relative_worsening_max"])
            and pcc_delta >= -float(thresholds["pcc_absolute_drop_max"])
        )
        catastrophic_failure = bool(
            any(delta > float(catastrophic["any_error_relative_worsening"]) for delta in deltas.values())
            or pcc_delta < -float(catastrophic["pcc_absolute_drop"])
        )
        quality_eligible = bool(candidate_id != W0 and material and quality_guardrails and not catastrophic_failure)

        parameters = float(group["trainable_parameters"].iloc[0])
        elements = float(group["model_input_elements_per_sample"].iloc[0])
        parameter_reduction = 1.0 - parameters / anchor_parameters
        element_reduction = 1.0 - elements / anchor_elements
        structural_reduction = max(parameter_reduction, element_reduction)
        structural_pass = bool(
            (candidate_id == W3 and element_reduction >= 0.40)
            or (candidate_id == D4 and parameter_reduction >= 0.20)
        )
        throughput = float(group["median_train_samples_per_second"].mean())
        allocated = float(group["peak_allocated_mib"].mean())
        throughput_gain = throughput / anchor_sps - 1.0
        allocated_reduction = 1.0 - allocated / anchor_allocated
        resource_pass = bool(
            throughput_gain >= float(efficiency["throughput_relative_improvement_min"])
            or allocated_reduction >= float(efficiency["peak_allocated_relative_reduction_min"])
        )
        efficiency_quality_guardrails = bool(
            all(delta <= float(efficiency["all_error_relative_worsening_max"]) for delta in deltas.values())
            and pcc_delta >= -float(efficiency["pcc_absolute_drop_max"])
        )
        efficiency_eligible = bool(
            candidate_id != W0
            and structural_pass
            and resource_pass
            and efficiency_quality_guardrails
            and not catastrophic_failure
        )
        descriptive_tradeoff = bool(
            candidate_id != W0
            and structural_pass
            and resource_pass
            and not catastrophic_failure
            and not efficiency_eligible
        )
        rows.append(
            {
                "candidate_order": CANDIDATE_ORDER.index(candidate_id),
                "candidate_id": candidate_id,
                "model_variant": MODEL_VARIANTS[candidate_id],
                "scientific_role": SCIENTIFIC_ROLES[candidate_id],
                **{f"{metric}_seed_mean": float(means[metric]) for metric in PRIMARY_METRICS},
                **{f"{metric}_sample_sd": float(sample_sd[metric]) for metric in PRIMARY_METRICS},
                **{f"{metric}_relative_delta": deltas[metric] for metric in ERROR_METRICS},
                "lag_aware_signed_pcc_absolute_delta": pcc_delta,
                **{f"{metric}_better_seed_count": better_counts[metric] for metric in PRIMARY_METRICS},
                "material_improvement_metrics": ";".join(material),
                "quality_guardrails_passed": quality_guardrails,
                "quality_eligible": quality_eligible,
                "efficiency_quality_guardrails_passed": efficiency_quality_guardrails,
                "trainable_parameters": int(parameters),
                "model_input_elements_per_sample": int(elements),
                "structural_reduction_fraction": structural_reduction,
                "structural_reduction_passed": structural_pass,
                "throughput_mean": throughput,
                "throughput_relative_gain": throughput_gain,
                "peak_allocated_mean_mib": allocated,
                "peak_allocated_relative_reduction": allocated_reduction,
                "resource_gate_passed": resource_pass,
                "efficiency_eligible": efficiency_eligible,
                "catastrophic_failure": catastrophic_failure,
                "descriptive_noncatastrophic_efficiency_tradeoff": descriptive_tradeoff,
                "strict_pool_qualified": bool(quality_eligible or efficiency_eligible),
            }
        )
    return pd.DataFrame(rows).sort_values("candidate_order").reset_index(drop=True)


def build_pareto_summary(eligibility: pd.DataFrame) -> dict[str, Any]:
    by_id = eligibility.set_index("candidate_id", drop=False)
    quality_pool = [candidate_id for candidate_id in NEW_CANDIDATES if bool(by_id.loc[candidate_id, "quality_eligible"])]
    efficiency_pool = [
        candidate_id for candidate_id in NEW_CANDIDATES if bool(by_id.loc[candidate_id, "efficiency_eligible"])
    ]
    tradeoffs = [
        candidate_id
        for candidate_id in NEW_CANDIDATES
        if bool(by_id.loc[candidate_id, "descriptive_noncatastrophic_efficiency_tradeoff"])
    ]
    quality_pareto = _quality_pareto(quality_pool, by_id)
    efficiency_pareto = _resource_pareto(efficiency_pool, by_id)
    descriptive_input = [
        candidate_id for candidate_id in NEW_CANDIDATES if candidate_id in set(quality_pool + efficiency_pool + tradeoffs)
    ]
    descriptive_pareto = _full_tradeoff_pareto(descriptive_input, by_id)
    quality_primary = min(
        quality_pareto,
        key=lambda name: (
            float(by_id.loc[name, "local_rr_mae_bpm_mean_seed_mean"]),
            float(by_id.loc[name, "whole_rr_abs_error_bpm_mean_seed_mean"]),
            -float(by_id.loc[name, "lag_aware_signed_pcc_mean_seed_mean"]),
        ),
    ) if quality_pareto else None
    efficiency_primary = max(
        efficiency_pareto,
        key=lambda name: (
            float(by_id.loc[name, "throughput_relative_gain"]),
            float(by_id.loc[name, "peak_allocated_relative_reduction"]),
            float(by_id.loc[name, "structural_reduction_fraction"]),
        ),
    ) if efficiency_pareto else None
    return {
        "protocol": PROTOCOL,
        "phase": "p4_pareto",
        "strict_quality_pool": quality_pool,
        "strict_efficiency_pool": efficiency_pool,
        "strict_quality_pareto": quality_pareto,
        "strict_efficiency_pareto": efficiency_pareto,
        "descriptive_noncatastrophic_efficiency_tradeoffs": tradeoffs,
        "descriptive_tradeoff_input": descriptive_input,
        "descriptive_quality_efficiency_pareto": descriptive_pareto,
        "mechanism_results_retained": [W1, W2],
        "p5_quality_primary_if_authorized": quality_primary,
        "p5_efficiency_candidate_if_authorized": efficiency_primary,
        "strict_eligibility_controls_p5_allowlist": True,
        "descriptive_pareto_does_not_override_hard_gates": True,
        "interpretation": {
            D4: (
                "non-catastrophic quality-efficiency trade-off: substantial parameter/throughput/peak-allocated "
                "benefit with mild but above-tolerance Local-RR/trajectory loss"
            ),
            W3: "strict quality candidate; scale reduction did not meet the registered runtime efficiency gate",
        },
        "no_total_score_used": True,
    }


def _quality_pareto(candidates: list[str], by_id: pd.DataFrame) -> list[str]:
    retained = []
    for candidate in candidates:
        dominated = any(
            other != candidate and _quality_tolerance_dominates(by_id.loc[other], by_id.loc[candidate])
            for other in candidates
        )
        if not dominated:
            retained.append(candidate)
    return retained


def _quality_tolerance_dominates(candidate: pd.Series, reference: pd.Series) -> bool:
    no_worse = bool(
        candidate["local_rr_mae_bpm_mean_seed_mean"] <= reference["local_rr_mae_bpm_mean_seed_mean"] * 1.005
        and candidate["whole_rr_abs_error_bpm_mean_seed_mean"]
        <= reference["whole_rr_abs_error_bpm_mean_seed_mean"] * 1.015
        and candidate["envelope_trajectory_mae_mean_seed_mean"]
        <= reference["envelope_trajectory_mae_mean_seed_mean"] * 1.015
        and candidate["global_envelope_modulation_error_mean_seed_mean"]
        <= reference["global_envelope_modulation_error_mean_seed_mean"] * 1.015
        and candidate["lag_aware_signed_pcc_mean_seed_mean"]
        >= reference["lag_aware_signed_pcc_mean_seed_mean"] - 0.003
    )
    substantive = bool(
        candidate["local_rr_mae_bpm_mean_seed_mean"] <= reference["local_rr_mae_bpm_mean_seed_mean"] * 0.995
        or candidate["whole_rr_abs_error_bpm_mean_seed_mean"]
        <= reference["whole_rr_abs_error_bpm_mean_seed_mean"] * 0.995
        or candidate["envelope_trajectory_mae_mean_seed_mean"]
        <= reference["envelope_trajectory_mae_mean_seed_mean"] * 0.995
        or candidate["global_envelope_modulation_error_mean_seed_mean"]
        <= reference["global_envelope_modulation_error_mean_seed_mean"] * 0.995
        or candidate["lag_aware_signed_pcc_mean_seed_mean"]
        >= reference["lag_aware_signed_pcc_mean_seed_mean"] + 0.002
    )
    return bool(no_worse and substantive)


def _resource_pareto(candidates: list[str], by_id: pd.DataFrame) -> list[str]:
    retained = []
    for candidate in candidates:
        row = by_id.loc[candidate]
        dominated = False
        for other in candidates:
            if other == candidate:
                continue
            other_row = by_id.loc[other]
            no_worse = bool(
                other_row["throughput_mean"] >= row["throughput_mean"]
                and other_row["peak_allocated_mean_mib"] <= row["peak_allocated_mean_mib"]
                and other_row["structural_reduction_fraction"] >= row["structural_reduction_fraction"]
            )
            strict = bool(
                other_row["throughput_mean"] > row["throughput_mean"]
                or other_row["peak_allocated_mean_mib"] < row["peak_allocated_mean_mib"]
                or other_row["structural_reduction_fraction"] > row["structural_reduction_fraction"]
            )
            dominated |= no_worse and strict
        if not dominated:
            retained.append(candidate)
    return retained


def _full_tradeoff_pareto(candidates: list[str], by_id: pd.DataFrame) -> list[str]:
    minimize = [f"{metric}_seed_mean" for metric in ERROR_METRICS] + [
        "peak_allocated_mean_mib",
        "trainable_parameters",
    ]
    maximize = [f"{PCC_METRIC}_seed_mean", "throughput_mean"]
    retained = []
    for candidate in candidates:
        row = by_id.loc[candidate]
        dominated = False
        for other in candidates:
            if other == candidate:
                continue
            other_row = by_id.loc[other]
            no_worse = all(float(other_row[column]) <= float(row[column]) for column in minimize) and all(
                float(other_row[column]) >= float(row[column]) for column in maximize
            )
            strict = any(float(other_row[column]) < float(row[column]) for column in minimize) or any(
                float(other_row[column]) > float(row[column]) for column in maximize
            )
            dominated |= no_worse and strict
        if not dominated:
            retained.append(candidate)
    return retained


def _build_summary(
    commit: str, formal_audit: pd.DataFrame, eligibility: pd.DataFrame, pareto: Mapping[str, Any]
) -> dict[str, Any]:
    selected = pareto["p5_quality_primary_if_authorized"]
    selected_rows = formal_audit.loc[formal_audit["candidate_id"] == selected] if selected else pd.DataFrame()
    allowlist = [
        {
            "candidate_id": str(row.candidate_id),
            "model_variant": str(row.model_variant),
            "seed": int(row.seed),
            "run_dir": str(row.run_dir),
            "selected_epoch": int(row.selected_epoch),
            "checkpoint_sha256": str(row.checkpoint_sha256),
        }
        for row in selected_rows.itertuples(index=False)
    ]
    d4 = eligibility.loc[eligibility["candidate_id"] == D4].iloc[0]
    return {
        "protocol": PROTOCOL,
        "phase": "p4_validation_summary",
        "status": "passed",
        "complete": True,
        "evidence_role": "validation-development; not unbiased held-out evidence",
        "git_commit": commit,
        "git_dirty": False,
        "candidate_lock_sha256": CANDIDATE_LOCK_SHA256,
        "p1_implementation_lock_sha256": P1_IMPLEMENTATION_LOCK_SHA256,
        "p3_implementation_lock_sha256": P3_IMPLEMENTATION_LOCK_SHA256,
        "source_cache_manifest_sha256": SOURCE_CACHE_MANIFEST_SHA256,
        "formal_run_count": int(len(formal_audit)),
        "candidate_count_including_anchor": int(len(CANDIDATE_ORDER)),
        "new_candidate_count": int(len(NEW_CANDIDATES)),
        "strict_quality_candidates": list(pareto["strict_quality_pool"]),
        "strict_efficiency_candidates": list(pareto["strict_efficiency_pool"]),
        "descriptive_quality_efficiency_pareto": list(pareto["descriptive_quality_efficiency_pareto"]),
        "p5_quality_primary_if_authorized": selected,
        "p5_efficiency_candidate_if_authorized": pareto["p5_efficiency_candidate_if_authorized"],
        "p5_checkpoint_allowlist_if_authorized": allowlist,
        "d4_tradeoff": {
            "strict_efficiency_eligible": bool(d4["efficiency_eligible"]),
            "descriptive_noncatastrophic_efficiency_tradeoff": bool(
                d4["descriptive_noncatastrophic_efficiency_tradeoff"]
            ),
            "parameter_reduction_fraction": float(d4["structural_reduction_fraction"]),
            "throughput_relative_gain": float(d4["throughput_relative_gain"]),
            "peak_allocated_relative_reduction": float(d4["peak_allocated_relative_reduction"]),
            "local_rr_relative_delta": float(d4["local_rr_mae_bpm_mean_relative_delta"]),
            "trajectory_relative_delta": float(d4["envelope_trajectory_mae_mean_relative_delta"]),
            "catastrophic_failure": bool(d4["catastrophic_failure"]),
            "interpretation": (
                "substantial compute reduction with mild non-catastrophic quality loss; retained descriptively, "
                "but excluded from the strict efficiency pool by preregistered quality guardrails"
            ),
        },
        "mechanism_results_retained": [W1, W2],
        "no_total_score_used": True,
        "samp_id_analysis_used": False,
        "research_test_used": False,
        "research_test_opened": False,
        "research_test_requires_new_user_authorization": True,
        "new_training_runs_remaining": 0,
    }


def _verify_source_locks() -> dict[str, Any]:
    for path, expected, label in (
        (CANDIDATE_LOCK, CANDIDATE_LOCK_SHA256, "candidate"),
        (P1_IMPLEMENTATION_LOCK, P1_IMPLEMENTATION_LOCK_SHA256, "P1 implementation"),
        (P3_IMPLEMENTATION_LOCK, P3_IMPLEMENTATION_LOCK_SHA256, "P3 implementation"),
        (SOURCE_CACHE_ROOT / "cache_manifest.json", SOURCE_CACHE_MANIFEST_SHA256, "source cache manifest"),
    ):
        if _sha256_file(path) != expected:
            raise RuntimeError(f"P4 {label} lock SHA-256 漂移: {path}")
    lock = json.loads(CANDIDATE_LOCK.read_text(encoding="utf-8"))
    if (
        lock.get("protocol") != PROTOCOL
        or lock.get("p0_outcome", {}).get("candidate_lock_complete") is not True
        or tuple(lock.get("stage_allowlist", {}).get("p1", ())) != P1_VARIANTS
        or tuple(lock.get("stage_allowlist", {}).get("p3", ())) != (P3_VARIANT,)
        or lock.get("forbidden", {}).get("samp_id_analysis") is not True
    ):
        raise RuntimeError("P4 candidate lock schema/allowlist 不合格")
    return lock


def _iter_tensors(value: Any) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for nested in value.values():
            yield from _iter_tensors(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _iter_tensors(nested)


def _artifact_set_sha256(formal_audit: pd.DataFrame) -> str:
    values = formal_audit.sort_values(["candidate_order", "seed"])["artifact_set_sha256"].astype(str).tolist()
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _hash_mapping(values: Mapping[str, str]) -> str:
    payload = "\n".join(f"{name}\t{values[name]}" for name in sorted(values))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_clean_git() -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    if dirty:
        raise RuntimeError("CRD-TF-W v2 P4 必须从干净 Git commit 生成")
    return commit


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
