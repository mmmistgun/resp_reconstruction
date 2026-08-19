from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from resp_train.crd.config import FORMAL_SEEDS, crd_dependency_versions, load_crd_config
from resp_train.crd.experiment import CRDExperiment
from resp_train.crd.tf_v1_p5 import METRIC_COLUMNS, SAMPLE_METRIC_COLUMNS
from resp_train.crd.tf_v1_research_test_data import (
    FROZEN_RESEARCH_TEST_CACHE_IDENTITY,
    FROZEN_RESEARCH_TEST_CACHE_MANIFEST_SHA256,
    FROZEN_RESEARCH_TEST_CACHE_ROOT,
)
from resp_train.crd.tf_v1_research_test_summary import SECONDARY_METRICS
from resp_train.crd.tf_v1_selection import ERROR_METRICS, PRIMARY_METRICS
from resp_train.crd.tf_w_v2 import (
    CANDIDATE_LOCK,
    CANDIDATE_LOCK_SHA256,
    P1_VARIANTS,
    PROTOCOL,
    SOURCE_CACHE_MANIFEST_SHA256,
    p1_variant_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST = REPO_ROOT / "docs/experiments/crd_tf_w_v2_p5_checkpoint_allowlist_20260819.json"
ALLOWLIST_SHA256 = "c3fe1320a8342a9580fff2864218c948c451b21c05a863db63efe269f1358b07"
P4_ROOT = REPO_ROOT / "runs/crd_tf_w_v2/p4_validation_summary"
P4_SUMMARY_SHA256 = "0f5a62448aa8db6b3bc9b07633bedf2e857b6369427792d37872851841cbd70b"
P4_MANIFEST_SHA256 = "d9f32d26fa9bf8b98ec739762a30717a881d62a32ccb44099e603fafe94b0e97"
P4_SOURCE_ARTIFACT_SET_SHA256 = "d3fad601cf82da21534621a8c1be88fdb77d4deea78a7819f8b5b6ced5751bfa"
V1_RESEARCH_SUMMARY_ROOT = REPO_ROOT / "runs/crd_tf_v1/research_test_summary"
V1_RESEARCH_SUMMARY_SHA256 = "e9430d3449e1e75cbab1804f1c887803ba8c12dcc4b11582f94090a6a1d7c6c0"
V1_RESEARCH_MANIFEST_SHA256 = "1c1a4571eaf281a2dbbbd86da633f45e9e44bed2f7d31b7b4320bd38e411e180"
V1_ACCESS_AUDIT_SHA256 = "eb75cce11e1827b983f6540719e772c1e0288e8ffda06b7f3ae917bc2920d377"
OUTPUT_ROOT = REPO_ROOT / "runs/crd_tf_w_v2/research_test"
EVALUATION_ROOT = OUTPUT_ROOT / "evaluations"
SUMMARY_ROOT = OUTPUT_ROOT / "summary"

EVIDENCE_ROLE = (
    "reused research/development evidence; research-test-informed; not untouched independent test"
)
CANDIDATE_ID = "W3_FULL_6V_FILM_D6"
CANDIDATE_VARIANT = "crd_tfw_v2_w3_full_6v_film_d6"
REUSED_REFERENCE_VARIANTS = ("crd_c201_decoder_10hz_cap", "crd_tf102_w")
EXPECTED_TEST_SAMPLES = 2310
EXPECTED_TEST_ROW_IDS_SHA256 = "184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8"
EXPECTED_STRATUM_TOTALS = {"low": 791, "medium": 957, "high": 562}
EVALUATION_FILES = (
    "research_test_metrics.csv",
    "research_test_metrics_summary.csv",
    "research_test_metrics_manifest.json",
)
SUMMARY_FILES = (
    "evaluation_audit.csv",
    "seed_metrics.csv",
    "aggregate_primary.csv",
    "aggregate_secondary.csv",
    "paired_vs_references.csv",
    "p5_summary.json",
    "p5_summary_manifest.json",
)


def evaluate_p5_research_test_seed(
    *,
    candidate_lock: str | Path = CANDIDATE_LOCK,
    allowlist_path: str | Path = ALLOWLIST,
    seed: int,
    device: str,
) -> Path:
    """评价一个冻结 W3 checkpoint；已存在且完整的 seed 仅校验后跳过。"""

    evaluation_commit = _require_clean_git("CRD-TF-W v2 P5 evaluation")
    if not str(device).startswith("cuda:"):
        raise ValueError("P5 research-test 必须显式使用 cuda:<index>")
    inputs = audit_p5_inputs(candidate_lock=candidate_lock, allowlist_path=allowlist_path)
    identities = {int(item["seed"]): item for item in inputs["checkpoint_identities"]}
    normalized_seed = int(seed)
    if normalized_seed not in identities:
        raise ValueError(f"seed 不在 P5 allowlist: {normalized_seed}")
    if SUMMARY_ROOT.exists():
        raise FileExistsError(f"P5 summary 已冻结，拒绝新增或重复 evaluation: {SUMMARY_ROOT}")

    identity = identities[normalized_seed]
    final_dir = _evaluation_dir(normalized_seed)
    if final_dir.exists():
        _audit_evaluation_output(final_dir, identity)
        return final_dir / "research_test_metrics.csv"

    final_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".seed_{normalized_seed}_", dir=final_dir.parent))
    metrics_path = temporary / "research_test_metrics.csv"
    checkpoint_path = Path(str(identity["checkpoint_path"]))
    cfg = load_crd_config(
        Path(str(identity["config_path"])),
        overrides=[f"training.device={device}", "training.show_progress=false"],
    )
    started_at = datetime.now().astimezone().isoformat()
    started = time.perf_counter()
    try:
        cuda_device = torch.device(device)
        if not torch.cuda.is_available():
            raise RuntimeError("P5 请求 CUDA，但当前环境不可用")
        torch.cuda.reset_peak_memory_stats(cuda_device)
        CRDExperiment(cfg).evaluate_checkpoint(
            checkpoint_path,
            split="test",
            metrics_output=metrics_path,
            tf_research_test_cache_path=FROZEN_RESEARCH_TEST_CACHE_ROOT,
        )
        torch.cuda.synchronize(cuda_device)
        elapsed = time.perf_counter() - started
        total_memory = float(torch.cuda.get_device_properties(cuda_device).total_memory) / float(1024**2)
        runtime = {
            "device": str(cuda_device),
            "wall_seconds": float(elapsed),
            "samples_per_second": float(EXPECTED_TEST_SAMPLES / elapsed),
            "peak_allocated_mib": float(torch.cuda.max_memory_allocated(cuda_device)) / float(1024**2),
            "peak_reserved_mib": float(torch.cuda.max_memory_reserved(cuda_device)) / float(1024**2),
            "total_device_memory_mib": total_memory,
        }
        runtime["peak_reserved_fraction"] = runtime["peak_reserved_mib"] / total_memory
        summary_path = temporary / "research_test_metrics_summary.csv"
        metrics = pd.read_csv(metrics_path)
        summary = pd.read_csv(summary_path).iloc[0]
        _validate_metrics(metrics, summary, variant=CANDIDATE_VARIANT)
        manifest = {
            "created_at": datetime.now().astimezone().isoformat(),
            "started_at": started_at,
            "command": list(sys.argv),
            "protocol": PROTOCOL,
            "phase": "tf_w_v2_p5_reused_research_test_evaluation",
            "status": "passed",
            "complete": True,
            "evidence_role": EVIDENCE_ROLE,
            "git_commit": evaluation_commit,
            "git_dirty": False,
            "research_test_used": True,
            "research_test_targets_used_for_metrics": True,
            "samp_id_analysis_used": False,
            "candidate_lock_sha256": CANDIDATE_LOCK_SHA256,
            "p5_checkpoint_allowlist_sha256": ALLOWLIST_SHA256,
            "p4_summary_sha256": P4_SUMMARY_SHA256,
            "candidate_id": CANDIDATE_ID,
            "variant": CANDIDATE_VARIANT,
            "seed": normalized_seed,
            "validation_selected_epoch": int(identity["selected_epoch"]),
            "selection_source": "frozen CRD-TF-W v2 P4 strict quality allowlist",
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": str(identity["checkpoint_sha256"]),
            "checkpoint_reselection_used": False,
            "test_w_cache_root": str(FROZEN_RESEARCH_TEST_CACHE_ROOT),
            "test_w_cache_identity_sha256": FROZEN_RESEARCH_TEST_CACHE_IDENTITY,
            "test_w_cache_manifest_sha256": FROZEN_RESEARCH_TEST_CACHE_MANIFEST_SHA256,
            "source_train_validation_w_manifest_sha256": SOURCE_CACHE_MANIFEST_SHA256,
            "test_w_cache_target_read": False,
            "w_view": "full_6v",
            "source_shape_per_sample": [97, 360],
            "model_input_shape_per_sample": [49, 360],
            "active_scale_count": 49,
            "view_index_sha256": "3734c4183c17744eb4779ac65ac8521de1897f6a12f364e913aeb02528e250d6",
            "source_cache_modified": False,
            "dataset_row_ids_sha256": EXPECTED_TEST_ROW_IDS_SHA256,
            "n_samples": EXPECTED_TEST_SAMPLES,
            "runtime": runtime,
            "dependency_versions": crd_dependency_versions(),
            "files": {
                metrics_path.name: {
                    "sha256": _sha256_file(metrics_path),
                    "size_bytes": metrics_path.stat().st_size,
                },
                summary_path.name: {
                    "sha256": _sha256_file(summary_path),
                    "size_bytes": summary_path.stat().st_size,
                },
            },
        }
        (temporary / "research_test_metrics_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        _audit_evaluation_output(temporary, identity, expected_commit=evaluation_commit)
        os.replace(temporary, final_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return final_dir / "research_test_metrics.csv"


def generate_p5_summary(
    *,
    candidate_lock: str | Path = CANDIDATE_LOCK,
    allowlist_path: str | Path = ALLOWLIST,
) -> Path:
    """冻结三项 W3 新评价与六项既有 C201/W0 证据的描述性 P5 汇总。"""

    summary_commit = _require_clean_git("CRD-TF-W v2 P5 summary")
    if SUMMARY_ROOT.exists():
        raise FileExistsError(f"P5 summary 已存在，拒绝覆盖: {SUMMARY_ROOT}")
    inputs = audit_p5_inputs(candidate_lock=candidate_lock, allowlist_path=allowlist_path)
    evaluation_rows = []
    for identity in inputs["checkpoint_identities"]:
        evaluation_rows.append(_audit_evaluation_output(_evaluation_dir(int(identity["seed"])), identity))
    evaluation_audit = pd.DataFrame(evaluation_rows).sort_values("seed").reset_index(drop=True)
    evaluation_commits = sorted(evaluation_audit["evaluation_commit"].astype(str).unique().tolist())
    if len(evaluation_commits) != 1:
        raise RuntimeError(f"P5 三个 evaluation 必须来自同一干净 commit: {evaluation_commits}")

    reused = _load_reused_reference_rows(inputs["allowlist"])
    new_seed_metrics = _new_seed_metrics(evaluation_audit)
    seed_metrics = pd.concat([reused, new_seed_metrics], ignore_index=True, sort=False)
    variant_order = [*REUSED_REFERENCE_VARIANTS, CANDIDATE_VARIANT]
    order = {name: index for index, name in enumerate(variant_order)}
    seed_metrics["variant_order"] = seed_metrics["variant"].map(order).astype(int)
    seed_metrics = seed_metrics.sort_values(["variant_order", "seed"]).reset_index(drop=True)
    aggregate_primary = _aggregate(seed_metrics, PRIMARY_METRICS, variant_order)
    aggregate_secondary = _aggregate(seed_metrics, tuple(SECONDARY_METRICS), variant_order)
    paired = paired_vs_references(seed_metrics)
    p4_summary = inputs["p4_summary"]
    summary = {
        "protocol": PROTOCOL,
        "phase": "tf_w_v2_p5_reused_research_test_summary",
        "status": "passed",
        "complete": True,
        "evidence_role": EVIDENCE_ROLE,
        "git_commit": summary_commit,
        "git_dirty": False,
        "evaluation_commit": evaluation_commits[0],
        "candidate_lock_sha256": CANDIDATE_LOCK_SHA256,
        "p5_checkpoint_allowlist_sha256": ALLOWLIST_SHA256,
        "p4_summary_sha256": P4_SUMMARY_SHA256,
        "reused_v1_research_summary_sha256": V1_RESEARCH_SUMMARY_SHA256,
        "reused_v1_research_manifest_sha256": V1_RESEARCH_MANIFEST_SHA256,
        "test_w_cache_identity_sha256": FROZEN_RESEARCH_TEST_CACHE_IDENTITY,
        "test_w_cache_manifest_sha256": FROZEN_RESEARCH_TEST_CACHE_MANIFEST_SHA256,
        "new_evaluation_variant": CANDIDATE_VARIANT,
        "new_evaluation_count": 3,
        "reused_reference_variants": list(REUSED_REFERENCE_VARIANTS),
        "reused_reference_evaluation_count": 6,
        "combined_seed_metric_count": 9,
        "expected_samples_per_evaluation": EXPECTED_TEST_SAMPLES,
        "validation_selected_epochs": [int(item["selected_epoch"]) for item in inputs["checkpoint_identities"]],
        "checkpoint_reselection_used": False,
        "validation_selection_changed": False,
        "research_test_used": True,
        "samp_id_analysis_used": False,
        "no_total_score_used": True,
        "comparison_role": "descriptive research-test comparison only; not a new selection gate",
        "d4_validation_tradeoff_retained": {
            **dict(p4_summary["d4_tradeoff"]),
            "research_test_evaluated": False,
            "exclusion_does_not_negate_compute_quality_tradeoff": True,
        },
        "decision": {
            "validation_selected_candidate_retained": CANDIDATE_ID,
            "strict_efficiency_candidate": None,
            "unique_research_test_winner_selected": False,
            "no_checkpoint_or_candidate_reselection_from_research_test": True,
        },
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".summary_", dir=OUTPUT_ROOT))
    try:
        frames = {
            "evaluation_audit.csv": evaluation_audit,
            "seed_metrics.csv": seed_metrics,
            "aggregate_primary.csv": aggregate_primary,
            "aggregate_secondary.csv": aggregate_secondary,
            "paired_vs_references.csv": paired,
        }
        for filename, frame in frames.items():
            frame.to_csv(temporary / filename, index=False)
        (temporary / "p5_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        generated = sorted(path for path in temporary.iterdir() if path.is_file())
        manifest = {
            "protocol": PROTOCOL,
            "phase": "tf_w_v2_p5_reused_research_test_summary_manifest",
            "status": "passed",
            "complete": True,
            "git_commit": summary_commit,
            "git_dirty": False,
            "research_test_used": True,
            "samp_id_analysis_used": False,
            "p5_checkpoint_allowlist_sha256": ALLOWLIST_SHA256,
            "files": {
                path.name: {"sha256": _sha256_file(path), "size_bytes": path.stat().st_size}
                for path in generated
            },
        }
        (temporary / "p5_summary_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        observed = tuple(sorted(path.name for path in temporary.iterdir() if path.is_file()))
        if observed != tuple(sorted(SUMMARY_FILES)):
            raise RuntimeError(f"P5 summary 输出 schema 不合格: {observed}")
        os.replace(temporary, SUMMARY_ROOT)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return SUMMARY_ROOT / "p5_summary.json"


def audit_p5_inputs(
    *,
    candidate_lock: str | Path = CANDIDATE_LOCK,
    allowlist_path: str | Path = ALLOWLIST,
) -> dict[str, Any]:
    candidate_lock_path = Path(candidate_lock).resolve()
    allowlist_resolved = Path(allowlist_path).resolve()
    if candidate_lock_path != CANDIDATE_LOCK.resolve():
        raise ValueError(f"P5 只接受冻结 candidate lock: {CANDIDATE_LOCK}")
    if allowlist_resolved != ALLOWLIST.resolve():
        raise ValueError(f"P5 只接受冻结 checkpoint allowlist: {ALLOWLIST}")
    _require_file_identity(candidate_lock_path, sha256=CANDIDATE_LOCK_SHA256)
    _require_file_identity(allowlist_resolved, sha256=ALLOWLIST_SHA256)
    _require_file_identity(P4_ROOT / "p4_summary.json", size_bytes=3336, sha256=P4_SUMMARY_SHA256)
    _require_file_identity(
        P4_ROOT / "p4_summary_manifest.json", size_bytes=1223, sha256=P4_MANIFEST_SHA256
    )
    _require_file_identity(
        V1_RESEARCH_SUMMARY_ROOT / "research_test_summary.json",
        size_bytes=1249,
        sha256=V1_RESEARCH_SUMMARY_SHA256,
    )
    _require_file_identity(
        V1_RESEARCH_SUMMARY_ROOT / "research_test_summary_manifest.json",
        size_bytes=1126,
        sha256=V1_RESEARCH_MANIFEST_SHA256,
    )
    _require_file_identity(
        V1_RESEARCH_SUMMARY_ROOT / "access_audit.csv",
        size_bytes=9293,
        sha256=V1_ACCESS_AUDIT_SHA256,
    )
    _require_file_identity(
        FROZEN_RESEARCH_TEST_CACHE_ROOT / "cache_manifest.json",
        size_bytes=6555,
        sha256=FROZEN_RESEARCH_TEST_CACHE_MANIFEST_SHA256,
    )

    lock = json.loads(candidate_lock_path.read_text(encoding="utf-8"))
    allowlist = json.loads(allowlist_resolved.read_text(encoding="utf-8"))
    p4_summary = json.loads((P4_ROOT / "p4_summary.json").read_text(encoding="utf-8"))
    p4_manifest = json.loads((P4_ROOT / "p4_summary_manifest.json").read_text(encoding="utf-8"))
    v1_manifest = json.loads(
        (V1_RESEARCH_SUMMARY_ROOT / "research_test_summary_manifest.json").read_text(encoding="utf-8")
    )
    cache_manifest = json.loads(
        (FROZEN_RESEARCH_TEST_CACHE_ROOT / "cache_manifest.json").read_text(encoding="utf-8")
    )
    if (
        lock.get("protocol") != PROTOCOL
        or lock.get("forbidden", {}).get("samp_id_analysis") is not True
        or (REPO_ROOT / lock.get("output_contract", {}).get("research_test_root", "")).resolve()
        != OUTPUT_ROOT.resolve()
    ):
        raise RuntimeError("P5 candidate lock schema/output contract 不合格")
    if (
        p4_summary.get("status") != "passed"
        or p4_summary.get("complete") is not True
        or p4_summary.get("strict_quality_candidates") != [CANDIDATE_ID]
        or p4_summary.get("strict_efficiency_candidates") != []
        or p4_summary.get("p5_quality_primary_if_authorized") != CANDIDATE_ID
        or len(p4_summary.get("p5_checkpoint_allowlist_if_authorized", [])) != 3
        or p4_summary.get("samp_id_analysis_used") is not False
    ):
        raise RuntimeError("P5 P4 selection source 不合格")
    if p4_manifest.get("source_artifact_set_sha256") != P4_SOURCE_ARTIFACT_SET_SHA256:
        raise RuntimeError("P5 P4 source artifact set 漂移")
    if (
        v1_manifest.get("files", {}).get("access_audit.csv", {}).get("sha256")
        != V1_ACCESS_AUDIT_SHA256
        or cache_manifest.get("cache_identity_sha256") != FROZEN_RESEARCH_TEST_CACHE_IDENTITY
        or cache_manifest.get("target_read") is not False
        or cache_manifest.get("test_target_array_read") is not False
        or int(cache_manifest.get("splits", {}).get("test", {}).get("count", -1))
        != EXPECTED_TEST_SAMPLES
    ):
        raise RuntimeError("P5 reused research-test/cache identity 不合格")
    constraints = allowlist.get("constraints", {})
    if (
        allowlist.get("protocol") != PROTOCOL
        or allowlist.get("user_authorized") is not True
        or allowlist.get("evidence_role") != EVIDENCE_ROLE
        or allowlist.get("candidate_id") != CANDIDATE_ID
        or allowlist.get("model_variant") != CANDIDATE_VARIANT
        or allowlist.get("fixed_seeds") != list(FORMAL_SEEDS)
        or allowlist.get("selection_source", {}).get("strict_efficiency_candidate") is not None
        or constraints.get("checkpoint_reselection_allowed") is not False
        or constraints.get("samp_id_analysis_allowed") is not False
        or constraints.get("d4_evaluation_allowed") is not False
        or constraints.get("w1_w2_evaluation_allowed") is not False
    ):
        raise RuntimeError("P5 checkpoint allowlist 边界不合格")
    contract = p1_variant_contract(CANDIDATE_VARIANT)
    test_cache = allowlist.get("test_w_cache", {})
    if (
        CANDIDATE_VARIANT not in P1_VARIANTS
        or contract.get("w_view") != "full_6v"
        or contract.get("view_index_sha256") != test_cache.get("view_index_sha256")
        or contract.get("model_input_shape_per_sample") != [49, 360]
        or test_cache.get("target_read") is not False
        or test_cache.get("manifest_sha256") != FROZEN_RESEARCH_TEST_CACHE_MANIFEST_SHA256
    ):
        raise RuntimeError("P5 W3 full-6V/test cache contract 不合格")

    identities = []
    p4_by_seed = {
        int(item["seed"]): item for item in p4_summary["p5_checkpoint_allowlist_if_authorized"]
    }
    for entry in allowlist.get("checkpoints", []):
        seed = int(entry["seed"])
        run_dir = (REPO_ROOT / str(entry["run_dir"])).resolve()
        checkpoint_path = run_dir / str(entry["checkpoint"]["filename"])
        config_path = run_dir / str(entry["config"]["filename"])
        training_manifest_path = run_dir / str(entry["training_manifest"]["filename"])
        _require_file_identity(checkpoint_path, **_size_hash(entry["checkpoint"]))
        _require_file_identity(config_path, **_size_hash(entry["config"]))
        _require_file_identity(training_manifest_path, **_size_hash(entry["training_manifest"]))
        cfg = load_crd_config(config_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        p4_entry = p4_by_seed.get(seed, {})
        if (
            str(cfg.model.variant) != CANDIDATE_VARIANT
            or int(cfg.training.seed) != seed
            or int(cfg.model.initialization_seed) != seed
            or int(entry["selected_epoch"]) != int(p4_entry.get("selected_epoch", -1))
            or str(entry["checkpoint"]["sha256"]) != p4_entry.get("checkpoint_sha256")
            or int(checkpoint.get("epoch", -1)) != int(entry["selected_epoch"])
        ):
            raise RuntimeError(f"P5 checkpoint/config/P4 identity 不合格: seed={seed}")
        identities.append(
            {
                "seed": seed,
                "selected_epoch": int(entry["selected_epoch"]),
                "run_dir": str(run_dir),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": str(entry["checkpoint"]["sha256"]),
                "config_path": str(config_path),
                "training_manifest_path": str(training_manifest_path),
            }
        )
    identities.sort(key=lambda item: int(item["seed"]))
    if [item["seed"] for item in identities] != list(FORMAL_SEEDS):
        raise RuntimeError("P5 checkpoint allowlist seeds 不完整")
    return {
        "candidate_lock": lock,
        "allowlist": allowlist,
        "p4_summary": p4_summary,
        "checkpoint_identities": identities,
    }


def _audit_evaluation_output(
    output_dir: Path,
    identity: Mapping[str, Any],
    *,
    expected_commit: str | None = None,
) -> dict[str, Any]:
    if not output_dir.is_dir():
        raise FileNotFoundError(f"P5 evaluation 尚未完成: {output_dir}")
    observed = tuple(sorted(path.name for path in output_dir.iterdir() if path.is_file()))
    if observed != tuple(sorted(EVALUATION_FILES)):
        raise RuntimeError(f"P5 evaluation 输出 schema 不合格: {output_dir}, files={observed}")
    metrics_path = output_dir / "research_test_metrics.csv"
    summary_path = output_dir / "research_test_metrics_summary.csv"
    manifest_path = output_dir / "research_test_metrics_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runtime = manifest.get("runtime", {})
    if (
        manifest.get("protocol") != PROTOCOL
        or manifest.get("phase") != "tf_w_v2_p5_reused_research_test_evaluation"
        or manifest.get("status") != "passed"
        or manifest.get("complete") is not True
        or manifest.get("evidence_role") != EVIDENCE_ROLE
        or manifest.get("git_dirty") is not False
        or (expected_commit is not None and manifest.get("git_commit") != expected_commit)
        or manifest.get("research_test_used") is not True
        or manifest.get("samp_id_analysis_used") is not False
        or manifest.get("candidate_id") != CANDIDATE_ID
        or manifest.get("variant") != CANDIDATE_VARIANT
        or int(manifest.get("seed", -1)) != int(identity["seed"])
        or int(manifest.get("validation_selected_epoch", -1)) != int(identity["selected_epoch"])
        or manifest.get("checkpoint_sha256") != identity["checkpoint_sha256"]
        or manifest.get("p5_checkpoint_allowlist_sha256") != ALLOWLIST_SHA256
        or manifest.get("test_w_cache_identity_sha256") != FROZEN_RESEARCH_TEST_CACHE_IDENTITY
        or manifest.get("test_w_cache_manifest_sha256") != FROZEN_RESEARCH_TEST_CACHE_MANIFEST_SHA256
        or manifest.get("test_w_cache_target_read") is not False
        or manifest.get("w_view") != "full_6v"
        or manifest.get("model_input_shape_per_sample") != [49, 360]
        or manifest.get("dataset_row_ids_sha256") != EXPECTED_TEST_ROW_IDS_SHA256
        or int(manifest.get("n_samples", -1)) != EXPECTED_TEST_SAMPLES
        or not _finite_positive_runtime(runtime)
    ):
        raise RuntimeError(f"P5 evaluation manifest identity 不合格: {manifest_path}")
    for path in (metrics_path, summary_path):
        metadata = manifest.get("files", {}).get(path.name, {})
        _require_file_identity(path, size_bytes=metadata.get("size_bytes"), sha256=metadata.get("sha256"))
    metrics = pd.read_csv(metrics_path)
    summary = pd.read_csv(summary_path).iloc[0]
    values = _validate_metrics(metrics, summary, variant=CANDIDATE_VARIANT)
    return {
        "candidate_id": CANDIDATE_ID,
        "variant": CANDIDATE_VARIANT,
        "seed": int(identity["seed"]),
        "checkpoint": str(identity["checkpoint_path"]),
        "validation_selected_epoch": int(identity["selected_epoch"]),
        "checkpoint_sha256": str(identity["checkpoint_sha256"]),
        "evaluation_commit": str(manifest["git_commit"]),
        "metrics_sha256": _sha256_file(metrics_path),
        "metrics_summary_sha256": _sha256_file(summary_path),
        "evaluation_manifest_sha256": _sha256_file(manifest_path),
        "dataset_row_ids_sha256": EXPECTED_TEST_ROW_IDS_SHA256,
        "wall_seconds": float(runtime["wall_seconds"]),
        "samples_per_second": float(runtime["samples_per_second"]),
        "peak_allocated_mib": float(runtime["peak_allocated_mib"]),
        "peak_reserved_mib": float(runtime["peak_reserved_mib"]),
        "peak_reserved_fraction": float(runtime["peak_reserved_fraction"]),
        **values,
    }


def _validate_metrics(metrics: pd.DataFrame, summary: pd.Series, *, variant: str) -> dict[str, float]:
    if (
        len(metrics) != EXPECTED_TEST_SAMPLES
        or not metrics["evaluation_split"].eq("test").all()
        or not metrics["split"].eq("test").all()
        or not metrics["method"].eq(variant).all()
        or metrics["dataset_row_id"].nunique() != EXPECTED_TEST_SAMPLES
        or int(summary["n_samples"]) != EXPECTED_TEST_SAMPLES
    ):
        raise RuntimeError("P5 evaluation sample/split/method identity 不合格")
    row_ids = metrics["dataset_row_id"].to_numpy(dtype=np.int64)
    row_hash = hashlib.sha256(row_ids.tobytes(order="C")).hexdigest()
    if row_hash != EXPECTED_TEST_ROW_IDS_SHA256:
        raise RuntimeError("P5 evaluation dataset_row_id 顺序与冻结 research-test 不一致")
    values = {name: float(summary[column]) for name, column in METRIC_COLUMNS.items()}
    values.update({name: float(summary[column]) for name, column in SECONDARY_METRICS.items()})
    if not np.isfinite(list(values.values())).all():
        raise RuntimeError("P5 evaluation summary 指标包含 NaN/Inf")
    for metric, column in SAMPLE_METRIC_COLUMNS.items():
        sample_values = metrics[column].to_numpy(dtype=np.float64)
        if not np.isfinite(sample_values).all() or not np.isclose(
            float(np.mean(sample_values)), values[metric], rtol=0.0, atol=1e-12
        ):
            raise RuntimeError(f"P5 per-sample/summary {metric} 不一致")
    if (
        bool(metrics["joint_prediction_degenerate"].astype(bool).any())
        or float(summary["joint_prediction_degenerate_fraction"]) != 0.0
        or float(summary["joint_target_eligible_fraction"]) != 1.0
        or any(
            int(summary[f"target_stratified_envelope_spearman_{name}_n_total"]) != total
            for name, total in EXPECTED_STRATUM_TOTALS.items()
        )
    ):
        raise RuntimeError("P5 evaluation eligibility/degeneracy 不合格")
    return values


def _load_reused_reference_rows(allowlist: Mapping[str, Any]) -> pd.DataFrame:
    frame = pd.read_csv(V1_RESEARCH_SUMMARY_ROOT / "access_audit.csv")
    selected = frame.loc[frame["variant"].isin(REUSED_REFERENCE_VARIANTS)].copy()
    for variant in REUSED_REFERENCE_VARIANTS:
        group = selected.loc[selected["variant"] == variant].sort_values("seed")
        if group["seed"].astype(int).tolist() != list(FORMAL_SEEDS):
            raise RuntimeError(f"P5 reused reference seeds 不完整: {variant}")
        if not group["dataset_row_ids_sha256"].eq(EXPECTED_TEST_ROW_IDS_SHA256).all():
            raise RuntimeError(f"P5 reused reference row identity 不合格: {variant}")
    if len(selected) != 6:
        raise RuntimeError("P5 reused reference evaluation count 必须为 6")
    selected["candidate_id"] = selected["variant"].map(
        {"crd_c201_decoder_10hz_cap": "C201_REUSED", "crd_tf102_w": "W0_REUSED"}
    )
    selected["evidence_source"] = "frozen_crd_tf_v1_research_test_summary"
    selected["new_evaluation"] = False
    selected["research_test_reused"] = True
    return selected


def _new_seed_metrics(evaluation_audit: pd.DataFrame) -> pd.DataFrame:
    frame = evaluation_audit.copy()
    frame["evidence_source"] = "crd_tf_w_v2_p5_new_evaluation"
    frame["new_evaluation"] = True
    frame["research_test_reused"] = False
    return frame


def _aggregate(
    seed_metrics: pd.DataFrame,
    metrics: Sequence[str],
    variant_order: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for variant in variant_order:
        group = seed_metrics.loc[seed_metrics["variant"] == variant].sort_values("seed")
        if group["seed"].astype(int).tolist() != list(FORMAL_SEEDS):
            raise RuntimeError(f"P5 aggregate seeds 不完整: {variant}")
        for metric in metrics:
            values = group[metric].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise RuntimeError(f"P5 aggregate {variant}/{metric} 包含 NaN/Inf")
            rows.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "sample_sd": float(np.std(values, ddof=1)),
                    **{f"seed_{seed}": float(value) for seed, value in zip(FORMAL_SEEDS, values)},
                }
            )
    return pd.DataFrame(rows)


def paired_vs_references(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    candidate = seed_metrics.loc[seed_metrics["variant"] == CANDIDATE_VARIANT].sort_values("seed")
    if candidate["seed"].astype(int).tolist() != list(FORMAL_SEEDS):
        raise RuntimeError("P5 paired comparison W3 seeds 不完整")
    for reference_variant in REUSED_REFERENCE_VARIANTS:
        reference = seed_metrics.loc[seed_metrics["variant"] == reference_variant].sort_values("seed")
        if reference["seed"].astype(int).tolist() != list(FORMAL_SEEDS):
            raise RuntimeError(f"P5 paired comparison reference seeds 不完整: {reference_variant}")
        for metric in PRIMARY_METRICS:
            candidate_values = candidate[metric].to_numpy(dtype=np.float64)
            reference_values = reference[metric].to_numpy(dtype=np.float64)
            delta = candidate_values - reference_values
            minimize = metric in ERROR_METRICS
            rows.append(
                {
                    "candidate": CANDIDATE_VARIANT,
                    "reference": reference_variant,
                    "metric": metric,
                    "direction": "minimize" if minimize else "maximize",
                    "candidate_mean": float(np.mean(candidate_values)),
                    "reference_mean": float(np.mean(reference_values)),
                    "mean_delta": float(np.mean(delta)),
                    "relative_delta": (
                        float(np.mean(candidate_values) / np.mean(reference_values) - 1.0)
                        if minimize
                        else np.nan
                    ),
                    "absolute_delta": float(np.mean(delta)) if not minimize else np.nan,
                    "better_seed_count": int(
                        np.sum(candidate_values < reference_values)
                        if minimize
                        else np.sum(candidate_values > reference_values)
                    ),
                    **{f"seed_{seed}_delta": float(value) for seed, value in zip(FORMAL_SEEDS, delta)},
                    "comparison_role": "descriptive_only_not_selection",
                }
            )
    return pd.DataFrame(rows)


def _evaluation_dir(seed: int) -> Path:
    return EVALUATION_ROOT / CANDIDATE_VARIANT / f"seed_{int(seed)}"


def _finite_positive_runtime(runtime: Mapping[str, Any]) -> bool:
    names = (
        "wall_seconds",
        "samples_per_second",
        "peak_allocated_mib",
        "peak_reserved_mib",
        "total_device_memory_mib",
        "peak_reserved_fraction",
    )
    try:
        values = np.asarray([float(runtime[name]) for name in names], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return False
    return bool(np.isfinite(values).all() and np.all(values > 0.0) and float(values[-1]) <= 1.0)


def _size_hash(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {"size_bytes": int(metadata["size_bytes"]), "sha256": str(metadata["sha256"])}


def _require_file_identity(
    path: Path,
    *,
    sha256: Any,
    size_bytes: Any | None = None,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if size_bytes is not None and path.stat().st_size != int(size_bytes):
        raise RuntimeError(f"P5 artifact size 漂移: {path}")
    if not isinstance(sha256, str) or _sha256_file(path) != sha256:
        raise RuntimeError(f"P5 artifact SHA-256 漂移: {path}")


def _require_clean_git(label: str) -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    if dirty:
        raise RuntimeError(f"{label} 必须从干净 Git commit 运行")
    return commit


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
