from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from resp_train.crd.config import (
    CRD_TF_RESEARCH_TEST_PROTOCOL_VERSION,
    FORMAL_SEEDS,
    crd_dependency_versions,
    load_crd_config,
)
from resp_train.crd.experiment import CRDExperiment
from resp_train.crd.tf_v1_research_test_data import (
    FROZEN_RESEARCH_TEST_CACHE_IDENTITY,
    FROZEN_RESEARCH_TEST_CACHE_MANIFEST_SHA256,
    FROZEN_RESEARCH_TEST_CACHE_ROOT,
)
from resp_train.utils.run import save_execution_manifest


REPO_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_LOCK = REPO_ROOT / "docs/experiments/crd_tf_v1_candidate_lock_20260812.json"
CANDIDATE_LOCK_SHA256 = "c8d4823500e6096fcacb8d2e8787f7b3422160813eabe31adf01b1f1f75cc139"
P5_ROOT = REPO_ROOT / "runs/crd_tf_v1/p5_validation_summary"
P5_SUMMARY_SHA256 = "afb6feba1600c9c5e07d713db9cac7c886aa87a033037649d3e9ac32cb753b4e"
P5_MANIFEST_SHA256 = "cecd35d862cc975cf7d99af9ad18fce4576ddb6b9aa561ad88322e3888992a65"
RESEARCH_TEST_VARIANTS = (
    "crd_c201_decoder_10hz_cap",
    "crd_tf101_m",
    "crd_tf102_w",
    "crd_tf203_ms",
)


def evaluate_tf_v1_research_test_checkpoint(
    *, checkpoint_path: str | Path, device: str
) -> Path:
    evaluation_commit = _require_clean_git()
    checkpoint_path = Path(checkpoint_path).resolve()
    identity = expected_research_test_checkpoints().get(checkpoint_path)
    if identity is None:
        raise ValueError(f"checkpoint 不在冻结 C201/M/W/MS × 3 research-test 矩阵: {checkpoint_path}")
    if _sha256_file(checkpoint_path) != identity["checkpoint_sha256"]:
        raise RuntimeError(f"research-test checkpoint SHA-256 不一致: {checkpoint_path}")
    output_path = checkpoint_path.parent / "research_test_metrics.csv"
    output_files = (
        output_path,
        output_path.with_name("research_test_metrics_summary.csv"),
        output_path.with_name("research_test_metrics_manifest.json"),
    )
    if any(path.exists() for path in output_files):
        raise FileExistsError(f"research-test 输出已存在，拒绝覆盖: {checkpoint_path.parent}")
    cfg = load_crd_config(
        checkpoint_path.parent / "config.yaml",
        overrides=[f"training.device={device}", "training.show_progress=false"],
    )
    if str(cfg.model.variant) != identity["variant"] or int(cfg.training.seed) != identity["seed"]:
        raise RuntimeError("research-test checkpoint/config variant 或 seed identity 不一致")
    CRDExperiment(cfg).evaluate_checkpoint(
        checkpoint_path,
        split="test",
        metrics_output=output_path,
        tf_research_test_cache_path=(
            FROZEN_RESEARCH_TEST_CACHE_ROOT if identity["variant"].startswith("crd_tf") else None
        ),
    )
    save_execution_manifest(
        output_path.with_name("research_test_metrics_manifest.json"),
        task=CRDExperiment.task_name,
        phase="tf_v1_reused_research_test_evaluation",
        protocol=CRD_TF_RESEARCH_TEST_PROTOCOL_VERSION,
        evidence_role="reused research/development evidence; not unbiased held-out evidence",
        research_test_used=True,
        checkpoint=str(checkpoint_path),
        checkpoint_sha256=identity["checkpoint_sha256"],
        variant=identity["variant"],
        seed=identity["seed"],
        validation_selected_epoch=identity["selected_epoch"],
        selection_source="frozen C201 lock / P5 validation summary",
        evaluation_commit=evaluation_commit,
        cache_root=(str(FROZEN_RESEARCH_TEST_CACHE_ROOT) if identity["variant"].startswith("crd_tf") else None),
        cache_identity=(FROZEN_RESEARCH_TEST_CACHE_IDENTITY if identity["variant"].startswith("crd_tf") else None),
        cache_manifest_sha256=(
            FROZEN_RESEARCH_TEST_CACHE_MANIFEST_SHA256 if identity["variant"].startswith("crd_tf") else None
        ),
        dependency_versions=crd_dependency_versions(),
    )
    return output_path


def expected_research_test_checkpoints() -> dict[Path, dict[str, Any]]:
    _audit_selection_inputs()
    expected: dict[Path, dict[str, Any]] = {}
    lock = json.loads(CANDIDATE_LOCK.read_text(encoding="utf-8"))
    for record in lock["seeds"]:
        run_dir = REPO_ROOT / str(record["run_dir"])
        path = (run_dir / "checkpoint_best_local_rr.pt").resolve()
        expected[path] = {
            "variant": "crd_c201_decoder_10hz_cap",
            "seed": int(record["seed"]),
            "selected_epoch": int(record["selected_epoch"]),
            "checkpoint_sha256": record["files"]["checkpoint_best_local_rr.pt"]["sha256"],
        }
    audit = pd.read_csv(P5_ROOT / "formal_run_audit.csv")
    for row in audit.loc[audit["variant"].isin(RESEARCH_TEST_VARIANTS[1:])].itertuples(index=False):
        path = (Path(str(row.run_dir)) / "checkpoint_best_local_rr.pt").resolve()
        expected[path] = {
            "variant": str(row.variant),
            "seed": int(row.seed),
            "selected_epoch": int(row.selected_epoch),
            "checkpoint_sha256": str(row.checkpoint_sha256),
        }
    observed = {(value["variant"], value["seed"]) for value in expected.values()}
    wanted = {(variant, seed) for variant in RESEARCH_TEST_VARIANTS for seed in FORMAL_SEEDS}
    if observed != wanted or len(expected) != 12:
        raise RuntimeError("research-test checkpoint matrix identity 不完整")
    return expected


def _audit_selection_inputs() -> None:
    if _sha256_file(CANDIDATE_LOCK) != CANDIDATE_LOCK_SHA256:
        raise RuntimeError("research-test C201 candidate lock SHA-256 不一致")
    if _sha256_file(P5_ROOT / "p5_summary.json") != P5_SUMMARY_SHA256:
        raise RuntimeError("research-test P5 summary SHA-256 不一致")
    manifest_path = P5_ROOT / "p5_summary_manifest.json"
    if _sha256_file(manifest_path) != P5_MANIFEST_SHA256:
        raise RuntimeError("research-test P5 manifest SHA-256 不一致")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    formal_identity = manifest.get("files", {}).get("formal_run_audit.csv", {})
    formal_path = P5_ROOT / "formal_run_audit.csv"
    if (
        formal_path.stat().st_size != int(formal_identity.get("size_bytes", -1))
        or _sha256_file(formal_path) != formal_identity.get("sha256")
    ):
        raise RuntimeError("research-test P5 formal audit identity 不一致")


def _require_clean_git() -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    if dirty:
        raise RuntimeError("CRD-TF research-test evaluation 必须从干净 Git commit 运行")
    return commit


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

