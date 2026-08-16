from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from resp_train.crd.config import CRD_TF_RESEARCH_TEST_PROTOCOL_VERSION, FORMAL_SEEDS
from resp_train.crd.tf_v1_p5 import METRIC_COLUMNS, SAMPLE_METRIC_COLUMNS
from resp_train.crd.tf_v1_research_test import (
    RESEARCH_TEST_VARIANTS,
    expected_research_test_checkpoints,
)
from resp_train.crd.tf_v1_research_test_data import (
    FROZEN_RESEARCH_TEST_CACHE_IDENTITY,
    FROZEN_RESEARCH_TEST_CACHE_MANIFEST_SHA256,
)
from resp_train.crd.tf_v1_selection import (
    PRIMARY_METRICS,
    paired_material_improvement,
    passes_base_guardrails,
    tolerance_pareto_set,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "runs/crd_tf_v1/research_test_summary"
EVALUATION_COMMIT = "9f429dae8f4a879c6b9530c1190df1949b6c420a"
EXPECTED_SAMPLES = 2310
BASE_VARIANT = "crd_c201_decoder_10hz_cap"
CANDIDATES = RESEARCH_TEST_VARIANTS[1:]
SECONDARY_METRICS = {
    "ibi_medae_sec": "ibi_medae_sec_mean",
    "ibi_coverage": "ibi_coverage_mean",
    "respiratory_band_coherence": "respiratory_band_coherence_mean",
    "constrained_ndtw": "constrained_ndtw_mean",
    "envelope_spearman_low": "target_stratified_envelope_spearman_low_mean",
    "envelope_spearman_medium": "target_stratified_envelope_spearman_medium_mean",
    "envelope_spearman_high": "target_stratified_envelope_spearman_high_mean",
    "ibi_interpretable_fraction": "ibi_interpretable_fraction",
}


def generate_research_test_summary() -> Path:
    commit = _require_clean_git()
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"research-test summary 已存在，拒绝覆盖: {OUTPUT_ROOT}")
    access_rows = _audit_evaluations()
    seed_metrics = pd.DataFrame(access_rows)
    primary = _aggregate(seed_metrics, PRIMARY_METRICS)
    secondary = _aggregate(seed_metrics, tuple(SECONDARY_METRICS))
    comparisons = _paired_vs_base(seed_metrics)
    eligibility, decision = _selection(primary, comparisons)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    frames = {
        "access_audit.csv": seed_metrics,
        "aggregate_primary.csv": primary,
        "aggregate_secondary.csv": secondary,
        "paired_vs_c201.csv": comparisons,
        "eligibility.csv": eligibility,
    }
    for filename, frame in frames.items():
        frame.to_csv(OUTPUT_ROOT / filename, index=False)
    summary = {
        "protocol": CRD_TF_RESEARCH_TEST_PROTOCOL_VERSION,
        "phase": "tf_v1_reused_research_test_summary",
        "status": "passed",
        "complete": True,
        "evidence_role": "reused research/development evidence; not unbiased held-out evidence",
        "research_test_used": True,
        "git_commit": commit,
        "git_dirty": False,
        "evaluation_commit": EVALUATION_COMMIT,
        "evaluation_count": len(access_rows),
        "expected_samples_per_evaluation": EXPECTED_SAMPLES,
        "candidate_matrix": list(RESEARCH_TEST_VARIANTS),
        "cache_identity": FROZEN_RESEARCH_TEST_CACHE_IDENTITY,
        "cache_manifest_sha256": FROZEN_RESEARCH_TEST_CACHE_MANIFEST_SHA256,
        "decision": decision,
        "no_total_score_used": True,
        "checkpoint_reselection_used": False,
    }
    summary_path = OUTPUT_ROOT / "research_test_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "protocol": CRD_TF_RESEARCH_TEST_PROTOCOL_VERSION,
        "phase": "tf_v1_reused_research_test_summary_manifest",
        "git_commit": commit,
        "git_dirty": False,
        "research_test_used": True,
        "files": {
            path.name: {"sha256": _sha256_file(path), "size_bytes": path.stat().st_size}
            for path in sorted(OUTPUT_ROOT.iterdir())
            if path.is_file()
        },
    }
    (OUTPUT_ROOT / "research_test_summary_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary_path


def _audit_evaluations() -> list[dict[str, Any]]:
    expected = expected_research_test_checkpoints()
    rows: list[dict[str, Any]] = []
    frozen_row_hash: str | None = None
    for checkpoint_path, identity in sorted(
        expected.items(), key=lambda item: (item[1]["variant"], item[1]["seed"])
    ):
        run_dir = checkpoint_path.parent
        metrics_path = run_dir / "research_test_metrics.csv"
        summary_path = run_dir / "research_test_metrics_summary.csv"
        manifest_path = run_dir / "research_test_metrics_manifest.json"
        missing = [path.name for path in (metrics_path, summary_path, manifest_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"research-test 产物不完整 {run_dir}: {missing}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        is_tf = str(identity["variant"]).startswith("crd_tf")
        if (
            manifest.get("git_commit") != EVALUATION_COMMIT
            or manifest.get("git_dirty") is not False
            or manifest.get("phase") != "tf_v1_reused_research_test_evaluation"
            or manifest.get("protocol") != CRD_TF_RESEARCH_TEST_PROTOCOL_VERSION
            or manifest.get("research_test_used") is not True
            or manifest.get("variant") != identity["variant"]
            or int(manifest.get("seed", -1)) != identity["seed"]
            or int(manifest.get("validation_selected_epoch", -1)) != identity["selected_epoch"]
            or manifest.get("checkpoint_sha256") != identity["checkpoint_sha256"]
            or manifest.get("cache_identity")
            != (FROZEN_RESEARCH_TEST_CACHE_IDENTITY if is_tf else None)
            or manifest.get("cache_manifest_sha256")
            != (FROZEN_RESEARCH_TEST_CACHE_MANIFEST_SHA256 if is_tf else None)
        ):
            raise RuntimeError(f"research-test evaluation manifest identity 不合格: {manifest_path}")
        metrics = pd.read_csv(metrics_path)
        summary = pd.read_csv(summary_path).iloc[0]
        if (
            len(metrics) != EXPECTED_SAMPLES
            or not metrics["evaluation_split"].eq("test").all()
            or not metrics["split"].eq("test").all()
            or not metrics["method"].eq(identity["variant"]).all()
            or metrics["dataset_row_id"].nunique() != EXPECTED_SAMPLES
            or int(summary["n_samples"]) != EXPECTED_SAMPLES
        ):
            raise RuntimeError(f"research-test sample/split identity 不合格: {metrics_path}")
        row_ids = metrics["dataset_row_id"].to_numpy(dtype=np.int64)
        row_hash = hashlib.sha256(row_ids.tobytes(order="C")).hexdigest()
        if frozen_row_hash is None:
            frozen_row_hash = row_hash
        elif row_hash != frozen_row_hash:
            raise RuntimeError("research-test evaluations 的 dataset_row_id 顺序不一致")
        values = _summary_metrics(summary)
        for metric, column in SAMPLE_METRIC_COLUMNS.items():
            sample_values = metrics[column].to_numpy(dtype=np.float64)
            if not np.isfinite(sample_values).all() or not np.isclose(
                float(np.mean(sample_values)), values[metric], rtol=0.0, atol=1e-12
            ):
                raise RuntimeError(f"research-test per-sample/summary {metric} 不一致: {metrics_path}")
        if (
            bool(metrics["joint_prediction_degenerate"].astype(bool).any())
            or float(summary["joint_prediction_degenerate_fraction"]) != 0.0
            or float(summary["joint_target_eligible_fraction"]) != 1.0
            or int(summary["target_stratified_envelope_spearman_low_n_total"]) != 791
            or int(summary["target_stratified_envelope_spearman_medium_n_total"]) != 957
            or int(summary["target_stratified_envelope_spearman_high_n_total"]) != 562
        ):
            raise RuntimeError(f"research-test eligibility/degeneracy 不合格: {metrics_path}")
        rows.append(
            {
                "variant": identity["variant"],
                "seed": identity["seed"],
                "checkpoint": str(checkpoint_path),
                "validation_selected_epoch": identity["selected_epoch"],
                "checkpoint_sha256": identity["checkpoint_sha256"],
                "metrics_sha256": _sha256_file(metrics_path),
                "metrics_summary_sha256": _sha256_file(summary_path),
                "evaluation_manifest_sha256": _sha256_file(manifest_path),
                "dataset_row_ids_sha256": row_hash,
                **values,
            }
        )
    if len(rows) != 12:
        raise RuntimeError(f"research-test evaluation 数必须为 12，实际 {len(rows)}")
    return rows


def _summary_metrics(summary: pd.Series) -> dict[str, float]:
    values = {name: float(summary[column]) for name, column in METRIC_COLUMNS.items()}
    values.update({name: float(summary[column]) for name, column in SECONDARY_METRICS.items()})
    if not np.isfinite(list(values.values())).all():
        raise RuntimeError("research-test summary 指标包含 NaN/Inf")
    return values


def _aggregate(seed_metrics: pd.DataFrame, metrics: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for variant, group in seed_metrics.groupby("variant", sort=False):
        ordered = group.sort_values("seed")
        if ordered["seed"].astype(int).tolist() != list(FORMAL_SEEDS):
            raise RuntimeError(f"research-test aggregate seeds 不完整: {variant}")
        for metric in metrics:
            values = ordered[metric].to_numpy(dtype=np.float64)
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


def _paired_vs_base(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    lookup = _seed_lookup(seed_metrics)
    rows = []
    for candidate in CANDIDATES:
        for metric in PRIMARY_METRICS:
            result = paired_material_improvement(
                metric, lookup[candidate][metric], lookup[BASE_VARIANT][metric]
            )
            rows.append({"candidate": candidate, "reference": BASE_VARIANT, "metric": metric, **result})
    return pd.DataFrame(rows)


def _selection(
    aggregate_primary: pd.DataFrame, comparisons: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    means = _mean_lookup(aggregate_primary)
    base = means[BASE_VARIANT]
    rows = []
    qualified = []
    for candidate in CANDIDATES:
        passed_metrics = comparisons.loc[
            (comparisons["candidate"] == candidate) & comparisons["passed"], "metric"
        ].tolist()
        guardrail = passes_base_guardrails(means[candidate], base)
        is_qualified = bool(guardrail and passed_metrics)
        if is_qualified:
            qualified.append(candidate)
        rows.append(
            {
                "variant": candidate,
                "base_guardrails_passed": guardrail,
                "material_improvement_metrics": ";".join(passed_metrics),
                "qualified": is_qualified,
            }
        )
    pareto = list(tolerance_pareto_set({name: means[name] for name in qualified})) if qualified else []
    local_rr_lead = min(qualified, key=lambda name: means[name]["local_rr_mae"]) if qualified else None
    decision = {
        "qualified_candidates": qualified,
        "tolerance_pareto": pareto,
        "local_rr_lead": local_rr_lead,
        "local_rr_lead_mean": means[local_rr_lead]["local_rr_mae"] if local_rr_lead else None,
        "unique_winner_selected": False,
        "decision": "retain_research_test_pareto_without_total_score" if pareto else "no_candidate_qualified",
    }
    return pd.DataFrame(rows), decision


def _seed_lookup(frame: pd.DataFrame) -> dict[str, dict[str, list[float]]]:
    return {
        str(variant): {
            metric: group.sort_values("seed")[metric].astype(float).tolist() for metric in PRIMARY_METRICS
        }
        for variant, group in frame.groupby("variant", sort=False)
    }


def _mean_lookup(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        str(variant): {str(row.metric): float(row.mean) for row in group.itertuples(index=False)}
        for variant, group in frame.groupby("variant", sort=False)
    }


def _require_clean_git() -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    if dirty:
        raise RuntimeError("CRD-TF research-test summary 必须从干净 Git commit 生成")
    return commit


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

