from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch

from resp_train.crd.config import CRD_TF_CACHE_PATH, CRD_TF_PROTOCOL_VERSION, FORMAL_SEEDS, load_crd_config
from resp_train.crd.tf_v1_data import FROZEN_CACHE_MANIFEST_SHA256, FROZEN_CACHE_TRANSFORM_SHA256
from resp_train.crd.tf_v1_model import TF_VARIANT_REPRESENTATIONS, TF_VARIANTS
from resp_train.crd.tf_v1_selection import (
    PRIMARY_METRICS,
    interaction_summary,
    pair_interaction,
    paired_material_improvement,
    passes_base_guardrails,
    tolerance_pareto_set,
    triple_interaction,
    validate_formal_matrix,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FORMAL_ROOT = REPO_ROOT / "runs/crd_tf_v1/formal"
OUTPUT_ROOT = REPO_ROOT / "runs/crd_tf_v1/p5_validation_summary"
CANDIDATE_LOCK = REPO_ROOT / "docs/experiments/crd_tf_v1_candidate_lock_20260812.json"
CANDIDATE_LOCK_SHA256 = "c8d4823500e6096fcacb8d2e8787f7b3422160813eabe31adf01b1f1f75cc139"
FORMAL_TRAINING_COMMIT = "68b3b85df2f18b3dc8ec59e02ac0780f2be42122"
EXPECTED_VALIDATION_SAMPLES = 2675

METRIC_COLUMNS = {
    "whole_rr_mae": "whole_rr_abs_error_bpm_mean",
    "local_rr_mae": "local_rr_mae_bpm_mean",
    "trajectory_mae": "envelope_trajectory_mae_mean",
    "global_envelope_error": "global_envelope_modulation_error_mean",
    "signed_pcc": "lag_aware_signed_pcc_mean",
}
SAMPLE_METRIC_COLUMNS = {
    "whole_rr_mae": "whole_rr_abs_error_bpm",
    "local_rr_mae": "local_rr_mae_bpm",
    "trajectory_mae": "envelope_trajectory_mae",
    "global_envelope_error": "global_envelope_modulation_error",
    "signed_pcc": "lag_aware_signed_pcc",
}

SINGLES = {
    "m": "crd_tf101_m",
    "w": "crd_tf102_w",
    "l": "crd_tf103_l",
    "s": "crd_tf104_s",
}
PAIRS = {
    ("m", "w"): "crd_tf201_mw",
    ("m", "l"): "crd_tf202_ml",
    ("m", "s"): "crd_tf203_ms",
    ("w", "l"): "crd_tf204_wl",
    ("w", "s"): "crd_tf205_ws",
    ("l", "s"): "crd_tf206_ls",
}
TRIPLES = {
    ("m", "l", "s"): "crd_tf301_mls",
    ("w", "l", "s"): "crd_tf302_wls",
}
CONTROL_BY_BRANCH_COUNT = {
    1: "crd_tf_ctrl1",
    2: "crd_tf_ctrl2",
    3: "crd_tf_ctrl3",
}


def generate_p5_summary() -> Path:
    commit = _require_clean_git()
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"P5 summary 已存在，拒绝覆盖: {OUTPUT_ROOT}")
    if _sha256_file(CANDIDATE_LOCK) != CANDIDATE_LOCK_SHA256:
        raise RuntimeError("TF000 candidate lock SHA-256 不一致")
    cache_manifest_path = Path(CRD_TF_CACHE_PATH) / "cache_manifest.json"
    if _sha256_file(cache_manifest_path) != FROZEN_CACHE_MANIFEST_SHA256:
        raise RuntimeError("CRD-TF cache manifest SHA-256 不一致")
    base_rows = _audit_base_anchor()
    formal_rows, excluded_rows = _audit_formal_runs()
    available: dict[str, list[int]] = {}
    for row in formal_rows:
        available.setdefault(str(row["variant"]), []).append(int(row["seed"]))
    validate_formal_matrix(available)

    seed_metrics = pd.DataFrame(base_rows + formal_rows)
    aggregate = _aggregate_metrics(seed_metrics)
    interactions = _interaction_table(seed_metrics)
    eligibility, capacity = _eligibility_tables(seed_metrics, aggregate, interactions)
    candidates = _candidate_sets(aggregate, eligibility)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    paths = {
        "formal_run_audit.csv": pd.DataFrame(formal_rows),
        "excluded_incomplete_runs.csv": pd.DataFrame(excluded_rows),
        "seed_metrics.csv": seed_metrics,
        "aggregate_metrics.csv": aggregate,
        "interactions.csv": interactions,
        "capacity_comparisons.csv": capacity,
        "eligibility.csv": eligibility,
    }
    for filename, frame in paths.items():
        frame.to_csv(OUTPUT_ROOT / filename, index=False)

    summary = {
        "protocol": CRD_TF_PROTOCOL_VERSION,
        "phase": "p5_validation_summary",
        "status": "passed",
        "complete": True,
        "evidence_role": "validation-development; not unbiased held-out evidence",
        "research_test_used": False,
        "git_commit": commit,
        "git_dirty": False,
        "formal_training_commit": FORMAL_TRAINING_COMMIT,
        "candidate_lock_sha256": _sha256_file(CANDIDATE_LOCK),
        "cache_path": CRD_TF_CACHE_PATH,
        "cache_transform_sha256": FROZEN_CACHE_TRANSFORM_SHA256,
        "cache_manifest_sha256": FROZEN_CACHE_MANIFEST_SHA256,
        "formal_run_count": len(formal_rows),
        "base_run_count": len(base_rows),
        "excluded_incomplete_run_count": len(excluded_rows),
        "excluded_incomplete_runs": excluded_rows,
        "candidate_sets": candidates,
        "qualified_variants": eligibility.loc[eligibility["qualified"], "variant"].tolist(),
        "no_total_score_used": True,
        "research_test_opened": False,
        "p6_opened": False,
    }
    summary_path = OUTPUT_ROOT / "p5_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "protocol": CRD_TF_PROTOCOL_VERSION,
        "phase": "p5_validation_summary_manifest",
        "git_commit": commit,
        "git_dirty": False,
        "research_test_used": False,
        "files": {
            path.name: {"sha256": _sha256_file(path), "size_bytes": path.stat().st_size}
            for path in sorted(OUTPUT_ROOT.iterdir())
            if path.is_file()
        },
    }
    (OUTPUT_ROOT / "p5_summary_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary_path


def _audit_base_anchor() -> list[dict[str, Any]]:
    lock = json.loads(CANDIDATE_LOCK.read_text(encoding="utf-8"))
    if (
        lock.get("protocol") != CRD_TF_PROTOCOL_VERSION
        or lock.get("lock_name") != "TF000_C201_ANCHOR"
        or lock.get("variant") != "crd_c201_decoder_10hz_cap"
        or lock.get("research_test_used_for_c201_selection") is not False
    ):
        raise RuntimeError("TF000 candidate lock identity 不合格")
    rows = []
    for seed_entry in lock["seeds"]:
        seed = int(seed_entry["seed"])
        run_dir = REPO_ROOT / str(seed_entry["run_dir"])
        for filename, metadata in seed_entry["files"].items():
            path = run_dir / filename
            if path.stat().st_size != int(metadata["size_bytes"]) or _sha256_file(path) != metadata["sha256"]:
                raise RuntimeError(f"TF000 lock file identity 不一致: {path}")
        summary = pd.read_csv(run_dir / "metrics_summary.csv").iloc[0]
        metrics = _metrics_from_summary(summary)
        rows.append(
            {
                "variant": "tf000_c201_anchor",
                "seed": seed,
                "run_dir": str(run_dir.resolve()),
                "selected_epoch": int(seed_entry["selected_epoch"]),
                "checkpoint_sha256": seed_entry["files"]["checkpoint_best_local_rr.pt"]["sha256"],
                "training_commit": lock["source_training_commit"],
                "peak_reserved_fraction": np.nan,
                **metrics,
            }
        )
    if sorted(row["seed"] for row in rows) != list(FORMAL_SEEDS):
        raise RuntimeError("TF000 anchor seeds 不完整")
    return rows


def _audit_formal_runs() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    required = (
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
    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for variant in TF_VARIANTS:
        for seed in FORMAL_SEEDS:
            parent = FORMAL_ROOT / variant / f"seed_{seed}"
            run_dirs = sorted(path for path in parent.glob("20*") if path.is_dir())
            complete = [path for path in run_dirs if all((path / name).is_file() for name in required)]
            if len(complete) != 1:
                raise RuntimeError(f"P5 要求 {variant} seed={seed} 恰有一个完整 run，实际 {len(complete)}")
            for path in run_dirs:
                if path not in complete:
                    _audit_incomplete_run_identity(path, variant, int(seed))
                    excluded.append(
                        {
                            "variant": variant,
                            "seed": int(seed),
                            "run_dir": str(path.resolve()),
                            "reason": "incomplete_lifecycle",
                            "present_files": ";".join(sorted(item.name for item in path.iterdir() if item.is_file())),
                        }
                    )
            rows.append(_audit_one_formal_run(complete[0], variant, int(seed)))
    return rows, excluded


def _audit_incomplete_run_identity(run_dir: Path, variant: str, seed: int) -> None:
    config_path = run_dir / "config.yaml"
    manifest_path = run_dir / "run_manifest.json"
    if not config_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"无法识别的 incomplete run，缺 config/manifest: {run_dir}")
    cfg = load_crd_config(config_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        str(cfg.model.variant) != variant
        or int(cfg.training.seed) != seed
        or str(cfg.protocol.run_role) != "formal"
        or manifest.get("git_commit") != FORMAL_TRAINING_COMMIT
        or manifest.get("git_dirty") is not False
        or manifest.get("protocol") != CRD_TF_PROTOCOL_VERSION
    ):
        raise RuntimeError(f"incomplete run identity 不合格: {run_dir}")


def _audit_one_formal_run(run_dir: Path, variant: str, seed: int) -> dict[str, Any]:
    cfg = load_crd_config(run_dir / "config.yaml")
    if (
        str(cfg.model.variant) != variant
        or list(cfg.model.tf_representations) != list(TF_VARIANT_REPRESENTATIONS[variant])
        or str(cfg.protocol.run_role) != "formal"
        or str(cfg.protocol.execution_gate) != "p4_formal"
        or int(cfg.training.seed) != seed
        or int(cfg.model.initialization_seed) != seed
        or (int(cfg.training.epochs), int(cfg.training.batch_size), int(cfg.training.gradient_accumulation_steps))
        != (80, 128, 1)
        or cfg.training.early_stopping_enabled is not False
        or str(cfg.data.tf_cache_path) != CRD_TF_CACHE_PATH
        or any(cfg.data.get(name) is not None for name in ("max_train_windows", "max_val_windows", "max_test_windows"))
    ):
        raise RuntimeError(f"P5 formal config identity 不一致: {run_dir}")
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("git_commit") != FORMAL_TRAINING_COMMIT
        or manifest.get("git_dirty") is not False
        or manifest.get("protocol") != CRD_TF_PROTOCOL_VERSION
        or manifest.get("stage") != "tf"
        or manifest.get("run_role") != "formal"
    ):
        raise RuntimeError(f"P5 formal manifest identity 不一致: {run_dir}")
    history = pd.read_csv(run_dir / "train_history.csv")
    if (
        len(history) != 80
        or history["epoch"].astype(int).tolist() != list(range(1, 81))
        or int(history.iloc[-1]["optimizer_update"]) != 6400
        or not np.isfinite(history.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)).all()
    ):
        raise RuntimeError(f"P5 formal history 不完整或非有限: {run_dir}")
    selected_epoch = int(history.loc[history["val_local_rr_mae"].idxmin(), "epoch"])
    checkpoint_hashes = {}
    for filename, expected_epoch, expected_update in (
        ("checkpoint_best_local_rr.pt", selected_epoch, selected_epoch * 80),
        ("checkpoint_final.pt", 80, 6400),
    ):
        path = run_dir / filename
        checkpoint = torch.load(path, map_location="cpu")
        extra = checkpoint.get("extra_state", {})
        tensors = list(_iter_tensors(checkpoint.get("model_state_dict"))) + list(
            _iter_tensors(checkpoint.get("optimizer_state_dict"))
        )
        if (
            int(checkpoint.get("epoch", -1)) != expected_epoch
            or int(extra.get("update_index", -1)) != expected_update
            or int(extra.get("total_updates", -1)) != 6400
            or not tensors
            or not all(bool(torch.isfinite(tensor).all()) for tensor in tensors)
        ):
            raise RuntimeError(f"P5 checkpoint 不合格: {path}")
        checkpoint_hashes[filename] = _sha256_file(path)
    metrics_frame = pd.read_csv(run_dir / "metrics.csv")
    summary = pd.read_csv(run_dir / "metrics_summary.csv").iloc[0]
    metrics = _metrics_from_summary(summary)
    if len(metrics_frame) != EXPECTED_VALIDATION_SAMPLES:
        raise RuntimeError(f"P5 validation sample count 错误: {run_dir}")
    for metric, column in SAMPLE_METRIC_COLUMNS.items():
        values = metrics_frame[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or not np.isclose(float(np.mean(values)), metrics[metric], rtol=0.0, atol=1e-12):
            raise RuntimeError(f"P5 per-sample/summary {metric} 不一致: {run_dir}")
    if bool(metrics_frame["joint_prediction_degenerate"].astype(bool).any()):
        raise RuntimeError(f"P5 prediction degeneracy 非零: {run_dir}")
    runtime = json.loads((run_dir / "runtime_summary.json").read_text(encoding="utf-8"))
    peak_reserved_fraction = float(runtime["peak_reserved_fraction"])
    if not np.isfinite(peak_reserved_fraction) or peak_reserved_fraction <= 0.0:
        raise RuntimeError(f"P5 runtime summary 不合格: {run_dir}")
    return {
        "variant": variant,
        "seed": seed,
        "run_dir": str(run_dir.resolve()),
        "selected_epoch": selected_epoch,
        "checkpoint_sha256": checkpoint_hashes["checkpoint_best_local_rr.pt"],
        "final_checkpoint_sha256": checkpoint_hashes["checkpoint_final.pt"],
        "training_commit": FORMAL_TRAINING_COMMIT,
        "peak_reserved_fraction": peak_reserved_fraction,
        **metrics,
    }


def _metrics_from_summary(summary: pd.Series) -> dict[str, float]:
    if int(summary["n_samples"]) != EXPECTED_VALIDATION_SAMPLES:
        raise RuntimeError("P5 metrics summary sample count 错误")
    metrics = {name: float(summary[column]) for name, column in METRIC_COLUMNS.items()}
    if not np.isfinite(list(metrics.values())).all() or float(summary["joint_prediction_degenerate_fraction"]) != 0.0:
        raise RuntimeError("P5 primary metrics 非有限或 prediction degeneracy 非零")
    return metrics


def _aggregate_metrics(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in seed_metrics.groupby("variant", sort=False):
        if sorted(group["seed"].astype(int).tolist()) != list(FORMAL_SEEDS):
            raise RuntimeError(f"P5 aggregate seeds 不完整: {variant}")
        for metric in PRIMARY_METRICS:
            values = group.sort_values("seed")[metric].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "sample_sd": float(np.std(values, ddof=1)),
                    "seed_20260811": float(values[0]),
                    "seed_20260812": float(values[1]),
                    "seed_20260813": float(values[2]),
                }
            )
    return pd.DataFrame(rows)


def _interaction_table(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    values = _seed_metric_lookup(seed_metrics)
    rows = []
    base = "tf000_c201_anchor"
    for components, variant in PAIRS.items():
        for metric in PRIMARY_METRICS:
            interaction = pair_interaction(
                metric,
                base=values[base][metric],
                arm_a=values[SINGLES[components[0]]][metric],
                arm_b=values[SINGLES[components[1]]][metric],
                pair=values[variant][metric],
            )
            rows.append(_interaction_row(variant, "+".join(components), "pair", metric, interaction))
    for components, variant in TRIPLES.items():
        a, b, c = components
        for metric in PRIMARY_METRICS:
            interaction = triple_interaction(
                metric,
                base=values[base][metric],
                arm_a=values[SINGLES[a]][metric],
                arm_b=values[SINGLES[b]][metric],
                arm_c=values[SINGLES[c]][metric],
                pair_ab=values[_pair_variant(a, b)][metric],
                pair_ac=values[_pair_variant(a, c)][metric],
                pair_bc=values[_pair_variant(b, c)][metric],
                triple=values[variant][metric],
            )
            rows.append(_interaction_row(variant, "+".join(components), "triple", metric, interaction))
    return pd.DataFrame(rows)


def _interaction_row(
    variant: str,
    components: str,
    order: str,
    metric: str,
    interaction: np.ndarray,
) -> dict[str, Any]:
    summary = interaction_summary(interaction)
    return {
        "variant": variant,
        "components": components,
        "order": order,
        "metric": metric,
        "seed_20260811": float(interaction[0]),
        "seed_20260812": float(interaction[1]),
        "seed_20260813": float(interaction[2]),
        **summary,
    }


def _pair_variant(left: str, right: str) -> str:
    target = frozenset((left, right))
    matches = [variant for components, variant in PAIRS.items() if frozenset(components) == target]
    if len(matches) != 1:
        raise RuntimeError(f"P5 pair mapping 不唯一: {left}+{right}")
    return matches[0]


def _eligibility_tables(
    seed_metrics: pd.DataFrame,
    aggregate: pd.DataFrame,
    interactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = _seed_metric_lookup(seed_metrics)
    means = _mean_metric_lookup(aggregate)
    base = means["tf000_c201_anchor"]
    eligibility_rows = []
    capacity_rows = []
    representation_variants = [variant for variant in TF_VARIANTS if TF_VARIANT_REPRESENTATIONS[variant]]
    for variant in representation_variants:
        branch_count = len(TF_VARIANT_REPRESENTATIONS[variant])
        control = CONTROL_BY_BRANCH_COUNT[branch_count]
        guardrail = passes_base_guardrails(means[variant], base)
        metric_passes = []
        for metric in PRIMARY_METRICS:
            comparison = paired_material_improvement(metric, values[variant][metric], values[control][metric])
            capacity_rows.append(
                {
                    "variant": variant,
                    "control": control,
                    "metric": metric,
                    **comparison,
                }
            )
            if bool(comparison["passed"]):
                metric_passes.append(metric)
        positive_interactions = interactions.loc[
            (interactions["variant"] == variant) & interactions["descriptive_positive"], "metric"
        ].tolist()
        eligibility_rows.append(
            {
                "variant": variant,
                "branch_count": branch_count,
                "matched_control": control,
                "base_guardrails_passed": guardrail,
                "capacity_improvement_passed": bool(metric_passes),
                "capacity_improvement_metrics": ";".join(metric_passes),
                "positive_interaction_metrics": ";".join(positive_interactions),
                "qualified": bool(guardrail and metric_passes),
            }
        )
    return pd.DataFrame(eligibility_rows), pd.DataFrame(capacity_rows)


def _candidate_sets(aggregate: pd.DataFrame, eligibility: pd.DataFrame) -> dict[str, list[str]]:
    means = _mean_metric_lookup(aggregate)
    qualified = eligibility.loc[eligibility["qualified"], "variant"].tolist()
    absolute = list(tolerance_pareto_set({variant: means[variant] for variant in qualified})) if qualified else []
    singles = [variant for variant in qualified if len(TF_VARIANT_REPRESENTATIONS[variant]) == 1]
    single_pareto = list(tolerance_pareto_set({variant: means[variant] for variant in singles})) if singles else []
    interaction = eligibility.loc[
        eligibility["qualified"] & (eligibility["positive_interaction_metrics"].astype(str) != ""), "variant"
    ].tolist()
    union = sorted(set(absolute) | set(single_pareto) | set(interaction))
    return {
        "absolute_pareto": absolute,
        "single_pareto": single_pareto,
        "positive_interaction": interaction,
        "union_for_future_p6": union,
    }


def _seed_metric_lookup(seed_metrics: pd.DataFrame) -> dict[str, dict[str, list[float]]]:
    lookup: dict[str, dict[str, list[float]]] = {}
    for variant, group in seed_metrics.groupby("variant", sort=False):
        ordered = group.sort_values("seed")
        lookup[variant] = {metric: ordered[metric].astype(float).tolist() for metric in PRIMARY_METRICS}
    return lookup


def _mean_metric_lookup(aggregate: pd.DataFrame) -> dict[str, dict[str, float]]:
    lookup: dict[str, dict[str, float]] = {}
    for variant, group in aggregate.groupby("variant", sort=False):
        lookup[variant] = {str(row.metric): float(row.mean) for row in group.itertuples(index=False)}
    return lookup


def _iter_tensors(value: Any) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for nested in value.values():
            yield from _iter_tensors(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _iter_tensors(nested)


def _require_clean_git() -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    if dirty:
        raise RuntimeError("CRD-TF P5 必须从干净 Git commit 生成")
    return commit


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
