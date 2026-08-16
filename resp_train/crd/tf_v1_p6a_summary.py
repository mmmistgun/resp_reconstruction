from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch

from resp_train.crd.config import (
    CRD_TF_CACHE_PATH,
    CRD_TF_P6A_PROTOCOL_VERSION,
    FORMAL_SEEDS,
    load_crd_config,
)
from resp_train.crd.tf_v1_model import TF_P6_VARIANT_REPRESENTATIONS, TF_P6_VARIANTS
from resp_train.crd.tf_v1_p5 import METRIC_COLUMNS, SAMPLE_METRIC_COLUMNS
from resp_train.crd.tf_v1_selection import (
    PRIMARY_METRICS,
    paired_material_improvement,
    passes_base_guardrails,
    tolerance_pareto_set,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FORMAL_ROOT = REPO_ROOT / "runs/crd_tf_v1/p6a_formal"
OUTPUT_ROOT = REPO_ROOT / "runs/crd_tf_v1/p6a_validation_summary"
P5_ROOT = REPO_ROOT / "runs/crd_tf_v1/p5_validation_summary"
P5_SUMMARY_SHA256 = "afb6feba1600c9c5e07d713db9cac7c886aa87a033037649d3e9ac32cb753b4e"
P5_MANIFEST_SHA256 = "cecd35d862cc975cf7d99af9ad18fce4576ddb6b9aa561ad88322e3888992a65"
P6A_ACCEPTANCE_RECEIPT = REPO_ROOT / (
    "runs/crd_tf_v1/p6a_acceptance_audit/"
    "9304ae7abd2056c9c28b09702d8fcfc88672a4b53ea412e2922d8d7c7ee821b1/"
    "p6a_acceptance.json"
)
P6A_ACCEPTANCE_RECEIPT_SHA256 = "a23c1dd9aca724ecae3d867429911043a0da1843ede61a3784578abbf99040a9"
P6A_FORMAL_TRAINING_COMMIT = "94033ce66a845d81c19db97114953da5413f60fd"
EXPECTED_VALIDATION_SAMPLES = 2675
UPDATES_PER_EPOCH = 80
PLANNED_TOTAL_UPDATES = 6400

COMPARISONS = (
    ("mws_add_vs_ms", "crd_tf401_mws_add", "crd_tf203_ms"),
    ("mws_add_vs_ctrl3", "crd_tf401_mws_add", "crd_tf_ctrl3"),
    ("mws_gate_vs_mws_add", "crd_tf402_mws_gate", "crd_tf401_mws_add"),
    ("mws_gate_vs_ctrl_gate", "crd_tf402_mws_gate", "crd_tf403_ctrl_gate"),
    ("ctrl_gate_vs_ctrl3", "crd_tf403_ctrl_gate", "crd_tf_ctrl3"),
)
REQUIRED_COMPARISONS = {
    "crd_tf401_mws_add": ("mws_add_vs_ms", "mws_add_vs_ctrl3"),
    "crd_tf402_mws_gate": ("mws_gate_vs_mws_add", "mws_gate_vs_ctrl_gate"),
}


def generate_p6a_summary() -> Path:
    commit = _require_clean_git()
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"P6a summary 已存在，拒绝覆盖: {OUTPUT_ROOT}")
    _audit_frozen_inputs()
    comparator_rows = _load_p5_comparators()
    formal_rows, excluded_rows = _audit_formal_runs()
    seed_metrics = pd.DataFrame(comparator_rows + formal_rows)
    aggregate = _aggregate_metrics(seed_metrics)
    comparisons = _comparison_table(seed_metrics)
    eligibility, decision = _eligibility_and_decision(aggregate, comparisons)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    frames = {
        "formal_run_audit.csv": pd.DataFrame(formal_rows),
        "excluded_incomplete_runs.csv": pd.DataFrame(excluded_rows),
        "seed_metrics.csv": seed_metrics,
        "aggregate_metrics.csv": aggregate,
        "paired_comparisons.csv": comparisons,
        "eligibility.csv": eligibility,
    }
    for filename, frame in frames.items():
        frame.to_csv(OUTPUT_ROOT / filename, index=False)
    summary = {
        "protocol": CRD_TF_P6A_PROTOCOL_VERSION,
        "phase": "p6a_validation_summary",
        "status": "passed",
        "complete": True,
        "evidence_role": "validation-development; not unbiased held-out evidence",
        "research_test_used": False,
        "git_commit": commit,
        "git_dirty": False,
        "formal_training_commit": P6A_FORMAL_TRAINING_COMMIT,
        "formal_run_count": len(formal_rows),
        "excluded_incomplete_run_count": len(excluded_rows),
        "excluded_incomplete_runs": excluded_rows,
        "early_stopped_run_count": int(sum(bool(row["early_stopping_triggered"]) for row in formal_rows)),
        "selected_epoch_min": int(min(row["selected_epoch"] for row in formal_rows)),
        "selected_epoch_max": int(max(row["selected_epoch"] for row in formal_rows)),
        "completed_epoch_min": int(min(row["completed_epoch"] for row in formal_rows)),
        "completed_epoch_max": int(max(row["completed_epoch"] for row in formal_rows)),
        "maximum_peak_reserved_fraction": float(max(row["peak_reserved_fraction"] for row in formal_rows)),
        "decision": decision,
        "no_total_score_used": True,
        "research_test_opened": False,
        "p6b_opened": False,
    }
    summary_path = OUTPUT_ROOT / "p6a_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "protocol": CRD_TF_P6A_PROTOCOL_VERSION,
        "phase": "p6a_validation_summary_manifest",
        "git_commit": commit,
        "git_dirty": False,
        "research_test_used": False,
        "files": {
            path.name: {"sha256": _sha256_file(path), "size_bytes": path.stat().st_size}
            for path in sorted(OUTPUT_ROOT.iterdir())
            if path.is_file()
        },
    }
    (OUTPUT_ROOT / "p6a_summary_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary_path


def _audit_frozen_inputs() -> None:
    if _sha256_file(P5_ROOT / "p5_summary.json") != P5_SUMMARY_SHA256:
        raise RuntimeError("P5 frozen summary SHA-256 不一致")
    manifest_path = P5_ROOT / "p5_summary_manifest.json"
    if _sha256_file(manifest_path) != P5_MANIFEST_SHA256:
        raise RuntimeError("P5 frozen manifest SHA-256 不一致")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for filename, identity in manifest.get("files", {}).items():
        path = P5_ROOT / filename
        if path.stat().st_size != int(identity["size_bytes"]) or _sha256_file(path) != identity["sha256"]:
            raise RuntimeError(f"P5 frozen output identity 不一致: {path}")
    if _sha256_file(P6A_ACCEPTANCE_RECEIPT) != P6A_ACCEPTANCE_RECEIPT_SHA256:
        raise RuntimeError("P6a acceptance receipt SHA-256 不一致")
    receipt = json.loads(P6A_ACCEPTANCE_RECEIPT.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "passed"
        or receipt.get("complete") is not True
        or receipt.get("batch_decision") != "128x1"
        or receipt.get("research_test_used") is not False
    ):
        raise RuntimeError("P6a acceptance receipt identity 不合格")


def _load_p5_comparators() -> list[dict[str, Any]]:
    frame = pd.read_csv(P5_ROOT / "seed_metrics.csv")
    variants = ("tf000_c201_anchor", "crd_tf203_ms", "crd_tf_ctrl3")
    subset = frame.loc[frame["variant"].isin(variants)].copy()
    for variant in variants:
        seeds = sorted(subset.loc[subset["variant"] == variant, "seed"].astype(int).tolist())
        if seeds != list(FORMAL_SEEDS):
            raise RuntimeError(f"P6a comparator seeds 不完整: {variant}={seeds}")
    columns = ["variant", "seed", "run_dir", "selected_epoch", "checkpoint_sha256", "training_commit",
               "peak_reserved_fraction", *PRIMARY_METRICS]
    return subset[columns].to_dict(orient="records")


def _audit_formal_runs() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    required = (
        "audit.csv", "checkpoint_best_local_rr.pt", "checkpoint_final.pt", "config.yaml", "metrics.csv",
        "metrics_summary.csv", "optimizer_parameter_groups.json", "run_manifest.json", "runtime_summary.json",
        "train_history.csv",
    )
    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for variant in TF_P6_VARIANTS:
        for seed in FORMAL_SEEDS:
            parent = FORMAL_ROOT / variant / f"seed_{seed}"
            run_dirs = sorted(path for path in parent.glob("20*") if path.is_dir())
            complete = [path for path in run_dirs if all((path / name).is_file() for name in required)]
            if len(complete) != 1:
                raise RuntimeError(f"P6a 要求 {variant} seed={seed} 恰有一个完整 run，实际 {len(complete)}")
            for path in run_dirs:
                if path not in complete:
                    _audit_incomplete_run(path, variant, int(seed))
                    excluded.append(
                        {"variant": variant, "seed": int(seed), "run_dir": str(path.resolve()),
                         "reason": "incomplete_lifecycle",
                         "present_files": ";".join(sorted(item.name for item in path.iterdir() if item.is_file()))}
                    )
            rows.append(_audit_one_run(complete[0], variant, int(seed)))
    return rows, excluded


def _audit_incomplete_run(run_dir: Path, variant: str, seed: int) -> None:
    if not (run_dir / "config.yaml").is_file() or not (run_dir / "run_manifest.json").is_file():
        raise RuntimeError(f"无法识别的 P6a incomplete run: {run_dir}")
    cfg = load_crd_config(run_dir / "config.yaml")
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if (
        str(cfg.model.variant) != variant or int(cfg.training.seed) != seed
        or manifest.get("git_commit") != P6A_FORMAL_TRAINING_COMMIT
        or manifest.get("git_dirty") is not False
    ):
        raise RuntimeError(f"P6a incomplete run identity 不合格: {run_dir}")


def _audit_one_run(run_dir: Path, variant: str, seed: int) -> dict[str, Any]:
    cfg = load_crd_config(run_dir / "config.yaml")
    if (
        str(cfg.model.variant) != variant
        or list(cfg.model.tf_representations) != list(TF_P6_VARIANT_REPRESENTATIONS[variant])
        or str(cfg.protocol.run_role) != "formal"
        or str(cfg.protocol.execution_gate) != "p6a_formal"
        or int(cfg.training.seed) != seed or int(cfg.model.initialization_seed) != seed
        or (int(cfg.training.epochs), int(cfg.training.batch_size), int(cfg.training.gradient_accumulation_steps))
        != (80, 128, 1)
        or cfg.training.early_stopping_enabled is not True
        or int(cfg.training.early_stopping_patience) != 30
        or float(cfg.training.early_stopping_min_delta) != 0.0
        or str(cfg.data.tf_cache_path) != CRD_TF_CACHE_PATH
        or any(cfg.data.get(name) is not None for name in ("max_train_windows", "max_val_windows", "max_test_windows"))
    ):
        raise RuntimeError(f"P6a formal config identity 不一致: {run_dir}")
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("git_commit") != P6A_FORMAL_TRAINING_COMMIT
        or manifest.get("git_dirty") is not False
        or manifest.get("protocol") != CRD_TF_P6A_PROTOCOL_VERSION
        or manifest.get("stage") != "tf_p6a" or manifest.get("run_role") != "formal"
    ):
        raise RuntimeError(f"P6a formal manifest identity 不一致: {run_dir}")
    history = pd.read_csv(run_dir / "train_history.csv")
    completed_epoch, selected_epoch, triggered = _audit_early_stopping_history(history, run_dir)
    checkpoint_hashes = _audit_checkpoints(run_dir, completed_epoch, selected_epoch, triggered)
    metrics_frame = pd.read_csv(run_dir / "metrics.csv")
    summary = pd.read_csv(run_dir / "metrics_summary.csv").iloc[0]
    metrics = _metrics_from_summary(summary)
    if len(metrics_frame) != EXPECTED_VALIDATION_SAMPLES:
        raise RuntimeError(f"P6a validation sample count 错误: {run_dir}")
    for metric, column in SAMPLE_METRIC_COLUMNS.items():
        values = metrics_frame[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or not np.isclose(float(np.mean(values)), metrics[metric], rtol=0.0, atol=1e-12):
            raise RuntimeError(f"P6a per-sample/summary {metric} 不一致: {run_dir}")
    if bool(metrics_frame["joint_prediction_degenerate"].astype(bool).any()):
        raise RuntimeError(f"P6a prediction degeneracy 非零: {run_dir}")
    if not np.isclose(float(history.loc[history["epoch"] == selected_epoch, "val_local_rr_mae"].iloc[0]),
                      metrics["local_rr_mae"], rtol=0.0, atol=1e-12):
        raise RuntimeError(f"P6a selected history/metrics Local RR 不一致: {run_dir}")
    runtime = json.loads((run_dir / "runtime_summary.json").read_text(encoding="utf-8"))
    peak_reserved_fraction = float(runtime["peak_reserved_fraction"])
    if not np.isfinite(peak_reserved_fraction) or peak_reserved_fraction <= 0.0:
        raise RuntimeError(f"P6a runtime summary 不合格: {run_dir}")
    return {
        "variant": variant, "seed": seed, "run_dir": str(run_dir.resolve()),
        "selected_epoch": selected_epoch, "completed_epoch": completed_epoch,
        "optimizer_updates": completed_epoch * UPDATES_PER_EPOCH,
        "early_stopping_triggered": triggered,
        "checkpoint_sha256": checkpoint_hashes["checkpoint_best_local_rr.pt"],
        "final_checkpoint_sha256": checkpoint_hashes["checkpoint_final.pt"],
        "training_commit": P6A_FORMAL_TRAINING_COMMIT,
        "peak_reserved_fraction": peak_reserved_fraction, **metrics,
    }


def _audit_early_stopping_history(history: pd.DataFrame, run_dir: Path) -> tuple[int, int, bool]:
    numeric = history.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    if history.empty or len(history) > 80 or not np.isfinite(numeric).all():
        raise RuntimeError(f"P6a history 长度或 finite 错误: {run_dir}")
    epochs = history["epoch"].astype(int).tolist()
    if epochs != list(range(1, len(history) + 1)):
        raise RuntimeError(f"P6a history epoch 不连续: {run_dir}")
    best = float("inf")
    wait = 0
    selected_epoch = -1
    for row in history.itertuples(index=False):
        value = float(row.val_local_rr_mae)
        improved = value < best
        if improved:
            best, wait, selected_epoch = value, 0, int(row.epoch)
        else:
            wait += 1
        triggered = wait >= 30
        if (
            int(row.optimizer_update) != int(row.epoch) * UPDATES_PER_EPOCH
            or int(row.early_stopping_improved) != int(improved)
            or int(row.early_stopping_wait) != wait
            or int(row.early_stopping_triggered) != int(triggered)
        ):
            raise RuntimeError(f"P6a early-stop history 语义错误: {run_dir}")
        if triggered and int(row.epoch) != len(history):
            raise RuntimeError(f"P6a history 在触发 early stop 后仍继续: {run_dir}")
    completed_epoch = len(history)
    final_triggered = bool(int(history.iloc[-1]["early_stopping_triggered"]))
    if completed_epoch < 80 and not final_triggered:
        raise RuntimeError(f"P6a 未到最大预算却未触发 early stop: {run_dir}")
    return completed_epoch, selected_epoch, final_triggered


def _audit_checkpoints(
    run_dir: Path, completed_epoch: int, selected_epoch: int, triggered: bool
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for filename, expected_epoch, expected_update, expected_triggered in (
        ("checkpoint_best_local_rr.pt", selected_epoch, selected_epoch * UPDATES_PER_EPOCH, False),
        ("checkpoint_final.pt", completed_epoch, completed_epoch * UPDATES_PER_EPOCH, triggered),
    ):
        path = run_dir / filename
        checkpoint = torch.load(path, map_location="cpu")
        extra = checkpoint.get("extra_state", {})
        early = extra.get("early_stopping", {})
        tensors = list(_iter_tensors(checkpoint.get("model_state_dict"))) + list(
            _iter_tensors(checkpoint.get("optimizer_state_dict"))
        )
        if (
            int(checkpoint.get("epoch", -1)) != expected_epoch
            or int(extra.get("update_index", -1)) != expected_update
            or int(extra.get("total_updates", -1)) != PLANNED_TOTAL_UPDATES
            or early.get("enabled") is not True or int(early.get("patience", -1)) != 30
            or float(early.get("min_delta", np.nan)) != 0.0
            or early.get("triggered") is not expected_triggered
            or not tensors or not all(bool(torch.isfinite(tensor).all()) for tensor in tensors)
        ):
            raise RuntimeError(f"P6a checkpoint 不合格: {path}")
        hashes[filename] = _sha256_file(path)
        del checkpoint, tensors
    return hashes


def _metrics_from_summary(summary: pd.Series) -> dict[str, float]:
    if int(summary["n_samples"]) != EXPECTED_VALIDATION_SAMPLES:
        raise RuntimeError("P6a metrics summary sample count 错误")
    metrics = {name: float(summary[column]) for name, column in METRIC_COLUMNS.items()}
    if not np.isfinite(list(metrics.values())).all() or float(summary["joint_prediction_degenerate_fraction"]) != 0.0:
        raise RuntimeError("P6a primary metrics 非有限或 prediction degeneracy 非零")
    return metrics


def _aggregate_metrics(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in seed_metrics.groupby("variant", sort=False):
        ordered = group.sort_values("seed")
        if ordered["seed"].astype(int).tolist() != list(FORMAL_SEEDS):
            raise RuntimeError(f"P6a aggregate seeds 不完整: {variant}")
        for metric in PRIMARY_METRICS:
            values = ordered[metric].to_numpy(dtype=np.float64)
            rows.append({"variant": variant, "metric": metric, "mean": float(np.mean(values)),
                         "sample_sd": float(np.std(values, ddof=1)),
                         **{f"seed_{seed}": float(value) for seed, value in zip(FORMAL_SEEDS, values)}})
    return pd.DataFrame(rows)


def _comparison_table(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    values = _seed_lookup(seed_metrics)
    rows = []
    for name, candidate, reference in COMPARISONS:
        for metric in PRIMARY_METRICS:
            result = paired_material_improvement(metric, values[candidate][metric], values[reference][metric])
            utility_delta = np.asarray(values[candidate][metric])
            reference_values = np.asarray(values[reference][metric])
            utility_delta = reference_values - utility_delta if metric != "signed_pcc" else utility_delta - reference_values
            rows.append({"comparison": name, "candidate": candidate, "reference": reference, "metric": metric,
                         "seed_20260811": float(utility_delta[0]), "seed_20260812": float(utility_delta[1]),
                         "seed_20260813": float(utility_delta[2]), **result})
    return pd.DataFrame(rows)


def _eligibility_and_decision(
    aggregate: pd.DataFrame, comparisons: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    means = _mean_lookup(aggregate)
    base = means["tf000_c201_anchor"]
    rows = []
    qualified: list[str] = []
    for variant, required in REQUIRED_COMPARISONS.items():
        comparison_passes = {
            name: comparisons.loc[(comparisons["comparison"] == name) & comparisons["passed"], "metric"].tolist()
            for name in required
        }
        guardrail = passes_base_guardrails(means[variant], base)
        is_qualified = bool(guardrail and all(comparison_passes[name] for name in required))
        if is_qualified:
            qualified.append(variant)
        rows.append({"variant": variant, "base_guardrails_passed": guardrail,
                     "required_comparisons": ";".join(required),
                     "comparison_passed_metrics": json.dumps(comparison_passes, sort_keys=True),
                     "qualified": is_qualified})
    pareto = list(tolerance_pareto_set({variant: means[variant] for variant in qualified})) if qualified else []
    decision = {
        "qualified_p6a_candidates": qualified,
        "qualified_tolerance_pareto": pareto,
        "selected_for_future_lock": pareto,
        "fallback_if_none": ["crd_tf101_m", "crd_tf102_w", "crd_tf203_ms"],
        "decision": "advance_p6a_candidate" if pareto else "no_p6a_candidate_retain_p5_pool",
        "comparison_passed_metrics": {
            name: comparisons.loc[(comparisons["comparison"] == name) & comparisons["passed"], "metric"].tolist()
            for name, _, _ in COMPARISONS
        },
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
        raise RuntimeError("CRD-TF P6a summary 必须从干净 Git commit 生成")
    return commit


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

