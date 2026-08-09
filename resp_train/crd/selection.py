from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from resp_train.crd.candidate_lock import DEFAULT_CANDIDATE_LOCK, verify_candidate_lock
from resp_train.crd.confirmation import (
    ACCESS_RECEIPT_FILENAME,
    DEFAULT_S1C_OUTPUT_ROOT,
    MANIFEST_FILENAME,
    METRICS_FILENAME,
    S1C_EXPECTED_TEST_SAMP_IDS,
    S1C_EXPECTED_TEST_WINDOWS,
    S1C_PROTOCOL_VERSION,
    SUMMARY_FILENAME,
    _assert_clean_repository,
    _validate_confirmation_metrics,
    _validate_s1c_lock,
)
from resp_train.metrics.task import summarize_task_metrics
from resp_train.utils.run import save_execution_manifest


SEED_SUMMARY_FILENAME = "s1c_seed_summary.csv"
VARIANT_SUMMARY_FILENAME = "s1c_variant_summary.csv"
SELECTION_FILENAME = "s1c_selection.json"
SELECTION_MANIFEST_FILENAME = "s1c_selection_manifest.json"

_IDENTITY_COLUMNS = {"role", "variant", "seed", "selected_epoch", "checkpoint_path"}


def summarize_crd_s1c(
    *,
    lock_path: str | Path = DEFAULT_CANDIDATE_LOCK,
    output_root: str | Path = DEFAULT_S1C_OUTPUT_ROOT,
) -> Path:
    """在 12 项结果齐备后，按 candidate lock 的冻结规则生成一次性 S1C 选择结果。"""

    verification = verify_candidate_lock(lock_path)
    _validate_s1c_lock(verification)
    _assert_clean_repository()
    output_root = Path(output_root)
    final_paths = {
        "seed": output_root / SEED_SUMMARY_FILENAME,
        "variant": output_root / VARIANT_SUMMARY_FILENAME,
        "selection": output_root / SELECTION_FILENAME,
        "manifest": output_root / SELECTION_MANIFEST_FILENAME,
    }
    existing = [path for path in final_paths.values() if path.exists()]
    if existing:
        raise FileExistsError("S1C 选择产物禁止覆盖: " + ", ".join(map(str, existing)))

    seed_rows: list[dict[str, Any]] = []
    for record in verification.records:
        result_dir = output_root / str(record["variant"]) / f"seed_{int(record['seed'])}"
        artifacts = {
            "access": result_dir / ACCESS_RECEIPT_FILENAME,
            "metrics": result_dir / METRICS_FILENAME,
            "summary": result_dir / SUMMARY_FILENAME,
            "manifest": result_dir / MANIFEST_FILENAME,
        }
        missing = [path for path in artifacts.values() if not path.exists()]
        if missing:
            raise FileNotFoundError("S1C 12 项结果尚未齐备，缺失: " + ", ".join(map(str, missing)))
        _validate_result_manifest(artifacts["access"], artifacts["manifest"], record, verification.lock_sha256)

        metrics = pd.read_csv(artifacts["metrics"])
        _validate_confirmation_metrics(
            metrics,
            variant=str(record["variant"]),
            expected_test_windows=S1C_EXPECTED_TEST_WINDOWS,
            expected_test_samp_ids=S1C_EXPECTED_TEST_SAMP_IDS,
        )
        recomputed = summarize_task_metrics(metrics)
        stored = pd.read_csv(artifacts["summary"])
        try:
            pd.testing.assert_frame_equal(
                stored,
                recomputed,
                check_dtype=False,
                check_exact=False,
                rtol=1e-12,
                atol=1e-12,
            )
        except AssertionError as exc:
            raise RuntimeError(f"S1C summary 与逐 sample metrics 重算不一致: {artifacts['summary']}") from exc
        seed_rows.append(
            {
                "role": str(record["role"]),
                "variant": str(record["variant"]),
                "seed": int(record["seed"]),
                "selected_epoch": int(record["selected_epoch"]),
                "checkpoint_path": str(record["checkpoint_path"]),
                **recomputed.iloc[0].to_dict(),
            }
        )

    seed_summary = pd.DataFrame(seed_rows)
    variant_summary, decision = apply_s1c_selection(seed_summary, verification.payload)
    output_root.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    temporary_paths = {name: output_root / f".{path.name}.{token}.tmp" for name, path in final_paths.items()}
    try:
        seed_summary.to_csv(temporary_paths["seed"], index=False)
        variant_summary.to_csv(temporary_paths["variant"], index=False)
        temporary_paths["selection"].write_text(
            json.dumps(
                {
                    "protocol": S1C_PROTOCOL_VERSION,
                    "candidate_lock": str(verification.lock_path),
                    "candidate_lock_id": str(verification.payload["lock_id"]),
                    "candidate_lock_sha256": verification.lock_sha256,
                    **decision,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        save_execution_manifest(
            temporary_paths["manifest"],
            task="crd_v1_s1c_research_confirmation",
            phase="s1c_frozen_selection",
            protocol=S1C_PROTOCOL_VERSION,
            candidate_lock=str(verification.lock_path),
            candidate_lock_sha256=verification.lock_sha256,
            complete_checkpoint_count=len(seed_summary),
            pareto_set=decision["pareto_set"],
        )
        for name in ("seed", "variant", "selection", "manifest"):
            os.replace(temporary_paths[name], final_paths[name])
    finally:
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)
    return final_paths["selection"]


def apply_s1c_selection(
    seed_summary: pd.DataFrame,
    lock_payload: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """纯函数实现冻结的 eligibility 与 primary Pareto，不使用 secondary 改写结果。"""

    rule = lock_payload["selection_rule"]
    primary = rule["primary_metrics"]
    required = _IDENTITY_COLUMNS | {str(item["name"]) for item in primary}
    missing = sorted(required - set(seed_summary.columns))
    if missing:
        raise ValueError(f"S1C seed summary 缺少选择字段: {missing}")
    if len(seed_summary) != 12:
        raise ValueError(f"S1C 选择要求恰有 12 个 seed summaries，实际 {len(seed_summary)}")

    expected_seeds = {int(seed) for seed in lock_payload["fixed_seeds"]}
    variant_order = [str(item["variant"]) for item in lock_payload["variants"]]
    aggregate_columns = [
        column
        for column in seed_summary.select_dtypes(include="number").columns
        if column not in {"seed", "selected_epoch", "n_samples"} and not column.endswith("_n")
    ]
    variant_rows: list[dict[str, Any]] = []
    for variant_item in lock_payload["variants"]:
        variant = str(variant_item["variant"])
        rows = seed_summary.loc[seed_summary["variant"].eq(variant)].sort_values("seed")
        actual_seeds = set(pd.to_numeric(rows["seed"]).astype(int))
        if len(rows) != 3 or actual_seeds != expected_seeds:
            raise ValueError(f"S1C {variant} seed 集不完整: {sorted(actual_seeds)}")
        row: dict[str, Any] = {"role": str(variant_item["role"]), "variant": variant}
        row.update({column: float(pd.to_numeric(rows[column]).mean()) for column in aggregate_columns})
        variant_rows.append(row)
    variant_summary = pd.DataFrame(variant_rows)

    anchor_variant = str(lock_payload["anchor_variant"])
    anchor = variant_summary.loc[variant_summary["variant"].eq(anchor_variant)].iloc[0]
    thresholds = rule["candidate_eligibility"]
    eligibility: dict[str, dict[str, Any]] = {}
    for variant in variant_order:
        role = str(variant_summary.loc[variant_summary["variant"].eq(variant), "role"].iloc[0])
        if role == "reference":
            eligibility[variant] = {"eligible": False, "reason": "reference_only"}
            continue
        if variant == anchor_variant:
            eligibility[variant] = {"eligible": True, "reason": "anchor_automatically_eligible"}
            continue
        candidate = variant_summary.loc[variant_summary["variant"].eq(variant)].iloc[0]
        local_improvement = _relative_improvement(
            float(anchor["local_rr_mae_bpm_mean"]),
            float(candidate["local_rr_mae_bpm_mean"]),
        )
        trajectory_worsening = _relative_worsening(
            float(anchor["envelope_trajectory_mae_mean"]),
            float(candidate["envelope_trajectory_mae_mean"]),
        )
        pcc_drop = float(anchor["lag_aware_signed_pcc_mean"]) - float(candidate["lag_aware_signed_pcc_mean"])
        anchor_local = seed_summary.loc[
            seed_summary["variant"].eq(anchor_variant), ["seed", "local_rr_mae_bpm_mean"]
        ].set_index("seed")["local_rr_mae_bpm_mean"]
        candidate_local = seed_summary.loc[
            seed_summary["variant"].eq(variant), ["seed", "local_rr_mae_bpm_mean"]
        ].set_index("seed")["local_rr_mae_bpm_mean"]
        paired_improvement_count = int((candidate_local.loc[sorted(expected_seeds)] < anchor_local.loc[sorted(expected_seeds)]).sum())
        checks = {
            "local_rr_relative_improvement_pass": local_improvement
            >= float(thresholds["local_rr_relative_improvement_min"]),
            "local_rr_paired_seed_pass": paired_improvement_count
            >= int(thresholds["local_rr_paired_seed_improvement_min_count"]),
            "signed_pcc_drop_pass": pcc_drop <= float(thresholds["signed_pcc_absolute_drop_max"]),
            "trajectory_worsening_pass": trajectory_worsening
            <= float(thresholds["trajectory_relative_worsening_max"]),
        }
        eligibility[variant] = {
            "eligible": bool(all(checks.values())),
            "local_rr_relative_improvement": local_improvement,
            "local_rr_paired_seed_improvement_count": paired_improvement_count,
            "signed_pcc_absolute_drop": pcc_drop,
            "trajectory_relative_worsening": trajectory_worsening,
            **checks,
        }

    for variant, details in eligibility.items():
        mask = variant_summary["variant"].eq(variant)
        variant_summary.loc[mask, "eligible"] = bool(details["eligible"])
        for key, value in details.items():
            if key != "eligible":
                variant_summary.loc[mask, key] = value

    eligible_variants = [
        variant
        for variant in variant_order
        if eligibility[variant]["eligible"] and eligibility[variant].get("reason") != "reference_only"
    ]
    pareto_set = [
        candidate
        for candidate in eligible_variants
        if not any(
            other != candidate and _dominates(variant_summary, other, candidate, primary)
            for other in eligible_variants
        )
    ]
    variant_summary["pareto_nondominated"] = variant_summary["variant"].isin(pareto_set)
    return variant_summary, {
        "complete_checkpoint_count": int(len(seed_summary)),
        "eligibility": eligibility,
        "eligible_variants": eligible_variants,
        "pareto_set": pareto_set,
        "outcome": "unique_nondominated_candidate" if len(pareto_set) == 1 else "retain_pareto_set",
        "secondary_metrics_used_for_selection": False,
    }


def _validate_result_manifest(
    access_path: Path,
    manifest_path: Path,
    record: dict[str, Any],
    lock_sha256: str,
) -> None:
    access = json.loads(access_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "protocol": S1C_PROTOCOL_VERSION,
        "evaluation_split": "test",
        "candidate_lock_sha256": lock_sha256,
        "checkpoint_sha256": str(record["checkpoint_sha256"]),
        "variant": str(record["variant"]),
        "seed": int(record["seed"]),
        "selected_epoch": int(record["selected_epoch"]),
    }
    for path, payload, phase in (
        (access_path, access, "research_test_access_started"),
        (manifest_path, manifest, "s1c_research_confirmation"),
    ):
        if payload.get("phase") != phase:
            raise RuntimeError(f"S1C manifest phase 异常: {path}")
        for key, value in expected.items():
            if payload.get(key) != value:
                raise RuntimeError(f"S1C manifest {key} 不一致: {path}")
        if payload.get("git_dirty") is not False:
            raise RuntimeError(f"S1C manifest 必须记录 git_dirty=false: {path}")
    if int(manifest.get("observed_test_windows", -1)) != S1C_EXPECTED_TEST_WINDOWS:
        raise RuntimeError(f"S1C manifest test windows 异常: {manifest_path}")
    if int(manifest.get("observed_test_samp_ids", -1)) != S1C_EXPECTED_TEST_SAMP_IDS:
        raise RuntimeError(f"S1C manifest test samp IDs 异常: {manifest_path}")


def _dominates(
    variant_summary: pd.DataFrame,
    left_variant: str,
    right_variant: str,
    primary: list[dict[str, str]],
) -> bool:
    left = variant_summary.loc[variant_summary["variant"].eq(left_variant)].iloc[0]
    right = variant_summary.loc[variant_summary["variant"].eq(right_variant)].iloc[0]
    no_worse = True
    strictly_better = False
    for item in primary:
        name = str(item["name"])
        left_value = float(left[name])
        right_value = float(right[name])
        if item["direction"] == "minimize":
            no_worse &= left_value <= right_value
            strictly_better |= left_value < right_value
        elif item["direction"] == "maximize":
            no_worse &= left_value >= right_value
            strictly_better |= left_value > right_value
        else:
            raise ValueError(f"未知 Pareto 方向: {item['direction']}")
    return bool(no_worse and strictly_better)


def _relative_improvement(anchor: float, candidate: float) -> float:
    if anchor <= 0:
        raise ValueError("相对改善 comparator 必须为正")
    return (anchor - candidate) / anchor


def _relative_worsening(anchor: float, candidate: float) -> float:
    if anchor <= 0:
        raise ValueError("相对恶化 comparator 必须为正")
    return (candidate - anchor) / anchor
