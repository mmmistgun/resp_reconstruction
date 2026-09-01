from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import numpy as np
import pandas as pd


PAPER_EVIDENCE_PROTOCOL_ID = "paper-evidence-closure-v1-20260901"
P0_SCHEMA_VERSION = "paper-evidence-p0-comparison-audit-v1"
PRIMARY_METRICS = (
    "whole_rr_abs_error_bpm",
    "local_rr_mae_bpm",
    "envelope_trajectory_mae",
    "global_envelope_modulation_error",
    "lag_aware_signed_pcc",
)
COMMON_CONTRACT = {
    "dataset_family": "research_v2_admitted_180s",
    "evaluation_split": "independent_test",
    "expected_test_rows": 2310,
    "target_carrier": "target_waveform_segment_soft_z_key",
    "sample_rate_hz": 100,
    "window_samples": 18000,
    "projection": "fft_hard_0.05_0.70hz_center_rms_eps1e-8",
    "primary_metric_set": "respiration_primary_v2_five",
    "aggregation": "sample_direct_mean_then_seed_mean_sample_sd_ddof1",
    "prediction_degradation": "rr_39bpm_pcc_minus1_checkpoint_nonfinite_fail",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def dataset_row_id_sha256(values: pd.Series | np.ndarray) -> str:
    row_ids = np.asarray(values, dtype=np.int64)
    if row_ids.ndim != 1 or row_ids.size == 0:
        raise ValueError("dataset_row_id 必须是一维非空数组")
    if np.unique(row_ids).size != row_ids.size:
        raise ValueError("dataset_row_id 不得重复")
    return hashlib.sha256(np.sort(row_ids).tobytes(order="C")).hexdigest()


def load_p0_source_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != PAPER_EVIDENCE_PROTOCOL_ID:
        raise ValueError("P0 source manifest protocol_id 不一致")
    if payload.get("schema_version") != P0_SCHEMA_VERSION:
        raise ValueError("P0 source manifest schema_version 不一致")
    methods = payload.get("methods")
    if not isinstance(methods, list) or not methods:
        raise ValueError("P0 source manifest methods 不得为空")
    ids = [str(item.get("method_id", "")) for item in methods]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("P0 method_id 必须非空且唯一")
    return payload


def run_p0_comparison_audit(
    *,
    source_manifest_path: str | Path,
    output_dir: str | Path,
    repo_root: str | Path,
    require_clean_git: bool = True,
) -> Path:
    """只读审计冻结逐样本/汇总产物；不加载 checkpoint、prediction 数组或数据集。"""

    root = Path(repo_root).resolve()
    source_path = Path(source_manifest_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"P0 输出禁止覆盖: {output}")
    if require_clean_git:
        _assert_clean_git(root)
    source = load_p0_source_manifest(source_path)

    temporary = output.parent / f".{output.name}.{uuid4().hex}.tmp"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        artifacts, compatibility, table = audit_methods(source, repo_root=root)
        rules = dict(source.get("representative_method_rule", {}))
        _validate_representative_rule(rules, compatibility)
        artifacts.to_csv(temporary / "source_artifact_audit.csv", index=False)
        compatibility.to_csv(temporary / "protocol_compatibility.csv", index=False)
        table.to_csv(temporary / "paper_primary_metrics_table.csv", index=False)
        (temporary / "representative_method_rule.json").write_text(
            json.dumps(rules, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        files = {
            path.name: {"sha256": sha256_file(path), "size_bytes": int(path.stat().st_size)}
            for path in sorted(temporary.iterdir())
            if path.is_file()
        }
        manifest = {
            "protocol_id": PAPER_EVIDENCE_PROTOCOL_ID,
            "schema_version": P0_SCHEMA_VERSION,
            "source_manifest": str(source_path),
            "source_manifest_sha256": sha256_file(source_path),
            "read_only_source_audit": True,
            "checkpoint_evaluation_used": False,
            "prediction_generation_used": False,
            "dataset_signal_or_target_read": False,
            "research_test_signal_or_target_read": False,
            "files": files,
            "decision": "audit_complete",
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output / "manifest.json"


def audit_methods(source: Mapping[str, Any], *, repo_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(repo_root).resolve()
    artifact_rows: list[dict[str, Any]] = []
    compatibility_rows: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    common_test_row_hash: str | None = None

    for method in source["methods"]:
        method_id = str(method["method_id"])
        contracts = dict(method.get("contracts", {}))
        scope = str(method.get("quality_scope", "independent_test"))
        reasons: list[str] = []
        if scope == "independent_test":
            for key, expected in COMMON_CONTRACT.items():
                if contracts.get(key) != expected:
                    reasons.append(f"{key}:unproven_or_mismatch")
        elif scope == "validation_only":
            if method.get("paper_role") != "validation-only efficiency trade-off":
                reasons.append("validation_only_role:mismatch")
        else:
            reasons.append("quality_scope:unknown")

        records = _expanded_records(method)
        if not isinstance(records, list) or not records:
            reasons.append("records:missing")
            records = []
        deterministic = bool(method.get("deterministic", False))
        if not str(contracts.get("input_carrier", "")):
            reasons.append("input_carrier:unproven")
        expected_selector = (
            "not_applicable_deterministic"
            if deterministic
            else "full_validation_local_rr_strict_lower_tie_earlier"
        )
        if contracts.get("checkpoint_selector") != expected_selector:
            reasons.append("checkpoint_selector:unproven_or_mismatch")
        if contracts.get("prediction_uses_target") is not False:
            reasons.append("prediction_uses_target:unproven_or_true")
        expected_seed_count = 0 if deterministic else int(method.get("expected_seed_count", 3))
        expected_record_count = int(method.get("expected_record_count", 1 if deterministic else expected_seed_count))
        observed_seeds = [record.get("seed") for record in records if record.get("seed") is not None]
        if deterministic and observed_seeds:
            reasons.append("deterministic_method_has_seed")
        if not deterministic and (
            len(records) != expected_record_count or len(set(observed_seeds)) != expected_seed_count
        ):
            reasons.append("trained_seed_set_incomplete")

        seed_summaries: list[dict[str, float]] = []
        method_row_hash: str | None = None
        for record in records:
            audit, summary, row_hash, record_reasons = _audit_record(
                method_id=method_id,
                record=record,
                method=method,
                repo_root=root,
            )
            artifact_rows.extend(audit)
            reasons.extend(record_reasons)
            if summary is not None:
                seed_summaries.append(summary)
            if scope == "independent_test" and row_hash is not None:
                if method_row_hash is None:
                    method_row_hash = row_hash
                elif row_hash != method_row_hash:
                    reasons.append("within_method_dataset_row_id_set_mismatch")

        if scope == "independent_test" and method_row_hash is not None:
            if common_test_row_hash is None:
                common_test_row_hash = method_row_hash
            elif method_row_hash != common_test_row_hash:
                reasons.append("cross_method_dataset_row_id_set_mismatch")

        compatible = not reasons
        lock_confirmed = bool(method.get("conclusion_lock_confirmed", True))
        include = bool(method.get("include_in_primary_table", False))
        table_eligible = compatible and lock_confirmed and include and bool(seed_summaries)
        compatibility_rows.append(
            {
                "method_id": method_id,
                "audited_candidate_ids": "|".join(_audited_candidate_ids(method)),
                "audited_candidate_count": len(_audited_candidate_ids(method)),
                "paper_role": str(method.get("paper_role", "")),
                "quality_scope": scope,
                "deterministic": deterministic,
                "expected_seed_count": expected_seed_count,
                "expected_record_count": expected_record_count,
                "observed_record_count": len(records),
                "dataset_row_id_sha256": method_row_hash,
                "compatibility": "compatible" if compatible else "incompatible",
                "conclusion_lock_confirmed": lock_confirmed,
                "predeclared_for_primary_table": include,
                "primary_table_eligible": table_eligible,
                "reason": ";".join(sorted(set(reasons))),
            }
        )
        if table_eligible:
            table_rows.append(_aggregate_method(method, seed_summaries))

    artifacts = pd.DataFrame.from_records(artifact_rows)
    compatibility = pd.DataFrame.from_records(compatibility_rows)
    table = pd.DataFrame.from_records(table_rows)
    return artifacts, compatibility, table


def _audit_record(
    *,
    method_id: str,
    record: Mapping[str, Any],
    method: Mapping[str, Any],
    repo_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, float] | None, str | None, list[str]]:
    audit_rows: list[dict[str, Any]] = []
    reasons: list[str] = []
    record_identity = str(
        record.get(
            "record_identity",
            dict(record.get("sample_filters", {})).get("candidate_id", method_id),
        )
    )
    sample_path = _resolve_source_path(repo_root, record.get("sample_metrics"))
    summary_path = _resolve_source_path(repo_root, record.get("summary"))
    for kind, path, expected in (
        ("sample_metrics", sample_path, record.get("sample_metrics_sha256")),
        ("summary", summary_path, record.get("summary_sha256")),
    ):
        exists = bool(path and path.is_file())
        actual_sha = sha256_file(path) if exists else None
        if not exists:
            reasons.append(f"{kind}:missing")
        elif expected is not None and actual_sha != str(expected):
            reasons.append(f"{kind}:sha256_mismatch")
        audit_rows.append(
            {
                "method_id": method_id,
                "record_identity": record_identity,
                "seed": record.get("seed"),
                "artifact_kind": kind,
                "path": str(path) if path is not None else None,
                "exists": exists,
                "size_bytes": int(path.stat().st_size) if exists else None,
                "sha256": actual_sha,
                "expected_sha256": expected,
                "hash_match": bool(exists and (expected is None or actual_sha == str(expected))),
            }
        )
    if sample_path is None or not sample_path.is_file() or summary_path is None or not summary_path.is_file():
        return audit_rows, None, None, reasons

    frame = _filter_frame(pd.read_csv(sample_path), record.get("sample_filters"), label="sample_metrics")
    required = {"dataset_row_id", *PRIMARY_METRICS}
    missing = sorted(required - set(frame.columns))
    if missing:
        reasons.append("sample_metrics_columns:" + ",".join(missing))
        return audit_rows, None, None, reasons
    expected_rows = int(method.get("contracts", {}).get("expected_test_rows", len(frame)))
    if len(frame) != expected_rows:
        reasons.append(f"sample_row_count:{len(frame)}!={expected_rows}")
    try:
        row_hash = dataset_row_id_sha256(frame["dataset_row_id"])
    except ValueError as exc:
        reasons.append(f"dataset_row_id:{exc}")
        row_hash = None
    if "split" not in frame.columns or set(frame["split"].astype(str)) != {str(record.get("split", "test"))}:
        reasons.append("sample_split:mismatch")
    summary: dict[str, float] = {}
    for metric in PRIMARY_METRICS:
        values = pd.to_numeric(frame[metric], errors="coerce").to_numpy(dtype=np.float64)
        if np.isinf(values).any():
            reasons.append(f"{metric}:infinite")
            continue
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            reasons.append(f"{metric}:no_finite_value")
            continue
        summary[metric] = float(np.mean(finite))

    declared = _filter_frame(pd.read_csv(summary_path), record.get("summary_filters"), label="summary")
    if len(declared) != 1:
        reasons.append("summary_row_count:not_one")
    else:
        mapping = dict(record.get("summary_columns", {metric: f"{metric}_mean" for metric in PRIMARY_METRICS}))
        for metric, recomputed in summary.items():
            column = mapping.get(metric)
            if column not in declared.columns:
                reasons.append(f"summary_column:{metric}")
                continue
            observed = float(declared.iloc[0][column])
            if not np.isfinite(observed) or not np.isclose(observed, recomputed, rtol=0.0, atol=1e-10):
                reasons.append(f"summary_value:{metric}")
    return audit_rows, summary if len(summary) == len(PRIMARY_METRICS) else None, row_hash, reasons


def _aggregate_method(method: Mapping[str, Any], seed_summaries: list[dict[str, float]]) -> dict[str, Any]:
    deterministic = bool(method.get("deterministic", False))
    row: dict[str, Any] = {
        "method_id": str(method["method_id"]),
        "display_name": str(method.get("display_name", method["method_id"])),
        "paper_role": str(method.get("paper_role", "")),
        "quality_scope": str(method.get("quality_scope", "independent_test")),
        "seed_count": 0 if deterministic else len(seed_summaries),
        "seed_semantics": "deterministic_no_seed" if deterministic else "three_seed",
    }
    for metric in PRIMARY_METRICS:
        values = np.asarray([item[metric] for item in seed_summaries], dtype=np.float64)
        row[f"{metric}_mean"] = float(np.mean(values))
        row[f"{metric}_sample_sd"] = float(np.std(values, ddof=1)) if values.size > 1 else np.nan
    return row


def _validate_representative_rule(rules: Mapping[str, Any], compatibility: pd.DataFrame) -> None:
    selected = [str(value) for value in rules.get("primary_table_method_ids", [])]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("representative rule 必须预注册非空且唯一的方法列表")
    known = set(compatibility["method_id"].astype(str))
    unknown = sorted(set(selected) - known)
    if unknown:
        raise ValueError(f"representative rule 含未知方法: {unknown}")
    declared = set(
        compatibility.loc[compatibility["predeclared_for_primary_table"], "method_id"].astype(str)
    )
    if declared != set(selected):
        raise RuntimeError("representative rule 与 method 的预声明 primary table 集合不一致")


def _resolve_source_path(repo_root: Path, value: Any) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    resolved = path.resolve() if path.is_absolute() else (repo_root / path).resolve()
    if resolved != repo_root and repo_root not in resolved.parents:
        raise ValueError(f"P0 source path 必须位于仓库内: {resolved}")
    return resolved


def _filter_frame(frame: pd.DataFrame, filters: Any, *, label: str) -> pd.DataFrame:
    if filters in (None, {}):
        return frame
    if not isinstance(filters, Mapping):
        raise TypeError(f"{label} filters 必须是 mapping")
    selected = frame
    for column, expected in filters.items():
        if str(column) not in selected.columns:
            raise ValueError(f"{label} 缺少 filter column={column!r}")
        selected = selected.loc[selected[str(column)].astype(str) == str(expected)]
    if selected.empty:
        raise ValueError(f"{label} filters 未匹配任何 row")
    return selected.reset_index(drop=True)


def _expanded_records(method: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = method.get("records")
    matrix = method.get("record_matrix")
    if records is not None and matrix is not None:
        raise ValueError("method 不得同时声明 records 与 record_matrix")
    if matrix is None:
        return list(records or [])
    if not isinstance(matrix, Mapping):
        raise TypeError("record_matrix 必须是 mapping")
    candidates = [str(value) for value in matrix.get("candidate_ids", [])]
    seeds = [int(value) for value in matrix.get("seeds", [])]
    if not candidates or not seeds:
        raise ValueError("record_matrix candidate_ids/seeds 不得为空")
    summary_columns = dict(
        matrix.get(
            "summary_columns",
            {metric: metric for metric in PRIMARY_METRICS},
        )
    )
    output = []
    for candidate_id in candidates:
        for seed in seeds:
            output.append(
                {
                    "seed": seed,
                    "split": str(matrix.get("split", "test")),
                    "sample_metrics": matrix["sample_metrics"],
                    "sample_filters": {"candidate_id": candidate_id, "seed": seed},
                    "summary": matrix["summary"],
                    "summary_filters": {"candidate_id": candidate_id, "seed": seed},
                    "summary_columns": summary_columns,
                }
            )
    return output


def _audited_candidate_ids(method: Mapping[str, Any]) -> list[str]:
    matrix = method.get("record_matrix")
    if isinstance(matrix, Mapping) and matrix.get("candidate_ids"):
        return [str(value) for value in matrix["candidate_ids"]]
    return [str(method["method_id"])]


def _assert_clean_git(repo_root: Path) -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"无法检查 Git 工作树: {result.stderr.strip()}")
    if result.stdout.strip():
        raise RuntimeError("P0 正式审计要求干净 Git 工作树")


__all__ = [
    "COMMON_CONTRACT",
    "P0_SCHEMA_VERSION",
    "PAPER_EVIDENCE_PROTOCOL_ID",
    "PRIMARY_METRICS",
    "audit_methods",
    "dataset_row_id_sha256",
    "load_p0_source_manifest",
    "run_p0_comparison_audit",
]
