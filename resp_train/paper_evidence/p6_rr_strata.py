from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from resp_train.crd.config import load_crd_config
from resp_train.data.cache import WholeNightCache
from resp_train.data.index import filter_index
from resp_train.data.research_v2 import read_research_v2_index
from resp_train.paper_evidence.p6_multi_attribute import (
    PRIMARY_METRICS,
    assign_rr_strata,
    build_target_attribute_frame,
    sha256_file,
)


PROTOCOL_ID = "paper-p6-rr-strata-v1-20260905"
REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = Path("configs/paper_evidence_v1/p6_rr_strata_v1.json")
CONTRACT_SHA256 = "d062e44868d5a721f9d111da4667ebc0c127f7c05767f2e80c104417a414e195"
TEST_TARGET_OUTPUT = Path("runs/paper_evidence_v1/p6_test_target_attributes")
SUMMARY_OUTPUT = Path("runs/paper_evidence_v1/p6_rr_strata_summary")


def load_contract(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    path = root / CONTRACT_PATH
    if sha256_file(path) != CONTRACT_SHA256:
        raise RuntimeError("P6 RR strata contract SHA-256 漂移")
    contract = json.loads(path.read_text(encoding="utf-8"))
    methods = contract.get("method_sources", {})
    if (
        contract.get("protocol_id") != PROTOCOL_ID
        or tuple(contract.get("rr_strata", {}).get("order", ())) != ("low", "medium", "high")
        or len(methods.get("method_allowlist", ())) != 10
        or len(set(methods.get("method_allowlist", ()))) != 10
        or tuple(methods.get("learned_seeds", ())) != (20260811, 20260812, 20260813)
        or Path(contract.get("outputs", {}).get("test_target_attributes", "")) != TEST_TARGET_OUTPUT
        or Path(contract.get("outputs", {}).get("rr_strata_summary", "")) != SUMMARY_OUTPUT
    ):
        raise RuntimeError("P6 RR strata contract identity/matrix 漂移")
    return contract


def validate_test_rows(rows: pd.DataFrame, dataset_contract: Mapping[str, Any]) -> None:
    required = {
        "dataset_row_id",
        "split",
        "input_set",
        "samp_id",
        "coupling_state_id",
        "target_source_npz",
        "target_signal_key",
        "window_start_sample",
        "window_end_sample",
    }
    missing = sorted(required - set(rows.columns))
    if (
        missing
        or len(rows) != int(dataset_contract["test_count"])
        or set(rows.get("split", pd.Series(dtype=str)).astype(str)) != {"test"}
    ):
        raise ValueError(f"P6 test rows schema/count/split 不合格: missing={missing}, rows={len(rows)}")
    row_ids = pd.to_numeric(rows["dataset_row_id"], errors="raise").to_numpy(dtype=np.int64)
    if np.unique(row_ids).size != row_ids.size or not np.all(np.diff(row_ids) > 0):
        raise ValueError("P6 test dataset_row_id 必须严格递增且无重复")
    if (
        hashlib.sha256(row_ids.astype("<i8", copy=False).tobytes(order="C")).hexdigest()
        != dataset_contract["test_row_ids_sha256"]
        or int(rows["samp_id"].nunique()) != int(dataset_contract["test_samp_id_count"])
        or not rows["target_signal_key"].eq(dataset_contract["target_key"]).all()
        or not (rows["window_end_sample"] - rows["window_start_sample"]).eq(
            int(dataset_contract["window_samples"])
        ).all()
    ):
        raise RuntimeError("P6 test row/samp/target identity 漂移")


def extract_test_target_attributes(rows: pd.DataFrame, cfg: Any) -> pd.DataFrame:
    cache = WholeNightCache(Path(str(cfg.data.dataset_root)) / str(cfg.data.index_csv))
    batches: list[pd.DataFrame] = []
    batch_rows: list[pd.Series] = []
    batch_targets: list[np.ndarray] = []
    for _, row in rows.iterrows():
        start, stop = int(row["window_start_sample"]), int(row["window_end_sample"])
        key = str(row["target_signal_key"])
        full_target = cache.get_arrays(str(row["target_source_npz"]), [key])[key]
        target = np.asarray(full_target[start:stop], dtype=np.float32).reshape(-1)
        if target.shape != (int(cfg.window.duration_samples),) or not np.isfinite(target).all():
            raise FloatingPointError(f"P6 test target row={int(row['dataset_row_id'])} shape/finite 不合格")
        batch_rows.append(row)
        batch_targets.append(target)
        if len(batch_targets) == 32:
            batches.append(
                build_target_attribute_frame(
                    pd.DataFrame(batch_rows), batch_targets, cfg, split="test"
                )
            )
            batch_rows, batch_targets = [], []
    if batch_targets:
        batches.append(
            build_target_attribute_frame(pd.DataFrame(batch_rows), batch_targets, cfg, split="test")
        )
    attributes = pd.concat(batches, ignore_index=True)
    metadata = rows[
        ["dataset_row_id", "input_set", "samp_id", "coupling_state_id"]
    ].copy()
    attributes = attributes.drop(columns=["samp_id"]).merge(
        metadata, on="dataset_row_id", how="inner", validate="one_to_one"
    )
    columns = [
        "dataset_row_id",
        "split",
        "input_set",
        "samp_id",
        "coupling_state_id",
        "target_sha256",
        "target_rr_bpm",
        "target_rr_eligible",
        "target_envelope_modulation",
    ]
    return attributes[columns].sort_values("dataset_row_id").reset_index(drop=True)


def rr_stratum_counts(attributes: pd.DataFrame, order: Sequence[str]) -> pd.DataFrame:
    rows = []
    for stratum_order, stratum in enumerate(order):
        group = attributes.loc[attributes["rr_stratum"].eq(stratum)]
        if group.empty:
            raise RuntimeError(f"P6 test RR stratum 为空: {stratum}")
        rows.append(
            {
                "rr_stratum": stratum,
                "stratum_order": stratum_order,
                "window_count": len(group),
                "samp_id_count": int(group["samp_id"].nunique()),
                "target_rr_min_bpm": float(group["target_rr_bpm"].min()),
                "target_rr_mean_bpm": float(group["target_rr_bpm"].mean()),
                "target_rr_max_bpm": float(group["target_rr_bpm"].max()),
            }
        )
    result = pd.DataFrame.from_records(rows)
    if int(result["window_count"].sum()) != len(attributes):
        raise RuntimeError("P6 test RR strata 未完整覆盖 target attributes")
    return result


def build_test_target_attributes(*, repo_root: str | Path, command: str) -> Path:
    root = Path(repo_root).resolve()
    output = root / TEST_TARGET_OUTPUT
    _reject_existing(output)
    contract = load_contract(root)
    commit = _require_clean_git(root)
    source_records: list[dict[str, Any]] = [
        _file_record(root / CONTRACT_PATH, "p6_rr_strata_contract", CONTRACT_SHA256)
    ]
    cfg_record = contract["target_config"]
    cfg_path = root / cfg_record["path"]
    source_records.append(_file_record(cfg_path, "target_metric_config", cfg_record["sha256"]))
    cfg = load_crd_config(cfg_path)
    dataset_contract = contract["dataset"]
    if (
        str(cfg.data.format) != "research_v2"
        or str(cfg.data.target_key) != dataset_contract["target_key"]
        or int(cfg.window.duration_samples) != int(dataset_contract["window_samples"])
        or float(cfg.window.target_fs) != float(dataset_contract["sample_rate_hz"])
        or float(cfg.loss.band_low_hz) != 0.05
        or float(cfg.loss.band_high_hz) != 0.7
    ):
        raise RuntimeError("P6 test target metric config 漂移")
    index_path = Path(str(cfg.data.dataset_root)) / str(cfg.data.index_csv)
    source_records.append(
        _file_record(index_path, "dataset_index", dataset_contract["dataset_index_sha256"])
    )
    cutpoints, rr_source_records = _load_train_cutpoints(root, contract)
    source_records.extend(rr_source_records)
    audited = read_research_v2_index(cfg.data.dataset_root, cfg.data.index_csv, cfg)
    rows = filter_index(
        audited,
        cfg,
        split="test",
        max_windows=None,
        sample_strategy=str(dataset_contract["sample_strategy"]),
        sample_seed=int(dataset_contract["sample_seed"]),
    )
    validate_test_rows(rows, dataset_contract)
    non_test_ids = audited.loc[
        audited["split"].astype(str).isin({"train", "val"}), "dataset_row_id"
    ].to_numpy(dtype=np.int64)
    if np.intersect1d(rows["dataset_row_id"].to_numpy(dtype=np.int64), non_test_ids).size:
        raise RuntimeError("P6 test rows 与 train/validation row identity 重叠")
    temporary = _temporary_output(output)
    try:
        attributes = extract_test_target_attributes(rows, cfg)
        if not attributes["target_rr_eligible"].isin([True, 1, "True"]).all():
            raise RuntimeError("P6 test target RR 存在 ineligible row")
        numeric = attributes[["target_rr_bpm", "target_envelope_modulation"]].apply(
            pd.to_numeric, errors="raise"
        ).to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise FloatingPointError("P6 test target RR/modulation 必须全部 finite")
        attributes["rr_stratum"] = assign_rr_strata(attributes["target_rr_bpm"], cutpoints)
        counts = rr_stratum_counts(attributes, contract["rr_strata"]["order"])
        attributes.to_csv(temporary / "test_target_attributes.csv", index=False)
        counts.to_csv(temporary / "test_rr_stratum_counts.csv", index=False)
        receipt = {
            "protocol_id": PROTOCOL_ID,
            "phase": "p6_test_target_attributes",
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "execution": _execution(commit, command),
            "contract_sha256": CONTRACT_SHA256,
            "inputs": source_records,
            "counts": {
                "test_rows": len(attributes),
                "test_samp_ids": int(attributes["samp_id"].nunique()),
                "rr_strata": counts[
                    ["rr_stratum", "window_count", "samp_id_count"]
                ].to_dict("records"),
            },
            "rr_cutpoints_bpm": list(cutpoints),
            "access": {
                "dataset_index_metadata_read": True,
                "test_target_read": True,
                "train_validation_target_read": False,
                "bcg_signal_read": False,
                "checkpoint_read": False,
                "existing_test_metrics_read": False,
                "model_inference_used": False,
                "training_used": False,
                "gpu_used": False,
            },
        }
        _verify_source_records(source_records)
        _finish_output(temporary, output, "target_attribute_receipt.json", receipt)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output / "target_attribute_receipt.json"


def validate_metric_frame(
    frame: pd.DataFrame,
    target_attributes: pd.DataFrame,
    dataset_contract: Mapping[str, Any],
    *,
    modulation_atol: float,
) -> pd.DataFrame:
    identity = ["dataset_row_id", "split", "input_set", "samp_id", "coupling_state_id"]
    required = {*identity, "method", "target_envelope_modulation", *PRIMARY_METRICS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"P6 source metrics schema 缺失: {missing}")
    ordered = frame.sort_values("dataset_row_id").reset_index(drop=True).copy()
    validate_test_rows(
        ordered.assign(
            target_source_npz="unused",
            target_signal_key=dataset_contract["target_key"],
            window_start_sample=0,
            window_end_sample=int(dataset_contract["window_samples"]),
        ),
        dataset_contract,
    )
    target = target_attributes.sort_values("dataset_row_id").reset_index(drop=True)
    if not ordered[identity].equals(target[identity]):
        raise RuntimeError("P6 metrics 与 test target attributes row/metadata 错位")
    observed_modulation = pd.to_numeric(
        ordered["target_envelope_modulation"], errors="raise"
    ).to_numpy(dtype=np.float64)
    target_modulation = pd.to_numeric(
        target["target_envelope_modulation"], errors="raise"
    ).to_numpy(dtype=np.float64)
    if not np.allclose(
        observed_modulation, target_modulation, rtol=0.0, atol=float(modulation_atol)
    ):
        raise RuntimeError("P6 metrics 与 target-only modulation 不一致")
    numeric = ordered[list(PRIMARY_METRICS)].apply(pd.to_numeric, errors="raise")
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise FloatingPointError("P6 source metrics 五主指标必须全部 finite")
    if (numeric[list(PRIMARY_METRICS[:-1])] < 0).any().any():
        raise ValueError("P6 source error metrics 不能为负")
    if not numeric[PRIMARY_METRICS[-1]].between(-1.0, 1.0, inclusive="both").all():
        raise ValueError("P6 source PCC 超出 [-1,1]")
    ordered["rr_stratum"] = target["rr_stratum"].to_numpy()
    return ordered


def summarize_metric_record(
    frame: pd.DataFrame,
    *,
    method_order: int,
    method_id: str,
    display_name: str,
    deterministic: bool,
    seed: int | None,
    source_method: str,
    source_path: str,
    strata_order: Sequence[str],
) -> list[dict[str, Any]]:
    rows = []
    for stratum_order, stratum in enumerate(strata_order):
        group = frame.loc[frame["rr_stratum"].eq(stratum)]
        if group.empty:
            raise RuntimeError(f"P6 metrics RR stratum 为空: {method_id}/{stratum}")
        row: dict[str, Any] = {
            "method_order": int(method_order),
            "method_id": method_id,
            "display_name": display_name,
            "deterministic": bool(deterministic),
            "seed": seed,
            "source_method": source_method,
            "source_path": source_path,
            "rr_stratum": stratum,
            "stratum_order": int(stratum_order),
            "window_count": len(group),
            "samp_id_count": int(group["samp_id"].nunique()),
        }
        for metric in PRIMARY_METRICS:
            row[metric] = float(group[metric].mean())
        rows.append(row)
    return rows


def aggregate_method_strata(
    record_metrics: pd.DataFrame,
    *,
    method_allowlist: Sequence[str],
    learned_seeds: Sequence[int],
    strata_order: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for method_order, method_id in enumerate(method_allowlist):
        for stratum_order, stratum in enumerate(strata_order):
            group = record_metrics.loc[
                record_metrics["method_id"].eq(method_id)
                & record_metrics["rr_stratum"].eq(stratum)
            ].sort_values("seed", na_position="first")
            if group.empty:
                raise RuntimeError(f"P6 method/stratum 缺失: {method_id}/{stratum}")
            deterministic = bool(group["deterministic"].iloc[0])
            if deterministic:
                if len(group) != 1 or group["seed"].notna().any():
                    raise RuntimeError("P6 deterministic record identity 不合格")
            elif len(group) != len(learned_seeds) or sorted(group["seed"].astype(int)) != list(
                learned_seeds
            ):
                raise RuntimeError("P6 learned method seed matrix 不完整")
            if (
                group["window_count"].nunique() != 1
                or group["samp_id_count"].nunique() != 1
                or group["display_name"].nunique() != 1
                or group["deterministic"].nunique() != 1
            ):
                raise RuntimeError("P6 method/stratum record provenance 不一致")
            row: dict[str, Any] = {
                "method_order": method_order,
                "method_id": method_id,
                "display_name": group["display_name"].iloc[0],
                "deterministic": deterministic,
                "record_count": len(group),
                "seed_count": 0 if deterministic else len(group),
                "seeds": "" if deterministic else ";".join(map(str, learned_seeds)),
                "rr_stratum": stratum,
                "stratum_order": stratum_order,
                "window_count": int(group["window_count"].iloc[0]),
                "samp_id_count": int(group["samp_id_count"].iloc[0]),
            }
            for metric in PRIMARY_METRICS:
                values = group[metric].to_numpy(dtype=np.float64)
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_sample_sd"] = (
                    float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
                )
            rows.append(row)
    result = pd.DataFrame.from_records(rows)
    if len(result) != len(method_allowlist) * len(strata_order):
        raise RuntimeError("P6 RR strata method summary matrix 不完整")
    return result


def summarize_test_rr_strata(*, repo_root: str | Path, command: str) -> Path:
    root = Path(repo_root).resolve()
    output = root / SUMMARY_OUTPUT
    _reject_existing(output)
    contract = load_contract(root)
    commit = _require_clean_git(root)
    target_attributes, target_records = _load_test_target_attributes(root, contract)
    methods, source_audit, p0_records = _load_p0_sources(root, contract)
    source_records = [
        _file_record(root / CONTRACT_PATH, "p6_rr_strata_contract", CONTRACT_SHA256),
        *target_records,
        *p0_records,
    ]
    method_contract = contract["method_sources"]
    strata_order = contract["rr_strata"]["order"]
    record_rows: list[dict[str, Any]] = []
    metric_records: list[dict[str, Any]] = []
    for method_order, method_id in enumerate(method_contract["method_allowlist"]):
        method = methods[method_id]
        records = method.get("records", [])
        deterministic = bool(method["deterministic"])
        expected_count = 1 if deterministic else len(method_contract["learned_seeds"])
        if (
            len(records) != expected_count
            or method.get("quality_scope") != "independent_test"
            or method.get("include_in_primary_table") is not True
            or method.get("conclusion_lock_confirmed") is not True
        ):
            raise RuntimeError(f"P6 method source matrix 不合格: {method_id}")
        if any(record.get("split") != "test" for record in records):
            raise RuntimeError(f"P6 method source split 漂移: {method_id}")
        observed_seeds = [int(record["seed"]) for record in records if "seed" in record]
        if deterministic and observed_seeds:
            raise RuntimeError(f"P6 deterministic method 不应有 seed: {method_id}")
        if not deterministic and observed_seeds != list(method_contract["learned_seeds"]):
            raise RuntimeError(f"P6 learned method seed 顺序/集合漂移: {method_id}")
        for record in records:
            metrics_path = (root / record["sample_metrics"]).resolve()
            audit_row = _metric_audit_row(source_audit, method_id, metrics_path)
            frame, file_record = _read_frozen_csv(
                metrics_path,
                role=f"{method_id}_seed_{record.get('seed', 'deterministic')}_sample_metrics",
                expected_hash=str(audit_row["sha256"]),
                expected_size=int(audit_row["size_bytes"]),
            )
            source_records.append(file_record)
            validated = validate_metric_frame(
                frame,
                target_attributes,
                contract["dataset"],
                modulation_atol=float(method_contract["target_envelope_modulation_atol"]),
            )
            source_methods = validated["method"].astype(str).unique().tolist()
            if len(source_methods) != 1:
                raise RuntimeError(f"P6 source method 列不唯一: {method_id}")
            record_rows.extend(
                summarize_metric_record(
                    validated,
                    method_order=method_order,
                    method_id=method_id,
                    display_name=str(method["display_name"]),
                    deterministic=deterministic,
                    seed=int(record["seed"]) if "seed" in record else None,
                    source_method=source_methods[0],
                    source_path=str(metrics_path.relative_to(root)),
                    strata_order=strata_order,
                )
            )
            metric_records.append(file_record)
    record_metrics = pd.DataFrame.from_records(record_rows)
    summary = aggregate_method_strata(
        record_metrics,
        method_allowlist=method_contract["method_allowlist"],
        learned_seeds=method_contract["learned_seeds"],
        strata_order=strata_order,
    )
    counts = rr_stratum_counts(target_attributes, strata_order)
    expected_counts = counts.set_index("rr_stratum")[["window_count", "samp_id_count"]]
    observed_counts = summary.set_index("rr_stratum")[["window_count", "samp_id_count"]]
    for stratum in strata_order:
        if not observed_counts.loc[stratum].eq(expected_counts.loc[stratum]).all().all():
            raise RuntimeError(f"P6 method summary RR stratum coverage 漂移: {stratum}")
    temporary = _temporary_output(output)
    try:
        counts.to_csv(temporary / "test_rr_stratum_counts.csv", index=False)
        record_metrics.to_csv(temporary / "rr_stratum_record_metrics.csv", index=False)
        summary.to_csv(temporary / "rr_stratum_method_summary.csv", index=False)
        pd.DataFrame(source_records).to_csv(temporary / "source_file_audit.csv", index=False)
        receipt = {
            "protocol_id": PROTOCOL_ID,
            "phase": "p6_test_rr_strata_summary",
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "execution": _execution(commit, command),
            "contract_sha256": CONTRACT_SHA256,
            "inputs": source_records,
            "counts": {
                "test_rows": len(target_attributes),
                "test_samp_ids": int(target_attributes["samp_id"].nunique()),
                "methods": len(method_contract["method_allowlist"]),
                "source_metric_records": len(metric_records),
                "record_stratum_rows": len(record_metrics),
                "method_stratum_rows": len(summary),
            },
            "aggregation": {
                "within_record": method_contract["record_aggregation"],
                "across_records": method_contract["method_aggregation"],
                "overlapping_windows_are_independent_clinical_samples": False,
                "model_selection_or_reranking_used": False,
            },
            "access": {
                "test_target_attribute_artifact_read": True,
                "existing_test_metrics_read": True,
                "dataset_signal_or_target_read": False,
                "checkpoint_read": False,
                "model_inference_used": False,
                "training_used": False,
                "gpu_used": False,
            },
        }
        _verify_source_records(source_records)
        _finish_output(temporary, output, "summary_receipt.json", receipt)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output / "summary_receipt.json"


def _load_train_cutpoints(
    root: Path, contract: Mapping[str, Any]
) -> tuple[tuple[float, float], list[dict[str, Any]]]:
    rr_contract = contract["rr_strata"]
    manifest_path = root / rr_contract["train_target_attribute_manifest"]
    manifest_record = _file_record(
        manifest_path,
        "train_validation_target_attribute_manifest",
        rr_contract["train_target_attribute_manifest_sha256"],
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _verify_artifact_manifest(manifest_path.parent, manifest, expected_protocol="paper-p6-multi-attribute-v1-20260905")
    cutpoint_path = manifest_path.parent / "rr_cutpoints.json"
    cutpoint_record = next(
        (record for record in manifest["files"] if record["filename"] == cutpoint_path.name), None
    )
    if cutpoint_record is None:
        raise RuntimeError("P6 train target artifact 缺少 RR cutpoints")
    _verify_file_record(cutpoint_path, cutpoint_record)
    payload = json.loads(cutpoint_path.read_text(encoding="utf-8"))
    cutpoints = (
        float(payload["low_max_q1_bpm"]),
        float(payload["medium_max_q2_bpm"]),
    )
    if cutpoints != tuple(map(float, rr_contract["cutpoints_bpm"])):
        raise RuntimeError("P6 train-frozen RR cutpoints 漂移")
    return cutpoints, [manifest_record, _file_record(cutpoint_path, "train_rr_cutpoints")]


def _load_test_target_attributes(
    root: Path, contract: Mapping[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    directory = root / contract["outputs"]["test_target_attributes"]
    manifest_path = directory / "artifact_manifest.json"
    receipt_path = directory / "target_attribute_receipt.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("protocol_id") != PROTOCOL_ID
        or receipt.get("phase") != "p6_test_target_attributes"
        or receipt.get("status") != "complete"
        or receipt.get("contract_sha256") != CONTRACT_SHA256
    ):
        raise RuntimeError("P6 test target attribute receipt identity 漂移")
    _verify_artifact_manifest(directory, manifest, expected_protocol=PROTOCOL_ID)
    attribute_meta = next(
        (
            record
            for record in manifest["files"]
            if record["filename"] == "test_target_attributes.csv"
        ),
        None,
    )
    if attribute_meta is None:
        raise RuntimeError("P6 test target artifact 缺少 attributes CSV")
    attributes, attribute_record = _read_frozen_csv(
        directory / "test_target_attributes.csv",
        role="test_target_attributes",
        expected_hash=attribute_meta["sha256"],
        expected_size=attribute_meta["size_bytes"],
    )
    required = {
        "dataset_row_id", "split", "input_set", "samp_id", "coupling_state_id",
        "target_sha256", "target_rr_bpm", "target_rr_eligible",
        "target_envelope_modulation", "rr_stratum",
    }
    if not required.issubset(attributes.columns):
        raise ValueError("P6 test target attributes schema 缺失")
    identity_rows = attributes.assign(
        target_source_npz="artifact",
        target_signal_key=contract["dataset"]["target_key"],
        window_start_sample=0,
        window_end_sample=int(contract["dataset"]["window_samples"]),
    )
    validate_test_rows(identity_rows, contract["dataset"])
    if (
        not attributes["target_rr_eligible"].isin([True, 1, "True"]).all()
        or set(attributes["rr_stratum"].astype(str)) != set(contract["rr_strata"]["order"])
    ):
        raise RuntimeError("P6 test target RR eligibility/strata 漂移")
    return attributes, [
        _file_record(manifest_path, "test_target_attribute_manifest"),
        _file_record(receipt_path, "test_target_attribute_receipt"),
        attribute_record,
    ]


def _load_p0_sources(
    root: Path, contract: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame, list[dict[str, Any]]]:
    method_contract = contract["method_sources"]
    p0_manifest_path = root / method_contract["p0_audit_manifest"]
    p0_manifest_record = _file_record(
        p0_manifest_path, "p0_audit_manifest", method_contract["p0_audit_manifest_sha256"]
    )
    p0_manifest = json.loads(p0_manifest_path.read_text(encoding="utf-8"))
    if p0_manifest.get("status", "complete") not in {"complete", None} and p0_manifest.get("decision") != "audit_complete":
        raise RuntimeError("P6 P0 audit manifest 未完成")
    source_manifest_path = root / method_contract["source_manifest"]
    source_manifest_record = _file_record(
        source_manifest_path, "p0_source_manifest", method_contract["source_manifest_sha256"]
    )
    sources = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if sources["representative_method_rule"]["primary_table_method_ids"] != method_contract["method_allowlist"]:
        raise RuntimeError("P6 method allowlist 与 P0 primary matrix 漂移")
    files = p0_manifest["files"]
    compatibility_path = p0_manifest_path.parent / "protocol_compatibility.csv"
    audit_path = p0_manifest_path.parent / "source_artifact_audit.csv"
    compatibility, compatibility_record = _read_frozen_csv(
        compatibility_path,
        role="p0_protocol_compatibility",
        expected_hash=files[compatibility_path.name]["sha256"],
        expected_size=files[compatibility_path.name]["size_bytes"],
    )
    source_audit, audit_record = _read_frozen_csv(
        audit_path,
        role="p0_source_artifact_audit",
        expected_hash=files[audit_path.name]["sha256"],
        expected_size=files[audit_path.name]["size_bytes"],
    )
    selected_compatibility = compatibility.loc[
        compatibility["method_id"].isin(method_contract["method_allowlist"])
    ].sort_values("method_id")
    if (
        len(selected_compatibility) != len(method_contract["method_allowlist"])
        or not selected_compatibility["compatibility"].eq("compatible").all()
        or not selected_compatibility["primary_table_eligible"].isin([True, 1, "True"]).all()
        or not selected_compatibility["dataset_row_id_sha256"].eq(
            contract["dataset"]["test_row_ids_sha256"]
        ).all()
    ):
        raise RuntimeError("P6 P0 method compatibility/row identity 不合格")
    methods = {method["method_id"]: method for method in sources["methods"]}
    if not set(method_contract["method_allowlist"]).issubset(methods):
        raise RuntimeError("P6 P0 source manifest 缺少方法")
    return methods, source_audit, [
        p0_manifest_record,
        source_manifest_record,
        compatibility_record,
        audit_record,
    ]


def _metric_audit_row(audit: pd.DataFrame, method_id: str, path: Path) -> pd.Series:
    selected = audit.loc[
        audit["method_id"].eq(method_id)
        & audit["artifact_kind"].eq("sample_metrics")
        & audit["path"].eq(str(path))
    ]
    unique = selected[["sha256", "size_bytes"]].drop_duplicates()
    if len(unique) != 1:
        raise RuntimeError(f"P6 P0 source metrics identity 缺失或冲突: {method_id}")
    return unique.iloc[0]


def _read_frozen_csv(
    path: Path, *, role: str, expected_hash: str, expected_size: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected_hash or len(data) != int(expected_size):
        raise RuntimeError(f"P6 frozen CSV identity 漂移: {role}")
    return pd.read_csv(io.BytesIO(data)), {
        "role": role,
        "path": str(path.resolve()),
        "sha256": digest,
        "size_bytes": len(data),
    }


def _reject_existing(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"P6 RR strata 输出目录禁止覆盖: {output}")


def _temporary_output(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{output.name}.incomplete_", dir=output.parent))


def _finish_output(
    temporary: Path, output: Path, receipt_name: str, receipt: Mapping[str, Any]
) -> None:
    _write_json(temporary / receipt_name, receipt)
    _write_json(
        temporary / "artifact_manifest.json",
        {"protocol_id": PROTOCOL_ID, "status": "complete", "files": _artifact_records(temporary)},
    )
    os.replace(temporary, output)


def _verify_artifact_manifest(
    directory: Path, manifest: Mapping[str, Any], *, expected_protocol: str
) -> None:
    if manifest.get("protocol_id") != expected_protocol or manifest.get("status") != "complete":
        raise RuntimeError("P6 source artifact manifest identity 漂移")
    files = manifest.get("files", [])
    if len({record["filename"] for record in files}) != len(files):
        raise RuntimeError("P6 source artifact manifest filename 重复")
    for record in files:
        _verify_file_record(directory / record["filename"], record)


def _verify_file_record(path: Path, record: Mapping[str, Any]) -> None:
    if (
        not path.is_file()
        or path.stat().st_size != int(record["size_bytes"])
        or sha256_file(path) != record["sha256"]
    ):
        raise RuntimeError(f"P6 source artifact 漂移: {path.name}")


def _file_record(path: Path, role: str, expected_hash: str | None = None) -> dict[str, Any]:
    digest = sha256_file(path)
    if expected_hash is not None and digest != expected_hash:
        raise RuntimeError(f"P6 frozen source SHA-256 漂移: {role}")
    return {
        "role": role,
        "path": str(path.resolve()),
        "sha256": digest,
        "size_bytes": path.stat().st_size,
    }


def _verify_source_records(records: Sequence[Mapping[str, Any]]) -> None:
    for record in records:
        path = Path(str(record["path"]))
        if path.stat().st_size != int(record["size_bytes"]) or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"P6 source 在执行期间发生变化: {record['role']}")


def _artifact_records(directory: Path) -> list[dict[str, Any]]:
    return [
        {"filename": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name != "artifact_manifest.json"
    ]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _execution(commit: str, command: str) -> dict[str, Any]:
    return {
        "command": command,
        "git_commit": commit,
        "git_dirty": False,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def _require_clean_git(root: Path) -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if commit.returncode != 0 or status.returncode != 0 or status.stdout.strip():
        raise RuntimeError("P6 RR strata 正式产物要求干净 Git commit")
    return commit.stdout.strip()


__all__ = [
    "CONTRACT_PATH",
    "CONTRACT_SHA256",
    "PROTOCOL_ID",
    "SUMMARY_OUTPUT",
    "TEST_TARGET_OUTPUT",
    "aggregate_method_strata",
    "build_test_target_attributes",
    "extract_test_target_attributes",
    "load_contract",
    "rr_stratum_counts",
    "summarize_metric_record",
    "summarize_test_rr_strata",
    "validate_metric_frame",
    "validate_test_rows",
]
