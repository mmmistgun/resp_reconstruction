from __future__ import annotations

import hashlib
import json
import os
import platform
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
from resp_train.data.research_v2 import ResearchV2WindowDataset, read_research_v2_index
from resp_train.metrics.task import (
    compute_log_rms_envelopes,
    compute_target_waveform_attributes,
    evaluate_task_predictions,
)


PROTOCOL_ID = "paper-p6-multi-attribute-v1-20260905"
REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = Path("configs/paper_evidence_v1/p6_multi_attribute_v1.json")
CONTRACT_SHA256 = "0b9c861d3934d1d5f247afb5737838a2d885ac0e1c454400dd7114d79e42af2c"
PRE_REPLAY_CONTRACT_SHA256 = "0c22bb56b8d5c54b604a4b7e2f2064f82c699df20d703744e6cea0cc243742ff"
TARGET_OUTPUT = Path("runs/paper_evidence_v1/p6_target_attributes")
SELECTION_OUTPUT = Path("runs/paper_evidence_v1/p6_waveform_selection")
EXPORT_OUTPUT = Path("runs/paper_evidence_v1/p6_waveform_export")
SEEDS = (20260811, 20260812, 20260813)
PRIMARY_METRICS = (
    "whole_rr_abs_error_bpm",
    "local_rr_mae_bpm",
    "envelope_trajectory_mae",
    "global_envelope_modulation_error",
    "lag_aware_signed_pcc",
)
CATEGORIES = (
    "typical",
    "rr_difficult",
    "effort_difficult",
    "rr_effort_inconsistent",
    "joint_failure",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    path = root / CONTRACT_PATH
    if sha256_file(path) != CONTRACT_SHA256:
        raise RuntimeError("P6 contract SHA-256 漂移")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if (
        contract.get("protocol_id") != PROTOCOL_ID
        or tuple(contract.get("waveform_selection", {}).get("selection_order", ())) != CATEGORIES
        or tuple(contract.get("waveform_export", {}).get("checkpoint_seeds", ())) != SEEDS
        or len(contract.get("method_allowlist", ())) != 10
        or len(set(contract.get("method_allowlist", ()))) != 10
    ):
        raise RuntimeError("P6 contract identity/matrix 漂移")
    return contract


def rr_cutpoints(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("train target RR 必须非空且全部 finite")
    low, high = np.quantile(array, [1.0 / 3.0, 2.0 / 3.0], method="linear")
    if not float(low) < float(high):
        raise ValueError("RR cutpoints 必须严格递增")
    return float(low), float(high)


def assign_rr_strata(values: Sequence[float], cutpoints: tuple[float, float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    low, high = map(float, cutpoints)
    if not low < high or not np.isfinite(array).all():
        raise ValueError("RR strata 输入或 cutpoints 不合格")
    return np.where(array <= low, "low", np.where(array <= high, "medium", "high"))


def build_target_attribute_frame(
    rows: pd.DataFrame,
    targets: Sequence[np.ndarray],
    cfg: Any,
    *,
    split: str,
) -> pd.DataFrame:
    if len(rows) != len(targets) or len(rows) == 0:
        raise ValueError("target rows/arrays 必须非空且一一对应")
    matrix = np.stack([np.asarray(value, dtype=np.float32).reshape(-1) for value in targets])
    rr, eligible, modulation = compute_target_waveform_attributes(matrix, cfg)
    records = []
    for index, (_, row) in enumerate(rows.reset_index(drop=True).iterrows()):
        target = np.ascontiguousarray(matrix[index], dtype=np.float32)
        records.append(
            {
                "dataset_row_id": int(row["dataset_row_id"]),
                "split": str(split),
                "samp_id": int(row["samp_id"]),
                "target_sha256": hashlib.sha256(target.tobytes(order="C")).hexdigest(),
                "target_rr_bpm": float(rr[index]),
                "target_rr_eligible": bool(eligible[index]),
                "target_envelope_modulation": float(modulation[index]),
            }
        )
    return pd.DataFrame.from_records(records)


def aggregate_w0_validation_metrics(
    frames: Mapping[int, pd.DataFrame], *, expected_count: int | None = None
) -> pd.DataFrame:
    if tuple(sorted(map(int, frames))) != SEEDS:
        raise ValueError("W0 validation metrics 必须完整覆盖三个冻结 seed")
    merged: pd.DataFrame | None = None
    metadata_reference: pd.DataFrame | None = None
    for seed in SEEDS:
        frame = frames[seed].copy()
        required = {
            "evaluation_split", "method", "dataset_row_id", "split", "samp_id",
            "target_envelope_modulation", *PRIMARY_METRICS,
        }
        if not required.issubset(frame.columns):
            raise ValueError(f"seed={seed} validation metrics schema 缺失")
        if (
            (expected_count is not None and len(frame) != int(expected_count))
            or not frame["evaluation_split"].eq("validation").all()
            or not frame["split"].eq("val").all()
            or not frame["method"].eq("crd_tf102_w").all()
        ):
            raise ValueError(f"seed={seed} validation metrics identity/count 漂移")
        frame["dataset_row_id"] = pd.to_numeric(frame["dataset_row_id"], errors="raise").astype(np.int64)
        if frame["dataset_row_id"].duplicated().any():
            raise ValueError(f"seed={seed} dataset_row_id 重复")
        frame = frame.sort_values("dataset_row_id").reset_index(drop=True)
        metadata = frame[["dataset_row_id", "samp_id", "target_envelope_modulation"]]
        numeric = frame[list(PRIMARY_METRICS)].apply(pd.to_numeric, errors="raise").to_numpy(np.float64)
        if not np.isfinite(numeric).all() or not np.isfinite(metadata["target_envelope_modulation"]).all():
            raise FloatingPointError("W0 validation 五主指标/target modulation 必须 finite")
        if metadata_reference is None:
            metadata_reference = metadata.copy()
            merged = metadata.copy()
        else:
            if not metadata[["dataset_row_id", "samp_id"]].equals(
                metadata_reference[["dataset_row_id", "samp_id"]]
            ) or not np.allclose(
                metadata["target_envelope_modulation"],
                metadata_reference["target_envelope_modulation"],
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError("W0 validation row/target identity 跨 seed 漂移")
        assert merged is not None
        for metric in PRIMARY_METRICS:
            merged[f"{metric}_seed_{seed}"] = frame[metric].to_numpy(np.float64)
    assert merged is not None
    for metric in PRIMARY_METRICS:
        merged[f"{metric}_three_seed_mean"] = merged[
            [f"{metric}_seed_{seed}" for seed in SEEDS]
        ].mean(axis=1)
    return merged


def build_waveform_selection(
    aggregate: pd.DataFrame,
    validation_attributes: pd.DataFrame,
    selection_contract: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    attrs = validation_attributes.copy()
    required_attrs = {
        "dataset_row_id", "split", "samp_id", "target_sha256", "target_rr_bpm",
        "target_rr_eligible", "target_envelope_modulation", "rr_stratum",
    }
    if not required_attrs.issubset(attrs.columns) or not attrs["split"].eq("val").all():
        raise ValueError("validation target attributes schema/split 不合格")
    if attrs["dataset_row_id"].duplicated().any() or aggregate["dataset_row_id"].duplicated().any():
        raise ValueError("selection 输入 dataset_row_id 重复")
    joined = aggregate.merge(attrs, on=["dataset_row_id", "samp_id"], how="inner", validate="one_to_one")
    if len(joined) != len(aggregate) or len(joined) != len(attrs):
        raise ValueError("W0 metrics 与 validation target attributes row 集合不一致")
    if not np.allclose(
        joined["target_envelope_modulation_x"],
        joined["target_envelope_modulation_y"],
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError("target envelope modulation 重算不一致")
    joined = joined.rename(columns={"target_envelope_modulation_y": "target_envelope_modulation"})
    joined = joined.drop(columns=["target_envelope_modulation_x"])
    eligibility = joined["target_rr_eligible"]
    if not eligibility.isin([True, 1, "True"]).all():
        raise ValueError("validation target RR 存在 ineligible row")
    # samp_id 只留在 target-only 属性中供 RR 分层 coverage 计数；波形候选和图件不携带该身份字段。
    joined = joined.drop(columns=["samp_id"])

    def normalized_rank(values: pd.Series) -> np.ndarray:
        ranks = values.rank(method="average", ascending=True).to_numpy(np.float64)
        return (ranks - 1.0) / float(len(values) - 1) if len(values) > 1 else np.zeros(1)

    whole = normalized_rank(joined["whole_rr_abs_error_bpm_three_seed_mean"])
    local = normalized_rank(joined["local_rr_mae_bpm_three_seed_mean"])
    trajectory = normalized_rank(joined["envelope_trajectory_mae_three_seed_mean"])
    global_error = normalized_rank(joined["global_envelope_modulation_error_three_seed_mean"])
    pcc_bad = normalized_rank(-joined["lag_aware_signed_pcc_three_seed_mean"])
    joined["rate_difficulty_score"] = (whole + local) / 2.0
    joined["effort_difficulty_score"] = (trajectory + global_error + pcc_bad) / 3.0

    canonical_rule = json.dumps(selection_contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    rule_hash = hashlib.sha256(canonical_rule.encode("utf-8")).hexdigest()
    selected_ids: set[int] = set()
    candidate_frames = []
    selected_rows = []
    for order, category in enumerate(CATEGORIES):
        rate = joined["rate_difficulty_score"].to_numpy(np.float64)
        effort = joined["effort_difficulty_score"].to_numpy(np.float64)
        if category == "typical":
            objective = np.hypot(rate - 0.5, effort - 0.5)
            ascending = True
        elif category == "rr_difficult":
            objective = np.hypot(rate - 1.0, effort)
            ascending = True
        elif category == "effort_difficult":
            objective = np.hypot(rate, effort - 1.0)
            ascending = True
        elif category == "rr_effort_inconsistent":
            objective = np.abs(rate - effort)
            ascending = False
        else:
            objective = np.hypot(rate - 1.0, effort - 1.0)
            ascending = True
        candidates = joined.copy()
        candidates.insert(0, "category", category)
        candidates.insert(1, "category_order", order)
        candidates["selection_objective"] = objective
        candidates["excluded_by_prior_selection"] = candidates["dataset_row_id"].isin(selected_ids)
        eligible = candidates.loc[~candidates["excluded_by_prior_selection"]].sort_values(
            ["selection_objective", "dataset_row_id"], ascending=[ascending, True], kind="mergesort"
        )
        if eligible.empty:
            raise RuntimeError(f"category={category} 没有可选 row")
        chosen = eligible.iloc[0].copy()
        chosen_id = int(chosen["dataset_row_id"])
        selected_ids.add(chosen_id)
        candidates["selected"] = candidates["dataset_row_id"].eq(chosen_id)
        candidate_frames.append(candidates)
        chosen["selection_rule_sha256"] = rule_hash
        chosen["selection_reason"] = str(selection_contract["rules"][category])
        selected_rows.append(chosen)
    selected = pd.DataFrame(selected_rows).sort_values("category_order").reset_index(drop=True)
    if selected["dataset_row_id"].nunique() != len(CATEGORIES):
        raise RuntimeError("P6 waveform selection 未得到五个不同 rows")
    candidates = pd.concat(candidate_frames, ignore_index=True)
    return candidates, selected, rule_hash


def build_numerical_replay_plan(
    selected: pd.DataFrame,
    validation_row_ids: Sequence[int],
    replay_contract: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[int]]:
    """校验冻结 row 的历史 batch slot，并给出只保存五项输出的抽取位置。"""

    row_ids = [int(value) for value in validation_row_ids]
    if len(row_ids) == 0 or len(set(row_ids)) != len(row_ids):
        raise ValueError("validation row ids 必须非空且唯一")
    selected_by_category = selected.set_index("category")
    if set(selected_by_category.index) != set(CATEGORIES):
        raise ValueError("numerical replay 必须完整覆盖五类 selected rows")
    source_batch = int(replay_contract["source_validation_batch_size"])
    if source_batch <= 0:
        raise ValueError("source validation batch size 必须为正")
    positions = {row_id: index for index, row_id in enumerate(row_ids)}
    observed_categories: set[str] = set()
    plans: list[dict[str, Any]] = []
    extraction_by_category: dict[str, int] = {}
    cumulative = 0
    for group_index, raw in enumerate(replay_contract["replay_batches"]):
        batch_size = int(raw["batch_size"])
        fill_category = str(raw["fill_category"])
        slots = {str(key): int(value) for key, value in raw["category_slots"].items()}
        if fill_category not in CATEGORIES or batch_size <= 0 or len(set(slots.values())) != len(slots):
            raise ValueError("numerical replay batch contract 不合格")
        entries = []
        for category, slot in slots.items():
            if category not in CATEGORIES or category in observed_categories or not 0 <= slot < batch_size:
                raise ValueError("numerical replay category/slot 重复或越界")
            row_id = int(selected_by_category.loc[category, "dataset_row_id"])
            if row_id not in positions:
                raise ValueError(f"selected row 不在完整 validation rows: {row_id}")
            source_position = positions[row_id]
            source_group_start = (source_position // source_batch) * source_batch
            expected_batch_size = min(source_batch, len(row_ids) - source_group_start)
            expected_slot = source_position - source_group_start
            if batch_size != expected_batch_size or slot != expected_slot:
                raise RuntimeError(
                    f"numerical replay 历史 batch shape/slot 漂移: {category} "
                    f"expected=({expected_batch_size},{expected_slot}) observed=({batch_size},{slot})"
                )
            observed_categories.add(category)
            extraction_by_category[category] = cumulative + slot
            entries.append(
                {
                    "category": category,
                    "dataset_row_id": row_id,
                    "source_position": source_position,
                    "slot": slot,
                }
            )
        plans.append(
            {
                "group_index": group_index,
                "batch_size": batch_size,
                "fill_category": fill_category,
                "entries": entries,
            }
        )
        cumulative += batch_size
    if observed_categories != set(CATEGORIES):
        raise ValueError("numerical replay contract 未完整覆盖五类")
    if cumulative != int(replay_contract["processed_batch_elements_per_checkpoint"]):
        raise ValueError("numerical replay processed elements contract 漂移")
    extraction = [extraction_by_category[category] for category in CATEGORIES]
    return plans, extraction


def build_p6_target_attributes(*, repo_root: str | Path, command: str) -> Path:
    root = Path(repo_root).resolve()
    output = root / TARGET_OUTPUT
    _reject_existing(output)
    commit = _require_clean_git(root)
    contract = load_contract(root)
    source_records = _audit_static_sources(root, contract, include_metrics=False, include_checkpoints=False)
    cfg = load_crd_config(root / contract["w0_sources"][0]["run_dir"] / "config.yaml")
    index_path = Path(str(cfg.data.dataset_root)) / str(cfg.data.index_csv)
    if sha256_file(index_path) != contract["dataset"]["dataset_index_sha256"]:
        raise RuntimeError("P6 dataset index SHA-256 漂移")
    source_records.append(
        _file_record(index_path, "dataset_index", contract["dataset"]["dataset_index_sha256"])
    )
    audited = read_research_v2_index(cfg.data.dataset_root, cfg.data.index_csv, cfg)
    temporary = _temporary_output(output)
    try:
        frames: dict[str, pd.DataFrame] = {}
        for split, cfg_key, label in (("train", "train", "train"), ("val", "validation", "validation")):
            rows = filter_index(
                audited,
                cfg,
                split=split,
                max_windows=None,
                sample_strategy="stratified_random",
                sample_seed=int(cfg.data.train_sample_seed if split == "train" else cfg.data.val_sample_seed),
            )
            _validate_dataset_rows(rows, contract["dataset"], cfg_key)
            frames[label] = _extract_target_attributes(rows, cfg, split=split)
        if not frames["train"]["target_rr_eligible"].all() or not frames["validation"]["target_rr_eligible"].all():
            raise RuntimeError("完整 admitted train/validation 中存在 target RR ineligible row")
        cutpoints = rr_cutpoints(frames["train"]["target_rr_bpm"])
        for frame in frames.values():
            frame["rr_stratum"] = assign_rr_strata(frame["target_rr_bpm"], cutpoints)
        frames["train"].to_csv(temporary / "train_target_attributes.csv", index=False)
        frames["validation"].to_csv(temporary / "validation_target_attributes.csv", index=False)
        _write_json(
            temporary / "rr_cutpoints.json",
            {
                "protocol_id": PROTOCOL_ID,
                "source_split": "train",
                "source_rows": len(frames["train"]),
                "quantiles": [1.0 / 3.0, 2.0 / 3.0],
                "method": "linear",
                "low_max_q1_bpm": cutpoints[0],
                "medium_max_q2_bpm": cutpoints[1],
                "boundary_rule": contract["rr_strata"]["boundary_rule"],
            },
        )
        receipt = {
            "protocol_id": PROTOCOL_ID,
            "phase": "p6_train_validation_target_attributes",
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "execution": _execution(commit, command),
            "contract_sha256": CONTRACT_SHA256,
            "inputs": source_records,
            "dataset_index_sha256": contract["dataset"]["dataset_index_sha256"],
            "counts": {
                "train_rows": len(frames["train"]),
                "validation_rows": len(frames["validation"]),
                "train_samp_ids": int(frames["train"]["samp_id"].nunique()),
                "validation_samp_ids": int(frames["validation"]["samp_id"].nunique()),
            },
            "rr_cutpoints_bpm": list(cutpoints),
            "access": {
                "train_target_read": True,
                "validation_target_read": True,
                "test_index_metadata_read": True,
                "test_target_read": False,
                "bcg_signal_read": False,
                "checkpoint_read": False,
                "model_inference_used": False,
                "training_used": False,
                "gpu_used": False,
            },
        }
        _verify_source_records(source_records)
        _finish_output(temporary, output, receipt)
    except BaseException as error:
        _record_failure(temporary, error)
        raise
    return output / "target_attribute_receipt.json"


def select_p6_validation_waveforms(*, repo_root: str | Path, command: str) -> Path:
    root = Path(repo_root).resolve()
    output = root / SELECTION_OUTPUT
    _reject_existing(output)
    commit = _require_clean_git(root)
    contract = load_contract(root)
    source_records = _audit_static_sources(root, contract, include_metrics=True, include_checkpoints=False)
    attrs, target_manifest_record = _load_target_attributes(root)
    frames = {
        int(source["seed"]): pd.read_csv(root / source["run_dir"] / "metrics.csv")
        for source in contract["w0_sources"]
    }
    aggregate = aggregate_w0_validation_metrics(
        frames, expected_count=int(contract["dataset"]["validation_count"])
    )
    candidates, selected, rule_hash = build_waveform_selection(
        aggregate, attrs, contract["waveform_selection"]
    )
    temporary = _temporary_output(output)
    try:
        candidates.to_csv(temporary / "waveform_selection_candidates.csv", index=False)
        selected.to_csv(temporary / "waveform_selected_rows.csv", index=False)
        receipt = {
            "protocol_id": PROTOCOL_ID,
            "phase": "p6_validation_waveform_selection",
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "execution": _execution(commit, command),
            "contract_sha256": CONTRACT_SHA256,
            "selection_rule_sha256": rule_hash,
            "inputs": [*source_records, target_manifest_record],
            "counts": {"candidate_rows": len(candidates), "selected_rows": len(selected)},
            "selected": selected[["category", "dataset_row_id", "target_rr_bpm", "rr_stratum"]].to_dict("records"),
            "access": {
                "frozen_validation_metrics_read": True,
                "validation_target_attributes_read": True,
                "waveform_signal_read": False,
                "checkpoint_read": False,
                "test_read": False,
                "model_inference_used": False,
                "training_used": False,
                "gpu_used": False,
            },
        }
        _verify_source_records([*source_records, target_manifest_record])
        _finish_output(temporary, output, receipt)
    except BaseException as error:
        _record_failure(temporary, error)
        raise
    return output / "waveform_selection_receipt.json"


def export_p6_validation_waveforms(
    *, repo_root: str | Path, command: str, device: str
) -> Path:
    import torch
    from torch.utils.data._utils.collate import default_collate

    from resp_train.crd.experiment import _validate_checkpoint_config
    from resp_train.crd.model import build_crd_model
    from resp_train.engine import collect_predictions
    from resp_train.protocols.respiration import as_batch_waveform_numpy

    root = Path(repo_root).resolve()
    output = root / EXPORT_OUTPUT
    _reject_existing(output)
    commit = _require_clean_git(root)
    if not str(device).startswith("cuda:") or not torch.cuda.is_available():
        raise RuntimeError("P6 validation waveform export 必须使用显式可用的 cuda:<index>")
    contract = load_contract(root)
    source_records = _audit_static_sources(root, contract, include_metrics=True, include_checkpoints=True)
    selected, selection_manifest_record = _load_selected_rows(root)
    selected_ids = selected.sort_values("category_order")["dataset_row_id"].astype(int).tolist()
    first = contract["w0_sources"][0]
    base_cfg = load_crd_config(
        root / first["run_dir"] / "config.yaml",
        overrides=[f"training.device={device}", "training.show_progress=false"],
    )
    audited = read_research_v2_index(base_cfg.data.dataset_root, base_cfg.data.index_csv, base_cfg)
    admitted_val_rows = filter_index(
        audited,
        base_cfg,
        split="val",
        max_windows=None,
        sample_strategy="stratified_random",
        sample_seed=int(base_cfg.data.val_sample_seed),
    )
    replay_contract = contract["waveform_export"]["numerical_replay"]
    replay_plan, extraction_positions = build_numerical_replay_plan(
        selected,
        admitted_val_rows["dataset_row_id"].astype(int).tolist(),
        replay_contract,
    )
    val_rows = admitted_val_rows.set_index("dataset_row_id").loc[selected_ids].reset_index()
    dataset = ResearchV2WindowDataset(
        Path(str(base_cfg.data.dataset_root)) / str(base_cfg.data.index_csv),
        val_rows,
        base_cfg,
        preload_windows=True,
        preload_show_progress=False,
    )
    category_to_index = {
        str(category): index
        for index, category in enumerate(selected.sort_values("category_order")["category"])
    }
    items = [dataset[index] for index in range(len(dataset))]
    replay_loader = []
    for group in replay_plan:
        fill_index = category_to_index[str(group["fill_category"])]
        replay_items = [items[fill_index] for _ in range(int(group["batch_size"]))]
        for entry in group["entries"]:
            replay_items[int(entry["slot"])] = items[category_to_index[str(entry["category"])] ]
        replay_loader.append(default_collate(replay_items))
    bcg = np.stack([dataset[index]["x"].numpy().reshape(-1) for index in range(len(dataset))])
    target = np.stack([dataset[index]["target"].numpy().reshape(-1) for index in range(len(dataset))])
    predictions = []
    metric_frames = []
    temporary = _temporary_output(output)
    try:
        for source in contract["w0_sources"]:
            seed = int(source["seed"])
            cfg = load_crd_config(
                root / source["run_dir"] / "config.yaml",
                overrides=[f"training.device={device}", "training.show_progress=false"],
            )
            checkpoint = torch.load(
                root / source["run_dir"] / "checkpoint_best_local_rr.pt",
                map_location="cpu",
                weights_only=False,
            )
            if int(checkpoint.get("epoch", -1)) != int(source["selected_epoch"]):
                raise RuntimeError(f"seed={seed} checkpoint selected epoch 漂移")
            _validate_checkpoint_config(checkpoint.get("config"), cfg)
            model = build_crd_model(cfg)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            del checkpoint
            predicted = collect_predictions(
                model,
                replay_loader,
                device=device,
                max_windows=int(replay_contract["processed_batch_elements_per_checkpoint"]),
                use_amp=True,
            )
            predicted = _slice_prediction_rows(predicted, extraction_positions)
            observed_ids = np.asarray(predicted["dataset_row_id"], dtype=np.int64).tolist()
            if observed_ids != selected_ids:
                raise RuntimeError("P6 waveform export row order 漂移")
            wave = as_batch_waveform_numpy(predicted["r_tho_hat"]).astype(np.float32, copy=False)
            predictions.append(wave)
            metrics = evaluate_task_predictions(predicted, cfg, method="crd_tf102_w")
            metrics.insert(0, "seed", seed)
            _anchor_export_metrics(
                metrics,
                pd.read_csv(root / source["run_dir"] / "metrics.csv"),
                atol=float(contract["waveform_export"]["primary_metric_anchor_atol"]),
            )
            category_by_id = selected.set_index("dataset_row_id")["category"]
            metrics.insert(
                1,
                "category",
                metrics["dataset_row_id"].map(category_by_id),
            )
            selected_by_id = selected.set_index("dataset_row_id")
            metrics.insert(
                3,
                "target_rr_bpm",
                metrics["dataset_row_id"].map(selected_by_id["target_rr_bpm"]),
            )
            metrics.insert(
                4,
                "rr_stratum",
                metrics["dataset_row_id"].map(selected_by_id["rr_stratum"]),
            )
            metric_frames.append(
                metrics[
                    ["seed", "category", "dataset_row_id", "target_rr_bpm", "rr_stratum", *PRIMARY_METRICS]
                ]
            )
            del model
            torch.cuda.empty_cache()
        prediction_matrix = np.stack(predictions)
        center_start, center_stop = map(int, contract["waveform_export"]["center_interval_samples"])
        envelope_bcg = compute_log_rms_envelopes(bcg, base_cfg)
        envelope_target = compute_log_rms_envelopes(target, base_cfg)
        envelope_prediction = np.stack(
            [compute_log_rms_envelopes(prediction_matrix[index], base_cfg) for index in range(len(SEEDS))]
        )
        np.savez_compressed(
            temporary / "validation_waveforms.npz",
            dataset_row_ids=np.asarray(selected_ids, dtype=np.int64),
            categories=selected.sort_values("category_order")["category"].to_numpy(str),
            bcg=bcg.astype(np.float32),
            whole_target=target.astype(np.float32),
            center_target=target[:, center_start:center_stop].astype(np.float32),
            checkpoint_seeds=np.asarray(SEEDS, dtype=np.int64),
            predictions=prediction_matrix.astype(np.float32),
            log_rms_bcg=envelope_bcg,
            log_rms_target=envelope_target,
            log_rms_predictions=envelope_prediction,
        )
        export_metrics = pd.concat(metric_frames, ignore_index=True)
        export_metrics.to_csv(temporary / "validation_waveform_metrics.csv", index=False)
        render_waveform_panel(
            selected.sort_values("category_order"),
            bcg,
            target,
            prediction_matrix,
            envelope_bcg,
            envelope_target,
            envelope_prediction,
            temporary / "validation_waveform_panel.png",
            fs=float(base_cfg.window.target_fs),
            center_samples=(center_start, center_stop),
        )
        receipt = {
            "protocol_id": PROTOCOL_ID,
            "phase": "p6_validation_waveform_export",
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "execution": _execution(commit, command),
            "contract_sha256": CONTRACT_SHA256,
            "inputs": [*source_records, selection_manifest_record],
            "counts": {
                "validation_rows": 5,
                "checkpoint_count": 3,
                "exported_row_checkpoint_pairs": 15,
                "replay_batches_per_checkpoint": len(replay_plan),
                "processed_batch_elements_per_checkpoint": int(
                    replay_contract["processed_batch_elements_per_checkpoint"]
                ),
                "processed_batch_elements_total": int(replay_contract["processed_batch_elements_total"]),
            },
            "numerical_replay": {
                "reason": "preserve historical validation batch shape and selected-row slot under BF16/Mamba",
                "padding_rows_reuse_selected_rows_only": True,
                "padding_outputs_saved": False,
                "plan": replay_plan,
            },
            "center_interval_samples": [center_start, center_stop],
            "input_carrier": str(base_cfg.data.bcg_input_key),
            "target_carrier": str(base_cfg.data.target_key),
            "access": {
                "validation_bcg_read": True,
                "validation_target_read": True,
                "validation_w_cache_read": True,
                "checkpoint_read": True,
                "test_read": False,
                "model_inference_used": True,
                "training_used": False,
                "gpu_used": True,
            },
        }
        for row_index, row in selected.sort_values("category_order").reset_index(drop=True).iterrows():
            observed_hash = hashlib.sha256(
                np.ascontiguousarray(target[row_index], dtype=np.float32).tobytes(order="C")
            ).hexdigest()
            if observed_hash != str(row["target_sha256"]):
                raise RuntimeError(f"P6 export target hash 漂移: row={int(row['dataset_row_id'])}")
        _verify_source_records([*source_records, selection_manifest_record])
        _finish_output(temporary, output, receipt)
    except BaseException as error:
        _record_failure(temporary, error)
        raise
    return output / "waveform_export_receipt.json"


def render_waveform_panel(
    selected: pd.DataFrame,
    bcg: np.ndarray,
    target: np.ndarray,
    predictions: np.ndarray,
    envelope_bcg: np.ndarray,
    envelope_target: np.ndarray,
    envelope_predictions: np.ndarray,
    path: str | Path,
    *,
    fs: float,
    center_samples: tuple[int, int],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if bcg.shape != target.shape or predictions.shape != (3, *target.shape):
        raise ValueError("P6 waveform panel array shape 不合格")
    figure, axes = plt.subplots(len(selected), 2, figsize=(15, 12), sharex="col")
    time = np.arange(target.shape[1], dtype=np.float64) / fs
    env_time = (np.arange(envelope_target.shape[1]) * 5.0) + 5.0
    colors = ("#0072B2", "#D55E00", "#009E73")
    center_start, center_stop = center_samples[0] / fs, center_samples[1] / fs
    for row_index, row in selected.reset_index(drop=True).iterrows():
        left, right = axes[row_index]
        left.axvspan(center_start, center_stop, color="#F0E442", alpha=0.12)
        left.plot(time[::10], bcg[row_index, ::10], color="0.75", linewidth=0.5, label="BCG")
        left.plot(time[::10], target[row_index, ::10], color="black", linewidth=1.0, label="Target")
        for seed_index, seed in enumerate(SEEDS):
            left.plot(
                time[::10], predictions[seed_index, row_index, ::10],
                color=colors[seed_index], linewidth=0.7, alpha=0.85, label=f"W0 {seed}",
            )
        left.set_ylabel(str(row["category"]))
        left.set_title(
            f"row {int(row['dataset_row_id'])}; target RR={float(row['target_rr_bpm']):.2f} bpm; {row['rr_stratum']}"
        )
        right.plot(env_time, envelope_bcg[row_index], color="0.75", linewidth=0.8, label="BCG")
        right.plot(env_time, envelope_target[row_index], color="black", linewidth=1.2, label="Target")
        for seed_index, seed in enumerate(SEEDS):
            right.plot(
                env_time, envelope_predictions[seed_index, row_index],
                color=colors[seed_index], linewidth=1.0, label=f"W0 {seed}",
            )
        right.set_title("Canonical log-RMS envelope")
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Envelope center time (s)")
    axes[0, 0].legend(ncol=5, fontsize=7, loc="upper right")
    figure.suptitle("Predeclared validation waveform examples")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight", metadata={"Software": "matplotlib"})
    plt.close(figure)


def _extract_target_attributes(rows: pd.DataFrame, cfg: Any, *, split: str) -> pd.DataFrame:
    cache = WholeNightCache(Path(str(cfg.data.dataset_root)) / str(cfg.data.index_csv))
    output = []
    batch_rows = []
    batch_targets = []
    for _, row in rows.iterrows():
        start, stop = int(row["window_start_sample"]), int(row["window_end_sample"])
        key = str(row["target_signal_key"])
        values = cache.get_arrays(str(row["target_source_npz"]), [key])[key]
        target = np.asarray(values[start:stop], dtype=np.float32).reshape(-1)
        if target.size != int(cfg.window.duration_samples) or not np.isfinite(target).all():
            raise ValueError(f"target row={int(row['dataset_row_id'])} shape/finite 不合格")
        batch_rows.append(row)
        batch_targets.append(target)
        if len(batch_targets) == 32:
            output.append(build_target_attribute_frame(pd.DataFrame(batch_rows), batch_targets, cfg, split=split))
            batch_rows, batch_targets = [], []
    if batch_targets:
        output.append(build_target_attribute_frame(pd.DataFrame(batch_rows), batch_targets, cfg, split=split))
    frame = pd.concat(output, ignore_index=True)
    return frame.sort_values("dataset_row_id").reset_index(drop=True)


def _validate_dataset_rows(rows: pd.DataFrame, dataset: Mapping[str, Any], key: str) -> None:
    count = int(dataset[f"{key}_count"])
    ids = rows["dataset_row_id"].astype("<i8").to_numpy()
    if (
        len(rows) != count
        or rows["dataset_row_id"].duplicated().any()
        or rows["samp_id"].nunique() != int(dataset[f"{key}_samp_id_count"])
        or hashlib.sha256(ids.tobytes(order="C")).hexdigest() != dataset[f"{key}_row_ids_sha256"]
    ):
        raise RuntimeError(f"P6 {key} admitted rows identity 漂移")


def _audit_static_sources(
    root: Path,
    contract: Mapping[str, Any],
    *,
    include_metrics: bool,
    include_checkpoints: bool,
) -> list[dict[str, Any]]:
    records = [_file_record(root / CONTRACT_PATH, "p6_contract", CONTRACT_SHA256)]
    for role, path_key, hash_key in (
        ("p0_source_manifest", "p0_source_manifest", "p0_source_manifest_sha256"),
        ("crd_tf_v1_p5_manifest", "crd_tf_v1_p5_manifest", "crd_tf_v1_p5_manifest_sha256"),
        ("train_validation_w_cache_manifest", "train_validation_w_cache_manifest", "train_validation_w_cache_manifest_sha256"),
    ):
        records.append(_file_record(root / contract["frozen_sources"][path_key], role, contract["frozen_sources"][hash_key]))
    p0 = json.loads((root / contract["frozen_sources"]["p0_source_manifest"]).read_text(encoding="utf-8"))
    if p0.get("representative_method_rule", {}).get("primary_table_method_ids") != contract["method_allowlist"]:
        raise RuntimeError("P6 method allowlist 与冻结 P0 primary methods 不一致")
    cache_manifest = json.loads(
        (root / contract["frozen_sources"]["train_validation_w_cache_manifest"]).read_text(encoding="utf-8")
    )
    if (
        cache_manifest.get("dataset_index_sha256") != contract["dataset"]["dataset_index_sha256"]
        or cache_manifest.get("splits", {}).get("train", {}).get("count") != contract["dataset"]["train_count"]
        or cache_manifest.get("splits", {}).get("val", {}).get("count") != contract["dataset"]["validation_count"]
        or cache_manifest.get("research_test_used") is not False
        or cache_manifest.get("target_read") is not False
    ):
        raise RuntimeError("P6 train/validation cache provenance 漂移")
    p5_manifest_path = root / contract["frozen_sources"]["crd_tf_v1_p5_manifest"]
    p5_manifest = json.loads(p5_manifest_path.read_text(encoding="utf-8"))
    audit_meta = p5_manifest.get("files", {}).get("formal_run_audit.csv", {})
    audit_path = p5_manifest_path.parent / "formal_run_audit.csv"
    records.append(_file_record(audit_path, "crd_tf_v1_formal_run_audit", audit_meta.get("sha256")))
    formal = pd.read_csv(audit_path)
    w0 = formal.loc[formal["variant"].eq("crd_tf102_w")].sort_values("seed")
    expected_identity = [
        (
            int(source["seed"]),
            int(source["selected_epoch"]),
            str(source["checkpoint_sha256"]),
            str((root / source["run_dir"]).resolve()),
        )
        for source in contract["w0_sources"]
    ]
    observed_identity = list(
        zip(
            w0["seed"].astype(int),
            w0["selected_epoch"].astype(int),
            w0["checkpoint_sha256"].astype(str),
            w0["run_dir"].astype(str),
        )
    )
    if observed_identity != expected_identity:
        raise RuntimeError("P6 W0 checkpoint allowlist 与冻结 P5 audit 不一致")
    for source in contract["w0_sources"]:
        run = root / source["run_dir"]
        records.append(_file_record(run / "config.yaml", f"w0_seed_{source['seed']}_config", source["config_sha256"]))
        _validate_w0_config(load_crd_config(run / "config.yaml"), contract)
        if include_metrics:
            records.append(_file_record(run / "metrics.csv", f"w0_seed_{source['seed']}_validation_metrics", source["validation_metrics_sha256"]))
        if include_checkpoints:
            records.append(_file_record(run / "checkpoint_best_local_rr.pt", f"w0_seed_{source['seed']}_checkpoint", source["checkpoint_sha256"]))
    return records


def _validate_w0_config(cfg: Any, contract: Mapping[str, Any]) -> None:
    dataset = contract["dataset"]
    if (
        str(cfg.protocol.name) != "crd-tf-v1-research-informed-20260812"
        or str(cfg.model.variant) != "crd_tf102_w"
        or list(cfg.model.tf_representations) != ["w"]
        or str(cfg.data.format) != "research_v2"
        or str(cfg.data.train_split) != "train"
        or str(cfg.data.val_split) != "val"
        or str(cfg.data.bcg_input_key) != dataset["bcg_input_key"]
        or str(cfg.data.target_key) != dataset["target_key"]
        or int(cfg.window.duration_samples) != int(dataset["window_samples"])
        or float(cfg.window.target_fs) != float(dataset["sample_rate_hz"])
        or float(cfg.loss.band_low_hz) != 0.05
        or float(cfg.loss.band_high_hz) != 0.7
    ):
        raise RuntimeError("P6 W0 scientific config 漂移")


def _load_target_attributes(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    directory = root / TARGET_OUTPUT
    manifest_path = directory / "artifact_manifest.json"
    receipt_path = directory / "target_attribute_receipt.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("protocol_id") != PROTOCOL_ID
        or receipt.get("contract_sha256") not in {PRE_REPLAY_CONTRACT_SHA256, CONTRACT_SHA256}
    ):
        raise RuntimeError("P6 target attribute receipt identity 漂移")
    _verify_artifact_manifest(directory, manifest)
    attrs = pd.read_csv(directory / "validation_target_attributes.csv")
    return attrs, _file_record(manifest_path, "target_attribute_manifest")


def _load_selected_rows(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    directory = root / SELECTION_OUTPUT
    manifest_path = directory / "artifact_manifest.json"
    receipt = json.loads((directory / "waveform_selection_receipt.json").read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        receipt.get("protocol_id") != PROTOCOL_ID
        or receipt.get("contract_sha256") not in {PRE_REPLAY_CONTRACT_SHA256, CONTRACT_SHA256}
    ):
        raise RuntimeError("P6 waveform selection receipt identity 漂移")
    _verify_artifact_manifest(directory, manifest)
    selected = pd.read_csv(directory / "waveform_selected_rows.csv")
    if tuple(selected.sort_values("category_order")["category"]) != CATEGORIES:
        raise RuntimeError("P6 selected category matrix 漂移")
    return selected, _file_record(manifest_path, "waveform_selection_manifest")


def _anchor_export_metrics(observed: pd.DataFrame, frozen: pd.DataFrame, *, atol: float) -> None:
    source = frozen.set_index("dataset_row_id")
    for _, row in observed.iterrows():
        row_id = int(row["dataset_row_id"])
        if row_id not in source.index:
            raise RuntimeError(f"冻结 validation metrics 缺少 row={row_id}")
        for metric in PRIMARY_METRICS:
            observed_value = float(row[metric])
            expected_value = float(source.loc[row_id, metric])
            if not np.isclose(observed_value, expected_value, rtol=0.0, atol=atol):
                raise RuntimeError(
                    f"P6 export FULL anchor 不一致: row={row_id}/{metric}; "
                    f"observed={observed_value:.17g}; expected={expected_value:.17g}; "
                    f"abs_delta={abs(observed_value - expected_value):.17g}; atol={atol:.17g}"
                )


def _slice_prediction_rows(
    predictions: Mapping[str, np.ndarray], positions: Sequence[int]
) -> dict[str, np.ndarray]:
    indices = np.asarray(positions, dtype=np.int64)
    if indices.shape != (len(CATEGORIES),) or len(np.unique(indices)) != len(indices):
        raise ValueError("P6 numerical replay extraction positions 不合格")
    output = {}
    total = len(np.asarray(predictions["dataset_row_id"]))
    if indices.min() < 0 or indices.max() >= total:
        raise ValueError("P6 numerical replay extraction position 越界")
    for key, value in predictions.items():
        array = np.asarray(value)
        if array.ndim == 0 or array.shape[0] != total:
            raise ValueError(f"P6 prediction field 第一维不一致: {key}")
        output[key] = array[indices]
    return output


def _reject_existing(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"P6 输出目录禁止覆盖: {output}")


def _temporary_output(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{output.name}.incomplete_", dir=output.parent))


def _record_failure(directory: Path, error: BaseException) -> None:
    try:
        _write_json(
            directory / "failure.json",
            {"status": "failed", "error_type": type(error).__name__, "error": str(error)},
        )
    except Exception:
        pass


def _finish_output(temporary: Path, output: Path, receipt: Mapping[str, Any]) -> None:
    receipt_name = {
        "p6_train_validation_target_attributes": "target_attribute_receipt.json",
        "p6_validation_waveform_selection": "waveform_selection_receipt.json",
        "p6_validation_waveform_export": "waveform_export_receipt.json",
    }[str(receipt["phase"])]
    _write_json(temporary / receipt_name, receipt)
    _write_json(
        temporary / "artifact_manifest.json",
        {"protocol_id": PROTOCOL_ID, "status": "complete", "files": _artifact_records(temporary)},
    )
    os.replace(temporary, output)


def _verify_artifact_manifest(directory: Path, manifest: Mapping[str, Any]) -> None:
    if manifest.get("protocol_id") != PROTOCOL_ID or manifest.get("status") != "complete":
        raise RuntimeError("P6 artifact manifest identity 漂移")
    for record in manifest.get("files", []):
        path = directory / record["filename"]
        if path.stat().st_size != int(record["size_bytes"]) or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"P6 artifact 漂移: {path.name}")


def _file_record(path: Path, role: str, expected_hash: str | None = None) -> dict[str, Any]:
    digest = sha256_file(path)
    if expected_hash is not None and digest != expected_hash:
        raise RuntimeError(f"P6 frozen source SHA-256 漂移: {role}")
    return {"role": role, "path": str(path.resolve()), "sha256": digest, "size_bytes": path.stat().st_size}


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
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if commit.returncode != 0 or status.returncode != 0 or status.stdout.strip():
        raise RuntimeError("P6 正式产物要求干净 Git commit")
    return commit.stdout.strip()


__all__ = [
    "CATEGORIES",
    "CONTRACT_PATH",
    "CONTRACT_SHA256",
    "EXPORT_OUTPUT",
    "PRIMARY_METRICS",
    "PROTOCOL_ID",
    "SEEDS",
    "SELECTION_OUTPUT",
    "TARGET_OUTPUT",
    "aggregate_w0_validation_metrics",
    "assign_rr_strata",
    "build_p6_target_attributes",
    "build_numerical_replay_plan",
    "build_target_attribute_frame",
    "build_waveform_selection",
    "export_p6_validation_waveforms",
    "load_contract",
    "render_waveform_panel",
    "rr_cutpoints",
    "select_p6_validation_waveforms",
]
