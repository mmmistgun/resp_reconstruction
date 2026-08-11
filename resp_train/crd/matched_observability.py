from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.optimize import linear_sum_assignment
from scipy.signal import coherence, welch

from resp_train.baselines.fixed_band import (
    FIXED_BAND_EXPECTED_SIGNAL_KEY,
    FIXED_BAND_METHOD,
    prepare_fixed_band_config,
)
from resp_train.crd.candidate_lock import DEFAULT_CANDIDATE_LOCK, REPO_ROOT, sha256_file, verify_candidate_lock
from resp_train.crd.config import (
    CRD_102_FAILURE_METADATA_PROTOCOL_VERSION,
    CRD_102_MATCHED_OBSERVABILITY_PROTOCOL_VERSION,
    FORMAL_SEEDS,
)
from resp_train.crd.failure_metadata import (
    DEFAULT_OUTPUT_ROOT as DEFAULT_METADATA_ROOT,
    MANIFEST_FILENAME as METADATA_MANIFEST_FILENAME,
    WINDOW_FILENAME as METADATA_WINDOW_FILENAME,
)
from resp_train.crd.s2_selection import BASE_VARIANT
from resp_train.data.research_v2 import ResearchV2WindowDataset, adapt_research_v2_index
from resp_train.metrics.task import evaluate_task_predictions
from resp_train.protocols.respiration import canonicalize_numpy
from resp_train.utils.run import save_execution_manifest


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs/crd_v1/crd_102_matched_observability_diagnostic"

PAIR_FILENAME = "crd102_observability_match_pairs.csv"
PAIR_AUDIT_FILENAME = "crd102_observability_match_audit.json"
PROXY_METRICS_FILENAME = "crd102_observability_proxy_metrics.csv"
DESCRIPTOR_FILENAME = "crd102_observability_signal_descriptors.csv"
PAIRED_TASK_FILENAME = "crd102_observability_paired_task_differences.csv"
PAIRED_TASK_SUMMARY_FILENAME = "crd102_observability_paired_task_summary.csv"
PAIRED_DESCRIPTOR_FILENAME = "crd102_observability_paired_descriptor_differences.csv"
PAIRED_DESCRIPTOR_SUMMARY_FILENAME = "crd102_observability_paired_descriptor_summary.csv"
DECISION_FILENAME = "crd102_observability_decision.json"
MANIFEST_FILENAME = "crd102_observability_manifest.json"

RAWISH_METHOD = "P0_rawish_direct_proxy"
RAWISH_EXPECTED_SIGNAL_KEY = "bcg_rawish_wideband_state_aligned_segment_soft_z"
TARGET_EXPECTED_SIGNAL_KEY = "tho_waveform_segment_soft_z"
MATCH_SCHEMES = ("exact_state_primary", "same_samp_sensitivity")
MATCH_VARIABLES = (
    "target_envelope_modulation",
    "waveform_confidence_score",
    "transient_motion_ratio",
)
EPISODE_STEP_SEC = 30.0
WINDOW_DURATION_SEC = 180.0
MATCH_CALIPER = 2.0
MIN_PRIMARY_PAIRS = 12
WORSE_FRACTION_THRESHOLD = 2.0 / 3.0
COHERENCE_NPERSEG = 2048
COHERENCE_NOVERLAP = 1024

TASK_METRICS: dict[str, str] = {
    "whole_rr_abs_error_bpm": "minimize",
    "local_rr_mae_bpm": "minimize",
    "envelope_trajectory_mae": "minimize",
    "global_envelope_modulation_error": "minimize",
    "lag_aware_signed_pcc": "maximize",
    "ibi_medae_sec": "minimize",
    "ibi_coverage": "maximize",
    "target_stratified_envelope_spearman": "maximize",
}
DESCRIPTORS: dict[str, str] = {
    "band_coherence_mean": "maximize",
    "band_coherence_median": "maximize",
    "dominant_frequency_abs_error_bpm": "minimize",
}


def summarize_crd102_matched_observability(
    *,
    candidate_lock_path: str | Path = DEFAULT_CANDIDATE_LOCK,
    metadata_root: str | Path = DEFAULT_METADATA_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    """用匹配的 high-modulation failure/control 波形判断输入可观测性签名。"""

    _assert_clean_repository()
    verification = verify_candidate_lock(candidate_lock_path)
    records = sorted(
        (record for record in verification.records if record["variant"] == BASE_VARIANT),
        key=lambda item: int(item["seed"]),
    )
    if len(records) != len(FORMAL_SEEDS):
        raise RuntimeError("candidate lock 中 CRD_102 checkpoint 不完整")

    metadata_root = Path(metadata_root)
    metadata_manifest_path = metadata_root / METADATA_MANIFEST_FILENAME
    metadata_windows_path = metadata_root / METADATA_WINDOW_FILENAME
    if not metadata_manifest_path.exists() or not metadata_windows_path.exists():
        raise FileNotFoundError("CRD_102 failure metadata diagnostic 产物不完整")
    metadata_manifest = json.loads(metadata_manifest_path.read_text(encoding="utf-8"))
    if (
        metadata_manifest.get("protocol") != CRD_102_FAILURE_METADATA_PROTOCOL_VERSION
        or metadata_manifest.get("git_dirty") is not False
        or metadata_manifest.get("research_test_used") is not False
        or metadata_manifest.get("candidate_lock_sha256") != verification.lock_sha256
    ):
        raise RuntimeError("CRD_102 failure metadata manifest 不满足冻结要求")

    config_paths = [Path(record["checkpoint_path"]).parent / "config.yaml" for record in records]
    configs = [OmegaConf.load(path) for path in config_paths]
    dataset_specs = {
        (str(cfg.data.dataset_root), str(cfg.data.index_csv), int(cfg.window.duration_samples), float(cfg.window.target_fs))
        for cfg in configs
    }
    if len(dataset_specs) != 1:
        raise RuntimeError(f"CRD_102 三 seed 数据/窗口配置不一致: {sorted(dataset_specs)}")
    cfg = configs[0]
    index_path = (Path(str(cfg.data.dataset_root)) / str(cfg.data.index_csv)).resolve()
    metadata = pd.read_csv(metadata_windows_path)
    index = pd.read_csv(index_path)

    pairs, match_audit = select_matched_windows(metadata)
    selected_ids = sorted(set(pairs["case_dataset_row_id"]) | set(pairs["control_dataset_row_id"]))
    selected_rows = index.loc[index["dataset_row_id"].isin(selected_ids)].copy()
    if len(selected_rows) != len(selected_ids):
        raise RuntimeError("matched observability 的 index rows 配对不完整")

    raw_predictions, raw_signals = _load_proxy_signals(selected_rows, index_path=index_path, cfg=cfg, fixed_band=False)
    fixed_predictions, fixed_signals = _load_proxy_signals(
        selected_rows,
        index_path=index_path,
        cfg=cfg,
        fixed_band=True,
    )
    if not np.array_equal(raw_predictions["dataset_row_id"], fixed_predictions["dataset_row_id"]):
        raise RuntimeError("rawish/fixed-band proxy identity 不一致")
    if not np.array_equal(raw_predictions["tho_ref"], fixed_predictions["tho_ref"]):
        raise RuntimeError("rawish/fixed-band proxy target 不一致")

    raw_metrics = evaluate_task_predictions(raw_predictions, cfg, method=RAWISH_METHOD)
    fixed_metrics = evaluate_task_predictions(fixed_predictions, cfg, method=FIXED_BAND_METHOD)
    proxy_metrics = pd.concat([raw_metrics, fixed_metrics], ignore_index=True)
    descriptors = pd.concat(
        [
            _signal_descriptors(raw_predictions["dataset_row_id"], raw_signals, raw_predictions["tho_ref"], cfg, RAWISH_METHOD),
            _signal_descriptors(
                fixed_predictions["dataset_row_id"],
                fixed_signals,
                fixed_predictions["tho_ref"],
                cfg,
                FIXED_BAND_METHOD,
            ),
        ],
        ignore_index=True,
    )
    paired_task, paired_task_summary = paired_task_descriptives(pairs, metadata, proxy_metrics)
    paired_descriptor, paired_descriptor_summary = paired_descriptor_descriptives(pairs, descriptors)
    decision = apply_observability_decision(paired_task_summary)

    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"CRD_102 matched observability diagnostic 禁止覆盖: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = output_root.parent / f".{output_root.name}.{uuid4().hex}.tmp"
    temporary_dir.mkdir(parents=False, exist_ok=False)
    try:
        pairs.to_csv(temporary_dir / PAIR_FILENAME, index=False)
        (temporary_dir / PAIR_AUDIT_FILENAME).write_text(
            json.dumps(match_audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        proxy_metrics.to_csv(temporary_dir / PROXY_METRICS_FILENAME, index=False)
        descriptors.to_csv(temporary_dir / DESCRIPTOR_FILENAME, index=False)
        paired_task.to_csv(temporary_dir / PAIRED_TASK_FILENAME, index=False)
        paired_task_summary.to_csv(temporary_dir / PAIRED_TASK_SUMMARY_FILENAME, index=False)
        paired_descriptor.to_csv(temporary_dir / PAIRED_DESCRIPTOR_FILENAME, index=False)
        paired_descriptor_summary.to_csv(temporary_dir / PAIRED_DESCRIPTOR_SUMMARY_FILENAME, index=False)
        (temporary_dir / DECISION_FILENAME).write_text(
            json.dumps(
                {
                    "protocol": CRD_102_MATCHED_OBSERVABILITY_PROTOCOL_VERSION,
                    "candidate_lock_sha256": verification.lock_sha256,
                    "source_metadata_manifest_sha256": sha256_file(metadata_manifest_path),
                    "source_metadata_windows_sha256": sha256_file(metadata_windows_path),
                    "dataset_index_sha256": sha256_file(index_path),
                    "research_test_used": False,
                    "model_inference_used": False,
                    "checkpoint_reselection_allowed": False,
                    "confirmatory_p_values_used": False,
                    "exploratory_diagnostic_only": True,
                    **decision,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        save_execution_manifest(
            temporary_dir / MANIFEST_FILENAME,
            task="crd_102_matched_observability_diagnostic",
            phase="crd102_high_modulation_matched_observability",
            protocol=CRD_102_MATCHED_OBSERVABILITY_PROTOCOL_VERSION,
            candidate_lock_sha256=verification.lock_sha256,
            source_metadata_manifest_sha256=sha256_file(metadata_manifest_path),
            source_metadata_windows_sha256=sha256_file(metadata_windows_path),
            dataset_index=str(index_path),
            dataset_index_sha256=sha256_file(index_path),
            primary_pair_count=decision["primary_pair_count"],
            diagnostic_outcome=decision["diagnostic_outcome"],
            research_test_used=False,
            model_inference_used=False,
        )
        os.replace(temporary_dir, output_root)
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
    return output_root / DECISION_FILENAME


def select_matched_windows(metadata: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {
        "dataset_row_id",
        "samp_id",
        "coupling_state_id",
        "window_start_s",
        "window_end_s",
        "envelope_target_stratum",
        "target_envelope_modulation",
        "waveform_confidence_score",
        "transient_motion_ratio",
        "local_rr_mae_bpm_persistent_failure",
        "persistent_core_failure_count",
    }
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"matched observability metadata 缺少字段: {missing}")
    if metadata.empty or metadata["dataset_row_id"].duplicated().any():
        raise ValueError("matched observability metadata 必须非空且 identity 唯一")

    ordered = metadata.sort_values(["samp_id", "window_start_s", "dataset_row_id"], kind="stable").reset_index(
        drop=True
    )
    failures = ordered.loc[ordered["local_rr_mae_bpm_persistent_failure"].astype(bool)].copy()
    new_episode = failures["samp_id"].ne(failures["samp_id"].shift()) | failures["window_start_s"].sub(
        failures["window_start_s"].shift()
    ).gt(EPISODE_STEP_SEC + 1e-12)
    failures["episode_id"] = new_episode.cumsum().astype(int)
    case_rows: list[dict[str, Any]] = []
    for episode_id, group in failures.groupby("episode_id", sort=True):
        group = group.sort_values(["window_start_s", "dataset_row_id"]).reset_index(drop=True)
        representative = group.iloc[(len(group) - 1) // 2]
        if len(group) < 2 or str(representative["envelope_target_stratum"]) != "high":
            continue
        row = representative.to_dict()
        row.update(
            {
                "episode_id": int(episode_id),
                "episode_window_n": int(len(group)),
                "episode_start_s": float(group["window_start_s"].min()),
                "episode_end_s": float(group["window_end_s"].max()),
            }
        )
        case_rows.append(row)
    cases = pd.DataFrame(case_rows)
    if cases.empty:
        raise RuntimeError("matched observability 没有 high-modulation multi-window failure episode")

    controls = ordered.loc[
        ordered["envelope_target_stratum"].astype(str).eq("high")
        & ~ordered["local_rr_mae_bpm_persistent_failure"].astype(bool)
        & ordered["persistent_core_failure_count"].eq(0)
    ].copy()
    nonoverlap: list[bool] = []
    for _, control in controls.iterrows():
        starts = failures.loc[failures["samp_id"].eq(control["samp_id"]), "window_start_s"]
        nonoverlap.append(bool(starts.sub(float(control["window_start_s"])).abs().ge(WINDOW_DURATION_SEC).all()))
    controls = controls.loc[nonoverlap].reset_index(drop=True)

    high = ordered.loc[ordered["envelope_target_stratum"].astype(str).eq("high")]
    scales = high[list(MATCH_VARIABLES)].quantile(0.75) - high[list(MATCH_VARIABLES)].quantile(0.25)
    if not np.isfinite(scales.to_numpy(dtype=np.float64)).all() or (scales <= 0.0).any():
        raise RuntimeError(f"matched observability matching IQR 非正或非有限: {scales.to_dict()}")

    pair_frames = [
        _optimal_match(cases, controls, scales, exact_state=True, scheme="exact_state_primary"),
        _optimal_match(cases, controls, scales, exact_state=False, scheme="same_samp_sensitivity"),
    ]
    pairs = pd.concat(pair_frames, ignore_index=True)
    primary_count = int(pairs["match_scheme"].eq("exact_state_primary").sum())
    if primary_count < MIN_PRIMARY_PAIRS:
        raise RuntimeError(f"matched observability primary pairs 不足: {primary_count} < {MIN_PRIMARY_PAIRS}")
    audit = {
        "high_modulation_multiwindow_case_count": int(len(cases)),
        "eligible_nonoverlap_control_count": int(len(controls)),
        "matching_variables": list(MATCH_VARIABLES),
        "matching_iqr": {column: float(scales[column]) for column in MATCH_VARIABLES},
        "normalized_l1_caliper": MATCH_CALIPER,
        "primary_exact_state_pair_count": primary_count,
        "sensitivity_same_samp_pair_count": int(pairs["match_scheme"].eq("same_samp_sensitivity").sum()),
        "primary_samp_ids": sorted(
            map(int, pairs.loc[pairs["match_scheme"].eq("exact_state_primary"), "case_samp_id"].unique())
        ),
        "sensitivity_samp_ids": sorted(
            map(int, pairs.loc[pairs["match_scheme"].eq("same_samp_sensitivity"), "case_samp_id"].unique())
        ),
    }
    return pairs, audit


def _optimal_match(
    cases: pd.DataFrame,
    controls: pd.DataFrame,
    scales: pd.Series,
    *,
    exact_state: bool,
    scheme: str,
) -> pd.DataFrame:
    invalid_cost = 1e9
    unmatched_cost = 1e6
    cost = np.full((len(cases), len(controls) + len(cases)), invalid_cost, dtype=np.float64)
    raw_cost = np.full((len(cases), len(controls)), np.nan, dtype=np.float64)
    for case_position, (_, case) in enumerate(cases.iterrows()):
        for control_position, (_, control) in enumerate(controls.iterrows()):
            if int(case["samp_id"]) != int(control["samp_id"]):
                continue
            if exact_state and int(case["coupling_state_id"]) != int(control["coupling_state_id"]):
                continue
            normalized = sum(
                abs(float(case[column]) - float(control[column])) / float(scales[column])
                for column in MATCH_VARIABLES
            )
            if normalized > MATCH_CALIPER:
                continue
            raw_cost[case_position, control_position] = normalized
            time_tie_break = 1e-9 * abs(float(case["window_start_s"]) - float(control["window_start_s"]))
            cost[case_position, control_position] = normalized + time_tie_break
    cost[:, len(controls) :] = unmatched_cost
    row_indices, column_indices = linear_sum_assignment(cost)

    rows: list[dict[str, Any]] = []
    for case_position, control_position in zip(row_indices, column_indices, strict=True):
        if control_position >= len(controls) or cost[case_position, control_position] >= unmatched_cost:
            continue
        case = cases.iloc[case_position]
        control = controls.iloc[control_position]
        rows.append(
            {
                "match_scheme": scheme,
                "pair_id": len(rows) + 1,
                "normalized_l1_cost": float(raw_cost[case_position, control_position]),
                "case_dataset_row_id": int(case["dataset_row_id"]),
                "control_dataset_row_id": int(control["dataset_row_id"]),
                "case_episode_id": int(case["episode_id"]),
                "case_episode_window_n": int(case["episode_window_n"]),
                "case_samp_id": int(case["samp_id"]),
                "control_samp_id": int(control["samp_id"]),
                "case_coupling_state_id": int(case["coupling_state_id"]),
                "control_coupling_state_id": int(control["coupling_state_id"]),
                "case_window_start_s": float(case["window_start_s"]),
                "control_window_start_s": float(control["window_start_s"]),
                **{
                    f"case_{column}": float(case[column])
                    for column in MATCH_VARIABLES
                },
                **{
                    f"control_{column}": float(control[column])
                    for column in MATCH_VARIABLES
                },
            }
        )
    return pd.DataFrame(rows)


def _load_proxy_signals(
    selected_rows: pd.DataFrame,
    *,
    index_path: Path,
    cfg: Any,
    fixed_band: bool,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    prepared = prepare_fixed_band_config(cfg) if fixed_band else OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    prepared.data.preload_windows = False
    prepared.training.device = "cpu"
    adapted = adapt_research_v2_index(selected_rows, prepared).sort_values("dataset_row_id").reset_index(drop=True)
    expected_source = FIXED_BAND_EXPECTED_SIGNAL_KEY if fixed_band else RAWISH_EXPECTED_SIGNAL_KEY
    source_keys = sorted(set(adapted["bcg_signal_key"].astype(str)))
    target_keys = sorted(set(adapted["target_signal_key"].astype(str)))
    if source_keys != [expected_source] or target_keys != [TARGET_EXPECTED_SIGNAL_KEY]:
        raise RuntimeError(
            f"observability signal key 异常: source={source_keys}/{expected_source}, "
            f"target={target_keys}/{TARGET_EXPECTED_SIGNAL_KEY}"
        )
    dataset = ResearchV2WindowDataset(index_path, adapted, prepared, preload_windows=False)
    signals: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    row_ids: list[int] = []
    samp_ids: list[int] = []
    states: list[int] = []
    for item in dataset:
        signals.append(item["x"].numpy().reshape(-1))
        targets.append(item["target"].numpy().reshape(-1))
        row_ids.append(int(item["meta"]["dataset_row_id"]))
        samp_ids.append(int(item["meta"]["samp_id"]))
        states.append(int(item["meta"]["coupling_state_id"]))
    signal_array = np.stack(signals).astype(np.float64)
    target_array = np.stack(targets).astype(np.float64)
    if not np.isfinite(signal_array).all() or not np.isfinite(target_array).all():
        raise FloatingPointError("observability proxy signals 包含 NaN/Inf")
    return (
        {
            "r_tho_hat": signal_array,
            "tho_ref": target_array,
            "dataset_row_id": np.asarray(row_ids, dtype=np.int64),
            "split": np.asarray(["val"] * len(row_ids)),
            "input_set": np.asarray(["research_v2_waveform"] * len(row_ids)),
            "samp_id": np.asarray(samp_ids, dtype=np.int64),
            "coupling_state_id": np.asarray(states, dtype=np.int64),
        },
        signal_array,
    )


def _signal_descriptors(
    row_ids: np.ndarray,
    signals: np.ndarray,
    targets: np.ndarray,
    cfg: Any,
    method: str,
) -> pd.DataFrame:
    fs = float(cfg.window.target_fs)
    low = float(cfg.loss.band_low_hz)
    high = float(cfg.loss.band_high_hz)
    _, signal_canonical = canonicalize_numpy(
        signals,
        fs=fs,
        low_hz=low,
        high_hz=high,
        scale_eps=float(cfg.loss.scale_eps),
    )
    _, target_canonical = canonicalize_numpy(
        targets,
        fs=fs,
        low_hz=low,
        high_hz=high,
        scale_eps=float(cfg.loss.scale_eps),
    )
    rows: list[dict[str, Any]] = []
    for position, row_id in enumerate(row_ids):
        frequency, cross = coherence(
            signal_canonical[position],
            target_canonical[position],
            fs=fs,
            nperseg=COHERENCE_NPERSEG,
            noverlap=COHERENCE_NOVERLAP,
        )
        band_mask = (frequency >= low) & (frequency <= high)
        signal_frequency, signal_power = welch(
            signal_canonical[position],
            fs=fs,
            nperseg=COHERENCE_NPERSEG,
            noverlap=COHERENCE_NOVERLAP,
        )
        target_frequency, target_power = welch(
            target_canonical[position],
            fs=fs,
            nperseg=COHERENCE_NPERSEG,
            noverlap=COHERENCE_NOVERLAP,
        )
        signal_band = (signal_frequency >= low) & (signal_frequency <= high)
        target_band = (target_frequency >= low) & (target_frequency <= high)
        signal_peak_hz = float(signal_frequency[signal_band][np.argmax(signal_power[signal_band])])
        target_peak_hz = float(target_frequency[target_band][np.argmax(target_power[target_band])])
        rows.append(
            {
                "method": method,
                "dataset_row_id": int(row_id),
                "band_coherence_mean": float(np.mean(cross[band_mask])),
                "band_coherence_median": float(np.median(cross[band_mask])),
                "signal_dominant_frequency_hz": signal_peak_hz,
                "target_dominant_frequency_hz": target_peak_hz,
                "dominant_frequency_abs_error_bpm": abs(signal_peak_hz - target_peak_hz) * 60.0,
            }
        )
    return pd.DataFrame(rows)


def paired_task_descriptives(
    pairs: pd.DataFrame,
    metadata: pd.DataFrame,
    proxy_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata_by_id = metadata.set_index("dataset_row_id")
    proxy_by_method = {
        method: frame.set_index("dataset_row_id")
        for method, frame in proxy_metrics.groupby("method", sort=False)
    }
    rows: list[dict[str, Any]] = []
    for _, pair in pairs.iterrows():
        case_id = int(pair["case_dataset_row_id"])
        control_id = int(pair["control_dataset_row_id"])
        for method in ("crd_102_seed_mean", RAWISH_METHOD, FIXED_BAND_METHOD):
            for metric, direction in TASK_METRICS.items():
                if method == "crd_102_seed_mean":
                    column = f"{metric}_seed_mean"
                    case_value = float(metadata_by_id.loc[case_id, column])
                    control_value = float(metadata_by_id.loc[control_id, column])
                else:
                    case_value = float(proxy_by_method[method].loc[case_id, metric])
                    control_value = float(proxy_by_method[method].loc[control_id, metric])
                if not np.isfinite(case_value) or not np.isfinite(control_value):
                    continue
                delta = case_value - control_value
                case_worse = delta > 0.0 if direction == "minimize" else delta < 0.0
                rows.append(
                    {
                        "match_scheme": pair["match_scheme"],
                        "pair_id": int(pair["pair_id"]),
                        "case_dataset_row_id": case_id,
                        "control_dataset_row_id": control_id,
                        "method": method,
                        "metric": metric,
                        "direction": direction,
                        "case_value": case_value,
                        "control_value": control_value,
                        "case_minus_control": delta,
                        "case_worse": bool(case_worse),
                    }
                )
    paired = pd.DataFrame(rows)
    return paired, _paired_summary(paired)


def paired_descriptor_descriptives(
    pairs: pd.DataFrame,
    descriptors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_method = {
        method: frame.set_index("dataset_row_id")
        for method, frame in descriptors.groupby("method", sort=False)
    }
    rows: list[dict[str, Any]] = []
    for _, pair in pairs.iterrows():
        case_id = int(pair["case_dataset_row_id"])
        control_id = int(pair["control_dataset_row_id"])
        for method in (RAWISH_METHOD, FIXED_BAND_METHOD):
            for descriptor, direction in DESCRIPTORS.items():
                case_value = float(by_method[method].loc[case_id, descriptor])
                control_value = float(by_method[method].loc[control_id, descriptor])
                delta = case_value - control_value
                case_worse = delta > 0.0 if direction == "minimize" else delta < 0.0
                rows.append(
                    {
                        "match_scheme": pair["match_scheme"],
                        "pair_id": int(pair["pair_id"]),
                        "case_dataset_row_id": case_id,
                        "control_dataset_row_id": control_id,
                        "method": method,
                        "metric": descriptor,
                        "direction": direction,
                        "case_value": case_value,
                        "control_value": control_value,
                        "case_minus_control": delta,
                        "case_worse": bool(case_worse),
                    }
                )
    paired = pd.DataFrame(rows)
    return paired, _paired_summary(paired)


def _paired_summary(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in paired.groupby(["match_scheme", "method", "metric", "direction"], sort=True):
        scheme, method, metric, direction = keys
        rows.append(
            {
                "match_scheme": scheme,
                "method": method,
                "metric": metric,
                "direction": direction,
                "pair_n": int(len(group)),
                "case_mean": float(group["case_value"].mean()),
                "control_mean": float(group["control_value"].mean()),
                "case_minus_control_mean": float(group["case_minus_control"].mean()),
                "case_minus_control_median": float(group["case_minus_control"].median()),
                "case_worse_count": int(group["case_worse"].sum()),
                "case_worse_fraction": float(group["case_worse"].mean()),
            }
        )
    return pd.DataFrame(rows)


def apply_observability_decision(summary: pd.DataFrame) -> dict[str, Any]:
    required = {"match_scheme", "method", "metric", "pair_n", "case_worse_fraction"}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"observability decision summary 缺少字段: {missing}")
    primary = summary.loc[summary["match_scheme"].eq("exact_state_primary")]
    decision_primary = primary.loc[
        primary["method"].isin((RAWISH_METHOD, FIXED_BAND_METHOD))
        & primary["metric"].isin(("local_rr_mae_bpm", "lag_aware_signed_pcc"))
    ]
    pair_counts = set(decision_primary["pair_n"].astype(int))
    if len(pair_counts) != 1 or next(iter(pair_counts)) < MIN_PRIMARY_PAIRS:
        raise RuntimeError(f"observability decision primary pair 数异常: {sorted(pair_counts)}")
    proxy_checks: dict[str, Any] = {}
    for method in (RAWISH_METHOD, FIXED_BAND_METHOD):
        method_rows = primary.loc[primary["method"].eq(method)].set_index("metric")
        if not {"local_rr_mae_bpm", "lag_aware_signed_pcc"}.issubset(method_rows.index):
            raise RuntimeError(f"observability decision 缺少 proxy primary: {method}")
        local_fraction = float(method_rows.loc["local_rr_mae_bpm", "case_worse_fraction"])
        pcc_fraction = float(method_rows.loc["lag_aware_signed_pcc", "case_worse_fraction"])
        proxy_checks[method] = {
            "local_rr_case_worse_fraction": local_fraction,
            "signed_pcc_case_worse_fraction": pcc_fraction,
            "local_rr_case_worse_pass": local_fraction >= WORSE_FRACTION_THRESHOLD,
            "signed_pcc_case_worse_pass": pcc_fraction >= WORSE_FRACTION_THRESHOLD,
            "observability_failure_signature": (
                local_fraction >= WORSE_FRACTION_THRESHOLD and pcc_fraction >= WORSE_FRACTION_THRESHOLD
            ),
        }
    passed = [method for method, details in proxy_checks.items() if details["observability_failure_signature"]]
    if len(passed) == 2:
        outcome = "input_observability_associated"
    elif len(passed) == 0:
        outcome = "model_specific_tracking_associated"
    else:
        outcome = "mixed_observability_and_model_tracking"
    return {
        "primary_pair_count": int(next(iter(pair_counts))),
        "primary_match_scheme": "exact_state_primary",
        "case_worse_fraction_threshold": WORSE_FRACTION_THRESHOLD,
        "proxy_checks": proxy_checks,
        "diagnostic_outcome": outcome,
        "causal_claim_allowed": False,
    }


def _assert_clean_repository() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode != 0:
        raise RuntimeError(f"无法读取 git 状态: {status.stderr.strip()}")
    if status.stdout.strip():
        raise RuntimeError("CRD_102 matched observability diagnostic 必须从干净 git commit 生成")
