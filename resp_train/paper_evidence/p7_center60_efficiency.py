from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from resp_train.crd.experiment import _validate_checkpoint_config
from resp_train.data.cache import WholeNightCache
from resp_train.data.index import filter_index
from resp_train.data.research_v2 import read_research_v2_index
from resp_train.paper_evidence import p7_iot_efficiency as p7
from resp_train.paper_evidence.center_context_cache import (
    CenterContextWCacheReader,
    center_context_cwt_features,
)
from resp_train.paper_evidence.center_context_config import load_center_context_config
from resp_train.paper_evidence.center_context_loss import pi60_torch
from resp_train.paper_evidence.center_context_model import build_center_context_model


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path("configs/paper_evidence_v1/p7_center60_efficiency_v1.json")
CONFIG_SHA256 = "10a8388843b225cdd32fabe8cd9d2f37e7c457076fe8936e226bff771fc08ff9"
PROTOCOL_ID = "paper-p7-center60-efficiency-v1-20260905"
OUTPUT_ROOT = Path("runs/paper_evidence_v1/p7_center60_efficiency")
C201 = "C201_CENTER60_60"
W_REDUCED = "W_REDUCED_CENTER60_60"
MODEL_ORDER = (C201, W_REDUCED)
SCENARIOS = {C201: ("waveform_only",), W_REDUCED: ("cached_w", "online_w")}
QUALITY_COLUMNS = (
    "center_rr_mae_bpm",
    "center_ibi_medae_sec",
    "center_envelope_trajectory_mae",
    "center_global_envelope_modulation_error",
    "center_lag_aware_signed_pcc",
    "center_ibi_coverage",
    "center_ibi_interpretable_fraction",
)
QUALITY_SOURCE_PREFIXES = {
    "center_rr_mae_bpm": "center_rr_mae_bpm_mean",
    "center_ibi_medae_sec": "center_ibi_medae_sec_mean",
    "center_envelope_trajectory_mae": "center_envelope_trajectory_mae_mean",
    "center_global_envelope_modulation_error": "center_global_envelope_modulation_error_mean",
    "center_lag_aware_signed_pcc": "center_lag_aware_signed_pcc_mean",
    "center_ibi_coverage": "center_ibi_coverage_mean",
    "center_ibi_interpretable_fraction": "center_ibi_interpretable_fraction",
}
REQUIRED_OUTPUTS = (
    "environment.json",
    "input_materialization.csv",
    "stage_latency_iterations.csv",
    "stage_latency_summary.csv",
    "memory_summary.csv",
    "model_resource_summary.csv",
    "quality_efficiency_table.csv",
    "descriptive_efficiency_context_table.csv",
    "manifest.json",
)


def load_contract(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    path = root / CONFIG_PATH
    if p7.sha256_file(path) != CONFIG_SHA256:
        raise RuntimeError("P7 center60 config SHA-256 漂移")
    contract = json.loads(path.read_text(encoding="utf-8"))
    validate_contract(contract)
    return contract


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != "paper-p7-center60-efficiency-config-v1":
        raise ValueError("P7 center60 schema 漂移")
    if contract.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("P7 center60 protocol identity 漂移")
    if contract.get("scientific_scope") != "separate_60s_input_60s_output_task":
        raise ValueError("P7 center60 scientific scope 漂移")
    if contract.get("authorization") != {
        "gpu_benchmark": True,
        "cpu_model_benchmark": False,
        "training": False,
        "cache_build": False,
        "research_test_signal_or_target_access": False,
    }:
        raise ValueError("P7 center60 authorization 漂移")
    platform_contract = contract.get("platform", {})
    if (
        platform_contract.get("device") != "cuda:0"
        or int(platform_contract.get("batch_size", -1)) != 1
        or platform_contract.get("amp_dtype") != "bfloat16"
        or platform_contract.get("host_memory") != "pageable"
    ):
        raise ValueError("P7 center60 platform 漂移")
    benchmark = contract.get("benchmark", {})
    if (
        int(benchmark.get("warmup_iterations", -1)) != 20
        or int(benchmark.get("timed_iterations", -1)) != 100
        or int(benchmark.get("rounds", -1)) != 5
        or int(benchmark.get("expected_timed_pipeline_iterations", -1)) != 1500
        or tuple(benchmark.get("c201_scenarios", ())) != SCENARIOS[C201]
        or tuple(benchmark.get("w_reduced_scenarios", ())) != SCENARIOS[W_REDUCED]
        or benchmark.get("profiler_position") != "after_all_latency_and_memory_measurements"
    ):
        raise ValueError("P7 center60 benchmark matrix 漂移")
    model_orders = benchmark.get("round_model_order", ())
    if len(model_orders) != 5 or any(set(order) != set(MODEL_ORDER) for order in model_orders):
        raise ValueError("P7 center60 model schedule 不完整")
    for candidate in MODEL_ORDER:
        counts = [sum(order[position] == candidate for order in model_orders) for position in (0, 1)]
        if abs(counts[0] - counts[1]) > 1:
            raise ValueError("P7 center60 model schedule 不平衡")
    scenario_orders = benchmark.get("w_round_scenario_order", ())
    if len(scenario_orders) != 5 or any(set(order) != set(SCENARIOS[W_REDUCED]) for order in scenario_orders):
        raise ValueError("P7 center60 W scenario schedule 不完整")
    if tuple(item.get("candidate_id") for item in contract.get("models", ())) != MODEL_ORDER:
        raise ValueError("P7 center60 model identity/order 漂移")
    if any(int(item.get("seed", -1)) != 20260811 for item in contract["models"]):
        raise ValueError("P7 center60 checkpoint seed 漂移")
    input_contract = contract.get("input", {})
    if (
        input_contract.get("split") != "val"
        or int(input_contract.get("dataset_row_id", -1)) != 12429
        or int(input_contract.get("input_samples", -1)) != 6000
        or int(input_contract.get("output_samples", -1)) != 6000
        or input_contract.get("input_slice") != [6000, 12000]
    ):
        raise ValueError("P7 center60 input/output identity 漂移")
    w_source = contract.get("w_source", {})
    if (
        w_source.get("online_extractor_device") != "host_cpu"
        or w_source.get("shape") != [49, 120]
        or int(w_source.get("online_scale_count", -1)) != 49
        or w_source.get("length_specific_scale_mapping") is not True
    ):
        raise ValueError("P7 center60 W contract 漂移")
    if contract.get("descriptive_reference", {}).get("cross_task_causal_window_claim_allowed") is not False:
        raise ValueError("P7 center60 cross-task interpretation boundary 漂移")
    if contract.get("output", {}).get("directory") != str(OUTPUT_ROOT):
        raise ValueError("P7 center60 output identity 漂移")


def _verify_file(root: Path, relative: str, expected_sha: str, expected_size: int | None = None) -> Path:
    path = root / relative
    if not path.is_file() or p7.sha256_file(path) != expected_sha:
        raise RuntimeError(f"P7 center60 source artifact 漂移: {relative}")
    if expected_size is not None and path.stat().st_size != int(expected_size):
        raise RuntimeError(f"P7 center60 source artifact size 漂移: {relative}")
    return path


def _load_inputs(
    root: Path, contract: Mapping[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, pd.DataFrame, dict[str, Any]]:
    input_contract = contract["input"]
    _verify_file(root, input_contract["selection_manifest"], input_contract["selection_manifest_sha256"])
    selection_path = _verify_file(
        root, input_contract["selection_rows"], input_contract["selection_rows_sha256"]
    )
    selected = pd.read_csv(selection_path)
    selected = selected.loc[selected["dataset_row_id"] == int(input_contract["dataset_row_id"])]
    if len(selected) != 1 or selected.iloc[0]["category"] != input_contract["selection_category"]:
        raise RuntimeError("P7 center60 predeclared row/category 漂移")

    source = contract["models"][0]
    config_path = _verify_file(root, f"{source['run_dir']}/resolved_config.yaml", source["config_sha256"])
    cfg = load_center_context_config(
        config_path, overrides=["training.device=cuda:0", "training.show_progress=false"]
    )
    audited = read_research_v2_index(cfg.data.dataset_root, cfg.data.index_csv, cfg)
    rows = filter_index(
        audited,
        cfg,
        split="val",
        max_windows=None,
        sample_strategy=str(cfg.data.val_sample_strategy),
        sample_seed=int(cfg.data.val_sample_seed),
    )
    rows = rows.loc[rows["dataset_row_id"] == int(input_contract["dataset_row_id"])]
    if len(rows) != 1:
        raise RuntimeError("P7 center60 validation row identity 漂移")
    row = rows.iloc[0]
    parent_start, parent_stop = int(row["window_start_sample"]), int(row["window_end_sample"])
    if parent_stop - parent_start != int(input_contract["parent_samples"]):
        raise RuntimeError("P7 center60 parent window length 漂移")

    io_rows = []
    cache = WholeNightCache(Path(str(cfg.data.dataset_root)) / str(cfg.data.index_csv))
    begin = time.perf_counter_ns()
    arrays = cache.get_arrays(str(row["source_npz"]), [str(row["bcg_signal_key"])])
    parent = np.asarray(
        arrays[str(row["bcg_signal_key"])][parent_start:parent_stop], dtype=np.float32
    ).copy()
    input_start, input_stop = map(int, input_contract["input_slice"])
    waveform_np = np.ascontiguousarray(parent[input_start:input_stop])
    io_rows.append(
        {
            "source": "validation_bcg_npz_and_center_crop",
            "dataset_row_id": int(row["dataset_row_id"]),
            "latency_ms": (time.perf_counter_ns() - begin) / 1e6,
            "measurement_count": 1,
            "os_cache_state": "uncontrolled_process_initial",
        }
    )
    if waveform_np.shape != (6000,) or not np.isfinite(waveform_np).all():
        raise FloatingPointError("P7 center60 BCG input shape/finite 不合格")

    w_contract = contract["w_source"]
    w_root = root / w_contract["cache_root"]
    _verify_file(
        root,
        f"{w_contract['cache_root']}/cache_manifest.json",
        w_contract["cache_manifest_sha256"],
    )
    begin = time.perf_counter_ns()
    w_reader = CenterContextWCacheReader(
        w_root, split="val", input_samples=6000, require_complete=True
    )
    cached_w = w_reader.get(int(row["dataset_row_id"])).contiguous()
    io_rows.append(
        {
            "source": "frozen_center60_validation_w_cache",
            "dataset_row_id": int(row["dataset_row_id"]),
            "latency_ms": (time.perf_counter_ns() - begin) / 1e6,
            "measurement_count": 1,
            "os_cache_state": "uncontrolled_process_initial",
        }
    )
    online_w, frequencies = center_context_cwt_features(waveform_np)
    delta = float(np.max(np.abs(online_w - cached_w.numpy())))
    if delta > float(w_contract["cached_online_max_abs_tolerance"]):
        raise RuntimeError(f"P7 center60 cached/online W 不一致: max_abs_delta={delta:.17g}")
    identity = {
        "dataset_row_id": int(row["dataset_row_id"]),
        "split": "val",
        "samp_id": int(row["samp_id"]),
        "input_slice": [input_start, input_stop],
        "waveform_sha256": hashlib.sha256(waveform_np.tobytes(order="C")).hexdigest(),
        "cached_w_sha256": hashlib.sha256(cached_w.numpy().tobytes(order="C")).hexdigest(),
        "cached_online_w_max_abs_delta": delta,
        "online_frequency_min_hz": float(frequencies[0]),
        "online_frequency_max_hz": float(frequencies[-1]),
    }
    return (
        torch.from_numpy(waveform_np).view(1, 1, 6000).contiguous(),
        cached_w.view(1, 49, 120),
        pd.DataFrame(io_rows),
        identity,
    )


def _load_model(
    root: Path, source: Mapping[str, Any], device: torch.device
) -> tuple[torch.nn.Module, Any]:
    manifest_path = _verify_file(
        root, f"{source['run_dir']}/artifact_manifest.json", source["artifact_manifest_sha256"]
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("experiment_id") != source["experiment_id"]
        or int(manifest.get("seed", -1)) != int(source["seed"])
        or int(manifest.get("best_epoch", -1)) != int(source["selected_epoch"])
    ):
        raise RuntimeError(f"P7 center60 run manifest identity 漂移: {source['candidate_id']}")
    config_path = _verify_file(
        root, f"{source['run_dir']}/resolved_config.yaml", source["config_sha256"]
    )
    checkpoint_path = _verify_file(
        root,
        f"{source['run_dir']}/checkpoint_best_center_rr.pt",
        source["checkpoint_sha256"],
        int(source["checkpoint_size_bytes"]),
    )
    cfg = load_center_context_config(
        config_path, overrides=[f"training.device={device}", "training.show_progress=false"]
    )
    if (
        str(cfg.model.variant) != source["variant"]
        or int(cfg.training.seed) != int(source["seed"])
        or int(cfg.window.input_samples) != 6000
        or int(cfg.window.output_samples) != 6000
    ):
        raise RuntimeError(f"P7 center60 model config identity 漂移: {source['candidate_id']}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("epoch", -1)) != int(source["selected_epoch"]):
        raise RuntimeError(f"P7 center60 selected epoch 漂移: {source['candidate_id']}")
    _validate_checkpoint_config(checkpoint.get("config"), cfg)
    model = build_center_context_model(cfg)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint
    model.eval().to(device)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable != int(source["trainable_parameters"]):
        raise RuntimeError(f"P7 center60 parameter count 漂移: {source['candidate_id']}")
    return model, cfg


def _amp_context(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def _timed_iteration(
    *,
    candidate_id: str,
    scenario: str,
    model: torch.nn.Module,
    waveform_host: torch.Tensor,
    cached_w_host: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, float], torch.Tensor]:
    if scenario not in SCENARIOS[candidate_id]:
        raise ValueError(f"P7 center60 {candidate_id} 不允许 scenario={scenario}")
    total_start = time.perf_counter_ns()
    stages: dict[str, float] = {}
    w_host: torch.Tensor | None = None
    if scenario == "online_w":
        begin = time.perf_counter_ns()
        values, _ = center_context_cwt_features(waveform_host.numpy().reshape(-1))
        w_host = torch.from_numpy(values).view(1, 49, 120).contiguous()
        stages["w_feature_extraction_direct49_60s_log_pool"] = (
            time.perf_counter_ns() - begin
        ) / 1e6
    elif scenario == "cached_w":
        w_host = cached_w_host

    begin = time.perf_counter_ns()
    waveform = waveform_host.to(device, non_blocking=False)
    tf = None
    if w_host is not None:
        tf = {"w": w_host.to(device, non_blocking=False)}
    torch.cuda.synchronize(device)
    transfer_stage = (
        "host_to_device_pageable_bcg"
        if tf is None
        else "host_to_device_pageable_bcg_and_49x120_w"
    )
    stages[transfer_stage] = (time.perf_counter_ns() - begin) / 1e6

    begin = time.perf_counter_ns()
    with torch.inference_mode(), _amp_context(device):
        output = model(waveform, tf=tf)
    torch.cuda.synchronize(device)
    stages["model_forward_60_to_60"] = (time.perf_counter_ns() - begin) / 1e6
    predicted = output["waveform"]
    if tuple(predicted.shape) != (1, 1, 6000):
        raise RuntimeError("P7 center60 model output shape 漂移")

    begin = time.perf_counter_ns()
    canonical = pi60_torch(predicted)
    torch.cuda.synchronize(device)
    stages["Pi_60"] = (time.perf_counter_ns() - begin) / 1e6
    stages["end_to_end"] = (time.perf_counter_ns() - total_start) / 1e6
    return stages, canonical


def _profile_model(
    *,
    candidate_id: str,
    model: torch.nn.Module,
    waveform_host: torch.Tensor,
    cached_w_host: torch.Tensor,
    device: torch.device,
) -> tuple[float, float, str]:
    waveform = waveform_host.to(device)
    tf = {"w": cached_w_host.to(device)} if candidate_id == W_REDUCED else None
    try:
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(
            activities=activities, record_shapes=True, with_flops=True
        ) as profile:
            with torch.inference_mode(), _amp_context(device):
                model(waveform, tf=tf)
            torch.cuda.synchronize(device)
        flops = float(sum(float(event.flops or 0) for event in profile.key_averages()))
        if not math.isfinite(flops) or flops <= 0:
            return math.nan, math.nan, "unavailable_no_positive_covered_flops"
        return (
            flops,
            flops / 2.0,
            "partial_standard_ops_only_mamba_custom_ops_may_be_uncovered",
        )
    except Exception as error:
        return math.nan, math.nan, f"unavailable_{type(error).__name__}"


def _resource_record(source: Mapping[str, Any], model: torch.nn.Module) -> dict[str, Any]:
    return {
        "candidate_id": source["candidate_id"],
        "experiment_id": source["experiment_id"],
        "model_label": source["model_label"],
        "model_variant": source["variant"],
        "checkpoint_seed": int(source["seed"]),
        "trainable_parameters": int(source["trainable_parameters"]),
        "total_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "bcg_input_elements": int(source["bcg_input_elements"]),
        "w_input_elements": int(source["w_input_elements"]),
        "total_model_input_elements": int(source["bcg_input_elements"])
        + int(source["w_input_elements"]),
        "w_scales": int(source["w_scales"]),
        "checkpoint_size_bytes": int(source["checkpoint_size_bytes"]),
        "checkpoint_sha256": source["checkpoint_sha256"],
        "config_sha256": source["config_sha256"],
    }


def build_quality_efficiency_table(
    contract: Mapping[str, Any],
    latency: pd.DataFrame,
    resources: pd.DataFrame,
    *,
    repo_root: str | Path,
) -> pd.DataFrame:
    root = Path(repo_root).resolve()
    source = contract["quality_source"]
    _verify_file(root, source["path"], source["sha256"])
    _verify_file(root, source["manifest_path"], source["manifest_sha256"])
    quality = pd.read_csv(root / source["path"])
    experiment_by_candidate = {
        item["candidate_id"]: item["experiment_id"] for item in contract["models"]
    }
    quality_rows = []
    for candidate_id in MODEL_ORDER:
        experiment_id = experiment_by_candidate[candidate_id]
        row = quality.loc[quality["experiment_id"] == experiment_id]
        if (
            len(row) != 1
            or int(row.iloc[0]["input_sec"]) != 60
            or int(row.iloc[0]["output_sec"]) != 60
        ):
            raise RuntimeError(f"P7 center60 quality source 不唯一: {candidate_id}")
        row = row.iloc[0]
        values: dict[str, Any] = {
            "candidate_id": candidate_id,
            "experiment_id": experiment_id,
            "quality_scope": source["scope"],
        }
        for metric in QUALITY_COLUMNS:
            prefix = QUALITY_SOURCE_PREFIXES[metric]
            values[f"{metric}_mean"] = float(row[f"{prefix}_seed_mean"])
            values[f"{metric}_sample_sd"] = float(row[f"{prefix}_seed_sample_sd"])
        quality_rows.append(values)
    quality_frame = pd.DataFrame(quality_rows)
    end_to_end = latency.loc[latency["stage"] == "end_to_end", [
        "candidate_id",
        "scenario",
        "median_ms",
        "p95_ms",
        "mean_ms",
        "sample_sd_ms",
        "min_ms",
        "max_ms",
        "p95_slack_ms",
        "p95_updates_per_second",
        "context_wait_seconds",
    ]]
    output = end_to_end.merge(resources, on="candidate_id", validate="many_to_one")
    return output.merge(quality_frame, on=["candidate_id", "experiment_id"], validate="many_to_one")


def build_descriptive_efficiency_context_table(
    contract: Mapping[str, Any],
    center60_quality_efficiency: pd.DataFrame,
    *,
    repo_root: str | Path,
) -> pd.DataFrame:
    root = Path(repo_root).resolve()
    reference = contract["descriptive_reference"]
    manifest_path = _verify_file(
        root, reference["manifest_path"], reference["manifest_sha256"]
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    table_record = manifest.get("files", {}).get("quality_efficiency_table.csv")
    if not isinstance(table_record, Mapping):
        raise RuntimeError("P7 180s reference manifest 缺少 quality table")
    table_path = manifest_path.parent / "quality_efficiency_table.csv"
    if (
        p7.sha256_file(table_path) != table_record["sha256"]
        or table_path.stat().st_size != int(table_record["size_bytes"])
    ):
        raise RuntimeError("P7 180s reference quality table 漂移")
    reference_table = pd.read_csv(table_path)
    rows = []
    for record in center60_quality_efficiency.to_dict("records"):
        rows.append(
            {
                "task_scope": "separate_60s_input_60s_output_task",
                "candidate_id": record["candidate_id"],
                "scenario": record["scenario"],
                "input_seconds": 60,
                "output_seconds": 60,
                "median_ms": record["median_ms"],
                "p95_ms": record["p95_ms"],
                "trainable_parameters": record["trainable_parameters"],
                "checkpoint_size_bytes": record["checkpoint_size_bytes"],
                "w_scales": record["w_scales"],
                "cross_task_causal_window_claim_allowed": False,
            }
        )
    for record in reference_table.to_dict("records"):
        rows.append(
            {
                "task_scope": reference["scope"],
                "candidate_id": record["candidate_id"],
                "scenario": record["scenario"],
                "input_seconds": 180,
                "output_seconds": 180,
                "median_ms": record["median_ms"],
                "p95_ms": record["p95_ms"],
                "trainable_parameters": record["trainable_parameters"],
                "checkpoint_size_bytes": record["checkpoint_size_bytes"],
                "w_scales": record["model_w_scales"],
                "cross_task_causal_window_claim_allowed": False,
            }
        )
    return pd.DataFrame(rows)


def run_center60_gpu_benchmark(
    *,
    repo_root: str | Path = REPO_ROOT,
    command: str,
    device: str,
    confirm: bool,
) -> Path:
    root = Path(repo_root).resolve()
    contract = load_contract(root)
    if not confirm:
        raise RuntimeError("P7 center60 GPU benchmark 需要显式确认")
    if device != contract["platform"]["device"] or not torch.cuda.is_available():
        raise RuntimeError("P7 center60 必须使用合同固定且可用的 cuda:0")
    if any(name in os.environ for name in contract["platform"]["required_unset_environment"]):
        raise RuntimeError("P7 center60 要求 unset LD_LIBRARY_PATH 与 LD_PRELOAD")
    if os.environ.get("SSQ_GPU", "0") != "0":
        raise RuntimeError("P7 center60 online-W 要求 SSQ_GPU=0")
    reference = contract["descriptive_reference"]
    _verify_file(root, reference["manifest_path"], reference["manifest_sha256"])
    output = root / OUTPUT_ROOT
    if output.exists():
        raise FileExistsError(f"P7 center60 output 已存在，拒绝覆盖: {output}")

    commit = p7._git_identity(root)
    torch.backends.cuda.matmul.allow_tf32 = bool(contract["platform"]["allow_tf32"])
    torch.backends.cudnn.benchmark = bool(contract["platform"]["cudnn_benchmark"])
    target_device = torch.device(device)
    waveform_host, cached_w_host, input_io, input_identity = _load_inputs(root, contract)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".p7_center60_efficiency_", dir=output.parent)
    )
    source_by_id = {item["candidate_id"]: item for item in contract["models"]}
    resources: dict[str, dict[str, Any]] = {}
    iteration_rows: list[dict[str, Any]] = []
    memory_rows: list[dict[str, Any]] = []
    benchmark = contract["benchmark"]
    last_output: torch.Tensor | None = None
    try:
        for round_index, model_order in enumerate(
            benchmark["round_model_order"], start=1
        ):
            for candidate_id in model_order:
                source = source_by_id[candidate_id]
                model, _ = _load_model(root, source, target_device)
                resources.setdefault(candidate_id, _resource_record(source, model))
                scenarios = (
                    benchmark["w_round_scenario_order"][round_index - 1]
                    if candidate_id == W_REDUCED
                    else benchmark["c201_scenarios"]
                )
                for scenario in scenarios:
                    torch.cuda.synchronize(target_device)
                    torch.cuda.empty_cache()
                    for _ in range(int(benchmark["warmup_iterations"])):
                        _, last_output = _timed_iteration(
                            candidate_id=candidate_id,
                            scenario=scenario,
                            model=model,
                            waveform_host=waveform_host,
                            cached_w_host=cached_w_host,
                            device=target_device,
                        )
                    torch.cuda.synchronize(target_device)
                    torch.cuda.reset_peak_memory_stats(target_device)
                    baseline_allocated = int(torch.cuda.memory_allocated(target_device))
                    baseline_reserved = int(torch.cuda.memory_reserved(target_device))
                    for iteration in range(1, int(benchmark["timed_iterations"]) + 1):
                        stages, last_output = _timed_iteration(
                            candidate_id=candidate_id,
                            scenario=scenario,
                            model=model,
                            waveform_host=waveform_host,
                            cached_w_host=cached_w_host,
                            device=target_device,
                        )
                        for stage, latency_ms in stages.items():
                            iteration_rows.append(
                                {
                                    "candidate_id": candidate_id,
                                    "model_variant": source["variant"],
                                    "checkpoint_seed": int(source["seed"]),
                                    "scenario": scenario,
                                    "round": round_index,
                                    "iteration": iteration,
                                    "stage": stage,
                                    "latency_ms": float(latency_ms),
                                }
                            )
                    memory_rows.append(
                        {
                            "candidate_id": candidate_id,
                            "scenario": scenario,
                            "round": round_index,
                            "baseline_allocated_bytes": baseline_allocated,
                            "baseline_reserved_bytes": baseline_reserved,
                            "peak_allocated_bytes": int(
                                torch.cuda.max_memory_allocated(target_device)
                            ),
                            "peak_reserved_bytes": int(
                                torch.cuda.max_memory_reserved(target_device)
                            ),
                        }
                    )
                    if last_output is None or not torch.isfinite(last_output).all():
                        raise FloatingPointError(
                            f"P7 center60 {candidate_id}/{scenario} Pi_60 output 非 finite"
                        )
                del model
                torch.cuda.empty_cache()

        iterations = pd.DataFrame(iteration_rows)
        observed_pipelines = iterations[
            ["candidate_id", "scenario", "round", "iteration"]
        ].drop_duplicates()
        if len(observed_pipelines) != int(benchmark["expected_timed_pipeline_iterations"]):
            raise RuntimeError("P7 center60 timed pipeline count 漂移")
        latency_summary = p7.summarize_stage_latencies(iterations, contract)
        memory_summary = p7.summarize_memory(pd.DataFrame(memory_rows))

        # Profiler 后置，避免污染正式时延和显存测量。
        for candidate_id in MODEL_ORDER:
            source = source_by_id[candidate_id]
            model, _ = _load_model(root, source, target_device)
            flops, macs, coverage = _profile_model(
                candidate_id=candidate_id,
                model=model,
                waveform_host=waveform_host,
                cached_w_host=cached_w_host,
                device=target_device,
            )
            resources[candidate_id].update(
                {
                    "profiler_scope": contract["profiler"]["scope"],
                    "profiler_covered_flops": flops,
                    "profiler_covered_macs": macs,
                    "profiler_coverage": coverage,
                }
            )
            del model
            torch.cuda.empty_cache()
        resource_frame = pd.DataFrame([resources[value] for value in MODEL_ORDER])
        quality = build_quality_efficiency_table(
            contract, latency_summary, resource_frame, repo_root=root
        )
        context_table = build_descriptive_efficiency_context_table(
            contract, quality, repo_root=root
        )

        environment = p7._environment(target_device, contract, input_identity)
        environment["protocol_id"] = PROTOCOL_ID
        environment["platform_scope"] = (
            "auxiliary_center60_desktop_gpu_measurement_not_edge_device_measurement"
        )
        environment["benchmark"] = dict(benchmark)
        environment["execution"] = {
            "git_commit": commit,
            "git_dirty": False,
            "command": command,
        }
        environment["contract_sha256"] = CONFIG_SHA256
        (temporary / "environment.json").write_text(
            json.dumps(environment, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        float_format = contract["output"]["csv_float_format"]
        frames = {
            "input_materialization.csv": input_io,
            "stage_latency_iterations.csv": iterations,
            "stage_latency_summary.csv": latency_summary,
            "memory_summary.csv": memory_summary,
            "model_resource_summary.csv": resource_frame,
            "quality_efficiency_table.csv": quality,
            "descriptive_efficiency_context_table.csv": context_table,
        }
        for filename, frame in frames.items():
            frame.to_csv(temporary / filename, index=False, float_format=float_format)
        generated = sorted(path for path in temporary.iterdir() if path.is_file())
        manifest = {
            "protocol_id": PROTOCOL_ID,
            "phase": "p7_center60_gpu_efficiency",
            "status": "complete",
            "decision": "measurement_complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "contract_sha256": CONFIG_SHA256,
            "descriptive_reference_manifest_sha256": reference["manifest_sha256"],
            "git_commit": commit,
            "git_dirty": False,
            "matrix": {
                "models": list(MODEL_ORDER),
                "scenarios": {key: list(value) for key, value in SCENARIOS.items()},
                "checkpoint_seed": 20260811,
                "input_seconds": 60,
                "output_seconds": 60,
                "warmup_iterations": 20,
                "timed_iterations": 100,
                "rounds": 5,
                "expected_timed_pipeline_iterations": 1500,
            },
            "access": {
                "validation_bcg_read": True,
                "validation_w_cache_read": True,
                "checkpoint_read": True,
                "research_test_metric_summary_read": True,
                "research_test_signal_or_target_read": False,
                "training_used": False,
                "cache_build_used": False,
                "gpu_benchmark_used": True,
                "cpu_model_benchmark_used": False,
            },
            "interpretation": (
                "separate_60_to_60_descriptive_measurement_no_cross_task_causal_window_claim"
            ),
            "files": {
                path.name: {
                    "sha256": p7.sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in generated
            },
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        observed = tuple(sorted(path.name for path in temporary.iterdir() if path.is_file()))
        if observed != tuple(sorted(REQUIRED_OUTPUTS)):
            raise RuntimeError(f"P7 center60 output schema 不完整: {observed}")
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output / "manifest.json"
