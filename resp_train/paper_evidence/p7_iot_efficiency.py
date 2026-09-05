from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import tempfile
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from resp_train.crd.config import crd_dependency_versions, load_crd_config
from resp_train.crd.tf_v1_data import TfV1CacheReader
from resp_train.crd.tf_v1_features import cwt_magnitude_features
from resp_train.crd.tf_w_v2 import FULL_6V_INDICES, SOURCE_CACHE_ROOT
from resp_train.data.cache import WholeNightCache
from resp_train.data.index import filter_index
from resp_train.data.research_v2 import read_research_v2_index
from resp_train.protocols.respiration import canonicalize_torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path("configs/paper_evidence_v1/p7_iot_efficiency_v1.json")
CONFIG_SHA256 = "8da08b682b2a7381fe81e0dc316295187b390ec29cd98ef1bb3c84aefd682b28"
PROTOCOL_ID = "paper-p7-iot-efficiency-v1-20260905"
OUTPUT_ROOT = Path("runs/paper_evidence_v1/p7_iot_efficiency")
MODEL_ORDER = ("W0_FULL_12V_FILM_D6", "W3_FULL_6V_FILM_D6", "D4_W0")
SCENARIO_ORDER = ("cached_w", "online_w")
PRIMARY_METRICS = (
    "whole_rr_abs_error_bpm",
    "local_rr_mae_bpm",
    "envelope_trajectory_mae",
    "global_envelope_modulation_error",
    "lag_aware_signed_pcc",
)
TEST_METRIC_NAMES = {
    "whole_rr_abs_error_bpm": "whole_rr_mae",
    "local_rr_mae_bpm": "local_rr_mae",
    "envelope_trajectory_mae": "trajectory_mae",
    "global_envelope_modulation_error": "global_envelope_error",
    "lag_aware_signed_pcc": "signed_pcc",
}
REQUIRED_OUTPUTS = (
    "environment.json",
    "input_materialization.csv",
    "stage_latency_iterations.csv",
    "stage_latency_summary.csv",
    "memory_summary.csv",
    "model_resource_summary.csv",
    "quality_efficiency_table.csv",
    "manifest.json",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    path = root / CONFIG_PATH
    if sha256_file(path) != CONFIG_SHA256:
        raise RuntimeError("P7 config SHA-256 漂移")
    contract = json.loads(path.read_text(encoding="utf-8"))
    validate_contract(contract)
    return contract


def validate_contract(contract: Mapping[str, Any]) -> None:
    benchmark = contract.get("benchmark", {})
    authorization = contract.get("authorization", {})
    platform_contract = contract.get("platform", {})
    models = contract.get("models", ())
    if contract.get("schema_version") != "paper-p7-iot-efficiency-config-v1":
        raise ValueError("P7 config schema 漂移")
    if contract.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("P7 protocol identity 漂移")
    if authorization != {
        "gpu_benchmark": True,
        "cpu_model_benchmark": False,
        "training": False,
        "cache_build": False,
        "research_test_access": False,
    }:
        raise ValueError("P7 authorization 漂移")
    if (
        platform_contract.get("kind") != "desktop_gpu"
        or platform_contract.get("device") != "cuda:0"
        or int(platform_contract.get("batch_size", -1)) != 1
        or platform_contract.get("host_memory") != "pageable"
        or platform_contract.get("amp_dtype") != "bfloat16"
    ):
        raise ValueError("P7 GPU platform contract 漂移")
    if (
        int(benchmark.get("warmup_iterations", -1)) != 20
        or int(benchmark.get("timed_iterations", -1)) != 100
        or int(benchmark.get("rounds", -1)) != 5
        or tuple(benchmark.get("scenarios", ())) != SCENARIO_ORDER
        or benchmark.get("model_forward_stage") != "model_forward_including_w_view"
    ):
        raise ValueError("P7 benchmark matrix 漂移")
    if tuple(item.get("candidate_id") for item in models) != MODEL_ORDER:
        raise ValueError("P7 model order/matrix 漂移")
    if any(int(item.get("seed", -1)) != 20260811 for item in models):
        raise ValueError("P7 只允许 seed=20260811 checkpoints")
    if contract.get("input", {}).get("split") != "val":
        raise ValueError("P7 只允许 validation BCG input")
    if int(contract.get("input", {}).get("dataset_row_id", -1)) != 12429:
        raise ValueError("P7 benchmark row 漂移")
    w_source = contract.get("w_source", {})
    if (
        w_source.get("online_extractor_device") != "host_cpu"
        or int(w_source.get("online_scale_count", -1)) != 97
        or w_source.get("w3_view") != "model-internal input[:,::2,:]"
    ):
        raise ValueError("P7 W source/extractor/view contract 漂移")
    if contract.get("output", {}).get("directory") != str(OUTPUT_ROOT):
        raise ValueError("P7 output identity 漂移")


def summarize_stage_latencies(iterations: pd.DataFrame, contract: Mapping[str, Any]) -> pd.DataFrame:
    required = {"candidate_id", "scenario", "round", "iteration", "stage", "latency_ms"}
    missing = sorted(required - set(iterations.columns))
    if missing or iterations.empty:
        raise ValueError(f"P7 latency iterations 不合格: missing={missing}, rows={len(iterations)}")
    if not np.isfinite(iterations["latency_ms"].to_numpy(dtype=np.float64)).all():
        raise FloatingPointError("P7 latency 包含 NaN/Inf")
    if (iterations["latency_ms"] < 0).any():
        raise ValueError("P7 latency 不能为负")
    expected_n = int(contract["benchmark"]["rounds"]) * int(contract["benchmark"]["timed_iterations"])
    rows: list[dict[str, Any]] = []
    for (candidate_id, scenario, stage), frame in iterations.groupby(
        ["candidate_id", "scenario", "stage"], sort=False
    ):
        values = frame["latency_ms"].to_numpy(dtype=np.float64)
        if values.size != expected_n:
            raise RuntimeError(
                f"P7 latency count 漂移: {candidate_id}/{scenario}/{stage}={values.size} != {expected_n}"
            )
        row = {
            "candidate_id": candidate_id,
            "scenario": scenario,
            "stage": stage,
            "n": int(values.size),
            "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95, method="linear")),
            "mean_ms": float(np.mean(values)),
            "sample_sd_ms": float(np.std(values, ddof=1)),
            "min_ms": float(np.min(values)),
            "max_ms": float(np.max(values)),
            "sliding_step_seconds": math.nan,
            "p95_slack_ms": math.nan,
            "p95_updates_per_second": math.nan,
            "context_wait_seconds": int(contract["benchmark"]["context_wait_seconds"]),
        }
        if stage == "end_to_end":
            step_ms = 1000.0 * float(contract["benchmark"]["sliding_step_seconds"])
            row["sliding_step_seconds"] = float(contract["benchmark"]["sliding_step_seconds"])
            row["p95_slack_ms"] = step_ms - row["p95_ms"]
            row["p95_updates_per_second"] = 1000.0 / row["p95_ms"] if row["p95_ms"] > 0 else math.inf
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_memory(round_rows: pd.DataFrame) -> pd.DataFrame:
    required = {
        "candidate_id", "scenario", "round", "baseline_allocated_bytes",
        "baseline_reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes",
    }
    missing = sorted(required - set(round_rows.columns))
    if missing or round_rows.empty:
        raise ValueError(f"P7 memory rows 不合格: missing={missing}, rows={len(round_rows)}")
    output = []
    for (candidate_id, scenario), frame in round_rows.groupby(["candidate_id", "scenario"], sort=False):
        row: dict[str, Any] = {
            "candidate_id": candidate_id,
            "scenario": scenario,
            "round_count": int(len(frame)),
        }
        for column in (
            "baseline_allocated_bytes", "baseline_reserved_bytes",
            "peak_allocated_bytes", "peak_reserved_bytes",
        ):
            values = frame[column].to_numpy(dtype=np.float64) / (1024.0**2)
            row[f"{column.removesuffix('_bytes')}_mean_mib"] = float(np.mean(values))
            row[f"{column.removesuffix('_bytes')}_sample_sd_mib"] = float(np.std(values, ddof=1))
            row[f"{column.removesuffix('_bytes')}_min_mib"] = float(np.min(values))
            row[f"{column.removesuffix('_bytes')}_max_mib"] = float(np.max(values))
        output.append(row)
    return pd.DataFrame(output)


def build_quality_efficiency_table(
    contract: Mapping[str, Any], latency_summary: pd.DataFrame, resources: pd.DataFrame, *, repo_root: str | Path
) -> pd.DataFrame:
    root = Path(repo_root).resolve()
    sources = contract["quality_sources"]
    for source in sources.values():
        if sha256_file(root / source["path"]) != source["sha256"]:
            raise RuntimeError(f"P7 quality source hash 漂移: {source['path']}")
        if "manifest_path" in source and (
            sha256_file(root / source["manifest_path"]) != source["manifest_sha256"]
        ):
            raise RuntimeError(f"P7 quality manifest hash 漂移: {source['manifest_path']}")
    test = pd.read_csv(root / sources["w0_w3"]["path"])
    validation = pd.read_csv(root / sources["d4"]["path"])
    variant_by_id = {item["candidate_id"]: item["variant"] for item in contract["models"]}
    quality_rows = []
    for candidate_id in MODEL_ORDER:
        variant = variant_by_id[candidate_id]
        values: dict[str, Any] = {"candidate_id": candidate_id, "model_variant": variant}
        if candidate_id == "D4_W0":
            row = validation.loc[validation["candidate_id"] == candidate_id]
            if len(row) != 1:
                raise RuntimeError("P7 D4 validation quality source 不唯一")
            row = row.iloc[0]
            values["quality_scope"] = sources["d4"]["scope"]
            for metric in PRIMARY_METRICS:
                values[f"{metric}_mean"] = float(row[f"{metric}_mean_seed_mean"])
                values[f"{metric}_sample_sd"] = float(row[f"{metric}_mean_sample_sd"])
        else:
            rows = test.loc[test["variant"] == variant]
            if set(rows["metric"]) != set(TEST_METRIC_NAMES.values()) or len(rows) != len(PRIMARY_METRICS):
                raise RuntimeError(f"P7 {candidate_id} test quality matrix 不完整")
            values["quality_scope"] = sources["w0_w3"]["scope"]
            by_metric = rows.set_index("metric")
            for metric in PRIMARY_METRICS:
                source_metric = TEST_METRIC_NAMES[metric]
                values[f"{metric}_mean"] = float(by_metric.loc[source_metric, "mean"])
                values[f"{metric}_sample_sd"] = float(by_metric.loc[source_metric, "sample_sd"])
        quality_rows.append(values)
    quality = pd.DataFrame(quality_rows)
    end_to_end = latency_summary.loc[latency_summary["stage"] == "end_to_end"].copy()
    selected = end_to_end[[
        "candidate_id", "scenario", "median_ms", "p95_ms", "mean_ms", "sample_sd_ms",
        "min_ms", "max_ms", "p95_slack_ms", "p95_updates_per_second", "context_wait_seconds",
    ]]
    output = selected.merge(resources, on="candidate_id", validate="many_to_one")
    output = output.merge(quality, on=["candidate_id", "model_variant"], validate="many_to_one")
    return output


def _git_identity(root: Path) -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=False, capture_output=True, text=True
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, check=False, capture_output=True, text=True
    )
    if commit.returncode != 0 or status.returncode != 0:
        raise RuntimeError("P7 无法读取 Git identity")
    if status.stdout.strip():
        raise RuntimeError("P7 benchmark 必须从干净 Git commit 启动")
    return commit.stdout.strip()


def _verify_file(root: Path, relative: str, expected_sha: str, expected_size: int | None = None) -> Path:
    path = root / relative
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise RuntimeError(f"P7 source artifact 漂移: {relative}")
    if expected_size is not None and path.stat().st_size != int(expected_size):
        raise RuntimeError(f"P7 source artifact size 漂移: {relative}")
    return path


def _load_inputs(root: Path, contract: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, pd.DataFrame, dict[str, Any]]:
    input_contract = contract["input"]
    _verify_file(root, input_contract["selection_manifest"], input_contract["selection_manifest_sha256"])
    selected_path = _verify_file(root, input_contract["selection_rows"], input_contract["selection_rows_sha256"])
    selected = pd.read_csv(selected_path)
    selected = selected.loc[selected["dataset_row_id"] == int(input_contract["dataset_row_id"])]
    if len(selected) != 1 or selected.iloc[0]["category"] != input_contract["selection_category"]:
        raise RuntimeError("P7 predeclared validation row/category 漂移")

    first = contract["models"][0]
    config_path = _verify_file(root, f"{first['run_dir']}/config.yaml", first["config_sha256"])
    cfg = load_crd_config(config_path, overrides=["training.device=cuda:0", "training.show_progress=false"])
    audited = read_research_v2_index(cfg.data.dataset_root, cfg.data.index_csv, cfg)
    rows = filter_index(
        audited, cfg, split="val", max_windows=None,
        sample_strategy="stratified_random", sample_seed=int(cfg.data.val_sample_seed),
    )
    rows = rows.loc[rows["dataset_row_id"] == int(input_contract["dataset_row_id"])]
    if len(rows) != 1:
        raise RuntimeError("P7 validation row identity/coverage 漂移")
    row = rows.iloc[0]
    start, stop = int(row["window_start_sample"]), int(row["window_end_sample"])
    if stop - start != int(input_contract["duration_samples"]):
        raise RuntimeError("P7 input window length 漂移")

    io_rows = []
    source_cache = WholeNightCache(Path(str(cfg.data.dataset_root)) / str(cfg.data.index_csv))
    begin = time.perf_counter_ns()
    source = source_cache.get_arrays(str(row["source_npz"]), [str(row["bcg_signal_key"])])
    waveform_np = np.asarray(source[str(row["bcg_signal_key"])][start:stop], dtype=np.float32).copy()
    io_rows.append({
        "source": "validation_bcg_npz", "dataset_row_id": int(row["dataset_row_id"]),
        "latency_ms": (time.perf_counter_ns() - begin) / 1e6,
        "measurement_count": 1, "os_cache_state": "uncontrolled_process_initial",
    })
    if waveform_np.shape != (18000,) or not np.isfinite(waveform_np).all():
        raise FloatingPointError("P7 BCG input shape/finite 不合格")

    if (root / contract["w_source"]["cache_root"]).resolve() != SOURCE_CACHE_ROOT.resolve():
        raise RuntimeError("P7 W cache root 漂移")
    begin = time.perf_counter_ns()
    cache = TfV1CacheReader(SOURCE_CACHE_ROOT, split="val", representations=("w",))
    cached_w = cache.get(int(row["dataset_row_id"]))["w"].contiguous()
    io_rows.append({
        "source": "frozen_validation_w_cache", "dataset_row_id": int(row["dataset_row_id"]),
        "latency_ms": (time.perf_counter_ns() - begin) / 1e6,
        "measurement_count": 1, "os_cache_state": "uncontrolled_process_initial",
    })
    if tuple(cached_w.shape) != (97, 360) or not torch.isfinite(cached_w).all():
        raise FloatingPointError("P7 cached W shape/finite 不合格")

    online_np, frequencies = cwt_magnitude_features(waveform_np)
    delta = float(np.max(np.abs(online_np - cached_w.numpy())))
    if delta > float(contract["w_source"]["cached_online_max_abs_tolerance"]):
        raise RuntimeError(f"P7 cached/online W 不一致: max_abs_delta={delta:.17g}")
    input_identity = {
        "dataset_row_id": int(row["dataset_row_id"]),
        "split": "val",
        "samp_id": int(row["samp_id"]),
        "waveform_sha256": hashlib.sha256(waveform_np.tobytes(order="C")).hexdigest(),
        "cached_w_sha256": hashlib.sha256(cached_w.numpy().tobytes(order="C")).hexdigest(),
        "cached_online_w_max_abs_delta": delta,
        "online_frequency_min_hz": float(frequencies[0]),
        "online_frequency_max_hz": float(frequencies[-1]),
    }
    waveform = torch.from_numpy(waveform_np).view(1, 1, -1).contiguous()
    return waveform, cached_w.view(1, 97, 360), pd.DataFrame(io_rows), input_identity


def _load_model(root: Path, source: Mapping[str, Any], device: torch.device) -> tuple[torch.nn.Module, Any]:
    from resp_train.crd.experiment import _validate_checkpoint_config
    from resp_train.crd.model import build_crd_model

    config_path = _verify_file(root, f"{source['run_dir']}/config.yaml", source["config_sha256"])
    checkpoint_path = _verify_file(
        root, f"{source['run_dir']}/checkpoint_best_local_rr.pt",
        source["checkpoint_sha256"], int(source["checkpoint_size_bytes"]),
    )
    cfg = load_crd_config(config_path, overrides=[f"training.device={device}", "training.show_progress=false"])
    if str(cfg.model.variant) != source["variant"] or int(cfg.training.seed) != int(source["seed"]):
        raise RuntimeError(f"P7 model config identity 漂移: {source['candidate_id']}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("epoch", -1)) != int(source["selected_epoch"]):
        raise RuntimeError(f"P7 selected epoch 漂移: {source['candidate_id']}")
    _validate_checkpoint_config(checkpoint.get("config"), cfg)
    model = build_crd_model(cfg)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint
    model.eval().to(device)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable != int(source["trainable_parameters"]):
        raise RuntimeError(f"P7 trainable parameter count 漂移: {source['candidate_id']}")
    return model, cfg


def _amp_context(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def _timed_iteration(
    *, model: torch.nn.Module, cfg: Any, waveform_host: torch.Tensor, cached_w_host: torch.Tensor,
    scenario: str, device: torch.device,
) -> tuple[dict[str, float], torch.Tensor]:
    start_total = time.perf_counter_ns()
    stages: dict[str, float] = {}
    if scenario == "online_w":
        begin = time.perf_counter_ns()
        online_w, _ = cwt_magnitude_features(waveform_host.numpy().reshape(-1))
        w_host = torch.from_numpy(online_w).view(1, 97, 360).contiguous()
        stages["w_feature_extraction_97scale_log_pool"] = (time.perf_counter_ns() - begin) / 1e6
    elif scenario == "cached_w":
        w_host = cached_w_host
    else:
        raise ValueError(f"未知 P7 scenario={scenario!r}")

    begin = time.perf_counter_ns()
    waveform = waveform_host.to(device, non_blocking=False)
    w = w_host.to(device, non_blocking=False)
    torch.cuda.synchronize(device)
    stages["host_to_device_pageable_bcg_and_full97_w"] = (time.perf_counter_ns() - begin) / 1e6

    begin = time.perf_counter_ns()
    with torch.inference_mode(), _amp_context(device):
        output = model(waveform, tf={"w": w})
    predicted = output["waveform"]
    if tuple(predicted.shape) != (1, 1, 18000):
        raise RuntimeError("P7 model output shape 不合格")
    torch.cuda.synchronize(device)
    stages["model_forward_including_w_view"] = (time.perf_counter_ns() - begin) / 1e6

    begin = time.perf_counter_ns()
    _, canonical = canonicalize_torch(
        predicted, fs=float(cfg.window.target_fs), low_hz=float(cfg.loss.band_low_hz),
        high_hz=float(cfg.loss.band_high_hz), scale_eps=float(cfg.loss.scale_eps),
    )
    torch.cuda.synchronize(device)
    stages["Pi_180"] = (time.perf_counter_ns() - begin) / 1e6
    stages["end_to_end"] = (time.perf_counter_ns() - start_total) / 1e6
    return stages, canonical


def _profile_model(
    model: torch.nn.Module, cfg: Any, waveform_host: torch.Tensor, cached_w_host: torch.Tensor,
    device: torch.device,
) -> tuple[float, float, str]:
    waveform = waveform_host.to(device)
    w = cached_w_host.to(device)
    try:
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=activities, record_shapes=True, with_flops=True) as profile:
            with torch.inference_mode(), _amp_context(device):
                output = model(waveform, tf={"w": w})
            torch.cuda.synchronize(device)
        covered_flops = float(sum(float(event.flops or 0) for event in profile.key_averages()))
        if not math.isfinite(covered_flops) or covered_flops <= 0:
            return math.nan, math.nan, "unavailable_no_positive_covered_flops"
        return covered_flops, covered_flops / 2.0, "partial_standard_ops_only_mamba_custom_ops_may_be_uncovered"
    except Exception as error:  # profiler coverage is descriptive and may vary with the installed PyTorch build.
        return math.nan, math.nan, f"unavailable_{type(error).__name__}"


def _environment(device: torch.device, contract: Mapping[str, Any], input_identity: Mapping[str, Any]) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    packages = {}
    for name in ("numpy", "pandas", "torch", "ssqueezepy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "protocol_id": PROTOCOL_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform_scope": "desktop_gpu_measurement_not_edge_device_measurement",
        "cpu_model_benchmark_used": False,
        "system": {
            "platform": platform.platform(), "python": platform.python_version(),
            "cpu": platform.processor(), "logical_cpu_count": os.cpu_count(),
            "torch_num_threads": torch.get_num_threads(),
        },
        "gpu": {
            "logical_device": str(device), "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": [int(properties.major), int(properties.minor)],
            "cuda_runtime": torch.version.cuda,
            "cudnn_runtime": torch.backends.cudnn.version(),
        },
        "numeric": {
            "amp_dtype": contract["platform"]["amp_dtype"],
            "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "host_memory": contract["platform"]["host_memory"],
        },
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "LD_LIBRARY_PATH_set": "LD_LIBRARY_PATH" in os.environ,
            "LD_PRELOAD_set": "LD_PRELOAD" in os.environ,
            "SSQ_GPU": os.environ.get("SSQ_GPU", "0"),
            "SSQ_PARALLEL": os.environ.get("SSQ_PARALLEL", "1"),
        },
        "packages": packages,
        "crd_dependencies": crd_dependency_versions(),
        "benchmark": dict(contract["benchmark"]),
        "input_identity": dict(input_identity),
    }


def run_p7_gpu_benchmark(
    *, repo_root: str | Path = REPO_ROOT, command: str, device: str, confirm: bool,
) -> Path:
    root = Path(repo_root).resolve()
    contract = load_contract(root)
    if not confirm:
        raise RuntimeError("P7 GPU benchmark 需要显式确认")
    if device != contract["platform"]["device"] or not torch.cuda.is_available():
        raise RuntimeError("P7 必须使用合同固定且可用的 cuda:0")
    if any(name in os.environ for name in contract["platform"]["required_unset_environment"]):
        raise RuntimeError("P7 要求 unset LD_LIBRARY_PATH 与 LD_PRELOAD")
    if os.environ.get("SSQ_GPU", "0") != "0":
        raise RuntimeError("P7 online-W 合同要求 ssqueezepy 在 host CPU 执行（SSQ_GPU=0）")
    output = root / OUTPUT_ROOT
    if output.exists():
        raise FileExistsError(f"P7 output 已存在，拒绝覆盖: {output}")
    commit = _git_identity(root)
    torch.backends.cuda.matmul.allow_tf32 = bool(contract["platform"]["allow_tf32"])
    torch.backends.cudnn.benchmark = bool(contract["platform"]["cudnn_benchmark"])
    target_device = torch.device(device)
    waveform_host, cached_w_host, input_io, input_identity = _load_inputs(root, contract)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".p7_iot_efficiency_", dir=output.parent))
    iteration_rows: list[dict[str, Any]] = []
    memory_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    last_output: torch.Tensor | None = None
    try:
        for source in contract["models"]:
            model, cfg = _load_model(root, source, target_device)
            flops, macs, coverage = _profile_model(model, cfg, waveform_host, cached_w_host, target_device)
            resource_rows.append({
                "candidate_id": source["candidate_id"], "model_variant": source["variant"],
                "checkpoint_seed": int(source["seed"]),
                "trainable_parameters": int(source["trainable_parameters"]),
                "total_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
                "model_input_elements": int(source["model_input_elements"]),
                "model_w_scales": int(source["model_w_scales"]),
                "full97_source_transfer_elements": 97 * 360,
                "checkpoint_size_bytes": int(source["checkpoint_size_bytes"]),
                "checkpoint_sha256": source["checkpoint_sha256"],
                "config_sha256": source["config_sha256"],
                "profiler_scope": contract["profiler"]["scope"],
                "profiler_covered_flops": flops,
                "profiler_covered_macs": macs,
                "profiler_coverage": coverage,
            })
            for scenario in SCENARIO_ORDER:
                for round_index in range(int(contract["benchmark"]["rounds"])):
                    for _ in range(int(contract["benchmark"]["warmup_iterations"])):
                        _, last_output = _timed_iteration(
                            model=model, cfg=cfg, waveform_host=waveform_host,
                            cached_w_host=cached_w_host, scenario=scenario, device=target_device,
                        )
                    torch.cuda.synchronize(target_device)
                    torch.cuda.reset_peak_memory_stats(target_device)
                    baseline_allocated = int(torch.cuda.memory_allocated(target_device))
                    baseline_reserved = int(torch.cuda.memory_reserved(target_device))
                    for iteration in range(int(contract["benchmark"]["timed_iterations"])):
                        stages, last_output = _timed_iteration(
                            model=model, cfg=cfg, waveform_host=waveform_host,
                            cached_w_host=cached_w_host, scenario=scenario, device=target_device,
                        )
                        for stage, latency_ms in stages.items():
                            iteration_rows.append({
                                "candidate_id": source["candidate_id"], "model_variant": source["variant"],
                                "checkpoint_seed": int(source["seed"]), "scenario": scenario,
                                "round": round_index + 1, "iteration": iteration + 1,
                                "stage": stage, "latency_ms": float(latency_ms),
                            })
                    memory_rows.append({
                        "candidate_id": source["candidate_id"], "scenario": scenario,
                        "round": round_index + 1,
                        "baseline_allocated_bytes": baseline_allocated,
                        "baseline_reserved_bytes": baseline_reserved,
                        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(target_device)),
                        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(target_device)),
                    })
                if last_output is None or not torch.isfinite(last_output).all():
                    raise FloatingPointError(
                        f"P7 {source['candidate_id']}/{scenario} Pi_180 output 包含 NaN/Inf"
                    )
            del model, cfg
            torch.cuda.empty_cache()
        if last_output is None or not torch.isfinite(last_output).all():
            raise RuntimeError("P7 未产生 finite output")
        iterations = pd.DataFrame(iteration_rows)
        latency_summary = summarize_stage_latencies(iterations, contract)
        memory_summary = summarize_memory(pd.DataFrame(memory_rows))
        resources = pd.DataFrame(resource_rows)
        quality = build_quality_efficiency_table(contract, latency_summary, resources, repo_root=root)
        environment = _environment(target_device, contract, input_identity)
        environment["execution"] = {"git_commit": commit, "git_dirty": False, "command": command}
        environment["contract_sha256"] = CONFIG_SHA256
        (temporary / "environment.json").write_text(
            json.dumps(environment, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        float_format = contract["output"]["csv_float_format"]
        input_io.to_csv(temporary / "input_materialization.csv", index=False, float_format=float_format)
        iterations.to_csv(temporary / "stage_latency_iterations.csv", index=False, float_format=float_format)
        latency_summary.to_csv(temporary / "stage_latency_summary.csv", index=False, float_format=float_format)
        memory_summary.to_csv(temporary / "memory_summary.csv", index=False, float_format=float_format)
        resources.to_csv(temporary / "model_resource_summary.csv", index=False, float_format=float_format)
        quality.to_csv(temporary / "quality_efficiency_table.csv", index=False, float_format=float_format)
        generated = sorted(path for path in temporary.iterdir() if path.is_file())
        manifest = {
            "protocol_id": PROTOCOL_ID,
            "phase": "p7_gpu_iot_efficiency_measurement",
            "status": "complete",
            "decision": "measurement_complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "contract_sha256": CONFIG_SHA256,
            "git_commit": commit,
            "git_dirty": False,
            "matrix": {
                "models": list(MODEL_ORDER), "scenarios": list(SCENARIO_ORDER),
                "checkpoint_seed": 20260811,
                "warmup_iterations": 20, "timed_iterations": 100, "rounds": 5,
                "expected_timed_pipeline_iterations": 3 * 2 * 5 * 100,
            },
            "access": {
                "validation_bcg_read": True, "validation_w_cache_read": True,
                "checkpoint_read": True, "research_test_metric_summary_read": True,
                "research_test_signal_or_target_read": False, "training_used": False,
                "cache_build_used": False, "gpu_benchmark_used": True,
                "cpu_model_benchmark_used": False,
            },
            "interpretation": "descriptive_measurement_only_no_deployment_verdict",
            "files": {
                path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
                for path in generated
            },
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        observed = tuple(sorted(path.name for path in temporary.iterdir() if path.is_file()))
        if observed != tuple(sorted(REQUIRED_OUTPUTS)):
            raise RuntimeError(f"P7 output schema 不完整: {observed}")
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output / "manifest.json"
