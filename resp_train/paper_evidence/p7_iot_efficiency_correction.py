from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from resp_train.paper_evidence import p7_iot_efficiency as p7


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path("configs/paper_evidence_v1/p7_iot_efficiency_correction_v2.json")
CONFIG_SHA256 = "94d23ed74bdf661c7507845ca7ab7af1da3decd63a3824da957a75868bfc5529"
PROTOCOL_ID = "paper-p7-iot-efficiency-correction-v2-20260905"
OUTPUT_ROOT = Path("runs/paper_evidence_v1/p7_iot_efficiency_correction_v2")
REQUIRED_OUTPUTS = (
    "environment.json",
    "input_materialization.csv",
    "stage_latency_iterations.csv",
    "stage_latency_summary.csv",
    "online_cwt_consistency.csv",
    "memory_summary.csv",
    "model_resource_summary.csv",
    "quality_efficiency_table.csv",
    "manifest.json",
)


def load_correction_contract(repo_root: str | Path = REPO_ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(repo_root).resolve()
    path = root / CONFIG_PATH
    if p7.sha256_file(path) != CONFIG_SHA256:
        raise RuntimeError("P7-v2 correction config SHA-256 漂移")
    correction = json.loads(path.read_text(encoding="utf-8"))
    base = p7.load_contract(root)
    validate_correction_contract(correction, base)
    return correction, base


def validate_correction_contract(correction: Mapping[str, Any], base: Mapping[str, Any]) -> None:
    if correction.get("schema_version") != "paper-p7-iot-efficiency-correction-config-v2":
        raise ValueError("P7-v2 correction schema 漂移")
    if correction.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("P7-v2 protocol identity 漂移")
    if correction.get("parent_contract") != {
        "path": str(p7.CONFIG_PATH),
        "sha256": p7.CONFIG_SHA256,
    }:
        raise ValueError("P7-v2 parent contract 漂移")
    if correction.get("authorization") != base.get("authorization"):
        raise ValueError("P7-v2 authorization 必须与 GPU-only parent 一致")
    benchmark = correction.get("benchmark", {})
    if (
        int(benchmark.get("warmup_iterations", -1)) != 20
        or int(benchmark.get("timed_iterations", -1)) != 100
        or int(benchmark.get("rounds", -1)) != 5
        or tuple(benchmark.get("scenarios", ())) != p7.SCENARIO_ORDER
        or benchmark.get("profiler_position") != "after_all_latency_and_memory_measurements"
        or benchmark.get("fresh_model_load_per_candidate_round") is not True
        or benchmark.get("cuda_empty_cache_before_each_scenario") is not True
    ):
        raise ValueError("P7-v2 benchmark contract 漂移")
    model_orders = benchmark.get("round_model_order", ())
    scenario_orders = benchmark.get("round_scenario_order", ())
    if len(model_orders) != 5 or any(tuple(order) == () or set(order) != set(p7.MODEL_ORDER) for order in model_orders):
        raise ValueError("P7-v2 round model schedule 不完整")
    if len(scenario_orders) != 5 or any(set(order) != set(p7.SCENARIO_ORDER) for order in scenario_orders):
        raise ValueError("P7-v2 round scenario schedule 不完整")
    for position in range(3):
        counts = {candidate: sum(order[position] == candidate for order in model_orders) for candidate in p7.MODEL_ORDER}
        if max(counts.values()) - min(counts.values()) > 1:
            raise ValueError("P7-v2 model position schedule 不平衡")
    audit = correction.get("online_cwt_consistency", {})
    if (
        audit.get("stage") != "w_feature_extraction_97scale_log_pool"
        or tuple(audit.get("statistics", ())) != ("median_ms", "p95_ms")
        or float(audit.get("maximum_relative_spread", -1)) != 0.1
        or audit.get("required_for_measurement_complete") is not True
    ):
        raise ValueError("P7-v2 online CWT consistency contract 漂移")
    if correction.get("output", {}).get("directory") != str(OUTPUT_ROOT):
        raise ValueError("P7-v2 output identity 漂移")


def build_online_cwt_consistency(
    latency_summary: pd.DataFrame, correction: Mapping[str, Any]
) -> tuple[pd.DataFrame, bool]:
    audit = correction["online_cwt_consistency"]
    rows = latency_summary.loc[
        (latency_summary["scenario"] == "online_w")
        & (latency_summary["stage"] == audit["stage"])
    ].copy()
    if set(rows["candidate_id"]) != set(p7.MODEL_ORDER) or len(rows) != len(p7.MODEL_ORDER):
        raise RuntimeError("P7-v2 online CWT summary matrix 不完整")
    maximum = float(audit["maximum_relative_spread"])
    spreads = {}
    for statistic in audit["statistics"]:
        values = rows[statistic].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or np.min(values) <= 0:
            raise FloatingPointError(f"P7-v2 CWT {statistic} 不合格")
        spreads[statistic] = float((np.max(values) - np.min(values)) / np.min(values))
    passed = all(value <= maximum for value in spreads.values())
    rows = rows[[
        "candidate_id", "n", "median_ms", "p95_ms", "mean_ms", "sample_sd_ms", "min_ms", "max_ms"
    ]].copy()
    rows["median_relative_spread_across_models"] = spreads["median_ms"]
    rows["p95_relative_spread_across_models"] = spreads["p95_ms"]
    rows["maximum_allowed_relative_spread"] = maximum
    rows["cross_model_consistency_passed"] = passed
    return rows, passed


def _resource_record(source: Mapping[str, Any], model: torch.nn.Module) -> dict[str, Any]:
    return {
        "candidate_id": source["candidate_id"],
        "model_variant": source["variant"],
        "checkpoint_seed": int(source["seed"]),
        "trainable_parameters": int(source["trainable_parameters"]),
        "total_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "model_input_elements": int(source["model_input_elements"]),
        "model_w_scales": int(source["model_w_scales"]),
        "full97_source_transfer_elements": 97 * 360,
        "checkpoint_size_bytes": int(source["checkpoint_size_bytes"]),
        "checkpoint_sha256": source["checkpoint_sha256"],
        "config_sha256": source["config_sha256"],
    }


def run_p7_correction_gpu_benchmark(
    *, repo_root: str | Path = REPO_ROOT, command: str, device: str, confirm: bool,
) -> Path:
    root = Path(repo_root).resolve()
    correction, base = load_correction_contract(root)
    if not confirm:
        raise RuntimeError("P7-v2 GPU benchmark 需要显式确认")
    if device != base["platform"]["device"] or not torch.cuda.is_available():
        raise RuntimeError("P7-v2 必须使用合同固定且可用的 cuda:0")
    if any(name in os.environ for name in base["platform"]["required_unset_environment"]):
        raise RuntimeError("P7-v2 要求 unset LD_LIBRARY_PATH 与 LD_PRELOAD")
    if os.environ.get("SSQ_GPU", "0") != "0":
        raise RuntimeError("P7-v2 online-W 要求 SSQ_GPU=0")
    prior = correction["prior_measurement"]
    prior_path = root / prior["path"]
    if p7.sha256_file(prior_path) != prior["sha256"]:
        raise RuntimeError("P7-v2 prior measurement manifest 漂移")
    prior_manifest = json.loads(prior_path.read_text(encoding="utf-8"))
    if prior_manifest.get("decision") != prior["decision"] or prior.get("retained_unchanged") is not True:
        raise RuntimeError("P7-v2 prior measurement identity/retention 漂移")

    output = root / OUTPUT_ROOT
    if output.exists():
        raise FileExistsError(f"P7-v2 output 已存在，拒绝覆盖: {output}")
    commit = p7._git_identity(root)
    torch.backends.cuda.matmul.allow_tf32 = bool(base["platform"]["allow_tf32"])
    torch.backends.cudnn.benchmark = bool(base["platform"]["cudnn_benchmark"])
    target_device = torch.device(device)
    waveform_host, cached_w_host, input_io, input_identity = p7._load_inputs(root, base)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".p7_iot_efficiency_correction_v2_", dir=output.parent))
    iteration_rows: list[dict[str, Any]] = []
    memory_rows: list[dict[str, Any]] = []
    resources: dict[str, dict[str, Any]] = {}
    source_by_id = {source["candidate_id"]: source for source in base["models"]}
    last_output: torch.Tensor | None = None
    benchmark = correction["benchmark"]
    try:
        # 每轮变换模型位置；profiler 完全移到时延/显存测量之后。
        for round_index, model_order in enumerate(benchmark["round_model_order"], start=1):
            scenario_order = benchmark["round_scenario_order"][round_index - 1]
            for candidate_id in model_order:
                source = source_by_id[candidate_id]
                model, cfg = p7._load_model(root, source, target_device)
                resources.setdefault(candidate_id, _resource_record(source, model))
                for scenario in scenario_order:
                    torch.cuda.synchronize(target_device)
                    torch.cuda.empty_cache()
                    for _ in range(int(benchmark["warmup_iterations"])):
                        _, last_output = p7._timed_iteration(
                            model=model, cfg=cfg, waveform_host=waveform_host,
                            cached_w_host=cached_w_host, scenario=scenario, device=target_device,
                        )
                    torch.cuda.synchronize(target_device)
                    torch.cuda.reset_peak_memory_stats(target_device)
                    baseline_allocated = int(torch.cuda.memory_allocated(target_device))
                    baseline_reserved = int(torch.cuda.memory_reserved(target_device))
                    for iteration in range(1, int(benchmark["timed_iterations"]) + 1):
                        stages, last_output = p7._timed_iteration(
                            model=model, cfg=cfg, waveform_host=waveform_host,
                            cached_w_host=cached_w_host, scenario=scenario, device=target_device,
                        )
                        for stage, latency_ms in stages.items():
                            iteration_rows.append({
                                "candidate_id": candidate_id, "model_variant": source["variant"],
                                "checkpoint_seed": int(source["seed"]), "scenario": scenario,
                                "round": round_index, "iteration": iteration,
                                "stage": stage, "latency_ms": float(latency_ms),
                            })
                    memory_rows.append({
                        "candidate_id": candidate_id, "scenario": scenario, "round": round_index,
                        "baseline_allocated_bytes": baseline_allocated,
                        "baseline_reserved_bytes": baseline_reserved,
                        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(target_device)),
                        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(target_device)),
                    })
                    if last_output is None or not torch.isfinite(last_output).all():
                        raise FloatingPointError(f"P7-v2 {candidate_id}/{scenario} Pi_180 output 非 finite")
                del model, cfg
                torch.cuda.empty_cache()

        iterations = pd.DataFrame(iteration_rows)
        latency_summary = p7.summarize_stage_latencies(iterations, base)
        memory_summary = p7.summarize_memory(pd.DataFrame(memory_rows))
        cwt_consistency, consistency_passed = build_online_cwt_consistency(latency_summary, correction)

        # Profiler 后置，避免其 allocator/后台状态污染正式 latency 与 memory。
        for candidate_id in p7.MODEL_ORDER:
            source = source_by_id[candidate_id]
            model, cfg = p7._load_model(root, source, target_device)
            flops, macs, coverage = p7._profile_model(
                model, cfg, waveform_host, cached_w_host, target_device
            )
            resources[candidate_id].update({
                "profiler_scope": base["profiler"]["scope"],
                "profiler_covered_flops": flops,
                "profiler_covered_macs": macs,
                "profiler_coverage": coverage,
            })
            del model, cfg
            torch.cuda.empty_cache()
        resource_frame = pd.DataFrame([resources[candidate] for candidate in p7.MODEL_ORDER])
        quality = p7.build_quality_efficiency_table(base, latency_summary, resource_frame, repo_root=root)
        quality["online_cwt_cross_model_consistency_passed"] = consistency_passed
        quality["online_cross_model_relative_interpretation_allowed"] = np.where(
            quality["scenario"].eq("online_w"), consistency_passed, True
        )

        environment = p7._environment(target_device, base, input_identity)
        environment["protocol_id"] = PROTOCOL_ID
        environment["benchmark"] = dict(benchmark)
        environment["correction"] = {
            "prior_measurement_manifest_sha256": prior["sha256"],
            "profiler_position": benchmark["profiler_position"],
            "balanced_model_order": True,
            "alternating_scenario_order": True,
            "online_cwt_consistency_passed": consistency_passed,
        }
        environment["execution"] = {"git_commit": commit, "git_dirty": False, "command": command}
        environment["contract_sha256"] = CONFIG_SHA256
        (temporary / "environment.json").write_text(
            json.dumps(environment, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        float_format = correction["output"]["csv_float_format"]
        frames = {
            "input_materialization.csv": input_io,
            "stage_latency_iterations.csv": iterations,
            "stage_latency_summary.csv": latency_summary,
            "online_cwt_consistency.csv": cwt_consistency,
            "memory_summary.csv": memory_summary,
            "model_resource_summary.csv": resource_frame,
            "quality_efficiency_table.csv": quality,
        }
        for filename, frame in frames.items():
            frame.to_csv(temporary / filename, index=False, float_format=float_format)
        generated = sorted(path for path in temporary.iterdir() if path.is_file())
        decision = "measurement_complete" if consistency_passed else "measurement_incomplete"
        manifest = {
            "protocol_id": PROTOCOL_ID,
            "phase": "p7_gpu_iot_efficiency_correction_v2",
            "status": "complete",
            "decision": decision,
            "incomplete_reason": None if consistency_passed else "online_cwt_cross_model_consistency_failed",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "contract_sha256": CONFIG_SHA256,
            "parent_contract_sha256": p7.CONFIG_SHA256,
            "prior_measurement_manifest_sha256": prior["sha256"],
            "git_commit": commit,
            "git_dirty": False,
            "matrix": {
                "models": list(p7.MODEL_ORDER), "scenarios": list(p7.SCENARIO_ORDER),
                "checkpoint_seed": 20260811, "warmup_iterations": 20,
                "timed_iterations": 100, "rounds": 5,
                "expected_timed_pipeline_iterations": 3000,
            },
            "online_cwt_consistency": {
                "passed": consistency_passed,
                "maximum_relative_spread": float(correction["online_cwt_consistency"]["maximum_relative_spread"]),
                "median_relative_spread": float(cwt_consistency["median_relative_spread_across_models"].iloc[0]),
                "p95_relative_spread": float(cwt_consistency["p95_relative_spread_across_models"].iloc[0]),
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
                path.name: {"sha256": p7.sha256_file(path), "size_bytes": path.stat().st_size}
                for path in generated
            },
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        observed = tuple(sorted(path.name for path in temporary.iterdir() if path.is_file()))
        if observed != tuple(sorted(REQUIRED_OUTPUTS)):
            raise RuntimeError(f"P7-v2 output schema 不完整: {observed}")
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output / "manifest.json"
