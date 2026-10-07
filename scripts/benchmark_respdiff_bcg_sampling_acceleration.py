#!/usr/bin/env python3
"""单 GPU 短时 benchmark：原串行、缓存 G1、缓存并行 G2/G4/G8；默认合成输入。"""

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from resp_train.respdiff_bcg.acceleration_checks import TOLERANCES, error_record, verify_grouped
from resp_train.respdiff_bcg.baseband import LPF_CONTRACT, lowpass_parent
from resp_train.respdiff_bcg.inference_v2 import load_frozen_source, state_digest
from resp_train.respdiff_bcg.runtime import (
    LOGGER, artifact_manifest, environment, run_logging, save_source_snapshot,
    seed_all, sha256, source_identity, write_json,
)
from resp_train.respdiff_bcg.sampling import SamplerSpec, ensemble_prefixes, reconstruct_ensemble_parent
from resp_train.respdiff_bcg.sampling_accelerated import AccelerationSpec, accelerated_prefixes
from resp_train.respdiff_bcg.signal import chunks_from_parent, prepare_parent
from scripts.run_respdiff_bcg_epsilon_inference_v2 import DEFAULT_SOURCE


def synthetic_condition(batch_size):
    """走完整 parent LPF→AA→13 chunk 合同，不读取真实数据。"""
    conditions, keys = [], []
    time_axis = np.arange(18000) / 100
    for parent in range((batch_size + 12) // 13):
        waveform = (np.sin(2 * np.pi * (.19 + .01 * parent) * time_axis)
                    + .1 * np.sin(2 * np.pi * 2 * time_axis))
        chunks = chunks_from_parent(prepare_parent(lowpass_parent(waveform)))
        for chunk in range(13):
            conditions.append(torch.from_numpy(chunks[chunk].copy()))
            keys.append(("synthetic", 100000 + parent, chunk))
    return torch.stack(conditions[:batch_size]), keys[:batch_size]


def sample_once(model, condition, keys, spec, acceleration, counts):
    """记录共享 prefix 的真实 readiness：同组 prefix 使用相同完成时间。"""
    device = condition.device
    ready, results = {}, {}
    groups_completed = 0
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    def completed(ids):
        nonlocal groups_completed
        torch.cuda.synchronize(device)
        groups_completed += 1
        wall = time.perf_counter() - started
        for count in counts:
            if ids[0] < count <= ids[-1] + 1:
                ready[count] = {"prefix_ready_seconds": wall, "completed_trajectories": ids[-1] + 1,
                                "actual_grouped_forward_calls": groups_completed * spec.nfe}
    generator = (ensemble_prefixes(model, condition, keys, spec, checkpoints=counts) if acceleration is None
        else accelerated_prefixes(model, condition, keys, spec, acceleration, checkpoints=counts,
                                  completion_observer=completed))
    for count, mean in generator:
        torch.cuda.synchronize(device)
        wall = time.perf_counter() - started
        results[count] = mean
        if acceleration is None:
            ready[count] = {"prefix_ready_seconds": wall, "completed_trajectories": count,
                            "actual_grouped_forward_calls": count * spec.nfe}
        ready[count].update({"N": count, "logical_denoiser_calls_per_chunk": count * spec.nfe,
            "actual_logical_calls_completed_per_chunk": ready[count]["completed_trajectories"] * spec.nfe,
            "prefix_delivery_seconds": wall,
            "runtime_definition": "shared max-N prefix readiness; not independent N timing"})
    torch.cuda.synchronize(device)
    return time.perf_counter() - started, results, list(ready.values())


def waveform_checks(means, references, spec, group_size):
    records = []
    for count, actual in means.items():
        reference = references[count]
        records.append(error_record(actual, reference, "prefix", G=group_size, N=count))
        # 原始 B64 包含 4 个完整 parent 和一个尾 parent；只重建完整 13 块。
        for parent in range(len(actual) // 13):
            begin, end = parent * 13, (parent + 1) * 13
            _, raw, post = reconstruct_ensemble_parent(actual[begin:end].cpu().numpy(), postfilter=spec.postfilter)
            _, expected_raw, expected_post = reconstruct_ensemble_parent(reference[begin:end].cpu().numpy(),
                                                                        postfilter=spec.postfilter)
            records.append(error_record(raw, expected_raw, "raw_waveform", G=group_size, N=count, parent=parent))
            records.append(error_record(post, expected_post, "postfiltered_waveform", G=group_size,
                                        N=count, parent=parent))
    return records


def benchmark_source_identity(cfg):
    identity = source_identity(cfg)
    for path in (Path(__file__).resolve(), ROOT / "scripts/run_respdiff_bcg_epsilon_inference_v2.py",
                 ROOT / "docs/experiments/respdiff_bcg_sampling_acceleration_v1_20261007.md"):
        identity["files"][str(path.relative_to(ROOT))] = sha256(path)
    return identity


def run(output, *, source_run=DEFAULT_SOURCE, device="cuda:0", sampler="ddim", n_trajectories=8,
        batch_size=64, warmup=1, repeats=2):
    if not str(device).startswith("cuda:") or not str(device)[5:].isdigit():
        raise ValueError("benchmark 须为显式单 GPU cuda:N")
    if not 13 <= batch_size <= 64 or warmup < 1 or repeats < 1:
        raise ValueError("batch_size 要求 13…64，warmup/repeats 至少为 1")
    spec = SamplerSpec(sampler=sampler, nfe=6 if sampler == "ddim" else 50,
                       n_trajectories=n_trajectories)
    counts = tuple(sorted({count for count in (1, 2, 4, 8, 16) if count <= n_trajectories} | {n_trajectories}))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    with run_logging(output):
        stage = "initialization"
        try:
            # 仅设置选定 GPU；noise 仍来自各身份独立的 CPU generator。
            seed_all(20260811, cpu_only=True)
            torch.cuda.set_device(torch.device(device))
            torch.cuda.manual_seed(20260811)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            model, cfg, checkpoint = load_frozen_source(source_run, torch.device(device))
            before = state_digest(model)
            condition, keys = synthetic_condition(batch_size)
            np.savez(output / "input_fixture.npz", condition=condition.numpy(), keys=np.array(keys, dtype=str))
            condition = condition.to(device)
            identity = benchmark_source_identity(cfg)
            write_json(output / "source_identity.json", identity)
            save_source_snapshot(output, identity)
            write_json(output / "environment.json", {**environment(),
                "gpu": torch.cuda.get_device_name(device), "device": str(device),
                "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                "cudnn_benchmark": torch.backends.cudnn.benchmark})
            write_json(output / "benchmark_identity.json", {"checkpoint": checkpoint, "sampler": asdict(spec),
                "groups": [1, 2, 4, 8], "counts": counts, "warmup": warmup, "repeats": repeats,
                "batch_size": batch_size, "input": "synthetic parent LPF fixture", "lowpass": LPF_CONTRACT,
                "tolerances": TOLERANCES, "parameter_hash_before": before})
            profiles, prefix_profiles, errors = [], [], []
            reference_means, baseline_time = None, None
            for label, acceleration in [("serial_reference", None), ("condition_cache_G1", AccelerationSpec(1)),
                ("condition_cache_G2", AccelerationSpec(2)), ("condition_cache_G4", AccelerationSpec(4)),
                ("condition_cache_G8", AccelerationSpec(8))]:
                stage = label
                group = acceleration.trajectory_group_size if acceleration is not None else 1
                LOGGER.info("benchmark %s：B=%d N=%d NFE=%d", label, batch_size, n_trajectories, spec.nfe)
                row = {"variant": label, "G": group, "condition_cache": acceleration is not None,
                    "N": n_trajectories, "NFE": spec.nfe, "original_batch_size": batch_size,
                    "effective_max_group_size": min(group, n_trajectories),
                    "largest_compute_batch_size": batch_size * min(group, n_trajectories),
                    "condition_encoding_calls_per_batch": 1 if acceleration is not None else n_trajectories * spec.nfe,
                    "logical_denoiser_calls_per_chunk": n_trajectories * spec.nfe,
                    "actual_grouped_denoiser_forward_calls": spec.nfe * (
                        (n_trajectories + group - 1) // group), "warmup": warmup, "repeats": repeats}
                try:
                    if acceleration is not None:
                        verified = verify_grouped(model, condition, keys, spec, acceleration, counts=counts)
                        errors.extend({"variant": label, **record} for record in verified)
                        pd.DataFrame(errors).to_csv(output / "numerical_errors.csv", index=False)
                        if not all(record["passed"] for record in verified):
                            raise FloatingPointError(f"{label} 逐 step/trajectory 超出预定容差")
                    gc.collect()
                    torch.cuda.empty_cache()
                    for _ in range(warmup):
                        _, values, _ = sample_once(model, condition, keys, spec, acceleration, counts)
                        del values
                    torch.cuda.reset_peak_memory_stats(device)
                    times = []
                    for repeat in range(repeats):
                        elapsed, values, prefix = sample_once(model, condition, keys, spec, acceleration, counts)
                        times.append(elapsed)
                        prefix_profiles.extend({"variant": label, "G": group, "repeat": repeat, **record}
                                               for record in prefix)
                        if label == "serial_reference" and reference_means is None:
                            reference_means = {count: value.cpu() for count, value in values.items()}
                        elif acceleration is not None and repeat == 0:
                            checked = waveform_checks(values, reference_means, spec, group)
                            errors.extend({"variant": label, **record} for record in checked)
                            pd.DataFrame(errors).to_csv(output / "numerical_errors.csv", index=False)
                            if not all(record["passed"] for record in checked):
                                raise FloatingPointError(f"{label} ensemble/重建波形超出预定容差")
                        del values
                    median = float(np.median(times))
                    if acceleration is None:
                        baseline_time = median
                    cache_time = next((p["sampling_seconds_median"] for p in profiles
                        if p["variant"] == "condition_cache_G1" and p["status"] == "passed"),
                        median if acceleration is not None and group == 1 else None)
                    row.update({"status": "passed", "finite": True, "sampling_seconds_median": median,
                        "sampling_seconds_repeats": json.dumps(times), "ensemble_chunks_per_second": batch_size / median,
                        "speedup_vs_serial": baseline_time / median,
                        "speedup_vs_cached_G1": cache_time / median if cache_time is not None else None,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)})
                    LOGGER.info("%s：median=%.3fs speedup=%.3fx allocated=%.2fGiB reserved=%.2fGiB", label,
                        median, row["speedup_vs_serial"], row["peak_allocated_bytes"] / 2**30,
                        row["peak_reserved_bytes"] / 2**30)
                except torch.cuda.OutOfMemoryError as error:
                    if acceleration is None:
                        raise
                    row.update({"status": "oom", "error": str(error), "finite": None})
                    (output / f"{label}_oom.txt").write_text(traceback.format_exc())
                    LOGGER.error("%s 显存不足，保持原 B/G，不自动降组：%s", label, error)
                profiles.append(row)
                pd.DataFrame(profiles).to_csv(output / "runtime_profile.csv", index=False)
                pd.DataFrame(prefix_profiles).to_csv(output / "shared_prefix_runtime_profile.csv", index=False)
                gc.collect()
                torch.cuda.empty_cache()
            if state_digest(model) != before or sha256(Path(source_run) / "final.pt") != checkpoint["checkpoint_sha256"]:
                raise ValueError("benchmark 改变了 checkpoint 文件、参数或 buffer")
            if benchmark_source_identity(cfg) != identity:
                raise ValueError("benchmark 期间源码或配置变化")
            successful = [row for row in profiles if row["condition_cache"] and row["status"] == "passed"]
            best = min(successful, key=lambda row: row["sampling_seconds_median"]) if successful else None
            write_json(output / "summary.json", {"best_measured_group_size": best["G"] if best else None,
                "profiles": profiles, "tolerances": TOLERANCES,
                "checkpoint_bytes_unchanged": True, "parameter_buffers_unchanged": True,
                "recommendation_scope": "only this GPU/B/sampler/N fixture; no full validation"})
            LOGGER.info("benchmark 完成：最快已通过组 G=%s", best["G"] if best else None)
            write_json(output / "receipt.json", {"status": "complete_with_oom" if any(
                row["status"] == "oom" for row in profiles) else "complete", "mode": "gpu_benchmark",
                "formal_model_evidence": False, "artifacts": artifact_manifest(output)})
        except BaseException as error:
            LOGGER.error("benchmark 失败：stage=%s %s: %s", stage, type(error).__name__, error)
            LOGGER.debug("异常堆栈", exc_info=True)
            write_json(output / "failure.json", {"status": "failed", "stage": stage,
                       "error_type": type(error).__name__, "error": str(error)})
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sampler", choices=("ddim", "ddpm"), default="ddim")
    parser.add_argument("--n-trajectories", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output, source_run=args.source_run, device=args.device, sampler=args.sampler,
        n_trajectories=args.n_trajectories, batch_size=args.batch_size, warmup=args.warmup, repeats=args.repeats)
