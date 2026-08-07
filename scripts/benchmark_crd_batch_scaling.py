from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.crd.config import (
    CRD_PROTOCOL_VERSION,
    check_crd_dependencies,
    crd_dependency_versions,
    load_crd_config,
)
from resp_train.crd.model import build_crd_model
from resp_train.crd.training import build_crd_optimizer, train_crd_one_epoch
from resp_train.data.factory import build_window_data
from resp_train.losses.task import RespirationTaskLoss
from resp_train.utils.run import resolve_device, save_execution_manifest, set_seed


EFFECTIVE_BATCH_SIZE = 128
DEFAULT_SCHEMES = ("32x4", "64x2", "128x1")
MIN_THROUGHPUT_GAIN = 0.10
MAX_RESERVED_FRACTION = 0.80


@dataclass(frozen=True)
class BatchScheme:
    physical_batch_size: int
    accumulation_steps: int

    @property
    def label(self) -> str:
        return f"{self.physical_batch_size}x{self.accumulation_steps}"

    @property
    def effective_batch_size(self) -> int:
        return self.physical_batch_size * self.accumulation_steps


def parse_batch_scheme(value: str) -> BatchScheme:
    normalized = str(value).strip().lower().replace("×", "x")
    pieces = normalized.split("x")
    if len(pieces) != 2:
        raise ValueError(f"batch scheme 必须形如 32x4，当前为 {value!r}")
    try:
        scheme = BatchScheme(int(pieces[0]), int(pieces[1]))
    except ValueError as exc:
        raise ValueError(f"batch scheme 必须使用整数，当前为 {value!r}") from exc
    if scheme.physical_batch_size <= 0 or scheme.accumulation_steps <= 0:
        raise ValueError("physical batch 与 accumulation 必须为正")
    if scheme.effective_batch_size != EFFECTIVE_BATCH_SIZE:
        raise ValueError(
            f"batch benchmark 必须保持 effective batch={EFFECTIVE_BATCH_SIZE}，"
            f"当前 {scheme.label}={scheme.effective_batch_size}"
        )
    return scheme


def summarize_benchmark(
    records: list[dict[str, Any]],
    *,
    total_memory_mib: float,
    expected_repeats: int,
    baseline_label: str = "32x4",
) -> dict[str, Any]:
    """应用预先冻结的工程选择规则；首轮只作 cold warmup，不进入吞吐中位数。"""

    labels = list(dict.fromkeys(str(record["scheme"]) for record in records))
    summaries: list[dict[str, Any]] = []
    for label in labels:
        selected = [record for record in records if record["scheme"] == label]
        passed = [record for record in selected if record.get("status") == "passed"]
        steady = [record for record in passed if int(record["repeat_index"]) > 1]
        if expected_repeats == 1:
            steady = passed
        all_passed = len(passed) == expected_repeats
        peak_reserved = max((float(record["peak_reserved_mib"]) for record in passed), default=float("inf"))
        peak_allocated = max((float(record["peak_allocated_mib"]) for record in passed), default=float("inf"))
        throughput = (
            statistics.median(float(record["samples_per_second"]) for record in steady)
            if steady and all_passed
            else 0.0
        )
        elapsed = (
            statistics.median(float(record["elapsed_seconds"]) for record in steady)
            if steady and all_passed
            else float("inf")
        )
        reserved_fraction = peak_reserved / float(total_memory_mib) if total_memory_mib > 0 else float("inf")
        summaries.append(
            {
                "scheme": label,
                "all_repeats_passed": all_passed,
                "steady_samples_per_second_median": throughput,
                "steady_elapsed_seconds_median": elapsed,
                "peak_allocated_mib_max": peak_allocated,
                "peak_reserved_mib_max": peak_reserved,
                "peak_reserved_fraction": reserved_fraction,
                "memory_safe": all_passed and reserved_fraction <= MAX_RESERVED_FRACTION,
            }
        )

    by_label = {summary["scheme"]: summary for summary in summaries}
    if baseline_label not in by_label or not by_label[baseline_label]["memory_safe"]:
        recommendation = baseline_label
        reason = "baseline benchmark 未完整通过，不能据此修订正式协议"
    else:
        baseline_throughput = float(by_label[baseline_label]["steady_samples_per_second_median"])
        eligible = []
        for summary in summaries:
            gain = (
                float(summary["steady_samples_per_second_median"]) / baseline_throughput - 1.0
                if baseline_throughput > 0.0
                else float("-inf")
            )
            summary["throughput_gain_vs_32x4"] = gain
            if summary["memory_safe"] and gain >= MIN_THROUGHPUT_GAIN:
                eligible.append(summary)
        if eligible:
            selected = max(eligible, key=lambda item: float(item["steady_samples_per_second_median"]))
            recommendation = str(selected["scheme"])
            reason = (
                f"相对 32x4 稳态吞吐提升至少 {MIN_THROUGHPUT_GAIN:.0%}，"
                f"且 peak reserved 不超过显存的 {MAX_RESERVED_FRACTION:.0%}"
            )
        else:
            recommendation = baseline_label
            reason = "没有候选同时达到 10% 稳态吞吐增益与 80% 显存安全线，保留冻结基线"
    return {
        "selection_rule": {
            "cold_repeat_excluded_from_throughput": expected_repeats > 1,
            "minimum_throughput_gain": MIN_THROUGHPUT_GAIN,
            "maximum_peak_reserved_fraction": MAX_RESERVED_FRACTION,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        },
        "schemes": summaries,
        "recommended_scheme": recommendation,
        "recommendation_reason": reason,
    }


def _run_repeat(
    *,
    cfg,
    dataset,
    scheme: BatchScheme,
    repeat_index: int,
    device: torch.device,
) -> dict[str, Any]:
    set_seed(int(cfg.training.seed))
    loader = DataLoader(
        dataset,
        batch_size=scheme.physical_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )
    expected_microbatches = scheme.accumulation_steps
    if len(loader) != expected_microbatches:
        raise RuntimeError(f"{scheme.label} 期望 {expected_microbatches} microbatches，实际 {len(loader)}")

    model = build_crd_model(cfg).to(device)
    loss_fn = RespirationTaskLoss(cfg).to(device)
    optimizer, _ = build_crd_optimizer(model, cfg)
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = torch.cuda.memory_allocated(device) / (1024**2)
    baseline_reserved = torch.cuda.memory_reserved(device) / (1024**2)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    start_event.record()
    wall_start = time.perf_counter()
    summary, next_update = train_crd_one_epoch(
        model,
        loader,
        loss_fn,
        optimizer,
        device=device,
        accumulation_steps=scheme.accumulation_steps,
        update_index=0,
        total_updates=1,
        max_learning_rate=float(cfg.training.max_learning_rate),
        min_learning_rate=float(cfg.training.min_learning_rate),
        warmup_fraction=float(cfg.training.warmup_fraction),
        grad_clip_norm=float(cfg.training.grad_clip_norm),
        use_amp=bool(cfg.training.use_amp),
        show_progress=False,
        epoch=1,
        total_epochs=1,
    )
    end_event.record()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - wall_start
    gpu_elapsed = start_event.elapsed_time(end_event) / 1000.0
    if next_update != 1 or int(summary["optimizer_updates"]) != 1:
        raise RuntimeError(f"{scheme.label} 没有严格产生一次 optimizer update")
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise FloatingPointError(f"{scheme.label} update 后模型参数包含 NaN/Inf")

    return {
        "scheme": scheme.label,
        "physical_batch_size": scheme.physical_batch_size,
        "accumulation_steps": scheme.accumulation_steps,
        "effective_batch_size": scheme.effective_batch_size,
        "repeat_index": repeat_index,
        "repeat_role": "cold" if repeat_index == 1 else "steady",
        "status": "passed",
        "elapsed_seconds": elapsed,
        "gpu_elapsed_seconds": gpu_elapsed,
        "samples_per_second": EFFECTIVE_BATCH_SIZE / elapsed,
        "baseline_allocated_mib": baseline_allocated,
        "baseline_reserved_mib": baseline_reserved,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024**2),
        "train_loss_total": float(summary["loss"]),
        "train_loss_sync": float(summary["loss_sync"]),
        "train_loss_effort": float(summary["loss_effort"]),
        "eligible_sync": int(summary["eligible_sync"]),
        "eligible_effort": int(summary["eligible_effort"]),
        "learning_rate": float(summary["last_learning_rate"]),
    }


def _cleanup_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="CRD_103 等效 batch-scaling 工程 benchmark")
    parser.add_argument("--config", default="configs/crd_v1/crd_103_direct_local_mamba.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--schemes", nargs="+", default=list(DEFAULT_SCHEMES))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.repeats < 2:
        raise SystemExit("benchmark 至少需要 2 repeats：首轮 cold，后续才是稳态")
    schemes = [parse_batch_scheme(value) for value in args.schemes]
    if len({scheme.label for scheme in schemes}) != len(schemes):
        raise SystemExit("--schemes 不能重复")
    problems = check_crd_dependencies()
    if problems:
        raise SystemExit("; ".join(problems))

    overrides = [
        "protocol.run_role=acceptance",
        "data.max_train_windows=128",
        "data.max_val_windows=32",
        "training.epochs=1",
        "training.batch_size=128",
        "training.gradient_accumulation_steps=1",
        f"training.device={args.device}",
        "training.show_progress=false",
    ]
    cfg = load_crd_config(args.config, overrides=overrides)
    if str(cfg.model.variant) != "crd_103_direct_local_mamba":
        raise SystemExit("本轮冻结 benchmark 只允许 CRD_103")
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise SystemExit("batch-scaling benchmark 必须在 CUDA 上执行")
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)

    window_data = build_window_data(
        cfg,
        split=str(cfg.data.train_split),
        max_windows=EFFECTIVE_BATCH_SIZE,
        sample_strategy=str(cfg.data.train_sample_strategy),
        sample_seed=int(cfg.data.train_sample_seed),
        shuffle=False,
    )
    if len(window_data.dataset) != EFFECTIVE_BATCH_SIZE:
        raise RuntimeError(f"benchmark 期望 {EFFECTIVE_BATCH_SIZE} windows，实际 {len(window_data.dataset)}")
    generator = torch.Generator().manual_seed(int(cfg.training.seed))
    order = torch.randperm(EFFECTIVE_BATCH_SIZE, generator=generator).tolist()
    dataset = Subset(window_data.dataset, order)

    output_path = (
        Path(args.output)
        if args.output
        else Path("/tmp") / f"crd_batch_scaling_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path.with_name(f"{output_path.stem}_manifest.json")
    records: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "protocol": CRD_PROTOCOL_VERSION,
        "role": "engineering_batch_scaling",
        "scientific_evidence": False,
        "config": str(Path(args.config).resolve()),
        "variant": str(cfg.model.variant),
        "seed": int(cfg.training.seed),
        "sample_count": EFFECTIVE_BATCH_SIZE,
        "sample_order": order,
        "device": str(device),
        "device_name": properties.name,
        "device_total_memory_mib": properties.total_memory / (1024**2),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "dependency_versions": crd_dependency_versions(),
        "schemes": [asdict(scheme) | {"label": scheme.label} for scheme in schemes],
        "repeats": int(args.repeats),
        "records": records,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for scheme in schemes:
        for repeat_index in range(1, int(args.repeats) + 1):
            print(f"benchmark {scheme.label} repeat {repeat_index}/{args.repeats}", flush=True)
            try:
                record = _run_repeat(
                    cfg=cfg,
                    dataset=dataset,
                    scheme=scheme,
                    repeat_index=repeat_index,
                    device=device,
                )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
                if not is_oom:
                    raise
                record = {
                    "scheme": scheme.label,
                    "physical_batch_size": scheme.physical_batch_size,
                    "accumulation_steps": scheme.accumulation_steps,
                    "effective_batch_size": scheme.effective_batch_size,
                    "repeat_index": repeat_index,
                    "repeat_role": "cold" if repeat_index == 1 else "steady",
                    "status": "oom",
                    "error": str(exc),
                    "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
                    "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024**2),
                }
            records.append(record)
            payload["records"] = records
            output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(record, ensure_ascii=False), flush=True)
            _cleanup_cuda()

    payload["summary"] = summarize_benchmark(
        records,
        total_memory_mib=float(payload["device_total_memory_mib"]),
        expected_repeats=int(args.repeats),
    )
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save_execution_manifest(
        manifest_path,
        task="crd_batch_scaling",
        phase="engineering_benchmark",
        scientific_evidence=False,
        config=str(Path(args.config).resolve()),
        output=str(output_path.resolve()),
        schemes=[scheme.label for scheme in schemes],
        repeats=int(args.repeats),
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(output_path)


if __name__ == "__main__":
    main()
