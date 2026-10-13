"""规则抽样部分汇总、独立计时与参数曲线图。"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import default_collate

from resp_train.metrics.task import summarize_task_metrics
from .sampling_accelerated import AccelerationSpec, accelerated_prefixes
from .study_diagnostics import chunk_keys
from .study_state import preserved_evaluation


def rule_sample_summary(setting, row_ids):
    """仅按预先给定 parent 身份汇总规则抽样部分，完整 cohort 的选择分数不变。"""
    setting = Path(setting)
    metrics = pd.read_csv(setting / "validation_postfiltered_per_parent.csv")
    diagnostics = pd.read_csv(setting / "validation_postfiltered_diagnostics.csv")
    selected = metrics[metrics.dataset_row_id.isin(row_ids)]
    if len(selected) != len(row_ids) or set(selected.dataset_row_id) != set(row_ids):
        raise ValueError("规则抽样部分的 parent 身份缺失或重复")
    summary = summarize_task_metrics(selected).iloc[0]
    result = {f"rule_sample_{key}": None if pd.isna(value) else float(value)
              for key,value in summary.items() if key.endswith(("_mean", "_n"))}
    subset = diagnostics[diagnostics.dataset_row_id.isin(row_ids)]
    result.update({"rule_sample_n_parents": len(row_ids),
                   "rule_sample_parent_max_p99": float(subset.prediction_max_abs.quantile(.99)),
                   "rule_sample_top1pct_energy_mean": float(subset.prediction_top_1pct_sample_energy_ratio.mean())})
    return result


def independent_timing(model, dataset, specs, *, batch_size=64, warmup=1, repeats=2):
    """最终配置的固定首批独立计时；每次重新创建 condition cache。"""
    if warmup < 0 or repeats < 1:
        raise ValueError("独立计时 warmup/repeats 非法")
    batch = default_collate([dataset[i] for i in range(min(batch_size, len(dataset)))])
    keys = chunk_keys(dataset, batch["index"].tolist())
    device = next(model.parameters()).device
    condition = batch["x"].to(device)
    records = []
    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    with preserved_evaluation(model):
        for spec in specs:
            times, allocated, reserved = [], [], []
            for repeat in range(-warmup, repeats):
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                synchronize()
                started = time.perf_counter()
                for _, mean in accelerated_prefixes(model, condition, keys, spec, AccelerationSpec(1),
                                                     checkpoints=(spec.n_trajectories,)):
                    # 同完整采样一样，将本次独立输出交付到 CPU。
                    mean.cpu()
                synchronize()
                wall = time.perf_counter() - started
                if repeat >= 0:
                    times.append(wall)
                    allocated.append(torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None)
                    reserved.append(torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None)
            records.append({"sampler": spec.sampler, "S": spec.nfe, "N": spec.n_trajectories,
                "sampling_seed": spec.noise_seed, "batch_size": len(keys), "keys": keys,
                "warmup": warmup, "timed_repeats": repeats, "sampling_seconds": times,
                "independent_batch_sampling_seconds_median": float(np.median(times)),
                "peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved,
                "definition": "one fixed input batch; standalone N; cache/noise/sampling/mean/CPU delivery",
                "trajectory_group_size": 1, "device": str(device)})
    return records


def plot_grid(curves, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = curves.groupby(["sampler", "nfe_per_trajectory", "sampling_seed"], sort=True)
    panels = [("postfiltered_lag_aware_signed_pcc", "PCC"),
              ("postfiltered_whole_rr_abs_error_bpm", "Whole RR MAE (bpm)"),
              ("postfiltered_parent_max_p99", "Parent max P99")]
    figure, axes = plt.subplots(1, 3, figsize=(16,4))
    for (kind, steps, seed), frame in groups:
        frame = frame.sort_values("N")
        for ax,(metric,label) in zip(axes,panels):
            ax.plot(frame.N, frame[metric], marker="o", label=f"{kind.upper()} S{steps} seed{seed}")
            ax.set_xscale("log",base=2)
            ax.set_xlabel("N trajectories")
            ax.set_ylabel(label)
            ax.grid(alpha=.25)
    axes[0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(Path(output)/"sampler_quality_and_tail_vs_N.png",dpi=130)
    plt.close(figure)
