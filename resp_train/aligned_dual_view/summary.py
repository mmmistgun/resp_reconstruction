from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from .artifacts import artifact_directory, sha256_file, verified_manifest, write_json
from .config import CORE_VIEWS, PROTOCOL, SEEDS, load_experiment_config
from .experiment import PRIMARY, check_primary_metrics


def summarize_runs(run_roots, *, output, include_scale_mean=False):
    """仅汇总完整预定义矩阵；不按中间效果缩小 arm 或 seed 集合。"""
    roots = [Path(value).resolve() for value in run_roots]
    views = (*CORE_VIEWS, "joint_scale_mean") if include_scale_mean else CORE_VIEWS
    expected = {(view, seed) for view in views for seed in SEEDS}
    if len(roots) != len(expected) or len(set(roots)) != len(roots):
        raise ValueError(f"必须提供完整 {len(expected)} 个不同 formal runs")
    cfg = load_experiment_config(roots[0] / "config.yaml")
    with artifact_directory(output, kind="summary", cfg=cfg) as (root, _):
        observed, rows, sources, comparisons = set(), [], [], set()
        for source_root in roots:
            manifest = verified_manifest(source_root, "train")
            key = (manifest["view"], manifest["seed"])
            if manifest["run_role"] != "formal" or key not in expected or key in observed:
                raise ValueError("汇总包含 smoke、重复或矩阵外 run")
            if manifest["epochs_completed"] != 80 or manifest.get("test_waveform_read") is not False:
                raise ValueError("formal run 未完整执行或访问边界不匹配")
            for filename, field in (("metrics.csv", "metrics_sha256"),
                                    ("metrics_summary.csv", "summary_sha256"),
                                    ("history.csv", "history_sha256"), ("config.yaml", "config_sha256"),
                                    ("samples.json", "samples_sha256")):
                if sha256_file(source_root / filename) != manifest[field]:
                    raise ValueError(f"汇总输入哈希不匹配: {source_root}/{filename}")
            for field, digest in (("selected_checkpoint", "selected_sha256"), ("final_checkpoint", "final_sha256")):
                if not re.fullmatch(r"checkpoints/epoch_[0-9]{3}\.pt", manifest[field]):
                    raise ValueError("checkpoint 路径不合格")
                if sha256_file(source_root / manifest[field]) != manifest[digest]:
                    raise ValueError("checkpoint 产物不完整或哈希不匹配")
            source_cfg = load_experiment_config(source_root / "config.yaml")
            if source_cfg.protocol.run_role != "formal" or (source_cfg.model.input_view, source_cfg.training.seed) != key:
                raise ValueError("run manifest 与 resolved config 不匹配")
            history = pd.read_csv(source_root / "history.csv")
            if history.epoch.tolist() != list(range(1, 81)) or not np.isfinite(history.val_local_rr_mae).all():
                raise ValueError("formal history 不完整或非有限")
            selected_epoch = int(history.iloc[np.argmin(history.val_local_rr_mae.to_numpy())].epoch)
            if selected_epoch != manifest["selected_epoch"]:
                raise ValueError("selected epoch 偏离 full-validation Local RR 最早最小值")
            metrics = pd.read_csv(source_root / "metrics.csv")
            check_primary_metrics(metrics)
            summary = pd.read_csv(source_root / "metrics_summary.csv")
            if len(summary) != 1 or int(summary.n_samples.iloc[0]) != len(metrics):
                raise ValueError("metrics summary 样本集合不匹配")
            record = {"view": key[0], "seed": key[1], "selected_epoch": selected_epoch}
            for metric in PRIMARY:
                value = float(metrics[metric].mean())
                if not np.isclose(value, float(summary[f"{metric}_mean"].iloc[0]), rtol=1e-12, atol=1e-12):
                    raise ValueError(f"{metric} direct mean 与 summary 不一致")
                record[metric] = value
            rows.append(record)
            observed.add(key)
            comparisons.add(manifest["comparison_id"])
            sources.append({"root": str(source_root), "manifest_sha256": sha256_file(source_root / "manifest.json")})
        if observed != expected or len(comparisons) != 1:
            raise ValueError("矩阵不完整或数据/预算/代码/依赖不一致")
        scores = pd.DataFrame(rows).sort_values(["view", "seed"])
        scores.to_csv(root / "per_seed.csv", index=False)
        aggregates, deltas = [], []
        anchor = scores[scores.view == "joint"].set_index("seed")
        for view in views:
            candidate = scores[scores.view == view].set_index("seed").loc[list(SEEDS)]
            for metric in PRIMARY:
                values = candidate[metric].to_numpy()
                aggregates.append({"view": view, "metric": metric, "mean": values.mean(),
                                   "seed_sd": values.std(ddof=1), "n_seeds": 3,
                                   "higher_is_better": metric == "lag_aware_signed_pcc"})
                for seed in SEEDS:
                    deltas.append({"view": view, "seed": seed, "metric": metric,
                                   "candidate_minus_joint": float(candidate.loc[seed, metric] - anchor.loc[seed, metric])})
        pd.DataFrame(aggregates).to_csv(root / "across_seed.csv", index=False)
        pd.DataFrame(deltas).to_csv(root / "paired_deltas.csv", index=False)
        write_json(root / "manifest.json", {
            "protocol": PROTOCOL, "kind": "summary", "sources": sources,
            "comparison_id": next(iter(comparisons)), "views": list(views), "seeds": list(SEEDS),
            "aggregation": "sample_direct_mean_then_equal_seed_mean", "test_waveform_read": False,
            "files": {name: sha256_file(root / name) for name in ("per_seed.csv", "across_seed.csv", "paired_deltas.csv")},
        })
    return Path(output).resolve()
