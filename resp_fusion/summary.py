from __future__ import annotations

from itertools import combinations
from pathlib import Path
import re

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from resp_train.metrics.task import summarize_task_metrics
from resp_train.aligned_dual_view.features import spec_digest
from .artifacts import artifact_directory, read_json, sha256_file, verified_manifest, write_json
from .config import ALL_ARMS, SEEDS, PROTOCOL, load_experiment_config
from .experiment import PRIMARY, check_primary_metrics, comparison_identity
from .model import ARMS
from .stopping import stopping_state


def factorial_contrasts(scores):
    """同 seed 的简单效应、边际效应与差分之差；不把 seed 当成人群重复。"""
    expected = {(arm, seed) for arm in ALL_ARMS for seed in SEEDS}
    if len(scores) != 18 or set(zip(scores.arm, scores.seed)) != expected:
        raise ValueError("因子分析要求恰好 18 个唯一 arm/seed")
    if not np.isfinite(scores[list(PRIMARY)].to_numpy()).all():
        raise ValueError("因子分析主指标必须有限")
    index = scores.set_index(["arm", "seed"])
    reverse = {value: key for key, value in ARMS.items()}
    rows = []
    methods = ("concat", "film", "attention")
    for seed in SEEDS:
        for metric in PRIMARY:
            def value(method, position):
                return float(index.loc[(reverse[method, position], seed), metric])
            def add(kind, contrast, delta):
                rows.append({"kind": kind, "contrast": contrast, "seed": seed,
                             "metric": metric, "delta": delta,
                             "higher_is_better": metric == "lag_aware_signed_pcc"})
            for anchor, candidate in combinations(methods, 2):
                pre = value(candidate, "pre") - value(anchor, "pre")
                post = value(candidate, "post") - value(anchor, "post")
                add("method_at_position", f"{candidate}-{anchor}@pre", pre)
                add("method_at_position", f"{candidate}-{anchor}@post", post)
                add("interaction", f"({candidate}-{anchor})@post-({candidate}-{anchor})@pre", post-pre)
                add("marginal_method", f"{candidate}-{anchor}:equal_positions", (pre+post)/2)
            positions = []
            for method in methods:
                delta = value(method, "post") - value(method, "pre")
                positions.append(delta)
                add("position_at_method", f"post-pre@{method}", delta)
            add("marginal_position", "post-pre:equal_methods", float(np.mean(positions)))
    return pd.DataFrame(rows)


def summarize_runs(run_roots, *, output):
    roots = [Path(value).resolve() for value in run_roots]
    expected = {(arm, seed) for arm in ALL_ARMS for seed in SEEDS}
    if len(roots) != 18 or len(set(roots)) != 18:
        raise ValueError("必须提供完整 18 个不同 formal runs")
    cfg = load_experiment_config(roots[0] / "config.yaml")
    with artifact_directory(output, kind="summary", cfg=cfg) as (root, started):
        observed, records, sources, comparisons = set(), [], [], set()
        shared_by_seed, fusion_by_method_seed = {}, {}
        row_sets = set()
        for source_root in roots:
            m = verified_manifest(source_root, "train")
            key = (m["arm"], m["seed"])
            if m["run_role"] != "formal" or key not in expected or key in observed:
                raise ValueError("汇总包含 smoke、重复或矩阵外 run")
            if not 30 <= m["epochs_completed"] <= 80 or m.get("test_waveform_read") is not False:
                raise ValueError("formal 未完成或访问边界不符")
            for filename, field in (("metrics.csv", "metrics_sha256"), ("metrics_summary.csv", "summary_sha256"),
                                    ("history.csv", "history_sha256"), ("config.yaml", "config_sha256"),
                                    ("samples.json", "samples_sha256"), ("initialization.json", "initialization_sha256"),
                                    ("model_report.json", "model_report_sha256")):
                if sha256_file(source_root / filename) != m[field]:
                    raise ValueError(f"产物哈希不匹配: {filename}")
            for field, digest in (("selected_checkpoint", "selected_sha256"), ("final_checkpoint", "final_sha256")):
                if not re.fullmatch(r"checkpoints/epoch_[0-9]{3}\.pt", m[field]):
                    raise ValueError("checkpoint 路径不合格")
                if sha256_file(source_root / m[field]) != m[digest]:
                    raise ValueError("checkpoint 哈希不匹配")
            source_cfg = load_experiment_config(source_root / "config.yaml")
            if (source_cfg.model.arm, source_cfg.training.seed) != key or source_cfg.protocol.run_role != "formal":
                raise ValueError("manifest/config arm 或 seed 不匹配")
            identity = m["identity"]
            if (identity["protocol"] != PROTOCOL or identity["code_sha256"] != started["code"]["sha256"]
                    or identity["config_sha256"] != spec_digest(OmegaConf.to_container(source_cfg, resolve=True))):
                raise ValueError("训练代码或配置 identity 漂移")
            source_start = read_json(source_root / "started.json")
            if source_start["dependencies"] != started["dependencies"] or source_start["code"] != started["code"]:
                raise ValueError("执行源码/依赖与汇总不一致")
            method, position = ARMS[key[0]]
            if (m["method"], m["position"]) != (method, position):
                raise ValueError("融合因子标签错配")
            samples = read_json(source_root / "samples.json")
            if set(samples) != {"train", "val"}:
                raise ValueError("samples split 不完整")
            sample_ids = {split: spec_digest(values) for split, values in samples.items()}
            if sample_ids != identity["samples"]:
                raise ValueError("sample 内容 identity 不一致")
            for split, count in (("train", 10141), ("val", 2675)):
                items = samples[split]
                ids = [item["identity"]["dataset_row_id"] for item in items]
                if len(items) != count or len(set(ids)) != count or ids != sorted(ids):
                    raise ValueError("sample 数量或 row 顺序不完整")
                if any(item["identity"]["split"] != split for item in items):
                    raise ValueError("sample split 错配")
            expected_comparison = comparison_identity(source_cfg, sample_ids, identity["cache_id"],
                                                      identity["code_sha256"], started["dependencies"])
            if m["comparison_id"] != expected_comparison:
                raise ValueError("comparison ID 无法从来源重建")
            comparisons.add((expected_comparison, identity["cache_manifest_sha256"]))
            initial = read_json(source_root / "initialization.json")
            shared = {name: value for name, value in initial["state"].items() if not name.startswith("fusion.")}
            shared_sha = spec_digest(shared)
            if shared_sha != initial["shared_sha256"] or shared_sha != m["shared_initialization_sha256"]:
                raise ValueError("共享初始化身份损坏")
            if shared_by_seed.setdefault(key[1], shared_sha) != shared_sha:
                raise ValueError("同 seed 六组共享模块未配对初始化")
            fusion = spec_digest({name: value for name, value in initial["state"].items() if name.startswith("fusion.")})
            if fusion_by_method_seed.setdefault((method, key[1]), fusion) != fusion:
                raise ValueError("同方式前后位置融合模块未配对初始化")
            history = pd.read_csv(source_root / "history.csv", float_precision="round_trip")
            actual_epochs = m["epochs_completed"]
            if (history.epoch.tolist() != list(range(1, actual_epochs+1))
                    or history.optimizer_update.tolist() != list(range(80, 80*actual_epochs+1, 80))
                    or not np.isfinite(history.to_numpy(dtype=float)).all()):
                raise ValueError("formal history/更新数不完整或非有限")
            stopping = stopping_state(source_cfg)
            for value in history.val_local_rr_mae:
                stopping.step(value)
            expected_stop = stopping.receipt()
            recorded_stop = dict(m.get("stopping", {}))
            recorded_best = recorded_stop.pop("best_value", float("nan"))
            expected_best = expected_stop.pop("best_value")
            if (recorded_stop != expected_stop or not np.isclose(recorded_best, expected_best, rtol=1e-12, atol=1e-12)
                    or m.get("updates_per_epoch") != 80
                    or m.get("optimizer_updates_completed") != 80*actual_epochs
                    or m.get("schedule_total_updates") != 6400):
                raise ValueError("停止原因、实际更新数或80epoch学习率预算不匹配")
            best = int(history.iloc[np.argmin(history.val_local_rr_mae)].epoch)
            if (best != m["selected_epoch"] or m["selected_checkpoint"] != f"checkpoints/epoch_{best:03d}.pt"
                    or m["final_checkpoint"] != f"checkpoints/epoch_{actual_epochs:03d}.pt"
                    or not np.isclose(history.val_local_rr_mae.min(), m["selected_local_rr_mae"], rtol=1e-12, atol=1e-12)):
                raise ValueError("selected checkpoint 偏离最早 Local RR 最小值")
            metrics = pd.read_csv(source_root / "metrics.csv")
            expected_ids = [item["identity"]["dataset_row_id"] for item in samples["val"]]
            if (metrics.dataset_row_id.tolist() != expected_ids or set(metrics.split) != {"val"}
                    or set(metrics.method) != {f"adv_fusion_v1_{key[0]}"}):
                raise ValueError("metrics row/split/method 错配")
            row_sets.add(spec_digest(expected_ids))
            check_primary_metrics(metrics)
            saved = pd.read_csv(source_root / "metrics_summary.csv")
            calculated = summarize_task_metrics(metrics)
            pd.testing.assert_frame_equal(saved, calculated, check_dtype=False, rtol=1e-12, atol=1e-12)
            records.append({"arm": key[0], "method": method, "position": position, "seed": key[1],
                            "selected_epoch": best, "epochs_completed": actual_epochs,
                            "optimizer_updates_completed": m["optimizer_updates_completed"],
                            "stop_reason": m["stopping"]["reason"],
                            **{metric: float(saved[f"{metric}_mean"].iloc[0]) for metric in PRIMARY}})
            observed.add(key)
            sources.append({"root": str(source_root), "manifest_sha256": sha256_file(source_root / "manifest.json")})
        if observed != expected or len(comparisons) != 1 or len(row_sets) != 1:
            raise ValueError("矩阵不完整或数据/代码/预算不一致")
        scores = pd.DataFrame(records).sort_values(["arm", "seed"])
        scores.to_csv(root / "per_seed.csv", index=False)
        across = []
        for arm, group in scores.groupby("arm"):
            for metric in PRIMARY:
                values = group[metric].to_numpy()
                across.append({"arm": arm, "metric": metric, "mean": values.mean(),
                               "seed_sd": values.std(ddof=1), "n_seeds": 3})
        pd.DataFrame(across).to_csv(root / "across_seed.csv", index=False)
        contrasts = factorial_contrasts(scores)
        contrasts.to_csv(root / "contrasts_per_seed.csv", index=False)
        agg = contrasts.groupby(["kind", "contrast", "metric", "higher_is_better"], sort=False).delta.agg(
            mean="mean", seed_sd="std", n_seeds="count").reset_index()
        agg.to_csv(root / "contrasts_across_seed.csv", index=False)
        names = ("per_seed.csv", "across_seed.csv", "contrasts_per_seed.csv", "contrasts_across_seed.csv")
        write_json(root / "manifest.json", {"protocol": PROTOCOL, "kind": "summary", "sources": sources,
                   "arms": list(ALL_ARMS), "seeds": list(SEEDS), "comparison_id": next(iter(comparisons))[0],
                   "aggregation": "sample_direct_mean_then_equal_seed_mean", "test_waveform_read": False,
                   "files": {name: sha256_file(root / name) for name in names}})
    return Path(output).resolve()
