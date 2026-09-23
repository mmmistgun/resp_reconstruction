from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from resp_train.aligned_dual_view.artifacts import read_json, sha256_file, write_json
from resp_train.aligned_dual_view.config import ALL_VIEWS, SEEDS
from resp_train.aligned_dual_view.experiment import PRIMARY
from resp_train.aligned_dual_view.features import spec_digest
from resp_train.metrics.task import summarize_task_metrics

from .artifacts import common_manifest, output_directory, verified_artifact
from .contract import LOCK_PATH, load_lock, require_confirmation, source_config
from .evaluate import check_test_metrics, evaluation_root


def summarize_test(cache_root, output, *, lock_path=LOCK_PATH, confirm_research_test=False):
    require_confirmation(confirm_research_test)
    lock = load_lock(lock_path)
    cache = verified_artifact(cache_root, kind="test_cache", lock=lock)
    cfg = source_config(lock, next(iter(lock["entries"])))
    cache_sha = sha256_file(Path(cache_root) / "manifest.json")
    with output_directory(output, kind="test_summary", cfg=cfg, lock=lock) as (root, started):
        records, sources, comparisons = [], [], set()
        for name, entry in lock["entries"].items():
            attempts = list((evaluation_root(cache_root) / name).glob("attempt_*"))
            successes = [path for path in attempts if (path / "completed.json").exists()]
            unfinished = [path for path in attempts if not (path / "completed.json").exists()
                          and not (path / "failed.json").exists()]
            if len(successes) != 1 or unfinished:
                raise ValueError(f"{name} 必须恰有一个成功评价，且所有尝试均已闭合")
            source = successes[0]
            manifest = verified_artifact(source, kind="test_evaluation", lock=lock)
            if (manifest["entry"] != name or manifest["view"] != entry["view"]
                    or manifest["seed"] != entry["seed"] or manifest["selected_epoch"] != entry["selected_epoch"]
                    or manifest["checkpoint_sha256"] != entry["checkpoint_sha256"]
                    or manifest["count"] != lock["test_count"]
                    or manifest["cache_manifest_sha256"] != cache_sha or manifest["cache_id"] != cache["cache_id"]):
                raise ValueError("test 评价身份偏离冻结候选/cache")
            if (manifest.get("checkpoint_reselected") is not False or manifest.get("target_read") is not True
                    or manifest.get("inference_completed_before_target_read") is not True):
                raise ValueError("test 评价访问边界不合格")
            for filename, field in (("metrics.csv", "metrics_sha256"), ("metrics_summary.csv", "summary_sha256"),
                                    ("samples.json", "sample_records_file_sha256"), ("access.jsonl", "access_sha256")):
                if sha256_file(source / filename) != manifest[field]:
                    raise ValueError(f"test 评价产物哈希漂移: {filename}")
            events = [json.loads(line) for line in (source / "access.jsonl").read_text(encoding="utf-8").splitlines()]
            if [event["event"] for event in events] != [
                "input_only_inference_started", "input_only_inference_completed",
                "target_read_started", "target_read_completed",
            ]:
                raise ValueError("test access trace 阶段顺序错误")
            if (events[0].get("expected_count") != lock["test_count"]
                    or events[1].get("count") != lock["test_count"]
                    or events[2].get("inference_count") != lock["test_count"]
                    or events[3].get("count") != lock["test_count"]
                    or events[0].get("target_read") is not False or events[1].get("target_read") is not False):
                raise ValueError("test access trace 样本数或读取边界错误")
            samples = read_json(source / "samples.json")["samples"]
            if spec_digest(samples) != manifest["samples_sha256"]:
                raise ValueError("test sample 内容身份错误")
            metrics = pd.read_csv(source / "metrics.csv")
            row_ids = metrics.dataset_row_id.to_numpy(dtype="<i8")
            if (len(metrics) != lock["test_count"] or set(metrics.split) != {"test"}
                    or not np.all(np.diff(row_ids) > 0)
                    or hashlib.sha256(row_ids.tobytes()).hexdigest() != lock["test_row_ids_sha256"]
                    or set(metrics.method) != {f"adv_v1_{entry['view']}"}):
                raise ValueError("test 指标行集合、顺序或方法标签错误")
            check_test_metrics(metrics)
            saved = pd.read_csv(source / "metrics_summary.csv")
            expected = summarize_task_metrics(metrics)
            if list(saved.columns) != list(expected.columns) or saved.shape != expected.shape:
                raise ValueError("test summary schema 错误")
            if not np.allclose(saved.to_numpy(dtype=float), expected.to_numpy(dtype=float),
                               rtol=1e-12, atol=1e-12, equal_nan=True):
                raise ValueError("test summary 与逐 sample 指标不一致")
            keys = ("lock_id", "cache_id", "samples_sha256", "code_sha256", "precision", "batch_size")
            if manifest["code_sha256"] != manifest["execution_code_sha256"]:
                raise ValueError("test execution code identity 不一致")
            if manifest["comparison_id"] != spec_digest({key: manifest[key] for key in keys}):
                raise ValueError("test comparison ID 错误")
            comparisons.add(manifest["comparison_id"])
            records.append({"view": entry["view"], "seed": entry["seed"],
                            "selected_epoch": entry["selected_epoch"], **saved.iloc[0].to_dict()})
            sources.append({"root": str(source.resolve()), "manifest_sha256": sha256_file(source / "manifest.json")})
        if len(records) != 12 or len(comparisons) != 1:
            raise ValueError("test 矩阵不完整或 sample/precision/code 不一致")
        scores = pd.DataFrame(records).sort_values(["view", "seed"])
        scores.to_csv(root / "per_seed.csv", index=False)
        mean_columns = [name for name in scores.columns if name.endswith("_mean")]
        across, deltas = [], []
        for view in ALL_VIEWS:
            candidate = scores[scores.view == view].set_index("seed").loc[list(SEEDS)]
            for column in mean_columns:
                values = candidate[column].to_numpy(dtype=float)
                # 辅助指标有合法 NA 时保持三 seed 均值为 NA，显式报告有限 seed 数。
                across.append({"view": view, "metric": column.removesuffix("_mean"),
                               "mean": values.mean(), "seed_sd": values.std(ddof=1),
                               "n_seeds": 3, "finite_seed_count": int(np.isfinite(values).sum())})
            for anchor_view in ("joint", "waveform"):
                anchor = scores[scores.view == anchor_view].set_index("seed")
                for seed in SEEDS:
                    for metric in PRIMARY:
                        deltas.append({"view": view, "anchor": anchor_view, "seed": seed, "metric": metric,
                                       "candidate_minus_anchor": candidate.loc[seed, f"{metric}_mean"]
                                       - anchor.loc[seed, f"{metric}_mean"]})
        pd.DataFrame(across).to_csv(root / "across_seed.csv", index=False)
        pd.DataFrame(deltas).to_csv(root / "paired_deltas.csv", index=False)
        write_json(root / "manifest.json", {
            **common_manifest("test_summary", lock, started), "sources": sources,
            "comparison_id": next(iter(comparisons)), "views": list(ALL_VIEWS), "seeds": list(SEEDS),
            "aggregation": "sample_direct_mean_then_equal_seed_mean",
            "cache_id": cache["cache_id"], "cache_manifest_sha256": cache_sha,
            "test_input_array_read": False, "test_target_array_read": False,
            "files": {name: sha256_file(root / name) for name in ("per_seed.csv", "across_seed.csv", "paired_deltas.csv")},
        })
    return Path(output).resolve()
