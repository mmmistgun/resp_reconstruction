"""M4-S1固定84-checkpoint的独立research-test入口；保持训练源码冻结。"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from resp_train import patch_aligned_tf_training as legacy
from resp_train.data.cache import WholeNightCache
from resp_train.data.factory import build_window_data
from resp_train.engine import collect_predictions
from resp_train.metrics.task import evaluate_task_predictions
from resp_train.paper_evidence import w0_structural_factorial_v1 as sf
from resp_train.paper_evidence.cwt_time_frequency_v1.features import transform
from scripts import m4_s1_cwt_parameters_artifacts as io
from scripts import m4_s1_cwt_parameters_data as data_ops
from scripts import m4_s1_cwt_parameters_runtime as runtime
from scripts import m4_s1_cwt_parameters_spec as spec
from scripts import m4_s1_cwt_parameters_summary as stats
from scripts.m4_s1_cwt_parameters_model import build_model
from scripts.run_m4_s1_receipt_compatibility import canonical_stage_bindings

PROTOCOL = "m4-s1-research-test-v1-20261022"
PROTOCOL_PATH = spec.ROOT / "docs/experiments/m4_s1_research_test_v1_protocol_20261022.md"
TRAIN_SHA = "87b201abb30dea105ca8c9ada5c2f48a33e1f5ebd47b78af11276f67e93740cc"
SUMMARY_SHA = "ea6b3a8767f8e1ca8f680ae255fc2aec5369489d57a9c6e5d7f49b7b411628a7"
TEST_ROW_HASH = "184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8"


def guard(confirmed):
    if confirmed is not True:
        raise PermissionError("需要本次--confirm-research-test")


def source(path):
    return {"path": str(Path(path).resolve()), **io.identity(path)}


def verify_source(record):
    io.verify(record["path"], {k: record[k] for k in ("sha256", "size_bytes")})


def prepare(training, output):
    """先核验完整development来源，再写一次allowlist；此函数不读取test。"""
    training, output = Path(training).resolve(), Path(output).resolve()
    frozen = io.load_session(training)
    if io.sha(training / "session.json") != TRAIN_SHA:
        raise ValueError("不是本轮冻结训练session")
    with canonical_stage_bindings():
        completion_sources = {c["cell"]: io.identity(training / "cells" / c["cell"] / "completed.json") for c in spec.plan()}
        summary = io.completed(training / "summary", io.binding(training, "summary", sources=completion_sources))
        development, metadata = data_ops.data_receipt(training, verify_inputs=False)
    if summary is None or io.sha(summary / "manifest.json") != SUMMARY_SHA:
        raise ValueError("完整validation汇总身份不符")
    output.mkdir(parents=True, exist_ok=False)
    entries = []
    for cell in spec.plan():
        arm, seed, name = cell["arm"], cell["seed"], cell["cell"]
        receipt = runtime.verified_cell(training, arm, seed)
        base = training / "cells" / name
        cfg = OmegaConf.load(base / "config.yaml")
        runtime.verify_training_result(base, receipt, cfg)
        entries.append({"arm": arm, "seed": seed, "cell": name,
            "epoch": receipt["best_epoch"], "parameters": receipt["parameters"],
            "representation": spec.ARMS[arm].representation,
            "checkpoint": source(base / receipt["best_checkpoint"]),
            "config": source(base / "config.yaml"), "completion": source(base / "completed.json"),
            "validation_metrics": source(base / receipt["metrics"])})
        print(f"freeze {name} epoch={receipt['best_epoch']}", flush=True)
    # 保存评价依赖的完整本地Python/配置快照；不依赖后续工作树修改。
    paths = {p for folder in ("scripts", "resp_train", "configs")
             for p in (spec.ROOT / folder).rglob("*") if p.is_file() and p.suffix in (".py", ".yaml", ".yml")}
    paths.update(spec.ROOT / p for p in frozen["code_sha256"])
    paths.update((PROTOCOL_PATH, spec.ROOT / "tests/test_eval_m4_s1_research_test.py"))
    sources = {str(p.relative_to(spec.ROOT)): io.identity(p) for p in sorted(paths)}
    with tarfile.open(output / "source_snapshot.tar.gz", "x:gz") as archive:
        for relative in sources:
            archive.add(spec.ROOT / relative, arcname=relative)
    for relative, identity in sources.items():
        io.verify(spec.ROOT / relative, identity)
    for name in ("train_rows.csv", "val_rows.csv"):
        shutil.copyfile(development / name, output / name)
    lock = {"protocol": PROTOCOL, "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv, "training_session": source(training / "session.json"),
        "validation_summary": source(summary / "manifest.json"), "validation_summary_path": str(summary),
        "development_metadata": metadata, "entries": entries,
        "development_rows": {s: io.identity(output / f"{s}_rows.csv") for s in ("train", "val")},
        "representations": frozen["representations"], "code": sources, "dependencies": io.dependencies(),
        "snapshot": io.identity(output / "source_snapshot.tar.gz"),
        "comparisons": stats.comparisons(), "effects": stats.effect_coefficients(),
        "test_contract": {"count": 2310, "subjects": 8, "row_sha256": TEST_ROW_HASH,
                          "batch_size": 32, "amp": "BF16", "threshold": .01},
        "test_data_accessed": False, "default_model_changed": False,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=spec.ROOT, text=True).strip()}
    io.write_json(output / "allowlist.json", lock)
    io.write_json(output / "allowlist_receipt.json", io.identity(output / "allowlist.json"))
    pd.DataFrame([{"cell": e["cell"], "arm": e["arm"], "seed": e["seed"], "epoch": e["epoch"],
                   "checkpoint": e["checkpoint"]["path"], "sha256": e["checkpoint"]["sha256"]}
                  for e in entries]).to_csv(output / "checkpoint_list.csv", index=False)
    return output


@lru_cache(maxsize=2)
def load(root):
    root = Path(root)
    io.verify(root / "allowlist.json", io.read_json(root / "allowlist_receipt.json"))
    lock = io.read_json(root / "allowlist.json")
    expected = {(p["arm"], p["seed"]) for p in spec.plan()}
    if (lock["protocol"] != PROTOCOL or len(lock["entries"]) != 84
            or {(e["arm"], e["seed"]) for e in lock["entries"]} != expected
            or lock["dependencies"] != io.dependencies()):
        raise ValueError("allowlist矩阵/协议/环境变化")
    for name, identity in lock["code"].items():
        io.verify(spec.ROOT / name, identity)
    io.verify(root / "source_snapshot.tar.gz", lock["snapshot"])
    for s, identity in lock["development_rows"].items():
        io.verify(root / f"{s}_rows.csv", identity)
    verify_source(lock["training_session"])
    verify_source(lock["validation_summary"])
    return lock


def key(root, phase, **fields):
    return {"protocol": PROTOCOL, "allowlist_sha256": io.sha(Path(root) / "allowlist.json"),
            "phase": phase, **fields}


def completed(root, phase, **fields):
    parent = Path(root) / phase
    for value in fields.values():
        parent = parent / str(value)
    return io.completed(parent, key(root, phase, **fields))


def get_data(root):
    path = completed(root, "data")
    if path is None:
        raise RuntimeError("缺少test数据准备")
    return path, io.read_json(path / "data.json")


def raw_bundle(cfg):
    return build_window_data(cfg, split="test", max_windows=None, shuffle=False,
        sample_strategy=cfg.data.test_sample_strategy, sample_seed=cfg.data.test_sample_seed)


def check_rows(rows, development):
    if (len(rows) != 2310 or rows.samp_id.nunique() != 8 or set(rows.split) != {"test"}
            or rows.dataset_row_id.duplicated().any()
            or data_ops.array_hash(rows.dataset_row_id.to_numpy(np.int64)) != TEST_ROW_HASH):
        raise ValueError("test样本身份/顺序/分母变化")
    if set(rows.samp_id.astype(int)) & set(development):
        raise ValueError("test与development受试者重叠")


def prepare_data(root, confirmed=False, retry=False):
    guard(confirmed)
    lock = load(root)
    prior = completed(root, "data")
    if prior:
        data_ops.verify_sources(io.read_json(prior / "data.json"))
        return prior
    with io.attempt(Path(root) / "data", key(root, "data"), retry_failed=retry) as out:
        io.write_json(out / "access.json", {"split": "test", "confirmed": True})
        cfg = OmegaConf.load(lock["entries"][0]["config"]["path"])
        index = (Path(cfg.data.dataset_root) / cfg.data.index_csv).resolve()
        verify_source(lock["development_metadata"]["index"])
        if index != Path(lock["development_metadata"]["index"]["path"]).resolve():
            raise ValueError("test index与训练不同")
        bundle = raw_bundle(cfg)
        rows = bundle.rows.reset_index(drop=True)
        dev = set()
        for s in ("train", "val"):
            dev.update(pd.read_csv(Path(root) / f"{s}_rows.csv").samp_id.astype(int))
        check_rows(rows, dev)
        rows.to_csv(out / "test_rows.csv", index=False)
        resolver = WholeNightCache(index)
        names = {resolver.resolve(str(v)).resolve() for col in ("source_npz", "target_source_npz") for v in rows[col].unique()}
        metadata = {"index": source(index), "source_files": {str(p): io.identity(p) for p in sorted(names)},
                    "row_sha256": TEST_ROW_HASH, "count": 2310, "subjects": 8}
        reference = np.lib.format.open_memmap(out / "test_reference.npy", mode="w+", dtype=np.float32, shape=(2310, 1, 18000))
        for i in range(len(bundle.dataset)):
            item = bundle.dataset[i]
            for field in ("x", "target"):
                if item[field].shape != (1, 18000) or not torch.isfinite(item[field]).all():
                    raise FloatingPointError("输入/target shape或有限性不符")
            if item["meta"]["dataset_row_id"] != int(rows.iloc[i].dataset_row_id):
                raise ValueError("公共target顺序变化")
            reference[i] = item["target"].numpy()
        reference.flush()
        del reference
        data_ops.verify_sources(metadata)
        io.write_json(out / "data.json", metadata)
    return out


def build_cache(root, rep_id, confirmed=False, retry=False):
    guard(confirmed)
    lock = load(root)
    rep = lock["representations"][rep_id]
    # B被多个消融共享；等待另一worker完成缓存后仅作核验与只读复用。
    with io.mutex(Path(root) / "cache" / f".{rep_id}.build.lock", blocking=True):
        prior = completed(root, "cache", representation=rep_id)
        if prior:
            return prior
        data_path, metadata = get_data(root)
        data_ops.verify_sources(metadata)
        with io.attempt(Path(root) / "cache" / rep_id, key(root, "cache", representation=rep_id), retry_failed=retry) as out:
            io.write_json(out / "access.json", {"split": "test", "confirmed": True, "target_read": False})
            rows = pd.read_csv(data_path / "test_rows.csv")
            resolver = WholeNightCache(metadata["index"]["path"])
            array = np.lib.format.open_memmap(out / "test_w.npy", mode="w+", dtype=np.float32, shape=(2310, *rep["shape"]))
            for i, row in enumerate(rows.itertuples(index=False)):
                signal = resolver.get_arrays(str(row.source_npz), [str(row.bcg_signal_key)])[str(row.bcg_signal_key)]
                value = transform(signal[int(row.window_start_sample):int(row.window_end_sample)], rep)
                if value.shape != tuple(rep["shape"]) or not np.isfinite(value).all():
                    raise FloatingPointError("CWT shape或有限性不符")
                array[i] = value
                if i % 500 == 0:
                    print(f"cache {rep_id} {i}/2310", flush=True)
            array.flush()
            del array
            np.save(out / "test_row_ids.npy", rows.dataset_row_id.to_numpy(np.int64), allow_pickle=False)
            io.write_json(out / "cache.json", {"representation": rep, "split": "test", "row_sha256": TEST_ROW_HASH,
                "data_manifest": source(data_path / "manifest.json"), "finite": True})
            data_ops.verify_sources(metadata)
        return out


class TestReader:
    def __init__(self, cache, rep, ids):
        metadata = io.read_json(Path(cache) / "cache.json")
        if metadata["representation"] != rep or metadata["split"] != "test" or metadata["finite"] is not True:
            raise ValueError("test缓存表示合同变化")
        self.ids = np.load(Path(cache) / "test_row_ids.npy", allow_pickle=False)
        self.values = np.load(Path(cache) / "test_w.npy", mmap_mode="r", allow_pickle=False)
        if (self.ids.dtype != np.int64 or not np.array_equal(self.ids, ids)
                or self.values.dtype != np.float32 or self.values.shape != (len(ids), *rep["shape"])):
            raise ValueError("test缓存身份/shape/dtype变化")

    def get(self, index, row_id):
        if int(self.ids[index]) != int(row_id):
            raise ValueError("CWT/batch样本错配")
        value = np.array(self.values[index], copy=True)
        if not np.isfinite(value).all():
            raise FloatingPointError("CWT非有限")
        return {"w": torch.from_numpy(value)}


def guarded_batches(loader, rows, shape):
    offset = 0
    for batch in loader:
        n = len(batch["x"])
        if n <= 0 or offset + n > len(rows):
            raise ValueError("batch分母错误")
        for col in ("dataset_row_id", "samp_id", "split"):
            value = batch["meta"][col]
            value = value.cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
            if not np.array_equal(value, rows.iloc[offset:offset+n][col].to_numpy()):
                raise ValueError(f"batch身份/顺序变化: {col}")
        tensors = [(batch["x"], (n, 1, 18000)), (batch["target"], (n, 1, 18000))]
        if set(batch.get("tf", {})) != ({"w"} if shape else set()):
            raise ValueError("CWT条件keys错误")
        if shape:
            tensors.append((batch["tf"]["w"], (n, *shape)))
        for value, expected in tensors:
            if tuple(value.shape) != expected or not torch.isfinite(value).all():
                raise FloatingPointError("batch shape/有限性错误")
        offset += n
        yield batch
    if offset != len(rows):
        raise ValueError("未覆盖完整test")


def evaluate(root, entry, device, confirmed=False, retry=False):
    guard(confirmed)
    lock = load(root)
    name, arm, seed = entry["cell"], entry["arm"], entry["seed"]
    prior = completed(root, "evaluation", cell=name)
    if prior:
        return prior
    runtime.require_cuda(device)
    if torch.cuda.get_device_name(torch.device(device)) != "NVIDIA GeForce RTX 4070 Ti SUPER":
        raise ValueError("GPU型号与本轮训练合同不同")
    rep_id = entry["representation"]
    cache = build_cache(root, rep_id, confirmed, retry) if rep_id else None
    data_path, metadata = get_data(root)
    data_ops.verify_sources(metadata)
    with io.attempt(Path(root) / "evaluation" / name, key(root, "evaluation", cell=name), retry_failed=retry) as out:
        started = time.monotonic()
        io.write_json(out / "access.json", {"split": "test", "confirmed": True, "entry": entry})
        for field in ("checkpoint", "config", "completion"):
            verify_source(entry[field])
        cfg = OmegaConf.load(entry["config"]["path"])
        saved = torch.load(entry["checkpoint"]["path"], map_location="cpu", weights_only=False)
        legacy.finite_tree(saved)
        if saved["config"] != legacy.config_identity(cfg) or saved["epoch"] != entry["epoch"]:
            raise ValueError("checkpoint配置/epoch不符")
        cfg.training.device = device
        cfg.outputs.run_root = str(out)
        OmegaConf.save(cfg, out / "evaluation_config.yaml")
        model = build_model(cfg).to(device).eval()
        model.load_state_dict(saved["model_state_dict"], strict=True)
        if sum(p.numel() for p in model.parameters()) != entry["parameters"]:
            raise ValueError("模型参数量变化")
        bundle = raw_bundle(cfg)
        rows = pd.read_csv(data_path / "test_rows.csv")
        if not bundle.rows[list(data_ops.ROW_COLUMNS)].astype(str).reset_index(drop=True).equals(rows[list(data_ops.ROW_COLUMNS)].astype(str)):
            raise ValueError("test完整样本元数据漂移")
        rep = lock["representations"][rep_id] if rep_id else None
        if rep:
            reader = TestReader(cache, rep, rows.dataset_row_id.to_numpy(np.int64))
            dataset = data_ops.ConditionDataset(bundle.dataset, reader)
            bundle = replace(bundle, dataset=dataset,
                loader=DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True, drop_last=False))
        prediction = collect_predictions(model, guarded_batches(bundle.loader, rows, rep["shape"] if rep else None),
            device=torch.device(device), max_windows=2310, use_amp=True)
        for field in ("r_tho_hat", "tho_ref"):
            if prediction[field].shape != (2310, 1, 18000) or not np.isfinite(prediction[field]).all():
                raise FloatingPointError("预测/target shape或有限性错误")
        if (not np.array_equal(prediction["dataset_row_id"], rows.dataset_row_id.to_numpy(np.int64))
                or not np.array_equal(prediction["tho_ref"], np.load(data_path / "test_reference.npy", mmap_mode="r", allow_pickle=False))):
            raise ValueError("预测顺序/公共target不一致")
        metrics = evaluate_task_predictions(prediction, cfg, include_test_only=True, method=spec.ARMS[arm].name)
        sf.validate_metrics(metrics, rows)
        metrics.insert(0, "arm", arm)
        metrics.insert(1, "seed", seed)
        metrics.insert(2, "evaluation_split", "test")
        metrics.to_csv(out / "metrics.csv", index=False)
        np.save(out / "prediction.npy", prediction["r_tho_hat"].astype(np.float32), allow_pickle=False)
        rows.to_csv(out / "test_rows.csv", index=False)
        io.write_json(out / "reference_source.json", source(data_path / "test_reference.npy"))
        io.write_json(out / "environment.json", {"dependencies": io.dependencies(), "gpu": torch.cuda.get_device_name(torch.device(device)),
            "device": device, "torch_cuda": torch.version.cuda, "command": sys.argv, "elapsed_seconds": time.monotonic()-started})
        io.write_json(out / "result.json", {"cell": name, "epoch": entry["epoch"], "parameters": entry["parameters"],
            "count": len(metrics), "subjects": metrics.samp_id.nunique(),
            "cache_manifest": source(cache / "manifest.json") if cache else None})
    print(f"completed {name}", flush=True)
    return out


def summary_tables(frame):
    """test独立检查split与分母；统计运算复用冻结validation的配对/效应函数。"""
    expected = {(p["arm"], p["seed"]) for p in spec.plan()}
    if set(frame[["arm", "seed"]].drop_duplicates().itertuples(index=False, name=None)) != expected:
        raise ValueError("汇总必须完整84-cell")
    records, first = [], None
    for (arm, seed), group in frame.groupby(["arm", "seed"], sort=False):
        group = group.reset_index(drop=True)
        check_rows(group, set())
        sf.validate_metrics(group, group[["dataset_row_id", "samp_id", "split"]])
        columns = ["dataset_row_id", "samp_id", "split"] + sorted(c for c in group if c.endswith("_target_eligible") or c == "local_rr_target_eligible_windows")
        identity = group[columns]
        if first is None:
            first = identity
        elif not first.equals(identity):
            raise ValueError("跨cell样本身份/顺序/target资格变化")
        for metric in sf.PRIMARY:
            values = group.loc[sf._metric_mask(group, metric), metric].to_numpy(float)
            if not len(values) or not np.isfinite(values).all():
                raise FloatingPointError("主指标缺少有限有效值")
            records.append({"arm": arm, "name": spec.ARMS[arm].name, "seed": seed, "metric": metric,
                            "mean": float(values.mean()), "n": len(values)})
    per_seed = pd.DataFrame(records)
    subjects = sf.subject_stratified_metrics(frame)
    macro = subjects.groupby(["arm", "seed", "metric"])["mean"].mean().reset_index()
    subject_pairs = stats.paired_table(subjects, subject=True)
    tables = {"per_seed": per_seed,
        "across_seed": per_seed.groupby(["arm", "name", "metric"])["mean"].agg(seed_mean="mean", seed_sd="std").reset_index(),
        "per_subject": subjects, "subject_macro": macro,
        "subject_macro_across_seed": macro.groupby(["arm", "metric"])["mean"].agg(seed_mean="mean", seed_sd="std").reset_index(),
        "per_subject_paired_delta": subject_pairs, "per_subject_paired_summary": stats.pair_summary(subject_pairs, subject=True),
        "local_rr_tail": sf.local_rr_tail_summary(frame), "denominators": sf.metric_denominators(frame)}
    for prefix, means in (("", per_seed), ("subject_macro_", macro)):
        pairs = stats.paired_table(means)
        tables[prefix+"paired_delta"] = pairs
        tables[prefix+"paired_summary"] = stats.pair_summary(pairs)
        tables[prefix+"effects"], tables[prefix+"effect_summary"] = stats.effect_tables(means)
    return tables


def summarize(root, confirmed=False, retry=False):
    guard(confirmed)
    lock = load(root)
    prior = completed(root, "summary")
    if prior:
        return prior
    sources = []
    # 首先核验全矩阵成功，再读取任何指标；partial结果不进入结论。
    for entry in lock["entries"]:
        path = completed(root, "evaluation", cell=entry["cell"])
        if path is None:
            raise RuntimeError(f"84矩阵未完成: {entry['cell']}")
        sources.append(path)
    tables = summary_tables(pd.concat([pd.read_csv(p / "metrics.csv") for p in sources], ignore_index=True))
    val_path = Path(lock["validation_summary_path"])
    io.verify_stage(val_path)
    with io.attempt(Path(root) / "summary", key(root, "summary"), retry_failed=retry) as out:
        for name, table in tables.items():
            table.to_csv(out / f"{name}.csv", index=False)
        lines = ["# M4-S1完整84-checkpoint research-test", "", "2310窗口/8受试者；84项全部完成后汇总。窗口加权为主，受试者宏平均为诊断。", "训练seed样本SD；delta=候选−参照；误差负值/PCC正值为改善；原生单位0.01为描述性阈值。", "test为reused research/development evidence；两split人群不同；默认模型未改变。", ""]
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for prefix, title in (("", "窗口加权"), ("subject_macro_", "受试者宏平均")):
            for suffix in ("across_seed", "paired_summary", "effect_summary", "paired_delta", "effects"):
                name = prefix + suffix
                val = pd.read_csv(val_path / f"{name}.csv")
                pd.concat([val.assign(evaluation_split="validation"), tables[name].assign(evaluation_split="test")], ignore_index=True).to_csv(out / f"cross_split_{name}.csv", index=False)
            values = tables[prefix+"across_seed"].set_index(["arm", "metric"])
            lines += [f"## {title}", "", "| 配置 | " + " | ".join(sf.PRIMARY) + " |", "|---|" + "---:|" * len(sf.PRIMARY)]
            for arm in spec.ARMS:
                lines.append("| " + arm + " | " + " | ".join(f"{values.loc[(arm,m),'seed_mean']:.6f} ± {values.loc[(arm,m),'seed_sd']:.6f}" for m in sf.PRIMARY) + " |")
            lines += [""]
            val = pd.read_csv(val_path / f"{prefix}paired_summary.csv")
            test = tables[prefix+"paired_summary"]
            merged = val.merge(test, on=["group", "comparison", "arm", "reference", "metric"], suffixes=("_validation", "_test"), validate="one_to_one")
            merged.to_csv(out / f"{prefix}validation_test_comparison.csv", index=False)
            arms = [a for a in spec.ARMS if a != "B"]
            fig, axes = plt.subplots(1, 5, figsize=(19, 11), sharey=True, constrained_layout=True)
            for axis, metric in zip(axes, sf.PRIMARY):
                part = merged[(merged.group == "baseline") & (merged.metric == metric)].set_index("arm").loc[arms]
                for split, shift in (("validation", -.13), ("test", .13)):
                    axis.errorbar(part[f"delta_mean_{split}"], np.arange(len(arms))+shift, xerr=part[f"delta_sd_{split}"], fmt="o", markersize=3, label=split)
                axis.axvline(0, color="gray", linewidth=.7)
                axis.set_title(metric.replace("_", " "), fontsize=8)
                axis.set_yticks(np.arange(len(arms)), arms, fontsize=7)
                axis.grid(axis="x", alpha=.2)
            axes[0].invert_yaxis()
            axes[0].legend()
            fig.savefig(out / f"{prefix}validation_test_deltas.png", dpi=150)
            plt.close(fig)
        (out / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
        io.write_json(out / "sources.json", {"allowlist": source(Path(root) / "allowlist.json"),
            "evaluations": [source(p / "manifest.json") for p in sources], "validation": lock["validation_summary"]})
    return out


def worker(root, device, shard, shards, confirmed=False, retry=False):
    guard(confirmed)
    lock = load(root)
    if not 0 <= shard < shards <= 84:
        raise ValueError("分片参数非法")
    # B共享缓存先由需要它的worker生成，独立互斥保证不重复。
    with io.mutex(Path(root) / f".device_{device.replace(':', '_')}.lock"):
        for entry in lock["entries"][shard::shards]:
            evaluate(root, entry, device, confirmed, retry)
    print(f"worker {shard}/{shards} completed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "prepare-data", "worker", "summarize", "status"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--confirm-research-test", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.training, args.root))
    elif args.command == "prepare-data":
        print(prepare_data(args.root, args.confirm_research_test, args.retry_failed))
    elif args.command == "worker":
        worker(args.root, args.device, args.shard, args.shards, args.confirm_research_test, args.retry_failed)
    elif args.command == "summarize":
        print(summarize(args.root, args.confirm_research_test, args.retry_failed))
    else:
        lock = load(args.root)
        print({"planned": len(lock["entries"]), "completed": sum(bool(list((args.root / "evaluation" / e["cell"]).glob("attempt_*/receipt.json"))) for e in lock["entries"]),
               "failed": len(list((args.root / "evaluation").glob("*/attempt_*/failed.json")))})


if __name__ == "__main__":
    main()
