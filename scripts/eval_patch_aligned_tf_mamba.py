"""固定九个 validation-selected checkpoint 的 research-test 入口。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import traceback
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from resp_train import patch_aligned_tf_training as training
from resp_train.crd import tf_v1_research_test_data as test_cache
from resp_train.crd.tf_v1_features import fixed_transform_spec
from resp_train.data.factory import build_window_data
from resp_train.data.index import filter_index
from resp_train.data.research_v2 import read_research_v2_index
from resp_train.engine import collect_predictions
from resp_train.metrics.task import evaluate_task_predictions
from resp_train.models.patch_aligned_tf_mamba import PatchAlignedTFMamba, PatchTFConfig
from resp_train.paper_evidence import w0_structural_factorial_v1 as sf
from resp_train.paper_evidence.w0_structural_factorial_v1_test import (
    IDENTITY_COLUMNS, TARGET_COLUMNS, check_test_rows, check_test_metrics, quality_flags,
)

PROTOCOL = "patch-aligned-tf-mamba-research-test-v1-20261006"
PROTOCOL_PATH = ROOT / "docs/experiments/patch_aligned_tf_mamba_research_test_v1_20261006.md"
TRAINING_SESSION_SHA256 = "ad34f3a03680cadd88000ee7bcf3c2a7053fd48d501cdb9153b2e4062f07ae3c"
VALIDATION_RECEIPT_SHA256 = "5394a9187db44588dbc1649f0c592982f742318b428c3d5a438607473996fd7f"
COUNT, SUBJECTS, SAMPLE_SEED = 2310, 8, 20260612
ROW_HASH = "184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8"
EVIDENCE_ROLE = "reused research/development evidence"
sha, read_json, write_json = training.sha, training.read_json, training.write_json


def require_confirmation(confirmed):
    if confirmed is not True:
        raise PermissionError("research-test 访问要求显式 --confirm-research-test")


def contained(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("产物路径越界")
    return path


def test_contract():
    return {"split": "test", "count": COUNT, "subjects": SUBJECTS, "sample_seed": SAMPLE_SEED,
        "row_hash": ROW_HASH, "batch_size": 64, "amp_dtype": "bfloat16", "include_test_only": True,
        "evidence_role": EVIDENCE_ROLE, "cache_root": str(test_cache.FROZEN_RESEARCH_TEST_CACHE_ROOT),
        "cache_manifest_sha256": test_cache.FROZEN_RESEARCH_TEST_CACHE_MANIFEST_SHA256,
        "cache_identity": test_cache.FROZEN_RESEARCH_TEST_CACHE_IDENTITY}


def code_identity():
    # 仅新增独立入口；训练 source_identity 所管理的文件保持不变。
    result = training.source_identity()
    for path in (Path(__file__), PROTOCOL_PATH, ROOT / "scripts/run_patch_tf_test_shard.sh",
                 ROOT / "tests/test_patch_aligned_tf_research_test.py"):
        result[str(path.relative_to(ROOT))] = sha(path)
    return result


def selected_epoch(history, completion):
    if [row["epoch"] for row in history] != list(range(1, completion["completed_epochs"] + 1)):
        raise ValueError("validation history 不完整或顺序错误")
    values = np.asarray([row["val_local_rr_mae"] for row in history], dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise FloatingPointError("validation selector 缺少有限完整轨迹")
    epoch = int(np.argmin(values)) + 1
    if epoch != completion["best_epoch"] or float(values[epoch - 1]) != completion["best_local_rr"]:
        raise ValueError("完成回执与完整 validation selector 不一致")
    return epoch


def prepare(training_session, output):
    """仅读取冻结的 development 产物，不打开 test manifest 或数组。"""
    session, root = Path(training_session).resolve(), Path(output).resolve()
    training.verify_session(session)
    if sha(session / "session.json") != TRAINING_SESSION_SHA256:
        raise ValueError("训练 session 不属于本轮已完成矩阵")
    if (root / "allowlist_receipt.json").exists():
        lock = load_allowlist(root)
        if lock["training_session"] != str(session):
            raise ValueError("allowlist 对应另一训练 session")
        return root
    root.mkdir(parents=True, exist_ok=True)
    with training.mutex(root / ".prepare.lock"):
        if set(p.name for p in root.iterdir()) - {".prepare.lock"}:
            raise FileExistsError("输出目录已有产物，请检查后使用独立路径")
        summaries = [p for p in (session / "summary").glob("*/receipt.json") if sha(p) == VALIDATION_RECEIPT_SHA256]
        if len(summaries) != 1:
            raise ValueError("缺少已冻结完整 validation 汇总")
        summary = read_json(summaries[0])
        for name, digest in summary["artifacts"].items():
            if sha(contained(summaries[0].parent, name)) != digest:
                raise ValueError("validation 汇总产物身份变化")
        source = read_json(ROOT / training.P2_LOCK_PATH)
        entries, development, row_identities, reference_frequencies = [], set(), None, None
        for cell in training.plan():
            directory = session / "cells" / cell["cell"]
            completion_path = directory / "completed.json"
            completion = read_json(completion_path)
            if sha(completion_path) != summary["sources"][cell["cell"]]:
                raise ValueError("cell 完成来源与 validation 汇总不一致")
            if (completion["patch_seconds"], completion["seed"]) != (cell["patch_seconds"], cell["seed"]):
                raise ValueError("完成 cell 标签不符")
            for name, digest in completion["artifacts"].items():
                if sha(contained(directory, name)) != digest:
                    raise ValueError(f"冻结产物身份变化: {cell['cell']}/{name}")
            cfg_path = directory / "config.yaml"
            cfg = OmegaConf.load(cfg_path)
            expected = training.config(cell["patch_seconds"], cell["seed"], directory, cfg.training.device)
            if training.config_identity(cfg) != training.config_identity(expected):
                raise ValueError("训练配置与矩阵不一致")
            final = torch.load(contained(directory, completion["final_checkpoint"]), map_location="cpu", weights_only=False)
            epoch = selected_epoch(final["history"], completion)
            if final["epoch"] != completion["completed_epochs"] or final["config"] != training.config_identity(cfg):
                raise ValueError("final checkpoint epoch/config 不符")
            del final
            checkpoint_path = contained(directory, completion["best_checkpoint"])
            best = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            training.finite_tree(best["model_state_dict"])
            if best["epoch"] != epoch or best["config"] != training.config_identity(cfg):
                raise ValueError("best checkpoint epoch/config 不符")
            frequencies = best["model_state_dict"]["condition.frequencies_hz"].cpu().numpy()
            if reference_frequencies is None:
                reference_frequencies = frequencies.copy()
            elif not np.array_equal(reference_frequencies, frequencies):
                raise ValueError("跨 cell H-CWT 频率坐标不一致")
            attempt = contained(directory, completion["metrics"]).parent
            current = {}
            for split, expected_count, expected_subjects in (("train", 10141, 32), ("val", 2675, 7)):
                rows = pd.read_csv(attempt / f"{split}_rows.csv")
                ids = rows.dataset_row_id.to_numpy(np.int64)
                digest = hashlib.sha256(np.sort(ids).tobytes()).hexdigest()
                if (len(rows) != expected_count or rows.samp_id.nunique() != expected_subjects
                        or set(rows.split) != {split} or len(np.unique(ids)) != len(ids)
                        or digest != source["cache_lock"]["row_identity"][f"{split}_row_content_sha256"]):
                    raise ValueError("development 行身份不符")
                development.update(rows.samp_id.astype(int))
                current[split] = sha(attempt / f"{split}_rows.csv")
            if row_identities is None:
                row_identities = current
            elif row_identities != current:
                raise ValueError("跨 cell development 样本不一致")
            entries.append({**cell, "selected_epoch": epoch,
                "checkpoint": {"path": str(checkpoint_path), "sha256": sha(checkpoint_path)},
                "config": {"path": str(cfg_path), "sha256": sha(cfg_path)},
                "training_config": OmegaConf.to_container(cfg, resolve=True),
                "completion": {"path": str(completion_path), "sha256": sha(completion_path)},
                "training_environment": read_json(directory / "environment.json")})
            del best
        sources = code_identity()
        with tarfile.open(root / "source_snapshot.tar.gz", "x:gz") as archive:
            for name in sources:
                archive.add(ROOT / name, arcname=name)
        if sources != code_identity():
            raise RuntimeError("allowlist 准备期间源码变化")
        payload = {"protocol": PROTOCOL, "training_session": str(session),
            "training_session_sha256": TRAINING_SESSION_SHA256,
            "validation_receipt_sha256": VALIDATION_RECEIPT_SHA256,
            "contract": test_contract(), "entries": entries, "development_subjects": sorted(development),
            "development_row_files": row_identities, "dataset_index": source["dataset_index"],
            "frequencies_hz": reference_frequencies.tolist(), "code_sha256": sources,
            "snapshot_sha256": sha(root / "source_snapshot.tar.gz"),
            "test_array_read": False, "test_manifest_read": False, "model_inference_used": False}
        write_json(root / "allowlist.json", payload)
        write_json(root / "allowlist_receipt.json", {"sha256": sha(root / "allowlist.json")})
    return root


def load_allowlist(root):
    root = Path(root)
    if sha(root / "allowlist.json") != read_json(root / "allowlist_receipt.json")["sha256"]:
        raise ValueError("allowlist SHA-256 不符")
    lock = read_json(root / "allowlist.json")
    if (lock["protocol"] != PROTOCOL or lock["contract"] != test_contract()
            or lock["training_session_sha256"] != TRAINING_SESSION_SHA256
            or lock["validation_receipt_sha256"] != VALIDATION_RECEIPT_SHA256
            or [{k: e[k] for k in ("patch_seconds", "seed", "cell")} for e in lock["entries"]] != training.plan()
            or lock["code_sha256"] != code_identity()
            or sha(root / "source_snapshot.tar.gz") != lock["snapshot_sha256"]):
        raise ValueError("allowlist 矩阵、合同或源码身份变化")
    return lock


class HTestReader:
    def __init__(self, checkpoint_frequencies, *, confirmed=False):
        require_confirmation(confirmed)
        self.source = test_cache.TfV1ResearchTestCacheReader(test_cache.FROZEN_RESEARCH_TEST_CACHE_ROOT,
            split="test", representations=("w",))
        if self.source.manifest["fixed_transform_spec"] != fixed_transform_spec():
            raise ValueError("test CWT 表示规格不符")
        actual = self.source._open("w_frequencies_hz.npy")
        if (actual.shape != (97,) or actual.dtype != np.float64 or not np.isfinite(actual).all()
                or np.any(actual <= 0) or np.any(np.diff(actual) < 0)):
            raise ValueError("test CWT 频率元数据不符")
        self.indices = np.flatnonzero((actual > .8) & (actual <= 8))
        if len(self.indices) != 41 or not np.array_equal(actual[self.indices], checkpoint_frequencies):
            raise ValueError("test 与 checkpoint 的 H-CWT 频率不一致")
        if self.source.arrays["w"].shape != (COUNT, 97, 360) or self.source.arrays["w"].dtype != np.float32:
            raise ValueError("test W cache shape/dtype 不符")

    def get(self, row_id):
        value = self.source.get(row_id)["w"]
        if not torch.isfinite(value).all():
            raise FloatingPointError("test W cache 包含非有限值")
        return {"w": value[self.indices].contiguous()}


class HTestDataset(Dataset):
    def __init__(self, base, reader):
        self.base, self.reader = base, reader

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = dict(self.base[index])
        item["tf"] = self.reader.get(item["meta"]["dataset_row_id"])
        return item


def guarded_batches(loader, rows):
    offset = 0
    for batch in loader:
        count = len(batch["x"])
        expected = rows.iloc[offset:offset + count]
        if count <= 0 or len(expected) != count or set(batch.get("tf", {})) != {"w"}:
            raise ValueError("test batch 数量/条件 keys 不符")
        for key in IDENTITY_COLUMNS:
            value = batch["meta"][key]
            value = value.cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
            if not np.array_equal(value, expected[key].to_numpy()):
                raise ValueError(f"test batch 身份/顺序不符: {key}")
        for value, shape in ((batch["x"], (count, 1, 18000)), (batch["target"], (count, 1, 18000)),
                             (batch["tf"]["w"], (count, 41, 360))):
            if value.shape != shape:
                raise ValueError("test batch shape 不符")
            if not torch.isfinite(value).all():
                raise FloatingPointError("test batch 包含非有限值")
        offset += count
        yield batch
    if offset != len(rows):
        raise ValueError("test loader 未完整覆盖样本")


def infer_metrics(model, loader, rows, cfg, device):
    predictions = collect_predictions(model, guarded_batches(loader, rows), device=device,
        max_windows=len(rows), use_amp=True)
    for key in ("r_tho_hat", "tho_ref"):
        if predictions[key].shape != (len(rows), 1, 18000) or not np.isfinite(predictions[key]).all():
            raise FloatingPointError("test 预测/参考 shape 或 finite 不符")
    metrics = evaluate_task_predictions(predictions, cfg, include_test_only=True, method=cfg.model.variant)
    summary = check_test_metrics(metrics, rows)
    return metrics, summary


def completed(root, parent, *, retry=False):
    paths = list(Path(parent).glob("attempt_*"))
    success = [p for p in paths if (p / "receipt.json").exists()]
    if len(success) > 1:
        raise ValueError("同一证据 identity 存在多个成功 attempt")
    if success:
        path = success[0]
        receipt = read_json(path / "receipt.json")
        if receipt["allowlist_sha256"] != sha(Path(root) / "allowlist.json"):
            raise ValueError("评价产物对应另一 allowlist")
        for name, digest in receipt["artifacts"].items():
            if sha(contained(path, name)) != digest:
                raise ValueError("评价产物身份变化")
        return path
    if paths and not retry:
        raise RuntimeError("已有未完成 attempt，检查后使用 --retry-failed")
    return None


@contextmanager
def attempt(root, parent):
    path = Path(parent) / f"attempt_{uuid4().hex}"
    path.mkdir(parents=True)
    write_json(path / "started.json", {"protocol": PROTOCOL, "command": sys.argv,
        "allowlist_sha256": sha(Path(root) / "allowlist.json")})
    try:
        yield path
        files = {str(p.relative_to(path)): sha(p) for p in path.iterdir() if p.is_file()}
        write_json(path / "receipt.json", {"allowlist_sha256": sha(Path(root) / "allowlist.json"), "artifacts": files})
    except BaseException:
        write_json(path / "failed.json", {"traceback": traceback.format_exc()})
        raise


def evaluate(root, cell_name, device, *, confirmed=False, retry=False):
    require_confirmation(confirmed)
    root = Path(root).resolve()
    lock = load_allowlist(root)
    entries = [e for e in lock["entries"] if e["cell"] == cell_name]
    if len(entries) != 1:
        raise ValueError("cell 不属于固定 allowlist")
    entry = entries[0]
    parent = root / "evaluation" / cell_name
    parent.mkdir(parents=True, exist_ok=True)
    with training.mutex(parent / ".run.lock"):
        previous = completed(root, parent, retry=retry)
        if previous:
            return previous
        with attempt(root, parent) as output:
            if torch.device(device).type != "cuda":
                raise ValueError("正式 test 入口要求 CUDA")
            torch.cuda.set_device(device)
            environment = training.runtime(device)
            if any(environment[k] != entry["training_environment"][k] for k in ("torch", "cuda", "packages", "gpu")):
                raise ValueError("test 环境与训练环境不一致")
            write_json(output / "environment.json", environment)
            for key in ("checkpoint", "config", "completion"):
                if sha(entry[key]["path"]) != entry[key]["sha256"]:
                    raise ValueError(f"test 来源变化: {key}")
            cfg = OmegaConf.load(entry["config"]["path"])
            if OmegaConf.to_container(cfg, resolve=True) != entry["training_config"]:
                raise ValueError("test 配置与 allowlist 不一致")
            if (cfg.data.test_split != "test" or cfg.data.max_test_windows is not None
                    or cfg.data.test_sample_seed != SAMPLE_SEED or cfg.training.batch_size != 64
                    or cfg.training.amp_dtype != "bfloat16" or not cfg.training.use_amp):
                raise ValueError("test 推理合同不符")
            checkpoint = torch.load(entry["checkpoint"]["path"], map_location="cpu", weights_only=False)
            if checkpoint["epoch"] != entry["selected_epoch"] or checkpoint["config"] != training.config_identity(cfg):
                raise ValueError("checkpoint 配置或选点不符")
            training.finite_tree(checkpoint["model_state_dict"])
            frequencies = checkpoint["model_state_dict"]["condition.frequencies_hz"].numpy()
            if not np.array_equal(frequencies, lock["frequencies_hz"]):
                raise ValueError("checkpoint 频率坐标不符")
            model = PatchAlignedTFMamba(frequencies, config=PatchTFConfig(**dict(cfg.model.patch_aligned_tf_mamba)),
                initialization_seed=int(cfg.model.initialization_seed))
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            del checkpoint
            cfg.training.device = device
            cfg.training.show_progress = False
            write_json(output / "access_started.json", {"split": "test", "count": COUNT,
                "evidence_role": EVIDENCE_ROLE, "checkpoint": entry["checkpoint"]})
            if sha(lock["dataset_index"]["path"]) != lock["dataset_index"]["sha256"]:
                raise ValueError("test dataset index 身份变化")
            audited = read_research_v2_index(cfg.data.dataset_root, cfg.data.index_csv, cfg)
            rows = filter_index(audited, cfg, split="test", max_windows=None,
                sample_strategy=cfg.data.test_sample_strategy, sample_seed=SAMPLE_SEED)
            check_test_rows(rows)
            if set(rows.samp_id.astype(int)) & set(lock["development_subjects"]):
                raise ValueError("test 与 development 受试者重叠")
            rows.to_csv(output / "test_rows.csv", index=False)
            # 数据工厂负责波形与参考；独立包装器负责冻结 H-CWT 条件。
            raw_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            raw_cfg.model.name = "crd_v1"
            raw_cfg.model.tf_representations = []
            raw_cfg.data.tf_cache_path = None
            raw_cfg.data.preload_windows = False
            bundle = build_window_data(raw_cfg, split="test", max_windows=None, shuffle=False, audited=audited,
                sample_strategy=cfg.data.test_sample_strategy, sample_seed=SAMPLE_SEED)
            check_test_rows(bundle.rows, rows)
            reader = HTestReader(frequencies, confirmed=confirmed)
            reader.source.verify_rows(rows.dataset_row_id)
            loader = DataLoader(HTestDataset(bundle.dataset, reader), batch_size=64, shuffle=False,
                num_workers=0, pin_memory=True, drop_last=False)
            OmegaConf.save(cfg, output / "evaluation_config.yaml")
            print(f"开始 research-test {cell_name}, epoch={entry['selected_epoch']}, {device}", flush=True)
            metrics, summary = infer_metrics(model, loader, rows, cfg, device)
            metrics.insert(0, "arm", f"p{entry['patch_seconds']}s")
            metrics.insert(1, "patch_seconds", entry["patch_seconds"])
            metrics.insert(2, "seed", entry["seed"])
            metrics.to_csv(output / "metrics.csv", index=False)
            summary.to_csv(output / "metrics_summary.csv", index=False)
            write_json(output / "evaluation.json", {"cell": cell_name, "selected_epoch": entry["selected_epoch"],
                "checkpoint_sha256": entry["checkpoint"]["sha256"], "rows": COUNT, "subjects": SUBJECTS,
                "evidence_role": EVIDENCE_ROLE, **quality_flags(metrics)})
            load_allowlist(root)
    del model, bundle, reader, loader, metrics
    gc.collect()
    torch.cuda.empty_cache()
    return output


def summary_tables(frame):
    expected = {(f"p{c['patch_seconds']}s", c["seed"]) for c in training.plan()}
    if set(frame[["arm", "seed"]].drop_duplicates().itertuples(index=False, name=None)) != expected:
        raise ValueError("汇总要求完整九组矩阵")
    reference, records, native = None, [], []
    for (arm, seed), group in frame.groupby(["arm", "seed"], sort=False):
        group = group.reset_index(drop=True)
        summary = check_test_metrics(group, group[list(IDENTITY_COLUMNS)])
        identity = group[[*IDENTITY_COLUMNS, *TARGET_COLUMNS]]
        if reference is None:
            reference = identity
        elif not identity.equals(reference):
            raise ValueError("跨 cell 样本或 target 资格变化")
        summary.insert(0, "arm", arm)
        summary.insert(1, "seed", seed)
        native.append(summary)
        for metric in sf.PRIMARY:
            values = group.loc[sf._metric_mask(group, metric), metric].to_numpy(float)
            if not len(values) or not np.isfinite(values).all():
                raise FloatingPointError("主指标缺失或非有限")
            records.append({"arm": arm, "seed": seed, "metric": metric, "mean": values.mean(), "n": len(values)})
    per_seed = pd.DataFrame(records)
    across = per_seed.groupby(["arm", "metric"])["mean"].agg(seed_mean="mean", seed_sd="std").reset_index()
    indexed = per_seed.set_index(["arm", "seed", "metric"])["mean"]
    deltas = []
    for arm in ("p1s", "p4s"):
        for seed in training.SEEDS:
            for metric in sf.PRIMARY:
                delta = indexed[arm, seed, metric] - indexed["p2s", seed, metric]
                deltas.append({"arm": arm, "reference": "p2s", "seed": seed, "metric": metric,
                    "delta": delta, "improvement": delta if metric == sf.PCC else -delta})
    subjects = sf.subject_stratified_metrics(frame)
    return {"per_seed": per_seed, "across_seed": across, "paired_delta": pd.DataFrame(deltas),
        "native_summary": pd.concat(native, ignore_index=True), "per_subject": subjects,
        "subject_macro": subjects.groupby(["arm", "seed", "metric"])["mean"].mean().reset_index(),
        "local_rr_tail": sf.local_rr_tail_summary(frame), "denominators": sf.metric_denominators(frame)}


def summarize(root, *, confirmed=False, retry=False):
    require_confirmation(confirmed)
    root = Path(root)
    lock = load_allowlist(root)
    parent = root / "summary"
    parent.mkdir(exist_ok=True)
    with training.mutex(parent / ".run.lock"):
        previous = completed(root, parent, retry=retry)
        if previous:
            return previous
        frames, sources = [], {}
        for entry in lock["entries"]:
            result = completed(root, root / "evaluation" / entry["cell"])
            if result is None:
                raise RuntimeError("九组 test 尚未全部完成")
            receipt = read_json(result / "evaluation.json")
            if receipt["cell"] != entry["cell"] or receipt["checkpoint_sha256"] != entry["checkpoint"]["sha256"]:
                raise ValueError("test cell 来源不符")
            frame = pd.read_csv(result / "metrics.csv")
            if not frame.seed.eq(entry["seed"]).all() or not frame.patch_seconds.eq(entry["patch_seconds"]).all():
                raise ValueError("test metrics cell 标签不符")
            frames.append(frame)
            sources[entry["cell"]] = sha(result / "receipt.json")
        tables = summary_tables(pd.concat(frames, ignore_index=True))
        with attempt(root, parent) as output:
            for name, frame in tables.items():
                frame.to_csv(output / f"{name}.csv", index=False)
            write_json(output / "sources.json", sources)
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_cmd = sub.add_parser("prepare")
    prepare_cmd.add_argument("--training-session", type=Path, required=True)
    prepare_cmd.add_argument("--output", type=Path, required=True)
    for command in ("evaluate", "status", "summarize"):
        p = sub.add_parser(command)
        p.add_argument("--allowlist", type=Path, required=True)
        if command != "status":
            p.add_argument("--confirm-research-test", action="store_true")
            p.add_argument("--retry-failed", action="store_true")
        if command == "evaluate":
            p.add_argument("--device", default="cuda:0")
            p.add_argument("--shard-index", type=int, default=0)
            p.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.command == "prepare":
        result = str(prepare(args.training_session, args.output))
    elif args.command == "status":
        lock = load_allowlist(args.allowlist)
        result = [{"cell": e["cell"], "status": "completed" if completed(args.allowlist,
            args.allowlist / "evaluation" / e["cell"], retry=True) else "pending_or_incomplete"} for e in lock["entries"]]
    elif args.command == "summarize":
        result = str(summarize(args.allowlist, confirmed=args.confirm_research_test, retry=args.retry_failed))
    else:
        require_confirmation(args.confirm_research_test)
        result = [str(evaluate(args.allowlist, c["cell"], args.device, confirmed=True, retry=args.retry_failed))
                  for c in training.plan(args.shard_index, args.shard_count)]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
