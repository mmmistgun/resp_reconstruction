"""固定 validation 选点的 BCG-CWT research-test 评价入口。"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from resp_train import bcg_cwt_training as training
from resp_train.crd import tf_v1_research_test_data as cache
from resp_train.crd.tf_v1_features import fixed_transform_spec
from resp_train.data.factory import build_window_data
from resp_train.data.index import filter_index
from resp_train.data.research_v2 import read_research_v2_index
from resp_train.engine import collect_predictions
from resp_train.metrics.task import evaluate_task_predictions
from resp_train.models.bcg_cwt_patch_mamba import BCGCWTPatchMamba, BCGCWTConfig, environment_versions
from resp_train.paper_evidence.w0_structural_factorial_v1_test import (
    COUNT, SUBJECTS, TEST_SAMPLE_SEED, ROW_ORDER_SHA256, TARGET_COLUMNS, IDENTITY_COLUMNS,
    check_test_rows, check_test_metrics, guarded_batches, quality_flags,
)

PROTOCOL = "bcg-cwt-patch-mamba-research-test-v1-20261015"
PROTOCOL_PATH = ROOT / "docs/experiments/bcg_cwt_patch_mamba_research_test_v1_20261015.md"
sha, read_json, write_json = training.sha256, training.read_json, training.write_json


def require_confirmation(confirmed):
    if confirmed is not True:
        raise PermissionError("本模型 research-test 要求 --confirm-research-test")


def select_entries(records, completion, directories):
    if [r["epoch"] for r in records] != list(range(1, completion["epochs"] + 1)):
        raise ValueError("训练 history 不完整")
    entries = []
    for role, values, epoch_key, score_key in (
        ("best_rr", [r["val_local_rr_mae"] for r in records], "best_epoch", "best_local_rr_mae"),
        ("best_loss", [r["validation"]["loss"] for r in records], "best_loss_epoch", "best_val_loss"),
    ):
        if not np.isfinite(values).all():
            raise FloatingPointError("validation 选点轨迹包含非有限值")
        epoch = int(np.argmin(values)) + 1
        if epoch != completion[epoch_key] or values[epoch - 1] != completion[score_key]:
            raise ValueError(f"{role} 完成回执与 validation 轨迹不符")
        path = directories[epoch] / "checkpoint.pt"
        entries.append({"role": role, "epoch": epoch, "checkpoint": str(path), "sha256": sha(path),
                        "validation_local_rr_mae": records[epoch - 1]["val_local_rr_mae"],
                        "validation_loss": records[epoch - 1]["validation"]["loss"]})
    return entries


def prepare(run_root, output):
    """只读训练产物；在任何 test 数据访问前固定两个 checkpoint 及评价合同。"""
    run_root, output = Path(run_root).resolve(), Path(output).resolve()
    if not output.is_relative_to(training.STORAGE.resolve()):
        raise ValueError("评价产物必须位于项目目录")
    completion = read_json(run_root / "completed.json")
    if completion["status"] != "completed":
        raise ValueError("训练尚未完成")
    directories = training.completed_epochs(run_root)
    records = [read_json(directories[e] / "record.json") for e in sorted(directories)]
    entries = select_entries(records, completion, directories)
    if sha(run_root / completion["best_loss_checkpoint"]) != entries[1]["sha256"]:
        raise ValueError("最低 loss 副本与选中 epoch 不符")
    cfg = OmegaConf.load(run_root / "resolved_config.yaml")
    if (cfg.model.name != "bcg_cwt_patch_mamba_v1" or cfg.data.test_split != "test"
            or cfg.data.max_test_windows is not None or cfg.data.test_sample_seed != TEST_SAMPLE_SEED):
        raise ValueError("训练配置与完整 test 合同不符")
    source = read_json(run_root / "source_manifest.json")
    for name, expected in source["files"].items():
        if name.endswith(".py") and sha(ROOT / name) != expected:
            raise ValueError(f"训练源码变化: {name}")
    frequencies = None
    for entry in entries:
        state = torch.load(entry["checkpoint"], map_location="cpu", weights_only=False)
        training.finite_tensors(state["model_state_dict"])
        if state["epoch"] != entry["epoch"] or state["config_identity"] != training.config_identity(cfg):
            raise ValueError("checkpoint epoch/config 不符")
        current = state["model_state_dict"]["frequencies_hz"].numpy()
        if frequencies is not None and not np.array_equal(current, frequencies):
            raise ValueError("两个 checkpoint 频率身份不一致")
        frequencies = current.copy()
        del state
    development_subjects, row_files = set(), {}
    for split, count, subjects in (("train", 10141, 32), ("val", 2675, 7)):
        path = run_root / f"{split}_rows.csv"
        rows = pd.read_csv(path)
        if (len(rows) != count or rows.samp_id.nunique() != subjects or set(rows.split) != {split}
                or rows.dataset_row_id.duplicated().any()):
            raise ValueError(f"{split} 来源行身份错误")
        development_subjects.update(rows.samp_id.astype(int))
        row_files[str(path)] = sha(path)
    output.mkdir(parents=True, exist_ok=False)
    training.save_sources(output)
    shutil.copyfile(PROTOCOL_PATH, output / "protocol.md")
    manifest = {"protocol": PROTOCOL, "run_root": str(run_root), "entries": entries,
                "config": OmegaConf.to_container(cfg, resolve=True),
                "training_environment": read_json(run_root / "started.json")["environment"],
                "development_subjects": sorted(development_subjects), "development_row_files": row_files,
                "frequencies_hz": frequencies.tolist(), "dataset": read_json(run_root / "access_started.json"),
                "training_completion_sha256": sha(run_root / "completed.json"),
                "protocol_sha256": sha(output / "protocol.md"),
                "source_manifest_sha256": sha(output / "source_manifest.json"),
                "source_snapshot_sha256": sha(output / "source.tar.gz"),
                "contract": {"count": COUNT, "subjects": SUBJECTS, "row_hash": ROW_ORDER_SHA256,
                             "batch_size": 64, "amp_dtype": "bfloat16", "include_test_only": True,
                             "cache_root": str(cache.FROZEN_RESEARCH_TEST_CACHE_ROOT),
                             "cache_manifest_sha256": cache.FROZEN_RESEARCH_TEST_CACHE_MANIFEST_SHA256},
                "test_accessed": False}
    write_json(output / "evaluation_manifest.json", manifest)
    write_json(output / "prepared.json", {"manifest_sha256": sha(output / "evaluation_manifest.json")})
    return output


def load_manifest(output):
    if sha(output / "evaluation_manifest.json") != read_json(output / "prepared.json")["manifest_sha256"]:
        raise ValueError("评价 manifest SHA-256 不符")
    manifest = read_json(output / "evaluation_manifest.json")
    if manifest["protocol"] != PROTOCOL or [e["role"] for e in manifest["entries"]] != ["best_rr", "best_loss"]:
        raise ValueError("评价协议或 checkpoint 清单不符")
    for file, key in (("protocol.md", "protocol_sha256"), ("source_manifest.json", "source_manifest_sha256"),
                      ("source.tar.gz", "source_snapshot_sha256")):
        if sha(output / file) != manifest[key]:
            raise ValueError(f"评价来源变化: {file}")
    for name, expected in read_json(output / "source_manifest.json")["files"].items():
        if name.endswith(".py") and sha(ROOT / name) != expected:
            raise ValueError(f"评价源码变化: {name}")
    for entry in manifest["entries"]:
        if sha(entry["checkpoint"]) != entry["sha256"]:
            raise ValueError("checkpoint SHA-256 不符")
    return manifest


class WTestReader:
    def __init__(self, frequencies, *, confirmed=False):
        require_confirmation(confirmed)
        self.source = cache.TfV1ResearchTestCacheReader(cache.FROZEN_RESEARCH_TEST_CACHE_ROOT,
                                                       split="test", representations=("w",))
        if self.source.manifest["fixed_transform_spec"] != fixed_transform_spec():
            raise ValueError("test CWT 变换规格不符")
        actual = self.source._open("w_frequencies_hz.npy")
        if (actual.shape != (97,) or actual.dtype != np.float64 or not np.isfinite(actual).all()
                or np.any(actual <= 0) or np.any(np.diff(actual) < 0)
                or not np.array_equal(actual, np.asarray(frequencies, dtype=np.float64))):
            raise ValueError("test 与 checkpoint 实际频率顺序不一致")
        w = self.source.arrays["w"]
        if w.shape != (COUNT, 97, 360) or w.dtype != np.float32:
            raise ValueError("test W shape/dtype 不符")

    def get(self, row_id):
        result = self.source.get(row_id)
        if not torch.isfinite(result["w"]).all():
            raise FloatingPointError("test W 包含 NaN/Inf")
        return result


class WTestDataset(Dataset):
    def __init__(self, base, reader):
        self.base, self.reader = base, reader

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = dict(self.base[i])
        item["tf"] = self.reader.get(item["meta"]["dataset_row_id"])
        return item


def infer(model, loader, rows, cfg, device):
    predictions = collect_predictions(model, guarded_batches(loader, rows), device=device,
                                      max_windows=len(rows), use_amp=True)
    for key in ("r_tho_hat", "tho_ref"):
        if predictions[key].shape != (len(rows), 1, 18000) or not np.isfinite(predictions[key]).all():
            raise FloatingPointError(f"test {key} shape/finite 不符")
    metrics = evaluate_task_predictions(predictions, cfg, include_test_only=True, method=cfg.model.name)
    return predictions, metrics, check_test_metrics(metrics, rows)


def evaluate(output, device, *, confirmed=False):
    require_confirmation(confirmed)
    output = Path(output).resolve()
    manifest = load_manifest(output)
    if torch.device(device).type != "cuda":
        raise ValueError("正式 research-test 要求 CUDA")
    if environment_versions() != manifest["training_environment"]:
        raise ValueError("评价与训练环境版本不同")
    with training.run_mutex(output):
        if (output / "evaluation_started.json").exists():
            raise FileExistsError("已有评价 attempt；保留产物并检查状态，不自动重跑")
        write_json(output / "evaluation_started.json", {"protocol": PROTOCOL, "command": sys.argv,
            "environment": environment_versions(), "device": device, "manifest_sha256": sha(output / "evaluation_manifest.json")})
        try:
            cfg = OmegaConf.create(manifest["config"])
            write_json(output / "access_started.json", {"split": "test", "count": COUNT,
                "subjects": SUBJECTS, "evidence_role": "reused research/development evidence",
                "entries": manifest["entries"], "cache": manifest["contract"]})
            if sha(manifest["dataset"]["index"]) != manifest["dataset"]["index_sha256"]:
                raise ValueError("数据 index 身份变化")
            audited = read_research_v2_index(cfg.data.dataset_root, cfg.data.index_csv, cfg)
            rows = filter_index(audited, cfg, split="test", max_windows=None,
                                sample_strategy=cfg.data.test_sample_strategy, sample_seed=TEST_SAMPLE_SEED)
            check_test_rows(rows)
            if set(rows.samp_id.astype(int)) & set(manifest["development_subjects"]):
                raise ValueError("test 与 train/validation 受试者重叠")
            rows.to_csv(output / "test_rows.csv", index=False)
            # 波形/目标继续用原数据层；完整 W 条件由本专项的只读 wrapper 注入。
            raw = OmegaConf.create(manifest["config"])
            raw.model.name, raw.model.tf_representations, raw.data.tf_cache_path = "crd_v1", [], None
            raw.data.preload_windows = False
            raw.training.device = device
            bundle = build_window_data(raw, split="test", max_windows=None, shuffle=False, audited=audited,
                sample_strategy=cfg.data.test_sample_strategy, sample_seed=TEST_SAMPLE_SEED)
            check_test_rows(bundle.rows, rows)
            reader = WTestReader(manifest["frequencies_hz"], confirmed=True)
            reader.source.verify_rows(rows.dataset_row_id)
            loader = DataLoader(WTestDataset(bundle.dataset, reader), batch_size=64, shuffle=False,
                                num_workers=0, drop_last=False, pin_memory=True)
            summaries, reference = [], None
            for entry in manifest["entries"]:
                directory = output / entry["role"]
                directory.mkdir()
                print(f"开始 {entry['role']}: epoch={entry['epoch']}, {COUNT} 窗口, {device}", flush=True)
                state = torch.load(entry["checkpoint"], map_location="cpu", weights_only=False)
                training.finite_tensors(state["model_state_dict"])
                model = BCGCWTPatchMamba(manifest["frequencies_hz"], config=BCGCWTConfig(cfg.model.architecture),
                                        initialization_seed=cfg.model.initialization_seed)
                model.load_state_dict(state["model_state_dict"], strict=True)
                del state
                predictions, metrics, summary = infer(model, loader, rows, cfg, device)
                identity = metrics[[*IDENTITY_COLUMNS, *TARGET_COLUMNS]]
                if reference is None:
                    reference = identity.copy()
                elif not identity.equals(reference):
                    raise ValueError("两个 checkpoint 的样本或 target eligibility 不一致")
                np.savez_compressed(directory / "predictions.npz", **predictions)
                metrics.to_csv(directory / "metrics.csv", index=False)
                summary.insert(0, "role", entry["role"])
                summary.insert(1, "epoch", entry["epoch"])
                summary.to_csv(directory / "metrics_summary.csv", index=False)
                write_json(directory / "receipt.json", {"entry": entry, **quality_flags(metrics),
                    "sha256": {name: sha(directory / name) for name in ("predictions.npz", "metrics.csv", "metrics_summary.csv")}})
                summaries.append(summary)
                print(f"完成 {entry['role']}", flush=True)
                del model, predictions, metrics
                gc.collect()
                torch.cuda.empty_cache()
            pd.concat(summaries, ignore_index=True).to_csv(output / "comparison.csv", index=False)
            load_manifest(output)
            write_json(output / "completed.json", {"protocol": PROTOCOL, "status": "completed",
                "entries": manifest["entries"], "count": COUNT, "subjects": SUBJECTS,
                "comparison_sha256": sha(output / "comparison.csv"),
                "receipts": {e["role"]: sha(output / e["role"] / "receipt.json") for e in manifest["entries"]}})
            return output
        except BaseException as exc:
            write_json(output / "failed.json", {"error": str(exc), "traceback": traceback.format_exc()})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--confirm-research-test", action="store_true")
    args = parser.parse_args()
    print(prepare(args.run, args.output) if args.command == "prepare" else
          evaluate(args.output, args.device, confirmed=args.confirm_research_test), flush=True)


if __name__ == "__main__":
    main()
