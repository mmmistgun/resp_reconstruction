from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from resp_train.aligned_dual_view.artifacts import code_identity, read_json, sha256_file, verified_manifest, write_json
from resp_train.aligned_dual_view.config import ALL_VIEWS, INDEX_SHA256, ROOT, SEEDS, load_experiment_config
from resp_train.aligned_dual_view.data import transform_identity
from resp_train.aligned_dual_view.features import representation_spec, spec_digest

PROTOCOL = "aligned-dual-view-v1-research-test-20260922"
EVIDENCE_ROLE = "reused research/development evidence"
TRAINING_CODE_SHA = "19b92747cb2a26a5cf45d6954aeebaae1ce70b3de5a01a266677de4761284040"
SUMMARY_ROOT = ROOT / "runs/aligned_dual_view_v1/summary_full_20260922_r1"
SUMMARY_SHA = "cdc03f23f299b24bbf6234cd73c5c1678d7e9d7b22065e7f8cef58f9a0d1577d"
LOCK_PATH = ROOT / "configs/aligned_dual_view_research_test_v1/candidate_lock_20260922.json"
LOCK_SHA = "146e89f4c196fd1b2742ba66bb3ad600f11e25a6ccf98ebfcbc7ece012947418"
PROTOCOL_PATH = ROOT / "docs/experiments/aligned_dual_view_v1_research_test_protocol_20260922.md"
TEST_COUNT = 2310
TEST_SUBJECT_COUNT = 8
TEST_ROW_IDS_SHA = "184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8"
TEST_SAMPLE_SEED = 20260612


def repo_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(ROOT) or ".." in Path(relative).parts:
        raise ValueError("来源路径必须位于当前工作树内")
    return path


def entry_id(view: str, seed: int) -> str:
    if view not in ALL_VIEWS or seed not in SEEDS:
        raise ValueError("候选不在四配置 × 三 seed 矩阵内")
    return f"{view}_seed{seed}"


def require_confirmation(confirmed: bool) -> None:
    if confirmed is not True:
        raise PermissionError("research-test 访问要求显式 --confirm-research-test")


def prepare_lock(output: str | Path = LOCK_PATH) -> Path:
    """仅从已完成的 validation 产物固定 12 个 checkpoint；不读取数据集。"""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"候选锁禁止覆盖: {output}")
    if sha256_file(SUMMARY_ROOT / "manifest.json") != SUMMARY_SHA:
        raise ValueError("validation summary 偏离既有来源")
    summary = verified_manifest(SUMMARY_ROOT, "summary")
    if summary["views"] != list(ALL_VIEWS) or summary["seeds"] != list(SEEDS):
        raise ValueError("validation 矩阵不完整")
    for name, digest in summary["files"].items():
        if sha256_file(SUMMARY_ROOT / name) != digest:
            raise ValueError("validation summary 文件哈希不匹配")
    current = code_identity()
    if current["sha256"] != TRAINING_CODE_SHA:
        raise ValueError("训练执行代码身份发生变化")
    entries, dependencies = {}, None
    for source in summary["sources"]:
        run = Path(source["root"]).resolve()
        if not run.is_relative_to(ROOT):
            raise ValueError("训练来源必须位于当前工作树")
        if sha256_file(run / "manifest.json") != source["manifest_sha256"]:
            raise ValueError("训练 manifest 与 validation 来源不一致")
        manifest = verified_manifest(run, "train")
        name = entry_id(manifest["view"], manifest["seed"])
        if name in entries or manifest["run_role"] != "formal" or manifest["epochs_completed"] != 80:
            raise ValueError("formal 候选重复或未完成")
        if manifest["comparison_id"] != summary["comparison_id"]:
            raise ValueError("候选 comparison ID 不匹配")
        for filename, key in (("config.yaml", "config_sha256"), ("history.csv", "history_sha256"),
                              ("metrics.csv", "metrics_sha256"), ("metrics_summary.csv", "summary_sha256"),
                              ("samples.json", "samples_sha256")):
            if sha256_file(run / filename) != manifest[key]:
                raise ValueError(f"训练来源哈希不匹配: {filename}")
        cfg = load_experiment_config(run / "config.yaml")
        if (cfg.model.input_view, cfg.training.seed) != (manifest["view"], manifest["seed"]):
            raise ValueError("配置与候选标签不匹配")
        if spec_digest(OmegaConf.to_container(cfg, resolve=True)) != manifest["identity"]["config_sha256"]:
            raise ValueError("配置内容 identity 不匹配")
        history = pd.read_csv(run / "history.csv")
        if history.epoch.tolist() != list(range(1, 81)) or not np.isfinite(history.val_local_rr_mae).all():
            raise ValueError("训练 history 不完整或非有限")
        epoch = int(history.iloc[np.argmin(history.val_local_rr_mae.to_numpy())].epoch)
        if epoch != manifest["selected_epoch"] or manifest["selected_checkpoint"] != f"checkpoints/epoch_{epoch:03d}.pt":
            raise ValueError("checkpoint 不是最早 validation Local RR 最小值")
        checkpoint = run / manifest["selected_checkpoint"]
        if sha256_file(checkpoint) != manifest["selected_sha256"]:
            raise ValueError("selected checkpoint 哈希不匹配")
        started = read_json(run / "started.json")
        if started["code"] != current or manifest["identity"]["code_sha256"] != TRAINING_CODE_SHA:
            raise ValueError("训练代码 provenance 不匹配")
        if dependencies is None:
            dependencies = started["dependencies"]
        elif dependencies != started["dependencies"]:
            raise ValueError("训练依赖版本不一致")
        entries[name] = {
            "view": manifest["view"], "seed": manifest["seed"], "selected_epoch": epoch,
            "run": str(run.relative_to(ROOT)), "run_manifest_sha256": source["manifest_sha256"],
            "config_sha256": manifest["config_sha256"],
            "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": manifest["selected_sha256"],
            "training_identity": manifest["identity"],
        }
    expected = {entry_id(view, seed) for view in ALL_VIEWS for seed in SEEDS}
    if set(entries) != expected or len(entries) != 12:
        raise ValueError("checkpoint 清单必须完整覆盖 12 个候选")
    payload = {
        "protocol": PROTOCOL, "evidence_role": EVIDENCE_ROLE,
        "validation_summary": str(SUMMARY_ROOT.relative_to(ROOT)), "validation_summary_sha256": SUMMARY_SHA,
        "comparison_id": summary["comparison_id"], "training_code": current, "dependencies": dependencies,
        "dataset_index_sha256": INDEX_SHA256, "test_count": TEST_COUNT,
        "test_subject_count": TEST_SUBJECT_COUNT, "test_row_ids_sha256": TEST_ROW_IDS_SHA,
        "test_sample_seed": TEST_SAMPLE_SEED, "representation": representation_spec(), "transform": transform_identity(),
        "include_test_only_metrics": True, "entries": entries,
        "selection": "validation_local_rr_minimum_earliest_tie", "test_data_accessed_during_lock": False,
    }
    payload["lock_id"] = spec_digest(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, payload)
    return output


def load_lock(path: str | Path = LOCK_PATH) -> dict:
    if sha256_file(path) != LOCK_SHA:
        raise ValueError("research-test candidate lock 哈希不匹配")
    lock = read_json(Path(path))
    content = {key: value for key, value in lock.items() if key != "lock_id"}
    if lock["protocol"] != PROTOCOL or lock["lock_id"] != spec_digest(content):
        raise ValueError("research-test lock identity 不匹配")
    expected = {entry_id(view, seed) for view in ALL_VIEWS for seed in SEEDS}
    if set(lock["entries"]) != expected:
        raise ValueError("research-test 固定矩阵不完整")
    if lock["training_code"] != code_identity():
        raise ValueError("冻结的训练代码或配置被修改")
    if lock["representation"] != representation_spec() or lock["transform"] != transform_identity():
        raise ValueError("前处理偏离冻结表示")
    return lock


def source_config(lock: dict, name: str):
    if name not in lock["entries"]:
        raise ValueError("checkpoint 不在冻结清单内")
    entry = lock["entries"][name]
    root = repo_path(entry["run"])
    if sha256_file(root / "manifest.json") != entry["run_manifest_sha256"]:
        raise ValueError("原训练 manifest 哈希漂移")
    verified_manifest(root, "train")
    if sha256_file(root / "config.yaml") != entry["config_sha256"]:
        raise ValueError("原训练配置哈希漂移")
    cfg = load_experiment_config(root / "config.yaml")
    if spec_digest(OmegaConf.to_container(cfg, resolve=True)) != entry["training_identity"]["config_sha256"]:
        raise ValueError("原训练配置 identity 漂移")
    return cfg
