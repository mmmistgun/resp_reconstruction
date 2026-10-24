"""计划运行的来源、数据身份和不可覆盖产物。"""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time

import pandas as pd
import torch
from omegaconf import OmegaConf

from resp_train.data.factory import build_window_data
from resp_train.respdiff.data import REQUIRED, assert_split_independence, validate_rows
from .baseband import (ROOT, SCHEMA as BASE_SCHEMA, LPF_CONTRACT, baseband_dataset,
                       config_contract, load_baseband_config, make_baseband_model)
from .inference_v2 import CHECKPOINT_SHA256, ParentSelection, state_digest
from .runtime import (LOGGER, environment, run_logging, save_source_snapshot, sha256,
                      source_file_inventory, source_identity, write_json)
from .study_contract import NAME, SCHEMA, PROBE_STEPS, PROBE_SEEDS, TrainingSpec
from .study_state import finite_tree

PLAN_FILE = "docs/experiments/respdiff_bcg_training_and_sampling_plan_v1_20261010.md"


def training_config(base, spec):
    cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
    cfg.protocol.name = NAME
    cfg.training.max_updates = spec.updates
    cfg.training.milestones = list(spec.milestones)
    cfg.training.seed = spec.seed
    cfg.diagnostics = {"probe_updates": list(spec.snapshots), "timesteps": list(PROBE_STEPS),
        "forward_noise_seeds": list(PROBE_SEEDS), "train_probe_parents": 8 if spec.regime == "overfit" else 32,
        "val_probe_parents": 32, "batch_size": 64}
    cfg.study = {"training": spec.identity()}
    return cfg


def validate_training_config(cfg):
    if "study" not in cfg:
        raise ValueError("缺少 study 训练身份")
    actual = OmegaConf.to_container(cfg, resolve=True)
    values = actual["study"]["training"]
    spec = TrainingSpec(values["regime"], values["seed"])
    base = config_contract("source_equivalent")
    base.training.device = cfg.training.device
    base.data.dataset_root = cfg.data.dataset_root
    expected = OmegaConf.to_container(training_config(base, spec), resolve=True)
    if actual != expected:
        raise ValueError("训练配置偏离计划合同")
    return spec


def verify_artifact(source, receipt, name):
    expected = receipt.get("artifacts", {}).get(name)
    if expected is None or sha256(source / name) != expected:
        raise ValueError(f"来源产物身份不匹配：{name}")


def load_source(source, device, *, expected_sha256=None):
    source = Path(source).resolve()
    receipt = json.loads((source / "receipt.json").read_text())
    if receipt.get("status") != "complete" or not receipt.get("formal_evidence"):
        raise ValueError("来源要求已完成的真实 train run；合成或失败产物不能进入科研评价")
    for name in ("final.pt", "resolved_config.yaml", "train_parents.csv", "val_parents.csv", "data_identity.json"):
        verify_artifact(source, receipt, name)
    digest = sha256(source / "final.pt")
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("checkpoint 与显式冻结 SHA256 不匹配")
    state = torch.load(source / "final.pt", map_location="cpu", weights_only=False)
    finite_tree(state)
    cfg = OmegaConf.load(source / "resolved_config.yaml")
    if state.get("schema") == BASE_SCHEMA:
        cfg = load_baseband_config(source / "resolved_config.yaml")
        if (digest != CHECKPOINT_SHA256 or receipt.get("objective") != "source_equivalent"
                or state.get("update") != 6400 or state.get("selector") != "final_update"
                or state.get("experiment", {}).get("seed") != 20260811):
            raise ValueError("历史来源必须匹配已核验的 source-equivalent final6400")
        seed, regime = 20260811, "baseline"
    elif state.get("schema") == SCHEMA:
        spec = validate_training_config(cfg)
        expected_training = {**spec.identity(), "actual_updates": spec.updates, "actual_batch_size": 64,
            "synthetic": False, "model_spec": OmegaConf.to_container(cfg.model, resolve=True),
            "objective": "source_equivalent"}
        if (state.get("selector") != "final_update" or state.get("update") != spec.updates
                or state.get("training") != expected_training
                or spec.regime == "overfit"):
            raise ValueError("候选必须是完整正式训练的 final-update 模型")
        if receipt.get("protocol") != NAME or receipt.get("phase") != "train":
            raise ValueError("候选训练 receipt 不匹配")
        verify_artifact(source,receipt,"actual_training_parents.csv")
        if (state.get("training_data_sha256") != sha256(source/"actual_training_parents.csv")
                or sha256(source/"actual_training_parents.csv") != sha256(source/"train_parents.csv")):
            raise ValueError("正式候选的实际训练父集合不完整或不匹配 checkpoint")
        seed, regime = spec.seed, spec.regime
    else:
        raise ValueError("不支持的 checkpoint schema")
    model = make_baseband_model(cfg)
    if state.get("model_spec") != vars(model.spec):
        raise ValueError("checkpoint model_spec 不匹配固定网络")
    model.load_state_dict(state["model"], strict=True)
    update = int(state["update"])
    del state
    model.to(device).eval()
    return model, cfg, {"source_run": str(source), "checkpoint_sha256": digest,
                        "training_seed": seed, "regime": regime, "update": update,
                        "parameter_sha256": state_digest(model)}


def source_path_mapping(recorded_paths, dataset_root):
    """迁移只替换同名数据集目录之前的主机前缀；目录内身份保持不变。"""
    root = Path(dataset_root).resolve(strict=True)
    mapping = {}
    for value in recorded_paths:
        path = Path(value)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("来源路径须为无上溯的绝对路径")
        if path.is_relative_to(root):
            relative = path.relative_to(root)
        else:
            positions = [i for i, part in enumerate(path.parts) if part == root.name]
            if len(positions) != 1:
                raise ValueError("来源路径无法唯一匹配同名数据集目录")
            relative = Path(*path.parts[positions[0]+1:])
        destination = (root / relative).resolve(strict=True)
        if not destination.is_relative_to(root) or not destination.is_file():
            raise ValueError("来源路径映射越界或不是文件")
        mapping[value] = str(destination)
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("多个来源文件映射到同一本地文件")
    return mapping


def verify_source_inventory(recorded, rows, mapping):
    if set(recorded) != set(mapping):
        raise ValueError("来源文件映射集合不完整")
    remapped = {mapping[path]: stat for path, stat in recorded.items()}
    if source_file_inventory(rows) != remapped:
        raise ValueError("数据来源文件的大小或时间身份偏离原训练记录")


def source_data(cfg, source, *, mapping_output=None):
    """完整 train/val 身份先核对；waveform 仍由选中 dataset 惰性读取。"""
    source = Path(source)
    recorded = json.loads((source / "data_identity.json").read_text())
    mapping = source_path_mapping(recorded["source_file_stats"], cfg.data.dataset_root)
    bundles = {}
    for split in ("train", "val"):
        bundle = build_window_data(cfg, split=split, max_windows=cfg.data[f"max_{split if split == 'train' else 'val'}_windows"],
            sample_strategy=str(cfg.data[f"{split}_sample_strategy"]),
            sample_seed=int(cfg.data[f"{split}_sample_seed"]), shuffle=False)
        expected = pd.read_csv(source / f"{split}_parents.csv")
        # 原始 receipt/CSV 保持冻结；仅在内存中统一经过验证的主机路径前缀。
        for column in ("source_npz", "target_source_npz"):
            expected[column] = expected[column].map(mapping)
            if expected[column].isna().any():
                raise ValueError(f"{split} 父清单包含未记录的来源文件")
        # ChunkDataset canonicalize 路径；构造数据包装不读取真实 waveform。
        dataset = baseband_dataset(bundle.dataset, bundle.rows)
        columns = list(REQUIRED) + [c for c in ("bcg_signal_key", "target_signal_key") if c in expected]
        if not expected[columns].reset_index(drop=True).equals(dataset.rows[columns].reset_index(drop=True)):
            raise ValueError(f"{split} 数据身份或顺序偏离来源")
        if sha256(bundle.index_path) != recorded["index_sha256"]:
            raise ValueError("数据索引 hash 偏离来源；路径迁移应先明确记录映射")
        bundles[split] = (bundle.dataset, dataset.rows, bundle.index_path)
    assert_split_independence(bundles["train"][1], bundles["val"][1])
    combined = pd.concat([part[1] for part in bundles.values()], ignore_index=True)
    verify_source_inventory(recorded["source_file_stats"], combined, mapping)
    if mapping_output is not None:
        write_json(Path(mapping_output), {"source_run": str(source.resolve()),
            "source_data_identity_sha256": sha256(source / "data_identity.json"),
            "index_sha256": recorded["index_sha256"], "dataset_root": str(Path(cfg.data.dataset_root).resolve()),
            "path_mapping": mapping, "n_changed_paths": sum(old != new for old, new in mapping.items()),
            "validation": "same index hash, ordered parent identity, signal keys, exact file size and mtime_ns"})
    return bundles


def stratified_indices(rows, count, *, seed=20261010):
    """samp_id 轮转，其内按连续 target 来源记录轮转，再选固定 hash 排序的窗口。"""
    validate_rows(rows)
    if type(count) is not int or not 1 <= count <= len(rows):
        raise ValueError("分层 parent 数量非法")
    def rank(kind, value):
        return hashlib.sha256(f"{NAME}|{seed}|{kind}|{value}".encode()).hexdigest()
    subjects = sorted(rows.samp_id.unique(), key=lambda value: rank("subject", int(value)))
    queues = {}
    for subject in subjects:
        records = []
        for _, frame in rows[rows.samp_id == subject].groupby("target_source_npz", sort=False):
            ids = sorted(frame.index.tolist(), key=lambda index: rank("row", int(rows.loc[index, "dataset_row_id"])))
            records.append(ids)
        records.sort(key=lambda values: rank("record", min(int(rows.loc[i, "dataset_row_id"]) for i in values)))
        queues[subject] = records
    chosen, turns = [], {subject: 0 for subject in subjects}
    while len(chosen) < count:
        for subject in subjects:
            records = queues[subject]
            for _ in range(len(records)):
                index = turns[subject] % len(records)
                turns[subject] += 1
                if records[index]:
                    chosen.append(records[index].pop(0))
                    break
            if len(chosen) == count:
                break
    return sorted(int(index) for index in chosen)


def selected_dataset(parents, rows, indices):
    selected = ParentSelection(parents, rows, indices)
    return baseband_dataset(selected, selected.rows)


def implementation_identity(cfg):
    identity = source_identity(cfg)
    paths = [ROOT / "scripts/run_respdiff_bcg_study_v1.py", ROOT / "scripts/run_respdiff_bcg_baseband_v1.py",
             ROOT / "configs/respdiff_bcg_v1/gpu_b64_b64.yaml", ROOT / PLAN_FILE]
    paths += sorted((ROOT / "configs/respdiff_bcg_study_v1").glob("*.yaml"))
    paths += [ROOT / "scripts/run_respdiff_bcg_cooldown_v1.py"]
    for path in paths:
        identity["files"][str(path.relative_to(ROOT))] = sha256(path)
    identity["lowpass"] = LPF_CONTRACT
    return identity


def save_provenance(output, cfg, *, cpu_only, execution):
    identity = implementation_identity(cfg)
    OmegaConf.save(cfg, output / "resolved_config.yaml")
    write_json(output / "source_identity.json", identity)
    save_source_snapshot(output, identity)
    write_json(output / "environment.json", environment(cpu_only=cpu_only))
    write_json(output / "execution.json", {"protocol": NAME, **execution})
    return identity


def data_identity(output, datasets, index_path):
    rows = pd.concat([dataset.rows for dataset in datasets.values()], ignore_index=True)
    identity = {"index_path": str(index_path), "index_sha256": sha256(index_path),
                "source_file_stats": source_file_inventory(rows), "lowpass": LPF_CONTRACT}
    write_json(output / "data_identity.json", identity)
    for split, dataset in datasets.items():
        dataset.rows.to_csv(output / f"{split}_parents.csv", index=False)
        dataset.manifest().to_csv(output / f"{split}_chunks.csv", index=False)
    return identity


def verify_data_unchanged(identity, rows):
    if (sha256(identity["index_path"]) != identity["index_sha256"]
            or source_file_inventory(rows) != identity["source_file_stats"]):
        raise ValueError("运行期间数据身份变化")


@contextmanager
def study_run(output, phase, *, synthetic, source=None):
    output = Path(output).resolve()
    if source is not None:
        source = Path(source).resolve()
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("新输出不得覆盖来源目录或位于来源目录内")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result = {"protocol": NAME, "phase": phase, "formal_evidence": not synthetic}
    try:
        with run_logging(output):
            try:
                yield output, result
                LOGGER.info("阶段完成：%s", phase)
            except BaseException as error:
                LOGGER.exception("阶段失败：%s", phase)
                write_json(output / "failure.json", {"phase": phase, "error_type": type(error).__name__,
                                                      "error": str(error), "status": "failed"})
                raise
    except BaseException:
        result["status"] = "failed"
        raise
    else:
        result.setdefault("status", "complete")
    finally:
        # 关闭日志后再计算 hash，receipt 不引用随后仍变化的日志内容。
        result["total_wall_seconds"] = time.perf_counter() - started
        result["artifacts"] = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*"))
                               if p.is_file() and p != output / "receipt.json"}
        write_json(output / "receipt.json", result)
