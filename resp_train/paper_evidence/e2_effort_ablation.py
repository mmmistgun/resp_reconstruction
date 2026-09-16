"""最终 W0 的相对努力损失消融：独立配置合同与原生训练生命周期。"""

from __future__ import annotations

import json
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from resp_train.crd.config import check_crd_dependencies, load_crd_config
from resp_train.crd.experiment import CRDExperiment
from resp_train.crd.model import build_crd_model
from resp_train.crd.tf_w_v2_audit import _w0_seed_entries
from resp_train.crd.training import build_crd_optimizer, crd_learning_rate, train_crd_one_epoch
from resp_train.data.index import filter_index
from resp_train.data.research_v2 import read_research_v2_index
from resp_train.losses.task import RespirationTaskLoss
from resp_train.metrics.task import summarize_task_metrics
from resp_train.paper_evidence.e1_scale_topology import PRIMARY, ERRORS, PCC, SEEDS, array_hash, validate_metrics
from resp_train.paper_evidence.e1_scale_topology_runtime import environment, git_state, identity, sha256_file, write_json

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "e2-w0-effort-ablation-v1-20260916"
ARM = "w0_sync_only"
SOURCE_LOCK = Path("docs/experiments/crd_tf_w_v2_candidate_lock_20260817.json")
SOURCE_SHA = "6ae35076bbd89bec688bfd4918cfecd20c7d5ea7f845f460034a88045432c7b6"
LOCK_PATH = Path("docs/experiments/e2_w0_effort_implementation_lock_20260916.json")
PROTOCOL_PATH = Path("docs/experiments/e2_w0_effort_ablation_protocol_20260916.md")
SCRIPT_PATH = Path("scripts/run_e2_w0_effort_ablation.py")
TEST_PATH = Path("tests/test_e2_effort_ablation.py")
OUTPUT = Path("runs/e2_w0_effort_ablation_v1")
COUNTS = {"train": 10141, "val": 2675}
EPOCHS = 80
UPDATES_PER_EPOCH = 80


def verify(path: Path, expected: Mapping[str, Any]) -> None:
    if not path.is_file() or identity(path) != {k: expected[k] for k in ("size_bytes", "sha256")}:
        raise RuntimeError(f"E2 文件身份漂移: {path}")


def derived_config(baseline: DictConfig, *, output_root: Path, device: str) -> DictConfig:
    """由同 seed 冻结 resolved config 派生；运算族仍是原生 TF，实验身份使用 E2。"""
    cfg = OmegaConf.create(OmegaConf.to_container(baseline, resolve=True))
    cfg.protocol.name = PROTOCOL
    cfg.protocol.execution_gate = "e2_formal"
    cfg.loss.effort_weight = 0.0
    cfg.training.device = device
    cfg.training.show_progress = False
    cfg.outputs.run_root = str(output_root)
    return cfg


def validate_config(cfg: DictConfig, baseline: DictConfig, *, output_root: Path, device: str) -> None:
    expected = derived_config(baseline, output_root=output_root, device=device)
    if OmegaConf.to_container(cfg, resolve=True) != OmegaConf.to_container(expected, resolve=True):
        raise ValueError("E2 配置必须与同 seed W0 合同一致，仅开放努力权重和实验/运行身份")
    if (str(baseline.model.variant) != "crd_tf102_w" or float(baseline.loss.effort_weight) != 0.25
            or float(baseline.loss.sync_weight) != 1.0 or int(baseline.training.seed) not in SEEDS
            or int(baseline.model.initialization_seed) != int(baseline.training.seed)
            or int(baseline.training.epochs) != EPOCHS or baseline.training.early_stopping_enabled
            or int(baseline.training.batch_size) != 128 or int(baseline.training.gradient_accumulation_steps) != 1
            or not baseline.training.use_amp or str(baseline.training.amp_dtype) != "bfloat16"
            or baseline.data.max_train_windows is not None or baseline.data.max_val_windows is not None):
        raise ValueError("E2 W0 来源配置不符合冻结矩阵")


def prepare_lock(root: Path = ROOT) -> Path:
    destination = root / LOCK_PATH
    if destination.exists():
        raise FileExistsError(f"E2 implementation lock 已存在: {destination}")
    if sha256_file(root / SOURCE_LOCK) != SOURCE_SHA:
        raise ValueError("E2 W0 candidate lock 漂移")
    source = json.loads((root / SOURCE_LOCK).read_text())
    entries = _w0_seed_entries(source)
    files = {str(SOURCE_LOCK): identity(root / SOURCE_LOCK)}
    baselines, templates = {}, {}
    for entry in entries:
        for key, name in (("checkpoint", "checkpoint_best_local_rr.pt"), ("config", "config.yaml"),
                          ("manifest", "run_manifest.json"), ("validation_summary", "metrics_summary.csv")):
            relative = str(Path(entry["run_dir"]) / name)
            verify(root / relative, entry[key])
            files[relative] = entry[key]
        cfg = load_crd_config(root / entry["run_dir"] / "config.yaml")
        output_root = root / OUTPUT / "formal" / f"seed_{entry['seed']}"
        candidate = derived_config(cfg, output_root=output_root, device="cuda:0")
        validate_config(candidate, cfg, output_root=output_root, device="cuda:0")
        baselines[str(entry["seed"])] = OmegaConf.to_container(cfg, resolve=True)
        templates[str(entry["seed"])] = OmegaConf.to_container(candidate, resolve=True)
    cache = source["cache_lock"]
    for key in ("manifest", "train_w", "val_w", "frequency_file"):
        entry = cache[key]
        expected = {"size_bytes": entry["size_bytes"], "sha256": entry.get("sha256", entry.get("file_sha256"))}
        verify(root / entry["path"], expected)
        files[entry["path"]] = expected
    for split in COUNTS:
        relative = str(Path(cache["root"]) / f"{split}_row_ids.npy")
        observed = identity(root / relative)
        if observed["sha256"] != cache["row_identity"][f"{split}_row_file_sha256"]:
            raise ValueError(f"E2 {split} row 文件漂移")
        files[relative] = observed
    cache_manifest = json.loads((root / cache["manifest"]["path"]).read_text())
    paths = sorted((root / "resp_train").rglob("*.py")) + [root / p for p in (SCRIPT_PATH, TEST_PATH, PROTOCOL_PATH)]
    lock = {"protocol": PROTOCOL, "arm": ARM, "seeds": list(SEEDS), "counts": COUNTS,
            "epochs": EPOCHS, "updates_per_epoch": UPDATES_PER_EPOCH, "w0_entries": entries,
            "baselines": baselines, "resolved_templates": templates, "cache_lock": cache,
            "dataset_index": {"path": cache_manifest["dataset_index"], "sha256": cache_manifest["dataset_index_sha256"]},
            "source_files": files, "code_files": {str(p.relative_to(root)): identity(p) for p in paths},
            "prepared_at": datetime.now(timezone.utc).isoformat(), "preparation_git": git_state(root),
            "status": "implementation_locked_gpu_and_training_pending"}
    write_json(destination, lock)
    return destination


def load_lock(root: Path = ROOT) -> tuple[dict[str, Any], str]:
    path = root / LOCK_PATH
    lock = json.loads(path.read_text())
    if (lock["protocol"] != PROTOCOL or lock["arm"] != ARM or tuple(lock["seeds"]) != SEEDS
            or lock["counts"] != COUNTS or lock["epochs"] != EPOCHS or lock["updates_per_epoch"] != UPDATES_PER_EPOCH):
        raise ValueError("E2 implementation lock 矩阵漂移")
    for relative, expected in lock["code_files"].items():
        verify(root / relative, expected)
    for seed in SEEDS:
        baseline = OmegaConf.create(lock["baselines"][str(seed)])
        template = OmegaConf.create(lock["resolved_templates"][str(seed)])
        validate_config(template, baseline, output_root=root / OUTPUT / "formal" / f"seed_{seed}", device="cuda:0")
    return lock, sha256_file(path)


@contextmanager
def attempt(parent: Path, lock_hash: str, phase: str, seed: int | None = None) -> Iterator[Path]:
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = parent / f"{phase}_{lock_hash[:12]}_{stamp}_{uuid4().hex[:12]}"
    path.mkdir(exist_ok=False)
    context = {"protocol": PROTOCOL, "arm": ARM, "seed": seed, "phase": phase,
               "implementation_lock_sha256": lock_hash, "command": sys.argv,
               "started_at": datetime.now(timezone.utc).isoformat()}
    write_json(path / "lifecycle_started.json", {**context, "status": "running"})
    print(f"E2 attempt: {path}", flush=True)
    try:
        yield path
        write_json(path / "lifecycle_completed.json", {**context, "status": "completed",
                   "ended_at": datetime.now(timezone.utc).isoformat()})
        manifest = {**context, "status": "completed", "files": {
            str(p.relative_to(path)): identity(p) for p in sorted(path.rglob("*")) if p.is_file()}}
        write_json(path / "manifest.json", manifest)
        write_json(path / "freeze_receipt.json", {"protocol": PROTOCOL, "manifest": identity(path / "manifest.json")})
    except BaseException as exc:
        write_json(path / "lifecycle_failed.json", {**context, "status": "failed", "error": str(exc),
                   "error_type": type(exc).__name__, "traceback": traceback.format_exc()})
        raise


def verify_attempt(path: Path, *, phase: str, lock_hash: str) -> dict[str, Any]:
    path = path.resolve()
    if (path / "lifecycle_failed.json").exists():
        raise ValueError(f"E2 attempt 失败: {path}")
    freeze = json.loads((path / "freeze_receipt.json").read_text())
    verify(path / "manifest.json", freeze["manifest"])
    manifest = json.loads((path / "manifest.json").read_text())
    if (manifest["protocol"] != PROTOCOL or manifest["phase"] != phase or manifest["status"] != "completed"
            or manifest["implementation_lock_sha256"] != lock_hash):
        raise ValueError("E2 attempt 协议/实现身份不一致")
    required = {"lifecycle_completed.json", "environment.json"}
    required.add("gpu_acceptance.json" if phase == "gpu_smoke" else "formal_receipt.json")
    if phase != "summary" and not required.issubset(manifest["files"]):
        raise ValueError("E2 attempt 缺少必需回执")
    for relative, expected in manifest["files"].items():
        file = (path / relative).resolve()
        if not file.is_relative_to(path):
            raise ValueError("E2 manifest 路径越界")
        verify(file, expected)
    return manifest


def runtime_preflight(device: str) -> dict[str, Any]:
    state = git_state()
    if state["status_porcelain"]:
        raise RuntimeError("E2 正式执行要求干净提交，请先提交本轮实现和协议")
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise ValueError("E2 GPU 阶段需要 CUDA；CPU 验证使用 synthetic 定向测试")
    problems = check_crd_dependencies()
    if problems:
        raise RuntimeError("E2 原生依赖检查失败: " + "; ".join(problems))
    return {**environment(device), "git": state}


def audit_sources(lock: Mapping[str, Any], cfg: DictConfig, output: Path) -> dict[str, pd.DataFrame]:
    write_json(output / "access_started.json", {"splits": ["train", "val"], "sources": list(lock["source_files"]),
               "dataset_index": lock["dataset_index"], "purpose": "E2 training and validation"})
    for relative, expected in lock["source_files"].items():
        verify(ROOT / relative, expected)
    index = lock["dataset_index"]
    if sha256_file(Path(index["path"])) != index["sha256"]:
        raise ValueError("E2 dataset index 漂移")
    audited = read_research_v2_index(cfg.data.dataset_root, cfg.data.index_csv, cfg)
    rows = {}
    for split, count in COUNTS.items():
        frame = filter_index(audited, cfg, split=split, max_windows=None,
                             sample_strategy=str(cfg.data[f"{split}_sample_strategy"]),
                             sample_seed=int(cfg.data[f"{split}_sample_seed"]))
        if (len(frame) != count or frame.dataset_row_id.duplicated().any()
                or set(frame.split.astype(str)) != {split}
                or frame.samp_id.nunique() != {"train": 32, "val": 7}[split]):
            raise ValueError(f"E2 {split} rows 不符合冻结合同")
        if array_hash(np.sort(frame.dataset_row_id.to_numpy())) != lock["cache_lock"]["row_identity"][f"{split}_row_content_sha256"]:
            raise ValueError(f"E2 {split} row 集合漂移")
        frame.to_csv(output / f"{split}_rows.csv", index=False)
        rows[split] = frame
    if set(rows["train"].samp_id) & set(rows["val"].samp_id):
        raise ValueError("E2 train/validation samp_id 交叉")
    write_json(output / "access_receipt.json", {"counts": COUNTS, "samp_ids": {"train": 32, "val": 7},
               "row_order_sha256": {s: array_hash(f.dataset_row_id.to_numpy()) for s, f in rows.items()},
               "source_files_verified": lock["source_files"], "dataset_index": index,
               "sample_seeds": {s: int(cfg.data[f"{s}_sample_seed"]) for s in COUNTS}})
    return rows


class EffortAblationExperiment(CRDExperiment):
    """只增强正式产物校验；训练、scheduler、梯度及 selector 由原生父类执行。"""

    task_name = ARM

    def __init__(self, cfg: DictConfig, validation_rows: pd.DataFrame):
        super().__init__(cfg)
        self.validation_rows = validation_rows

    def _evaluate_model(self, model, loader, **kwargs):
        frame = super()._evaluate_model(model, loader, **kwargs)
        validate_metrics(frame, self.validation_rows)
        frame.insert(0, "arm", ARM)
        frame.insert(0, "seed", int(self.cfg.training.seed))
        return frame


def finite_tree(value: Any) -> None:
    if torch.is_tensor(value):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError("E2 checkpoint/optimizer tensor 非有限")
    elif isinstance(value, Mapping):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            finite_tree(item)
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise FloatingPointError("E2 checkpoint scalar 非有限")


def validate_history(history: pd.DataFrame, cfg: DictConfig) -> int:
    if len(history) != EPOCHS or not np.array_equal(history.epoch, np.arange(1, EPOCHS + 1)):
        raise ValueError("E2 history 必须完整覆盖 80 epochs")
    if not np.isfinite(history.select_dtypes(include=np.number).to_numpy()).all():
        raise FloatingPointError("E2 history 非有限")
    if not np.array_equal(history.optimizer_update, np.arange(1, EPOCHS + 1) * UPDATES_PER_EPOCH):
        raise ValueError("E2 optimizer update 数不完整")
    if not np.allclose(history.train_loss_total, history.train_loss_sync, atol=1e-12, rtol=0):
        raise ValueError("E2 total loss 与纯同步目标不一致")
    total = EPOCHS * UPDATES_PER_EPOCH
    for row in history.itertuples():
        for column, update in (("first_learning_rate", (row.epoch - 1) * UPDATES_PER_EPOCH),
                               ("last_learning_rate", row.epoch * UPDATES_PER_EPOCH - 1)):
            expected = crd_learning_rate(update, total_updates=total,
                       max_learning_rate=float(cfg.training.max_learning_rate), min_learning_rate=float(cfg.training.min_learning_rate),
                       warmup_fraction=float(cfg.training.warmup_fraction))
            if not np.isclose(getattr(row, column), expected, atol=1e-15, rtol=0):
                raise ValueError("E2 learning-rate schedule 漂移")
    return int(history.iloc[int(np.argmin(history.val_local_rr_mae.to_numpy()))].epoch)


def validate_run(run_dir: Path, cfg: DictConfig, rows: pd.DataFrame) -> dict[str, Any]:
    saved = OmegaConf.load(run_dir / "config.yaml")
    if OmegaConf.to_container(saved, resolve=True) != OmegaConf.to_container(cfg, resolve=True):
        raise ValueError("E2 保存配置漂移")
    history = pd.read_csv(run_dir / "train_history.csv")
    best_epoch = validate_history(history, cfg)
    for name, expected_epoch in (("checkpoint_best_local_rr.pt", best_epoch), ("checkpoint_final.pt", EPOCHS)):
        checkpoint = torch.load(run_dir / name, map_location="cpu", weights_only=False)
        finite_tree(checkpoint)
        if int(checkpoint["epoch"]) != expected_epoch or checkpoint["config"] != OmegaConf.to_container(cfg, resolve=True):
            raise ValueError("E2 checkpoint epoch/config identity 漂移")
        history_row = history.loc[history.epoch.eq(expected_epoch)].iloc[0]
        for key in ("val_local_rr_mae", "train_loss_total", "optimizer_update"):
            if not np.isclose(float(checkpoint["metrics"][key]), float(history_row[key]), atol=1e-12, rtol=0):
                raise ValueError(f"E2 checkpoint 与对应 history 不一致: {key}")
        extra = checkpoint["extra_state"]
        if (extra["protocol"] != PROTOCOL or int(extra["update_index"]) != expected_epoch * UPDATES_PER_EPOCH
                or int(extra["total_updates"]) != EPOCHS * UPDATES_PER_EPOCH):
            raise ValueError("E2 checkpoint update/protocol 不一致")
    metrics = pd.read_csv(run_dir / "metrics.csv")
    validate_metrics(metrics, rows)
    if not metrics.arm.eq(ARM).all() or not metrics.seed.eq(int(cfg.training.seed)).all():
        raise ValueError("E2 metrics arm/seed identity 不一致")
    summary = pd.read_csv(run_dir / "metrics_summary.csv")
    expected = summarize_task_metrics(metrics)
    for key in PRIMARY:
        if (not np.isfinite(summary.iloc[0][key + "_mean"])
                or not np.isclose(summary.iloc[0][key + "_mean"], expected.iloc[0][key + "_mean"], atol=1e-12, rtol=0)
                or int(summary.iloc[0][key + "_n"]) != int(expected.iloc[0][key + "_n"])):
            raise ValueError(f"E2 summary 或分母不一致: {key}")
    return {"protocol": PROTOCOL, "arm": ARM, "seed": int(cfg.training.seed), "epochs": EPOCHS,
            "updates": EPOCHS * UPDATES_PER_EPOCH, "selected_epoch": best_epoch, "validation_rows": len(metrics),
            "checkpoint_and_history_finite": True, "prediction_degeneracy": 0,
            "validation_row_order_sha256": array_hash(rows.dataset_row_id.to_numpy())}


def run_formal(seed: int, *, gpu_receipt: Path, device: str = "cuda:0") -> Path:
    if seed not in SEEDS:
        raise ValueError("E2 seed 必须属于冻结三 seed 矩阵")
    lock, lock_hash = load_lock()
    parent = ROOT / OUTPUT / "formal" / f"seed_{seed}"
    for existing in parent.glob("*/freeze_receipt.json"):
        manifest = json.loads((existing.parent / "manifest.json").read_text())
        if manifest["implementation_lock_sha256"] == lock_hash:
            verify_attempt(existing.parent, phase="formal", lock_hash=lock_hash)
            raise FileExistsError(f"E2 同一身份 seed 已完成: {existing.parent}")
    with attempt(parent, lock_hash, "formal", seed) as output:
        write_json(output / "environment.json", runtime_preflight(device))
        verify_attempt(gpu_receipt, phase="gpu_smoke", lock_hash=lock_hash)
        smoke = json.loads((gpu_receipt / "gpu_acceptance.json").read_text())
        if smoke.get("passed") is not True or smoke.get("effort_weight") != 0.0:
            raise ValueError("E2 synthetic GPU 验收未通过")
        write_json(output / "gpu_acceptance_source.json", {"path": str(gpu_receipt.resolve()), "manifest": identity(gpu_receipt / "manifest.json")})
        write_json(output / "implementation_lock.json", lock)
        baseline = OmegaConf.create(lock["baselines"][str(seed)])
        cfg = derived_config(baseline, output_root=output / "training", device=device)
        validate_config(cfg, baseline, output_root=output / "training", device=device)
        rows = audit_sources(lock, cfg, output)
        run_dir = EffortAblationExperiment(cfg, rows["val"]).train()
        receipt = validate_run(run_dir, cfg, rows["val"])
        receipt["run_dir"] = str(run_dir.relative_to(output))
        write_json(output / "formal_receipt.json", receipt)
    return output


def paired_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected = {(arm, seed) for arm in ("W0_FULL", ARM) for seed in SEEDS}
    if frame[["arm", "seed"]].duplicated().any() or set(frame[["arm", "seed"]].itertuples(index=False, name=None)) != expected:
        raise ValueError("E2 配对汇总要求两个 arm 的完整三个 seed")
    columns = [key + "_mean" for key in PRIMARY]
    if not np.isfinite(frame[columns].to_numpy()).all():
        raise FloatingPointError("E2 汇总指标非有限")
    full = frame[frame.arm.eq("W0_FULL")].set_index("seed").loc[list(SEEDS)]
    candidate = frame[frame.arm.eq(ARM)].set_index("seed").loc[list(SEEDS)]
    if (full[[key + "_mean" for key in ERRORS]].to_numpy() <= 0).any():
        raise ValueError("E2 FULL error 分母必须大于零")
    paired, aggregate = [], []
    for metric in PRIMARY:
        f = full[metric + "_mean"].to_numpy(dtype=float)
        c = candidate[metric + "_mean"].to_numpy(dtype=float)
        delta = f - c if metric == PCC else 100 * (c - f) / f
        mean_delta = float(f.mean() - c.mean()) if metric == PCC else float(100 * (c.mean() - f.mean()) / f.mean())
        if not np.isfinite(delta).all() or not np.isfinite(mean_delta):
            raise FloatingPointError("E2 paired delta 非有限")
        for seed, a, b, d in zip(SEEDS, f, c, delta, strict=True):
            paired.append({"seed": seed, "metric": metric, "W0_FULL": float(a), ARM: float(b), "delta": float(d),
                           "unit": "absolute_drop" if metric == PCC else "relative_percent"})
        aggregate.append({"metric": metric, "full_mean": float(f.mean()), "full_sample_sd": float(f.std(ddof=1)),
                          "sync_only_mean": float(c.mean()), "sync_only_sample_sd": float(c.std(ddof=1)),
                          "paired_delta_mean": float(delta.mean()), "paired_delta_sample_sd": float(delta.std(ddof=1)),
                          "delta_of_seed_means": mean_delta, "full_better_seeds": int((delta > 0).sum()),
                          "sync_only_better_seeds": int((delta < 0).sum()), "equal_seeds": int((delta == 0).sum())})
    return pd.DataFrame(paired), pd.DataFrame(aggregate)


def summarize(runs: list[Path]) -> Path:
    lock, lock_hash = load_lock()
    if len(runs) != 3 or len({p.resolve() for p in runs}) != 3:
        raise ValueError("E2 summary 需要三个不同的完成 attempt")
    source_runs, candidates = {}, []
    for output in runs:
        manifest = verify_attempt(output, phase="formal", lock_hash=lock_hash)
        receipt = json.loads((output / "formal_receipt.json").read_text())
        seed = int(receipt["seed"])
        if seed not in SEEDS or seed in source_runs or manifest["seed"] != seed:
            raise ValueError("E2 summary seed 重复或越界")
        run_dir = (output / receipt["run_dir"]).resolve()
        if not run_dir.is_relative_to(output.resolve()):
            raise ValueError("E2 training run_dir 越界")
        cfg = OmegaConf.load(run_dir / "config.yaml")
        baseline = OmegaConf.create(lock["baselines"][str(seed)])
        validate_config(cfg, baseline, output_root=output.resolve() / "training", device=str(cfg.training.device))
        validate_run(run_dir, cfg, pd.read_csv(output / "val_rows.csv"))
        summary = pd.read_csv(run_dir / "metrics_summary.csv")
        summary.insert(0, "seed", seed)
        summary.insert(0, "arm", ARM)
        candidates.append(summary)
        source_runs[seed] = {"path": str(output.resolve()), "manifest": identity(output / "manifest.json"), "selected_epoch": receipt["selected_epoch"]}
    if set(source_runs) != set(SEEDS):
        raise ValueError("E2 summary seed 矩阵不完整")
    full_rows = []
    for entry in lock["w0_entries"]:
        path = ROOT / entry["run_dir"] / "metrics_summary.csv"
        verify(path, entry["validation_summary"])
        frame = pd.read_csv(path)
        frame.insert(0, "seed", entry["seed"])
        frame.insert(0, "arm", "W0_FULL")
        full_rows.append(frame)
    combined = pd.concat([*full_rows, *candidates], ignore_index=True)
    for metric in PRIMARY:
        if set(combined[metric + "_n"].astype(int)) != {COUNTS["val"]}:
            raise ValueError("E2 主指标分母与冻结 W0 不一致")
    paired, aggregate = paired_tables(combined)
    parent = ROOT / OUTPUT / "summary"
    for previous in parent.glob("*/freeze_receipt.json"):
        m = json.loads((previous.parent / "manifest.json").read_text())
        if m["implementation_lock_sha256"] == lock_hash:
            raise FileExistsError(f"E2 该实现身份已有完整汇总: {previous.parent}")
    with attempt(parent, lock_hash, "summary") as output:
        combined.to_csv(output / "seed_metrics.csv", index=False)
        paired.to_csv(output / "paired_seed_delta.csv", index=False)
        aggregate.to_csv(output / "three_seed_comparison.csv", index=False)
        write_json(output / "summary_receipt.json", {"protocol": PROTOCOL, "seeds": list(SEEDS),
                   "source_runs": source_runs, "w0_sources": lock["w0_entries"],
                   "delta_definition": "positive means W0_FULL better; error=(sync_only-full)/full*100; PCC=full-sync_only",
                   "interpretation": "complete paired continuous effects and seed directions; effort axes primary, RR/PCC reported jointly"})
    return output


def run_gpu_smoke(device: str = "cuda:0") -> Path:
    lock, lock_hash = load_lock()
    with attempt(ROOT / OUTPUT / "gpu_smoke", lock_hash, "gpu_smoke") as output:
        write_json(output / "environment.json", runtime_preflight(device))
        baseline = OmegaConf.create(lock["baselines"][str(SEEDS[0])])
        cfg = derived_config(baseline, output_root=output / "synthetic", device=device)
        validate_config(cfg, baseline, output_root=output / "synthetic", device=device)
        torch.manual_seed(SEEDS[0])
        model = build_crd_model(cfg).to(device)
        initial = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
        reference = build_crd_model(baseline)
        if not all(torch.equal(value, reference.state_dict()[key]) for key, value in initial.items()):
            raise RuntimeError("E2 与 W0 的同 seed 初始化不一致")
        del reference
        t = torch.arange(18000, dtype=torch.float32) / 100
        target = ((1 + 0.3 * torch.sin(2 * torch.pi * 0.02 * t)) * torch.sin(2 * torch.pi * 0.23 * t))[None, None]
        batch = {"x": target + 0.05 * torch.randn_like(target), "target": target,
                 "tf": {"w": torch.rand(1, 97, 360)}}
        loss_fn = RespirationTaskLoss(cfg).to(device)
        optimizer, _ = build_crd_optimizer(model, cfg)
        summary, updates = train_crd_one_epoch(model, [batch], loss_fn, optimizer, device=device,
                           accumulation_steps=1, update_index=0, total_updates=EPOCHS * UPDATES_PER_EPOCH,
                           max_learning_rate=float(cfg.training.max_learning_rate), min_learning_rate=float(cfg.training.min_learning_rate),
                           warmup_fraction=float(cfg.training.warmup_fraction), grad_clip_norm=float(cfg.training.grad_clip_norm),
                           use_amp=True, show_progress=False)
        finite_tree(model.state_dict())
        finite_tree(optimizer.state_dict())
        finite_tree(summary)
        if updates != 1 or not np.isclose(summary["loss"], summary["loss_sync"], atol=1e-12, rtol=0):
            raise ValueError("E2 synthetic optimizer/loss 合同不一致")
        if not any(not torch.equal(initial[key], value.detach().cpu()) for key, value in model.state_dict().items()):
            raise ValueError("E2 synthetic step 没有更新参数")
        for parameter in model.parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError("E2 synthetic gradient 非有限")
        write_json(output / "gpu_acceptance.json", {"protocol": PROTOCOL, "passed": True, "batch_size": 1,
                   "effort_weight": 0.0, "initialization_matches_w0": True, "optimizer_updates": updates,
                   "finite_parameters_gradients_optimizer": True, "summary": summary, "fixture": "synthetic dynamic waveform and random W"})
        write_json(output / "access_receipt.json", {"waveforms": "synthetic", "checkpoint": "fresh initialization", "historical_artifacts_read": False})
    return output
