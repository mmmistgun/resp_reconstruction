"""E4-v2 四候选×三 seed 的冻结合同、原生训练与完整矩阵汇总。"""

from __future__ import annotations

import json
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from resp_train.crd.config import load_crd_config
from resp_train.crd.experiment import CRDExperiment
from resp_train.crd.training import build_crd_optimizer, crd_learning_rate
from resp_train.crd.tf_w_v2_audit import _w0_seed_entries
from resp_train.metrics.task import summarize_task_metrics
from resp_train.paper_evidence import e4_scale_aggregation as previous
from resp_train.paper_evidence.e1_scale_topology import PRIMARY, ERRORS, PCC, SEEDS, array_hash
from resp_train.paper_evidence.e1_scale_topology_runtime import identity, sha256_file, write_json, git_state
from resp_train.paper_evidence.e4_aggregation_v2_model import (
    PROTOCOL, ARMS, ADDED_PARAMETERS, W0_PARAMETERS, W0_BRANCH_PARAMETERS,
    aggregation_contract, validate_frequency_grid, build_model,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path("runs/e4_scale_aggregation_v2")
LOCK_PATH = Path("docs/experiments/e4_scale_aggregation_v2_implementation_lock_20260917.json")
PROTOCOL_PATH = Path("docs/experiments/e4_scale_aggregation_v2_protocol_20260917.md")
SCRIPT_PATH = Path("scripts/run_e4_aggregation_v2.py")
TEST_PATHS = (Path("tests/test_e4_aggregation_v2_model.py"), Path("tests/test_e4_aggregation_v2.py"))
COUNTS = {"train": 10141, "val": 2675}
EPOCHS, UPDATES_PER_EPOCH = 80, 80
verify, finite_tree, validate_metrics = previous.verify, previous.finite_tree, previous.validate_metrics
phase_guard, runtime_preflight = previous.phase_guard, previous.runtime_preflight


def derived_config(baseline, arm: str, frequencies, *, output_root: Path, device: str):
    cfg = OmegaConf.create(OmegaConf.to_container(baseline, resolve=True))
    cfg.protocol.name, cfg.protocol.execution_gate = PROTOCOL, "e4_v2_formal"
    cfg.model.aggregation_v2 = aggregation_contract(arm)
    cfg.model.aggregation_frequencies_hz = list(map(float, frequencies))
    cfg.training.device, cfg.training.show_progress = device, False
    cfg.outputs.run_root = str(output_root)
    return cfg


def validate_config(cfg, baseline, arm: str, frequencies, *, output_root: Path, device: str):
    expected = derived_config(baseline, arm, frequencies, output_root=output_root, device=device)
    if OmegaConf.to_container(cfg, resolve=True) != OmegaConf.to_container(expected, resolve=True):
        raise ValueError("E4-v2 配置必须与同 seed W0 合同一致，仅开放聚合与运行身份")
    if (baseline.model.variant != "crd_tf102_w" or baseline.loss.effort_weight != .25 or baseline.loss.sync_weight != 1
            or baseline.training.seed not in SEEDS or baseline.model.initialization_seed != baseline.training.seed
            or baseline.training.epochs != EPOCHS or baseline.training.early_stopping_enabled
            or baseline.training.batch_size != 128 or baseline.training.gradient_accumulation_steps != 1
            or not baseline.training.use_amp or baseline.training.amp_dtype != "bfloat16"
            or baseline.data.max_train_windows is not None or baseline.data.max_val_windows is not None):
        raise ValueError("E4-v2 W0 来源不符合冻结合同")
    validate_frequency_grid(np.asarray(frequencies, dtype=np.float64))


def prepare_lock(root: Path = ROOT) -> Path:
    path = root / LOCK_PATH
    if path.exists():
        raise FileExistsError("E4-v2 implementation lock 已存在")
    source_path = root / previous.SOURCE_LOCK
    if sha256_file(source_path) != previous.SOURCE_SHA:
        raise ValueError("E4-v2 W0 来源锁漂移")
    source = json.loads(source_path.read_text())
    entries = _w0_seed_entries(source)
    files = {str(previous.SOURCE_LOCK): identity(source_path)}
    audit_path = Path("docs/experiments/e4_w0_scale_aggregation_source_audit_20260917.json")
    audit = json.loads((root / audit_path).read_text())
    files[str(audit_path)] = identity(root / audit_path)
    baselines = {}
    for entry in entries:
        for key, filename in (("checkpoint", "checkpoint_best_local_rr.pt"), ("config", "config.yaml"),
                              ("manifest", "run_manifest.json"), ("validation_summary", "metrics_summary.csv")):
            relative = str(Path(entry["run_dir"]) / filename)
            verify(root / relative, entry[key]); files[relative] = entry[key]
        relative = str(Path(entry["run_dir"]) / "metrics.csv")
        verify(root / relative, audit["verified_files"][relative]); files[relative] = audit["verified_files"][relative]
        cfg = load_crd_config(root / entry["run_dir"] / "config.yaml")
        baselines[str(entry["seed"])] = OmegaConf.to_container(cfg, resolve=True)
    cache = source["cache_lock"]
    for key in ("manifest", "train_w", "val_w", "frequency_file"):
        item = cache[key]
        expected = {"size_bytes": item["size_bytes"], "sha256": item.get("sha256", item.get("file_sha256"))}
        verify(root / item["path"], expected); files[item["path"]] = expected
    for split in COUNTS:
        relative = str(Path(cache["root"]) / f"{split}_row_ids.npy")
        observed = identity(root / relative)
        if observed["sha256"] != cache["row_identity"][f"{split}_row_file_sha256"]:
            raise ValueError("E4-v2 row identity 漂移")
        files[relative] = observed
    frequency = validate_frequency_grid(np.load(root / cache["frequency_file"]["path"], allow_pickle=False))
    templates = {}
    for arm in ARMS:
        templates[arm] = {}
        for seed in SEEDS:
            baseline = OmegaConf.create(baselines[str(seed)])
            output_root = root / OUTPUT / "formal" / arm / f"seed_{seed}"
            cfg = derived_config(baseline, arm, frequency["values_hz"], output_root=output_root, device="cuda:0")
            validate_config(cfg, baseline, arm, frequency["values_hz"], output_root=output_root, device="cuda:0")
            templates[arm][str(seed)] = OmegaConf.to_container(cfg, resolve=True)
    manifest = json.loads((root / cache["manifest"]["path"]).read_text())
    # test 专属控制器由后续 test lock 绑定；其维护不改变已完成训练的代码 identity。
    paths = [p for p in sorted((root / "resp_train").rglob("*.py"))
             if p.relative_to(root).as_posix() != "resp_train/paper_evidence/e4_aggregation_v2_test.py"]
    paths += [root / p for p in (SCRIPT_PATH, PROTOCOL_PATH, *TEST_PATHS)]
    lock = {"protocol": PROTOCOL, "arms": list(ARMS), "seeds": list(SEEDS), "counts": COUNTS,
            "epochs": EPOCHS, "updates_per_epoch": UPDATES_PER_EPOCH,
            "contracts": {arm: aggregation_contract(arm) for arm in ARMS},
            "baselines": baselines, "resolved_templates": templates, "w0_entries": entries,
            "frequency": frequency, "cache_lock": cache, "source_files": files,
            "dataset_index": {"path": manifest["dataset_index"], "sha256": manifest["dataset_index_sha256"]},
            "code_files": {str(p.relative_to(root)): identity(p) for p in paths},
            "preparation_git": git_state(root), "prepared_at": datetime.now(timezone.utc).isoformat()}
    write_json(path, lock)
    return path


def load_lock(root: Path = ROOT):
    path = root / LOCK_PATH
    lock = json.loads(path.read_text())
    if (lock["protocol"] != PROTOCOL or lock["arms"] != list(ARMS) or lock["seeds"] != list(SEEDS)
            or lock["counts"] != COUNTS or lock["epochs"] != EPOCHS or lock["updates_per_epoch"] != UPDATES_PER_EPOCH
            or lock["contracts"] != {arm: aggregation_contract(arm) for arm in ARMS}
            or validate_frequency_grid(np.asarray(lock["frequency"]["values_hz"], dtype=np.float64)) != lock["frequency"]):
        raise ValueError("E4-v2 lock 科学矩阵漂移")
    for relative, expected in lock["code_files"].items():
        verify(root / relative, expected)
    for arm in ARMS:
        for seed in SEEDS:
            validate_config(OmegaConf.create(lock["resolved_templates"][arm][str(seed)]),
                            OmegaConf.create(lock["baselines"][str(seed)]), arm, lock["frequency"]["values_hz"],
                            output_root=root / OUTPUT / "formal" / arm / f"seed_{seed}", device="cuda:0")
    return lock, sha256_file(path)


@contextmanager
def attempt(parent: Path, lock_hash: str, phase: str, arm: str | None = None, seed: int | None = None):
    parent.mkdir(parents=True, exist_ok=True)
    output = parent / f"{phase}_{lock_hash[:12]}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:12]}"
    output.mkdir(exist_ok=False)
    context = {"protocol": PROTOCOL, "phase": phase, "arm": arm, "seed": seed,
               "implementation_lock_sha256": lock_hash, "command": sys.argv,
               "started_at": datetime.now(timezone.utc).isoformat()}
    write_json(output / "lifecycle_started.json", {**context, "status": "running"})
    print(f"E4-v2 attempt: {output}", flush=True)
    try:
        yield output
        write_json(output / "lifecycle_completed.json", {**context, "status": "completed", "ended_at": datetime.now(timezone.utc).isoformat()})
        write_json(output / "manifest.json", {**context, "status": "completed", "files": {
            str(p.relative_to(output)): identity(p) for p in sorted(output.rglob("*")) if p.is_file()}})
        write_json(output / "freeze_receipt.json", {"protocol": PROTOCOL, "manifest": identity(output / "manifest.json")})
    except BaseException as exc:
        write_json(output / "lifecycle_failed.json", {**context, "status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
        raise


def verify_attempt(output: Path, *, phase: str, lock_hash: str):
    output = output.resolve()
    if (output / "lifecycle_failed.json").exists():
        raise ValueError("E4-v2 attempt 已失败")
    freeze = json.loads((output / "freeze_receipt.json").read_text())
    verify(output / "manifest.json", freeze["manifest"])
    manifest = json.loads((output / "manifest.json").read_text())
    if (manifest["protocol"] != PROTOCOL or manifest["phase"] != phase or manifest["status"] != "completed"
            or manifest["implementation_lock_sha256"] != lock_hash):
        raise ValueError("E4-v2 attempt identity 不匹配")
    required = {"lifecycle_completed.json"} | {
        "formal": {"formal_receipt.json", "environment.json", "implementation_lock.json", "access_receipt.json"},
        "summary": {"summary_receipt.json", "seed_metrics.csv", "paired_seed_delta.csv", "four_arm_comparison.csv"},
        "gpu_acceptance": {"gpu_acceptance.json", "environment.json", "access_receipt.json"},
        "benchmark": {"benchmark.json", "environment.json", "access_receipt.json"},
    }[phase]
    if not required.issubset(manifest["files"]):
        raise ValueError("E4-v2 manifest 缺少必需产物")
    for relative, expected in manifest["files"].items():
        path = (output / relative).resolve()
        if not path.is_relative_to(output):
            raise ValueError("E4-v2 manifest 路径越界")
        verify(path, expected)
    return manifest


def reject_completed(parent: Path, phase: str, lock_hash: str):
    for previous_receipt in parent.glob("*/freeze_receipt.json"):
        manifest = json.loads((previous_receipt.parent / "manifest.json").read_text())
        if manifest["implementation_lock_sha256"] == lock_hash:
            verify_attempt(previous_receipt.parent, phase=phase, lock_hash=lock_hash)
            raise FileExistsError(f"E4-v2 相同身份已完成: {previous_receipt.parent}")


def audit_sources(lock, cfg, output):
    return previous.audit_sources(lock, cfg, output)


class AggregationExperiment(CRDExperiment):
    task_name = "e4_scale_aggregation_v2"

    def __init__(self, cfg, validation_rows):
        super().__init__(cfg)
        self.validation_rows = validation_rows

    def _build_model(self):
        return build_model(self.cfg)

    def _evaluate_model(self, model, loader, **kwargs):
        metrics = super()._evaluate_model(model, loader, **kwargs)
        validate_metrics(metrics, self.validation_rows)
        metrics.insert(0, "arm", str(self.cfg.model.aggregation_v2.arm))
        metrics.insert(0, "seed", int(self.cfg.training.seed))
        return metrics


def validate_history(history, cfg):
    if len(history) != EPOCHS or not np.array_equal(history.epoch, np.arange(1, EPOCHS + 1)):
        raise ValueError("E4-v2 history epoch 不完整")
    if not np.array_equal(history.optimizer_update, np.arange(1, EPOCHS + 1) * UPDATES_PER_EPOCH):
        raise ValueError("E4-v2 history updates 不完整")
    if not np.isfinite(history.select_dtypes(include=np.number).to_numpy()).all():
        raise FloatingPointError("E4-v2 history 非有限")
    if not np.allclose(history.train_loss_total, history.train_loss_sync + .25 * history.train_loss_effort, rtol=0, atol=1e-12):
        raise ValueError("E4-v2 完整 loss 漂移")
    for row in history.itertuples():
        for key, update in (("first_learning_rate", (row.epoch - 1) * UPDATES_PER_EPOCH), ("last_learning_rate", row.epoch * UPDATES_PER_EPOCH - 1)):
            expected = crd_learning_rate(update, total_updates=EPOCHS * UPDATES_PER_EPOCH,
                max_learning_rate=cfg.training.max_learning_rate, min_learning_rate=cfg.training.min_learning_rate,
                warmup_fraction=cfg.training.warmup_fraction)
            if not np.isclose(getattr(row, key), expected, rtol=0, atol=1e-15):
                raise ValueError("E4-v2 LR schedule 漂移")
    return int(history.iloc[int(np.argmin(history.val_local_rr_mae.to_numpy()))].epoch)


def validate_run(run_dir: Path, cfg, rows):
    config = OmegaConf.to_container(cfg, resolve=True)
    if OmegaConf.to_container(OmegaConf.load(run_dir / "config.yaml"), resolve=True) != config:
        raise ValueError("E4-v2 保存配置漂移")
    history = pd.read_csv(run_dir / "train_history.csv")
    best = validate_history(history, cfg)
    model = build_model(cfg)
    optimizer, partition = build_crd_optimizer(model, cfg)
    groups = json.loads((run_dir / "optimizer_parameter_groups.json").read_text())
    if groups != {"weight_decay": float(cfg.training.weight_decay), "decay": list(partition.decay_names), "no_decay": list(partition.no_decay_names)}:
        raise ValueError("E4-v2 optimizer 分组漂移")
    for filename, epoch in (("checkpoint_best_local_rr.pt", best), ("checkpoint_final.pt", EPOCHS)):
        checkpoint = torch.load(run_dir / filename, map_location="cpu", weights_only=False)
        finite_tree(checkpoint)
        if checkpoint["config"] != config or checkpoint["epoch"] != epoch:
            raise ValueError("E4-v2 checkpoint 配置/epoch 错误")
        record = history.loc[history.epoch.eq(epoch)].iloc[0]
        for key in history.columns:
            if key not in checkpoint["metrics"] or not np.isclose(checkpoint["metrics"][key], record[key], atol=1e-12, rtol=1e-12):
                raise ValueError(f"E4-v2 checkpoint 与 history 不一致: {key}")
        extra = checkpoint["extra_state"]
        if extra["protocol"] != PROTOCOL or extra["update_index"] != epoch * UPDATES_PER_EPOCH or extra["total_updates"] != EPOCHS * UPDATES_PER_EPOCH:
            raise ValueError("E4-v2 checkpoint update/protocol 错误")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                state = optimizer.state.get(parameter, {})
                if (not {"step", "exp_avg", "exp_avg_sq"}.issubset(state) or float(state["step"]) != epoch * UPDATES_PER_EPOCH
                        or state["exp_avg"].shape != parameter.shape or state["exp_avg_sq"].shape != parameter.shape):
                    raise ValueError("E4-v2 optimizer state/step 不完整")
    metrics = pd.read_csv(run_dir / "metrics.csv")
    degeneracy = validate_metrics(metrics, rows)
    arm, seed = str(cfg.model.aggregation_v2.arm), int(cfg.training.seed)
    if not metrics.arm.eq(arm).all() or not metrics.seed.eq(seed).all():
        raise ValueError("E4-v2 metrics arm/seed 错误")
    saved = pd.read_csv(run_dir / "metrics_summary.csv")
    expected = summarize_task_metrics(metrics)
    if len(saved) != 1:
        raise ValueError("E4-v2 summary 必须恰有一行")
    for key in PRIMARY:
        if (not np.isfinite(saved.iloc[0][key + "_mean"])
                or not np.isclose(saved.iloc[0][key + "_mean"], expected.iloc[0][key + "_mean"], rtol=0, atol=1e-12)
                or saved.iloc[0][key + "_n"] != expected.iloc[0][key + "_n"]):
            raise ValueError("E4-v2 summary 数值/分母不一致")
    return {"protocol": PROTOCOL, "arm": arm, "seed": seed, "epochs": EPOCHS, "updates": EPOCHS * UPDATES_PER_EPOCH,
            "selected_epoch": best, "validation_rows": len(metrics), "prediction_degeneracy": degeneracy,
            "quality_acceptance_passed": not any(degeneracy.values()), "validation_row_order_sha256": array_hash(rows.dataset_row_id.to_numpy())}


def run_formal(arm: str, seed: int, *, gpu_receipt: Path, device: str = "cuda:0"):
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError("E4-v2 arm/seed 不在固定矩阵")
    lock, digest = load_lock()
    parent = ROOT / OUTPUT / "formal" / arm / f"seed_{seed}"
    with phase_guard(parent, digest):
        reject_completed(parent, "formal", digest)
        with attempt(parent, digest, "formal", arm, seed) as output:
            env = runtime_preflight(device)
            write_json(output / "environment.json", env)
            verify_attempt(gpu_receipt, phase="gpu_acceptance", lock_hash=digest)
            gpu = json.loads((gpu_receipt / "gpu_acceptance.json").read_text())
            if gpu.get("passed") is not True or gpu.get("arms") != list(ARMS) or gpu.get("seeds") != list(SEEDS) or gpu.get("physical_batch") != 128:
                raise ValueError("E4-v2 全矩阵 GPU 验收不完整")
            old_env = json.loads((gpu_receipt / "environment.json").read_text())
            for key in ("git", "packages", "gpu_name", "cuda", "cudnn", "cudnn_benchmark", "cudnn_deterministic", "matmul_allow_tf32", "cudnn_allow_tf32", "deterministic_algorithms"):
                if env.get(key) != old_env.get(key):
                    raise ValueError(f"E4-v2 GPU 验收与训练环境不一致: {key}")
            write_json(output / "gpu_source.json", {"path": str(gpu_receipt.resolve()), "manifest": identity(gpu_receipt / "manifest.json")})
            write_json(output / "implementation_lock.json", lock)
            baseline = OmegaConf.create(lock["baselines"][str(seed)])
            cfg = derived_config(baseline, arm, lock["frequency"]["values_hz"], output_root=output / "training", device=device)
            validate_config(cfg, baseline, arm, lock["frequency"]["values_hz"], output_root=output / "training", device=device)
            rows = audit_sources(lock, cfg, output)
            run = AggregationExperiment(cfg, rows["val"]).train()
            receipt = validate_run(run, cfg, rows["val"])
            previous.validate_anchor_rows(pd.read_csv(run / "metrics.csv"), lock, seed)
            write_json(output / "formal_receipt.json", {**receipt, "run_dir": str(run.relative_to(output))})
    return output


def paired_tables(frame: pd.DataFrame, *, split: str):
    expected = {(arm, seed) for arm in ("W0_FULL", *ARMS) for seed in SEEDS}
    if (frame[["arm", "seed"]].duplicated().any()
            or set(frame[["arm", "seed"]].itertuples(index=False, name=None)) != expected
            or set(frame.split) != {split}):
        raise ValueError("E4-v2 配对表要求 W0 与四候选的完整三 seed 矩阵和唯一 split")
    if not np.isfinite(frame[[key + "_mean" for key in PRIMARY]].to_numpy()).all():
        raise FloatingPointError("E4-v2 汇总指标非有限")
    if (frame[[key + "_mean" for key in ERRORS]].to_numpy() < 0).any():
        raise ValueError("E4-v2 error 不得为负")
    indexed = frame.set_index(["arm", "seed"])
    paired, aggregate = [], []
    for arm in ARMS:
        for metric in PRIMARY:
            f = indexed.loc[[("W0_FULL", s) for s in SEEDS], metric + "_mean"].to_numpy(dtype=float)
            c = indexed.loc[[(arm, s) for s in SEEDS], metric + "_mean"].to_numpy(dtype=float)
            raw = f - c if metric == PCC else c - f
            defined = np.ones(3, dtype=bool) if metric == PCC else f != 0
            delta = raw if metric == PCC else np.divide(100 * raw, f, out=np.full_like(f, np.nan), where=defined)
            mean_delta = float(raw.mean()) if metric == PCC else (float(100 * raw.mean() / f.mean()) if f.mean() else np.nan)
            if not np.isfinite(raw).all() or not np.isfinite(delta[defined]).all() or (metric == PCC or f.mean() != 0) and not np.isfinite(mean_delta):
                raise FloatingPointError("E4-v2 配对差值非有限")
            for i, seed in enumerate(SEEDS):
                item = {"arm": arm, "seed": seed, "split": split, "metric": metric, "W0_FULL": f[i],
                        "candidate": c[i], "raw_delta": raw[i], "delta": delta[i], "relative_delta_defined": bool(defined[i]),
                        "unit": "absolute_drop" if metric == PCC else "relative_percent"}
                for side, source_arm in (("full", "W0_FULL"), ("candidate", arm)):
                    row = indexed.loc[(source_arm, seed)]
                    for key in ("selected_epoch", "source_sha256"):
                        if key in row:
                            item[f"{side}_{key}"] = row[key]
                    item[f"{side}_n"] = int(row[metric + "_n"])
                paired.append(item)
            aggregate.append({"arm": arm, "split": split, "metric": metric, "full_mean": f.mean(), "full_sample_sd": f.std(ddof=1),
                "candidate_mean": c.mean(), "candidate_sample_sd": c.std(ddof=1), "paired_delta_mean": delta.mean(),
                "paired_delta_sample_sd": delta.std(ddof=1), "paired_raw_delta_mean": raw.mean(), "paired_raw_delta_sample_sd": raw.std(ddof=1),
                "relative_delta_defined_seeds": int(defined.sum()), "delta_of_seed_means": mean_delta,
                "candidate_better_seeds": int((raw < 0).sum()), "full_better_seeds": int((raw > 0).sum()), "equal_seeds": int((raw == 0).sum())})
    return pd.DataFrame(paired), pd.DataFrame(aggregate)


def summarize(runs: list[Path]):
    lock, digest = load_lock()
    if len(runs) != len(ARMS) * len(SEEDS) or len({p.resolve() for p in runs}) != len(runs):
        raise ValueError("E4-v2 汇总要求 12 个不同的完成 attempt")
    parent = ROOT / OUTPUT / "summary"
    with phase_guard(parent, digest):
        reject_completed(parent, "summary", digest)
        frames, sources = [], {}
        for output in runs:
            manifest = verify_attempt(output, phase="formal", lock_hash=digest)
            receipt = json.loads((output / "formal_receipt.json").read_text())
            arm, seed = receipt["arm"], receipt["seed"]
            key = f"{arm}/{seed}"
            if arm not in ARMS or seed not in SEEDS or key in sources or manifest["arm"] != arm or manifest["seed"] != seed:
                raise ValueError("E4-v2 汇总 arm/seed 身份重复或越界")
            run = (output / receipt["run_dir"]).resolve()
            if not run.is_relative_to(output.resolve()):
                raise ValueError("E4-v2 training run_dir 越界")
            cfg = OmegaConf.load(run / "config.yaml")
            validate_config(cfg, OmegaConf.create(lock["baselines"][str(seed)]), arm, lock["frequency"]["values_hz"],
                            output_root=output.resolve() / "training", device=str(cfg.training.device))
            checked = validate_run(run, cfg, pd.read_csv(output / "val_rows.csv"))
            if any(receipt.get(k) != v for k, v in checked.items()):
                raise ValueError("E4-v2 formal receipt 与产物不一致")
            previous.validate_anchor_rows(pd.read_csv(run / "metrics.csv"), lock, seed)
            summary = pd.read_csv(run / "metrics_summary.csv")
            for name, value in {"arm": arm, "seed": seed, "split": "val", "selected_epoch": receipt["selected_epoch"],
                                "source_sha256": sha256_file(run / "metrics.csv"), "quality_acceptance_passed": checked["quality_acceptance_passed"]}.items():
                summary.insert(0, name, value)
            frames.append(summary)
            sources[key] = {"arm": arm, "seed": seed, "path": str(output.resolve()), "manifest": identity(output / "manifest.json"), "selected_epoch": receipt["selected_epoch"]}
        for entry in lock["w0_entries"]:
            path = ROOT / entry["run_dir"] / "metrics_summary.csv"
            verify(path, entry["validation_summary"])
            summary = pd.read_csv(path)
            for key, value in {"arm": "W0_FULL", "seed": entry["seed"], "split": "val", "selected_epoch": entry["selected_epoch"],
                               "source_sha256": sha256_file(path), "quality_acceptance_passed": True}.items():
                summary.insert(0, key, value)
            frames.append(summary)
        combined = pd.concat(frames, ignore_index=True)
        for metric in PRIMARY:
            if set(combined[metric + "_n"]) != {COUNTS["val"]}:
                raise ValueError("E4-v2 validation 主指标分母漂移")
        paired, comparison = paired_tables(combined, split="val")
        with attempt(parent, digest, "summary") as output:
            combined.to_csv(output / "seed_metrics.csv", index=False)
            paired.to_csv(output / "paired_seed_delta.csv", index=False, na_rep="NA")
            comparison.to_csv(output / "four_arm_comparison.csv", index=False, na_rep="NA")
            write_json(output / "summary_receipt.json", {"protocol": PROTOCOL, "split": "val", "arms": list(ARMS), "seeds": list(SEEDS),
                "source_runs": sources, "w0_sources": lock["w0_entries"], "implementation_lock_sha256": digest,
                "delta_definition": "error=(candidate-W0)/W0*100; PCC=W0-candidate; positive=worse"})
            write_json(output / "parameter_compute_report.json", parameter_compute_report())
    return output


def completed_runs():
    """只收集当前锁下每个固定 cell 的唯一成功 attempt，随后交给汇总器完整核验。"""
    _, digest = load_lock()
    result = []
    for arm in ARMS:
        for seed in SEEDS:
            parent = ROOT / OUTPUT / "formal" / arm / f"seed_{seed}"
            matches = []
            for receipt in parent.glob("*/freeze_receipt.json"):
                manifest = json.loads((receipt.parent / "manifest.json").read_text())
                if manifest.get("implementation_lock_sha256") == digest:
                    if manifest.get("arm") != arm or manifest.get("seed") != seed or manifest.get("phase") != "formal":
                        raise ValueError("E4-v2 完成目录与 manifest cell 不一致")
                    matches.append(receipt.parent.resolve())
            if len(matches) != 1:
                raise ValueError(f"E4-v2 {arm}/{seed} 需要唯一成功 attempt，实际 {len(matches)}")
            result.extend(matches)
    return result


def parameter_compute_report():
    return {"arms": {arm: {"added_parameters": ADDED_PARAMETERS[arm], "model_parameters": W0_PARAMETERS + ADDED_PARAMETERS[arm],
                "branch_parameters": W0_BRANCH_PARAMETERS + ADDED_PARAMETERS[arm],
                "attention_content_score_macs": (96 * 8 + 8) * 97 * 360 if "attention" in arm else 0,
                "frequency_projection_multiplies": 8 * 97 if arm == "frequency_attention" else 0,
                "weighted_pool_products": 96 * (4 if arm == "channel_region" else 97) * 360}
            for arm in ARMS}, "unit": "per 180s window, forward",
            "coverage": "aggregation score convolution MACs and weighted pooling products reported separately",
            "whole_model_flops": None, "excluded": ["exp/normalization/reductions", "SiLU", "interpolation", "Mamba/FFT", "backward/recomputation"]}
