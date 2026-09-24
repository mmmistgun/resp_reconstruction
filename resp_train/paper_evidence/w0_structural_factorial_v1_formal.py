"""W0 三因素结构对照 P3：单 cell 正式训练与产物验收。"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from resp_train.crd.config import check_crd_dependencies, crd_dependency_versions
from resp_train.crd.training import build_crd_optimizer
from resp_train.data.index import filter_index
from resp_train.data.research_v2 import read_research_v2_index
from resp_train.metrics.task import summarize_task_metrics
from resp_train.paper_evidence import w0_structural_factorial_v1 as sf
from resp_train.paper_evidence.w0_structural_factorial_v1_engineering import finite_tree
from resp_train.paper_evidence.w0_structural_factorial_v1_model import (
    ARMS,
    ARM_SPECS,
    build_w0_structural_factorial_model,
)


P2_LOCK_PATH = Path("docs/experiments/w0_structural_factorial_v1_implementation_lock_r2_20260924.json")
P2_LOCK_SHA256 = "32141eab672ea41435055c45cbd7ec96325f8ddef2481a210db2941222c9e9f3"
P2_RECEIPT_PATH = Path("docs/experiments/w0_structural_factorial_v1_p2_engineering_receipt_20260924.md")
FORMAL_SOURCE_PATH = Path("resp_train/paper_evidence/w0_structural_factorial_v1_formal.py")

P2_ACCEPTANCE = Path(
    "runs/w0_structural_factorial_v1_es30p15/gpu_acceptance/"
    "gpu_acceptance_32141eab672e_20260924T074944Z_44fa12038854"
)
P2_BENCHMARK = Path(
    "runs/w0_structural_factorial_v1_es30p15/benchmark/"
    "benchmark_32141eab672e_20260924T075037Z_ab5670de3e59"
)
P2_IDENTITIES: dict[str, dict[str, dict[str, Any]]] = {
    "gpu_acceptance": {
        "gpu_acceptance.json": {
            "size_bytes": 12_635,
            "sha256": "0a1d642795871d85bfb039c55fcf0284443c31922c695dcc5231ac7e95fe843e",
        },
        "manifest.json": {
            "size_bytes": 4_890,
            "sha256": "6d49aa1b5889c67d176d205e7a035d5d541a45f5ab473a5b4502d7ff54674dd9",
        },
        "freeze_receipt.json": {
            "size_bytes": 190,
            "sha256": "48669632ad609f2884ca535236a511d0c12cf498696a7778996a1a0fd462f9a6",
        },
    },
    "benchmark": {
        "benchmark.json": {
            "size_bytes": 26_401,
            "sha256": "f21a073c269351baa2b0a6ee7924adf0557efe726f98cf2061147c9d2dc561fe",
        },
        "manifest.json": {
            "size_bytes": 6_265,
            "sha256": "214f4cd8b555569600f7dcf5577a2a6899d5bf00650c3f5be817b2527118568b",
        },
        "freeze_receipt.json": {
            "size_bytes": 190,
            "sha256": "234f4a15b6923fb9c105b14e148c222290e8977c0852b83373dd9c6ecc773e9d",
        },
    },
}


def array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def state_dict_identity(state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    digest = hashlib.sha256()
    tensors: dict[str, dict[str, Any]] = {}
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
        item_hash = hashlib.sha256(raw).hexdigest()
        shape = list(tensor.shape)
        dtype = str(tensor.dtype)
        digest.update(name.encode("utf-8"))
        digest.update(dtype.encode("ascii"))
        digest.update(json.dumps(shape, separators=(",", ":")).encode("ascii"))
        digest.update(bytes.fromhex(item_hash))
        tensors[name] = {
            "shape": shape,
            "dtype": dtype,
            "sha256": item_hash,
        }
    return {
        "state_sha256": digest.hexdigest(),
        "tensor_count": len(tensors),
        "tensors": tensors,
    }


def _p2_evidence_payload() -> dict[str, Any]:
    return {
        "implementation_lock": {
            "path": str(P2_LOCK_PATH),
            **sf.identity(sf.ROOT / P2_LOCK_PATH),
        },
        "gpu_acceptance": {
            "path": str(P2_ACCEPTANCE),
            "files": P2_IDENTITIES["gpu_acceptance"],
        },
        "benchmark": {
            "path": str(P2_BENCHMARK),
            "files": P2_IDENTITIES["benchmark"],
        },
        "engineering_receipt": {
            "path": str(P2_RECEIPT_PATH),
            **sf.identity(sf.ROOT / P2_RECEIPT_PATH),
        },
    }


def formal_source_identity() -> dict[str, Any]:
    paths = tuple(dict.fromkeys((*sf.critical_paths(), FORMAL_SOURCE_PATH, P2_RECEIPT_PATH)))
    return {
        "git": sf.git_state(),
        "files": {str(path): sf.identity(sf.ROOT / path) for path in paths},
    }


def _verify_frozen_attempt(path: Path, expected: Mapping[str, Mapping[str, Any]], phase: str) -> dict[str, Any]:
    for filename, file_identity in expected.items():
        sf.verify_identity(path / filename, file_identity)
    freeze = json.loads((path / "freeze_receipt.json").read_text(encoding="utf-8"))
    sf.verify_identity(path / "manifest.json", freeze["manifest"])
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("protocol") != sf.PROTOCOL
        or manifest.get("phase") != phase
        or manifest.get("status") != "completed"
        or manifest.get("implementation_lock_sha256") != P2_LOCK_SHA256
    ):
        raise ValueError(f"P2 {phase} lifecycle identity 不一致")
    for relative, file_identity in manifest["files"].items():
        target = (path / relative).resolve()
        if not target.is_relative_to(path.resolve()):
            raise ValueError("P2 manifest 路径越界")
        sf.verify_identity(target, file_identity)
    return manifest


def verify_p2_evidence(evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
    expected = _p2_evidence_payload() if evidence is None else dict(evidence)
    lock_entry = expected["implementation_lock"]
    if lock_entry["path"] != str(P2_LOCK_PATH):
        raise ValueError("P2 implementation lock 路径漂移")
    sf.verify_identity(sf.ROOT / P2_LOCK_PATH, lock_entry)
    if lock_entry["sha256"] != P2_LOCK_SHA256:
        raise ValueError("P2 implementation lock SHA-256 漂移")

    acceptance_entry = expected["gpu_acceptance"]
    acceptance_path = sf.SOURCE_ROOT / acceptance_entry["path"]
    _verify_frozen_attempt(acceptance_path, acceptance_entry["files"], "gpu_acceptance")
    acceptance = json.loads((acceptance_path / "gpu_acceptance.json").read_text(encoding="utf-8"))
    batch1 = acceptance.get("batch1", [])
    if (
        acceptance.get("passed") is not True
        or acceptance.get("implementation_lock_sha256") != P2_LOCK_SHA256
        or [item.get("arm") for item in batch1] != list(ARMS)
        or any(item.get("batch_size") != 1 for item in batch1)
        or acceptance.get("max_resource_batch128", {}).get("batch_size") != 128
        or acceptance.get("max_resource_batch128", {}).get("updates") != 3
        or acceptance.get("native_lifecycle", {}).get("train_windows") != 128
        or acceptance.get("native_lifecycle", {}).get("validation_windows") != 32
    ):
        raise ValueError("P2 GPU acceptance 合同不完整")

    benchmark_entry = expected["benchmark"]
    benchmark_path = sf.SOURCE_ROOT / benchmark_entry["path"]
    _verify_frozen_attempt(benchmark_path, benchmark_entry["files"], "benchmark")
    benchmark = json.loads((benchmark_path / "benchmark.json").read_text(encoding="utf-8"))
    measurements = benchmark.get("measurements", [])
    expected_cells = {(arm, mode) for arm in ARMS for mode in ("eval", "train")}
    observed_cells = {(item.get("arm"), item.get("mode")) for item in measurements}
    if (
        benchmark.get("implementation_lock_sha256") != P2_LOCK_SHA256
        or len(measurements) != len(expected_cells)
        or observed_cells != expected_cells
        or any(item.get("peak_reserved_fraction", 1.0) > 0.8 for item in measurements)
        or any(item.get("batch_size") != (1 if item.get("mode") == "eval" else 128) for item in measurements)
    ):
        raise ValueError("P2 benchmark 合同不完整")

    receipt_entry = expected["engineering_receipt"]
    if receipt_entry["path"] != str(P2_RECEIPT_PATH):
        raise ValueError("P2 工程回执路径漂移")
    sf.verify_identity(sf.ROOT / P2_RECEIPT_PATH, receipt_entry)
    return {
        "gpu_acceptance_manifest": sf.identity(acceptance_path / "manifest.json"),
        "benchmark_manifest": sf.identity(benchmark_path / "manifest.json"),
        "max_acceptance_reserved_fraction": float(
            acceptance["max_resource_batch128"]["peak_reserved_fraction"]
        ),
        "max_benchmark_reserved_fraction": float(
            max(item["peak_reserved_fraction"] for item in measurements)
        ),
    }


def load_formal_contract(root: Path = sf.ROOT) -> tuple[dict[str, Any], str]:
    """复用实验锁的科学合同；正式实现身份由运行时 Git 与源码 manifest 固化。"""

    path = root / P2_LOCK_PATH
    if sf.sha256_file(path) != P2_LOCK_SHA256:
        raise ValueError("实验锁身份漂移")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if (
        lock.get("protocol") != sf.PROTOCOL
        or tuple(lock.get("arms", ())) != ARMS
        or tuple(lock.get("seeds", ())) != sf.SEEDS
        or lock.get("counts") != sf.COUNTS
        or lock.get("samp_ids") != sf.SAMP_IDS
        or lock.get("epochs") != sf.EPOCHS
        or lock.get("planned_updates") != sf.PLANNED_UPDATES
        or lock.get("early_stopping") != sf._MATRIX["early_stopping"]
        or lock.get("arm_contracts") != sf._arm_spec_payload()
        or lock.get("status") != "implementation_locked_p2_authorized"
        or lock.get("source_repository_root") != str(sf.SOURCE_ROOT)
        or lock.get("artifact_root") != str(sf.SOURCE_ROOT / sf.OUTPUT_ROOT)
    ):
        raise ValueError("正式训练科学合同漂移")
    sf.verify_identity(root / sf.SPEC_PATH, lock["spec"])
    verify_p2_evidence()
    for seed in sf.SEEDS:
        baseline = OmegaConf.create(lock["baselines"][str(seed)])
        for arm in ARMS:
            output = sf.SOURCE_ROOT / sf.OUTPUT_ROOT / "formal" / arm / f"seed_{seed}"
            template = OmegaConf.create(lock["resolved_templates"][str(seed)][arm])
            sf.validate_config(
                template,
                baseline,
                arm=arm,
                output_root=output,
                device="cuda:0",
            )
    return lock, P2_LOCK_SHA256


def runtime_preflight(device: str) -> dict[str, Any]:
    state = sf.git_state()
    if state["status_porcelain"]:
        raise RuntimeError("P3 正式训练要求干净 Git 工作树")
    resolved = torch.device(device)
    if resolved.type != "cuda" or resolved.index is None or not torch.cuda.is_available():
        raise RuntimeError("P3 正式训练要求显式可用的 cuda:<index>")
    problems = check_crd_dependencies()
    if problems:
        raise RuntimeError("P3 原生依赖检查失败: " + "; ".join(problems))
    torch.cuda.set_device(resolved)
    properties = torch.cuda.get_device_properties(resolved)
    return {
        "git": state,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "dependencies": crd_dependency_versions(),
        "device": str(resolved),
        "device_name": properties.name,
        "device_total_bytes": int(properties.total_memory),
        "amp_dtype": "bfloat16",
    }


def _verify_runtime_against_p2(runtime: Mapping[str, Any], evidence: Mapping[str, Any]) -> None:
    path = sf.SOURCE_ROOT / evidence["gpu_acceptance"]["path"] / "environment.json"
    accepted = json.loads(path.read_text(encoding="utf-8"))
    for key in ("python", "torch", "cuda_runtime", "cudnn", "dependencies", "device_name", "device_total_bytes", "amp_dtype"):
        if runtime.get(key) != accepted.get(key):
            raise ValueError(f"P3 runtime 与 P2 工程环境不一致: {key}")


def audit_sources(lock: Mapping[str, Any], cfg: DictConfig, output: Path) -> dict[str, pd.DataFrame]:
    sf.write_json(
        output / "access_started.json",
        {
            "purpose": "P3 formal training and validation",
            "splits": ["train", "val"],
            "research_test_evaluation": False,
            "dataset_index": lock["dataset_index"],
            "source_files": list(lock["source_files"]),
        },
    )
    for relative, expected in lock["source_files"].items():
        sf.verify_identity(sf.SOURCE_ROOT / relative, expected)
    index = lock["dataset_index"]
    index_path = Path(index["path"])
    if sf.sha256_file(index_path) != index["sha256"]:
        raise ValueError("P3 dataset index 身份漂移")
    audited = read_research_v2_index(cfg.data.dataset_root, cfg.data.index_csv, cfg)
    rows: dict[str, pd.DataFrame] = {}
    for split, count in sf.COUNTS.items():
        frame = filter_index(
            audited,
            cfg,
            split=split,
            max_windows=None,
            sample_strategy=str(cfg.data[f"{split}_sample_strategy"]),
            sample_seed=int(cfg.data[f"{split}_sample_seed"]),
        )
        expected_hash = lock["cache_lock"]["row_identity"][f"{split}_row_content_sha256"]
        if (
            len(frame) != count
            or frame.dataset_row_id.duplicated().any()
            or set(frame.split.astype(str)) != {split}
            or frame.samp_id.nunique() != sf.SAMP_IDS[split]
            or array_hash(np.sort(frame.dataset_row_id.to_numpy())) != expected_hash
        ):
            raise ValueError(f"P3 {split} row 合同漂移")
        frame.to_csv(output / f"{split}_rows.csv", index=False)
        rows[split] = frame
    if set(rows["train"].samp_id) & set(rows["val"].samp_id):
        raise ValueError("P3 train/validation subject 隔离失败")
    sf.write_json(
        output / "access_receipt.json",
        {
            "counts": sf.COUNTS,
            "samp_ids": sf.SAMP_IDS,
            "row_order_sha256": {
                split: array_hash(frame.dataset_row_id.to_numpy()) for split, frame in rows.items()
            },
            "sample_seeds": {
                split: int(cfg.data[f"{split}_sample_seed"]) for split in sf.COUNTS
            },
            "dataset_index": index,
            "source_files_verified": lock["source_files"],
            "evaluation_splits": ["train", "val"],
            "research_test_evaluation": False,
        },
    )
    return rows


class FormalStructuralFactorialExperiment(sf.StructuralFactorialExperiment):
    def __init__(
        self,
        cfg: DictConfig,
        validation_rows: pd.DataFrame,
        arm: str,
        initialization_path: Path,
    ):
        self.initialization_path = initialization_path
        super().__init__(cfg, validation_rows, arm)

    def _build_model(self):
        model = build_w0_structural_factorial_model(self.cfg)
        identity = state_dict_identity(model.state_dict())
        identity.update(
            {
                "arm": self.arm,
                "seed": int(self.cfg.training.seed),
                "trainable_parameters": ARM_SPECS[self.arm].trainable_parameters,
            }
        )
        sf.write_json(self.initialization_path, identity)
        return model


def _checkpoint_contract(
    checkpoint: Mapping[str, Any],
    *,
    expected_epoch: int,
    history_row: pd.Series,
    cfg: DictConfig,
) -> None:
    finite_tree(checkpoint, label="checkpoint")
    resolved = OmegaConf.to_container(cfg, resolve=True)
    if checkpoint["epoch"] != expected_epoch or checkpoint["config"] != resolved:
        raise ValueError("P3 checkpoint epoch/config identity 漂移")
    for key in history_row.index:
        if key not in checkpoint["metrics"] or not np.isclose(
            float(checkpoint["metrics"][key]),
            float(history_row[key]),
            atol=1e-12,
            rtol=1e-12,
        ):
            raise ValueError(f"P3 checkpoint/history 不一致: {key}")
    extra = checkpoint["extra_state"]
    if (
        extra.get("protocol") != sf.PROTOCOL
        or int(extra.get("update_index", -1)) != expected_epoch * sf.UPDATES_PER_EPOCH
        or int(extra.get("total_updates", -1)) != sf.PLANNED_UPDATES
    ):
        raise ValueError("P3 checkpoint update/protocol 漂移")
    early = extra.get("early_stopping")
    if (
        not isinstance(early, Mapping)
        or early.get("enabled") is not True
        or early.get("monitor") != "validation_local_rr_mae_full_split"
        or int(early.get("min_epoch", -1)) != sf.EARLY_STOP_MIN_EPOCH
        or int(early.get("patience", -1)) != sf.EARLY_STOP_PATIENCE
        or float(early.get("min_delta", np.nan)) != sf.EARLY_STOP_MIN_DELTA
        or int(early.get("planned_epochs", -1)) != sf.EPOCHS
        or int(early.get("epochs_without_improvement", -1)) != int(history_row.early_stopping_wait)
        or bool(early.get("triggered")) != bool(history_row.early_stopping_triggered)
    ):
        raise ValueError("P3 checkpoint early-stopping 合同漂移")


def validate_anchor_rows(metrics: pd.DataFrame, lock: Mapping[str, Any], seed: int) -> None:
    entry = next(item for item in lock["w0_entries"] if int(item["seed"]) == int(seed))
    relative = str(Path(entry["run_dir"]) / "metrics.csv")
    path = sf.SOURCE_ROOT / relative
    sf.verify_identity(path, lock["source_files"][relative])
    anchor = pd.read_csv(path).sort_values("dataset_row_id").reset_index(drop=True)
    current = metrics.sort_values("dataset_row_id").reset_index(drop=True)
    for key in (
        "dataset_row_id",
        "samp_id",
        "split",
        "whole_rr_target_eligible",
        "local_rr_target_eligible",
        "joint_target_eligible",
    ):
        if not np.array_equal(anchor[key].to_numpy(), current[key].to_numpy()):
            raise ValueError(f"P3 与 W0 validation anchor 不一致: {key}")


def validate_formal_run(
    run_dir: Path,
    cfg: DictConfig,
    rows: pd.DataFrame,
    *,
    arm: str,
    initialization_path: Path,
) -> dict[str, Any]:
    saved = OmegaConf.load(run_dir / "config.yaml")
    if OmegaConf.to_container(saved, resolve=True) != OmegaConf.to_container(cfg, resolve=True):
        raise ValueError("P3 保存配置漂移")
    history = pd.read_csv(run_dir / "train_history.csv")
    best_epoch = sf.validate_history(history, cfg)
    final_epoch = int(history.iloc[-1].epoch)
    if not np.isfinite(history[["train_elapsed_seconds", "train_samples_per_second"]].to_numpy()).all():
        raise FloatingPointError("P3 history runtime 非有限")

    model = build_w0_structural_factorial_model(cfg)
    expected_initialization = state_dict_identity(model.state_dict())
    saved_initialization = json.loads(initialization_path.read_text(encoding="utf-8"))
    if saved_initialization.get("state_sha256") != expected_initialization["state_sha256"]:
        raise ValueError("P3 initialization identity 漂移")
    optimizer, partition = build_crd_optimizer(model, cfg)
    groups = json.loads((run_dir / "optimizer_parameter_groups.json").read_text(encoding="utf-8"))
    if groups != {
        "weight_decay": float(cfg.training.weight_decay),
        "decay": list(partition.decay_names),
        "no_decay": list(partition.no_decay_names),
    }:
        raise ValueError("P3 optimizer 参数分组漂移")

    for filename, epoch in (
        ("checkpoint_best_local_rr.pt", best_epoch),
        ("checkpoint_final.pt", final_epoch),
    ):
        checkpoint = torch.load(run_dir / filename, map_location="cpu", weights_only=False)
        history_row = history.loc[history.epoch.eq(epoch)].iloc[0]
        _checkpoint_contract(
            checkpoint,
            expected_epoch=epoch,
            history_row=history_row,
            cfg=cfg,
        )
        if filename == "checkpoint_final.pt" and int(
            checkpoint["extra_state"]["early_stopping"].get("completed_epochs", -1)
        ) != final_epoch:
            raise ValueError("P3 final checkpoint completed_epochs 漂移")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                state = optimizer.state.get(parameter, {})
                if (
                    not {"step", "exp_avg", "exp_avg_sq"}.issubset(state)
                    or float(state["step"]) != epoch * sf.UPDATES_PER_EPOCH
                    or state["exp_avg"].shape != parameter.shape
                    or state["exp_avg_sq"].shape != parameter.shape
                ):
                    raise ValueError("P3 optimizer state/step 不完整")

    metrics = pd.read_csv(run_dir / "metrics.csv")
    degeneracy = sf.validate_metrics(metrics, rows)
    if not metrics.arm.eq(arm).all() or not metrics.seed.eq(int(cfg.training.seed)).all():
        raise ValueError("P3 metrics arm/seed identity 漂移")
    summary = pd.read_csv(run_dir / "metrics_summary.csv")
    expected_summary = summarize_task_metrics(metrics)
    if len(summary) != 1:
        raise ValueError("P3 metrics summary 必须恰有一行")
    for metric in sf.PRIMARY:
        if (
            not np.isfinite(summary.iloc[0][metric + "_mean"])
            or not np.isclose(
                summary.iloc[0][metric + "_mean"],
                expected_summary.iloc[0][metric + "_mean"],
                atol=1e-12,
                rtol=0,
            )
            or int(summary.iloc[0][metric + "_n"])
            != int(expected_summary.iloc[0][metric + "_n"])
        ):
            raise ValueError(f"P3 summary 数值/分母漂移: {metric}")
    runtime = json.loads((run_dir / "runtime_summary.json").read_text(encoding="utf-8"))
    if runtime.get("peak_reserved_fraction") is None or not np.isfinite(runtime["peak_reserved_fraction"]):
        raise FloatingPointError("P3 runtime summary 不完整")
    return {
        "protocol": sf.PROTOCOL,
        "arm": arm,
        "seed": int(cfg.training.seed),
        "planned_epochs": sf.EPOCHS,
        "completed_epochs": final_epoch,
        "planned_updates": sf.PLANNED_UPDATES,
        "completed_updates": final_epoch * sf.UPDATES_PER_EPOCH,
        "early_stopping_triggered": bool(
            final_epoch < sf.EPOCHS or history.iloc[-1].early_stopping_triggered
        ),
        "selected_epoch": best_epoch,
        "validation_rows": len(metrics),
        "prediction_degeneracy": degeneracy,
        "validation_row_order_sha256": array_hash(rows.dataset_row_id.to_numpy()),
        "initialization_state_sha256": saved_initialization["state_sha256"],
        "runtime": runtime,
    }


def run_formal(arm: str, seed: int, *, device: str = "cuda:0") -> Path:
    if arm not in ARMS or int(seed) not in sf.SEEDS:
        raise ValueError("P3 arm/seed 不属于冻结矩阵")
    seed = int(seed)
    lock, lock_hash = load_formal_contract()
    evidence = _p2_evidence_payload()
    parent = sf.SOURCE_ROOT / sf.OUTPUT_ROOT / "formal" / arm / f"seed_{seed}"
    with sf.exclusive_attempt(
        parent,
        phase="formal",
        lock_hash=lock_hash,
        arm=arm,
        seed=seed,
        reject_completed=True,
    ) as output:
        runtime = runtime_preflight(device)
        _verify_runtime_against_p2(runtime, evidence)
        sf.write_json(output / "environment.json", runtime)
        evidence_summary = verify_p2_evidence(evidence)
        sf.write_json(
            output / "p2_source.json",
            {"evidence": evidence, "verification": evidence_summary},
        )
        sf.write_json(output / "source_code.json", formal_source_identity())
        sf.write_json(output / "implementation_lock.json", lock)
        baseline = OmegaConf.create(lock["baselines"][str(seed)])
        cfg = sf.derived_config(
            baseline,
            arm=arm,
            output_root=output / "training",
            device=device,
        )
        sf.validate_config(
            cfg,
            baseline,
            arm=arm,
            output_root=output / "training",
            device=device,
        )
        rows = audit_sources(lock, cfg, output)
        initialization_path = output / "initialization.json"
        experiment = FormalStructuralFactorialExperiment(
            cfg,
            rows["val"],
            arm,
            initialization_path,
        )
        torch.cuda.synchronize(device)
        training_started = time.perf_counter()
        run_dir = experiment.train()
        torch.cuda.synchronize(device)
        formal_wall_seconds = time.perf_counter() - training_started
        if formal_wall_seconds <= 0 or not np.isfinite(formal_wall_seconds):
            raise RuntimeError("P3 formal wall time 非法")
        receipt = validate_formal_run(
            run_dir,
            cfg,
            rows["val"],
            arm=arm,
            initialization_path=initialization_path,
        )
        validate_anchor_rows(pd.read_csv(run_dir / "metrics.csv"), lock, seed)
        sf.write_json(
            output / "formal_receipt.json",
            {
                **receipt,
                "implementation_lock_sha256": lock_hash,
                "run_dir": str(run_dir.relative_to(output)),
                "formal_wall_seconds": formal_wall_seconds,
            },
        )
    return output


def verify_formal_attempt(
    path: Path,
    *,
    lock_hash: str,
    arm: str,
    seed: int,
) -> dict[str, Any]:
    path = path.resolve()
    if (path / "lifecycle_failed.json").exists():
        raise ValueError(f"P3 formal attempt 失败: {path}")
    freeze = json.loads((path / "freeze_receipt.json").read_text(encoding="utf-8"))
    sf.verify_identity(path / "manifest.json", freeze["manifest"])
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("protocol") != sf.PROTOCOL
        or manifest.get("phase") != "formal"
        or manifest.get("status") != "completed"
        or manifest.get("implementation_lock_sha256") != lock_hash
        or manifest.get("arm") != arm
        or int(manifest.get("seed", -1)) != int(seed)
    ):
        raise ValueError("P3 formal attempt lifecycle identity 漂移")
    required = {
        "lifecycle_started.json",
        "lifecycle_completed.json",
        "environment.json",
        "p2_source.json",
        "source_code.json",
        "implementation_lock.json",
        "access_started.json",
        "access_receipt.json",
        "train_rows.csv",
        "val_rows.csv",
        "initialization.json",
        "formal_receipt.json",
    }
    if not required.issubset(manifest["files"]):
        raise ValueError("P3 formal attempt 缺少必需产物")
    for relative, expected in manifest["files"].items():
        target = (path / relative).resolve()
        if not target.is_relative_to(path):
            raise ValueError("P3 formal manifest 路径越界")
        sf.verify_identity(target, expected)
    receipt = json.loads((path / "formal_receipt.json").read_text(encoding="utf-8"))
    if (
        receipt.get("protocol") != sf.PROTOCOL
        or receipt.get("implementation_lock_sha256") != lock_hash
        or receipt.get("arm") != arm
        or int(receipt.get("seed", -1)) != int(seed)
        or int(receipt.get("validation_rows", -1)) != sf.COUNTS["val"]
    ):
        raise ValueError("P3 formal receipt identity 漂移")
    run_dir = (path / receipt["run_dir"]).resolve()
    if not run_dir.is_relative_to(path) or not run_dir.is_dir():
        raise ValueError("P3 formal training run_dir 越界或缺失")
    return manifest


def formal_plan() -> list[dict[str, Any]]:
    return [
        {
            "arm": arm,
            "seed": seed,
            "output_parent": str(
                sf.SOURCE_ROOT / sf.OUTPUT_ROOT / "formal" / arm / f"seed_{seed}"
            ),
        }
        for arm in ARMS
        for seed in sf.SEEDS
    ]


def matrix_status(lock_hash: str | None = None) -> dict[str, Any]:
    if lock_hash is None:
        _lock, lock_hash = load_formal_contract()
    rows: list[dict[str, Any]] = []
    for cell in formal_plan():
        parent = Path(cell["output_parent"])
        completed: list[str] = []
        failed: list[str] = []
        running: list[str] = []
        if parent.is_dir():
            for attempt in sorted(path for path in parent.iterdir() if path.is_dir()):
                started = attempt / "lifecycle_started.json"
                if not started.is_file():
                    continue
                context = json.loads(started.read_text(encoding="utf-8"))
                if context.get("implementation_lock_sha256") != lock_hash:
                    continue
                if (attempt / "freeze_receipt.json").is_file():
                    verify_formal_attempt(
                        attempt,
                        lock_hash=lock_hash,
                        arm=cell["arm"],
                        seed=cell["seed"],
                    )
                    completed.append(str(attempt))
                elif (attempt / "lifecycle_failed.json").is_file():
                    failed.append(str(attempt))
                else:
                    running.append(str(attempt))
        if len(completed) > 1:
            raise RuntimeError(f"P3 cell 存在多个完成 attempt: {cell['arm']}/{cell['seed']}")
        status = "completed" if completed else ("running" if running else ("failed" if failed else "pending"))
        rows.append({**cell, "status": status, "completed": completed, "failed": failed, "running": running})
    counts = {status: sum(row["status"] == status for row in rows) for status in ("pending", "running", "failed", "completed")}
    return {
        "protocol": sf.PROTOCOL,
        "implementation_lock_sha256": lock_hash,
        "counts": counts,
        "cells": rows,
    }


__all__ = [
    "FORMAL_SOURCE_PATH",
    "P2_ACCEPTANCE",
    "P2_BENCHMARK",
    "P2_LOCK_PATH",
    "P2_LOCK_SHA256",
    "FormalStructuralFactorialExperiment",
    "array_hash",
    "audit_sources",
    "formal_plan",
    "load_formal_contract",
    "matrix_status",
    "run_formal",
    "state_dict_identity",
    "validate_anchor_rows",
    "validate_formal_run",
    "verify_formal_attempt",
    "verify_p2_evidence",
]
