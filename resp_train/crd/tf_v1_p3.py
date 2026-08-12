from __future__ import annotations

import gc
import hashlib
import json
import math
import subprocess
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch

from resp_train.crd.config import CRD_TF_CACHE_PATH, CRD_TF_PROTOCOL_VERSION, load_crd_config
from resp_train.crd.model import CRDCoarseModel, build_crd_model
from resp_train.crd.tf_v1_data import FROZEN_CACHE_MANIFEST_SHA256, FROZEN_CACHE_TRANSFORM_SHA256
from resp_train.crd.tf_v1_model import (
    TF_BRANCH_CHECKPOINT_BATCH_CHUNK,
    TF_VARIANT_REPRESENTATIONS,
    TF_VARIANTS,
    trainable_parameter_count,
)
from resp_train.losses.task import RespirationTaskLoss
from resp_train.utils.run import resolve_device, set_seed


P3_ACCEPTANCE_VARIANTS = ("crd_tf102_w", "crd_tf204_wl", "crd_tf302_wls")
P3_SYNTHETIC_SEED = 20260811
P3_MAX_RESERVED_FRACTION = 0.80


def require_clean_git() -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if dirty:
        raise RuntimeError("CRD-TF P3 必须从干净 Git commit 运行")
    return commit


def run_cuda_synthetic_matrix(
    *,
    config_path: str | Path,
    device_name: str,
    output_root: str | Path,
) -> Path:
    commit = require_clean_git()
    device = resolve_device(device_name)
    if device.type != "cuda":
        raise ValueError("CRD-TF P3 synthetic 必须使用 CUDA")
    torch.cuda.set_device(device)
    output_dir = _new_receipt_dir(Path(output_root), commit)
    receipt_path = output_dir / "synthetic_receipt.json"
    results: list[dict[str, Any]] = []
    status = "passed"
    error: str | None = None
    try:
        for variant in TF_VARIANTS:
            results.append(_run_one_synthetic(config_path, variant, device))
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        payload = {
            "protocol": CRD_TF_PROTOCOL_VERSION,
            "phase": "p3_cuda_synthetic",
            "status": status,
            "complete": status == "passed" and len(results) == len(TF_VARIANTS),
            "git_commit": commit,
            "git_dirty": False,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "seed": P3_SYNTHETIC_SEED,
            "tf_branch_checkpoint_batch_chunk": TF_BRANCH_CHECKPOINT_BATCH_CHUNK,
            "cache_transform_sha256": FROZEN_CACHE_TRANSFORM_SHA256,
            "cache_manifest_sha256": FROZEN_CACHE_MANIFEST_SHA256,
            "expected_variants": list(TF_VARIANTS),
            "results": results,
            "error": error,
        }
        receipt_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(receipt_path.resolve())
    return receipt_path


def _run_one_synthetic(config_path: str | Path, variant: str, device: torch.device) -> dict[str, Any]:
    representations = TF_VARIANT_REPRESENTATIONS[variant]
    cfg = load_crd_config(
        config_path,
        overrides=(
            f"model.variant={variant}",
            f"model.tf_representations=[{','.join(representations)}]",
        ),
    )
    set_seed(P3_SYNTHETIC_SEED)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = build_crd_model(cfg).to(device).train()
    new_encoder_dropout_zero = _new_encoder_dropout_is_zero(model)
    if not new_encoder_dropout_zero:
        raise RuntimeError(f"{variant} 新增 TF encoder dropout 不为 0")
    loss_fn = RespirationTaskLoss(cfg).to(device)
    target, sensor, features = _synthetic_batch(device, representations)
    started = time.perf_counter()
    with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=True):
        output = model(sensor, tf=features or None)
    with torch.amp.autocast(device.type, enabled=False):
        components = loss_fn.differentiable_component_sums(output, target.float())
        sync = components["loss_sync_sum"] / components["loss_sync_count"].clamp_min(1)
        effort = components["loss_effort_sum"] / components["loss_effort_count"].clamp_min(1)
        loss = loss_fn.sync_weight * sync + loss_fn.effort_weight * effort
    loss.backward()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    waveform = output["waveform"]
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    all_finite = bool(
        torch.isfinite(waveform).all()
        and sensor.grad is not None
        and torch.isfinite(sensor.grad).all()
        and gradients
        and all(gradient is not None and bool(torch.isfinite(gradient).all()) for gradient in gradients)
    )
    branch_projection_nonzero = _branch_projection_gradients_nonzero(model)
    if not all_finite or not branch_projection_nonzero:
        raise FloatingPointError(f"{variant} synthetic output/gradient contract 失败")
    total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**2)
    result = {
        "variant": variant,
        "representations": list(representations),
        "status": "passed",
        "waveform_shape": list(waveform.shape),
        "loss": float(loss.detach().cpu()),
        "sync_eligible": int(components["loss_sync_count"].item()),
        "effort_eligible": int(components["loss_effort_count"].item()),
        "all_output_input_parameter_gradients_finite": all_finite,
        "all_branch_final_projection_gradients_nonzero": branch_projection_nonzero,
        "all_new_encoder_dropout_zero": new_encoder_dropout_zero,
        "trainable_parameters": trainable_parameter_count(model),
        "incremental_parameters_vs_c201": trainable_parameter_count(model) - _c201_parameter_count(),
        "elapsed_seconds": elapsed,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024**2),
        "peak_reserved_fraction": torch.cuda.max_memory_reserved(device) / (1024**2) / total_memory,
    }
    del model, loss_fn, target, sensor, features, output, loss, components, gradients
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _synthetic_batch(
    device: torch.device, representations: tuple[str, ...]
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    sample_time = torch.arange(18000, device=device, dtype=torch.float32) / 100.0
    envelope = 1.0 + 0.3 * torch.sin(2.0 * torch.pi * 0.01 * sample_time)
    target = (envelope * torch.sin(2.0 * torch.pi * 0.2 * sample_time))[None, None, :]
    sensor = (target + 0.1 * torch.randn_like(target)).requires_grad_(True)
    features: dict[str, torch.Tensor] = {}
    if "m" in representations:
        features["m_slow"] = torch.randn(1, 36, 101, device=device)
        features["m_fast"] = torch.randn(1, 44, 349, device=device)
    if "w" in representations:
        features["w"] = torch.randn(1, 97, 360, device=device)
    if "l" in representations:
        features["l_spectrum"] = torch.randn(1, 9001, device=device, dtype=torch.complex64)
    if "s" in representations:
        features["s"] = torch.randn(1, 12, 360, device=device)
    return target, sensor, features


def _branch_projection_gradients_nonzero(model: torch.nn.Module) -> bool:
    branches = list(model.branches.values()) + list(model.controls)
    return bool(
        branches
        and all(
            branch.final_projection.weight.grad is not None
            and torch.isfinite(branch.final_projection.weight.grad).all()
            and torch.count_nonzero(branch.final_projection.weight.grad) > 0
            for branch in branches
        )
    )


def _new_encoder_dropout_is_zero(model: torch.nn.Module) -> bool:
    branches = list(model.branches.values()) + list(model.controls)
    dropout = [
        module
        for branch in branches
        for module in branch.modules()
        if isinstance(module, torch.nn.Dropout)
    ]
    return bool(dropout and all(module.p == 0.0 for module in dropout))


@lru_cache(maxsize=1)
def _c201_parameter_count() -> int:
    # initialization seed 不改变参数数量；矩阵内只构造一次 CPU reference。
    model = CRDCoarseModel("crd_c201_decoder_10hz_cap", P3_SYNTHETIC_SEED)
    count = trainable_parameter_count(model)
    del model
    return count


def acceptance_overrides(variant: str, *, device: str, output_root: str | Path) -> tuple[str, ...]:
    if variant not in P3_ACCEPTANCE_VARIANTS:
        raise ValueError(f"P3 acceptance 只允许最大 single/pair/triple: {list(P3_ACCEPTANCE_VARIANTS)}")
    representations = TF_VARIANT_REPRESENTATIONS[variant]
    return (
        "protocol.run_role=acceptance",
        "protocol.execution_gate=p3_cuda_acceptance",
        f"model.variant={variant}",
        f"model.tf_representations=[{','.join(representations)}]",
        "data.max_train_windows=128",
        "data.max_val_windows=32",
        "training.epochs=1",
        "training.batch_size=128",
        "training.gradient_accumulation_steps=1",
        f"training.device={device}",
        f"outputs.run_root={Path(output_root).resolve() / variant}",
    )


def audit_p3_acceptance(
    *,
    synthetic_receipt: str | Path,
    run_dirs: Mapping[str, str | Path],
    output_root: str | Path,
) -> Path:
    synthetic_path = Path(synthetic_receipt).resolve()
    synthetic = json.loads(synthetic_path.read_text(encoding="utf-8"))
    if (
        synthetic.get("protocol") != CRD_TF_PROTOCOL_VERSION
        or synthetic.get("status") != "passed"
        or synthetic.get("complete") is not True
        or synthetic.get("git_dirty") is not False
        or synthetic.get("tf_branch_checkpoint_batch_chunk") != TF_BRANCH_CHECKPOINT_BATCH_CHUNK
        or [row.get("variant") for row in synthetic.get("results", [])] != list(TF_VARIANTS)
        or not all(row.get("status") == "passed" for row in synthetic.get("results", []))
        or not all(row.get("all_new_encoder_dropout_zero") is True for row in synthetic.get("results", []))
    ):
        raise RuntimeError("CRD-TF P3 synthetic receipt 不合格")
    expected_commit = str(synthetic["git_commit"])
    if set(run_dirs) != set(P3_ACCEPTANCE_VARIANTS):
        raise ValueError(f"acceptance run_dirs 必须严格包含 {list(P3_ACCEPTANCE_VARIANTS)}")
    audits = [
        _audit_acceptance_run(variant, Path(run_dirs[variant]).resolve(), expected_commit)
        for variant in P3_ACCEPTANCE_VARIANTS
    ]
    identity = {
        "synthetic_receipt_sha256": _sha256_file(synthetic_path),
        "runs": {row["variant"]: row["run_dir"] for row in audits},
    }
    audit_id = _sha256_json(identity)
    output_dir = Path(output_root).resolve() / audit_id
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "p3_acceptance.json"
    payload = {
        "protocol": CRD_TF_PROTOCOL_VERSION,
        "phase": "p3_acceptance_audit",
        "status": "passed",
        "complete": True,
        "git_commit": expected_commit,
        "git_dirty": False,
        "research_test_used": False,
        "cache_transform_sha256": FROZEN_CACHE_TRANSFORM_SHA256,
        "cache_manifest_sha256": FROZEN_CACHE_MANIFEST_SHA256,
        "synthetic_receipt": str(synthetic_path),
        "synthetic_receipt_sha256": identity["synthetic_receipt_sha256"],
        "acceptance_runs": audits,
        "batch_decision": "128x1",
        "maximum_peak_reserved_fraction": max(row["peak_reserved_fraction"] for row in audits),
        "minimum_train_samples_per_second": min(row["train_samples_per_second"] for row in audits),
        "audit_id": audit_id,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def _audit_acceptance_run(variant: str, run_dir: Path, expected_commit: str) -> dict[str, Any]:
    required = (
        "audit.csv",
        "checkpoint_best_local_rr.pt",
        "checkpoint_final.pt",
        "config.yaml",
        "metrics.csv",
        "metrics_summary.csv",
        "optimizer_parameter_groups.json",
        "run_manifest.json",
        "runtime_summary.json",
        "train_history.csv",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"P3 acceptance 产物不完整 {variant}: {missing}")
    cfg = load_crd_config(run_dir / "config.yaml")
    if (
        str(cfg.model.variant) != variant
        or str(cfg.protocol.execution_gate) != "p3_cuda_acceptance"
        or str(cfg.protocol.run_role) != "acceptance"
        or str(cfg.training.device).startswith("cuda:") is False
        or (int(cfg.training.epochs), int(cfg.training.batch_size), int(cfg.training.gradient_accumulation_steps))
        != (1, 128, 1)
        or (int(cfg.data.max_train_windows), int(cfg.data.max_val_windows), cfg.data.max_test_windows)
        != (128, 32, None)
        or str(cfg.data.tf_cache_path) != CRD_TF_CACHE_PATH
    ):
        raise RuntimeError(f"P3 acceptance resolved config identity 不一致: {run_dir}")
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("git_commit") != expected_commit
        or manifest.get("git_dirty") is not False
        or manifest.get("protocol") != CRD_TF_PROTOCOL_VERSION
        or manifest.get("stage") != "tf"
        or manifest.get("run_role") != "acceptance"
    ):
        raise RuntimeError(f"P3 acceptance manifest identity 不一致: {run_dir}")
    history = pd.read_csv(run_dir / "train_history.csv")
    numeric = history.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    if (
        len(history) != 1
        or int(history.iloc[0]["epoch"]) != 1
        or int(history.iloc[0]["optimizer_update"]) != 1
        or not np.isfinite(numeric).all()
        or float(history.iloc[0]["train_elapsed_seconds"]) <= 0.0
        or float(history.iloc[0]["train_samples_per_second"]) <= 0.0
    ):
        raise RuntimeError(f"P3 acceptance train history 不合格: {run_dir}")
    for filename in ("checkpoint_best_local_rr.pt", "checkpoint_final.pt"):
        checkpoint = torch.load(run_dir / filename, map_location="cpu")
        tensors = list(_iter_tensors(checkpoint.get("model_state_dict"))) + list(
            _iter_tensors(checkpoint.get("optimizer_state_dict"))
        )
        extra = checkpoint.get("extra_state", {})
        if (
            int(checkpoint.get("epoch", -1)) != 1
            or int(extra.get("update_index", -1)) != 1
            or int(extra.get("total_updates", -1)) != 1
            or not tensors
            or not all(bool(torch.isfinite(tensor).all()) for tensor in tensors)
        ):
            raise RuntimeError(f"P3 acceptance checkpoint 不合格: {run_dir / filename}")
    metrics = pd.read_csv(run_dir / "metrics_summary.csv").iloc[0]
    primary = (
        "whole_rr_abs_error_bpm_mean",
        "local_rr_mae_bpm_mean",
        "envelope_trajectory_mae_mean",
        "global_envelope_modulation_error_mean",
        "lag_aware_signed_pcc_mean",
    )
    if (
        int(metrics["n_samples"]) != 32
        or not np.isfinite([float(metrics[name]) for name in primary]).all()
        or float(metrics["joint_prediction_degenerate_fraction"]) != 0.0
    ):
        raise RuntimeError(f"P3 acceptance validation metrics 不合格: {run_dir}")
    runtime = json.loads((run_dir / "runtime_summary.json").read_text(encoding="utf-8"))
    reserved_fraction = float(runtime.get("peak_reserved_fraction", math.nan))
    if not (0.0 < reserved_fraction <= P3_MAX_RESERVED_FRACTION):
        raise RuntimeError(f"P3 acceptance peak reserved fraction 越界: {variant}={reserved_fraction}")
    return {
        "variant": variant,
        "run_dir": str(run_dir),
        "training_commit": expected_commit,
        "physical_batch_size": 128,
        "gradient_accumulation_steps": 1,
        "optimizer_updates": 1,
        "validation_samples": 32,
        "train_elapsed_seconds": float(history.iloc[0]["train_elapsed_seconds"]),
        "train_samples_per_second": float(history.iloc[0]["train_samples_per_second"]),
        "peak_allocated_mib": float(runtime["peak_allocated_mib"]),
        "peak_reserved_mib": float(runtime["peak_reserved_mib"]),
        "peak_reserved_fraction": reserved_fraction,
        "checkpoint_best_sha256": _sha256_file(run_dir / "checkpoint_best_local_rr.pt"),
        "checkpoint_final_sha256": _sha256_file(run_dir / "checkpoint_final.pt"),
    }


def _iter_tensors(value: Any) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for nested in value.values():
            yield from _iter_tensors(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _iter_tensors(nested)


def _new_receipt_dir(root: Path, commit: str) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    output = root.resolve() / f"{commit[:12]}_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    return output


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
