from __future__ import annotations

import gc
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from resp_train.crd.config import CRD_TF_P6A_PROTOCOL_VERSION, load_crd_config
from resp_train.crd.model import CRDCoarseModel, build_crd_model
from resp_train.crd.tf_v1_model import (
    P6_GATE_PARAMETER_COUNT,
    TF_BRANCH_CHECKPOINT_BATCH_CHUNK,
    TF_P6_VARIANT_REPRESENTATIONS,
    TF_P6_VARIANTS,
    trainable_parameter_count,
)
from resp_train.crd.tf_v1_p3 import (
    _branch_projection_gradients_nonzero,
    _new_encoder_dropout_is_zero,
    _synthetic_batch,
    require_clean_git,
)
from resp_train.losses.task import RespirationTaskLoss
from resp_train.utils.run import resolve_device, set_seed


P6A_SYNTHETIC_SEED = 20260811
P6A_ACCEPTANCE_VARIANTS = ("crd_tf402_mws_gate", "crd_tf403_ctrl_gate")
P6A_EXPECTED_INCREMENTAL_PARAMETERS = {
    "crd_tf401_mws_add": 450_080,
    "crd_tf402_mws_gate": 463_427,
    "crd_tf403_ctrl_gate": 463_203,
}


def run_p6a_cuda_synthetic(
    *, config_path: str | Path, device_name: str, output_root: str | Path
) -> Path:
    commit = require_clean_git()
    device = resolve_device(device_name)
    if device.type != "cuda":
        raise ValueError("CRD-TF P6a synthetic 必须使用 CUDA")
    torch.cuda.set_device(device)
    output_dir = _new_receipt_dir(Path(output_root), commit)
    receipt_path = output_dir / "synthetic_receipt.json"
    results: list[dict[str, Any]] = []
    status = "passed"
    error: str | None = None
    try:
        for variant in TF_P6_VARIANTS:
            results.append(_run_one_synthetic(config_path, variant, device))
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        payload = {
            "protocol": CRD_TF_P6A_PROTOCOL_VERSION,
            "phase": "p6a_cuda_synthetic",
            "status": status,
            "complete": status == "passed" and len(results) == len(TF_P6_VARIANTS),
            "git_commit": commit,
            "git_dirty": False,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "seed": P6A_SYNTHETIC_SEED,
            "tf_branch_checkpoint_batch_chunk": TF_BRANCH_CHECKPOINT_BATCH_CHUNK,
            "expected_variants": list(TF_P6_VARIANTS),
            "results": results,
            "error": error,
        }
        receipt_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(receipt_path.resolve())
    return receipt_path


def _run_one_synthetic(
    config_path: str | Path, variant: str, device: torch.device
) -> dict[str, Any]:
    representations = TF_P6_VARIANT_REPRESENTATIONS[variant]
    cfg = load_crd_config(
        config_path,
        overrides=(
            f"model.variant={variant}",
            f"model.tf_representations=[{','.join(representations)}]",
        ),
    )
    set_seed(P6A_SYNTHETIC_SEED)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = build_crd_model(cfg).to(device).train()
    if not _new_encoder_dropout_is_zero(model):
        raise RuntimeError(f"{variant} 新增 condition encoder dropout 不为 0")
    gate_initially_identity = _gate_initially_identity(model, device)
    if variant.endswith("_gate") and not gate_initially_identity:
        raise RuntimeError(f"{variant} gate 初始 factor 不严格为 1")
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
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    all_finite = bool(
        torch.isfinite(output["waveform"]).all()
        and sensor.grad is not None
        and torch.isfinite(sensor.grad).all()
        and gradients
        and all(gradient is not None and bool(torch.isfinite(gradient).all()) for gradient in gradients)
    )
    projection_gradients_nonzero = _branch_projection_gradients_nonzero(model)
    base_count = trainable_parameter_count(
        CRDCoarseModel("crd_c201_decoder_10hz_cap", P6A_SYNTHETIC_SEED)
    )
    incremental = trainable_parameter_count(model) - base_count
    if incremental != P6A_EXPECTED_INCREMENTAL_PARAMETERS[variant]:
        raise RuntimeError(
            f"{variant} incremental parameters={incremental}，期望 {P6A_EXPECTED_INCREMENTAL_PARAMETERS[variant]}"
        )
    if not all_finite or not projection_gradients_nonzero:
        raise FloatingPointError(f"{variant} synthetic output/gradient contract 失败")
    result = {
        "variant": variant,
        "representations": list(representations),
        "status": "passed",
        "waveform_shape": list(output["waveform"].shape),
        "loss": float(loss.detach().cpu()),
        "all_output_input_parameter_gradients_finite": all_finite,
        "all_branch_final_projection_gradients_nonzero": projection_gradients_nonzero,
        "all_new_encoder_dropout_zero": True,
        "gate_initially_identity": gate_initially_identity,
        "gate_parameters": P6_GATE_PARAMETER_COUNT if model.fusion_gate is not None else 0,
        "trainable_parameters": trainable_parameter_count(model),
        "incremental_parameters_vs_c201": incremental,
        "elapsed_seconds": elapsed,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024**2),
    }
    del model, loss_fn, target, sensor, features, output, loss, components, gradients
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _gate_initially_identity(model: torch.nn.Module, device: torch.device) -> bool:
    if model.fusion_gate is None:
        return True
    with torch.no_grad():
        factors = model.fusion_gate(torch.randn(1, 96, 1800, device=device))
    return bool(torch.equal(factors, torch.ones_like(factors)))


def p6a_acceptance_overrides(
    variant: str, *, device: str, output_root: str | Path
) -> tuple[str, ...]:
    if variant not in P6A_ACCEPTANCE_VARIANTS:
        raise ValueError(f"P6a acceptance 只允许最大 representation/control gate: {list(P6A_ACCEPTANCE_VARIANTS)}")
    representations = TF_P6_VARIANT_REPRESENTATIONS[variant]
    return (
        "protocol.run_role=acceptance",
        "protocol.execution_gate=p6a_cuda_acceptance",
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


def _new_receipt_dir(root: Path, commit: str) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    output = root.resolve() / f"{commit[:12]}_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    return output

