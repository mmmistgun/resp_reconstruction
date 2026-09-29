"""E9 潜在宽度 × 条件末端实验的配置、P1 合同与受控计划。"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from resp_train.crd.config import load_crd_config
from resp_train.crd.experiment import CRDExperiment
from resp_train.paper_evidence.e9_latent_width_condition_refiner_v1_model import (
    ARMS,
    ARM_SPECS,
    PROTOCOL,
    build_e9_latent_width_condition_refiner_model,
    model_contract,
    trainable_parameter_count,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path("/mnt/disk_code/marques/resp_reconstruction")
SPEC_PATH = Path("configs/e9_latent_width_condition_refiner_v1/experiment.yaml")
PROTOCOL_PATH = Path(
    "docs/experiments/e9_latent_width_condition_refiner_v1_protocol_20260929.md"
)
W0_CONFIG_PATH = Path("configs/crd_tf_v1/crd_tf102_w_formal.yaml")
OUTPUT_ROOT = Path("runs/e9_latent_width_condition_refiner_v1")
FORMAL_LOCK_PATH = Path(
    "docs/experiments/e9_latent_width_condition_refiner_v1_implementation_lock_20260929.json"
)
SUMMARY_ROOT = OUTPUT_ROOT / "validation_summary"
TEST_ROOT = OUTPUT_ROOT / "research_test"

SEEDS = (20260811, 20260812, 20260813)
EPOCHS = 80
UPDATES_PER_EPOCH = 80
PLANNED_UPDATES = 6_400
EARLY_STOP_MIN_EPOCH = 30
EARLY_STOP_PATIENCE = 15
EARLY_STOP_MIN_DELTA = 0.0

PRIMARY_METRICS = (
    "whole_rr_abs_error_bpm",
    "local_rr_mae_bpm",
    "envelope_trajectory_mae",
    "global_envelope_modulation_error",
    "lag_aware_signed_pcc",
)

PLANNED_CONTRASTS = (
    "e9a_d96_h64-e9a_d96_h65",
    "e9a_d96_h48-e9a_d96_h65",
    "e9a_d96_h48-e9a_d96_h64",
    "e9b_d64_h64-e9b_d64_direct",
    "e9b_d64_h48-e9b_d64_direct",
    "e9b_d64_h48-e9b_d64_h64",
)


def _arm_payload() -> dict[str, Any]:
    return {
        arm: {
            "family": spec.family,
            "latent_channels": spec.latent_channels,
            "condition_refiner": spec.condition_refiner,
            "hidden_channels": spec.hidden_channels,
            "trainable_parameters": spec.trainable_parameters,
            "condition_refiner_macs": spec.condition_refiner_macs,
            "declared_covered_macs": spec.declared_covered_macs,
        }
        for arm, spec in ARM_SPECS.items()
    }


def load_experiment_spec(path: Path | None = None) -> DictConfig:
    path = ROOT / SPEC_PATH if path is None else Path(path)
    cfg = OmegaConf.load(path)
    OmegaConf.resolve(cfg)
    expected = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "p0_p1_engineering_design_formal_blocked",
        "output_roots": {
            "experiment": str(OUTPUT_ROOT),
            "validation_summary": str(SUMMARY_ROOT),
            "research_test": str(TEST_ROOT),
        },
        "implementation_lock": str(FORMAL_LOCK_PATH),
        "source": {
            "w0_config": str(W0_CONFIG_PATH),
            "e8_protocol": "docs/experiments/e8_film_decoder_redesign_v1_protocol_20260925.md",
            "e8_closeout": "docs/experiments/e8_film_decoder_redesign_v1_closeout_20260926.md",
        },
        "arms": _arm_payload(),
        "matrix": {
            "seeds": list(SEEDS),
            "cell_count": len(ARMS) * len(SEEDS),
            "epochs": EPOCHS,
            "updates_per_epoch": UPDATES_PER_EPOCH,
            "planned_updates": PLANNED_UPDATES,
            "early_stopping": {
                "enabled": True,
                "min_epoch": EARLY_STOP_MIN_EPOCH,
                "patience": EARLY_STOP_PATIENCE,
                "min_delta": EARLY_STOP_MIN_DELTA,
            },
            "physical_batch": 128,
            "gradient_accumulation_steps": 1,
            "amp_dtype": "bfloat16",
            "checkpoint_selector": "full_validation_local_rr_strict_minimum_earliest_tie",
            "split": "train_validation",
        },
        "analysis": {
            "primary_metrics": list(PRIMARY_METRICS),
            "planned_contrasts": list(PLANNED_CONTRASTS),
            "error_relative_tolerance": 0.005,
            "pcc_absolute_tolerance": 0.002,
            "seed_sd_ddof": 1,
            "cross_metric_score": None,
        },
    }
    if OmegaConf.to_container(cfg, resolve=True) != expected:
        raise ValueError("E9 experiment spec 科学合同漂移")
    return cfg


def load_w0_baseline(seed: int, root: Path = ROOT) -> DictConfig:
    if int(seed) not in SEEDS:
        raise ValueError("seed 必须属于 E9 三 seed 矩阵")
    return load_crd_config(
        root / W0_CONFIG_PATH,
        overrides=[f"training.seed={int(seed)}", f"model.initialization_seed={int(seed)}"],
    )


def derived_config(
    baseline: DictConfig,
    *,
    arm: str,
    output_root: Path,
    device: str,
) -> DictConfig:
    if arm not in ARM_SPECS:
        raise ValueError(f"未知 E9 arm={arm!r}")
    cfg = OmegaConf.create(OmegaConf.to_container(baseline, resolve=True))
    cfg.protocol.name = PROTOCOL
    cfg.protocol.stage = "e9_latent_width_condition_refiner_v1"
    cfg.protocol.execution_gate = "e9_formal_blocked_until_p2_p3_lock"
    cfg.model.e9_latent_width_condition_refiner_v1 = model_contract(arm)
    cfg.training.early_stopping_enabled = True
    cfg.training.early_stopping_min_epoch = EARLY_STOP_MIN_EPOCH
    cfg.training.early_stopping_patience = EARLY_STOP_PATIENCE
    cfg.training.early_stopping_min_delta = EARLY_STOP_MIN_DELTA
    cfg.training.device = str(device)
    cfg.training.show_progress = False
    cfg.outputs.run_root = str(output_root)
    return cfg


def validate_config(
    cfg: DictConfig,
    baseline: DictConfig,
    *,
    arm: str,
    output_root: Path,
    device: str,
) -> None:
    expected = derived_config(
        baseline,
        arm=arm,
        output_root=output_root,
        device=device,
    )
    if OmegaConf.to_container(cfg, resolve=True) != OmegaConf.to_container(expected, resolve=True):
        raise ValueError("E9 derived config 与同 seed W0 来源合同不一致")
    if (
        str(baseline.protocol.name) != "crd-tf-v1-research-informed-20260812"
        or str(baseline.model.variant) != "crd_tf102_w"
        or list(baseline.model.tf_representations) != ["w"]
        or int(baseline.training.seed) not in SEEDS
        or int(baseline.model.initialization_seed) != int(baseline.training.seed)
        or int(baseline.training.epochs) != EPOCHS
        or bool(baseline.training.early_stopping_enabled)
        or int(baseline.training.batch_size) != 128
        or int(baseline.training.gradient_accumulation_steps) != 1
        or not bool(baseline.training.use_amp)
        or str(baseline.training.amp_dtype) != "bfloat16"
        or float(baseline.loss.sync_weight) != 1.0
        or float(baseline.loss.effort_weight) != 0.25
    ):
        raise ValueError("E9 W0 来源配置不符合冻结合同")


def formal_plan() -> list[dict[str, Any]]:
    return [
        {
            "arm": arm,
            "seed": seed,
            "family": ARM_SPECS[arm].family,
            "latent_channels": ARM_SPECS[arm].latent_channels,
            "condition_refiner": ARM_SPECS[arm].condition_refiner,
            "output_parent": str(
                SOURCE_ROOT / OUTPUT_ROOT / "formal" / arm / f"seed_{seed}"
            ),
            "status": "blocked_until_gpu_acceptance_and_implementation_lock",
            "command": (
                "PYTHONPATH=. ./.venv/bin/python "
                "scripts/run_e9_latent_width_condition_refiner_v1.py formal "
                f"--arm {arm} --seed {seed} --device cuda:0 "
                "--confirm-formal-training"
            ),
        }
        for seed in SEEDS
        for arm in ARMS
    ]


def _serialized_state_dict_bytes(model: torch.nn.Module) -> int:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return len(buffer.getvalue())


def _finite_state_dict(model: torch.nn.Module) -> None:
    for name, value in model.state_dict().items():
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"E9 state_dict 包含非有限 tensor: {name}")


def check_p1() -> dict[str, Any]:
    """只做配置、结构、参数、state round-trip 与 finite；不读取真实数据。"""

    load_experiment_spec()
    for seed in SEEDS:
        baseline = load_w0_baseline(seed)
        for arm in ARMS:
            output = SOURCE_ROOT / OUTPUT_ROOT / "formal" / arm / f"seed_{seed}"
            cfg = derived_config(
                baseline,
                arm=arm,
                output_root=output,
                device="cuda:0",
            )
            validate_config(
                cfg,
                baseline,
                arm=arm,
                output_root=output,
                device="cuda:0",
            )

    models: dict[str, dict[str, int]] = {}
    baseline = load_w0_baseline(SEEDS[0])
    for arm in ARMS:
        cfg = derived_config(
            baseline,
            arm=arm,
            output_root=SOURCE_ROOT / OUTPUT_ROOT / "p1_disposable" / arm,
            device="cpu",
        )
        model = build_e9_latent_width_condition_refiner_model(cfg)
        count = trainable_parameter_count(model)
        if count != ARM_SPECS[arm].trainable_parameters:
            raise RuntimeError(f"E9 参数合同错误: {arm}")
        _finite_state_dict(model)
        clone = build_e9_latent_width_condition_refiner_model(cfg)
        clone.load_state_dict(model.state_dict(), strict=True)
        _finite_state_dict(clone)
        models[arm] = {
            "trainable_parameters": count,
            "serialized_state_dict_bytes": _serialized_state_dict_bytes(model),
            "condition_refiner_macs": ARM_SPECS[arm].condition_refiner_macs,
            "declared_covered_macs": ARM_SPECS[arm].declared_covered_macs,
        }
        del clone, model
    return {
        "protocol": PROTOCOL,
        "arms": len(ARMS),
        "formal_cells": len(formal_plan()),
        "models": models,
        "data_access": False,
        "test_access": False,
        "gpu_used": False,
        "formal_gate": "closed_until_p2_p3",
    }


class E9LatentWidthConditionRefinerExperiment(CRDExperiment):
    """E9 原生 trainer 适配器；formal 仍由 P2/P3 implementation lock 门控。"""

    def _build_model(self):
        return build_e9_latent_width_condition_refiner_model(self.cfg)
