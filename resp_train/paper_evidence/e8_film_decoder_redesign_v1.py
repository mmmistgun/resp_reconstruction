"""E8 FiLM/decoder 析因实验的配置与 P1 静态合同。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from resp_train.crd.config import load_crd_config
from resp_train.crd.experiment import CRDExperiment
from resp_train.paper_evidence.e8_film_decoder_redesign_v1_model import (
    ARMS,
    ARM_SPECS,
    PROTOCOL,
    build_e8_film_decoder_redesign_model,
    model_contract,
    trainable_parameter_count,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path("/mnt/disk_code/marques/resp_reconstruction")
SPEC_PATH = Path("configs/e8_film_decoder_redesign_v1/experiment.yaml")
PROTOCOL_PATH = Path("docs/experiments/e8_film_decoder_redesign_v1_protocol_20260925.md")
W0_CONFIG_PATH = Path("configs/crd_tf_v1/crd_tf102_w_formal.yaml")
OUTPUT_ROOT = Path("runs/e8_film_decoder_redesign_v1")

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


def _arm_payload() -> dict[str, Any]:
    return {
        arm: {
            "condition_refiner": spec.condition_refiner,
            "decoder": spec.decoder,
            "trainable_parameters": spec.trainable_parameters,
            "factor_covered_macs": spec.factor_covered_macs,
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
        "output_root": str(OUTPUT_ROOT),
        "source": {
            "w0_config": str(W0_CONFIG_PATH),
            "w0_closeout": "docs/experiments/w0_structural_factorial_v1_closeout_20260925.md",
            "e7_protocol": "docs/experiments/e7_scale_encoding_aggregation_factorial_protocol_20260924.md",
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
            "error_relative_tolerance": 0.005,
            "pcc_absolute_tolerance": 0.002,
            "seed_sd_ddof": 1,
        },
    }
    if OmegaConf.to_container(cfg, resolve=True) != expected:
        raise ValueError("E8 experiment spec 科学合同漂移")
    return cfg


def load_w0_baseline(seed: int, root: Path = ROOT) -> DictConfig:
    if int(seed) not in SEEDS:
        raise ValueError("seed 必须属于 E8 三 seed 矩阵")
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
        raise ValueError(f"未知 E8 arm={arm!r}")
    cfg = OmegaConf.create(OmegaConf.to_container(baseline, resolve=True))
    cfg.protocol.name = PROTOCOL
    cfg.protocol.stage = "e8_film_decoder_redesign_v1"
    cfg.protocol.execution_gate = "e8_formal_unopened"
    cfg.model.e8_film_decoder_redesign_v1 = model_contract(arm)
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
        raise ValueError("E8 derived config 与同 seed W0 合同不一致")
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
        raise ValueError("E8 W0 来源配置不符合冻结合同")


def formal_plan() -> list[dict[str, Any]]:
    return [
        {
            "arm": arm,
            "seed": seed,
            "condition_refiner": ARM_SPECS[arm].condition_refiner,
            "decoder": ARM_SPECS[arm].decoder,
            "status": "blocked_until_engineering_acceptance",
        }
        for seed in SEEDS
        for arm in ARMS
    ]


def check_p1() -> dict[str, Any]:
    """只做配置、结构和初始化静态核验；不读取数据或运行训练。"""

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

    counts: dict[str, int] = {}
    baseline = load_w0_baseline(SEEDS[0])
    for arm in ARMS:
        cfg = derived_config(
            baseline,
            arm=arm,
            output_root=SOURCE_ROOT / OUTPUT_ROOT / "formal" / arm / f"seed_{SEEDS[0]}",
            device="cpu",
        )
        model = build_e8_film_decoder_redesign_model(cfg)
        counts[arm] = trainable_parameter_count(model)
        if counts[arm] != ARM_SPECS[arm].trainable_parameters:
            raise RuntimeError(f"E8 参数合同错误: {arm}")
    return {
        "protocol": PROTOCOL,
        "arms": len(ARMS),
        "formal_cells": len(formal_plan()),
        "parameters": counts,
        "data_access": False,
        "formal_gate": "closed",
    }


class E8FilmDecoderExperiment(CRDExperiment):
    """预留给后续受控 formal 入口的原生 trainer 适配器。"""

    def _build_model(self):
        return build_e8_film_decoder_redesign_model(self.cfg)
