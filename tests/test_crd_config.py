from __future__ import annotations

from pathlib import Path

import pytest

from resp_train.crd.config import (
    CRD_DIAGNOSTIC_PROTOCOL_VERSION,
    CRD_DIAGNOSTIC_VARIANTS,
    CRD_EXPLORATORY_PROTOCOL_VERSION,
    CRD_EXPLORATORY_VARIANTS,
    CRD_PROTOCOL_VERSION,
    CRD_S1F_PROTOCOL_VERSION,
    CRD_S1F_VARIANTS,
    CRD_S2A_PROTOCOL_VERSION,
    CRD_S2A_VARIANTS,
    CRD_S2BR_PROTOCOL_VERSION,
    CRD_S2BR_VARIANTS,
    CRD_CONTROLS_PROTOCOL_VERSION,
    load_crd_config,
)
from resp_train.crd.model import CRD_C1_VARIANTS, CRD_C2_VARIANTS


def test_all_crd_configs_are_formal_and_frozen() -> None:
    paths = sorted(Path("configs/crd_v1").glob("*.yaml"))
    assert len(paths) == 18
    for path in paths:
        cfg = load_crd_config(path)
        if cfg.model.variant in CRD_C1_VARIANTS | CRD_C2_VARIANTS:
            expected_protocol = CRD_CONTROLS_PROTOCOL_VERSION
        elif cfg.model.variant in CRD_DIAGNOSTIC_VARIANTS:
            expected_protocol = CRD_DIAGNOSTIC_PROTOCOL_VERSION
        elif cfg.model.variant in CRD_EXPLORATORY_VARIANTS:
            expected_protocol = CRD_EXPLORATORY_PROTOCOL_VERSION
        elif cfg.model.variant in CRD_S1F_VARIANTS:
            expected_protocol = CRD_S1F_PROTOCOL_VERSION
        elif cfg.model.variant in CRD_S2A_VARIANTS:
            expected_protocol = CRD_S2A_PROTOCOL_VERSION
        elif cfg.model.variant in CRD_S2BR_VARIANTS:
            expected_protocol = CRD_S2BR_PROTOCOL_VERSION
        else:
            expected_protocol = CRD_PROTOCOL_VERSION
        assert cfg.protocol.name == expected_protocol
        assert cfg.protocol.run_role == "formal"
        assert cfg.training.epochs == 80
        assert cfg.training.batch_size == 128
        assert cfg.training.gradient_accumulation_steps == 1
        assert cfg.model.initialization_seed == cfg.training.seed
        if cfg.model.variant in CRD_C1_VARIANTS:
            assert cfg.protocol.stage == "c1"
        elif cfg.model.variant in CRD_C2_VARIANTS:
            assert cfg.protocol.stage == "c2"
        elif cfg.model.variant in CRD_S2A_VARIANTS:
            assert cfg.protocol.stage == "s2"
        elif cfg.model.variant in CRD_S2BR_VARIANTS:
            assert cfg.protocol.stage == "s2br"


def test_formal_config_rejects_unregistered_batch_change() -> None:
    with pytest.raises(ValueError, match="physical batch=128"):
        load_crd_config(
            "configs/crd_v1/crd_101_b0_coarse.yaml",
            overrides=["training.batch_size=16"],
        )


def test_smoke_role_allows_only_bounded_engineering_overrides() -> None:
    cfg = load_crd_config(
        "configs/crd_v1/crd_101_b0_coarse.yaml",
        overrides=[
            "protocol.run_role=smoke",
            "training.epochs=1",
            "training.batch_size=2",
            "training.gradient_accumulation_steps=1",
            "training.device=cpu",
            "data.max_train_windows=2",
            "data.max_val_windows=1",
        ],
    )
    assert cfg.protocol.run_role == "smoke"
    assert cfg.data.max_train_windows == 2


def test_acceptance_role_uses_benchmark_selected_physical_batch() -> None:
    cfg = load_crd_config(
        "configs/crd_v1/crd_103_direct_local_mamba.yaml",
        overrides=[
            "protocol.run_role=acceptance",
            "data.max_train_windows=128",
            "data.max_val_windows=32",
            "training.epochs=1",
            "training.device=cuda:0",
        ],
    )
    assert cfg.training.batch_size == 128
    assert cfg.training.gradient_accumulation_steps == 1


def test_crd_config_rejects_resume_and_hidden_model_knobs() -> None:
    with pytest.raises(ValueError, match="training.resume"):
        load_crd_config(
            "configs/crd_v1/crd_101_b0_coarse.yaml",
            overrides=["training.resume=true"],
        )
    with pytest.raises(ValueError, match="model 配置字段"):
        load_crd_config(
            "configs/crd_v1/crd_101_b0_coarse.yaml",
            overrides=["model.local_blocks=5"],
        )
    with pytest.raises(ValueError, match="training 配置字段"):
        load_crd_config(
            "configs/crd_v1/crd_101_b0_coarse.yaml",
            overrides=["training.learning_rate=0.1"],
        )
