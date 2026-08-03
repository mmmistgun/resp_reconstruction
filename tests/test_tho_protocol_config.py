from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from resp_train.config import load_config
from resp_train.experiments.tho import _validate_checkpoint_config
from resp_train.models import build_model


def test_research_v2_config_is_the_frozen_time_only_baseline() -> None:
    cfg = load_config("configs/tho_research_v2.yaml")
    assert cfg.model.name == "patch_mixer1d"
    assert cfg.model.overlap_window == "hann"
    assert cfg.data.max_train_windows is None
    assert cfg.data.max_val_windows is None
    assert cfg.loss.band_low_hz == 0.05
    assert cfg.loss.band_high_hz == 0.70
    assert cfg.loss.max_lag_sec == 0.30
    assert cfg.loss.sync_weight == 1.0
    assert cfg.loss.effort_weight == 0.25
    assert "rhythm_weight" not in cfg.loss
    assert "pol_start_weight" not in cfg.loss
    assert cfg.evaluation.local_rr_window_sec == 60
    assert cfg.evaluation.local_rr_step_sec == 15
    assert cfg.training.lr_scheduler == "none"
    assert cfg.training.grad_clip_norm is None
    assert cfg.training.use_amp is False
    assert "patience" not in cfg.training
    assert "min_delta" not in cfg.training
    assert "baseline" not in cfg


def test_checkpoint_reevaluation_only_allows_runtime_changes() -> None:
    cfg = load_config("configs/tho_research_v2.yaml")
    checkpoint_config = OmegaConf.to_container(cfg, resolve=True)

    runtime_only = OmegaConf.create(checkpoint_config)
    runtime_only.training.device = "cpu"
    _validate_checkpoint_config(checkpoint_config, runtime_only)

    changed_protocol = OmegaConf.create(checkpoint_config)
    changed_protocol.data.test_split = "another_test"
    with pytest.raises(ValueError, match="data"):
        _validate_checkpoint_config(checkpoint_config, changed_protocol)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("loss.band_low_hz=0.10", "0.05–0.70"),
        ("loss.sync_weight=0.5", "sync_weight=1.0"),
        ("loss.effort_weight=0.5", "effort_weight=0.25"),
        ("training.lr_scheduler=cosine", "不使用 learning-rate scheduler"),
        ("training.use_amp=true", "use_amp=false"),
    ],
)
def test_frozen_protocol_rejects_semantic_drift(override: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_config("configs/tho_research_v2.yaml", overrides=[override])


def test_m1_multiscale_candidate_is_frozen_and_parameter_matched() -> None:
    baseline_cfg = load_config("configs/tho_research_v2.yaml")
    candidate_cfg = load_config(
        "configs/tho_research_v2.yaml",
        overrides=[
            "model.name=multiscale_patch_mixer1d",
            "model.base_channels=1",
            "model.patch_lengths=[256,512,1024,2048]",
            "model.patch_stride_ratio=0.5",
        ],
    )
    baseline_parameters = sum(parameter.numel() for parameter in build_model(baseline_cfg).parameters())
    candidate_parameters = sum(parameter.numel() for parameter in build_model(candidate_cfg).parameters())
    assert candidate_parameters == 11664
    assert abs(candidate_parameters - baseline_parameters) / baseline_parameters < 0.05


@pytest.mark.parametrize(
    "override",
    [
        "model.base_channels=2",
        "model.patch_lengths=[256,512,1024]",
        "model.patch_stride_ratio=0.25",
        "model.mixer_layers=3",
    ],
)
def test_m1_multiscale_candidate_rejects_architecture_drift(override: str) -> None:
    with pytest.raises(ValueError, match="M1 冻结要求"):
        load_config(
            "configs/tho_research_v2.yaml",
            overrides=[
                "model.name=multiscale_patch_mixer1d",
                "model.base_channels=1",
                "model.patch_lengths=[256,512,1024,2048]",
                "model.patch_stride_ratio=0.5",
                override,
            ],
        )
