from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from resp_train.config import load_config
from resp_train.experiments.tho import _validate_checkpoint_config


def test_research_v2_config_is_the_frozen_time_only_baseline() -> None:
    cfg = load_config("configs/tho_research_v2.yaml")
    assert cfg.model.name == "patch_mixer1d"
    assert cfg.model.overlap_window == "hann"
    assert cfg.data.max_train_windows is None
    assert cfg.data.max_val_windows is None
    assert cfg.loss.band_low_hz == 0.05
    assert cfg.loss.band_high_hz == 0.70
    assert cfg.loss.max_lag_sec == 0.30
    assert cfg.loss.local_rhythm_window_sec == 30
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
        ("loss.local_rhythm_window_sec=40", "local_rhythm_window_sec=30"),
        ("training.lr_scheduler=cosine", "不使用 learning-rate scheduler"),
        ("training.use_amp=true", "use_amp=false"),
    ],
)
def test_frozen_protocol_rejects_semantic_drift(override: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_config("configs/tho_research_v2.yaml", overrides=[override])


@pytest.mark.parametrize(
    ("weight_override", "weight_key"),
    [
        ("loss.rhythm_weight=0", "rhythm_weight"),
        ("loss.effort_weight=0", "effort_weight"),
        ("loss.pol_start_weight=0", "pol_start_weight"),
    ],
)
def test_registered_loss_ablation_variants_are_allowed(
    weight_override: str,
    weight_key: str,
) -> None:
    cfg = load_config(
        "configs/tho_research_v2.yaml",
        overrides=[weight_override],
    )
    assert float(cfg.loss[weight_key]) == 0.0


@pytest.mark.parametrize(
    "overrides",
    [
        ["loss.sync_weight=0"],
        ["loss.rhythm_weight=0.25"],
        ["loss.effort_weight=0.1"],
        ["loss.pol_start_weight=0.01"],
        ["loss.rhythm_weight=0", "loss.effort_weight=0"],
    ],
)
def test_loss_ablation_guard_rejects_unregistered_weight_combinations(overrides: list[str]) -> None:
    with pytest.raises(ValueError, match="不开放任意 loss 权重搜索"):
        load_config("configs/tho_research_v2.yaml", overrides=overrides)
