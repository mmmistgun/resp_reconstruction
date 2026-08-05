from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from resp_train.config import load_config
from resp_train.experiments.tho import _validate_checkpoint_config
from resp_train.models import build_model


def test_research_v2_config_is_the_frozen_t2_candidate() -> None:
    cfg = load_config("configs/tho_research_v2.yaml")
    assert cfg.model.name == "time_stft_dual1d"
    assert cfg.model.experiment_variant == "t2_g3c_wide_native"
    assert cfg.model.overlap_window == "hann"
    assert cfg.model.stft_win == 2000
    assert cfg.model.stft_hop == 250
    assert cfg.model.stft_low_hz == 0.05
    assert cfg.model.stft_high_hz == 8.0
    assert cfg.model.stft_encoder_type == "conv2d"
    assert cfg.model.fusion_mode == "native_inject"
    assert cfg.model.stft_inject_position == "pre_mixer"
    assert sum(parameter.numel() for parameter in build_model(cfg).parameters()) == 14192
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


@pytest.mark.parametrize(
    ("config_path", "variant", "parameters"),
    [
        ("configs/tho_research_v2.yaml", "t2_g3c_wide_native", 14192),
        ("configs/tho_research_v2_t3_concat.yaml", "t3_e3a0_wide_concat", 16305),
        ("configs/tho_research_v2_t4_bandenergy.yaml", "t4_g3c_bandenergy_native", 12752),
    ],
)
def test_frozen_time_frequency_candidates_build(config_path: str, variant: str, parameters: int) -> None:
    cfg = load_config(config_path)
    assert cfg.model.experiment_variant == variant
    assert sum(parameter.numel() for parameter in build_model(cfg).parameters()) == parameters


@pytest.mark.parametrize(
    "override",
    [
        "model.base_channels=8",
        "model.stft_win=3000",
        "model.stft_hop=500",
        "model.stft_encoder_type=bandenergy",
        "model.fusion_mode=concat_generic",
        "model.stft_inject_position=post_mixer",
        "model.fb_aux_head=enc1_min_aux",
    ],
)
def test_t2_candidate_rejects_architecture_drift(override: str) -> None:
    with pytest.raises(ValueError, match="t2_g3c_wide_native"):
        load_config("configs/tho_research_v2.yaml", overrides=[override])


@pytest.mark.parametrize(
    "config_path",
    ["configs/tho_research_v2_t3_concat.yaml", "configs/tho_research_v2_t4_bandenergy.yaml"],
)
def test_other_time_frequency_candidates_reject_unfrozen_fields(config_path: str) -> None:
    with pytest.raises(ValueError, match="不接受未冻结模型字段"):
        load_config(config_path, overrides=["model.cross_attention_heads=2"])
