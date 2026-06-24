from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from resp_train.experiments.tho_e5_a2 import ThoE5A2Experiment
from resp_train.models.registry import build_model


def _cfg(checkpoint: Path | None = None):
    return OmegaConf.create(
        {
            "window": {"target_fs": 100, "duration_samples": 4096},
            "data": {},
            "model": {
                "name": "time_stft_dual1d",
                "in_channels": 1,
                "out_channels": 1,
                "base_channels": 8,
                "branch_mode": "dual",
                "time_backbone": "patch_mixer1d",
                "patch_len": 128,
                "patch_stride": 64,
                "mixer_layers": 1,
                "overlap_window": "hann",
                "stft_win": 512,
                "stft_hop": 128,
                "stft_low_hz": 0.05,
                "stft_high_hz": 8.0,
                "stft_out_channels": 16,
                "stft_norm": "n0",
                "stft_encoder_type": "conv2d",
                "fusion_mode": "cross_attention_inject",
                "stft_inject_position": "pre_mixer",
                "cross_attention_heads": 2,
            },
            "training": {
                "learning_rate": 0.001,
                "time_backbone_learning_rate": 0.0001,
                "warm_start_checkpoint": str(checkpoint) if checkpoint is not None else None,
                "warm_start_prefixes": ["time_backbone."],
            },
        }
    )


def test_e5_a2_build_model_loads_time_backbone_prefix_only(tmp_path):
    source = build_model(_cfg())
    with torch.no_grad():
        source.time_backbone.patch_embed.bias.fill_(3.0)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": source.state_dict()}, checkpoint)

    model = ThoE5A2Experiment(_cfg(checkpoint)).build_model()

    assert torch.allclose(model.time_backbone.patch_embed.bias, torch.full_like(model.time_backbone.patch_embed.bias, 3.0))
    assert torch.allclose(model.cross_attention_adapter.out_proj.weight, torch.zeros_like(model.cross_attention_adapter.out_proj.weight))


def test_e5_a2_optimizer_uses_lower_lr_for_time_backbone():
    model = build_model(_cfg())
    opt = ThoE5A2Experiment(_cfg()).build_optimizer(model)

    lrs = sorted({group["lr"] for group in opt.param_groups})

    assert lrs == [0.0001, 0.001]
    time_ids = {id(param) for param in model.time_backbone.parameters()}
    time_group = next(group for group in opt.param_groups if group["lr"] == 0.0001)
    assert {id(param) for param in time_group["params"]} <= time_ids


def test_e5_a2_warm_start_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ThoE5A2Experiment(_cfg(tmp_path / "missing.pt")).build_model()
