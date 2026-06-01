import pytest
from omegaconf import OmegaConf
import torch

from resp_train.models import build_model, list_models


def test_unet1d_tiny_is_registered():
    assert "unet1d_tiny" in list_models()


def test_build_unet1d_tiny_preserves_waveform_shape():
    cfg = OmegaConf.create(
        {
            "model": {
                "name": "unet1d_tiny",
                "in_channels": 1,
                "out_channels": 1,
                "base_channels": 8,
            }
        }
    )
    model = build_model(cfg)

    x = torch.randn(2, 1, 18000)
    y = model(x)

    assert y.shape == x.shape


def test_build_unet1d_tiny_uses_configured_channels_and_handles_odd_length():
    cfg = OmegaConf.create(
        {
            "model": {
                "name": "unet1d_tiny",
                "in_channels": 2,
                "out_channels": 1,
                "base_channels": 4,
            }
        }
    )
    model = build_model(cfg)

    y = model(torch.randn(2, 2, 18001))

    assert y.shape == (2, 1, 18001)


def test_build_model_unknown_name_lists_available_models():
    cfg = OmegaConf.create({"model": {"name": "unknown"}})

    with pytest.raises(KeyError, match="未知模型.*unet1d_tiny"):
        build_model(cfg)
