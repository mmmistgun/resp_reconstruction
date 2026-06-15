import pytest
from omegaconf import OmegaConf
import torch

from resp_train.models import build_model, list_models


def test_unet1d_tiny_is_registered():
    assert "unet1d_tiny" in list_models()


def test_unet1d_tiny_noskip1_is_registered():
    assert "unet1d_tiny_noskip1" in list_models()


@pytest.mark.parametrize(
    "model_name",
    [
        "periodic_unet1d_tiny",
        "patch_mixer1d",
        "dlinear_waveform",
    ],
)
def test_time_series_library_inspired_models_are_registered(model_name):
    assert model_name in list_models()


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


def test_build_unet1d_tiny_noskip1_preserves_waveform_shape():
    cfg = OmegaConf.create(
        {
            "model": {
                "name": "unet1d_tiny_noskip1",
                "in_channels": 1,
                "out_channels": 1,
                "base_channels": 4,
            }
        }
    )
    model = build_model(cfg)

    y = model(torch.randn(2, 1, 18001))

    assert y.shape == (2, 1, 18001)


@pytest.mark.parametrize(
    "model_name",
    [
        "periodic_unet1d_tiny",
        "patch_mixer1d",
        "dlinear_waveform",
    ],
)
def test_build_time_series_library_inspired_models_preserve_waveform_shape(model_name):
    cfg = OmegaConf.create(
        {
            "model": {
                "name": model_name,
                "in_channels": 2,
                "out_channels": 1,
                "base_channels": 4,
                "patch_len": 128,
                "patch_stride": 64,
                "mixer_layers": 2,
                "moving_avg": 101,
            }
        }
    )
    model = build_model(cfg)

    y = model(torch.randn(2, 2, 1025))

    assert y.shape == (2, 1, 1025)


def test_build_model_unknown_name_lists_available_models():
    cfg = OmegaConf.create({"model": {"name": "unknown"}})

    with pytest.raises(KeyError, match="未知模型.*unet1d_tiny"):
        build_model(cfg)
