from __future__ import annotations

import pytest
import torch

from resp_train.config import load_config
from resp_train.models import build_model


@pytest.mark.parametrize(
    ("config_path", "band_bins", "energy_bands"),
    [
        ("configs/tho_research_v2.yaml", 160, 0),
        ("configs/tho_research_v2_t3_concat.yaml", 239, 0),
        ("configs/tho_research_v2_t4_bandenergy.yaml", 160, 5),
    ],
)
def test_frozen_time_frequency_candidates_preserve_full_waveform_contract(
    config_path: str,
    band_bins: int,
    energy_bands: int,
) -> None:
    cfg = load_config(config_path)
    model = build_model(cfg).eval()
    x = torch.randn(1, 1, 18000)

    with torch.no_grad():
        prediction = model(x)

    assert prediction.shape == (1, 1, 18000)
    assert torch.isfinite(prediction).all()
    assert model.stft_encoder.band_bin_count() == band_bins
    assert model.stft_encoder.energy_band_count() == energy_bands


def test_t2_native_projection_starts_as_exact_time_only_model() -> None:
    cfg = load_config("configs/tho_research_v2.yaml")
    model = build_model(cfg).eval()
    x = torch.randn(1, 1, 18000)

    with torch.no_grad():
        expected = model.time_backbone(x)
        actual = model(x)

    assert torch.equal(actual, expected)
