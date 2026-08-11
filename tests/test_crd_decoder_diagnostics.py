from __future__ import annotations

import numpy as np
import pytest
import torch

from resp_train.crd.decoder_diagnostics import residual_spectral_rows


def test_residual_spectral_rows_separates_formal_band_and_out_of_band() -> None:
    time = torch.arange(18000, dtype=torch.float32) / 100.0
    residual = torch.stack(
        (
            torch.sin(2.0 * torch.pi * 0.20 * time),
            torch.sin(2.0 * torch.pi * 2.00 * time),
        )
    )[:, None, :]
    rows = residual_spectral_rows(residual)
    assert rows["residual_in_band_energy_fraction"][0] == pytest.approx(1.0, abs=1e-6)
    assert rows["residual_out_of_band_energy_fraction"][0] == pytest.approx(0.0, abs=1e-6)
    assert rows["residual_in_band_energy_fraction"][1] == pytest.approx(0.0, abs=1e-6)
    assert rows["residual_out_of_band_energy_fraction"][1] == pytest.approx(1.0, abs=1e-6)
    assert np.isfinite(np.concatenate(list(rows.values()))).all()


def test_residual_spectral_rows_rejects_zero_or_nonfinite_residual() -> None:
    with pytest.raises(FloatingPointError, match="energy 为 0"):
        residual_spectral_rows(torch.zeros(1, 1, 18000))
    residual = torch.zeros(1, 1, 18000)
    residual[0, 0, 0] = torch.nan
    with pytest.raises(FloatingPointError, match="NaN/Inf"):
        residual_spectral_rows(residual)

