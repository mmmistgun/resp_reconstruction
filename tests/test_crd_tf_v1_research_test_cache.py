from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from resp_train.crd.config import load_crd_config
from resp_train.crd.tf_v1_features import RidgeParameters
from resp_train.crd.tf_v1_research_test_cache import (
    EXPECTED_TEST_COUNT,
    _validate_config,
    _validate_test_rows,
    _write_test_cache,
)


def test_research_test_cache_config_and_full_row_contract() -> None:
    cfg = load_crd_config("configs/crd_tf_v1/crd_tf203_ms_formal.yaml")
    _validate_config(cfg)
    rows = pd.DataFrame(
        {
            "dataset_row_id": np.arange(EXPECTED_TEST_COUNT),
            "split": ["test"] * EXPECTED_TEST_COUNT,
        }
    )
    _validate_test_rows(rows, "test")
    with pytest.raises(RuntimeError, match="完整窗口数"):
        _validate_test_rows(rows.iloc[:-1], "test")


def test_research_test_cache_writer_reads_input_and_omits_l_spectrum(monkeypatch, tmp_path) -> None:
    cfg = load_crd_config("configs/crd_tf_v1/crd_tf203_ms_formal.yaml")
    rows = pd.DataFrame(
        [
            {
                "dataset_row_id": 7,
                "bcg_signal_key": "bcg",
                "window_start_sample": 0,
                "window_end_sample": 18000,
                "source_npz": "source.npz",
                "samp_id": 3,
            }
        ]
    )

    class _SourceCache:
        def __init__(self, _path):
            pass

        def get_arrays(self, _source, keys):
            assert keys == ["bcg"]
            return {"bcg": np.zeros(18000, dtype=np.float32)}

    monkeypatch.setattr("resp_train.crd.tf_v1_research_test_cache.WholeNightCache", _SourceCache)
    monkeypatch.setattr(
        "resp_train.crd.tf_v1_research_test_cache.multires_stft_features",
        lambda _waveform: {
            "m_slow": np.zeros((36, 101), dtype=np.float32),
            "m_fast": np.zeros((44, 349), dtype=np.float32),
        },
    )
    monkeypatch.setattr(
        "resp_train.crd.tf_v1_research_test_cache.cwt_magnitude_features",
        lambda _waveform: (np.zeros((97, 360), dtype=np.float32), np.arange(97, dtype=np.float64)),
    )
    monkeypatch.setattr(
        "resp_train.crd.tf_v1_research_test_cache.wsst_energy_map",
        lambda _waveform: (np.zeros((387, 360), dtype=np.float32), np.arange(387, dtype=np.float64)),
    )
    monkeypatch.setattr(
        "resp_train.crd.tf_v1_research_test_cache.wsst_ridge_features",
        lambda *_args: np.zeros((12, 360), dtype=np.float32),
    )
    manifest = _write_test_cache(
        rows=rows,
        cfg=cfg,
        index_path=tmp_path / "index.csv",
        output_dir=tmp_path,
        ridge_parameters=RidgeParameters(2.0, 2),
        frequency_identity={},
        show_progress=False,
    )

    assert manifest["count"] == 1
    assert manifest["sample_seed"] == 20260612
    assert (tmp_path / "test_m_slow.npy").is_file()
    assert (tmp_path / "test_w.npy").is_file()
    assert (tmp_path / "test_s.npy").is_file()
    assert not (tmp_path / "test_l_spectrum.npy").exists()

