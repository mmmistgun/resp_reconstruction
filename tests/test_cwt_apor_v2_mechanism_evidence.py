"""机制产物分析的synthetic CPU测试，不读取实验数据。"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def analysis():
    path = Path(__file__).resolve().parents[1] / "scripts/analyze_cwt_apor_v2_mechanism_evidence.py"
    spec = importlib.util.spec_from_file_location("mechanism_evidence", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_psd_is_invariant_to_scale_and_dc(analysis):
    value = np.array([[[3., 1., 3.]], [[99., 2., 6.]]])
    original = value.copy()
    np.testing.assert_allclose(analysis.normalized_psd(value)[:, 0], [[0., .25, .75], [0., .25, .75]])
    np.testing.assert_array_equal(value, original)


@pytest.mark.parametrize("value", [np.zeros((1, 4)), [[0., np.nan]], [[0., -1., 2.]]])
def test_invalid_psd_fails(analysis, value):
    with pytest.raises((ValueError, FloatingPointError)):
        analysis.normalized_psd(value)


def test_pairing_rejects_reordering_and_duplicate_identity(analysis):
    analysis.check_ids(pd.DataFrame({"dataset_row_id": [3, 8]}), [3, 8])
    for identities in ([8, 3], [3, 3], [3]):
        with pytest.raises(ValueError):
            analysis.check_ids(pd.DataFrame({"dataset_row_id": identities}), [3, 8])


def test_separated_spectral_peaks_recover_power_ratio(analysis):
    # 两个正弦分离6个FFT bin；功率比为振幅比的平方。
    t = np.arange(18000) / 100
    single = np.sin(2*np.pi*.2*t)
    one = analysis.peak_competition(single)
    two = analysis.peak_competition(single + .8*np.sin(2*np.pi*.5*t))
    assert len(two) == 9 and all(v["defined"] for v in two)
    assert max(v["peak_ratio"] for v in one) < .01
    np.testing.assert_allclose([v["peak_ratio"] for v in two], .64, atol=.01)
    np.testing.assert_allclose([v["peak_gap_bpm"] for v in two], 18, atol=1e-9)


def test_zero_signal_retains_undefined_denominator(analysis):
    records = analysis.peak_competition(np.zeros(18000))
    assert len(records) == 9
    assert all(not record["defined"] and np.isnan(record["peak_ratio"]) for record in records)
    with pytest.raises(FloatingPointError):
        analysis.peak_competition(np.full(18000, np.inf))
