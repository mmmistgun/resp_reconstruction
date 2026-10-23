"""THO描述性计数的合成信号定向检查。"""
import numpy as np
import pytest

from scripts.count_tho_window_breaths import count_window


@pytest.mark.parametrize('frequency, expected', [(0.1, 18), (0.2, 36), (0.5, 90), (0.65, 117)])
def test_known_breath_counts(frequency, expected):
    signal = np.sin(2 * np.pi * frequency * np.arange(18000) / 100)
    peaks, dynamic, _ = count_window(signal)
    assert len(peaks) == expected
    assert dynamic


def test_constant_has_no_peaks():
    peaks, dynamic, _ = count_window(np.zeros(18000))
    assert len(peaks) == 0
    assert not dynamic


@pytest.mark.parametrize('signal', [np.full(18000, np.nan), np.full(18000, np.inf), np.zeros(17999)])
def test_invalid_window_fails(signal):
    with pytest.raises(ValueError):
        count_window(signal)
