"""窗口事件交集与冻结指标汇总的 synthetic CPU 检查。"""
import numpy as np
import pandas as pd
import pytest
from scripts.analyze_h_only_sa_presence import interval_hits, summarize, SEEDS, METRICS


def test_half_open_intersections_and_multiple_events():
    windows = [[0, 180], [180, 360], [360, 540]]
    events = [[-20, 0], [179, 181], [200, 220], [540, 560]]
    hits = interval_hits(windows, events)
    assert hits.tolist() == [[False, True, False, False], [False, True, True, False], [False, False, False, False]]
    assert interval_hits(windows, np.empty((0, 2))).shape == (3, 0)


def test_invalid_bounds_rejected():
    for bounds in ([[1, 1]], [[2, 1]], [[np.nan, 2]], [[1, np.inf]]):
        with pytest.raises(ValueError):
            interval_hits([[0, 180]], bounds)


def test_window_weighting_recomposition_and_failure():
    records = []
    for seed in SEEDS:
        for subject, group, value in [(1, 'sa_present', 0), (2, 'sa_present', 4), (2, 'sa_present', 4), (2, 'no_sa_annotation', 8)]:
            row = dict(seed=seed, samp_id=subject, sa_group=group)
            row.update({metric: value for metric in METRICS})
            row.update({flag: True for flag in METRICS.values()})
            records.append(row)
    frame = pd.DataFrame(records)
    per_seed, across = summarize(frame)
    assert np.allclose(per_seed.loc[per_seed.sa_group.eq('sa_present'), 'mean'], 8/3)
    assert per_seed.loc[per_seed.sa_group.eq('all'), 'mean'].eq(4).all()
    assert across.seed_sd.eq(0).all()
    frame.loc[0, 'local_rr_mae_bpm'] = np.nan
    with pytest.raises(ValueError, match='非有限'):
        summarize(frame)
