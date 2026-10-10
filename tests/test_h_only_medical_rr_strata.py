"""医学阈值端点、主体错配和窗口权重的 synthetic CPU 检查。"""
import numpy as np
import pandas as pd
import pytest

from scripts.analyze_h_only_medical_rr_strata import (
    METRICS, SEEDS, check_alignment, classify_rr, summarize,
)


def test_medical_endpoints_and_invalid_reference():
    values = [11.999999, 12, 12.000001, 19.999999, 20, 20.000001]
    assert classify_rr(values).tolist() == ['slow', *(['reference_range'] * 4), 'fast']
    for bad in ([np.nan], [np.inf], [0], [-1]):
        with pytest.raises(ValueError):
            classify_rr(bad)


def test_window_weighting_eligibility_empty_stratum_and_nonfinite():
    rows = []
    for seed in SEEDS:
        # 两个主体贡献1与3窗，窗口均值为3，主体等权均值为2。
        for subject, value in [(1, 0), (2, 4), (2, 4), (2, 4)]:
            row = dict(seed=seed, samp_id=subject, rr_stratum='slow')
            row.update({metric: value for metric in METRICS})
            row.update({flag: True for flag in METRICS.values()})
            rows.append(row)
    frame = pd.DataFrame(rows)
    per_seed, across = summarize(frame)
    assert per_seed.loc[per_seed.rr_stratum.eq('slow'), 'mean'].eq(3).all()
    assert across.loc[across.rr_stratum.eq('slow'), 'seed_sd'].eq(0).all()
    empty = per_seed.loc[per_seed.rr_stratum.eq('fast')]
    assert empty.eligible_n.eq(0).all() and empty['mean'].isna().all()
    frame.loc[0, 'whole_rr_target_eligible'] = False
    per_seed, _ = summarize(frame)
    value = per_seed.loc[per_seed.seed.eq(SEEDS[0]) & per_seed.rr_stratum.eq('slow') &
                         per_seed.metric.eq('whole_rr_abs_error_bpm')].iloc[0]
    assert value.eligible_n == 3 and value['mean'] == 4
    frame.loc[1, 'local_rr_mae_bpm'] = np.nan
    with pytest.raises(ValueError, match='非有限'):
        summarize(frame)


def test_same_count_but_subject_or_order_mismatch_rejected():
    rows = pd.DataFrame(dict(dataset_row_id=[1, 2], samp_id=[10, 20], split=['test', 'test']))
    metrics = rows.assign(seed=SEEDS[0], arm='H')
    check_alignment(metrics, rows, SEEDS[0])
    for bad in (metrics.iloc[::-1], metrics.assign(samp_id=[20, 10])):
        with pytest.raises(ValueError):
            check_alignment(bad, rows, SEEDS[0])
