"""确定性基线不能被复制为多个seed或伪造seed标准差。"""
import numpy as np
import pandas as pd
from scripts.analyze_m4_iewt_strata import summarize, METRICS


def test_single_iewt_and_three_seed_m4_are_aggregated_separately():
    rows = []
    for model, instances in [('IEWT', [7]), ('M4', [1, 2, 3])]:
        for instance in instances:
            for subject, value in [(1, instance), (2, instance+4), (2, instance+4)]:
                row = dict(model=model, instance=str(instance), samp_id=subject, rr_stratum='slow', sa_group='sa_present')
                row.update({metric:value for metric in METRICS})
                row.update({flag:True for flag in METRICS.values()})
                rows.append(row)
    _, summary = summarize(pd.DataFrame(rows))
    selected = summary.loc[summary.group.eq('all')]
    single = selected.loc[selected.model.eq('IEWT')]
    assert single.instance_n.eq(1).all() and single.instance_sd.isna().all()
    assert np.allclose(single['mean'], 7+8/3)
    seeds = selected.loc[selected.model.eq('M4')]
    assert seeds.instance_n.eq(3).all() and np.allclose(seeds.instance_sd, 1)
    assert np.allclose(seeds['mean'], 2+8/3)
