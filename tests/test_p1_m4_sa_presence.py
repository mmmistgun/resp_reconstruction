"""同数量窗口的THO定位/模型身份漂移必须拒绝。"""
import pandas as pd
import pytest
from scripts.analyze_p1_m4_sa_presence import validate_rows


def test_row_identity_includes_tho_position_and_model():
    rows = pd.DataFrame(dict(dataset_row_id=[1, 2], samp_id=[10, 20], split=['test']*2,
                            target_source_npz=['a', 'b'], target_signal_key=['tho']*2,
                            window_start_sample=[0, 100], window_end_sample=[18000, 18100], target_fs=[100]*2))
    attrs = rows[['dataset_row_id', 'samp_id', 'split']]
    metrics = attrs.assign(arm='M4', seed=20260811, method='P1-Full-Add')
    validate_rows(metrics, rows, rows, attrs, 20260811)
    with pytest.raises(ValueError, match='THO定位'):
        validate_rows(metrics, rows.assign(window_start_sample=[1, 100]), rows, attrs, 20260811)
    with pytest.raises(ValueError, match='模型身份'):
        validate_rows(metrics.assign(arm='M0'), rows, rows, attrs, 20260811)
    with pytest.raises(ValueError, match='行顺序'):
        validate_rows(metrics.iloc[::-1], rows, rows, attrs, 20260811)
