"""独立test入口的合成故障注入；不读取真实数据或使用GPU。"""
import numpy as np
import pandas as pd
import pytest
import torch

from scripts import eval_m4_s1_research_test as rt


def fixture_batch():
    rows = pd.DataFrame({"dataset_row_id": [1, 2], "samp_id": [10, 11], "split": ["test", "test"]})
    batch = {"x": torch.zeros(2, 1, 18000), "target": torch.zeros(2, 1, 18000),
             "meta": {c: rows[c].to_numpy() for c in rows}}
    return rows, batch


def test_guard_requires_explicit_authorization():
    for value in (False, None, 1, "true"):
        with pytest.raises(PermissionError):
            rt.guard(value)
    rt.guard(True)


def test_batch_rejects_permutation_nonfinite_and_incomplete():
    rows, batch = fixture_batch()
    assert len(list(rt.guarded_batches([batch], rows, None))) == 1
    with pytest.raises(ValueError, match="顺序"):
        list(rt.guarded_batches([batch], rows.iloc[::-1], None))
    with pytest.raises(ValueError, match="完整"):
        list(rt.guarded_batches([], rows, None))
    batch["x"][0, 0, 3] = float("nan")
    with pytest.raises(FloatingPointError):
        list(rt.guarded_batches([batch], rows, None))


def test_batch_requires_native_cwt_shape():
    rows, batch = fixture_batch()
    batch["tf"] = {"w": torch.zeros(2, 3, 4)}
    assert len(list(rt.guarded_batches([batch], rows, [3, 4]))) == 1
    with pytest.raises(FloatingPointError):
        list(rt.guarded_batches([batch], rows, [4, 3]))
    with pytest.raises(ValueError, match="keys"):
        list(rt.guarded_batches([batch], rows, None))


def test_cache_rejects_row_mismatch_and_nonfinite(tmp_path):
    rep = {"shape": [3, 4]}
    rt.io.write_json(tmp_path / "cache.json", {"representation": rep, "split": "test", "finite": True})
    np.save(tmp_path / "test_row_ids.npy", np.array([1, 2], dtype=np.int64))
    values = np.zeros((2, 3, 4), dtype=np.float32)
    values[1, 0, 0] = np.nan
    np.save(tmp_path / "test_w.npy", values)
    with pytest.raises(ValueError):
        rt.TestReader(tmp_path, rep, np.array([2, 1]))
    reader = rt.TestReader(tmp_path, rep, np.array([1, 2]))
    assert reader.get(0, 1)["w"].shape == (3, 4)
    with pytest.raises(ValueError):
        reader.get(0, 2)
    with pytest.raises(FloatingPointError):
        reader.get(1, 2)


def test_partial_matrix_never_summarized():
    with pytest.raises(ValueError, match="完整84"):
        rt.summary_tables(pd.DataFrame({"arm": ["B"], "seed": [rt.spec.SEEDS[0]]}))


def test_test_summary_matches_frozen_validation_math(monkeypatch):
    # 仅缩小合成样本门禁，完整28×3矩阵仍由真实统计函数检查。
    monkeypatch.setattr(rt, "check_rows", lambda rows, development: None)
    rows = []
    for i, arm in enumerate(rt.spec.ARMS):
        for seed in rt.spec.SEEDS:
            for row_id in range(4):
                row = {"arm": arm, "seed": seed, "dataset_row_id": row_id, "samp_id": row_id//2,
                       "split": "test", "joint_prediction_degenerate": False,
                       "envelope_spearman_prediction_degenerate": False,
                       "local_rr_target_eligible_windows": 1}
                row.update({m: .2 + i*.003 + row_id*.001 for m in rt.sf.PRIMARY})
                row.update({flag: True for flag in rt.sf.ELIGIBILITY.values()})
                rows.append(row)
    frame = pd.DataFrame(rows)
    test = rt.summary_tables(frame)
    validation = rt.stats.summary_tables(frame.assign(split="val"))
    for name in test:
        pd.testing.assert_frame_equal(test[name], validation[name])
    broken = frame.copy()
    broken.loc[0, rt.sf.PRIMARY[0]] = np.inf
    with pytest.raises(FloatingPointError):
        rt.summary_tables(broken)
