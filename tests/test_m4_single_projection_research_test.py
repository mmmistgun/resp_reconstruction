"""单投影 research-test 的确认、固定来源与矩阵接口检查。"""
import numpy as np
import pandas as pd
import pytest
import torch
from scripts import eval_m4_single_projection as run


def test_contract_and_confirmation_gate(tmp_path):
    assert run.COUNT == 2310 and run.SUBJECTS == 8 and run.BATCH_SIZE == 32
    assert run.test_contract()["comparisons"] == {"SP": "M4"}
    assert run.test_contract()["historical_reference_arm"] == "M4"
    for action in (lambda: run.evaluate(tmp_path, "SP_seed20260811", "cpu"),
                   lambda: run.summarize(tmp_path)):
        with pytest.raises((ValueError, PermissionError)):
            action()


def test_reference_cannot_be_reevaluated(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "load_allowlist", lambda root:
        {"entries": [{"arm": "M4", "cell": "M4_seed20260811"}]})
    with pytest.raises(ValueError, match="只读"):
        run.evaluate(tmp_path, "M4_seed20260811", "cpu", confirmed=True)


@pytest.mark.parametrize("arm", ["M4", "SP"])
def test_checkpoint_frequency_identity(arm):
    f = np.r_[np.geomspace(.03, .8, 56), np.geomspace(.81, 8, 41)]
    state = {"condition.frequencies_hz": torch.tensor(f)}
    np.testing.assert_array_equal(run.checkpoint_frequencies(state, arm, f), f)
    with pytest.raises(ValueError):
        run.checkpoint_frequencies(state, arm, f * 1.01)


def test_summary_requires_complete_pairing():
    # 完整test指标校验由共享入口负责；此处只验证本轮完整矩阵门槛。
    with pytest.raises(ValueError, match="完整"):
        run.training.summary_tables(pd.DataFrame({"arm": ["SP"], "seed": [20260811]}))
