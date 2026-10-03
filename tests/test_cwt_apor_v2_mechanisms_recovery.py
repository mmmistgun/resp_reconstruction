"""CPU合成数据验证机制指标构造及真实案例导出路径。"""
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader
from scripts.recover_cwt_apor_v2_mechanisms import prediction_keys
from resp_train.engine.train import _prediction_dict_from_arrays
from resp_train.paper_evidence.cwt_apor_v2 import interventions as iv
from resp_train.paper_evidence.cwt_apor_v2.model import build_model
from resp_train.paper_evidence.cwt_apor_v2.spec import config, SEEDS


def test_prediction_adapter_uses_standard_keys_and_restores_on_failure():
    arrays = np.ones((1, 1, 18000), np.float32)
    metadata = [{"dataset_row_id": 7, "samp_id": 11, "split": "val"}]
    expected = _prediction_dict_from_arrays(arrays, arrays, metadata, pred_key="r_tho_hat", target_key="tho_ref")
    with pytest.raises(RuntimeError, match="fixture"):
        with prediction_keys():
            actual = iv._prediction_dict_from_arrays(arrays, arrays, metadata)
            assert actual.keys() == expected.keys()
            for name in actual:
                np.testing.assert_array_equal(actual[name], expected[name])
            raise RuntimeError("fixture")
    assert iv._prediction_dict_from_arrays is _prediction_dict_from_arrays


def test_full_loader_metrics_cases_and_short_final_batch(tmp_path):
    torch.manual_seed(42)
    rep = {"arm": {"name": "A0", "pool_samples": 50}, "shape": [97, 360],
           "frequencies_hz": np.geomspace(.0366, 7.995, 97),
           "time_seconds": (np.arange(360)*50+24.5)/100}
    model = build_model(SEEDS[0], rep).eval()
    # CPU合成测试保留真实条件/FiLM/解码路径；Mamba原生CUDA路径由既有GPU验收覆盖。
    model.base.local_blocks = nn.ModuleList([nn.Identity() for _ in range(6)])
    with torch.no_grad():
        model.branches['w'].final_projection.weight.normal_(0, .005)
    t = torch.arange(18000)/100
    target = ((1+.15*torch.sin(2*torch.pi*.02*t))*torch.sin(2*torch.pi*.25*t))[None]
    dataset = [{"x": target.clone(), "target": target.clone(), "tf": {"w": torch.rand(97,360)},
                "meta": {"dataset_row_id": i, "samp_id": 11, "split": "val"}} for i in range(3)]
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    cfg = config("A0", SEEDS[0], tmp_path)
    shifts = np.tile([60, 120, 240], (3,1))
    broken = tmp_path / "before"
    broken.mkdir()
    with pytest.raises(TypeError, match="pred_key.*target_key"):
        iv.evaluate_loader(model, loader, cfg, rep, shifts, {2}, broken, SEEDS[0])
    fixed = tmp_path / "after"
    fixed.mkdir()
    with prediction_keys():
        frame = iv.evaluate_loader(model, loader, cfg, rep, shifts, {2}, fixed, SEEDS[0])
    assert len(frame) == 3*18
    assert frame.groupby('condition').size().eq(3).all()
    assert set(frame.condition) == set(iv.CONDITIONS)
    assert len(list(fixed.glob('case_*.npz'))) == 18
    assert len(list(fixed.glob('case_*_metrics.json'))) == 18
    with np.load(fixed / 'case_2_FULL__NAT.npz') as case:
        assert case['Z'].shape == case['Z_prime'].shape == (96,140)
        np.testing.assert_array_equal(case['prediction_delta'], 0)
        np.testing.assert_array_equal(case['reference'], target.numpy())
    assert (fixed/'metrics.csv').is_file() and (fixed/'gn_stats.csv').stat().st_size > 0
    assert iv._prediction_dict_from_arrays is _prediction_dict_from_arrays
