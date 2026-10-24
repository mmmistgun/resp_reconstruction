"""中断缓存恢复必须与冻结采样逐值一致，损坏缓存须显式失败。"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

spec = importlib.util.spec_from_file_location('recover', Path(__file__).with_name('recover.py'))
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
from resp_train.respdiff.model import RespDiffSpec
from resp_train.respdiff_bcg.baseband import RespDiffBCGBaseband
from resp_train.respdiff_bcg.inference_v2 import predict_prefixes, state_digest
from resp_train.respdiff_bcg.sampling_accelerated import AccelerationSpec
from resp_train.respdiff_bcg.study_contract import StudySamplerSpec
from resp_train.respdiff_bcg.runtime import seed_all


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    old = torch.get_num_threads(); torch.set_num_threads(1)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    def forbidden(*args, **kwargs): raise AssertionError('CPU 合成检查不得初始化 CUDA')
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize('sampler,steps', [('ddim', 6), ('ddpm', 50)])
def test_interruption_resume_matches_frozen_operator(tmp_path, monkeypatch, sampler, steps):
    seed_all(12, cpu_only=True)
    model = RespDiffBCGBaseband(RespDiffSpec(8, 1, 4), objective='source_equivalent').eval()
    before = state_digest(model)
    class Data:
        rows = pd.DataFrame({'split':['synthetic'], 'dataset_row_id':[42]})
        def __len__(self): return 13
        def __getitem__(self, i):
            return {'x':torch.sin(torch.arange(600).float() * .01 + i).reshape(1,600), 'index':i}
    data = Data(); sampler_spec = StudySamplerSpec(sampler, steps, 2)
    reference = tmp_path/'reference';reference.mkdir()
    expected, _ = predict_prefixes(model, data, sampler_spec, (1,2), reference, batch_size=4, acceleration=AccelerationSpec(1))
    original = module.commit_batch
    def interrupt(*args, **kwargs):
        original(*args, **kwargs)
        if args[2] == 4: raise RuntimeError('synthetic interruption after two committed batches')
    with monkeypatch.context() as mp:
        mp.setattr(module,'commit_batch',interrupt)
        with pytest.raises(RuntimeError,match='interruption'):
            module.predict_cached(model, data, sampler_spec, (1,2), tmp_path/'interrupted', {'seed':12}, batch_size=4)
    calls=[]; kernel=module.accelerated_prefixes
    def counted(*args,**kwargs):
        calls.append(len(args[1]));return kernel(*args,**kwargs)
    monkeypatch.setattr(module,'accelerated_prefixes',counted)
    actual, profile = module.predict_cached(model, data, sampler_spec, (1,2), tmp_path/'resumed', {'seed':12},
        batch_size=4,resume_caches=[tmp_path/'interrupted/batch_cache'])
    assert calls == [4,1]
    for n in (1,2):assert np.array_equal(actual[n],expected[n])
    assert all(p['reused_batches']==2 for p in profile)
    assert state_digest(model)==before
    identity=json.loads((tmp_path/'interrupted/batch_cache/identity.json').read_text())
    with (tmp_path/'interrupted/batch_cache/batch_000000.npz').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='hash'):
        module.load_cached_batch(tmp_path/'interrupted/batch_cache',identity,0,4,(1,2))
    with pytest.raises(ValueError,match='合同'):
        module.predict_cached(model,data,sampler_spec,(1,2),tmp_path/'wrong_identity',{'seed':13},
            batch_size=4,resume_caches=[tmp_path/'resumed/batch_cache'])


def test_cli_requires_authorization(tmp_path):
    with pytest.raises(SystemExit) as error:module.main(['run','--output',str(tmp_path/'out')])
    assert error.value.code==2
