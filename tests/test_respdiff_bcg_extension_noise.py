"""新增采样 seed 合同与敏感性汇总：CPU synthetic fixture。"""
import json
import numpy as np
import pandas as pd
import pytest
import torch

from resp_train.respdiff.model import RespDiffSpec
from resp_train.respdiff_bcg.baseband import RespDiffBCGBaseband
from resp_train.respdiff_bcg.runtime import seed_all,sha256
from resp_train.respdiff_bcg.sampling_accelerated import AccelerationSpec,accelerated_prefixes
from resp_train.respdiff_bcg.study_contract import METRICS,NAME,StudySamplerSpec
from resp_train.respdiff_bcg.study_extension import CHECKPOINT,ExtensionSpec
from resp_train.respdiff_bcg.study_extension_noise import (REVISION,ExtensionNoiseSpec,validate_sensitivity,summarize_noise)
from scripts.run_respdiff_bcg_extension_noise_v1 import main


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    previous=torch.get_num_threads();torch.set_num_threads(1)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    def forbidden(*args,**kwargs):raise AssertionError('CPU fixture 禁止 CUDA 初始化')
    monkeypatch.setattr(torch.cuda,'_lazy_init',forbidden)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('kind,steps,n',[('ddim',6,64),('ddpm',50,16)])
@pytest.mark.parametrize('seed',[20261013,20261014])
def test_same_seed_old_prefix_remains_exact(kind,steps,n,seed):
    seed_all(7,cpu_only=True)
    model=RespDiffBCGBaseband(RespDiffSpec(8,1,4),objective='source_equivalent').eval()
    condition=torch.randn(2,1,600);keys=[('val',3,0),('val',5,1)]
    old=StudySamplerSpec(kind,steps,n//2,seed)
    new=ExtensionNoiseSpec(kind,steps,n,seed)
    with torch.no_grad():
        expected=dict(accelerated_prefixes(model,condition,keys,old,AccelerationSpec(1),checkpoints=(n//2,)))
        actual=dict(accelerated_prefixes(model,condition,keys,new,AccelerationSpec(1),checkpoints=(n//2,n)))
    assert torch.equal(actual[n//2],expected[n//2])
    assert torch.isfinite(actual[n]).all()


def test_seed_scope_and_original_extension_remain_strict():
    for args in [('ddim',6,64,20261003),('ddpm',50,32,20261013),('ddim',10,64,20261014)]:
        with pytest.raises(ValueError):ExtensionNoiseSpec(*args)
    with pytest.raises(ValueError):ExtensionSpec('ddim',6,64,20261013)


def row(kind,s,n,seed):
    return {'sampler':kind,'nfe_per_trajectory':s,'N':n,'sampling_seed':seed,'training_seed':20260811,
        'checkpoint_sha256':CHECKPOINT,'n_parents':66,
        METRICS[0]:.8,METRICS[1]:1.,METRICS[2]:2.,METRICS[3]:.1}


def test_nine_point_sensitivity_requires_all_fixed_identities():
    curve=pd.DataFrame([row(k,s,n,seed) for k,s,n in [('ddim',6,64),('ddpm',50,16),('ddim',6,16)]
                        for seed in (20261003,20261013,20261014)])
    assert len(validate_sensitivity(curve))==12
    with pytest.raises(ValueError,match='九项'):validate_sensitivity(curve.iloc[:-1])
    curve.loc[0,'checkpoint_sha256']='different'
    with pytest.raises(ValueError,match='身份'):validate_sensitivity(curve)


def test_cli_requires_gpu_data_authorization(tmp_path):
    args=['run']
    for name in ('output','grid-run','reference-noise-run','selection-run','source-run','main-run'):
        args+=['--'+name,str(tmp_path/name)]
    args+=['--sampler','ddim']
    for flags in ([],['--confirm-data-access','--device','cpu']):
        with pytest.raises(SystemExit) as e:main(args+flags)
        assert e.value.code==2


def test_summary_copies_frozen_selection_and_reuses_reference(tmp_path):
    paths={k:tmp_path/k for k in ('grid','ref','selected','ddim','ddpm')}
    for p in paths.values():p.mkdir();(p/'val_parents.csv').write_text('dataset_row_id\n1\n')
    choice={'selection_noise_seed':20261003,'selected':[{'sampler':'ddpm','S':50,'N':16,'role':'selected'},
        {'sampler':'ddim','S':6,'N':64,'role':'selected'},{'sampler':'ddim','S':6,'N':16,'role':'fixed_reference'}]}
    def receipt(key,phase,**extra):
        p=paths[key]
        r={'status':'complete','protocol':NAME,'formal_evidence':True,'phase':phase,
           'checkpoint':{'checkpoint_sha256':CHECKPOINT,'parameter_sha256':'same'},**extra,
           'artifacts':{f.name:sha256(f) for f in p.iterdir() if f.name!='receipt.json'}}
        (p/'receipt.json').write_text(json.dumps(r));return sha256(p/'receipt.json')
    gh=receipt('grid','grid')
    pd.DataFrame([row('ddim',6,16,seed) for seed in (20261013,20261014)]).to_csv(paths['ref']/'sampler_grid.csv',index=False)
    rh=receipt('ref','noise-check')
    pd.DataFrame([row(k,s,n,20261003) for k,s,n in [('ddim',6,64),('ddpm',50,16),('ddim',6,16)]]).to_csv(paths['selected']/'sampler_grid.csv',index=False)
    (paths['selected']/'selection.json').write_text(json.dumps(choice))
    (paths['selected']/'sources.json').write_text(json.dumps({'grid_receipt_sha256':gh,'ddim_receipt_sha256':'main-ddim','ddpm_receipt_sha256':'main-ddpm'}))
    sh=receipt('selected','select-extension')
    for kind,s,n in [('ddim',6,64),('ddpm',50,16)]:
        pd.DataFrame([row(kind,s,n,seed) for seed in (20261013,20261014)]).to_csv(paths[kind]/'sampler_grid.csv',index=False)
        receipt(kind,'extension-noise-check',revision=REVISION,
            specs=[ExtensionNoiseSpec(kind,s,n,seed).identity() for seed in (20261013,20261014)],
            sources={'selection_receipt_sha256':sh,'grid_receipt_sha256':gh,'reference_noise_receipt_sha256':rh,'main_receipt_sha256':'main-'+kind},
            parameter_sha256_after='same',prefix_verifications=[{'passed':True},{'passed':True}])
    summarize_noise(tmp_path/'summary',selection_run=paths['selected'],grid_run=paths['grid'],
                    reference_noise_run=paths['ref'],ddim_run=paths['ddim'],ddpm_run=paths['ddpm'])
    assert json.loads((tmp_path/'summary/selection.json').read_text())==choice
    assert len(pd.read_csv(tmp_path/'summary/sensitivity.csv'))==9
