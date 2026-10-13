"""P4-N2 独立合同的 CPU synthetic 定向验证。"""
import json
import numpy as np
import pandas as pd
import pytest
import torch

from resp_train.respdiff.model import RespDiffSpec
from resp_train.respdiff_bcg.baseband import RespDiffBCGBaseband
from resp_train.respdiff_bcg.inference_v2 import reconstruct_ensemble_parent
from resp_train.respdiff_bcg.runtime import seed_all
from resp_train.respdiff_bcg.sampling_accelerated import AccelerationSpec,accelerated_prefixes
from resp_train.respdiff_bcg.study_contract import GRID,METRICS,StudySamplerSpec
from resp_train.respdiff_bcg.study_extension import ExtensionSpec,compare_prefix,select_extended
from scripts.run_respdiff_bcg_extension_v1 import main


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    previous=torch.get_num_threads();torch.set_num_threads(1)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    def forbidden(*args,**kwargs):raise AssertionError('CPU fixture 禁止 CUDA 初始化')
    monkeypatch.setattr(torch.cuda,'_lazy_init',forbidden)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('kind,steps,new_n',[('ddim',6,64),('ddpm',50,16)])
def test_extended_sampler_keeps_exact_old_prefix(kind,steps,new_n):
    seed_all(7,cpu_only=True)
    model=RespDiffBCGBaseband(RespDiffSpec(8,1,4),objective='source_equivalent').eval()
    condition=torch.randn(2,1,600)
    keys=[('val',3,0),('val',5,1)]
    new=ExtensionSpec(kind,steps,new_n)
    old=StudySamplerSpec(kind,steps,new_n//2)
    assert old.timesteps==new.timesteps
    with torch.no_grad():
        expected=dict(accelerated_prefixes(model,condition,keys,old,AccelerationSpec(1),checkpoints=(new_n//2,)))
        actual=dict(accelerated_prefixes(model,condition,keys,new,AccelerationSpec(1),checkpoints=(new_n//2,new_n)))
    assert torch.equal(actual[new_n//2],expected[new_n//2])
    assert torch.isfinite(actual[new_n]).all()
    with pytest.raises(ValueError):StudySamplerSpec(kind,steps,new_n)


def test_extension_rejects_unapproved_budgets_and_noise():
    for args in [('ddim',6,32),('ddpm',50,32),('ddim',10,64),('ddim',6,128),('ddim',6,64,20261013)]:
        with pytest.raises(ValueError):ExtensionSpec(*args)


def test_old_waveform_prefix_guard(tmp_path):
    chunks=np.random.default_rng(5).normal(size=(2,13,1,600)).astype(np.float32)
    expected=np.stack([reconstruct_ensemble_parent(parent,postfilter=True)[2] for parent in chunks])
    path=tmp_path/'reference.npz';np.savez(path,prediction_postfiltered=expected)
    result=compare_prefix(chunks,path,tmp_path/'pass.json')
    assert result['passed'] and result['exact_equal']
    np.savez(tmp_path/'changed.npz',prediction_postfiltered=expected+1)
    with pytest.raises(ValueError,match='前缀'):
        compare_prefix(chunks,tmp_path/'changed.npz',tmp_path/'fail.json')
    assert json.loads((tmp_path/'fail.json').read_text())['passed'] is False


def test_selection_requires_full_24_points_and_preserves_rule():
    keys=[(kind,s,n) for (kind,s),ns in GRID.items() for n in ns]+[('ddim',6,64),('ddpm',50,16)]
    records=[]
    for kind,s,n in keys:
        records.append({'sampler':kind,'nfe_per_trajectory':s,'N':n,'denoiser_calls_per_chunk':s*n,
            METRICS[0]:.8 if (kind,s,n) in [('ddim',6,64),('ddpm',50,16)] else .7,
            METRICS[1]:1.,METRICS[2]:2.,METRICS[3]:.1})
    curve=pd.DataFrame(records)
    result=select_extended(curve)
    assert [(r['sampler'],r['N']) for r in result['selected'][:2]]==[('ddpm',16),('ddim',64)]
    assert result['selected'][2]['role']=='fixed_reference'
    assert len(result['new_doubling_comparisons'])==2
    with pytest.raises(ValueError,match='24'):select_extended(curve.iloc[:-1])
    curve.loc[0,METRICS[1]]=np.nan
    with pytest.raises(ValueError,match='数值'):select_extended(curve)


def test_cli_rejects_missing_authorization_and_cpu(tmp_path):
    args=['run','--output',str(tmp_path/'out'),'--grid-run',str(tmp_path/'grid'),
          '--source-run',str(tmp_path/'source'),'--sampler','ddim']
    for flags in ([],['--confirm-data-access','--device','cpu']):
        with pytest.raises(SystemExit) as error:main(args+flags)
        assert error.value.code==2


def test_summary_binds_both_completed_arms_to_original_grid(tmp_path):
    from resp_train.respdiff_bcg.runtime import sha256
    from resp_train.respdiff_bcg.study_contract import NAME
    from resp_train.respdiff_bcg.study_extension import CHECKPOINT,REVISION,summarize_extensions
    def row(kind,s,n):
        return {'sampler':kind,'nfe_per_trajectory':s,'N':n,'denoiser_calls_per_chunk':s*n,
            METRICS[0]:.8,METRICS[1]:1.,METRICS[2]:2.,METRICS[3]:.1,
            'sampling_seed':20261003,'training_seed':20260811,'checkpoint_sha256':CHECKPOINT}
    grid=tmp_path/'grid';grid.mkdir()
    pd.DataFrame([row(kind,s,n) for (kind,s),ns in GRID.items() for n in ns]).to_csv(grid/'sampler_grid.csv',index=False)
    (grid/'val_parents.csv').write_text('dataset_row_id\n1\n')
    base={'status':'complete','protocol':NAME,'formal_evidence':True}
    def receipt(path,values):
        obj={**base,**values,'artifacts':{name:sha256(path/name) for name in ['sampler_grid.csv','val_parents.csv']}}
        (path/'receipt.json').write_text(json.dumps(obj));return obj
    receipt(grid,{'phase':'grid'})
    paths=[]
    for kind,s,n in [('ddim',6,64),('ddpm',50,16)]:
        path=tmp_path/kind;path.mkdir();paths.append(path)
        pd.DataFrame([row(kind,s,n)]).to_csv(path/'sampler_grid.csv',index=False)
        (path/'val_parents.csv').write_bytes((grid/'val_parents.csv').read_bytes())
        receipt(path,{'phase':'grid-extension','revision':REVISION,'spec':ExtensionSpec(kind,s,n).identity(),
            'prefix_verification':{'passed':True},'grid_receipt_sha256':sha256(grid/'receipt.json'),
            'checkpoint':{'checkpoint_sha256':CHECKPOINT,'parameter_sha256':'same'},'parameter_sha256_after':'same'})
    summarize_extensions(tmp_path/'summary',grid_run=grid,ddim_run=paths[0],ddpm_run=paths[1])
    assert len(pd.read_csv(tmp_path/'summary/sampler_grid.csv'))==24
    bad=json.loads((paths[1]/'receipt.json').read_text());bad['spec']['noise_seed']=20261013
    (paths[1]/'receipt.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='身份'):
        summarize_extensions(tmp_path/'invalid',grid_run=grid,ddim_run=paths[0],ddpm_run=paths[1])
