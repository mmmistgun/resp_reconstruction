"""P2-A1 的定向 CPU 合成检查；不访问真实数据或 GPU。"""
import json
import math

import pytest
import torch

from resp_train.respdiff.model import RespDiffSpec
from resp_train.respdiff_bcg.baseband import RespDiffBCGBaseband
from resp_train.respdiff_bcg.inference_v2 import state_digest
from resp_train.respdiff_bcg.runtime import seed_all,sha256
from resp_train.respdiff_bcg.study_audit import (comparison,context_metrics,context_predictions,
    gradient_case,gradient_metrics,posterior_reference,sampling_parity,_load_checkpoint)
from resp_train.respdiff_bcg.study_contract import StudySamplerSpec
from scripts.audit_respdiff_bcg_training_sampling_v1 import main


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    previous=torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    def forbid(name):
        def forbidden(*args,**kwargs):
            raise AssertionError(f'CPU 合成检查调用了 CUDA: {name}')
        return forbidden
    for name in ('_lazy_init','synchronize','manual_seed_all','get_rng_state','get_rng_state_all'):
        monkeypatch.setattr(torch.cuda,name,forbid(name))
    yield
    torch.set_num_threads(previous)


def model():
    seed_all(20261011,cpu_only=True)
    return RespDiffBCGBaseband(RespDiffSpec(8,1,4),objective='source_equivalent')


def test_production_loss_gradient_parity_and_state():
    m=model()
    x,y,noise=[torch.randn(3,1,600) for _ in range(3)]
    before=state_digest(m)
    for t in (0,19,49):
        values,groups,checks,_=gradient_case(m,x,y,noise,t)
        assert all(row['passed'] for row in checks),checks
        assert values['loss_noise']>0 and values['loss_spectral_weighted']>0
        assert groups[0]['n_parameters']==sum(p.numel() for p in m.parameters())
        assert all(p.grad is None for p in m.parameters())
    assert state_digest(m)==before


def test_gradient_geometry_and_undefined_zero_norm():
    names=[('diffusion_model.de.fc1.weight',None)]
    n=[torch.tensor([1.,2.])]
    rows=gradient_metrics(names,n,[torch.tensor([-2.,-4.])])
    assert rows[0]['spectral_over_noise_gradient_l2']==pytest.approx(2)
    assert rows[0]['gradient_cosine']==pytest.approx(-1)
    assert rows[0]['total_vs_noise_cosine']==pytest.approx(-1)
    assert rows[0]['total_gradient_l2']==pytest.approx(math.sqrt(5))
    empty=gradient_metrics(names,[torch.zeros(2)],[torch.ones(2)])[0]
    assert empty['zero_noise_norm'] and empty['gradient_cosine'] is None
    with pytest.raises(ValueError,match='非有限'):
        gradient_metrics(names,[torch.tensor([float('nan'),0.])],n)


def test_posterior_all_steps_and_t0_noise_guard():
    m=model();x,e,z=[torch.randn(3,1,31) for _ in range(3)]
    for t in range(50):
        actual=m.reverse_step(x,e,step=t,noise=z)
        ref=posterior_reference(m,x,e,t,z)
        assert comparison('posterior',actual,ref,atol=2e-6,rtol=2e-5)['passed']
    assert torch.equal(m.reverse_step(x,e,step=0,noise=z),m.reverse_step(x,e,step=0,noise=z*100))


@pytest.mark.parametrize('kind,steps',[('ddim',6),('ddpm',50)])
def test_actual_sampler_entry_parity(kind,steps):
    m=model().eval();x=torch.randn(2,1,600);before=state_digest(m)
    checks=sampling_parity(m,x,[('train',0,0),('train',0,1)],StudySamplerSpec(kind,steps,1))
    assert len(checks)==3*steps+1
    assert all(row['passed'] for row in checks)
    assert state_digest(m)==before


def test_context_comparison_aligns_chunk_identity():
    m=model().eval();x=torch.randn(6,1,600);y=torch.randn_like(x)
    x[3:]*=3
    keys=[('train',i//2,i%2) for i in range(6)]
    before=state_digest(m)
    a,fine,coarse=context_predictions(m,x,y,keys,list(range(6)),batch_size=4,timesteps=(19,49))
    b,fb,cb=context_predictions(m,x,y,keys,[5,2,1,3,0,4],batch_size=4,timesteps=(19,49))
    rows=context_metrics(a,b,fine,fb,coarse,cb,y)
    assert len(rows)==12 and {r['global_chunk_index'] for r in rows}==set(range(6))
    assert max(r['fine_condition_context_rms'] for r in rows)<1e-5
    assert max(r['coarse_condition_context_rms'] for r in rows)>1e-3
    assert state_digest(m)==before


def test_checkpoint_hash_and_cli_guards(tmp_path):
    m=model();checkpoint=tmp_path/'final.pt'
    torch.save({'update':6400,'model_spec':vars(m.spec),'model':m.state_dict(),'selector':'final_update'},checkpoint)
    digest=sha256(checkpoint)
    (tmp_path/'receipt.json').write_text(json.dumps({'status':'complete','formal_evidence':True,
        'objective':'source_equivalent','artifacts':{'final.pt':digest}}))
    assert _load_checkpoint(m,tmp_path,'final.pt',digest,6400,'baseline')==state_digest(m)
    with pytest.raises(ValueError,match='哈希'):
        _load_checkpoint(m,tmp_path,'final.pt','0'*64,6400,'baseline')
    with pytest.raises(SystemExit):
        main(['--baseline-run',str(tmp_path),'--p2-run',str(tmp_path),'--output',str(tmp_path/'out')])
    assert not (tmp_path/'out').exists()
