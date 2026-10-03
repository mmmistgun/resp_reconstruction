"""同batch波形硬检查和跨batch诊断的合成CPU定向测试。"""
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader
from scripts.run_cwt_apor_v2_mechanisms_r2 import wave_check, checked_pairing, cross_batch_report, validate_matrix, reuse_same_batch_audit, PRIOR_AUDIT, io
from scripts.recover_cwt_apor_v2_mechanisms import prediction_keys
from resp_train.paper_evidence.cwt_apor_v2 import interventions as iv
from resp_train.paper_evidence.cwt_apor_v2.model import build_model
from resp_train.paper_evidence.cwt_apor_v2.spec import config, SEEDS


def test_waveform_gate_rejects_local_drift_and_nonfinite():
    a=torch.ones(2,1,18000)
    assert max(r['max_abs'] for r in wave_check(a,a.clone(),[1,2]))==0
    b=a.clone()
    b[0,0,9000]+=.001
    with pytest.raises(AssertionError):
        wave_check(a,b,[1,2])
    b[0,0,9000]=float('nan')
    with pytest.raises(FloatingPointError):
        wave_check(a,b,[1,2])


def test_checked_loader_and_cross_batch_contract(tmp_path):
    torch.manual_seed(42)
    rep={'arm':{'name':'A0','pool_samples':50},'shape':[97,360],
         'frequencies_hz':np.geomspace(.0366,7.995,97),'time_seconds':(np.arange(360)*50+24.5)/100}
    model=build_model(SEEDS[0],rep).eval()
    model.base.local_blocks=nn.ModuleList([nn.Identity() for _ in range(6)])
    with torch.no_grad():
        model.branches['w'].final_projection.weight.normal_(0,.005)
    t=torch.arange(18000)/100
    target=((1+.15*torch.sin(2*torch.pi*.02*t))*torch.sin(2*torch.pi*.25*t))[None]
    dataset=[{'x':target.clone(),'target':target.clone(),'tf':{'w':torch.rand(97,360)},
              'meta':{'dataset_row_id':i,'samp_id':11,'split':'val'}} for i in range(3)]
    cfg=config('A0',SEEDS[0],tmp_path)
    original=iv.paired_batch
    audit=[]
    with prediction_keys(), checked_pairing([0,1,2],audit):
        frame=iv.evaluate_loader(model,DataLoader(dataset,batch_size=2),cfg,rep,
                                  np.tile([60,120,240],(3,1)),{2},tmp_path,SEEDS[0])
    assert iv.paired_batch is original
    assert [r['dataset_row_id'] for r in audit]==[0,1,2]
    assert all(r['max_abs']==0 for r in audit)
    source=frame[frame.condition.eq('FULL__NAT')].reset_index(drop=True).copy()
    validate_matrix(frame,source[['dataset_row_id','samp_id','split']],SEEDS[0])
    source['whole_rr_abs_error_bpm']+=.2
    report=cross_batch_report(frame,source)
    assert not report.set_index('metric').loc['whole_rr_abs_error_bpm','previous_mean_gate_passed']
    source.loc[0,'dataset_row_id']=100
    with pytest.raises(ValueError,match='身份'):
        cross_batch_report(frame,source)
    assert len(list(tmp_path.glob('case_*.npz')))==18
    # 已执行完的波形检查可以独立认证；有任何非零记录时不能借历史指标门槛修订通过。
    prior=tmp_path/PRIOR_AUDIT
    prior.mkdir(parents=True)
    entry={'seed':SEEDS[0]}
    io.write_json(prior/'failed.json',{'traceback':'Not equal to tolerance rtol=1e-07, atol=1e-08'})
    io.write_json(prior/'access.json',{'entry':entry,'amendment':{}})
    pd.DataFrame(audit).to_csv(prior/'same_batch_waveform_check.csv',index=False)
    current=frame[frame.condition.eq('FULL__NAT')].reset_index(drop=True)
    current.to_csv(prior/'replayed_full_metrics.csv',index=False)
    certificate=tmp_path/'certificate'
    certificate.mkdir()
    assert len(reuse_same_batch_audit(tmp_path,entry,frame,certificate))==3
    audit[0]['max_abs']=1.
    pd.DataFrame(audit).to_csv(prior/'same_batch_waveform_check.csv',index=False)
    with pytest.raises(ValueError,match='波形验收'):
        reuse_same_batch_audit(tmp_path,entry,frame,certificate)


def test_pairing_context_restores_after_error():
    original=iv.paired_batch
    with pytest.raises(RuntimeError,match='fixture'):
        with checked_pairing([0],[]):
            raise RuntimeError('fixture')
    assert iv.paired_batch is original
