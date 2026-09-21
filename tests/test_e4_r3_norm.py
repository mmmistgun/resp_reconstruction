"""R3/GN synthetic CPU 合同与微型流程；不读取真实波形/cache。"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from resp_train.crd.tf_v1_model import CwtBranch
from resp_train.crd.config import load_crd_config
from resp_train.paper_evidence import e4_r3_norm as core
from resp_train.paper_evidence import e4_r3_norm_runtime as runtime
from resp_train.paper_evidence import e4_r3_signals as signals
from resp_train.paper_evidence.e4_aggregation_v2_model import cpu_module_seed
from resp_train.paper_evidence.e4_band_audit import paired_changes


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    previous=torch.get_num_threads(); torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    yield
    torch.set_num_threads(previous)


def fixture_model():
    with cpu_module_seed(7,'tf_branch_w'): branch=CwtBranch()
    with torch.no_grad():
        branch.final_projection.weight.fill_(.015)
        # temporal 残差打开，使全部 GN 控制都参与输出。
        for block in branch.temporal.blocks: block.project.weight.fill_(.01)
    class Model(nn.Module):
        def __init__(self):
            super().__init__(); self.branches=nn.ModuleDict({'w':branch})
        def forward(self,x,*,tf):
            gamma,beta=self.branches['w'](tf)
            return {'waveform':x+F.interpolate((gamma+beta).mean(1,keepdim=True),size=x.shape[-1],mode='linear',align_corners=False)}
    return Model().eval()


def fixture_config():
    return load_crd_config(Path(__file__).resolve().parents[1]/'configs/crd_tf_v1/crd_tf102_w_formal.yaml')


@pytest.mark.parametrize('kind',core.TRANSFORMS)
def test_transform_invariants_and_immutable_source(kind):
    w=torch.rand(2,97,360,generator=torch.Generator().manual_seed(19)); original=w.clone()
    reference=torch.linspace(.02,.3,97); shifts=torch.from_numpy(core.make_shifts(2))
    changed=core.transform_w(w,kind,reference,shifts)
    assert torch.equal(w,original) and torch.equal(changed[:,:73],w[:,:73])
    if kind.startswith('SHIFT') or kind=='REVERSE':
        torch.testing.assert_close(changed[:,73:].sort(-1).values,w[:,73:].sort(-1).values,rtol=0,atol=0)
        np.testing.assert_allclose(np.abs(np.fft.rfft(changed[:,73:].numpy())),np.abs(np.fft.rfft(w[:,73:].numpy())),rtol=1e-12,atol=1e-12)
    if kind=='WINDOW_FLAT':
        torch.testing.assert_close(changed[:,73:].mean(-1),w[:,73:].mean(-1))
        assert torch.count_nonzero(changed[:,73:].std(-1))==0
    if kind=='TRAIN_MEAN': torch.testing.assert_close(changed[:,73:],reference[None,73:,None].expand(2,24,360))


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('shape',[(2,48,97,360),(2,96,1800)])
def test_native_gn_stats_replay_and_counterfactual_formula(dtype,shape):
    x=torch.randn(shape,generator=torch.Generator().manual_seed(3)).to(dtype)
    module=nn.GroupNorm(8,shape[1]).eval()
    with torch.no_grad():
        module.weight.copy_(torch.linspace(.7,1.3,shape[1])); module.bias.copy_(torch.linspace(-.2,.2,shape[1]))
        y=module(x); stats=core.native_stats(module,x,y)
        replay=core.frozen_norm(module,x,stats,y.dtype)
        torch.testing.assert_close(replay,y,rtol=1e-5,atol=1e-6)
        shifted=core.GNStats(stats.mean+.1,stats.rstd*.9)
        counter=core.frozen_norm(module,x,shifted,y.dtype)
        groups=x.float().reshape(shape[0],8,-1)
        expected=((groups-shifted.mean[:,:,None])*shifted.rstd[:,:,None]).reshape(shape)
        broad=(1,shape[1],*([1]*(len(shape)-2)))
        expected=expected*module.weight.reshape(broad)+module.bias.reshape(broad)
        torch.testing.assert_close(counter.float(),expected,rtol=.008 if dtype==torch.bfloat16 else 1e-5,atol=.008 if dtype==torch.bfloat16 else 1e-6)
        assert not torch.equal(counter,y)


def test_full_22_conditions_statistics_selection_and_protected_scales():
    model=fixture_model(); batch=runtime.prior.synthetic_batch(1,7)
    batch['meta']={'dataset_row_id':torch.tensor([1]),'samp_id':torch.tensor([9]),'split':['val']}
    reference=torch.linspace(.02,.3,97); shifts=torch.from_numpy(core.make_shifts(1)); seen=[]; mean_stats=None
    for condition,result,trace,baseline in runtime.batch_conditions(model,batch,reference,shifts,'cpu'):
        seen.append(condition)
        if condition=='TRAIN_MEAN__NAT': mean_stats=trace.natural['norm']
        if condition.endswith('GN1_FIXED'):
            assert torch.equal(trace.used['norm'].mean,baseline.natural['norm'].mean)
            torch.testing.assert_close(trace.points['gn_front'][:,:,:71],baseline.points['gn_front'][:,:,:71],rtol=1e-5,atol=1e-6)
        if condition.endswith('ALL_W_GN_FIXED'):
            for name in core.GN_NAMES: assert torch.equal(trace.used[name].rstd,baseline.natural[name].rstd)
        if condition=='STAT_ONLY_TRAIN_MEAN':
            assert torch.equal(trace.points['conv_in'],baseline.points['conv_in'])
            assert torch.equal(trace.used['norm'].mean,mean_stats.mean)
            assert not torch.equal(trace.points['gn_front'],baseline.points['gn_front'])
        else:
            assert torch.equal(trace.points['conv_in'][:,:,:71],baseline.points['conv_in'][:,:,:71])
        assert np.isfinite(result['r_tho_hat']).all()
    assert seen==list(core.CONDITIONS)
    assert not any(m._forward_hooks for m in model.modules())


def test_invalid_stats_and_missing_source_fail():
    module=nn.GroupNorm(8,48); x=torch.zeros(1,48,97,360)
    with pytest.raises(ValueError): core.frozen_norm(module,x,core.GNStats(torch.zeros(2,8),torch.ones(2,8)),torch.float32)
    with pytest.raises(FloatingPointError): core.frozen_norm(module,x,core.GNStats(torch.zeros(1,8),torch.zeros(1,8)),torch.float32)
    model=fixture_model(); wrapper=core.NormIntervention(model,'FULL__GN1_FIXED',torch.zeros(97),torch.from_numpy(core.make_shifts(1))).eval()
    with torch.no_grad(),pytest.raises(ValueError,match='缺少'): wrapper(torch.zeros(1,1,18000),tf={'w':torch.zeros(1,97,360)})


def test_hooks_are_removed_after_failure():
    model=fixture_model(); module=model.branches['w'].final_projection
    def fail(*args): raise RuntimeError('synthetic downstream')
    handle=module.register_forward_pre_hook(fail)
    wrapper=core.NormIntervention(model,'FULL__NAT',torch.zeros(97),torch.from_numpy(core.make_shifts(1))).eval()
    with torch.no_grad(),pytest.raises(RuntimeError,match='synthetic downstream'):
        wrapper(torch.zeros(1,1,18000),tf={'w':torch.zeros(1,97,360)})
    handle.remove(); assert not any(m._forward_hooks for m in model.modules())


def test_signal_associations_missingness_and_frozen_time_axis(tmp_path):
    cfg=fixture_config(); t=np.arange(18000)/100
    target=np.sin(2*np.pi*.2*t)*(1+.1*np.sin(2*np.pi*.015*t))
    grid=(np.arange(360)*50+24.5)/100
    w=np.tile(.3+.1*np.sin(2*np.pi*.2*grid),(97,1)).astype(np.float32)
    row={'dataset_row_id':1,'samp_id':9,'split':'val'}
    frame=pd.DataFrame(signals.signal_window(w,target,[61,177,249],np.ones(97)*.2,cfg,row,tmp_path))
    aggregate=frame[(frame.representation=='R3_MEAN')&(frame.association=='cycle')&(frame.support=='full')]
    assert aggregate[aggregate['transform']=='FULL'].correlation.iloc[0]>.99
    assert not aggregate[aggregate['transform'].isin(['TRAIN_MEAN','WINDOW_FLAT'])].defined.any()
    assert set(frame['transform'])==set(core.TRANSFORMS)
    assert frame.representation.nunique()==25
    summary=signals.summarize_signals(frame); assert set(summary.scope)=={'pooled','samp_id'}
    paired=signals.paired_correlations(frame)
    assert paired.loc[paired['transform']=='FULL','absolute_drop'].dropna().eq(0).all()
    saved=np.load(tmp_path/'row_1.npz'); np.testing.assert_array_equal(saved['time'],grid)
    assert (tmp_path/'row_1.png').is_file()


def test_vectorized_envelope_matches_frozen_definition():
    from resp_train.metrics.task import TaskMetricConfig,_log_rms_envelope
    from dataclasses import replace
    task=replace(TaskMetricConfig.from_config(fixture_config()),envelope_window=8,envelope_step=2)
    values=np.random.default_rng(7).normal(size=(25,360))
    expected=np.stack([_log_rms_envelope(v,task) for v in values])
    np.testing.assert_array_equal(signals.log_rms_envelopes(values,task),expected)


def test_factorial_uses_raw_oriented_metric_and_interaction():
    frame=pd.DataFrame([dict(seed=1,scope='pooled',group='ALL',metric='error',condition=c,degradation=v)
        for c,v in [('TRAIN_MEAN__GN1_FIXED',2.),('STAT_ONLY_TRAIN_MEAN',3.),('TRAIN_MEAN__NAT',9.)]])
    result=core.factorial_table(frame).iloc[0]
    assert result.content==2 and result.statistics==3 and result.interaction==4 and result.total==9


def test_lifecycle_failure_tamper_and_no_overwrite(tmp_path,monkeypatch):
    monkeypatch.setattr(runtime,'ROOT',tmp_path); digest='f'*64
    with pytest.raises(RuntimeError):
        with runtime.attempt('fixture',digest) as failed: raise RuntimeError('fixture')
    assert (failed/'lifecycle_failed.json').exists()
    with runtime.attempt('fixture',digest) as good: (good/'data').write_text('a')
    runtime.verify_attempt(good,digest,'fixture')
    with pytest.raises(FileExistsError):
        with runtime.attempt('fixture',digest): pass
    (good/'data').write_text('b')
    with pytest.raises(RuntimeError): runtime.verify_attempt(good,digest,'fixture')


def test_bf16_full_replays_with_all_gn_fixed():
    model=fixture_model(); batch=runtime.prior.synthetic_batch(1,19)
    baseline=None; expected=None
    with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
        for mode in core.MODES:
            wrapper=core.NormIntervention(model,f'FULL__{mode}',torch.zeros(97),torch.from_numpy(core.make_shifts(1)),baseline).eval()
            result=wrapper(batch['x'],tf=batch['tf'])
            if baseline is None: baseline=wrapper.trace; expected=result
            torch.testing.assert_close(result['waveform'],expected['waveform'],rtol=1e-5,atol=1e-6)


@pytest.mark.parametrize('shape,groups', [((2,48,9,17),8),((2,96,31),12)])
def test_replay_changed_statistics_against_independent_float64_oracle(shape,groups):
    module=nn.GroupNorm(groups,shape[1]).eval()
    x=torch.randn(shape,generator=torch.Generator().manual_seed(2109))
    with torch.no_grad():
        module.weight.copy_(torch.linspace(-1.3,1.1,shape[1]))
        module.bias.copy_(torch.linspace(-.4,.3,shape[1]))
        native=module(x); natural=core.native_stats(module,x,native)
        selected=core.GNStats(natural.mean+.2,natural.rstd*.7)
        result=core.replay_norm(module,x,selected,natural,native)
        grouped=x.double().reshape(shape[0],groups,-1)
        oracle=((grouped-selected.mean.double()[:,:,None])*selected.rstd.double()[:,:,None]).reshape(shape)
        broad=(1,shape[1],*([1]*(len(shape)-2)))
        oracle=oracle*module.weight.double().reshape(broad)+module.bias.double().reshape(broad)
        torch.testing.assert_close(result.double(),oracle,rtol=1e-5,atol=1e-6)
        assert not torch.equal(result,native)
        copied=core.GNStats(natural.mean.clone(),natural.rstd.clone())
        assert torch.equal(core.replay_norm(module,x,copied,natural,native),native)


def test_native_rounding_residual_does_not_cross_bf16_boundary_on_replay():
    # 模拟两个正确 FP32 kernel 的一 ULP 差异，随后 BF16 量化会把误差放大。
    module=nn.GroupNorm(1,1).eval(); x=torch.tensor([[[-1.,1.]]])
    with torch.no_grad():
        module.bias.fill_(1.00390625-module(x)[0,0,1])
        native=module(x); natural=core.native_stats(module,x,native)
        direct=core.frozen_norm(module,x,natural,native.dtype)
        assert direct[0,0,1]==1.00390625
        native[0,0,1]=torch.nextafter(direct[0,0,1],torch.tensor(float('inf')))
        torch.testing.assert_close(direct,native,rtol=1e-5,atol=1e-6)
        assert not torch.equal(direct.bfloat16(),native.bfloat16())
        selected=core.GNStats(natural.mean.clone(),natural.rstd.clone())
        replay=core.replay_norm(module,x,selected,natural,native)
        assert torch.equal(replay,native) and torch.equal(replay.bfloat16(),native.bfloat16())


def test_full_failure_records_condition_and_keeps_gate(tmp_path,monkeypatch):
    model=fixture_model(); batch=runtime.prior.synthetic_batch(1,7)
    batch['meta']={'dataset_row_id':torch.tensor([1]),'samp_id':torch.tensor([9]),'split':['val']}
    original=runtime.collect_predictions; calls=0
    def faulty(*args,**kwargs):
        nonlocal calls
        calls+=1; result=original(*args,**kwargs)
        if calls==3: result['r_tho_hat']=result['r_tho_hat']+.004
        return result
    monkeypatch.setattr(runtime,'collect_predictions',faulty)
    with pytest.raises(AssertionError) as error:
        list(runtime.batch_conditions(model,batch,torch.zeros(97),torch.from_numpy(core.make_shifts(1)),
                                      'cpu',audit=runtime.condition_audit(tmp_path,seed=1)))
    records=[json.loads(line) for line in (tmp_path/'condition_diagnostics.jsonl').read_text().splitlines()]
    assert records[-1]['status']=='failed' and records[-1]['condition']=='FULL__GN1_FIXED'
    assert records[-1]['waveform_max_abs']>.0039 and records[-1]['waveform_mismatched']>0
    assert any('FULL__GN1_FIXED' in note for note in error.value.__notes__)
    assert not any(m._forward_hooks for m in model.modules())


def test_reuse_signals_requires_frozen_code_contract_and_exact_artifact(tmp_path,monkeypatch):
    monkeypatch.setattr(runtime,'ROOT',tmp_path); monkeypatch.setattr(runtime,'COUNT',2)
    core_path='resp_train/paper_evidence/e4_r3_norm.py'
    runtime_path='resp_train/paper_evidence/e4_r3_norm_runtime.py'
    sources={core_path:'TRANSFORMS=("FULL",)\ndef make_shifts(n): return n\n',
             runtime_path:'COUNT=2\ndef signals(): return 1\ndef validation_data(): return 2\n',
             'resp_train/paper_evidence/e4_r3_signals.py':'def signal_window(): return 3\n'}
    for rel,text in sources.items():
        path=tmp_path/rel; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(text)
    fields=('protocol','seeds','conditions','count','entries','validation_rows','plot_row_ids','cache_files',
            'dataset_index','frequencies','shifts','source_lock','reference','reference_manifest')
    old={key:[] for key in fields}; old['shifts']=[[60,61,62],[63,64,65]]
    old['code_files']={rel:runtime.identity(tmp_path/rel) for rel in sources}
    source=tmp_path/'old_lock.json'; runtime.write_json(source,old)
    old_digest=runtime.sha256_file(source)
    monkeypatch.setattr(runtime,'R1_LOCK',source); monkeypatch.setattr(runtime,'R1_SHA',old_digest)
    with runtime.attempt('signals',old_digest) as signal:
        runtime.write_json(signal/'environment.json',{'git':{'commit':'fixture','status_porcelain':''}})
        runtime.write_json(signal/'signals_receipt.json',dict(count=2,transforms=list(core.TRANSFORMS),
            offsets=old['shifts'],reference=old['reference'],model_inference=False))
        for name in ('associations.csv','paired_associations.csv','rows.csv'): (signal/name).write_text('fixture\n1\n')
    monkeypatch.setattr(runtime,'R1_SIGNALS',signal)
    monkeypatch.setattr(runtime.subprocess,'check_output',lambda command,**kwargs:sources[command[-1].split(':',1)[1]].encode())
    compatible=runtime.prepare_signal_reuse(old)
    lock=dict(old,compatible_signals=compatible); digest='c'*64
    runtime.verify_signals(signal,lock,digest)
    with pytest.raises(ValueError): runtime.verify_signals(signal,{},digest)
    (tmp_path/runtime_path).write_text(sources[runtime_path].replace('return 1','return 9'))
    with pytest.raises(ValueError,match='信号函数'): runtime.prepare_signal_reuse(old)
    (tmp_path/runtime_path).write_text(sources[runtime_path])
    with pytest.raises(ValueError,match='合同变化'): runtime.prepare_signal_reuse(dict(old,shifts=[]))
    (signal/'associations.csv').write_text('tampered\n')
    with pytest.raises(RuntimeError): runtime.verify_signals(signal,lock,digest)


def test_full_evaluation_runtime_native_metrics(tmp_path,monkeypatch):
    from resp_train.engine import collect_predictions
    from resp_train.metrics.task import evaluate_task_predictions
    monkeypatch.setattr(runtime,'ROOT',tmp_path); monkeypatch.setattr(runtime,'COUNT',2)
    digest='d'*64; environment={'fixture':'cpu'}
    monkeypatch.setattr(runtime.prior.source,'runtime_preflight',lambda device:environment)
    cfg=fixture_config(); model=fixture_model()
    batch=runtime.prior.synthetic_batch(2,1700)
    batch['meta']={'dataset_row_id':torch.tensor([1,2]),'samp_id':torch.tensor([7,8]),'split':['val','val']}
    rows=pd.DataFrame({'dataset_row_id':[1,2],'samp_id':[7,8],'split':['val','val']})
    pred=collect_predictions(model,[batch],device='cpu',max_windows=2,use_amp=False)
    evaluate_task_predictions(pred,cfg,include_test_only=False,method='W0_FULL').to_csv(tmp_path/'frozen.csv',index=False)
    np.save(tmp_path/'reference.npy',np.linspace(.02,.3,97,dtype=np.float32))
    reference={'path':str(tmp_path/'reference.npy'),**runtime.identity(tmp_path/'reference.npy')}
    checkpoint={'path':'synthetic','size_bytes':1,'sha256':'fixture'}
    entry={'seed':runtime.SEEDS[0],'selected_epoch':13,'files':{'metrics.csv':{'path':str(tmp_path/'frozen.csv')},'checkpoint_best_local_rr.pt':checkpoint}}
    lock={'entries':[entry],'reference':reference,'shifts':core.make_shifts(2).tolist(),'plot_row_ids':[1]}
    monkeypatch.setattr(runtime,'load_lock',lambda:(lock,digest))
    monkeypatch.setattr(runtime,'model_for',lambda entry,device:(model,cfg))
    def data(lock,cfg,out):
        rows.to_csv(out/'rows.csv',index=False)
        return SimpleNamespace(loader=[batch]),rows
    monkeypatch.setattr(runtime,'validation_data',data)
    with runtime.attempt('signals',digest) as signal:
        runtime.write_json(signal/'signals_receipt.json',dict(count=2,transforms=list(core.TRANSFORMS),offsets=lock['shifts'],reference=reference,model_inference=False))
        for name in ('rows.csv','associations.csv','paired_associations.csv'): rows.to_csv(signal/name,index=False)
    with runtime.attempt('gpu_smoke',digest) as gpu:
        runtime.write_json(gpu/'environment.json',environment)
        runtime.write_json(gpu/'gpu_smoke.json',dict(passed=True,records=[dict(seed=s,batch_size=b,conditions=list(core.CONDITIONS),
            full_replay_passed=True,peak_reserved_fraction=.1) for s in runtime.SEEDS for b in (1,128)]))
    output=runtime.evaluate(runtime.SEEDS[0],'cpu',signal,gpu)
    runtime.verify_attempt(output,digest,'evaluation')
    paired=pd.read_csv(output/'paired_changes.csv')
    assert len(paired)==22*3*5
    assert len(pd.read_csv(output/'train_mean_factorial.csv'))==3*5
    for mode in core.MODES:
        assert paired[paired.condition.eq(f'FULL__{mode}')].degradation.abs().max()<1e-5
    assert len(pd.read_csv(output/'layer_changes.csv'))==44
    assert len(list((output/'examples'/'row_1').glob('*.npz')))==22


def test_complete_summary_and_missing_seed(tmp_path,monkeypatch):
    monkeypatch.setattr(runtime,'ROOT',tmp_path)
    digest='e'*64
    entries=[dict(seed=s,selected_epoch=e,files={'checkpoint_best_local_rr.pt':{'path':str(s),'sha256':str(s),'size_bytes':1}}) for s,e in zip(runtime.SEEDS,(13,15,14))]
    monkeypatch.setattr(runtime,'load_lock',lambda:({'entries':entries},digest))
    with pytest.raises(ValueError): runtime.summarize()
    with runtime.attempt('signals',digest) as signal:
        runtime.write_json(signal/'signals_receipt.json',{'fixture':True})
        for name in ('associations.csv','paired_associations.csv','rows.csv'): pd.DataFrame({'fixture':[1]}).to_csv(signal/name,index=False)
    signal_info={'path':str(signal),'manifest':runtime.identity(signal/'manifest.json')}
    for seed in runtime.SEEDS:
        entry=next(e for e in entries if e['seed']==seed)
        with runtime.attempt('evaluation',digest,seed) as out:
            runtime.write_json(out/'evaluation_receipt.json',dict(seed=seed,count=runtime.COUNT,conditions=list(core.CONDITIONS),full_reproduced=True,
                signals=signal_info,selected_epoch=entry['selected_epoch'],checkpoint=entry['files']['checkpoint_best_local_rr.pt']))
            records=[]
            for condition in core.CONDITIONS:
                for group in ['ALL',*map(str,range(7))]:
                    for metric in runtime.PRIMARY:
                        records.append(dict(seed=seed,condition=condition,scope='pooled' if group=='ALL' else 'samp_id',group=group,metric=metric,
                            degradation=0. if condition.startswith('FULL__') else .1))
            pd.DataFrame(records).to_csv(out/'paired_changes.csv',index=False)
            layer=pd.DataFrame([dict(condition=c,dataset_row_id=1,samp_id=7,split='val',z_relative_delta=.1) for c in core.CONDITIONS])
            layer.to_csv(out/'layer_changes.csv',index=False)
            pd.DataFrame([dict(condition=c,dataset_row_id=1,samp_id=7,split='val',layer='norm',group=0,
                mean=1.,rstd=2.,full_mean=0.,full_rstd=1.,used_mean=0.,used_rstd=1.,
                normalized_mean_shift=1.,log_variance_plus_eps_ratio=-1.386) for c in core.CONDITIONS]).to_csv(out/'gn_statistics.csv',index=False)
            pd.DataFrame({'dataset_row_id':[1],'samp_id':[7],'split':['val']}).to_csv(out/'rows.csv',index=False)
    result=runtime.summarize()
    assert len(pd.read_csv(result/'paired_seed_subject_changes.csv'))==3*22*8*5
    assert len(pd.read_csv(result/'train_mean_factorial.csv'))==3*8*5
    assert len(pd.read_csv(result/'three_seed_direction.csv'))==22*8*5
    diagnostics=pd.read_csv(result/'gn_statistics_summary.csv')
    assert diagnostics.used_mean_shift_mean.eq(0).all()
    assert diagnostics.mean_shift_mean.eq(1).all()


def test_signal_phase_complete_with_disposable_inputs(tmp_path,monkeypatch):
    from omegaconf import OmegaConf
    monkeypatch.setattr(runtime,'ROOT',tmp_path); monkeypatch.setattr(runtime,'COUNT',1)
    monkeypatch.setattr(runtime,'git_state',lambda *args,**kwargs:{'commit':'fixture','status_porcelain':''})
    monkeypatch.setattr(runtime.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*2**30))
    cfg=fixture_config(); batch=runtime.prior.synthetic_batch(1,19)
    batch['meta']={'dataset_row_id':torch.tensor([1]),'samp_id':torch.tensor([7]),'split':['val']}
    rows=pd.DataFrame({'dataset_row_id':[1],'samp_id':[7],'split':['val']})
    np.save(tmp_path/'reference.npy',np.linspace(.02,.3,97,dtype=np.float32))
    lock={'entries':[{'config':OmegaConf.to_container(cfg,resolve=True)}],
          'reference':{'path':str(tmp_path/'reference.npy'),**runtime.identity(tmp_path/'reference.npy')},
          'shifts':core.make_shifts(1).tolist(),'plot_row_ids':[]}
    monkeypatch.setattr(runtime,'load_lock',lambda:(lock,'a'*64))
    def data(lock,cfg,out):
        rows.to_csv(out/'rows.csv',index=False)
        return SimpleNamespace(loader=[batch]),rows
    monkeypatch.setattr(runtime,'validation_data',data)
    out=runtime.signals(); runtime.verify_attempt(out,'a'*64,'signals')
    associations=pd.read_csv(out/'associations.csv')
    assert set(associations['transform'])==set(core.TRANSFORMS)
    assert associations.representation.nunique()==25
    profile=associations[associations.support.str.startswith('common_lag')]
    assert profile.lag_sec.min()==-.3 and profile.lag_sec.max()==.3 and profile.lag_sec.nunique()==61
    assert set(profile.association)=={'cycle'}
    assert (out/'paired_association_subject_macro.csv').is_file()
