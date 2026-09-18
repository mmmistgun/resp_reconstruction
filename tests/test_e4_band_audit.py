"""仅用 disposable CPU fixture 检验干预定义、固定特征和来源生命周期。"""
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
from resp_train.paper_evidence.e4_aggregation_v2_model import AggregationCwtBranch, cpu_module_seed
from resp_train.paper_evidence import e4_band_audit as core
from resp_train.paper_evidence import e4_band_audit_runtime as runtime
from resp_train.paper_evidence.e4_band_audit_store import FeatureStore, feature_reader


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    previous=torch.get_num_threads(); torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def frequencies():
    p=Path(__file__).resolve().parents[1]/"docs/experiments/e4_w0_scale_aggregation_source_audit_20260917.json"
    return np.asarray(json.loads(p.read_text())["frequency"]["values_hz"],dtype=np.float64)


def branch_for(arm,frequencies):
    with cpu_module_seed(7,"tf_branch_w"):
        branch=CwtBranch() if arm=="W0_FULL" else AggregationCwtBranch(arm,7,frequencies)
    with torch.no_grad():
        branch.final_projection.weight.fill_(.01)
        if hasattr(branch,"aggregation"):
            module=branch.aggregation
            if hasattr(module,"logits"):
                module.logits.copy_(torch.linspace(-.7,.5,module.logits.numel()).reshape_as(module.logits))
            else:
                module.score.weight.fill_(.3)
    return branch


class FixtureModel(nn.Module):
    def __init__(self,branch):
        super().__init__(); self.branches=nn.ModuleDict({"w":branch})

    def forward(self,x,*,tf):
        gamma,beta=self.branches["w"](tf)
        return {"waveform":x+F.interpolate((gamma+beta).mean(1,keepdim=True),size=x.shape[-1],mode="linear",align_corners=False)}


@pytest.mark.parametrize("region",range(4))
@pytest.mark.parametrize("channels",[1,96])
def test_reset_preserves_prior_inside_and_ratios_outside(region,channels):
    generator=torch.Generator().manual_seed(10)
    alpha=torch.softmax(torch.randn(2,channels,97,3,generator=generator),dim=2)
    original=alpha.clone(); changed=core.reset_band(alpha,region)
    start,stop=core.REGIONS[region]
    torch.testing.assert_close(changed[:,:,start:stop],torch.full_like(changed[:,:,start:stop],1/97),rtol=0,atol=0)
    mask=torch.ones(97,dtype=torch.bool); mask[start:stop]=False
    ratio=changed[:,:,mask]/alpha[:,:,mask]
    torch.testing.assert_close(ratio,ratio[:,:,:1].expand_as(ratio),rtol=2e-6,atol=1e-7)
    assert torch.equal(alpha,original)


@pytest.mark.parametrize("reference",["MEAN","ZERO"])
@pytest.mark.parametrize("region",range(4))
def test_input_replacement_is_local_and_source_immutable(reference,region):
    x=torch.arange(97*360).reshape(1,97,360).float(); original=x.clone()
    baseline=torch.arange(97).float()
    actual=core.replace_input(x,f"INPUT_{reference}_R{region}",baseline)
    start,stop=core.REGIONS[region]
    expected=original.clone()
    expected[:,start:stop]=baseline[None,start:stop,None] if reference=="MEAN" else 0
    assert torch.equal(actual,expected) and torch.equal(x,original)


@pytest.mark.parametrize("arm",runtime.ARMS)
def test_wrapper_full_matches_native_and_reuses_identical_features(arm,frequencies):
    branch=branch_for(arm,frequencies); model=FixtureModel(branch).eval()
    generator=torch.Generator().manual_seed(17)
    x=torch.randn(1,1,1800,generator=generator)
    tf={"w":torch.rand(1,97,360,generator=generator)}
    captured={}
    with torch.no_grad():
        expected=model(x,tf=tf)
        core.AuditModel(model,"FULL",observe=lambda offset,data:captured.update(data)).eval()(x,tf=tf)
        replay=core.AuditModel(model,"FULL").eval()(x,tf=tf)
        assert torch.equal(expected["waveform"],replay["waveform"])
        saved=captured['x'].clone()
        # 更换入口不会改变缓存 X；前卷积若被错误调用，直接失败。
        def forbidden(*args):
            raise AssertionError("聚合干预重新计算了前卷积")
        handle=branch.conv_in.register_forward_pre_hook(forbidden)
        try:
            for condition in core.CONDITIONS[1:6]:
                wrapper=core.AuditModel(model,condition,fixed_features=lambda offset,count:saved).eval()
                changed=wrapper(x,tf={"w":tf["w"]+100})
                assert wrapper.offset==1 and torch.equal(saved,captured['x'])
                if arm=="W0_FULL":
                    assert torch.equal(changed['waveform'],expected['waveform'])
        finally:
            handle.remove()
    assert branch.forward.__self__ is branch


def test_region_expansion_matches_unequal_count_prior(frequencies):
    branch=branch_for('channel_region',frequencies)
    with torch.no_grad():
        branch.aggregation.logits.zero_()
        a,d=core.scale_weights(branch,torch.ones(1,96,97,360))
    torch.testing.assert_close(a,torch.full_like(a,1/97),atol=1e-8,rtol=1e-6)
    assert torch.count_nonzero(d)==0
    for start,stop in core.REGIONS:
        torch.testing.assert_close(a[:,:,start:stop].sum(2),torch.full_like(a.sum(2),(stop-start)/97),rtol=1e-6,atol=1e-7)


@pytest.mark.parametrize("bad",["nan","negative","zero_complement"])
def test_bad_weights_fail(bad):
    a=torch.ones(1,1,97,1)/97
    if bad=='nan': a[:,:,0]=float('nan')
    elif bad=='negative': a[:,:,0]=-1
    else: a.zero_(); a[:,:,:25]=1/25
    with pytest.raises((ValueError,FloatingPointError)):
        core.reset_band(a,0)


@pytest.mark.parametrize("dtype",[torch.float32,torch.bfloat16])
def test_full_feature_storage_roundtrip_and_zero_denominator(tmp_path,dtype,frequencies):
    rows=pd.DataFrame({'dataset_row_id':[1,2],'samp_id':[7,8],'split':['val','val']})
    store=FeatureStore(tmp_path,rows,[],frequencies)
    x=torch.zeros(1,96,97,360,dtype=dtype)
    alpha=torch.full((1,1,97,1),1/97)
    for i in range(2):
        store(i,dict(x=x,z=x.mean(2),alpha=alpha,delta=torch.zeros_like(alpha)))
    store.finish()
    read=feature_reader(tmp_path)
    assert torch.equal(read(0,2),x.expand(2,-1,-1,-1)) and read(0,1).dtype==dtype
    stats=pd.read_csv(tmp_path/'feature_statistics.csv')
    assert stats.delta_to_mean.isna().all() and stats.cancellation_ratio.isna().all()
    assert stats.mean_is_zero.all() and not stats.cancellation_defined.any()


def metrics_fixture():
    frame=pd.DataFrame({'dataset_row_id':[1,2,3],'samp_id':[7,7,8],'split':['val']*3})
    for key in ('whole_rr_target_eligible','local_rr_target_eligible','local_rr_target_eligible_windows',
                'joint_target_eligible','envelope_spearman_target_eligible'):
        frame[key]=True
    for metric in core.PRIMARY:
        frame[metric]=[1.,2.,3.]
    return frame


def test_paired_subject_direction_and_pcc_sign():
    full=metrics_fixture(); changed=full.copy()
    for metric in core.PRIMARY: changed[metric]+=1
    result=core.paired_changes(full,changed,'fixture',7,'UNIFORM')
    assert len(result)==15
    assert result.loc[result.metric.eq(core.PCC),'degradation'].eq(-1).all()
    assert result.loc[~result.metric.eq(core.PCC),'degradation'].eq(1).all()
    assert result[result.scope.eq('samp_id')].group.unique().tolist()==['7','8']
    changed.loc[0,'dataset_row_id']=100
    with pytest.raises(ValueError): core.paired_changes(full,changed,'fixture',7,'UNIFORM')


def test_attempt_failure_completion_and_tamper(tmp_path,monkeypatch):
    monkeypatch.setattr(runtime,'ROOT',tmp_path)
    digest='b'*64
    with pytest.raises(RuntimeError):
        with runtime.attempt('fixture',digest) as failed:
            raise RuntimeError('fixture failure')
    assert (failed/'lifecycle_failed.json').exists()
    with runtime.attempt('fixture',digest) as good:
        (good/'payload').write_text('a')
    runtime.verify_attempt(good,digest,'fixture')
    with pytest.raises(FileExistsError):
        with runtime.attempt('fixture',digest): pass
    (good/'payload').write_text('b')
    with pytest.raises(RuntimeError): runtime.verify_attempt(good,digest,'fixture')


def test_summary_requires_full_matrix_and_preserves_reference(tmp_path,monkeypatch):
    monkeypatch.setattr(runtime,'ROOT',tmp_path)
    monkeypatch.setattr(runtime,'COUNT',3)
    digest='c'*64; lock={}
    monkeypatch.setattr(runtime,'load_lock',lambda:(lock,digest))
    with pytest.raises(ValueError): runtime.summarize()
    for arm in runtime.ARMS:
        for seed in runtime.SEEDS:
            with runtime.attempt('evaluation',digest,arm,seed) as out:
                records=[]
                for condition in core.CONDITIONS:
                    for group in ['ALL',*map(str,range(7))]:
                        for metric in core.PRIMARY:
                            records.append(dict(arm=arm,seed=seed,condition=condition,scope='pooled' if group=='ALL' else 'samp_id',
                                group=group,metric=metric,n=3,full=1.,intervened=1.1,degradation=.1,relative_degradation_pct=10.))
                pd.DataFrame(records).to_csv(out/'paired_changes.csv',index=False)
                runtime.write_json(out/'evaluation_receipt.json',dict(arm=arm,seed=seed,conditions=list(core.CONDITIONS),
                    count=runtime.COUNT,full_reproduced=True,reference_manifest={'sha256':'reference','size_bytes':1}))
                rows=pd.DataFrame(dict(dataset_row_id=[1,2,3],samp_id=[7,8,9],split=['val']*3))
                rows.to_csv(out/'rows.csv',index=False)
                (out/'FULL').mkdir()
                rows.assign(delta_to_mean=[0.,.1,.2]).to_csv(out/'FULL'/'feature_statistics.csv',index=False)
    output=runtime.summarize()
    assert len(pd.read_csv(output/'paired_seed_subject_changes.csv'))==15*14*8*5
    assert len(pd.read_csv(output/'three_seed_direction.csv'))==5*14*8*5
    assert len(pd.read_csv(output/'subject_macro_by_seed.csv'))==15*14*5
    assert len(pd.read_csv(output/'feature_statistics_by_window.csv'))==45
    assert len(pd.read_csv(output/'feature_statistics_by_seed_subject.csv'))==60


def test_train_reference_matches_full_mean_and_rejects_nonfinite():
    values=np.random.default_rng(7).normal(size=(65,97,360)).astype(np.float32)
    expected=values.astype(np.float64).mean(axis=(0,2)).astype(np.float32)
    np.testing.assert_array_equal(core.training_reference(values),expected)
    values[64,0,0]=np.nan
    with pytest.raises(FloatingPointError): core.training_reference(values)


def test_observation_detects_cancellation_and_plots_channels(tmp_path,frequencies):
    rows=pd.DataFrame({'dataset_row_id':[1],'samp_id':[7],'split':['val']})
    store=FeatureStore(tmp_path,rows,[1],frequencies)
    x=torch.ones(1,96,97,360)
    delta=torch.zeros(1,96,97,1); delta[:,:,0]=.002; delta[:,:,1]=-.002
    alpha=torch.full_like(delta,1/97)+delta
    store(0,dict(x=x,z=x.mean(2),alpha=alpha,delta=delta)); store.finish()
    stats=pd.read_csv(tmp_path/'feature_statistics.csv').iloc[0]
    assert stats.alpha_tv_mean>0 and stats.delta_to_mean==0 and stats.cancellation_ratio==0
    assert (tmp_path/'row_1.png').is_file() and (tmp_path/'region_channels.png').is_file()


def test_full_runtime_with_synthetic_native_metrics(tmp_path,monkeypatch,frequencies):
    from resp_train.crd.config import load_crd_config
    from resp_train.engine import collect_predictions
    from resp_train.metrics.task import evaluate_task_predictions
    monkeypatch.setattr(runtime,'ROOT',tmp_path)
    monkeypatch.setattr(runtime,'COUNT',2)
    monkeypatch.setattr(runtime.shutil,'disk_usage',lambda path:SimpleNamespace(free=100*2**30))
    digest='d'*64; environment={'fixture':'cpu'}
    monkeypatch.setattr(runtime.source,'runtime_preflight',lambda device:environment)
    cfg=load_crd_config(Path(__file__).resolve().parents[1]/'configs/crd_tf_v1/crd_tf102_w_formal.yaml')
    model=FixtureModel(branch_for('W0_FULL',frequencies)).eval()
    batch=runtime.synthetic_batch(2,1700)
    batch['meta']={'dataset_row_id':torch.tensor([1,2]),'samp_id':torch.tensor([7,8]),'split':['val','val']}
    rows=pd.DataFrame({'dataset_row_id':[1,2],'samp_id':[7,8],'split':['val','val']})
    predictions=collect_predictions(model,[batch],device='cpu',max_windows=2,use_amp=False)
    metrics=evaluate_task_predictions(predictions,cfg,include_test_only=False,method='fixture')
    metrics.to_csv(tmp_path/'frozen.csv',index=False)
    (tmp_path/'cache').write_bytes(b'fixture')
    item={'path':str(tmp_path/'cache'),**runtime.identity(tmp_path/'cache')}
    entry={'arm':'W0_FULL','seed':runtime.SEEDS[0],'selected_epoch':13,
           'files':{'checkpoint_best_local_rr.pt':item,'metrics.csv':{'path':str(tmp_path/'frozen.csv')}}}
    lock={'entries':[entry],'cache_files':{k:item for k in ('val_w','val_rows','manifest','frequency_file')},
          'dataset_index':item,'validation_rows':rows.to_dict('records'),'plot_row_ids':[],
          'frequencies':frequencies.tolist()}
    monkeypatch.setattr(runtime,'load_lock',lambda:(lock,digest))
    monkeypatch.setattr(runtime,'load_model',lambda entry,device:(model,cfg))
    monkeypatch.setattr(runtime,'build_window_data',lambda *args,**kwargs:SimpleNamespace(rows=rows,dataset=[1,2],loader=[batch]))
    with runtime.attempt('reference',digest) as reference:
        np.save(reference/'reference.npy',np.linspace(-.1,.1,97,dtype=np.float32))
    with runtime.attempt('gpu_smoke',digest) as smoke:
        runtime.write_json(smoke/'environment.json',environment)
        slots={(a,s,1) for a in runtime.ARMS for s in runtime.SEEDS}|{(a,runtime.SEEDS[0],128) for a in runtime.ARMS}
        runtime.write_json(smoke/'gpu_smoke.json',{'passed':True,'records':[dict(arm=a,seed=s,batch_size=b,
            full_replay_passed=True,cached_full_replay_passed=True,conditions=list(core.CONDITIONS),
            peak_reserved_fraction=.1) for a,s,b in sorted(slots)]})
    output=runtime.evaluate('W0_FULL',runtime.SEEDS[0],'cpu',reference,smoke)
    runtime.verify_attempt(output,digest,'evaluation')
    assert len(pd.read_csv(output/'paired_changes.csv'))==14*3*5
    for condition in core.CONDITIONS[1:6]:
        pd.testing.assert_frame_equal(pd.read_csv(output/'FULL'/'metrics.csv'),pd.read_csv(output/condition/'metrics.csv'))
    assert feature_reader(output/'FULL')(0,2).shape==(2,96,97,360)


@pytest.mark.parametrize('arm',['channel_region','frequency_attention'])
def test_bf16_wrapper_replay_and_uniform_exact_mean(arm,frequencies):
    model=FixtureModel(branch_for(arm,frequencies)).eval()
    x=torch.zeros(1,1,1800); tf={'w':torch.ones(1,97,360)}; captured={}
    with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
        native=model(x,tf=tf)
        wrapper=core.AuditModel(model,'FULL',observe=lambda offset,data:captured.update(data)).eval()
        actual=wrapper(x,tf=tf)
        assert torch.equal(actual['waveform'],native['waveform'])
        assert captured['x'].dtype==torch.bfloat16
        uniform=core.aggregate(model.branches['w'],captured['x'],'UNIFORM')
        assert torch.equal(uniform,captured['x'].mean(2))


def test_wrapper_restores_forward_after_downstream_exception(frequencies):
    model=FixtureModel(branch_for('W0_FULL',frequencies)).eval()
    branch=model.branches['w']; original=branch.forward
    def fail(*args):
        raise RuntimeError('downstream fixture')
    hook=branch.temporal.register_forward_pre_hook(fail)
    with torch.no_grad(),pytest.raises(RuntimeError,match='downstream fixture'):
        core.AuditModel(model,'FULL').eval()(torch.zeros(1,1,1800),tf={'w':torch.zeros(1,97,360)})
    hook.remove()
    assert branch.forward==original
