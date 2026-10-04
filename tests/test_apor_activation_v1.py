from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from scripts import apor_activation_v1_runtime as exp
from scripts.apor_activation_v1_model import ARMS, SEEDS, ActivationModel, model_contract
from resp_train.paper_evidence.patch_apor_v1_model import PatchAporModel
from resp_train.losses.task import RespirationTaskLoss
from resp_train.crd.training import build_crd_optimizer


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    torch.set_num_threads(previous)


def replace_trunk(model):
    model.base.local_blocks = nn.ModuleList([nn.Identity() for _ in model.base.local_blocks])
    return model


@pytest.mark.parametrize("seed", SEEDS)
def test_exact_four_activation_changes_and_common_initialization(seed):
    original = PatchAporModel("A0", seed)
    u0, u1 = ActivationModel("U0", seed), ActivationModel("U1", seed)
    for model in (u0, u1):
        assert sum(p.numel() for p in model.parameters()) == 1077640
        assert set(original.state_dict()) == set(model.state_dict())
        for name, value in original.state_dict().items():
            assert torch.equal(value, model.state_dict()[name])
    changed = []
    for name, module in u0.named_modules():
        other = dict(u1.named_modules())[name]
        if type(module) is not type(other):
            assert isinstance(module, nn.GELU) and isinstance(other, nn.SiLU)
            changed.append(name)
    assert len(changed) == 4 and all(name.startswith('base.frontend.encoder.blocks.') for name in changed)
    for model in (original, u0, u1):
        replace_trunk(model).eval()
    batch = exp.synthetic_batch(1, 31)
    with torch.no_grad():
        original.branches['w'].final_projection.weight.normal_(std=.01)
        u0.load_state_dict(original.state_dict());u1.load_state_dict(original.state_dict())
        expected = original(batch['x'], tf=batch['tf'])['waveform']
        torch.testing.assert_close(expected, u0(batch['x'], tf=batch['tf'])['waveform'], rtol=0, atol=0)
        assert not torch.equal(expected, u1(batch['x'], tf=batch['tf'])['waveform'])


def test_contract_and_three_seed_partition():
    exp.load_spec()
    assert len(exp.plan()) == 3 and len(exp.scientific_plan()) == 6
    layout = exp.parallel_layout(['cuda:0', 'cuda:1'])
    assert [len(w['cells']) for w in layout['workers']] == [2, 1]
    ids = {(c['arm'], c['seed']) for w in layout['workers'] for c in w['cells']}
    assert ids == {('U1', s) for s in SEEDS}
    for arm in ARMS:
        cfg = exp.config(arm, SEEDS[0], Path('SYNTHETIC'))
        old = exp.legacy.config('A0', SEEDS[0], Path('OLD'))
        assert exp.scientific_config(cfg) == exp.scientific_config(old)


def test_u1_gradient_finite_and_roundtrip():
    cfg = exp.config('U1', SEEDS[0], Path('SYNTHETIC'))
    model = replace_trunk(ActivationModel('U1', SEEDS[0]))
    optimizer, partition = build_crd_optimizer(model, cfg)
    assert set(partition.decay_names) | set(partition.no_decay_names) == set(dict(model.named_parameters()))
    batch = exp.synthetic_batch(1, 22)
    loss_fn = RespirationTaskLoss(cfg)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = loss_fn(model(batch['x'], tf=batch['tf']), batch['target'])
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    clone = replace_trunk(ActivationModel('U1', SEEDS[0])).eval()
    clone.load_state_dict(model.state_dict());model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(batch['x'], tf=batch['tf'])['waveform'], clone(batch['x'], tf=batch['tf'])['waveform'], rtol=0, atol=0)


def metric_fixture():
    rows = []
    for arm in ARMS:
        for seed in SEEDS:
            for index in range(4):
                row = {'arm': arm, 'seed': seed, 'dataset_row_id': index, 'samp_id': 1 if index < 3 else 2, 'split': 'val',
                       'whole_rr_target_eligible': True, 'local_rr_target_eligible': True, 'joint_target_eligible': True,
                       'joint_prediction_degenerate': False, 'envelope_spearman_prediction_degenerate': False}
                row.update({m: .8 if m == exp.sf.PCC else 1. for m in exp.sf.PRIMARY})
                rows.append(row)
    return pd.DataFrame(rows)


def test_quality_preserving_decision_checks_both_aggregation_scopes(monkeypatch):
    monkeypatch.setitem(exp.COUNTS, 'val', 4)
    frame = metric_fixture()
    tables = exp.summary_tables(frame)
    assert len(tables['per_seed']) == 30 and len(tables['paired_delta']) == 15
    assert exp.validation_decision(tables)['candidate'] == 'U1'
    frame.loc[frame.arm.eq('U1'), 'local_rr_mae_bpm'] = 1.004
    assert exp.validation_decision(exp.summary_tables(frame))['candidate'] == 'U1'
    frame.loc[frame.arm.eq('U1'), 'local_rr_mae_bpm'] = 1.006
    assert exp.validation_decision(exp.summary_tables(frame))['candidate'] == 'U0'
    frame.loc[frame.arm.eq('U1') & frame.samp_id.eq(1), 'local_rr_mae_bpm'] = .97
    frame.loc[frame.arm.eq('U1') & frame.samp_id.eq(2), 'local_rr_mae_bpm'] = 1.06
    assert exp.validation_decision(exp.summary_tables(frame))['candidate'] == 'U0'
    with pytest.raises(ValueError, match='6-cell'):
        exp.summary_tables(frame[frame.arm.eq('U1')])


def test_native_trainer_lifecycle(tmp_path, monkeypatch):
    cfg = exp.config('U1', SEEDS[0], tmp_path / 'training')
    cfg.training.epochs=2;cfg.training.early_stopping_min_epoch=1;cfg.training.batch_size=2;cfg.training.use_amp=False

    class Loader(list):
        dataset=range(2)

    builder = exp.build_model
    monkeypatch.setattr(exp, 'build_model', lambda cfg: replace_trunk(builder(cfg)))

    class SyntheticExperiment(exp.ModuleExperiment):
        def _build_data(self):
            train=exp.synthetic_batch(2,241,split='train');val=exp.synthetic_batch(2,242,split='val',row_offset=100)
            return SimpleNamespace(train=SimpleNamespace(loader=Loader([train])),val=SimpleNamespace(loader=Loader([val])),
                                   audit_summary=pd.DataFrame({'synthetic':[True]}))

    initial=tmp_path/'initialization.json';run=SyntheticExperiment(cfg,initialization_path=initial).train()
    frame=pd.read_csv(run/'metrics.csv');assert len(frame)==2 and frame.arm.eq('U1').all()
    assert np.load(run/'validation_prediction.npy').shape==(2,1,18000)
    monkeypatch.setattr(exp,'UPDATES_PER_EPOCH',1);monkeypatch.setattr(exp,'PLANNED_UPDATES',2)
    monkeypatch.setattr(exp.sf,'UPDATES_PER_EPOCH',1);monkeypatch.setattr(exp.sf,'PLANNED_UPDATES',2)
    monkeypatch.setattr(exp.sf,'EPOCHS',2);monkeypatch.setattr(exp.sf,'EARLY_STOP_MIN_EPOCH',1)
    assert exp.validate_run(run,cfg,frame[['dataset_row_id','samp_id','split']],initial)['validation_rows']==2


@pytest.mark.parametrize('exit_codes',[(0,0),(0,1)])
def test_parallel_acceptance_barrier_and_summary(tmp_path,monkeypatch,exit_codes):
    exp.write_json(tmp_path/'session.json',{'fixture':True})
    monkeypatch.setattr(exp,'load_session',lambda p:{})
    monkeypatch.setattr(exp,'cuda_environment',lambda d:{'device_uuid':d})
    monkeypatch.setattr(torch.cuda,'set_device',lambda d:None);monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    accepted=[];calls=[];summaries=[]
    monkeypatch.setattr(exp,'run_gpu',lambda s,d,r:accepted.append(d) or tmp_path/d)
    monkeypatch.setattr(exp,'verify_gpu',lambda *a:None)

    class Process:
        def __init__(self,cmd,**kwargs):
            assert accepted==['cuda:0','cuda:1'];self.returncode=exit_codes[len(calls)];self.pid=23400+len(calls);calls.append(cmd)
        def poll(self):return self.returncode
        def wait(self,**kwargs):return self.returncode

    monkeypatch.setattr(exp.subprocess,'Popen',Process)
    monkeypatch.setattr(exp,'summarize',lambda *a:summaries.append(a) or tmp_path/'summary')
    if exit_codes==(0,0):
        exp.run_parallel(tmp_path,['cuda:0','cuda:1']);assert len(summaries)==1
    else:
        with pytest.raises(RuntimeError):exp.run_parallel(tmp_path,['cuda:0','cuda:1'])
        assert not summaries


@pytest.mark.parametrize('training_fails',[False,True])
def test_pipeline_freezes_validation_before_test_and_stops_on_failure(tmp_path,monkeypatch,training_fails):
    from scripts import run_apor_activation_v1 as cli
    from scripts import eval_apor_activation_v1_research_test as rt
    calls=[]

    def train(*args):
        calls.append('validation')
        if training_fails:raise RuntimeError('fixture training failed')
        return tmp_path/'validation'

    def prepare(session):
        assert calls==['validation'];calls.append('allowlist');return tmp_path/'allowlist'

    def test(*args):
        assert calls==['validation','allowlist'];calls.append('test');return tmp_path/'test_summary'

    monkeypatch.setattr(cli.experiment,'run_parallel',train)
    monkeypatch.setattr(rt,'prepare_allowlist',prepare)
    monkeypatch.setattr(rt,'parallel',test)
    if training_fails:
        with pytest.raises(RuntimeError):cli.run_pipeline(tmp_path,['cuda:0','cuda:1'])
        assert calls==['validation'] and not (tmp_path/'pipeline_result.json').exists()
    else:
        cli.run_pipeline(tmp_path,['cuda:0','cuda:1'])
        assert calls==['validation','allowlist','test'] and (tmp_path/'pipeline_result.json').exists()
