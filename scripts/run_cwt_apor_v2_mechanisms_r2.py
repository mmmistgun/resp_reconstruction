"""同batch逐窗FULL验收；复核已有机制产物并完成剩余seed。"""
from contextlib import contextmanager
from pathlib import Path
import argparse
import signal
import sys
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.recover_cwt_apor_v2_mechanisms import prediction_keys
from resp_train.engine.train import _extract_meta, _prediction_dict_from_arrays
from resp_train.metrics.task import evaluate_task_predictions
from resp_train.paper_evidence import w0_structural_factorial_v1 as sf
from resp_train.paper_evidence.cwt_apor_v2 import artifacts as io, interventions as iv, research_test as test
from resp_train.paper_evidence.cwt_apor_v2.model import build_model
from resp_train.paper_evidence.cwt_apor_v2.spec import SEEDS, OUTPUT_ROOT

NOTE = ROOT / 'docs/experiments/cwt_apor_v2_mechanisms_r2_20261002.md'
TEST = ROOT / 'tests/test_cwt_apor_v2_mechanisms_r2.py'
RTOL, ATOL = 1e-5, 1e-6
LEGACY = {
    20260811: 'attempt_20261002T061007Z_0d99fce0f3ac',
    20260812: 'attempt_20261002T062624Z_a2c4728c941a',
}
PRIOR_AUDIT = 'research_test/mechanisms_r2/revision_c69a010bd938/seed_20260811/attempt_20261002T082234Z_a3a0196ab113'


def wave_check(plain, full, row_ids):
    """逐点硬检查，同时记录每个窗口的差异；不以均值抵消局部错误。"""
    if plain.shape != full.shape or len(plain) != len(row_ids):
        raise ValueError('FULL波形shape或窗口身份不符')
    if not torch.isfinite(plain).all() or not torch.isfinite(full).all():
        raise FloatingPointError('FULL/普通前向含非有限输出')
    torch.testing.assert_close(full, plain, rtol=RTOL, atol=ATOL)
    difference = (full.float()-plain.float()).abs().flatten(1)
    return [{'dataset_row_id': int(row), 'max_abs': float(d.max()), 'mean_abs': float(d.mean())}
            for row, d in zip(row_ids, difference)]


def cross_batch_report(frame, source):
    """样本/资格仍为硬检查；跨batch的指标差值完整报告。"""
    current = frame[frame.condition.eq('FULL__NAT')].reset_index(drop=True)
    if not current[['dataset_row_id','samp_id','split']].equals(source[['dataset_row_id','samp_id','split']].reset_index(drop=True)):
        raise ValueError('跨batch FULL样本身份或顺序不一致')
    sf.validate_metrics(current, source[['dataset_row_id','samp_id','split']])
    records = []
    for metric in sf.PRIMARY:
        mask = sf._metric_mask(source, metric)
        if not np.array_equal(mask, sf._metric_mask(current, metric)):
            raise ValueError('FULL重放目标资格发生变化')
        a, b = current.loc[mask,metric].to_numpy(), source.loc[mask,metric].to_numpy()
        delta = a-b
        records.append({'metric': metric, 'n': len(a), 'reference_mean': b.mean(), 'full_mean': a.mean(),
                        'mean_delta': delta.mean(), 'window_max_abs': abs(delta).max(),
                        'previous_mean_gate_passed': bool(np.isclose(a.mean(),b.mean(),rtol=1e-3,atol=0))})
    return pd.DataFrame(records)


@contextmanager
def checked_pairing(row_ids, audit):
    """新seed在原18条件流程内同步比较同batch普通前向和FULL。"""
    original, offset = iv.paired_batch, 0
    def checked(model, x, w, frequencies, shifts):
        nonlocal offset
        seen = False
        for item in original(model, x, w, frequencies, shifts):
            if item[0] == 'FULL__NAT':
                if seen:
                    raise ValueError('重复FULL条件')
                # 保持原机制路径为首次前向，再追加同batch普通前向；避免改变首次数值执行路径。
                with torch.inference_mode(), torch.autocast(x.device.type, enabled=x.is_cuda, dtype=torch.bfloat16):
                    plain = model(x, tf={'w':w})['waveform']
                audit.extend(wave_check(plain, item[1], row_ids[offset:offset+len(x)]))
                seen = True
            yield item
        if not seen:
            raise ValueError('缺少FULL条件')
        offset += len(x)
    iv.paired_batch = checked
    try:
        yield
        if offset != len(row_ids):
            raise ValueError('同batch验收窗口不完整')
    finally:
        iv.paired_batch = original


def audit_existing(model, loader, cfg, frame, output, rep, shifts):
    """只重放普通前向/FULL，已有18条件指标与案例采用只读引用。"""
    audit, measured = [], []
    for batch in loader:
        device = next(model.parameters()).device
        x, w = batch['x'].to(device), batch['tf']['w'].to(device)
        offset=len(audit)
        batch_shifts=torch.as_tensor(shifts[offset:offset+len(x)],dtype=torch.int64,device=device)
        paired=iv.paired_batch(model,x,w,rep['frequencies_hz'],batch_shifts)
        try:
            first=next(paired)
            if first[0] != 'FULL__NAT':
                raise ValueError('原机制入口首次条件不是FULL')
            full=first[1]
            with torch.inference_mode(), torch.autocast(x.device.type, enabled=x.is_cuda, dtype=torch.bfloat16):
                plain = model(x, tf={'w':w})['waveform']
        finally:
            paired.close()
        metadata = [_extract_meta(batch['meta'],i) for i in range(len(x))]
        audit.extend(wave_check(plain,full,[m['dataset_row_id'] for m in metadata]))
        blob = _prediction_dict_from_arrays(full.float().cpu().numpy(),batch['target'].numpy(),metadata,
                                             pred_key='r_tho_hat',target_key='tho_ref')
        measured.append(evaluate_task_predictions(blob,cfg,include_test_only=False,method='FULL__NAT'))
    current = pd.concat(measured,ignore_index=True)
    current.to_csv(output/'replayed_full_metrics.csv',index=False)
    pd.DataFrame(audit).to_csv(output/'same_batch_waveform_check.csv',index=False)
    saved = frame[frame.condition.eq('FULL__NAT')].reset_index(drop=True)
    if not current[['dataset_row_id','samp_id','split']].equals(saved[['dataset_row_id','samp_id','split']]):
        raise ValueError('已有FULL指标的样本身份不一致')
    cross_batch_report(current.assign(condition='FULL__NAT'),saved).to_csv(output/'historical_replay_diagnostic.csv',index=False)
    return audit


def reuse_same_batch_audit(session, entry, frame, output):
    """复用已执行完且所有波形差为0的检查；原失败只来自额外的跨运行指标断言。"""
    source=session/PRIOR_AUDIT
    failure=io.read_json(source/'failed.json')['traceback']
    if 'tolerance rtol=1e-07, atol=1e-08' not in failure or io.read_json(source/'access.json')['entry'] != entry:
        raise ValueError('历史波形验收来源不匹配')
    for relative,identity in io.read_json(source/'access.json')['amendment'].items():
        io.verify(source/Path(relative).name,identity)
    identities={str(p.relative_to(source)):io.identity(p) for p in source.rglob('*') if p.is_file()}
    table=pd.read_csv(source/'same_batch_waveform_check.csv')
    current=pd.read_csv(source/'replayed_full_metrics.csv')
    saved=frame[frame.condition.eq('FULL__NAT')].reset_index(drop=True)
    if (not table.dataset_row_id.equals(saved.dataset_row_id) or
        not table[['max_abs','mean_abs']].eq(0).all().all()):
        raise ValueError('历史逐窗波形验收未完整通过')
    cross_batch_report(current.assign(condition='FULL__NAT'),saved).to_csv(output/'historical_replay_diagnostic.csv',index=False)
    table.to_csv(output/'same_batch_waveform_check.csv',index=False)
    io.write_json(output/'prior_audit_source.json',{'path':str(source),'files':identities,'same_batch_check_reused':True})
    return table.to_dict('records')


def validate_matrix(frame, rows, seed):
    if len(frame) != len(rows)*len(iv.CONDITIONS) or set(frame.condition) != set(iv.CONDITIONS) or not frame.seed.eq(seed).all():
        raise ValueError('机制矩阵不完整')
    for condition in iv.CONDITIONS:
        part = frame[frame.condition.eq(condition)].reset_index(drop=True)
        if not part.arm.eq(condition).all():
            raise ValueError('arm/condition身份不一致')
        sf.validate_metrics(part,rows)
    natural = frame[frame.condition.eq('FULL__NAT')].reset_index(drop=True)
    fixed = frame[frame.condition.eq('FULL__FIXED')].reset_index(drop=True)
    for metric in sf.PRIMARY:
        np.testing.assert_array_equal(natural[metric],fixed[metric])


def certify_seed(session, seed, device, root, identities):
    key = test.test_key(session,'test_mechanisms_r2',seed=seed,amendment=identities)
    parent = root/f'seed_{seed}'
    prior = io.completed(parent,key)
    if prior:
        return prior
    frozen = io.load_session(session)
    _, allowed = test.allowlist(session)
    entry = next(e for e in allowed['entries'] if (e['arm'],e['seed']) == ('A0',seed))
    test.require_gpu(session,device)
    reference = io.completed(session/f'research_test/evaluation/A0/seed_{seed}',test.test_key(session,'test_evaluation',arm='A0',seed=seed))
    data_path,_ = test.test_data(session)
    rows = pd.read_csv(data_path/'test_rows.csv')[['dataset_row_id','samp_id','split']]
    case_ids = set(io.read_json(data_path/'cases.json')['dataset_row_ids'])
    with io.mutex(OUTPUT_ROOT/f'.device_{torch.device(device).index}.mutex'), io.attempt(parent,key) as output:
        for relative,identity in identities.items():
            path=ROOT/relative
            io.verify(path,identity)
            (output/path.name).write_bytes(path.read_bytes())
        io.write_json(output/'access.json',{'confirmed':True,'entry':entry,'same_batch':8,'dtype':'bfloat16',
                      'rtol':RTOL,'atol':ATOL,'amendment':identities})
        for identity in entry['sources'].values():
            io.verify(identity['path'],identity)
        cfg=OmegaConf.load(entry['sources']['config']['path'])
        cfg.training.device=device
        model=build_model(seed,frozen['representations']['A0']).to(device).eval()
        checkpoint=torch.load(entry['sources']['checkpoint']['path'],map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        bundle=test.test_bundle(session,'A0',cfg,confirmed=True)
        loader=DataLoader(bundle.dataset,batch_size=8,shuffle=False,num_workers=0)
        if seed in LEGACY:
            source=session/f'research_test/mechanisms/seed_{seed}'/LEGACY[seed]
            if seed==20260811:
                io.verify_stage(source,test.test_key(session,'test_mechanisms',seed=seed))
            else:
                failure=io.read_json(source/'failed.json')['traceback']
                if 'verify_full_replay' not in failure or 'Not equal to tolerance rtol=0.001, atol=0' not in failure:
                    raise ValueError('来源失败不是已授权修订的跨batch验收')
                if io.read_json(source/'access.json')['entry'] != entry:
                    raise ValueError('失败产物checkpoint身份不符')
            recovery=io.read_json(source/'recovery_source.json')
            if (recovery['pred_key'],recovery['target_key']) != ('r_tho_hat','tho_ref'):
                raise ValueError('既有导出适配不一致')
            for relative,identity in recovery['files'].items():
                io.verify(ROOT/relative,identity)
                io.verify(source/Path(relative).name,identity)
            source_files={str(p.relative_to(source)):io.identity(p) for p in source.rglob('*') if p.is_file()}
            io.write_json(output/'referenced_files.json',source_files)
            frame=pd.read_csv(source/'metrics.csv')
            validate_matrix(frame,rows,seed)
            if len(list(source.glob('case_*.npz'))) != len(case_ids)*18 or len(list(source.glob('case_*_metrics.json'))) != len(case_ids)*18:
                raise ValueError('已有案例导出不完整')
            if seed==20260811:
                audit=reuse_same_batch_audit(session,entry,frame,output)
            else:
                audit=audit_existing(model,loader,cfg,frame,output,frozen['representations']['A0'],
                                     np.load(data_path/'test_shift_frames.npy',allow_pickle=False))
            for relative,identity in source_files.items():
                io.verify(source/relative,identity)
            metrics_path=source/'metrics.csv'
        else:
            audit=[]
            with prediction_keys(), checked_pairing(rows.dataset_row_id.tolist(),audit):
                frame=iv.evaluate_loader(model,loader,cfg,frozen['representations']['A0'],
                      np.load(data_path/'test_shift_frames.npy',allow_pickle=False),case_ids,output,seed)
            validate_matrix(frame,rows,seed)
            source=output
            metrics_path=output/'metrics.csv'
        if [r['dataset_row_id'] for r in audit] != rows.dataset_row_id.tolist():
            raise ValueError('逐窗波形验收不完整或顺序不符')
        if seed not in LEGACY:
            pd.DataFrame(audit).to_csv(output/'same_batch_waveform_check.csv',index=False)
        cross_batch_report(frame,pd.read_csv(reference/'metrics.csv')).to_csv(output/'cross_batch_diagnostic.csv',index=False)
        io.write_json(output/'result.json',{'seed':seed,'conditions':18,'windows':len(rows),'same_batch_passed':True,
                      'maximum_waveform_difference':max(r['max_abs'] for r in audit),
                      'metrics':{'path':str(metrics_path),**io.identity(metrics_path)},'case_source':str(source),
                      'reused_mechanism_outputs':seed in LEGACY})
        io.verify_provenance(frozen['provenance'],session)
    return output


def summarize(session,root,certificates,identities):
    from resp_train.paper_evidence.cwt_apor_v2.summary import tables
    key=test.test_key(session,'test_mechanisms_r2_summary',amendment=identities)
    frames=[]
    for path in certificates:
        io.verify_stage(path)
        record=io.read_json(path/'result.json')['metrics']
        io.verify(record['path'],record)
        frames.append(pd.read_csv(record['path']))
    full=pd.concat(frames,ignore_index=True)
    with io.attempt(root/'summary',key) as output:
        io.write_json(output/'sources.json',[{'path':str(p),**io.identity(p/'manifest.json')} for p in certificates])
        for view,mask in [('full',np.ones(len(full),bool)),('exclude670',full.samp_id.ne(670)),('subject670',full.samp_id.eq(670))]:
            subset=full.loc[mask]
            for _,group in subset.groupby(['arm','seed']):
                if (len(group),group.samp_id.nunique()) != test.VIEW_COUNTS[view]:
                    raise ValueError('机制视图分母不符')
            folder=output/view
            folder.mkdir()
            for name,table in tables(subset,{(c,s) for c in iv.CONDITIONS for s in SEEDS},[(c,'FULL__NAT') for c in iv.CONDITIONS if c!='FULL__NAT']).items():
                table.to_csv(folder/f'{name}.csv',index=False)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,required=True)
    parser.add_argument('--confirm-research-test',action='store_true')
    args=parser.parse_args()
    if not args.confirm_research_test:
        parser.error('需要 --confirm-research-test')
    def stop(signum,_):
        raise KeyboardInterrupt(f'收到停止信号{signum}')
    signal.signal(signal.SIGTERM,stop)
    session=args.session.resolve()
    io.load_session(session)
    paths=(Path(__file__).resolve(),NOTE,TEST,ROOT/'scripts/recover_cwt_apor_v2_mechanisms.py')
    identities={str(p.relative_to(ROOT)):io.identity(p) for p in paths}
    root=session/'research_test/mechanisms_r2'/('revision_'+io.json_hash(identities)[:12])
    key=test.test_key(session,'test_mechanisms_r2_pipeline',amendment=identities)
    with io.attempt(root/'pipeline',key) as output:
        print(f'PIPELINE={output}',flush=True)
        io.write_json(output/'authorization.json',{'user_instruction':'可以，推进吧','amendment':identities,
                      'scope':'same_batch_FULL_gate_and_three_seed_mechanism_completion','command':sys.argv})
        certificates=[]
        for index,seed in enumerate(SEEDS):
            print(f'PHASE_START=seed_{seed}',flush=True)
            path=certify_seed(session,seed,f'cuda:{index%2}',root,identities)
            certificates.append(path)
            print(f'PHASE_COMPLETE=seed_{seed} OUTPUT={path}',flush=True)
        result=summarize(session,root,certificates,identities)
        io.write_json(output/'completion.json',{'completed':True,'summary':str(result),'seeds':list(SEEDS)})
    print('COMPLETED=mechanisms_r2',flush=True)


if __name__=='__main__':
    main()
