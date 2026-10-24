"""在冻结M4-S1运行时上按validation总loss选点，独立评价与比较。"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'm4-s1-loss-selected-research-test-v1-20261024'
DOC = ROOT / 'docs/experiments/m4_s1_loss_selected_research_test_v1_20261024.md'
PREVIOUS_SHA = 'e02211e4c330644a30324e0ada597059bb14f9289d1cce595c732a4b48c90338'


def read(path):
    return json.loads(Path(path).read_text())


def identity(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return {'sha256': digest.hexdigest(), 'size_bytes': Path(path).stat().st_size}


def select_loss(history):
    """只接受完整连续且有限的validation总loss轨迹；并列最早。"""
    if not history or [r['epoch'] for r in history] != list(range(1, len(history)+1)):
        raise ValueError('history必须连续且按epoch排序')
    for row in history:
        value = row.get('validation', {}).get('loss')
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('validation总loss缺失或非有限')
    return min(history, key=lambda r: (r['validation']['loss'], r['epoch']))


def runtime(root):
    sys.path.insert(0, str(Path(root)/'runtime'))
    from scripts import eval_m4_s1_research_test as rt
    return rt


def previous_key(rt, previous, phase, **fields):
    return {'protocol': 'm4-s1-research-test-v1-20261022',
            'allowlist_sha256': rt.io.sha(previous/'allowlist.json'), 'phase': phase, **fields}


def previous_stage(rt, previous, phase, **fields):
    parent=previous/phase
    for value in fields.values():
        parent=parent/str(value)
    path=rt.io.completed(parent, previous_key(rt, previous, phase, **fields))
    if path is None:
        raise ValueError(f'前次阶段缺失: {phase} {fields}')
    return path


def prepare(root, previous):
    root,previous=Path(root).resolve(),Path(previous).resolve()
    expected=read(previous/'allowlist_receipt.json')
    if identity(previous/'allowlist.json') != expected or expected['sha256'] != PREVIOUS_SHA:
        raise ValueError('前次allowlist身份变化')
    old=read(previous/'allowlist.json')
    if identity(previous/'source_snapshot.tar.gz') != old['snapshot']:
        raise ValueError('冻结runtime快照变化')
    root.mkdir(parents=True, exist_ok=False)
    (root/'runtime').mkdir()
    with tarfile.open(previous/'source_snapshot.tar.gz') as archive:
        archive.extractall(root/'runtime', filter='data')
    rt=runtime(root)
    rt.load(previous)
    training=Path(old['training_session']['path']).parent
    rt.io.load_session(training)
    entries=[]
    for entry in old['entries']:
        cell=training/'cells'/entry['cell']
        receipt=rt.runtime.verified_cell(training, entry['arm'], entry['seed'])
        cfg=rt.OmegaConf.load(cell/'config.yaml')
        history=rt.runtime.verify_training_result(cell,receipt,cfg)
        picked=select_loss(history)
        epochs=dict(rt.legacy.completed_epochs(cell))
        checkpoint=epochs[picked['epoch']]/'checkpoint.pt'
        saved=rt.torch.load(checkpoint,map_location='cpu',weights_only=False)
        rt.legacy.finite_tree(saved)
        if (saved['history'] != history[:picked['epoch']] or saved['epoch'] != picked['epoch']
                or saved['config'] != rt.legacy.config_identity(cfg)):
            raise ValueError('loss选点checkpoint内部身份不符')
        previous_record=history[entry['epoch']-1]
        selected={**entry, 'epoch':picked['epoch'], 'checkpoint':rt.source(checkpoint),
            'selected_validation_loss':picked['validation']['loss'], 'selected_validation_local_rr':picked['val_local_rr_mae'],
            'previous_epoch':entry['epoch'], 'previous_validation_loss':previous_record['validation']['loss'],
            'previous_validation_local_rr':previous_record['val_local_rr_mae'], 'previous_checkpoint':entry['checkpoint'],
            'reuse_previous':identity(checkpoint)=={k:entry['checkpoint'][k] for k in ('sha256','size_bytes')}}
        entries.append(selected)
        print('selected',entry['cell'],entry['epoch'],'->',picked['epoch'],flush=True)
    implementation={str(p):identity(p) for p in (Path(__file__).resolve(),DOC,ROOT/'tests/test_m4_s1_loss_selected_test.py')}
    snapshot=root/'implementation';snapshot.mkdir()
    for filename in implementation:
        shutil.copyfile(filename,snapshot/Path(filename).name)
    lock={**old,'protocol':PROTOCOL,'entries':entries,'previous_root':str(previous),
        'previous_allowlist':rt.source(previous/'allowlist.json'),'previous_snapshot':rt.source(previous/'source_snapshot.tar.gz'),
        'implementation':implementation,'selection_rule':'minimum_validation_total_loss_earliest_tie',
        'created_utc':datetime.now(timezone.utc).isoformat(),'command':sys.argv,
        'implementation_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'selection_scope':'completed_epochs_of_original_local_rr_early_stopped_training',
        'test_data_accessed':False,'default_model_changed':False}
    # development身份继续由旧allowlist保护；新输出保留相同样本表。
    for split in ('train','val'):
        shutil.copyfile(previous/f'{split}_rows.csv',root/f'{split}_rows.csv')
    rt.io.write_json(root/'allowlist.json',lock)
    rt.io.write_json(root/'allowlist_receipt.json',identity(root/'allowlist.json'))
    rt.pd.DataFrame([{k:v for k,v in e.items() if k in ('arm','seed','cell','epoch','previous_epoch','selected_validation_loss',
        'selected_validation_local_rr','previous_validation_loss','previous_validation_local_rr','reuse_previous')}
        | {'checkpoint':e['checkpoint']['path'],'checkpoint_sha256':e['checkpoint']['sha256']} for e in entries]).to_csv(root/'checkpoint_list.csv',index=False)
    return root


@contextmanager
def configured(root):
    """复用冻结推理与样本门禁，仅替换本轮协议/清单加载；退出后恢复。"""
    root=Path(root).resolve();rt=runtime(root)
    rt.io.verify(root/'allowlist.json',read(root/'allowlist_receipt.json'))
    lock=read(root/'allowlist.json');previous=Path(lock['previous_root'])
    if lock['protocol']!=PROTOCOL or len(lock['entries'])!=84:
        raise ValueError('本轮协议/矩阵不符')
    old_load,old_protocol=rt.load,rt.PROTOCOL
    old_load(previous)
    rt.verify_source(lock['previous_allowlist']);rt.verify_source(lock['previous_snapshot'])
    for filename,expected in lock['implementation'].items():
        rt.io.verify(filename,expected)
        rt.io.verify(root/'implementation'/Path(filename).name,expected)
    if {(e['arm'],e['seed']) for e in lock['entries']} != {(c['arm'],c['seed']) for c in rt.spec.plan()}:
        raise ValueError('84项身份不完整')
    def load(path):
        if Path(path).resolve()!=root:
            raise ValueError('本轮入口不接受其他输出目录')
        rt.io.verify(root/'allowlist.json',read(root/'allowlist_receipt.json'))
        return lock
    rt.PROTOCOL,rt.load=PROTOCOL,load
    try:
        yield rt,lock
    finally:
        rt.PROTOCOL,rt.load=old_protocol,old_load


def prepare_inputs(root, confirmed=False):
    with configured(root) as (rt,lock):
        rt.guard(confirmed)
        previous=Path(lock['previous_root'])
        for phase,fields in [('data',{})]+[('cache',{'representation':rep}) for rep in lock['representations']]:
            if rt.completed(root,phase,**fields):
                continue
            source=previous_stage(rt,previous,phase,**fields)
            parent=Path(root)/phase
            for value in fields.values():parent=parent/value
            with rt.io.attempt(parent,rt.key(root,phase,**fields)) as out:
                for path in source.iterdir():
                    if path.is_file() and path.name not in ('manifest.json','receipt.json','started.json','access.json'):
                        subprocess.run(['cp','--reflink=auto','--',str(path),str(out/path.name)],check=True)
                        rt.io.verify(out/path.name,identity(path))
                rt.io.write_json(out/'access.json',{'split':'test','confirmed':True,'kind':'verified_frozen_copy'})
                rt.io.write_json(out/'reuse_source.json',rt.source(source/'manifest.json'))
            print('prepared',phase,fields,flush=True)


def worker(root,device,shard,shards,confirmed=False,retry=False):
    with configured(root) as (rt,lock):
        rt.guard(confirmed)
        if not 0<=shard<shards<=84:raise ValueError('非法分片')
        with rt.io.mutex(Path(root)/f'.device_{device.replace(":","_")}.lock'):
            for entry in lock['entries'][shard::shards]:
                if rt.completed(root,'evaluation',cell=entry['cell']):continue
                if entry['reuse_previous']:
                    previous=Path(lock['previous_root'])
                    rt.verify_source(entry['checkpoint']);rt.verify_source(entry['previous_checkpoint'])
                    if entry['checkpoint']!=entry['previous_checkpoint']:
                        raise ValueError('复用要求完整checkpoint身份相同')
                    old=previous_stage(rt,previous,'evaluation',cell=entry['cell'])
                    with rt.io.attempt(Path(root)/'evaluation'/entry['cell'],rt.key(root,'evaluation',cell=entry['cell']),retry_failed=retry) as out:
                        rt.io.write_json(out/'reuse.json',{'entry':entry,'evaluation_manifest':rt.source(old/'manifest.json')})
                    print('reused',entry['cell'],flush=True)
                else:
                    rt.evaluate(root,entry,device,confirmed,retry)
        print('worker completed',shard,shards,flush=True)


def result_path(rt,root,entry):
    path=rt.completed(root,'evaluation',cell=entry['cell'])
    if path is None:raise ValueError('完整矩阵未完成: '+entry['cell'])
    if (path/'reuse.json').exists():
        record=read(path/'reuse.json')
        if record['entry']!=entry or not entry['reuse_previous']:
            raise ValueError('结果复用身份不符')
        rt.verify_source(record['evaluation_manifest'])
        path=Path(record['evaluation_manifest']['path']).parent
        rt.io.verify_stage(path)
    return path


def summarize(root,confirmed=False):
    with configured(root) as (rt,lock):
        rt.guard(confirmed)
        prior=rt.completed(root,'summary')
        if prior:return prior
        paths=[result_path(rt,root,e) for e in lock['entries']]
        frames=[rt.pd.read_csv(p/'metrics.csv') for p in paths]
        tables=rt.summary_tables(rt.pd.concat(frames,ignore_index=True))
        previous=Path(lock['previous_root'])
        old_summary=previous_stage(rt,previous,'summary')
        with rt.io.attempt(Path(root)/'summary',rt.key(root,'summary')) as out:
            for name,table in tables.items():table.to_csv(out/f'{name}.csv',index=False)
            comparison_outputs={}
            for prefix in ('','subject_macro_'):
                new=tables['per_seed' if not prefix else 'subject_macro']
                old=rt.pd.read_csv(old_summary/('per_seed.csv' if not prefix else 'subject_macro.csv'))
                paired=new[['arm','seed','metric','mean']].merge(old[['arm','seed','metric','mean']],on=['arm','seed','metric'],suffixes=('_loss','_local_rr'),validate='one_to_one')
                paired['delta']=paired.mean_loss-paired.mean_local_rr
                paired['improvement']=rt.np.where(paired.metric==rt.sf.PCC,paired.delta,-paired.delta)
                paired['group']='checkpoint_selector';paired['comparison']=paired.arm+'_loss_vs_local_rr';paired['reference']=paired.arm
                summary=rt.stats.pair_summary(paired)
                paired.to_csv(out/f'{prefix}selector_paired.csv',index=False)
                summary.to_csv(out/f'{prefix}selector_summary.csv',index=False)
                comparison_outputs[prefix]=summary
            lines=['# M4-S1 validation总loss选点research-test','',
                '完整84项固定checkpoint；对照为同cell原Local RR选点。delta=loss选点−Local RR选点；误差负值/PCC正值为改善。',
                '原训练按Local RR早停；此次比较限定已保存轨迹。原结果及默认模型保持不变。','']
            for prefix,title in (('','窗口加权'),('subject_macro_','受试者宏平均')):
                frame=tables[prefix+'across_seed'].set_index(['arm','metric'])
                lines += ['## '+title,'','| 配置 | '+' | '.join(rt.sf.PRIMARY)+' |','|---|'+'---:|'*5]
                for arm in rt.spec.ARMS:
                    lines.append('| '+arm+' | '+' | '.join(f"{frame.loc[(arm,m),'seed_mean']:.6f} ± {frame.loc[(arm,m),'seed_sd']:.6f}" for m in rt.sf.PRIMARY)+' |')
                lines+=['','### 两种选点的test配对变化','','| 配置 | 指标 | delta mean ± SD | 改善/退步seed |','|---|---|---:|---|']
                for row in comparison_outputs[prefix].itertuples():
                    lines.append(f'| {row.arm} | {row.metric} | {row.delta_mean:+.6f} ± {row.delta_sd:.6f} | {row.improved_seeds}/{row.worsened_seeds} |')
                lines+=['']
            (out/'report.md').write_text('\n'.join(lines)+'\n')
            rt.io.write_json(out/'sources.json',{'allowlist':rt.source(Path(root)/'allowlist.json'),
                'evaluations':[rt.source(p/'manifest.json') for p in paths],'previous_summary':rt.source(old_summary/'manifest.json'),
                'new_evaluations':sum(not e['reuse_previous'] for e in lock['entries']),
                'reused_evaluations':sum(e['reuse_previous'] for e in lock['entries'])})
        return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','prepare-inputs','worker','summarize','status'])
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--previous',type=Path)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--confirm-research-test',action='store_true');parser.add_argument('--retry-failed',action='store_true')
    a=parser.parse_args();a.root=a.root.resolve()
    if a.command=='prepare':print(prepare(a.root,a.previous))
    elif a.command=='prepare-inputs':prepare_inputs(a.root,a.confirm_research_test)
    elif a.command=='worker':worker(a.root,a.device,a.shard,a.shards,a.confirm_research_test,a.retry_failed)
    elif a.command=='summarize':print(summarize(a.root,a.confirm_research_test))
    else:
        lock=read(a.root/'allowlist.json')
        print({'planned':len(lock['entries']),'completed':sum(bool(list((a.root/'evaluation'/e['cell']).glob('attempt_*/receipt.json'))) for e in lock['entries'])})


if __name__=='__main__':main()
