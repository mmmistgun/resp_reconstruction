"""R3/GN 独立锁、信号检查、synthetic GPU 门控与完整 validation 配对。"""
from __future__ import annotations
import json
import shutil
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime,timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from resp_train.data.factory import build_window_data
from resp_train.engine import collect_predictions
from resp_train.metrics.task import evaluate_task_predictions,summarize_task_metrics
from resp_train.paper_evidence import e4_band_audit_runtime as prior
from resp_train.paper_evidence.e4_band_audit import paired_changes
from resp_train.paper_evidence.e1_scale_topology import PRIMARY,PCC,SEEDS
from resp_train.paper_evidence.e1_scale_topology_runtime import identity,sha256_file,write_json,git_state,checked_batches
from resp_train.paper_evidence.e4_r3_norm import CONDITIONS,TRANSFORMS,GN_NAMES,Trace,NormIntervention,make_shifts,describe,factorial_table
from resp_train.paper_evidence.e4_r3_signals import signal_window,summarize_signals,paired_correlations,summarize_paired

ROOT=Path(__file__).resolve().parents[2]
PROTOCOL='e4-r3-temporal-normalization-v1-20260921'
OUTPUT=Path('runs/e4_r3_temporal_normalization_v1')
LOCK=Path('docs/experiments/e4_r3_temporal_normalization_lock_20260921.json')
DOC=Path('docs/experiments/e4_r3_temporal_normalization_protocol_20260921.md')
SCRIPT=Path('scripts/eval_e4_r3_temporal_normalization.py')
TEST=Path('tests/test_e4_r3_norm.py')
SOURCE_LOCK=Path('docs/experiments/e4_band_encoding_aggregation_lock_20260918.json')
SOURCE_SHA='41084408b8b597ed7daaf6ecf081083e25397a25a647db035b692205989d6879'
REFERENCE=Path('runs/e4_band_encoding_aggregation_v1/reference/reference_41084408b8b5_20260918T112447Z_1498cff12931')
COUNT=2675


def prepare_lock():
    if (ROOT/LOCK).exists(): raise FileExistsError('R3/GN 实现锁已存在')
    if sha256_file(ROOT/SOURCE_LOCK)!=SOURCE_SHA: raise ValueError('来源锁 identity 漂移')
    old=json.loads((ROOT/SOURCE_LOCK).read_text())
    entries=[e for e in old['entries'] if e['arm']=='W0_FULL']
    if [(e['seed'],e['selected_epoch']) for e in entries]!=list(zip(SEEDS,(13,15,14))): raise ValueError('W0 allowlist 不完整')
    for entry in entries:
        for item in entry['files'].values(): prior.source.verify(Path(item['path']),item)
        if entry['config']['model']['variant']!='crd_tf102_w': raise ValueError('需要原 W0 模型')
        for section in ('window','loss','evaluation'):
            if entry['config'][section]!=entries[0]['config'][section]: raise ValueError('三 seed 的信号/指标合同不一致')
    # 此处只复用已冻结参考的文件身份，不加载 train/validation 数组。
    prior.verify_attempt(ROOT/REFERENCE,SOURCE_SHA,'reference')
    reference_meta=json.loads((ROOT/REFERENCE/'reference.json').read_text())
    if reference_meta['source']!=old['cache_files']['train_w']: raise ValueError('参考 train cache 来源漂移')
    code=sorted((ROOT/'resp_train').rglob('*.py'))+[ROOT/p for p in (DOC,SCRIPT,TEST)]
    result=dict(protocol=PROTOCOL,seeds=list(SEEDS),conditions=list(CONDITIONS),count=COUNT,entries=entries,
        validation_rows=old['validation_rows'],plot_row_ids=old['plot_row_ids'],cache_files=old['cache_files'],
        dataset_index=old['dataset_index'],frequencies=old['frequencies'],shifts=make_shifts(COUNT).tolist(),
        source_lock={'path':str(ROOT/SOURCE_LOCK),**identity(ROOT/SOURCE_LOCK)},
        reference={'path':str(ROOT/REFERENCE/'reference.npy'),**identity(ROOT/REFERENCE/'reference.npy')},
        reference_manifest={'path':str(ROOT/REFERENCE/'manifest.json'),**identity(ROOT/REFERENCE/'manifest.json')},
        code_files={str(p.relative_to(ROOT)):identity(p) for p in code},preparation_git=git_state(ROOT),
        prepared_at=datetime.now(timezone.utc).isoformat())
    write_json(ROOT/LOCK,result)
    return ROOT/LOCK


def load_lock():
    lock=json.loads((ROOT/LOCK).read_text())
    if lock['protocol']!=PROTOCOL or lock['seeds']!=list(SEEDS) or lock['conditions']!=list(CONDITIONS) or lock['count']!=COUNT:
        raise ValueError('R3/GN 矩阵漂移')
    if [(e['seed'],e['selected_epoch']) for e in lock['entries']]!=list(zip(SEEDS,(13,15,14))): raise ValueError('W0 allowlist 漂移')
    if not np.array_equal(np.asarray(lock['shifts']),make_shifts(COUNT)): raise ValueError('冻结偏移漂移')
    for key in ('source_lock','reference','reference_manifest'):
        item=lock[key]; prior.source.verify(Path(item['path']),item)
    for name,expected in lock['code_files'].items(): prior.source.verify(ROOT/name,expected)
    return lock,sha256_file(ROOT/LOCK)


@contextmanager
def attempt(phase,digest,seed=None):
    parent=ROOT/OUTPUT/phase
    if seed is not None: parent=parent/f'seed_{seed}'
    with prior.source.phase_guard(parent,digest):
        for old in parent.glob('*/freeze_receipt.json'):
            manifest=json.loads((old.parent/'manifest.json').read_text())
            if manifest['implementation_lock_sha256']==digest: raise FileExistsError(f'同身份已完成: {old.parent}')
        output=parent/f"{phase}_{digest[:12]}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:12]}"
        output.mkdir(parents=True,exist_ok=False)
        meta=dict(protocol=PROTOCOL,phase=phase,seed=seed,implementation_lock_sha256=digest,command=sys.argv,
                  started_at=datetime.now(timezone.utc).isoformat())
        write_json(output/'lifecycle_started.json',meta)
        print(f'R3/GN attempt: {output}',flush=True)
        try:
            yield output
            write_json(output/'lifecycle_completed.json',dict(meta,status='completed'))
            write_json(output/'manifest.json',dict(meta,status='completed',files={str(p.relative_to(output)):identity(p)
                for p in sorted(output.rglob('*')) if p.is_file()}))
            write_json(output/'freeze_receipt.json',{'manifest':identity(output/'manifest.json')})
        except BaseException as exc:
            write_json(output/'lifecycle_failed.json',dict(meta,error=str(exc),traceback=traceback.format_exc()))
            raise


def verify_attempt(output,digest,phase):
    output=Path(output).resolve()
    if (output/'lifecycle_failed.json').exists(): raise ValueError('R3/GN attempt 已失败')
    prior.source.verify(output/'manifest.json',json.loads((output/'freeze_receipt.json').read_text())['manifest'])
    manifest=json.loads((output/'manifest.json').read_text())
    if any(manifest[k]!=v for k,v in dict(protocol=PROTOCOL,phase=phase,implementation_lock_sha256=digest,status='completed').items()):
        raise ValueError('R3/GN attempt identity 漂移')
    required={
        'signals':{'signals_receipt.json','associations.csv','paired_associations.csv','rows.csv'},
        'gpu_smoke':{'gpu_smoke.json','environment.json'},
        'evaluation':{'evaluation_receipt.json','paired_changes.csv','layer_changes.csv','gn_statistics.csv','rows.csv'},
        'summary':{'summary_receipt.json','paired_seed_subject_changes.csv','train_mean_factorial.csv'},
    }.get(phase,set()) | {'lifecycle_completed.json'}
    if not required.issubset(manifest['files']): raise ValueError('R3/GN manifest 缺少必要产物')
    for rel,expected in manifest['files'].items():
        path=(output/rel).resolve()
        if not path.is_relative_to(output): raise ValueError('manifest 路径越界')
        prior.source.verify(path,expected)
    return manifest


def validation_data(lock,cfg,output):
    write_json(output/'access_started.json',{'split':'val','count':COUNT,'cache_files':{k:lock['cache_files'][k] for k in ('val_w','val_rows','frequency_file','manifest')},
        'dataset_index':lock['dataset_index']})
    for key in ('val_w','val_rows','frequency_file','manifest'):
        item=lock['cache_files'][key]; prior.source.verify(Path(item['path']),item)
    if sha256_file(Path(lock['dataset_index']['path']))!=lock['dataset_index']['sha256']: raise ValueError('数据索引漂移')
    data=build_window_data(cfg,split='val',max_windows=None,sample_strategy=str(cfg.data.val_sample_strategy),
                           sample_seed=int(cfg.data.val_sample_seed),shuffle=False)
    rows=pd.DataFrame(lock['validation_rows'])
    pd.testing.assert_frame_equal(data.rows[rows.columns].reset_index(drop=True),rows,check_dtype=False)
    if len(data.dataset)!=COUNT: raise ValueError('validation 数量漂移')
    rows.to_csv(output/'rows.csv',index=False)
    return data,rows


def model_for(entry,device):
    model,cfg=prior.load_model(entry,device)
    if getattr(model,'gamma_coefficient',.5)!=.5 or getattr(model,'beta_coefficient',.5)!=.5:
        raise ValueError('需要原 W0 FiLM 0.5/0.5')
    if len(model.base.local_blocks)!=6 or model.branches['w'].parameter_fill.expand.out_channels!=65:
        raise ValueError('W0 depth/fill 漂移')
    return model,cfg


def signals():
    lock,digest=load_lock()
    with attempt('signals',digest) as output:
        if shutil.disk_usage(output).free<4*2**30: raise RuntimeError('信号检查至少需要 4 GiB 可用空间')
        write_json(output/'environment.json',dict(git=git_state(ROOT,require_clean=True),python=sys.version,
            numpy=np.__version__,torch=torch.__version__))
        cfg=OmegaConf.create(lock['entries'][0]['config']); cfg.training.device='cpu'; cfg.training.show_progress=False
        data,rows=validation_data(lock,cfg,output)
        reference=np.load(lock['reference']['path'],allow_pickle=False)
        records=[]; cursor=0; figures=output/'figures'; figures.mkdir()
        for batch in checked_batches(data.loader,rows):
            for i in range(len(batch['x'])):
                index=cursor+i; row=rows.iloc[index].to_dict()
                records.extend(signal_window(batch['tf']['w'][i].numpy(),batch['target'][i].numpy().reshape(-1),
                    lock['shifts'][index],reference,cfg,row,figures if row['dataset_row_id'] in lock['plot_row_ids'] else None))
            cursor+=len(batch['x'])
            print(f'信号对应 {cursor}/{COUNT}',flush=True)
        if cursor!=COUNT: raise ValueError('信号检查样本不完整')
        frame=pd.DataFrame(records)
        frame.to_csv(output/'associations.csv',index=False)
        summary=summarize_signals(frame); summary.to_csv(output/'association_summary.csv',index=False)
        paired=paired_correlations(frame); paired.to_csv(output/'paired_associations.csv',index=False)
        paired_summary=summarize_paired(paired); paired_summary.to_csv(output/'paired_association_summary.csv',index=False)
        groupkeys=['transform','representation','association','lag_sec','support']
        summary[summary.scope.eq('samp_id')].groupby(groupkeys,sort=False).agg(
            subject_absolute_mean=('absolute_mean','mean'),subject_signed_mean=('signed_mean','mean'),defined_subjects=('absolute_mean','count')).to_csv(output/'association_subject_macro.csv')
        paired_summary[paired_summary.scope.eq('samp_id')].groupby(groupkeys,sort=False).agg(
            subject_absolute_drop=('absolute_drop_mean','mean'),subject_signed_change=('signed_change_mean','mean'),
            defined_subjects=('absolute_drop_mean','count')).to_csv(output/'paired_association_subject_macro.csv')
        write_json(output/'signals_receipt.json',dict(count=COUNT,transforms=list(TRANSFORMS),sample_ids=sorted(rows.samp_id.unique().tolist()),
            reference=lock['reference'],offsets=np.asarray(lock['shifts']).tolist(),model_inference=False))
    return output


def batch_conditions(model,batch,reference,shifts,device,native_predictions=None):
    """同一 batch 顺序跑完整矩阵，FULL 的层间 tensor 只在该 batch 暂存。"""
    baseline=None; mean_trace=None
    native=native_predictions if native_predictions is not None else collect_predictions(model,[batch],device=device,max_windows=len(batch['x']),use_amp=True)
    for condition in CONDITIONS:
        wrapper=NormIntervention(model,condition,reference,shifts,baseline,mean_trace)
        result=collect_predictions(wrapper,[batch],device=device,max_windows=len(batch['x']),use_amp=True)
        if condition=='FULL__NAT': baseline=wrapper.trace
        if condition=='TRAIN_MEAN__NAT':
            # STAT_ONLY 只需要首层统计，不能把 TRAIN_MEAN 的整批大特征一直留在 GPU。
            mean_trace=Trace(natural={'norm':wrapper.trace.natural['norm']})
        if condition.startswith('FULL__'):
            np.testing.assert_allclose(result['r_tho_hat'],native['r_tho_hat'],rtol=1e-5,atol=1e-6)
        yield condition,result,wrapper.trace,baseline


def gpu_smoke(device):
    lock,digest=load_lock()
    with attempt('gpu_smoke',digest) as output:
        write_json(output/'environment.json',prior.source.runtime_preflight(device))
        write_json(output/'access_started.json',{'inputs':'synthetic','checkpoints':'three selected W0','real_inputs':False})
        records=[]
        for entry in lock['entries']:
            model,cfg=model_for(entry,device)
            for batch_size in (1,128):
                write_json(output/f"seed_{entry['seed']}_batch_{batch_size}_started.json",dict(seed=entry['seed'],batch_size=batch_size,conditions=list(CONDITIONS)))
                print(f"GPU smoke: seed={entry['seed']}, batch={batch_size}",flush=True)
                batch=prior.synthetic_batch(batch_size,20260921,'cpu')
                batch['meta']={'dataset_row_id':torch.arange(batch_size),'samp_id':torch.ones(batch_size,dtype=torch.int64),'split':['val']*batch_size}
                shifts=torch.from_numpy(make_shifts(batch_size)); reference=torch.linspace(.02,.3,97)
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
                seen=[]; replay=0.
                for condition,result,trace,baseline in batch_conditions(model,batch,reference,shifts,device):
                    if not np.isfinite(result['r_tho_hat']).all(): raise FloatingPointError('smoke prediction 非有限')
                    seen.append(condition); replay=max(replay,*trace.replay_max_abs.values())
                    del trace
                fraction=torch.cuda.max_memory_reserved(device)/torch.cuda.get_device_properties(device).total_memory
                if fraction>.8: raise RuntimeError(f'GPU reserved {fraction} 超过 80%')
                record=dict(seed=entry['seed'],batch_size=batch_size,conditions=seen,full_replay_passed=True,
                    norm_self_replay_max_abs=replay,peak_reserved_fraction=fraction,peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
                write_json(output/f"seed_{entry['seed']}_batch_{batch_size}.json",record); records.append(record)
            del model
        write_json(output/'gpu_smoke.json',dict(passed=True,records=records))
    return output


def validate_gpu(path,digest,environment):
    verify_attempt(path,digest,'gpu_smoke')
    if json.loads((path/'environment.json').read_text())!=environment: raise ValueError('GPU 环境或 commit 与验收不符')
    receipt=json.loads((path/'gpu_smoke.json').read_text())
    if not receipt['passed'] or len(receipt['records'])!=6 or {(x['seed'],x['batch_size']) for x in receipt['records']}!={(s,b) for s in SEEDS for b in (1,128)}:
        raise ValueError('GPU smoke 矩阵不完整')
    if not all(x['conditions']==list(CONDITIONS) and x['full_replay_passed'] and x['peak_reserved_fraction']<=.8 for x in receipt['records']):
        raise ValueError('GPU smoke 未覆盖所有条件')


def evaluate(seed,device,signal_receipt,gpu_receipt):
    lock,digest=load_lock(); environment=prior.source.runtime_preflight(device)
    validate_gpu(gpu_receipt,digest,environment)
    verify_attempt(signal_receipt,digest,'signals')
    signal_meta=json.loads((signal_receipt/'signals_receipt.json').read_text())
    if (signal_meta['count']!=COUNT or signal_meta['transforms']!=list(TRANSFORMS) or signal_meta['offsets']!=lock['shifts']
            or signal_meta['reference']!=lock['reference'] or signal_meta['model_inference'] is not False):
        raise ValueError('信号回执矩阵/参考/偏移不匹配')
    entry=next(e for e in lock['entries'] if e['seed']==seed)
    with attempt('evaluation',digest,seed) as output:
        if shutil.disk_usage(output).free<8*2**30: raise RuntimeError('每 seed 评价至少需要 8 GiB 可用空间')
        write_json(output/'environment.json',environment); write_json(output/'implementation_lock.json',lock)
        model,cfg=model_for(entry,device); OmegaConf.save(cfg,output/'resolved_config.yaml')
        data,rows=validation_data(lock,cfg,output)
        # 先通过完整原生 FULL 门槛，再开始 22 条件；只保留约 0.2 GiB 的 CPU 波形参照。
        native=collect_predictions(model,checked_batches(data.loader,rows),device=device,max_windows=COUNT,use_amp=True)
        native_metrics=evaluate_task_predictions(native,cfg,include_test_only=False,method='W0_FULL')
        prior.source.validate_metrics(native_metrics,rows)
        frozen=pd.read_csv(entry['files']['metrics.csv']['path'])
        paired_changes(frozen,native_metrics,'W0_FULL',seed,'NATIVE_BASELINE')
        for metric in PRIMARY:
            if not np.isclose(native_metrics[metric].mean(),frozen[metric].mean(),rtol=1e-3,atol=0):
                raise RuntimeError(f'原生 FULL 未复现冻结指标: {metric}')
        native_metrics.to_csv(output/'native_baseline_metrics.csv',index=False)
        write_json(output/'baseline_reproduction.json',{'passed':True,'rtol':1e-3,'atol':0,'count':COUNT})
        reference=torch.from_numpy(np.load(lock['reference']['path'],allow_pickle=False))
        metric_frames={c:[] for c in CONDITIONS}; cursor=0
        eps={name:module.eps for name,module in model.branches['w'].named_modules() if name in GN_NAMES}
        for batch in checked_batches(data.loader,rows):
            layer_records=[]; stats_records=[]
            count=len(batch['x']); slice_rows=rows.iloc[cursor:cursor+count].to_dict('records')
            shifts=torch.tensor(lock['shifts'][cursor:cursor+count],dtype=torch.int64)
            reference_predictions={'r_tho_hat':native['r_tho_hat'][cursor:cursor+count]}
            for condition,predictions,trace,baseline in batch_conditions(model,batch,reference,shifts,device,reference_predictions):
                metrics=evaluate_task_predictions(predictions,cfg,include_test_only=False,method='W0_FULL')
                prior.source.validate_metrics(metrics,rows.iloc[cursor:cursor+count])
                metric_frames[condition].append(metrics)
                layers,stats=describe(trace,baseline,slice_rows,condition,eps)
                layer_records.extend(layers); stats_records.extend(stats)
                # 只保存预固定窗口的层间示例，所有示例共享原窗口 identity。
                selected=[(i,row) for i,row in enumerate(slice_rows) if row['dataset_row_id'] in lock['plot_row_ids']]
                for i,row in selected:
                    directory=output/'examples'/f"row_{row['dataset_row_id']}"; directory.mkdir(parents=True,exist_ok=True)
                    example={}
                    for name,value in trace.points.items():
                        local=value[i].float()
                        example[f'{name}_channel_mean']=local.mean(0).cpu().numpy()
                        example[f'{name}_channel_rms']=local.square().mean(0).sqrt().cpu().numpy()
                    example['prediction']=predictions['r_tho_hat'][i]
                    example['target']=predictions['tho_ref'][i]
                    np.savez(directory/f'{condition}.npz',**example)
                del trace
            pd.DataFrame(layer_records).to_csv(output/'layer_changes.csv',index=False,mode='a',header=cursor==0)
            pd.DataFrame(stats_records).to_csv(output/'gn_statistics.csv',index=False,mode='a',header=cursor==0)
            cursor+=count; print(f'完成 seed={seed}: {cursor}/{COUNT} ×22 条件',flush=True)
        if cursor!=COUNT: raise ValueError('validation 未覆盖全部窗口')
        full=pd.concat(metric_frames['FULL__NAT'],ignore_index=True)
        paired_changes(frozen,full,'W0_FULL',seed,'FULL__NAT')
        for metric in PRIMARY:
            if not np.isclose(full[metric].mean(),frozen[metric].mean(),rtol=1e-3,atol=0): raise RuntimeError(f'FULL 未复现冻结指标: {metric}')
        comparisons=[]; quality={}
        for condition,frames in metric_frames.items():
            metrics=pd.concat(frames,ignore_index=True); quality[condition]=prior.source.validate_metrics(metrics,rows)
            summary=summarize_task_metrics(metrics)
            if not np.isfinite(summary[[m+'_mean' for m in PRIMARY]].to_numpy()).all(): raise FloatingPointError('主指标非有限')
            directory=output/condition; directory.mkdir()
            metrics.to_csv(directory/'metrics.csv',index=False); summary.to_csv(directory/'summary.csv',index=False)
            comparisons.append(paired_changes(full,metrics,'W0_FULL',seed,condition))
        paired=pd.concat(comparisons,ignore_index=True); paired.to_csv(output/'paired_changes.csv',index=False)
        factorial_table(paired).to_csv(output/'train_mean_factorial.csv',index=False)
        write_json(output/'evaluation_receipt.json',dict(seed=seed,selected_epoch=entry['selected_epoch'],count=COUNT,conditions=list(CONDITIONS),
            checkpoint=entry['files']['checkpoint_best_local_rr.pt'],signals={'path':str(signal_receipt),'manifest':identity(signal_receipt/'manifest.json')},
            quality=quality,full_reproduced=True))
    return output


def summarize():
    lock,digest=load_lock(); frames=[]; fact=[]; sources={}; signal_source=None
    for seed in SEEDS:
        paths=[]
        for freeze in (ROOT/OUTPUT/'evaluation'/f'seed_{seed}').glob('*/freeze_receipt.json'):
            manifest=json.loads((freeze.parent/'manifest.json').read_text())
            if manifest['implementation_lock_sha256']==digest: paths.append(freeze.parent)
        if len(paths)!=1: raise ValueError(f'{seed} 需要唯一成功评价')
        path=paths[0]; manifest=verify_attempt(path,digest,'evaluation'); receipt=json.loads((path/'evaluation_receipt.json').read_text())
        if manifest['seed']!=seed or receipt['seed']!=seed or receipt['count']!=COUNT or receipt['conditions']!=list(CONDITIONS) or not receipt['full_reproduced']:
            raise ValueError('评价回执不完整')
        entry=next(e for e in lock['entries'] if e['seed']==seed)
        if receipt['checkpoint']!=entry['files']['checkpoint_best_local_rr.pt'] or receipt['selected_epoch']!=entry['selected_epoch']:
            raise ValueError('汇总 checkpoint allowlist 不匹配')
        if signal_source is None: signal_source=receipt['signals']
        elif signal_source!=receipt['signals']: raise ValueError('信号来源混用')
        frame=pd.read_csv(path/'paired_changes.csv')
        if len(frame)!=22*8*5 or not frame.seed.eq(seed).all() or set(frame.condition)!=set(CONDITIONS): raise ValueError('配对矩阵不完整')
        frames.append(frame); fact.append(factorial_table(frame)); sources[str(seed)]={'path':str(path),'manifest':identity(path/'manifest.json')}
    combined=pd.concat(frames,ignore_index=True); factorial=pd.concat(fact,ignore_index=True)
    verify_attempt(Path(signal_source['path']),digest,'signals')
    prior.source.verify(Path(signal_source['path'])/'manifest.json',signal_source['manifest'])
    summary=[]
    for key,g in combined.groupby(['condition','scope','group','metric'],sort=False,dropna=False):
        if len(g)!=3 or set(g.seed)!=set(SEEDS): raise ValueError('三 seed 配对缺失')
        summary.append(dict(zip(['condition','scope','group','metric'],key),defined_seeds=int(g.degradation.notna().sum()),
            degradation_mean=g.degradation.mean(),degradation_sd=g.degradation.std(ddof=1),
            worse_seeds=int((g.degradation>0).sum()),better_seeds=int((g.degradation<0).sum())))
    macro=combined[combined.scope.eq('samp_id')].groupby(['seed','condition','metric'],sort=False).agg(
        subject_macro_degradation=('degradation','mean'),defined_subjects=('degradation','count'),
        worse_subjects=('degradation',lambda x:int((x>0).sum())),better_subjects=('degradation',lambda x:int((x<0).sum())))
    with attempt('summary',digest) as output:
        combined.to_csv(output/'paired_seed_subject_changes.csv',index=False)
        pd.DataFrame(summary).to_csv(output/'three_seed_direction.csv',index=False)
        factorial.to_csv(output/'train_mean_factorial.csv',index=False); macro.to_csv(output/'subject_macro_by_seed.csv')
        summarize_diagnostics(sources,output)
        write_json(output/'summary_receipt.json',dict(models=3,conditions_per_model=22,condition_windows=3*22*COUNT,
            sources=sources,signals=signal_source,delta='error=condition-FULL; PCC=FULL-condition; positive=worse'))
    return output


def summarize_diagnostics(sources,output):
    """层间描述量按窗口汇总；大 GN CSV 分块求和与最大值，保留组和受试者。"""
    layer_records=[]; norm_frames=[]
    names=('mean_shift','log_variance','used_mean_shift','used_log_variance')
    for seed,source_info in sources.items():
        path=Path(source_info['path'])
        layer=pd.read_csv(path/'layer_changes.csv')
        fields=[c for c in layer.columns if c not in ('dataset_row_id','samp_id','split','condition')]
        for condition,frame in layer.groupby('condition',sort=False):
            groups=[('pooled','ALL',frame),*(('samp_id',str(s),g) for s,g in frame.groupby('samp_id'))]
            for scope,subject,g in groups:
                for metric in fields:
                    v=g[metric]
                    layer_records.append(dict(seed=int(seed),condition=condition,scope=scope,group=subject,metric=metric,
                        n=len(v),defined_n=int(v.notna().sum()),mean=v.mean(),median=v.median(),p90=v.quantile(.9)))
        pieces=[]; keys=['condition','layer','gn_group','samp_id']
        for chunk in pd.read_csv(path/'gn_statistics.csv',chunksize=100000):
            chunk=chunk.rename(columns={'group':'gn_group'})
            chunk['mean_shift']=chunk.normalized_mean_shift.abs()
            chunk['log_variance']=chunk.log_variance_plus_eps_ratio.abs()
            chunk['used_mean_shift']=((chunk.used_mean-chunk.full_mean)*chunk.full_rstd).abs()
            chunk['used_log_variance']=(2*np.log(chunk.full_rstd/chunk.used_rstd)).abs()
            agg={'n':('mean','size')}
            for name in names:
                agg[name+'_sum']=(name,'sum'); agg[name+'_max']=(name,'max')
            pieces.append(chunk.groupby(keys,sort=False).agg(**agg).reset_index())
        merged=pd.concat(pieces,ignore_index=True)
        reduction={c:('max' if c.endswith('_max') else 'sum') for c in merged.columns if c not in keys}
        per_subject=merged.groupby(keys,as_index=False,sort=False).agg(reduction)
        pooled=per_subject.groupby(keys[:-1],as_index=False,sort=False).agg(reduction)
        per_subject['scope']='samp_id'; per_subject['group']=per_subject.samp_id.astype(str)
        pooled['scope']='pooled'; pooled['group']='ALL'
        combined=pd.concat([per_subject.drop(columns='samp_id'),pooled],ignore_index=True)
        for name in names: combined[name+'_mean']=combined[name+'_sum']/combined.n
        combined.insert(0,'seed',int(seed)); norm_frames.append(combined)
    pd.DataFrame(layer_records).to_csv(output/'layer_change_summary.csv',index=False)
    pd.concat(norm_frames,ignore_index=True).to_csv(output/'gn_statistics_summary.csv',index=False)
