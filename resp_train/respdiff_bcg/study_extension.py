"""P4-N2 独立扩展：仅 DDIM6/N64 与 DDPM50/N16，保留原首轮上限。"""
from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch

from .inference_v2 import evaluate_setting, predict_prefixes, reconstruct_ensemble_parent, state_digest
from .runtime import LOGGER, environment, save_source_snapshot, sha256, write_json
from .sampling_accelerated import AccelerationSpec
from .study_contract import GRID, METRICS, StudySamplerSpec, tolerance_candidates
from .study_evaluation import rule_sample_summary
from .study_io import (ROOT, data_identity, implementation_identity, selected_dataset, study_run,
                       verify_artifact, verify_data_unchanged)
from .study_runtime import _initialize, _check_source, read_stage
from .study_state import preserved_evaluation

REVISION='P4-N2-ddim64-ddpm16-v1'
CHECKPOINT='c1e16cfdfba3498a60eafad11e1a72007b5e9969ba921d2eb7b679690d558731'
LIMITS={('ddim',6):64,('ddpm',50):16}


@dataclass(frozen=True)
class ExtensionSpec(StudySamplerSpec):
    def __post_init__(self):
        if (type(self.nfe) is not int or type(self.n_trajectories) is not int
                or LIMITS.get((self.sampler,self.nfe))!=self.n_trajectories
                or type(self.noise_seed) is not int or self.noise_seed!=20261003
                or self.postfilter is not True or self.eta!=0):
            raise ValueError('P4-N2 只允许 DDIM6/N64、DDPM50/N16，固定 sampling seed20261003')


def extension_identity(cfg):
    identity=implementation_identity(cfg)
    name='scripts/run_respdiff_bcg_extension_v1.py'
    identity['files'][name]=sha256(ROOT/name)
    return identity


def compare_prefix(chunks, reference_path, output):
    """新最大 N 内自然得到的旧前缀，用旧 waveform 验证跨设备数值一致性。"""
    with np.load(reference_path) as reference:
        expected=reference['prediction_postfiltered']
        actual=np.stack([reconstruct_ensemble_parent(parent,postfilter=True)[2] for parent in chunks])
        if actual.shape!=expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise ValueError('旧 N 前缀 shape 或有限性不匹配')
        delta=np.abs(actual-expected)
        # 跨同型号 GPU 允许小量浮点差；阈值在采样前固定，不据结果放宽。
        passed=bool(np.all(delta<=1e-5+1e-4*np.abs(expected)))
        report={'passed':passed,'atol':1e-5,'rtol':1e-4,'max_abs_difference':float(delta.max()),
            'rmse':float(np.sqrt(np.mean((actual.astype(np.float64)-expected)**2))),
            'reference_sha256':sha256(reference_path),'exact_equal':bool(np.array_equal(actual,expected))}
    write_json(output,report)
    if not passed:raise ValueError('新运行的旧 N 前缀偏离原结果，见 prefix_verification.json')
    return report


def run_extension(output, *, sampler, source_run, grid_run, device='cuda:0'):
    steps=6 if sampler=='ddim' else 50
    spec=ExtensionSpec(sampler,steps,LIMITS[(sampler,steps)])
    grid,grid_receipt=read_stage(grid_run,'grid')
    for name in ('grid_identity.json','val_parents.csv','selection.json','source_identity.json'):
        verify_artifact(grid,grid_receipt,name)
    previous=json.loads((grid/'grid_identity.json').read_text())
    chosen=json.loads((grid/'selection.json').read_text())['selected']
    if (previous['checkpoint']['checkpoint_sha256']!=CHECKPOINT or previous['batch_size']!=64
            or previous['trajectory_group_size']!=1 or not any(r['sampler']==sampler and
                r['S']==steps and r['N']==spec.n_trajectories//2 and r['role']=='selected' for r in chosen)):
        raise ValueError('P4-N2 来源不是已冻结的对应边界优选')
    old_files=json.loads((grid/'source_identity.json').read_text())['files']
    for name,h in old_files.items():
        if name.startswith(('resp_train/','scripts/','configs/')) and sha256(ROOT/name)!=h:
            raise ValueError(f'首轮数值实现或配置发生变化：{name}')
    reference_name=f'{sampler}_S{steps}_seed20261003_N{spec.n_trajectories//2}/validation_postfiltered_waveforms.npz'
    verify_artifact(grid,grid_receipt,reference_name)
    with study_run(output,'grid-extension',synthetic=False,source=source_run) as (out,receipt):
        model,cfg,checkpoint,bundles,_,index=_initialize(out,'grid-extension',source_run,device,
            tiny=False,expected_sha256=CHECKPOINT)
        parents,rows,_=bundles['val']
        dataset=selected_dataset(parents,rows,previous['source_parent_indices'])
        if dataset.rows.to_csv(index=False).encode()!=(grid/'val_parents.csv').read_bytes():
            raise ValueError('扩展 parent 身份或顺序改变')
        identity=extension_identity(cfg)
        OmegaConf.save(cfg,out/'resolved_config.yaml')
        write_json(out/'source_identity.json',identity)
        save_source_snapshot(out,identity)
        write_json(out/'environment.json',environment(cpu_only=False))
        write_json(out/'execution.json',{'revision':REVISION,'checkpoint':checkpoint,'spec':spec.identity(),
            'grid_run':str(grid),'grid_receipt_sha256':sha256(grid/'receipt.json'),
            'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'logical_device':device})
        data=data_identity(out,{'val':dataset},index)
        rule_ids=rows.iloc[np.linspace(0,len(rows)-1,64,dtype=int)].dataset_row_id.astype(int).tolist()
        sampling=out/'sampling';sampling.mkdir()
        gpu=torch.device(device)
        properties=torch.cuda.get_device_properties(gpu)
        torch.cuda.synchronize(gpu)
        torch.cuda.reset_peak_memory_stats(gpu)
        with preserved_evaluation(model):
            predictions,profile=predict_prefixes(model,dataset,spec,
                (spec.n_trajectories//2,spec.n_trajectories),sampling,batch_size=64,
                acceleration=AccelerationSpec(1))
            torch.cuda.synchronize(gpu)
            memory={'device_name':properties.name,'total_memory_bytes':properties.total_memory,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(gpu),
                'peak_reserved_bytes':torch.cuda.max_memory_reserved(gpu),
                'batch_size':64,'trajectory_group_size':1,'N':spec.n_trajectories,
                'scope':'whole extended-N sampling sweep; before CPU reconstruction/metrics'}
            write_json(out/'memory_profile.json',memory)
            LOGGER.info('扩展采样显存：allocated=%.3f GiB reserved=%.3f GiB',
                        memory['peak_allocated_bytes']/2**30,memory['peak_reserved_bytes']/2**30)
            prefix=compare_prefix(predictions.pop(spec.n_trajectories//2),grid/reference_name,
                                  out/'prefix_verification.json')
            setting=out/f'{sampler}_S{steps}_seed20261003_N{spec.n_trajectories}'
            curve=evaluate_setting(predictions.pop(spec.n_trajectories),dataset,cfg,setting,spec,
                spec.n_trajectories,method_prefix=f'RespDiff-study-v1/{REVISION}')
            curve.update(rule_sample_summary(setting,rule_ids))
        curve.update({'training_seed':20260811,'sampling_seed':20261003,'checkpoint_sha256':CHECKPOINT})
        pd.DataFrame([curve]).to_csv(out/'sampler_grid.csv',index=False)
        pd.DataFrame(profile).to_csv(out/'runtime_profile.csv',index=False)
        verify_data_unchanged(data,dataset.rows)
        _check_source(checkpoint,source_run,model)
        if extension_identity(cfg)!=identity:raise ValueError('扩展期间源码或协议变化')
        receipt.update({'revision':REVISION,'checkpoint':checkpoint,'spec':spec.identity(),
            'grid_receipt_sha256':sha256(grid/'receipt.json'),'parameter_sha256_after':state_digest(model),
            'prefix_verification':prefix,'memory_profile':memory})


def select_extended(curve):
    """仅接受原 22 点与两个授权新增点，质量/成本筛选顺序沿用首轮。"""
    expected={(kind,s,n) for (kind,s),counts in GRID.items() for n in counts}|{('ddim',6,64),('ddpm',50,16)}
    if len(curve)!=24 or set(zip(curve.sampler,curve.nfe_per_trajectory,curve.N))!=expected:
        raise ValueError('P4-N2 汇总要求完整 24 配置')
    columns=['nfe_per_trajectory','N','denoiser_calls_per_chunk',*METRICS]
    if (not np.isfinite(curve[columns].to_numpy(dtype=float)).all()
            or not (curve.denoiser_calls_per_chunk==curve.nfe_per_trajectory*curve.N).all()
            or (curve[list(METRICS[1:])]<0).any().any() or (curve[METRICS[0]].abs()>1+1e-6).any()):
        raise ValueError('扩展曲线存在非法数值或计算预算')
    selected=[]
    for kind in ('ddpm','ddim'):
        candidates=tolerance_candidates(curve[curve.sampler==kind])
        row=candidates.sort_values(['denoiser_calls_per_chunk',METRICS[0],'nfe_per_trajectory','N'],
                                   ascending=[True,False,True,True]).iloc[0]
        selected.append({'sampler':kind,'S':int(row.nfe_per_trajectory),'N':int(row.N),'role':'selected'})
    if not any(r['sampler']=='ddim' and r['S']==6 and r['N']==16 for r in selected):
        selected.append({'sampler':'ddim','S':6,'N':16,'role':'fixed_reference'})
    comparisons=[]
    for kind,steps,n in [('ddim',6,32),('ddpm',50,8)]:
        f=curve[(curve.sampler==kind)&(curve.nfe_per_trajectory==steps)].set_index('N')
        relative={key:(abs(float(f.loc[2*n,key]-f.loc[n,key]))/abs(float(f.loc[n,key]))
            if f.loc[n,key] else (0. if f.loc[2*n,key]==0 else None)) for key in METRICS[1:]}
        pcc=abs(float(f.loc[2*n,METRICS[0]]-f.loc[n,METRICS[0]]))
        comparisons.append({'sampler':kind,'S':steps,'N':n,'next_N':2*n,'pcc_change':pcc,
            'relative_changes':relative,'passed':pcc<.01 and all(v is not None and v<.05 for v in relative.values())})
    return {'revision':REVISION,'selected':selected,'new_doubling_comparisons':comparisons,
            'selection_noise_seed':20261003,'independent_timing_used':False,
            'stability_note':'Earlier adjacent comparisons all failed; one new pass is preliminary only.'}


def summarize_extensions(output, *, grid_run, ddim_run, ddpm_run):
    grid,receipt=read_stage(grid_run,'grid')
    verify_artifact(grid,receipt,'sampler_grid.csv')
    frames=[pd.read_csv(grid/'sampler_grid.csv')]
    sources={'grid_receipt_sha256':sha256(grid/'receipt.json')}
    for kind,path in [('ddim',ddim_run),('ddpm',ddpm_run)]:
        path,r=read_stage(path,'grid-extension')
        for name in ('sampler_grid.csv','val_parents.csv'):verify_artifact(path,r,name)
        steps=6 if kind=='ddim' else 50
        expected_spec=ExtensionSpec(kind,steps,LIMITS[(kind,steps)]).identity()
        if (r.get('revision')!=REVISION or r['spec']!=expected_spec or not r['prefix_verification']['passed'] or
                r['grid_receipt_sha256']!=sources['grid_receipt_sha256'] or
                r['checkpoint']['checkpoint_sha256']!=CHECKPOINT or
                r['parameter_sha256_after']!=r['checkpoint']['parameter_sha256'] or
                (path/'val_parents.csv').read_bytes()!=(grid/'val_parents.csv').read_bytes()):
            raise ValueError('两臂扩展身份不匹配')
        frames.append(pd.read_csv(path/'sampler_grid.csv'))
        sources[kind+'_receipt_sha256']=sha256(path/'receipt.json')
    curve=pd.concat(frames,ignore_index=True)
    if (set(curve.sampling_seed)!={20261003} or set(curve.training_seed)!={20260811}
            or set(curve.checkpoint_sha256)!={CHECKPOINT}):
        raise ValueError('汇总曲线模型或噪声 seed 身份不一致')
    result=select_extended(curve)
    with study_run(output,'select-extension',synthetic=False) as (out,receipt):
        curve.to_csv(out/'sampler_grid.csv',index=False)
        write_json(out/'selection.json',result)
        write_json(out/'sources.json',sources)
        receipt.update({'revision':REVISION,'selected':result['selected']})
