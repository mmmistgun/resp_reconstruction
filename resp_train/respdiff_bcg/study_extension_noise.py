"""P4-N2 固定 N 的两组额外采样噪声核验；不重新选择配置。"""
from dataclasses import dataclass
import json
import os

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch

from .inference_v2 import evaluate_setting,predict_prefixes,state_digest
from .runtime import LOGGER,environment,save_source_snapshot,sha256,write_json
from .sampling_accelerated import AccelerationSpec
from .study_contract import METRICS
from .study_evaluation import rule_sample_summary
from .study_extension import CHECKPOINT,LIMITS,ExtensionSpec,compare_prefix,extension_identity
from .study_io import ROOT,data_identity,selected_dataset,study_run,verify_artifact,verify_data_unchanged
from .study_runtime import _initialize,_check_source,read_stage
from .study_state import preserved_evaluation

REVISION='P4-N2-noise-check-v1'
SEEDS=(20261013,20261014)


@dataclass(frozen=True)
class ExtensionNoiseSpec(ExtensionSpec):
    def __post_init__(self):
        if type(self.noise_seed) is not int or self.noise_seed not in SEEDS:
            raise ValueError('P4-N2 噪声核验固定 seeds20261013/20261014')
        ExtensionSpec(self.sampler,self.nfe,self.n_trajectories,postfilter=self.postfilter,eta=self.eta)


def noise_identity(cfg):
    identity=extension_identity(cfg)
    name='scripts/run_respdiff_bcg_extension_noise_v1.py'
    identity['files'][name]=sha256(ROOT/name)
    return identity


def frozen_selection(path):
    path,receipt=read_stage(path,'select-extension')
    for name in ('selection.json','sources.json','sampler_grid.csv'):verify_artifact(path,receipt,name)
    selection=json.loads((path/'selection.json').read_text())
    selected={(r['sampler'],r['S'],r['N']) for r in selection['selected'] if r['role']=='selected'}
    if selected!={('ddim',6,64),('ddpm',50,16)} or selection['selection_noise_seed']!=20261003:
        raise ValueError('噪声核验要求已冻结的 DDIM6/N64 与 DDPM50/N16')
    return path,receipt,selection


def run_noise(output, *, sampler, source_run, grid_run, reference_noise_run, selection_run, main_run, device='cuda:0'):
    steps=6 if sampler=='ddim' else 50
    n=LIMITS[(sampler,steps)]
    specs=[ExtensionNoiseSpec(sampler,steps,n,seed) for seed in SEEDS]
    grid,grid_receipt=read_stage(grid_run,'grid')
    ref,ref_receipt=read_stage(reference_noise_run,'noise-check')
    main,main_receipt=read_stage(main_run,'grid-extension')
    selected,_,selection=frozen_selection(selection_run)
    sources=json.loads((selected/'sources.json').read_text())
    if (sources['grid_receipt_sha256']!=sha256(grid/'receipt.json') or
            sources[sampler+'_receipt_sha256']!=sha256(main/'receipt.json') or
            main_receipt['spec']!=ExtensionSpec(sampler,steps,n).identity()):
        raise ValueError('冻结选择与主采样/首轮来源不匹配')
    for path,r in ((grid,grid_receipt),(ref,ref_receipt),(main,main_receipt)):
        for name in ('val_parents.csv','source_identity.json'):verify_artifact(path,r,name)
        if (r['checkpoint']['checkpoint_sha256']!=CHECKPOINT or
                (path/'val_parents.csv').read_bytes()!=(grid/'val_parents.csv').read_bytes()):
            raise ValueError('噪声核验的模型或 parent 顺序改变')
        # 所有既有数值实现保持冻结；本轮只新增外层固定噪声编排。
        for name,h in json.loads((path/'source_identity.json').read_text())['files'].items():
            if name.startswith(('resp_train/','scripts/','configs/')) and sha256(ROOT/name)!=h:
                raise ValueError(f'来源数值实现发生变化：{name}')
    verify_artifact(grid,grid_receipt,'grid_identity.json')
    previous=json.loads((grid/'grid_identity.json').read_text())
    if previous['batch_size']!=64 or previous['trajectory_group_size']!=1:
        raise ValueError('必须复用原 B64/G1 身份')
    reference_names={seed:f'{sampler}_S{steps}_seed{seed}_N{n//2}/validation_postfiltered_waveforms.npz' for seed in SEEDS}
    for name in reference_names.values():verify_artifact(ref,ref_receipt,name)
    frozen={'selection_receipt_sha256':sha256(selected/'receipt.json'),
        'grid_receipt_sha256':sha256(grid/'receipt.json'),'reference_noise_receipt_sha256':sha256(ref/'receipt.json'),
        'main_receipt_sha256':sha256(main/'receipt.json')}
    with study_run(output,'extension-noise-check',synthetic=False,source=source_run) as (out,receipt):
        model,cfg,checkpoint,bundles,_,index=_initialize(out,'extension-noise-check',source_run,device,
            tiny=False,expected_sha256=CHECKPOINT)
        parents,rows,_=bundles['val']
        dataset=selected_dataset(parents,rows,previous['source_parent_indices'])
        if dataset.rows.to_csv(index=False).encode()!=(grid/'val_parents.csv').read_bytes():
            raise ValueError('当前父集合/顺序与冻结来源不一致')
        identity=noise_identity(cfg)
        OmegaConf.save(cfg,out/'resolved_config.yaml')
        write_json(out/'source_identity.json',identity);save_source_snapshot(out,identity)
        write_json(out/'environment.json',environment(cpu_only=False))
        write_json(out/'selection.json',selection)
        write_json(out/'sources.json',frozen)
        write_json(out/'execution.json',{'revision':REVISION,'checkpoint':checkpoint,
            'specs':[spec.identity() for spec in specs],'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'logical_device':device,'selection_run':str(selected),'reference_noise_run':str(ref)})
        data=data_identity(out,{'val':dataset},index)
        rule_ids=rows.iloc[np.linspace(0,len(rows)-1,64,dtype=int)].dataset_row_id.astype(int).tolist()
        gpu=torch.device(device);properties=torch.cuda.get_device_properties(gpu)
        curves,profiles,memories,prefixes=[],[],[],[]
        with preserved_evaluation(model):
            for spec in specs:
                seed=spec.noise_seed
                sampling=out/f'sampling_seed{seed}';sampling.mkdir()
                torch.cuda.synchronize(gpu);torch.cuda.reset_peak_memory_stats(gpu)
                predictions,profile=predict_prefixes(model,dataset,spec,(n//2,n),sampling,
                    batch_size=64,acceleration=AccelerationSpec(1))
                torch.cuda.synchronize(gpu)
                memories.append({'sampling_seed':seed,'peak_allocated_bytes':torch.cuda.max_memory_allocated(gpu),
                    'peak_reserved_bytes':torch.cuda.max_memory_reserved(gpu),'total_memory_bytes':properties.total_memory,
                    'device_name':properties.name,'batch_size':64,'trajectory_group_size':1})
                write_json(out/'memory_profile.json',memories)
                prefix=compare_prefix(predictions.pop(n//2),ref/reference_names[seed],out/f'prefix_seed{seed}.json')
                prefixes.append({'sampling_seed':seed,**prefix})
                setting=out/f'{sampler}_S{steps}_seed{seed}_N{n}'
                curve=evaluate_setting(predictions.pop(n),dataset,cfg,setting,spec,n,
                    method_prefix=f'RespDiff-study-v1/{REVISION}')
                curve.update(rule_sample_summary(setting,rule_ids))
                curve.update({'training_seed':20260811,'sampling_seed':seed,'checkpoint_sha256':CHECKPOINT})
                curves.append(curve);profiles.extend({**p,'sampling_seed':seed} for p in profile)
                pd.DataFrame(curves).to_csv(out/'sampler_grid.csv',index=False)
                pd.DataFrame(profiles).to_csv(out/'runtime_profile.csv',index=False)
                LOGGER.info('%s N=%d sampling seed=%d 核验完成，prefix exact=%s',sampler,n,seed,prefix['exact_equal'])
        verify_data_unchanged(data,dataset.rows);_check_source(checkpoint,source_run,model)
        if noise_identity(cfg)!=identity:raise ValueError('核验期间源码或协议变化')
        receipt.update({'revision':REVISION,'checkpoint':checkpoint,'specs':[s.identity() for s in specs],
            'sources':frozen,'prefix_verifications':prefixes,'parameter_sha256_after':state_digest(model)})


def validate_sensitivity(curve):
    expected={(kind,s,n,seed) for kind,s,n in [('ddim',6,64),('ddpm',50,16),('ddim',6,16)]
              for seed in (20261003,*SEEDS)}
    if len(curve)!=9 or set(zip(curve.sampler,curve.nfe_per_trajectory,curve.N,curve.sampling_seed))!=expected:
        raise ValueError('敏感性汇总要求完整九项：三个配置 × 三个 sampling seed')
    if (set(curve.checkpoint_sha256)!={CHECKPOINT} or set(curve.training_seed)!={20260811}
            or set(curve.n_parents)!={66} or not np.isfinite(curve[list(METRICS)].to_numpy()).all()):
        raise ValueError('敏感性曲线的身份或指标不合法')
    return [{'sampler':kind,'S':int(s),'N':int(n),'metric':metric,
             'min':float(group[metric].min()),'max':float(group[metric].max()),
             'mean':float(group[metric].mean()),'std_sampling_seeds':float(group[metric].std(ddof=1))}
            for (kind,s,n),group in curve.groupby(['sampler','nfe_per_trajectory','N']) for metric in METRICS]


def summarize_noise(output, *, selection_run, grid_run, reference_noise_run, ddim_run, ddpm_run):
    selected,_,selection=frozen_selection(selection_run)
    grid,gr=read_stage(grid_run,'grid');ref,rr=read_stage(reference_noise_run,'noise-check')
    for path,r in ((grid,gr),(ref,rr)):
        verify_artifact(path,r,'val_parents.csv')
        if (r['checkpoint']['checkpoint_sha256']!=CHECKPOINT or
                (path/'val_parents.csv').read_bytes()!=(grid/'val_parents.csv').read_bytes()):
            raise ValueError('历史参照的模型或 parent 顺序不匹配')
    source_hashes={'selection_receipt_sha256':sha256(selected/'receipt.json'),
        'grid_receipt_sha256':sha256(grid/'receipt.json'),'reference_noise_receipt_sha256':sha256(ref/'receipt.json')}
    original_sources=json.loads((selected/'sources.json').read_text())
    if original_sources['grid_receipt_sha256']!=source_hashes['grid_receipt_sha256']:
        raise ValueError('主选择与首轮网格来源不同')
    main_curve=pd.read_csv(selected/'sampler_grid.csv')
    frames=[main_curve[((main_curve.sampler=='ddim')&(main_curve.nfe_per_trajectory==6)&main_curve.N.isin([16,64]))|
                       ((main_curve.sampler=='ddpm')&(main_curve.N==16))]]
    verify_artifact(ref,rr,'sampler_grid.csv');old_noise=pd.read_csv(ref/'sampler_grid.csv')
    frames.append(old_noise[(old_noise.sampler=='ddim')&(old_noise.nfe_per_trajectory==6)&(old_noise.N==16)])
    for kind,path in [('ddim',ddim_run),('ddpm',ddpm_run)]:
        path,r=read_stage(path,'extension-noise-check')
        for name in ('sampler_grid.csv','val_parents.csv'):verify_artifact(path,r,name)
        s=6 if kind=='ddim' else 50
        expected=[ExtensionNoiseSpec(kind,s,LIMITS[(kind,s)],seed).identity() for seed in SEEDS]
        if (r.get('revision')!=REVISION or r['specs']!=expected or
                any(r['sources'][k]!=v for k,v in source_hashes.items()) or
                r['sources']['main_receipt_sha256']!=original_sources[kind+'_receipt_sha256'] or
                r['checkpoint']['checkpoint_sha256']!=CHECKPOINT or
                r['checkpoint']['parameter_sha256']!=r['parameter_sha256_after'] or
                len(r['prefix_verifications'])!=2 or not all(p['passed'] for p in r['prefix_verifications']) or
                (path/'val_parents.csv').read_bytes()!=(grid/'val_parents.csv').read_bytes()):
            raise ValueError('噪声核验两臂的身份或前缀验证不一致')
        frames.append(pd.read_csv(path/'sampler_grid.csv'))
    curve=pd.concat(frames,ignore_index=True);ranges=validate_sensitivity(curve)
    with study_run(output,'summarize-extension-noise',synthetic=False) as (out,receipt):
        curve.to_csv(out/'sensitivity.csv',index=False)
        pd.DataFrame(ranges).to_csv(out/'noise_ranges.csv',index=False)
        write_json(out/'selection.json',selection)
        write_json(out/'sources.json',{**source_hashes,'ddim_run':str(ddim_run),'ddpm_run':str(ddpm_run)})
        receipt.update({'revision':REVISION,'selection_changed':False,'configurations':9})
