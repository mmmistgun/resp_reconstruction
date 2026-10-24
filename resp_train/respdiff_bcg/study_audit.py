"""P2-A1：固定 checkpoint 的损失梯度、数值一致性和 batch 上下文诊断。"""

import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch

from resp_train.respdiff.model import finite
from .baseband import make_baseband_model
from .inference_v2 import state_digest
from .runtime import LOGGER, environment, save_source_snapshot, seed_all, sha256, source_file_inventory, write_json
from .sampling import sample_trajectory
from .sampling_accelerated import encode_condition, forward_cached, sample_group
from .study_contract import PROBE_SEEDS, PROBE_STEPS, StudySamplerSpec
from .study_diagnostics import chunk_keys, forward_noise
from .study_io import (ROOT, implementation_identity, selected_dataset, source_data,
                       study_run, validate_training_config, verify_artifact, verify_data_unchanged)
from .study_state import finite_tree

AUDIT_ID = 'respdiff-bcg-p2-audit-a1-20261011'
SCRIPT = 'scripts/audit_respdiff_bcg_training_sampling_v1.py'
CHECKPOINTS = (
    ('baseline_6400', 'baseline', 'final.pt', 6400,
     '08713b693ca709a659eaa29898f689814f32651136a42ce2fe61223cffb8f6f6'),
    ('p2_1600', 'p2', 'checkpoints/update_001600.pt', 1600,
     'dca9d5e4226f7aaf6cf99559d08cfbb7bb74b9d38acc1de4403cb8320149e4cc'),
    ('p2_3200', 'p2', 'final.pt', 3200,
     'b4f4f3a8af56efd6942e120cec75a488ea3f06688c0ae568d89e3aea1021c47d'),
)
GROUPS = ('condition', 'noise_encoder', 'time_embedding', 'rnn', 'decoder')


def comparison(name, actual, reference, *, atol=2e-4, rtol=1e-3):
    finite(name, actual)
    finite(name + ' reference', reference)
    if actual.shape != reference.shape:
        raise ValueError(f'{name} shape 不匹配')
    a, b = actual.detach().double(), reference.detach().double()
    error = (a-b).abs()
    return {'check': name, 'max_abs_error': float(error.max()),
            'rms_error': float(error.square().mean().sqrt()),
            'max_tolerance_ratio': float((error/(atol+rtol*b.abs())).max()),
            'passed': bool(torch.all(error <= atol+rtol*b.abs())), 'atol': atol, 'rtol': rtol}


def _group(name):
    if name == 'diffusion_model.weight' or '.ppg_encoder' in name:
        return 'condition'
    for prefix, group in (('noise_encoder', 'noise_encoder'), ('diffusion_embedding', 'time_embedding'),
                          ('diff_model', 'rnn'), ('de', 'decoder')):
        if name.startswith('diffusion_model.' + prefix + '.'):
            return group
    raise ValueError(f'未声明的参数组：{name}')


def gradient_metrics(named_parameters, noise_gradients, spectral_gradients):
    """报告原始参数梯度，不将 L2 范数冒充 Adam 实际更新贡献。"""
    sums = {name: np.zeros(5, dtype=np.float64) for name in ('all', *GROUPS)}
    for (name, _), noise, spectral in zip(named_parameters, noise_gradients, spectral_gradients, strict=True):
        finite('noise gradient ' + name, noise)
        finite('spectral gradient ' + name, spectral)
        n, s = noise.detach().double(), spectral.detach().double()
        values = np.array([float(n.square().sum()), float(s.square().sum()), float((n*s).sum()),
                           float((n+s).square().sum()), n.numel()])
        sums['all'] += values
        sums[_group(name)] += values
    rows = []
    for group, (n2, s2, dot, total2, count) in sums.items():
        n, s, total = np.sqrt(n2), np.sqrt(s2), np.sqrt(total2)
        rows.append({'parameter_group': group, 'n_parameters': int(count),
            'noise_gradient_l2': float(n), 'weighted_spectral_gradient_l2': float(s),
            'total_gradient_l2': float(total), 'gradient_dot': float(dot),
            'spectral_over_noise_gradient_l2': float(s/n) if n else None,
            'gradient_cosine': float(dot/(n*s)) if n and s else None,
            'total_vs_noise_cosine': float((n2+dot)/(total*n)) if total and n else None,
            'zero_noise_norm': bool(n == 0), 'zero_spectral_norm': bool(s == 0),
            'gradient_space': 'raw_parameter_gradient_before_Adam'})
    return rows


def posterior_reference(model, current, predicted, step, reverse_noise):
    """用独立的 posterior x0/xt 系数形式及 FP64 验证 DDPM 均值、方差。"""
    alpha = model.alpha[step].double()
    previous = model.alpha[step-1].double() if step else torch.ones_like(alpha)
    beta = model.beta[step].double()
    alpha_hat = model.alpha_hat[step].double()
    x0 = (current.double()-(1-alpha).sqrt()*predicted.double())/alpha.sqrt()
    result = beta*previous.sqrt()/(1-alpha)*x0
    result += alpha_hat.sqrt()*(1-previous)/(1-alpha)*current.double()
    if step:
        result += (beta*(1-previous)/(1-alpha)).sqrt()*reverse_noise.double()
    return result.to(current.dtype)


def gradient_case(model, condition, target, noise, timestep):
    """从生产 training_loss 捕获预测，分离两个真实损失项的梯度。"""
    if not torch.is_grad_enabled():
        raise ValueError('梯度审计不能处于 no_grad/inference_mode')
    if model.objective != 'source_equivalent':
        raise ValueError('P2-A1 仅核对既有 source-equivalent 目标')
    model.train()  # cuDNN RNN backward 必须保留训练前向的 reserve。
    steps = torch.full((len(target),), timestep, dtype=torch.long, device=target.device)
    captured = {}
    def capture(module, args, prediction):
        captured.update(xt=args[1], prediction=prediction)
    hook = model.diffusion_model.register_forward_hook(capture)
    try:
        losses = model.training_loss(condition, target, step=steps, noise=noise)
    finally:
        hook.remove()
    predicted, xt = captured['prediction'], captured['xt']
    alpha = model.alpha_torch[steps]
    weight = .01/128
    # 直接 x0→FFT 是独立表达；不把它替换到实际训练目标中。
    independent_prediction = predicted.detach().clone().requires_grad_(True)
    x0 = (xt.detach()-(1-alpha).sqrt()*independent_prediction)/alpha.sqrt()
    direct_fft = (torch.fft.fft(x0, norm='ortho').abs()
                  - torch.fft.fft(target, norm='ortho').abs()).square().mean()
    direct_noise = (noise-independent_prediction).square().mean()
    direct_weighted = weight*direct_fft
    production_output_gradient = torch.autograd.grad(losses['loss_spec_weighted'], predicted, retain_graph=True)[0]
    direct_output_gradient = torch.autograd.grad(direct_weighted, independent_prediction)[0]
    checks = [comparison('noise_loss_formula', losses['loss_noise'], direct_noise, atol=1e-6),
              comparison('spectral_loss_formula', losses['loss_spec_weighted'], direct_weighted, atol=1e-6),
              comparison('spectral_output_gradient', production_output_gradient, direct_output_gradient, atol=1e-6)]
    named = list(model.named_parameters())
    parameters = [p for _, p in named]
    noise_gradients = torch.autograd.grad(losses['loss_noise'], parameters, retain_graph=True)
    spectral_gradients = torch.autograd.grad(losses['loss_spec_weighted'], parameters)
    groups = gradient_metrics(named, noise_gradients, spectral_gradients)
    gain = float(((1-alpha[0])/alpha[0]).sqrt())
    epsilon_mse = float(losses['loss_noise'].detach())
    record = {'loss_noise': epsilon_mse, 'loss_spectral_weighted': float(losses['loss_spec_weighted'].detach()),
        'loss_total': float(losses['loss'].detach()), 'x0_error_gain': gain,
        'spectral_residual_coefficient': weight*gain*gain,
        'pooled_x0_rmse': gain*np.sqrt(epsilon_mse),
        'target_rms': float(target.double().square().mean().sqrt())}
    del noise_gradients, spectral_gradients, production_output_gradient, direct_output_gradient
    model.eval()
    with torch.no_grad():
        evaluation = model.diffusion_model(condition, xt.detach(), steps)
        cache = encode_condition(model.diffusion_model, condition)
        cached = forward_cached(model.diffusion_model, condition, cache, xt.detach(), steps)
        checks.extend([comparison('train_vs_eval_epsilon', predicted, evaluation),
                       comparison('cache_G1_vs_eval_epsilon', cached, evaluation)])
        # 固定非零 z 只用于代数核验，不作为采样噪声或生成质量证据。
        reverse_noise = torch.full_like(xt, .37)
        actual = model.reverse_step(xt.detach(), evaluation, step=timestep, noise=reverse_noise)
        reference = posterior_reference(model, xt.detach(), evaluation, timestep, reverse_noise)
        checks.append(comparison('ddpm_posterior_formula', actual, reference, atol=.02))
    return record, groups, checks, evaluation.detach().cpu()


@torch.inference_mode()
def sampling_parity(model, condition, keys, spec):
    """对同一 N1 轨迹逐步比较实际串行入口和实际缓存 G1 入口。"""
    model.eval()
    reference, checks = [], []
    def capture(module, args, output):
        reference.append({'step': int(args[2][0]), 'current': args[1].detach().clone(),
                          'epsilon': output.detach().clone()})
    hook = model.diffusion_model.register_forward_hook(capture)
    try:
        serial = sample_trajectory(model, condition, keys, spec, 0)
    finally:
        hook.remove()
    if [row['step'] for row in reference] != list(spec.timesteps):
        raise ValueError('串行采样实际 timestep 序列不匹配')
    def observe(event):
        i = len(checks)//3
        row = reference[i]
        if event['timestep'] != row['step'] or tuple(event['trajectory_ids']) != (0,):
            raise ValueError('缓存采样轨迹身份不匹配')
        following = reference[i+1]['current'] if i+1 < len(reference) else serial
        for name, actual, expected, atol in (
            ('sampler_current', event['current'][0], row['current'], .02),
            ('sampler_epsilon', event['epsilon'][0], row['epsilon'], 2e-4),
            ('sampler_reverse', event['reverse'][0], following, .02)):
            checks.append({'timestep': row['step'], **comparison(name, actual, expected, atol=atol)})
    cache = encode_condition(model.diffusion_model, condition)
    grouped = sample_group(model, condition, keys, spec, (0,), cache, observer=observe)[0]
    checks.append({'timestep': -1, **comparison('sampler_final', grouped, serial, atol=.02)})
    return checks


@torch.inference_mode()
def context_predictions(model, condition, target, keys, order, *, batch_size=64, timesteps=(19,39,49)):
    """保留每个 chunk 的输入与噪声身份，仅改变伙伴及 B64/B40 分组。"""
    if sorted(order) != list(range(len(keys))):
        raise ValueError('batch 上下文顺序必须是完整双射')
    model.eval()
    result = {t: torch.empty_like(target, device='cpu') for t in timesteps}
    coarse = torch.empty(len(keys),192,target.shape[-1],dtype=target.dtype)
    fine = torch.empty_like(coarse)
    for start in range(0,len(keys),batch_size):
        ids = order[start:start+batch_size]
        x, y = condition[ids], target[ids]
        cache = encode_condition(model.diffusion_model,x)
        fine[ids], coarse[ids] = cache.fine.cpu(), cache.weighted_coarse.cpu()
        noise = forward_noise([keys[i] for i in ids],PROBE_SEEDS[0],x.device)
        for t in timesteps:
            alpha=model.alpha_torch[t]
            xt=alpha.sqrt()*y+(1-alpha).sqrt()*noise
            steps=torch.full((len(ids),),t,dtype=torch.long,device=x.device)
            result[t][ids]=forward_cached(model.diffusion_model,x,cache,xt,steps).cpu()
    return result, fine, coarse


def context_metrics(reference, altered, fine_reference, fine_altered, coarse_reference, coarse_altered, target):
    rows=[]
    def per_chunk_rms(tensor):
        return tensor.double().flatten(1).square().mean(1).sqrt()
    fine_error=per_chunk_rms(fine_altered-fine_reference)
    coarse_error=per_chunk_rms(coarse_altered-coarse_reference)
    for t in reference:
        error=per_chunk_rms(altered[t]-reference[t])
        for i in range(len(target)):
            rows.append({'timestep':t,'global_chunk_index':i,'epsilon_context_rms':float(error[i]),
                         'fine_condition_context_rms':float(fine_error[i]),
                         'coarse_condition_context_rms':float(coarse_error[i])})
    return rows


def audit_identity(cfg):
    identity=implementation_identity(cfg)
    identity['files'][SCRIPT]=sha256(ROOT/SCRIPT)
    return identity


def _load_checkpoint(model, run, filename, expected_hash, update, role):
    receipt=json.loads((run/'receipt.json').read_text())
    if receipt.get('status') != 'complete' or not receipt.get('formal_evidence'):
        raise ValueError('审计来源必须是已完成的真实运行')
    verify_artifact(run,receipt,filename)
    if receipt['artifacts'][filename] != expected_hash:
        raise ValueError('checkpoint 偏离 P2-A1 固定哈希')
    state=torch.load(run/filename,map_location='cpu',weights_only=False)
    finite_tree(state['model'])
    if state['update'] != update or state['model_spec'] != vars(model.spec):
        raise ValueError('checkpoint update/model_spec 不匹配')
    if role == 'p2':
        sidecar=str(Path(filename).with_suffix('.identity.json'))
        verify_artifact(run,receipt,sidecar)
        if (receipt.get('phase') != 'overfit' or state['training']['regime'] != 'overfit'
                or state['training_data_sha256'] != sha256(run/'actual_training_parents.csv')):
            raise ValueError('P2 checkpoint 训练身份不匹配')
        if state['selector'] != ('final_update' if update==3200 else 'diagnostic_update'):
            raise ValueError('P2 checkpoint selector 不匹配')
    elif receipt.get('objective') != 'source_equivalent' or state['selector'] != 'final_update':
        raise ValueError('baseline 来源身份不匹配')
    model.load_state_dict(state['model'],strict=True)
    del state
    return state_digest(model)


def run_audit(output, baseline_run, p2_run, *, device='cuda:0'):
    baseline_run, p2_run=Path(baseline_run).resolve(),Path(p2_run).resolve()
    with study_run(output,'training-sampling-audit',synthetic=False,source=p2_run) as (out,receipt):
        seed_all(20261011)
        source_receipt=json.loads((p2_run/'receipt.json').read_text())
        if source_receipt.get('status') != 'complete' or source_receipt.get('phase') != 'overfit':
            raise ValueError('P2 尚未完成')
        for name in ('resolved_config.yaml','actual_training_parents.csv','actual_training_chunks.csv'):
            verify_artifact(p2_run,source_receipt,name)
        baseline_receipt=json.loads((baseline_run/'receipt.json').read_text())
        if (baseline_receipt.get('status')!='complete' or not baseline_receipt.get('formal_evidence')
                or baseline_receipt.get('objective')!='source_equivalent'):
            raise ValueError('历史来源须为完整 source-equivalent 运行')
        for name in ('resolved_config.yaml','train_parents.csv','val_parents.csv','data_identity.json'):
            verify_artifact(baseline_run,baseline_receipt,name)
        cfg=OmegaConf.load(p2_run/'resolved_config.yaml')
        if validate_training_config(cfg).regime != 'overfit':
            raise ValueError('要求已冻结 P2 配置')
        identity=audit_identity(cfg)
        OmegaConf.save(cfg,out/'resolved_config.yaml')
        write_json(out/'source_identity.json',identity)
        save_source_snapshot(out,identity)
        write_json(out/'environment.json',environment(cpu_only=False))
        budget={'audit_id':AUDIT_ID,'checkpoint_labels':[row[0] for row in CHECKPOINTS],
            'n_parents':8,'n_chunks':104,'batch_sizes':[64,40], 'timesteps':list(PROBE_STEPS),
            'forward_noise_seeds':list(PROBE_SEEDS),'gradient_cases':108,'network_backward_calls':216,
            'full_trajectory_pairs':12,'trajectory_N':1,'samplers':['DDIM6','DDPM50'],
            'optimizer_updates':0,'rebatch_seed':20261030,'context_timesteps':[19,39,49],
            'source_runs':{'baseline':str(baseline_run),'p2':str(p2_run)}}
        write_json(out/'execution.json',{'device':device,**budget})
        LOGGER.info('P2-A1 固定预算：%s',budget)
        bundles=source_data(cfg,baseline_run,mapping_output=out/'source_path_mapping.json')
        parents,rows,index=bundles['train']
        all_rows=pd.concat([bundle[1] for bundle in bundles.values()],ignore_index=True)
        data={'index_path':str(index),'index_sha256':sha256(index),'source_file_stats':source_file_inventory(all_rows)}
        write_json(out/'data_identity.json',data)
        selected=pd.read_csv(p2_run/'actual_training_parents.csv')
        indices=[]
        for row in selected.dataset_row_id:
            found=rows.index[rows.dataset_row_id==row].tolist()
            if len(found)!=1:
                raise ValueError('P2 parent 未唯一匹配来源')
            indices.extend(found)
        dataset=selected_dataset(parents,rows,indices)
        columns=['dataset_row_id','split','samp_id','source_npz','target_source_npz',
                 'window_start_sample','window_end_sample','bcg_signal_key','target_signal_key']
        if len(dataset)!=104 or not dataset.rows[columns].equals(selected[columns]):
            raise ValueError('实际审计输入偏离 P2 的八个 parent')
        dataset.rows.to_csv(out/'actual_parents.csv',index=False)
        dataset.manifest().to_csv(out/'actual_chunks.csv',index=False)
        items=[dataset[i] for i in range(len(dataset))]
        condition=torch.stack([item['x'] for item in items]).to(device)
        target=torch.stack([item['target'] for item in items]).to(device)
        del items
        np.savez_compressed(out/'fixed_inputs.npz',condition=condition.cpu().numpy(),target=target.cpu().numpy())
        keys=chunk_keys(dataset,list(range(len(dataset))))
        order=np.random.default_rng(20261030).permutation(len(dataset)).tolist()
        write_json(out/'batch_context.json',{'ordered':list(range(len(dataset))),'shuffled':order,
                   'batch_sizes':[64,40],'sampling_noise_seed':20261003})
        model=make_baseband_model(cfg).to(device)
        torch.cuda.reset_peak_memory_stats(torch.device(device))
        gradients,groups,checks,contexts,states=[],[],[],[],[]
        def flush():
            for filename,records in (('gradient_per_case.csv',gradients),('gradient_per_group.csv',groups),
                                      ('consistency_checks.csv',checks),('batch_context_metrics.csv',contexts)):
                if records:
                    pd.DataFrame(records).to_csv(out/filename,index=False)
        for label,role,filename,update,digest in CHECKPOINTS:
            source=baseline_run if role=='baseline' else p2_run
            before=_load_checkpoint(model,source,filename,digest,update,role)
            saved_probe=None
            if role=='p2':
                probe_name=f'monitor/update_{update:06d}/train_representative/forward/timestep_metrics.csv'
                verify_artifact(p2_run,source_receipt,probe_name)
                saved_probe=pd.read_csv(p2_run/probe_name)
            LOGGER.info('开始 checkpoint %s',label)
            for batch,start in enumerate((0,64)):
                x,y=condition[start:start+64],target[start:start+64]
                batch_keys=keys[start:start+64]
                for seed in PROBE_SEEDS:
                    noise=forward_noise(batch_keys,seed,x.device)
                    for t in PROBE_STEPS:
                        meta={'checkpoint':label,'update':update,'batch':batch,'batch_size':len(x),
                              'timestep':t,'noise_seed':seed}
                        row,group_rows,case_checks,_=gradient_case(model,x,y,noise,t)
                        if saved_probe is not None:
                            matched=saved_probe[(saved_probe.batch==batch)&(saved_probe.noise_seed==seed)
                                                &(saved_probe.timestep==t)&(saved_probe.view=='correct')]
                            if len(matched)!=len(x):
                                raise ValueError('保存 probe 与当前 batch 身份不匹配')
                            case_checks.append(comparison('saved_P2_probe_epsilon_mse',
                                torch.tensor(row['loss_noise']),torch.tensor(float(matched.epsilon_mse.mean())),atol=1e-6))
                        gradients.append({**meta,**row,**group_rows[0]})
                        groups.extend({**meta,**g} for g in group_rows)
                        checks.extend({**meta,**check} for check in case_checks)
                        flush()
                    LOGGER.info('%s 梯度：batch=%d seed=%d 已完成六个 t',label,batch,seed)
                for kind,steps in (('ddim',6),('ddpm',50)):
                    spec=StudySamplerSpec(kind,steps,1)
                    checks.extend({'checkpoint':label,'batch':batch,'batch_size':len(x),'sampler':kind,
                                   **check} for check in sampling_parity(model,x,batch_keys,spec))
                    flush()
                    LOGGER.info('%s 采样一致性：batch=%d %s N1 完成',label,batch,kind)
            reference,fine,coarse=context_predictions(model,condition,target,keys,list(range(len(keys))))
            altered,fine_changed,coarse_changed=context_predictions(model,condition,target,keys,order)
            for row in context_metrics(reference,altered,fine,fine_changed,coarse,coarse_changed,target):
                i=row['global_chunk_index']; t=row['timestep']; gain=float(((1-model.alpha_torch[t])/model.alpha_torch[t]).sqrt())
                contexts.append({'checkpoint':label,'dataset_row_id':keys[i][1],'chunk_index':keys[i][2],
                                 'x0_context_rms':row['epsilon_context_rms']*gain,**row})
            del reference,altered,fine,coarse,fine_changed,coarse_changed
            after=state_digest(model)
            if before!=after or any(p.grad is not None for p in model.parameters()):
                raise ValueError('梯度审计改变模型状态或残留参数 .grad')
            if sha256(source/filename)!=digest:
                raise ValueError('审计期间 checkpoint 字节变化')
            states.append({'checkpoint':label,'sha256':digest,'parameter_sha256_before':before,'parameter_sha256_after':after})
            flush()
            LOGGER.info('checkpoint %s 审计完成，状态未改变',label)
        if audit_identity(cfg)!=identity:
            raise ValueError('审计期间代码或计划变化')
        verify_data_unchanged(data,all_rows)
        write_json(out/'checkpoint_states.json',states)
        summary={'audit_id':AUDIT_ID,'gradient_cases':len(gradients),'consistency_checks':len(checks),
            'failed_consistency_checks':sum(not item['passed'] for item in checks),
            'context_rows':len(contexts),'optimizer_updates':0,'scientific_interpretation':'pending_review',
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(torch.device(device)),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(torch.device(device))}
        write_json(out/'summary.json',summary)
        if summary['gradient_cases']!=108 or summary['context_rows']!=936:
            raise ValueError('审计矩阵不完整')
        if summary['failed_consistency_checks']:
            raise ValueError('存在超出预定容差的一致性检查，详见 consistency_checks.csv')
        receipt.update(summary)
