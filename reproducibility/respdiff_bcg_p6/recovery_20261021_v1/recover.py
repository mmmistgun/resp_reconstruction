"""P6 恢复：核验完整 DDIM 配置，原采样内核逐批持久化。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import default_collate
from resp_train.metrics.task import summarize_task_metrics
from resp_train.respdiff_bcg import research_test as rt
from resp_train.respdiff_bcg.inference_v2 import evaluate_setting, state_digest
from resp_train.respdiff_bcg.runtime import LOGGER, Progress, environment, sha256, write_json, save_source_snapshot, seed_all
from resp_train.respdiff_bcg.sampling_accelerated import AccelerationSpec, accelerated_prefixes
from resp_train.respdiff_bcg.study_evaluation import independent_timing
from resp_train.respdiff_bcg.study_io import data_identity, load_source, study_run, verify_data_unchanged
from resp_train.respdiff_bcg.study_state import preserved_evaluation
from omegaconf import OmegaConf

REVISION = 'P6-recovery-batch-cache-v1-20261021'
STUDY = ROOT / rt.STUDY
MANIFEST = STUDY / 'manifest_v1'
PREFLIGHT = STUDY / 'remote_preflight_v1'
SEEDS = tuple(rt.CHECKPOINTS)


def exclusive_json(path, value):
    """数据先落盘，再以原子提交的 JSON 标记该批可复用。"""
    pending = path.with_suffix(path.suffix + '.pending')
    with pending.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    os.link(pending, path)
    pending.unlink()
    fd = os.open(path.parent, os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def audit(output):
    manifest = rt.read_manifest(MANIFEST)
    report = {'revision': REVISION, 'manifest_sha256': sha256(MANIFEST / 'research_test_manifest.json'),
              'program_sha256': sha256(Path(__file__)), 'seeds': {}}
    for seed in SEEDS:
        source = STUDY / f'test_s{seed}_v1'
        if (source / 'receipt.json').exists():
            raise ValueError('原运行出现回执，应先重新判定完成状态')
        if sha256(source / 'research_test_manifest.json') != report['manifest_sha256']:
            raise ValueError('原运行 manifest 不匹配')
        execution = json.loads((source / 'execution.json').read_text())
        cp = execution['checkpoint']
        expected = next(row for row in manifest['models'] if row['training_seed'] == seed)
        if {k: v for k, v in cp.items() if k != 'source_run'} != expected:
            raise ValueError('原运行模型身份不匹配')
        if sha256(source / 'test_canonical_parents.csv') != manifest['splits']['test']['canonical_rows_sha256']:
            raise ValueError('原运行测试集合不匹配')
        original = json.loads((source / 'source_identity.json').read_text())
        if original['files'] != manifest['files']:
            raise ValueError('原执行源码身份不匹配')
        inventory = {str(p.relative_to(source)): sha256(p) for p in sorted(source.rglob('*')) if p.is_file()}
        reused = []
        grid_path = source / 'sampler_grid.csv'
        if grid_path.exists():
            grid = pd.read_csv(grid_path)
            if len(grid) != 2 or set(zip(grid.sampler, grid.N)) != {('ddim', 16), ('ddim', 64)}:
                raise ValueError('中断配置状态不同于预期，停止自动恢复')
            sampling = json.loads((source / 'ddim_sampling/sampling_identity.json').read_text())
            if (sampling['ordered_chunk_indices'] != list(range(manifest['n_chunks']))
                    or sampling['counts'] != [16, 64] or sampling['spec'] != vars(rt.ResearchTestSpec('ddim', 6, 64))):
                raise ValueError('DDIM 采样身份或完整性不符')
            parents = pd.read_csv(source / 'test_parents.csv')
            for n in (16, 64):
                setting = source / f'ddim_S6_N{n}'
                saved = json.loads((setting / 'setting_summary.json').read_text())
                row = grid[grid.N == n].iloc[0]
                if (row.training_seed != seed or row.sampling_seed != 20261003 or
                        row.checkpoint_sha256 != rt.CHECKPOINTS[seed] or saved['n_parents'] != manifest['n_parents']):
                    raise ValueError('DDIM 配置来源不符')
                for key, value in saved.items():
                    if isinstance(value, (int, float)):
                        if not np.isfinite(value) or not np.isclose(row[key], value, rtol=1e-12, atol=1e-12):
                            raise ValueError(f'DDIM 汇总字段不匹配：{key}')
                for view in ('raw', 'postfiltered'):
                    prefix = f'research_test_{view}'
                    with np.load(setting / f'{prefix}_waveforms.npz', allow_pickle=False) as data:
                        if (set(data['split']) != {'test'} or not np.array_equal(data['dataset_row_id'], parents.dataset_row_id)
                                or not np.array_equal(data['samp_id'], parents.samp_id)):
                            raise ValueError('已保存波形的 parent 身份不匹配')
                        for key in (f'prediction_{view}', 'tho_ref'):
                            values = data[key]
                            if values.shape != (manifest['n_parents'], 18000) or not np.isfinite(values).all():
                                raise ValueError('已保存波形不完整或非有限')
                    metrics = pd.read_csv(setting / f'{prefix}_per_parent.csv')
                    if len(metrics) != len(parents) or not metrics.dataset_row_id.equals(parents.dataset_row_id) or set(metrics.split) != {'test'}:
                        raise ValueError('已保存逐 parent 指标不完整')
                    summary = summarize_task_metrics(metrics).iloc[0]
                    for metric in ('whole_rr_abs_error_bpm', 'local_rr_mae_bpm', 'envelope_trajectory_mae',
                                   'global_envelope_modulation_error', 'lag_aware_signed_pcc'):
                        if not np.isfinite(metrics[metric]).all() or not np.isclose(summary[f'{metric}_mean'], saved[f'{view}_{metric}'], rtol=1e-12, atol=1e-12):
                            raise ValueError('已保存任务指标不匹配')
                reused.append(n)
        report['seeds'][str(seed)] = {'source': str(source), 'reused_ddim_counts': reused,
            'original_files': inventory, 'original_receipt_present': False}
        LOGGER.info('恢复核验 seed%d：完整 DDIM 配置=%s，文件=%d', seed, reused, len(inventory))
    with study_run(output, 'research-test-recovery-audit', synthetic=False) as (out, receipt):
        receipt['research_test_protocol'] = rt.PROTOCOL
        write_json(out / 'recovery_audit.json', report)


def cache_key(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def load_cached_batch(cache, identity, start, stop, counts):
    marker = cache / f'batch_{start:06d}.json'
    if not marker.exists():
        if (cache / f'batch_{start:06d}.npz').exists():
            LOGGER.warning('保留无提交回执的批次文件，重新计算：%s', cache / f'batch_{start:06d}.npz')
        return None
    entry = json.loads(marker.read_text())
    path = cache / f'batch_{start:06d}.npz'
    if (entry['identity_sha256'] != cache_key(identity) or entry['start'] != start or entry['stop'] != stop
            or entry['counts'] != list(counts) or sha256(path) != entry['sha256']):
        raise ValueError('批次缓存身份或内容 hash 不匹配')
    with np.load(path, allow_pickle=False) as data:
        if not np.array_equal(data['chunk_indices'], np.arange(start, stop)):
            raise ValueError('批次缓存 chunk 顺序不匹配')
        values = {n: data[f'N{n}'].copy() for n in counts}
    for value in values.values():
        if value.dtype != np.float32 or value.shape != (stop-start, 1, 600) or not np.isfinite(value).all():
            raise ValueError('批次缓存 shape/dtype/有限性不匹配')
    return values, entry['prefix_seconds']


def commit_batch(cache, identity, start, stop, counts, values, times):
    path = cache / f'batch_{start:06d}.npz'
    with path.open('xb') as stream:
        np.savez(stream, chunk_indices=np.arange(start, stop), **{f'N{n}': values[n] for n in counts})
        stream.flush(); os.fsync(stream.fileno())
    exclusive_json(cache / f'batch_{start:06d}.json', {'identity_sha256': cache_key(identity),
        'start': start, 'stop': stop, 'counts': list(counts), 'sha256': sha256(path), 'prefix_seconds': times})


def predict_cached(model, dataset, spec, counts, output, identity, *, batch_size=64, resume_caches=()):
    cache = output / 'batch_cache'; cache.mkdir(parents=True, exist_ok=False)
    identity = {**identity, 'spec': spec.identity(), 'counts': list(counts), 'batch_size': batch_size,
                'trajectory_group_size': 1, 'n_chunks': len(dataset)}
    exclusive_json(cache / 'identity.json', identity)
    for previous in resume_caches:
        if json.loads((Path(previous) / 'identity.json').read_text()) != identity:
            raise ValueError('恢复缓存合同不同')
    parts = {n: [] for n in counts}; elapsed = {str(n): 0. for n in counts}; reused = 0
    device = next(model.parameters()).device
    total_batches = (len(dataset) + batch_size - 1) // batch_size
    progress = Progress(f'{spec.sampler.upper()} durable ensemble', total_batches, every=1)
    def sync():
        if device.type == 'cuda': torch.cuda.synchronize(device)
    for batch_number, start in enumerate(range(0, len(dataset), batch_size), 1):
        stop = min(start + batch_size, len(dataset)); found = None
        for previous in resume_caches:
            found = load_cached_batch(Path(previous), identity, start, stop, counts)
            if found is not None: break
        if found is None:
            batch = default_collate([dataset[i] for i in range(start, stop)])
            if batch['index'].tolist() != list(range(start, stop)):
                raise ValueError('真实 batch 的 chunk 顺序不匹配')
            keys = [(str(dataset.rows.iloc[i//13].split), int(dataset.rows.iloc[i//13].dataset_row_id), i % 13)
                    for i in range(start, stop)]
            condition = batch['x'].to(device); sync(); begin = time.perf_counter()
            values, times = {}, {}
            for n, mean in accelerated_prefixes(model, condition, keys, spec, AccelerationSpec(1), checkpoints=counts):
                sync(); values[n] = mean.cpu().numpy(); times[str(n)] = time.perf_counter() - begin
        else:
            values, times = found; reused += 1
        commit_batch(cache, identity, start, stop, counts, values, times)
        for n in counts:
            parts[n].append(values[n]); elapsed[str(n)] += times[str(n)]
        progress.update(batch_number, f'chunks={stop}/{len(dataset)} reused_batches={reused}; batch committed')
    predictions = {n: np.concatenate(parts[n]).reshape(len(dataset.rows), 13, 1, 600) for n in counts}
    profiles = [{'sampler': spec.sampler, 'N': n, 'nfe_per_trajectory': spec.nfe,
        'denoiser_calls_per_chunk': n * spec.nfe, 'sampling_seconds': elapsed[str(n)], 'n_parents': len(dataset.rows),
        'n_chunks': len(dataset), 'batch_size': batch_size, 'trajectory_group_size': 1, 'reused_batches': reused,
        'runtime_definition': 'sum of original per-batch prefix delivery times; excludes resume skips and durable disk writes'} for n in counts]
    write_json(output / 'sampling_identity.json', {'spec': spec.identity(), 'counts': list(counts),
        'ordered_chunk_indices': list(range(len(dataset))), 'profiles': profiles, 'cache_identity': identity})
    return predictions, profiles


def run(output, *, seed, audit_run, device, resume_run=None):
    manifest = rt.read_manifest(MANIFEST)
    proof = json.loads((Path(audit_run) / 'recovery_audit.json').read_text())
    audit_receipt = json.loads((Path(audit_run) / 'receipt.json').read_text())
    if (audit_receipt['status'] != 'complete' or audit_receipt['artifacts']['recovery_audit.json'] != sha256(Path(audit_run) / 'recovery_audit.json')
            or proof['program_sha256'] != sha256(Path(__file__)) or proof['manifest_sha256'] != sha256(MANIFEST / 'research_test_manifest.json')):
        raise ValueError('恢复核验或程序身份变化')
    reference = proof['seeds'][str(seed)]; source = Path(reference['source'])
    for name, expected in reference['original_files'].items():
        if sha256(source / name) != expected: raise ValueError(f'中断来源被改变：{name}')
    with study_run(output, 'research-test', synthetic=False, source=rt.model_path(seed)) as (out, receipt):
        receipt.update(research_test_protocol=rt.PROTOCOL, recovery_revision=REVISION,
                       recovery_audit_sha256=sha256(Path(audit_run) / 'recovery_audit.json'))
        seed_all(20260811)
        model, cfg, checkpoint = load_source(rt.model_path(seed), torch.device(device), expected_sha256=rt.CHECKPOINTS[seed])
        expected = next(item for item in manifest['models'] if item['training_seed'] == seed)
        if {k: v for k, v in checkpoint.items() if k != 'source_run'} != expected:
            raise ValueError('恢复模型状态不同于冻结清单')
        datasets, index = rt.datasets_for(cfg)
        rt.verify_preflight(PREFLIGHT, MANIFEST, datasets, index)
        for split, ds in datasets.items():
            if rt.frame_hash(rt.canonical_rows(ds.rows, cfg.data.dataset_root)) != manifest['splits'][split]['canonical_rows_sha256']:
                raise ValueError('恢复数据身份改变')
        dataset = datasets['test']; code = rt.identity(cfg)
        OmegaConf.save(cfg, out / 'resolved_config.yaml')
        write_json(out / 'source_identity.json', code); save_source_snapshot(out, code)
        shutil.copy2(Path(__file__), out / 'recovery_program.py')
        write_json(out / 'recovery_audit.json', proof)
        write_json(out / 'environment.json', environment(cpu_only=False))
        write_json(out / 'execution.json', {'protocol': rt.PROTOCOL, 'recovery_revision': REVISION,
            'checkpoint': checkpoint, 'device': device, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'reused_source': str(source), 'program_sha256': proof['program_sha256']})
        write_json(out / 'research_test_manifest.json', manifest)
        data = data_identity(out, {'test': dataset}, index)
        rt.canonical_rows(dataset.rows, cfg.data.dataset_root).to_csv(out / 'test_canonical_parents.csv', index=False)
        curves, profiles = [], []
        if reference['reused_ddim_counts']:
            for n in reference['reused_ddim_counts']:
                shutil.copytree(source / f'ddim_S6_N{n}', out / f'ddim_S6_N{n}')
            shutil.copytree(source / 'ddim_sampling', out / 'ddim_sampling')
            curves = pd.read_csv(source / 'sampler_grid.csv').to_dict('records')
            profiles = pd.read_csv(source / 'runtime_profile.csv').to_dict('records')
            LOGGER.info('复用已完整核验的 DDIM N16/N64，原来源保留：%s', source)
        grids = [] if reference['reused_ddim_counts'] else [(rt.ResearchTestSpec('ddim', 6, 64), (16, 64))]
        grids.append((rt.ResearchTestSpec('ddpm', 50, 16), (16,)))
        pd.DataFrame(curves).to_csv(out / 'sampler_grid.csv', index=False)
        base = {'revision': REVISION, 'manifest_sha256': proof['manifest_sha256'], 'training_seed': seed,
                'checkpoint_sha256': rt.CHECKPOINTS[seed], 'test_identity': manifest['splits']['test']['canonical_rows_sha256']}
        with preserved_evaluation(model):
            for spec, counts in grids:
                sampling = out / f'{spec.sampler}_sampling'
                previous = Path(resume_run) / f'{spec.sampler}_sampling/batch_cache' if resume_run else None
                caches = [previous] if previous is not None and previous.exists() else []
                predictions, profile = predict_cached(model, dataset, spec, counts, sampling, base, resume_caches=caches)
                profiles.extend(profile)
                for n in counts:
                    actual = rt.ResearchTestSpec(spec.sampler, spec.nfe, n)
                    row = evaluate_setting(predictions.pop(n), dataset, cfg, out / f'{spec.sampler}_S{spec.nfe}_N{n}',
                        actual, n, method_prefix=rt.PROTOCOL, output_prefix='research_test')
                    row.update(training_seed=seed, sampling_seed=20261003, checkpoint_sha256=rt.CHECKPOINTS[seed],
                        role=next(c['role'] for c in rt.CONFIGURATIONS if (c['sampler'], c['N']) == (spec.sampler, n)))
                    curves.append(row)
                    pd.DataFrame(curves).to_csv(out / 'sampler_grid.csv', index=False)
                    pd.DataFrame(profiles).to_csv(out / 'runtime_profile.csv', index=False)
        rt.validate_curves(pd.DataFrame(curves), manifest, all_seeds=False)
        timings = independent_timing(model, dataset, [rt.ResearchTestSpec(c['sampler'], c['S'], c['N']) for c in rt.CONFIGURATIONS], batch_size=64)
        write_json(out / 'independent_runtime.json', timings)
        verify_data_unchanged(data, dataset.rows); rt.read_manifest(MANIFEST)
        if (state_digest(model) != checkpoint['parameter_sha256'] or sha256(rt.model_path(seed) / 'final.pt') != rt.CHECKPOINTS[seed]
                or rt.identity(cfg) != code or sha256(Path(__file__)) != proof['program_sha256']):
            raise ValueError('恢复期间模型/数据/实现改变')
        receipt.update(checkpoint=checkpoint, n_parents=len(dataset.rows), n_chunks=len(dataset),
            manifest_sha256=proof['manifest_sha256'], parameter_sha256_after=state_digest(model),
            reused_ddim_counts=reference['reused_ddim_counts'])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('audit', 'run')); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--confirm-research-test', action='store_true'); p.add_argument('--seed', type=int, choices=SEEDS)
    p.add_argument('--audit-run', type=Path); p.add_argument('--device', default='cuda:0')
    p.add_argument('--resume-run', type=Path)
    args = p.parse_args(argv)
    if not args.confirm_research_test: p.error('需要 --confirm-research-test')
    if args.phase == 'audit': audit(args.output)
    else:
        if args.seed is None or args.audit_run is None or not args.device.startswith('cuda:') or not args.device[5:].isdigit():
            p.error('需要 seed、audit-run 与 cuda:N')
        run(args.output, seed=args.seed, audit_run=args.audit_run, device=args.device, resume_run=args.resume_run)


if __name__ == '__main__': main()
