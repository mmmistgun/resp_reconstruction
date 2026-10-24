"""P6 seed20260812 双卡迁移；仅拆分原始 B64 批次，复用冻结恢复器。"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
spec = importlib.util.spec_from_file_location('p6_recovery', STUDY / 'recovery_20261021_v1/recover.py')
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)
SEED = 20260812
CUT = 280 * 64
OLD = STUDY / 'test_s20260812_v2'
NEW = STUDY / 'test_s20260812_v3'
AUDIT = STUDY / 'recovery_audit_20261021_v1'
HEAD = HERE / 'gpu2_ddpm_head'
TAIL = HERE / 'gpu0_ddpm_tail'
ORIGINAL_PREDICT = rc.predict_cached


def base_identity(manifest):
    return {'revision': rc.REVISION, 'manifest_sha256': rc.sha256(rc.MANIFEST / 'research_test_manifest.json'),
            'training_seed': SEED, 'checkpoint_sha256': rc.rt.CHECKPOINTS[SEED],
            'test_identity': manifest['splits']['test']['canonical_rows_sha256']}


def full_identity(base, sampler, counts, size):
    return {**base, 'spec': sampler.identity(), 'counts': list(counts), 'batch_size': 64,
            'trajectory_group_size': 1, 'n_chunks': size}


def compute_range(model, dataset, sampler, counts, output, identity, start, stop):
    """边界与原批次对齐，尾批保持原大小；缓存身份始终表示完整数据集。"""
    if start % 64 or (stop != len(dataset) and stop % 64) or not 0 <= start < stop <= len(dataset):
        raise ValueError('分片边界不符合原始批次')
    output.mkdir(exist_ok=False)
    cache = output / 'batch_cache'; cache.mkdir()
    identity = full_identity(identity, sampler, counts, len(dataset))
    rc.exclusive_json(cache / 'identity.json', identity)
    device = next(model.parameters()).device
    progress = rc.Progress(f'{sampler.sampler} shard [{start},{stop})', (stop-start+63)//64, every=1)
    for number, begin in enumerate(range(start, stop, 64), 1):
        end = min(begin + 64, stop)
        batch = rc.default_collate([dataset[i] for i in range(begin, end)])
        if batch['index'].tolist() != list(range(begin, end)):
            raise ValueError('分片 batch 索引改变')
        keys = [(str(dataset.rows.iloc[i//13].split), int(dataset.rows.iloc[i//13].dataset_row_id), i % 13)
                for i in range(begin, end)]
        condition = batch['x'].to(device)
        if device.type == 'cuda': rc.torch.cuda.synchronize(device)
        clock = time.perf_counter()
        values, times = {}, {}
        for n, mean in rc.accelerated_prefixes(model, condition, keys, sampler, rc.AccelerationSpec(1), checkpoints=counts):
            if device.type == 'cuda': rc.torch.cuda.synchronize(device)
            values[n] = mean.cpu().numpy(); times[str(n)] = time.perf_counter() - clock
        rc.commit_batch(cache, identity, begin, end, counts, values, times)
        progress.update(number, f'chunks={end}; batch committed')
    rc.exclusive_json(output / 'complete.json', {'status': 'complete', 'start': start, 'stop': stop,
        'identity_sha256': rc.cache_key(identity), 'program_sha256': rc.sha256(Path(__file__))})


def require_partition(caches, identity, size, cut=CUT):
    """先证实两个分片恰好覆盖全部批次，禁止收集时补算遗漏批次。"""
    for cache, start, stop in ((caches[0], 0, cut), (caches[1], cut, size)):
        done = json.loads((cache.parent / 'complete.json').read_text())
        if done != {'status': 'complete', 'start': start, 'stop': stop,
                    'identity_sha256': rc.cache_key(identity), 'program_sha256': rc.sha256(Path(__file__))}:
            raise ValueError('分片完成身份不匹配')
        if json.loads((cache / 'identity.json').read_text()) != identity:
            raise ValueError('分片合同改变')
        expected = {f'batch_{i:06d}.json' for i in range(start, stop, 64)}
        if {p.name for p in cache.glob('batch_*.json')} != expected:
            raise ValueError('分片存在遗漏、重复或越界批次')
        for i in range(start, stop, 64):
            if rc.load_cached_batch(cache, identity, i, min(i+64, size), (16,)) is None:
                raise ValueError('分片批次未提交')


def gpu0_equivalence(model, dataset, sampler, counts, base):
    identity = full_identity(base, sampler, counts, len(dataset))
    cached, _ = rc.load_cached_batch(OLD / 'ddim_sampling/batch_cache', identity, 0, 64, counts)
    batch = rc.default_collate([dataset[i] for i in range(64)])
    keys = [(str(dataset.rows.iloc[i//13].split), int(dataset.rows.iloc[i//13].dataset_row_id), i % 13) for i in range(64)]
    report = {'atol': 1e-5, 'rtol': 1e-4, 'batch_start': 0, 'batch_stop': 64, 'prefixes': {}}
    before = rc.state_digest(model)
    for n, mean in rc.accelerated_prefixes(model, batch['x'].to(next(model.parameters()).device), keys,
                                          sampler, rc.AccelerationSpec(1), checkpoints=counts):
        actual = mean.cpu().numpy()
        if not rc.np.isfinite(actual).all(): raise ValueError('跨 GPU 输出非有限')
        difference = actual.astype('float64') - cached[n]
        passed = rc.np.allclose(actual, cached[n], atol=1e-5, rtol=1e-4)
        report['prefixes'][str(n)] = {'passed': bool(passed), 'bitwise_equal': bool(rc.np.array_equal(actual, cached[n])),
            'max_abs_difference': float(abs(difference).max()), 'rmse': float(rc.np.sqrt((difference**2).mean()))}
    report['passed'] = len(report['prefixes']) == len(counts) and all(v['passed'] for v in report['prefixes'].values()) and before == rc.state_digest(model)
    rc.exclusive_json(HERE / 'gpu0_equivalence.json', report)
    if not report['passed']: raise ValueError('跨 GPU 固定首批一致性验证失败')


def migrated_predict(model, dataset, sampler, counts, output, identity, **kwargs):
    if sampler.sampler == 'ddim':
        rc.shutil.copy2(Path(__file__), output.parent / 'migration_program.py')
        rc.exclusive_json(output.parent / 'migration_identity.json', {'program_sha256': rc.sha256(Path(__file__)),
            'controller': str(HERE), 'previous_run': str(OLD), 'ddim_gpu': 0,
            'ddpm_ranges': [{'gpu': 2, 'start': 0, 'stop': CUT}, {'gpu': 0, 'start': CUT, 'stop': len(dataset)}]})
        gpu0_equivalence(model, dataset, sampler, counts, identity)
        return ORIGINAL_PREDICT(model, dataset, sampler, counts, output, identity, **kwargs)
    compute_range(model, dataset, sampler, counts, TAIL, identity, CUT, len(dataset))
    rc.LOGGER.info('GPU0 DDPM 分片已完成，等待 GPU2 分片回执')
    while not (HEAD / 'receipt.json').exists():
        if (HERE / 'gpu2_exit.json').exists():
            raise RuntimeError('GPU2 已退出但没有完整回执')
        time.sleep(10)
    receipt = json.loads((HEAD / 'receipt.json').read_text())
    if receipt['status'] != 'complete': raise ValueError('GPU2 分片失败')
    for name, digest in receipt['artifacts'].items():
        if rc.sha256(HEAD / name) != digest: raise ValueError(f'GPU2 分片产物改变：{name}')
    caches = [HEAD / 'sampling/batch_cache', TAIL / 'batch_cache']
    require_partition(caches, full_identity(identity, sampler, counts, len(dataset)), len(dataset))
    kwargs['resume_caches'] = caches
    return ORIGINAL_PREDICT(model, dataset, sampler, counts, output, identity, **kwargs)


def worker():
    manifest = rc.rt.read_manifest(rc.MANIFEST)
    with rc.study_run(HEAD, 'research-test-cache-shard', synthetic=False, source=rc.rt.model_path(SEED)) as (out, receipt):
        rc.seed_all(20260811)
        model, cfg, checkpoint = rc.load_source(rc.rt.model_path(SEED), rc.torch.device('cuda:0'), expected_sha256=rc.rt.CHECKPOINTS[SEED])
        expected = next(v for v in manifest['models'] if v['training_seed'] == SEED)
        if {k:v for k,v in checkpoint.items() if k != 'source_run'} != expected: raise ValueError('模型身份不匹配')
        datasets, index = rc.rt.datasets_for(cfg)
        rc.rt.verify_preflight(rc.PREFLIGHT, rc.MANIFEST, datasets, index)
        for split, dataset in datasets.items():
            if rc.rt.frame_hash(rc.rt.canonical_rows(dataset.rows, cfg.data.dataset_root)) != manifest['splits'][split]['canonical_rows_sha256']:
                raise ValueError('数据身份不匹配')
        dataset = datasets['test']; code = rc.rt.identity(cfg)
        rc.write_json(out / 'environment.json', rc.environment(cpu_only=False))
        rc.write_json(out / 'execution.json', {'checkpoint': checkpoint, 'physical_gpu': 2, 'start': 0, 'stop': CUT,
            'program_sha256': rc.sha256(Path(__file__)), 'manifest_sha256': rc.sha256(rc.MANIFEST / 'research_test_manifest.json')})
        data = rc.data_identity(out, {'test': dataset}, index)
        with rc.preserved_evaluation(model):
            compute_range(model, dataset, rc.rt.ResearchTestSpec('ddpm', 50, 16), (16,), out / 'sampling', base_identity(manifest), 0, CUT)
        rc.verify_data_unchanged(data, dataset.rows); rc.rt.read_manifest(rc.MANIFEST)
        if rc.state_digest(model) != checkpoint['parameter_sha256'] or rc.rt.identity(cfg) != code:
            raise ValueError('模型或源码在计算期间改变')
        receipt.update(research_test_protocol=rc.rt.PROTOCOL, checkpoint=checkpoint, start=0, stop=CUT)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('role', choices=('gpu0', 'gpu2'))
    p.add_argument('--confirm-research-test', action='store_true')
    args = p.parse_args()
    if not args.confirm_research_test: p.error('需要 --confirm-research-test')
    if args.role == 'gpu2': worker()
    else:
        rc.predict_cached = migrated_predict
        rc.run(NEW, seed=SEED, audit_run=AUDIT, device='cuda:0', resume_run=OLD)


if __name__ == '__main__': main()
