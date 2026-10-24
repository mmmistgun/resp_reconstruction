"""只读复核 P6 已保存产物；不加载模型、不读取原始数据、不调用 GPU。"""
import hashlib
import json
from pathlib import Path
import datetime

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
STUDY = OUT.parent
ROOT = STUDY.parents[1]
COL = STUDY / 'collected_20261024_v1'
RUNS = {20260811: 'test_s20260811_v2', 20260812: 'test_s20260812_v3', 20260813: 'test_s20260813_v2'}
METRICS = ['whole_rr_abs_error_bpm', 'local_rr_mae_bpm', 'envelope_trajectory_mae',
           'global_envelope_modulation_error', 'lag_aware_signed_pcc']
CONFIGS = [('ddim', 6, 16), ('ddim', 6, 64), ('ddpm', 50, 16)]


def read(path): return json.loads(path.read_text())
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()
def save(name, value):
    with (OUT / name).open('x') as f: json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
def close(a, b):
    if not np.allclose(a, b, atol=1e-12, rtol=1e-12): raise ValueError(f'数值不一致：{a} != {b}')
def same_frame(a, b, keys):
    pd.testing.assert_frame_equal(a.sort_values(keys).reset_index(drop=True),
        b[a.columns].sort_values(keys).reset_index(drop=True), check_dtype=False, atol=1e-12, rtol=1e-12)


def main():
    inventory = read(COL / 'collection_inventory_20261024_v1.json')
    hashes = {}
    for name, entry in inventory['files'].items():
        path = COL / name
        if path.stat().st_size != entry['size']: raise ValueError(f'文件大小不一致：{name}')
        digest = sha(path)
        if digest != entry['sha256']: raise ValueError(f'远端与本地 SHA256 不同：{name}')
        hashes[name] = digest
    print(f'本地与远端 {len(hashes)} 项 SHA256 一致', flush=True)
    checked_receipts, artifact_references = [], 0
    for name in hashes:
        if Path(name).name != 'receipt.json': continue
        receipt = read(COL / name)
        if receipt['status'] != 'complete': raise ValueError(f'来源未完成：{name}')
        for child, expected in receipt['artifacts'].items():
            key = str(Path(name).parent / child)
            if hashes.get(key) != expected: raise ValueError(f'回执产物不一致：{key}')
            artifact_references += 1
        checked_receipts.append(name)
    manifest = read(COL / 'manifest_v1/research_test_manifest.json')
    manifest_sha = sha(COL / 'manifest_v1/research_test_manifest.json')
    for name, digest in {**manifest['files'], **manifest['input_files']}.items():
        if sha(ROOT / name) != digest: raise ValueError(f'本地冻结来源改变：{name}')
    summary = COL / 'summary_v3'
    nine = pd.read_csv(summary / 'research_test_nine_configurations.csv')
    keys = ['training_seed', 'sampler', 'N']
    expected_cells = {(seed, sampler, n) for seed in RUNS for sampler, _, n in CONFIGS}
    if len(nine) != 9 or set(map(tuple, nine[keys].to_numpy())) != expected_cells:
        raise ValueError('九项配置不完整')
    if not (nine.n_parents == 2310).all() or not (nine.sampling_seed == 20261003).all():
        raise ValueError('样本或采样 seed 改变')
    source_receipts = {Path(v['run']).name: v['receipt_sha256'] for v in read(summary / 'sources.json')}
    curves, paired_source, subjects, macro, worst, timings = [], {}, [], [], [], []
    target_hashes, waveform_count, metric_row_count = {}, 0, 0
    subject_counts = None
    for seed, name in RUNS.items():
        run = COL / name; receipt = read(run / 'receipt.json')
        checkpoint = receipt['checkpoint']
        if source_receipts[name] != sha(run / 'receipt.json'): raise ValueError('自动汇总来源不一致')
        if receipt['manifest_sha256'] != manifest_sha or checkpoint['checkpoint_sha256'] != manifest['checkpoints'][str(seed)]:
            raise ValueError('模型或 manifest 不一致')
        if receipt['parameter_sha256_after'] != checkpoint['parameter_sha256']: raise ValueError('模型状态改变')
        if sha(run / 'test_canonical_parents.csv') != manifest['splits']['test']['canonical_rows_sha256']:
            raise ValueError('规范化父集合改变')
        parents = pd.read_csv(run / 'test_parents.csv')
        if len(parents) != 2310 or set(parents.split) != {'test'} or parents.dataset_row_id.duplicated().any():
            raise ValueError('父集合不完整或重复')
        counts = parents.groupby('samp_id').size().to_dict()
        if subject_counts is None: subject_counts = counts
        elif counts != subject_counts: raise ValueError('受试者集合改变')
        curve = pd.read_csv(run / 'sampler_grid.csv'); curves.append(curve)
        for sampler, steps, n in CONFIGS:
            setting = run / f'{sampler}_S{steps}_N{n}'
            saved = read(setting / 'setting_summary.json')
            row = curve[(curve.sampler == sampler) & (curve.N == n)].iloc[0]
            for key, val in saved.items():
                if isinstance(val, (int, float)): close(row[key], val)
            for view in ('raw', 'postfiltered'):
                table = pd.read_csv(setting / f'research_test_{view}_per_parent.csv')
                if len(table) != 2310 or set(table.split) != {'test'}:
                    raise ValueError('逐窗口指标缺失或 split 改变')
                for key in ('dataset_row_id', 'samp_id'):
                    if not np.array_equal(table[key], parents[key]): raise ValueError('逐窗口指标错序')
                if not np.isfinite(table[METRICS].to_numpy()).all(): raise ValueError('核心指标非有限')
                for metric in METRICS: close(table[metric].mean(), saved[f'{view}_{metric}'])
                metric_row_count += len(table)
                if view == 'postfiltered':
                    paired_source[(seed, sampler, n)] = table
                    subject = table.groupby('samp_id')[METRICS].mean()
                    for subject_id, values in subject.iterrows():
                        subjects.append({'training_seed': seed, 'sampler': sampler, 'N': n, 'samp_id': int(subject_id),
                                         'n_parents': counts[subject_id], **values.to_dict()})
                    macro.append({'training_seed': seed, 'sampler': sampler, 'N': n, **subject.mean().to_dict()})
                with np.load(setting / f'research_test_{view}_waveforms.npz', allow_pickle=False) as data:
                    for key in ('dataset_row_id', 'samp_id'):
                        if not np.array_equal(data[key], parents[key]): raise ValueError('波形身份错序')
                    if set(data['split']) != {'test'}: raise ValueError('波形 split 不匹配')
                    prediction, target = data[f'prediction_{view}'], data['tho_ref']
                    for value in (prediction, target):
                        if value.shape != (2310,18000) or not np.isfinite(value).all(): raise ValueError('波形维度或有限性失败')
                    target_digest = hashlib.sha256(target.tobytes()).hexdigest()
                    if view in target_hashes and target_hashes[view] != target_digest: raise ValueError('配置间参考波形改变')
                    target_hashes[view] = target_digest
                    maxima = np.abs(prediction).max(axis=1)
                    close(float(maxima.max()), saved[f'{view}_global_max'])
                    close(float(np.quantile(maxima, .99)), saved[f'{view}_parent_max_p99'])
                    if view == 'postfiltered':
                        i = int(maxima.argmax())
                        worst.append({'training_seed':seed,'sampler':sampler,'N':n,'dataset_row_id':int(parents.iloc[i].dataset_row_id),
                            'samp_id':int(parents.iloc[i].samp_id),'prediction_max_abs':float(maxima[i]),
                            'target_max_abs':float(np.abs(target[i]).max())})
                    waveform_count += 1
                del prediction, target
        execution = read(run / 'execution.json')
        for timing in read(run / 'independent_runtime.json'):
            if timing['warmup'] != 1 or timing['timed_repeats'] != 2 or timing['batch_size'] != 64:
                raise ValueError('计时合同不同')
            values = np.asarray(timing['sampling_seconds'])
            if not np.isfinite(values).all() or not (values > 0).all(): raise ValueError('计时无效')
            close(np.median(values), timing['independent_batch_sampling_seconds_median'])
            timings.append({'training_seed':seed, 'sampler':timing['sampler'], 'N':timing['N'],
                'physical_gpu':execution['cuda_visible_devices'], 'seconds':float(np.median(values)),
                'peak_allocated_gib':max(timing['peak_allocated_bytes'])/2**30,
                'peak_reserved_gib':max(timing['peak_reserved_bytes'])/2**30})
        print(f'seed{seed}：双视图波形、逐窗口指标与模型身份通过', flush=True)
    same_frame(nine, pd.concat(curves, ignore_index=True), keys)
    metrics = [c for c in nine if c.startswith(('raw_', 'postfiltered_')) and not c.endswith('_n_defined')]
    if not np.isfinite(nine[metrics].to_numpy()).all(): raise ValueError('汇总存在非有限值')
    grouped = nine.groupby(['sampler','nfe_per_trajectory','N','role'])[metrics].agg(['mean','std','min','max'])
    grouped.columns = ['__'.join(c) for c in grouped.columns]
    same_frame(grouped.reset_index(), pd.read_csv(summary/'training_seed_summary.csv'), ['sampler','N'])
    comparison = pd.read_csv(summary/'validation_test_comparison.csv')
    same_frame(nine, comparison[comparison.evaluation_split=='test'], keys)
    val = pd.read_csv(STUDY/'inputs_v1/p5_summary/full_validation_nine_configurations.csv')
    same_frame(val, comparison[comparison.evaluation_split=='val'], keys)
    paired = []
    for seed in RUNS:
        for label, a, b in [('ddim64_minus_ddpm16',('ddim',64),('ddpm',16)), ('ddim64_minus_ddim16',('ddim',64),('ddim',16))]:
            left, right = paired_source[(seed,*a)], paired_source[(seed,*b)]
            for metric in METRICS:
                delta = left[metric] - right[metric]
                subject_delta = delta.groupby(left.samp_id).mean()
                better = delta > 0 if metric == 'lag_aware_signed_pcc' else delta < 0
                better_subject = subject_delta > 0 if metric == 'lag_aware_signed_pcc' else subject_delta < 0
                paired.append({'training_seed':seed,'comparison':label,'metric':metric,'mean_delta':float(delta.mean()),
                    'improved_parent_fraction':float(better.mean()), 'improved_subjects':int(better_subject.sum()),
                    'n_subjects':len(subject_delta),'subject_macro_delta':float(subject_delta.mean())})
    pd.DataFrame(paired).to_csv(OUT/'paired_effects.csv',index=False)
    pd.DataFrame(subjects).to_csv(OUT/'subject_metrics.csv',index=False)
    pd.DataFrame(macro).to_csv(OUT/'subject_macro_descriptive.csv',index=False)
    pd.DataFrame(worst).to_csv(OUT/'worst_amplitude_cases.csv',index=False)
    pd.DataFrame(timings).to_csv(OUT/'independent_runtime.csv',index=False)
    stats = comparison.groupby(['evaluation_split','sampler','N'])[metrics].agg(['mean','std','min','max'])
    stats.columns = ['__'.join(c) for c in stats.columns]
    stats.to_csv(OUT/'cross_split_summary.csv')
    nine[['training_seed','sampler','N']+[f'postfiltered_{c}' for c in METRICS]].to_csv(OUT/'core_metrics_by_seed.csv',index=False)
    receipt = {'status':'complete','finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':'read_only_saved_artifacts','manifest_sha256':manifest_sha,
        'collection_inventory_sha256':sha(COL/'collection_inventory_20261024_v1.json'),
        'transferred_files_verified':len(hashes),'transferred_bytes':sum(v['size'] for v in inventory['files'].values()),
        'receipts_verified':checked_receipts,'artifact_references_verified':artifact_references,
        'frozen_files_verified':len(manifest['files']),'input_files_verified':len(manifest['input_files']),
        'n_models':3,'n_configurations':9,'n_parents_per_configuration':2310,'n_chunks_per_configuration':30030,
        'n_subjects':len(subject_counts),'subject_parent_counts':{str(k):int(v) for k,v in subject_counts.items()},
        'waveform_files_verified':waveform_count,'metric_rows_verified':metric_row_count,
        'target_waveform_sha256':target_hashes,'original_summary_verified':True,'validation_comparison_verified':True,
        'artifacts':{p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='console.log'}}
    save('review.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('artifacts','receipts_verified')},ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    try: main()
    except BaseException as error:
        save('failure.json', {'status':'failed','error_type':type(error).__name__,'error':str(error)})
        raise
