"""从冻结汇总整理 E4 性能、代价与机制证据数据，并登记来源身份。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT/'docs/experiments/e4_closeout_artifact_index_20260922.json'
SEEDS = [20260811, 20260812, 20260813]
METRICS = ['whole_rr_abs_error_bpm', 'local_rr_mae_bpm', 'envelope_trajectory_mae',
           'global_envelope_modulation_error', 'lag_aware_signed_pcc']
ARMS = ['W0_FULL', 'w0_mr4_residual', 'static_scale', 'scale_attention', 'frequency_attention', 'channel_region']
SOURCES: dict[str, dict] = {}


def identity(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''): h.update(block)
    return {'size_bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def source_dir(inventory, root_name, phase):
    rows = [r for r in inventory['completed_attempts']
            if Path(r['path']).parts[1] == root_name and r['phase'] == phase]
    if len(rows) != 1: raise ValueError(f'需要唯一冻结来源: {root_name}/{phase}')
    row = rows[0]; path = ROOT/row['path']
    if identity(path/'manifest.json') != row['manifest']: raise ValueError('来源 manifest 漂移')
    SOURCES[str((path/'manifest.json').relative_to(ROOT))] = row['manifest']
    return path


def read_source(directory, filename):
    path = directory/filename
    manifest = json.loads((directory/'manifest.json').read_text())
    expected = manifest['files'][filename]
    if identity(path) != expected: raise ValueError(f'来源内容漂移: {path}')
    SOURCES[str(path.relative_to(ROOT))] = expected
    return pd.read_csv(path) if path.suffix == '.csv' else json.loads(path.read_text())


def prepare_table(inventory, output):
    comparisons = {}
    for group, root_name in [('A', 'e4_w0_scale_aggregation_v1'), ('B', 'e4_scale_aggregation_v2')]:
        test_root = 'e4_w0_scale_aggregation_test_v1' if group == 'A' else 'e4_scale_aggregation_v2_test'
        filename = 'three_seed_comparison.csv' if group == 'A' else 'four_arm_comparison.csv'
        for split, name in [('val', root_name), ('test', test_root)]:
            frame = read_source(source_dir(inventory, name, 'summary'), filename)
            if 'arm' not in frame: frame['arm'] = 'w0_mr4_residual'
            if not frame.split.eq(split).all(): raise ValueError('比较 split 不符')
            comparisons[group, split] = frame
    baseline = {}
    for (group, split), frame in comparisons.items():
        for metric in METRICS:
            rows = frame[frame.metric.eq(metric)]
            values = rows[['full_mean', 'full_sample_sd']].to_numpy()
            np.testing.assert_allclose(values, np.tile(values[0], (len(values), 1)), rtol=0, atol=0)
            if split in baseline:
                np.testing.assert_allclose(values[0], baseline[split][metric], rtol=1e-12, atol=1e-14)
        if split not in baseline:
            baseline[split] = {m: frame[frame.metric.eq(m)][['full_mean','full_sample_sd']].iloc[0].to_numpy() for m in METRICS}
    records = []; bench_rows = []
    for group, root_name, arms in [('A', 'e4_w0_scale_aggregation_v1', ARMS[:2]),
                                   ('B', 'e4_scale_aggregation_v2', [ARMS[0], *ARMS[2:]])]:
        directory = source_dir(inventory, root_name, 'benchmark')
        benchmark = read_source(directory, 'benchmark.json')
        parameters = read_source(directory, 'parameter_compute_report.json')
        for arm in arms:
            cost = {}
            for mode in ['eval', 'train']:
                rows = [r for r in benchmark['measurements'] if r['arm'] == arm and r['mode'] == mode]
                if sorted(r['group'] for r in rows) != [0, 1, 2]: raise ValueError('benchmark 测量组不完整')
                values = np.array([r['median_seconds']*1000 for r in rows])
                cost[mode+'_ms_mean'] = values.mean()
                cost[mode+'_ms_measurement_sd'] = values.std(ddof=1)
                for kind in ['allocated', 'reserved']:
                    cost[mode+'_peak_'+kind+'_gib_max'] = max(r['peak_'+kind+'_bytes'] for r in rows)/2**30
                for r in rows:
                    np.testing.assert_allclose(np.median(r['seconds']), r['median_seconds'], rtol=1e-12)
                    bench_rows.append(dict(benchmark_group=group, arm=arm, mode=mode, measurement_group=r['group'],
                        median_ms=r['median_seconds']*1000, iqr_ms=r['iqr_seconds']*1000,
                        peak_allocated_bytes=r['peak_allocated_bytes'], peak_reserved_bytes=r['peak_reserved_bytes'],
                        gpu=r['environment']['gpu_name'], warmup=r['warmup'], repeats=r['repeats']))
            count = 1219850 if arm == 'W0_FULL' else parameters['model_parameters'] if group == 'A' else parameters['arms'][arm]['model_parameters']
            for split in ['val', 'test']:
                frame = comparisons[group, split]
                record = dict(benchmark_group=group, arm=arm, split=split, n_seeds=3,
                    windows_per_seed=2675 if split == 'val' else 2310, parameters=count,
                    added_parameters=count-1219850, **cost)
                for metric in METRICS:
                    if arm == 'W0_FULL': mean, sd = baseline[split][metric]
                    else:
                        row = frame[frame.arm.eq(arm) & frame.metric.eq(metric)]
                        if len(row) != 1: raise ValueError('候选指标缺失/重复')
                        mean, sd = row[['candidate_mean', 'candidate_sample_sd']].iloc[0]
                    record[metric+'_mean'], record[metric+'_seed_sd'] = mean, sd
                records.append(record)
    table = pd.DataFrame(records)
    if len(table) != 14 or not np.isfinite(table.select_dtypes('number')).all().all(): raise ValueError('总表形状/有限性异常')
    table.to_csv(output/'performance_cost.csv', index=False)
    pd.DataFrame(bench_rows).to_csv(output/'benchmark_measurement_groups.csv', index=False)
    return table


def prepare_mechanism(inventory, output):
    directory = source_dir(inventory, 'e4_r3_temporal_normalization_v1', 'summary')
    paired = read_source(directory, 'paired_seed_subject_changes.csv')
    receipt = read_source(directory, 'summary_receipt.json')
    conditions = [f'SHIFT_{i}__ALL_W_GN_FIXED' for i in (1,2,3)]
    frame = paired[paired.condition.isin(conditions) & paired.metric.isin(METRICS[2::2])].copy()
    if len(frame) != 3*3*8*2: raise ValueError('机制配对矩阵不完整')
    if frame.duplicated(['seed','condition','scope','group','metric']).any(): raise ValueError('重复配对行')
    if set(frame.seed) != set(SEEDS): raise ValueError('seed 集合漂移')
    expected = np.where(frame.metric.eq(METRICS[4]), frame.full-frame.intervened, frame.intervened-frame.full)
    np.testing.assert_allclose(frame.degradation, expected, rtol=1e-10, atol=1e-14)
    subjects = sorted(frame.loc[frame.scope.eq('samp_id'), 'group'].astype(int).unique())
    if len(subjects) != 7: raise ValueError('受试者集合不完整')
    counts = []
    for (condition, metric, subject), g in frame[frame.scope.eq('samp_id')].groupby(['condition','metric','group']):
        if set(g.seed) != set(SEEDS) or len(g) != 3: raise ValueError('受试者 seed 不完整')
        counts.append(dict(condition=condition,metric=metric,subject=int(subject),
            worse_seeds=int((g.degradation>0).sum()),better_seeds=int((g.degradation<0).sum()),
            equal_seeds=int((g.degradation==0).sum()),n_seeds=3))
    directions = pd.DataFrame(counts)
    signal_dir = source_dir(inventory, 'e4_r3_temporal_normalization_v1', 'signals')
    if identity(signal_dir/'manifest.json') != receipt['signals']['manifest']: raise ValueError('signals 来源不匹配')
    signal = read_source(signal_dir, 'association_summary.csv')
    signal = signal[signal.representation.eq('R3_MEAN') & signal.support.eq('full') & signal.lag_sec.eq(0)
                    & signal['transform'].isin(['FULL','SHIFT_1','SHIFT_2','SHIFT_3'])].copy()
    if len(signal) != 4*2*8 or not signal.defined_n.eq(signal.n).all(): raise ValueError('信号对应数据缺失')
    drops = read_source(signal_dir, 'paired_association_summary.csv')
    drops = drops[drops.representation.eq('R3_MEAN') & drops.support.eq('full') & drops.lag_sec.eq(0)
                  & drops['transform'].isin(['SHIFT_1','SHIFT_2','SHIFT_3'])].copy()
    if len(drops) != 3*2*8: raise ValueError('信号配对矩阵不完整')
    if not drops[drops.scope.eq('samp_id')].absolute_drop_mean.gt(0).all(): raise ValueError('信号方向与结项记录不符')
    frame.to_csv(output/'mechanism_paired_points.csv',index=False)
    directions.to_csv(output/'mechanism_subject_directions.csv',index=False)
    signal.to_csv(output/'mechanism_signal_associations.csv',index=False)
    drops.to_csv(output/'mechanism_signal_paired_changes.csv',index=False)
    return frame, directions, signal, subjects


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'docs/experiments/e4_closeout_data_20260922')
    args=parser.parse_args()
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    inventory=json.loads(INDEX.read_text())
    table=prepare_table(inventory,output)
    frame,directions,signal,subjects=prepare_mechanism(inventory,output)
    write_json(output/'source_manifest.json',{
        'index':{'path':str(INDEX.relative_to(ROOT)),**identity(INDEX)},
        'sources':SOURCES,
        'script':{'path':str(Path(__file__).resolve().relative_to(ROOT)),**identity(Path(__file__))},
        'statistics':{'performance':'3 training seeds, mean and sample SD (ddof=1)',
            'timing':'mean and sample SD of 3 independent measurement-group medians',
            'memory':'maximum allocated/reserved bytes across measurement groups',
            'signal':'shared analysis; subject means and window-pooled means; no seed replication'},
        'environment':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__}})
    write_json(output/'manifest.json',{'kind':'e4_closeout_dataset','status':'completed',
        'table_rows':len(table),'paired_rows':len(frame),'subject_direction_rows':len(directions),
        'signal_association_rows':len(signal),'subjects':list(map(int,subjects)),
        'files':{str(p.relative_to(output)):identity(p) for p in sorted(output.iterdir()) if p.is_file()}})
    print(output)


if __name__=='__main__': main()
