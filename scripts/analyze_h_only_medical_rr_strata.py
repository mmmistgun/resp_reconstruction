"""基于冻结 THO 参考和 H-only 指标进行医学参考 RR 三区间描述。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback
import uuid

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from resp_train.metrics.task import TaskMetricConfig, compute_target_whole_rr_bpm

PROTOCOL = ROOT / 'docs/experiments/h_only_medical_rr_strata_v1_protocol_20261005.md'
SOURCE = Path('/home/marques/.codex/worktrees/cwt-time-frequency-v1/resp_reconstruction')
SESSION = SOURCE / 'runs/cwt_apor_v2/session_20261001T042244Z_e5e53ad16731'
SEEDS = (20260811, 20260812, 20260813)
GROUPS = ('slow', 'reference_range', 'fast')
EVALS = ('attempt_20261002T054308Z_d7e28870553f',
         'attempt_20261002T054403Z_7dccc688c2d1',
         'attempt_20261002T054457Z_ddefb506eda0')
METRICS = {
    'whole_rr_abs_error_bpm': 'whole_rr_target_eligible',
    'local_rr_mae_bpm': 'local_rr_target_eligible',
    'envelope_trajectory_mae': 'joint_target_eligible',
    'global_envelope_modulation_error': 'joint_target_eligible',
    'lag_aware_signed_pcc': 'joint_target_eligible',
}


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'path': str(path.resolve()), 'sha256': digest.hexdigest(), 'size_bytes': path.stat().st_size}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def verify(path, expected, sources):
    observed = identity(path)
    for key in ('sha256', 'size_bytes'):
        if observed[key] != expected[key]:
            raise ValueError(f'来源身份不一致：{path} {key}')
    sources[str(Path(path).resolve())] = observed


def read_manifest(directory, sources):
    path = directory / 'manifest.json'
    sources[str(path.resolve())] = identity(path)
    payload = json.loads(path.read_text())
    if payload['status'] != 'completed':
        raise ValueError(f'来源阶段未完成：{directory}')
    return payload


def classify_rr(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('参考 RR 必须为一维有限正数')
    return np.where(values < 12, 'slow', np.where(values <= 20, 'reference_range', 'fast'))


def summarize(frame):
    records = []
    for seed in SEEDS:
        part = frame.loc[frame.seed.eq(seed)]
        for group in (*GROUPS, 'all'):
            subset = part if group == 'all' else part.loc[part.rr_stratum.eq(group)]
            for metric, flag in METRICS.items():
                if not subset[flag].isin([True, False]).all():
                    raise ValueError(f'非法资格标志：{flag}')
                values = subset.loc[subset[flag].eq(True), metric].to_numpy(float)
                if not np.isfinite(values).all():
                    raise ValueError(f'主指标非有限：{seed} {group} {metric}')
                records.append(dict(seed=seed, rr_stratum=group, metric=metric,
                                    window_n=len(subset), subject_n=subset.samp_id.nunique(),
                                    eligible_n=len(values), mean=float(values.mean()) if len(values) else np.nan))
    per_seed = pd.DataFrame(records)
    across = per_seed.groupby(['rr_stratum', 'metric'], sort=False).agg(
        seed_mean=('mean', 'mean'), seed_sd=('mean', 'std'),
        window_n=('window_n', 'first'), subject_n=('subject_n', 'first'),
        eligible_n=('eligible_n', 'first'), valid_seed_n=('mean', 'count')).reset_index()
    return per_seed, across


def check_alignment(metrics, rows, seed):
    keys = ['dataset_row_id', 'samp_id', 'split']
    if len(metrics) != len(rows) or metrics.dataset_row_id.duplicated().any():
        raise ValueError('指标窗口数/唯一性不一致')
    if not metrics[keys].reset_index(drop=True).equals(rows[keys].reset_index(drop=True)):
        raise ValueError('指标顺序/受试者/split不一致')
    if not metrics.seed.eq(seed).all() or not metrics.arm.eq('H').all():
        raise ValueError('模型或seed身份不一致')


def run(output, sources):
    allow_dir = SESSION / 'research_test/allowlist/attempt_20261002T045020Z_d38da46561ce'
    allow_manifest = read_manifest(allow_dir, sources)
    verify(allow_dir / 'allowlist.json', allow_manifest['files']['allowlist.json'], sources)
    allowed = json.loads((allow_dir / 'allowlist.json').read_text())
    entries = [e for e in allowed['entries'] if e['arm'] == 'H']
    if sorted(e['seed'] for e in entries) != list(SEEDS):
        raise ValueError('H 三seed allowlist不完整')
    data_dir = SESSION / 'research_test/data/attempt_20261002T045020Z_0be812da89ce'
    data_manifest = read_manifest(data_dir, sources)
    for name in ('test_rows.csv', 'test_reference.npy'):
        verify(data_dir / name, data_manifest['files'][name], sources)
    rows = pd.read_csv(data_dir / 'test_rows.csv')
    row_hash = hashlib.sha256(rows.dataset_row_id.to_numpy(dtype='<i8').tobytes()).hexdigest()
    if (len(rows) != 2310 or rows.samp_id.nunique() != 8 or rows.dataset_row_id.duplicated().any()
            or not rows.split.eq('test').all() or not rows.target_signal_key.eq('tho_waveform_segment_soft_z').all()
            or row_hash != '184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8'):
        raise ValueError('固定 THO test 行身份不一致')
    # 使用和原评价字节一致的公共算法，禁止用当前漂移后的实现复算参考RR。
    for relative in ('resp_train/metrics/task.py', 'resp_train/protocols/respiration.py'):
        verify(ROOT / relative, identity(SOURCE / relative), sources)
    configs, checkpoints = [], []
    for entry in entries:
        for name in ('config', 'checkpoint'):
            verify(entry['sources'][name]['path'], entry['sources'][name], sources)
        verify(entry['formal_manifest']['path'], entry['formal_manifest'], sources)
        cfg = OmegaConf.load(entry['sources']['config']['path'])
        configs.append(cfg)
        checkpoints.append({'seed': entry['seed'], 'selected_epoch': entry['selected_epoch'],
                            'checkpoint': entry['sources']['checkpoint']})
    settings = [asdict(TaskMetricConfig.from_config(cfg)) for cfg in configs]
    if any(value != settings[0] for value in settings[1:]):
        raise ValueError('三seed参考RR设置不同')
    write_json(output / 'metric_config.json', settings[0])
    write_json(output / 'checkpoints.json', checkpoints)
    target = np.load(data_dir / 'test_reference.npy', mmap_mode='r', allow_pickle=False)
    if target.shape != (2310, 1, 18000):
        raise ValueError('THO参考shape不一致')
    rr, eligible = compute_target_whole_rr_bpm(target, configs[0])
    if not eligible.all():
        raise ValueError('参考RR存在未定义窗口，需先明确处理口径')
    attrs = rows[['dataset_row_id', 'samp_id', 'split']].copy()
    attrs['target_rr_bpm'] = rr
    attrs['rr_stratum'] = classify_rr(rr)
    attrs.to_csv(output / 'window_strata.csv', index=False)
    frames = []
    for seed, attempt in zip(SEEDS, EVALS):
        directory = SESSION / f'research_test/evaluation/H/seed_{seed}' / attempt
        manifest = read_manifest(directory, sources)
        if manifest['binding']['allowlist'] != identity(allow_dir / 'manifest.json')['sha256']:
            raise ValueError('评价allowlist身份不一致')
        for name in ('metrics.csv', 'reference_source.json'):
            verify(directory / name, manifest['files'][name], sources)
        ref = json.loads((directory / 'reference_source.json').read_text())
        verify(data_dir / 'test_reference.npy', ref, sources)
        metrics = pd.read_csv(directory / 'metrics.csv')
        check_alignment(metrics, rows, seed)
        if not np.array_equal(metrics.whole_rr_target_eligible.to_numpy(), eligible):
            raise ValueError('参考RR资格与原评价不同')
        frames.append(metrics.assign(target_rr_bpm=rr, rr_stratum=attrs.rr_stratum.to_numpy()))
    frame = pd.concat(frames, ignore_index=True)
    frame[['seed', 'dataset_row_id', 'samp_id', 'rr_stratum', 'target_rr_bpm', *METRICS]].to_csv(
        output / 'window_metrics.csv', index=False)
    per_seed, across = summarize(frame)
    per_seed.to_csv(output / 'metrics_per_seed.csv', index=False)
    across.to_csv(output / 'metrics_across_seed.csv', index=False)
    counts = []
    for group in GROUPS:
        part = attrs.loc[attrs.rr_stratum.eq(group)]
        counts.append(dict(rr_stratum=group, window_n=len(part), window_percent=100 * len(part) / len(attrs),
                           subject_n=part.samp_id.nunique(), rr_min=part.target_rr_bpm.min(),
                           rr_median=part.target_rr_bpm.median(), rr_max=part.target_rr_bpm.max()))
    pd.DataFrame(counts).to_csv(output / 'stratum_counts.csv', index=False)
    composition = attrs.groupby(['rr_stratum', 'samp_id']).size().rename('window_n').reset_index()
    composition['within_stratum_percent'] = 100 * composition.window_n / composition.groupby('rr_stratum').window_n.transform('sum')
    composition.to_csv(output / 'subject_composition.csv', index=False)
    quality = rows[['dataset_row_id', 'rate_confidence_level', 'waveform_confidence_level', 'allowed_losses']].merge(
        attrs[['dataset_row_id', 'rr_stratum']], validate='one_to_one', on='dataset_row_id')
    quality_records = []
    for column in ('rate_confidence_level', 'waveform_confidence_level', 'allowed_losses'):
        q = quality.groupby(['rr_stratum', column], dropna=False).size().rename('window_n').reset_index().rename(columns={column: 'value'})
        quality_records.append(q.assign(attribute=column))
    pd.concat(quality_records).to_csv(output / 'quality_composition.csv', index=False)
    # 分层加权重组必须恢复原全量均值，同时和旧冻结summary逐seed交叉核对。
    summary_dir = SESSION / 'research_test/summary/attempt_20261002T054624Z_e22f113cfb30'
    summary_manifest = read_manifest(summary_dir, sources)
    verify(summary_dir / 'full/per_seed.csv', summary_manifest['files']['full/per_seed.csv'], sources)
    original = pd.read_csv(summary_dir / 'full/per_seed.csv')
    original = original.loc[original.arm.eq('H')]
    differences = []
    for (seed, metric), subset in per_seed.groupby(['seed', 'metric']):
        groups = subset.loc[subset.rr_stratum.ne('all')]
        full = subset.loc[subset.rr_stratum.eq('all')].iloc[0]
        if groups.eligible_n.sum() != full.eligible_n:
            raise ValueError('分层有效分母不能恢复全量')
        restored = (groups['mean'].fillna(0) * groups.eligible_n).sum() / full.eligible_n
        old = original.loc[original.seed.eq(seed) & original.metric.eq(metric)]
        if len(old) != 1 or not np.isclose(restored, full['mean'], rtol=0, atol=1e-12) or not np.isclose(old.iloc[0]['mean'], full['mean'], rtol=0, atol=1e-12):
            raise ValueError('分层复算/冻结全量summary不一致')
        differences.append(max(abs(restored-full['mean']), abs(old.iloc[0]['mean']-full['mean'])))
    write_json(output / 'verification.json', {'window_n': len(attrs), 'subject_n': 8, 'seed_n': 3,
                'row_order_sha256': row_hash, 'reference_rr_eligible_n': int(eligible.sum()),
                'all_15_full_mean_checks_passed': True, 'max_absolute_difference': float(max(differences))})
    return across, pd.DataFrame(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm-research-test', action='store_true')
    args = parser.parse_args()
    if not args.confirm_research_test:
        parser.error('需要本次 research-test 授权及 --confirm-research-test')
    output = ROOT / 'runs/h_only_medical_rr_strata_v1' / (
        'attempt_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:12])
    output.mkdir(parents=True, exist_ok=False)
    sources = {}
    shutil.copy2(__file__, output / Path(__file__).name)
    shutil.copy2(PROTOCOL, output / PROTOCOL.name)
    write_json(output / 'started.json', {'protocol': 'h-only-medical-rr-strata-v1-20261005',
               'research_test_authorized': True, 'command': sys.argv, 'python': sys.version,
               'numpy': np.__version__, 'pandas': pd.__version__, 'platform': platform.platform(),
               'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()})
    try:
        across, counts = run(output, sources)
        write_json(output / 'sources.json', sources)
        files = {p.name: identity(p) for p in output.iterdir() if p.is_file()}
        write_json(output / 'manifest.json', {'status': 'completed', 'files': files})
        write_json(output / 'receipt.json', {'status': 'completed', 'output': str(output),
                   'manifest': identity(output / 'manifest.json')})
        print(str(output))
        print(counts.to_string(index=False))
        print(across.to_string(index=False))
    except Exception:
        write_json(output / 'sources.json', sources)
        write_json(output / 'receipt.json', {'status': 'failed', 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
