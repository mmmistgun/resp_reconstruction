"""用同轴 PSG 秒级标签描述 H-only 含 SA/无 SA 标注窗口的性能。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_h_only_medical_rr_strata import (
    SESSION, SEEDS, EVALS, METRICS, identity, verify, read_manifest, write_json, check_alignment,
)

PROTOCOL = ROOT / 'docs/experiments/h_only_sa_presence_v1_protocol_20261006.md'
RAW = Path('/mnt/disk_wd/marques_dataset/Resp_Pair_Dataset/HYS/Raw')
PSG = Path('/mnt/disk_wd/marques_dataset/DataCombine2023/HYS/PSG_Aligned')
BANK = Path('/mnt/disk_code/marques/resp_prepare/dataset/20260620_research_v2_resp_reconstruction_stage2_1_segrobustz_bcgstagee_log1psoftz_robustconf/whole_night/signal_bank')
EVENT_TYPES = ('Obstructive apnea', 'Central apnea', 'Mixed apnea', 'Hypopnea')
GROUPS = ('sa_present', 'no_sa_annotation')


def interval_hits(windows, events):
    """半开区间有正时长交集才计入，端点接触和零长事件不能冒充事件。"""
    windows, events = np.asarray(windows, float), np.asarray(events, float)
    for value in (windows, events):
        if value.ndim != 2 or value.shape[1] != 2 or not np.isfinite(value).all() or (value[:, 1] <= value[:, 0]).any():
            raise ValueError('区间必须为有限且end>start的n×2矩阵')
    return np.minimum(windows[:, None, 1], events[None, :, 1]) > np.maximum(windows[:, None, 0], events[None, :, 0])


def window_groups(rows, sources):
    attributes, audits = [], []
    for sid, windows in rows.groupby('samp_id', sort=True):
        label = RAW / str(sid) / f'{sid}_SA Label.csv'
        synced = PSG / str(sid) / 'SA Label_Sync.csv'
        sources[str(synced)] = identity(synced)
        verify(label, sources[str(synced)], sources)
        raw_tho = list((RAW / str(sid)).glob(f'{sid}_Effort Tho_*.txt'))
        sync_tho = list((PSG / str(sid)).glob('Effort Tho_*.txt'))
        if len(raw_tho) != 1 or len(sync_tho) != 1:
            raise ValueError(f'THO来源不唯一：{sid}')
        sources[str(sync_tho[0])] = identity(sync_tho[0])
        verify(raw_tho[0], sources[str(sync_tho[0])], sources)
        meta_path = BANK / str(sid) / 'research_v2_signal_bank.json'
        sources[str(meta_path)] = identity(meta_path)
        meta = json.loads(meta_path.read_text())
        duration = float(meta['duration_s'])
        if meta['target_fs'] != 100 or not np.isfinite(duration) or duration <= 0:
            raise ValueError('THO覆盖范围/采样率非法')
        events = pd.read_csv(label, encoding='gbk')
        if not events['Event type'].isin(EVENT_TYPES).all():
            raise ValueError(f'未知SA事件类型：{sid}')
        bounds = events[['Start', 'End']].apply(pd.to_numeric, errors='raise').to_numpy(float)
        starts = windows.window_start_s.to_numpy(float)
        ends = windows.window_end_s.to_numpy(float)
        if (starts < 0).any() or (ends > duration).any() or not np.all(ends-starts == 180):
            raise ValueError('窗口超出THO覆盖或非180秒')
        hits = interval_hits(np.column_stack([starts, ends]), bounds)
        within = (bounds[:, 1] > 0) & (bounds[:, 0] < duration)
        # 覆盖范围外事件仍写入审计表；合格窗口不能与它们相交。
        if hits[:, ~within].any():
            raise ValueError('覆盖范围外事件与窗口相交')
        table = windows[['dataset_row_id', 'samp_id', 'split']].copy()
        table['sa_event_n'] = hits.sum(axis=1)
        table['sa_group'] = np.where(hits.any(axis=1), 'sa_present', 'no_sa_annotation')
        attributes.append(table)
        audits.append(pd.DataFrame(dict(samp_id=int(sid), label_row=np.arange(len(events)) + 2,
                      event_type=events['Event type'], start_s=bounds[:, 0], end_s=bounds[:, 1],
                      source_duration_s=pd.to_numeric(events.Duration, errors='raise'),
                      tho_duration_s=duration, in_tho_coverage=within,
                      fully_in_tho_coverage=(bounds[:, 0] >= 0) & (bounds[:, 1] <= duration),
                      touching_test_window_n=hits.sum(axis=0))))
    attrs = pd.concat(attributes).set_index('dataset_row_id').loc[rows.dataset_row_id].reset_index()
    return attrs, pd.concat(audits, ignore_index=True)


def summarize(frame):
    records = []
    for seed in SEEDS:
        for group in (*GROUPS, 'all'):
            subset = frame.loc[frame.seed.eq(seed)]
            if group != 'all':
                subset = subset.loc[subset.sa_group.eq(group)]
            for metric, flag in METRICS.items():
                if not subset[flag].isin([True, False]).all():
                    raise ValueError('指标资格非法')
                values = subset.loc[subset[flag].eq(True), metric].to_numpy(float)
                if not np.isfinite(values).all():
                    raise ValueError('主指标非有限')
                records.append(dict(seed=seed, sa_group=group, metric=metric,
                                    window_n=len(subset), subject_n=subset.samp_id.nunique(),
                                    eligible_n=len(values), mean=float(values.mean()) if len(values) else np.nan))
    per_seed = pd.DataFrame(records)
    for _, part in per_seed.groupby(['sa_group', 'metric']):
        if part.window_n.nunique() != 1 or part.eligible_n.nunique() != 1:
            raise ValueError('三seed分母不一致')
    across = per_seed.groupby(['sa_group', 'metric'], sort=False).agg(
        seed_mean=('mean', 'mean'), seed_sd=('mean', 'std'), window_n=('window_n', 'first'),
        subject_n=('subject_n', 'first'), eligible_n=('eligible_n', 'first'), valid_seed_n=('mean', 'count')).reset_index()
    return per_seed, across


def run(output, sources):
    allow_dir = SESSION / 'research_test/allowlist/attempt_20261002T045020Z_d38da46561ce'
    allow_manifest = read_manifest(allow_dir, sources)
    verify(allow_dir / 'allowlist.json', allow_manifest['files']['allowlist.json'], sources)
    entries = [e for e in json.loads((allow_dir / 'allowlist.json').read_text())['entries'] if e['arm'] == 'H']
    if sorted(e['seed'] for e in entries) != list(SEEDS):
        raise ValueError('缺少H三seed身份')
    for entry in entries:
        for name in ('config', 'checkpoint'):
            verify(entry['sources'][name]['path'], entry['sources'][name], sources)
    write_json(output / 'checkpoints.json', entries)
    data_dir = SESSION / 'research_test/data/attempt_20261002T045020Z_0be812da89ce'
    manifest = read_manifest(data_dir, sources)
    verify(data_dir / 'test_rows.csv', manifest['files']['test_rows.csv'], sources)
    rows = pd.read_csv(data_dir / 'test_rows.csv')
    import hashlib
    row_hash = hashlib.sha256(rows.dataset_row_id.to_numpy(dtype='<i8').tobytes()).hexdigest()
    if (len(rows) != 2310 or rows.samp_id.nunique() != 8 or rows.dataset_row_id.duplicated().any()
            or not rows.split.eq('test').all() or not rows.target_signal_key.eq('tho_waveform_segment_soft_z').all()
            or row_hash != '184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8'):
        raise ValueError('test行身份不一致')
    attrs, events = window_groups(rows, sources)
    if attrs.sa_group.value_counts().to_dict() != {'no_sa_annotation': 1880, 'sa_present': 430}:
        raise ValueError('分组与预检不一致')
    attrs.to_csv(output / 'window_groups.csv', index=False)
    events.to_csv(output / 'event_coverage_audit.csv', index=False)
    frames, base_flags = [], None
    for seed, attempt in zip(SEEDS, EVALS):
        directory = SESSION / f'research_test/evaluation/H/seed_{seed}' / attempt
        manifest = read_manifest(directory, sources)
        if manifest['binding']['allowlist'] != identity(allow_dir / 'manifest.json')['sha256']:
            raise ValueError('指标allowlist不一致')
        verify(directory / 'metrics.csv', manifest['files']['metrics.csv'], sources)
        metrics = pd.read_csv(directory / 'metrics.csv')
        check_alignment(metrics, rows, seed)
        flags = metrics[sorted(set(METRICS.values()))]
        if base_flags is not None and not flags.equals(base_flags):
            raise ValueError('三seed资格不一致')
        base_flags = flags
        frames.append(metrics.assign(sa_group=attrs.sa_group.to_numpy()))
    frame = pd.concat(frames, ignore_index=True)
    frame[['seed', 'dataset_row_id', 'samp_id', 'sa_group', *METRICS, *sorted(set(METRICS.values()))]].to_csv(
        output / 'window_metrics.csv', index=False)
    per_seed, across = summarize(frame)
    per_seed.to_csv(output / 'metrics_per_seed.csv', index=False)
    across.to_csv(output / 'metrics_across_seed.csv', index=False)
    composition = pd.crosstab(attrs.samp_id, attrs.sa_group).reindex(columns=GROUPS, fill_value=0)
    composition.to_csv(output / 'subject_composition.csv')
    summary_dir = SESSION / 'research_test/summary/attempt_20261002T054624Z_e22f113cfb30'
    manifest = read_manifest(summary_dir, sources)
    verify(summary_dir / 'full/per_seed.csv', manifest['files']['full/per_seed.csv'], sources)
    original = pd.read_csv(summary_dir / 'full/per_seed.csv')
    errors = []
    for (seed, metric), part in per_seed.groupby(['seed', 'metric']):
        groups, full = part.loc[part.sa_group.ne('all')], part.loc[part.sa_group.eq('all')].iloc[0]
        old = original.loc[original.arm.eq('H') & original.seed.eq(seed) & original.metric.eq(metric)]
        if groups.eligible_n.sum() != full.eligible_n or len(old) != 1 or full.eligible_n == 0:
            raise ValueError('分母或旧summary身份错误')
        restored = (groups['mean'].fillna(0) * groups.eligible_n).sum() / full.eligible_n
        error = max(abs(restored-full['mean']), abs(old.iloc[0]['mean']-full['mean']))
        if error > 1e-12:
            raise ValueError('分组/冻结summary复算不一致')
        errors.append(error)
    # 执行结束再核对读取来源，捕获并发修改。
    for path, expected in list(sources.items()):
        verify(path, expected, sources)
    write_json(output / 'verification.json', dict(window_n=2310, subject_n=8, seed_n=3,
               sa_present_n=430, no_sa_annotation_n=1880, event_n=len(events),
               outside_tho_event_n=int((~events.in_tho_coverage).sum()),
               all_15_full_mean_checks_passed=True, max_absolute_difference=float(max(errors))))
    return across


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm-research-test', action='store_true')
    args = parser.parse_args()
    if not args.confirm_research_test:
        parser.error('需要本次授权及 --confirm-research-test')
    output = ROOT / 'runs/h_only_sa_presence_v1' / ('attempt_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:12])
    output.mkdir(parents=True, exist_ok=False)
    sources = {}
    for path in (Path(__file__), PROTOCOL, ROOT / 'scripts/analyze_h_only_medical_rr_strata.py'):
        shutil.copy2(path, output / path.name)
        sources[str(path)] = identity(path)
    write_json(output / 'started.json', dict(protocol='h-only-sa-presence-v1-20261006', research_test_authorized=True,
               command=sys.argv, python=sys.version, numpy=np.__version__, pandas=pd.__version__,
               platform=platform.platform(), git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()))
    try:
        across = run(output, sources)
        write_json(output / 'sources.json', sources)
        write_json(output / 'manifest.json', dict(status='completed', files={p.name: identity(p) for p in output.iterdir() if p.is_file()}))
        write_json(output / 'receipt.json', dict(status='completed', output=str(output), manifest=identity(output / 'manifest.json')))
        print(output)
        print(across.to_string(index=False))
    except Exception:
        write_json(output / 'sources.json', sources)
        write_json(output / 'receipt.json', dict(status='failed', traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
