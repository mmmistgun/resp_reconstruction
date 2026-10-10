"""复用固定SA窗口分组，汇总P1-Full-Add M4的research-test性能。"""
from __future__ import annotations

import argparse
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_h_only_medical_rr_strata import identity, verify, read_manifest, write_json, SEEDS, METRICS, SESSION
from scripts.analyze_h_only_sa_presence import summarize, GROUPS

PROTOCOL = ROOT / 'docs/experiments/p1_m4_sa_presence_v1_protocol_20261007.md'
SA = ROOT / 'runs/h_only_sa_presence_v1/attempt_20261005T175626Z_3c7db0d6459b'
P1 = Path('/home/marques/.codex/worktrees/model-architecture-review/resp_reconstruction/runs/p1_components_v1/research_test_v1_20261007')
ATTEMPTS = ('attempt_381d565d5cd24490b8cf04149dbbb888', 'attempt_ada310604cea4620889def97f1f695e4', 'attempt_085e269f1a0746919724c546f422a32d')
ALLOW_SHA = '8ac8b2305f66cd4f54f6a317f7c1dad9336bd88d393206fa84b5f300ad68fa47'
KEYS = ['dataset_row_id', 'samp_id', 'split']
POSITION = [*KEYS, 'target_source_npz', 'target_signal_key', 'window_start_sample', 'window_end_sample', 'target_fs']


def checked(path, sha, sources):
    observed = identity(path)
    if observed['sha256'] != sha:
        raise ValueError(f'冻结来源哈希不一致：{path}')
    sources[str(Path(path).resolve())] = observed


def validate_rows(metrics, rows, reference, attrs, seed):
    """同row ID仍须核对THO定位，避免仅按数量把异源窗口贴上标签。"""
    if len(metrics) != len(attrs) or metrics.dataset_row_id.duplicated().any() or rows.dataset_row_id.duplicated().any():
        raise ValueError('窗口数或唯一性不一致')
    for left, right, columns in ((metrics, attrs, KEYS), (metrics, rows, KEYS), (rows, reference, POSITION)):
        if not left[columns].reset_index(drop=True).equals(right[columns].reset_index(drop=True)):
            raise ValueError('行顺序、主体或THO定位不一致')
    if not metrics.arm.eq('M4').all() or not metrics.seed.eq(seed).all() or not metrics.method.eq('P1-Full-Add').all():
        raise ValueError('M4模型身份不一致')


def run(output, sources):
    sa_receipt = json.loads((SA / 'receipt.json').read_text())
    if sa_receipt['status'] != 'completed':
        raise ValueError('SA来源未完成')
    sources[str(SA / 'receipt.json')] = identity(SA / 'receipt.json')
    verify(SA / 'manifest.json', sa_receipt['manifest'], sources)
    manifest = read_manifest(SA, sources)
    for name in ('window_groups.csv', 'window_metrics.csv', 'subject_composition.csv'):
        verify(SA / name, manifest['files'][name], sources)
    attrs = pd.read_csv(SA / 'window_groups.csv')
    if (len(attrs) != 2310 or attrs.samp_id.nunique() != 8 or attrs.dataset_row_id.duplicated().any()
            or not attrs.split.eq('test').all()
            or attrs.sa_group.value_counts().to_dict() != {'no_sa_annotation': 1880, 'sa_present': 430}
            or hashlib.sha256(attrs.dataset_row_id.to_numpy(dtype='<i8').tobytes()).hexdigest() != '184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8'):
        raise ValueError('SA来源行身份/分组不一致')
    attrs.to_csv(output / 'window_groups.csv', index=False)
    shutil.copy2(SA / 'subject_composition.csv', output / 'subject_composition.csv')
    data = SESSION / 'research_test/data/attempt_20261002T045020Z_0be812da89ce'
    manifest = read_manifest(data, sources)
    verify(data / 'test_rows.csv', manifest['files']['test_rows.csv'], sources)
    reference = pd.read_csv(data / 'test_rows.csv')
    checked(P1 / 'allowlist.json', ALLOW_SHA, sources)
    entries = [e for e in json.loads((P1 / 'allowlist.json').read_text())['entries'] if e['arm'] == 'M4']
    if sorted(e['seed'] for e in entries) != list(SEEDS):
        raise ValueError('缺少M4三seed身份')
    for entry in entries:
        for name in ('checkpoint', 'config'):
            checked(entry[name]['path'], entry[name]['sha256'], sources)
    write_json(output / 'checkpoints.json', entries)
    old_sa = pd.read_csv(SA / 'window_metrics.csv')
    frames = []
    flags = sorted(set(METRICS.values()))
    for seed, attempt in zip(SEEDS, ATTEMPTS):
        directory = P1 / f'evaluation/M4_seed{seed}' / attempt
        receipt_path = directory / 'receipt.json'
        sources[str(receipt_path)] = identity(receipt_path)
        receipt = json.loads(receipt_path.read_text())
        if receipt['allowlist_sha256'] != ALLOW_SHA:
            raise ValueError('评价allowlist不一致')
        for name in ('metrics.csv', 'test_rows.csv', 'evaluation.json'):
            checked(directory / name, receipt['artifacts'][name], sources)
        evaluation = json.loads((directory / 'evaluation.json').read_text())
        entry = next(e for e in entries if e['seed'] == seed)
        if evaluation['checkpoint_sha256'] != entry['checkpoint']['sha256'] or evaluation['selected_epoch'] != entry['selected_epoch']:
            raise ValueError('评价checkpoint不一致')
        metrics = pd.read_csv(directory / 'metrics.csv')
        rows = pd.read_csv(directory / 'test_rows.csv')
        validate_rows(metrics, rows, reference, attrs, seed)
        prior = old_sa.loc[old_sa.seed.eq(seed)].reset_index(drop=True)
        eligibility_keys = ['dataset_row_id', 'samp_id', *flags]
        if not metrics[eligibility_keys].equals(prior[eligibility_keys]):
            raise ValueError('M4与SA来源target资格不一致')
        frames.append(metrics.assign(sa_group=attrs.sa_group.to_numpy()))
    frame = pd.concat(frames, ignore_index=True)
    frame[['arm', 'seed', *KEYS, 'sa_group', *METRICS, *flags]].to_csv(output / 'window_metrics.csv', index=False)
    per_seed, across = summarize(frame)
    per_seed.to_csv(output / 'metrics_per_seed.csv', index=False)
    across.to_csv(output / 'metrics_across_seed.csv', index=False)
    summary = P1 / 'summary/attempt_b4769c4f0e724872b110fe20c6db99eb'
    receipt_path = summary / 'receipt.json'
    sources[str(receipt_path)] = identity(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    if receipt['allowlist_sha256'] != ALLOW_SHA:
        raise ValueError('汇总allowlist不一致')
    checked(summary / 'per_seed.csv', receipt['artifacts']['per_seed.csv'], sources)
    original = pd.read_csv(summary / 'per_seed.csv')
    differences = []
    for (seed, metric), part in per_seed.groupby(['seed', 'metric']):
        groups, full = part.loc[part.sa_group.ne('all')], part.loc[part.sa_group.eq('all')].iloc[0]
        old = original.loc[original.arm.eq('M4') & original.seed.eq(seed) & original.metric.eq(metric)]
        if groups.eligible_n.sum() != full.eligible_n or len(old) != 1 or full.eligible_n == 0:
            raise ValueError('分母或冻结汇总身份错误')
        restored = (groups['mean'].fillna(0) * groups.eligible_n).sum() / full.eligible_n
        difference = max(abs(restored-full['mean']), abs(old.iloc[0]['mean']-full['mean']))
        if difference > 1e-12:
            raise ValueError('全量复算不一致')
        differences.append(difference)
    for path, expected in list(sources.items()):
        verify(path, expected, sources)
    write_json(output / 'verification.json', dict(window_n=2310, subject_n=8, seed_n=3,
               sa_present_n=430, no_sa_annotation_n=1880, all_15_full_mean_checks_passed=True,
               max_absolute_difference=float(max(differences)), same_tho_positions_and_target_eligibility=True))
    return across


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm-research-test', action='store_true')
    args = parser.parse_args()
    if not args.confirm_research_test:
        parser.error('需要本次授权及 --confirm-research-test')
    output = ROOT / 'runs/p1_m4_sa_presence_v1' / ('attempt_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:12])
    output.mkdir(parents=True, exist_ok=False)
    sources = {}
    for path in (Path(__file__), PROTOCOL, ROOT / 'scripts/analyze_h_only_sa_presence.py', ROOT / 'scripts/analyze_h_only_medical_rr_strata.py'):
        shutil.copy2(path, output / path.name)
        sources[str(path)] = identity(path)
    write_json(output / 'started.json', dict(protocol='p1-m4-sa-presence-v1-20261007', research_test_authorized=True,
               command=sys.argv, python=sys.version, numpy=np.__version__, pandas=pd.__version__, platform=platform.platform(),
               git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()))
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
