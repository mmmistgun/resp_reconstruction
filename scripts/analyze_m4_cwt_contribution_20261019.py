"""从冻结逐受试者指标整理原始 M4 的 Full/NoTF/H/L 贡献与 670 敏感性。"""
from pathlib import Path
import hashlib
import json
import shutil

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / 'runs/home_storage_migration_20261010/preserved_worktrees/worktrees_20261010/model-architecture-review/runs/m4_components_v1'
SOURCES = {
    'validation': HISTORY / 'formal_v1_20261008/summary/9cbebaa8b4b44b5d8bf14b4d53fa7396',
    'research_test': HISTORY / 'research_test_v1_20261009/summary/attempt_a1790585cd644d79969174cb7cc1a40b',
}
OUT = ROOT / 'runs/m4_cwt_contribution_analysis_20261019/analysis_01'
ARMS = {'B0': 'Full', 'B1': 'NoTF', 'B4': 'H', 'B5': 'L'}
SEEDS = {20260811, 20260812, 20260813}
METRICS = ['whole_rr_abs_error_bpm', 'local_rr_mae_bpm', 'envelope_trajectory_mae',
           'global_envelope_modulation_error', 'lag_aware_signed_pcc']
PAIRS = [('Full', 'NoTF'), ('H', 'NoTF'), ('L', 'NoTF'), ('Full', 'L'), ('Full', 'H'), ('H', 'L')]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    records, provenance, subject_pairs = [], {}, []
    for split, source in SOURCES.items():
        receipt = json.loads((source / 'receipt.json').read_text())
        path = source / 'per_subject.csv'
        expected = receipt['artifacts']['per_subject.csv']
        assert sha(path) == expected
        provenance[split] = {'path': str(path), 'sha256': expected, 'receipt_sha256': sha(source/'receipt.json')}
        data = pd.read_csv(path)
        data = data[data.arm.isin(ARMS)].copy()
        data['model'] = data.arm.map(ARMS)
        assert set(data.seed) == SEEDS and set(data.metric) == set(METRICS)
        assert not data.duplicated(['model', 'seed', 'samp_id', 'metric']).any()
        assert np.isfinite(data['mean']).all()
        expected_subjects = 7 if split == 'validation' else 8
        assert len(data) == 4*3*expected_subjects*5
        counts = data.pivot(index=['seed', 'samp_id', 'metric'], columns='model', values='eligible_n')
        assert (counts.nunique(axis=1) == 1).all()
        values = data.pivot(index=['seed', 'samp_id', 'metric'], columns='model', values='mean').reset_index()
        for candidate, reference in PAIRS:
            p = values[['seed', 'samp_id', 'metric']].copy()
            p['comparison'], p['split'] = f'{candidate}-vs-{reference}', split
            # improvement 统一为正表示候选较好；保留完整逐 seed、逐人配对。
            sign = np.where(p.metric == 'lag_aware_signed_pcc', 1, -1)
            p['improvement'] = sign * (values[candidate]-values[reference])
            subject_pairs.append(p)
        for view in (['full'] if split == 'validation' else ['full', 'exclude670']):
            selected = data[data.samp_id != 670] if view == 'exclude670' else data
            expected_n = 2675 if split == 'validation' else (2231 if view == 'exclude670' else 2310)
            for (model, seed, metric), group in selected.groupby(['model', 'seed', 'metric']):
                assert group.eligible_n.sum() == expected_n
                for weighting in ('window', 'subject_macro'):
                    weights = group.eligible_n if weighting == 'window' else np.ones(len(group))
                    records.append(dict(split=split, view=view, weighting=weighting, model=model,
                        seed=seed, metric=metric, value=np.average(group['mean'], weights=weights),
                        subjects=len(group), windows=expected_n))
    seed_values = pd.DataFrame(records)
    model_summary = seed_values.groupby(['split','view','weighting','model','metric'],as_index=False).agg(
        value=('value','mean'),seed_sd=('value','std'))
    comparisons = []
    index = ['split','view','weighting','seed','metric']
    wide = seed_values.pivot(index=index,columns='model',values='value').reset_index()
    for candidate, reference in PAIRS:
        p = wide[index].copy()
        p['comparison'] = f'{candidate}-vs-{reference}'
        p['reference'] = wide[reference]
        p['candidate'] = wide[candidate]
        sign = np.where(p.metric == 'lag_aware_signed_pcc',1,-1)
        p['improvement'] = sign*(wide[candidate]-wide[reference])
        comparisons.append(p)
    paired = pd.concat(comparisons,ignore_index=True)
    comparison_summary = paired.groupby(['split','view','weighting','comparison','metric'],as_index=False).agg(
        reference=('reference','mean'),candidate=('candidate','mean'),improvement=('improvement','mean'),
        seed_sd=('improvement','std'),better_seeds=('improvement',lambda x:int((x>0).sum())))
    subjects = pd.concat(subject_pairs,ignore_index=True)
    OUT.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(__file__,OUT/'analysis_source.py')
    for name, table in [('seed_values',seed_values),('model_summary',model_summary),('paired_seeds',paired),
                        ('comparison_summary',comparison_summary),('paired_subjects',subjects)]:
        table.to_csv(OUT/f'{name}.csv',index=False)
    (OUT/'receipt.json').write_text(json.dumps({'sources':provenance,
        'scope':'Original P1-M4 B0/B1/B4/B5; frozen metrics only; no model inference',
        'exclude670_role':'post-hoc sensitivity; full population retained',
        'positive_improvement':'reference minus candidate for errors; candidate minus reference for PCC',
        'artifacts':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()}},ensure_ascii=False,indent=2))
    print('OUTPUT',OUT)
    for view in ('full','exclude670'):
        print('\nTEST',view)
        selected=model_summary[(model_summary.split=='research_test')&(model_summary.view==view)&(model_summary.weighting=='window')]
        print(selected.pivot(index='model',columns='metric',values='value').to_string())
        selected=comparison_summary[(comparison_summary.split=='research_test')&(comparison_summary.view==view)&(comparison_summary.weighting=='window')]
        print(selected[['comparison','metric','improvement','better_seeds']].to_string(index=False))


if __name__=='__main__':
    main()
