"""复用冻结THO RR与SA组别，汇总M4三seed及IEWT确定性基线。"""
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
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_h_only_medical_rr_strata import identity, verify, read_manifest, write_json, METRICS, SEEDS
from scripts.analyze_p1_m4_sa_presence import P1, ATTEMPTS
PROTOCOL = ROOT / 'docs/experiments/m4_iewt_strata_v1_protocol_20261007.md'
RR = Path('/mnt/disk_code/marques/resp_reconstruction/runs/h_only_medical_rr_strata_v1/attempt_20261005T154759Z_17fabd96d171')
M4 = ROOT / 'runs/p1_m4_sa_presence_v1/attempt_20261007T150849Z_4d5ba53300e8'
IEWT = Path('/mnt/disk_code/marques/resp_reconstruction/runs/tho_iewt_research_test/20260807_134559_593049')
AXES = {'rr_stratum': ('slow', 'reference_range', 'fast'), 'sa_group': ('sa_present', 'no_sa_annotation')}
KEYS = ['dataset_row_id', 'samp_id', 'split']


def frozen(directory, names, sources):
    receipt_path = directory / 'receipt.json'
    sources[str(receipt_path)] = identity(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    if receipt['status'] != 'completed':
        raise ValueError('来源分析未完成')
    verify(directory / 'manifest.json', receipt['manifest'], sources)
    manifest = read_manifest(directory, sources)
    for name in names:
        verify(directory / name, manifest['files'][name], sources)


def summarize(frame):
    records = []
    for (model, instance), subset in frame.groupby(['model', 'instance'], sort=False):
        for axis, groups in AXES.items():
            for group in (*groups, 'all'):
                part = subset if group == 'all' else subset.loc[subset[axis].eq(group)]
                for metric, flag in METRICS.items():
                    if not part[flag].isin([True, False]).all():
                        raise ValueError('指标资格非法')
                    values = part.loc[part[flag].eq(True), metric].to_numpy(float)
                    if not np.isfinite(values).all():
                        raise ValueError('主指标非有限')
                    records.append(dict(model=model, instance=instance, axis=axis, group=group, metric=metric,
                                window_n=len(part), subject_n=part.samp_id.nunique(), eligible_n=len(values),
                                mean=float(values.mean()) if len(values) else np.nan))
    per = pd.DataFrame(records)
    for _, part in per.groupby(['model', 'axis', 'group', 'metric']):
        if part.eligible_n.nunique() != 1:
            raise ValueError('实例间分母不同')
    across = per.groupby(['model', 'axis', 'group', 'metric'], sort=False).agg(
        mean=('mean', 'mean'), instance_sd=('mean', 'std'), instance_n=('mean', 'count'),
        window_n=('window_n', 'first'), subject_n=('subject_n', 'first'), eligible_n=('eligible_n', 'first')).reset_index()
    return per, across


def run(output, sources):
    frozen(RR, ['window_strata.csv'], sources)
    frozen(M4, ['window_metrics.csv', 'window_groups.csv', 'metrics_per_seed.csv', 'checkpoints.json', 'sources.json'], sources)
    rr, groups = pd.read_csv(RR / 'window_strata.csv'), pd.read_csv(M4 / 'window_groups.csv')
    if not rr[KEYS].equals(groups[KEYS]) or len(rr) != 2310:
        raise ValueError('RR与SA来源行身份不同')
    if rr.rr_stratum.value_counts().to_dict() != {'reference_range':1125, 'slow':1122, 'fast':63}:
        raise ValueError('RR分母不同')
    attrs = rr.assign(sa_group=groups.sa_group)
    attrs.to_csv(output / 'window_groups.csv', index=False)
    m4 = pd.read_csv(M4 / 'window_metrics.csv')
    if set(m4.seed) != set(SEEDS) or not m4.arm.eq('M4').all():
        raise ValueError('M4三seed身份不同')
    for seed in SEEDS:
        part = m4.loc[m4.seed.eq(seed)].reset_index(drop=True)
        if not part[KEYS].equals(attrs[KEYS]) or not part.sa_group.equals(attrs.sa_group):
            raise ValueError('M4组别或行顺序不同')
    m4 = m4.merge(attrs[['dataset_row_id', 'rr_stratum']], on='dataset_row_id', validate='many_to_one', sort=False)
    m4['model'], m4['instance'] = 'M4', m4.seed.astype(str)
    for name in ('run_manifest.json', 'resolved_config.yaml', 'sample_metrics.csv', 'summary.csv'):
        sources[str(IEWT / name)] = identity(IEWT / name)
    manifest = json.loads((IEWT / 'run_manifest.json').read_text())
    if manifest['experiment_id'] != 'IEWT' or manifest['split'] != 'test' or manifest['n_samples'] != 2310 or manifest['algorithm'] != 'protocolized_python_iewt':
        raise ValueError('IEWT身份不同')
    cfg = OmegaConf.load(IEWT / 'resolved_config.yaml')
    if (str(cfg.data.dataset_root) != '/mnt/disk_code/marques/resp_prepare/dataset/20260620_research_v2_resp_reconstruction_stage2_1_segrobustz_bcgstagee_log1psoftz_robustconf'
            or cfg.data.index_csv != 'training/dataset_index.csv' or cfg.data.target_key != 'target_waveform_segment_soft_z_key'
            or cfg.data.bcg_input_key != 'bcg_rawish_segment_soft_z_key'):
        raise ValueError('IEWT数据/输入/THO配置不同')
    iewt = pd.read_csv(IEWT / 'sample_metrics.csv')
    if not iewt[KEYS].equals(attrs[KEYS]) or not iewt.method.eq('IEWT').all() or iewt.dataset_row_id.duplicated().any():
        raise ValueError('IEWT窗口身份不同')
    # 用原M4指标中的目标属性与输入身份辅助核对历史IEWT target合同。
    original_path = P1 / f'evaluation/M4_seed{SEEDS[0]}' / ATTEMPTS[0] / 'metrics.csv'
    m4_sources = json.loads((M4 / 'sources.json').read_text())
    verify(original_path, m4_sources[str(original_path)], sources)
    original = pd.read_csv(original_path)
    flags = sorted(set(METRICS.values()))
    if not iewt[['input_set', 'coupling_state_id', *flags]].equals(original[['input_set', 'coupling_state_id', *flags]]):
        raise ValueError('IEWT输入身份或target资格不同')
    if not np.allclose(iewt.target_envelope_modulation, original.target_envelope_modulation, rtol=0, atol=1e-12):
        raise ValueError('IEWT target属性不同')
    if not iewt.bcg_signal_key.eq(manifest['source_signal_key']).all() or manifest['source_signal_key'] != 'bcg_rawish_wideband_state_aligned_segment_soft_z':
        raise ValueError('IEWT输入信号key不同')
    iewt = iewt.merge(attrs[['dataset_row_id', 'rr_stratum', 'sa_group']], on='dataset_row_id', validate='one_to_one', sort=False)
    iewt['model'], iewt['instance'] = 'IEWT', 'deterministic'
    frame = pd.concat([m4, iewt], ignore_index=True)
    frame[['model', 'instance', *KEYS, 'rr_stratum', 'sa_group', *METRICS, *flags]].to_csv(output / 'window_metrics.csv', index=False)
    per, across = summarize(frame)
    per.to_csv(output / 'metrics_per_instance.csv', index=False)
    across.to_csv(output / 'metrics_summary.csv', index=False)
    differences = []
    old_m4, old_iewt = pd.read_csv(M4 / 'metrics_per_seed.csv'), pd.read_csv(IEWT / 'summary.csv')
    for (model, instance, axis, metric), part in per.groupby(['model', 'instance', 'axis', 'metric']):
        subset, full = part.loc[part.group.ne('all')], part.loc[part.group.eq('all')].iloc[0]
        if subset.eligible_n.sum() != full.eligible_n or full.eligible_n == 0:
            raise ValueError('全量分母恢复失败')
        reconstructed = (subset['mean'].fillna(0)*subset.eligible_n).sum()/full.eligible_n
        old = (old_iewt.iloc[0][metric+'_mean'] if model == 'IEWT' else
               old_m4.loc[old_m4.seed.eq(int(instance)) & old_m4.sa_group.eq('all') & old_m4.metric.eq(metric), 'mean'].item())
        difference = max(abs(reconstructed-full['mean']), abs(old-full['mean']))
        if difference > 1e-12:
            raise ValueError('冻结全量对账失败')
        differences.append(difference)
    for axis, groups in AXES.items():
        composition = pd.crosstab(attrs.samp_id, attrs[axis]).reindex(columns=groups, fill_value=0)
        composition.to_csv(output / f'{axis}_subject_composition.csv')
    for path, expected in list(sources.items()):
        verify(path, expected, sources)
    write_json(output / 'verification.json', dict(all_40_full_mean_checks_passed=True, max_absolute_difference=float(max(differences)),
               m4_seed_n=3, iewt_instance_n=1, same_row_identity_and_target_eligibility=True))
    return across


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm-research-test', action='store_true')
    args = parser.parse_args()
    if not args.confirm_research_test:
        parser.error('需要本次授权及 --confirm-research-test')
    output = ROOT / 'runs/m4_iewt_strata_v1' / ('attempt_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:12])
    output.mkdir(parents=True, exist_ok=False)
    sources = {}
    for path in (Path(__file__), PROTOCOL, ROOT/'scripts/analyze_h_only_medical_rr_strata.py', ROOT/'scripts/analyze_p1_m4_sa_presence.py', ROOT/'scripts/analyze_h_only_sa_presence.py'):
        shutil.copy2(path, output/path.name)
        sources[str(path)] = identity(path)
    write_json(output/'started.json', dict(protocol='m4-iewt-strata-v1-20261007', research_test_authorized=True, command=sys.argv,
               python=sys.version, numpy=np.__version__, pandas=pd.__version__, platform=platform.platform(),
               git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()))
    try:
        summary = run(output, sources)
        write_json(output/'sources.json', sources)
        write_json(output/'manifest.json', dict(status='completed', files={p.name:identity(p) for p in output.iterdir() if p.is_file()}))
        write_json(output/'receipt.json', dict(status='completed', output=str(output), manifest=identity(output/'manifest.json')))
        print(output)
        print(summary.to_string(index=False))
    except Exception:
        write_json(output/'sources.json', sources)
        write_json(output/'receipt.json', dict(status='failed', traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
