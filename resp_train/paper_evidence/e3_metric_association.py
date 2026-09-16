"""E3：仅从冻结指标 CSV 与时间元数据生成多属性分布统计。"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = 'e3-w0-metric-association-v1-20260916'
SEEDS = (20260811, 20260812, 20260813)
COUNTS = {'validation': (2675, 7), 'test': (2310, 8)}
PRIMARY = ('whole_rr_abs_error_bpm', 'local_rr_mae_bpm', 'envelope_trajectory_mae',
           'global_envelope_modulation_error', 'lag_aware_signed_pcc')
AXES = (*PRIMARY[:4], 'signed_pcc_error')
FLAGS = ('whole_rr_target_eligible', 'local_rr_target_eligible', 'joint_target_eligible')
DEGENERACY = ('joint_prediction_degenerate', 'envelope_spearman_prediction_degenerate')
QUANTILES = (0.5, 0.75, 0.9)
RR_LIMITS = (0.5, 1.0, 2.0)
LOCK_PATH = Path('docs/experiments/e3_w0_metric_association_lock_20260916.json')
PROTOCOL_PATH = Path('docs/experiments/e3_w0_metric_association_protocol_20260916.md')
OUTPUT = Path('runs/e3_w0_metric_association_v1/analysis')
TRAIN_LOCK = Path('docs/experiments/e2_w0_effort_implementation_lock_20260916.json')
TRAIN_SHA = '29ff147334a358be0b807a5e068e280891522f7ceef831b4ed16105988589edf'
TEST_LOCK = Path('docs/experiments/e2_w0_effort_test_lock_20260916.json')
TEST_SHA = '81fa0622c3fba8060de5f531ad899417654b16d8bf10bc092fcadda7c0a03898'
TEST_SUMMARY = Path('runs/e2_w0_effort_test_v1/summary/summary_81fa0622c3fb_20260916T080746Z_f2ef33859245')
SUMMARY_SHA = '978d3d365fe9cab574244c1fd0761b0512a03f670c95a6fc5e2636cfa7b6d6cf'
CODE_PATHS = (Path('resp_train/paper_evidence/e3_metric_association.py'),
              Path('scripts/analyze_e3_w0_metrics.py'), Path('tests/test_e3_metric_association.py'), PROTOCOL_PATH,
              Path('resp_train/__init__.py'), Path('resp_train/paper_evidence/__init__.py'),
              Path('resp_train/paper_evidence/comparison_audit.py'),
              Path('resp_train/paper_evidence/center_context_config.py'))


def identity(path: Path) -> dict:
    data = path.read_bytes()
    return {'size_bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def verify(path: Path, expected: dict) -> None:
    if identity(path) != {key: expected[key] for key in ('size_bytes', 'sha256')}:
        raise ValueError(f'E3 文件身份漂移: {path}')


def write_json(path: Path, value: dict) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def git_state(root: Path) -> dict:
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=root, text=True).strip()
    return {'commit': git('rev-parse', 'HEAD'), 'status_porcelain': git('status', '--porcelain')}


def contract() -> dict:
    return {'protocol': PROTOCOL, 'seeds': list(SEEDS), 'counts': {k: list(v) for k, v in COUNTS.items()},
            'primary': list(PRIMARY), 'axes': list(AXES), 'quantiles': list(QUANTILES),
            'rr_limits_bpm': list(RR_LIMITS), 'quantile_method': 'linear',
            'calibration': 'validation unique-window three-seed metric mean',
            'rr_rule': 'whole<=limit AND local<=limit', 'high_error_rule': 'error>threshold'}


def validate_frame(metrics: pd.DataFrame, rows: pd.DataFrame, summary: pd.DataFrame, split: str) -> pd.DataFrame:
    """固定来源应全部合格；来源异常显式失败，避免缩小统计分母。"""
    count, subjects = COUNTS[split]
    split_value = 'val' if split == 'validation' else 'test'
    for frame in (rows, metrics):
        if (len(frame) != count or frame.dataset_row_id.dtype.kind not in 'iu'
                or frame.dataset_row_id.duplicated().any() or frame.samp_id.isna().any()
                or frame.samp_id.nunique() != subjects or set(frame.split) != {split_value}):
            raise ValueError('E3 窗口数量、唯一身份、samp_id 或 split 错误')
    for key in ('dataset_row_id', 'samp_id', 'split', 'coupling_state_id'):
        if not np.array_equal(metrics[key].to_numpy(), rows[key].to_numpy()):
            raise ValueError(f'E3 指标与元数据顺序不一致: {key}')
    if set(metrics.method) != {'crd_tf102_w'} or set(metrics.evaluation_split) != {split}:
        raise ValueError('E3 模型或 evaluation_split 错误')
    values = metrics[list(PRIMARY)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise FloatingPointError('E3 五主指标必须全部有限')
    if (values[:, :4] < 0).any() or (np.abs(values[:, 4]) > 1 + 1e-12).any():
        raise ValueError('E3 error/PCC 值域错误')
    for column, expected in [(c, True) for c in FLAGS] + [(c, False) for c in DEGENERACY]:
        flags = metrics[column]
        if flags.isna().any() or not flags.isin([True, False]).all() or not flags.eq(expected).all():
            raise ValueError(f'E3 资格或预测退化错误: {column}')
    if len(summary) != 1:
        raise ValueError('E3 source summary 必须只有一行')
    for key in PRIMARY:
        if (int(summary.iloc[0][key + '_n']) != count or not np.isclose(
                metrics[key].mean(), float(summary.iloc[0][key + '_mean']), rtol=1e-10, atol=1e-12)):
            raise ValueError(f'E3 逐窗口均值与冻结 summary 不匹配: {key}')
    time = rows[['window_start_s', 'window_end_s', 'window_duration_s']].to_numpy(dtype=float)
    if (not np.isfinite(time).all() or (time[:, 0] < 0).any()
            or not np.allclose(time[:, 1] - time[:, 0], 180, rtol=0, atol=1e-8)
            or not np.allclose(time[:, 2], 180, rtol=0, atol=1e-8)):
        raise ValueError('E3 窗口时间必须为有效的 180 秒区间')
    if rows.source_npz.isna().any() or (rows.groupby('samp_id').source_npz.nunique() != 1).any():
        raise ValueError('E3 同一 samp_id 的时间坐标必须来自同一个 source_npz')
    frame = rows[['dataset_row_id', 'samp_id', 'split', 'window_start_s', 'window_end_s']].copy()
    for key in PRIMARY[:4]:
        frame[key] = metrics[key].to_numpy()
    frame[AXES[-1]] = 1 - metrics[PRIMARY[-1]].to_numpy()
    return frame


def nonoverlap_ids(rows: pd.DataFrame) -> set[int]:
    """跨 segment 连续检查时间，半开区间允许相邻端点相接。"""
    selected = set()
    for _, group in rows.groupby('samp_id', sort=True):
        last_end = -np.inf
        for row in group.sort_values(['window_start_s', 'dataset_row_id']).itertuples():
            if row.window_start_s >= last_end:
                selected.add(int(row.dataset_row_id))
                last_end = row.window_end_s
    return selected


def calibrate(validation: dict[int, pd.DataFrame]) -> pd.DataFrame:
    if set(validation) != set(SEEDS):
        raise ValueError('E3 阈值校准需要完整三个 validation seed')
    reference = validation[SEEDS[0]]
    for frame in validation.values():
        if set(frame.split) != {'val'} or not np.array_equal(frame.dataset_row_id, reference.dataset_row_id):
            raise ValueError('E3 阈值只允许身份对齐的 validation')
    means = np.mean([validation[seed][list(AXES[2:])].to_numpy() for seed in SEEDS], axis=0)
    return pd.DataFrame([{'axis': axis, 'quantile': quantile,
                          'threshold': float(np.quantile(means[:, index], quantile, method='linear')),
                          'calibration_windows': len(reference), 'calibration_seeds': len(SEEDS)}
                         for index, axis in enumerate(AXES[2:]) for quantile in QUANTILES])


def associations(frame: pd.DataFrame) -> list[dict]:
    groups = [('pooled_windows', 'all', frame)]
    groups.extend(('within_samp', str(samp), part) for samp, part in frame.groupby('samp_id', sort=True))
    groups.append(('samp_means', 'all', frame.groupby('samp_id', sort=True)[list(AXES)].mean()))
    result = []
    for level, samp, group in groups:
        for left, right in combinations(AXES, 2):
            n = len(group)
            estimate, status = np.nan, 'insufficient_units' if n < 3 else 'ok'
            if status == 'ok':
                if group[left].nunique() < 2 or group[right].nunique() < 2:
                    status = 'constant_input'
                else:
                    # Pearson 相关作用于平均秩即 Spearman；只计算系数，不生成独立性 p-value。
                    estimate = float(np.corrcoef(rankdata(group[left]), rankdata(group[right]))[0, 1])
                    if not np.isfinite(estimate):
                        raise FloatingPointError('E3 Spearman 计算出现非有限值')
            result.append({'level': level, 'samp_id': samp, 'axis_x': left, 'axis_y': right,
                           'n_units': n, 'rho': estimate, 'status': status})
    return result


def discordance(frame: pd.DataFrame, thresholds: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    result = []
    groups = [('pooled_windows', 'all', frame)]
    groups.extend(('within_samp', str(samp), part) for samp, part in frame.groupby('samp_id', sort=True))
    for level, samp, group in groups:
        for limit in RR_LIMITS:
            rr = group[AXES[0]].le(limit) & group[AXES[1]].le(limit)
            total, n_rr = len(group), int(rr.sum())
            for threshold in thresholds.itertuples():
                high = group[threshold.axis].gt(threshold.threshold)
                n_high, n_joint = int(high.sum()), int((rr & high).sum())
                result.append({'level': level, 'samp_id': samp, 'axis': threshold.axis,
                               'rr_limit_bpm': limit, 'quantile': threshold.quantile, 'threshold': threshold.threshold,
                               'n_total': total, 'n_rr': n_rr, 'n_high': n_high, 'n_joint': n_joint,
                               'joint_fraction': n_joint / total, 'high_fraction': n_high / total,
                               'conditional_fraction': n_joint / n_rr if n_rr else np.nan,
                               'status': 'ok' if n_rr else 'no_rr_qualified_windows'})
    macro = []
    individual = pd.DataFrame([r for r in result if r['level'] == 'within_samp'])
    for keys, group in individual.groupby(['axis', 'rr_limit_bpm', 'quantile'], sort=True):
        defined = group.conditional_fraction.notna()
        macro.append({'axis': keys[0], 'rr_limit_bpm': keys[1], 'quantile': keys[2],
                      'threshold': float(group.threshold.iloc[0]), 'n_samp': len(group),
                      'n_samp_conditional_defined': int(defined.sum()), 'n_samp_zero_rr': int((~defined).sum()),
                      'joint_fraction': float(group.joint_fraction.mean()),
                      'high_fraction': float(group.high_fraction.mean()),
                      'conditional_fraction': float(group.loc[defined, 'conditional_fraction'].mean()) if defined.any() else np.nan,
                      'status': 'ok' if defined.all() else 'partial_samp' if defined.any() else 'no_rr_qualified_windows'})
    return result, macro


def seed_summary(frame: pd.DataFrame, keys: list[str], values: list[str]) -> pd.DataFrame:
    result = []
    for index, group in frame.groupby(keys, sort=True, dropna=False):
        if set(group.seed) != set(SEEDS) or len(group) != len(SEEDS):
            raise ValueError('E3 seed 汇总矩阵不完整或重复')
        context = dict(zip(keys, index if isinstance(index, tuple) else (index,), strict=True))
        for value in values:
            valid = group[value].notna()
            defined = group.loc[valid, value]
            result.append({**context, 'statistic': value, 'n_seeds': len(group), 'n_seeds_defined': len(defined),
                           'mean': float(defined.mean()) if len(defined) else np.nan,
                           'sample_sd': float(defined.std(ddof=1)) if len(defined) >= 2 else np.nan,
                           'status': 'ok' if valid.all() else 'partial_seeds' if valid.any() else 'undefined'})
    return pd.DataFrame(result)


def prepare_lock(root: Path = ROOT) -> Path:
    """绑定历史字节身份；阈值及新统计在正式 analyze 阶段生成。"""
    if (root / LOCK_PATH).exists():
        raise FileExistsError('E3 lock 已存在')
    sources = {}

    def source(path: Path, expected: dict | None = None) -> str:
        path = path.resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('E3 来源必须为仓库内冻结结果或元数据')
        if expected is not None:
            verify(path, expected)
        sources[str(path)] = identity(path)
        return str(path)

    for relative, expected_sha in ((TRAIN_LOCK, TRAIN_SHA), (TEST_LOCK, TEST_SHA),
                                   (TEST_SUMMARY / 'manifest.json', SUMMARY_SHA)):
        path = root / relative
        if identity(path)['sha256'] != expected_sha:
            raise ValueError(f'E3 来源锚点漂移: {relative}')
        source(path)
    train = json.loads((root / TRAIN_LOCK).read_text())
    test = json.loads((root / TEST_LOCK).read_text())
    manifest = json.loads((root / TEST_SUMMARY / 'manifest.json').read_text())
    summary_receipt = root / TEST_SUMMARY / 'summary_receipt.json'
    source(summary_receipt, manifest['files']['summary_receipt.json'])
    receipt = json.loads(summary_receipt.read_text())
    if tuple(receipt['seeds']) != SEEDS or tuple(test['seeds']) != SEEDS:
        raise ValueError('E3 冻结来源 seed 矩阵错误')
    entries, metadata = [], {}
    for seed, epoch in zip(SEEDS, (13, 15, 14), strict=True):
        old = next(item for item in train['w0_entries'] if item['seed'] == seed)
        entry = next(item for item in test['entries'] if item['seed'] == seed)
        if old['selected_epoch'] != epoch:
            raise ValueError('E3 W0 checkpoint identity 错误')
        run = root / old['run_dir']
        for split, metrics_name, summary_name in (
                ('validation', 'metrics.csv', 'metrics_summary.csv'),
                ('test', 'research_test_metrics.csv', 'research_test_metrics_summary.csv')):
            expected_metrics = None if split == 'validation' else entry['w0_test'][metrics_name]
            expected_summary = old['validation_summary'] if split == 'validation' else entry['w0_test'][summary_name]
            entries.append({'split': split, 'seed': seed, 'selected_epoch': epoch,
                            'checkpoint_sha256': old['checkpoint']['sha256'],
                            'metrics': source(run / metrics_name, expected_metrics),
                            'summary': source(run / summary_name, expected_summary)})
        # 三 seed 共享同一 split；其窗口顺序还会在 analyze 中逐项复核。
        if seed == SEEDS[0]:
            val_rows = Path(entry['training_attempt']) / 'val_rows.csv'
            metadata['validation'] = source(val_rows, test['source_files'][str(val_rows)])
            evaluation = Path(receipt['source_runs'][str(seed)]['path'])
            source(evaluation / 'manifest.json', receipt['source_runs'][str(seed)]['manifest'])
            evaluation_manifest = json.loads((evaluation / 'manifest.json').read_text())
            metadata['test'] = source(evaluation / 'test_rows.csv', evaluation_manifest['files']['test_rows.csv'])
    lock = {'contract': contract(), 'entries': entries, 'metadata': metadata, 'source_files': sources,
            'code_files': {str(path): identity(root / path) for path in CODE_PATHS},
            'prepared_at': datetime.now(timezone.utc).isoformat(), 'git': git_state(root),
            'access': 'existing metric CSV bytes, frozen summaries and window metadata; no signal arrays'}
    write_json(root / LOCK_PATH, lock)
    return root / LOCK_PATH


def load_lock(root: Path) -> tuple[dict, str]:
    path = root / LOCK_PATH
    lock = json.loads(path.read_text())
    if lock['contract'] != contract() or set(lock['code_files']) != set(map(str, CODE_PATHS)):
        raise ValueError('E3 统计合同或源码清单漂移')
    for relative, expected in lock['code_files'].items():
        verify(root / relative, expected)
    expected_matrix = {(split, seed) for split in COUNTS for seed in SEEDS}
    if len(lock['entries']) != len(expected_matrix) or {(e['split'], e['seed']) for e in lock['entries']} != expected_matrix:
        raise ValueError('E3 来源矩阵不完整或重复')
    required = [e[key] for e in lock['entries'] for key in ('metrics', 'summary')] + list(lock['metadata'].values())
    if set(lock['metadata']) != set(COUNTS) or not set(required).issubset(lock['source_files']):
        raise ValueError('E3 来源身份清单不完整')
    return lock, identity(path)['sha256']


def verify_sources(lock: dict) -> None:
    for path, expected in lock['source_files'].items():
        verify(Path(path), expected)


def analyze_tables(lock: dict) -> dict[str, pd.DataFrame]:
    frames, audit, membership = {}, [], []
    thresholds = None
    for split in ('validation', 'test'):
        rows = pd.read_csv(lock['metadata'][split])
        selected = nonoverlap_ids(rows)
        membership.append(rows[['dataset_row_id', 'samp_id', 'split', 'window_start_s', 'window_end_s']].assign(
            analysis_split=split, nonoverlap_selected=rows.dataset_row_id.isin(selected)))
        for seed in SEEDS:
            entry = next(e for e in lock['entries'] if e['split'] == split and e['seed'] == seed)
            metrics = pd.read_csv(entry['metrics'])
            summary = pd.read_csv(entry['summary'])
            frame = validate_frame(metrics, rows, summary, split)
            frames[(split, seed)] = frame
            audit.append({'split': split, 'seed': seed, 'expected_windows': COUNTS[split][0],
                          'observed_windows': len(frame), 'eligible_windows': len(frame), 'excluded_windows': 0,
                          'samp_count': frame.samp_id.nunique(), 'nonoverlap_windows': len(selected),
                          'prediction_degeneracy': 0})
        if split == 'validation':
            thresholds = calibrate({seed: frames[(split, seed)] for seed in SEEDS})
        else:
            if set(rows.samp_id) & set(frames[('validation', SEEDS[0])].samp_id):
                raise ValueError('E3 validation 与 test samp_id 交叉')
    association_rows, discordance_rows, macro_rows = [], [], []
    for (split, seed), frame in frames.items():
        selected = nonoverlap_ids(frame)
        for view, data in (('all_windows', frame), ('nonoverlap', frame[frame.dataset_row_id.isin(selected)])):
            context = {'split': split, 'seed': seed, 'view': view}
            association_rows.extend({**context, **r} for r in associations(data))
            ratios, macro = discordance(data, thresholds)
            discordance_rows.extend({**context, **r} for r in ratios)
            macro_rows.extend({**context, **r} for r in macro)
    association = pd.DataFrame(association_rows)
    ratios, macro = pd.DataFrame(discordance_rows), pd.DataFrame(macro_rows)
    ratio_for_summary = pd.concat([ratios[ratios.level.eq('pooled_windows')],
                                  macro.assign(level='macro_samp')], ignore_index=True)
    return {'source_audit': pd.DataFrame(audit), 'window_membership': pd.concat(membership, ignore_index=True),
            'thresholds': thresholds, 'associations': association, 'discordance': ratios, 'discordance_macro': macro,
            'association_seed_summary': seed_summary(association,
                ['split', 'view', 'level', 'samp_id', 'axis_x', 'axis_y'], ['rho']),
            'discordance_seed_summary': seed_summary(ratio_for_summary,
                ['split', 'view', 'level', 'axis', 'rr_limit_bpm', 'quantile'],
                ['joint_fraction', 'high_fraction', 'conditional_fraction'])}


TABLES = ('source_audit', 'window_membership', 'thresholds', 'associations', 'discordance',
          'discordance_macro', 'association_seed_summary', 'discordance_seed_summary')


def verify_completed(output: Path, lock_hash: str) -> None:
    if (output / 'lifecycle_failed.json').exists():
        raise ValueError('E3 attempt 有失败回执')
    freeze = json.loads((output / 'freeze_receipt.json').read_text())
    verify(output / 'manifest.json', freeze['manifest'])
    manifest = json.loads((output / 'manifest.json').read_text())
    if (manifest['protocol'] != PROTOCOL or manifest['lock_sha256'] != lock_hash
            or manifest['status'] != 'completed'):
        raise ValueError('E3 完成产物身份错误')
    required = {f'{name}.csv' for name in TABLES} | {'analysis_receipt.json', 'implementation_lock.json',
                'environment.json', 'lifecycle_started.json', 'lifecycle_completed.json'}
    if not required.issubset(manifest['files']):
        raise ValueError('E3 完成产物不完整')
    for relative, expected in manifest['files'].items():
        path = (output / relative).resolve()
        if not path.is_relative_to(output.resolve()):
            raise ValueError('E3 manifest 路径越界')
        verify(path, expected)


def run_analysis(root: Path = ROOT) -> Path:
    lock, lock_hash = load_lock(root)
    parent = root / OUTPUT
    parent.mkdir(parents=True, exist_ok=True)
    for old in sorted(parent.glob(f'analysis_{lock_hash[:12]}_*')):
        if (old / 'lifecycle_completed.json').exists() or (old / 'freeze_receipt.json').exists():
            verify_sources(lock)
            verify_completed(old, lock_hash)
            return old
    state = git_state(root)
    if state['status_porcelain']:
        raise RuntimeError('E3 正式分析前请提交实现与锁，保持工作树干净')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = parent / f'analysis_{lock_hash[:12]}_{stamp}_{uuid4().hex[:12]}'
    output.mkdir(exist_ok=False)
    context = {'protocol': PROTOCOL, 'lock_sha256': lock_hash, 'command': sys.argv,
               'started_at': datetime.now(timezone.utc).isoformat()}
    write_json(output / 'lifecycle_started.json', {**context, 'status': 'running'})
    print(f'E3 attempt: {output}', flush=True)
    try:
        write_json(output / 'implementation_lock.json', lock)
        write_json(output / 'environment.json', {'git': state, 'python': sys.version, 'platform': platform.platform(),
                   'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__})
        verify_sources(lock)
        tables = analyze_tables(lock)
        for name, table in tables.items():
            table.to_csv(output / f'{name}.csv', index=False, mode='x')
        # 防止读取期间发生来源变化；任何变化都保留失败 attempt。
        verify_sources(lock)
        write_json(output / 'analysis_receipt.json', {'contract': contract(),
                   'sources': lock['source_files'], 'table_rows': {k: len(v) for k, v in tables.items()},
                   'thresholds_identity': identity(output / 'thresholds.csv'),
                   'evidence_role': 'descriptive fixed-W0 distributions; repeated-use research test',
                   'undefined_statistics': {'association_rows': int(tables['associations'].rho.isna().sum()),
                     'conditional_fraction_rows': int(tables['discordance'].conditional_fraction.isna().sum())}})
        write_json(output / 'lifecycle_completed.json', {**context, 'status': 'completed',
                   'ended_at': datetime.now(timezone.utc).isoformat()})
        write_json(output / 'manifest.json', {**context, 'status': 'completed',
                   'files': {path.name: identity(path) for path in sorted(output.iterdir()) if path.is_file()}})
        write_json(output / 'freeze_receipt.json', {'protocol': PROTOCOL, 'manifest': identity(output / 'manifest.json')})
    except BaseException as error:
        write_json(output / 'lifecycle_failed.json', {**context, 'status': 'failed', 'error': str(error),
                   'error_type': type(error).__name__, 'traceback': traceback.format_exc()})
        raise
    verify_completed(output, lock_hash)
    return output
