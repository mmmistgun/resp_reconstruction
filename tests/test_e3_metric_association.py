"""E3 的 synthetic CPU 校验，全部来源在 pytest 临时目录。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from resp_train.paper_evidence import e3_metric_association as e3


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def synthetic(split='validation', seed_index=0):
    offset = 0 if split == 'validation' else 100
    samps = np.array([1] * 4 + [2] * 3 + [3] * 2) + offset
    starts = np.array([0, 60, 180, 240, 0, 60, 180, 0, 180])
    rows = pd.DataFrame({'dataset_row_id': np.arange(9) + offset, 'samp_id': samps,
                         'split': 'val' if split == 'validation' else 'test', 'coupling_state_id': 1,
                         'window_start_s': starts, 'window_end_s': starts + 180,
                         'window_duration_s': 180, 'source_npz': [f'{s}/signal.npz' for s in samps],
                         'segment_id': [0, 1, 1, 2, 0, 0, 1, 0, 1]})
    metrics = rows[['dataset_row_id', 'samp_id', 'split', 'coupling_state_id']].copy()
    metrics['method'] = 'crd_tf102_w'
    metrics['evaluation_split'] = split
    base = np.arange(1, 10) / 10
    for i, key in enumerate(e3.PRIMARY[:4]):
        metrics[key] = base * (i + 1) + seed_index * 0.01
    metrics[e3.PRIMARY[-1]] = 1 - base / 2 - seed_index * 0.01
    for flag in e3.FLAGS:
        metrics[flag] = True
    for flag in e3.DEGENERACY:
        metrics[flag] = False
    summary = pd.DataFrame([{**{k + '_mean': metrics[k].mean() for k in e3.PRIMARY},
                             **{k + '_n': len(metrics) for k in e3.PRIMARY}}])
    return metrics, rows, summary


@pytest.fixture
def counts(monkeypatch):
    monkeypatch.setattr(e3, 'COUNTS', {'validation': (9, 3), 'test': (9, 3)})


@pytest.fixture
def sources(tmp_path, monkeypatch, counts):
    """构造原 E2 身份锚点的最小真实文件链，覆盖 prepare-lock 到 completed。"""
    train = {'w0_entries': []}
    test = {'seeds': list(e3.SEEDS), 'entries': [], 'source_files': {}}
    receipt = {'seeds': list(e3.SEEDS), 'source_runs': {}}
    for i, (seed, epoch) in enumerate(zip(e3.SEEDS, (13, 15, 14), strict=True)):
        run = tmp_path / 'w0' / str(seed)
        run.mkdir(parents=True)
        formal = tmp_path / 'e2' / str(seed)
        formal.mkdir(parents=True)
        evaluation = tmp_path / 'test' / str(seed)
        evaluation.mkdir(parents=True)
        full = {}
        for split in e3.COUNTS:
            metrics, rows, summary = synthetic(split, i)
            metrics_name = 'metrics.csv' if split == 'validation' else 'research_test_metrics.csv'
            summary_name = 'metrics_summary.csv' if split == 'validation' else 'research_test_metrics_summary.csv'
            metrics.to_csv(run / metrics_name, index=False)
            summary.to_csv(run / summary_name, index=False)
            if split == 'test':
                full = {name: {'path': str(run / name), **e3.identity(run / name)} for name in (metrics_name, summary_name)}
                rows.to_csv(evaluation / 'test_rows.csv', index=False)
                save_json(evaluation / 'manifest.json', {'files': {'test_rows.csv': e3.identity(evaluation / 'test_rows.csv')}})
            else:
                rows.to_csv(formal / 'val_rows.csv', index=False)
        train['w0_entries'].append({'seed': seed, 'selected_epoch': epoch, 'run_dir': str(run),
                                   'checkpoint': {'sha256': 'a' * 64},
                                   'validation_summary': e3.identity(run / 'metrics_summary.csv')})
        test['entries'].append({'seed': seed, 'training_attempt': str(formal), 'w0_test': full})
        test['source_files'][str(formal / 'val_rows.csv')] = e3.identity(formal / 'val_rows.csv')
        receipt['source_runs'][str(seed)] = {'path': str(evaluation), 'manifest': e3.identity(evaluation / 'manifest.json')}
    save_json(tmp_path / e3.TRAIN_LOCK, train)
    save_json(tmp_path / e3.TEST_LOCK, test)
    save_json(tmp_path / e3.TEST_SUMMARY / 'summary_receipt.json', receipt)
    save_json(tmp_path / e3.TEST_SUMMARY / 'manifest.json', {
        'files': {'summary_receipt.json': e3.identity(tmp_path / e3.TEST_SUMMARY / 'summary_receipt.json')}})
    for constant, path in [('TRAIN_SHA', e3.TRAIN_LOCK), ('TEST_SHA', e3.TEST_LOCK), ('SUMMARY_SHA', e3.TEST_SUMMARY / 'manifest.json')]:
        monkeypatch.setattr(e3, constant, e3.identity(tmp_path / path)['sha256'])
    for path in e3.CODE_PATHS:
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text('synthetic source identity\n')
    monkeypatch.setattr(e3, 'git_state', lambda root: {'commit': 'synthetic', 'status_porcelain': ''})
    return tmp_path


def test_rank_ties_and_pcc_direction(counts):
    metrics, rows, summary = synthetic()
    frame = e3.validate_frame(metrics, rows, summary, 'validation')
    frame[e3.AXES[0]] = [1, 1, 2, 3, 3, 3, 4, 5, 6]
    frame[e3.AXES[1]] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    table = pd.DataFrame(e3.associations(frame))
    row = table.query("level == 'pooled_windows'").iloc[0]
    expected = np.corrcoef([1.5, 1.5, 3, 5, 5, 5, 7, 8, 9], np.arange(1, 10))[0, 1]
    assert row.rho == pytest.approx(expected)
    pcc = table[(table.level == 'pooled_windows') & (table.axis_x == e3.AXES[2]) & (table.axis_y == e3.AXES[-1])]
    assert pcc.iloc[0].rho == pytest.approx(1)
    frame[e3.AXES[0]] = 1
    assert e3.associations(frame)[0]['status'] == 'constant_input'
    assert e3.associations(frame.iloc[:2])[0]['status'] == 'insufficient_units'


def test_nonoverlap_cross_segment_and_order():
    _, rows, _ = synthetic()
    assert e3.nonoverlap_ids(rows) == {0, 2, 4, 6, 7, 8}
    assert e3.nonoverlap_ids(rows.sample(frac=1, random_state=3)) == {0, 2, 4, 6, 7, 8}


@pytest.mark.parametrize('fault', ['nan', 'infinity', 'duplicate', 'order', 'split', 'eligibility', 'degenerate', 'summary', 'duration', 'source_clock'])
def test_bad_sources_fail(counts, fault):
    metrics, rows, summary = synthetic()
    if fault == 'nan': metrics.loc[0, e3.PRIMARY[2]] = np.nan
    elif fault == 'infinity': metrics.loc[0, e3.PRIMARY[0]] = np.inf
    elif fault == 'duplicate': metrics.loc[0, 'dataset_row_id'] = 1
    elif fault == 'order': metrics = metrics.iloc[::-1]
    elif fault == 'split': metrics.loc[0, 'split'] = 'test'
    elif fault == 'eligibility': metrics.loc[0, e3.FLAGS[0]] = False
    elif fault == 'degenerate': metrics.loc[0, e3.DEGENERACY[0]] = True
    elif fault == 'summary': summary.loc[0, e3.PRIMARY[0] + '_mean'] += 0.1
    elif fault == 'duration': rows.loc[0, 'window_end_s'] = 200
    elif fault == 'source_clock': rows.loc[0, 'source_npz'] = 'other.npz'
    with pytest.raises((ValueError, FloatingPointError)):
        e3.validate_frame(metrics, rows, summary, 'validation')


def test_calibration_uses_unique_validation_window_means(counts):
    validation = {seed: e3.validate_frame(*synthetic(seed_index=i), 'validation') for i, seed in enumerate(e3.SEEDS)}
    thresholds = e3.calibrate(validation)
    values = np.arange(1, 10) / 10 * 3 + 0.01
    assert thresholds.iloc[1].threshold == pytest.approx(np.quantile(values, 0.75))
    assert len(thresholds) == 9
    validation[e3.SEEDS[0]] = validation[e3.SEEDS[0]].assign(split='test')
    with pytest.raises(ValueError, match='validation'):
        e3.calibrate(validation)


def test_conditional_and_macro_denominators(counts):
    frame = e3.validate_frame(*synthetic(), 'validation')
    frame[e3.AXES[0]] = 0.5
    frame[e3.AXES[1]] = 0.5
    frame.loc[frame.samp_id.eq(3), e3.AXES[1]] = 3
    frame[e3.AXES[2]] = [2, 2, 2, 0, 0, 0, 0, 2, 2]
    thresholds = pd.DataFrame([{'axis': e3.AXES[2], 'quantile': 0.75, 'threshold': 1.0}])
    ratios, macro = map(pd.DataFrame, e3.discordance(frame, thresholds))
    pooled = ratios[(ratios.level == 'pooled_windows') & (ratios.rr_limit_bpm == 1)].iloc[0]
    assert (pooled.n_total, pooled.n_rr, pooled.n_high, pooled.n_joint) == (9, 7, 5, 3)
    assert pooled.conditional_fraction == pytest.approx(3 / 7)
    assert pooled.joint_fraction == pytest.approx(3 / 9)
    row = macro[macro.rr_limit_bpm == 1].iloc[0]
    assert row.conditional_fraction == pytest.approx((3 / 4 + 0) / 2)
    assert row.joint_fraction == pytest.approx((3 / 4 + 0 + 0) / 3)
    assert (row.n_samp_conditional_defined, row.n_samp_zero_rr) == (2, 1)
    frame[e3.AXES[0]] = 100
    ratios, macro = e3.discordance(frame, thresholds)
    assert all(r['status'] == 'no_rr_qualified_windows' and np.isnan(r['conditional_fraction']) for r in ratios)
    assert all(np.isnan(r['conditional_fraction']) for r in macro)


def test_full_lifecycle_sources_unchanged_and_reuse(sources):
    lock_path = e3.prepare_lock(sources)
    lock = json.loads(lock_path.read_text())
    with pytest.raises(FileExistsError): e3.prepare_lock(sources)
    output = e3.run_analysis(sources)
    e3.verify_sources(lock)
    e3.verify_completed(output, e3.identity(lock_path)['sha256'])
    assert e3.run_analysis(sources) == output
    audit = pd.read_csv(output / 'source_audit.csv')
    assert len(audit) == 6 and set(audit.excluded_windows) == {0}
    assert set(audit.nonoverlap_windows) == {6}
    assert len(pd.read_csv(output / 'thresholds.csv')) == 9
    association = pd.read_csv(output / 'associations.csv')
    assert len(association) == 2 * 3 * 2 * (3 + 2) * 10
    ratios = pd.read_csv(output / 'discordance.csv')
    assert len(ratios) == 2 * 3 * 2 * (3 + 1) * 3 * 3 * 3
    assert len(pd.read_csv(output / 'window_membership.csv')) == 18
    (output / 'thresholds.csv').write_text('corrupt')
    with pytest.raises(ValueError, match='身份漂移'): e3.run_analysis(sources)


def test_test_values_do_not_change_thresholds(sources):
    lock = json.loads(e3.prepare_lock(sources).read_text())
    before = e3.analyze_tables(lock)['thresholds']
    for entry in lock['entries']:
        if entry['split'] != 'test': continue
        frame = pd.read_csv(entry['metrics'])
        frame[e3.PRIMARY[2]] *= 100
        frame.to_csv(entry['metrics'], index=False)
        summary = pd.read_csv(entry['summary'])
        summary[e3.PRIMARY[2] + '_mean'] = frame[e3.PRIMARY[2]].mean()
        summary.to_csv(entry['summary'], index=False)
    after = e3.analyze_tables(lock)['thresholds']
    pd.testing.assert_frame_equal(before, after)


def test_source_drift_preserves_failed_attempt(sources):
    lock = json.loads(e3.prepare_lock(sources).read_text())
    Path(lock['entries'][0]['metrics']).write_text('changed source')
    with pytest.raises(ValueError, match='身份漂移'): e3.run_analysis(sources)
    failed = list((sources / e3.OUTPUT).glob('*/lifecycle_failed.json'))
    assert len(failed) == 1
    assert not (failed[0].parent / 'freeze_receipt.json').exists()


def test_code_drift_and_incomplete_seed_matrix(sources):
    e3.prepare_lock(sources)
    source = sources / e3.CODE_PATHS[0]
    source.write_text('changed code')
    with pytest.raises(ValueError, match='身份漂移'): e3.run_analysis(sources)
    incomplete = pd.DataFrame({'split': ['test', 'test'], 'seed': list(e3.SEEDS[:2]), 'rho': [0.1, 0.2]})
    with pytest.raises(ValueError, match='矩阵不完整'): e3.seed_summary(incomplete, ['split'], ['rho'])


def test_seed_sd_and_undefined_counts():
    frame = pd.DataFrame({'split': ['test'] * 3, 'seed': e3.SEEDS, 'rho': [1.0, 0.0, np.nan]})
    row = e3.seed_summary(frame, ['split'], ['rho']).iloc[0]
    assert row['mean'] == 0.5 and row.sample_sd == pytest.approx(np.sqrt(0.5))
    assert row.n_seeds_defined == 2 and row.status == 'partial_seeds'
