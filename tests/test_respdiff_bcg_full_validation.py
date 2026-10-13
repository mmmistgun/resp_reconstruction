"""P5 的 synthetic CPU 定向验证：前缀、冻结身份、全量范围与九项汇总。"""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from resp_train.respdiff.model import RespDiffSpec
from resp_train.respdiff_bcg.baseband import RespDiffBCGBaseband
from resp_train.respdiff_bcg.runtime import seed_all, sha256
from resp_train.respdiff_bcg.sampling_accelerated import AccelerationSpec, accelerated_prefixes
from resp_train.respdiff_bcg.study_contract import METRICS, NAME, StudySamplerSpec
from resp_train.respdiff_bcg.study_extension import ExtensionSpec
from resp_train.respdiff_bcg import study_full_validation as full
from scripts.run_respdiff_bcg_full_validation_v1 import main


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    def forbidden(*args, **kwargs):
        raise AssertionError('synthetic 测试禁止 CUDA 初始化')
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)
    yield
    torch.set_num_threads(previous)


def manifest():
    return {'revision': full.REVISION, 'configurations': copy.deepcopy(full.CONFIGURATIONS),
            'sampling_seed': 20261003, 'batch_size': 64, 'trajectory_group_size': 1,
            'checkpoint_sources': [{'training_seed': seed, 'checkpoint_sha256': digest,
                'regime': 'long_stretched', 'update': 16000} for seed, digest in full.CHECKPOINTS.items()]}


def test_shared_n16_matches_standalone_and_preserves_model():
    seed_all(5, cpu_only=True)
    model = RespDiffBCGBaseband(RespDiffSpec(8, 1, 4), objective='source_equivalent').eval()
    before = full.state_digest(model)
    condition = torch.randn(2, 1, 600)
    keys = [('val', 2, 0), ('val', 7, 1)]
    with torch.no_grad():
        standalone = dict(accelerated_prefixes(model, condition, keys, StudySamplerSpec('ddim', 6, 16),
                          AccelerationSpec(1), checkpoints=(16,)))
        shared = dict(accelerated_prefixes(model, condition, keys, ExtensionSpec('ddim', 6, 64),
                     AccelerationSpec(1), checkpoints=(16, 64)))
    assert torch.equal(standalone[16], shared[16])
    assert torch.isfinite(shared[64]).all()
    assert full.state_digest(model) == before


@pytest.mark.parametrize('change', ['duplicate_seed', 'checkpoint', 'N', 'noise', 'batch'])
def test_manifest_rejects_contract_drift(change):
    value = manifest()
    full.validate_manifest(value)
    if change == 'duplicate_seed': value['checkpoint_sources'][1] = value['checkpoint_sources'][0]
    elif change == 'checkpoint': value['checkpoint_sources'][0]['checkpoint_sha256'] = 'changed'
    elif change == 'N': value['configurations'][0]['N'] = 128
    elif change == 'noise': value['sampling_seed'] = 20261013
    else: value['batch_size'] = 32
    with pytest.raises(ValueError): full.validate_manifest(value)


def test_selection_requires_fixed_reference():
    value = {'selected': copy.deepcopy(full.CONFIGURATIONS), 'selection_noise_seed': 20261003}
    full.validate_selection(value)
    value['selected'].pop()
    with pytest.raises(ValueError): full.validate_selection(value)


def test_cli_requires_data_authorization(tmp_path):
    args = ['run', '--output', str(tmp_path / 'out'), '--source-run', str(tmp_path / 'source'),
            '--manifest-run', str(tmp_path / 'manifest')]
    for flags in ([], ['--confirm-data-access', '--device', 'cpu']):
        with pytest.raises(SystemExit) as error: main(args + flags)
        assert error.value.code == 2


def test_full_orchestration_and_summary_bind_every_parent(tmp_path, monkeypatch):
    """模拟数值内核，实际执行保存/回执/九项汇总，并拒绝不完整或篡改产物。"""
    model = torch.nn.Linear(1, 1).eval()
    rows = pd.DataFrame({'dataset_row_id': [10, 11, 12]})
    chunks = pd.DataFrame({'index': list(range(39))})
    class Dataset:
        def __init__(self): self.rows = rows
        def __len__(self): return 39
    dataset = Dataset()
    reference = tmp_path / 'reference'; reference.mkdir()
    rows.to_csv(reference / 'parents.csv', index=False)
    chunks.to_csv(reference / 'chunks.csv', index=False)
    frozen = tmp_path / 'freeze'; frozen.mkdir()
    value = manifest()
    value.update(n_parents=3, n_chunks=39, files={},
        source_val_parents_sha256=sha256(reference / 'parents.csv'),
        source_val_chunks_sha256=sha256(reference / 'chunks.csv'))
    for checkpoint in value['checkpoint_sources']:
        checkpoint.update(source_run=str(tmp_path / str(checkpoint['training_seed'])),
                          parameter_sha256=full.state_digest(model))
    (frozen / 'final_evaluation_manifest.json').write_text(json.dumps(value))
    monkeypatch.setattr(full, 'read_manifest', lambda path: (frozen, value))
    def initialize(out, phase, source, device, **kwargs):
        checkpoint = next(c for c in value['checkpoint_sources'] if c['source_run'] == str(source))
        assert kwargs['expected_sha256'] == checkpoint['checkpoint_sha256']
        return model, OmegaConf.create({}), checkpoint, {'val': (None, rows, None)}, None, None
    monkeypatch.setattr(full, '_initialize', initialize)
    def select(parents, actual_rows, indices):
        assert indices == [0, 1, 2]
        return dataset
    monkeypatch.setattr(full, 'selected_dataset', select)
    monkeypatch.setattr(full, 'full_identity', lambda cfg: {'files': {}})
    monkeypatch.setattr(full, 'environment', lambda **kwargs: {})
    def save_data(out, datasets, index):
        assert datasets == {'val': dataset}
        rows.to_csv(out / 'val_parents.csv', index=False)
        chunks.to_csv(out / 'val_chunks.csv', index=False)
        return {}
    monkeypatch.setattr(full, 'data_identity', save_data)
    monkeypatch.setattr(full, 'verify_data_unchanged', lambda *args: None)
    monkeypatch.setattr(full, '_check_source', lambda cp, source, model: None)
    calls = []
    def predict(model, ds, spec, counts, out, **kwargs):
        assert ds is dataset and kwargs['batch_size'] == 64
        calls.append((spec.sampler, counts))
        return {n: np.zeros((3, 13, 1, 600), dtype=np.float32) for n in counts}, [{'N': n} for n in counts]
    monkeypatch.setattr(full, 'predict_prefixes', predict)
    def evaluate(predictions, ds, cfg, setting, spec, n, **kwargs):
        assert predictions.shape[0] == 3 and spec.n_trajectories == n
        return {'sampler': spec.sampler, 'N': n, 'nfe_per_trajectory': spec.nfe,
            'denoiser_calls_per_chunk': n * spec.nfe, 'n_parents': 3,
            METRICS[0]: .8, METRICS[1]: .5, METRICS[2]: 10., METRICS[3]: .1}
    monkeypatch.setattr(full, 'evaluate_setting', evaluate)
    def timing(model, ds, specs, **kwargs):
        assert [(s.sampler, s.n_trajectories) for s in specs] == [('ddim', 64), ('ddpm', 16), ('ddim', 16)]
        assert kwargs == {'batch_size': 64, 'warmup': 1, 'repeats': 2}
        return []
    monkeypatch.setattr(full, 'independent_timing', timing)
    runs = []
    for cp in value['checkpoint_sources']:
        run = tmp_path / f"run{cp['training_seed']}"
        full.run_full(run, source_run=cp['source_run'], manifest_run=frozen, device='cpu')
        assert json.loads((run / 'receipt.json').read_text())['status'] == 'complete'
        runs.append(run)
    assert calls == [('ddim', (16, 64)), ('ddpm', (16,))] * 3
    full.summarize_full(tmp_path / 'summary', manifest_run=frozen, runs=runs)
    frame = pd.read_csv(tmp_path / 'summary/full_validation_nine_configurations.csv')
    assert len(frame) == 9
    with pytest.raises(ValueError): full.validate_curves(frame.iloc[:-1], value, all_seeds=True)
    bad = frame.copy(); bad.loc[0, 'n_parents'] = 2
    with pytest.raises(ValueError): full.validate_curves(bad, value, all_seeds=True)
    bad = frame.copy(); bad.loc[0, METRICS[0]] = np.nan
    with pytest.raises(ValueError): full.validate_curves(bad, value, all_seeds=True)
    (runs[0] / 'sampler_grid.csv').write_text('tampered')
    with pytest.raises(ValueError, match='身份'):
        full.summarize_full(tmp_path / 'bad', manifest_run=frozen, runs=runs)
