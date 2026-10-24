"""P6 专项授权、噪声域、数据隔离与数值兼容的 synthetic CPU 验收。"""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from resp_train.respdiff.model import RespDiffSpec
from resp_train.respdiff_bcg.baseband import RespDiffBCGBaseband, baseband_dataset, config_contract
from resp_train.respdiff_bcg.inference_v2 import evaluate_setting
from resp_train.respdiff_bcg.research_test import ResearchTestSpec, canonical_rows, check_isolation, frame_hash
from resp_train.respdiff_bcg.runtime import seed_all
from resp_train.respdiff_bcg.sampling import trajectory_parent_noise, sample_trajectory
from resp_train.respdiff_bcg.sampling_accelerated import AccelerationSpec, accelerated_prefixes
from resp_train.respdiff_bcg.signal import STARTS, parent_noise
from resp_train.respdiff_bcg.study_contract import StudySamplerSpec
from scripts.run_respdiff_bcg_research_test_v1 import main


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    old = torch.get_num_threads(); torch.set_num_threads(1)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    def forbidden(*args, **kwargs): raise AssertionError('CPU 合成检查禁止 CUDA')
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize('split', ['val', 'test'])
@pytest.mark.parametrize('trajectory,step', [(0, None), (3, None), (2, 49)])
def test_noise_preserves_hash_algorithm_and_real_split(split, trajectory, step):
    args = dict(split=split, row_id=7, trajectory_id=trajectory, noise_seed=20261003, reverse_step=step)
    if split == 'test':
        with pytest.raises(ValueError): trajectory_parent_noise(**args)
    actual = trajectory_parent_noise(**args, allow_research_test=True)
    values = (['respdiff-bcg-noise-v1', split, 7, 20261003] if trajectory == 0 and step is None
              else ['respdiff-bcg-trajectory-noise-v2', split, 7, trajectory, 20261003, 'initial' if step is None else step])
    seed = int.from_bytes(hashlib.sha256(json.dumps(values).encode()).digest()[:8], 'little') % 2**63
    field = torch.randn(4200, generator=torch.Generator(device='cpu').manual_seed(seed))
    expected = torch.stack([field[s:s+600] for s in STARTS])[:, None]
    assert torch.equal(actual, expected)
    if split == 'test':
        assert not torch.equal(actual, trajectory_parent_noise(**{**args, 'split': 'val'}))


@pytest.mark.parametrize('kind,steps,n', [('ddim', 6, 16), ('ddpm', 50, 16)])
def test_cached_test_trajectory_matches_reference(kind, steps, n):
    seed_all(17, cpu_only=True)
    model = RespDiffBCGBaseband(RespDiffSpec(8, 1, 4), objective='source_equivalent').eval()
    condition = torch.randn(2, 1, 600)
    spec = ResearchTestSpec(kind, steps, n)
    keys = [('test', 3, 0), ('test', 4, 0)]
    expected = sample_trajectory(model, condition, keys, spec, 0)
    from resp_train.respdiff_bcg.sampling_accelerated import encode_condition, sample_group
    actual = sample_group(model, condition, keys, spec, (0,), encode_condition(model.diffusion_model, condition))[0]
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    assert torch.isfinite(actual).all()


def test_shared_test_prefix_and_val_compatibility():
    seed_all(8, cpu_only=True)
    model = RespDiffBCGBaseband(RespDiffSpec(8, 1, 4), objective='source_equivalent').eval()
    x = torch.randn(2, 1, 600)
    for split in ('val', 'test'):
        keys = [(split, 1, 0), (split, 2, 0)]
        old = ResearchTestSpec('ddim', 6, 16) if split == 'test' else StudySamplerSpec('ddim', 6, 16)
        expected = dict(accelerated_prefixes(model, x, keys, old, AccelerationSpec(1), checkpoints=(16,)))
        actual = dict(accelerated_prefixes(model, x, keys, ResearchTestSpec('ddim', 6, 64), AccelerationSpec(1), checkpoints=(16, 64)))
        assert torch.equal(actual[16], expected[16])


def rows(tmp_path, split, subject, row_id):
    file = tmp_path / f'{subject}.npz'; file.touch()
    return pd.DataFrame([dict(dataset_row_id=row_id, split=split, samp_id=subject,
        source_npz=str(file), target_source_npz=str(file), window_start_sample=0, window_end_sample=18000)])


def test_isolation_rejects_subject_record_and_row_overlap(tmp_path):
    frames = {s: rows(tmp_path, s, i, i) for i, s in enumerate(('train', 'val', 'test'), 1)}
    check_isolation({s: SimpleNamespace(rows=f) for s, f in frames.items()})
    for column in ('samp_id', 'dataset_row_id', 'source_npz'):
        bad = {s: f.copy() for s, f in frames.items()}
        bad['test'][column] = bad['train'][column].iloc[0]
        with pytest.raises(ValueError): check_isolation({s: SimpleNamespace(rows=f) for s, f in bad.items()})


def test_data_and_output_keep_test_identity(tmp_path):
    frame = rows(tmp_path, 'test', 3, 12)
    class Parents:
        rows = frame
        index_csv_path = str(tmp_path / 'index.csv')
        def __len__(self): return 1
        def __getitem__(self, i):
            values = torch.sin(torch.arange(18000) * (2 * np.pi * .2 / 100)).reshape(1, -1)
            return {'x': values, 'target': values, 'meta': {'dataset_row_id': 12}}
    with pytest.raises(ValueError): baseband_dataset(Parents(), frame)
    ds = baseband_dataset(Parents(), frame, allow_research_test=True)
    chunks = np.stack([ds[i]['target'].numpy() for i in range(13)])[None]
    spec = ResearchTestSpec('ddim', 6, 16)
    with pytest.raises(ValueError): evaluate_setting(chunks, ds, config_contract('source_equivalent'), tmp_path/'bad', spec, 16)
    evaluate_setting(chunks, ds, config_contract('source_equivalent'), tmp_path/'out', spec, 16, output_prefix='research_test')
    with np.load(tmp_path/'out/research_test_postfiltered_waveforms.npz') as f: assert set(f['split']) == {'test'}
    assert set(pd.read_csv(tmp_path/'out/research_test_postfiltered_per_parent.csv').split) == {'test'}
    assert not list((tmp_path/'out').glob('validation*'))


def test_relative_identity_is_portable_and_ordered(tmp_path):
    a=tmp_path/'a'/'dataset'; b=tmp_path/'b'/'dataset'; a.mkdir(parents=True); b.mkdir(parents=True)
    left = rows(a, 'test', 7, 12); right = rows(b, 'test', 7, 12)
    assert frame_hash(canonical_rows(left, a)) == frame_hash(canonical_rows(right, b))
    right['dataset_row_id'] = 13
    assert frame_hash(canonical_rows(left, a)) != frame_hash(canonical_rows(right, b))


def test_budget_and_cli_authorization(tmp_path):
    for args in [('ddim', 6, 32), ('ddpm', 50, 4), ('ddim', 10, 64), ('ddim', 6, 64, 20261013)]:
        with pytest.raises(ValueError): ResearchTestSpec(*args)
    for phase in ('freeze', 'preflight', 'run', 'gpu-check'):
        with pytest.raises(SystemExit) as error: main([phase, '--output', str(tmp_path/'out')])
        assert error.value.code == 2
    with pytest.raises(ValueError): parent_noise(split='test', row_id=1, seed=0)
