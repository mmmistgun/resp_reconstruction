"""仅 synthetic CPU：分片覆盖、尾批、复用数值与损坏拒绝。"""
import importlib.util
from pathlib import Path
import json
import pytest

spec = importlib.util.spec_from_file_location('migration', Path(__file__).with_name('migrate.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
rc = m.rc


def test_partition_collection_and_missing_rejection(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(rc.torch.cuda, 'is_available', lambda: False)
    def forbidden(*args, **kwargs): raise AssertionError('禁止 CUDA 或重复采样')
    monkeypatch.setattr(rc.torch.cuda, '_lazy_init', forbidden)
    class Data:
        rows = rc.pd.DataFrame({'split': ['synthetic'] * 10, 'dataset_row_id': range(10)})
        def __len__(self): return 130
        def __getitem__(self, i): return {'x': rc.torch.full((1,600), float(i)), 'index': i}
    class Sampler:
        sampler = 'ddpm'; nfe = 50
        def identity(self): return {'synthetic': True, 'sampler': self.sampler}
    def kernel(model, condition, keys, spec, acceleration, checkpoints):
        for n in checkpoints: yield n, condition + n
    monkeypatch.setattr(rc, 'accelerated_prefixes', kernel)
    model = rc.torch.nn.Linear(1,1); data = Data(); sampler = Sampler(); base = {'synthetic': True}
    caches = []
    for name, start, stop in [('head',0,64), ('tail',64,130)]:
        out = tmp_path / name
        m.compute_range(model, data, sampler, (16,), out, base, start, stop)
        caches.append(out / 'batch_cache')
    identity = m.full_identity(base, sampler, (16,), 130)
    m.require_partition(caches, identity, 130, cut=64)
    monkeypatch.setattr(rc, 'accelerated_prefixes', forbidden)
    result, profile = m.ORIGINAL_PREDICT(model, data, sampler, (16,), tmp_path/'collected', base, resume_caches=caches)
    expected = rc.np.broadcast_to(rc.np.arange(130, dtype='float32')[:,None,None]+16, (130,1,600)).reshape(10,13,1,600)
    assert rc.np.array_equal(result[16], expected)
    assert profile[0]['reused_batches'] == 3
    tail = rc.load_cached_batch(caches[1], identity, 128, 130, (16,))[0][16]
    assert tail.shape == (2,1,600)
    marker = caches[1]/'batch_000128.json'; marker.unlink()
    with pytest.raises(ValueError, match='遗漏'): m.require_partition(caches, identity, 130, cut=64)


def test_bad_partition_boundary_rejected(tmp_path):
    with pytest.raises(ValueError, match='边界'):
        m.compute_range(None, list(range(130)), None, (16,), tmp_path/'invalid', {}, 1, 64)


def test_cli_requires_explicit_authorization(monkeypatch):
    monkeypatch.setattr('sys.argv', ['migrate.py', 'gpu2'])
    with pytest.raises(SystemExit) as error: m.main()
    assert error.value.code == 2
