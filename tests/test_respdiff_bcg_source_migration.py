"""跨主机迁移只改变路径前缀；样本、索引与文件身份仍严格检查。"""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from resp_train.respdiff_bcg import study_io
from resp_train.respdiff_bcg.baseband import config_contract
from resp_train.respdiff_bcg.runtime import sha256, source_file_inventory
from scripts.run_respdiff_bcg_baseband_v1 import fixture


@pytest.mark.parametrize('change', [None, 'row', 'order', 'signal', 'index', 'size', 'mtime'])
def test_migrated_source_keeps_all_identity_checks(tmp_path, monkeypatch, change):
    root, source = tmp_path / 'dataset', tmp_path / 'source'
    root.mkdir()
    source.mkdir()
    cfg = config_contract('source_equivalent')
    cfg.data.dataset_root = str(root)
    data = {split: fixture(root, cfg, split) for split in ('train', 'val')}
    rows = pd.concat([dataset.rows for dataset in data.values()], ignore_index=True)
    index = root / 'combined_index.csv'
    rows.to_csv(index, index=False)
    old_prefix = Path('/original-host/dataset')
    inventory = source_file_inventory(rows)
    inverse = {path: str(old_prefix / Path(path).relative_to(root)) for path in inventory}
    for split, dataset in data.items():
        recorded = dataset.rows.copy()
        for column in ('source_npz', 'target_source_npz'):
            recorded[column] = recorded[column].map(inverse)
        recorded.to_csv(source / f'{split}_parents.csv', index=False)
    identity = {'index_sha256': sha256(index),
                'source_file_stats': {inverse[path]: stat for path, stat in inventory.items()}}
    (source / 'data_identity.json').write_text(json.dumps(identity))
    if change in ('row', 'order', 'signal'):
        altered = pd.read_csv(source / 'train_parents.csv')
        if change == 'row':
            altered.loc[0, 'dataset_row_id'] += 100
        elif change == 'signal':
            altered.loc[0, 'target_signal_key'] = 'another_signal'
        else:
            altered = altered.iloc[::-1]
        altered.to_csv(source / 'train_parents.csv', index=False)
    elif change == 'index':
        with index.open('a') as stream:
            stream.write('\n')
    elif change in ('size', 'mtime'):
        selected = Path(next(iter(inventory)))
        if change == 'size':
            with selected.open('ab') as stream:
                stream.write(b'changed')
        else:
            stat = selected.stat()
            os.utime(selected, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    before = {path.name: sha256(path) for path in source.iterdir()}
    def bundle(cfg, *, split, **kwargs):
        selected = data[split]
        return SimpleNamespace(dataset=selected.parents.raw, rows=selected.rows, index_path=index)
    monkeypatch.setattr(study_io, 'build_window_data', bundle)
    report = tmp_path / 'mapping.json'
    if change is None:
        actual = study_io.source_data(cfg, source, mapping_output=report)
        assert set(actual) == {'train', 'val'}
        assert json.loads(report.read_text())['n_changed_paths'] == 8
    else:
        with pytest.raises(ValueError):
            study_io.source_data(cfg, source, mapping_output=report)
        assert not report.exists()
    assert before == {path.name: sha256(path) for path in source.iterdir()}


def test_source_path_mapping_rejects_ambiguous_or_escaping_paths(tmp_path):
    root = tmp_path / 'dataset'
    root.mkdir()
    file = root / 'one.npz'
    file.write_bytes(b'synthetic')
    assert study_io.source_path_mapping([str(file)], root) == {str(file): str(file)}
    bad = [['/old/another_dataset/one.npz'], ['/dataset/old/dataset/one.npz'],
           ['/old/dataset/../one.npz'], ['/host1/dataset/one.npz', '/host2/dataset/one.npz']]
    for paths in bad:
        with pytest.raises(ValueError):
            study_io.source_path_mapping(paths, root)
    outside = tmp_path / 'outside.npz'
    outside.write_bytes(b'synthetic')
    (root / 'escape.npz').symlink_to(outside)
    with pytest.raises(ValueError):
        study_io.source_path_mapping(['/old/dataset/escape.npz'], root)
