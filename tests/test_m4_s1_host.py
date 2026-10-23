"""物理项目路径映射使用临时目录验证，不访问真实数据。"""
from copy import deepcopy

import pytest

from scripts.run_m4_s1_host import CANONICAL_PROJECT, canonical_output_identity, relocate_project_path


def test_project_relocation_and_traversal_rejection(tmp_path):
    original = CANONICAL_PROJECT / 'runs/source/cache/B/attempt/manifest.json'
    assert relocate_project_path(original, tmp_path) == tmp_path / original.relative_to(CANONICAL_PROJECT)
    for path in ('relative/file', str(CANONICAL_PROJECT) + '_other/file',
                 str(CANONICAL_PROJECT / '..' / 'other' / 'file')):
        with pytest.raises(ValueError):
            relocate_project_path(path, tmp_path)


def test_symlink_escape_rejected(tmp_path):
    (tmp_path / 'escape').symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match='越界'):
        relocate_project_path(CANONICAL_PROJECT / 'escape' / 'file', tmp_path)


def test_config_preserves_everything_except_physical_output_root(tmp_path):
    canonical = {'outputs': {'run_root': str(CANONICAL_PROJECT / 'runs/session/cells/B')},
                 'data': {'dataset_root': '/frozen/data', 'split': 'train'},
                 'model': {'seed': 20260814}, 'training': {'batch_size': 32}}
    remote = deepcopy(canonical)
    remote['outputs']['run_root'] = str(tmp_path / 'runs/session/cells/B')
    assert canonical_output_identity(remote, tmp_path) == canonical
    assert canonical_output_identity(deepcopy(canonical), tmp_path) == canonical
    changed = deepcopy(remote)
    changed['training']['batch_size'] = 16
    assert canonical_output_identity(changed, tmp_path) != canonical
    with pytest.raises(ValueError):
        canonical_output_identity({'outputs': {'run_root': '/unrelated/output'}}, tmp_path)
