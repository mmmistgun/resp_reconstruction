"""M4-S1跨主机执行适配：只映射项目物理位置，保留规范配置与全部来源核验。"""
from contextlib import contextmanager
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_m4_s1_receipt_compatibility import canonical_stage_bindings

CANONICAL_PROJECT = Path('/mnt/disk_code/marques/resp_reconstruction')


def relocate_project_path(path, project_root):
    """历史引用必须位于指定项目根内；不接受旁路路径或父目录跳转。"""
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('项目引用不是合法绝对路径')
    try:
        relative = path.relative_to(CANONICAL_PROJECT)
    except ValueError as exc:
        raise ValueError('历史引用不属于本项目') from exc
    root = Path(project_root).resolve()
    result = root / relative
    if not result.resolve().is_relative_to(root):
        raise ValueError('映射后的项目引用越界')
    return result


def canonical_output_identity(value, project_root):
    """仅将输出目录的物理根归一；数据、训练与模型配置均保持原值。"""
    output = Path(value['outputs']['run_root'])
    root = Path(project_root).resolve()
    if output.is_relative_to(root):
        value['outputs']['run_root'] = str(CANONICAL_PROJECT / output.relative_to(root))
    elif not output.is_relative_to(CANONICAL_PROJECT):
        raise ValueError('配置输出目录不属于规范根或当前主机根')
    return value


@contextmanager
def host_paths(project_root):
    from scripts import m4_s1_cwt_parameters_artifacts as io
    from scripts import m4_s1_cwt_parameters_data as data
    from resp_train import patch_aligned_tf_training as legacy

    original_identity, original_cache = legacy.config_identity, data.cache_receipt

    def config_identity(cfg):
        return canonical_output_identity(original_identity(cfg), project_root)

    def cache_receipt(session, representation):
        frozen = io.require_data_scope(session)
        data_path, metadata = data.data_receipt(session, verify_inputs=False)
        path = io.completed(Path(session) / 'cache' / representation,
                            data.cache_binding(session, representation, data_path))
        if path is None:
            raise RuntimeError(f'未完成CWT缓存: {representation}')
        record = io.read_json(path / 'cache_record.json')
        if record['kind'] == 'local':
            return path, path
        if record['kind'] != 'legacy_reference':
            raise ValueError('未知cache来源类型')
        source = relocate_project_path(record['source_session'], project_root)
        io.verify(source / 'session.json', record['source_session_identity'])
        # 使用原完整检查核验数据/rows/源码/表示/依赖/缓存全部字节，仅翻译物理位置。
        cache = data.verify_legacy_cache(source, representation,
                                        frozen['representations'][representation], metadata, data_path)
        expected = relocate_project_path(record['cache'], project_root)
        if cache.resolve() != expected.resolve():
            raise ValueError('历史cache映射位置变化')
        io.verify(cache / 'manifest.json', record['cache_manifest'])
        return path, cache

    legacy.config_identity, data.cache_receipt = config_identity, cache_receipt
    try:
        yield
    finally:
        legacy.config_identity, data.cache_receipt = original_identity, original_cache


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--host-project-root', type=Path, required=True)
    args, rest = parser.parse_known_args()
    expected_source = args.host_project_root.resolve() / '.worktrees/m4-s1-cwt-parameters-v1'
    if Path(__file__).resolve().parents[1] != expected_source:
        raise ValueError('执行源码不属于声明的主机项目根')
    from scripts.run_m4_s1_cwt_parameters import main as frozen_main
    with canonical_stage_bindings(), host_paths(args.host_project_root):
        frozen_main(rest)


if __name__ == '__main__':
    main()
