"""P6 固定九项 research-test：跨主机身份、显式准入和三模型汇总。"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch

from resp_train.data.factory import build_window_data
from resp_train.respdiff.data import REQUIRED, assert_split_independence, validate_rows
from .baseband import baseband_dataset, config_contract, make_baseband_model
from .inference_v2 import evaluate_setting, predict_prefixes, state_digest
from .runtime import LOGGER, environment, save_source_snapshot, seed_all, sha256, write_json, source_file_inventory
from .sampling import sample_trajectory
from .sampling_accelerated import AccelerationSpec, accelerated_prefixes
from .study_contract import METRICS, StudySamplerSpec
from .study_evaluation import independent_timing
from .study_full_validation import CHECKPOINTS, CONFIGURATIONS, full_identity, validate_curves
from .study_io import ROOT, data_identity, load_source, study_run, verify_artifact, verify_data_unchanged
from .study_state import preserved_evaluation

PROTOCOL = 'respdiff-bcg-research-test-v1-20261019'
STUDY = Path('runs/respdiff_bcg_research_test_v1')
INPUTS = STUDY / 'inputs_v1'
PROTOCOL_FILE = 'docs/experiments/respdiff_bcg_research_test_v1_protocol_20261019.md'
CHANGED_FILES = {'resp_train/respdiff/data.py', 'resp_train/respdiff_bcg/data.py',
    'resp_train/respdiff_bcg/baseband.py', 'resp_train/respdiff_bcg/signal.py',
    'resp_train/respdiff_bcg/sampling.py', 'resp_train/respdiff_bcg/sampling_accelerated.py',
    'resp_train/respdiff_bcg/inference_v2.py'}


@dataclass(frozen=True)
class ResearchTestSpec(StudySamplerSpec):
    research_test: bool = True

    def __post_init__(self):
        if (type(self.nfe) is not int or type(self.n_trajectories) is not int
                or (self.sampler, self.nfe, self.n_trajectories) not in {('ddim', 6, 16), ('ddim', 6, 64), ('ddpm', 50, 16)}
                or type(self.noise_seed) is not int or self.noise_seed != 20261003
                or self.eta != 0 or self.postfilter is not True or self.research_test is not True):
            raise ValueError('P6 只允许三组冻结配置与 sampling seed20261003')


def identity(cfg):
    result = full_identity(cfg)
    for name in ('scripts/run_respdiff_bcg_research_test_v1.py', PROTOCOL_FILE):
        result['files'][name] = sha256(ROOT / name)
    return result


def model_path(seed):
    return ROOT / INPUTS / f'seed{seed}'


def canonical_rows(rows, dataset_root):
    """迁移只改变数据集根前缀；其下相对文件身份和行顺序均保留。"""
    root = Path(dataset_root).resolve()
    frame = rows[list(REQUIRED) + [c for c in ('bcg_signal_key', 'target_signal_key') if c in rows]].copy()
    for column in ('source_npz', 'target_source_npz'):
        def relative(value):
            path = Path(value)
            if not path.is_absolute():
                path = root / 'training' / path
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                positions = [i for i, part in enumerate(path.parts) if part == root.name]
                if len(positions) != 1 or '..' in path.parts:
                    raise ValueError('数据路径不能唯一映射到冻结数据集')
                resolved = (root / Path(*path.parts[positions[0] + 1:])).resolve()
            if not resolved.is_relative_to(root) or not resolved.is_file():
                raise ValueError('数据来源文件越界或缺失')
            return str(resolved.relative_to(root))
        frame[column] = frame[column].map(relative)
    return frame.reset_index(drop=True)


def frame_hash(frame):
    return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()


def check_isolation(datasets):
    rows = {split: ds.rows for split, ds in datasets.items()}
    assert_split_independence(rows['train'], rows['val'])
    validate_rows(rows['test'], allow_research_test=True)
    if set(rows['test'].split) != {'test'}:
        raise ValueError('专项数据集必须全部为 test')
    for split in ('train', 'val'):
        for column in ('samp_id', 'dataset_row_id'):
            if set(rows[split][column]) & set(rows['test'][column]):
                raise ValueError(f'{split}/test 隔离失败：{column}')
        records = set(rows[split].source_npz) | set(rows[split].target_source_npz)
        if records & (set(rows['test'].source_npz) | set(rows['test'].target_source_npz)):
            raise ValueError(f'{split}/test 来源文件交叉')


def datasets_for(cfg):
    datasets, index = {}, None
    for split in ('train', 'val', 'test'):
        bundle = build_window_data(cfg, split=split, max_windows=None, sample_strategy='stratified_random',
                                   sample_seed=20260612 if split == 'test' else int(cfg.data[f'{split}_sample_seed']),
                                   shuffle=False)
        datasets[split] = baseband_dataset(bundle.dataset, bundle.rows, allow_research_test=split == 'test')
        index = bundle.index_path
    check_isolation(datasets)
    return datasets, index


def read_manifest(path):
    path = Path(path)
    receipt = json.loads((path / 'receipt.json').read_text())
    if receipt.get('status') != 'complete' or receipt.get('research_test_protocol') != PROTOCOL:
        raise ValueError('P6 manifest 未完整冻结')
    verify_artifact(path, receipt, 'research_test_manifest.json')
    result = json.loads((path / 'research_test_manifest.json').read_text())
    if (result['protocol'] != PROTOCOL or result['configurations'] != CONFIGURATIONS
            or result['sampling_seed'] != 20261003 or result['batch_size'] != 64
            or result['trajectory_group_size'] != 1 or result['checkpoints'] != {str(k): v for k, v in CHECKPOINTS.items()}):
        raise ValueError('P6 冻结配置改变')
    for name, expected in result['files'].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f'P6 执行文件变化：{name}')
    return result


def freeze(output):
    """正式预测前冻结全体样本及其文件内容；不执行模型前向。"""
    with study_run(output, 'freeze-research-test', synthetic=False) as (out, receipt):
        receipt['research_test_protocol'] = PROTOCOL
        p5root = ROOT / INPUTS / 'p5_manifest'
        p5receipt = json.loads((p5root / 'receipt.json').read_text())
        if p5receipt.get('status') != 'complete':
            raise ValueError('P5 来源清单未完成')
        verify_artifact(p5root, p5receipt, 'final_evaluation_manifest.json')
        p5 = json.loads((p5root / 'final_evaluation_manifest.json').read_text())
        if p5['configurations'] != CONFIGURATIONS or p5['sampling_seed'] != 20261003:
            raise ValueError('P5 冻结选择不匹配')
        changes = {}
        for name, expected in p5['files'].items():
            actual = sha256(ROOT / name)
            if actual != expected:
                if name not in CHANGED_FILES:
                    raise ValueError(f'未声明的 P5 源码变化：{name}')
                changes[name] = {'p5': expected, 'p6': actual}
        models = []
        for seed, expected in CHECKPOINTS.items():
            model, cfg, cp = load_source(model_path(seed), torch.device('cpu'), expected_sha256=expected)
            if cp['training_seed'] != seed or cp['regime'] != 'long_stretched' or cp['update'] != 16000:
                raise ValueError('模型训练合同不匹配')
            models.append({k: v for k, v in cp.items() if k != 'source_run'})
            del model
        datasets, index = datasets_for(cfg)
        identities = {split: canonical_rows(ds.rows, cfg.data.dataset_root) for split, ds in datasets.items()}
        for seed in CHECKPOINTS:
            source = model_path(seed)
            old = json.loads((source / 'data_identity.json').read_text())
            if sha256(index) != old['index_sha256']:
                raise ValueError('测试索引不同于原训练数据版本')
            for split in ('train', 'val'):
                if frame_hash(canonical_rows(pd.read_csv(source / f'{split}_parents.csv'), cfg.data.dataset_root)) != frame_hash(identities[split]):
                    raise ValueError('train/validation 身份偏离原训练')
        names = sorted({str(value) for frame in identities.values()
                        for column in ('source_npz', 'target_source_npz') for value in frame[column]})
        data_root = Path(cfg.data.dataset_root).resolve()
        content = {}
        for i, name in enumerate(names, 1):
            content[name] = sha256(data_root / name)
            LOGGER.info('冻结数据内容 %d/%d：%s', i, len(names), name)
        current = identity(cfg)
        manifest = {'protocol': PROTOCOL, 'checkpoints': {str(k): v for k, v in CHECKPOINTS.items()},
            'models': models, 'configurations': CONFIGURATIONS, 'sampling_seed': 20261003,
            'batch_size': 64, 'trajectory_group_size': 1, 'files': current['files'],
            'dataset_name': data_root.name, 'index_sha256': sha256(index), 'data_content_sha256': content,
            'input_files': {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / INPUTS).rglob('*')) if p.is_file()},
            'splits': {split: {'n_parents': len(ds.rows), 'n_chunks': len(ds),
                       'subject_ids': sorted(ds.rows.samp_id.astype(int).unique().tolist()),
                       'canonical_rows_sha256': frame_hash(identities[split])} for split, ds in datasets.items()},
            'n_parents': len(datasets['test'].rows), 'n_chunks': len(datasets['test']),
            'p5_manifest_sha256': sha256(p5root / 'final_evaluation_manifest.json'), 'p5_to_p6_changes': changes}
        write_json(out / 'research_test_manifest.json', manifest)
        for split, frame in identities.items():
            frame.to_csv(out / f'{split}_canonical_parents.csv', index=False)
        write_json(out / 'source_identity.json', current)
        save_source_snapshot(out, current)
        write_json(out / 'environment.json', environment(cpu_only=True))


def preflight(output, manifest_run):
    manifest = read_manifest(manifest_run)
    with study_run(output, 'research-test-preflight', synthetic=False) as (out, receipt):
        receipt['research_test_protocol'] = PROTOCOL
        for name, expected in manifest['input_files'].items():
            if sha256(ROOT / name) != expected:
                raise ValueError(f'传输输入损坏：{name}')
        cfg = OmegaConf.load(model_path(20260811) / 'resolved_config.yaml')
        datasets, index = datasets_for(cfg)
        for split, ds in datasets.items():
            if frame_hash(canonical_rows(ds.rows, cfg.data.dataset_root)) != manifest['splits'][split]['canonical_rows_sha256']:
                raise ValueError(f'远程 {split} 有序身份不匹配')
        if sha256(index) != manifest['index_sha256']:
            raise ValueError('远程索引内容变化')
        root = Path(cfg.data.dataset_root).resolve()
        for i, (name, expected) in enumerate(manifest['data_content_sha256'].items(), 1):
            if sha256(root / name) != expected:
                raise ValueError(f'远程数据内容不一致：{name}')
            LOGGER.info('远程内容核验 %d/%d', i, len(manifest['data_content_sha256']))
        rows = pd.concat([ds.rows for ds in datasets.values()], ignore_index=True)
        write_json(out / 'remote_identity.json', {'manifest_sha256': sha256(Path(manifest_run) / 'research_test_manifest.json'),
            'dataset_root': str(root), 'index_sha256': sha256(index), 'source_file_stats': source_file_inventory(rows),
            'host': os.uname().nodename, 'root': str(ROOT)})
        write_json(out / 'environment.json', environment(cpu_only=True))


def verify_preflight(path, manifest_run, datasets, index):
    path = Path(path)
    receipt = json.loads((path / 'receipt.json').read_text())
    if receipt.get('status') != 'complete' or receipt.get('research_test_protocol') != PROTOCOL:
        raise ValueError('远程 preflight 未完成')
    verify_artifact(path, receipt, 'remote_identity.json')
    value = json.loads((path / 'remote_identity.json').read_text())
    rows = pd.concat([ds.rows for ds in datasets.values()], ignore_index=True)
    if (value['manifest_sha256'] != sha256(Path(manifest_run) / 'research_test_manifest.json')
            or value['host'] != os.uname().nodename or value['root'] != str(ROOT)
            or value['index_sha256'] != sha256(index) or value['source_file_stats'] != source_file_inventory(rows)):
        raise ValueError('当前数据/主机不同于已核验部署')


def gpu_check(output, *, device):
    with study_run(output, 'research-test-gpu-check', synthetic=True) as (out, receipt):
        receipt['research_test_protocol'] = PROTOCOL
        cfg = config_contract('source_equivalent')
        seed_all(20261019)
        model = make_baseband_model(cfg).to(device).eval()
        before = state_digest(model)
        condition = torch.randn(64, 1, 600, device=device)
        keys = [('synthetic', i // 13, i % 13) for i in range(64)]
        records = []
        for sampler, steps in [('ddim', 6), ('ddpm', 50)]:
            spec = SimpleNamespace(sampler=sampler, nfe=steps, n_trajectories=1, noise_seed=20261003,
                timesteps=(49, 39, 29, 19, 9, 0) if steps == 6 else tuple(range(49, -1, -1)), research_test=True)
            expected = sample_trajectory(model, condition, keys, spec, 0)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            actual = dict(accelerated_prefixes(model, condition, keys, spec, AccelerationSpec(1), checkpoints=(1,)))[1]
            torch.cuda.synchronize()
            passed = bool(torch.allclose(expected, actual, atol=1e-5, rtol=1e-4))
            record = {'sampler': sampler, 'passed': passed, 'batch_size': 64, 'N': 1,
                'sampling_seconds': time.perf_counter() - started,
                'max_abs_difference': float((expected - actual).abs().max()),
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'atol': 1e-5, 'rtol': 1e-4}
            records.append(record); write_json(out / 'comparisons.json', records)
            if not passed or not torch.isfinite(actual).all():
                raise ValueError('原尺寸 GPU synthetic 数值对比失败')
        if state_digest(model) != before:
            raise ValueError('GPU 验收改变模型参数/buffer')
        write_json(out / 'environment.json', {**environment(cpu_only=False),
                   'gpu': torch.cuda.get_device_name(), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')})
        receipt.update(comparisons=records, parameter_sha256_after=before)


def run_test(output, *, seed, manifest_run, preflight_run, device):
    manifest = read_manifest(manifest_run)
    with study_run(output, 'research-test', synthetic=False, source=model_path(seed)) as (out, receipt):
        receipt['research_test_protocol'] = PROTOCOL
        seed_all(20260811)
        model, cfg, checkpoint = load_source(model_path(seed), torch.device(device), expected_sha256=CHECKPOINTS[seed])
        expected = next(row for row in manifest['models'] if row['training_seed'] == seed)
        if {k: v for k, v in checkpoint.items() if k != 'source_run'} != expected:
            raise ValueError('实际模型与 P6 冻结清单不符')
        datasets, index = datasets_for(cfg)
        verify_preflight(preflight_run, manifest_run, datasets, index)
        for split, ds in datasets.items():
            if frame_hash(canonical_rows(ds.rows, cfg.data.dataset_root)) != manifest['splits'][split]['canonical_rows_sha256']:
                raise ValueError('运行样本身份改变')
        dataset = datasets['test']
        code = identity(cfg)
        OmegaConf.save(cfg, out / 'resolved_config.yaml')
        write_json(out / 'source_identity.json', code); save_source_snapshot(out, code)
        write_json(out / 'environment.json', environment(cpu_only=False))
        write_json(out / 'execution.json', {'protocol': PROTOCOL, 'seed': seed, 'checkpoint': checkpoint,
            'device': device, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'manifest_run': str(Path(manifest_run).resolve()), 'preflight_run': str(Path(preflight_run).resolve())})
        write_json(out / 'research_test_manifest.json', manifest)
        data = data_identity(out, {'test': dataset}, index)
        canonical_rows(dataset.rows, cfg.data.dataset_root).to_csv(out / 'test_canonical_parents.csv', index=False)
        curves, profiles = [], []
        with preserved_evaluation(model):
            for spec, counts in [(ResearchTestSpec('ddim', 6, 64), (16, 64)), (ResearchTestSpec('ddpm', 50, 16), (16,))]:
                sampling = out / f'{spec.sampler}_sampling'; sampling.mkdir()
                predictions, profile = predict_prefixes(model, dataset, spec, counts, sampling,
                    batch_size=64, acceleration=AccelerationSpec(1))
                profiles.extend(profile)
                for n in counts:
                    actual = ResearchTestSpec(spec.sampler, spec.nfe, n)
                    row = evaluate_setting(predictions.pop(n), dataset, cfg,
                        out / f'{spec.sampler}_S{spec.nfe}_N{n}', actual, n,
                        method_prefix=PROTOCOL, output_prefix='research_test')
                    row.update(training_seed=seed, sampling_seed=20261003, checkpoint_sha256=CHECKPOINTS[seed],
                        role=next(c['role'] for c in CONFIGURATIONS if (c['sampler'], c['N']) == (spec.sampler, n)))
                    curves.append(row)
                    pd.DataFrame(curves).to_csv(out / 'sampler_grid.csv', index=False)
                    pd.DataFrame(profiles).to_csv(out / 'runtime_profile.csv', index=False)
        validate_curves(pd.DataFrame(curves), manifest, all_seeds=False)
        timings = independent_timing(model, dataset, [ResearchTestSpec(c['sampler'], c['S'], c['N']) for c in CONFIGURATIONS], batch_size=64)
        write_json(out / 'independent_runtime.json', timings)
        verify_data_unchanged(data, dataset.rows)
        read_manifest(manifest_run)
        if (state_digest(model) != checkpoint['parameter_sha256'] or sha256(model_path(seed) / 'final.pt') != CHECKPOINTS[seed]
                or identity(cfg) != code):
            raise ValueError('运行期间模型/源码身份改变')
        receipt.update(checkpoint=checkpoint, n_parents=len(dataset.rows), n_chunks=len(dataset),
            manifest_sha256=sha256(Path(manifest_run) / 'research_test_manifest.json'),
            parameter_sha256_after=state_digest(model))


def summarize(output, *, runs, manifest_run):
    manifest = read_manifest(manifest_run)
    frames, sources = [], []
    for value in runs:
        path = Path(value); receipt = json.loads((path / 'receipt.json').read_text())
        if (receipt.get('status') != 'complete' or receipt.get('phase') != 'research-test'
                or receipt.get('research_test_protocol') != PROTOCOL
                or receipt['manifest_sha256'] != sha256(Path(manifest_run) / 'research_test_manifest.json')
                or receipt['parameter_sha256_after'] != receipt['checkpoint']['parameter_sha256']):
            raise ValueError('research-test 运行未完整完成或身份改变')
        for name in ('sampler_grid.csv', 'test_canonical_parents.csv', 'independent_runtime.json'):
            verify_artifact(path, receipt, name)
        if sha256(path / 'test_canonical_parents.csv') != manifest['splits']['test']['canonical_rows_sha256']:
            raise ValueError('test 样本集合变化')
        frames.append(pd.read_csv(path / 'sampler_grid.csv'))
        sources.append({'run': str(path.resolve()), 'receipt_sha256': sha256(path / 'receipt.json')})
    frame = pd.concat(frames, ignore_index=True)
    validate_curves(frame, manifest, all_seeds=True)
    cols = [c for c in frame if c.startswith(('raw_', 'postfiltered_')) and not c.endswith('_n_defined')]
    if not np.isfinite(frame[cols].to_numpy(dtype=float)).all():
        raise ValueError('汇总指标存在非有限值')
    summary = frame.groupby(['sampler', 'nfe_per_trajectory', 'N', 'role'])[cols].agg(['mean', 'std', 'min', 'max'])
    summary.columns = ['__'.join(c) for c in summary.columns]
    with study_run(output, 'summarize-research-test', synthetic=False) as (out, receipt):
        receipt['research_test_protocol'] = PROTOCOL
        frame.to_csv(out / 'research_test_nine_configurations.csv', index=False)
        summary.to_csv(out / 'training_seed_summary.csv')
        write_json(out / 'sources.json', sources)
        write_json(out / 'research_test_manifest.json', manifest)
        val = pd.read_csv(ROOT / INPUTS / 'p5_summary/full_validation_nine_configurations.csv')
        pd.concat([val.assign(evaluation_split='val'), frame.assign(evaluation_split='test')]).to_csv(out / 'validation_test_comparison.csv', index=False)
