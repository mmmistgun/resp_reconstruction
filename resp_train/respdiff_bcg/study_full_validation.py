"""P5：固定三个模型与扩展 N 配置的全量 validation 编排。"""
import json
import os
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch

from .inference_v2 import evaluate_setting, predict_prefixes, state_digest
from .runtime import environment, save_source_snapshot, sha256, write_json
from .sampling_accelerated import AccelerationSpec
from .study_contract import METRICS, NAME, StudySamplerSpec
from .study_evaluation import independent_timing
from .study_extension import ExtensionSpec
from .study_extension_noise import frozen_selection
from .study_io import (ROOT, data_identity, implementation_identity, load_source,
                       selected_dataset, study_run, verify_artifact, verify_data_unchanged)
from .study_runtime import _check_source, _initialize, read_stage
from .study_state import preserved_evaluation

REVISION = 'P5-extended-full-validation-v1'
CHECKPOINTS = {
    20260811: 'c1e16cfdfba3498a60eafad11e1a72007b5e9969ba921d2eb7b679690d558731',
    20260812: '5a0c9f1b06879ea62aa6173571ad29a54de28b07ca919699db9bccaea12aab0e',
    20260813: 'b07c5efc3ed135b43578db38b197c0a0b4922ab55b595525f09414872f0c36cb',
}
CONFIGURATIONS = [
    {'sampler': 'ddim', 'S': 6, 'N': 64, 'role': 'selected'},
    {'sampler': 'ddpm', 'S': 50, 'N': 16, 'role': 'selected'},
    {'sampler': 'ddim', 'S': 6, 'N': 16, 'role': 'fixed_reference'},
]


def specs():
    return [ExtensionSpec('ddim', 6, 64), ExtensionSpec('ddpm', 50, 16), StudySamplerSpec('ddim', 6, 16)]


def full_identity(cfg):
    identity = implementation_identity(cfg)
    name = 'scripts/run_respdiff_bcg_full_validation_v1.py'
    identity['files'][name] = sha256(ROOT / name)
    return identity


def verify_files(files):
    for name, expected in files.items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f'冻结实现或协议变化：{name}')


def validate_selection(selection):
    rows = selection['selected']
    if (len(rows) != 3 or {tuple(sorted(r.items())) for r in rows}
            != {tuple(sorted(r.items())) for r in CONFIGURATIONS}
            or selection['selection_noise_seed'] != 20261003):
        raise ValueError('P5 必须使用已选定的两个配置和固定参照')


def validate_manifest(manifest):
    models = manifest['checkpoint_sources']
    if (manifest.get('revision') != REVISION or manifest.get('configurations') != CONFIGURATIONS
            or manifest.get('sampling_seed') != 20261003 or manifest.get('batch_size') != 64
            or manifest.get('trajectory_group_size') != 1 or len(models) != 3
            or {m['training_seed'] for m in models} != set(CHECKPOINTS)):
        raise ValueError('P5 清单配置或训练 seed 不完整')
    for item in models:
        if (item['checkpoint_sha256'] != CHECKPOINTS[item['training_seed']]
                or item['regime'] != 'long_stretched' or item['update'] != 16000):
            raise ValueError('P5 清单模型不匹配冻结的 final16000')


def freeze_full(output, *, source_runs, grid_run, selection_run):
    selected, _, selection = frozen_selection(selection_run)
    validate_selection(selection)
    grid, receipt = read_stage(grid_run, 'grid')
    for name in ('grid_identity.json', 'source_identity.json'):
        verify_artifact(grid, receipt, name)
    sources = json.loads((selected / 'sources.json').read_text())
    if sources['grid_receipt_sha256'] != sha256(grid / 'receipt.json'):
        raise ValueError('P4-N2 选择与原 grid 来源不一致')
    original = json.loads((grid / 'grid_identity.json').read_text())
    old_files = json.loads((grid / 'source_identity.json').read_text())['files']
    verify_files({k: v for k, v in old_files.items() if k.startswith(('resp_train/', 'scripts/', 'configs/'))})
    with study_run(output, 'freeze-full-validation', synthetic=False) as (out, result):
        checkpoints, parent_hashes, chunk_hashes = [], set(), set()
        for source in source_runs:
            model, cfg, checkpoint = load_source(source, torch.device('cpu'))
            if vars(model.spec) != original['model_spec']:
                raise ValueError('最终模型结构与 P4 不一致')
            source = Path(source)
            train_receipt = json.loads((source / 'receipt.json').read_text())
            verify_artifact(source, train_receipt, 'val_chunks.csv')
            parent_hashes.add(sha256(source / 'val_parents.csv'))
            chunk_hashes.add(sha256(source / 'val_chunks.csv'))
            checkpoints.append(checkpoint)
            del model
        if len(parent_hashes) != 1 or len(chunk_hashes) != 1 or original['source_val_parents_sha256'] not in parent_hashes:
            raise ValueError('三个模型的全量 validation 身份不一致')
        identity = full_identity(cfg)
        manifest = {'protocol': NAME, 'revision': REVISION, 'checkpoint_sources': checkpoints,
            'configurations': CONFIGURATIONS, 'sampling_seed': 20261003, 'batch_size': 64,
            'trajectory_group_size': 1, 'selection_run': str(selected),
            'selection_receipt_sha256': sha256(selected / 'receipt.json'),
            'selection_sha256': sha256(selected / 'selection.json'),
            'source_val_parents_sha256': next(iter(parent_hashes)),
            'source_val_chunks_sha256': next(iter(chunk_hashes)),
            'n_parents': len(pd.read_csv(source / 'val_parents.csv')),
            'n_chunks': len(pd.read_csv(source / 'val_chunks.csv')),
            'files': identity['files'], 'model_spec': original['model_spec']}
        validate_manifest(manifest)
        write_json(out / 'final_evaluation_manifest.json', manifest)
        write_json(out / 'selection.json', selection)
        write_json(out / 'source_identity.json', identity)
        save_source_snapshot(out, identity)
        write_json(out / 'environment.json', environment(cpu_only=True))
        result['revision'] = REVISION


def read_manifest(path):
    root, receipt = read_stage(path, 'freeze-full-validation')
    for name in ('final_evaluation_manifest.json', 'selection.json'):
        verify_artifact(root, receipt, name)
    manifest = json.loads((root / 'final_evaluation_manifest.json').read_text())
    validate_manifest(manifest)
    validate_selection(json.loads((root / 'selection.json').read_text()))
    selected = Path(manifest['selection_run'])
    if (sha256(selected / 'receipt.json') != manifest['selection_receipt_sha256']
            or sha256(selected / 'selection.json') != manifest['selection_sha256']):
        raise ValueError('P4 冻结选择发生变化')
    verify_files(manifest['files'])
    return root, manifest


def evaluate_fixed(model, dataset, cfg, out, checkpoint):
    """N16 来自同一次 DDIM N64 的在线均值前缀，独立计时仍逐配置执行。"""
    curves, profiles = [], []
    with preserved_evaluation(model):
        for spec, counts in [(ExtensionSpec('ddim', 6, 64), (16, 64)),
                             (ExtensionSpec('ddpm', 50, 16), (16,))]:
            sampling = out / f'{spec.sampler}_sampling'
            sampling.mkdir()
            predictions, profile = predict_prefixes(model, dataset, spec, counts, sampling,
                                                     batch_size=64, acceleration=AccelerationSpec(1))
            profiles.extend(profile)
            for n in counts:
                # 小 N 使用自己的 spec，避免评价 metadata 被标成最大 N。
                actual = StudySamplerSpec('ddim', 6, 16) if spec.sampler == 'ddim' and n == 16 else spec
                setting = out / f'{spec.sampler}_S{spec.nfe}_N{n}'
                row = evaluate_setting(predictions.pop(n), dataset, cfg, setting, actual, n,
                                       method_prefix=f'RespDiff-study-v1/{REVISION}')
                role = next(r['role'] for r in CONFIGURATIONS if (r['sampler'], r['N']) == (spec.sampler, n))
                row.update(training_seed=checkpoint['training_seed'], sampling_seed=20261003,
                           checkpoint_sha256=checkpoint['checkpoint_sha256'], role=role)
                curves.append(row)
                pd.DataFrame(curves).to_csv(out / 'sampler_grid.csv', index=False)
                pd.DataFrame(profiles).to_csv(out / 'runtime_profile.csv', index=False)
    return pd.DataFrame(curves)


def validate_curves(frame, manifest, *, all_seeds):
    expected = {(seed, r['sampler'], r['S'], r['N']) for seed in CHECKPOINTS for r in CONFIGURATIONS}
    keys = list(zip(frame.training_seed, frame.sampler, frame.nfe_per_trajectory, frame.N))
    if not all_seeds:
        if frame.training_seed.nunique() != 1:
            raise ValueError('单模型评价包含多个训练 seed')
        expected = {k for k in expected if k[0] == int(frame.training_seed.iloc[0])}
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('全量评价缺少配置或包含重复/额外配置')
    if not np.isfinite(frame[list(METRICS)].to_numpy(dtype=float)).all():
        raise ValueError('全量核心指标非有限')
    if (frame[METRICS[0]].abs() > 1 + 1e-6).any() or (frame[list(METRICS[1:])] < 0).any().any():
        raise ValueError('全量核心指标范围非法')
    for row in frame.itertuples():
        wanted = next(r for r in CONFIGURATIONS if (r['sampler'], r['N']) == (row.sampler, row.N))
        if (row.n_parents != manifest['n_parents'] or row.sampling_seed != 20261003
                or row.checkpoint_sha256 != CHECKPOINTS[row.training_seed] or row.role != wanted['role']
                or row.denoiser_calls_per_chunk != row.nfe_per_trajectory * row.N):
            raise ValueError('全量行的 parent/模型/采样身份不匹配')


def run_full(output, *, source_run, manifest_run, device='cuda:0'):
    frozen, manifest = read_manifest(manifest_run)
    matches = [m for m in manifest['checkpoint_sources'] if Path(m['source_run']).resolve() == Path(source_run).resolve()]
    if len(matches) != 1:
        raise ValueError('模型来源不在冻结清单中')
    with study_run(output, 'full-validation-extended', synthetic=False, source=source_run) as (out, receipt):
        model, cfg, checkpoint, bundles, _, index = _initialize(out, 'full-validation-extended', source_run,
            device, tiny=False, expected_sha256=matches[0]['checkpoint_sha256'])
        if checkpoint != matches[0]:
            raise ValueError('实际模型身份与冻结清单不匹配')
        parents, rows, _ = bundles['val']
        dataset = selected_dataset(parents, rows, list(range(len(rows))))
        if len(rows) != manifest['n_parents'] or len(dataset) != manifest['n_chunks']:
            raise ValueError('全量 validation 数量改变')
        identity = full_identity(cfg)
        OmegaConf.save(cfg, out / 'resolved_config.yaml')
        write_json(out / 'source_identity.json', identity)
        save_source_snapshot(out, identity)
        write_json(out / 'environment.json', environment(cpu_only=False))
        write_json(out / 'final_evaluation_manifest.json', manifest)
        write_json(out / 'execution.json', {'revision': REVISION, 'checkpoint': checkpoint,
            'manifest_run': str(frozen), 'manifest_sha256': sha256(frozen / 'final_evaluation_manifest.json'),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'device': device})
        data = data_identity(out, {'val': dataset}, index)
        if (sha256(out / 'val_parents.csv') != manifest['source_val_parents_sha256']
                or sha256(out / 'val_chunks.csv') != manifest['source_val_chunks_sha256']):
            raise ValueError('全量 validation 的 parent/chunk 顺序改变')
        frame = evaluate_fixed(model, dataset, cfg, out, checkpoint)
        validate_curves(frame, manifest, all_seeds=False)
        timings = independent_timing(model, dataset, specs(), batch_size=64, warmup=1, repeats=2)
        write_json(out / 'independent_runtime.json', timings)
        verify_data_unchanged(data, dataset.rows)
        _check_source(checkpoint, source_run, model)
        read_manifest(manifest_run)
        if full_identity(cfg) != identity:
            raise ValueError('评价期间源码或配置变化')
        receipt.update(revision=REVISION, checkpoint=checkpoint, n_parents=len(rows),
            n_chunks=len(dataset), manifest_sha256=sha256(frozen / 'final_evaluation_manifest.json'),
            parameter_sha256_after=state_digest(model))


def summarize_full(output, *, manifest_run, runs):
    frozen, manifest = read_manifest(manifest_run)
    frames, sources = [], []
    for run in runs:
        path, receipt = read_stage(run, 'full-validation-extended')
        for name in ('sampler_grid.csv', 'val_parents.csv', 'val_chunks.csv', 'independent_runtime.json'):
            verify_artifact(path, receipt, name)
        if (receipt.get('revision') != REVISION or receipt['manifest_sha256'] != sha256(frozen / 'final_evaluation_manifest.json')
                or receipt['parameter_sha256_after'] != receipt['checkpoint']['parameter_sha256']
                or sha256(path / 'val_parents.csv') != manifest['source_val_parents_sha256']
                or sha256(path / 'val_chunks.csv') != manifest['source_val_chunks_sha256']):
            raise ValueError('全量结果来源身份不一致')
        frames.append(pd.read_csv(path / 'sampler_grid.csv'))
        sources.append({'run': str(path), 'receipt_sha256': sha256(path / 'receipt.json')})
    frame = pd.concat(frames, ignore_index=True)
    validate_curves(frame, manifest, all_seeds=True)
    columns = [c for c in frame if c.startswith(('raw_', 'postfiltered_')) and not c.endswith('_n_defined')]
    if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
        raise ValueError('完整汇总指标存在非有限值')
    summary = frame.groupby(['sampler', 'nfe_per_trajectory', 'N', 'role'])[columns].agg(['mean', 'std', 'min', 'max'])
    summary.columns = ['__'.join(c) for c in summary.columns]
    with study_run(output, 'summarize-full-validation', synthetic=False) as (out, receipt):
        frame.to_csv(out / 'full_validation_nine_configurations.csv', index=False)
        summary.to_csv(out / 'training_seed_summary.csv')
        write_json(out / 'sources.json', sources)
        write_json(out / 'final_evaluation_manifest.json', manifest)
        receipt.update(revision=REVISION, training_seeds=sorted(CHECKPOINTS), n_configurations=9)
