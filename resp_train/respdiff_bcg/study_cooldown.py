"""P2-A2：从固定 P2 状态仅改变后续学习率，并逐步核对随机输入身份。"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .baseband import make_baseband_model
from .runtime import LOGGER, seed_all, sha256, write_json
from .study_contract import PROBE_SEEDS, PROBE_STEPS, TrainingSpec
from .study_diagnostics import evaluate_anchors, run_forward_probes
from .study_io import (ROOT, data_identity, implementation_identity, save_provenance,
                       training_config, verify_artifact, verify_data_unchanged)
from .study_runtime import _initialize, _check_source, read_stage, training_datasets
from .study_io import study_run
from .study_state import preserved_evaluation, restore_rng, rng_state
from .study_training import ResumableBatches, load_resume, train_with_snapshots

PARENT_SHA = 'dca9d5e4226f7aaf6cf99559d08cfbb7bb74b9d38acc1de4403cb8320149e4cc'
CONTROL_SHA = 'b4f4f3a8af56efd6942e120cec75a488ea3f06688c0ae568d89e3aea1021c47d'


def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal_tree(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(equal_tree(a, b) for a, b in zip(left, right))
    return left == right


def draw_identity(step, noise):
    digest = hashlib.sha256()
    for value in (step, noise):
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class CooldownFork:
    """唯一降档合同；fixture 只缩小更新数，实际运行固定 1600→3200。"""
    learning_rate = 1e-5

    def __init__(self, control_run, output, *, synthetic=False):
        self.control_run, self.output = Path(control_run).resolve(), Path(output)
        self.synthetic = synthetic
        self.start = 2 if synthetic else 1600
        self.end = 4 if synthetic else 3200
        self.parent = self.control_run / f'checkpoints/update_{self.start:06d}.pt'
        self.final = self.control_run / 'final.pt'
        self.parent_sha, self.control_sha = sha256(self.parent), sha256(self.final)
        if not synthetic and (self.parent_sha != PARENT_SHA or self.control_sha != CONTROL_SHA):
            raise ValueError('P2-A2 固定来源 checkpoint hash 不匹配')
        self.expected = {}
        self.seen = []

    def identity(self):
        return {'revision': 'P2-A2-lr-cooldown-v1', 'parent_checkpoint_sha256': self.parent_sha,
                'control_final_sha256': self.control_sha, 'control_run': str(self.control_run),
                'start_update': self.start, 'end_update': self.end,
                'learning_rate_before': 1e-4, 'learning_rate_after': self.learning_rate,
                'selector': 'final_update', 'synthetic': self.synthetic}

    def prepare(self, state, identity, data_digest, device):
        if (identity['regime'] != 'overfit' or identity['milestones'] or
                identity['synthetic'] != self.synthetic or state['update'] != self.start or
                identity['actual_updates'] != self.end or state['selector'] != 'diagnostic_update'):
            raise ValueError('降档起点或 overfit 合同不匹配')
        control = load_resume(self.final, training=identity, data_digest=data_digest)
        if control['update'] != self.end or control['selector'] != 'final_update':
            raise ValueError('控制路径终点不完整')
        for value in (state, control):
            if any(g['lr'] != 1e-4 for g in value['optimizer']['param_groups']):
                raise ValueError('控制路径不是固定 1e-4')
        history = [json.loads(line) for line in (self.control_run/'history.jsonl').read_text().splitlines()]
        if [row['update'] for row in history] != list(range(1, self.end+1)):
            raise ValueError('控制路径 history 不完整')
        future = history[self.start:]
        stream_state = state['batch_stream']
        stream = ResumableBatches(stream_state['size'], stream_state['batch_size'], stream_state['seed'])
        stream.load_state_dict(stream_state)
        saved = rng_state(device)
        try:
            restore_rng(state['rng'], device)
            # 原 training_loss 每步先 randint，再 randn；网络无随机层。
            # 用控制终点全部 RNG 与 batch 游标验证这一消费顺序，随后逐步验证实际降档训练。
            for row in future:
                indices = stream.next()
                if indices != row['chunk_indices'] or row['lr'] != 1e-4:
                    raise ValueError('控制路径后续 batch 顺序或学习率不匹配')
                step = torch.randint(50, (len(indices),), device=device)
                noise = torch.randn((len(indices), 1, 600), device=device)
                self.expected[row['update']] = {'chunk_indices': indices, 'draw_sha256': draw_identity(step, noise)}
            if not equal_tree(rng_state(device), control['rng']):
                raise ValueError('重放 t/noise 后 RNG 与原控制终点不一致')
            if not equal_tree(stream.state_dict(), control['batch_stream']):
                raise ValueError('重放 batch 后游标与原控制终点不一致')
        finally:
            restore_rng(saved, device)
        self.final_rng, self.final_stream = control['rng'], control['batch_stream']
        write_json(self.output/'cooldown_identity.json', self.identity())
        write_json(self.output/'sequence_preflight.json', {'status': 'passed',
            'future_updates': len(future), 'control_terminal_rng_equal': True,
            'control_terminal_batch_stream_equal': True, 'draws': self.expected})
        LOGGER.info('P2-A2 序列预检通过：%d 个后续 batch，t/noise 重放终点 RNG 完全一致', len(future))

    def loss(self, model, condition, target, update, indices):
        step = torch.randint(50, (len(target),), device=target.device)
        noise = torch.randn(target.shape, dtype=target.dtype, device=target.device)
        actual = {'chunk_indices': indices, 'draw_sha256': draw_identity(step, noise)}
        if actual != self.expected[update]:
            raise ValueError(f'update={update} 实际 batch/t/noise 偏离控制路径')
        self.seen.append(update)
        return model.training_loss(condition, target, step=step, noise=noise)

    def finish(self, rng, stream):
        if (self.seen != list(range(self.start+1, self.end+1)) or
                not equal_tree(rng, self.final_rng) or not equal_tree(stream, self.final_stream)):
            raise ValueError('降档路径随机序列或终点游标不一致')
        write_json(self.output/'sequence_verification.json', {'status': 'passed',
            'verified_updates': len(self.seen), 'terminal_rng_equal': True, 'terminal_batch_stream_equal': True})


def run_cooldown(output, baseline_run, p1_run, p2_run, audit_run, *, device='cuda:0'):
    p2, control_receipt = read_stage(p2_run, 'overfit')
    audit = Path(audit_run).resolve()
    audit_receipt = json.loads((audit/'receipt.json').read_text())
    if (audit_receipt.get('status') != 'complete' or audit_receipt.get('phase') != 'training-sampling-audit'
            or not audit_receipt.get('formal_evidence') or audit_receipt.get('gradient_cases') != 108
            or audit_receipt.get('failed_consistency_checks') != 0):
        raise ValueError('要求已完成 P2-A1 审计')
    for name in ('checkpoints/update_001600.pt', 'checkpoints/update_001600.identity.json',
                 'final.pt', 'final.identity.json', 'history.jsonl', 'actual_training_parents.csv',
                 'source_identity.json', 'train_representative_parents.csv', 'val_representative_parents.csv'):
        verify_artifact(p2, control_receipt, name)
    with study_run(output, 'overfit_cooldown_a2', synthetic=False, source=p2) as (out, receipt):
        fork = CooldownFork(p2, out)
        old_model, base, checkpoint, bundles, full, index = _initialize(
            out, 'overfit_cooldown_a2', baseline_run, device, tiny=False, training=True)
        del old_model
        p1, p1_receipt = read_stage(p1_run, 'diagnose')
        verify_artifact(p1, p1_receipt, 'diagnostic_identity.json')
        diagnostic_digest = sha256(p1/'diagnostic_identity.json')
        if diagnostic_digest != control_receipt['diagnostic_identity_sha256']:
            raise ValueError('P2 与 P1 数据选择身份不一致')
        train, probes = training_datasets(bundles, p1, overfit=True)
        if train.rows.to_csv(index=False).encode() != (p2/'actual_training_parents.csv').read_bytes():
            raise ValueError('降档训练 parent 身份改变')
        for key, dataset in probes.items():
            if dataset.rows.to_csv(index=False).encode() != (p2/f'{key}_parents.csv').read_bytes():
                raise ValueError('降档评价 batch parent 顺序改变')
        old_files = json.loads((p2/'source_identity.json').read_text())['files']
        # 数值模型、预处理、采样、指标保持原版本；只允许新增独立编排与训练分支钩子。
        for relative, digest in old_files.items():
            if relative.startswith('resp_train/') and relative not in {
                    'resp_train/respdiff_bcg/study_training.py',
                    'resp_train/respdiff_bcg/study_io.py'} and sha256(ROOT/relative) != digest:
                raise ValueError(f'控制路径数值源码改变：{relative}')
        spec = TrainingSpec('overfit')
        cfg = training_config(base, spec)
        cfg.training.device = device
        cfg.study.cooldown = fork.identity()
        cfg.training.learning_rate = fork.learning_rate
        cfg.diagnostics.probe_updates = [3200]
        seed_all(spec.seed)
        model = make_baseband_model(cfg).to(torch.device(device))
        identity = save_provenance(out, cfg, cpu_only=False, execution={'phase': 'overfit_cooldown_a2',
            'source': checkpoint, 'cooldown': fork.identity(), 'diagnostic_run': str(p1),
            'audit_run': str(audit), 'audit_receipt_sha256': sha256(audit/'receipt.json')})
        data = data_identity(out, full, index)
        train.rows.to_csv(out/'actual_training_parents.csv', index=False)
        train.manifest().to_csv(out/'actual_training_chunks.csv', index=False)
        for key, dataset in probes.items():
            dataset.rows.to_csv(out/f'{key}_parents.csv', index=False)

        def monitor(update):
            with preserved_evaluation(model):
                for key, dataset in probes.items():
                    path = out/'monitor'/f'update_{update:06d}'/key
                    run_forward_probes(model, dataset, path/'forward', batch_size=64,
                        steps=PROBE_STEPS, noise_seeds=PROBE_SEEDS, pairing_seeds=())
                    evaluate_anchors(model, dataset, cfg, path/'anchors', batch_size=64)

        outcome = train_with_snapshots(model, train, cfg, spec, out, callback=monitor, cooldown=fork)
        verify_data_unchanged(data, pd.concat([d.rows for d in full.values()], ignore_index=True))
        _check_source(checkpoint, baseline_run)
        if (implementation_identity(cfg) != identity or sha256(fork.parent) != fork.parent_sha or
                sha256(fork.final) != fork.control_sha):
            raise ValueError('运行期间来源或实现身份改变')
        receipt.update({'outcome': outcome, 'cooldown': fork.identity(),
            'diagnostic_identity_sha256': diagnostic_digest, 'scientific_interpretation': 'pending_review'})
