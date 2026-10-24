"""T630 三卡 P6 队列：内容核验 → 三卡验收 → 三 seed 并行 → 完整汇总。"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
STUDY = ROOT / 'runs/respdiff_bcg_research_test_v1'
PY = ROOT.parents[1] / '.venv/bin/python'
ENTRY = ROOT / 'scripts/run_respdiff_bcg_research_test_v1.py'
MANIFEST = STUDY / 'manifest_v1'
PREFLIGHT = STUDY / 'remote_preflight_v1'
SUMMARY = STUDY / 'summary_v1'
SEEDS = (20260811, 20260812, 20260813)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(name, value):
    with (OUT / name).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def base_env(gpu=''):
    env = os.environ.copy(); env.pop('LD_LIBRARY_PATH', None)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    return env


def check_idle():
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    if output.strip():
        raise RuntimeError('GPU 上已有计算进程，停止提交：' + output.strip())


def run_cpu(phase, output, extra):
    command = [str(PY), str(ENTRY), phase, '--output', str(output), *extra]
    save(f'{phase}_command.json', {'command': command, 'started_utc': now()})
    with (OUT / f'{phase}_console.log').open('x') as log:
        subprocess.run(command, cwd=ROOT, env=base_env(), stdout=log, stderr=subprocess.STDOUT, check=True)


def run_parallel(phase):
    processes, logs, outputs = {}, {}, {}
    try:
        for gpu, seed in enumerate(SEEDS):
            output = STUDY / (f'gpu_check_{gpu}_v1' if phase == 'gpu-check' else f'test_s{seed}_v1')
            args = (['--confirm-gpu-check'] if phase == 'gpu-check' else
                    ['--confirm-research-test', '--training-seed', str(seed), '--manifest-run', str(MANIFEST),
                     '--preflight-run', str(PREFLIGHT)])
            command = [str(PY), str(ENTRY), phase, '--output', str(output), '--device', 'cuda:0', *args]
            logs[gpu] = (OUT / f'{phase}_gpu{gpu}_console.log').open('x')
            processes[gpu] = subprocess.Popen(command, cwd=ROOT, env=base_env(gpu),
                                              stdout=logs[gpu], stderr=subprocess.STDOUT)
            outputs[gpu] = output
            save(f'{phase}_gpu{gpu}_process.json', {'pid': processes[gpu].pid, 'physical_gpu': gpu,
                 'training_seed': seed if phase == 'run' else None, 'command': command, 'started_utc': now()})
            print(f'{phase} GPU{gpu} 已启动，PID {processes[gpu].pid}', flush=True)
        exited = {}
        while len(exited) != 3:
            for gpu, process in processes.items():
                if gpu not in exited and process.poll() is not None:
                    exited[gpu] = process.returncode
                    save(f'{phase}_gpu{gpu}_exit.json', {'exit_code': process.returncode, 'finished_utc': now()})
                    print(f'{phase} GPU{gpu} 退出，code={process.returncode}', flush=True)
            if len(exited) != 3:
                time.sleep(10)
        if any(exited.values()):
            raise RuntimeError(f'{phase} 有失败任务：{exited}')
        receipts = []
        for gpu, output in outputs.items():
            receipt = json.loads((output / 'receipt.json').read_text())
            expected = 'research-test-gpu-check' if phase == 'gpu-check' else 'research-test'
            if receipt.get('status') != 'complete' or receipt.get('phase') != expected:
                raise ValueError(f'{phase} GPU{gpu} 回执不完整')
            if phase == 'run' and receipt.get('checkpoint', {}).get('training_seed') != SEEDS[gpu]:
                raise ValueError('GPU/训练 seed 身份不匹配')
            if phase == 'gpu-check' and (len(receipt.get('comparisons', [])) != 2 or
                    not all(item['passed'] for item in receipt['comparisons'])):
                raise ValueError('三卡数值验收不完整')
            receipts.append(receipt)
        return receipts
    finally:
        for log in logs.values(): log.close()


def main():
    paths = [PREFLIGHT, SUMMARY, *[STUDY / f'gpu_check_{gpu}_v1' for gpu in range(3)],
             *[STUDY / f'test_s{seed}_v1' for seed in SEEDS]]
    for path in paths:
        if path.exists(): raise FileExistsError(path)
    save('controller.json', {'pid': os.getpid(), 'started_utc': now(), 'root': str(ROOT),
        'manifest_sha256': hashlib.sha256((MANIFEST / 'research_test_manifest.json').read_bytes()).hexdigest()})
    check_idle()
    print('开始远程输入与数据内容核验', flush=True)
    run_cpu('preflight', PREFLIGHT, ['--manifest-run', str(MANIFEST), '--confirm-research-test'])
    check_idle()
    print('远程内容核验通过，开始三卡 synthetic 数值验收', flush=True)
    acceptance = run_parallel('gpu-check')
    manifest = json.loads((MANIFEST / 'research_test_manifest.json').read_text())
    batches = (manifest['n_chunks'] + 63) // 64
    budget = []
    for gpu, receipt in enumerate(acceptance):
        per_n = {r['sampler']: r['sampling_seconds'] for r in receipt['comparisons']}
        budget.append({'physical_gpu': gpu, 'training_seed': SEEDS[gpu], 'full_batches': batches,
                       'estimated_sampling_hours': batches * (per_n['ddim'] * 64 + per_n['ddpm'] * 16) / 3600})
    save('budget_estimate.json', {'definition': 'synthetic B64/N1 extrapolation; overhead and load may differ', 'models': budget})
    check_idle()
    print('三卡验收全部通过，提交三个冻结 seed 正式 research-test', flush=True)
    run_parallel('run')
    run_cpu('summarize', SUMMARY, ['--manifest-run', str(MANIFEST), '--runs',
            *[str(STUDY / f'test_s{seed}_v1') for seed in SEEDS]])
    save('complete.json', {'status': 'complete', 'finished_utc': now(), 'summary': str(SUMMARY)})
    print('三个模型与九项 research-test 汇总全部完成', flush=True)


if __name__ == '__main__':
    try: main()
    except BaseException as error:
        save('failure.json', {'status': 'failed', 'finished_utc': now(),
                             'error_type': type(error).__name__, 'error': str(error)})
        raise
