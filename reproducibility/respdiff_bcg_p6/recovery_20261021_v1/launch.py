"""T630 三卡恢复控制器；已完成 DDIM 由核验清单复用。"""
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
AUDIT = STUDY / 'recovery_audit_20261021_v1'
MANIFEST = STUDY / 'manifest_v1'
SEEDS = (20260811, 20260812, 20260813)


def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(name, value):
    with (OUT / name).open('x') as stream: json.dump(value, stream, ensure_ascii=False, indent=2)


def env_for(gpu=''):
    env = os.environ.copy(); env.pop('LD_LIBRARY_PATH', None)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
    return env


def wait_idle():
    last = None
    while True:
        state = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True).strip()
        if not state: return
        if state != last: print('等待现有 GPU 任务释放显卡：' + state, flush=True); last = state
        time.sleep(30)


def main():
    for path in [STUDY / 'summary_v2', *[STUDY / f'test_s{seed}_v2' for seed in SEEDS]]:
        if path.exists(): raise FileExistsError(path)
    audit = json.loads((AUDIT / 'recovery_audit.json').read_text())
    digest = hashlib.sha256((OUT / 'recover.py').read_bytes()).hexdigest()
    if audit['program_sha256'] != digest: raise ValueError('恢复代码不同于已核验版本')
    save('controller.json', {'pid': os.getpid(), 'started_utc': now(), 'audit_run': str(AUDIT),
                            'recovery_program_sha256': digest})
    wait_idle()
    logs, processes = {}, {}
    try:
        for gpu, seed in enumerate(SEEDS):
            command = [str(PY), str(OUT / 'recover.py'), 'run', '--confirm-research-test',
                '--seed', str(seed), '--audit-run', str(AUDIT), '--device', 'cuda:0',
                '--output', str(STUDY / f'test_s{seed}_v2')]
            logs[gpu] = (OUT / f'gpu{gpu}_console.log').open('x')
            processes[gpu] = subprocess.Popen(command, cwd=ROOT, env=env_for(gpu), stdout=logs[gpu], stderr=subprocess.STDOUT)
            save(f'gpu{gpu}_process.json', {'pid': processes[gpu].pid, 'physical_gpu': gpu,
                'training_seed': seed, 'command': command, 'started_utc': now()})
            print(f'恢复 seed{seed} → GPU{gpu}，PID {processes[gpu].pid}', flush=True)
        exited = {}
        while len(exited) != 3:
            for gpu, process in processes.items():
                if gpu not in exited and process.poll() is not None:
                    exited[gpu] = process.returncode
                    save(f'gpu{gpu}_exit.json', {'exit_code': process.returncode, 'finished_utc': now()})
                    print(f'GPU{gpu} 退出，code={process.returncode}', flush=True)
            if len(exited) != 3: time.sleep(10)
        if any(exited.values()): raise RuntimeError(f'恢复任务存在失败，保留全部结果：{exited}')
        command = [str(PY), str(ROOT / 'scripts/run_respdiff_bcg_research_test_v1.py'), 'summarize',
            '--manifest-run', str(MANIFEST), '--runs', *[str(STUDY / f'test_s{seed}_v2') for seed in SEEDS],
            '--output', str(STUDY / 'summary_v2')]
        with (OUT / 'summary_console.log').open('x') as log:
            subprocess.run(command, cwd=ROOT, env=env_for(), stdout=log, stderr=subprocess.STDOUT, check=True)
        save('complete.json', {'status': 'complete', 'finished_utc': now(), 'summary': str(STUDY / 'summary_v2')})
        print('恢复后的三模型完整九项汇总已完成', flush=True)
    finally:
        for log in logs.values(): log.close()


if __name__ == '__main__':
    try: main()
    except BaseException as error:
        save('failure.json', {'status':'failed', 'finished_utc':now(), 'error_type':type(error).__name__, 'error':str(error)})
        raise
