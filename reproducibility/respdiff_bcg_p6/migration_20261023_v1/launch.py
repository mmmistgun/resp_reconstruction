"""双卡迁移控制器，两个分片成功后汇总完整三 seed。"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[1]
PY = ROOT.parents[1] / '.venv/bin/python'


def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(name, data):
    with (HERE / name).open('x') as f: json.dump(data, f, ensure_ascii=False, indent=2)
def env_for(gpu):
    env = os.environ.copy(); env.pop('LD_LIBRARY_PATH', None)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
    return env


def main():
    for path in [STUDY/'test_s20260812_v3', STUDY/'summary_v3', HERE/'gpu2_ddpm_head', HERE/'gpu0_ddpm_tail']:
        if path.exists(): raise FileExistsError(path)
    completed = [STUDY/f'test_s{seed}_v2' for seed in (20260811,20260813)]
    for source in completed:
        receipt = json.loads((source/'receipt.json').read_text())
        if receipt['status'] != 'complete': raise ValueError('其他 seed 尚未完成')
        for name, sha in receipt['artifacts'].items():
            if digest(source/name) != sha: raise ValueError(f'已完成产物发生改变：{source/name}')
    state = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'], text=True)
    for uuid in ('GPU-5f031bf4-55cf-bb6e-b088-e7bf2b92c19d','GPU-a03e0829-f209-e05f-d8cf-b23db5ff47c6'):
        if uuid in state: raise RuntimeError('目标 GPU 被占用，停止启动')
    save('controller.json', {'pid': os.getpid(), 'started_utc': now(), 'program_sha256': digest(HERE/'migrate.py'),
        'recovery_sha256': digest(STUDY/'recovery_20261021_v1/recover.py'),
        'completed_seed_receipts': {str(p):digest(p/'receipt.json') for p in completed},
        'interrupted_seed_receipt': digest(STUDY/'test_s20260812_v2/receipt.json')})
    processes, logs = {}, {}
    try:
        for gpu in (2,0):
            command = [str(PY),str(HERE/'migrate.py'),f'gpu{gpu}','--confirm-research-test']
            logs[gpu] = (HERE/f'gpu{gpu}_console.log').open('x')
            processes[gpu] = subprocess.Popen(command,cwd=ROOT,env=env_for(gpu),stdout=logs[gpu],stderr=subprocess.STDOUT)
            save(f'gpu{gpu}_process.json', {'pid':processes[gpu].pid,'physical_gpu':gpu,'command':command,'started_utc':now()})
            print(f'GPU{gpu} PID {processes[gpu].pid}',flush=True)
        exited = {}
        while len(exited) < 2:
            for gpu, process in processes.items():
                if gpu not in exited and process.poll() is not None:
                    exited[gpu] = process.returncode
                    save(f'gpu{gpu}_exit.json',{'exit_code':process.returncode,'finished_utc':now()})
            if len(exited) < 2: time.sleep(10)
        if any(exited.values()): raise RuntimeError(f'双卡任务存在失败：{exited}')
        command = [str(PY),str(ROOT/'scripts/run_respdiff_bcg_research_test_v1.py'),'summarize',
            '--manifest-run',str(STUDY/'manifest_v1'),'--runs',str(completed[0]),str(STUDY/'test_s20260812_v3'),str(completed[1]),
            '--output',str(STUDY/'summary_v3')]
        save('summary_command.json', {'command':command})
        with (HERE/'summary_console.log').open('x') as log:
            subprocess.run(command,cwd=ROOT,env=env_for(''),stdout=log,stderr=subprocess.STDOUT,check=True)
        save('complete.json',{'status':'complete','finished_utc':now(),'summary':str(STUDY/'summary_v3')})
    finally:
        for log in logs.values(): log.close()


if __name__ == '__main__':
    try: main()
    except BaseException as error:
        save('failure.json',{'status':'failed','error_type':type(error).__name__,'error':str(error),'finished_utc':now()})
        raise
