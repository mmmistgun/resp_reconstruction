"""后台串联合成校准和多卡独立工程验收；会话不允许真实数据或训练。"""
from __future__ import annotations
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    parser.add_argument("--calibration", type=Path, help="复用已完成且变换依赖未改变的校准，不重跑校准")
    parser.add_argument("--confirm-synthetic", action="store_true")
    args = parser.parse_args()
    if not args.confirm_synthetic:
        parser.error("需要 --confirm-synthetic")
    import torch
    from resp_train.paper_evidence.cwt_time_frequency_v1 import artifacts as io
    from resp_train.paper_evidence.cwt_time_frequency_v1.calibration import calibrate, prepare
    from resp_train.paper_evidence.cwt_time_frequency_v1.spec import OUTPUT_ROOT, load_spec
    devices = [str(torch.device(d)) for d in args.devices]
    if (not devices or len(set(devices)) != len(devices)
            or any(torch.device(d).type != "cuda" or torch.device(d).index is None for d in devices)):
        parser.error("设备必须为不同的显式 cuda:<index>")
    key = {"phase": "synthetic_engineering_background", "spec": io.json_hash(load_spec()), "devices": devices}
    parent = OUTPUT_ROOT / "engineering_background" / io.stamp()
    processes, streams = [], []

    def stop_signal(signum, _):
        raise KeyboardInterrupt(f"收到停止信号 {signum}")

    signal.signal(signal.SIGTERM, stop_signal)
    with io.attempt(parent, key) as output:
        print(f"BACKGROUND={output}", flush=True)
        io.write_json(output / "scope.json", {"synthetic_only": True, "real_data": False, "training": False,
                      "research_test": False, "devices": devices, "pid": os.getpid()})
        print("PHASE=calibration_reuse" if args.calibration else "PHASE=synthetic_calibration", flush=True)
        calibration = args.calibration.resolve() if args.calibration else calibrate()
        if args.calibration:
            from resp_train.paper_evidence.cwt_time_frequency_v1.engineering import prepare_reusing_calibration
            session = prepare_reusing_calibration(calibration)
        else:
            session = prepare(calibration, engineering_only=True)
        io.write_json(output / "calibration_source.json", {"path": str(calibration), **io.identity(calibration / "manifest.json")})
        print(f"CALIBRATION={calibration}", flush=True)
        io.write_json(output / "session_source.json", {"path": str(session), **io.identity(session / "session.json")})
        print(f"SESSION={session}", flush=True)
        try:
            for index, device in enumerate(devices):
                log = output / f"worker_{index}_{device.replace(':', '_')}.log"
                stream = log.open("x")
                streams.append(stream)
                env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", PYTHONPATH=str(ROOT))
                env.pop("LD_LIBRARY_PATH", None)
                env.pop("LD_PRELOAD", None)
                command = [sys.executable, str(ROOT / "scripts/run_cwt_time_frequency_v1.py"), "gpu-acceptance",
                           "--session", str(session), "--device", device, "--confirm-synthetic"]
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                processes.append(process)
                print(f"WORKER={device} PID={process.pid} LOG={log}", flush=True)
            io.write_json(output / "workers.json", [{"device": d, "pid": p.pid} for d, p in zip(devices, processes)])
            while any(p.poll() is None for p in processes):
                if any(p.poll() not in (None, 0) for p in processes):
                    raise RuntimeError("GPU验收worker失败；停止其他worker，保留日志和失败现场")
                time.sleep(2)
            if any(p.returncode != 0 for p in processes):
                raise RuntimeError("GPU验收worker非零退出")
            from resp_train.paper_evidence.cwt_time_frequency_v1.engineering import require_gpu
            receipts = {d: str(require_gpu(session, d)) for d in devices}
            io.write_json(output / "completion.json", {"passed": True, "session": str(session), "gpu_receipts": receipts,
                          "real_data": False, "training": False, "research_test": False})
            print("COMPLETED=synthetic_calibration_and_gpu_acceptance", flush=True)
        finally:
            for p in processes:
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGINT)
            for p in processes:
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
            for stream in streams:
                stream.close()


if __name__ == "__main__":
    main()
