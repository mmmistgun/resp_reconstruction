"""固定60-checkpoint research-test与A0机制分析的双卡后台调度。"""
from __future__ import annotations
import argparse
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROTOCOL_NOTE = ROOT / "docs/experiments/cwt_apor_v2_research_test_execution_20261002.md"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    parser.add_argument("--confirm-research-test", action="store_true")
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.confirm_research_test:
        parser.error("需要当次 --confirm-research-test")
    if args.devices != ["cuda:0", "cuda:1"]:
        parser.error("本轮已验收设备固定为cuda:0和cuda:1")
    if args.worker is not None and args.worker not in (0, 1):
        parser.error("worker分片只能为0或1")
    from resp_train.paper_evidence.cwt_apor_v2 import artifacts as io
    from resp_train.paper_evidence.cwt_apor_v2 import research_test as test
    from resp_train.paper_evidence.cwt_apor_v2.spec import ARMS, SEEDS
    session = args.session.resolve()

    def stop(signum, _):
        raise KeyboardInterrupt(f"收到停止信号{signum}")

    signal.signal(signal.SIGTERM, stop)

    def step(label, action):
        print(f"PHASE_START={label}", flush=True)
        result = action()
        print(f"PHASE_COMPLETE={label} OUTPUT={result}", flush=True)
        return result

    if args.worker is not None:
        device = args.devices[args.worker]
        # 每种表示的缓存仅构建一次，随后顺序评价全部三seed；两分片不共享GPU。
        for arm in list(ARMS)[args.worker::len(args.devices)]:
            step(f"cache_{arm}", lambda: test.build_test_cache(session, arm, confirmed=True))
            for seed in SEEDS:
                step(f"evaluate_{arm}_{seed}", lambda: test.evaluate(session, arm, seed, device, confirmed=True))
        return

    io.require_data_scope(session)
    processes, streams = [], []
    key = io.binding(session, "research_test_pipeline")
    with io.attempt(session / "research_test" / "pipeline", key) as output:
        print(f"PIPELINE={output}", flush=True)
        for path in (Path(__file__), PROTOCOL_NOTE):
            (output / path.name).write_bytes(path.read_bytes())
        io.write_json(output / "authorization.json", {
            "user_instruction": "开启test", "date_local": "2026-10-02", "timezone": "Asia/Shanghai",
            "session": str(session), "checkpoint_cells": 60, "mechanism_seeds": list(SEEDS),
            "devices": args.devices, "training": False, "research_test": True,
            "command": sys.argv, "protocol_note": io.identity(PROTOCOL_NOTE),
        })
        allowlist = step("freeze_allowlist", lambda: test.prepare_allowlist(session))
        io.write_json(output / "allowlist_source.json", {"path": str(allowlist), **io.identity(allowlist / "manifest.json")})
        step("prepare_test_data", lambda: test.prepare_test_data(session, confirmed=True))
        try:
            for index, device in enumerate(args.devices):
                log = output / f"worker_{index}_{device.replace(':', '_')}.log"
                stream = log.open("x")
                streams.append(stream)
                command = [sys.executable, "-u", str(Path(__file__).resolve()), "--session", str(session),
                           "--devices", *args.devices, "--confirm-research-test", "--worker", str(index)]
                process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                processes.append(process)
                print(f"WORKER={device} PID={process.pid} LOG={log}", flush=True)
            io.write_json(output / "workers.json", [{"device": d, "pid": p.pid} for d, p in zip(args.devices, processes)])
            while any(p.poll() is None for p in processes):
                if any(p.poll() not in (None, 0) for p in processes):
                    raise RuntimeError("test worker失败；停止其他worker并保留现场")
                time.sleep(2)
            if any(p.returncode != 0 for p in processes):
                raise RuntimeError("test worker非零退出")
        finally:
            for p in processes:
                if p.poll() is None:
                    p.send_signal(signal.SIGINT)
            for p in processes:
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
            for stream in streams:
                stream.close()
        summary = step("test_summary", lambda: test.summarize_test(session))
        # 机制评价沿用已验收batch8，三个seed依次运行以控制案例导出内存与磁盘峰值。
        for index, seed in enumerate(SEEDS):
            step(f"mechanisms_{seed}", lambda: test.mechanisms(session, seed, args.devices[index % 2], confirmed=True))
        mechanisms = step("mechanisms_summary", lambda: test.summarize_mechanisms(session))
        io.write_json(output / "completion.json", {"completed": True, "checkpoint_cells": 60,
                      "test_summary": str(summary), "mechanisms_summary": str(mechanisms)})
    print("COMPLETED=research_test_and_mechanisms", flush=True)


if __name__ == "__main__":
    main()
