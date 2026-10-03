"""已验收60-cell正式训练后台入口；含train/validation必要准备，不含test。"""
import argparse
import os
import signal
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engineering-session", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    parser.add_argument("--confirm-training", action="store_true")
    args = parser.parse_args()
    if not args.confirm_training:
        parser.error("正式训练需要--confirm-training")
    import torch
    from resp_train.paper_evidence.cwt_apor_v2.formal_launch import prepare_formal, execute
    devices = [str(torch.device(d)) for d in args.devices]
    if (not devices or len(set(devices)) != len(devices)
            or any(torch.device(d).type != "cuda" or torch.device(d).index is None for d in devices)):
        parser.error("GPU必须为不同的cuda:<index>")
    def stop(signum, _):
        raise KeyboardInterrupt(f"收到停止信号{signum}")
    signal.signal(signal.SIGTERM, stop)
    print(f"PID={os.getpid()} PHASE=prepare_formal_session", flush=True)
    session = prepare_formal(args.engineering_session, devices)
    print(f"SESSION={session}", flush=True)
    print(execute(session, devices), flush=True)


if __name__ == "__main__":
    main()
