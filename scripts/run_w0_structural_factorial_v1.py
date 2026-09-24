from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resp_train.paper_evidence.w0_structural_factorial_v1 import (
    ARMS,
    derived_config,
    load_experiment_spec,
    load_lock,
    load_w0_baseline,
    parameter_compute_report,
    prepare_lock,
    validate_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="W0 三因素结构对照 v1")
    commands = parser.add_subparsers(dest="phase", required=True)
    commands.add_parser("check-config", help="核验八组结构与训练合同")
    commands.add_parser("prepare-lock", help="在干净提交上复核来源并生成实现锁")
    commands.add_parser("check-lock", help="复核已生成的实现锁与当前源码")
    describe = commands.add_parser("describe", help="输出参数与 covered-MAC 合同")
    describe.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()

    if args.phase == "check-config":
        load_experiment_spec()
        for seed in (20260811, 20260812, 20260813):
            baseline = load_w0_baseline(seed)
            for arm in ARMS:
                output = ROOT / "runs/w0_structural_factorial_v1_es30p15/formal" / arm / f"seed_{seed}"
                cfg = derived_config(baseline, arm=arm, output_root=output, device="cuda:0")
                validate_config(cfg, baseline, arm=arm, output_root=output, device="cuda:0")
        result: str | dict = "W0 structural factorial config OK"
    elif args.phase == "prepare-lock":
        result = str(prepare_lock())
    elif args.phase == "check-lock":
        _lock, lock_hash = load_lock()
        result = {"implementation_lock_sha256": lock_hash, "status": "passed"}
    else:
        report = parameter_compute_report()
        result = report if args.arm is None else report["arms"][args.arm]
    print(result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
