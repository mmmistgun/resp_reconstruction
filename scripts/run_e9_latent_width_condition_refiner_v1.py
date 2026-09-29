from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resp_train.paper_evidence.e9_latent_width_condition_refiner_v1 import (
    check_p1,
    formal_plan,
)
from resp_train.paper_evidence.e9_latent_width_condition_refiner_v1_engineering import (
    run_gpu_acceptance,
)
from resp_train.paper_evidence.e9_latent_width_condition_refiner_v1_formal import (
    load_formal_lock,
    matrix_status,
    prepare_formal_amendment,
    prepare_formal_lock,
    run_formal,
)
from resp_train.paper_evidence.e9_latent_width_condition_refiner_v1_model import (
    ARMS,
    ARM_SPECS,
    model_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="E9 潜在宽度 × 条件末端结构实验")
    commands = parser.add_subparsers(dest="phase", required=True)
    commands.add_parser("check-p1", help="核验六臂 P1 CPU/static 合同；不读取真实数据")
    commands.add_parser("formal-plan", help="输出固定 18-cell 计划及逐 cell 命令")
    commands.add_parser("prepare-formal-lock", help="在干净提交上生成唯一 implementation lock")
    commands.add_parser("check-formal-lock", help="回载核验 implementation lock、P2 与数据来源")
    commands.add_parser("prepare-formal-amendment", help="生成 post-training validation schema 修订锁")
    commands.add_parser("matrix-status", help="回载核验当前 formal 18-cell 状态")
    gpu = commands.add_parser("gpu-acceptance", help="运行 P2 synthetic GPU 工程验收")
    gpu.add_argument("--device", default="cuda:0")
    describe = commands.add_parser("describe", help="输出结构、参数与受管 MAC 合同")
    describe.add_argument("--arm", choices=ARMS)
    formal = commands.add_parser("formal", help="执行一个锁定的 arm×seed formal cell")
    formal.add_argument("--arm", required=True, choices=ARMS)
    formal.add_argument(
        "--seed", required=True, type=int, choices=(20260811, 20260812, 20260813)
    )
    formal.add_argument("--device", default="cuda:0")
    formal.add_argument("--confirm-formal-training", action="store_true")
    args = parser.parse_args()

    if args.phase == "check-p1":
        result = check_p1()
    elif args.phase == "formal-plan":
        plan = formal_plan()
        result = {"cells": plan, "count": len(plan)}
    elif args.phase == "prepare-formal-lock":
        result = str(prepare_formal_lock())
    elif args.phase == "check-formal-lock":
        lock, lock_hash, _source = load_formal_lock()
        result = {
            "protocol": lock["protocol"],
            "implementation_lock_sha256": lock_hash,
            "status": "passed",
        }
    elif args.phase == "prepare-formal-amendment":
        result = str(prepare_formal_amendment())
    elif args.phase == "matrix-status":
        result = matrix_status()
    elif args.phase == "gpu-acceptance":
        result = str(run_gpu_acceptance(args.device))
    elif args.phase == "formal":
        if not args.confirm_formal_training:
            raise SystemExit("E9 formal 要求显式传入 --confirm-formal-training")
        result = str(run_formal(args.arm, args.seed, device=args.device))
    else:
        arms = {
            arm: {
                **model_contract(arm),
                "structure": (
                    f"D={spec.latent_channels}, "
                    + (
                        "condition direct"
                        if spec.hidden_channels is None
                        else f"condition {spec.latent_channels}→{spec.hidden_channels}→{spec.latent_channels}"
                    )
                ),
            }
            for arm, spec in ARM_SPECS.items()
        }
        result = arms if args.arm is None else arms[args.arm]
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
