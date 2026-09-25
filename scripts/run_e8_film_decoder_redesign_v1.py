from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resp_train.paper_evidence.e8_film_decoder_redesign_v1 import check_p1
from resp_train.paper_evidence.e8_film_decoder_redesign_v1_engineering import (
    benchmark_worker,
    run_benchmark,
    run_gpu_acceptance,
)
from resp_train.paper_evidence.e8_film_decoder_redesign_v1_model import (
    ARMS,
    ARM_SPECS,
    RECOMMENDED_ARM,
    REFERENCE_ARM,
    model_contract,
)
from resp_train.paper_evidence.e8_film_decoder_redesign_v1_formal import (
    formal_plan,
    load_formal_lock,
    matrix_status,
    prepare_formal_lock,
    run_formal,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="E8 FiLM 条件末端 × 波形解码端析因实验")
    commands = parser.add_subparsers(dest="phase", required=True)
    commands.add_parser("check-p1", help="核验 12 个模型与 36 个 derived config；不读取数据")
    commands.add_parser("formal-plan", help="输出固定的 36-cell formal 计划")
    commands.add_parser("prepare-formal-lock", help="在干净提交上生成唯一 formal execution lock")
    commands.add_parser("check-formal-lock", help="回载核验 formal lock、P2 与 train/validation 来源")
    commands.add_parser("matrix-status", help="回载核验当前 formal 36-cell 状态")
    formal = commands.add_parser("formal", help="执行一个固定 arm×seed formal cell")
    formal.add_argument("--arm", required=True, choices=ARMS)
    formal.add_argument("--seed", required=True, type=int, choices=(20260811, 20260812, 20260813))
    formal.add_argument("--device", default="cuda:0")
    formal.add_argument("--confirm-formal-training", action="store_true")
    gpu = commands.add_parser("gpu-acceptance", help="运行 P2 synthetic GPU 工程验收")
    gpu.add_argument("--device", default="cuda:0")
    benchmark = commands.add_parser("benchmark", help="运行 P2 独立进程 model-only benchmark")
    benchmark.add_argument("--device", default="cuda:0")
    worker = commands.add_parser("_benchmark-worker", help=argparse.SUPPRESS)
    worker.add_argument("--arm", required=True, choices=ARMS)
    worker.add_argument("--mode", required=True, choices=("eval", "train"))
    worker.add_argument("--device", required=True)
    worker.add_argument("--output", required=True, type=Path)
    describe = commands.add_parser("describe", help="输出结构、参数与局部 MAC 合同")
    describe.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()

    if args.phase == "check-p1":
        result = check_p1()
    elif args.phase == "formal-plan":
        result = {"cells": formal_plan(), "count": len(formal_plan())}
    elif args.phase == "prepare-formal-lock":
        result = str(prepare_formal_lock())
    elif args.phase == "check-formal-lock":
        lock, lock_hash, _source = load_formal_lock()
        result = {
            "protocol": lock["protocol"],
            "implementation_lock_sha256": lock_hash,
            "status": "passed",
        }
    elif args.phase == "matrix-status":
        result = matrix_status()
    elif args.phase == "formal":
        if not args.confirm_formal_training:
            raise SystemExit("E8 formal 要求显式传入 --confirm-formal-training")
        result = str(run_formal(args.arm, args.seed, device=args.device))
    elif args.phase == "gpu-acceptance":
        result = str(run_gpu_acceptance(args.device))
    elif args.phase == "benchmark":
        result = str(run_benchmark(args.device))
    elif args.phase == "_benchmark-worker":
        result = str(
            benchmark_worker(
                arm=args.arm,
                mode=args.mode,
                device=args.device,
                destination=args.output,
            )
        )
    else:
        arms = {
            arm: {
                **model_contract(arm),
                "role": (
                    "same-round W0 reference"
                    if arm == REFERENCE_ARM
                    else "proposed clean redesign"
                    if arm == RECOMMENDED_ARM
                    else "factorial contrast"
                ),
            }
            for arm in ARM_SPECS
        }
        result = arms if args.arm is None else arms[args.arm]
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
