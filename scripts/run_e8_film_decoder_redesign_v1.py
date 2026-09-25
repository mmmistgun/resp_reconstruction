from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resp_train.paper_evidence.e8_film_decoder_redesign_v1 import check_p1, formal_plan
from resp_train.paper_evidence.e8_film_decoder_redesign_v1_model import (
    ARMS,
    ARM_SPECS,
    RECOMMENDED_ARM,
    REFERENCE_ARM,
    model_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="E8 FiLM 条件末端 × 波形解码端析因实验")
    commands = parser.add_subparsers(dest="phase", required=True)
    commands.add_parser("check-p1", help="核验 12 个模型与 36 个 derived config；不读取数据")
    commands.add_parser("formal-plan", help="输出冻结前的 36-cell 计划；formal gate 仍关闭")
    describe = commands.add_parser("describe", help="输出结构、参数与局部 MAC 合同")
    describe.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()

    if args.phase == "check-p1":
        result = check_p1()
    elif args.phase == "formal-plan":
        result = {"cells": formal_plan(), "count": len(formal_plan()), "formal_gate": "closed"}
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
