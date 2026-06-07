from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.experiments.tho import _resolve_config_path, _validate_checkpoint_config, evaluate_tho_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="用 checkpoint 生成 THO 验证预测和指标")
    parser.add_argument("--config", default="", help="配置文件路径；为空时优先使用 checkpoint 同目录 config.yaml")
    parser.add_argument("--checkpoint", required=True, help="训练产生的 checkpoint.pt")
    parser.add_argument("--output", required=True, help="预测 NPZ 输出路径")
    parser.add_argument("--metrics-output", default="", help="可选指标 CSV 输出路径")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="OmegaConf dotlist 覆盖，可重复传入")
    args = parser.parse_args()

    output = evaluate_tho_checkpoint(
        checkpoint_path=args.checkpoint,
        config_path=args.config or None,
        output_path=args.output,
        metrics_output_path=args.metrics_output or None,
        overrides=args.overrides,
    )
    print(f"写出预测: {output}")


if __name__ == "__main__":
    main()
