from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.config import check_required_packages, load_config
from resp_train.experiments.tho import ThoExperiment


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 THO 呼吸努力估计实验")
    parser.add_argument("--config", default="configs/tho_small.yaml", help="配置文件路径")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="OmegaConf dotlist 覆盖，可重复传入")
    args = parser.parse_args()

    missing = check_required_packages()
    if missing:
        raise SystemExit(f"缺少依赖: {missing}; 请先确认是否安装。")

    cfg = load_config(args.config, overrides=args.overrides)
    run_dir = ThoExperiment(cfg).train()
    print(run_dir)


if __name__ == "__main__":
    main()
