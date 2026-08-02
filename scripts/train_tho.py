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
    parser = argparse.ArgumentParser(description="训练冻结版 THO 呼吸重建协议")
    parser.add_argument("--config", default="configs/tho_research_v2.yaml", help="配置文件路径")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="OmegaConf dotlist 覆盖，可重复传入")
    args = parser.parse_args()
    missing = check_required_packages()
    if missing:
        raise SystemExit(f"缺少依赖: {missing}")
    cfg = load_config(args.config, overrides=args.overrides)
    print(ThoExperiment(cfg).train())


if __name__ == "__main__":
    main()
