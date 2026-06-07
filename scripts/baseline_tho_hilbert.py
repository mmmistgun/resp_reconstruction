from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.config import load_config
from resp_train.data.factory import build_tho_data
from resp_train.metrics.baseline import evaluate_baseline_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 THO 平凡基线指标")
    parser.add_argument("--config", default="configs/tho_small.yaml", help="配置文件路径")
    parser.add_argument("--output", default="baseline_metrics.csv", help="输出 CSV 路径")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="OmegaConf dotlist 覆盖，可重复传入")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    data = build_tho_data(cfg)
    frame = evaluate_baseline_dataset(data.val.dataset, cfg)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"写出 baseline: {output}")


if __name__ == "__main__":
    main()
