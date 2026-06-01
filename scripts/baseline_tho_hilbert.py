from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.config import load_config
from resp_train.metrics.baseline import run_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 THO 平凡基线指标")
    parser.add_argument("--config", default="configs/tho_small.yaml", help="配置文件路径")
    parser.add_argument("--output", default="baseline_metrics.csv", help="输出 CSV 路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = run_baseline(cfg, Path(args.output))
    print(f"写出平凡基线指标: {args.output} rows={len(df)}")


if __name__ == "__main__":
    main()
