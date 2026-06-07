from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from resp_train.config import load_config
from resp_train.data.factory import build_tho_data


def main() -> None:
    parser = argparse.ArgumentParser(description="生成胸带小规模训练数据审计表")
    parser.add_argument("--config", default="configs/tho_small.yaml")
    parser.add_argument("--output", default="audit.csv")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="OmegaConf dotlist 覆盖，可重复传入")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    data = build_tho_data(cfg)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data.audit_summary.to_csv(output, index=False)
    print(f"写出审计: {output}")


if __name__ == "__main__":
    main()
