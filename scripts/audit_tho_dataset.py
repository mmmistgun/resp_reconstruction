from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from resp_train.config import load_config
from resp_train.data.audit import add_usable_flag, summarize_audit
from resp_train.data.index import read_index


def main() -> None:
    parser = argparse.ArgumentParser(description="生成胸带小规模训练数据审计表")
    parser.add_argument("--config", default="configs/tho_small.yaml")
    parser.add_argument("--output", default="audit.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = read_index(cfg.data.dataset_root, cfg.data.index_csv)
    audited = add_usable_flag(df, cfg)
    summary = summarize_audit(audited)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(f"写出审计摘要: {output} rows={len(summary)}")


if __name__ == "__main__":
    main()
