from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from resp_train.config import load_config
from resp_train.data.factory import build_tho_data
from resp_train.data.independence import audit_split_independence


def main() -> None:
    parser = argparse.ArgumentParser(description="审计 THO train/val split 的个体和片段独立性")
    parser.add_argument("--config", required=True, help="训练配置路径")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="OmegaConf dotlist 覆盖")
    parser.add_argument("--output-dir", required=True, help="审计 CSV 输出目录")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    data = build_tho_data(cfg)
    report = audit_split_independence(data.train.rows, data.val.rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in report.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)

    summary = report["summary"].iloc[0].to_dict()
    print(
        "split_independence "
        f"train_windows={summary['train_windows']} "
        f"val_windows={summary['val_windows']} "
        f"overlap_samp_id_count={summary['overlap_samp_id_count']} "
        f"overlap_segment_count={summary['overlap_segment_count']}"
    )


if __name__ == "__main__":
    main()
