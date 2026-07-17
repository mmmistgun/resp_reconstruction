from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from resp_train.config import load_config
from resp_train.data.factory import build_tho_data, build_window_data
from resp_train.data.independence import audit_all_split_independence


def build_full_split_rows(cfg: DictConfig) -> dict[str, pd.DataFrame]:
    """按同一过滤口径构造全量 train/val/test；审计不继承训练抽样上限。"""

    cfg.data.max_train_windows = None
    cfg.data.max_val_windows = None
    cfg.data.preload_windows = False
    data = build_tho_data(cfg)

    test_split = cfg.data.get("test_split")
    if not test_split:
        raise ValueError("独立性审计需要 data.test_split")
    test_seed = cfg.data.get("test_sample_seed")
    test = build_window_data(
        cfg,
        split=str(test_split),
        max_windows=None,
        sample_strategy=cfg.data.get("test_sample_strategy"),
        sample_seed=int(test_seed) if test_seed is not None else None,
        shuffle=False,
        audited=data.audited,
    )
    if test.rows.empty:
        raise RuntimeError("test 数据为空，请检查 test_split、input_set 和可用性过滤配置。")
    return {"train": data.train.rows, "val": data.val.rows, "test": test.rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="全量审计 THO train/val/test 的个体和片段独立性")
    parser.add_argument("--config", required=True, help="训练配置路径")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="OmegaConf dotlist 覆盖；max_*_windows 在审计中固定为 null",
    )
    parser.add_argument("--output-dir", required=True, help="审计 CSV 输出目录")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    split_rows = build_full_split_rows(cfg)
    report = audit_all_split_independence(split_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in report.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)

    for summary in report["summary"].to_dict("records"):
        print(
            "split_independence "
            f"pair={summary['left_split']}-{summary['right_split']} "
            f"left_windows={summary['left_windows']} "
            f"right_windows={summary['right_windows']} "
            f"overlap_samp_id_count={summary['overlap_samp_id_count']} "
            f"overlap_segment_count={summary['overlap_segment_count']}"
        )


if __name__ == "__main__":
    main()
