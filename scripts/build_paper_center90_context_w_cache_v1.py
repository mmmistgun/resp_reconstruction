from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.data.index import filter_index
from resp_train.data.research_v2 import read_research_v2_index
from resp_train.paper_evidence.center90_cache import (
    build_center90_w_cache_from_rows,
    fixed_center90_w_transform_spec,
    sha256_file,
)
from resp_train.paper_evidence.center90_config import load_center90_config


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 center90 的 135 s train/validation input-only W cache")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-sec", type=int, choices=(135,), required=True)
    parser.add_argument("--output-root", default="runs/paper_evidence_v1/center90_context_w_cache")
    parser.add_argument("--max-windows-per-split", type=int)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--confirm-cache-build", action="store_true")
    args = parser.parse_args()
    if not args.confirm_cache_build:
        raise SystemExit("center90 W cache build 需要显式 --confirm-cache-build")
    if args.max_windows_per_split is None:
        _assert_clean_repo()
    elif not args.allow_partial or args.max_windows_per_split <= 0:
        raise SystemExit("partial cache 要求正数 max-windows 与 --allow-partial")
    length = args.input_sec * 100
    cfg = load_center90_config(
        args.config,
        overrides=[f"window.input_samples={length}", f"window.input_sec={args.input_sec}"],
    )
    index_path = (Path(str(cfg.data.dataset_root)) / str(cfg.data.index_csv)).resolve()
    audited = read_research_v2_index(cfg.data.dataset_root, cfg.data.index_csv, cfg)
    rows_by_split = {
        split: filter_index(
            audited,
            cfg,
            split=split,
            max_windows=args.max_windows_per_split,
            sample_strategy=str(cfg.data[f"{split if split == 'train' else 'val'}_sample_strategy"]),
            sample_seed=int(cfg.data[f"{split if split == 'train' else 'val'}_sample_seed"]),
        )
        for split in ("train", "val")
    }
    if args.max_windows_per_split is None and {name: len(rows) for name, rows in rows_by_split.items()} != {
        "train": 10141,
        "val": 2675,
    }:
        raise RuntimeError("center90 完整 W cache row count 漂移")
    identity = {
        "spec": fixed_center90_w_transform_spec(length),
        "dataset_index_sha256": sha256_file(index_path),
        "train_row_ids": rows_by_split["train"]["dataset_row_id"].astype(int).tolist(),
        "val_row_ids": rows_by_split["val"]["dataset_row_id"].astype(int).tolist(),
        "partial": args.max_windows_per_split is not None,
    }
    suffix = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if args.max_windows_per_split is not None:
        suffix += "-partial"
    output = Path(args.output_root).resolve() / f"{args.input_sec}s_{suffix}"
    print(
        build_center90_w_cache_from_rows(
            output_dir=output,
            index_path=index_path,
            rows_by_split=rows_by_split,
            input_samples=length,
            dataset_index_sha256=sha256_file(index_path),
            complete=args.max_windows_per_split is None,
        )
    )


def _assert_clean_repo() -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    if result.returncode != 0 or result.stdout.strip():
        raise RuntimeError("完整 center90 W cache 要求干净 Git 工作树")


if __name__ == "__main__":
    main()
