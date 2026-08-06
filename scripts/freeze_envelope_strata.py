from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from resp_train.config import load_config
from resp_train.data.factory import build_window_data
from resp_train.metrics.task import compute_target_envelope_modulations, envelope_strata_cutpoints


def main() -> None:
    parser = argparse.ArgumentParser(description="使用完整 admitted training targets 冻结包络三分层阈值")
    parser.add_argument("--config", default="configs/tho_research_v2.yaml")
    parser.add_argument("--output", default="runs/envelope_strata_train.json")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="OmegaConf dotlist 覆盖，可重复传入")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    cfg.data.preload_windows = False
    split = str(cfg.data.train_split)
    data = build_window_data(
        cfg,
        split=split,
        max_windows=cfg.data.get("max_train_windows"),
        sample_strategy=str(cfg.data.train_sample_strategy),
        sample_seed=int(cfg.data.train_sample_seed),
        shuffle=False,
    )

    chunks: list[np.ndarray] = []
    for batch in data.loader:
        target = batch["target"].detach().cpu().numpy()
        chunks.append(compute_target_envelope_modulations(target, cfg))
    if not chunks:
        raise RuntimeError("admitted training split 为空，无法冻结包络分层阈值")

    values = np.concatenate(chunks).astype(np.float64, copy=False)
    method = str(cfg.evaluation.get("envelope_quantile_method", "linear"))
    low, high = envelope_strata_cutpoints(values, method=method)
    row_ids = data.rows["dataset_row_id"].to_numpy(dtype=np.int64, copy=True)
    artifact = {
        "config": str(Path(args.config)),
        "split": split,
        "n_training_targets": int(values.size),
        "dataset_row_ids_sha256": sha256(row_ids.tobytes()).hexdigest(),
        "train_sample_strategy": str(cfg.data.train_sample_strategy),
        "train_sample_seed": int(cfg.data.train_sample_seed),
        "quantile_method": method,
        "quantiles": [1.0 / 3.0, 2.0 / 3.0],
        "envelope_strata_low": low,
        "envelope_strata_high": high,
        "target_modulation_min": float(np.min(values)),
        "target_modulation_median": float(np.median(values)),
        "target_modulation_max": float(np.max(values)),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact, ensure_ascii=False, indent=2))
    print(f"写出包络分层冻结产物: {output}")


if __name__ == "__main__":
    main()
