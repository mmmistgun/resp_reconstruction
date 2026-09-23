#!/usr/bin/env python3
"""只读已有训练 history，保存提前停止的回顾性回放；不读取原始波形或 test 指标。"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adv-root", type=Path, required=True)
    parser.add_argument("--w0-root", type=Path, required=True)
    parser.add_argument("--gamma-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = [("ADV", sorted(args.adv_root.glob("*_seed*_formal_*/history.csv")), 12),
              ("W0", sorted(args.w0_root.glob("seed_*/*/train_history.csv")), 3),
              ("W0_gamma", sorted(args.gamma_root.glob("gamma*/seed_*/*/train_history.csv")), 6)]
    args.output.mkdir(parents=True, exist_ok=False)
    summaries, replays, sources, curves = [], [], [], []
    try:
        for group, paths, count in groups:
            if len(paths) != count:
                raise ValueError(f"{group} 历史数量应为 {count}")
            for path in paths:
                frame = pd.read_csv(path)
                values = frame.val_local_rr_mae.to_numpy()
                if (frame.epoch.tolist() != list(range(1, 81))
                        or frame.optimizer_update.tolist() != list(range(80, 6401, 80))
                        or not np.isfinite(frame.to_numpy(dtype=float)).all()):
                    raise ValueError(f"history 预算或有限性错误: {path}")
                name = path.parent.name if group == "ADV" else f"{path.parent.parent.parent.name}_{path.parent.parent.name}"
                traincol = "train_loss" if "train_loss" in frame else "train_loss_total"
                valcol = "val_loss" if "val_loss" in frame else "val_core_loss"
                selected = int(np.argmin(values))
                summaries.append({"group": group, "run": name, "best_epoch": selected+1,
                    "best_local_rr": values[selected], "final_local_rr": values[-1],
                    "train_loss_at_best": frame[traincol].iloc[selected], "train_loss_final": frame[traincol].iloc[-1],
                    "val_loss_at_best": frame[valcol].iloc[selected], "val_loss_final": frame[valcol].iloc[-1]})
                for minimum in (20, 30):
                    for patience in (10, 15, 20, 25):
                        best, last, stop = float("inf"), 0, 80
                        for epoch, value in enumerate(values, 1):
                            if value < best:
                                best, last = value, epoch
                            if epoch >= minimum and epoch-last >= patience:
                                stop = epoch
                                break
                        replays.append({"group": group, "run": name, "min_epochs": minimum, "patience": patience,
                            "min_delta": 0, "stop_epoch": stop, "saved_epochs": 80-stop,
                            "selected_epoch": int(np.argmin(values[:stop]))+1,
                            "missed_full_budget_best": bool(values[:stop].min() > values.min()),
                            "local_rr_regret": float(values[:stop].min()-values.min())})
                sources.append({"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
                curves.append((group, name, frame, traincol, valcol))
        pd.DataFrame(summaries).to_csv(args.output / "histories.csv", index=False)
        replay = pd.DataFrame(replays)
        replay.to_csv(args.output / "replay_per_run.csv", index=False)
        aggregate = replay.groupby(["group", "min_epochs", "patience"]).agg(
            runs=("run", "count"), mean_stop=("stop_epoch", "mean"), min_stop=("stop_epoch", "min"),
            max_stop=("stop_epoch", "max"), missed=("missed_full_budget_best", "sum"),
            max_regret=("local_rr_regret", "max"), saved_epochs=("saved_epochs", "sum")).reset_index()
        aggregate.to_csv(args.output / "replay_summary.csv", index=False)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)
        for i, group in enumerate(("ADV", "W0", "W0_gamma")):
            for g, name, frame, traincol, valcol in curves:
                if g != group:
                    continue
                for j, col in enumerate((traincol, valcol, "val_local_rr_mae")):
                    axes[i, j].plot(frame.epoch, frame[col], label=name, alpha=.85, linewidth=1)
            for j, label in enumerate(("Train loss", "Validation loss", "Validation Local RR (bpm)")):
                axes[i, j].set(title=f"{group}: {label}", xlabel="Epoch")
                axes[i, j].grid(alpha=.2)
            axes[i, 2].legend(fontsize=5)
        fig.savefig(args.output / "training_curves.png", dpi=160)
        plt.close(fig)
        payload = {"status": "complete", "scope": "read_only_stored_train_validation_histories",
            "replay": "strict Local RR improvement; min_delta=0; fixed original 80-epoch LR schedule",
            "sources": sources, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir()}}
        (args.output / "manifest.json").write_text(json.dumps(payload, indent=2)+"\n")
        print(aggregate.to_string(index=False))
        print(pd.DataFrame(summaries).to_string(index=False))
    except BaseException as exc:
        (args.output / "failed.json").write_text(json.dumps({"error":str(exc)})+"\n")
        raise


if __name__ == "__main__":
    main()
