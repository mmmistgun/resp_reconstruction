"""M4配对重训：RR负责早停，validation总loss仅额外记录checkpoint。"""
from pathlib import Path
import time
from uuid import uuid4

import torch

from resp_train import patch_aligned_tf_training as legacy
from resp_train.crd.training import optimizer_updates_per_epoch
from resp_train.engine import validate


def run_epochs(model, data, loss, optimizer, cfg, cell, *, resume=False, log_path=None):
    """按原RR规则训练；两种selector共享完整epoch快照，各自保留最早严格最小值。"""
    cell = Path(cell)
    (cell / "epochs").mkdir(exist_ok=True)
    prior = legacy.completed_epochs(cell)
    if prior and not resume:
        raise FileExistsError("已有epoch，请显式使用 --resume")
    updates_per_epoch = optimizer_updates_per_epoch(len(data.train.loader), int(cfg.training.gradient_accumulation_steps))
    total = int(cfg.training.epochs) * updates_per_epoch
    update, history, best, best_epoch, wait, stopped = 0, [], float("inf"), None, 0, False
    best_loss, best_loss_epoch = float("inf"), None
    if prior:
        state = torch.load(prior[-1][1] / "checkpoint.pt", map_location="cpu", weights_only=False)
        legacy.finite_tree(state)
        if state["config"] != legacy.config_identity(cfg) or state["total_updates"] != total:
            raise ValueError("恢复checkpoint配置或预算不一致")
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        update, history, best, best_epoch, wait, stopped = (state[k] for k in
            ("update_index", "history", "best_local_rr", "best_epoch", "wait", "stopped"))
        best_loss, best_loss_epoch = state["best_validation_loss"], state["best_loss_epoch"]
        if len(history) != len(prior) or history[-1]["epoch"] != prior[-1][0]:
            raise ValueError("checkpoint/history/epoch不一致")
        if (update != len(history) * updates_per_epoch or best_epoch not in range(1, len(history) + 1)
                or best_loss_epoch not in range(1, len(history) + 1)):
            raise ValueError("checkpoint更新计数或selector epoch不一致")
        if min(history, key=lambda r: r["val_local_rr_mae"])["epoch"] != best_epoch:
            raise ValueError("恢复RR选点与history不符")
        selected_loss = min(history, key=lambda r: r["validation"]["loss"])
        if selected_loss["epoch"] != best_loss_epoch or selected_loss["validation"]["loss"] != best_loss:
            raise ValueError("恢复loss选点与history不符")
        legacy.restore_rng(state["rng"], cfg.training.device)
    for epoch in range(len(history) + 1, int(cfg.training.epochs) + 1):
        if stopped:
            break
        started = time.perf_counter()
        train_summary, update = legacy._train_epoch(model, data.train.loader, loss, optimizer, cfg, update, total, epoch)
        val_summary, predictions = validate(model, data.val.loader, loss, device=cfg.training.device,
            use_amp=bool(cfg.training.use_amp), show_progress=False, return_predictions=True)
        score = float(legacy.validation_local_rr_mean(predictions, cfg))
        val_loss = float(val_summary["loss"])
        legacy.finite_tree((train_summary, val_summary, score, val_loss))
        if score < best - float(cfg.training.early_stopping_min_delta):
            best, best_epoch, wait = score, epoch, 0
        else:
            wait += 1
        # 仅更新loss选点，不改变RR的best/wait/stopped；并列保留最早epoch。
        if val_loss < best_loss:
            best_loss, best_loss_epoch = val_loss, epoch
        stopped = epoch >= int(cfg.training.early_stopping_min_epoch) and wait >= int(cfg.training.early_stopping_patience)
        record = {"epoch": epoch, "optimizer_update": update, "train": train_summary,
            "validation": val_summary, "val_local_rr_mae": score, "best_epoch": best_epoch,
            "best_validation_loss": best_loss, "best_loss_epoch": best_loss_epoch,
            "wait": wait, "stopped": stopped, "elapsed_seconds": time.perf_counter() - started}
        history.append(record)
        state = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "config": legacy.config_identity(cfg), "epoch": epoch, "update_index": update, "total_updates": total,
            "history": history, "best_local_rr": best, "best_epoch": best_epoch, "wait": wait,
            "best_validation_loss": best_loss, "best_loss_epoch": best_loss_epoch,
            "stopped": stopped, "rng": legacy.rng_state(cfg.training.device)}
        legacy.finite_tree(state)
        directory = cell / "epochs" / f"epoch_{epoch:04d}_{uuid4().hex[:12]}"
        directory.mkdir()
        with (directory / "checkpoint.pt").open("xb") as stream:
            torch.save(state, stream)
        legacy.write_json(directory / "record.json", record)
        legacy.write_json(directory / "receipt.json", {"epoch": epoch, "checkpoint_sha256": legacy.sha(directory / "checkpoint.pt")})
        line = (f"epoch={epoch} update={update}/{total} val_local_rr={score:.6f} best_epoch={best_epoch} "
            f"val_loss={val_loss:.6f} best_loss_epoch={best_loss_epoch} wait={wait}")
        print(line, flush=True)
        if log_path is not None:
            with Path(log_path).open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
    if best_epoch is None or best_loss_epoch is None:
        raise RuntimeError("没有有效validation checkpoint")
    epochs = dict(legacy.completed_epochs(cell))
    return {"best_epoch": best_epoch, "best_local_rr": best,
        "best_checkpoint": str((epochs[best_epoch] / "checkpoint.pt").relative_to(cell)),
        "best_loss_epoch": best_loss_epoch, "best_validation_loss": best_loss,
        "best_loss_checkpoint": str((epochs[best_loss_epoch] / "checkpoint.pt").relative_to(cell)),
        "final_checkpoint": str((epochs[len(history)] / "checkpoint.pt").relative_to(cell)),
        "completed_epochs": len(history), "optimizer_updates": update, "early_stopped": stopped}
