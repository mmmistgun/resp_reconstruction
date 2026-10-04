"""按本配置完整目标验收 history；所有变体均由完整 Local RR 选点。"""
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from resp_train.crd.training import crd_learning_rate
from resp_train.crd.experiment import _early_stopping_step, _early_stopping_should_stop


def validate_history(history, cfg):
    train = cfg.training
    if (train.epochs != 80 or not train.early_stopping_enabled
            or (train.early_stopping_min_epoch, train.early_stopping_patience, train.early_stopping_min_delta) != (30, 15, 0)
            or train.batch_size != 128 or train.gradient_accumulation_steps != 1
            or (train.max_learning_rate, train.min_learning_rate, train.warmup_fraction) != (3e-4, 3e-5, .05)
            or cfg.loss.sync_weight != 1 or cfg.loss.effort_weight not in (0, .25)):
        raise ValueError("训练/早停/损失合同漂移")
    count = len(history)
    if (not 30 <= count <= 80 or not np.array_equal(history.epoch, np.arange(1, count+1))
            or not np.isfinite(history.select_dtypes(include=np.number).to_numpy()).all()
            or not np.array_equal(history.optimizer_update, np.arange(1, count+1)*80)):
        raise ValueError("history epoch/finite/update 不一致")
    total = history.train_loss_sync + float(cfg.loss.effort_weight)*history.train_loss_effort
    if not np.allclose(history.train_loss_total, total, atol=1e-12, rtol=0):
        raise ValueError("history loss 与本配置目标不一致")
    best, wait, triggered = float("inf"), 0, []
    for row in history.itertuples():
        for column, update in (("first_learning_rate", (row.epoch-1)*80), ("last_learning_rate", row.epoch*80-1)):
            lr = crd_learning_rate(update, total_updates=6400, max_learning_rate=3e-4, min_learning_rate=3e-5, warmup_fraction=.05)
            if not np.isclose(getattr(row, column), lr, atol=1e-15, rtol=0):
                raise ValueError("固定6400-update LR计划漂移")
        improved, wait = _early_stopping_step(value=float(row.val_local_rr_mae), best=best,
                           epochs_without_improvement=wait, min_delta=0)
        if improved:
            best = float(row.val_local_rr_mae)
        stop = _early_stopping_should_stop(epoch=int(row.epoch), min_epoch=30, epochs_without_improvement=wait, patience=15)
        if (row.early_stopping_improved != int(improved) or row.early_stopping_wait != wait
                or row.early_stopping_triggered != int(stop) or row.early_stopping_min_epoch != 30):
            raise ValueError("早停轨迹漂移")
        triggered.append(stop)
    if any(triggered[:-1]) or (count < 80 and not triggered[-1]):
        raise ValueError("训练停止位置不符")
    # np.argmin 在相等值时返回最早索引。
    return int(history.iloc[int(np.argmin(history.val_local_rr_mae.to_numpy()))].epoch)


def selected_epoch(run):
    return validate_history(pd.read_csv(run / "train_history.csv"), OmegaConf.load(run / "config.yaml"))
