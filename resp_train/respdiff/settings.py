"""固定论文/发布源码的方法设置；THO数据选择独立于这些参数。"""

import math

import torch


# 论文§IV-A给出30 Hz与5秒，§III-C给出lambda_spec=0.01，表II给出50 NFE。
# Adam、batch、epochs与scheduler来自breathing_bidmc_fft.py:153–175；
# betas/eps/weight_decay/amsgrad为该Adam调用采用的库默认值，现显式保存。
SOURCE_TRAINING = {
    "epochs": 400, "batch_size": 128, "shuffle": True, "drop_last": False,
    "gradient_accumulation_steps": 1, "gradient_clip_norm": None, "early_stopping": False,
    "checkpoint_policy": "final_epoch", "validation_timing": "after_training",
    "dtype": "float32", "optimizer": "adam", "learning_rate": 1e-4,
    "weight_decay": 0, "betas": [0.9, 0.999], "eps": 1e-8, "amsgrad": False,
    "scheduler": {"name": "multi_step_lr", "milestones": [280, 396],
                  "gamma": 0.1, "step_unit": "epoch_end"},
}
SOURCE_INFERENCE = {
    "batch_size": 64, "n_samples": 100, "aggregation": "mean",
    "shuffle": False, "drop_last": False,
}
SOURCE_MODEL = {"profile": "source_fft", "hidden_dim": 1024, "num_layers": 6, "output_dim": 128}


def validate_source_settings(config):
    """拒绝源方法参数漂移；小模型仅由独立synthetic_smoke字段声明。"""
    for section, expected in (("model", SOURCE_MODEL), ("training", SOURCE_TRAINING),
                              ("inference", SOURCE_INFERENCE)):
        actual = config[section]
        allowed = set(expected) | ({"seeds"} if section == "training" else set())
        if set(actual) != allowed:
            raise ValueError(f"{section}字段与来源设置不一致")
        for name, value in expected.items():
            if actual[name] != value:
                raise ValueError(f"{section}.{name}偏离论文/发布源码设置")
    seeds = config["training"]["seeds"]
    if (not isinstance(seeds, list) or not seeds or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("项目seeds须为非负且不重复的整数列表")


def build_source_optimizer(model, config):
    validate_source_settings(config)
    training = config["training"]
    return torch.optim.Adam(model.parameters(), lr=training["learning_rate"],
                            betas=tuple(training["betas"]), eps=training["eps"],
                            weight_decay=training["weight_decay"], amsgrad=training["amsgrad"])


def build_source_scheduler(optimizer, config):
    """与来源一样，每个完整epoch结束后调用scheduler.step()。"""
    validate_source_settings(config)
    scheduler = config["training"]["scheduler"]
    return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler["milestones"],
                                               gamma=scheduler["gamma"])


def dataset_dependent_budget(config, *, train_chunks, validation_chunks):
    """只推导规模，不读数据/运行网络；调用次数对应当前逐轨迹串行采样。"""
    validate_source_settings(config)
    if any(type(value) is not int or value <= 0 for value in (train_chunks, validation_chunks)):
        raise ValueError("train/validation chunk数须为正整数")
    updates_per_epoch = math.ceil(train_chunks / config["training"]["batch_size"])
    validation_batches = math.ceil(validation_chunks / config["inference"]["batch_size"])
    return {
        "train_chunks": train_chunks, "validation_chunks": validation_chunks,
        "updates_per_epoch": updates_per_epoch,
        "total_updates_per_seed": updates_per_epoch * config["training"]["epochs"],
        "validation_batches": validation_batches,
        "denoiser_calls_per_validation": validation_batches * config["diffusion"]["steps"]
                                         * config["inference"]["n_samples"],
    }
