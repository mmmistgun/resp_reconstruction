"""M4 池化位置路径实验：复用四种池化定义和原始 P1 初始化。"""
from dataclasses import asdict

from resp_train.models.patch_aligned_tf_mamba import PatchTFConfig
from scripts.m4_pool_position_model import (
    ALL_ARMS, PoolPositionModel, architecture, band_indices, frequency_metadata,
)

ARMS = dict(ALL_ARMS)
TRAIN_ARMS = ("C", "D")
REFERENCES = ("A", "B")


def contract(arm):
    return {"arm": arm, **asdict(ARMS[arm])}


def build_model(cfg, *, mamba_factory=None):
    value = dict(cfg.model.m4_pool_position_paths)
    arm = value["arm"]
    if arm not in TRAIN_ARMS or value != contract(arm) or cfg.model.name != "m4_pool_position_paths_v1":
        raise ValueError("M4 池化位置路径训练合同不符")
    return PoolPositionModel(arm, int(cfg.model.initialization_seed), frequency_metadata(cfg.data.tf_cache_path),
        cfg=PatchTFConfig(**dict(cfg.model.architecture)), mamba_factory=mamba_factory)
