"""M4 片段池化位置消融；保持原 P1 的共有参数初始化及双投影。"""
from dataclasses import asdict, dataclass

import torch
from torch.nn import functional as F

from resp_train.crd.initialization import named_subseed
from resp_train.models.patch_aligned_tf_mamba import ContinuousPatchEncoder, PatchTFConfig, _local_call
from scripts.p1_components_model import P1ComponentsModel, architecture, band_indices, frequency_metadata


@dataclass(frozen=True)
class PositionArm:
    name: str
    score_position: bool
    value_position: bool
    band: str = "W-full"
    mamba: bool = True


ALL_ARMS = {
    "A": PositionArm("M4-PoolPosition-Both", True, True),
    "B": PositionArm("M4-PoolPosition-Content", False, False),
    "C": PositionArm("M4-PoolPosition-Score", True, False),
    "D": PositionArm("M4-PoolPosition-Value", False, True),
}
# 第二阶段只定义数学实现；正式入口和配置矩阵仅开放 A/B。
ARMS = {a: ALL_ARMS[a] for a in ("A", "B")}


class PositionPatchEncoder(ContinuousPatchEncoder):
    def __init__(self, cfg, score_position, value_position):
        # 按冻结构造顺序消耗 RNG，再移除 B 的位置模块，使 score/project 精确配对。
        # B 最终 state_dict、optimizer 和前向图均不含位置网络。
        super().__init__(cfg)
        self.score_position = score_position
        self.value_position = value_position
        if not (score_position or value_position):
            del self.position

    def _pool_chunk(self, features, scores, position, position_scores):
        cfg = self.config
        windows = features.unfold(-1, cfg.patch_samples, cfg.patch_hop_samples)
        logits = scores.unfold(-1, cfg.patch_samples, cfg.patch_hop_samples).float()
        if self.score_position:
            logits = logits + position_scores.float()
        weights = logits.softmax(-1)
        pooled = torch.einsum("bnp,bcnp->bnc", weights.to(features.dtype), windows)
        if self.value_position:
            pooled = pooled + torch.einsum("bnp,pc->bnc", weights.to(position.dtype), position)
        return pooled

    def forward(self, waveform):
        if self.score_position and self.value_position:
            return super().forward(waveform)
        features = self.stem(waveform)
        position = (self.position(self.relative_seconds[:, None]).to(features.dtype)
                    if self.score_position or self.value_position else None)
        scores = self.score(features.transpose(1, 2)).squeeze(-1)
        position_scores = F.linear(position, self.score.weight).squeeze(-1) if self.score_position else None
        cfg, chunks = self.config, []
        for start in range(0, cfg.patch_count, cfg.patch_chunk_size):
            stop = min(start + cfg.patch_chunk_size, cfg.patch_count)
            lo = start * cfg.patch_hop_samples
            hi = (stop - 1) * cfg.patch_hop_samples + cfg.patch_samples
            chunks.append(_local_call(self._pool_chunk, cfg.checkpoint_local,
                features[..., lo:hi], scores[..., lo:hi], position, position_scores))
        return self.project(torch.cat(chunks, dim=1))


class PoolPositionModel(P1ComponentsModel):
    def __init__(self, arm, seed, frequencies_hz, *, cfg=None, mamba_factory=None):
        if arm not in ALL_ARMS:
            raise ValueError("未知片段池化位置配置")
        super().__init__("M4", seed, frequencies_hz, cfg=cfg, mamba_factory=mamba_factory)
        self.arm = arm
        option = ALL_ARMS[arm]
        # 重建仅使用 CPU generator，并恢复其状态；训练随机序列沿用原 M4 构造结果。
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(named_subseed(seed, "p1_components.waveform_encoder"))
            self.waveform_encoder = PositionPatchEncoder(self.config, option.score_position, option.value_position)


def contract(arm):
    return {"arm": arm, **asdict(ARMS[arm])}


def build_model(cfg, *, mamba_factory=None):
    value = dict(cfg.model.m4_pool_position)
    arm = value["arm"]
    if arm not in ARMS or value != contract(arm) or cfg.model.name != "m4_pool_position_v1":
        raise ValueError("M4 池化位置第一阶段合同不符")
    return PoolPositionModel(arm, int(cfg.model.initialization_seed), frequency_metadata(cfg.data.tf_cache_path),
        cfg=PatchTFConfig(**dict(cfg.model.architecture)), mamba_factory=mamba_factory)
