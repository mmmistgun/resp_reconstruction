"""可变CWT网格与M4-S1模块消融，沿用L1共有参数初始化和训练数学。"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache

import numpy as np
import torch
from torch import nn

from resp_train.crd.initialization import module_seed
from resp_train.models.patch_aligned_tf_mamba import LocalTFAttention, PatchTFConfig, _finite, _local_call
from scripts.m4_components_model import ZeroCoordinateBias
from scripts.m4_linear_fusion_model import LinearFusionModel
from scripts.m4_pool_position_model import PositionPatchEncoder
from scripts.p1_components_model import UniformPatchEncoder, architecture
from scripts.m4_s1_cwt_parameters_spec import ARMS, CWT_SETTINGS, contract


@lru_cache(maxsize=1)
def baseline_frequencies():
    from resp_train.crd.tf_v1_features import morlet_scales_and_frequencies
    _, values = morlet_scales_and_frequencies(12, 97)
    values = np.sort(values, kind="stable").copy()
    values.setflags(write=False)
    return values


def validate_representation(arm, rep):
    key = ARMS[arm].representation
    if key is None:
        if rep is not None:
            raise ValueError("NoTF不接收CWT表示")
        return
    settings = CWT_SETTINGS[key]
    expected = {"name": key, **{k: v for k, v in settings.items() if k != "id"}}
    if not isinstance(rep, dict) or rep.get("arm") != expected:
        raise ValueError("CWT表示与实验臂不符")
    frequencies = np.asarray(rep["frequencies_hz"], dtype=np.float64)
    frames = 18000 // settings["pool_samples"]
    if (frequencies.ndim != 1 or not len(frequencies) or not np.isfinite(frequencies).all()
            or np.any(frequencies <= 0) or np.any(np.diff(frequencies) < 0)
            or rep["shape"] != [len(frequencies), frames]
            or rep["sample_rate"] != 100 or rep["window_samples"] != 18000
            or rep["feature"] != "mean_pool_log1p_abs" or rep["padtype"] != "reflect"
            or rep["library"] != "ssqueezepy==0.6.6"):
        raise ValueError("CWT频率、shape或变换合同非法")
    expected_times = (np.arange(frames) * settings["pool_samples"] + (settings["pool_samples"] - 1) / 2) / 100
    if not np.array_equal(rep["time_seconds"], expected_times):
        raise ValueError("CWT时间坐标非法")
    if settings["high_only"] and not np.all((frequencies > .8) & (frequencies <= 8)):
        raise ValueError("H频带越界")
    if settings["high_hz"] != 8 and np.any(frequencies > settings["high_hz"]):
        raise ValueError("CWT频带上限越界")


class GridAttention(LocalTFAttention):
    """先编码原生F×T网格，再在固定360个物理中心上读取，保留原分块注意力。"""

    def __init__(self, original, rep, *, uniform=False, frequency_bias=True):
        nn.Module.__init__(self)
        self.config, self.heads, self.head_dim = original.config, original.heads, original.head_dim
        self.encoder = original.encoder
        self.uniform = uniform
        self.pool_samples = int(rep["arm"]["pool_samples"])
        if not uniform:
            self.query, self.key = original.query, original.key
        # 保持B的模块注册/optimizer分组/梯度裁剪遍历顺序与原生L1一致。
        self.value, self.output = original.value, original.output
        if not uniform:
            self.frequency_bias = original.frequency_bias if frequency_bias else ZeroCoordinateBias(self.heads)
            self.time_bias = original.time_bias
        self.register_buffer("frequencies_hz", torch.tensor(rep["frequencies_hz"], dtype=torch.float64))
        self.register_buffer("relative_seconds", original.relative_seconds.detach().clone())
        # q[k]=(50k+24.5-(p-1)/2)/p；固定网格共享相邻patch读取点。
        q = (torch.arange(360, dtype=torch.float32) * 50 + 24.5 - (self.pool_samples - 1) / 2) / self.pool_samples
        q = q.clamp(0, rep["shape"][1] - 1)
        low = q.floor().long()
        self.register_buffer("sample_lo", low, persistent=False)
        self.register_buffer("sample_hi", (low + 1).clamp(max=rep["shape"][1] - 1), persistent=False)
        self.register_buffer("sample_weight", q - low.float(), persistent=False)

    def encoded_grid(self, condition):
        features = self.encoder(condition[:, None])
        if self.pool_samples == 50:
            return features
        # 先转float32再做线性插值，之后恢复encoder输出dtype，保持AMP边界。
        with torch.autocast(device_type=features.device.type, enabled=False):
            lo = features.index_select(-1, self.sample_lo).float()
            hi = features.index_select(-1, self.sample_hi).float()
            output = lo * (1 - self.sample_weight) + hi * self.sample_weight
        return output.to(features.dtype)

    def _uniform_chunk(self, values, count):
        with torch.autocast(device_type=values.device.type, enabled=False):
            # 每头在原生尺度和两个读取位置上使用1/(2F)，V和output保持原参数化。
            context = (values[:, :, :count].float().sum(1) + values[:, :, 1:count + 1].float().sum(1))
            context = context / (2 * values.shape[1])
        return context.flatten(-2)

    def forward(self, tokens, condition):
        features = self.encoded_grid(condition)
        cells = features.movedim(1, -1)
        b, n, _ = tokens.shape
        # 正常分支保持原K/V/Q计算顺序，B不引入插值或额外数值转换。
        if not self.uniform:
            key = self.key(cells).unflatten(-1, (self.heads, self.head_dim))
        value = self.value(cells).unflatten(-1, (self.heads, self.head_dim))
        if not self.uniform:
            query = self.query(tokens).reshape(b, n, self.heads, self.head_dim)
            frequency_bias = self.frequency_bias(self.frequencies_hz.log().float()[:, None])
            time_bias = self.time_bias(self.relative_seconds[:, None])
            bias = (frequency_bias[:, None, :] + time_bias[None, :, :]).permute(2, 0, 1)
        contexts = []
        for start in range(0, n, self.config.patch_chunk_size):
            stop = min(start + self.config.patch_chunk_size, n)
            if self.uniform:
                contexts.append(_local_call(self._uniform_chunk, self.config.checkpoint_local,
                                           value[:, :, start:stop + 1], stop - start))
            else:
                contexts.append(_local_call(self._attend_chunk, self.config.checkpoint_local,
                    query[:, start:stop], key[:, :, start:stop + 1], value[:, :, start:stop + 1], bias))
        return self.output(torch.cat(contexts, dim=1).to(tokens.dtype))


class M4S1CWTModel(LinearFusionModel):
    def __init__(self, arm, seed, representation=None, *, cfg=None, mamba_factory=None, canonical_frequencies=None):
        if arm not in ARMS:
            raise ValueError("未知M4-S1实验臂")
        validate_representation(arm, representation)
        canonical = baseline_frequencies() if canonical_frequencies is None else canonical_frequencies
        super().__init__("L1", seed, canonical, cfg=cfg, mamba_factory=mamba_factory)
        self.arm, self.spec = arm, ARMS[arm]
        self.representation = deepcopy(representation)
        if self.spec.representation is None:
            self.condition, self.fusion = None, None
        else:
            self.condition = GridAttention(self.condition, representation,
                uniform=self.spec.readout == "uniform", frequency_bias=self.spec.frequency_bias)
        if not self.spec.mamba:
            self.trunk = nn.Identity()
        if self.spec.pooling == "uniform":
            with module_seed(seed, "p1_components.waveform_encoder"):
                self.waveform_encoder = UniformPatchEncoder(self.config)
        elif not (self.spec.score_position and self.spec.value_position):
            with module_seed(seed, "p1_components.waveform_encoder"):
                self.waveform_encoder = PositionPatchEncoder(self.config, self.spec.score_position, self.spec.value_position)

    def forward(self, x, *, tf=None, **kwargs):
        if (x.ndim != 3 or x.shape[1:] != (1, 18000) or len(x) < 1 or not x.is_floating_point()):
            raise ValueError("M4-S1输入要求浮点(B,1,18000)")
        _finite(x, "BCG")
        if self.condition is None:
            if tf:
                raise ValueError("NoTF不接收CWT条件")
        else:
            if (tf is None or set(tf) != {"w"} or tf["w"].shape != (len(x), *self.representation["shape"])
                    or tf["w"].device != x.device or not tf["w"].is_floating_point()):
                raise ValueError("CWT条件shape/dtype/device不符")
            _finite(tf["w"], "CWT")
        z = self.waveform_encoder(x)
        u = z if self.condition is None else self.fusion(z, self.condition(z, tf["w"]))
        output = self.overlap(self.decoder(self.trunk(u)))
        _finite(output, "M4-S1重建输出")
        return {"waveform": output}


def build_model(cfg, *, mamba_factory=None):
    from omegaconf import OmegaConf
    value = OmegaConf.to_container(cfg.model.m4_s1_cwt_parameters, resolve=True)
    arm = value["arm"]
    if cfg.model.name != "m4_s1_cwt_parameters_v1" or value != contract(arm):
        raise ValueError("M4-S1模型合同不符")
    rep = OmegaConf.to_container(cfg.model.cwt_representation, resolve=True) if cfg.model.cwt_representation is not None else None
    return M4S1CWTModel(arm, int(cfg.model.initialization_seed), rep,
                      cfg=PatchTFConfig(**dict(cfg.model.architecture)), mamba_factory=mamba_factory)
