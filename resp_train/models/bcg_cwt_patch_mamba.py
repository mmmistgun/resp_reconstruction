"""连续双表征编码、同区间交叉注意力与 BiMamba2 呼吸重建。"""
from __future__ import annotations

from collections.abc import Mapping, Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from importlib.metadata import version, PackageNotFoundError
import math
from typing import Any

import torch
from torch import nn

from resp_train.crd.blocks import BidirectionalMamba2Block
from resp_train.models.patch_aligned_tf_mamba import LocalResidual, ScaleTimePad


MODEL_NAME = "bcg_cwt_patch_mamba_v1"
DEFAULT_ARCHITECTURE = {
    "sample_rate": 100, "window_samples": 18000, "patch_samples": 100,
    "patch_hop_samples": 100, "cwt_pool_samples": 50, "cwt_frequency_count": 97,
    "dimension": 96,
    "waveform_encoder": {
        "stage1_channels": 32, "stage1_kernel": 7, "stage1_residual_kernel": 5,
        "stage2_channels": 64, "stage2_kernel": 5, "stage2_residual_kernel": 3,
        "context_kernel": 3, "context_dilations": [1, 2, 4, 8, 16],
        "residual_type": "pre_norm_depthwise_gelu_pointwise",
        "norm": "per_position_channel_layernorm", "norm_eps": 1e-5,
        "padding_mode": "reflect", "activation": "gelu", "conv_bias": True,
    },
    "patch_embedding": {"type": "ordered_feature_patch_projection", "implementation": "conv1d",
                        "padding": 0, "bias": True},
    "tf_encoder": {"channels": 32, "stem_kernel": 3, "stem_norm": "groupnorm",
                   "norm_groups": 8, "norm_eps": 1e-5, "local_depthwise_kernel": 3,
                   "activation": "gelu", "conv_bias": True, "time_padding": "reflect",
                   "frequency_padding": "replicate"},
    "tf_position": {"type": "additive_frequency_and_local_time_tables", "initialization": "normal",
                    "initialization_std": .02},
    "fusion": {"type": "aligned_patch_cross_attention_block", "layers": 1, "heads": 4,
               "key_input_dimension": 32, "value_input_dimension": 32,
               "attention_dropout": 0., "residual_dropout": .1, "ffn_hidden_dimension": 384,
               "ffn_dropout": .1, "activation": "gelu", "norm": "layernorm", "norm_eps": 1e-5,
               "bias": True, "patch_chunk_size": 16},
    "temporal": {"type": "bidirectional_mamba2", "placement": "after_fusion", "layers": 6,
                 "parameter_sharing": "independent_blocks_and_directions", "external_norm": "rmsnorm",
                 "external_norm_eps": 1e-5, "residual_dropout": .1,
                 "mamba2": {"d_state": 64, "d_conv": 4, "expand": 2, "headdim": 32,
                            "ngroups": 1, "chunk_size": 256, "rmsnorm": True, "bias": False,
                            "conv_bias": True, "use_mem_eff_path": True}},
    "decoder": {"type": "shared_patch_mlp", "hidden_dimension": 96, "activation": "gelu", "bias": True},
    "reconstruction": {"type": "chronological_patch_concatenation", "output_dtype": "float32"},
}


def _merge_checked(defaults, supplied, prefix="architecture"):
    """严格接收已定义字段，避免 YAML 拼写错误或未实现字段被静默忽略。"""
    if not isinstance(supplied, Mapping):
        raise TypeError(f"{prefix} 必须是 mapping")
    unknown = set(supplied) - set(defaults)
    if unknown:
        raise ValueError(f"{prefix} 未知字段: {sorted(unknown)}")
    result = deepcopy(defaults)
    for key, value in supplied.items():
        path = f"{prefix}.{key}"
        expected = defaults[key]
        if isinstance(expected, dict):
            result[key] = _merge_checked(expected, value, path)
        elif isinstance(expected, list):
            if not isinstance(value, (list, tuple)) or not value or any(type(v) is not int or v <= 0 for v in value):
                raise ValueError(f"{path} 必须是非空正整数序列")
            result[key] = list(value)
        else:
            if isinstance(expected, float):
                valid = type(value) in (int, float) and math.isfinite(value)
            else:
                valid = type(value) is type(expected)
            if not valid:
                raise ValueError(f"{path} 类型或数值无效: {value!r}")
            if isinstance(expected, (str, bool)) and value != expected:
                raise ValueError(f"{path} 当前实现要求 {expected!r}")
            if type(expected) is int and key != "padding" and value <= 0:
                raise ValueError(f"{path} 必须为正整数")
            if "dropout" in key and not 0 <= value < 1:
                raise ValueError(f"{path} 必须在 [0,1) 内")
            if ("eps" in key or key == "initialization_std") and value <= 0:
                raise ValueError(f"{path} 必须为正数")
            result[key] = value
    return result


@dataclass
class BCGCWTConfig:
    """独立结构配置；正式时间几何固定，计算分块大小可调。"""

    architecture: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        from omegaconf import OmegaConf
        if OmegaConf.is_config(self.architecture):
            self.architecture = OmegaConf.to_container(self.architecture, resolve=True)
        a = _merge_checked(DEFAULT_ARCHITECTURE, self.architecture)
        for key in ("sample_rate", "window_samples", "patch_samples", "patch_hop_samples",
                    "cwt_pool_samples", "cwt_frequency_count"):
            if a[key] != DEFAULT_ARCHITECTURE[key]:
                raise ValueError(f"architecture.{key} 正式几何要求 {DEFAULT_ARCHITECTURE[key]}")
        w, t, f, m = a["waveform_encoder"], a["tf_encoder"], a["fusion"], a["temporal"]
        fixed = {"waveform_encoder.norm_eps": (w["norm_eps"], 1e-5),
                 "patch_embedding.padding": (a["patch_embedding"]["padding"], 0),
                 "tf_encoder.stem_kernel": (t["stem_kernel"], 3),
                 "tf_encoder.local_depthwise_kernel": (t["local_depthwise_kernel"], 3),
                 "fusion.layers": (f["layers"], 1),
                 "temporal.external_norm_eps": (m["external_norm_eps"], 1e-5)}
        for name, (actual, expected) in fixed.items():
            if actual != expected:
                raise ValueError(f"architecture.{name} 当前实现要求 {expected}")
        if a["dimension"] % f["heads"]:
            raise ValueError("dimension 必须是 fusion.heads 的整数倍")
        if t["channels"] % t["norm_groups"]:
            raise ValueError("tf_encoder.channels 必须是 norm_groups 的整数倍")
        if (f["key_input_dimension"], f["value_input_dimension"]) != (t["channels"], t["channels"]):
            raise ValueError("fusion key/value_input_dimension 必须等于 tf_encoder.channels")
        ssm = m["mamba2"]
        inner = a["dimension"] * ssm["expand"]
        if inner % ssm["headdim"] or (inner // ssm["headdim"]) % ssm["ngroups"]:
            raise ValueError("Mamba2 内部维度/headdim/ngroups 不可整除")
        for key in ("stage1_kernel", "stage2_kernel", "stage1_residual_kernel", "stage2_residual_kernel", "context_kernel"):
            if w[key] % 2 != 1:
                raise ValueError(f"waveform_encoder.{key} 必须为奇数")
        pads = [w[k] // 2 for k in ("stage1_kernel", "stage2_kernel", "stage1_residual_kernel", "stage2_residual_kernel")]
        pads += [w["context_kernel"] // 2 * d for d in w["context_dilations"]]
        if max(pads) >= a["window_samples"]:
            raise ValueError("waveform_encoder reflect padding 必须小于窗口长度")
        self.architecture = a

    def to_dict(self):
        return deepcopy(self.architecture)

    @property
    def patch_count(self):
        a = self.architecture
        return (a["window_samples"] - a["patch_samples"]) // a["patch_hop_samples"] + 1

    @property
    def bins_per_patch(self):
        return self.architecture["patch_samples"] // self.architecture["cwt_pool_samples"]

    @property
    def step_tf(self):
        return self.architecture["patch_hop_samples"] // self.architecture["cwt_pool_samples"]

    def alignment(self):
        a = self.architecture
        starts = torch.arange(self.patch_count) * a["patch_hop_samples"]
        slots = torch.arange(self.bins_per_patch) * a["cwt_pool_samples"]
        return {"sample_starts": starts, "cwt_columns": starts[:, None] // a["cwt_pool_samples"] + torch.arange(self.bins_per_patch),
                "patch_centers_seconds": (starts + (a["patch_samples"] - 1) / 2) / a["sample_rate"],
                "cwt_centers_seconds": (starts[:, None] + slots + (a["cwt_pool_samples"] - 1) / 2) / a["sample_rate"],
                "relative_seconds": (slots + a["cwt_pool_samples"] / 2 - a["patch_samples"] / 2) / a["sample_rate"]}


def require_finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} 包含 NaN/Inf (dtype={value.dtype}, shape={tuple(value.shape)})")


def frequency_tensor(frequencies: Sequence[float]):
    value = torch.as_tensor(frequencies, dtype=torch.float64).detach().clone()
    if value.shape != (97,):
        raise ValueError("frequencies_hz 需要 97 个实际频率")
    require_finite(value, "frequencies_hz")
    if not bool((value > 0).all()) or not bool((value[1:] >= value[:-1]).all()):
        raise ValueError("frequencies_hz 必须为正值且按输入行顺序非降序排列")
    return value


def environment_versions():
    try:
        mamba = version("mamba-ssm")
    except PackageNotFoundError:
        mamba = None
    return {"torch": str(torch.__version__), "cuda": torch.version.cuda, "mamba_ssm": mamba}


def local_cells(features, bins=2, step=2):
    """返回 B,N,F,R,C 视图；扁平单元下标为 f*R+r。"""
    return features.unfold(-1, bins, step).permute(0, 3, 2, 4, 1)


def concatenate_patches(patches):
    return patches.reshape(patches.shape[0], 1, -1).float()


class AlignedCrossAttention(nn.Module):
    def __init__(self, config: BCGCWTConfig):
        super().__init__()
        a = config.to_dict()
        self.bins, self.step = config.bins_per_patch, config.step_tf
        self.chunk_size = a["fusion"]["patch_chunk_size"]
        d, c, f = a["dimension"], a["tf_encoder"]["channels"], a["fusion"]
        self.frequency_embedding = nn.Parameter(torch.empty(a["cwt_frequency_count"], c))
        self.local_time_embedding = nn.Parameter(torch.empty(self.bins, c))
        nn.init.normal_(self.frequency_embedding, std=a["tf_position"]["initialization_std"])
        nn.init.normal_(self.local_time_embedding, std=a["tf_position"]["initialization_std"])
        self.query_norm = nn.LayerNorm(d, eps=f["norm_eps"])
        self.kv_norm = nn.LayerNorm(c, eps=f["norm_eps"])
        self.ffn_norm = nn.LayerNorm(d, eps=f["norm_eps"])
        self.mha = nn.MultiheadAttention(d, f["heads"], kdim=c, vdim=c,
                                         dropout=f["attention_dropout"], bias=f["bias"], batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, f["ffn_hidden_dimension"]), nn.GELU(),
                                 nn.Dropout(f["ffn_dropout"]), nn.Linear(f["ffn_hidden_dimension"], d))
        self.attention_output_dropout = nn.Dropout(f["residual_dropout"])
        self.ffn_output_dropout = nn.Dropout(f["residual_dropout"])

    def positioned_cells(self, features):
        return (local_cells(features, self.bins, self.step)
                + self.frequency_embedding[None, None, :, None, :]
                + self.local_time_embedding[None, None, None, :, :])

    def forward(self, tokens, features, *, return_attention=False):
        b, n, d = tokens.shape
        contexts, weights = [], []
        for start in range(0, n, self.chunk_size):
            stop = min(start + self.chunk_size, n)
            # CNN 已在整窗执行；这里只组织当前片段的完整 F×R 集合。
            grid = features[..., start * self.step:(stop - 1) * self.step + self.bins]
            cells = self.positioned_cells(grid)
            q = self.query_norm(tokens[:, start:stop]).reshape(b * (stop - start), 1, d)
            kv = self.kv_norm(cells).reshape(b * (stop - start), -1, cells.shape[-1])
            context, attention = self.mha(q, kv, kv, need_weights=return_attention,
                                          average_attn_weights=False)
            contexts.append(context.reshape(b, stop - start, d))
            if return_attention:
                weights.append(attention.reshape(b, stop - start, self.mha.num_heads, cells.shape[2], self.bins))
        # 随机残差层和 FFN 在分块合并后调用，使其调用位置固定。
        u = tokens + self.attention_output_dropout(torch.cat(contexts, dim=1))
        fused = u + self.ffn_output_dropout(self.ffn(self.ffn_norm(u)))
        return fused, torch.cat(weights, dim=1) if return_attention else None


class BCGCWTPatchMamba(nn.Module):
    def __init__(self, frequencies_hz, *, config: BCGCWTConfig | None = None,
                 initialization_seed: int = 20261013, mamba_factory: Callable[..., nn.Module] | None = None):
        super().__init__()
        self.config = BCGCWTConfig((config or BCGCWTConfig()).to_dict())
        if type(initialization_seed) is not int or not 0 <= initialization_seed < 2**63:
            raise ValueError("initialization_seed 必须为 [0,2**63) 的整数")
        self.initialization_seed = initialization_seed
        self.official_mamba = mamba_factory is None
        self.register_buffer("frequencies_hz", frequency_tensor(frequencies_hz))
        self.register_buffer("relative_seconds", self.config.alignment()["relative_seconds"])
        a = self.config.to_dict()
        w, t, m = a["waveform_encoder"], a["tf_encoder"], a["temporal"]
        d, c1, c2, ct = a["dimension"], w["stage1_channels"], w["stage2_channels"], t["channels"]
        # 一次构造种子覆盖全模型；退出后恢复调用者的 CPU RNG。
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(initialization_seed)
            self.waveform_encoder = nn.Sequential(
                nn.Conv1d(1, c1, w["stage1_kernel"], padding=w["stage1_kernel"] // 2, padding_mode="reflect"),
                nn.GELU(), LocalResidual(c1, w["stage1_residual_kernel"]),
                nn.Conv1d(c1, c2, w["stage2_kernel"], padding=w["stage2_kernel"] // 2, padding_mode="reflect"),
                nn.GELU(), LocalResidual(c2, w["stage2_residual_kernel"]),
                *[LocalResidual(c2, w["context_kernel"], dilation) for dilation in w["context_dilations"]])
            self.patch_embedding = nn.Conv1d(c2, d, a["patch_samples"], stride=a["patch_hop_samples"], padding=0)
            self.tf_encoder = nn.Sequential(
                ScaleTimePad(), nn.Conv2d(1, ct, 3), nn.GroupNorm(t["norm_groups"], ct, eps=t["norm_eps"]), nn.GELU(),
                ScaleTimePad(), nn.Conv2d(ct, ct, 3, groups=ct), nn.GELU(), nn.Conv2d(ct, ct, 1))
            self.fusion = AlignedCrossAttention(self.config)
            ssm = {key: value for key, value in m["mamba2"].items()
                   if key not in {"rmsnorm", "bias", "conv_bias", "use_mem_eff_path"}}
            self.temporal = nn.Sequential(*[BidirectionalMamba2Block(d, **ssm,
                dropout=m["residual_dropout"], mamba_factory=mamba_factory) for _ in range(m["layers"])])
            self.patch_decoder = nn.Sequential(nn.Linear(d, a["decoder"]["hidden_dimension"]), nn.GELU(),
                                                nn.Linear(a["decoder"]["hidden_dimension"], a["patch_samples"]))
        self.register_load_state_dict_pre_hook(self._check_checkpoint)

    def get_extra_state(self):
        return {"model_name": MODEL_NAME, "architecture": self.config.to_dict(),
                "initialization_seed": self.initialization_seed, "frequencies_hz": self.frequencies_hz.cpu().tolist(),
                "official_mamba": self.official_mamba, "environment": environment_versions()}

    def set_extra_state(self, state):
        # pre-hook 已检查身份；保存原始构造种子供下一次 checkpoint 使用。
        self.initialization_seed = state["initialization_seed"]

    def _check_checkpoint(self, module, state_dict, prefix, *args):
        """在任何参数/buffer 被覆盖前核对结构与数据频率身份。"""
        extra = state_dict.get(prefix + "_extra_state")
        if not isinstance(extra, dict) or extra.get("model_name") != MODEL_NAME:
            raise ValueError("checkpoint 缺少 BCG-CWT 模型身份")
        if extra.get("architecture") != self.config.to_dict():
            raise ValueError("checkpoint architecture/时间几何与当前模型不兼容")
        if extra.get("official_mamba") != self.official_mamba:
            raise ValueError("checkpoint Mamba2 实现身份不兼容")
        for name, expected in (("frequencies_hz", self.frequencies_hz), ("relative_seconds", self.relative_seconds)):
            value = state_dict.get(prefix + name)
            if not isinstance(value, torch.Tensor) or not torch.equal(value.cpu(), expected.cpu()):
                raise ValueError(f"checkpoint {name} 与数据元信息不兼容")
        if extra.get("frequencies_hz") != self.frequencies_hz.cpu().tolist():
            raise ValueError("checkpoint 频率元信息与 buffer 不一致")
        for name, value in state_dict.items():
            if name.startswith(prefix) and isinstance(value, torch.Tensor):
                require_finite(value, f"checkpoint.{name}")

    def forward(self, x, *, tf: Mapping[str, torch.Tensor], return_attention=False, epoch=None):
        a = self.config.architecture
        if not isinstance(tf, Mapping) or "w" not in tf:
            raise ValueError("tf 必须包含 w")
        w = tf["w"]
        for name, value, tail in (("x", x, (1, a["window_samples"])),
                                   ("tf['w']", w, (a["cwt_frequency_count"], a["window_samples"] // a["cwt_pool_samples"]))):
            if not isinstance(value, torch.Tensor) or value.ndim != 3 or tuple(value.shape[1:]) != tail or value.shape[0] < 1:
                raise ValueError(f"{name} 期望 [B,{tail[0]},{tail[1]}] 且 B>0")
            if not value.is_floating_point():
                raise TypeError(f"{name} 必须是浮点 tensor")
            if value.device != self.patch_embedding.weight.device:
                raise ValueError(f"{name} 与模型 device 不一致")
            require_finite(value, name)
        if x.shape[0] != w.shape[0]:
            raise ValueError("x 与 tf['w'] batch 不一致")
        if x.dtype != w.dtype:
            raise TypeError("x 与 tf['w'] dtype 不一致")
        if not torch.is_autocast_enabled(x.device.type) and x.dtype != self.patch_embedding.weight.dtype:
            raise TypeError("x/tf['w'] dtype 必须与模型一致，或在 autocast 中运行")
        wave_features = self.waveform_encoder(x)
        tf_features = self.tf_encoder(w[:, None])
        require_finite(wave_features, "waveform_encoder")
        require_finite(tf_features, "tf_encoder")
        tokens = self.patch_embedding(wave_features).transpose(1, 2)
        fused, attention = self.fusion(tokens, tf_features, return_attention=return_attention)
        require_finite(fused, "fusion")
        latent = self.temporal(fused)
        require_finite(latent, "temporal")
        prediction = concatenate_patches(self.patch_decoder(latent))
        require_finite(prediction, "waveform")
        result = {"waveform": prediction}
        if return_attention:
            result["attention"] = attention  # B,N,heads,F,R
        return result


def build_bcg_cwt_patch_mamba(cfg, *, mamba_factory=None):
    from omegaconf import OmegaConf
    raw = OmegaConf.to_container(cfg.model, resolve=True) if OmegaConf.is_config(cfg.model) else dict(cfg.model)
    if raw.get("name", MODEL_NAME) != MODEL_NAME:
        raise ValueError(f"model.name 必须为 {MODEL_NAME}")
    unknown = set(raw) - {"name", "initialization_seed", "architecture", "frequencies_hz", "tf_representations"}
    if unknown:
        raise ValueError(f"model 未知字段: {sorted(unknown)}")
    if raw.get("tf_representations", ["w"]) != ["w"]:
        raise ValueError("model.tf_representations 必须为 [w]")
    config = BCGCWTConfig(raw.get("architecture", {}))
    if "window" in cfg and (cfg.window.target_fs, cfg.window.duration_samples) != (100, 18000):
        raise ValueError("window.target_fs/duration_samples 与 architecture 时间几何冲突")
    frequencies = raw.get("frequencies_hz")
    cache = cfg.get("data", {}).get("tf_cache_path")
    if cache:
        from resp_train.data.bcg_cwt import frozen_w_metadata
        actual = frozen_w_metadata(cache)
        if frequencies is not None and not torch.equal(frequency_tensor(frequencies), frequency_tensor(actual)):
            raise ValueError("model.frequencies_hz 与 data.tf_cache_path 频率身份冲突")
        frequencies = actual
    if frequencies is None:
        raise ValueError("需要数据实际 model.frequencies_hz 或 data.tf_cache_path；合成测试须显式提供测试网格")
    return BCGCWTPatchMamba(frequencies, config=config,
                           initialization_seed=raw.get("initialization_seed", 20261013), mamba_factory=mamba_factory)
