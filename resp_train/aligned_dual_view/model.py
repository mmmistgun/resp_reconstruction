from __future__ import annotations

import importlib.metadata
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from resp_train.crd.blocks import BidirectionalMamba2Block
from resp_train.crd.initialization import module_seed
from resp_train.crd.spectral_ops import fourier_interpolate

from .features import CWTBatch, SCALE_COUNT, representation_spec, spec_digest
from .signal import LATENT_LENGTH, WINDOW_SAMPLES, CenteredFIRDecimate10, require_finite


@dataclass(frozen=True)
class ModelConfig:
    input_view: str = "joint"
    dimension: int = 64
    depth: int = 6
    waveform_kernel: int = 51
    initialization_seed: int = 20260811

    def __post_init__(self) -> None:
        if self.input_view not in {"joint", "waveform", "cwt", "joint_scale_mean"}:
            raise ValueError(f"未知 input_view={self.input_view!r}")
        for name in ("dimension", "depth", "waveform_kernel", "initialization_seed"):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} 必须为整数")
        if self.dimension < 32 or self.dimension % 32:
            raise ValueError("dimension 必须为至少 32 的 32 倍数")
        if self.depth < 1 or self.initialization_seed < 0:
            raise ValueError("depth 必须为正，initialization_seed 必须非负")
        if self.waveform_kernel < 1 or self.waveform_kernel % 2 != 1 or self.waveform_kernel >= WINDOW_SAMPLES:
            raise ValueError("waveform_kernel 必须为小于输入长度的正奇数")

    @property
    def architecture_id(self) -> str:
        values = asdict(self)
        values.pop("initialization_seed")
        return spec_digest({"name": "aligned-dual-view-v1", "version": 1, **values})


def load_model_config(path: str | Path) -> ModelConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(payload) != {"schema", "model"} or payload["schema"] != "aligned-dual-view-v1-model-v1":
        raise ValueError("ADV-v1 model config schema 不匹配")
    return ModelConfig(**payload["model"])


class AlignedDualViewV1(nn.Module):
    """可学习网络返回 100 Hz 波形；任务 Pi 留在既有 loss/evaluator 中。"""

    def __init__(
        self, config: ModelConfig | None = None, *,
        mamba_factory: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if mamba_factory is None:
            for name, expected in (("mamba-ssm", "2.3.2.post1"), ("causal-conv1d", "1.6.2.post1")):
                actual = importlib.metadata.version(name)
                if actual != expected:
                    raise RuntimeError(f"ADV-v1 要求 {name}=={expected}，当前 {actual}")
        self.config = config or ModelConfig()
        cfg = self.config
        self.uses_waveform = cfg.input_view != "cwt"
        self.uses_cwt = cfg.input_view != "waveform"
        joint = self.uses_waveform and self.uses_cwt
        view_channels = cfg.dimension // 2 if joint else cfg.dimension
        self.representation_id = representation_spec()["representation_id"]
        self.downsample = CenteredFIRDecimate10()
        # 独立子 seed 避免可选分支改变共享主干/读出的初始化。
        if self.uses_waveform:
            with module_seed(cfg.initialization_seed, "adv-v1.waveform"):
                self.waveform_encoder = nn.Sequential(
                    nn.Conv1d(1, view_channels, kernel_size=cfg.waveform_kernel,
                              padding=cfg.waveform_kernel // 2, padding_mode="reflect", bias=True),
                    nn.SiLU(),
                )
        if self.uses_cwt:
            with module_seed(cfg.initialization_seed, "adv-v1.cwt"):
                self.scale_projection = nn.Sequential(
                    nn.Conv1d(1 if cfg.input_view == "joint_scale_mean" else SCALE_COUNT,
                              view_channels, kernel_size=1, bias=True),
                    nn.SiLU(),
                )
        with module_seed(cfg.initialization_seed, "adv-v1.backbone"):
            self.blocks = nn.ModuleList([
                BidirectionalMamba2Block(cfg.dimension, d_state=64, d_conv=4, expand=2,
                                         headdim=32, ngroups=1, chunk_size=256, dropout=0.0,
                                         mamba_factory=mamba_factory)
                for _ in range(cfg.depth)
            ])
        with module_seed(cfg.initialization_seed, "adv-v1.readout"):
            self.readout = nn.Conv1d(cfg.dimension, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor, *, cwt: CWTBatch | None = None) -> dict[str, torch.Tensor]:
        if x.ndim != 3 or x.shape[1:] != (1, WINDOW_SAMPLES) or not len(x):
            raise ValueError("ADV-v1 输入期望非空 (B,1,18000)")
        if not x.is_floating_point():
            raise TypeError("ADV-v1 输入必须为浮点 tensor")
        require_finite("输入", x)
        views = []
        if self.uses_waveform:
            views.append(self.downsample(self.waveform_encoder(x)))
        if self.uses_cwt:
            if not isinstance(cwt, CWTBatch) or cwt.representation_id != self.representation_id:
                raise ValueError("必须传入表示身份匹配的 ADV-v1 CWTBatch")
            values = cwt.values
            if values.shape != (x.shape[0], SCALE_COUNT, LATENT_LENGTH):
                raise ValueError("ADV-v1 CWT 期望 (B,97,1800)，不能使用旧 2 Hz 特征")
            if values.device != x.device or values.dtype != torch.float32:
                raise ValueError("ADV-v1 CWT 必须与输入同设备且为 float32")
            require_finite("CWT 输入", values)
            if self.config.input_view == "joint_scale_mean":
                values = values.mean(dim=1, keepdim=True)
            views.append(self.scale_projection(values))
        elif cwt is not None:
            raise ValueError("waveform 单视图不接收 CWTBatch")
        latent = torch.cat(views, dim=1).transpose(1, 2).contiguous()
        for block in self.blocks:
            latent = block(latent)
        waveform_10hz = self.readout(latent.transpose(1, 2))
        require_finite("10 Hz prediction", waveform_10hz)
        waveform = fourier_interpolate(waveform_10hz, target_length=WINDOW_SAMPLES)
        require_finite("100 Hz prediction", waveform)
        return {"waveform": waveform, "waveform_10hz": waveform_10hz}
