# M9 低自由度模型结构实验实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 固定已收口的 signed direction loss，新增并比较 5 个带低频归纳偏置的模型结构，验证是否能在方向正确的前提下降低局部尖峰自由度和 `rr_peak_band_abs_error`。

**架构：** M9 不继续扩大 loss 网格，而是在 `resp_train.models.lowfreq` 中新增低频模型组：低频 basis decoder、多尺度分解 mixer、TimesNet-lite 周期块、frequency bottleneck、downsampled SSM-like decoder。所有模型统一接入现有 registry、训练入口、评价与汇总脚本，先跑 2 个 seed 试验，通过后再扩 seed。

**技术栈：** Python、PyTorch、OmegaConf、pandas、pytest、现有 `resp_train` 训练与 `summarize_tho_runs.py` 汇总脚本。M9 首轮不引入新第三方依赖；true Mamba 只作为后续可选扩展，需要单独确认安装。

---

## 背景与固定口径

当前阶段结论：

- Loss 已阶段性收口，默认候选为 `high_freq_weight=0.2`、`relative_envelope_weight=0.03`、`phase_alignment_weight=0.0`、`signed_corr_weight=0.1`。
- checkpoint direction gate 保留：`training.checkpoint_gate.metric=auto_direction`，`training.checkpoint_gate.max=0.5`。
- `signed_corr_weight=0.1` 能稳定解决方向，但会带来局部峰值和 raw peak 代价。
- 下一阶段主问题是模型输出自由度过高，不是继续发明新 loss。

M9 固定训练口径：

- 输入：`bcg_rawish_wideband_state_aligned`
- 数据窗口：全量，`data.max_train_windows=null`，`data.max_val_windows=null`
- 数据 seed：`data.train_sample_seed=20260610`，`data.val_sample_seed=20260611`
- Loss：
  - `loss.high_freq_weight=0.2`
  - `loss.relative_envelope_weight=0.03`
  - `loss.phase_alignment_weight=0.0`
  - `loss.signed_corr_weight=0.1`
  - `loss.signed_cosine_weight=0.0`
  - `loss.signed_cosine_schedule.mode=none`
- 训练：
  - `training.epochs=50`
  - `training.batch_size=128`
  - `training.patience=8`
  - `training.min_delta=0.001`
  - `training.use_amp=false`
  - `training.show_progress=false`
  - `training.checkpoint_gate.metric=auto_direction`
  - `training.checkpoint_gate.max=0.5`
- 输出：`runs/tho_research_v2_model_m9_lowfreq_structure`
- 汇总：`runs/tho_research_v2_model_m9_lowfreq_structure_summary.csv`

M9 首轮 seed：

- 困难 seed：`20260710`
- guard seed：`20260700`

通过条件：

1. `auto_direction` 对应的验证方向分项不超过 `0.5`，并且 `band_limited_corr_mean > 0`。
2. 主指标 `selection_task_rr_peak_band_abs_error_mean` 不明显劣于同 seed Patch-Hann baseline。
3. `selection_task_rr_spec_abs_error_mean` 不出现明显频谱 RR 退化。
4. `selection_task_relative_envelope_mae_mean` 不明显劣于 baseline。
5. raw `selection_waveform_rr_peak_abs_error_mean` 不能出现灾难性尖峰；该列只做诊断，不作为单独选模主指标。

## 依赖策略

M9 首轮不需要安装新依赖：

- `basis_decoder1d`：用 PyTorch buffer 构造 Fourier/DCT-like basis，不依赖 `torch-dct`。
- `multiscale_decomp_mixer1d`：用 `avg_pool1d`、`interpolate`、`Conv1d` 和 existing moving average。
- `timesnet_lite1d`：用 `torch.fft.rfft` 估计低频候选周期，再用轻量 2D convolution。
- `frequency_bottleneck1d`：用 `torch.fft.irfft` 从低频 bins 回到时域；首轮强制 `training.use_amp=false`，避免半精度 FFT 约束。
- `downsampled_ssm1d`：用纯 PyTorch gated recurrent/state block，不使用 `mamba-ssm`。

可选依赖：

- 若 `downsampled_ssm1d` 首轮表现接近通过，但长程建模不足，再单独评估安装 `mamba-ssm` 或 `causal-conv1d`。安装需要 CUDA/PyTorch ABI 匹配，属于单独决策，不纳入本计划默认执行。

## 文件结构

- 创建：`resp_train/models/lowfreq.py`
  - 职责：集中实现 M9 新模型与共享低频 helper，避免继续膨胀 `timeseries.py`。
- 修改：`resp_train/models/registry.py`
  - 职责：注册 `basis_decoder1d`、`multiscale_decomp_mixer1d`、`timesnet_lite1d`、`frequency_bottleneck1d`、`downsampled_ssm1d`。
- 修改：`resp_train/models/__init__.py`
  - 职责：保持现有 `build_model` / `list_models` 导出方式不变；只有需要显式导出新类时才更新。
- 修改：`tests/test_model_registry.py`
  - 职责：覆盖 5 个新模型的注册、shape、关键结构约束。
- 本地生成：`runs/tho_research_v2_model_m9_lowfreq_structure/`
  - 职责：保存正式训练产物，不进入 Git。
- 本地生成：`runs/tho_research_v2_model_m9_lowfreq_structure_summary.csv`
  - 职责：保存汇总表，不进入 Git。
- 修改：`docs/experiments/rawish_state_aligned_l0_l1.md`
  - 职责：追加 M9 结果、对照表和阶段判断。

## 任务 1：新增 M9 模型注册测试

**文件：**
- 修改：`tests/test_model_registry.py`

- [ ] **步骤 1：编写失败的注册测试**

在 `tests/test_model_registry.py` 增加：

```python
@pytest.mark.parametrize(
    "model_name",
    [
        "basis_decoder1d",
        "multiscale_decomp_mixer1d",
        "timesnet_lite1d",
        "frequency_bottleneck1d",
        "downsampled_ssm1d",
    ],
)
def test_lowfreq_structure_models_are_registered(model_name):
    assert model_name in list_models()
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
./.venv/bin/python -m pytest tests/test_model_registry.py::test_lowfreq_structure_models_are_registered -v
```

预期：失败，提示至少一个 M9 模型名不在 registry。

- [ ] **步骤 3：编写 shape 参数化测试**

在 `tests/test_model_registry.py` 增加：

```python
@pytest.mark.parametrize(
    "model_name,extra",
    [
        ("basis_decoder1d", {"basis_count": 64, "encoder_stride": 20}),
        ("multiscale_decomp_mixer1d", {"downsample_factors": [1, 4, 16], "mixer_layers": 2}),
        ("timesnet_lite1d", {"period_top_k": 3, "period_min_sec": 2.0, "period_max_sec": 20.0}),
        ("frequency_bottleneck1d", {"max_freq_hz": 0.7, "freq_bins": 128}),
        ("downsampled_ssm1d", {"latent_stride": 20, "state_layers": 2}),
    ],
)
def test_lowfreq_structure_models_preserve_waveform_shape(model_name, extra):
    cfg = OmegaConf.create(
        {
            "window": {"target_fs": 100, "duration_samples": 1800},
            "model": {
                "name": model_name,
                "in_channels": 1,
                "out_channels": 1,
                "base_channels": 8,
                **extra,
            },
        }
    )
    model = build_model(cfg)

    y = model(torch.randn(2, 1, 1800))

    assert y.shape == (2, 1, 1800)
    assert torch.isfinite(y).all()
```

- [ ] **步骤 4：运行 shape 测试验证失败**

运行：

```bash
./.venv/bin/python -m pytest tests/test_model_registry.py::test_lowfreq_structure_models_preserve_waveform_shape -v
```

预期：失败，提示模型未注册或类未定义。

## 任务 2：实现共享低频 helper

**文件：**
- 创建：`resp_train/models/lowfreq.py`

- [ ] **步骤 1：创建 helper 骨架**

新增 `resp_train/models/lowfreq.py`：

```python
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def _match_length(x: torch.Tensor, length: int) -> torch.Tensor:
    diff = int(length) - x.size(-1)
    if diff > 0:
        return F.pad(x, (0, diff))
    if diff < 0:
        return x[..., :length]
    return x


def _odd_kernel(value: int, *, minimum: int = 3) -> int:
    value = max(int(value), minimum)
    return value if value % 2 == 1 else value + 1


class ChannelNorm1D(nn.Module):
    """按窗口做可逆标准化，减少不同片段幅值尺度影响。"""

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(self.eps)
        return (x - mean) / std, mean, std


class DepthwiseMovingAverage1D(nn.Module):
    """每个通道独立移动平均，用于低频趋势提取。"""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        kernel_size = _odd_kernel(kernel_size)
        self.padding = kernel_size // 2
        weight = torch.full((int(channels), 1, kernel_size), 1.0 / kernel_size)
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padded = F.pad(x, (self.padding, self.padding), mode="reflect")
        return F.conv1d(padded, self.weight, groups=x.size(1))
```

- [ ] **步骤 2：运行静态编译**

运行：

```bash
./.venv/bin/python -m py_compile resp_train/models/lowfreq.py
```

预期：无输出，退出码为 0。

## 任务 3：实现 `basis_decoder1d`

**文件：**
- 修改：`resp_train/models/lowfreq.py`
- 修改：`resp_train/models/registry.py`
- 测试：`tests/test_model_registry.py`

- [ ] **步骤 1：实现低频 basis decoder**

在 `resp_train/models/lowfreq.py` 增加：

```python
class BasisDecoder1D(nn.Module):
    """用少量低频基函数重建整段呼吸轨迹，显式限制输出自由度。"""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        basis_count: int = 96,
        encoder_stride: int = 20,
        duration_samples: int = 18000,
    ) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        self.basis_count = max(int(basis_count), 8)
        self.duration_samples = int(duration_samples)
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=9, stride=int(encoder_stride), padding=4),
            nn.GroupNorm(1, base_channels),
            nn.SiLU(),
            nn.Conv1d(base_channels, base_channels, kernel_size=9, padding=4),
            nn.GroupNorm(1, base_channels),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.coeff_head = nn.Linear(base_channels, out_channels * self.basis_count)
        basis = self._build_basis(self.duration_samples, self.basis_count)
        self.register_buffer("basis", basis, persistent=False)

    @staticmethod
    def _build_basis(length: int, basis_count: int) -> torch.Tensor:
        t = torch.linspace(0.0, 1.0, int(length), dtype=torch.float32)
        cols = [torch.ones_like(t)]
        max_harmonics = max((int(basis_count) - 1) // 2, 1)
        for k in range(1, max_harmonics + 1):
            cols.append(torch.sin(2.0 * math.pi * k * t))
            if len(cols) >= basis_count:
                break
            cols.append(torch.cos(2.0 * math.pi * k * t))
            if len(cols) >= basis_count:
                break
        basis = torch.stack(cols[:basis_count], dim=0)
        return basis / basis.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(-1)
        features = self.pool(self.encoder(x)).squeeze(-1)
        coeff = self.coeff_head(features).view(x.size(0), self.out_channels, self.basis_count)
        basis = self.basis.to(device=x.device, dtype=x.dtype)
        if basis.size(-1) != length:
            basis = F.interpolate(basis.unsqueeze(0), size=length, mode="linear", align_corners=False).squeeze(0)
        return torch.einsum("bck,kl->bcl", coeff, basis)
```

- [ ] **步骤 2：注册 `basis_decoder1d`**

在 `resp_train/models/registry.py` 导入并注册：

```python
from resp_train.models.lowfreq import BasisDecoder1D

"basis_decoder1d": lambda cfg: BasisDecoder1D(
    in_channels=int(cfg.model.in_channels),
    out_channels=int(cfg.model.out_channels),
    base_channels=int(cfg.model.base_channels),
    basis_count=int(cfg.model.get("basis_count", 96)),
    encoder_stride=int(cfg.model.get("encoder_stride", 20)),
    duration_samples=int(cfg.window.get("duration_samples", 18000)),
),
```

- [ ] **步骤 3：运行对应测试**

运行：

```bash
./.venv/bin/python -m pytest tests/test_model_registry.py::test_lowfreq_structure_models_are_registered tests/test_model_registry.py::test_lowfreq_structure_models_preserve_waveform_shape -v
```

预期：`basis_decoder1d` 相关断言通过，其它未实现模型仍失败。

## 任务 4：实现 `multiscale_decomp_mixer1d`

**文件：**
- 修改：`resp_train/models/lowfreq.py`
- 修改：`resp_train/models/registry.py`
- 测试：`tests/test_model_registry.py`

- [ ] **步骤 1：实现多尺度分解 mixer**

在 `resp_train/models/lowfreq.py` 增加：

```python
class MultiScaleDecompMixer1D(nn.Module):
    """TimeMixer/MICN-lite：多尺度低频分解后融合输出。"""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        downsample_factors: list[int] | tuple[int, ...] = (1, 4, 16),
        mixer_layers: int = 2,
    ) -> None:
        super().__init__()
        self.factors = tuple(int(v) for v in downsample_factors)
        self.branches = nn.ModuleList()
        for factor in self.factors:
            layers: list[nn.Module] = [
                nn.Conv1d(in_channels, base_channels, kernel_size=9, padding=4),
                nn.GroupNorm(1, base_channels),
                nn.SiLU(),
            ]
            for _ in range(max(int(mixer_layers), 1)):
                layers.extend(
                    [
                        nn.Conv1d(base_channels, base_channels, kernel_size=7, padding=3, groups=base_channels),
                        nn.Conv1d(base_channels, base_channels, kernel_size=1),
                        nn.GroupNorm(1, base_channels),
                        nn.SiLU(),
                    ]
                )
            self.branches.append(nn.Sequential(*layers))
        self.fuse = nn.Sequential(
            nn.Conv1d(base_channels * len(self.factors), base_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(base_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(-1)
        outputs = []
        for factor, branch in zip(self.factors, self.branches):
            if factor > 1:
                pooled = F.avg_pool1d(x, kernel_size=factor, stride=factor, ceil_mode=True)
            else:
                pooled = x
            y = branch(pooled)
            if y.size(-1) != length:
                y = F.interpolate(y, size=length, mode="linear", align_corners=False)
            outputs.append(y)
        return self.fuse(torch.cat(outputs, dim=1))
```

- [ ] **步骤 2：注册 `multiscale_decomp_mixer1d`**

在 `resp_train/models/registry.py` 增加：

```python
"multiscale_decomp_mixer1d": lambda cfg: MultiScaleDecompMixer1D(
    in_channels=int(cfg.model.in_channels),
    out_channels=int(cfg.model.out_channels),
    base_channels=int(cfg.model.base_channels),
    downsample_factors=list(cfg.model.get("downsample_factors", [1, 4, 16])),
    mixer_layers=int(cfg.model.get("mixer_layers", 2)),
),
```

- [ ] **步骤 3：运行对应测试**

运行：

```bash
./.venv/bin/python -m pytest tests/test_model_registry.py::test_lowfreq_structure_models_preserve_waveform_shape -v
```

预期：`basis_decoder1d` 与 `multiscale_decomp_mixer1d` 相关断言通过，其它未实现模型仍失败。

## 任务 5：实现 `timesnet_lite1d`

**文件：**
- 修改：`resp_train/models/lowfreq.py`
- 修改：`resp_train/models/registry.py`
- 测试：`tests/test_model_registry.py`

- [ ] **步骤 1：实现 TimesNet-lite 周期块**

在 `resp_train/models/lowfreq.py` 增加：

```python
class TimesNetLite1D(nn.Module):
    """在低频表示上做周期重排和 2D 卷积，避免直接让 BCG 高频主导周期选择。"""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        period_top_k: int = 3,
        period_min_sec: float = 2.0,
        period_max_sec: float = 20.0,
        sample_rate: float = 100.0,
        lowpass_kernel: int = 401,
    ) -> None:
        super().__init__()
        self.period_top_k = int(period_top_k)
        self.min_period = max(int(period_min_sec * sample_rate), 1)
        self.max_period = max(int(period_max_sec * sample_rate), self.min_period)
        self.lowpass = DepthwiseMovingAverage1D(in_channels, lowpass_kernel)
        self.embed = nn.Conv1d(in_channels, base_channels, kernel_size=1)
        self.period_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=(3, 5), padding=(1, 2), groups=base_channels),
            nn.Conv2d(base_channels, base_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=(3, 5), padding=(1, 2), groups=base_channels),
            nn.Conv2d(base_channels, base_channels, kernel_size=1),
        )
        self.out = nn.Conv1d(base_channels, out_channels, kernel_size=1)

    def _periods(self, x: torch.Tensor) -> list[int]:
        spectrum = torch.fft.rfft(x.float(), dim=-1).abs().mean(dim=(0, 1))
        spectrum[0] = 0
        length = x.size(-1)
        min_bin = max(length // self.max_period, 1)
        max_bin = min(length // self.min_period, spectrum.numel() - 1)
        masked = torch.zeros_like(spectrum)
        masked[min_bin : max_bin + 1] = spectrum[min_bin : max_bin + 1]
        top = torch.topk(masked, k=min(self.period_top_k, max_bin - min_bin + 1)).indices
        periods = [max(length // int(idx.item()), 1) for idx in top if int(idx.item()) > 0]
        return periods or [self.min_period]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(-1)
        z = self.embed(self.lowpass(x))
        outputs = []
        for period in self._periods(x):
            padded_len = math.ceil(length / period) * period
            z_pad = _match_length(z, padded_len)
            grid = z_pad.view(z.size(0), z.size(1), padded_len // period, period)
            y = self.period_conv(grid).reshape(z.size(0), z.size(1), padded_len)[..., :length]
            outputs.append(y)
        mixed = torch.stack(outputs, dim=0).mean(dim=0)
        return self.out(mixed + z)
```

- [ ] **步骤 2：注册 `timesnet_lite1d`**

在 `resp_train/models/registry.py` 增加：

```python
"timesnet_lite1d": lambda cfg: TimesNetLite1D(
    in_channels=int(cfg.model.in_channels),
    out_channels=int(cfg.model.out_channels),
    base_channels=int(cfg.model.base_channels),
    period_top_k=int(cfg.model.get("period_top_k", 3)),
    period_min_sec=float(cfg.model.get("period_min_sec", 2.0)),
    period_max_sec=float(cfg.model.get("period_max_sec", 20.0)),
    sample_rate=float(cfg.window.get("target_fs", 100)),
    lowpass_kernel=int(cfg.model.get("lowpass_kernel", 401)),
),
```

- [ ] **步骤 3：运行对应测试**

运行：

```bash
./.venv/bin/python -m pytest tests/test_model_registry.py::test_lowfreq_structure_models_preserve_waveform_shape -v
```

预期：前三个 M9 模型相关断言通过，未实现的 `frequency_bottleneck1d`、`downsampled_ssm1d` 仍失败。

## 任务 6：实现 `frequency_bottleneck1d`

**文件：**
- 修改：`resp_train/models/lowfreq.py`
- 修改：`resp_train/models/registry.py`
- 测试：`tests/test_model_registry.py`

- [ ] **步骤 1：实现低频频谱瓶颈**

在 `resp_train/models/lowfreq.py` 增加：

```python
class FrequencyBottleneck1D(nn.Module):
    """只预测低频 spectrum bins，再 irfft 回到时域，限制高频输出自由度。"""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        freq_bins: int = 128,
        max_freq_hz: float = 0.7,
        sample_rate: float = 100.0,
        duration_samples: int = 18000,
    ) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        self.duration_samples = int(duration_samples)
        max_bins = int(max_freq_hz * self.duration_samples / sample_rate) + 1
        self.freq_bins = max(2, min(int(freq_bins), max_bins))
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=17, stride=20, padding=8),
            nn.GroupNorm(1, base_channels),
            nn.SiLU(),
            nn.Conv1d(base_channels, base_channels, kernel_size=9, padding=4),
            nn.GroupNorm(1, base_channels),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.real_head = nn.Linear(base_channels, out_channels * self.freq_bins)
        self.imag_head = nn.Linear(base_channels, out_channels * self.freq_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(-1)
        features = self.encoder(x).squeeze(-1)
        real = self.real_head(features).view(x.size(0), self.out_channels, self.freq_bins)
        imag = self.imag_head(features).view(x.size(0), self.out_channels, self.freq_bins)
        imag[..., 0] = 0.0
        spectrum = torch.zeros(
            x.size(0),
            self.out_channels,
            length // 2 + 1,
            dtype=torch.complex64,
            device=x.device,
        )
        low = torch.complex(real.float(), imag.float())
        spectrum[..., : self.freq_bins] = low
        y = torch.fft.irfft(spectrum, n=length, dim=-1)
        return y.to(dtype=x.dtype)
```

- [ ] **步骤 2：注册 `frequency_bottleneck1d`**

在 `resp_train/models/registry.py` 增加：

```python
"frequency_bottleneck1d": lambda cfg: FrequencyBottleneck1D(
    in_channels=int(cfg.model.in_channels),
    out_channels=int(cfg.model.out_channels),
    base_channels=int(cfg.model.base_channels),
    freq_bins=int(cfg.model.get("freq_bins", 128)),
    max_freq_hz=float(cfg.model.get("max_freq_hz", 0.7)),
    sample_rate=float(cfg.window.get("target_fs", 100)),
    duration_samples=int(cfg.window.get("duration_samples", 18000)),
),
```

- [ ] **步骤 3：运行对应测试**

运行：

```bash
./.venv/bin/python -m pytest tests/test_model_registry.py::test_lowfreq_structure_models_preserve_waveform_shape -v
```

预期：除 `downsampled_ssm1d` 外，其它 M9 模型相关断言通过。

## 任务 7：实现 `downsampled_ssm1d`

**文件：**
- 修改：`resp_train/models/lowfreq.py`
- 修改：`resp_train/models/registry.py`
- 测试：`tests/test_model_registry.py`

- [ ] **步骤 1：实现纯 PyTorch 下采样状态模型**

在 `resp_train/models/lowfreq.py` 增加：

```python
class GatedStateBlock1D(nn.Module):
    """轻量 SSM-like 门控状态块，用卷积门控近似长程状态更新。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.filter = nn.Conv1d(channels, channels, kernel_size=9, padding=4, groups=channels)
        self.gate = nn.Conv1d(channels, channels, kernel_size=1)
        self.mix = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        update = torch.tanh(self.filter(z))
        gate = torch.sigmoid(self.gate(z))
        return x + self.mix(update * gate)


class DownsampledSSM1D(nn.Module):
    """先降采样到低频 latent，再用门控状态块建模，最后上采样输出。"""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        latent_stride: int = 20,
        state_layers: int = 2,
    ) -> None:
        super().__init__()
        self.latent_stride = max(int(latent_stride), 1)
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=9, padding=4),
            nn.GroupNorm(1, base_channels),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(*[GatedStateBlock1D(base_channels) for _ in range(max(int(state_layers), 1))])
        self.output_proj = nn.Sequential(
            nn.Conv1d(base_channels, base_channels, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(base_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(-1)
        if self.latent_stride > 1:
            z = F.avg_pool1d(x, kernel_size=self.latent_stride, stride=self.latent_stride, ceil_mode=True)
        else:
            z = x
        z = self.blocks(self.input_proj(z))
        z = F.interpolate(z, size=length, mode="linear", align_corners=False)
        return self.output_proj(z)
```

- [ ] **步骤 2：注册 `downsampled_ssm1d`**

在 `resp_train/models/registry.py` 增加：

```python
"downsampled_ssm1d": lambda cfg: DownsampledSSM1D(
    in_channels=int(cfg.model.in_channels),
    out_channels=int(cfg.model.out_channels),
    base_channels=int(cfg.model.base_channels),
    latent_stride=int(cfg.model.get("latent_stride", 20)),
    state_layers=int(cfg.model.get("state_layers", 2)),
),
```

- [ ] **步骤 3：运行 M9 模型测试**

运行：

```bash
./.venv/bin/python -m pytest tests/test_model_registry.py::test_lowfreq_structure_models_are_registered tests/test_model_registry.py::test_lowfreq_structure_models_preserve_waveform_shape -v
```

预期：新增 M9 相关测试全部通过。

## 任务 8：全量模型测试、编译与实现提交

**文件：**
- 修改：`resp_train/models/lowfreq.py`
- 修改：`resp_train/models/registry.py`
- 修改：`tests/test_model_registry.py`

- [ ] **步骤 1：运行模型测试**

运行：

```bash
./.venv/bin/python -m pytest tests/test_model_registry.py -q
```

预期：`tests/test_model_registry.py` 全部通过。

- [ ] **步骤 2：运行编译检查**

运行：

```bash
./.venv/bin/python -m py_compile resp_train/models/lowfreq.py resp_train/models/registry.py
```

预期：无输出，退出码为 0。

- [ ] **步骤 3：运行空白检查**

运行：

```bash
git diff --check
```

预期：无输出，退出码为 0。

- [ ] **步骤 4：Commit**

运行：

```bash
git add resp_train/models/lowfreq.py resp_train/models/registry.py tests/test_model_registry.py
git commit -m "feat: 添加 M9 低频结构模型候选"
```

预期：只提交模型实现与测试；`AGENTS.md` 如果仍有本地改动，不纳入提交。

## 任务 9：GPU smoke 训练检查

**文件：**
- 本地生成：`runs/tho_research_v2_model_m9_lowfreq_smoke/`

- [ ] **步骤 1：运行 5 个模型的一轮小样本 smoke**

逐个运行以下命令，替换 `<MODEL>` 和 `<EXTRA_SET>`：

```bash
./.venv/bin/python scripts/train_tho_small.py \
  --config configs/tho_research_v2.yaml \
  --set data.max_train_windows=32 \
  --set data.max_val_windows=16 \
  --set data.train_sample_seed=20260610 \
  --set data.val_sample_seed=20260611 \
  --set model.name=<MODEL> \
  <EXTRA_SET> \
  --set loss.high_freq_weight=0.2 \
  --set loss.relative_envelope_weight=0.03 \
  --set loss.phase_alignment_weight=0.0 \
  --set loss.signed_corr_weight=0.1 \
  --set loss.signed_cosine_weight=0.0 \
  --set loss.signed_cosine_schedule.mode=none \
  --set training.epochs=1 \
  --set training.batch_size=8 \
  --set training.patience=1 \
  --set training.use_amp=false \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --set outputs.run_root=runs/tho_research_v2_model_m9_lowfreq_smoke
```

模型参数：

```text
basis_decoder1d:
  --set model.basis_count=96 --set model.encoder_stride=20
multiscale_decomp_mixer1d:
  --set model.downsample_factors=[1,4,16] --set model.mixer_layers=2
timesnet_lite1d:
  --set model.period_top_k=3 --set model.period_min_sec=2.0 --set model.period_max_sec=20.0 --set model.lowpass_kernel=401
frequency_bottleneck1d:
  --set model.freq_bins=128 --set model.max_freq_hz=0.7
downsampled_ssm1d:
  --set model.latent_stride=20 --set model.state_layers=2
```

预期：每个模型都生成 `checkpoint.pt`、`metrics.csv` 和 `train_history.csv`，无 CUDA shape 或 FFT 报错。

- [ ] **步骤 2：汇总 smoke run**

运行：

```bash
./.venv/bin/python scripts/summarize_tho_runs.py \
  --runs-root runs/tho_research_v2_model_m9_lowfreq_smoke \
  --output /tmp/m9_lowfreq_smoke_summary.csv
```

预期：输出表有 5 行，且 `selection_task_rr_peak_band_abs_error_mean` 为有限数。

- [ ] **步骤 3：Commit smoke 结论**

若 smoke 发现需要代码修复，先修复并重复任务 8 与任务 9。若 smoke 无代码改动，只在实验文档新增一句 smoke 结果，再提交：

```bash
git add docs/experiments/rawish_state_aligned_l0_l1.md
git commit -m "exp: 记录 M9 模型 smoke 结果"
```

## 任务 10：正式训练第一批 2 seed

**文件：**
- 本地生成：`runs/tho_research_v2_model_m9_lowfreq_structure/`

- [ ] **步骤 1：运行 Patch-Hann signed baseline，seed 20260710**

```bash
./.venv/bin/python scripts/train_tho_small.py \
  --config configs/tho_research_v2.yaml \
  --set data.max_train_windows=null \
  --set data.max_val_windows=null \
  --set data.train_sample_seed=20260610 \
  --set data.val_sample_seed=20260611 \
  --set model.name=patch_mixer1d \
  --set model.patch_len=256 \
  --set model.patch_stride=128 \
  --set model.mixer_layers=2 \
  --set model.overlap_window=hann \
  --set model.output_smoothing_kernel=1 \
  --set loss.high_freq_weight=0.2 \
  --set loss.relative_envelope_weight=0.03 \
  --set loss.phase_alignment_weight=0.0 \
  --set loss.signed_corr_weight=0.1 \
  --set loss.signed_cosine_weight=0.0 \
  --set loss.signed_cosine_schedule.mode=none \
  --set training.seed=20260710 \
  --set training.epochs=50 \
  --set training.batch_size=128 \
  --set training.patience=8 \
  --set training.min_delta=0.001 \
  --set training.use_amp=false \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --set training.checkpoint_gate.metric=auto_direction \
  --set training.checkpoint_gate.max=0.5 \
  --set outputs.run_root=runs/tho_research_v2_model_m9_lowfreq_structure
```

预期：run 目录包含 `config.yaml`、`train_history.csv`、`metrics.csv`、`checkpoint.pt`。

- [ ] **步骤 2：运行 Patch-Hann signed baseline，seed 20260700**

重复步骤 1，仅修改：

```bash
--set training.seed=20260700
--set training.device=cuda:1
```

预期：run 目录完整。

- [ ] **步骤 3：运行 5 个 M9 模型，seed 20260710 与 20260700**

对以下模型分别运行两条训练命令，通用参数与步骤 1 相同，只替换 `model.name`、模型专属参数、`training.seed`、`training.device`：

```text
basis_decoder1d:
  --set model.basis_count=96 --set model.encoder_stride=20

multiscale_decomp_mixer1d:
  --set model.downsample_factors=[1,4,16] --set model.mixer_layers=2

timesnet_lite1d:
  --set model.period_top_k=3 --set model.period_min_sec=2.0 --set model.period_max_sec=20.0 --set model.lowpass_kernel=401

frequency_bottleneck1d:
  --set model.freq_bins=128 --set model.max_freq_hz=0.7

downsampled_ssm1d:
  --set model.latent_stride=20 --set model.state_layers=2
```

GPU 安排：

```text
cuda:0: seed 20260710 的候选
cuda:1: seed 20260700 的候选
```

预期：每个模型每个 seed 生成一个完整 run。若实际机器只有一张可用 GPU，则顺序执行同一批命令并统一使用 `cuda:0`。

- [ ] **步骤 4：每个正式模型完成后保存 git 状态**

如果训练过程没有代码或文档改动，不强行提交空 commit。每完成一个模型的 2 seed 后，至少追加实验进度到 `docs/experiments/rawish_state_aligned_l0_l1.md` 并提交：

```bash
git add docs/experiments/rawish_state_aligned_l0_l1.md
git commit -m "exp: 记录 M9 <模型名> 首轮结果"
```

预期：不会积压多个模型结论在同一个未提交工作区中。

## 任务 11：汇总与首轮筛选

**文件：**
- 本地生成：`runs/tho_research_v2_model_m9_lowfreq_structure_summary.csv`
- 本地生成：`/tmp/m9_lowfreq_model_table.csv`
- 修改：`docs/experiments/rawish_state_aligned_l0_l1.md`

- [ ] **步骤 1：生成正式 summary**

运行：

```bash
./.venv/bin/python scripts/summarize_tho_runs.py \
  --runs-root runs/tho_research_v2_model_m9_lowfreq_structure \
  --output runs/tho_research_v2_model_m9_lowfreq_structure_summary.csv
```

预期：summary 至少包含 12 行：Patch-Hann baseline 2 行 + 5 个新模型各 2 行。

- [ ] **步骤 2：生成可读对照表**

运行：

```bash
./.venv/bin/python -c "import pandas as pd; p='runs/tho_research_v2_model_m9_lowfreq_structure_summary.csv'; df=pd.read_csv(p); cols=['run_id','best_val_loss','selection_task_rr_peak_band_abs_error_mean','selection_task_rr_spec_abs_error_mean','selection_task_relative_envelope_mae_mean','selection_task_relative_envelope_corr_mean','selection_waveform_band_limited_corr_mean','selection_waveform_best_lag_corr_mean','selection_waveform_rr_peak_abs_error_mean']; out=df[cols].copy(); out.to_csv('/tmp/m9_lowfreq_model_table.csv', index=False); print(out.to_string(index=False))"
```

预期：输出所有正式 run 的核心指标，且无 `NaN` 主指标。

- [ ] **步骤 3：追加 M9 首轮表格**

在 `docs/experiments/rawish_state_aligned_l0_l1.md` 新增 `## M9 低自由度模型结构实验` 小节。表格列固定为：

```text
label | run | model | seed | best val loss | rr_peak_band_abs_error mean / median | rr_spec_abs_error mean | relative_envelope_mae mean | relative_envelope_corr mean | band_limited_corr mean | best_lag_corr mean | raw rr_peak_abs_error mean | 结论
```

- [ ] **步骤 4：写首轮筛选结论**

按以下规则写阶段判断：

- 若模型方向 gate 失败，标记为“不通过方向护栏”，不进入扩 seed。
- 若模型方向通过但 `rr_peak_band_abs_error_mean` 明显劣于 baseline，标记为“低频形态可用但任务指标不通过”。
- 若模型方向通过且主指标接近或优于 baseline，同时 raw peak 更低，进入 4-6 seed 扩展。
- 若 `frequency_bottleneck1d` 或 `downsampled_ssm1d` 失败，先判断是结构不适合还是实现容量不足；不因为复杂模型失败就回退 loss 阶段。

- [ ] **步骤 5：验证文档与测试**

运行：

```bash
./.venv/bin/python -m pytest tests/test_model_registry.py tests/test_losses.py tests/test_base_experiment.py -q
./.venv/bin/python -m py_compile resp_train/models/lowfreq.py resp_train/models/registry.py scripts/summarize_tho_runs.py
git diff --check
```

预期：pytest 全部通过；py_compile 无输出；`git diff --check` 无输出。

- [ ] **步骤 6：Commit M9 首轮总结**

运行：

```bash
git add docs/superpowers/plans/2026-06-18-model-m9-lowfreq-structure.md docs/experiments/rawish_state_aligned_l0_l1.md
git commit -m "exp: 记录 M9 低频结构模型首轮实验"
```

预期：只提交计划和实验台账；`runs/` 产物仍留在本地。

## 任务 12：扩 seed 决策门

**文件：**
- 修改：`docs/experiments/rawish_state_aligned_l0_l1.md`

- [ ] **步骤 1：选择扩展候选**

根据任务 11 的筛选结论，最多选择 2 个候选进入扩 seed。排序优先级：

```text
1. direction gate 通过
2. rr_peak_band_abs_error_mean 最低
3. rr_spec_abs_error_mean 不退化
4. relative_envelope_mae_mean 不退化
5. raw rr_peak_abs_error_mean 更低
```

- [ ] **步骤 2：运行扩展 seed**

对入选模型运行：

```text
training.seed=20260650
training.seed=20260660
training.seed=20260690
training.seed=20260720
```

训练命令沿用任务 10 的正式训练口径，只替换模型名、模型专属参数和 `training.seed`。

预期：每个入选模型新增 4 个 run。

- [ ] **步骤 3：扩展汇总**

运行：

```bash
./.venv/bin/python scripts/summarize_tho_runs.py \
  --runs-root runs/tho_research_v2_model_m9_lowfreq_structure \
  --output runs/tho_research_v2_model_m9_lowfreq_structure_summary.csv
```

预期：summary 行数增加，入选模型至少各有 6 个 seed。

- [ ] **步骤 4：写扩 seed 结论并提交**

在 `docs/experiments/rawish_state_aligned_l0_l1.md` 的 M9 小节追加扩 seed 聚合表，然后提交：

```bash
git add docs/experiments/rawish_state_aligned_l0_l1.md
git commit -m "exp: 记录 M9 候选扩 seed 结果"
```

## 验收清单

- [ ] 5 个新模型都能通过 registry 与 shape 测试。
- [ ] 5 个新模型都能完成一轮 GPU smoke 训练并产出 `metrics.csv`。
- [ ] 正式训练固定 loss、输入、数据 seed、batch size 与 checkpoint gate，不混入新 loss 变量。
- [ ] 首轮正式训练包含 Patch-Hann signed baseline 和 5 个新模型，每个模型至少 2 个 seed。
- [ ] 筛选以 `selection_task_rr_peak_band_abs_error_mean` 为主，方向 gate 为硬护栏，raw peak 为诊断。
- [ ] 若需要安装 `mamba-ssm`，先单独向用户说明 CUDA/PyTorch ABI 风险并获得确认。
- [ ] 每个正式模型实验阶段完成后都有 git 保存，不把多个实验结论长期混在未提交工作区里。
