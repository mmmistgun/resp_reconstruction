# E1 STFT 输入信息增益验证 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 把纯时序训练栈扩展到能跑“时序 + STFT 双分支”，产出 E0 收尾 + E1a/a'/b/c/d 对照，回答“STFT 输入是否有稳定信息增益”。

**架构：** STFT 在模型内部用 `torch.stft` 现场算（路径 A），dataset/engine 零改动；新增通用包装器 `TimeStftDual1D` 持有“时序主干（return_features 出 `(B,C,T')` 特征）+ STFTEncoder + 融合头”，按 `branch_mode` 切换 time_only/stft_only/dual 三模式；现有主干加 `return_features=False` 开关保证逐元素向后兼容。

**技术栈：** PyTorch、OmegaConf、pytest、现有 `resp_train` 训练栈（`build_model` 注册表 + `ThoExperiment`）。

参考规格：`docs/superpowers/specs/2026-06-21-e1-stft-info-gain-design.md`

---

## 文件结构

**创建：**
- `resp_train/models/stft_branch.py` — STFTEncoder（固定算子+轻编码）、TimeStftDual1D（包装器）、FusionHead、align_to_time 工具。
- `tests/test_stft_branch.py` — STFTEncoder / TimeStftDual1D / 向后兼容 单元测试。
- `scripts/precompute_stft_band_scale.py` — N1 归一化的 per-freq-bin 鲁棒尺度（IQR）预统计脚本。
- `scripts/run_e1_stft_info_gain.py` — E1 批次编排（笛卡尔积 + override）。
- `tests/test_run_e1_overrides.py` — E1 编排脚本的 override 生成单测（不实际训练）。

**修改：**
- `resp_train/models/timeseries.py` — `PatchMixer1D.forward` 加 `return_features` 开关。
- `resp_train/models/lowfreq.py` — `MultiScaleDecompMixer1D.forward` 加 `return_features` 开关。
- `resp_train/models/registry.py` — 注册 `time_stft_dual1d`。
- `tests/test_model_registry.py` — 补 `return_features` 向后兼容与契约测试、`time_stft_dual1d` 注册测试。

**完全不动：** dataset / engine / collate / loss / checkpoint 校验 / 指标。

**关键契约（贯穿全计划，命名固定）：**
- 主干 `forward(x, return_features: bool = False)`：
  - `False` → 返回 `(B,1,L)` 波形（现状）。
  - `True` → 返回 `tuple(features, length)`，`features` 形状 `(B,C,T')`，`length` 为 int（原始输入长度）。
- `STFTEncoder.forward(x) -> Tensor`：`(B,1,L)` → `(B,Cs,Ts')`。
- `TimeStftDual1D.forward(x) -> Tensor`：`(B,1,L)` → `(B,1,L)`。
- cfg 字段：`model.name=time_stft_dual1d`、`model.branch_mode∈{time_only,stft_only,dual}`、`model.time_backbone`（主干名）、`model.stft_win`、`model.stft_hop`、`model.stft_low_hz`、`model.stft_high_hz`、`model.stft_norm∈{n0,n1}`、`model.fuse_len`、`model.stft_band_scale_path`（N1 用，可空）。

---

## 任务 1：PatchMixer1D 加 return_features 开关

**文件：**
- 修改：`resp_train/models/timeseries.py`（`PatchMixer1D.forward`，约 172-184 行）
- 测试：`tests/test_model_registry.py`

- [ ] **步骤 1：编写失败的测试**

在 `tests/test_model_registry.py` 末尾追加：

```python
def test_patch_mixer_return_features_is_backward_compatible():
    model = PatchMixer1D(in_channels=1, out_channels=1, base_channels=8, patch_len=128, patch_stride=64)
    model.eval()
    x = torch.randn(2, 1, 4096)

    with torch.no_grad():
        default_out = model(x)
        explicit_out = model(x, return_features=False)

    assert torch.equal(default_out, explicit_out)


def test_patch_mixer_return_features_shape_contract():
    model = PatchMixer1D(in_channels=1, out_channels=1, base_channels=8, patch_len=128, patch_stride=64)
    x = torch.randn(2, 1, 4096)

    features, length = model(x, return_features=True)

    assert features.dim() == 3
    assert features.shape[0] == 2
    assert features.shape[1] == 8  # base_channels = C
    assert length == 4096
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_model_registry.py::test_patch_mixer_return_features_shape_contract -v`
预期：FAIL，`forward()` 不接受 `return_features` 关键字（TypeError）。

- [ ] **步骤 3：编写最少实现代码**

把 `resp_train/models/timeseries.py` 中 `PatchMixer1D.forward` 改为（仅在 `patch_head` 之前插入提前返回分支，其余不变）：

```python
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        batch, _, length = x.shape
        padded_length = self._padded_length(length)
        x_padded = _match_length(x, padded_length)
        patches = x_padded.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        patch_count = patches.size(2)
        tokens = patches.permute(0, 2, 1, 3).reshape(batch, patch_count, -1)
        tokens = self.patch_embed(tokens).transpose(1, 2)
        for block in self.blocks:
            tokens = block(tokens)
        if return_features:
            # tokens: (B, base_channels, patch_count) 即 (B, C, T')，供双分支特征级融合
            return tokens, length
        patch_values = self.patch_head(tokens.transpose(1, 2))
        patch_values = patch_values.view(batch, patch_count, self.out_channels, self.patch_len)
        return self.output_smoother(self._overlap_add(patch_values, length, padded_length))
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_model_registry.py -k "patch_mixer_return_features" -v`
预期：两条均 PASS。

- [ ] **步骤 5：运行 PatchMixer 全部既有测试确认零回归**

运行：`pytest tests/test_model_registry.py -k "patch_mixer" -v`
预期：全部 PASS（含既有的 overlap_add / smoothing 测试）。

- [ ] **步骤 6：Commit**

```bash
git add resp_train/models/timeseries.py tests/test_model_registry.py
git commit -m "feat: PatchMixer1D 增加 return_features 中间特征开关"
```

---

## 任务 2：MultiScaleDecompMixer1D 加 return_features 开关

**文件：**
- 修改：`resp_train/models/lowfreq.py`（`MultiScaleDecompMixer1D.forward`，约 142-154 行）
- 测试：`tests/test_model_registry.py`

- [ ] **步骤 1：编写失败的测试**

在 `tests/test_model_registry.py` 顶部 import 区补充：

```python
from resp_train.models.lowfreq import MultiScaleDecompMixer1D
```

在文件末尾追加：

```python
def test_multiscale_decomp_return_features_is_backward_compatible():
    model = MultiScaleDecompMixer1D(in_channels=1, out_channels=1, base_channels=8, downsample_factors=[1, 4, 16])
    model.eval()
    x = torch.randn(2, 1, 4096)

    with torch.no_grad():
        default_out = model(x)
        explicit_out = model(x, return_features=False)

    assert torch.equal(default_out, explicit_out)


def test_multiscale_decomp_return_features_shape_contract():
    model = MultiScaleDecompMixer1D(in_channels=1, out_channels=1, base_channels=8, downsample_factors=[1, 4, 16])
    x = torch.randn(2, 1, 4096)

    features, length = model(x, return_features=True)

    assert features.shape == (2, 8 * 3, 4096)  # base_channels * n_scale，时间网格 = length
    assert length == 4096
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_model_registry.py::test_multiscale_decomp_return_features_shape_contract -v`
预期：FAIL，`forward()` 不接受 `return_features`（TypeError）。

- [ ] **步骤 3：编写最少实现代码**

把 `resp_train/models/lowfreq.py` 中 `MultiScaleDecompMixer1D.forward` 改为（仅在 `self.fuse` 之前插入提前返回）：

```python
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
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
        fused_input = torch.cat(outputs, dim=1)
        if return_features:
            # fused_input: (B, base_channels*n_scale, length) 即 (B, C, T')
            return fused_input, length
        return self.fuse(fused_input)
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_model_registry.py -k "multiscale_decomp_return_features" -v`
预期：两条均 PASS。

- [ ] **步骤 5：运行多尺度模型既有测试确认零回归**

运行：`pytest tests/test_model_registry.py -k "multiscale_decomp" -v`
预期：全部 PASS。

- [ ] **步骤 6：Commit**

```bash
git add resp_train/models/lowfreq.py tests/test_model_registry.py
git commit -m "feat: MultiScaleDecompMixer1D 增加 return_features 中间特征开关"
```

---

## 任务 3：STFTEncoder（固定算子 + N0 归一化 + 轻编码）

**文件：**
- 创建：`resp_train/models/stft_branch.py`
- 测试：`tests/test_stft_branch.py`

本任务只实现 N0（仅 log1p）。N1（频带鲁棒尺度）在任务 6 加挂。

- [ ] **步骤 1：编写失败的测试**

创建 `tests/test_stft_branch.py`：

```python
import pytest
import torch

from resp_train.models.stft_branch import STFTEncoder


def _encoder(high_hz: float) -> STFTEncoder:
    return STFTEncoder(
        sample_rate=100.0,
        stft_win=3000,
        stft_hop=500,
        low_hz=0.05,
        high_hz=high_hz,
        out_channels=16,
        norm="n0",
    )


def test_stft_encoder_output_is_3d_time_series():
    enc = _encoder(3.0)
    x = torch.randn(2, 1, 18000)

    feats = enc(x)

    assert feats.dim() == 3
    assert feats.shape[0] == 2
    assert feats.shape[1] == 16  # out_channels


def test_stft_encoder_freq_bins_increase_with_high_hz():
    x = torch.randn(1, 1, 18000)
    bins_3 = _encoder(3.0).band_bin_count()
    bins_8 = _encoder(8.0).band_bin_count()
    bins_12 = _encoder(12.0).band_bin_count()

    assert bins_3 < bins_8 < bins_12


def test_stft_encoder_handles_zero_and_nan_input_without_crash():
    enc = _encoder(8.0)
    zero = enc(torch.zeros(1, 1, 18000))
    assert torch.isfinite(zero).all()

    x = torch.randn(1, 1, 18000)
    x[0, 0, :100] = float("nan")
    out = enc(x)
    # NaN 输入不应让前向抛错；产出可含 nan，但不得崩溃
    assert out.shape[0] == 1
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_stft_branch.py -v`
预期：FAIL，`resp_train.models.stft_branch` 模块不存在（ImportError）。

- [ ] **步骤 3：编写最少实现代码**

创建 `resp_train/models/stft_branch.py`：

```python
from __future__ import annotations

import torch
from torch import nn


def _band_indices(stft_win: int, sample_rate: float, low_hz: float, high_hz: float) -> tuple[int, int]:
    """返回 rfft 频点中落在 [low_hz, high_hz] 的起止索引（含 low、不含 high+1 之外）。"""
    freqs = torch.fft.rfftfreq(stft_win, d=1.0 / sample_rate)
    lo = int(torch.searchsorted(freqs, torch.tensor(float(low_hz)), right=False))
    hi = int(torch.searchsorted(freqs, torch.tensor(float(high_hz)), right=True))
    hi = max(hi, lo + 1)
    return lo, hi


class STFTEncoder(nn.Module):
    """对单通道时序做 STFT → log1p magnitude → 频带裁剪 → 归一化 → 轻编码，输出 (B,Cs,Ts')。"""

    def __init__(
        self,
        sample_rate: float = 100.0,
        stft_win: int = 3000,
        stft_hop: int = 500,
        low_hz: float = 0.05,
        high_hz: float = 3.0,
        out_channels: int = 16,
        norm: str = "n0",
    ) -> None:
        super().__init__()
        self.sample_rate = float(sample_rate)
        self.stft_win = int(stft_win)
        self.stft_hop = int(stft_hop)
        self.norm = str(norm)
        self._lo, self._hi = _band_indices(self.stft_win, self.sample_rate, low_hz, high_hz)
        self.register_buffer("stft_window", torch.hann_window(self.stft_win), persistent=False)
        # N1 用的 per-freq-bin 鲁棒尺度，默认 1（即不缩放，等价 N0）；任务 6 注入真实值。
        self.register_buffer("band_scale", torch.ones(self._hi - self._lo), persistent=True)
        freq_bins = self._hi - self._lo
        # 轻编码：把 (freq, time) 视作 (C=freq, L=time) 的 1D 序列，conv 压成 out_channels。
        self.encoder = nn.Sequential(
            nn.Conv1d(freq_bins, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.SiLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )

    def band_bin_count(self) -> int:
        return self._hi - self._lo

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        signal = x.reshape(x.shape[0], -1)  # (B, L)
        spec = torch.stft(
            signal,
            n_fft=self.stft_win,
            hop_length=self.stft_hop,
            win_length=self.stft_win,
            window=self.stft_window.to(signal.dtype),
            center=True,
            return_complex=True,
        )  # (B, F, T)
        mag = spec.abs()
        log_mag = torch.log1p(mag)
        band = log_mag[:, self._lo : self._hi, :]  # (B, freq_bins, T)
        if self.norm == "n1":
            band = band / self.band_scale.view(1, -1, 1).clamp_min(1e-6)
        return self.encoder(band)  # (B, out_channels, T)
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_stft_branch.py -v`
预期：全部 PASS。

- [ ] **步骤 5：Commit**

```bash
git add resp_train/models/stft_branch.py tests/test_stft_branch.py
git commit -m "feat: 新增 STFTEncoder 时频分支（N0 归一化）"
```

---

## 任务 4：FusionHead 与 align_to_time 工具

**文件：**
- 修改：`resp_train/models/stft_branch.py`（追加 `align_to_time`、`FusionHead`）
- 测试：`tests/test_stft_branch.py`

- [ ] **步骤 1：编写失败的测试**

在 `tests/test_stft_branch.py` 追加：

```python
from resp_train.models.stft_branch import FusionHead, align_to_time


def test_align_to_time_resamples_to_target_length():
    feats = torch.randn(2, 8, 31)

    aligned = align_to_time(feats, target_len=600)

    assert aligned.shape == (2, 8, 600)


def test_fusion_head_outputs_waveform_shape():
    head = FusionHead(in_channels=24, out_length=18000, hidden=16)
    fused = torch.randn(2, 24, 600)

    out = head(fused)

    assert out.shape == (2, 1, 18000)
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_stft_branch.py -k "align_to_time or fusion_head" -v`
预期：FAIL，`align_to_time` / `FusionHead` 未定义（ImportError）。

- [ ] **步骤 3：编写最少实现代码**

在 `resp_train/models/stft_branch.py` 追加：

```python
import torch.nn.functional as F


def align_to_time(feats: torch.Tensor, target_len: int) -> torch.Tensor:
    """把 (B,C,T') 沿时间轴线性插值到公共网格 (B,C,target_len)。"""
    if feats.shape[-1] == target_len:
        return feats
    return F.interpolate(feats, size=int(target_len), mode="linear", align_corners=False)


class FusionHead(nn.Module):
    """把对齐后的拼接特征 (B,C,fuse_len) 解码并上采样回 (B,1,out_length) 波形。"""

    def __init__(self, in_channels: int, out_length: int, hidden: int = 16) -> None:
        super().__init__()
        self.out_length = int(out_length)
        self.decoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(1, hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, 1, kernel_size=1),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        decoded = self.decoder(fused)  # (B,1,fuse_len)
        if decoded.shape[-1] != self.out_length:
            decoded = F.interpolate(decoded, size=self.out_length, mode="linear", align_corners=False)
        return decoded
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_stft_branch.py -k "align_to_time or fusion_head" -v`
预期：两条均 PASS。

- [ ] **步骤 5：Commit**

```bash
git add resp_train/models/stft_branch.py tests/test_stft_branch.py
git commit -m "feat: 新增 align_to_time 与 FusionHead 融合工具"
```

---

## 任务 5：TimeStftDual1D 包装器与三模式

**文件：**
- 修改：`resp_train/models/stft_branch.py`（追加 `TimeStftDual1D`）
- 测试：`tests/test_stft_branch.py`

- [ ] **步骤 1：编写失败的测试**

在 `tests/test_stft_branch.py` 追加：

```python
import pytest
from resp_train.models.stft_branch import TimeStftDual1D


def _dual(branch_mode: str) -> TimeStftDual1D:
    return TimeStftDual1D(
        time_backbone_name="patch_mixer1d",
        time_backbone_kwargs=dict(in_channels=1, out_channels=1, base_channels=8, patch_len=128, patch_stride=64),
        time_feat_channels=8,
        branch_mode=branch_mode,
        out_length=18000,
        fuse_len=600,
        stft_kwargs=dict(sample_rate=100.0, stft_win=3000, stft_hop=500, low_hz=0.05, high_hz=3.0, out_channels=16, norm="n0"),
    )


@pytest.mark.parametrize("branch_mode", ["time_only", "stft_only", "dual"])
def test_dual_outputs_waveform_shape(branch_mode):
    model = _dual(branch_mode)
    x = torch.randn(2, 1, 18000)

    out = model(x)

    assert out.shape == (2, 1, 18000)
    assert torch.isfinite(out).all()


def test_dual_mode_uses_both_branches_gradients():
    model = _dual("dual")
    x = torch.randn(2, 1, 18000)

    model(x).square().mean().backward()

    time_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.time_backbone.parameters())
    stft_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.stft_encoder.parameters())
    assert time_grad and stft_grad


def test_stft_only_mode_has_no_time_backbone():
    model = _dual("stft_only")
    assert model.time_backbone is None


def test_time_only_mode_has_no_stft_encoder():
    model = _dual("time_only")
    assert model.stft_encoder is None
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_stft_branch.py -k "dual or stft_only_mode or time_only_mode" -v`
预期：FAIL，`TimeStftDual1D` 未定义。

- [ ] **步骤 3：编写最少实现代码**

在 `resp_train/models/stft_branch.py` 追加（`build_time_backbone` 用一个局部 import 避免与 registry 循环依赖）：

```python
def _build_time_backbone(name: str, kwargs: dict):
    from resp_train.models.timeseries import PatchMixer1D
    from resp_train.models.lowfreq import MultiScaleDecompMixer1D

    table = {
        "patch_mixer1d": PatchMixer1D,
        "multiscale_decomp_mixer1d": MultiScaleDecompMixer1D,
    }
    if name not in table:
        raise KeyError(f"time_backbone 暂不支持: {name}，可用: {sorted(table)}")
    return table[name](**kwargs)


class TimeStftDual1D(nn.Module):
    """通用双分支包装器：时序主干(出特征) + STFT 分支，特征级 concat 后融合出波形。"""

    def __init__(
        self,
        time_backbone_name: str,
        time_backbone_kwargs: dict,
        time_feat_channels: int,
        branch_mode: str,
        out_length: int,
        fuse_len: int,
        stft_kwargs: dict,
        fusion_hidden: int = 16,
    ) -> None:
        super().__init__()
        self.branch_mode = str(branch_mode)
        if self.branch_mode not in {"time_only", "stft_only", "dual"}:
            raise ValueError(f"未知 branch_mode: {self.branch_mode}")
        self.fuse_len = int(fuse_len)

        use_time = self.branch_mode in {"time_only", "dual"}
        use_stft = self.branch_mode in {"stft_only", "dual"}
        self.time_backbone = _build_time_backbone(time_backbone_name, time_backbone_kwargs) if use_time else None
        self.stft_encoder = STFTEncoder(**stft_kwargs) if use_stft else None

        fused_channels = 0
        if use_time:
            fused_channels += int(time_feat_channels)
        if use_stft:
            fused_channels += int(stft_kwargs["out_channels"])
        self.fusion_head = FusionHead(in_channels=fused_channels, out_length=int(out_length), hidden=fusion_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = []
        if self.time_backbone is not None:
            f_t, _ = self.time_backbone(x, return_features=True)
            feats.append(align_to_time(f_t, self.fuse_len))
        if self.stft_encoder is not None:
            f_s = self.stft_encoder(x)
            feats.append(align_to_time(f_s, self.fuse_len))
        return self.fusion_head(torch.cat(feats, dim=1))
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_stft_branch.py -v`
预期：全部 PASS。

- [ ] **步骤 5：Commit**

```bash
git add resp_train/models/stft_branch.py tests/test_stft_branch.py
git commit -m "feat: 新增 TimeStftDual1D 双分支包装器与三模式"
```

---

## 任务 6：N1 频带鲁棒尺度（预统计脚本 + 注入）

**文件：**
- 创建：`scripts/precompute_stft_band_scale.py`
- 修改：`resp_train/models/stft_branch.py`（`STFTEncoder` 支持从文件加载 `band_scale`）
- 测试：`tests/test_stft_branch.py`

- [ ] **步骤 1：编写失败的测试**

在 `tests/test_stft_branch.py` 追加：

```python
def test_stft_encoder_n1_divides_by_band_scale(tmp_path):
    import numpy as np

    enc = _encoder(3.0)  # norm="n0"
    n_bins = enc.band_bin_count()
    scale = np.full(n_bins, 2.0, dtype=np.float32)
    scale_path = tmp_path / "band_scale_3hz.npy"
    np.save(scale_path, scale)

    enc_n1 = STFTEncoder(
        sample_rate=100.0, stft_win=3000, stft_hop=500, low_hz=0.05, high_hz=3.0,
        out_channels=16, norm="n1", band_scale_path=str(scale_path),
    )

    assert torch.allclose(enc_n1.band_scale, torch.full((n_bins,), 2.0))
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_stft_branch.py::test_stft_encoder_n1_divides_by_band_scale -v`
预期：FAIL，`STFTEncoder.__init__` 不接受 `band_scale_path`。

- [ ] **步骤 3：编写最少实现代码**

修改 `resp_train/models/stft_branch.py` 的 `STFTEncoder.__init__` 签名与 buffer 初始化，新增 `band_scale_path` 参数：

```python
    def __init__(
        self,
        sample_rate: float = 100.0,
        stft_win: int = 3000,
        stft_hop: int = 500,
        low_hz: float = 0.05,
        high_hz: float = 3.0,
        out_channels: int = 16,
        norm: str = "n0",
        band_scale_path: str | None = None,
    ) -> None:
        super().__init__()
        self.sample_rate = float(sample_rate)
        self.stft_win = int(stft_win)
        self.stft_hop = int(stft_hop)
        self.norm = str(norm)
        self._lo, self._hi = _band_indices(self.stft_win, self.sample_rate, low_hz, high_hz)
        self.register_buffer("stft_window", torch.hann_window(self.stft_win), persistent=False)
        n_bins = self._hi - self._lo
        if self.norm == "n1" and band_scale_path:
            import numpy as np

            scale = torch.from_numpy(np.load(band_scale_path).astype("float32"))
            if scale.numel() != n_bins:
                raise ValueError(f"band_scale 长度 {scale.numel()} 与频点数 {n_bins} 不一致")
        else:
            scale = torch.ones(n_bins)
        self.register_buffer("band_scale", scale, persistent=True)
        # encoder 部分保持任务 3 原样
        self.encoder = nn.Sequential(
            nn.Conv1d(n_bins, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.SiLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )
```

创建 `scripts/precompute_stft_band_scale.py`：

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resp_train.config import load_config
from resp_train.data.factory import build_tho_data
from resp_train.models.stft_branch import _band_indices


def main() -> None:
    parser = argparse.ArgumentParser(description="预统计 STFT per-freq-bin 鲁棒尺度(IQR)，用于 N1 归一化")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--high-hz", type=float, required=True)
    parser.add_argument("--low-hz", type=float, default=0.05)
    parser.add_argument("--stft-win", type=int, default=3000)
    parser.add_argument("--stft-hop", type=int, default=500)
    parser.add_argument("--max-windows", type=int, default=512, help="抽样窗口数，控制统计成本")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    fs = float(cfg.window.target_fs)
    lo, hi = _band_indices(args.stft_win, fs, args.low_hz, args.high_hz)
    window = torch.hann_window(args.stft_win)

    data = build_tho_data(cfg)
    rows = data.train.dataset
    n = min(len(rows), args.max_windows)
    bands = []
    for i in range(n):
        x = rows[i]["x"].reshape(1, -1)
        spec = torch.stft(
            x, n_fft=args.stft_win, hop_length=args.stft_hop, win_length=args.stft_win,
            window=window, center=True, return_complex=True,
        ).abs()
        band = torch.log1p(spec)[0, lo:hi, :]  # (freq_bins, T)
        bands.append(band)
    stacked = torch.cat(bands, dim=1)  # (freq_bins, sum_T)
    q1 = torch.quantile(stacked, 0.25, dim=1)
    q3 = torch.quantile(stacked, 0.75, dim=1)
    iqr = (q3 - q1).clamp_min(1e-6).numpy().astype("float32")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, iqr)
    print(f"saved band_scale: {out_path} shape={iqr.shape} high_hz={args.high_hz}")


if __name__ == "__main__":
    main()
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_stft_branch.py::test_stft_encoder_n1_divides_by_band_scale -v`
预期：PASS。

- [ ] **步骤 5：运行 stft_branch 全部测试确认零回归**

运行：`pytest tests/test_stft_branch.py -v`
预期：全部 PASS（N0 路径不受影响）。

- [ ] **步骤 6：Commit**

```bash
git add resp_train/models/stft_branch.py scripts/precompute_stft_band_scale.py tests/test_stft_branch.py
git commit -m "feat: STFT N1 频带鲁棒尺度预统计与注入"
```

---

## 任务 7：在 registry 注册 time_stft_dual1d

**文件：**
- 修改：`resp_train/models/registry.py`
- 测试：`tests/test_model_registry.py`

- [ ] **步骤 1：编写失败的测试**

在 `tests/test_model_registry.py` 末尾追加：

```python
def test_time_stft_dual1d_is_registered():
    assert "time_stft_dual1d" in list_models()


@pytest.mark.parametrize("branch_mode", ["time_only", "stft_only", "dual"])
def test_build_time_stft_dual1d_preserves_waveform_shape(branch_mode):
    cfg = OmegaConf.create(
        {
            "window": {"target_fs": 100, "duration_samples": 18000},
            "model": {
                "name": "time_stft_dual1d",
                "in_channels": 1,
                "out_channels": 1,
                "base_channels": 8,
                "branch_mode": branch_mode,
                "time_backbone": "patch_mixer1d",
                "patch_len": 128,
                "patch_stride": 64,
                "mixer_layers": 1,
                "stft_win": 3000,
                "stft_hop": 500,
                "stft_low_hz": 0.05,
                "stft_high_hz": 3.0,
                "stft_out_channels": 16,
                "stft_norm": "n0",
                "fuse_len": 600,
            },
        }
    )
    model = build_model(cfg)

    y = model(torch.randn(2, 1, 18000))

    assert y.shape == (2, 1, 18000)
    assert torch.isfinite(y).all()
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_model_registry.py -k "time_stft_dual1d" -v`
预期：FAIL，`time_stft_dual1d` 未注册（KeyError）。

- [ ] **步骤 3：编写最少实现代码**

在 `resp_train/models/registry.py` 顶部 import 区追加：

```python
from resp_train.models.stft_branch import TimeStftDual1D
```

在 `_REGISTRY` 字典中追加一个工厂条目（放在 `downsampled_ssm1d` 之后、闭括号之前）。该工厂解析双分支配置，并按 `time_backbone` 推断时序主干特征通道数（= base_channels）：

```python
    "time_stft_dual1d": lambda cfg: TimeStftDual1D(
        time_backbone_name=str(cfg.model.get("time_backbone", "patch_mixer1d")),
        time_backbone_kwargs=_time_backbone_kwargs(cfg),
        time_feat_channels=int(cfg.model.base_channels),
        branch_mode=str(cfg.model.get("branch_mode", "dual")),
        out_length=int(cfg.window.get("duration_samples", 18000)),
        fuse_len=int(cfg.model.get("fuse_len", 600)),
        stft_kwargs=dict(
            sample_rate=float(cfg.window.get("target_fs", 100)),
            stft_win=int(cfg.model.get("stft_win", 3000)),
            stft_hop=int(cfg.model.get("stft_hop", 500)),
            low_hz=float(cfg.model.get("stft_low_hz", 0.05)),
            high_hz=float(cfg.model.get("stft_high_hz", 3.0)),
            out_channels=int(cfg.model.get("stft_out_channels", 16)),
            norm=str(cfg.model.get("stft_norm", "n0")),
            band_scale_path=(str(cfg.model.get("stft_band_scale_path")) if cfg.model.get("stft_band_scale_path") else None),
        ),
    ),
```

在 `_REGISTRY` 定义之前新增辅助函数（解析两类主干各自的构造参数）：

```python
def _time_backbone_kwargs(cfg: Any) -> dict:
    """按 time_backbone 类型解析时序主干构造参数，仅支持 E1 用到的两类。"""
    name = str(cfg.model.get("time_backbone", "patch_mixer1d"))
    common = dict(
        in_channels=int(cfg.model.in_channels),
        out_channels=int(cfg.model.out_channels),
        base_channels=int(cfg.model.base_channels),
    )
    if name == "patch_mixer1d":
        return dict(
            **common,
            patch_len=int(cfg.model.get("patch_len", 256)),
            patch_stride=int(cfg.model.get("patch_stride", 128)),
            mixer_layers=int(cfg.model.get("mixer_layers", 2)),
            overlap_window=str(cfg.model.get("overlap_window", "uniform")),
            output_smoothing_kernel=int(cfg.model.get("output_smoothing_kernel", 1)),
        )
    if name == "multiscale_decomp_mixer1d":
        return dict(
            **common,
            downsample_factors=list(cfg.model.get("downsample_factors", [1, 4, 16])),
            mixer_layers=int(cfg.model.get("mixer_layers", 2)),
        )
    raise KeyError(f"time_backbone 暂不支持: {name}")
```

注意：`Any` 已在 `registry.py` 顶部从 `typing` 导入（现状第 3 行 `from typing import Any, Callable`），无需重复导入。

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_model_registry.py -k "time_stft_dual1d" -v`
预期：4 条（注册 + 3 个 branch_mode）均 PASS。

- [ ] **步骤 5：运行全部模型注册测试确认零回归**

运行：`pytest tests/test_model_registry.py -v`
预期：全部 PASS。

- [ ] **步骤 6：Commit**

```bash
git add resp_train/models/registry.py tests/test_model_registry.py
git commit -m "feat: 注册 time_stft_dual1d 双分支模型"
```

---

## 任务 8：双分支模型集成训练冒烟测试

**文件：**
- 测试：`tests/test_engine_smoke.py`（追加）

确认 `time_stft_dual1d` 能进现有训练栈：前向→loss→反向→checkpoint 存取往返一致。

- [ ] **步骤 1：先查看现有冒烟测试的搭建方式**

运行：`pytest tests/test_engine_smoke.py -v` 并阅读该文件，确认其如何构造最小 cfg、如何调用 engine（沿用其 fixture/helper，不要新造数据集）。

- [ ] **步骤 2：编写失败的测试**

在 `tests/test_engine_smoke.py` 追加（若该文件已有构造小 batch 的 helper，复用之；下面给出自包含版本，仅依赖现有 `collect_predictions` 与模型前向，不依赖真实数据集）：

```python
import torch
from omegaconf import OmegaConf

from resp_train.models import build_model


def test_time_stft_dual1d_forward_backward_and_state_roundtrip():
    cfg = OmegaConf.create(
        {
            "window": {"target_fs": 100, "duration_samples": 18000},
            "model": {
                "name": "time_stft_dual1d",
                "in_channels": 1,
                "out_channels": 1,
                "base_channels": 8,
                "branch_mode": "dual",
                "time_backbone": "multiscale_decomp_mixer1d",
                "downsample_factors": [1, 4, 16],
                "mixer_layers": 1,
                "stft_win": 3000,
                "stft_hop": 500,
                "stft_low_hz": 0.05,
                "stft_high_hz": 8.0,
                "stft_out_channels": 16,
                "stft_norm": "n0",
                "fuse_len": 600,
            },
        }
    )
    model = build_model(cfg)
    x = torch.randn(2, 1, 18000)
    target = torch.randn(2, 1, 18000)

    pred = model(x)
    loss = (pred - target).square().mean()
    loss.backward()
    grad_norm = sum(
        float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None
    )
    assert pred.shape == (2, 1, 18000)
    assert grad_norm > 0.0

    # checkpoint 往返：state_dict 存取后输出一致
    state = model.state_dict()
    fresh = build_model(cfg)
    fresh.load_state_dict(state)
    model.eval()
    fresh.eval()
    with torch.no_grad():
        assert torch.allclose(model(x), fresh(x), atol=1e-5)
```

- [ ] **步骤 3：运行测试验证失败或通过**

运行：`pytest tests/test_engine_smoke.py::test_time_stft_dual1d_forward_backward_and_state_roundtrip -v`
预期：若任务 1-7 正确，应直接 PASS；若 FAIL，按报错回到对应任务修复（这是集成护栏）。

- [ ] **步骤 4：Commit**

```bash
git add tests/test_engine_smoke.py
git commit -m "test: time_stft_dual1d 前向/反向/checkpoint 往返集成冒烟"
```

---

## 任务 9：E0 收尾——split 审计与 ssm 补跑脚本

**文件：**
- 修改：`scripts/run_20260620_softz_candidates.py`（`DEFAULT_MODELS` 与 `MODEL_OVERRIDES` 补 `downsampled_ssm1d`）

此任务不写自动化测试（属一次性实验编排），通过手动运行验证。

- [ ] **步骤 1：确认 ssm 是否已在候选脚本中**

运行：`grep -n "downsampled_ssm1d" scripts/run_20260620_softz_candidates.py`
预期：当前无输出（未纳入）。

- [ ] **步骤 2：在 MODEL_OVERRIDES 补 ssm 条目**

在 `scripts/run_20260620_softz_candidates.py` 的 `MODEL_OVERRIDES` 字典追加：

```python
    "downsampled_ssm1d": [
        "model.name=downsampled_ssm1d",
        "model.latent_stride=20",
        "model.state_layers=2",
    ],
```

并在 `DEFAULT_MODELS` 列表追加 `"downsampled_ssm1d"`。

- [ ] **步骤 3：本地干跑校验 override 拼接（不实际训练）**

运行：
```bash
python -c "import scripts.run_20260620_softz_candidates as m; print('downsampled_ssm1d' in m.MODEL_OVERRIDES and 'downsampled_ssm1d' in m.DEFAULT_MODELS)"
```
预期：输出 `True`。

- [ ] **步骤 4：跑 split 独立性审计（确认数据隔离）**

运行：
```bash
python scripts/audit_split_independence.py --config configs/tho_research_v2.yaml --output-dir runs/audit_split_20260620
```
预期：stdout 打印 `overlap_samp_id_count=0`，`runs/audit_split_20260620/summary.csv` 中 `has_samp_id_leakage=False`。
**若 `overlap_samp_id_count` 非 0，停止并报告——E1 的高频捷径护栏前提不成立。**

- [ ] **步骤 5：Commit**

```bash
git add scripts/run_20260620_softz_candidates.py
git commit -m "feat: 候选脚本补 downsampled_ssm1d 用于 E0 polarity 反例"
```

---

## 任务 10：E1 批次编排脚本

**文件：**
- 创建：`scripts/run_e1_stft_info_gain.py`
- 测试：`tests/test_run_e1_overrides.py`

脚本沿用 `run_20260620_softz_candidates.py` 的“笛卡尔积 + subprocess + --skip”模式，调用 `scripts/train_tho_small.py --config configs/tho_research_v2.yaml`。

- [ ] **步骤 1：编写失败的测试（只测 override 生成，不训练）**

创建 `tests/test_run_e1_overrides.py`：

```python
import scripts.run_e1_stft_info_gain as e1


def test_build_overrides_covers_four_labels():
    labels = {spec["label"] for spec in e1.build_run_specs()}
    assert {"E1a", "E1a_prime", "E1b", "E1c"} <= labels


def test_e1a_is_plain_backbone_not_wrapper():
    spec = next(s for s in e1.build_run_specs() if s["label"] == "E1a" and s["time_backbone"] == "patch_mixer1d")
    joined = " ".join(spec["overrides"])
    # E1a 直接用现有模型，不经包装器
    assert "model.name=patch_mixer1d" in joined
    assert "time_stft_dual1d" not in joined


def test_e1b_dual_sweeps_three_bands():
    bands = sorted(
        float(o.split("=")[1])
        for s in e1.build_run_specs() if s["label"] == "E1b" and s["time_backbone"] == "patch_mixer1d"
        for o in s["overrides"] if o.startswith("model.stft_high_hz=")
    )
    assert bands == [3.0, 8.0, 12.0]


def test_e1a_prime_is_wrapper_time_only():
    spec = next(s for s in e1.build_run_specs() if s["label"] == "E1a_prime" and s["time_backbone"] == "patch_mixer1d")
    joined = " ".join(spec["overrides"])
    assert "model.name=time_stft_dual1d" in joined
    assert "model.branch_mode=time_only" in joined
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_run_e1_overrides.py -v`
预期：FAIL，`scripts.run_e1_stft_info_gain` 不存在。

- [ ] **步骤 3：编写最少实现代码**

创建 `scripts/run_e1_stft_info_gain.py`：

```python
from __future__ import annotations

import argparse
import subprocess
import sys

COMMON_OVERRIDES = [
    "data.max_train_windows=null",
    "data.max_val_windows=null",
    "data.train_sample_seed=20260610",
    "data.val_sample_seed=20260611",
    "loss.phase_alignment_weight=0.0",
    "loss.signed_corr_weight=0.2",
    "training.epochs=50",
    "training.batch_size=128",
    "training.patience=8",
    "training.min_delta=0.001",
    "training.device=cuda:0",
    "training.show_progress=false",
    "training.checkpoint_gate.metric=auto_direction",
    "training.checkpoint_gate.max=0.5",
    "outputs.run_root=runs/tho_research_v2_20260620_e1_stft_info_gain",
]

TIME_BACKBONES = ["patch_mixer1d", "multiscale_decomp_mixer1d"]
HIGH_HZ_BANDS = [3.0, 8.0, 12.0]
SEEDS = [20260700, 20260710, 20260837]
STFT_BASE = [
    "model.name=time_stft_dual1d",
    "model.base_channels=16",
    "model.mixer_layers=2",
    "model.patch_len=256",
    "model.patch_stride=128",
    "model.stft_win=3000",
    "model.stft_hop=500",
    "model.stft_low_hz=0.05",
    "model.stft_out_channels=16",
    "model.stft_norm=n0",
    "model.fuse_len=600",
]


def _plain_backbone_overrides(backbone: str) -> list[str]:
    base = [f"model.name={backbone}", "model.base_channels=16", "model.mixer_layers=2"]
    if backbone == "patch_mixer1d":
        base += ["model.patch_len=256", "model.patch_stride=128"]
    return base


def build_run_specs() -> list[dict]:
    """生成 E1 全部 run 规格（label/time_backbone/high_hz/seed/overrides）。"""
    specs: list[dict] = []
    for backbone in TIME_BACKBONES:
        for seed in SEEDS:
            # E1a：现有模型直跑，不经包装器，与频带无关
            specs.append({
                "label": "E1a", "time_backbone": backbone, "high_hz": None, "seed": seed,
                "overrides": [*_plain_backbone_overrides(backbone), f"training.seed={seed}"],
            })
            # E1a'：包装器 time_only，与频带无关
            specs.append({
                "label": "E1a_prime", "time_backbone": backbone, "high_hz": None, "seed": seed,
                "overrides": [*STFT_BASE, f"model.time_backbone={backbone}",
                              "model.branch_mode=time_only", "model.stft_high_hz=3.0",
                              f"training.seed={seed}"],
            })
            for high_hz in HIGH_HZ_BANDS:
                # E1b：dual，扫三频带
                specs.append({
                    "label": "E1b", "time_backbone": backbone, "high_hz": high_hz, "seed": seed,
                    "overrides": [*STFT_BASE, f"model.time_backbone={backbone}",
                                  "model.branch_mode=dual", f"model.stft_high_hz={high_hz}",
                                  f"training.seed={seed}"],
                })
                # E1c：stft_only，扫三频带
                specs.append({
                    "label": "E1c", "time_backbone": backbone, "high_hz": high_hz, "seed": seed,
                    "overrides": [*STFT_BASE, f"model.time_backbone={backbone}",
                                  "model.branch_mode=stft_only", f"model.stft_high_hz={high_hz}",
                                  f"training.seed={seed}"],
                })
    return specs


def _tag(spec: dict) -> str:
    band = "na" if spec["high_hz"] is None else f"{spec['high_hz']:g}hz"
    return f"{spec['label']}_{spec['time_backbone']}_{band}_{spec['seed']}"


def main() -> None:
    parser = argparse.ArgumentParser(description="E1 STFT 信息增益批次编排")
    parser.add_argument("--skip", action="append", default=[], help="跳过的 run tag")
    parser.add_argument("--dry-run", action="store_true", help="只打印将运行的 tag，不实际训练")
    args = parser.parse_args()
    skipped = set(args.skip)

    for spec in build_run_specs():
        tag = _tag(spec)
        if tag in skipped:
            print(f"skip {tag}", flush=True)
            continue
        if args.dry_run:
            print(f"plan {tag}", flush=True)
            continue
        cmd = [sys.executable, "scripts/train_tho_small.py", "--config", "configs/tho_research_v2.yaml"]
        for override in [*COMMON_OVERRIDES, *spec["overrides"]]:
            cmd.extend(["--set", override])
        print(f"start {tag}", flush=True)
        subprocess.run(cmd, check=True)
        print(f"done {tag}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_run_e1_overrides.py -v`
预期：全部 PASS。

- [ ] **步骤 5：dry-run 校验 run 总数（应为约 48）**

运行：`python scripts/run_e1_stft_info_gain.py --dry-run | wc -l`
预期：`2 backbone × 3 seed ×（E1a 1 + E1a' 1 + E1b 3 + E1c 3）= 2×3×8 = 48`，输出 `48`。

- [ ] **步骤 6：Commit**

```bash
git add scripts/run_e1_stft_info_gain.py tests/test_run_e1_overrides.py
git commit -m "feat: E1 STFT 信息增益批次编排脚本"
```

---

## 任务 11：全量回归与文档收尾

**文件：**
- 修改：`docs/experiments/time_frequency_input_fusion_plan.md`（“当前建议”补一句实现状态指针，可选）

- [ ] **步骤 1：运行全部测试套件**

运行：`pytest tests/test_model_registry.py tests/test_stft_branch.py tests/test_engine_smoke.py tests/test_run_e1_overrides.py -v`
预期：全部 PASS，无回归。

- [ ] **步骤 2：确认现有纯时序模型行为零变化（抽样）**

运行：`pytest tests/test_model_registry.py -k "preserve_waveform_shape or backward" -v`
预期：全部 PASS（向后兼容护栏）。

- [ ] **步骤 3（可选）：更新规划文档实现状态**

若需要，在 `docs/experiments/time_frequency_input_fusion_plan.md` 的“当前建议”末尾追加一行指针：

```markdown
> 实现状态：E1 双分支训练栈与批次脚本见 `scripts/run_e1_stft_info_gain.py`，
> 设计规格见 `docs/superpowers/specs/2026-06-21-e1-stft-info-gain-design.md`。
```

- [ ] **步骤 4：Commit**

```bash
git add -A
git commit -m "docs: E1 双分支实现收尾，链接规格与批次脚本"
```

---

## 实验执行（代码完成后的手动步骤，非自动化任务）

代码与测试全绿后，按规格“实验编排”执行（GPU 资源充足时可并行）：

1. **E0 收尾**：
   - `python scripts/audit_split_independence.py --config configs/tho_research_v2.yaml --output-dir runs/audit_split_20260620`（确认 `has_samp_id_leakage=False`）。
   - 补跑 ssm：`python scripts/run_20260620_softz_candidates.py`（仅 ssm，可用 `--skip` 跳过已跑模型）。
2. **N1 频带尺度预统计**（仅 N1 对照档需要）：
   - `python scripts/precompute_stft_band_scale.py --config configs/tho_research_v2.yaml --high-hz 3 --output runs/stft_band_scale/band_scale_3hz.npy`
3. **E1 首波**：`python scripts/run_e1_stft_info_gain.py`（约 48 run，支持 `--skip` 续跑）。
4. **汇总**：`python scripts/summarize_tho_runs.py --runs-root runs/tho_research_v2_20260620_e1_stft_info_gain --output runs/tho_research_v2_20260620_e1_stft_info_gain_summary.csv`。
5. 按规划“记录模板”填 E1a/a'/b/c/d 指标 + 分层诊断，给出研究判定。

---

## 自检结果

**规格覆盖度核对：**
- STFTEncoder（含 N0/N1、频带裁剪、log1p、轻编码）→ 任务 3、6 ✓
- 中间特征开关（PatchMixer / MultiScaleDecomp，向后兼容）→ 任务 1、2 ✓
- TimeStftDual1D 三模式 + 融合头 + align_to_time → 任务 4、5 ✓
- registry 注册 + cfg 字段 → 任务 7 ✓
- 集成（进 engine、checkpoint 往返）→ 任务 8 ✓
- E0 收尾（split 审计 + ssm 补跑）→ 任务 9 ✓
- E1 编排（48 run、E1a 不经包装器、E1a' 容量桩、三频带）→ 任务 10 ✓
- 测试策略（向后兼容护栏、契约、三模式、集成）→ 任务 1-8 ✓
- 验收（全量回归、零变化）→ 任务 11 ✓

**占位符扫描：** 无 TODO/待定；每个代码步骤均含完整代码。

**类型一致性：** `return_features` 返回 `(features, length)` 在任务 1/2 定义、任务 5 消费一致；`STFTEncoder` 构造参数（`out_channels`/`norm`/`band_scale_path`）在任务 3/6/7 一致；`TimeStftDual1D` 构造参数（`time_feat_channels`/`stft_kwargs.out_channels`）在任务 5/7 一致；`build_run_specs` 返回的 `label`（E1a/E1a_prime/E1b/E1c）在任务 10 测试与实现一致。
