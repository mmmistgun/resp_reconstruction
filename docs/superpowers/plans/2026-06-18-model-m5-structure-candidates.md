# M5 模型结构候选实验实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 固定当前 N3 loss 和 Research v2 数据口径，比较一批低风险模型结构候选，验证 skip 泄漏和显式频率前端是否能优于当前 Patch-Hann baseline。

**架构：** M5 不改 loss，只新增两个轻量模型：完全去浅层/深层 skip 的 `unet1d_tiny_noskip_all`，以及带可学习 FIR 呼吸频带前端的 `patch_mixer1d_fir_frontend`。训练阶段使用同 seed Patch-Hann baseline、已有 `unet1d_tiny_noskip1`、新增 `unet1d_tiny_noskip_all`、新增 FIR PatchMixer 做并行对照。

**技术栈：** Python、PyTorch、OmegaConf、pandas、pytest、现有 `resp_train` 训练与 `summarize_tho_runs.py` 汇总脚本。

---

## 背景与固定口径

M4 结论：

- `patch_mixer1d + hann + output_smoothing_kernel=1` 是当前主任务 baseline。
- 简单输出平滑能降低 raw `rr_peak_abs_error`，但会损伤 `rr_peak_band_abs_error` 主指标。
- 下一步应优先尝试模型结构的低频归纳偏置，而不是继续加大输出端 moving average。

M5 只验证两个模型假设：

1. **skip 泄漏假设**：U-Net 浅层 skip 可能把 BCG 高频结构直通输出。现有 `unet1d_tiny_noskip1` 只去掉最后一层浅 skip；M5 新增 `unet1d_tiny_noskip_all`，完全去掉两个 decoder skip，对照 skip 是否是主要问题。
2. **显式频率前端假设**：PatchMixer 的主任务 RR 好，但 raw peak 毛刺高。M5 新增 `patch_mixer1d_fir_frontend`，在 Patch-Hann 前加入初始化为呼吸频带的 depthwise FIR 前端，让模型先看带通后的输入，再做 patch mixing。

M5 不纳入：

- `dual_stream_lpf_unet`：这是推荐主线，但它同时引入低频分支、残差门控和融合 decoder，变量更多。若 M5 证明 skip 或 FIR 方向有效，再进入 M6。
- `frequency_bottleneck` / `FreTS-lite`：科研味更强，但实现和调参风险高，放在 M7。
- `Mamba` / 大 Transformer / 完整 PatchTST：长窗 `18000` 点下计算和接口风险高，暂不进入当前批次。

固定训练口径：

- 输入：`bcg_rawish_wideband_state_aligned`
- 目标：`tho_waveform_ref`
- 数据窗口：全量（`data.max_train_windows=null`，`data.max_val_windows=null`）
- 数据 seed：`data.train_sample_seed=20260610`，`data.val_sample_seed=20260611`
- Loss：N3 主线，`high_freq_weight=0.2`，`relative_envelope_weight=0.03`，`phase_alignment_weight=0.005`
- 训练：`epochs=50`，`batch_size=128`，`patience=8`，`min_delta=0.001`，`use_amp=false`
- 训练 seed：先跑 `20260650`；若无硬失败，再跑 `20260660`
- 输出：`runs/tho_research_v2_model_m5_structure_candidates`
- 汇总：`runs/tho_research_v2_model_m5_structure_candidates_summary.csv`

选择标准：

1. 主指标：`rr_peak_band_abs_error_mean`，越低越好。
2. 方向护栏：`band_limited_corr_mean` 必须为正；负相关候选不能作为通过模型。
3. 频域护栏：`rr_spec_abs_error_mean` 不应明显恶化。
4. 相对努力：`relative_envelope_mae_mean` 不应明显高于同 seed baseline。
5. 毛刺诊断：raw `rr_peak_abs_error_mean/median` 只用于判断局部尖峰，不作为单独通过标准。

## 文件结构

- 修改：`resp_train/models/unet1d.py`
  - 职责：新增 `UNet1DTinyNoSkipAll`，复用 `ConvBlock` 和现有上下采样结构。
- 修改：`resp_train/models/timeseries.py`
  - 职责：新增 FIR band-pass 前端和 `FIRFrontendPatchMixer1D`。
- 修改：`resp_train/models/registry.py`
  - 职责：注册 `unet1d_tiny_noskip_all` 与 `patch_mixer1d_fir_frontend`。
- 修改：`tests/test_model_registry.py`
  - 职责：覆盖新增模型注册、形状保持和 FIR 前端配置。
- 本地生成：`runs/tho_research_v2_model_m5_structure_candidates/`
  - 职责：保存 M5 训练 run，不进入 Git。
- 本地生成：`runs/tho_research_v2_model_m5_structure_candidates_summary.csv`
  - 职责：保存 M5 汇总表，不进入 Git。
- 修改：`docs/experiments/rawish_state_aligned_l0_l1.md`
  - 职责：追加 M5 结果和阶段判断。

## 任务 1：新增完全去 skip 的 U-Net

**文件：**
- 修改：`resp_train/models/unet1d.py`
- 修改：`resp_train/models/registry.py`
- 修改：`tests/test_model_registry.py`

- [ ] **步骤 1：编写失败测试**

在 `tests/test_model_registry.py` 增加：

```python
def test_unet1d_tiny_noskip_all_is_registered():
    assert "unet1d_tiny_noskip_all" in list_models()


def test_build_unet1d_tiny_noskip_all_preserves_waveform_shape():
    cfg = OmegaConf.create(
        {
            "model": {
                "name": "unet1d_tiny_noskip_all",
                "in_channels": 1,
                "out_channels": 1,
                "base_channels": 4,
            }
        }
    )
    model = build_model(cfg)

    y = model(torch.randn(2, 1, 18001))

    assert y.shape == (2, 1, 18001)
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
./.venv/bin/python -m pytest tests/test_model_registry.py::test_unet1d_tiny_noskip_all_is_registered tests/test_model_registry.py::test_build_unet1d_tiny_noskip_all_preserves_waveform_shape -v
```

预期：失败，提示 `unet1d_tiny_noskip_all` 未注册。

- [ ] **步骤 3：实现 `UNet1DTinyNoSkipAll`**

在 `resp_train/models/unet1d.py` 增加：

```python
class UNet1DTinyNoSkipAll(nn.Module):
    """完全去掉 decoder skip 的一维 U-Net，用于验证高频细节直通假设。"""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 16) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.down1 = nn.Conv1d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.enc2 = ConvBlock(base_channels * 2, base_channels * 2)
        self.down2 = nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 4)
        self.up2 = nn.ConvTranspose1d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.dec2 = ConvBlock(base_channels * 2, base_channels * 2)
        self.up1 = nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.dec1 = ConvBlock(base_channels, base_channels)
        self.out = nn.Conv1d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        z = self.bottleneck(self.down2(e2))
        d2 = self.dec2(UNet1DTiny._match_length(self.up2(z), e2))
        d1 = self.dec1(UNet1DTiny._match_length(self.up1(d2), e1))
        return self.out(d1)
```

在 `resp_train/models/registry.py` 导入并注册：

```python
from resp_train.models.unet1d import UNet1DTiny, UNet1DTinyNoSkip1, UNet1DTinyNoSkipAll

"unet1d_tiny_noskip_all": lambda cfg: UNet1DTinyNoSkipAll(
    in_channels=int(cfg.model.in_channels),
    out_channels=int(cfg.model.out_channels),
    base_channels=int(cfg.model.base_channels),
),
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
./.venv/bin/python -m pytest tests/test_model_registry.py -v
```

预期：`test_model_registry.py` 全部通过。

- [ ] **步骤 5：Commit**

运行：

```bash
git add resp_train/models/unet1d.py resp_train/models/registry.py tests/test_model_registry.py
git commit -m "feat: 添加完全去 skip 的 U-Net 候选"
```

## 任务 2：新增 FIR 前端 PatchMixer

**文件：**
- 修改：`resp_train/models/timeseries.py`
- 修改：`resp_train/models/registry.py`
- 修改：`tests/test_model_registry.py`

- [ ] **步骤 1：编写失败测试**

在 `tests/test_model_registry.py` 增加：

```python
def test_fir_frontend_patch_mixer_is_registered():
    assert "patch_mixer1d_fir_frontend" in list_models()


def test_build_fir_frontend_patch_mixer_preserves_waveform_shape():
    cfg = OmegaConf.create(
        {
            "model": {
                "name": "patch_mixer1d_fir_frontend",
                "in_channels": 1,
                "out_channels": 1,
                "base_channels": 4,
                "patch_len": 128,
                "patch_stride": 64,
                "mixer_layers": 1,
                "overlap_window": "hann",
                "fir_kernel_size": 401,
                "fir_low_hz": 0.05,
                "fir_high_hz": 0.7,
                "fir_sample_rate": 100.0,
                "fir_trainable": True,
            }
        }
    )
    model = build_model(cfg)

    y = model(torch.randn(2, 1, 1025))

    assert y.shape == (2, 1, 1025)
    assert model.fir.weight.requires_grad
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
./.venv/bin/python -m pytest tests/test_model_registry.py::test_fir_frontend_patch_mixer_is_registered tests/test_model_registry.py::test_build_fir_frontend_patch_mixer_preserves_waveform_shape -v
```

预期：失败，提示 `patch_mixer1d_fir_frontend` 未注册。

- [ ] **步骤 3：实现 FIR 前端**

在 `resp_train/models/timeseries.py` 增加：

```python
def _sinc_lowpass(cutoff_hz: float, kernel_size: int, sample_rate: float) -> torch.Tensor:
    kernel_size = _odd_kernel(kernel_size)
    half = kernel_size // 2
    t = torch.arange(-half, half + 1, dtype=torch.float32)
    cutoff = float(cutoff_hz) / float(sample_rate)
    kernel = 2.0 * cutoff * torch.sinc(2.0 * cutoff * t)
    window = torch.hamming_window(kernel_size, periodic=False, dtype=torch.float32)
    kernel = kernel * window
    return kernel / kernel.sum().clamp_min(1e-8)


class FIRBandpass1D(nn.Module):
    """初始化为呼吸频带的 depthwise FIR 前端。"""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 401,
        low_hz: float = 0.05,
        high_hz: float = 0.7,
        sample_rate: float = 100.0,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        kernel_size = _odd_kernel(kernel_size)
        high = _sinc_lowpass(high_hz, kernel_size, sample_rate)
        low = _sinc_lowpass(low_hz, kernel_size, sample_rate)
        band = high - low
        weight = band.view(1, 1, kernel_size).repeat(int(channels), 1, 1)
        self.weight = nn.Parameter(weight, requires_grad=bool(trainable))
        self.padding = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padded = F.pad(x, (self.padding, self.padding), mode="reflect")
        return F.conv1d(padded, self.weight, groups=x.size(1))
```

在同文件增加 wrapper：

```python
class FIRFrontendPatchMixer1D(nn.Module):
    """先用呼吸频带 FIR 前端滤波，再交给 PatchMixer。"""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        patch_len: int = 256,
        patch_stride: int = 128,
        mixer_layers: int = 2,
        overlap_window: str = "hann",
        output_smoothing_kernel: int = 1,
        fir_kernel_size: int = 401,
        fir_low_hz: float = 0.05,
        fir_high_hz: float = 0.7,
        fir_sample_rate: float = 100.0,
        fir_trainable: bool = True,
    ) -> None:
        super().__init__()
        self.fir = FIRBandpass1D(
            channels=in_channels,
            kernel_size=fir_kernel_size,
            low_hz=fir_low_hz,
            high_hz=fir_high_hz,
            sample_rate=fir_sample_rate,
            trainable=fir_trainable,
        )
        self.backbone = PatchMixer1D(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            patch_len=patch_len,
            patch_stride=patch_stride,
            mixer_layers=mixer_layers,
            overlap_window=overlap_window,
            output_smoothing_kernel=output_smoothing_kernel,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.fir(x))
```

在 `resp_train/models/registry.py` 导入并注册：

```python
from resp_train.models.timeseries import DLinearWaveform, FIRFrontendPatchMixer1D, PatchMixer1D, PeriodicUNet1DTiny

"patch_mixer1d_fir_frontend": lambda cfg: FIRFrontendPatchMixer1D(
    in_channels=int(cfg.model.in_channels),
    out_channels=int(cfg.model.out_channels),
    base_channels=int(cfg.model.base_channels),
    patch_len=int(cfg.model.get("patch_len", 256)),
    patch_stride=int(cfg.model.get("patch_stride", 128)),
    mixer_layers=int(cfg.model.get("mixer_layers", 2)),
    overlap_window=str(cfg.model.get("overlap_window", "hann")),
    output_smoothing_kernel=int(cfg.model.get("output_smoothing_kernel", 1)),
    fir_kernel_size=int(cfg.model.get("fir_kernel_size", 401)),
    fir_low_hz=float(cfg.model.get("fir_low_hz", 0.05)),
    fir_high_hz=float(cfg.model.get("fir_high_hz", 0.7)),
    fir_sample_rate=float(cfg.model.get("fir_sample_rate", 100.0)),
    fir_trainable=bool(cfg.model.get("fir_trainable", True)),
),
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
./.venv/bin/python -m pytest tests/test_model_registry.py -v
```

预期：`test_model_registry.py` 全部通过。

- [ ] **步骤 5：Commit**

运行：

```bash
git add resp_train/models/timeseries.py resp_train/models/registry.py tests/test_model_registry.py
git commit -m "feat: 添加 FIR 前端 PatchMixer 候选"
```

## 任务 3：运行 M5 首轮 seed20260650

**文件：**
- 本地生成：`runs/tho_research_v2_model_m5_structure_candidates/`

- [ ] **步骤 1：运行同 seed Patch-Hann baseline**

运行：

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
  --set loss.phase_alignment_weight=0.005 \
  --set training.seed=20260650 \
  --set training.epochs=50 \
  --set training.batch_size=128 \
  --set training.patience=8 \
  --set training.min_delta=0.001 \
  --set training.use_amp=false \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --set outputs.run_root=runs/tho_research_v2_model_m5_structure_candidates
```

预期：写出 baseline run，包含 `metrics.csv` 和 `config.yaml`。

- [ ] **步骤 2：运行 `unet1d_tiny_noskip1`**

运行：

```bash
./.venv/bin/python scripts/train_tho_small.py \
  --config configs/tho_research_v2.yaml \
  --set data.max_train_windows=null \
  --set data.max_val_windows=null \
  --set data.train_sample_seed=20260610 \
  --set data.val_sample_seed=20260611 \
  --set model.name=unet1d_tiny_noskip1 \
  --set loss.high_freq_weight=0.2 \
  --set loss.relative_envelope_weight=0.03 \
  --set loss.phase_alignment_weight=0.005 \
  --set training.seed=20260650 \
  --set training.epochs=50 \
  --set training.batch_size=128 \
  --set training.patience=8 \
  --set training.min_delta=0.001 \
  --set training.use_amp=false \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --set outputs.run_root=runs/tho_research_v2_model_m5_structure_candidates
```

预期：写出 `unet1d_tiny_noskip1` run，包含 `metrics.csv` 和 `config.yaml`。

- [ ] **步骤 3：运行 `unet1d_tiny_noskip_all`**

运行：

```bash
./.venv/bin/python scripts/train_tho_small.py \
  --config configs/tho_research_v2.yaml \
  --set data.max_train_windows=null \
  --set data.max_val_windows=null \
  --set data.train_sample_seed=20260610 \
  --set data.val_sample_seed=20260611 \
  --set model.name=unet1d_tiny_noskip_all \
  --set loss.high_freq_weight=0.2 \
  --set loss.relative_envelope_weight=0.03 \
  --set loss.phase_alignment_weight=0.005 \
  --set training.seed=20260650 \
  --set training.epochs=50 \
  --set training.batch_size=128 \
  --set training.patience=8 \
  --set training.min_delta=0.001 \
  --set training.use_amp=false \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --set outputs.run_root=runs/tho_research_v2_model_m5_structure_candidates
```

预期：写出 `unet1d_tiny_noskip_all` run，包含 `metrics.csv` 和 `config.yaml`。

- [ ] **步骤 4：运行 `patch_mixer1d_fir_frontend`**

运行：

```bash
./.venv/bin/python scripts/train_tho_small.py \
  --config configs/tho_research_v2.yaml \
  --set data.max_train_windows=null \
  --set data.max_val_windows=null \
  --set data.train_sample_seed=20260610 \
  --set data.val_sample_seed=20260611 \
  --set model.name=patch_mixer1d_fir_frontend \
  --set model.patch_len=256 \
  --set model.patch_stride=128 \
  --set model.mixer_layers=2 \
  --set model.overlap_window=hann \
  --set model.output_smoothing_kernel=1 \
  --set model.fir_kernel_size=401 \
  --set model.fir_low_hz=0.05 \
  --set model.fir_high_hz=0.7 \
  --set model.fir_sample_rate=100.0 \
  --set model.fir_trainable=true \
  --set loss.high_freq_weight=0.2 \
  --set loss.relative_envelope_weight=0.03 \
  --set loss.phase_alignment_weight=0.005 \
  --set training.seed=20260650 \
  --set training.epochs=50 \
  --set training.batch_size=128 \
  --set training.patience=8 \
  --set training.min_delta=0.001 \
  --set training.use_amp=false \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --set outputs.run_root=runs/tho_research_v2_model_m5_structure_candidates
```

预期：写出 FIR PatchMixer run，包含 `metrics.csv` 和 `config.yaml`。

## 任务 4：汇总首轮并决定第二 seed 范围

**文件：**
- 本地生成：`runs/tho_research_v2_model_m5_structure_candidates_summary.csv`

- [ ] **步骤 1：生成 summary**

运行：

```bash
./.venv/bin/python scripts/summarize_tho_runs.py \
  --runs-root runs/tho_research_v2_model_m5_structure_candidates \
  --output runs/tho_research_v2_model_m5_structure_candidates_summary.csv
```

预期：summary 至少包含四行，并包含 `selection_task_rr_peak_band_abs_error_mean`。

- [ ] **步骤 2：生成可读对照表**

运行：

```bash
./.venv/bin/python -c "import pandas as pd, yaml, pathlib; df=pd.read_csv('runs/tho_research_v2_model_m5_structure_candidates_summary.csv'); rows=[]; \
for _, r in df.iterrows(): \
    cfg=yaml.safe_load((pathlib.Path(r['run_dir'])/'config.yaml').read_text()); model=cfg['model']; train=cfg['training']; \
    rows.append({'run_id':r['run_id'],'model':model['name'],'seed':train['seed'],'best_val_loss':r['best_val_loss'],'rr_peak_band_mean':r['selection_task_rr_peak_band_abs_error_mean'],'rr_peak_band_median':r['selection_task_rr_peak_band_abs_error_median'],'rr_spec_mean':r['selection_task_rr_spec_abs_error_mean'],'rel_env_mae':r['selection_task_relative_envelope_mae_mean'],'rel_env_corr':r['selection_task_relative_envelope_corr_mean'],'band_corr':r['selection_waveform_band_limited_corr_mean'],'best_lag_corr':r['selection_waveform_best_lag_corr_mean'],'raw_peak_mean':r['selection_waveform_rr_peak_abs_error_mean']}); \
out=pd.DataFrame(rows).sort_values(['seed','model']); out.to_csv('/tmp/m5_model_candidates_table.csv', index=False); print(out.to_string(index=False))"
```

预期：表格能按 model 和 seed 显示 M5 首轮结果。

- [ ] **步骤 3：决定 seed20260660 跑法**

判断：

- 若某个新模型 `band_limited_corr_mean <= 0`，该模型不进入第二 seed。
- 若某个新模型 `rr_peak_band_abs_error_mean` 比同 seed Patch-Hann baseline 高出 `0.08 bpm` 以上，且没有明显 `relative_envelope_mae` 或 raw peak 收益，该模型不进入第二 seed。
- 其余模型进入 seed20260660。
- Patch-Hann baseline 必须进入 seed20260660，用于同 seed 对照。

## 任务 5：运行 seed20260660 复核

**文件：**
- 本地生成：`runs/tho_research_v2_model_m5_structure_candidates/`

- [ ] **步骤 1：运行 Patch-Hann baseline seed20260660**

运行任务 3 步骤 1 的命令，仅把：

```bash
--set training.seed=20260650
```

改为：

```bash
--set training.seed=20260660
```

预期：写出第二个 baseline run。

- [ ] **步骤 2：运行通过首轮筛选的新模型 seed20260660**

对任务 4 步骤 3 保留的每个模型，复用任务 3 中对应命令，仅把：

```bash
--set training.seed=20260650
```

改为：

```bash
--set training.seed=20260660
```

预期：每个保留模型写出第二个 seed 的 run。

- [ ] **步骤 3：重新生成 summary 与对照表**

运行：

```bash
./.venv/bin/python scripts/summarize_tho_runs.py \
  --runs-root runs/tho_research_v2_model_m5_structure_candidates \
  --output runs/tho_research_v2_model_m5_structure_candidates_summary.csv
```

再运行任务 4 步骤 2 的对照表命令。

预期：summary 和 `/tmp/m5_model_candidates_table.csv` 包含第二 seed 结果。

## 任务 6：记录、验证与提交

**文件：**
- 修改：`docs/experiments/rawish_state_aligned_l0_l1.md`

- [ ] **步骤 1：追加 M5 表格**

表格列：

```text
label | run | model | seed | best epoch | best val loss | rr_peak_band_abs_error mean / median | rr_spec_abs_error mean | relative_envelope_mae mean | relative_envelope_corr mean | band_limited_corr mean | best_lag_corr mean | raw rr_peak_abs_error mean | 结论
```

- [ ] **步骤 2：写阶段判断**

判断必须覆盖：

- `unet1d_tiny_noskip1` 与 `unet1d_tiny_noskip_all` 是否支持 skip 泄漏假设。
- `patch_mixer1d_fir_frontend` 是否在主指标、频域护栏、相对努力和 raw peak 之间给出更好的折中。
- 是否进入 M6 `dual_stream_lpf_unet`，以及进入条件：
  - 若 FIR 前端方向优于 Patch-Hann 或 raw peak 明显更稳，M6 优先做 `dual_stream_lpf_unet`。
  - 若 noskip 系列方向优于 Patch-Hann，M6 优先做 gated/limited skip U-Net。
  - 若所有候选均不优于 Patch-Hann baseline，M6 不继续扩大模型复杂度，先回到数据/标签或目标指标诊断。

- [ ] **步骤 3：运行验证**

运行：

```bash
./.venv/bin/python -m pytest tests/test_model_registry.py tests/test_signal_metrics.py tests/test_eval_metrics.py tests/test_diagnostics_scripts.py -v
./.venv/bin/python -m py_compile resp_train/models/unet1d.py resp_train/models/timeseries.py resp_train/models/registry.py
git diff --check
```

预期：pytest 全部通过；py_compile 退出码为 0；diff check 无输出。

- [ ] **步骤 4：提交实验记录**

运行：

```bash
git add docs/experiments/rawish_state_aligned_l0_l1.md
git commit -m "exp: 记录 M5 模型结构候选实验"
```

预期：生成一个只包含 M5 记录更新的提交。

## 验收清单

- [ ] 新增模型都通过 `tests/test_model_registry.py` 的注册和形状测试。
- [ ] M5 不修改 loss 权重与选模口径。
- [ ] 每个训练候选都有同 seed Patch-Hann baseline 对照。
- [ ] 方向护栏 `band_limited_corr_mean > 0` 是通过模型的硬条件。
- [ ] `runs/` 产物不进入 Git。
- [ ] 阶段完成后提交 Git 状态。
