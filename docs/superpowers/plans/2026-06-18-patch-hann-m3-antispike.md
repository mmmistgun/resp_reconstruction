# M3 Patch-Hann 抗毛刺实验计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在 M2 确认 `patch_mixer1d + hann` 更稳定后，降低它的普通局部毛刺和 raw peak 偏高，同时守住带通 RR 与正低频方向。

**架构：** 先给 `PatchMixer1D` 增加可关闭的输出移动平均平滑，默认行为不变；再并行跑两个 M3 候选：更大 overlap 的 `patch_stride=64`，以及轻量输出平滑的 `output_smoothing_kernel=5/11`。所有实验继续以 `rr_peak_band_abs_error` 为主指标，原始 `rr_peak_abs_error` 只做尖峰诊断。

**技术栈：** Python、PyTorch、OmegaConf、pandas、pytest、现有 `resp_train` 训练与评价脚本。

---

## 背景与固定口径

M2 结论：

- `periodic_unet1d_tiny + output_smoothing_kernel=21` 多 seed 不稳定；两个新 seed 均低频反相。
- `patch_mixer1d + overlap_window=hann` 两个 seed 均保持正 `band_limited_corr` 和高 `best_lag_corr`。
- `patch_hann_seed20260620` 的 `rr_peak_band_abs_error_mean=0.486101`，接近反相锚点 `patch_uniform` 的 `0.457180`。
- `patch_hann` 的剩余问题是 raw `rr_peak_abs_error` 偏高，说明普通局部毛刺仍会污染原始峰检测。

本轮沿用：

- 输入：`bcg_rawish_wideband_state_aligned`
- 目标：`tho_waveform_ref`
- 数据窗口：全量
- loss：N3 主线，`high_freq_weight=0.2`，`relative_envelope_weight=0.03`，`phase_alignment_weight=0.005`
- 训练：`epochs=50`，`batch_size=128`，`patience=8`，`min_delta=0.001`，`use_amp=false`
- seed：先用 `training.seed=20260620`，保证与 M2 `patch_hann` 可比
- 输出：`runs/tho_research_v2_patch_hann_m3_antispike`

选择标准：

1. 主指标：`rr_peak_band_abs_error_mean` 不应明显劣于 `patch_hann_seed20260620`。
2. 次护栏：`rr_spec_abs_error_mean` 不应明显恶化。
3. 方向护栏：`band_limited_corr_mean` 必须保持为正。
4. 诊断收益：raw `rr_peak_abs_error_mean`、worst raw peak 窗口和 `row=10196` 应下降。

## 文件结构

- 修改：`resp_train/models/timeseries.py`
  - 职责：给 `PatchMixer1D` 增加 `output_smoothing_kernel` 参数，默认关闭。
- 修改：`resp_train/models/registry.py`
  - 职责：把 `cfg.model.output_smoothing_kernel` 传给 `PatchMixer1D`。
- 修改：`tests/test_model_registry.py`
  - 职责：覆盖 PatchMixer 输出平滑构造、形状保持和局部跳变降低。
- 本地生成：`runs/tho_research_v2_patch_hann_m3_antispike/`
  - 职责：保存 M3 训练 run，不进入 Git。
- 本地生成：`runs/tho_research_v2_patch_hann_m3_antispike_summary.csv`
  - 职责：保存 M3 汇总表，不进入 Git。
- 修改：`docs/experiments/rawish_state_aligned_l0_l1.md`
  - 职责：追加 M3 结果和阶段判断。

## 任务 1：PatchMixer 输出平滑能力

**文件：**
- 修改：`tests/test_model_registry.py`
- 修改：`resp_train/models/timeseries.py`
- 修改：`resp_train/models/registry.py`

- [ ] **步骤 1：编写失败测试**

在 `tests/test_model_registry.py` 增加：

```python
def test_patch_mixer_output_smoothing_reduces_local_jitter():
    raw = PatchMixer1D(
        in_channels=1,
        out_channels=1,
        base_channels=1,
        patch_len=8,
        patch_stride=8,
        mixer_layers=0,
        overlap_window="uniform",
        output_smoothing_kernel=1,
    )
    smooth = PatchMixer1D(
        in_channels=1,
        out_channels=1,
        base_channels=1,
        patch_len=8,
        patch_stride=8,
        mixer_layers=0,
        overlap_window="uniform",
        output_smoothing_kernel=5,
    )
    smooth.load_state_dict(raw.state_dict(), strict=False)
    for model in (raw, smooth):
        model.patch_embed.weight.data.zero_()
        model.patch_embed.bias.data.zero_()
        model.patch_head.weight.data.zero_()
        model.patch_head.bias.data.copy_(torch.tensor([1.0, -1.0] * 4))

    x = torch.zeros(1, 1, 64)
    raw_y = raw(x)
    smooth_y = smooth(x)

    assert smooth_y.shape == raw_y.shape
    assert torch.mean(torch.abs(smooth_y[..., 1:] - smooth_y[..., :-1])).item() < torch.mean(
        torch.abs(raw_y[..., 1:] - raw_y[..., :-1])
    ).item()
```

在 registry 构建测试的 patch mixer 配置里加入：

```python
"output_smoothing_kernel": 5,
```

- [ ] **步骤 2：运行测试验证失败**

```bash
./.venv/bin/python -m pytest tests/test_model_registry.py::test_patch_mixer_output_smoothing_reduces_local_jitter -v
```

预期：失败，提示 `PatchMixer1D.__init__()` 不接受 `output_smoothing_kernel`。

- [ ] **步骤 3：实现最少代码**

`PatchMixer1D.__init__()` 增加参数：

```python
output_smoothing_kernel: int = 1,
```

并增加：

```python
if int(output_smoothing_kernel) > 1:
    self.output_smoother = MovingAverage1D(out_channels, int(output_smoothing_kernel))
else:
    self.output_smoother = nn.Identity()
```

`forward()` 最后一行改为：

```python
return self.output_smoother(self._overlap_add(patch_values, length, padded_length))
```

`registry.py` 的 `PatchMixer1D` factory 增加：

```python
output_smoothing_kernel=int(cfg.model.get("output_smoothing_kernel", 1)),
```

- [ ] **步骤 4：运行模型测试**

```bash
./.venv/bin/python -m pytest tests/test_model_registry.py -v
```

预期：全部通过。

- [ ] **步骤 5：Commit**

```bash
git add resp_train/models/timeseries.py resp_train/models/registry.py tests/test_model_registry.py
git commit -m "feat: 添加 PatchMixer 输出平滑开关"
```

## 任务 2：运行 M3 小网格

**文件：**
- 本地生成：`runs/tho_research_v2_patch_hann_m3_antispike/`

- [ ] **步骤 1：运行 M3c stride64**

```bash
./.venv/bin/python scripts/train_tho_small.py \
  --config configs/tho_research_v2.yaml \
  --set data.max_train_windows=null \
  --set data.max_val_windows=null \
  --set data.train_sample_seed=20260610 \
  --set data.val_sample_seed=20260611 \
  --set model.name=patch_mixer1d \
  --set model.patch_len=256 \
  --set model.patch_stride=64 \
  --set model.mixer_layers=2 \
  --set model.overlap_window=hann \
  --set loss.high_freq_weight=0.2 \
  --set loss.relative_envelope_weight=0.03 \
  --set loss.phase_alignment_weight=0.005 \
  --set training.seed=20260620 \
  --set training.epochs=50 \
  --set training.batch_size=128 \
  --set training.patience=8 \
  --set training.min_delta=0.001 \
  --set training.use_amp=false \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --set outputs.run_root=runs/tho_research_v2_patch_hann_m3_antispike
```

- [ ] **步骤 2：运行 M3a smoothing5**

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
  --set model.output_smoothing_kernel=5 \
  --set loss.high_freq_weight=0.2 \
  --set loss.relative_envelope_weight=0.03 \
  --set loss.phase_alignment_weight=0.005 \
  --set training.seed=20260620 \
  --set training.epochs=50 \
  --set training.batch_size=128 \
  --set training.patience=8 \
  --set training.min_delta=0.001 \
  --set training.use_amp=false \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --set outputs.run_root=runs/tho_research_v2_patch_hann_m3_antispike
```

- [ ] **步骤 3：如 smoothing5 有收益，再运行 smoothing11**

触发条件：`smoothing5` 保持正 `band_limited_corr`，且 raw `rr_peak_abs_error_mean`
低于 M2 `patch_hann_seed20260620`。

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
  --set model.output_smoothing_kernel=11 \
  --set loss.high_freq_weight=0.2 \
  --set loss.relative_envelope_weight=0.03 \
  --set loss.phase_alignment_weight=0.005 \
  --set training.seed=20260620 \
  --set training.epochs=50 \
  --set training.batch_size=128 \
  --set training.patience=8 \
  --set training.min_delta=0.001 \
  --set training.use_amp=false \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --set outputs.run_root=runs/tho_research_v2_patch_hann_m3_antispike
```

- [ ] **步骤 4：生成 summary**

```bash
./.venv/bin/python scripts/summarize_tho_runs.py \
  --runs-root runs/tho_research_v2_patch_hann_m3_antispike \
  --output runs/tho_research_v2_patch_hann_m3_antispike_summary.csv
```

## 任务 3：记录与验证

**文件：**
- 修改：`docs/experiments/rawish_state_aligned_l0_l1.md`

- [ ] **步骤 1：追加 M3 表格**

表格列：

```text
label | run | best epoch | best val loss | rr_peak_band_abs_error mean / median | rr_spec_abs_error mean | relative_envelope_mae mean | relative_envelope_corr mean | band_limited_corr mean | best_lag_corr mean | raw rr_peak_abs_error mean | 结论
```

- [ ] **步骤 2：写阶段判断**

判断规则：

- 若 `patch_stride=64` 改善 raw peak 但明显拖慢训练或恶化带通 RR，则保留为备选，不作为默认。
- 若 `output_smoothing_kernel=5` 同时降低 raw peak 且保持带通 RR/正低频方向，则进入下一轮多 seed。
- 若二者都不能降低 raw peak，则普通毛刺可能不是简单 overlap 或小平滑能解决，需要设计更显式的低自由度/带限 decoder。

- [ ] **步骤 3：运行验证**

```bash
./.venv/bin/python -m pytest tests/test_model_registry.py tests/test_signal_metrics.py tests/test_eval_metrics.py tests/test_diagnostics_scripts.py -v
./.venv/bin/python -m py_compile resp_train/models/timeseries.py resp_train/models/registry.py
git diff --check
```

预期：pytest 全部通过；py_compile 和 diff check 无输出。

- [ ] **步骤 4：Commit**

```bash
git add docs/experiments/rawish_state_aligned_l0_l1.md
git commit -m "exp: 记录 M3 Patch-Hann 抗毛刺实验"
```

## 验收清单

- [ ] `PatchMixer1D` 默认行为不变，`output_smoothing_kernel=1` 等于关闭。
- [ ] M3 至少包含 `patch_stride=64` 和 `output_smoothing_kernel=5` 两个候选。
- [ ] 主选模仍使用 `rr_peak_band_abs_error_mean`。
- [ ] 低频方向必须保持为正。
- [ ] `runs/` 产物不进入 Git。
