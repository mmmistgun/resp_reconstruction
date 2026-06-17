# Phase-Aware STFT Loss 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在 rawish state-aligned 全量口径下，系统评估 phase-aware / lag-tolerant loss 与低频 STFT loss 是否能改善 THO 呼吸任务指标。

**架构：** 先复用现有 `phase_lag_loss` 做 L2 低成本实验，再在 `WeakSyncLoss` 中新增低频 STFT magnitude / phase / complex loss 做 L3 实验。所有实验继续以 `rr_peak_abs_error` 为主护栏，波形相关指标只作诊断；实验产物留在本地 `runs/`，只提交代码、计划和人工台账。

**技术栈：** Python、PyTorch、OmegaConf、pandas、pytest、现有 `resp_train` 训练与评价框架。

---

## 背景与约束

已完成的 fresh batch128 L0/L1 结果记录在：

```text
docs/experiments/rawish_state_aligned_l0_l1.md
```

关键事实：

- L0：`runs/tho_research_v2_patch_mixer_rawish_eff_l0_l1/20260617_173758_049730`
- L1a：`band_waveform_weight=0.05`，低频波形诊断明显改善，但 `rr_peak_abs_error` mean 从 `0.962277` 恶化到 `4.056376`
- L1b：`band_waveform_weight=0.10`，低频波形诊断明显改善，但 `rr_peak_abs_error` mean 恶化到 `4.497422`

因此下一轮不能只追求“波形更像”。所有候选 loss 必须接受任务指标护栏：

- 主护栏：`rr_peak_abs_error_mean <= L0 + 0.15 bpm`
- 主护栏：`rr_peak_abs_error_median <= L0 + 0.05 bpm`
- 次护栏：`rr_spec_abs_error_mean <= L0 + 0.05 bpm`
- 任务辅助：`relative_envelope_mae` 不应明显恶化，`relative_envelope_corr` 优先改善
- 波形诊断：`band_limited_corr`、`best_lag_corr`、`abs(best_lag_sec)` 只用于解释，不单独判定通过

固定训练口径：

- 输入：`bcg_rawish_wideband_state_aligned`
- 目标：`tho_waveform_ref`
- 数据窗口：全量（`data.max_train_windows=null`，`data.max_val_windows=null`）
- 模型：`patch_mixer1d`
- Patch 参数：`patch_len=256`，`patch_stride=128`，`mixer_layers=2`
- 训练：`epochs=50`，`batch_size=128`，`use_amp=false`
- 早停：`patience=8`，`min_delta=0.001`
- seed：`training.seed=20260610`，`data.train_sample_seed=20260610`，`data.val_sample_seed=20260611`
- 基础 loss：`high_freq_weight=0.2`，`relative_envelope_weight=0.01`
- 输出根目录：`runs/tho_research_v2_patch_mixer_rawish_phase_stft_l2_l3`

## 实验分层

### L2：Phase-Aware / Lag-Tolerant

先复用现有 `WeakSyncLoss._phase_lag_loss()`，不新增代码即可开跑。它在 `0.05-0.7Hz`
带限波形上搜索可微 soft-best lag correlation，适合验证“允许小范围非生理时移是否能避免惩罚合理相位差”。

候选：

| label | `phase_lag_weight` | `phase_lag_max_sec` | `phase_lag_step_sec` | `phase_lag_temperature` | `band_waveform_weight` |
|---|---:|---:|---:|---:|---:|
| L2a | 0.01 | 1.0 | 0.2 | 0.10 | 0.0 |
| L2b | 0.03 | 1.0 | 0.2 | 0.10 | 0.0 |
| L2c | 0.01 | 1.0 | 0.2 | 0.10 | 0.005 |

### L3：Low-Frequency STFT

新增低频 STFT loss，只看 `0.05-0.7Hz`，避免高频 BCG 结构主导。

候选：

| label | `stft_mag_weight` | `stft_phase_weight` | `stft_complex_weight` | 说明 |
|---|---:|---:|---:|---|
| L3a | 0.02 | 0.0 | 0.0 | 只验证局部低频幅度结构 |
| L3b | 0.05 | 0.0 | 0.0 | 稍强 magnitude 约束 |
| L3c | 0.02 | 0.005 | 0.0 | 小权重相位一致性 |
| L3d | 0.02 | 0.0 | 0.005 | normalized complex STFT 替代相位分支 |

L3c 与 L3d 不应同时打开。一个是显式 phase loss，一个是 complex loss；同时开会让相位约束过强，不利于判断哪一项造成指标变化。

## 文件结构

- 修改：`resp_train/losses/weak.py`
  - 职责：新增低频 STFT magnitude / phase / complex loss，保持默认权重为 0 时行为不变。
- 修改：`configs/tho_research_v2.yaml`
  - 职责：新增 STFT loss 默认配置项。
- 修改：`tests/test_losses.py`
  - 职责：覆盖 STFT loss 的禁用、频带限制、相位惩罚和可微性。
- 修改：`docs/experiments/rawish_state_aligned_l0_l1.md`
  - 职责：追加 L2/L3 实验入口和阶段记录，避免与 L0/L1 结论混淆。
- 修改：`scripts/README.md`
  - 职责：补充 phase-aware / STFT loss 实验命令和模型选择护栏。
- 本地生成：`runs/tho_research_v2_patch_mixer_rawish_phase_stft_l2_l3/`
  - 职责：保存 L2/L3 训练产物；该目录被 `.gitignore` 忽略，不进入 Git。
- 本地生成：`runs/tho_research_v2_patch_mixer_rawish_phase_stft_l2_l3_summary.csv`
  - 职责：保存 L2/L3 汇总；该文件被 `.gitignore` 忽略，不进入 Git。

---

### 任务 1：为 STFT loss 写失败测试

**文件：**
- 修改：`tests/test_losses.py`

- [ ] **步骤 1：在 `_cfg()` 增加 STFT 默认配置**

将 `_cfg()` 的 `loss` 字典补充为：

```python
"stft_mag_weight": 0.0,
"stft_phase_weight": 0.0,
"stft_complex_weight": 0.0,
"stft_window_sec": 32.0,
"stft_hop_sec": 4.0,
"stft_n_fft_sec": 64.0,
"stft_low_hz": 0.05,
"stft_high_hz": 0.7,
"stft_log_magnitude": True,
"stft_eps": 1e-6,
```

- [ ] **步骤 2：扩展 component 集合断言**

把 `test_weak_sync_loss_returns_components_and_scalar()` 中的 `set(parts)` 改为包含：

```python
"stft_magnitude",
"stft_phase",
"stft_complex",
```

- [ ] **步骤 3：新增禁用权重不改变总 loss 的测试断言**

在 `test_optional_loss_weights_zero_keep_total_loss_unchanged()` 中加入：

```python
cfg.loss.stft_mag_weight = 0.0
cfg.loss.stft_phase_weight = 0.0
cfg.loss.stft_complex_weight = 0.0
```

并把 expected total 扩展为：

```python
+ cfg.loss.stft_mag_weight * parts["stft_magnitude"]
+ cfg.loss.stft_phase_weight * parts["stft_phase"]
+ cfg.loss.stft_complex_weight * parts["stft_complex"]
```

- [ ] **步骤 4：新增 magnitude 只关注低频的测试**

添加测试：

```python
def test_stft_magnitude_loss_ignores_high_frequency_noise_when_band_limited():
    cfg = _cfg()
    cfg.loss.envelope_weight = 0.0
    cfg.loss.spectrum_weight = 0.0
    cfg.loss.smooth_weight = 0.0
    cfg.loss.high_freq_weight = 0.0
    cfg.loss.stft_mag_weight = 1.0
    loss_fn = WeakSyncLoss(cfg)
    fs = float(cfg.window.target_fs)
    time = torch.arange(0, 180, 1 / fs)
    target = torch.sin(2 * torch.pi * 0.25 * time).reshape(1, 1, -1)
    pred_low_ok = target + 0.5 * torch.sin(2 * torch.pi * 8.0 * time).reshape(1, 1, -1)
    pred_low_bad = torch.sin(2 * torch.pi * 0.35 * time).reshape(1, 1, -1)

    _, ok_parts = loss_fn(pred_low_ok, target)
    _, bad_parts = loss_fn(pred_low_bad, target)

    assert ok_parts["stft_magnitude"] < bad_parts["stft_magnitude"] * 0.5
```

- [ ] **步骤 5：新增 phase loss 惩罚同频相位偏移的测试**

添加测试：

```python
def test_stft_phase_loss_penalizes_same_frequency_phase_shift():
    cfg = _cfg()
    cfg.loss.envelope_weight = 0.0
    cfg.loss.spectrum_weight = 0.0
    cfg.loss.smooth_weight = 0.0
    cfg.loss.high_freq_weight = 0.0
    cfg.loss.stft_phase_weight = 1.0
    loss_fn = WeakSyncLoss(cfg)
    fs = float(cfg.window.target_fs)
    time = torch.arange(0, 180, 1 / fs)
    target = torch.sin(2 * torch.pi * 0.25 * time).reshape(1, 1, -1)
    same = target.clone()
    shifted = torch.sin(2 * torch.pi * 0.25 * time + torch.pi / 2).reshape(1, 1, -1)

    _, same_parts = loss_fn(same, target)
    _, shifted_parts = loss_fn(shifted, target)

    assert same_parts["stft_phase"] < 0.05
    assert shifted_parts["stft_phase"] > same_parts["stft_phase"] + 0.2
```

- [ ] **步骤 6：新增 complex loss 可微性测试**

添加测试：

```python
def test_stft_complex_loss_is_differentiable_when_enabled():
    cfg = _cfg()
    cfg.loss.stft_complex_weight = 0.01
    loss_fn = WeakSyncLoss(cfg)
    pred = torch.randn(2, 1, 18000, requires_grad=True)
    target = torch.randn(2, 1, 18000)

    total, parts = loss_fn(pred, target)
    total.backward()

    assert parts["stft_complex"] >= 0
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
```

- [ ] **步骤 7：运行测试确认失败**

运行：

```bash
./.venv/bin/python -m pytest tests/test_losses.py -v
```

预期：至少 1 个测试失败，报错应指向 `stft_*` component 或配置属性尚未实现。

- [ ] **步骤 8：Commit 测试**

```bash
git add tests/test_losses.py
git commit -m "test: 补充低频 STFT 损失测试"
```

---

### 任务 2：实现低频 STFT loss

**文件：**
- 修改：`resp_train/losses/weak.py`

- [ ] **步骤 1：在 `__init__` 读取 STFT 配置并校验**

在 `WeakSyncLoss.__init__()` 中 `phase_lag_temperature` 后加入：

```python
self.stft_mag_weight = float(cfg.loss.get("stft_mag_weight", 0.0))
self.stft_phase_weight = float(cfg.loss.get("stft_phase_weight", 0.0))
self.stft_complex_weight = float(cfg.loss.get("stft_complex_weight", 0.0))
self.stft_window_sec = float(cfg.loss.get("stft_window_sec", 32.0))
self.stft_hop_sec = float(cfg.loss.get("stft_hop_sec", 4.0))
self.stft_n_fft_sec = float(cfg.loss.get("stft_n_fft_sec", 64.0))
self.stft_low_hz = float(cfg.loss.get("stft_low_hz", cfg.loss.spectrum_low_hz))
self.stft_high_hz = float(cfg.loss.get("stft_high_hz", cfg.loss.spectrum_high_hz))
self.stft_log_magnitude = bool(cfg.loss.get("stft_log_magnitude", True))
self.stft_eps = float(cfg.loss.get("stft_eps", 1e-6))
```

在已有 phase lag 校验后加入：

```python
for name, value in (
    ("stft_mag_weight", self.stft_mag_weight),
    ("stft_phase_weight", self.stft_phase_weight),
    ("stft_complex_weight", self.stft_complex_weight),
):
    if value < 0:
        raise ValueError(f"{name} 必须非负，当前={value}")
if self.stft_window_sec <= 0:
    raise ValueError(f"stft_window_sec 必须为正数，当前={self.stft_window_sec}")
if self.stft_hop_sec <= 0:
    raise ValueError(f"stft_hop_sec 必须为正数，当前={self.stft_hop_sec}")
if self.stft_n_fft_sec <= 0:
    raise ValueError(f"stft_n_fft_sec 必须为正数，当前={self.stft_n_fft_sec}")
if not 0 < self.stft_low_hz < self.stft_high_hz < self.fs / 2:
    raise ValueError(
        f"STFT 频带非法: low={self.stft_low_hz} high={self.stft_high_hz} fs={self.fs}"
    )
if self.stft_eps <= 0:
    raise ValueError(f"stft_eps 必须为正数，当前={self.stft_eps}")
```

- [ ] **步骤 2：在 `forward()` 接入新 loss**

在 `phase_lag` 计算后加入：

```python
if self.stft_mag_weight > 0 or self.stft_phase_weight > 0 or self.stft_complex_weight > 0:
    pred_stft, target_stft = self._low_frequency_stft_pair(pred_loss, target_loss)
    stft_magnitude = self._stft_magnitude_loss(pred_stft, target_stft)
    stft_phase = self._stft_phase_loss(pred_stft, target_stft)
    stft_complex = self._stft_complex_loss(pred_stft, target_stft)
else:
    stft_magnitude = pred_loss.new_tensor(0.0)
    stft_phase = pred_loss.new_tensor(0.0)
    stft_complex = pred_loss.new_tensor(0.0)
```

把 `total` 扩展为：

```python
+ self.stft_mag_weight * stft_magnitude
+ self.stft_phase_weight * stft_phase
+ self.stft_complex_weight * stft_complex
```

把返回 parts 扩展为：

```python
"stft_magnitude": stft_magnitude.detach(),
"stft_phase": stft_phase.detach(),
"stft_complex": stft_complex.detach(),
```

- [ ] **步骤 3：新增 STFT helper**

在 `_phase_lag_loss()` 后加入：

```python
def _low_frequency_stft_pair(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pred_stft = self._low_frequency_stft(pred)
    target_stft = self._low_frequency_stft(target)
    return pred_stft, target_stft

def _low_frequency_stft(self, x: torch.Tensor) -> torch.Tensor:
    centered = self._center(x).squeeze(1)
    n_samples = centered.shape[-1]
    n_fft = min(n_samples, max(16, int(round(self.stft_n_fft_sec * self.fs))))
    win_length = min(n_fft, max(8, int(round(self.stft_window_sec * self.fs))))
    hop_length = max(1, int(round(self.stft_hop_sec * self.fs)))
    window = torch.hann_window(win_length, device=x.device, dtype=centered.dtype)
    spectrum = torch.stft(
        centered,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / self.fs).to(x.device)
    mask = (freqs >= self.stft_low_hz) & (freqs <= self.stft_high_hz)
    if not bool(mask.any()):
        raise ValueError(
            f"STFT 频带为空: low_hz={self.stft_low_hz} high_hz={self.stft_high_hz} "
            f"fs={self.fs} n_fft={n_fft}"
        )
    return spectrum[:, mask, :]
```

- [ ] **步骤 4：新增 magnitude / phase / complex loss**

继续加入：

```python
def _stft_magnitude_loss(self, pred_stft: torch.Tensor, target_stft: torch.Tensor) -> torch.Tensor:
    pred_mag = pred_stft.abs()
    target_mag = target_stft.abs()
    if self.stft_log_magnitude:
        pred_mag = torch.log1p(pred_mag)
        target_mag = torch.log1p(target_mag)
    return torch.mean(torch.abs(pred_mag - target_mag))

def _stft_phase_loss(self, pred_stft: torch.Tensor, target_stft: torch.Tensor) -> torch.Tensor:
    target_mag = target_stft.abs()
    weights = target_mag / torch.clamp(target_mag.sum(dim=(1, 2), keepdim=True), min=self.stft_eps)
    phase_diff = torch.angle(pred_stft * target_stft.conj())
    return torch.mean(torch.sum(weights * (1.0 - torch.cos(phase_diff)), dim=(1, 2)))

def _stft_complex_loss(self, pred_stft: torch.Tensor, target_stft: torch.Tensor) -> torch.Tensor:
    pred_scale = torch.clamp(pred_stft.abs().mean(dim=(1, 2), keepdim=True), min=self.stft_eps)
    target_scale = torch.clamp(target_stft.abs().mean(dim=(1, 2), keepdim=True), min=self.stft_eps)
    pred_norm = pred_stft / pred_scale
    target_norm = target_stft / target_scale
    return torch.mean(torch.abs(pred_norm - target_norm))
```

- [ ] **步骤 5：运行 loss 测试**

运行：

```bash
./.venv/bin/python -m pytest tests/test_losses.py -v
```

预期：全部通过，CUDA AMP 测试在无 CUDA 环境可 skip。

- [ ] **步骤 6：运行编译检查**

运行：

```bash
./.venv/bin/python -m py_compile resp_train/losses/weak.py tests/test_losses.py
```

预期：无输出且退出码为 `0`。

- [ ] **步骤 7：Commit 实现**

```bash
git add resp_train/losses/weak.py tests/test_losses.py
git commit -m "feat: 添加低频 STFT 损失"
```

---

### 任务 3：补充配置与文档入口

**文件：**
- 修改：`configs/tho_research_v2.yaml`
- 修改：`scripts/README.md`
- 修改：`docs/experiments/rawish_state_aligned_l0_l1.md`

- [ ] **步骤 1：在配置文件增加默认项**

在 `configs/tho_research_v2.yaml` 的 `loss:` 下加入：

```yaml
  stft_mag_weight: 0.0
  stft_phase_weight: 0.0
  stft_complex_weight: 0.0
  stft_window_sec: 32.0
  stft_hop_sec: 4.0
  stft_n_fft_sec: 64.0
  stft_low_hz: 0.05
  stft_high_hz: 0.7
  stft_log_magnitude: true
  stft_eps: 1.0e-6
```

- [ ] **步骤 2：在 README 增加 L2/L3 命令入口**

在 `scripts/README.md` 的 L1 段落后加入：

````markdown
### L2/L3 Phase-aware 与 STFT loss

L2/L3 继续使用 `bcg_rawish_wideband_state_aligned` 全量 batch128 口径。
模型选择仍以 `rr_peak_abs_error` 为主护栏；STFT/phase/complex loss 只允许
在不破坏 RR 任务指标的情况下改善低频形态。

L2 先复用现有 lag-tolerant loss：

```bash
for w in 0.01 0.03; do
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
    --set loss.high_freq_weight=0.2 \
    --set loss.relative_envelope_weight=0.01 \
    --set loss.band_waveform_weight=0.0 \
    --set loss.phase_lag_weight="$w" \
    --set loss.phase_lag_max_sec=1.0 \
    --set loss.phase_lag_step_sec=0.2 \
    --set loss.phase_lag_temperature=0.10 \
    --set training.epochs=50 \
    --set training.batch_size=128 \
    --set training.patience=8 \
    --set training.min_delta=0.001 \
    --set training.use_amp=false \
    --set training.device=cuda:0 \
    --set training.show_progress=false \
    --set outputs.run_root=runs/tho_research_v2_patch_mixer_rawish_phase_stft_l2_l3
done
```

L3 再开启低频 STFT magnitude / phase / complex 分支。
````

- [ ] **步骤 3：在实验台账增加 L2/L3 预注册段落**

在 `docs/experiments/rawish_state_aligned_l0_l1.md` 末尾加入：

```markdown
## L2/L3 预注册

下一轮实验输出到 `runs/tho_research_v2_patch_mixer_rawish_phase_stft_l2_l3`。
L2 复用 `phase_lag_loss`，L3 新增低频 STFT magnitude / phase / complex loss。
候选 run 只有在 `rr_peak_abs_error` 主护栏不明显恶化时，才允许根据波形诊断继续比较。
```

- [ ] **步骤 4：运行文档关键词检查**

运行：

```bash
rg -n "stft_mag_weight|phase_lag_weight|phase_stft_l2_l3|rr_peak_abs_error" configs/tho_research_v2.yaml scripts/README.md docs/experiments/rawish_state_aligned_l0_l1.md
```

预期：三个文件均能检索到相应关键词。

- [ ] **步骤 5：Commit 配置与文档**

```bash
git add configs/tho_research_v2.yaml scripts/README.md docs/experiments/rawish_state_aligned_l0_l1.md
git commit -m "docs: 记录 phase-aware STFT 实验入口"
```

---

### 任务 4：运行 L2 phase-aware 实验

**文件：**
- 本地生成：`runs/tho_research_v2_patch_mixer_rawish_phase_stft_l2_l3/`

- [ ] **步骤 1：确认 GPU 可见**

运行：

```bash
./.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

预期：输出 `True` 且 device count 大于 `0`。如果输出 `False`，停止并报告，不要用 CPU 代跑正式实验。

- [ ] **步骤 2：并行启动 L2a/L2b/L2c**

如果有多张 GPU，使用 `CUDA_VISIBLE_DEVICES=<物理 GPU id>` 隔离每个训练进程，并在进程内继续设置 `training.device=cuda:0`。

L2a：

```bash
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python scripts/train_tho_small.py \
  --config configs/tho_research_v2.yaml \
  --set data.max_train_windows=null \
  --set data.max_val_windows=null \
  --set data.train_sample_seed=20260610 \
  --set data.val_sample_seed=20260611 \
  --set model.name=patch_mixer1d \
  --set model.patch_len=256 \
  --set model.patch_stride=128 \
  --set model.mixer_layers=2 \
  --set loss.high_freq_weight=0.2 \
  --set loss.relative_envelope_weight=0.01 \
  --set loss.band_waveform_weight=0.0 \
  --set loss.phase_lag_weight=0.01 \
  --set loss.phase_lag_max_sec=1.0 \
  --set loss.phase_lag_step_sec=0.2 \
  --set loss.phase_lag_temperature=0.10 \
  --set training.epochs=50 \
  --set training.batch_size=128 \
  --set training.patience=8 \
  --set training.min_delta=0.001 \
  --set training.use_amp=false \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --set outputs.run_root=runs/tho_research_v2_patch_mixer_rawish_phase_stft_l2_l3
```

L2b：

```bash
CUDA_VISIBLE_DEVICES=1 ./.venv/bin/python scripts/train_tho_small.py \
  --config configs/tho_research_v2.yaml \
  --set data.max_train_windows=null \
  --set data.max_val_windows=null \
  --set data.train_sample_seed=20260610 \
  --set data.val_sample_seed=20260611 \
  --set model.name=patch_mixer1d \
  --set model.patch_len=256 \
  --set model.patch_stride=128 \
  --set model.mixer_layers=2 \
  --set loss.high_freq_weight=0.2 \
  --set loss.relative_envelope_weight=0.01 \
  --set loss.band_waveform_weight=0.0 \
  --set loss.phase_lag_weight=0.03 \
  --set loss.phase_lag_max_sec=1.0 \
  --set loss.phase_lag_step_sec=0.2 \
  --set loss.phase_lag_temperature=0.10 \
  --set training.epochs=50 \
  --set training.batch_size=128 \
  --set training.patience=8 \
  --set training.min_delta=0.001 \
  --set training.use_amp=false \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --set outputs.run_root=runs/tho_research_v2_patch_mixer_rawish_phase_stft_l2_l3
```

L2c：

```bash
CUDA_VISIBLE_DEVICES=2 ./.venv/bin/python scripts/train_tho_small.py \
  --config configs/tho_research_v2.yaml \
  --set data.max_train_windows=null \
  --set data.max_val_windows=null \
  --set data.train_sample_seed=20260610 \
  --set data.val_sample_seed=20260611 \
  --set model.name=patch_mixer1d \
  --set model.patch_len=256 \
  --set model.patch_stride=128 \
  --set model.mixer_layers=2 \
  --set loss.high_freq_weight=0.2 \
  --set loss.relative_envelope_weight=0.01 \
  --set loss.band_waveform_weight=0.005 \
  --set loss.phase_lag_weight=0.01 \
  --set loss.phase_lag_max_sec=1.0 \
  --set loss.phase_lag_step_sec=0.2 \
  --set loss.phase_lag_temperature=0.10 \
  --set training.epochs=50 \
  --set training.batch_size=128 \
  --set training.patience=8 \
  --set training.min_delta=0.001 \
  --set training.use_amp=false \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --set outputs.run_root=runs/tho_research_v2_patch_mixer_rawish_phase_stft_l2_l3
```

预期：每个命令最终输出一个 run 目录，且目录内包含 `config.yaml`、`train_history.csv`、`metrics.csv`、`checkpoint.pt`。

- [ ] **步骤 3：汇总 L2**

运行：

```bash
./.venv/bin/python scripts/summarize_tho_runs.py \
  --runs-root runs/tho_research_v2_patch_mixer_rawish_phase_stft_l2_l3 \
  --output runs/tho_research_v2_patch_mixer_rawish_phase_stft_l2_l3_summary.csv
```

预期：输出 `rows=3` 或更多，具体取决于同目录内是否已有后续 L3 run。

- [ ] **步骤 4：打印 L2 护栏表**

运行：

```bash
./.venv/bin/python - <<'PY'
from pathlib import Path
import pandas as pd
root = Path("runs/tho_research_v2_patch_mixer_rawish_phase_stft_l2_l3")
rows = []
for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    cfg = (run_dir / "config.yaml").read_text()
    if "phase_lag_weight: 0.0" in cfg and "stft_mag_weight: 0.0" in cfg:
        continue
    metrics = pd.read_csv(run_dir / "metrics.csv")
    history = pd.read_csv(run_dir / "train_history.csv")
    rows.append({
        "run_id": run_dir.name,
        "best_val_loss": float(history["val_loss"].min()),
        "rr_peak_mean": float(metrics["rr_peak_abs_error"].mean()),
        "rr_peak_median": float(metrics["rr_peak_abs_error"].median()),
        "rr_spec_mean": float(metrics["rr_spec_abs_error"].mean()),
        "rel_env_mae": float(metrics["relative_envelope_mae"].mean()),
        "rel_env_corr": float(metrics["relative_envelope_corr"].mean()),
        "band_corr": float(metrics["band_limited_corr"].mean()),
        "best_lag_corr": float(metrics["best_lag_corr"].mean()),
        "abs_best_lag_sec": float(metrics["best_lag_sec"].abs().mean()),
    })
print(pd.DataFrame(rows).to_string(index=False))
PY
```

- [ ] **步骤 5：Commit L2 结果台账**

把 L2 表格和阶段判断写入 `docs/experiments/rawish_state_aligned_l0_l1.md`，然后提交：

```bash
git add docs/experiments/rawish_state_aligned_l0_l1.md
git commit -m "exp: 记录 phase-aware L2 实验"
```

---

### 任务 5：运行 L3 STFT 实验

**文件：**
- 本地生成：`runs/tho_research_v2_patch_mixer_rawish_phase_stft_l2_l3/`
- 修改：`docs/experiments/rawish_state_aligned_l0_l1.md`

- [ ] **步骤 1：并行启动 L3a/L3b**

L3a：

```bash
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python scripts/train_tho_small.py \
  --config configs/tho_research_v2.yaml \
  --set data.max_train_windows=null \
  --set data.max_val_windows=null \
  --set data.train_sample_seed=20260610 \
  --set data.val_sample_seed=20260611 \
  --set model.name=patch_mixer1d \
  --set model.patch_len=256 \
  --set model.patch_stride=128 \
  --set model.mixer_layers=2 \
  --set loss.high_freq_weight=0.2 \
  --set loss.relative_envelope_weight=0.01 \
  --set loss.band_waveform_weight=0.0 \
  --set loss.phase_lag_weight=0.0 \
  --set loss.stft_mag_weight=0.02 \
  --set loss.stft_phase_weight=0.0 \
  --set loss.stft_complex_weight=0.0 \
  --set training.epochs=50 \
  --set training.batch_size=128 \
  --set training.patience=8 \
  --set training.min_delta=0.001 \
  --set training.use_amp=false \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --set outputs.run_root=runs/tho_research_v2_patch_mixer_rawish_phase_stft_l2_l3
```

L3b：

```bash
CUDA_VISIBLE_DEVICES=1 ./.venv/bin/python scripts/train_tho_small.py \
  --config configs/tho_research_v2.yaml \
  --set data.max_train_windows=null \
  --set data.max_val_windows=null \
  --set data.train_sample_seed=20260610 \
  --set data.val_sample_seed=20260611 \
  --set model.name=patch_mixer1d \
  --set model.patch_len=256 \
  --set model.patch_stride=128 \
  --set model.mixer_layers=2 \
  --set loss.high_freq_weight=0.2 \
  --set loss.relative_envelope_weight=0.01 \
  --set loss.band_waveform_weight=0.0 \
  --set loss.phase_lag_weight=0.0 \
  --set loss.stft_mag_weight=0.05 \
  --set loss.stft_phase_weight=0.0 \
  --set loss.stft_complex_weight=0.0 \
  --set training.epochs=50 \
  --set training.batch_size=128 \
  --set training.patience=8 \
  --set training.min_delta=0.001 \
  --set training.use_amp=false \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --set outputs.run_root=runs/tho_research_v2_patch_mixer_rawish_phase_stft_l2_l3
```

- [ ] **步骤 2：如果 L3a/L3b 未触发主护栏，再启动 L3c/L3d**

L3c：

```bash
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python scripts/train_tho_small.py \
  --config configs/tho_research_v2.yaml \
  --set data.max_train_windows=null \
  --set data.max_val_windows=null \
  --set data.train_sample_seed=20260610 \
  --set data.val_sample_seed=20260611 \
  --set model.name=patch_mixer1d \
  --set model.patch_len=256 \
  --set model.patch_stride=128 \
  --set model.mixer_layers=2 \
  --set loss.high_freq_weight=0.2 \
  --set loss.relative_envelope_weight=0.01 \
  --set loss.band_waveform_weight=0.0 \
  --set loss.phase_lag_weight=0.0 \
  --set loss.stft_mag_weight=0.02 \
  --set loss.stft_phase_weight=0.005 \
  --set loss.stft_complex_weight=0.0 \
  --set training.epochs=50 \
  --set training.batch_size=128 \
  --set training.patience=8 \
  --set training.min_delta=0.001 \
  --set training.use_amp=false \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --set outputs.run_root=runs/tho_research_v2_patch_mixer_rawish_phase_stft_l2_l3
```

L3d：

```bash
CUDA_VISIBLE_DEVICES=1 ./.venv/bin/python scripts/train_tho_small.py \
  --config configs/tho_research_v2.yaml \
  --set data.max_train_windows=null \
  --set data.max_val_windows=null \
  --set data.train_sample_seed=20260610 \
  --set data.val_sample_seed=20260611 \
  --set model.name=patch_mixer1d \
  --set model.patch_len=256 \
  --set model.patch_stride=128 \
  --set model.mixer_layers=2 \
  --set loss.high_freq_weight=0.2 \
  --set loss.relative_envelope_weight=0.01 \
  --set loss.band_waveform_weight=0.0 \
  --set loss.phase_lag_weight=0.0 \
  --set loss.stft_mag_weight=0.02 \
  --set loss.stft_phase_weight=0.0 \
  --set loss.stft_complex_weight=0.005 \
  --set training.epochs=50 \
  --set training.batch_size=128 \
  --set training.patience=8 \
  --set training.min_delta=0.001 \
  --set training.use_amp=false \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --set outputs.run_root=runs/tho_research_v2_patch_mixer_rawish_phase_stft_l2_l3
```

如果 L3a/L3b 已明显破坏 `rr_peak_abs_error`，停止 L3c/L3d，不继续叠加相位或 complex 约束。

- [ ] **步骤 3：汇总 L3**

运行：

```bash
./.venv/bin/python scripts/summarize_tho_runs.py \
  --runs-root runs/tho_research_v2_patch_mixer_rawish_phase_stft_l2_l3 \
  --output runs/tho_research_v2_patch_mixer_rawish_phase_stft_l2_l3_summary.csv
```

- [ ] **步骤 4：写入 L3 结论**

在 `docs/experiments/rawish_state_aligned_l0_l1.md` 追加：

```markdown
## L3 Low-Frequency STFT

| run | label | STFT mag | STFT phase | STFT complex | best val loss | rr peak mean / median | rr spec mean | relative envelope MAE / corr | band corr | best lag corr | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
```

结论必须明确：

- 是否通过 `rr_peak_abs_error` 主护栏
- STFT magnitude 是否比 L2 更稳
- phase 或 complex 是否造成额外 RR 恶化
- 是否值得进入下一轮模型结构实验

- [ ] **步骤 5：Commit L3 结果台账**

```bash
git add docs/experiments/rawish_state_aligned_l0_l1.md
git commit -m "exp: 记录低频 STFT L3 实验"
```

---

## 并行执行建议

如果 GPU 和显存足够，可按以下批次并行：

1. L2a、L2b、L2c 同时跑。
2. L3a、L3b 同时跑。
3. L3c、L3d 只在 L3a/L3b 未触发主护栏时同时跑。

不要把 L2 和 L3 第一批混在一起启动：L3 依赖新增代码，L2 可先用现有 phase lag 低成本验证；分批更容易定位失败原因。

若使用多张 GPU：

把物理 GPU 0 暴露给进程后，进程内部仍然用 `cuda:0`：

```bash
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python scripts/train_tho_small.py --config configs/tho_research_v2.yaml --set training.device=cuda:0
```

把物理 GPU 1 暴露给另一个进程后，进程内部也继续用 `cuda:0`：

```bash
CUDA_VISIBLE_DEVICES=1 ./.venv/bin/python scripts/train_tho_small.py --config configs/tho_research_v2.yaml --set training.device=cuda:0
```

若使用同一张大显存 GPU 并行多个进程，必须先确认显存余量；如果出现 OOM，应降低并行度，不改 batch size 来保留与 L0/L1 的可比性。

## 最终验收

- [ ] `tests/test_losses.py` 覆盖 STFT magnitude / phase / complex。
- [ ] 默认配置下 STFT loss 权重全为 `0.0`，不改变现有 L0/L1 行为。
- [ ] L2/L3 训练命令都使用 `bcg_rawish_wideband_state_aligned`、全量窗口、batch128、同一 seed。
- [ ] 每个正式 run 完成后都有 `metrics.csv` 和可解释台账。
- [ ] `runs/` 产物不进入 Git。
- [ ] 每完成一个紧密实验批次提交一次仓库状态。
- [ ] 只有未触发 `rr_peak_abs_error` 主护栏的候选，才允许作为下一轮模型结构实验依据。
