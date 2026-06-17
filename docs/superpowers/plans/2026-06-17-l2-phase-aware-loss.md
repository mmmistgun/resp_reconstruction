# L2 Phase-Aware Physiological Loss 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在 L1 低频波形损失未带来整体收益后，新增可微的 phase-invariant alignment loss，让训练目标允许 PSG 胸带参考与 BCG 推断之间存在小范围生理/对齐时移。

**架构：** 复用 L1 已实现的 FFT band-limited waveform 与 lag-aware evaluation。训练侧新增 `phase_lag` 分量：对预测和目标限制到呼吸频带，在 ±`phase_lag_max_sec` 内计算多 lag 相关系数，用 softmax 聚合最佳相关，形成可微的 `1 - soft_best_corr` 损失。cycle consistency 暂不进入训练目标，只在本计划末尾作为 L3 诊断候选记录。

**技术栈：** Python、PyTorch FFT、OmegaConf、pytest、现有 `resp_train` 训练与评价框架。

---

## L1 证据与 L2 切入点

L1 对照使用相同 `4096/1024` 窗口、训练 seed 和验证 seed：

- L0：`runs/tho_research_v2/20260617_120905_055953`
- L1：`runs/tho_research_v2/20260617_121328_250481`
- 差异：`loss.band_waveform_weight` 从 `0.0` 到 `0.2`

收尾诊断：

- `band_limited_corr` 均值从 `0.783169` 降到 `0.778134`
- `best_lag_corr` 均值从 `0.843794` 降到 `0.841708`
- `abs_best_lag_sec` 均值从 `0.173818` 升到 `0.179590`

判断：

- 朴素带限波形 L1 会惩罚相位错位，不能作为下一阶段主线继续加权。
- L2 应把训练目标从固定逐点相位约束改为小范围 lag 容忍的相位/形态约束。
- L1 的 `band_waveform` 可以保留为小权重辅助项，但 L2 pilot 的第一组实验应先单独验证 `phase_lag`。

## 范围与非目标

本计划包含：

- 在 `WeakSyncLoss` 中加入默认关闭的 `phase_lag` 分量。
- 增加单元测试，验证 `phase_lag` 对小范围 shift 不过度惩罚，对超出容忍范围的 shift 仍有惩罚。
- 在默认配置中加入 `phase_lag_weight=0.0` 等参数。
- 给出 L2 pilot 实验命令与结果记录框架。

本计划不包含：

- 不引入 soft-DTW 依赖。
- 不实现 L3 cycle consistency loss。
- 不改模型结构。
- 不改 split 生成策略。
- 不重新解释 L1 为正结果；L1 只作为失败形态证据。

## 文件结构

- 修改：`resp_train/losses/weak.py`
  - 职责：新增可微 soft-lag phase loss，返回 `phase_lag` 分项。
- 修改：`tests/test_losses.py`
  - 职责：覆盖默认关闭、shift 容忍、超范围惩罚和梯度。
- 修改：`configs/tho_research_v2.yaml`
  - 职责：显式加入默认关闭的 L2 loss 参数。
- 修改：`docs/tho_small_training.md`
  - 职责：记录 L2 实验口径和 L1 到 L2 的依据。
- 修改：`scripts/README.md`
  - 职责：补充 L2 pilot 命令。
- 创建：`docs/experiments/l2_phase_aware_loss.md`
  - 职责：记录 L2 结果表和进入 L3 的判定规则。

---

### 任务 1：为 Phase-Lag Loss 编写失败测试

**文件：**
- 修改：`tests/test_losses.py`

- [ ] **步骤 1：扩展分量断言**

把 `tests/test_losses.py::test_weak_sync_loss_returns_components_and_scalar` 中的分量断言改成：

```python
    assert set(parts) == {
        "envelope",
        "spectrum",
        "smooth",
        "high_freq",
        "relative_envelope",
        "band_waveform",
        "phase_lag",
    }
```

- [ ] **步骤 2：扩展可选 loss 权重归零测试**

在 `test_optional_loss_weights_zero_keep_total_loss_unchanged()` 中加入：

```python
    cfg.loss.phase_lag_weight = 0.0
```

并在 expected 中加入：

```python
        + cfg.loss.phase_lag_weight * parts["phase_lag"]
```

- [ ] **步骤 3：新增小范围 shift 容忍测试**

在 `tests/test_losses.py` 新增：

```python
def test_phase_lag_loss_tolerates_small_time_shift():
    cfg = _cfg()
    cfg.loss.envelope_weight = 0.0
    cfg.loss.spectrum_weight = 0.0
    cfg.loss.smooth_weight = 0.0
    cfg.loss.high_freq_weight = 0.0
    cfg.loss.band_waveform_weight = 0.0
    cfg.loss.phase_lag_weight = 1.0
    cfg.loss.phase_lag_max_sec = 1.0
    cfg.loss.phase_lag_step_sec = 0.1
    cfg.loss.phase_lag_temperature = 0.05
    loss_fn = WeakSyncLoss(cfg)
    fs = float(cfg.window.target_fs)
    time = torch.arange(0, 60, 1 / fs)
    target = torch.sin(2 * torch.pi * 0.25 * time).reshape(1, 1, -1)
    shifted = torch.roll(target, shifts=int(round(0.5 * fs)), dims=-1)

    total, parts = loss_fn(shifted, target)

    assert parts["phase_lag"] < 0.05
    assert total < 0.05
```

- [ ] **步骤 4：新增超范围 shift 惩罚测试**

在 `tests/test_losses.py` 新增：

```python
def test_phase_lag_loss_penalizes_shift_outside_tolerance():
    cfg = _cfg()
    cfg.loss.envelope_weight = 0.0
    cfg.loss.spectrum_weight = 0.0
    cfg.loss.smooth_weight = 0.0
    cfg.loss.high_freq_weight = 0.0
    cfg.loss.band_waveform_weight = 0.0
    cfg.loss.phase_lag_weight = 1.0
    cfg.loss.phase_lag_max_sec = 0.5
    cfg.loss.phase_lag_step_sec = 0.1
    cfg.loss.phase_lag_temperature = 0.05
    loss_fn = WeakSyncLoss(cfg)
    fs = float(cfg.window.target_fs)
    time = torch.arange(0, 60, 1 / fs)
    target = torch.sin(2 * torch.pi * 0.25 * time).reshape(1, 1, -1)
    shifted = torch.roll(target, shifts=int(round(1.5 * fs)), dims=-1)

    _, parts = loss_fn(shifted, target)

    assert parts["phase_lag"] > 0.2
```

- [ ] **步骤 5：新增梯度测试**

在 `tests/test_losses.py` 新增：

```python
def test_phase_lag_loss_is_differentiable_when_enabled():
    cfg = _cfg()
    cfg.loss.phase_lag_weight = 0.5
    cfg.loss.phase_lag_max_sec = 0.5
    cfg.loss.phase_lag_step_sec = 0.1
    cfg.loss.phase_lag_temperature = 0.05
    loss_fn = WeakSyncLoss(cfg)
    pred = torch.randn(2, 1, 1800, requires_grad=True)
    target = torch.randn(2, 1, 1800)

    total, parts = loss_fn(pred, target)
    total.backward()

    assert parts["phase_lag"] >= 0
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
```

- [ ] **步骤 6：运行测试验证失败**

运行：

```bash
./.venv/bin/python -m pytest \
  tests/test_losses.py::test_weak_sync_loss_returns_components_and_scalar \
  tests/test_losses.py::test_optional_loss_weights_zero_keep_total_loss_unchanged \
  tests/test_losses.py::test_phase_lag_loss_tolerates_small_time_shift \
  tests/test_losses.py::test_phase_lag_loss_penalizes_shift_outside_tolerance \
  tests/test_losses.py::test_phase_lag_loss_is_differentiable_when_enabled \
  -v
```

预期：FAIL，报错包含 `phase_lag` 分量不存在。

- [ ] **步骤 7：Commit 测试**

运行：

```bash
git add tests/test_losses.py
git commit -m "test: 补充相位容忍损失测试"
```

---

### 任务 2：实现 Soft-Lag Phase Loss

**文件：**
- 修改：`resp_train/losses/weak.py`

- [ ] **步骤 1：读取配置参数**

在 `WeakSyncLoss.__init__()` 中加入：

```python
        self.phase_lag_weight = float(cfg.loss.get("phase_lag_weight", 0.0))
        self.phase_lag_max_sec = float(cfg.loss.get("phase_lag_max_sec", 1.0))
        self.phase_lag_step_sec = float(cfg.loss.get("phase_lag_step_sec", 0.1))
        self.phase_lag_temperature = float(cfg.loss.get("phase_lag_temperature", 0.05))
        if self.phase_lag_max_sec < 0:
            raise ValueError(f"phase_lag_max_sec 必须非负，当前={self.phase_lag_max_sec}")
        if self.phase_lag_step_sec <= 0:
            raise ValueError(f"phase_lag_step_sec 必须为正数，当前={self.phase_lag_step_sec}")
        if self.phase_lag_temperature <= 0:
            raise ValueError(f"phase_lag_temperature 必须为正数，当前={self.phase_lag_temperature}")
```

- [ ] **步骤 2：接入 forward 总损失**

在 `forward()` 中 `band_waveform = ...` 后加入：

```python
        phase_lag = self._phase_lag_loss(pred, target)
```

在 total 中加入：

```python
            + self.phase_lag_weight * phase_lag
```

在返回 parts 中加入：

```python
            "phase_lag": phase_lag.detach(),
```

- [ ] **步骤 3：实现 lag 列表**

在 `WeakSyncLoss` 中加入：

```python
    def _phase_lag_samples(self, n_samples: int) -> list[int]:
        max_lag = int(round(self.phase_lag_max_sec * self.fs))
        step = max(1, int(round(self.phase_lag_step_sec * self.fs)))
        max_lag = min(max_lag, max(0, n_samples - 2))
        lags = list(range(-max_lag, max_lag + 1, step))
        if 0 not in lags:
            lags.append(0)
        return sorted(set(lags))
```

- [ ] **步骤 4：实现单 lag 相关**

在 `WeakSyncLoss` 中加入：

```python
    def _lagged_corr(self, pred_band: torch.Tensor, target_band: torch.Tensor, lag_samples: int) -> torch.Tensor:
        n = pred_band.shape[-1]
        if lag_samples > 0:
            pred_slice = pred_band[..., lag_samples:]
            target_slice = target_band[..., : n - lag_samples]
        elif lag_samples < 0:
            lead = -lag_samples
            pred_slice = pred_band[..., : n - lead]
            target_slice = target_band[..., lead:]
        else:
            pred_slice = pred_band
            target_slice = target_band
        pred_norm = self._zscore(pred_slice)
        target_norm = self._zscore(target_slice)
        return torch.mean(pred_norm * target_norm, dim=-1).squeeze(1)
```

- [ ] **步骤 5：实现 soft-lag phase loss**

在 `WeakSyncLoss` 中加入：

```python
    def _phase_lag_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_band = self._band_limited_waveform(pred)
        target_band = self._band_limited_waveform(target)
        lag_corrs = [self._lagged_corr(pred_band, target_band, lag) for lag in self._phase_lag_samples(pred.shape[-1])]
        corr = torch.stack(lag_corrs, dim=-1)
        weights = torch.softmax(corr / self.phase_lag_temperature, dim=-1)
        soft_best_corr = torch.sum(weights * corr, dim=-1)
        return torch.mean(1.0 - soft_best_corr)
```

实现说明：

- 这是 differentiable lag search，不需要新依赖。
- 正 lag 表示 `pred` 相对 `target` 滞后，和现有 `best_lag_correlation()` 评价语义保持一致。
- 第一版不返回 best lag 到训练日志，只记录 `phase_lag` 标量，避免训练日志膨胀。

- [ ] **步骤 6：运行 loss 测试验证通过**

运行：

```bash
./.venv/bin/python -m pytest tests/test_losses.py -v
```

预期：PASS。

- [ ] **步骤 7：运行训练 smoke 测试验证通过**

运行：

```bash
./.venv/bin/python -m pytest tests/test_engine_smoke.py tests/test_tho_experiment.py -v
```

预期：PASS。

- [ ] **步骤 8：Commit 实现**

运行：

```bash
git add resp_train/losses/weak.py
git commit -m "feat: 增加相位容忍训练损失"
```

---

### 任务 3：更新默认配置与文档口径

**文件：**
- 修改：`configs/tho_research_v2.yaml`
- 修改：`docs/tho_small_training.md`
- 修改：`scripts/README.md`
- 创建：`docs/experiments/l2_phase_aware_loss.md`

- [ ] **步骤 1：默认配置保持关闭**

在 `configs/tho_research_v2.yaml` 的 `loss` 下加入：

```yaml
  phase_lag_weight: 0.0
  phase_lag_max_sec: 1.0
  phase_lag_step_sec: 0.1
  phase_lag_temperature: 0.05
```

- [ ] **步骤 2：补充训练文档**

在 `docs/tho_small_training.md` 的 L1 阶段结论后加入：

```markdown
## 2026-06-17 L2 计划口径：phase-aware training loss

L1 的 `band_waveform_weight=0.2` 没有提升整体 `band_limited_corr` 或 `best_lag_corr`，说明固定相位的带限波形约束会惩罚合理的小范围时移。L2 因此转向可微的 soft-lag phase loss：在呼吸频带内搜索 ±1 秒内的相关性，并用 softmax 聚合形成训练损失。

L2 第一轮只验证 `phase_lag_weight`，不改模型结构；`band_waveform_weight` 只作为后续小权重辅助项候选。
```

- [ ] **步骤 3：补充脚本说明**

在 `scripts/README.md` 的 L1 小节后加入：

```markdown
### L2 phase-aware loss 对照

L2 用于验证小范围 phase/lag 容忍是否比固定相位的带限波形损失更适合胸带呼吸波形恢复。

L2a 只打开 phase-lag loss：

```bash
./.venv/bin/python scripts/train_tho_small.py \
  --config configs/tho_research_v2.yaml \
  --set data.max_train_windows=4096 \
  --set data.max_val_windows=1024 \
  --set training.epochs=3 \
  --set training.device=cuda:0 \
  --set loss.phase_lag_weight=0.2 \
  --set loss.phase_lag_max_sec=1.0 \
  --set loss.phase_lag_step_sec=0.1 \
  --set loss.phase_lag_temperature=0.05 \
  --set outputs.run_root=runs/tho_research_v2_l2_phase_lag_020
```

若 L2a 提升 `best_lag_corr` 或降低 `abs(best_lag_sec)`，再考虑 L2b：
`phase_lag_weight=0.2 + band_waveform_weight=0.05`。如果 L2a 无收益，不进入 L2b。
```

- [ ] **步骤 4：创建 L2 实验记录模板**

创建 `docs/experiments/l2_phase_aware_loss.md`：

```markdown
# L2 Phase-Aware Physiological Loss 实验

## 实验问题

L1 低频波形损失没有整体改善低频相关或最佳 lag 相关。本实验验证 phase-invariant alignment loss 是否能在允许小范围时移的前提下，提高胸带呼吸波形的相位连续性和局部形态。

## 对照 Run

| 实验 | run | phase_lag_weight | band_waveform_weight | best epoch | best val loss | band_limited_corr mean | best_lag_corr mean | abs_best_lag_sec mean | rr_peak_abs_error mean | spectrum_similarity mean | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| L0 | `runs/tho_research_v2/20260617_120905_055953` | 0.0 | 0.0 | 3 | 0.611827 | 0.783169 | 0.843794 | 0.173818 | NaN | NaN | L1 对照基线 |
| L2a | 未运行 | 0.2 | 0.0 | NaN | NaN | NaN | NaN | NaN | NaN | NaN | 未运行 |
| L2b | 未运行 | 0.2 | 0.05 | NaN | NaN | NaN | NaN | NaN | NaN | NaN | 仅 L2a 通过后运行 |

## 判定规则

- L2a 需要提升 `best_lag_corr`，或在不降低 `band_limited_corr` 的前提下降低 `abs_best_lag_sec`。
- 若 L2a 只降低 train/val loss，但 `best_lag_corr`、`band_limited_corr` 和诊断图无改善，不进入 L2b。
- 若 L2a 改善 lag-aware 指标但 RR 或 spectrum 明显恶化，先调低 `phase_lag_weight`，不直接进入 L3。
- L3 cycle consistency 只在 L2 有明确收益后规划。
```

- [ ] **步骤 5：运行文档关键词检查**

运行：

```bash
rg -n "phase_lag|Phase-Aware|L2|soft-lag" configs docs scripts
```

预期：输出包含 `configs/tho_research_v2.yaml`、`docs/tho_small_training.md`、`scripts/README.md` 和 `docs/experiments/l2_phase_aware_loss.md`。

- [ ] **步骤 6：Commit 文档**

运行：

```bash
git add configs/tho_research_v2.yaml docs/tho_small_training.md scripts/README.md docs/experiments/l2_phase_aware_loss.md
git commit -m "docs: 记录相位容忍损失实验方案"
```

---

### 任务 4：L2 Pilot 执行与收尾

**文件：**
- 生成：`runs/tho_research_v2_l2_phase_lag_020/` 下的时间戳 run 子目录
- 生成：`runs/l2_phase_lag_020_summary.csv`
- 修改：`docs/experiments/l2_phase_aware_loss.md`
- 可生成：`runs/tho_research_v2/l2_closeout_20260617/`

- [ ] **步骤 1：运行完整相关测试**

运行：

```bash
./.venv/bin/python -m pytest \
  tests/test_losses.py \
  tests/test_signal_metrics.py \
  tests/test_eval_metrics.py \
  tests/test_engine_smoke.py \
  tests/test_tho_experiment.py \
  -v
```

预期：PASS。

- [ ] **步骤 2：GPU 执行 L2a pilot**

运行：

```bash
./.venv/bin/python scripts/train_tho_small.py \
  --config configs/tho_research_v2.yaml \
  --set data.max_train_windows=4096 \
  --set data.max_val_windows=1024 \
  --set training.epochs=3 \
  --set training.device=cuda:0 \
  --set loss.phase_lag_weight=0.2 \
  --set loss.phase_lag_max_sec=1.0 \
  --set loss.phase_lag_step_sec=0.1 \
  --set loss.phase_lag_temperature=0.05 \
  --set outputs.run_root=runs/tho_research_v2_l2_phase_lag_020
```

预期：退出码 0，run 目录中存在 `config.yaml`、`train_history.csv`、`metrics.csv` 和 `predictions.npz`。

- [ ] **步骤 3：汇总 L2a**

运行：

```bash
./.venv/bin/python scripts/summarize_tho_runs.py \
  --runs-root runs/tho_research_v2_l2_phase_lag_020 \
  --output runs/l2_phase_lag_020_summary.csv
```

预期：退出码 0，`runs/l2_phase_lag_020_summary.csv` 存在。

- [ ] **步骤 4：生成 L2a 诊断图**

运行：

```bash
./.venv/bin/python scripts/plot_tho_predictions.py \
  --run-dir "$(find runs/tho_research_v2_l2_phase_lag_020 -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)" \
  --sort-by best_lag_corr \
  --limit 32
```

预期：退出码 0，run 目录下生成诊断图。

- [ ] **步骤 5：更新实验记录**

根据 L2a 的 `metrics.csv`、`train_history.csv`、`runs/l2_phase_lag_020_summary.csv` 更新 `docs/experiments/l2_phase_aware_loss.md`。

记录必须包含：

- L2a run 路径。
- `phase_lag_weight`、`phase_lag_max_sec`、`phase_lag_step_sec`、`phase_lag_temperature`。
- `band_limited_corr`、`best_lag_corr`、`abs_best_lag_sec`、`rr_peak_abs_error`、`spectrum_similarity`。
- 是否进入 L2b 或 L3 的判断。

- [ ] **步骤 6：每次实验后 git 保存**

运行：

```bash
git add runs/tho_research_v2_l2_phase_lag_020 runs/l2_phase_lag_020_summary.csv docs/experiments/l2_phase_aware_loss.md
git commit -m "exp: 记录 L2 phase lag pilot 结果"
```

此步骤是硬性工作流：完成 L2a 实验后必须保存仓库状态。

---

## 自检清单

- [ ] L2 计划明确基于 L1 失败证据，而不是忽略 L1 结果。
- [ ] 第一版 L2 不引入新第三方依赖。
- [ ] `phase_lag_weight` 默认关闭，现有 L0 行为不漂移。
- [ ] 训练目标只新增 phase-lag loss，不同步推进模型结构。
- [ ] L3 cycle consistency 只作为后续候选，不进入当前实现范围。
- [ ] 每次实验后都有独立 git commit 保存结果。
