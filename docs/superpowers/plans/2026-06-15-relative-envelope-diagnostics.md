# 相对包络诊断与约束实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在不引入绝对幅度标定 loss 的前提下，先新增相对包络变化诊断指标，判断模型是否错过目标中的相对下降/增强；只有证据成立时，再加入很小权重的 `relative_envelope_loss`。

**架构：** 保留当前 z-score envelope corr 作为主包络一致性指标，不约束预测和目标的绝对幅度。新增诊断指标只比较局部包络相对自身基线的变化轨迹，先进入评估 CSV 和汇总脚本；训练 loss 的改动放在第二阶段，并由诊断结果触发。

**技术栈：** Python、NumPy、PyTorch、pandas、OmegaConf、pytest、现有 `resp_train.metrics` / `resp_train.losses` / `scripts` 工具链。

---

## 背景与约束

当前 run：

`runs/tho_research_v2_patch_mixer_rawish_hfpenalty_gpu_es50/20260614_171417_631400`

观察结论：

- 模型已经显著改善呼吸率和频谱指标，说明主频建模有效。
- 诊断图显示预测更像窄带、规整的振荡器，可能抹平了目标中的相对包络增强/下降。
- 当前不需要绝对幅度标定，因为任务目标不要求预测恢复 THO 的真实物理幅度。
- 现有 z-score envelope corr 仍保留，但它对局部增强/下降不够精细。

必须遵守：

- 不新增绝对幅度标定 loss。
- 不把原始 `std_ratio`、绝对能量比、原始幅度 MSE 作为训练目标。
- 第一阶段只加诊断指标和可视化/汇总，不改变训练行为。
- 第二阶段只有在诊断证明模型漏掉相对包络变化时才执行。
- 任何新增配置默认值必须保持现有训练行为不变。

## 文件结构

- 修改：`resp_train/metrics/signal.py`
  - 新增相对包络特征与诊断指标函数。
  - 只使用归一化后的包络变化，不暴露绝对幅度训练目标。
- 修改：`resp_train/metrics/evaluate.py`
  - 在逐窗口 metrics 中写入相对包络诊断列。
- 修改：`tests/test_signal_metrics.py`
  - 覆盖相对包络指标的行为、尺度不变性和退化输入。
- 修改：`tests/test_eval_metrics.py`
  - 确认评估 CSV 包含新诊断列。
- 修改：`scripts/plot_tho_predictions.py`
  - 在诊断图指标文本中显示关键相对包络指标，辅助人工复核。
- 修改：`scripts/summarize_tho_runs.py`
  - 把新指标纳入 run 汇总，支持跨实验对比。
- 修改：`resp_train/losses/weak.py`
  - 第二阶段才新增 `relative_envelope_loss`，默认权重 `0.0`。
- 修改：`tests/test_losses.py`
  - 第二阶段才覆盖新增 loss 分量和默认不改变行为。
- 修改：`configs/tho_research_v2_patch_mixer.yaml`
  - 第二阶段才加入 `relative_envelope_weight: 0.0` 默认项。

---

## 阶段一：相对包络诊断指标

### 任务 1：实现相对包络诊断函数

**文件：**
- 修改：`resp_train/metrics/signal.py`
- 测试：`tests/test_signal_metrics.py`

- [ ] **步骤 1：编写失败测试：相对包络指标对绝对幅度缩放不敏感**

在 `tests/test_signal_metrics.py` 的 import 中加入 `relative_envelope_metrics`：

```python
from resp_train.metrics.signal import (
    bandpass_filter,
    estimate_peak_rate_bpm,
    estimate_spectral_rate_bpm,
    relative_envelope_metrics,
    rms_envelope,
    spectrum_similarity,
)
```

在文件末尾新增：

```python
def test_relative_envelope_metrics_忽略绝对幅度缩放():
    fs = 100.0
    t = np.arange(0, 120, 1 / fs)
    carrier = np.sin(2 * np.pi * 0.25 * t)
    mod = 1.0 + 0.5 * np.sin(2 * np.pi * 0.02 * t)
    target = mod * carrier
    pred = 5.0 * target

    metrics = relative_envelope_metrics(pred, target, fs=fs, envelope_window_sec=2.0, trend_window_sec=20.0)

    assert metrics["relative_envelope_corr"] > 0.99
    assert metrics["relative_envelope_mae"] < 0.01
```

- [ ] **步骤 2：编写失败测试：漏掉相对增强会被诊断出来**

继续在 `tests/test_signal_metrics.py` 末尾新增：

```python
def test_relative_envelope_metrics_识别相对增强缺失():
    fs = 100.0
    t = np.arange(0, 120, 1 / fs)
    carrier = np.sin(2 * np.pi * 0.25 * t)
    target_mod = np.ones_like(t)
    target_mod[(t >= 45) & (t <= 75)] = 1.8
    target = target_mod * carrier
    pred = carrier.copy()

    metrics = relative_envelope_metrics(pred, target, fs=fs, envelope_window_sec=2.0, trend_window_sec=20.0)

    assert metrics["relative_envelope_corr"] < 0.8
    assert metrics["relative_envelope_mae"] > 0.05
```

- [ ] **步骤 3：运行测试验证失败**

运行：

```bash
/home/marques/.conda/envs/lighting/bin/python -m pytest tests/test_signal_metrics.py -v
```

预期：FAIL，报错包含 `cannot import name 'relative_envelope_metrics'`。

- [ ] **步骤 4：实现最少诊断函数**

在 `resp_train/metrics/signal.py` 中新增函数。放在 `rms_envelope` 后，保持指标定义集中：

```python
def relative_envelope_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    fs: float,
    envelope_window_sec: float = 2.0,
    trend_window_sec: float = 20.0,
) -> dict[str, float]:
    """比较包络相对自身趋势的变化，用于诊断增强/下降是否被模型捕捉。"""
    fs = float(fs)
    env_window = max(1, int(round(fs * float(envelope_window_sec))))
    trend_window = max(env_window, int(round(fs * float(trend_window_sec))))
    pred_rel = _relative_envelope_trace(pred, env_window, trend_window)
    target_rel = _relative_envelope_trace(target, env_window, trend_window)
    if not np.isfinite(pred_rel).all() or not np.isfinite(target_rel).all():
        return {"relative_envelope_corr": float("nan"), "relative_envelope_mae": float("nan")}
    corr = _corrcoef_or_nan(pred_rel, target_rel)
    mae = float(np.mean(np.abs(pred_rel - target_rel)))
    return {"relative_envelope_corr": corr, "relative_envelope_mae": mae}


def _relative_envelope_trace(signal: np.ndarray, env_window: int, trend_window: int) -> np.ndarray:
    env = rms_envelope(signal, env_window)
    trend = _moving_average_reflect(env, trend_window)
    rel = np.log(np.clip(env, 1e-8, None)) - np.log(np.clip(trend, 1e-8, None))
    return rel - float(np.mean(rel))


def _moving_average_reflect(values: np.ndarray, window_samples: int) -> np.ndarray:
    x = _as_1d_float(values)
    window = int(window_samples)
    if window <= 0:
        raise ValueError(f"window_samples 必须为正数，当前={window_samples}")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.pad(x, (window // 2, window - 1 - window // 2), mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


def _corrcoef_or_nan(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) <= 0 or np.std(b) <= 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])
```

- [ ] **步骤 5：运行测试验证通过**

运行：

```bash
/home/marques/.conda/envs/lighting/bin/python -m pytest tests/test_signal_metrics.py -v
```

预期：PASS。

- [ ] **步骤 6：Commit**

```bash
git add resp_train/metrics/signal.py tests/test_signal_metrics.py
git commit -m "feat: add relative envelope diagnostics"
```

### 任务 2：把相对包络指标写入评估结果

**文件：**
- 修改：`resp_train/metrics/evaluate.py`
- 测试：`tests/test_eval_metrics.py`

- [ ] **步骤 1：编写失败测试：评估输出包含新指标列**

在 `tests/test_eval_metrics.py::test_evaluate_prediction_dict_returns_window_metrics` 的断言末尾新增：

```python
    assert frame.loc[0, "relative_envelope_corr"] > 0.99
    assert frame.loc[0, "relative_envelope_mae"] < 0.01
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
/home/marques/.conda/envs/lighting/bin/python -m pytest tests/test_eval_metrics.py::test_evaluate_prediction_dict_returns_window_metrics -v
```

预期：FAIL，报错包含 `KeyError: 'relative_envelope_corr'`。

- [ ] **步骤 3：接入评估指标**

在 `resp_train/metrics/evaluate.py` import 中加入：

```python
    relative_envelope_metrics,
```

在循环中计算：

```python
        rel_env = relative_envelope_metrics(
            pred,
            target,
            fs=fs,
            envelope_window_sec=float(cfg.loss.envelope_window_sec),
        )
```

在 `records.append` 的字典中加入：

```python
                "relative_envelope_corr": rel_env["relative_envelope_corr"],
                "relative_envelope_mae": rel_env["relative_envelope_mae"],
```

- [ ] **步骤 4：运行测试验证通过**

运行：

```bash
/home/marques/.conda/envs/lighting/bin/python -m pytest tests/test_eval_metrics.py::test_evaluate_prediction_dict_returns_window_metrics -v
```

预期：PASS。

- [ ] **步骤 5：运行相关指标测试**

运行：

```bash
/home/marques/.conda/envs/lighting/bin/python -m pytest tests/test_signal_metrics.py tests/test_eval_metrics.py -v
```

预期：PASS。

- [ ] **步骤 6：Commit**

```bash
git add resp_train/metrics/evaluate.py tests/test_eval_metrics.py
git commit -m "feat: report relative envelope metrics"
```

### 任务 3：让诊断图和 run 汇总暴露新指标

**文件：**
- 修改：`scripts/plot_tho_predictions.py`
- 修改：`scripts/summarize_tho_runs.py`
- 测试：`tests/test_diagnostics_scripts.py`

- [ ] **步骤 1：编写失败测试：诊断图指标文本包含相对包络字段**

在 `tests/test_diagnostics_scripts.py` 中找到构造 metrics 行的测试数据，把两列加入第一条样本：

```python
                "relative_envelope_corr": 0.42,
                "relative_envelope_mae": 0.18,
```

在 `test_plot_run_predictions_writes_diagnostic_four_panel_png` 中不需要解析图片文本，只需保证绘图不因新列失败。新增独立测试 `_metric_text` 更直接：

```python
def test_metric_text_includes_relative_envelope_metrics():
    from scripts.plot_tho_predictions import _metric_text

    text = _metric_text(
        _metric_row={
            "relative_envelope_corr": 0.42,
            "relative_envelope_mae": 0.18,
        },
        predictions={},
        pred_idx=0,
    )

    assert "relative_envelope_corr=0.420" in text
    assert "relative_envelope_mae=0.180" in text
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
/home/marques/.conda/envs/lighting/bin/python -m pytest tests/test_diagnostics_scripts.py::test_metric_text_includes_relative_envelope_metrics -v
```

预期：FAIL，文本中缺少新字段。

- [ ] **步骤 3：更新诊断图指标文本**

在 `scripts/plot_tho_predictions.py::_metric_text` 的指标字段元组中加入：

```python
        "relative_envelope_corr",
        "relative_envelope_mae",
```

完整元组保持为：

```python
    for key in (
        "rr_spec_abs_error",
        "rr_peak_abs_error",
        "envelope_corr",
        "relative_envelope_corr",
        "relative_envelope_mae",
        "spectrum_similarity",
    ):
```

- [ ] **步骤 4：更新 run 汇总字段**

在 `scripts/summarize_tho_runs.py` 的指标字段列表中加入：

```python
    "relative_envelope_corr",
    "relative_envelope_mae",
```

要求：

- `relative_envelope_corr` 汇总均值、中位数时越高越好。
- `relative_envelope_mae` 汇总均值、中位数时越低越好。
- 如果脚本已有统一数值列聚合逻辑，只需把新列加入白名单。

- [ ] **步骤 5：运行诊断脚本测试**

运行：

```bash
/home/marques/.conda/envs/lighting/bin/python -m pytest tests/test_diagnostics_scripts.py -v
```

预期：PASS。

- [ ] **步骤 6：Commit**

```bash
git add scripts/plot_tho_predictions.py scripts/summarize_tho_runs.py tests/test_diagnostics_scripts.py
git commit -m "feat: surface relative envelope diagnostics"
```

### 任务 4：复算当前 run 并形成是否进入第二阶段的证据

**文件：**
- 输入：`runs/tho_research_v2_patch_mixer_rawish_hfpenalty_gpu_es50/20260614_171417_631400/predictions.npz`
- 输出：`runs/tho_research_v2_patch_mixer_rawish_hfpenalty_gpu_es50/20260614_171417_631400/metrics.csv`
- 输出：`runs/tho_research_v2_patch_mixer_rawish_hfpenalty_gpu_es50/20260614_171417_631400/plots_diagnostic/`

- [ ] **步骤 1：重新评估当前 run**

运行项目现有评估命令，复用同一个 checkpoint 和 run 目录：

```bash
/home/marques/.conda/envs/lighting/bin/python scripts/eval_tho_small.py \
  --checkpoint runs/tho_research_v2_patch_mixer_rawish_hfpenalty_gpu_es50/20260614_171417_631400/checkpoint.pt \
  --output runs/tho_research_v2_patch_mixer_rawish_hfpenalty_gpu_es50/20260614_171417_631400/predictions.npz \
  --metrics-output runs/tho_research_v2_patch_mixer_rawish_hfpenalty_gpu_es50/20260614_171417_631400/metrics.csv
```

预期：`metrics.csv` 包含 `relative_envelope_corr` 和 `relative_envelope_mae`。

- [ ] **步骤 2：重画诊断图**

运行：

```bash
/home/marques/.conda/envs/lighting/bin/python scripts/plot_tho_predictions.py \
  --run-dir runs/tho_research_v2_patch_mixer_rawish_hfpenalty_gpu_es50/20260614_171417_631400 \
  --output-dir runs/tho_research_v2_patch_mixer_rawish_hfpenalty_gpu_es50/20260614_171417_631400/plots_diagnostic \
  --sort-by relative_envelope_mae \
  --max-plots 8
```

预期：输出 8 张 PNG，标题指标文本中包含相对包络指标。

- [ ] **步骤 3：汇总判断是否进入第二阶段**

运行：

```bash
/home/marques/.conda/envs/lighting/bin/python -c "import pandas as pd; p='runs/tho_research_v2_patch_mixer_rawish_hfpenalty_gpu_es50/20260614_171417_631400/metrics.csv'; m=pd.read_csv(p); cols=['envelope_corr','relative_envelope_corr','relative_envelope_mae','rr_peak_abs_error','spectrum_similarity']; print(m[cols].describe(percentiles=[.25,.5,.75,.9]).to_string()); print('\\nworst relative envelope:'); print(m.sort_values('relative_envelope_mae', ascending=False).head(12)[['dataset_row_id']+cols].to_string(index=False))"
```

进入第二阶段的判据：

- `relative_envelope_corr` 中位数低于 `0.45`，或
- `relative_envelope_mae` 的 75 分位明显高于人工可接受阈值 `0.20`，或
- 最差 8 张诊断图中至少 5 张显示目标有明确相对增强/下降，而预测未跟随。

不进入第二阶段的判据：

- 相对包络指标整体良好，但 `envelope_corr` 仍低，优先继续查相位/对齐问题。
- 只有少数样本异常，优先查数据质量或标签对齐，不改 loss。

- [ ] **步骤 4：记录阶段结论**

在当前计划文件末尾追加执行记录。记录必须包含以下信息：

- 当前 run 路径。
- `relative_envelope_corr` 的 median 数值。
- `relative_envelope_mae` 的 p75 数值。
- 最差诊断图的人工判断结果和代表性 `dataset_row_id`。
- 明确决策：进入第二阶段，或暂不进入第二阶段。

记录示例：

```markdown
## 执行记录

### 阶段一诊断结论

- run：`runs/tho_research_v2_patch_mixer_rawish_hfpenalty_gpu_es50/20260614_171417_631400`
- `relative_envelope_corr` median：0.38
- `relative_envelope_mae` p75：0.24
- 最差诊断图人工判断：未通过，代表 row 为 7954、7956、7946
- 决策：进入第二阶段
```

- [ ] **步骤 5：Commit**

```bash
git add runs/tho_research_v2_patch_mixer_rawish_hfpenalty_gpu_es50/20260614_171417_631400/metrics.csv \
  runs/tho_research_v2_patch_mixer_rawish_hfpenalty_gpu_es50/20260614_171417_631400/plots_diagnostic \
  docs/superpowers/plans/2026-06-15-relative-envelope-diagnostics.md
git commit -m "docs: record relative envelope diagnostic decision"
```

---

## 阶段二：小权重相对包络约束

只有阶段一满足进入判据时执行本阶段。

### 任务 5：在 WeakSyncLoss 中加入默认关闭的 relative_envelope_loss

**文件：**
- 修改：`resp_train/losses/weak.py`
- 修改：`configs/tho_research_v2_patch_mixer.yaml`
- 测试：`tests/test_losses.py`

- [ ] **步骤 1：编写失败测试：默认返回新分量但权重为 0 不改变总损失**

在 `tests/test_losses.py::test_weak_sync_loss_returns_components_and_scalar` 中，把分量断言改为：

```python
    assert set(parts) == {"envelope", "spectrum", "smooth", "high_freq", "relative_envelope"}
```

新增测试：

```python
def test_relative_envelope_weight_zero_keeps_total_loss_unchanged():
    cfg = _cfg()
    cfg.loss.relative_envelope_weight = 0.0
    loss_fn = WeakSyncLoss(cfg)
    pred = torch.randn(2, 1, 1800)
    target = torch.randn(2, 1, 1800)

    total, parts = loss_fn(pred, target)
    expected = (
        cfg.loss.envelope_weight * parts["envelope"]
        + cfg.loss.spectrum_weight * parts["spectrum"]
        + cfg.loss.smooth_weight * parts["smooth"]
        + cfg.loss.high_freq_weight * parts["high_freq"]
    )

    assert torch.allclose(total, expected)
```

- [ ] **步骤 2：编写失败测试：相对增强缺失会增加新分量**

在 `tests/test_losses.py` 新增：

```python
def test_relative_envelope_loss_penalizes_missing_relative_boost():
    cfg = _cfg()
    cfg.loss.relative_envelope_weight = 0.01
    loss_fn = WeakSyncLoss(cfg)
    fs = float(cfg.window.target_fs)
    time = torch.arange(0, 60, 1 / fs)
    carrier = torch.sin(2 * torch.pi * 0.25 * time)
    boost = torch.ones_like(time)
    boost[(time >= 20) & (time <= 40)] = 1.8
    target = (boost * carrier).reshape(1, 1, -1)
    good = (3.0 * target).clone()
    bad = carrier.reshape(1, 1, -1)

    _, good_parts = loss_fn(good, target)
    _, bad_parts = loss_fn(bad, target)

    assert bad_parts["relative_envelope"] > good_parts["relative_envelope"] + 0.05
```

- [ ] **步骤 3：运行测试验证失败**

运行：

```bash
/home/marques/.conda/envs/lighting/bin/python -m pytest tests/test_losses.py -v
```

预期：FAIL，分量缺少 `relative_envelope`。

- [ ] **步骤 4：实现相对包络 loss**

在 `resp_train/losses/weak.py::__init__` 中新增：

```python
        self.relative_env_weight = float(cfg.loss.get("relative_envelope_weight", 0.0))
        self.relative_env_trend_window = max(self.envelope_window, int(round(self.fs * 20.0)))
```

在 `forward` 中新增：

```python
        relative_env = self._relative_envelope_loss(pred, target)
```

总损失加入：

```python
            + self.relative_env_weight * relative_env
```

返回分量加入：

```python
            "relative_envelope": relative_env.detach(),
```

新增方法：

```python
    def _relative_envelope_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_rel = self._relative_envelope_trace(pred)
        target_rel = self._relative_envelope_trace(target)
        return torch.mean(torch.abs(pred_rel - target_rel))

    def _relative_envelope_trace(self, x: torch.Tensor) -> torch.Tensor:
        env = self._rms_envelope(x)
        pad = self.relative_env_trend_window // 2
        trend = F.avg_pool1d(env, kernel_size=self.relative_env_trend_window, stride=1, padding=pad)
        if trend.shape[-1] > env.shape[-1]:
            trend = trend[..., : env.shape[-1]]
        rel = torch.log(torch.clamp(env, min=1e-8)) - torch.log(torch.clamp(trend, min=1e-8))
        return rel - rel.mean(dim=-1, keepdim=True)
```

- [ ] **步骤 5：配置默认值保持关闭**

在 `configs/tho_research_v2_patch_mixer.yaml` 的 `loss` 下加入：

```yaml
  relative_envelope_weight: 0.0
```

- [ ] **步骤 6：运行 loss 测试**

运行：

```bash
/home/marques/.conda/envs/lighting/bin/python -m pytest tests/test_losses.py -v
```

预期：PASS。

- [ ] **步骤 7：Commit**

```bash
git add resp_train/losses/weak.py configs/tho_research_v2_patch_mixer.yaml tests/test_losses.py
git commit -m "feat: add optional relative envelope loss"
```

### 任务 6：跑小权重实验并比较

**文件：**
- 输入：`configs/tho_research_v2_patch_mixer.yaml`
- 输出：`runs/tho_research_v2_patch_mixer_rawish_relenv001/`

- [ ] **步骤 1：启动小权重实验**

运行：

```bash
/home/marques/.conda/envs/lighting/bin/python scripts/train_tho_small.py \
  --config configs/tho_research_v2_patch_mixer.yaml \
  --set outputs.run_root=runs/tho_research_v2_patch_mixer_rawish_relenv001 \
  --set loss.relative_envelope_weight=0.01 \
  --set training.epochs=50 \
  --set training.patience=8
```

预期：生成新的 run 目录，包含 `metrics.csv`、`train_history.csv`、`plots_diagnostic/`。

- [ ] **步骤 2：比较新旧 run**

运行：

```bash
/home/marques/.conda/envs/lighting/bin/python scripts/summarize_tho_runs.py \
  --runs-root runs/tho_research_v2_patch_mixer_rawish_relenv001 \
  --output runs/tho_research_v2_patch_mixer_rawish_relenv001/summary.csv
```

如果需要和旧 run 并排比较，运行：

```bash
/home/marques/.conda/envs/lighting/bin/python -c "import pandas as pd; old='runs/tho_research_v2_patch_mixer_rawish_hfpenalty_gpu_es50/20260614_171417_631400/metrics.csv'; new='runs/tho_research_v2_patch_mixer_rawish_relenv001/summary.csv'; print(pd.read_csv(old)[['relative_envelope_corr','relative_envelope_mae','rr_peak_abs_error','spectrum_similarity']].mean().rename('old')); print(pd.read_csv(new).filter(regex='model_(relative_envelope_corr|relative_envelope_mae|rr_peak_abs_error|spectrum_similarity)_mean|run_id|run_dir').to_string(index=False))"
```

验收标准：

- `relative_envelope_corr` 中位数提升，或 `relative_envelope_mae` 中位数下降。
- `rr_peak_abs_error` 不允许明显恶化，均值恶化超过 `0.15 bpm` 视为不通过。
- `spectrum_similarity` 不允许明显恶化，均值下降超过 `0.01` 视为不通过。
- 诊断图中相对增强/下降跟随更好，且没有明显增加高频伪影。

- [ ] **步骤 3：根据结果记录决策**

在本计划的执行记录中追加阶段二结论。记录必须包含以下信息：

- 新 run 路径。
- `relative_envelope_corr` 均值或中位数相对旧 run 的变化。
- `relative_envelope_mae` 均值或中位数相对旧 run 的变化。
- `rr_peak_abs_error` 均值相对旧 run 的变化。
- `spectrum_similarity` 均值相对旧 run 的变化。
- 明确决策：保留 `relative_envelope_weight=0.01`、降到 `0.005`，或放弃该 loss。

记录示例：

```markdown
### 阶段二实验结论

- 对比 run：`runs/tho_research_v2_patch_mixer_rawish_relenv001/20260615_120000_000000`
- `relative_envelope_corr` median 变化：+0.07
- `relative_envelope_mae` median 变化：-0.03
- `rr_peak_abs_error` mean 变化：+0.04 bpm
- `spectrum_similarity` mean 变化：-0.003
- 决策：保留 `relative_envelope_weight=0.01`
```

- [ ] **步骤 4：Commit**

```bash
git add docs/superpowers/plans/2026-06-15-relative-envelope-diagnostics.md
git commit -m "docs: record relative envelope loss experiment"
```

---

## 验收标准

- 第一阶段完成后，`metrics.csv` 必须包含：
  - `relative_envelope_corr`
  - `relative_envelope_mae`
- 诊断图指标文本必须显示上述两列。
- 现有 `envelope_corr` 的定义和数值口径不变。
- 不新增绝对幅度标定训练目标。
- 第二阶段默认配置 `relative_envelope_weight=0.0`，旧训练行为不变。
- 只有阶段一诊断满足进入判据时，才允许训练小权重相对包络 loss。

## 回归验证命令

阶段一完成后运行：

```bash
/home/marques/.conda/envs/lighting/bin/python -m pytest tests/test_signal_metrics.py tests/test_eval_metrics.py tests/test_diagnostics_scripts.py -v
```

阶段二完成后追加运行：

```bash
/home/marques/.conda/envs/lighting/bin/python -m pytest tests/test_losses.py tests/test_signal_metrics.py tests/test_eval_metrics.py tests/test_diagnostics_scripts.py -v
```

## 风险与反向检查

- 如果相对包络指标低但峰值/频谱指标高，不应直接判定模型失败；需要人工查看最差诊断图，确认目标中确实存在有意义的相对增强/下降。
- 如果相对包络 loss 改善了诊断指标但损害呼吸率指标，应优先降低权重到 `0.005` 或放弃 loss。
- 如果坏例集中在少数 `samp_id` 或数据质量类别，应优先处理数据/对齐问题，而不是扩大 loss。
- 如果相对包络指标和现有 `envelope_corr` 高度一致，说明新增 loss 价值有限，只保留诊断指标即可。
