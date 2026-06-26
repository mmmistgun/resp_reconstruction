# F-A STFT Loss Prep 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 为 F-A waveform 输出 + 目标 STFT 派生 loss 实验补齐代码、单测、runner、汇总入口和交接命令；正式全量训练由用户手动启动。

**架构：** 在 `resp_train/losses/stft.py` 增加专用 target-STFT loss 组件，由 `WeakSyncLoss` 按权重组合进当前训练目标；新增 F-A runner 固定 F0/F-A 同 seed 配对；新增 summarizer 从 manifest 回连 run 目录并输出 paired delta 和分层诊断。

**技术栈：** PyTorch STFT、OmegaConf dotlist、pytest、现有 `scripts/batch_utils.py` 批量调度工具。

---

### 任务 1：F-A STFT Loss

**文件：**
- 创建：`resp_train/losses/stft.py`
- 修改：`resp_train/losses/weak.py`
- 测试：`tests/test_fa_stft_loss.py`

- [ ] **步骤 1：编写失败测试**

测试覆盖：`L_dist` 对主峰/谐波分布敏感、`L_bandE` 对分频带能量轨迹敏感、`WeakSyncLoss` 能按权重纳入 STFT loss 并回传梯度、空频带配置报错。

- [ ] **步骤 2：运行测试验证失败**

运行：`./.venv/bin/python -m pytest tests/test_fa_stft_loss.py -q`

预期：FAIL，原因是 `resp_train.losses.stft` 或 STFT loss 配置尚未实现。

- [ ] **步骤 3：实现最少代码**

实现 `TargetStftLoss`，使用 30s/5s 默认口径、`center=False`、JSD 分布约束和分频带 log-energy SmoothL1；`WeakSyncLoss` 增加 `stft_dist`、`stft_band_energy` 分项和可选组件梯度范数接口。

- [ ] **步骤 4：运行测试验证通过**

运行：`./.venv/bin/python -m pytest tests/test_fa_stft_loss.py -q`

预期：PASS。

### 任务 2：F-A Runner

**文件：**
- 创建：`scripts/run_f_a_stft_loss_probe.py`
- 测试：`tests/test_run_f_a_stft_loss_probe.py`

- [ ] **步骤 1：编写失败测试**

测试覆盖：5 个 label × 3 seed，F0 time-only 与 F0 STFT anchor 可配对，F-A 候选只改 STFT loss 权重，命令包含全量窗口、方向门控、batch128、loss grad norm logging。

- [ ] **步骤 2：运行测试验证失败**

运行：`./.venv/bin/python -m pytest tests/test_run_f_a_stft_loss_probe.py -q`

预期：FAIL，原因是 runner 尚不存在。

- [ ] **步骤 3：实现 runner**

复用 E3 runner 风格，支持 `--dry-run`、`--device`、`--max-parallel`、`--manifest` 和 `--start-stagger-sec`。

- [ ] **步骤 4：运行测试验证通过**

运行：`./.venv/bin/python -m pytest tests/test_run_f_a_stft_loss_probe.py -q`

预期：PASS。

### 任务 3：F-A Summary

**文件：**
- 创建：`scripts/summarize_f_a_stft_loss.py`
- 测试：`tests/test_summarize_f_a_stft_loss.py`

- [ ] **步骤 1：编写失败测试**

测试覆盖：从 manifest 解析 run root、读取 metrics/train_history、输出逐 run detail、按 `paired_f0_label + seed` 计算 paired delta、输出 baseline hard/easy/low-spectrum/fast-RR 分层。

- [ ] **步骤 2：运行测试验证失败**

运行：`./.venv/bin/python -m pytest tests/test_summarize_f_a_stft_loss.py -q`

预期：FAIL，原因是 summarizer 尚不存在。

- [ ] **步骤 3：实现 summarizer**

读取 manifest，选择每个 spec 最新完整 run，计算核心指标 mean/median、`frac_gt_1/2`、paired delta 和分层 delta。

- [ ] **步骤 4：运行测试验证通过**

运行：`./.venv/bin/python -m pytest tests/test_summarize_f_a_stft_loss.py -q`

预期：PASS。

### 任务 4：记录执行边界

**文件：**
- 修改：`docs/experiments/f_series_stft_loss_plan.md`

- [ ] **步骤 1：追加短说明**

在第一阶段执行清单附近记录：本轮 agent 只准备实现、测试、manifest、smoke 和命令；全量 F-A pilot 由用户手动运行。

- [ ] **步骤 2：检查 diff**

运行：`git diff -- docs/experiments/f_series_stft_loss_plan.md`

预期：只新增执行分工说明，不改动既有实验结论。

### 任务 5：验证与交接

**文件：**
- 修改：相关测试文件与实现文件

- [ ] **步骤 1：运行聚焦测试**

运行：`./.venv/bin/python -m pytest tests/test_fa_stft_loss.py tests/test_run_f_a_stft_loss_probe.py tests/test_summarize_f_a_stft_loss.py tests/test_losses.py tests/test_engine_smoke.py -q`

预期：PASS。

- [ ] **步骤 2：运行 runner dry-run**

运行：`./.venv/bin/python scripts/run_f_a_stft_loss_probe.py --dry-run --manifest /tmp/f_a_manifest.csv`

预期：打印 15 个 plan tag，写出 manifest，不启动训练。

- [ ] **步骤 3：整理全量运行命令**

提供：split 审计命令、dry-run 命令、正式 runner 命令、summary 命令、peak-band 分布诊断命令。
