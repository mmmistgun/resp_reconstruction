# E4 v2 工程验收修订 r2

日期：2026-09-17。对应协议：`e4-scale-aggregation-v2-20260917`。

本附件仅覆盖[主协议](e4_scale_aggregation_v2_protocol_20260917.md)第 6.2 节的 synthetic GPU 更新窗口、诊断落盘，以及第 8 节的当前实现锁路径。主协议中初始化、模型、数据、loss、指标、12 次正式训练预算、checkpoint selector 和 test 门控继续适用。当前 GPU 验收尚待用户执行。

## 1. 触发与证据

r1 实现锁 SHA-256：`39cd320e67c02b59dc102c6f2f0d4e2d821b0fe96aff1f6049a0030af5fb2495`。
失败 attempt：`runs/e4_scale_aggregation_v2/gpu_acceptance/gpu_acceptance_39cd320e67c0_20260917T134553Z_c573c28fb76d`。

执行顺序及已落盘记录将失败用例定位为 `scale_attention / seed 20260813 / batch=1`。此前 `static_scale` 三 seed 的 batch-1 和 batch-128、`scale_attention` 前两个 seed 的 batch-1 均已有通过记录。r1 在检查失败后才准备保存用例记录，因此失败用例的具体参数、梯度和变化量没有保留下来，无法断言此次 GPU 失败究竟是未更新还是零梯度。

已保存的 `scale_attention / seed 20260812` 记录显示：第一步新增参数梯度全零，第二步 score 梯度打开，第三步 content 梯度打开；第五步 content.bias 梯度范数约 `1.195e-11`。原生 warm-up 前五步 LR 为 `9.375e-7` 至 `4.6875e-6`。梯度通路打开与 FP32 参数发生可见变化是不同条件，固定五步没有数值充分性保证。

CPU 定向 fixture 使用 FP32 参数 0.1、有限梯度 `5e-12`、原生 6400-update warm-up 和 AdamW eps=1e-8，复现第五步梯度非零但参数变化为零，随后在有界窗口内出现真实变化。此 fixture 验证舍入机制；完整模型的 Mamba CUDA 算子不支持 CPU forward，实际 GPU 根因需新诊断确认。

## 2. 修订后的验收

- 每个用例至少完成 5 次、最多完成 20 次连续原生更新，从 update index=0 开始，复用原生 LR、loss、optimizer 分组及精度。
- 第 5 步起，每步检查全部新增参数张量：本步梯度范数大于零，且相对初始值至少一个元素真实改变，两项同时满足才通过。范数使用 FP64 计算，避免诊断平方求和下溢。
- 至第 20 步仍不满足则失败，错误列出未更新参数名和本步零梯度参数名。梯度缺失、非有限 forward/loss/gradient/state 立即失败。参数已变化不能替代末步非零梯度。
- 仍执行四 arm×三 seed 的 batch-1 初始 W0 对照，以及每 arm、seed 20260811 的 batch-128/chunk=8 用例；共 16 个用例，合计 80–320 次 synthetic updates。初始等值容差和 peak reserved ≤80% 门槛继续适用。
- 每个用例独立创建 `*_diagnostics/`：每个成功完成的更新即时排他写入 `step_NNN.json`，包含 LR/loss summary、逐参数梯度范数、相对初值的最大变化量和变化标志。正常结束或异常退出均保存 `case.json`，记录 arm、seed、batch、实际步数、阶段及错误。OOM 等中途异常保留此前完整步骤，外层继续登记 failed lifecycle 并返回失败。
- 成功时顶层 `*_batch1.json` / `*_batch128.json` 及总回执保持原位置，增加实际步数和诊断字段。benchmark 继续使用固定 5 次 warm-up、20 次计时，与此验收窗口独立。

这是工程参与更新的有界检查；20 步是新的失败上限，不是对所有 GPU/seed 必然通过的保证。GPU 仍失败时，以具体参数和逐步诊断判断原因，不改变模型以迁就验收。

## 3. 实现身份与影响范围

当前实现锁：`docs/experiments/e4_scale_aggregation_v2_implementation_lock_r2_20260917.json`。新锁登记本附件、工程定向测试及 r1 锁的字节身份。r1 主协议、实现锁和失败 attempt 保持原样；代码可从失败时提交 `3e74bd3cae624e2260be4ae8c5d0d3d8900bf871` 追溯。

修订时输出根仅发现上述失败 GPU attempt，尚无 v2 formal、benchmark 或冻结汇总。新 attempt 使用 r2 锁 SHA 生成独立 identity，完整执行 16 个工程用例；r1 部分成功记录只作故障追溯。模型源码、四种聚合合同和 12 个正式训练模板保持相同，因此没有已完成正式训练需要重训。历史 E4 v1 训练及 test 锁保留原身份。

## 4. 验证与用户执行

定向验证覆盖 warm-up 舍入、最少步数、最多步数、未更新、零梯度、缺失梯度、NaN/Inf、极小非零梯度范数及中途异常现场保留；同时检查训练和 test 控制器对当前训练锁路径的兼容性。全部使用 CPU disposable fixture。

本次验证结果：**58 passed，51.64 s**（工程 10 项、训练控制器 22 项、test 控制器 26 项）。GPU 验收及真实数据任务尚未代跑。

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python -m pytest \
  tests/test_e4_aggregation_v2_engineering.py tests/test_e4_aggregation_v2.py \
  tests/test_e4_aggregation_v2_test.py -q
```

提交本次代码、附件和 r2 锁，保持工作树干净后，由用户运行：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  ./.venv/bin/python scripts/run_e4_aggregation_v2.py gpu-acceptance --device cuda:0
```

验收标准：新 attempt 生成 `gpu_acceptance.json`、`manifest.json` 和 `freeze_receipt.json`，16 个用例全部通过，并保存各自实际步数、非零梯度、真实参数变化和显存比例。新失败仍保留独立目录及具体错误；只有成功的 r2 验收目录可用于下一步正式训练。
