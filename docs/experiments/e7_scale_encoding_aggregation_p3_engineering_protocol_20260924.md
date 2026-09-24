# E7 P3：GPU 工程验收与效率测量附件

日期：2026-09-24。协议 ID：`e7-scale-encoding-aggregation-p3-engineering-v1-20260924`。

状态：**P3 GPU acceptance 与 benchmark 入口、CPU fixture 和 engineering lock 已实现；GPU 阶段由用户执行，当前尚无 GPU 结果。**

## 1. 来源身份

本附件引用 E7 P1 implementation lock：

```text
docs/experiments/e7_scale_encoding_aggregation_p1_implementation_lock_20260924.json
SHA-256 299448a12c4006f2f918f56645000838f83f044fbbf31d7cc281faa6540b11e1
```

P3 工程实现由独立锁固定：

```text
docs/experiments/e7_scale_encoding_aggregation_p3_engineering_lock_20260924.json
```

GPU acceptance 和 benchmark 都必须在干净提交上运行，并同时登记 P1 与 P3 lock identity、环境、命令、生命周期和文件 SHA。输入只使用解析 synthetic waveform、target 与 `[B,97,360]` W tensor；不读取真实数据或 research-test。

## 2. GPU acceptance

固定矩阵为：

- 六 arm × 三 seed 的 batch-1 验收，共 18 cells；
- 六 arm 使用 seed `20260811` 的 batch-128/chunk-8 验收，共 6 cells；
- 每 cell 至少 5、最多 20 次原生 `loss/backward/clip/AdamW` update；
- BF16、完整 `L_sync + 0.25 L_effort`、planned 6400-update LR；
- peak reserved/device total 必须不超过 80%。

Batch-1 同时核验公共 W0 state、CPU/CUDA RNG、聚合前后 shape、有限性和完整 waveform 初始等价。多步验收跟踪每个 encoder/aggregation 参数的 FP64 gradient norm 与相对初值最大变化；`s0_shallow__mean` 跟踪原 W 分支 final projection。成功要求所有 tracked 参数具有有限非零梯度并发生可见更新。每一步单独保存诊断，失败 lifecycle 和已完成诊断保持可追溯。

## 3. Benchmark

六 arm 使用两个场景：batch-1 eval forward 与 batch-128 native train update。每项在独立进程中 warm-up 5 次、记录 20 次，共 `6×2×3=36` 个测量进程。三组顺序固定为正序、逆序和循环移位，减弱顺序性热状态影响。

每个进程执行 CUDA synchronize，记录逐次秒数、median、IQR、peak allocated/reserved 和设备总显存。范围是 synthetic cached-input model-only，不包含 CWT/cache 构建与磁盘 I/O；局部 MAC 报告不解释为完整模型 FLOPs。

## 4. 用户执行命令

提交 engineering lock 并保持工作树干净后执行：

```bash
./.venv/bin/python scripts/run_e7_scale_encoding_aggregation_p3.py check-lock

./.venv/bin/python scripts/run_e7_scale_encoding_aggregation_p3.py \
  gpu-acceptance --device cuda:0

./.venv/bin/python scripts/run_e7_scale_encoding_aggregation_p3.py \
  benchmark --device cuda:0
```

GPU acceptance 与 benchmark 各自排他创建 attempt。同一 P3 lock identity 已完成的阶段拒绝重跑；失败后保留现场，修复需要新的实现与 lock identity。
