# E7 P4：18-cell formal train/validation 执行附件

日期：2026-09-24。协议 ID：`e7-scale-encoding-aggregation-p4-formal-v1-20260924`。

状态：**P4 formal 入口与单一 execution lock 已实现并完成 CPU 定向验证；18 次正式训练由用户执行，当前尚未启动。**

## 1. 冻结来源

- P1 implementation lock：`299448a12c4006f2f918f56645000838f83f044fbbf31d7cc281faa6540b11e1`。
- P3 engineering lock：`773f5e78c5e8d7b08cadaac00797ede8dcd9f3f7a77d0775fc5679650a29d32d`。
- P3 closeout：`docs/experiments/e7_scale_encoding_aggregation_p3_closeout_20260924.json`，SHA-256 `29334d33e4a3fb6666230bb5cf6915d16c5cf34e8a2cf49a07bd58d499433182`。
- P4 execution lock：`docs/experiments/e7_scale_encoding_aggregation_p4_execution_lock_20260924.json`。

P4 lock 固定六臂×三 seed 的完整矩阵、P1/P3/closeout identity、唯一 GPU acceptance receipt、formal 源码和训练/validation 合同。该锁由 18 次 formal 与后续 P5 汇总共同复用。

## 2. Formal 合同

每个 cell 从头训练，使用同 seed 初始化与数据顺序。最大 80 epochs、每完整 epoch 80 updates；early stopping 为 `min_epoch=30 / patience=15 / min_delta=0`，wait 从 epoch 1 累计，学习率始终按 planned 6400 updates 计算。Checkpoint selector 为完整 validation Local RR 严格最小，平局取最早 epoch。

固定 physical/effective batch 128、accumulation 1、BF16、branch checkpoint chunk 8、完整尾 batch、`L_sync + 0.25 L_effort`、AdamW 和 W0 数据/split/指标语义。每个 cell 完整保存 history、best/final checkpoint、optimizer state、逐窗口 validation metrics、summary、row identity、环境、来源审计及 lifecycle。

Formal 入口逐项回放 early-stop、LR、checkpoint epoch、optimizer step、指标分母和 W0 validation row/eligibility identity。有限退化结果保留质量标志；非有限关键量显式失败。相同 execution lock 下已完成 cell 拒绝重跑，失败现场保留并允许该 cell 新建 attempt。

P4 只读取 train/validation。Selected checkpoint 的完整表征诊断、checkpoint 内干预与 18-cell 析因汇总在 P5 一次性执行，不访问 research-test。

## 3. 唯一 GPU receipt

全部 18 次训练复用 P3 closeout 固定的成功 GPU receipt：

```text
runs/e7_scale_encoding_aggregation/gpu_acceptance/
gpu_acceptance_773f5e78c5e8_20260924T082610Z_62e8f9b7eaef
```

每次 formal 都重新核验其 freeze receipt、manifest、24-cell 完整性、P1/P3 identity 及 GPU/CUDA/cuDNN/PyTorch 环境。Benchmark 只作为效率证据，不是 formal 门控。

## 4. 用户执行命令

先确认代码、P4 lock 和 GPU receipt 均已提交/冻结：

```bash
git status --short  # 必须无输出

env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  ./.venv/bin/python scripts/run_e7_scale_encoding_aggregation_p4.py check-lock
```

随后使用同一个 receipt 顺序执行完整 18-cell；任一失败即停止，修复后从未完成 cell 继续：

```bash
E7_GPU_RECEIPT='runs/e7_scale_encoding_aggregation/gpu_acceptance/gpu_acceptance_773f5e78c5e8_20260924T082610Z_62e8f9b7eaef'

for E7_SEED in 20260811 20260812 20260813; do
  for E7_ARM in \
    s0_shallow__mean \
    s0_shallow__frequency_attention \
    s1_deep_local__mean \
    s1_deep_local__frequency_attention \
    s2_axis_spanning__mean \
    s2_axis_spanning__frequency_attention; do
    env -u LD_LIBRARY_PATH -u LD_PRELOAD \
      ./.venv/bin/python scripts/run_e7_scale_encoding_aggregation_p4.py formal \
      --arm "$E7_ARM" \
      --seed "$E7_SEED" \
      --device cuda:0 \
      --gpu-receipt "$E7_GPU_RECEIPT" || break 2
  done
done
```

全部完成后只读核验 18-cell 唯一成功矩阵：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  ./.venv/bin/python scripts/run_e7_scale_encoding_aggregation_p4.py check-completed
```
