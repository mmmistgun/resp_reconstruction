# E8 P3：36-cell formal train/validation 执行附件

日期：2026-09-25。协议 ID：`e8-film-decoder-redesign-factorial-v1-20260925`。

状态：**Formal 入口已实现并完成 CPU 定向验证；唯一 execution lock 待在干净提交上生成。正式训练由用户执行，当前尚未启动。**

## 1. 固定来源

- 主协议：`docs/experiments/e8_film_decoder_redesign_v1_protocol_20260925.md`。
- P2 收口：`docs/experiments/e8_film_decoder_redesign_v1_p2_closeout_20260925.md`，SHA-256 `7e83f2045a586a0b6f7aabfcf4a3b6a71a3662ce9c397c600394dd99b3402010`。
- P2 engineering identity：`aa80977f0c737906bdf011bd9b39367d1c77b39130404a39db9c1f6b2670f3d4`。
- W0 train/validation 来源锁：`docs/experiments/w0_structural_factorial_v1_implementation_lock_r2_20260924.json`，SHA-256 `32141eab672ea41435055c45cbd7ec96325f8ddef2481a210db2941222c9e9f3`。
- E8 formal execution lock：`docs/experiments/e8_film_decoder_redesign_v1_formal_execution_lock_20260925.json`。

Execution lock 固定 12 arms × 3 seeds、P2 两份成功 attempt、W0 train/validation 来源、配置、停止合同和 formal 关键源码。它由全部 36 次训练与后续 validation 汇总共同复用。

## 2. Formal 合同

- 每个 cell 从头训练，使用同 seed 初始化与数据顺序。
- 最大 80 epochs，每完整 epoch 80 optimizer updates；planned 上限 6,400 updates。
- Early stopping：`min_epoch=30 / patience=15 / min_delta=0`，wait 从 epoch 1 累积。
- Selector：完整 validation Local RR 严格最小，并列取最早 epoch。
- Physical/effective batch 128、accumulation 1、BF16、branch checkpoint chunk 8、完整尾 batch。
- Loss：`L_sync + 0.25 L_effort`；AdamW、weight decay、gradient clipping 和 planned-update learning-rate schedule 保持 W0 合同。
- 只读取 train/validation；每个 attempt 启动前核验数据/cache/source identity、row identity、subject 隔离、P2 evidence 与运行环境兼容性。
- 保存完整 history、best/final checkpoint、optimizer state、逐窗口 validation metrics、summary、初始化 identity、环境、来源审计与不可覆盖 lifecycle。

相同 execution lock 下已完成 cell 拒绝重跑；失败现场保留，可在修复后为未完成 cell 建立新 attempt。不得删除失败目录或用它冒充成功单元格。

## 3. Execution lock

在 formal 实现提交且工作树干净后生成一次：

```bash
./.venv/bin/python scripts/run_e8_film_decoder_redesign_v1.py prepare-formal-lock
```

将 lock 提交后回载核验：

```bash
./.venv/bin/python scripts/run_e8_film_decoder_redesign_v1.py check-formal-lock
./.venv/bin/python scripts/run_e8_film_decoder_redesign_v1.py matrix-status
```

`matrix-status` 在训练前必须为 `pending=36 / running=0 / failed=0 / completed=0`。

## 4. 用户执行命令

第二张物理 GPU 通过 `CUDA_VISIBLE_DEVICES=1` 隔离后，在进程内使用 `cuda:0`。按固定 seed-major、arm-minor 顺序运行，任一失败立即停止：

```bash
cd /mnt/disk_code/marques/resp_reconstruction/.worktrees/e8_film_decoder_redesign_v1

for E8_SEED in 20260811 20260812 20260813; do
  for E8_ARM in \
    e8_fill65_pointwise \
    e8_fill65_single \
    e8_fill65_temporal \
    e8_direct_pointwise \
    e8_direct_single \
    e8_direct_temporal \
    e8_res96_pointwise \
    e8_res96_single \
    e8_res96_temporal \
    e8_res192_pointwise \
    e8_res192_single \
    e8_res192_temporal; do
    env -u LD_LIBRARY_PATH -u LD_PRELOAD \
      CUDA_VISIBLE_DEVICES=1 \
      PYTHONPATH=. \
      /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python \
      scripts/run_e8_film_decoder_redesign_v1.py formal \
      --arm "$E8_ARM" \
      --seed "$E8_SEED" \
      --device cuda:0 \
      --confirm-formal-training || break 2
  done
done
```

完成或中断后只读核验：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  CUDA_VISIBLE_DEVICES=1 \
  PYTHONPATH=. \
  /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python \
  scripts/run_e8_film_decoder_redesign_v1.py matrix-status
```

## 5. 后续门控

只有 36/36 个唯一成功 cell 完成后，才开放一次性 validation 汇总。Research-test 仍需后续专项协议、固定全部 validation-selected checkpoints 的 allowlist，以及当次用户明确授权；当前 formal 不读取 test。
