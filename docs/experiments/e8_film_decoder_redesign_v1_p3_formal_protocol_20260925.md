# E8 P3：36-cell formal train/validation 执行附件

日期：2026-09-25。协议 ID：`e8-film-decoder-redesign-factorial-v1-20260925`。

状态：**Formal 入口与 base execution lock 已完成；2026-09-26 建立仅修改同型号 GPU 容量兼容门控的增量修订。修订前矩阵固定为 `completed=11 / failed=1 / pending=24 / running=0`，科学合同未改变。**

## 1. 固定来源

- 主协议：`docs/experiments/e8_film_decoder_redesign_v1_protocol_20260925.md`。
- P2 收口：`docs/experiments/e8_film_decoder_redesign_v1_p2_closeout_20260925.md`，SHA-256 `7e83f2045a586a0b6f7aabfcf4a3b6a71a3662ce9c397c600394dd99b3402010`。
- P2 engineering identity：`aa80977f0c737906bdf011bd9b39367d1c77b39130404a39db9c1f6b2670f3d4`。
- W0 train/validation 来源锁：`docs/experiments/w0_structural_factorial_v1_implementation_lock_r2_20260924.json`，SHA-256 `32141eab672ea41435055c45cbd7ec96325f8ddef2481a210db2941222c9e9f3`。
- E8 formal execution lock：`docs/experiments/e8_film_decoder_redesign_v1_formal_execution_lock_20260925.json`，SHA-256 `40b30fc6ecdcc9b40750c393be8fd2faf4e223da0566432df739aacb513713c5`。
- Runtime amendment：`docs/experiments/e8_film_decoder_redesign_v1_formal_runtime_amendment_20260926.json`，SHA-256 `517bdf281a0db0902e0c10481b0ea210a7c514d627eb6397e9e982d95ba270e3`。

Execution lock 固定 12 arms × 3 seeds、P2 两份成功 attempt、W0 train/validation 来源、配置、停止合同和 formal 关键源码。它由全部 36 次训练与后续 validation 汇总共同复用。

### 1.1 Runtime amendment

P2 在物理 GPU 1 上记录 `device_total_bytes=16,717,840,384`；同型号 GPU 0 报告 `16,710,500,352`，差 `7,340,032` bytes（约 0.044%），两者的型号、Python/PyTorch/CUDA/cuDNN、依赖与 BF16 合同一致。Base gate 使用逐 byte `current>=P2`，因此在任何训练开始前拒绝 GPU 0。

修订只把容量判断改为同时满足：

1. 原有软件栈、device name 与 BF16 字段继续精确相等；
2. `device_total_bytes >= 16,710,500,352`；
3. P2 最大 `peak_reserved=10,643,046,400` bytes 在当前设备上的比例不超过 0.8。

GPU 0 对应比例约为 `0.6369`。模型、参数、初始化、数据、loss、optimizer、batch、停止合同、selector、指标和输出 identity 均未改变。修订前 11 个成功 cell 的路径与 manifest 身份全部进入 amendment allowlist；后续成功 cell 必须登记 amendment SHA。GPU 0 容量门控失败 attempt 保留为修订依据。

用户要求移除的最后一个 Ctrl-C 中断 attempt（`e8_fill65_temporal / seed_20260812`）已移到可恢复隔离路径 `/tmp/e8_removed_failed_formal_40b30fc6ecdc_20260925T190025Z_1c8015a2f49a`，未作为科学证据或成功 cell 使用。

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
