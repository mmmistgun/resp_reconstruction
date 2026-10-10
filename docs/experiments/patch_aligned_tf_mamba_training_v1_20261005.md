# Patch-aligned TF-Mamba 时长敏感性 train/validation 协议

协议 ID：`patch-aligned-tf-mamba-v1-20261005`。用户已确定 1/2/4 秒 patch × 三 seed 矩阵，并要求正式运行由用户执行。当前阶段为训练入口实现与合成验证，正式训练尚未启动。

## 实验合同

9 个 cell：patch 时长 1、2、4 秒；seed 为 20260811、20260812、20260813。几何统一使用 100 Hz 输入、0.5 秒 hop、0.5 秒 CWT pooling。除片段时长及其派生位置/解码尺寸外，模型使用同一配置：D=96、6 层 BiMamba2、79 点连续 stem、GroupNorm CWT 编码、有界零初始化 FiLM、正权重 Hann 合成。各 cell 的解码参数量随片段长度变化。

数据来源、split、sample seed、target、loss 与 metrics 沿用 `configs/crd_tf_v1/crd_tf102_w_formal.yaml` 及 W0 来源记录。train=10141 窗口/32 受试者，validation=2675 窗口/7 受试者。模型仅接收冻结 W cache 中实际频率 `(0.8,8] Hz` 对应的 41 行；不建立新 cache。只允许 train/val，test 不在本轮入口内。

训练为物理 batch=64、梯度累积=2、BF16，有效 batch=128。尾组按真实 eligible 数归一化，复用 `train_crd_one_epoch`，每 epoch 80 次 optimizer 更新，最大 80 epoch/6400 次更新。AdamW 的分组、lr=3e-4→3e-5、5% warmup+cosine、weight decay=1e-4、梯度裁剪 1.0 沿用基线。loss 为既有 sync+0.25×effort，累积组按各项有效样本总数独立归一化。

每 epoch 完整 validation，按 Local RR MAE 严格最小值选 checkpoint，相等时保留最早 epoch。early stopping：min_epoch=30、patience=15、min_delta=0，等待次数从训练开始累计。完成后只评价该 cell 的 best checkpoint，保存逐样本 validation metrics 与汇总。正式配置、数据口径和 selector 的改变应使用新修订及新 session。

## 运行生命周期

`prepare` 保存完整矩阵、源码 SHA-256、Git 身份及源码快照，不读取真实数据。两张 GPU 共享 session；各自串行运行固定分片，cell 文件锁防止同组重复执行。源码或配置改变后拒绝继续旧 session。

每个 cell 启动时先用合成信号、正式 task loss、正式 optimizer 分组及 64×2 累积完成两次更新与一次 loss validation。失败时该 cell 不进入真实数据训练。该验收覆盖所运行的 patch 时长；此前 2 秒 MSE 工程检查不替代此门槛。

真实运行记录 resolved config、环境、命令、样本行身份、初始化 seed、optimizer 分组、逐 epoch train/validation 指标、完整 checkpoint 与逐样本评价。epoch checkpoint 包含模型、optimizer、精确 update index、早停状态、Python/NumPy/CPU/CUDA RNG。每个完成 epoch 写入独立目录，最后写入带 checkpoint SHA-256 的 receipt。

`--resume` 从最后一个完整 epoch 边界恢复。中途未完成的 epoch 重新执行；所有已有 epoch 产物保持原位。num_workers=0、固定数据及保存 RNG 支持恢复后的采样/dropout连续性。同配置可迁移到另一 CUDA 设备；正式恢复仍应保持相同软件和硬件条件。每 epoch 保存一次完整模型及 optimizer，9-cell 最大约需数至十余 GiB checkpoint 空间。

完成 cell 再次调用会校验并复用成功产物；未完成 cell 需要显式 `--resume`。汇总要求完整 9-cell，输出每 seed 及跨 seed mean/std。历史科研产物、当前用户其他实验目录不参与写入。

## 实现验证

2026-10-05，14 项定向 CPU 测试通过，覆盖矩阵/分片、基线 loss 与指标合同、64×2 更新预算、session 身份和排他写入、正式 task loss 累积路径、epoch 中断恢复、严格改善与早停、完整矩阵汇总及产物校验。恢复测试确认模型参数及 AdamW 状态与连续运行逐项一致；CPU 模型检查使用显式 Mamba 替身。shell 脚本通过 `bash -n`。

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=. /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python -m pytest -q tests/test_patch_aligned_tf_training.py tests/test_crd_training.py
```

1/2/4 秒正式 loss 的官方 CUDA 验收由用户启动入口后自动执行；正式 train/validation 尚未运行。

## 用户运行命令

在本 worktree 根目录执行。以下 session 名必须与其他独立实验区分；已建同身份 session 可重复执行 prepare 检查身份。

```bash
cd /home/marques/.codex/worktrees/model-architecture-review/resp_reconstruction
PATCH_TF_PYTHON=/mnt/disk_code/marques/resp_reconstruction/.venv/bin/python
PATCH_TF_SESSION="$PWD/runs/patch_aligned_tf_mamba/formal_v1_20261005"
env -u LD_LIBRARY_PATH -u LD_PRELOAD "$PATCH_TF_PYTHON" scripts/run_patch_aligned_tf_mamba.py plan
env -u LD_LIBRARY_PATH -u LD_PRELOAD "$PATCH_TF_PYTHON" scripts/run_patch_aligned_tf_mamba.py prepare --session "$PATCH_TF_SESSION"
```

终端 1：

```bash
bash scripts/run_patch_tf_shard.sh "$PATCH_TF_SESSION" 0 0
```

终端 2（重新设置同一个 session 绝对路径）：

```bash
cd /home/marques/.codex/worktrees/model-architecture-review/resp_reconstruction
PATCH_TF_SESSION="$PWD/runs/patch_aligned_tf_mamba/formal_v1_20261005"
bash scripts/run_patch_tf_shard.sh "$PATCH_TF_SESSION" 1 1
```

shell 脚本内部使用 `env -u LD_LIBRARY_PATH -u LD_PRELOAD`。等价的直接分拆命令如下，可在两个终端分别启动：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 "$PATCH_TF_PYTHON" -u scripts/run_patch_aligned_tf_mamba.py run --session "$PATCH_TF_SESSION" --shard-index 0 --shard-count 2 --device cuda:0
env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 "$PATCH_TF_PYTHON" -u scripts/run_patch_aligned_tf_mamba.py run --session "$PATCH_TF_SESSION" --shard-index 1 --shard-count 2 --device cuda:1
```

恢复时在上述命令末尾加 `--resume`。单组运行使用 `run --session ... --patch-seconds 4 --seed 20260811 --device cuda:1`；不能与分片参数同时指定。

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD "$PATCH_TF_PYTHON" scripts/run_patch_aligned_tf_mamba.py status --session "$PATCH_TF_SESSION"
env -u LD_LIBRARY_PATH -u LD_PRELOAD "$PATCH_TF_PYTHON" scripts/run_patch_aligned_tf_mamba.py summarize --session "$PATCH_TF_SESSION"
```

默认分片：GPU0 执行 p1s/seed11、p4s/seed11、p2s/seed12、p1s/seed13、p4s/seed13；GPU1 执行 p2s/seed11、p1s/seed12、p4s/seed12、p2s/seed13，其中 seed11/12/13 分别表示上述三个完整 seed。
