# M4-S1 CWT 参数与模块消融：正式 train/validation 执行

日期：2026-10-20。设计 ID：`m4-s1-cwt-parameters-v1-20261020`。

用户已授权启动完整84-cell正式train/validation，使用两张空闲GPU并行，按冻结协议保存产物；research-test暂不开放。本记录将执行授权推进到正式训练阶段，科学合同、冻结源码与配置保持原定义。[设计协议](m4_s1_cwt_parameters_v1_design_20261020.md)中的先前阶段授权状态保留为冻结时记录；已关闭的[数据与GPU验收](m4_s1_cwt_parameters_v1_acceptance_results_20261020.md)只读复用。

## 执行配置

- 28配置 × seeds `20260812/20260813/20260814`，全部从头训练。
- 每cell微批32、累积4、BF16；最多80 epoch，每epoch80次更新。
- 早停：最少30 epoch、patience15、min_delta0；完整validation Local RR严格最小选点，并列保留最早epoch。
- 两个独立worker按冻结 `plan(shard_index, 2)` 分片，每卡42个cell；已核验并集84、交集为空。运行中不按partial结果调整矩阵、顺序、阈值或预算。
- GPU0：RTX 4070 Ti SUPER，UUID `GPU-ce2f5f29-2322-0e50-3627-4bc7ac242aac`，首cell `Q_mu6_v4_seed20260812`。
- GPU1：RTX 4070 Ti SUPER，UUID `GPU-4bef190e-71da-30f7-1ffd-e3d9e60dd5c2`，首cell `Q_mu6_v8_seed20260812`。
- 两卡均通过启动前空闲检查；可用磁盘约227 GiB、可用主存约90 GiB。

两worker设置 `CUDA_VISIBLE_DEVICES=0,1`，分别使用 `cuda:0` 和 `cuda:1`。训练入口按设备与环境核验B及当前臂的合成验收；GPU0已有成功回执只读复用，GPU1缺少的设备验收由入口在相应cell训练前完成。

## 产物与生命周期

沿用已冻结的train/validation session，不重建数据与缓存：

`/mnt/disk_code/marques/resp_reconstruction/runs/m4_s1_cwt_parameters_v1/acceptance_v1_20261020`

`session.json` SHA256：`87b201abb30dea105ca8c9ada5c2f48a33e1f5ebd47b78af11276f67e93740cc`。

正式产物位于该session的 `cells/<arm>_seed<seed>/`。入口保存配置、初始化、环境、数据访问与缓存来源、optimizer分组、每个完整epoch的checkpoint/record/receipt、完整history、最优与最终checkpoint引用、逐窗口validation指标、资源记录及完成manifest；失败写入独立attempt并保留现场。

启动记录：

`/mnt/disk_code/marques/resp_reconstruction/runs/m4_s1_cwt_parameters_v1/formal_launch_v1_20261020/`

`worker.py`只负责调用冻结训练入口并保存进程生命周期。各 `gpu0/`、`gpu1/` 目录保存 `launch.json`、`started.json`、`console.log`，退出时保存 `completed.json` 或 `failed.json`。启动manifest另存完整两份cell清单、用户授权、入口与session身份。后台任务持续执行到各自分片完成或显式失败。

单worker失败会保留日志并停止其分片；不得静默跳过cell。恢复时先定位失败原因，再使用冻结入口的 `--resume` 从完整epoch边界恢复；成功cell只读核验。完成全部84-cell之前不生成完整结果结论。research-test须另行授权和专项合同。

## 命令

两个worker各调用：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES=0,1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python -u \
  scripts/run_m4_s1_cwt_parameters.py run \
  --session /mnt/disk_code/marques/resp_reconstruction/runs/m4_s1_cwt_parameters_v1/acceptance_v1_20261020 \
  --device cuda:0 --shard-index 0 --shard-count 2 --confirm-training
```

第二个worker将 `--device cuda:0 --shard-index 0` 换为 `--device cuda:1 --shard-index 1`；工作目录为 `/mnt/disk_code/marques/resp_reconstruction/.worktrees/m4-s1-cwt-parameters-v1`。
