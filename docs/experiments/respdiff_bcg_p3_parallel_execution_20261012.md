# RespDiff-BCG P3：双 GPU 并行调度修订

当前状态：本次并行执行已按用户要求停止；B1 保留恢复点，B2 CUDA 失败目录已清理并独立保留证据，详见主计划第 21 节。下文为此次调度的历史执行记录。

2026-10-12 用户明确要求：“把另一个任务提交到GPU1并行。这样就不用等太久”。本修订仅替换[训练与采样计划第 20 节](respdiff_bcg_training_and_sampling_plan_v1_20261010.md#20-p3-继续执行决定与后台队列2026-10-12)的顺序调度安排；两臂模型、数据、seed、更新预算、学习率、快照、评价与方案选择规则沿用原合同。

| 训练臂 | 物理 GPU | 运行 identity | 启动时状态 |
|---|---|---|---|
| B1 long_absolute | 0，UUID `GPU-ce2f5f29-2322-0e50-3627-4bc7ac242aac` | `p3_long_absolute_s20260811_v1` | 原训练 PID 260849 继续执行 |
| B2 long_stretched | 1，UUID `GPU-4bef190e-71da-30f7-1ffd-e3d9e60dd5c2` | `p3_long_stretched_s20260811_v1` | 2026-10-12 00:52:03（Asia/Shanghai）启动，PID 271197 |

B2 使用 `CUDA_VISIBLE_DEVICES=1`，进程内逻辑设备为 `cuda:0`。执行入口、环境与命令记录在 `runs/respdiff_bcg_study_v1/launch_p3_parallel_v1/b2_process.json` 和对应运行产物中。

新调度任务为 `j-ulrij7`，入口 `runs/respdiff_bcg_study_v1/launch_p3_parallel_v1/parallel.py`，综合日志 `console.log`，B2 控制台日志 `b2_console.log`。原任务 `j-qqtm83` 的调度 bash（PID 260846）已单独暂停，B1 子进程保持运行；原 bash 不再提交 B2 或执行方案选择。新调度在确认 B1 进程自然退出后结束旧调度 bash。此时原任务的调度退出码可能非零，B1 的科研执行状态以自身 receipt 为准。

并行调度等待两臂完成，核对两个 complete/train/16000-update 回执后，唯一执行 `p3_selection_v1`。任一臂失败时保留其产物，另一臂继续运行，训练方案选择停止。

正在运行的源码与主计划保持原冻结 hash；主计划 SHA256 为 `8170e5cb9a323b95a95155a9b5b9338044348b94e93b5996fede465cc7d3919c`。本资源调度修订单独保存，完整接管记录位于 `launch_p3_parallel_v1/handoff.json`。两臂结果复核时合并读取本修订；记录的墙钟包含并行资源环境，不作为独立性能 benchmark。

启动前已核对 GPU1 空闲、目标 PID 与父子关系、B2/selection identity 尚未创建，以及新调度脚本语法。此次没有修改训练数值实现，也未重跑已关闭的验收或实验。
