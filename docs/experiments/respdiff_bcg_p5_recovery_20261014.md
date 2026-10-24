# RespDiff-BCG P5 队列恢复记录（2026-10-14）

作用域：`respdiff-bcg-training-and-sampling-v1-20261010` 第 32 节 P5 全量 validation。用户明确要求“队列中断了，帮我恢复”，沿用物理 GPU0 单卡、既有三个模型和固定配置合同。

原后台任务 `j-88wy68` 被记录为进程结束但无退出记录，可能为机器重启或外部终止；现有证据不能确定具体原因。恢复检查时已无原队列/评价进程，两张 GPU 均空闲。中断运行未留下 Python 异常或完成回执。

- `p5_full_s20260811_v1` 已于 2026-10-14 13:14（Asia/Shanghai）完整完成。49 项 artifact SHA256、三配置完整性、2675-parent/34775-chunk 身份、冻结 checkpoint 与参数状态均核验通过，继续只读复用。
- `p5_full_s20260812_v1` 日志最后记录为 13:21:30 的 DDIM 第 6/544 批，约 5 分 44 秒采样进度。采样结果当时保存在进程内存中，尚无可恢复的 waveform、采样状态或完整配置结果，因此该 seed 在新目录重新执行。
- `seed20260813` 尚未启动。原首 seed 结果、中断目录及原启动器日志全部保留；中断目录的 11 个文件 hashes 保存于恢复 preflight。

本次继续使用既有 `p5_full_manifest_v1`。其 242 项冻结实现/协议文件均匹配当前版本；训练、采样、metrics、checkpoint、完整 validation 样本和三个最终配置均保持原合同。原计划文件继续冻结，本记录作为独立执行附件保存。

| 次序 | GPU | 操作 | 输出 identity（相对 study 根目录） |
|---|---:|---|---|
| 引用 | — | 复用 seed20260811 三项完整结果 | `p5_full_s20260811_v1` |
| 1 | 0 | 重新开始 seed20260812 的全量评价 | `p5_full_s20260812_v2` |
| 2 | 0 | 执行 seed20260813 的全量评价 | `p5_full_s20260813_v1` |
| 3 | CPU | 汇总三个模型的九项完整结果 | `p5_full_summary_v1` |

新启动器位于 `runs/respdiff_bcg_study_v1/launch_p5_full_gpu0_recovery_v1/launch.py`，同目录保存 `recovery_preflight.json`、逐 seed 命令/进程/退出回执、console 日志和队列结束回执。每项仍为 DDIM6/N64、DDPM50/N16、DDIM6/N16，sampling seed20261003、B64/G1。第一项完整成功后才启动下一项，失败时停止后续队列并保留产物；两项完成后才合并首 seed 生成九项汇总。

按首 seed 已完成的实际阶段耗时约 27.1 小时，剩余两个模型预留约 **54 小时**，实际时间随系统负载变化。恢复时磁盘剩余约 241 GiB。新的队列编排通过 CPU 模拟子进程检查；底层评价源码未变，沿用原定向测试和首 seed 的完整执行证据。
