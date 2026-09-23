# ADV 系列产物归位记录

日期：2026-09-23。状态：完成。

## 1. 目的与代码身份

ADV-v1 与 ADV 融合因子实验均已关闭。两条实验线的实现、协议、来源锁和收尾报告已通过提交
`6ca07b7f0351c0bbb44828923a8086bb5be0b564` 合入 `main`。该合并包含原
`codex/aligned-dual-view-v1` 提交 `f1a4236` 和原 `codex/adv-fusion-factorial-v1`
提交 `9702913` 的完整历史。

运行产物统一归位到主仓库的 `runs/`。本次操作为同一文件系统内的目录重命名；冻结文件内容、
目录内部结构和实验 identity 均保持不变。

## 2. 产物位置

| 实验阶段 | 当前路径 |
|---|---|
| ADV-v1 train/validation | `runs/aligned_dual_view_v1/` |
| ADV-v1 research-test | `runs/aligned_dual_view_v1_research_test/` |
| ADV 融合因子 train/validation | `runs/adv_fusion_factorial_v1/` |
| ADV 融合因子 research-test | `runs/adv_fusion_factorial_v1_research_test/` |

历史 summary 和曲线审查 manifest 中保存了运行时绝对路径。当前机器通过本地兼容链接将这些路径
解析到上述目录；兼容链接不属于 Git 产物。跨机器恢复时，以本记录中的当前路径、候选锁、产物
manifest 和源码快照共同确定来源，不改写历史 manifest。

## 3. 迁移后核对

迁移后只读核对结果：

- ADV-v1：12 个锁定 checkpoint 的运行 manifest 和 checkpoint SHA-256 全部匹配；train/validation
  为 18 个完整 lifecycle，research-test 为 14 个完整 lifecycle，未发现失败 lifecycle。
- ADV 融合因子：18 个锁定 checkpoint 的运行 manifest 和 checkpoint SHA-256 全部匹配；
  train/validation 为 32 个完整 lifecycle，research-test 为 20 个完整 lifecycle，未发现失败 lifecycle。
- ADV 融合因子的历史曲线审查使用独立的只读 manifest；其 21 个来源和 4 个输出文件哈希均匹配。
- 四份冻结 summary manifest 的 SHA-256 分别为：
  - ADV-v1 validation：`cdc03f23f299b24bbf6234cd73c5c1678d7e9d7b22065e7f8cef58f9a0d1577d`
  - ADV-v1 research-test：`b21da372dfca62807831e97362c1a3cd81fe0b84657c1b5bca7972cd74b12bc3`
  - ADV 融合 validation：`ab47bb367b74c476605d70d8e2ab7b17940fbc905dbd4d1ae610be6a72c76e11`
  - ADV 融合 research-test：`892d6735f3d4dba7d9a79c187fe0c7a6d3f608230b7817009299997e1aede640`

本次核对读取生命周期回执、manifest、候选锁、锁定 checkpoint、核心 CSV、访问记录和源码身份。
大体积 CWT payload 保持原文件，不重新计算特征或重放模型。原始数据、research-test 波形和 target
未被访问。

## 4. 主分支兼容边界

正式候选锁保存运行时的完整训练源码身份。实验关闭后，`main` 中的
`resp_train/crd/experiment.py` 为 E5/E6 增加了 early-stopping `min_epoch` 支持，因此当前主分支
不再等于 ADV 正式运行时的全仓训练身份。生产锁的严格加载入口继续拒绝这一差异；这不会改变已保存
checkpoint、指标或 summary。复核正式执行源码时使用 Git 中的 `f1a4236` / `9702913` 提交，或各产物
的 `source_snapshot/`。当前主分支上的 synthetic 测试使用 disposable 数据、权重和当前源码身份，
不替换生产候选锁。

## 5. 保留边界

上述四个目录是关闭实验的冻结证据，继续遵循各自收尾协议：不得覆盖、补写、使用相同 identity
重跑或删除。恢复与复核同时依赖 Git 历史、产物内 `source_snapshot/`、候选锁和完成回执。
