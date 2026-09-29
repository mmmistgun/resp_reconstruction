# E5 W0 时域前端替换：结项记录

日期：2026-09-27。状态：**实现、三 seed formal train/validation、validation 汇总、三个固定 checkpoint 的 reused research-test 评价及 test 汇总均已完成并冻结；E5 不替换 W0，实验关闭。**

本文件是 E5 的当前状态、结果与证据入口。训练结果报告和 test 协议保留各自形成时的阶段状态；其中 test 协议开头的“尚未执行”是进入 test 前的历史快照，并已绑定 test lock，保持原文用于复核。实际完成状态以本结项、原始 lifecycle、evaluation receipt、summary receipt 和 manifest 为准。

## 1. 最终决定

E5 候选 `e5_tfe101_aa10_res_w0` 在 validation 和 reused research-test 上均不替换 W0：

1. Validation 的五项候选均值全部不利；Local RR 与 PCC 为三个 seed 全部退化，结论为当前完整前端 package 的 `net negative`。
2. Research-test 进一步扩大 Whole RR、Local RR 和 global modulation 差距；这三项及 PCC 均为 W0 在三个 seed 全部更优，trajectory 为 W0 在两个 seed 更优。
3. Subject-macro 在 test 五项指标上同样整体不利；该结果不是少数高窗口数 subject 单独驱动。
4. Test 使用 validation 已固定的 checkpoint，不重选 epoch、不追加训练，也不改变 validation 决定。
5. E5 只约束本轮 `100→10 Hz` 固定抗混叠、无前端 normalization、`k11` embedding 与 zero-init depthwise residual 的完整 package；不证明 patch 前端普遍最优，也不能把退化归因于单一组件。

Research-test 已被既往研究重复使用，证据属性为 **reused research/development evidence**，只作固定候选的开发性复核，不是首次 held-out 或独立确认。

## 2. 固定实验范围

- 候选：`e5_tfe101_aa10_res_w0`；对照：相同 seed 的冻结 W0。
- Seeds：`20260811`、`20260812`、`20260813`。
- Train/validation：10,141/2,675 窗口；test：每 seed 2,310 窗口、8 个 `samp_id`。
- Formal run 均在 epoch 30 合法 early-stop，完成 2,400 optimizer updates；selected epoch 分别为 9、13、5。
- Test 严格加载三个 validation-selected `checkpoint_best_local_rr.pt`，共新增 6,930 条逐窗口指标。
- Test 三项 evaluation 的主指标均有限、prediction degeneracy 均为 0、quality acceptance 均通过，target eligibility 与 W0 完全一致。

## 3. Validation 与 research-test 结果

下表为三 seed arithmetic mean。四项 error 的变化为 `(E5−W0)/W0×100%`，PCC 为 `W0−E5`；正值表示 E5 更差。

| 指标 | Validation W0 | Validation E5 | Validation 变化 | Test W0 | Test E5 | Test 变化 | Test W0 更优 seed |
|---|---:|---:|---:|---:|---:|---:|---:|
| Whole RR absolute error | 0.499298 | 0.518247 | +3.7951% | 0.617234 | 0.755534 | +22.4064% | 3/3 |
| Local RR MAE | 0.551309 | 0.590073 | +7.0312% | 0.609566 | 0.713314 | +17.0200% | 3/3 |
| Envelope trajectory MAE | 0.152729 | 0.155769 | +1.9903% | 0.139546 | 0.143666 | +2.9520% | 2/3 |
| Global envelope modulation error | 0.191051 | 0.201522 | +5.4808% | 0.173418 | 0.189031 | +9.0026% | 3/3 |
| Lag-aware signed PCC | 0.865300 | 0.848395 | +0.016905 | 0.876577 | 0.864083 | +0.012494 | 3/3 |

Validation 中的少量单 seed 局部改善没有跨 seed 稳定。Test 中 trajectory 仍有一个 seed 的候选更优，但其三 seed 均值及其他四项均不支持候选替换。

## 4. Test subject-macro

先在每个 seed×`samp_id` 内按 eligibility 聚合，再对 8 个 `samp_id` 等权。下表把三个 seed 的 subject-macro delta 再等权平均；正值表示 E5 更差，方向分母为 `3×8=24`。

| 指标 | Subject-macro 平均变化 | W0 更优 | E5 更优 |
|---|---:|---:|---:|
| Whole RR | +10.5369% | 14/24 | 10/24 |
| Local RR | +11.0186% | 20/24 | 4/24 |
| Envelope trajectory | +3.5001% | 16/24 | 8/24 |
| Global envelope modulation | +8.8160% | 18/24 | 6/24 |
| Signed PCC drop | +0.009146 | 20/24 | 4/24 |

这些方向数是描述性证据，不是把窗口、subject 或训练 seed 当作独立人群的显著性检验。

## 5. 冻结身份与产物

| 内容 | 路径或身份 |
|---|---|
| 训练协议 | `docs/experiments/e5_temporal_frontend_protocol_20260922.md` |
| Validation 结果 | `docs/experiments/e5_temporal_frontend_results_20260923.md` |
| 训练 implementation lock SHA-256 | `d7cf90ccb432f2bda4563f6719385eb98564c1594c43c8b75c10f10374dce7f5` |
| Formal 产物 | `runs/e5_temporal_frontend/formal/seed_{seed}/{attempt}`，3/3 完成 |
| Validation summary | `runs/e5_temporal_frontend/summary/summary_d7cf90ccb432_20260923T022609Z_937e34d10c67` |
| Validation summary manifest SHA-256 | `153b581b0ec1ca0ba5b286d990bdf248bb8e7a35949e79d36331b5cf6740d7f8` |
| Research-test 协议 | `docs/experiments/e5_temporal_frontend_test_protocol_20260923.md` |
| Research-test implementation lock SHA-256 | `5b452091dc28725786cfa93225446e78a48cca89c938c079d3953c986e77c109` |
| Test 入口实现提交 | `75281b6a16d2048ed31f5b435b942a50d649d2f7` |
| Test 来源锁提交 | `d58495a61fdc34d83e352a128d3a95f123117e74` |
| Test evaluation | `runs/e5_temporal_frontend_test/evaluation/seed_{seed}/{attempt}`，3/3 完成 |
| Test summary | `runs/e5_temporal_frontend_test/summary/summary_5b452091dc28_20260923T031427Z_c128f3353d19` |
| Test summary manifest SHA-256 | `ebfea505978d9fa55e3fa22607b11778588661a65f57b8e2fb3759ecfdf072d1` |
| Test summary receipt SHA-256 | `1dde8f5ba90ddb4a8e0beb823fe0ec6c88afcab0d4cd132f943099cfba8b031b` |

三个 test evaluation 分别固定 epoch 9、13、5，manifest SHA-256 为：

- seed 20260811：`124f832ba4b2acdc422f22f40ffe53748437ff01bf0829c30f6268ea887a67a8`；
- seed 20260812：`e0eaed8e9c340c63cf27af6b553e88a3a096b18d80f5e922e55591177a2ef24e`；
- seed 20260813：`4093e5e6dbd6c8f20afa1b6ca882a3c783dcabde05ee23c0670604624f10b0f6`。

Test summary 只读取三项已完成 evaluation 和冻结 W0 指标，没有再次执行推理。逐窗口 metrics、test rows、环境、access receipt、evaluation receipt、配对 seed/subject 表及完成回执均保存在上述目录。

## 6. 关闭与后续边界

- E5 的 formal、validation、test evaluation 和 summary 均关闭，不重跑、不覆盖、不补写。
- 原 test 协议中的执行前状态和命令作为历史合同保留；本文件只更新当前状态，不修改该锁定协议。
- E5 不产生新的主模型，不改写 W0、E6 或后续结构实验的冻结身份。
- 后续前端研究须建立新协议、独立实现身份与独立输出目录，并明确其与 E5/E6 已知结果及 reused research-test 偏差的关系。
