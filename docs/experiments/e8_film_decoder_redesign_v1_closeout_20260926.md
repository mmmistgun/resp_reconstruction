# E8：FiLM 条件末端 × 解码端重设计结项

日期：2026-09-26。状态：**P0–P6 全部完成并关闭；36 项 formal train/validation、P5 validation 汇总、36 项固定 checkpoint research-test 评价及完整汇总均已冻结。**

本文件是 E8 的最终状态、结果与证据入口。已完成训练、评价、allowlist 和汇总不得重复执行、覆盖或补写。

## 1. 最终决定

**保留原 W0 `fill65 + pointwise`，本轮不替换 FiLM 条件末端或 decoder。**

原因：

1. Validation 中，`direct-fill65` 的边际效应表现为 tolerance-aware Pareto 改善；但 reused research-test 上方向反转，Whole RR 和 Local RR 分别恶化 1.8783% 和 1.3448%，只保留 global modulation 改善。删除 65-channel 参数填充没有形成跨 split 稳定收益。
2. Validation 最值得继续确认的简化候选 `direct+single` 在 test 上相对 W0 的 Whole RR、Local RR、trajectory 分别恶化 5.6004%、4.3970%、1.2440%，PCC 下降 0.003000；仅 global modulation 改善 1.4881%。该候选退出替换建议。
3. 设计时优先候选 `direct+temporal` 在 test 上相对 W0 的 Whole RR、Local RR、trajectory 分别恶化 6.8199%、1.0885%、0.6631%，global modulation 改善 1.7240%，PCC 增加 0.001043（容差内）。它仍是属性交换，不是全面改进。
4. Test 的 tolerance-aware Pareto 集只剩 `fill65/pointwise`、`direct/temporal`、`res96/single`。其中 W0 获得最低 Whole RR、最低 Local RR，并在 subject-macro RR/PCC 上保持明显优势；另两组分别偏向 global modulation/PCC 或 trajectory，不足以取代 W0 的综合平衡。
5. `res192` 扩宽在 test 边际上同时恶化 Whole RR、Local RR、trajectory 和 global modulation；额外末端容量没有保留依据。

因此，本轮回答了最初疑惑：**65 并非理论上不可删除，但在当前训练合同和跨 split 证据下，直接投影的 validation 优势不稳定；时间残差 decoder 也没有解决这种不稳定。工程上更简洁，不等于科研上已证明可替换。**

## 2. 证据属性

- Validation：36 个同轮从头训练 cell，三 seed 描述优化随机性。
- Research-test：36 个 validation-selected checkpoint 在任何 test 访问前由 allowlist 固定；test 不参与 checkpoint、结构或阈值选择。
- Test split 已参与既往研究开发，证据属性为 **reused research/development evidence**，不声称独立 held-out 确认。
- 窗口重叠且 seed 不是独立人群，不报告窗口级或 seed-level 人群显著性。

## 3. Research-test 五主指标

表中为三个 seed 的 window-direct arithmetic mean ± sample SD。前四项越低越好，PCC 越高越好。

| Arm | Whole RR | Local RR | Trajectory | Global modulation | PCC |
|---|---:|---:|---:|---:|---:|
| `fill65/pointwise` | **0.613661 ± 0.029358** | **0.610338 ± 0.014579** | 0.139546 ± 0.000987 | 0.173789 ± 0.006452 | 0.876463 ± 0.001347 |
| `fill65/single` | 0.681243 ± 0.050756 | 0.654627 ± 0.014465 | 0.140269 ± 0.003025 | 0.178238 ± 0.004277 | 0.873374 ± 0.001348 |
| `fill65/temporal` | 0.639216 ± 0.019910 | 0.616594 ± 0.019113 | 0.140643 ± 0.000574 | 0.172999 ± 0.004702 | 0.875505 ± 0.003932 |
| `direct/pointwise` | 0.666727 ± 0.064405 | 0.652552 ± 0.062506 | 0.140615 ± 0.003446 | 0.176416 ± 0.010858 | 0.873554 ± 0.001046 |
| `direct/single` | 0.648274 ± 0.042872 | 0.637496 ± 0.043163 | 0.141279 ± 0.001089 | 0.171114 ± 0.003129 | 0.873463 ± 0.002632 |
| `direct/temporal` | 0.656110 ± 0.053749 | 0.617129 ± 0.033399 | 0.140472 ± 0.001426 | 0.170762 ± 0.005093 | **0.877506 ± 0.000923** |
| `res96/pointwise` | 0.672815 ± 0.048460 | 0.634446 ± 0.013595 | 0.140903 ± 0.002378 | 0.174368 ± 0.002799 | 0.874825 ± 0.004638 |
| `res96/single` | 0.651235 ± 0.017358 | 0.634075 ± 0.032195 | 0.138177 ± 0.002228 | **0.166806 ± 0.001005** | 0.876012 ± 0.003934 |
| `res96/temporal` | 0.685961 ± 0.030637 | 0.641653 ± 0.035117 | **0.138146 ± 0.001226** | 0.169651 ± 0.007562 | 0.876555 ± 0.002575 |
| `res192/pointwise` | 0.682848 ± 0.027449 | 0.650545 ± 0.017314 | 0.140577 ± 0.002767 | 0.172059 ± 0.002699 | 0.875174 ± 0.001743 |
| `res192/single` | 0.688406 ± 0.018075 | 0.657704 ± 0.010880 | 0.140579 ± 0.001045 | 0.173700 ± 0.001982 | 0.875041 ± 0.003524 |
| `res192/temporal` | 0.671439 ± 0.018462 | 0.629173 ± 0.035997 | 0.141029 ± 0.005390 | 0.170892 ± 0.006683 | 0.874742 ± 0.003225 |

单项 test seed mean 最优分散在三个 arm：Whole/Local RR=`fill65/pointwise`，trajectory=`res96/temporal`，global modulation=`res96/single`，PCC=`direct/temporal`。这再次说明不存在单一全面赢家。

## 4. Test 计划边际对比

方向统一后的跨 seed mean：error 使用相对改善比例，PCC 使用绝对差；正值表示候选改善。

| Contrast | Whole RR | Local RR | Trajectory | Global modulation | PCC |
|---|---:|---:|---:|---:|---:|
| `direct-fill65` | -1.8783% | -1.3448% | -0.4580% | +1.2853% | -0.000273 |
| `res96-direct` | -2.2001% | -0.3160% | +1.2108% | +1.4193% | +0.000956 |
| `res192-res96` | -1.7176% | -1.5297% | -1.2010% | -1.1460% | -0.000811 |
| `single-pointwise` | -1.3808% | -1.4340% | +0.2350% | +0.9533% | -0.000531 |
| `temporal-pointwise` | -0.7594% | +1.6409% | +0.2363% | +1.7587% | +0.001073 |
| `temporal-single` | +0.5973% | +3.0386% | +0.0034% | +0.8048% | +0.001604 |

在 test 上，`temporal-single` 是边际 Pareto 改善，但这只说明 temporal 优于 single 的平均比较；它没有证明 temporal 优于原 W0 pointwise。`temporal-pointwise` 仍有 Whole RR 恶化，具体 `direct/temporal` 和其他 temporal arm 也没有全面超越 W0。

## 5. Validation→test 反转

最关键的跨 split 结果：

- `direct-fill65`：validation 为 Whole RR/global modulation 改善且其余容差内；test 转为 Whole/Local RR 实质性恶化。
- `direct/single`：validation window-direct 形成相对 W0 的 Pareto 改善；test 的 Whole RR、Local RR、trajectory、PCC 均转差。
- `res192/pointwise`：validation 获得最优 RR/PCC，但 test RR 明显落后 W0。
- W0：validation 不在 tolerance-aware Pareto 集，test 则进入三组 Pareto 集并获得最佳 Whole/Local RR。

这说明当前 validation 人群不足以稳定决定条件末端和 decoder 替换；不应把 validation 上的结构简化写成已泛化结论。

## 6. Subject 与 Local RR 尾部

Test subject-macro 中：

| Arm | Whole RR | Local RR | Trajectory | Global modulation | PCC |
|---|---:|---:|---:|---:|---:|
| `fill65/pointwise` | 1.320771 | 1.164372 | 0.159327 | 0.212636 | 0.834280 |
| `direct/single` | 1.448388 | 1.214320 | 0.163342 | 0.213241 | 0.830096 |
| `direct/temporal` | 1.450205 | 1.199614 | 0.161207 | 0.207837 | 0.833676 |
| `res96/single` | 1.502903 | 1.256798 | 0.157976 | 0.204089 | 0.833979 |

W0 在 Whole/Local RR 与 PCC 上保持优势。Local RR tail 也没有支持 `direct/single`：W0 与 direct/single 的 median=`0.082644/0.085151`、P90=`1.458046/1.516784`、P95=`3.159712/3.289366`、`>2 bpm=7.5758%/7.9942%`、`>5 bpm=2.9582%/3.2468%`。

## 7. 完整性与冻结身份

- Test：2,310 windows、8 subjects；36 项共 83,160 行。
- 36/36 evaluation quality acceptance 通过；每项 row identity、target eligibility 与 cache/checkpoint identity 一致。
- Whole RR、Local RR、PCC、envelope 与 test-only target eligibility 均为 2,310；prediction degeneracy 全部为0。
- Allowlist SHA-256：`76ef0c8ac05cdddcf89caab46abb0a03b78d0d3e72d09b6ed5773521dc2576db`。
- Test summary：

```text
runs/e8_film_decoder_redesign_v1/research_test/summary/
summary_76ef0c8ac05c_20260926T113826Z_0214686a8ade
```

- Test summary manifest SHA-256：`c0356ab3ae649310933e36cd4f132656c0b6fd193bd878793243412ca377eaa6`。
- Decision SHA-256：`aa27c9639f57b3f23c83420248d5df2442688d3855e5a0d2f408ec7745514180`。
- Summary receipt SHA-256：`28ae58e3fd23d4f9cb262b896c5830c6b0d86e306256e5b6363af870a3aa6cff`。
- Test summary 源码 commit：`57f28109c25ea9f5d2dae23f7f9f8e0f433aeb80`。

Test summary 只读取已冻结 evaluation metrics；没有再次读取 test arrays、checkpoint 内容或执行推理。

## 8. 工程与解释风险

- Formal 阶段实际 CUDA reserved 峰值约 81.4%–81.65%，暴露 P2 未覆盖 batch-128 validation 峰值；不影响已完成数值，但未来工程验收需补足该路径。
- Validation→test 的结构效应反转是本轮最重要的科学风险。继续在相同 research-test 上追加结构并选择赢家只会扩大 research-test-informed development 偏差。
- 若未来重新研究 direct FiLM 或 decoder，应使用新的独立人群/外部数据或预先固定的确认数据，而不是继续复用当前 test 进行迭代选择。

## 9. 关闭决定

E8 全部阶段关闭。保留 W0 `fill65/pointwise`；`direct/single`、`direct/temporal`、`res96/single` 等只作为属性取舍和机制证据，不进入当前主模型替换。所有成功与失败 lifecycle、allowlist、evaluation、summary 和 checkpoint 原位保留，不重跑、不补写、不删除。
