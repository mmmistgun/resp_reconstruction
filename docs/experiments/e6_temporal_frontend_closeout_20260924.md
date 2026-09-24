# E6 20-Hz 学习解调前端：结项记录

日期：2026-09-24。

状态：**E6 实现、GPU 验收、效率测量、三 seed train/validation、validation 汇总、三 seed reused research-test 和 test 汇总均已完成并冻结。候选 `e6_tfe201_aa20_demod10_w0` 不替换 W0；E6 实验关闭。**

## 1. 最终决定

E6 检验的完整结构为：

```text
100-Hz input
→ fixed anti-aliased 100→20 Hz
→ learned carrier-sensitive filtering + channel-only normalization + SiLU at 20 Hz
→ fixed anti-aliased 20→10 Hz
→ [B,96,1800]
→ frozen W0 trunk / W-FiLM / refinement / decoder
```

Validation 显示 pooled envelope trajectory MAE 改善，但 Whole RR、Local RR、global envelope 和 PCC 退化；subject-macro 未复现 trajectory 收益。Reused research-test 上五项 candidate mean 全部不利，RR 与 global-envelope 退化进一步扩大。因此：

- 冻结 W0 继续作为保留模型；
- E6 不进入模型替换、论文主模型或后续 checkpoint 竞争；
- 不追加相同结构的 seed、epoch、test 重跑或 selector 调整；
- E6 结果可作为时域前端结构证据，但不能证明 exact `TemporalStem` 中任一单独组件的因果作用；
- W0 `2×2×2` 结构因子实验具有独立协议、worktree 和输出 identity，不复用 E6 生命周期。

## 2. 冻结身份与产物索引

### 2.1 实现与工程阶段

- 训练协议：`e6-temporal-frontend-v1-20260923`。
- 训练 implementation lock：`docs/experiments/e6_temporal_frontend_implementation_lock_20260923.json`。
- 训练 lock SHA-256：`d139aca0a26bd7fb1715e7f1fbd802e113e2e9cbc347ed17b4d2133ff613aea3`。
- 实现提交：`4142582cec2a0a32ba674535d4c5cb7e0d279af3`。
- 锁提交：`b3294200d6dacefe568f62d8ccaf71b39d1ffc7a`。
- GPU acceptance：`runs/e6_temporal_frontend/gpu_acceptance/gpu_acceptance_d139aca0a26b_20260923T070435Z_fe7afb8289bf`；manifest SHA-256 `2216086832a2273fc3cf3e99fbb2e9a7f860c2e13ac7f2204bc71c748c4827c7`。
- Benchmark：`runs/e6_temporal_frontend/benchmark/benchmark_d139aca0a26b_20260923T070601Z_3d56a0bded6e`；manifest SHA-256 `0cf81831e3f644509a5830fe98fb51afc21c89e5294b107a4424afaaad81c916`。

GPU acceptance 覆盖三个 seed 的 batch-1 三步更新与一个 batch-128 三步更新，公共 W0 state、两级 shape、全部前端梯度/参数更新、finite 和显存合同均通过。

### 2.2 Formal validation

| Seed | Completed epoch | Selected epoch | 完成 attempt |
|---|---:|---:|---|
| 20260811 | 30 | 4 | `runs/e6_temporal_frontend/formal/seed_20260811/formal_d139aca0a26b_20260923T071617Z_61bfe5eca7d9` |
| 20260812 | 30 | 4 | `runs/e6_temporal_frontend/formal/seed_20260812/formal_d139aca0a26b_20260923T111319Z_21267b72bee0` |
| 20260813 | 30 | 4 | `runs/e6_temporal_frontend/formal/seed_20260813/formal_d139aca0a26b_20260923T114928Z_c3aa1615a422` |

三者均按 `min_epoch=30 / patience=15 / min_delta=0` 在 epoch 30 early-stop，完成 2,400 optimizer updates；selected checkpoint 均为 epoch 4。每 seed 完整评价 2,675 个 validation 窗口，checkpoint/history 全有限，prediction-degeneracy count 为 0，row identity 一致。

Validation summary：

```text
runs/e6_temporal_frontend/summary/summary_d139aca0a26b_20260923T143709Z_4382db584e0d
```

Manifest SHA-256：`66b93fee3e1a66d7807528e80a4c7236895501a0d38848653944063861d1b0c4`。

三个 seed-20260812 工程失败 attempt 原地保留：两次 clean-worktree gate 失败，一次 GPU acceptance/formal commit identity 不一致；均不进入正式汇总。

### 2.3 Reused research-test

- Test 协议：`e6-temporal-frontend-test-v1-20260924`。
- Test implementation lock：`docs/experiments/e6_temporal_frontend_test_lock_20260924.json`。
- Test lock SHA-256：`f16461fbfba40f022b633b3dff67c460a802b436982ef1b64297c46907dccfd9`。
- Test 实现提交：`fce0c4a720b31bc2953cb49c63765d0d62585330`。
- Test 锁提交：`e182ba37e40751eb21f05dedc1ea30adcaa0143f`。

| Seed | Fixed epoch | 完成 evaluation attempt |
|---|---:|---|
| 20260811 | 4 | `runs/e6_temporal_frontend_test/evaluation/seed_20260811/evaluation_f16461fbfba4_20260924T031045Z_b0e8b4364d18` |
| 20260812 | 4 | `runs/e6_temporal_frontend_test/evaluation/seed_20260812/evaluation_f16461fbfba4_20260924T031200Z_99405798be57` |
| 20260813 | 4 | `runs/e6_temporal_frontend_test/evaluation/seed_20260813/evaluation_f16461fbfba4_20260924T031314Z_0643332dee48` |

Test summary：

```text
runs/e6_temporal_frontend_test/summary/summary_f16461fbfba4_20260924T031751Z_bb389e6cef28
```

Manifest SHA-256：`1cb7a77e64302b309b36d4550399212e8e1b998b92a48e7cfb570f7ee76b49fd`。每 seed 2,310 窗口、8 个 `samp_id`，共新增 6,930 条逐窗口指标；五主指标完整有限，quality acceptance 全部通过，target eligibility 与 W0 一致。

## 3. Validation 与 test 主结果

变化列对四项 error 使用 `(E6−W0)/W0×100%`，PCC 使用 `W0−E6`；正值统一表示 E6 更差。

| 指标 | Validation W0 | Validation E6 | Validation 变化 | Test W0 | Test E6 | Test 变化 |
|---|---:|---:|---:|---:|---:|---:|
| Whole RR absolute error | 0.499298 | 0.509212 | +1.9856% | 0.617234 | 0.748268 | +21.2292% |
| Local RR MAE | 0.551309 | 0.576253 | +4.5246% | 0.609566 | 0.739197 | +21.2661% |
| Envelope trajectory MAE | 0.152729 | 0.148718 | −2.6263% | 0.139546 | 0.140186 | +0.4580% |
| Global envelope modulation error | 0.191051 | 0.199689 | +4.5215% | 0.173418 | 0.192367 | +10.9262% |
| Lag-aware signed PCC | 0.865300 | 0.849693 | +0.015607 drop | 0.876577 | 0.865360 | +0.011217 drop |

Validation 中 Local RR/PCC 为 `0/3` seed 改善；test 中 Whole RR、Local RR、global envelope 和 PCC 均为 `0/3` 改善。Validation pooled trajectory 的局部改善在 test 变为轻微退化。

## 4. Subject-macro 稳健性

| 指标 | Validation subject-macro | Validation W0/E6 更优 | Test subject-macro | Test W0/E6 更优 |
|---|---:|---:|---:|---:|
| Whole RR | +15.4901% | 10/11 | +9.3039% | 12/12 |
| Local RR | +7.5204% | 14/7 | +13.5131% | 18/6 |
| Envelope trajectory | +2.0269% | 14/7 | +1.0595% | 12/12 |
| Global envelope modulation | +6.6744% | 13/8 | +10.6404% | 19/5 |
| Signed PCC drop | +0.018241 | 20/1 | +0.006680 | 19/5 |

Validation 为 `3 seeds × 7 samp_id=21` 个单元，test 为 `3×8=24`。Subject-macro 在两个 split 的五项指标上均不利，且 validation pooled trajectory 收益没有通过受试者等权检查。

## 5. 效率记录

Benchmark 在 RTX 4070 Ti SUPER 上以三组交替顺序、每组 20 次计时完成。以下为合并 60 次记录的描述性中位数与三组最大显存：

| 场景 | W0 | E6 | E6 相对变化 |
|---|---:|---:|---:|
| batch-1 eval latency | 20.892 ms | 20.431 ms | −2.21% |
| batch-128 train-update latency | 554.479 ms | 618.927 ms | +11.62% |
| train peak allocated | 9.200 GB | 9.987 GB | +8.55% |
| train peak reserved | 10.612 GB | 11.409 GB | +7.51% |

Eval 三组中有一组 E6 较慢，因此 pooled eval 差异只作运行记录。训练侧三组均显示 E6 增加时间与显存；效率不能补偿质量结论。

## 6. 科学解释边界

相对 E5，E6 validation 的五项候选均值均有恢复，说明在最终压至 10 Hz 前保留 learned nonlinearity 是合理的结构方向。但 E6 相对 W0 的 Local RR、PCC 和 subject-macro 均不利，reused research-test 也没有复现 validation pooled trajectory 收益。

该结果约束的是 exact RTM `TemporalStem` 与冻结 W0 其余模块组成的完整 package。E5/E6 的差异还包括宽度、归一化和局部编码，因此不能把恢复量单独归因于 20-Hz 中间网格。Test split 已被历史工作重复使用，只能作为开发性复核，不升级为独立确认性证据。

## 7. 关闭与保留规则

- 所有实现锁、checkpoint、metrics、summary、benchmark、失败 lifecycle 和 access/provenance 回执原地保留。
- 已完成身份不重跑、不覆盖、不补写；E6 入口只作历史复现索引。
- 不据 partial seed、test 或跨实验结果修改 E6 selector、early stopping、候选结构或判断阈值。
- E6 不产生新的主模型 checkpoint，不改写冻结 W0、E4 或论文主表。
- 后续结构研究须使用新协议、独立 implementation lock、独立 worktree/输出 identity，并明确其是否受到 E6 validation/test 结果启发。
