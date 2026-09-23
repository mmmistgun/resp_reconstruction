# E5 W0 时域前端替换：validation 结果

日期：2026-09-23。协议：`e5-temporal-frontend-v1-20260922`。

## 1. 结论

**`e5_tfe101_aa10_res_w0` 在冻结 validation 口径下属于 `net negative for this candidate`，不替换 W0，也不据此开放 test。**

候选相对同 seed W0 的五项 candidate-mean 均超过预注册 material tolerance 且方向不利：四项 error 均增加，signed PCC 明显下降。Local RR 与 PCC 为三个 seed 全部变差；受试者等权汇总也在这两项上分别有 `20/21` 和 `21/21` 个 seed×samp_id 组合变差。

该结论只约束当前 W0 专用的 `100→10 Hz` 固定抗混叠、无前端 normalization、`k11` embedding 加 zero-init depthwise residual 的完整 package，以及本次 early-stop 训练合同。它不证明旧 patch bridge 普遍最优，也不单独否定抗混叠、10-Hz 输入或卷积前端。

## 2. 来源与完整性

三个 formal attempt 为：

| Seed | Completed epoch | Selected epoch | Attempt |
|---|---:|---:|---|
| 20260811 | 30 | 9 | `runs/e5_temporal_frontend/formal/seed_20260811/formal_d7cf90ccb432_20260922T180755Z_44b4c5de8e8b` |
| 20260812 | 30 | 13 | `runs/e5_temporal_frontend/formal/seed_20260812/formal_d7cf90ccb432_20260922T184045Z_f02856cef079` |
| 20260813 | 30 | 5 | `runs/e5_temporal_frontend/formal/seed_20260813/formal_d7cf90ccb432_20260922T191356Z_0b29a1dcb165` |

三者均绑定 implementation lock SHA-256 `d7cf90ccb432f2bda4563f6719385eb98564c1594c43c8b75c10f10374dce7f5`，完成 2,400 optimizer updates、2,675 条 validation、完整 best/final checkpoint 与 optimizer state 核验；checkpoint/history 全有限，两个 prediction-degeneracy count 均为 0，validation row-order SHA-256 均为 `b68a51b101bb80033c4de18c9c21cdfbb8924d1bfe134330f047617cf3b0915a`。

三个 run 均按 `min_epoch=30 / patience=15 / min_delta=0` 在 epoch 30 触发 early stopping。一次性汇总位于：

```text
runs/e5_temporal_frontend/summary/summary_d7cf90ccb432_20260923T022609Z_937e34d10c67
```

汇总 manifest SHA-256 为 `153b581b0ec1ca0ba5b286d990bdf248bb8e7a35949e79d36331b5cf6740d7f8`。完整核验覆盖 formal manifest、来源锁、resolved config、history、LR、early-stop 轨迹、checkpoint、optimizer、逐窗口 identity/eligibility、五指标分母和 W0 配对来源。未访问 test。

## 3. 五主指标

下表为三 seed arithmetic mean ± sample SD。变化列对四项 error 使用“三 seed 均值之差 / W0 三 seed 均值”，PCC 使用绝对下降；正值统一表示 E5 更差。方向为 E5 优于 W0 的 seed 数。

| 指标 | W0 | E5 | E5 变化 | E5 改善 seed |
|---|---:|---:|---:|---:|
| Whole RR absolute error (bpm) | 0.499298 ± 0.018320 | 0.518247 ± 0.031855 | +3.7951% | 1/3 |
| Local RR MAE (bpm) | 0.551309 ± 0.012064 | 0.590073 ± 0.004570 | +7.0312% | 0/3 |
| Envelope trajectory MAE | 0.152729 ± 0.001423 | 0.155769 ± 0.006624 | +1.9903% | 1/3 |
| Global envelope modulation error | 0.191051 ± 0.003022 | 0.201522 ± 0.017101 | +5.4808% | 1/3 |
| Lag-aware signed PCC | 0.865300 ± 0.001491 | 0.848395 ± 0.001499 | −0.016905 | 0/3 |

预注册 tolerance 为 error 相对 `0.5%`、PCC 绝对 `0.002`。五项均为 material degradation，没有 material improvement，因此不是 attribute trade-off，而是本候选的 net negative。

逐 seed 可见少量局部改善：seed 20260811 的 Whole RR 改善 6.85%，seed 20260813 的 trajectory 改善 2.19%，seed 20260812 的 global-envelope 改善 0.76%。但它们没有跨 seed 稳定，同时 Local RR/PCC 在每个 seed 均达到 material degradation，不能据单点收益保留候选。

## 4. 受试者等权汇总

先在每个 seed×samp_id 内按对应 eligibility 求均值，再对 7 个 samp_id 等权。下表把三个 seed 的 subject-macro delta 再等权平均；正值仍表示 E5 更差。方向数的分母为 `3 seeds × 7 samp_id = 21`。

| 指标 | Subject-macro 平均变化 | W0 更优 | E5 更优 |
|---|---:|---:|---:|
| Whole RR | +15.5375% | 13/21 | 8/21 |
| Local RR | +11.3717% | 20/21 | 1/21 |
| Envelope trajectory | +4.3930% | 16/21 | 5/21 |
| Global envelope modulation | +5.6775% | 13/21 | 8/21 |
| Signed PCC drop | +0.018424 | 21/21 | 0/21 |

受试者等权结果与 pooled sample-direct mean 的整体判断一致，尤其 Local RR 和 PCC 的退化具有很强方向一致性。窗口重叠、同一受试者的窗口和三个模型 seed 不作为新增独立受试者；这些方向数是描述证据，不是确认性显著性检验。

## 5. 解释边界

结果说明，当前“时域前端只负责 ≤4.5-Hz 有符号局部波形，高频幅度结构交给 W 分支”的分工没有改善固定 W0。可能原因包括：

- 去掉旧前端的全窗归一化/patch mixing 后，主干接收到的表示尺度和统计结构发生了不利变化；
- 直接在任何学习非线性之前降至 10 Hz，可能丢失了 W 分支的 log-magnitude FiLM 不能完全替代的时域 carrier 调制或相位线索；
- 极简 depthwise residual 可能不足以在进入 Mamba 前形成合适的局部跨滤波器交互；
- 前端、采样、归一化和局部编码同时变化，当前结果不能在这些解释之间作因果分配。

这些是与结果一致的候选解释，不是已经验证的机制结论。不能看到负结果后立即追加多个 normalization、20-Hz carrier path、pointwise mixer 或容量变体；若继续 E5，应先提出一个新的、单一可证伪问题和对应控制。

三个 seed 都在最小允许 epoch 30 停止，而 selected epoch 为 9/13/5。这说明在 epoch 30 前已经连续至少 15 epoch 没有更好的 Local RR checkpoint；但历史 CRD 中存在晚恢复，因此仍不能把本结果外推为“80 epoch 固定训练也必然相同”。当前科学结论明确绑定本次用户指定的 early-stop 合同。

## 6. 当前决定

- 保留冻结 W0；E5 首轮候选不进入论文主模型替换。
- 后续 test 仅按专项附件评价已由 validation 固定的 checkpoint，作为复用 research-test 上的开发性描述；不改变 validation 失败结论。
- 不重跑相同 identity，不改变 selector 或 early-stop 后补跑本候选。
- 如继续研究，另立新协议；当前 summary、三个 formal attempt 和失败结论保持不可覆盖。
