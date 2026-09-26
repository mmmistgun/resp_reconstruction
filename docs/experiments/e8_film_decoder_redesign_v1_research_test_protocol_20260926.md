# E8：固定 36-checkpoint research-test 协议

日期：2026-09-26。协议 ID：`e8-film-decoder-redesign-v1-research-test-20260926`。

状态：**专项代码与协议正在实现；当前只允许生成/核验 checkpoint allowlist，不授权读取 test 数组。真正评价必须在 allowlist 冻结后由用户再次明确授权，并显式传入 `--confirm-research-test`。**

## 1. 证据角色与问题

本阶段固定评价 E8 validation 前已完成的全部 `12 arms × 3 seeds=36` 个 validation-selected checkpoint，描述结构效应能否跨到既有 research-test split。该 split 已参与既往研究开发，因此证据角色固定为 **reused research/development evidence**，不构成新的独立 held-out 确认。

Test 不用于：

- 重选 arm、seed、epoch 或 checkpoint；
- 修改材料性阈值、指标、eligibility、FiLM/decoder 结构或训练合同；
- 回改已冻结的 validation 结论；
- 只挑选 validation 有利的 `direct/single` 或其他子集。

## 2. 固定来源

- E8 validation 结果：`docs/experiments/e8_film_decoder_redesign_v1_validation_results_20260926.md`。
- P5 summary：`runs/e8_film_decoder_redesign_v1/summary/summary_40b30fc6ecdc_20260926T081641Z_d1cb238ad1b5`。
- P5 manifest SHA-256：`3d9ad92fa42df7d2f0e9e66a20a134b4328e62d3e3940c04c15bdafa7805871f`。
- P5 decision SHA-256：`1743548f3894cae2238827016cce0fef1fe9fc50ef46b48b3a1ca65794388b51`。
- Formal execution lock：`40b30fc6ecdcc9b40750c393be8fd2faf4e223da0566432df739aacb513713c5`。
- Runtime amendment：`517bdf281a0db0902e0c10481b0ea210a7c514d627eb6397e9e982d95ba270e3`。

Allowlist 必须逐项固定 checkpoint、config、history、formal manifest/receipt、selected epoch、development subject 集及文件 SHA。生成 allowlist 时只读取 validation 产物、checkpoint 字节和 test cache manifest，不读取 dataset index 内容或 test arrays。

## 3. Test 固定合同

- Split=`test`，2,310 windows，8 个 `samp_id`，sample seed=`20260612`。
- 冻结 W cache：97 scales × 360 time bins，逐 row identity 与 frequency file 沿用现有 CRD-TF research-test cache lock。
- Batch=128、BF16、完整尾 batch、无 shuffle；模型数学结构与 checkpoint state 不改变。
- 每个 checkpoint 只执行 input-only inference；随后按固定任务算子计算 test metrics。
- 五主指标、eligibility、Local RR、PCC、trajectory、global modulation 口径与 validation 相同；同时保留 test-only IBI、coherence、nDTW 与分层 envelope Spearman。
- 非有限 input、target、W、prediction、checkpoint 或关键指标显式失败；不丢弃样本后继续。
- 36 个 cell 必须全部完成后一次性汇总；不依据 partial test 改变剩余队列。

## 4. 阶段门控

### P6a：Allowlist

在代码提交、工作树干净后执行：

```bash
./.venv/bin/python scripts/run_e8_film_decoder_redesign_v1_test.py prepare-allowlist
```

允许访问：P5 validation summary、36 个 formal receipt/config/history/checkpoint、test cache manifest。禁止解码或加载 test arrays。

### P6b：评价

只有 allowlist 路径与 SHA 固定、当次用户明确授权后，才提供并执行 36-cell 评价命令。每个 cell 使用独立不可覆盖 lifecycle；已成功 cell 拒绝重跑，失败现场保留。

### P6c：汇总

仅在 `completed=36 / pending=running=failed=0` 后开放。汇总只读 36 份 test metrics，不再加载 checkpoint 或 test arrays；输出 test 主指标、计划对比、interaction、subject-macro、Local RR 尾部、secondary metrics 及 validation→test 差值。

## 5. 解释规则

Test 汇总沿用 validation 的 0.5% error / 0.002 PCC 材料性阈值，但只作描述性跨 split 复核。重点报告：

1. `direct-fill65` 的 validation 边际 Pareto 改善是否保持；
2. `direct/single` 的 window-direct 收益、subject-macro PCC 风险与 seed 不稳定是否重复；
3. temporal decoder 的 RR—global modulation 交换是否保持；
4. Pareto 集与材料性交互如何变化；
5. 全部 36 个固定 checkpoint 的 test 结果，而非新一轮结构选择。

无论 test 结果如何，都不把 reused research-test 写成独立确认，不按 test 单项最好值创建新主模型。
