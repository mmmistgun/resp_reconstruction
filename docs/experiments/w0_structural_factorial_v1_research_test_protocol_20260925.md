# W0 三因素结构对照：固定 checkpoint research-test 协议

日期：2026-09-25
协议 ID：`w0-structural-factorial-v1-research-test-20260925`
状态：T0 实现与 synthetic CPU 定向验证完成；真实 research-test 访问等待单独授权

## 1. 目的与证据属性

本阶段把 P4 前已由 validation 固定的八组 × 三 seed 共 24 个 checkpoint 应用于完整 research-test。Test split 已参与既有研究开发，因此本阶段结果属于 **reused research/development evidence**，用于描述结构效应在既有 test 数据上的表现，不表述为独立 held-out 确认。

本阶段固定模型、checkpoint、阈值、指标、eligibility、样本集合和分析函数。不根据任何 test 结果更换 epoch、增删 arm、改变训练合同或重训模型。Validation P4 的结构判断保持原身份；test 汇总另行报告描述性一致与差异。

## 2. 固定评价矩阵

候选集合为 P4 `summary_32141eab672e_20260925T072509Z_dbcda95df316` 的完整 24-cell 来源矩阵。每项只使用对应 formal attempt 的 `checkpoint_best_local_rr.pt`，selected epoch 由完整 validation Local-RR 最小值预先确定。

`prepare-allowlist` 从 P4 source manifest 逐项复核：

- arm、seed、formal manifest 和训练源码身份；
- resolved config、完整 history、early stopping 与 selected epoch；
- checkpoint epoch/config/state 和文件 SHA-256；
- train/validation `samp_id` 集合；
- P4 summary manifest 与决策身份。

Allowlist 使用 manifest/receipt 保存，不建立新的实验锁；其 preparation identity 绑定 P4 manifest/decision、训练实现锁、test cache manifest、干净 Git commit 和实现文件身份。评价 identity 由 allowlist SHA-256、resolved config、checkpoint/cache identity 和输出 manifest 共同确定。

## 3. Research-test 数据合同

沿用冻结 CRD-TF research-test：

- split：`test`；
- 完整窗口数：2,310；
- `samp_id`：8 个；
- sample seed：`20260612`；
- 有序 row-ID SHA-256：`184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8`；
- W 表示：`float32 [2310,97,360]`，使用冻结 input-only cache；
- physical batch：128，BF16 AMP，eval/no-grad。

Allowlist 阶段只读取 validation/P4 产物、checkpoint 和 test cache manifest，不打开 test 数组。实际评价前重新核验 dataset index、cache 文件、完整 row 顺序及 train/validation/test subject 隔离。模型 forward 只接收 BCG 与 W；target 仅进入冻结指标计算。

## 4. 指标与分析

逐窗口输出五主指标、IBI/coverage、分层 envelope Spearman、respiratory-band coherence 与 constrained nDTW。五主指标继续使用 eligible sample direct mean；每个 arm 先逐 seed 汇总，再报告三 seed arithmetic mean/sample SD。

完整矩阵汇总复用 validation P4 的：

- 条件效应、A/B/C 等权主效应；
- AB/AC/BC 与 ABC 交互；
- Local RR median、P90/P95、`>2/>5 bpm`；
- 完整受试者分层、8-subject 等权 macro 和有效分母；
- validation-test 描述性对照。

三个 seed 表示训练随机性，重叠窗口不作为独立样本；不计算 window-level 或 seed-level p-value。Local RR 仍是窗口级频谱主峰误差，不等同逐呼吸相位跟踪。Test 汇总不重新选择结构或生成跨属性总分。

## 5. 生命周期与输出

阶段划分：

| 阶段 | 内容 | 当前状态 |
|---|---|---|
| T0 | 协议、入口与 synthetic CPU 定向测试 | 已完成（57 项相关回归通过） |
| T1 | 生成固定 24-checkpoint allowlist receipt | 等待代码提交 |
| T2 | 24 次 GPU research-test evaluation | 等待单独授权 |
| T3 | 一次性完整矩阵汇总 | T2 完整后开放 |

输出根：

```text
runs/w0_structural_factorial_v1_es30p15/research_test/
  allowlist/<attempt>/
  evaluation/<arm>/seed_<seed>/<attempt>/
  summary/<attempt>/
```

每个 attempt 不可覆盖；失败 lifecycle 保留。相同 allowlist/arm/seed 的成功评价拒绝重复。T3 要求 24 个 cell 各有唯一成功 evaluation，并重新核验全部文件身份、2,310 行顺序、target eligibility、finite、分母和 checkpoint 来源。

## 6. 执行入口

实现入口为：

```bash
# 提交实现后生成 allowlist；本步不读取 test 数组。
./.venv/bin/python scripts/run_w0_structural_factorial_v1_test.py prepare-allowlist

# 实际 test 访问须再次获得用户授权并显式确认。
./.venv/bin/python scripts/run_w0_structural_factorial_v1_test.py evaluate \
  --allowlist <allowlist-attempt> --arm <arm> --seed <seed> --device cuda:0 \
  --confirm-research-test

# 24 项完整后执行一次性汇总。
./.venv/bin/python scripts/run_w0_structural_factorial_v1_test.py summary \
  --allowlist <allowlist-attempt> --confirm-research-test
```

Codex 本轮实现与验证仅使用 synthetic/disposable fixture，不执行以上真实 research-test 评价。
