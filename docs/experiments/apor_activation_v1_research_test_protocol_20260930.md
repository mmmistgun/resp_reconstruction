# APOR激活规范化：固定checkpoint research-test附件

日期：2026-09-30。协议ID：`apor-activation-v1-research-test-20260930`。

用户当次要求继续后续实验，包括test。本附件定义U1三seed训练/validation完成后的固定评价，U0复用原A0三份test。执行顺序为：完整validation汇总与质量决定冻结→prepare-allowlist绑定三个U1 checkpoint→全部test评价→完整六单元三视图汇总。训练或汇总失败则停止，不进入test。

## 1. 固定科学范围

- 训练依据[激活规范化协议](apor_activation_v1_protocol_20260930.md)，结构已由v2 validation冻结为N0。
- U0为原A0，来源由v2 `validation_decision.json`绑定；复用`runs/patch_apor_v1/research_test/allowlist_20260930T071732Z_9b85989bdc0b`中A0三个seed的成功评价，核对checkpoint、epoch、样本与文件身份。
- U1只有四处GELU→SiLU变化，seeds=20260811/20260812/20260813；checkpoint均按完整validation Local RR最小、并列最早选择。
- 评价全部三个U1 checkpoint，不根据validation保护线是否通过或partial test结果省略单元。Validation候选决定保持独立，test不改epoch、阈值、结构或激活。

## 2. 样本和指标

完整test共2310窗口、8个samp_id，180 s/100 Hz、sample seed20260612，有序row ID SHA256=`184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8`。原数据索引、input-only W缓存、资格、Pi与原生test指标均保持不变。

固定输出full（2310窗口/8人）、exclude670（2231窗口/7人）、subject670（79窗口/1人）三个视图。所有窗口先完整推理并核验，再派生子集统计。两种多受试者视图同时报告窗口均值与subject-macro；670为单病例描述，病例背景来自用户提供的信息。

推理为strict checkpoint加载、eval、batch128、BF16和完整末尾batch。检查真实input/target/W、模型与预测finite，样本顺序与development受试者隔离。保存原五主指标及原生IBI/coverage、coherence、constrained nDTW、分层包络Spearman、资格与失败分母；不通过过滤预测失败窗口改善结果。

三seed均值、sample SD、U1−U0配对差和方向数、逐受试者、subject-macro及RR尾部全部保存。新增6930行窗口指标，完整科学矩阵13860行。本test是reused research/development evidence，不作为新的独立人群确认。

## 3. 运行身份与命令

入口：`scripts/eval_apor_activation_v1_research_test.py`。训练完成后prepare-allowlist验证全部来源、history、checkpoint config/epoch和validation决定，固定新checkpoint哈希并保存源码快照；此时不读取test数组。输出使用`runs/apor_activation_v1/research_test/allowlist_<UTC>_<uuid>`。

```bash
./.venv/bin/python scripts/eval_apor_activation_v1_research_test.py prepare-allowlist --session /实际激活训练session
```

双GPU固定2/1分片、每worker CPU线程4；已完成结果校验后复用，失败/中断保留，显式重试才创建新attempt。整个六单元矩阵齐备后统一汇总，不按partial结果调整。

```bash
APOR_ACT_TEST='/实际allowlist路径'
env -u LD_LIBRARY_PATH -u LD_PRELOAD PYTHONUNBUFFERED=1 \
  ./.venv/bin/python scripts/eval_apor_activation_v1_research_test.py parallel \
  --allowlist "$APOR_ACT_TEST" --devices cuda:0 cuda:1 --confirm-research-test
```

训练入口的`pipeline --confirm-training --confirm-research-test`按同样顺序衔接两阶段；任一阶段失败终止并保留现场。源码、配置、protocol、checkpoint、原始指标及历史结果不覆盖，完成状态以各阶段manifest/receipt和pipeline_result记录。
