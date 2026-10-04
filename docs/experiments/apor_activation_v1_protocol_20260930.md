# APOR：固定N0结构的激活规范化协议

日期：2026-09-30。协议ID：`apor-activation-v1-20260930`。

状态：用户明确要求继续后续实验。本阶段实施已在[APOR v2研究方案](apor_grid_decoder_v2_research_plan_20260930.md)预设的第二阶段，复用U0三seed、新训练U1三seed；v2 test运行结束后启动两卡验收与训练。结果与阶段完成状态以独立session和receipt为准。

## 1. 结构选择与参照

v2完整validation已选择N0（原A0完整APOR），该决定在v2新test前冻结：

- `runs/apor_grid_decoder_v2/session_20260930T093614Z_511f2ff06f0f/summary/attempt_20260930T111411Z_d03d6929f568/validation_decision.json`。
- SHA256=`4e69b062349e4139746cc83c7b7aff4f2e305b27afbdf10b2cdc9fd3a4810a8c`。
- U0固定为原PATCH/APOR session的A0，selected epoch=8/13/13，seeds=20260811/20260812/20260813。参照checkpoint必须与上述decision中的三项身份完全一致。

v2 test结果不改变本阶段结构、候选矩阵、激活位置、阈值或训练合同。

## 2. 单因素比较

| Arm | PatchMixer隐藏层激活 | 其他计算 |
|---|---|---|
| U0 | 原GELU | 原A0/N0，复用冻结三seed |
| U1 | SiLU | 从共同初始化训练三seed |

精确替换两个PatchMixerBlock中的`patch_mixer[2]`与`channel_mixer[1]`，共四个GELU。模块权重、参数名称、初始tensor、参数量均不变，两臂均1077640参数。条件编码、adapter、H65、片段MLP中原有SiLU不改；FiLM的tanh、线性波形输出、Mamba内部算子、GN/RMSNorm及所有分组/epsilon保持原值。

U1从头训练，不使用已训练U0权重继续优化。该比较检验工程规范化能否保持质量，不将激活替换预设为独立方法创新。

## 3. 固定训练与质量规则

沿用v2/v1训练合同：train/validation=10141/2675窗口、32/7人，180 s/100 Hz，sample seed20260610/20260611；训练seed同上；原Pi、`L_sync+0.25L_effort`、五主指标、eligibility、AdamW分组、batch128、BF16、最大80epochs/6400updates、min30/patience15。Checkpoint由完整validation Local RR严格最小、并列最早选择。

保存完整三seed窗口平均与subject-macro结果、sample SD、配对差及方向、逐受试者、尾部和分母。U1在两种聚合口径内，四项error均值相对U0退化均不超过0.5%、PCC下降不超过0.002时，成为统一激活候选；否则保留U0。这是预设工程保护线，不作为统计非劣效或临床阈值。全部seed完成后统一判定，不按中间结果裁剪。

后续固定checkpoint test沿用已授权的研究开发评价范围，读取前另行绑定U1的三个既定checkpoint并复用U0冻结test。固定报告完整2310窗口、排除670的2231窗口和670单病例79窗口。Test不回选checkpoint，不改变上述validation决定；属于复用开发性证据。

## 4. 执行与来源

独立入口`scripts/run_apor_activation_v1.py`；模型/运行器分别为`scripts/apor_activation_v1_model.py`、`scripts/apor_activation_v1_runtime.py`，配置为`configs/apor_activation_v1/experiment.yaml`，输出根`runs/apor_activation_v1/`。历史v1、v2源码与产物保持冻结。

准备阶段核对v2结构决定和U0三份来源、配置、history、checkpoint、metrics及环境；完整绑定共同初始tensor。Synthetic CPU覆盖替换位置和数量、U0等价、U1梯度、保存回载、原生训练器及质量规则。原生GPU验收覆盖两臂×三seed batch1三步更新，以及两臂batch128训练/eval；U0与原A0的严格FP32/BF16前后向等价沿用隔离确定性验收，正式训练保持历史算法设置。

双GPU验收均通过后，固定2/1分片训练三个U1 seed。CPU线程4，完整训练成功后汇总U0/U1六个单元并固定validation候选及checkpoint。非有限状态显式失败；成功、失败与中断产物均保留，禁止同identity覆盖或隐式重跑。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. ./.venv/bin/python -m pytest tests/test_apor_activation_v1.py -q
./.venv/bin/python scripts/run_apor_activation_v1.py prepare
```

使用返回session路径：

```bash
APOR_ACT_SESSION='/实际session路径'
env -u LD_LIBRARY_PATH -u LD_PRELOAD PYTHONUNBUFFERED=1 \
  ./.venv/bin/python scripts/run_apor_activation_v1.py parallel \
  --session "$APOR_ACT_SESSION" --devices cuda:0 cuda:1 --confirm-training
```

本阶段新增训练3次，上限19200updates。效率从formal运行记录提取，独立benchmark不随训练自动启动。所有已完成阶段保持冻结，后续完成记录另建文件。

实现验证：9项模型/训练synthetic CPU检查通过（19.69 s）；pipeline顺序与失败门禁2项、独立test入口13项检查通过（合计35.60 s）。覆盖恰好四处激活替换、公共初始化与U0重放、梯度、原生训练器、两种聚合口径保护线、三视图分母、原生test指标及并行完成门禁。

当次已授权的train/validation→固定test可使用`pipeline`串行衔接；完整validation决定先于test allowlist冻结，三个U1 checkpoint全部评价，执行合同见[test附件](apor_activation_v1_research_test_protocol_20260930.md)：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD PYTHONUNBUFFERED=1 \
  ./.venv/bin/python scripts/run_apor_activation_v1.py pipeline \
  --session "$APOR_ACT_SESSION" --devices cuda:0 cuda:1 \
  --confirm-training --confirm-research-test
```
