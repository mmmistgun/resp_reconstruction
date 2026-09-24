# E7：聚合前尺度编码 × 尺度聚合析因实验

日期：2026-09-24。协议 ID：`e7-scale-encoding-aggregation-factorial-v1-20260924`。

状态：**P1 独立六臂模型、严格 spec、训练/汇总控制器、表示诊断与 synthetic CPU 验收已完成；26 项定向测试通过。来源审计、实现锁、GPU 工程验收、正式训练和评价尚未执行。当前阶段不读取独立测试集。**

## 1. 科学问题与证据边界

W0 的 CWT 条件分支在 97 个尺度槽位上依次使用 `k=(5,3)` 普通卷积和 `k=(3,3)` depthwise 卷积，随后立即沿尺度轴求均值。两个尺度卷积对应约 7 槽的有效尺度感受野。E4 已比较多种尺度聚合器，并形成聚合器局部功能与结构取舍证据；它没有把聚合前尺度编码深度或尺度跨度作为实验因子。

E7 专门回答：

> 在 W0 的数据、CWT、主干、损失和选择器固定时，聚合前表示是否能从更深的局部尺度编码或跨轴尺度编码中获得稳定收益，以及这种收益是否依赖聚合方式？

核心证据来自同轮 `3 encoder × 2 aggregation × 3 seeds` 的 18 次从头训练。历史 W0/E4 checkpoint 只作来源和机制背景，不进入本轮单元格。结论限于本协议定义的卷积尺度编码器、训练预算和 validation/research-test 人群。

## 2. 析因矩阵

### 2.1 聚合前尺度编码因子

原 W0 聚合输入记为 `X0 ∈ R^(B×96×97×360)`。

| Encoder ID | 定义 | 相对原始 W 的理论尺度感受野 | 主要问题 |
|---|---|---:|---|
| `s0_shallow` | 原 W0 编码，直接输出 `X0` | 7 | 同轮基线 |
| `s1_deep_local` | `X0` 后串联四个 `k=(9,1), dilation=(1,1)` 尺度残差块 | 39 | 增加深度、非线性和容量是否有用 |
| `s2_axis_spanning` | `X0` 后串联四个 dilation 为 `1/2/4/8` 的同构尺度残差块 | 127 | 更宽尺度跨度是否在相同深度和参数量下额外有用 |

S1 与 S2 的层数、通道、可学习参数量、激活、归一化和投影完全相同，只有 dilation 与对应 padding 不同。127 是内部位置的理论最大跨度；same padding 下边缘位置的真实输入覆盖会被边界截断。S2 的可解释范围是跨轴尺度跨度，逐槽位覆盖范围仍由其位置和边界共同决定。

每个残差块固定为：

```text
X [B,96,97,360]
→ channel-only LayerNorm（每个 b,s,t 独立跨 96 channels）
→ SiLU
→ DWConv2d(96, k=(9,1), dilation=(d,1), padding=(4d,0), groups=96)
→ Conv2d(96→96, k=1)
→ residual add
```

卷积权重使用项目原生 `kaiming_normal_(fan_in, relu)` 初始化，卷积 bias 使用原生零偏置，LayerNorm affine 使用 weight=1、bias=0。S1/S2 使用相同命名子 seed，使同 seed 下形状相同的可学习 tensor 初值逐 tensor 相等。新增模块在独立 CPU RNG 上下文中构造，公共 W0 state 和训练 CUDA RNG 保持同 seed 可配对。

### 2.2 聚合因子

| Aggregation ID | 定义 |
|---|---|
| `mean` | `X.mean(dim=2)` |
| `frequency_attention` | E4 frequency-aware content attention 的内容/真实频率坐标/稳定归一化结构 |

Frequency attention 固定为：

```text
h[b,:,s,t] = SiLU(W_content @ X[b,:,s,t] + b_content + v_frequency*q_s)
logit[b,s,t] = W_score @ h[b,:,s,t]
weight = softmax(logit, dim=scale)  # FP32 稳定计算
z[b,c,t] = Σ_s weight[b,s,t] * X[b,c,s,t]
```

其中 hidden channels=8，`q_s` 使用冻结真实频率的 log-minmax 坐标，输出权重跨 96 channels 共享。所有可学习卷积使用上述项目原生初始化和独立命名子 seed。聚合输出、score、权重及 correction 必须有限。

### 2.3 完整单元格

```text
s0_shallow × mean
s0_shallow × frequency_attention
s1_deep_local × mean
s1_deep_local × frequency_attention
s2_axis_spanning × mean
s2_axis_spanning × frequency_attention
× seeds 20260811 / 20260812 / 20260813
```

六组在同 seed 下共享 W0 公共参数初始化、数据顺序和 sample seed。S1/S2 编码器初值配对，三个 frequency-attention arm 的聚合器初值配对。各 arm 的初始函数允许由其已定义结构自然产生；正式比较估计完整结构及其优化参数化的整体效果。

## 3. 冻结变量

以下合同在 18 个单元格中统一：

1. `research_v2` 的 input、target、admission、train/validation split、subject/session 隔离及逐 row identity。
2. train/validation W cache、97 个尺度槽位、冻结真实频率数组与排列。
3. 原 W0 waveform frontend、六层 BiMamba2、W 条件分支 temporal mixer、active fill、FiLM 位置与系数、refinement 和 decoder。
4. TM3、REF2 及全部既有 shape、finite、eligibility 与退化标志语义。
5. `L_sync + 0.25 L_effort`，AdamW、weight decay、gradient clipping、warm-up cosine 与 planned-update 学习率定义。
6. seeds `20260811/20260812/20260813`；model initialization seed 与 training seed 相同；train/validation sample seed 沿用 W0。
7. physical/effective batch 128、accumulation=1、BF16、完整尾 batch、branch checkpoint chunk=8、drop_last=false、resume=false。
8. 五主指标、逐窗口直接平均、subject-macro、eligibility 分母和 validation Local RR checkpoint selector。

结构新增参数不触发 active fill 重新匹配。参数量、覆盖算子 MAC、实测显存与速度单独报告。

## 4. 训练与 early stopping

- 最大 80 epochs，每个完整 epoch 为 80 optimizer updates，planned 上限为 6,400 updates。
- early stopping 监控每 epoch 完整 validation 的 Local RR MAE：`min_epoch=30 / patience=15 / min_delta=0`。
- 只有严格降低 Local RR 才更新 best 并清零 wait；相等值计入 wait。wait 从 epoch 1 开始累计，达到 epoch 30 后允许停止。
- 学习率始终按 planned 6,400 updates 计算，不随实际停止点重标定。
- checkpoint selector 是已执行完整 validation 轨迹中的 Local RR 严格最小；相等时保留最早 epoch。
- 每个 run 保存 best checkpoint、实际末轮 checkpoint、停止原因、实际 epochs/updates、wait 轨迹和固定 planned budget。
- 汇总器逐 epoch 重放停止状态，要求 run 恰好结束在首个合法停止点或 epoch 80。

18 个单元格必须全部完成后一次性冻结 validation 汇总。不得依据 partial seed、单个单元格、中间指标或运行耗时改变矩阵、初始化、停止合同、容差或选择器。

## 5. 指标方向、容差与 Pareto 规则

五主指标为：

- `whole_rr_abs_error_bpm`
- `local_rr_mae_bpm`
- `envelope_trajectory_mae`
- `global_envelope_modulation_error`
- `lag_aware_signed_pcc`

统计表保留原始量纲。用于方向与交互解释时定义效用值 `Y`：四项 error 取负值，PCC 保持原值，因此正差统一表示改善。

材料性容差预注册为：

- 四项 error：相对变化 `0.5%`；基线为零时只报告原始差并标记相对量未定义。
- PCC：绝对变化 `0.002`。

每个五指标对比按每 seed 配对差、三 seed mean/sample SD、改善/容差内/恶化方向数、逐 `samp_id` 配对差和 subject-macro 联合解释。一个结构只有在至少一个属性形成材料性改善、其余属性均无材料性退化时才构成 tolerance-aware Pareto 改善；存在任一材料性退化时报告属性权衡。三个训练 seed 描述优化随机性，不作为三个独立人群，也不与七个 validation `samp_id` 相乘构造独立样本量。不构造跨指标总分。

## 6. 计划对比

令 `Sij` 表示 encoder `i∈{0,1,2}` 与 aggregation `j∈{M,A}` 的效用值。

### 6.1 条件简单效应

```text
MEAN 下局部加深：      S1M - S0M
MEAN 下宽尺度跨度：    S2M - S1M
ATTN 下局部加深：      S1A - S0A
ATTN 下宽尺度跨度：    S2A - S1A

S0 聚合效应：          S0A - S0M
S1 聚合效应：          S1A - S1M
S2 聚合效应：          S2A - S2M
```

### 6.2 边际 encoder 对比与交互

```text
局部加深边际效应：
C_local = 0.5 * [(S1M-S0M) + (S1A-S0A)]

宽尺度跨度边际效应：
C_span = 0.5 * [(S2M-S1M) + (S2A-S1A)]

局部加深 × 聚合交互：
I_local = (S1A-S1M) - (S0A-S0M)

宽尺度跨度 × 聚合交互：
I_span = (S2A-S2M) - (S1A-S1M)

端点交互（描述性）：
I_endpoint = (S2A-S2M) - (S0A-S0M) = I_local + I_span
```

`I_local` 与 `I_span` 是两项线性独立交互；端点交互只作便于阅读的合成描述，不计为第三项独立证据。

### 6.3 解释模板

- `C_local` 稳定改善而 `C_span` 处于容差内：支持增加聚合前深度/容量，未显示额外跨度收益。
- `C_span` 稳定改善：支持在相同深度与参数量下扩大尺度跨度。
- `S0` 的聚合效应更强且 `I_local/I_span` 为负：attention 的收益随编码增强而减弱，符合其补偿浅尺度编码的解释。
- `S2` 的聚合效应更强且相关交互为正：attention 更能利用尺度上下文化表示。
- S1/S2 在两种聚合下均未形成材料性 Pareto 改善，且三 seed 与 subject-macro 没有稳定改善方向：在本结构族、预算和容差内，为浅层编码基本充分提供有限支持。
- 任一改善伴随 Local RR、PCC 或其他属性的材料性退化：结论为属性权衡。

交互的正负表示聚合效应之差，只有结合相应简单效应才能解释为总体收益。

## 7. 表征与优化诊断

诊断使用每个 validation-selected checkpoint 的完整 validation split，并同时保留窗口直接平均与 subject-macro。所有累计使用 FP64 或具有等价稳定性的流式算法，避免保存完整四维激活。

### 7.1 聚合前表征

对 `X0` 和最终聚合输入 `XE` 分别计算：

1. 跨尺度方差：先对每个 `(window,channel,time)` 向量计算 97 槽方差，再按窗口与 subject 汇总。
2. 尺度相关矩阵：将 `(window,channel,time)` 作为观测、97 个尺度槽位作为变量，流式累计均值、方差和 `97×97` 相关矩阵。
3. entropy effective rank：对相关矩阵非负特征值归一化为 `p_i`，计算 `exp(-Σ p_i log p_i)`；同时保存特征值与数值秩容差。
4. 固定尺度距离相关：分别汇总 slot lag `1`、`8–16`、`48–96` 的相关系数；另保存按真实 log-frequency 距离分箱的结果。

### 7.2 模块幅度与优化通路

- 每层 residual ratio：`RMS(block_output-block_input) / RMS(block_input)`。
- 四层联合 residual ratio：`RMS(XE-X0) / RMS(X0)`。
- attention correction ratio：`RMS(attention_output-mean(XE)) / RMS(mean(XE))`。
- attention 权重的 total variation、entropy、四个既有 E4 区域权重质量与 subject-macro。
- 每个新参数 tensor 在首个 backward、第一次 optimizer step 后、第五次 step 后和 selected checkpoint 的 gradient norm、parameter norm、相对更新 norm。
- 公共参数与新增参数的 optimizer 分组、step、有限性和实际更新状态。

W 条件分支的最终 FiLM projection 延续原 W0 初始化，因此验收必须覆盖多步通路开启；单步梯度只描述该步计算图状态。

### 7.3 checkpoint 内干预

在 selected checkpoint 上执行适用条件的完整 validation 配对评价：

- encoder residual bypass；
- attention 恢复 uniform aggregation；
- 同时启用两项干预。

S0/MEAN 的对应模块状态作为恒等参照。输出逐窗口、逐 seed、逐 subject 和五主指标差值。该分析只用于判断训练后模块是否被模型利用，不替代六组从头训练的结构比较。

## 8. 工程验收与资源报告

正式训练前完成以下阶段：

1. Synthetic CPU：shape、padding、dilation、理论感受野、标准初始化、S1/S2 参数与初值配对、frequency coordinate、attention 归一化、finite、state round-trip、optimizer 分组、planned contrasts 和 early-stop 回放。
2. Synthetic GPU batch 1：三 seed 六组完整 forward/backward/multi-step update，核对公共 state、梯度通路与所有新增参数实际更新。
3. Synthetic GPU batch 128、chunk 8：六组至少五次原生 loss/update，记录峰值 allocated/reserved、单步时间、finite 与 OOM 状态；显存上限固定为 device total 的 80%。
4. 独立进程 benchmark：六组 batch-1 eval 与 batch-128 train，固定 warm-up、计时次数和轮换顺序，报告 median/IQR、吞吐和显存。

解析计算报告至少覆盖四个残差块的 DWConv/pointwise Conv 与 attention scorer，并明确未覆盖的算子；不把局部 MAC 当作完整模型 FLOPs。工程验收不得自动改变 batch、chunk、精度、层数或 dilation 后沿用原实现身份。

## 9. 产物与汇总

每个 formal run 至少保存：resolved config、命令、代码与实现锁、环境、初始化/训练/sample seeds、数据与 cache identity、完整 history、best/final checkpoint、optimizer 分组与 state、逐窗口 validation metrics、五主指标 summary、early-stop 轨迹、诊断流式统计和 lifecycle receipt。

完整 validation 汇总至少生成：

- `per_seed.csv`：18 行，含 arm、seed、selected/final epoch、停止原因和实际 updates。
- `across_seed.csv`：6 arms × 5 metrics。
- `simple_effects_per_seed.csv` 与 `simple_effects_across_seed.csv`。
- `factorial_contrasts_per_seed.csv` 与 `factorial_contrasts_across_seed.csv`。
- `subject_macro.csv`：逐 arm、seed、metric、`samp_id` 及等权宏平均。
- `representation_diagnostics.csv`、`optimization_diagnostics.csv`。
- `checkpoint_interventions.csv`。
- `manifest.json`：全部来源、文件 SHA、行数、分母和冻结身份。

汇总器只接受同一实现锁下 18 个唯一完成 attempt；缺失、重复、配置漂移、停止点漂移、非有限关键量、row identity 或 eligibility 不一致均显式失败。

## 10. Test 门控

当前协议范围为 train/validation。18 个 validation run、完整汇总、计划对比和诊断全部冻结后，另建 research-test 专项协议与 checkpoint allowlist。该附件必须固定六组×三 seed 的全部 18 个 validation-selected checkpoint，并在任何 test 数组读取前完成来源、split 隔离和访问审计。

Research-test 阶段完整评价 18 个 checkpoint，沿用五主指标、subject-macro、配对方向和材料性容差；test 不参与 early stopping、checkpoint 选择、结构筛选或阈值修改。既有 E4 research-test 已参与设计背景，因此 E7 test 结果按 research-test-informed development 证据解释。

## 11. 阶段计划

P1 实现入口：

```text
configs/e7_scale_encoding_aggregation/experiment.yaml
resp_train/paper_evidence/e7_scale_encoding_aggregation_model.py
resp_train/paper_evidence/e7_scale_encoding_aggregation.py
scripts/run_e7_scale_encoding_aggregation.py
tests/test_e7_scale_encoding_aggregation.py
```

当前允许的只读/CPU 验收命令：

```bash
./.venv/bin/python scripts/run_e7_scale_encoding_aggregation.py check-p1
./.venv/bin/python -m pytest tests/test_e7_scale_encoding_aggregation.py -q
```

| 阶段 | 内容 | 当前状态 |
|---|---|---|
| P0 | 科学问题、矩阵、停止合同、分析与诊断 | 已完成 |
| P1 | 独立模型、训练/汇总控制器、synthetic CPU 测试 | 已完成；26 tests passed |
| P2 | 来源审计与不可覆盖实现锁 | 待 P1 提交后执行 |
| P3 | Synthetic GPU acceptance 与 benchmark | 由用户执行 |
| P4 | 18 次 formal train/validation | 由用户执行 |
| P5 | 完整 validation 汇总与冻结 | 随完整 P4 开放 |
| P6 | Research-test 专项协议、allowlist 与 18 项评价 | 待 P5 冻结及当次授权 |

历史 E4 代码、协议、锁、checkpoint、评价和结项产物保持原身份。E7 使用独立配置、源码入口、输出根、实现锁和 lifecycle。
