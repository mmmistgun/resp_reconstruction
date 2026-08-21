# Respiration Temporal Modeling v1 协议

日期：2026-08-20

状态：**Signal-first、candidate/CPU/GPU locks、双GPU formal 15/15、validation summary与独立测试集评价均已完成并通过完整性验收。测试结果结论尚待用户确认锁定。**

## 1. 权威性、独立命名与当前边界

本文是 `docs/experiments/loss_metrics_restart_plan_20260729.md` 第 50 节引用的规范性附件；冲突时以主协议为准。协议 ID 固定为：

```text
resp-temporal-v1-validation-20260820
```

简称 `RTM-v1`。它是新的 validation-only 时序模型家族比较，不属于 CRD-v1.1、CRD-TF v1、CRD-TF-W v2 或旧 THO 阶段的补跑。旧阶段的 checkpoint、summary、配置和冻结结论只作只读背景，不进入本协议的候选集合，也不得覆盖或重跑。

Validation用于checkpoint选择和本阶段模型比较，因此validation结果的证据名称固定为：

```text
validation-development evidence
```

RTM-v1 formal只允许读取train/validation，并由用户通过唯一单run入口手动执行；入口不提供split、data、checkpoint、epoch、batch、device、output或resume override。该validation阶段没有独立评价入口或test split。用户确认validation lock后另行明确授权的独立测试集评价属于后续独立协议，规范性执行附件保留为`docs/experiments/resp_temporal_v1_research_test_protocol_20260821.md`。本文后续统一将其称为“独立测试集”：独立性指与train/validation严格隔离，且不参与本阶段candidate、checkpoint或超参数选择；不要求测试集在整个研究过程中只能访问一次。

## 2. 现有时序证据的正确角色

### 2.1 PatchMixer B0

现有 B0 为 `patch_len=256 / stride=128 / channels=16 / blocks=2`，共 11,408 参数。每个 block 的 learnable token mixing 只是 `kernel=3` depthwise convolution；两层可学习内容传播约覆盖 5 个 token，即约 7.68 秒。GroupNorm 会引入全窗均值/方差依赖，但不等于可学习的远距离内容交换。

B0 是稳定的协议 baseline，不是合理容量 PatchMixer 家族探针。

### 2.2 M1 参数匹配多尺度控制

M1 使用 256/512/1024/2048 点四个 patch 分支，每分支 `base_channels=1`、两个 mixer block，最终只用四个 softmax scalar 做 waveform 融合，总参数 11,664。所有 mixer block 与融合 scalar 合计约 140 参数，仅占约 1.2%；绝大多数参数位于每分支 patch embedding/head。

因此 M1 只回答严格参数预算下的多尺度容量重分配，不回答合理宽度、多尺度 feature interaction 或层级时序建模是否有效。其 validation 负/混合结果不得写成“多尺度时序网络无效”。

### 2.3 CRD_102 Mamba 与 C1 TCN

CRD_102 为 6×D96 BiMamba2，C1 为 10×`C=96/H=488` full-context TCN。C1 感受野 4093 个 10-Hz tokens、总参数 1,062,001，相对 CRD_102 少 0.6310%；它本身是合理容量的一个 TCN 点，但 H=488 来自严格参数匹配，且没有第二个合理深度/宽度点。

C1 改善 Whole/Local RR，但 trajectory、global envelope、PCC 和 IBI 形成任务交换，并因预注册 trajectory guardrail 失败而保留 Mamba。该结论属于单一 backbone replacement control，不证明 Mamba 家族普遍优于 TCN。

两者共同使用旧 PatchTokenFrontend：先生成 140 个 PatchMixer token，再线性插值为 1800；插值不产生新的时间信息。因此它也不是对所有合理 raw-signal temporal frontend 的覆盖。

### 2.4 D4 与 LSTM

D4 只在 W0 full-12V FiLM 条件下把 Local BiMamba2 从 6 层降到 4 层，属于特定表示下的深度—效率控制，不是纯时域 Mamba family sweep。

仓库在 RTM-v1 前没有 LSTM 实现、配置或结果。离线双向 LSTM 是与卷积、全局 mixer 和 SSM 不同的门控递归归纳偏置，应作为合理时序家族加入，但不能直接处理不必要的 18,000-step 原始序列。

## 3. 科学问题与不可回答的问题

唯一主问题：

> 在只使用 train 的 BCG—呼吸信号审计所固定的可观测机制、采样率与时间尺度下，各时序家族的一个信号充分、家族标准代表在共同输入输出契约中形成怎样的 validation 质量 Pareto；其质量收益分别对应哪些局部周期、速率变化、努力趋势或整窗上下文假设？

辅助问题：

1. 任一 temporal trunk 是否相对无 trunk 的公共 stem/head 控制产生净任务收益；
2. dilated convolution、state-space recurrence、gated recurrence、feature pyramid，以及条件进入的 global token mixing 分别表现出哪些任务交换；
3. 共同 stem 是否在降至 10 Hz 前保留了 train 数据中可观测的低频位移与 carrier-modulation 信息；
4. 参数、吞吐、延迟与显存作为描述性工程结果，是否改变候选的可部署性；它们不参与科学候选选择。

本协议不回答：

- 各家族经过 family-specific frontend、optimizer 或大规模调参后的理论能力上界；
- causal/streaming 能力；
- Mamba、TCN、LSTM 或 PatchMixer 的普遍优劣；
- subject-level 确认性统计推断；
- 外部泛化或未触碰 test 表现；
- 高频 BCG 表征、时频融合、decoder placement 或 loss 的新问题。

本协议中的角色必须严格分开：

- T0 是 temporal-trunk attribution control；
- 不同家族的锁定代表是架构能力比较，不是参数匹配控制；
- 同一家族的第二个深度或宽度若未来开放，只能称为 capacity-sensitivity control，不能替代该家族代表或伪装成新的架构能力证据；
- 工程不可运行只说明当前硬件下未评估，不说明该家族无效或容量不足。

## 4. 公共数据、任务与训练口径

以下项目冻结不变：

- research v2 数据、admission、train/validation split 与 subject/session 隔离；
- 输入 `bcg_rawish_segment_soft_z_key`，shape `[B,1,18000]`、100 Hz、180秒；
- target `target_waveform_segment_soft_z_key`；
- 正式输出 `Pi=S(B(.))`，频带 `0.05–0.70 Hz`；
- `L_sync + 0.25 L_effort` 与全部 eligibility/finite 语义；
- 五项 primary、IBI/coverage 与 train-frozen envelope strata；
- 完整 validation Local-RR 最小 epoch selector，严格 `<` 时更新；
- sample direct mean 与三个固定 training seeds `20260811/12/13`；
- 不构造总分，不计算确认性 p-value。

未来 formal 训练语义预冻结为：

```text
epochs = 80
optimizer updates = 6400
optimizer = AdamW
max_lr / min_lr = 3e-4 / 3e-5
warmup = 5% updates
schedule = exact cosine
weight_decay = 1e-4
grad_clip_norm = 1.0
AMP = bf16
effective batch = 128
early stopping = false
resume = false
```

当前 implementation-only 配置记录这些未来语义，但 `formal_training_enabled=false`；它们不能作为正式训练入口。

## 5. Signal-first 公共 substrate：先审计、后锁定

### 5.1 10-Hz latent 的依据

正式频带的最快 0.70 Hz 周期在 10 Hz 上仍约有 14.3 点，最慢 0.05 Hz 周期约有 200 点；10秒 effort、60秒 Local RR 和180秒整窗分别对应 100/600/1800 tokens。既有 C0 已证明 canonical target 经 10-Hz 抽取与 Fourier 100-Hz 恢复后在正式指标上近似无损；RTM-v1 复用该**输出表示**结论，不重跑已关闭的 C0。

但 C0 不证明 100-Hz BCG 在进入 temporal trunk 前可直接压到 10 Hz，也不证明高频 carrier 已被充分解调。主协议已有依据认为 BCG 的 0.05–8 Hz 可能同时包含呼吸位移、心动及呼吸调制信息。因此公共 stem、carrier 解调顺序和多尺度降采样必须先通过独立的 train-only signal audit；规范见 `docs/experiments/resp_temporal_v1_signal_audit_20260820.md`。不得用 validation 质量选择 stem 或频带。

审计前 implementation probe 的公共 stem 为：

```text
[B,1,18000] at 100 Hz
→ Conv1d 1→48, k=101, stride=5, padding=50, bias=false
→ channel-only LayerNorm → SiLU
→ DWConv1d 48, k=5, padding=2, bias=false
→ channel-only LayerNorm → SiLU
→ Conv1d 48→96, k=5, stride=2, padding=2, bias=false
→ channel-only LayerNorm → SiLU
→ [B,96,1800] at 10 Hz
```

第一层在 100 Hz 上先提取约 1.01 秒局部宽频信息，再逐级降采样；这避免让 LSTM/Mamba 直接处理 18,000 steps，也避免旧 140-token bridge 对快呼吸时序过粗。但 train-only audit 已确认 learned strided convolution 本身不能充当显式 anti-alias 证明，因此该 probe stem **不得原样升格**。

公共 substrate 的科学顺序现冻结为：显式 anti-aliased `100→20 Hz`，在20 Hz执行所有家族共享的 learned carrier-sensitive filtering + nonlinearity，再显式 anti-aliased `20→10 Hz`，形成 `[B,96,1800]` latent。参考算子固定 Kaiser beta `8.6`、`padtype=line`：100→20 使用255 taps/cutoff 9.0 Hz，20→10 使用127 taps/cutoff 4.5 Hz。Exact torch 实现已固定20-Hz `1→48 k21`、48-channel depthwise `k5` 与 `48→96 k5` 学习路径，两次固定降采样分别位于该路径前后；float64 audit 等价 tolerance 固定为 `atol=rtol=5e-12`。四个固定 decimator 的确定性 max-abs error 为 `4.44e-16 / 4.44e-16 / 1.11e-15 / 6.66e-16`，不增加手工 analytic-envelope 或固定等 RMS proxy 输入分支。

公共 decoder 保留 channel-only norm、`96→64 k5`、64-channel depthwise k5、`64→32→1` coarse head，加与 C201 相同的 1,057-parameter zero-init pointwise nonlinear residual；生成 10-Hz raw waveform 后用冻结 Fourier interpolation 恢复 100 Hz。模型内部不执行 `Pi`。

同 seed 的公共 stem/decoder 使用独立命名子 seed并要求跨所有 family 逐 tensor 相同。共同 substrate 用于提高 trunk 归因清晰度，不代表每个家族的端到端最优 frontend。

### 5.2 离线边界

所有锁定 trunk 均允许完整 180 秒双向上下文：TCN 对称 padding、Mamba 双向扫描、LSTM 双向 recurrence、多尺度分支 reflect-FIR/block-center decimation 与对称 convolution。因此结果只能解释为离线重建能力。

## 6. 家族信号假设与当前 implementation probes

家族进入正式矩阵的依据是其是否对应独立、可证伪的信号建模假设，而不是参数量、显存或与其他模型的规模接近程度。旧 Full/Compact 只用于验证结构可实现、shape 正确和容量范围合理；candidate lock 已按 signal mechanism 与 architecture convention 独立选择代表，未通过资源 gate 二选一。

唯一核心代表现冻结为：T0 locked stem/head control；9-block dilation `1…256`、H=384 TCN；6×D96 BiMamba2；2×H96 BiLSTM；以及 H=384、`10/2/1 Hz` feature-level multiscale。深度/宽度规则分别来自完整感受野、标准4× expansion、成熟六层项目 convention 和最小非平凡 stacked recurrence。同家族第二容量点全部关闭，只能在新协议中称 capacity-sensitivity control。

### 6.1 T0 无 trunk 控制

T0 只包含 locked common stem/decoder。旧 probe 参数62,882仅作 provenance；显式 anti-alias substrate 实现后须重算参数合同。T0 是控制实验，不是第五个架构家族，也不因参数小而进入“公平参数排名”。

### 6.2 Global token mixer（条件家族）

当前 B0 的局部 k3 token mixer不进入新候选。当前 RTM-v1 probe 每个 block 使用：

```text
channel-only norm
→ Linear(1800→token_hidden) → GELU → Linear(token_hidden→1800)
→ residual
→ channel-only norm
→ Conv1x1(96→192) → GELU → Dropout(0.10) → Conv1x1(192→96)
→ residual
```

- Compact：2 blocks、`token_hidden=64`，固定 602,482 参数；
- Full：4 blocks、`token_hidden=96`，固定 1,603,010 参数。

两者都具备完整 180秒 token mixing；Compact 不是短上下文版本。该实现实际是固定 1800 位置上的低秩 global token MLP，不是传统 patch hierarchy，也不具备卷积式时间平移等变性。Train-only audit 支持局部、层级与连续状态机制，但没有形成固定位置、弱局部先验 global mixing 的独立信号依据；用户已确认关闭该家族，不因已有实现而运行。

### 6.3 Dilated TCN

九个 residual blocks 的 dilation 固定为：

```text
1,2,4,8,16,32,64,128,256
```

每 block 为 channel-only norm、depthwise k5、pointwise expansion、SiLU/dropout、zero-init projection。理论感受野：

```text
1 + 4 × sum(dilations) = 2045 tokens = 204.5 s
```

- Compact：`H=256`，固定 512,162 参数；
- Full：`H=384`，固定 733,346 参数。

两个 tier 都覆盖完整 180秒，宽度差异不改变上下文资格。TCN 代表固定9 blocks、H=384；九层由204.5秒完整感受野决定，H=384按D96的标准4× expansion锁定。H=256关闭为 capacity-sensitivity control。

### 6.4 BiMamba2

沿用固定依赖与官方 fast path 的 D=96 双向 Mamba2 block；每方向独立，concat+linear 后残差：

- Compact：4 blocks，固定 697,138 参数；
- Full：6 blocks，固定 1,014,266 参数。

不得使用卷积式 SSM-like fallback。Mamba 代表固定6×D96，沿用成熟项目 convention；4层关闭为 capacity-sensitivity control。依赖或 CUDA fast path 不满足时只能登记当前环境未评估，不得以 fallback 或4层版本自动替代。

### 6.5 BiLSTM

在 `[B,1800,96]` 上先做 channel LayerNorm，再运行 H=96 的 bidirectional LSTM，双向输出经 `192→96` projection、dropout 后与输入残差相加：

- Compact：2 layers，固定 453,314 参数；
- Full：3 layers，固定 676,034 参数。

不开放 raw 18,000-step LSTM、单向 LSTM、GRU、hidden-size 或 projection 搜索。BiLSTM 代表固定2×H96，这是最小非平凡 stacked bidirectional gated recurrence；3层关闭为 capacity-sensitivity control。

### 6.6 多尺度 feature pyramid

Locked multiscale 对共同 10-Hz latent 构造三个 feature 分支：

| 分支 | Grid | 长度 | dilation | 主要作用 |
|---|---:|---:|---|---|
| fine | 10 Hz | 1800 | 1/2/4/8 | 快速局部形态与相位 |
| medium | 2 Hz | 360 | 1/2/4/8 | 呼吸周期与局部变化 |
| coarse | 1 Hz | 180 | 1/2/4/8/16/32 | effort 与整窗 context；不得承担完整波形 |

降采样固定为 signal lock 的显式 low-pass + decimation，关闭 average pooling；编码后回到1800并在 feature channel 上 concat+1×1融合。Coarse 分支增加 dilation32，理论感受野253个1-Hz tokens/253秒，以覆盖完整180秒。所有分支保持96 channels，不生成独立 waveform head：

- 代表固定 branch expansion H=384；旧 H=192 probe 关闭为 capacity-sensitivity control；
- substrate/grid变化后须重算参数量，旧1,059,074只作0.5-Hz average-pooling probe provenance。

它与旧 M1 的关键区别是 feature-level、合理宽度、多尺度交互，而非四个单通道 waveform 分支的标量融合。审计中2-Hz显式低通对 target 的 round-trip NRMSE/Local RR error 为`0.003213/0.000101 bpm`；1-Hz context effort Spearman为`0.994398`，而0.5-Hz为`0.496835`且47.94% target source power位于其Nyquist以上。因此正式 grid 固定10/2/1 Hz，0.5 Hz与average pooling均关闭。

## 7. Signal-substrate lock、candidate lock 与工程可行性

选择顺序固定为：

1. 只用 train 完成信号审计，登记 input mechanism、时间尺度、采样/抗混叠结论；
2. 先按信号假设锁定公共 substrate、家族集合及每家族唯一核心代表；
3. candidate lock 完成后，才在同一干净 commit 上做盲于 validation 质量的统一工程 benchmark；
4. 工程 benchmark 只登记能否执行和实际成本，不得把 Full/Compact、深度、宽度或家族加入/退出作为资源优化问题。

本协议不要求参数相等，也不设置用于科学选择的统一参数上限。目标设备仍记录为 RTX 4070 Ti SUPER 16 GiB；effective batch 固定128，physical batch 只允许从 `128/64/32` 选择，对应 accumulation `1/2/4`。允许为可执行性选择 physical batch，但不得改变模型结构、optimizer updates 或候选身份。

统一工程 receipt 必须至少报告 trainable parameters、batch-1 inference latency、固定 batch forward/forward+backward、peak allocated/reserved、候选 physical-batch update throughput 与预估完整 wall time。此前的 `2M parameters / 85% peak-reserved / 4 h` 只可保留为项目规划参考，不能成为候选选择 gate。

若已锁定代表无法在目标设备上以允许的 physical batch finite 运行：

- 标记 `not_evaluated_on_current_hardware` 并暂停该 arm；
- 不得自动替换为 Compact、减少层数/宽度或引入 fallback；
- 若要研究较小容量，必须另立 capacity-sensitivity control，并经用户再次确认；
- 工程失败不得写成家族能力负证据。

Signal audit receipt、双lock与exact CPU implementation均已冻结。GPU engineering v1 failure receipt/manifest SHA-256=`3ae8e96ebfbcb2ef1eeb75134088d364bea085ed37fbe9e72041d569f50d3ad0 / 96b8c5ddddf62df2d890e3293ceb69804c97ca96dee37702981222a8daeae28c`，原因是cuDNN 9.8/9.20环境冲突；v1不得删除、覆盖或重跑。V2从干净commit `d51659d18c8905a5dc62e9a00ed399df198c89fc`完整执行，receipt/manifest SHA-256=`c8d34d5f1a9f944af58945f74110b0c9ff74e15696c7a7f240465ce4c83de38a / e90df905b2708853a761328995c4fd0c6db1e2564b1bb16eb9328ed709f25812`，5/5 candidates均以128×1通过、OOM=0、numeric nonfinite=0。用户确认的GPU engineering lock为`docs/experiments/resp_temporal_v1_gpu_engineering_lock_20260820.json`，SHA-256=`5632b0e404f943e61c646a8ddcf1761c2391884412915ef1d79aa342f8bc0183`，decision=`all_five_hardware_feasible_at_128x1`；formal training继续关闭。

## 8. 最小实验矩阵

signal-first candidate lock 后的核心正式矩阵为：

| 模型 | configs | seeds | runs |
|---|---:|---|---:|
| T0 control | 1 | 20260811/12/13 | 3 |
| TCN representative | 1 | 同上 | 3 |
| Mamba representative | 1 | 同上 | 3 |
| BiLSTM representative | 1 | 同上 | 3 |
| Multiscale representative | 1 | 同上 | 3 |

核心矩阵固定15个 future formal runs、96,000 optimizer updates。Global token mixer 已关闭，不再保留18-run分支。五项 locked candidate ID 为 `rtm_v1_t0_locked_stem_head / rtm_v1_tcn_d9_h384 / rtm_v1_bimamba2_d96_l6 / rtm_v1_bilstm_h96_l2 / rtm_v1_multiscale_10_2_1_h384`。旧 Full/Compact 均不是 formal arm；工程 benchmark/implementation smoke 不计为科研 run，但成本单独记录。

任一 formal arm 一旦开始，三个 seed 必须全部完成。不得依据第一个 seed 的 validation 数值取消其余 seed；只有 OOM、非有限、checkpoint/lifecycle、identity 或 prediction-collapse 工程失败可以暂停队列。

## 9. 汇总、Pareto 与停止线

正式结果先逐 sample direct mean，再报告三个 seed 的 arithmetic mean ± sample SD及 paired-seed 方向。T0 作为 trunk-attribution control 单独列示。

必须分别生成：

1. 五项 primary 的质量 Pareto；
2. 在不改变质量结论的前提下，另列五项 primary + trainable parameters + standardized throughput + peak allocated 的描述性质量—效率 Pareto；
3. batch-1 latency、peak reserved、IBI/coverage、三层 envelope Spearman 只作补充解释。

Tolerance-aware materiality 固定为：

- error primary：相对0.5%；
- signed PCC：绝对0.002；
- throughput与peak allocated：相对5%。

质量 Pareto 与描述性质量—效率 Pareto 必须分开解释。只有 A 在相应 Pareto 的全部比较维度不差于 tolerance，并至少一项实质更优时，才称 A dominate B。若多个候选互不支配，保留 Pareto set；不得加权、排序求和或强制唯一赢家。资源维度不能反向改写“哪个结构更符合信号建模假设”的科学解释。

停止线：

- 任一已锁定 representative 在当前硬件上不可执行：暂停并登记未评估，不用较小结构替代；
- 全部 trunk family 相对 T0 均没有任何 material primary 改善：阶段停止，不追加容量搜索；
- 出现多个 Pareto 候选：保留集合，不根据 secondary 选唯一赢家；
- 汇总完成后协议关闭，不追加宽度、LR、epoch、dilation、LSTM hidden、Mamba depth或多尺度分支；
- 不访问 research-test，不建立 test evaluator。

冻结validation summary从干净commit `670d3bd36182fb4c4b982427616e8d1bc4fef42d`一次性完成。Access receipt/manifest SHA-256=`6f9f1e873b8910b22241bc0e9f2c510909835b0bbc2a9edc12f0e1788fa59aad / aab22094d6efd11927c952e9f12bcbab24e30cedc9a2a5f282f61056f1f193dc`；15项formal、40125行validation、5项candidate summary、50项paired-seed与40项dominance audit均闭合，numeric finite/null/nonfinite=`3425/120/0`，其中null仅为quality-only dominance表中不适用的资源列。Summarizer未读取checkpoint内容、dataset/index、signal/target或research-test，且未使用GPU。

按预注册candidate-mean tolerance，质量Pareto仅含`rtm_v1_multiscale_10_2_1_h384`，其对T0五项primary分别为Whole/Local/trajectory/global相对改善`4.5777% / 5.9896% / 6.9115% / 32.9483%`、signed PCC绝对增加`0.037931`，五项paired-seed material improvement均为`3/3`。它亦在冻结candidate mean/tolerance下支配TCN、BiMamba2和BiLSTM representative；但该关系不是逐seed一致：相对BiLSTM的Whole/trajectory原始更优仅`1/3 / 2/3`，相对BiMamba2的Whole仅`2/3`，相对TCN的trajectory仅`2/3`。因此只能表述为当前冻结representative package的validation-development quality Pareto结果，不得改写为“唯一最佳模型”或multiscale family普遍优越。

加入参数、standardized throughput与peak allocated后，质量—效率Pareto保留全部五项且无overall winner；T0仍只是trunk-attribution control，BiLSTM只构成有竞争力的描述性质量—效率trade-off，资源维度不改写质量锁。停止线固定为`at_least_one_trunk_material_primary_improvement`，不追加容量或超参数搜索。用户确认的不可变validation lock位于`docs/experiments/resp_temporal_v1_validation_lock_20260821.json`，SHA-256=`989ef0a3a5941ead3e80aba25606878f88d315bd23a1cf5ca4260317f3cffce6`；协议关闭且research-test不开放。

用户随后要求只固化五候选validation `mean ± SD`主指标表，并明确授权开始独立测试集评价。表格固定为`docs/experiments/resp_temporal_v1_primary_metrics_table_20260821.md`，SHA-256=`417fe73b491d459fe649a365fdad590d00c1d0aa992df0e150b144dde4ba8f21`。该后续授权不修改validation lock，也不允许用test重选candidate/checkpoint。

## 10. 证据边界与反方风险

1. 共同 stem/decoder 提高 trunk 归因清晰度，但不代表 family-specific end-to-end 最优 package。
2. 固定 AdamW/LR 可能对不同 family 的优化适配程度不同，因此结果不是理论能力上界。
3. Validation 同时承担 checkpoint 与 family comparison，且已被历史研究多次使用，只能形成 development evidence。
4. 三 seed 描述初始化随机性，不构成确认性推断。
5. 10-Hz latent 对正式输出频带充分，但不保证公共 stem 已最优利用输入 BCG 的全部高频 carrier 信息。
6. 参数不是容量或计算的充分代理，必须与实测吞吐、延迟和显存同时报告。
7. 所有结构均离线非因果，不能外推到 streaming。
8. 单个 representative 的失败只能否定该冻结候选，不能写成整个 family 无效。

禁止表述包括：

- “Mamba 普遍优于 TCN/LSTM”；
- “TCN 不适合本任务”；
- “多尺度时序网络无效”；
- “RTM-v1 找到了唯一最佳模型”；
- “validation 结果证明无偏泛化”；
- “资源更小等价于建模能力更弱”。

## 11. 当前实现锁与开放条件

当前已建立：

- `resp_train/temporal/` 下 locked common substrate、五项 candidate、config 与命名子 seed 实现；
- `configs/resp_temporal_v1/` 下恰好五项 strict implementation-only candidate configs；旧11项 Full/Compact probes只作历史 implementation provenance；
- fixed-filter 对 audit operator 的float64数值等价、可微性、参数/shape/感受野、五项公共 state tensor identity、invalid config 与非有限输入定向测试；
- 配置固定 `gpu_engineering_enabled=false / formal_training_enabled=false / validation_evaluation_enabled=false / research_test_enabled=false`；`resource_lock_required=true` 只阻止越阶段运行，不表示资源决定候选；
- 完整 train-only signal audit、冻结 receipt/manifest/summary 与全部 hash/access/finite 验收；
- 用户确认的 signal-substrate lock、五项 scientific candidate lock，以及 exact CPU implementation receipt；
- `docs/experiments/resp_temporal_v1_gpu_engineering_protocol_20260820.md`、冻结synthetic benchmark config、fail-closed harness、严格receipt/manifest schema与GPU engineering implementation receipt。
- v1 cuDNN环境失败的冻结receipt/manifest，以及`docs/experiments/resp_temporal_v1_gpu_engineering_correction_v2_20260820.md`规定的unset环境、cuDNN 92000与LSTM canary preflight。
- 完整v2 execution receipt/manifest、五项candidate engineering summary与用户确认的GPU engineering lock；formal五项据此统一physical batch=128、accumulation=1。
- `docs/experiments/resp_temporal_v1_formal_protocol_20260820.md`与dual-GPU v2 runner；用户已完成5×3=15项，每项80 epochs/6400 updates、2675 validation rows，全部receipt/lifecycle/hash/access/finite闭合且无失败。
- formal implementation receipt SHA-256=`0d2dae3fea51f18821a47a11e69d8a8b2ca765898c3d84c228d2fedffcaff0fa`；66项RTM-v1 CPU定向测试通过，未读取dataset/index/split或既有checkpoint，未使用GPU。
- 用户在0/15时新增双GPU调度要求；`docs/experiments/resp_temporal_v1_formal_dual_gpu_correction_20260820.md`将旧plan冻结为`superseded_before_execution`，新plan固定`gpu_0` 8项/`gpu_1` 7项并强制、记录`CUDA_VISIBLE_DEVICES`。Correction/plan/implementation receipt SHA-256=`1c2cad1c5cc401776d894a0bb039e0ad75a8cd3030f66ea70b48817f2de1b84a / c6594d160a3f7ecb1f39a4e036996491dd6f1919aedeea28949ff8533a1dd8c3 / fa07aae4f83b74c86bf5113840058af99a768d33b16aad883dc6d6368998d888`，68项RTM-v1 CPU定向测试通过。
- `docs/experiments/resp_temporal_v1_validation_summary_protocol_20260821.md`冻结三seed mean±sample SD、50项paired-seed方向、独立quality/quality-efficiency tolerance-aware Pareto、T0停止线与validation-only access。Config/implementation receipt SHA-256=`21f723580c9cbbe39a3b85c68290e5ace26c6dadf0a9b37abdee9dd8a046816f / ffe4e658ddf9ddb9fb0f0456cf0425c7b2f58994ef4af4d53c693d4911f6149c`；81项RTM-v1 CPU定向测试通过。
- Validation summary已一次性完成，receipt/manifest SHA-256=`6f9f1e873b8910b22241bc0e9f2c510909835b0bbc2a9edc12f0e1788fa59aad / aab22094d6efd11927c952e9f12bcbab24e30cedc9a2a5f282f61056f1f193dc`。用户确认的validation lock SHA-256=`989ef0a3a5941ead3e80aba25606878f88d315bd23a1cf5ca4260317f3cffce6`；质量Pareto仅含multiscale representative，质量—效率Pareto保留全部五项，overall winner为空。
- 独立测试集协议/config/一次性评价器已建立并完成执行：固定五候选×三seed的15个validation-selected checkpoint，只输出逐sample metrics、15行seed direct mean与5行candidate `mean ± sample SD`，不计算secondary、Pareto、排名、paired方向或p-value。Protocol/config/implementation receipt SHA-256=`1da00281ad435a14cc0b1e9a26b84554cae35ec02784ea7f2f369ec539be8184 / 3d8551989fbfa07e8ef9454fbb348f2908151f35c681e15a6191b61a0c60a406 / e9e31c01b8f2da7a441d26b115f446c9fd71a7fc213ea3b07b5528b63a1615e6`；execution receipt/manifest/主指标表 SHA-256=`f9b1b216f80c4bde5a7be27aa9df76c66b8e3765ebe491ebe45fa880dd380a1e / a18d1459a3e4e9f11728c2bcebffd64f1f016f65129345c359794eee2a39ce4b / 91be5de5607de03a678b34f597100991ce81051f150b7c5cfdd7dfb508613454`。15/15 checkpoint、34650 rows、finite/nonfinite与access flags全部闭合。

当前明确未建立：

- 独立测试集结果的用户确认结论锁。

GPU engineering、formal训练、validation summary与独立测试集评价均已关闭且不得重跑，validation lock保持不变。下一步只允许用户确认当前结果结论；不得重复评价、追加搜索或根据测试结果重选candidate/checkpoint。
