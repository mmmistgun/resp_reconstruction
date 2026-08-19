# Respiration Temporal Modeling v1 协议

日期：2026-08-20

状态：**signal-first 修订、train-only 信号审计规范、严格配置、脚本、receipt schema 与定向确定性测试已建立；完整 train 审计尚未执行。现有 Full/Compact 结构只保留为 implementation probe，不是正式候选锁；formal runner、GPU engineering、formal training、validation summary 和 research-test 均未开放。**

## 1. 权威性、独立命名与当前边界

本文是 `docs/experiments/loss_metrics_restart_plan_20260729.md` 第 50 节引用的规范性附件；冲突时以主协议为准。协议 ID 固定为：

```text
resp-temporal-v1-validation-20260820
```

简称 `RTM-v1`。它是新的 validation-only 时序模型家族比较，不属于 CRD-v1.1、CRD-TF v1、CRD-TF-W v2 或旧 THO 阶段的补跑。旧阶段的 checkpoint、summary、配置和冻结结论只作只读背景，不进入本协议的候选集合，也不得覆盖或重跑。

本阶段由既有 validation/research-test 历史启发，因此即使未来完成正式矩阵，证据名称也只能是：

```text
validation-development evidence; research-history-informed;
not untouched held-out or unbiased generalization evidence
```

RTM-v1 只允许读取 train/validation。当前实现不提供训练或评价入口，更不提供 test split；未来若开放 validation runner，也必须保持 validation-only。Research-test 不属于本协议任何阶段，不能因 validation 结果自动开放。

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

当前 implementation probe 的公共 stem 为：

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

第一层在 100 Hz 上先提取约 1.01 秒局部宽频信息，再逐级降采样；这避免让 LSTM/Mamba 直接处理 18,000 steps，也避免旧 140-token bridge 对快呼吸时序过粗。Normalization 只沿 channel，不允许公共 stem 通过全窗归一化替代 temporal trunk 的长程建模。该结构尚不能因 shape 与输出采样率合理就称为 signal-locked：审计必须判断 20 Hz 非线性处理是否足以保留 0.8–8 Hz carrier modulation，并明确后续降采样的抗混叠要求。

公共 decoder 暂定为 channel-only norm、`96→64 k5`、64-channel depthwise k5、`64→32→1` coarse head，加与 C201 相同的 1,057-parameter zero-init pointwise nonlinear residual；生成 10-Hz raw waveform 后用冻结 Fourier interpolation 恢复 100 Hz。模型内部不执行 `Pi`。其输出采样语义已有 C0 支持，但只有 signal audit 登记后，stem + decoder 才能共同写入 signal-substrate lock。

同 seed 的公共 stem/decoder 使用独立命名子 seed并要求跨所有 family 逐 tensor 相同。共同 substrate 用于提高 trunk 归因清晰度，不代表每个家族的端到端最优 frontend。

### 5.2 离线边界

所有 trunk 均允许完整 180 秒双向上下文：PatchMixer 全窗 mixing、TCN 对称 padding、Mamba 双向扫描、LSTM 双向 recurrence、多尺度分支对称 pooling/convolution。因此结果只能解释为离线重建能力。

## 6. 家族信号假设与当前 implementation probes

家族进入正式矩阵的依据是其是否对应独立、可证伪的信号建模假设，而不是参数量、显存或与其他模型的规模接近程度。当前代码中的 Full/Compact 仅用于验证结构可实现、shape 正确和容量范围合理；它们尚未成为 formal candidates，不能通过资源 gate 自动二选一。

signal audit 完成后，每个保留家族只锁定一个核心代表。深度、宽度无法由信号唯一推出时，应采用明确的家族标准规则（例如固定 latent width、标准 expansion ratio、最小合理层数），并如实标记为 architecture convention。若确有必要保留第二点，必须另立为 family-internal capacity-sensitivity control，且不允许依据 validation 结果追加。

### 6.1 T0 无 trunk 控制

`rtm_v1_t0_stem_head` 只包含公共 stem/decoder，当前 probe 参数 62,882。它是控制实验，不是第六个架构家族，也不因参数小而进入“公平参数排名”。其作用是判断 temporal trunk 是否提供净增益；若 signal audit 修改公共 stem，其参数合同随 substrate lock 一次性更新。

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

两者都具备完整 180秒 token mixing；Compact 不是短上下文版本。该实现实际是固定 1800 位置上的低秩 global token MLP，不是传统 patch hierarchy，也不具备卷积式时间平移等变性。只有在协议明确保留“弱局部先验的全窗位置混合”这一反方假设时，它才进入正式矩阵；否则关闭该家族，不因已有实现而自动运行。

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

两个 tier 都覆盖完整 180秒，宽度差异不改变上下文资格。TCN 的信号依据是局部准周期结构、层级 dilation 与完整 180秒上下文；`H=256/384` 只是当前容量 probes，不是由呼吸信号推导的宽度。

### 6.4 BiMamba2

沿用固定依赖与官方 fast path 的 D=96 双向 Mamba2 block；每方向独立，concat+linear 后残差：

- Compact：4 blocks，固定 697,138 参数；
- Full：6 blocks，固定 1,014,266 参数。

不得使用卷积式 SSM-like fallback。Mamba 的信号假设是用选择性状态表示长程、非平稳的速率与努力变化；4/6 层是当前容量 probes。依赖或 CUDA fast path 不满足时只能登记当前环境未评估，不得以 fallback 或 Compact 自动替代科学代表。

### 6.5 BiLSTM

在 `[B,1800,96]` 上先做 channel LayerNorm，再运行 H=96 的 bidirectional LSTM，双向输出经 `192→96` projection、dropout 后与输入残差相加：

- Compact：2 layers，固定 453,314 参数；
- Full：3 layers，固定 676,034 参数。

不开放 raw 18,000-step LSTM、单向 LSTM、GRU、hidden-size 或 projection 搜索。BiLSTM 的信号假设是显式门控状态能够追踪连续的呼吸相位、速率与努力；2/3 层是当前容量 probes，不是资源分档。

### 6.6 多尺度 feature pyramid

当前 probe 对共同 10-Hz latent 构造三个 feature 分支：

| 分支 | Grid | 长度 | dilation | 主要作用 |
|---|---:|---:|---|---|
| fine | 10 Hz | 1800 | 1/2/4/8 | 快速局部形态与相位 |
| medium | 2 Hz | 360 | 1/2/4/8 | 呼吸周期与局部变化 |
| coarse | 0.5 Hz | 90 | 1/2/4/8/16 | effort、Local-RR 与整窗上下文 |

降采样固定为 average pooling，编码后线性插值回1800并在 feature channel 上 concat+1×1融合。Coarse 分支感受野为125个0.5-Hz tokens，即250秒。所有分支保持96 channels，不生成独立 waveform head：

- Compact：branch expansion H=192，固定 579,842 参数；
- Full：H=384，固定 1,059,074 参数。

它与旧 M1 的关键区别是 feature-level、合理宽度、多尺度交互，而非四个单通道 waveform 分支的标量融合。正式分支采样率、低通/抗混叠算子和时间尺度必须由 signal audit 锁定。尤其 0.5-Hz 分支 Nyquist 仅0.25 Hz，只能承担慢上下文/努力建模，不能声称保留完整 0.05–0.70 Hz 呼吸波形；简单 average pooling 在审计前不视为充分的抗混叠证明。

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

当前既没有 signal audit receipt、signal-substrate lock 或 formal candidate lock，也没有 GPU engineering receipt，因此全部 formal training 继续关闭。

## 8. 最小实验矩阵

signal-first candidate lock 后的核心正式矩阵为：

| 模型 | configs | seeds | runs |
|---|---:|---|---:|
| T0 control | 1 | 20260811/12/13 | 3 |
| TCN representative | 1 | 同上 | 3 |
| Mamba representative | 1 | 同上 | 3 |
| BiLSTM representative | 1 | 同上 | 3 |
| Multiscale representative | 1 | 同上 | 3 |

核心矩阵共15个 formal runs、96,000 optimizer updates。若 signal audit 后明确保留 global token mixer 的独立反方假设，可在 candidate lock 中增加其唯一代表 ×3 seeds，使绝对上限为18 runs、115,200 updates。当前 Full/Compact 均不是 formal arm；工程 benchmark/implementation smoke 不计为科研 run，但成本单独记录。

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

- `resp_train/temporal/` 下的独立 initialization、blocks、model 和 config；
- `configs/resp_temporal_v1/` 下 T0 与五家族 Full/Compact 共11个 implementation-only probes；它们不是 formal candidates；
- 参数、shape、感受野、公共 state identity、invalid config 与非有限输入的定向测试；
- 配置固定 `formal_training_enabled=false / research_test_enabled=false`；现有 `resource_lock_required=true` 仅是阻止运行的旧实现字段，不再表示资源决定候选；
- train-only signal audit 的规范、严格冻结配置、确定性算子、不可覆盖输出、receipt schema 与定向测试；完整 train 执行结果和正式 receipt 尚不存在。

当前明确未建立：

- train/eval CLI；
- 完整 train signal-audit execution receipt、signal-substrate lock 与 formal candidate lock；
- GPU engineering benchmark receipt；
- formal config/runner/preflight；
- validation summary；
- research-test cache、allowlist 或 evaluator。

下一阶段只能由用户在提交实现且工作树干净后，手动执行固定的完整 train-only signal audit；Codex 不代跑长 CPU 审计。审计 receipt 登记并由用户确认 signal-substrate/family/candidate lock 后，才可另行授权 GPU engineering。GPU receipt 登记到主协议前不得开放 formal training；formal 结果完成后也不自动开放 research-test。
