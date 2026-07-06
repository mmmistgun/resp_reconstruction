# 从信号本体出发的实验框架思考存档

日期：2026-07-04

## 定位

本文是一次思考存档，不是执行计划、预注册实验或当前主线替换方案。它记录一次从呼吸信号本体出发的框架回顾：暂时不被当前数据集形态约束，包括窗长、步长、质量标签、目标定义、输入缓存、训练入口和既有评价口径都可以被重新想象。

现有实验结果在本文中只作为“失败模式”和“线索”参考，不作为最终排除某方向的定理。许多失败可能来自当前数据构造、窗口尺度、标签质量、输出定义、优化盆地或评价口径，而不一定说明对应信号假设无效。

本文也不改变任何数据 split、标签定义、指标口径或主线实验结论。若后续要把其中某个想法变成正式实验，需要另写具体设计、影响面说明、manifest、smoke、验证和记录。

## 参考背景

本次回顾主要参考：

- `docs/experiments/time_frequency_input_fusion_plan.md`
- `docs/experiments/e1_stft_info_gain_20260622.md`
- `docs/experiments/f_series_stft_loss_plan.md`
- `docs/experiments/g_series_stft_input_resolution_band_plan.md`
- `docs/experiments/rawish_state_aligned_l0_l1.md`
- `docs/experiments/softz_20260620_model_candidates.md`
- `docs/experiments/tho_research_v2_performance.md`

这些文档里稳定出现了几类现象：纯 waveform 回归容易受极性、局部峰拓扑、谐波误拣和窗口质量影响；STFT/高频输入常能改善 hard 或 low-spectrum 窗口，但容易损伤 easy 或 fast-RR 窗口；输出带限和低自由度结构能显著降低局部尖峰，但可能损伤相对努力表达；方向约束可以修正极性盆地，但会带来 RR 或 raw peak 代价。

## 从信号本体重述任务

当前任务表面上是：

```text
BCG window -> THO waveform window
```

但从信号本体看，更合理的隐藏结构可能是：

```text
BCG observation
  -> latent respiratory state
     -> THO-like observation / waveform / RR / events
```

其中 latent respiratory state 至少包含：

- 呼吸相位 `phase(t)`
- 瞬时呼吸率或相位速度 `RR(t)` / `d phase / dt`
- 呼吸幅度或 effort envelope `A(t)`
- 呼吸事件拓扑：peak、trough、zero-crossing、cycle boundary
- 极性或方向 `polarity`
- 小范围 lag / alignment uncertainty
- 局部质量、可识别性和不确定性

THO waveform 是这个状态的一种观测，不一定是唯一正确输出。BCG 到 THO 的映射存在不适定性：BCG 中既有呼吸位移，也有心冲击、体动、姿态、接触状态和设备噪声。某些窗口可以恢复稳定 RR，但不一定能恢复胸带局部波形形状；某些窗口胸带本身也可能多峰、错位或质量不足。

因此，后续探索可以把“一维 waveform”降级为一个可渲染观察，而把核心监督转到呼吸状态、事件和节律上。

## 可放开的数据形态

如果暂时不受当前数据集形态限制，数据构造本身可以重新设计。

### 多尺度窗口

固定 180s 窗口对频率分辨率和稳定统计有好处，但会混合多个状态、事件和质量区间。可以考虑同时构造：

- `20-30s`：局部峰形、事件、短时节律和 fast-RR。
- `60s`：稳定 RR、呼吸努力 envelope 和局部质量。
- `180s`：整段状态、质量、低频漂移和 hard/low-spectrum 诊断。
- 长上下文加短目标：模型读取 180s 或更长上下文，但只监督中间 30-60s，减少边界和状态混合。

这能把“需要长窗频率分辨率”和“需要短窗局部拓扑”拆开，而不是让一个窗口尺度同时承担所有目标。

### 可变步长和事件对齐采样

当前固定步长容易产生大量高度相邻窗口，也会让少数事件段重复影响统计。可以考虑：

- 稳态段使用较大步长，降低重复。
- 状态切换、事件、低质量、高 ambiguity 区间使用更密步长。
- 按呼吸 cycle 或事件中心采样，而不是只按时间网格采样。
- 对相邻窗口建立 group 或 sequence，避免把连续坏例当作独立证据。

### 标签重构

质量标签不应只有一个粗粒度等级。更有用的标签可能是：

- input BCG quality
- target THO quality
- alignment quality
- peak ambiguity
- harmonic / subharmonic ambiguity
- motion dominance
- fast-RR ambiguity
- polarity confidence
- learnability / observability

这些标签不一定都来自人工标注，可以由 signal audit、规则诊断和少量人工复核组合生成。关键是区分“输入不可见”“目标不可靠”“映射不唯一”和“模型没学会”。

### 目标不再单一

每个窗口可以同时保存多种 target：

- 原始或 soft-z THO waveform
- bandpassed THO waveform
- low-frequency phase
- instantaneous RR trajectory
- breath event mask
- cycle duration
- envelope / effort
- target STFT 或 band-energy 诊断量
- target ambiguity / uncertainty

训练时可以根据窗口质量和任务阶段选择 target 组合，而不是对所有样本强行使用同一种 waveform loss。

## 输入方向

### 低频主输入

BCG 的低频呼吸分量仍应作为主路径之一，但不应只依赖 raw waveform。可以探索：

- raw / wideband BCG
- respiratory-band BCG
- trend / baseline drift
- multi-resolution lowpass traces
- derivative / velocity-like trace
- local SNR 或 respiratory-band energy ratio

这些输入可以作为多通道输入，也可以作为显式低频状态 encoder。

### 高频心冲击调制

1-8Hz 高频不宜直接视为“噪声”或“捷径”。它可能包含心冲击幅度被呼吸调制、接触状态、姿态和体动信息。更合适的用法可能不是 dense map 直接注入 waveform，而是先提取调制/质量特征：

- cardiac band envelope
- ridge frequency
- ridge energy
- ridge stability
- high-band entropy
- second-ridge ratio
- high / low band quality ratio
- cardiac envelope 在 0.05-0.7Hz 上的调制功率

这些特征更像条件变量，用于判断当前 BCG 是否能支持呼吸重建，而不是替代低频呼吸路径。

### 时频输入的重新定位

已有 STFT、CWT、SST 实验提示：时频表示能提供 hard-window 线索，但复杂 dense map 和 attention 未稳定通过。后续可以把时频输入重新定位为：

- 质量/ambiguity 估计器。
- hard-window 修正器。
- 高频调制上下文。
- 训练期诊断或分层标签生成器。

而不是默认把 STFT/CWT/SST 当成第二个强主输入。

### 自监督预训练

BCG 本身很长，且标签质量有限。可以先预训练输入 encoder：

- masked segment reconstruction
- future segment prediction
- contrastive subject/session/state representation
- respiratory-band reconstruction
- cardiac modulation prediction
- motion/quality proxy prediction

预训练目标应避免把 subject identity 变成捷径。后续使用时需要跨 `samp_id` 或更严格 split 验证。

## 模型框架方向

### Latent respiratory state model

把模型分成两段：

```text
BCG encoder -> respiratory state decoder -> waveform/event renderer
```

state decoder 输出低频状态，例如：

- `phase(t)`
- `RR(t)` 或 `phase_velocity(t)`
- `amplitude(t)`
- `baseline(t)`
- `event logits(t)`
- `uncertainty(t)`
- `polarity / lag confidence`

renderer 可以是固定公式、可微 spline、cycle template 或小型 neural renderer。这样 waveform 不再由任意 18000 点自由产生，而由可解释状态生成。

### Phase-amplitude renderer

一种具体形式：

```text
y_hat(t) = baseline(t) + amplitude(t) * template(phase(t), shape_params(t))
```

约束：

- `phase(t)` 单调递增。
- `phase_velocity(t)` 平滑但允许局部变化。
- `amplitude(t)` 慢变。
- `template` 可以从简单 sinusoid 开始，再加入少量 shape 参数。

这条线适合解决局部尖峰、额外峰、呼吸次数和相位方向问题。

### Event topology model

另一条线是不直接生成连续 waveform，而是先预测呼吸事件：

- peak probability
- trough probability
- upward/downward zero-crossing probability
- cycle boundary
- invalid/ambiguous event mask

然后从事件生成 RR、breath count 或再渲染 waveform。它直接针对当前主指标中的 peak-band RR 和 count error。

### 条件修正，而非强融合

STFT/high-frequency 分支更适合作为条件修正：

```text
state_low = low_encoder(BCG_low)
context_high = high_encoder(BCG_high or modulation)
state = state_low + small_gate(context_high, state_low)
```

关键是 small-init、energy cap、hard/ambiguity proxy 和 easy/fast guard。不要让高频或时频分支直接接管最终 waveform。

### 双路径输出

可以拆成：

- 节律路径：负责 phase/RR/count。
- 努力路径：负责 envelope/amplitude。
- 形态路径：只给少量 shape correction。

当前 M10c 类带限输出在节律和降尖峰上有价值，但相对努力略弱。一个自然方向是保留带限主输出，再补 effort branch，而不是回到全自由 waveform。

## 输出空间方向

### 不建议把 STFT 作为主输出

低频 complex STFT 主输出在当前记录中暴露了明显风险：方向弱、RR 和 waveform 相关性退化。即使忽略具体数值，这个方向也有信号本体上的问题：

- STFT framewise peak 不等于全局呼吸 cycle。
- 多峰 target STFT 会误导 fast-RR。
- magnitude 或局部 complex 约束不能自然保证 cycle topology。
- iSTFT 边界、phase、NOLA 和 length 都增加新歧义。

更合适的输出空间是 phase/event/state，而不是 dense STFT。

### 低自由度 waveform 仍值得保留

带限输出、control point、cycle spline、局部 basis 都属于低自由度 waveform。但全局 basis 或过硬 frequency bottleneck 可能压掉相对努力。更合理的是局部低自由度：

- 分段 basis。
- cycle-level spline。
- low-frequency control points。
- bandlimited residual with small gate。
- template plus local shape parameters。

### 多输出而非单输出

一次 forward 可以输出：

- waveform
- bandpassed waveform
- event logits
- RR trajectory
- envelope
- uncertainty
- polarity/lag logits

训练和评价按任务分层使用这些输出。这样可以保留 waveform 可视化，同时不让 waveform loss 主导所有结论。

## Loss 方向

### 从点对点波形转向拓扑和状态

值得优先探索：

- soft breath count loss
- soft zero-crossing count loss
- differentiable peak interval loss
- event cross entropy / focal loss
- phase velocity smoothness
- circular phase loss
- amplitude envelope loss
- cycle duration loss
- peak topology consistency

这些 loss 更贴近“呼吸节律是否对”，而不是“每个采样点是否像 THO”。

### Polarity 和 lag 的 latent-variable 处理

现有 signed loss 能修正方向，但也带来 RR 代价。可以把 sign/lag 当成 latent variable：

- 训练早期对 `{+1, -1}` 和小 lag 做 min/marginal likelihood。
- 中期逐渐收紧到 state-aligned 正方向。
- 同时训练 polarity confidence head。
- 对低 polarity confidence 窗口降低 waveform 监督权重。

这比单纯提高 signed loss 权重更能表达“这个窗口本身方向证据不足”。

### Anti-harmonic / anti-subharmonic

STFT peak-anchor 的教训是：framewise target peak 可能和全局 RR 不一致。若做频域约束，应更依赖 cycle-level 或 global RR proxy：

- 只在 target peak 与全局 cycle/RR 一致时启用 anchor。
- 对 fast-RR 的低频误吸引单独加 guard。
- 用多峰分布而非单峰 hard target。
- 用 harmonic rank / subharmonic risk 做诊断权重。

重点不是继续调一个 scalar，而是避免错误锚点。

### Heteroscedastic / uncertainty loss

不可学习或目标不可靠窗口不应被同等强度监督。模型可输出 `sigma(t)` 或 window-level uncertainty：

```text
L = error / sigma^2 + log sigma
```

需要防止模型用无限 uncertainty 逃避训练，可加先验或上限。它适合和质量标签、event ambiguity、motion dominance 结合。

## 训练策略方向

### Curriculum

可以分阶段训练：

1. 高质量、稳定呼吸窗口，只学 RR/phase/envelope。
2. 加入正常质量窗口，打开 waveform renderer。
3. 加入 hard/low-spectrum/fast 窗口，打开条件高频或时频修正。
4. 最后微调不确定性和 hard-window loss。

这样比一开始让所有窗口、所有 loss、所有分支同时竞争更清楚。

### Balanced sampler

采样不应只按窗口数量。可以按以下因素平衡：

- RR bin：slow / normal / fast。
- quality bin。
- spectrum ambiguity。
- subject/session。
- baseline easy/hard 的训练期 proxy。
- event / transition presence。

如果 fast-RR 或 dirty-easy 很少，均值会掩盖系统性失败。

### Checkpoint selection protocol

现有经验说明 val loss 不够。checkpoint selection 可以是：

```text
direction gate
+ task RR
+ breath count
+ easy/fast guard
+ hard/low-spectrum benefit
```

这不是新模型，而是选择协议。它应和训练 loss 分开记录，避免把 checkpoint selection 收益误写成模型结构收益。

## 评价和诊断方向

如果输出改成 respiratory state，评价也应分层：

- waveform metrics：只作为可视化和形态诊断。
- RR metrics：peak-band、spec、event-based RR。
- count metrics：zero-cross count、event count。
- phase metrics：phase error、phase velocity error、polarity。
- envelope metrics：relative effort。
- uncertainty calibration：错误是否集中在高不确定性窗口。
- topology metrics：多峰、漏峰、额外峰、cycle duration outlier。

分层应继续保留：

- baseline easy / hard。
- clean easy / dirty easy。
- low spectrum similarity。
- fast RR。
- low confidence。
- high ambiguity。
- motion dominated。

## 候选实验想法清单

以下只是想法清单，不是执行顺序。

### H0：respiratory state target 审计

不训练模型。先从 THO 生成多目标审计表：

- peaks / troughs / zero-crossing。
- cycle duration。
- instantaneous RR。
- phase proxy。
- envelope。
- event ambiguity。
- target multi-peak / harmonic risk。
- clean easy / dirty easy 拆分。

产物应服务后续所有 state/event 训练。

### H1：phase-amplitude renderer

模型输出 `phase_velocity + amplitude + baseline + shape`，renderer 生成 waveform。主比较不是 waveform loss，而是 RR、count、phase、envelope 和 easy/fast guard。

### H2：event topology head

在当前强 baseline 或 M10c 类模型上增加 event head，训练 peak/trough/zero-crossing logits，并用 event-based RR 监督。目标是降低 peak topology 错误，而不是让 waveform 更像。

### H3：latent polarity / lag marginalization

把 sign 和小 lag 作为 latent variable，训练时边际化或最小化，后续再用 direction head 收紧。目标是降低对 signed loss 权重的依赖。

### H4：高频 modulation conditional gate

只使用高频调制特征，不使用 dense CWT/SST map。通过 small gate 影响 respiratory state，目标是保留 hard/low-spectrum 收益，同时不伤 clean/easy 和 fast-RR。

### H5：uncertainty-aware loss

训练 window-level 或 time-level uncertainty，让模型识别不可学习、目标多峰或低质量窗口。重点看 calibration 和 hard/easy 分层，而不是只看均值。

### H6：多尺度窗口模型

长上下文 encoder + 短目标 decoder。输入可以是 180s 或更长，监督只在中心 30-60s。目标是减少边界、状态混合和长窗平均化问题。

### H7：effort branch for bandlimited output

保留带限节律输出，单独增加 effort/envelope branch，补 M10c 类模型的相对努力损失。

### H8：training-time ambiguity proxy

从输入和 target 本身生成无泄漏 proxy，用于判断训练期是否开启 STFT/high-frequency/residual 约束。重点避免使用验证 paired delta。

## 暂时降优先级的方向

这些方向不一定无效，但从信号本体和现有失败模式看，短期高后悔成本较高：

- 继续扩大 dense CWT/SST map、attention pooling 或 TF residual head。
- 继续做 STFT 主输出空间、full-band complex output 或 magnitude-only output。
- 继续围绕 `stft_peak_anchor_weight`、`sigma_bins`、confidence scalar 做小网格。
- 继续加深 cross-attention 或复杂融合结构。
- 简单 moving-average 输出平滑作为主线。
- 全局 basis decoder 或过硬 frequency bottleneck。
- 只按 val loss 或单一 RR 均值选模型。

## 风险与边界

如果后续把这些想法变成正式实验，以下操作属于高影响变更，必须另行说明影响面：

- 改变 target 定义，例如从 THO waveform 改成 phase/event/state。
- 改变窗口长度、步长、样本独立性或 split 逻辑。
- 引入新质量标签、过滤规则或样本权重。
- 改变核心评价指标或模型选择协议。
- 用训练期 proxy 替代现有质量标签。
- 启动长训练或大规模搜索。

尤其需要标注旧结果是否可比。若 target、split、窗口或指标变了，旧 run 通常只能作为失败模式参考，不能横向比较。

## 当前思考收束

如果只从呼吸信号本体出发，下一阶段最值得关注的是：

1. 把输出从任意 waveform 改为 respiratory state / event / renderer。
2. 把质量从粗标签改成 ambiguity、learnability 和 uncertainty。
3. 把高频和时频信息当作条件上下文，而不是强主输入或主输出。
4. 把 polarity / lag 当作 latent uncertainty，而不是只用固定 signed loss 硬压。
5. 把 checkpoint selection 和训练 loss 分开设计，避免 val loss 或单一均值误导。

一句话总结：当前问题不像是“还缺一个更复杂的 STFT/CWT/attention 模型”，更像是“任务表述仍过度绑定 18000 点胸带 waveform”。更可信的下一类实验，应围绕呼吸状态、事件拓扑、低自由度输出和不确定性建模展开。
