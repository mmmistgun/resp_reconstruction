# Research v2 数据准备改造设计

## 背景

当前训练仓库在 `mixed_zscore` 小规模实验中已经暴露出一个关键问题：模型可以学到一定呼吸节律和频谱信息，但输出波形与 THO 目标仍然距离较远，且常把 BCG 中的高频或尖峰结构带到预测结果中。继续单纯增加 smooth loss 或改模型结构，不能直接解决“训练监督是否正确表达下游任务”的问题。

对 `/mnt/disk_code/marques/resp_prepare` 的只读审查显示，现有数据准备流程工程上是闭环的：Stage A-E 完成基础预处理，Stage 1.2-1.4 做 lag 诊断和对齐策略，Stage 1.5 生成训练前片段候选，Stage 2.1 生成 fixed-window dataset index。现有数据集适合作为 baseline，但它更偏向“保守、干净、可交接”的训练集，而不是面向呼吸率、相位事件、峰谷、过零点和弱监督重建的研究型数据集。

当前 Stage 1.5 总时长约 `392.8h`，其中：

- `include_candidate` 约 `156.6h`，约 `39.9%`
- `review_required` 约 `19.5h`，约 `5.0%`
- `exclude` 约 `64.4h`，约 `16.4%`
- `below_min_duration` 约 `152.3h`，约 `38.8%`

这说明大量未进入训练的数据不一定完全不可用，而是被当前片段切分和最小时长规则排除。另一方面，下游 `mixed_zscore` 实际训练窗口的对齐并非完全失败：窗口加权的片段 residual abs median 中位数约 `0.21s`，但 90 分位可到约 `0.72s`，部分片段的 p90 residual 可达数秒。现有流程做了不少对齐，但还没有把“这个窗口支持哪类监督”明确交给下游训练。

因此，本轮目标不是简单“重做数据集”，而是设计一条 Research v2 数据准备分支：在保留现有 baseline 的同时，重新定义对齐、耦合状态和监督可信度，让更多弱信号数据可以进入训练，同时避免把不可靠片段错误地当作强波形监督。

## 目标

- 保留现有 Stage A-E 和 Stage 2.1 baseline 产物，不覆盖当前数据集。
- 从研究视角重构 Stage 1.2 之后的数据准备逻辑，形成并行的 Research v2 分支。
- 引入 `coupling state segment` 作为中心概念，表示 BCG 与 THO 呼吸参考之间耦合关系相对稳定的一段时间。
- 区分短时体动、转身/姿态变化、长期耦合状态变化、幅值不可靠和真正不可用。
- 同时输出 raw/timebase-aligned input、reference-assisted state-aligned input 和 lag/confidence metadata。
- 每个窗口输出 rate、phase、event、waveform、normalization、alignment 等监督可信度，用于下游 loss weighting、采样和分层评估。
- 第一版训练窗口仍保持 `180s`，以便和当前 `tho_small` 实验可比；同时记录短窗口候选诊断，为后续多尺度训练预留空间。

## 非目标

- 不在当前训练仓库直接实现 `resp_prepare` 的代码改造。
- 不推翻 Stage A-E 的基础处理链路。
- 不把 reference-assisted alignment 当作真实可部署推理流程。
- 不把所有弱信号都强行用于 waveform loss。
- 不引入腹带或胸腹综合 effort 任务。
- 不在第一版引入多尺度训练窗口。
- 不把 confidence 设计成一次性定死的阈值体系；第一版应保留诊断和阈值校准空间。

## Stage A-E 重新解释

Stage A-E 不建议推翻，但需要重新定义其在 Research v2 中的角色。

### Stage A

Stage A 负责原始 BCG、THO 和睡眠分期加载，统一重采样到 100Hz 并按最短公共时长裁剪。现有 summary 未显示明显时长异常，可作为可信基础层保留。

Stage A 不解决 BCG 与 PSG 起点偏移、全局 lag 或采样时基漂移。这些问题应由后续 Research v2 对齐层处理。

### Stage B

Stage B 当前输出 THO/BCG extreme motion 和 unusable mask。Research v2 中应将这些 mask 重新解释为多级质量证据，而不是简单硬排除。

建议区分：

- `hard_invalid_sec`：离床、通道失效、长时间 flatline、明显饱和、极端剧烈体动、大面积缺失或非有限值。
- `transient_motion_sec`：短时体动，污染当前瞬时幅值和局部波形，但不一定改变之后耦合状态。
- `posture_transition_candidate_sec`：转身或姿态切换候选，可能导致后续长时间耦合状态变化。
- `amplitude_unreliable_sec`：绝对幅值不可靠，但节律或相位仍可能可用。
- `normalization_unreliable_sec`：不应参与归一化参考池的秒点。

短时体动默认不作为 hard exclude；它应降低 waveform、amplitude 和 normalization confidence。转身不能被当作普通短时噪声，因为它可能改变后续较长时间的 BCG-THO 耦合状态。

### Stage C

Stage C 当前输出 `0.05-0.5Hz` 的 `tho_on_axis` 和 `bcg_on_axis`。这对稳定低频呼吸很干净，但对较快呼吸、双峰、下降顿挫等形态可能偏窄。

Research v2 应保留当前 Stage C 输出作为 baseline，同时增加并行表征：

- `resp_band_0p05_0p5`
- `resp_band_0p05_0p7`
- 必要时保留宽频 rawish BCG 作为辅助输入

Stage C 输出不再直接决定训练可用性，而是作为对齐、事件提取和 waveform reference 的候选表征。

### Stage D

Stage D 当前基于 amp level 构造幅值状态、motion 和 low quality。Research v2 中，Stage D 应从“硬切训练片段的依据”降级为“耦合状态和监督可信度的证据来源”。

短时幅值跳变不应直接触发长期状态切分。只有持续足够久、前后稳定、且不能由短时体动解释的变化，才作为 coupling state 边界候选。

### Stage E

Stage E 当前基于 amp level 做连续归一化、segment gain 和 extreme motion compression，产出 `tho_normalized_zscore` 等信号。它适合做稳定 waveform target 的候选，但不应成为 phase/event/rate 的唯一参考。

Research v2 应拆分 THO 参考语义：

- `tho_event_phase_ref`：优先来自 Stage C `tho_on_axis` 或 `0.05-0.7Hz` 表征，用于峰谷、过零点和相位事件。
- `tho_waveform_ref`：可来自 Stage E `tho_normalized_zscore`，用于 waveform loss，但只在 normalization 可信时使用。
- `tho_rate_ref`：由 event/phase reference 或频谱估计得到，用于呼吸率和谱监督。

## 中心概念：Coupling State Segment

`coupling_state_segment` 表示 BCG 与 THO 呼吸参考之间耦合关系相对稳定的一段时间。它不是单纯幅值分段，也不是简单 valid segment。

每个 state 应记录：

- `coupling_state_id`
- `state_start_s`
- `state_end_s`
- `state_duration_s`
- `state_bcg_resp_power`
- `state_bcg_amp_level`
- `state_lag_median_s`
- `state_lag_iqr_s`
- `state_drift_s_per_hour`
- `state_rate_agreement`
- `state_phase_agreement`
- `state_event_agreement`
- `state_waveform_corr`
- `state_alignment_confidence`
- `state_confidence`
- `state_reason`

状态最小时长采用平衡策略：

- `>= 300s`：normal coupling state
- `180-300s`：short valid state，可进入训练但 confidence 上限受限
- `< 180s`：diagnostic only，不进入第一版 180s 训练窗口

短时体动不切断 state，只降低局部窗口可信度。转身或姿态变化作为候选边界，只有当其前后 BCG-THO 关系发生持续变化时才确认新 state。

## Research v2 数据流

Research v2 作为并行分支，不覆盖现有 Stage 1.5 / Stage 2.1。

现有 Stage 1.2-1.4 的人工结论只作为历史参考和初始先验，不作为 Research v2 的不可推翻约束。除已经明确排除的两份不可自动对齐数据外，其余样本都应允许重新评估对齐、耦合状态和监督可信度。

```text
Stage A-E baseline preprocessing
        ↓
Stage R1: multi-representation signal bank
        ↓
Stage R2a: coupling state discovery
        ↓
Stage R2b: state 内 supervision confidence 评估
        ↓
Stage R3: state-level reference-assisted alignment
        ↓
Stage R4: window-level confidence estimation
        ↓
Stage R5: research dataset index/export
```

建议在 `resp_prepare` 中使用独立产物目录，避免污染旧链路：

```text
artifacts/research_v2_signal_bank/
artifacts/research_v2_coupling_states/
artifacts/research_v2_alignment/
artifacts/research_v2_confidence/
artifacts/research_v2_dataset/
```

## Stage R1：Signal Bank

R1 不推翻 Stage A-E，只构造下游任务需要的最小多表征信号库。

BCG 表征：

- `bcg_rawish_wideband`：接近当前 mixed BCG，频带约 `0.03-20Hz`，保留心搏、复杂耦合和弱信号线索。
- `bcg_resp_band_0p05_0p7`：主对齐和 phase/event 表征，用于捕获呼吸节律、峰谷、过零点和稍快呼吸。
- `bcg_legacy_on_axis_0p05_0p5`：当前 Stage C 低频表征，用于 baseline 和稳定低频对照。

THO 参考：

- `tho_event_phase_ref`：优先来自 Stage C 或 `0.05-0.7Hz` 表征，用于 phase/event。
- `tho_waveform_ref`：来自 Stage E `tho_normalized_zscore`，用于 waveform 监督候选。
- `tho_rate_ref`：由 event/phase ref 或频谱估计派生，用于 rate 监督。

## Stage R2：Coupling State Discovery

R2 分两级。

R2 的核心约束是：早期逐窗 lag 诊断可能存在周期歧义、弱峰或局部跳变，不能被当作逐窗真值直接切段。真实 BCG-THO 时间关系可以缓慢变化，也可以在转身/姿态变化后进入新稳定状态，但不应频繁跳变。现有 Stage 1.2-1.4 的稳定 offset、drift 和人工复核结果可以作为初始化或对照先验，但 Research v2 必须允许重新估计。

### R2a：状态发现

R2a 用于发现 BCG-THO 关系相对稳定的 coupling state。主依据是：

- respiratory-band lag median 的稳定估计
- lag IQR / lag stability / lag trajectory smoothness
- rate agreement
- BCG respiratory-band power
- BCG amplitude/envelope state
- posture transition candidate
- hard invalid / long low-observable gap

R2a 不要求 waveform 高度相似，也不要求峰谷完全一致。它只回答：这段时间是否处于同一种稳定耦合关系。

状态发现应采用“先验约束 + 稳健估计”的方式：

- 用现有 Stage 1.2-1.4 的 sample-level 或 region-level 稳定 lag 作为初始候选。
- 对逐窗 lag 使用中值、分位数、HMM/动态规划或平滑分段模型做稳健化，避免被单个窗口的周期错配牵引。
- 限制 state 内 lag 只能缓慢变化；只有转身/姿态变化或持续的关系变化才能触发新状态。
- 对频繁正负跳变、贴近呼吸周期倍数的 lag 序列，优先标记为 `lag_ambiguous`，降低 phase/event/waveform confidence，而不是直接切出大量短状态。
- 对信号很弱但 rate 一致的片段，允许保留 rate supervision；不要因为 lag 不稳定就整体丢弃。

边界策略采用混合策略：

- 体动/转身提供候选边界。
- 信号关系变化负责确认边界。

候选边界来源：

- `posture_transition_candidate`
- sustained amp level change
- sustained BCG respiratory-band power change
- 稳定化后的 lag regime change
- phase agreement change
- long low-observable interval

候选边界确认时，建议比较前后 `120-180s` 的稳定观察窗。如果前后变化持续存在，确认新 coupling state；否则视为 transient motion 或局部污染。

### R2b：state 内监督可信度

R2b 在每个 coupling state 内评估其支持哪些监督：

- rate confidence：主频或 RR 是否一致。
- phase confidence：允许固定 lag 后相位是否稳定。
- event confidence：峰、谷、过零点是否有稳定对应。
- waveform confidence：band-limited 波形是否相关。
- normalization confidence：幅值、体动和归一化参考池是否可信。

这样可以避免因为 waveform 不像，就丢掉 rate/phase 有价值的数据。

## Stage R3：Reference-Assisted Alignment

R3 同时输出 raw/timebase-aligned input、state-aligned input 和 lag metadata。

每个 coupling state 内按复杂度选择对齐方式：

- `constant_shift`：state 内 lag 稳定。
- `linear_drift`：state 内 drift 明显且拟合残差小。
- `none`：lag 不稳定或周期歧义，保留 raw，只降低 confidence。

同一个 state-level lag/drift 同时应用到 respiratory-band BCG 和 wideband BCG，输出：

- `bcg_rawish_wideband_to_tho_timebase`
- `bcg_resp_band_to_tho_timebase`
- `bcg_legacy_on_axis_to_tho_timebase`
- `bcg_rawish_wideband_state_aligned`
- `bcg_resp_band_state_aligned`
- `bcg_legacy_on_axis_state_aligned`
- `state_alignment_method`
- `state_alignment_lag_s`
- `state_alignment_drift_s_per_hour`
- `state_alignment_confidence`
- `state_alignment_is_reference_assisted`
- `state_alignment_valid_sec`
- `state_alignment_edge_invalid_sec`

`state_aligned_*` 是 reference-assisted diagnostic input。它用于诊断对齐误差是否是瓶颈，不代表真实部署时可直接获得的输入。

## Stage R4：Supervision Confidence

R4 输出连续分数和离散等级。

建议字段：

- `rate_confidence_score`
- `rate_confidence_level`
- `phase_confidence_score`
- `phase_confidence_level`
- `event_confidence_score`
- `event_confidence_level`
- `waveform_confidence_score`
- `waveform_confidence_level`
- `normalization_confidence_score`
- `normalization_confidence_level`
- `alignment_confidence_score`
- `alignment_confidence_level`
- `supervision_confidence_score`
- `supervision_confidence_level`

初版等级可采用：

```text
high: score >= 0.75
medium: 0.45 <= score < 0.75
low: score < 0.45
```

这些阈值应通过诊断图和分布校准，不应在第一版被视为固定科学结论。

设计原则：

- `rate_confidence` 最宽松，不要求波形相似。
- `waveform_confidence` 最严格，需同时考虑 alignment、waveform 和 normalization。
- 不用一个总分决定所有 loss。
- `supervision_confidence_score` 可用于采样或摘要，但具体 loss 应看对应任务的 confidence。

示例组合：

```text
rate task: rate_confidence
phase task: min(alignment_confidence, phase_confidence)
event task: min(alignment_confidence, event_confidence)
waveform task: min(alignment_confidence, waveform_confidence, normalization_confidence)
```

## Stage R5：Research Dataset Index

第一版训练窗口保持：

```text
window_s = 180
step_s = 30
```

窗口规则：

- 窗口不能跨 coupling state。
- `hard_valid_ratio` 达标。
- `state_alignment_valid_ratio` 达标。
- 至少具备一种 medium+ 监督：rate、phase、event 或 waveform。

同时记录短窗口候选诊断：

```text
short_window_candidate_s = 60 / 120
```

短窗口候选不进入第一版训练，只用于分析哪些数据因为 180s 太长而未进入训练。

`dataset_index_research_v2.csv` 建议字段：

```text
dataset_row_id
split
samp_id
coupling_state_id
window_start_s
window_end_s
window_duration_s
source_npz
bcg_input_key
bcg_input_aligned_key
target_event_phase_key
target_waveform_key
target_rate_key
hard_valid_ratio
state_alignment_valid_ratio
transient_motion_ratio
posture_transition_ratio
amplitude_reliable_ratio
normalization_reliable_ratio
rate_confidence_score
rate_confidence_level
phase_confidence_score
phase_confidence_level
event_confidence_score
event_confidence_level
waveform_confidence_score
waveform_confidence_level
alignment_confidence_score
alignment_confidence_level
supervision_confidence_score
supervision_confidence_level
state_alignment_method
state_alignment_lag_s
state_alignment_drift_s_per_hour
state_alignment_is_reference_assisted
allowed_losses
reason
```

`allowed_losses` 示例：

```text
rate
rate;phase
rate;phase;event
rate;phase;event;waveform
```

## 实验设计

Research v2 支持两阶段实验。

### 阶段一：诊断上限实验

使用 reference-assisted state-aligned input，回答：对齐修正后模型能否显著改善 rate、phase、event 或 waveform 指标。

如果改善明显，说明现有数据准备中的对齐误差是主要瓶颈之一。如果仍不改善，则问题更可能在输入表征、loss、模型或 BCG-THO 生理耦合本身。

### 阶段二：可部署逼近实验

使用 raw/timebase-aligned BCG input，不在输入侧依赖 THO 参考做 state alignment。保留 lag/confidence/coupling metadata 用于训练采样、loss weighting 和评估分层，但不能在推理时依赖 THO 生成 input alignment。

建议对照：

- current dataset baseline
- raw input + confidence-aware loss
- state-aligned input + confidence-aware loss
- high-confidence-only subset
- raw input + lag-tolerant loss

## 风险与边界

- Reference-assisted alignment 可能带来 oracle 上限偏差，必须在实验命名和解释中明确。
- Confidence 指标容易工程化。第一版应保持连续分数、离散等级和诊断图并存，不把阈值当最终理论。
- Coupling state 不应被幅值变化单独决定。幅值变化必须结合 lag、rate、phase 或 event 关系确认。
- 过度放宽数据纳入可能污染训练。Research v2 应允许 rate-only 或 phase-only 监督，而不是把所有数据都推给 waveform loss。
- Stage E 归一化目标不应替代 THO event/phase reference。

## 验收标准

第一版 Research v2 数据准备设计完成后，应能回答：

- 每个样本有多少 coupling state。
- 当前未进入 baseline 的时长中，有多少可作为 rate/phase/event/waveform 不同等级监督重新纳入。
- 每个窗口支持哪些 loss。
- raw input 和 state-aligned input 的窗口数量、质量分布和 split 分布。
- reference-assisted alignment 是否明显改善下游诊断实验。
- confidence 分层是否能解释模型成功和失败的窗口类型。

## 后续实施位置

本设计文档先保存在训练仓库，用作跨仓库研究设计底稿。真正实现应在 `/mnt/disk_code/marques/resp_prepare` 中另起对话或工作分支完成，并保留当前 baseline 数据集作为对照。
