# CRD-TF-W v2 融合、小波与 Local BiMamba 深度实验计划

日期：2026-08-16

状态：**已于 2026-08-17 被 `docs/experiments/crd_tf_w_v2_protocol_20260817.md` 取代。本文只保留为历史设计记录，不得据此实现、验收、训练、汇总或访问 research-test。**

## 1. 权威性与阶段定位

本文曾是 `docs/experiments/loss_metrics_restart_plan_20260729.md` 第 49 节引用的规范性附件；现已失效，执行口径只见 2026-08-17 修订版与主协议。

新阶段暂命名为 `CRD-TF-W v2`，只围绕已冻结候选 `crd_tf102_w` 回答三个问题：

1. 当前 zero-init FiLM 相对同位置、同 decoder、近似等容量的其他融合方法是否更合理；
2. W 分支的有效信息来自哪些频带，Morlet 时间—频率分辨率是否需要调整；
3. 当前 6 层 Local BiMamba2 是否处于合理的质量—效率位置。

本阶段是 **research-test-informed development**。实验可根据阶段性合理结果调整后续研究叙事，不要求在项目执行顺序上形成一次性线性故事；但不得修改已经发生的访问事实、伪造预注册状态，或把已访问并影响后续研究的现有 research-test 表述为“未触碰的独立测试”。

## 2. “稳健/合理结果”的操作定义

本阶段的“稳健”不要求 test split 完全独立，而要求结果在数值、优化、机制与任务交换上合理：

1. run 生命周期完整，checkpoint、optimizer、prediction 和全部正式指标 finite，prediction degeneracy 为 0；
2. 正式晋级结论使用三个固定训练 seed `20260811 / 20260812 / 20260813`；单 seed 只允许承担预注册的失败筛除，不形成正式优胜结论；
3. 相对当阶段 anchor，至少一项 primary 达到实质改善：Whole/Local RR、trajectory 或 global-envelope error 相对改善至少 `0.5%`，或 signed PCC 绝对增加至少 `0.002`，并至少 `2/3` paired seeds 同方向；
4. Whole RR、Local RR、trajectory、global-envelope 任一相对恶化超过 `3%`，或 signed PCC 绝对下降超过 `0.005`，默认判为不晋级；这类结果仍完整保留为任务交换或失败证据；
5. 不构造加权总分。若候选在不同任务轴互不支配，保留 tolerance-aware Pareto set，不强行选择唯一赢家；
6. validation 与 reused research-test 不要求效应量相同，但论文中的核心方向应可解释；若方向不一致，必须报告 split sensitivity，不得删除不利结果；
7. 显著增加参数、显存或延迟的候选必须产生相称的质量收益；轻量候选可以凭质量近似与明确效率收益进入 efficiency Pareto。

上述 `3%` 是本新阶段的“明显不合理退化”停止线，不追溯修改 P5/P6a 已冻结的 `1.5% / 0.005` 门槛和历史决定。

## 3. 现有证据与禁止重复项

### 3.1 直接复用的冻结证据

- C201 anchor：`crd_c201_decoder_10hz_cap × seeds 20260811/12/13`；
- W anchor：`crd_tf102_w × seeds 20260811/12/13` 的 validation-selected checkpoints；
- P5 matched-capacity `crd_tf_ctrl1`；
- P5 的 M/W/S/MS、全部 W-containing pair/triple、capacity 与 interaction 结果；
- P6a 的 MWS additive/gate 结果；
- C1 的六层 Local BiMamba2 与 parameter-matched full-context TCN 结果；
- C2 的 10-Hz decoder capacity 与 100-Hz placement 结果；
- 当前 Morlet calibration、train/validation cache 与现有 research-test cache/summary。

### 3.2 不再重复的实验

- 不重复 W vs C201、W vs CTRL1 或原 M/W/S/MS 矩阵；
- 不重复 MW/WL/WS/WLS 或 MWS 简单相加；
- 不重复 P6a 三分支 gate；其结果不能被外推为 single-W 下 FiLM 已优于所有 gate，但足以降低再次优先研究 branch gate 的必要性；
- 不重复 Local Mamba vs TCN；层数实验继续使用 Mamba；
- 不重新打开 100-Hz placement、其他 decoder 或 TCN+decoder；
- 旧 T3 concat-deep 只作负面背景，不能替代本阶段的同位置 clean concat；
- local cross-attention 保持关闭，除非本阶段简单融合全部失败且用户另立新协议。

## 4. 公共训练与工程口径

除本协议明确改变的科学因素外，全部新候选冻结：

```text
dataset / admission / split / target = 现有 CRD-TF v1
input = BCG 100 Hz × 180 s
loss = L_sync + 0.25 L_effort
metrics = 现有冻结 primary + secondary
checkpoint selector = full-validation Local RR minimum
optimizer = AdamW
max_lr / min_lr = 3e-4 / 3e-5
effective batch = 128
physical batch = 128, accumulation = 1（首选）
max_epochs = 80
early stopping = patience 30, min_delta 0
formal seeds = 20260811 / 20260812 / 20260813
AMP = bf16
```

P4 的 W 三个 selected epoch 为 `13 / 15 / 14`。在已完成的 80-epoch history 上回放 `patience=30` 不会改变这三个 validation-selected checkpoint，因此 W anchor 可复用，不因启用已冻结的 P6a early-stop 规则而重训。

每个新 variant 在 formal 前必须完成：

1. 严格 config/schema、参数数和初始化 identity 测试；
2. CUDA bf16 batch-1 forward/core-loss/backward，input/parameter gradient finite；
3. `128 train / 32 validation / 1 update` 独立 lifecycle acceptance；
4. best/final checkpoint、optimizer、逐 sample metrics finite，prediction nondegenerate；
5. 记录 peak allocated/reserved、throughput、trainable parameters；
6. 若 `128×1` 不可行，不得只为单 arm 临时改变 batch。统一 fallback 与必要 batch control 须先修订本协议。

当前 W formal peak reserved fraction 为 `81.524%`，因此新增 concat 或 D8 必须先通过实际目标 GPU 验收；不得仅依据参数数推断可运行。

## 5. P0：新阶段 candidate lock 与实现前冻结

P0 仅登记并审计现有 C201/W/CTRL1 checkpoint、config、manifest、metrics、cache 和 commit identity，不训练、不评价模型。

P0 完成前必须把本协议仍未确定的内容写死：

- 各融合 arm 的精确 tensor 公式、norm、projection 和参数匹配方法；
- screening failure-only 门槛及 screening run 是否可计入后续三 seed；
- Morlet Q 的数值定义与 input-only calibration 方法；
- D4/D8 的准确参数数、module seed 和 acceptance 资源线；
- 每阶段最多保留几个 Pareto 候选；
- 新输出目录、不可覆盖规则和汇总 schema。

P0 当前未开放。

## 6. P1：single-W 融合方式

固定当前 full-band Morlet W、97×360 cache、6 层 Local BiMamba2、FiLM 现有插入位置、refinement/head/decoder、数据与训练协议。所有新融合初始函数必须等于同 seed C201 anchor；W 分支总增量保持 `150,048 ±2%`，parameter-fill 必须参与 forward。

| ID | 融合 | 规范性问题 | 新 formal runs |
|---|---|---|---:|
| `F0_FILM` | `z' = z*(1+0.5*tanh(gamma)) + 0.5*tanh(beta)` | 当前 anchor | 0，复用 W |
| `F1_ADD` | `z' = z + 0.5*tanh(beta)` | W 是否只需加性条件修正 | 3 |
| `F2_SCALE` | `z' = z*(1+0.5*tanh(gamma))` | W 是否主要承担尺度重标定 | 3 |
| `F3_CONCAT_RES` | `z' = z + ZeroInitProject([Norm(z), c_W])` | 同位置 clean concat 是否优于仿射条件 | 3 |

`F3_CONCAT_RES` 不得更换 decoder、增加 deep FusionHead、移动融合位置或引入 attention。否则将重现旧 T3 的多因素混杂，不能回答 FiLM 比较。

P1 完成后：

- 若 F0 仍在 Pareto，继续保留 FiLM；
- 若 F1/F2/F3 之一满足第 2 节晋级定义，最多保留两个 Pareto 融合进入 P2；
- 若没有新 arm 晋级，P2 固定继续使用 F0，不因负结果追加 gate/attention 搜索。

P1 当前未开放。

## 7. P2：W 信息来源与 Morlet 参数

P2 使用 P1 冻结的单一融合 anchor；若 P1 保留多个 Pareto，不做“融合 × 小波”全组合，须在 P1 汇总中预先选择一个主 anchor，另一个只作论文背景。

### 7.1 P2A：频带来源

保持现有 `[97,360]` shape、encoder和参数数，只对 W cache 的 scale 轴使用固定、input-only mask：

| ID | 保留频率 | 作用 |
|---|---|---|
| `W0_FULL` | `0.03125–8.00 Hz` | 当前 anchor |
| `W1_RESP` | `0.03125–0.80 Hz` | 直接呼吸低频 |
| `W2_CARRIER` | `(0.80,8.00] Hz` | 高频载波/调制信息 |

`0.80 Hz` 固定归入 respiratory arm，避免重叠。W1/W2 首先各运行一个明确标记为 screening 的 `20260811` seed；只有生命周期合格且未触发第 2 节明显退化线，才补齐另两个 seed。若 screening 配置、代码和 protocol identity 不变且生命周期完整，可在 P0 冻结后规定其是否计入三 seed；不得事后决定。

### 7.2 P2B：Morlet 时间—频率分辨率

只在 P2A 冻结的频带语义上比较三档 effective Q：

| ID | effective Q | 含义 |
|---|---:|---|
| `WQ_LOW` | 当前 Q 的约 `0.7×` | 更强时间定位、较宽频率响应 |
| `WQ_BASE` | 当前 `mu=13.4` 的实际 Q | anchor |
| `WQ_HIGH` | 当前 Q 的约 `1.4×` | 更强频率选择性、更长时间支撑 |

不得直接凭整数指定新的 `mu`。P2 实现前须使用不读取真实 target 的 synthetic/input-only calibration，从实际 Morlet 离散频响测量 Q，并反求、登记和冻结 `mu_low / mu_high`。三臂保持相同目标中心频率、12 voices/octave、0.5 秒池化、encoder、参数预算和边界规则。每个新 transform 使用新的不可覆盖 cache identity。

WQ_LOW/WQ_HIGH 同样允许预注册的 failure-only screening，再由合格 arm 补齐三 seed。

### 7.3 条件候选，不默认开放

- `6 voices/octave` 只作为 cache/计算效率候选；当前 12 voices 为 anchor；
- 不默认运行 24/48 voices CWT，因为 180 秒窗口的低频离散映射已存在重复中心，继续加密可能主要增加冗余与资源；
- analytic generalized Morse 只在 Q 实验显示明确敏感性、且论文问题需要比较母小波形状时开放；必须匹配中心频率与 effective Q；
- 不同时搜索 mother wavelet、Q、voices、频带、pooling 和边界方式。

P2 当前未开放。

## 8. P3：Local BiMamba2 质量—效率深度

固定 P1/P2 选出的融合与 W 参数，只改变 Local BiMamba2 block 数：

| ID | blocks | 预计总参数 | 角色 |
|---|---:|---:|---|
| `D4` | 4 | 约 `902,722` | 轻量候选 |
| `D6` | 6 | `1,219,850` | 当前 anchor |
| `D8` | 8 | 约 `1,536,978` | 增深候选 |

准确参数数须由实现测试登记。D4/D8 先各运行一个 failure-only screening seed，合格后补齐三 seed。

解释边界：自然增减 block 会同时改变深度与参数量，因此本比较首先回答“完整实际架构应选择几层”，不能直接表述为纯深度因果。若 D8 明显优于 D6，须在论文作深度因果主张前增加一个 D6 等参数、active capacity control；若不增加该 control，只能称为 D8 完整 package 收益。若 D4 质量落在预注册近似容差内且吞吐提高至少 `10%`，可进入 efficiency Pareto，即使不在质量轴支配 D6。

P3 当前未开放。

## 9. P4：最终候选与回退消融

P1–P3 完成后，允许依据合理结果重新组织论文主线，并冻结一个最终主候选 `C*` 与最多一个 Pareto 辅候选。不得为了形成线性故事篡改实验发生顺序或声称结果知情假设为事前预注册。

围绕 `C*` 只补齐此前没有覆盖的最小 leave-one-component-out 矩阵：

| 模型 | 目的 |
|---|---|
| C201 | 无 W 条件基线，复用 |
| TF102-W | 当前 W anchor，复用 |
| `C*` | 最终候选 |
| `C*` 恢复 F0 | 最终上下文中的融合贡献 |
| `C*` 恢复 WQ_BASE/full-band | 最终上下文中的小波贡献 |
| `C*` 恢复 D6 | 最终上下文中的深度贡献 |
| 必要 capacity control | 排除容量解释 |

若 `C*` 只改变一个因素，不为论文形式强行补齐无意义的其他回退；已被 P1–P3 完整覆盖的 arm 不重复训练。

P4 当前未开放。

## 10. P5：reused research-test 与最终合理性

用户已明确本阶段的稳健性不要求 test 完全独立。P5 因此允许在模型和 validation 决策冻结后，对以下最小集合使用现有 research-test：

- C201：复用既有结果；
- TF102-W：复用既有结果；
- `C*` 三个 validation-selected checkpoints；
- 最关键的一个回退或 capacity control（仅当其对核心论文解释必要）。

证据名称固定为：

```text
reused research/development evidence; research-test-informed; not untouched independent test
```

若小波 transform identity 改变，必须先构建新的完整 test input-only cache，登记 dataset-row identity、transform/cache hash 与 `target_read=false`，再开放专用 checkpoint allowlist；不得覆盖现有 cache，不得调用或扩展已关闭的旧 research-test 队列来旁路新协议。

最终“合理”要求：目标方向在 validation 与 research-test 上可解释，至少 `2/3` seeds 支持核心方向，无 collapse/degeneracy，资源成本与收益匹配。若 research-test 改变排序，可据此形成新的论文主线或 split-sensitivity 结论；不得删除不利轴或改写访问历史。

P5 当前未开放。

## 11. 预计成本与自适应停止

固定 formal 上限不是承诺全部执行：

- P1：9 个新 runs；
- P2A/P2B：4 个初始 screening runs，晋级 arm 再补 2 seeds；典型共 6–10 runs；
- P3：2 个初始 screening runs，晋级 arm 再补 2 seeds；典型共 4–6 runs；
- P4：只补缺失组合，0–6 runs；
- P5：最终 allowlist evaluation，不重训。

典型新增训练量约 `21–27` runs，最坏约 `33` runs。禁止执行“融合 × 小波 × 深度”全因子矩阵。每阶段必须先冻结 summary 与下一阶段 anchor，失败 arm 不因论文叙事需要追加超参数补救。

## 12. 当前下一步

当前只允许讨论并修订：

1. P1 是否保留 F2 gamma-only；
2. 第 2 节 `3%` 明显退化线是否合适；
3. P2A 的 `0.80 Hz` 频带边界与 P2B Q 比例；
4. screening seed 是否计入正式三 seed；
5. P1–P4 的最大 GPU run 预算。

在用户明确确认上述问题、协议修订完成并建立 P0 candidate lock 前，不实现 variant/config/cache builder，不运行测试、smoke、acceptance、正式训练或 research-test。
