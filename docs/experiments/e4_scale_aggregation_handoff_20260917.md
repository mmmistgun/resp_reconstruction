# E4：尺度聚合结构实验立项交接

日期：2026-09-17。工作目录：`/mnt/disk_code/marques/resp_reconstruction`。

## 1. 新会话任务与授权

用户明确要求重新立项推进 E4，并由新会话承接本文件提供的上下文。第一阶段任务是核对当前代码与冻结 W0，确定一个首轮候选，制定可审查的专项方案、独立实验身份和验收设计。候选尚未最终确定；先交付方案，再进入实现与训练阶段。

本次立项依据是尺度聚合的结构问题，覆盖历史文件中 E4 依赖“E1 基本不敏感”的旧启动条件。E1 的已完成结果仍按原证据解释。用户希望助手自行确定能够合理确定的技术和统计细节，涉及关键科学范围或无法可靠确定的问题时再请用户介入。

沟通默认中文，开头“我在！”。先读取仓库 `AGENTS.md` 与 `/home/marques/.codex/AGENTS.dl.md`。长时间训练、GPU 推理、真实数据 smoke、正式 benchmark 由用户执行；可以运行数分钟内的 synthetic CPU 定向验证。数据范围可按匹配协议使用 train/validation/test；需要 test 时建立本实验专项定义，不因旧 test 已被评价而一概拒绝。解释不预设效果门槛，完成结果后结合效应量、seed 方向和其他属性代价判断。

## 2. 用户确定的动机与科学问题

当前 CWT 分支的数据流：

```text
B×1×97×360
→ 二维卷积和归一化
→ B×96×97×360
→ 沿尺度平均
→ B×96×360
→ 时间对齐到 1800 点
→ temporal mixer、条件投影、FiLM
```

应区分两个问题：

- 通道扩展 1→96 增加特征检测通道，张量增大本身不能证明通道过多。本轮固定 96 通道。
- 尺度压缩 97→1 把每个通道的尺度分布汇总为单一均值，可能压缩有用的尺度位置和分布信息。前置卷积可提前编码部分信息，因此这是待检验假设。

E4 科学问题：**当前尺度聚合是否压缩了对重建有用的尺度分布信息？保留更多频率位置信息，能否改善多属性重建？**

E1 已显示尺度重排敏感性；这不能决定尺度平均是不是最合适的聚合方式。此次 E4 围绕新结构问题立项。

## 3. 候选方向与命名

| 候选 | 主要检验问题 | 命名 |
|---|---|---|
| 所有输入共用的可学习尺度权重 | 是否需要不同的全局尺度重要性 | 可学习尺度加权 / 加权池化 |
| 根据当前输入或时间生成尺度权重 | 是否需要动态选择尺度 | 尺度注意力 |
| 输入相关权重同时使用显式频率坐标 | 频率坐标是否帮助动态尺度选择 | 频率感知尺度注意力 |
| 多个连续频率区域分别聚合，再投影融合 | 是否需要多个尺度摘要缓解单一摘要压缩 | 多区域尺度聚合 |

**优先研究多区域聚合**，因为它更直接对应压缩问题。单组注意力仍输出一个尺度摘要。首轮只确定一个候选；区域数 K、划分边界、融合投影及初始化均待代码核对后制定，不将任何具体 K 或分区方式当作已获用户确认。

应依据冻结实际频率网格确定区域顺序、边界、每区尺度数及权重归一化。明确等尺度索引区间和等 log-frequency 区间是否等价，处理 97 个尺度无法等分、边界尺度与频率覆盖问题。物理频带不能按观察到的 test 效果挑选。

## 4. 已核对的代码事实及关键实现风险

源文件：`/mnt/disk_code/marques/resp_reconstruction/resp_train/crd/tf_v1_model.py`。

- `CwtBranch` 约第 186–207 行；当前核心聚合在第 205 行：`F.silu(self.conv_out(self.depthwise(value))).mean(dim=2)`。
- `conv_in`：Conv2d(1,48,kernel=(5,3),bias=False)，720 参数。
- `norm`：GroupNorm(8,48)，96 个 affine 参数。
- `depthwise`：Conv2d(48,48,kernel=(3,3),groups=48,bias=False)，432 参数。
- `conv_out`：Conv2d(48,96,kernel=1,bias=True)，4704 参数。
- 聚合前上述模块共 **5952 参数**。这不是完整 CWT 分支或全模型参数量。
- 时间对齐为 linear interpolate 至 1800，`align_corners=False`；随后 `_TemporalMixer`、`project_condition`。
- `_ConditionBranch.finalize_parameter_match()` 约第 111–123 行按分支现有参数数自动计算 `_ActiveParameterFill` 的隐藏宽度。这个 fill 是参与 forward 的 pointwise residual（约第 87–100 行），不是闲置参数。

**必须处理的控制变量问题：**如果直接插入新聚合投影再调用原 `finalize_parameter_match()`，它可能自动缩减 active fill 宽度，导致聚合之外的结构也变化。方案须明确如何保持 W0 其余模块、参数和初始化语义一致，以及新参数与容量效应如何报告。不能只以最终总参数相近就声称严格单变量。

新增模块还可能改变 RNG 消耗顺序，需核对原 module 子 seed / 初始化实现，并设计 synthetic 测试，确保声明保持相同初始化的公共模块确实一致。所有新增参数必须参与前向与梯度。

## 5. 首轮实验边界

固定 CWT 97 尺度网格及 cache、96 通道、六层主干、FiLM 位置、refinement/decoder、输入/参考/任务投影、损失、训练预算、优化器、AMP、batch、数据划分及 checkpoint selector。

完整 W0 复用冻结对照；新候选按三个 seed 从头训练。窗口内通道宽度研究另立问题。首轮不同时加入多个聚合候选、改变编码器深度或组成全因子矩阵。

沿用 W0 的训练合同，具体字段需由原配置复核：

- seed：20260811、20260812、20260813。
- 每 seed 80 epochs / 6400 updates；physical/effective batch=128，accumulation=1，保留尾 batch。
- 原生 AdamW、学习率 3e-4→3e-5、step-exact warm-up cosine，BF16 AMP。
- 完整目标 `L_sync + 0.25 L_effort`。
- selector：完整 validation Local RR 严格最小，平局选择更早 epoch。
- 五主指标：Whole RR absolute error、Local RR MAE、envelope trajectory MAE、global envelope modulation error、lag-aware signed PCC。
- train/validation/test 原窗口数为 10141/2675/2310；32/7/8 个 samp_id。样本 seed 分别为 20260610/20260611/20260612。

需报告新聚合器、完整分支及全模型参数，明确 FLOPs/MACs 计算口径，以及内存与时间的预期影响。真实运行耗时和 GPU 峰值显存由用户执行匹配的测量命令。若首轮只能比较单一新结构与 W0，应说明可归因范围；额外容量对照属于需单独列明的训练规模，不静默扩大首轮矩阵。

## 6. 冻结 W0 与协议入口

W0 正式配置入口：`configs/crd_tf_v1/crd_tf102_w_formal.yaml`。完整模型可训练参数为 1,219,850；应与原 run resolved config/checkpoint 核对。

原始 W0 run（路径均位于仓库根下）：

| Seed | W0 selected epoch | Run |
|---|---:|---|
| 20260811 | 13 | `runs/crd_tf_v1/formal/crd_tf102_w/seed_20260811/20260812_210725_400861` |
| 20260812 | 15 | `runs/crd_tf_v1/formal/crd_tf102_w/seed_20260812/20260812_223203_048130` |
| 20260813 | 14 | `runs/crd_tf_v1/formal/crd_tf102_w/seed_20260813/20260812_235740_452006` |

各 run 保存 `config.yaml`、`checkpoint_best_local_rr.pt`、`metrics.csv`、`metrics_summary.csv`、`research_test_metrics.csv` 及其 summary/manifest。

相关协议与可参考实现：

- `docs/experiments/loss_metrics_restart_plan_20260729.md`：冻结 loss/metrics 与数据定义；按任务相关章节读取。
- `docs/experiments/crd_tf_v1_protocol_20260812.md`：W0 原始训练协议。
- `docs/experiments/crd_tf_w_v2_candidate_lock_20260817.json`：W0 候选来源。
- `docs/experiments/e2_w0_effort_ablation_protocol_20260916.md`、`resp_train/paper_evidence/e2_effort_ablation.py`：三 seed 新训练、锁、GPU synthetic、history/selector 验收和配对汇总模式。
- `docs/experiments/e2_w0_effort_test_protocol_20260916.md`、`resp_train/paper_evidence/e2_effort_test.py`：独立 test 附件与来源核对模式。
- `docs/experiments/e3_w0_metric_association_lock_20260916.json`：已有 W0 validation/test CSV 的字节身份及元数据来源。

这些已经完成的 E1/E2/E3 源码、协议、锁和产物作为冻结来源保留。为 E4 新建专项协议、代码入口、实现锁与不可覆盖输出目录；可拟定 `runs/e4_w0_scale_aggregation_v1/`，具体身份在方案中确定。若调整公共模型文件，须明确历史结果仍绑定原 commit/锁，并避免破坏旧入口的严格配置校验；优先独立候选入口和显式科学配置校验。

## 7. E1–E3 已完成证据摘要

### E1：尺度重排

记录：`docs/experiments/e1_w0_scale_results_20260915.md`。

完整 validation、三个 seed、2675 窗口/seed，FULL、尺度循环移动12位、反转、固定置换已完成。固定置换使 global modulation error 配对增加 20.014% ± 17.594%，PCC 绝对下降 0.005919 ± 0.001286，两项均三个 seed 恶化。其他指标及操作存在混合响应。结论为指定尺度重排的属性相关功能敏感性，操作同时涉及尺度槽位、方向/邻接及卷积边界，不能直接证明某种聚合结构最优。

### E2：相对努力损失消融

记录：`docs/experiments/e2_w0_effort_results_20260916.md`。

三个 seed 的同结构/初始化/预算训练、validation/test 及汇总均完成。移除 effort 项后，轨迹误差在 validation/test 分别配对增加 5.268%/5.180%，均为三个 seed 同方向；test global 增加 5.878%、PCC 下降 0.006233，也为三个 seed 同方向。validation global 的方向相反，RR 存在取舍。E4 保持完整目标。

### E3：五指标关联与条件比例

记录：`docs/experiments/e3_w0_metric_association_results_20260917.md`。

完整 W0 冻结 CSV 再分析已完成。Whole/Local RR 均≤1 bpm、另一个属性超过 validation 共同参考 Q75 时，test 轨迹/global/1−PCC 条件比例为 6.82%/17.71%/10.22%；非重叠视图为 9.17%/19.93%/13.35%。总体相关但低 RR 误差仍可伴随其他属性高误差；比例受阈值与窗口视图影响，受试者条件分母稀疏时宏平均需谨慎。E3 用于多属性评价的论证背景，不用它替 E4 预选最有利阈值或新指标。

## 8. 当前仓库与论文状态

截至交接前，实验仓库最近提交：

- `bd779e3`：E2 结果与论文证据收尾。
- `d7beea5`：E3 实现、专项协议及来源锁。
- `76a0e23`：E3 分布结果与论文证据收尾。

论文工作区：`/mnt/disk_code/marques/paper/resp_rec`，有既有未提交编辑。E1–E3 的结果已同步到中文稿、内部底稿、计划、科学证据账本和结果汇总。旧 E4 条件仍可能出现在以下文档，本次方案应明确新立项依据并同步当前可修改状态：

- `paper_rewriting_output/实验项目待补实验清单_20260915.md`
- `paper_rewriting_output/实验补充与证据闭环计划.md`
- `内部底稿_中文稿事实与证据映射.md`

先读论文 `AGENTS.md` 再改论文材料。论文目录可能不在实验会话写入权限内；按实际工具权限操作，可先在 `/tmp` 生成完整候选、原字节备份及 SHA 清单，审查后原子安装。保留既有改动，不将整个论文工作区顺带提交。

## 9. 第一阶段交付清单

1. 核对真实张量流、5952 参数拆分、原频率数组顺序、W0 原配置与来源身份。
2. 提出一个首轮候选，明确 K、分区、池化、融合、初始化、输出 shape 与参数公式，说明选择理由和可归因范围。
3. 明确 active parameter fill 的处理和公共模块同 seed 初始化验证；列出容量变化及计算代价。
4. 制定三个 seed 的完整训练/validation 矩阵、五指标配对统计、checkpoint 与失败/完成验收；是否进入 test 在专项附件中明确。
5. 编写 E4 新专项协议/立项方案与独立输出身份，同步旧 E4 条件到本次科学问题。
6. 给出后续实现和用户执行步骤，标明 synthetic CPU 可直接验证的内容，以及需用户执行的 GPU/训练阶段。

第一阶段完成后向用户交付可审查方案。E4 具体聚合结构、实现与正式实验尚待该阶段确定和推进。
