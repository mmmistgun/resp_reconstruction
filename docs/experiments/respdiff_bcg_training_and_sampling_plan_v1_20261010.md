# RespDiff-BCG：训练诊断、训练方案与最终采样评定计划 v1

计划 ID：`respdiff-bcg-training-and-sampling-v1-20261010`。

当前状态（2026-10-13）：三个训练 seed 已完成并关闭，结果见第 31 节。用户已授权推进 P5 三模型全量 validation，沿用 GPU0 单卡串行；新入口、冻结清单、队列与预算见第 32 节。配置固定为 DDIM6/N64、DDPM50/N16 及 DDIM6/N16 参照，N 尚未稳定的结论不变。独立 research-test 尚未开放。

**最终交付必须回到固定模型上的 DDPM／DDIM、采样步数与独立轨迹数 N 的比较，并完成两类 sampler 所选配置的全量 validation。训练 loss 或小样本过拟合结果不作为最终评定结果。** 独立 research-test 是配置冻结后的单独阶段，按匹配协议和当次授权开放。

## 1. 文件位置与证据入口

本计划在主工作区与实施工作树的 `docs/experiments/` 同步保存。本轮实现位于项目内独立工作树 `/mnt/disk_code/marques/resp_reconstruction/.worktrees/respdiff-training-sampling-v1`，分支 `codex/respdiff-training-sampling-v1`，基于已核对的 RespDiff 提交 `df37b3d4437489694f08a6b7817aa3fe37d1b4e8`。下文新增代码和输出相对路径均相对于该工作树。

2026-10-10 实施期间主工作区更新了项目内存储规则。本轮使用上述新位置；GPU 启动前已确认历史来源的实体位置为 `/mnt/disk_code/marques/resp_reconstruction/.worktrees/respdiff-paper`，原 `/home/marques/.codex/worktrees/respdiff-paper/resp_reconstruction` 路径已映射到该位置。新增运行直接使用项目内路径，历史 checkpoint 按文件 hash 核对。

| 证据 | 用途 |
| --- | --- |
| [双低通三臂协议](/home/marques/.codex/worktrees/respdiff-paper/resp_reconstruction/docs/experiments/respdiff_bcg_baseband_v1_protocol_20261006.md) | 数据、模型、loss 与原训练合同 |
| [双低通三臂结果](/home/marques/.codex/worktrees/respdiff-paper/resp_reconstruction/docs/experiments/respdiff_bcg_baseband_v1_results_20261007.md) | 当前 source-equivalent 基线及尾部问题 |
| [ε 推理 v2 协议](/home/marques/.codex/worktrees/respdiff-paper/resp_reconstruction/docs/experiments/respdiff_bcg_epsilon_inference_v2_protocol_20261007.md) | 噪声身份、DDPM/DDIM 更新、ensemble 与重建口径 |
| [ε 推理 v2 结果](/home/marques/.codex/worktrees/respdiff-paper/resp_reconstruction/docs/experiments/respdiff_bcg_epsilon_inference_v2_results_20261007.md) | 已完成的子集曲线与完整 DDIM6 N16 validation |
| [推理加速记录](/home/marques/.codex/worktrees/respdiff-paper/resp_reconstruction/docs/experiments/respdiff_bcg_sampling_acceleration_v1_20261007.md) | G4 工程证据、耗时口径与成本估算 |

固定历史来源：

- source run：`runs/respdiff_bcg_baseband_v1_from_t630_20261007/source_equivalent_seed20260811_v1/`。
- checkpoint：上述目录的 `final.pt`，train seed=20260811、final update=6400。
- checkpoint SHA256：`08713b693ca709a659eaa29898f689814f32651136a42ce2fe61223cffb8f6f6`。
- 已完成训练、评价、benchmark 和失败产物保留，以既有 receipt 读取。本计划的新实验使用独立 identity。

已知事实：训练覆盖约 3.107 遍；后期平均 noise loss 下降放缓；固定 train probe 的 t=49 单步 x0 RMSE 约 38.20；DDIM6 从 N1 到 N16 的全量 PCC 从 0.5313 提高到约 0.6513，但幅值长尾仍存在。原 probe 只有前 64 chunks，来自前 5 个 parent。以上支持继续诊断，尚未证明预算不足或学习率下降过早。

## 2. 本轮固定合同与研究问题

基础任务沿用 BCG→THO、research-v2 soft-z、parent 双 1 Hz LPF、100→20 Hz、180 s parent、30 s chunk、15 s hop、13 chunks、reflect padding、Hann OLA 与 100 Hz 插值。保持原 train/validation 身份和隔离规则。

起点模型保持六层双向 RNN、H1024、output=128、epsilon prediction、50 步 linear beta `[0.0001, 0.5]`、FP32、Adam 与 source-equivalent loss。训练和推理 batch 上限均为 64，保留尾批。最终预测使用既有 parent 1 Hz LPF，并同时保存保幅 raw/postfiltered 输出。

本计划回答四个问题：

1. 当前实现与 checkpoint 能否稳定完成条件去噪，错误主要出现在何种 t、样本和时间位置？
2. 增加训练更新、改变学习率下降时机，能否改善固定 sampler 下的重建和尾部？
3. 固定训练方案后，DDIM 步数、DDPM 与 N 分别带来多少质量、稳定性和成本变化？
4. 最终应报告哪些 DDPM/DDIM 配置，其结果是否跨训练 seed 和采样噪声保持稳定？

本轮首先改变诊断覆盖、训练预算／学习率，以及推理搜索范围。若证据要求修改网络、loss、扩散 schedule、归一化或 target，应先在本计划中说明科学合同变化及旧结果的可比范围，再建立独立候选；不将这种变更混入预算对照。历史结果不重标、不补写。

## 3. 阶段路线与完成条件

| 阶段 | 核心工作 | 产物与进入下一阶段的条件 |
| --- | --- | --- |
| P0 来源与实现准备 | 核对来源，补齐诊断、训练快照和新 sampler 配置能力 | CPU 合成定向检查通过；真实运行前形成准确命令、配置、预算和输出 identity |
| P1 固定 checkpoint 诊断 | 分 t 误差、条件错配、反推轨迹和尾部定位 | 给出证据支持的故障分类及未决项；实现错误先修复 |
| P2 小样本可拟合性 | 固定训练样本，持续重采噪声与 t | 区分训练样本记忆能力、高 t 估计与完整生成能力 |
| P3 训练预算／学习率对照 | 两个 16000-update 方案与历史基线比较 | 固定训练方案及 final-update selector，选定进入推理评定的模型 |
| P4 sampler／步数／N 搜索 | 同一 checkpoint、同一选择子集与噪声身份 | DDPM/DDIM 曲线、预算边界、每类最终配置与固定低成本参照 |
| P5 最终完整 validation | 三个训练 seed，冻结后的 DDPM/DDIM 配置 | 全量质量、幅值尾部、稳定性与成本表；形成最终 validation 结论 |
| P6 独立 research-test | 匹配协议与当次授权开放后执行 | 原样评价已冻结模型和配置，形成独立测试结果 |

P1/P2 的失败阻止未经解释的扩大训练。若新增训练没有形成更好的候选，使用已核验的历史 source-equivalent 方案进入 P4/P5，仍完成 DDPM/DDIM 评定。实验表现不佳也可以形成完整结论，不以不断扩大预算作为完成条件。

## 4. P0：最小实现准备

现有 `resp_train/respdiff_bcg/sampling.py` 的 `SamplerSpec` 仅开放 DDIM6、N≤16 与 DDPM50、N≤4；旧执行器还固定了来源 checkpoint 和阶段合同。新增范围应使用本计划对应的新配置／入口，保留历史入口的约束。

需要补齐：

- 接受明确来源 identity 的诊断入口，保存逐 t、逐 parent／chunk 误差及 paired condition 对照。
- 训练中的固定 update checkpoint、无扰动 probe、固定验证子集评价；保存完整恢复所需的 RNG、loader 顺序及游标。
- DDIM 的显式 timestep 序列、更大 N 的 nested prefixes、新模型来源加载和确定性配置选择。
- 每阶段的执行预算、进度、失败回执、配置／代码／数据身份与结果文件 manifest。

最小 CPU 合成检查覆盖：epsilon↔x0 恒等式及解析基线；新旧 DDIM6 N1 兼容；DDIM timestep 单调唯一与终点；DDPM posterior 更新及 t=0 不加噪；nested N 与在线均值；condition 错配保留 batch 输入集合；诊断不改变模型／optimizer／RNG；训练恢复顺序和 selector；非有限显式失败。

真实 GPU 运行前执行与新增能力相关的短时合成验收。已通过且未受影响的历史 benchmark 读取已有记录；更改 GPU、batch 或合批策略时，在新条件下验收。加速 G 与轨迹数 N 分开记录，condition BN 始终按原 B64／尾批计算。

## 5. P1：固定 checkpoint 的诊断

### 5.1 固定样本与噪声

从 train 和 validation 各按主体和来源记录分层、确定性抽取 32 个 parent；不按预测质量选样本。实现使用 `samp_id` 表示主体、`target_source_npz` 表示连续来源记录：先按主体轮转，其内按来源记录轮转，再从固定 hash 排序中选择窗口。实际人数、记录数和身份写入 manifest，不把来源记录冒充额外的 session 标注。只使用既有准入规则。

已知异常 row 10382、12226、10322 若不在上述集合，单列为最多三个压力案例，不混入代表性子集的均值。历史 64-chunk probe 继续作为追踪参照。

固定 t=`[0,9,19,29,39,49]`，独立 forward-noise seeds=`[20261010,20261011,20261012]`，条件置换 seeds=`[20261020,20261021,20261022]`。forward noise 按 protocol、seed、split、parent row 和 chunk 身份生成；同一比较保持输入、target、噪声、batch 组成和顺序一致。全部身份在运行前保存。

### 5.2 三组诊断

1. **前向加噪诊断**：记录 epsilon MSE/RMS/P99/P99.9/max、换算后的 x0 RMSE、按 target RMS 归一化的误差，以及逐样本、逐时间位置残差。加入 `epsilon_hat=x_t/sqrt(1-alpha_bar_t)` 的零 x0 解析参照；低于或高于该参照只解释单步估计能力。
2. **条件错配诊断**：在原 batch 内确定性置换 condition，保持 condition 的多重集合、x_t、target、t 和噪声不变，优先让 donor 来自不同 parent；无法满足时明确记录。至少三组固定置换。分别比较分 t 损失与固定 DDIM6 N16、DDPM50 N4 的重建变化。这检验输入配对的作用，不把指标变化换算成“BCG 信息百分比”。
3. **反推轨迹诊断**：在正确 condition 下保存两个固定 sampler 的各步 x_t／x0 统计、首步尾部位置及最终 parent 波形，区分加噪样本上的单步误差与从初始噪声出发的生成误差。

BN 不保存 running stats，子集 batch 与全量 batch 可能产生不同预测。所有组内对照必须保持 batch 身份；子集结果不冒充全量同一行的数值复现。未改代码的条件错配实验使用当前 checkpoint，不更新参数。

输出 `diagnostic_identity.json`、`timestep_metrics.csv`、`condition_pairing_metrics.csv`、`reverse_trace_summary.csv`、固定案例图和 receipt。结论至少区分：实现／数值错误、明显训练拟合问题、条件配对作用弱、采样路径敏感；允许多种现象并存。

## 6. P2：小样本可拟合性

- 从 P1 的 train 集合中按固定身份规则取 8 个 parent，保留全部 104 chunks；B64、尾批 B40 的规则在训练和诊断中固定。
- 新初始化 seed=20260811，source-equivalent loss 与原网络；Adam lr=`1e-4`，预算 3200 updates，期间不降学习率。该任务用于诊断，不进入正式模型竞争。
- updates=`[0,400,800,1600,3200]` 保存状态；每次训练重采 t 与 epsilon。诊断使用上述独立固定噪声，避免只记住固定噪声实现。
- 同时检查分 t 误差，以及训练 parent 上 DDIM6 N16／DDPM50 N4 的完整生成结果。需要同时看到错误幅度、PCC 和尾部，不能只看 loss。

高 t 的 x0 RMSE 在独立噪声上仍不及零 x0 参照，或目标样本上的完整生成仍有严重幅值异常，记为尚未通过基本可拟合性检查；先沿 P1 的定位修复或解释。优于解析参照只是最低检查，不等于充分拟合。训练样本表现良好而 validation 较差时，再检验泛化、对齐与输入配对依赖。

3200 updates 未解决时保留全部曲线和失败／未决状态；追加优化或架构候选须有明确假设和独立配置。P2 不通过时，P4/P5 仍可评价历史基线的实际性能。

## 7. P3：预算与学习率对照

进入条件：来源与实现核验通过；P1/P2 未留下足以使训练结论失效的错误。2026-10-12 用户在知悉 P2/A2 未通过可拟合性检查后明确要求进入 P3，对应继续执行决定见第 20 节。首轮仅使用 train seed=20260811，预先固定以下配置：

| 方案 | updates | LR milestones | 作用 |
| --- | ---: | --- | --- |
| B0 历史基线 | 6400 | 4480、6336 | 读取已有 final checkpoint／结果 |
| B1 仅延长预算 | 16000 | 4480、6336 | 保持原绝对 LR 时间表，检验其后继续更新的收益 |
| B2 延长并推迟降 LR | 16000 | 11200、15840 | 相同长预算下，与 B1 比较下降时机 |

三者初始 lr=`1e-4`、gamma=0.1，其余训练定义一致。B1 后段处于 `1e-6`，没有改善只能说明这个低学习率续训方案无效，不能排除其他长训练方案。B2 相对 B0 同时改变预算和调度，两者的净作用需结合 B1 解释。

B1/B2 使用新 run identity。复用 B0 checkpoint 续训前，必须证明 optimizer、scheduler、全局 RNG、shuffle 顺序和 epoch 内游标可以精确恢复；现有 checkpoint 含 RNG 不等于已经具备完整恢复能力。当前实现明确拒绝将历史 B0 checkpoint 作为精确恢复点，因此 B1/B2 均从头运行；新训练快照支持在相同已定预算内精确恢复。CPU 合成检查已验证新旧训练循环逐值一致；真实设备上的原 6400-update 前缀兼容性仍须由实际记录核对。

快照／probe updates=`[0,1600,3200,4800,6400,9600,12800,16000]`；训练过程保存所有 step 的 loss 分项与 lr。以 P1 固定 train/validation 样本、固定噪声记录分 t 曲线；在 update=`[6400,9600,12800,16000]` 上用固定 DDIM6 N16 记录验证重建曲线。诊断与验证不得消耗后续训练 RNG 或改变 BN 状态。

各方案正式候选只取其预定 final update，不从中途曲线挑 best epoch。最终 B1/B2 还在 P4 的固定选择子集上用 DDIM6 N16 与 DDPM50 N4 两个 anchor 比较，B0 相同 identity 的既有 anchor 直接读取。比较误差、幅值尾部和 sampler 交互，不单凭平均训练 loss 排序。

训练方案选择采用第 9 节的质量／尾部顺序：先对两个 anchor 的 PCC、Whole RR MAE 分别取等权均值，P99 和 top1% 能量分别取两个 anchor 中的较大值，再应用预定容差与成本优先级，选出一个训练方案。该汇总仅用于配置选择；结果表仍分别报告两个 sampler 的真实指标与排序差异。若 B0 也留在最终容差集合内，优先其较少的训练更新。P4 中所有 sampler 因而使用同一训练方案、同一 checkpoint。

选定训练合同后，补齐 seeds=`[20260811,20260812,20260813]` 的 final-update 模型用于 P5。既有其他 target／loss 合同的三 seed checkpoint 不能替代本轮对应 seed。只在一个 seed 上完成时，交付标记为单 seed 阶段结果。

## 8. P4：DDPM／DDIM／N 的参数搜索

### 8.1 公平比较设置

参数搜索首先使用 seed20260811 的候选 checkpoint。复用 ε 推理 v2 的固定 66-parent validation 选择集合及顺序；已知两个异常 parent 的加入方式沿用历史定义。另报去除额外压力案例的规则抽样部分结果，避免把含压力案例的均值当作无偏泛化估计。

固定 B64、保留尾批、noise seed=20261003、同一 initial trajectory 身份和相同重建／后滤波。DDPM 还固定按 reverse step keyed 的噪声。相同 N 在不同 sampler 之间共享初始噪声；各 N 是同一次最大 N 执行的 nested prefixes。

checkpoint 参数／buffer 在前后不变。一个曲线内按轨迹顺序 FP64 累加，保存请求 N、实际完成轨迹数、NFE×N、实际 batched forward 次数、显存及时间。合批 G 的 prefix-ready 时间不能当作该 N 独立运行耗时。

### 8.2 首轮固定矩阵

| sampler | 每轨迹步数 S | N 网格 | 最大逻辑 calls/chunk |
| --- | ---: | --- | ---: |
| DDIM | 6 | 1、2、4、8、16、32 | 192 |
| DDIM | 10 | 1、2、4、8、16 | 160 |
| DDIM | 20 | 1、2、4、8 | 160 |
| DDIM | 50 | 1、2、4 | 200 |
| DDPM | 50 | 1、2、4、8 | 400 |

DDIM eta=0。S6 保留历史序列 `[49,39,29,19,9,0]`；S10/S20 使用含 49 和 0 的等距、四舍五入整数序列，保存完整列表并检查无重复、严格递减；S50 为 49…0。DDIM 最后到干净状态 alpha_bar=1。DDPM50 保留原 posterior variance 和完整反推步序。

这是不同 timestep 网格的实际配置比较，不把所有差异唯一归因于步数。重点同时比较同 N 与近似同 calls/chunk：例如 DDIM6 N8／DDPM50 N1、DDIM6 N16／DDPM50 N2、DDIM50 N4／DDPM50 N4，以及 DDIM10 N16／DDIM20 N8。

### 8.3 N 稳定性和预算扩展

沿用历史四项变化口径：post parent-max P99、top1% 能量均值和 Whole RR MAE 的相对变化分别 <5%，PCC 绝对变化 <0.01。相对变化使用绝对差除以前一值绝对值；零分母且两者同为零记 0，否则无定义并判未通过。每个 S 分别报告相邻 N 曲线，不以一个指标改善替代其余指标。

本计划把连续两次翻倍均通过作为“观察到局部稳定区间”的证据；将该区间起点 N 标为稳定性候选，并保留较大 N 结果。只有一次通过记为初步稳定；到上限仍不满足，明确记为预算受限。所有已评估网格点仍按第 9 节统一选择，稳定性标签不替代质量排序。稳定不代表质量已经足够，也不是 N→∞ 收敛证明。

首轮预算若不足以稳定，扩展优先限定于每类 sampler 的一个非支配配置：DDPM50 可依次 N16、32、64、100；选定 DDIM 可按其当前上限依次翻倍至 64，再到 100。每次扩展先根据已有 profile 给出子集／全量耗时与显存估算，再获得该范围的执行授权。N64→100 不是翻倍，单独报告边际收益。

N100 是可选的源码采样数参照，不能因作者代码使用它就认定本任务必须使用它。若预算不扩展，P5 仍以已评估范围内的配置形成最终结果，同时明确搜索上限和 N 稳定性状态。

## 9. 配置选择、指标与统计口径

五项公共指标及 eligibility 沿用现有实现：lag-aware signed PCC、Whole RR MAE、Local RR MAE、包络轨迹 MAE、全局包络调制误差。保幅诊断包括 parent-max P95/P99/global max、prediction/target std mean/median、top1% 能量及频带能量；同时保存对应 target 统计。

选择优先级在新结果出现前固定如下：

1. 非有限、样本集合改变、身份或实现核验失败的配置判执行失败，不参与排名；有限但幅值很差的配置保留为有效失败表现。
2. 在每类 sampler 内先列出质量、尾部与成本的非支配配置，完整保留 trade-off。
3. 对有效配置，取该类最高 PCC 的 0.01 以内者；再保留其中 Whole RR MAE 不超过这组最优值 5% 者。若 RR 最优值为零，只保留同为零者。
4. 在余下配置中，取 post parent-max P99 不超过这组最优值 5% 者，再取 top1% 能量均值不超过剩余组最优值 5% 者；零值处理同上。
5. 选择余下配置中 calls/chunk 最小者；并列时依次按独立测量采样时间、PCC 较高、S 较小、N 较小决定。独立时间未测量时跳过时间项，完成测量后不据此重新看质量指标调规则。

上述 0.01／5% 是预定工程选择容差，不是显著性阈值或绝对达标线。P3 比较固定 anchor 下的训练方案时使用同样的质量／尾部顺序，成本并列优先训练更新更少者。P4 最终至少给出 DDPM 优选、DDIM 优选，以及 DDIM6 N16 的固定效率参照；若与优选重复，只保留一个执行单元。

选择使用 noise seed=20261003；选定后用额外 sampling seeds=`[20261013,20261014]` 在同一子集检验敏感性，保存全部结果，不选择最有利的噪声 seed。若出现排序反转或尾部明显不稳定，报告该配置的限制，后续扩展按第 8 节执行。

主体汇总保持原 parent-level 均值和有效数，另报训练 seed 间的 mean/std。相关 parent 不视为独立样本；如给出区间或配对差异推断，以数据隔离主体（subject；必要时同时保留 session 层级）进行 cluster bootstrap，并记录抽样单位与 seed。P1 条件错配结果只支持配对依赖分析。

## 10. P5／P6：最终结果与冻结

P4 结束后固定训练配置、final-update selector、全部 checkpoint SHA256、DDPM/DDIM 的 S／完整 timestep／N／eta、采样噪声规则、batch 划分、后处理、指标及代码版本。上述身份写入 `final_evaluation_manifest.json`，执行过程的状态与产物由 receipt 管理。

P5 必须完成：

- seeds20260811/20260812/20260813 的同合同模型，在相同完整 validation 上评价选定 DDPM 和 DDIM；主评价 sampling seed 固定为 20261003。当前来源为 2675 parents，实际执行前核对父身份 hash。
- 固定 DDIM6 N16 参照同时报告；相同模型、数据、sampler、batch、噪声与数值实现身份的既有完整结果可直接引用，否则生成新 identity 的评价。
- 每个配置报告 raw/postfiltered 五项指标、尾部诊断、所有有效数、GPU 与环境、实际采样和总墙钟、峰值显存、S×N。参数选择只使用 P4 子集；完整 validation 的结果用于评定，不再回头改 N 或挑 checkpoint。
- 对最终配置进行短时独立耗时测量，将其与共享 prefix 执行时间分开。历史 2080 Ti 与 4070 Ti SUPER 耗时不能直接混排。
- 最终表一行一个 `(train_seed, checkpoint, sampler, S, N, sampling_seed)`，另给跨训练 seed 汇总与子集噪声敏感性表。明确“最佳已评估配置”“N 是否出现稳定区间”“幅值异常是否仍存在”。

P5 使用了参与配置开发的 validation，只支持开发阶段的最终评定。若需独立泛化结论，进入 P6：在匹配 test 协议和当次授权开放后，对冻结的模型与 sampler 清单原样评价。test 结果不触发剩余矩阵、阈值、N 或 checkpoint 的变更。

交付文件至少包括：`training_diagnostics.md`、`training_comparison.csv`、`sampler_grid.csv`、`selection.json`、`final_evaluation_manifest.json`、`final_validation_per_parent.csv`、`final_validation_summary.csv`、`runtime_profile.csv` 和一份 closeout。原始逐 run 结果与 receipt 保留。

## 11. 资源预算与执行清单

执行前按设备实测填写预算。已有 T630／RTX 2080 Ti、G4 的合成 profile 推算：DDPM50 N8 单 checkpoint 全量 validation 采样约 16.7 小时，N16 约 33.3 小时，N100 约 208 小时；均为外推，不含全部 I/O、指标和机器负载影响。当前新入口固定 G1 condition cache，以减少首轮数值变量；按旧 G1 profile 外推，DDPM50 N8 全量采样约 20.8 小时，不能直接套用 G4 时间。三训练 seed 的开销需相应累计。详见来源加速记录。

P4 每个 checkpoint 的首轮五行最大 N 共享执行共需 1112 逻辑 calls/chunk；不按每个 prefix 重复计费。实际批调用数、GPU 时间和 BN companion／尾批影响以实现与 profile 为准。P3 最多先执行两个单 seed 16000-update 方案；额外两个训练 seed 在训练方案确定后补齐。

| 执行项 | 当前状态 | 执行前要补齐 |
| --- | --- | --- |
| P0 实现与 CPU 合成验证 | 已完成 | native GPU 工程验收；正式执行保存实际源码快照／hash |
| P1 有限真实数据诊断 | 待执行 | parent／batch 清单、GPU 预算、执行授权 |
| P2 小样本诊断 | 待执行 | 八 parent 身份、独立 run、预算、执行授权 |
| P3 两个训练对照 | 待执行 | 恢复能力结论、final selector、预算、执行授权 |
| P4 首轮参数矩阵 | 待执行 | 候选 checkpoint、完整 timestep、子集／噪声身份、预算、执行授权 |
| P4 更大 N 扩展 | 条件触发 | 现有曲线、具体增量 N、成本、该范围授权 |
| P5 三 seed 最终 validation | 待执行 | 冻结配置与 checkpoints、完整父身份、预算、执行授权 |
| P6 research-test | 未开放 | 匹配专项协议、冻结清单、当次 test 授权 |

下一步先执行 native GPU 合成工程验收，再执行固定 P1 诊断。已获授权的同阶段必要实现、定向检查和失败修复持续推进；不因例行步骤重复请求确认。运行期间保持被记录的源码、配置和本计划文件不变，阶段状态由运行目录的 receipt 管理；结束后再更新本计划。

## 12. P0 实现记录与执行命令

统一入口：`scripts/run_respdiff_bcg_study_v1.py`。固定配置：`configs/respdiff_bcg_study_v1/plan.yaml`。新增功能位于 `resp_train/respdiff_bcg/study_*.py`；历史 sampler 的预算限制保留。公共评价辅助函数只新增可选的 `method_prefix` 参数，旧调用的默认行为已通过原合成流程检查。

### 12.1 已执行的 CPU 验证

全部使用小网络（H8、1 层、output4）和 `/tmp` 下的合成／disposable NPZ；CUDA 不可见，新增测试还将 CUDA 初始化、同步、全设备 seed 和 CUDA RNG 调用设为失败。没有读取真实 waveform、加载真实 checkpoint 做推理，或执行 GPU 验收。

- 17 个定向案例通过，24.00 s：参数合同、epsilon→x0 恒等式与零波形基线、旧 sampler 精确兼容、新 timestep 终点、nested N、保 batch 的置换、分层选择、无扰动诊断、与旧训练循环及恢复训练的参数／optimizer／RNG 逐值一致、选择规则、三 seed 冻结清单及授权门。
- 4 个合成流程案例通过，116.72 s：新 P1 诊断、新训练快照、新五行 sampler 矩阵，以及原 ε-v2 流程。
- 新训练 resolved config 同步记录实际 probe noise seeds 与 parent 数量后，配置合同及训练合成流程两个受影响案例再次通过，18.95 s。
- 按项目新增绘图要求加入首行 BCG 输入后，P1 合成流程再次通过，38.46 s；已检查生成 PNG 的输入口径与布局。
- 完成训练后的零更新恢复及中途恢复专项复验通过，12.75 s；模型、optimizer、scheduler、RNG 与 batch 游标保持一致。
- 首轮 CUDA mock 复用了同一函数对象，引发 PyTorch Dynamo 规则冲突；改为独立 mock 后通过。一次合并测试触及 60 s 工具时限，保留临时产物，后续在有界测试进程中完成；这不是科研实验失败或 GPU 结果。

对应命令（先按第 12.2 节设置工作目录和 `PY`；只有源代码变化或新增风险时再运行受影响部分）：

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PY" -m pytest tests/test_respdiff_bcg_study.py \
  -k 'not synthetic_stage_artifacts' -q -p no:cacheprovider

env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PY" -m pytest tests/test_respdiff_bcg_study.py tests/test_respdiff_bcg_sampling.py \
  -k 'synthetic_stage_artifacts or synthetic_workflow_outputs' -q -p no:cacheprovider
```

独立 `synthetic-smoke` 模式将三条新合成流程放入一个新输出目录，并保存总 receipt。新检查点的恢复合同包含实际 batch 顺序、epoch 内 cursor、下一 epoch 的 loader RNG、Python／NumPy／Torch RNG、optimizer 和 scheduler；中间快照标为 `diagnostic_update`，只有计划最后一步是 `final_update`。

### 12.2 当前应执行：GPU 合成验收与 P1

本机初始化命令如下。在 T630 执行时，先同步本次代码，将工作目录和 `PY` 改为该主机项目路径与 `.venv/bin/python`，并把 `SOURCE_RUN` 改为当地原训练目录 `runs/respdiff_bcg_baseband_v1/source_equivalent_seed20260811_v1`。来源目录应包含原完整 receipt、配置、父清单、数据身份及 checkpoint。

```bash
cd /mnt/disk_code/marques/resp_reconstruction/.worktrees/respdiff-training-sampling-v1
PY=/mnt/disk_code/marques/resp_reconstruction/.venv/bin/python
BASELINE_ROOT=/mnt/disk_code/marques/resp_reconstruction/.worktrees/respdiff-paper
SOURCE_RUN="$BASELINE_ROOT/runs/respdiff_bcg_baseband_v1_from_t630_20261007/source_equivalent_seed20260811_v1"
BASELINE_INFERENCE_RUN="$BASELINE_ROOT/runs/respdiff_bcg_epsilon_inference_v2/subset_seed20261003_v1"
STUDY_ROOT=runs/respdiff_bcg_study_v1

env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 "$PY" \
  scripts/run_respdiff_bcg_study_v1.py gpu-check --confirm-gpu-check \
  --device cuda:0 --output "$STUDY_ROOT/gpu_synthetic_b64_v1"

env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 "$PY" \
  scripts/run_respdiff_bcg_study_v1.py diagnose --confirm-data-access \
  --device cuda:0 --source-run "$SOURCE_RUN" \
  --checkpoint-sha256 08713b693ca709a659eaa29898f689814f32651136a42ce2fe61223cffb8f6f6 \
  --output "$STUDY_ROOT/p1_source_s20260811_v2"
```

GPU 工程验收覆盖原尺寸 B64 三次合成训练更新，以及 DDIM6/10/20/50、DDPM50 的 N1/N2 采样。它验证新路径可运行，不提供真实数据性能或 N 上限的耗时结论。P1 使用固定 G1，三组置乱逐 batch 保持 condition 集合，初始噪声及 target 不随置乱变化。

先执行 `gpu-check`，确认其 `receipt.json status=complete` 后，再执行 `diagnose`。每条命令的输出目录必须不存在；失败状态与日志保留，重试使用新版本 identity。`run.log` 自动保存进度及异常，P1 的 `budget.json` 在采样前保存实际 parent／chunk／batch 数和 denoiser 调用预算。

P1 主要产物：

- 根目录：`diagnostic_identity.json`、`budget.json`、各 cohort 的 parent/chunk 清单、`anchor_metrics.csv`、来源快照、数据身份和 receipt。
- `train_representative/forward/` 与 `val_representative/forward/`：`timestep_metrics.csv`、`condition_pairing_metrics.csv`、`position_error_metrics.csv`、`condition_permutations.json`。
- 各 cohort 的 `anchors/`：两个 sampler 的正确／错配 condition 原始与后滤波波形、公共逐 parent 指标、幅值诊断、第一条轨迹的逐步统计和固定案例 PNG。案例图首行显示实际输入 chunks 重建的 BCG，并标注 parent LPF、抗混叠与 20 Hz 采样口径。
- 额外压力案例位于 `val_pressure/`；已经属于代表性集合的案例不重复加入。

验收要求：全部预定 cohort、t、噪声、置乱与 sampler 有记录；关键张量和指标有限；参数／buffer、checkpoint 字节及来源数据身份前后不变。`diagnostic_complete=true` 只代表执行完成；是否支持进入长训练要结合上述结果解释，不能把 complete 当作模型通过科学验收。

### 12.3 后续阶段入口

以下是各阶段对应的命令模板，按前置证据、预算和执行范围逐阶段使用。`P1_RUN`、`P2_RUN`、`ABS_RUN`、`STRETCH_RUN`、`GRID_RUN` 等变量必须指向对应已完成、不可覆盖的运行。

```bash
# P2：先读取 P1 结果，再执行固定八 parent 小样本诊断。
P1_RUN="$STUDY_ROOT/p1_source_s20260811_v2"
P2_RUN="$STUDY_ROOT/p2_overfit_s20260811_v1"
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 "$PY" \
  scripts/run_respdiff_bcg_study_v1.py overfit --confirm-data-access --confirm-training \
  --source-run "$SOURCE_RUN" --diagnostic-run "$P1_RUN" --output "$P2_RUN"

# P3：两个独立长预算对照，均从头初始化。
ABS_RUN="$STUDY_ROOT/p3_long_absolute_s20260811_v1"
STRETCH_RUN="$STUDY_ROOT/p3_long_stretched_s20260811_v1"
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 "$PY" \
  scripts/run_respdiff_bcg_study_v1.py train --confirm-data-access --confirm-training \
  --regime long_absolute --training-seed 20260811 --source-run "$SOURCE_RUN" \
  --diagnostic-run "$P1_RUN" --overfit-run "$P2_RUN" --output "$ABS_RUN"
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 "$PY" \
  scripts/run_respdiff_bcg_study_v1.py train --confirm-data-access --confirm-training \
  --regime long_stretched --training-seed 20260811 --source-run "$SOURCE_RUN" \
  --diagnostic-run "$P1_RUN" --overfit-run "$P2_RUN" --output "$STRETCH_RUN"

# P3 选择：只读取已有三组 anchor；B0 使用已完成 ε-v2 subset 的保存曲线。
TRAIN_SELECTION="$STUDY_ROOT/p3_selection_v1"
"$PY" scripts/run_respdiff_bcg_study_v1.py select-training \
  --baseline-inference-run "$BASELINE_INFERENCE_RUN" \
  --absolute-run "$ABS_RUN" --stretched-run "$STRETCH_RUN" --output "$TRAIN_SELECTION"

# P4：SELECTED_RUN 指向 training_selection.json 确定的首 seed 模型目录。
GRID_RUN="$STUDY_ROOT/p4_grid_s20260811_v1"
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 "$PY" \
  scripts/run_respdiff_bcg_study_v1.py grid --confirm-data-access \
  --source-run "$SELECTED_RUN" --training-selection-run "$TRAIN_SELECTION" --output "$GRID_RUN"
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 "$PY" \
  scripts/run_respdiff_bcg_study_v1.py noise-check --confirm-data-access \
  --source-run "$SELECTED_RUN" --grid-run "$GRID_RUN" \
  --output "$STUDY_ROOT/p4_noise_s20260811_v1"

# 先按选定训练合同补齐另外两个 seed；SEED12_RUN/SEED13_RUN 为对应完整训练目录。
FINAL_MANIFEST="$STUDY_ROOT/p5_manifest_v1"
"$PY" scripts/run_respdiff_bcg_study_v1.py freeze-final --grid-run "$GRID_RUN" \
  --source-runs "$SELECTED_RUN" "$SEED12_RUN" "$SEED13_RUN" --output "$FINAL_MANIFEST"

# P5：三个模型分别执行相同冻结配置；此处展示首 seed，另外两个使用各自新输出目录。
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 "$PY" \
  scripts/run_respdiff_bcg_study_v1.py full-validation --confirm-data-access \
  --source-run "$SELECTED_RUN" --grid-run "$GRID_RUN" --final-manifest-run "$FINAL_MANIFEST" \
  --output "$STUDY_ROOT/p5_full_s20260811_v1"
```

补齐训练 seed 时，复用 `train` 命令，将 `--regime` 设为 `baseline`／`long_absolute`／`long_stretched` 的已选值，`--training-seed` 分别设为 20260812、20260813，并使用新输出目录。历史来源 `SOURCE_RUN` 仍用于核对原数据合同；新训练从相应 seed 初始化。

新格式训练中断后，保持原 regime、seed 和最终预算，使用新输出目录并添加 `--resume-checkpoint <原 run/checkpoints/update_XXXXXX.pt>`；原 `.identity.json` 侧车和数据合同必须同时匹配。若训练已结束、随后 monitor／anchor 评价失败，可从原 `final.pt` 在新目录恢复，优化器更新次数为零，仅补齐本阶段评价；父 checkpoint 的 SHA256 写入新状态与 receipt。旧格式或不同预算的 checkpoint 会被拒绝。

P4 输出完整 `sampler_grid.csv`、`selection.json`、N 曲线图和共享 prefix 耗时；另报规则抽样 64 parent 的结果。`noise-check` 固定使用两组额外噪声，不改变所选参数。P5 将三个 checkpoint 逐一绑定到 `final_evaluation_manifest.json`，每个完整评价额外保存 `independent_runtime.json`：固定首批、warmup=1、repeats=2 的独立 N 耗时和显存，与全量墙钟及共享 prefix 时间分开解释。

当前入口仅开放首轮 N 网格；更大 N 的范围修订及独立 test 仍按前述对应阶段执行。P1 的 GPU 回执与诊断产物是下一次推进的直接输入。

### 12.4 本次后台执行

- 授权：用户 2026-10-10 明确授权 GPU，并要求长任务放后台。
- 设备：物理 GPU 0，NVIDIA GeForce RTX 4070 Ti SUPER，UUID `GPU-ce2f5f29-2322-0e50-3627-4bc7ac242aac`；启动前两卡无 compute 进程。
- 调度：`runs/respdiff_bcg_study_v1/launch_gpu_p1_v1/launch.sh` 顺序执行第 12.2 节两条命令；任一阶段非零退出或工程 receipt 非 complete 时停止后续步骤。
- 日志：启动器的 `console.log` 和各阶段的 `run.log`；源码快照、环境、实际预算和产物 hash 由阶段入口保存。执行期间冻结本工作树的被记录源文件及本计划，状态读取运行目录。
- 本次队列只排入 GPU 工程验收和 P1；P2 及后续依赖 P1 结果解释，未提前排入。

GPU 工程验收结果：`gpu_synthetic_b64_v1/receipt.json` 为 complete，三次原尺寸 B64 合成训练更新及五种 sampler 步数组合的 N1/N2 采样通过，peak allocated 为 7624451072 bytes。工程验收不提供真实数据性能结论。

P1 v1 失败原因及修复：

- `launch_gpu_p1_v1/source_path_audit.json` 记录只读审计：train 10141 parent、validation 2675 parent，row／主体／split／时间边界／顺序／signal key 一致；原索引 SHA256 一致；78 个 NPZ 来源文件的大小及纳秒修改时间全部匹配。
- 历史绝对路径前缀 `/data/disk1/cxh/code/resp_prepare/dataset/` 对应本机 `/mnt/disk_code/marques/resp_prepare/dataset/`。新来源检查在内存中按同名数据集目录及完整相对路径归一化，并继续严格检查索引、逐行身份与文件元数据；不修改历史 CSV 或 receipt。
- 路径歧义、映射碰撞、目录逃逸、样本／顺序／signal key／索引／文件大小／mtime 变化仍拒绝。8 个合成定向案例通过，5.42 s；对应 `tests/test_respdiff_bcg_source_migration.py`。
- 新运行保存 `source_path_mapping.json`，其 hash 随 receipt 归档。该映射沿用原文件身份合同，不把 size／mtime 核验表述为 NPZ 内容 SHA256 比较。
- 重启入口为 `launch_p1_source_s20260811_v2/launch.sh`，日志为同目录 `console.log`；复用已完成 GPU 工程回执，不重跑未受影响的 GPU 数值路径。只有 P1 使用新 identity。

## 13. P1 结果与阶段判断（2026-10-11 只读复核）

来源：实施工作树内 `runs/respdiff_bcg_study_v1/p1_source_s20260811_v2/`。该运行于 2026-10-10 17:59（Asia/Shanghai）完成，receipt 为 complete，阶段墙钟 2886.10 s（48.10 min）。本次复核核对了 receipt 中全部 281 个文件 hash；参数／buffer 的前后摘要一致。完整覆盖 6 个 t、3 个 forward-noise seed、3 个条件置乱、2 个 sampler：代表 train／val 各 29952 条分 t 记录，压力组 2808 条，以及 24 个重建设置。关键数值有限，置乱均无自身配对或同 parent 配对。

代表 train 为 32 parent／32 主体，代表 validation 为 32 parent／7 主体；压力组 row 10322、10382、12226 单独解释。子集依赖其固定 batch，不能作为全量 validation 相同行的数值复现；本轮未执行 P2 或后续阶段。

### 13.1 模型对 BCG 配对有明确依赖

以下为后滤波 parent 指标；置乱列取三次固定置乱的均值。

| 集合 | 配置 | 正确配对 PCC | 置乱 PCC | 正确 Whole RR MAE / bpm | 置乱 Whole RR MAE / bpm |
| --- | --- | ---: | ---: | ---: | ---: |
| train 32 | DDIM6 N16 | 0.67590 | 0.03751 | 0.74507 | 2.74057 |
| train 32 | DDPM50 N4 | 0.74191 | 0.02408 | 0.72349 | 2.53571 |
| validation 32 | DDIM6 N16 | 0.71464 | 0.02508 | 0.36316 | 2.66017 |
| validation 32 | DDPM50 N4 | 0.72946 | 0.02545 | 0.36018 | 2.42661 |

这支持 BCG 的正确配对对生成质量有实质作用。置乱同时改变所配对的局部波形、相位和幅值等信息，不能据此计算“BCG 信息占比”。在 t=49，置乱前后平均 epsilon MSE 比值仅为 train 1.0175、val 0.9797；这一高噪声步的弱区分度与完整生成过程的强条件依赖可以同时成立。

### 13.2 高噪声步在 train 上也明显不足

聚合方式为同长度 chunk × 三个固定噪声实现等权：池化 x0 RMSE = `sqrt(mean(x0_rmse_i**2))`；target RMS 用同样二次聚合。归一化误差为两者之比，数值 1 对应直接输出零 x0 的解析参照。三个噪声实现不作为额外独立主体。

| t | train x0 RMSE | val x0 RMSE | train / target RMS | val / target RMS |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.00373 | 0.00373 | 0.0030 | 0.0038 |
| 9 | 0.17576 | 0.17326 | 0.1420 | 0.1777 |
| 19 | 0.42922 | 0.41398 | 0.3468 | 0.4247 |
| 29 | 0.85314 | 0.70371 | 0.6892 | 0.7219 |
| 39 | 1.48690 | 1.39789 | 1.2012 | 1.4340 |
| 49 | 15.26563 | 17.40368 | 12.3327 | 17.8532 |

train／val 的池化 target RMS 分别为 1.23782／0.97482。t=49 的 epsilon MSE 约为 4.85e-5／6.30e-5，但误差经 epsilon→x0 放大后，两个代表集合的全部 1248 个 chunk×noise 实现均差于零 x0 参照。t=39 仅 train 8.17%、val 3.04% 的实现优于该参照。这是单步估计能力的判断，不能替代最终生成指标。

t=49 的两端各 1 s（共 40/600 个时间点）占残差平方和的 train 21.81%、val 23.86%；边界误差集中，但大部分误差仍在其余位置。首尾位置值得跟踪，不能据此认定全部问题都由边界造成。原历史前 64-chunk probe 与本轮样本和噪声不同，不能用两次 RMSE 的差值声称训练改善。

### 13.3 DDPM 改善典型输出与压力案例，最坏尾部仍存在

| 集合 | 配置 | 后滤波 PCC | Whole RR MAE / bpm | parent 峰值 P99 | 最大绝对幅值 | 最高 1% 时间点能量占比均值 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| train 32 | DDIM6 N16 | 0.67590 | 0.74507 | 29.5065 | 32.4169 | 33.33% |
| train 32 | DDPM50 N4 | 0.74191 | 0.72349 | 10.4607 | 12.4526 | 10.24% |
| validation 32 | DDIM6 N16 | 0.71464 | 0.36316 | 31.8207 | 34.8539 | 26.54% |
| validation 32 | DDPM50 N4 | 0.72946 | 0.36018 | 47.3905 | 63.4503 | 12.57% |
| 压力 3 | DDIM6 N16 | 0.55527 | 0.42615 | 24.2304 | 24.4393 | 58.45% |
| 压力 3 | DDPM50 N4 | 0.81950 | 0.21334 | 1.7949 | 1.7954 | 5.52% |

幅值单位为既有 soft-z 空间；表内使用相同后滤波口径。代表 validation 的 parent 峰值中位数从 DDIM 4.6432 降到 DDPM 2.0501，但 row 13001 的 DDPM 峰值达到 63.4503（THO 目标峰值 2.6464，prediction/target std=6.2976，最高 1% 时间点占能量 94.15%）。因此平均与典型输出改善并不代表极端尾部全部改善。该异常是本次结果的报告对象，不修改预先固定的压力集合或选择规则。

第一条轨迹的压力组最大 x0 在 t=49 为 1884.92；到最终 t=0，DDIM 为 1847.98，DDPM 为 4.51。代表 validation 的最终单轨迹最大值仍分别为 533.05／255.23。轨迹统计位于 chunk 空间，不能与经 N 条均值、OLA 和后滤波后的 parent 峰值混用。两种配置的逻辑预算分别为 96／200 次 denoiser 调用每 chunk，当前对照同时改变 sampler、步数和 N，不识别单个因素的独立效应。

固定案例图为 `val_pressure/anchors/row_12226.png`：首行为实际输入 BCG；DDIM 结果出现约 24.44 的尖峰，DDPM 同一 parent 峰值约 1.77。它说明该压力案例的路径敏感性，不能取代上表全部代表样本的结果。

### 13.4 下一阶段判断

P1 的执行、配对与来源检查通过；本轮证据支持“模型已经使用条件信息，但高 t 单步估计及完整生成尾部尚未解决”。train 也有明显误差，因此不能仅归结为泛化；这仍未证明增加训练预算或推迟学习率下降就能解决。

下一步按第 6 节的 P2，固定 8 个 train parent、104 chunks、3200 updates、不降学习率，观察独立噪声上的高 t 误差和两个固定 sampler 的完整生成是否共同改善。P2 后再决定 P3 的长训练门槛；最终仍按 P4/P5 完成 DDPM／DDIM／N 的统一评定。本次只读取已完成产物、核验及归纳结果，未启动新 GPU 阶段；运行 receipt 保持原样，科研解释保存在本节。

## 14. P2 后台执行（2026-10-11）

用户在阅读 P1 结果及第 6 节下一步说明后明确“可以，开始吧”。本次范围为固定 P2，不自动进入 P3。

- 运行：`runs/respdiff_bcg_study_v1/p2_overfit_s20260811_v1/`。
- 启动器与控制台日志：`runs/respdiff_bcg_study_v1/launch_p2_overfit_s20260811_v1/launch.sh`、同目录 `console.log`。
- 设备：物理 GPU 0，RTX 4070 Ti SUPER，UUID `GPU-ce2f5f29-2322-0e50-3627-4bc7ac242aac`；启动前两张卡均空闲。
- 来源：固定历史 source-equivalent 数据合同；P1 身份绑定 `p1_source_s20260811_v2/diagnostic_identity.json`。模型按 seed=20260811 重新初始化。
- 八个 parent 按已实现的固定规则选择，row ID 为 `[1533,2224,4840,6347,10167,11497,13451,14288]`；共 104 chunks，B64／尾批 B40。
- 训练：3200 updates，Adam lr=1e-4，全程无学习率下降；网络、source-equivalent loss 和数据预处理沿用计划。
- updates 0／400／800／1600／3200 保存状态和固定分 t probe。400／800／1600／3200 同时记录训练八 parent 与 P1 固定 validation 32 parent 的 DDIM6 N16／DDPM50 N4 anchor，供训练内拟合和泛化对照；不据中间结果改变本轮预算。
- 产物包含 resolved config、实际 parent/chunk 清单、完整 history、训练状态快照、独立噪声下的分 t 误差、anchor 指标与波形、源码及环境和 receipt。运行期间冻结被记录源文件与本计划。
- 完成后按第 6 节判断高 t 误差和完整生成是否共同改善，再确定后续阶段。P2 是诊断模型，不进入正式候选竞争。

## 15. P2 结果与关闭记录（2026-10-11）

**执行完成，可拟合性检查未通过。** 来源为 `runs/respdiff_bcg_study_v1/p2_overfit_s20260811_v1/`，于 2026-10-11 17:33:59（Asia/Shanghai）记录阶段完成，墙钟 3250.10 s（54.17 min）。全部 251 个 receipt 产物 hash 复核通过；3200 条 history 更新连续，学习率均为 1e-4。每两次更新恰好是 B64／B40，各遍历一次全部 104 chunks，共 1600 个完整小样本 epoch，累计 166400 个 chunk 呈现。t 与噪声在训练中持续重采样；这些重复呈现不作为独立样本量。

五个 update × train／validation 的分 t probe 共十组，以及四个非零 update × 两个集合 × 两个 sampler 的 16 组 anchor 均完整，关键数值有限。P2 从 seed20260811 重新初始化，实际只用八个 parent；它不是原全量 baseline 的继续训练或公平预算对照，也不进入最终候选竞争。

只读复核产物位于 `runs/respdiff_bcg_study_v1/p2_review_20261011_v1/`：`review.py`、`review.json`、`timestep_summary.csv`、`anchor_curves.csv`、`p2_diagnostic_curves.png`。输入 CSV 与 receipt 的 hash 绑定；原训练产物保持原样。绘图仅平滑训练 loss（后向 100-update 平均），probe 和 anchor 使用原固定节点数值。

### 15.1 平均 loss 下降，高 t 的后期误差反升

| 更新区间 | 平均总 loss | 平均 noise loss | 平均加权 spectral loss |
| --- | ---: | ---: | ---: |
| 1–100 | 1.66379 | 0.29203 | 1.37177 |
| 301–400 | 0.23470 | 0.15771 | 0.07699 |
| 701–800 | 0.12635 | 0.10678 | 0.01956 |
| 1501–1600 | 0.10420 | 0.08212 | 0.02208 |
| 3101–3200 | 0.06975 | 0.05008 | 0.01967 |

与第 13.2 节相同，池化 x0 RMSE 是同长度 chunk × 三个固定噪声实现的平方误差等权聚合。

| update | train t=49 x0 RMSE | val t=49 x0 RMSE | train t=39 x0 RMSE | val t=39 x0 RMSE |
| --- | ---: | ---: | ---: | ---: |
| 0 | 2181.9783 | 2185.6074 | 105.0252 | 105.2002 |
| 400 | 97.8122 | 107.9814 | 4.7763 | 5.2445 |
| 800 | 60.0151 | 63.7780 | 3.5644 | 3.6859 |
| 1600 | 45.1189 | 48.6090 | 3.3075 | 3.3913 |
| 3200 | 54.2278 | 55.1099 | 7.0272 | 7.1198 |

最后 train／val 的 target RMS 为 0.95921／0.97482，t=49 归一化 RMSE 均约 56.53；两个集合在 t=39／49 的全部固定 chunk×noise 实现均差于零 x0 参照。最终 train t=29 归一化 RMSE 也为 1.1989，仅 1.28% 实现优于零 x0 参照；低 t=0／9／19 则分别为 0.0054／0.2205／0.6846。

相对初始模型，高 t 误差有大幅下降；但从 1600 到 3200，train t=49 增加约 20.2%、t=39 增加约 112.5%，与同期随机训练 batch 平均 loss 的下降不同步。曲线还显示加权 spectral loss 多次突增。该标量现象是后续审计线索，尚未记录分 t／分损失的梯度贡献，不能仅凭 loss 大小判定主因。

### 15.2 最终训练样本的生成仍严重失真

以下使用 final update=3200、正确 BCG 配对及既有后滤波口径。幅值比是各 parent 的 prediction STD / target STD 的中位数。

| 集合 | sampler | PCC | Whole RR MAE / bpm | 幅值比中位数 | 最大绝对幅值 |
| --- | --- | ---: | ---: | ---: | ---: |
| train 8 | DDIM6 N16 | 0.60594 | 0.36133 | 8.0717 | 74.2262 |
| train 8 | DDPM50 N4 | 0.22650 | 9.23606 | 12.1138 | 164.0978 |
| validation 32 | DDIM6 N16 | 0.56425 | 0.59408 | 8.7036 | 166.0476 |
| validation 32 | DDPM50 N4 | 0.22896 | 9.39659 | 9.8446 | 233.0621 |

DDIM 的相关性和 RR 可以看起来尚可，但其典型幅值仍扩大约八倍；DDPM 同时存在严重节律与幅值问题。异常出现在实际参与训练的八个 parent，因此 P2 没有形成“训练集已拟合、仅 validation 失败”的证据。

过程节点也未提供通过依据：train 的 DDIM 幅值比中位数在 400／800／1600／3200 步分别为 10.3766／9.4479／4.1767／8.0717；DDPM 为 0.9767／7.2507／0.1188／12.1138，且四个节点的 DDPM PCC 仅 0.1389–0.2265。这表明不能用某一次幅值比接近 1 或某个较好中间指标代替完整可拟合性检查。final-update selector 保持 3200，中间快照仅用于诊断，不重新挑选候选。

### 15.3 对后续计划的影响

P2 执行合同和有限性检查通过，但第 6 节的科学检查未通过；第 7 节的 P3 进入条件尚未满足。当前证据支持先定位高 t 单步误差、训练目标与生成质量的脱节，以及过程中的幅值不稳定；增加全量训练预算的收益仍未得到证明。

建议下一步先作有针对性的训练／采样一致性审计，并为分 t、分损失梯度诊断制定明确修订。学习率、loss、扩散 schedule、网络或采样定义若需改变，应独立声明可比范围和运行 identity，不混入原 P3 的两臂预算对照。本次未执行这些新增实验。

最终 DDPM／DDIM／N 评定目标继续保留。按第 3 节的既有分支，若暂不产生通过诊断的新训练方案，仍可使用已冻结的历史 baseline 进入 P4/P5；P2 模型不作为替代候选。启动哪一后续 GPU 阶段需明确相应执行范围，当前没有排队的后续训练。

## 16. P2-A1 定向审计修订（2026-10-11）

修订 ID：`respdiff-bcg-p2-audit-a1-20261011`。用户在阅读 P2 结果及“分 t 损失／梯度、训练与采样一致性排查”的下一步建议后明确同意。本修订只诊断已有权重，不修改模型、loss、扩散 schedule、split、指标、selector 或后续矩阵；不产生新的训练候选。旧 P1/P2 的结论和产物保持原样。

### 16.1 已完成的代码核对与 CPU 检查

- 当前前向为 `xt=sqrt(alpha_bar)*target+sqrt(1-alpha_bar)*epsilon`，epsilon→x0 符号与 DDPM 反向公式一致；DDIM 最后一步使用干净端点 alpha_bar=1。原作者 DDIM 的全 batch min-max 与当前保幅合同的差别已由历史协议声明，本轮不重新引入该变换。
- source-equivalent 谱项为 `(.01/128)/SNR(t) * mean((|sqrt(SNR)*FFT(target)+FFT(epsilon-epsilon_hat)|-|sqrt(SNR)*FFT(target)|)^2)`，等价于固定权重的直接 x0 FFT 幅度误差。该表达中 epsilon 残差的谱项系数在 t=29／39／49 分别约 0.01116／0.87067／375.80426；这不是实测参数梯度比。
- condition 粗分支使用 `BatchNorm1d(track_running_stats=False)`，训练与评价都依赖当前 batch 统计。RNN dropout=0。相同 batch 的 mode 数值差异与改变 batch 伙伴的效应分别检查。
- 新增 `tests/test_respdiff_bcg_audit.py`：7 项 CPU 合成定向案例通过，8.88 s，覆盖实际训练损失及输出梯度等价、原始梯度几何与零范数状态、50 步独立 posterior 公式及 t=0 无加噪、两种实际 sampler 的逐步串行/缓存一致性、batch 重组的 chunk 身份对齐、checkpoint hash 与 CLI 授权门。所有输入为合成张量，CUDA 调用被 mock 拒绝。

### 16.2 固定模型、输入和预算

| 标签 | 来源 | update | SHA256 |
| --- | --- | ---: | --- |
| baseline_6400 | 原 source-equivalent final.pt | 6400 | `08713b693ca709a659eaa29898f689814f32651136a42ce2fe61223cffb8f6f6` |
| p2_1600 | P2 checkpoints/update_001600.pt | 1600 | `dca9d5e4226f7aaf6cf99559d08cfbb7bb74b9d38acc1de4403cb8320149e4cc` |
| p2_3200 | P2 final.pt | 3200 | `b4f4f3a8af56efd6942e120cec75a488ea3f06688c0ae568d89e3aea1021c47d` |

P2 两个节点用于解释后期变化，baseline 提供固定权重参照，不据审计结果更换 selector。输入为 P2 原八 parent、全部 104 chunks，按原实际清单顺序组成 B64／B40。数据源、信号 key、预处理、索引和文件元数据复用严格核验，模型实际输入保存为本次 `fixed_inputs.npz`。

1. **梯度矩阵**：三个模型 × 两个 batch × t=`[0,9,19,29,39,49]` × 三个原固定 forward-noise seed，共 108 组；每组由生产 `training_loss` 提供真实 noise／weighted spectral 项，通过两次网络反向求导分别取得参数梯度，共 216 次网络 backward。另作 FFT 输出梯度的代数核对，优化器更新次数为零。
2. **数值一致性**：每组比较稳定谱表达与直接 x0→FFT 的 loss 和输出梯度、同 batch 的 train/eval epsilon、普通 eval 与缓存 G1 epsilon，以及独立 posterior x0/xt 系数形式的 DDPM 更新。P2 两个模型同时与原保存的同 batch／t／noise probe MSE 核对。
3. **完整采样路径**：每个模型、每个 batch 上，以同一 trajectory 0／sampling seed20261003，比较串行与缓存 G1 的 DDIM6 N1、DDPM50 N1，共 12 对轨迹；保存每一步 current、epsilon、reverse 的误差统计。此项不计算新的 parent 质量排名或 N 曲线。
4. **batch 上下文**：用 seed20261030 固定重排全部 104 chunks，仍分成 B64／B40。在 t=19／39／49、forward-noise seed20261010 下，保持每个 chunk 的 x／target／noise 身份，比较原顺序与新伙伴的逐 chunk epsilon 及 condition 两分支特征，共 936 条上下文记录。该差异是模型行为测量，不以“必须接近零”作为通过条件。

原始梯度分别报告全参数以及 condition、noise encoder、time embedding、RNN、decoder 组的 L2 范数、加权谱项／noise 范数比、两项夹角和总梯度与 noise 梯度夹角。零范数时显式记录未定义比例／夹角，不生成 NaN 或静默丢弃。参数梯度不是 Adam 的实际更新方向；六个指定 t 的结果也不冒充全部 50 个 t 在实际训练中的完整贡献份额。

### 16.3 预先固定的数值门槛与失败处理

- epsilon：`atol=2e-4, rtol=1e-3`；sampler current/reverse/终态及独立 DDPM posterior：`atol=.02, rtol=1e-3`，沿用既有 FP32 推理一致性门槛。
- loss 标量、谱项输出梯度及保存 probe MSE：`atol=1e-6, rtol=1e-3`。
- 保留全部检查的最大绝对差、RMS 差和最大容差占比。任何非有限输入、输出或梯度立即失败；完整矩阵中的容差超限保存为失败，不能在看到结果后自动放宽门槛。
- 每个模型的参数／buffer 摘要前后一致，checkpoint 字节 hash 不变，参数 `.grad` 不留存；代码、配置、计划、索引和源文件元数据前后复核。
- 工程检查通过只说明这些有限输入与路径的一致性；梯度主导或冲突的观察不能单独证明 P2 失真的因果，也不自动授权改变 loss、学习率或网络。

### 16.4 入口与产物

入口 `scripts/audit_respdiff_bcg_training_sampling_v1.py`，实现 `resp_train/respdiff_bcg/study_audit.py`。本次物理 GPU 0：RTX 4070 Ti SUPER，UUID `GPU-ce2f5f29-2322-0e50-3627-4bc7ac242aac`，启动前空闲。使用既有环境、FP32、原生 B64/B40、G1，后台执行。

```bash
cd /mnt/disk_code/marques/resp_reconstruction/.worktrees/respdiff-training-sampling-v1
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python \
  scripts/audit_respdiff_bcg_training_sampling_v1.py --confirm-data-access --confirm-gpu-audit \
  --device cuda:0 \
  --baseline-run /mnt/disk_code/marques/resp_reconstruction/.worktrees/respdiff-paper/runs/respdiff_bcg_baseband_v1_from_t630_20261007/source_equivalent_seed20260811_v1 \
  --p2-run runs/respdiff_bcg_study_v1/p2_overfit_s20260811_v1 \
  --output runs/respdiff_bcg_study_v1/p2_training_sampling_audit_a1_v1
```

输出包括 `execution.json`、实际 parent/chunk 清单、`fixed_inputs.npz`、`gradient_per_case.csv`、`gradient_per_group.csv`、`consistency_checks.csv`、`batch_context_metrics.csv`、`checkpoint_states.json`、`summary.json`、源码／环境／数据身份及 receipt。控制台日志位于 `runs/respdiff_bcg_study_v1/launch_p2_audit_a1_v1/console.log`。阶段完成前冻结被记录代码与计划；失败重试使用新 identity，旧状态和产物保留。

## 17. P2-A1 审计结果与后续建议（2026-10-11）

来源为 `runs/respdiff_bcg_study_v1/p2_training_sampling_audit_a1_v1/`，于 19:54:53（Asia/Shanghai）完成，receipt 为 complete，阶段墙钟 295.79 s（4.93 min）。全部 18 个产物 hash 复核通过；108 组梯度、648 条参数组记录、1740 项一致性检查、936 条上下文记录完整。优化器更新为零，三个模型的参数／buffer 摘要与 checkpoint 字节前后不变；peak allocated／reserved 为 11359433216／12528386048 bytes。

派生汇总位于 `runs/respdiff_bcg_study_v1/p2_audit_a1_review_20261011_v1/`：`gradient_summary.csv`、`group_summary.csv`、`context_summary.csv`、`consistency_summary.csv`、`audit_findings.png`、`review.py` 和 `review.json`。该汇总只读审计产物，没有再次运行模型。

### 17.1 本轮覆盖的训练／采样路径一致

1740 项检查均通过预定门槛。其中：

| 检查 | 本轮最大绝对误差 |
| --- | ---: |
| 相同 batch 的 train/eval epsilon | 0 |
| 同 batch 普通 eval／缓存 G1 epsilon | 0 |
| 串行／缓存 G1 逐步 current、epsilon、reverse、终态 | 0 |
| noise loss 独立公式 | 0 |
| 直接 x0 FFT／稳定谱表达的加权 loss | 4.47e-8 |
| 两种谱表达对 epsilon 预测的梯度 | 1.14e-8 |
| 独立 posterior 形式／实际 DDPM 更新 | 4.77e-7 |
| 当前输入与原保存 P2 probe 的 epsilon MSE | 1.19e-7 |

这支持本次检查覆盖的公式、mode、缓存路径及 P2 输入关联的一致性。它不等于验证了所有数据和所有可能运行状态，也不把串行与缓存一致性当作生成质量通过。

### 17.2 高 t 的谱梯度很强，且同 t 两项梯度高度同向

下表是每个 checkpoint、每个 t 的六个诊断 case 中位数（B64／B40 × 三个固定噪声），均为 Adam 处理前的全参数原始梯度。

| 模型 | t | 谱项／noise 梯度 L2 比 | 两项梯度 cosine |
| --- | ---: | ---: | ---: |
| baseline6400 | 29 | 0.01163 | 0.93477 |
| baseline6400 | 39 | 0.70123 | 0.93614 |
| baseline6400 | 49 | 395.1808 | 0.99685 |
| P2-1600 | 29 | 0.00942 | 0.90321 |
| P2-1600 | 39 | 0.85932 | 0.99747 |
| P2-1600 | 49 | 378.4278 | 0.99978 |
| P2-3200 | 29 | 0.00972 | 0.96411 |
| P2-3200 | 39 | 0.86126 | 0.99953 |
| P2-3200 | 49 | 374.7711 | 0.99997 |

108 个 case 的全参数同 t cosine 全部为正。当前观察显示，高 t 并不缺少谱项的原始梯度加权；同一个 t 内两项梯度高度同向。未测量不同 t 之间的梯度夹角或 Adam 的实际更新贡献，因此不能把本结果扩展为“整个训练不存在梯度冲突”。

t=49 的绝对梯度进一步区分模型状态：

| 模型 | noise 梯度 L2 | 加权谱梯度 L2 |
| --- | ---: | ---: |
| baseline6400 | 0.00462 | 1.79013 |
| P2-1600 | 0.02804 | 10.58826 |
| P2-3200 | 0.14002 | 52.47535 |

P2 后期加权谱梯度增大约 4.96 倍。decoder 对应值从 4.63284 增至 46.02297（约 9.93 倍），RNN 从 9.34051 增至 25.07256；condition 组从 1.70171 变为 1.53318。baseline 也有很大的谱项／noise 比值，因此“比值大”本身不能单独解释 P2 的失败。快照梯度增大也可能是误差增大的结果；这些值不是学习率导致失稳的因果证明。

### 17.3 batch 伙伴改变会显著改变高 t 单步输出

保持每个 chunk 的输入、target、noise 和模型不变，仅将同一 104 chunks 重新分成 B64／B40。下表为全体 chunk 的 x0 等效预测差异 RMS，不是 target 误差，也不是最终生成指标。

| 模型 | t=19 | t=39 | t=49 |
| --- | ---: | ---: | ---: |
| baseline6400 | 0.03714 | 0.49852 | 8.10547 |
| P2-1600 | 0.03392 | 0.66772 | 13.14522 |
| P2-3200 | 0.04666 | 0.86087 | 12.77409 |

细 condition 分支的差异为零；带 BatchNorm 的加权粗分支差异 RMS 分别约 0.2170／0.2278／0.2143。结果与该分支使用当前 batch 统计的实现一致。使用本次同一 noise seed 的原 batch teacher RMSE 作量级参照，t=49 差异分别约为其 45.56%／29.13%／21.12%；这些比值不是误差的因果归因比例。

相同 batch 的 train/eval 输出相等，与更换 batch 伙伴后输出改变是两件不同的事。后续训练候选的评价 batch 身份应继续固定；本次尚未比较重排伙伴后的完整 ensemble 生成，不能声称 batch 重排已经解决幅值异常。

### 17.4 审计后的建议：单独检验学习率降档（后续授权见第 18 节）

当前审计未提供需要立即修正反向公式或缓存实现的证据。优先检验一个窄假设：在现有高 t 梯度尺度下，P2 后期保持 1e-4 是否妨碍稳定收敛。

建议以 P2 update1600 的同一模型、Adam 状态、数据游标和 RNG 为起点，**只增加一个 lr=1e-5 的独立诊断分支，再执行 1600 次更新，到总 update3200**。原 P2 的恒定 1e-4 路径直接作为已有控制，不重跑。loss、网络、BN、数据、未来 batch／t／noise 身份、sampler／N 和评价 batch 均保持相同。沿用既有独立噪声 probe、两个 anchor 和 P2 可拟合性标准，不根据中途较好结果改 selector。

这需要独立的恢复合同和运行 identity，明确允许的唯一变化为学习率；不能放宽原通用 resume 的合同检查来绕过限制。是否改善高 t RMSE、幅值和重建必须共同判断。该分支仍是小样本诊断，不直接晋级全量正式候选。P3 和最终 sampler/N 评定的既有科学门槛继续保留。

## 18. P2-A2：学习率降档对照修订与执行（2026-10-11）

用户在第 17 节建议后明确要求“启动吧”，授权本节实现、定向验证及 GPU 后台执行。本阶段仅新增一个分支，使用 GPU 0，输出 `runs/respdiff_bcg_study_v1/p2_cooldown_a2_s20260811_v1/`；启动器与控制台日志位于 `runs/respdiff_bcg_study_v1/launch_p2_cooldown_a2_v1/`。旧 P2 及 P2-A1 产物保持冻结。

### 18.1 独立恢复合同

- 起点：P2 `checkpoints/update_001600.pt`，SHA256 `dca9d5e4226f7aaf6cf99559d08cfbb7bb74b9d38acc1de4403cb8320149e4cc`。
- 控制：已有 P2 的恒定 `1e-4` 路径，终点 SHA256 `b4f4f3a8af56efd6942e120cec75a488ea3f06688c0ae568d89e3aea1021c47d`。
- 恢复模型、Adam 动量与步数、scheduler 更新计数、batch 游标、Python/NumPy/CPU/CUDA RNG；唯一数值干预为后续学习率 `1e-4 → 1e-5`，同步 scheduler 对学习率的记录，milestones 仍为空。
- 新增 1600 次更新：总步数 1601–3200，batch 仍为 B64/B40，八个 train parent、104 chunks、loss、BN、预处理均沿用 P2。分支使用 `P2-A2-lr-cooldown-v1` 身份；通用 resume 继续要求原合同完全一致。
- 训练前从起点重放全部 1600 步 batch、t 和 Gaussian noise 抽样，核对旧 history 的逐步 batch，以及旧终点全部 RNG 和完整 batch 游标；任一不匹配立即失败。记录逐步 t/noise SHA256，并在实际训练中逐步核对。
- 总 update3200 评价固定 train8 / val32，保持 P2 的 parent 顺序、B64、六个 t、三个 forward noise seed、DDIM6/N16 与 DDPM50/N4 的 noise seed 和 G1。复用旧 update1600 评价；本分支只生成 update3200 评价，selector 固定 final update。
- 训练末再次验证 RNG、游标、数据、来源 checkpoint 与实现身份。失败输出保留，新重试使用独立 identity。

### 18.2 实现与验收

独立入口：`scripts/run_respdiff_bcg_cooldown_v1.py`；实现：`resp_train/respdiff_bcg/study_cooldown.py`。要求 `--confirm-data-access --confirm-training`，入口固定单臂、单 seed 和 GPU device。新增的训练钩子保留原默认训练数值路径。

CPU synthetic 验证已通过：降档分支与手工恢复 Adam 后按同序列更新的模型、optimizer、scheduler、RNG 逐值一致；错误 batch 顺序被拒绝；降档产物不能被原合同通用 resume 接受；CLI 未授权或 CPU device 被拒绝。两项新定向测试通过（10.62 秒）；原精确恢复测试通过。初轮新测试在完成数值比较后因缺失文件的异常类型断言失败，修正断言后通过，未因此启动真实失败训练。

关键产物：`sequence_preflight.json`、`cooldown_identity.json`、完整 `history.jsonl`、`final.pt` 与侧车、两组 final probes/anchors、`sequence_verification.json`、来源/环境/代码快照、最终 receipt。

### 18.3 判定与范围

与旧 P2 update3200 做相同 parent/batch 的配对比较，重点联合检查高 t 的 x0 RMSE、幅值 STD/尾部及两个 anchor 的波形与 RR 指标。沿用第 6 节 P2 可拟合性要求；降低 loss 或仅改善 PCC 不单独视为通过。该单 seed 小样本分支只检验降档是否改善后期优化，不能直接证明学习率是唯一原因，亦不能消除已观察到的 BN 批依赖。

本次授权不新增 P3 长训练、P4 网格或独立测试集任务。完成后先审核本分支结果。

## 19. P2-A2 配对结果与关闭（2026-10-12 复核）

### 19.1 执行与证据完整性

阶段于 **2026-10-11 20:50:43（Asia/Shanghai）完成**，receipt 为 complete，耗时 1221.33 秒（20.36 分钟）；后台任务 `j-bx7isj` 退出码 0。从既有总 update1600 恢复，新增 update1601–3200，1600 条 history 连续且学习率均为 `1e-5`。对应原控制的逐步 batch 顺序相同，训练前重放与训练中核验覆盖全部 t/noise 抽样，最终全部 RNG 和完整 batch 游标与原控制一致。

本次只读复核验证新运行 **76 项**及旧控制 **251 项**全部 artifact SHA256。训练 parent、两组评价 parent 顺序一致；每个分支 final probes 共 9360 行（train 104 chunks、val 416 chunks × 六个 t × 三个 noise seed），无缺项、重复或非有限值；四组 final anchors 完整。未重跑模型、原始 waveform、训练或评价。

来源：`runs/respdiff_bcg_study_v1/p2_cooldown_a2_s20260811_v1/` 与 `p2_overfit_s20260811_v1/`。新增派生复核产物独立保存在 `runs/respdiff_bcg_study_v1/p2_cooldown_a2_review_20261012_v1/`，包含 `review.py`、`review.json`、分 t 汇总、配对效应、anchor 对照及 PNG；原始运行保持冻结。

### 19.2 单步误差有所改善，高 t 仍未通过零 x0 参照

下表使用相同 chunk/t/noise 身份，跨等长 chunks 与三个固定 noise seed 计算 `sqrt(mean(x0_rmse²))`。

| 集合 | t | 恒定 1e-4 | 后期 1e-5 | 相对变化 |
|---|---:|---:|---:|---:|
| train8 | 39 | 7.0272 | 3.0962 | −55.94% |
| val32 | 39 | 7.1198 | 3.2007 | −55.04% |
| train8 | 49 | 54.2278 | 41.8675 | −22.79% |
| val32 | 49 | 55.1099 | 45.0342 | −18.28% |

降档后 t=49 RMSE 仍分别为目标 RMS 的 **43.65 倍、46.20 倍**；t=39 分别为 **3.23 倍、3.28 倍**。两个高 t 的全部 chunk×noise 组合仍劣于零 x0 参照。t=49 配对 RMSE 改善比例为 train 91.03%、val 89.34%；这些改善不足以达到可拟合性要求。

改善也非覆盖全部 t：t=0/9/19 的 train pooled RMSE 分别上升 80.27%/19.05%/4.49%，val 上升 68.19%/16.80%/3.19%；t=0 的绝对误差仍较小。t=29 两组约降低 15.6%。最后 100 步平均总 loss 为 0.069752→0.072183，noise 项 0.050078→0.067118，weighted spectral 项 0.019674→0.005064，反映不同组成项的权衡，不能将谱项降低等同于完整重建改善。

### 19.3 完整生成：幅值部分收缩，波形相关性全部下降

以下均为沿用原口径的 postfiltered 指标；箭头表示恒定 `1e-4` 控制 → 后期 `1e-5` 分支。PCC 为 lag-aware signed PCC；幅值为 parent 级 prediction STD/target STD 的中位数；RR 为 whole-parent MAE（bpm）。

| 集合与采样配置 | PCC | 幅值比中位数 | RR MAE |
|---|---:|---:|---:|
| train8 DDIM6/N16 | 0.6059 → 0.4990 | 8.07 → 7.04 | 0.361 → 0.695 |
| train8 DDPM50/N4 | 0.2265 → 0.1949 | 12.11 → 5.54 | 9.236 → 10.732 |
| val32 DDIM6/N16 | 0.5642 → 0.4751 | 8.70 → 7.82 | 0.594 → 1.332 |
| val32 DDPM50/N4 | 0.2290 → 0.2191 | 9.84 → 2.59 | 9.397 → 8.556 |

尾部问题依然明显：降档后 DDPM 的 parent 内 top 1% 样本能量占比均值为 train **74.81%**、val **71.30%**（旧控制 67.78%、66.94%）；全局最大绝对幅值仍为 141.07、213.28。DDIM 的全局最大绝对幅值为 train 53.94、val 172.01，后者高于旧控制 166.05。因此，幅值比中位数下降不能证明尖峰问题解除。

### 19.4 阶段结论

**P2-A2 执行与序列一致性验收通过，科学可拟合性门槛未通过，本分支关闭。** 在这个固定起点、seed 和预算下，后期学习率降档能缓解高 t 单步误差与部分幅值膨胀，却不足以恢复正确生成；不能将失败仅归因于后期恒定学习率，也不能由单次降档对照排除其他学习率策略。既有 BN 批依赖证据仍需纳入后续训练设计。

当前结果不支持直接启动 P3 两个 16000-step 长训练臂。保留固定 final-update selector，不将某个指标改善或历史中间节点转为候选选择依据。DDPM/DDIM/N 最终评定仍属于后续固定模型阶段；本轮未产生正式模型或 sampler/N 最终结论。

## 20. P3 继续执行决定与后台队列（2026-10-12）

用户在收到第 19 节结果后明确指令 **“进入P3”**。本节据此开放第 7 节既定的首 seed 两臂训练、final anchors 和三方案选择，覆盖此前暂缓执行的决定。P2/A2 的科学失败状态保持不变；本阶段直接检验完整训练集上的预算与学习率调度，不预设小样本生成问题已经解决。

### 20.1 固定合同与输出身份

| 队列顺序 | 方案 | 初始化与 seed | 更新预算 | LR milestones | 输出目录（相对 study 根目录） |
|---|---|---|---:|---|---|
| 1 | B1 long_absolute | 从头，20260811 | 16000 | 4480、6336 | `p3_long_absolute_s20260811_v1` |
| 2 | B2 long_stretched | 从头，20260811 | 16000 | 11200、15840 | `p3_long_stretched_s20260811_v1` |
| 3 | B0/B1/B2 固定规则选择 | 只读取已保存结果 | 0 | 不适用 | `p3_selection_v1` |

study 根目录为 `runs/respdiff_bcg_study_v1/`。两个训练臂均使用初始 lr=1e-4、gamma=0.1、Adam、B64/保留尾批、原 source-equivalent loss、完整 train 父集合及全部 chunks。P2/A2 checkpoint 不用于初始化。P1 固定 train32/val32 probes、八个快照节点、四个 validation DDIM6/N16 节点，以及 final 66-parent 双 anchor 均按第 7 节执行。预定 selector 仍为 final update16000。

B0 直接引用历史 `respdiff-paper/runs/respdiff_bcg_epsilon_inference_v2/subset_seed20261003_v1`；启动前已核对其 complete/subset 回执和 `ensemble_curve.csv`、`inference_identity.json`、`validation_parents.csv` 的冻结 hash。B0 source checkpoint 仍为第 3 节的 6400-update 模型；不重训或重评历史 B0。

### 20.2 执行、前置验收与边界

- 物理 GPU 0：RTX 4070 Ti SUPER，UUID `GPU-ce2f5f29-2322-0e50-3627-4bc7ac242aac`；启动前两卡均空闲，项目盘可用约 261 GiB。两个训练臂在同一 GPU 顺序执行。
- 启动器：`runs/respdiff_bcg_study_v1/launch_p3_s20260811_v1/launch.sh`；日志为同目录 `console.log` 与各阶段 `run.log`。
- 每臂使用独立 CLI 进程，完成其 probes、validation 曲线、final anchors 后核对 complete receipt；任何非零退出或失败回执停止后续队列，保留失败产物。
- 本轮不修改训练数值实现，复用已通过的 CPU 精确恢复、新旧训练循环兼容性、GPU 原尺寸验收及 P1/P2/A1/A2 的已有工程证据。启动前检查新增 launcher 语法、两份协议一致性、产物身份可用性和 Git diff 格式；不重复已关闭实验。
- 源码与本计划在队列运行期间冻结；实际 resolved config、种子、训练父清单、chunk 顺序、完整 history、checkpoint、环境、源码快照和 receipt 由现有入口保存。原 B0 前 6400 步的真实设备兼容性仍按实际记录核对，不预先声称逐值相同。
- 第三步自动应用第 7/9 节既定训练方案选择规则，保留两个 sampler 的分项指标和失败表现；固定规则选中某一臂不等于已解决幅值问题，仍需结果复核。

本次队列止于 P3 首 seed 训练方案选择。补齐另外两个 seed、P4 sampler/N 网格、P5 全量 validation 和独立测试集不在本次队列中。

## 21. P3 停止与失败产物清理（2026-10-12）

用户先要求将 B2 提交 GPU1 并行，具体调度接管见[双 GPU 修订](respdiff_bcg_p3_parallel_execution_20261012.md)，随后明确要求“帮我把两个任务都停一下，清理失败产物”。已依次停止并行调度任务 `j-ulrij7` 与原任务 `j-qqtm83` 的进程树；训练 PID 260849、271197 和原调度 PID 260846 均已退出，方案选择未启动。

- **B1 long_absolute**：属于用户主动中止，history 最后完整记录为 update1850；`checkpoints/update_001600.pt` 及侧车 hash 已复核，可恢复节点为 update1600。完整未完成运行目录 `p3_long_absolute_s20260811_v1` 保留，未补写 complete receipt，不作为最终训练结果。后续如恢复，应使用独立新 identity，按原 16000-step 合同从该 checkpoint 恢复；1601–1850 之间没有可恢复快照。
- **B2 long_stretched**：于 00:56:04（Asia/Shanghai）在 update0 的 train probe 期间失败，receipt 为 failed，错误为 `CUDA error: unspecified launch failure`，尚无 checkpoint 或训练更新。停止后 `nvidia-smi` 返回 `Unable to determine the device handle for GPU1 ... Unknown Error`；当前证据不足以判断具体驱动或硬件原因，未执行设备重置、重启或新 GPU 实验。
- **清理范围**：仅删除失败的 `p3_long_stretched_s20260811_v1` 运行目录。删除前核验其所有文件与原 receipt 的 hash，并独立保存 receipt、failure、run.log、resolved config、environment、execution、source identity 及源码快照。净释放约 **150.20 MiB**。原 B2 identity 已退役，后续不得复用同名目录重跑。
- **证据位置**：`runs/respdiff_bcg_study_v1/p3_stop_cleanup_20261012_v1/cleanup_receipt.json` 记录授权、停止状态、B1 恢复点、B2 删除前文件大小/hash、保留项与清理结果；`b2_failure_evidence/` 保存诊断材料。原两个 launcher 的日志和接管记录保留。

当前 P3 未完成，B0/B1/B2 训练方案尚未选择；不自动重启训练或后续阶段。已完成的历史运行、来源 checkpoint 和原始数据均未清理。

## 22. P3 双卡重新提交（2026-10-12）

用户在第 21 节停止与清理后明确要求“帮我把这两个任务提交到GPU吧”，重新开放 P3 两个已定训练臂及完成后的固定规则方案选择。沿用用户此前指定的 GPU0/GPU1 并行安排，不新增科学候选、训练 seed 或后续阶段。

### 22.1 前置状态与恢复范围

重新提交前，`nvidia-smi` 能正常读取两张 RTX 4070 Ti SUPER，UUID 与前次一致；两卡各占用 2 MiB、利用率 0%，无 compute 进程。该检查只说明设备当时可访问，不据此推定前次 GPU1 launch failure 的根因已查明。本轮没有执行设备重置、重启或环境安装。

B1 来源 `p3_long_absolute_s20260811_v1/checkpoints/update_001600.pt` 的 SHA256 已核对为 `7f06f96ad46ef498e7d23073fd5a74ea84b2fe6e6c0755dddd66f76f99964eab`。现有训练代码、入口与配置文件均与 B1 保存的 source identity 相同。恢复模型、Adam、scheduler、batch 游标和 RNG；新增执行 updates1601–16000，共 14400 次更新。旧 history 的 1601–1850 不作为恢复状态，其日志保留用于追溯，最终训练历史按旧 1–1600 与新 1601–16000 连接。

### 22.2 固定运行身份

| 训练臂 | 物理 GPU | 起点 | 输出 identity（相对 study 根目录） |
|---|---:|---|---|
| B1 long_absolute | 0 | 原 B1 update1600 完整状态 | `p3_long_absolute_s20260811_resume1600_v2` |
| B2 long_stretched | 1 | 从头初始化，seed20260811 | `p3_long_stretched_s20260811_v2` |
| B0/B1/B2 选择 | CPU | 两个完整 v2 回执及历史 B0 anchors | `p3_selection_v2` |

两臂总预算、LR milestones、数据、loss、网络、快照、probes、validation anchors 与 final-update selector 均沿用第 7 节。GPU 通过各子进程的 `CUDA_VISIBLE_DEVICES=0/1` 隔离，逻辑设备均为 `cuda:0`。已删除的 B2 v1 identity 保持退役，不复用原目录。

首个启动器因路径拼接 TypeError 在创建任何训练进程前退出，记录保留在 `launch_p3_resume_v2/`，训练输出身份尚未使用。修正后以模拟子进程做 CPU 启动器验证，覆盖两条命令、GPU 映射、B1 恢复参数及缺失回执阻止选择，全部通过。

实际启动器与日志位于 `runs/respdiff_bcg_study_v1/launch_p3_resume_v2_retry1/`：`launch.py`、`console.log`、`b1_console.log`、`b2_console.log` 和进程/结束回执。任一臂失败时保留失败产物，另一臂继续其已授权预算；只有两臂 complete、phase=train、end_update=16000 后才执行唯一的 `p3_selection_v2`。全部进程由同一个后台任务管理，便于统一停止。

运行期间冻结源码及本计划；恢复入口继续执行原有合同、数据身份与 checkpoint hash 检查。前次停止的运行、失败诊断及清理证据保持原状。此次队列仍止于首 seed 的 P3 方案选择，不自动进入 P4/P5 或独立测试集。

## 23. P3 首 seed 结果与关闭（2026-10-12）

### 23.1 执行和来源核验

后台任务 `j-v6lsmi` 退出码 0。B1 本次恢复段完成 14400 次更新，总步数达到 16000，耗时 15821.67 秒（4.39 小时）；B2 从头完成 16000 次更新，耗时 22016.31 秒（6.12 小时）。完成回执分别于 05:38、07:21（Asia/Shanghai）被调度器确认；训练方案选择在 07:21:22 完成。墙钟包含并行资源环境及 probes/anchors，不用于独立效率比较。

复核通过 B1 **145 项**、B2 **165 项**、selection **3 项**全部 artifact SHA256。B1 的旧 1–1600 与新 1601–16000 组成连续 history；B2 为连续 1–16000。两臂完整 16000 步的 batch 索引与 t 序列一致；LR 实际切换为 B1 的 4481/6337、B2 的 11201/15841（milestone 更新后改变下一步所用 LR）。每个完整 epoch 覆盖全部 131833 chunks，B64、尾批 B57。

固定 checkpoint 身份：

| 方案 | 运行目录（相对 study 根目录） | final SHA256 |
|---|---|---|
| B0 | 历史 `source_equivalent_seed20260811_v1`，只读引用 | `08713b693ca709a659eaa29898f689814f32651136a42ce2fe61223cffb8f6f6` |
| B1 | `p3_long_absolute_s20260811_resume1600_v2` | `6b89ad505aa12e2f6ff79aefedf03045cdb2e1529677bcda3e08898803da7116` |
| B2 | `p3_long_stretched_s20260811_v2` | `c1e16cfdfba3498a60eafad11e1a72007b5e9969ba921d2eb7b679690d558731` |

选择依据保存在 `p3_selection_v2/training_selection.json` 与 `training_comparison.csv`；重新应用预设规则仍选中 B2。新派生复核位于 `p3_review_20261012_v1/`，含输入/输出 hashes、`prefix_audit.json`、完整指标、分 t 汇总、过程曲线、配对效果与 PNG；本次只读 CSV/JSON 和 CPU checkpoint，没有执行模型、真实 waveform 或新训练。

### 23.2 固定 66-parent 双 anchor 比较

本节使用训练 seed20260811 的 final-update 模型，validation 选择集合及顺序相同（64 个规则抽样 parent 加 2 个既定压力案例）。下表均为 postfiltered 结果；RR 为 whole-parent MAE（bpm），幅值为 parent 级 prediction STD/target STD 中位数，P99 为 parent 最大绝对预测值的 99 分位数。

| 方案 | sampler / N | PCC ↑ | RR ↓ | 幅值比 | P99 ↓ | top1% 能量均值 ↓ |
|---|---|---:|---:|---:|---:|---:|
| B0 6400 | DDIM6 / 16 | 0.6593 | 0.9832 | 0.9516 | 97.6078 | 36.53% |
| B0 6400 | DDPM50 / 4 | 0.7586 | 0.9458 | 0.8294 | 79.2577 | 13.41% |
| B1 16000 | DDIM6 / 16 | 0.7769 | 0.8693 | 0.8626 | 82.4522 | 20.15% |
| B1 16000 | DDPM50 / 4 | 0.7586 | 0.6784 | 0.9092 | 76.3159 | 13.76% |
| B2 16000 | DDIM6 / 16 | 0.8069 | 0.7038 | 0.7914 | 59.5957 | 11.72% |
| B2 16000 | DDPM50 / 4 | 0.7882 | 0.7580 | 0.8446 | 41.5375 | 9.09% |

按预定两个 anchor 等权平均 PCC/RR、较差者 P99/top1% 统计：

| 方案 | 平均 PCC | 平均 RR | 较差 P99 | 较差 top1% 能量 |
|---|---:|---:|---:|---:|
| B0 | 0.708945 | 0.964480 | 97.607761 | 36.53% |
| B1 | 0.767737 | 0.773852 | 82.452202 | 20.15% |
| **B2（选中）** | **0.797565** | **0.730895** | **59.595715** | **11.72%** |

B2 在最高平均 PCC 的 0.01 容差内是唯一方案，因此按原规则选中；没有修改容差或 selector。相对 B1，B2 的 parent 配对 PCC 改善比例为 DDIM 77.27%、DDPM 84.85%。但 DDPM 的 RR 均值由 B1 的 0.6784 上升到 0.7580，不能描述为所有指标一致最优。

在 B2 上，DDIM6/N16 的 PCC 和 RR 好于 DDPM50/N4，后者的尾部更小。这只是两个 anchor 的比较，尚不能代替 P4 的步数与 N 搜索；两者计算预算也不同（96 与 200 calls/chunk）。

### 23.3 高 t 与尾部仍有剩余误差

B2 final 的固定 train32/val32 probes 中，t=39 的 x0 RMSE/target RMS 为 0.9334/0.9869，分别有 65.54%/56.17% chunk×noise 组合优于零 x0 参照；t=49 仍为 **8.3330/11.4823**，全部组合仍劣于零 x0 参照。B0 的对应 t=49 比值为 12.3327/17.8532，说明改善并不等于问题解除。

在 final 66-parent anchors 中，B2 最大异常位于 row9658：target 最大绝对值 9.6068，DDIM 最大绝对预测值 113.4544，DDPM 为 63.5615。幅值中位数低于 1 与少量巨大尖峰可以同时存在，不能用中位数替代尾部检查。

过程中的固定 val32 DDIM6/N16 PCC，B1 在 6400/9600/12800/16000 步为 0.7268/0.7511/0.7742/0.7913；B2 为 0.3985/0.7496/0.7794/0.8113。B2 的早期结果明显较差，最终按完整预定预算得到改善；这些过程指标仅用于诊断，未据此挑选中间 checkpoint。

### 23.4 真实设备前缀与恢复复现边界

第 7 节要求的真实设备前缀核验现已完成，**不能宣称逐值复现历史 B0 或原中断前的全部数值轨迹**：

- B0 与 B1 的前 6400 步 batch、t、LR 完全一致；update6400 的 Python、NumPy、CPU、CUDA RNG 状态也一致。但 loss 从第 2 步起出现差异（首次绝对差 2.38e-6），整个前缀的 loss 绝对差中位数 0.004247、最大 2.966196；6400-step model 的 170 个 tensor 中仅 1 个完全相等，最大元素差 0.082558。
- B1 恢复后与旧运行保留的 1601–1850 这 250 步比较，batch/t/LR 完全一致，首次 loss 差异在 update1602，为 7.45e-9；loss 差值中位数 0.001070、最大 0.064843。
- B1 与 B2 的共同 LR 前缀 1–4480 步同样具有一致 batch/t/LR，loss 在第 4 步首次出现 1.43e-5 差异。

这些记录表明完整状态和随机序列恢复并不保证本环境下 GPU 训练路径逐值相同。本轮未定位具体 kernel 或数值差异来源；不能把 B1 描述为直接接续历史 B0 参数的纯追加训练，也不能把单 seed 的全部差异唯一归因于预算或 LR。CPU 逐值兼容性验收与真实 GPU 的上述证据分别保留。固定协议下的模型实测比较和预设选择结果有效，跨 seed 稳健性仍待后续验证。

### 23.5 关闭与下一阶段

**P3 首 seed 已完成并关闭，选定训练合同为 `long_stretched`，首 seed 模型为 B2 final16000。** 下一步按第 8 节，在这个固定 checkpoint 上进入 P4，比较 DDIM6/10/20/50、DDPM50 及预设 N 网格。其他两个训练 seed 和 P5 全量 validation 仍待对应授权执行；本次复核未启动新 GPU 阶段，也未访问独立测试集。

## 24. P4 首轮网格与采样噪声核验执行（2026-10-12）

用户在 P3 结果与进入 P4 的建议后回复“可以”，授权执行第 8/9 节既定参数搜索与选定后噪声敏感性检查。本次固定使用 B2 `p3_long_stretched_s20260811_v2/final.pt`，SHA256 `c1e16cfdfba3498a60eafad11e1a72007b5e9969ba921d2eb7b679690d558731`，并绑定 `p3_selection_v2/training_selection.json`。启动前已核对模型和选择回执的 hash；不更换模型、不重选训练方案。

### 24.1 运行合同与身份

首轮共 22 个配置：DDIM6 N=1/2/4/8/16/32，DDIM10 N=1/2/4/8/16，DDIM20 N=1/2/4/8，DDIM50 N=1/2/4，DDPM50 N=1/2/4/8。固定 66-parent validation 选择集合与顺序，858 chunks，B64/尾批 B26，共 14 batches；另报既定规则抽样 64-parent 结果。

固定采样 seed20261003、G1、每个 timestep 网格复用最大 N 的 nested prefixes，FP64 按轨迹顺序累加。五个最大 N 的总预算为 1112 calls/chunk，即 15568 次 batched denoiser forward。模型参数和 buffer 在前后核对，配置选择使用第 9 节原有质量/尾部/成本顺序；完整网格前不根据中间表现修改剩余配置。

| 队列步骤 | 输出 identity（相对 study 根目录） | 内容 |
|---|---|---|
| grid | `p4_grid_b2_s20260811_v1` | 完整 22 配置、指标/尾部、运行 profile、N 稳定性及每类 sampler 的选择 |
| noise-check | `p4_noise_b2_s20260811_v1` | 对选定 DDIM、DDPM 及固定 DDIM6/N16 参照（去重），在同一集合上使用 sampling seeds20261013/20261014；保持所选配置，不据此重选 |

noise-check 的精确计算量由已定规则选中的 S/N 决定，最多 19488 次 batched forward（两个额外采样 seed，每个 seed 的 DDIM/DDPM 与固定参照最大预算合计 696 calls/chunk）。两组采样噪声不是新增训练 seed。

### 24.2 调度与验收

使用物理 GPU0（RTX 4070 Ti SUPER，UUID `GPU-ce2f5f29-2322-0e50-3627-4bc7ac242aac`），通过现有统一入口顺序执行 grid 与 noise-check。启动前两卡均空闲，项目盘可用约 254 GiB。启动器与控制台日志为 `runs/respdiff_bcg_study_v1/launch_p4_b2_s20260811_v1/launch.sh`、`console.log`；每阶段保存独立 receipt 和来源身份。

本次沿用已通过 CPU 合成网格验收和 native GPU 采样验收的实现，未修改采样、指标、checkpoint 或选择数值逻辑。启动前检查 launcher 语法、新 identity 可用性和协议同步；运行期间冻结源码与本计划。grid 非零退出或非 complete 回执时停止后续 noise-check，保留失败输出。

阶段复核要求：22 个主配置完整、raw/postfiltered 指标及有效数可核对、父集合与 batch 身份不变、checkpoint 参数/字节不变；选择按原规则重算一致；额外两个 sampling seed 的结果完整报告。共享 prefix 时间不作为各 N 的独立运行时间，独立耗时测量按 P5 原计划执行。

本次队列止于 P4 网格和固定噪声核验。更大 N、其余两个训练 seed、P5 全量 validation 与独立测试集不自动启动。

## 25. P4 首轮结果与关闭（2026-10-12）

### 25.1 完成状态及完整性

后台任务 `j-a7e0ng` 退出码 0。grid 于 08:53:54（Asia/Shanghai）完成，耗时 2431.51 秒（40.53 分钟）；noise-check 于 09:42:37 完成，耗时 2917.14 秒（48.62 分钟）。固定 22 配置及三个所选/参照配置 × 两个额外 sampling seed 完整结束，模型参数/buffer 的前后摘要相同。

复核验证 grid **262 项**、noise-check **87 项**全部 artifact SHA256；22 个主配置、6 个额外噪声配置无缺项，五项公共指标及核心尾部统计均有限；固定 66-parent 身份和顺序一致，规则抽样报告均为 64 parents。重新应用原选择规则得到相同配置，未按额外噪声结果重选。

原始结果为 `p4_grid_b2_s20260811_v1/` 与 `p4_noise_b2_s20260811_v1/`；独立派生复核位于 `p4_review_20261012_v1/`，含完整网格简表、三组噪声结果、范围统计、极端案例、比较 PNG、输入/输出 hashes 和预算估算。本次复核只读取保存的 CSV/JSON，没有执行模型或新 GPU 任务。

### 25.2 主采样 seed20261003 的选择结果

以下均为固定 B2、训练 seed20260811、66-parent validation 选择集合上的 postfiltered 结果。P99 指 parent 最大绝对预测值的 99 分位数。

| 角色 | 配置 | calls/chunk | PCC ↑ | Whole RR MAE ↓ | P99 ↓ | top1% 能量均值 ↓ |
|---|---|---:|---:|---:|---:|---:|
| DDIM 优选 | **DDIM6/N32** | 192 | **0.819926** | 0.700927 bpm | 58.295759 | 9.10% |
| DDPM 优选 | **DDPM50/N8** | 400 | 0.803554 | **0.478145 bpm** | **37.874733** | 9.28% |
| 固定参照 | DDIM6/N16 | 96 | 0.806950 | 0.703831 bpm | 59.595715 | 11.72% |

本轮选择结果是**已评估范围内的优选**，不是所有 S/N 的全局最优或独立泛化结果。DDIM6/N32 在主 seed 下有更高 PCC、约一半逻辑 denoiser 调用；DDPM50/N8 的主 seed RR 和 P99 更小。后者的这种优势在额外噪声中并不稳定，见下一节。

增大 DDIM 步数没有形成普遍收益：近似同预算的 DDIM6/N32（192 calls/chunk）与 DDIM50/N4（200 calls/chunk）相比，PCC 为 0.8199 vs 0.7286，RR 为 0.7009 vs 0.9059，P99 为 58.30 vs 152.92。这只支持本模型、本集合及已评估配置的比较，不能推广为步数越少越好。

### 25.3 采样噪声敏感性

| 配置 | sampling seed | PCC | RR MAE（bpm） | P99 |
|---|---:|---:|---:|---:|
| DDIM6/N32 | 20261003 | 0.819926 | 0.700927 | 58.295759 |
| DDIM6/N32 | 20261013 | 0.821249 | 0.695634 | 53.858991 |
| DDIM6/N32 | 20261014 | 0.812923 | 0.641704 | 68.849470 |
| DDPM50/N8 | 20261003 | 0.803554 | 0.478145 | 37.874733 |
| DDPM50/N8 | 20261013 | 0.801815 | 1.036560 | 125.857156 |
| DDPM50/N8 | 20261014 | 0.806499 | 0.400874 | 18.920190 |

DDPM 的 PCC 相对平稳，但 RR 范围为 **0.4009–1.0366 bpm**，P99 为 **18.92–125.86**，包含相对 DDIM 的排序反转。seed20261013 时最大绝对预测值为 163.74（row12883，对应 target 最大值 10.43）。DDIM6/N32 的 RR 范围为 0.6417–0.7009、P99 为 53.86–68.85，但仍存在明显尖峰，不能称为尾部达标。

该现象不只是额外两个压力 parent 导致：规则抽样 64-parent 的 DDPM RR 为 0.4809/1.0649/0.4093，P99 为 38.12/127.02/19.13。三组噪声上的比较保持完整，未挑选有利噪声 seed。

固定参照 DDIM6/N16 在三组噪声的 PCC 为 0.806950/0.817988/0.808196，RR 为 0.703831/0.702884/0.639071；增加到 N32 对 PCC 的改善为 0.012976/0.003261/0.004727，边际收益具有采样噪声依赖。

### 25.4 N 尚未稳定

五条 S/sampler 曲线的 **17 次相邻 N 翻倍比较均未同时通过四项稳定判据**，更未形成连续两次通过的稳定区间，全部保留 `budget_limited` 状态。

- DDIM6 从 N16 到 N32：PCC 增加 0.012976，超过 0.01 变化阈值；top1% 能量变化 22.38%，超过 5%。RR 和 P99 的变化较小不能替代其余条件。
- DDPM50 从 N4 到 N8：PCC 增加 0.015374，RR 变化 36.92%，P99 变化 8.82%，均未达到局部稳定要求。

两类优选都位于当前 N 上限。因此本轮结果不能用于宣称 ensemble 已收敛。有限但较差的配置和所有中间 N 均保留，没有因其异常幅值缩小样本集合或删除结果。

### 25.5 下一轮建议与计算预算（未授权执行）

建议按第 8.3 节，只将当前两类优选扩展一档：**DDIM6/N64、DDPM50/N16**，固定同一 B2 checkpoint、parent/batch/noise 身份和指标口径，使用独立修订与输出 identity。需要先扩展现有仅允许首轮上限的运行合同和入口；本次结果复核没有实施或启动该扩展。一次新增翻倍最多补充一次稳定比较，本身不能保证满足连续两次通过的判据。

基于本轮 G1 最大 N 的完整采样 sweep，DDIM6/N32 用时 389.21 秒、DDPM50/N8 用时 814.69 秒。按 calls 与 batch 数线性估算：

| 扩展配置 | 同一 66-parent 子集，单 sampling seed | 2675-parent 全量，单模型/单 sampling seed |
|---|---:|---:|
| DDIM6/N64 | 约 13.0 分钟 | 约 8.4 GPU 小时 |
| DDPM50/N16 | 约 27.2 分钟 | 约 17.6 GPU 小时 |
| 两者合计 | 约 **40.1 分钟** | 约 **26.0 GPU 小时** |

若同一子集继续检查三组 sampling seed，两配置合计约 2.0 GPU 小时；以上均不含初始化、指标计算和 I/O，亦不含固定参照或其他训练 seed。全量估算按 544 batches 相对当前 14 batches 放大。这是预算估算，不是独立 N 性能 benchmark，不能将本轮共享 prefix 的中间 ready 时间当作独立运行耗时。

本轮入口没有落盘采样峰值显存，不能给出实测 GiB 值。G1 逐轨迹执行下，增加 N 预计不会使峰值显存按 N 线性增长；如后续授权扩展，应在新运行中补充 B64/G1 的峰值显存记录并确认资源预算，不能补写旧产物的未测数据。P5 的独立时间/显存测量仍按原计划保留。

P4 首轮至此关闭；N 扩展、其他两个训练 seed、P5 全量 validation 及独立测试集均未启动。

## 26. P4-N2：双 GPU 扩展一档 N（2026-10-12）

用户明确授权：“可以，扩到 DDIM6/N64 和 DDPM50/N16，分别放到两张 GPU 上后台并行跑。”本节新增独立扩展合同 `P4-N2-ddim64-ddpm16-v1`，仅开放以下两个点，并在两臂完成后只读合并原 22 点与新增 2 点，按同一质量/尾部/调用成本顺序更新比较与选择。

| 臂 | 物理 GPU | 新增配置 | 输出 identity（相对 study 根目录） |
|---|---:|---|---|
| DDIM | 0 | DDIM6/N64 | `p4_n2_ddim6_n64_s20260811_v1` |
| DDPM | 1 | DDPM50/N16 | `p4_n2_ddpm50_n16_s20260811_v1` |
| 合并选择 | CPU | 原 22 点＋新 2 点 | `p4_n2_selection_v1` |

### 26.1 固定身份和比较合同

- 模型：B2 `p3_long_stretched_s20260811_v2/final.pt`，SHA256 `c1e16cfdfba3498a60eafad11e1a72007b5e9969ba921d2eb7b679690d558731`；训练 seed20260811。
- 来源：`p4_grid_b2_s20260811_v1` 的 complete 回执、选择及父身份。所有已有数值源码/配置与其 source identity 逐文件 hash 相同；仅新增独立扩展模块/CLI，原 `StudySamplerSpec` 首轮预算上限保持不变。
- 数据：同一 66-parent validation 集合及顺序、858 chunks、B64/尾批 B26、G1、sampling seed20261003、原初始/反向噪声键和 FP64 轨迹均值。DDIM eta=0、timestep `[49,39,29,19,9,0]`；DDPM 保持完整 50 步。
- 最大 N 内自然得到旧前缀：DDIM 的 N32、DDPM 的 N8。仅为新扩展的数值核验保存其前缀输出，与首轮对应 postfiltered waveform 做逐样本比较，预先固定 `atol=1e-5, rtol=1e-4`；记录最大差、RMSE、是否逐值相等。超出容差即失败，不事后放宽阈值，不改写旧结果。
- 只对新最大 N 计算并保存完整 raw/postfiltered 指标和规则抽样 64-parent 汇总。旧前缀核验不额外增加 denoiser 调用。两个新点与原 22 点的合并必须完整，不能按某一臂的提前结果修改另一臂或原选择规则。
- N 稳定性继续按原 PCC/三项相对变化判据。首轮较小 N 的翻倍比较均未通过，新增一次通过最多记为初步稳定，不构成连续两次通过。

### 26.2 显存和计算预算

DDIM6/N64 为 384 calls/chunk、5376 次 batched forward；DDPM50/N16 为 800 calls/chunk、11200 次 batched forward。两臂分别使用两张 RTX 4070 Ti SUPER，启动前均空闲，项目盘可用约 253 GiB。参考本轮 profile 的纯采样估算分别为 13.0/27.2 分钟，并行时墙钟受较慢臂和初始化/指标开销影响；这不是完成时限或独立耗时 benchmark。

每臂在采样前重置 CUDA peak memory 计数，保存本次完整 G1/B64 采样 sweep 的 `max_memory_allocated`、`max_memory_reserved`、设备名、总显存与环境映射。该记录用于本次真实峰值，不回填首轮未测值；运行失败明确记录并保留，不缩小 batch 或改变模型来绕过错误。

### 26.3 实现、验收与后台执行

新增入口 `scripts/run_respdiff_bcg_extension_v1.py`，实现 `resp_train/respdiff_bcg/study_extension.py`。定向 CPU synthetic 验证 **7 项通过**（基础 6 项 26.75 秒，汇总身份检查 1 项 4.02 秒）：两类 sampler 的扩展旧前缀与原预算逐值一致、原入口继续拒绝超限、未授权预算/采样 seed 被拒绝、保存 waveform 的前缀偏移显式失败、完整 24 点选择合同、CLI 数据授权与 GPU device 守卫，以及两臂汇总绑定原网格并拒绝被改动的采样 seed。并行启动器另经模拟子进程验证两条命令、GPU 映射与合并入口。无需重新运行已关闭 GPU 验收。

调度与日志位于 `runs/respdiff_bcg_study_v1/launch_p4_n2_parallel_v1/`。DDIM 子进程使用 `CUDA_VISIBLE_DEVICES=0`，DDPM 使用 `CUDA_VISIBLE_DEVICES=1`，各自逻辑设备均为 `cuda:0`。运行保存新的 resolved config、execution、环境、源码快照、父/chunk 身份、采样 profile、显存、前缀核验、指标和 receipt；源码与本计划在运行期间冻结。

两臂独立运行；任一失败时另一臂继续已授权预算，合并选择停止。只有两臂 complete、旧前缀核验通过、模型/数据身份一致后，才执行 24 点合并选择。历史运行和失败记录不覆盖；新增运行 identity 只使用一次。

本次授权仅执行主 sampling seed20261003 的两个新增点和 CPU 汇总。额外噪声核验、更大 N、其余训练 seed、P5 全量 validation 与独立测试集不排入队列。

## 27. P4-N2 结果与关闭（2026-10-12）

### 27.1 完成状态、前缀与显存

后台任务 `j-4d8uj8` 退出码 0，两臂和 24 点合并均为 complete。DDIM6/N64 总墙钟 875.10 秒（14.59 分钟），DDPM50/N16 为 3414.85 秒（56.91 分钟）；DDPM 于 16:26:06（Asia/Shanghai）结束。复核验证两臂各 26 项、合并 4 项，共 **56 项** artifact SHA256，模型参数/buffer 摘要未变。

DDIM 的 N32、DDPM 的 N8 前缀，与首轮保存的对应 postfiltered waveform **逐值完全一致**，两者最大绝对差与 RMSE 均为 0。此次两张 GPU 的新增 N 比较通过了前缀数值验收。

两臂实测 CUDA 峰值 allocated 均为 **2286900736 bytes（2.1298 GiB）**，峰值 reserved 均为 **2426404864 bytes（2.2598 GiB）**。这是完整 G1/B64 最大 N 采样 sweep 的计数，覆盖模型与采样分配，不等同于整卡所有进程的总占用。

原始产物：`p4_n2_ddim6_n64_s20260811_v1/`、`p4_n2_ddpm50_n16_s20260811_v1/`、`p4_n2_selection_v1/`。独立派生复核为 `p4_n2_review_20261012_v1/`，含输入/输出 hashes、五个相关点的完整公共指标/尾部比较、parent 配对效果、耗时/显存表及 PNG。只读复核未执行模型或新 GPU 任务。

### 27.2 主 sampling seed20261003 的新增结果

固定 B2 checkpoint、相同 66-parent validation 选择集合；下表为 postfiltered 指标。

| 配置 | calls/chunk | PCC ↑ | Whole RR MAE ↓ | Local RR MAE ↓ | parent-max P99 ↓ | top1% 能量均值 ↓ |
|---|---:|---:|---:|---:|---:|---:|
| DDIM6/N32 | 192 | 0.819926 | 0.700927 | 1.133345 | 58.295759 | 9.0983% |
| **DDIM6/N64** | 384 | **0.826575** | **0.684889** | **1.125606** | **56.918615** | **7.8399%** |
| DDPM50/N8 | 400 | 0.803554 | **0.478145** | **1.034226** | **37.874733** | 9.2753% |
| **DDPM50/N16** | 800 | **0.817811** | 0.714112 | 1.134487 | 47.380904 | **8.8843%** |

DDIM N32→64 的 PCC 增加 0.006649，RR 减少 0.016038 bpm，P99 降低 2.36%，top1% 能量降低 13.83%。66 个 parent 中，PCC 改善比例为 74.24%，RR 改善比例为 62.12%。但全局最大绝对幅值从 75.36 升至 78.15，整体统计改善仍不代表每个尾部案例都改善。

DDPM N8→16 的 PCC 增加 0.014256，但 RR 增加 **0.235967 bpm（49.35%）**，P99 增加 **25.10%**，全局最大绝对幅值从 45.76 升至 80.07。PCC 在 92.42% parent 上改善；RR 在 51.52% parent 上改善，但均值被其他 parent 的退步拉高，不能描述为 N 增加后所有样本或所有指标都变好。

规则抽样 64-parent 上，新 DDIM6/N64 的 PCC/RR 为 0.824216/0.702810，新 DDPM50/N16 为 0.814941/0.729171；当前两者的主要排序不依赖额外两个压力 parent。两配置逻辑计算量为 384 vs 800 calls/chunk，仍不是相同预算。

### 27.3 24 点选择及稳定性

重新应用原质量/尾部/调用成本顺序，选择 DDIM6/N64、DDPM50/N16，保留 DDIM6/N16 固定参照。选择规则未变：DDPM N8 的 PCC 比 N16 低 0.014256，已超出最高 PCC 的 0.01 容差，因此其较小 RR 不能在后续筛选中使其胜出；这说明“按规则选中”不等于所有指标全面占优。DDIM N32 与 N64 均通过前面的 PCC/RR/P99 容差，N64 的 top1% 能量优势决定了后续筛选。

**两个新翻倍比较均未通过稳定判据：**DDIM 的 PCC/RR/P99 变化满足对应阈值，但 top1% 能量仍变化 13.83%；DDPM 的 PCC 变化 0.014256、RR 变化 49.35%、P99 变化 25.10% 均超阈值。不得标记为局部稳定或 N 已收敛。

新增 N64/N16 仅评价 sampling seed20261003。首轮 N32/N8 的两组额外噪声结果不自动构成新配置的敏感性证据，也不能用此次单 seed 结果认定先前 DDPM 的噪声不稳定已经消失。

### 27.4 本轮耗时与后续预算

DDIM6/N64 完整采样 sweep 为 778.68 秒（12.98 分钟），与前轮线性估算接近。DDPM50/N16 为 3318.69 秒（55.31 分钟），明显高于前轮 27.2 分钟估算；其同一个 N8 前缀本次为 1652.77 秒，是前轮 GPU0 的 814.69 秒的 2.03 倍。这反映执行速度发生变化，原因尚未定位，不能将该倍数归因于新增 N，也不能据跨设备/并行环境的墙钟直接宣称 sampler 独立加速比。

若下一步补充两个固定采样 seed（20261013、20261014），仅对 N64/N16 产生四项新评价，按本轮 profile 估算纯采样分别约 26 分钟（DDIM 两 seed）和 111 分钟（DDPM 两 seed）；双卡并行墙钟主要受 DDPM 臂限制，另加初始化、核验与指标开销。这个建议尚未实施或启动。

### 27.5 关闭与建议

**P4-N2 已完成并关闭。** 当前证据支持将 DDIM6/N64 与 DDPM50/N16 保留为扩展范围内的规则优选，同时完整保留旧 N 的性能权衡与尚未稳定的状态。建议先给两个新增配置补齐 sampling seeds20261013/20261014 的敏感性核验，再决定配置冻结、进一步 N 扩展或全量评价。其他训练 seed、P5 与独立测试集均未启动。

## 28. P4-N2 额外采样噪声核验（2026-10-12）

用户明确授权给 DDIM6/N64 和 DDPM50/N16 补齐两组采样噪声，并继续分别放到两张 GPU 后台并行。本轮独立修订为 `P4-N2-noise-check-v1`，只新增 **4 项评价**：每个配置各执行 sampling seed20261013、20261014；主 seed20261003 和 DDIM6/N16 参照的既有结果只读复用。

### 28.1 固定合同与前缀来源

模型为 B2 final16000，SHA256 `c1e16cfdfba3498a60eafad11e1a72007b5e9969ba921d2eb7b679690d558731`。配置绑定 `p4_n2_selection_v1/selection.json`，保持 DDIM6/N64、DDPM50/N16；两个额外 seed 的结果不触发 S/N 或训练模型重选。

沿用同一 66-parent validation 顺序、858 chunks、B64/尾批 B26、G1、原 timestep、postfilter 和逐轨迹 FP64 均值。每个 sampler 的两个 sampling seed 在同一 GPU 进程中按 20261013、20261014 的固定顺序执行。

每个最大 N 自然产生的旧 N 前缀，分别与 `p4_noise_b2_s20260811_v1` 中**同采样 seed** 的 DDIM6/N32、DDPM50/N8 保存 waveform 核对，容差仍为 `atol=1e-5, rtol=1e-4`。主 seed 的 waveform 不充当额外 seed 参照。前缀不匹配则显式失败，保留记录；不放宽阈值或改变剩余矩阵。

每次采样前重置 CUDA peak 计数，保存该 seed 的 allocated/reserved 峰值和设备信息。源码、模型、父集合、原 P4 网格、N2 主运行、旧噪声参照及冻结选择均通过来源 hash 绑定；运行结束再次核验参数、数据和实现身份。

### 28.2 GPU、运行身份与预算

| 臂 | 物理 GPU | 固定配置与 sampling seeds | 输出 identity（相对 study 根目录） |
|---|---:|---|---|
| DDIM | 0 | DDIM6/N64，20261013/20261014 | `p4_n2_noise_ddim6_n64_v1` |
| DDPM | 1 | DDPM50/N16，20261013/20261014 | `p4_n2_noise_ddpm50_n16_v1` |
| 只读汇总 | CPU | 两个优选及固定参照 × 三个 sampling seed | `p4_n2_noise_summary_v1` |

启动前两张 RTX 4070 Ti SUPER 均空闲、各显示 2 MiB 占用；来源数值代码与此前保存身份逐文件相同，项目盘可用约 253 GiB。DDIM 两 seed 共 10752 次 batched denoiser forward，DDPM 共 22400 次。按第 27 节最近一次实际 profile 估算，纯采样约 26/111 分钟，双卡并行墙钟主要受 DDPM 臂限制，另加初始化、指标与 I/O；不将估算当作完成时限。

调度目录：`runs/respdiff_bcg_study_v1/launch_p4_n2_noise_parallel_v1/`。`launch.py` 同时启动两条进程，各臂通过 `CUDA_VISIBLE_DEVICES=0/1` 使用对应物理卡，逻辑设备均为 `cuda:0`；`ddim_console.log`、`ddpm_console.log` 和 `console.log` 分别记录两臂及整体状态。任一臂失败时另一臂继续其既定预算；只有两臂完整成功后才自动汇总。

### 28.3 实现与验收

新增独立模块 `resp_train/respdiff_bcg/study_extension_noise.py` 和入口 `scripts/run_respdiff_bcg_extension_noise_v1.py`。原主采样扩展合同仍只接受 seed20261003，所有已有模型/采样/指标代码保持原 hash。本轮外层合同只接受精确的 N64/N16 与两个已授权 sampling seed。

CPU synthetic 定向测试 **8 项通过，49.25 秒**：两类 sampler × 两个额外 seed 的旧前缀逐值一致、未授权 seed/预算拒绝、原主采样合同不放宽、九项完整性与模型身份校验、CLI GPU/数据授权守卫，以及来源绑定并复制原选择的真实文件汇总 fixture。并行启动器经模拟子进程验证两条命令、GPU 映射、主运行/旧噪声来源参数和汇总入口。未重新执行已关闭 GPU 验收。

两臂保存各自完整配置、来源摘要、源码快照、环境、父/chunk 身份、两个 seed 的采样 profile、显存、前缀核验、波形和指标。汇总完整保留 N64/N16 的三组 sampling seed，加上已保存 DDIM6/N16 参照的三组结果，共 **9 行配置级结果**；不重新选择配置。min/max/mean/std 仅描述采样随机性，不作为三训练 seed 的稳定性证据。

本次队列止于四项额外评价与只读敏感性汇总。更大 N、其他训练 seed、P5 全量 validation 和独立测试集不自动启动。

## 29. P4-N2 采样噪声敏感性结果与关闭（2026-10-12）

### 29.1 执行与复核

后台任务 `j-rgz0uw` 退出码 0。DDIM 两组额外 seed 的总墙钟为 1686.28 秒（28.10 分钟），DDPM 为 6775.52 秒（112.93 分钟）。两臂各 41 项、汇总 5 项，共 **87 项** artifact hash 校验通过，四个新增评价完整。模型参数/buffer 未改变；两个 sampler × 两个额外 seed 的旧 N 前缀均与历史同 seed waveform 逐值一致（最大差与 RMSE 为 0）。

四次采样 sweep 的峰值 allocated 均为 **2.1298 GiB**，reserved 均为 **2.2598 GiB**。DDIM 两个 seed 的纯采样耗时为 787.77/798.40 秒，DDPM 为 3335.28/3340.49 秒，延续前轮 GPU1 上较慢的实测速率；这些时间不用于跨设备独立效率排名。

来源为 `p4_n2_noise_ddim6_n64_v1/`、`p4_n2_noise_ddpm50_n16_v1/` 和 `p4_n2_noise_summary_v1/`。汇总的 `sensitivity.csv` 包含两个优选及固定参照 × 三个采样 seed 的完整九项结果，`selection.json` 与主采样选择逐项相同。独立复核摘要、来源 hashes 与采样 seed 统计表位于 `p4_n2_noise_review_20261012_v1/`。未重新选择配置，未执行新 GPU 任务。

### 29.2 新配置的三组采样噪声结果

固定 B2 模型（训练 seed20260811）、同一 66-parent 选择集合，以下均为 postfiltered 指标。

| 配置 | sampling seed | PCC ↑ | Whole RR MAE（bpm）↓ | parent-max P99 ↓ |
|---|---:|---:|---:|---:|
| DDIM6/N64 | 20261003 | 0.826575 | 0.684889 | 56.918615 |
| DDIM6/N64 | 20261013 | 0.825185 | 0.623198 | 54.990679 |
| DDIM6/N64 | 20261014 | 0.824295 | 0.699625 | 61.136485 |
| DDPM50/N16 | 20261003 | 0.817811 | 0.714112 | 47.380904 |
| DDPM50/N16 | 20261013 | 0.819465 | 0.965417 | 78.271295 |
| DDPM50/N16 | 20261014 | 0.820127 | 0.373197 | 27.128615 |

三组 **sampling seed** 的描述统计（均值 ± 样本标准差）如下；不是三组训练 seed 的结果，也不构成显著性推断。

| 配置 | PCC | Whole RR MAE（bpm） | parent-max P99 |
|---|---:|---:|---:|
| 固定参照 DDIM6/N16 | 0.811044 ± 0.006045 | 0.681929 ± 0.037119 | 61.021273 ± 1.320104 |
| DDIM6/N64 | **0.825352 ± 0.001149** | **0.669238 ± 0.040546** | 57.681926 ± 3.143202 |
| DDPM50/N16 | 0.819134 ± 0.001193 | 0.684242 ± 0.297238 | **50.926938 ± 25.755081** |

在本模型与子集上，DDIM6/N64 的 PCC 三次均高于 DDPM50/N16，RR/尾部的跨采样噪声变化较小。DDPM 的平均 P99 较低，但范围为 27.13–78.27，RR 为 **0.3732–0.9654 bpm**，仍有相对 DDIM 的排序反转。两个额外 seed 中的最差 DDPM 幅值为 row12883 的 102.73，对应 target 最大值 10.43；增加 N 后仍不能认为尖峰问题消失。

规则抽样 64-parent 上，DDPM 三个 seed 的 RR 为 0.7292/0.9879/0.3808，P99 为 48.39/79.02/27.40；因此敏感性不只是额外两个压力案例造成。

### 29.3 增加 N 的跨噪声效果

从 DDIM N32 到 N64，三组 seed 的 PCC 均提高，平均 PCC 为 0.818033→0.825352；RR 均值为 0.679422→0.669238，但 seed20261014 的 RR 反而上升 0.057921 bpm。不能将平均改善表述为每个 seed 的所有指标都改善。

从 DDPM N8 到 N16，三组 seed 的 PCC 也均提高，平均为 0.803956→0.819134。P99 的跨 seed 标准差由 57.06 降至 25.76，最坏 P99 由 125.86 降至 78.27，表明尾部波动有所缓解；但 RR 均值由 **0.638526 升至 0.684242**，标准差仅由 0.346866 降至 0.297238。增大 N 没有解决全部 RR/尾部问题。

固定 DDIM6/N16 参照的平均 RR 与 N64 接近，而 PCC 低约 0.0143；N64 使用其四倍逻辑 calls/chunk。这一质量/成本权衡应在最终评价中保留。

### 29.4 关闭与下一阶段

**本轮噪声核验已完成并关闭。** 原选择保持 DDIM6/N64、DDPM50/N16，并保留 DDIM6/N16 参照；本轮仅补充其敏感性证据，不改变第 27 节 N 尚未满足稳定判据的状态。

若按当前已评估预算推进最终评定，下一步应先补齐 `long_stretched` 的训练 seeds20260812、20260813，再按冻结配置进行 P5 全量 validation；始终标明这些是预算内优选，而非 N 收敛证明。完整 validation 的 GPU 预算需依据当前实测吞吐另行核对，不能直接套用早期较快的 DDPM 估算。本次没有启动追加训练、全量评价、更多 N 或独立测试集。

## 30. 补齐两个训练 seed：GPU0 单卡串行（2026-10-12）

### 30.1 授权与固定合同

用户在第 29 节结果和补齐训练 seed 的建议后明确回复“可以，这次用GPU0一颗显卡”。本次在物理 GPU0 上按 20260812→20260813 顺序串行运行两个独立训练。各进程设置 `CUDA_VISIBLE_DEVICES=0`、逻辑设备 `cuda:0`，沿用项目 `.venv`，移除 `LD_LIBRARY_PATH`，OMP/MKL 线程各为 4。

两个模型均从相应 seed 初始化，使用已选定的 `long_stretched`：16000 updates、初始 LR=1e-4、milestones=11200/15840、gamma=0.1、B64 与原尾批规则。数据、split、target、网络、loss、快照、固定 train/validation probes、过程 DDIM6/N16 评价、最终双 anchors 及 final-update selector 均沿用第 7 节。历史 source run 仅用于核对数据合同，不作为新 seed 的参数初始化。

| 次序 | 物理 GPU | training seed | 输出 identity（相对 study 根目录） |
|---|---:|---:|---|
| 1 | 0 | 20260812 | `p3_long_stretched_s20260812_v1` |
| 2 | 0 | 20260813 | `p3_long_stretched_s20260813_v1` |

启动器与日志位于 `runs/respdiff_bcg_study_v1/launch_p3_remaining_seeds_gpu0_v1/`，每个 seed 保存独立 console、进程命令和退出回执。第一项完成且回执满足 train/complete/end_update=16000 后才启动第二项；失败则停止队列并保留全部产物。两个训练完成后保存队列 `complete.json`。

### 30.2 启动核验、预算与结束条件

启动前已确认 GPU0 空闲（2 MiB、0%、无 compute 进程），项目磁盘剩余约 254 GiB。首 seed B2 保存的训练源码与配置逐文件 hash 均匹配当前版本，唯一历史身份变化为持续更新的本计划文档；不改变训练数值代码。复用已通过的 CPU/GPU 验证；本次仅对新启动器做 CPU 模拟子进程检查。队列保存前置回执、选择文件、来源 checkpoint、源码和本计划的 hash，并在各 seed 开始前核对。

历史 B2 完整训练含 probes/anchors 在 GPU1 上耗时约 6.12 小时，只作为单 seed 预算参考；两项串行可按约 12 小时预留，实际 GPU0 吞吐和系统负载可能不同。训练期间冻结源码、配置和本计划。

验收以两个完整 train 回执、各自连续 1–16000 步 history、final checkpoint、全部快照/probes/anchors 及 artifact hashes 为准；既有 seed20260811 模型只读复用。本次队列止于这两个训练的既定阶段评价。P5 全量 validation 待训练完成后核对冻结清单、扩展 N 入口与单卡预算，再单独安排；独立 research-test 仍使用专项授权边界。

## 31. 三个训练 seed 完成与固定子集结果（2026-10-13）

### 31.1 执行、验收与模型身份

GPU0 串行后台任务 `j-jha49p` 退出码 0。seed20260812 于 2026-10-13 00:38:53、seed20260813 于 05:40:27（Asia/Shanghai）完成；两项阶段墙钟分别为 18156.60 秒（5.04 小时）和 18079.67 秒（5.02 小时），包含训练及既定 probes/anchors。整个队列从 2026-10-12 19:36 启动，历时约 10 小时 4 分钟。

两个新增 seed 各 **165 项**、合计 **330 项** artifact SHA256 全部通过；用于合并的首 seed 五项历史引用也校验通过。两份 history 均连续覆盖 updates1–16000，loss 及记录的数值均有限，LR 在 updates11201/15841 切换。每个 seed 含七个完整 epoch，各完整覆盖 131833 chunks；共 15993 个 B64 和七个 B57 尾批。两个训练均从头初始化，无恢复父 checkpoint。

三个 seed 的实际 train parents/chunks、固定 train/val probe parents 和 final anchor parents 文件 hash 相同，数据与评价集合保持一致。全部模型使用 final-update selector，固定 checkpoint 如下：

| training seed | 运行目录（相对 study 根目录） | final SHA256 |
|---|---|---|
| 20260811 | `p3_long_stretched_s20260811_v2` | `c1e16cfdfba3498a60eafad11e1a72007b5e9969ba921d2eb7b679690d558731` |
| 20260812 | `p3_long_stretched_s20260812_v1` | `5a0c9f1b06879ea62aa6173571ad29a54de28b07ca919699db9bccaea12aab0e` |
| 20260813 | `p3_long_stretched_s20260813_v1` | `b07c5efc3ed135b43578db38b197c0a0b4922ab55b595525f09414872f0c36cb` |

本次复核只读取既有产物，未执行模型或新 GPU 任务。派生结果保存在 `runs/respdiff_bcg_study_v1/p3_remaining_seeds_review_20261013_v1/`：`review.json`、`anchor_metrics_three_training_seeds.csv`、`anchor_training_seed_summary.csv`，记录来源回执、校验数量、checkpoint hashes、history 检查与统计表 hashes。

### 31.2 既定 66-parent anchors

以下均为 postfiltered、同一 66-parent validation 选择子集的指标。训练阶段按第 7 节保存的 anchors 为 DDIM6/N16 和 DDPM50/N4；保持原采样身份。这些结果用于观察固定训练合同的跨 seed 表现。

| training seed | sampler / N | PCC ↑ | Whole RR MAE（bpm）↓ | parent-max P99 ↓ |
|---|---|---:|---:|---:|
| 20260811 | DDIM6 / 16 | 0.806950 | 0.703831 | 59.595715 |
| 20260812 | DDIM6 / 16 | 0.817833 | 0.446364 | 15.862569 |
| 20260813 | DDIM6 / 16 | 0.808900 | 0.636896 | 29.636856 |
| 20260811 | DDPM50 / 4 | 0.788181 | 0.757959 | 41.537492 |
| 20260812 | DDPM50 / 4 | 0.770902 | 0.532655 | 11.861286 |
| 20260813 | DDPM50 / 4 | 0.766392 | 0.696359 | 8.874902 |

三个 **training seed** 的描述统计（均值 ± 样本标准差）：

| sampler / N | PCC | Whole RR MAE（bpm） | parent-max P99 |
|---|---:|---:|---:|
| DDIM6 / 16 | 0.811228 ± 0.005803 | 0.595697 ± 0.133586 | 35.031713 ± 22.360129 |
| DDPM50 / 4 | 0.775158 ± 0.011501 | 0.662325 ± 0.116444 | 20.757893 ± 18.057503 |

在每个训练 seed 的配对比较中，DDIM6/N16 的 PCC 更高、RR MAE 更低；DDPM50/N4 的幅值尾部 P99 更低。三个 seed 的幅值尾部差异仍较大。两个新增 seed 的尾部小于首 seed，但继续按预定三个模型共同评价，不挑选其中一个替代训练矩阵。

本表使用训练阶段 anchors，与第 29 节在单模型上更换 sampling seed 的实验具有不同的变化因素；也不能替代 DDIM6/N64、DDPM50/N16 最终配置的三模型全量结果。固定选择子集上的比较不构成完整 validation 结论或显著性推断。

### 31.3 关闭与下一阶段

**两个补充训练及三 seed anchor 汇总已完成并关闭。** `long_stretched` 三模型已齐备；P4 选定配置和第 27 节 N 未满足稳定判据的记录保持不变。

下一阶段为 P5：绑定这三个 final checkpoint，按 DDIM6/N64、DDPM50/N16 和固定 DDIM6/N16 参照执行全量 validation。执行前需完成扩展 N 的全量入口与冻结清单核验，并结合 GPU0 实测采样吞吐给出单卡预算；本次队列没有启动该阶段。独立 research-test 仍按专项协议与当次授权开放。

## 32. P5 三模型全量 validation：GPU0 单卡队列（2026-10-13）

### 32.1 授权、范围与数据合同

用户在第 31 节结果与 P5 推进建议后回复“可以”，授权本阶段的必要入口实现、冻结核验和正式执行。沿用上一阶段指定的物理 GPU0 单卡；按 training seeds20260811→20260812→20260813 串行评价第 31.1 节的三个 final16000 checkpoint。完成三模型后自动执行 CPU 汇总。

每个模型完整评价 **2675 个 validation parent、34775 个 chunk**，配置为 DDIM6/N64、DDPM50/N16，以及固定 DDIM6/N16 参照；sampling seed 固定 20261003、B64、尾批 B23、G1、FP32、原 parent LPF、原波形重建和公共 metrics。train/validation 身份与隔离核验复用原入口；评价只读取 validation waveform。三个训练模型全部进入汇总，不按途中结果改变模型、配置、指标或样本集合。

首轮 `StudySamplerSpec` 的上限保持冻结。新增 `resp_train/respdiff_bcg/study_full_validation.py` 和 `scripts/run_respdiff_bcg_full_validation_v1.py` 仅编排本次固定配置，复用既有 `ExtensionSpec`、采样、重建、metrics 与独立计时实现。两个 DDIM 配置使用同一次 N64 采样中保存的 N16/N64 在线均值前缀；DDPM 单独执行 N16。各配置仍执行独立首批计时，warmup=1、repeats=2，保存时间和显存；共享前缀耗时不当作独立 N 耗时。

### 32.2 运行身份与入口

| 阶段 | 输出 identity（相对 study 根目录） |
|---|---|
| CPU 冻结三个模型与配置 | `p5_full_manifest_v1` |
| seed20260811 全量评价 | `p5_full_s20260811_v1` |
| seed20260812 全量评价 | `p5_full_s20260812_v1` |
| seed20260813 全量评价 | `p5_full_s20260813_v1` |
| 完整九项 CPU 汇总 | `p5_full_summary_v1` |

冻结清单绑定三个 checkpoint hashes、原 P4 grid 来源、`p4_n2_selection_v1` 的选择及回执、完整 validation parent/chunk hashes、实际实现与本计划。CPU 冻结阶段加载 checkpoint 核对有限性、模型结构、训练合同和 final selector；不运行真实 waveform 或模型前向。历史数值源码逐文件核对，新增外层编排单独进入本轮源码快照。

命令（在本计划第 1 节工作树中执行）：

```bash
PY=/mnt/disk_code/marques/resp_reconstruction/.venv/bin/python
STUDY_ROOT=runs/respdiff_bcg_study_v1
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PY" \
  scripts/run_respdiff_bcg_full_validation_v1.py freeze \
  --grid-run "$STUDY_ROOT/p4_grid_b2_s20260811_v1" \
  --selection-run "$STUDY_ROOT/p4_n2_selection_v1" \
  --source-runs "$STUDY_ROOT/p3_long_stretched_s20260811_v2" \
    "$STUDY_ROOT/p3_long_stretched_s20260812_v1" "$STUDY_ROOT/p3_long_stretched_s20260813_v1" \
  --output "$STUDY_ROOT/p5_full_manifest_v1"

# 启动器为三个独立 run 逐项设置 GPU0、逻辑 cuda:0 与显式数据访问确认。
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PY" \
  "$STUDY_ROOT/launch_p5_full_gpu0_v1/launch.py"
```

启动器实际命令、逐 seed 进程与退出回执、console 日志均保存在 `launch_p5_full_gpu0_v1/`。任一模型失败则停止后续队列，保留失败产物；仅三个 complete 回执齐备后执行九项汇总。输出目录不可覆盖；后续恢复须使用独立身份并明确已完成配置的复用范围。

### 32.3 验证与预算

新增 CPU synthetic 定向检查 **9 项通过**：tiny 模型上 DDIM N64 的 N16 前缀与独立 N16 逐值相同且模型状态不变；重复 seed、错误 checkpoint、N/采样 seed/B64 漂移和缺失固定参照均被拒绝；模拟全量编排覆盖全部 parent、两次采样与三配置独立计时；九项汇总拒绝缺行、样本数量变化、非有限指标及产物 hash 篡改；CLI 要求显式真实数据访问授权。训练、采样及 metrics 数值源码保持原样，已完成的历史 GPU 验收不重跑。

GPU0 最近一次训练 anchor 的满 B64 约为 DDIM6/N16 每批 14–15 秒、DDPM50/N4 每批 30 秒。完整 validation 含 544 批，按 N 线性粗估，单模型 DDIM6/N64 约 8.8 小时、DDPM50/N16 约 18.1 小时，三个模型纯采样约 **81 小时**；加上数据准备、重建、指标、保存和独立计时，单卡预留 **3–4 天**。该估计来自既有日志，不是新增 benchmark；实际吞吐受设备与系统负载影响。启动检查时 GPU0 空闲，项目磁盘剩余约 246 GiB。

运行期间冻结源码、配置与本计划。验收要求为三个完整全量回执、九项配置、每项完整 2675-parent 指标与 raw/postfiltered 波形、冻结父/chunk 身份一致、参数和 buffer 不变、来源 checkpoint 不变、独立计时及可校验的产物 hashes。汇总输出逐 seed 九项表和三训练 seed 的均值/样本标准差/范围。本阶段用于形成最终 validation 结果；N 收敛状态仍按第 27 节解释，独立 research-test 仍须专项协议与当次授权。
