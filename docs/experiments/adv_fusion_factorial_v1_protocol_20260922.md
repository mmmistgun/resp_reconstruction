# ADV 融合方式 × 融合位置对照 v1

日期：2026-09-22。协议 ID：`adv-fusion-factorial-v1-es30p15-train-val-20260922`。本修订统一启用最短30epochs、patience15的提前停止策略，替代本轮尚未执行的固定80epoch训练合同；上一版实现锁保留为工程历史。

状态：提前停止修订已实现，CPU合成验证23项通过（50.39 s）；GPU 合成验收、有限真实数据 smoke、缓存数组访问和 formal 训练尚未执行，由用户执行。实验固定为三种方式 × 两种位置 × 三个 seed，共 18 次 formal 训练。当前协议只开放 train/validation，test 不属于本轮执行范围。当前实现身份与CPU验证记录见 `adv_fusion_factorial_v1_implementation_lock_es30p15_20260922.json`。

分支：`codex/adv-fusion-factorial-v1`；工作树：`/mnt/disk_code/marques/resp_reconstruction/.worktrees/adv_fusion_factorial_v1`；基点为 ADV 收尾提交 `f1a4236`。本轮实现位于 `resp_fusion/`，配置位于 `configs/adv_fusion_factorial_v1/`，运行入口为 `scripts/run_adv_fusion_factorial_v1.py`。

## 1. 依据与研究问题

来源：[ADV 收尾](aligned_dual_view_v1_closeout_20260922.md)、[ADV validation](aligned_dual_view_v1_results_20260922.md)、[ADV research-test](aligned_dual_view_v1_research_test_results_20260922.md)、[W0 条件调制定义](w0_cwt_film_behavior_protocol_20260918.md)、[W0 训练参照](w0_film_gamma_training_results_20260919.md)、[W0 research-test](crd_tf_v1_research_test_protocol_20260816.md)。

历史参照的方向需要区分 split 和配置：validation 中 ADV joint 的 Whole RR / Local RR / trajectory 均值为 0.484406 / 0.548861 / 0.150516，W0 为 0.499298 / 0.551309 / 0.152729；ADV joint 的 PCC / global modulation 为 0.842308 / 0.215527，W0 为 0.865300 / 0.191051。Research-test 中 W0 的 Whole/Local RR 为 0.617234/0.609566，优于已有 ADV 四组；ADV 尺度均值的 trajectory=0.137495，略低于 W0 的 0.139546。这些历史比较存在结构、宽度、前端、表示和执行差异，只构成设计动机。

本轮核心归因来自六组受控比较：相同位置比较融合方式，相同方式比较位置，再计算方式 × 位置交互。卷积前端是固定工作基础，结论限于该前端；Patch Mixer 的价值不由本轮判定。固定完整 97→32 CWT 投影，尺度均值与前端深度不作为因子。历史 ADV joint 的计算图不同，不能充当 A 组。

## 2. 前向与维度

输入 x 为 `(B,1,18000)`，100 Hz、180 s。

- 波形编码：Conv1d(1,32,kernel=51,reflect padding=25,bias=True) → SiLU → 固定 D10。
- CWT：复用 ADV 冻结的 97 scales、ssqueezepy 0.6.6 Morlet(mu=13.4)、100 Hz log1p(abs(CWT)) → 同一 D10，float32 `(B,97,1800)`；沿用原点0、间隔0.1 s的网格。
- D10：中心对称 Kaiser FIR501、beta8.6、cutoff4 Hz、reflect250、stride10；固定系数与累加均 float32，CUDA cudnn convolution 设为 IEEE。
- CWT 编码：Conv1d(97,32,1,bias=True) → SiLU。
- 两路适配：分别使用 Linear(32,64,bias=True)，逐时刻应用，不附加激活或归一化。输出 U,V 均为 `(B,1800,64)`。
- T：六层既有 BidirectionalMamba2Block，D64、d_state64、d_conv4、expand2、headdim32、ngroups1、chunk_size256、dropout0；保留既有每块 RMSNorm、正反向 Mamba、方向拼接线性合并和残差。
- R：Conv1d(64,1,1,bias=True)，既有 Fourier 插值回 18000 点；任务 Pi 仍由 loss/evaluator 外置应用。

| Arm | 方式 | 位置 | 前向 |
|---|---|---|---|
| A | concat | pre | R(T(F_concat(U,V))) |
| B | concat | post | R(F_concat(T(U),V)) |
| C | FiLM | pre | R(T(F_film(U,V))) |
| D | FiLM | post | R(F_film(T(U),V)) |
| E | 双视图 attention | pre | R(T(F_attention(U,V))) |
| F | 双视图 attention | post | R(F_attention(T(U),V)) |

Post 严格在第六层 Mamba 后、读出前。两路编码深度、时间网格和适配层在六组一致；V 不额外经过时间主干。无新增跨时间 attention、池化、融合 norm、激活或 dropout。

## 3. 融合与初始化

拼接：`Linear_128→64(concat[h,V])`，weight 初始化 `[I,0]`，bias=0。

FiLM：`h * (1 + 0.5*tanh(Linear_gamma(V))) + 0.5*tanh(Linear_beta(V))`。Gamma/beta 为两个独立 Linear(64,64,bias=True)，weight/bias 全零；系数固定为0.5/0.5。

Attention：每个时间位置 t，q=Wq h_t，两个 key/value 为 h_t 与 V_t 的共享 Wk/Wv 投影；D64、4 heads、每头16维。`softmax(q·k/sqrt(16))` 的轴长度严格为2；float32 logits/softmax，再将权重转为 value dtype。按权重合并两个 value、拼合4头，经过 Linear(64,64,bias=True) 输出投影，再与 h 残差相加。Attention dropout=0。

Q/K/V 投影均为 Linear(64,64,bias=False)，使用 PyTorch 默认 Kaiming-uniform 初始化；共享 key bias 会给两个 logits 加同一常数而在 softmax 中抵消，因此统一采用无 bias 的 Q/K/V。输出投影 weight/bias 零初始化。

所有共享前端、adapter、T、R 通过独立命名子 seed 构造；六组在相同 seed 下逐 tensor 一致。同一方式的 pre/post 融合参数也逐 tensor 一致。新 adapter 使用 PyTorch 默认线性初始化；原 ADV 共享模块构造保持原实现。

六组初始函数均为 R(T(U))，但优化参数化不同，不能将“初始函数一致”等同于“优化轨迹一致”。初次反传：CWT 上游梯度为零；concat 的 CWT 权重列、FiLM gamma/beta 头、attention 输出投影可以更新。Attention 的 Q/K/V 首次梯度也为零。更新后需验证条件分支、Q/K/V 权重及两个输入视图均获得非零且有限的梯度。AdamW 对零梯度但非零参数仍可能施加 weight decay，不能将首步零梯度描述为完全冻结。

每个 run 保存 `initialization.json` 的逐 state tensor 哈希和共享身份；汇总检查六组共享初始化与同方式 pre/post 的配对。自然参数量由原生模型 `describe` 实测，不以无效参数补齐。

## 4. 参数、计算和显存口径

融合模块参数量：concat=8,256；FiLM=8,320；attention=16,448。两路适配层各2,112。模型报告逐模块列出前端、主干、读出、adapter 和 fusion 参数量。

原生实例统计：波形编码1,664、CWT编码3,136、主干463,248、读出65；总参数A/B为480,593，C/D为480,657，E/F为488,785。相对concat，FiLM自然多64参数，attention多8,192参数。同方式的前后位置参数量相同。

融合一次前向、单窗口1800点的 MAC：concat/FiLM 各14,745,600；attention=44,697,600。Attention 包括 Q 一次、K/V 各两个 token、输出投影一次的线性运算，以及两个点积和加权求和。1 MAC 计2 FLOPs；此解析统计不计 bias、tanh、SiLU、softmax、逐元素 FiLM/残差、主干与前端，也不是端到端速度指标。

Attention 权重每窗口仅 `1800×4×2=14400` 个元素，Q/K/V、autograd 和原生 Mamba 仍会占显存。显存和实际耗时不由参数量推断：GPU 合成报告提供相同 batch32 下的峰值 allocated/reserved bytes 和完整模型+任务 loss+反向+AdamW step 时间（2步预热、3步记录），含有限性/梯度验收检查与同步，不计离线 CWT 和文件 I/O。这是工程验收，不代替正式吞吐 benchmark。

Attention 组效应包括其参数量、投影及残差等整个融合模块，不能归因于权重选择本身。Attention 权重仅描述输入依赖，不直接解释为生理贡献或可靠性。

## 5. 训练、缓存与产物

六组均使用 seeds 20260811/20260812/20260813。沿用 ADV 的完整 train10141/validation2675、sample seeds20260610/20260611、admission、输入/target、L_sync+0.25 L_effort、AdamW、max/min LR3e-4/3e-5、5% warmup与按更新步 cosine、weight decay1e-4、grad clip1。严格 physical batch32、accumulation4，bf16、drop_last=false。最大80epochs、最多6400更新；学习率始终按完整6400更新计算，不随实际停止点压缩。按已执行的full-validation Local RR最小值选checkpoint，并列取最早；resume=false。

统一停止合同：`enabled=true、min_epochs=30、patience=15、min_delta=0、max_epochs=80`。每次完整validation后，严格降低Local RR才更新最佳点并重置未改善计数；并列不重置。最短训练期内也累计未改善次数，达到epoch30且距最近改善已满15epochs时即可停止。若epoch30首次改善，最早在epoch45满足patience。达到epoch80时终止原因记为`max_epochs`，更早触发则记为`patience_exhausted`。Smoke保持最多2epoch的独立工程预算，不触发formal早停。

终止时保存实际末轮checkpoint，再重载最佳checkpoint进行最终评价；manifest记录实际epoch/更新数、固定学习率总预算、停止原因、最佳epoch和未改善计数。所有18项使用同一策略，但允许实际训练长度不同。历史结果保持原选点；本修订只影响本轮后续训练，不需要重算既有ADV/W0。

新的协议、配置、输出 identity 与源码快照独立保存；非有限输入、target、prediction、loss、gradient、checkpoint 或 eligible 指标显式失败。失败目录与回执保留；目录必须新建，成功产物不可覆盖。

旧 CWT 仅允许复用 ADV formal cache：
`/mnt/disk_code/marques/resp_reconstruction/.worktrees/aligned_dual_view_v1/runs/aligned_dual_view_v1/cache_formal_20260921_r1`。
Manifest SHA=`ecce7f6c67bd4bc585e98e20f8c441ad19c328cde6c73c83e6e84ba6b6c1f424`；旧执行源码身份=`19b92747cb2a26a5cf45d6954aeebaae1ce70b3de5a01a266677de4761284040`。

正式读取时需核验旧成功回执、固定 manifest/代码身份、完整选择合同与 index SHA、逐 row 身份/顺序、数学表示、transform 源码和依赖、selection/rows/完整特征文件哈希、逐输入与特征内容哈希；训练前再扫描输入/target/mask 并保存 provenance。任一不匹配直接失败。W0 的 `[97,360]` 缓存不满足数学定义，拒绝使用。有限 smoke 自建本轮 cache；若正式缓存不可用，仅能在独立新路径按本协议构建。

每个训练 run 保存 config、命令、环境、Git及源码快照、sample identities、input/target/mask 哈希、初始化、模型参数、optimizer分组、完整 history、改进与最终 checkpoints、五指标逐窗口 CSV/summary 和完成/失败回执。

## 6. 完整矩阵分析

汇总只接受18个唯一formal run，每项30..80epochs、每epoch80更新，满足同一停止合同，学习率计划总预算固定6400。逐条回放完整Local RR history，必须恰在首个合法停止点结束；无停止依据的截断、触发后继续训练、伪造停止回执或混用旧合同均拒绝。同seed六组共享初始化、同方式pre/post融合初始化、数据/target/mask、cache、代码、依赖、最大预算和指标口径一致。验证来源哈希、选点与history、checkpoint文件、metrics行/顺序/方法及逐窗口汇总数值。

每个 seed 使用五主指标 sample direct mean，跨seed等权 mean/sample SD。输出：

- `per_seed.csv`：18行，含实际epochs、更新数及停止原因；`across_seed.csv`：6×5=30行。
- `contrasts_per_seed.csv`：240行；分别为两位置下三组方式差、三种方式的位置差、三组交互，以及等权边际方式/位置效应。
- `contrasts_across_seed.csv`：80行，逐对比的三seed mean/sample SD。
- `manifest.json`：18项来源及文件SHA。

方式差定义为 FiLM−concat、attention−concat、attention−FiLM；位置差为 post−pre；每种方式对 m2/m1 的交互为 `(m2_post−m2_pre)−(m1_post−m1_pre)`。分别保留全部五指标：四项误差负值为改善，PCC正值为改善；交互的符号只表示位置效应之差，不直接称为总体收益。不生成跨指标总分或基于局部结果缩小矩阵。

三种两两交互中只有两个线性独立，attention−FiLM 的交互等于 attention−concat 减 FiLM−concat；不将三者视为三个独立证据。

三个seed描述训练随机性，不能作为三个独立人群。W0与已有ADV是历史参照，核心归因限于本轮六组在统一停止策略下的差异，不将其解释为等实际计算量比较。既有test已参与设计动机，此设计属于结果知情开发。

## 7. 用户执行与验收

先读 [执行命令](adv_fusion_factorial_v1_commands_20260922.md)。顺序为：GPU合成六组 → 独立小cache → 六组有限smoke → 完整18组formal → 一次完整summary。任一失败即停止当前shell矩阵，保留失败目录。

GPU验收：原生Mamba前向/反向、bf16、任务loss、输出形状和全部梯度有限；更新后的CWT与Q/K/V梯度非零；记录参数、计算口径、显存与耗时。CPU替身验证不能代替该验收。

Smoke验收：每组最多64 train/val窗口、最多2epochs，完整保存失效/成功生命周期，cache仅输入，训练/选点/重载/指标管线可贯通。Formal验收：18项各满足30/15/0/80停止策略，实际epoch与更新数一致、停止点回放正确、同完整样本身份、选点及初始化校验通过；五主指标完整且有限或与冻结eligibility一致。Summary验收：18/30/240/80行、全部来源校验通过。

Early stopping 的历史回放与限制见 [训练过程评估](adv_fusion_factorial_v1_earlystop_review_20260922.md)。输出identity使用`es30p15`，停止规则在全部18组执行前统一固定，不按部分组的中间结果调整。
