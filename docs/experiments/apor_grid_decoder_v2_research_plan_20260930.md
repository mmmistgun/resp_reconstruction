# APOR v2：主干时间网格 × 调制后时间细化研究方案

日期：2026-09-30。方案ID：`apor-grid-decoder-v2-20260930`。

状态：**第一阶段实现完成，用户已明确授权“开始实现，后台启动”。** 本次开放两卡synthetic GPU验收、参照等价性核验、N1/D1共六次train/validation及12-cell完整validation汇总。执行身份与进度由新session、engineering/formal/summary回执记录。激活规范化、benchmark及research-test保持后续阶段。既有PATCH/APOR、W0及其他已关闭实验保持冻结。

## 1. 研究目标与证据起点

核心问题：**原生Patch主干的节律表现，是否受主干处理网格和FiLM后缺少可学习时间细化的共同影响？能否通过一个职责明确的条件解码阶段，保持APOR的波形相关性并改善节律？**

已有证据见[validation与排除670的test综合结论](patch_apor_v1_validation_test_exclude670_conclusions_20260930.md)：

- A0相对M0：validation五项窗口均值均改善；排除670的test中Local RR增加8.47%，PCC仍在三个seed改善。
- A3相对A0：validation两项RR均退化；排除670的test两项RR改善12.43%/11.40%，但PCC下降。
- 删除M0的解码残差，其validation收益没有跨split复现；时间处理与读出之间存在值得研究的节律—形态取舍。
- TM3×REF2在test子集保留窗口Local RR收益，受试者等权收益接近零，作为后续专项机制问题保留。

上述事实产生研究假设，不预设新模块有效，也不将A3指定为最终模型。新方案已受到历史research-test结果启发，其证据属性为研究开发。

## 2. 统一的功能架构

最终叙事与实现接口统一为四个功能阶段：局部编码、序列建模、条件融合、波形解码。

```text
x[B,1,18000]
  → Patch局部编码[B,16,140]
  → 指定主干网格上的adapter与BiMamba2
  → 回到公共Patch坐标[B,96,140] ───────────────────┐
                                                ↓
W[B,97,360] → 浅层尺度编码 → 五点条件对齐 → 有界FiLM
                                                ↓
                         条件解码：时间细化或Identity
                                                ↓
                           片段MLP → Hann重叠合成 → Pi
```

时间细化是条件解码阶段内部的可选计算，不引入新的输入分支或新的训练目标。主干网格和调制后处理以外的既有计算保持相同。M0、W0用于外部参照，不填入本轮二因素矩阵。

## 3. 第一阶段：2×2结构矩阵

| Arm | 主干网格 | FiLM后时间细化 | 固定来源/新增内容 |
|---|---|---|---|
| N0 | 原生140-token | Identity | 既有A0的计算图 |
| N1 | 原生140-token | 一个时间残差块 | N0增加条件解码时间细化 |
| D0 | 1800点密集网格，主干后采回140点 | Identity | 既有A3的计算图 |
| D1 | 1800点密集网格，主干后采回140点 | 同一个时间残差块 | D0增加条件解码时间细化 |

Seeds固定为20260811、20260812、20260813，科学矩阵为12个arm×seed单元。各臂从对应公共初始化训练，不从A0/A3的已训练checkpoint继续优化。

### 3.1 公共计算与物理坐标

- 输入180 s、100 Hz、18000点。Patch长度256、步长128，右补48点，140个Patch。
- 第j个Patch起点128j、中心128j+127.5（原始100-Hz采样坐标）。
- 原生网格直接在140个Patch上执行16→96 adapter/GN/SiLU与六层D96双向Mamba2。
- 密集网格严格沿用A3：将16通道特征从Patch中心采到`0,10,...,17990`，在1800点上执行adapter与主干，再采回Patch中心；采用原CoordinateSample线性插值及边界延拓。
- 所有臂在FiLM前已经回到`[B,96,140]`，条件融合与新增时间细化的时间网格完全一致。
- W保持97尺度、360时间点；原浅层二维编码、scale mean、每Patch五位置条件、480→96映射、H65残差、原gamma/beta投影。
- FiLM保持`U=Z*(1+0.5*tanh(gamma))+0.5*tanh(beta)`；最终投影零初始化。
- 片段解码保持`Linear(96,96)→SiLU→Linear(96,256)`，原Hann权重与float32重叠合成，裁到18000点，原Pi执行一次。

主干网格因素同时改变序列长度、局部卷积的物理跨度、状态更新序列及归一化统计。它研究的是该网格处理方案的整体效应，不单独归因为token数量。140个多维token不等同于140个标量波形采样点。

### 3.2 新增时间残差块的精确定义

输入、输出均为`U[B,96,140]`，位于FiLM之后、片段MLP之前：

```text
V = GroupNorm(12 groups, 96 channels, eps=1e-5, affine=True)(U)
V = SiLU(DepthwiseConv1d(96,96,kernel=3,stride=1,padding=1,
                        dilation=1,groups=96,bias=False)(V))
V = SiLU(Conv1d(96,192,kernel=1,bias=False)(V))
T(U) = U + Conv1d(192,96,kernel=1,bias=True)(V)
```

- 单块、无新增dropout；末端96通道投影weight/bias均零初始化，初始`T(U)=U`。
- GN初始weight=1、bias=0；其余卷积初始化沿用现有`_initialize_conv`约定。
- 新参数使用独立命名子seed，建议命名`apor_grid_decoder_v2_postfilm_temporal`；同seed的N1/D1新增块逐tensor初始化相同，公共模块不受构造顺序影响。
- 单块新增参数理论值37440；N0/D0为1077640，N1/D1为1115080，实施时构造核对。
- 时间核读取三个相邻Patch，中心间距1.28 s，首尾中心跨度2.56 s；这不是整个网络的感受野，GN统计和既有主干还有更广的依赖。

该因素评价的是包含归一化、时间卷积与通道映射的完整残差块。若得到收益，首先称为“调制后时间细化模块的贡献”，进一步将收益归因于平滑、相位一致性或主峰稳定性，需要对应诊断支持。

## 4. 冻结参照与训练数量

优先采用**复用6个冻结参照单元、新训练N1/D1共6个单元**的模式。该模式须在新训练前通过以下来源与等价性检查：

1. 固定历史训练session `runs/patch_apor_v1/session_20260929T181607Z_77e89165fef0`，SHA256=`cdf6337db77d73724f93508a3568669863a17011066e7d332cf69b18b4e98fa2`。
2. N0/D0分别绑定该session的A0/A3，seeds、完整config、history、selector、checkpoint、逐窗口指标和formal manifest可追溯。
3. 新入口的N0/D0与对应历史图在命名参数、buffer、初始化、输入表示、物理坐标、前向及反向上等价；新增块末端为零时，N1/D1前向分别重放N0/D0。
4. 数据、样本顺序生成、batch、优化器分组、学习率、精度、停止与checkpoint合同一致；实际软件环境与GPU型号/精度路径可核对。
5. 逐seed数值等价检查覆盖FP32和实际BF16路径。精确复用同一算子的路径要求一致；若计算顺序改变导致差异，必须定位并记录，不能通过放宽容差把不同计算图标成相同。

源码与身份审计读取历史manifest中列出的冻结来源，并另记新实现依赖。历史session可能按整个`resp_train`文件集合冻结，新增文件造成的集合变化要与旧文件字节变化区分；不得修改旧manifest或绕过实际依赖核验。

若无法满足上述条件，训练前将执行模式明确修订为**独立12-cell同轮重训**，保留历史参照。这一模式改变须说明额外6次训练并获得相应资源授权；不能在结果出现后按表现决定复用或重训。

## 5. 训练合同

沿用[PATCH/APOR v1合同](patch_apor_v1_protocol_20260930.md)：

| 项目 | 固定设置 |
|---|---|
| 数据 | train/validation=10141/2675窗口，32/7个samp_id；180 s/100 Hz |
| Sample seed | train=20260610，validation=20260611 |
| 训练seed | 20260811、20260812、20260813 |
| 目标 | 原Pi、`L_sync + 0.25 L_effort`、eligibility与五主指标 |
| Batch/精度 | physical/effective batch128，accumulation1，BF16 |
| 优化 | 原AdamW分组，LR 3e-4→3e-5，warmup0.05，gradient clip1 |
| 预算 | 最大80epochs；80updates/epoch；LR始终按6400updates计划 |
| 停止 | min_epoch30、patience15、min_delta0 |
| Checkpoint | 完整validation Local RR严格最小，并列最早 |

每个cell完成后保存完整history及best/final checkpoint。非有限input、target、预测、loss、梯度、checkpoint或核心指标按原合同显式失败，保留失败现场。中间seed结果不改变剩余矩阵或训练预算。

## 6. 预设比较与判断

### 6.1 主要比较

按同seed配对，保存原始单位差、误差相对百分比、三seed均值/SD与改善方向数：

- N1−N0：原生网格的时间细化作用。
- D1−D0：密集网格的时间细化作用。
- D0−N0：无时间细化时的主干网格作用。
- D1−N1：有时间细化时的主干网格作用。
- 交互：`I=(D1−D0)−(N1−N0)`，在每个指标的原始单位上计算；不相减使用不同分母的百分比。

四项error中，`I<0`表示细化对密集网格更有利；PCC中方向相反。交互同时报告逐seed与subject-macro，不用一个跨指标总分替代五项结果。

### 6.2 结构候选与停止条件

机制结论先报告完整矩阵。进入第二阶段的结构候选只由完整validation决定：

1. N1/D1各自相对对应N0/D0检查：窗口Local RR均值至少改善0.5%，至少2/3 seed同向。
2. 两种聚合口径内，Whole RR、Local RR、轨迹及全局调制均值退化不超过0.5%，PCC下降不超过0.002。
3. 上述容差是预先指定的工程研究阈值，沿用历史实验的材料性量级，不是临床阈值、统计显著性或非劣效证明。
4. 若新增臂通过，进入候选集合；N0/D0为保留参照。集合内以三seed窗口Local RR均值最低者作为下一阶段APOR结构候选；精确并列依次比较subject-macro Local RR、参数量、arm固定顺序N0/N1/D0/D1。
5. 若新增臂均未通过，记录机制结果；下一阶段可保留validation选择的既有参照，新增时间细化不自动成为最终结构。

结构候选及第二阶段U0身份在第一阶段research-test访问前记录并冻结。本规则选出APOR家族内的研究候选，不自动替换W0或本轮既定M0。逐受试者差异与留一受试者后比较方向作为稳定性描述，不能当作重新训练的交叉验证。

## 7. 机制诊断与可解释产物

主要诊断使用完整validation与已固定checkpoint。指标计算、阈值及结构选择不依据research-test案例继续调整。

| 诊断 | 输出 | 能回答的问题 |
|---|---|---|
| RR与尾部 | Whole/Local RR、P90/P95、>2/>5 bpm、逐受试者配对 | 增益是否集中在部分人或尾部 |
| RR频谱证据 | 复用原RR窗口、频带和插值，保存预测/参考RR及相应频谱峰位置、功率 | 误差变化是否伴随主峰位置/相对功率改变 |
| 重叠片段一致性 | FiLM后decoder输出在相邻Patch重叠区的归一化差异 | 细化是否改变片段间一致性 |
| 波形与条件案例 | 同一窗口并列四臂预测、参考、RR频谱及重叠差异 | 观察节律、形态与片段一致性的对应关系 |

重叠诊断定义：相邻Patch的`p_j[128:256]`与`p_(j+1)[0:128]`配对，按窗口对全部139对、128点计算MSE，再除以两段平方能量均值加1e-8。它是辅助诊断量，不替代主指标；低能量情况额外标记并报告，不通过删窗改善结果。重叠更一致不自动证明RR更准确。

案例从固定参照的validation Local RR误差分位位置选择，使用稳定row ID并列全部臂；完整定量分析覆盖全部窗口。671在历史test中的结果只作为提出假设的依据，后续test图用于冻结后的描述性复核。倍频/半频或相位漂移若未作专门核验，不直接命名为已证实机制。

建议保存selected checkpoint的validation预测与公共参考（float32、带row ID及hash），避免后续重复推理。12份预测加一份参考约2.50 GB十进制，不含checkpoint和其他产物；中间特征只保存所需标量诊断。历史参照若没有预测数组，在获准的validation诊断阶段各导出一次，写入新目录并引用旧checkpoint。

## 8. Research-test评价规则

训练与validation完成后，另以本方案的增量附件固定全部checkpoint、样本、访问范围和执行命令，并取得当次执行授权。

- 评价完整12-cell科学矩阵；严格匹配的既有N0/D0 test产物优先复用。复用模式新增推理6个checkpoint；同轮重训模式评价12个新checkpoint。
- 预设三种视图：完整2310窗口/8人；排除670的2231窗口/7人敏感性分析；670的79窗口单病例描述。每个视图均列分母，前两种同时报告窗口平均和subject-macro。
- 完整test为默认评价集合；排除670不是替换默认人群的规则。用户提供的重度OSA伴体动背景单独注明，病例性质不由模型误差反推，剩余7人不命名为正常人群。
- 五项主指标、资格/失败、IBI及coverage、coherence/nDTW等沿用原test口径。全部checkpoint在访问前固定，完整矩阵完成后统一汇总。
- 原research-test已用于开发，本轮仍为开发性复核。新独立人群若可用，应在最终模型冻结后通过独立方案确认；现有单病例分析不建立疾病亚组结论。

## 9. 第二阶段：激活规范化

结构候选由第6节固定后，再单独比较激活配置。其余结构与训练合同保持一致：

| 条件 | 改动 | 训练数量 |
|---|---|---|
| U0 | 第一阶段选定结构的原激活配置 | 复用其三seed完整来源，等价性条件同第4节 |
| U1 | 两个PatchMixer中共4个GELU统一替换为SiLU | 新训练3seed |

条件编码、适配器、新时间残差块和片段MLP已使用SiLU；FiLM限幅tanh、线性波形输出及Mamba内部算子按原功能保留。GN/RMSNorm的统计轴、分组、epsilon、H65宽度、条件采样与Hann均保持不变。

U1若在validation两种聚合口径内，四项error相对U0退化均不超过0.5%、PCC下降不超过0.002，则作为统一激活的候选进入后续固定评价；否则保留U0，并记录具体取舍。这里是保持质量的工程检查，不要求激活替换本身成为论文贡献。即使均值通过，也完整报告seed差异。

U1的test访问仍需单独阶段授权并固定checkpoint；test描述其跨split表现，不回选其他epoch。若最终修改归一化、宽度或其他算子，应作为新的设计变更说明，不能并入本次激活比较。

## 10. 实现、验收与运行组织

实现采用独立`apor_grid_decoder_v2`版本。入口为`scripts/run_apor_grid_decoder_v2.py`，模型与运行器位于`scripts/apor_grid_decoder_v2_model.py`、`scripts/apor_grid_decoder_v2_runtime.py`，配置为`configs/apor_grid_decoder_v2/experiment.yaml`，输出根为`runs/apor_grid_decoder_v2/`。独立脚本位置保留历史`resp_train`源码集合及内容身份。

1. **来源与模型实现：**固定依赖、各臂计算图与参考映射；模型实现避免改动历史受管源码。工程整理需通过等价性检查，不能静默改变数值图。
2. **快速CPU验证：**synthetic/disposable fixture覆盖坐标、shape、有限值、零初始化恒等、公共初始化、参数数量、梯度、保存回载、矩阵与summary计算。Mamba替身只验证外围逻辑，不能替代原生模型验收。
3. **GPU工程验收：**获准后，原生四臂×三seed batch1多步训练；四臂batch128训练与eval；检查新增块初始恒等、外层投影及内部参数在门控打开后获得梯度、原生loss finite、optimizer覆盖、peak reserved/total≤0.85。验收失败保留现场，修复后使用新attempt。
4. **冻结执行身份：**在真实训练前固定源码快照、resolved configs、依赖环境、来源receipt、执行模式、cell计划和命令。一个版本复用一个session身份，各阶段用receipt表达状态。
5. **正式运行：**可沿用两张GPU固定交错分片；每个worker CPU线程4，完整矩阵先固定再调度。Worker失败停止另一路并保留产物；完整汇总只接受全部必需来源。
6. **效率记录：**formal运行记录参数、selected/completed epoch及显存峰值；若需要论文时延结论，再获准开展同硬件、同精度的匹配benchmark，同时区分cached-input与包含CWT的完整路径。

新产物不可覆盖旧identity。失败、恢复与引用来源均保存；完成阶段不重复启动。最终命令由实现完成后的CLI帮助和实际验收结果生成，不把规划命令当成现成工具。

## 11. 预算与阶段交付

| 阶段 | 预期新增训练 | 最大optimizer updates | 交付 |
|---|---:|---:|---|
| 第一阶段，参照复用通过 | 6 | 38400 | 12-cell完整比较、交互、逐受试者与机制诊断 |
| 第一阶段，改为同轮重训 | 12 | 76800 | 同轮12-cell完整比较及历史参照 |
| 第二阶段，激活规范化 | 3 | 19200 | U0/U1三seed配对结果 |

优先路径合计新增9次训练；若第一阶段改为同轮重训则合计15次。上述是预算上限计数，不估计未经新结构测量的GPU小时。各阶段独立说明资源范围与执行授权，第二阶段不在第一阶段运行中自动展开。

## 12. 论文组织与完成标准

预期论文主线为“时频条件下的呼吸波形重建及节律—形态协调”。最终方法用统一计算图、坐标表和模块职责解释；激活/归一化使用规则与实际配置一致。

关键证据分别承担明确任务：

- 2×2比较回答主干网格与调制后时间细化是否存在交互。
- 频谱、片段一致性和逐受试者结果解释性能变化，避免仅凭平均指标命名机制。
- 激活比较检验最终实现规范化能否保持质量。
- 完整人群、病例敏感性与独立确认分别界定适用范围。

第一阶段完成标准是完整矩阵、来源一致、预设比较与诊断均可审阅，无论新增模块是否改善。论文主模型决定在结构、规范化和相应评价证据齐备后单独记录，不由某一test子集的单项排名自动产生。

## 13. 当前实现与执行命令

CPU synthetic定向验证覆盖模型/初始化/前向反向等价、新增块三步梯度、原生训练器与selected-checkpoint预测导出、完整矩阵交互及validation选择、失败保留、双卡验收先于训练和失败汇总门禁。历史六份参照的session源码、配置、history、checkpoint和metrics来源已通过只读核验；原生GPU等价性仍由工程阶段验收。

工程修订（2026-09-30）：首个GPU验收session `session_20260930T093034Z_b539544e8fa0`在FP32公共梯度逐位比较处停止，尚未读取真实数据训练。原始模型自身重复反向也出现约1e-11量级差异、前向逐位一致，定位为GPU反向非确定性。严格等价复核改在隔离子进程启用PyTorch确定性算法与`CUBLAS_WORKSPACE_CONFIG=:4096:8`，仍要求FP32/BF16前向及公共梯度逐位一致；已先验证N0/D0首seed通过。正式训练和训练验收保留历史默认算法设置，确定性设置不传播给训练worker。原失败session及源码快照保留，修订后创建新session；本修订只改变工程检查方式。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. ./.venv/bin/python -m pytest tests/test_apor_grid_decoder_v2.py -q
./.venv/bin/python scripts/run_apor_grid_decoder_v2.py plan
./.venv/bin/python scripts/run_apor_grid_decoder_v2.py prepare
```

使用prepare返回的session路径启动第一阶段；两张卡全部验收成功后才启动训练worker：

```bash
APOR_V2_SESSION='/prepare返回的实际路径'
env -u LD_LIBRARY_PATH -u LD_PRELOAD PYTHONUNBUFFERED=1 \
  ./.venv/bin/python scripts/run_apor_grid_decoder_v2.py parallel \
  --session "$APOR_V2_SESSION" --devices cuda:0 cuda:1 --confirm-training
```

六次新训练按固定交错分片各3个cell，两张卡均包含两种网格。每个新cell在原生selected-checkpoint评价时保存validation预测、公共参考来源及逐窗口重叠诊断；不增加一次重复评价。全部训练成功后，汇总六份冻结参照和六份新结果，并记录下一阶段候选及checkpoint身份。历史参照的波形导出和完整机制图表在训练完成后的诊断阶段落实；本入口不自动执行第二阶段或test。

```bash
./.venv/bin/python scripts/run_apor_grid_decoder_v2.py status --session "$APOR_V2_SESSION"
./.venv/bin/python scripts/run_apor_grid_decoder_v2.py summarize --session "$APOR_V2_SESSION"
```

若失败，先检查对应failed.json与日志，再在相同session的源码身份仍一致时显式传`--retry-failed`创建新attempt；已成功cell核验后复用。若修复改变源码身份，保留原session，记录原因后使用新的执行身份，不覆盖已有产物。
