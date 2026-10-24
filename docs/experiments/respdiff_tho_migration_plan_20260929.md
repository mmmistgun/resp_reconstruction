# RespDiff-THO 移植方案

日期：2026-09-29。工作分支：`codex/resp-diff`。

状态：历史准备草案，已由[论文与代码对照后的v2方案](respdiff_tho_migration_plan_v2_20260929.md)取代。下文保留调研过程记录；实施以v2为准。

## 1. 目标与交付顺序

目标是在现有 BCG→THO 任务上建立可追溯的 RespDiff 适配实现，输出与 W0 相同的180秒波形，支持同数据身份、同评价函数的比较。发布代码行为与THO适配变化分别记录，结果命名为 RespDiff-THO。

先交付可变长度模型与synthetic CPU验证，再完成数据/训练/评价入口；原始规模的资源验收决定正式可用的长度、batch和运行预算。此时没有把尚未测量的运行预算写成冻结实验配置。W0及E9等既有训练、指标、来源锁和产物保持原身份。

## 2. 推荐科学设置与待定项

| 项目 | 推荐方案 | 需要明确的影响 |
|---|---|---|
| 数据身份 | 复用W0的BCG/THO数据键、主体划分、资格、row集合与顺序 | 新训练使用原样本身份；PPG→BCG属于模态适配 |
| 时间上下文 | 主比较优先完整180秒 | 保留整窗节律与努力包络；原RNN序列明显变长，必须先验收资源 |
| 输入采样率 | 优先100 Hz，直接保留现有18000点输入信息 | 保留原卷积核采样点数时，其物理时间尺度不同于原30 Hz模型；30 Hz作为需明确批准的资源替代，且会去除约15 Hz以上输入成分 |
| 归一化 | 保留现有segment soft-z，不新增逐5秒或逐窗min-max | 输入/目标尺度改变属于THO适配；不使用目标统计反归一化预测 |
| 主版本 | 优先发布的FFT版本，架构和loss作为一个整体身份 | 噪声分支不加step embedding；普通版本作为来源一致性验证profile，首轮不自动扩成双臂实验 |
| 网络规模 | 保留多尺度卷积、6层双向tanh RNN、hidden=1024、原decoder | 不按W0参数预算缩窄；参数量、显存与时延据实报告 |
| 扩散调度 | 原50步DDPM，线性beta=[0.0001,0.5] | 首轮不引入DDIM加速与额外归一化 |
| 训练loss | 先精确保留发布公式及FFT权重0.01 | 不替换为W0任务loss；否则研究问题同时改变 |
| checkpoint | 使用完整validation的项目Local RR，严格最小、并列最早 | 属于项目适配，区别于原末轮checkpoint；随机推理必须固定验证随机数方案 |
| 报告指标 | 建议采用20260927最终五指标实现 | validation selector仍使用历史Local RR；报告指标与selector分别命名、分别记录 |
| 推理采样数 | N=100为来源参考，实际validation/final N在资源验收后一次确定 | 不先用少采样选模型，再悄悄改为另一采样合同；若二者N不同，预先注册并冻结 |
| seeds/预算 | 正式候选三seed：20260811/12/13；预算验收后确定 | 原400 epochs与W0的80 epochs不构成天然等价预算；不得把尚未测量的完整矩阵视为可执行承诺 |

100 Hz方案的优势是主比较保留同一输入信息，代价是每条去噪序列18000点，为原150点的120倍。30 Hz整窗为5400点，仍为原长度36倍。两种选择都不能称为与原5秒训练设定完全相同。若100 Hz整窗不能满足资源预算，应先报告30 Hz适配的带宽与时间尺度变化，再确定新的主合同；不自动切5秒分块。分块会新增边界、上下文和拼接问题，对包络任务尤其重要。

## 3. 模块与接口

以下为计划新增文件，尚未实现：

| 路径 | 职责 |
|---|---|
| `resp_train/respdiff/model.py` | 多尺度编码器、双向RNN与噪声预测，明确source_fft/source_plain两个来源profile |
| `resp_train/respdiff/diffusion.py` | 调度buffer、加噪、原公式loss与DDPM采样；显式随机数接口 |
| `resp_train/respdiff/data.py` | 复用research_v2数据读取与row身份核验，控制waveform-only输入和可选采样率适配 |
| `resp_train/respdiff/experiment.py` | 独立训练循环、完整validation采样、selector、checkpoint与失败/完成记录 |
| `configs/respdiff_tho_v1/experiment.yaml` | 通过准备阶段后落地的专属配置；正式未决字段应显式阻止正式运行 |
| `scripts/run_respdiff_tho_v1.py` | synthetic CPU检查、后续GPU工程验收、train与validation命令 |
| `tests/test_respdiff.py` | 模型/数学/随机数/数据适配/生命周期的最小定向验证 |

模型核心接口分开表达训练与生成：

```python
epsilon = denoiser(condition, noisy_target, step)  # [B,1,L]
losses = diffusion.training_loss(condition, target, generator=generator)
prediction = diffusion.sample_mean(condition, n_samples=n, sampling_identity=identity)
# 项目适配层输出 {"waveform": [B,1,18000]}，供既有指标函数消费。
```

不把昂贵的采样藏在普通forward里。训练只做一个随机时间步的噪声预测；validation显式调用sample_mean，使用eval和inference_mode。N条轨迹逐条累加均值，避免保留[B,N,1,L]全量张量。原始轨迹保存只作为显式诊断功能。

复用点已经核对：

- `resp_train/data/factory.py:66` 的 `build_tho_data` 构造train/validation；`ResearchV2WindowDataset`输出x/target/meta，保持100 Hz原始row语义。
- `resp_train/data/research_v2.py:75` 表明TF cache是可选项。新配置显式设置相关cache为空、tf_representations为空，不继承W0配置中的cache/output路径。
- `resp_train/metrics/task.py` 的Local RR逻辑用于checkpoint选择；`resp_train/metrics/final_evaluation.py` 的evaluate_window/汇总与中央窗选择用于报告。
- `CRDExperiment.train` 固定构造RespirationTaskLoss，并将训练和validation记录绑定sync/effort。仅替换_build_model不足以接入，因此编写小型独立runner，不修改冻结CRD流程。
- 最终W0评价runner包含W0来源锁和checkpoint限定，仅复用其纯指标函数，另建RespDiff来源校验。

如批准30 Hz适配，先按原18000点加载与验证，再用成熟重采样函数将输入/训练目标转换为5400点；输出恢复为18000点，评价参考始终使用原100 Hz目标。精确滤波器、边界方式和长度对齐纳入配置与合成信号测试；不直接把dataset的duration_samples改为5400。

## 4. 保真与工程修正

### 来源一致性

记录上游提交和四个Python文件SHA-256，保留层名称与tensor shape映射。将上游必要模型定义作为只读参考做对照，移除未使用import与脚本副作用。优先使用现有torch/numpy/scipy环境，不安装新依赖；CPU测试调用主仓库现有解释器并显式设置新worktree的PYTHONPATH。

两个profile只验证各自发布行为。FFT profile关闭噪声分支step embedding，plain profile开启；不把两者结果解释成单因素loss消融。

### Loss与batch

原单通道噪声loss为sum(error²)/L，FFT为batch均值。先实现原公式，并显式返回噪声项、FFT项和总loss用于记录。用固定噪声与时间步验证不同batch及尾batch的缩放，禁止静默改为mean。

若采用梯度累积，必须定义一次optimizer update中的数学归约，并与同数据大batch目标比较。BatchNorm在小physical batch和大batch下统计不同，梯度累积不能宣称恢复大batch等价；loss归约正确与归一化统计等价是两个问题。physical/effective batch与尾batch策略在正式训练前分别冻结。

### 采样与随机数

调度数值保持与上游对应，设备相关数据注册buffer；边界t=0不读取不需要的前一时刻调度。训练RNG与validation采样RNG分离。验证使用由训练seed、split、row ID、轨迹编号、噪声用途导出的稳定随机数身份，固定epoch间验证噪声，降低selector受随机采样噪声干扰。不能使用Python进程随机化hash作为持久seed。

原BN的track_running_stats=False保持。validation batch大小、row分组、顺序固定记录；同batch条件下采样可重复性必须验证，但不承诺改变batch后输出不变。为提高速度，将N条轨迹并入batch会改变BN统计，不作为透明优化。

调度/FFT/还原x0中小alpha的除法存在数值放大，默认FP32验证，AMP须另验收。任何输入、target、loss、梯度、checkpoint、逐步采样状态或指标非有限都明确失败并保留现场。

## 5. 分阶段验收

### P0：核心移植与CPU定向验证

使用短synthetic tensor完成模型接口、loss和采样。来源对照重点是相同权重/输入/step下噪声输出，以及相同固定噪声下loss和一步反向更新；依赖/设备迁移差异必须有明确容差。资源较小的fixture检验真实梯度与optimizer更新、state_dict严格回载、非法shape/非有限显式失败、可重复采样和N条均值归约。完整规模可用meta tensor核对参数shape与数量；小模型测试不能替代完整模型工程验收。

可由Codex执行，限数分钟内CPU、synthetic/disposable fixture。完成标准是来源一致性和失败行为有证据，不包含真实数据或GPU性能结论。

### P1：数据与runner接通

使用临时NPZ/index fixture走通train/validation、完整row覆盖、checkpoint selector、中央窗选择与汇总；测试新输出拒绝覆盖、失败中止与test入口拒绝。新data adapter必须从dataset实际rows回核identity，避免过滤器二次筛选后悄悄改变集合。

不在此阶段读取真实数据。准备正式数据来源核验代码与命令，后续真实运行记录resolved config、源文件哈希、row身份、模型/优化器、完整history、环境和命令。

### P2：原始规模synthetic资源验收

由用户执行或另行明确授权。保留hidden=1024、6层等原规模，在目标GPU上按150→5400→18000点顺序验证；一次只跑一项，先batch=1、FP32，测前向/反向/optimizer更新/峰值显存及一次完整50步采样，长项使用有界运行预算。首次结果已显示超预算则停止更大项，保留失败记录。

先记录每个batch一条完整轨迹的时间，再估计全validation N=100、每epoch验证与三seed总成本，报告估算假设。确认可行后才校准完整N和候选batch/精度。不能只凭训练step时间估计扩散validation总耗时。

通过条件包括有限输出/梯度、有效参数更新、明确显存余量，以及满足用户接受的单轮/全流程时间预算。资源上限未指定时提交测量结果供确定，不默认长时运行或占用现有E9设备。

### P3：冻结适配合同并准备正式运行

根据P2结果一次确定采样率、context、physical/effective batch、精度、loss归约、optimizer/schedule/最大updates、早停、完整validation频率、N/steps及三seed矩阵。原Adam 1e-4、400 epochs与W0训练合同列为预算参考，不自动混合成未经验证的训练配方。

优先以Git提交、resolved config、source manifest和receipt记录身份；只有存在额外跨运行证据冻结需求才建正式锁。新输出建议为worktree内 `runs/respdiff_tho_v1/<profile>/seed_<seed>/<attempt>`，失败目录保留。正式train/validation与真实data smoke由用户执行或另行授权。独立test后续须有匹配专项协议、固定checkpoint身份及当次授权。

## 6. 影响与本轮产物

新增RespDiff需要从头训练。现有数据、split、W0/E9参数与历史指标均不改动，因本次准备无需重算旧结果。比较采用最终五指标时引用相同版本的W0评价产物，若来源尚未确认完成则先核实，不能把历史五指标表换名复用。

最大的未决项是长序列双向RNN加100轨迹采样的资源可行性。100 Hz完整输入有利于同输入信息比较，30 Hz有利于保留原模型时间尺度并降低序列长度；二者各有成本，不能在运行失败后静默切换并继续同一实验identity。

本轮完成接口核对与本文。尚未新增模型/runner/config或执行模型测试；下一交付是P0的可审查核心实现及CPU验证结果。
