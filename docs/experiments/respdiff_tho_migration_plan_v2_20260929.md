# RespDiff-THO 移植方案：论文与发布代码对照

日期：2026-09-29。修订：v2。分支：`codex/resp-diff`。

状态：核心模型、信号适配与CPU合成入口已实现，历史21项定向测试通过；详见[实现与CPU验收记录](respdiff_tho_implementation_20260929.md)。2026-10-03按用户要求完成[方法参数与约定审查](respdiff_parameter_alignment_20261003.md)，采用来源训练/末轮评价设置，并移除额外Local RR选模组件；最新相关17项测试通过。正式训练入口与数据归一化仍待准备。本方案取代 `respdiff_tho_migration_plan_20260929.md`。

## 1. 目标与来源

在现有BCG→THO任务上评估RespDiff，保留论文的 **30 Hz、5秒片段生成、拼接长波形后评价** 路线。网络每次处理150点，项目接口每次交付180秒、18000点；生成上下文与评价窗口分别记录。

- 论文：[Miao等，RespDiff](</mnt/disk_code/marques/paper/resp_rec/reference_paper/2024 - Miao 等 - RespDiff An end-to-end multi-scale RNN diffusion model for respiratory waveform estimation from PPG.pdf>)，5页，首页标注arXiv:2410.04366v1，2024-10-06。已读取全文并核对第3–4页渲染中的图1、式(10–11)、实验设置与表I–II。
- PDF SHA-256：`745336adab2a01286adc45e41846495b3506cdf7b07e8f8ed33953a84ebba173`。
- 代码：`/mnt/disk_code/marques/reference_repos/RespDiff`，提交 `3ff05545c34f67e1ad415e36e4daea80269880b0`。
- 项目参照：W0 `crd_tf102_w`，沿用其数据准入与主体隔离；报告指标建议采用 `w0_final_evaluation_protocol_20260927.md` 的最终五指标。

论文实验对象为BIDMC的53条8分钟PPG/阻抗呼吸记录。移植后的输入模态、目标、划分与评价属于THO适配合同；论文1.18 bpm仅作来源信息，不是本任务验收阈值。

## 2. 论文—代码对照与取舍

| 事项 | 论文证据 | 源码证据 | 移植决定 |
|---|---|---|---|
| 生成/评价长度 | p.3 §IV-A：30 Hz、5秒片段、拼接8分钟；p.4：60秒评价 | `breathing_bidmc_fft.py:56–85,225–259` | 150点生成，36段拼成项目180秒，再执行统一评价 |
| 参考模态 | p.3：impedance pneumography | 变量名co2实际读`ref.resp_sig` | 来源为阻抗呼吸，本任务为THO，不按变量名解释为CO₂ |
| 输入低通 | p.3文字称呼吸与PPG都做1 Hz低通 | 第78–80行仅过滤参考，PPG只重采样 | 按可执行源码：BCG只重采样，训练target做1 Hz低通 |
| 归一化范围 | p.3按“breathing and PPG…respectively”字面为呼吸[-1,1]、PPG[0,1] | 第128–135行明确PPG[-1,1]、呼吸[0,1]，逐5秒处理 | 来源测试按源码；THO主配置保留segment soft-z相对幅值，明确记为适配 |
| 频谱loss | p.3式(10–11)：还原x0后比较FFT幅值，权重0.01 | `model_fft.py:16–51,251–263`为ortho全FFT幅值平方误差 | 精确实现源码；式(11)范数写法未明确代码中的平方/批归约，不能断言逐项一致 |
| 编码器细节 | p.2–3：fine/coarse、多尺度、双向RNN；p.4列fine核1/3/5/7/9/11 | coarse核3/5/7/9/11/13、dilation 1/2/4；RNN 6层/1024 hidden | 论文未完整给出的超参数由源码补全 |
| 融合权重 | 式(7)写λ_ppg | `model_fft.py:183`为可学习[1,192,1]权重 | 保留逐通道参数 |
| FFT消融 | 表II称有/无spectral loss | `model.py:149`与`model_fft.py:188`还差噪声分支step embedding | 主实现采用FFT文件；纯loss消融须固定架构后另注册 |
| 采样方式 | p.3–4：DDPM/DDIM，表II列50/6 NFE | 训练脚本只调用flag=1的DDPM50；FFT文件另有DDIM | DDPM50为质量主路径，DDIM6为后续加速路径 |
| 采样条数 | 论文未明确轨迹平均条数N | 脚本第214–215行N=100、取均值 | N与NFE分开记录，不能把代码N=100当成论文表II已明确的设置 |
| DDIM细节 | 论文未给出完整实现 | 第318行每步对全batch做min-max；eta未实际使用 | 来源DDIM单独对照；THO版需定义保持目标尺度的更新并验收 |
| 训练/选择 | LOSO，未完整列出优化预算 | Adam 1e-4、batch128、400 epochs、280/396降lr、末轮checkpoint | 保留来源训练/末轮选择设置；主体split与评价指标使用项目合同 |
| 评价 | 60秒FFT主峰RR与波形MAE | RR窗60秒/步20秒、分母漏+1；另算整条波形MAE | 使用项目指标函数，修复行为不沿用原错误分母 |

论文“6步生成8分钟用7秒”未配齐硬件、batch和N等条件，不能推算本机耗时。50/6 NFE指一条轨迹的调用数；若N=100，每个条件batch分别需要5000/600次网络调用。

## 3. 主配置与来源校验

**主比较命名为RespDiff-THO。**保留多尺度卷积、双向RNN、扩散与频谱目标、30 Hz/5秒生成、来源训练设置及顺序拼接；适配BCG/THO模态、数据幅值空间、固定split与项目评价。建议首轮只有这一主配置的三seed，正式归一化在数据合同中明确。

**来源行为校验使用synthetic fixture。**对照原forward、loss、DDPM单步以及逐5秒min-max预处理，不自动增加正式实验臂。若另需THO上的`source_minmax`训练对照，应另注册；推理不能读取目标统计恢复幅值。该目标变换消除了跨5秒片段相对幅值，其包络成绩只能描述该版本，不能代替主适配模型的能力。完整BIDMC数值复现属于独立LOSO任务。

## 4. 数据与信号流程

建议主配置按以下步骤实施，变换放入新adapter，保留原数据文件与W0读取器：

```text
冻结parent row：BCG/THO各180秒 × 100 Hz = 18000点
    ↓ 沿用segment soft-z，校验身份/长度/有限性
180秒整窗 Fourier resample → 30 Hz、5400点
    ↓ 仅训练target：八阶Butterworth 1 Hz、filtfilt
按时间拆成36个不重叠5秒片段，每段150点
    ↓ RespDiff生成，各片段N条轨迹取均值
按parent row与chunk_index拼回5400点
    ↓ 对拼接预测做同源1 Hz低通
整窗 Fourier resample → 100 Hz、18000点
    ↓ 与原100 Hz参考比较，执行统一五指标
```

1. 使用现有SciPy的resample、butter和filtfilt。dtype、padding、输出长度与系数参数在实现时固定并验证。整窗处理后切片，滤波支持范围为180秒，区别于原代码8分钟整条记录的边界条件。
2. 输入/target使用项目segment soft-z，不逐5秒重缩放；推理不使用目标最值、能量或包络恢复幅值。保留片段间相对幅值信息，但是否能学到它须实验证实。
3. 低通后的训练target与原100 Hz评价reference分别保留。新增适配必须从头训练；W0与历史数据/结果无需重算。
4. 30 Hz重采样会去除约15 Hz以上输入信息；结果是整套方法适配比较，不能宣称“输入带宽完全相同的纯架构对照”。论文的BCG高频贡献讨论应披露此范围。
5. 先按父级split隔离，再展开chunk。每段保存parent row ID、split、samp_id、源记录、绝对起止sample及chunk_index=0…35。不能对子片段随机重划split。
6. 保持现有parent窗口抽样分布。重叠parent可能带来重复物理片段，登记unique source interval与重复次数；不静默去重改变权重，也不把chunk数当成独立统计样本数。
7. 每个父窗口必须完整收集36段，缺失、重复、错序显式失败。顺序直接拼接，不使用参考做相位/幅值校准。

**任务适配风险：**5秒片段对0.05 Hz呼吸仅覆盖四分之一周期，独立生成可能出现边界不连续、慢变化或努力包络失真；低通平滑不能保证恢复缺失信息。CPU验收加入慢呼吸、幅值调制与拼接边界信号测试，报告保留PCC和包络指标。5秒FFT栅格是0.2 Hz，即12 bpm；频谱loss不直接等价于60秒精细RR优化。

## 5. 网络、损失与推理

### 网络

按`model_fft.py`保留：fine核[1,3,5,7,9,11]；coarse核[3,5,7,9,11,13]，各支32通道，dilation 1/2/4残差卷积；192维可学习融合权重。条件和噪声各192通道，拼接384；6层双向tanh RNN hidden=1024；线性2048→1024→128、decoder 128→512→1。step embedding由128映射192，仅加在条件分支。

保留BatchNorm的track_running_stats=False。eval仍受batch组成影响，验证batch、顺序、分组、尾batch均记入身份。BN仅在条件coarse编码器中：若整个固定条件batch按相同倍数重复以并行采样，理论均值/方差不变，不能笼统认定这种并行必然改变模型。并行前须在固定逐轨迹噪声下验证数值等价、顺序还原与显存；改变条件样本的相对组成仍会改变BN统计。固定batch内的条件编码也可考虑在推理时缓存，须验证与逐步重算等价。

### Loss

随机t∈{0,…,49}，按原调度加噪并还原x0。源码合同为：

```text
L_noise = sum((noise - predicted_noise)^2) / 150
L_fft   = mean((abs(FFT_ortho(x0_hat)) - abs(FFT_ortho(target)))^2)
L_total = L_noise + 0.01 * L_fft
```

单通道batch为B时，L_noise=B×逐元素MSE，而FFT项是batch均值。首轮保留该公式并分别记录三项；physical batch建议128、保留完整尾batch，尾batch相对权重变化属于来源行为。以后改mean或梯度累积须记录优化合同变化；累积不能恢复大batch BN统计。先FP32验证小alpha下x0还原及FFT有限性，AMP另行验收。

### DDPM与DDIM

- DDPM主路径：50步、beta线性0.0001→0.5，建议N=100均值。调度注册buffer，保持数值合同，t=0不读取无用前一时刻项。
- 训练/评价RNG分离。采样身份包含seed/split/parent/chunk/trajectory/step，用于固定末轮checkpoint的可重复评价；batch分组按已记录的推理合同执行。
- `sample_mean`使用eval+inference_mode，逐轨迹累加，避免保存全部轨迹；明确浮点容差和跨环境可重复边界。
- DDIM6有论文加速依据，列为第二阶段可选推理评估。源码的每步全batch min-max会改变幅值空间、耦合样本；THO版应定义eta=0、固定timesteps、保持目标尺度的更新公式，单独登记与来源DDIM的差异。改变sampler不重训，但产生新评价身份。
- 主合同按源码使用400 epochs末轮checkpoint与DDPM50评价。DDIM6作为独立推理评估，不按中途质量替换已确定主结果。

## 6. 训练预算与评价成本

按既有记录规模，10141×36=365076训练chunk，2675×36=96300 validation chunk；实际运行仍须核验父级row集合，数字不是新的准入规则。

**方法预算按来源400 epochs定义，update数随实际数据规模推导。**全chunk batch128每次完整遍历需2853次更新，400 epochs为1141200次更新/seed。此设置有源码依据，但不代表与BIDMC等更新量或等计算成本。

- Adam lr=1e-4、weight decay=0、physical batch128、FP32；seeds 20260811/12/13为项目重复性设置。
- 按完整train chunk集合遍历400轮，第280/396轮结束后学习率×0.1；保留尾batch，不增加早停或梯度累积。
- 使用第400轮完成后的checkpoint，训练后完整validation；项目Local RR和五指标用于报告，不重新选择主checkpoint。
- DDPM50、N=100均值，推理batch64；真实资源验收不会自动修改这些参数。
- 报告updates、实际chunk数/信号时长暴露、wall-time、params、显存和推理成本；W0作为同任务参照，不宣称等算力架构比较。

96300个validation chunk、batch64需要1505个batch；当前串行DDPM50×N100约7525000次去噪网络调用/完整validation。原规模训练及一次完整验证的资源需求仍须测量。若无法承担，应说明开销并明确修订实验，而不是自动缩减来源设置。

## 7. 接口与实施顺序

| 计划新增文件 | 职责 |
|---|---|
| `resp_train/respdiff/model.py` | 来源FFT网络与tensor映射 |
| `resp_train/respdiff/diffusion.py` | loss、调度、sample_mean、随机数接口 |
| `resp_train/respdiff/data.py` | parent→36 chunks、信号处理、身份与拼接 |
| `resp_train/respdiff/experiment.py` | 更新组件、固定checkpoint的完整父窗口评价、产物 |
| `configs/respdiff_tho_v1/experiment.yaml` | 模型/变换/采样/预算/新输出根，未决合同阻止正式运行 |
| `scripts/run_respdiff_tho_v1.py` | CPU检查、工程验收、train/validation入口 |
| `tests/test_respdiff.py` | 数学一致性、信号对齐、生命周期测试 |

复用ResearchV2WindowDataset的父级读取与资格，关闭TF/SST cache。CRDExperiment绑定波形loss，使用独立runner；复用metrics/task.py的Local RR与metrics/final_evaluation.py纯函数。输出`{"waveform": [B,1,18000]}`，按原parent row评价和聚合。

**P0，来源与adapter：**固定paper/code身份；synthetic CPU对照forward、三项loss、一次backward、DDPM单步和state_dict。小fixture测试梯度与失败行为，原规模meta核对参数。验证18000→5400→36×150→5400→18000长度/时间位置、已知正弦/脉冲/慢幅值调制与边界；来源min-max另做测试，主预测路径检查无目标统计输入。

**P1，runner：**临时NPZ/index验证主体隔离、chunk映射、完整重组、五指标资格/分母和中央选窗；验证新identity防覆盖、失败中止与test入口拒绝。固定batch条件下测试可重复采样。

**P2，资源验收：**由用户执行或另行授权；保持L=150、6层/1024 hidden，batch1逐步到训练128/验证64，FP32测forward/backward/update、50步单轨迹，再校准完整N100采样与36段拼接成本。逐项限时，保留失败；据实估计400 epochs、一次完整validation和三seed成本。DDIM6另验收公式与幅值合同。

**P3，正式运行：**确定全部合同后，记录Git/source manifest、resolved config、父子row身份、seeds、命令、环境、模型/优化器、history、checkpoint、逐parent指标和完成/失败receipt。输出独立不可覆盖的 `runs/respdiff_tho_v1/<identity>/seed_<seed>/<attempt>`。真实数据smoke、GPU和正式训练遵循仓库授权；独立test须匹配专项协议及当次授权。W0最终五指标产物确认完成且版本一致后再并表。

## 8. 来源哈希与本轮范围

| 源码 | SHA-256 |
|---|---|
| `model.py` | `48d95640b8e0c7a0ebadfc1efd0975b2bca93e85fd003465ad487e4f357ed59d` |
| `model_fft.py` | `e46ada04c541cf0742656fad283a0455883bd7400700a8db48977cfa678b798f` |
| `breathing_bidmc.py` | `2e15c37f7faf732c2b9d195fcbc77a186b65638df4c199a0d10a6c61a67c0360` |
| `breathing_bidmc_fft.py` | `38b6191e163a37ca6b9dc3832cba7de3b495d007459c4915f3086f77ce07fec8` |

方案调研阶段只完成论文—代码对照；后续已新增核心实现并执行CPU合成更新/采样，当前证据见实现记录。2026-10-03已按来源固定方法参数；正式归一化仍待数据合同确定，既有实验结果不因此改变。

## 9. 完整性与合理性复核

结论：本方案保留核心网络和5秒生成机制，是THO任务适配，尚不构成BIDMC论文数值复现或已验证的源码等价实现。方案中的合理动机与已验证效果必须区分。

- **归一化是实质科学变化。**由逐5秒min-max改为segment soft-z，会改变target的均值、能量、范围及固定alpha调度下的信噪比，也会影响频谱项的数值尺度。数据空间映射须明确记录；synthetic来源测试只证明实现行为，不能证明适配后的实际质量。额外归一化对照属于独立研究，不作为当前复现的强制前置阶段。
- **400 epochs采用来源设置，开销仍须验收。**THO展开chunk数更大，约114万updates/seed是固定方法参数在当前数据规模上的代价。配置符合来源不等于资源可行或效果已证实；不能用test决定是否缩减预算。
- **完整DDPM50×N100评价尚无资源证据。**主合同训练后评价一次；第6节调用数对应当前串行实现，条件缓存和等价轨迹并行可能降低成本，但必须分别验证。
- **公平性有三种不同范围。**同主体划分/同评价窗口/同指标可支持同任务方法比较；5秒与180秒上下文、30 Hz与100 Hz带宽、不同训练目标和数据暴露不能支持严格等信息或等算力的纯架构结论。最终指标对参考及预测统一处理，但不能补回片段归一化或降采样已丢失的信息。
- **完整复现仍缺实际证据。**核心实现的CPU来源forward/loss/采样对照与chunk身份/重组已通过；原规模更新、实际BN分组合同、真实完整validation和三seed结果仍待验证。若声称复现论文1.18 bpm及表II，还需BIDMC LOSO、论文与代码冲突处理、sampler/N及原指标口径的独立复现证据。

源码一致性实现与短片段工程验收服务于来源设置的可运行性；数据身份和归一化映射确定后，按来源训练及末轮评价流程执行。DDIM6保留为有论文依据的可选路径。
