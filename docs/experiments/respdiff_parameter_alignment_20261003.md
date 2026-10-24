# RespDiff 方法参数核对与数据适配边界

日期：2026-10-03。状态：按用户要求，方法参数以论文为先，论文未列出的实现细节由固定发布源码补齐；已完成来源约定审查，并移除移植入口中额外的Local RR选模组件。真实数据/GPU运行未开放。

本文是当前参数入口，覆盖[移植方案v2](respdiff_tho_migration_plan_v2_20260929.md)中训练预算、验证频率和checkpoint选择的初始提案。数据身份、归一化和统一报告口径继续按数据适配范围管理。

## 来源与优先级

1. 本地论文：[RespDiff，arXiv:2410.04366v1](</mnt/disk_code/marques/paper/resp_rec/reference_paper/2024 - Miao 等 - RespDiff An end-to-end multi-scale RNN diffusion model for respiratory waveform estimation from PPG.pdf>)，重点复核第2–4页方法、实验设置及表II。
2. 作者公开代码：`/mnt/disk_code/marques/reference_repos/RespDiff`，commit=`3ff05545c34f67e1ad415e36e4daea80269880b0`。文件SHA-256沿用`resp_train/respdiff/provenance.py`并由测试回核。
3. 发布代码未显式传递的库默认参数注明为默认值；不把它们写成论文明确报告的超参数。
4. 论文/代码的预处理冲突继续显式记录，不以“全部按论文”掩盖实际差异。

## 固定方法参数

| 参数 | 采用设置 | 证据 |
|---|---|---|
| 模型采样率/片段 | 30 Hz、5 s、150点 | 论文p.3 §IV-A；脚本第56–85行 |
| 细尺度卷积核 | 1/3/5/7/9/11，各32通道 | 论文p.4 §IV-D列核大小；通道数见`model_fft.py:79–94` |
| 粗尺度卷积 | 核3/5/7/9/11/13，dilation 1/2/4，各32通道 | `model_fft.py:101–140` |
| 双向RNN | tanh，6层，hidden=1024 | 论文规定双向RNN；具体值见模型第53–64行与训练脚本第160行 |
| 融合/读出 | 384输入；2048→1024→128；decoder 128→512→1 | `model_fft.py:53–75,171–190` |
| 条件融合系数 | 可学习192通道权重，初始化为1 | `model_fft.py:183` |
| 扩散步嵌入 | 128维→192维；FFT主版本仅注入条件分支 | `model_fft.py:143–190` |
| BatchNorm | `track_running_stats=False` | `model_fft.py:107–109` |
| 训练扩散调度 | 50步，线性beta=0.0001…0.5 | `model_fft.py:198–204` |
| 噪声loss | 单通道`sum(error²)/L` | 论文式(4)给出噪声目标；具体归约见代码第261行 |
| 频谱loss | ortho全FFT幅值平方误差，mean；系数0.01 | 系数见论文p.3 §III-C；FFT与归约见代码第16–51、262–263行 |
| 优化器 | Adam，lr=1e-4 | `breathing_bidmc_fft.py:167`；论文未列出 |
| Adam默认项 | betas=(0.9,0.999)、eps=1e-8、weight_decay=0、amsgrad=false | 来源调用的库默认值，在新配置显式保存，并与本环境原调用核对 |
| 训练batch | 128，shuffle=true，保留尾batch | 脚本第153行；drop_last=false为DataLoader默认 |
| 训练轮数 | 400完整epoch | 脚本第170行；论文未列出 |
| 学习率调度 | MultiStepLR；milestones=[280,396]，gamma=0.1；epoch结束时step | 脚本第172–175、196行 |
| 训练方式 | FP32、每batch一次更新；无梯度累积、梯度裁剪或早停 | 来源脚本第177–196行执行路径 |
| 主checkpoint | 第400轮完成后的末轮checkpoint | 脚本第201–202行；best_val_loss声明未参与选择 |
| 主推理 | DDPM50，每输入100条轨迹算术平均 | 论文表II包含50 NFE；脚本第214–215行明确DDPM与N=100 |
| 推理batch | 64，shuffle=false，保留尾batch | 脚本第154行 |
| 开发集评价时机 | 训练完成后评价固定末轮模型 | 来源脚本训练后评价；在本项目映射至validation而非LOSO留出subject |

论文还报告6步DDIM，作为可选独立推理评估保留。当前DDPM主配置不自动切换sampler；来源DDIM每步全batch min-max的幅值影响需单独定义和验证。论文“7秒生成8分钟”未配齐资源与N等条件，不作为本机时延承诺。

## 属于本数据集或项目协议的设置

| 项目 | 当前处理 |
|---|---|
| 输入/目标 | BCG→THO；不换为论文的PPG→阻抗呼吸 |
| subject/session与split | 沿用项目既有隔离；不改成BIDMC 53折LOSO |
| 父窗口 | 保留100 Hz、180秒、18000点身份，内部拆36个150点片段再重组 |
| 归一化 | `formal_normalization: null`；保留source_minmax与segment_soft_z两个开发profile。前者有源码依据，后者为努力幅值任务适配，正式选择须明确 |
| 输入/target预处理 | 论文写两路1 Hz低通，源码只过滤target；当前adapter遵循可执行源码。该差异与BCG输入带宽有关，继续单独记录 |
| 最终报告指标 | 使用项目统一五指标，与W0同口径；Local RR可作诊断，不改变来源末轮checkpoint规则 |
| 训练/评价规模 | 由实际准入父row及chunk映射推导，不能硬编码论文53人或BIDMC窗口数 |
| 随机种子与产物 | 保留项目三seed与不可覆盖identity；作者未报告这些seed，不归因于论文 |
| 设备和实现环境 | 实际GPU/软件版本在工程验收记录，不硬编码源码cuda:2 |

训练轮数、主采样数、batch与调度现在有来源固定值；剩余待定的是正式数据/归一化合同和资源执行准备。数据与指标处理不因本次参数核对被重新定义，W0等历史结果无需重算。

## 计算规模与运行边界

若继续使用10141训练父窗口、2675 validation父窗口，并各展开36段：

- train chunk=365076；batch128每epoch=2853次更新；400 epochs=1141200次更新/seed。
- validation chunk=96300；batch64=1505个batch；DDPM50×100轨迹=7525000次去噪网络调用/一次完整validation（当前串行轨迹实现）。
- 这些是给定规模的算术推导，未读取真实数据回核行数，也没有本机耗时测量。主合同训练后完整评价一次。

资源不足时应报告实测开销，不能静默缩小原模型、batch、400 epochs或N=100并继续沿用相同实验身份。任何加速近似或训练预算修改须明确为修订。CPU合成检查仍用小RNN、单次更新、N=1，是开发测试配置，不是正式复现设置。

## 实现与验证

- `configs/respdiff_tho_v1/experiment.yaml`写入完整训练/推理默认值，并保留`formal_enabled: false`。
- 新增`resp_train/respdiff/settings.py`：来源设置校验、Adam/MultiStepLR构造，以及仅根据样本数推导更新数/调用数的函数。独立synthetic字段允许小规模测试，方法配置漂移显式失败。
- 合成CLI复用来源Adam构造，评价固定的合成checkpoint并输出指标；正式主配置使用来源末轮checkpoint。
- 新测试从已核验上游脚本AST中提取优化器、epoch和scheduler赋值，在标量CPU模型上逐epoch核对400次调度；同时核对源码batch/N、数据规模推导及非法参数修改失败。

执行：

```bash
cd /home/marques/.codex/worktrees/respdiff-paper/resp_reconstruction
PYTHONPATH=. /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python -m pytest \
  tests/test_respdiff_settings.py \
  tests/test_respdiff.py::test_cpu_cli_completion_and_failure_receipts -q
```

本节参数对齐时的结果为**12 passed in 11.95s**。同日完成下方约定审查后，更新的相关测试结果为**17 passed in 12.72s**。既有21项来源/适配测试的完整结果见9月29日CPU验收记录；均未访问真实波形或执行GPU。

## 来源约定审查

用户要求：论文和发布代码未采用的额外训练/选模规则，不作为RespDiff复现的默认要求。本次审查区分来源方法、BCG适配候选与复现工程，实际设置仍由本文件上方来源表和配置共同定义。

### 训练与选模

来源脚本`breathing_bidmc_fft.py:167–215`执行Adam训练400 epochs，在训练循环结束后保存state_dict并评价。`best_val_loss`只是未使用的变量；源码没有计算逐epoch validation排名或据其保存best checkpoint。

| 审查项 | 来源情况 | 当前处理 |
|---|---|---|
| proxy验证子集、分层proxy抽样 | 论文/代码未采用 | 不进入默认训练流程；此前仅为讨论建议 |
| proxy top-K=5再完整验证 | 论文/代码未采用 | 不进入默认流程，未实现过此筛选器 |
| 每epoch/每N updates完整validation选优 | 来源仅训练后评价 | 使用末轮checkpoint和训练后评价，不以其他周期验证替代proxy |
| 按Local RR选择best checkpoint | 来源使用末轮 | 删除`LocalRRSelector`和`select_validation_checkpoint`及CLI依赖，指标仅用于报告 |
| early stopping | 来源没有 | 保持关闭 |
| 6400或15600固定update预算 | 论文/代码未定义 | 不作为默认预算；来源400 epochs保持，更新数由实际数据量推导 |
| 强制额外归一化桥接实验 | 来源没有 | 归一化属于数据合同选择，不把额外对照训练列为复现前置步骤 |

### 方法改动与数据适配建议

| 讨论中的约定 | 是否属于原方法 | 默认边界 |
|---|---|---|
| 20 Hz、Kaiser降采样 | 来源为30 Hz、Fourier resample | 属于BCG适配候选，不当作原方法固定要求 |
| 30秒窗、15秒hop、reflect padding、tapered OLA | 来源为5秒不重叠片段顺序拼接 | 属于用户提出的BCG适配方向，尚未实现或写入当前来源配置 |
| padded父窗口共享4200点噪声场 | 来源逐片段独立高斯噪声 | 属于新增重叠重建设计，不作为默认要求 |
| 每个父窗口13块组成推理batch | 来源推理batch64 | 属于适配推理分组建议，不替代来源batch64 |
| batch-invariant loss，参考B=128重标定 | 来源为sum/L噪声项与mean频谱项 | 当前保留原loss及batch128；不自动改写objective，尾batch效应据实记录 |
| DDIM6 | 论文表II有依据，源码提供DDIM入口 | 可以作为有出处的可选采样设置；当前实现仍是来源主脚本DDPM50 |
| DDIM6、N=1的组合 | 论文未明确N；主脚本采用N=100 | N=1不能归因于论文；当前来源推理配置保留N=100 |
| 删除DDIM每步全batch min-max | 与发布DDIM代码不同 | 属于幅值任务适配，不声称为源码原行为；尚未实现DDIM路径 |
| parent soft-z、取消target 1 Hz低通 | 与来源逐片段min-max、target低通不同 | 属于监督空间适配，未作为自动默认修订；正式归一化仍待确定 |

这张表不擅自撤销用户提出的BCG适配方向，而是标明其证据属性。必要的数据接入包括BCG/THO数据键、既有主体split、父窗口身份、共同评价口径；这些要求与作者的BIDMC数据设置分别记录。任何已明确选择的适配应独立命名和记录，不借“复现”自动添加其他实验规则。

### 保留的工程实现

固定seed、来源哈希、配置记录、非有限失败、防覆盖、synthetic测试和batch=1索引兼容修复用于可重复运行及数据安全。它们不增加超参数搜索、checkpoint排名或新的科学指标。现有keyed-noise保持独立标准高斯抽样，没有实现跨重叠chunk共享噪声场；固定seed来源标为项目工程设置。

移除选模组件后执行：

```bash
PYTHONPATH=. /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python -m pytest \
  tests/test_respdiff_settings.py \
  tests/test_respdiff.py::test_complete_validation \
  tests/test_respdiff.py::test_cpu_cli_completion_and_failure_receipts -q
```

结果：**17 passed in 12.72s**。覆盖来源训练设置、拒绝未注册proxy/top-K/update预算、完整父窗口指标与合成完成/失败回执。没有重跑历史实验或改变W0等已冻结结果。

## 工作树恢复

此前托管worktree已归档清理；14个源码/配置/文档/测试文件保存在Git快照`0bda24e7243a29ee5b8f7226165becd6b7aff08b`。本次从该快照恢复到：

- 路径：`/home/marques/.codex/worktrees/respdiff-paper/resp_reconstruction`。
- 分支：`codex/respdiff-paper-settings`。

恢复保留了原移植文件，未改动当前主工作区。后续使用此路径继续参数审查与工程实现。
