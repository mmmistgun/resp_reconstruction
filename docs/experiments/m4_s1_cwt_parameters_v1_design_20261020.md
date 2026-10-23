# M4-S1 的 CWT 参数与模块消融实验

设计 ID：`m4-s1-cwt-parameters-v1-20261020`。日期：2026-10-20。

状态：28配置/84次训练的模型适配、独立训练入口与定向测试已实现，**80项定向CPU测试通过（223.18秒）**。用户于2026-10-20追加授权正式train/validation数据与缓存兼容性核验、GPU合成验收，明确暂不启动正式训练。当前冻结版本用于上述验收；阶段产物和结果另行记录。正式训练与research-test未开放。

## 1. 研究问题与基线

本轮研究 **M4-S1 的重建质量如何随 CWT 表示及关键模块变化**，并量化节律、包络和波形相关性之间的取舍。四个问题分别由独立子矩阵回答：

1. Morlet μ 与每倍频程尺度密度 voices 的联合设置如何影响重建？用完整 3×4 矩阵检查两者的交互。
2. 固定 M4-S1 的物理读取位置，CWT 时间池化精度是否影响重建？
3. 哪些频带范围适合当前模型，扩大频带和采用单频带分别带来什么变化？
4. CWT条件分支、时频自适应权重、频率bias、BiMamba2、波形patch注意力池化及池化相对位置表示各自提供什么增量？

用户于 2026-10-20 确认：M4-S1 对应 `LinearFusionModel("L1")`，即 10 月 19 日简化版，保留交叉频率 bias、Δt bias 为零，拼接多头后使用一个零初始化输出投影，直接与时域 token 相加，参数量 **1,046,945**。定义见 [linear fusion 协议](m4_linear_fusion_v1_protocol_20261019.md)。

基准结构：100 Hz、180 秒输入；1 秒 patch、0.5 秒 hop、359 个 token；D96、4 头交叉注意力、32 通道 CWT 编码器、6 层 BiMamba2、原评分/值位置路径、局部波形解码及 positive Hann overlap-add。CWT参数组保持此结构；模块组只作第2.2节规定的替换。所有配置保持时域输入、patch几何及波形解码合同。

代码基点：`8d575be6ee32b7d40d3809e0e94569e48dfaebb1`。

- 分支：`codex/m4-s1-cwt-parameters-v1`。
- worktree：`/mnt/disk_code/marques/resp_reconstruction/.worktrees/m4-s1-cwt-parameters-v1`。
- 设计清单：`configs/m4_s1_cwt_parameters_v1/design.yaml`。
- 新产物根：`/mnt/disk_code/marques/resp_reconstruction/runs/m4_s1_cwt_parameters_v1/`；运行时创建独立 session。

前序 [CWT-APOR v2](cwt_apor_v2_results_summary_20261002.md) 提供参数范围与变换实现依据。本轮采用相同范围，完整保留各设置；结论对象为 M4-S1。本设计是在已知历史研究测试结果后形成的开发实验，后续复用研究测试集的评价保持这一证据定位。

## 2. 矩阵与预算

基准 `B`：完整M4-S1，μ=13.4、voices=12、池化50点（0.5秒）、名义频带0.03125–8 Hz、97×360表示。全部配置使用三个训练seed：**20260812、20260813、20260814**，包括B在内均在本轮独立session中从头训练。

| 子矩阵 | 设置 | 其他设置 | 唯一配置数 | 相对 B 新增训练 |
|---|---|---|---:|---:|
| Q：小波与尺度密度 | μ=6/13.4/20 × voices=4/8/12/24 | 池化0.5秒、上限8 Hz | 12，含 B | 11×3=33 |
| P：时间池化 | 0.25/0.5/1秒 | μ13.4、voices12、上限8 Hz | 3，共享 B | 2×3=6 |
| C：频带上限 | 0.8/2/4/8/12/20 Hz | μ13.4、voices12、池化0.5秒 | 6，共享 B | 5×3=15 |
| H/L：频带组成 | L≤0.8 Hz、H=(0.8,8] Hz、L+H | 同基准小波与池化 | 3，共享 C_0p8、B | 1×3=3 |
| A：核心模块消融 | NoTF、UniformTFReadout、NoFrequencyBias、NoMamba、UniformPatchPool、NoPoolPosition | 基准CWT设置；NoTF不读取CWT | 6 | 6×3=18 |
| A：位置路径补充 | ScorePositionOnly、ValuePositionOnly | 基准CWT设置 | 2 | 2×3=6 |
| B：完整基准训练 | 完整M4-S1 | 基准CWT设置 | 已计入Q | 1×3=3 |

共 **28 个唯一配置×3 seed=84次新训练**。CWT参数组20配置、核心模块组7配置（21次训练）、位置路径补充2配置（6次训练），共享完整基准B，因此20+7+2−1=28。H和C_0p8从参数组共享为补充频带对照；包含它们的模块汇总为11配置，等价计数为20+11−3=28。正常完成时训练预算下限为2,520 cell-epochs/201,600 optimizer updates，上限6,720 cell-epochs/537,600 updates；实际耗时与显存需工程阶段测量。

本轮包含μ×voices的3×4矩阵、池化评分位置×值位置的2×2矩阵，以及围绕B的其余分组对照。模块消融使用基准CWT设置；不与20组CWT设置作全交叉。各子矩阵的最佳设置不能直接拼成一个已验证的联合配置。

### 2.1 CWT参数组：20配置

| ID | μ | voices | 池化点数 | 频带上限/选择 |
|---|---:|---:|---:|---|
| Q_mu6_v4 | 6 | 4 | 50 | 8 Hz |
| Q_mu6_v8 | 6 | 8 | 50 | 8 Hz |
| Q_mu6_v12 | 6 | 12 | 50 | 8 Hz |
| Q_mu6_v24 | 6 | 24 | 50 | 8 Hz |
| Q_mu13p4_v4 | 13.4 | 4 | 50 | 8 Hz |
| Q_mu13p4_v8 | 13.4 | 8 | 50 | 8 Hz |
| B | 13.4 | 12 | 50 | 8 Hz |
| Q_mu13p4_v24 | 13.4 | 24 | 50 | 8 Hz |
| Q_mu20_v4 | 20 | 4 | 50 | 8 Hz |
| Q_mu20_v8 | 20 | 8 | 50 | 8 Hz |
| Q_mu20_v12 | 20 | 12 | 50 | 8 Hz |
| Q_mu20_v24 | 20 | 24 | 50 | 8 Hz |
| P_025 | 13.4 | 12 | 25 | 8 Hz |
| P_100 | 13.4 | 12 | 100 | 8 Hz |
| C_0p8 | 13.4 | 12 | 50 | ≤0.8 Hz，L |
| C_2 | 13.4 | 12 | 50 | ≤2 Hz |
| C_4 | 13.4 | 12 | 50 | ≤4 Hz |
| C_12 | 13.4 | 12 | 50 | ≤12 Hz |
| C_20 | 13.4 | 12 | 50 | ≤20 Hz |
| H | 13.4 | 12 | 50 | (0.8,8] Hz |

### 2.2 模块组：核心7配置，位置路径补充2配置，共享频带2配置

模块组围绕条件增量、读取权重、显式序列主干和patch汇聚设计，共有模块使用B的初始化与训练合同。实现参考 [原始M4组件实验](m4_components_v1_protocol_20261008.md) 及 `scripts/m4_pool_position_model.py` 中评分/值位置路径的定义。

| ID | 名称 | 相对B的改动 | 与参数组的关系 |
|---|---|---|---|
| B | 完整M4-S1 | 完整条件编码、交叉注意力、频率bias、单输出投影、BiMamba2及patch注意力池化 | 共享B |
| A_NoTF | 纯时域 | 移除整个CWT条件编码/读取/融合分支，直接将时域token送入BiMamba2 | 新增 |
| A_UniformTFReadout | 均匀时频读取 | 将各头的F×2注意力权重固定为1/(2F)，保留原CWT encoder、V和输出投影 | 新增 |
| A_NoFrequencyBias | 内容注意力 | 移除频率bias网络，保留Q/K/V、单输出投影及物理时间对齐；Δt bias保持零 | 新增 |
| A_NoMamba | 去BiMamba2 | 六层BiMamba2替换为Identity，保留时域CNN、条件分支及解码器 | 新增 |
| A_UniformPatchPool | 均匀patch池化 | 时域patch注意力权重替换为均匀权重，移除score参数，保留位置值与输出映射 | 新增 |
| A_NoPoolPosition | 去池化位置表示 | 同时移除评分位置项和值位置项，保留内容注意力score、CNN和输出映射 | 新增 |
| A_ScorePositionOnly | 仅评分位置 | 保留评分位置项，移除值聚合的位置项，保留内容注意力与位置网络 | 新增，位置路径补充 |
| A_ValuePositionOnly | 仅值位置 | 移除评分位置项，保留值聚合的位置项，保留内容注意力与位置网络 | 新增，位置路径补充 |
| H | H-only（补充） | 条件表示为(0.8,8] Hz的41尺度 | 共享H |
| C_0p8 | L-only（补充） | 条件表示为≤0.8 Hz的56尺度 | 共享C_0p8 |

UniformTFReadout对每个patch、每个head取 `context_h = sum(V_h(encoded[f,r]))/(2F)`，拼接heads后经原 `attention.output` 残差相加；V为原带偏置Linear(32,96)，输出为原零初始化Linear(96,96)。保持原物理F×2取样位置、V/output初始化和float32聚合。先按B构造保证RNG一致，再移除不参与固定权重前向的Q/K及坐标bias模块；最终state_dict和optimizer只保留活动参数。参数量减少须报告，但值路径的参数化与B一致。

B−UniformTFReadout检验完整学习权重机制（包含内容与频率bias）的增量。NoFrequencyBias−UniformTFReadout在相同值路径下检验内容自适应权重；B−NoFrequencyBias定向检验频率bias。工程测试须在打开输出投影后，核对该臂与原读取器强制均匀权重时的前向及encoder/V/output梯度。

UniformPatchPool保留 `project(mean(features)+mean(position))`；位置参与原注意力评分的作用随均匀池化一起退出，位置值路径保留。NoMamba检验显式序列主干的增量，其余CNN、CWT及整窗GN仍可提供时间依赖。各模块臂参数量在实现阶段由正式模型构造核验，不能套用原双投影M4的参数量。

NoPoolPosition使用 `w = softmax(score(features))`，输出 `project(sum(w*features))`。评分中的位置项和值聚合中的位置项同时移除，位置网络退出最终模型；原stem、score、project及其他模块保持同seed共有初始化。该臂相对B测量两条位置路径的联合增量，保留内容自适应池化。它与UniformPatchPool是分别相对B的两个对照。

### 2.3 池化位置路径：完整2×2矩阵

令patch内时域特征为h_i、位置网络输出为p_i，评分层为a(h_i)=w_s^T h_i+b_s，输出映射为project。评分位置路径只加入 `w_s^T p_i`，避免重复加入评分bias；值位置路径在加权聚合中加入p_i。

| 配置 | 评分位置 s | 值位置 v | 池化定义 |
|---|---:|---:|---|
| B（Full） | 1 | 1 | α=softmax(a(h)+w_s^T p)，project(Σα_i(h_i+p_i)) |
| A_NoPoolPosition | 0 | 0 | α=softmax(a(h))，project(Σα_i h_i) |
| A_ScorePositionOnly | 1 | 0 | α=softmax(a(h)+w_s^T p)，project(Σα_i h_i) |
| A_ValuePositionOnly | 0 | 1 | α=softmax(a(h))，project(Σα_i(h_i+p_i)) |

四臂均保留内容自适应权重。仅评分位置/仅值位置均保留原位置网络及其同seed初始化，通过对应活动路径训练；其余共有模块与B逐张量配对。NoPoolPosition的最终模型移除位置网络。CWT设置、模型其他模块、训练与选点合同在四臂中相同。该子矩阵共4×3=12个cell，已计入总预算。

新增两臂使评分路径、值路径的条件效应及交互可以分别估计。空间卷积与逐位置映射的对照仍需独立的通道和归一化设计。

## 3. CWT 与 M4-S1 的接口

### 3.1 变换定义

沿用 `resp_train/paper_evidence/cwt_time_frequency_v1/features.py` 的变换数学和 ssqueezepy 0.6.6：Morlet、reflect padding、先 `log1p(abs(CWT))`，再不重叠均值池化。使用原 M4-S1 冻结的 BCG 输入键及预处理；数据适配时逐项核对旧 CWT cache 的输入身份。

- μ×voices 基于相同名义八 octave 网格，尺度数分别为33/65/97/193。每个 μ 使用自身实际映射频率。
- B 精确保留历史尺度数组和重复离散中心；频率升序排列保持稳定排序，不去重。
- C_0p8/C_2/C_4 与 H 按 B 的实际中心频率取子集。C_12/C_20 只向高频延伸原尺度数组，保留既有低频格。
- 频带标签为筛选上限，报告实际最高中心频率；例如20 Hz标签不意味着一定存在20 Hz中心。
- 时间帧数为720/360/180，中心坐标为 `(k*p+(p-1)/2)/100` 秒，p为池化点数。
- 保存 scales、实际频率、排序、重复中心数、时间坐标、shape、库版本和变换哈希。非有限值显式失败。

### 3.2 固定两个物理读取位置

原实现将 `cwt_pool_samples=50`、patch hop和 `unfold` 步长绑定，并要求97尺度元数据。本轮独立适配器将原生CWT表示与固定物理读取网格分开，保留原实验入口及其校验。

第 j 个 patch（j=0…358）的中心为 `(50*j+49.5)/100` 秒；两个读取位置固定为：

`t[j,r] = (50*j + 24.5 + 50*r)/100` 秒，r=0,1。

先在各配置原生 F×T 网格上执行原 CWT encoder，再在编码特征的时间轴上读取这两个位置。对池化点数 p，连续帧坐标为：

`q[j,r] = (50*j+24.5+50*r-(p-1)/2)/p`。

采用相邻帧线性插值，超出首尾中心时使用最近端点；坐标和权重以float32计算，插值累计后回到原特征dtype。p=50时 q=j+r，走整数 gather 路径；与原连续两个时间格严格对应。p=25时读取 `2*j+0.5` 与 `2*j+2.5`；p=100时为 `0.5*j-0.25` 与 `0.5*j+0.25`，首尾各有一次端点延拓。

K/V 在取样特征上使用原投影，Q来自原时域token；注意力在 F×2 位置上 softmax。频率 bias 输入该配置的真实 `log(f_Hz)`，保持同一坐标网络和初始化；Δt bias保持零。CWT参数组每个配置的参数量目标均为1,046,945；模块组按第2.2节替换对应计算路径。

这个定义固定了物理读取位置与每尺度读取数。P组测量的是**原生时间池化、原生时间卷积及固定位置读取的整体表示效应**。编码器两层3格卷积的局部路径覆盖5帧，池化变化会改变其物理时间跨度；GroupNorm还在全表示上计算统计。因此不能把P组差异单独解释为池化的信息损失。

同样，voices变化会改变尺度卷积的频率跨度，频带裁剪会改变卷积边界与GN支持域；这些均属于固定编码器下的CWT配置效应。若要进一步区分归一化或物理感受野机制，需要另行定义对照。

### 3.3 初始化与等价验收

全部配置沿用L1的 `p1_components.*` 模块隔离种子。先保持原构造流，再执行模块替换；保留的共有可训练参数逐张量一致。CWT参数组仅表示相关buffer、索引及插值权重随配置改变。UniformTFReadout保留原V和零初始化输出投影；位置路径四臂按原waveform encoder子种子构造，再按(s,v)开关配置路径，保持stem/score/project及外部RNG一致，仅(0,0)移除位置网络。所有含条件分支的臂初始条件残差为零；NoTF直接使用原时域token。

B须同时通过：原生L1与适配模型的state_dict及参数遍历顺序对应、AdamW分组一致、零投影恒等、打开输出投影后的完整前向/梯度比较。零输出时波形相等不足以验证CWT路径。CPU合成测试用固定替身主干验证几何和两次优化后的逐张量一致；官方Mamba2在CPU构造核验参数量与初始化。GPU阶段使用官方Mamba2、FP32/BF16及32×4训练形态；三seed打开输出投影的前向/梯度比较容差预设为FP32 `rtol=2e-5, atol=2e-6`，BF16 `rtol=2e-2, atol=2e-3`，逐项保存回执。

## 4. 数据、训练与来源

沿用L1的解析配置和样本身份：train=10,141窗/32人，validation=2,675窗/7人；原subject隔离、数据sample seed、target、Pi、loss、metrics和资格分母保持一致。训练seed与模型初始化seed统一为20260812/20260813/20260814；本轮28配置全部从头训练，同seed使用相同样本顺序合同。历史模型权重和指标仅保留为历史证据，不抵扣本轮84个cell。

训练合同：batch32、累积4、BF16、每epoch80次optimizer update；AdamW及原参数分组，LR 3e-4→3e-5、5% warmup、按6,400 updates规划调度、weight decay=1e-4、clip=1；loss=`L_sync+0.25 L_effort`。最多80epoch，min_epoch=30、patience=15、min_delta=0；按完整validation Local RR严格最小选择checkpoint，并列最早。新的运行器须与L1的loss累积归一化及epoch恢复逻辑一致。

### 4.1 结构来源与同期基准

M4-S1结构和训练合同的只读来源根：

`/mnt/disk_code/marques/resp_reconstruction/runs/m4_linear_fusion_v1/formal_v1_20261019`

| 身份 | SHA-256 |
|---|---|
| session.json | f84949530ceb877a9a1e9e58460dca533f2231b10cccae29ace2e1f76fb0bb7c |
| summary/0b3bdd6476c14144a9e5c6dd86bef39e/receipt.json | 836b554c5796920c838a0e8060797000aad2a641ff3fc8854c733aa647359ac6 |

上述session和summary receipt已在设计阶段只读计算哈希，并与已有结果记录一致，用于结构和训练合同追溯。本轮B按新seed清单在新session下完整训练，所有配对使用本轮B的相同seed结果。

正式准备阶段核验源码快照、模型/初始化、resolved config、loss/optimizer/selector、软件和GPU环境、row身份与cache字节。每个新cell具有独立输出identity；已有20260812/13的历史训练保持冻结，不覆盖或追加。基准适配等价失败时阻断训练，修复后再进入完整矩阵。

### 4.2 CWT 缓存复用

前序 W0/APOR 的20份train/validation CWT缓存是潜在只读来源；先核对实际文件位置、manifest、数据输入/rows、表示定义、依赖版本及字节身份。相同CWT数学不自动保证数据身份相同。通过核验的缓存只读引用，缺失或不匹配的表示在新session内新建。授权范围由执行阶段明确，保留所有历史成功和失败产物。

本次来源预审识别到唯一预处理源码差异：`resp_train/data/research_v2.py` 新增Patch-aligned TF-Mamba专用H缓存分支。旧SHA256为 `37f013106ecd3270a79a87305c05f5105e0d046fc3a941c46f084a777eb024fb`，当前为 `3ff5abab6594f9d3a228b8925b5cb6ccf6193ba91c7d5fd002e68f7cad33ea77`。本轮raw配置的模型名、空tf_cache_path和空tf_representations使该分支不可达，波形切片、target和row身份逻辑保持一致。兼容检查仅接受这对已审阅哈希并核验旧快照字节，记录到各cache引用；其他源码漂移仍显式失败。临时数据的新旧加载器逐样本输入/target/meta等价与错误配置拒绝检查、原缓存引用回归共 **2 passed（58.82秒）**。

训练入口只构建train/validation；research-test须另建固定checkpoint清单及专项执行授权。当前设计不访问测试集波形、cache或checkpoint进行新推理。

## 5. 预设比较与判断

每个非B配置均按同seed与本轮B配对，完整报告27组差异，其中CWT参数组19组、额外模块臂8组。Q组额外报告固定μ的voices变化、固定voices的μ变化，及相对μ13.4/voices12的差中差：

`I(mu,v) = [M(mu,v)-M(13.4,v)] - [M(mu,12)-M(13.4,12)]`。

I按seed计算后汇总，用于描述交互，不将三个seed当作独立受试者。C组按0.8→2→4→8→12→20 Hz报告相邻增量；H/L组报告B−L、B−H及H−L。P组报告P_025−B、P_100−B及P_025−P_100。

模块组另报告：B−A_NoTF（完整条件分支增量）、A_UniformTFReadout−A_NoTF（均匀读取条件增量）、A_NoFrequencyBias−A_UniformTFReadout（内容自适应权重）、B−A_UniformTFReadout（完整学习权重机制）、B−A_NoFrequencyBias（频率bias增量）、B−A_NoPoolPosition（池化位置联合增量），以及H−A_NoTF、C_0p8−A_NoTF。差值统一按“前者−后者”保存，误差负值/PCC正值为改善。全部比较覆盖五项主指标。

位置路径子矩阵记每个同seed指标为M_sv，M_11=B、M_00=NoPoolPosition、M_10=ScorePositionOnly、M_01=ValuePositionOnly。对全部五项主指标及窗口加权/受试者宏平均视图报告：

- 评分路径条件效应：值路径关闭时 `M_10−M_00`，开启时 `M_11−M_01`。
- 值路径条件效应：评分路径关闭时 `M_01−M_00`，开启时 `M_11−M_10`。
- 评分路径平均主效应：`[(M_10−M_00)+(M_11−M_01)]/2`。
- 值路径平均主效应：`[(M_01−M_00)+(M_11−M_10)]/2`。
- 交互效应：`I_pool = M_11−M_10−M_01+M_00`。

以上先按seed计算，再汇总mean/sample SD和方向数；全部保留指标原生单位。交互为零表示该指标上的加性关系；非零表示一条路径的效果随另一条路径开关改变，需结合四臂原值和条件效应解释。三个seed描述训练随机性，不能仅据交互均值的符号宣称统计显著或普遍协同。

五项主指标完整保留：Local RR MAE、Whole RR AE、包络轨迹MAE、全局包络调制误差、lag-aware signed PCC。主视图为窗口加权，逐seed结果、mean/sample SD（ddof=1）、同seed差值及方向数均输出；受试者宏平均、逐人三seed平均差异和Local RR尾部作为诊断，保留资格分母及退化计数。

沿用L1的描述性0.01判据：各指标原生单位的绝对均值差<0.01记为均值无明显差异，否则按方向判断。该判据不代表统计等效，也不替代连续效应量和逐seed一致性。checkpoint selector保持Local RR，其他指标用于解释取舍。

本轮首先交付完整参数敏感性结果；若需推荐配置，按三seed窗口加权validation Local RR均值排序并同时列出另外四项指标及宏平均。多个配置差异小于0.01时保留并列候选和B，不自动把微小排序差转成新默认。模型的最终采用另作决策记录。不得按partial seed、test或个别案例删改矩阵。

CWT相对纯时域的增量由B/H/C_0p8与同期A_NoTF对照回答；固定模型干预和参数敏感性结果用于补充解释，不代替这一重新训练对照。

## 6. 实施阶段与验收交付

| 阶段 | 工作 | 通过条件 |
|---|---|---|
| 设计（已完成） | 分支/worktree、模型身份、矩阵与来源登记 | 28配置/84新cell、核心7臂及位置2×2、三seed与物理坐标定义一致 |
| CPU实现（已通过） | 独立配置/adapter/runner、synthetic定向验证 | B完整等价；八种模块替换、均匀读取的值路径/梯度等价、位置2×2的开关/初始化/活动路径梯度及效应公式、变量F/T、边界插值、频率排序、有限性、矩阵/来源拒绝路径通过 |
| 数据准备与GPU验收 | 核验cache；正式shape、32×4、BF16两次更新 | B等价通过；各臂输出/梯度有限，含条件臂打开输出投影后上游有任务梯度；显存容量可用 |
| train/validation | 固定完整矩阵、每卡独立worker、每seed配对 | 84个新completion；分母/身份一致；history与选点相符 |
| 汇总 | 全部预设比较、受试者诊断、质量与成本曲线 | 可追溯CSV、PNG及中文报告；全部配置覆盖 |
| research-test | validation完成后独立定义 | 先固定完整checkpoint清单、比较和视图，再获准执行 |

资源不足时保留失败回执，在训练前调整统一执行合同。不能单独对高成本臂降低batch、更换精度或减少训练预算后仍声称只改变CWT配置或指定模块。效率记录包含cache空间、单epoch训练/validation耗时、显存峰值和实际训练updates；比较需记录同GPU型号、batch、精度和计时范围。

独立入口为 `scripts/run_m4_s1_cwt_parameters.py`。`m4_s1_cwt_parameters_spec.py` 固定矩阵与配置，`model.py` 实现网格/模块适配，`artifacts.py` 管理session和阶段回执，`data.py` 管理原始dataset与独立cache，`runtime.py` 复用原L1训练引擎及epoch恢复，`summary.py` 完成全矩阵和交互效应统计（上述后缀文件均带 `m4_s1_cwt_parameters_` 前缀，位于scripts目录）。两个定向测试文件分别覆盖模型和运行生命周期。

每个session保存Git/source snapshot、resolved config、数据/sample/training seed、环境、初始化摘要、表示元数据、来源引用、命令、完整history、checkpoint、逐窗口metrics、成功/失败状态与manifest。使用一次session身份和各阶段receipt管理，输出排他创建。案例图第一行为实际预处理后的BCG，固定样本规则，图只保存PNG。

## 7. 操作入口

所有持久输出直接使用项目主目录下的独立session；当前worktree不另建结果副本。`plan`只加载矩阵；`prepare`只冻结源码、实际网格、CPU初始化摘要和84份解析配置，不加载真实波形。默认 `synthetic_only` session只能校准/合成验收，不能升级为正式session。`train_validation` session须显式指定范围；真实数据操作和训练另由命令标志开放。入口不提供test选项。

### CPU定向验证

```bash
cd /mnt/disk_code/marques/resp_reconstruction/.worktrees/m4-s1-cwt-parameters-v1
M4_CWT_PY=/mnt/disk_code/marques/resp_reconstruction/.venv/bin/python
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  "$M4_CWT_PY" -m pytest -q -p no:cacheprovider tests/test_m4_s1_cwt_parameters_model.py tests/test_m4_s1_cwt_parameters_runtime.py
"$M4_CWT_PY" scripts/run_m4_s1_cwt_parameters.py plan --summary
```

定向测试使用合成张量和临时NPZ/CSV；正式Mamba2的CPU检查仅构造模块。实际信号的预处理/cache兼容性、官方CUDA前后向和显存容量由后续阶段验证。

### 已授权执行：准备与缓存

以下命令用于本次数据与合成GPU验收。session名称必须尚未被其他实验占用；源码/配置一经prepare冻结，运行中修改依赖会被拒绝。

```bash
M4_CWT_SESSION=/mnt/disk_code/marques/resp_reconstruction/runs/m4_s1_cwt_parameters_v1/acceptance_v1_20261020
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  "$M4_CWT_PY" scripts/run_m4_s1_cwt_parameters.py prepare --session "$M4_CWT_SESSION" --scope train_validation --confirm-data
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  "$M4_CWT_PY" scripts/run_m4_s1_cwt_parameters.py calibrate --session "$M4_CWT_SESSION"
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  "$M4_CWT_PY" scripts/run_m4_s1_cwt_parameters.py prepare-data --session "$M4_CWT_SESSION" --confirm-data
```

`calibrate`以合成BCG验证20份CWT表示、基准与原变换一致、L/H/低频上限子集一致及时间池化关系。`prepare-data`核验冻结index与train/val row身份、受试者隔离及源NPZ哈希，建立公共validation target和新的数据receipt。非有限样本显式失败。

优先核验已有W0时频实验的20份cache（APOR已复用同一来源），只保存新session下的引用：

```bash
M4_CWT_SOURCE=/mnt/disk_code/marques/resp_reconstruction/runs/home_storage_migration_20261010/preserved_worktrees/worktrees_20261010/cwt-time-frequency-v1/runs/cwt_time_frequency_v1/session_20260930T184031Z_812b424a1c92
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' \
  "$M4_CWT_PY" scripts/run_m4_s1_cwt_parameters.py import-cache --session "$M4_CWT_SESSION" --source-session "$M4_CWT_SOURCE" --all --confirm-data
```

该入口核验旧session、manifest、数据输入/target/rows、库版本、变换源码及所有cache字节。来源不匹配时显式失败，不自动重建。确定需构建某个表示后，可使用 `cache --session "$M4_CWT_SESSION" --representation ID --confirm-data`；已有失败现场时追加 `--retry-failed`，新attempt保留旧现场。全部20份均需新建时使用 `--all`。NoTF直接读取原时域dataset；其余模块臂共享B缓存。

### GPU验收及后续训练入口

本次只执行`smoke`。下方`run`与`summarize`用于后续单独获准的正式训练阶段。

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  "$M4_CWT_PY" scripts/run_m4_s1_cwt_parameters.py smoke --session "$M4_CWT_SESSION" --device cuda:0 --confirm-synthetic
env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  "$M4_CWT_PY" scripts/run_m4_s1_cwt_parameters.py run --session "$M4_CWT_SESSION" --device cuda:0 --confirm-training
"$M4_CWT_PY" scripts/run_m4_s1_cwt_parameters.py status --session "$M4_CWT_SESSION"
"$M4_CWT_PY" scripts/run_m4_s1_cwt_parameters.py summarize --session "$M4_CWT_SESSION"
```

`smoke`默认覆盖28臂，使用正式32×4、BF16、task loss与两次optimizer update；模型先零输出再检查条件上游任务梯度和参数更新，显存reserved硬上限为设备容量95%。CPU替身回执不能用于GPU正式训练。B还执行三seed非零条件输出的原生L1等价核验。`run`进入任一cell前核验同设备/环境的B及该臂成功验收；相同身份的成功验收只读复用。

单cell可指定 `--arm ID --seed 20260814`。多GPU可在不同终端使用相同session、不同 `--device`，分别指定 `--shard-index i --shard-count N`；完整84-cell计划按seed/arm顺序固定分片，不因结果变化。分片与单cell选择互斥；同cell互斥，软件/GPU型号一致性在session内统一登记。

训练恢复在原run命令后加 `--resume`，从最后一个完整epoch恢复模型、optimizer、RNG和选点/早停状态。若训练已到合法终点但导出失败，恢复只重新导出到新attempt；原epochs和失败产物保持原样。数据/校准/GPU验收/汇总的失败重试使用 `--retry-failed`。已完成cell及成功阶段只读校验，不覆盖重跑。

验收产物包括：session/source snapshot/84份resolved config；公共数据receipt和cache记录；每臂工程回执；84个cell的完整epoch、history、最优/最终checkpoint、逐窗口validation指标和completion；全矩阵的窗口加权/宏平均/逐人配对、μ×voices及位置2×2效应、原生指标/分母/tail/选点/参数量/效率CSV、两张PNG及中文报告。任何cell缺失、row/资格漂移或关键值非有限均阻断完整汇总。

## 8. 当前检查记录

2026-10-20：完整定向CPU集合 **80 passed，223.18秒**。检查使用合成张量与临时NPZ/CSV，覆盖：

- 三seed官方Mamba2的CPU构造、全部28臂共有参数/外部RNG核验，20个CWT参数臂均为1,046,945参数；仅评分/仅值位置的参数量与B一致。
- B与原生L1在非零条件输出下FP32/BF16完整前向和梯度逐张量相同；模块/参数顺序、AdamW分组和两次优化后的权重逐张量一致。
- 全部28臂的输出/梯度有限，变量F/T、0.25/0.5/1秒物理坐标及边界、均匀读取的原V/output值路径与梯度、位置2×2公式和活动路径梯度。
- 合成信号上20份实际CWT变换校准，基准、频带子集、时间池化关系一致；真实CRD加载器保持历史seed边界，本轮独立入口支持20260814。
- 沿用正式task loss和4步梯度累积，CPU替身以微批1完成两次optimizer update；验证首步零投影与后续上游梯度、非有限值拒绝。GPU正式微批32的验收待执行。
- 临时数据的train/val隔离、cache与实际输入切片对应、缓存只读引用及来源篡改拒绝、公共validation target一致；索引中的测试集哨兵文件未被访问。
- session/snapshot/config身份检查、失败attempt保留、显式恢复、完整epoch/RNG恢复后与不中断训练权重一致、最早并列选点、合法早停终点。
- 完整84-cell的配对、宏平均、μ×voices与位置2×2效应，以及完整汇总的只读复用；缺失cell、样本漂移及非有限指标拒绝。报告只生成PNG。

`plan --summary`返回28配置、指定三个seed、84个训练cell且历史训练复用为false；10个新增Python文件AST语法检查和 `git diff --check` 通过。官方GPU内核、正式32×4显存及真实数据/cache兼容性尚未验收，本轮未创建正式运行session或启动训练。

前序设计静态检查已覆盖28配置、三seed、核心7臂与位置补充/共享频带映射、84次新训练预算、27组主配对、位置2×2效应系数及文档/YAML一致性。
