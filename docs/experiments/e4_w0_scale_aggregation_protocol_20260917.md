# E4：W0 四区域尺度聚合专项方案

协议 ID：`e4-w0-scale-aggregation-v1-20260917`；日期：2026-09-17。

状态：**用户已授权实施；独立模型、训练/验收/汇总入口和来源锁已准备，synthetic CPU 定向验证 57 项通过。GPU 验收、效率测量和正式三 seed 训练由用户执行，尚未运行。**

## 1. 任务、依据与范围

本方案承接 [E4 交接](e4_scale_aggregation_handoff_20260917.md)。用户本次授权为核对代码与冻结来源、确定一个首轮候选、完成专项方案、控制变量和验收设计。科学问题为：当前尺度平均是否压缩了对多属性呼吸重建有用的尺度分布信息，多个连续频率区域的摘要能否改善重建？本方案在 E4 的立项范围内取代旧计划中依赖“E1 基本不敏感”的启动条件。E1 的冻结证据仍解释为指定尺度重排下的属性相关功能敏感性。

首轮确定一个候选 `w0_mr4_residual`，复用三个冻结 W0 对照，新增三个 seed 的从头训练。实验改变尺度聚合结构及其必要参数，固定其余网络与训练合同。新结构需要新训练；原 W0、E1–E3 结果继续使用各自冻结身份，无须重算、重评或重训。首轮结果属于 research-test-informed development/validation evidence。

2026-09-17 用户在方案交付后明确要求“开始实现吧”，开放本协议的独立实现、相关文档及 synthetic CPU 定向验证。来源核对阶段没有读取真实波形、test 指标或 test cache。当前首轮实际执行范围为 train/validation；GPU、真实数据训练与正式 benchmark 由用户执行。test 的专项阶段规则见第 9 节。

## 2. 已完成的来源核对

机器可读记录：[来源与设计审计](e4_w0_scale_aggregation_source_audit_20260917.json)。本次起点为 `76a0e231bed795baf6b75d644a5d189577307e1d`，工作树原有未跟踪交接文档予以保留。

| Seed | Selected epoch | 冻结 run（仓库相对路径） |
|---|---:|---|
| 20260811 | 13 | `runs/crd_tf_v1/formal/crd_tf102_w/seed_20260811/20260812_210725_400861` |
| 20260812 | 15 | `runs/crd_tf_v1/formal/crd_tf102_w/seed_20260812/20260812_223203_048130` |
| 20260813 | 14 | `runs/crd_tf_v1/formal/crd_tf102_w/seed_20260813/20260812_235740_452006` |

核对结果：

- 22 个来源文件的字节身份已登记，其中 W0 checkpoint/config/run manifest/validation summary 对照 W-v2 candidate lock 经 E2 实现锁核验，validation metrics/summary 对照 E3 来源锁核验；另登记三份来源锁自身身份。实际频率文件、cache manifest 和 train/val row-ID 文件也已核对。
- 三个 checkpoint 在 CPU 读取，epoch 与锁一致、内嵌 config 与 run resolved config 完全一致、模型 state 全有限，并严格加载到当前 W0；完整模型均为 **1,219,850** 参数。
- 三个原 run 均来自干净训练提交 `68b3b85df2f18b3dc8ec59e02ac0780f2be42122`。当前 `model.py` / `tf_v1_model.py` 增加了后续候选和可选深度，但 W0 路径保留六层、97 尺度、同一 FiLM 计算。用该训练提交的两个模型文件与当前模型分别构造，三个 seed 的 **194 个 state tensor 全部逐 tensor 相等**。这验证初始化与 state 结构，不替代后续 GPU forward 验收。
- 原 `training.py`、loss、初始化、基本 block 与 CWT 特征文件仍与训练提交字节一致。训练器的后续 early-stopping 分支在 W0 的 `early_stopping_enabled=false`、默认 `min_delta=0` 下保留严格最小选择；metrics 文件新增的是辅助导出函数，原五主指标计算未改。上述关键文件均与 E2 实现锁一致。
- train/val 大型 W cache 本轮仅继承原锁中的 SHA、shape 和 transform identity，没有重新扫描内容。实现锁准备阶段须重新核验实际文件字节身份；该待办不等同于已完成全 cache 验收。数据索引 SHA 从冻结 manifest 继承，本轮未重做真实数据审计。

真实张量流为 `B×1×97×360 → B×48×97×360 → B×96×97×360 → B×96×360 → B×96×1800 → temporal → active fill → 192-channel gamma/beta`。FiLM 使用 `0.5*tanh`，注入位置与主干保持 W0 定义。

| W0 分支组件 | 参数数 |
|---|---:|
| conv_in：1→48，5×3，无 bias | 720 |
| GroupNorm：8 groups / 48 channels | 96 |
| depthwise：48 channels，3×3，无 bias | 432 |
| conv_out：48→96，1×1，有 bias | 4,704 |
| 上述聚合前小计 | **5,952** |
| 三个 temporal residual blocks | 112,896 |
| active fill：96→65→96 | 12,576 |
| final projection：96→192 | 18,624 |
| 完整 W 条件分支 | **150,048** |

## 3. 唯一首轮候选：四区域均值与残差融合

### 3.1 频率分区

使用冻结 cache 的 `w_frequencies_hz.npy`，文件 SHA-256 为 `15cc722c38e5572b3284c92137cfdeec7cbc4153b4588c2c1e086f26ae3238b3`。实际数组按频率非降序排列，共 97 个槽位，覆盖 `0.03662109375–7.99560546875 Hz`。索引 1、2 的映射频率均为 `0.0396728515625 Hz`；它们来自原有不同尺度槽位，保留两个槽位的原始权重。

选 **K=4**，以实际端点构造等 log-frequency 边界：`e_k = exp(log(f_min) + k/4 * log(f_max/f_min))`。这是固定分辨率的粗区域方案：相比单一摘要保留区域身份，同时把额外投影限制在 36,864 参数。K=4 是首轮工程与科学折中，不声称为最优区域数，也不由 validation/test 效果选择。

| 区域（低→高） | Python 切片 | 槽位数 n_k | 边界区间 Hz | 实际首末中心频率 Hz |
|---|---|---:|---|---|
| R0 | `[0:25]` | 25 | `[0.03662109375, 0.14077039278558345)` | 0.03662109375–0.140380859375 |
| R1 | `[25:49]` | 24 | `[0.14077039278558345, 0.5411171938306031)` | 0.146484375–0.537109375 |
| R2 | `[49:73]` | 24 | `[0.5411171938306031, 2.080038363642994)` | 0.567626953125–2.0721435546875 |
| R3 | `[73:97]` | 24 | `[2.080038363642994, 7.99560546875]` | 2.191162109375–7.99560546875 |

内部边界若恰有中心频率，归入右区；最后端点包含。每槽位恰好属于一区，完整覆盖，重复频率归同区。实际 log-frequency 间距并不严格均匀，等索引与等 log-frequency 在一般情形下不等价；**本冻结数组上**，上述规则恰与 `array_split(arange(97),4)` 的 25/24/24/24 分组一致。实现锁保存完整频率数组、SHA、边界与最终索引，不在运行时用近似十进制边界重新决定分组。这些区域是统计聚合区，不赋予预设生理频带含义。

### 3.2 运算、归一化和初始化

令原 SiLU 后特征为 `X ∈ R^(B×96×97×360)`：

```text
g = X.mean(dim=2)                                  # B×96×360，原 W0 路径
r_k = X[:, :, region_k, :].mean(dim=2)              # 每槽位权重 1/n_k
r = cat([r_0, r_1, r_2, r_3], dim=1)              # B×384×360，先区域再通道
z = g + Conv1d(384, 96, kernel_size=1, bias=False)(r)
z = interpolate(z, size=1800, mode='linear', align_corners=False)
```

投影矩阵 `A∈R^(96×384)` **全零初始化**；没有新增归一化、激活、dropout 或门控。全局均值仍对 97 槽位等权；区域均值各自按 n_k 归一化。数学上 `g=Σ(n_k/97)r_k`，因此融合等效为四块 `W_k=(n_k/97)I+A_k`。系数可正可负，不施加 softmax。实现保留原全局 mean 的计算路径，避免用四个已舍入的区域均值重新合成 g；A=0 时初始聚合数值可严格回到原输出。

pooling 与投影跟随 W0 的 AMP/dtype 语义，不另设全分支 FP32 模式。本次独立公式检查已在 synthetic FP32/BF16 输入上验证零残差精确等值、`B×96×360` 输出和投影全部 36,864 个梯度元素有限且非零；这只验证公式，不是候选模型已实现的声明。

该结构在融合前保留四个区域摘要，通过可学习的跨通道、跨区域线性融合表达位置差异；最终通道仍为 96。区内分布仍被平均，前置卷积也跨区域边界，因此正结果不能解释为完整保留了 97 尺度信息或定位了某一生理频带。单候选对照也不能把额外容量、优化参数化与区域位置利用的贡献完全分离。

## 4. 控制变量和实现约束

1. 固定 CWT 算子、97 尺度和 `[97,360]` cache、96 通道、原卷积/GN、六层主干、FiLM、temporal mixer、refinement/decoder、输入/参考/任务投影及所有公共参数 shape。
2. **active fill 固定为 96→65→96。**先按同 seed 的原构造顺序完成全部 W0 公共模块，再追加 E4 投影；不得再按 150,000 目标回填。新投影若在原 finalize 前插入，会使基础分支从 137,472 增至 174,336 参数，原公式得 hidden=-127 并报错。若为通过检查压缩其他模块，会引入第二个结构变化。
3. 独立 E4 模型/构造入口使用显式 `186,912` 分支、`1,256,714` 全模型合同；旧 TF 的 ±2% 检查继续服务旧候选。实现优先用独立子类/局部工厂扩展 W0，保留公共参数键；不得放宽旧注册表或把新结构伪装成无差异 W0。
4. 新模块构造使用独立 `module_seed(seed, 'e4_mr4_aggregation')`，构造后置零。即使最终零初始化，Conv 构造器仍消耗 RNG，必须隔离。公共 `tf_branch_w` 内的初始化顺序完全保留，比较三个 seed 的公共 state 逐 tensor、CPU RNG 前后状态、相同输入的训练随机流；不能仅以 module 名称有子 seed 推断分支内部一致。
5. 从同 seed 冻结 resolved config 派生；允许差异仅为 E4 协议/arm/aggregation 字段、输出身份和设备/日志显示。明确记录模型科学身份 `w0_mr4_residual` 与原始基座 `crd_tf102_w`。独立构造注入原生 trainer，保持 optimizer、loss、metrics、scheduler 和 selector；若需新增窄工厂扩展点，必须验证旧配置路径不变。
6. 原生 optimizer 按参数属性分组，新投影权重进入 weight decay 组，与原生 LR 相同；不设置新 LR/正则。公共参数的 decay/no-decay 分组须逐名相同。
7. 同 seed 保持训练 shuffle seed、样本顺序、physical batch、末 batch、CPU/CUDA RNG 设置和 branch checkpoint chunk=8。数学等值不意味着后续训练轨迹相同；新增聚合和梯度裁剪将自然影响优化轨迹。

| 合同 | 冻结值 |
|---|---|
| 输入任务 | 180 s，100 Hz，18000 点；原 BCG 输入、THO 参考与 Pi |
| Train / validation | 10141 / 2675 窗口，32 / 7 个 samp_id；原 admission、row identity、subject/session 隔离 |
| 训练 seed | 20260811、20260812、20260813；初始化 seed 与训练 seed 相同 |
| Sample seed | train=20260610，val=20260611；完整 split |
| 预算 | 80 epochs × 80 updates = 6400 updates/seed；early stopping=false，resume=false |
| Batch | physical/effective=128，accumulation=1，drop_last=false；79×128+29=10141 |
| Optimizer | 原生 AdamW，betas=(0.9,0.999)，eps=1e-8，weight decay=1e-4 |
| LR | 3e-4→3e-5，step-exact warm-up cosine，warm-up fraction=0.05，clip=1.0 |
| 精度 | BF16 AMP；原生 loss FP32 路径 |
| 目标 | `L_sync + 0.25 L_effort` |
| Selector | 每 epoch 完整 validation Local RR 严格最小；平局取最早 epoch |

初始公共参数与 W0 同 seed 相等指从头初始化，不迁移已训练 W0 checkpoint。

## 5. 容量与效率口径

| 项目 | W0 | E4 | 增量 |
|---|---:|---:|---:|
| 聚合器 trainable 参数 | 0 | 36,864 | `K*C*C=4*96*96` |
| 完整 W 分支参数 | 150,048 | 186,912 | +24.5681% |
| 全模型参数 | 1,219,850 | 1,256,714 | +3.0220% |
| 聚合投影 MACs / 180 s 窗口 | 0 | 13,271,040 | `360*384*96` |

MAC 计一次乘加为一次；乘和加各算一个 FLOP 时，该投影为 26,542,080 FLOPs。另增加区域求和约 `96*360*(97−4)=3,214,080` 次加法、138,240 次均值缩放，以及 34,560 次残差加法，另列而不混入 MAC。concat、内存传输、归一化、SiLU、interpolate 和 Mamba scan 不由这个投影数字代表。实现验收报告分支/全模型解析或 profiler 覆盖范围及未支持算子，不能把新增投影 MAC 称为全模型 FLOPs。

区域 concat 单次物化为每样本 138,240 元素，BF16 270 KiB、FP32 540 KiB；实际 autograd、区域张量与 concat 可能同时存活。新增 FP32 参数/梯度/Adam 两个 moments 的简单合计为 0.5625 MiB，不包括转换副本、allocator、workspace。二维激活和旧公共模块保持同规格；因此预计增加投影计算和额外摘要内存，但本轮没有实测延迟或峰值显存，不承诺增量百分比。

用户执行 GPU 效率测量时：同硬件、依赖、设备、dtype、相同 synthetic 输入，串行测 W0/E4；在独立进程分别 warm-up 5 次、计时 20 次、CUDA synchronize 和重置峰值统计，保存每次时长、median/IQR、allocated/reserved 峰值及设备总显存。测 batch-1 eval forward、batch-128 training update 两种场景，后者包含原生 loss/backward/clip/AdamW，保留 chunk=8；CWT cache 读取和在线变换不计入模型内耗时。两臂采用交替顺序的三组测量，用于描述工程波动。这是待实现的有界用户测量步骤。

首轮仅三次新训练，未包含容量匹配重训练臂。若正结果需要进一步归因于尺度区域结构，另立三 seed 容量对照及其明确预算。结论只能称为这个聚合方案相对 W0 的整体收益/代价。

## 6. 矩阵与统计设计

| Arm | Seeds | 训练 | 最终 validation |
|---|---|---|---|
| W0_FULL | 三个固定 seed | 复用冻结 run | 复用原 selected checkpoint 对应 CSV/summary |
| w0_mr4_residual | 同三个 seed | 从头 3×80 epochs，共 19,200 updates | 每个新 selected checkpoint 完整 2675 窗口 |

五主指标完整并列报告：`whole_rr_abs_error_bpm`、`local_rr_mae_bpm`、`envelope_trajectory_mae`、`global_envelope_modulation_error`、`lag_aware_signed_pcc`。指标内与窗口间聚合沿用冻结定义：local 先 sample 内平均，再 target-eligible sample 直接平均；两项包络覆盖全部 admitted sample，prediction 失败按原定义计入。Local RR 仅承担 selector，不作为 E4 唯一科学终点。

对四个 error，逐 seed 保存原始差 `E4−W0` 和相对差 `100*(E4−W0)/W0`；PCC 保存原值并用 `W0−E4` 表示恶化方向。正 delta 统一表示 E4 较差。保留三 seed 原值、各 arm mean/sample SD（ddof=1）、配对 delta 的 mean/sample SD 和改善/相等/恶化方向数；误差另报三 seed 均值之比，与“逐 seed 百分比的均值”分栏。基线 error 若为零，相对差明确 NA，保留原始差且不加 epsilon。

不设预先的效果合格门槛或五项加权总分；依据完整效应量、三个 seed 的方向、跨属性取舍、容量和效率判断。工程验收通过不等于科学假设成立。方向混合时如实报告，零/负结果仅约束本候选和预算，不能证明原尺度平均最优。正结果也不自动替换论文冻结主模型。

三 seed SD 描述训练随机性；窗口重叠且同受试者相关，2675 个窗口不是独立重复实验。本轮不以窗口级显著性检验或仅三个 seed 的 p-value 作确认性结论。训练预算不随部分 seed 的效果改变。补充受试者统计或频带消融须另行定义，避免结果后寻找有利视图。

计划汇总文件为 `seed_metrics.csv`（2×3 行）、`paired_seed_delta.csv`（3×5 行）、`three_seed_comparison.csv`（5 行）、`parameter_compute_report.json`、`source_receipt.json` 和完成 manifest。各表记录来源 SHA、arm、seed、split、selected epoch 和各指标分母。

## 7. 工程验收与失败处理

### 7.1 Codex 可执行的 synthetic CPU 定向检查

实现阶段覆盖以下性质，使用 disposable fixture，不接真实数据：

1. 冻结频率和四区 membership 全覆盖、无重叠、边界归属、重复槽位保留；每区 1/n_k、concat 顺序、shape/dtype 契约。构造区域常值/脉冲，验证正确区域的投影响应，避免只测输出尺寸。
2. 三 seed 的全部公共 state、fill=65、六层主干、参数数、optimizer 分组、RNG 隔离与 W0 对照；A=0 时先检查聚合和 temporal 输入严格等值，再检查 branch 输出。最终 FiLM 零初始化会遮蔽错误，不能只比较初始 waveform。
3. 聚合直接梯度测试须验证四块权重均得到非零有限梯度、输入梯度有限。整体模型的 final projection 初始为零，第一次任务 backward 可使上游 A 梯度为零；用受控的多步 synthetic 更新检查通路打开后的 A 梯度与参数变化，不能据第一步上游零梯度误判未参与 forward，也不能永久豁免梯度验收。
4. A 非零时的 branch chunked/unchunked 输出与梯度等价、state_dict round-trip、错误 shape、NaN/Inf 和未知科学配置显式失败；候选加载必须严格匹配 arm，不能用 `strict=False` 接受 W0 checkpoint。
5. 短程原生 trainer fixture（例如 2 epochs，tiny synthetic 模型/数据）核验末 batch、update/LR、最早严格最小 selector、best/final 与 history 对应、三 seed 完整性、丢 row/重复 row/错误分母/混锁/失败回执/重复运行拒绝。fixture 不承担真实 Mamba/GPU 正确性证明。

### 7.2 用户执行的 GPU 验收

- 真实候选与原生 W0 从新初始化开始，三 seed batch-1 BF16 对照初始聚合/branch/waveform；关闭 dropout 或同步重放 RNG，比对时避免随机数差异。FP32 聚合要求严格等值；同一 dtype 的零残差路径亦要求等值，完整 GPU 算子误差容限在实现测试中锁定并保存实际最大误差。
- 每 seed 至少 3 次原生任务 loss/backward/AdamW synthetic update，检查 input/target/prediction、loss、梯度、参数及 optimizer state 全有限；通路打开后 A 的四区域梯度均非零且实际更新，原有 active 模块保持参与。3 次仍未打开则验收失败并诊断，不能仅按 loss 有限放行。
- 单独 batch-128 acceptance 使用动态合成波形与随机 W、原生 chunk=8，至少 3 updates；显存不能 OOM，记录峰值。保持原工程上限 peak reserved/device total ≤80%。若失败，记录现场；不自动减 batch、改精度或缩窄 fill。需要改变训练 substrate 时修订方案和 W0 可比性。
- 验收回执绑定科学配置、实现锁、干净 commit、依赖版本、GPU 和 CUDA 环境。训练只接受同实现身份的成功验收回执。

### 7.3 正式完成与失败验收

每 seed 应保存 resolved config、命令、代码/来源/实现锁、数据索引及有序 rows/hash、初始化和训练 seed、依赖环境、访问记录、optimizer 分组、80 行 history、best/final checkpoint、完整逐窗口 metrics 和 summary。验证 history epochs=1…80、更新数=80…6400、LR 轨迹符合合同、best 为最早严格最小 Local RR、final epoch=80/update=6400，checkpoint 内配置、epoch、history 和 optimizer step 彼此一致。

按 validation row identity 核验 2675 窗口、7 samp_id、无缺失/重复、原生 target eligibility 与分母一致，五主指标按原定义有限；记录 prediction degeneration。出现 degeneration 不过滤样本或挑其他 checkpoint：保留失败值和诊断，标记质量验收失败，该科学结果仍需披露，不能因效果不好自动重训。

每个 attempt 先排他建目录并写 `lifecycle_started.json`，成功后生成 `manifest.json` / `freeze_receipt.json`；工程失败写 `lifecycle_failed.json`，保留全部现场。当前 seed 因工程失败停止时，不删除其他已完成 seed；同实现/合同下排除执行故障后以新 attempt 从头执行失败 seed。若涉及代码/科学配置修订，升级实现 identity 并说明旧 seed 可比性，禁止混锁汇总。完成 seed 禁止同 identity 重跑。三 seed 全部完成后一次性汇总；不依据部分结果改候选、阈值或 selector。

## 8. 独立身份及后续实现顺序

| 对象 | 实现路径/身份 |
|---|---|
| Arm | `w0_mr4_residual` |
| 独立模型 | `resp_train/paper_evidence/e4_scale_aggregation_model.py` |
| 实验控制器 | `resp_train/paper_evidence/e4_scale_aggregation.py` |
| GPU 工程验收与测量 | `resp_train/paper_evidence/e4_scale_aggregation_engineering.py` |
| CLI | `scripts/run_e4_w0_scale_aggregation.py` |
| 定向测试 | `tests/test_e4_scale_aggregation.py` |
| 实现锁 | `docs/experiments/e4_w0_scale_aggregation_implementation_lock_20260917.json` |
| 输出根 | `runs/e4_w0_scale_aggregation_v1/` |

以上代码路径和 CLI 已实现。实现锁包括本协议与源码 SHA、三个完整 W0 config、三个 E4 resolved templates、全部来源文件 SHA、完整 frequency/membership 及明确参数合同；CLI 与本节命令一同绑定源码/协议身份。来源审计 JSON 保留第一阶段检查事实，运行使用独立 implementation lock。

实施顺序为：独立模型及科学配置检查 → 定向 CPU 验证 → 核验 train/val cache 字节身份并准备实现锁 → 提交本次所需文件 → 用户 GPU 验收/效率测量 → 用户三 seed 正式训练 → 完整 validation 汇总和结果记录。执行命令如下；已生成的实现锁直接复用，`prepare-lock` 拒绝覆盖：

```bash
# CPU synthetic 定向验证与一次性锁准备。
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python -m pytest \
  tests/test_e4_scale_aggregation.py tests/test_e2_effort_ablation.py -q
./.venv/bin/python scripts/run_e4_w0_scale_aggregation.py prepare-lock

# 用户：synthetic GPU 多步检查、batch-128 acceptance 与匹配效率测量。
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python \
  scripts/run_e4_w0_scale_aggregation.py gpu-acceptance --device cuda:0
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python \
  scripts/run_e4_w0_scale_aggregation.py benchmark --device cuda:0

# 用户：取得成功验收回执后，固定三 seed 顺序。
E4_GPU_RECEIPT='/实际完成的E4验收attempt目录'
for E4_SEED in 20260811 20260812 20260813; do
  env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python \
    scripts/run_e4_w0_scale_aggregation.py formal --device cuda:0 \
    --seed "$E4_SEED" --gpu-receipt "$E4_GPU_RECEIPT" || break
done

# 三个完成 attempt 齐全后，核验完整矩阵并配对汇总。
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python \
  scripts/run_e4_w0_scale_aggregation.py summarize --runs \
  '/seed_20260811的完成attempt目录' \
  '/seed_20260812的完成attempt目录' \
  '/seed_20260813的完成attempt目录'
```

每阶段输出位于 `<输出根>/<phase>/[seed_<seed>/]<phase>_<lock前缀>_<UTC>_<UUID>/`，时间戳不是唯一 identity。汇总命令须显式接受三个完成 attempt，验证其来源和同实现锁。运行期缓存/波形访问由 scoped access receipt 登记，先登记再读取。

## 9. Test 决定与证据边界

首轮专项执行定义为 train/validation。当前代码/来源审计无需读取 test 结果来决定 K、分区或初始化。本次没有授权代跑 test，也未建立 E4 test 执行附件。

三 seed train/validation 完成并锁定后，如果研究问题需要跨 split 检查，由 E4 独立附件 `e4_w0_scale_aggregation_test_protocol_<date>.md` 明确三枚已选 checkpoint、W0 对照 CSV 身份、2310 窗口/8 samp_id、sample seed=20260612、cache/row identity、五主指标及原生诊断、不可覆盖输出和当次执行授权。全部候选 seed 使用已锁的 validation-selected checkpoint；不因 test 重选、调结构或只评价有利 seed。既有 W0 test CSV 可作为匹配附件的冻结对照来源。

复用该 test 属于 research-test，不能称为首次、无偏外部确认。需要更强泛化结论时另行锁定未参与开发的数据。是否进入此阶段按新增科学问题决定，不套用旧实验的 test 禁止或自动开放规则。

## 10. 本阶段交付与验收状态

- 已完成：代码流和参数核对、三个冻结 run 的来源/CPU checkpoint 检查、历史与当前 W0 三 seed 初始化一致性、实际频率四区划分、synthetic 聚合公式 FP32/BF16 等值和梯度验证、本专项方案及来源审计记录。
- 实现阶段已交付独立模型、控制器、GPU 工程验收/独立进程 benchmark、CLI 和集成 CPU 定向测试。独立 implementation lock 保存重新核验的 train/val W cache 字节身份及运行合同。GPU 多步与 batch-128 验收、效率实测、三 seed 正式训练/validation 和最终解释待用户执行阶段。
- 本轮不改变数据划分、loss、指标、selector 或论文冻结结果。经用户明确授权，论文三份实验计划/内部事实索引已同步为“E4 已重新立项、方案待审查”，保留既有其他编辑。原字节备份、SHA 清单、diff 与安装回执位于 `/tmp/e4_paper_sync_20260917/`。

## 11. 实现约束的落实

- 公共代码唯一扩展为 `CRDExperiment._build_model()`，默认仍调用原 `build_crd_model(self.cfg)`；E4 子类覆盖该方法。原训练更新循环、loss、metrics 和 selector 均复用。普通 CRD 配置校验器未放宽，E4 配置由独立入口逐字段核验。
- 历史 E1–E3 源码/协议/锁和所有产物保持原字节。公共训练器的新扩展改变了该文件 SHA，历史完整代码锁仍绑定原提交，不能通过重写旧锁接受新工作树；历史复现使用原提交。E4 将本次公共扩展纳入新的实现锁，E2 synthetic 回归覆盖默认构造路径。
- GPU 初始整模型比较使用 `rtol=1e-5, atol=1e-6` 并保存最大绝对误差；聚合输出和 temporal 输入要求逐 tensor 精确相等。batch-1 验收覆盖三个 seed，batch-128 acceptance 使用固定 seed 20260811，每项均执行 3 个原生 update。正式训练同时核对验收 commit、关键包版本、GPU/CUDA 和精度相关设置。
- 独立 benchmark 共 3 组×2 场景×2 arms，每项在新进程 warm-up 5 次、测量 20 次，保留每次时间与进程日志；场景为 batch-1 eval、batch-128 native training update。全模型 FLOPs 暂不报告缺少完整算子覆盖的数值；报告新增投影和完整 W 分支卷积 MAC（W0 458,438,400，E4 471,709,440），明确排除归一化、激活、插值、Mamba/FFT 等算子。
- 相同 seed/phase/实现锁使用进程文件锁互斥。正式完成核对每个 optimizer 参数的 moment shape 与 step，严格加载完整候选 checkpoint。prediction degeneration 保留在逐窗口结果和汇总中，`quality_acceptance_passed=false` 作为质量失败标志；完成身份仍拒绝按效果重跑。

2026-09-17 最终 synthetic CPU 定向验证：`tests/test_e4_scale_aggregation.py` 与 `tests/test_e2_effort_ablation.py` 合计 **57 passed，28.14 s**。覆盖三个真实 W0/E4 模型的 CPU 初始化、FP32/BF16 聚合、非零投影的分块前向/梯度、零 FiLM 多步梯度打开、独立配置/锁漂移、旧训练器默认构造路径，以及 tiny synthetic fixture 的三 seed 原生训练/汇总生命周期。该 fixture 使用 2 epochs、3 个 train sample（batch=2，保留尾 batch）和 2 个 validation sample，不读取真实波形；正式入口仍固定 80 epochs、128 batch 和完整 split。另已检查 CLI 帮助、Python 语法和工作树补丁格式。
