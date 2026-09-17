# E4 v2：四种尺度聚合的配对比较

协议 ID：`e4-scale-aggregation-v2-20260917`；日期：2026-09-17。

状态：**独立模型、train/validation 控制器、GPU 工程入口及完整 test 入口已实现，100 项 synthetic CPU 定向验证通过。训练身份由独立实现锁登记。GPU 验收、效率测量、12 次正式训练和真实 test 评价由用户执行，尚未运行。**

## 1. 科学问题与适用范围

本协议比较全局静态尺度权重、内容自适应尺度注意力、显式频率条件注意力及逐通道静态区域权重，考察 W0 的等权尺度平均在多属性呼吸重建中的收益与代价。冻结四个候选、三个训练 seed 和全部控制变量；所有候选均与同 seed 的冻结 W0 配对。新聚合器需要从头训练，共新增 12 次训练。原 W0 与既有 E4 v1 证据保留各自身份，新结果单独归属于本协议。

本轮授权覆盖独立实现、专项协议与锁、synthetic CPU 定向验证，以及供用户执行的 train/validation 和独立 test 入口。test 实现预先准备；实际 checkpoint allowlist 只能在 12 次训练全部完成、完整 validation 矩阵汇总冻结后生成。四个候选的三个 validation-selected checkpoint 全部进入 test，共 12 次评价。开发期间不根据 partial seed 或中间 test 表现修改矩阵、训练预算、指标或 selector。

本实验在固定 train/validation/test 划分上进行同 seed 配对比较。三 seed 反映训练随机性，重叠窗口不作为独立重复实验。结果按五项属性、训练波动及效率代价共同解释。

## 2. W0、数据与来源身份

W0 来源沿用 [E4 v1 主协议](e4_w0_scale_aggregation_protocol_20260917.md)、其 [实现锁](e4_w0_scale_aggregation_implementation_lock_20260917.json)及 [test 附件](e4_w0_scale_aggregation_test_protocol_20260917.md)中锁定的原始来源。新实现锁记录依赖锁本身及实际来源文件的 SHA-256；历史协议、代码、锁和实验产物保持各自冻结身份。

| Seed | W0 selected epoch | 冻结 W0 run |
|---|---:|---|
| 20260811 | 13 | `runs/crd_tf_v1/formal/crd_tf102_w/seed_20260811/20260812_210725_400861` |
| 20260812 | 15 | `runs/crd_tf_v1/formal/crd_tf102_w/seed_20260812/20260812_223203_048130` |
| 20260813 | 14 | `runs/crd_tf_v1/formal/crd_tf102_w/seed_20260813/20260812_235740_452006` |

数据来自 `research_v2` 数据集，根目录为 `/mnt/disk_code/marques/resp_prepare/dataset/20260620_research_v2_resp_reconstruction_stage2_1_segrobustz_bcgstagee_log1psoftz_robustconf`，索引为 `training/dataset_index.csv`。输入使用原 BCG `bcg_rawish_segment_soft_z_key`，目标使用原 THO `target_waveform_segment_soft_z_key`；180 s、100 Hz、18000 点，admission、Pi、target eligibility、subject/session 隔离和 row identity 沿用冻结合同。

| Split | 完整窗口数 | samp_id 数 | Sample seed |
|---|---:|---:|---:|
| train | 10141 | 32 | 20260610 |
| val | 2675 | 7 | 20260611 |
| test | 2310 | 8 | 20260612 |

train/val W cache 来源为 `runs/crd_tf_v1/cache/bd6cea7348f6b51ed768b89cf9b3425530b6358a82ba78277844517a1c27fea0/`，W shape 固定为 `[97,360]`。来源锁保存数据索引、cache manifest、数组及有序 row identity。准备 test 锁仅从冻结 metadata 登记身份，真实 test 数组和波形在用户执行评价时才受控读取。

test cache 为 `runs/crd_tf_v1/research_test_cache/40a24df424b2ff9182cfcc6f5b7c12d287578b0df1ed1e25b56b7af1c7f73839/`；manifest SHA-256 为 `5d43ecf34596d5a6dd7cbaba75d91f9b7cbbb00214ae7594a4755e2afe510745`，有序 row SHA-256 为 `184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8`。W0 test 直接复用三个原 run 的 `research_test_metrics.csv`、`research_test_metrics_summary.csv` 与评价 manifest，来源闭合到 `runs/crd_tf_v1/research_test_summary/access_audit.csv`，该文件 SHA-256 为 `eb75cce11e1827b983f6540719e772c1e0288e8ffda06b7f3ae917bc2920d377`。复用来源必须与同 seed W0 checkpoint/selected epoch 相符。

## 3. 冻结频率网格与四区域

频率文件 `w_frequencies_hz.npy` 的 SHA-256 固定为 `15cc722c38e5572b3284c92137cfdeec7cbc4153b4588c2c1e086f26ae3238b3`。97 个尺度槽位覆盖 `0.03662109375–7.99560546875 Hz`，按实际频率非降序排列；索引 1、2 的重复频率 `0.0396728515625 Hz` 保留为两个原始尺度槽位。

四区域沿用冻结 E4 v1 的等 log-frequency 边界和 membership：

| 区域 | Python 切片 | 槽位数 | 边界 Hz |
|---|---|---:|---|
| R0 | `[0:25]` | 25 | `[0.03662109375, 0.14077039278558345)` |
| R1 | `[25:49]` | 24 | `[0.14077039278558345, 0.5411171938306031)` |
| R2 | `[49:73]` | 24 | `[0.5411171938306031, 2.080038363642994)` |
| R3 | `[73:97]` | 24 | `[2.080038363642994, 7.99560546875]` |

实现锁保存完整频率数组、边界和 membership，运行时以已锁索引聚合。每个槽位恰属于一区，内部边界归右区，末端点包含。这些区域是统计聚合区，不预设生理频带含义。

频率注意力使用固定坐标 `q_s = 2*(log(f_s)-log(f_min))/(log(f_max)-log(f_min))-1`，由真实频率而非尺度序号生成；其端点为 −1 和 1，重复频率具有相同坐标。坐标为固定 buffer，不计为可学习参数。

## 4. 四个聚合候选

聚合输入为原 SiLU 后的 `X ∈ R^(B×96×97×360)`，原 W0 聚合为 `g=X.mean(dim=2)`。四种候选均保留这条全局均值计算路径，并增加相对初始权重的加权修正，输出仍为 `B×96×360`，随后使用原插值、temporal blocks、active fill 和 FiLM。

| Arm ID | 权重依赖 | 新增可学习参数 |
|---|---|---:|
| `static_scale` | 97 个静态尺度 logits，全部样本、时间、通道共用 | 97 |
| `scale_attention` | 每个样本/尺度/时间的全部 96 通道内容；输出权重在通道间共享 | 784 |
| `frequency_attention` | 同一内容特征及固定真实频率坐标；输出权重在通道间共享 | 792 |
| `channel_region` | 每个通道独立的 4 个静态区域 logits，样本与时间共用 | 384 |

### 4.1 静态尺度与两种注意力

`static_scale` 使用 `a_s`，全零初始化。`scale_attention` 的共享评分器为：

```text
h[b,:,s,t] = SiLU(W_content @ X[b,:,s,t] + b_content)
l[b,s,t] = W_score @ h[b,:,s,t]
```

其中 `W_content` 为 `8×96`，`b_content` 为 8，`W_score` 为 `1×8`，末层没有 bias；参数数为 `768+8+8=784`。`frequency_attention` 在相同 SiLU 前增加 `v_frequency*q_s`，其中 `v_frequency` 为 8 个可学习系数，合计 792 参数。频率项进入非线性前，才能与内容形成非加性评分；固定标量坐标本身不额外学习。

两种注意力的公共内容层使用 `cpu_module_seed(seed, 'e4_v2_attention_content')` 和原生 Conv2d 默认初始化，逐 tensor 相等；末层 `W_score` 全零初始化。频率候选的额外 8 个系数使用独立 `cpu_module_seed(seed, 'e4_v2_frequency_coefficient')`，服从均值 0、标准差 `96**(-0.5)` 的正态初始化。该上下文以 `torch.default_generator.manual_seed(named_subseed)` 仅设置 CPU generator 并恢复其状态，不触碰 CUDA generator；新增 replacement branch 同样使用 CPU 专用上下文和原 `tf_branch_w` 子 seed，CPU 初始 state 保持冻结定义。原 W0 父构造继续使用冻结 `module_seed`，新模块在父构造完成后添加；新增构造不得增加 CUDA 播种调用或改变训练 CUDA 随机流。静态尺度及区域 logits 全零。

对前三个候选，沿 97 尺度归一化。以 FP32 计算 `mass_s=exp(l_s-max(l))`、`sum_mass=Σ_s mass_s`，使用

```text
delta_s = (97*mass_s - sum_mass) / (97*sum_mass)
correction[b,c,t] = Σ_s delta[b,s,t] * X[b,c,s,t]
z = g + correction.to(g.dtype)
```

静态尺度的 logits/权重按对应维度广播。该式数学上等价于 `softmax(l)-1/97`，零评分时分子数值精确为零；避免先算近似的 `1/97` 后相减造成初始化残差。FP32 稳定指数、归一化及加权修正后，仅修正结果转回 `g.dtype`。初始化聚合和 temporal 输入必须与原 W0 严格相等。

### 4.2 逐通道区域权重

区域摘要 `r[b,c,k,t]` 是区域内的算术均值；logits `a[c,k]` 形状为 `96×4`。每区使用槽位数 prior `n=(25,24,24,24)`：

```text
mass[c,k] = n[k] * exp(a[c,k] - max_k(a[c,k]))
sum_mass[c] = Σ_k mass[c,k]
delta[c,k] = (97*mass[c,k] - n[k]*sum_mass[c]) / (97*sum_mass[c])
correction[b,c,t] = Σ_k delta[c,k] * r[b,c,k,t]
z = g + correction.to(g.dtype)
```

该质量归一化等价于 `softmax(a+log(n))`，初始区域质量为 `n/97`，保证原始 97 槽位等权，而不是四区域等权。FP32 计算保证零 logits 时修正分子精确为零。每个通道独立学习区域权重；区域内仍等权平均，区域间不增加跨通道线性混合。

### 4.3 容量与可解释边界

| Arm | 聚合器参数 | W 分支参数 | 全模型参数 |
|---|---:|---:|---:|
| W0 | 0 | 150048 | 1219850 |
| `static_scale` | 97 | 150145 | 1219947 |
| `scale_attention` | 784 | 150832 | 1220634 |
| `frequency_attention` | 792 | 150840 | 1220642 |
| `channel_region` | 384 | 150432 | 1220234 |

active fill 固定为 `96→65→96`。新增参数不触发原 150000 参数目标的重新回填，也不以压缩公共模块匹配容量。四候选仍存在参数量、运算量及优化参数化差异；注意力权重和区域权重不能直接视为因果归因或生理频带定位。各候选的收益只能先解释为所定义聚合方案的整体效果。

## 5. 固定训练合同与矩阵

公共 W0 模块保持原初始化顺序和相同 seed 的全部 state：原聚合前卷积/GN/SiLU、六层主干、96 通道、FiLM `0.5*tanh`、temporal mixer、refinement/decoder、输入/参考/任务投影均固定。所有候选从头初始化，不迁移已训练 W0 参数。独立模型工厂复用原生 trainer、loss、metrics 和 checkpoint selector。

| 合同 | 冻结值 |
|---|---|
| 训练及初始化 seeds | 20260811、20260812、20260813 |
| 每次训练 | 80 epochs × 80 updates = 6400 updates；early stopping=false，resume=false |
| Physical/effective batch | 128；accumulation=1，drop_last=false；每 epoch 为 79×128+29 |
| Optimizer | 原生 AdamW；betas=(0.9,0.999)，eps=1e-8，weight decay=1e-4 |
| LR | 3e-4→3e-5；step-exact warm-up cosine；warm-up fraction=0.05；clip=1.0 |
| 精度及 chunk | BF16 AMP、原生 loss FP32 路径；branch checkpoint chunk=8 |
| 完整目标 | `L_sync + 0.25 L_effort`，各项原始内部计算和系数不变 |
| Checkpoint selector | 每 epoch 完整 validation Local RR 严格最小；平局取最早 epoch |

新参数按照原生 optimizer 的参数属性规则进入 decay/no-decay 分组，使用相同 LR 和正则合同；公共参数分组逐名匹配。shuffle、sample seed、末 batch、训练随机流与原 W0 合同一致。初始聚合等值不意味着训练轨迹相同，新增参数梯度及全模型梯度裁剪会影响后续优化。

正式矩阵为四 arm × 三 seed：新增 **12×6400=76800 updates**。W0 的三次训练及 selected validation CSV/summary 直接复用。每个新 selected checkpoint 完整评价 2675 个 validation 窗口。全部 12 次正式训练完成前不发布可用于删减矩阵的候选筛选结论。

## 6. 工程验收与效率测量

### 6.1 Synthetic CPU 定向验证

Codex 可执行数分钟内的静态/语法检查和 disposable synthetic CPU 测试，覆盖：

1. 冻结频率/区域 membership、重复频率、真实频率归一化；常值、脉冲和可解析权重输入下的四种聚合行为，以及 shape/dtype 合同。
2. 三 seed 公共 W0 state、fill=65、六层主干、参数数及 optimizer 分组；两种注意力公共内容初始化一致；新增构造 RNG 隔离，并通过 mock 核对 CUDA 播种调用序列与 W0 完全一致。
3. FP32/BF16 的零评分精确修正零、聚合与 temporal 输入逐 tensor 等值；非零评分符合显式权重公式。初始 FiLM 的零投影会掩盖上游问题，因此初始 waveform 相等不能替代聚合级验收。
4. 直接聚合梯度与受控多步任务更新，检查新参数通路在零末层/零 FiLM 打开后得到有限非零梯度并实际变化；chunked/unchunked 前向和梯度、严格 state round-trip、错误 arm/config/state 拒绝。
5. tiny synthetic trainer 的尾 batch、update/LR、最早严格最小 selector、history/best/final 一致；完整 12-cell validation/test 矩阵及 W0 配对统计；缺失/重复 cell、混锁、漂移、非有限、错误分母和 identity 显式失败。
6. 独立 test 准备只依赖 metadata、allowlist 固定为全四 arm×三 seed、target identity、受试者隔离、退化质量标志、互斥及不可覆盖生命周期。

### 6.2 用户执行的 GPU 验收

每个候选的三个 seed 均做 batch-1 BF16 初始 W0 对照和 **5 次**原生 loss/backward/AdamW synthetic updates。初始聚合及 temporal 输入要求精确相等，整模型使用锁定容差 `rtol=1e-5, atol=1e-6` 并保存实际最大误差；同步重放 RNG 或关闭 dropout 以消除比较中的随机流差异。

每个候选另以 seed 20260811 执行 batch-128、chunk=8 的 **5 次**原生 updates，记录峰值显存，要求无 OOM 且 peak reserved/device total ≤80%。input、target、prediction、loss、梯度、参数及 optimizer state 全有限，通路打开后的新增参数组和原 active 模块参与更新。注意力 score 与 FiLM 投影均有零初始化，验收须覆盖打开后的内容层，以及频率候选的额外频率系数，不能只检查末层 score。若 5 次内通路仍未打开，验收失败并诊断。不得自动减 batch、改精度或改变 fill 后沿用原验收身份。

验收回执绑定四候选合同、实现锁、干净代码 commit、依赖、GPU/CUDA 和精度设置。正式训练仅接受与实现身份和运行环境匹配的成功回执。GPU 验收使用合成输入，但仍由用户执行。

### 6.3 Benchmark

固定 W0 加四候选共 **5 arms × 2 场景 × 3 组**，共 30 个独立测量进程。每进程 warm-up 5 次、计时 20 次，CUDA synchronize，重置峰值统计；五 arm 顺序按预先固定的轮换/反向顺序变化，保存实际顺序。场景为 batch-1 eval forward 和 batch-128 native training update，后者包含原生 loss/backward/clip/AdamW，保留 chunk=8。

同硬件、依赖、dtype 和合成输入，记录逐次时间、median/IQR、峰值 allocated/reserved、设备总显存及进程日志。CWT/cache 读取不计入模型内耗时。参数数为严格计数；MAC/FLOPs 报告必须列明算子覆盖，不能用局部评分器计算代替全模型 FLOPs。工程合格与科学效果分开判断。

## 7. 完整指标与配对统计

五主指标全部报告：`whole_rr_abs_error_bpm`、`local_rr_mae_bpm`、`envelope_trajectory_mae`、`global_envelope_modulation_error`、`lag_aware_signed_pcc`。调用冻结原生指标定义，保留 target eligibility、Local RR eligible windows、envelope/IBI 等诊断字段；本协议不追加新的主指标或五项加权总分。Local RR 只承担预先固定的 checkpoint selector。

Local 指标先在单窗口内按原生资格计算，再在 target-eligible sample 上直接平均；两项包络按原定义覆盖全部 admitted sample。这里的 sample 为数据窗口，`samp_id` 另表征来源主体/记录身份。每个 split 的实际资格计数和分母须与冻结 W0 逐窗口参照核验，不能用过滤预测退化的方式缩小集合。

每个候选分别与同 seed W0 配对：四项 error 的原始差为 `candidate−W0`，相对差为 `100*(candidate−W0)/W0`；PCC 使用绝对下降 `W0−candidate`。正值统一表示恶化，PCC 不计算误差型百分比。保存三 seed 原值、各 arm mean/sample SD（ddof=1）、配对差 mean/sample SD、改善/相等/恶化方向数，以及四项 error 的三 seed 均值之比；后者与逐 seed 百分比的均值分栏。W0 分母为零时百分比为 NA，保留原始差和该 seed，不加 epsilon。

非有限 input、target、prediction、checkpoint 或关键指标显式失败。有限退化预测按原定义保留，记录计数和 `quality_acceptance_passed=false`；其科学结果仍进入完整汇总，不删除样本、不替换 checkpoint、不按效果重跑。工程失败与有限结果的质量标志分开保存。

validation/test 各自输出完整五 arm×三 seed 的 `seed_metrics.csv`（15 行）、四候选×三 seed×五指标的 `paired_seed_delta.csv`（60 行）及四候选×五指标的 `four_arm_comparison.csv`（20 行），另保存来源回执和 manifest。逐 seed 表记录 split、arm、seed、selected epoch、分母和来源 SHA；比较表按 arm/metric 汇总，可回溯到上述配对明细。test 与 validation 不混合为新 selector。三 seed SD 只描述本预算的训练随机性，结果解释结合属性取舍、方向、容量和实测效率。

## 8. 独立实现、锁与生命周期

| 对象 | 路径/身份 |
|---|---|
| 模型 | `resp_train/paper_evidence/e4_aggregation_v2_model.py` |
| Train/validation 控制器 | `resp_train/paper_evidence/e4_aggregation_v2.py` |
| GPU 工程验收/benchmark | `resp_train/paper_evidence/e4_aggregation_v2_engineering.py` |
| Test 控制器 | `resp_train/paper_evidence/e4_aggregation_v2_test.py` |
| Train/validation CLI | `scripts/run_e4_aggregation_v2.py` |
| Test CLI | `scripts/eval_e4_aggregation_v2_test.py` |
| 实现锁 | `docs/experiments/e4_scale_aggregation_v2_implementation_lock_20260917.json` |
| Train/validation 输出根 | `runs/e4_scale_aggregation_v2/` |
| Test 协议身份 | `e4-scale-aggregation-v2-test-v1-20260917` |
| Test 锁 | `docs/experiments/e4_scale_aggregation_v2_test_lock_20260917.json` |
| Test 输出根 | `runs/e4_scale_aggregation_v2_test/` |

实现锁保存协议与代码 SHA、全部 W0 原 config、12 个候选 resolved templates、来源文件 SHA、频率/区域/初始化合同、严格参数数、完整训练与统计定义。配置键 `model.aggregation_v2` 保存完整聚合合同，`model.aggregation_frequencies_hz` 保存 97 元实际频率数组，并由独立构造器逐字段验证。训练实现锁准备读取已冻结 W0/E4 来源 metadata 与频率文件，并重新计算 train/val cache 实体文件的字节 SHA；test 锁准备只读取已冻结结果与 metadata，不加载 test 数组，test 运行阶段再核验其实际 cache 文件身份。旧代码锁继续绑定旧提交，不通过改写历史锁接受新工作树。新构造及控制逻辑由独立模块承载。

每个 arm/seed/phase/lock 使用进程互斥锁，并排他创建包含实现身份、UTC 和 UUID 的新 attempt；时间戳不是唯一 identity。开始保存 lifecycle，成功保存 manifest/freeze receipt，失败保存 failed lifecycle 和完整现场。同身份已完成 cell 拒绝重跑。工程故障排除后以新 attempt 从头执行失败 cell，保留其他已完成 cell；影响实现或科学合同的修订使用新 identity 并说明可比性。

每次正式训练保存 resolved config、命令、代码/来源/实现锁、环境、初始化/训练/sample seed、数据索引及有序 rows/hash、访问记录、optimizer 分组、完整 history、best/final checkpoint 和逐窗口 validation metrics/summary。完成验收要求 history epochs=1…80、updates=80…6400、LR 轨迹正确、best 是最早严格最小 Local RR、final epoch=80/update=6400，checkpoint/config/history/optimizer step 一致。

validation 汇总只接受四 arm×三 seed 的全部 12 个完成 attempt，核验同实现锁、来源、完整 rows 和分母后冻结。test 汇总同样要求全部 12 个已锁 checkpoint 的完成评价，不从中选取有利子集。

## 9. 后续 test 门控及用户执行顺序

先完成 synthetic CPU 定向验证并锁定实现，再由用户运行 GPU 验收/benchmark、12 次 formal 训练及完整 validation 汇总。正式运行使用独立 CLI 的 `--arm`、`--seed` 与实际成功验收目录；每个失败返回值必须停止相应执行流程，保留现场。

test 准备阶段必须显式提供实际冻结 validation summary 目录：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python \
  scripts/eval_e4_aggregation_v2_test.py prepare-lock \
  --validation-summary '/实际完成的v2完整validation汇总目录'
```

`prepare-lock` 核验 summary 的完整 12-cell 矩阵、完成/冻结身份和 SHA，并固定每个 arm/seed 的 `checkpoint_best_local_rr.pt`、selected epoch、完整训练配置/history、checkpoint 字节身份及 train/validation samp_id。新的 test 锁还登记协议/源码/训练锁、W0 test 来源、cache/index/row/frequency identity；排他创建且拒绝覆盖。此时只检查 metadata 与来源字节身份，不解码真实 test 数组。

用户随后按固定四 arm×三 seed 运行 `evaluate --arm <arm> --seed <seed> --device cuda:0`。评价前写 access_started，核验 cache 实体文件/index、有序 test rows 与 W0 的 row/samp/split identity、与该训练实例 train/validation samp_id 的隔离，再用对应独立模型 `strict=True` 加载 allowlist checkpoint。原训练 config 单独保存；test data config 仅增加匹配的 test cache 路径及允许的设备/显示设置。

每次评价固定 2310 窗口、8 samp_id、batch=128、shuffle=false、eval mode、BF16 和完整尾 batch（18×128+6），共生成 **12×2310=27720 条**新逐窗口记录。逐 batch 核对 identity、顺序、W shape、input/target finite；逐窗口 target 属性与冻结 W0 对照一致。评价交付 metrics/summary、test rows、resolved config、锁快照、环境、access/evaluation receipts 和 lifecycle。全部 12 次完成后汇总并冻结第 7 节的完整表。

GPU/真实数据任务均留由用户执行；本协议的代码实现授权不等于 Codex 代跑这些阶段。实施完成时补充实际 synthetic CPU 检查命令、通过数和运行时间，工程实测与科学结果以独立运行产物记录。

## 10. 实现交付与执行命令

2026-09-17 联合验证：**100 passed，71.37 s**，包括模型 52 项、训练控制器 22 项、test 控制器 26 项。模型验证覆盖四候选非零聚合时的 chunked/unchunked 输出、输入梯度及参数梯度等价，公共 optimizer 分组不变；CPU 构造 W0 与四候选时 mock CUDA 播种调用序列一致。GPU 入口还会实际比较选定 CUDA 设备的初始 RNG state。

完整生命周期 fixture 使用 tiny 模型与临时合成波形：train=3、batch=2 保留尾 batch，2 epochs/4 updates；test=4、batch=3+1，2 个合成 samp_id。正式合同仍为第 5 节及第 9 节的完整预算与数据规模。测试中的 checkpoint/cache/index 均为 disposable fixture。没有运行真实数据训练或评价。

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python -m pytest \
  tests/test_e4_aggregation_v2_model.py tests/test_e4_aggregation_v2.py \
  tests/test_e4_aggregation_v2_test.py -q
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python \
  scripts/run_e4_aggregation_v2.py prepare-lock
```

实现锁排他创建，已有锁复用；当前训练锁绑定模型、训练、工程验收、共同依赖及相应测试/协议，test 专属控制器由后续 test 锁独立绑定。提交本轮代码、协议和训练锁后，保持工作树干净，用户执行：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python \
  scripts/run_e4_aggregation_v2.py gpu-acceptance --device cuda:0
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python \
  scripts/run_e4_aggregation_v2.py benchmark --device cuda:0

E4_V2_GPU_RECEIPT='/实际成功的v2_gpu_acceptance目录'
(
  for E4_V2_SEED in 20260811 20260812 20260813; do
    for E4_V2_ARM in static_scale scale_attention frequency_attention channel_region; do
      env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python \
        scripts/run_e4_aggregation_v2.py formal --arm "$E4_V2_ARM" --seed "$E4_V2_SEED" \
        --device cuda:0 --gpu-receipt "$E4_V2_GPU_RECEIPT" || exit $?
    done
  done
)
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python \
  scripts/run_e4_aggregation_v2.py summarize --completed
```

固定顺序为 seed 外层、arm 内层；子 shell 在工程失败时停止，后续只执行尚未完成的 cell。`--completed` 在当前锁对应目录中按 arm/seed 定位唯一成功 attempt，缺失或重复即失败；汇总器仍重新核验全部字节身份与矩阵。两个汇总入口也支持用 `--runs` 显式传入 12 个目录。

test 锁须按第 9 节从已完成的完整 validation 汇总准备并提交；随后用户执行：

```bash
(
  for E4_V2_SEED in 20260811 20260812 20260813; do
    for E4_V2_ARM in static_scale scale_attention frequency_attention channel_region; do
      env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python \
        scripts/eval_e4_aggregation_v2_test.py evaluate \
        --arm "$E4_V2_ARM" --seed "$E4_V2_SEED" --device cuda:0 || exit $?
    done
  done
)
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python \
  scripts/eval_e4_aggregation_v2_test.py summarize --completed
```

训练实现锁可在当前阶段生成；test checkpoint allowlist 由后续 12 次训练的实际 selector 结果生成，当前不预填 selected epoch。
