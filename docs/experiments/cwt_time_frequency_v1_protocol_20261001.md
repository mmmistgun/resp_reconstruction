# CWT 时频信息与频带作用 v1：实现与分阶段执行协议

日期：2026-10-01。协议 ID：`cwt-time-frequency-v1-20260930`。

状态（2026-10-01检查更新）：**初始12项定向测试通过（11.10 s）；新增仅合成会话访问边界测试通过（1项，5.29 s）；17个Python文件语法检查及79文件源码依赖闭包检查通过。** 当次用户授权后台完成合成校准与GPU工程验收；真实数据、训练及research-test不在本次执行范围。后台运行结果以独立attempt回执为准，不由此声明GPU已通过。

研究范围和对照理由见 [设计记录](cwt_time_frequency_v1_research_design_20260930.md)。20 配置×3 seed=60 cell，原 W0 为同期基准 B；固定 80 epochs，关闭早停。L 与完整 B 在各子矩阵共享，TF-S5 只新增 H。历史训练不抵扣本轮同期矩阵。

## 1. 实现入口和独立性

- 配置：`configs/cwt_time_frequency_v1/experiment.yaml`。
- 实现包：`resp_train/paper_evidence/cwt_time_frequency_v1/`。
- 准备/训练/validation：`scripts/run_cwt_time_frequency_v1.py`。
- Research-test：`scripts/eval_cwt_time_frequency_v1_research_test.py`。
- 测试：`tests/test_cwt_time_frequency_v1.py`。
- 输出：新 worktree 下的 `runs/cwt_time_frequency_v1/`。

新包复用 `CRDExperiment`、原数据资格规则与 Dataset、optimizer、Pi、loss、metrics、W0 CWT 编码层和 GN 重放函数。原入口的固定缓存身份及 49/97-scale 合同保持原样。新 Dataset wrapper 只为 batch 添加已核验的 W；原样本、target 和 meta 沿用原 Dataset。

## 2. 变换与模型合同

- B 严格调用历史 `morlet_scales_and_frequencies(12,97)`；μ=13.4，名义 0.03125–8 Hz，pool50，97×360。保留重复离散中心频率。
- μ∈{6,13.4,20}、voices∈{4,8,12,24}，八 octave 基础名义网格为 33/65/97/193 个尺度。其他 μ/voices 使用与历史相同的 `freq_to_scale` 参数与 ssqueezepy 0.6.6。
- C_0p8、C_2、C_4 从基准尺度数组按实际中心频率 `≤上限` 取子集；H 按 `(0.8,8]` 取子集，均以原生尺度数进入编码器。
- C_12、C_20 保留所有 B 尺度，按 `base_min_scale × 2^(-k/12)` 向高频延伸，再按实际中心频率筛选上限；不会移动既有低频尺度。
- 先 `log1p(abs(CWT))`，再对25/50/100点做不重叠均值，产生720/360/180帧。输入为有限 float32 18000点，reflect padding；全部 actual scales/frequencies、频率排序、时间坐标与重复中心数随 session 保存。
- 原 W0 编码器保持卷积槽位核、GN、scale mean、线性对齐1800点、三个 temporal blocks、H65、0.5/0.5 FiLM及原读出。全部臂理论参数量为1,219,850，构造时强制核验；同 seed 完整参数初始化逐张量核验。

代码中的端点规则是本轮待校准的具体实现。`calibrate` 保存所有表示的响应和合成样本，并核对 B 与旧变换、L/H 与 B 子集的实际数值。默认容差 `rtol=1e-6, atol=1e-7`；失败保留现场。

`prepare --parameter-review` 保存用户对校准结果、20-Hz高端点和预处理有效带宽的核查说明。代码不会凭校准 shape/finite 成功自动宣称高频有任务价值。若频率定义需改变，先修订配置/实现/协议，再产生新校准及新 session。

仅做工程验收时使用`prepare-engineering --calibration ...`，session固定`execution_scope=synthetic_only`、`parameter_review=null`；真实数据准备、训练与test入口显式拒绝该会话。后续进入真实数据阶段需要按上述核查要求另建`train_validation`会话，不能修改工程会话范围。

本次授权的后台串联入口为`scripts/run_cwt_time_frequency_v1_engineering.py --devices cuda:0 cuda:1 --confirm-synthetic`。它先执行合成校准，再建立仅合成会话，两卡各自独立执行完整验收；任一worker失败会停止另一worker并保留日志。不会调用prepare-data、cache、signals、formal或test。

首次后台启动在合成真值NPZ写入时发现`transient`字段重名，尚未进入GPU阶段。分量字段已改为`transient_component`，新增NPZ导出回归测试通过（1项，4.46 s）；原失败attempt保留，修复后使用新源码身份重新校准。

## 3. 数据、训练和选点

沿用 W0 基础配置解析后的科学合同：train/validation=10141/2675窗口、32/7个samp_id，180 s/100 Hz；sample seed=20260610/20260611。训练 seed=20260811/20260812/20260813。

原 `L_sync + 0.25 L_effort`、任务 Pi 与五项指标函数保持不变。Batch128、accumulation1、BF16、AdamW、LR3e-4→3e-5、warmup0.05、gradient clip1；80 updates/epoch、80 epochs、6400 planned updates。完整 validation Local RR 严格最小选择 checkpoint，并列保留最早。

`prepare-data`核对历史来源锁、数据索引字节身份、train/val row集合、顺序及隔离；冻结使用的源/target NPZ字节身份。该操作需要读取这些文件用于哈希，属于用户执行的数据准备阶段。原波形与缓存不修改。

公共 validation reference 在准备阶段由单一进程保存；训练 worker 只读逐元素核验。每臂缓存独立绑定 session、表示、数据来源、row顺序、库与源码。缓存读取核验完整文件身份、实际shape/dtype/row，逐样本再次检查finite。本版直接构建各臂缓存，不自动复用历史缓存或推断0.25 s特征。

每 cell 保存解析配置、初始化身份、optimizer分组、80轮history、best/final checkpoint、完整metrics、selected validation prediction、row IDs、reference来源及资源信息。新包不改旧 trainer 的 per-epoch logging；运行总耗时和峰值显存另外记录。

## 4. TF-S1 信号分析定义

合成校准覆盖周期漂移、共同幅度调制、独立幅度调制、瞬态干扰，固定种子20260930；保存真值、波形、各臂CWT。冲激响应记录有限180 s窗口内的功率半高宽、99%能量半径、首尾能量比例，以及五/七尺度卷积实际覆盖。低频响应受有限窗长影响，报告值不作为无限长小波支持的估计。

真实信号阶段使用C_20表示，完整覆盖train/validation：

- 原BCG在0.03–0.8、0.8–2、2–4、4–8、8–12、12–20 Hz内的Hann功率谱密度积分。
- 各尺度log幅度在2 Hz上去均值、Hann窗的调制谱，保存逐窗口PSD数组、频率坐标和有序rows；频率分辨率1/180 Hz。
- CWT幅度序列投影到0.05–0.70 Hz；THO按原Pi频带投影后以同50点窗口均值对齐。零时延有符号与绝对Pearson关联分别保存；包络使用10 s/5 s log-RMS。
- 各尺度按合成冲激99%能量半径裁掉两侧支持；三个SHIFT与FULL采用同一支持。支持不足或标准差≤1e-8时记NA并报告分母，非有限输入则失败。
- FULL与每row三个共同循环SHIFT；PCG64，偏移60…300帧无放回抽样；train与val使用明确不同种子，保存实际数组。按samp_id及现有residual_quality_class汇总。
- S3分层：参考局部RR曲线相邻有效点绝对变化均值、参考log-RMS包络相邻变化均值；train的1/3和2/3分位数固定阈值，val/test直接应用。RR无相邻有效点时另列undefined层。

压缩前信号的来源未提供，本版不加载其他输入键猜测其身份。预处理滤波器的实际传递响应需由参数核查说明引用正确来源；信号功率分布不能替代滤波响应核查。

## 5. GPU验收、生命周期和恢复

验收覆盖三个seed的B原生/新模型非零FiLM等价（FP32、BF16），以及20臂×3seed的batch1三步更新、20臂batch128三步更新及eval。逐模块检查梯度和更新，检查finite、shape与显存reserved/total≤0.95。CPU测试仅覆盖独立组件，不能替代原生Mamba GPU验收。

2026-10-01显存策略修订：用户明确允许放宽到90%乃至95%。原两卡验收均在`Q_mu6_v24`的batch128完成更新/eval后触发85%资源门槛，日志未报CUDA OOM。硬上限现为95%，超过90%的记录设置`elevated_memory=true`；保存allocated/reserved峰值字节、占比及设备总显存。记录先落盘再判断门槛，资源失败仍保留测量值；实际OOM继续显式失败。此修订不改变batch128、精度、模型、训练或科学指标。

已完成的合成校准不重跑。后台入口可指定`--calibration /已完成校准attempt`；复用时核验原manifest/receipt/全部快照与产物，要求数学规格、依赖环境、源码集合一致。只允许工程验收文件、后台入口、定向测试及本协议四个文件的显式差异，记录每个差异的旧/新身份。CWT变换、模型、校准计算等任一其他文件改变均拒绝该复用路径。新建仅合成session及GPU验收attempt，保留原失败session。

本修订的显存边界、测量一致性与校准复用范围测试共6项通过（5.34 s）；来源校准完整产物核验及变换依赖不变检查已通过。GPU新运行状态由对应新session回执记录。

另外在B三个seed的batch1/128上运行18个机制条件，核对原生GN同源重放、固定统计公式、FULL_FIXED等价和受保护输入，并保存独立GN验收回执。

每个session只有一份源码/配置/环境身份。源码集合从明确入口和静态import闭包建立，核验冻结清单中的文件与快照；添加无关Python文件不会自动使旧session失败，修改实际依赖会失败。prepare前应把实现修复完毕；运行中发生源码变化时建立新session。

各阶段创建独立attempt，失败写traceback并保留文件。成功阶段再次调用只读核验并返回已有路径；失败重试需要`--retry-failed`，它重新开始该阶段。`formal --retry-failed`从头训练该cell。

训练的最终checkpoint/history写完后，先写`training_complete.json`绑定字节身份，再开始validation导出。如仅收尾失败，使用`recover-export`：核验完整80轮和固定checkpoint，输出到新的exports attempt，不改原失败目录，不重训。

`parallel`固定按完整60-cell计划交错分片；设备映射首次写入后不变。每worker独立进程、单GPU、OMP/MKL/OpenBLAS各4线程；worker失败会通知其余worker停止并保留现场。此机制是cell级并行。

## 6. TF-S6/S7与research-test

干预只使用B三个固定selected checkpoints。H=(0.8,8]、H2=(2,8]按实际频率定义；每区域WINDOW_MEAN、三个共同SHIFT，分别采用NAT和ALL_W_GN_FIXED，另含共享FULL/NAT和FULL/FIXED，共18条件/seed。

固定统计来自同窗口FULL的四层W-GN。每次hook检查同源重放与固定公式，异常也移除hook；检查BCG、原W与FiLM前Z保持一致。FULL/FIXED重放FULL/NAT采用`rtol=1e-5, atol=1e-6`，FULL/NAT均值另核对常规validation/test指标`rtol=1e-3, atol=0`。

机制导出使用batch8，所有条件同batch配对。固定每subject在row顺序中的首/中/末案例，完整180 s，不依据预测效果挑选。案例保存实际CWT、频率/时间坐标、gamma_raw/beta_raw/g/b/Z/Z_prime、修正量、配对预测与Z_prime差、参考/预测指标曲线及有效掩码。全样本保存紧凑指标及GN统计，运行前检查估算磁盘空间。

全矩阵validation和signals完成后才可生成60-checkpoint allowlist，包含较差配置；每臂选点仍仅取自己的完整validation Local RR。allowlist生成不打开test数据。

Research-test的prepare-data、cache、evaluate、mechanisms需要当次命令的`--confirm-research-test`。检查冻结2310-window row身份、8人及与train/val隔离；使用同一表示构建独立test缓存。输出full=2310/8、exclude670=2231/7、subject670=79/1三视图，保持相同allowlist；这里的test属于复用研究测试证据。

## 7. 用户执行命令

以下为分阶段命令。测试执行状态见文首；正式数据和训练命令仍未执行。先进入本worktree，并使用已存在的解释器，不安装或升级依赖：

```bash
cd /home/marques/.codex/worktrees/cwt-time-frequency-v1/resp_reconstruction
CWT_PY=/mnt/disk_code/marques/resp_reconstruction/.venv/bin/python
export PYTHONPATH=.
```

后续允许检查时，先执行定向测试和矩阵检查：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$CWT_PY" -m pytest tests/test_cwt_time_frequency_v1.py -q
"$CWT_PY" scripts/run_cwt_time_frequency_v1.py plan
```

合成校准后填入打印的绝对路径，并记录真实的参数核查结论：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD "$CWT_PY" scripts/run_cwt_time_frequency_v1.py calibrate --confirm-synthetic
CWT_CALIBRATION='/实际校准attempt绝对路径'
"$CWT_PY" scripts/run_cwt_time_frequency_v1.py prepare --calibration "$CWT_CALIBRATION" \
  --parameter-review '填写频率映射、边界影响和预处理有效带宽核查结果及依据'
CWT_SESSION='/实际session绝对路径'
"$CWT_PY" scripts/run_cwt_time_frequency_v1.py prepare-data --session "$CWT_SESSION" --confirm-data
"$CWT_PY" scripts/run_cwt_time_frequency_v1.py cache --session "$CWT_SESSION" --arm C_20 --confirm-data
"$CWT_PY" scripts/run_cwt_time_frequency_v1.py signals --session "$CWT_SESSION" --confirm-data
```

完整构建其余缓存，C_20的成功结果会核验后复用：

```bash
for CWT_ARM in Q_mu6_v4 Q_mu6_v8 Q_mu6_v12 Q_mu6_v24 Q_mu13p4_v4 Q_mu13p4_v8 B Q_mu13p4_v24 \
  Q_mu20_v4 Q_mu20_v8 Q_mu20_v12 Q_mu20_v24 P_025 P_100 C_0p8 C_2 C_4 C_12 C_20 H; do
  env -u LD_LIBRARY_PATH -u LD_PRELOAD "$CWT_PY" scripts/run_cwt_time_frequency_v1.py cache \
    --session "$CWT_SESSION" --arm "$CWT_ARM" --confirm-data || break
done
```

双GPU入口先各自synthetic验收，再运行固定cell分片，最终汇总：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD "$CWT_PY" scripts/run_cwt_time_frequency_v1.py parallel \
  --session "$CWT_SESSION" --devices cuda:0 cuda:1 --confirm-training
```

也可单独运行`gpu-acceptance --device cuda:0 --confirm-synthetic`和`formal --arm B --seed 20260811 --device cuda:0 --confirm-training`；均须提供`--session`。指定cell的收尾恢复：

```bash
"$CWT_PY" scripts/run_cwt_time_frequency_v1.py recover-export --session "$CWT_SESSION" \
  --arm B --seed 20260811 --source-attempt '/原失败formal attempt绝对路径' --device cuda:0 --confirm-inference
```

机制评价与汇总：

```bash
for CWT_SEED in 20260811 20260812 20260813; do
  env -u LD_LIBRARY_PATH -u LD_PRELOAD "$CWT_PY" scripts/run_cwt_time_frequency_v1.py mechanisms \
    --session "$CWT_SESSION" --seed "$CWT_SEED" --device cuda:0 --confirm-inference || break
done
"$CWT_PY" scripts/run_cwt_time_frequency_v1.py summarize-mechanisms --session "$CWT_SESSION"
"$CWT_PY" scripts/run_cwt_time_frequency_v1.py prepare-allowlist --session "$CWT_SESSION"
```

用户开放本轮test后，独立test入口按`prepare-data → 各arm cache → 各arm/seed evaluate → summarize`顺序执行。前三类操作均加`--confirm-research-test`；不提供自动train→test联跑以混淆两阶段访问边界。test机制另用`mechanisms --seed ... --device ... --confirm-research-test`及`summarize-mechanisms`。

## 8. 验收与资源边界

定向CPU检查已通过；实际频率映射、GPU兼容性、GN精度及显存预算仍需相应回执验证。完整矩阵最多384,000次optimizer更新；正式训练GPU小时尚无测量。

缓存按`窗口数×尺度数×帧数×4 bytes`估算；每cell的validation prediction约184 MiB。signals额外保存C_20逐窗口调制谱，机制保存固定案例与完整GN统计。checkpoint、各attempt源码快照和失败现场另计。正式执行前根据calibration实际shape核算总预算，资源不足时失败，不自动缩小batch、矩阵或分母。

交付结果包含per_seed、across_seed、paired_delta、paired_summary、per_subject、subject_macro、per_subject配对、subject_macro配对、local_rr_tail、denominators及S3参考变化分层。差值正值表示变差：error=候选−参照、PCC=参照−候选；零分母的相对变化明确标为undefined。
