# PATCH/TM0/REF0 模块消融与 APOR：统一训练协议

日期：2026-09-30。协议：`patch-apor-module-v1-20260930`。

状态：13臂模型与双GPU调度已实现；正式矩阵为 **13 臂×3 seed＝39 cells**。当次授权的GPU验收与train/validation运行状态以对应session回执为准。

## 1. 模型与比较关系

M0 严格沿用三因素实验的 `PATCH/TM0/REF0`：D96、六层双向 Mamba2、97-scale W、H65、原 FiLM 与 C201 读出。E9 的 D96/H64 属于其自身冻结选择，当前 M0 仍绑定三因素实验来源。

| Arm | 模块定义 | 对照 |
|---|---|---|
| M0 | 完整 PATCH/TM0/REF0 | 基准 |
| M1 | 删除 W 分支与 FiLM；数据加载不打开 W cache | M0 |
| M2 | 六层 BiMamba2 替换为空 block stack | M0 |
| M3 | W conv_in (5,3)→(1,3)，depthwise (3,3)→(1,3) | M0 |
| M4 | H65 条件残差替换为 Identity | M0 |
| M5 | 删除 C201 residual；保留完整 base head 与 Fourier 插值 | M0 |
| M6 | FiLM 仅 beta，z'=z+0.5tanh(beta) | M0 |
| M7 | FiLM 仅 gamma，z'=z×(1+0.5tanh(gamma)) | M0 |
| A0 | 完整 APOR | M0 |
| A1 | APOR 五位置条件改为中心单点 | A0 |
| A2 | APOR MLP decoder 改为 Linear(96,256) | A0 |
| A3 | APOR 主干在真实坐标1800点网格运行，再采回140点 | A0 |
| A4 | APOR overlap-add 权重改为 uniform | A0 |

每个变体均相对对应基准独立改变模块，从头训练。公共模块的初始化逐tensor共享；M3新卷积、APOR条件适配和decoder使用独立命名子seed。M6/M7仅注册所需96通道投影，并从同seed原零初始化投影的beta/gamma部分取值。

CPU构造实测参数数量：M0=1,031,690；M1=994,538；M2=80,306；M3=1,030,826；M4=1,019,114；M5=1,030,633；M6/M7各1,022,378；A0/A3/A4各1,077,640；A1=1,040,776；A2=1,068,328。模型构造时验证这些数量。参数数不能替代GPU时延或完整计算量测量。

M2保留PatchMixer与局部时间卷积，结果只回答主干的增量作用。M3保留时间核、通道、GN及scale mean，结果针对尺度邻域卷积。A3改变时间网格也改变局部卷积的物理跨度与GN统计，按整个时间网格设计解释。

## 2. APOR精确计算图

输入 `[B,1,18000]`，右补48个零，L256/S128产生140个patch。原16通道Patch encoder与两层mixer → 原16→96 adapter/GN/SiLU → 六层双向Mamba2，得到 `[B,96,140]`。

W `[B,97,360]` 经原浅层二维编码与scale mean得到 `[B,96,360]`。每个patch内提取五个位置，展开为480通道，经1×1线性投影到96，经过H65残差与原zero-init FiLM投影。FiLM始终位于主干后。

物理坐标以原100-Hz采样索引计：

- Patch j起点128j，中心128j+127.5；局部条件位置为0、63.75、127.5、191.25、255。
- CWT槽位k中心50k+24.5，间距50。坐标线性取样，边界最近值延拓。
- A1仅中心127.5，条件适配为96→96；其余结构与A0相同。
- A3先将原16通道patch特征采样到0、10、…、17990，运行adapter与主干，再在FiLM前采样到patch中心。

调制后的每个token由 `Linear(96,96)→SiLU→Linear(96,256)` 生成局部有符号片段。A2使用单层Linear(96,256)。以S128重叠合成到18048点，裁回18000；Hann窗口periodic=false并clamp_min=1e-3。A4使用全1权重。

`y[n] = Σ_j w[n−128j]p_j[n−128j] / Σ_j w[n−128j]`。

坐标权重与重叠累加使用float32。输出仍为raw waveform，原外置Pi执行一次。APOR直接替换旧coarse head/C201 residual/Fourier读出，没有未使用的旧读出参数。

140个token是多维局部表示，每个token输出256点片段；不能把它视为140个波形标量。工程fixture检查0.7-Hz、变频、幅度变化、首尾覆盖及合成梯度；正式质量还需训练结果验证。

## 3. 数据、训练与指标

- Train/validation沿用10141/2675窗口、32/7个samp_id、180 s/100 Hz和既有subject/session划分。
- Seeds：20260811、20260812、20260813；sample seed：train20260610、val20260611。
- 原 `L_sync + 0.25 L_effort`、Pi、eligibility与五指标完整复用。
- physical/effective batch128，accumulation1，BF16；原AdamW分组、学习率3e-4→3e-5、warmup0.05、gradient clip1。
- 每epoch80次更新；最多80epochs，学习率始终按6400updates计划；min_epoch30、patience15、min_delta0。
- 完整validation Local RR严格最小选择checkpoint，并列最早。每个cell单独early-stop，矩阵不根据partial结果裁剪。
- 五指标：Whole RR、Local RR、envelope trajectory、global envelope modulation、lag-aware signed PCC。

来源元数据复用三因素实验冻结文件 `w0_structural_factorial_v1_implementation_lock_r2_20260924.json`，固定SHA256为 `32141eab672ea41435055c45cbd7ec96325f8ddef2481a210db2941222c9e9f3`。只使用其中的数据索引、train/validation rows与W cache身份，不加载历史checkpoint初始化。

正式运行前复核dataset index和适用cache字节身份、row集合/顺序、samp_id数及隔离；实际loader再次核对样本。输入、输出、loss、梯度、checkpoint与关键指标非有限均失败。prediction退化计数单独保留，不静默排除。来源索引固定数据口径；本协议不重新进行原始NPZ全量字节审计。

本入口只构建train/validation。Research-test需后续匹配专项协议及当次授权。

## 4. 生命周期与验收

单个session统一覆盖全部39个cell与GPU验收，使用独立 `runs/patch_apor_v1/session_<UTC>_<uuid>`。session.json保存resolved配置、源码哈希、完整Python源码快照、Git状态和依赖版本；session_receipt绑定其哈希。无需重复建立阶段实验锁。提交状态与源码快照共同保存，可在包含本次未提交实现的工作树运行。

后续运行回载验证源码集合、字节哈希、resolved配置和环境。运行中修改受管代码会阻止后续cell启动；修复涉及实现身份时应建立新session，不混用旧cell。串行运行与汇总取得会话独占锁；并行worker持会话共享锁，并分别取得分片与设备独占锁。调度表写入后不可改变分片数量或设备映射。

双GPU固定按39-cell计划交错分片：GPU0处理第0、2、4…个cell，共20个；GPU1处理第1、3、5…个cell，共19个。每臂三个seed分布到两张卡，分片集合互斥且并集等于完整矩阵。每张卡使用独立进程，OMP/MKL/OpenBLAS线程各设4，依旧每个cell使用physical batch128、accumulation1。该分片不改变训练样本、更新预算、loss或selector。

GPU synthetic验收先覆盖13臂×3seed、batch1各3次原生训练更新，再覆盖13臂batch128各3次更新与batch128 eval/loss。检查shape、finite、optimizer覆盖、第三步逐模块非零梯度与参数更新，GPU peak reserved/total≤0.85。FiLM零初始化下前两步用于打开条件内部梯度。

每个worker在其GPU验收通过后才读取真实数据；验收产物分别保存到engineering/cuda_0与engineering/cuda_1。worker按固定分片顺序训练，不根据中间结果重新分配cell。任何worker失败时，调度器通知另一worker以SIGINT停止并保留现场，不自动改变batch、结构或精度。全部worker成功退出后，父进程取得会话独占锁并汇总39个cell。

每个attempt排他创建started、failed或manifest/freeze receipt。成功cell回载验证后跳过；失败/中断需用户检查后使用 `--retry-failed` 创建新attempt，原产物保留。此选项从头重试该cell，不从partial checkpoint续训。

训练完成核验原生config、history停止/学习率轨迹、best/final checkpoint、初始化身份、optimizer state、逐窗口指标与summary。完整39-cell后生成：per_seed、across_seed、paired_delta、paired_summary、per_subject、subject_macro、local_rr_tail、denominators及来源manifest。summary只读冻结validation指标。

seed SD描述优化随机性；subject-macro按samp_id等权，窗口和seed均不作为独立人群重复。

## 5. 执行命令

仓库根：`/mnt/disk_code/marques/resp_reconstruction`。

CPU定向验证：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. ./.venv/bin/python -m pytest tests/test_patch_apor_v1.py -q
./.venv/bin/python scripts/run_patch_apor_v1.py plan
```

双GPU完整入口，一次启动两个worker，各自完成synthetic GPU验收与训练，最后统一汇总：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  ./.venv/bin/python scripts/run_patch_apor_v1.py parallel \
  --devices cuda:0 cuda:1 --confirm-training
```

启动后打印 `SESSION=/绝对路径`及两个worker的PID/日志路径，保留这些信息。日志与调度回执位于session/dispatch/attempt_*。已有会话的继续执行：

```bash
PATCH_APOR_SESSION='/实际打印的session绝对路径'
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  ./.venv/bin/python scripts/run_patch_apor_v1.py parallel \
  --session "$PATCH_APOR_SESSION" --devices cuda:0 cuda:1 --confirm-training --retry-failed
```

查看状态与只读指标汇总：

```bash
./.venv/bin/python scripts/run_patch_apor_v1.py status --session "$PATCH_APOR_SESSION"
./.venv/bin/python scripts/run_patch_apor_v1.py summarize --session "$PATCH_APOR_SESSION"
```

也提供prepare、describe、gpu-acceptance、formal单cell及run-all串行入口，参数见 `--help`。run-shard由并行调度器调用并核对固定分片。没有指定session的parallel/run-all代表新的独立运行身份；恢复现有矩阵必须显式给出session。

## 6. 当前实现验证

实现文件：`resp_train/paper_evidence/patch_apor_v1_model.py`、`patch_apor_v1.py`；配置：`configs/patch_apor_v1/experiment.yaml`；入口：`scripts/run_patch_apor_v1.py`。

CPU测试覆盖三seed M0初始化与非零FiLM计算等价、13臂结构与公共参数、原生loss三次更新、状态回载、物理坐标、重叠重建、nonfinite失败、39-cell汇总和不可覆盖生命周期。CPU模型前后向fixture将Mamba替换为Identity；原生Mamba GPU验收为正式训练前的必经步骤。

2026-09-30：完整定向集合33项通过（29.98 s），包含M0/M1/A0的原生训练器两epoch合成生命周期及收尾验证。模型参数构造与CLI describe/plan也已检查。GPU、真实数据训练与research-test未在实现阶段执行。
