# W0 R3 幅度时序与 GN 统计控制：执行协议

协议 ID：`e4-r3-temporal-normalization-v1-20260921`。日期：2026-09-21。

状态：依据 [合并设计](e4_r3_temporal_normalization_design_20260921.md) 完成独立运行入口与 synthetic CPU 验证：**22 项通过，完整集合 21.39 s；最后的显存引用优化另复核 2 项关键流程通过（17.44 s）**。真实信号检查、GPU smoke、validation 干预与最终汇总留由用户执行；尚未运行。

## 1. 来源与执行边界

问题：R3 幅度时间结构与当前输入/THO 的对应是否被模型利用，以及入口替换导致的 GN 统计漂移是否解释部分损失。该实验只分析冻结 checkpoint，不训练、不改变 selected epoch、不读取独立 test。

固定原 W0 三 seed：20260811/13 epoch、20260812/15 epoch、20260813/14 epoch。原始模型 `crd_tf102_w`、六层主干、fill=65，FiLM gamma/beta 均为 0.5。当前原生代码包含其他专项的 FiLM 参数入口，本入口显式拒绝非原值，并通过原生 FULL 重放核对行为。

来源锁 `e4_band_encoding_aggregation_lock_20260918.json`，SHA `41084408b8b597ed7daaf6ecf081083e25397a25a647db035b692205989d6879`。来源 checkpoint/config/冻结 validation CSV 的字节身份在新锁中登记。旧代码锁仅作为历史来源；本次对当前执行代码建立独立快照，不修改历史锁以接受代码变动。

训练均值复用既有 `runs/e4_band_encoding_aggregation_v1/reference/reference_41084408b8b5_20260918T112447Z_1498cff12931/reference.npy`。准备锁核验参考 manifest 及原 train cache 身份关联，只读取来源文件、metadata 和字节身份，不重新计算 train 参考。

完整 validation=2675 窗口、7 个 samp_id，原有序 rows/sample seed=20260611、BCG/THO、eligibility、五指标和分母。batch=128，保留 115 个窗口的尾 batch，BF16 autocast、eval/no_grad。

## 2. 固定 22 条件/seed

输入变换为 FULL、TRAIN_MEAN、WINDOW_FLAT、SHIFT_1、SHIFT_2、SHIFT_3、REVERSE。R3 始终是槽位 `[73:97]`，其余 W、BCG、target 不变。

| 变换 | 定义 |
|---|---|
| FULL | 原输入 |
| TRAIN_MEAN | 复用冻结训练集逐尺度均值，覆盖 R3 时间轴 |
| WINDOW_FLAT | 每窗口、每尺度自身 360 点均值，覆盖该尺度时间轴 |
| SHIFT_1/2/3 | R3 全部 24 尺度共同循环平移同一偏移 |
| REVERSE | R3 全部尺度共同反序 |

SHIFT 按冻结 row 顺序，使用 `numpy.random.Generator(PCG64(20260920))`，每窗口从 bin 60…300 无放回抽三个偏移，对应 30–150 s；实际数组写入实现锁。三 seed 使用同一数组；不根据 target/模型效果重抽。值分布及离散傅里叶功率保持不意味着卷积后 GN 统计不变，也不保证每次偏移都能充分破坏周期对应。

每个变换分别使用 NAT、GN1_FIXED、ALL_W_GN_FIXED，共 21 条件，额外增加 STAT_ONLY_TRAIN_MEAN，总计 **22×3=66 条件、176550 条条件窗口评价**。

| 模式 | 执行 |
|---|---|
| NAT | 全部 GN 正常计算当前输入统计 |
| GN1_FIXED | 仅第一层 GN 使用同窗口 FULL 的统计，后续 GN 正常 |
| ALL_W_GN_FIXED | W 分支四层 GN 均使用同窗口 FULL 的对应层/组统计 |
| STAT_ONLY_TRAIN_MEAN | 输入 FULL；仅第一层 GN 使用同窗口自然 TRAIN_MEAN 的统计 |

四层路径固定为 `norm`、`temporal.blocks.0.norm`、`.1.norm`、`.2.norm`。第一层 8 组、temporal 每层 12 组；三层 temporal 统计来自原 FULL 正常轨迹。其余网络归一化保持正常，不能将结论外推为全网络归一化均已控制。

## 3. GN 数值定义与重放验收

使用原生 GN 算子在原实际输出 dtype 下记录每窗口/组均值及逆标准差 `rstd=1/sqrt(var+eps)`。rstd 是重放的主记录，方差描述由 `max(0,1/rstd²−eps)` 数值重构，不替代原生 rstd。均值、rstd 必须有限，rstd 必须为正；形状必须对应当前 batch、当前层的组数。

固定统计的输出按等价仿射实现：`scale=gamma*rstd`，`bias=beta−mu*scale`，用融合乘加 `bias+U*scale` 计算后回到原 GN 输出 dtype。保留原 affine 参数、epsilon 和有偏方差定义。

已对照当前 PyTorch 2.12.0 的 [CUDA GN 源码](https://raw.githubusercontent.com/pytorch/pytorch/v2.12.0/aten/src/ATen/native/cuda/group_norm_kernel.cu)：当前空间输入走逐通道 affine 路径。CUDA bias 同样使用融合乘加计算；CPU 路径按合成重放验证。源码核对不替代用户运行 GPU smoke。

每次 forward 都核对同源统计重放与原生 GN 输出，容差 `rtol=1e-5, atol=1e-6`，记录实际最大误差。不得以“统计相同就直接返回原输出”代替此验收。CPU 验证不替代 CUDA 验收；GPU 不达标时停止诊断并保留失败现场。

同一 batch 先计算 FULL/NAT，再保存短期基准层间 tensor 与四层统计；随后的 21 条件共享该窗口基准。自然 TRAIN_MEAN 的第一层统计提供给最后的统计量单独变化条件。每次调用结束移除 hooks，异常路径也必须移除；不修改模型参数或原 cache。

## 4. 信号对应检查：单独运行一次

`signals` 阶段用 CPU 读取固定 validation 输入，模型不参与推理，三个模型 seed 共享同一冻结回执。主描述为 `a(t)=mean_{s∈R3} W_s(t)`，保留全部 24 个单尺度的对应结果，不根据 THO 选择尺度或权重。

- 原 W 是 `log1p(abs(CWT))`、每 50 点平均，采样率 2 Hz。THO 使用原任务 `canonicalize_numpy` 的 FFT 带投影，再以相同 50 点窗口平均。bin 中心为 `(50j+24.5)/100` 秒。
- 周期关联：R3 幅度图用原 FFT 投影定义适配 2 Hz，通带 0.05–0.70 Hz；与投影/池化后的 THO 计算零滞后 signed correlation 和预定义 absolute correlation。
- 相对幅度关联：在两个周期曲线上使用原任务 10 s 窗、5 s 步长、eps=1e-8 的 log-RMS 包络。2 Hz 上对应 20 点窗口、10 点步长；滑窗批量实现经测试与冻结公式一致。
- 低方差阈值使用原 `dynamic_eps=1e-8`，有一侧标准差不超过该值时关联记 NA/low_variance，记录有定义数。它不改变主任务的窗口集合。
- 辅助周期 lag profile 只用于 R3 平均曲线。原范围为 ±0.3 s、100 Hz lag 网格，即 61 个 lag；由于幅度图是 2 Hz，明确使用线性插值取样，并在所有 lag 上固定裁去两端各一个幅度 bin 的公共支持区。正 lag 表示取 R3 在 `t+lag` 的值与 THO 在 t 的值比较。不选择最优 lag，不据插值声称具有 0.01 s 的实际时间分辨率。包络不做此细粒度扫描。
- 每个变换的所有尺度零滞后结果及主曲线完整 lag profile 均保存。原样—干预关联差按同窗口/表示/支持区配对，保留配对有定义数；按 samp_id 汇总后给等权宏平均。
- 21 个展示 row 复用旧锁预选的各 samp_id 首/中/末窗口。图包含原始 R3 平均 log-magnitude、周期对齐（只做显示缩放，极性不变）、相对 log-RMS 包络，以及原样/三错位/反序/均值条件。图和 NPZ 保存相同时间轴与曲线。

信号关联只支持呼吸相关性；与模型干预共同解释时才讨论是否被模型利用。THO 包络是归一化输入下的相对幅度量，不是校准潮气量；时间错位还可能影响 R3 与 BCG/其他尺度的跨路径一致性。

## 5. 完整 validation 与配对

每 seed 先完整运行当前原生 W0，核对冻结五项 validation 均值，门槛 `rtol=1e-3, atol=0`；通过后才开始干预矩阵。该工程基准额外计算不计入 66 个科学条件。

对每个 batch 顺序运行 22 条件，FULL/NAT、FULL/GN1_FIXED、FULL/ALL_W_GN_FIXED 的 waveform 均须复现原生 FULL，`rtol=1e-5, atol=1e-6`。调用原生五指标函数逐 batch 评价，窗口级指标独立，完整拼接后使用原生汇总函数；保留完整尾 batch。FULL/NAT 再核对冻结均值，所有条件核对 identity、target-only eligibility、样本集合和质量标志。有限退化预测保留且记录，非有限预测或关键指标显式失败。

差值统一正值为变差：error=condition−FULL，PCC=FULL−condition。保存各 seed pooled、各 samp_id、samp_id 等权宏平均以及三 seed mean/sample SD 和方向数。三错位、模型 seed、七个 samp_id 和重叠窗口不合并为独立重复。

TRAIN_MEAN 2×2 使用同一 oriented difference：content 为 TRAIN_MEAN/GN1_FIXED，statistics 为 STAT_ONLY，total 为 TRAIN_MEAN/NAT，interaction=total−content−statistics。只对这个完整 2×2 计算差分恒等式，不对缺少统计量单独条件的 SHIFT/REVERSE 计算机制贡献，更不把比值包装成机制解释比例。

## 6. 层间诊断和资源

每窗口/条件保存 conv_in、第一层 GN 后、聚合 z、原始 FiLM gamma/beta 的基准 RMS、变化 RMS、相对变化；基准为零时比例为 NA。前两层另记录保护区域 `[0:71]` 的变化，检查第一层统计传播和卷积局部性。GN 表逐层/组保存自然统计、实际使用的统计、FULL 统计、标准化均值漂移及含 epsilon 的方差比。

GN 统计约 `2675×22×44=2589400` 行/seed，分 batch 流式写入；最终按条件、层、组和 samp_id 分块汇总，保留自然/实际使用统计偏移的均值和最大值。层间描述量也按 seed、条件、samp_id 给出 mean/median/p90 与有定义数。

每个预固定窗口保存 22 条件的层间通道 mean/RMS 图数据、prediction 和 target（NPZ），不是完整通道特征。原始 FULL 层间 tensor 仅在当前 batch 暂存，随后释放。不再保存约 250 GiB 的全量 X。signals 阶段要求至少 4 GiB 可用磁盘，每 seed evaluate 要求至少 8 GiB；失败不自动裁剪矩阵。

## 7. 来源锁、门控与交付

- 独立模型/干预：`resp_train/paper_evidence/e4_r3_norm.py`。
- 信号分析：`resp_train/paper_evidence/e4_r3_signals.py`。
- 生命周期/执行：`resp_train/paper_evidence/e4_r3_norm_runtime.py`。
- CLI：`scripts/eval_e4_r3_temporal_normalization.py`。
- 测试：`tests/test_e4_r3_norm.py`。
- 实现锁：`docs/experiments/e4_r3_temporal_normalization_lock_20260921.json`。
- 输出根：`runs/e4_r3_temporal_normalization_v1/`。

新锁绑定当前代码、协议、测试、原 W0 allowlist、有序 rows、预选窗口、全部偏移及旧参考的字节身份。所有阶段排他创建新 attempt，完成后写 manifest/freeze receipt，失败保存 traceback 与已有产物；同身份成功阶段拒绝重跑。三个 seed 评价必须指向同一信号回执。

GPU smoke 只用 synthetic 输入和原 selected checkpoint：三 seed × batch 1/128，共 6 个用例，每例完整 22 条件。核对原生/重放一致性、每个 GN 同源重放、全部预测 finite 和 reserved/device total≤80%。环境与干净代码 commit 必须与正式 evaluate 一致。

最终输出包含：2640 行 seed/条件/指标/pooled/subject 配对、880 行三 seed 方向、330 行受试者宏平均、120 行 TRAIN_MEAN factorial、层间和 GN 分层汇总，以及共享信号回执。信号阶段另提供全部关联、配对差、各受试者汇总、等权宏平均、21 个预固定窗口图及曲线数据。

## 8. 用户运行顺序

实现锁生成后提交待执行代码、协议和锁，保持工作树干净。先由用户完成 synthetic GPU smoke，确认 GN 数值重放与显存门槛，再进入真实 validation 信号检查：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python \
  scripts/eval_e4_r3_temporal_normalization.py gpu-smoke --device cuda:0
```

GPU smoke 成功后：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python \
  scripts/eval_e4_r3_temporal_normalization.py signals
```

两阶段成功后，填入实际输出目录：

```bash
E4_R3_SIGNALS='/实际成功的signals目录'
E4_R3_GPU='/实际成功的gpu_smoke目录'
(
  for E4_R3_SEED in 20260811 20260812 20260813; do
    env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python \
      scripts/eval_e4_r3_temporal_normalization.py evaluate \
      --seed "$E4_R3_SEED" --device cuda:0 \
      --signals "$E4_R3_SIGNALS" --gpu-receipt "$E4_R3_GPU" || exit $?
  done
  env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python \
    scripts/eval_e4_r3_temporal_normalization.py summarize
)
```

CPU synthetic 验证命令：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ./.venv/bin/python -m pytest tests/test_e4_r3_norm.py -q
```

锁准备入口为 `scripts/eval_e4_r3_temporal_normalization.py prepare-lock`，只准备一次。已有锁直接使用，后续修订需新 identity。
