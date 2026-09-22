# ADV-v1：时间对齐双视图网络实现与实验建议

日期：2026-09-20。协议 ID：`aligned-dual-view-v1-engineering-20260920`。

执行契约 ID：`aligned-dual-view-v1-train-val-20260920`。

**当前状态：网络核心、固定 CWT 前处理、独立 cache、训练、validation 和完整矩阵汇总入口已实现。用户本次授权继续实现；已执行 CPU 合成验证，真实数据 cache/smoke、GPU、正式训练和 test 均未执行。第 7 节定义可运行的工程契约与用户执行命令；第 4、5 节仍区分核心对照和可选扩展，不自动启动任何矩阵。**

本任务从提交 `cebc7f5a478da492458fe2c0f3f8d8acdf2f2001` 创建分支 `codex/aligned-dual-view-v1`。工作树为 `/mnt/disk_code/marques/resp_reconstruction/.worktrees/aligned_dual_view_v1`。

## 1. 任务与来源

本实现对应 [2026-09-20 V1 设计](/mnt/disk_code/marques/paper/resp_rec/design_proposals/20260920_aligned_dual_view_v1/DESIGN.md) 和该目录的 `architecture_spec.json`、`FRONTEND_REVIEW.md`。来源哈希见同目录 `aligned_dual_view_v1_source_lock_20260920.json`。

该 source lock 记录首轮网络核心交付时的状态；后续管线实现与当前测试记录见 `aligned_dual_view_v1_runtime_lock_20260920.json`，后者取代前者对管线是否实现的状态描述。两份记录均保留。

本协议只约束 ADV-v1，不替代 CRD、RTM 或论文证据闭环的历史协议。已关闭阶段、冻结 checkpoint/cache/summary 保持冻结。沿用输入 BCG 100 Hz × 180 s、既有 Pi、`L_sync + 0.25 L_effort` 与五主指标；本次没有修改数据、split、target、loss、metric 或 selector 实现。

首轮建议按提案使用完整 180 s 的既有 CRD 任务口径。论文 center-30/60/90 附件的评价区间属于另一实验定义，其数值不能直接与本轮混表。若改成中心区间重建，须先明确匹配的监督区间、评价区间和 anchor。

## 2. 实现定义

- 波形：`Conv1d(1,32,51,padding=25,reflect,bias=True) → SiLU → D10`。
- CWT：读取 `resp_train/aligned_dual_view/w0_grid.json` 中冻结的 97 个实际 scales；ssqueezepy 0.6.6、Morlet(mu=13.4)、float32、L1 normalization、reflect、完整窗口。按旧实现的实际频率 stable argsort 排列，低频重复离散中心频率仍保持对应尺度身份。
- CWT 幅度：在 100 Hz 上计算 `log1p(abs(CWT(x))) → D10 → Conv1d(97,32,1,bias=True) → SiLU`。
- D10：SciPy `firwin` 生成 501-tap、cutoff=4 Hz、Kaiser beta=8.6 的对称 DC 归一核；reflect250 后 valid stride10，输出原点为原输入第 0 点。两路复用同一实现。核和滤波累加保持 float32；允许 autocast，拒绝将固定核整体转成 bf16/fp16。CUDA 卷积要求 `torch.backends.cudnn.conv.fp32_precision='ieee'`，防止 TF32 降精度；GPU 检查入口显式设置此项。
- 拼接为 `(B,1800,64)`，六个既有 `BidirectionalMamba2Block`；参数按原提案，dropout=0，正反向独立。
- `Conv1d(64,1,1,bias=True)` 后使用既有 `fourier_interpolate`。返回 `waveform` 与 `waveform_10hz`；前者是供既有 loss/evaluator 应用 Pi 的网络输出。读出不限制符号。
- 可学习 Conv/Linear 沿用 PyTorch 默认初始化；Mamba、RMSNorm、方向合并沿用既有块初始化。各模块采用独立命名子 seed，四个输入配置共享主干和读出的初始参数。
- CWT 是固定离线前处理，参数梯度经过可学习投影和主干；原始输入的 CWT 路径不提供 autograd。波形路前端和 D10 支持反向传播。

代码入口：`resp_train/aligned_dual_view/`。`configs/aligned_dual_view_v1/*.json` 描述单独网络；`experiment.yaml` 是封闭 schema 的执行配置。独立 CLI 为 `scripts/run_aligned_dual_view_v1.py`。训练通过 `RuntimeModel` 适配既有 CRD epoch/optimizer 和 validation engine。

```python
from resp_train.aligned_dual_view import AlignedDualViewV1, prepare_cwt_batch
import torch

# x_cpu: CPU float32 (B,1,18000)，须已按既有数据口径准备。
cwt_batch = prepare_cwt_batch(x_cpu)
torch.backends.cudnn.conv.fp32_precision = "ieee"  # CUDA 固定滤波的精度契约
model = AlignedDualViewV1().to(device)
prediction = model(x_cpu.to(device), cwt=cwt_batch.to(device))
# prediction 直接交给既有 RespirationTaskLoss；Pi 由该 loss/evaluator 处理。
```

`CWTBatch` 校验表示身份、shape、dtype、device 与有限性。正式管线另外核对 dataset/index、split、sample ID、输入内容、行顺序、特征内容、代码/依赖及表示哈希；训练前扫描全部已选窗口，并记录 target 和 mask 的内容哈希。`prepare_cwt_batch` 继续用于小批准备，完整 cache 使用独立 CLI。

## 3. 设计复核结论与主要风险

结构可实现，当前未发现必须先加深前端才能修复的计算错误。模型是否有效仍需要训练证据。

| 问题 | 判断与处理 |
|---|---|
| 同时更换前端、融合、宽度、表示时间网格和读出 | 与 W0 的差异只能解释为整个方案的差异。若要单独声称早期拼接优于 FiLM，须另做保持新前端、表示、宽度、读出相同的融合方式对照。 |
| 10 Hz 是否值得五倍缓存 | 输出带宽 0.7 Hz 不能直接决定输入幅度轨迹的充分采样率；幅度视图还包含高分析频率载波的调制。需要直接证据才能声称 10 Hz 必要。 |
| 两路尺度/幅值不平衡 | 每时刻 RMSNorm 不保证两路贡献平衡。先记录前端输出 RMS、梯度范数和塌缩情况；如失败再预定义仅从 train 拟合的尺度处理，不根据 val/test 任意选择归一化。 |
| 97→32 投影 | 明确有损，保留的是固定尺度列身份，不能声称保留全部尺度信息。尺度均值对照可以检验完整尺度读入的价值，但也含前端参数变化。 |
| 边界 | FIR reflect、CWT reflect 与 Fourier 周期延拓是不同算子的边界语义。必须分别报告内部区间与端点诊断，保持正式五指标统计区间不变。 |
| 原生 Mamba 运行 | CPU 接线测试使用显式测试替身；原生实例仅做结构统计。bf16、反向传播、长度 1800 的 fused kernel 运行仍需 GPU smoke。 |
| 模型效率 | 参数少不代表端到端快。必须计入 CWT、滤波、缓存读取、主干及同步开销；离线前处理成本与在线模型耗时分别报告。 |

旧 W0 CWT 是每 50 点平均后的 2 Hz 表示，时间块中心与新原点不同。插值旧缓存既不是本模型指定的抗混叠滤波，也不能被标记为新 10 Hz 表示。新特征每窗 `97×1800×4=698400` 字节；若沿用旧 train+val manifest 的 12,816 窗，仅特征 payload 约 **8.336 GiB**，不含索引、文件头和中间 CWT。这个估算没有打开真实波形或创建缓存。

抗混叠的通用依据见 [Zhang, ICML 2019](https://proceedings.mlr.press/v97/zhang19a.html)；Mamba-2 的算子来源见 [Dao & Gu, ICML 2024](https://arxiv.org/abs/2405.21060)。这些文献不提供本任务性能证据。Fourier 重采样的周期假设及非周期端点振铃见 [SciPy 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample.html)；本实现仍使用仓库明确丢弃源 Nyquist 的既有函数，其 Nyquist 约定不等同于直接调用 SciPy resample。

## 4. 建议的最小对照矩阵

**建议先完成三种输入 × 三个 seed，共 9 次新训练；若论文主张固定尺度身份的价值，再加入尺度均值 arm，总计 12 次。** 先锁定全部 arm、预算、失败规则和统计口径，再执行完整矩阵。

| 配置 | 前端进入主干的通道 | 主干 | 原生实际参数 | 主要问题 |
|---|---|---|---:|---|
| `joint.json` | 波形32 + CWT32 | D64、6层 | 468,113 | 主方案 |
| `waveform.json` | 波形64 | D64、6层 | 466,641 | CWT 输入是否增加价值 |
| `cwt.json` | CWT64 | D64、6层 | 469,585 | 带符号波形输入是否增加价值 |
| `joint_scale_mean.json` | 波形32 + 尺度均值→32 | D64、6层 | 465,041 | 完整尺度表示与均值压缩的比较 |

前三组前端自然参数差小于主模型总量的 0.4%，且主干、读出、序列长度与训练预算一致。单视图的活动前端直接输出 64 通道，不用无效参数补齐。它检验的是固定主干宽度下的输入表示 package；双路各32与单路64仍是解释边界。若希望补充自然规模单视图 D32，应另列效率实验，不混作唯一模块必要性证据。

W0、C201 优先作为已有完整 180 s validation 结果的上下文 anchor。只有 split、sample集合、target/loss/五指标、聚合和 selector 均能对齐时才并列比较；冻结结果不因本任务重新训练、重评或重选。

执行配置已采用以下公共口径；扩大科学矩阵时须保留这些可比条件：

- train/validation、admission、输入与 target key 复用既有 CRD；保存实际 sample 清单及其哈希。
- loss 与五主指标保持冻结定义；dataset 聚合为逐 sample direct mean，checkpoint 选择为 full-validation Local RR minimum。
- seeds 建议为 20260811、20260812、20260813；相同 seed 的共享主干初始化已配对。
- 训练使用 AdamW、LR 3e-4→3e-5、effective batch 128、80 epochs、warmup 5%、weight decay=1e-4、clip=1、bf16、无 early stopping。默认 physical batch=32、accumulation=4；允许改成乘积仍为 128 的组合，但同一比较矩阵必须保持同一组合。参数分组、按 optimizer update 的精确 LR 和按 eligible count 的梯度累积直接复用 `resp_train/crd/training.py`。
- 一次报告五主指标及 paired-seed 方向；如 RR 与形态/努力指标取舍，不凭其中一项宣布全面优越。三个训练 seed 不是三个独立人群。
- 不在看到单个 seed 结果后停掉剩余 arm/seed；工程失败保留完整 lifecycle。阈值和任何成功判据须在看结果前说明任务依据。
- 新模型仍属于已有研究方向下的 development evidence；更换网络不会使已使用的 research-test 恢复未触及状态。本协议不开放 test。

## 5. 多参数实验的优先级

**需要有限敏感性检查，不建议首轮同时搜索宽度、深度、卷积核、dropout、CWT 分辨率和学习率的笛卡尔积。**

1. **先做工程检查与核心输入对照。** 原生 GPU 合成 smoke、train-only 的前端幅值/梯度诊断和训练稳定性检查，应先于大矩阵。失败先定位实现或优化问题。
2. **主方案有稳定收益后，再做单因素容量检查。** 可预定义 D={64,96}、深度={4,6} 的三个点 `(64,6)`、`(96,6)`、`(64,4)`；已有主点可复用，新增 2×3=6 次。宽度改变同时改变视图投影宽度，因此是整体宽度敏感性，不单独归因于 97→32 瓶颈。若要识别宽度×深度交互，再加 `(96,4)` 三次，而非假定效果可相加。
3. **10 Hz 的成本是否必要，是独立问题。** 有成本压力时预先定义一项新 2 Hz 低通/采样及带时间戳重建到共同网格的对照。它改变幅度时间带宽与滤波 package；需要独立表示 identity，不能用旧缓存插值冒充。当前代码没有实现这项候选。
4. **T-local / TF-local 由失败模式触发。** 分别增加抽取前时域局部层或尺度压缩前局部二维层，先锁定核/通道/预算，一次只改一项。两者各自有效后，组合仍需交互验证。当前保持 V1 前端。
5. **时间/尺度干预后置。** 冻结主模型后才做预定义的错位和尺度重排功能诊断；时间偏移需规定正负方向、偏移量、边界对照和有效区间。不能将推理干预当作重训练消融，也不能据此回头选择 checkpoint。

若首轮全面劣于既有方案，先检查表示、边界与训练收敛，再决定是否投入容量实验。相同任务协议用于公平比较；若额外调学习率，必须给各 arm 相同调参预算并区分默认协议结果与调优后结果。

## 6. 已执行验证与后续验收

已执行的验证仅使用 synthetic/disposable 数据：

- 15 项 CPU 定向测试通过，包括 float32 FIR 通/阻带、独立卷积参考、完整边界、抽取原点、AMP 精度、滤波梯度、四种输入配置、配对初始化、非有限拒绝、真实 ssqueezepy 合成 CWT 的极性不变性与 AM 时相，以及既有 loss 的前后向接入。
- 原生 Mamba2 在 CPU 上实例化并统计了上表参数；没有执行原生 Mamba 前后向。
- float32 FIR 的 262144 点 FFT 核验：0–0.7 Hz 最大绝对偏差约 0.000193079 dB；5–50 Hz 最大响应约 −91.687581 dB。

在新工作树运行以下命令。复用已有虚拟环境，不安装或升级依赖：

```bash
cd /mnt/disk_code/marques/resp_reconstruction/.worktrees/aligned_dual_view_v1
ADV_PY=/mnt/disk_code/marques/resp_reconstruction/.venv/bin/python
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  SSQ_PARALLEL=0 SSQ_GPU=0 NUMBA_CACHE_DIR=/tmp/adv1-numba-cache \
  "$ADV_PY" -m pytest -q tests/test_aligned_dual_view_v1.py
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' \
  "$ADV_PY" scripts/check_aligned_dual_view_v1.py describe
```

以下 GPU 合成验证默认由用户执行，或获得当次明确授权后代跑；本次未执行：

```bash
ADV_CHECK_DIR=$(mktemp -d /tmp/adv1-gpu-check.XXXXXX)
for precision in fp32 bf16; do
  env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SSQ_PARALLEL=0 SSQ_GPU=0 \
    NUMBA_CACHE_DIR=/tmp/adv1-numba-cache \
    "$ADV_PY" scripts/check_aligned_dual_view_v1.py gpu-smoke \
      --device cuda:0 --precision "$precision" \
      --output "$ADV_CHECK_DIR/$precision.json" || break
done
```

预期报告 `status=passed`、`native_forward_checked=true`、`native_backward_checked=true`，两输出形状分别为 `[1,1,18000]` 与 `[1,1,1800]`，loss 与所有参数梯度有限。报告独占创建，失败写入失败记录，禁止覆盖。该 smoke 不建立性能、收敛或泛化结论。

后续管线的 CPU 测试见 `tests/test_aligned_dual_view_runtime.py`：合成 NPZ 的实际 CWT cache→两个训练 epochs→最早 Local RR 最小值选点→重载 validation；另覆盖 test 配置拒绝、subject 隔离、输入/特征/target 错配、缓存版本漂移、失败保留、输出防覆盖、checkpoint 非有限和完整矩阵汇总。时序网络使用显式测试替身，这些结果不替代原生 GPU 验收。

本次合并运行模型与管线的最小定向集合，**27 项测试通过**（23.88 s）。未访问真实波形或执行 GPU。

正式 MACs、显存和端到端时延在匹配硬件与明确统计口径下另测。

## 7. Cache、训练和评价执行契约

### 7.1 数据与产物边界

- `formal` 固定 research-v2 数据根、dataset index SHA=`f65ae6524632187c7c2795bd7b5e25434afb649b5a04fe39084b70cc59f9b04f`，admitted train=10,141、val=2,675；保留既有 admission 与 sample seeds。
- `smoke` 最多两个 epochs、每个 split 显式限制 1..64 窗。部分缓存不能供 formal 运行。smoke 和 formal 各用独立目录，选择与缓存契约必须完全相同。
- 缓存仅访问 train/val 的输入数组。训练和 validation 使用原有 dataset/loss/metric；索引元数据可用于隔离审计，不打开 test 波形。CLI 没有 test 入口，修改 split 或添加 test 配置字段会在数据访问前失败。
- 同一缓存可供三个 seed、四种输入配置共用。它绑定实际 sample 清单、输入哈希、实际 scales、滤波系数、前处理实现和数值依赖；读取时先验证全文件哈希，再在取样时验证输入与该行特征哈希。
- 时域单视图也校验相同的 sample/input identity，其 batch 不载入 CWT tensor。缓存校验属于运行前审计，不能被当成该模型在线推理成本。
- 所有输出目录要求事先不存在；拒绝向已有 run/cache 补写子运行，也拒绝把产物写入数据源或源码目录。run ID 由代码身份和随机 UUID 组成，不仅依赖时间戳。
- `started.json` 保存命令、代码版本、Git 状态、依赖和源文件哈希；`source_snapshot/` 保存实际执行源码、冻结配置和本协议，支持复现未提交的开发实现。执行期间文件变化会使本次 lifecycle 失败。
- 失败/中断保留已生成文件及 `failed.json`；只有成功产物有与 manifest 哈希绑定的 `completed.json`。强制杀进程可能只留下 started 状态，该产物同样不能被读取为完整运行。
- `history.jsonl` 在每个完整 epoch 后追加，训练完成再写 `history.csv`。改善 epoch 与最终 epoch 的 checkpoint 使用各自文件名并永久保留；selector 严格取 full-validation Local RR 最早最小值。当前不支持 resume。
- 每个成功训练保存 optimizer 参数分组、完整 history、selected/final checkpoint、逐 sample 五主指标及附加诊断、direct-mean summary、实际选择行、input/target/mask 哈希。符合原定义的 target-ineligible NA 保留；eligible 主指标出现 NaN/Inf 或越界直接失败。
- 独立 validation 仅加载 manifest 指定的 selected checkpoint，核对源码、依赖、配置、缓存、target/mask、checkpoint 哈希和有限性；不会重新选择 epoch。

### 7.2 用户执行命令

以下命令供用户执行；Codex 代跑真实数据/GPU 阶段仍需当次明确授权。应先完成第 6 节原生 GPU 合成验收，再运行有限真实数据 smoke，之后决定正式矩阵。预期物理 batch 是否适合硬件，由 smoke 检查，未在本任务中测量。

下面 smoke 使用 train=64、val=32，覆盖默认 physical batch=32；其有限窗口验收不保证完整数据集的耗时或显存上界。

```bash
cd /mnt/disk_code/marques/resp_reconstruction/.worktrees/aligned_dual_view_v1
ADV_PY=/mnt/disk_code/marques/resp_reconstruction/.venv/bin/python
adv_run() {
  env -u LD_LIBRARY_PATH -u LD_PRELOAD \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    SSQ_PARALLEL=0 SSQ_GPU=0 NUMBA_CACHE_DIR=/tmp/adv1-numba-cache \
    "$ADV_PY" scripts/run_aligned_dual_view_v1.py "$@"
}

# 有限真实数据 smoke：这些目录应尚不存在；失败目录也保留。
adv_run cache --output runs/aligned_dual_view_v1/cache_smoke_v1 \
  --set protocol.run_role=smoke --set training.epochs=1 \
  --set data.max_train_windows=64 --set data.max_val_windows=32
adv_run train --cache runs/aligned_dual_view_v1/cache_smoke_v1 \
  --output runs/aligned_dual_view_v1/joint_smoke_v1 \
  --set protocol.run_role=smoke --set training.epochs=1 \
  --set data.max_train_windows=64 --set data.max_val_windows=32

# 完整新表示 cache：只需构建一次，三个 seed / 各 arm 共用。
adv_run cache --output runs/aligned_dual_view_v1/cache_formal_v1

# 正式单次训练示例；默认 joint、seed20260811、80 epochs、32×4。
adv_run train --cache runs/aligned_dual_view_v1/cache_formal_v1 \
  --output runs/aligned_dual_view_v1/joint_seed20260811_v1

# 更换输入候选与 seed（其余配置保持相同）。
adv_run train --cache runs/aligned_dual_view_v1/cache_formal_v1 \
  --output runs/aligned_dual_view_v1/waveform_seed20260812_v1 \
  --set model.input_view=waveform --set training.seed=20260812

# 训练结束已生成 selected validation 指标；以下是需要独立重放时的入口。
adv_run validation --cache runs/aligned_dual_view_v1/cache_formal_v1 \
  --run runs/aligned_dual_view_v1/joint_seed20260811_v1 \
  --output runs/aligned_dual_view_v1/joint_seed20260811_validation_v1
```

每次失败后的新尝试必须选择新输出目录，不能删除失败记录后使用同名目录重跑。完整 cache 验收需 `completed.json`、对应 manifest、两个 split 的 selection/rows 文件与 `[N,97,1800]` float32 特征；smoke/正式训练验收需完整 epoch/update 数、可重载 selected/final checkpoint、完整 validation 样本数及有限的 eligible 主指标。

### 7.3 完整矩阵汇总

`summary` 默认要求 `{joint,waveform,cwt} × {20260811,20260812,20260813}` 的九个完整 formal run，读取已保存的 validation metrics，不运行模型。通过 `--include-scale-mean` 扩展为十二个完整 run。会拒绝 partial seed、smoke、重复输入以及数据/预算/执行代码/依赖不一致的组合。

```bash
ADV_RUN_ARGS=()
for view in joint waveform cwt; do
  for seed in 20260811 20260812 20260813; do
    ADV_RUN_ARGS+=(--run "runs/aligned_dual_view_v1/${view}_seed${seed}_v1")
  done
done
adv_run summary "${ADV_RUN_ARGS[@]}" --output runs/aligned_dual_view_v1/summary_core_v1
```

产物包括 `per_seed.csv`、`across_seed.csv`、`paired_deltas.csv` 和来源哈希 manifest。先保持每个 seed 的 sample direct mean，再对三个 seed 等权汇总；seed SD 仅描述优化随机性，不是人群置信区间，不构建加权总分或自动选赢家。
