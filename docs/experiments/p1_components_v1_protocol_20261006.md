# P1 组件消融 train/validation 协议

协议 ID：`p1-components-v1-20261006`。本轮为独立的 10 臂 × 3 seed 组件消融；用户已确认包括七个主臂和三个补充臂，并保留现有 GroupNorm。当前为实现与合成验证阶段，正式运行由用户执行。

## 主模型与矩阵

主候选 M0（P1-Full）定义为 **1 秒 patch + W-full CWT + local cross-attention + bounded FiLM + BiMamba2**。

W-full 指已有冻结 97-scale W 表示，频率上限约 8 Hz，频率坐标使用实际映射值。H 为 `(0.8,8] Hz` 的 41 scales，L 为 `≤0.8 Hz` 的 56 scales。三种表示采用各自支持域上的相同 GroupNorm 算子，比较定义为 representation variant：频率支持变化同时改变归一化统计域及频率边界邻域。

| Arm | 名称 | 定义 | 预设参照 |
|---|---|---|---|
| M0 | P1-Full | W-full attention → FiLM → BiMamba2 | 主候选 |
| M1 | P1-NoTF | waveform token → BiMamba2 | M0 |
| M2 | P1-H | H 表示，其余结构同 M0 | M0 |
| M3 | P1-Full-MeanTF | W-full 对齐区域均匀聚合 | M0 |
| M4 | P1-Full-Add | 零初始化线性 residual-add | M0 |
| M5 | P1-Full-PostFiLM | 由原始 z 读取 c，FiLM 放在 BiMamba2 后 | M0 |
| M6 | P1-Full-NoMamba | 条件化 token 经 Identity 后解码 | M0 |
| S1 | P1-L | L 表示，其余结构同 M0 | M0 |
| S2 | P1-H-MeanTF | H 表示的对齐区域均匀聚合 | M2 |
| S3 | P1-Full-UniformPatchPool | waveform patch 使用均匀池化 | M0 |

全部使用 seeds 20260811/20260812/20260813。1 秒为本轮固定 architecture decision：P=100 点、hop=50 点、N=359；每个 patch 读取 R=2 个 CWT 时间格，相对中心坐标为 ±0.25 秒。D=96，适用时 BiMamba2 深度为 6；continuous stem、MLP decoder、positive Hann 合成及相关默认数值参数统一。

## 变量定义与初始化

M1 不构造条件编码器、时频注意力或融合模块；数据与模型构造均不读取 CWT cache/频率元数据。M6 的 temporal trunk 为 Identity。参数量按实际可训练模块报告。

M3/S2 的条件为 `c=W(mean_{f,r} F[j,f,r])`。保留同一二维编码器、相同物理读取区间、FiLM、trunk 与 decoder；均值作用于编码后的全部尺度和两个对齐时间格。

M4 为 `u=z+Wc`，线性层 W 的权重及偏置零初始化。FiLM 各臂使用 `u=z*(1+0.5*tanh(gamma))+0.5*tanh(beta)`，生成 gamma/beta 的最终投影同样零初始化。

M0 为 `BiMamba(FiLM(z,c))`；M5 为 `FiLM(BiMamba(z),c)`。两者的 `c=Attention(z,F)` 都以原始 z 为 Query。

S3 使用 `z=W(mean_t(h_t+p_t))`，保留 continuous CNN、相对物理位置表示及 token 维度；池化权重固定均匀。

各共有模块按名称隔离初始化 seed。waveform encoder、条件编码器、同型 attention/fusion、trunk、decoder 的共有权重在同 seed 下相同，移除或替换其他模块不改变这些初始化。S3 保持其共有 stem、位置网络和输出投影与 M0 初始化一致。本轮所有 cell 使用新的实验身份训练，前序 H-only 结果保持冻结。

NoMamba 衡量显式 BiMamba2 模块的增量。其条件仍包含 CWT 小波时间支持、GroupNorm 整窗统计及连续 CNN 的邻域；1 秒描述的是 patch 读取几何，而非全部输入信息的严格支持范围。

## 数据、优化与选点

沿用 `configs/crd_tf_v1/crd_tf102_w_formal.yaml` 的数据、sample seeds、target、loss、metrics 和既有 W0 来源身份。train=10141 窗口/32 受试者，validation=2675 窗口/7 受试者。test 不属于本轮入口。

有条件的各臂读取相同冻结 97-scale W cache，数据 adapter 固定选择 W-full/H/L；训练时不计算 CWT。NoTF 只读取波形与监督目标。

统一物理 batch=32、累积=4、有效 batch=128、BF16。32×4 为较宽 W-full 支持下的统一显存设置。按真实 eligible 数对每个累积组的 loss 分量归一化，尾组完整保留，每 epoch 80 次 optimizer 更新。

AdamW 参数分组与基线一致，lr=3e-4→3e-5，5% update warmup+cosine，weight decay=1e-4、clip=1.0。loss 为既有 sync+0.25×effort。最多 80 epoch/6400 updates；early stopping min_epoch=30、patience=15、min_delta=0。

每 epoch 完整 validation，按 Local RR MAE 严格最小值选择 checkpoint，并列保留最早 epoch。各臂共享选择规则和最大预算，实际完成 epoch 可因早停不同。完成后评价固定 best checkpoint 并保存逐样本指标。

## 汇总与后续机制

完整 30-cell 后汇总逐 seed、跨 seed mean/SD、预设配对差、受试者等权结果、Local RR tail 和资格分母。W-full/H × attention/mean 的 interaction 为“W-full 下 attention 改善 − H 下 attention 改善”，误差下降/PCC上升均记正改善。GroupNorm 的 representation-variant 解释边界同时适用于该交互。

参数量记录于各 cell；同一设备与合成条件下的训练耗时、峰值 allocated/reserved memory 保存在工程验收 receipt。两步验收时间含初次内核开销，仅作为工程资源记录。FLOPs 本轮不提供估计值，避免将未覆盖 Mamba 自定义算子的计数用作完整模型 FLOPs。

正式 Mamba2 模块在 CPU 上构造得到的参数量如下；同 seed 共有参数逐张量一致性检查通过。

| Arm | 参数量 |
|---|---:|
| M0 | 1,074,981 |
| M1 | 1,020,125 |
| M2 | 1,074,981 |
| M3 | 1,052,989 |
| M4 | 1,056,357 |
| M5 | 1,074,981 |
| M6 | 123,597 |
| S1 | 1,074,981 |
| S2 | 1,052,989 |
| S3 | 1,074,916 |

固定 M0 的频带均值替换、时间位移及 attention 分布分析属于后续独立机制协议，待模型冻结后制定。Attention mass 描述模型的权重分配，功能作用需要相应干预证据。

## 生命周期与执行

2026-10-06：35 项定向 CPU 测试通过，覆盖矩阵和优化合同、共有初始化、零初始化融合、PostFiLM Query/位置、MeanTF 与均匀 patch 池化、全部臂的完整梯度、BF16、临时冻结 cache 数据链、正式 task loss 的四微批累积、验收复用及交互计算方向。模型前后向测试使用显式 Mamba 替身；正式 Mamba2 已完成 CPU 构造与共有参数一致性检查。GPU 合成验收及正式训练尚未执行。

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=. /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python -m pytest -q tests/test_p1_components.py
```

实现位于 `scripts/p1_components_model.py`、`scripts/p1_components_runtime.py` 和独立配置目录。前序已冻结的模型、训练协议与 test 来源字节保持原位。

`prepare` 保存矩阵、源码身份与快照，不读取真实数据。`run` 在真实训练前执行对应臂的官方 GPU 合成验收：32×4 累积、正式 task loss/optimizer、两次更新及 validation loss。相同来源、设备环境与正式批量的已完成验收复用。每个 cell 使用互斥锁，并按完整 epoch 保存模型、optimizer、update index、早停状态、Python/NumPy/CPU/CUDA RNG。`--resume` 从最后完整 epoch 继续；完成产物校验后复用。

先准备一次：

```bash
cd /home/marques/.codex/worktrees/model-architecture-review/resp_reconstruction
env -u LD_LIBRARY_PATH -u LD_PRELOAD /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python scripts/run_p1_components.py prepare --session runs/p1_components_v1/formal_v1_20261006
```

两个终端均进入上述目录，分别启动，每张 GPU 串行执行 15 个 cell：

```bash
# GPU 0
bash scripts/run_p1_components_shard.sh runs/p1_components_v1/formal_v1_20261006 0 0
```

```bash
# GPU 1
bash scripts/run_p1_components_shard.sh runs/p1_components_v1/formal_v1_20261006 1 1
```

脚本包含 `env -u LD_LIBRARY_PATH -u LD_PRELOAD`。恢复时末尾加 `--resume`。单组执行可使用 `run --session ... --arm M0 --seed 20260811 --device cuda:0`；GPU 合成验收可独立执行 `smoke --session ... --device cuda:0 --arms M0 M1`。

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python scripts/run_p1_components.py plan
env -u LD_LIBRARY_PATH -u LD_PRELOAD /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python scripts/run_p1_components.py status --session runs/p1_components_v1/formal_v1_20261006
env -u LD_LIBRARY_PATH -u LD_PRELOAD /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python scripts/run_p1_components.py summarize --session runs/p1_components_v1/formal_v1_20261006
```
