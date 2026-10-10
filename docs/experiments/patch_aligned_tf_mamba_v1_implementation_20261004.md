# Patch-aligned TF-Mamba 模型实现

更新日期：2026-10-05。当前阶段为模型与训练数据接口工程验证。训练矩阵、loss、metrics、checkpoint selector 和正式 validation 协议由后续实验定义；当前工程检查不产生重建性能结论。分支已合并 main 的 `b40c2d1`，包含 CWT-APOR 时频表示与机制记录。

## 物理几何

输入为 100 Hz、180 秒 BCG 和对应冻结 H-CWT。配置包含 sample_rate=100、window_samples=18000、patch_samples=200、patch_hop_samples=50、cwt_pool_samples=50、patch_chunk_size=16、checkpoint_local=true。

设输入长度 L、片段长度 P、步长 S、CWT 池化长度 Q。约束为 `P % Q == 0`、`S == Q`、`(L-P) % S == 0`，且 `0<P<=L`。冻结表示要求采样率、输入长度和池化分别为 100、18000、50；修改这三项需要独立表示合同。

`N=(L-P)/S+1`、`R=P/Q`。1/2/4 秒分别得到 359/357/353 个 patch，各读取 2/4/8 个 CWT 时间格。相对时间坐标为 `(r*Q+(Q-1)/2-(P-1)/2)/sample_rate`；2 秒时为 `[-0.75,-0.25,0.25,0.75]` 秒。

## 连续波形表示

完整 BCG 经 `Conv(1,32,k7) → GELU → 残差(k5) → Conv(32,64,k5) → GELU → 残差(k3)`，再接五个 k3 残差块，dilation 为 `1/2/4/8/16`。残差块使用逐时间位置通道 LayerNorm、depthwise 卷积、GELU 和 pointwise 卷积；时间卷积使用 reflect padding。

感受野为 `17+2*sum(stem_dilations)=79` 点，即 0.79 秒。`stem_dilations` 随配置保存。该增强使局部特征在池化前具备亚秒级组合信息，其任务收益仍需 validation 验证。

位置池化使用相对中心物理时间。线性 score 分为内容项与位置项，相加后沿片段时间 softmax；局部特征与位置向量分别加权求和，再投影为 D 维 token。这与对 `feature+position` 评分并聚合在数学上等价；中间张量按 patch 分块计算。

## 时频条件与合成

H-CWT 为 `(B,41,360)`，频率范围 `(0.8,8] Hz`。编码器为：时间 reflect/频率 replicate padding、3×3 Conv、GroupNorm、GELU、同样的分轴 padding、3×3 DWConv、GELU、1×1 Conv。默认通道数 32，GroupNorm 为 8 组。

GroupNorm 在每个样本内按组覆盖通道和时频空间统计。卷积支持为 5×5 邻域，归一化依赖整个时频网格。局部幅度扰动检查验证编码器可感知局部变化；这不等于对整窗统一增益保持绝对幅度。

每个波形 token 作为 Query，读取同区间 `41×R` 个 K/V。K/V 在原生 `(B,41,360,D)` 上投影一次，按时间偏移及 patch 分块计算。位置偏置来自缓存的实际 log-frequency 和相对中心秒数；默认 4 头。点积、softmax、累计使用 float32，局部激活在反向时重算。分块限制后端布局转换的临时存储。

FiLM 为 `u=z*(1+0.5*tanh(gamma))+0.5*tanh(beta)`，条件 MLP 为 `D→D→2D`，最后一层权重、偏置零初始化。第一步由输出投影开始学习；投影更新后，梯度进入条件编码器和注意力。

随后为 6 层独立正反向 Mamba2，D=96、d_state=64、d_conv=4、expand=2、headdim=32、ngroups=1、chunk_size=256、dropout=0.1。宽度和深度沿用 APOR 容量，作为结构验证起点。

解码器为 `Linear(D,D) → GELU → Linear(D,P)`，使用对称 Hann 窗及正下限 epsilon=0.001，经累计权重归一化合成完整窗口。epsilon 是记录在配置中的数值参数。双向结构适用于离线重建。

## 冻结缓存与训练接口

模型入口为 `resp_train.models.patch_aligned_tf_mamba.PatchAlignedTFMamba`，注册名为 `patch_aligned_tf_mamba`。将 `configs/patch_aligned_tf_mamba/model.yaml` 与调用方的数据、训练配置合并，并显式赋值 `data.tf_cache_path`。此 YAML 是模型与缓存接口片段，尚不是正式实验配置。

`model.tf_representations=[w]`。`ResearchV2WindowDataset` 按模型名选择 `FrozenHCWTReader`，复用原 `TfV1CacheReader` 的冻结 manifest、row identity 和只读映射验证。adapter 从冻结 `(N,97,360)` W cache 按实际频率选取 41 行，以 `tf['w']` 交给模型。

模型 builder 与 reader 共用 `w_frequencies_hz.npy`，验证 manifest 和频率文件 SHA-256、表示规格、频率顺序及维度；读取样本时验证整条源记录 finite。模型构造和训练读取均不计算 CWT。train/val 为当前开放 split；该接口拒绝 test，后续评价须由对应协议与授权开放。

标准 DataLoader、`batch_tf_to_device`、engine 的特征转发和 `train_one_epoch` 可沿用。合成临时缓存验证了 dataset→loader→engine→model 两个优化步骤。`h_cwt_features()` 保留用于独立合成检查。

实际频率、相对时间、合成窗保存在 state_dict；结构配置与初始化 seed 随 resolved config 保存。输入、条件、FiLM 投影和输出非有限值显式失败。此前模型 checkpoint 的参数布局与本轮新增 stem、归一化结构不兼容，不能作为本结构的严格恢复 checkpoint。

## CPU 验证

2026-10-05：模型定向测试 27 项通过，数据加载及特征转发相关回归 12 项通过；默认结构结合原生 CWT 的 CPU BF16 两步 AdamW 检查通过。CPU 检查均使用合成输入或临时 fixture，Mamba 为显式测试替身。后续官方 CUDA 检查已完成，结果见下文。

在本 worktree 根目录，使用现有环境：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python -m pytest -q tests/test_patch_aligned_tf_mamba.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python scripts/check_patch_aligned_tf_mamba.py --device cpu --dtype bfloat16
```

测试使用合成信号、临时 cache 与显式 Mamba 替身，覆盖几何、OLA 首尾与梯度、0.79 秒 stem 支持、局部幅度响应、边界延拓、分块参考前向/梯度、保存张量边界、BF16、元数据篡改、test 拒绝及两个 engine 优化步骤。

## CUDA 工程验证

经用户授权，官方 Mamba2 FP32 batch=1 与 BF16 batch=32/64 的两步优化检查通过；BF16 batch=128 显存不足。batch=64 峰值分配为 9.100 GiB、峰值预留为 9.938 GiB。详情与产物身份见 [CUDA 工程验证记录](patch_aligned_tf_mamba_cuda_validation_20261005.md)。

由用户在目标 GPU 执行。先检查小 batch 官方内核，再测目标 batch 的 BF16 训练 step 与显存。报告路径须尚不存在，失败也保存状态。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python scripts/check_patch_aligned_tf_mamba.py --device cuda:0 --dtype float32 --batch-size 1 --steps 2 --report /tmp/patch_tf_cuda_fp32_b1_20261005.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python scripts/check_patch_aligned_tf_mamba.py --device cuda:0 --dtype bfloat16 --batch-size 32 --steps 2 --patch-chunk-size 16 --report /tmp/patch_tf_cuda_bf16_b32_recheck.json
```

验收为 `status=passed`、`mamba=official_mamba2`、输出 `[B,1,18000]`、两步有限 loss/参数/梯度，以及第二步条件梯度非零。报告包含配置、batch、精度、软件版本、设备、参数量、耗时、`peak_allocated_bytes` 和 `peak_reserved_bytes`。峰值统计覆盖 forward/backward/AdamW step，包括优化器状态分配；耗时仅作工程检查，不用于正式效率比较。

本轮目标 GPU 上的直接批量 128 检查失败；批量 64 已通过。正式训练前还需匹配任务 loss、训练精度与 optimizer 配置的获准验证；有效批量和梯度累积由正式协议定义。
