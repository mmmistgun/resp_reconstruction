# Patch-aligned TF-Mamba v1 模型实现

本文件定义本轮模型实现与定向工程验证。当前阶段为模型实现；训练矩阵、数据划分、loss、metrics、checkpoint selector 和 research-test 访问由后续独立实验协议定义。本轮不产生模型性能结论。

## 结构

输入 BCG 为 `(B,1,18000)`，采样率 100 Hz。连续小核 CNN 提取 100 Hz 特征，再用 200 点窗口、50 点步长形成 357 个观察单元。相对中心时间编码参与注意力池化，生成 `(B,357,96)` token。

连续 CNN：`Conv(1,32,k7) → GELU → 残差(k5) → Conv(32,64,k5) → GELU → 残差(k3)`。残差块使用逐位置通道 LayerNorm、depthwise 卷积、GELU 和 pointwise 卷积。连续 stem 的时间感受野为 17 个采样点。位置池化在 200 个特征位置上运算；其对应原始信号感受野包含 CNN 的邻域。

H-CWT 通过已有 `cwt_magnitude_features` 生成原生 Morlet `log1p(abs(CWT))` 的 50 点均值，从实际映射频率中选取 `(0.8,8] Hz`，保留 41 行的升序排列。返回的频率元数据与特征行一一对应。该选取与 H-only 分支所用的原生网格、频带与池化定义一致；实现复用本分支已有变换函数。

二维条件编码器为 `Conv2d(1,32,3×3) → 通道LayerNorm → GELU → DWConv2d(3×3) → GELU → Conv2d(1×1)`，保留 `(41,360)` 网格。逐位置通道归一化使其时间感受野保持在 5 个 CWT 时间格。

第 j 个 token 读取 CWT 编码特征的 `j:j+4`，每个 query 对应 164 个 K/V。K/V 在整张特征图上投影一次，再形成局部窗口。注意力使用 4 头，位置偏置来自实际频率的自然对数和相对中心时间 `[-0.75,-0.25,0.25,0.75]` 秒。读取位置对齐；CWT 和编码器的实际感受野包含邻域。

FiLM 为 `u=z*(1+0.5*tanh(gamma))+0.5*tanh(beta)`。条件 MLP 为 `96→96→192`，最终投影权重、偏置均零初始化。FiLM 后接 6 层现有独立正反向 Mamba2 block，D=96，d_state=64，d_conv=4，expand=2，headdim=32，ngroups=1，chunk_size=256，dropout=0.1。

解码器 `Linear(96,96) → GELU → Linear(96,200)` 输出局部波形。对称 Hann 窗取正下限 epsilon=0.001，按累计权重归一化重叠合成，完整覆盖 18000 点。epsilon 是记录在配置中的数值参数。双向时序建模适用于离线窗口重建。

## 接口与复现

- 模型：`resp_train.models.patch_aligned_tf_mamba.PatchAlignedTFMamba`。
- 通用模型注册名：`patch_aligned_tf_mamba`。
- 结构配置：`configs/patch_aligned_tf_mamba/model.yaml`。
- 调用：`model(x, tf={"w": h_cwt})["waveform"]`。H-CWT 的频率行顺序必须与构造模型时的元数据相同。
- `h_cwt_features(waveform)` 返回 `(41,360)` 特征及 41 个实际频率；注册 builder 按同一原生映射构造频率元数据。
- 实际频率、相对时间与合成窗保存为 state_dict buffer；模型结构配置及初始化 seed 随未来运行的 resolved config 一起保存。
- 输入、条件、FiLM 投影和输出非有限值显式失败。CPU 测试显式注入替身；正式构造调用官方 Mamba2。

2 秒窗口作为局部观察尺度，与四个 0.5 秒 CWT 区间建立确定几何关系。窗口时长和局部 token 压缩能力仍需后续 validation 实验检验。

## 定向验证

2026-10-04：11 项合成 CPU 单元测试通过；默认结构的原生 CWT 合成检查通过，输出 `[1,1,18000]`，条件与参数梯度检查通过。官方 Mamba2 CUDA 前后向尚未执行。

在本 worktree 根目录执行，复用现有 Python 环境：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python \
  -m pytest -q tests/test_patch_aligned_tf_mamba.py

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python \
  scripts/check_patch_aligned_tf_mamba.py --device cpu
```

单元测试覆盖时间索引、局部读取的显式参考计算、连续卷积感受野、合成首尾与梯度、FiLM 零初始化、条件梯度、输入失败、state_dict 恢复和模型注册。合成检查脚本使用真实原生 CWT 变换和默认结构；CPU 模式明确使用 Mamba 测试替身。

官方 Mamba2 CUDA 前后向由用户执行：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python \
  scripts/check_patch_aligned_tf_mamba.py --device cuda:0
```

验收为 JSON 报告 `status=passed`、`mamba=official_mamba2`、输出 shape `[1,1,18000]`、有限且有效的条件梯度以及有限参数梯度。命令只生成合成信号并向终端输出报告，不读取数据集或写入实验产物。GPU 显存、混合精度与正式训练效率仍需后续获准阶段验证。
