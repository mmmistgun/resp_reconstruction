# M4-v2 单投影时频残差 train/validation 协议

协议 ID：`m4-residual-v2-20261007`。用户已确认固定 6 臂 × seeds 20260811/20260812/20260813，共 18 个 cell。当前范围为实现、合成 CPU 验证及来源准备；正式 GPU 验收和 train/validation 由用户执行。

## 研究问题与证据起点

本轮回答：在 1 秒 patch、连续波形 CNN、BiMamba2 和局部解码合同下，CWT residual 条件分支是否提供值得保留的增量，以及该增量来自均匀时频信息、content attention 还是物理坐标 bias。模型层面综合阅读固定五项指标、受试者等权结果、尾部误差及 seed 配对；checkpoint 层面继续按 Local RR 选点。

前序结果见 [P1 validation](p1_components_v1_validation_results_20261007.md) 与 [P1 research-test](p1_components_research_test_results_20261007.md)。M4 的 test Whole RR 三 seed 均值为 0.605136 bpm，NoTF 为 0.643841；同 seed 的 M4−NoTF 差为 −0.206325、+0.033990、+0.056221 bpm，均值 −0.038705、sample SD 0.145589，改善方向为 1/3。该均值优势是本轮研究动机，尚未建立跨 seed 稳定优势。M4 的其余四项主指标均值及 subject-macro Local RR 低于 NoTF 的表现。

前序 test 曾参与研究开发，本轮设计属于受既有 research-test 证据影响的开发迭代。旧研究结果保持原身份；后续 test 评价需要独立专项协议及当次执行授权。本轮入口只开放 train/validation。

## 模型与矩阵

公共结构：100 Hz、180 秒波形；continuous small-kernel CNN；P=100、hop=50 的 patch pooling，N=359、D=96；适用时 6 层 BiMamba2；每 token 解码 100 点，以 positive Hann（epsilon=0.001）归一化 overlap-add 合成。waveform encoder 的相对位置表示与 attention pooling 沿用现有合同。

所有条件臂使用完整 97-scale W，360 个时间格；每个 token 读取 97 scales × 2 个对齐时间格，其相对中心坐标为 ±0.25 秒。条件编码器保持 Conv/GroupNorm/GELU、depthwise Conv/GELU、pointwise Conv；C=32，时间 reflect、频率 replicate padding。使用既有冻结 W cache，实际频率逐值校验。

| Arm | 名称 | 条件读取与融合 | 序列模块 |
|---|---|---|---|
| A0 | M4-v2 | 物理对齐 cross-attention → 单一零初始化 D→D 输出投影 → 加到 z | BiMamba2 ×6 |
| A1 | NoTF | u=z | BiMamba2 ×6 |
| A2 | MeanTF-Residual | 对齐 F×2 编码特征均值 → 单一零初始化 C→D 输出投影 → 加到 z | BiMamba2 ×6 |
| A3 | NoMamba | 与 A0 相同 | Identity |
| S1 | Query RMSNorm | 与 A0 相同，仅 Q 输入使用 RMSNorm | BiMamba2 ×6 |
| S2 | ContentAttention | 与 A0 相同的物理窗口和 Q/K/V，仅使用 content logits | BiMamba2 ×6 |

A0：`q=Wq(z)`，`k=Wk(F)`，`v=Wv(F)`；logits=`qkᵀ/√d + bf(log f) + bt(Δt)`，对 97×2 个局部格 softmax。concat heads 后由唯一 `Linear(96,96,bias=True)` 生成 delta，`u=z+delta`；权重和偏置均为零初始化。两个连续 affine 可以合成为一个 affine，函数表达类别一致，但重新参数化改变优化路径，因此 A0 使用独立实验身份从头训练。

A2 直接从 C=32 的编码特征均值经 `Linear(32,96,bias=True)` 生成 delta，同样将权重与偏置零初始化。S1 使用既有 CustomRMSNorm（eps=1e-5、learnable scale 初始为 1），只计算 `q=Wq(RMSNorm(z))`，残差仍为原始 z。S2 移除两个坐标 bias 网络，保持 CWT 窗口几何、CNN、Q/K/V 和输出映射一致；该对照检验显式坐标 bias，不能据此推断物理对齐读取本身的作用。

各共有模块按名称隔离初始化：waveform encoder、CWT encoder、Q/K/V、同型输出映射、坐标网络、trunk、decoder。在同 seed 下，共有参数逐张量一致；可选模块不会扰动其他模块的初始化。A0/A2/S1/S2 在初始 eval 状态与 A1 输出一致。首个更新可打开输出投影，随后梯度进入条件编码器和 attention 上游。全矩阵重新训练，共享本轮初始化合同。

A1 不构造条件模块，不读取 CWT cache 或频率元数据。A3 检验 residual-add 模型中显式 BiMamba2 的增量；其输入仍有小波支持、整窗 GroupNorm 统计及连续 CNN 邻域。前序 M6 使用 FiLM，不能替代本轮 A3 的同结构消融。

正式 Mamba2 在 CPU 上构造得到的参数量如下；同 seed 共有参数逐张量一致性已核验。A0 相对旧 M4 的 1,056,357 参数减少 9,312，等于一个带偏置的 96→96 affine。

| Arm | 参数量 |
|---|---:|
| A0 | 1,047,045 |
| A1 | 1,020,125 |
| A2 | 1,025,053 |
| A3 | 95,661 |
| S1 | 1,047,141 |
| S2 | 1,046,845 |

## 数据、优化与选点

沿用 `configs/crd_tf_v1/crd_tf102_w_formal.yaml` 的数据、sample seeds、target、loss、metrics 及已有 W0 来源身份。train=10141 窗口/32 人，validation=2675 窗口/7 人。每次运行检查 dataset index SHA、样本 ID 内容、分母和受试者隔离。

统一 microbatch=32、gradient accumulation=4、有效 batch=128、BF16；每 epoch 80 次 optimizer 更新。各累积组按真实 eligible 数归一化各 loss 分量，完整保留尾组。AdamW 的参数分组、lr=3e-4→3e-5、5% update warmup+cosine、weight decay=1e-4、clip=1 保持既有合同；loss 为 sync+0.25×effort。

最多 80 epoch/6400 updates；early stopping min_epoch=30、patience=15、min_delta=0。每 epoch 完整 validation；按 Local RR MAE 严格最小选 checkpoint，并列保留最早 epoch。最终对固定 best checkpoint 保存逐样本指标。模型、数据、预测及关键指标非有限时显式失败。

## 预设分析与决策

完整 18-cell 后汇总逐 seed、三 seed mean/sample SD、五项主指标、逐受试者及 subject-macro、Local RR tail、资格分母与参数量。主配对为 A1/A2/A3/S1/S2 各自相对 A0。补充预设递增链为：

| 功能增量 | 候选 − 参照 | 解释 |
|---|---|---|
| TF information | A2 − A1 | 均匀 CWT residual 的增量 |
| Content attention | S2 − A2 | 同支持域下 content-dependent 读取及其 Q/K/V 参数化的增量 |
| Physical coordinates | A0 − S2 | 显式频率/相对时间 bias 的增量 |

所有配对均按同 seed；`delta=候选−参照`，误差下降和 PCC 上升记为正 improvement。S1 检验 Query RMSNorm 的多指标表现与 seed 波动；训练日志和非有限失败记录提供优化稳定性依据。完整矩阵后综合评估，不预设单一指标的自动架构胜者，不以 partial seed 调整矩阵或训练合同。三个 seed 共用同一 validation 人群，且该集参与 checkpoint 选择，汇总为开发阶段描述性证据。

## 生命周期与执行

模型、训练生命周期与入口位于 `scripts/m4_residual_model.py`、`scripts/m4_residual_runtime.py`、`scripts/run_m4_residual.py`。复用现有数据、task loss、metrics、optimizer 和完整 epoch 恢复实现；本轮 source identity 包含依赖源码、配置、协议与定向测试。准备阶段保存一次 session 及源码快照，阶段进展由工程 receipt、epoch 产物、completion 和 summary 记录。

`prepare` 不读取真实数据。`run` 在真实训练前执行对应臂的 GPU 合成验收：正式 Mamba2、32×4 累积、正式 task loss/optimizer，两次更新后检查上游梯度/参数变化及 validation loss；相同 session、设备环境和正式批量的已完成验收复用。验收记录参数量、耗时及 allocated/reserved 显存，时间含首次内核成本，只作工程资源记录。

每个 cell 使用互斥锁；完整 epoch 保存模型、optimizer、scheduler update index、早停与 Python/NumPy/CPU/CUDA RNG。`--resume` 恢复最后完整 epoch，失败 attempt 原地保留；已完成 cell 校验产物后复用。

准备一次：

```bash
cd /home/marques/.codex/worktrees/model-architecture-review/resp_reconstruction
env -u LD_LIBRARY_PATH -u LD_PRELOAD /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python scripts/run_m4_residual.py prepare --session runs/m4_residual_v2/formal_v1_20261007
```

两个终端进入上述目录，各卡串行执行 9 个 cell：

```bash
# GPU 0
bash scripts/run_m4_residual_shard.sh runs/m4_residual_v2/formal_v1_20261007 0 0
```

```bash
# GPU 1
bash scripts/run_m4_residual_shard.sh runs/m4_residual_v2/formal_v1_20261007 1 1
```

恢复在 shell 命令末尾加 `--resume`。两个入口均包含 `env -u LD_LIBRARY_PATH -u LD_PRELOAD`；单组执行为 `run --session ... --arm A0 --seed 20260811 --device cuda:0`。独立工程验收可使用 `smoke --session ... --arms A0 A2 --device cuda:0`。

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python scripts/run_m4_residual.py status --session runs/m4_residual_v2/formal_v1_20261007
env -u LD_LIBRARY_PATH -u LD_PRELOAD /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python scripts/run_m4_residual.py summarize --session runs/m4_residual_v2/formal_v1_20261007
```

验收标准：18 个 cell 完成；源码、配置及样本身份一致；各 validation 2675 窗口/7 人，完整汇总样本与 target 资格一致；五指标及配对按资格为有限值，预测退化另行记录并保持分母。正式实验只由用户启动。

## 合成验证

2026-10-07：36 项 CPU 定向测试通过，覆盖矩阵/优化合同、共有初始化、单 affine 合成等价、初始 NoTF 输出等价、零输出投影后的梯度开启、全部臂有限梯度与 BF16、Query RMSNorm 作用范围、坐标消融几何、MeanTF 对齐区域、临时 cache 到 task loss/optimizer 链路、两步四微批累积、验收复用及递增配对方向。官方 Mamba2 已完成 CPU 构造、参数量及共有参数一致性核验。

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=. /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python -m pytest -q tests/test_m4_residual.py
bash -n scripts/run_m4_residual_shard.sh
```

合成测试使用显式 Mamba 替身验证模型和 task loss 链路；GPU 官方内核、正式批量显存及真实数据训练由上述用户执行阶段验收。
