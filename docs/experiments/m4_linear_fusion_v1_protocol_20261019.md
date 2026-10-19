# M4 Linear fusion 消融 v1

协议 ID：`m4-linear-fusion-v1-20261019`。

2026-10-19，用户授权基于 M4 简化基线开展三个 seed 的 train/validation 消融，seed 分配至不同空闲 GPU 后台执行，全部成功后自动与冻结基线比较。授权阶段包含 synthetic CPU 定向验证、正式配置 GPU 合成验收、prepare、训练及 validation 汇总。当前运行状态以 session、cell completion 和 summary receipt 为准；本协议在 prepare 后保持冻结身份。

## 研究问题与结构合同

基线采用交叉频率 bias 保留、Δt 时间 bias 移除的 M4，选择依据及身份见 [M4 简化基线决策](m4_simplified_baseline_decision_20261019.md)。本轮检验额外 linear fusion 投影带来的参数化与优化效果。

令 `u` 为交叉注意力加权聚合结果，`z` 为时域表示。L0 计算 `z + fusion.projection(attention.output(u))`；L1 计算 `context = attention.output(u)`，随后直接 `z + context`。

| Arm | 条件残差路径 | 初始化 | 本轮角色 | 参数量 |
|---|---|---|---|---:|
| L0 双投影基线 | attention.output → fusion.projection → residual add | fusion.projection 权重与偏置为零 | 复用冻结 A1 三 seed | 1,056,257 |
| L1 单投影 | attention.output → residual add | attention.output 权重与偏置为零 | 新训练三 seed | 1,046,945 |

L1 保留一个 D96→D96 affine，移除额外投影的 9,312 个参数。两个连续 affine 之间没有非线性，函数上可以合成为一个 affine；零初始化位置与参数化改变后，梯度传播和优化路径会变化。因此本轮结果衡量这一完整参数化变更的效果。

模型构造沿用基线的初始化调用顺序，再执行预定替换与清零；CPU/CUDA RNG 调用序列保持相同。除 `attention.output` 的预定零初始化外，其余共有参数与同 seed 基线逐张量匹配。两臂初始条件残差均为零。prepare 保存三个 seed 的官方 CPU 初始化摘要，正式运行核对初始化身份、参数量及共有状态。

共有结构为：连续波形 CNN、1 秒 patch/0.5 秒 hop 的时域 attention pooling 及其评分/值位置表示、W-full 97-scale CWT、每 patch 97×2 个物理对齐时频格、原条件编码器与 GroupNorm、Q/K/V、100 参数的交叉频率 bias、6 层 BiMamba2、局部波形解码及 Hann overlap-add。

历史 M4-v2 单投影实验使用不同初始化 namespace 并保留 Δt 时间 bias，科学合同与本轮不同；本轮配套参照固定为下述 L0。

## 冻结参照与来源

- 参照 session：`runs/m4_cross_time_bias_v1/formal_r2_20261018/`。
- L0 cell：上述目录的 `cells/A1_seed20260811`、`cells/A1_seed20260812`、`cells/A1_seed20260813`。
- session SHA-256：`96975f061d9e1e0e51cbd75d8f0154f383a07389535d6423c0b9853d68c5ee43`。
- validation summary receipt SHA-256：`e3f6206ef63157aaf9ebde106d6c9ffb10f5e5d2c350107e047866c44e1f0458`。
- 配对 seeds：20260811、20260812、20260813；冻结基线最佳 epoch 分别为 25、5、8。

prepare 核验参照 session、validation receipt、源码快照、配置、完成产物哈希、环境摘要与共有科学合同，并保存本轮来源记录、官方 CPU 初始化摘要及源码快照。L0 仅在本轮汇总内映射参照标签；历史文件保持冻结。正式环境的软件版本、关键包及 GPU 型号与参照核对，GPU 编号可不同。来源或身份核验失败时显式停止。

## 数据、训练与选点

复用 `configs/crd_tf_v1/crd_tf102_w_formal.yaml` 的数据、划分、target、loss、指标及种子口径。train 为 10,141 窗口/32 人，validation 为 2,675 窗口/7 人；窗口 180 秒、100 Hz。使用既有冻结 train/validation W-full cache，核验 index 身份、行 ID 摘要和受试者隔离。

batch=32、梯度累积4、BF16；最多80 epoch，最早30 epoch停止，patience=15，min_delta=0。AdamW 学习率从 `3e-4` 调度至 `3e-5`，warmup 占5%；每 epoch 80 次 optimizer update，其他优化器和 scheduler 参数沿用基线。每个 epoch 在完整 validation 上评价，按 Local RR 严格最小、并列最早选择 best checkpoint。

GPU 合成验收在正式训练前自动执行，使用正式 batch=32、BF16、累积4的两个 optimizer update、官方 Mamba2 和正式 RespirationTaskLoss，检查条件分支、梯度/状态有限及短 validation。正式训练前按 seed 重置随机状态，保证合成验收与正式初始化的独立性。

非有限 input、target、prediction、checkpoint、loss 或关键指标显式失败。每个 cell 保存 resolved config、数据与 sample seed、训练 seed、原始命令、代码版本、软件/GPU 环境、数据访问与 row identity、初始化摘要、optimizer groups、完整 history、checkpoint、逐窗口指标和成功或失败 receipt。

## 配对比较与结论口径

三个 L1 cell 全部成功后，与三个冻结 L0 cell 组成完整六 cell 比较。主指标为 Local RR MAE、Whole RR AE、envelope trajectory MAE、global envelope modulation error、lag-aware signed PCC；保留原生辅助指标、分母、预测退化、逐人结果和 Local RR tail。

同 seed 的 `delta = L1 − L0`；误差下降、PCC 上升对应改善。输出各 seed 原值与差值、跨 seed mean/sample SD、原始方向计数和达阈值方向计数。配对前核验样本身份、顺序及 target 资格一致。

结构选择采用窗口加权汇总，受试者宏平均作为诊断。五项主指标分别按其原生单位判断：绝对均值差 **<0.01** 记为“均值无明显差异”，达到或超过 0.01 按方向记为改善或变差，同时检查逐 seed 差值。该描述性阈值不改变 checkpoint selector，也不构成统计等效检验。完整三 seed 结果形成前，矩阵、模型、预算、指标与判据保持冻结。

自动汇总生成逐 seed/跨 seed、同 seed 配对、受试者宏平均配对、0.01 判据、尾部、原生辅助指标、来源 CSV、PNG 配对图及中文 `report.md`。本协议证据范围为 train/validation；research-test 须由匹配的专项协议和当次授权开放。

## 生命周期与执行

正式输出 identity 为 `runs/m4_linear_fusion_v1/formal_v1_20261019`。已生成的 session、日志、失败记录和实验产物保持可追溯。训练恢复通过 CLI 显式 `--resume`，沿用冻结引擎的已完成 epoch 恢复语义与原配置。

```bash
cd /mnt/disk_code/marques/resp_reconstruction
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_m4_linear_fusion.py
bash -n scripts/run_m4_linear_fusion_parallel.sh
env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python scripts/run_m4_linear_fusion.py plan --include-references
env -u LD_LIBRARY_PATH -u LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python scripts/run_m4_linear_fusion.py prepare --session runs/m4_linear_fusion_v1/formal_v1_20261019
bash scripts/run_m4_linear_fusion_parallel.sh runs/m4_linear_fusion_v1/formal_v1_20261019 0 1
```

启动前核对 GPU 空闲状态；最后一条命令由后台任务管理器托管。shell 参数为 `SESSION GPU_FIRST GPU_SECOND`，GPU 编号限定为不同的0和1。第一个 GPU 执行 shard 0/2，即 seed 20260811、20260813；第二个 GPU 执行 shard 1/2，即 seed 20260812。每个 worker 的日志为 session 下 `gpu_INDEX.log`，通过原子排他创建保留日志身份。shell 等待两个 worker 均退出；任一失败则保留非零退出状态、日志和失败产物；均成功后自动执行 summarize。

`plan` 默认列出三个新 cell，`--include-references` 列出六个 cell。状态入口为 `scripts/run_m4_linear_fusion.py status --session ...`。完成验收要求三个新 completion、三个持续通过来源核验的冻结参照、固定 validation 分母、完整六 cell 配对汇总，以及配置、源码和产物哈希一致。
