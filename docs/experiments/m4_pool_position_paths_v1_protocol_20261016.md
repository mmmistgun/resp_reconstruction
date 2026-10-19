# M4 片段池化位置评分与值路径实验

协议 ID：`m4-pool-position-paths-v1-20261016`。2026-10-16 用户授权开始第二阶段 C/D，沿用第一阶段三个 seed 和训练预算，在空闲 GPU 后台执行六个 train/validation cell。当前实现与 13 项 synthetic CPU 定向核验已完成；运行状态以 `runs/m4_pool_position_paths_v1/formal_v1_20261016/` 的 session、engineering 和 completion 为准。完整汇总包含四条件 × 三 seed 共十二个 cell。

## 研究问题与矩阵

本轮分解片段内相对位置在注意力评分与值聚合中的作用。第一阶段的 [A/B 结果](m4_pool_position_v1_validation_results_20261016.md) 显示 Local RR 平均值接近、seed 方向混合，Whole RR 与全局包络调制存在取舍；本轮关注两条位置路径的条件效应与交互。

沿用 [第一阶段协议](m4_pool_position_v1_protocol_20261015.md) 的原始 P1-M4：

`α_jt = softmax_t[s(h_jt + a p_t)]`

`z_j = W_pool Σ_t α_jt (h_jt + b p_t) + b_pool`

`p_t=MLP((t−49.5)/100)`，t=0,…,99；评分 s 为原线性层。

| 配置 | a：评分位置 | b：值位置 | 本轮角色 |
| --- | ---: | ---: | --- |
| A | 1 | 1 | 第一阶段冻结参照 |
| B | 0 | 0 | 第一阶段冻结参照 |
| C | 1 | 0 | 从头训练 |
| D | 0 | 1 | 从头训练 |

seeds 固定为 20260811、20260812、20260813。C/D 的定义与第一阶段预先列出的第二阶段候选一致；启动依据来自第一阶段 validation，因此本轮仍作为开发阶段机制比较。

## 结构、初始化与科学合同

复用冻结 `scripts/m4_pool_position_model.py` 的四种数学定义、原 P1 模块子 seed，以及仅设置 CPU generator 的波形编码器重建逻辑。C/D 与同 seed A 的完整初始 state_dict 逐张量一致，A/B/C/D 的共有参数一致；GPU seed 分发调用和构造结束后的 CPU RNG 状态均与原始 P1-M4 一致。

C/D 均保留完整位置 MLP（4,288 参数），有效可训练参数量均为 1,056,357。A 为 1,056,357，B 为 1,052,069。保留连续 CNN、P=100/hop=50、96 维 token、97-scale W-full CWT、物理对齐 cross-attention、独立 attention 输出与零初始化融合投影、六层 BiMamba2、MLP decoder 和 positive Hann 合成。

数据划分、input/target、sample seeds、loss、metrics、optimizer 与 checkpoint selector 均与已完成 A/B 的保存配置逐项核对。train=10141 窗口/32 人，validation=2675 窗口/7 人，BCG 输入为 `bcg_rawish_segment_soft_z_key`，100 Hz、180 秒窗口。

训练使用物理 batch=32、累积=4、有效 batch=128、BF16；原 AdamW 分组、weight decay=1e-4、clip=1.0，lr 3e-4→3e-5、5% update warmup+cosine。loss 为原 sync+0.25×effort，eligible 加权累积并保留尾组。每 epoch 80 次更新，最大 80 epoch/6400 updates；early stopping min_epoch=30、patience=15、min_delta=0。每 epoch 完整 validation，按 Local RR MAE 严格最小、并列最早选 checkpoint，再评价固定 best checkpoint。

本轮仅使用 train/validation。三 seed 共用七名 validation 受试者，validation 同时参与选点。阈值保持未设置状态，完整十二 cell 后按描述性效应与指标取舍解释；原始输出的全频形态不由呼吸频带标准化指标保证。

## A/B 复用与来源

冻结参照 session 为 `runs/m4_pool_position_v1/formal_r3_20261015/`，session.json SHA-256：`ae108d17e8fb24ec5b19a789fedfa8c549ca4992fc5920347f465024ec4a318f`。

`prepare` 对六个 A/B cell 的全部 completion artifacts 校验 SHA-256，核对保存配置的 data/window/loss/evaluation/training、结构几何、参数量、初始化摘要和环境记录。新的四条件 CPU 构造报告还会与 A/B 保存的实际训练初始化摘要比较。每个 C/D cell 再核验自身构造摘要，然后进入 GPU。

A/B 的 checkpoint、指标、history 和案例按保存路径引用。正式队列及工程验收入口只接受 C/D；四条件汇总通过 session 内的 reference 路径定位 A/B。C/D 与 A/B 的 Torch、CUDA、Mamba/causal-conv1d 版本及 GPU 型号必须一致，设备编号可以不同。

本轮新增文件位于独立的 model/runtime/CLI、配置目录和协议中。第一阶段源码、协议、准备 session、训练及汇总产物保持原身份。

## 配对效应与交互

主指标 Local RR；同时完整报告 Whole RR、包络轨迹、全局包络调制、PCC 及既有辅助指标。以下 delta 保留指标原方向，误差负值/PCC 正值代表改善；汇总另外保存统一的 improvement 方向。

| 效应 | 公式 |
| --- | --- |
| 值路径关闭时的评分位置效应 | C−B |
| 值路径开启时的评分位置效应 | A−D |
| 评分路径关闭时的值位置效应 | D−B |
| 评分路径开启时的值位置效应 | A−C |
| 评分位置平均主效应 | [(C−B)+(A−D)]/2 |
| 值位置平均主效应 | [(D−B)+(A−C)]/2 |
| 两路径交互 | A−C−D+B |

交互衡量一条路径的效果是否随另一条路径改变；它的数值不等同于任一模型的绝对性能。固定 checkpoint 对应各配置独立重训练后的表现，路径比较解释为整个训练系统适应后的结构效应。

保存逐 seed 原始差、跨 seed mean/sample SD（ddof=1）和方向计数，提供窗口加权、受试者等权及逐受试者结果。基础对照另外保留 B−A、C−A、D−A。所有 cell 必须具有相同样本顺序和 target 资格集合，并报告实际分母、退化计数、最佳 epoch、完成 epoch、参数量和 Local RR tail。

既有 PCC 允许 ±0.30 秒时移，须结合 lag 中位数、P95 和触边比例阅读。完整指标取舍与配对一致性用于结构讨论；本轮不预设等效结论。

## 案例与生命周期

固定每名 validation 受试者最小 `dataset_row_id`，全部四条件和三个 seed 使用相同窗口。C/D 案例复用固定 best checkpoint 的完整 validation 预测，A/B 使用已保存案例。输出 21 张四条件对比 PNG：第一行展示模型实际输入 BCG，第二行展示 prediction/target 经 0.05–0.70 Hz 整窗 FFT 投影与中心化 RMS 标准化后的完整 180 秒波形，第三行固定放大 60–120 秒，保持原时间轴。

运行前保存六份 C/D resolved config、完整四条件合同、初始化报告、A/B 来源记录、Git 身份及源码快照。每臂正式 batch 的 GPU synthetic 验收执行两个 optimizer 更新与 validation loss。正式训练保存完整 history、每 epoch checkpoint/RNG/optimizer、数据行身份及逐窗口指标。环境登记采用短临界区阻塞文件锁，失败保留独立 attempt。全矩阵完成后汇总会校验十二个 cell 的所有 completion artifacts；相同来源的已完成汇总校验后复用。

## 运行命令

```bash
cd /mnt/disk_code/marques/resp_reconstruction
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/run_m4_pool_position_paths.py plan --include-references
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/run_m4_pool_position_paths.py prepare --session runs/m4_pool_position_paths_v1/formal_v1_20261016
```

当前使用空闲物理 GPU 1，六个 C/D cell 串行执行：

```bash
m4_paths_log=$(mktemp runs/m4_pool_position_paths_v1/formal_v1_20261016/train_XXXXXX.log)
nohup bash -c '
set -e
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python -u scripts/run_m4_pool_position_paths.py run --session runs/m4_pool_position_paths_v1/formal_v1_20261016 --device cuda:0
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python -u scripts/run_m4_pool_position_paths.py summarize --session runs/m4_pool_position_paths_v1/formal_v1_20261016
' > "$m4_paths_log" 2>&1 < /dev/null &
```

恢复时在 `run` 子命令的参数末尾添加 `--resume` 并创建新日志；单 cell 使用 `--arm C --seed 20260811`。后台 wrapper 在六个 cell 全部成功后执行完整汇总，任一步失败即退出。状态查询和独立汇总入口：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD .venv/bin/python scripts/run_m4_pool_position_paths.py status --session runs/m4_pool_position_paths_v1/formal_v1_20261016
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/run_m4_pool_position_paths.py summarize --session runs/m4_pool_position_paths_v1/formal_v1_20261016
```

## 当前核验

2026-10-16：13 项 CPU 定向测试通过，覆盖六项训练/十二项比较矩阵、冻结参照执行边界、配置一致性、三 seed 完整初始化及设备 seed 调用、C/D FP32/BF16 全梯度、正式 loss 两次更新、七个路径效应公式及符号、A/B 来源与环境验证、参照字节保留和完整四条件汇总/绘图。

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_m4_pool_position_paths.py
```
