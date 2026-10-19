# M4 片段注意力池化相对位置消融 v1

协议 ID：`m4-pool-position-v1-20261015`，执行修订 r3。2026-10-15 用户授权在空闲 GPU 后台执行完整 A/B × 三 seed 的 train/validation。实际意义阈值留待后续研究确定，配置为 `local_rr_margin_bpm=null`、`decision.confirmed=false`，本轮报告描述性配对效应及指标取舍。正式 session 使用 `runs/m4_pool_position_v1/formal_r3_20261015/`，运行状态和结果以该 session 的产物为准。C/D 的预算和执行安排属于第二阶段，须另行确认。

## 来源与研究问题

研究对象为原始 P1 M4 / P1-Full-Add。[仓库归并记录](repository_consolidation_20261010.md) 将原 `codex/patch-aligned-tf-mamba` 对应到提交 `d45d16c553313b3f3a5673e497c183ede402e2d6`；本轮在归并后仓库中新增独立入口，共享冻结 P1 源码。历史 [P1 协议](p1_components_v1_protocol_20261006.md)、[P1 validation](p1_components_v1_validation_results_20261007.md) 和 [M4 组件结果](m4_components_v1_validation_results_20261009.md) 用于来源及背景核对。

研究问题是：片段内相对位置表示是否改善重新训练后的呼吸重建，其作用是否来自评分路径或值聚合路径。连续 CNN 提取特征后，以 100 点窗口、50 点步长聚合为 96 维 token；时频 cross-attention 从同一 BCG 的物理对齐 CWT 读取条件。两种表示可能包含重叠信息，但重叠程度不是本轮预设结论。

既有 M4 attention/均匀池化比较同时改变聚合权重，不能单独识别相对位置的贡献。本轮 A 同期重训练；历史 M4 不加入 A/B 的配对效果计算。

## 定义与矩阵

令 `p_t=MLP(Δt)`，`Δt=(t−49.5)/100` 秒，`t=0,…,99`。

`α_jt = softmax_t[s(h_jt + a p_t)]`

`z_j = W_pool Σ_t α_jt (h_jt + b p_t) + b_pool`

| 配置 | a | b | 当前执行范围 |
| --- | ---: | ---: | --- |
| A | 1 | 1 | 第一期：同期原 M4 |
| B | 0 | 0 | 第一期：内容池化 |
| C | 1 | 0 | 第二期候选：位置参与评分 |
| D | 0 | 1 | 第二期候选：位置参与值聚合 |

评分仍为原线性层。`s(h+p)=wᵀh+wᵀp+β`，因此评分位置项是共享片段内位置偏好，值路径是加权位置摘要；第一期的 A/B 不能独立分解这两种作用。第二期若开放，可按相同 seed 计算评分效应 C−B、A−D，值效应 D−B、A−C，以及交互 A−C−D+B；需先声明指标方向。

第一期固定 **A/B × seeds 20260811、20260812、20260813 = 6 cell**，均从头训练。配置为 `configs/m4_pool_position_v1/experiment.yaml`。CLI 不接受 C/D 正式运行。

## 不变量与配对初始化

沿用原始 M4：连续 CNN、97-scale W-full CWT、物理对齐 local cross-attention、六层 BiMamba2、原 MLP decoder、positive Hann 合成。保留 `condition.output` 与其后的独立 `fusion.projection`，融合权重及偏置零初始化。

构造沿用 `p1_components.*` 的模块 seed。B 按原顺序初始化后移除位置 MLP，最终模型、state_dict、optimizer 和前向均无此网络；这一步保持 score/project 的初始化随机数位置。B 的固定坐标 buffer 可保留，但不参与前向。A/B 参数量分别为 **1,056,357 / 1,052,069**，位置 MLP 为 **4,288** 个参数；均按实际 `requires_grad` 参数记录。

波形编码器重建使用 `torch.random.default_generator` 设置同名 CPU 子 seed，并通过 `fork_rng(devices=[])` 恢复 CPU 状态。构造结束后的设备随机状态与原 P1-M4 一致，从而保持训练 dropout 的起始随机序列。CPU 回归测试追踪 `torch.manual_seed` 的 CUDA 分发调用，逐 seed 核对 A/B 与原 M4 的完整调用序列及 CPU RNG 状态。

`audit-initialization` 对三个 seed 逐张量比较 A 与原 M4 的完整状态，以及 A/B 全部共有状态，核对唯一删除的四个 tensor 来自位置 MLP。`prepare` 将官方 Mamba2 CPU 构造所得报告写入 session。每个正式 cell 构造后再次与报告的初始化哈希比较，再转至 GPU。该哈希排除 `condition.frequencies_hz`（真实坐标由冻结 cache 元数据独立校验），其余参数及结构 buffer 均参与。

## 数据、训练与选点

继承 `configs/crd_tf_v1/crd_tf102_w_formal.yaml` 及原 P1 的 data/window/loss/evaluation/training 科学定义：train=10141 窗口/32 人，validation=2675 窗口/7 人，原 sample seeds、subject 隔离与 target 不变。正式数据入口复用原 W0 来源锁，核验 dataset index 哈希和 train/val row identity，记录实际行清单、cache 来源与数据 receipt。模型读取既有冻结 W cache。

输入为 `bcg_rawish_segment_soft_z_key`（数据集已有 segment soft-z BCG，100 Hz），180 秒窗口；案例直接保存模型收到的输入。物理 batch=32，累积=4，有效 batch=128，BF16；AdamW 原参数分组、weight decay=1e-4、clip=1.0；lr 3e-4→3e-5，5% update warmup+cosine。loss 为原 sync+0.25×effort；按 eligible 数归一化累积组并保留尾组。每 epoch 80 次更新，最大 80 epoch/6400 updates；early stopping min_epoch=30、patience=15、min_delta=0。

每 epoch 完整 validation，Local RR MAE 严格最小选 checkpoint，并列取最早。每个 cell 以固定 best checkpoint 完整评价一次；案例复用该次预测。A/B 使用相同软件版本及 GPU 型号，设备编号可以不同。epoch 边界恢复保留模型、optimizer、更新计数、早停状态及全部 RNG。完成 cell 只在哈希验证通过后复用，失败 attempt 保留。

多个分片通过阻塞式文件锁串行登记 `training_environment.json`，持锁范围仅包括短暂的 JSON 读取、比较与首次写入。各 cell 的运行锁继续防止同一 cell 被重复执行。环境采集、session 一致性检查与恢复环境检查都处于 attempt 的异常捕获范围，失败时保存 `failed.json`；已登记的环境保持原值。

独立测试集不在第一期授权范围。已有 research-test 曾影响相关模型设计，本轮 validation 又用于选点；三 seed 共用七名 validation 受试者，结果是开发阶段证据，不能将窗口或 seed 当成独立受试者重复。

## 描述性汇总与结构判断

主效应定义为 `g_s=LocalRR(B,s)−LocalRR(A,s)`，正值表示位置设计有益。r3 将数值阈值与本轮训练分离：完整训练按固定数据、矩阵、预算及 selector 执行，汇总保留效应大小、方向和波形取舍；`decision.csv` 的阈值及第二阶段门槛字段为空。阈值字段为空不影响模型训练或 checkpoint 选择。若后续依据本轮结果制定阈值，该阈值下对本轮结果的判断属于事后探索性解释，前瞻性确认需要新的预设实验；已准备 session 保持原身份。

1. 完整六 cell 后，报告每个 `g_s`、均值、sample SD 及正/负/零方向计数。
2. 同时检查受试者等权及逐受试者差异。若窗口加权与受试者等权方向反转，明确记录聚合依赖，不宣称统一人群收益。
3. 报告 Whole RR、包络轨迹、全局包络调制、PCC、IBI、包络分层及 lag 统计，完整呈现主指标与辅助指标间的取舍；结构保留仍由完整证据和用户确认决定，CLI 不自动启动 C/D。
4. A/B 的比较同时考虑参数差额与实际效果；跨 seed 方向混合或指标冲突时，明确记录证据边界。第二阶段是否开展由完整六 cell 结果讨论决定。
5. 当前三个 seed 和七人 validation 只做描述性报告；等效结论需要预设等效界值、统计单位及相应不确定性分析。

完整报告逐 seed 与 paired mean/sample SD（ddof=1），保留方向计数。现有 PCC 允许 ±0.30 秒时移；必须连同绝对 lag 中位数、P95、触边比例解释，PCC 改善不能单独证明时间对齐改善。波形图使用原时间轴。

## 产物与运行命令

`prepare` 不读取真实信号；保存 resolved 六 cell 配置、矩阵、代码 SHA、Git commit/status、源码快照、数据来源记录身份及三 seed 初始化报告。正式 cell 保存完整 history、最佳 epoch、checkpoint、优化器组、环境、数据来源、逐窗口指标及全部原生汇总。`summarize` 要求完整六 cell 和相同窗口/target 资格集合，验证所有 completion artifacts；输出逐 seed/跨 seed、B−A 配对、受试者配对、subject macro、Local RR tail、分母、参数量、选点与全指标表。

案例规则在运行前固定为：每名 validation 受试者选择最小 `dataset_row_id`，三个 seed 和 A/B 共用这些窗口。图第一行显示实际 BCG 输入，第二行显示 180 秒 target/A/B，第三行固定 60–120 秒局部区域；保留原幅度及时间轴。全部输出 PNG。案例用于形态检查，不代表总体性能。

所有命令在项目根目录运行，Python 使用项目 `.venv/bin/python`。r2 的 A/B 官方 GPU 合成验收已通过，来源为 `runs/m4_pool_position_v1/preparation_r2_20261015/engineering/`。r3 按当前源码身份准备正式 session；`run` 在各臂训练前执行本 session 的正式 batch=32、累积=4、两次更新及 validation loss 工程检查，并保存 receipt。当前授权使用空闲的物理 GPU 1，按 seed 顺序串行执行 A/B 六个 cell；全矩阵完成后执行汇总。

```bash
cd /mnt/disk_code/marques/resp_reconstruction
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/run_m4_pool_position.py plan
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/run_m4_pool_position.py prepare --session runs/m4_pool_position_v1/formal_r3_20261015
```

单 GPU 顺序执行六个 cell；`run` 会先执行/复用每臂正式批量、两次更新的 synthetic GPU 验收：

```bash
m4_train_log=$(mktemp runs/m4_pool_position_v1/formal_r3_20261015/train_XXXXXX.log)
nohup env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 .venv/bin/python -u scripts/run_m4_pool_position.py run --session runs/m4_pool_position_v1/formal_r3_20261015 --device cuda:0 > "$m4_train_log" 2>&1 < /dev/null &
```

恢复时在输出重定向之前添加 `--resume`，并创建新日志。单 cell 用 `--arm A --seed 20260811`。可选同型号双 GPU 分片，在上述 run 参数中分别添加 `--shard-count 2 --shard-index 0/1` 并分别设置 `CUDA_VISIBLE_DEVICES=0/1`；每个分片串行三 cell，使用同一个预先准备的 session 和各自独立的日志。

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD .venv/bin/python scripts/run_m4_pool_position.py status --session runs/m4_pool_position_v1/formal_r3_20261015
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/run_m4_pool_position.py summarize --session runs/m4_pool_position_v1/formal_r3_20261015
```

验收标准：六个 completed、样本及资格一致、全部关键张量/指标有限、每个初始化哈希匹配、所有 artifact 哈希通过、最佳 epoch 可由 history 复核、完整汇总及 7×3 张案例 PNG。正式训练前须有对应源码、设备环境和批量的官方 Mamba2 GPU 合成验收 receipt；该证据与 CPU fixture 前后向检查分别记录。

## 当前实现核验

2026-10-15，r3：**28 项 CPU 定向测试通过**。在 r2 检查基础上，覆盖阈值留空时的运行入口与汇总语义，以及已确认/待确认阈值的门槛字段处理。模型、数据、loss、metrics、训练预算及 checkpoint selector 均沿用上述定义。

2026-10-15，r2：**24 项 CPU 定向测试通过**。覆盖四种池化公式与梯度、A 对原 M4 的精确前向一致性、三个 seed 的共有初始化和设备 seed 分发、A/B BF16 前后向、正式 loss 两步累积、并发环境登记等待、三类环境失败记录、配置及产物完整性、固定案例选择和完整矩阵汇总/PNG 生成。

正式 Mamba2 已完成 CPU 构造及三 seed 配对检查。共享 P1 模型、runtime 与基础 patch 模型源码经 Git 比较与 `d45d16c` 相同；三个历史 M4 保存配置的 data/window/loss/training/evaluation 与本轮对应字段一致。补充 synthetic CPU 比较中，两次正式 loss 更新后的 A 与原 M4 参数及训练统计精确一致，新旧评价入口的 29 列结果精确一致。GPU 合成验收的证据由本节所述 r2 session 单独记录；逐 seed 正式指标、最佳 epoch、跨 seed 结果和结构选择结论待正式训练后产生。

初次准备产物 `runs/m4_pool_position_v1/preparation_v1_20261015/` 以及 r2 准备与 GPU 验收产物 `runs/m4_pool_position_v1/preparation_r2_20261015/` 保持原位和原始字节，各自源码快照用于恢复所属版本。r3 使用独立的正式 session，保存六 cell 配置、当前源码快照和配对初始化报告。源码身份校验分别绑定所属 session。

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_m4_pool_position.py
```
