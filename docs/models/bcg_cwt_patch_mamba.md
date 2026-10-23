# BCG–CWT patch 呼吸波形重建

`bcg_cwt_patch_mamba_v1` 用连续卷积编码同一窗口的 BCG 与完整 CWT 幅值图，在一秒片段内做交叉注意力，再通过六层双向 Mamba2 和共享输出头重建呼吸波形。模型用于整窗离线重建。

本轮单 seed train/validation 与两个固定 checkpoint research-test 已完成并关闭；指标、历史参照与产物位置见[结果记录](../experiments/bcg_cwt_patch_mamba_v1_results_20261015.md)。

## 输入、时间几何与主路径

```python
prediction = model(x, tf={"w": w})
# x: [B,1,18000]；w: [B,97,360]
# prediction["waveform"]: [B,1,18000]，float32
```

采样率为 100 Hz，每窗 180 秒。连续波形 CNN 输出 `[B,64,18000]`，由 `Conv1d(64,96,100,stride=100,padding=0)` 形成 180 个有序 token。该投影等价于按通道、片段内时间展平 `[64,100]` 后应用共享 `Linear(6400,96)`。

连续 TF CNN 输出 `[B,32,97,360]`，时间轴使用 reflect padding，频率轴使用 replicate padding。第 `j` 个 query 读取第 `2j,2j+1` 列，全频率共 194 个单元，扁平索引为 `f*2+r`。首片段对应采样 `0…99`、CWT 列 `0,1`，末片段对应采样 `17900…17999`、CWT 列 `358,359`。

频率表 `[97,32]` 与局部时间表 `[2,32]` 在全部片段、batch 间共享，各初始化为 `Normal(0,0.02)`。位置仅在 K/V 归一化之前相加一次。相对中心 `[-0.25,+0.25]` 秒保存为 buffer；首片段中心为 0.495 秒，对应 CWT 中心为 0.245、0.745 秒。

融合使用标准四头 `MultiheadAttention(embed_dim=96,kdim=32,vdim=32)`，继以残差和 `96→384→96` FFN。默认每次处理 16 个片段的完整候选集合，合并 context 后执行残差 dropout 和 FFN。连续 CNN 始终处理整窗。TF CNN 的 GroupNorm 具有整窗统计域，CWT 具有尺度相关的时间支持。

融合后的 `[B,180,96]` 经过六个独立 `BidirectionalMamba2Block`，共十二个方向性 Mamba2 实例。共享 `96→96→100` 输出头生成片段，按 `prediction[b,0,j*100+r]=patches[b,j,r]` 拼接。

调试时可用 `model(x, tf={"w": w}, return_attention=True)`，额外返回 `[B,180,4,97,2]` 的 `attention`。这些数值表示注意力计算权重。

## 配置与参数量

[默认结构配置](../../configs/bcg_cwt_patch_mamba_v1/model.yaml) 保留配套 YAML 的全部结构字段和值，构造种子为 `20261013`。`BCGCWTConfig` 独立校验时间几何、整除约束、padding、层类型及未知字段。可调整的宽度、层数、dropout 和分块参数均用于实际模块构造；固定实现语义的字段传入其他值会报出字段名。

官方 `mamba-ssm==2.3.2.post1` 构造的默认模型参数量：

| 模块 | 可训练参数 |
|---|---:|
| 连续波形 CNN | 39,136 |
| 有序 patch 投影 | 614,496 |
| 连续 TF CNN | 1,760 |
| 位置表与交叉注意力融合 | 102,784 |
| 六层双向 Mamba2 | 951,384 |
| 共享 patch 输出头 | 19,012 |
| 总计 | **1,728,572** |

构造种子覆盖一次完整模型初始化，并恢复调用者 CPU RNG。普通层采用 PyTorch 默认初始化；Mamba2 保留官方初始化及优化器标记。CPU 测试替身需要通过 `mamba_factory` 显式注入，其参数量不代表官方模型。

## 数据与任务接口

真实数据通过既有 `ResearchV2WindowDataset` 接入。BCG 字段为 `bcg_rawish_segment_soft_z_key`，目标字段为 `target_waveform_segment_soft_z_key`；实际数组键由数据索引解析。两者沿用各自的预处理与尺度语义。输出如需恢复尺度，应使用目标端元信息。

`FrozenWCWTReader` 读取既有 train/val W cache 的全部 97 行，验证冻结 manifest、频率文件身份、行身份、shape 和有限值。CWT 定义沿用 `cwt_magnitude_features`：整窗系数取 `log1p(abs(coefficients))`，按连续 50 点平均，按实际频率非降序排列。模型前向直接接收缓存特征。

模型工厂从 `data.tf_cache_path` 读取实际 `w_frequencies_hz.npy`。直接构造模型时必须显式提供与特征行对应的频率。结构 YAML 本身不包含实际频率向量。若同时提供 `model.frequencies_hz` 和缓存，两者必须完全一致。

`RespirationTaskLoss` 直接接受输出字典及 `[B,1,18000]` 的 target，使用原有 sync 与 effort 计算及 target eligibility；没有新增损失辅助字段。真实数据训练的 split、目标、损失、评价和 Local RR checkpoint selector 见[本模型训练协议](../experiments/bcg_cwt_patch_mamba_v1_protocol_20261014.md)。

## 构造与 checkpoint

在本 worktree 根目录运行：

```python
from resp_train.bcg_cwt_training import load_training_config
from resp_train.models import build_model

cfg = load_training_config("configs/bcg_cwt_patch_mamba_v1/train.yaml")
model = build_model(cfg).to("cuda:0")
```

该工厂示例会读取配置指定的缓存元数据。已有张量和数据频率时可以直接构造：

```python
from omegaconf import OmegaConf
from resp_train.models.bcg_cwt_patch_mamba import BCGCWTConfig, BCGCWTPatchMamba

cfg = OmegaConf.load("configs/bcg_cwt_patch_mamba_v1/model.yaml")
model = BCGCWTPatchMamba(
    frequencies_hz=data_frequencies_hz,  # 与 w 的 97 行一一对应
    config=BCGCWTConfig(cfg.model.architecture),
    initialization_seed=cfg.model.initialization_seed,
)
```

`state_dict` 含实际频率、相对中心以及 `_extra_state`：模型类型、解析后结构、构造种子、Mamba 实现身份和环境版本。加载 pre-hook 在复制权重或 buffer 前比较当前数据频率、结构与时间几何，并检查 checkpoint tensor 有限性。

```python
import torch

checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
model = build_model(cfg)  # 先按当前数据元信息构造
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
```

完整训练 checkpoint 含 NumPy RNG 状态，以上 `weights_only=False` 仅用于自己生成并信任的训练产物。单独保存的 `state_dict` 可以使用 `weights_only=True`。纯 `state_dict` 的加载方式为 `model.load_state_dict(torch.load(path, weights_only=True))`。

## 运行命令

当前环境 Python 路径为 `/mnt/disk_code/marques/resp_reconstruction/.venv/bin/python`。以下命令中的 `python` 代表该解释器或已激活的同等项目环境。

```bash
# CPU 外围与任务损失检查，显式使用 Mamba 接口替身。
python scripts/check_bcg_cwt_patch_mamba.py --synthetic-mamba --batch-size 2

# 官方内核，合成 BCG 与人工 CWT 特征，长度和网格均为默认值。
python scripts/check_bcg_cwt_patch_mamba.py --device cuda:0 --batch-size 1
python scripts/check_bcg_cwt_patch_mamba.py --device cuda:0 --precision bfloat16 --batch-size 2

# 仅解析真实训练配置。
python scripts/train_bcg_cwt_patch_mamba.py --print-config

# run02 历史训练命令；该 identity 已完成，新实验须另设输出目录与授权。
python scripts/train_bcg_cwt_patch_mamba.py \
  --set outputs.run_root=/mnt/disk_code/marques/resp_reconstruction/runs/bcg_cwt_patch_mamba_v1/seed_20261013_run02

# 从该运行最后一个完整 epoch 恢复；直接读取保存的解析配置。
python scripts/train_bcg_cwt_patch_mamba.py \
  --config /mnt/disk_code/marques/resp_reconstruction/runs/bcg_cwt_patch_mamba_v1/seed_20261013_run02/resolved_config.yaml \
  --resume
```

`--output 新文件.json` 可保存 smoke 结果，已有文件会被拒绝。smoke 输入为人工 BCG/CWT 张量，测试频率网格明确作为人工元信息。训练命令默认读取完整 train/val、最多 80 epoch、微批量 64、梯度累积 2 次、有效 batch 128、bfloat16 AMP。按 10,141 个训练窗口和 `drop_last=false`，每 epoch 更新 80 次，最大预算 6,400 次，5% warmup 为 320 次更新。训练种子与模型构造种子独立记录，多 seed 运行时可分别通过 `training.seed` 与 `model.initialization_seed` 设置。

训练目录保存 resolved config、源码快照及哈希、环境、访问记录、实际 train/val 行表、优化器分组、每 epoch 的任务统计、逐样本 validation metrics、checkpoint 和完成回执。新运行使用全新目录，恢复时显式指定 `--resume`（也可设置 `training.resume=true`）。失败会记录 `failed.json` 并抛出错误。

恢复以写完 `receipt.json` 的完整 epoch 为边界，先验证 checkpoint、record、逐样本指标的 SHA-256 和 epoch 连续性，再核对解析配置、Python 源码、环境版本、数据 index/cache、实际样本行顺序及更新预算。模型、优化器、Python/NumPy/PyTorch RNG、当前 CUDA 设备 RNG、更新计数、RR 最优记录、早停参考与等待计数、loss 最优记录共同恢复。配置比较仅忽略恢复开关和终端进度显示；默认设备、batch、累积、最大 epoch 等仍须与原运行一致。

中断的 epoch 目录原样保留，重试该轮使用 `epoch_NNNN_retry_<唯一标识>`；恢复尝试的日志、访问记录和失败状态写入 `attempts/<唯一标识>/`。首轮中断时按原种子重建后重试。已经早停或达到预算的 checkpoint 只补齐导出阶段；已有完成回执的运行经校验后直接返回结果。运行互斥只用于防止并发恢复。此入口支持当前版本生成的完整恢复状态，历史检查产物保持原位。

默认开启 early stopping，唯一监控指标为 validation Local RR MAE：`early_stopping_min_epoch=30`、`early_stopping_patience=15`、`early_stopping_min_delta=0.0`。从首轮累计未改善次数，指标严格低于早停参考值减去 min_delta 时更新参考值并清零；到达最小轮数且等待次数达到 patience 时，在保存完整本轮产物后停止。达到 epoch 上限也正常结束。每轮 record/checkpoint 保存 `early_stopping_best`、`wait`、`stopped`，完成回执保存 `early_stopped`。设 `early_stopping_enabled=false` 可运行完整 epoch 预算；学习率计划始终按配置的最大预算计算。

每轮 validation 只调用一次 `evaluate_task_predictions`。Local RR selector 从该结果的 `local_rr_mae_bpm` 按 `local_rr_target_eligible` 取有效窗口均值，沿用原 selector 的统计口径。目标不适用行保留原有标记，eligible 行出现 NaN/Inf 或没有 eligible 窗口时显式失败。

训练正常结束（含触发早停）时还会生成 `best_loss.pt`，内容与已完成 epoch 中 validation 总 loss 最低的 checkpoint 一致，loss 相同时选更早 epoch。`completed.json` 分别记录 Local RR 最优的 `best_checkpoint` 和 loss 最优的 `best_loss_checkpoint`、`best_loss_epoch`、`best_val_loss` 及来源文件校验和。每轮 record/checkpoint 同样记录当时的 loss 最优 epoch；中断时可据此定位已保存模型。loss 改善只更新辅助模型选择，不重置 RR 早停计数。Local RR 最优模型始终按严格最小值保存；自定义非零 min_delta 仅影响早停参考值。

## 实际检查记录

检查日期：2026-10-14。环境：Python 3.12、PyTorch 2.12.0+cu130、CUDA 13.0、mamba-ssm 2.3.2.post1、pytest 9.0.3、OmegaConf 2.3.0；GPU 为 NVIDIA GeForce RTX 4070 Ti SUPER。

模型与数据接入检查记录：**45 passed，0 failed，0 skipped**，耗时 53.45 秒。其中 3 项使用官方 CUDA 内核，40 项为本模型 CPU 外围/配置/数据/训练集成测试，2 项为原 H-CWT 数据路径回归。另有 12 条来自官方依赖的 autocast API 弃用告警。

训练恢复与累积检查（2026-10-14）：`BCG_CWT_RUN_CUDA=1 python -m pytest -q tests/test_bcg_cwt_training.py -rs`，**24 passed，0 failed，0 skipped**，耗时 16.39 秒。23 项 CPU 检查及 1 项 CUDA/bfloat16 小模型检查覆盖：2×64 累积只执行一次正确更新；CPU 和 CUDA 下中断恢复与连续运行的模型、优化器、RNG 及选点状态逐值一致；首轮重试、失败轮保留、早停后导出恢复、配置/源码/数据行/checkpoint 身份拒绝、并发互斥；Local RR 复用结果与原 selector 相同且不重复执行信号处理。RR 早停与最低 loss 独立保存的既有用例继续通过。累积与恢复测试使用小型可微模型和既有任务损失，官方 Mamba2 模型验收见下方独立记录。

当前默认微批量 64 × 累积 2 的 CPU 定向检查：`test_default_training_config_preserves_task_semantics` 与 `test_effective_batch_128_performs_one_correct_update[64-2]`，**2 passed**，耗时 12.23 秒。验证有效 batch 128、每 epoch 80 次更新，以及累积梯度和一次优化器更新的正确性。该检查使用合成小模型；完整官方模型的正式 GPU 训练状态和结果见运行目录回执。

```bash
BCG_CWT_RUN_CUDA=1 python -m pytest -q \
  tests/test_bcg_cwt_patch_mamba.py tests/test_bcg_cwt_training.py \
  tests/test_patch_aligned_tf_mamba.py::test_frozen_adapter_dataset_engine_training_step \
  tests/test_patch_aligned_tf_mamba.py::test_frozen_cache_tampering_and_missing_rows_fail -rs
```

验收覆盖：全部 180 片段的编号索引、Conv/Linear 前向与全部梯度等价、共享位置表、连续网格尺寸、Mamba 十二次构造参数、分块/整批 attention 的输出及输入/参数梯度、batch=1/2 默认长度前后向、任务资格计数、优化器更新、错误输入、配置冲突、checkpoint 提前拒绝不兼容身份、工厂和缓存接入、合成训练及失败产物保留。官方 CUDA 检查另含 checkpoint 保存/加载后的 eval 一致性和 no-weight-decay 标记。

Conv/Linear 等价使用 float64，容差 `rtol=atol=1e-10`。attention 输出与梯度比较使用 float64 的 `1e-8/1e-9`、float32 的 `2e-4/2e-5`；CPU 序列化比较为逐值一致，官方 CUDA 序列化容差为 `1e-5/1e-6`。

独立官方 smoke 实测：

| 精度 | batch | 状态 | 参数量 | 一步耗时 | 峰值 allocated | 峰值 reserved |
|---|---:|---|---:|---:|---:|---:|
| float32 | 1 | passed | 1,728,572 | 15.896 秒 | 556,030,976 B | 834,666,496 B |
| bfloat16 AMP | 2 | passed | 1,728,572 | 17.813 秒 | 646,441,472 B | 939,524,096 B |

计时区间包含前向、任务损失、反向、梯度检查和一次 AdamW 更新，显存包含测试用参数快照。这是单次合成检查的资源记录。两次检查输出均为 float32，所有参数梯度有限，各主分支梯度范数为正且完成更新。

原始记录位于本 worktree 的 `runs/bcg_cwt_patch_mamba_v1/engineering_20261014_01/`：`pytest_final.log`、`cuda_float32_b1.json`、`cuda_bfloat16_b2.json`、`parsed_training_config.yaml`。生成记录按仓库规则不进入 Git。

训练恢复与累积的记录位于 `runs/bcg_cwt_patch_mamba_v1/engineering_20261014_resume_01/pytest.log`。

当前环境可以执行 CUDA 检查。CPU 默认测试会将三个官方内核检查和一个 CUDA 恢复检查记为 skipped；开启 CUDA 检查后，缺少 CUDA、缺少官方依赖或设备不支持 bfloat16 时分别给出具体原因，内核运行错误会作为 failed 抛出。依赖内存在 autocast API 弃用告警，当前检查可正常完成。

以上工程检查使用合成输入。正式 train/val 状态与结果由 `runs/bcg_cwt_patch_mamba_v1/` 中各运行的回执和指标记录提供；重建精度、片段边界质量及跨受试者表现据实际训练与评价判断。独立测试集访问按专项协议开放。

## 文件职责

| 文件 | 职责 |
|---|---|
| `resp_train/models/bcg_cwt_patch_mamba.py` | 结构配置、模型、几何、序列化校验与构造函数 |
| `resp_train/models/registry.py` | 注册 `bcg_cwt_patch_mamba_v1` |
| `resp_train/data/bcg_cwt.py` | 97 行缓存读取与频率身份检查 |
| `resp_train/data/research_v2.py` | 数据集选择本模型 adapter |
| `resp_train/bcg_cwt_training.py` | 独立配置解析、train/val 循环及产物保存 |
| `configs/bcg_cwt_patch_mamba_v1/{model,train}.yaml` | 模型默认值与训练起始设置 |
| `scripts/check_bcg_cwt_patch_mamba.py` | 合成前后向和任务损失 smoke |
| `scripts/train_bcg_cwt_patch_mamba.py` | 配置检查及真实训练 CLI |
| `tests/test_bcg_cwt_patch_mamba.py` | 模型、梯度、序列化和官方内核测试 |
| `tests/test_bcg_cwt_training.py` | 临时 cache/NPZ 的数据与训练集成测试 |
