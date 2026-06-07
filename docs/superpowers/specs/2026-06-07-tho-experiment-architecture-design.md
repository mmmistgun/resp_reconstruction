# THO 实验工程骨架重建设计

## 背景

当前仓库已经完成胸带参考 `r_tho_hat(t)` 的首版小规模训练闭环，包含数据审计、平凡基线、训练、验证、checkpoint、评价指标、诊断预测和绘图脚本。首版设计偏向「最小跑通」，核心逻辑主要集中在 `scripts/train_tho_small.py` 和 `scripts/eval_tho_small.py`。

经过 2026-06-07 的 `mixed_zscore` 小规模实验后，当前最影响后续研究可信度的不是模型数量，而是实验组织与采样口径：

- 当前 `max_train_windows` 和 `max_val_windows` 默认取排序后的前缀窗口，容易引入样本顺序、片段和质量分布偏差。
- 训练脚本承担数据准备、审计、baseline、模型构建、训练、评价和诊断保存等多重职责。
- 训练和评价脚本各自重建数据管线，后续容易产生口径分叉。
- 文档已有首版训练说明、实验记录和计划文档，但缺少一个面向下一阶段实验工程的结构化设计。

真实索引快速检查显示，`mixed_zscore` 训练集按 `dataset_row_id` 前缀取 512 个窗口时，样本主要集中在 `samp_id=88` 和 `samp_id=221`；验证集按前缀取 256 个窗口时，样本集中在 `samp_id=952`。由于数据来自滑动窗口采样，相邻窗口高度相关，前缀取样会放大单个受试者和单个片段的影响，不适合作为正式训练或验证默认策略。

本轮目标是在尊重既有数据事实的前提下，重建实验工程骨架，使后续可以稳定推进随机采样、多 seed 消融、质量分层评价和新模型 baseline。首版代码和文档只作为历史参考，不作为新骨架的兼容包袱。

## 目标

本轮采用「实验骨架重建，新实验优先」路线：

- 抽出统一数据工厂，集中管理索引读取、审计、过滤、抽样、Dataset 和 DataLoader 构建。
- 支持 `random` 和 `stratified_random` 两种正式抽样策略，默认使用 `stratified_random`。
- 保留 `head` 作为 debug 策略，只用于快速复现和排查，不作为训练或验证默认。
- 新增公共实验基类和 THO 任务特化实验类，统一一次训练 run 的准备、训练、评价、产物保存和日志记录。
- 简化训练与评价脚本，使脚本只负责 CLI 参数解析和调用实验层。
- 更新文档，让使用说明、脚本索引和实验记录与新结构一致。
- 清理过时或重复说明；历史文档保留为归档资料，不约束新实现。

## 非目标

本轮不做以下内容：

- 不引入腹带、胸腹综合努力或多任务输出。
- 不新增 PatchTST、DLinear、TimesNet 等模型 baseline。
- 不改变 `WeakSyncLoss` 的默认训练目标。
- 不加入 RR 辅助头或 RR 派生损失。
- 不迁移到完整 Hydra 工程。
- 不重做 Stage 2.1 数据集制作流程。
- 不把参考仓库代码直接并入当前仓库。
- 不保证旧 run 的采样口径继续作为推荐口径。
- 不为历史脚本内部结构保留兼容层；只尽量保留常用 CLI 路径和参数。

这些内容在实验骨架稳定后再分阶段推进。

## 参考模式

参考 `/mnt/disk_code/marques/reference_repos/Time-Series-Library` 的工程组织方式，但只吸收适合当前阶段的部分：

- `run.py`：入口只做参数解析、任务路由和运行控制。
- `exp/exp_basic.py`：实验类负责模型构建、设备选择和公共接口。
- `exp/exp_long_term_forecasting.py`：训练、验证、测试和结果保存形成闭环。
- `data_provider/data_factory.py`：数据集与 DataLoader 通过统一工厂创建。

本仓库不复制其多任务框架，只借鉴「薄脚本 + 实验层 + 数据工厂 + 模型注册」的边界划分。

## 目标结构

新增和调整后的核心结构：

```text
resp_train/
  data/
    factory.py
    index.py
    audit.py
    dataset.py
    cache.py
  experiments/
    __init__.py
    base.py
    tho.py
  engine/
    train.py
  losses/
    weak.py
  metrics/
    baseline.py
    evaluate.py
    signal.py
  models/
    registry.py
    unet1d.py
  utils/
    run.py

scripts/
  train_tho_small.py
  eval_tho_small.py
  audit_tho_dataset.py
  baseline_tho_hilbert.py
  plot_tho_predictions.py
  summarize_tho_runs.py

docs/
  tho_small_training.md
  experiments/
    tho_small_mixed_zscore_20260607.md
  superpowers/
    specs/
    plans/
```

暂不移动脚本目录。脚本数量仍然可控，保持路径稳定能减少使用成本。若后续加入腹带、整夜推理和模型消融脚本，再单独迁移到 `scripts/data/`、`scripts/train/`、`scripts/diagnostics/`。

## 数据工厂设计

新增 `resp_train/data/factory.py`，提供面向实验层的统一入口。

核心接口：

```python
def build_window_data(
    cfg,
    *,
    split: str,
    max_windows: int | None,
    sample_strategy: str | None = None,
    sample_seed: int | None = None,
    shuffle: bool,
) -> WindowDataBundle:
    ...
```

`WindowDataBundle` 至少包含：

- `rows`：过滤和抽样后的索引行。
- `dataset`：`RespWindowDataset`。
- `loader`：PyTorch `DataLoader`。
- `audit`：完整审计 DataFrame 或摘要信息。

数据流：

1. `read_index(cfg.data.dataset_root, cfg.data.index_csv)` 读取索引。
2. `add_usable_flag(df, cfg)` 增加可用性标记。
3. `filter_index(..., split=..., max_windows=...)` 根据 `input_set`、`split`、`usable` 和抽样策略筛选窗口。
4. `RespWindowDataset(..., preload_windows=cfg.data.preload_windows)` 构建 Dataset。
5. `DataLoader` 根据训练或评价场景设置 `shuffle`、`batch_size` 和 `num_workers`。

训练和评价必须复用同一数据工厂，避免 split、过滤、抽样或 preload 行为分叉。

## 抽样策略

在 `configs/tho_small.yaml` 中新增：

```yaml
data:
  train_sample_strategy: stratified_random
  val_sample_strategy: stratified_random
  train_sample_seed: 20260601
  val_sample_seed: 20260602
  stratify_column: residual_quality_class
```

支持策略：

- `random`：过滤后按 `sample_seed` 随机抽取 `max_windows` 个窗口，再按 `dataset_row_id` 排序，保证训练可复现。
- `stratified_random`：按 `stratify_column` 分层抽样，尽量保持质量类别比例，再按 `dataset_row_id` 排序。
- `head`：按 `dataset_row_id` 排序后取前 `max_windows` 个窗口，仅用于 debug 和定位问题。

边界规则：

- `max_windows=None` 时不抽样，只排序返回。
- 可用窗口数小于等于 `max_windows` 时返回全部窗口。
- train 和 val 可分别指定策略，默认都使用 `stratified_random`。
- train 和 val 使用独立 seed。多 seed 训练实验默认只改变训练随机性和训练抽样，验证集 seed 保持固定，避免每个 run 的验证集不同导致超参数比较失真。
- `stratified_random` 的分层列不存在时 fail-fast 报错。
- 分层配额使用比例分配，余数按小数部分从大到小补齐；若某层样本不足，将剩余配额分配给仍有可用样本的层。
- 抽样结果必须可复现，测试中固定 seed 后行 ID 完全一致。

当前实验阶段默认使用 `stratified_random` 做后续消融，避免 train 或 val 集过度偏向某一类残差质量窗口。`head` 不再被视为正式训练策略；如果需要排查某个固定窗口序列，可显式传入 `--set data.train_sample_strategy=head` 或 `--set data.val_sample_strategy=head`。

后续可以进一步加入 `group_stratified_random`，按 `samp_id` 或 `(samp_id, segment_id)` 做组级抽样，降低相邻滑动窗口泄漏和重复片段影响。本轮先完成窗口级分层随机，组级抽样作为下一阶段数据划分增强。

## 后续 split 研究备忘

以下想法只作为后续研究备忘，不进入本轮实现主线。

数据量较小时，严格的 train/val/test 划分需要同时考虑个体间差异、个体内片段差异和滑动窗口相关性。同一人的数据是否可以同时出现在训练、验证和测试中，需要按研究问题拆开判断：

- 如果目标是评估同一受试者内的片段泛化，可以允许同一 `samp_id` 分布在不同 split，但要避免高度重叠窗口跨 split。
- 如果目标是评估跨受试者泛化，应采用 subject-level holdout，保证同一 `samp_id` 只出现在一个 split。
- 如果目标是模型开发阶段的稳定调参，可以固定一个分层随机 val 集，同时另设更严格的 subject-level test 集。

后续可以考虑做 split 预校验，而不是只依赖原始波形相似性：

- 个体级特征：`samp_id`、段数、窗口数、可用窗口比例、残差质量类别分布。
- 信号级特征：输入和目标 RMS、主频分布、频谱能量分布、包络统计、valid ratio。
- 任务级特征：平凡基线指标、BCG 与参考的频谱主峰差、峰谷读数稳定性。
- 相似性或覆盖度：用上述特征做聚类、距离矩阵或分布差异检查，辅助判断 train/val/test 是否覆盖相似难度区间。

这类相似性预校验可以帮助发现分布极不均衡的 split，但不应过早变成强制均衡算法。过度追求 split 间相似，可能削弱真实泛化评估。因此建议先实现诊断报告，再讨论是否引入 `group_stratified_random`、subject-level holdout 或基于特征的分层策略。

## 训练策略

参考 Time-Series-Library 的训练闭环，本轮只吸收必要训练策略：

- **best checkpoint**：每轮验证后保存当前最佳模型，最终评价只使用最佳 checkpoint。
- **early stopping**：新增 `training.patience` 和 `training.min_delta`，默认启用早停，避免小样本 weak loss 下继续训练放大过拟合。
- **学习率调度**：新增轻量 `training.lr_scheduler`，支持 `none` 和 `cosine`。默认先使用 `none`，保留 `cosine` 用于后续较长训练。
- **梯度裁剪**：新增 `training.grad_clip_norm`，默认关闭；若频谱损失或新模型导致梯度不稳定，可用配置开启。
- **多 seed 运行记录**：每次 run 必须记录训练 seed、train 抽样 seed、val 抽样 seed、抽样策略和最终选中的窗口数量。
- **AMP**：新增 `training.use_amp=false`，首版默认关闭。只有在 CUDA 环境和较大模型训练时再开启。

不吸收的训练策略：

- 不加入复杂分布式训练。
- 不加入自动多模型 benchmark runner。
- 不加入会改变当前研究问题的 forecasting decoder 结构。

训练日志至少记录：

- 设备、seed、抽样策略和窗口数量。
- 每轮 train/val loss 及各 loss 分量。
- best epoch、best val loss 和早停原因。
- baseline 指标和模型指标文件路径。

## 公共实验层设计

新增 `resp_train/experiments/base.py`，放置跨 THO、ABD 和 effort 任务共享的实验编排能力。公共基类只承载确定会复用的工程流程，不提前抽象任务语义。

推荐接口：

```python
class BaseExperiment:
    def __init__(self, cfg):
        ...

    def train(self) -> Path:
        ...

    def evaluate_checkpoint(self, checkpoint_path: Path, *, output: Path, metrics_output: Path | None) -> None:
        ...
```

`BaseExperiment` 负责：

1. 创建 run 目录。
2. 保存 resolved `config.yaml`。
3. 初始化 logger、seed 和 device。
4. 调用任务特化方法构建 train/val 数据。
5. 构建 model、loss 和 optimizer。
6. 执行训练与验证循环。
7. 保存最佳 `checkpoint.pt`。
8. 调用任务特化方法生成评价指标和诊断预测。
9. 打印 run 目录。

公共基类提供以下可覆盖钩子：

```python
class BaseExperiment:
    task_name: str

    def build_data(self) -> ExperimentData:
        ...

    def build_model(self) -> torch.nn.Module:
        ...

    def build_loss(self) -> torch.nn.Module:
        ...

    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        ...

    def run_baseline(self, data: ExperimentData, run_dir: Path) -> None:
        ...

    def evaluate_best(self, model: torch.nn.Module, data: ExperimentData, run_dir: Path) -> None:
        ...
```

`ExperimentData` 是轻量数据容器，至少包含 `train`、`val`、`audit_frame` 和 `audit_summary`。这样 ABD 和 effort 后续可以复用 run 生命周期、checkpoint、日志、配置快照和训练循环，只替换数据键、target 语义、baseline 和评价口径。

公共函数组织原则：

- `resp_train/engine/` 只放与任务无关的 PyTorch 训练循环、验证循环、checkpoint 和 prediction collection。
- `resp_train/data/` 只放数据事实、索引、审计、抽样、缓存和 Dataset，不放任务指标。
- `resp_train/metrics/signal.py` 放任务无关信号工具，如频谱、包络、主频、峰谷检测。
- `resp_train/metrics/baseline.py` 和 `resp_train/metrics/evaluate.py` 可以按任务继续拆分；当 ABD 或 effort 进入时，新增任务特化文件，避免 THO 语义污染公共指标。
- `resp_train/experiments/` 负责任务编排，把公共训练生命周期和任务特化评价连接起来。

## THO 实验层设计

新增 `resp_train/experiments/tho.py`，定义 `ThoExperiment(BaseExperiment)`。

`ThoExperiment` 负责：

1. 使用数据工厂构建 THO train/val 数据。
2. 保存 `audit.csv`。
3. 运行 THO val 子集平凡基线并保存 `baseline_metrics.csv`。
4. 构建 `r_tho_hat(t)` 模型和 `WeakSyncLoss`。
5. 载入最佳 checkpoint，生成完整 val `metrics.csv`。
6. 保存少量 `predictions.npz` 作为诊断样本。

`evaluate_checkpoint()` 负责：

1. 解析 checkpoint 同目录 `config.yaml` 或显式 `--config`。
2. 校验 checkpoint 内配置与当前评价配置的一致性。
3. 通过数据工厂构建 val 数据。
4. 保存诊断预测和可选完整指标。

`ThoExperiment` 保持 THO 任务语义清晰。公共基类只放生命周期，不把 `r_tho_hat(t)`、胸带 baseline 或 THO 指标提升为全局概念。

## 脚本调整

`scripts/train_tho_small.py` 调整为薄入口：

- 解析 `--config` 和重复 `--set`。
- 检查必需依赖。
- 调用 `ThoExperiment(cfg).train()`。
- 打印 run 目录。
- 常用 CLI 路径保留；脚本内部结构可以完全重写。

`scripts/eval_tho_small.py` 调整为薄入口：

- 解析 checkpoint、output、metrics-output、config 和 overrides。
- 调用实验层评价函数。

`scripts/audit_tho_dataset.py` 和 `scripts/baseline_tho_hilbert.py` 继续保留独立入口，但内部应复用数据工厂，保证口径统一。

## 文档整理

需要更新：

- `docs/tho_small_training.md`：加入抽样策略、实验层结构、推荐 smoke test 和新的 run 产物说明。
- `scripts/README.md`：说明脚本仍平铺，但训练/评价已复用实验层。
- `docs/experiments/tho_small_mixed_zscore_20260607.md`：追加说明既有 4 个 run 使用前缀采样，结论只能作为首版 smoke 和现象记录，不能作为正式消融结论。

需要保留：

- `docs/superpowers/specs/2026-06-01-tho-small-train-design.md`：作为首版训练骨架历史设计，不再约束新实现。
- `docs/superpowers/plans/2026-06-01-tho-small-train.md`：作为首版实现计划历史记录，不再约束新实现。
- `docs/experiments/tho_small_mixed_zscore_20260607.md`：作为真实实验记录。

可以清理：

- 明显过时的脚本路径说明。
- 与当前默认配置冲突的训练命令。
- 继续窄范围 smooth 消融的旧建议；新骨架完成前，超参数结论不应继续扩大解释。
- 任何把 `head` 前缀采样描述为默认训练策略的内容。

## 测试策略

新增或更新测试：

- `tests/test_data_index_audit.py`
  - `random` 在固定 seed 下可复现。
  - `stratified_random` 是默认策略。
  - `stratified_random` 保持主要质量类别比例。
  - train 和 val 使用独立抽样 seed。
  - `head` 仅在显式指定时使用，并按排序前缀返回。
  - 分层列缺失时报错。

- `tests/test_data_factory.py`
  - 能从临时索引和临时 NPZ 构建 train/val DataLoader。
  - train 和 val 能使用不同抽样策略。
  - 空 train 或 val 数据时给出明确错误。

- `tests/test_base_experiment.py`
  - 公共实验基类能创建 run 目录、保存配置、保存 checkpoint，并调用任务钩子。
  - 早停在验证损失不改善时触发。
  - 公共基类不包含 THO 专属指标或 baseline 逻辑。

- `tests/test_tho_experiment.py`
  - 使用小型临时数据集跑 1 epoch smoke。
  - 断言 run 目录包含 `config.yaml`、`checkpoint.pt`、`train_history.csv`、`metrics.csv` 和 `predictions.npz`。

保留既有测试：

- 配置测试。
- Dataset/cache 测试。
- loss 测试。
- signal metrics 测试。
- engine smoke 测试。
- 诊断脚本测试。

## 验证命令

实现完成后至少运行：

```bash
./.venv/bin/python -m pytest tests/test_data_index_audit.py tests/test_data_factory.py tests/test_base_experiment.py tests/test_tho_experiment.py -v
./.venv/bin/python -m pytest tests -q
```

再运行一个小型训练 smoke：

```bash
./.venv/bin/python scripts/train_tho_small.py \
  --config configs/tho_small.yaml \
  --set data.train_sample_seed=1 \
  --set data.val_sample_seed=2 \
  --set data.max_train_windows=16 \
  --set data.max_val_windows=8 \
  --set training.epochs=1 \
  --set model.base_channels=4 \
  --set outputs.max_prediction_windows=4
```

若环境缺少依赖或数据路径不可访问，应停止并报告具体阻塞，不伪造通过。

## 风险与处理

- **抽样改变实验结果口径**：这是有意改变。旧实验按前缀采样归档，新实验默认使用分层随机，避免滑动窗口前缀偏差。
- **公共基类过度抽象**：公共基类只承载 run 生命周期和训练生命周期，不承载任务语义；THO、ABD 和 effort 指标留在任务类。
- **脚本兼容性变化**：保留常用脚本路径和主要 CLI 参数，但允许重写内部结构。
- **测试数据构造成本上升**：测试使用临时小型 NPZ 和最小索引，不依赖真实数据集。
- **文档清理误删研究上下文**：历史设计、历史计划和实验记录不删除，但明确标注为历史资料。

## 验收标准

- 默认训练和验证不再使用 `head` 前缀采样。
- 抽样策略可通过 `--set data.train_sample_strategy=...` 和 `--set data.val_sample_strategy=...` 分别切换。
- train 与 val 抽样 seed 独立记录，多 seed 实验可以固定验证集。
- 训练支持 best checkpoint 和 early stopping。
- 训练和评价复用同一数据工厂。
- 公共 `BaseExperiment` 承担 run 生命周期，`ThoExperiment` 承担 THO 任务语义。
- `scripts/train_tho_small.py` 不再直接承担完整训练编排。
- 小型 smoke run 可以生成完整 run 产物。
- 测试覆盖抽样、数据工厂、公共基类和 THO 实验层。
- 文档明确区分历史前缀采样实验和后续分层随机实验。
