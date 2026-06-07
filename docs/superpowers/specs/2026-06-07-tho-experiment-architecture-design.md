# THO 实验工程骨架重建设计

## 背景

当前仓库已经完成胸带参考 `r_tho_hat(t)` 的首版小规模训练闭环，包含数据审计、平凡基线、训练、验证、checkpoint、评价指标、诊断预测和绘图脚本。首版设计偏向「最小跑通」，核心逻辑主要集中在 `scripts/train_tho_small.py` 和 `scripts/eval_tho_small.py`。

经过 2026-06-07 的 `mixed_zscore` 小规模实验后，当前最影响后续研究可信度的不是模型数量，而是实验组织与采样口径：

- 当前 `max_train_windows` 和 `max_val_windows` 默认取排序后的前缀窗口，容易引入样本顺序、片段和质量分布偏差。
- 训练脚本承担数据准备、审计、baseline、模型构建、训练、评价和诊断保存等多重职责。
- 训练和评价脚本各自重建数据管线，后续容易产生口径分叉。
- 文档已有首版训练说明、实验记录和计划文档，但缺少一个面向下一阶段实验工程的结构化设计。

本轮目标是在保留首版能力的前提下，重建 THO 实验工程骨架，使后续可以稳定推进随机采样、多 seed 消融、质量分层评价和新模型 baseline。

## 目标

本轮采用「实验骨架重建，保留当前能力」路线：

- 抽出统一数据工厂，集中管理索引读取、审计、过滤、抽样、Dataset 和 DataLoader 构建。
- 支持 `head`、`random`、`stratified_random` 三种窗口抽样策略，默认保持 `head` 以兼容既有实验。
- 新增 THO 实验编排层，统一一次训练 run 的准备、训练、评价、产物保存和日志记录。
- 简化训练与评价脚本，使脚本只负责 CLI 参数解析和调用实验层。
- 更新文档，让使用说明、脚本索引和实验记录与新结构一致。
- 清理明显过时或重复的说明，但保留有研究上下文的历史文档。

## 非目标

本轮不做以下内容：

- 不引入腹带、胸腹综合努力或多任务输出。
- 不新增 PatchTST、DLinear、TimesNet 等模型 baseline。
- 不改变 `WeakSyncLoss` 的默认训练目标。
- 不加入 RR 辅助头或 RR 派生损失。
- 不迁移到完整 Hydra 工程。
- 不重做 Stage 2.1 数据集制作流程。
- 不把参考仓库代码直接并入当前仓库。

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
def build_window_data(cfg, *, split: str, max_windows: int | None, shuffle: bool) -> WindowDataBundle:
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
  sample_strategy: head
  sample_seed: 20260601
  stratify_column: residual_quality_class
```

支持策略：

- `head`：按 `dataset_row_id` 排序后取前 `max_windows` 个窗口，保持历史兼容。
- `random`：过滤后按 `sample_seed` 随机抽取 `max_windows` 个窗口，再按 `dataset_row_id` 排序，保证训练可复现。
- `stratified_random`：按 `stratify_column` 分层抽样，尽量保持质量类别比例，再按 `dataset_row_id` 排序。

边界规则：

- `max_windows=None` 时不抽样，只排序返回。
- 可用窗口数小于等于 `max_windows` 时返回全部窗口。
- `stratified_random` 的分层列不存在时 fail-fast 报错。
- 分层配额使用比例分配，余数按小数部分从大到小补齐；若某层样本不足，将剩余配额分配给仍有可用样本的层。
- 抽样结果必须可复现，测试中固定 seed 后行 ID 完全一致。

当前实验阶段推荐使用 `stratified_random` 做后续消融，避免 val 集过度偏向某一类残差质量窗口。

## THO 实验层设计

新增 `resp_train/experiments/tho.py`，定义一个轻量实验编排类或函数集合。

推荐接口：

```python
class ThoExperiment:
    def __init__(self, cfg):
        ...

    def train(self) -> Path:
        ...

    def evaluate_checkpoint(self, checkpoint_path: Path, *, output: Path, metrics_output: Path | None) -> None:
        ...
```

`train()` 负责：

1. 创建 run 目录。
2. 保存 resolved `config.yaml`。
3. 初始化 logger、seed 和 device。
4. 构建 train/val 数据。
5. 保存 `audit.csv`。
6. 运行 val 子集平凡基线并保存 `baseline_metrics.csv`。
7. 构建 model、loss 和 optimizer。
8. 执行训练与验证循环。
9. 保存最佳 `checkpoint.pt`。
10. 载入最佳 checkpoint，生成完整 val `metrics.csv`。
11. 保存少量 `predictions.npz` 作为诊断样本。
12. 打印 run 目录。

`evaluate_checkpoint()` 负责：

1. 解析 checkpoint 同目录 `config.yaml` 或显式 `--config`。
2. 校验 checkpoint 内配置与当前评价配置的一致性。
3. 通过数据工厂构建 val 数据。
4. 保存诊断预测和可选完整指标。

实验层应保持 THO 任务语义清晰，不提前抽象成多任务基类。后续出现 ABD 或 effort 任务时，再抽出公共基类。

## 脚本调整

`scripts/train_tho_small.py` 调整为薄入口：

- 解析 `--config` 和重复 `--set`。
- 检查必需依赖。
- 调用 `ThoExperiment(cfg).train()`。
- 打印 run 目录。

`scripts/eval_tho_small.py` 调整为薄入口：

- 解析 checkpoint、output、metrics-output、config 和 overrides。
- 调用实验层评价函数。

`scripts/audit_tho_dataset.py` 和 `scripts/baseline_tho_hilbert.py` 可以继续保留独立入口，但内部应逐步复用数据工厂，保证口径统一。

## 文档整理

需要更新：

- `docs/tho_small_training.md`：加入抽样策略、实验层结构、推荐 smoke test 和新的 run 产物说明。
- `scripts/README.md`：说明脚本仍平铺，但训练/评价已复用实验层。
- `docs/experiments/tho_small_mixed_zscore_20260607.md`：追加说明既有 4 个 run 使用历史 `head` 前缀采样，后续结论需要用随机或分层随机复核。

需要保留：

- `docs/superpowers/specs/2026-06-01-tho-small-train-design.md`：作为首版训练骨架历史设计。
- `docs/superpowers/plans/2026-06-01-tho-small-train.md`：作为首版实现计划历史记录。
- `docs/experiments/tho_small_mixed_zscore_20260607.md`：作为真实实验记录。

可以清理：

- 明显过时的脚本路径说明。
- 与当前默认配置冲突的训练命令。
- 已经不再推荐继续做窄范围 smooth 消融、但没有说明抽样偏差的下一步建议。

## 测试策略

新增或更新测试：

- `tests/test_data_index_audit.py`
  - `head` 保持历史排序前缀行为。
  - `random` 在固定 seed 下可复现。
  - `random` 与 `head` 在构造样例中产生不同窗口集合。
  - `stratified_random` 保持主要质量类别比例。
  - 分层列缺失时报错。

- `tests/test_data_factory.py`
  - 能从临时索引和临时 NPZ 构建 train/val DataLoader。
  - 空 train 或 val 数据时给出明确错误。

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
./.venv/bin/python -m pytest tests/test_data_index_audit.py tests/test_data_factory.py tests/test_tho_experiment.py -v
./.venv/bin/python -m pytest tests -q
```

再运行一个小型训练 smoke：

```bash
./.venv/bin/python scripts/train_tho_small.py \
  --config configs/tho_small.yaml \
  --set data.sample_strategy=stratified_random \
  --set data.sample_seed=1 \
  --set data.max_train_windows=16 \
  --set data.max_val_windows=8 \
  --set training.epochs=1 \
  --set model.base_channels=4 \
  --set outputs.max_prediction_windows=4
```

若环境缺少依赖或数据路径不可访问，应停止并报告具体阻塞，不伪造通过。

## 风险与处理

- **抽样改变实验结果口径**：默认保持 `head`，文档明确旧实验为前缀采样；新实验用 `random` 或 `stratified_random` 单独记录。
- **实验层过度抽象**：本轮只做 `ThoExperiment`，不提前做通用多任务框架。
- **脚本兼容性变化**：保留原脚本路径和主要 CLI 参数。
- **测试数据构造成本上升**：测试使用临时小型 NPZ 和最小索引，不依赖真实数据集。
- **文档清理误删研究上下文**：历史设计、历史计划和实验记录不删除，只更新当前使用说明。

## 验收标准

- 旧的训练命令仍可运行。
- 新增抽样策略可通过 `--set data.sample_strategy=...` 切换。
- 训练和评价复用同一数据工厂。
- `scripts/train_tho_small.py` 不再直接承担完整训练编排。
- 小型 smoke run 可以生成完整 run 产物。
- 测试覆盖抽样、数据工厂和 THO 实验层。
- 文档明确区分历史 head 采样实验和后续随机/分层随机实验。
