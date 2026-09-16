# E3：W0 五指标关联与 RR—努力不一致比例

协议 ID：`e3-w0-metric-association-v1-20260916`。

状态：统计定义、独立入口与 synthetic CPU 验证已完成；正式再分析由用户执行。E2 完成状态见 `e2_w0_effort_results_20260916.md`。

## 1. 对象与科学问题

回答：在冻结 W0 的窗口分布中，RR 误差较小时，包络轨迹、全局包络调制及波形形态误差如何分布？分析三个 W0（seed 20260811/20260812/20260813，validation-selected epoch 13/15/14）的既有五主指标；validation（2675 窗口、7 个 samp_id）与 test（2310 窗口、8 个 samp_id）分别报告。

本轮复用已保存的 CSV 和窗口时间元数据。输出是新的分布性证据，既有模型、指标、checkpoint、split 与 E1/E2 结果保持其冻结身份。test 为有重复访问历史的研究测试集。

## 2. 统计定义

### 2.1 五指标关联

四项 error 保留原单位；signed PCC 转为 `1 − PCC`，使五个分析轴均为越大误差越大。每个 seed、split 分别计算 10 对 Spearman 秩相关（平均秩处理并列值）：

- `pooled_windows`：该 split 的窗口分布，保留全部重叠窗口，作为主要描述。
- `within_samp`：每个 samp_id 内分别计算，显示个体间异质性。
- `samp_means`：先在每个 samp_id 内平均五指标，再对 7/8 个 samp_id 的均值计算相关；与窗口相关分别报告。

主表保留每个 seed 的估计；三 seed mean/sample SD（ddof=1）描述模型随机性。常量输入或不足 3 个单位时，相关系数记为空并写明原因、有效单位数，汇总列明有定义的 seed 数。相关计算不输出 p-value 或独立窗口置信区间。

### 2.2 不一致比例与参照阈值

描述性主设置：Whole RR absolute error 与 Local RR MAE **同时 ≤1 bpm**；另一属性误差 **严格大于 validation 第 75 百分位**。轨迹误差、全局包络调制误差与 `1−PCC` 三个属性各自报告。

validation 参照阈值通过以下固定算法取得：先按同一 dataset_row_id 对三个 seed 的该项窗口指标求均值，再对 2675 个唯一窗口计算线性插值分位数（NumPy `method="linear"`）。此均值只用于共同参照分布；每个 seed 的关联和不一致比例使用该 seed 的原始指标。三个 seed 及两个 split 使用同一组阈值。test 数值不参与阈值估计。

完整敏感性矩阵为 RR 门槛 `{0.5, 1.0, 2.0}` bpm × validation 分位点 `{0.50, 0.75, 0.90}`，事后按全表解释。分位数定义的是相对 validation 参照分布的高误差，RR 门槛为便于描述的操作定义，均不代表临床合格线。

每项比例保存：总窗口 N、RR 满足窗口数 R、高属性误差窗口数 H、两者同时成立的窗口数 D，以及 `D/N`（总体不一致占比）、`D/R`（RR 满足条件下的高属性误差比例）、`H/N`（参照高误差占比）。R=0 时条件比例为空，显式状态为 `no_rr_qualified_windows`。

同时提供逐 samp_id 比例与 samp_id 等权宏平均；条件比例的宏平均明确记录有定义及分母为零的 samp_id 数。主 pooled 比例保留窗口权重，宏平均显示记录长度对结果的影响。

### 2.3 重叠窗口与分母

完整窗口是主分析视图。敏感性视图按每个 samp_id 的 `window_start_s`、`dataset_row_id` 排序，选取最早窗口，随后贪心选择 `start >= 上一已选 end` 的窗口。同一 samp_id 的时间坐标必须来自同一个 source_npz；选择覆盖该 samp_id 的所有 segment，不在 segment 边界重置时间。窗口为半开区间 `[start,end)`。选择仅依赖元数据，并为三个 seed 共用。

逐行保存两个视图的成员身份；非重叠视图继续使用完整 validation 的同一组参照阈值。这一视图检验采样重叠敏感性，仍存在同一受试者内相关。

冻结 W0 的五主指标分母已完整：本阶段要求每个来源的行数、顺序、samp_id、split 与元数据一致，五主指标全部有限，Whole/Local RR 与 joint target eligibility 全为真，预测退化为零。缺行、非有限值、资格变化或重复身份均显式失败，修复前不产生完成统计。分母审计列出 expected、observed、eligible 和 excluded（合同下为零）窗口数。

## 3. 来源与可复现产物

源身份锚点为冻结 E2 训练/test 实现锁及 E2 test 汇总 manifest。W0 validation 使用原 `metrics.csv` 与冻结 `metrics_summary.csv`；test 使用原 `research_test_metrics.csv` 及 summary。validation 逐窗口文件在本轮锁定字节身份，并核对其五指标均值与已有冻结 summary。窗口元数据来自完成的 E2 `val_rows.csv` / `test_rows.csv`，以其原 manifest 或锁中的 SHA-256 核对。

独立入口：`scripts/analyze_e3_w0_metrics.py`。`prepare-lock` 只核验既有文件身份并冻结代码/协议与输入清单；正式 `analyze` 校验清单后读取 CSV，先从 validation 计算参照阈值，再计算两个 split 全部统计。

独立输出：`runs/e3_w0_metric_association_v1/analysis/<attempt>/`。输出包括：

- `source_audit.csv`、`window_membership.csv`、`thresholds.csv`；
- `associations.csv`、`association_seed_summary.csv`；
- `discordance.csv`、`discordance_macro.csv`、`discordance_seed_summary.csv`；
- 实现锁快照、环境/命令/版本、结果口径与来源回执、lifecycle、manifest 和 freeze receipt。

每次 attempt 排他创建；失败状态与部分产物保留。相同锁已有成功分析时校验并返回已有产物。缺失或损坏完成产物显式失败。

## 4. 验证与运行

允许的实现验证使用 synthetic CPU fixture，覆盖相关方向与并列秩、阈值仅依赖 validation、分母为零、samp 等权宏平均、跨 segment 的不重叠选择、输入身份/资格/finite 失败、来源哈希漂移，以及完整成功/失败生命周期。

2026-09-16 定向验证：19 passed。测试只使用临时合成来源；完整原始 CSV 的 E3 统计尚待正式命令生成。

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  ./.venv/bin/python -m pytest tests/test_e3_metric_association.py -q
./.venv/bin/python scripts/analyze_e3_w0_metrics.py prepare-lock
```

提交实现与锁后，由用户运行 CPU 再分析：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  ./.venv/bin/python scripts/analyze_e3_w0_metrics.py analyze
```

验收：两个 split × 三个 seed 的来源与分母完整；两种窗口视图、10 对相关和完整阈值矩阵齐全；宏平均和 seed 汇总报告定义数；完整 manifest/freeze receipt 成功。结果解释按相关幅度、条件比例、参照高误差比例、个体差异及敏感性视图共同判断。
