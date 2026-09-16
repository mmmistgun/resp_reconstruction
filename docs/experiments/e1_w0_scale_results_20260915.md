# E1 尺度重排实验结果与论文证据记录

状态：validation 的 12 项评价已完成并形成不可覆盖 manifest；论文证据整理完成。

## 1. 冻结身份

- 协议：`e1-w0-scale-topology-v1-20260915`；执行口径为 r2，FULL 容差 `rtol=1e-3, atol=0`。
- 干净执行 commit：`d8bb76304c39d2ca30bf3b4d2ef6b50135a9824a`。
- 三个 seed：`20260811/20260812/20260813`；validation-selected epoch：`13/15/14`。
- 运行目录：`/mnt/disk_code/marques/resp_reconstruction/runs/e1_w0_scale_topology_v1/validation/validation_6ed77404bc56_20260915T144337Z_ffd960924da7`。
- implementation lock SHA-256：`6ed77404bc56c56eff7d0328700eda8a2348ad167b2172735556077e2bf82f79`。
- 固定置换：NumPy PCG64，种子 `20260915`；索引 SHA-256：`fd7284a27c81b75d156229b275a42ab502790d8fba2956ef669ebbd25a0293e9`。
- manifest SHA-256：`bd9adfd02534259d27a48a856e621537078182afc8aaa2cf775090e1313a365e`；对应 `freeze_receipt.json`。
- 主来源：`paired_delta_summary.csv`、`paired_seed_delta.csv`、`three_seed_summary.csv`、`seed_summary.csv`；完整逐窗口产物位于各 `seed_<seed>/<condition>/` 子目录。

## 2. 验收与核对范围

每条件 3×2675=8025 条记录；四条件共 32100 条 metrics 和 32100 条 FiLM 配对记录，覆盖 7 个 samp_id。逐窗口身份、顺序及逐 seed 汇总一致，五项主指标全部有限，prediction degeneracy 为 0。三个 FULL 按 r2 合同通过；实际最大绝对偏差为 `1.100059492692429e-6`，最大相对偏差为 `5.727871674223612e-6`。

本次只读核对验证了 manifest 与冻结回执，以及 67 个非 raw 文件的 SHA-256；6 个大型 raw FiLM 文件核对大小，其 SHA-256 沿用运行时 manifest 记录。本次没有重新读取 raw 张量计算统计或运行模型。

## 3. 五指标配对变化

下表为同 seed 干预相对 FULL 的变化，再计算三个 seed 的均值 ± 样本 SD（ddof=1）。四项 error 使用百分比，PCC 使用绝对下降；正值表示恶化，括号为恶化 seed 数。原始 `delta_of_seed_means` 列对四项 error 另存三 seed 指标均值之间的相对百分比变化，对 PCC 另存均值的绝对下降；它与下表的逐 seed 配对变化均值分别记录。

| 条件 | Whole RR (%) | Local RR (%) | trajectory (%) | global modulation (%) | PCC 绝对下降 |
|---|---:|---:|---:|---:|---:|
| 尺度循环平移 12 位 | +0.001 ± 1.975 (1/3) | -0.208 ± 0.986 (1/3) | -0.400 ± 0.559 (1/3) | +1.704 ± 2.147 (2/3) | +0.000070 ± 0.000592 (2/3) |
| 尺度反转 | +3.009 ± 2.172 (3/3) | +0.956 ± 1.082 (2/3) | +0.325 ± 0.987 (2/3) | +1.060 ± 1.104 (3/3) | +0.001512 ± 0.000463 (3/3) |
| 固定尺度置换 | -2.190 ± 5.388 (1/3) | +0.485 ± 1.650 (2/3) | -0.367 ± 1.875 (2/3) | +20.014 ± 17.594 (3/3) | +0.005919 ± 0.001286 (3/3) |

## 4. FiLM 响应

统计对象为 `tanh` 前的 raw gamma/beta。同 seed、同窗口、同通道和时间位置作差取绝对值，先在窗口内平均，再在窗口及 seed 层面汇总。以下为三 seed 均值 ± 样本 SD。

| 条件 | raw gamma 配对 MAE | raw beta 配对 MAE |
|---|---:|---:|
| 尺度循环平移 12 位 | 0.118974 ± 0.006071 | 0.114680 ± 0.012775 |
| 尺度反转 | 0.149538 ± 0.023537 | 0.147459 ± 0.031787 |
| 固定尺度置换 | 0.471923 ± 0.067217 | 0.437518 ± 0.070617 |

## 5. 解释与后续实验

固定置换使 global modulation error 与 signed PCC 在三个 seed 中均恶化；其中 global error 分别增加 38.37%、3.30%、18.37%，变化幅度具有明显 seed 差异。反转使五项主指标的配对变化均值均朝恶化方向，其中 Whole RR、global error 和 PCC 为 3/3 seed 恶化；其余两项为 2/3。循环平移的均值响应较弱，Whole RR 的近零均值同时伴随约 1.98 个百分点的 seed SD。

这些结果支持冻结 W0 对指定尺度重排具有属性相关的功能敏感性，global modulation 与 PCC 的响应最一致。三个操作同时涉及尺度槽位、方向或邻接及卷积边界；解释范围为完整 validation、指定重排和三个已训练模型。结果后解释采用完整连续效应量、SD 和方向数。当前 E4 所需的“基本不依赖尺度结构”前提未由 E1 支持，E4 保持条件关闭。

### E2 状态

E2 三个 seed 的训练、validation 与 test 配对汇总已完成。相对努力损失在两个 split 上均改善包络轨迹重建，test 上还改善全局包络调制误差与 signed PCC；RR 指标存在取舍。冻结身份、完整五指标和解释边界见[E2 完成记录](/mnt/disk_code/marques/resp_reconstruction/docs/experiments/e2_w0_effort_results_20260916.md)。

## 6. 论文映射

- 内部底稿第 5.3.1 节保存完整身份和数值。
- 中文稿 V-C 的 Table VI(b) 呈现尺度重排结果，Table VI(a) 保留条件内容与时间干预。
- `results_validation.md` 和科学证据账本增加 E1 支持性结果；结果汇总登记完整 validation 证据。
- 论文计划与待补实验清单登记 E1、E2 的完成状态与冻结结果。
