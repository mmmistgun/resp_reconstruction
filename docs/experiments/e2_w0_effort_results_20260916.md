# E2 相对努力损失消融：完成记录

状态：**三个 seed 的训练、validation、test 及配对汇总均已完成。** 本记录承接冻结训练协议与 test 附件的完成状态；后续引用下列产物。

## 1. 实验身份与完成范围

- 训练协议：`e2-w0-effort-ablation-v1-20260916`；test 协议：`e2-w0-effort-test-v1-20260916`。
- 科学变量：最终 W0 的 `effort_weight` 从 0.25 改为 0；同结构、同 seed 初始化、同训练预算、同数据与五主指标。完整目标对照复用原 W0 冻结结果。
- seed：20260811、20260812、20260813；每 seed 80 epochs / 6400 updates。
- E2 checkpoint 由完整 validation 的最早最小 Local RR 选择，epoch 分别为 31/36/33；W0 对照为 13/15/14。test 评价使用预先固定的 checkpoint。
- validation：每 seed 2675 窗口、7 个 samp_id，共 8025 条新增指标记录；test：每 seed 2310 窗口、8 个 samp_id，共 6930 条新增指标记录。
- 五主指标有限、分母完整，prediction degeneracy 为零；新旧臂逐窗口身份与 target eligibility 对齐。

| 身份 | 训练 / validation | test |
|---|---|---|
| 执行 commit | `4b8a56211b31c74d997110482c3d1c7a829db1ed` | `916ec1bdcd44ee65b278c2a3acdc4eaaa86e0c18` |
| 实现锁 SHA-256 | `29ff147334a358be0b807a5e068e280891522f7ceef831b4ed16105988589edf` | `81fa0622c3fba8060de5f531ad899417654b16d8bf10bc092fcadda7c0a03898` |
| 汇总 manifest SHA-256 | `ab0165f4846a2b74e83a7f0bf70bc9bd09f80cee53fecf76a322e9d8f54489ad` | `978d3d365fe9cab574244c1fd0761b0512a03f670c95a6fc5e2636cfa7b6d6cf` |

完整逐 seed 来源、selected epoch 与文件身份由各汇总的 `summary_receipt.json` 记录：

- [validation 汇总](/mnt/disk_code/marques/resp_reconstruction/runs/e2_w0_effort_ablation_v1/summary/summary_29ff147334a3_20260916T064752Z_180c97e12740/summary_receipt.json)
- [test 汇总](/mnt/disk_code/marques/resp_reconstruction/runs/e2_w0_effort_test_v1/summary/summary_81fa0622c3fb_20260916T080746Z_f2ef33859245/summary_receipt.json)

## 2. 统计口径与配对变化

先在每个 seed 内对完整窗口计算 sample-direct mean，再配对比较两臂。四项 error 的变化为 `100 × (sync_only − FULL) / FULL`；PCC 为 `FULL − sync_only`。正值表示移除努力项后恶化。表中为三个配对变化的均值 ± 样本 SD（ddof=1），括号为完整目标更好的 seed 数。

| Split | Whole RR (%) | Local RR (%) | trajectory (%) | global modulation (%) | PCC 绝对下降 |
|---|---:|---:|---:|---:|---:|
| validation | −2.583 ± 3.085 (0/3) | +1.497 ± 3.675 (2/3) | +5.268 ± 2.378 (3/3) | −3.068 ± 3.652 (1/3) | +0.003337 ± 0.004529 (2/3) |
| test | +2.423 ± 5.071 (2/3) | −1.515 ± 5.128 (1/3) | +5.180 ± 1.130 (3/3) | +5.878 ± 4.431 (3/3) | +0.006233 ± 0.001708 (3/3) |

两臂的原始指标、各自 mean/sample SD、逐 seed delta、方向数和 `delta_of_seed_means` 均保留在各 split 的 `seed_metrics.csv`、`paired_seed_delta.csv`、`three_seed_comparison.csv`。四项 error 的配对百分比均值与两个 seed 均值之间的百分比变化是不同统计量，正文引用应标明所用口径。

## 3. 结果解释

相对努力损失对包络轨迹重建的贡献在两个 split 上最一致：移除后 validation 误差增加 5.27%，test 增加 5.18%，均为三个 seed 同方向。test 的全局包络调制误差与 signed PCC 同样在三个 seed 上支持完整目标。

全局包络调制误差的 validation 均值更有利于同步损失单独训练，表明该属性的收益具有 split 依赖。Whole/Local RR 方向混合，且部分方向在两个 split 间改变。结论应同时呈现努力属性收益和 RR 取舍。

建议论文表述：相对努力损失在 validation 与 test 上均稳定改善包络轨迹重建；在 test 上还改善了全局包络调制误差和波形相关性，而呼吸率误差呈现指标间取舍。

seed SD 描述训练随机性；重叠窗口不构成独立重复。test 沿用既有受试者隔离数据，已有重复访问历史，证据解释为固定 validation-selected checkpoint 在复用研究测试集上的比较。

## 4. 验证与收尾

- 实现阶段 synthetic CPU 定向测试：训练消融 24 passed，test 入口 15 passed。
- validation 收尾已核验三个训练 run 的 history、checkpoint 与 80 epochs / 6400 updates，确认 selector 为 31/36/33，逐窗口与聚合值相符。
- test 汇总收尾只读复核了 55 项文件身份（含汇总、三个评价 attempt 与 W0 来源），并独立重算 15 个配对差值与五行 mean/sample SD、方向数，全部通过。
- 已完成的源码、协议、锁及运行产物作为冻结来源保留；本记录与论文证据文档承接当前状态。E2 的实验执行与统计汇总已收尾。
- 论文工作区 10 份相关文档已同步：内部底稿 5.5.2、正文 V-F / Table VIII、计划与待补清单、结果核验、科学证据账本及检查、全部实验汇总、贡献证据索引。候选补丁经独立只读审核通过，逐文件原 SHA 核验后安装并回读；保留了论文工作区既有编辑。
- 下一阶段 E3 独立建立冻结结果再分析协议，分析对象、相关系数、阈值、分母和窗口重叠口径由其专项协议定义。
