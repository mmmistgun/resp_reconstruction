# W0 CWT-FiLM 独立探索最终阶段总结

状态：本独立探索的行为分析、局部敏感性分析、gamma 完整重训验证与固定 research/development test 比较均已完成。本文件仅汇总既有正式证据与阶段决策，不新增实验、选择规则或结果口径。

## 1. 范围与证据链

本阶段围绕 W0 `crd_tf102_w` 的 CWT-FiLM 条件调制回答四个递进问题：现有 FiLM 是否实际生效、gamma/beta 哪条路径对输出更敏感、局部扰动方向能否经完整重训保留，以及 validation 候选能否在复用 research/development test 上维持改善方向。

正式证据按以下顺序形成：

1. [W0 CWT-FiLM 调制行为分析](./w0_cwt_film_behavior_results_20260918.md)：在三个冻结 W0 validation checkpoint 上描述实际调制强度、限幅边缘、质量关联和误差关联。
2. [W0 FiLM gamma/beta 局部系数敏感性](./w0_film_local_sensitivity_results_20260918.md)：在同一批冻结 checkpoint 附近进行 gamma-only 与 beta-only inference 扰动，定位优先训练变量。
3. [W0 FiLM gamma 系数训练验证](./w0_film_gamma_training_results_20260919.md)：按冻结训练合同完整重训 `GAMMA_030` 与 `GAMMA_040`，并以 `GAMMA_050` 为既有基线进行 validation-only 判定。
4. [W0 FiLM gamma 系数 research-test 比较](./w0_film_gamma_test_results_20260919.md)：固定 validation-selected checkpoint，仅比较唯一 validation quality candidate `GAMMA_040` 与默认 `GAMMA_050` 在复用 research/development test split 上的表现。

这条证据链从描述性观察逐步收窄到单一训练候选，再进行固定比较；后续阶段不以 test 结果反向改变 checkpoint、候选集合、指标或判定口径。

## 2. 各阶段核心发现

| 阶段 | 数据与操作 | 核心发现 | 对后续决策的含义 |
|---|---|---|---|
| 调制行为 | 3 个冻结 checkpoint；完整 validation；不训练 | `R_total` 窗口中位数约为 0.36–0.39；shift/scale 中位比约为 0.63–0.65，beta/shift 不是可忽略路径。`tau=0.99` 时 gamma、beta 边缘元素均仅约 1%，不支持广泛贴边饱和。调制强度随独立质量元数据变化，但 pooled、分层和 within-samp 相关均只能作描述 | 不以“分支近似恒等”“beta 冗余”或“普遍饱和”为由删减 shift/beta 或全局放宽限幅；质量关联不作因果解释 |
| 局部敏感性 | 冻结 checkpoint；`0.4/0.6` 单路径 inference 扰动；validation-only | gamma 的局部响应整体强于 beta。`GAMMA_040` 在 Whole RR、Local RR、trajectory、`1-PCC` 四轴均为 3/3 seeds 改善；global modulation 方向不稳定。beta 同方向信号更弱且跨 seed、主体和指标的一致性不足 | 只支持开展窄范围 gamma-only 完整训练；不支持删除 beta、宣布 `0.4` 更优或扩展为广泛扫描 |
| 完整重训 | `GAMMA_030/040` 各 3 seeds 完整训练；`GAMMA_050` 复用；validation checkpoint 按 Local RR 选择 | `GAMMA_040` 是冻结矩阵中唯一通过预注册 guardrails 的 quality candidate 和 tolerance-aware Pareto 条件。其 Local RR、trajectory 为 3/3 seeds 改善；PCC 平均下降 0.000547，仍在容差内。`GAMMA_030` 的 PCC 平均下降 0.004097，超过 0.003 guardrail | `GAMMA_040` 获得且仅获得 validation candidate 身份；`GAMMA_030` 不晋级 |
| 固定 test 比较 | 固定 validation-selected checkpoints；3 seeds；复用 research/development test | `GAMMA_040` 相对 `GAMMA_050` 的 paired mean 为 Whole RR **+3.305%**、Local RR **+0.625%**、trajectory **+0.445%**、global modulation **+2.259%**、PCC **-0.002340**；四项 error 均恶化，PCC 也下降。validation 上的平均改善方向未保持，且 seed/subject 异质性明显 | 现有证据不支持用 `GAMMA_040` 替换 `GAMMA_050`；test 不新增 selector，也不触发 test 驱动调参 |

## 3. 最终技术与实验决策

1. `GAMMA_050`（$c_\gamma=0.5,c_\beta=0.5$）继续作为 W0 默认配置。该决定表示当前没有足够证据支持替换默认值，不表示已经证明 0.5 是最优系数。
2. `GAMMA_040` 保留为本冻结训练矩阵中唯一的 **validation quality candidate**，但不提升为新的 W0 默认，也不将其 validation 收益表述为独立泛化收益。
3. `GAMMA_030` 因 PCC guardrail 失败而不晋级；其部分 error 轴收益不足以覆盖预注册的 PCC 风险。
4. shift/beta 路径继续保留。其实际调制幅度、质量关联以及局部响应均不足以支持将其删除或视为无效路径。
5. 冻结本轮固定比较；不基于已复用的 test 结果追加 gamma 系数、扩大扫描、调整指标、改选 checkpoint、回看其他候选或继续调参。

## 4. 可复现入口索引

以下仅列本证据链对应的四个脚本入口及其定向测试：

| 阶段 | 脚本入口 | 对应测试 |
|---|---|---|
| 调制行为分析 | `scripts/analyze_w0_cwt_film_behavior.py` | `tests/test_w0_cwt_film_behavior.py` |
| gamma/beta 局部敏感性 | `scripts/analyze_w0_film_local_sensitivity.py` | `tests/test_w0_film_local_sensitivity.py` |
| gamma 完整训练验证 | `scripts/run_w0_film_gamma_training.py` | `tests/test_w0_film_gamma_training.py` |
| gamma 固定 test 比较 | `scripts/eval_w0_film_gamma_test.py` | `tests/test_w0_film_gamma_test.py` |

具体冻结身份、运行矩阵、产物路径、manifest 哈希、统计定义与完整性核验以第 1 节链接的四份正式结果文档为准。

## 5. 证据边界

- 行为分析、局部敏感性和完整重训选择均基于 validation；窗口存在重叠，validation 仅 7 个 `samp_id`。窗口不能视为独立受试者，seed/主体方向计数也不是独立显著性推断。
- research/development test split 已用于既往研究评价，本轮不是首次触碰的 held-out test，不能证明无偏的独立泛化、部署收益或对新总体的稳健性。
- 调制与质量、误差或 target-modulation 分层之间的关联均可能受到主体构成、任务难度和质量轴混杂影响；它们不证明 FiLM、gamma 或 beta 对性能的因果作用。
- 局部 inference 扰动不等价于完整重训；完整重训比较又同时包含表示学习、优化路径、selected epoch 和 checkpoint 选择的影响，不能把差异单独归因于 gamma 系数。
- checkpoint 由 validation Local RR 选择，其他指标是同一 selected checkpoint 上的伴随描述。复用 test 未定义新的 quality selector，因此其结果不改写 `GAMMA_040` 的既有 validation candidate 身份，也不产生新的 test gate 标签。
- 当前证据既不能证明 0.5 为最优，也不能证明 0.4 或其他系数在独立数据上更优；只能支持维持现有默认、保留已冻结候选身份并停止 test 驱动的继续搜索。
