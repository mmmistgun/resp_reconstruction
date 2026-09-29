# ADV 节律模块 v1 未执行实验关闭记录

日期：2026-09-27。状态：**仅完成 CPU baseline 来源审计；未进入 GPU 工程验收、真实数据 smoke、formal 训练、validation 汇总或 research-test，实验关闭。**

协议 ID：`adv-rhythm-modules-v1-es30p15-train-val-20260923`。

本文件是 ADV 节律模块 v1 的唯一当前状态入口。它只整理已经存在的只读证据，不恢复原执行入口，不补写历史产物，也不赋予任何训练、评价或测试集访问权限。若以后重新研究节律模块，必须建立新的协议和输出 identity；不得沿用本轮身份继续运行。

## 1. 关闭决定与结论边界

1. 本轮没有产生候选模型 checkpoint、训练 history、逐窗口 validation metrics、跨 seed summary 或 research-test 结果。
2. 不存在 M_A、M_B、M_C、M_AB、M_AC、M_BC 之间的效果比较，也不存在相对 ADV 融合实验 D 基线的质量结论。
3. 唯一完成动作是 `audit-baseline`：只读核对三个历史 D seed 的来源与初始化身份。该动作不是模型训练、模型评价或科学结果。
4. 本轮不得进入论文结果表、模型选择、机制归因、消融结论或后续候选晋级证据。历史协议中的矩阵、阈值和分析计划均属于未执行设计。
5. 关闭不改变 ADV-v1 与 ADV 融合因子实验已经冻结的结果；其当前入口仍分别为 [ADV-v1 收尾索引](aligned_dual_view_v1_closeout_20260922.md) 和 [ADV 融合因子实验收尾索引](adv_fusion_factorial_v1_closeout_20260923.md)。

## 2. 原计划范围，仅作设计历史

原协议计划以 ADV 融合因子实验的 D 配置为历史基线，考察三个节律模块及其两两组合：

- A：替换波形前端的 patch 编码模块；
- B：CWT 投影后、FiLM 条件映射前的时间卷积模块；
- C：FiLM 后、线性读出前的时间卷积模块；
- 六个候选为 M_A、M_B、M_C、M_AB、M_AC、M_BC；每个候选原计划运行 seeds `20260811/20260812/20260813`，共 18 项新训练；
- 历史 D 三项原计划只读接入，形成 21 项汇总；不训练 M0，不包含 ABC；
- 原计划只开放 train/validation，未开放 research-test。

以上内容没有进入执行阶段，不应按完成矩阵、失败矩阵或负结果解释。

## 3. 实际完成的 baseline 审计

唯一产物目录为：

`runs/adv_rhythm_modules_v1/baseline_audit_cpu_20260923_r1`

完成回执记录 run ID 为 `baseline_audit-e47f150852b5-cd6a9811adb24b108c460167a95867a2`，结束时间为 `2026-09-23T04:14:02.459874+00:00`。审计核对了 ADV 融合因子实验 D 的三个 seed 来源：

| Seed | 历史来源 | 初始化核对 |
|---|---|---|
| 20260811 | `runs/adv_fusion_factorial_v1/D_seed20260811_formal_es30p15_20260922_r1` | 通过 |
| 20260812 | `runs/adv_fusion_factorial_v1/D_seed20260812_formal_es30p15_20260922_r1` | 通过 |
| 20260813 | `runs/adv_fusion_factorial_v1/D_seed20260813_formal_es30p15_20260922_r1` | 通过 |

审计 manifest 明确记录：

- `raw_data_accessed=false`；
- `checkpoint_content_loaded=false`；
- `native_forward_executed=false`；
- baseline lock SHA-256 为 `0d6b7a51a850d4836ec3ec674826a028e1c4bb359214c34d9357844cf01b39f7`。

因此，本次完成状态只表示 baseline provenance 检查成功，不表示模型、数据管线、GPU 路径或训练合同通过验收。

## 4. 来源身份与现存材料

- 开发基点：`97029137bacc482f2ff23f81c120f79fb7c81ca8`，该提交已属于当前 `main` 历史。
- 历史协议声明的开发分支：`codex/adv-rhythm-modules-v1`；该分支当前不存在。
- 历史协议声明的工作树：`/mnt/disk_code/marques/resp_reconstruction/.worktrees/adv_rhythm_modules_v1`；该工作树当前不存在。
- 审计启动时，节律模块配置、协议、实现、入口和测试均为未跟踪文件；它们不属于开发基点提交。
- 产物中的 source snapshot 仍保存 19 个实验专属文件：8 个配置/锁文件、9 个 `resp_rhythm` 包文件、1 份协议和 1 个执行入口。它们仅用于历史审计，不恢复为当前可运行源码。
- 启动记录还列出 `adv_rhythm_modules_v1_design_20260923.md`、`adv_rhythm_modules_v1_commands_20260923.md` 和 `tests/test_adv_rhythm_modules_v1.py` 三份未跟踪材料；这三份文件不在当前仓库，也未进入现存 source snapshot，不能声称本轮设计或测试材料完整可复现。

| 证据 | SHA-256 |
|---|---|
| `started.json` | `eb43afb4540e976ec86c093be01c057e250ff911956a90b97e1447e9b7885e9c` |
| `config.yaml` | `3928c40b37b3269caf89a1bf7c74a2263d785392b2079686038d9d4be9469365` |
| `manifest.json` | `27daedf79ee2d50676899d43e78441430091ca48046ef896d635c096e8f4d436` |
| `completed.json` | `a6e9dbfe209a3120ab7ac22fc3da261e2e414d1fc233efd2e49221f5a957b359` |
| 历史协议快照 | `0ba85d95138de94123f980f7fe37f986a89fb42ecd9ede57bb57ccf029be8a66` |
| source snapshot 聚合身份 | `e47f150852b5425a5beaae7a7c2050c4663227eb0618aed04c7c7d02cc09a372` |

上述哈希按关闭时的原位文件计算或由原始启动记录保存。`runs/` 不进入 Git；对应目录、回执与 source snapshot 必须原位保留，不得删除、覆盖、补写或使用相同 identity 重跑。

## 5. 后续独立实验的研究问题覆盖

ADV 节律模块 v1 本身没有执行，但其计划考察的三个结构问题后来由独立协议、独立实现和独立输出 identity 实际研究。下表只说明科学问题与结构位置的对应关系，不表示这些实验继承了本轮 provenance，也不把后续结果回填为本轮结果。

| 原计划问题 | 后续独立实验 | 已完成证据 | 关系边界 |
|---|---|---|---|
| A：波形前端的 patch/卷积表示 | [E5 时域前端](e5_temporal_frontend_closeout_20260927.md)、[E6 20-Hz 学习解调前端](e6_temporal_frontend_closeout_20260924.md)、[W0 三因素结构对照](w0_structural_factorial_v1_closeout_20260925.md) | E5/E6 各完成三 seed validation 与 reused research-test；W0 完成 `PATCH/CONV20` 因子 | 模块宽度、采样、归一化、基线与训练合同均不完全相同 |
| B：CWT 条件路径中的时序/尺度表示 | [W0 三因素结构对照](w0_structural_factorial_v1_closeout_20260925.md)、[E7 尺度编码 × 聚合](e7_scale_encoding_aggregation_test_results_20260926.md) | W0 完成 `TM3/TM0`；E7 完成六臂三 seed validation 与 18-checkpoint reused research-test | E7 扩展的是尺度编码与聚合问题，不等同于原 B 模块的直接复现 |
| C：FiLM 后时间细化与解码 | [W0 三因素结构对照](w0_structural_factorial_v1_closeout_20260925.md)、[E8 条件末端 × decoder](e8_film_decoder_redesign_v1_closeout_20260926.md) | W0 完成 `REF2/REF0`；E8 完成 36 项 validation 与 36-checkpoint reused research-test | E8 使用更完整的条件末端与 decoder 矩阵，实验身份独立 |
| A/B/C 条件依赖与交互 | [W0 三因素结构对照](w0_structural_factorial_v1_closeout_20260925.md) | 完整 `2×2×2`、24 项训练与 24 项 reused research-test | W0 为 D96 完整结构合同，不是原 D64 ADV rhythm 矩阵的续跑 |

其中 W0 是最完整的结构性覆盖：它同时检验 `PATCH/CONV20`、`TM3/TM0`、`REF2/REF0` 及二阶、三阶交互。结果保留完整 `PATCH/TM3/REF2`，同时确认结构作用依赖上下文且存在 RR 与形态/PCC 的属性取舍。E5/E6、E7、E8 又分别扩展了前端、CWT 条件表示和 FiLM/decoder 问题。后续证据已经覆盖原计划的科学动机，因此关闭本轮不会留下必须按旧身份补跑的实验缺口。

上述关系是基于结构位置、对照因素和科学问题的对应整理；现有 provenance 没有声明 W0 或其他实验是 `adv-rhythm-modules-v1` 的改名、分支合并或直接续作。

## 6. 阶段状态

| 阶段 | 状态 | 可形成的证据 |
|---|---|---|
| CPU baseline 来源审计 | 已完成 | 仅 baseline 来源和初始化身份 |
| CPU synthetic 模型/训练验收 | 未形成可核验完成回执 | 无 |
| GPU synthetic 工程验收 | 未执行 | 无 |
| 真实数据 cache 或 smoke | 未执行 | 无 |
| Formal train/validation | 未执行 | 无 |
| Validation summary | 未执行 | 无 |
| Research-test | 未开放、未执行 | 无 |

至此，ADV 节律模块 v1 以“未执行实验”关闭。任何后续工作都应视为新的研究阶段，不得把本记录转换为旧协议的续跑授权。
