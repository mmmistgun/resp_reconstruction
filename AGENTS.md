# 仓库 Agent 工作手册

## 项目定位

这是 THO 呼吸重建科研仓库。涉及数据、split、loss、metrics、checkpoint、实验执行或结论时，按任务需要应用 `~/.codex/AGENTS.dl.md` 的相关规则。

本文件只保留稳定的仓库边界和协议路由。阶段状态、运行命令、冻结哈希与实验结果以对应协议和产物 manifest 为准，不在此重复维护。

## 协议路由

- CWT-APOR v2 H-only 医学参考呼吸率三区间：本轮 research-test 产物分析已完成并关闭；口径、结果和来源入口为 `docs/experiments/h_only_medical_rr_strata_v1_results_20261005.md`。

- E4 尺度聚合及频带机制检查：当前阶段与证据入口为 `docs/experiments/e4_closeout_20260922.md`；各执行协议与来源锁按该索引定位。
- E5 W0 时域前端替换：三 seed train/validation 与固定 checkpoint research-test 均已完成并关闭；当前状态、validation/test 结果与来源身份统一由 `docs/experiments/e5_temporal_frontend_closeout_20260927.md` 路由。
- E6 20-Hz 学习解调前端：三 seed train/validation、效率记录与固定 checkpoint research-test 均已完成并关闭；当前状态与证据入口为 `docs/experiments/e6_temporal_frontend_closeout_20260924.md`。
- E7 聚合前尺度编码 × 尺度聚合：本轮 validation 与 research-test 均已完成；当前状态、协议、结果和来源锁统一由 `docs/experiments/e7_scale_encoding_aggregation_test_results_20260926.md` 路由。
- W0 三因素结构对照 v1：train/validation 与 research-test 均已完成并关闭；当前状态、结果、协议与冻结身份统一由 `docs/experiments/w0_structural_factorial_v1_closeout_20260925.md` 路由。
- E8 FiLM 条件末端 × 波形解码端：formal、validation 与固定36-checkpoint research-test 均已完成并关闭；最终状态、结果、来源与冻结身份统一由 `docs/experiments/e8_film_decoder_redesign_v1_closeout_20260926.md` 路由。
- E9 潜在宽度 × 条件末端：18-cell formal train/validation 与固定18-checkpoint research-test均已完成并关闭；当前状态、validation/test结果与冻结身份统一由 `docs/experiments/e9_latent_width_condition_refiner_v1_closeout_20260929.md` 路由。
- APOR统一GELU、H64与Direct独立对照：train/validation已完成并关闭，当前状态、结果与来源由 `docs/experiments/apor_gelu_refiner_v1_validation_results_20261001.md` 路由；前序SiLU对照的完成状态与来源见 `docs/experiments/apor_activation_v1_results_20260930.md`。
- APOR统一GELU、H64与Direct固定checkpoint research-test：九项新增评价及完整12-cell汇总已完成并关闭，当前状态、结果与来源由 `docs/experiments/apor_gelu_refiner_v1_research_test_results_20261001.md` 路由。

先确定任务所属实验，再读取对应协议中与当前任务有关的章节；不要默认把某一协议应用到整个仓库。

- THO loss/metrics restart：`docs/experiments/loss_metrics_restart_plan_20260729.md`
- CRD-v1 与控制实验：`docs/experiments/crd_v1_protocol_20260808.md`、`docs/experiments/crd_v1_controls_protocol_20260811.md`
- CRD-TF v1：`docs/experiments/crd_tf_v1_protocol_20260812.md`；P6a 和独立测试集分别使用对应附件
- CRD-TF-W v2：`docs/experiments/crd_tf_w_v2_protocol_20260817.md`
- RTM-v1：`docs/experiments/resp_temporal_v1_protocol_20260820.md`；formal、validation summary 和独立测试集使用对应附件
- 论文证据闭环：`docs/experiments/paper_evidence_closure_protocol_20260901.md`；center-30/60/90、research-test、P5、P6 等任务使用对应专项协议，存在 `supersedes` 或 v2 声明时按协议声明使用
- ADV-v1 时间对齐双视图网络：本轮阶段已关闭；当前状态、执行协议、validation 与 research-test 结果及来源锁统一由 `docs/experiments/aligned_dual_view_v1_closeout_20260922.md` 路由。
- ADV 融合方式 × 位置 v1：本轮train/validation与research-test均已完成并关闭；当前状态、协议、结果与来源锁统一由 `docs/experiments/adv_fusion_factorial_v1_closeout_20260923.md` 路由。
- ADV 节律模块 v1：仅完成 CPU baseline 来源审计，未进入 GPU 验收、smoke、训练、validation 或 research-test，实验已关闭；证据边界、后续独立实验的研究问题覆盖与本地产物入口统一由 `docs/experiments/adv_rhythm_modules_v1_unexecuted_closeout_20260927.md` 路由。

专项协议只在其明确作用域内覆盖总协议。协议发生冲突时，先核对任务身份、协议 ID、状态和日期；仍无法唯一确定时再询问用户。

## 冻结状态与产物

- 是否允许继续某阶段，以相关协议开头的当前状态和冻结决定为准。已完成或关闭的训练、评价、审计、汇总、cache 和 benchmark 不自行重跑。
- 历史 `runs/`、checkpoint、日志、CSV、图表、cache、manifest 和原始数据不得删除、覆盖、补写或用相同 identity 重跑。
- 不在仓库内创建 `archive/`；旧代码、配置和说明通过 Git 历史恢复，淘汰原因写入协议或决策记录。
- 新实验必须使用新协议或明确修订，并使用独立、不可覆盖的输出 identity。失败 lifecycle 和失败产物也必须保留。
- RTM-v1 独立测试集评价已冻结，但测试结论锁尚未由用户确认；不得据此重评、重选或改写论文冻结表。只读代码、文档和 provenance 审计不因此暂停。

### 实验锁使用原则

- 实验锁不是默认步骤；只有确需跨运行冻结实验身份或受控证据边界，且 Git commit、配置、manifest 与 receipt 不足以满足复现和审计要求时，才建立实验锁。
- 每个实验版本原则上只建立一个正式锁，并由工程验收、正式运行和汇总共同复用；阶段状态与结果使用 manifest、receipt 或 closeout 记录。
- 科学合同实质变化时优先使用增量修订；新增独立证据边界时可建立专项锁。并发互斥锁只用于运行控制，不作为科研证据锁。

## 数据与科研边界

- 数据、split/fold、subject/session 隔离、target、loss、核心指标、聚合方式或 checkpoint selector 发生变化前，说明影响范围以及旧结果需要重算、重评、重训还是退出当前结论。
- 普通 CRD 训练和 `scripts/eval_crd.py` 只允许 train/validation。任何独立测试集访问必须由匹配的专项协议和当次用户授权共同开放；一个任务的 test 授权不外推到其他任务。
- `scripts/eval_tho.py --split test --confirm-research-test` 只适用于 THO 协议允许的独立测试集评价，不构成 CRD、RTM 或论文附件的通用 test 入口。
- 非有限 input、target、prediction、checkpoint 或关键指标必须显式失败，不得过滤后继续或静默缩小样本集合。
- 正式实验按协议保存 resolved config、数据与 sample seed、训练 seed、命令、代码版本、checkpoint、完整 history、逐 sample metrics、运行环境和 access/provenance 信息。
- 不根据 partial seed、中间 test 结果或单个示例修改剩余矩阵、候选、阈值、指标或 checkpoint 选择。

## 执行与验证

- Codex 可以直接运行与改动相关的静态检查、语法检查和快速 CPU 定向测试，前提是只使用 synthetic/disposable fixture、不访问真实数据或独立测试集，且预期数分钟内完成。
- 正式训练、GPU 任务、真实数据 smoke、全量 cache、全量回归、acceptance、benchmark、后台服务和独立测试集访问默认由用户执行；只有用户在当次任务中明确授权时，Codex 才可代跑。
- 默认只运行能验证本次改动的最小测试集合。受影响测试通过后，除非出现失败、跨模块影响或未解决风险，不扩大或重复测试。
- 修改 loss、metrics 或 checkpoint 逻辑时，同步更新对应协议并运行允许范围内的定向测试；需要长时或 GPU 验证时，提供命令、预期产物和验收标准。
- 入口、历史命令和完整测试命令见 `scripts/README.md`。数据与 split 审计入口分别为 `scripts/audit_tho_dataset.py` 和 `scripts/audit_split_independence.py`。

## Git 与产物

- Git 提交消息使用简洁中文。
- `runs/`、checkpoint、日志和生成图不进入 Git。
- 工作树可能包含用户改动；保留无关修改，不覆盖或回退用户内容。
