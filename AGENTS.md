# 仓库 Agent 工作手册

## 项目定位

这是 THO 呼吸重建科研仓库。涉及训练、指标、数据或结论时先读取 `~/.codex/AGENTS.dl.md`，并以 `docs/experiments/loss_metrics_restart_plan_20260729.md` 为当前唯一实验协议。

## 当前边界

- 当前主线从头建立，不继承旧 loss、metrics、checkpoint 选择、实验 runner 或结论。
- 不在仓库内创建 `archive/`；旧代码、旧配置和旧说明通过 Git 历史恢复。
- 历史 `runs/`、checkpoint、日志、CSV、图表和原始数据不得删除、覆盖或改写。
- 模型注册表与数据基础设施保留；旧阶段以 `patch_mixer1d` 为 baseline，T2–T4 复用冻结的 `time_stft_dual1d` 结构。
- CRD-v1.1 S0/S1/S1C/S1F/S2A/S2B-R、CRD_102 failure/matched-observability diagnostics 与 C0/C1/C2 控制线均已完成并关闭。C0 冻结为 `roundtrip_negligible=true`；C1 固定为 `mamba_retained_control_failure / retain CRD_102`；C2 固定为 `decoder_capacity_supported_100hz_placement_not_supported`，选择 CRD_102 Mamba backbone + `crd_c201_decoder_10hz_cap`。CRD-TF v1 P0–P6a 亦已完成并关闭；P6a 没有 qualified 新候选。C201/M/W/MS × 3 reused research-test 与冻结汇总已完成并关闭；M/W/MS 均 qualified，Pareto=W/MS，Local-RR lead=W，但不构造总分或唯一赢家。P6b/local cross-attention、TCN+decoder、其他 decoder、S3 与旧 gate/auxiliary/final ablation 均关闭。
- CRD-TF-W v2 当前规范为 `docs/experiments/crd_tf_w_v2_protocol_20260817.md`；2026-08-16 附件已失效。P0/P−1/correction 均已完成并冻结，P2 固定为 `retain_film_no_p2_training` 并关闭。P1 三臂实现锁为 `docs/experiments/crd_tf_w_v2_p1_implementation_lock_20260817.json`、SHA-256=`fb822ca7f8607e45443e07a94d91150bcc108217fb25c47f0b4504fbdf644f58`；W1 RESP mask、W2 CARRIER mask、W3 full-band 6V view、六份 config、source/view provenance、stress receipt 与 formal preflight 已实现，`112 passed`，尚未运行 GPU。下一步只允许用户确认后按 W1→W2→W3 依次运行 isolation stress，每臂结果必须先核验；三臂 stress 登记前，P1 formal、P3 实现与 research-test 均关闭。剩余预算固定为 P1 九 runs 加 P3 三 runs，共 12 个新 training runs。暂不做 `samp_id` 分析，Q/concat/D8/gate/attention 与复合候选均关闭。
- CRD 训练阶段仍只使用 train/validation；普通 `eval_crd.py` 保持 validation-only。S1C 的 12 项 access receipt 已齐备，队列关闭，不得重复调用 `eval_crd_s1c.py`；不存在新协议时不得新增或调用其他 CRD test 入口。

## 当前入口

- 配置：T2 使用 `configs/tho_research_v2.yaml`；T3/T4 使用对应的 `tho_research_v2_t3_concat.yaml` / `tho_research_v2_t4_bandenergy.yaml`
- 训练：`./.venv/bin/python scripts/train_tho.py --config configs/tho_research_v2.yaml --set training.device=cuda:0`
- 复评：`./.venv/bin/python scripts/eval_tho.py --checkpoint runs/<run>/checkpoint_best_local_rr.pt --split val`
- Research-test：复评命令额外传入 `--split test --confirm-research-test`。该 split 可在阶段性模型整理后重复观察，也可形成后续独立研究问题，但不得用于重选已训练 run 的 epoch/checkpoint；所有结果均属于 development/research evidence，不表述为无偏 held-out 证据。
- CRD 配置：`configs/crd_v1/` 下保留八个 S0/S1、三个 S2A、四个 S2B-R、一个已关闭 C1 配置与两个已关闭 C2 配置；训练入口 `./.venv/bin/python scripts/train_crd.py --config configs/crd_v1/<variant>.yaml --set training.device=cuda:0` 只作通用 provenance。C2 选择 `crd_c201_decoder_10hz_cap`，现有 formal 不重跑；不得增加其他 decoder 或 TCN+decoder。
- CRD validation 复评：`./.venv/bin/python scripts/eval_crd.py --checkpoint runs/<run>/checkpoint_best_local_rr.pt`；该入口故意不提供 test split。
- CRD S1C research-test 与冻结汇总均已完成；入口和命令只作 provenance，不再重复运行。
- CRD S2A prototype 与冻结汇总均已完成；`scripts/eval_crd_morphology_prototypes.py`、`scripts/summarize_crd_s2a.py` 及其命令只作 provenance，不再重复运行。
- CRD S2B-R 正式训练与冻结汇总均已完成；`scripts/summarize_crd_s2br.py` 及其命令只作 provenance，不再重复运行，冻结产物位于 `runs/crd_v1/crd_s2br_validation_summary/`。
- CRD_102 failure diagnostic 两层均已完成；`scripts/summarize_crd102_failures.py` 与 `scripts/summarize_crd102_failure_metadata.py` 只保留 provenance，不得重复运行。冻结产物位于 `runs/crd_v1/crd_102_failure_diagnostic/` 与 `runs/crd_v1/crd_102_failure_metadata_diagnostic/`。
- CRD_102 matched observability 已完成；`scripts/summarize_crd102_matched_observability.py` 只保留 provenance，不得重复运行，冻结产物位于 `runs/crd_v1/crd_102_matched_observability_diagnostic/`。
- CRD_102 C0 decoder round-trip 已完成并冻结；`scripts/audit_crd102_decoder_roundtrip.py` 只保留 provenance，不得重复运行，固定产物位于 `runs/crd_v1/crd_102_decoder_roundtrip_audit/`。
- CRD C1 formal 与冻结汇总已完成；`scripts/summarize_crd_c1.py` 只保留 provenance，不得重复运行，冻结产物位于 `runs/crd_v1/crd_c1_validation_summary/`。
- CRD C2 formal、C202 residual diagnostics 与冻结汇总已完成；`scripts/eval_crd_c202_residual_spectrum.py`、`scripts/summarize_crd_c2.py` 及其命令只作 provenance，不得重复运行，冻结产物位于 `runs/crd_v1/crd_c2_validation_summary/`。
- CRD-TF v1 P5 已从干净 commit `c7b65b1` 一次性完成，固定产物位于 `runs/crd_tf_v1/p5_validation_summary/`；summary SHA-256 为 `afb6feba1600c9c5e07d713db9cac7c886aa87a033037649d3e9ac32cb753b4e`。P6a 规范为 `docs/experiments/crd_tf_v1_p6a_protocol_20260815.md`；9/9 formal 与冻结汇总已完成，无 incomplete，全部 early stop，summary SHA-256 为 `b970a6ea6d77e6d6858d8b7dbd77ed4c2ed8ff633c7f48eca15aeba227ef6e64`，decision=`no_p6a_candidate_retain_p5_pool`。`scripts/summarize_crd_tf_v1.py` 与 `scripts/summarize_crd_tf_v1_p6a.py` 均只保留 provenance，不得重复运行。
- CRD-TF reused research-test 协议为 `docs/experiments/crd_tf_v1_research_test_protocol_20260816.md`；cache manifest SHA-256=`5d43ecf34596d5a6dd7cbaba75d91f9b7cbbb00214ae7594a4755e2afe510745`，summary SHA-256=`e9430d3449e1e75cbab1804f1c887803ba8c12dcc4b11582f94090a6a1d7c6c0`。`scripts/eval_crd_tf_v1_research_test.py` 与 `scripts/summarize_crd_tf_v1_research_test.py` 均只保留 provenance，不得重复运行；普通 `eval_crd.py` 仍不得用于 test。
- CRD-TF-W v2 P1 stress 使用 `configs/crd_tf_w_v2/*_stress.yaml` 与通用 `scripts/train_crd.py`；必须从干净 commit 串行执行，固定 engineering root 不可覆盖。成功 run 必须包含 `p1_stress_receipt.json`；缺失或不合格时 formal preflight 拒绝运行。当前 stress 尚未获得本阶段执行确认，不得自行启动。
- 数据审计：`scripts/audit_tho_dataset.py`
- Split 审计：`scripts/audit_split_independence.py`
- 详细旧阶段 smoke/batch 128 与 CRD smoke/physical-batch-128 acceptance/正式 seed 命令见 `scripts/README.md`。

## 科研约束

- 数据、split、subject/session 隔离、target 或核心指标口径发生变化前，先说明影响面与旧结论状态。
- 正式实验需保留 resolved config、数据/sample seed、训练 seed、命令、代码版本、checkpoint 和逐 sample metrics。
- 不默认启动正式训练、长时间 GPU 任务或大规模搜索；先完成实现、定向单测和 smoke，由用户确认正式运行。
- 凡可能长时间运行或持续占用计算资源的任务，包括结构测试、全量回归、CPU/GPU smoke、acceptance、benchmark、训练和后台服务，默认均由用户执行；Codex 只提供命令、预期产物与验收口径，并根据用户返回结果继续分析。只有用户在当次任务中明确授权代跑时，Codex 才可执行。
- 修改 loss/metrics/checkpoint 后必须同步协议文档，并运行对应测试。
- 非有限 prediction 不能被静默丢弃；数据泄漏、标签错位、shape 或 split 风险优先报告。

## 当前验证

- 定向协议测试：`./.venv/bin/python -m pytest tests/test_respiration_protocol.py tests/test_respiration_metrics.py tests/test_tho_protocol_config.py tests/test_tho_current_experiment.py tests/test_time_stft_fusion.py tests/test_tho_time_frequency_candidates.py tests/test_crd_spectral_ops.py tests/test_crd_representations.py tests/test_crd_capacity.py tests/test_crd_models.py tests/test_crd_config.py tests/test_crd_training.py tests/test_crd_experiment.py tests/test_crd_batch_scaling.py tests/test_crd_confirmation.py tests/test_crd_s2_selection.py tests/test_crd_s2br_selection.py tests/test_crd_failure_diagnostics.py tests/test_crd_failure_metadata.py tests/test_crd_matched_observability.py tests/test_crd_decoder_roundtrip.py tests/test_crd_c1_selection.py tests/test_crd_decoder_diagnostics.py tests/test_crd_c2_selection.py tests/test_crd_tf_v1_features.py tests/test_crd_tf_v1_cache.py tests/test_crd_tf_w_v2_audit.py tests/test_crd_tf_w_v2_p1.py`
- 全量当前测试：`./.venv/bin/python -m pytest tests`
- GPU 正式运行必须在沙盒外执行；CPU smoke 只用于实现验收，不形成科研结论。

## Git 与产物

- Git 提交消息使用简洁中文。
- `runs/`、checkpoint、日志和生成图不进入 Git。
- 工作树可能包含用户改动；不得覆盖无关修改。
