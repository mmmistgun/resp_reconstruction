# 仓库 Agent 工作手册

## 项目定位

这是 THO 呼吸重建科研仓库。涉及训练、指标、数据或结论时先读取 `~/.codex/AGENTS.dl.md`，并以 `docs/experiments/loss_metrics_restart_plan_20260729.md` 为当前唯一实验协议。

## 当前边界

- 当前主线从头建立，不继承旧 loss、metrics、checkpoint 选择、实验 runner 或结论。
- 不在仓库内创建 `archive/`；旧代码、旧配置和旧说明通过 Git 历史恢复。
- 历史 `runs/`、checkpoint、日志、CSV、图表和原始数据不得删除、覆盖或改写。
- 模型注册表与数据基础设施保留；旧阶段以 `patch_mixer1d` 为 baseline，T2–T4 复用冻结的 `time_stft_dual1d` 结构。
- CRD-v1.1 S0/S1/S1C/S1F/S2A/S2B-R、CRD_102 failure/matched-observability diagnostics 与 C0/C1/C2 控制线均已完成并关闭。C0 冻结为 `roundtrip_negligible=true`；C1 固定为 `mamba_retained_control_failure / retain CRD_102`；C2 固定为 `decoder_capacity_supported_100hz_placement_not_supported`，选择 CRD_102 Mamba backbone + `crd_c201_decoder_10hz_cap`。CRD-TF v1 P0–P6a 亦已完成并关闭；P6a 没有 qualified 新候选，冻结候选池仍为 `crd_tf101_m / crd_tf102_w / crd_tf203_ms`，且不选择唯一赢家；证据仅属于 validation-development。P6b/local cross-attention、research-test、TCN+decoder、其他 decoder、S3 与旧 gate/auxiliary/final ablation 均关闭。
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

- 定向协议测试：`./.venv/bin/python -m pytest tests/test_respiration_protocol.py tests/test_respiration_metrics.py tests/test_tho_protocol_config.py tests/test_tho_current_experiment.py tests/test_time_stft_fusion.py tests/test_tho_time_frequency_candidates.py tests/test_crd_spectral_ops.py tests/test_crd_representations.py tests/test_crd_capacity.py tests/test_crd_models.py tests/test_crd_config.py tests/test_crd_training.py tests/test_crd_experiment.py tests/test_crd_batch_scaling.py tests/test_crd_confirmation.py tests/test_crd_s2_selection.py tests/test_crd_s2br_selection.py tests/test_crd_failure_diagnostics.py tests/test_crd_failure_metadata.py tests/test_crd_matched_observability.py tests/test_crd_decoder_roundtrip.py tests/test_crd_c1_selection.py tests/test_crd_decoder_diagnostics.py tests/test_crd_c2_selection.py tests/test_crd_tf_v1_features.py tests/test_crd_tf_v1_cache.py`
- 全量当前测试：`./.venv/bin/python -m pytest tests`
- GPU 正式运行必须在沙盒外执行；CPU smoke 只用于实现验收，不形成科研结论。

## Git 与产物

- Git 提交消息使用简洁中文。
- `runs/`、checkpoint、日志和生成图不进入 Git。
- 工作树可能包含用户改动；不得覆盖无关修改。
