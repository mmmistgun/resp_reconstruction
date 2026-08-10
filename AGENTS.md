# 仓库 Agent 工作手册

## 项目定位

这是 THO 呼吸重建科研仓库。涉及训练、指标、数据或结论时先读取 `~/.codex/AGENTS.dl.md`，并以 `docs/experiments/loss_metrics_restart_plan_20260729.md` 为当前唯一实验协议。

## 当前边界

- 当前主线从头建立，不继承旧 loss、metrics、checkpoint 选择、实验 runner 或结论。
- 不在仓库内创建 `archive/`；旧代码、旧配置和旧说明通过 Git 历史恢复。
- 历史 `runs/`、checkpoint、日志、CSV、图表和原始数据不得删除、覆盖或改写。
- 模型注册表与数据基础设施保留；旧阶段以 `patch_mixer1d` 为 baseline，T2–T4 复用冻结的 `time_stft_dual1d` 结构。
- CRD-v1.1 S0/S1/S1C/S1F/S2A 已完成；S2A 单分支均失败并保留 candidate-lock 中的 CRD_102。研究者在获知结果后明确激活 result-informed exploratory S2B-R：当前只实现/验收 `205 BASE+E+M / 206 BASE+A+M / 207–208 capacity controls`，不得改写 S2A 或表述为原条件自然触发。S3、gate/auxiliary/TCN/final ablation 继续关闭。规范附件由主协议第 35–42 节纳入。
- CRD 训练阶段仍只使用 train/validation；普通 `eval_crd.py` 保持 validation-only。S1C 的 12 项 access receipt 已齐备，队列关闭，不得重复调用 `eval_crd_s1c.py`；不存在新协议时不得新增或调用其他 CRD test 入口。

## 当前入口

- 配置：T2 使用 `configs/tho_research_v2.yaml`；T3/T4 使用对应的 `tho_research_v2_t3_concat.yaml` / `tho_research_v2_t4_bandenergy.yaml`
- 训练：`./.venv/bin/python scripts/train_tho.py --config configs/tho_research_v2.yaml --set training.device=cuda:0`
- 复评：`./.venv/bin/python scripts/eval_tho.py --checkpoint runs/<run>/checkpoint_best_local_rr.pt --split val`
- Research-test：复评命令额外传入 `--split test --confirm-research-test`。该 split 可在阶段性模型整理后重复观察，也可形成后续独立研究问题，但不得用于重选已训练 run 的 epoch/checkpoint；所有结果均属于 development/research evidence，不表述为无偏 held-out 证据。
- CRD 配置：`configs/crd_v1/` 下保留八个 S0/S1、三个 S2A 配置，并新增 `crd_205_base_em_static / crd_206_base_am_static / crd_207_base_cap_em / crd_208_base_cap_am` 四个 S2B-R 配置；训练入口 `./.venv/bin/python scripts/train_crd.py --config configs/crd_v1/<variant>.yaml --set training.device=cuda:0`。
- CRD validation 复评：`./.venv/bin/python scripts/eval_crd.py --checkpoint runs/<run>/checkpoint_best_local_rr.pt`；该入口故意不提供 test split。
- CRD S1C research-test 与冻结汇总均已完成；入口和命令只作 provenance，不再重复运行。
- CRD S2A prototype 与冻结汇总均已完成；`scripts/eval_crd_morphology_prototypes.py`、`scripts/summarize_crd_s2a.py` 及其命令只作 provenance，不再重复运行。
- 数据审计：`scripts/audit_tho_dataset.py`
- Split 审计：`scripts/audit_split_independence.py`
- 详细旧阶段 smoke/batch 128 与 CRD smoke/physical-batch-128 acceptance/正式 seed 命令见 `scripts/README.md`。

## 科研约束

- 数据、split、subject/session 隔离、target 或核心指标口径发生变化前，先说明影响面与旧结论状态。
- 正式实验需保留 resolved config、数据/sample seed、训练 seed、命令、代码版本、checkpoint 和逐 sample metrics。
- 不默认启动正式训练、长时间 GPU 任务或大规模搜索；先完成实现、定向单测和 smoke，由用户确认正式运行。
- 修改 loss/metrics/checkpoint 后必须同步协议文档，并运行对应测试。
- 非有限 prediction 不能被静默丢弃；数据泄漏、标签错位、shape 或 split 风险优先报告。

## 当前验证

- 定向协议测试：`./.venv/bin/python -m pytest tests/test_respiration_protocol.py tests/test_respiration_metrics.py tests/test_tho_protocol_config.py tests/test_tho_current_experiment.py tests/test_time_stft_fusion.py tests/test_tho_time_frequency_candidates.py tests/test_crd_spectral_ops.py tests/test_crd_representations.py tests/test_crd_capacity.py tests/test_crd_models.py tests/test_crd_config.py tests/test_crd_training.py tests/test_crd_experiment.py tests/test_crd_batch_scaling.py tests/test_crd_confirmation.py tests/test_crd_s2_selection.py`
- 全量当前测试：`./.venv/bin/python -m pytest tests`
- GPU 正式运行必须在沙盒外执行；CPU smoke 只用于实现验收，不形成科研结论。

## Git 与产物

- Git 提交消息使用简洁中文。
- `runs/`、checkpoint、日志和生成图不进入 Git。
- 工作树可能包含用户改动；不得覆盖无关修改。
