# 仓库 Agent 工作手册

## 项目定位

这是 THO 呼吸重建科研仓库。涉及训练、指标、数据或结论时先读取 `~/.codex/AGENTS.dl.md`，并以 `docs/experiments/loss_metrics_restart_plan_20260729.md` 为当前唯一实验协议。

## 当前边界

- 当前主线从头建立，不继承旧 loss、metrics、checkpoint 选择、实验 runner 或结论。
- 不在仓库内创建 `archive/`；旧代码、旧配置和旧说明通过 Git 历史恢复。
- 历史 `runs/`、checkpoint、日志、CSV、图表和原始数据不得删除、覆盖或改写。
- 模型注册表与数据基础设施保留；旧阶段以 `patch_mixer1d` 为 baseline，T2–T4 复用冻结的 `time_stft_dual1d` 结构。
- 当前新增 CRD-v1.1 的 S0/S1；CRD_101 触发原 gate 后仅额外激活结果后 S1D 诊断，规范附件为 `docs/experiments/crd_v1_protocol_20260808.md`，由主协议第 35–36 节纳入。D0 paired-final 已排除 selector 主因，当前允许推进 CRD_105；原 CRD_102 暂停。S2 以后、AM/Morphology/gate/auxiliary/control 尚未激活。
- CRD-S0/S1 只使用 train/validation，不读取 research-test；不存在锁定队列和协议修订时不得新增 CRD test 入口。

## 当前入口

- 配置：T2 使用 `configs/tho_research_v2.yaml`；T3/T4 使用对应的 `tho_research_v2_t3_concat.yaml` / `tho_research_v2_t4_bandenergy.yaml`
- 训练：`./.venv/bin/python scripts/train_tho.py --config configs/tho_research_v2.yaml --set training.device=cuda:0`
- 复评：`./.venv/bin/python scripts/eval_tho.py --checkpoint runs/<run>/checkpoint_best_local_rr.pt --split val`
- Research-test：复评命令额外传入 `--split test --confirm-research-test`。该 split 可在阶段性模型整理后重复观察，也可形成后续独立研究问题，但不得用于重选已训练 run 的 epoch/checkpoint；所有结果均属于 development/research evidence，不表述为无偏 held-out 证据。
- CRD 配置：`configs/crd_v1/` 下六个原 S0/S1 配置与一个 `crd_105_direct_coarse` 诊断配置；训练入口 `./.venv/bin/python scripts/train_crd.py --config configs/crd_v1/<variant>.yaml --set training.device=cuda:0`。
- CRD validation 复评：`./.venv/bin/python scripts/eval_crd.py --checkpoint runs/<run>/checkpoint_best_local_rr.pt`；该入口故意不提供 test split。
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

- 定向协议测试：`./.venv/bin/python -m pytest tests/test_respiration_protocol.py tests/test_respiration_metrics.py tests/test_tho_protocol_config.py tests/test_tho_current_experiment.py tests/test_time_stft_fusion.py tests/test_tho_time_frequency_candidates.py tests/test_crd_spectral_ops.py tests/test_crd_models.py tests/test_crd_config.py tests/test_crd_training.py tests/test_crd_experiment.py tests/test_crd_batch_scaling.py`
- 全量当前测试：`./.venv/bin/python -m pytest tests`
- GPU 正式运行必须在沙盒外执行；CPU smoke 只用于实现验收，不形成科研结论。

## Git 与产物

- Git 提交消息使用简洁中文。
- `runs/`、checkpoint、日志和生成图不进入 Git。
- 工作树可能包含用户改动；不得覆盖无关修改。
