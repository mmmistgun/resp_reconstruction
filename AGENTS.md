# 仓库 Agent 工作手册

## 项目定位

这是 THO 呼吸重建科研仓库。涉及训练、指标、数据或结论时先读取 `~/.codex/AGENTS.dl.md`，并以 `docs/experiments/loss_metrics_restart_plan_20260729.md` 为当前唯一实验协议。

## 当前边界

- 当前主线从头建立，不继承旧 loss、metrics、checkpoint 选择、实验 runner 或结论。
- 不在仓库内创建 `archive/`；旧代码、旧配置和旧说明通过 Git 历史恢复。
- 历史 `runs/`、checkpoint、日志、CSV、图表和原始数据不得删除、覆盖或改写。
- 模型注册表与数据基础设施保留；纯时域 `patch_mixer1d` 是协议 baseline，当前候选为精简的 `time_stft_fusion1d`。
- 若未来要恢复历史比较、桥接验证、分层统计或新模型路线，必须作为独立任务重新定义，不提前保留兼容分支。

## 当前入口

- 配置：`configs/tho_research_v2.yaml`
- 训练：`./.venv/bin/python scripts/train_tho.py --config configs/tho_research_v2.yaml --set training.device=cuda:0`
- 复评：`./.venv/bin/python scripts/eval_tho.py --checkpoint runs/<run>/checkpoint_best_local_rr.pt --split val`
- Designated test：复评命令额外传入 `--split test --confirm-designated-test`；不得用 test 反向调协议或选模。
- 数据审计：`scripts/audit_tho_dataset.py`
- Split 审计：`scripts/audit_split_independence.py`
- 详细 smoke、batch 128 验收和正式 seed 命令见 `scripts/README.md`。

## 科研约束

- 数据、split、subject/session 隔离、target 或核心指标口径发生变化前，先说明影响面与旧结论状态。
- 正式实验需保留 resolved config、数据/sample seed、训练 seed、命令、代码版本、checkpoint 和逐 sample metrics。
- 不默认启动正式训练、长时间 GPU 任务或大规模搜索；先完成实现、定向单测和 smoke，由用户确认正式运行。
- 修改 loss/metrics/checkpoint 后必须同步协议文档，并运行对应测试。
- 非有限 prediction 不能被静默丢弃；数据泄漏、标签错位、shape 或 split 风险优先报告。

## 当前验证

- 定向协议测试：`./.venv/bin/python -m pytest tests/test_respiration_protocol.py tests/test_respiration_metrics.py tests/test_tho_protocol_config.py tests/test_tho_current_experiment.py tests/test_time_stft_fusion.py`
- 全量当前测试：`./.venv/bin/python -m pytest tests`
- GPU 正式运行必须在沙盒外执行；CPU smoke 只用于实现验收，不形成科研结论。

## Git 与产物

- Git 提交消息使用简洁中文。
- `runs/`、checkpoint、日志和生成图不进入 Git。
- 工作树可能包含用户改动；不得覆盖无关修改。
