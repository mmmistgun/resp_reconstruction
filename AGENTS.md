# 仓库 Agent 工作手册

## 使用方式

- 通用协作偏好遵循全局记忆；本文只记录本仓库特有规则、入口、约束和验证方式。
- 本仓库属于深度学习科研代码；涉及数据探索、模型训练、消融、baseline、评价指标或结果表述时，先读取并遵循 `~/.codex/AGENTS.dl.md`。
- 长实验背景、阶段结论和命令细节按需读取 `scripts/README.md` 与 `docs/experiments/*.md`。

## 项目速览

- 这是呼吸重建/THO 相关深度学习科研实验仓库。
- 默认围绕 `configs/`、`resp_train/`、`scripts/`、`tests/`、`docs/experiments/` 和 `runs/` 推进训练、评估、诊断与实验记录。
- 当前训练主线以 `configs/tho_research_v2.yaml`、`scripts/README.md` 和 `docs/experiments/time_frequency_input_fusion_plan.md` 的最新记录为准；旧 rawish state-aligned 结果只能作为候选来源，不能直接和 20260620 soft-z 口径横向比较。
- 后续 agent 需要先理解数据口径、指标口径、split 关系和已有 run 证据，再决定是否改代码或继续训练。

## 科研仓库例外

- 本仓库不适用“始终最小必要改动”的保守产品维护策略。面对科研假设验证、训练流程、数据适配、模型结构和评估口径时，可以接受大规模重构、目录调整、接口变化和不兼容变更。
- 判断标准从“少改”改为“更有利于形成可信证据”：能减少实验歧义、清理错误抽象、统一数据/指标口径、降低后续推理成本的改动，应优先考虑根治性方案。
- 不需要为历史临时脚本、失败实验分支或早期探索接口保持兼容；但要保留足够信息，让旧结果的配置、指标和结论仍可追溯。
- 若改动会改变数据集构造、split 逻辑、核心评价指标或既有结论解释，先停下说明影响面和旧结论是否需要重算，再继续。
- 对实验性代码，允许先做清晰的破坏性改造，再补验证；对已经作为当前主线的训练入口、汇总脚本和实验文档，改动后必须同步验证和记录。

## 项目规则与约束

- 需要调用 GPU 的命令应使用提权执行，让命令在沙盒外访问 GPU；不要为了绕过沙盒而改用 CPU 跑正式实验或验证。
- Git 提交消息使用中文，保持简洁明确。
- 每次正式实验完成后，及时保存 git 状态；不要把多个实验结果、指标解释和代码变化长期混在一个未提交工作区里。
- 允许更新长期文档来反映稳定结论；探索期的临时猜测、一次性失败记录和未复核判断不要写入长期文档。
- 实验实现、runner/manifest、结果汇总脚本完成后，不能只交付命令和 CSV；汇总结论、通过/停止判断、后续动作必须回写到对应 `docs/experiments/*.md` 或 `findings.md`，并在不确定时标注证据缺口。
- 引入新第三方依赖前仍需说明理由并确认，除非它只用于本地一次性分析且不会进入仓库依赖或主流程。
- 以下属于本仓库高后悔成本操作，必须先说明风险并得到明确指令：改动原始数据、历史实验结果、checkpoint、日志、图表；改变 split、subject/session 隔离、标签定义或核心指标口径；启动长训练、大规模搜索/下载；安装或升级影响复现的依赖。
- 训练、评价或诊断流程不得绕过数据泄漏、标签/时间对齐、shape、split 和指标合理性检查；发现风险先报告。
- 多因素对照 run 必须能追溯配置和配对关系；同一计算图的对照 run 可以复用 `time_only` substrate，但 manifest 中必须记录复用关系。
- 解释模型结果前，优先检查 run 目录中的 `config.yaml`、`audit.csv`、`baseline_metrics.csv`、`train_history.csv`、`metrics.csv` 和诊断图。
- 默认不要替用户启动正式全量训练；优先准备实现、单测、smoke、manifest、dry-run 和可复制命令，待用户确认后由用户或按明确指令提权运行。

## 实验策略

- 对结构性问题优先根治，例如训练入口职责混乱、指标含义漂移、配置覆盖链条不清、脚本与训练工厂口径不一致。
- 正式对照实验应固定验证 seed 和全量验证窗口，避免验证集变化掩盖模型、loss 或输入差异。
- F-A target STFT loss 这轮全量 pilot 由用户手动运行；agent 负责准备代码、测试、manifest、dry-run 和汇总命令。

## 运行环境

- 默认解释器：`./.venv/bin/python`。
- 配置系统：`resp_train/config.py` 使用 OmegaConf，脚本常用 `--set key=value` 做 dotlist 覆盖。
- 主配置：`configs/tho_small.yaml` 用于小规模胸带训练和 smoke；`configs/tho_research_v2.yaml` 用于当前 research v2 主线；`configs/e4_*.yaml` 用于 E4/SST 相关实验。

## 代码结构

- `resp_train/data/`：数据索引、research v2 数据集、缓存、审计、split 独立性和 DataLoader 工厂。
- `resp_train/experiments/`：`BaseExperiment` 管通用实验生命周期，`ThoExperiment` 及其变体管 THO 任务逻辑。
- `resp_train/models/`：模型注册表、1D 时序模型、低频/分解模型和 STFT 分支。
- `resp_train/losses/weak.py`：弱同步相关 loss 组合，是当前研究假设的重要入口。
- `resp_train/metrics/`：baseline、信号处理和评价指标；指标口径变更会影响历史结论解释。
- `scripts/`：训练、评价、诊断、manifest 和批量实验入口；脚本仍保持平铺，职责说明见 `scripts/README.md`。
- `tests/`：配置、数据、loss、模型注册、脚本 overrides、指标和实验流程回归测试。

## 常用入口

- 全量测试：`./.venv/bin/python -m pytest tests`。
- 配置覆盖语法：`--set data.train_sample_seed=1 --set training.batch_size=128`。
- 数据审计：`./.venv/bin/python scripts/audit_tho_dataset.py --config configs/tho_research_v2.yaml --output /tmp/tho_audit.csv`。
- Split 独立性审计：`./.venv/bin/python scripts/audit_split_independence.py --config configs/tho_research_v2.yaml --output-dir runs/audits/split_independence_research_v2`。
- 单 run 训练原子入口：`./.venv/bin/python scripts/train_tho_small.py --config configs/tho_research_v2.yaml --set ...`。
- 从 checkpoint 复评：`./.venv/bin/python scripts/eval_tho_small.py --checkpoint runs/<root>/<timestamp>/checkpoint.pt --metrics-output /tmp/tho_metrics.csv`。
- 预测图诊断：`./.venv/bin/python scripts/plot_tho_predictions.py --run-dir runs/<root>/<timestamp> --sort-by rr_peak_band_abs_error --max-plots 8`。
- 多 run 汇总：`./.venv/bin/python scripts/summarize_tho_runs.py --runs-root runs/<root> --output /tmp/tho_runs_summary.csv`。
- 当前批量实验、manifest 和 E3/E4/E5 入口不要从记忆中猜，先读 `scripts/README.md` 的对应小节。

## 验证方式

- 单元/回归测试优先使用 `./.venv/bin/python -m pytest tests`。
- 修改配置解析、数据工厂、loss、metrics、模型注册或脚本 override 后，至少运行对应 `tests/test_*.py`；影响共享路径时再跑全量 `tests`。
- 修改训练入口或实验生命周期后，补跑最小 smoke：小窗口、1 epoch、低通道数，并检查 run 产物是否完整。
- 修改评价指标、mask、RR 口径或 split 逻辑后，必须说明旧 run 是否需要重算，并同步更新相关实验文档。
- 正式实验验证要保留 resolved `config.yaml`、manifest、summary 和诊断图；单个 summary 表不能替代波形复核。
- 正式实验汇总后，相关计划文档应记录运行范围、manifest/summary 路径、主指标结论、分层结论、是否进入下一阶段；若结论不足以推进，也要明确停止或补跑条件。

## 详细背景索引

- `scripts/README.md`：脚本职责、推荐 THO workflow、当前 soft-z/research v2 训练口径、批量实验和诊断命令。涉及运行命令时先读。
- `docs/tho_small_training.md`：早期小规模 THO 训练设计、产物说明、评价口径和历史阶段结论。维护旧小规模入口时读取。
- `docs/experiments/time_frequency_input_fusion_plan.md`：时频输入、多分支融合、E1-E5 路线和当前建议。修改 STFT/fusion/E3/E4/E5 相关代码或结论时读取。
- `docs/experiments/e1_stft_info_gain_20260622.md`：E1 STFT 信息增益、疑点核查和频域输入探索收口。复核 STFT 是否提供信息增益时读取。
- `docs/experiments/softz_20260620_model_candidates.md`：20260620 soft-z 模型候选重跑。比较 soft-z 模型候选时读取。
- `docs/experiments/rawish_state_aligned_l0_l1.md`：rawish state-aligned 历史 L/M/N 系列实验。只在追溯旧结果或迁移旧模型想法时读取。
- `docs/experiments/tho_research_v2_performance.md`：research v2 性能对比和坏例观察。解释 historical research v2 结果时读取。
- `docs/experiments/f_series_stft_loss_plan.md`：F 系列 STFT 约束与输出空间草稿。重启 STFT loss/输出空间路线时读取。
- `~/.codex/AGENTS.dl.md`：深度学习科研任务规范。涉及训练、消融、baseline、数据探索、指标口径、复现或结果表述时读取。

## 待确认事项

- 当前没有 `findings.md`。若后续需要沉淀跨阶段稳定结论，优先新建 `findings.md` 并在本文件只保留读取条件。
- 若 `scripts/` 迁移到子目录结构，需要同步更新本文件、`scripts/README.md`、测试和常用命令。
