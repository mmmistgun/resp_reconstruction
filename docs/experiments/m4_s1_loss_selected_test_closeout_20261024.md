# M4-S1 validation总loss选点实验收尾索引

状态：2026-10-24，完整84项固定checkpoint research-test、自动汇总、逐项核验和选点对照已完成并关闭。82项新增评价，2项复用相同checkpoint的冻结结果。默认模型与训练早停规则保持不变。

## 协议、结果与代码

- [执行协议](m4_s1_loss_selected_research_test_v1_20261024.md)：在原Local RR早停轨迹内按validation总loss最低选点，并列最早；28配置×3 seed。
- [完整结果与逐项对照](m4_s1_loss_selected_research_test_results_20261024.md)：五指标、窗口加权与受试者宏平均、seed/受试者差异及结构效应。
- [方向计数精度修订](m4_s1_loss_selected_test_precision_note_20261024.md)：两侧逐窗CSV使用相同计算路径，使相同checkpoint精确持平；均值与0.01阈值结论保持一致。
- [评价入口](../../scripts/eval_m4_s1_loss_selected_test.py)与[合成选点测试](../../tests/test_m4_s1_loss_selected_test.py)：保留实际执行时冻结的字节身份。
- [核验入口](../../scripts/audit_m4_s1_loss_selected_test.py)：整理自实际通过的 `audit_v2.py`，完整核验主体AST一致；增加命令行路径、显式test访问确认、禁止Python优化模式，以及新输出目录保护。

核验入口读取固定实验的checkpoint来源、缓存、预测与逐窗指标，复算原生表和选点对照；它不是通用实验工具。完整核验已经通过，本次整理仅验证新增入口保护和源码一致性。后续如需重新核验，须有匹配的当次授权，使用项目内全新的 `--output`；禁止覆盖已有收尾目录。调用形式为 `audit_m4_s1_loss_selected_test.py --root TEST_ROOT --output NEW_AUDIT_ROOT --confirm-research-test`，使用项目 `.venv/bin/python`，不得使用 `-O` 或 `PYTHONOPTIMIZE`。

## 冻结证据

以下路径相对于项目根 `/mnt/disk_code/marques/resp_reconstruction`：

| 内容 | 路径 |
|---|---|
| 本轮评价根 | `runs/m4_s1_cwt_parameters_v1/loss_selected_research_test_v1_20261024/` |
| 自动汇总 | 上述根的 `summary/attempt_20261024T091035Z_8046093ea8ab/` |
| 独立核验与收尾 | `runs/m4_s1_cwt_parameters_v1/loss_selected_closeout_20261024/` |
| 原Local RR选点参照 | `runs/m4_s1_cwt_parameters_v1/research_test_v1_20261022/` |

最终方向计数使用收尾目录内 `selector_paired_verified.csv`、`selector_summary_verified.csv` 及对应 `subject_macro_` 表。逐项对照为 `cell_comparison_84.csv`；受试者对照为 `subject_selector_paired.csv` 和 `subject_selector_summary.csv`。原自动表、首次失败审计和全部运行日志均保留，修订明细为 `precision_direction_corrections.csv`。

| 证据 | SHA256 |
|---|---|
| 本轮 `allowlist.json` | `4521211dd5f65116c222897209edda696ae4356733091b138714f0e366eda471` |
| 自动汇总 `manifest.json` | `cb8c8bf3c22e0fa32d5f72b452f304905174292ba6769722f987252fd767466c` |
| 收尾 `audit.json` | `5a4b753ba06116d6b6e96203201825f772da057d514f83cadbb68b5118b2aacf` |
| 收尾 `closeout.json` | `95430ccc76a069e4f23c952380387b2e3d9f26e47c72f11f65e2c417f24f12e3` |

`allowlist.json` 绑定执行时入口、协议、测试和原runtime；`audit.json` 绑定核验来源；`closeout.json` 绑定结果页、精度修订、核验表和PNG图。提交时核对上述绑定，冻结文件保持原字节。

运行辅助源码保留在产物目录，以下身份供追溯；这些脚本已执行，按原路径重跑会触及冻结输出，不能作为新任务入口：

| 脚本 | 所在目录 | SHA256 |
|---|---|---|
| `execute_pipeline.py` | 本轮评价根 | `e644ddedd1d69e7accb5d4f7713ea15bd385faaced4d24a011985e46b1bc4409` |
| `audit_v2.py` | 收尾目录 | `3a4670d288875ef30df5eecacbc1d5be53fcae2b777e63fa312f34bf5ee3d207` |
| `write_results.py` | 收尾目录 | `9ceb9fc37956265155b84bc4411d689ef11182c3e84db74ffd25aa6bee72ed8e` |
| `closeout.py` | 收尾目录 | `7886870b07766aad7a6fa76f023cfcb51dc56364cbc82328b34dd4a18101eafb` |

## 整理验证与证据范围

选点合成测试此前10项通过。本次整理通过Python语法检查、核验主体AST一致性检查，以及synthetic CLI的未确认访问、优化模式和已有输出目录三项拒绝检查；冻结入口/协议/测试及收尾绑定文件的哈希均一致。实际84项评价与完整核验的结论沿用上述冻结回执。

Git保存入口、测试、协议、结果说明与本索引；checkpoint、缓存、预测、CSV、日志和PNG保留在项目产物目录。结果限定于既有训练轨迹及research-test人群；各消融结论必须注明checkpoint selector。
