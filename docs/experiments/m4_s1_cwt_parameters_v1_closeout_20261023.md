# M4-S1 CWT参数与模块消融收尾索引

整理日期：2026-10-23。实验协议：`m4-s1-cwt-parameters-v1-20261020`；固定checkpoint评价协议：`m4-s1-research-test-v1-20261022`。

**完整28配置×3 seed的84项train/validation、固定84-checkpoint research-test及跨split汇总均已完成并关闭。默认模型保持不变。** 本页为当前状态入口；下列历史设计、验收和执行文档保留各自阶段当时的授权与状态。

## 结果与分析

| 内容 | 入口 |
|---|---|
| 完整validation结果、选点与训练来源 | [validation结果](m4_s1_cwt_parameters_v1_validation_results_20261022.md) |
| 完整research-test、validation对照、seed与受试者差异 | [research-test结果](m4_s1_research_test_results_20261022.md) |
| 28配置、20种CWT表示与预设配对/交互 | [设计合同](m4_s1_cwt_parameters_v1_design_20261020.md) |
| 数据、缓存与GPU合成验收 | [验收结果](m4_s1_cwt_parameters_v1_acceptance_results_20261020.md) |
| 正式训练、JSON绑定兼容与四卡调度 | [正式执行](m4_s1_cwt_parameters_v1_formal_execution_20261020.md)、[兼容修订](m4_s1_cwt_parameters_v1_execution_amendment_20261020.md)、[跨主机执行](m4_s1_cwt_parameters_v1_distributed_execution_20261021.md) |
| 固定84-checkpoint清单、test范围与分析合同 | [research-test专项协议](m4_s1_research_test_v1_protocol_20261022.md) |
| test缓存的来源核验及表示标签映射 | [缓存复用记录](m4_s1_research_test_cache_reuse_20261022.md) |

跨split结果中，BiMamba的Whole RR、全局调制误差与PCC贡献方向最稳定；NoTF的test RR均值改善存在seed分歧，且Local RR净改善主要来自受试者670。池化位置路径与CWT参数呈现跨split差异。完整数值、0.01描述性阈值、受试者贡献及研究开发集的证据边界以结果页为准。

## 源码与运行环境

- `configs/m4_s1_cwt_parameters_v1/design.yaml`：固定矩阵与训练/分析合同。
- `scripts/run_m4_s1_cwt_parameters.py` 及 `scripts/m4_s1_cwt_parameters_*.py`：独立session、模型、数据、缓存、训练及完整矩阵统计。
- `scripts/run_m4_s1_receipt_compatibility.py`：JSON元组/列表绑定规范化；`scripts/run_m4_s1_host.py`：跨主机物理路径适配。
- `scripts/eval_m4_s1_research_test.py`：固定清单、test数据/缓存、逐项评价与跨split汇总。
- [Ubuntu186环境记录](../../configs/hosts/ubuntu186.yaml)及[SSH别名](../../configs/hosts/ssh_config)：双卡训练服务器、项目解释器、同步和回收来源。该远端承担30项train/validation；research-test全部在本机双卡执行。
- 使用对应主机项目内的 `.venv/bin/python`；环境版本与源码身份由冻结session/allowlist核验。

## 产物位置与固定身份

实体产物根为 `/mnt/disk_code/marques/resp_reconstruction/runs/m4_s1_cwt_parameters_v1/`。下表路径相对该根。

| 阶段 | 目录/文件 | 关键SHA-256 |
|---|---|---|
| 训练session | `acceptance_v1_20261020/session.json` | `87b201abb30dea105ca8c9ada5c2f48a33e1f5ebd47b78af11276f67e93740cc` |
| validation汇总 | `acceptance_v1_20261020/summary/attempt_20261022T114304Z_0a0fef94d95b/manifest.json` | `ea6b3a8767f8e1ca8f680ae255fc2aec5369489d57a9c6e5d7f49b7b411628a7` |
| test固定清单 | `research_test_v1_20261022/allowlist.json` | `e02211e4c330644a30324e0ada597059bb14f9289d1cce595c732a4b48c90338` |
| test自动汇总 | `research_test_v1_20261022/summary/attempt_20261022T144637Z_e4476b04e88b/manifest.json` | `f39fee99e3cb628df141f1e98565406860b9dadd1f6dfea882f30f3442827113` |
| test只读核验 | `research_test_closeout_20261022/audit.json` | `a937457b0456dcf5ba83d4ca85c2b1e031f71d8ae279aad86d60ea5fcd817596` |
| test收尾 | `research_test_closeout_20261022/closeout.json` | `95fb7092b2a9c1964c19a30b710f86145fad768d0ff92abf08ba3ea5e133c1a5` |

训练回收与完整性证据另见 `closeout_v1_20261022/`；跨主机分工与恢复来源见 `distributed_v1_20261021/`。test目录保存可读的 `checkpoint_list.csv`、两个worker日志、`pipeline_summary_exit.json`、20个cache阶段和84项evaluation。每项test均为2310窗/8人，五个主指标有效n均为2310。自动汇总包含29张CSV、2张PNG及中文报告。

源码、配置、协议、结果说明和测试纳入Git；checkpoint、缓存、逐窗预测/指标、日志、生成图与原始数据留在产物目录，由manifest及receipt追溯。上述已完成阶段按原身份只读使用；后续新增实验另建独立版本及输出目录。

## 已完成验证

- 基础模型与运行时：80项synthetic CPU定向测试通过，记录见设计合同。
- 主机路径与JSON绑定兼容：6项synthetic CPU定向测试通过，记录见跨主机执行文档。
- 独立research-test入口：6项synthetic CPU定向测试通过，覆盖显式授权、样本顺序、原生CWT形状、非有限值、完整矩阵及统计一致性。
- 全部84项test及20缓存的来源/完整manifest、配置、样本、预测有限性通过；17张test统计表及12张跨split/合并表通过只读数值核验。最终结果页的28配置表、40项模块配对指标和19个CWT候选对照已逐值核对。

本次整理保留冻结源码、协议及结果页的字节身份；Git提交用于保存已完成实验的完整实现和追溯入口。
