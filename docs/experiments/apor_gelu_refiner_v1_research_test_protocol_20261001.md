# APOR GELU/H64/Direct：固定checkpoint research-test附件

日期：2026-10-01（Asia/Shanghai）。协议ID：`apor-gelu-refiner-v1-research-test-20261001`。

用户当次明确要求“开启test”。本附件开放已完成GELU/H64/Direct九个固定checkpoint的research-test，复用B0三份原A0评价。它接续[validation完成记录](apor_gelu_refiner_v1_validation_results_20261001.md)，在本次test访问范围内替代训练阶段“test尚未开放”的状态；冻结训练源码、配置及科学合同保持原身份。

## 1. 固定来源与矩阵

- Training session：`runs/apor_gelu_refiner_v1/session_20260930T154903Z_6fec6cef5eff`，SHA256=`3c349e55f8f77c306607bfe19a3734afde3ce1380b6c5635679953fd2ac587b1`。
- Validation summary：session下`summary/attempt_20260930T175426Z_6500442256e9`，manifest SHA256=`37381d5a81b80989aabec54e854e5f6e343665797c1be2dd43d9d08d9c3aa7ed`。
- 完整validation决定为`eligible_arms=[B0]`、`test_used=false`，其中已经绑定全部12个checkpoint。Test评价全部九个新增候选，不因保护线是否通过或partial test结果裁剪矩阵。
- Seeds为20260811/20260812/20260813；GELU best epoch=10/13/11，H64=8/13/14，DIRECT=14/13/61。均由完整validation Local RR最小、并列最早选择。
- B0复用`runs/patch_apor_v1/research_test/allowlist_20260930T071732Z_9b85989bdc0b`的A0三份成功test，checkpoint与当前B0参照逐项一致，epoch=8/13/13。

Test不改选checkpoint，不改变validation决定或后续候选定义。B0不重新训练或推理。入口在冻结训练文件集合之外，通过独立allowlist绑定新评价实现与全部来源。

## 2. 样本、指标及证据边界

沿用既有research-test：2310窗口、8个samp_id，180 s、100 Hz，sample seed20260612。有序row ID SHA256=`184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8`。原数据索引、input-only W cache、Pi、指标定义与target资格保持不变。

固定报告三个视图：full=2310窗口/8人，exclude670=2231窗口/7人，subject670=79窗口/1人。先对完整样本推理并核验，再派生子集统计；子集不作为新增独立样本。Test与train/validation受试者必须隔离。

新增九项评价共20790行窗口metrics；加入B0三份复用结果，完整12-cell科学矩阵27720行。保存五项主指标、原生IBI与coverage、coherence、constrained nDTW、分层包络Spearman、资格及失败分母。预测退化保留并报告，非有限input/target/W/prediction/checkpoint或关键指标显式失败。

汇总保存三seedmean/sample SD、GELU/H64/DIRECT各自相对B0的配对差及改善seed数、逐受试者、subject-macro、RR尾部和分母。三个seed描述优化随机性；该test已用于既往研究开发，证据角色为`reused research/development evidence`，不作为新的独立人群确认。

## 3. 执行与验收

推理为strict checkpoint加载、eval、batch128、BF16，完整末尾batch。GPU型号与训练来源一致，依赖环境和代码快照必须通过身份检查。原生训练GPU验收已覆盖四臂，此阶段不重跑训练或benchmark。

入口`scripts/eval_apor_gelu_refiner_v1_research_test.py`。准备allowlist先验证完整validation、九个新增checkpoint、B0来源以及冻结样本/cache manifest；该步骤不打开test数组。双GPU按固定矩阵交错拆分5/4，worker独立进程组、每进程CPU线程4，全部成功后统一汇总。

```bash
./.venv/bin/python scripts/eval_apor_gelu_refiner_v1_research_test.py prepare-allowlist \
  --session runs/apor_gelu_refiner_v1/session_20260930T154903Z_6fec6cef5eff
```

使用返回的独立allowlist目录：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD PYTHONUNBUFFERED=1 \
  ./.venv/bin/python scripts/eval_apor_gelu_refiner_v1_research_test.py parallel \
  --allowlist /返回的allowlist目录 --devices cuda:0 cuda:1 --confirm-research-test
```

本次由Codex按用户授权在持久后台任务中执行，避免交互终端网络断开影响父调度器。成功产物校验后复用；失败、中断现场保留，显式重试使用新attempt。输出根为`runs/apor_gelu_refiner_v1/research_test/allowlist_<UTC>_<uuid>`，保存源码/来源、逐cell配置与指标、访问记录、manifest/freeze及完整三视图summary。

验收要求：9/9新增评价成功、两worker退出码0、完整12-cell三视图汇总；样本与target资格跨cell一致，固定checkpoint、validation决定和原始产物身份不漂移。失败时停止后续汇总，不过滤失败样本或替换checkpoint。

实现验证：19项synthetic CPU定向测试通过（36.10 s），覆盖固定九checkpoint矩阵、validation来源完整性、三种模型到原生test指标的数据流、样本/finite门禁、三视图分母、checkpoint selector、失败保留、CLI访问确认及并行完成门禁。CPU模型fixture用Identity替代Mamba；本轮实际推理仍使用训练时的原生Mamba。
