# APOR v2：固定checkpoint research-test附件

日期：2026-09-30。协议ID：`apor-grid-decoder-v2-research-test-20260930`。

状态：用户当次明确要求“继续后面的实验，比如test”。本附件开放已完成v2矩阵的research-test，复用N0/D0六份冻结结果，新增N1/D1六次评价，并按预设三种人群视图完整汇总。完成状态以新allowlist、evaluation及summary回执为准。

## 1. 固定来源

- 训练session：`runs/apor_grid_decoder_v2/session_20260930T093614Z_511f2ff06f0f`。
- Session SHA256：`0c54b200006030b5a4c692f5a3b386d8e9caf551a1b3bf0415768ed334340386`。
- Validation summary：该session下`summary/attempt_20260930T111411Z_d03d6929f568`，manifest SHA256=`650fb26bfbed456e867ea66ea1cd146cc07e5577b39d810a77591e76c81bdcc8`。
- 新评价：N1 selected epoch=8/9/22，D1=29/19/10；seeds=20260811/20260812/20260813。只使用已固定的`checkpoint_best_local_rr.pt`。
- 参照复用：N0=A0，D0=A3，来自`runs/patch_apor_v1/research_test/allowlist_20260930T071732Z_9b85989bdc0b`；checkpoint hash和epoch须与v2冻结参照完全一致，原test manifest与文件逐一核验。
- Validation已选择N0作为后续激活规范化的U0。该决定及三个checkpoint在当前test访问前已冻结；本test不改变结构或epoch选择。

## 2. 样本与视图

完整test固定2310窗口、8个samp_id，sample seed20260612，180 s/100 Hz。有序row ID SHA256=`184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8`。沿用历史input-only W缓存与原数据索引身份，逐batch核验input/target/W finite、shape、顺序与development受试者隔离。

固定报告以下视图，不按部分结果改变人群：

| 视图 | 窗口数 | samp_id数 | 用途 |
|---|---:|---:|---|
| full | 2310 | 8 | 完整开发性test评价 |
| exclude670 | 2231 | 7 | 已声明的病例敏感性分析 |
| subject670 | 79 | 1 | 单病例描述 |

排除670只在全部窗口推理、指标完整核验后派生统计，不改变推理样本或过滤失败。用户提供670为重度OSA伴一定体动的背景，本附件不由模型误差推断临床状态。剩余7人不命名为正常人群。

## 3. 推理、指标与统计

严格加载固定模型state dict，eval模式、batch128、BF16、完整末尾batch。核心五指标、Pi、资格与失败规则沿用v2训练及原test实现；启用原生IBI/coverage、coherence、constrained nDTW和分层包络Spearman，并报告各自分母。

每种视图均保存三seed mean/sample SD、四个预设配对、原始单位交互`(D1−D0)−(N1−N0)`、subject-macro、逐受试者、Local RR尾部、资格和辅助指标分母。单病例的macro等于该病例窗口均值，不构成人群重复。复用六个参照加六个新评价形成完整12-cell，共27720行完整窗口指标，其中新增13860行。

Research-test属于reused research/development evidence。当前结果用于固定矩阵的开发性复核，不作为新的独立确认，也不回选checkpoint或修改后续激活比较定义。

## 4. 生命周期与执行

入口：`scripts/eval_apor_grid_decoder_v2_research_test.py`。新脚本、附件、测试及全部依赖在allowlist中保存快照；历史训练、协议和test文件不修改。Allowlist准备在test数组访问前绑定六个新checkpoint、六个参照、validation决定、数据及缓存身份。

已完成cell经校验后复用；失败与中断保留，显式`--retry-failed`创建新attempt。两卡按固定3/3分片评价，各CPU线程4；任何worker失败停止另一路，完整12-cell之后才生成三视图汇总。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. ./.venv/bin/python -m pytest tests/test_apor_grid_decoder_v2_research_test.py -q
./.venv/bin/python scripts/eval_apor_grid_decoder_v2_research_test.py prepare-allowlist
```

使用返回的实际allowlist路径：

```bash
APOR_V2_TEST='/实际allowlist路径'
env -u LD_LIBRARY_PATH -u LD_PRELOAD PYTHONUNBUFFERED=1 \
  ./.venv/bin/python scripts/eval_apor_grid_decoder_v2_research_test.py parallel \
  --allowlist "$APOR_V2_TEST" --devices cuda:0 cuda:1 --confirm-research-test
```

全部完成后保留一次性汇总，结果另建记录。本附件不触发重新训练，激活规范化采用其自身来源与运行身份。

实现验证：15项synthetic CPU定向检查通过（36.34 s），覆盖六个新checkpoint合同、参照禁止重新推理、原生test指标、三种固定视图的分母/交互、有限值与顺序、成功复用、失败保留及双卡汇总门禁。
