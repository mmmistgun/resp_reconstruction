# E9：固定18-checkpoint research-test协议

日期：2026-09-29。协议ID：`e9-latent-width-condition-refiner-v1-research-test-20260929`。

状态：**执行协议与代码已固定；用户已明确授权本轮research-test。必须先生成并冻结18-entry allowlist，再评价完整矩阵，最后一次性汇总。**

## 1. 证据角色

本阶段评价E9 validation完成前预注册的全部`6 arms × 3 seeds = 18`个validation-selected checkpoint。当前test split已经参与既往研究开发，因此证据角色固定为 **reused research/development evidence**，用于描述validation结构方向能否跨split保持，不构成新的独立held-out确认。

Test不参与：

- checkpoint、epoch、arm或seed选择；
- 材料性阈值、指标、eligibility或结构修改；
- 根据partial结果改变剩余队列；
- 追加新的D/H候选。

## 2. 冻结validation来源

- Validation summary：`runs/e9_latent_width_condition_refiner_v1/validation_summary/summary_8bb3b1992f64_20260929T144128Z_f1242dd035e9`。
- Summary manifest SHA-256：`93ae65b2a19c3b76bb3fb6656ce4190d26e994abb0028e810a9c1c2e1a29c81f`。
- Validation decision SHA-256：`ed7159d43894acea366beac74bdc1c056636bde02a3131f861fa353e85b690a6`。
- Training implementation lock：`8bb3b1992f64fc90959d3c16116e5ee9ac6463131ac01bb0f2473c068298367b`。
- Formal runtime amendment：`1c8b58249d7173743128f8c8b603c5ca643d9cfabf97f5e521b207d51fb79566`。

Allowlist逐项固定checkpoint、config、history、formal manifest/receipt、selected epoch、development subject集合和文件SHA-256。Allowlist阶段只读取validation产物、checkpoint字节与test cache manifest，不读取test arrays或dataset index内容。

## 3. Test合同

- Split=`test`，2,310 windows，8个`samp_id`，sample seed=`20260612`。
- 冻结W cache：97 scales × 360 time bins；row identity与frequency identity沿用现有CRD-TF research-test cache lock。
- Batch=128、BF16、完整尾batch、无shuffle；模型结构与checkpoint state保持不变。
- 每个checkpoint只执行input-only inference，随后按冻结任务算子计算test metrics。
- 五主指标、eligibility、Local RR、PCC、trajectory与global modulation口径和validation一致；同时保存test-only IBI、coherence、nDTW与分层envelope Spearman。
- 非有限input、target、W、prediction、checkpoint或关键指标显式失败。
- 18项全部完成后才开放一次性summary；summary只读取冻结metrics，不再次加载checkpoint或test arrays。

## 4. 阶段门控

### P6a：Allowlist

```bash
PYTHONPATH=. ./.venv/bin/python \
  scripts/run_e9_latent_width_condition_refiner_v1_test.py prepare-allowlist
```

Allowlist成功回执必须明确：`test_array_read=false`、`dataset_index_read=false`、`model_inference_used=false`。

### P6b：完整评价

每个cell必须引用同一allowlist并显式传入`--confirm-research-test`。每项使用独立不可覆盖lifecycle；已完成cell拒绝重跑，失败现场保留。

### P6c：一次性汇总

只有`completed=18 / pending=running=failed=0`后执行：

```bash
PYTHONPATH=. ./.venv/bin/python \
  scripts/run_e9_latent_width_condition_refiner_v1_test.py summary \
  --allowlist <ALLOWLIST_DIR> \
  --confirm-research-test
```

## 5. 解释重点

Test汇总沿用error相对0.5%、PCC绝对0.002材料性阈值，重点回答：

1. Validation中`D96-H64−H65`的五属性Pareto改善是否保持；
2. H48相对H64的形态/调制交换是否保持；
3. D64-direct的节律—容量工作点是否保持；
4. D64-H64的trajectory/global modulation优势与RR/PCC交换是否保持；
5. D64相对D96-H64的质量保护线是否仍未满足。

无论test结果如何，全部18项均进入汇总，不按test单项最好值重选结构，也不把reused split表述为独立确认。

