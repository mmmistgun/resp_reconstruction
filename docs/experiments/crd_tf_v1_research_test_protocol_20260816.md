# CRD-TF v1 冻结候选池 research-test 协议

日期：2026-08-16

状态：**用户已明确授权；R0 协议与 input-only cache builder 已实现，完整 test cache 尚未生成；模型评价尚未开放**

协议标识：`crd-tf-v1-research-test-development-20260816`

## 1. 授权、目的与证据属性

用户在获知 P6a 没有 qualified 新候选后明确表示“不 care 独不独立，开始 research-test”。本协议据此开放已经多次参与研究决策的 test split。结果允许用于当前研究开发、比较与后续选择，但必须始终标记为 **reused research/development evidence**；不得表述为未触碰 held-out、无偏泛化或独立确认。

本阶段不重选任何训练 run 的 epoch/checkpoint，不重训模型，不改变数据 admission、test split、target、loss、指标、逐 sample 聚合或 eligibility。Research-test target 只在模型 inference 后进入冻结评价器；cache builder 只读取 test BCG 输入，不读取 target array。

## 2. 冻结评价矩阵

只评价 P5/P6a 后仍保留的候选池与 C201 anchor：

| 角色 | variant | seeds | checkpoint |
|---|---|---|---|
| anchor | `crd_c201_decoder_10hz_cap` | 20260811/12/13 | candidate lock 中既有 validation-selected checkpoint |
| candidate M | `crd_tf101_m` | 20260811/12/13 | P5 已审计的 validation-selected checkpoint |
| candidate W | `crd_tf102_w` | 20260811/12/13 | P5 已审计的 validation-selected checkpoint |
| candidate MS | `crd_tf203_ms` | 20260811/12/13 | P5 已审计的 validation-selected checkpoint |

共 12 次固定 checkpoint evaluation。P6a 的 MWS-ADD/MWS-GATE、S、其他 P4 arms、C202 和旧模型不进入本矩阵；不得根据前几个 research-test 结果临时增加或删除模型。

## 3. Test input-only cache

M/W/MS 需要固定 M/W/S 表示。Train/validation cache 保持原样且不得加入 test 文件；research-test 使用独立不可覆盖目录：

```text
runs/crd_tf_v1/research_test_cache/<identity>/
```

Cache 固定覆盖完整 2310 个 test windows，使用既有 test sample seed `20260612` 和相同 calibration/fixed transform，只保存：

- `test_m_slow [2310,36,101]`；
- `test_m_fast [2310,44,349]`；
- `test_w [2310,97,360]`；
- `test_s [2310,12,360]`；
- 严格递增的 `test_row_ids` 与 W/S frequency identity。

不生成 L spectrum，因为冻结候选不含 L。Manifest 必须记录 `research_test_used=true / research_test_input_used=true / test_target_array_read=false / target_read=false / test_cache_created=true / model_inference_used=false`，并固定 calibration、candidate lock、P5 summary、dataset index、row IDs、文件 hash/shape/dtype/finite 与干净 Git commit。

## 4. 阶段开关

| 阶段 | 内容 | 状态 |
|---|---|---|
| R0 | 协议、cache builder 与 CPU 定向测试 | 已实现，待提交 |
| R1 | 完整 2310-window input-only cache | 待用户执行 |
| R2 | cache reader、12-checkpoint 受控评价入口与 CPU 测试 | R1 结果冻结后开放 |
| R3 | 12 次 GPU research-test evaluation | 关闭 |
| R4 | 一次性审计、paired-seed 汇总与候选比较 | 关闭 |

当前唯一需要用户执行的长任务是 R1 cache。Cache manifest 返回并审计前，不评价任何 checkpoint，也不读取 test target。

## 5. 未来汇总口径

R4 将分别报告五项 primary、IBI/coverage、分层 envelope Spearman、coherence 与 nDTW；primary 按 seed 报 mean、sample SD 和相对 C201 的 paired direction，不构造总分。可以按用户授权将结果用于 research/development 选择，但结论必须附带 test 已被重复使用且受前序研究方向影响的限制。

