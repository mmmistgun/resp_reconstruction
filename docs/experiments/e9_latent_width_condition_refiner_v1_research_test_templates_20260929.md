# E9：research-test allowlist 与汇总模板

状态：**模板；当前不授权 test array 访问、推理或指标计算。** 只有18/18 train/validation、一次性 validation 汇总、checkpoint selector 和结构决策冻结后，才能生成 allowlist。评价仍需用户当次明确授权。

## 1. Allowlist 头部

| 字段 | 必填值 |
|---|---|
| Protocol | `<new E9 research-test protocol id>` |
| Evidence role | `reused research/development evidence` |
| Validation summary path/manifest/decision SHA | `<frozen>` |
| Implementation lock path/SHA | `<frozen>` |
| Matrix | `6 arms × 3 seeds = 18 checkpoints` |
| Test used for selection | `false` |
| Additional width/refiner candidates after test | `forbidden` |

## 2. 18-entry allowlist schema

每个 `(arm,seed)` 恰有一项：

| 字段 | 内容 |
|---|---|
| `arm`, `seed` | 属于冻结18-cell矩阵 |
| `selected_epoch` | 完整 validation Local RR 严格最小、并列最早 |
| `checkpoint_path`, `size_bytes`, `sha256` | validation-selected checkpoint 字节身份 |
| `config_path`, `sha256` | 保存配置及 E9 model contract |
| `history_path`, `sha256` | selected epoch 可回放 |
| `formal_manifest`, `formal_receipt`, `freeze_receipt` | formal lifecycle 身份 |
| `initialization_state_sha256` | from-scratch 初始化身份 |
| `development_subjects` | train/validation subject 集合 |
| `test_cache_manifest_identity` | 只读 manifest 身份，不含 test array 访问 |

Allowlist 生成回执必须为：

```text
validation_artifacts_read = true
checkpoint_bytes_read = true
test_cache_manifest_read = true
test_array_read = false
dataset_index_content_read = false
model_inference_used = false
```

## 3. Test 执行门控

- allowlist path/SHA 固定并回载成功；
- 用户当次明确授权，并显式传入 `--confirm-research-test`；
- 一次性评价18项完整矩阵，partial 结果不改变队列；
- 每个 cell 使用不可覆盖 attempt，失败现场保留；
- test 不重选 epoch/checkpoint/arm，不改阈值、指标或结构。

## 4. Test 汇总模板

### 五主指标

| Arm | Whole RR | Local RR | Trajectory | Global modulation | PCC |
|---|---:|---:|---:|---:|---:|
| `<six E9 arms; three-seed mean ± sample SD>` |  |  |  |  |  |

### 计划对比与 validation→test

| Contrast | Test Whole RR | Test Local RR | Test Trajectory | Test Global modulation | Test PCC | Validation classification | Test classification | Direction retained? |
|---|---:|---:|---:|---:|---:|---|---|---|
| `H64−H65` |  |  |  |  |  |  |  |  |
| `H48−H65` |  |  |  |  |  |  |  |  |
| `H48−H64` |  |  |  |  |  |  |  |  |
| `D64-H64−D64-direct` |  |  |  |  |  |  |  |  |
| `D64-H48−D64-direct` |  |  |  |  |  |  |  |  |
| `D64-H48−D64-H64` |  |  |  |  |  |  |  |  |

并列报告 window-direct、subject-macro、Local RR median/P90/P95、`>2/>5 bpm`、五主指标分母、prediction degeneracy、paired-seed方向，以及完整 frozen E8 pointwise 历史表。跨宽度表继续同时携带参数与 covered MAC。

## 5. 证据边界措辞

> E9 research-test 评价使用在既往研究开发中已被访问的固定 split，因此结果只描述 validation 结构方向在该 reused split 上是否保持。它不构成独立 held-out 或外部确认，不用于追加 H/D 候选、重选 checkpoint 或把 test 单项最好结构升级为主模型。

若 validation 与 test 反转，应把反转本身作为主要风险报告；不得选择只支持偏好结构的 split 或指标。
