# CRD-TF v1 冻结候选池 research-test 协议

日期：2026-08-16

状态：**12-checkpoint evaluation 与一次性冻结汇总已完成；research-test 阶段关闭**

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
| R0 | 协议、cache builder 与 CPU 定向测试 | 已完成 |
| R1 | 完整 2310-window input-only cache | 已完成并冻结 |
| R2 | cache reader、12-checkpoint 受控评价入口与 CPU 测试 | 已完成 |
| R3 | 12 次 GPU research-test evaluation | 已完成 |
| R4 | 一次性审计、paired-seed 汇总与候选比较 | 已完成并冻结 |

固定 12 次 evaluation 已完成且没有缺项。评价入口与 summary 入口现只保留 provenance，不得重复运行或覆盖结果。

## 5. 未来汇总口径

R4 将分别报告五项 primary、IBI/coverage、分层 envelope Spearman、coherence 与 nDTW；primary 按 seed 报 mean、sample SD 和相对 C201 的 paired direction，不构造总分。可以按用户授权将结果用于 research/development 选择，但结论必须附带 test 已被重复使用且受前序研究方向影响的限制。

## 6. R1 cache 冻结结果

完整 cache 由用户从干净 commit `dfd931379c01609fbd154549117b16950f16ec3f` 生成，耗时 `29:43`。Manifest 与 7 个受管文件重新计算 identity 后全部一致：

```text
runs/crd_tf_v1/research_test_cache/40a24df424b2ff9182cfcc6f5b7c12d287578b0df1ed1e25b56b7af1c7f73839/cache_manifest.json
SHA-256 = 5d43ecf34596d5a6dd7cbaba75d91f9b7cbbb00214ae7594a4755e2afe510745
```

Cache 共 `2310 windows / 8 samp_id / 538,086,928 bytes managed files / 514 MiB directory`；row IDs 唯一、严格递增，row-id SHA-256 为 `184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8`。所有数组 shape/dtype/finite 与文件 SHA-256 通过；manifest 固定 `test_target_array_read=false / target_read=false / model_inference_used=false`。

Reader 每个 evaluation 启动时重新校验 manifest SHA、所需数组文件 SHA/size/shape/dtype、row-id content hash，并要求完整 2310 个 dataset rows 一一存在；不得 fallback 为在线计算。

## 7. R2 受控评价入口

唯一入口为：

```text
scripts/eval_crd_tf_v1_research_test.py
```

入口从冻结 C201 candidate lock 与 P5 formal audit 构造精确 12-checkpoint allowlist，逐 checkpoint 校验 path、variant、seed、selected epoch 和 SHA-256。普通 `eval_crd.py` 继续 validation-only。每项成功后在原 run 目录新增且不覆盖：

```text
research_test_metrics.csv
research_test_metrics_summary.csv
research_test_metrics_manifest.json
```

Manifest 记录 checkpoint/cache identity、evaluation commit、完整命令和 reused evidence 属性。任何一项失败时保留已有成功结果并停止对应 shell；修复前不得跳过失败项生成 R4 summary。

## 8. R3/R4 冻结结果

12/12 evaluations 全部来自干净 commit `9f429dae8f4a879c6b9530c1190df1949b6c420a`。每项严格对应冻结 checkpoint path/hash/validation-selected epoch，TF variants 使用冻结 cache；每项均为相同顺序的 2310 个 test rows，五项 primary finite、joint target eligibility=1、prediction degeneracy=0。没有重选 checkpoint。

三 seed research-test mean ± sample SD：

| variant | Whole RR | Local RR | Trajectory | Global envelope | Signed PCC |
|---|---:|---:|---:|---:|---:|
| C201 | 0.702232 ± 0.032241 | 0.663832 ± 0.011575 | 0.142537 ± 0.003638 | 0.171310 ± 0.009071 | 0.876464 ± 0.000996 |
| M | 0.704849 ± 0.023883 | 0.657296 ± 0.014508 | 0.142040 ± 0.003138 | 0.169338 ± 0.005093 | 0.873427 ± 0.003614 |
| W | 0.617234 ± 0.027788 | 0.609566 ± 0.018472 | 0.139546 ± 0.001037 | 0.173418 ± 0.006625 | 0.876577 ± 0.001329 |
| MS | 0.643805 ± 0.067921 | 0.633577 ± 0.011256 | 0.141044 ± 0.002264 | 0.168844 ± 0.011908 | 0.874493 ± 0.001670 |

相对 C201，按预注册 mean 实质门槛 + 至少 2/3 paired seeds 同方向：

- M：Local RR、global envelope 通过；base guardrails 通过；
- W：Whole RR、Local RR、trajectory 通过且均为 3/3；base guardrails 通过；
- MS：Whole RR、Local RR、trajectory、global envelope 通过，均至少 2/3；base guardrails 通过。

W 相对 C201 的 Whole/Local RR mean 改善分别为 `12.10% / 8.17%`，trajectory 改善 `2.10%`，PCC 略高 `0.000113`，但 global envelope 恶化 `1.23%`。MS 的 Whole/Local RR 改善为 `8.32% / 4.56%`，global envelope 改善 `1.44%`，PCC 下降 `0.001971`。W 相对 MS 在 Whole/Local RR/trajectory/PCC 上更好，但 global envelope 更差，因此两者互不容差支配。

Secondary 不支持“W 全指标获胜”：C201 的 coherence、nDTW 和多数 envelope Spearman 均优于 W；W 的 IBI coverage 略高，但 IBI MedAE 略差。这些描述不覆盖 primary/Pareto 规则。

最终三项候选均 qualified；tolerance-aware Pareto 固定为 `W / MS`，M 被容差支配。由于不构造总分，`unique_winner_selected=false`；按既有 Local-RR checkpoint 目标，描述性 lead 为 W，Local RR=`0.609566`。若项目需要一个实际主模型，W 是与当前任务选择目标一致的首选，MS 保留为 envelope trade-off 候选。

冻结产物：

```text
runs/crd_tf_v1/research_test_summary/research_test_summary.json
SHA-256 = e9430d3449e1e75cbab1804f1c887803ba8c12dcc4b11582f94090a6a1d7c6c0

runs/crd_tf_v1/research_test_summary/research_test_summary_manifest.json
SHA-256 = 1c1a4571eaf281a2dbbbd86da633f45e9e44bed2f7d31b7b4320bd38e411e180
```

汇总来自干净 commit `a1ce90c9e82ba044449587c1692b0073f7dce889`，decision=`retain_research_test_pareto_without_total_score`。证据属性继续是 reused research/development evidence。
