# W0 FiLM 局部剂量敏感性方案

协议 ID：w0-film-local-sensitivity-v1-20260918。

状态：结果知情的 validation-only 探索方案；设计、独立实现和 synthetic CPU 定向验证已完成，implementation lock 待生成。真实 GPU 推理由用户执行。

## 1. 动机与证据边界

本方案承接 w0-cwt-film-behavior-v1-20260918 的完成结果：

- W0 的 R_total 窗口中位数约 0.36–0.39，条件分支并非近似恒等；
- shift 强度约为 scale 的 63%–65%，不支持直接删除 beta；
- tau=0.99 的 g/b 边缘比例约 1%，不支持广泛饱和；
- pooled 中更强调制通常伴随较低误差；
- 但在 high target-envelope modulation 层，g/b 的 tau=0.95 边缘比例与 Local RR、trajectory 和 1-PCC 转为正相关。

因此下一步不启动训练，而先回答：在已训练 W0 附近，小幅增强或减弱 gamma/beta 路径，输出质量对哪条路径、哪个任务层更敏感？high target-modulation 中的边缘—误差同向关系是一般全局规律，还是只在困难层出现的局部响应？

本阶段是结果知情、探索性、冻结 checkpoint 的局部干预。它不能证明重新训练时某个系数更优，也不能把 inference perturbation 表述为结构消融或因果机制定论。

## 2. 冻结对象

复用三个 W0 crd_tf102_w validation-selected checkpoint：

| Seed | Selected epoch | 冻结 run |
|---|---:|---|
| 20260811 | 13 | runs/crd_tf_v1/formal/crd_tf102_w/seed_20260811/20260812_210725_400861 |
| 20260812 | 15 | runs/crd_tf_v1/formal/crd_tf102_w/seed_20260812/20260812_223203_048130 |
| 20260813 | 14 | runs/crd_tf_v1/formal/crd_tf102_w/seed_20260813/20260812_235740_452006 |

只使用完整 validation：2675 窗口、7 samp_id、sample seed=20260611。沿用冻结输入、W cache、预处理、batch=128、BF16 AMP、原生任务投影与五项主指标。不得访问 test，不训练，不修改 checkpoint/cache。

FULL 直接复用冻结 metrics.csv，不重复完整 native inference。新执行只运行四个局部干预；FULL 的当前代码/环境能力由真实 batch-1 smoke 和前序 FiLM 行为分析的 exact hook anchor 共同约束。

## 3. 固定干预矩阵

设原始条件头输出 gamma_raw、beta_raw，冻结 W0 为：

$$
g=0.5\tanh(\gamma_{\mathrm{raw}}),\qquad
b=0.5\tanh(\beta_{\mathrm{raw}}).
$$

局部干预固定为：

| ID | gamma coefficient | beta coefficient | 新推理 |
|---|---:|---:|---:|
| FULL | 0.5 | 0.5 | 0，复用冻结 metrics |
| GAMMA_040 | 0.4 | 0.5 | 3 seed |
| GAMMA_060 | 0.6 | 0.5 | 3 seed |
| BETA_040 | 0.5 | 0.4 | 3 seed |
| BETA_060 | 0.5 | 0.6 | 3 seed |

共 12 次新 validation inference、32100 条新逐窗口 metrics。0.4/0.6 是围绕训练值 0.5 的对称 ±20% 局部扰动，不声称为候选最优值。

gamma=0.6 时实际乘法增益范围仍为 [0.4,1.6]，保持为正；beta 有界于 [-0.6,0.6]。一次只改变一条路径，另一条固定为 0.5。首轮不增加 gamma×beta factorial、按窗口自适应系数、raw clipping、按结果选阈值或任意系数 CLI。

既有 P−1 的 GAMMA_ONLY/BETA_ONLY/CONDITION_OFF 是极端路径干预；本轮回答基线附近的局部敏感性，不重复零路径。

## 4. 非侵入与实现要求

FULL 必须调用原生模型 forward。synthetic CPU fixture 和用户真实 batch-1 smoke 另构造 identity wrapper（0.5/0.5），要求与原生 waveform/waveform_10hz 逐 tensor一致；GPU 最大绝对差超过 1e-6 失败。

四个干预路径只复用原生：

    Z = model.base.encode_local(x)
    gamma_raw, beta_raw = model.branches["w"]({"w": W})
    Z_prime = Z * (1 + c_gamma * tanh(gamma_raw)) + c_beta * tanh(beta_raw)
    output = model.base.decode_local(Z_prime)

不得在 decoder 前计算额外统计；不得修改 tf_v1_model.py；不得使用 strict=False。每个 checkpoint strict-load，model.eval()、torch.inference_mode()、BF16 AMP。

## 5. 固定评价轴

逐窗口继续使用冻结五轴：

- whole_rr_abs_error_bpm；
- local_rr_mae_bpm；
- envelope_trajectory_mae；
- global_envelope_modulation_error；
- 1-lag_aware_signed_pcc。

每个 candidate 与同 seed、同 dataset_row_id 的 FULL 配对。统一使用 error-aligned 值，正 delta 表示 candidate 更差：

$$
\Delta_{i,m}=E_{i,m}^{candidate}-E_{i,m}^{FULL}.
$$

对四个 error 另报相对变化；FULL=0 时相对变化为空，不加 epsilon。PCC 轴使用 1-PCC，因此原始等价形式为 FULL PCC−candidate PCC。

不构造加权总分、p-value 或单一赢家。

## 6. 预定义分层与统计

### 6.1 主层级

每 seed、condition、metric 报告：

- FULL/candidate mean；
- paired delta mean、median、P05/P25/P75/P95；
- worse/equal/better 数；
- 四项 error 的 paired relative delta；
- 全部 target eligibility 和 prediction degeneracy 分母。

三 seed 报告各自估计及 arithmetic mean/sample SD。

### 6.2 结果知情的主要分层

本轮主要解释层为冻结 envelope_target_stratum：

- low；
- medium；
- high。

high stratum 是本次结果知情动机，但必须与 low/medium 完整并列，不能只发布 high。

### 6.3 独立质量与主体敏感性

同时报告：

- waveform_confidence_level：实际 high/medium；
- waveform_confidence_score、transient_motion_ratio 的连续分层描述；
- 每个 samp_id 的 paired delta；
- high target stratum 内按 confidence level 的交叉描述；样本数不足时只保存计数，不推断。

窗口重叠，不能把窗口当独立受试者。首版不计算置信区间；若后续需要，只按 samp_id 重采样并说明 n=7。

## 7. 局部响应量

对 gamma 或 beta 路径、每 seed/metric/stratum，令 error-aligned 聚合值为 M(c)，固定计算：

$$
S_{\mathrm{central}}=\frac{M(0.6)-M(0.4)}{0.2},
$$

$$
C_{\mathrm{local}}=M(0.6)-2M(0.5)+M(0.4).
$$

S_central 正值表示提高该路径系数局部趋向恶化，负值表示局部趋向改善。C_local 只描述三点非线性，不称为二阶导数或最优性证明。

另分别保存 down/up 相对 FULL 的 paired delta，避免中心斜率掩盖单边非对称。

## 8. 决策口径

本轮不自动选择训练系数。完成后按以下顺序解释：

1. 先看 FULL 复用身份、四干预完整性、finite、eligibility 与 degeneracy；
2. 再看 high/medium/low 三层的方向是否一致；
3. 再看三个 seed 和 samp_id 内方向；
4. 最后才讨论是否值得新增训练实验。

只有当同一路径在 high stratum 中 down/up 呈可解释的局部方向、至少 2/3 seed 同向，并且其他 strata/质量组没有明显相反代价时，才可提出训练系数敏感性候选。即使满足，也必须另立训练协议。

若 high 层改善但 low/medium 恶化，优先考虑条件自适应或困难层机制研究，不把一个全局系数宣布为更优。若所有局部扰动影响很小，则边缘比例更可能是困难度标记，而不是当前性能瓶颈。

## 9. 实现、产物与运行顺序

已实现：

- 核心/运行：resp_train/paper_evidence/w0_film_local_sensitivity.py
- CLI：scripts/analyze_w0_film_local_sensitivity.py
- tests：tests/test_w0_film_local_sensitivity.py
- implementation lock：docs/experiments/w0_film_local_sensitivity_implementation_lock_20260918.json
- 输出根：/mnt/disk_code/marques/resp_reconstruction/runs/w0_film_local_sensitivity_v1/

阶段：

1. prepare-lock：锁定 W0 来源、冻结 metrics、前序 FiLM summary/结果依据及代码；
2. smoke：用户执行真实 batch-1 native/identity wrapper 一致性；
3. analyze：用户按 seed 执行四个干预；
4. summarize：CPU 汇总完整 3×5 矩阵；
5. finalize：中文结论、manifest 与 freeze receipt。

每次 attempt 排他创建；失败现场保留。正式入口不提供 test、任意 checkpoint、任意 coefficient 或覆盖输出参数。

最小产物：

- 每 seed 的 condition_metrics.csv、paired_window_deltas.csv、condition_summary.csv；
- 完整汇总 paired_window_deltas.csv、stratum_summary.csv、seed_summary.csv、local_response.csv；
- quality_subject_summary.csv、matrix_receipt.json；
- coefficient_response.png、high_stratum_response.png、quality_response.png；
- conclusions_zh.md、artifact_manifest.json、freeze_receipt.json。

## 10. 验收

Codex 只运行 synthetic/disposable CPU 测试：

- 0.5/0.5 identity wrapper 与原生输出相等；
- 四个条件公式、shape、finite 与实际增益正值；
- condition/seed/row identity 完整；
- error-aligned delta、相对变化、分层与局部响应公式；
- 缺 condition/seed、eligibility 漂移、degeneracy、非有限、重复 row 显式失败；
- 排他 lifecycle、manifest/hash。

用户运行真实 GPU：

- batch-1 smoke；
- 三个 seed 的完整 validation inference。

所有新推理完成前不查看部分 seed 后删减矩阵或改变系数。

当前 synthetic CPU 与原 W0 模型联合验证为 34 passed。implementation lock 生成并提交后，固定命令为：

~~~bash
FILM_WT=/mnt/disk_code/marques/resp_reconstruction_w0_cwt_film
FILM_PY=/mnt/disk_code/marques/resp_reconstruction/.venv/bin/python

cd "$FILM_WT"

env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  "$FILM_PY" -m pytest \
  tests/test_w0_film_local_sensitivity.py \
  tests/test_w0_cwt_film_behavior.py \
  tests/test_crd_tf_v1_model.py -q

env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  "$FILM_PY" scripts/analyze_w0_film_local_sensitivity.py prepare-lock

# 用户：真实 validation batch-1 一致性。
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  "$FILM_PY" scripts/analyze_w0_film_local_sensitivity.py smoke --device cuda:0

# 用户：smoke 通过后，固定三个 seed、每 seed 四个新条件。
FILM_LOCAL_SMOKE=/完成的smoke_attempt目录
for FILM_SEED in 20260811 20260812 20260813; do
  env -u LD_LIBRARY_PATH -u LD_PRELOAD \
    "$FILM_PY" scripts/analyze_w0_film_local_sensitivity.py analyze \
    --seed "$FILM_SEED" --device cuda:0 \
    --smoke-receipt "$FILM_LOCAL_SMOKE" || break
done

env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  "$FILM_PY" scripts/analyze_w0_film_local_sensitivity.py summarize \
  --runs /三个完成analysis_attempt目录
~~~

## 11. 证据边界

- 结果知情的 exploratory validation evidence；
- 不是重新训练消融；
- 不是 test 泛化证据；
- 不证明 0.4/0.5/0.6 中任一训练系数最优；
- high target-modulation 的分层动机来自前序观察，必须明确披露；
- test 或新增训练另行讨论。
