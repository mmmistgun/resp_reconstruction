# W0 FiLM gamma 系数 research-test 评价附件

协议 ID：`w0-film-gamma-test-v1-20260919`。

状态：固定 checkpoint test 评价协议、实现和 synthetic CPU 验证已完成；专项锁待生成，真实 test cache 数组、test 波形和 target 尚未读取。

## 1. 授权、目的与证据属性

承接 `w0-film-gamma-training-v1-20260918` 的冻结 validation 决策，仅评价唯一 quality candidate `GAMMA_040` 与同 seed `GAMMA_050` W0 基线。用户本次授权编写 test 相关协议、代码、身份锁与 synthetic CPU 测试；本阶段不执行真实 test 推理，也不新建或重建 cache。

本项目的 test split 已用于既往 CRD-TF 研究评价，因此本附件的证据定位为：**固定 validation-selected checkpoints 在复用 research/development test split 上的比较**。不得表述为首次触碰的 held-out、无偏独立确认或部署泛化证明。

test 结果不用于重选训练 epoch、追加系数、回看 GAMMA_030、修改模型结构或改变评价指标。

## 2. 固定 checkpoint 矩阵

| Seed | GAMMA_040 selected epoch | GAMMA_050 selected epoch |
|---|---:|---:|
| 20260811 | 13 | 13 |
| 20260812 | 30 | 15 |
| 20260813 | 14 | 14 |

GAMMA_040 来源固定为：

```text
runs/w0_film_gamma_training_v1/formal/gamma040/seed_20260811/20260918_235034_257608
runs/w0_film_gamma_training_v1/formal/gamma040/seed_20260812/20260919_011752_936734
runs/w0_film_gamma_training_v1/formal/gamma040/seed_20260813/20260919_024514_370431
```

三者均完成 80 epochs / 6400 updates，并从完整 validation 按 Local RR 最小值选择 checkpoint。训练 summary 固定为：

```text
runs/w0_film_gamma_training_v1/summary/20260919_115215_255816
```

- summary manifest SHA-256：`2df739a0a1a0d987a9c4733107230a5f74662ae78dd92c481d3bca92feae1b7b`；
- training implementation lock SHA-256：`2bf4b72a15111e1caa76b6bed19abbde3242edc451795176c307d160cc49368c`；
- training commit：`0f9686ccbc64a8b4ee9578b96eae88f5b3bdf4eb`。

GAMMA_050 复用既有 W0 三个 validation-selected checkpoint 及其冻结 research-test metrics，不重复推理。新执行固定为 3 次 GAMMA_040 evaluation，共 6930 条新逐窗口记录。

## 3. Test 数据与 cache 合同

- 完整 test：2310 windows、8 个 `samp_id`；test sample seed=`20260612`；
- 冻结 input-only cache：`runs/crd_tf_v1/research_test_cache/40a24df424b2ff9182cfcc6f5b7c12d287578b0df1ed1e25b56b7af1c7f73839/`；
- cache manifest SHA-256：`5d43ecf34596d5a6dd7cbaba75d91f9b7cbbb00214ae7594a4755e2afe510745`；
- test row-order SHA-256：`184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8`；
- W tensor shape=`[97,360]`，batch=128，BF16 AMP，eval mode，shuffle=false；
- 完整尾 batch 为 6，不过滤、不补齐、不缩小样本集合。

prepare-lock 只读取冻结 manifest、训练产物、既有 W0 test 指标和 dataset metadata，不加载或解码 `test_w.npy`，不读取 test 波形/target。正式 evaluate 启动后才核验 cache 数组字节身份并登记 `access_started`。

每次正式评价必须：

1. 先核验 checkpoint/config/history/receipt、cache manifest、dataset index 和代码锁；
2. 严格回放 `c_gamma=0.4,c_beta=0.5` resolved training config；
3. `strict=True` 加载 checkpoint，selected epoch 必须匹配 allowlist；
4. 核验 test 与该 seed 的 train/validation `samp_id` 无交叉；
5. 逐 batch 检查 row identity/order、W shape 与 input/target/cache finite；
6. 使用原生 `evaluate_task_predictions(..., include_test_only=False)`。

## 4. 指标与配对统计

五主指标固定为：

- Whole RR absolute error；
- Local RR MAE；
- envelope trajectory MAE；
- global envelope modulation error；
- lag-aware signed PCC。

逐 seed 与同 seed GAMMA_050 按完整 test rows 对齐。四项 error 报：

$$
100\times\frac{E_{040}-E_{050}}{E_{050}},
$$

PCC 报绝对差 `PCC_040-PCC_050`。统一保留候选/基线原值、三-seed mean ± sample SD、paired-seed 方向和逐 `samp_id` 差异。所有表显式标记 `split=test`。

本附件不构造加权总分，也不以 test 新设 quality selector。解释重点是 validation 方向是否在复用 test 上保持、效应量大小、seed/subject 异质性与属性取舍。三 seed SD 描述训练实例差异；重叠窗口不作为独立显著性检验样本。

## 5. 质量与失败口径

- 五主指标的有效分母必须为 2310；target eligibility 必须逐窗口匹配 W0；
- 非有限 input、target、prediction、checkpoint 或应有资格的主指标显式失败；
- prediction degeneracy 保留在完整结果中并标记质量状态，不据此删样本或换 checkpoint；
- row/order、cache/checkpoint/config/seed/epoch 漂移属于工程失败；
- 工程失败保留 lifecycle 和现场，不生成不完整 summary；
- 同 test lock 的已完成 seed 和 summary 拒绝重复运行或覆盖。

## 6. 实现与预期产物

```text
resp_train/paper_evidence/w0_film_gamma_test.py
scripts/eval_w0_film_gamma_test.py
tests/test_w0_film_gamma_test.py
docs/experiments/w0_film_gamma_test_lock_20260919.json
runs/w0_film_gamma_test_v1/
```

每 seed 输出独立 evaluation attempt：

- `metrics.csv`、`metrics_summary.csv`、`test_rows.csv`；
- `resolved_config.yaml`、implementation lock 快照与环境；
- `access_started.json`、`access_receipt.json`、`evaluation_receipt.json`；
- lifecycle、artifact manifest 与 freeze receipt。

完整三个 seed 后汇总输出：

- `seed_metrics.csv`：GAMMA_050/GAMMA_040 共 6 行；
- `paired_seed_delta.csv`：3 seeds × 5 metrics，共 15 行；
- `three_seed_comparison.csv`：5 行；
- `subject_summary.csv` 与中文描述性结论；
- 来源回执、artifact manifest 与 freeze receipt。

## 7. 阶段与执行边界

1. T0：协议、实现、专项锁、synthetic CPU 测试；
2. T1：用户执行 3 个固定 checkpoint 的真实 GPU test evaluation；
3. T2：完整矩阵汇总、结果文档和冻结回执。

本轮只完成 T0。真实 test 访问由用户在实现提交后执行；不得根据前一个 seed 的 test 结果改变后续矩阵或统计口径。

## 8. 验证、命令与验收

synthetic CPU 定向测试共 13 passed，fixture 固定为 4 个合成 test windows、2 个合成 `samp_id`、batch=3+1；正式合同仍为 2310 windows、8 个 `samp_id`、batch=128。prepare-lock fixture 使用不可解码的 cache 字节，并禁止 `numpy.load`，用于确认准备阶段不会提前打开 test arrays。

实现与锁准备命令：

~~~bash
GAMMA_TEST_WT=/mnt/disk_code/marques/resp_reconstruction_w0_cwt_film
GAMMA_TEST_PY=/mnt/disk_code/marques/resp_reconstruction/.venv/bin/python

cd "$GAMMA_TEST_WT"

env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  "$GAMMA_TEST_PY" -m pytest tests/test_w0_film_gamma_test.py -q

env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  "$GAMMA_TEST_PY" scripts/eval_w0_film_gamma_test.py prepare-lock
~~~

提交实现与专项锁并保持 clean commit 后，真实评价由用户执行：

~~~bash
for GAMMA_TEST_SEED in 20260811 20260812 20260813; do
  env -u LD_LIBRARY_PATH -u LD_PRELOAD \
    "$GAMMA_TEST_PY" scripts/eval_w0_film_gamma_test.py evaluate \
    --seed "$GAMMA_TEST_SEED" --device cuda:0 || break
done
~~~

三个完成 attempt 通过核验后再执行：

~~~bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  "$GAMMA_TEST_PY" scripts/eval_w0_film_gamma_test.py summarize --runs \
  /三个完成的evaluation_attempt目录
~~~

每 seed 验收：固定 checkpoint/epoch/config/系数 identity、2310 个唯一 test rows、8 个 `samp_id`、五主指标分母完整、target eligibility 匹配 W0、cache/row order 一致、质量状态和 access receipt 齐全。summary 必须恰含 6/15/5 行 seed/paired/three-seed 主表，并保存 subject summary、中文结论和冻结回执。
