# Center-90 上下文效应独立测试集协议

日期：2026-09-04

协议 ID：`paper-center90-context-research-test-v1-20260904`

状态：**P4-U-T1 的 135 s input-only test W cache 已完成并冻结；18-checkpoint evaluator、allowlist 与双 GPU
顺序队列已开放，等待用户手动执行。**

## 1. 科学问题与证据边界

本附件检验 center-90 validation 上“上下文长度效应弱且依赖表征”的描述能否外推到独立测试集。固定输出中心 90 s，
输入为 90/135/180 s；C201-center90 与 W-reduced-center90 分别作为独立轨迹报告。

独立测试集与 train/validation 在 subject/session 上严格隔离，且未参与模型、epoch、checkpoint、长度矩阵或 validation
结论选择。评价只使用已经冻结的 validation-selected best-RR checkpoints。Test 结果形成 center-90 独立 panel，统计任务内
paired-seed 相对变化，不与 center-30、center-60 或 180→180 合并绝对指标。

## 2. 冻结矩阵与 test 身份

| 模型 | 输入 | seeds | checkpoints |
|---|---|---|---:|
| C201-center90 | 90/135/180 s | 20260811/12/13 | 9 |
| W-reduced-center90 | 90/135/180 s | 20260811/12/13 | 9 |

共 18 个 checkpoints，由 center-90 validation summary 唯一锁定：receipt/manifest SHA-256=
`b8e3428ec4f43094ec7508b4897f9d2b07a54c702dc07749e3fd5fbb9f959886 / cc532fe3930861dd27ef4eb0f8efbb55381ffd26ad5cb88b7cf5f3c0839d47bc`。

Test identity 固定为：

- dataset index SHA-256=`f65ae6524632187c7c2795bd7b5e25434afb649b5a04fe39084b70cc59f9b04f`；
- split=`test`，2310 windows，8 个 `samp_id`；
- row-ID SHA-256=`184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8`；
- input key=`bcg_rawish_segment_soft_z_key`，target 固定为 `[4500,13500)`；
- 18 项使用完全相同的 2310 row IDs，不按输出或指标过滤样本。

## 3. P4-U-T0/T1：最小增量 W cache

90 s 与 180 s input slice、transform spec 与已有 center-30/60 联合 test W cache 完全相同，直接复用：

```text
runs/paper_evidence_v1/context_length_research_test_w_cache/
  cf89c6e678bb243801c0ca577ec14d69d724603e51b3335c4e1f474e7f999d5c/
```

其 manifest SHA-256=`9b475926258d129851fb7b9c10d2b342ac2e833b121842b18437585059bf1b98`。

仅补建 135 s：每个 test parent BCG 加载一次并裁 `[2250,15750)`，执行冻结的 49-scale W transform，生成
`[2310,49,270]` float32 finite feature。计算量为 2310 次 CWT，feature 为 122,245,200 bytes（约 116.6 MiB）。
Cache 只读取 test BCG input；输出目录、lifecycle 与文件均不可覆盖。

用户从干净实现 commit 执行：

```bash
./.venv/bin/python scripts/build_paper_center90_research_test_w_cache_v1.py \
  --confirm-research-test-cache-build
```

返回 `cache_manifest.json` 后，先只读复核 lifecycle、全部文件 size/SHA-256、2310-row identity、shape/dtype/finite、
frequency identity 与 access flags，并冻结 path/hash。

P4-U-T1 已由用户从干净 commit `14871006ce42fbbf54f694b9ea6b6d2ef7004506` 完成，耗时 `01:24`。固定输出为：

```text
runs/paper_evidence_v1/center90_context_research_test_w_cache/
  fff524795f290969e4e3892b15a4f9ac2722c145e1ec3cd40e698c742e327f8a/
```

`cache_manifest.json` SHA-256=`4a7c7ad6bdd8fa21d1d8b2dde05ed5bf2ca8ac3746664e835542661d6e897072`。
只读验收确认 lifecycle=`complete`、2310 个严格唯一 row IDs、8 个 `samp_id`、non-test overlap=0；feature
shape=`[2310,49,270]`、float32、全量 finite，49 个 frequency centers 有限非降且无重复。Manifest 登记的 3 个
`.npy` 文件 size/SHA-256 全部匹配；access identity 确认只读取 test BCG input，未执行模型 inference。

## 4. P4-U-T2：18-checkpoint inference

Cache 冻结后实现受控 evaluator。每次调用固定一个 `model/input-sec/seed` identity，只接受本附件 18 项 allowlist：

- C201 从 test parent 裁对应 input；
- W-reduced 的 90/180 s 读取既有联合 cache，135 s 读取 P4-U-T1 补充 cache；
- 加载 `checkpoint_best_center90_rr.pt`，核验 checkpoint path、size、SHA-256 与 selected epoch；
- 输出 2310-row `center90_*` 五主指标、IBI coverage/interpretable 与身份列；
- 每项使用不可覆盖 evaluation identity，可分配到两张 GPU，进程内均使用逻辑 `cuda:0`。

预计成本为 `18×2310=41,580` 个样本推理。18/18 完成后才进行只读冻结汇总。

固定入口为 `scripts/eval_paper_center90_research_test_v1.py`。每次调用必须显式给出 `model/input-sec/seed` 和
`--confirm-research-test`；device 固定为逻辑 `cuda:0`，物理卡仅由 `CUDA_VISIBLE_DEVICES` 指定。Evaluator 从冻结
validation summary 逐项核验 run lifecycle、artifact manifest、resolved config、formal commit、checkpoint size/hash 与
selected epoch，再首次读取 test target。每项输出到不可覆盖目录：

```text
runs/paper_evidence_v1/center90_context_research_test/<experiment_id>/seed_<seed>/
```

定向 CPU 测试覆盖 18 项 allowlist、三种 target/input crop、W cache 分流、checkpoint payload identity 与 CLI gate。

## 5. P4-U-T3：冻结汇总

逐 seed 报告，再计算 arithmetic mean ± sample SD (`ddof=1`)。按模型计算 90→135、135→180、90→180 的
paired-seed 有向变化；四项 error 使用相对 `0.5%` 材料阈值，PCC 使用绝对 `0.002`。IBI 同时报告 coverage、
interpretable fraction 与 eligible count。

结果用于评价 center-90 上下文效应的独立测试集复现性；不构造加权总分、统一模型排名或跨输出长度绝对指标曲线。
