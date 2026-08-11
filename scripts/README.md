# 当前 THO 与 CRD 实验入口

本文只描述当前冻结的新呼吸重建协议。旧 E/F/G probe、旧 loss、旧 metrics、旧 gate/topK 和历史 checkpoint 语义不再属于当前 workflow；旧代码与说明通过 Git 追溯，历史 run 原地保留。

唯一主协议见 `docs/experiments/loss_metrics_restart_plan_20260729.md`；CRD-v1.1 S0/S1 的规范附件见 `docs/experiments/crd_v1_protocol_20260808.md`。

## 当前固定口径

- 数据：2026-06-20 research v2 soft-z。
- 输入：`bcg_rawish_segment_soft_z_key`。
- target：`target_waveform_segment_soft_z_key`。
- 当前 research-test 集合：B0 PatchMixer、T2 宽频 native、T4 宽频 bandenergy、F0 固定呼吸带和 IEWT。T1/T3 已由 validation 退出。
- 训练 loss：`L_sync + 0.25 L_effort`；rhythm 与短期 polarity 已由消融删除。
- 正式输出：$\Pi=S\circ B$，统一 `0.05–0.70 Hz`。
- checkpoint：完整 validation Local RR MAE 最小 epoch。
- early stopping：关闭。
- 包络主指标：`envelope_trajectory_mae` 与 `global_envelope_modulation_error`；
  `target_stratified_envelope_spearman` 只按 train-frozen Low/Medium/High 分层补充报告。
- research-test：现有 `test` 可在阶段性整理后重复观察，并可形成后续独立科研问题；它不是无偏 held-out 证据，不得用于重选既有 run 的 epoch/checkpoint。
- CRD 训练与普通 `eval_crd.py` 仍只读 train/validation；S1C 只允许 candidate lock 中的 12 个 checkpoint 通过专用入口各读取一次现有 research-test。

## 数据与 split 审计

数据审计：

```bash
./.venv/bin/python scripts/audit_tho_dataset.py \
  --config configs/tho_research_v2.yaml \
  --output /tmp/tho_restart_audit.csv
```

Split 独立性审计：

```bash
./.venv/bin/python scripts/audit_split_independence.py \
  --config configs/tho_research_v2.yaml \
  --output-dir runs/audits/split_independence_restart
```

CRD S1C 使用严格 CRD 配置 loader 的复现命令：

```bash
./.venv/bin/python scripts/audit_split_independence.py \
  --config-kind crd \
  --config configs/crd_v1/crd_105_direct_coarse.yaml \
  --output-dir runs/crd_v1/crd_s1c_split_audit_20260809
```

这些审计不是每个训练 run 的重复前置步骤。协议首次实现或数据/split 发生变化时执行并保存结果；普通 run 只保留加载、shape 与 finite 断言。

## 训练

统一入口：

```bash
./.venv/bin/python scripts/train_tho.py \
  --config configs/tho_research_v2.yaml \
  --set training.device=cuda:0
```

### T2 实现 smoke

Smoke 不是科研结果：

```bash
./.venv/bin/python scripts/train_tho.py \
  --config configs/tho_research_v2.yaml \
  --set data.max_train_windows=32 \
  --set data.max_val_windows=32 \
  --set training.epochs=2 \
  --set training.batch_size=8 \
  --set training.seed=20260811 \
  --set training.device=cuda:0 \
  --set outputs.run_root=/tmp/tho_restart_t2_smoke
```

T2、T3、T4 均已分别通过一次性 CPU 生命周期 smoke；数值不构成科研结果。

### Batch 128 GPU 验收（已通过）

T2 native 路径：

```bash
./.venv/bin/python scripts/train_tho.py \
  --config configs/tho_research_v2.yaml \
  --set data.max_train_windows=128 \
  --set data.max_val_windows=32 \
  --set training.epochs=1 \
  --set training.batch_size=128 \
  --set training.seed=20260811 \
  --set training.device=cuda:0 \
  --set outputs.run_root=/tmp/tho_restart_t2_batch128_acceptance
```

T3 concat-deep 使用不同 decoder，需单独验收：

```bash
./.venv/bin/python scripts/train_tho.py \
  --config configs/tho_research_v2_t3_concat.yaml \
  --set data.max_train_windows=128 \
  --set data.max_val_windows=32 \
  --set training.epochs=1 \
  --set training.batch_size=128 \
  --set training.seed=20260811 \
  --set training.device=cuda:0 \
  --set outputs.run_root=/tmp/tho_restart_t3_batch128_acceptance
```

T4 与 T2 共用内存更低的 native 路径，T2 通过后不重复 batch 128 验收。

验收目录分别为 `/tmp/tho_restart_t2_batch128_acceptance/20260805_155009_160323` 和 `/tmp/tho_restart_t3_batch128_acceptance/20260805_155030_379024`。

### T2 三 seed 正式 validation

先只运行第一顺位 T2：

```bash
for seed in 20260811 20260812 20260813; do
  ./.venv/bin/python scripts/train_tho.py \
    --config configs/tho_research_v2.yaml \
    --set training.epochs=50 \
    --set training.seed="${seed}" \
    --set training.device=cuda:0 \
    --set outputs.run_root="runs/tho_restart_t2_g3c_wide_native/seed_${seed}" \
    || exit 1
done
```

T2、T3、T4 validation 均已完成。当前保留 T2/T4，并与 B0、F0、IEWT 一起进入阶段性 research-test；B0/M1/T1/T3 和 loss 消融由各自 run manifest 与 Git 历史追溯。

## CRD-v1.1 S0/S1/S2A/S2B-R

八个 S0/S1、三个 S2A 与四个 S2B-R 配置：

```text
configs/crd_v1/crd_001_b0_retrain.yaml
configs/crd_v1/crd_002_t4_retrain.yaml
configs/crd_v1/crd_101_b0_coarse.yaml
configs/crd_v1/crd_102_b0_local_mamba.yaml
configs/crd_v1/crd_103_direct_local_mamba.yaml
configs/crd_v1/crd_104_direct_hier_mamba.yaml
configs/crd_v1/crd_105_direct_coarse.yaml
configs/crd_v1/crd_106_b0_hier_mamba.yaml
configs/crd_v1/crd_202_base_legacy_energy.yaml
configs/crd_v1/crd_203_base_analytic_am.yaml
configs/crd_v1/crd_204_base_morphology.yaml
configs/crd_v1/crd_205_base_em_static.yaml
configs/crd_v1/crd_206_base_am_static.yaml
configs/crd_v1/crd_207_base_cap_em.yaml
configs/crd_v1/crd_208_base_cap_am.yaml
```

先确认固定原生依赖：

```bash
./.venv/bin/python -c \
  "from resp_train.crd.config import check_crd_dependencies; p=check_crd_dependencies(); print(p); raise SystemExit(bool(p))"
```

在目标 GPU 上执行 actual CRD BiMamba2 的 bf16 fast-path forward/backward 短验收：

```bash
./.venv/bin/python scripts/check_crd_mamba.py --device cuda:0
```

选定 variant 的完整 synthetic 单 microbatch 链路验收：

```bash
./.venv/bin/python scripts/check_crd_variant.py \
  --config configs/crd_v1/crd_103_direct_local_mamba.yaml \
  --device cuda:0 \
  --batch-size 1
```

### CRD CPU 生命周期 smoke

CPU smoke 只建议用于不含 Mamba 的 001/002/101/105；下面以 101 为例。它不形成科研结果：

```bash
./.venv/bin/python scripts/train_crd.py \
  --config configs/crd_v1/crd_101_b0_coarse.yaml \
  --set protocol.run_role=smoke \
  --set data.max_train_windows=4 \
  --set data.max_val_windows=2 \
  --set training.epochs=1 \
  --set training.batch_size=2 \
  --set training.gradient_accumulation_steps=1 \
  --set training.device=cpu \
  --set training.show_progress=false \
  --set outputs.run_root=/tmp/crd_101_cpu_smoke
```

### CRD GPU physical-batch-128 acceptance

每种实际要进入正式队列的结构都应单独验收。下面的一次 run 包含 1 个 physical-batch-128 microbatch、一次 optimizer update 和 32 个 validation windows：

```bash
./.venv/bin/python scripts/train_crd.py \
  --config configs/crd_v1/crd_103_direct_local_mamba.yaml \
  --set protocol.run_role=acceptance \
  --set data.max_train_windows=128 \
  --set data.max_val_windows=32 \
  --set training.epochs=1 \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --set outputs.run_root=/tmp/crd_103_batch32_acceptance
```

如发生 OOM，停止，不在正式命令中临时改变 batch/accumulation；按协议先复验并统一修订为 runner-up `64×2`。Acceptance 目录与数值不进入正式比较。

正式队列前的等效 batch-scaling 工程 benchmark：

```bash
./.venv/bin/python scripts/benchmark_crd_batch_scaling.py \
  --config configs/crd_v1/crd_103_direct_local_mamba.yaml \
  --device cuda:0 \
  --schemes 32x4 64x2 128x1 \
  --repeats 3 \
  --output /tmp/crd_batch_scaling_20260808.json
```

首轮 cold 不进入吞吐中位数；脚本记录 PyTorch allocated/reserved 峰值并按协议的 10% 吞吐增益、80% 显存安全线给出工程 recommendation。它不运行 validation/test，也不形成模型效果证据。

2026-08-08 的固定 CRD_103 benchmark 中，`32×4 / 64×2 / 128×1` 稳态吞吐分别为 `215.899 / 249.812 / 281.438 samples/s`，peak reserved 分别为 `3114 / 5448 / 10868 MiB`。`128×1` 相对基线提升 30.36%，占 16 GiB 设备显存的 68.17%，因此当前 formal/acceptance 配置已统一冻结为 physical batch 128、accumulation 1；effective batch 仍为 128。

### CRD 正式三 seed

必须按协议决策链逐 variant 推进，而不是一次性启动 001–104。以下模板只在对应上一阶段已经通过且工作树/commit 已冻结时使用：

```bash
variant=crd_001_b0_retrain
for seed in 20260811 20260812 20260813; do
  ./.venv/bin/python scripts/train_crd.py \
    --config "configs/crd_v1/${variant}.yaml" \
    --set training.seed="${seed}" \
    --set training.device=cuda:0 \
    --set outputs.run_root="runs/crd_v1/${variant}/seed_${seed}" \
    || exit 1
done
```

不要覆盖 `epochs/batch/accumulation` 或任何 `max_*_windows`；formal loader 会拒绝。S0 的 001/002 与 S1 的 101 已完成。101 已触发 signed-PCC 停止线，原 `101 → 102 → 103 → 可选104` 队列关闭；当前只允许按协议第 15–16 节执行 final-checkpoint 归因复评与 105 诊断。

CRD validation checkpoint 复评：

```bash
./.venv/bin/python scripts/eval_crd.py \
  --checkpoint runs/crd_v1/<variant>/seed_<seed>/<timestamp>/checkpoint_best_local_rr.pt \
  --set training.device=cuda:0 \
  --metrics-output /tmp/crd_validation_metrics.csv
```

该入口只读 validation，不接受 `--split test`。

### CRD_001/101 paired final-checkpoint 诊断复评

以下复评不改变原 Local-RR checkpoint 结论，输出进入独立诊断目录，不写回六个正式 run。先复评 CRD_001：

```bash
./.venv/bin/python scripts/eval_crd.py \
  --checkpoint runs/crd_v1/crd_001_b0_retrain/seed_20260811/20260808_035754_258179/checkpoint_final.pt \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --metrics-output runs/crd_v1/crd_final_checkpoint_diagnostic/crd_001_b0_retrain/seed_20260811/validation_metrics.csv

./.venv/bin/python scripts/eval_crd.py \
  --checkpoint runs/crd_v1/crd_001_b0_retrain/seed_20260812/20260808_042713_710965/checkpoint_final.pt \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --metrics-output runs/crd_v1/crd_final_checkpoint_diagnostic/crd_001_b0_retrain/seed_20260812/validation_metrics.csv

./.venv/bin/python scripts/eval_crd.py \
  --checkpoint runs/crd_v1/crd_001_b0_retrain/seed_20260813/20260808_045849_155730/checkpoint_final.pt \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --metrics-output runs/crd_v1/crd_final_checkpoint_diagnostic/crd_001_b0_retrain/seed_20260813/validation_metrics.csv
```

再复评 CRD_101：

```bash
./.venv/bin/python scripts/eval_crd.py \
  --checkpoint runs/crd_v1/crd_101_b0_coarse/seed_20260811/20260808_154855_237029/checkpoint_final.pt \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --metrics-output runs/crd_v1/crd_final_checkpoint_diagnostic/crd_101_b0_coarse/seed_20260811/validation_metrics.csv

./.venv/bin/python scripts/eval_crd.py \
  --checkpoint runs/crd_v1/crd_101_b0_coarse/seed_20260812/20260808_155229_535580/checkpoint_final.pt \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --metrics-output runs/crd_v1/crd_final_checkpoint_diagnostic/crd_101_b0_coarse/seed_20260812/validation_metrics.csv

./.venv/bin/python scripts/eval_crd.py \
  --checkpoint runs/crd_v1/crd_101_b0_coarse/seed_20260813/20260808_161853_510867/checkpoint_final.pt \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --metrics-output runs/crd_v1/crd_final_checkpoint_diagnostic/crd_101_b0_coarse/seed_20260813/validation_metrics.csv
```

若目标文件已经存在，先停止并核对，不得覆盖。D0 结果只判断 selector 是否可能贡献 PCC 下降，不能事后重选 CRD_101 或开放 CRD_102。

D0 已完成：fixed-final 下 CRD_001/101 的 Local RR seed mean 为 `0.661868/0.640469`，signed PCC seed mean 为 `0.841586/0.788269`；PCC 仍下降 `0.053317`，三个配对 seed 全部下降。selector 不是原 PCC 退化的主要解释，CRD_102 继续暂停，下一步进入 CRD_105。

### CRD_105 Direct-Coarse 诊断

先在目标 GPU 验证完整 synthetic 链路，再运行独立 physical-batch-128 acceptance：

```bash
./.venv/bin/python scripts/check_crd_variant.py \
  --config configs/crd_v1/crd_105_direct_coarse.yaml \
  --device cuda:0 \
  --batch-size 1

./.venv/bin/python scripts/train_crd.py \
  --config configs/crd_v1/crd_105_direct_coarse.yaml \
  --set protocol.run_role=acceptance \
  --set data.max_train_windows=128 \
  --set data.max_val_windows=32 \
  --set training.epochs=1 \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --set outputs.run_root=/tmp/crd_105_batch128_acceptance
```

两项通过、代码提交且工作树干净后，才运行三 formal seeds：

上述两项已在 commit `be21ba0` 下通过；acceptance 路径为 `/tmp/crd_105_batch128_acceptance/20260808_204245_249478`。该单 epoch 数值只属于工程验收，不作效果解释。当前允许执行：

```bash
for seed in 20260811 20260812 20260813; do
  ./.venv/bin/python scripts/train_crd.py \
    --config configs/crd_v1/crd_105_direct_coarse.yaml \
    --set training.seed="${seed}" \
    --set training.device=cuda:0 \
    --set training.show_progress=false \
    --set outputs.run_root="runs/crd_v1/crd_105_direct_coarse/seed_${seed}" \
    || exit 1
done
```

105 通过 CRD_001 coarse gate 后才允许按 `105 → 103 → 可选104` 推进；105 失败则停止。原 CRD_102 仍处于暂停状态。

CRD_105 三 formal seeds 已在 commit `2bee3e5` 下完成：Local RR seed mean `0.581234`、signed PCC seed mean `0.846330`，相对 CRD_001 分别改善 8.1006% 和增加 0.006043，明确通过 gate。当前跳过 CRD_102 并开放 CRD_103；CRD_104 仍等待 `103 vs 105` 结果。未来 CRD_103/104 formal run 与 105 一样记录 `crd-v1.1-s1d-20260808`，不再使用原队列的 protocol 标识。

CRD_103 三 formal seeds 已在 commit `ed9d68e` 下完成：相对 CRD_105，Local RR 恶化 0.6905%、配对方向为 0/3、trajectory MAE 恶化 5.2364%，违反三项必要 gate；signed PCC 虽增加 0.006461，不能单独推翻停止规则。S1D 当前保留 CRD_105、CRD_103 不保留，CRD_102/104 不进入模型选择队列；其后仅可按下面的 S1E 身份探索性补跑。

### CRD S1E 探索性补全

结果后协议第 17 节允许补跑 CRD_102/104，但不重新选择本轮模型。两者使用独立 `crd-v1.1-s1e-20260809` protocol；先分别执行 synthetic 与 acceptance：

```bash
for variant in crd_102_b0_local_mamba crd_104_direct_hier_mamba; do
  ./.venv/bin/python scripts/check_crd_variant.py \
    --config "configs/crd_v1/${variant}.yaml" \
    --device cuda:0 \
    --batch-size 1 \
    || exit 1

  ./.venv/bin/python scripts/train_crd.py \
    --config "configs/crd_v1/${variant}.yaml" \
    --set protocol.run_role=acceptance \
    --set data.max_train_windows=128 \
    --set data.max_val_windows=32 \
    --set training.epochs=1 \
    --set training.device=cuda:0 \
    --set training.show_progress=false \
    --set outputs.run_root="/tmp/${variant}_batch128_acceptance" \
    || exit 1
done
```

两个 variant 均通过后，执行完整三 seed：

两项 synthetic/acceptance 已在 commit `d60b050`、`git_dirty=false` 下通过；acceptance 路径分别为 `/tmp/crd_102_b0_local_mamba_batch128_acceptance/20260809_021443_770877` 与 `/tmp/crd_104_direct_hier_mamba_batch128_acceptance/20260809_021601_879119`。其单 epoch 指标不作效果解释。工作树重新确认干净后即可执行：

```bash
for variant in crd_102_b0_local_mamba crd_104_direct_hier_mamba; do
  for seed in 20260811 20260812 20260813; do
    ./.venv/bin/python scripts/train_crd.py \
      --config "configs/crd_v1/${variant}.yaml" \
      --set training.seed="${seed}" \
      --set training.device=cuda:0 \
      --set training.show_progress=false \
      --set outputs.run_root="runs/crd_v1/${variant}/seed_${seed}" \
      || exit 1
  done
done
```

S1E 运行当时不得覆盖 epochs/batch/accumulation、任何 `max_*_windows` 或读取 research-test。102/104 的原保留条件只作描述性参照；S1E 结果不会自动推翻当前保留的 CRD_105。后续 test 授权只来自下面另立的 S1C。

六个 S1E run 已在 commit `f8fa658` 下完成并通过审计。102 相对 101 除 trajectory 恶化 3.4048% 外，在 RR、PCC、global envelope 与 IBI 上均大幅改善；104 相对 103 描述性满足原四项条件。相对冻结的 105，102 的 Local RR 改善 4.3138%、3/3 paired 改善、PCC 增加 0.018342且 trajectory 仅恶化 0.2882%；104 的 Local RR/PCC 也改善，但 trajectory 恶化 3.0375%。这些结果只用于下一版确认协议设计；S1E 本身不重选模型、不授权 research-test。

### CRD candidate lock

CRD_102/104/105 的九个候选 checkpoint 与 CRD_001 三个 reference 已冻结在：

```text
docs/experiments/crd_v1_candidate_lock_20260809.json
```

验证 checkpoint、resolved config、run manifest 和 validation summary 的大小/SHA-256：

```bash
./.venv/bin/python scripts/verify_crd_candidate_lock.py
```

选择规则固定为相对 CRD_105 的四项资格门槛后进行五项 primary Pareto；多于一个非支配候选时保留 Pareto set，不强制单赢家。

### CRD S1C research-test 确认

现有 research-test 已为上述 12 个 frozen checkpoints 完成一次评价。它曾在旧模型阶段被观察，因此结果属于 development/research confirmation evidence，不是无偏 held-out。以下冻结命令仅保留作 provenance，12 份 access receipt 已齐备，**不得再次运行**：

```bash
./.venv/bin/python scripts/verify_crd_candidate_lock.py

set -o pipefail
./.venv/bin/python scripts/verify_crd_candidate_lock.py --print-checkpoints | \
while IFS= read -r checkpoint; do
  ./.venv/bin/python scripts/eval_crd_s1c.py \
    --checkpoint "${checkpoint}" \
    --device cuda:0 \
    --confirm-research-test \
    || exit 1
done
```

冻结规则汇总器也已运行，以下命令只作 provenance：

```bash
./.venv/bin/python scripts/summarize_crd_s1c.py
```

运行使用干净 commit `3b28001`。12 份 metrics 各 2310 行，总计 27720 行，manifest/lock/split/finite 检查全部通过。冻结结果为：102 通过全部 eligibility，104 未通过 Local-RR 与 trajectory 门槛；eligible set 为 `{102,105}`，102 在五项 primary 上严格支配 105，因而是唯一 non-dominated candidate。输出固定在 `runs/crd_v1/crd_s1c_research_confirmation/`。CRD_001 reference 的 Whole RR、IBI-MedAE 与 coherence 仍优于 102，不能把选择结果写成所有轴全面占优。S1C 队列现已关闭；下一阶段仍需另立 research-test-informed 协议。

### CRD S1F：CRD_106 B0-Hier-Mamba

S1F 只补齐 `CRD_102 + CRD_104 同构 global/FiLM` 这一格，不开放其他 S1/S2 模型，也不读取 research-test。提交后先执行 synthetic 与独立 physical-batch-128 acceptance：

```bash
./.venv/bin/python scripts/check_crd_variant.py \
  --config configs/crd_v1/crd_106_b0_hier_mamba.yaml \
  --device cuda:0 \
  --batch-size 1

./.venv/bin/python scripts/train_crd.py \
  --config configs/crd_v1/crd_106_b0_hier_mamba.yaml \
  --set protocol.run_role=acceptance \
  --set data.max_train_windows=128 \
  --set data.max_val_windows=32 \
  --set training.epochs=1 \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --set outputs.run_root=/tmp/crd_106_b0_hier_mamba_batch128_acceptance
```

两项已在干净 commit `8fa56f8` 下通过并完成产物审计；acceptance 路径为 `/tmp/crd_106_b0_hier_mamba_batch128_acceptance/20260809_142221_810744`。现允许三个 formal seeds：

```bash
for seed in 20260811 20260812 20260813; do
  ./.venv/bin/python scripts/train_crd.py \
    --config configs/crd_v1/crd_106_b0_hier_mamba.yaml \
    --set training.seed="${seed}" \
    --set training.device=cuda:0 \
    --set training.show_progress=false \
    --set outputs.run_root="runs/crd_v1/crd_106_b0_hier_mamba/seed_${seed}" \
    || exit 1
done
```

三个 formal runs 已在干净 commit `80e6350` 下完成并通过完整性审计。相对冻结 CRD_102，106 的 Local RR 改善 `2.2123%`、3/3 paired seeds 改善，PCC 下降 `0.003178` 仍在护栏内；但 trajectory 恶化 `5.0351%`，超过 `1.5%`。因此 106 不满足四项全通过条件，不保留；S1F 关闭并固定未来 `S2 BASE=102`，不追加 variant、不读取 research-test。Whole RR 改善 `8.5536%` 只作任务交换背景，不能覆盖停止规则。

### CRD S2 表征分支

S2 BASE 固定为 candidate lock 中的 CRD_102，别名 `CRD_201_BASE`，不重训。当前只激活以下三个 S2A variant 的实现与工程验收：

```text
CRD_202_BASE_LEGACY_ENERGY
CRD_203_BASE_ANALYTIC_AM
CRD_204_BASE_MORPHOLOGY
```

三者都在 BASE PatchTokenFrontend 后、local Mamba 前以 zero-init static residual 加入；不含 Direct/global/gate/auxiliary。当前冻结的配置与 trainable parameter 数为：

| Variant | 配置 | Trainable params |
|---|---|---:|
| `CRD_202_BASE_LEGACY_ENERGY` | `configs/crd_v1/crd_202_base_legacy_energy.yaml` | 1,071,449 |
| `CRD_203_BASE_ANALYTIC_AM` | `configs/crd_v1/crd_203_base_analytic_am.yaml` | 1,197,785 |
| `CRD_204_BASE_MORPHOLOGY` | `configs/crd_v1/crd_204_base_morphology.yaml` | 1,106,857 |

先逐一运行 synthetic forward/backward，再运行彼此独立的 physical-batch-128 acceptance：

```bash
for variant in \
  crd_202_base_legacy_energy \
  crd_203_base_analytic_am \
  crd_204_base_morphology; do
  ./.venv/bin/python scripts/check_crd_variant.py \
    --config "configs/crd_v1/${variant}.yaml" \
    --device cuda:0 \
    --batch-size 1 \
    || exit 1

  ./.venv/bin/python scripts/train_crd.py \
    --config "configs/crd_v1/${variant}.yaml" \
    --set protocol.run_role=acceptance \
    --set data.max_train_windows=128 \
    --set data.max_val_windows=32 \
    --set training.epochs=1 \
    --set training.device=cuda:0 \
    --set training.show_progress=false \
    --set outputs.run_root="/tmp/${variant}_batch128_acceptance" \
    || exit 1
done
```

每个 synthetic 必须报告 finite output/gradient；每个 acceptance 必须完成一次 optimizer update、validation 和完整 checkpoint lifecycle。每个 run 的 `runtime_summary.json` 会固化 peak allocated/reserved 及 reserved/总显存比例，尤其检查 204。三项结果返回并审计、代码 commit 固定前，不启动 formal seeds。Energy 和 morphology 使用不同的 primary/guardrail，S2B 只在两类都 eligible 时条件开放；S2 全程禁止 research-test。

首轮结果中，202/203 已通过，peak reserved fraction 为 `62.18%/74.25%`；204 synthetic 通过但 batch-128 训练 OOM。204 现已加入不改变数学定义的 per-chunk activation checkpoint，保持 physical batch 128。只重跑 204：

```bash
./.venv/bin/python scripts/check_crd_variant.py \
  --config configs/crd_v1/crd_204_base_morphology.yaml \
  --device cuda:0 \
  --batch-size 1

./.venv/bin/python scripts/train_crd.py \
  --config configs/crd_v1/crd_204_base_morphology.yaml \
  --set protocol.run_role=acceptance \
  --set data.max_train_windows=128 \
  --set data.max_val_windows=32 \
  --set training.epochs=1 \
  --set training.device=cuda:0 \
  --set training.show_progress=false \
  --set outputs.run_root=/tmp/crd_204_base_morphology_checkpointed_batch128_acceptance
```

验收要求仍是完整 lifecycle、output/gradient/checkpoint finite、所有 eligible primary metrics finite，且 `runtime_summary.json` 的 `peak_reserved_fraction≤0.80`；按 eligibility flag 显式未定义的指标不伪造有限值。若仍失败，停止，不临时减小 batch 或 chunk。

204 checkpointed 重验已在干净 commit `a149913` 下通过，peak allocated/reserved 为 `8,636.81/10,406 MiB`、reserved fraction `65.30%`。三个 S2A variant 至此均完成工程验收；正式队列在本记录提交且工作树干净后开放。依次运行九个 formal runs：

```bash
for variant in \
  crd_202_base_legacy_energy \
  crd_203_base_analytic_am \
  crd_204_base_morphology; do
  for seed in 20260811 20260812 20260813; do
    ./.venv/bin/python scripts/train_crd.py \
      --config "configs/crd_v1/${variant}.yaml" \
      --set training.seed="${seed}" \
      --set training.device=cuda:0 \
      --set training.show_progress=false \
      --set outputs.run_root="runs/crd_v1/${variant}/seed_${seed}" \
      || exit 1
  done
done
```

不得覆盖 epochs、batch、accumulation 或任何 `max_*_windows`，不得读取 research-test。九个 runs 完成后先做完整性与 checkpoint 集合审计，再按附件第 21.4–21.5 节分别判断 energy representation 与 morphology eligibility；在结果审计前不实现 S2B。

九个 formal runs 已在统一干净 commit `41ed41d` 下完成并通过完整性审计。Selected epochs 为 202=`10/18/12`、203=`22/11/12`、204=`8/11/12`。冻结门槛得到 `X=none` 且 M 不 eligible，因此 S2B/S3 关闭并保留 CRD_102；prototype 描述不能推翻该结果。为满足第 21.5 节的完整报告要求，只读三个 204 validation-selected checkpoints：

```bash
for checkpoint in \
  runs/crd_v1/crd_204_base_morphology/seed_20260811/20260809_224710_306894/checkpoint_best_local_rr.pt \
  runs/crd_v1/crd_204_base_morphology/seed_20260812/20260810_011253_712445/checkpoint_best_local_rr.pt \
  runs/crd_v1/crd_204_base_morphology/seed_20260813/20260810_033710_435683/checkpoint_best_local_rr.pt; do
  ./.venv/bin/python scripts/eval_crd_morphology_prototypes.py \
    --checkpoint "${checkpoint}" \
    --device cuda:0 \
    || exit 1
done
```

三项完成后，在干净工作树运行冻结汇总：

```bash
./.venv/bin/python scripts/summarize_crd_s2a.py
```

汇总入口会重新审计 BASE lock、九个 formal runs、逐 sample identity/metrics summary、checkpoint finite/hash、prototype manifest/hash，并输出 seed mean±sample SD、paired window/samp 描述、prototype usage/entropy 与最终 decision；已有输出拒绝覆盖。

上述三项 prototype 描述与冻结汇总已在干净 commit `3c3598c` 下完成。Prototype 未发生全局单类坍缩：三个 seed 的 global dominant hard fraction 为 `37.26%/47.50%/34.06%`，soft-usage entropy 为 `0.9021/0.8809/0.9540`；但这不能覆盖 M 已失败的 PCC、Local-RR 与 coverage 门槛。最终产物：

```text
runs/crd_v1/crd_s2a_validation_summary/s2a_decision.json
runs/crd_v1/crd_s2a_validation_summary/s2a_seed_summary.csv
runs/crd_v1/crd_s2a_validation_summary/s2a_variant_summary.csv
runs/crd_v1/crd_s2a_validation_summary/s2a_paired_descriptives.csv
runs/crd_v1/crd_s2a_validation_summary/s2a_prototype_seed_summary.csv
runs/crd_v1/crd_s2a_validation_summary/s2a_summary_manifest.json
```

Decision 固定为 `X=none / M ineligible / S2B=false / S3=false / retain CRD_102`。S2A 已关闭，以上 prototype/summary 命令只保留 provenance，不得重复执行或用 research-test 重选。

### CRD S2B-R 双因素交互补救

研究者在获知 S2A 结果后明确要求探索单因素失败、多因素非线性补偿，因此新增的 S2B-R 是 result-informed exploratory stage，不改写上面的 S2A decision，也不是原 S2B 条件自然触发。四个模型为：

| Variant | 结构 | Params | Capacity match |
|---|---|---:|---|
| `crd_205_base_em_static` | BASE + E + M | 1,109,561 | — |
| `crd_206_base_am_static` | BASE + A + M | 1,235,897 | — |
| `crd_207_base_cap_em` | BASE + capacity control | 1,109,257 | `N=2,H=104`，差 `-304` |
| `crd_208_base_cap_am` | BASE + capacity control | 1,235,785 | `N=4,H=216`，差 `-112` |

以下四项 synthetic 与独立 physical-batch-128 acceptance 命令现只保留 provenance：

```bash
for variant in \
  crd_205_base_em_static \
  crd_206_base_am_static \
  crd_207_base_cap_em \
  crd_208_base_cap_am; do
  ./.venv/bin/python scripts/check_crd_variant.py \
    --config "configs/crd_v1/${variant}.yaml" \
    --device cuda:0 \
    --batch-size 1 \
    || exit 1

  ./.venv/bin/python scripts/train_crd.py \
    --config "configs/crd_v1/${variant}.yaml" \
    --set protocol.run_role=acceptance \
    --set data.max_train_windows=128 \
    --set data.max_val_windows=32 \
    --set training.epochs=1 \
    --set training.device=cuda:0 \
    --set training.show_progress=false \
    --set outputs.run_root="/tmp/${variant}_batch128_acceptance" \
    || exit 1
done
```

每项要求完整 lifecycle、eligible primary finite、joint prediction nondegenerate 且 `peak_reserved_fraction≤0.80`。若某项 OOM 或越线，停止，不临时减 batch/结构。S3 gate 仍未实现，S2B-R 全程禁止 research-test。

四项已在统一干净 commit `0e541f8` 下通过；peak reserved fraction 为 `67.37%/67.15%/65.41%/71.46%`。以下 12 个 formal runs 命令现只保留 provenance，不得重复运行：

```bash
for variant in \
  crd_205_base_em_static \
  crd_206_base_am_static \
  crd_207_base_cap_em \
  crd_208_base_cap_am; do
  for seed in 20260811 20260812 20260813; do
    ./.venv/bin/python scripts/train_crd.py \
      --config "configs/crd_v1/${variant}.yaml" \
      --set training.seed="${seed}" \
      --set training.device=cuda:0 \
      --set training.show_progress=false \
      --set outputs.run_root="runs/crd_v1/${variant}/seed_${seed}" \
      || exit 1
  done
done
```

12 项已在统一干净 training commit `c2bcfb0` 下完成；selected epochs 为 205=`8/11/12`、206=`19/11/9`、207=`13/11/12`、208=`10/18/15`。冻结汇总命令如下，现只保留 provenance：

```bash
./.venv/bin/python scripts/summarize_crd_s2br.py
```

产物位于：

```text
runs/crd_v1/crd_s2br_validation_summary/s2br_decision.json
runs/crd_v1/crd_s2br_validation_summary/s2br_seed_summary.csv
runs/crd_v1/crd_s2br_validation_summary/s2br_variant_summary.csv
runs/crd_v1/crd_s2br_validation_summary/s2br_paired_descriptives.csv
runs/crd_v1/crd_s2br_validation_summary/s2br_factorial_seed_summary.csv
runs/crd_v1/crd_s2br_validation_summary/s2br_factorial_paired_descriptives.csv
runs/crd_v1/crd_s2br_validation_summary/s2br_summary_manifest.json
```

两个组合的 BASE/control/best-constituent Local RR 门槛均失败：205 的相对改善为 `-1.0387%/-0.9260%/-2.5769%`，206 为 `-1.0295%/-1.4515%/-1.0295%`。Factorial descriptives 显示非线性补偿方向存在，但绝对组合效果不足；decision 固定为 `passing_combinations=[] / retain CRD_102 / S3=false`。S2B-R 已关闭，以上训练与汇总入口不得重复执行或用于读取 research-test 后重选。

### CRD_102 validation 误差分层与失败模式

该诊断只读取 candidate-lock 中 CRD_102 三个冻结 run 的既有 validation `metrics.csv`，不需要 GPU、不重新 eval、不访问 research-test。以下命令现只保留 provenance，不得重复运行：

```bash
./.venv/bin/python -m pytest tests/test_crd_failure_diagnostics.py
./.venv/bin/python scripts/summarize_crd102_failures.py
```

第一层已从干净 commit `81fab55` 输出到 `runs/crd_v1/crd_102_failure_diagnostic/`；worst-decile/persistent failure、固定分层、association 和签名口径见协议附件第 23 节。

第一层满足 metadata follow-up 条件后，以下命令已从新的干净 commit 运行，现只保留 provenance：

```bash
./.venv/bin/python -m pytest tests/test_crd_failure_metadata.py
./.venv/bin/python scripts/summarize_crd102_failure_metadata.py
```

第二层已从干净 commit `7a1b29b` 输出到 `runs/crd_v1/crd_102_failure_metadata_diagnostic/`，只连接冻结 validation consensus 与 dataset index；没有读取波形、重新 eval 或访问 research-test。核心结果为：high modulation 占 Local-RR/multimetric persistent failures 的 `70.61%/91.76%`，三 seed Local-RR 难例高度一致，且 `84.73%/80.40%` 的相应失败窗口位于多窗口连续片段。两层均已关闭，所有入口不得用于重选 checkpoint 或自动触发新模型阶段。

### CRD_102 matched observability

该入口按附件第 24 节冻结规则选择 high-modulation failure/control pairs，只读取匹配窗口的 rawish/fixed-band/target 波形并计算 CPU proxy metrics，不执行模型 inference。以下命令现只保留 provenance：

```bash
./.venv/bin/python -m pytest tests/test_crd_matched_observability.py
./.venv/bin/python scripts/summarize_crd102_matched_observability.py
```

唯一结果从干净 commit `c951325` 输出到 `runs/crd_v1/crd_102_matched_observability_diagnostic/`。21 个 exact-state primary pairs 中 rawish proxy 的 Local-RR/PCC case-worse 为 `16/21、13/21`，fixed-band 为 `17/21、16/21`，decision 为 `mixed_observability_and_model_tracking`；28 个 same-samp sensitivity pairs 也不支持纯输入受限结论。该阶段已关闭，入口不得重复运行，结果不构成因果证据或新候选选择。

## 固定呼吸带传统基线

`F0_fixed_band_bcg` 直接使用当前数据集的
`bcg_resp_band_state_aligned_segment_soft_z` 作为预测，数据 admission、validation split、
canonical 输出算子、五项 primary、IBI、eligibility 和逐 sample direct mean 均与深度学习一致。
该方法是确定性基线，不训练、不选择 checkpoint，也不报告 seed 方差。

完整 validation 统计：

```bash
./.venv/bin/python scripts/eval_tho_fixed_band_baseline.py \
  --config configs/tho_research_v2.yaml \
  --split val \
  --run-root runs/tho_fixed_band_baseline
```

实现 smoke 可以额外传入 `--max-windows 8`，但 smoke 不构成科研结果。Research-test 需
显式添加 `--split test --confirm-research-test`。

## 包络分层阈值复现

当前配置已经冻结 training target modulation 的三分层边界。以下命令只用于复现阈值与 provenance，
不得用 validation/test 重算：

```bash
./.venv/bin/python scripts/freeze_envelope_strata.py \
  --config configs/tho_research_v2.yaml \
  --output runs/envelope_strata_train_20260805.json
```

冻结结果为 `low=0.30875308839006915`、`high=0.7031542121234101`，来自 10141 个 admitted
training targets、linear quantile、train sample seed `20260610`。

## IEWT 传统基线

`IEWT` 使用与深度学习相同的
`bcg_rawish_wideband_state_aligned_segment_soft_z`，在 100 Hz 下执行协议化 Python IEWT：
三阶多项式去趋势、三阶 1 Hz 零相位低通、35 秒上下文生成 30 秒输出、六块拼接及整段
1 Hz 零相位后低通。两处低通均使用 `sosfiltfilt`，阶数和截止频率与原适配协议一致；
它只读取 BCG，target 只进入公共评价器。

完整 validation 命令为：

```bash
./.venv/bin/python scripts/eval_tho_iewt_baseline.py \
  --config configs/tho_research_v2.yaml \
  --split val \
  --run-root runs/tho_iewt_baseline
```

实现 smoke 可传入 `--max-windows 1` 或其他小值。零相位版本已完成单 sample CPU smoke 和完整
validation 新包络指标重评；此前因果版本只作系统群延迟诊断。当前实现不要求 MATLAB 数值对照，只能描述为依据
现有 MATLAB 源码定义并进行零相位适配的 Python IEWT，不能声称逐点等价。Research-test 要求
`--split test --confirm-research-test`。

该入口保存 `resolved_config.yaml`、`run_manifest.json`、`sample_metrics.csv` 和
`summary.csv`；不生成 checkpoint。

## Checkpoint 复评

Validation 复评：

```bash
./.venv/bin/python scripts/eval_tho.py \
  --checkpoint runs/<run>/checkpoint_best_local_rr.pt \
  --split val \
  --metrics-output /tmp/tho_restart_val_metrics.csv
```

Research-test 必须显式确认：

```bash
./.venv/bin/python scripts/eval_tho.py \
  --checkpoint runs/<run>/checkpoint_best_local_rr.pt \
  --split test \
  --confirm-research-test \
  --metrics-output runs/<run>/research_test_metrics.csv
```

Research-test 命令会同时计算五项 primary、IBI + coverage、三层 envelope Spearman、coherence 与 nDTW。结果可解释阶段性差异并形成后续独立研究任务，但不得重选既有 run 的 epoch/checkpoint；所有结论都必须注明是 research/development evidence。

### 当前阶段 research-test 命令

B0、T2、T4 各运行三个 validation-selected seed checkpoint，逐 run 保存结果：

```bash
for checkpoint in \
  runs/tho_restart_b0_final_loss_patchmixer/seed_*/20*/checkpoint_best_local_rr.pt \
  runs/tho_restart_t2_g3c_wide_native/seed_*/20*/checkpoint_best_local_rr.pt \
  runs/tho_restart_t4_g3c_bandenergy_native/seed_*/20*/checkpoint_best_local_rr.pt; do
  run_dir="$(dirname "${checkpoint}")"
  ./.venv/bin/python scripts/eval_tho.py \
    --checkpoint "${checkpoint}" \
    --split test \
    --confirm-research-test \
    --set training.device=cuda:0 \
    --metrics-output "${run_dir}/research_test_metrics.csv" \
    || exit 1
done
```

F0 与 IEWT 不训练、无 seed：

```bash
./.venv/bin/python scripts/eval_tho_fixed_band_baseline.py \
  --config configs/tho_research_v2.yaml \
  --split test \
  --confirm-research-test \
  --run-root runs/tho_fixed_band_research_test

./.venv/bin/python scripts/eval_tho_iewt_baseline.py \
  --config configs/tho_research_v2.yaml \
  --split test \
  --confirm-research-test \
  --run-root runs/tho_iewt_research_test
```

## 当前 run 产物

- `config.yaml`：resolved config。
- `run_manifest.json`：运行命令、Git commit 与 dirty 状态。
- `audit.csv`：数据加载审计摘要。
- `train_history.csv`：每 epoch 仅含 `train_loss_total`、`train_loss_sync`、`train_loss_effort`、`val_core_loss` 和 `val_local_rr_mae`。
- `checkpoint_best_local_rr.pt`：Local RR 严格最小 epoch；完全并列时保留更早 epoch。
- `checkpoint_final.pt`：固定预算最后 epoch，仅用于追溯。
- `metrics.csv`：选中 checkpoint 的完整 validation 逐 sample 指标。
- `metrics_summary.csv`：逐 sample direct-mean validation 汇总。
- `research_test_metrics.csv` / `research_test_metrics_summary.csv`：显式 research-test 评价产物。
- `*_metrics_manifest.json`：checkpoint 复评的命令、split、配置与代码版本。
- `train.log`：训练日志。

CRD run 额外保存 `optimizer_parameter_groups.json` 与 `runtime_summary.json`；后者记录训练到最终 validation 复评期间的 CUDA peak allocated/reserved 和显存占比。`train_history.csv` 还记录 optimizer update、每 epoch 首末 LR。CRD checkpoint 的 `extra_state` 保存协议版本、update index/total updates、依赖版本与 `resume_supported=false`。

不再生成或解释旧 `checkpoint.pt`、`checkpoint_best_rr.pt`、`checkpoint_best_task.pt`、`checkpoint_topN.pt`、`epoch_metrics.csv`、旧 target-feature cache 或旧指标 summary。
