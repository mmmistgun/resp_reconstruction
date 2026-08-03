# 当前 THO 实验入口

本文只描述当前冻结的新呼吸重建协议。旧 E/F/G probe、旧 loss、旧 metrics、旧 gate/topK 和历史 checkpoint 语义不再属于当前 workflow；旧代码与说明通过 Git 追溯，历史 run 原地保留。

协议定义见 `docs/experiments/loss_metrics_restart_plan_20260729.md`。

## 当前固定口径

- 数据：2026-06-20 research v2 soft-z。
- 输入：`bcg_rawish_segment_soft_z_key`。
- target：`target_waveform_segment_soft_z_key`。
- 最终学习模型：纯时域 `patch_mixer1d`；M1 多尺度候选未入选。
- 训练 loss：`L_sync + 0.25 L_effort`；rhythm 与短期 polarity 已由消融删除。
- 正式输出：$\Pi=S\circ B$，统一 `0.05–0.70 Hz`。
- checkpoint：完整 validation Local RR MAE 最小 epoch。
- early stopping：关闭。
- designated test：最终 PatchMixer 三个 seed 与固定带传统参照已经冻结，尚未执行。

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

这些审计不是每个训练 run 的重复前置步骤。协议首次实现或数据/split 发生变化时执行并保存结果；普通 run 只保留加载、shape 与 finite 断言。

## 训练

统一入口：

```bash
./.venv/bin/python scripts/train_tho.py \
  --config configs/tho_research_v2.yaml \
  --set training.device=cuda:0
```

### 实现 smoke

Smoke 不是科研结果：

```bash
./.venv/bin/python scripts/train_tho.py \
  --config configs/tho_research_v2.yaml \
  --set data.max_train_windows=32 \
  --set data.max_val_windows=32 \
  --set training.epochs=2 \
  --set training.batch_size=8 \
  --set training.seed=20260802 \
  --set training.device=cuda:0 \
  --set outputs.run_root=runs/tho_restart_b0_smoke
```

### 单 seed 预算 pilot

默认 config 即 pilot：完整 train/validation、50 epochs、seed `20260802`、batch 128。只需指定设备：

```bash
./.venv/bin/python scripts/train_tho.py \
  --config configs/tho_research_v2.yaml \
  --set training.device=cuda:0
```

若 pilot 的最小 validation Local RR epoch 位于 41–50，正式预算统一为 80 epochs；否则为 50。Pilot 不进入正式结果，也不查看 test。

### 三 seed 正式 baseline（已完成）

Pilot 已将正式预算冻结为 50 epochs；正式 baseline 已按 seed `20260811 / 20260812 / 20260813` 完成：

```bash
./.venv/bin/python scripts/train_tho.py \
  --config configs/tho_research_v2.yaml \
  --set training.epochs=50 \
  --set training.seed=20260811 \
  --set training.device=cuda:0 \
  --set outputs.run_root=runs/tho_restart_b0_formal/seed_20260811
```

另外两个 seed 使用完全相同配置，只改变训练 seed 与输出目录。

### 三个 loss 消融（已完成）

三个消融保持 PatchMixer、数据、metrics、checkpoint selector、50 epochs 和正式 seeds 不变，只把一个辅助 loss 权重精确置零。配置拒绝任意中间权重与未预注册组合；不进行权重搜索。以下三个循环已经完成，未打开 designated test。

`A1_no_rhythm`：

```bash
for seed in 20260811 20260812 20260813; do
  ./.venv/bin/python scripts/train_tho.py \
    --config configs/tho_research_v2.yaml \
    --set loss.rhythm_weight=0 \
    --set training.epochs=50 \
    --set training.seed="${seed}" \
    --set training.device=cuda:0 \
    --set outputs.run_root="runs/tho_restart_a1_no_rhythm/seed_${seed}" \
    || exit 1
done
```

`A2_no_effort`：

```bash
for seed in 20260811 20260812 20260813; do
  ./.venv/bin/python scripts/train_tho.py \
    --config configs/tho_research_v2.yaml \
    --set loss.effort_weight=0 \
    --set training.epochs=50 \
    --set training.seed="${seed}" \
    --set training.device=cuda:0 \
    --set outputs.run_root="runs/tho_restart_a2_no_effort/seed_${seed}" \
    || exit 1
done
```

`A3_no_pol`：

```bash
for seed in 20260811 20260812 20260813; do
  ./.venv/bin/python scripts/train_tho.py \
    --config configs/tho_research_v2.yaml \
    --set loss.pol_start_weight=0 \
    --set training.epochs=50 \
    --set training.seed="${seed}" \
    --set training.device=cuda:0 \
    --set outputs.run_root="runs/tho_restart_a3_no_pol/seed_${seed}" \
    || exit 1
done
```

9 个 run 的 validation 结果支持删除 rhythm、保留 effort、删除短期 polarity。详细决定见协议第 18 节。

### 最终 loss 的 PatchMixer baseline（已完成）

第五个实验 `B0_final_loss_patchmixer` 在物理精简前同时设置 `rhythm_weight=0` 与 `pol_start_weight=0`，三个 seed 已完成并确认没有联合删除交互，结果位于 `runs/tho_restart_b0_final_loss_patchmixer`。当前默认配置已经直接采用最终两项 loss，并拒绝旧字段重新进入训练；旧命令由对应 run manifest 追溯，不再作为当前可运行入口保留。

结果接受 `dirty-but-runtime-audited` provenance 例外，详见协议第 20 节。零权重分量已经从当前实现物理删除。

### M1 参数匹配多尺度纯时域模型（已完成，未入选）

M1 使用 `multiscale_patch_mixer1d`，四个 patch 尺度为 `256 / 512 / 1024 / 2048` 点，`base_channels=1`，总参数 `11664`；B0 为 `11408`。模型 raw head 不内置频带投影，继续统一使用公共 $\Pi=S\circ B$。以下命令只追溯已完成的验收和正式运行，不再用于追加预算或调参。

先做一次 batch 128 GPU 验收；该结果不构成科研结果：

```bash
./.venv/bin/python scripts/train_tho.py \
  --config configs/tho_research_v2.yaml \
  --set model.name=multiscale_patch_mixer1d \
  --set model.base_channels=1 \
  --set 'model.patch_lengths=[256,512,1024,2048]' \
  --set model.patch_stride_ratio=0.5 \
  --set model.mixer_layers=2 \
  --set data.max_train_windows=128 \
  --set data.max_val_windows=32 \
  --set training.epochs=1 \
  --set training.batch_size=128 \
  --set training.seed=20260802 \
  --set training.device=cuda:0 \
  --set outputs.run_root=/tmp/tho_restart_m1_batch128_acceptance
```

验收通过后，正式三个 seed 顺序运行：

```bash
for seed in 20260811 20260812 20260813; do
  ./.venv/bin/python scripts/train_tho.py \
    --config configs/tho_research_v2.yaml \
    --set model.name=multiscale_patch_mixer1d \
    --set model.base_channels=1 \
    --set 'model.patch_lengths=[256,512,1024,2048]' \
    --set model.patch_stride_ratio=0.5 \
    --set model.mixer_layers=2 \
    --set training.epochs=50 \
    --set training.seed="${seed}" \
    --set training.device=cuda:0 \
    --set outputs.run_root="runs/tho_restart_m1_multiscale_time/seed_${seed}" \
    || exit 1
done
```

M1 validation 提高了 effort，但 Whole/Local RR、signed PCC、IBI 与 coverage 均退化，因此未入选最终模型集合，也不进入 designated test。详细结果见协议第 22 节。

### 最终模型集合

- 学习模型：`B0_final_loss_patchmixer` 三个 Local RR 最优 checkpoint。
- 确定性参照：`F0_fixed_band_bcg`。
- M1：仅作 validation 负结果留档。

在 designated test 执行前，不再根据 validation 修改 loss、模型、epoch、频带、阈值或 detector。

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

实现 smoke 可以额外传入 `--max-windows 8`，但 smoke 不构成科研结果。Designated test 仍需
显式添加 `--split test --confirm-designated-test`，且只能在最终模型集合冻结后统一运行。

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

Designated test 必须显式确认：

```bash
./.venv/bin/python scripts/eval_tho.py \
  --checkpoint runs/<run>/checkpoint_best_local_rr.pt \
  --split test \
  --confirm-designated-test \
  --metrics-output runs/<run>/test_metrics.csv
```

Test 命令会同时计算五项 primary、IBI + coverage、coherence 与 nDTW。不得用 test 结果重选 epoch、模型、频带或阈值。

## 当前 run 产物

- `config.yaml`：resolved config。
- `run_manifest.json`：运行命令、Git commit 与 dirty 状态。
- `audit.csv`：数据加载审计摘要。
- `train_history.csv`：每 epoch 仅含 `train_loss_total`、`train_loss_sync`、`train_loss_effort`、`val_core_loss` 和 `val_local_rr_mae`。
- `checkpoint_best_local_rr.pt`：Local RR 严格最小 epoch；完全并列时保留更早 epoch。
- `checkpoint_final.pt`：固定预算最后 epoch，仅用于追溯。
- `metrics.csv`：选中 checkpoint 的完整 validation 逐 sample 指标。
- `metrics_summary.csv`：逐 sample direct-mean validation 汇总。
- `test_metrics.csv` / `test_metrics_summary.csv`：显式 designated test 评价产物。
- `*_metrics_manifest.json`：checkpoint 复评的命令、split、配置与代码版本。
- `train.log`：训练日志。

不再生成或解释旧 `checkpoint.pt`、`checkpoint_best_rr.pt`、`checkpoint_best_task.pt`、`checkpoint_topN.pt`、`epoch_metrics.csv`、旧 target-feature cache 或旧指标 summary。
