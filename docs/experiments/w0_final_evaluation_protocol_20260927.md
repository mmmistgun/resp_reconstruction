# W0 最终五指标评价 v1

协议 ID：`w0-final-evaluation-v1-20260927`。日期：2026-09-27。

状态：实现与 synthetic CPU 定向验证已完成（20 项通过）；真实 test 待用户执行。用户本次要求重新实现最终五指标，并明确选择“你实现并提供命令，我来运行”。本专项开放下面固定三 checkpoint 的新口径评价。历史阶段与产物保持冻结。

## 范围与来源

模型为原始 `crd_tf102_w` W0，默认 gamma/beta 系数各 0.5；seed `20260811/20260812/20260813`，固定 validation-selected epoch 分别 `13/15/14`。来源为 `w0_film_gamma_training_implementation_lock_20260918.json` 的 `baseline_runs` 和 `w0_film_gamma_test_lock_20260919.json` 的 baseline test/cache/index 身份；复用来源锁，不另建实验锁。对应 SHA-256 为 `2bf4b72a15111e1caa76b6bed19abbde3242edc451795176c307d160cc49368c`、`0d2bbbcc5ee6fee8e442475600c450de55c8e03ba31f18f5043e0557ac1ae712`。

数据为同一冻结 test：2310 个 180 s 窗口、8 个 samp_id，100 Hz；sample seed 20260612。train/validation/test 划分、target、loss、训练和 checkpoint selector 均沿用原来源。本次仅重评固定 checkpoint，无需重训；旧指标表不能换名后作为新协议结果。完整三 seed 结果从本次输出引用。Research-test 已用于开发，证据属性仍为 reused research/development evidence。

## 固定指标定义

所有数值评价使用 float64。预测和参考分别减均值、rFFT 硬掩码保留闭区间 `[0.05,0.70] Hz`、irFFT 回原长度，再减均值并除以 `sqrt(mean(centered_band²)+1e-8)`。整窗 N=18000；中央 60 s 从原始波形 `[6000:12000]` 独立执行 N=6000 投影归一化。

1. `rr180_mae_bpm`：整窗归一化波形取起点 0/3000/6000/9000/12000 的五个 6000 点窗口；每窗减均值、乘对称 Hann、计算未补零 rFFT 功率；逐 bin 取五谱中位数。在呼吸频带选择最大值。内部峰对 `log(power+1e-12)` 作三点抛物线插值，delta 裁剪到 ±0.5；频带边界或零二阶差分取 delta=0。RR=`60*(k+delta)*100/6000`。谱峰完全并列取最低频率 bin。取预测/参考 RR 绝对差。
2. `rr60_nonoverlap_mae_bpm`：中央原始波形独立投影归一化，使用单个对称 Hann 周期图和相同 RR 估计。按 `(samp_id, 规范化后的 target_source_npz 绝对路径)` 表示受试者与连续整晚记录，按绝对开始采样点排序，贪心保留最早中央窗，此后要求 start≥上次 end。状态/segment 是同一记录内的分段，不重启选择。选窗在首次预测前完成，不依据参考资格或预测值补选，三 seed 共享同一 CSV；每个保留且参考有效的窗口等权。
3. `aligned_waveform_mae`：在完整归一化波形上，复用第 5 项的最优 lag，取 `mean(abs(pred[30+k:17970+k]-target[30:17970]))`，固定 17940 点；不再拟合幅值或移动包络。
4. `relative_envelope_mae`：完整归一化波形以 1000 点窗、500 点步长取得 35 个 `.5*log(mean(v²)+1e-8)`；预测与参考各自减去其包络中位数，然后计算 35 点绝对差平均。
5. `lag_signed_pcc`：固定参考区间 `[30:17970]`，预测配对区间 `[30+k:17970+k]`，每个配对片段独立减均值。PCC 分母为 `sqrt(sum(pred_centered²)+1e-8)*sqrt(sum(target_centered²)+1e-8)`。最大化有符号 PCC，不取绝对值；按 `0,-1,+1,...,-30,+30` 遍历，仅严格大于时更新，落实完全并列优先级。

前两项单位次/min，其余无量纲；前四项越低越好，PCC 越高越好。每 seed 各项先按有效窗口等权平均，再对完整三 seed 计算 arithmetic mean 和 `ddof=1` sample SD。固定频带/IEWT 的单结果聚合受指标模块支持，本专项运行矩阵仅含 W0。

## 资格、分母和失败

沿用冻结索引与配置的数据资格：`allowed_losses` 含 waveform、reason 为空、hard_valid_ratio≥0.8、state_alignment_valid_ratio≥0.8，并核验冻结 test row 集合与顺序。任何 input/target/prediction/checkpoint 的非有限值显式失败；不静默删除窗口。

“去中心化频带能量”固定定义为归一化前频带波形的 `mean((band-mean(band))²)`。完整 RR 和包络参考资格为整窗能量严格大于 1e-8；PCC/波形 MAE 还要求参考固定公共区间能量严格大于 1e-8；中央 RR 资格为独立中央投影后的能量严格大于 1e-8。包络变化平坦本身不是不合格条件。失格参考保留原 row、显式 flag 和空指标，不进入相应均值；零有效分母失败。

所有预测窗口检查完整 180 s 能量和每个 PCC 候选配对片段能量；被选中的中央窗另查独立中央投影能量。任一相关预测能量≤1e-8，即使参考失格，也立即失败。归一化不能掩盖低能量。非入选中央窗不参与中央指标与中央动态性检查。

失败时保存 `failure.json`：方法、checkpoint、seed、stage、row（能定位时）、batch row IDs、原因和 traceback；终止后续 seed 与总体汇总。已生成的逐窗/逐 seed 文件仅是失败 lifecycle 的部分产物，不能作为完整总体结果。只有 `receipt.json` 与 `artifact_manifest.json` 都为 complete 且不存在 failure.json 时才接受结果。

## 执行与产物

推理沿用 batch=128、尾 batch=6、BF16 AMP、eval+inference_mode、shuffle=false；三个 seed 顺序执行。预先核验三个 checkpoint/config/history/run manifest，test cache/index 身份，test 与开发主体隔离。首尾核验源码、checkpoint、cache、索引及原始 NPZ 哈希；保存实际源码快照、训练/运行配置、环境、命令、Git 状态、来源 manifest、统一窗口表和中央选择表。

在仓库根目录由用户运行（目录必须尚不存在）：

```bash
./.venv/bin/python scripts/eval_w0_final_metrics.py \
  --device cuda:0 \
  --output runs/w0_final_evaluation_v1/three_seed_01 \
  --confirm-research-test
```

若失败，保留失败目录，修复后使用新的输出 identity。入口拒绝覆盖或原目录续跑。

产物：

- `results.md`、`summary.json`：最终五指标 mean±sample SD 及各项每 seed 有效窗口数。
- `seed_metrics.csv`、`seed_<seed>/summary.json`：三个 seed 独立结果和分母。
- `seed_<seed>/window_metrics.jsonl` / `.csv`：每个窗口的五指标、RR、best lag、参考资格与中央选窗标记；JSONL 逐行持久化。
- `test_rows.csv`、`center_selection.csv`：统一测试集合、中央窗绝对起止时间和记录身份，三 seed 共用。
- `reference.npy`、`seed_<seed>/prediction.npy`：原始输出空间波形，按 test_rows 顺序保存，便于后续离线复核。四个 float32 数组约 665 MB 十进制，另有源码与表格；预留至少 1 GB。
- `execution.json`、`source_manifest.json`、`source_code/`、各 seed config、访问/完成/失败回执及产物 manifest。

验收：三 seed 均完整覆盖 2310 行，中央选择和逐窗资格一致，所有有效指标有限，各项分母显式报告，source 首尾身份一致，完整完成回执和产物 manifest 齐全。中央有效数量由元数据选择和参考资格决定，不预先假定为 1155。

## 本地验证

```bash
./.venv/bin/python -m pytest -q tests/test_final_evaluation.py
```

仅使用 synthetic/disposable fixture：频带与归一化、Hann 和插值/边界、中位谱稳健性、lag 正负方向/边界/并列、有符号 PCC、MAE 复用 lag、35 点包络、中央独立投影、按记录时间选窗、退化与非有限失败、显式分母、sample SD、完整覆盖及防覆盖。另以 CPU 合成数据和替代模型走通运行入口：验证三 seed 完成、产物哈希、第二 seed 非有限失败时的准确 row/checkpoint 记录、终止后续 seed 和总体汇总。2026-09-27 执行结果为 `20 passed`；未访问真实信号或运行 GPU 推理。实际 GPU 推理和真实 test 指标待用户执行，不以 CPU 定向测试替代真实运行验收。
