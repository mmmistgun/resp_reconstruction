# W0 FiLM gamma 系数训练验证协议

协议 ID：`w0-film-gamma-training-v1-20260918`。

状态：结果知情、validation-only 的训练可行性实验；设计、implementation 与 CPU 定向验证已完成，GPU acceptance 待用户执行。

## 1. 目标与证据来源

冻结 W0 的局部 inference 扰动显示：降低 gamma 路径系数在 Whole RR、Local RR、trajectory 和 $1-\mathrm{PCC}$ 上具有较稳定的改善方向，且 high target-modulation 层响应更强。该结果记录于：

- `docs/experiments/w0_film_local_sensitivity_results_20260918.md`；
- `runs/w0_film_local_sensitivity_v1/summary/summary_7b11a15d39e6_20260918T092051Z_99a66682abe9`；
- `runs/w0_film_local_sensitivity_v1/final/final_7b11a15d39e6_20260918T093454Z_78ad894bc739`。

本实验回答：改变训练期 FiLM gamma 系数后，网络重新优化得到的 validation-selected checkpoint 是否仍保留该方向。它不把冻结模型的 inference perturbation 当作训练结果，也不访问 test。

## 2. 固定训练矩阵

W0 融合写为：

$$
g=c_\gamma\tanh(\gamma_{\mathrm{raw}}),\qquad
b=c_\beta\tanh(\beta_{\mathrm{raw}}),\qquad
Z'=Z\odot(1+g)+b.
$$

固定条件：

| 条件 | $c_\gamma$ | $c_\beta$ | 角色 | 新训练 |
|---|---:|---:|---|---:|
| GAMMA_030 | 0.3 | 0.5 | 新增训练候选 | 3 seeds |
| GAMMA_040 | 0.4 | 0.5 | 局部敏感性优先候选 | 3 seeds |
| GAMMA_050 | 0.5 | 0.5 | 冻结 W0 基线 | 0，复用 3 个既有 runs |

正式 seeds 固定为 `20260811 / 20260812 / 20260813`，共 6 个新 formal runs。`0.3` 是用户在实现前加入的预注册点；矩阵一旦进入 GPU acceptance，不根据部分 seed 结果删减或新增系数。

首轮固定 $c_\beta=0.5$，不加入 beta 训练、gamma×beta factorial、自适应系数或更密集的 gamma 扫描。

## 3. 冻结训练与数据口径

除 FiLM gamma 系数外，复用 W0 `crd_tf102_w` 的完整训练合同：

- 相同 train/validation split、sample seed、`dataset_row_id` 与 W cache；
- 相同 full-12V W 输入 `[97,360]`、六层 Local BiMamba2、条件头和 decoder；
- 相同初始化规则，`model.initialization_seed=training.seed`；
- AdamW、相同 learning-rate schedule、loss、AMP BF16、physical batch `128×1`；
- 固定 80 epochs、每 run 6400 optimizer updates、不开 early stopping；
- checkpoint selector 为完整 validation Local RR minimum；
- 每个 selected checkpoint 保存完整 2675-window validation metrics；
- 不读取 test，不修改 checkpoint、cache、split、loss 或 metrics。

冻结基线为：

| Seed | Selected epoch | Run |
|---|---:|---|
| 20260811 | 13 | `runs/crd_tf_v1/formal/crd_tf102_w/seed_20260811/20260812_210725_400861` |
| 20260812 | 15 | `runs/crd_tf_v1/formal/crd_tf102_w/seed_20260812/20260812_223203_048130` |
| 20260813 | 14 | `runs/crd_tf_v1/formal/crd_tf102_w/seed_20260813/20260812_235740_452006` |

## 4. 实现约束

原 W0 默认构造必须继续等价于 $c_\gamma=c_\beta=0.5$，state-dict key 与参数量不变，既有 checkpoint 继续 strict-load。新训练配置必须显式记录：

- `film_gamma_coefficient`；
- `film_beta_coefficient`；
- 条件 ID、seed、代码 commit 与 implementation lock；
- resolved config、checkpoint、history、逐窗口 metrics、runtime 与 receipt。

正式入口只接受固定配置、两个新增系数和三个 seeds，不提供任意系数 override。输出根固定为：

```text
/mnt/disk_code/marques/resp_reconstruction/runs/w0_film_gamma_training_v1/
```

每个 condition×seed 只允许一个正式 run；已有目录时拒绝静默重复。失败现场保留，修复后必须更新 implementation lock，并使用新的实验身份。

## 5. GPU acceptance

正式训练前只运行一次 `GAMMA_030 × seed 20260811` acceptance：

- 1 epoch；
- 128 train windows、32 validation windows；
- physical batch 128、accumulation 1、BF16 AMP；
- 检查 forward/backward、非零且有限的输入/参数梯度、checkpoint/optimizer/history finite；
- 检查 resolved coefficient、模型参数量、输出 shape、prediction degeneracy 与 test access；
- acceptance 与 formal 必须来自同一 clean commit 和 implementation lock。

`0.3` 是离基线更远的新增点，因此用它覆盖 acceptance。acceptance 只作工程闸门，不形成科学效果证据。

## 6. 评价与配对统计

每个候选与同 seed 的 GAMMA_050 基线按 `dataset_row_id` 配对。主轴固定为：

- `whole_rr_abs_error_bpm`；
- `local_rr_mae_bpm`；
- `envelope_trajectory_mae`；
- `global_envelope_modulation_error`；
- `lag_aware_signed_pcc`。

报告：

- 每 seed 的 candidate/baseline mean 与 paired delta；
- 三 seed 的 mean ± sample SD 与方向计数；
- `samp_id` 级配对汇总；
- low/medium/high target-modulation 与 waveform-confidence 分层；
- selected epoch、完整 history、prediction degeneracy 和 finite 审计。

窗口存在重叠，不作窗口独立推断。若后续计算置信区间，只按 `samp_id` 重采样，并披露主体数为 7。

## 7. 预注册决策口径

沿用 CRD-TF-W v2 的质量候选门槛。相对 GAMMA_050，候选必须同时满足：

1. 至少一项 primary 达到实质改善：某个 error 相对改善至少 0.5%，或 PCC 绝对提高至少 0.002；
2. 被声明改善的 primary 至少 2/3 paired seeds 同方向；
3. Local RR 相对恶化不超过 0.5%；
4. Whole RR、trajectory、global modulation 各自相对恶化不超过 1.5%；
5. PCC 绝对下降不超过 0.003；
6. 生命周期完整、全部 formal 数值 finite、prediction degeneracy 为 0。

任一 error 三-seed mean 恶化超过 3%，或 PCC 下降超过 0.005，记为明显失败。若 0.3 与 0.4 均通过，只报告 tolerance-aware Pareto 与分层差异，不根据单一指标自动宣布赢家。

## 8. 阶段与停止边界

1. implementation：代码、固定 configs、tests、implementation lock；
2. acceptance：用户执行单次短程 GPU 工程验证；
3. formal：用户执行 6 个完整训练 run；
4. summary：完整矩阵核验、配对统计、中文结论与冻结清单；
5. test：关闭，需另行协议和用户授权。

Codex 可运行 synthetic/disposable CPU 测试。GPU acceptance、formal training 和长时间推理由用户执行；不得根据 partial seed 结果改变剩余矩阵。

## 9. 实现、命令与产物

实现文件：

- 固定 configs：`configs/w0_film_gamma_training/`；
- 模型系数接口：`resp_train/crd/tf_v1_model.py`；
- 训练闸门与汇总：`resp_train/crd/w0_film_gamma_training.py`；
- CLI：`scripts/run_w0_film_gamma_training.py`；
- tests：`tests/test_w0_film_gamma_training.py`；
- implementation lock：`docs/experiments/w0_film_gamma_training_implementation_lock_20260918.json`。

CPU 定向及相关回归共 92 passed。implementation lock 提交后，固定执行顺序为：

~~~bash
GAMMA_WT=/mnt/disk_code/marques/resp_reconstruction_w0_cwt_film
GAMMA_PY=/mnt/disk_code/marques/resp_reconstruction/.venv/bin/python

cd "$GAMMA_WT"

# 用户：一次 GPU acceptance。
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  "$GAMMA_PY" scripts/run_w0_film_gamma_training.py acceptance

# acceptance 通过后，完整执行固定 2×3 formal 矩阵。
for GAMMA_CONDITION in GAMMA_030 GAMMA_040; do
  for GAMMA_SEED in 20260811 20260812 20260813; do
    env -u LD_LIBRARY_PATH -u LD_PRELOAD \
      "$GAMMA_PY" scripts/run_w0_film_gamma_training.py formal \
      --condition "$GAMMA_CONDITION" --seed "$GAMMA_SEED" || break 2
  done
done

# 六个 formal run 全部完成后；路径顺序不限，receipt 决定 identity。
env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  "$GAMMA_PY" scripts/run_w0_film_gamma_training.py summarize \
  --runs /六个完成的formal_run目录
~~~

acceptance 最小验收：`gamma_training_receipt.json` 中 `status=passed`，resolved coefficients 为 `0.3/0.5`，梯度、history、checkpoint、optimizer 和 primary 全部 finite，prediction degeneracy 为 0，`research_test_used=false`。

每个 formal run 最小验收：80 epochs、6400 updates、2675 validation windows、selected checkpoint 存在且 finite、零 prediction degeneracy，并与 acceptance 保持相同 commit 和 implementation lock。summary 必须接收完整 6-run 矩阵，否则显式失败。
