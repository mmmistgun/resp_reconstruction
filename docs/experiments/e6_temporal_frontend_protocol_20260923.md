# E6：20-Hz 学习解调时域前端专项方案

日期：2026-09-23。协议 ID：`e6-temporal-frontend-v1-20260923`。

状态：**P1 独立模型、严格配置、训练/汇总控制器、GPU/benchmark 入口和 synthetic CPU 验收已完成；early stopping 固定为 `min_epoch=30 / patience=15 / min_delta=0`。实现锁、GPU 阶段、真实训练和 test 尚未执行。**

## 1. 科学问题与候选身份

E5 已完成三 seed validation 和复用 research-test，`100 Hz→固定抗混叠直接降至10 Hz→学习编码` 的完整 package 在五主指标上整体退化。历史 C0/C2 说明 10 Hz 足以承载最终呼吸波形和 temporal latent；RTM train-only signal audit 则说明，这不等价于原始 100-Hz BCG 可以在任何可学习非线性之前直接降至 10 Hz。

E6 只回答：

> 固定 W0 的 W 条件分支、六层 BiMamba2、FiLM、decoder、数据、损失和训练选择器时，在 20 Hz 完成 learned carrier-sensitive filtering + nonlinearity、再显式降至 10 Hz，能否改善重建？

唯一候选为 `e6_tfe201_aa20_demod10_w0`；对照为同 seed 冻结 W0，内部标签 `e6_tfe000_patch_w0`。三个候选从头训练，不加载 W0 或 RTM checkpoint。E5 只作为历史机制参照，不进入 E6 selector 或阈值调整。

机器可读来源审计为 `docs/experiments/e6_temporal_frontend_source_audit_20260923.json`。E6 使用新协议、配置、输出根和未来实现锁，不复用或覆盖 E5 identity。

## 2. 结构定义

候选直接复用当前冻结字节身份下的 `resp_train.temporal.blocks.TemporalStem`：

```text
x: [B,1,18000] at 100 Hz
→ fixed Kaiser line-boundary FIR: down=5, 255 taps, cutoff=9.0 Hz
→ [B,1,3600] at 20 Hz
→ Conv1d 1→48, k21, p10, bias=false
→ channel-only LayerNorm per time step → SiLU
→ DWConv1d 48, k5, p2, bias=false
→ channel-only LayerNorm per time step → SiLU
→ Conv1d 48→96, k5, p2, bias=false
→ channel-only LayerNorm per time step → SiLU
→ fixed Kaiser line-boundary FIR: down=2, 127 taps, cutoff=4.5 Hz
→ [B,96,1800] at 10 Hz
→ 原 W0 六层 BiMamba2 / W-FiLM / refinement / decoder
```

第一阶段显式保留到 9 Hz，使 20-Hz 学习路径能够在最终 4.5-Hz 抗混叠之前将 carrier-sensitive 响应通过非线性变换为低频调制表示。第二阶段对已经完成学习变换的 96-channel latent 显式降至 10 Hz。两次固定算子都在 autocast 外执行；学习层遵循原 BF16 训练路径。

前端为 24,672 个训练参数，全模型为 1,235,738 个训练参数。参数量只用于资源报告，不是设计、匹配或淘汰条件。选择 exact `TemporalStem` 的理由是顺序已有 train-only signal audit、实现已有确定性等价与训练可行性记录、接口恰为 `[B,96,1800]`；这不声称该 stem 是 W0 的唯一最优实现。

E6 同时改变 W0 patch bridge 的采样、局部编码、归一化和边界语义，因此正式结果只能归因于完整 E6 frontend package。正负结果都不能把因果单独归到 20 Hz、某个卷积或 LayerNorm。

## 3. 冻结来源

- W0 来源锁：`docs/experiments/e4_w0_scale_aggregation_implementation_lock_20260917.json`，SHA-256 `464e073dbd5707a30575d606dec2a84dcd89a161945e537c463b214a15c2b493`。
- RTM signal substrate lock：SHA-256 `11bfcad00f4532d4bdfe1413a375b5f06f46eb8ac67dfcd475701872322fee69`。
- RTM CPU implementation receipt：SHA-256 `6ef3ca0d48e1ac819ff55bab4247d542c8bae0adfddb6951c8a026e81fea7872`。
- `resp_train/temporal/blocks.py`：SHA-256 `c43b8deca4e5ea185c2808a6e6c8b06172b77f811449cbb3d9e3249a3dd83435`。
- E5 validation summary manifest：SHA-256 `153b581b0ec1ca0ba5b286d990bdf248bb8e7a35949e79d36331b5cf6740d7f8`。
- E5 test summary manifest：SHA-256 `ebfea505978d9fa55e3fa22607b11778588661a65f57b8e2fb3759ecfdf072d1`。

实现锁准备时必须重新核验三 seed W0 checkpoint/config/validation metrics、train/validation W cache manifest 与实际所需文件、dataset index、上述 RTM/E5 来源和全部 E6 关键代码字节身份。工作树必须干净；实现锁排他生成且不得覆盖。

## 4. 控制变量

除 `base.frontend` 外全部固定：

1. research_v2 输入、target、admission、train/validation split 与 subject/session 隔离；train/validation 为 10,141/2,675 窗口、32/7 个 `samp_id`。
2. seed 为 `20260811/20260812/20260813`；model initialization seed 与 training seed 相同；sample seed 沿用 W0。
3. 97-scale W-only cache、CWT 分支、temporal mixer、active fill、FiLM 位置和 `gamma/beta=0.5/0.5`。
4. 六层 D=96 BiMamba2、refinement、coarse head、nonlinear decoder residual 与 Fourier 10→100 Hz 恢复。
5. `L_sync + 0.25 L_effort`、finite/eligibility/penalty 语义。
6. AdamW、physical/effective batch 128、accumulation=1、BF16、grad clip、weight decay 和 planned-update LR 日程。
7. 五主指标、secondary diagnostics、sample-direct mean 和受试者等权描述统计。

候选先完整构造 W0，再在独立子 seed `e6_temporal_frontend_aa20_demod10` 下替换 `base.frontend`。除 `base.frontend.*` 外的公共 state 必须与同 seed W0 逐 tensor 相等；固定 FIR taps 不进入 optimizer，六个 convolution/LayerNorm 参数必须进入原生分组并具有有限梯度。

## 5. 训练与 early stopping

- 最多 80 epochs、每 epoch 80 optimizer updates、planned 上限 6,400 updates。
- early stopping 固定监控完整 validation Local RR MAE：`min_epoch=30`、`patience=15`、`min_delta=0`。
- wait 从 epoch 1 累计，只有 `epoch≥30` 才允许停止；最早完成 epoch 为 30。
- LR 始终按 planned 6,400 updates 计算，不因提前停止重标定。
- checkpoint selector 为完整 validation Local RR 严格 `<` 最小，平局保留最早 epoch；保存 best 与 final。
- 不按 partial seed、中间结果、其他主指标或 test 修改停止、选择器、候选或阈值。

三个 formal seed 必须全部完成工程核验后才能一次性汇总。若某 seed 工程失败，保留失败 lifecycle，只修复工程问题并用新 attempt 继续；相同实现锁下已成功的 seed 不重跑。

## 6. 指标、判断与解释

并列报告 Whole RR absolute error、Local RR MAE、envelope trajectory MAE、global envelope modulation error 和 lag-aware signed PCC。对 error 使用 `(E6−W0)/W0×100%`，PCC 使用 `W0−E6`；正值统一表示 E6 更差。

输出包括两臂三 seed mean±sample SD、同 seed 配对差与方向、三 seed 均值之比、逐 seed×samp_id 配对差和受试者等权宏平均。不构造总分，不以参数量或运行速度补偿质量，不把三个模型 seed 当作独立受试者。

判断沿用 E5 的 material tolerance：error 相对 `0.5%`，PCC 绝对 `0.002`。结论基于完整五指标方向和属性取舍；“优于 E5”本身不构成替换 W0 的充分条件。

正式范围只含 train/validation。Validation 决定冻结前不建立 E6 test 入口，不读取 test input、target、prediction 或指标。

## 7. 实现、验收和阶段

```text
configs/e6_temporal_frontend/e6_tfe201_aa20_demod10_w0.yaml
resp_train/paper_evidence/e6_temporal_frontend_model.py
resp_train/paper_evidence/e6_temporal_frontend.py
resp_train/paper_evidence/e6_temporal_frontend_engineering.py
scripts/run_e6_temporal_frontend.py
tests/test_e6_temporal_frontend.py
runs/e6_temporal_frontend/
```

Synthetic CPU 测试至少覆盖：严格 spec、两个 decimator 与冻结 SciPy operator 等价、exact `TemporalStem` state、shape/finite、每个学习参数的梯度、同 seed W0 公共初始化、optimizer 分组、early-stop 轨迹、checkpoint selector、来源/锁漂移、不可覆盖 lifecycle、三 seed formal fixture 和汇总。

GPU acceptance 使用解析 synthetic input/target/W：三个 seed 各 batch-1 三步并核对公共 state；一个 batch-128 三步核对物理 batch、所有前端参数非零有限梯度、参数真实更新和显存上限。Benchmark 在独立进程中比较 W0/E6 的 batch-1 eval 和 batch-128 train，报告实测延迟、吞吐及显存；不将参数量推算成完整 FLOPs。

| 阶段 | 内容 | 状态 |
|---|---|---|
| P0 | 来源与结构方案 | 已完成 |
| P1 | 代码与 synthetic CPU 验收 | 已完成；专项与相邻回归共 97 tests passed |
| P2 | 实现锁 | 待 P1 提交后另行执行 |
| P3 | synthetic GPU acceptance 与 benchmark | 未授权 |
| P4 | 三 seed formal train/validation | 未授权 |
| P5 | 一次性 validation 汇总 | 随完整 P4 开放 |

当前只允许执行：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./.venv/bin/python -m pytest \
  tests/test_e6_temporal_frontend.py tests/test_e5_temporal_frontend.py -q

./.venv/bin/python scripts/run_e6_temporal_frontend.py check-config
```

实现提交并保持工作树干净后，下一步才是 `prepare-lock`；GPU、正式训练和 test 不在当前授权范围。
