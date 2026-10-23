# M4与单投影SP双选点重训

协议ID：`m4-dual-selector-v1-20261013`。用户于2026-10-13授权重新训练并保存最低loss模型，明确最低loss不参与early stopping，且确认原始M4/SP两臂×seeds 20260811/20260812/20260813，共六组，沿用GPU1。本轮只开放train/validation；新的test评价需要固定checkpoint清单与专项授权。

## 科研合同

模型、模块隔离初始化、数据/预处理/split/sample seeds、task loss、优化器、batch32×累积4、BF16、最大80epoch/6400updates与原[单投影协议](m4_single_projection_v1_protocol_20261012.md)一致。两臂均从头训练，使用全新session。原训练、validation及research-test产物保持冻结。

每个完整epoch记录两个独立selector：

| Selector | 指标 | 同值处理 | 是否影响early stopping |
|---|---|---|---|
| rr | 完整validation Local RR MAE严格最小 | 保留最早epoch | 是，沿用原规则 |
| loss | 完整validation总loss：sync + 0.25×effort严格最小 | 保留最早epoch | 否 |

最低loss指validation总loss。early stopping仍为RR的min_epoch=30、patience=15、min_delta=0；loss下降不重置RR的wait，不延长训练，也不改变LR或梯度。两个selector都可选择任一已完成epoch；min_epoch仅限制最早停止时间。最低loss模型限定为本次实际训练完成的epoch范围，不声称覆盖未执行的80轮。

每个epoch保存不可覆盖的完整checkpoint，包含模型/optimizer/RNG、history、RR早停状态与loss选点状态。新增的`best_validation_loss`、`best_loss_epoch`独立恢复；RR或loss并列均保留最早epoch。结果`completed.json`中的`best_checkpoint`指向RR最佳checkpoint，`best_loss_checkpoint`指向实际保存的最低loss完整checkpoint（`epochs/epoch_.../checkpoint.pt`）。同一epoch可同时满足两者，复用同一文件，不覆盖历史快照。

正式结束后分别评价两个固定checkpoint的完整validation；若二者同epoch，只推理一次。保存`validation_rr_metrics.csv`、`validation_loss_metrics.csv`及各自原生summary。全部六组完成后，两种selector分别输出五项主指标的逐seed、mean/sample SD、SP−M4配对差、受试者等权、tail、资格分母及原生辅助指标；`selector_delta.csv`报告同arm/seed的loss选点减RR选点。`selection.csv`包含12条选点记录、checkpoint路径、RR与loss值、训练轮数和早停状态；参数量与同口径工程效率另表保存。

本轮设计已受到既有research-test观察影响，属于研究开发迭代。不能用两个selector的test表现再选每seed的获胜模型；当前阶段不访问test。新结果只解释实际参数化及选点合同，不能单独归因于投影表达能力。

## 实现与执行

独立代码为`scripts/m4_dual_selector_training.py`、`scripts/m4_dual_selector_runtime.py`和`scripts/run_m4_dual_selector.py`。复用原冻结模型、数据/训练epoch函数、validation与指标；原`run_epochs`保持不变。一次session/source snapshot冻结本轮依赖及协议。失败attempt保留；`--resume`从最后完整epoch恢复全部训练和selector状态。

```bash
cd /mnt/disk_code/marques/resp_reconstruction/.worktrees/m4-single-projection
PY=/mnt/disk_code/marques/resp_reconstruction/.venv/bin/python
SESSION=/mnt/disk_code/marques/resp_reconstruction/runs/m4_dual_selector_v1/formal_v1_20261013
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' "$PY" scripts/run_m4_dual_selector.py prepare --session "$SESSION"
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 "$PY" scripts/run_m4_dual_selector.py run --session "$SESSION" --device cuda:0
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' "$PY" scripts/run_m4_dual_selector.py summarize --session "$SESSION"
```

按seed交错运行M4、SP，共六组。每臂先做官方GPU synthetic验收，32×4累积两次更新，通过后训练。同一session/环境/arm的成功工程验收可复用。每组完整validation为2675窗口/7人，训练10141窗口/32人，来源和分母与原合同一致。GPU1由`CUDA_VISIBLE_DEVICES=1`限定，进程内为`cuda:0`。

CPU定向检查：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=. "$PY" -m pytest -q tests/test_m4_dual_selector.py
```

验收重点：loss持续改善也不能阻止RR规则早停；两selector并列保留最早；最低loss checkpoint确实存在且与history一致；中断恢复保持两种选点和训练轨迹；非有限loss显式失败；完整六组×双选点汇总校验通过。测试仅使用synthetic/disposable fixture。

2026-10-13：上述CPU定向测试6项全部通过（9.76秒），其中同一synthetic训练过程与原RR-only runner的停止epoch、RR选点及最终权重逐张量一致。GPU验收和正式训练由本次用户授权开放。
