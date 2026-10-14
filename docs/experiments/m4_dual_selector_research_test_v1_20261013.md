# M4/SP双选点固定12-checkpoint research-test

协议ID：`m4-dual-selector-research-test-v1-20261013`。用户于2026-10-13授权GPU1评价本轮两臂三个seed的最低validation总loss模型，并与最低RR选点比较，重点检查20260811。因此固定本轮六个训练cell×rr/loss两个selector，共12个checkpoint；全部来源于本轮重训，不混用前轮test结果。

## 固定来源与选点

训练session：`/mnt/disk_code/marques/resp_reconstruction/runs/m4_dual_selector_v1/formal_v1_20261013`。session SHA-256：`511459634b9d6022bab019efaacf030bb4f8220dad27bff5fca79a31ba21dc74`；validation汇总receipt SHA-256：`40bad985b37c22e6cdc1bf35136b69328c3b52a4be179c1b0cdee537a1dc6fff`。

| Arm / selector | 20260811 | 20260812 | 20260813 |
|---|---:|---:|---:|
| M4 / rr | 25 | 5 | 8 |
| SP / rr | 4 | 5 | 8 |
| M4 / loss | 8 | 7 | 6 |
| SP / loss | 8 | 7 | 6 |

rr为完整validation Local RR MAE最早严格最小值；loss为完整validation总loss（sync+0.25×effort）最早严格最小值，未影响训练early stopping。prepare核验全部训练completion、哈希、配置、完整history、两种选点、实际checkpoint state与development样本；将12个checkpoint路径/SHA、epoch、selector、配置和环境冻结至allowlist，保存源码快照。不重新训练或依据test改变任何选点。

## 数据、推理及指标

复用已有[P1固定checkpoint test合同](p1_components_research_test_v1_20261007.md)：2310窗口/8受试者，100Hz×180秒，sample seed=20260612，有序row ID SHA为`184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8`。dataset index来源哈希与development保持一致；验证test与train/validation受试者无交集。冻结W-full cache=97尺度×360格，原reader校验manifest/shape/dtype/row identity/实际频率。

batch=32、BF16、官方Mamba2、eval模式、strict加载、保留尾批，配置只改运行设备和进度显示。逐batch身份/shape/finite检查，模型state、输入、目标、预测及资格域内关键指标非有限时显式失败，退化计数独立报告、保留分母。六组重训的模型/损失/评价口径不变，test没有参与训练早停。

保留五项主指标（Whole RR AE、Local RR MAE、包络轨迹MAE、全局包络调制误差、lag-aware signed PCC）、原生IBI/coverage、coherence、constrained nDTW、分层Spearman、lag与资格/退化记录。完整12组后才汇总：

- 每种selector分别报告六组逐seed、三seed mean/sample SD、SP−M4配对及其SD/改善seed数、受试者等权、tail和资格分母。
- 同arm/seed报告`loss选点−rr选点`，跨seed汇总差值mean/sample SD和改善seed数；误差负差、PCC正差为改善。
- 单独输出20260811的两臂×五指标选点差。结合逐受试者结果检查670/671的变化，并保留其余受试者。

本轮设计受此前research-test观察影响，证据角色为reused research/development evidence。结果是固定选点的开发阶段比较；不按test为每个seed择优混合两个selector，不将选点差异等同于单投影结构的独立因果效应。

## 执行

独立入口`scripts/eval_m4_dual_selector.py`，只新增本轮协议/入口/测试，冻结训练源码保持不变。prepare只读development产物；evaluate/summarize使用本次授权对应的`--confirm-research-test`。每cell独立attempt/互斥锁，失败保留并停止，检查原因后才使用`--retry-failed`；成功回执及产物哈希匹配后复用。

```bash
cd /mnt/disk_code/marques/resp_reconstruction/.worktrees/m4-single-projection
PY=/mnt/disk_code/marques/resp_reconstruction/.venv/bin/python
TRAIN=/mnt/disk_code/marques/resp_reconstruction/runs/m4_dual_selector_v1/formal_v1_20261013
TEST=/mnt/disk_code/marques/resp_reconstruction/runs/m4_dual_selector_v1/research_test_v1_20261013
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' "$PY" scripts/eval_m4_dual_selector.py prepare --training-session "$TRAIN" --output "$TEST"
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 "$PY" scripts/eval_m4_dual_selector.py evaluate --allowlist "$TEST" --device cuda:0 --confirm-research-test
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' "$PY" scripts/eval_m4_dual_selector.py summarize --allowlist "$TEST" --confirm-research-test
```

产物为allowlist/receipt/source_snapshot、12份evaluation（metrics、summary、test_rows、config、环境、访问及checkpoint身份）、完整summary及sources。验收须12组全部成功、每组2310窗口/8人、样本及target资格跨selector/arm/seed一致、来源不漂移且规定指标有限。两种selector的全部结果共同呈现。
