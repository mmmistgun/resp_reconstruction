# 原始 M4 单投影固定 checkpoint research-test

协议 ID：`m4-single-projection-research-test-v1-20261013`。用户于2026-10-13明确授权“开始test”，范围为本轮固定 M4/SP × seeds 20260811/20260812/20260813 的 research-test 评价及完整汇总，使用 GPU1。train/validation 已完成；训练实现、配置、选点规则和原训练协议保持冻结。

## 固定来源

训练代码提交 `fadef32e9b7fd5c9e5e4ea855d99e1291e92d994`。训练 session 为 `/mnt/disk_code/marques/resp_reconstruction/runs/m4_single_projection_v1/formal_v1_20261012`；session SHA-256：`fcc8154cc9d80131fea39abbbe7490f4eb8b79e26e303f5c1a35a3ee9a1d5f03`。完整 validation 汇总 receipt SHA-256：`21520734264149e0603a4fe24746882cd8cd9c842bc44939f4850086011bf48e`。

| Arm | seed 20260811 | seed 20260812 | seed 20260813 | 本轮处理 |
|---|---:|---:|---:|---|
| 原始 M4 | epoch 25 | epoch 5 | epoch 8 | 只读复用历史 test |
| SP | epoch 4 | epoch 5 | epoch 8 | 新增固定 checkpoint 推理 |

每个 checkpoint 均由完整 validation Local RR MAE 严格最小、并列最早选出。prepare 校验最终训练 history、best epoch/config/state、completion 和全部已记录产物哈希；固定实际频率、参数量、训练环境及 development 样本身份。allowlist 保存全部六个 checkpoint 的路径与 SHA-256，和本轮源码快照。

原始 M4 test 来自 `p1_components_v1/research_test_v1_20261007` 的冻结产物，实体位于项目 `runs/home_storage_migration_20261010/preserved_worktrees/worktrees_20261010/model-architecture-review/runs/` 下。历史 allowlist SHA：`8ac8b2305f66cd4f54f6a317f7c1dad9336bd88d393206fa84b5f300ad68fa47`；summary receipt SHA：`bae9f68d1834483dc9cd1b734f01d98ea5826186b9f1c41d8c423af7d962d29e`。逐项核对 checkpoint、配置、选点、数据/推理合同及成功产物；不重评已关闭的 M4。

## 评价合同

沿用 [P1 research-test 数据及指标合同](p1_components_research_test_v1_20261007.md)：test=2310窗口/8人、180秒/100Hz、sample seed=20260612，有序 row ID SHA：`184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8`。校验 dataset index 身份、与 train/val 受试者零交集、冻结97-scale W cache 的 manifest/数组/频率/row identity。W cache identity：`40a24df424b2ff9182cfcc6f5b7c12d287578b0df1ed1e25b56b7af1c7f73839`，manifest SHA：`5d43ecf34596d5a6dd7cbaba75d91f9b7cbbb00214ae7594a4755e2afe510745`。

batch=32、BF16、eval模式、完整尾批、官方 Mamba2、strict checkpoint加载；评价配置仅改设备和进度显示。SP 参数量1,047,045；M4参数量1,056,357。逐batch检查身份/顺序/shape/finite；预测、checkpoint及资格域内指标非有限时失败。prediction degeneracy 单独报告并保留分母。

五项主指标、逐seed、跨seed mean/sample SD（ddof=1）、SP−M4配对差/SD/改善seed数、逐受试者、subject-macro、Local RR tail、资格分母、参数量与全部原生 test 辅助指标（IBI/coverage、coherence、constrained nDTW、分层Spearman、lag）完整保存。六组齐全后才生成汇总。误差下降、PCC上升为改善；不按部分seed或中间test结果改变模型、矩阵、指标、阈值或选点。

该 test 曾参与既有研究开发，证据角色为 reused research/development evidence；本轮结果用于固定模型的跨split复核。参数化与优化路径同时变化，不能单独解释为额外投影表达能力的作用。

## 执行与产物

独立入口 `scripts/eval_m4_single_projection.py`；训练 source identity 内的文件保持原字节。prepare 不访问test数据源，之后 evaluate/summarize 使用本次授权对应的 `--confirm-research-test`。每cell独立attempt和互斥锁，失败保留并停止；经检查后使用 `--retry-failed` 创建新attempt，成功产物校验后复用。

```bash
cd /mnt/disk_code/marques/resp_reconstruction/.worktrees/m4-single-projection
PY=/mnt/disk_code/marques/resp_reconstruction/.venv/bin/python
TRAIN=/mnt/disk_code/marques/resp_reconstruction/runs/m4_single_projection_v1/formal_v1_20261012
TEST=/mnt/disk_code/marques/resp_reconstruction/runs/m4_single_projection_v1/research_test_v1_20261013_r3
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' "$PY" scripts/eval_m4_single_projection.py prepare --training-session "$TRAIN" --output "$TEST"
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES=1 "$PY" scripts/eval_m4_single_projection.py evaluate --allowlist "$TEST" --device cuda:0 --confirm-research-test
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' "$PY" scripts/eval_m4_single_projection.py summarize --allowlist "$TEST" --confirm-research-test
```

预期产物：allowlist及receipt/source_snapshot；三份新增evaluation attempt（环境、访问记录、test_rows、evaluation_config、逐sample metrics、原生summary、checkpoint身份和哈希receipt）；六组完整summary及sources。验收：三份新增成功receipt与三份历史引用均匹配allowlist，每组2310窗口/8人，样本/资格一致，五项主指标和规定资格域内辅助指标有限。CPU定向检查仅使用synthetic fixtures；正式test由本次用户授权开放。

准备记录：前两次 prepare 均在 test 数据访问前失败，依次为历史 P1 来源集合受归并新增10个H-only文件影响、历史配置的输出兼容路径被factory解析为实体路径。原冻结文件、合同及快照均未改变；本入口按历史allowlist逐文件校验原集合，配置比对保留历史输出路径字符串。失败记录保留于原目录及 `_r2`，新准备使用上述 `_r3` 独立目录，科研合同不变。5项synthetic CPU定向测试已通过。
