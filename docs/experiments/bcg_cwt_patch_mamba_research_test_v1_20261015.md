# BCG–CWT 固定 checkpoint research-test 评价

协议 ID：`bcg-cwt-patch-mamba-research-test-v1-20261015`。

当前状态：两个固定 checkpoint 的完整 research-test 已完成并关闭，结果与来源见[完成记录](bcg_cwt_patch_mamba_v1_results_20261015.md)。本次评价由 2026-10-15 用户授权执行；实际执行合同以运行目录中的协议副本、manifest 和源码快照为准。

## 固定评价对象

训练来源：`runs/bcg_cwt_patch_mamba_v1/seed_20261013_run02`，训练源码提交 `0fe6e47f87d892c70ca0b93187b56cb1164a00b0`，seed=20261013，训练已于 epoch 30 早停。

固定两个 validation-selected checkpoint，评价前由完整 history 与完成回执重新核对其选择身份，并将文件 SHA-256 写入 `evaluation_manifest.json`：

- 主结果 `best_rr`：epoch 6，validation Local RR MAE=0.554724507640297 bpm。
- 辅助结果 `best_loss`：epoch 3，validation loss=0.25404847051495705。

两者分别报告，主结果身份由 validation 选点确定；test 不参与再次选 checkpoint、调整训练或更改阈值。

## 数据与计算合同

使用既有完整 research-test：2,310 窗口、8 名受试者，sample seed=20260612，行顺序哈希 `184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8`。读取既有冻结 input-only W cache，manifest 哈希为 `5d43ecf34596d5a6dd7cbaba75d91f9b7cbbb00214ae7594a4755e2afe510745`，复用 `TfV1ResearchTestCacheReader` 的身份校验。该测试集已被项目历史研究使用，本次属于复用的 research/development evidence。

BCG、呼吸目标、预处理、100 Hz/18000 点几何和指标口径保持训练及既有评价实现。CWT 读取完整 97×360 网格，先比对缓存实际频率与 checkpoint，再加载权重；batch=64、官方 Mamba2、bfloat16 AMP、float32 输出。检查 test 与 train/validation 受试者隔离、逐批行身份/顺序、所有输入/目标/预测有限性及指标资格。指标使用 `evaluate_task_predictions(include_test_only=True)`，包含 RR、包络、PCC、IBI、coherence、nDTW；汇总沿用逐窗口口径，分别保存两个模型的分母和质量标记。

## 执行与产物

`scripts/eval_bcg_cwt_patch_mamba.py prepare --run <训练目录> --output <新目录>` 只读取训练来源，保存评价清单、协议副本与源码快照；此步骤不读取 test cache 或 test 目标。

`evaluate --output <准备目录> --device cuda:0 --confirm-research-test` 执行固定清单。通过 `CUDA_VISIBLE_DEVICES=1` 选择物理 GPU1。评价使用独立 identity，原训练和历史评价产物保留。已有 attempt 拒绝自动重复执行；失败记录原位保留。

输出包含 `evaluation_manifest.json`、数据访问记录、test 行表、每个 checkpoint 的预测/参考 NPZ、逐样本 metrics、汇总、质量标记与哈希回执，以及固定两项的 `comparison.csv`。本专项仅开放上述两个 checkpoint 的本次 research-test，不扩大到其他训练或候选。

工程检查：`python -m pytest -q tests/test_bcg_cwt_research_test.py -rs`，4 passed，耗时 7.46 秒；使用人工输入和缓存替身验证确认门禁、完整 history 选点与平局规则、频率身份、97 行条件、完整指标和逐批行身份。`git diff --check` 通过。
