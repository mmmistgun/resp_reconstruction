# ADV 融合因子实验：固定候选 research-test

日期：2026-09-23。协议 ID：`adv-fusion-factorial-v1-es30p15-research-test-20260923`。

状态：代码实现及CPU合成验证已完成，定向集合13项通过（44.95 s）；实际test cache、GPU推理与真实test汇总尚未执行，由用户执行。仅评价已完成validation矩阵的18个固定checkpoint。Test split曾参与既有研究，本轮证据标记为 **reused research/development evidence**，不作为未触及held-out的独立确认。

## 1. 来源与固定矩阵

来源：[完整validation结果](adv_fusion_factorial_v1_results_20260923.md)、[训练合同](adv_fusion_factorial_v1_protocol_20260922.md)。Validation summary为`runs/adv_fusion_factorial_v1/summary_full_es30p15_20260922_r1`，manifest SHA-256=`ab47bb367b74c476605d70d8e2ab7b17940fbc905dbd4d1ae610be6a72c76e11`。

训练执行代码身份=`f9381a7d1bf48a736f4a34e549d2bb440c9eb057e2f2e789a4e5364737870bce`。Test实现位于`resp_eval/fusion_test/`，使用独立配置目录`configs/adv_fusion_research_test_v1/`，不改变训练代码身份。

固定seed为20260811/20260812/20260813，下表为各自selected epoch：

| Arm | 方式与位置 | seed20260811 | seed20260812 | seed20260813 |
|---|---|---:|---:|---:|
| A | 前融合拼接 | 8 | 17 | 18 |
| B | 后融合拼接 | 12 | 7 | 4 |
| C | 前融合FiLM | 9 | 17 | 4 |
| D | 后融合FiLM | 9 | 16 | 4 |
| E | 前融合双视图attention | 9 | 17 | 4 |
| F | 后融合双视图attention | 9 | 3 | 5 |

候选锁`candidate_lock_20260923.json`由固定validation来源只读生成：核对summary及各run哈希、config、完整history、初始化文件、代码/依赖和checkpoint文件；回放30/15/0/80停止规则及最早Local RR选点。锁记录每项arm、方式、位置、seed、训练停止epoch/回执、checkpoint path/hash及原训练identity。只保留整套18项矩阵，不根据test结果调整候选、checkpoint、阈值或early stopping规则。

候选锁文件SHA-256：`2c80d570168b0de907a348bd1a1bdd8b476203623fc83ba64462377dd0ea8add`。锁准备仅读取已有validation来源与checkpoint字节，未加载原始test数据或模型推理。

## 2. 数据与前处理

沿用冻结数据源、admission、BCG输入、target、mask、Pi和窗口direct mean。数据身份常数引用既有research-test协议和已完成ADV记录；锁准备不读取真实index或波形：

- 完整test为2310窗口、8个samp_id；sample seed20260612。
- Index SHA：`f65ae6524632187c7c2795bd7b5e25434afb649b5a04fe39084b70cc59f9b04f`。
- 有序little-endian int64 row-ID SHA：`184e9d6a934b6719a4b679ebf6224e20dda1101c1920ed5b9e22ea80f0f293e8`。
- 实际访问时核验index、train/val/test的subject及源文件隔离、admission数量、row-ID顺序和窗口身份。

本轮cache绑定当前候选锁，使用独立input-only输出identity。按ADV冻结97scales、100Hz log1p(abs(CWT))、同一FIR501/D10构建float32 `[2310,97,1800]` 特征；payload约1.503GiB。缓存保存数学定义、transform源码/依赖、selection、逐窗口input/feature哈希、全特征文件哈希和成功回执；不读取target数组，不执行模型。

## 3. 评价与访问顺序

`cache`、`evaluate`、`summary`均要求显式`--confirm-research-test`。用户执行命令表明本阶段操作意图；Codex代跑真实数据或GPU仍需当次明确授权。普通train/validation入口保持原范围。

每项先校验候选锁、训练配置、checkpoint字节哈希，再strict-load原`FusionModel`；核验所有checkpoint状态有限、固定FIR及训练identity，检查epoch与锁一致。默认CUDA、bf16、physical batch32，eval/no_grad，FIR保持float32 IEEE。采用原图及读出，不调整early stopping或重新训练。

**每个候选完成全部2310个input-only预测后，才开始实例化target dataset并读取target/mask。** 四阶段`access.jsonl`记录推理开始、推理完成、target开始、target完成及完整计数。Target阶段再次检查输入内容未变。非有限输入、target、预测、checkpoint或eligible指标均显式失败。

指标沿用原函数：五主指标及IBI/coverage、target分层包络Spearman、呼吸带coherence、constrained nDTW。原定义允许的target-ineligible NA保留；IBI误差需结合其可解释计数与coverage。合法NA不被静默过滤为更小seed集合。

每项输出逐窗口metrics及summary、sample/input/target/mask哈希、prediction哈希、selected checkpoint与cache身份、完整源码快照、环境、命令和生命周期。预测波形不单独落盘；预测内容哈希用于记录本次推理身份。

## 4. 完整矩阵汇总

要求每项恰有一个成功evaluation，所有尝试均闭合。核验18个锁定checkpoint/epoch、arm/方式/位置/训练停止epoch、同一cache/sample/precision/batch/code、文件哈希、访问阶段顺序/计数、行集合以及逐sample指标到summary的一致性。成功后拒绝同cache/entry重复评价；失败保留并用新attempt重试，同一entry文件锁防并发。更换cache名称不构成重评成功候选的授权。

输出：

- `per_seed.csv`：18行，五主指标、辅助指标及其分母，附训练selected/stopped epoch。
- `across_seed.csv`：全部`*_mean`列的三seed mean/sample SD、n_seeds、finite_seed_count；存在合法NA时保留三seed均值为NA。
- `contrasts_per_seed.csv`：五主指标240行，复用冻结训练阶段的因子对比函数。
- `contrasts_across_seed.csv`：80行，逐对比的三seed mean/sample SD。
- 来源manifest及完成回执，汇总不读取input/target数组或重放模型。

对比为FiLM−拼接、attention−拼接、attention−FiLM，在pre/post分别计算；位置为post−pre，交互为两方式位置效应之差。包括等权边际方式/位置效应。三个两两交互只有两个线性独立；三训练seed不作为独立人群。报告RR、PCC、trajectory和global modulation的取舍，不生成跨属性总分。历史W0/ADV保留原结果，本轮不追加其评价。Attention效应仍限于整个融合模块，不将权重解释为生理贡献。

## 5. 用户执行命令

在Bash执行，任一步失败停止当前矩阵并保留失败产物。每个输出目录必须事先不存在。运行期间保持本协议与执行源码固定。

```bash
set -euo pipefail
cd /mnt/disk_code/marques/resp_reconstruction/.worktrees/adv_fusion_factorial_v1
FUSION_PY=/mnt/disk_code/marques/resp_reconstruction/.venv/bin/python
fusion_test() {
  env -u LD_LIBRARY_PATH -u LD_PRELOAD \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    SSQ_PARALLEL=0 SSQ_GPU=0 NUMBA_CACHE_DIR=/tmp/adv-fusion-test-numba \
    "$FUSION_PY" scripts/eval_adv_fusion_factorial_v1_research_test.py "$@"
}

# 候选锁已生成；此检查仅读取锁文件与源码。
fusion_test check-lock

FUSION_TEST_ROOT=runs/adv_fusion_factorial_v1_research_test
FUSION_TEST_CACHE="$FUSION_TEST_ROOT/cache_es30p15_20260923_r1"
fusion_test cache --confirm-research-test --output "$FUSION_TEST_CACHE"
for arm in A B C D E F; do
  for seed in 20260811 20260812 20260813; do
    fusion_test evaluate --confirm-research-test --cache "$FUSION_TEST_CACHE" \
      --entry "${arm}_seed${seed}" --device cuda:0 --attempt r1 || exit
  done
done
fusion_test summary --confirm-research-test --cache "$FUSION_TEST_CACHE" \
  --output "$FUSION_TEST_ROOT/summary_full_es30p15_20260923_r1"
```

评价目录自动确定为`${FUSION_TEST_CACHE}_evaluations/{entry}/attempt_r1`。验收为独立cache成功回执、18项各2310个相同行/内容身份、完整访问日志、有限eligible主指标和一次完整矩阵汇总。最大CPU开销包含nDTW等辅助指标，实际耗时由用户执行记录确定。

工程验证命令仅使用disposable数据与CPU Mamba替身：

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  SSQ_PARALLEL=0 SSQ_GPU=0 NUMBA_CACHE_DIR=/tmp/adv-fusion-test-numba \
  "$FUSION_PY" -m pytest -q -p no:cacheprovider tests/test_adv_fusion_research_test.py \
  tests/test_adv_fusion_factorial_v1.py::test_factorial_contrast_sign_and_complete_matrix
```

实现哈希、候选锁哈希及实际CPU验证结果见`adv_fusion_factorial_v1_research_test_implementation_lock_20260923.json`。
