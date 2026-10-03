# RespDiff-THO 核心移植与CPU验收

日期：2026-09-29。分支：`codex/resp-diff`。依据：[论文与代码对照方案v2](respdiff_tho_migration_plan_v2_20260929.md)。

状态：核心移植和合成数据接入已实现，21项CPU定向测试通过（13.08秒）。原规模GPU验收、真实数据访问和正式训练尚未执行。当前CLI只提供来源核验与CPU合成流程。

## 已实现范围

| 文件 | 实现与证据 |
|---|---|
| `resp_train/respdiff/model.py` | FFT/plain两个来源profile，保留state_dict键、初始化顺序及时间步嵌入差异 |
| `resp_train/respdiff/diffusion.py` | 原归约噪声loss、FFT loss、50步DDPM、按样本身份固定噪声、逐轨迹均值 |
| `resp_train/respdiff/data.py` | segment soft-z/source min-max两种开发profile，重采样/低通、36块身份、完整重组、主体/记录隔离和重复区间统计 |
| `resp_train/respdiff/experiment.py` | 单次Adam更新检查、checkpoint严格回载与防覆盖、完整父窗口五指标、独立Local RR selector |
| `resp_train/respdiff/provenance.py` | 固定上游4个文件SHA-256；不导入有训练副作用的脚本 |
| `configs/respdiff_tho_v1/experiment.yaml` | 开发合同、原规模网络声明、小模型合成配置；正式归一化、预算和验证频率保留null |
| `scripts/run_respdiff_tho_v1.py` | check-source、synthetic-smoke；来源/配置/源码哈希、完成/失败receipt |
| `tests/test_respdiff.py` | 来源对照、DDPM、信号适配、临时ResearchV2 NPZ及CLI生命周期 |

source_plain用于验证来源行为，主移植网络采用source_fft。两个归一化profile均可测试，正式主配置尚未选择；实现可运行不代表归一化合理性已通过真实实验。

## 数值一致性与工程修正

- 两个profile在相同初始化seed下，全部来源state_dict键和值逐元素相同；相同输入/step的forward逐元素相同。
- 使用相同训练噪声和step，对照来源loss、全部参数梯度及一次Adam更新，按测试内显式容差通过。单通道噪声项仍为sum/L，FFT项仍为batch均值；重复batch测试验证该缩放。
- 完整50步采样使用相同初始噪声与每步噪声，与来源batch=2输出在rtol=2e-5、atol=2e-4内一致。同身份同环境重复采样逐元素一致；验证采样不消耗全局训练RNG。
- **来源batch=1兼容失败：**在本次PyTorch 2.12.0+cu130、NumPy 1.26.4环境中，上游以单元素torch tensor索引NumPy调度数组得到标量，随后`coeff1[0]`触发IndexError。测试保留来源失败证据，移植版用明确的torch标量系数完成采样；t=0不读取无用的前一步调度。
- CPU噪声按SHA-256派生的seed逐样本生成，再转模型device。固定噪声可消除batch拆分对随机数身份的影响，但原BN仍会受条件batch组成影响；不宣称预测跨batch不变。
- 原规模6层/hidden1024网络仅以meta tensor核对state_dict形状和参数总数与来源一致；实际CPU前向/反向/采样使用小RNN，卷积与50步调度保留。
- 所有新增实现要求FP32，未验收AMP、DDIM、条件缓存或轨迹并行。常量source min-max、非有限输入/输出/梯度/optimizer/checkpoint与FP32转换溢出明确失败。

## 数据与生命周期验证

18000→5400→36×150→5400→18000的长度、脉冲位置、已知呼吸频率、缓慢幅值变化和低通抑制通过合成测试。segment soft-z路径不重新归一化5秒片段；source min-max分别映射输入[-1,1]和target[0,1]。输入准备函数不接收target或其统计。

临时NPZ/CSV fixture实际调用ResearchV2WindowDataset，再由ChunkDataset包装；核对父dataset实际row身份，显式拒绝test、主体交叉、错误父row、缺块、重复、错序。父窗口重叠导致的重复区间按原采样权重保留并统计。

合成CLI测试执行一次小模型Adam更新，保存/回载checkpoint，使用DDPM50、N=1生成36块，重组180秒输出并计算最终五指标及独立的原Local RR selector。该fixture仅验证流程，没有真实train/validation质量含义。测试还验证相同目录拒绝覆盖、注入训练失败后保存failure.json并不写完成receipt。

## 复查命令

在新worktree执行，解释器复用主仓库现有环境；无需安装依赖：

```bash
cd /home/marques/.codex/worktrees/resp-diff/resp_reconstruction
PYTHONPATH=. /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python \
  -m pytest tests/test_respdiff.py -q
```

已执行结果：`21 passed in 13.08s`。这是与新增模块相关的最小集合，未运行全仓回归或既有冻结实验验收。

源码身份核验：

```bash
PYTHONPATH=. /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python \
  scripts/run_respdiff_tho_v1.py check-source
```

保存独立合成流程产物（输出目录必须尚不存在）：

```bash
PYTHONPATH=. /mnt/disk_code/marques/resp_reconstruction/.venv/bin/python \
  scripts/run_respdiff_tho_v1.py synthetic-smoke \
  --output /tmp/respdiff_tho_synthetic_01
```

产物包括execution.json、synthetic.pt、prediction.npy、chunks.csv、metrics.json与receipt.json。失败保留failure.json及已生成文件；换新的输出identity重新执行。receipt包含当前产物SHA-256，execution记录实际小模型、单轨迹设置、配置、依赖、源码与指标来源。

## 下一阶段边界

工程入口和正式训练runner尚待原规模资源合同与预算确定后补齐；目前没有读取真实数据的CLI命令。下一步是原规模150点网络的GPU资源验收，并在train/validation开发范围解决归一化桥接对照与充分训练规则。正式归一化、训练预算、验证频率、采样N与三seed矩阵冻结后才开放真实训练。独立test仍需匹配协议与当次授权。

本次仅新增隔离实现和文档，未修改既有loss/metrics、数据、split、checkpoint或W0/E9产物；无需重算旧结果。
