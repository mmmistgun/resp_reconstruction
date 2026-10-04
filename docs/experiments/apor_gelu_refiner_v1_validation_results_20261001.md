# APOR：统一GELU、H64与Direct的validation完成记录

日期：2026-10-01（Asia/Shanghai）。协议ID：`apor-gelu-refiner-v1-20260930`。

状态：**9/9新增训练与完整12-cell validation汇总均已完成并冻结，失败0。** B0复用原A0三seed；三个独立候选均未通过预设工程保护线，`eligible_arms=[B0]`。本轮train/validation执行关闭，research-test尚未实施。

## 1. 主要判断

当前APOR保留混合激活与H65条件残差。统一GELU改善全局调制均值，但增加Whole RR误差；H64与H65的窗口均值接近，受试者等权下仍出现节律与调制取舍；Direct删除残差后，Whole RR、包络及PCC整体不利，支持保留条件残差映射。

这些结果回答的是当前结构下的三个独立改变。它们不能证明65是特殊最优宽度、混合激活普遍优于统一激活，或残差瓶颈是所有模型不可缺少的模块。H64初始化使用同seed H65前64个隐藏单元；比较及初始化边界见[执行协议](apor_gelu_refiner_v1_protocol_20260930.md)。

## 2. 模型、样本与checkpoint

| Arm | 对照内容 | 参数量 | Best epoch，按seed顺序 | 完成epoch |
|---|---|---:|---|---|
| B0 | 原A0/N0，混合激活、H65 | 1,077,640 | 8 / 13 / 13 | 30 / 30 / 30 |
| GELU | 五处普通隐藏层SiLU→GELU | 1,077,640 | 10 / 13 / 11 | 30 / 30 / 30 |
| H64 | 条件残差96→64→96 | 1,077,448 | 8 / 13 / 14 | 30 / 30 / 30 |
| DIRECT | 条件残差替换为Identity | 1,065,064 | 14 / 13 / 61 | 30 / 30 / 76 |

Seeds为20260811/20260812/20260813。每cell validation为2675窗口、7人，完整矩阵32100行，其中新增24075行。数据、split、损失、Pi、五项主指标与完整validation Local RR严格最小checkpoint selector均沿用原合同。

DIRECT seed 20260813在第61 epoch产生最优Local RR，随后等待15个epoch，在第76 epoch正常早停。该轨迹符合原min30/patience15/max80合同，未因中途网络断开重训或更换selector。

## 3. 窗口均值

表中为先计算每seed窗口均值、再取三seed均值。四项error越低越好，PCC越高越好；RR单位bpm。sample SD及逐seed结果保存在summary。

| Arm | Whole RR | Local RR | 包络轨迹 | 全局调制 | PCC |
|---|---:|---:|---:|---:|---:|
| B0 | 0.453363 | 0.526085 | 0.146643 | 0.165625 | 0.871033 |
| GELU | 0.468856 | 0.527205 | 0.147129 | 0.163876 | 0.871407 |
| H64 | 0.451804 | 0.528797 | 0.146174 | 0.165866 | 0.870273 |
| DIRECT | 0.468204 | 0.526503 | 0.151677 | 0.172278 | 0.868316 |

相对B0的变化，error为相对百分比，PCC为绝对差：

| Arm | Whole RR | Local RR | 包络轨迹 | 全局调制 | PCC |
|---|---:|---:|---:|---:|---:|
| GELU | +3.417% | +0.213% | +0.331% | −1.056% | +0.000374 |
| H64 | −0.344% | +0.515% | −0.320% | +0.146% | −0.000760 |
| DIRECT | +3.274% | +0.079% | +3.433% | +4.017% | −0.002717 |

GELU的调制误差2/3 seed改善，PCC仅1/3改善；Whole RR仅1/3改善。H64的Local RR虽然2/3 seed改善，三seed均值仍增加0.515%，说明不能仅凭方向票数判断总体质量。DIRECT的Whole RR及全局调制误差均3/3退化，轨迹/PCC均仅1/3改善。

## 4. 受试者等权结果与保护线

下表先计算每seed各受试者窗口均值，再对受试者等权，最后取三seed均值。

| Arm | Whole RR | Local RR | 包络轨迹 | 全局调制 | PCC |
|---|---:|---:|---:|---:|---:|
| B0 | 0.549703 | 0.627546 | 0.159056 | 0.188713 | 0.859111 |
| GELU | 0.590163 | 0.629080 | 0.159334 | 0.188596 | 0.859582 |
| H64 | 0.552494 | 0.618132 | 0.158707 | 0.190966 | 0.858069 |
| DIRECT | 0.591603 | 0.623021 | 0.164899 | 0.192912 | 0.855732 |

预设要求两种聚合口径内各项error相对退化不超过0.5%，PCC下降不超过0.002。未通过项为：

- GELU：窗口Whole RR +3.417%；subject-macro Whole RR +7.360%，后者三个seed均退化。
- H64：窗口Local RR +0.515%；subject-macro Whole RR +0.508%、全局调制 +1.194%。其中前两项接近阈值，但调制退化不只是舍入边界，且三个seed均退化。Macro Local RR反而改善1.500%，三个seed均改善，体现聚合口径下的取舍。
- DIRECT：窗口Whole RR、轨迹、调制及PCC未通过；subject-macro相同四项分别变化+7.622%、+3.673%、+2.225%、−0.003378。Macro Local RR改善0.721%，不能抵消其余属性损失。

因此保留B0。H64应描述为接近基准且存在属性取舍，不能据此声称65显著优于64。Direct仅减少12,576参数，约占基准1.17%，本轮质量变化不支持用这一规模的参数节省替换基准。统计只描述三个训练seed和当前validation人群，不作显著性或独立人群外推。

## 5. 网络中断与完整性核验

用户报告网络断开后，检查发现原调度父进程已退出，但两个worker在独立进程组内运行。worker 1完成4个cell并正常输出完成记录；worker 0继续完成5个cell。中途检查时已有8个成功cell，最后的DIRECT seed 20260813日志持续更新。

全部训练结束后，用户通过原CLI的`summarize`子命令补做完整汇总。旧dispatch缺少父进程完成receipt，原目录保持原状；本记录以各cell及summary的manifest/freeze作为完成依据，不补造调度退出码。

完成后只读校验全部12个来源cell及summary的manifest/freeze，核对12个固定checkpoint身份；从保存的逐窗口metrics独立复算三seedmean/sample SD与subject-macro，均通过。各cell的窗口数、row顺序及target资格一致，五项主指标全部有限，joint/envelope Spearman prediction degeneracy均为0。本次没有重新训练、推理或改选checkpoint。

## 6. 来源与后续边界

- Session：`runs/apor_gelu_refiner_v1/session_20260930T154903Z_6fec6cef5eff`。
- Session SHA256：`3c349e55f8f77c306607bfe19a3734afde3ce1380b6c5635679953fd2ac587b1`。
- Summary：session下`summary/attempt_20260930T175426Z_6500442256e9`，于北京时间2026-10-01 01:54创建。
- Summary manifest SHA256：`37381d5a81b80989aabec54e854e5f6e343665797c1be2dd43d9d08d9c3aa7ed`。
- `validation_decision.json`：`eligible_arms=[B0]`，`test_used=false`，保存全部12个fixed checkpoint身份。

本阶段完成并冻结。Research-test尚未实施，若开展，须使用另行定义的固定checkpoint附件及当次访问授权。当前结果也不回答480→96五点条件压缩是否必要；本轮只改变H65残差映射，五点条件与140-token结构均保持原定义。
