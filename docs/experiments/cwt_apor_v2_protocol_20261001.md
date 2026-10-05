# CWT时频信息与频带作用：APOR/A0 v2协议

日期：2026-10-01。协议：`cwt-apor-v2-20261001`。

状态：按用户要求停止原W0实验，修订为APOR/A0；14项CPU定向测试通过（12+2分次，24.44 s及19.17 s）。用户随后明确要求“修订完成后启动”，授权新的APOR GPU验收及通过后自动执行正式train/validation。矩阵为20配置×3seed=60个科学cell；research-test关闭。实际阶段以新会话manifest和运行日志为准。

## 1. 研究对象与矩阵

主模型为完整APOR/A0：D96、六层双向Mamba2、140个patch、每patch五点条件采样、H65条件末端、双支有界FiLM、MLP 256点片段解码、Hann重叠合成及原Pi。原GELU/SiLU设置保持A0，参数量1,077,640。基础模型入口为`patch_apor_v1_model.PatchAporModel("A0", seed)`。

| 子矩阵 | 设置 | 唯一配置/训练预算 |
|---|---|---|
| TF-S2 | μ=6/13.4/20 × voices=4/8/12/24；至8 Hz、0.5 s | 12配置，36cell，含同期A0 |
| TF-S3 | A0其他参数；池化0.25/0.5/1 s | 新增2配置、6cell；共享A0 |
| TF-S4 | 基准μ/voices/池化；上限0.8/2/4/8/12/20 Hz | 新增5配置、15cell；共享8-Hz A0 |
| TF-S5 | L、H、L+H | 新增H一配置、3cell；L共享C_0p8，L+H共享A0 |

TF-S6/S7使用本轮A0自己的validation-selected checkpoints。W0中已训练的cell仅保留在其历史实验身份，不进入A0矩阵。当前预算按A0同期三seed重训练计算；历史A0复用须另完成缓存、数值执行和来源合同审计。

## 2. 统一训练停止合同

- 最大80epochs；early_stopping_enabled=true。
- min_epoch=30、patience=15、min_delta=0。
- 完整validation Local RR严格改善才重置patience；并列checkpoint取最早。
- 每epoch80次optimizer更新；学习率始终按6400次planned updates计算，不随实际早停轮数压缩。
- 最终checkpoint绑定实际停止epoch和实际更新数，best checkpoint仍按完整已运行validation轨迹选点。未达到合法停止条件的中断history拒绝作为完成结果。
- 原batch128、accumulation1、BF16、AdamW、LR3e-4→3e-5、warmup0.05、clip1、`L_sync+0.25L_effort`及数据资格/指标保持原样。

以上合同同时应用于全部20配置、训练验证、停止轨迹验收、最终checkpoint检查和导出恢复。正式cell不从被停止的W0 checkpoint续训。

## 3. 条件物理坐标与模型接口

每patch j的五个条件位置（原100-Hz采样索引）为：

`128*j + [0, 63.75, 127.5, 191.25, 255]`，j=0…139。

池化p点时，CWT第k帧中心为`p*k+(p-1)/2`，步长p；p=25/50/100对应720/360/180帧。新实现同步替换CoordinateSample的length/origin/step，保持线性取样、边界最近值延拓、坐标权重float32及原输入dtype回转行为。

W编码保持原卷积槽位核、scale mean和通道数；原生尺度/时间shape按各表示校验。五位置展开为480通道后使用原96通道投影及H65，不引入W0的三个temporal blocks。共享参数逐对象保留；基准A0的完整state_dict（含采样buffer）与原实现一致。

## 4. 模型无关产物复用

来源会话：`runs/cwt_time_frequency_v1/session_20260930T184031Z_812b424a1c92`。

已有20份train/val缓存、数据身份/公共参考、TF-S1信号分析、分层阈值、shift数组及案例规则均已完成。v2以只读引用复用，不复制大数组，不覆盖或补写旧产物。

复用前检查：原session/manifest/receipt、全部文件身份、依赖环境、数据/指标公共实现、CWT实现字节，以及每臂完整representation（含scales、实际频率、pool、shape、时间坐标）。基准名B→A0仅作身份映射，其余变换字段必须相等。半秒缓存不会被当作0.25或1秒缓存。任何不匹配显式失败。

原W0 GPU回执与训练结果不进入APOR验收；v2要求新的APOR FP32/BF16原生计算等价、不同shape的三步更新、batch128和APOR条件机制检查。95%显存硬上限、超过90%标记继续使用。

## 5. APOR的TF-S6/S7

干预矩阵保持18条件：FULL/NAT、FULL/FIXED，以及H=(0.8,8]和H2=(2,8]各自WINDOW_MEAN、三个共同SHIFT×NAT/FIXED。

APOR W分支实际只有`norm`这一层GroupNorm，固定统计只作用于它；前端时域GN不属于被干预的W分支。代码核对实际拓扑，拒绝套用W0四层GN清单。

Z由最后一层主干输出捕获，Z_prime由patch decoder入口捕获；gamma_raw、beta_raw及g/b均为`[B,96,140]`。特征时间坐标为patch中心`(128*j+127.5)/100`秒，另保存五位置条件采样坐标；输入CWT另存实际频率/时间坐标。保留同批FULL配对、保护输入、固定GN同源重放及正式指标曲线。

这些结果描述A0对条件信息的功能响应；WINDOW_MEAN与共同SHIFT仍不足以单独识别整体强度和跨尺度相对结构的贡献。

## 6. 历史A0候选参照的初步审计

已只读核对PATCH/APOR源会话`session_20260929T181607Z_77e89165fef0`的三个A0配置/结果：最大80轮、30/15/0早停、batch128、BF16和学习率设置一致；三个实例均完成30轮，selected epoch为8/13/13。

这仅确认训练停止合同兼容。v2目前不把三个历史cell登记为已完成；仍需核对实际缓存字节、FP32/BF16图等价、初始化、optimizer和数值环境。当前方案保留60-cell预算，若后续确立复用，则须在正式执行前明确修订为57次新增训练，不根据新结果临时替换参照。

## 7. 执行边界与入口

实现包：`resp_train/paper_evidence/cwt_apor_v2/`；配置：`configs/cwt_apor_v2/experiment.yaml`；输出：`runs/cwt_apor_v2/`。

入口分别为`run_cwt_apor_v2.py`、`run_cwt_apor_v2_engineering.py`、`run_cwt_apor_v2_formal.py`和`eval_cwt_apor_v2_research_test.py`。这些入口不修改原v1协议或产物。worker继承受管后台任务的进程组，停止主任务时避免留下独立GPU进程。

2026-10-01用户明确要求“修订完成后启动”，现开放新APOR GPU验收及通过后自动交接60-cell正式train/validation。只读复用既有模型无关产物；不开放research-test。任何工程失败均停止交接并保留失败生命周期。

CPU验证覆盖三seed原生A0完整state_dict与采样buffer一致、非零FiLM下的Identity主干fixture等价、各时间池化的物理坐标与梯度、APOR单W-GN的18条件配对及hook清理、30轮早停与最早并列选点、固定6400-update LR、80轮上限、早停final checkpoint校验、禁止W0验收跨模型继承。CPU前向fixture替换了Mamba，不能据此声称原生GPU数值路径已通过。

另已确认20份旧缓存metadata与新表示定义一致，CWT实现字节未变化；这里只读核对metadata，完整缓存字节与所有回执由后续`prepare-reuse`执行核验。
