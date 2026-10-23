# M4-S1 四卡续跑与跨主机执行记录

日期：2026-10-21。适用设计：`m4-s1-cwt-parameters-v1-20261020`。

用户授权检查进度、清理无法继续的失败产物、恢复完整train/validation，并加入新服务器Ubuntu186并行。本次沿用[冻结设计](m4_s1_cwt_parameters_v1_design_20261020.md)、[正式合同](m4_s1_cwt_parameters_v1_formal_execution_20261020.md)及[JSON回执修订](m4_s1_cwt_parameters_v1_execution_amendment_20261020.md)。research-test未开放。

## 进度与恢复

启动前本机队列已停止，未发现存活worker；现有产物为21个完成cell、2个中断cell、61个未开始cell。21个完成cell的完整产物哈希已核验。两项恢复边界：

| cell | 已完成epoch | 已完成更新 | 恢复checkpoint SHA256 |
|---|---:|---:|---|
| `A_NoFrequencyBias_seed20260812` | 5 | 400 | `eb6672d3b45da7cda3278a925a812d93c66cbd7ff4422d9cfaf2ce1f9be89d0a` |
| `A_UniformTFReadout_seed20260812` | 12 | 960 | `8f3c02abe8b547291e310e4f5251b97bdf1ff3ba113b4d3b436831b8436c94c8` |

两个中断cell从最后完整epoch的模型、optimizer、RNG与选点/早停状态恢复。既有完整epoch继续保留；不将中断cell重新初始化，也不重跑21个完成cell。

## 清理范围

清理3个被后续成功attempt替代、无checkpoint且无成功receipt的首次启动失败目录，共8,682 bytes。原失败原因、JSON内容、文件路径、大小和SHA256完整保存于 `cleanup_inventory.json`；清理动作回执为 `cleanup_completed.json`。没有删除成功产物、可恢复checkpoint或完整epoch。不存在需要清理的大型损坏checkpoint。此项按用户本次明确清理指令执行，更新先前保留失败目录的执行约定。

## 四卡分工

assignment在继续训练前固定，不使用指标值分配：两个中断cell留本机；61个未开始cell按原84-cell计划顺序轮转分配四卡。21个完成cell退出待执行队列，剩余63项并集完整且互不重叠。

| worker | 主机 | GPU | cell数 |
|---|---|---:|---:|
| `local_gpu0` | 本机 | 0 | 17 |
| `local_gpu1` | 本机 | 1 | 16 |
| `ubuntu186_gpu0` | Ubuntu186 | 0 | 15 |
| `ubuntu186_gpu1` | Ubuntu186 | 1 | 15 |

两台机器均为双RTX 4070 Ti SUPER，使用各自项目 `.venv/bin/python`。worker逐cell调用冻结入口，每cell独立进程；失败写明退出码并停止所属队列，不跳过失败cell。源码/数据/环境仍按原入口检查。正式微批32、累积4、BF16、最多80epoch、每epoch80update及早停与Local RR选点合同均不变。

## 跨主机身份与路径

规范项目根为 `/mnt/disk_code/marques/resp_reconstruction`；Ubuntu186物理项目根为 `/data/disk2/marques/marques/resp_reconstruction`。两者使用相同相对session路径：

`runs/m4_s1_cwt_parameters_v1/acceptance_v1_20261020`

两端 `session.json` 字节身份保持 `87b201abb30dea105ca8c9ada5c2f48a33e1f5ebd47b78af11276f67e93740cc`。远端只复制冻结session元数据、resolved config、校准、train/validation数据回执及所需只读缓存；不复制本机cell来重新训练。远端物理文件位于远端项目根内。

新增执行适配 `scripts/run_m4_s1_host.py`：缓存记录仍保留原规范路径，读取时仅将项目根翻译为远端物理根；原完整来源检查继续校验session、manifest、数据/rows、表示、依赖和缓存字节。配置身份只将 `outputs.run_root` 的物理项目根规范化，所有数据/模型/训练字段原样保留。映射拒绝项目外路径、父目录跳转和符号链接越界。原冻结源码与回执不改写；适配器与worker单独保存快照和身份。

路径映射与原JSON兼容层的synthetic定向验证共 **6 passed（0.18秒）**，覆盖合法映射、越界拒绝、仅输出路径归一、真实配置差异拒绝、回执与文件哈希拒绝路径。远端启动前还须通过冻结源码/依赖、78份train/validation源文件完整哈希、公共数据回执和全部分配所需cache核验。远端不继承本机GPU验收回执，按本机设备在各臂训练前完成B及相应臂合成验收。

## 运行产物

两端相对调度目录：`runs/m4_s1_cwt_parameters_v1/distributed_v1_20261021/`。

- `assignment.json`及receipt：21个完成cell、四worker完整63项分工和授权。
- `deployment.json`及receipt：session、分工、worker、执行适配与测试快照的身份。
- `remote_preflight.json`及receipt：远端来源与缓存校验结果；成功后才允许远端worker启动。
- `<worker>/started.json`、`console.log`、逐cell进程/命令/完成或失败记录；整队列退出时保存完成或失败回执。
- `startup_verified.json`：四卡启动后的checkpoint和进程证据，状态以实际记录为准。

正式cell产物保存在其分配主机的session `cells/` 下，包含完整epoch、恢复状态、训练history、最优/最终checkpoint、逐窗口validation指标和完成清单。远端结果回收时只同步assignment标记为远端拥有的cell，并核验完成manifest；远端工程回执按主机保存，不混入本机同设备键的成功阶段目录。最终84-cell汇总需先核验两端完整并集。
