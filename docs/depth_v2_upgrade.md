# V2 升级与验收（2026-10-15）

本轮依据 [2026-10-15 I-S-O 计划](MiniInfer_V1_V2_求职改进策略.md)逐项升级，每个小版本完成后单独 Git commit。起点为 `bf55478`：四个 CPU Worker、每 Worker 独占 ORT Session；这些行为需要保留。V2.4 必须实现完整 RGB8 GPU 预处理，局部归一化路径不能替代它。

## 证据与范围

恢复利用旧提交 `04a0ee6`、`9262958` 及中断备份中的可用实现，并重新与当前 SDK 集成。原始 [2026-10-11 分阶段记录](depth_v2_validation.md)及 `validation/v21*`、`validation/v22*` 是历史证据，不作为本轮二进制的性能结果。本轮数据独立保存于 `validation/v2_upgrade/`，不覆盖旧测量。

沿用 [负载与质量门槛](../data/depth/v2_experiment.json)，新增 [完整 GPU 预处理冻结契约](../data/depth/v2_preprocess_experiment.json)。图像仍是原 V2 开发/最终划分；本轮属于固定样本回归，不宣称重新建立了未见测试集。单机结果只代表本机、该负载和记录的运行条件。

| 小版本 | 交付 | 当前验收状态 |
|---|---|---|
| V2.1 | 定时回放、计数、分段 NVTX、输入上传字节、工具链与基线 | 通过，见下文 |
| V2.2 | 518/392、FP32/混合 FP16 配置与质量取舍 | 通过，选择518 FP32 |
| V2.3 | 拥有型异步句柄、背压、缓冲复用、1/2 GPU 槽位 | 通过，选择复用、单槽位 |
| V2.4 | A/B/C 三路径，完整 CUDA 基础/优化实现及实测 | 待验收 |
| V2.5 | 过载、错误、慢消费者、关闭与至少30分钟长稳 | 待验收 |
| V2.6 | 最终组合复验、模型包与使用说明 | 待验收 |

CPU/网络回归与 GPU 设备检查分开执行；Nsight 和正常性能计时分开。硬件计数器不可用时记录工具原始错误，不能以普通计时代替计数器验证。CUDA 同步和 TensorRT 上下文隔离依据 [CUDA 异步执行](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)与 [TensorRT 10.x 性能说明](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/performance/optimization.html)。

## V2.1：当前基线

完整 Release 构建及 CTest **36/36** 通过，包含 CPU/网络回归、ORT Worker 所有权、关闭和回放统计。测量使用 518 混合 FP16 包、CPU 预处理、4 个 SDK Worker、1 个 GPU 槽位、8 个回放调用线程、队列4、128 MiB请求预算。各轮预热2秒，正式50秒、20请求/秒；三轮 **3000/3000成功**，无拒绝、错误或生成漏发，成功吞吐均为20.0请求/秒，SDK P99依次 **37.13 / 36.13 / 35.83 ms**。保留原冻结目标，无需放宽。

当前 GPU 为 RTX 4080 Laptop（12282 MiB），驱动617.14、CUDA runtime12.4、TensorRT10.8、GCC13.3、WSL Ubuntu24.04。未锁频、桌面后台未隔离，供电状态未采样；每轮记录温度、时钟、RSS和设备总显存。设备总显存包含其他进程，不冒充本进程显存。详见 [验收摘要](validation/v2_upgrade/v21/acceptance.json)、[全部计时与遥测](validation/v2_upgrade/v21/baseline/report.json)及 [CTest日志](validation/v2_upgrade/v21/ctest-release.log)。

独立 Nsight Systems 2026.5.1 时间线真实采到12257个kernel和129次memcpy。60次调用的NVTX平均CPU预处理8.75ms，其中转换1.62ms、resize3.42ms、补边/归一化/布局0.57ms；范围包含不同开销且有嵌套，不与GPU时间相加。这支持分别测B的归一化迁移和C的完整像素处理迁移，不能将整个CPU预处理时长当作既得收益。[分阶段与GPU活动统计](validation/v2_upgrade/v21/profile-stats.csv)

回放记录每图RGB8候选上传量与实际FP32上传量。开发集原图693594～2073600像素，RGB8为2080782～6220800字节，覆盖小于和大于固定518 FP32输入3219888字节的情形。当前A路径实际上传FP32；成功后端完成统计不包含失败执行中可能已发起的复制。

复现：按部署指南准备模型后，运行 `tools/run_depth_replay.py --executable BUILD/bin/depth_replay --package data/generated/depth/packages/trt-fp16 --backend trt --images-list data/depth/replay_v2.txt --output NEW_DIR --duration 50 --warmup 2 --rate 20 --workers 8 --sdk-workers 4 --max-outstanding 8 --queue 4 --budget 134217728 --repeats 3`。Profile另行构建并设置 `MINIINFER_NVTX_INCLUDE`，使用 `nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none`；trace哈希已保存，二进制位于忽略的 `data/generated/depth/v2-upgrade/`。

## V2.2：模型尺寸与精度

支持518与392的静态契约、CPU/ORT/TRT推理、原始张量校验及模型包身份检查。Release CTest **37/37** 通过。四组配置各3×1000个正式请求，全部成功、无生成漏发，成功吞吐均20.0请求/秒；下表P99逐轮报告，不混合独立百分位。所有候选均通过其同尺寸的数值误差门限，但392新增一对排序错误（`v2_indoor_0` 的第3对），不满足跨尺寸零新增错误门槛。

| 配置 | 开发集排序 | 三轮SDK P99（ms） | engine（MiB） | 稳态RSS峰值（MiB） | 设备总显存峰值（MiB） | 选择资格 |
|---|---|---|---:|---:|---:|---|
| 518 FP32 | 13/13 | 34.94 / 33.63 / 35.49 | 100.70 | 676.22 | 1049 | 通过，选中 |
| 518 混合FP16 | 13/13 | 34.97 / 35.88 / 34.24 | 93.49 | 697.13 | 1105 | 通过 |
| 392 FP32 | 12/13 | 36.98 / 36.26 / 36.01 | 100.64 | 676.29 | 955 | 排序退化，不选 |
| 392 混合FP16 | 12/13 | 34.75 / 34.29 / 35.11 | 93.74 | 690.62 | 872 | 排序退化，不选 |

共同权重文件94.62MiB，两种ONNX约94.36MiB；实际字节数和SHA256见[验收数据](validation/v2_upgrade/v22/acceptance.json)。设备总显存包含桌面等其他进程；应用固定I/O显存为518的4.09MiB或392的2.34MiB，不能替代上下文、权重和库内存。完整逐轮遥测、上传量、CPU阶段和启动时间保留在各 `replay-*/report.json`。

按已冻结规则，在精度合格配置中选择“三次独立重复的最差SDK P99最小者”，因此本轮选518 FP32。与混合FP16差距仅0.39ms，且时钟未锁定，不宣称普遍性能优势。只用开发集选择后，最终8张回归图的数值门限全部通过、排序 **17/17**，预处理张量最大误差0、原始输出最大绝对误差9.43e-5；未调整容差。[决策](validation/v2_upgrade/v22/decision.json)、[最终精度](validation/v2_upgrade/v22/quality-518-fp32-final.json)

所选模型已创建为 `data/generated/depth/packages/v2-upgrade-selected`，包含16个清单文件和最终验证证据；模型资产仍按既有规则放在Git忽略目录。[打包记录](validation/v2_upgrade/v22/package.log)。本轮测量使用冻结二进制和源码树，后续V2.3修改未进入这些测量。

## V2.3：异步所有权、复用与槽位

新增 move-only `DepthRequest`：提交时复制输入并预留数量与字节预算，FIFO 出队；已就绪但未消费的结果继续占用预算。`get()` 交付拥有型结果并归还预算，销毁句柄安全放弃结果；执行中的放弃请求在执行完成后释放预算。`shutdown()` 停止接收、排空已接收任务并释放后端资源，仍可从保留的句柄获取结果。CPU 预处理在取得 GPU 槽位前运行，四个 CPU Worker 与每 Worker 独占 ORT Session 保持不变；两个 GPU 槽位各有独立 engine、context、stream 和缓冲。

Release CTest **37/37**，CPU Debug、ASan/UBSan、TSan 各 **7/7** 通过。TSan 首次因系统地址空间布局失败，保留原错误并使用仓库既有的 `setarch x86_64 -R` 方式复测通过；不声称第三方 GPU 库经过 CPU Sanitizer 插桩。真实 GPU 测试覆盖1/2槽位、主机/设备复用四种组合、双图交错、全图一致性及关闭/析构后的句柄交付。[CPU检查](validation/v2_upgrade/v23/cpu_checks.json)、[GPU检查](validation/v2_upgrade/v23/async_gpu.log)、[Release](validation/v2_upgrade/v23/release_ctest.log)

固定518 FP32、4个SDK Worker、8个客户端、队列4，各轮预热2秒、正式50秒；每项3次独立进程。分配策略按原冻结实验逐步消融：

| 配置 | 请求率 | 三轮SDK P99（ms） | 三轮成功QPS | 三轮Busy数 |
|---|---:|---|---|---|
| 每请求分配，1槽位 | 20 | 49.15 / 49.14 / 47.33 | 20 / 20 / 19.98 | 0 / 0 / 1 |
| 仅复用GPU/页锁定缓冲，1槽位 | 20 | 35.56 / 35.95 / 35.06 | 20 / 20 / 20 | 0 / 0 / 0 |
| CPU/GPU均复用，1槽位 | 20 | 33.89 / 34.71 / 34.86 | 20 / 20 / 20 | 0 / 0 / 0 |
| CPU/GPU均复用，2槽位 | 20 | 33.81 / 34.51 / 33.67 | 20 / 20 / 20 | 0 / 0 / 0 |
| CPU/GPU均复用，1槽位 | 80 | 34.48 / 35.47 / 37.61 | 64.68 / 60.70 / 59.68 | 764 / 963 / 1015 |
| CPU/GPU均复用，2槽位 | 80 | 38.27 / 39.65 / 36.06 | 60.50 / 59.98 / 60.46 | 973 / 1000 / 976 |

每请求分配配置有一轮正常负载出现拒绝，因此不通过正常负载门槛。其余正常负载均通过，所有配置均无已接收请求失败、无生成漏发，关闭后计数守恒且无残留请求/缓冲。复用使每个单槽位进程的设备/页锁定分配各从2080次降为2次；四个CPU工作区启动时共24个预分配块。单/双槽位应用I/O设备与页锁定缓冲分别各4,293,184/8,586,368字节，上下文静态需求156,788,224/313,576,448字节，正常负载稳态RSS峰值约641.54/661.72MiB。这里的I/O分配、上下文需求、RSS和设备总显存是不同口径，不能相加冒充精确进程显存。[逐轮消融与资源](validation/v2_upgrade/v23/ablation-summary.json)

真实 Nsight 时间线识别出双槽位的两个推理流，但约4秒窗口内实际 kernel 交叠仅 **6.84ms**；CPU预处理与GPU活动交叠约 **1027.55ms**。单槽位也有约 **933.85ms** 的CPU/GPU交叠。结合双槽位未提高过载成功吞吐且资源翻倍，本轮采用 **CPU/GPU缓冲复用、1个GPU槽位**；保留可配置双槽位，不把创建两个stream等同于有效并行加速。[单槽位时间线](validation/v2_upgrade/v23/timeline-slot1.json)、[双槽位时间线](validation/v2_upgrade/v23/timeline-slot2.json)

复现入口：`tools/run_depth_v23.py --executable BUILD/bin/depth_replay --package PACKAGE --images-list data/depth/replay_v2.txt --output NEW_DIR`。测量使用独立冻结源码树和二进制；源码树标识见 [source-tree.txt](validation/v2_upgrade/v23/source-tree.txt)，二进制和输入哈希保存在各轮报告中。
