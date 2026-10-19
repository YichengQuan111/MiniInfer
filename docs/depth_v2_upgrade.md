# V2 升级与验收（2026-10-15—19）

本轮依据 [2026-10-15 I-S-O 计划](MiniInfer_V1_V2_求职改进策略.md)逐项升级，每个小版本完成后单独 Git commit。起点为 `bf55478`：四个 CPU Worker、每 Worker 独占 ORT Session；这些行为需要保留。V2.4 必须实现完整 RGB8 GPU 预处理，局部归一化路径不能替代它。

## 证据与范围

恢复利用旧提交 `04a0ee6`、`9262958` 及中断备份中的可用实现，并重新与当前 SDK 集成。原始 [2026-10-11 分阶段记录](depth_v2_validation.md)及 `validation/v21*`、`validation/v22*` 是历史证据，不作为本轮二进制的性能结果。本轮数据独立保存于 `validation/v2_upgrade/`，不覆盖旧测量。

沿用 [负载与质量门槛](../data/depth/v2_experiment.json)，新增 [完整 GPU 预处理冻结契约](../data/depth/v2_preprocess_experiment.json)。图像仍是原 V2 开发/最终划分；本轮属于固定样本回归，不宣称重新建立了未见测试集。单机结果只代表本机、该负载和记录的运行条件。

| 小版本 | 交付 | 当前验收状态 |
|---|---|---|
| V2.1 | 定时回放、计数、分段 NVTX、输入上传字节、工具链与基线 | 通过，见下文 |
| V2.2 | 518/392、FP32/混合 FP16 配置与质量取舍 | 通过，选择518 FP32 |
| V2.3 | 拥有型异步句柄、背压、缓冲复用、1/2 GPU 槽位 | 通过，选择复用、单槽位 |
| V2.4 | A/B/C 三路径，完整 CUDA 基础/优化实现及实测 | 通过，选择完整C基础、单槽位 |
| V2.5 | 过载、错误、慢消费者、关闭与至少30分钟长稳 | 通过，1800秒36000/36000成功 |
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

## V2.4：完整 GPU 预处理

四种可选模式共用拥有型请求、槽位租用和同一模型契约：`cpu`（A）在CPU完成全部预处理；`cuda-partial`（B）在CPU做除255、resize和补边，再上传FP32 HWC，由GPU归一化并转NCHW；`cuda-full-basic` / `cuda-full-fused`（C）上传原图RGB8，在GPU完成全部像素运算并直接写入TensorRT输入。基础实现每线程处理一个通道，融合实现每线程处理RGB三通道以复用坐标计算；两者都严格在半像素双线性插值前除255，并按零RGB补边后归一化，禁止改变舍入语义的FMA。

完整C路径为每个槽位预留6,291,456字节RGB8设备缓冲及同容量页锁定缓冲，按原图上限而非518/392模型面积定容。B另有FP32 HWC设备缓冲；原FP32输入页锁定区为兼容 `run_raw` 保留。每个CPU工作区现有7个固定块，包含B的HWC暂存区；A/C也保留该工作区容量，在资源报告中照实计入。复用开启后稳态不再调用 `cudaMalloc` / `cudaHostAlloc`。

Release CTest **38/38**，四种模式各自的单/双槽位、CPU/GPU复用四组合、双图交错及拥有型句柄检查全部通过。CUDA合同测试覆盖28组形状/边界×3种kernel，包括518/392、横竖图、奇数补边、单像素/单轴、最大像素/边长与输出哨兵，最大绝对误差 **0**。[集成检查命令与结果](validation/v2_upgrade/v24/integration/checks.json)

开发8图×518/392×FP32/混合FP16×A/B/C基础/C融合全部通过冻结数值门槛；C相对同配置A的输入、raw和恢复输出最大误差均为 **0**，无新增排序错误。518为13/13，392为12/13，392在V2.2已有的跨尺寸退化仍存在，因此不重新取得选择资格。最终518 FP32的8图、四路径回归全部通过、排序17/17。校验读取的是送入TensorRT的实际设备NCHW，回读只在正确性工具启用，不进入性能回放。[开发集质量](validation/v2_upgrade/v24/quality-development.json)、[最终回归](validation/v2_upgrade/v24/quality-final.json)。四种路径还分别通过SDK模型包golden核对。

经用户授权启用Windows GPU调试接口后，对最终集成代码重跑 Compute Sanitizer 的 memcheck、initcheck、racecheck、synccheck 均为0错误/风险；完整C融合异步SDK测试覆盖单/双槽位与复用组合，memcheck为 **0字节泄漏、0错误**。原权限失败日志仍保留。[kernel设备检查](validation/v2_upgrade/v24/sanitizer/memcheck.log)、[异步SDK设备内存检查](validation/v2_upgrade/v24/sanitizer/sdk-memcheck.log)

CUDA events微基准在上传完成后计时，每进程预热20次、测200次，各形状/图像独立运行3次。518、1024×768小图的C基础平均耗时逐轮0.01516/0.01690/0.01781ms，C融合0.01461/0.01337/0.01219ms；1920×1080大图对应0.01560/0.01436/0.01400ms与0.01307/0.01324/0.01228ms。392也完成相同比较，某轮融合更慢，说明极短kernel受时钟/调度影响明显；这些数值不包含H2D、CPU工作和模型，不能替代端到端收益。[全部独立kernel结果](validation/v2_upgrade/v24/micro/518-v2_indoor_0-1.json)

小图原RGB8为2,359,296字节，低于518 FP32的3,219,888字节；大图RGB8为6,220,800字节，反而更大。完整C的动机是迁移CPU像素处理，并不保证传输量减小。回放单列实际H2D字节及H2D/kernel/model/D2H CUDA events，host预处理、槽位等待、输入页锁定复制、同步等待与后处理另列；同步等待和设备工作重叠，不相加。

对三个kernel分别实际运行Nsight Compute，均收到 `ERR_NVGPUCTRPERM`，未采到硬件计数器。因此本轮不报告occupancy、DRAM吞吐或roofline结论；被Profiler附着影响的耗时不作为正常计时数据。[原始错误](validation/v2_upgrade/v24/ncu/preprocess_scalar.log)。正常性能、Nsight Systems时间线与计数器尝试分开运行。

固定518 FP32、全部缓冲复用、单GPU槽位，各项3个独立进程。正常/过载分别20/80请求每秒、正式50秒；单并发为1个SDK Worker、1个客户端、最多1个未交付请求，10请求每秒、正式20秒。各轮预热2秒。正常负载四种模式均3000/3000成功、无拒绝/失败/生成漏发，正常与过载均无已接收请求失败；单并发每轮200/200成功。

| 路径 | 正常SDK P99（ms） | 过载成功QPS | 过载Busy数 | 单并发SDK P99（ms） |
|---|---|---|---|---|
| A：CPU | 29.29 / 34.04 / 35.62 | 63.94 / 61.14 / 59.28 | 802 / 942 / 1035 | 46.95 / 54.80 / 45.32 |
| B：局部CUDA | 35.14 / 33.16 / 35.04 | 59.78 / 58.94 / 58.34 | 1009 / 1052 / 1082 | 54.96 / 53.32 / 45.33 |
| C：完整CUDA基础 | 24.80 / 24.24 / 23.80 | 65.90 / 66.82 / 66.30 | 703 / 658 / 684 | 45.95 / 45.66 / 45.79 |
| C：完整CUDA融合 | 25.25 / 27.28 / 26.98 | 67.22 / 66.36 / 66.46 | 638 / 681 / 675 | 45.56 / 46.55 / 46.09 |

单并发10请求/秒与冻结正常负载的并发度不同，不用其P99替代20请求/秒的SLO；低占用下GPU频率更低。原单并发C融合测试受会话中断影响，其不完整批次保存在 `interrupted-single-cuda-full-fused/`，不进入汇总；上表使用10月16日重新完成的3个独立进程，其余主矩阵为10月15日测量。逐轮计数、各阶段P50/P95/P99和遥测见 [路径决策](validation/v2_upgrade/v24/path-decision.json)及 [消融汇总](validation/v2_upgrade/v24/ablation-summary.json)。

| 路径 | 正常进程CPU占用，100%=1核（逐轮%） | 正常稳态RSS最大峰值（MiB） | 单槽位设备I/O（字节） | 单槽位页锁定I/O（字节） |
|---|---|---:|---:|---:|
| A | 46.82 / 51.32 / 52.79 | 653.89 | 4,293,184 | 4,293,184 |
| B | 53.36 / 50.72 / 52.68 | 653.52 | 7,513,072 | 4,293,184 |
| C基础 | 36.45 / 36.50 / 36.28 | 566.62 | 10,584,640 | 10,584,640 |
| C融合 | 38.21 / 40.11 / 41.02 | 567.14 | 10,584,640 | 10,584,640 |

四个CPU工作区固定总容量均为144,381,568字节。C每轮正常负载实际上传4,961,080,500字节，大于A/B的3,219,888,000字节；C降低CPU像素处理开销，付出了RGB缓冲与传输成本。所有进程关闭后请求、预算、设备/页锁定I/O和CPU工作区均归零。上下文静态需求、应用I/O、RSS及设备总显存仍分别报告。

正常负载A的逐轮平均SM频率为2315/1802/1744MHz，C基础为2161/2153/2203MHz，C融合为2016/1774/1690MHz。频率、温度和跨时段条件明显不同，因此不能将全部端到端差异归因于kernel实现，也不能宣称基础kernel普遍快于融合kernel。依本轮规则，在质量与正常负载门槛均合格的配置中，选择三轮最差SDK P99最低的 **`cuda-full-basic`**；融合实现保留为可选项。

对选中C基础另做双槽位3轮正常/过载复测：正常SDK P99为 **26.34 / 26.74 / 29.26ms**，全部1000/1000成功；过载成功QPS为 **75.48 / 69.14 / 67.34**，Busy为225/541/632，已接收失败为0。其正常RSS峰值593.87MiB，设备/页锁定I/O各21,169,280字节，上下文静态需求313,576,448字节。双槽位取得较高过载吞吐，但该复测正常平均SM频率仅1194/1195/1221MHz，不能从非锁频实验推出精确并行收益。按正常尾延迟和资源成本，本轮仍选 **1槽位**，保留需要更高容量时复测双槽位的入口。[槽位决策](validation/v2_upgrade/v24/slot-decision.json)

独立Nsight Systems实测A/B/C基础/C融合分别捕获45466/48960/52380/52380个kernel；A/B的CPU像素预处理与GPU活动交叠约1003/969ms，C已经迁移像素运算，此项为0。选中C的双槽位时间线中两个流实际kernel交叠仅 **7.00ms/4.13秒**，不能把两个stream视为模型充分并行的证据。[双槽位实际时间线](validation/v2_upgrade/v24/timeline-selected-slot2.json)。

完整测量使用 [冻结源码树](validation/v2_upgrade/v24/source-tree.txt)和独立构建；各报告记录可执行文件、模型包和输入哈希。运行入口 `tools/run_depth_v24.py` 支持 `--mode`、`--load` 和 `--slots`；最终显式运行配置保存在 [v2_selected.json](../data/depth/v2_selected.json)，SDK不会自动加载该文件。V2.5将使用该配置重新进行输出一致性与长稳验收。

## V2.5：背压、结果检查与长稳

最终配置固定为518 FP32、C基础、1槽位、4个SDK Worker、8个客户端、队列4、128MiB请求预算，主机/设备缓冲均复用。回放v4在启动时建立每图独立基线，之后每个成功结果检查尺寸和64点位模式，每128轮图像序列对接纳的各图执行完整哈希。基线调用与预热单独记账，输出检查位于SDK终结发布之后，不计入CUDA events；它仍产生真实的客户端CPU与交付开销。

SDK每秒采样请求守恒、预算、当前/峰值I/O和分配计数，正式负载开始前同步写入带标记的边界样本。审查发现原周期采样会漏掉首个周期之前的分配增长，已修复并新增“早期增长、缺失/迟到/重复边界、允许预热分配”回归。严格审计重新读取CSV，核对报告统计与完整哈希检查次数；RSS使用60秒窗口，中位数增长及充分采样尾窗均受64MiB门槛约束，所有阶段的采样RSS峰值不超过2GiB。

同一冻结代码Release CTest **39/39**；CPU Debug、ASan/UBSan、TSan各 **7/7**。统计单测15项、稳定性门禁单测20项包含在CTest中。最终C基础的真实GPU压力检查覆盖单/双槽位，各有执行中放弃与析构后交付、ready字节预算、四线程提交与双线程关闭竞争，共6场景；全图逐位结果、请求ID与H2D对应、账本及回收均通过。Compute Sanitizer memcheck同时通过，**0错误、0字节泄漏**。[Release](validation/v2_upgrade/v25/frozen-ctest.log)、[CPU检查](validation/v2_upgrade/v25/cpu/checks.json)、[GPU场景](validation/v2_upgrade/v25/selected-stress.json)、[设备检查](validation/v2_upgrade/v25/selected-stress-memcheck.log)

每轮预热2秒、正式50秒，各场景3个独立进程；慢消费者在结果ready后延迟200ms再get。成功吞吐按窗口内终结发布计数，尾部drain结果仍计入总成功数；过载允许Busy，已接收失败、输出差异与回收失败仍不允许。

| 场景 | 三轮SDK P99（ms） | 三轮成功QPS | 三轮成功总数 | 三轮Busy数 |
|---|---|---|---|---|
| 正常20请求/秒 | 26.33 / 26.22 / 26.35 | 20 / 20 / 20 | 1000 / 1000 / 1000 | 0 / 0 / 0 |
| 容量探针40请求/秒 | 18.84 / 20.20 / 20.95 | 40 / 40 / 40 | 2000 / 2000 / 2000 | 0 / 0 / 0 |
| 持续过载80请求/秒 | 31.32 / 32.16 / 32.11 | 73.40 / 70.34 / 70.42 | 3671 / 3518 / 3522 | 329 / 482 / 478 |
| 慢消费者，20请求/秒 | 42.90 / 42.09 / 43.39 | 10 / 10 / 10 | 500 / 500 / 500 | 500 / 500 / 500 |

四组均通过正式审计，无已接收失败、输出差异或生成漏发。40请求/秒未饱和，因此只称容量探针；80请求/秒已经出现背压，不追加160请求/秒实验。慢消费者可见ready结果持续占用额度，消费/关闭后归零。正常三轮replay P99为28.14/27.47/27.91ms，保留原50/55ms门槛。测量未锁频，40请求/秒的较低延迟不代表请求率增加必然降低延迟。[正常审计](validation/v2_upgrade/v25/final/normal-acceptance.json)、[容量探针](validation/v2_upgrade/v25/final/saturation-probe-acceptance.json)、[过载](validation/v2_upgrade/v25/final/overload-acceptance.json)、[慢消费者](validation/v2_upgrade/v25/final/slow-consumer-acceptance.json)

10月19日完成连续 **1800秒** 正式长稳：20请求/秒，**36,000/36,000成功**，Busy、失败、超时和生成漏发均为0；SDK P50/P95/P99为24.41/26.12/**26.85ms**，replay P99为 **27.06ms**。包括40次预热的36,040次结果抽检、296次周期完整哈希均无差异；8次启动基线另计。30个逐分钟窗口均为20.0成功请求/秒，SDK P99范围23.07–29.19ms，最大replay P99为29.30ms，没有平均这些百分位。[完整严格审计](validation/v2_upgrade/v25/final/soak-acceptance.json)、[逐分钟延迟](validation/v2_upgrade/v25/soak-latency-windows.json)

共35,715个稳态RSS采样，实际覆盖1799.936秒，最大间隔74.60ms；所有阶段RSS峰值 **566.08MiB**。29个完整60秒窗口和最后约59.92秒充分采样尾窗的中位数增长均 **0MiB**，满足冻结64MiB门槛。设备/页锁定分配计数从正式边界到关闭均保持3次，CPU工作区28次；单槽位I/O设备与页锁定容量各10,584,640字节。关闭后inflight、ready、预算、I/O及主机工作区全部归零；上下文需求等历史字段保留，不能解释为关闭后仍持有这些资源。

本轮长稳通过全部原冻结质量、计数、SLO、采样密度、内存增长与回收门槛。10月16日的短测和10月19日长稳使用同一冻结二进制；未锁频、桌面后台未隔离、供电状态未采样，结果仍限本机条件。中断的容量探针、两次不完整长稳及冻结前3秒preflight保留为诊断，不混入正式结果，也不拼接时长；证据口径见 [V2.5目录说明](validation/v2_upgrade/v25/README.md)。

复现入口 `tools/run_depth_v25.py --executable BUILD/bin/depth_replay --package PACKAGE --images-list data/depth/replay_v2.txt --output NEW_DIR --preprocess cuda-full-basic --slots 1 --case soak`。它运行真实1800秒并调用严格审计；`tools/summarize_depth_soak.py AUDIT --output NEW.json`在核对审计和原始文件哈希后生成逐分钟统计。
