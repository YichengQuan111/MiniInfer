# MiniInfer V1 验收记录

本文件记录本机实际执行证据。各阶段只有对应检查通过后才标记完成。

| 阶段 | 状态 | 证据 |
|---|---|---|
| V1.1 环境与模型契约 | 通过 | 16 图 Python CPU FP32；518 静态 ONNX checker；真实 CUDA/TRT 执行；Python 29 项、CPU/TCP Debug 24/24、Release 24/24 |
| V1.2 ONNX / ORT | 通过 | 16 图分层对齐；前后处理逐元素一致；Debug 29/29；Release 28/28 加验收驱动测试通过 |
| V1.3 TensorRT FP32 / FP16 | 通过 | 两配置最终集各 8/8；真实 GPU 生命周期/并发/坏引擎恢复；全套 31/31 |
| V1.4 C++ SDK | 通过 | 主构建 32/32；CPU-only 27/27；SDK ASan/UBSan；安装后独立外部项目的三个模型包 golden 通过 |
| V1.5 TCP 深度接入 | 通过 | 全套 33/33；网络与旧 TCP 并发/关闭 ASan/UBSan 3/3；三个实际后端 TCP 输出与本地逐字节一致 |
| V1.6 部署与基线 | 未开始 | — |

前后处理、版本和复现方式见[部署契约](depth_deployment.md)。既有 CPU / D16 的历史验收与性能记录保留原义。

## V1.1：环境、数据与导出可行性

2026-10-11 在 WSL Ubuntu 24.04、RTX 4080 Laptop（12,282 MiB，驱动 617.14，计算能力 8.9）执行。使用独立虚拟环境，`pip check` 无依赖冲突；[环境和最小设备证据](validation/v11_environment.json)、[完整 Python 版本](validation/v11_pip_freeze.txt)已保存。最小网络 `y=x+x` 在 CUDA stream 完成后核对输出 `[2,-4,7,0]`，通过。

[Python 参考清单](validation/v11_reference_manifest.json)记录全部 16 张图、中间张量、模型源码与权重哈希。它验证基线可生成，近远排序与跨后端数值验收在后续阶段进行。[静态导出清单](validation/v11_export_manifest.json)记录 ONNX SHA-256 `a7d66386939681431d3b7c407b897d00cf205c44c607a33189eb4238bb7c9f80`，大小 98,941,039 字节，opset 17，输入 `[1,3,518,518]`，输出 `[1,518,518]`，checker 通过。

导出日志中的 TracerWarning 来自官方模型将固定 518 形状转为 Python 整数或布尔值；本版本明确仅接收固定形状，已核验导出输入输出，动态尺寸不在契约内。未安装可选 xFormers，执行使用 PyTorch 路径。

验证日志：[参考执行](validation/v11_reference.log)、[导出](validation/v11_export.log)、[设备执行](validation/v11_probe.log)、[Python CTest](validation/v11_python_tests.log)、[Debug 回归](validation/v11_debug.log)、[Release 回归](validation/v11_release.log)。Python 前后处理 15 项与统计 14 项测试在 WSL 通过；既有两套 24 项回归全部通过。

## V1.2：C++ ORT 和分层数值验收

[精度报告](depth_accuracy.md)记录门槛冻结和开发/最终集结果。最终集 8/8 张图通过，人工排序 Python 与 ORT 均 11/13，保留了基线中的两组错误；没有将模型本身的错误隐去。[ONNX 工件记录](../data/depth/onnx.json)将静态导出提升为已验收部署工件。

[Debug 全套](validation/v12_debug_tests.log)29/29；[Release](validation/v12_tests.log)28/28，随后新增的 [验收驱动测试](validation/v12_driver_tests.log)单独通过。ORT 关闭的独立 C++ 图像库构建与测试也通过。C++ 测试包含手算前后处理、输入边界、输出所有权、文件损坏/防覆盖，ORT 使用小型模型实测错误形状/名字和正常执行；12 项 Python 驱动测试覆盖损坏工件、错误退出、分层误差与失败报告。

## V1.3：TensorRT 单槽位与混合精度

FP32 engine 构建耗时 **15.608645 秒**，最终 FP16 v2 构建耗时 **151.955208 秒**；这是 `buildSerializedNetwork` 的实测主机耗时，包含构建期间设备工作，不包括之前的 ONNX 解析和之后的清单/文件写入。引擎大小分别为 105,586,948 和 98,029,604 字节。运行环境和哈希见[FP32 清单](validation/v13_fp32_engine.manifest)、[FP16 清单](validation/v13_fp16_engine.manifest)；完整层格式见同目录 `v13_*_layers.json`。

首版 FP16 的 raw 精度失败后，保持门槛不变，将 backbone 保留 FP32、卷积 head 使用 FP16；最终两个配置都通过固定最终集，详见[精度报告](depth_accuracy.md)。

[全套 CTest](validation/v13_tests.log)31/31；SHA-256 使用标准空串、abc、百万 a 向量验证，并由实际模型工件与 Python 哈希交叉核对。GPU 的[FP32](validation/v13_fp32_lifecycle.log)与[FP16](validation/v13_fp16_lifecycle.log)测试覆盖并发串行、结果跨销毁存活、配置标签错配、SHA 错配、反序列化失败后的资源回收和有效模型重载。日志中故意损坏 engine 的 TensorRT 报错是负例证据。

CUDA stream 同步之后才读取 events 和拷贝输出。WSL 的 CPU/GPU 独立时钟曾测得 compute 大于 host wall，故只分别验证计时有限/非负，不以跨时钟大小比较作为完成条件；性能报告保留两套观测。

## V1.4：SDK、安装导出和模型交付

[全套 Release](validation/v14_tests.log)32/32，[不启用 ORT/TRT 的独立 Release](validation/v14_cpu_tests.log)27/27；[SDK ASan/UBSan](validation/v14_asan_tests.log)通过。生命周期测试覆盖独立输入/输出所有权、执行串行、队列和字节限流、已完成但尚未交付的结果占用额度、失败回收、并发关闭和排空，以及坏清单、模型路径逃逸、哈希错配和不可用后端。

通过 `cmake --install` 生成 CMake package，再从 `examples/depth_sdk` 独立 `find_package` 构建外部程序；不使用源码 `add_subdirectory`，不直接链接工作目录中的库。见[安装记录](validation/v14_install.log)与[外部项目构建](validation/v14_external_build.log)。导出的公共头文件没有 ORT/TRT 类型，依赖库由使用方提供路径。

外部程序从包内读取 RGB 图，推理、关闭 SDK 后写出 owning 结果。三个包都通过完整文件哈希检查及固定阈值 golden 检查：[ORT FP32](validation/v14_sdk_ort-fp32.json)、[TensorRT FP32](validation/v14_sdk_trt-fp32.json)、[TensorRT 混合 FP16](validation/v14_sdk_trt-fp16.json)。该单图用于部署冒烟回归，不替代 V1.2/V1.3 的 8 图最终集；此图本来存在的人工排序错误也原样记录，没有宣称模型任务精度满分。

## V1.5：真实网络结果和资源边界

[全套 Release](validation/v15_tests.log)33/33，包含既有 D16 协议、客户端、服务端、并发与关闭回归；[ASan/UBSan](validation/v15_asan_tests.log)的深度网络、旧 TCP 并发及关闭 3/3。新增测试覆盖手算 wire、大小端、非法元数据/版本/方向/长度、最大尺寸、非有限输出、重复 ID、半包、慢收发、数量/字节限流与恢复、断连及停止后的已接纳任务完成。

[实际 TCP 验证报告](validation/v15_tcp_report.json)记录 ORT FP32、TRT FP32、TRT 混合 FP16 每包两次网络请求。六次输出均与各自本地 SDK 的 1024×683 FP32 输出逐字节一致；三个服务器均 SIGTERM 正常结束，请求/响应/SDK 成功交付计数一致，网络/SDK 在途、等待、执行和字节预留全部为零。该验证是有限请求的正确性回归，持续过载与 30 分钟长稳属于 V2。
