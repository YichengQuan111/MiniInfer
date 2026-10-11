# MiniInfer V1 验收记录

本文件记录本机实际执行证据。各阶段只有对应检查通过后才标记完成。

| 阶段 | 状态 | 证据 |
|---|---|---|
| V1.1 环境与模型契约 | 通过 | 16 图 Python CPU FP32；518 静态 ONNX checker；真实 CUDA/TRT 执行；Python 29 项、CPU/TCP Debug 24/24、Release 24/24 |
| V1.2 ONNX / ORT | 通过 | 16 图分层对齐；前后处理逐元素一致；Debug 29/29；Release 28/28 加验收驱动测试通过 |
| V1.3 TensorRT FP32 / FP16 | 通过 | 两配置最终集各 8/8；真实 GPU 生命周期/并发/坏引擎恢复；全套 31/31 |
| V1.4 C++ SDK | 未开始 | — |
| V1.5 TCP 深度接入 | 未开始 | — |
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
