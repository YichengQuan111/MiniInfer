# MiniInfer V1 验收记录

本文件记录本机实际执行证据。各阶段只有对应检查通过后才标记完成。

| 阶段 | 状态 | 证据 |
|---|---|---|
| V1.1 环境与模型契约 | 通过 | 16 图 Python CPU FP32；518 静态 ONNX checker；真实 CUDA/TRT 执行；Python 29 项、CPU/TCP Debug 24/24、Release 24/24 |
| V1.2 ONNX / ORT | 未开始 | — |
| V1.3 TensorRT FP32 / FP16 | 未开始 | — |
| V1.4 C++ SDK | 未开始 | — |
| V1.5 TCP 深度接入 | 未开始 | — |
| V1.6 部署与基线 | 未开始 | — |

前后处理、版本和复现方式见[部署契约](depth_deployment.md)。既有 CPU / D16 的历史验收与性能记录保留原义。

## V1.1：环境、数据与导出可行性

2026-10-11 在 WSL Ubuntu 24.04、RTX 4080 Laptop（12,282 MiB，驱动 617.14，计算能力 8.9）执行。使用独立虚拟环境，`pip check` 无依赖冲突；[环境和最小设备证据](validation/v11_environment.json)、[完整 Python 版本](validation/v11_pip_freeze.txt)已保存。最小网络 `y=x+x` 在 CUDA stream 完成后核对输出 `[2,-4,7,0]`，通过。

[Python 参考清单](validation/v11_reference_manifest.json)记录全部 16 张图、中间张量、模型源码与权重哈希。它验证基线可生成，近远排序与跨后端数值验收在后续阶段进行。[静态导出清单](validation/v11_export_manifest.json)记录 ONNX SHA-256 `a7d66386939681431d3b7c407b897d00cf205c44c607a33189eb4238bb7c9f80`，大小 98,941,039 字节，opset 17，输入 `[1,3,518,518]`，输出 `[1,518,518]`，checker 通过。

导出日志中的 TracerWarning 来自官方模型将固定 518 形状转为 Python 整数或布尔值；本版本明确仅接收固定形状，已核验导出输入输出，动态尺寸不在契约内。未安装可选 xFormers，执行使用 PyTorch 路径。

验证日志：[参考执行](validation/v11_reference.log)、[导出](validation/v11_export.log)、[设备执行](validation/v11_probe.log)、[Python CTest](validation/v11_python_tests.log)、[Debug 回归](validation/v11_debug.log)、[Release 回归](validation/v11_release.log)。Python 前后处理 15 项与统计 14 项测试在 WSL 通过；既有两套 24 项回归全部通过。
