# MiniInfer

MiniInfer 是一个基于 **C++17 / Linux（支持 WSL2）** 的深度估计推理运行时，将训练好的 **Depth Anything V2 Small** 封装成可供其他 C++ 程序调用的同步 SDK。输入一张 RGB 图片，返回与原图尺寸一致的 FP32 相对深度图；数值越大表示越近，单位不是米。

模型计算由 ONNX Runtime 或 TensorRT 执行，项目重点是 **C++ 资源管理、并发请求调度、异常安全和部署验证**。

```text
接收图片 → 校验与资源预留 → 深拷贝并入队 → Worker 执行
        → 预处理 → ORT / TensorRT → 后处理 → 返回独立拥有数据的结果
```

## 核心能力

- **CPU / GPU 推理**：支持 ORT CPU FP32、TensorRT FP32 与混合 FP16。模型输入固定为 `[1, 3, 518, 518]`，SDK 完成等比缩放、补边、归一化及输出尺寸还原。
- **同步接口与并发接入**：`DepthSdk::run()` 等待结果后返回；内部通过线程池和 `std::future` 调度任务、传递结果与异常。默认 4 个 Worker，每个 Worker 独占一个 ORT Session，可并行推理不同图片；`ort_threads` 控制每个 Session 的计算线程数。多个 Session 会增加内存和线程开销。TensorRT 串行使用单个 GPU 执行槽位。
- **独立所有权**：输入深拷贝隔离调用方存储，输出对象拥有自己的数组；后续推理及 SDK 销毁不会使已经返回的结果失效。内存、模型会话和线程等资源通过 RAII 管理。
- **有界资源与安全关闭**：预分配环形队列，结合请求数量和字节预算限制接纳，超限返回 `Busy`；关闭时停止接收、排空已接收任务、等待结果交付和线程退出，再释放后端。
- **部署入口**：提供本地 CLI、RGB8 / FP32 TCP 请求、可安装的 CMake SDK 和带校验清单的模型包；TensorRT engine 可在目标环境重建并复验。

## SDK 使用示例

准备好依赖和模型包后，通过 `find_package(MiniInfer CONFIG REQUIRED)` 链接 `miniinfer::depth_sdk`。下面使用 ORT FP32 模型包；包内清单决定后端。

```cpp
#include <miniinfer/depth/sdk.hpp>

int main() {
    using namespace miniinfer::depth;

    SdkConfig config;
    config.package_dir = "/path/to/ort-fp32-package";
    config.worker_count = 4;
    DepthSdk sdk(config);

    const auto image = read_ppm("input.ppm");
    const auto depth = sdk.run(image);  // 同步等待完整结果
    sdk.shutdown();
    write_f32("output.f32", depth.values());  // SDK 关闭后结果仍然有效
}
```

完整构建、模型准备和外部工程接入方式见 [部署指南](docs/depth_deployment.md)与 [SDK 示例](examples/depth_sdk/README.md)。

## 并发推理性能

实测环境：Intel Core i9-13980HX、NVIDIA RTX 4080 Laptop GPU、WSL2 Ubuntu，Release 构建。使用 Depth Anything V2 Small，同一张 1024×683 RGB8 图片，模型输入为 518×518，batch=1。

固定 **4 个 Worker**；ORT 每个 Worker 独占 Session，每个 Session 使用 4 个计算线程；TensorRT 使用单个 GPU 执行槽位。下表的“调用并发”是同时调用 SDK 的线程数。测试队列容量为 8，请求字节预算为 512 MiB。

| 后端 | 调用并发 | P50（ms） | P99（ms） | QPS（请求/秒） | 相对单路吞吐 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ORT CPU FP32 | 1 | 329.48 | 355.15 | 3.00 | 1.00× |
| ORT CPU FP32 | 4 | 773.34 | 843.53 | 5.09 | 1.70× |
| ORT CPU FP32 | 8 | 1592.22 | 1732.21 | 4.96 | 1.66× |
| TensorRT GPU FP32 | 1 | 16.77 | 18.16 | 47.47 | 1.00× |
| TensorRT GPU FP32 | 4 | 46.31 | 47.97 | 78.35 | 1.65× |
| TensorRT GPU FP32 | 8 | 95.85 | 98.12 | 79.34 | 1.67× |
| TensorRT GPU 混合 FP16 | 1 | 16.34 | 18.29 | 47.96 | 1.00× |
| TensorRT GPU 混合 FP16 | 4 | 41.98 | 63.56 | 85.47 | 1.78× |
| TensorRT GPU 混合 FP16 | 8 | 88.85 | 91.06 | 85.34 | 1.78× |

每档先执行 1 次校验基线和 20 次预热，再测 300 次请求，独立重复 3 次；表中取三轮指标的中位数，共 **8,100 次请求全部成功**。吞吐倍率相对于同一后端的单路调用结果计算。

P50/P99 是完整同步 `run()` 的请求耗时，包含输入拷贝、排队、前后处理和推理；QPS 按整个稳态阶段的成功数与耗时计算，包含结果校验和调度，排除初始化与预热。这是有限样本的本机实测。

4 路调用时，CPU 吞吐达到单路的 **1.70 倍**，GPU FP32 / 混合 FP16 分别达到 **1.65 / 1.78 倍**。增加并发主要提升总吞吐，单请求延迟会随排队增加；GPU 数据体现完整 SDK 的并发吞吐，其执行槽位仍串行使用。

原始性能数据：[CPU FP32](docs/validation/sdk_worker_sessions_20261014/performance/ort-fp32/report.json)、[GPU FP32](docs/validation/sdk_worker_sessions_20261014/performance/trt-fp32/report.json)、[GPU 混合 FP16](docs/validation/sdk_worker_sessions_20261014/performance/trt-fp16/report.json)。
