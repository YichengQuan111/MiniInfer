# MiniInfer

MiniInfer 是一个基于 **C++17 / Linux（支持 WSL2）** 的深度估计推理运行时，将训练好的 **Depth Anything V2 Small** 封装成可供其他 C++ 程序调用的同步 / 异步 SDK。输入一张 RGB 图片，返回与原图尺寸一致的 FP32 相对深度图；数值越大表示越近，单位不是米。

模型计算由 ONNX Runtime 或 TensorRT 执行，项目重点是 **C++ 资源管理、并发请求调度、异常安全和部署验证**。

```text
接收图片 → 校验与资源预留 → 深拷贝并入队 → Worker 执行
        → 预处理 → ORT / TensorRT → 后处理 → 返回独立拥有数据的结果
```

## 核心能力

- **CPU / GPU 推理**：支持 ORT CPU FP32、TensorRT FP32 与混合 FP16；每个模型包使用固定的518或392输入尺寸。SDK 完成等比缩放、补边、归一化及输出尺寸还原。
- **同步 / 异步接口**：`run()` 同步返回，`submit()` 返回拥有型 `DepthRequest`，支持等待、超时观察、一次性交付和安全放弃。默认4个CPU Worker，每个ORT Worker独占Session；GPU槽位可独立配置为1或2，每槽位独占engine、context、stream和缓冲。
- **完整 CUDA 预处理**：支持CPU全路径、CPU resize/补边+GPU归一化路径，以及RGB8上传后在GPU完成全部像素处理的基础/融合路径。CUDA可选构建，CPU模式保持默认；GPU模式需要显式设置。
- **独立所有权**：输入深拷贝隔离调用方存储，输出对象拥有自己的数组；后续推理及 SDK 销毁不会使已经返回的结果失效。内存、模型会话和线程等资源通过 RAII 管理。
- **有界资源与安全关闭**：请求数量和字节预算覆盖排队、执行及已就绪未消费结果，超限返回 `Busy`。关闭后返回 `Closed`，排空执行并释放后端；保留句柄仍可在SDK关闭或销毁后交付结果。
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
    auto request = sdk.submit(image);  // 拷贝输入、预留预算、FIFO入队
    sdk.shutdown();
    const auto depth = request.get();  // 关闭后仍可交付；也可用sdk.run(image)同步调用
    write_f32("output.f32", depth.values());  // SDK 关闭后结果仍然有效
}
```

完整构建、模型准备和外部工程接入方式见 [部署指南](docs/depth_deployment.md)与 [SDK 示例](examples/depth_sdk/README.md)。

## V2 升级与验证

V2.1–V2.6 P0已按I-S-O完成，并逐个小版本提交；完整证据与限制见 [V2升级报告](docs/depth_v2_upgrade.md)。包含模型尺寸/精度选择、真实Nsight时间线、复用和槽位消融、A/B/C预处理对照、CUDA设备检查及长稳验收。固定回归数据与门槛保存在 `data/depth/v2_experiment.json` 和 `v2_preprocess_experiment.json`；P1实时帧/INT8及真实设备E1未计入本轮成果。

本机选定配置为 **518 FP32、完整 CUDA 基础预处理、1 个 GPU 槽位、4 个 CPU Worker、主机/设备缓冲复用**，详见 [显式配置](data/depth/v2_selected.json)。使用 TRT 模型包时，设置 `config.preprocess_mode = PreprocessMode::CudaFullBasic`、`config.gpu_slots = 1`；SDK 的兼容默认值仍为 CPU 预处理，配置文件不会自动加载。四路径对照中，该配置在20请求/秒下三轮SDK P99为24.80/24.24/23.80ms；频率未锁定，这些数值只代表记录的本机条件。

最终配置连续1800秒、20请求/秒长稳 **36,000/36,000成功**，SDK P99为26.85ms，输出抽检零差异，RSS峰值566.08MiB且分钟窗口中位数增长为0。Release回归39/39、CPU Sanitizer和CUDA内存检查通过；安装SDK的独立C++示例及最终8图、16次TCP请求通过。交付包位于 `data/generated/depth/v2-delivery`，模型与二进制保持Git忽略，源代码、工具和验收证据已版本化。

## 历史并发推理性能（2026-10-14）

以下保留V1同步闭环测量，不作为V2定时开环回放的结果。实测环境：Intel Core i9-13980HX、NVIDIA RTX 4080 Laptop GPU、WSL2 Ubuntu，Release 构建。使用 Depth Anything V2 Small，同一张 1024×683 RGB8 图片，模型输入为 518×518，batch=1。

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
