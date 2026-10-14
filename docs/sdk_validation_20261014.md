# SDK 并发与 Sanitizer 验收（2026-10-14）

本记录验证当前同步 `DepthSdk::run()` 的资源管理、并发正确性和实际模型性能。测试基线为 `b13dddb` 加本次提交中的修改；[最终源码哈希](validation/sdk_validation_20261014/measured_sources_final.json)记录实际编译的文件内容，性能报告另含可执行文件、图片和模型包的 SHA-256。修复 ORT Session 数据竞争之前的[源码哈希](validation/sdk_validation_20261014/measured_sources.json)和初始运行记录另行保留。

## 测试结果

| 构建 / 检查 | 结果 | 原始记录 |
| --- | --- | --- |
| Release 全套 CTest | 35/35 通过 | [日志](validation/sdk_validation_20261014/release/ort-serialized/ctest.log) |
| ASan + UBSan 全套 CTest | 35/35 通过 | [日志](validation/sdk_validation_20261014/asan-ubsan/ort-serialized/ctest.log) |
| TSan + 预编译 ORT，全套 CTest | 35/35 通过 | [日志](validation/sdk_validation_20261014/tsan/ort-serialized/ctest.log) |
| TSan + 插桩 ORT，全套 CTest / 真实模型 1/4/8 | 35/35；各 16/16 请求通过 | [完整记录](validation/sdk_validation_20261014/ort-tsan-instrumented/validation-serialized/commands.json) |
| 真实 ORT，ASan + UBSan，并发 1/4/8 | 各 16/16 请求通过 | [报告](validation/sdk_validation_20261014/asan-smoke-serialized/asan-ubsan-ort/report.json) |
| 真实 TensorRT FP32 / 混合 FP16 生命周期 | 两种精度均通过 | [FP32](validation/sdk_validation_20261014/smoke/trt-fp32-lifecycle.log)、[FP16](validation/sdk_validation_20261014/smoke/trt-fp16-lifecycle.log) |
| TensorRT FP32 / 混合 FP16，并发 1/4/8 预检 | 每种精度、每档各 16/16 请求通过 | [FP32](validation/sdk_validation_20261014/gpu-smoke/trt-fp32/report.json)、[FP16](validation/sdk_validation_20261014/gpu-smoke/trt-fp16/report.json) |

CTest 包括 SDK 输入 / 输出所有权、资源限额、故障恢复、线程池、并发关闭、网络协议及压测统计验证。ASan 开启泄漏检测，UBSan / TSan 遇错退出；没有配置告警屏蔽。TSan 单独构建，并使用 `setarch x86_64 -R` 固定地址布局。Python 测试与预编译第三方库本身不属于 C++ Sanitizer 插桩覆盖；实际 GPU 生命周期测试使用 Release 构建。

首轮的 TensorRT 缺失文件测试触发 NVIDIA 驱动侧泄漏 / 线程泄漏报告。修复将 manifest、契约和文件哈希校验移到 CUDA 初始化之前，并增加不可用设备编号下的离线拒绝回归测试。保留[首轮 ASan 失败记录](validation/sdk_validation_20261014/asan-ubsan/ctest.log)及[首轮 TSan 失败记录](validation/sdk_validation_20261014/tsan/ctest.log)。有效 engine 的并发推理、对象销毁后结果可用性、损坏文件拒绝和重新加载恢复均通过实际 GPU 测试。

额外真实模型检查中，预编译 ORT 1.21.0 在调用并发 1/4/8、ORT 内部线程数 4 时均触发 TSan 告警，堆栈两端位于 ORT 内部的分配 / 释放，见[原始报告](validation/sdk_validation_20261014/smoke/tsan-ort/report.json)。未插桩第三方库的同步可能无法被 TSan 观察到，但这不能单独证明本次告警是假阳性，参见 [TSan 官方说明](https://github.com/google/sanitizers/wiki/ThreadSanitizerCppManual#non-instrumented-code)。因此从同版本源码构建了带 TSan 的 ORT 及其 C/C++ 依赖。

插桩复验进一步捕获到共享 Session 的实际数据竞争：[失败日志](validation/sdk_validation_20261014/ort-tsan-instrumented/validation/ctest.log)指向 ORT `inference_session.cc:2747`，多个 `Run()` 同时更新非原子的 telemetry 计数。修复在每个 `OrtRunner::Impl` 中增加互斥锁，保护完整的 `Session.Run()`；输入私有拷贝、输出深拷贝和 SDK 前后处理仍在锁外执行。该修复同样用于普通构建，没有跳过并发测试或屏蔽告警。修复后，使用插桩 ORT 的完整 CTest 35/35 通过，真实模型并发 1/4/8 各 16/16 请求通过，未出现 TSan 报告。见[CTest 日志](validation/sdk_validation_20261014/ort-tsan-instrumented/validation-serialized/ctest.log)和[并发报告](validation/sdk_validation_20261014/ort-tsan-instrumented/validation-serialized/concurrency/report.json)。

插桩库来自 ORT v1.21.0，1,012 条 C/C++ 编译命令均包含 `-fsanitize=thread`；运行前检查了库路径、插桩符号及 SHA-256。构建与依赖来源见[构建记录](validation/sdk_validation_20261014/ort-tsan-instrumented/commands.json)，运行时身份见[复验记录](validation/sdk_validation_20261014/ort-tsan-instrumented/validation-serialized/commands.json)。Eigen 的 GitLab 下载受阻后改用同一提交的 GitHub 镜像，并验证对应归档哈希。覆盖范围是主机 C/C++ 路径，不包含汇编、GPU 设备代码或 Python；GCC 关于 `atomic_thread_fence` 的 TSan 支持限制告警保留在[构建日志](validation/sdk_validation_20261014/ort-tsan-instrumented/build_native_2.log)中。

## 并发压测

环境：i9-13980HX，WSL2 / Ubuntu 24.04.5，32 个可见逻辑 CPU、约 16 GiB 内存；RTX 4080 Laptop 12 GiB，驱动 617.14；GCC 13.3、ORT 1.21.0、TensorRT 10.8.0.43。完整环境见 [environment.json](validation/sdk_validation_20261014/environment.json)。本机测量没有独占硬件或固定频率。

- Release 构建，三个模型包分别运行，各档和各次重复均串行执行。
- ORT 共享 Session 的 `Run()` 加锁串行，单次 ORT 推理使用内部线程；不同请求的前后处理可重叠。TensorRT 仍为单个 GPU 执行槽位。
- 同一 SDK 固定 4 个 Worker、ORT 内部线程数 4、队列容量 8、字节预算 512 MiB；调用线程数依次为 1、4、8。SDK 默认队列 4、预算 128 MiB，本次显式增大，以容纳 8 个完整图片请求。
- 同一张 1024×683 RGB8 图片，batch=1，模型输入固定为 518×518；每次请求预留 30,065,664 字节。预算只约束接纳预留，不是进程总内存上限。
- 每次独立进程先执行 1 次正确性基线、20 次预热，再统计总计 300 次请求；每档重复 3 次。调用线程统一起跑，完成一次后发下一次，属于闭环负载；拒绝和失败保留且不重试。
- P50 / P99 使用全部成功请求的 `run()` 墙钟耗时做线性插值，包含接纳、输入拷贝、排队、预处理、后端推理和后处理。
- QPS = 成功请求数 / 整段稳态墙钟秒数。该时间包含输出校验和调用线程调度，不包含初始化、预热、文件读写及关闭；不能用各线程耗时相加作为分母。
- 每次检查尺寸、有限值、精确 checksum 和独立存储，保留旧结果检查后续推理及 shutdown 后的数据；结束时校验计数、排空状态和零预留。

以下数据取 3 次独立重复结果的中位数；没有把每线程分位数平均，也没有把 3 次样本合并。每档共 900 个稳态请求。完整分位数范围、原始逐请求 CSV、失败计数和复算结果保存在各报告中。这是有限样本基准，不是长时间稳定性或生产 P99 保证。

| 后端 | 调用并发 | P50（ms） | P99（ms） | QPS | 成功 / 总数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ORT CPU FP32 | 1 | 349.79 | 381.35 | 2.82 | 900/900 |
| ORT CPU FP32 | 4 | 1341.72 | 1411.73 | 2.95 | 900/900 |
| ORT CPU FP32 | 8 | 2635.70 | 2724.79 | 3.02 | 900/900 |
| TensorRT FP32 | 1 | 16.74 | 18.63 | 47.48 | 900/900 |
| TensorRT FP32 | 4 | 45.87 | 49.23 | 79.12 | 900/900 |
| TensorRT FP32 | 8 | 96.84 | 104.08 | 78.03 | 900/900 |
| TensorRT 混合 FP16 | 1 | 16.38 | 18.37 | 47.77 | 900/900 |
| TensorRT 混合 FP16 | 4 | 42.54 | 43.79 | 84.68 | 900/900 |
| TensorRT 混合 FP16 | 8 | 90.11 | 96.67 | 83.71 | 900/900 |

共 8,100 个稳态请求全部成功，Busy / Closed / Failure 均为 0，所有统计复算、所有权及排空检查通过。本次各后端的共享模型执行均串行；增加调用线程可以重叠前后处理，也会增加排队等待。三种后端从并发 4 增到 8 后，吞吐变化很小，P50 接近翻倍；本配置下继续增加调用线程主要增加等待。修复前的性能记录保留在 `validation/sdk_validation_20261014/performance/`，不作为最终实现的性能结论。

原始结果：[ORT FP32](validation/sdk_validation_20261014/performance-serialized/ort-fp32/report.json)、[TRT FP32](validation/sdk_validation_20261014/performance-serialized/trt-fp32/report.json)、[TRT 混合 FP16](validation/sdk_validation_20261014/performance-serialized/trt-fp16/report.json)。

## 复现入口

先按[部署指南](depth_deployment.md)准备依赖与模型包。在 WSL 仓库根目录执行：

```bash
bash tools/configure_depth.sh /tmp/miniinfer-release Release
cmake --build /tmp/miniinfer-release -j4
source tools/depth_env.sh
ctest --test-dir /tmp/miniinfer-release --output-on-failure

python3 tools/run_depth_concurrency.py \
  --executable /tmp/miniinfer-release/bin/depth_concurrency_benchmark \
  --backend ort --package data/generated/depth/packages/ort-fp32 \
  --image data/generated/depth/packages/ort-fp32/golden/original.ppm \
  --output /tmp/miniinfer-ort-results \
  --concurrencies 1 4 8 --iterations 300 --warmup 20 --repeats 3 \
  --worker-count 4 --ort-threads 4 --queue-capacity 8 --byte-budget 536870912
```

TRT 改用 `--backend trt` 和对应包，输出到新目录。工具拒绝覆盖已有证据。ASan+UBSan 与 TSan 各使用独立 Debug 构建目录，在 `configure_depth.sh` 后分别配置 `-DMINIINFER_ENABLE_ASAN_UBSAN=ON` 或 `-DMINIINFER_ENABLE_TSAN=ON`。确切命令与运行环境见 [ASan+UBSan](validation/sdk_validation_20261014/asan-ubsan/ort-serialized/commands.json)、[TSan](validation/sdk_validation_20261014/tsan/ort-serialized/commands.json)及[性能运行记录](validation/sdk_validation_20261014/performance-serialized/commands.json)。
