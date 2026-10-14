# 每 Worker 独占 ORT Session 验收（2026-10-14）

本次将 SDK 从共享一个 ORT Session 改为每个 Worker 独占一个 Session。基线提交为 `42e2a59`，实测源码由[哈希记录](validation/sdk_worker_sessions_20261014/measured_sources.json)固定。上一版共享 Session 加锁串行的实测保留在[历史报告](sdk_validation_20261014.md)。

## 实现与资源边界

- 初始化时按 `worker_count` 创建独立 `OrtRunner`，全部创建成功后启动线程池。任务按线程池提供的固定 `worker_id` 选择对应实例，同一个实例不会被两个 SDK Worker 使用。
- 每个 `OrtRunner` 仍保留自身的 `Run()` 锁，保护独立调用它时的 ORT 1.21 telemetry；不同 Worker 使用不同实例和锁，可以并行推理不同图片。
- `ort_threads` 表示每个 Session 的计算线程数。默认配置是 4 个 Session、每 Session 4 个计算线程；模型状态、分配区和线程开销随 Session 数量增加。请求字节预算不包含这些固定资源。
- 部分 Session 构造失败时，已创建实例自动释放。`shutdown()` 停止接纳，排空任务并等待结果交付后释放全部实例，已有输出保持独立拥有数据。
- TensorRT 继续共用一个执行槽位。公开 SDK 接口、默认参数、输入/输出所有权和拒绝策略保持兼容。

## 测试结果

| 检查 | 结果 | 记录 |
| --- | --- | --- |
| Release 全套 CTest | 35/35 通过 | [日志](validation/sdk_worker_sessions_20261014/release/final/ctest.log) |
| ASan+UBSan 全套 CTest | 35/35 通过 | [日志](validation/sdk_worker_sessions_20261014/asan-ubsan/final/ctest.log) |
| TSan + 预编译 ORT，全套 CTest | 35/35 通过 | [日志](validation/sdk_worker_sessions_20261014/tsan/final/ctest.log) |
| TSan + 插桩 ORT，全套 CTest | 35/35 通过 | [日志](validation/sdk_worker_sessions_20261014/ort-tsan-instrumented/final/ctest.log) |
| 真实模型，TSan + 插桩 ORT，并发 1/4/8 | 每档 16/16 请求通过 | [报告](validation/sdk_worker_sessions_20261014/ort-tsan-instrumented/final/concurrency/report.json) |
| 真实模型，ASan+UBSan，并发 1/4/8 | 每档 16/16 请求通过 | [报告](validation/sdk_worker_sessions_20261014/asan-model/asan-ubsan-ort/report.json) |

新增确定性测试通过阻塞门控制两轮请求，验证四个后端能够同时执行、各实例固定绑定同一个 Worker 且不重入；第三个实例构造失败会释放前两个实例；关闭时保留仍在执行或等待交付的资源，交付结束后释放所有实例，SDK 销毁后返回值仍可用。既有所有权、限流、失败恢复、并发关闭及网络测试继续通过。

ASan 开启泄漏检测，UBSan/TSan 遇错退出，没有配置告警屏蔽。TSan 使用独立构建和 `setarch x86_64 -R`；插桩 ORT v1.21.0 的库 SHA-256、加载路径与插桩符号见[身份记录](validation/sdk_worker_sessions_20261014/ort-tsan-instrumented/final/commands.json)。ASan+UBSan 使用预编译 ORT，第三方库内部没有对应的编译插桩；TSan 的额外源码插桩覆盖主机 C/C++，不覆盖汇编、Python 或 GPU 设备代码。TensorRT 的实际模型压测使用 Release。

## 并发性能

测量使用同一台 i9-13980HX / RTX 4080 Laptop、WSL2 Ubuntu 环境及同一模型包。环境见[记录](validation/sdk_worker_sessions_20261014/environment.json)。

- Release 构建；固定 4 个 Worker，每个 ORT Session 4 个计算线程，队列容量 8，请求字节预算 512 MiB。
- 同一张 1024×683 RGB8 图片，batch=1，模型输入为 518×518；调用线程数依次为 1、4、8。
- 每个独立进程先执行 1 次校验基线及 20 次顺序预热，再测 300 次请求。每档重复 3 次，所有案例串行执行，拒绝和失败不重试。
- P50/P99 对全部成功请求的同步 `run()` 墙钟耗时作线性插值，包含拷贝、排队、前后处理和后端推理；QPS 使用成功数除以整个稳态墙钟时间，包含结果校验和线程调度，排除初始化、预热及写文件。
- 表中取三次独立运行指标的中位数；原始逐请求 CSV、失败计数、独立统计复算、输出一致性、所有权和排空检查随报告保存。这是有限样本基准，没有独占硬件或固定频率，不代表生产环境长期 P99。

| 后端 | 调用并发 | P50（ms） | P99（ms） | QPS | 成功 / 总数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ORT CPU FP32 | 1 | 329.48 | 355.15 | 3.00 | 900/900 |
| ORT CPU FP32 | 4 | 773.34 | 843.53 | 5.09 | 900/900 |
| ORT CPU FP32 | 8 | 1592.22 | 1732.21 | 4.96 | 900/900 |
| TensorRT FP32 | 1 | 16.77 | 18.16 | 47.47 | 900/900 |
| TensorRT FP32 | 4 | 46.31 | 47.97 | 78.35 | 900/900 |
| TensorRT FP32 | 8 | 95.85 | 98.12 | 79.34 | 900/900 |
| TensorRT 混合 FP16 | 1 | 16.34 | 18.29 | 47.96 | 900/900 |
| TensorRT 混合 FP16 | 4 | 41.98 | 63.56 | 85.47 | 900/900 |
| TensorRT 混合 FP16 | 8 | 88.85 | 91.06 | 85.34 | 900/900 |

8,100 个稳态请求全部成功，拒绝及失败均为零。完整结果：[ORT](validation/sdk_worker_sessions_20261014/performance/ort-fp32/report.json)、[TRT FP32](validation/sdk_worker_sessions_20261014/performance/trt-fp32/report.json)、[TRT 混合 FP16](validation/sdk_worker_sessions_20261014/performance/trt-fp16/report.json)。

## 与共享 Session 串行方案对比

CPU 吞吐对比使用 `42e2a59` 的同机、同模型、同参数历史结果，两次测量不属于交替运行的受控 A/B 实验。峰值内存另用本次保存的旧可执行文件与新文件串行实测，核对旧文件与原报告的 SHA-256 一致；每种设计测一次，4 路调用、64 次稳态请求、20 次预热。GNU `time` 的最大 RSS 包含整个进程的模型、分配区、输入/输出等内存，单位换算为 MiB；不等于单独模型大小或请求预算。

| 调用并发 | 共享 Session QPS | 每 Worker Session QPS | QPS 变化 | 旧 P99（ms） | 新 P99（ms） |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.82 | 3.00 | +6.3% | 381.35 | 355.15 |
| 4 | 2.95 | 5.09 | +72.3% | 1411.73 | 843.53 |
| 8 | 3.02 | 4.96 | +64.1% | 2724.79 | 1732.21 |

两版均设置 `ort_threads=4`，新方案创建 4 个 Session，潜在计算线程数也随之增加；上述变化反映整体并发配置的改变。

| 设计 | 进程峰值 RSS（MiB） | 样本 |
| --- | ---: | --- |
| 共享 Session，加锁串行 | 607.51 | 64/64 请求成功 |
| 每 Worker 独立 Session | 1548.20 | 64/64 请求成功 |

内存原始记录与逐次输出校验见[报告](validation/sdk_worker_sessions_20261014/memory/report.json)。

## 复现

按[部署指南](depth_deployment.md)配置 Release 构建，并运行：

```bash
source tools/depth_env.sh
python3 tools/run_depth_concurrency.py \
  --executable /path/to/release/bin/depth_concurrency_benchmark \
  --backend ort --package data/generated/depth/packages/ort-fp32 \
  --image data/generated/depth/packages/ort-fp32/golden/original.ppm \
  --output /tmp/miniinfer-worker-sessions-results \
  --concurrencies 1 4 8 --iterations 300 --warmup 20 --repeats 3 \
  --worker-count 4 --ort-threads 4 --queue-capacity 8 --byte-budget 536870912
```

TRT 使用 `--backend trt` 及相应模型包。TSan 与 ASan+UBSan 分别构建；TSan 真实模型检查使用上一轮构建的同版本插桩 ORT，运行前重新验证库路径和插桩标记，不能直接把未插桩第三方库的报警当成误报。实际命令与返回码见[执行记录](validation/sdk_worker_sessions_20261014/commands.json)。
