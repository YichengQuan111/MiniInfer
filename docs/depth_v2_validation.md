# MiniInfer V2 历史验收记录（2026-10-11）

> 本文为回滚前的历史测量，恢复后保留原日期与结果。本轮源码集成、复测与最终进度见 [2026-10-15 升级记录](depth_v2_upgrade.md)。

需求与冻结标准见 [I-S-O 计划](MiniInfer_V1_V2_求职改进策略.md)和 [实验配置](../data/depth/v2_experiment.json)。以下只记已执行的验证，各小版本单独提交。V1 原有数据、包和报告保留。

## V2.1：负载、测量与基线（2026-10-11）

新增 `depth_replay` 开放环定时回放、Python 统计与资源采样、SDK CPU 墙钟阶段计时和可选 NVTX 范围。输入在计时前解码，按固定序列循环；成功、拒绝、失败、生成端容量不足与迟到均有终态，未提交请求仍进入计划数分母。窗口内成功吞吐和排空后成功率分别统计，不把排空时间内的完成计入窗口吞吐。

V2.1 适配器使用并发调用方调用同步 SDK；接纳与排队阶段采用 SDK 内部 CPU 时长，不声称已有公开异步接口。`ready` 在 SDK 返回时记录，随后销毁 owning 结果并记录 `delivered`。数值统计缓存、输入图像缓存、未完成请求数和输出日志均有边界。

### 数据和环境

- 新增 `samples_v2.json`：八个场景各两张，共 16 张，开发 / 最终各八张；原始来源与 V1 不重叠。按路径排序跳过各场景前两张符合条件的图像，不按模型预测筛选。共 30 对标注，开发 13 对、最终 17 对；样本量只支持本地回归结论。
- 回放只使用开发集八张，原始图像 693,594–2,073,600 像素；不同于 V1 的单张图负载，不能直接比较两套 P99。
- i9-13980HX、RTX 4080 Laptop 12,282 MiB、WSL Ubuntu 24.04、GCC 13.3、TensorRT 10.8.0.43、CUDA runtime 12.4。交流供电，电池 100%；时钟未锁定，桌面后台未隔离。
- 工具由 NVIDIA 官方分发清单与哈希锁定，安装在用户缓存目录，无驱动升级。版本与路径见 [工具记录](validation/v21_toolchain.json)。Nsight 使用进程 CUDA/NVTX 追踪；关闭需要更高权限的系统 CPU 采样。

### 实测与冻结标准

预试为各一次 10 秒正式窗口、2 秒预热：20 请求/秒为 200/200 成功、P99 34.07 ms；40 请求/秒为 387/400 成功、13 次 Busy、P99 63.42 ms；80 请求/秒为 400/800 成功、400 次 Busy、P99 74.34 ms。均无生成端漏发或失败。此后冻结正常 20、饱和探测 40、过载 80 请求/秒及质量、内存、长稳门槛，再开始优化。

正式基线：518、混合 FP16（`backbone-fp32-head-fp16-v2`）、CPU 前后处理、单槽位；每次 2 秒预热及 50 秒正式窗口，8 个调用线程、最多 8 个未完成、SDK 队列 4、128 MiB 预留预算。

| 重复 | 成功 / 计划 | 成功请求/s | SDK P50 / P95 / P99 ms | 回放 P99 ms |
|---|---:|---:|---:|---:|
| 1 | 1000 / 1000 | 20.0 | 25.90 / 32.72 / 35.74 | 35.97 |
| 2 | 1000 / 1000 | 20.0 | 30.49 / 34.22 / 37.77 | 38.04 |
| 3 | 1000 / 1000 | 20.0 | 24.42 / 30.81 / 33.38 | 33.77 |

三次均无拒绝、错误、迟到或容量漏发，计数校验通过，满足已冻结正常负载目标。原始逐请求 CSV、资源 CSV、命令、图像与工件哈希、分位数和环境保存在 [基线目录](validation/v21_baseline/)。P99 逐次报告，不平均分位数，也不删慢请求。

### Profiler 与测试

独立采集的 Nsight 时间线覆盖 60 次调用（含预热），CPU NVTX 均值：预处理 6.78 ms、TRT 包含拷贝与等待 12.94 ms、后处理 2.84 ms。CPU 预处理已有可测代价，可为后续归一化与布局转换实验提供依据；它包含 resize 等工作，不能将整个预处理时长视为 CUDA 融合收益。范围嵌套，百分比不可相加；CUDA 与 CPU 时钟独立，不能相减推导开销。追踪数据不并入正式性能表。

- [Nsight 汇总](validation/v21_profile_stats.csv)；本机完整时间线 `data/generated/depth/v21_profile/timeline.nsys-rep`（二进制忽略提交），SHA256 `80b83c615ee91ad0e2ebad38ce9a82b8da25bf572e06b0d7ad703e77f23f707b`。
- Release 全量 CTest **35/35 通过**，包含原 V1 34 项与新增回放统计测试；[日志](validation/v21_tests.log)。新增 Python 测试 8 项，涵盖计数、分位数、窗口 / 排空、迟到 / 容量及采样边界；CLI 非法输入和帮助也已检查。
- 工具链完成 nvcc 编译链接 smoke；CUDA kernel、Compute Sanitizer 与 Nsight Compute 的实际验收留待 V2.4，不能据此宣称已经通过。

### 复现

先按 V1 文档准备环境、权重、原包；用 `tools/prepare_depth_samples.py --help` 按 V2 清单的 offset 和前缀生成独立图像。随后在 WSL 项目根目录执行：

```bash
source tools/depth_env.sh
bash tools/configure_depth.sh /home/qyc/builds/MiniInfer/v21 Release
cmake --build /home/qyc/builds/MiniInfer/v21 -j 8
ctest --test-dir /home/qyc/builds/MiniInfer/v21 --output-on-failure
/home/qyc/.venvs/miniinfer-depth/bin/python tools/run_depth_replay.py \
  --executable /home/qyc/builds/MiniInfer/v21/bin/depth_replay \
  --package data/generated/depth/packages/trt-fp16 --backend trt \
  --images-list data/depth/replay_v2.txt --output /tmp/miniinfer-v21-new \
  --duration 50 --warmup 2 --rate 20 --workers 8 --max-outstanding 8 --repeats 3
```

输出目录必须不存在，以免覆盖历史证据。Profiler 构建需另设 `MINIINFER_NVTX_INCLUDE` 指向 NVTX3 头文件；完整采集命令见 profile 日志及其对应环境记录。此阶段不作默认精度、双槽位或 CUDA 路径选择。

### GPU 时间线补验

原 Nsight 2023.4 文件只有 CPU/NVTX，不能用来判断 GPU overlap。进一步用同一 CUDA smoke 对照，2024.5 明确报告不支持当前 driver API 13.4；升级用户缓存内的 Nsight Systems CLI 到 **2026.5.1.161** 后，保持 CUDA runtime 12.4 与驱动不变，smoke 与真实 TRT 回放均采到了 kernel / memcpy。安装命令为 `python tools/setup_depth_v2_tools.py --help` 所示 `--nsys-release cli-2026.5.1` 选项；工具归档及哈希锁定在安装脚本。

[兼容性诊断](validation/v21_profile_diagnosis.md)、[修复后活动统计与完整 trace 哈希](validation/v21_profile_fixed.json)、[实际 GPU / NVTX 汇总](validation/v21_profile_fixed_stats.csv)保留了失败与修复证据。修复后仍独立采集，含初始化与预热，不用于性能排名。

## V2.2：静态尺寸、精度与部署身份

已贯通 518 / 392 两套静态 shape：C++ 前后处理、ORT / TRT、CLI、Python 参考与验收、SDK 模型包。两种 shape 使用独立 contract；旧 `depth-v1` 518 包继续可读，392 的包 / 参考 / 阈值 / 引擎 shape 必须一致。包的新版 identity 将输入 SHA256、参考 manifest、模型及阈值绑定；尺寸错配和不受支持的形状在执行前拒绝，不增加动态 shape。

使用同一权重与 V2 独立样本生成本尺寸 PyTorch 参考；保持 FP32 I/O。混合 FP16 仍为 V1 已验证的 `backbone-fp32-head-fp16-v2`，并非全模型 FP16。四组同尺寸部署精度均通过；缩小到 392 后，两种精度都使开发集 `v2_indoor_0` 第三对排序由对变错。因此 392 保留为受支持的实验选项，但未通过“相对 518 PyTorch 零新增错对”的冻结默认选择门槛。

每个候选三次独立进程，20 请求/s、2 秒预热、50 秒正式窗口；均为 1000/1000 成功、20.0 成功请求/s，零拒绝、错误与生成漏发。正式矩阵共 12,000 个请求。

| 配置 | 同尺寸精度 | 开发排序 | SDK P99 三次 ms | engine bytes | 稳态 RSS 最大 KiB | 选择资格 |
|---|---|---:|---|---:|---:|---|
| 518 FP32 | 通过 | 13/13 | 33.63 / 35.13 / 35.72 | 105,586,948 | 583,900 | 通过，选中 |
| 518 混合 FP16 | 通过 | 13/13 | 36.03 / 37.96 / 34.75 | 98,029,604 | 616,196 | 通过 |
| 392 FP32 | 通过 | 12/13 | 30.36 / 31.27 / 29.36 | 105,524,148 | 594,424 | 新增一对错误 |
| 392 混合 FP16 | 通过 | 12/13 | 30.27 / 33.68 / 35.76 | 98,291,716 | 608,024 | 新增一对错误 |

通过资格的候选按“三次 SDK P99 中最大值最小，engine 大小为平局规则”选择，故本阶段选择 **518 FP32**。最终集未参与选择，之后独立复验 8 张、17/17 对排序正确；raw MAE 1.90e-6、restored MAE 9.98e-7，前后处理独立误差为 0，全部冻结数值门槛通过。这里的排序样本仍很小，不作泛化保证。

518 的应用 device I/O 与 pinned I/O 各 4,293,184 bytes；392 各 2,458,624 bytes。它们不含 TRT 权重、context、库内分配。RSS 与设备总显存另由回放采样，不混称“进程显存”。权重、ONNX、参考与包的大小 / 哈希见 [工件清单](validation/v22/artifacts.json)。各次 P50/P95/P99、阶段时长、温度 / 时钟 / 设备总显存、逐请求分母和失败候选均见 [决定依据](validation/v22/decision.json)及 [完整原始记录](validation/v22/)。

独立导出的 V2.2 源码完成 Release **36/36 CTest**，验证时不混入正在开发的 V2.3。392 实际 TensorRT 推理与资源寿命检查、392 ORT 开发集数值回归、选中包的 SDK golden 均通过。ORT 首次补验因手写了不存在的 CLI 名称而未执行推理，修正为 `depth_cli` 后在新目录重跑，错误记录保留。日志：[CTest](validation/v22_verify_tests.log)、[392 TRT](validation/v22_gpu_392.log)、[392 ORT](validation/v22/quality-392-ort-development.json)、[SDK golden](validation/v22/sdk-golden.json)。

选中模型包位于 `data/generated/depth/packages/v22-518-fp32`。参考 / 模型体积较大，不提交二进制；以 `tools/depth/reference.py --help` 生成 518 reference 与 392 export，再依次执行 `tools/run_depth_v22.py engines|development|benchmark|decision|final|package --build BUILD`（每次一个 phase）。脚本使用明确路径且不覆盖已有证据。配置控制和后续改进均以此包为基准。
