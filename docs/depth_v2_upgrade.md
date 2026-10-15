# V2 升级与验收（2026-10-15）

本轮依据 [2026-10-15 I-S-O 计划](MiniInfer_V1_V2_求职改进策略.md)逐项升级，每个小版本完成后单独 Git commit。起点为 `bf55478`：四个 CPU Worker、每 Worker 独占 ORT Session；这些行为需要保留。V2.4 必须实现完整 RGB8 GPU 预处理，局部归一化路径不能替代它。

## 证据与范围

恢复利用旧提交 `04a0ee6`、`9262958` 及中断备份中的可用实现，并重新与当前 SDK 集成。原始 [2026-10-11 分阶段记录](depth_v2_validation.md)及 `validation/v21*`、`validation/v22*` 是历史证据，不作为本轮二进制的性能结果。本轮数据独立保存于 `validation/v2_upgrade/`，不覆盖旧测量。

沿用 [负载与质量门槛](../data/depth/v2_experiment.json)，新增 [完整 GPU 预处理冻结契约](../data/depth/v2_preprocess_experiment.json)。图像仍是原 V2 开发/最终划分；本轮属于固定样本回归，不宣称重新建立了未见测试集。单机结果只代表本机、该负载和记录的运行条件。

| 小版本 | 交付 | 当前验收状态 |
|---|---|---|
| V2.1 | 定时回放、计数、分段 NVTX、输入上传字节、工具链与基线 | 通过，见下文 |
| V2.2 | 518/392、FP32/混合 FP16 配置与质量取舍 | 待验收 |
| V2.3 | 拥有型异步句柄、背压、缓冲复用、1/2 GPU 槽位 | 待验收 |
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
