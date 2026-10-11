# V1 部署核验与排查记录

固定模型、输入处理和阈值见 [部署契约](depth_deployment.md)，各版本执行证据见 [验收记录](depth_validation.md)。以下记录来自实际开发与测试，不将故意构造的负例当成线上故障。

## FP16 首版 raw 输出未过门槛

输入：固定开发集 8 图，原始 RGB、Python 预处理张量与输出已保存。首版策略 `attention-normalization-fp32-v1` 在 TensorRT 中将注意力/归一化保留 FP32，其余允许 FP16。

检查顺序：先核对输入图、预处理哈希和变换元数据；再分别用 C++ RGB 路径与相同的 Python 输入张量执行；最后比较 raw 输出、恢复后的输出与独立近远标注。前后处理对齐，但首版 raw P99 绝对误差约 0.08724、最大约 0.24620，`adverse_style_0` 和 `aerial_0` 两个开发样本未通过冻结门槛。仅看恢复输出或排序会漏掉 raw 数值退化，原始失败报告保存在 [v13_fp16_rejected_development.json](validation/v13_fp16_rejected_development.json)。该引擎没有进入最终集测试。

修复：保留 backbone FP32，卷积 head 允许 FP16，策略命名 `backbone-fp32-head-fp16-v2`；没有改变阈值或对原始结果做缩放。最终层检查实际包含 Half 输出，因此称为混合 FP16。固定最终集 8/8 通过，raw MAE 0.001853、P99 0.010976，超容差元素约 0.00326%；任务排序与 Python 相同。完整数值及适用边界见 [精度报告](depth_accuracy.md)。

该证据将问题定位到模型图的混合精度执行范围，但没有证明每一个误差由某个单独算子造成；V1 不加入未经实测的逐算子因果结论。性能也应由实测决定，不因启用 FP16 就宣称加速。

## 引擎加载失败与包不匹配

测试使用有效引擎副本修改精度标签、修改哈希、提供与哈希匹配但内容损坏的 engine，分别触发清单兼容性拒绝、损坏拒绝和 TensorRT 反序列化失败；随后加载有效引擎并执行成功。见 [FP32 生命周期记录](validation/v13_fp32_lifecycle.log)和 [FP16 记录](validation/v13_fp16_lifecycle.log)。

诊断顺序：`package_depth.py verify` 检查整包；核对 `manifest.txt` 后端/精度/模型哈希；核对 engine 清单的 ONNX、静态形状、精度策略、硬件、CUDA/TRT 与缓存标识；最后确认加载和实际执行。目标环境变化时，从 ONNX 在当地重建并按冻结阈值复验，不改清单伪装兼容。SDK 的 `ModelIncompatible` 与 `BackendFailure` 分别表达包契约失败和后端加载/执行失败。

## WSL CPU 与 GPU 时间不能相减

GPU 生命周期检查实际出现 FP32 `compute=53.126 ms`、`host=50.326 ms`，以及 FP16 `compute=61.4571 ms`、`host=57.7823 ms`。原先 `host >= compute` 的断言不成立；CPU steady clock 与虚拟化环境中的 CUDA event 时钟是独立观测，不能据其大小关系判断 stream 是否完成。

修正检查：执行路径仍先同步 stream，再读取 events 和交付输出；分别要求耗时有限且为正。性能报告以主机 wall time 比较同口径调用，同时保留 raw TRT 的 H2D/compute/D2H events，不跨时钟相减计算“框架开销”。这不是精度或同步门槛的放宽。

## 独立安装与常见执行边界

`find_package(MiniInfer CONFIG REQUIRED)` 应指向安装目录；使用方提供启用后端的 ORT/TRT/CUDA 共享库路径。若动态加载器找不到 `.so`，使用项目的 `tools/depth_env.sh` 或部署系统配置的 loader 路径，再启动程序。CMake 导出不固化开发机的绝对库路径，安装目录提供头文件、静态库、CLI、契约和独立示例源文件；模型包单独选择。

所有读取入口先检查尺寸/形状和有限性，生成工具拒绝覆盖工件。`Busy` 表示本次未接受，`Closed` 表示关闭后不再接受；已接受设备计算完成后才释放资源。网络 Busy 可能发生在 RGB payload 发送完之前，应重建连接；推理过程中断连不意味着 GPU 计算已经取消。

新机器先执行环境探针，再导出/重建与 golden 回归，最后测性能。V1.6 的“全新构建”是同一已记录 WSL 和依赖环境中的新 CMake 构建目录，不冒充另一台机器或 ARM/Jetson 验证；启动也不宣称清空操作系统文件缓存。大文件下载可从 Windows 侧缓存后核验哈希，GPU 推理本身无需 Python 服务。
