# MiniInfer V1 深度部署契约与复现

V1 以 Depth Anything V2 Small 为唯一视觉负载，模型输入固定 `[1,3,518,518]`。阶段完成状态与实测证据见 [V1 验收记录](depth_validation.md)。V1.1—V1.6 分别提交 Git，原 CPU MLP 与 D16 回归继续保留。

## 模型与图像契约

机器可读规范位于 [contract.json](../data/depth/contract.json)。源码固定为官方提交 `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`，Small 权重 SHA-256 为 `715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378`。权重以 `weights_only=True` 严格加载。

外部图像是连续 RGB8 HWC，宽高在 1—4096，像素总数不超过 2,097,152。先将像素转为 FP32 并除以 255，再等比缩放到最长边 518；缩放尺寸采用整数 round-half-up，最短边至少为 1。双线性插值使用 half-pixel 坐标与边界 clamp，不启用抗锯齿。中心 padding 使用 RGB=0，多余一像素放右侧或下侧；随后按 ImageNet mean/std 归一化并变为 NCHW。

Python 中间运算使用 FP32，插值按先水平后垂直的 `a+(b-a)*w` 顺序计算。C++ 对应路径禁用浮点收缩，不使用 fast-math。原始深度为 FP32 `[1,518,518]`，后处理去掉 padding 并按相同插值恢复原图尺寸。对外返回独立拥有的行主序 FP32 HW 数据；数值越大表示越近，单位不是米。伪彩色与逐图 min-max 不参与数值验收。

此几何包装是项目契约，与[官方默认处理](https://github.com/DepthAnything/Depth-Anything-V2/blob/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf/depth_anything_v2/dpt.py)区分；模型图结构和权重保持原样。

## 样本和任务标注

[samples.json](../data/depth/samples.json)固定官方 [DA-2K](https://github.com/DepthAnything/Depth-Anything-V2/blob/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf/DA-2K.md)中的 16 张图像与 30 组独立近远标注。8 个场景各取按路径排序后满足尺寸约束的前 2 张，第 1 张用于开发，第 2 张用于最终验收；选择过程不读取模型预测。原始标注的 `[y,x]` 转为项目 `[x,y]`，保留来源与哈希；相等预测算排序错误，不确定点对排除。

这只是部署与任务保持的小规模检查集，不代表完整 DA-2K 成绩。最终样本不用于选择容差。图片和模型放在忽略的 `data/generated/depth`，Git 保存选择清单、契约、生成工具与检查证据。

## 独立环境与下载

源码继续在 Windows 工作区编辑，Linux 工具从 `/mnt/c/Users/ASUS/Documents/MiniInfer` 读取同一份源码。GPU 环境独立于已有 CPU 虚拟环境，默认 `/home/qyc/.venvs/miniinfer-depth`；模型与 C++ 后端依赖缓存默认 `/home/qyc/.cache/miniinfer-depth`。

在项目根目录的 WSL shell 中执行：

```bash
bash tools/bootstrap_depth.sh
```

环境固定 PyTorch 2.6.0+cu124、torchvision 0.21.0+cu124、ORT 1.21.0、TensorRT 10.8.0.43，其他直接依赖见 [depth-requirements.txt](../tools/depth-requirements.txt)。选择一组固定版本进行实际兼容验证，不自动追随最新版本。NVIDIA 驱动来自宿主系统，安装脚本不替换驱动。

WSL 直连下载较慢时，可先在 Windows 项目根目录运行 `python tools/fetch_depth_assets.py --output data/generated/depth/assets`，再在 WSL 使用已有下载；工具核验已冻结的 SHA-256。模型源码 ZIP 解压到缓存目录，使用明确的 `--repo-commit`，参考清单另保存每个 Python 源文件的哈希。

`tools/fetch_depth_wheels.py --output data/generated/depth/wheels` 可从 Windows 获取 Linux CPython 3.12 的辅助依赖，保存 PyPI 发布哈希。它是下载补充，不修改 Windows Python 环境，也不包含完整 PyTorch / NVIDIA CUDA wheel 集；完整环境以安装记录与 `pip freeze` 为准。

样本重新生成到新清单文件后核对，不覆盖已冻结清单：

```bash
python tools/prepare_depth_samples.py \
  --archive data/generated/depth/assets/DA-2K.zip \
  --images data/generated/depth/images \
  --manifest data/depth/samples-reproduced.json
```

## Python 参考与静态导出

以下命令中的 `python` 应为独立深度环境解释器，输出目录必须不存在：

```bash
python tools/probe_depth_env.py --output data/generated/depth/environment-new.json
python tools/depth/reference.py reference \
  --repo /home/qyc/.cache/miniinfer-depth/Depth-Anything-V2-a561b849ebae10a6f5ef49e26c83cbbcd36c71bf \
  --repo-commit a561b849ebae10a6f5ef49e26c83cbbcd36c71bf \
  --weights data/generated/depth/assets/depth_anything_v2_vits.pth \
  --samples data/depth/samples.json --output data/generated/depth/reference-v1
python tools/depth/reference.py export \
  --repo /home/qyc/.cache/miniinfer-depth/Depth-Anything-V2-a561b849ebae10a6f5ef49e26c83cbbcd36c71bf \
  --repo-commit a561b849ebae10a6f5ef49e26c83cbbcd36c71bf \
  --weights data/generated/depth/assets/depth_anything_v2_vits.pth \
  --samples data/depth/samples.json --reference-dir data/generated/depth/reference-v1 \
  --output data/generated/depth/export-v1
```

参考逐图保存原始 RGB、PPM、变换参数、预处理张量、原始深度和还原深度；FP32 工件均为 little-endian 连续无头文件。Python 模型在 CPU FP32、eval/no_grad、禁 TF32 条件下运行，默认 4 个线程。静态导出使用 opset 17 和 legacy exporter，输入名 `image`、输出名 `depth`；导出成功本身不代表 ORT / TRT 精度已通过。

## 基础验证

```bash
python -m unittest discover -s tests -p test_depth_reference.py -v
cmake --preset debug -DMINIINFER_DEPTH_PYTHON=/home/qyc/.venvs/miniinfer-depth/bin/python
cmake --build --preset debug
ctest --preset debug
```

`MINIINFER_DEPTH_PYTHON` 为空时不要求深度 Python 依赖，原 CPU / TCP 仍可独立构建。

## V1.2：ORT C++ 执行与验收

```bash
cmake --preset release -DMINIINFER_ENABLE_ORT=ON \
  -DMINIINFER_ORT_ROOT=$HOME/.cache/miniinfer-depth/onnxruntime-linux-x64-1.21.0
cmake --build --preset release
$HOME/builds/MiniInfer/release/bin/depth_cli \
  --model data/generated/depth/export-v1/depth_anything_v2_vits_518.onnx \
  --image data/generated/depth/images/indoor_0.ppm --output data/generated/depth/local
python tools/depth/validate_backend.py \
  --cli $HOME/builds/MiniInfer/release/bin/depth_cli \
  --model data/generated/depth/export-v1/depth_anything_v2_vits_518.onnx \
  --reference data/generated/depth/reference-v1 --samples data/depth/samples.json \
  --split final --thresholds data/depth/thresholds_ort_fp32.json \
  --output data/generated/depth/ort-final-reproduced
```

CLI 同时支持 `--tensor INPUT.f32 --width W --height H` 隔离图执行。它输出 `.preprocessed.f32`、`.raw.f32`、`.depth.f32` 和含几何/计时的 `.json`，拒绝覆盖。`model_seconds` 包括 ORT 同步执行及返回输出，`total_seconds` 额外包含读取、初始化、前后处理，均不含工件写盘；此工具逐图启动，仅用于精度检查，不用于稳态性能报告。

ORT 只通过可选 `miniinfer::depth_ort` 目标链接，默认关闭；图像契约库 `miniinfer::depth_image` 不依赖 ORT、CUDA 或 Python。模型初始化严格校验唯一输入/输出的名称、形状与 FP32 类型。

## V1.3：TensorRT C++ 引擎

`bootstrap_depth.sh` 同时安装 CUDA runtime/compiler headers 与固定 TensorRT/ONNX parser headers，无须系统 nvcc 编译器。项目不在本阶段加入自定义 CUDA kernel。

```bash
bash tools/configure_depth.sh
cmake --build $HOME/builds/MiniInfer/depth-release -j4
mkdir -p data/generated/depth/engines
$HOME/builds/MiniInfer/depth-release/bin/depth_engine_builder \
  --onnx data/generated/depth/export-v1/depth_anything_v2_vits_518.onnx \
  --output data/generated/depth/engines/rebuilt-fp16.engine --precision fp16
$HOME/builds/MiniInfer/depth-release/bin/depth_trt_cli \
  --model data/generated/depth/engines/rebuilt-fp16.engine \
  --image data/generated/depth/images/indoor_0.ppm --output data/generated/depth/trt-local
```

FP32 与 FP16 均关闭 TF32，workspace 上限为 1 GiB。FP16 的最终策略是 `backbone-fp32-head-fp16-v2`，具体开发取舍和数值门槛见[精度报告](depth_accuracy.md)。构建过程在接收业务请求之前完成，输出 engine、`.manifest` 和 `.layers.json`，拒绝覆盖已有路径。

引擎清单记录 ONNX/engine SHA-256、契约及形状、精度策略、GPU 型号/计算能力、CUDA runtime/driver API 版本、TensorRT 版本、平台、workspace、缓存标识及构建耗时。缓存标识覆盖配置，不包含构建耗时或 engine 二进制哈希；运行时重新计算并验证自洽性，再检查当前环境和 engine 哈希。清单是损坏/误配检测，不是数字签名；模型包由部署方可信提供。engine 不承诺跨硬件、驱动或软件版本通用，应在目标环境重建并复验。

`TrtRunner` 拥有单个 context/stream、设备输入输出和 pinned host 缓冲；并发调用在该实例内串行。返回前等待 stream 完成，输出单独拥有存储。设备执行错误会使该槽位失效，后续调用要求重新初始化；非法输入在设备提交前拒绝。构造失败、损坏 engine、配置错配和反复加载有单独检查。更上层的数量/字节接纳在 V1.4 SDK 实现。

`TrtTiming` 仅在显式请求时记录 CUDA events；H2D、compute、D2H 在同一 stream 计时且同步后读取。当前 WSL 实测曾出现 device compute 时间略大于 host wall 时间，两种时钟分别记录，不用相加、相减或大小关系校验设备完成。

## V1.4：同步 SDK 与模型包

`miniinfer::depth_sdk` 提供 `DepthSdk(SdkConfig)` 初始化、`run(RgbImage)` 同步返回 owning `DepthMap`、`config()`、`snapshot()` 和 `shutdown()`。对外头文件不暴露 ORT/TRT 类型。一个实例固定一个后端，默认使用 4 个 Worker，可通过 `SdkConfig::worker_count` 配置；每个 ORT Worker 使用独立 Session，通过线程池提供的固定 Worker 编号选择，能够并行推理不同图片。所有 Session 在线程池启动前创建，排空执行和结果交付后释放；创建中途失败时已创建资源自动释放。

`ort_threads` 是每个 Session 的计算线程数，默认 4 个 Worker × 每 Session 4 个线程，需要结合 CPU 核心数调整。各 Session 有独立的模型状态和内存分配区，这些固定开销不受请求字节预算限制。单独使用 `OrtRunner` 时，同一实例仍通过自身的锁保护 ORT 1.21 的统计字段；不同 Worker 不会争用同一把锁。GPU 后端仍串行访问单个执行槽位。

在途请求数量上限为 `queue_capacity + worker_count`，同时受字节预算限制。调用者可并发调用 `run`，输出在下次调用或 SDK 关闭后仍有效。完整接口、错误分类和外部集成示例见 [examples/depth_sdk](../examples/depth_sdk/README.md)，实测见[每 Worker Session 验收](sdk_worker_sessions_20261014.md)。

默认使用 4 个 Worker 和 4 个等待队列槽位，未交付请求的计数上限为 8，字节预算可能进一步限制可接纳数量；每个请求在复制图像前预留 `19 × 像素数 + 16 MiB`，128 MiB 总预算不足时立即返回 `Busy`。计数和字节预算同时覆盖等待、执行及等待交付的结果，结果交付后释放。固定模型权重、后端 arena、CUDA context 和调用者持有的已交付结果不属于此预算，不能把预算当作进程 RSS 上限。`shutdown()` 先拒绝新请求，再排空已接受任务及结果交付，最后释放后端；支持重复或并发关闭。

模型包生成要求匹配且通过的最终集报告，不会在打包时放宽阈值。包内包含 ONNX、可选 TensorRT engine/环境清单/层格式、契约、阈值、最终验收报告、参考清单、样本来源和一张 golden 图及中间/最终张量。`package.json` 对所有附带文件记录大小和 SHA-256；SDK 初始化校验最小 `manifest.txt`、模型哈希和 TRT 环境，离线完整检查使用 `verify`。包不是数字签名。

```bash
python tools/package_depth.py create --backend trt --precision fp16 \
  --model data/generated/depth/engines/depth-fp16-v2.engine \
  --onnx data/generated/depth/export-v1/depth_anything_v2_vits_518.onnx \
  --reference data/generated/depth/reference-v1 --samples data/depth/samples.json \
  --thresholds data/depth/thresholds_trt_fp16.json \
  --report docs/validation/v13_fp16_final.json --output /tmp/depth-package-fp16
python tools/package_depth.py verify /tmp/depth-package-fp16
cmake --install $HOME/builds/MiniInfer/depth-release --prefix $HOME/miniinfer-install
# 独立 find_package 构建及依赖库路径参数见 examples/depth_sdk/README.md。
source tools/depth_env.sh
python tools/verify_depth_sdk.py --cli /tmp/miniinfer-sdk-example/depth_sdk_example \
  --package /tmp/depth-package-fp16 --output /tmp/sdk-golden-report
```

所有工件输出目录和结果文件默认拒绝覆盖。项目中已生成的三个包位于 `data/generated/depth/packages/{ort-fp32,trt-fp32,trt-fp16}`，大体积模型和图像不提交到 Git；从固定源、哈希与命令重建。

## V1.5：TCP 深度入口

`depth_server --package DIR` 在模型初始化后接纳连接，`depth_client --image PPM --output F32` 完成一次 RGB8 → FP32 深度请求。MINI v2 的完整格式、数量/字节限流、慢读者和关闭语义见 [TCP 契约](depth_tcp.md)。旧 Echo/MLP 使用独立的 MINI v1 路径并持续回归。

## V1.6：重新构建、安装和测量

在已按前文准备好固定依赖、ONNX 和参考工件的 WSL 环境中：

```bash
source tools/depth_env.sh
bash tools/configure_depth.sh "$HOME/builds/MiniInfer/new-depth-release"
cmake --build "$HOME/builds/MiniInfer/new-depth-release" -j4
ctest --test-dir "$HOME/builds/MiniInfer/new-depth-release" --output-on-failure
cmake --install "$HOME/builds/MiniInfer/new-depth-release" --prefix "$HOME/miniinfer-install"
bash tools/rebuild_depth_engines.sh /tmp/depth-rebuilt "$HOME/builds/MiniInfer/new-depth-release"
```

重建脚本依次生成 FP32 和混合 FP16 引擎，使用已冻结的最终集和阈值复验，再打包并执行 SDK golden 回归；任一步失败立即非零退出，保留日志。它不会覆盖旧输出，也不会改阈值。引擎二进制与构建耗时不要求重复构建完全相同，兼容清单与精度验收必须通过。

安装目录提供 `include/`、`lib/`、`bin/`、`lib/cmake/MiniInfer/` 和 `share/MiniInfer/`，最后一项包含契约、部署说明及外部示例源码。安装 SDK 与已验收模型包共同构成部署交付；C++ 在线路径不启动 Python 服务。共享库由记录的兼容环境提供，不复制宿主 NVIDIA 驱动。可从安装目录的 `share/MiniInfer/examples/depth_sdk` 独立构建示例，参数见其 README。

```bash
python tools/run_depth_suite.py \
  --bin-dir "$HOME/builds/MiniInfer/new-depth-release/bin" \
  --packages data/generated/depth/packages --output /tmp/depth-baseline-new
```

基准脚本串行执行三个后端配置的 raw/SDK/TCP 九个案例，每例三个独立调用进程。ORT 每次 100 个稳态请求、5 次预热；TRT 每次 300 个、20 次预热，首请求在预热前独立记录。完整时间边界、失败记账、RSS/设备显存口径见 [基线方法](depth_benchmark_method.md)，实际结果见 [性能报告](depth_performance.md)，真实问题与定位证据见 [排查记录](depth_troubleshooting.md)。
