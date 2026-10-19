# MiniInfer 深度部署契约与复现

当前接口已扩展为拥有型异步 SDK 和四种可选预处理模式，当前用法见文末「V2：当前构建、选择与复现」及 [外部 SDK 示例](../examples/depth_sdk/README.md)。当前质量回归已通过，性能比较和长稳验收尚未全部结束；最终状态与部署组合只以 [V2 升级与验收](depth_v2_upgrade.md)为准，不能据此宣称 V2 全部完成或预选最终 GPU 模式。

以下从模型契约至 V1.6 的章节记录 **V1 历史条件与重建流程**：V1 以 Depth Anything V2 Small 为唯一视觉负载，模型输入固定 `[1,3,518,518]`。阶段完成状态与实测证据见 [V1 验收记录](depth_validation.md)。V1.1—V1.6 分别提交 Git，原 CPU MLP 与 D16 回归继续保留。历史命令和版本限制予以保留；其中用户目录、虚拟环境和缓存绝对路径属于当时本机记录，复现时须替换为自己的兼容环境，不是通用安装位置。V1 的单 GPU 槽位、同步交付关闭和无自定义 CUDA kernel 描述仅适用于该历史阶段，现行语义以文末 V2 章节为准。

## 模型与图像契约

机器可读规范位于 [contract.json](../data/depth/contract.json)。源码固定为官方提交 `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`，Small 权重 SHA-256 为 `715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378`。权重以 `weights_only=True` 严格加载。

外部图像是连续 RGB8 HWC，宽高在 1—4096，像素总数不超过 2,097,152。先将像素转为 FP32 并除以 255，再等比缩放到最长边 518；缩放尺寸采用整数 round-half-up，最短边至少为 1。双线性插值使用 half-pixel 坐标与边界 clamp，不启用抗锯齿。中心 padding 使用 RGB=0，多余一像素放右侧或下侧；随后按 ImageNet mean/std 归一化并变为 NCHW。

Python 中间运算使用 FP32，插值按先水平后垂直的 `a+(b-a)*w` 顺序计算。C++ 对应路径禁用浮点收缩，不使用 fast-math。原始深度为 FP32 `[1,518,518]`，后处理去掉 padding 并按相同插值恢复原图尺寸。对外返回独立拥有的行主序 FP32 HW 数据；数值越大表示越近，单位不是米。伪彩色与逐图 min-max 不参与数值验收。

此几何包装是项目契约，与[官方默认处理](https://github.com/DepthAnything/Depth-Anything-V2/blob/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf/depth_anything_v2/dpt.py)区分；模型图结构和权重保持原样。

## 样本和任务标注

[samples.json](../data/depth/samples.json)固定官方 [DA-2K](https://github.com/DepthAnything/Depth-Anything-V2/blob/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf/DA-2K.md)中的 16 张图像与 30 组独立近远标注。8 个场景各取按路径排序后满足尺寸约束的前 2 张，第 1 张用于开发，第 2 张用于最终验收；选择过程不读取模型预测。原始标注的 `[y,x]` 转为项目 `[x,y]`，保留来源与哈希；相等预测算排序错误，不确定点对排除。

这只是部署与任务保持的小规模检查集，不代表完整 DA-2K 成绩。最终样本不用于选择容差。图片和模型放在忽略的 `data/generated/depth`，Git 保存选择清单、契约、生成工具与检查证据。

## 独立环境与下载

V1 记录中，源码在 Windows 工作区编辑，Linux 工具从 `/mnt/c/Users/ASUS/Documents/MiniInfer` 读取同一份源码。GPU 环境独立于已有 CPU 虚拟环境，默认 `/home/qyc/.venvs/miniinfer-depth`；模型与 C++ 后端依赖缓存默认 `/home/qyc/.cache/miniinfer-depth`。

在项目根目录的 WSL shell 中执行：

```bash
bash tools/bootstrap_depth.sh
```

V1 验证环境固定 PyTorch 2.6.0+cu124、torchvision 0.21.0+cu124、ORT 1.21.0、TensorRT 10.8.0.43，其他直接依赖见 [depth-requirements.txt](../tools/depth-requirements.txt)。选择一组固定版本进行实际兼容验证，不自动追随最新版本。NVIDIA 驱动来自宿主系统，安装脚本不替换驱动。

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

V1.3 的 `bootstrap_depth.sh` 安装 CUDA runtime/compiler headers 与固定 TensorRT/ONNX parser headers；当时未加入自定义 CUDA kernel，所以该历史构建不要求系统 nvcc。V2 的完整 CUDA 预处理构建另需兼容的 toolkit/nvcc，见文末。

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

V1 的 `TrtTiming` 仅在显式请求时记录 CUDA events；H2D、compute、D2H 在同一 stream 计时且同步后读取。当时 WSL 实测曾出现 device compute 时间略大于 host wall 时间，两种时钟分别记录，不用相加、相减或大小关系校验设备完成。

## V1.4：同步 SDK 与模型包

V1.4 的 `miniinfer::depth_sdk` 提供 `DepthSdk(SdkConfig)` 初始化、`run(RgbImage)` 同步返回 owning `DepthMap`、`config()`、`snapshot()` 和 `shutdown()`。对外头文件不暴露 ORT/TRT 类型。一个实例固定一个后端，默认使用 4 个 Worker，可通过 `SdkConfig::worker_count` 配置；每个 ORT Worker 使用独立 Session，通过线程池提供的固定 Worker 编号选择，能够并行推理不同图片。所有 Session 在线程池启动前创建，排空执行和结果交付后释放；创建中途失败时已创建资源自动释放。

`ort_threads` 是每个 Session 的计算线程数，默认 4 个 Worker × 每 Session 4 个线程，需要结合 CPU 核心数调整。各 Session 有独立的模型状态和内存分配区，这些固定开销不受请求字节预算限制。单独使用 `OrtRunner` 时，同一实例仍通过自身的锁保护 ORT 1.21 的统计字段；不同 Worker 不会争用同一把锁。该 V1 实现的 GPU 后端串行访问单个执行槽位；V2 的 `gpu_slots` 已可独立配置为 1 或 2。

在途请求数量上限为 `queue_capacity + worker_count`，同时受字节预算限制。调用者可并发调用 `run`，输出在下次调用或 SDK 关闭后仍有效。完整接口、错误分类和外部集成示例见 [examples/depth_sdk](../examples/depth_sdk/README.md)，实测见[每 Worker Session 验收](sdk_worker_sessions_20261014.md)。

默认使用 4 个 Worker 和 4 个等待队列槽位，未交付请求的计数上限为 8，字节预算可能进一步限制可接纳数量；每个请求在复制图像前预留 `19 × 像素数 + 16 MiB`，128 MiB 总预算不足时立即返回 `Busy`。计数和字节预算同时覆盖等待、执行及等待交付的结果，结果交付后释放。固定模型权重、后端 arena、CUDA context 和调用者持有的已交付结果不属于此预算，不能把预算当作进程 RSS 上限。在该 V1 同步实现中，`shutdown()` 拒绝新请求后还等待已接受任务的执行及同步结果交付，再释放后端；支持重复或并发关闭。V2 已改为只排空执行，不等待 ready 句柄消费。

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


## V2：当前构建、选择与复现

### 契约与模式

V2 保留 V1 的 RGB8 输入限制、half-pixel/clamp 插值、中心补边、ImageNet 归一化、相对深度语义和独立拥有的返回结果。静态尺寸由模型包决定：`depth-v1` 对应 518，`depth-v2-392` 对应 392，分别见 [518 契约](../data/depth/contract.json)和 [392 契约](../data/depth/contract_392.json)。392 必须使用同尺寸的 ONNX、engine、参考、阈值和包身份，不能只改尺寸后复用 518 工件。V1 的权重提交/SHA、opset 17、输入输出名称和版本兼容检查继续适用；engine 仍须在目标环境重建并复验。

`SdkConfig::preprocess_mode` 默认 `PreprocessMode::Cpu`，保留旧调用的 CPU 预处理行为；GPU 模式显式选择。`parse_preprocess_mode` / `preprocess_mode_name` 使用下列精确字符串，拒绝未知字符串或枚举：

| 路径 | MODE / 枚举 | CPU 工作 | 上传与 GPU 工作 |
|---|---|---|---|
| A | `cpu` / `Cpu` | 转 FP32、resize、补边、归一化与 NCHW | TRT 上传 FP32 NCHW；ORT 也支持此模式 |
| B | `cuda-partial` / `CudaPartial` | 转 FP32、resize、补边为 HWC | 上传 FP32 HWC，GPU 归一化/转换布局后推理 |
| C-basic | `cuda-full-basic` / `CudaFullBasic` | 几何元数据 | 原始 RGB8 上传，GPU 完整预处理直接写 TRT input |
| C-fused | `cuda-full-fused` / `CudaFullFused` | 几何元数据 | 同 C-basic，复用 RGB 三通道坐标计算 |

B 的 CPU 工作在取得 GPU 槽位前完成；C 不把中间 NCHW 下载回 CPU 再上传。`TrtRunner::run_raw(PreparedImage)` 保留 A 的 FP32 上传语义；`run_image_raw` 按实例模式执行。验证专用 `captured_nchw` 非空时，在模型执行前回读实际 device input，性能路径不传此参数。图像 CLI 为写出 `.preprocessed.f32` 会启用验证回读，不能用其逐图 `model_seconds` 代替稳态测量。

ORT 拒绝 B/C；没有编译 CUDA preprocessing 的 TRT 构建也在初始化 CUDA 前明确拒绝 B/C。模型包决定后端、尺寸和精度，不隐式决定预处理模式。V2 P0的质量、性能、1800秒长稳与交付验证已通过；最终结果与选择见 [V2 升级与验收](depth_v2_upgrade.md)。

### CUDA 构建与外部消费

项目仍要求 Linux/WSL、CMake 3.28+ 和兼容 GCC/Clang。构建方提供与 TensorRT、CUDA runtime、驱动及主机编译器匹配的依赖和 CUDA toolkit；V1 的 bootstrap 版本记录不等于已提供 nvcc。以下路径需替换为部署方实际依赖位置；`89` 只是目标计算能力示例，应改为目标 GPU 对应值。使用新的构建目录：

```bash
export BUILD="$HOME/builds/MiniInfer/depth-v2-cuda"
export PREFIX="$HOME/miniinfer-v2-install"
export CUDACXX=/path/to/cuda-toolkit/bin/nvcc
export MINIINFER_BUILD_CUDA_PREPROCESS=1
export MINIINFER_CUDA_ARCHITECTURES=89
export MINIINFER_CUDA_FLAGS=''
export DEPTH_PYTHON=/path/to/depth-venv/bin/python
export LD_LIBRARY_PATH="/path/to/tensorrt_libs:/path/to/nvidia/cuda_runtime/lib:/path/to/onnxruntime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cmake -S . -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON -DMINIINFER_ENABLE_ORT=ON -DMINIINFER_ENABLE_TRT=ON \
  -DMINIINFER_ENABLE_CUDA_PREPROCESS="$MINIINFER_BUILD_CUDA_PREPROCESS" \
  -DCMAKE_CUDA_ARCHITECTURES="$MINIINFER_CUDA_ARCHITECTURES" \
  -DCMAKE_CUDA_FLAGS="$MINIINFER_CUDA_FLAGS" \
  -DMINIINFER_DEPTH_PYTHON="$DEPTH_PYTHON" \
  -DMINIINFER_ORT_ROOT=/path/to/onnxruntime \
  -DMINIINFER_TRT_INCLUDE=/path/to/tensorrt/include \
  -DMINIINFER_TRT_LIB=/path/to/tensorrt_libs \
  -DMINIINFER_CUDA_RUNTIME=/path/to/nvidia/cuda_runtime \
  -DMINIINFER_CUDA_NVCC_INCLUDE=/path/to/cuda-toolkit/include
cmake --build "$BUILD" -j4
ctest --test-dir "$BUILD" --output-on-failure
cmake --install "$BUILD" --prefix "$PREFIX"
```

`MINIINFER_CUDA_RUNTIME` 是含 `include/` 和 `lib/` 的 runtime 根目录；compiler headers 目录须含 `crt/host_defines.h`；TensorRT headers 须含 `NvInfer.h` 和 `NvOnnxParser.h`。这些依赖和运行时 loader 路径由兼容部署环境提供。可关闭未使用的 ORT；CUDA preprocessing 要求 TensorRT。`MINIINFER_DEPTH_PYTHON` 用于 Python 质量/回归工具，在线 C++ 推理不启动 Python。

默认不绕过 nvcc 的主机编译器检查。部署方已验证特定 toolkit/host compiler 组合且确需显式 override 时，在重新配置前设置：

```bash
export MINIINFER_CUDA_FLAGS='--allow-unsupported-compiler'
```

它只跳过版本检查，不证明二进制兼容；实际版本与此参数要随构建证据保存。便利脚本 `tools/configure_depth.sh "$BUILD" Release` 读取 `CUDACXX`、`MINIINFER_BUILD_CUDA_PREPROCESS`、`MINIINFER_CUDA_ARCHITECTURES` 和 `MINIINFER_CUDA_FLAGS`，但要求 `MINIINFER_DEPTH_ENV` Python 环境和 `MINIINFER_DEPTH_CACHE` 固定依赖布局已准备好。自有依赖布局可直接使用上述显式 CMake 路径，不必复制某台机器的缓存目录。

仅用 A/ORT 或 TRT CPU 预处理时，可以关闭 `MINIINFER_ENABLE_CUDA_PREPROCESS`，沿用前文构建流程而不要求 nvcc。外部消费方继续 `find_package(MiniInfer CONFIG REQUIRED)` 并链接 `miniinfer::depth_sdk`，指定安装前缀和已启用后端的依赖库位置。消费已构建 static SDK 不需要 nvcc 编译自己的 C++ 项目，仍须提供匹配的 ORT/TRT/CUDA runtime 共享库，完整命令见 [外部示例](../examples/depth_sdk/README.md)。

### 当前 SDK、CLI 与服务器

`submit(image)` 返回前复制输入，返回 move-only `DepthRequest`。`id()` 关联请求，`ready()` 查询执行是否终结，`wait()` 等待；`wait_for(milliseconds)` 超时仅停止等待，不取消、不释放额度。`get(&timing)` 一次性交付 owning 结果或存储错误，异常路径也消费句柄并归还额度；`run(image, &timing)` 使用相同路径。ready 但未 get 的结果继续占用数量和字节预算。

`abandon()` 或句柄析构放弃交付：ready 请求立即释放，等待/执行中的请求继续安全执行，结束后释放；不提供 GPU 取消。`shutdown()` 拒绝新任务、排空已接受执行、释放后端，不等待 ready 句柄消费。句柄与结果可晚于 SDK 销毁存在，保留句柄仍可 get。访问 SDK 对象的提交/同步调用/关闭须在析构前完成；同一请求句柄不可由多线程同时操作。支持重复/并发关闭，FIFO 出队允许乱序完成。可复制 C++ 用例见 [外部示例](../examples/depth_sdk/README.md)。

`worker_count` 默认 4，控制 CPU Worker 数；每 Worker 独占 ORT Session 和默认复用的 CPU workspace，`ort_threads` 默认 4、作用于每个 Session；同一 `OrtRunner` 的内部锁保留。`gpu_slots` 独立配置为 1 或 2，不把 ORT Worker 数改成槽位数。每个 TRT 槽位独占 engine/context、stream、events、I/O 和 pinned 缓冲，双槽位增加固定资源但不保证加速。`reuse_host_workspace` / `reuse_gpu_buffers` 默认 true，可显式关闭做消融。

```bash
export PACKAGE=/path/to/validated-model-package
export MODE=cpu  # 配置示例；B/C 需要 CUDA 构建与 TRT 包。
"$BUILD/bin/depth_sdk_cli" "$PACKAGE" input.ppm /tmp/depth-output-new.f32 "$MODE" 1
"$BUILD/bin/depth_server" --package "$PACKAGE" --host 127.0.0.1 --port 9001 \
  --preprocess "$MODE" --gpu-slots 1 --sdk-workers 4
```

SDK CLI 尾部参数是 `[MODE [SLOTS]]`，默认 cpu、1；服务器使用 `--preprocess`、`--gpu-slots`、`--sdk-workers`。MINI v2 线协议仍为 RGB8 → FP32 相对深度；网络连接、网络请求预算与 SDK 预算分别配置，见 [TCP 契约](depth_tcp.md)。

### 配额、上传容量与计时

默认等待队列 4、Worker 4、请求预算 128 MiB；接纳前预留 `19 × 原图像素数 + 16 MiB`。等待、执行、ready 未消费结果都计入 `queue_capacity + worker_count` 和字节预算，数量之和须能由 `size_t` 表示。等待队列满或总数量/字节不足返回 Busy，关闭后提交返回 Closed；致命 GPU 故障也会关闭接纳并让已接受工作安全终结。额度在 get（含异常）或安全放弃完成时只释放一次。

C 的原始 RGB8 每槽 pinned/device staging 上限为 `3 × 2,097,152 = 6,291,456` 字节，按原图最大像素数预留；实际上传 `3 × 原图宽 × 原图高`，可能大于或小于固定模型输入。A/B 上传 `3 × S × S × sizeof(float)`，518 为 3,219,888 字节，392 为 1,843,968 字节。C 仍需 FP32 模型输入、深度输出和上下文，RGB8 上传量不是全部显存预算。

`snapshot()` 的 live I/O、固定 host workspace、分配次数、峰值和静态需求是不同口径。请求预算不含这些固定资源、模型权重、ORT arena、CUDA context，以及调用者交付后保留的结果，不是 RSS 或完整 VRAM 上限。`context_required_bytes` 是 TRT 静态上下文需求，`engine_serialized_bytes` 是每槽序列化 plan 大小之和，不是精确 GPU 权重驻留。关闭后 live I/O/host 字节归零，次数、峰值与 engine/context 指标保留历史值；槽位峰值相加不意味着峰值同时发生。设备总显存还可包含桌面等其他进程。

`h2d_ms`、`gpu_preprocess_ms`、`model_ms`、`d2h_ms` 是每槽同一 stream 的 CUDA-event 区间，同步后读取；`TrtTiming::compute_ms` 保留为模型阶段别名。host wall 包括队列、CPU 预处理、`slot_wait_ms`、backend 调用、`synchronize_ms` 和后处理。`pinned_copy_ms` 仅输入复制到 pinned staging，输出拥有化复制留在 host_ms/backend_ms 余量。backend_ms 已包含槽位等待，host 同步又与设备执行重叠，字段有包含关系且使用不同钟源，不相加、不相减推导重叠、不用大小关系证明完成；重叠需独立 Nsight 时间线。`ready_at` 是终结发布时刻，`h2d_bytes` 记录成功调用的实际上传（ORT 为零），不包含失败执行中可能已发起的传输。

### V2 复现与结果状态

使用固定 [V2 样本](../data/depth/samples_v2.json)与同尺寸/精度的 engine、参考、阈值，以及新的输出目录。先验包和 SDK golden。MODE=cpu 为兼容示例，比较 B/C 时显式更改，不代表最终选择：

```bash
python tools/package_depth.py verify "$PACKAGE"
python tools/verify_depth_sdk.py --cli "$BUILD/bin/depth_sdk_cli" \
  --package "$PACKAGE" --preprocess "$MODE" --slots 1 --output /tmp/sdk-v2-golden-new
python tools/run_depth_replay.py --executable "$BUILD/bin/depth_replay" \
  --package "$PACKAGE" --backend trt --images-list data/depth/replay_v2.txt \
  --output /tmp/depth-v2-replay-new --duration 50 --warmup 2 --rate 20 \
  --workers 8 --sdk-workers 4 --max-outstanding 8 --queue 4 --budget 134217728 \
  --async-sdk --slots 1 --preprocess "$MODE" --repeats 3
```

ORT 回放使用 ORT 包、`--backend ort`、`--preprocess cpu` 和单槽位参数。`tools/run_depth_v24.py --executable BUILD/bin/depth_replay --package PACKAGE --images-list data/depth/replay_v2.txt --output NEW_DIR` 是固定四模式、normal/overload/单在途探针的比较入口，可用 --mode/--load 缩小一次重现范围。完整模型和实际 device NCHW 检查由 `tools/validate_depth_gpu_pipeline.py --cli BUILD/bin/test_depth_gpu_pipeline --output NEW_DIR` 驱动，按参数显式提供兼容 engine 与同尺寸冻结参考，不能仅以 kernel 测试替代模型质量。保留输入/二进制哈希、各轮日志、拒绝/失败、资源遥测与真实供电/时钟/后台条件；profile 与正常性能分开，不并行构建或运行其他 GPU 负载。

本机已选定518 FP32、`cuda-full-basic`、1个GPU槽位、4个SDK Worker，两个复用开关开启；队列4、请求预算134217728字节。[v2_selected.json](../data/depth/v2_selected.json)记录模型哈希及决策来源，需显式转换为 `SdkConfig` 或CLI参数，不会被SDK自动加载。其对应枚举为 `PreprocessMode::CudaFullBasic`；兼容默认模式仍为 `Cpu`。

稳定性复现使用新的输出目录，各短场景运行三轮；`soak` 会在预热后真实执行1800秒，并重新解析原始CSV进行严格验收：

```bash
MODE=cuda-full-basic
for CASE in normal saturation-probe overload slow-consumer soak; do
  python tools/run_depth_v25.py --executable "$BUILD/bin/depth_replay" \
    --package "$PACKAGE" --images-list data/depth/replay_v2.txt \
    --output /tmp/depth-v2-stability-new --preprocess "$MODE" --slots 1 --case "$CASE" || break
done
```

输出抽检与SDK每秒采样包含在该入口中；正式开始前同步采样以锁定复用分配基线。通过结果应同时包含 `soak-acceptance.json` 的 `passed=true` 和 `require_soak=true`，不能用配置中的1800秒、短preflight或被中断记录替代。完整结束后可用 `tools/summarize_depth_soak.py AUDIT --output NEW.json`生成逐分钟P50/P95/P99和成功吞吐，不平均P99。

最终交付目录采用 `sdk/`（安装SDK）、`model/`（通过清单验证的模型包）、`evidence/`（选择、最终质量和长稳审计）及 `inventory.json`（逐文件字节数/SHA256）。`tools/assemble_depth_delivery.py --sdk-install INSTALL --package PACKAGE --selection data/depth/v2_selected.json --quality FINAL_QUALITY.json --stability PASSED_SOAK_AUDIT.json --output NEW_BUNDLE` 会检查模型/路径/槽位与证据身份，复制后重新核对哈希。驱动、CUDA runtime、TensorRT和ORT共享库由目标环境提供；更换环境须重新构建engine并验收。

V2.1–V2.6 P0已验收，证据与限制统一见 [V2 升级与验收](depth_v2_upgrade.md)。本机交付目录为 `data/generated/depth/v2-delivery`，其中的SDK已通过独立C++工程和最终8图、16次TCP请求验证；P1/E1仍未计入成果。兼容默认值与示例参数不替代显式部署选择。
