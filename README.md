# MiniInfer

MiniInfer 是一个基于 **C++17 / Linux（支持 WSL2）** 的深度估计推理运行时，将训练好的 **Depth Anything V2 Small** 封装成可供其他 C++ 程序调用的同步 SDK。输入一张 RGB 图片，返回与原图尺寸一致的 FP32 相对深度图；数值越大表示越近，单位不是米。

模型计算由 ONNX Runtime 或 TensorRT 执行，项目重点是 **C++ 资源管理、并发请求调度、异常安全和部署验证**。

```text
接收图片 → 校验与资源预留 → 深拷贝并入队 → Worker 执行
        → 预处理 → ORT / TensorRT → 后处理 → 返回独立拥有数据的结果
```

## 核心能力

- **CPU / GPU 推理**：支持 ORT CPU FP32、TensorRT FP32 与通过精度验收的混合 FP16。模型输入固定为 `[1, 3, 518, 518]`，SDK 完成等比缩放、补边、归一化及输出尺寸还原。
- **同步接口与并发接入**：`DepthSdk::run()` 等待结果后返回；内部通过线程池和 `std::future` 调度任务、传递结果与异常。默认 4 个 Worker，可配置为 1；多个请求的预处理、后处理可并行。ORT 1.21 共享 Session 的执行加锁，避免其内部统计数据竞争；TensorRT 串行使用单个 GPU 执行槽位。
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

## 测试与项目范围

测试覆盖 Python 参考结果对齐、输入边界、数量 / 字节限流、故障注入、并发关闭、任务排空及输入 / 输出生命周期。通过可控阻塞稳定构造竞争场景，检查异常后资源释放和后续请求恢复。最新的 Sanitizer 检查、1/4/8 并发 P50 / P99 / QPS 及具体覆盖范围见[本次验收记录](docs/sdk_validation_20261014.md)。

仓库同时保留早期手写 CPU MLP 算子、支持 `submit()` / `run()` 的 Runtime 及 TCP 基础实现。深度 SDK 当前对外提供同步接口；公开异步接口、GPU 多槽位并行、自定义 CUDA 优化和长时间稳定性测试属于后续工作。历史性能与测试结论对应各自记录的版本和配置。

- [分版本验收与证据](docs/depth_validation.md)、[精度对齐](docs/depth_accuracy.md)
- [性能基线与测量范围](docs/depth_performance.md)
- [TCP 协议](docs/depth_tcp.md)、[部署排查](docs/depth_troubleshooting.md)

<details>
<summary>展开 V0.1 / D16 历史开发与验收记录</summary>

## V0.1 / D16 历史记录

以下 CPU/D16 状态和“后续计划”保留为 V1 实施前的历史快照；当前 V1 进度以以上入口为准。

MiniInfer v0.1 是一个面向 Linux / WSL2 的 C++17 CPU 推理项目：固定 `64 → 128 → 10` MLP，实现标量 FP32 算子、资源所有权、有界任务队列与安全关闭。完整范围见 [项目开发规范](<MiniInfer 项目说明与 AI Agent 开发规范.md>)。

当前完成 **D1–D15 的 CPU 基础及 D16 TCP 扩展**：同步 / 异步推理、有界计算与连接队列、远程 MLP、关闭竞争、阶段计时及 Release 对照。Debug / Release / ASan+UBSan 各 **24/24 CTest 通过**；独立 TSan 的 9 项网络测试均通过（关闭测试修正线程运行时基线后单项复验，详见文末）。D13 的 perf 未安装，未实施可选优化。

设计与当前边界见 [架构说明](docs/architecture.md)，测量方法及原始结果见 [基准报告](docs/benchmark.md)。[D10 开发交接](docs/HANDOFF_D10.md)保留为历史快照，当前进度以本 README 和源码为准。

## 后续计划

2026-10-11 已完成 D16，逐阶段验收记录见文末。V1 / V2 仍为待实施计划，按 I【洞察】→ S【策略】→ O【行动】聚焦 **MiniInfer：C++ 视觉模型部署与推理性能优化**，以 Depth Anything V2 Small 为实际负载。主线对应 C++ AI 软件、模型部署与推理性能优化，车企 / 机器人 / 边缘计算作为重点应用方向。

| 阶段 | 目标与进入下一阶段的条件 |
|---|---|
| [D16 原生 TCP 网络编程与推理服务](docs/D16_TCP_网络编程学习扩展.md) | Echo → 消息协议 → 远程 MLP → 有限并发 → 安全退出；正确收发、过载、断连和关闭有测试 |
| [V1 P0：正确部署与 C++ SDK](docs/MiniInfer_V1_V2_求职改进策略.md) | Python → ONNX → ORT → TensorRT FP32 / FP16 精度对齐；固定形状、Batch=1、单槽位；C++ SDK、本地 / TCP、可重建模型包与启动 / 稳态基线 |
| [V2 P0：可靠异步与性能优化](docs/MiniInfer_V1_V2_求职改进策略.md) | 两种静态尺寸 / 精度取舍；可靠异步 SDK、有界 FIFO、GPU 缓冲复用与 1 / 2 槽位；实际链路 CUDA 优化、Profiler、P99 / 吞吐 / 内存、过载与长稳 |
| P1：选做分支 | INT8 PTQ、实时帧策略、epoll / eventfd、独立 LayerNorm / GEMM、TensorRT Plugin、ROS2 等，按岗位或瓶颈每次先选一个 |

D16 已提供 TCP 服务；当前没有 GPU 后端。现有电脑和 NVIDIA GPU 用于完成 V1 / V2 本机验证，真实端侧适配另列 E1，未验证板端时延或功耗。P0 全部验收后才能标记对应版本完成；P1 / E1 不阻塞主线。INT8 若选做，仍须证明真实量化执行并完成精度 / 性能对照，最终可保留 FP16。详细验收以主计划为准。

## 已实现

- [导出脚本](tools/export_mlp.py)：固定分层划分、训练参数与随机种子，导出连续行主序、little-endian FP32 权重；W1 / W2 使用 `[out,in]` 布局。
- [正式参考工件](data/reference/manifest.json)：四份权重、64 个固定输入、原始 logits、预测类别、真值及原数据索引；manifest 记录参数、环境和 SHA-256。
- [数据契约](data/README.md)：输入预处理固定为 `pixel / 16.0`，导出输入已经归一化；分类取 `argmax(logits)`，并列取最小索引。
- [契约测试](tests/test_export_mlp.py)：工件完整性、样本来源和顺序、独立标量 FP32 前向计算、重复导出、非法参数与已有文件保护。
- [CMake 工程](CMakeLists.txt)与 [构建配置](CMakePresets.json)：C++17 静态库 `miniinfer_core`、`infer_cli`、CTest；Debug / Release / ASan+UBSan / TSan 使用独立 Linux 构建目录。
- [推理 CLI](apps/infer_cli.cpp)：提供 `--help`、`-h`、`--version` 及实际推理；成功输出标签和 logits JSON，参数错误退出 2，加载 / 执行错误退出 1。
- [Buffer](include/miniinfer/buffer.hpp) 与 [安全乘法](include/miniinfer/checked_size.hpp)：独占 CPU FP32 内存、自动释放、禁止复制、支持无异常移动；尺寸错误在分配前拒绝。
- [TensorView](include/miniinfer/tensor_view.hpp)：借用连续 CPU FP32 内存，区分可写和只读视图，检查指针、形状、容量及访问边界。
- [MlpModel](include/miniinfer/mlp_model.hpp) 与 [加载器](src/mlp_model.cpp)：拥有四份固定形状权重，只暴露只读视图；严格校验字节数、读取结果和有限数值。
- [CpuOps](include/miniinfer/cpu_ops.hpp)、[CpuExecutor](include/miniinfer/cpu_executor.hpp) 与 [Workspace](include/miniinfer/workspace.hpp)：标量 Linear / 原地 ReLU / Argmax，串联两层 MLP 并复用隐藏层存储。
- [Runtime](include/miniinfer/runtime.hpp) 与 [ThreadPool](include/miniinfer/thread_pool.hpp)：输入 / 结果拥有数据，固定 Worker ID、有界等待队列、future 异常交付、drain / join。
- [基准程序](apps/benchmark_cli.cpp)：direct_cpu / Runtime 1/2/4 Worker、有限 future 窗口、成功 / 失败 / 拒绝记账、阶段延迟、独立直接执行器分配探测和 RSS；[采集驱动](tools/run_runtime_baseline.py)按独立进程保存重复结果。
- [Runtime 演示](apps/runtime_demo.cpp)：混合 Batch、拥有型输入、future 溢出异常与恢复、重复关闭、关闭后拒绝、析构后保留结果并对照参考。
- [TCP 网络层](include/miniinfer/net/server.hpp)：独立 `miniinfer_net` 目标，fd RAII、大端帧协议、绝对收发截止时间、固定连接线程、有界等待连接、推理错误映射和 drain / join；提供 Echo 与远程推理两个客户端/服务端入口。

## Buffer 与尺寸计算契约

`Buffer(element_count)` 拥有固定长度、连续且满足 `float` 对齐的 FP32 数组，非零长度的初值为零。`size()` 是 float 元素数，`size_bytes()` 是字节数；默认构造与 `Buffer(0)` 均为空，`data()` 返回空指针。当前没有扩容接口。

移动构造和赋值转移所有权与底层地址，源对象变为空且可以再次赋值使用。移动赋值自动释放目标原有内存，自移动保持有效；复制构造和赋值被删除。`data()` 是借用指针，const Buffer 只暴露 `const float*`；借用不延长内存寿命，移动后的借用依赖新所有者存活。

`checked_multiply(a,b)` 接受零因子，拒绝 `size_t` 乘法溢出并抛出 `std::overflow_error`。Buffer 先检查元素数乘 `sizeof(float)`，再拒绝超过 `PTRDIFF_MAX` 字节的对象并抛出 `std::length_error`，之后才分配；分配异常直接传播。TensorView / MlpModel 管理形状与模型语义。

## TensorView 与模型契约

`TensorView(data, available_elements, rows, cols)` 借用连续、行主序的 CPU FP32 内存。维度必须大于零，指针非空且满足 float 对齐，可用元素数至少覆盖形状；多余容量可留给后续 Workspace。调用者必须如实提供可访问容量并保持存储存活，视图不分配、不释放，也不能承载设备地址。

`rows()` / `cols()` 表示形状，`size()` / `size_bytes()` 表示逻辑元素数 / 字节数；`row(r)` 和 `at(r,c)` 检查索引。非法指针、维度或容量抛出 `std::invalid_argument`，乘法溢出抛出 `std::overflow_error`，超过 `PTRDIFF_MAX` 字节抛出 `std::length_error`，访问越界抛出 `std::out_of_range`。`ConstTensorView` 只返回 `const float*` / `const float&`；`as_const()` 将可写视图转为只读视图。const TensorView 本身仍可写底层数据；跨线程读写由调用者同步。

`MlpModel::load(directory)` 只在四份权重全部加载成功后返回模型。W1 / W2 视图为 `[128,64]` / `[10,128]`，沿用 `weights[out,in]`；偏置视图使用 `[1,128]` / `[1,10]`，二进制布局与数据契约的 `[128]` / `[10]` 相同。加载器验证 little-endian IEEE 754 FP32 平台，拒绝目录错误、缺失或非普通文件、过短 / 过长数据、读取失败以及 NaN / Inf；文件错误以含路径的 `std::runtime_error` 报告，分配异常直接传播，部分加载资源由 RAII 回收。

模型禁止复制和赋值，可无异常移动构造；getter 仅在存活的左值模型上提供只读借用。模型应在发布给 Worker 前完成加载和移动；发布后可以通过 const 引用并发读取，读取期间不得移动或销毁所有者。视图不延长模型寿命。V1 / V2 的设备存储及真实模型契约按策略独立增加。

## CPU 推理与请求所有权

`cpu_ops::linear(input, weights, bias, output)` 接受 `[B,in]`、`[out,in]`、`[1,out]`、`[B,out]`，按 FP32 标量顺序累加后加 bias；output 不得与任何输入重叠。`relu(view)` 原地执行，负数和正 / 负零变为正零；`argmax(logits, labels, available_labels)` 为每行写一个 int32 类别，并列取最小索引，labels 不得与 logits 重叠。三个算子在写入前检查输入有限性、形状与必要的容量 / 地址范围；参数错误抛出 `std::invalid_argument`，Linear 计算产生非有限结果抛出 `std::overflow_error`。算术错误不保证 output 回滚，执行器不会返回部分结果。

`MlpInput(batch, values)` 拥有 `[B,64]` 的归一化有限 FP32 数据；B 必须正且元素数精确匹配。传入 vector 左值产生独立副本，右值转移存储。`Workspace(max_batch=256)` 预分配 `max_batch*128` 个 float；`hidden(batch)` 仅借用逻辑前缀，不重新分配。Workspace 禁止复制 / 赋值，可移动构造，移动源容量归零。一次直接执行独占一个 Workspace；它不能供多个线程同时写入。

`CpuExecutor{}.run(model, input, workspace)` 同步返回独立拥有 logits `[B,10]` 与 labels `[B]` 的 `MlpResult`。B 超过 Workspace 容量立即拒绝；尺寸乘法先检查溢出，分配失败传播。Result 禁止复制、支持移动，后续执行或销毁 Runtime 都不改变已返回结果。Input / Result 的视图不延长所有者寿命；赋值、移动及销毁时需停止使用失效的借用，移动源不用于推理。

## Runtime 与有界线程池

`Runtime(MlpModel model, RuntimeConfig config={})` 独占模型，默认 `workers=1, queue_capacity=64, max_batch=256`。所有配置必须正；每 Worker 的 Workspace 在线程启动前准备，模型只读共享。`submit(MlpInput input)` 返回 `future<MlpResult>`：左值复制，右值移动；`run(input)` 等价于 `submit(std::move(input)).get()`，包含入队成本及拒绝行为，与直接 CpuExecutor 基线区分。

队列预分配固定环形槽位，容量只计算等待任务，执行中的任务已出队。接收判断和入队在同一 mutex 下完成；满时抛出 `QueueFull`，关闭后抛出 `PoolClosed`（关闭优先于满队列），不等待空位。Input 校验发生在调用线程；被接受任务的执行错误通过该 future 交付，Worker 继续执行后续任务。任务与 future / Result 的额外存储不包含在等待队列容量中。

`shutdown()` 先停止接收并唤醒 Worker，再排空已接受任务，在队列锁外 join；多个关闭调用由单独 mutex 序列化，重复调用有效。状态依次为 Running、Draining、Stopped，分别由接收标志与 join 完成表示。析构自动执行该流程，线程池先于 Workspace 和模型销毁；部分线程启动失败时也关闭并 join 已启动线程。

通用 ThreadPool 的任务获得稳定 Worker ID，不依赖 MLP 或算子。支持外部调用者并发 submit / shutdown；调用者须保证池 / Runtime 对象本身存活，不能在所属 Worker 回调中销毁池、阻塞等待同池任务或调用 shutdown（自关闭抛出 `std::logic_error`）。可选第三个构造参数 ThreadFactory 用于启动失败注入，默认 std::thread；创建失败关闭并回收已启动线程，非 joinable 工厂结果被拒绝。这些路径及关闭竞争已经测试。

`submit(input, shared_ptr<TaskTiming>)` 可选记录入队 / 开始 / 结束时间，普通 submit 不读时钟。每请求使用独立记录，只在对应 future 完成后读取；拒绝没有完成记录。记录的可见性与错误时完成边界见架构说明。

## 本机运行方式

Codex 和 Windows VS Code 使用 `C:\Users\ASUS\Documents\MiniInfer` 主目录。Python 由 Windows PowerShell 调用 WSL 执行；已有虚拟环境只提供解释器和依赖，源码始终来自 Windows 主目录。

```powershell
Set-Location -LiteralPath 'C:\Users\ASUS\Documents\MiniInfer'
$linuxProject = '/mnt/c/Users/ASUS/Documents/MiniInfer'
$linuxPython = '/home/qyc/projects/MiniInfer/.venv/bin/python'

wsl -d Ubuntu-24.04 --cd $linuxProject -- $linuxPython -m unittest discover -s tests -p test_export_mlp.py -v
```

重新生成时使用新的空目录；脚本会拒绝覆盖非空目录：

```powershell
wsl -d Ubuntu-24.04 --cd $linuxProject -- $linuxPython tools/export_mlp.py --output-dir data/generated/digits_seed42 --samples 64 --seed 42 --max-iter 500
```

Python 基线为 3.12.3，包版本见 [requirements](tools/requirements.txt)。当前环境已可直接运行；新环境重新导出才需要安装该依赖清单。C++ 推理只需现有参考工件，不需要训练或 Python 包；CTest 共 24 项，其中 5 项使用 Python 标准库验证程序及基准驱动，无 Python 时注册 19 项。D1 的 Python 导出测试单独运行。

## C++ 构建、测试与 CLI

本机使用 CMake 3.28.3、Ninja 1.11.1、GCC 13.3.0。Presets 限定在 Linux / WSL 运行，构建输出为 `$HOME/builds/MiniInfer/<preset>`，当前即 `/home/qyc/builds/MiniInfer/`。在 Windows PowerShell 顺序执行，每步成功后再继续：

```powershell
$linuxProject = '/mnt/c/Users/ASUS/Documents/MiniInfer'
wsl -d Ubuntu-24.04 --cd $linuxProject -- cmake --preset debug
wsl -d Ubuntu-24.04 --cd $linuxProject -- cmake --build --preset debug
wsl -d Ubuntu-24.04 --cd $linuxProject -- ctest --preset debug
wsl -d Ubuntu-24.04 -- /home/qyc/builds/MiniInfer/debug/bin/infer_cli --help
```

Release 使用相同的三个构建/测试命令，将 preset 名称改成 `release`。每个 CTest preset 在没有注册测试时返回错误。`model_contract` 加载正式权重；`inference_contract` 对照正式输入 / logits / labels；CLI 测试执行 64 样本推理。

执行正式参考输入，输出单行 JSON（输入文件必须精确包含 `batch*64` 个 FP32 值，已经归一化）：

```powershell
wsl -d Ubuntu-24.04 --cd $linuxProject --exec /home/qyc/builds/MiniInfer/release/bin/infer_cli --model data/reference --input data/reference/inputs.f32 --batch 64
```

同步及 TCP 基准命令与原始测量见 [benchmark](docs/benchmark.md)。`benchmark_cli` 支持直接 CPU 与 Runtime 1/2/4 Worker 对照；`infer_client` 从发送前计时到完整响应校验完成，建连耗时单列。

Windows VS Code 直接打开 Windows 主目录：`Ctrl+Shift+B` 使用默认的 `MiniInfer: build Debug (WSL)`；“终端 → 运行任务”可选择 `MiniInfer: test Debug (WSL)` 或对应 Release 任务。测试任务依次完成配置、构建、CTest；定义见 [.vscode/tasks.json](.vscode/tasks.json)。这些任务使用当前项目的固定 WSL 路径，移动源码目录时需同步修改。

`asan-ubsan` 和 `tsan` preset 分别启用独立安全检查选项；同时开启 `MINIINFER_ENABLE_ASAN_UBSAN` 与 `MINIINFER_ENABLE_TSAN` 会在配置时报错。全套 ASan+UBSan 与独立 TSan 并发检查已经执行，命令及限制见下方 D11–D15 记录。目录、工具和后续升级约定见 [技术栈](docs/tech_stack.md)。

## D1 验收记录

2026-10-10，在上述 WSL Python 环境执行：

- 4 项 unittest 全部通过，耗时 14.656 秒；包括已保存的正式参考工件。
- 默认参数重新训练并导出，全部 10 个文件（含 manifest）与正式工件逐字节一致；只验证了同一环境内的复现。
- 训练集 1437 项，留出测试集 360 项；实际训练 189 次迭代，无收敛告警。
- 导出 FP32 模型的留出准确率为 **354/360 = 98.33%**，固定参考子集为 **62/64 = 96.88%**；两个集合的预测均与 sklearn 一致。
- 标量计算与参考 logits 满足 `abs(actual-reference) <= 1e-4 + 1e-4 × abs(reference)`，预测类别一致。

以上是 D1 的 Python 导出与数据契约证据。C++ 前向推理、安全检查和同步基准的新增证据见下方 D5–D10 记录。

## D2 验收记录

2026-10-10，通过 Windows PowerShell 调用 WSL 实际执行：

- Debug、Release 均从不存在的构建目录完成配置与构建，无编译警告；所有 C++ 编译单元使用 `-std=c++17 -Wall -Wextra -Wpedantic`。
- 两个配置的 CTest 均 **2/2 通过**：`project_contract` 检查版本格式，`cli_contract` 检查六种 CLI 调用及输出、退出码。断言采用显式失败分支，在 Release 的 `NDEBUG` 下仍有效。
- ASan+UBSan / TSan 的独立 preset 配置成功，生成的编译标志分别正确；同时启用两个选项的配置被明确拒绝。
- VS Code 的六个任务定义已检查，构建/测试使用与上面已验证命令相同的调用方式；尚未通过 VS Code 界面点击运行。

## D3 验收记录

2026-10-10，通过 Windows PowerShell 调用 WSL 实际执行：

- Debug / Release 构建无编译警告，各 **3/3 CTest 通过**。新增 `buffer_contract` 的 **8 组测试**覆盖空状态、初始化与访问、移动与源复用、自移动、作用域销毁、乘法边界和尺寸拒绝；Release 保留全部运行期检查。
- ASan+UBSan 构建成功，针对 `buffer_contract` 的测试 **1/1 通过**，启用泄漏检查与错误即停止；本次执行没有报告泄漏、重复释放、越界或未定义行为。该结论限于已执行的 Buffer / 尺寸测试。

复现安全检查，使用前面设置的 `$linuxProject`：

```powershell
wsl -d Ubuntu-24.04 --cd $linuxProject -- cmake --preset asan-ubsan
wsl -d Ubuntu-24.04 --cd $linuxProject -- cmake --build --preset asan-ubsan
wsl -d Ubuntu-24.04 --cd $linuxProject -- env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 ctest --preset asan-ubsan -R '^buffer_contract$' -V
```

## D4 验收记录

2026-10-10，通过 Windows PowerShell 调用 WSL 实际执行：

- Debug / Release 构建无编译警告，各 **5/5 CTest 通过**；新增 `tensor_view_contract` 和 `model_contract`，Release 同样保留运行期检查。
- TensorView 的 **7 组测试**覆盖非方阵行主序访问、可写借用与只读转换、容量、空指针 / 对齐 / 零维度、越界、元素数 / 字节数溢出及 `PTRDIFF_MAX` 边界。
- 模型的 **6 组测试**加载正式权重，逐元素核对独立生成的确定性权重，验证移动后的模型和只读接口。四份文件各覆盖缺失、短文件、空文件、长文件、目录替代、NaN、Inf，并在每次错误后恢复文件重新加载；另验证不存在 / 非目录路径及不可读文件。不可读用例在当前普通 Linux 用户下实际执行；能绕过文件权限的特权进程会明确报告跳过该用例。
- ASan+UBSan 下 Buffer / TensorView / 模型加载 **3/3 CTest 通过**，启用泄漏检查与错误即停止，执行未报告泄漏、越界或未定义行为。线程安全检查和完整前向推理仍属于后续任务。

复现 D3 / D4 安全检查，使用前面设置的 `$linuxProject`；`--exec` 直接传递正则表达式给 CTest：

```powershell
wsl -d Ubuntu-24.04 --cd $linuxProject -- cmake --preset asan-ubsan
wsl -d Ubuntu-24.04 --cd $linuxProject -- cmake --build --preset asan-ubsan
wsl -d Ubuntu-24.04 --cd $linuxProject --exec env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 ctest --preset asan-ubsan -R '^(buffer_contract|tensor_view_contract|model_contract)$' -V
```

## D5–D10 验收记录

2026-10-10，Windows 主源码、WSL GCC 13.3.0 实际执行：

| 工作单元 | 实现与本轮证据 |
|---|---|
| D5 | CpuOps 7 组测试：非方阵、多 Batch、手算、bias、ReLU 零、Argmax 并列、非法形状 / 指针 / 容量 / 重叠 / 非有限值、算术溢出 |
| D6 | 同步执行器与 CLI；Batch 1/8/32/64/256 共比较 512 行、5120 logits，最大绝对误差 `7.62939453e-06`，最大非零参考相对误差 `1.27173634e-05`，类别全一致；256 行是循环复用参考，不是新增独立样本 |
| D7 | 输入 / 文件 / 尺寸 / 移动源 / Workspace 超限校验、结果独立寿命、错误恢复；全套 ASan+UBSan 实际通过 |
| D8 | Release `direct_cpu`，Batch 1/8/32 各 3 次独立进程，预热 200 / 正式 1000；每次单独 100 次分配探测得到 200 个 C++ new 事件；保存环境、SHA、指标和内存口径 |
| D9 | 有界 FIFO 环形队列；用 promise gate 确定性占住 Worker 后填满等待槽位，验证 QueueFull；关闭排空和停止接收通过 |
| D10 | 通用 ThreadPool + Runtime，Worker ID 稳定；双 Worker Batch 1/8 与直接执行器一致，Input / Result 所有权、异常 future 后恢复、重复关闭、析构回收均通过 |

- Debug / Release / ASan+UBSan 均构建成功，无编译警告，各 **10/10 CTest 通过**。ASan 启用泄漏检查，UBSan 错误即停止，本次没有报告泄漏、越界或未定义行为。该结果限于已执行契约测试，不替代 D11 / D12 压力与数据竞争检查。
- 同步参考验证通过后才进入并发；D10 已具备可用的异步执行路径。这是 D10 当时的验收快照；Runtime 统计、完整演示及之后的进度见 D11–D15。

复现全套安全检查：

```powershell
wsl -d Ubuntu-24.04 --cd $linuxProject --exec cmake --preset asan-ubsan
wsl -d Ubuntu-24.04 --cd $linuxProject --exec cmake --build --preset asan-ubsan
wsl -d Ubuntu-24.04 --cd $linuxProject --exec env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 ctest --preset asan-ubsan
```

V1 / V2 升级边界保持 [实施策略](docs/MiniInfer_V0.1_ISO_策略.md) 中的约定，后续阶段按用户选择推进。

## D11–D15 最终验收记录

2026-10-10，继续使用同一 Windows 主源码 / WSL 构建；没有重新训练或覆盖正式模型。

| 工作单元 | 完成内容与证据 |
|---|---|
| D11 | 四生产者 × 每人 40 个已接受请求，1/2/4 Worker、混合 Batch 1/8/32/64/256；逐元素精确对照 CpuExecutor，重试 / QueueFull 单独记账；每生产者 future 窗口 2。每配置含附加用例共接受 167，成功 166、预期任务失败 1，结果隔离和异常后恢复通过；[详细输出](docs/validation/d11_d12_details.log) |
| D12 | gate 控制满队列 / Worker 状态，验证关闭优先级、并发 / 重复关闭、drain、恰好一次、Runtime 提交竞争和析构后收取 future；工厂第 0 / 2 次启动注入失败及非 joinable 返回。独立 TSan 的 Runtime / Pool / 压力 / 关闭 / 计时五项通过 |
| D13 | 独立直接执行器探测每请求 2 次 new；perf 实查未安装，保留现有实现，未做可选热点优化；[报告](docs/benchmark.md)记录覆盖范围 |
| D14 | direct_cpu / Runtime 1/2/4 Worker，Batch 1/8/32 各 3 次独立进程，窗口 8 / 队列 64 / 预热 200 / 尝试 1000，共 36 次；阶段计时、正确记账与 RSS 口径完整；[原始数据](docs/benchmark_data/d14_runtime.json) |
| D15 | README、[架构](docs/architecture.md)、[基准](docs/benchmark.md)、自验证 Runtime 演示；不存在的 Linux 构建目录完成 Release 配置 / 构建 / 15 项测试 |

最终 Debug / Release / ASan+UBSan 各 **15/15**，构建无编译警告，ASan 泄漏检查开启且 UBSan 错误即停止，无报告泄漏、越界或 UB。日志：[Debug](docs/validation/d15_debug.log)、[Release](docs/validation/d15_release.log)、[ASan/UBSan](docs/validation/d15_asan_ubsan.log)、[干净构建](docs/validation/d15_clean_release.log)。检查结果仅覆盖已执行路径。

TSan 首次默认布局出现 unexpected memory mapping，部分启动失败；[首次日志](docs/validation/d12_tsan.log)保留。使用进程级 `setarch x86_64 -R` 后五项通过，无竞争报告，见 [复测日志](docs/validation/d12_tsan_no_aslr.log)。随后增加的调用线程启动中断 / 异常退栈用例也在 TSan 下 [通过](docs/validation/d12_tsan_unwind.log)，确保先关闭池再 join 无限重试的生产者。该命令只影响此进程及其子进程，不修改系统配置。Windows PowerShell 复现：

```powershell
wsl -d Ubuntu-24.04 --cd $linuxProject --exec cmake --preset tsan
wsl -d Ubuntu-24.04 --cd $linuxProject --exec cmake --build --preset tsan
wsl -d Ubuntu-24.04 --cd $linuxProject --exec setarch x86_64 -R env TSAN_OPTIONS=halt_on_error=1 ctest --preset tsan -R '^(thread_pool_contract|runtime_contract|concurrency_contract|shutdown_contract|timing_contract)$' -V
```

运行完整 Runtime 演示与基准（Release 构建完成后）：

```powershell
wsl -d Ubuntu-24.04 --cd $linuxProject --exec /home/qyc/builds/MiniInfer/release/bin/runtime_demo data/reference
wsl -d Ubuntu-24.04 --cd $linuxProject --exec /usr/bin/python3 tools/run_runtime_baseline.py --cli /home/qyc/builds/MiniInfer/release/bin/benchmark_cli --reference data/reference --output docs/benchmark_data/d14_runtime_rerun.json --iterations 1000 --warmup 200 --repeats 3 --window 8 --queue-capacity 64
```

演示接受五个任务，四个成功、一个预期溢出异常，成功批大小 1/8/32/1；验证异常后恢复、关闭后拒绝以及 Runtime 析构后的结果。实际 [输出](docs/validation/d15_demo.json)含标签；程序自身对照 FP32 参考，失败退出 1，参数错误退出 2。基准输出路径须未存在，复跑选择新路径；输入必须精确匹配 Batch。

在相近 Linux 环境从零复现，不需要现有本机构建缓存。准备 GCC / Clang、CMake≥3.28、Ninja 和 Python3 标准库，仓库已有正式权重，进入仓库根目录执行（构建路径选择不存在的目录）：

```bash
cmake -S . -B "$HOME/builds/MiniInfer/clean-release" -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build "$HOME/builds/MiniInfer/clean-release" -j 4
ctest --test-dir "$HOME/builds/MiniInfer/clean-release" --output-on-failure
"$HOME/builds/MiniInfer/clean-release/bin/runtime_demo" data/reference
python3 tools/run_runtime_baseline.py --cli "$HOME/builds/MiniInfer/clean-release/bin/benchmark_cli" --reference data/reference --output docs/benchmark_data/clean_runtime.json
```

本机最终干净构建目录为 `/home/qyc/builds/MiniInfer/d15-clean-final-release-20261010`。源码和模型仍在 Windows 主目录，智能体保持 Windows；以上 Linux 命令仅用于构建复现。可选 perf 优化、到达率扫描、GPU / 真实视觉模型属于后续工作，v0.1 不扩展这些范围。

## D16 TCP 使用方式

客户端与服务端先都在 WSL 中运行，默认只监听 `127.0.0.1:9000`。构建沿用上述 presets。在一个 PowerShell 终端启动服务，在另一个终端运行客户端；服务端输出 `Listening on ...` 后即可连接：

```powershell
$linuxProject = '/mnt/c/Users/ASUS/Documents/MiniInfer'
wsl -d Ubuntu-24.04 --cd $linuxProject -- /home/qyc/builds/MiniInfer/release/bin/infer_server --model data/reference
```

```powershell
$linuxProject = '/mnt/c/Users/ASUS/Documents/MiniInfer'
wsl -d Ubuntu-24.04 --cd $linuxProject -- /home/qyc/builds/MiniInfer/release/bin/infer_client --input data/reference/inputs.f32 --batch 64 --requests 10
```

输入必须恰好包含 `batch*64` 个已归一化 little-endian FP32，不再除以 16；Batch 限制 1–256。`--requests` 在一个连接上依次发送独立请求，不重试失败请求；JSON 包含最后一次成功响应的 logits/labels、正式成功/错误计数和 P50/P95/P99。`--warmup N` 默认 0，预热成功后才开始正式请求；预热失败单列并中止。`connect_ms` 不包含在请求延迟中。`attempts=successes+business_errors+protocol_errors+connection_errors+timeouts`，`requests_planned=attempts+skipped`；建连错误另计。无成功样本时百分位为 0，须同时读取 `latency_samples`。

Echo 演示仍是单连接顺序处理，使用相同帧协议，默认接受一个连接后退出；分别在两个终端执行：

```powershell
wsl -d Ubuntu-24.04 -- /home/qyc/builds/MiniInfer/release/bin/tcp_echo_server --port 9001
wsl -d Ubuntu-24.04 -- /home/qyc/builds/MiniInfer/release/bin/tcp_echo_client --port 9001 --message hello
```

| infer_server 配置 | 默认值与含义 |
|---|---|
| `--connection-threads` / `--waiting-connections` | 4 / 8，固定连接处理线程及额外等待连接上限 |
| `--queue-timeout-ms` | 1000，等待连接在出队时检查是否过期 |
| `--workers` / `--queue-capacity` | 1 / 64，CPU 推理 Worker 及等待计算上限 |
| `--read-timeout-ms` / `--write-timeout-ms` | 5000 / 5000，每次整帧接收（含空闲等待）/发送的绝对期限 |
| `--connections` | 0，无限接入直到退出信号；正数限制累计 accept 次数，含容量拒绝 |
| `--host` / `--port` | IPv4 字面地址 / 端口；服务端端口 0 由系统分配并通过 readiness 输出 |

`infer_client --timeout-ms` 默认 5000，分别限制建连、整帧发送和整帧接收。连接层满直接关闭新连接；只有完整合法推理请求进入已满计算队列才返回 `BUSY`。

| 协议 status | 连接行为 |
|---|---|
| 0 成功 | 可继续发送下一个递增 ID |
| 1 `BAD_REQUEST` | 完整可信请求参数错误，回包后关闭 |
| 2 `BUSY` | 可继续发送下一独立请求 |
| 3 `SHUTTING_DOWN` | 回包后关闭 |
| 4 `INFERENCE_FAILED` | 当前计算失败，连接可继续使用 |

损坏 magic/version/type、全局超长或半帧 EOF 直接关闭连接。客户端退出码：0 全部请求成功，2 CLI 参数错误，3 正式业务响应包含错误，1 加载/连接/协议/超时或预热失败；JSON `status` 为整体成功标志，具体业务状态读 `bad_request/busy/shutting_down/inference_failed`。服务端参数错误退出 2，运行故障退出 1。

对 `infer_server` 按 Ctrl+C 或发 SIGTERM：停止接入、回收等待连接、让接收退出，Runtime 排空已接受计算，限时回包并 join。对端仅关闭写方向仍可收到已完成请求的结果；断连不取消已接受计算。计算 drain 本身没有硬性耗时上限。所有权、停止竞争边界及计数细节见 [架构](docs/architecture.md)。

## D16 验收记录

2026-10-11 在同一 Windows 源码 / WSL 环境按 D16.1→D16.5 顺序实现验收，未修改既有 CPU 算子、模型、Buffer 或 Runtime 实现。

| 阶段 | 实际证据 |
|---|---|
| D16.1 | 原始阻塞 Echo、UniqueFd、真实回环/重复连接/fd 回收，阶段测试 [2/2](docs/validation/d16_1_echo.log)；后续 Echo 已升级为共享帧协议 |
| D16.2 | 固定字节、逐字节/拼帧、短收发、EINTR/EAGAIN、所有半帧 EOF、整帧截止和上限；网络测试 [4/4](docs/validation/d16_2_protocol_io.log) |
| D16.3 | Batch 1/8/32/64/256 对照参考；运行中删除临时权重仍正确，证明只加载一次；异常后恢复/半关闭，网络测试 [5/5](docs/validation/d16_3_inference.log) |
| D16.4 | 四客户端 40 个混合 Batch 请求；gate 在 workers=1/queue=1 稳定制造 BUSY 后恢复；active=1/waiting=1 拒绝与过期，网络测试 [6/6](docs/validation/d16_4_concurrency.log) |
| D16.5 | 空闲/半帧/等待连接退出，执行与排队任务恰好完成一次，4 路并发 shutdown、提交竞争、RST、小缓冲填满至 EAGAIN 的慢读写超时、12 次启停资源回收；完整日志见下方 |

全套 [Debug 24/24](docs/validation/d16_debug.log)、[Release 24/24](docs/validation/d16_release.log)、[ASan+UBSan 24/24](docs/validation/d16_asan_ubsan.log) 通过。TSan 使用进程级 `setarch x86_64 -R`，网络套件 [首次 8/9](docs/validation/d16_tsan.log)：功能路径无竞争报告，但线程回收基线在 TSan 首次启动后台线程前采样。测试改为先启动/join 空线程并确认该线程退出，再采基线，所有严格资源断言保留；[TSan 关闭用例复验通过](docs/validation/d16_tsan_shutdown_recheck.log)。同一测试修改在 [Debug](docs/validation/d16_debug_shutdown_recheck.log)、[Release](docs/validation/d16_release_shutdown_recheck.log)、[ASan+UBSan](docs/validation/d16_asan-ubsan_shutdown_recheck.log) 也各复验通过；其余代码未变，不重复整套检查。

复现完整验收（各命令成功后再执行下一条，其他构建类型先 configure/build 对应 preset）：

```powershell
wsl -d Ubuntu-24.04 --cd $linuxProject -- ctest --preset debug
wsl -d Ubuntu-24.04 --cd $linuxProject -- ctest --preset release
wsl -d Ubuntu-24.04 --cd $linuxProject -- env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 ctest --preset asan-ubsan
wsl -d Ubuntu-24.04 --cd $linuxProject -- setarch x86_64 -R env TSAN_OPTIONS=halt_on_error=1 ctest --preset tsan -R net_
```

Release 的进程内 Runtime 与 TCP 各完成 15,000 个正式请求，业务/协议/连接错误及超时均为 0；配置、时延口径及原始记录见 [D16 基准](docs/benchmark.md)。Windows→WSL、局域网、epoll、TLS、GPU 与 V1/V2 仍未实施。

</details>
