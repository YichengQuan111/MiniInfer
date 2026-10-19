# MiniInfer 技术栈与开发基线

本文环境、命令与完成范围为 2026-10-10 的 v0.1 实测基线及 2026-10-11 的 D16 历史快照；主体中的“当前”以该快照为范围。D1–D15 的 CPU 基础与 [D16 TCP 扩展](D16_TCP_网络编程学习扩展.md)已完成。第5节保留版本技术范围；现有V2 CUDA实现、实际工具链与验收状态见 [V2升级报告](depth_v2_upgrade.md)及 [项目README](../README.md)。

## 1. 技术栈总览

**v0.1：Linux / WSL2 + C++17 + GCC + CMake / Ninja / CTest + 标准库并发 + 自写标量 FP32 算子。Python 只用于离线模型与参考数据。**

| 层次 | 采用的技术 | 在项目中的职责 |
|---|---|---|
| 开发与运行平台 | Ubuntu 24.04 LTS、WSL2、x86_64、little-endian | C++ 构建、测试、运行和性能采集的目标环境 |
| C++ 语言与编译器 | C++17；GCC 主工具链，Clang 备用 | 实现 Runtime；编译警告、调试及独立安全检查 |
| 构建与测试 | CMake、Ninja、CTest | 管理库/CLI/测试目标和构建配置；统一执行已注册的测试 |
| 内存与数据结构 | RAII、移动语义、标准库容器、独占所有权 | Buffer、拥有输入/结果的数据容器、只读权重、Worker 私有 Workspace |
| 张量与计算 | 主机可访问 TensorView、连续 FP32、行主序；自写 Linear / ReLU / Argmax | 固定两层 MLP 的 CPU 前向推理 |
| 并发与任务 | 标准库 thread / mutex / condition_variable / promise / future | 有界排队、固定 Worker、任务完成与异常交付、drain / join |
| 离线参考环境 | Python 3、NumPy、scikit-learn，独立 `.venv` | digits 模型训练、FP32 权重导出、固定输入和参考 logits / labels |
| 调试与安全检查 | GDB、ASan / UBSan、独立 TSan | 定位故障，检查实际执行路径的内存、未定义行为和数据竞争 |
| 性能与统计 | steady_clock、Release 可执行程序、Linux RSS；可选 perf | 延迟、吞吐、排队/执行时间、内存及热点分析 |
| 版本与文档 | Git、Markdown、机器可读基准输出 | 保存代码与小型参考工件，追溯契约、环境和实验结果 |

C++ 运行部分保持零第三方强依赖；Python 包不进入 C++ 链接或运行依赖。测试先采用标准库实现，不增加测试框架。CPU 阶段不调用 BLAS、Eigen 或成熟推理框架替代自写算子。

## 2. 当前实测环境与依赖版本

以下是本机已安装版本的快照，不是“最低支持版本”或“最新版本”声明。编译工具执行版本查询成功；列出的 Python 包实际导入成功。

| 工具 / 依赖 | 当前版本 | 用途与状态 |
|---|---|---|
| Ubuntu | 24.04.5 LTS | WSL2 发行版；架构 x86_64，字节序 little |
| WSL Linux 内核 | 6.18.40.1-microsoft-standard-WSL2 | 实际运行内核，性能报告需记录 |
| GCC / g++ | 13.3.0 | 默认编译器；D15 Debug / Release 实际构建通过 |
| Clang / clang++ | 18.1.3 | 可用的备用编译器；尚未构建项目 |
| CMake / CTest | 3.28.3 | 工程与测试已接入；D15 Debug / Release / ASan+UBSan 各 15/15 CTest 通过 |
| Ninja | 1.11.1 | 默认构建执行器 |
| Git for Windows | 2.54.0.windows.1 | Windows 主目录已初始化 main；统一从 Windows 管理主仓库 |
| WSL Git | 2.43.0 | 旧 Linux 目录有一个无提交仓库，保留为旧副本 |
| GDB | 15.1 | 版本查询通过；已有骨架程序，尚未实际调试 |
| Python | 3.12.3 | 项目 `.venv` 的解释器 |
| NumPy | 2.5.3 | 直接依赖；FP32 布局转换、参考计算与二进制导出 |
| scikit-learn | 1.9.1 | 直接依赖；digits 数据与 MLPClassifier |
| SciPy | 1.18.1 | 当前 Python 环境依赖，导入成功 |
| joblib | 1.6.0 | 当前 Python 环境依赖，导入成功 |
| threadpoolctl | 3.7.0 | 当前 Python 环境依赖，导入成功 |
| perf | 未找到 | 可选项，后续按 WSL 内核及权限条件验证，不阻塞首版 |

ASan / UBSan / TSan 的独立配置与互斥检查已验证。D15 全套 ASan+UBSan 在启用泄漏检查时 15/15 通过；独立 TSan 在进程级关闭地址随机化后五项相关测试通过，默认布局启动失败与成功复测均有日志。C++ 已对照正式参考 logits / labels，D14 的直接 CPU / Runtime 对照原始结果已保存。以上沿用既有验收记录，不代表新增网络路径已验证。Python 包基线已锁定在 `tools/requirements.txt`，实际环境记录保存在 `data/reference/manifest.json`；版本更新需重新生成并验证参考工件。

## 3. Windows 编辑、WSL 编译的目录约定

用户明确要求智能体保留在 Windows，Windows VS Code 阅读和编辑代码。采用一份源码、两种访问路径、独立 Linux 构建目录：

| 内容 | 固定位置 | 使用方式 |
|---|---|---|
| 唯一源码与 Git 仓库 | `C:\Users\ASUS\Documents\MiniInfer` | Windows Codex 修改文件，Windows VS Code 打开此目录，Windows Git 管理版本 |
| 同一源码的 WSL 路径 | `/mnt/c/Users/ASUS/Documents/MiniInfer` | WSL 编译器与 Python 直接读取；它不是第二份源码 |
| Linux 构建与可执行程序 | `/home/qyc/builds/MiniInfer` | CMake 缓存、目标文件、Linux 程序和安全检查构建放在 Linux 文件系统 |
| 已有 Python 环境 | `/home/qyc/projects/MiniInfer/.venv/bin/python` | 暂时复用已验证环境；脚本来自 Windows 主目录，产出写回主目录的 data |
| 旧 WSL 项目与仓库 | `/home/qyc/projects/MiniInfer` | 保留旧文档及空仓库，不再从这里编辑源码或构建，不持续同步两套工程 |

已确认 WSL 能读取当前 Windows 目录，且旧工程与主目录中的规划文档此前内容一致。Windows 主目录使用 Git main，提交与远端状态以 `git status`、`git log -1` 和 `git remote -v` 为准；旧目录不删除。后续数据导出、构建和测试必须指向 `/mnt/c/...` 主源码路径。

微软说明 Windows 盘通过 `/mnt/c` 在 WSL 中访问，并建议按主要工具所在环境选择文件位置。本方案根据用户的 Windows 编辑约束选择 Windows 源码，把构建输出放到 Linux 文件系统；源码读取仍存在跨文件系统成本，没有实测加速承诺。[微软文件系统说明](https://learn.microsoft.com/en-us/windows/wsl/filesystems)

后续目录按任务逐步建立，不提前创建空文件：

```text
C:\Users\ASUS\Documents\MiniInfer\
├── .git\
├── CMakeLists.txt          # D2 已创建
├── CMakePresets.json       # 独立 Linux 构建配置
├── .vscode\tasks.json      # Windows VS Code 调用 WSL
├── include\miniinfer\     # 公共接口
├── src\                   # C++ 实现
├── apps\                  # 推理与基准 CLI
├── tests\                 # 功能、错误和并发测试
├── tools\                 # 离线导出脚本
├── data\                  # 模型与固定参考工件
└── docs\                  # 规划、技术栈与实验报告
```

Git 统一使用 Windows 客户端。D1 已创建 `.gitignore` 与 `.gitattributes`：忽略虚拟环境、构建、缓存和临时重生成目录，文本统一 LF；`.f32` / `.i32` 二进制参考工件禁止行尾转换。Windows 和 WSL 不复用同一 CMake 构建目录。

| 构建配置 | 当前目录 | 使用目的 |
|---|---|---|
| Debug | `/home/qyc/builds/MiniInfer/debug` | 常规功能测试与 GDB 调试 |
| Release | `/home/qyc/builds/MiniInfer/release` | 正式性能测量，记录实际优化选项 |
| ASan + UBSan | `/home/qyc/builds/MiniInfer/asan-ubsan` | 内存与未定义行为检查，不参与正式计时 |
| TSan | `/home/qyc/builds/MiniInfer/tsan` | 独立并发检查，不与 ASan 组合 |

默认主线使用 GCC 与 Ninja，启用 C++17 和 `-Wall -Wextra -Wpedantic`；并发链接由 CMake 的 Threads 支持管理。Clang 用于出现工具链问题时的验证，首版不增加完整双编译器测试矩阵。已创建 CPU 静态库、推理 / 基准 CLI、十项 CTest 与四个独立构建配置；Debug / Release / ASan+UBSan 实际构建并测试，TSan 只完成配置。Release 同步基准采用实际 `-O3 -DNDEBUG`，计时与分配探测分开，口径见 [benchmark](benchmark.md)。

### 在 Windows PowerShell 中编译与测试

WSL 支持从 PowerShell 直接调用 Linux 工具，参数中的路径应使用 Linux 格式。这只调用编译命令，不改变 Windows 智能体的工作目录。[微软 WSL 命令互操作说明](https://learn.microsoft.com/en-us/windows/wsl/filesystems#run-linux-tools-from-a-windows-command-line)

以下是显式指定源码和构建目录的等效命令。D2 实际执行并验证的是 [项目 README](../README.md) 中的 preset 命令，其源码和构建位置与下列约定相同。按顺序运行，任何一步失败先处理错误：

```powershell
$linuxSource = '/mnt/c/Users/ASUS/Documents/MiniInfer'
$linuxBuild = '/home/qyc/builds/MiniInfer/debug'

wsl -d Ubuntu-24.04 -- cmake -S $linuxSource -B $linuxBuild -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
wsl -d Ubuntu-24.04 -- cmake --build $linuxBuild -j 4
wsl -d Ubuntu-24.04 -- ctest --test-dir $linuxBuild --output-on-failure --no-tests=error
```

CMake 的 `-S` 指定源码、`-B` 指定独立构建目录；CTest 的 `--no-tests=error` 避免没有测试时被当作成功。[CMake 3.28 命令说明](https://cmake.org/cmake/help/v3.28/manual/cmake.1.html)、[CTest 3.28 命令说明](https://cmake.org/cmake/help/v3.28/manual/ctest.1.html)

Release 改用独立的 release 目录与 `CMAKE_BUILD_TYPE=Release`。运行生成的程序也通过 `wsl -d Ubuntu-24.04 --` 调用其 Linux 路径；权重路径传 `/mnt/c/.../data/...`。CTest 的参考数据路径由工程显式传入源码的 data 目录，不依赖测试进程当前目录。计时前先完成数据加载和预热；若另测模型加载或图像 I/O，单独记录文件系统与数据位置。

Windows VS Code 日常直接打开主目录，使用 PowerShell 终端执行上述命令；D2 已配置 `.vscode/tasks.json`，默认构建任务和 Debug / Release 测试任务均通过 `wsl.exe` 调用 Linux 工具。Linux 可执行程序位于构建目录的 `bin/`，由 WSL 运行；调试使用 WSL GDB。若以后需要 Linux 头文件感知和断点调试，可以单独打开 VS Code 的 WSL 窗口访问同一源码；智能体仍使用 Windows 主目录。[VS Code WSL C++ 调试说明](https://code.visualstudio.com/docs/cpp/config-wsl)

## 4. 数值与模块实现约定

模型固定为 `64 → 128 → 10`。输入是归一化后的 `[B,64]` FP32，隐藏层 `[B,128]`，输出为 `[B,10]` logits 和每行一个类别。W1 / W2 使用 `[out,in]` 布局，四份权重文件仅包含 little-endian FP32。加载前验证平台的 float 表示、文件长度、读取结果和尺寸计算。

已实现默认配置：Worker=1、等待队列容量=64、最大 Batch=256。每 Worker 预留隐藏层 Workspace，ReLU 原地执行；Input / Result 分别拥有请求与 logits / labels。argmax 并列取最小索引；参考比较采用 `atol=1e-4, rtol=1e-4`，Batch 1/8/32/64/256 均通过，最大绝对误差 `7.62939453e-06`。

Buffer 已实现 CPU FP32 数组的固定容量与独占所有权：禁止复制、无异常移动、源对象变空、零长度不分配，按元素数和字节数分别查询；乘法溢出和超过 PTRDIFF_MAX 的字节数在分配前拒绝。TensorView 已实现对存活主机内存的借用，检查正维度、指针对齐、容量、溢出和访问边界；ConstTensorView 只暴露 const 指针和引用。MlpModel 已完整加载四份固定权重，只提供只读视图；偏置的 C++ 视图为行向量，文件字节布局不变。模型发布后允许并发只读访问，读取期间所有者不得移动或销毁；可写视图的数据同步由调用者负责。加载错误与 API 失败行为见项目 README。

CpuExecutor 已串联两层 Linear / ReLU / Argmax，Workspace 复用隐藏层；算子拒绝非有限输入、非法形状及有害内存重叠，计算溢出明确报错。ThreadPool 已实现固定环形等待队列和稳定 Worker ID，不包含模型形状与算子；Runtime 在线程启动前准备按 Worker ID 索引的私有资源。同步 run=submit().get()，与异步共用接收和关闭协议；直接 CpuExecutor 用于数值验证和计算基线。

输入错误、QueueFull、已关闭在调用时明确拒绝；已接受任务的执行异常通过对应 future 返回。关闭停止接收、执行完已接受任务、join 后释放资源。运行时完成的含义是拥有的结果可用，后续 GPU 路径仍须遵守这一约定。

## 5. D16 与 V1 / V2 技术栈如何增加

| 阶段 | 在对应阶段增加的技术 | 接入方式与需要验证的事项 |
|---|---|---|
| D16：原生 TCP 入门 | Linux socket API、TCP、fd RAII、长度帧、完整收发、poll 截止时间辅助、固定连接线程 | 网络层独立于 CPU 核心；模型只加载一次；连接、缓冲和计算队列分别有界；断连/超时/关闭有测试 |
| V1：深度模型参考与导出 | PyTorch、ONNX、Depth Anything V2 Small（ViT-S）；图像处理依赖按实际需求选定 | 固定 Batch=1 和模型形状；明确项目几何变换、相对深度语义与尺寸还原；自行验证导出，固定工件与参考样本 |
| V1：CPU 部署 | ONNX Runtime C++ API | 先建立同步 CPU 路径，与 Python 的预处理张量和原始输出对照 |
| V1：GPU 部署 | TensorRT C++、CUDA Runtime | 独立引擎与执行适配、设备资源 RAII；FP32 / FP16 原始深度张量、尺寸还原与任务保持对照 |
| V1：部署 SDK 与模型包 | C++ 接口 / 示例、模型与配置清单、引擎构建 / 缓存流程 | 兼容性检查、黄金回归、启动 / 首请求 / 稳态分别计时；源码与 ONNX 可移交，engine 按目标环境重建 |
| V1：深度网络服务 | 复用 D16 帧协议与连接线程，新增 RGB8 图像请求和 FP32 稠密相对深度响应 | 本地正确后再接入；单 GPU 槽位；请求 / 响应数量与字节预算；本地和 TCP 基线分别测量 |
| V2 P0：配置取舍 | 两种静态尺寸、FP32 / FP16 | 比较任务精度、延迟、内存和工件体积，选择配置；不依赖 INT8 |
| V2 P0：可靠异步与资源复用 | 拥有型任务 / future、有界 FIFO、独立上下文、CUDA Stream / Event、1 / 2 GPU 槽位、定时图像请求回放；每槽独立预分配输入输出、scratch、原图 RGB8 上传缓冲与所需 Pinned Memory | 原图缓冲按支持的输入上限与槽位数限制容量和字节；H2D → GPU 预处理 → TRT → D2H → CPU 后处理遵守依赖，结果就绪后交付，安全完成后复用；验证结果关联、关闭与槽位隔离 |
| V2.4 P0：完整 GPU 预处理 | 上传已解码 RGB8；GPU 完成除以 255、双线性 resize、padding、通道归一化及 HWC→NCHW，直接写入 TRT 输入缓冲；CUDA Event 与设备检查 | 必须实现、集成与验收 C 完整 GPU 路径，保留 A CPU / B 局部 GPU 对照；基础 / 改进 kernel、全模型正确性、端到端及 Profiler 对照齐全，默认部署按实测选择，不要求 C 更快 |
| V2 P0：性能与稳定性 | Nsight Systems、目标 kernel 的 Nsight Compute、分阶段计时与受控输入 | 解释系统 / kernel 瓶颈；测量 Batch=1 P99、成功吞吐、拒绝 / 失败、内存 / 显存、过载与长稳 |
| P1.Q：INT8 PTQ | 兼容 PTQ 工具、显式 ONNX Q/DQ、敏感层分析 | 校准 / 调参 / 最终测试隔离；真实 INT8 覆盖及同尺寸 FP16 对照；允许最终选 FP16，未做不阻塞 P0 |
| P1.F：实时帧 | 按需接入视频解码、最新帧淘汰、结果年龄与发布顺序 | 显式模式、丢弃 / 超期终态和时效统计；在途 GPU 资源安全完成后释放 |
| P1.N / K / R：按需进阶 | epoll LT / eventfd、独立 LayerNorm / GEMM、TensorRT Plugin、ROS2 薄封装或控制资源干扰实验 | 按岗位或 Profile 每次先选一个，不阻塞主线；仍需完整正确性与性能回归 |
| E1：真实端侧适配 | 设备到位后选择一个 ARM / Jetson / NPU 平台及对应 SDK | 目标架构构建、引擎重建、精度、持续延迟 / 内存、温度与功率口径；当前未验证 |

D16 已用系统 API 和标准库实现，不增加第三方网络库；上表是各版本的技术规划，实际接入状态见 README。V1 各项及标注 V2 P0 的内容为必做，P1 / E1 独立验收。主题为“MiniInfer：C++ 视觉模型部署与推理性能优化”，当前仅有电脑与 NVIDIA GPU，主线在本机验证。依赖、驱动兼容性和 API 在实施时核验，PTQ 工具只在选做量化时接入。网络为独立 CMake 目标，CPU 核心和 MLP 回归可独立构建；epoll 不属于 D16，也不是 V2 必达项。

V2.4 的 CPU 保留输入校验、入口需要时的解码、几何元数据计算、调度与结果后处理；解码不属于这项 CUDA 实作。A 在 CPU 完成全部像素处理并上传 FP32 NCHW；B 在 CPU 完成除以 255 / resize / padding，上传 FP32 HWC 后由 GPU 完成归一化 / 布局；C 上传原图 RGB8 并在 GPU 完成全部像素处理。三路径沿用相同数值 / 几何契约，现有混合 FP16 engine 仍使用 FP32 I/O。B / C已实现，CUDA12.4、TensorRT10.8、Nsight和Compute Sanitizer的本机检查及限制见 [V2升级报告](depth_v2_upgrade.md)；完整无收益实验也可按原门槛验收。详细边界、缓冲预算与验收以 [V1 / V2 计划](MiniInfer_V1_V2_求职改进策略.md)2026-10-15 修订的 S3.3 / S3.4 / O V2.4 为准。

固定 MLP 的输入和 `.f32` 工件继续保留。真实视觉模型使用新契约；主机 TensorView 不承载可直接解引用的设备地址。共用执行接口在第二种实际执行路径出现时提取，避免提前设计通用框架。

## 6. v0.1 / D16 历史完成范围

当前交付已覆盖 D1–D15 的 v0.1 必达范围：Windows 编辑 / WSL 编译与 Git 主目录、真实 CPU MLP、有界队列 / 通用 ThreadPool / Runtime、多生产者压力、关闭竞争 / 创建失败、阶段计时、Release 对照、演示与干净构建。源码、工件、测试及文档均维护在 Windows 主目录；旧 Linux 副本不再作为活动源码或文档维护。

D1 的四项导出测试及同环境逐字节复现保留原验收证据；本轮不重训。最终 Debug / Release / ASan+UBSan 各 15/15 CTest 通过，独立 TSan 用进程级 setarch 关闭地址随机化后五项相关测试通过；默认布局启动失败及成功复测日志均保存。D14 保存 36 次独立进程的直接 CPU / Runtime 1/2/4 Worker 结果，RSS 与高水位分别解释。Python3 可选注册驱动测试；无 Python 时 CTest 为 14 项。

当前没有安装 perf，也没有做可选热点优化；D16 原生 TCP 已实现。以上为历史验收记录，V1 / V2 的实际接入与完成状态以 README 为准。v0.1 API / 所有权见 [architecture](architecture.md)，验收及复现命令见 [README](../README.md)，实际测量与限制见 [benchmark](benchmark.md)。D10 交接及 I/S 文档中的历史快照不能替代当前证据。
