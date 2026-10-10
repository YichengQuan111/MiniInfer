# MiniInfer 技术栈与开发基线

日期：2026-10-10。I-S-O 阶段：O，已完成 D1–D10：固定 CPU MLP、同步 / 异步执行、有界线程池和同步基准。本文落实 [S 阶段策略](MiniInfer_V0.1_ISO_策略.md)，作为编码依据；多生产者压力、TSan 与完整性能对照尚待后续。当前实现与验收记录见 [项目 README](../README.md)。

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
| GCC / g++ | 13.3.0 | 默认编译器；D10 Debug / Release 实际构建通过 |
| Clang / clang++ | 18.1.3 | 可用的备用编译器；尚未构建项目 |
| CMake / CTest | 3.28.3 | 工程与测试已接入；D10 Debug / Release / ASan+UBSan 各 10/10 CTest 通过 |
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

ASan / UBSan / TSan 的独立配置与生成编译标志已验证，互斥选项的错误路径通过。本轮全套算子、同步 / 异步推理及基础队列测试已在 ASan+UBSan 下启用泄漏检查运行，10/10 通过；TSan 尚未构建或运行。C++ 已对照正式参考 logits / labels，Release 同步基准原始结果已保存。Python 包基线已锁定在 `tools/requirements.txt`，实际环境记录保存在 `data/reference/manifest.json`；版本更新需重新生成并验证参考工件。

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

## 5. V1 / V2 技术栈如何增加

| 阶段 | 在对应阶段增加的技术 | 接入方式与需要验证的事项 |
|---|---|---|
| V1：真实模型参考与导出 | PyTorch、ONNX、一个固定轻量视觉模型；图像处理依赖按实际需求选定 | 增加模型专属输入/输出、颜色与布局、预处理和后处理契约；固定工件与参考样本 |
| V1：CPU 部署 | ONNX Runtime C++ API | 先建立同步 CPU 路径，与 Python 的预处理张量和原始输出对照 |
| V1：GPU 部署 | TensorRT C++、CUDA Runtime | 独立引擎与执行适配、设备资源 RAII；FP32 / FP16 同步结果与任务指标对照 |
| V2：资源与并发 | 独立执行上下文、设备缓冲、CUDA Stream / Event、GPU 并发槽位 | 每条同时执行的路径隔离可写资源，CPU Worker 数和 GPU 并发数分开配置；future 和关闭等待实际完成 |
| V2：性能与 CUDA 实践 | Nsight Systems、CUDA 算子、设备计时 | 根据 Profile 完成一次优化实验；分别测量排队、预处理、传输、计算、后处理和端到端耗时 |

这些新增依赖的版本、驱动兼容性、API 行为与可运行性在对应阶段核验并记录。本轮不安装 GPU / 部署依赖，也不为未实现后端写接口代码。将来通过独立、可选的 CMake 目标接入，CPU 核心和 MLP 回归仍应能独立构建。

固定 MLP 的输入和 `.f32` 工件继续保留。真实视觉模型使用新契约；主机 TensorView 不承载可直接解引用的设备地址。共用执行接口在第二种实际执行路径出现时提取，避免提前设计通用框架。

## 6. 本轮完成范围与下一项工作

当前交付：Windows 编辑 / WSL 编译与 Git 主目录，D1–D4 数据与基础资源，D5–D7 算子 / 前向 / 输入校验与安全检查，D8 同步基准与分配记录，D9–D10 有界队列 / 通用 ThreadPool / Runtime。源码、工件、测试及文档均维护在 Windows 主目录；旧 Linux 副本不再作为活动源码或文档维护。

D1 的 4 项 Python 契约测试及同环境逐字节复现已验收。D10 主线 Debug / Release / ASan+UBSan 各 10/10 CTest 通过，正式推理类别一致，基本异步结果 / 所有权 / 错误交付 / drain 成立。D8 保存 Batch 1/8/32 各三次独立进程的 Release 原始结果和环境，分配探测每请求两次 new；RSS 与进程高水位口径分别记录。并发竞争安全和完整 Runtime 性能对照仍待后续工作单元；运行命令与证据见项目 README 和 benchmark。

下一项编码任务是 D11：多生产者与 1/2/4 Worker 正确性 / 压力测试；D12 补并发关闭、构造失败注入和 TSan。随后完成 Runtime 基准与最终交付，V1 / V2 依赖仍在对应阶段核验。
