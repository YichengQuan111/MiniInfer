# MiniInfer 开发交接：D10 完成后继续 D11–D15

更新日期：2026-10-10。本文件是新会话的启动快照，记录当前实现和已运行证据；不代表 v0.1 最终验收完成。新会话先核对工作区是否发生变化，再按用户指定范围继续。本次仅整理交接，没有改代码或重新运行测试。

## 1. 目标、范围与阅读顺序

v0.1 使用 C++17 / Linux 从零实现 CPU 神经网络推理 Runtime：固定 `64→128→10` FP32 MLP、资源所有权、标量算子、有界队列 / Worker、错误交付、安全关闭、正确性与可复现测量。D1–D10 已完成，D6 同步正确性关卡与 D10 可用异步路径关卡均已通过。

按以下顺序读取必要内容，不要重复调研或重建已完成模块：

1. 本交接文件。
2. [项目开发规范](<../MiniInfer 项目说明与 AI Agent 开发规范.md>)：资源、线程、错误、测试、最终完成标准。
3. [实施策略](MiniInfer_V0.1_ISO_策略.md)：后续执行依据，重点第 4–7 节。策略已将基本 drain / 异常交付提前到 D10，D12 是补全竞争验证，不能把安全关闭当作尚不存在。
4. [README](../README.md)：现有 API 契约与 D1–D10 验收记录；[技术栈](tech_stack.md)：Windows / WSL 路径与版本；[数据契约](../data/README.md)。
5. [基准报告](benchmark.md)及相关源码 / 测试；V1 / V2 背景按需读 [升级策略](MiniInfer_V1_V2_求职改进策略.md)，不要把未来范围加入 v0.1。

旧 I / S / O 规划文档可能保留制定时的“下一任务”文字；实际进度以源码和 README 的已执行记录为准。用户最新指令优先，若本快照与实际文件不一致，说明差异后按实际证据处理。

## 2. 必须保持的工作方式

| 项目 | 位置 / 约定 |
|---|---|
| 唯一源码与 Git 主目录 | `C:\Users\ASUS\Documents\MiniInfer` |
| Codex 智能体 | 保持在 Windows，使用 PowerShell；用户明确表示切换智能体到 WSL 会出现 bug |
| VS Code | Windows VS Code 直接打开上述主目录 |
| 同一源码的 WSL 访问路径 | `/mnt/c/Users/ASUS/Documents/MiniInfer`，不是另一个副本 |
| Linux 工具执行 | 从 Windows 通过 `wsl -d Ubuntu-24.04 --cd ... --exec ...` 调用 |
| 构建目录 | `/home/qyc/builds/MiniInfer/{debug,release,asan-ubsan,tsan}`，可执行文件在各自 `bin/` |
| Python 导出 / 契约测试环境 | `/home/qyc/projects/MiniInfer/.venv/bin/python`，只复用解释器与包 |
| 同步基准驱动 | `/usr/bin/python3`，仅需 Python 标准库 |
| 旧 WSL 工程 | `/home/qyc/projects/MiniInfer` 保留，不从这里编辑 / 构建，不同步两份源码 |

Git 主分支为 `main`，GitHub 发布目标为公开仓库 [YichengQuan111/MiniInfer](https://github.com/YichengQuan111/MiniInfer)。本交接文档最初生成时仓库尚无提交；用户随后授权创建首次提交并上传。新会话先用 `git status`、`git log -1` 和 `git remote -v` 确认最新状态，不再将“所有文件未跟踪”视为当前状态。保留现有文件与提交，不重做 D1–D10。

使用项目现有 CMake Presets，不在 Windows 源码目录生成 Linux 构建物，也不改用 Windows 原生编译器。现有 `.vscode/tasks.json` 包含六个 Debug / Release 配置、构建、测试任务，默认 Ctrl+Shift+B 构建 Debug；调用方式已检查，尚未通过 VS Code 界面点击验收。

Windows 调用 WSL 偶尔打印 localhost 代理警告，工具输出可能呈 UTF-16 乱码；此前编译 / 测试仍成功。以退出码及实际诊断判断。带 `|`、括号等的 CTest 正则用 `--exec` 直接传给工具；曾用默认 shell 传递而出现转义错误，不能把它当作测试本身失败。

## 3. 协作偏好

- 默认用中文，先交代目标和最小步骤，再持续推进授权范围；需要较长工作时简短报告发现和验证进展。
- 用户已使用 Luna Max 并行开发。复杂任务仅在预计减少主智能体工作量时委派；简单任务直接做。
- 子智能体明确设置 `model=gpt-6-luna`、`reasoning_effort=max`、`fork_turns=none`；同时最多五个，子智能体不得再委派。新会话须实际检查工具是否支持配置，不虚称已经生效。
- 每项委派只传目标、材料位置、文件边界、冻结 API 和验收标准；同一工作区按文件划分，主智能体负责跨模块判断、关键证据核查和最终验收。
- 搜索先 `rg` / `rg --files`，先定位再读取；独立查询批量执行，大文件 / JSON 在工具侧按字段筛选。避免重复整份读取和重复运行已通过且未受改动影响的检查。
- 完整完成、质量达标优先。这些偏好不是硬性 token / 额度预算，不宣称未经测量的节省比例。
- 不因为例行可逆编辑反复请求确认；缺失信息确实影响后续时才询问。按任务完成后交付简短结论与文件链接，详细记录留在文档。

## 4. 当前代码地图与不可破坏的契约

| 模块 / 文件 | 当前实现与边界 |
|---|---|
| `buffer.hpp` / `checked_size.hpp` | 连续 CPU FP32 独占存储；禁止复制，无异常移动，移动源为空；乘法溢出 / 超过 PTRDIFF_MAX 字节在分配前拒绝 |
| `tensor_view.hpp` | 非拥有二维行主序 TensorView / ConstTensorView；正维度、非空 / 对齐指针、容量 / 溢出 / 索引检查；可用容量由调用者如实提供，只支持主机可解引用地址 |
| `mlp_model.hpp` / `data_io.hpp` | MlpModel 拥有四份固定权重，只读视图；little-endian IEEE FP32，精确文件长度、读取、有限值校验；JSON manifest 用于追溯，不引入 C++ JSON 依赖 |
| `cpu_ops.hpp` | `linear(input,weights,bias,output)`、原地 `relu(view)`、`argmax(logits,labels,capacity)`；检查形状 / 非有限值 / 有害重叠，Linear 结果非有限抛 overflow_error |
| `mlp_input.hpp` | MlpInput(batch,vector<float>) 拥有 `[B,64]` 归一化输入；复制深拷贝、移动转移；零维度、错误元素数、NaN / Inf 拒绝 |
| `mlp_result.hpp` | 独立拥有 logits `[B,10]` 与 int32 labels `[B]`；禁止复制、支持移动；Result 不借用 Workspace，后续执行不覆盖旧结果 |
| `workspace.hpp` | 按 max_batch 预分配 `max_batch*128` float 隐藏层；只借用当前 Batch 前缀；独占写入，禁止复制 / 赋值，移动源容量归零 |
| `cpu_executor.hpp` | 无状态 `run(const MlpModel&,const MlpInput&,Workspace&) const -> MlpResult`，不管理线程 / 队列 / 计时 |
| `bounded_queue.hpp` | 固定 vector<optional<T>> 环形槽位；T 要求无异常移动 / 析构；push / pop / close，mutex + cv；关闭后排空再返回空 |
| `thread_pool.hpp` | 模型无关的 ThreadPool；submit(callable(worker_id)) 返回 future，支持 move-only capture；稳定 Worker ID、QueueFull / PoolClosed、序列化 shutdown、构造部分失败回收 |
| `runtime.hpp` | Runtime 拥有 model、按 Worker ID 索引的独立 Workspace、pool；默认 workers=1 / queue_capacity=64 / max_batch=256 |
| `apps/infer_cli.cpp` | 真实同步推理，`--model DIR --input FILE --batch N`；N 为 1–256，成功单行 JSON；参数错误 exit2，加载 / 执行错误 exit1 |
| `apps/benchmark_cli.cpp` | 当前仅 `direct_cpu`；正式计时与独立 global C++ new 事件探测、RSS / 高水位采样；尚无 Runtime 基准模式 |
| `tools/export_mlp.py` | 已训练 / 导出并验证正式数据，不需要为继续 Runtime 开发重训 |

重要约定：

- 固定 MLP `64→128→10`；输入 `pixel/16.0` 已在工件中归一化。W1 `[128,64]`、W2 `[10,128]` 使用 `weights[out,in]`。偏置文件 `[128]` / `[10]`，C++ 视图 `[1,128]` / `[1,10]`，字节布局一致。
- Linear 从零按输入维度顺序 FP32 累加，最后加 bias；ReLU 将负数与正 / 负零转为正零；Argmax 并列取最小索引。不随意改布局、预处理或错误语义。
- TensorView 不延长所有者寿命，临时模型 / Input / Result 的借用 getter 被禁用；停止使用移动 / 赋值 / 销毁导致失效的借用。异步任务必须拥有 Input，不能捕获调用者的裸视图。
- `Runtime::submit(MlpInput input)`：左值复制、右值移动，调用线程先校验输入，再入队；`run(input)=submit(std::move(input)).get()`，同步 API 也有排队与拒绝成本。
- 队列容量只计等待任务；执行中的任务已出队。接收判断与入队在同一锁下完成，满时立即 QueueFull，不等待空位；队列已关闭时 PoolClosed 优先于队列满。任务执行错误通过 future 交付，Worker 后续仍可运行。
- Runtime 成员顺序为 model / config / workspaces / pool：pool 最先析构，drain / join 后才释放资源。基本关闭已实现，多个 shutdown 调用由单独 mutex 串行化，join 在队列锁外；Running / Draining / Stopped 由接收标志及 join 完成隐式表示。
- 外部可以并发调用 API，调用者负责 Runtime / pool 对象寿命；同池 Worker 不阻塞等待同池任务、不销毁池，不自 shutdown（抛 logic_error）。
- ThreadPool 的线程启动失败 catch 分支已实现，但**尚未做实际失败注入测试**；不能把“有代码”写成“已验证”。

## 5. 已验证结果与局限

证据集中于 [README 的 D5–D10 记录](../README.md#d5d10-验收记录)、测试源码和 [benchmark](benchmark.md)，无需重做整套历史工作。

| 项目 | 已运行结果 |
|---|---|
| D1 Python | 4 项 unittest 通过；默认重新导出 10 个文件在同环境逐字节一致 |
| 正式模型 | 留出集 354/360=98.33%，固定参考子集 62/64=96.88%；是模型分类效果，不能等同工程验收或速度 |
| C++ Debug / Release / ASan+UBSan | 三个配置均成功构建，无编译警告，各 10/10 CTest 通过；ASan detect_leaks=1，UBSan 错误即停止，无报告泄漏 / 越界 / UB |
| 同步参考 | Batch 1/8/32/64/256，共 512 行 / 5120 logits；atol=rtol=1e-4，最大绝对误差 7.62939453e-06，最大非零参考相对误差 1.27173634e-05，类别全一致 |
| 基本异步 | Runtime 两 Worker、Batch 1/8 与 CpuExecutor 一致；输入副本不受调用方修改影响、旧 Result 不被覆盖、错误 future 后恢复、重复关闭及析构后收结果通过 |
| 队列 / Pool | 6 组基础测试，包含受控饱和 / 拒绝、FIFO / 关闭排空、稳定 Worker ID、move-only、异常交付和自 shutdown 拒绝 |
| D8 基准 | Batch 1/8/32 各 3 次独立进程，预热200 / 正式1000；分别另探测100次，均200个 C++ new 事件，即每请求2次，用于独立 Result 存储 |

十项 CTest 名称：`project_contract`、`cli_contract`、`buffer_contract`、`tensor_view_contract`、`model_contract`、`cpu_ops_contract`、`inference_contract`、`runtime_contract`、`thread_pool_contract`、`benchmark_cli_contract`。当前统一 TIMEOUT=20 秒，内部 CLI 调用另有超时。

不可扩大解读：

- 256 行验证是循环复用 64 行正式参考，不是新增独立评测样本。
- ASan / UBSan 不证明没有数据竞争；TSan 仅配置过，尚未构建或运行。1/2/4 Worker 多生产者压力、并发 submit / shutdown、构造失败注入仍待做。
- 不可读权重测试当前普通 Linux 用户下实际执行；若进程能绕过文件权限，会显式打印 skip，不冒充该分支通过。
- D8 只测直接执行器，包含输入 / 算子校验、结果分配 / 消费 / 销毁和计时开销，不能当作纯 kernel 耗时。new 探测不覆盖所有 malloc / mmap 或其他线程。
- 原始九次数据是 `docs/benchmark_data/d8_sync.json`；`d8_peak_probe.json` 为不同启动方式的短诊断。Python 启动的高水位 RSS 17.457–17.707 MiB，直接 WSL 短诊断约3.766 MiB；可能受启动阶段影响，不应把高水位解释成执行器独占峰值。预热后 / 正式计时后 RSS 分别3.660–3.715 / 3.723–3.785 MiB。
- Batch=1 吞吐三次为约101065–129816 req/s；未绑核，未控制供电 / 频率 / 后台负载。后续必须同条件比较、保存原始结果，不虚构稳定速度或加速比。

## 6. 剩余开发顺序与完成标准

| 任务 | 需要完成 | 验收 / 约束 |
|---|---|---|
| D11 | 多生产者与1/2/4 Worker 的正确性 / 压力；混合 Batch、结果隔离、任务计数 | 所有成功结果与同步基线一致，无串扰；每个已接受任务完成一次，成功 / 失败均能收回；QueueFull 不算成功，不无限积累 future / Result |
| D12 | 并发 submit / shutdown、重复 / 并发关闭、拒绝边界、任务失败恢复、线程构造失败回收；独立 TSan | 所有接受任务完成，关闭后拒绝，drain / join 无悬空资源；用可控创建失败注入，不靠耗尽系统线程；TSan 实际运行或记录准确失败原因 |
| D13 | 时间允许时 perf / 分配分析，最多选择一个小优化 | 先有热点证据；改动后跑相关正确性和同条件基准；此前 perf 未找到，后续再核实，无工具 / 无收益可以保留原基线，不阻塞 P0 |
| D14 | benchmark_cli 增加 Runtime 路径与必要计时，完整 Release 对照 / 内存记录 | direct_cpu 与 Runtime1/2/4 Worker 区分；相同输入序列 / Batch / 在途窗口 / 采集方式；记录尝试 / 接受 / 成功 / 失败 / 拒绝，排队 / 执行 / 回收延迟，RSS口径和重复数据 |
| D15 | README、architecture、benchmark、演示、干净构建复现 | 新环境可按文档跑测试与基准，全部必达项有证据；工具限制 / 未完成 P1 明确。当前尚没有最终 architecture 文档或完整 Runtime 演示 |

下一步建议只从 **D11** 开始，完成后再进入 D12；若用户明确授权连续完成多项，按依赖逐项验收后继续。D12 安全验证优先于复杂统计 / 优化，最终交付不要求一定有加速。

具体注意：

- 并发测试用 promise / condition_variable 明确控制 Worker 已占用、队列已满等状态，**不用 sleep 猜测**。复用 `tests/test_thread_pool.cpp` 的 gate 模式。
- Gate / worker 捕获的 promise 必须比 pool 活得长，释放守卫在 pool 后声明：异常退栈先释放 gate，再析构 pool join，最后销毁 gate / promise。此前已修正该测试生命周期，不能退回先销毁同步对象再 join 的写法。
- Runtime 目前没有每请求入队 / 开始执行 / 执行完成 / 调用者取得结果时间戳。D14 需要明确加入测量边界，不将提交耗时冒称排队耗时；不把 MLP 数学写入通用队列 / Pool。
- D14 主负载起点：队列64、固定有限在途窗口、Batch1主 / Batch8与32对照、预热200 / 正式至少1000 / 重复至少3。成功吞吐=完成成功数/正式墙钟，样本数另计；结束时drain并核验接受数=成功数+失败数。
- 不提前引入 CUDA / ONNX Runtime / TensorRT。V1 增加真实模型同步部署；V2 增加独立设备资源、Stream / Event / 并发槽位；主机 TensorView 不承载设备地址，多 CPU Worker 不证明 GPU 后端共享安全。

## 7. 续开发常用命令

在 Windows PowerShell 逐条执行、检查退出码后再继续。现有工具基线为 GCC13.3.0 / CMake3.28.3 / Ninja1.11.1 / Ubuntu24.04.5，Clang18.1.3可用但尚未构建项目。

```powershell
Set-Location -LiteralPath 'C:\Users\ASUS\Documents\MiniInfer'
$linuxProject = '/mnt/c/Users/ASUS/Documents/MiniInfer'

git status --short
wsl -d Ubuntu-24.04 --cd $linuxProject --exec cmake --preset debug
wsl -d Ubuntu-24.04 --cd $linuxProject --exec cmake --build --preset debug
wsl -d Ubuntu-24.04 --cd $linuxProject --exec ctest --preset debug
```

Release 将三个 preset 名称改为 `release`。新增 D11 测试后按实际改动跑检查，不因交接重新训练或重复测全部历史基准。

```powershell
# 针对当前队列 / Runtime 的定位测试
wsl -d Ubuntu-24.04 --cd $linuxProject --exec ctest --preset debug -R '^(thread_pool_contract|runtime_contract)$' -V

# 独立 ASan+UBSan 配置
wsl -d Ubuntu-24.04 --cd $linuxProject --exec cmake --preset asan-ubsan
wsl -d Ubuntu-24.04 --cd $linuxProject --exec cmake --build --preset asan-ubsan
wsl -d Ubuntu-24.04 --cd $linuxProject --exec env ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 ctest --preset asan-ubsan

# D12 的 TSan 尝试：这些命令列为下一步，不是已通过证据
wsl -d Ubuntu-24.04 --cd $linuxProject --exec cmake --preset tsan
wsl -d Ubuntu-24.04 --cd $linuxProject --exec cmake --build --preset tsan
wsl -d Ubuntu-24.04 --cd $linuxProject --exec env TSAN_OPTIONS=halt_on_error=1 ctest --preset tsan -R '^(thread_pool_contract|runtime_contract)$' -V
```

ASan+UBSan 与 TSan 互斥，不能启用在同一构建目录。定位并发问题用单独配置，不能把 TSan 启动 / 平台问题等同代码通过，也不能因为预期 WSL 有限制而跳过实际尝试。

```powershell
# 实际推理：64行正式参考
wsl -d Ubuntu-24.04 --cd $linuxProject --exec /home/qyc/builds/MiniInfer/release/bin/infer_cli --model data/reference --input data/reference/inputs.f32 --batch 64

# 如新改动需要基准对照，选未存在的输出路径
wsl -d Ubuntu-24.04 --cd $linuxProject --exec /usr/bin/python3 tools/run_sync_baseline.py --cli /home/qyc/builds/MiniInfer/release/bin/benchmark_cli --reference data/reference --output docs/benchmark_data/d8_sync_newrun.json --iterations 1000 --warmup 200 --repeats 3

# 仅当修改导出 / 数据契约时运行 Python 契约测试
wsl -d Ubuntu-24.04 --cd $linuxProject --exec /home/qyc/projects/MiniInfer/.venv/bin/python -m unittest discover -s tests -p test_export_mlp.py -v
```

基准输入精确匹配 Batch；驱动会从64行正式输入截取前B行到临时文件，结束后清理。不要拿整份 `inputs.f32` 直接请求 Batch1。保留已有正式数据和测量文件；重生成 / 复测选择新目录或输出文件。

## 8. 直接粘贴到新会话的启动提示

```text
请继续 MiniInfer v0.1 的剩余开发。项目目录是 C:\Users\ASUS\Documents\MiniInfer。
先读取 docs/HANDOFF_D10.md，再按其中的阅读顺序核对开发规范、实施策略、README 和当前代码。

当前 D1–D10 已完成；本次从 D11 开始，只完成 D11 的多生产者、1/2/4 Worker 正确性与压力测试，实际验收后向我报告，暂不提前实现 D12–D15。
Codex 智能体和 VS Code 保持 Windows；只通过 Windows PowerShell 调用 WSL Ubuntu-24.04 编译 / 运行同一份 /mnt/c/... 源码，构建目录保持 /home/qyc/builds/MiniInfer。
仓库使用 main，GitHub 发布目标为公开仓库 YichengQuan111/MiniInfer；请先检查 git status、git log -1 和 git remote -v，保留既有文件与提交，不重做 D1–D10，不重训正式模型。

采用交接文档中的省额度工作规则；适合独立推进的子任务允许调用 model=gpt-6-luna、reasoning_effort=max、fork_turns=none 的子智能体，同一时间最多5个，子智能体不得再委派，主智能体负责集成与最终验收。
先检查现状，给最小改动范围，然后开始实现；不要只给计划。并发测试必须使用确定性同步，不能靠 sleep 判断队列 / 线程状态。
保持 Input / Result 拥有数据、模型只读共享、Worker 私有 Workspace、有界拒绝、future 异常和 drain/join 契约。
完成后简短报告改动、实际测试结果、未解决问题和下一步，并更新 README 与必要文档。不要虚构 TSan / 压力 / 性能结果。
```

如果希望新会话连续完成 D11–D15，将启动提示中的任务范围一句改为“本次连续完成 D11–D15，按依赖逐项验证后推进”；其余约束保留。
