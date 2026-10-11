# D16 Runtime / TCP 实测报告

2026-10-11 在 Release 下完成 15 组本地/TCP 对照；原始命令、配置、二进制与参考工件 SHA-256、环境和每次结果见 [d16_tcp.json](benchmark_data/d16_tcp.json)。本地和 TCP 各 **15,000 个正式请求成功**，业务拒绝、协议/连接错误和超时均为 0。成功数来自客户端完整解析/校验响应，未用服务端发送计数代替。

环境为同一 Windows 源码 / Ubuntu 24.04 WSL2、i9-13980HX、32 个来宾逻辑 CPU，内核 `6.18.40.1-microsoft-standard-WSL2`，MemTotal 16177044 kB。GCC 13.3.0、C++17、Release `-O3 -DNDEBUG`，无 sanitizer 或 fast-math；未绑核，供电、频率及后台负载未控制。正式测量在本轮 sanitizer 检查结束后执行。

固定模型 `64→128→10`，Batch 1/8/32/64/256；取参考前 B 行，256 由 64 行重复四次。每个 Batch 预热 100、正式 1000、重复 3 次。每组先启动独立进程内 Runtime，再启动独立 TCP 服务端/客户端；Worker=1，推理队列=64，Runtime future 窗口=1；TCP 并发客户端=1、连接线程=1、等待连接=8，一条连接贯穿预热和正式请求。无流水线、重试或到达率控制。

客户端用 `steady_clock` 从 `send_frame` 前量到完整响应校验/解码后；建连单列。输入 FP32 的请求体编码在正式循环外，帧构造/发送、服务端解码/入队/计算/回包及客户端解码都计入请求耗时。Runtime 仍按既有口径包含拥有型输入复制和 future 取得结果。两者不等同于相同算子的微基准，不能将差值全部解释为纯网络耗时，也不与 D14 的 window=8 结果直接计算加速比。

下表单位为 **μs**，每列为三次独立运行指标的中位数。每次运行的百分位采用最近秩 `ceil(p*N)`，只含成功的正式请求；不将不同进程的百分位合并重算。吞吐、每次完整百分位及计数保留在 JSON。

| Batch | Runtime P50 | P95 | P99 | TCP P50 | P95 | P99 | 建连 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 31.381 | 109.726 | 186.237 | 58.530 | 190.271 | 305.436 | 123.218 |
| 8 | 68.740 | 184.771 | 276.961 | 90.434 | 241.707 | 337.316 | 61.759 |
| 32 | 166.682 | 235.640 | 295.805 | 205.726 | 336.966 | 400.840 | 111.779 |
| 64 | 303.645 | 417.653 | 476.708 | 345.026 | 498.573 | 629.425 | 136.073 |
| 256 | 1167.052 | 1345.800 | 1520.766 | 1260.972 | 1558.699 | 1813.405 | 75.881 |

本轮 TCP 中位延迟高于本地路径，Batch 越小额外调度、序列化和收发成本越显著。这只是回环、单并发下的小规模基线；不推断局域网、负载饱和、多客户端扩展性或生产性能，也未据此修改算子或网络模型。BUSY 的正确映射与拒绝记账由 gate 契约测试单独验证，本轮未测饱和拒绝率。

PowerShell 复现（输出必须选择新文件）：

```powershell
$linuxProject = '/mnt/c/Users/ASUS/Documents/MiniInfer'
wsl -d Ubuntu-24.04 --cd $linuxProject -- cmake --preset release
wsl -d Ubuntu-24.04 --cd $linuxProject -- cmake --build --preset release
wsl -d Ubuntu-24.04 --cd $linuxProject -- /usr/bin/python3 tools/run_tcp_baseline.py --bin-dir /home/qyc/builds/MiniInfer/release/bin --reference data/reference --output docs/benchmark_data/d16_tcp_rerun.json --iterations 1000 --warmup 100 --repeats 3
```

驱动只用 Python 标准库，使用系统分配的回环端口并等待服务就绪，临时输入精确匹配 Batch。每次核验客户端末次 logits/labels 和全部响应计数，再与 Runtime 每成功请求 checksum 对照；失败路径限时终止并回收子进程。正式完整输出已实测，驱动的短流程和已有输出保护也纳入 CTest。

# D14 CPU / Runtime 历史实测报告

2026-10-10，Release 下完成 **36 次独立进程测量**。原始参数、环境、工件摘要、命令和完整 JSON 保存在 [d14_runtime.json](benchmark_data/d14_runtime.json)。这是本机固定小 MLP 的测量，未控制供电、频率和后台负载，不承诺稳定加速。

## 环境与负载

Windows 主源码经 Ubuntu 24.04 / WSL2 执行，内核 `6.18.40.1-microsoft-standard-WSL2`；来宾 lscpu 为 i9-13980HX、32 个逻辑 CPU，MemTotal 16177024 KiB。GCC 13.3.0、C++17、`-O3 -DNDEBUG -Wall -Wextra -Wpedantic`，无 sanitizer / fast-math / 指定本机 ISA。亲和集合、采集时系统负载、实际编译命令及可执行文件 SHA-256 保留在 JSON；未绑核。

Batch=1/8/32，重复正式参考输入的前 B 行，模型和输入不改变。每个 Batch / 重复按 direct_cpu、Runtime 1/2/4 Worker 顺序执行；预热 200 次、正式 **1000 次提交尝试**、重复 3 次。队列容量 64，调用者 future 窗口 8。直接路径是串行执行，实际在途为 1；Runtime 会扫描全部已就绪 future 后补充提交，仅在窗口满或全部尝试完成时阻塞等待。无独立生产速率控制。这轮各进程接受 / 成功均为 1000，任务失败 / 拒绝均为 0，总成功 36000；各 Batch 的每成功请求 checksum 一致。

两种模式都包含拥有型 Input 的复制。直接路径没有队列或 future；Runtime 普通 API 另有输入校验和任务 / future 构造。不能把这两条路径当作相同的 kernel 微基准。

## 时间与统计口径

使用 steady_clock：call_before 在输入复制前，enqueued 在队列锁内接受后，start / finish 在 Worker 的 callable 内，acquired 在 future.get 返回后。记录 finish 后才发布 future 完成；异常退出也记录 finish。

| 指标 | 计算与范围 |
|---|---|
| 端到端 mean / P50 / P95 / P99 | acquired−call_before；成功请求，包含输入复制和结果取得，不包含随后 checksum 消费 / 销毁 |
| queue_* | start−enqueued；直接执行为 0 |
| execution_* | finish−start；Runtime 包含 CpuExecutor callable 的执行，不是纯 Linear kernel |
| completion_* | finish−call_before；包含提交和执行，便于区分回收延迟 |
| collection_* | acquired−finish；包括完成发布、调度、future 扫描与采集等待；直接执行为 0 |
| submit_* | submit_return−call_before；直接路径为拥有型输入复制时间，不冒称排队时间 |
| 成功请求吞吐 | successes / 正式墙钟秒数；样本吞吐另乘 Batch |

正式墙钟还包括拒绝尝试、就绪扫描、等待、结果检查 / checksum 消费 / 销毁和指标记录；不含准备、预热、独立分配探测、最终 shutdown / 报告。结束前回收本轮全部接受请求，核验 `attempts=accepted+rejected`、`accepted=successes+failures`。拒绝不重试；错误不计成功吞吐。延迟统计只含成功样本，全部失败时统计为 0、样本数为 0。所有计时含测量开销。

百分位采用最近秩 `ceil(p*N)-1`。下表为三次进程指标的中位数，延迟单位均为 **μs**；各阶段分别取中位数，不能把不同列直接相加。原始文件包含每次所有百分位与吞吐、计数。

| Batch | 模式 / Worker | req/s | P50 | P95 | P99 | 排队均值 | 执行均值 | 回收均值 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | direct | 66736 | 14.702 | 17.119 | 27.239 | 0.000 | 14.837 | 0.000 |
| 1 | Runtime 1 | 37398 | 187.296 | 329.764 | 413.385 | 89.845 | 16.198 | 91.558 |
| 1 | Runtime 2 | 45954 | 119.890 | 203.019 | 262.723 | 51.256 | 19.388 | 50.323 |
| 1 | Runtime 4 | 50389 | 37.657 | 93.795 | 173.839 | 8.634 | 14.053 | 17.585 |
| 8 | direct | 20388 | 37.387 | 88.428 | 102.712 | 0.000 | 48.811 | 0.000 |
| 8 | Runtime 1 | 19817 | 376.586 | 629.020 | 785.600 | 205.958 | 43.455 | 145.698 |
| 8 | Runtime 2 | 25878 | 269.612 | 526.093 | 598.964 | 123.693 | 58.495 | 96.385 |
| 8 | Runtime 4 | 36223 | 154.928 | 329.749 | 433.131 | 48.348 | 63.767 | 62.612 |
| 32 | direct | 7179 | 137.061 | 154.766 | 205.970 | 0.000 | 138.766 | 0.000 |
| 32 | Runtime 1 | 6676 | 1271.308 | 1526.389 | 1651.838 | 586.138 | 139.458 | 464.759 |
| 32 | Runtime 2 | 12108 | 732.969 | 845.212 | 874.679 | 272.801 | 143.861 | 214.596 |
| 32 | Runtime 4 | 17753 | 417.437 | 697.504 | 981.348 | 123.231 | 159.395 | 132.534 |

Batch=1 的 Runtime 4 吞吐三次为 34437–61014 req/s，直接执行为 48810–78200 req/s，小任务上线程、future 和采集成本明显；本轮中位数没有超过直接执行。Batch=32 的 Runtime 4 为 17446–20516 req/s，直接为 6488–7208 req/s，计算增加后观察到并行收益；端到端尾延迟仍含窗口排队与回收。Batch=32 从 2 增到 4 Worker 时吞吐提高而 P99 中位数增大，不能仅凭吞吐宣称尾延迟改善，也不能把三次短运行推广到生产服务。

## 内存与 D13 分配分析

RSS 为 `/proc/self/statm` resident pages × page_size。下表覆盖所有 Batch / 重复，单位 MiB，采样时 Runtime 和线程仍存活，结果取得后及时消费并销毁；最多保留窗口内 futures。六组正式延迟数组预留 `6*1000*sizeof(double)=48000` 字节；这属于采集存储。

| 模式 | 预热后 RSS 范围 | 正式计时后 RSS 范围 |
|---|---:|---:|
| direct | 3.770–3.828 | 3.809–3.875 |
| Runtime 1 | 4.008–4.105 | 4.055–4.160 |
| Runtime 2 | 4.145–4.250 | 4.195–4.297 |
| Runtime 4 | 4.402–4.543 | 4.449–4.590 |

每 Worker 的隐藏层 Workspace 预留 128 KiB，四份权重有效载荷共 38440 字节，每份 Result 有效载荷 `B*44` 字节。RSS 另含输入加载 Buffer、拥有型 Input、队列 / future、线程实际驻留栈、标准库与分配器保留，不能只按这些有效载荷求和。

getrusage 高水位在本轮 36 次均为 **19.695 MiB**，包含启动、准备、预热和报告阶段，可能继承 Python 启动过程的高水位；没有在本轮分离启动贡献。它不是 Runtime 独占峰值，稳态比较使用以上两次 RSS。D8 也观察到启动方式相关差异，原始数据均保留。

直接执行独立探测 100 次，每进程 **200 个可替换 C++ new/new[] 事件**（每请求 2 次 Result 存储）；不含输入复制、计时记录、pool / Worker，亦不覆盖 malloc / mmap。Runtime 该字段 `allocation_probe_available=false`，计数 0 表示未测，不能解释为零分配。D13 实查 perf 命令不存在，`perf_event_paranoid=2`，见 [诊断](validation/d13_perf.log)；本轮没有热点采样，没有选择算子或内存优化，P1 不阻塞 v0.1 验收。

## 拒绝诊断与复现

[饱和诊断](benchmark_data/d14_saturation_probe.json)单独使用 Batch=64、Worker=1、队列=1、窗口=64、预热=20、尝试=1000：实际接受 / 成功 11、拒绝 989、任务失败 0；成功吞吐约 2214 req/s，以 11 个成功请求计算。此单次诊断用于核验拒绝记账，不作为主对照、稳定拒绝率或到达率曲线。

Windows PowerShell，从项目主目录执行，输出路径必须未存在：

```powershell
$linuxProject = '/mnt/c/Users/ASUS/Documents/MiniInfer'
wsl -d Ubuntu-24.04 --cd $linuxProject --exec cmake --preset release
wsl -d Ubuntu-24.04 --cd $linuxProject --exec cmake --build --preset release
wsl -d Ubuntu-24.04 --cd $linuxProject --exec /usr/bin/python3 tools/run_runtime_baseline.py --cli /home/qyc/builds/MiniInfer/release/bin/benchmark_cli --reference data/reference --output docs/benchmark_data/d14_runtime_rerun.json --iterations 1000 --warmup 200 --repeats 3 --window 8 --queue-capacity 64
```

Linux 在仓库根目录使用相同驱动，将 CLI 路径换为本机构建目录。驱动只需 Python 标准库，临时截取精确 Batch 输入，逐项校验有限 JSON、计数关系与每成功请求 checksum；文件已存在时拒绝覆盖。C++ 基准的参数错误退出 2，加载 / 设置错误退出 1，正常统计（包含已接受任务失败）退出 0；正式驱动会拒绝任务失败、无成功或契约不一致的报告。

D14 对齐了输入复制、提交阶段和取得结果边界，增加了阶段记录；D8 当时的单次延迟包含结果消费和销毁。两轮口径不同，不用 D8 数字计算本轮加速比。下面保留历史原始记录。

# D8 历史同步 CPU 基准

2026-10-10 已测量直接 CpuExecutor 路径。原始九次结果及环境见 [d8_sync.json](benchmark_data/d8_sync.json)，短内存诊断见 [d8_peak_probe.json](benchmark_data/d8_peak_probe.json)。这一节保留 D8 当时的计时口径；当前 Runtime 对照见上方 D14。

## 环境与方法

- Windows 唯一源码，由 Ubuntu 24.04.5 / WSL2 执行；内核 `6.18.40.1-microsoft-standard-WSL2`，x86_64。WSL `lscpu` 报告 Intel i9-13980HX、32 个逻辑 CPU、16 core/socket、1 socket；这是来宾拓扑信息。WSL 可见 MemTotal 16177040 KiB，约 15.43 GiB。
- GCC 13.3.0，Release，实际 CpuOps 编译选项 `-O3 -DNDEBUG -std=c++17 -Wall -Wextra -Wpedantic`，未启用 sanitizer、fast-math 或指定本机 ISA。CPU 亲和集合和完整选项保存在原始文件；本轮未绑核，未控制电源 / 频率或后台负载。
- 固定 `64→128→10` 标量 FP32 MLP；模型和输入 SHA-256 保存在原始结果。Batch=1/8/32，各取正式参考输入的前 B 行，每次请求重复同一个 Batch。输入已归一化；当前没有队列、生产速率或多 Worker 参数。
- 每个 Batch 独立启动三个进程，各预热 200 次、正式执行 1000 次；一次只保留当前 Result，消费后销毁。权重加载、输入准备、Workspace 分配、预热和报告均在正式计时之外，隐藏层最大 Batch=256，预分配 128 KiB。
- `steady_clock`：总墙钟包含 1000 次执行、结果检查 / checksum 消费 / 销毁、单次时钟及延迟记录。每次延迟包含 run、结果检查 / 消费和销毁，不含该次写入延迟数组；这是直接执行请求成本，包含校验及结果分配，不能当作纯算子时间。
- 请求吞吐为成功请求数 / 正式墙钟；样本吞吐为请求吞吐 × Batch。原始结果中成功均为 1000，失败 / 拒绝均为 0。P50/P95/P99 按排序后的最近秩 `ceil(p*N)-1` 取样。

## 实测结果

下表对每次进程的指标取三次中位数，吞吐同时展示三次范围；百分位不是把三次延迟混合后重新计算的百分位。完整每次数据保留在 JSON。

| Batch | req/s 中位数 [最小,最大] | P50 μs | P95 μs | P99 μs |
|---:|---:|---:|---:|---:|
| 1 | 119661 [101065,129816] | 7.875 | 9.204 | 9.491 |
| 8 | 24477 [24078,25280] | 37.195 | 62.354 | 62.495 |
| 32 | 7091 [6440,7115] | 139.292 | 147.632 | 164.762 |

请求单位对应整个 Batch，不能直接把三个 req/s 当作相同样本负载。Batch=1 吞吐和 Batch=32 尾延迟有波动；本表保留三次观测作为本机基线，后续对照须使用相同输入、编译选项和计时流程，并记录重复结果。

## 分配与内存口径

正式计时关闭计数，另运行 100 次直接执行：九个进程均记录 **200 次 C++ new 事件，即每请求 2 次**，对应 Result 的 logits 数组与 labels vector。Workspace 地址复用由契约测试验证；不宣称整次推理零分配。计数覆盖本基准程序可替换的 global `new/new[]` 及 aligned 形式，不覆盖绕过它们的 malloc、mmap、库内部来源或全进程其他线程；实际计时程序仍包含关闭状态的计数分支。Result 两块数据有效载荷为 `B*44` 字节，不等于分配器或进程总占用。

RSS 使用 `/proc/self/statm` resident pages × `_SC_PAGESIZE`，在预热后与正式计时后采样；九次分别为 **3.660–3.715 MiB** 和 **3.723–3.785 MiB**。后一时点包含预先保留的延迟数组。CLI 的输入加载 Buffer 与拥有型 Input 在本次流程中均保持存活，进程 RSS 还包含标准库、映射与分配器；不能用权重 / Workspace 的有效载荷推导全部内存。

峰值使用 `getrusage(RUSAGE_SELF).ru_maxrss`，Linux KiB 转字节，采样于报告构造后，包含准备 / 预热 / 分配探测及报告阶段。Python 子进程启动的九次高水位为 **17.457–17.707 MiB**；直接 WSL 启动的 Batch=64 短诊断高水位仅 **3.766 MiB**。这表明该高水位口径可能受启动器及启动阶段影响，不能把 17 MiB 当作执行器独占峰值；两个诊断流程的负载不同，未作内存优化对照。原始值如实保留，稳态使用上述两次 RSS 采样。

## 复现

在 Windows PowerShell 顺序执行：

```powershell
$linuxProject = '/mnt/c/Users/ASUS/Documents/MiniInfer'
wsl -d Ubuntu-24.04 --cd $linuxProject --exec cmake --preset release
wsl -d Ubuntu-24.04 --cd $linuxProject --exec cmake --build --preset release
wsl -d Ubuntu-24.04 --cd $linuxProject --exec /usr/bin/python3 tools/run_sync_baseline.py --cli /home/qyc/builds/MiniInfer/release/bin/benchmark_cli --reference data/reference --output docs/benchmark_data/d8_sync_rerun.json --iterations 1000 --warmup 200 --repeats 3
```

驱动只用 Python 标准库；从正式输入截取前 B 行到临时文件，每个配置 / 重复启动独立进程，成功后清理临时输入。输出路径必须未存在，复跑选择新路径；JSON 中的临时输入路径用于追溯调用，复现由驱动重新创建。输入必须精确为 `B*64` 个 little-endian FP32 值，不能直接将 64 行正式文件作为 Batch=1 的输入。
