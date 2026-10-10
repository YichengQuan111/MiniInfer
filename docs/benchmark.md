# D8 同步 CPU 基准

2026-10-10 已测量直接 CpuExecutor 路径。原始九次结果及环境见 [d8_sync.json](benchmark_data/d8_sync.json)，短内存诊断见 [d8_peak_probe.json](benchmark_data/d8_peak_probe.json)。Runtime 多 Worker 基准、队列耗时和拒绝负载实验属于后续任务，当前数据没有加速对照。

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
