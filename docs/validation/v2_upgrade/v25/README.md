# V2.5 测量证据

`source-tree.txt` 标识正式测量的 Git 索引源码树；构建目录为 `/home/qyc/builds/MiniInfer/v25-upgrade-frozen`。各正式报告另外记录可执行文件、模型包和输入的 SHA256。配置为 `data/depth/v2_selected.json`，源码冻结后不在正式测量期间构建或运行其他 GPU 任务。

- `frozen-build.log`、`frozen-ctest.log`：正式 Release 构建及39项回归。
- `cpu/`：同一冻结代码的 CPU Debug、ASan/UBSan、TSan，各7项。TSan按既有环境要求使用 `setarch x86_64 -R`；不将第三方GPU库声称为CPU Sanitizer插桩对象。
- `selected-stress.json`、`selected-stress-memcheck.log`：最终完整CUDA基础模式、1/2槽位，真实GPU结果关联、放弃、ready预算、析构后交付及并发关闭检查；memcheck同时检查泄漏。
- `final/`：20请求/秒正常、40请求/秒容量探针、80请求/秒过载、200ms慢消费者，以及预热后1800秒长稳。40请求/秒是探针，不预设它使最终配置饱和。若80请求/秒未出现背压，另标160请求/秒测量。
- 每个 `*-acceptance.json` 重新读取原始CSV核对计数、输出抽检、资源上限和回收。只有 `require_soak=true` 且 `passed=true` 的完整长稳审计可作为30分钟验收证据。

每个成功请求核对尺寸及64个固定采样点；每128轮图像序列对所有图像各做一次完整位模式哈希。启动时每图一次基线调用另计，不计入正式吞吐。输出检查在SDK终结发布后执行，其CPU成本影响客户端交付，不能归入模型CUDA-event时间。

正式运行的 `sdk.csv` 含 `phase_boundary`：提交正式负载前在互斥保护下同步采样，防止首次一秒周期采样之前的分配增长漏检。分配计数从该边界到关闭必须保持恒定；运行中每行均检查三条请求守恒关系及预算。RSS约每50ms采样，完整60秒窗口及不少于30秒的充分采样尾窗评估64MiB增长门槛；所有阶段的RSS采样峰值遵守2GiB上限。设备总显存只作观测，不冒充进程显存。

`build.log`、`ctest-release.log`、`stress-cpu.*`、`stress-cuda-full-fused.*` 和 `preflight/` 是冻结前的开发检查。`preflight-acceptance.json` 是旧采样器/审计器的3秒诊断，不含同步阶段边界，不能作为当前正式稳定性门禁或30分钟证据；原文件保留，不改写为新实现的通过结果。

`interrupted-saturation-probe/` 保存会话中断时40请求/秒第三轮约6秒处停止的原批次；它不参与正式汇总。已通过的正常负载不重测，容量探针完整三轮及剩余场景从同一冻结源码/二进制接续。

`interrupted-soak-20261016/`、`interrupted-soak-20261017/` 分别保存约5分钟、约1分钟处终止的长稳尝试，无完整终态和审计，不作为30分钟证据。10月19日重新执行完整1800秒；正式短测与长稳分属不同日期，源码、二进制、模型和运行参数保持一致，不拼接中断前后的时间。
