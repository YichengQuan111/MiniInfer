# V2.1 CUDA trace 缺失诊断与工具修复

2026-10-11 在现有 Ubuntu-24.04 / WSL2、RTX 4080 Laptop、Windows driver 617.14 上完成。结果与完整命令、输出、SQLite 哈希见 [机器可读诊断](v21_profile_diagnosis.json)。原 V2.1 Profile 文件保留，不作为已具备 GPU 时间线的证据。

原 `timeline.sqlite` 有 14,750 条 CUDA runtime API、66 条同步事件和 NVTX，**不存在 CUDA kernel / memcpy 活动表**。`nsys profile` 返回 0、NVTX 存在及 CUDA API 可见都不足以证明 GPU 时间线有效。

诊断使用相同 CUDA 源码、nvcc 12.4.131、`-arch=sm_89 --fmad=false` 和现有驱动，分别编译运行极小程序：256 个线程执行一次加一，H2D / D2H 各 1,024 字节，并核对所有输出值。没有运行 TensorRT 或持续 GPU 负载；不同临时目录中的二进制哈希并不相同，源码哈希一致。

| Nsight Systems | 程序输出 | kernel 表行数 | memcpy 表行数 |
|---|---|---:|---:|
| 2023.4.4.54 | PASS | 表不存在 | 表不存在 |
| 2024.5.1.113 | PASS | 表不存在 | 表不存在 |
| **2026.5.1.161** | **PASS** | **1** | **2** |

2024.5 的 SQLite `DIAGNOSTIC_EVENT` 明确报告：`Installed CUDA driver version (13.4) is not supported by this build of Nsight Systems`，随后使用其自带 `libcupti.so.12.6`。2026.5 实际加载自带 `libcupti.so.13.4`，其 SQLite 有 `CUPTI_ACTIVITY_KIND_KERNEL=1` 与 `CUPTI_ACTIVITY_KIND_MEMCPY=2`。因此当前机器的缺失可通过升级 Nsight Systems 修复，不能归因于 WSL 的固有限制。该结论的直接验收范围是极小 CUDA 程序；原 TensorRT 回放仍需另行重采。

[NVIDIA 官方发行说明](https://developer.nvidia.com/nsight-systems/get-started)声明 2026.5 支持 CUDA 13.4，并列出 WSL-Ubuntu。工具选择必须兼顾宿主驱动 API，不能只按应用的 CUDA runtime 12.4 选择。PyTorch 2.6.0+cu124、TensorRT 10.8.0.43、CUDA runtime 12.4.127 与 nvcc 12.4.131 均保留。

## 局部安装与复现

工具来自 [NVIDIA 官方 Ubuntu 24.04 仓库](https://developer.download.nvidia.com/devtools/repos/ubuntu2404/amd64/Packages.gz)，安装使用 `dpkg-deb --extract`，不执行系统包安装或驱动更新。

- CLI 包：`NsightSystems-linux-cli-public-2026.5.1.161-3889610.deb`，225,594,886 字节。
- 包 SHA-256：`61829db6392e5c293ada1319df86a97c3356bc810335a1975cb551fcfe3eca08`。
- 官方 Packages.gz 快照 SHA-256：`35623a7aaa47b9f8c99f3a904ef4966891e5f52768447d6db1753a5e9d24087a`。

仓库索引会变化，应保留下载缓存中的已核验 `nsight-packages.gz`；脚本遇到不匹配的索引会拒绝，避免静默更换版本。CUDA 核心组件继续由原 12.4.1 manifest 锁定。

Windows 项目根目录下载：

```powershell
python tools/setup_depth_v2_tools.py fetch --downloads tmp/v2-tools --nsys-release cli-2026.5.1
```

WSL 项目根目录安装：

```bash
python3 tools/setup_depth_v2_tools.py install --downloads tmp/v2-tools \
  --nsys-release cli-2026.5.1 --report tmp/v2-tools/toolchain-nsys2026.json
source /home/qyc/.cache/miniinfer-depth/v2-tools/env.sh
source tools/depth_env.sh
```

实际 CLI：

```text
/home/qyc/.cache/miniinfer-depth/v2-tools/nsight_systems-2026.5.1.161/opt/nvidia/nsight-systems-cli/2026.5.1/target-linux-x64/nsys
```

极小验证使用独占新目录，GPU 正式计时期间不要并行运行：

```bash
python3 tools/setup_depth_v2_tools.py profile-smoke \
  --nsys "$(command -v nsys)" \
  --profile-output /home/qyc/.cache/miniinfer-depth/v2-tools/profile-smoke-reproduced \
  --report tmp/v2-tools/profile-smoke-reproduced.json
```

实际通过的短 trace 保存在 `/home/qyc/.cache/miniinfer-depth/v2-tools/profile-nsys2026/timeline.nsys-rep` 和同目录 `timeline.sqlite`。正式 Profile 应保持 `--trace=cuda,nvtx --sample=none --cpuctxsw=none`，在正常计时结束后单独重采原 V2.1 二进制，并再次检查实际 kernel / memcpy 表。Nsight Compute 计数器权限及 Compute Sanitizer 设备安全检查不由本次验证代替。
