# MLP 数据契约与参考工件

本目录保存 v0.1 固定 `64 → 128 → 10` MLP 的小型真实权重和参考样本。`reference/` 中的工件由 `tools/export_mlp.py` 生成，按项目约定纳入版本管理；C++ 模型测试直接加载四份权重，不要求先训练模型。D6 已对照正式输入 / logits / labels 验证同步前向，D10 的双 Worker 基本测试与直接执行器一致。

## 数据与计算约定

- 数据来自 scikit-learn 的 [digits 数据集](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)，8×8 图像展开为 64 个特征。
- 输入预处理为 FP32 `pixel / 16.0`，导出的输入已经归一化；后续 C++ 不再重复除以 16。
- 固定随机种子 42，按类别分层进行 80% / 20% 训练与测试划分；默认参考集取划分后测试索引序列的前 64 项，顺序保留。
- 使用 [MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) 的单隐藏层 ReLU 模型。Python 原始权重 `[in,out]` 在导出时转为 `[out,in]`，所有数组连续行主序。
- 参考 logits 由写出后重新读取的 FP32 输入、权重和偏置计算。logits 是第二层线性输出，不是概率或 softmax。
- 全部浮点文件为 little-endian FP32，整数文件为 little-endian int32；无文件头。argmax 并列取最小索引。

```text
X [N,64]
Z1 = X × W1ᵀ + b1
A1 = max(Z1, 0)
logits = A1 × W2ᵀ + b2
labels = argmax(logits, axis=1)
```

## 文件契约

默认 `N=64`。字节数为元素数量乘 4；加载器必须拒绝缺失文件、短文件、额外字节和读取失败，不能截断或补零。

| 文件 | 类型 | 形状 | N=64 时字节数 |
|---|---|---|---:|
| `w1.f32` | FP32 | `[128,64]` | 32768 |
| `b1.f32` | FP32 | `[128]` | 512 |
| `w2.f32` | FP32 | `[10,128]` | 5120 |
| `b2.f32` | FP32 | `[10]` | 40 |
| `inputs.f32` | FP32 | `[N,64]` | 16384 |
| `logits.f32` | FP32 | `[N,10]` | 2560 |
| `labels.i32` | int32 | `[N]` | 256 |
| `targets.i32` | int32 | `[N]` | 256 |
| `sample_indices.i32` | int32 | `[N]` | 256 |
| `manifest.json` | UTF-8 JSON | 元数据 | 以实际文件为准 |

`labels` 是预测类别，`targets` 是数据集真值，两者用途不同；`sample_indices` 用于核验样本来源、预处理与顺序。Runtime 默认最大请求 Batch 为 256；参考样本总数是工件集大小，不等于单请求 Batch。重复参考行可以验证较大 Batch，但不能算新增评测样本。

`manifest.json` 的 `format_version=1`。`files` 以文件名为键，保存 `dtype`（`<f4` / `<i4`）、`shape`、`elements`、`bytes`、`sha256`。`model` / `preprocessing` / `reference` 保存数值契约；`training` 保存种子、划分、参数、迭代次数、loss 和收敛告警；`environment` 保存依赖版本与线程限制；`metrics` 保存实测准确率及 sklearn 对照。JSON 用于工件追溯，不要求 C++ 引入 JSON 库。

## 重新生成

在 Windows PowerShell 中运行；脚本从主源码读取，已有 WSL Python 环境执行，产物回到 Windows 目录：

```powershell
$linuxProject = '/mnt/c/Users/ASUS/Documents/MiniInfer'
$linuxPython = '/home/qyc/projects/MiniInfer/.venv/bin/python'

wsl -d Ubuntu-24.04 --cd $linuxProject -- $linuxPython tools/export_mlp.py --output-dir data/generated/digits_seed42 --samples 64 --seed 42 --max-iter 500
```

输出目录须不存在或为空；非空目录会明确拒绝，重跑时选择另一个空目录。`data/generated/` 是临时重生成区，已被 Git 忽略；不要直接覆盖版本化的 `reference/`。正式参考工件的升级应对照元数据、文件摘要和测试后进行。

默认训练配置为 Adam、隐藏层 128、ReLU、Batch=64、学习率 0.001、alpha=0.0001、最多 500 次迭代、tol=1e-4、n_iter_no_change=20、shuffle=True、early_stopping=False；实际完整参数写入 manifest。训练和参考计算限制数值库为一个线程。收敛告警会记录，不以预设准确率代替实际结果。

当前依赖基线见 `tools/requirements.txt`。固定种子不保证跨不同 NumPy / BLAS / 平台得到逐字节相同工件；相同实测环境内的重复导出由测试验证，已有参考工件的身份以 SHA-256 为准。

## 正确性验收

运行 `tests/test_export_mlp.py`，验证非法参数、已有文件保护、同环境重复导出、形状与字节数、文件摘要、样本来源和独立标量 FP32 计算。初始数值标准为 `abs(actual-reference) <= 1e-4 + 1e-4 × abs(reference)`，同时要求预测类别一致。

D1 不包含 C++ 推理、CMake 工程、线程池或性能结果。模型测试准确率反映该固定划分上的分类效果，不能作为未来 C++ 推理速度或工程正确性的证明。
