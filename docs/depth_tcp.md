# V1.5 深度 TCP 契约

深度服务复用 D16 的 socket、带截止时间的收发、有限连接线程和关闭排空。`TcpServer` 增加会话回调重载；原 Echo/MLP 构造和 MINI v1 协议保持原行为。深度入口独立使用 MINI v2 与类型 3，不把旧 MLP 的消息长度限制放大。

## 使用

```bash
source tools/depth_env.sh
$HOME/builds/MiniInfer/depth-release/bin/depth_server \
  --package data/generated/depth/packages/trt-fp16 --host 127.0.0.1 --port 9001
# 另一个终端
$HOME/builds/MiniInfer/depth-release/bin/depth_client \
  --image data/generated/depth/images/indoor_1.ppm \
  --output /tmp/depth-network.f32 --host 127.0.0.1 --port 9001
```

服务器先初始化模型，成功后输出包含 PID、实际端口、后端和精度的 `ready` JSON；`--port 0` 自动分配端口。SIGINT/SIGTERM 停止接纳并排空，`stopped` JSON 记录请求/响应、错误、预算峰值和最终占用。客户端只在收到完整有效输出时创建结果文件，拒绝覆盖；超时覆盖连接、发送和接收整个操作。

## 帧格式

固定 16 字节 header，各整数字段大端：magic 4 字节 ASCII `MINI`；version uint16 = 2；type uint16 = 3（请求）或 0x8003（响应）；request_id uint32；body_size uint32。同一连接请求 ID 严格递增；一次完成一个请求/响应，可复用连接。

| 消息 | body 前 20 字节，依次为五个大端 uint32 | 后续数据 |
|---|---|---|
| 请求 | width、height、channels=3、format=1、reserved=0 | RGB8 HWC，连续 `3 × W × H` 字节 |
| 成功响应 | status=0、width、height、format=1、reserved=0 | 小端 IEEE FP32 HW，连续 `4 × W × H` 字节 |
| 错误响应 | status、0、0、0、0 | 无；body_size 必须为 20 |

输出恢复到原始 W/H，是相对逆深度，数值越大越近，不是米制距离，不做逐图 min/max 归一化。width/height 为正，单边不超过 4096，总像素不超过 2,097,152。请求和成功响应 body 长度必须与尺寸精确相符，最大分别为 6,291,476 和 8,388,628 字节。输出不允许 NaN/Inf。

status：0 Ok、1 BadRequest、2 Busy、3 ShuttingDown、4 InferenceFailed。服务端不在 wire 上发送内部异常文本。非法 header 直接关闭；合法 header 下的非法元数据或重复 ID 尽可能回复 BadRequest 后关闭；截断、读写超时结束会话。

## 两层预算与退出

默认 4 个连接线程、8 个等待连接，网络层共享 4 个在途请求与 128 MiB 字节预算。读取固定 header 和固定元数据并校验后，先预留 `14 × 像素数 + 1 MiB`，再分配 RGB 接收缓冲；预留覆盖接收、执行、结果及发送，直到响应发送成功或失败才释放。FP32 响应分块编码，慢读者不能导致无限缓存响应。

网络预算不足时，在读取 RGB payload 前回复 Busy 并关闭；客户端发送剩余 payload 可能先遇到断连，应重建连接后重试。SDK 接纳不足也映射 Busy。SDK 自身仍以 `19 × 像素数 + 16 MiB` 预留请求，默认 128 MiB；网络预算和 SDK 预算各管其拥有的存储，不表示进程总内存上限。

默认读、写截止时间各为 5 秒，分别由 `--read-timeout-ms` / `--write-timeout-ms` 设置。开始执行的推理不因客户端断连或停止信号取消；先完成设备工作，再做有截止时间的响应发送并释放预算。停止先中断新接纳和未完成读取，已接纳的业务按上述语义结束，最后关闭 SDK。GPU 卡死不属于网络超时能取消的范围。

## 实测回归

```bash
python tools/verify_depth_tcp.py --bin-dir $HOME/builds/MiniInfer/depth-release/bin \
  --package data/generated/depth/packages/ort-fp32 \
  --package data/generated/depth/packages/trt-fp32 \
  --package data/generated/depth/packages/trt-fp16 --output /tmp/depth-tcp-verification
```

工具为每个包启动全新服务器，将两次真实 TCP 客户端输出与相同包的本地 SDK 输出逐字节比较，验证 FP32 数量/有限性，再检查 SIGTERM 退出、请求计数和最终零预算。协议负例、最大图、断连、慢收发、预算拒绝后恢复、停止时完成已接纳推理由 C++ `depth_net_contract` 覆盖。验收证据见 [V1 记录](depth_validation.md)。
