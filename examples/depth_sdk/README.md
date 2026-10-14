# External synchronous depth SDK example

Install MiniInfer with its CMake package configuration, then build this directory as an independent project:

```sh
cmake -S examples/depth_sdk -B /tmp/miniinfer-sdk-example \
  -DCMAKE_PREFIX_PATH=/path/to/miniinfer/install \
  -DMINIINFER_ORT_ROOT=/path/to/onnxruntime \
  -DMINIINFER_TRT_LIB=/path/to/tensorrt_libs \
  -DMINIINFER_CUDA_RUNTIME=/path/to/nvidia/cuda_runtime
cmake --build /tmp/miniinfer-sdk-example
source tools/depth_env.sh # From the MiniInfer checkout for its pinned pip runtime.
/tmp/miniinfer-sdk-example/depth_sdk_example /path/to/model-package input.ppm output.f32
```

Only pass dependency roots for backends enabled in the installed SDK. Runtime shared libraries must be on the loader search path (system installation or `LD_LIBRARY_PATH`). The installed CMake exports refer to imported dependency targets and do not embed the original build machine's absolute library paths.

`find_package(MiniInfer CONFIG REQUIRED)` supplies `miniinfer::depth_sdk`; its public headers contain no ORT or TensorRT types. The SDK chooses one backend during initialization from the package's UTF-8 `manifest.txt`, with exactly these six key/value fields:

```text
schema=miniinfer-model-1
contract=depth-v1
backend=ort
precision=fp32
model=depth.onnx
model_sha256=REPLACE_WITH_THE_64_CHARACTER_LOWERCASE_SHA256
```

For TensorRT set `backend=trt`, use `precision=fp32` or `fp16`, and name the engine file. Its existing `<engine>.manifest` must match that precision and pass the backend's hardware/software compatibility checks. The model must resolve to one file inside the package; path traversal, symlink escapes, hash mismatches and unavailable backends raise `ModelIncompatible`. Initialization failures from the backend raise `BackendFailure`.

`DepthSdk(SdkConfig)` initializes one model and a thread pool with four workers by default. Set `SdkConfig::worker_count` to another positive value, including one for serial request execution. `run(const RgbImage&)` returns an owned FP32 row-major depth map restored to the original dimensions. Larger values mean nearer objects; these are relative values, without per-image min/max scaling or meter units. The example writes little-endian `.f32` exclusively and refuses to overwrite an existing output.

`config()` exposes the immutable initialization configuration; `snapshot()` reports lifecycle and admission counters. The default waiting queue capacity is four, worker count four, ORT intra-op thread count four, device zero and request byte budget 128 MiB. `worker_count` controls concurrent SDK requests; `ort_threads` controls the shared ORT session's internal computation threads. Admission reserves `19*pixels+16 MiB` before copying caller pixels and holds the reservation through result delivery. Count admission includes waiting, executing and ready results: at most `queue_capacity+worker_count` reservations, with the byte budget potentially admitting fewer. The sum must not overflow `std::size_t`. This budget excludes caller-owned input/results after delivery and fixed backend allocations such as weights, ORT arena and GPU context; it does not bound total process RSS or GPU memory.

Concurrent `run` calls can execute on up to `worker_count` workers. The ORT CPU backend shares its session while keeping each request's input, output and temporary storage independent; completion order need not match dequeue order. The TensorRT backend still serializes access to its single execution context/stream/buffer slot internally. Keep the SDK and the caller's input alive throughout each call, and do not mutate an input while its call is copying it. `Busy` rejects count/byte exhaustion, `InvalidInput` rejects malformed inputs, and `Closed` rejects requests after shutdown starts. `shutdown()` stops admission, drains accepted work, waits for their synchronous delivery and releases backend resources. Repeated/concurrent shutdown calls are supported; previously returned results retain independent storage. Caller code must finish all calls before destroying the SDK object. No public asynchronous interface or GPU cancellation is provided.
