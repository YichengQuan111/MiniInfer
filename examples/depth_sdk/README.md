# External depth SDK example

Install MiniInfer with its CMake package configuration, then build this directory as an independent C++17 project:

```sh
cmake -S examples/depth_sdk -B /tmp/miniinfer-sdk-example \
  -DCMAKE_PREFIX_PATH=/path/to/miniinfer/install \
  -DMINIINFER_ORT_ROOT=/path/to/onnxruntime \
  -DMINIINFER_TRT_LIB=/path/to/tensorrt_libs \
  -DMINIINFER_CUDA_RUNTIME=/path/to/nvidia/cuda_runtime
cmake --build /tmp/miniinfer-sdk-example
/tmp/miniinfer-sdk-example/depth_sdk_example /path/to/model-package input.ppm output.f32 cpu 1
```

Only pass dependency roots for backends enabled in the installed SDK. Runtime shared libraries must be on the loader search path (system installation or `LD_LIBRARY_PATH`). `source tools/depth_env.sh` is available from a prepared MiniInfer checkout using its configured Python environment/cache; it is not a universal dependency installer. The installed CMake exports resolve imported dependency targets on the consumer machine and do not embed the original build machine's library paths.

`find_package(MiniInfer CONFIG REQUIRED)` supplies `miniinfer::depth_sdk`; its public headers expose no ORT, TensorRT or CUDA types. A consumer linking an already built static SDK needs the matching runtime libraries and a C++ compiler, but does not need nvcc to compile this example. Building the SDK's CUDA preprocessing implementation itself requires a compatible CUDA toolkit/nvcc. The explicit `MINIINFER_BUILD_CUDA_PREPROCESS=1`, `CUDACXX`, architecture and CUDA compiler flag setup is in the [deployment guide](../../docs/depth_deployment.md).

## Package and command line

The example and installed SDK CLI accept the same positional arguments:

```text
depth_sdk_example PACKAGE_DIR IMAGE.ppm OUTPUT.f32 [MODE [SLOTS]]
depth_sdk_cli     PACKAGE_DIR IMAGE.ppm OUTPUT.f32 [MODE [SLOTS]]
```

`MODE` defaults to `cpu`; `SLOTS` defaults to `1` and accepts only `1` or `2`. For example, `... output.f32 cuda-full-fused 2` explicitly requests the fused CUDA path and two GPU slots; it is a configuration example, not a final performance recommendation. Each output is an independently owned FP32 row-major depth map restored to the original image dimensions. Larger values mean nearer objects; values are relative, without per-image min/max scaling or meter units. The example writes little-endian `.f32` exclusively and refuses to overwrite existing output.

The SDK selects one backend at initialization from the package's UTF-8 `manifest.txt`, with exactly these six key/value fields:

```text
schema=miniinfer-model-1
contract=depth-v1
backend=ort
precision=fp32
model=depth.onnx
model_sha256=REPLACE_WITH_THE_64_CHARACTER_LOWERCASE_SHA256
```

`depth-v1` identifies static `[1,3,518,518]` input; `depth-v2-392` identifies `[1,3,392,392]`. The package, model and engine must agree on their static contract. ORT supports FP32 packages. For TensorRT, set `backend=trt`, use `precision=fp32` or `fp16`, and name the engine file. Its `<engine>.manifest` must match that precision and pass the hardware/software compatibility checks. Engines do not promise portability across GPU, driver or backend versions; rebuild and validate in the target environment.

The model must resolve to one file inside the package. Path traversal, symlink escapes, hash mismatches and unavailable backends raise `ModelIncompatible`; backend initialization/execution failures raise `BackendFailure`. `InvalidInput` covers invalid configuration/input, including a GPU preprocessing mode with ORT. The package checks detect corruption and misconfiguration; they are not a digital signature. Use trusted packages and the offline package verifier described in the deployment guide.

## Synchronous and owning asynchronous calls

`run(image, &timing)` uses the same admission/execution/result path as `submit(image).get(&timing)`. `submit` copies the RGB8 input before returning. The caller must keep the image stable during that copy; after return, the accepted request owns its input. Callers may use different requests concurrently while keeping the SDK alive through calls to `submit`, `run` and `shutdown`.

```cpp
#include <miniinfer/depth/sdk.hpp>
#include <chrono>

miniinfer::depth::DepthMap infer(const std::filesystem::path& package,
                               const miniinfer::depth::RgbImage& image) {
    using namespace miniinfer::depth;
    DepthRequest request;
    {
        SdkConfig config;
        config.package_dir = package;
        config.worker_count = 4;
        config.gpu_slots = 1;
        config.preprocess_mode = PreprocessMode::Cpu; // Explicit compatible default.
        DepthSdk sdk(config);
        request = sdk.submit(image);
        const auto request_id = request.id();
        (void)request_id;
        if (!request.wait_for(std::chrono::milliseconds(5))) {
            // Timeout does not cancel: do other work, or wait for completion.
            request.wait();
        }
        if (request.ready()) {
            // A ready result still holds its admission reservation.
        }
        sdk.shutdown(); // Drains execution; does not wait for request.get().
    }
    DepthTiming timing;
    return request.get(&timing); // Handle and result can outlive the SDK.
}
```

`DepthRequest` is move-only. `id()` associates a result/error with its submission; `valid()` identifies an unconsumed handle. `ready()` polls, `wait()` blocks, and `wait_for(milliseconds)` returns whether execution reached its terminal state. A timeout does not cancel or release quota. `get()` consumes the handle once, including when it throws a correlated backend error; another `get()` on that handle raises `std::logic_error`. Read `DepthTiming` only after `get` or `run` returns/throws.

To discard a result, call `request.abandon()` or destroy the handle. Abandoning ready work releases its reservation immediately; abandoning queued/executing work keeps its copied input and reservation until execution safely completes. It does not cancel a GPU submission. For example:

```cpp
auto unused = sdk.submit(image);
unused.abandon();
sdk.shutdown(); // Still drains that accepted request.
```

`shutdown()` rejects new work, drains all accepted execution, and releases backend resources without waiting for ready handles to be consumed. Repeated/concurrent shutdown calls are supported. Handles retain their shared result/accounting state after SDK destruction; their `get()` can still succeed or report the stored error. Finish calls that access the SDK object before destroying it. A single request handle does not support simultaneous method calls from multiple threads. FIFO dequeue order does not imply FIFO completion.

## Workers, preprocessing and resource limits

`worker_count` defaults to four and is independent of `gpu_slots` (one or two). Each ORT worker owns a separate Session and CPU workspace; `ort_threads` defaults to four intra-op threads per Session. Calls through one `OrtRunner` retain its instance lock; different SDK workers use different Sessions. TensorRT slots each own their engine/context, stream, events, I/O and pinned buffers. CPU preparation for A/B completes before borrowing a GPU slot, allowing CPU work to overlap device execution. Creating two slots does not guarantee a speedup.

| Mode / configuration enum | Host preparation | Device input path |
|---|---|---|
| A: `cpu` / `PreprocessMode::Cpu` | Convert, resize, pad, normalize, NCHW | Upload prepared FP32; ORT also supports this mode |
| B: `cuda-partial` / `CudaPartial` | Convert, resize, pad FP32 HWC | Upload FP32 HWC, normalize/layout on GPU, then infer |
| C-basic: `cuda-full-basic` / `CudaFullBasic` | Geometry metadata | Upload original RGB8, complete GPU preprocessing, then infer |
| C-fused: `cuda-full-fused` / `CudaFullFused` | Geometry metadata | Original RGB8 with fused RGB coordinate work, then infer |

GPU modes require a TensorRT SDK built with CUDA preprocessing enabled; a legacy build rejects them before CUDA initialization. `parse_preprocess_mode` / `preprocess_mode_name` convert the exact strings above. `reuse_host_workspace` and `reuse_gpu_buffers` default to true and can be disabled for allocation comparisons. The C paths reserve a bounded RGB8 pinned/device staging capacity of `3 * max_pixels = 6,291,456` bytes per slot, derived from the original-image limit of 2,097,152 pixels, not from the 518/392 model resolution. Actual C H2D bytes are `3 * original_width * original_height`; A/B upload `3 * model_size * model_size * sizeof(float)`.

The default waiting queue capacity is four, request byte budget 128 MiB and device zero. Admission reserves `19*pixels + 16 MiB` before copying input. Waiting, executing and ready-but-unconsumed requests all count toward `queue_capacity + worker_count` and the byte budget; the sum must fit `std::size_t`. `Busy` rejects waiting-queue/count/byte exhaustion, `Closed` rejects new work once shutdown starts or a fatal GPU backend error closes admission. Quota is released by successful/error `get`, ready abandonment, or safe completion of abandoned pending work, once only.

`config()` exposes immutable initialization settings; `snapshot()` reports lifecycle/accounting, workspace/I/O capacities and allocation counters. The request budget excludes fixed workspaces/slots, model weights, ORT arenas, CUDA contexts and caller storage/results after delivery. It is not a process RSS or total VRAM limit. I/O counters omit backend/library allocations; `context_required_bytes` is a TensorRT static requirement, while `engine_serialized_bytes` sums serialized plan sizes per slot and is not exact GPU weight residency. After shutdown live host/I/O bytes are zero, but allocation counts, peaks and static engine/context indicators remain historical. Per-slot peak sums need not have occurred simultaneously.

CUDA-event `h2d_ms`, `gpu_preprocess_ms`, `model_ms` and `d2h_ms` use the slot's stream and are read after synchronization. `TrtTiming::compute_ms` remains the model-stage alias. Host wall measurements include queue/slot wait, CPU preprocessing, backend call, synchronization and restoration. `pinned_copy_ms` covers only input staging into pinned memory; the owning output copy remains in `host_ms` / `backend_ms`. `backend_ms` includes separately reported slot wait, and `synchronize_ms` overlaps device execution. These clocks and nested scopes are not additive and their differences do not prove overlap; use a separate timeline capture. `ready_at` records terminal publication rather than later consumer observation, and `h2d_bytes` counts actual successful input transfers (zero for ORT).

Current V2 quality evidence, pending performance/soak work and the eventual deployment selection are tracked in the [V2 upgrade report](../../docs/depth_v2_upgrade.md). The CPU default preserves existing callers; this README does not choose a final GPU mode.
