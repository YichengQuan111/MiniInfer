# V1.6 depth baseline method

`depth_benchmark` measures one synchronous caller with one model instance loaded once per process. `tools/run_depth_benchmark.py` repeats one case in three independent **client/benchmark** processes by default and writes a new directory containing `report.json` and `repeat_01/`, `repeat_02/`, `repeat_03/`. Each repeat contains `raw.csv`, `summary.json`, and `memory.csv`. An initialization failure can leave only a CSV header; the repeat is marked invalid, its stderr is preserved, and the runner exits nonzero after writing its report.

Run from Linux/WSL, using Linux executable, model, image, output, and PID paths. The runner does not start or stop a TCP server, build TensorRT engines, select numerical thresholds, or evaluate the final model set.

## Commands

```sh
# ORT raw inference; defaults: 100 steady attempts, 5 warmups, 3 repeats.
python3 tools/run_depth_benchmark.py --executable build/depth_benchmark \
  --mode raw --backend ort --model assets/model.onnx --image assets/image.ppm \
  --output reports/ort_raw

# SDK package; for TRT defaults: 300 steady attempts, 20 warmups, 3 repeats.
python3 tools/run_depth_benchmark.py --executable build/depth_benchmark \
  --mode sdk --backend trt --package packages/trt_fp32 --image assets/image.ppm \
  --output reports/trt_sdk

# External server must already be ready. Pass its Linux PID to sample server RSS.
python3 tools/run_depth_benchmark.py --executable build/depth_benchmark \
  --mode tcp --backend trt --host 127.0.0.1 --port 9000 --server-pid 12345 \
  --engine-manifest packages/trt_fp32/model.engine.manifest \
  --image assets/image.ppm --output reports/trt_tcp

# Minimal direct C++ run (CSV destination must not exist).
build/depth_benchmark --mode raw --backend trt --model assets/model.engine \
  --image assets/image.ppm --iterations 300 --warmup 20 --output raw.csv

python3 -m unittest discover -s tests -p test_depth_benchmark.py -v
```

Override `--iterations`, `--warmup` (zero permitted), or `--repeats` explicitly when needed and disclose these values. TCP uses `--timeout-ms 30000` for connect and each complete request/response operation; the runner also has a `--process-timeout-seconds 3600` overall deadline per repeat. These are termination limits, not a requested test duration. A positive iteration count is required. TCP metadata `--backend` describes the externally configured server; the transport does not attest its backend. Set the external server to the same settings and verify its package independently.

ORT threads are fixed at 4. SDK configuration uses one backend worker/GPU context slot, queue capacity 4, and a 128 MiB admission budget. The byte budget covers per-request reservations and excludes weights, backend arenas, CUDA context, and caller storage. Raw/TCP runs have one synchronous caller. The external TCP server must use equivalent worker/slot, queue, and admission settings; the runner cannot change them. Three independent TCP client processes reuse the same external server: they are not three independent server/model startups.

## Timing boundaries

1. `startup_ms` begins at the first instruction in C++ `main`, before argument parsing, image decoding, raw preprocessing, and model/SDK construction. It excludes OS process creation and dynamic loading. It ends when local setup is ready; TCP connection setup follows and appears as `connect_ms`. TCP has no local model initialization.
2. C++ emits and flushes a `ready` JSON line after initialization and, for TCP, after connection. The Python runner measures `process_to_ready_ms` from immediately before `Popen` through receipt of this line. This includes process launch, dynamic loading, setup, model loading, TCP connect if applicable, and pipe reception delay. `main_to_ready_ms` in the ready event is the corresponding C++ main-relative duration.
3. Phase `first` executes exactly one real request **before** warmup. `first_ms` and its CSV row represent that request, including first-use allocations/JIT effects. TCP's first request is the first request on this new connection; the external server may already have run inference.
4. Phase `warmup` executes the requested count; every attempt is recorded and counted but excluded from steady distributions and throughput.
5. Phase `steady` executes the requested count. Every attempt produces one row, including failures and rejections. Successful latency percentiles use only successful steady rows. Throughput is `successful steady count / elapsed_steady_ms * 1000`. The denominator covers the entire steady loop, including rejected/failed attempts, CSV writes, and result disposal between requests.

The per-request clock stops while the returned owned output is still alive, before output disposal and CSV formatting. Raw mode preprocesses the PPM exactly once before ready, then times `run_raw(PreparedImage)` through owned raw output completion. SDK mode times RGB input through `DepthSdk::run` returning an owned restored `DepthMap`, including preprocessing, admission/copying, execution, and postprocessing. TCP decodes the PPM before ready, keeps one connection, and times request encoding/sending through complete response receive/decode/validation. The deadline covers both send and receive.

All APIs must synchronously complete the operation before returning. Raw TensorRT adds CUDA events on the actual stream around H2D, compute, and D2H and reads them only after stream synchronization. CSV contains these device durations only for raw TRT. Other backend/layer rows leave device fields empty; host wall time includes additional host scheduling and output handling and must not be reconstructed by adding GPU event durations. TRT `host_ms` exists in the backend timing API but the benchmark's wall clock is the reported host metric.

Device event instrumentation adds overhead to the raw TRT case; disclose this when comparing against SDK/TCP. It does not imply SDK/TCP GPU event time was zero.

The current WSL lifecycle checks observed FP32 GPU `compute_ms = 53.126` with host `host_ms = 50.326`, and FP16 GPU `compute_ms = 61.4571` with host `host_ms = 57.7823`, while stream synchronization and numerical checks passed. These are observations from that environment, not suite performance results. CPU and GPU use separate clocks here: do not assume cross-clock ordering, subtract the durations to infer overhead, or force GPU intervals to fit the host wall. Use measured host wall for the primary performance comparison and retain device intervals as independent observations with their original values.

## Formats and statistics

CSV columns are `phase,index,status,wall_ms,h2d_ms,compute_ms,d2h_ms`; index is zero-based within each phase. Status is one of `succeeded`, `busy`, `closed`, `failure`, `timeout`. SDK Busy and Closed map directly; other SDK/backend exceptions map to failure. TCP Busy closes the client connection because that protocol response ends the server session. EOF/truncated/stopped I/O maps to closed, timeout maps to timeout, and remaining I/O/protocol/backend errors map to failure. After an unusable connection, remaining planned attempts are recorded as closed; the benchmark does not reconnect. These terminal attempts remain part of the planned denominator and the successful-only latency distribution excludes them.

C++ stdout contains JSON ready, phase, and summary events. Summary counts cover first + warmup + steady, with a separate `steady_counts`; their sums must be `1 + warmup + iterations` and `iterations`. The Python runner checks counts against the CSV, row order/schema, finite nonnegative timings, first-request consistency, and phase ordering. Initialization/process/report format errors are separate from request statuses.

P50/P95/P99 use sorted samples and linear interpolation at `(n - 1) * q / 100`, matching NumPy's `method="linear"`. Empty distributions have count 0 and null percentiles. Each repeat is reported individually; percentiles across repeats are not averaged or silently pooled. A run can be a valid measurement while containing request failures; `all_runs_valid` means the measurement/report checks passed, not that every request succeeded. Inspect `phase_counts` and successful counts alongside latency and throughput.

For TRT, `engine_build_seconds.seconds` is read from the existing manifest's `build_seconds` field, with the source path preserved. Raw defaults to `MODEL.manifest`, SDK to `PACKAGE/model.engine.manifest`, and TCP requires `--engine-manifest` to identify it. Missing or malformed fields yield null with a reason. This offline build duration is not an inference latency or startup duration, and no engine builder is invoked.

The report records executable/image hashes and a raw model hash when supplied. Hashing and earlier repeats can warm filesystem caches; independent processes do not imply a cold OS cache. Keep the SDK package manifests, engine manifests, build configuration, environment record, and external server command with the measurement for comparisons; an image or executable hash alone cannot establish identical environments.

## Memory sampling and limits

The runner reads `/proc/PID/status` VmRSS approximately every 20 ms. Raw/SDK RSS belongs to the benchmark. TCP client RSS and optional `--server-pid` RSS are separate series; client RSS must never be interpreted as server model memory. External server samples begin at client launch; they do not capture server initialization peaks.

`nvidia-smi --id=0 --query-gpu=memory.used` is sampled at a target interval of 200 ms. Each invocation has a three-second timeout, so slow queries can lower the actual sampling rate. Values are total GPU-device-0 used MiB, **including background processes**, not memory allocated by this process. CUDA backend device is fixed at zero. Missing tool/data is unavailable, not zero. No background subtraction or process attribution is performed.

`memory.csv` records elapsed time, phase, role, metric, and value (empty when unavailable). JSON summaries give observed peak and valid sample count for startup/first/warmup/steady/teardown and whole benchmark lifetime. Phases are assigned using stdout **reception** timestamps. Flush/pipe scheduling and sampler/query delay can place observations near a boundary in the adjacent phase. Short phases can have zero samples; their peak is null. Sampled peaks are lower bounds on peaks between samples, not allocator limits. Query observations completed after process exit are discarded.

SDK snapshot currently exposes final reservations and counters, not reservation peaks. The report includes `reservation_bytes_per_request` and final `reserved_bytes`; it does not label either as a measured peak.

This baseline is concurrency 1 and a finite requested attempt count. It establishes no overload/rejection-rate claim and no 30-minute stability claim. Model numerical acceptance remains a separate prerequisite; these small unit tests exercise statistics, failure accounting, CSV integrity, and memory role/phase handling without fake model performance tests.

## Sequential nine-case suite

`tools/run_depth_suite.py` orchestrates the full matrix in a fixed order: ORT FP32 raw/SDK/TCP, TRT FP32 raw/SDK/TCP, then TRT FP16 raw/SDK/TCP. It runs each case completely before starting the next, using three independent benchmark/client processes per case. ORT uses 100 steady requests and 5 warmups; TRT uses 300 and 20. No cases run in parallel. The package parent must contain `ort-fp32/`, `trt-fp32/`, and `trt-fp16/`, and the binary directory must contain `depth_benchmark` and `depth_server`.

```sh
python3 tools/run_depth_suite.py --bin-dir build-depth --packages packages \
  --output reports/depth_v16_suite

# Optional common image and bounded lifecycle timeout overrides.
python3 tools/run_depth_suite.py --bin-dir build-depth --packages packages \
  --image assets/image.ppm --output reports/depth_v16_suite_custom \
  --ready-timeout-seconds 180 --run-timeout-seconds 1800 --shutdown-timeout-seconds 60
```

The default image is `ort-fp32/golden/original.ppm` and all nine cases use that same image. The suite records package JSON, SDK manifest, model and engine manifest hashes, binary/image hashes, platform, `lscpu`, `nvidia-smi --version`, and GPU/driver identity from local tools. It checks package backend/precision and model inventory identity before launching benchmarks. It does not change precision criteria, rerun numerical acceptance, or build engines. Hashing and earlier cases can warm OS caches; this does not establish disk-cache-cold startup.

Each TCP case starts a fresh `depth_server --port 0` with one connection worker, SDK queue 4, 128 MiB SDK/network admission budgets, network admission count 4, and 30-second read/write deadlines. Its ready JSON supplies the PID, actual port, backend, precision, and main-based `startup_ms`; readiness requires no connection or inference probe. The suite separately records server Popen-to-ready time and `/proc` server RSS during startup at approximately 20 ms intervals in `server_startup_memory.csv/json`; device-0 total GPU-used MiB is sampled when `nvidia-smi` is available. Startup GPU samples can be absent in short phases. These server startup measurements remain separate from the existing runner's client Popen-to-ready and client/server RSS during requests.

For TCP, repeat 1's `first` row is this new server's actual first inference request, before any warmup or check request. Repeat 2 and 3's `first` rows are first requests on new client connections to the same already exercised server. They must not be presented as fresh model first-use measurements. Raw/SDK repeats each initialize their own model independently.

After the three clients finish, SIGTERM requests draining shutdown. The suite waits up to the shutdown deadline, captures the complete server `stopped` JSON and stdout/stderr, and kills the owned process group if cleanup cannot finish. Ready defaults to 180 seconds, the entire three-repeat runner case to 1800 seconds, and graceful server shutdown to 60 seconds; runner and benchmark descendants are cleaned up on case timeout. The suite sends no extra requests during startup or cleanup.

`suite.json` contains every command, package/model identity, runner report path, server lifecycle evidence, and per-case errors/pass status; `suite.inprogress.json` retains completed-case evidence during execution. Each case directory contains `case.json` and `measurements/report.json` alongside raw repeat artifacts and captured logs. `all_runs_valid` requires all nine runner reports to pass measurement validation. Suite `passed` additionally requires every first/warmup/steady attempt to succeed, all request error/rejection counters to be zero, and each TCP server's stopped counters to show the expected requests/responses, three completed connections, zero remaining reservations/work, and completed SDK shutdown. A suite process exit code of zero means that stricter check passed; an invalid or failed case remains recorded and later cases still run serially.
