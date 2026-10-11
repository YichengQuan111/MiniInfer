"""Small D16 Release comparison: Runtime and a single reused TCP connection."""
import argparse
from datetime import datetime, timezone, timedelta
import json
import math
import pathlib
import selectors
import struct
import subprocess
import tempfile

from run_runtime_baseline import collect_environment, execute_one, sha256_file


def ensure(ok, message):
    if not ok:
        raise ValueError(message)


def cleanup(server):
    if server.poll() is None:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)
    server.communicate(timeout=5)


def validate_tcp(result, batch, iterations, warmup, logits, labels):
    for name in ("business_errors", "connection_errors", "protocol_errors", "timeouts", "connect_errors", "connect_timeouts", "warmup_failures", "skipped"):
        ensure(result[name] == 0, f"TCP benchmark reported {name}")
    ensure(result["warmup_successes"] == warmup and result["warmup_attempts"] == warmup, "warmup mismatch")
    for name in ("attempts", "successes", "latency_samples", "requests_completed", "requests_planned"):
        ensure(result[name] == iterations, f"TCP {name} mismatch")
    ensure(result["batch"] == batch and result["status"] == 0, "batch/status mismatch")
    ensure(len(result["logits"]) == batch * 10 and len(result["labels"]) == batch, "result size mismatch")
    for i, value in enumerate(result["logits"]):
        expected = logits[i % 640]
        ensure(math.isfinite(value) and abs(value - expected) <= 1e-4 + 1e-4 * abs(expected), "TCP reference logits mismatch")
    ensure(result["labels"] == [labels[i % 64] for i in range(batch)], "TCP labels mismatch")
    for name in ("connect_ms", "wall_ms", "requests_per_sec", "p50_ms", "p95_ms", "p99_ms"):
        ensure(math.isfinite(result[name]) and result[name] >= 0, f"invalid {name}")
    ensure(result["p50_ms"] <= result["p95_ms"] <= result["p99_ms"], "unordered percentiles")
    checksum = sum(value * (1 + i / 100) for i, value in enumerate(result["logits"]))
    checksum += sum(value * (1 + i / 100) for i, value in enumerate(result["labels"]))
    return checksum


def tcp_once(bin_dir, reference, input_file, batch, args, expected_logits, expected_labels):
    server_command = [str(bin_dir / "infer_server"), "--model", str(reference), "--port", "0",
                      "--connections", "1", "--workers", "1", "--queue-capacity", "64",
                      "--connection-threads", "1", "--waiting-connections", "8"]
    server = subprocess.Popen(server_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(server.stdout, selectors.EVENT_READ)
            ensure(selector.select(5), "server readiness deadline")
            line = server.stdout.readline().strip()
        ensure(line.startswith("Listening on 127.0.0.1:"), f"bad readiness: {line!r}")
        port = int(line.rsplit(":", 1)[1])
        command = [str(bin_dir / "infer_client"), "--port", str(port), "--input", str(input_file),
                   "--batch", str(batch), "--requests", str(args.iterations), "--warmup", str(args.warmup)]
        process = subprocess.run(command, capture_output=True, text=True, timeout=60)
        ensure(process.returncode == 0 and not process.stderr, f"client failed: {process.stderr}")
        result = json.loads(process.stdout)
        checksum = validate_tcp(result, batch, args.iterations, args.warmup, expected_logits, expected_labels)
        out, err = server.communicate(timeout=5)
        ensure(server.returncode == 0 and not out and not err, f"server failed: {out} {err}")
        # The complete last result has been checked above; keep only metrics in
        # the report instead of repeating thousands of reference numbers.
        result.pop("logits")
        result.pop("labels")
        result["checksum_per_success"] = checksum
        return {"server_command": server_command, "command": command, "result": result}
    finally:
        cleanup(server)


def positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=pathlib.Path, required=True)
    parser.add_argument("--reference", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--iterations", type=positive, default=1000)
    parser.add_argument("--warmup", type=positive, default=100)
    parser.add_argument("--repeats", type=positive, default=3)
    args = parser.parse_args()
    ensure(not args.output.exists(), "output already exists; choose a new file")
    bin_dir, reference = args.bin_dir.resolve(), args.reference.resolve()
    inputs = (reference / "inputs.f32").read_bytes()
    ensure(len(inputs) == 64 * 256, "expected 64 reference input rows")
    logits = struct.unpack("<640f", (reference / "logits.f32").read_bytes())
    labels = struct.unpack("<64i", (reference / "labels.i32").read_bytes())
    report = {
        "schema": "miniinfer-d16-tcp-baseline-v1",
        "recorded_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "environment": collect_environment(bin_dir / "benchmark_cli"),
        "binaries": {name: sha256_file(bin_dir / name) for name in ("benchmark_cli", "infer_server", "infer_client")},
        "reference_sha256": {name: sha256_file(reference / name) for name in ("w1.f32", "b1.f32", "w2.f32", "b2.f32", "inputs.f32", "logits.f32", "labels.i32")},
        "workload": {"model": "64->128->10 FP32", "batches": [1, 8, 32, 64, 256],
                     "iterations": args.iterations, "warmup": args.warmup, "repeats": args.repeats,
                     "workers": 1, "queue_capacity": 64, "concurrent_clients": 1,
                     "connection_threads": 1, "waiting_connections": 8, "connection_reuse": True,
                     "request_order": "repeat the first B reference rows; B=256 repeats 64 rows four times",
                     "ordering": "for each batch/repeat: isolated Runtime process, then fresh server/client processes"},
        "measurement": {
            "tcp_latency": "client steady_clock before send_frame through complete response validation/decode; successful formal requests only",
            "tcp_wall": "formal loop including per-request accounting and last-result ownership, excluding connect/warmup/output",
            "connect": "client connect duration reported separately",
            "runtime_latency": "benchmark_cli call through future result acquisition, includes owned input copy; window=1",
            "comparison_limits": "TCP input encoded before formal loop; Runtime copies input per request. Different process/scheduling/serialization costs; no pure-network subtraction claim.",
            "percentiles": "nearest rank ceil(p*N); per run, not pooled",
        },
        "runs": [],
    }
    with tempfile.TemporaryDirectory(prefix="miniinfer-tcp-bench-") as directory:
        for batch in report["workload"]["batches"]:
            input_file = pathlib.Path(directory) / f"batch_{batch}.f32"
            input_file.write_bytes((inputs * 4)[:batch * 256])
            for repeat in range(1, args.repeats + 1):
                command = [str(bin_dir / "benchmark_cli"), "--model", str(reference), "--input", str(input_file),
                           "--batch", str(batch), "--iterations", str(args.iterations), "--warmup", str(args.warmup),
                           "--mode", "runtime", "--workers", "1", "--window", "1", "--queue-capacity", "64"]
                runtime, _ = execute_one(command, {"mode": "runtime", "workers": 1, "batch": batch,
                                                   "iterations": args.iterations, "warmup": args.warmup,
                                                   "window": 1, "queue_capacity": 64})
                ensure(runtime["successes"] == args.iterations and runtime["rejected"] == 0, "unexpected Runtime failure/rejection")
                tcp = tcp_once(bin_dir, reference, input_file, batch, args, logits, labels)
                ensure(math.isclose(runtime["checksum"] / runtime["successes"], tcp["result"]["checksum_per_success"],
                                    rel_tol=1e-6, abs_tol=1e-6), "local/TCP checksum mismatch")
                report["runs"].append({"batch": batch, "repeat": repeat,
                                       "runtime": {"command": command, "result": runtime}, "tcp": tcp})
                print(f"batch={batch} repeat={repeat}: local={runtime['successes']} TCP={tcp['result']['successes']}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(report, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write("\n")


if __name__ == "__main__":
    main()
