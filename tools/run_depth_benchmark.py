#!/usr/bin/env python3
"""One-case, concurrency-one depth baseline; run on Linux/WSL near the benchmark."""
import argparse
import bisect
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time


STATUSES = ("succeeded", "busy", "closed", "failure", "timeout")
PHASES = ("startup", "first", "warmup", "steady", "teardown")
CSV_FIELDS = ("phase", "index", "status", "wall_ms", "h2d_ms", "compute_ms", "d2h_ms")


def percentile(values, q):
    """numpy percentile(method='linear') equivalent, with null for no samples."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values):
    return {"samples": len(values), "p50_ms": percentile(values, 50),
            "p95_ms": percentile(values, 95), "p99_ms": percentile(values, 99)}


def finite_number(value, name):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"invalid {name}: {value}")
    return result


def summarize_csv(path, summary):
    counts = {status: 0 for status in STATUSES}
    by_phase = {phase: dict(counts) for phase in ("first", "warmup", "steady")}
    lengths = {"first": 1, "warmup": summary["warmup"], "steady": summary["iterations"]}
    expected = [(phase, index) for phase, length in lengths.items() for index in range(length)]
    latencies, devices = [], {key: [] for key in CSV_FIELDS[4:]}
    all_wall_ms = 0
    first_ms = None
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(CSV_FIELDS):
            raise ValueError("CSV schema mismatch")
        rows = list(reader)
    if len(rows) != len(expected):
        raise ValueError("CSV planned attempt count mismatch")
    for row, (phase, index) in zip(rows, expected):
        if row["phase"] != phase or row["index"] != str(index) or row["status"] not in STATUSES:
            raise ValueError("CSV phase/index/status mismatch")
        if None in row or any(value is None for value in row.values()):
            raise ValueError("CSV column count mismatch")
        status = row["status"]
        counts[status] += 1
        by_phase[phase][status] += 1
        wall = finite_number(row["wall_ms"], "wall_ms")
        if phase == "steady":
            all_wall_ms += wall
        if phase == "first":
            first_ms = wall
        for key in devices:
            if row[key] != "":
                value = finite_number(row[key], key)
                if phase == "steady" and status == "succeeded":
                    devices[key].append(value)
        if phase == "steady" and status == "succeeded":
            latencies.append(wall)
    if counts != summary["counts"] or by_phase["steady"] != summary["steady_counts"]:
        raise ValueError("summary/CSV terminal counts mismatch")
    if not math.isclose(first_ms, finite_number(summary["first_ms"], "first_ms"), rel_tol=1e-10, abs_tol=1e-9):
        raise ValueError("first request mismatch")
    elapsed = finite_number(summary["elapsed_steady_ms"], "elapsed_steady_ms")
    if elapsed < all_wall_ms - 1e-6:
        raise ValueError("steady elapsed shorter than request wall times")
    for key in ("startup_ms", "connect_ms"):
        finite_number(summary[key], key)
    return {"attempt_counts": counts, "phase_counts": by_phase,
            "successful_steady_latency": distribution(latencies),
            "successful_throughput_per_second": len(latencies) * 1000 / elapsed if elapsed > 0 else None,
            "elapsed_steady_ms": elapsed,
            "device_successful_steady": {key: distribution(value) for key, value in devices.items()}}


def rss_kib(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return None


def phase_at(timestamp, boundaries):
    position = bisect.bisect_right([item[0] for item in boundaries], timestamp) - 1
    return boundaries[max(position, 0)][1]


def summarize_memory(samples):
    result = {}
    for role, metric in sorted({(sample["role"], sample["metric"]) for sample in samples}):
        matching = [sample for sample in samples if sample["role"] == role and sample["metric"] == metric]
        result[f"{role}:{metric}"] = {}
        for phase in ("whole",) + PHASES:
            values = [sample["value"] for sample in matching
                      if sample["value"] is not None and (phase == "whole" or sample["phase"] == phase)]
            result[f"{role}:{metric}"][phase] = {"samples": len(values), "peak": max(values) if values else None}
    return result


def json_exclusive(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def engine_build_seconds(args):
    if args.backend != "trt":
        return None
    if args.engine_manifest:
        path = args.engine_manifest
    elif args.model:
        path = Path(str(args.model) + ".manifest")
    elif args.package:
        path = args.package / "model.engine.manifest"
    else:
        return {"seconds": None, "source": None, "reason": "TCP server engine manifest not supplied"}
    try:
        fields = dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
        return {"seconds": finite_number(fields["build_seconds"], "build_seconds"), "source": str(path), "manifest_field": "build_seconds"}
    except (OSError, KeyError, ValueError) as error:
        return {"seconds": None, "source": str(path), "reason": str(error)}


def run_once(args, directory):
    directory.mkdir()
    command = [str(args.executable), "--mode", args.mode, "--backend", args.backend,
               "--image", str(args.image), "--iterations", str(args.iterations),
               "--warmup", str(args.warmup), "--output", str(directory / "raw.csv")]
    if args.mode == "raw":
        command += ["--model", str(args.model)]
    elif args.mode == "sdk":
        command += ["--package", str(args.package)]
    else:
        command += ["--host", args.host, "--port", str(args.port), "--timeout-ms", str(args.timeout_ms)]
    start = time.perf_counter()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    events = queue.Queue()
    stdout_lines, stderr_lines, samples, errors = [], [], [], []
    boundaries = [(start, "startup")]
    gpu_stop = threading.Event()
    process_to_ready_ms = None
    summary = None
    def read_stdout():
        for line in process.stdout:
            timestamp = time.perf_counter()
            stdout_lines.append(line)
            try:
                events.put((timestamp, json.loads(line)))
            except ValueError:
                events.put((timestamp, {"event": "invalid", "line": line.strip()}))
    def read_stderr():
        for line in process.stderr:
            stderr_lines.append(line)
    def gpu_sample():
        executable = shutil.which("nvidia-smi")
        if not executable:
            samples.append({"timestamp": time.perf_counter(), "role": "gpu_device_0_total", "metric": "used_mib", "value": None})
            return
        while not gpu_stop.is_set():
            began = time.perf_counter()
            value = None
            try:
                result = subprocess.run([executable, "--id=0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                        capture_output=True, text=True, timeout=3)
                if result.returncode == 0:
                    value = finite_number(result.stdout.strip(), "GPU total used MiB")
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
            samples.append({"timestamp": time.perf_counter(), "role": "gpu_device_0_total", "metric": "used_mib", "value": value})
            gpu_stop.wait(max(0, .2 - (time.perf_counter() - began)))
    threads = [threading.Thread(target=read_stdout), threading.Thread(target=read_stderr), threading.Thread(target=gpu_sample)]
    for thread in threads:
        thread.start()
    roles = {"client" if args.mode == "tcp" else "benchmark": process.pid}
    if args.server_pid:
        roles["server"] = args.server_pid
    def consume_events():
        nonlocal process_to_ready_ms, summary
        while True:
            try:
                timestamp, event = events.get_nowait()
            except queue.Empty:
                break
            if event.get("event") == "ready":
                if process_to_ready_ms is not None:
                    errors.append("duplicate ready event")
                process_to_ready_ms = (timestamp - start) * 1000
            elif event.get("event") == "phase":
                phase = event.get("phase")
                if phase not in PHASES[1:4]:
                    errors.append("invalid phase event")
                else:
                    boundaries.append((timestamp, phase))
            elif event.get("event") == "summary":
                if summary is not None:
                    errors.append("duplicate summary event")
                summary = event
                boundaries.append((timestamp, "teardown"))
            else:
                errors.append("unknown or malformed stdout event")
    try:
        while True:
            consume_events()
            timestamp = time.perf_counter()
            for role, pid in roles.items():
                samples.append({"timestamp": timestamp, "role": role, "metric": "rss_kib", "value": rss_kib(pid)})
            if process.poll() is not None:
                break
            if timestamp - start > args.process_timeout_seconds:
                errors.append("whole benchmark process deadline exceeded")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                break
            time.sleep(.02)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        process_end = time.perf_counter()
        gpu_stop.set()
        for thread in threads:
            thread.join(timeout=5)
        consume_events()
        process.stdout.close()
        process.stderr.close()
    if process.returncode != 0:
        errors.append(f"process exited {process.returncode}")
    if process_to_ready_ms is None or summary is None:
        errors.append("ready/summary event missing")
    if [phase for _, phase in boundaries if phase in PHASES[1:4]] != ["first", "warmup", "steady"]:
        errors.append("phase events missing or out of order")
    statistics = None
    if summary is not None:
        try:
            if summary["warmup"] != args.warmup or summary["iterations"] != args.iterations:
                raise ValueError("summary requested counts mismatch")
            statistics = summarize_csv(directory / "raw.csv", summary)
            if args.mode != "raw" or args.backend != "trt":
                if any(value["samples"] for value in statistics["device_successful_steady"].values()):
                    raise ValueError("device timings reported outside raw TRT")
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(str(error))
    boundaries.sort(key=lambda item: item[0])
    samples = [sample for sample in samples if sample["timestamp"] <= process_end]
    with (directory / "memory.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("elapsed_ms", "phase", "role", "metric", "value"))
        writer.writeheader()
        for sample in sorted(samples, key=lambda item: item["timestamp"]):
            sample["phase"] = phase_at(sample["timestamp"], boundaries)
            writer.writerow({"elapsed_ms": (sample["timestamp"] - start) * 1000, "phase": sample["phase"],
                             "role": sample["role"], "metric": sample["metric"], "value": sample["value"]})
    result = {"command": command, "pid": process.pid, "server_pid": args.server_pid,
              "process_to_ready_ms": process_to_ready_ms, "process_summary": summary,
              "statistics": statistics, "memory": summarize_memory(samples), "errors": errors,
              "returncode": process.returncode, "stderr": "".join(stderr_lines), "stdout": "".join(stdout_lines)}
    json_exclusive(directory / "summary.json", result)
    return result


def positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def nonnegative(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--mode", choices=("raw", "sdk", "tcp"), required=True)
    parser.add_argument("--backend", choices=("ort", "trt"), required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new report directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=positive)
    parser.add_argument("--server-pid", type=positive)
    parser.add_argument("--engine-manifest", type=Path)
    parser.add_argument("--iterations", type=positive)
    parser.add_argument("--warmup", type=nonnegative)
    parser.add_argument("--repeats", type=positive, default=3)
    parser.add_argument("--timeout-ms", type=positive, default=30000)
    parser.add_argument("--process-timeout-seconds", type=positive, default=3600)
    args = parser.parse_args()
    if os.name != "posix" or not Path("/proc/self/status").exists():
        parser.error("Linux/WSL with /proc required")
    if args.mode == "raw" and not args.model:
        parser.error("raw requires --model")
    if args.mode == "sdk" and not args.package:
        parser.error("sdk requires --package")
    if args.mode == "tcp" and (not args.port or args.port > 65535):
        parser.error("tcp requires --port in 1..65535")
    if args.server_pid and args.mode != "tcp":
        parser.error("--server-pid applies to TCP")
    if args.server_pid and rss_kib(args.server_pid) is None:
        parser.error("--server-pid has no readable VmRSS")
    args.iterations = args.iterations if args.iterations is not None else (300 if args.backend == "trt" else 100)
    args.warmup = args.warmup if args.warmup is not None else (20 if args.backend == "trt" else 5)
    for key in ("executable", "model", "package", "image", "output", "engine_manifest"):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    args.output.mkdir()  # Exclusive new destination; no silently merged reruns.
    report = {"schema": "miniinfer-depth-baseline-v1", "recorded_at": datetime.now(timezone.utc).isoformat(),
              "case": {"mode": args.mode, "backend": args.backend, "image": str(args.image), "model": str(args.model) if args.model else None,
                       "package": str(args.package) if args.package else None, "concurrency": 1,
                       "iterations": args.iterations, "warmup": args.warmup, "repeats": args.repeats,
                       "cpu_threads": 4, "gpu_slots": 1, "sdk_queue_capacity": 4, "sdk_byte_budget": 128 * 1024 * 1024},
              "identity": {"executable_sha256": sha256(args.executable), "image_sha256": sha256(args.image)},
              "engine_build_seconds": engine_build_seconds(args),
              "measurement": {"rss_interval_ms": 20, "gpu_target_interval_ms": 200,
                              "gpu_memory_scope": "device 0 total used MiB, includes background processes; NOT process allocation",
                              "phase_boundary": "stdout event reception time; sampling and pipe scheduling can lag the operation boundary",
                              "rss_scope": "VmRSS; TCP client and optional external server reported separately",
                              "startup_ms_scope": "C++ main entry through setup and model/SDK initialization; TCP connect separately",
                              "process_to_ready_ms_scope": "before Popen through flushed ready line reception, includes OS launch and dynamic loader",
                              "first_request": "before any warmup; actual first request after ready",
                              "percentiles": "successful steady attempts only; numpy linear definition",
                              "throughput": "successful steady count / elapsed entire steady phase seconds",
                              "gpu_samples": "sampling can miss short phases; unavailable observations remain null",
                              "tcp_server": "externally started and already ready; server model initialization and first-ever model inference not measured"},
              "runs": []}
    if args.model:
        report["identity"]["model_sha256"] = sha256(args.model)
    for repeat in range(args.repeats):
        result = run_once(args, args.output / f"repeat_{repeat + 1:02d}")
        report["runs"].append(result)
        print(json.dumps({"repeat": repeat + 1, "errors": result["errors"], "statistics": result["statistics"]}), flush=True)
    report["valid_runs"] = sum(not run["errors"] for run in report["runs"])
    report["all_runs_valid"] = report["valid_runs"] == args.repeats
    json_exclusive(args.output / "report.json", report)
    return 0 if report["all_runs_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
