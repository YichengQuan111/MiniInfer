#!/usr/bin/env python3
"""Sequential nine-case depth suite with a new, unprobed server per TCP case."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time

from run_depth_benchmark import finite_number, json_exclusive, positive, rss_kib, sha256


VARIANTS = (("ort-fp32", "ort", "fp32", 100, 5),
            ("trt-fp32", "trt", "fp32", 300, 20),
            ("trt-fp16", "trt", "fp16", 300, 20))
MODES = ("raw", "sdk", "tcp")
REPEATS = 3


def command_evidence(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        return {"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": str(error)}


def package_evidence(root, backend, precision):
    metadata_path = root / "package.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["backend"] != backend or metadata["precision"] != precision:
        raise ValueError(f"package backend/precision mismatch: {root}")
    model = root / ("model.onnx" if backend == "ort" else "model.engine")
    model_hash = sha256(model)
    inventory = metadata["files"]
    if inventory[model.name]["sha256"] != model_hash or inventory[model.name]["bytes"] != model.stat().st_size:
        raise ValueError(f"package model inventory mismatch: {root}")
    return {"package": str(root), "package_json_sha256": sha256(metadata_path),
            "manifest_txt_sha256": sha256(root / "manifest.txt"), "model": str(model), "model_sha256": model_hash,
            "backend": backend, "precision": precision,
            "engine_manifest_sha256": sha256(root / "model.engine.manifest") if backend == "trt" else None}


class CapturedProcess:
    """Capture pipes continuously and own a process group for bounded cleanup."""
    def __init__(self, command):
        self.command = command
        self.started = time.perf_counter()
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, bufsize=1, start_new_session=True)
        self.stdout, self.stderr, self.events = [], [], []
        self.incoming = queue.Queue()
        self.threads = [threading.Thread(target=self._read_stdout), threading.Thread(target=self._read_stderr)]
        for thread in self.threads:
            thread.start()

    def _read_stdout(self):
        for line in self.process.stdout:
            timestamp = time.perf_counter()
            self.stdout.append(line)
            try:
                event = json.loads(line)
            except ValueError:
                event = {"event": "invalid", "line": line.strip()}
            self.events.append(event)
            self.incoming.put((timestamp, event))

    def _read_stderr(self):
        for line in self.process.stderr:
            self.stderr.append(line)

    def terminate_group(self, grace=5):
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                pass
        # A killed runner can leave a benchmark or nvidia-smi alive in its group.
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if self.process.poll() is None:
            self.process.wait(timeout=5)

    def close(self, directory, stem):
        for thread in self.threads:
            thread.join(timeout=5)
        if any(thread.is_alive() for thread in self.threads):
            self.terminate_group(grace=0)
            for thread in self.threads:
                thread.join(timeout=5)
        self.process.stdout.close()
        self.process.stderr.close()
        (directory / f"{stem}.stdout.jsonl").write_text("".join(self.stdout), encoding="utf-8")
        (directory / f"{stem}.stderr.txt").write_text("".join(self.stderr), encoding="utf-8")


def wait_server_ready(server, directory, timeout):
    samples, stop = [], threading.Event()
    gpu_available = shutil.which("nvidia-smi")
    ready_time = None
    ready = None
    def sample_gpu():
        if not gpu_available:
            return
        while not stop.is_set():
            began = time.perf_counter()
            value = None
            try:
                result = subprocess.run([gpu_available, "--id=0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                        capture_output=True, text=True, timeout=3)
                if result.returncode == 0:
                    value = finite_number(result.stdout.strip(), "GPU total used MiB")
            except (OSError, subprocess.TimeoutExpired, ValueError):
                pass
            samples.append((time.perf_counter(), "gpu_device_0_total", "used_mib", value))
            stop.wait(max(0, .2 - (time.perf_counter() - began)))
    gpu_thread = threading.Thread(target=sample_gpu)
    gpu_thread.start()
    try:
        deadline = server.started + timeout
        while ready is None:
            now = time.perf_counter()
            samples.append((now, "server", "rss_kib", rss_kib(server.process.pid)))
            try:
                timestamp, event = server.incoming.get(timeout=.02)
            except queue.Empty:
                if server.process.poll() is not None:
                    raise RuntimeError("server exited before ready")
                if time.perf_counter() >= deadline:
                    raise TimeoutError("server ready timeout")
                continue
            if event.get("event") != "ready":
                raise ValueError("server emitted non-ready stdout before ready")
            ready, ready_time = event, timestamp
        if ready["pid"] != server.process.pid or not isinstance(ready["port"], int) or not 1 <= ready["port"] <= 65535:
            raise ValueError("server ready PID/port mismatch")
        if ready_time > deadline:
            raise TimeoutError("server ready event arrived after deadline")
        finite_number(ready["startup_ms"], "server startup_ms")
        return {"ready": ready, "process_to_ready_ms": (ready_time - server.started) * 1000}
    finally:
        stop.set()
        gpu_thread.join(timeout=4)
        end = ready_time if ready_time is not None else time.perf_counter()
        samples = [sample for sample in samples if sample[0] <= end]
        with (directory / "server_startup_memory.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("elapsed_ms", "phase", "role", "metric", "value"))
            writer.writerows(((timestamp - server.started) * 1000, "server_startup", role, metric, value)
                            for timestamp, role, metric, value in sorted(samples))
        metrics = {}
        for role, metric in (("server", "rss_kib"), ("gpu_device_0_total", "used_mib")):
            values = [value for _, r, m, value in samples if r == role and m == metric and value is not None]
            metrics[f"{role}:{metric}"] = {"samples": len(values), "observed_peak": max(values) if values else None}
        json_exclusive(directory / "server_startup_memory.json", {"metrics": metrics, "gpu_tool_available": bool(gpu_available),
                       "phase_end": "server ready stdout receipt", "rss_interval_ms": 20, "gpu_target_interval_ms": 200,
                       "gpu_scope": "device 0 total used MiB; includes background, not server allocation"})


def validate_report(report, iterations, warmup):
    errors = []
    if report.get("all_runs_valid") is not True or report.get("valid_runs") != REPEATS or len(report.get("runs", [])) != REPEATS:
        errors.append("runner did not produce three valid repeats")
    for index, run in enumerate(report.get("runs", []), 1):
        if run.get("errors") or run.get("returncode") != 0:
            errors.append(f"repeat {index} process/report error")
        summary = run.get("process_summary") or {}
        counts = summary.get("counts") or {}
        steady = summary.get("steady_counts") or {}
        if counts.get("succeeded") != 1 + warmup + iterations or any(counts.get(key) != 0 for key in ("busy", "closed", "failure", "timeout")):
            errors.append(f"repeat {index} request errors or count mismatch")
        if steady.get("succeeded") != iterations or any(steady.get(key) != 0 for key in ("busy", "closed", "failure", "timeout")):
            errors.append(f"repeat {index} steady request errors or count mismatch")
    return errors


def validate_stopped(stopped, iterations, warmup):
    errors = []
    planned = REPEATS * (1 + warmup + iterations)
    for key in ("rejected", "expired", "discarded", "bad_requests", "protocol_errors", "session_failures", "accept_errors",
                "busy", "inference_failures", "io_failures", "sdk_failed", "sdk_rejected_busy", "sdk_rejected_closed", "sdk_rejected_invalid",
                "active", "pending", "network_bytes", "network_inflight", "sdk_bytes", "sdk_inflight", "sdk_queued", "sdk_executing"):
        if stopped.get(key) != 0:
            errors.append(f"server stopped {key} missing/nonzero")
    for key in ("requests", "responses", "sdk_accepted", "sdk_succeeded", "sdk_delivered"):
        if stopped.get(key) != planned:
            errors.append(f"server stopped {key} planned count mismatch")
    if stopped.get("accepted") != REPEATS or stopped.get("completed") != REPEATS:
        errors.append("server connection count mismatch")
    if stopped.get("sdk_shutdown_complete") is not True or stopped.get("sdk_accepting") is not False:
        errors.append("server SDK did not finish shutdown")
    return errors


def run_case(args, variant, mode, identity):
    name, backend, precision, iterations, warmup = variant
    directory = args.output / f"{name}_{mode}"
    directory.mkdir()
    runner_script = Path(__file__).with_name("run_depth_benchmark.py").resolve()
    command = [sys.executable, str(runner_script), "--executable", str(args.bin_dir / "depth_benchmark"),
               "--mode", mode, "--backend", backend, "--image", str(args.image), "--iterations", str(iterations),
               "--warmup", str(warmup), "--repeats", str(REPEATS), "--output", str(directory / "measurements"),
               "--process-timeout-seconds", str(args.run_timeout_seconds)]
    result = {"case": directory.name, "backend": backend, "precision": precision, "mode": mode,
              "identity": identity, "command": command, "report_path": str(directory / "measurements" / "report.json"),
              "errors": [], "passed": False, "runner_all_runs_valid": False}
    server = runner = None
    try:
        if mode == "raw":
            command += ["--model", identity["model"]]
        elif mode == "sdk":
            command += ["--package", identity["package"]]
        else:
            server_command = [str(args.bin_dir / "depth_server"), "--package", identity["package"], "--host", "127.0.0.1",
                              "--port", "0", "--connections", "0", "--connection-threads", "1", "--waiting-connections", "4",
                              "--queue-capacity", "4", "--sdk-bytes", "134217728", "--network-requests", "4", "--network-bytes", "134217728",
                              "--read-timeout-ms", "30000", "--write-timeout-ms", "30000"]
            result["server_command"] = server_command
            server = CapturedProcess(server_command)
            result["server"] = {"pid": server.process.pid}
            result["server"].update(wait_server_ready(server, directory, args.ready_timeout_seconds))
            ready = result["server"]["ready"]
            if ready["backend"] != backend or ready["precision"] != precision:
                raise ValueError("server ready backend/precision mismatch")
            command += ["--host", "127.0.0.1", "--port", str(ready["port"]), "--server-pid", str(ready["pid"])]
            if backend == "trt":
                command += ["--engine-manifest", str(Path(identity["package"]) / "model.engine.manifest")]
            result["tcp_first_request_scope"] = {"repeat_01": "server's first-ever inference request; no earlier probe/warmup",
                                                 "repeat_02_and_03": "first request of new client connection; same warmed server"}
        runner = CapturedProcess(command)
        try:
            runner.process.wait(timeout=args.run_timeout_seconds)
        except subprocess.TimeoutExpired:
            result["errors"].append("runner group whole-case deadline exceeded")
            runner.terminate_group()
        result["runner_returncode"] = runner.process.returncode
        if runner.process.returncode != 0:
            result["errors"].append(f"runner exited {runner.process.returncode}")
        path = Path(result["report_path"])
        if not path.exists():
            raise ValueError("runner report.json missing")
        report = json.loads(path.read_text(encoding="utf-8"))
        result["runner_all_runs_valid"] = report.get("all_runs_valid") is True
        result["errors"].extend(validate_report(report, iterations, warmup))
    except (OSError, ValueError, KeyError, RuntimeError, TimeoutError) as error:
        result["errors"].append(str(error))
    finally:
        if runner is not None:
            if runner.process.poll() is None:
                runner.terminate_group()
            runner.close(directory, "runner")
        if server is not None:
            if server.process.poll() is None:
                server.process.send_signal(signal.SIGTERM)
                try:
                    server.process.wait(timeout=args.shutdown_timeout_seconds)
                except subprocess.TimeoutExpired:
                    result["errors"].append("server graceful shutdown timeout; killed process group")
                    server.terminate_group(grace=0)
            server.close(directory, "server")
            result["server"]["returncode"] = server.process.returncode
            result["server"]["events"] = server.events
            stopped = [event for event in server.events if event.get("event") == "stopped"]
            result["server"]["stopped"] = stopped[0] if len(stopped) == 1 else None
            if server.process.returncode != 0 or len(stopped) != 1:
                result["errors"].append("server exit failure or missing/duplicate stopped event")
            else:
                result["errors"].extend(validate_stopped(stopped[0], iterations, warmup))
            if len([event for event in server.events if event.get("event") == "ready"]) != 1 or any(event.get("event") not in ("ready", "stopped") for event in server.events):
                result["errors"].append("server stdout event sequence invalid")
    result["passed"] = result["runner_all_runs_valid"] and not result["errors"]
    json_exclusive(directory / "case.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--packages", type=Path, required=True, help="parent containing ort-fp32, trt-fp32, trt-fp16")
    parser.add_argument("--output", type=Path, required=True, help="new exclusive suite directory")
    parser.add_argument("--image", type=Path, help="defaults to ort-fp32/golden/original.ppm")
    parser.add_argument("--ready-timeout-seconds", type=positive, default=180)
    parser.add_argument("--run-timeout-seconds", type=positive, default=1800, help="whole runner case, all three repeats")
    parser.add_argument("--shutdown-timeout-seconds", type=positive, default=60)
    args = parser.parse_args()
    if os.name != "posix" or not Path("/proc/self/status").exists():
        parser.error("Linux/WSL with /proc required")
    args.bin_dir, args.packages, args.output = args.bin_dir.resolve(), args.packages.resolve(), args.output.resolve()
    args.image = (args.image or (args.packages / "ort-fp32" / "golden" / "original.ppm")).resolve()
    for executable in (args.bin_dir / "depth_benchmark", args.bin_dir / "depth_server"):
        if not executable.is_file() or not os.access(executable, os.X_OK):
            parser.error(f"executable missing/not executable: {executable}")
    if not args.image.is_file():
        parser.error(f"image missing: {args.image}")
    identities = {}
    try:
        for name, backend, precision, _, _ in VARIANTS:
            identities[name] = package_evidence(args.packages / name, backend, precision)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    args.output.mkdir()
    suite = {"schema": "miniinfer-depth-nine-case-suite-v1", "recorded_at": datetime.now(timezone.utc).isoformat(),
             "environment": {"platform": platform.platform(), "uname": dict(platform.uname()._asdict()),
                             "python": sys.version, "lscpu": command_evidence(["lscpu"]),
                             "nvidia_smi_version": command_evidence(["nvidia-smi", "--version"]),
                             "gpu_identity": command_evidence(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv"])},
             "binaries": {name: {"path": str(args.bin_dir / name), "sha256": sha256(args.bin_dir / name)}
                          for name in ("depth_benchmark", "depth_server")},
             "image": {"path": str(args.image), "sha256": sha256(args.image)}, "packages": identities,
             "method": {"order": "ort-fp32 then trt-fp32 then trt-fp16; raw then sdk then tcp for each",
                        "parallel_cases": False, "repeats": REPEATS, "cpu_threads": 4, "gpu_slots": 1,
                        "sdk_workers": 4, "queue_capacity": 4, "admission_bytes": 134217728,
                        "tcp": "fresh server per case; three serial new client processes; no probe requests",
                        "server_initialization_memory": "separate server_startup_memory.csv/json, not client startup RSS",
                        "gpu_memory": "device 0 total used MiB includes background; not per-process attribution",
                        "startup": "server Popen-to-ready and main-based ready.startup_ms separate from runner client startup",
                        "pass": "all nine runner reports valid; every first/warmup/steady request successful; server drains with zero errors",
                        "timeouts_seconds": {"server_ready": args.ready_timeout_seconds, "whole_case_runner": args.run_timeout_seconds,
                                             "server_shutdown": args.shutdown_timeout_seconds}},
             "cases": []}
    for variant in VARIANTS:
        for mode in MODES:
            result = run_case(args, variant, mode, identities[variant[0]])
            suite["cases"].append(result)
            suite["all_runs_valid"] = len(suite["cases"]) == 9 and all(case["runner_all_runs_valid"] for case in suite["cases"])
            suite["passed"] = len(suite["cases"]) == 9 and all(case["passed"] for case in suite["cases"])
            temporary = args.output / "suite.inprogress.json"
            temporary.write_text(json.dumps(suite, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            print(json.dumps({"case": result["case"], "passed": result["passed"], "errors": result["errors"]}), flush=True)
    json_exclusive(args.output / "suite.json", suite)
    return 0 if suite["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
