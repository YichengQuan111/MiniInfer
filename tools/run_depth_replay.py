#!/usr/bin/env python3
"""Fixed-schedule local SDK replay; Linux/WSL, independent processes per repeat."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time

from run_depth_benchmark import distribution, finite_number as baseline_finite_number, json_exclusive, rss_kib, sha256

MAX_REQUESTS = 1_000_000
STATUSES = ("succeeded", "busy", "closed", "invalid_input", "failure", "timeout",
            "not_submitted_capacity", "not_submitted_late")
STAGES = ("admission_ms", "queue_ms", "preprocess_ms", "backend_ms", "postprocess_ms")
CSV_FIELDS = ("phase", "request_id", "index", "image_index", "planned_ms", "generated_ms",
              "submit_ms", "ready_ms", "delivered_ms", "status", "accepted") + STAGES
CSV_FIELDS_V2 = CSV_FIELDS + ("request_handle_id",)
SCHEMAS = ("miniinfer-depth-replay-v1", "miniinfer-depth-replay-v2")


def finite_number(value, name):
    if isinstance(value, bool):
        raise ValueError(f"invalid {name}: Boolean is not a timing/count")
    return baseline_finite_number(value, name)


def integer_field(value, name, minimum=0, maximum=MAX_REQUESTS):
    if isinstance(value, bool):
        raise ValueError(f"invalid {name}")
    result = int(value)
    if str(result) != str(value) or not minimum <= result <= maximum:
        raise ValueError(f"invalid {name}")
    return result


def protocol_version(summary):
    schema = summary["schema"]
    if schema not in SCHEMAS:
        raise ValueError("summary schema mismatch")
    if schema == SCHEMAS[1]:
        if not isinstance(summary["async_sdk"], bool):
            raise ValueError("async_sdk must be Boolean")
        return 2
    return 1


def summarize_csv(path, summary):
    """Bounded exact percentiles; stream rows and keep only <=1M numeric samples."""
    version = protocol_version(summary)
    asynchronous = version == 2 and summary["async_sdk"]
    handle_ids = set()  # Bounded by the validated <=1M planned requests.
    rate = finite_number(summary["rate"], "rate")
    image_count = integer_field(summary["image_count"], "image_count", 1, 1024)
    max_lag = finite_number(summary["max_lag_ms"], "max_lag_ms")
    if rate <= 0 or rate > 100000 or max_lag <= 0:
        raise ValueError("invalid rate/lag limit")
    planned_total = sum(integer_field(summary[p]["planned"], "planned") for p in ("warmup", "steady"))
    if planned_total > MAX_REQUESTS:
        raise ValueError("too many planned requests")
    states = {}
    offset = 0
    for phase in ("warmup", "steady"):
        spec = summary[phase]
        planned = integer_field(spec["planned"], "planned")
        duration = finite_number(spec["duration_ms"], "duration_ms")
        drain = finite_number(spec["drain_ms"], "drain_ms")
        if (phase == "steady" and duration <= 0) or planned != math.ceil(duration * rate / 1000):
            raise ValueError("planned count/duration mismatch")
        states[phase] = {"seen": bytearray(planned), "offset": offset, "duration": duration,
                         "drain_ms": drain, "counts": dict.fromkeys(STATUSES, 0),
                         "submitted": 0, "accepted": 0, "accepted_succeeded": 0, "accepted_failed": 0,
                         "completed_in_window": 0, "succeeded_in_window": 0, "drain_completed": 0,
                         "delivered_in_window": 0, "delivered_during_drain": 0,
                         "sdk": [], "replay": [], "delivery": [], "lag": [],
                         "stages": {stage: [] for stage in STAGES}, "rows": 0}
        offset += planned
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(CSV_FIELDS_V2 if version == 2 else CSV_FIELDS):
            raise ValueError("CSV schema mismatch")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("CSV column count mismatch")
            phase, status = row["phase"], row["status"]
            if phase not in states or status not in STATUSES:
                raise ValueError("CSV phase/status mismatch")
            state = states[phase]
            index = integer_field(row["index"], "index")
            if index >= len(state["seen"]) or state["seen"][index]:
                raise ValueError("duplicate/out-of-range request")
            state["seen"][index] = 1
            if integer_field(row["request_id"], "request_id", 1) != state["offset"] + index + 1:
                raise ValueError("request ID mismatch")
            if integer_field(row["image_index"], "image_index") != index % image_count:
                raise ValueError("image sequence mismatch")
            planned = finite_number(row["planned_ms"], "planned_ms")
            generated = finite_number(row["generated_ms"], "generated_ms")
            if not math.isclose(planned, index * 1000 / rate, rel_tol=1e-10, abs_tol=1e-8):
                raise ValueError("fixed schedule mismatch")
            if planned >= state["duration"] or generated < planned - 1e-6:
                raise ValueError("invalid generation timestamp")
            state["lag"].append(max(0, generated - planned))
            state["counts"][status] += 1
            state["rows"] += 1
            if status.startswith("not_submitted_"):
                if any(row[key] != "" for key in ("submit_ms", "ready_ms", "delivered_ms", "accepted") + STAGES):
                    raise ValueError("unsubmitted request has execution fields")
                if version == 2 and row["request_handle_id"] != "":
                    raise ValueError("unsubmitted request has a handle ID")
                if status == "not_submitted_late" and generated < state["duration"] and generated - planned <= max_lag:
                    raise ValueError("late status without missed deadline")
                continue
            submit, ready, delivered = (finite_number(row[key], key) for key in ("submit_ms", "ready_ms", "delivered_ms"))
            if not generated <= submit <= ready <= delivered:
                raise ValueError("execution timestamp order")
            if delivered > state["duration"] + state["drain_ms"] + 1e-5:
                raise ValueError("delivery outside drain")
            accepted = integer_field(row["accepted"], "accepted", 0, 1)
            if (status == "succeeded" and not accepted) or (status in ("busy", "closed", "invalid_input") and accepted):
                raise ValueError("invalid acceptance/status relation")
            if version == 2:
                handle = row["request_handle_id"]
                if asynchronous and accepted:
                    identity = integer_field(handle, "request_handle_id", 1, 2**64-1)
                    if identity in handle_ids:
                        raise ValueError("duplicate SDK request handle ID")
                    handle_ids.add(identity)
                elif handle != "":
                    raise ValueError("rejected or synchronous request has a handle ID")
            state["submitted"] += 1
            state["accepted"] += accepted
            if accepted:
                state["accepted_succeeded" if status == "succeeded" else "accepted_failed"] += 1
            if ready < state["duration"]:
                state["completed_in_window"] += 1
                if status == "succeeded":
                    state["succeeded_in_window"] += 1
            else:
                state["drain_completed"] += 1
            if delivered < state["duration"]:
                state["delivered_in_window"] += 1
            else:
                state["delivered_during_drain"] += 1
            timings = {stage: finite_number(row[stage], stage) for stage in STAGES}
            if status == "succeeded":
                # Percentiles cover successful requests planned in this phase,
                # including drain; throughput uses only ready inside the window.
                state["sdk"].append(ready - submit)
                state["replay"].append(ready - planned)
                state["delivery"].append(delivered - ready)
                for stage, value in timings.items():
                    state["stages"][stage].append(value)
    result = {}
    for phase, state in states.items():
        spec = summary[phase]
        if state["rows"] != len(state["seen"]) or state["counts"] != spec["counts"]:
            raise ValueError("CSV planned/terminal counts mismatch")
        for key in ("submitted", "completed_in_window", "succeeded_in_window", "drain_completed"):
            if integer_field(spec[key], key) != state[key]:
                raise ValueError(f"CSV/summary {key} mismatch")
        expected_outstanding = state["delivered_during_drain"] if version == 2 else state["drain_completed"]
        if integer_field(spec["outstanding_at_window_end"], "outstanding_at_window_end") != expected_outstanding:
            raise ValueError("window pending/drain count mismatch")
        total = state["rows"]
        result[phase] = {"planned": total, "terminal_counts": state["counts"], "submitted": state["submitted"],
                         "accepted": state["accepted"], "rejected": state["submitted"] - state["accepted"],
                         "accepted_succeeded": state["accepted_succeeded"], "accepted_failed": state["accepted_failed"],
                         "success_rate_of_planned": state["counts"]["succeeded"] / total if total else None,
                         "window_success_rate_of_planned": state["succeeded_in_window"] / total if total else None,
                         "succeeded_in_window": state["succeeded_in_window"],
                         "successful_throughput_per_second": state["succeeded_in_window"] * 1000 / state["duration"] if state["duration"] else None,
                         "outstanding_at_window_end": expected_outstanding,
                         "sdk_unfinished_at_window_end": state["drain_completed"],
                         "undelivered_at_window_end": state["delivered_during_drain"],
                         "delivered_in_window": state["delivered_in_window"],
                         "delivered_during_drain": state["delivered_during_drain"],
                         "drain_completed": state["drain_completed"],
                         "drain_ms": state["drain_ms"], "sdk_successful_latency": distribution(state["sdk"]),
                         "replay_successful_latency": distribution(state["replay"]), "delivery_successful_latency": distribution(state["delivery"]),
                         "generator_lag": distribution(state["lag"]),
                         "successful_cpu_wall_stages": {key: distribution(values) for key, values in state["stages"].items()}}
    snapshot = summary["sdk_snapshot"]
    for key, expected in (("accepted", sum(s["accepted"] for s in states.values())),
                          ("succeeded", sum(s["accepted_succeeded"] for s in states.values())),
                          ("failed", sum(s["accepted_failed"] for s in states.values())),
                          ("delivered", sum(s["accepted"] for s in states.values())),
                          ("rejected_busy", sum(s["counts"]["busy"] for s in states.values())),
                          ("rejected_closed", sum(s["counts"]["closed"] for s in states.values())),
                          ("rejected_invalid", sum(s["counts"]["invalid_input"] for s in states.values())),
                          ("inflight", 0), ("reserved_bytes", 0)):
        if integer_field(snapshot[key], key, maximum=2**63-1) != expected:
            raise ValueError(f"SDK snapshot {key} conservation mismatch")
    if version == 2:
        for key in ("queued", "executing", "ready", "abandoned", "abandoned_completed"):
            if integer_field(snapshot[key], key) != 0:
                raise ValueError(f"SDK final {key} conservation mismatch")
        if len(handle_ids) != sum(s["accepted"] for s in states.values()) and asynchronous:
            raise ValueError("SDK handle/accepted count mismatch")
    return result


class Protocol:
    """Bounded stdout protocol; raw lines go to files, never an in-memory log."""
    def __init__(self):
        self.ready = None
        self.summary = None
        self.phase = "startup"
        self.order = []
        self.errors = []

    def error(self, message):
        if len(self.errors) < 32:
            self.errors.append(str(message)[:1024])

    def consume(self, line, elapsed):
        try:
            event = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            if not isinstance(event, dict):
                raise ValueError("stdout event must be object")
            kind = event.get("event")
            if kind == "ready" and self.ready is None and not self.order:
                finite_number(event["startup_ms"], "startup_ms")
                self.ready = {"process_to_ready_ms": elapsed * 1000, "event": event}
            elif kind == "phase" and self.ready is not None and self.summary is None:
                phase = event["phase"]
                source = event.get("source")
                token = (phase, source)
                allowed = [("warmup", None), ("drain", "warmup"), ("steady", None), ("drain", "steady")]
                position = len(self.order)
                if self.order == [("warmup", None)] and token == ("steady", None):
                    position = 2  # Zero-duration warmup may omit its drain.
                elif self.order == [("warmup", None), ("steady", None)]:
                    position = 3
                if position >= len(allowed) or token != allowed[position]:
                    raise ValueError("phase events out of order")
                self.order.append(token)
                self.phase = phase
            elif kind == "summary" and self.ready is not None and self.summary is None and self.order and self.order[-1] == ("drain", "steady"):
                protocol_version(event)
                if event["warmup"]["duration_ms"] > 0 and ("drain", "warmup") not in self.order:
                    raise ValueError("warmup drain missing")
                self.summary = event
                self.phase = "teardown"
            else:
                raise ValueError("unknown/duplicate/out-of-order stdout event")
        except (ValueError, KeyError, TypeError, OverflowError, RecursionError) as error:
            self.error(error)


def build_command(args, directory):
    command = [str(args.executable), "--package", str(args.package), "--backend", args.backend,
               "--images-list", str(args.sequence), "--output", str(directory / "raw.csv"),
               "--duration", str(args.duration), "--warmup", str(args.warmup), "--rate", str(args.rate),
               "--sdk-workers", str(args.sdk_workers), "--workers", str(args.workers), "--max-outstanding", str(args.max_outstanding),
               "--queue", str(args.queue), "--budget", str(args.budget), "--max-lag-ms", str(args.max_lag_ms)]
    if args.async_sdk:
        command += ["--async-sdk", "1", "--slots", str(args.slots), "--reuse-host", str(args.reuse_host),
                    "--reuse-gpu", str(args.reuse_gpu), "--consumer-delay-ms", str(args.consumer_delay_ms)]
    return command


def validate_configuration(args, summary):
    version = protocol_version(summary)
    if args.async_sdk and version != 2:
        raise ValueError("async request requires v2 replay protocol")
    if version == 2:
        expected = {"async_sdk": args.async_sdk, "slots": args.slots,
                    "reuse_host": bool(args.reuse_host), "reuse_gpu": bool(args.reuse_gpu),
                    "consumer_delay_ms": args.consumer_delay_ms}
        for key, value in expected.items():
            if key in ("async_sdk", "reuse_host", "reuse_gpu") and not isinstance(summary[key], bool):
                raise ValueError(f"summary {key} must be Boolean")
            if key == "slots":
                integer_field(summary[key], key, 1, 2)
            elif key == "consumer_delay_ms":
                integer_field(summary[key], key, 0, 60000)
            if summary[key] != value:
                raise ValueError(f"requested configuration/summary {key} mismatch")


def run_once(args, directory):
    directory.mkdir()
    command = build_command(args, directory)
    protocol = Protocol()
    events = queue.Queue(maxsize=32)
    start = time.perf_counter()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stop = threading.Event()
    memory_lock = threading.Lock()
    peaks = {}
    telemetry_errors = set()

    def read_stdout():
        with (directory / "stdout.log").open("xb") as log:
            remaining = 1024 * 1024
            while True:
                line = process.stdout.readline(16385)
                if not line:
                    break
                if len(line) > 16384 or not line.endswith(b"\n") or len(line) > remaining:
                    protocol.error("stdout line/log limit exceeded")
                    while line and not line.endswith(b"\n"):
                        line = process.stdout.readline(16385)
                    continue
                remaining -= len(line)
                log.write(line)
                try:
                    events.put((line.decode("utf-8"), time.perf_counter() - start), timeout=1)
                except (UnicodeError, queue.Full):
                    protocol.error("invalid UTF-8 or stdout event queue overflow")

    def read_stderr():
        with (directory / "stderr.log").open("xb") as log:
            remaining = 16 * 1024 * 1024
            while True:
                chunk = process.stderr.read(8192)
                if not chunk:
                    break
                log.write(chunk[:remaining])
                remaining = max(0, remaining - len(chunk))
                if not remaining:
                    protocol.error("stderr log limit exceeded")

    memory_stream = (directory / "memory.csv").open("x", newline="", encoding="utf-8")
    writer = csv.writer(memory_stream)
    writer.writerow(("elapsed_ms", "phase", "role", "metric", "value"))

    def sample(role, metric, value):
        with memory_lock:
            phase = protocol.phase
            writer.writerow(((time.perf_counter() - start) * 1000, phase, role, metric, value))
            for scope in ("whole", phase):
                key = f"{role}:{metric}:{scope}"
                item = peaks.setdefault(key, {"samples": 0, "peak": None})
                if value is not None:
                    item["samples"] += 1
                    item["peak"] = value if item["peak"] is None else max(item["peak"], value)

    def gpu_sample():
        executable = shutil.which("nvidia-smi")
        metrics = ("used_mib", "temperature_c", "sm_clock_mhz", "memory_clock_mhz")
        if not executable:
            telemetry_errors.add("nvidia-smi unavailable")
            for metric in metrics:
                sample(f"gpu_device_{args.device}_total", metric, None)
            return
        while not stop.is_set():
            began = time.perf_counter()
            values = [None] * 4
            try:
                observation = subprocess.run([executable, f"--id={args.device}",
                    "--query-gpu=memory.used,temperature.gpu,clocks.sm,clocks.mem", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=3)
                if observation.returncode:
                    raise ValueError("nvidia-smi query failed")
                fields = observation.stdout.strip().split(",")
                if len(fields) != 4:
                    raise ValueError("nvidia-smi field count")
                for index, field in enumerate(fields):
                    try:
                        values[index] = finite_number(field, metrics[index])
                    except ValueError:
                        telemetry_errors.add(f"nvidia-smi {metrics[index]} unavailable")
            except (OSError, ValueError, subprocess.TimeoutExpired) as error:
                telemetry_errors.add(str(error)[:256])
            for metric, value in zip(metrics, values):
                sample(f"gpu_device_{args.device}_total", metric, value)
            stop.wait(max(0, .2 - (time.perf_counter() - began)))

    threads = [threading.Thread(target=read_stdout), threading.Thread(target=read_stderr), threading.Thread(target=gpu_sample)]
    for thread in threads:
        thread.start()

    def consume():
        while True:
            try:
                line, elapsed = events.get_nowait()
            except queue.Empty:
                break
            protocol.consume(line, elapsed)

    try:
        while True:
            consume()
            sample("replay", "rss_kib", rss_kib(process.pid))
            if process.poll() is not None:
                break
            if time.perf_counter() - start > args.process_timeout_seconds:
                protocol.error("whole process deadline exceeded; incomplete rows are invalid, never inferred")
                process.kill()
                process.wait(timeout=5)
                break
            stop.wait(.05)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        stop.set()
        for thread in threads:
            thread.join(timeout=5)
            if thread.is_alive():
                protocol.error("reader/sampler failed to join")
        consume()
        process.stdout.close()
        process.stderr.close()
        memory_stream.close()
    if process.returncode:
        protocol.error(f"process exited {process.returncode}")
    if protocol.ready is None or protocol.summary is None:
        protocol.error("ready/summary event missing")
    statistics = None
    if protocol.summary is not None:
        try:
            summary = protocol.summary
            validate_configuration(args, summary)
            if (summary["rate"] != args.rate or summary["image_count"] != len(args.images)
                or summary["max_lag_ms"] != args.max_lag_ms
                or summary["steady"]["duration_ms"] != args.duration * 1000
                or summary["warmup"]["duration_ms"] != args.warmup * 1000):
                raise ValueError("requested configuration/summary mismatch")
            if args.warmup and ("drain", "warmup") not in protocol.order:
                raise ValueError("warmup drain missing")
            statistics = summarize_csv(directory / "raw.csv", summary)
            for phase in ("warmup", "steady"):
                if summary[phase]["peak_outstanding"] > args.max_outstanding:
                    raise ValueError("client outstanding bound exceeded")
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
            protocol.error(error)
    result = {"command": command, "pid": process.pid, "returncode": process.returncode,
              "ready": protocol.ready, "process_summary": protocol.summary, "statistics": statistics,
              "memory_peaks": peaks, "telemetry_errors": sorted(telemetry_errors), "errors": protocol.errors,
              "stdout": "stdout.log", "stderr": "stderr.log", "memory_csv": "memory.csv", "request_csv": "raw.csv"}
    json_exclusive(directory / "summary.json", result)
    return result


def positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def finite_positive(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def finite_nonnegative(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--executable", type=Path, required=True)
    result.add_argument("--package", type=Path, required=True)
    result.add_argument("--backend", choices=("ort", "trt"), required=True)
    source = result.add_mutually_exclusive_group(required=True)
    source.add_argument("--images", type=Path, nargs="+")
    source.add_argument("--images-list", type=Path)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--duration", type=finite_positive, default=10.)
    result.add_argument("--warmup", type=finite_nonnegative, default=2., help="seconds; drained before steady")
    result.add_argument("--rate", type=finite_positive, required=True)
    result.add_argument("--sdk-workers", type=positive, default=4, help="SDK CPU workers, independent from replay callers")
    result.add_argument("--workers", type=positive, default=4)
    result.add_argument("--max-outstanding", type=positive, default=4)
    result.add_argument("--queue", type=positive, default=4)
    result.add_argument("--budget", type=positive, default=128 * 1024 * 1024, help="SDK admission bytes, excludes image sequence/model")
    result.add_argument("--max-lag-ms", type=finite_positive, default=100.)
    result.add_argument("--repeats", type=positive, default=3)
    result.add_argument("--async-sdk", action="store_true", help="submit on schedule; SDK ready_at separates execution from consumption")
    result.add_argument("--slots", type=int, choices=(1, 2), default=1)
    result.add_argument("--reuse-host", type=int, choices=(0, 1), default=1)
    result.add_argument("--reuse-gpu", type=int, choices=(0, 1), default=1)
    result.add_argument("--consumer-delay-ms", type=int, default=0, help="0..60000; delay after wait, before get; async only")
    result.add_argument("--device", type=int, choices=(0,), default=0, help="V1 SDK uses device 0")
    result.add_argument("--process-timeout-seconds", type=finite_positive, default=3600.)
    result.add_argument("--environment-note", default="", help="power/background load/build type and external profiler conditions")
    return result


def validate_args(args):
    if not 0 <= args.consumer_delay_ms <= 60000:
        raise ValueError("consumer delay must be in 0..60000 ms")
    if not args.async_sdk and (args.slots != 1 or args.reuse_host != 1 or args.reuse_gpu != 1 or args.consumer_delay_ms != 0):
        raise ValueError("nondefault slots/reuse/consumer delay require --async-sdk")
    if args.backend == "ort" and args.slots != 1:
        raise ValueError("ORT supports one SDK slot")
    if (args.duration > 86400 or args.warmup > 3600 or args.rate > 100000 or args.sdk_workers > 256 or args.workers > 256
        or args.max_outstanding > 100000 or args.queue > 100000 or args.repeats > 100
        or math.ceil(args.duration * args.rate) + math.ceil(args.warmup * args.rate) > MAX_REQUESTS):
        raise ValueError("replay exceeds bounded duration/rate/threads/requests/repeats")
    if args.process_timeout_seconds <= args.duration + args.warmup:
        raise ValueError("process timeout must leave time for startup and drain")
    if args.process_timeout_seconds > 172800:
        raise ValueError("process timeout exceeds two days")
    if args.images_list:
        source = args.images_list.resolve()
        if source.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("image list exceeds 8MiB")
        args.images = []
        with source.open(encoding="utf-8") as stream:
            for line in stream:
                if len(line) > 8192:
                    raise ValueError("image list line exceeds 8192 characters")
                value = line.rstrip("\r\n")
                if value and not value.startswith("#"):
                    path = Path(value)
                    args.images.append((source.parent / path).resolve() if not path.is_absolute() else path.resolve())
                    if len(args.images) > 1024:
                        raise ValueError("at most 1024 images")
    else:
        args.images = [path.resolve() for path in args.images]
    if not args.images or len(args.images) > 1024:
        raise ValueError("1..1024 image paths required")
    for key in ("executable", "package", "output"):
        setattr(args, key, getattr(args, key).resolve())
    if not args.executable.is_file() or not args.package.is_dir() or any(not path.is_file() for path in args.images):
        raise ValueError("executable/package/images must exist")
    if any("\n" in str(path) or "\r" in str(path) for path in args.images):
        raise ValueError("image paths cannot contain newlines")


def environment():
    result = {"platform": os.uname()._asdict() if hasattr(os.uname(), "_asdict") else list(os.uname()),
              "python": os.sys.version, "environment": {key: os.environ.get(key) for key in
                  ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "LD_LIBRARY_PATH")}}
    for name, command in (("gpu", ["nvidia-smi"]), ("git", ["git", "rev-parse", "HEAD"]),
                          ("git_status", ["git", "status", "--porcelain"]),
                          ("compiler", ["c++", "--version"]), ("cmake", ["cmake", "--version"])):
        try:
            item = subprocess.run(command, capture_output=True, text=True, timeout=5)
            result[name] = {"command": command, "returncode": item.returncode, "stdout": item.stdout[:8192], "stderr": item.stderr[:1024]}
        except (OSError, subprocess.TimeoutExpired) as error:
            result[name] = {"command": command, "error": str(error)}
    return result


def main():
    cli = parser()
    args = cli.parse_args()
    try:
        validate_args(args)
    except (OSError, ValueError) as error:
        cli.error(str(error))
    if os.name != "posix" or not Path("/proc/self/status").exists():
        cli.error("Linux/WSL with /proc required")
    args.output.mkdir()
    args.sequence = args.output / "sequence.txt"
    args.sequence.write_text("".join(str(path) + "\n" for path in args.images), encoding="utf-8")
    package_identity = []
    for path in sorted(args.package.rglob("*")):
        if path.is_file():
            package_identity.append({"path": str(path.relative_to(args.package)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    report = {"schema": "miniinfer-depth-replay-report-v2", "recorded_at": datetime.now(timezone.utc).isoformat(),
              "case": {key: getattr(args, key) for key in ("backend", "duration", "warmup", "rate", "sdk_workers", "workers", "max_outstanding", "queue", "budget", "max_lag_ms", "repeats", "device", "process_timeout_seconds", "environment_note", "async_sdk", "slots", "reuse_host", "reuse_gpu", "consumer_delay_ms")},
              "identity": {"executable": str(args.executable), "executable_sha256": sha256(args.executable),
                           "package": str(args.package), "package_files": package_identity,
                           "images": [{"path": str(path), "sha256": sha256(path)} for path in args.images], "sequence_sha256": sha256(args.sequence)},
              "environment": environment(),
              "measurement": {"schedule": "phase-relative index/rate; fixed image order, decoded before ready",
                              "unsubmitted": "explicit capacity or generator-late; lateness beyond max-lag or phase end never catches up silently",
                              "latency": ("successful requests planned in phase, including drain; async ready is actual SDK ready_at publication after synchronized execution and owning restoration; delivered follows get/consumption, including configured consumer delay"
                                          if args.async_sdk else "successful requests planned in phase, including drain; synchronous ready observed immediately at run return while output is alive; delivered after output consumption/destruction"),
                              "window_end": "SDK unfinished uses ready timestamp; client undelivered uses delivered timestamp; v1 outstanding_at_window_end retains its historical ready-based definition",
                              "acceptance": "DepthTiming; backend/admission failure distinguished; absolute admission/dequeue not exposed",
                              "stages": "CPU wall; backend wall is not CUDA Event device compute time",
                              "upload_bytes": "ready event lists RGB8 source sizes and FP32 model input bytes; phase h2d_bytes/fp32_h2d_bytes count actual transfers for successful backend completions; rgb8_h2d_bytes is zero in the CPU preprocessing baseline; CPU backend transfers are zero",
                              "throughput": "success ready in [0,duration) / full duration; drain count and time separate",
                              "success_rate": "all planned requests denominator; eventual and in-window rates separate",
                              "rss": "process VmRSS sampled about every 50ms; excludes sampler process",
                              "gpu": "device total, includes background load; target 200ms; unavailable fields null; temperatures and clocks included",
                              "sampling_phase": "stdout reception boundary, pipe/scheduling delay possible",
                              "bounds": "<=1M requests, <=1024 images, <=256MiB decoded; stream CSV/logs/telemetry; bounded numeric samples for exact linear percentiles",
                              "repeats": "independent SDK process each time; do not average P99"}, "runs": []}
    for repeat in range(args.repeats):
        result = run_once(args, args.output / f"repeat_{repeat + 1:02d}")
        report["runs"].append(result)
        print(json.dumps({"repeat": repeat + 1, "errors": result["errors"], "statistics": result["statistics"]}, allow_nan=False), flush=True)
    report["all_runs_valid"] = all(not run["errors"] for run in report["runs"])
    json_exclusive(args.output / "report.json", report)
    return 0 if report["all_runs_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
