#!/usr/bin/env python3
"""Sequential real DepthSdk closed-loop concurrency cases, each in a fresh process."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess


STATUSES = ("succeeded", "busy", "closed", "failure")
CSV_FIELDS = ("index", "caller", "status", "wall_ms", "checksum", "error")


def percentile(values, q):
    """Linear interpolation over all successful requests; never average callers."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def finite(value, name, positive=False):
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f"invalid {name}")
    return result


def summarize_csv(path, summary):
    """Independently recompute summary; overlapping request times must not be summed."""
    iterations, concurrency = summary["iterations"], summary["concurrency"]
    if not isinstance(iterations, int) or not isinstance(concurrency, int) or not 1 <= concurrency <= iterations:
        raise ValueError("invalid planned request/concurrency count")
    elapsed = finite(summary["elapsed_steady_ms"], "elapsed_steady_ms", positive=True)
    counts = dict.fromkeys(STATUSES, 0)
    caller_counts = [0] * concurrency
    latencies = []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(CSV_FIELDS):
            raise ValueError("CSV schema mismatch")
        rows = list(reader)
    if len(rows) != iterations:
        raise ValueError("CSV total attempt count mismatch")
    for index, row in enumerate(rows):
        if None in row or any(value is None for value in row.values()):
            raise ValueError("CSV column count mismatch")
        if row["index"] != str(index) or row["status"] not in counts:
            raise ValueError("CSV index/status mismatch")
        caller = int(row["caller"])
        if not 0 <= caller < concurrency:
            raise ValueError("CSV caller outside concurrency")
        caller_counts[caller] += 1
        status = row["status"]
        counts[status] += 1
        wall = finite(row["wall_ms"], "wall_ms")
        if wall > elapsed + 1e-6:
            raise ValueError("single request exceeds steady wall interval")
        if status == "succeeded":
            if row["checksum"] != str(summary["baseline_checksum"]) or row["error"]:
                raise ValueError("successful output checksum/error mismatch")
            latencies.append(wall)
    if counts != summary["counts"]:
        raise ValueError("CSV/summary counts mismatch")
    result = {"counts": counts, "p50_ms": percentile(latencies, 50),
              "p99_ms": percentile(latencies, 99),
              "successful_qps": len(latencies) * 1000 / elapsed,
              "elapsed_steady_ms": elapsed, "caller_attempt_counts": caller_counts}
    for key in ("p50_ms", "p99_ms", "successful_qps"):
        expected, actual = result[key], summary[key]
        if expected is None:
            if actual is not None:
                raise ValueError(f"nonempty {key} for zero successes")
        elif not math.isclose(expected, finite(actual, key), rel_tol=1e-10, abs_tol=1e-8):
            raise ValueError(f"CSV/summary {key} mismatch")
    if any(summary.get(key) is not True for key in ("ownership_ok", "counters_ok", "drained")):
        raise ValueError("ownership/counter/drain validation failed")
    snapshot = summary["sdk_snapshot"]
    if any(snapshot[key] != 0 for key in ("inflight", "queued", "executing", "reserved_bytes")):
        raise ValueError("SDK still holds requests/reservations")
    return result


def metric_ranges(runs):
    """Ranges of independently measured repeats, not pooled or averaged percentiles."""
    result = {}
    for key in ("p50_ms", "p99_ms", "successful_qps", "elapsed_steady_ms"):
        values = [run["statistics"][key] for run in runs
                  if not run["errors"] and run["statistics"][key] is not None]
        result[key] = {"repeats": len(values), "min": min(values) if values else None,
                       "median": statistics.median(values) if values else None,
                       "max": max(values) if values else None}
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


def command_for(args, concurrency, directory):
    command = [str(args.executable), "--backend", args.backend, "--package", str(args.package),
               "--image", str(args.image), "--csv", str(directory / "raw.csv"),
               "--summary", str(directory / "summary.json")]
    for name, value in (("concurrency", concurrency), ("iterations", args.iterations),
                        ("warmup", args.warmup), ("worker-count", args.worker_count),
                        ("ort-threads", args.ort_threads), ("queue-capacity", args.queue_capacity),
                        ("byte-budget", args.byte_budget)):
        command.extend((f"--{name}", str(value)))
    return command


def run_once(args, concurrency, directory):
    directory.mkdir()
    command = command_for(args, concurrency, directory)
    errors, summary, computed = [], None, None
    returncode = None
    stdout = stderr = ""
    try:
        process = subprocess.run(command, capture_output=True, text=True,
                                 timeout=args.process_timeout_seconds)
        returncode, stdout, stderr = process.returncode, process.stdout, process.stderr
        if returncode != 0:
            errors.append(f"benchmark process exited {returncode}")
    except subprocess.TimeoutExpired as error:
        # subprocess.run kills and waits for the child before raising.
        errors.append("whole process timeout; killed and reaped")
        stdout = (error.stdout or b"").decode(errors="replace") if isinstance(error.stdout, bytes) else (error.stdout or "")
        stderr = (error.stderr or b"").decode(errors="replace") if isinstance(error.stderr, bytes) else (error.stderr or "")
    except OSError as error:
        errors.append(str(error))
    (directory / "stdout.txt").write_text(stdout, encoding="utf-8")
    (directory / "stderr.txt").write_text(stderr, encoding="utf-8")
    try:
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        expected = {"backend": args.backend, "concurrency": concurrency, "iterations": args.iterations,
                    "warmup": args.warmup, "worker_count": args.worker_count, "ort_threads": args.ort_threads,
                    "queue_capacity": args.queue_capacity, "byte_budget": args.byte_budget}
        if any(summary[key] != value for key, value in expected.items()):
            raise ValueError("actual benchmark configuration differs from requested configuration")
        computed = summarize_csv(directory / "raw.csv", summary)
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(str(error))
    result = {"command": command, "returncode": returncode, "errors": errors,
              "summary": summary, "statistics": computed, "directory": str(directory)}
    json_exclusive(directory / "verified.json", result)
    return result


def positive(value):
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def nonnegative(value):
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a nonnegative integer") from error
    if result < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--backend", choices=("ort", "trt"), required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new evidence directory; existing directory refused")
    parser.add_argument("--concurrencies", type=positive, nargs="+", default=[1, 4, 8])
    parser.add_argument("--iterations", type=positive, default=1000, help="total steady attempts per repeat, not per caller")
    parser.add_argument("--warmup", type=nonnegative, default=20, help="sequential SDK warmup calls after one baseline")
    parser.add_argument("--repeats", type=positive, default=3)
    parser.add_argument("--worker-count", type=positive, default=4)
    parser.add_argument("--ort-threads", type=positive, default=4)
    parser.add_argument("--queue-capacity", type=positive, default=4)
    parser.add_argument("--byte-budget", type=positive, default=512 * 1024 * 1024)
    parser.add_argument("--process-timeout-seconds", type=positive, default=7200)
    args = parser.parse_args()
    if len(set(args.concurrencies)) != len(args.concurrencies) or max(args.concurrencies) > min(256, args.iterations):
        parser.error("concurrencies must be distinct and <= min(256, iterations)")
    for key in ("executable", "package", "image", "output"):
        setattr(args, key, getattr(args, key).resolve())
    args.output.mkdir()  # Exclusive root: all child files belong to this new run.
    report = {"schema": "miniinfer-depth-concurrency-suite-v1",
              "recorded_at": datetime.now(timezone.utc).isoformat(),
              "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "identity": {"executable_sha256": sha256(args.executable), "image_sha256": sha256(args.image),
                           "package_files_sha256": {str(path.relative_to(args.package)): sha256(path)
                                                    for path in sorted(args.package.rglob("*")) if path.is_file()}},
              "measurement": {"load": "closed-loop synchronous callers sharing one SDK; rejects retained without retry",
                              "iterations": "total steady attempts per repeat",
                              "qps": "successful count / entire steady wall seconds, including output validation overhead",
                              "latency": "SDK run only, including copy, admission, queue, pre/postprocess and backend",
                              "percentiles": "linear interpolation, successful requests across all callers",
                              "repeat_ranges": "min/median/max of separately measured repeats; percentiles are not pooled or averaged"},
              "cases": []}
    for concurrency in args.concurrencies:
        case = {"concurrency": concurrency, "runs": []}
        for repeat in range(1, args.repeats + 1):
            result = run_once(args, concurrency, args.output / f"c{concurrency}_repeat{repeat:02d}")
            case["runs"].append(result)
            print(json.dumps({"concurrency": concurrency, "repeat": repeat, "errors": result["errors"],
                              "statistics": result["statistics"]}, allow_nan=False), flush=True)
        case["repeat_ranges"] = metric_ranges(case["runs"])
        report["cases"].append(case)
        print(json.dumps({"concurrency": concurrency, "repeat_ranges": case["repeat_ranges"]}, allow_nan=False), flush=True)
    runs = [run for case in report["cases"] for run in case["runs"]]
    report["all_runs_valid"] = all(not run["errors"] for run in runs)
    report["all_requests_successful"] = all(run["statistics"] is not None and
        run["statistics"]["counts"]["succeeded"] == args.iterations for run in runs)
    json_exclusive(args.output / "report.json", report)
    return 0 if report["all_runs_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
