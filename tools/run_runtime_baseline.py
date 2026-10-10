"""Run isolated direct-CPU and runtime benchmark measurements.

Each measurement is a fresh benchmark_cli process. The driver uses only the
Python standard library and writes a report only when the requested output
path does not already exist.
"""

import argparse
import hashlib
import json
import math
import os
import pathlib
import platform
import re
import shlex
import subprocess
import tempfile
from datetime import datetime, timezone


INPUT_ROWS = 64
INPUT_FEATURES = 64
FLOAT32_BYTES = 4
BATCHES = (1, 8, 32)
CONFIGURATIONS = (
    ("direct_cpu", 0),
    ("runtime", 1),
    ("runtime", 2),
    ("runtime", 4),
)
TIMING_GROUPS = ("queue", "execution", "completion", "collection", "submit")
TIMING_PERCENTILES = ("mean_ms", "p50_ms", "p95_ms", "p99_ms")
SCRIPT_VERSION = "1"


def sha256_file(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_json_constant(value):
    raise ValueError(f"non-finite JSON constant {value!r}")


def _check_finite_tree(value, location="result"):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{location} contains a non-finite number")
    if isinstance(value, dict):
        for key, child in value.items():
            _check_finite_tree(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _check_finite_tree(child, f"{location}[{index}]")


def validate_result(result, expected):
    """Validate one benchmark_cli JSON object against the D14 contract."""
    if not isinstance(result, dict):
        raise ValueError("benchmark output must be a JSON object")
    _check_finite_tree(result)

    required = {
        "mode", "workers", "batch", "iterations", "warmup", "window",
        "queue_capacity", "attempts", "accepted", "successes", "failures",
        "rejected", "queue_full_rejected", "closed_rejected", "wall_ms",
        "req_per_sec", "samples_per_sec", "mean_ms", "p50_ms", "p95_ms",
        "p99_ms", "checksum", "rss_after_warmup_bytes",
        "rss_after_timing_bytes", "peak_rss_bytes", "max_inflight",
    }
    required.update(
        f"{group}_{percentile}"
        for group in TIMING_GROUPS
        for percentile in TIMING_PERCENTILES
    )
    missing = sorted(required.difference(result))
    if missing:
        raise ValueError("benchmark output is missing fields: " + ", ".join(missing))

    for name, value in expected.items():
        actual = result.get(name)
        if type(actual) is not type(value) or actual != value:
            raise ValueError(f"benchmark field {name!r} is {actual!r}, expected {value!r}")

    integer_fields = (
        "workers", "batch", "iterations", "warmup", "window", "queue_capacity",
        "attempts", "accepted", "successes", "failures", "rejected",
        "queue_full_rejected", "closed_rejected", "max_inflight",
    )
    for name in integer_fields:
        value = result[name]
        if type(value) is not int or value < 0:
            raise ValueError(f"benchmark field {name!r} must be a nonnegative integer")

    numeric_fields = {
        "wall_ms", "req_per_sec", "samples_per_sec", "mean_ms", "p50_ms",
        "p95_ms", "p99_ms", "rss_after_warmup_bytes",
        "rss_after_timing_bytes", "peak_rss_bytes",
    }
    numeric_fields.update(
        f"{group}_{percentile}"
        for group in TIMING_GROUPS
        for percentile in TIMING_PERCENTILES
    )
    numeric_fields.add("checksum")
    for name in numeric_fields:
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"benchmark field {name!r} must be numeric")
        if not math.isfinite(value):
            raise ValueError(f"benchmark field {name!r} must be finite")
        if name != "checksum" and value < 0:
            raise ValueError(f"benchmark field {name!r} must be nonnegative")

    if result["attempts"] != result["accepted"] + result["rejected"]:
        raise ValueError("attempts must equal accepted + rejected")
    if result["accepted"] != result["successes"] + result["failures"]:
        raise ValueError("accepted must equal successes + failures")
    if result["rejected"] != result["queue_full_rejected"] + result["closed_rejected"]:
        raise ValueError("rejected must equal queue_full_rejected + closed_rejected")
    if result["iterations"] != result["attempts"]:
        raise ValueError("iterations must equal attempts")
    if result["failures"] != 0:
        raise ValueError("benchmark reported failed accepted requests")
    if result["successes"] <= 0:
        raise ValueError("benchmark reported no successful requests")
    if result["max_inflight"] > result["window"]:
        raise ValueError("max_inflight exceeds window")
    if result["max_inflight"] < 1:
        raise ValueError("max_inflight must be positive when requests succeed")

    if result["mode"] == "direct_cpu":
        if result["workers"] != 0:
            raise ValueError("direct_cpu mode must use workers=0")
        if result["max_inflight"] != 1:
            raise ValueError("direct_cpu mode must report max_inflight=1")
        for name in (f"queue_{p}" for p in TIMING_PERCENTILES):
            if result[name] != 0:
                raise ValueError(f"direct_cpu mode must report {name}=0")
        for name in (f"collection_{p}" for p in TIMING_PERCENTILES):
            if result[name] != 0:
                raise ValueError(f"direct_cpu mode must report {name}=0")
    elif result["mode"] != "runtime":
        raise ValueError("mode must be 'direct_cpu' or 'runtime'")
    return result


def parse_result(stdout, expected):
    try:
        result = json.loads(stdout, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"benchmark stdout is not valid finite JSON: {error}") from error
    return validate_result(result, expected)


def _read_environment_file(path):
    try:
        return pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        return {"unavailable": str(error)}


def _lscpu_info():
    try:
        completed = subprocess.run(
            ["lscpu", "--json"], capture_output=True, text=True, check=False, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"unavailable": str(error)}
    if completed.returncode != 0:
        return {"unavailable": completed.stderr.strip() or f"exit {completed.returncode}"}
    try:
        parsed = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        return {"unavailable": f"lscpu returned invalid JSON: {error}"}
    return parsed


def _compile_configuration(cli_path):
    database = cli_path.resolve().parent.parent / "compile_commands.json"
    if not database.is_file():
        return {"available": False, "compile_commands": str(database), "flags": {}}
    try:
        commands = json.loads(database.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"available": False, "compile_commands": str(database),
                "error": str(error), "flags": {}}

    wanted = {"benchmark_cli.cpp", "cpu_ops.cpp", "runtime.cpp", "thread_pool.cpp"}
    flags = {}
    full_commands = {}
    for entry in commands:
        source = str(entry.get("file", ""))
        normalized = source.replace("\\", "/")
        basename = normalized.rsplit("/", 1)[-1]
        if basename not in wanted:
            continue
        if "arguments" in entry:
            argv = entry["arguments"]
        else:
            argv = shlex.split(entry.get("command", ""))
        flags[source] = [
            token for token in argv
            if token.startswith(("-O", "-D", "-std=", "-f", "-W"))
        ]
        full_commands[source] = argv
    return {
        "available": bool(flags),
        "compile_commands": str(database),
        "flags": flags,
        "commands": full_commands,
    }


def _project_version(cli_path):
    candidates = []
    resolved_cli = cli_path.resolve()
    candidates.extend(parent / "CMakeLists.txt" for parent in resolved_cli.parents)

    database = resolved_cli.parent.parent / "compile_commands.json"
    try:
        commands = json.loads(database.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        commands = []
    for entry in commands:
        source_text = entry.get("file")
        if not source_text:
            continue
        source = pathlib.Path(source_text)
        if not source.is_absolute():
            source = pathlib.Path(entry.get("directory", ".")) / source
        candidates.extend(parent / "CMakeLists.txt" for parent in source.resolve().parents)

    pattern = re.compile(r"project\s*\(\s*MiniInfer\s+VERSION\s+([^\s)]+)", re.IGNORECASE)
    for cmake_file in candidates:
        try:
            match = pattern.search(cmake_file.read_text(encoding="utf-8"))
        except OSError:
            continue
        if match:
            return match.group(1)
    return None


def collect_environment(cli_path):
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    try:
        load_average = list(os.getloadavg())
    except (AttributeError, OSError):
        load_average = None

    mem_total = None
    try:
        for line in pathlib.Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                mem_total = line.strip()
                break
    except OSError:
        pass

    binary_hash = sha256_file(cli_path) if cli_path.is_file() else None
    return {
        "os_release": _read_environment_file("/etc/os-release"),
        "kernel": platform.release(),
        "kernel_version": platform.version(),
        "lscpu": _lscpu_info(),
        "memory_total": mem_total,
        "cpu_affinity": affinity,
        "observed_load_average_1_5_15": load_average,
        "compile_configuration": _compile_configuration(cli_path),
        "tool_versions": {
            "driver": SCRIPT_VERSION,
            "python": platform.python_version(),
            "miniinfer_project": _project_version(cli_path),
            "benchmark_cli_sha256": binary_hash,
        },
        "control_notes": (
            "CPU affinity was recorded, but no affinity pinning, background-load "
            "isolation, frequency-governor, or power-state control was applied; "
            "system load and power state were uncontrolled."
        ),
    }


def build_command(cli, reference, input_file, batch, args, mode, workers):
    return [
        str(cli), "--model", str(reference), "--input", str(input_file),
        "--batch", str(batch), "--iterations", str(args.iterations),
        "--warmup", str(args.warmup), "--mode", mode, "--workers", str(max(workers, 1)),
        "--window", str(args.window), "--queue-capacity", str(args.queue_capacity),
    ]


def execute_one(command, expected):
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=60
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"benchmark timed out after 60 seconds: {command!r}") from error
    if completed.returncode != 0:
        raise RuntimeError(
            f"benchmark exited with status {completed.returncode}: {command!r}\n"
            f"stdout: {completed.stdout}\nstderr: {completed.stderr}"
        )
    if completed.stderr:
        raise RuntimeError(f"benchmark emitted stderr: {completed.stderr}")
    result = parse_result(completed.stdout, expected)
    return result, completed.stdout


def _check_batch_consistency(runs):
    by_batch = {}
    for run in runs:
        result = run["result"]
        baseline = by_batch.get(run["batch"])
        if baseline is None:
            by_batch[run["batch"]] = result
            continue
        if not math.isclose(result["checksum"] / result["successes"],
                            baseline["checksum"] / baseline["successes"],
                            rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"batch {run['batch']} has inconsistent checksums")


def _model_hashes(reference):
    model_files = ("w1.f32", "b1.f32", "w2.f32", "b2.f32")
    hashes = {name: sha256_file(reference / name) for name in model_files}
    return hashes


def _write_report(output_path, report):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents overwriting a file created during collection.
    with output_path.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=pathlib.Path, required=True)
    parser.add_argument("--reference", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--queue-capacity", type=int, default=64)
    args = parser.parse_args(argv)

    if args.iterations < 1 or args.repeats < 1:
        parser.error("iterations and repeats must be positive")
    if args.warmup < 0:
        parser.error("warmup must be nonnegative")
    if args.window < 1 or args.queue_capacity < 1:
        parser.error("window and queue-capacity must be positive")
    if os.path.lexists(args.output):
        parser.error("output exists; choose a new output path")

    cli = args.cli.resolve()
    reference = args.reference.resolve()
    if not cli.is_file():
        parser.error(f"CLI is not a file: {cli}")
    if not reference.is_dir():
        parser.error(f"reference is not a directory: {reference}")

    try:
        inputs = (reference / "inputs.f32").read_bytes()
    except OSError as error:
        parser.error(f"cannot read reference inputs: {error}")
    expected_input_bytes = INPUT_ROWS * INPUT_FEATURES * FLOAT32_BYTES
    if len(inputs) != expected_input_bytes:
        parser.error(f"expected {expected_input_bytes} bytes in reference inputs.f32")
    try:
        model_sha256 = _model_hashes(reference)
    except OSError as error:
        parser.error(f"cannot read model files: {error}")
    manifest = reference / "manifest.json"
    manifest_sha256 = sha256_file(manifest) if manifest.is_file() else None

    report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": collect_environment(cli),
        "cli": str(cli),
        "reference": str(reference),
        "model_sha256": model_sha256,
        "input_sha256": sha256_file(reference / "inputs.f32"),
        "manifest_sha256": manifest_sha256,
        "workload": {
            "batches": list(BATCHES),
            "input_rows": "first B rows of reference inputs.f32; each row has 64 little-endian FP32 values",
            "processes": "one fresh benchmark_cli process per batch, repeat, and mode/worker configuration",
            "ordering": "for each batch and repeat: direct_cpu, runtime workers 1, 2, then 4",
            "checksum_tolerance": {"relative": 1e-6, "absolute": 1e-6},
        },
        "parameters": {
            "iterations": args.iterations,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "window": args.window,
            "queue_capacity": args.queue_capacity,
        },
        "runs": [],
    }

    with tempfile.TemporaryDirectory(prefix="miniinfer_d14_") as temporary:
        temp_dir = pathlib.Path(temporary)
        for batch in BATCHES:
            input_file = temp_dir / f"input_{batch}.f32"
            row_bytes = INPUT_FEATURES * FLOAT32_BYTES
            input_file.write_bytes(inputs[:batch * row_bytes])
            for repeat in range(1, args.repeats + 1):
                for mode, workers in CONFIGURATIONS:
                    expected = {
                        "mode": mode,
                        "workers": workers,
                        "batch": batch,
                        "iterations": args.iterations,
                        "warmup": args.warmup,
                        "window": args.window,
                        "queue_capacity": args.queue_capacity,
                    }
                    command = build_command(
                        cli, reference, input_file, batch, args, mode, workers
                    )
                    result, raw_json = execute_one(command, expected)
                    report["runs"].append({
                        "batch": batch,
                        "repeat": repeat,
                        "mode": mode,
                        "workers": workers,
                        "command": command,
                        "result": result,
                        "raw_json": raw_json,
                    })

    _check_batch_consistency(report["runs"])
    expected_runs = len(BATCHES) * args.repeats * len(CONFIGURATIONS)
    if len(report["runs"]) != expected_runs:
        raise RuntimeError(f"collected {len(report['runs'])} runs; expected {expected_runs}")
    _write_report(args.output, report)
    print(f"Saved {len(report['runs'])} independent-process measurements to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
