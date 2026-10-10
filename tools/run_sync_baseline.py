"""Run D8 direct-CPU measurements in separate processes; Python standard library only."""

import argparse
import hashlib
import json
import os
import pathlib
import platform
import shlex
import subprocess
import tempfile
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=pathlib.Path, required=True)
    parser.add_argument("--reference", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0 or args.repeats < 1:
        parser.error("iterations/repeats must be positive and warmup nonnegative")
    if args.output.exists():
        parser.error("output exists; choose a new output path")

    reference = args.reference.resolve()
    inputs = (reference / "inputs.f32").read_bytes()
    if len(inputs) != 64 * 64 * 4:
        parser.error("expected the official 64-row FP32 reference inputs")
    cli = str(args.cli.resolve())
    compile_database = args.cli.resolve().parent.parent / "compile_commands.json"
    compile_flags = None
    if compile_database.exists():
        commands = json.loads(compile_database.read_text())
        cpu_ops = next(entry for entry in commands if entry["file"].endswith("/src/cpu_ops.cpp"))
        compile_flags = [flag for flag in shlex.split(cpu_ops["command"])
                         if flag.startswith(("-O", "-D", "-std=", "-f", "-W"))]
    cpu_info = json.loads(subprocess.check_output(["lscpu", "--json"], text=True))
    wanted = {"Architecture:", "CPU(s):", "Model name:", "Thread(s) per core:",
              "Core(s) per socket:", "Socket(s):", "Hypervisor vendor:"}
    cpu = {row["field"].rstrip(":"): row["data"] for row in cpu_info["lscpu"]
           if row["field"] in wanted}
    mem_total = next(line.strip() for line in pathlib.Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemTotal:"))
    files = ["w1.f32", "b1.f32", "w2.f32", "b2.f32", "inputs.f32", "manifest.json"]
    report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"os_release": pathlib.Path("/etc/os-release").read_text(),
                        "kernel": platform.release(), "cpu": cpu, "memory": mem_total,
                        "cpu_affinity": sorted(os.sched_getaffinity(0)),
                        "cpu_ops_compile_flags": compile_flags,
                        "control_notes": "No pinning, power/frequency or background-load control."},
        "cli": cli,
        "reference_sha256": {name: hashlib.sha256((reference / name).read_bytes()).hexdigest()
                             for name in files},
        "workload": "Repeat the first batch rows of official normalized inputs; no request queue.",
        "runs": [],
    }
    with tempfile.TemporaryDirectory(prefix="miniinfer_d8_") as temporary:
        for batch in (1, 8, 32):
            input_file = pathlib.Path(temporary) / f"input_{batch}.f32"
            input_file.write_bytes(inputs[:batch * 64 * 4])
            for repeat in range(args.repeats):
                command = [cli, "--model", str(reference), "--input", str(input_file),
                           "--batch", str(batch), "--iterations", str(args.iterations),
                           "--warmup", str(args.warmup)]
                completed = subprocess.run(command, text=True, capture_output=True,
                                           check=True, timeout=60)
                if completed.stderr:
                    raise RuntimeError(f"benchmark emitted stderr: {completed.stderr}")
                report["runs"].append({"repeat": repeat + 1, "command": command,
                                       "result": json.loads(completed.stdout)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Refuse to overwrite another run, even if it appeared during collection.
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
    print(f"Saved {len(report['runs'])} independent-process measurements to {args.output}")


if __name__ == "__main__":
    main()
