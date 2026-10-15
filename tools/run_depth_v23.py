#!/usr/bin/env python3
"""Allocation and GPU slot ablation; each case uses three independent processes.

Run in WSL after sourcing tools/depth_env.sh. Output directories are exclusive.
The package is the quality-qualified V2.2 choice, not selected by this script.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("allocate-1", 1, 0, 0, 20),
    ("gpu-reuse-1", 1, 0, 1, 20),
    ("reuse-1", 1, 1, 1, 20),
    ("reuse-2", 2, 1, 1, 20),
    ("overload-1", 1, 1, 1, 80),
    ("overload-2", 2, 1, 1, 80),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--images-list", type=Path, default=ROOT / "data/depth/replay_v2.txt")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case", choices=[c[0] for c in CASES])
    parser.add_argument("--environment-note", default="clocks unlocked; background desktop not isolated; AC not sampled; no concurrent builds/profiling/GPU workload")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, slots, host, gpu, rate in CASES:
        if args.case and name != args.case:
            continue
        command = [sys.executable, str(ROOT / "tools/run_depth_replay.py"),
            "--executable", str(args.executable.resolve()), "--package", str(args.package.resolve()),
            "--backend", "trt", "--images-list", str(args.images_list.resolve()),
            "--output", str(args.output / name), "--duration", "50", "--warmup", "2", "--rate", str(rate),
            "--workers", "8", "--sdk-workers", "4", "--max-outstanding", "8", "--queue", "4",
            "--budget", "134217728", "--repeats", "3", "--async-sdk", "--slots", str(slots),
            "--reuse-host", str(host), "--reuse-gpu", str(gpu), "--environment-note", args.environment_note]
        with (args.output / (name + ".log")).open("x", encoding="utf-8") as log:
            log.write(json.dumps(command) + "\n")
            log.flush()
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{name} failed; preserve evidence and diagnose before continuing")
        print(json.dumps({"case": name, "completed": True}), flush=True)


if __name__ == "__main__":
    main()
