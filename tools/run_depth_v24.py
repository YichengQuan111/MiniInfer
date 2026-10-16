#!/usr/bin/env python3
"""Three-path replay comparison with a fixed model/precision and one GPU slot.

Normal/overload use the frozen 50-second window and three independent processes.
The separately labelled single-outstanding probe uses 200 arrivals at 10/s per
process, one CPU worker, and one consumer; it is not a throughput benchmark.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODES = ("cpu", "cuda-partial", "cuda-full-basic", "cuda-full-fused")
LOADS = {"normal": (50, 20, 4, 8, 8, 4),
         "overload": (50, 80, 4, 8, 8, 4),
         "single": (20, 10, 1, 1, 1, 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--images-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--load", choices=LOADS)
    parser.add_argument("--slots", type=int, choices=(1, 2), default=1,
                        help="use a fresh output directory for the optional selected-path slot comparison")
    parser.add_argument("--environment-note", default="clocks unlocked; background desktop not isolated; AC not sampled; no concurrent builds/profiling/GPU workload")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for load, (seconds, rate, cpu, callers, outstanding, queue) in LOADS.items():
        if args.load and args.load != load:
            continue
        for mode in MODES:
            if args.mode and args.mode != mode:
                continue
            name = f"{load}-{mode}"
            command = [sys.executable, str(ROOT / "tools/run_depth_replay.py"),
                "--executable", str(args.executable.resolve()), "--package", str(args.package.resolve()),
                "--backend", "trt", "--images-list", str(args.images_list.resolve()),
                "--output", str((args.output / name).resolve()), "--duration", str(seconds),
                "--warmup", "2", "--rate", str(rate), "--workers", str(callers),
                "--sdk-workers", str(cpu), "--max-outstanding", str(outstanding), "--queue", str(queue),
                "--budget", "134217728", "--repeats", "3", "--async-sdk", "--slots", str(args.slots),
                "--preprocess", mode, "--environment-note", args.environment_note]
            with (args.output / (name + ".log")).open("x", encoding="utf-8") as log:
                log.write(json.dumps(command) + "\n")
                log.flush()
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(name + " failed; keep evidence and diagnose")
            print(json.dumps({"case": name, "completed": True}), flush=True)


if __name__ == "__main__":
    main()
