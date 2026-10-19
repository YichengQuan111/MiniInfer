#!/usr/bin/env python3
"""Frozen V2 normal/load/slow-consumer and real 1800-second soak scenarios."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CASES = {"normal": (20, 50, 3, 0, False),
         "saturation-probe": (40, 50, 3, 0, True),
         "overload": (80, 50, 3, 0, True),
         "slow-consumer": (20, 50, 3, 200, True),
         "capacity-relative-overload": (160, 50, 3, 0, True),
         "soak": (20, 1800, 1, 0, False)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("executable", "package", "images-list", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--preprocess", choices=("cpu", "cuda-partial", "cuda-full-basic", "cuda-full-fused"), required=True)
    parser.add_argument("--slots", type=int, choices=(1, 2), required=True)
    parser.add_argument("--case", choices=CASES, required=True,
        help="capacity-relative-overload is separate evidence only if 80/s did not overload the optimized path")
    args = parser.parse_args()
    rate, seconds, repeats, delay, stress = CASES[args.case]
    args.output.mkdir(parents=True, exist_ok=True)
    destination = (args.output / args.case).resolve()
    command = [sys.executable, str(ROOT / "tools/run_depth_replay.py"),
        "--executable", str(args.executable.resolve()), "--package", str(args.package.resolve()),
        "--backend", "trt", "--images-list", str(args.images_list.resolve()), "--output", str(destination),
        "--duration", str(seconds), "--warmup", "2", "--rate", str(rate), "--repeats", str(repeats),
        "--sdk-workers", "4", "--workers", "8", "--max-outstanding", "8", "--queue", "4", "--budget", "134217728",
        "--async-sdk", "--slots", str(args.slots), "--preprocess", args.preprocess,
        "--consumer-delay-ms", str(delay), "--check-outputs", "--sdk-telemetry",
        "--environment-note", "clocks unlocked; background desktop not isolated; AC not sampled; no concurrent builds/profiling/GPU workload"]
    with (args.output / (args.case + ".log")).open("x", encoding="utf-8") as log:
        log.write(json.dumps(command) + "\n"); log.flush()
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    audit = [sys.executable, str(ROOT / "tools/analyze_depth_stability.py"), str(destination / "report.json"),
             "--output", str((args.output / (args.case + "-acceptance.json")).resolve())]
    if stress:
        audit.append("--stress")
    if args.case == "soak":
        audit.append("--require-soak")
    subprocess.run(audit, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
