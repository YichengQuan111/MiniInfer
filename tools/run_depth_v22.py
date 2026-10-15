#!/usr/bin/env python3
"""Reproducible V2.2 static-size/precision experiment; phases never overwrite evidence.

Run from WSL after source tools/depth_env.sh, using the depth Python environment.
Reference generation is intentionally separate (see docs/depth_v2_validation.md).
Candidate descriptors are local measurement inputs, not validated release packages.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from depth.reference import sha256

ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "data/generated/depth"
ARTIFACTS = GENERATED / "v22"
REPORTS = ROOT / "docs/validation/v22"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def candidates():
    result = []
    for size in (518, 392):
        for precision in ("fp32", "fp16"):
            key = f"{size}-{precision}"
            result.append(dict(key=key, size=size, precision=precision,
                reference=GENERATED / ("v22-reference-518" if size == 518 else "v22-export-392"),
                onnx=GENERATED / ("export-v1" if size == 518 else "v22-export-392") / f"depth_anything_v2_vits_{size}.onnx",
                engine=(GENERATED / "engines" / ("depth-fp32.engine" if precision == "fp32" else "depth-fp16-v2.engine"))
                       if size == 518 else ARTIFACTS / "engines" / f"{key}.engine",
                thresholds=ROOT / "data/depth" / f"thresholds_trt_{precision}{'_392' if size == 392 else ''}.json"))
    return result


def run(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps([str(v) for v in command]) + "\n")
        stream.flush()
        return subprocess.run([str(v) for v in command], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode


def descriptor(candidate):
    directory = ARTIFACTS / "candidates" / candidate["key"]
    directory.mkdir(parents=True, exist_ok=False)
    for suffix in ("", ".manifest", ".layers.json"):
        shutil.copyfile(str(candidate["engine"]) + suffix, directory / ("model.engine" + suffix))
    entries = {"schema": "miniinfer-model-1", "contract": "depth-v1" if candidate["size"] == 518 else "depth-v2-392",
               "backend": "trt", "precision": candidate["precision"], "model": "model.engine",
               "model_sha256": sha256(directory / "model.engine")}
    (directory / "manifest.txt").write_text("".join(f"{k}={v}\n" for k, v in sorted(entries.items())), encoding="utf-8")
    (directory / "EXPERIMENT.txt").write_text("Unreleased measurement descriptor. Quality eligibility is in the V2.2 decision report.\n")
    return directory


def quality(candidate, split, build):
    key = candidate["key"]
    output = ARTIFACTS / "quality" / f"{key}-{split}"
    command = [sys.executable, "-m", "tools.depth.validate_backend", "--cli", build / "bin/depth_trt_cli",
               "--model", candidate["engine"], "--reference", candidate["reference"],
               "--samples", ROOT / "data/depth/samples_v2.json", "--thresholds", candidate["thresholds"],
               "--model-size", candidate["size"], "--split", split, "--output", output]
    code = run(command, REPORTS / f"quality-{key}-{split}.log")
    report = read(output / "report.json")
    save(REPORTS / f"quality-{key}-{split}.json", report)
    if report["errors"]:
        raise RuntimeError(f"{key}: validation tool failed: {report['errors']}")
    # A measured numerical/quality failure remains in the matrix, never hidden.
    if code not in (0, 1):
        raise RuntimeError(f"unexpected validation exit {code}")


def decide():
    baseline = read(REPORTS / "quality-518-fp32-development.json")
    baseline_pairs = {(s["id"], i): p["python"]["correct"] for s in baseline["samples"]
                      for i, p in enumerate(s["pair_results"]) if p["status"] == "valid"}
    rows = []
    for c in candidates():
        quality_report = read(REPORTS / f"quality-{c['key']}-development.json")
        new_errors = [{"sample": s["id"], "pair": i} for s in quality_report["samples"]
                      for i, p in enumerate(s["pair_results"]) if p["status"] == "valid"
                      and baseline_pairs[(s["id"], i)] and not p["candidate"]["correct"]]
        repeats = [read(REPORTS / ("replay-" + c["key"]) / f"repeat_{i:02}/summary.json") for i in (1, 2, 3)]
        measurements = []
        for repeat in repeats:
            stats = repeat["statistics"]["steady"]
            measurements.append({"sdk": stats["sdk_successful_latency"], "replay": stats["replay_successful_latency"],
                "success_rate": stats["success_rate_of_planned"], "throughput": stats["successful_throughput_per_second"],
                "terminal_counts": stats["terminal_counts"], "memory_peaks": repeat["memory_peaks"], "errors": repeat["errors"]})
        normal_passed = all(not r["errors"] and r["success_rate"] == 1.0 and r["sdk"]["p99_ms"] <= 50
                            and r["replay"]["p99_ms"] <= 55 and
                            r["memory_peaks"]["replay:rss_kib:steady"]["peak"] * 1024 <= 2 * 1024**3
                            for r in measurements)
        rows.append({"key": c["key"], "model_size": c["size"], "precision": c["precision"],
            "same_size_quality_passed": quality_report["passed"], "ordering": quality_report["aggregate"]["ordering"],
            "new_pair_errors_vs_518_python": new_errors, "normal_passed": normal_passed,
            "eligible": quality_report["passed"] and not new_errors and normal_passed,
            "engine_bytes": c["engine"].stat().st_size, "engine_sha256": sha256(c["engine"]),
            "application_io_device_bytes": 4 * c["size"]**2 * 4,
            "application_io_pinned_bytes": 4 * c["size"]**2 * 4,
            "repeats": measurements})
    eligible = [r for r in rows if r["eligible"]]
    if not eligible:
        raise RuntimeError("No candidate meets frozen development/normal criteria; retain evidence and investigate.")
    selected = min(eligible, key=lambda r: (max(v["sdk"]["p99_ms"] for v in r["repeats"]), r["engine_bytes"]))
    save(REPORTS / "decision.json", {"rule": "eligible only; smallest worst-of-three SDK P99, engine size breaks ties",
        "split": "development", "final_split_used_for_selection": False, "selected": selected["key"], "candidates": rows,
        "memory_scope": "IO allocations exclude TRT context/weights and CUDA libraries; RSS and device-total telemetry are separate."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("engines", "development", "benchmark", "decision", "final", "package"))
    parser.add_argument("--build", type=Path, required=True)
    args = parser.parse_args()
    build = args.build.resolve()
    all_candidates = candidates()
    if args.phase == "engines":
        (ARTIFACTS / "engines").mkdir(parents=True, exist_ok=True)
        for c in all_candidates:
            if c["size"] == 392:
                if run([build / "bin/depth_engine_builder", "--onnx", c["onnx"], "--output", c["engine"],
                        "--precision", c["precision"]], REPORTS / f"build-{c['key']}.log"):
                    raise RuntimeError("engine build failed: " + c["key"])
    elif args.phase == "development":
        for c in all_candidates:
            quality(c, "development", build)
    elif args.phase == "benchmark":
        for c in all_candidates:
            package = descriptor(c)
            if run([sys.executable, ROOT / "tools/run_depth_replay.py", "--executable", build / "bin/depth_replay",
                    "--package", package, "--backend", "trt", "--images-list", ROOT / "data/depth/replay_v2.txt",
                    "--output", REPORTS / ("replay-" + c["key"]), "--duration", 50, "--warmup", 2, "--rate", 20,
                    "--workers", 8, "--max-outstanding", 8, "--queue", 4, "--budget", 134217728, "--repeats", 3,
                    "--environment-note", "V2.2 Release; AC not sampled; unlocked clocks; record background load separately"],
                   REPORTS / f"replay-{c['key']}.log"):
                raise RuntimeError("replay failed: " + c["key"])
    elif args.phase == "decision":
        decide()
    else:
        chosen = read(REPORTS / "decision.json")["selected"]
        c = next(c for c in all_candidates if c["key"] == chosen)
        if args.phase == "final":
            quality(c, "final", build)
        else:
            output = GENERATED / "packages" / ("v22-" + chosen)
            golden = next(s["id"] for s in read(ROOT / "data/depth/samples_v2.json")["samples"] if s["split"] == "final")
            if run([sys.executable, ROOT / "tools/package_depth.py", "create", "--model", c["engine"], "--onnx", c["onnx"],
                    "--output", output, "--reference", c["reference"], "--samples", ROOT / "data/depth/samples_v2.json",
                    "--thresholds", c["thresholds"], "--report", REPORTS / f"quality-{chosen}-final.json",
                    "--backend", "trt", "--precision", c["precision"], "--model-size", c["size"], "--golden", golden],
                   REPORTS / "package.log"):
                raise RuntimeError("final package validation failed")
    print(json.dumps({"phase": args.phase, "completed": True}))


if __name__ == "__main__":
    main()
