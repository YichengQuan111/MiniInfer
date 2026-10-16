#!/usr/bin/env python3
"""Summarize A/B/C independent repeats and select a quality-qualified path."""
import argparse
import json
from pathlib import Path
from run_depth_v24 import MODES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="V2.4 evidence directory with replay/ and quality-development.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    quality = json.loads((args.directory / "quality-development.json").read_text())
    if quality.get("passed") is not True:
        raise ValueError("development quality evidence did not pass")
    configuration = next(row for row in quality["configurations"] if row["model_size"] == 518 and row["precision"] == "fp32")
    names = dict(zip(MODES, ("A", "B", "C-basic", "C-fused")))
    result = {"model_size": 518, "precision": "fp32", "gpu_slots": 1, "cases": {},
              "rule": "quality and normal gates first; lowest worst-of-three SDK P99, then replay P99; separate processes, no averaged P99",
              "scope": "local unlocked-clock comparison; the selected path is not a general hardware/precision claim", "passed": False}
    eligible = []
    for mode in MODES:
        candidate = {"quality_passed": configuration["aggregate"][names[mode]]["passed"], "loads": {}}
        for load in ("normal", "overload", "single"):
            report = json.loads((args.directory / "replay" / f"{load}-{mode}" / "report.json").read_text())
            if not report["all_runs_valid"] or len(report["runs"]) != 3:
                raise ValueError(f"invalid or missing independent repeats: {load}-{mode}")
            rows = []
            for run in report["runs"]:
                stat = run["statistics"]["steady"]
                snapshot = run["process_summary"]["sdk_snapshot"]
                if run["process_summary"]["preprocess_mode"] != mode or run["process_summary"]["slots"] != 1:
                    raise ValueError("mode/slot identity mismatch")
                telemetry = run["memory_peaks"]
                rows.append({"sdk_p99_ms": stat["sdk_successful_latency"]["p99_ms"],
                    "replay_p99_ms": stat["replay_successful_latency"]["p99_ms"],
                    "sdk_p50_ms": stat["sdk_successful_latency"]["p50_ms"],
                    "success_qps": stat["successful_throughput_per_second"], "counts": stat["terminal_counts"],
                    "cpu_one_core_mean_percent": telemetry.get("replay:cpu_percent_one_core:steady", {}).get("mean"),
                    "rss_peak_kib": telemetry["replay:rss_kib:steady"]["peak"],
                    "device_total_peak_mib": telemetry.get("gpu_device_0_total:used_mib:steady", {}).get("peak"),
                    "gpu_sm_clock_mean_mhz": telemetry.get("gpu_device_0_total:sm_clock_mhz:steady", {}).get("mean"),
                    "gpu_temperature_mean_c": telemetry.get("gpu_device_0_total:temperature_c:steady", {}).get("mean"),
                    "initial_resources": run["ready"]["event"], "final_resources": snapshot,
                    "host_stages": stat["successful_cpu_wall_stages"], "device_stages": stat["successful_device_stages"],
                    "h2d_bytes": stat["h2d_bytes"]})
            candidate["loads"][load] = rows
        normal = candidate["loads"]["normal"]
        candidate["normal_passed"] = all(row["sdk_p99_ms"] <= 50 and row["replay_p99_ms"] <= 55
            and row["counts"]["succeeded"] == 1000 and all(value == 0 for key, value in row["counts"].items() if key != "succeeded")
            and row["rss_peak_kib"] * 1024 <= 2**31 for row in normal)
        candidate["lifecycle_passed"] = all(
            all(row["counts"][key] == 0 for key in ("failure", "timeout", "closed", "invalid_input"))
            and all(row["final_resources"][key] == 0 for key in ("inflight", "ready", "queued", "executing", "reserved_bytes", "io_device_bytes", "io_pinned_bytes", "host_workspace_bytes"))
            and not row["final_resources"]["fatal_backend"]
            and row["final_resources"]["peak_io_device_bytes"] + row["final_resources"]["context_required_bytes"] <= 2**31
            for rows in candidate["loads"].values() for row in rows)
        if candidate["quality_passed"] and candidate["normal_passed"] and candidate["lifecycle_passed"]:
            eligible.append((max(row["sdk_p99_ms"] for row in normal), max(row["replay_p99_ms"] for row in normal), mode))
        result["cases"][mode] = candidate
    if eligible:
        result["selected_preprocess_mode"] = min(eligible)[2]
        result["passed"] = True
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": result["passed"], "selected": result.get("selected_preprocess_mode"), "output": str(args.output)}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
