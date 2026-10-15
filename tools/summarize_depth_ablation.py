#!/usr/bin/env python3
"""Collect validated replay reports without merging independent percentiles."""
import argparse
import json
from pathlib import Path


def summarize(report):
    if not report["all_runs_valid"] or any(run["errors"] for run in report["runs"]):
        raise ValueError("invalid replay evidence")
    result = []
    for run in report["runs"]:
        steady = run["statistics"]["steady"]
        terminal = steady["terminal_counts"]
        result.append({
            "planned": steady["planned"], "counts": terminal,
            "sdk": steady["sdk_successful_latency"], "replay": steady["replay_successful_latency"],
            "delivery": steady["delivery_successful_latency"],
            "successful_qps": steady["successful_throughput_per_second"],
            "stages": steady["successful_cpu_wall_stages"],
            "device_stages": steady.get("successful_device_stages", {}),
            "normal_targets_passed": (terminal["succeeded"] == steady["planned"]
                and steady["sdk_successful_latency"]["p99_ms"] <= 50
                and steady["replay_successful_latency"]["p99_ms"] <= 55),
            "memory": run["memory_peaks"], "startup": run["ready"],
            "resources_after_shutdown": run["process_summary"]["sdk_snapshot"],
        })
    return {"case": report["case"], "identity": report["identity"], "repeats": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = {}
    for path in args.reports:
        name = path.parent.name
        if name in rows:
            raise ValueError("duplicate case name: " + name)
        rows[name] = summarize(json.loads(path.read_text(encoding="utf-8")))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump({"scope": "Independent process distributions; GPU device-total memory includes other processes.",
                   "cases": rows}, stream, indent=2)
        stream.write("\n")
    for name, case in rows.items():
        print(json.dumps({"case": name,
            "sdk_p99_ms": [r["sdk"]["p99_ms"] for r in case["repeats"]],
            "success_qps": [r["successful_qps"] for r in case["repeats"]],
            "busy": [r["counts"]["busy"] for r in case["repeats"]]}))


if __name__ == "__main__":
    main()
