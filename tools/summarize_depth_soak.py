#!/usr/bin/env python3
"""Report minute-by-minute latency from an already passed measured soak audit."""
import argparse
import csv
import json
import math
from pathlib import Path

from run_depth_benchmark import distribution, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    if audit.get("passed") is not True or audit.get("require_soak") is not True:
        raise ValueError("a passed real long-soak audit is required")
    report_path = Path(audit["report"])
    if sha256(report_path) != audit["report_sha256"]:
        raise ValueError("report changed after its audit")
    report = json.loads(report_path.read_text())
    result = {"audit_sha256": sha256(args.audit), "report_sha256": sha256(report_path),
        "window_seconds": 60,
        "scope": "Latency grouped by planned arrival minute; throughput counts successful ready times inside each minute. Window P99 values are never averaged.",
        "runs": []}
    for index, run in enumerate(report["runs"], 1):
        duration_ms = run["process_summary"]["steady"]["duration_ms"]
        count = math.ceil(duration_ms / 60000)
        windows = [{"minute": i + 1, "start_ms": i * 60000,
                    "seconds": min(60000, duration_ms - i * 60000) / 1000,
                    "counts": {}, "sdk": [], "replay": [], "completed": 0} for i in range(count)]
        raw = report_path.parent / f"repeat_{index:02d}" / run["request_csv"]
        if sha256(raw) != audit["runs"][index - 1]["input_sha256"].get(str(raw)):
            raise ValueError("request CSV changed after its audit")
        with raw.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row["phase"] != "steady":
                    continue
                window = windows[int(float(row["planned_ms"]) // 60000)]
                status = row["status"]
                window["counts"][status] = window["counts"].get(status, 0) + 1
                if status == "succeeded":
                    ready = float(row["ready_ms"])
                    window["sdk"].append(ready - float(row["submit_ms"]))
                    window["replay"].append(ready - float(row["planned_ms"]))
                    if ready < duration_ms:
                        windows[int(ready // 60000)]["completed"] += 1
        assert sum(sum(w["counts"].values()) for w in windows) == run["statistics"]["steady"]["planned"]
        assert sum(w["completed"] for w in windows) == run["statistics"]["steady"]["succeeded_in_window"]
        for window in windows:
            window["sdk_latency"] = distribution(window.pop("sdk"))
            window["replay_latency"] = distribution(window.pop("replay"))
            window["success_qps"] = window["completed"] / window["seconds"]
        result["runs"].append({"raw_sha256": sha256(raw), "windows": windows})
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "runs": len(result["runs"])}))


if __name__ == "__main__":
    main()
