#!/usr/bin/env python3
"""Audit replay artifacts without running inference: REPORT --output NEW_JSON.

The normal gate is a local experiment target, not a hard real-time guarantee.
--stress permits explicit Busy/capacity/late backpressure and skips normal
throughput/P99 gates; accepted failures, ownership, output and resource checks
remain mandatory. --require-soak additionally requires measured long coverage.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

from run_depth_replay import STATUSES, summarize_csv

MIB = 1024 ** 2
GIB = 1024 ** 3
LIMITS = {"rss_growth_bytes": 64 * MIB, "rss_peak_bytes": 2 * GIB,
          "application_io_plus_context_bytes": 2 * GIB, "sdk_p99_ms": 50,
          "replay_p99_ms": 55, "window_ms": 60000, "soak_seconds": 1800,
          "minimum_full_windows": 29, "rss_target_interval_ms": 50,
          "rss_tail_minimum_coverage_ms": 30000,
          "rss_median_interval_limit_ms": 100, "rss_max_gap_ms": 1000,
          "phase_boundary_tolerance_ms": 2000, "sdk_max_gap_ms": 3000}
SDK_FIELDS = ("elapsed_ms", "phase", "inflight", "queued", "executing", "ready", "reserved_bytes",
    "accepted", "succeeded", "failed", "delivered", "abandoned", "abandoned_completed", "io_device_bytes",
    "io_pinned_bytes", "host_workspace_bytes", "peak_io_device_bytes", "peak_io_pinned_bytes",
    "device_allocation_calls", "pinned_allocation_calls", "host_workspace_allocation_calls",
    "context_required_bytes", "engine_serialized_bytes", "gpu_slots", "shutdown_complete", "phase_boundary")
SDK_PHASES = ("startup", "warmup", "drain_warmup", "steady", "drain_steady", "teardown")
CUMULATIVE = ("accepted", "succeeded", "failed", "delivered", "abandoned", "abandoned_completed",
              "device_allocation_calls", "pinned_allocation_calls", "host_workspace_allocation_calls",
              "peak_io_device_bytes", "peak_io_pinned_bytes")
ZERO_FINAL = ("inflight", "queued", "executing", "ready", "reserved_bytes", "io_device_bytes",
              "io_pinned_bytes", "host_workspace_bytes")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, name):
    require(not isinstance(value, bool) and isinstance(value, (int, float, str)), f"invalid {name}")
    result = float(value)
    require(math.isfinite(result) and result >= 0, f"non-finite/negative {name}")
    return result


def integer(value, name):
    result = number(value, name)
    require(result.is_integer(), f"non-integer {name}")
    return int(result)


def load_json(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, f"duplicate JSON key: {key}")
            value[key] = item
        return value
    def nonfinite(value):
        raise ValueError(f"non-finite JSON constant: {value}")
    result = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique, parse_constant=nonfinite)
    require(isinstance(result, dict), f"expected JSON object: {path}")
    return result


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(MIB), b""):
            value.update(block)
    return value.hexdigest()


def artifact(root, name):
    require(isinstance(name, str) and name and not Path(name).is_absolute(), "artifact name must be relative")
    path = (root / name).resolve(strict=True)
    require(path.is_relative_to(root.resolve()) and path.is_file(), "artifact escapes its repeat directory")
    return path


def phase_boundary(run):
    events = run.get("phase_events", [])
    require(isinstance(events, list), "phase_events must be a list")
    previous = -1.0
    start = end = None
    for event in events:
        require(isinstance(event, dict), "invalid phase event")
        elapsed = number(event["elapsed_ms"], "phase event elapsed_ms")
        require(elapsed >= previous, "phase event clock moved backwards")
        previous = elapsed
        if event.get("phase") == "steady":
            require(start is None and end is None, "duplicate steady phase boundary")
            start = elapsed
        if event.get("phase") == "drain" and event.get("source") == "steady":
            require(start is not None and end is None, "invalid steady drain boundary")
            end = elapsed
    require((start is None) == (end is None), "incomplete observed steady phase boundaries")
    return {"start_ms": start, "end_ms": end,
            "observed_duration_ms": end - start if start is not None else None,
            "clock": "Python process-relative monotonic clock at stdout reception",
            "limitation": "pipe and reader scheduling can shift reception boundaries by milliseconds"}


def rss_windows(samples):
    """Only complete [start,end) windows supported by a later actual sample."""
    require(bool(samples), "missing steady RSS samples")
    origin, last = samples[0][0], samples[-1][0]
    count = int((last - origin) // LIMITS["window_ms"])
    groups = [[] for _ in range(count)]
    tail = []
    for elapsed, value in samples:
        index = int((elapsed - origin) // LIMITS["window_ms"])
        if index < count:
            groups[index].append(value)
        else:
            tail.append((elapsed, value))
    windows = []
    for index, values in enumerate(groups):
        require(bool(values), "empty complete RSS window")
        windows.append({"index": index, "start_elapsed_ms": origin + index * LIMITS["window_ms"],
                        "end_elapsed_ms": origin + (index + 1) * LIMITS["window_ms"],
                        "samples": len(values), "median_bytes": statistics.median(values), "peak_bytes": max(values)})
    medians = [window["median_bytes"] for window in windows]
    tail_values = [value for _, value in tail]
    tail_gaps = [after[0] - before[0] for before, after in zip(tail, tail[1:])]
    tail_span = tail[-1][0] - tail[0][0] if tail else 0
    tail_median = statistics.median(tail_values) if tail else None
    tail_minimum_samples = math.ceil(tail_span / LIMITS["rss_median_interval_limit_ms"]) + 1
    tail_sampled = (tail_span >= LIMITS["rss_tail_minimum_coverage_ms"]
        and len(tail) >= tail_minimum_samples and bool(tail_gaps)
        and statistics.median(tail_gaps) <= LIMITS["rss_median_interval_limit_ms"]
        and max(tail_gaps) <= LIMITS["rss_max_gap_ms"])
    tail_used = bool(medians) and tail_sampled
    trend = {"slope_mib_per_minute": None, "r_squared": None}
    if len(windows) >= 2:
        xs = [index + 0.5 for index in range(len(windows))]  # minutes from first sample
        ys = [value / MIB for value in medians]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        residual = sum((y - (my + slope * (x - mx))) ** 2 for x, y in zip(xs, ys))
        total = sum((y - my) ** 2 for y in ys)
        trend = {"slope_mib_per_minute": slope, "r_squared": 1 - residual / total if total else 1.0}
    return {"complete_windows": windows, "complete_window_count": count,
            "partial_tail": {"samples": len(tail), "minimum_samples_for_coverage": tail_minimum_samples,
                             "start_elapsed_ms": tail[0][0] if tail else None,
                             "end_elapsed_ms": tail[-1][0] if tail else None, "observed_span_ms": tail_span,
                             "median_interval_ms": statistics.median(tail_gaps) if tail_gaps else None,
                             "maximum_gap_ms": max(tail_gaps) if tail_gaps else None,
                             "median_bytes": tail_median, "peak_bytes": max(tail_values) if tail else None,
                             "sufficient_coverage_and_sampling": tail_sampled,
                             "median_growth_from_first_complete_window_bytes": tail_median - medians[0] if medians and tail else None,
                             "used_for_growth_gate": tail_used},
            "last_minus_first_median_bytes": medians[-1] - medians[0] if len(medians) >= 2 else None,
            "highest_minus_first_median_bytes": max(medians) - medians[0] if len(medians) >= 2 else None,
            "complete_window_growth_assessed": len(medians) >= 2,
            "growth_assessed": len(medians) >= 2 or tail_used, "trend_on_window_medians": trend}


def read_memory(path):
    rss = []
    whole_rss = []
    observations = defaultdict(lambda: {"samples": 0, "missing": 0, "sum": 0.0, "peak": None})
    missing_steady_rss = 0
    previous = -1.0
    steady_ended = False
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == ["elapsed_ms", "phase", "role", "metric", "value"], "memory.csv schema mismatch")
        for row in reader:
            require(None not in row and all(value is not None for value in row.values()), "memory.csv column mismatch")
            elapsed = number(row["elapsed_ms"], "memory elapsed_ms")
            require(elapsed >= previous, "memory clock moved backwards")
            previous = elapsed
            value = number(row["value"], "memory value") if row["value"] else None
            if row["role"] == "replay" and row["metric"] == "rss_kib":
                if value is not None:
                    whole_rss.append(value * 1024)
                if row["phase"] == "steady":
                    require(not steady_ended, "RSS returned to steady after leaving the phase")
                    if value is None:
                        missing_steady_rss += 1
                    else:
                        rss.append((elapsed, value * 1024))
                elif rss:
                    steady_ended = True
            if row["role"].startswith("gpu_device_") or row["metric"] == "cpu_percent_one_core":
                for phase in (row["phase"], "whole"):
                    item = observations[f"{row['role']}:{row['metric']}:{phase}"]
                    if value is None:
                        item["missing"] += 1
                    else:
                        item["samples"] += 1; item["sum"] += value
                        item["peak"] = value if item["peak"] is None else max(item["peak"], value)
    require(len(rss) >= 2, "insufficient measured steady RSS coverage")
    gaps = [after[0] - before[0] for before, after in zip(rss, rss[1:])]
    require(all(gap > 0 for gap in gaps), "duplicate steady RSS timestamp")
    result = rss_windows(rss)
    result.update(samples=len(rss), first_elapsed_ms=rss[0][0], last_elapsed_ms=rss[-1][0],
                  observed_span_ms=rss[-1][0] - rss[0][0], median_interval_ms=statistics.median(gaps),
                  maximum_gap_ms=max(gaps), missing_samples=missing_steady_rss,
                  peak_bytes=max(value for _, value in rss), peak_scope="steady RSS samples",
                  whole_process_peak_bytes=max(whole_rss), whole_process_peak_scope="all sampled phases, including startup and teardown")
    observed = {key: {"samples": value["samples"], "missing": value["missing"],
        "mean": value["sum"] / value["samples"] if value["samples"] else None, "peak": value["peak"]}
        for key, value in sorted(observations.items())}
    return result, observed


def check_ledger(row, name):
    require(row["inflight"] == row["queued"] + row["executing"] + row["ready"], f"{name}: inflight conservation")
    require(row["accepted"] == row["delivered"] + row["abandoned_completed"] + row["inflight"],
            f"{name}: accepted conservation")
    require(row["succeeded"] + row["failed"] == row["delivered"] + row["abandoned_completed"] + row["ready"],
            f"{name}: terminal conservation")
    require(row["abandoned_completed"] <= row["abandoned"], f"{name}: abandonment conservation")
    require((row["inflight"] == 0) == (row["reserved_bytes"] == 0), f"{name}: quota lifetime mismatch")


def read_sdk(path, case, summary):
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == list(SDK_FIELDS), "sdk.csv schema mismatch")
        for line, raw in enumerate(reader, 2):
            require(None not in raw and all(value is not None for value in raw.values()), "sdk.csv column mismatch")
            row = {key: (raw[key] if key == "phase" else number(raw[key], key) if key == "elapsed_ms"
                         else integer(raw[key], key)) for key in SDK_FIELDS}
            require(row["phase"] in SDK_PHASES, "unknown SDK sampling phase")
            require(row["shutdown_complete"] in (0, 1), "invalid shutdown flag")
            require(row["phase_boundary"] in (0, 1), "invalid phase boundary flag")
            if rows:
                require(row["elapsed_ms"] >= rows[-1]["elapsed_ms"], "SDK clock moved backwards")
                require(SDK_PHASES.index(row["phase"]) >= SDK_PHASES.index(rows[-1]["phase"]), "SDK phase moved backwards")
                for key in CUMULATIVE:
                    require(row[key] >= rows[-1][key], f"cumulative SDK counter decreased: {key}")
            check_ledger(row, f"sdk.csv line {line}")
            require(row["reserved_bytes"] <= integer(case["budget"], "budget"), "SDK byte budget exceeded")
            require(row["inflight"] <= integer(case["queue"], "queue") + integer(case["sdk_workers"], "sdk_workers"),
                    "SDK inflight admission count exceeded")
            require(row["queued"] <= integer(case["queue"], "queue")
                    and row["executing"] <= integer(case["sdk_workers"], "sdk_workers"), "SDK queue/worker bound exceeded")
            require(row["gpu_slots"] == integer(summary["slots"], "slots"), "SDK slot count differs")
            require(row["io_device_bytes"] <= row["peak_io_device_bytes"]
                    and row["io_pinned_bytes"] <= row["peak_io_pinned_bytes"], "SDK current allocation exceeds its peak")
            require(row["failed"] == 0, "SDK reports real backend failures")
            if row["shutdown_complete"]:
                require(row["phase"] == "teardown" and all(row[key] == 0 for key in ZERO_FINAL),
                        "shutdown retained requests, quota or fixed resources")
            rows.append(row)
    require(bool(rows) and rows[0]["phase"] == "startup", "missing startup SDK sampling phase")
    final = rows[-1]
    require(final["phase"] == "teardown" and final["shutdown_complete"] == 1,
            "missing final shutdown SDK telemetry row")
    require(all(final[key] == 0 for key in ZERO_FINAL), "final SDK resources/requests/quota not zero")
    for key, value in summary["sdk_snapshot"].items():
        if key in final:
            require(value == final[key], f"SDK final telemetry/summary differs: {key}")
    steady = [row for row in rows if row["phase"] == "steady"]
    require(bool(steady), "missing steady SDK telemetry")
    require(steady[0]["phase_boundary"] == 1
            and sum(row["phase_boundary"] for row in steady) == 1,
            "missing, late or duplicate synchronous steady SDK phase boundary")
    require(steady[0]["inflight"] == 0, "steady SDK boundary must precede request submission")
    intervals = [b["elapsed_ms"] - a["elapsed_ms"] for a, b in zip(steady, steady[1:])]
    allocations = {}
    for key, reuse in (("device_allocation_calls", summary["reuse_gpu"]),
                       ("pinned_allocation_calls", summary["reuse_gpu"]),
                       ("host_workspace_allocation_calls", summary["reuse_host"])):
        require(isinstance(reuse, bool), "reuse flag must be Boolean")
        values = [row[key] for row in steady]
        allocations[key] = {"reuse": reuse, "first": values[0], "last": values[-1], "final": final[key],
                            "growth": final[key] - values[0],
                            "passed": not reuse or min(values) == max(values) == final[key]}
    return {"sample_count": len(rows), "steady_samples": len(steady),
            "steady_boundary": steady[0],
            "steady_observed_span_ms": steady[-1]["elapsed_ms"] - steady[0]["elapsed_ms"],
            "steady_maximum_gap_ms": max(intervals) if intervals else None,
            "maximum_reserved_bytes": max(row["reserved_bytes"] for row in rows),
            "maximum_inflight": max(row["inflight"] for row in rows),
            "maximum_application_io_plus_context_bytes": max(row["io_device_bytes"] + row["context_required_bytes"] for row in rows),
            "peak_io_plus_context_requirement_bytes": max(row["peak_io_device_bytes"] + row["context_required_bytes"] for row in rows),
            "allocation_counters": allocations, "final": final,
            "memory_scope": "application CUDA IO plus TensorRT static context requirement; excludes weights, libraries, driver and other allocations; not total GPU memory",
            "shutdown_scope": "owned IO/pinned/host scratch and request quota zero; context_required_bytes and engine_serialized_bytes remain historical requirements"}


def analyze_run(run, root, case, require_soak, stress):
    result = {"repeat_directory": str(root), "passed": False, "errors": [], "checks": {}, "input_sha256": {}}
    try:
        require(isinstance(run.get("errors"), list) and not run["errors"], "replay run errors are present/missing")
        require(type(run.get("returncode")) is int and run["returncode"] == 0, "replay process did not exit successfully")
        summary = run["process_summary"]
        require(summary["schema"] == "miniinfer-depth-replay-v4", "v4 output checks/SDK telemetry required; old reports cannot prove stability")
        require(summary["output_checks"]["enabled"] is True and summary["sdk_telemetry_enabled"] is True,
                "output checking and SDK sampling must both be enabled")
        require(summary["sdk_snapshot"].get("fatal_backend") is False, "fatal backend state/missing fatal flag")
        for key in ("slots", "reuse_host", "reuse_gpu", "async_sdk"):
            if key in case:
                require(case[key] == summary[key], f"case/process summary differs: {key}")
        raw_path = artifact(root, run.get("request_csv", "raw.csv"))
        memory_path = artifact(root, run.get("memory_csv", "memory.csv"))
        sdk_path = artifact(root, run.get("sdk_csv", "sdk.csv"))
        for path in (raw_path, memory_path, sdk_path):
            result["input_sha256"][str(path)] = digest(path)
        statistics_recomputed = summarize_csv(raw_path, summary)
        require(run.get("statistics") == statistics_recomputed, "stored replay statistics differ from recomputed raw.csv")
        expected_full = 0
        with raw_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row["status"] == "succeeded" and (int(row["index"]) // summary["image_count"]) % 128 == 0:
                    expected_full += 1
        output = summary["output_checks"]
        require(integer(output["full"], "full checks") == expected_full, "periodic full-map output check count mismatch")
        require(integer(output["failures"], "output failures") == 0, "output consistency failures")
        result["output_checks"] = dict(output, expected_full=expected_full)
        result["statistics"] = {phase: {key: value for key, value in statistics_recomputed[phase].items()
            if key in ("planned", "submitted", "accepted", "accepted_succeeded", "accepted_failed", "terminal_counts",
                       "successful_throughput_per_second", "sdk_successful_latency", "replay_successful_latency")}
            for phase in ("warmup", "steady")}
        rss, observations = read_memory(memory_path)
        sdk = read_sdk(sdk_path, case, summary)
        boundary = phase_boundary(run)
        result.update(rss=rss, sdk=sdk, observations=observations, observed_phase=boundary,
                      telemetry_errors_observed=run.get("telemetry_errors", []))
        checks = result["checks"]
        checks["output_and_counter_consistency"] = True
        checks["sdk_ledger_budget_and_shutdown"] = True
        checks["fixed_reuse_allocations"] = all(value["passed"] for value in sdk["allocation_counters"].values())
        checks["application_io_plus_context_bound"] = sdk["peak_io_plus_context_requirement_bytes"] <= LIMITS["application_io_plus_context_bytes"]
        checks["rss_peak_bound"] = rss["whole_process_peak_bytes"] <= LIMITS["rss_peak_bytes"]
        checks["rss_sampling_density"] = (rss["median_interval_ms"] <= LIMITS["rss_median_interval_limit_ms"]
            and rss["maximum_gap_ms"] <= LIMITS["rss_max_gap_ms"] and rss["missing_samples"] == 0)
        checks["rss_window_growth"] = (rss["last_minus_first_median_bytes"] <= LIMITS["rss_growth_bytes"]
            and rss["highest_minus_first_median_bytes"] <= LIMITS["rss_growth_bytes"]) if rss["complete_window_growth_assessed"] else None
        checks["rss_tail_growth"] = (rss["partial_tail"]["median_growth_from_first_complete_window_bytes"]
            <= LIMITS["rss_growth_bytes"]) if rss["partial_tail"]["used_for_growth_gate"] else None
        for phase, metrics in statistics_recomputed.items():
            counts = metrics["terminal_counts"]
            checks[f"{phase}_no_failures"] = all(counts[name] == 0 for name in ("failure", "timeout", "invalid_input", "closed"))
            checks[f"{phase}_normal_no_reject_or_missed_generation"] = (
                metrics["accepted_succeeded"] == metrics["planned"] and all(
                    counts[name] == 0 for name in STATUSES if name != "succeeded")) if not stress else None
        steady = statistics_recomputed["steady"]
        require(steady["accepted_succeeded"] > 0, "no successful steady samples")
        checks["normal_sdk_p99"] = steady["sdk_successful_latency"]["p99_ms"] <= LIMITS["sdk_p99_ms"] if not stress else None
        checks["normal_replay_p99"] = steady["replay_successful_latency"]["p99_ms"] <= LIMITS["replay_p99_ms"] if not stress else None
        duration = number(summary["steady"]["duration_ms"], "steady duration_ms")
        tolerance = LIMITS["phase_boundary_tolerance_ms"]
        measured = boundary["observed_duration_ms"]
        if measured is not None:
            require(measured > 0, "observed steady phase has no duration")
            checks["observed_phase_duration_matches_process"] = abs(measured - duration) <= tolerance
            checks["rss_phase_edges"] = (abs(rss["first_elapsed_ms"] - boundary["start_ms"]) <= tolerance
                and abs(rss["last_elapsed_ms"] - boundary["end_ms"]) <= tolerance)
        checks["rss_covers_reported_steady_duration"] = rss["observed_span_ms"] >= max(0, duration - tolerance)
        checks["sdk_covers_reported_steady_duration"] = (sdk["steady_observed_span_ms"] >= max(0, duration - tolerance)
            and (sdk["steady_maximum_gap_ms"] is None or sdk["steady_maximum_gap_ms"] <= LIMITS["sdk_max_gap_ms"]))
        soak_checks = {"process_steady_at_least_1800s": duration >= LIMITS["soak_seconds"] * 1000,
            "measured_phase_boundary_available": measured is not None,
            "observed_phase_at_least_1800s_with_2s_reception_tolerance": measured is not None and measured >= LIMITS["soak_seconds"] * 1000 - tolerance,
            "rss_actual_span_at_least_1798s": rss["observed_span_ms"] >= LIMITS["soak_seconds"] * 1000 - tolerance,
            "at_least_29_complete_windows": rss["complete_window_count"] >= LIMITS["minimum_full_windows"]}
        result["soak_evidence"] = {"required": require_soak, "process_steady_duration_ms": duration,
            "checks": soak_checks, "eligible": all(soak_checks.values()),
            "sampling_note": "an 1800s phase sampled every 50ms normally spans only 1799.95s; boundary tolerance is frozen at 2s, while sample gaps/density are checked independently; configured case.duration alone is never evidence"}
        if require_soak:
            checks.update({f"soak_{key}": value for key, value in soak_checks.items()})
        result["passed"] = all(value is not False for value in checks.values())
    except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError) as error:
        result["errors"].append(f"{type(error).__name__}: {error}")
    return result


def analyze(report_path, require_soak=False, stress=False):
    report_path = Path(report_path).resolve(strict=True)
    result = {"schema": "miniinfer-depth-stability-audit-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "report": str(report_path), "report_sha256": digest(report_path), "require_soak": require_soak,
              "stress": stress, "limits": LIMITS.copy(), "runs": [], "errors": [], "passed": False,
              "observational_metrics": "GPU device-total and sampled CPU means/peaks include background/system effects and are not acceptance gates"}
    try:
        source = load_json(report_path)
        result.update(case=source.get("case"), identity=source.get("identity"))
        require(source.get("schema") in ("miniinfer-depth-replay-report-v1", "miniinfer-depth-replay-report-v2"), "unknown replay report schema")
        require(source.get("all_runs_valid") is True, "source replay report was not valid")
        runs = source["runs"]
        require(isinstance(runs, list) and bool(runs), "report must contain runs")
        require(integer(source["case"]["repeats"], "repeats") == len(runs), "report repeats/run count mismatch")
        for index, run in enumerate(runs, 1):
            root = report_path.parent / f"repeat_{index:02d}"
            result["runs"].append(analyze_run(run, root, source["case"], require_soak, stress))
        result["passed"] = all(run["passed"] for run in result["runs"])
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        result["errors"].append(f"{type(error).__name__}: {error}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", required=True, type=Path, help="new exclusive JSON file")
    parser.add_argument("--require-soak", action="store_true")
    parser.add_argument("--stress", action="store_true")
    args = parser.parse_args()
    try:
        result = analyze(args.report, args.require_soak, args.stress)
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"passed": result["passed"], "output": str(args.output), "runs": len(result["runs"])}))
        return 0 if result["passed"] else 1
    except (OSError, ValueError) as error:
        print(json.dumps({"passed": False, "error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
