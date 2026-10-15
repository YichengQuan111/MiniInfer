#!/usr/bin/env python3
"""Summarize real Nsight Systems intervals for the measured depth pipeline.

Supply the SQLite export and model output size. Device overlap is interval
intersection across application streams, not a sum of event/host durations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3


def union(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def intersect(a, b):
    left = right = 0
    total = 0
    while left < len(a) and right < len(b):
        total += max(0, min(a[left][1], b[right][1]) - max(a[left][0], b[right][0]))
        if a[left][1] <= b[right][1]:
            left += 1
        else:
            right += 1
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--model-size", type=int, choices=(518, 392), default=518)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    db = sqlite3.connect(f"file:{args.sqlite.resolve()}?mode=ro", uri=True)
    # Repeated restored-model raw D2H copies identify the application streams.
    # Context and stream together prevent accidental merging of unrelated IDs.
    streams = db.execute("SELECT contextId,streamId,count(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY "
                         "WHERE copyKind=2 AND bytes=? GROUP BY contextId,streamId HAVING count(*)>=2",
                         (4 * args.model_size**2,)).fetchall()
    if not streams:
        raise RuntimeError("no repeated raw-output D2H copies found")
    nvtx = db.execute("SELECT n.start,n.end,coalesce(n.text,s.value) FROM NVTX_EVENTS n "
                      "LEFT JOIN StringIds s ON n.textId=s.id WHERE n.end>n.start").fetchall()
    slot_ranges = [(a,b) for a,b,name in nvtx if name in ("depth.gpu_slot0", "depth.gpu_slot1")]
    if not slot_ranges:
        raise RuntimeError("no SDK GPU-slot ranges found")
    begin, end = min(a for a,b in slot_ranges), max(b for a,b in slot_ranges)
    per_stream = []
    all_intervals = []
    all_compute = []
    for context, stream, copies in streams:
        kernels = db.execute("SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL "
                             "WHERE contextId=? AND streamId=? AND start>=? AND end<=?",
                             (context, stream, begin, end)).fetchall()
        copies_intervals = db.execute("SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMCPY "
                                     "WHERE contextId=? AND streamId=? AND start>=? AND end<=?",
                                     (context, stream, begin, end)).fetchall()
        compute = union(kernels)
        intervals = union(kernels + copies_intervals)
        all_intervals.append(intervals)
        all_compute.append(compute)
        per_stream.append({"context": context, "stream": stream, "raw_output_copies": copies,
                           "kernel_count": len(kernels),
                           "kernel_busy_ms": sum(b-a for a,b in compute)/1e6})
    combined = union([interval for stream in all_intervals for interval in stream])
    cpu = union([(a,b) for a,b,name in nvtx if name in ("depth.preprocess.cpu", "depth.preprocess.reuse", "depth.preprocess.cpu.hwc", "depth.preprocess.reuse.hwc")])
    stages = {}
    for a,b,name in nvtx:
        if name and name.startswith("depth."):
            item = stages.setdefault(name, {"calls":0,"total_ms":0})
            item["calls"] += 1
            item["total_ms"] += (b-a)/1e6
    for item in stages.values():
        item["mean_ms"] = item["total_ms"]/item["calls"]
    result = {"source":str(args.sqlite), "scope":"Separate profiler run, includes warmup; stream overlap is actual interval intersection.",
              "streams":per_stream, "observed_interval_ms":(end-begin)/1e6,
              "application_gpu_busy_ms":sum(b-a for a,b in combined)/1e6,
              "two_stream_kernel_overlap_ms":intersect(all_compute[0],all_compute[1])/1e6 if len(streams)==2 else None,
              "cpu_preprocess_gpu_overlap_ms":intersect(cpu,combined)/1e6, "nvtx":stages}
    trace = args.sqlite.with_suffix(".nsys-rep")
    if trace.is_file():
        with trace.open("rb") as stream:
            result["trace_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result,stream,indent=2)
        stream.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("nvtx",)}))


if __name__ == "__main__":
    main()
