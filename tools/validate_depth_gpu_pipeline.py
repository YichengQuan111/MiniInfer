#!/usr/bin/env python3
"""Compare persistent TensorRT A/B/C paths with frozen V2 images and references.

Defaults cover 518/392 and fp32/fp16 on development; final requires --split final.
This is a correctness run with actual device-input readback, not a timing run.
Example (WSL, after building test_depth_gpu_pipeline with GPU-path support)::

    python tools/validate_depth_gpu_pipeline.py --cli /path/to/test_depth_gpu_pipeline \
        --output docs/validation/v2_upgrade/v24/development
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from depth.metrics import numerical_metrics, ordering_metrics
from depth.preprocess import DepthError, contract_version
from depth.reference import sha256, write_json
from depth.validate_backend import (
    _aggregate, _array, _json, _ordering_total, _passes, load_thresholds, verify_reference,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "generated" / "depth"
MODES = ("A", "B", "C-basic", "C-fused")
PREPROCESS_ATOL = 2e-6
PREPROCESS_RTOL = 1e-6


def _manifest(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if not separator or not key or key in result:
            raise DepthError(f"invalid/duplicate engine manifest field: {path}")
        result[key] = value
    return result


def _new_ordering_errors(baseline: np.ndarray, candidate: np.ndarray, pairs: list) -> dict:
    """Count newly failed baseline-correct pairs; improvements cannot cancel them."""
    indices = []
    for index, pair in enumerate(pairs):
        if pair["status"] != "valid":
            continue
        before = ordering_metrics(baseline, [pair])
        after = ordering_metrics(candidate, [pair])
        if before["correct"] and not after["correct"]:
            indices.append(index)
    return {"count": len(indices), "pair_indices": indices, "passed": not indices}


def _numerical(expected: np.ndarray, actual: np.ndarray, thresholds: dict, preprocessing: bool) -> dict:
    metric = numerical_metrics(expected, actual,
        PREPROCESS_ATOL if preprocessing else thresholds["atol"],
        PREPROCESS_RTOL if preprocessing else thresholds["rtol"])
    metric["passed"] = metric["violations"] == 0 if preprocessing else _passes(metric, thresholds, True)
    return metric


def _read_outputs(directory: Path, identifier: str, width: int, height: int, size: int, hashes: dict) -> dict:
    result = {}
    for name, shape in (("nchw", (1, 3, size, size)), ("raw", (1, size, size)),
                        ("restored", (height, width))):
        path = directory / f"{identifier}.{name}.f32"
        result[name] = _array(path, shape)
        hashes[str(path)] = sha256(path)
    return result


def _run_configuration(args, size: int, precision: str, verified: list[dict], report: dict) -> dict:
    selected = [sample for sample in verified if sample["entry"]["split"] == args.split]
    if not selected:
        raise DepthError(f"empty requested split: {args.split}")
    engine = Path(getattr(args, f"engine_{size}_{precision}")).resolve(strict=True)
    engine_manifest_path = Path(str(engine) + ".manifest")
    engine_manifest = _manifest(engine_manifest_path)
    threshold_path = ROOT / "data/depth" / f"thresholds_trt_{precision}{'_392' if size == 392 else ''}.json"
    thresholds = load_thresholds(threshold_path, size)
    if thresholds["backend"] != f"tensorrt-{precision}":
        raise DepthError("precision thresholds have a different backend identity")
    if engine_manifest.get("precision") != precision or engine_manifest.get("contract") != contract_version(size):
        raise DepthError(f"engine precision/contract differs: {engine}")
    engine_hash = sha256(engine)
    if engine_manifest.get("engine_sha256") != engine_hash:
        raise DepthError(f"engine manifest SHA256 differs: {engine}")
    report["input_sha256"][str(engine)] = engine_hash
    for path in (engine_manifest_path, threshold_path):
        report["input_sha256"][str(path)] = sha256(path)
    name = f"{size}-{precision}"
    configuration = {"model_size": size, "precision": precision, "engine": str(engine),
                     "thresholds": thresholds, "samples": [], "aggregate": {}, "passed": False}
    report["configurations"].append(configuration)
    sample_list = Path(args.output).resolve() / f"{name}-samples.tsv"
    with sample_list.open("x", encoding="utf-8", newline="\n") as stream:
        for sample in selected:
            transform = sample["transform"]
            image = sample["directory"] / "original.ppm"
            # Send the hash-verified frozen PPM, ensuring byte-identical input on
            # every path. Preserve the original source identity separately.
            columns = [sample["id"], str(image), str(transform.original_width),
                       str(transform.original_height), sha256(image)]
            if any("\t" in column or "\n" in column or "\r" in column for column in columns):
                raise DepthError("sample path contains a TSV delimiter")
            stream.write("\t".join(columns) + "\n")
    artifact_directory = sample_list.parent / name
    command = [str(Path(args.cli).resolve(strict=True)), str(engine), str(size), precision,
               str(sample_list), str(artifact_directory), str(args.device)]
    run = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=args.timeout, check=False)
    configuration["command"] = {"argv": command, "returncode": run.returncode,
        "stdout": run.stdout[-8192:], "stderr": run.stderr[-8192:]}
    if run.returncode:
        raise DepthError(f"GPU artifact producer exited {run.returncode} for {name}: {run.stderr[-1000:]}")
    manifest_path = artifact_directory / "run.json"
    produced = _json(manifest_path)
    expected_samples = [{"id": sample["id"], "width": sample["transform"].original_width,
        "height": sample["transform"].original_height,
        "source_sha256": sha256(sample["directory"] / "original.ppm")} for sample in selected]
    if (produced.get("schema_version") != 1 or produced.get("model_size") != size
            or produced.get("precision") != precision or produced.get("engine_sha256") != engine_hash
            or produced.get("samples_list_sha256") != sha256(sample_list)
            or produced.get("captured_nchw") != "actual-device-input-readback"
            or produced.get("modes") != list(MODES) or produced.get("samples") != expected_samples):
        raise DepthError(f"artifact producer metadata differs from requested inputs: {name}")
    report["output_sha256"][str(manifest_path)] = sha256(manifest_path)
    report["input_sha256"][str(sample_list)] = sha256(sample_list)
    # The existing aggregate helper computes pixel-weighted MAE/RMSE and exact
    # global P99 from the recorded errors, never averages image percentiles.
    numerical_rows, absolute_errors = {}, {}
    ordering_rows = {baseline: {mode: [] for mode in MODES} for baseline in ("python", "A")}
    new_errors = {baseline: {mode: 0 for mode in MODES} for baseline in ("python", "A")}
    for sample in selected:
        identifier, transform = sample["id"], sample["transform"]
        width, height = transform.original_width, transform.original_height
        reference = {"nchw": _array(sample["directory"] / "preprocessed.f32", (1, 3, size, size)),
            "raw": _array(sample["directory"] / "raw.f32", (1, size, size)),
            "restored": _array(sample["directory"] / "restored.f32", (height, width))}
        cpu_path = artifact_directory / "cpu" / f"{identifier}.nchw.f32"
        cpu_nchw = _array(cpu_path, (1, 3, size, size))
        report["output_sha256"][str(cpu_path)] = sha256(cpu_path)
        actual = {mode: _read_outputs(artifact_directory / mode, identifier, width, height, size,
                                     report["output_sha256"]) for mode in MODES}
        pairs = sample["entry"].get("near_far_pairs", [])
        python_ordering = ordering_metrics(reference["restored"], pairs)
        a_ordering = ordering_metrics(actual["A"]["restored"], pairs)
        row = {"id": identifier, "source_sha256": sample["entry"].get("sha256"),
               "python_ordering": python_ordering, "A_ordering": a_ordering, "modes": {}, "passed": True}
        for mode in MODES:
            candidate = actual[mode]
            mode_row = {"numerical": {}, "ordering": ordering_metrics(candidate["restored"], pairs),
                        "new_ordering_errors": {}, "ordering_drop": {}, "passed": True}
            comparisons = {"cpu_input": (cpu_nchw, candidate["nchw"], True)}
            for baseline, values in (("python", reference), ("A", actual["A"])):
                for tensor in ("nchw", "raw", "restored"):
                    comparisons[f"{baseline}_{tensor}"] = (values[tensor], candidate[tensor], tensor == "nchw")
                mode_row["new_ordering_errors"][baseline] = _new_ordering_errors(
                    values["restored"], candidate["restored"], pairs)
                baseline_ordering = python_ordering if baseline == "python" else a_ordering
                drop = (baseline_ordering["accuracy"] - mode_row["ordering"]["accuracy"]
                        if baseline_ordering["total"] else None)
                mode_row["ordering_drop"][baseline] = drop
                mode_row["passed"] &= mode_row["new_ordering_errors"][baseline]["passed"]
                mode_row["passed"] &= drop is None or drop <= thresholds["max_ordering_drop"]
                ordering_rows[baseline][mode].append(mode_row["ordering"])
                new_errors[baseline][mode] += mode_row["new_ordering_errors"][baseline]["count"]
            for comparison, (expected, value, preprocessing) in comparisons.items():
                metric = _numerical(expected, value, thresholds, preprocessing)
                mode_row["numerical"][comparison] = metric
                mode_row["passed"] &= metric["passed"]
                key = (mode, comparison)
                numerical_rows.setdefault(key, []).append(metric)
                absolute_errors.setdefault(key, []).append(np.abs(value.astype(np.float64) - expected).ravel())
            row["modes"][mode] = mode_row
            row["passed"] &= mode_row["passed"]
        configuration["samples"].append(row)
    for mode in MODES:
        aggregate = {"numerical": {}, "ordering": _ordering_total(ordering_rows["python"][mode]),
                     "new_ordering_errors": {baseline: new_errors[baseline][mode] for baseline in new_errors},
                     "passed": all(row["modes"][mode]["passed"] for row in configuration["samples"])}
        for key, rows in numerical_rows.items():
            if key[0] != mode:
                continue
            metric = _aggregate(rows, absolute_errors[key])
            preprocessing = key[1] == "cpu_input" or key[1].endswith("_nchw")
            metric["passed"] = metric["violations"] == 0 if preprocessing else _passes(metric, thresholds, True)
            aggregate["numerical"][key[1]] = metric
            aggregate["passed"] &= metric["passed"]
        configuration["aggregate"][mode] = aggregate
    configuration["aggregate"]["python_ordering"] = _ordering_total(
        [row["python_ordering"] for row in configuration["samples"]])
    configuration["aggregate"]["A_ordering"] = _ordering_total(
        [row["A_ordering"] for row in configuration["samples"]])
    configuration["passed"] = all(configuration["aggregate"][mode]["passed"] for mode in MODES)
    return configuration


def validate(args) -> dict:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "split": args.split, "scope": "frozen 16-image V2 smoke subset; not official DA-2K score",
              "preprocess_tolerance": {"atol": PREPROCESS_ATOL, "rtol": PREPROCESS_RTOL},
              "new_ordering_errors_allowed": 0, "input_sha256": {}, "output_sha256": {},
              "configurations": [], "errors": [], "passed": False}
    try:
        if len(set(args.model_sizes)) != len(args.model_sizes) or len(set(args.precisions)) != len(args.precisions):
            raise DepthError("duplicate requested model size or precision")
        samples_path = Path(args.samples).resolve(strict=True)
        cli = Path(args.cli).resolve(strict=True)
        report["input_sha256"].update({str(samples_path): sha256(samples_path), str(cli): sha256(cli)})
        report["tools_sha256"] = {str(path.relative_to(ROOT)): sha256(path) for path in
            (Path(__file__), ROOT / "tests/test_depth_gpu_pipeline.cpp", ROOT / "tools/depth/metrics.py",
             ROOT / "tools/depth/preprocess.py", ROOT / "tools/depth/reference.py", ROOT / "tools/depth/validate_backend.py")}
        for size in args.model_sizes:
            reference = Path(getattr(args, f"reference_{size}")).resolve(strict=True)
            verified = verify_reference(reference, samples_path, report["input_sha256"], size)
            for precision in args.precisions:
                _run_configuration(args, size, precision, verified, report)
        report["passed"] = bool(report["configurations"]) and all(row["passed"] for row in report["configurations"])
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    write_json(output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", required=True, help="built test_depth_gpu_pipeline binary")
    parser.add_argument("--output", required=True, help="new exclusive output directory")
    parser.add_argument("--samples", default=str(ROOT / "data/depth/samples_v2.json"))
    parser.add_argument("--split", choices=("development", "final"), default="development")
    parser.add_argument("--model-sizes", type=int, choices=(518, 392), nargs="+", default=[518, 392])
    parser.add_argument("--precisions", choices=("fp32", "fp16"), nargs="+", default=["fp32", "fp16"])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=1800, help="seconds per persistent four-path invocation")
    parser.add_argument("--reference-518", default=str(DATA / "v22-reference-518"))
    parser.add_argument("--reference-392", default=str(DATA / "v22-export-392"))
    parser.add_argument("--engine-518-fp32", default=str(DATA / "engines/depth-fp32.engine"))
    parser.add_argument("--engine-518-fp16", default=str(DATA / "engines/depth-fp16-v2.engine"))
    parser.add_argument("--engine-392-fp32", default=str(DATA / "v22/engines/392-fp32.engine"))
    parser.add_argument("--engine-392-fp16", default=str(DATA / "v22/engines/392-fp16.engine"))
    args = parser.parse_args()
    if args.device < 0 or args.timeout <= 0:
        parser.error("device must be non-negative and timeout must be positive")
    try:
        report = validate(args)
    except Exception as error:
        print(json.dumps({"passed": False, "error": str(error)}), file=sys.stderr)
        return 1
    summary = {"passed": report["passed"], "split": args.split,
               "configurations": len(report["configurations"]), "errors": report["errors"],
               "report": str(Path(args.output).resolve() / "report.json")}
    print(json.dumps(summary, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
