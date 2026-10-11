#!/usr/bin/env python3
"""Validate a depth CLI against the frozen Python bundle on one named split.

No thresholds are inferred. Pass a frozen JSON threshold file. Report totals
weight numerical statistics by pixels and ordering accuracy by valid pairs.
Python .py CLIs are run with the current interpreter (useful for smoke tests).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

if __package__:
    from .metrics import numerical_metrics, ordering_metrics
    from .preprocess import DepthError, geometry, postprocess, read_ppm
    from .reference import CONTRACT, load_samples, sha256, write_json
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from depth.metrics import numerical_metrics, ordering_metrics
    from depth.preprocess import DepthError, geometry, postprocess, read_ppm
    from depth.reference import CONTRACT, load_samples, sha256, write_json


def _json(path: Path) -> dict:
    def reject(value):
        raise DepthError(f"non-finite JSON constant in {path}: {value}")
    def unique(items):
        value = {}
        for key, item in items:
            if key in value:
                raise DepthError(f"duplicate JSON key in {path}: {key}")
            value[key] = item
        return value
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject, object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise DepthError(f"expected JSON object: {path}")
    return value


def load_thresholds(path: Path) -> dict:
    value = _json(path)
    required = {"schema_version", "backend", "atol", "rtol", "max_violation_fraction",
                "preprocess_atol", "postprocess_atol", "max_ordering_drop"}
    allowed = required | {"max_p99_abs", "max_mae"}
    if set(value) - allowed or required - set(value):
        raise DepthError(f"threshold keys must be {sorted(required)}; optional max_p99_abs/max_mae")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise DepthError("threshold schema_version must be 1")
    if not isinstance(value["backend"], str) or not value["backend"].strip():
        raise DepthError("threshold backend must be a nonempty string")
    for name in allowed - {"schema_version", "backend"}:
        if name in value:
            number = value[name]
            if type(number) not in (int, float) or not np.isfinite(number) or number < 0:
                raise DepthError(f"threshold {name} must be finite and non-negative")
    for name in ("max_violation_fraction", "max_ordering_drop"):
        if value[name] > 1:
            raise DepthError(f"threshold {name} must be in [0,1]")
    return value


def _inside(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise DepthError("reference artifact path must be relative")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root.resolve()):
        raise DepthError(f"reference artifact escapes bundle: {relative}")
    return path


def _check_hash(path: Path, expected, hashes: dict) -> None:
    actual = sha256(path)
    hashes[str(path)] = actual
    if not isinstance(expected, str) or actual != expected:
        raise DepthError(f"SHA256 mismatch: {path}")


def verify_reference(root: Path, samples_path: Path, hashes: dict) -> list[dict]:
    """Verify all artifact hashes, including the split not executed by this run."""
    manifest_path = root / "manifest.json"
    hashes[str(manifest_path)] = sha256(manifest_path)
    manifest = _json(manifest_path)
    if (type(manifest.get("format_version")) is not int or manifest["format_version"] != 1
            or manifest.get("contract_version") != "depth-v1"):
        raise DepthError("reference must be format_version=1, contract_version=depth-v1")
    _check_hash(samples_path, manifest.get("samples_manifest_sha256"), hashes)
    _check_hash(_inside(root, manifest.get("samples_manifest_file", "samples.json")),
                manifest.get("samples_manifest_sha256"), hashes)
    contract_path = _inside(root, "contract.json")
    _check_hash(contract_path, manifest.get("contract_sha256"), hashes)
    hashes[str(CONTRACT)] = sha256(CONTRACT)
    if _json(contract_path) != _json(CONTRACT):
        raise DepthError("reference contract differs from current frozen contract")
    samples = load_samples(samples_path)
    current = {s["entry"]["id"]: s for s in samples}
    saved = manifest.get("samples")
    if not isinstance(saved, list) or not saved:
        raise DepthError("reference contains no samples")
    result, seen = [], set()
    for item in saved:
        if not isinstance(item, dict) or item.get("id") not in current or item["id"] in seen:
            raise DepthError("reference sample identifiers differ or are duplicated")
        identifier = item["id"]
        seen.add(identifier)
        sample = current[identifier]
        hashes[str(sample["path"])] = sample["sha256"]
        metadata_path = _inside(root, item.get("metadata"))
        _check_hash(metadata_path, item.get("metadata_sha256"), hashes)
        metadata = _json(metadata_path)
        entry, rgb = sample["entry"], sample["rgb"]
        transform = geometry(rgb.shape[1], rgb.shape[0])
        if (metadata.get("id") != identifier or metadata.get("split") != entry["split"]
                or metadata.get("source_sha256") != sample["sha256"]
                or metadata.get("near_far_pairs") != entry.get("near_far_pairs", [])
                or metadata.get("geometry") != transform.to_dict()):
            raise DepthError(f"reference sample metadata differs from current samples: {identifier}")
        for dimension, actual in (("width", rgb.shape[1]), ("height", rgb.shape[0])):
            if dimension in entry and entry[dimension] != actual:
                raise DepthError(f"sample {identifier}: declared {dimension} differs from image")
        files = metadata.get("files")
        required = {"original.ppm", "preprocessed.f32", "raw.f32", "restored.f32"}
        if not isinstance(files, dict) or not required.issubset(files):
            raise DepthError(f"reference sample {identifier} lacks required artifacts")
        for name, info in files.items():
            path = _inside(metadata_path.parent, name)
            if not isinstance(info, dict):
                raise DepthError(f"invalid artifact metadata: {name}")
            _check_hash(path, info.get("sha256"), hashes)
            if type(info.get("bytes")) is not int or path.stat().st_size != info["bytes"]:
                raise DepthError(f"reference byte count mismatch: {path}")
        original = read_ppm(metadata_path.parent / "original.ppm")
        if not np.array_equal(original, rgb):
            raise DepthError(f"reference original pixels differ: {identifier}")
        for name, shape in (("preprocessed.f32", (1, 3, 518, 518)),
                            ("raw.f32", (1, 518, 518)),
                            ("restored.f32", (rgb.shape[0], rgb.shape[1]))):
            info = files[name]
            if info.get("dtype") != "<f4" or info.get("shape") != list(shape) or info.get("layout") != "C":
                raise DepthError(f"reference tensor contract mismatch: {identifier}/{name}")
            _array(metadata_path.parent / name, shape)
        result.append({"id": identifier, "entry": entry, "directory": metadata_path.parent,
                       "transform": transform})
    if seen != set(current):
        raise DepthError("reference and current sample identifier sets differ")
    return result


def _array(path: Path, shape: tuple) -> np.ndarray:
    expected = int(np.prod(shape)) * 4
    if path.stat().st_size != expected:
        raise DepthError(f"tensor byte count mismatch: {path}; expected {expected}")
    result = np.fromfile(path, dtype="<f4").reshape(shape)
    if not np.isfinite(result).all():
        raise DepthError(f"non-finite tensor: {path}")
    return result


def _run(command: list[str], prefix: Path, transform, report: dict) -> dict:
    record = {"command": command, "returncode": None}
    report["commands"].append(record)
    completed = subprocess.run(command, capture_output=True, text=True, errors="replace", check=False)
    record.update(returncode=completed.returncode, stdout=completed.stdout[-16384:],
                  stderr=completed.stderr[-16384:])
    if completed.returncode:
        raise DepthError(f"CLI exited {completed.returncode}: {prefix.name}; {completed.stderr[-1000:]}")
    paths = {name: Path(str(prefix) + suffix) for name, suffix in
             (("preprocessed", ".preprocessed.f32"), ("raw", ".raw.f32"),
              ("restored", ".depth.f32"), ("metadata", ".json"))}
    metadata = _json(paths["metadata"])
    if metadata.get("backend", "").replace("_", "-") != report["thresholds"]["backend"].replace("_", "-"):
        raise DepthError(f"CLI backend differs from threshold identity: {prefix}")
    if metadata.get("geometry") != transform.to_dict():
        raise DepthError(f"CLI geometry differs: {prefix}")
    seconds = metadata.get("seconds")
    if not isinstance(seconds, dict) or not seconds:
        raise DepthError(f"CLI seconds must be a nonempty object: {prefix}")
    if any(type(v) not in (int, float) or not np.isfinite(v) or v < 0 for v in seconds.values()):
        raise DepthError(f"CLI timing must be finite and non-negative: {prefix}")
    record["output_sha256"] = {str(p): sha256(p) for p in paths.values()}
    record["seconds"] = seconds
    return {"preprocessed": _array(paths["preprocessed"], (1, 3, 518, 518)),
            "raw": _array(paths["raw"], (1, 518, 518)),
            "restored": _array(paths["restored"], (transform.original_height, transform.original_width))}


def _passes(metrics: dict, thresholds: dict, inference: bool) -> bool:
    allowed = thresholds["max_violation_fraction"] if inference else 0
    if metrics["violation_fraction"] > allowed:
        return False
    return not inference or all(metrics[metric] <= thresholds[threshold] for metric, threshold in
                                (("p99_abs", "max_p99_abs"), ("mae", "max_mae"))
                                if threshold in thresholds)


def _aggregate(values: list[dict], errors: list[np.ndarray]) -> dict:
    count = sum(v["count"] for v in values)
    violations = sum(v["violations"] for v in values)
    return {"count": count, "mae": sum(v["mae"] * (v["count"] / count) for v in values),
            "rmse": float(np.sqrt(sum(v["rmse"] ** 2 * (v["count"] / count) for v in values))),
            "max_abs": max(v["max_abs"] for v in values),
            "p99_abs": float(np.quantile(np.concatenate(errors), 0.99, method="linear")),
            "nonfinite_count": sum(v["nonfinite_count"] for v in values),
            "violations": violations, "violation_fraction": violations / count}


def _ordering_total(values: list[dict]) -> dict:
    result = {name: sum(v[name] for v in values) for name in ("correct", "total", "ignored", "ties")}
    result["accuracy"] = result["correct"] / result["total"] if result["total"] else None
    return result


def validate(args: argparse.Namespace) -> dict:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "split": args.split, "input_sha256": {}, "commands": [], "samples": [],
              "aggregate": {}, "errors": [], "passed": False}
    try:
        paths = {name: Path(getattr(args, name)).resolve(strict=True)
                 for name in ("cli", "model", "reference", "samples", "thresholds")}
        report["inputs"] = {name: str(path) for name, path in paths.items()}
        for name in ("cli", "model", "samples", "thresholds"):
            report["input_sha256"][str(paths[name])] = sha256(paths[name])
        report["tools_sha256"] = {p.name: sha256(p) for p in (Path(__file__),
                                      Path(__file__).with_name("metrics.py"),
                                      Path(__file__).with_name("preprocess.py"),
                                      Path(__file__).with_name("reference.py"))}
        thresholds = load_thresholds(paths["thresholds"])
        report["thresholds"] = thresholds
        samples = verify_reference(paths["reference"], paths["samples"], report["input_sha256"])
        selected = [s for s in samples if s["entry"]["split"] == args.split]
        if not selected:
            raise DepthError(f"no samples in split {args.split}")
        extra = list(args.extra_arg or [])
        reserved = {"--model", "--image", "--tensor", "--width", "--height", "--output"}
        if any(v.split("=", 1)[0] in reserved for v in extra):
            raise DepthError("extra arguments must not override validation model/input/output")
        cli = [str(paths["cli"])]
        if paths["cli"].suffix.lower() == ".py":
            cli.insert(0, sys.executable)
        errors = {}
        for sample in selected:
            identifier, transform, directory = sample["id"], sample["transform"], sample["directory"]
            row = {"id": identifier, "split": args.split, "numerical": {}, "passed": False}
            report["samples"].append(row)
            destination = output / identifier
            destination.mkdir()
            base = cli + ["--model", str(paths["model"])] + extra
            image_prefix, tensor_prefix = destination / "image", destination / "tensor"
            image = _run(base + ["--image", str(directory / "original.ppm"), "--output", str(image_prefix)],
                         image_prefix, transform, report)
            tensor = _run(base + ["--tensor", str(directory / "preprocessed.f32"),
                                  "--width", str(transform.original_width), "--height", str(transform.original_height),
                                  "--output", str(tensor_prefix)], tensor_prefix, transform, report)
            reference = {"preprocessed": _array(directory / "preprocessed.f32", (1, 3, 518, 518)),
                         "raw": _array(directory / "raw.f32", (1, 518, 518)),
                         "restored": _array(directory / "restored.f32", (transform.original_height, transform.original_width))}
            if not np.array_equal(tensor["preprocessed"], reference["preprocessed"]):
                raise DepthError(f"same-input CLI changed supplied tensor: {identifier}")
            layers = {
                "preprocessing": (reference["preprocessed"], image["preprocessed"], "preprocess_atol", False),
                "raw": (reference["raw"], image["raw"], "atol", True),
                "restored": (reference["restored"], image["restored"], "atol", True),
                "postprocessing": (postprocess(image["raw"], transform), image["restored"], "postprocess_atol", False),
                "same_input_raw": (reference["raw"], tensor["raw"], "atol", True),
                "raw_path_consistency": (image["raw"], tensor["raw"], "atol", True),
                "same_input_restored": (reference["restored"], tensor["restored"], "atol", True),
                "same_input_postprocessing": (postprocess(tensor["raw"], transform), tensor["restored"], "postprocess_atol", False),
            }
            for name, (expected, actual, tolerance, inference) in layers.items():
                metric = numerical_metrics(expected, actual, thresholds[tolerance], thresholds["rtol"] if inference else 0)
                metric["passed"] = _passes(metric, thresholds, inference)
                row["numerical"][name] = metric
                errors.setdefault(name, []).append(np.abs(actual.astype(np.float64) - expected.astype(np.float64)).ravel())
            pairs = sample["entry"].get("near_far_pairs", [])
            row["near_far_pairs"] = pairs
            row["ordering"] = {"python": ordering_metrics(reference["restored"], pairs),
                               "candidate": ordering_metrics(image["restored"], pairs),
                               "same_input_candidate": ordering_metrics(tensor["restored"], pairs)}
            row["pair_results"] = []
            for pair in pairs:
                nx, ny = pair["near"]
                fx, fy = pair["far"]
                result = {"near": pair["near"], "far": pair["far"], "status": pair["status"]}
                for backend, depth in (("python", reference["restored"]), ("candidate", image["restored"]),
                                       ("same_input_candidate", tensor["restored"])):
                    near, far = float(depth[ny, nx]), float(depth[fy, fx])
                    result[backend] = {"near_value": near, "far_value": far,
                                       "correct": near > far if pair["status"] == "valid" else None,
                                       "tie": near == far if pair["status"] == "valid" else None}
                row["pair_results"].append(result)
            accuracies = row["ordering"]
            drop = (accuracies["python"]["accuracy"] - accuracies["candidate"]["accuracy"]
                    if accuracies["python"]["total"] else None)
            row["ordering_drop"] = drop
            row["same_input_ordering_drop"] = (accuracies["python"]["accuracy"]
                - accuracies["same_input_candidate"]["accuracy"] if accuracies["python"]["total"] else None)
            row["ordering_passed"] = all(v is None or v <= thresholds["max_ordering_drop"]
                                         for v in (drop, row["same_input_ordering_drop"]))
            row["passed"] = all(m["passed"] for m in row["numerical"].values()) and row["ordering_passed"]
        aggregate = report["aggregate"]
        aggregate["numerical"] = {name: _aggregate([s["numerical"][name] for s in report["samples"]], chunks)
                                  for name, chunks in errors.items()}
        for name, metric in aggregate["numerical"].items():
            metric["passed"] = _passes(metric, thresholds, name not in
                                       ("preprocessing", "postprocessing", "same_input_postprocessing"))
        aggregate["ordering"] = {backend: _ordering_total([s["ordering"][backend] for s in report["samples"]])
                                 for backend in ("python", "candidate", "same_input_candidate")}
        python, candidate = aggregate["ordering"]["python"], aggregate["ordering"]["candidate"]
        aggregate["ordering_drop"] = python["accuracy"] - candidate["accuracy"] if python["total"] else None
        aggregate["same_input_ordering_drop"] = (python["accuracy"]
            - aggregate["ordering"]["same_input_candidate"]["accuracy"] if python["total"] else None)
        aggregate["ordering_passed"] = all(v is None or v <= thresholds["max_ordering_drop"]
                                           for v in (aggregate["ordering_drop"], aggregate["same_input_ordering_drop"]))
        report["passed"] = (all(s["passed"] for s in report["samples"])
                            and all(m["passed"] for m in aggregate["numerical"].values())
                            and aggregate["ordering_passed"])
    except Exception as exc:
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    write_json(output / "report.json", report)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cli", "model", "reference", "samples", "thresholds", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--split", required=True, choices=("development", "final"))
    parser.add_argument("--extra-arg", action="append", default=[], help="repeat; use --extra-arg=--flag for flags")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        report = validate(args)
    except Exception as exc:
        print(f"depth validation failed before report creation: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"report": str(Path(args.output).resolve() / "report.json"), "passed": report["passed"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
