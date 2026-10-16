#!/usr/bin/env python3
"""Run a standalone SDK CLI against its package's golden image and thresholds."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from depth.metrics import numerical_metrics, ordering_metrics
from depth.preprocess import Geometry, contract_version, geometry_model_size, validate_model_size
from depth.reference import sha256, write_json
from depth.validate_backend import load_thresholds
from package_depth import verify

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-size", type=int, choices=(518, 392), help="assert package model size")
    parser.add_argument("--preprocess", choices=("cpu", "cuda-partial", "cuda-full-basic", "cuda-full-fused"), default="cpu")
    parser.add_argument("--slots", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    root = Path(args.package).resolve(strict=True)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"passed": False, "errors": []}
    try:
        package = verify(root)
        model_size = validate_model_size(package.get("model_size", 518))
        if args.model_size is not None and args.model_size != model_size:
            raise ValueError("requested model_size differs from SDK package")
        report.update(model_size=model_size, contract=contract_version(model_size),
                      reference_manifest_sha256=sha256(root / "reference-manifest.json"))
        thresholds = load_thresholds(root / "thresholds.json", model_size)
        metadata = json.loads((root / "golden/metadata.json").read_text(encoding="utf-8"))
        transform = metadata["geometry"]
        if geometry_model_size(Geometry(**transform)) != model_size:
            raise ValueError("golden geometry differs from SDK package model_size")
        shape = (transform["original_height"], transform["original_width"])
        target = output / "sdk.f32"
        command = [str(Path(args.cli).resolve(strict=True)), str(root), str(root / "golden/original.ppm"), str(target)]
        if args.preprocess != "cpu" or args.slots != 1:
            command += [args.preprocess, str(args.slots)]
        run = subprocess.run(command, capture_output=True, text=True, timeout=180)
        report.update(command=command, returncode=run.returncode, stdout=run.stdout, stderr=run.stderr,
                      cli_sha256=sha256(Path(args.cli)), package_sha256=sha256(root / "package.json"))
        if run.returncode:
            raise ValueError("SDK CLI failed")
        reference = np.fromfile(root / "golden/restored.f32", dtype="<f4").reshape(shape)
        candidate = np.fromfile(target, dtype="<f4").reshape(shape)
        metric = numerical_metrics(reference, candidate, thresholds["atol"], thresholds["rtol"])
        pairs = metadata["near_far_pairs"]
        baseline, actual = ordering_metrics(reference, pairs), ordering_metrics(candidate, pairs)
        passed = metric["violation_fraction"] <= thresholds["max_violation_fraction"]
        passed &= all(metric[key] <= thresholds[bound] for key, bound in
                      (("mae", "max_mae"), ("p99_abs", "max_p99_abs")) if bound in thresholds)
        if baseline["total"]:
            passed &= baseline["accuracy"] - actual["accuracy"] <= thresholds["max_ordering_drop"]
        report.update(passed=bool(passed), numerical=metric, python_ordering=baseline,
                      sdk_ordering=actual, output_sha256=sha256(target))
    except Exception as error:
        report["errors"].append(str(error))
    write_json(output / "report.json", report)
    print(json.dumps({"passed": report["passed"], "report": str(output / "report.json")}))
    return 0 if report["passed"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
