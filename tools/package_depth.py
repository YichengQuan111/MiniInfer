#!/usr/bin/env python3
"""Create an exclusive, verifiable SDK package from a passed final-split report."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile

from depth.reference import CONTRACT, sha256, write_json
from depth.validate_backend import load_thresholds, verify_reference


def read_manifest(path):
    entries = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split("=", 1)
        if key in entries or not value:
            raise ValueError("invalid/duplicate engine manifest entry")
        entries[key] = value
    return entries


def create(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("package output already exists")
    model, onnx = Path(args.model).resolve(strict=True), Path(args.onnx).resolve(strict=True)
    reference, samples = Path(args.reference).resolve(strict=True), Path(args.samples).resolve(strict=True)
    thresholds_path, report_path = Path(args.thresholds).resolve(strict=True), Path(args.report).resolve(strict=True)
    thresholds = load_thresholds(thresholds_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_backend = "ort-cpu-fp32" if args.backend == "ort" else "tensorrt-" + args.precision
    if args.backend == "ort" and args.precision != "fp32":
        raise ValueError("ORT package supports CPU FP32 only")
    if report.get("passed") is not True or report.get("split") != "final" or report.get("thresholds") != thresholds:
        raise ValueError("package requires the matching passed final-split report")
    if thresholds["backend"] != expected_backend:
        raise ValueError("threshold backend/precision mismatch")
    saved_hashes = report["input_sha256"]
    for key, path in (("model", model), ("samples", samples), ("thresholds", thresholds_path)):
        if saved_hashes.get(report["inputs"][key]) != sha256(path):
            raise ValueError("validated " + key + " changed")
    verified = verify_reference(reference, samples, {})
    if {s["id"] for s in verified if s["entry"]["split"] == "final"} != {s["id"] for s in report["samples"]}:
        raise ValueError("final report sample set mismatch")
    engine_metadata = None
    if args.backend == "trt":
        engine_metadata = read_manifest(Path(str(model) + ".manifest"))
        if (engine_metadata.get("precision") != args.precision or engine_metadata.get("contract") != "depth-v1"
                or engine_metadata.get("engine_sha256") != sha256(model)
                or engine_metadata.get("onnx_sha256") != sha256(onnx)):
            raise ValueError("engine metadata/hash mismatch")
    elif model != onnx and sha256(model) != sha256(onnx):
        raise ValueError("ORT model must be the declared ONNX")
    # Portable filenames deliberately have no path components in SDK manifest.
    model_name = "model.onnx" if args.backend == "ort" else "model.engine"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="depth-package-", dir=output.parent) as temporary:
        stage = Path(temporary)
        shutil.copyfile(model, stage / model_name)
        if args.backend == "trt":
            shutil.copyfile(onnx, stage / "model.onnx")
            shutil.copyfile(str(model) + ".manifest", stage / "model.engine.manifest")
            shutil.copyfile(str(model) + ".layers.json", stage / "model.engine.layers.json")
        shutil.copyfile(CONTRACT, stage / "contract.json")
        shutil.copyfile(thresholds_path, stage / "thresholds.json")
        shutil.copyfile(report_path, stage / "validation.json")
        shutil.copyfile(reference / "manifest.json", stage / "reference-manifest.json")
        shutil.copyfile(samples, stage / "samples.json")
        golden = next(s for s in verified if s["id"] == args.golden and s["entry"]["split"] == "final")
        (stage / "golden").mkdir()
        for filename in ("original.ppm", "preprocessed.f32", "raw.f32", "restored.f32", "metadata.json"):
            shutil.copyfile(golden["directory"] / filename, stage / "golden" / filename)
        manifest = {"schema": "miniinfer-model-1", "contract": "depth-v1", "backend": args.backend,
                    "precision": args.precision, "model": model_name, "model_sha256": sha256(stage / model_name)}
        (stage / "manifest.txt").write_text("".join(f"{k}={v}\n" for k, v in sorted(manifest.items())), encoding="utf-8")
        (stage / "README.txt").write_text(
            "MiniInfer depth-v1 model package\n"
            "Input: owning RGB8 HWC; max side4096, max pixels2097152. Output: FP32 HW relative inverse depth.\n"
            "Larger means nearer; not meters. Validation uses fixed 518 model input.\n"
            "SDK: DepthSdk(SdkConfig{package_dir=THIS_DIRECTORY}); run(image); shutdown().\n"
            "TensorRT engines are environment-specific; manifest checks happen before loading.\n"
            "Rebuild: depth_engine_builder --onnx model.onnx --output rebuilt.engine --precision " + args.precision + "\n"
            "Revalidate rebuilt.engine using the frozen reference/thresholds, then create a new package.\n"
            "Source/model: https://github.com/DepthAnything/Depth-Anything-V2 (Small, Apache-2.0).\n"
            "Golden data: https://huggingface.co/datasets/depth-anything/DA-2K ; see samples.json for provenance.\n"
            "Hashes detect corruption; this is not a signed trust boundary. Only load trusted model packages.\n", encoding="utf-8")
        inventory = {str(p.relative_to(stage)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                     for p in sorted(stage.rglob("*")) if p.is_file()}
        write_json(stage / "package.json", {"schema_version": 1, "contract": "depth-v1",
                   "backend": args.backend, "precision": args.precision, "golden": args.golden,
                   "onnx_sha256": sha256(onnx), "engine_environment": engine_metadata,
                   "files": inventory})
        output.mkdir(exist_ok=False)
        for item in stage.iterdir():
            shutil.move(str(item), output / item.name)
    print(json.dumps({"package": str(output), "files": len(inventory), "passed_final_report": True}))


def verify(path):
    root = Path(path).resolve(strict=True)
    metadata = json.loads((root / "package.json").read_text(encoding="utf-8"))
    for name, expected in metadata["files"].items():
        file = (root / name).resolve(strict=True)
        if not file.is_relative_to(root) or not file.is_file():
            raise ValueError("package inventory path escapes root")
        if file.stat().st_size != expected["bytes"] or sha256(file) != expected["sha256"]:
            raise ValueError("package content mismatch: " + name)
    print(json.dumps({"package": str(root), "verified_files": len(metadata["files"])}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("create")
    for name in ("model", "onnx", "output", "reference", "samples", "thresholds", "report"):
        build.add_argument("--" + name, required=True)
    build.add_argument("--backend", choices=("ort", "trt"), required=True)
    build.add_argument("--precision", choices=("fp32", "fp16"), required=True)
    build.add_argument("--golden", default="indoor_1")
    check = sub.add_parser("verify")
    check.add_argument("path")
    args = parser.parse_args()
    if args.command == "create": create(args)
    else: verify(args.path)


if __name__ == "__main__":
    main()
