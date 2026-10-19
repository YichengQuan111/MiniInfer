#!/usr/bin/env python3
"""Bundle an installed SDK, verified model and passed V2 deployment evidence.

External CUDA/TensorRT/ORT libraries and the NVIDIA driver remain dependencies.
The selected runtime configuration is explicit documentation, not auto-loaded
by DepthSdk. All destinations are new; existing deliveries are never replaced.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from depth.reference import sha256, write_json
from package_depth import verify


def assemble(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("delivery output already exists")
    sdk = Path(args.sdk_install).resolve(strict=True)
    package = Path(args.package).resolve(strict=True)
    if not (sdk / "lib/cmake/MiniInfer/MiniInferConfig.cmake").is_file():
        raise ValueError("SDK install has no exported CMake package")
    for executable in ("depth_sdk_cli", "depth_server", "depth_client", "depth_replay"):
        if not (sdk / "bin" / executable).is_file():
            raise ValueError("SDK install missing " + executable)
    metadata = verify(package)
    evidence = {name: Path(getattr(args, name)).resolve(strict=True)
                for name in ("selection", "quality", "stability")}
    selection, quality, stability = [json.loads(evidence[name].read_text(encoding="utf-8"))
                                     for name in ("selection", "quality", "stability")]
    if selection["model_size"] != metadata["model_size"] or selection["precision"] != metadata["precision"]:
        raise ValueError("runtime selection differs from model package")
    if quality.get("passed") is not True or quality.get("split") != "final":
        raise ValueError("final pipeline quality evidence must pass")
    mode = {"cpu": "A", "cuda-partial": "B", "cuda-full-basic": "C-basic", "cuda-full-fused": "C-fused"}[selection["preprocess_mode"]]
    matching = [row for row in quality["configurations"] if row["model_size"] == selection["model_size"]
                and row["precision"] == selection["precision"]]
    if len(matching) != 1 or matching[0]["aggregate"][mode]["passed"] is not True:
        raise ValueError("selected model/path lacks passing final evidence")
    engine_hash = sha256(package / "model.engine")
    if quality["input_sha256"].get(matching[0]["engine"]) != engine_hash or selection["engine_sha256"] != engine_hash:
        raise ValueError("quality/selection evidence names a different engine")
    if stability.get("passed") is not True or stability.get("require_soak") is not True:
        raise ValueError("passed measured long-soak evidence is required")
    if stability["case"]["preprocess"] != selection["preprocess_mode"] or stability["case"]["slots"] != selection["gpu_slots"]:
        raise ValueError("long-soak configuration differs from selection")
    for selected_key, measured_key in (("backend", "backend"), ("worker_count", "sdk_workers"),
            ("queue_capacity", "queue"), ("byte_budget", "budget"),
            ("reuse_host_workspace", "reuse_host"), ("reuse_gpu_buffers", "reuse_gpu")):
        if stability["case"].get(measured_key) != selection[selected_key]:
            raise ValueError("long-soak configuration differs: " + selected_key)
    measured_engine = [row for row in stability["identity"]["package_files"] if row["path"] == "model.engine"]
    if len(measured_engine) != 1 or measured_engine[0]["sha256"] != engine_hash:
        raise ValueError("long-soak evidence names a different engine")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(sdk, output / "sdk")
    shutil.copytree(package, output / "model")
    (output / "evidence").mkdir()
    for name, path in evidence.items():
        shutil.copyfile(path, output / "evidence" / (name + ".json"))
    (output / "README.txt").write_text(
        "MiniInfer V2 deployment bundle\n"
        "sdk/: installed C++ headers, libraries, executables, CMake export and documentation.\n"
        "model/: verified environment-specific TensorRT engine, ONNX, manifests and golden data.\n"
        "evidence/selection.json: explicit SdkConfig/CLI selection; not loaded automatically.\n"
        "Read sdk/share/MiniInfer/docs/depth_deployment.md for dependency and rebuild instructions.\n"
        "External CUDA runtime, TensorRT, ONNX Runtime and NVIDIA driver are not bundled.\n"
        "Rebuild and repeat quality/performance checks when the target environment differs.\n"
        "inventory.json hashes detect accidental corruption; they are not signatures.\n", encoding="utf-8")
    files = {path.relative_to(output).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path)}
             for path in sorted(output.rglob("*")) if path.is_file()}
    manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "selection": selection, "files": files}
    write_json(output / "inventory.json", manifest)
    # Read back the completed bundle, including copied evidence and executables.
    for name, info in files.items():
        path = output / name
        if path.stat().st_size != info["bytes"] or sha256(path) != info["sha256"]:
            raise ValueError("delivery copy/hash verification failed: " + name)
    print(json.dumps({"passed": True, "output": str(output), "files": len(files),
                      "inventory_sha256": sha256(output / "inventory.json")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sdk-install", "package", "selection", "quality", "stability", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    assemble(args)


if __name__ == "__main__":
    main()
