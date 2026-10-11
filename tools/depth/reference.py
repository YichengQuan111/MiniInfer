#!/usr/bin/env python3
"""Reproduce the project depth-v1 baseline and static ONNX feasibility export.

Example: python tools/depth/reference.py reference --repo /path/to/official/repo
 --weights /path/to/depth_anything_v2_vits.pth --samples data/depth/samples.json
 --output data/generated/depth/reference-v1
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

if __package__:
    from .preprocess import DepthError, MODEL_SIZE, postprocess, ppm_bytes, preprocess, read_image
else:
    from preprocess import DepthError, MODEL_SIZE, postprocess, ppm_bytes, preprocess, read_image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = PROJECT_ROOT / "data" / "depth" / "contract.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def write_array(path: Path, value: np.ndarray, dtype: str) -> dict:
    if value.dtype.kind == "f" and not np.isfinite(value).all():
        raise DepthError(f"refusing non-finite tensor: {path.name}")
    with path.open("xb") as stream:
        stream.write(np.ascontiguousarray(value, dtype=dtype).tobytes(order="C"))
    return {"dtype": dtype, "shape": list(value.shape), "bytes": path.stat().st_size,
            "sha256": sha256(path), "layout": "C"}


def load_samples(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise DepthError("samples manifest must have schema_version=1")
    samples = document.get("samples")
    if not isinstance(samples, list) or not samples:
        raise DepthError("samples manifest must contain a nonempty samples list")
    identifiers = set()
    validated = []
    for sample in samples:
        if not isinstance(sample, dict):
            raise DepthError("each sample must be an object")
        identifier = sample.get("id")
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", identifier):
            raise DepthError("sample id must be 1-64 ASCII letters, digits, underscores or hyphens")
        if identifier in identifiers:
            raise DepthError(f"duplicate sample id: {identifier}")
        identifiers.add(identifier)
        if sample.get("split") not in ("development", "final"):
            raise DepthError(f"sample {identifier}: split must be development or final")
        if not isinstance(sample.get("path"), str) or not sample["path"]:
            raise DepthError(f"sample {identifier}: path must be a nonempty string")
        image_path = Path(sample["path"])
        if not image_path.is_absolute():
            image_path = path.parent / image_path
        image_path = image_path.resolve(strict=True)
        digest = sha256(image_path)
        expected = sample.get("sha256")
        if expected is not None and (not isinstance(expected, str) or digest != expected.lower()):
            raise DepthError(f"sample {identifier}: source SHA256 mismatch")
        pairs = sample.get("near_far_pairs", [])
        if not isinstance(pairs, list):
            raise DepthError(f"sample {identifier}: near_far_pairs must be a list")
        rgb = read_image(image_path)
        for pair in pairs:
            if not isinstance(pair, dict) or pair.get("status") not in ("valid", "uncertain"):
                raise DepthError(f"sample {identifier}: pair must have valid or uncertain status")
            for name in ("near", "far"):
                point = pair.get(name)
                if not isinstance(point, list) or len(point) != 2 or any(type(v) is not int for v in point):
                    raise DepthError(f"sample {identifier}: {name} must be integer [x,y]")
                if not 0 <= point[0] < rgb.shape[1] or not 0 <= point[1] < rgb.shape[0]:
                    raise DepthError(f"sample {identifier}: pair point outside original image")
            if pair["near"] == pair["far"]:
                raise DepthError(f"sample {identifier}: pair points must differ")
        validated.append({"entry": sample, "path": image_path, "sha256": digest, "rgb": rgb})
    return validated


def source_provenance(repo: Path, explicit_commit: str | None = None) -> dict:
    if not (repo / "depth_anything_v2" / "dpt.py").is_file():
        raise DepthError("--repo must point to the official Depth-Anything-V2 source checkout")
    def git(*args: str) -> str:
        result = subprocess.run(["git", "-C", str(repo), *args], check=True,
                                capture_output=True, text=True)
        return result.stdout.strip()
    if (repo / ".git").exists():
        commit = git("rev-parse", "HEAD")
        if explicit_commit is not None and explicit_commit != commit:
            raise DepthError("--repo-commit differs from source checkout HEAD")
        status = git("status", "--porcelain")
        commit_evidence = "git_HEAD"
    else:
        if explicit_commit is None:
            raise DepthError("source archive has no .git; pass --repo-commit for its pinned official revision")
        commit, status, commit_evidence = explicit_commit, None, "caller_pinned_archive_revision"
    if not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        raise DepthError("official source checkout has an invalid Git commit")
    return {"repo_path": str(repo), "commit": commit,
            "git_status": status, "commit_evidence": commit_evidence,
            "source_sha256": {str(p.relative_to(repo)): sha256(p)
                              for p in sorted((repo / "depth_anything_v2").rglob("*.py"))}}


def load_model(repo: Path, weights: Path, threads: int):
    try:
        import torch
    except ImportError as exc:
        raise DepthError("PyTorch is required for reference/export; NumPy transforms do not need it") from exc
    torch.set_num_threads(threads)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    sys.path.insert(0, str(repo))
    module = importlib.import_module("depth_anything_v2.dpt")
    if not Path(module.__file__).resolve().is_relative_to(repo):
        raise DepthError("imported Depth Anything source is outside --repo")
    model = module.DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    state = torch.load(str(weights), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return torch, model.to(device="cpu", dtype=torch.float32).eval()


def generate(args: argparse.Namespace) -> dict:
    output = Path(args.output).resolve()
    if output.exists():
        raise DepthError(f"output path already exists; refusing to overwrite: {output}")
    repo = Path(args.repo).resolve(strict=True)
    weights = Path(args.weights).resolve(strict=True)
    samples_path = Path(args.samples).resolve(strict=True)
    samples = load_samples(samples_path)
    provenance = source_provenance(repo, args.repo_commit)
    weight_hash = sha256(weights)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if provenance["commit"] != contract["model"]["repository_commit"]:
        raise DepthError("source revision differs from frozen depth-v1 contract")
    if weight_hash != contract["model"]["weights_sha256"]:
        raise DepthError("weight SHA256 differs from frozen depth-v1 contract")
    reference_bundle = None
    if args.reference_dir is not None:
        reference_path = Path(args.reference_dir).resolve(strict=True)
        reference_manifest_path = reference_path / "manifest.json"
        previous = json.loads(reference_manifest_path.read_text(encoding="utf-8"))
        if previous.get("model", {}).get("weights_sha256") != weight_hash:
            raise DepthError("reference bundle weights differ from --weights")
        if previous.get("model", {}).get("source_sha256") != provenance["source_sha256"]:
            raise DepthError("reference bundle official source differs from --repo")
        if previous.get("samples_manifest_sha256") != sha256(samples_path):
            raise DepthError("reference bundle samples differ from --samples")
        if sha256(reference_path / "contract.json") != previous.get("contract_sha256"):
            raise DepthError("reference bundle contract hash mismatch")
        if json.loads((reference_path / "contract.json").read_text(encoding="utf-8")) != json.loads(CONTRACT.read_text(encoding="utf-8")):
            raise DepthError("reference bundle preprocessing contract differs from current contract")
        if not previous.get("samples"):
            raise DepthError("reference bundle contains no saved samples")
        for saved in previous["samples"]:
            metadata_path = reference_path / saved["metadata"]
            if sha256(metadata_path) != saved["metadata_sha256"]:
                raise DepthError("reference bundle sample metadata hash mismatch")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            for filename, entry in metadata["files"].items():
                if sha256(metadata_path.parent / filename) != entry["sha256"]:
                    raise DepthError(f"reference bundle tensor hash mismatch: {filename}")
        reference_bundle = {"path": str(reference_path), "manifest_sha256": sha256(reference_manifest_path),
                            "samples": len(previous["samples"])}
    torch, model = load_model(repo, weights, args.torch_threads)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".depth-reference-", dir=output.parent) as temporary:
        stage = Path(temporary)
        (stage / "samples").mkdir()
        write_json(stage / "contract.json", contract)
        shutil.copyfile(samples_path, stage / "samples.json")
        entries = []
        for sample in ([] if reference_bundle is not None else samples):
            entry, rgb = sample["entry"], sample["rgb"]
            directory = stage / "samples" / entry["id"]
            directory.mkdir()
            tensor, transform = preprocess(rgb)
            start = time.perf_counter()
            with torch.no_grad():
                raw = model(torch.from_numpy(tensor)).detach().cpu().numpy()
            inference_seconds = time.perf_counter() - start
            restored = postprocess(raw, transform)
            files = {"original.rgb8": write_array(directory / "original.rgb8", rgb, "|u1"),
                     "preprocessed.f32": write_array(directory / "preprocessed.f32", tensor, "<f4"),
                     "raw.f32": write_array(directory / "raw.f32", raw, "<f4"),
                     "restored.f32": write_array(directory / "restored.f32", restored, "<f4")}
            (directory / "original.ppm").write_bytes(ppm_bytes(rgb))
            files["original.ppm"] = {"sha256": sha256(directory / "original.ppm"),
                                     "bytes": (directory / "original.ppm").stat().st_size}
            pairs = entry.get("near_far_pairs", [])
            results = []
            for pair in pairs:
                if pair["status"] == "valid":
                    nx, ny = pair["near"]
                    fx, fy = pair["far"]
                    near, far = float(restored[ny, nx]), float(restored[fy, fx])
                    results.append({"near": pair["near"], "far": pair["far"],
                                    "near_value": near, "far_value": far, "correct": near > far})
            metadata = {"id": entry["id"], "split": entry["split"],
                        "source_path": str(sample["path"]), "source_sha256": sample["sha256"],
                        "source_info": entry.get("source"),
                        "geometry": transform.to_dict(), "files": files,
                        "cpu_model_seconds": inference_seconds,
                        "near_far_pairs": pairs, "ordering_results": results,
                        "task_semantics_evidence": "annotated_pairs" if results else "not_verified_no_valid_pairs"}
            write_json(directory / "metadata.json", metadata)
            entries.append({"id": entry["id"], "metadata": f"samples/{entry['id']}/metadata.json",
                            "metadata_sha256": sha256(directory / "metadata.json")})
        export = None
        if args.command == "export":
            import onnx
            onnx_path = stage / "depth_anything_v2_vits_518.onnx"
            with torch.no_grad():
                torch.onnx.export(model, torch.zeros((1, 3, MODEL_SIZE, MODEL_SIZE), dtype=torch.float32),
                                  str(onnx_path), input_names=["image"], output_names=["depth"],
                                  opset_version=17, dynamic_axes=None, dynamo=False,
                                  do_constant_folding=True)
            graph = onnx.load(str(onnx_path), load_external_data=False)
            onnx.checker.check_model(graph)
            inputs, outputs = graph.graph.input, graph.graph.output
            shape = lambda value: [dim.dim_value for dim in value.type.tensor_type.shape.dim]
            if len(inputs) != 1 or inputs[0].name != "image" or shape(inputs[0]) != [1, 3, 518, 518]:
                raise DepthError("exported ONNX input violates the static contract")
            if len(outputs) != 1 or outputs[0].name != "depth" or shape(outputs[0]) != [1, 518, 518]:
                raise DepthError("exported ONNX output violates the static contract")
            if any(v.type.tensor_type.elem_type != onnx.TensorProto.FLOAT for v in [inputs[0], outputs[0]]):
                raise DepthError("exported ONNX must expose float32 input/output")
            export = {"file": onnx_path.name, "sha256": sha256(onnx_path),
                      "bytes": onnx_path.stat().st_size, "opset": 17, "dynamo": False,
                      "status": "feasibility_only", "checker": "passed",
                      "backend_numerical_validation": "pending_V1.2"}
        manifest = {"format_version": 1, "contract_version": contract["contract_version"],
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "mode": args.command, "model": {"name": "Depth Anything V2 Small", "encoder": "vits",
                    "features": 64, "out_channels": [48, 96, 192, 384],
                    "weights_path": str(weights), "weights_sha256": weight_hash, **provenance},
                    "environment": {"python": platform.python_version(), "platform": platform.platform(),
                    "numpy": np.__version__, "torch": torch.__version__, "device": "cpu",
                    "dtype": "float32", "eval": True, "no_grad": True, "tf32": False,
                    "deterministic_algorithms": True, "torch_threads": args.torch_threads},
                    "contract_sha256": sha256(stage / "contract.json"),
                    "samples_manifest_path": str(samples_path), "samples_manifest_file": "samples.json",
                    "samples_manifest_sha256": sha256(stage / "samples.json"),
                    "tools_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                    "samples": entries, "reference_bundle": reference_bundle, "onnx": export}
        if export is not None:
            manifest["environment"]["onnx"] = onnx.__version__
        write_json(stage / "manifest.json", manifest)
        # Reserve the destination exclusively only after generation succeeds.
        output.mkdir(exist_ok=False)
        for item in stage.iterdir():
            shutil.move(str(item), str(output / item.name))
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("reference", "export"))
    parser.add_argument("--repo", required=True, help="official Depth-Anything-V2 Git checkout")
    parser.add_argument("--weights", required=True, help="official vits state_dict weights")
    parser.add_argument("--samples", required=True, help="versioned image/pair manifest")
    parser.add_argument("--output", required=True, help="destination directory; must not exist")
    parser.add_argument("--repo-commit", help="pinned official revision (required for source ZIP without .git)")
    parser.add_argument("--reference-dir", help="export only: reuse a verified existing reference bundle")
    parser.add_argument("--torch-threads", "--threads", type=int, default=4, help="CPU model threads (default: 4)")
    args = parser.parse_args(argv)
    if args.torch_threads < 1:
        parser.error("--torch-threads must be positive")
    if args.reference_dir is not None and args.command != "export":
        parser.error("--reference-dir is only supported with export")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = generate(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(Path(args.output).resolve()), "samples": len(manifest["samples"]),
                      "onnx": manifest["onnx"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
