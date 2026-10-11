#!/usr/bin/env python3
"""Optional Windows-side wheel cache for the Linux CPython 3.12 environment."""
import concurrent.futures
import hashlib
import json
import urllib.request
from pathlib import Path
import argparse

try:
    from packaging.tags import cpython_tags, compatible_tags
    from packaging.utils import parse_wheel_filename
except ImportError:
    from pip._vendor.packaging.tags import cpython_tags, compatible_tags
    from pip._vendor.packaging.utils import parse_wheel_filename

VERSIONS = {
    "numpy": "2.2.6", "sympy": "1.13.1", "mpmath": "1.3.0",
    "networkx": "3.4.2", "Jinja2": "3.1.6", "MarkupSafe": "3.0.3",
    "filelock": "3.18.0", "typing_extensions": "4.13.2", "fsspec": "2025.3.0",
    "setuptools": "75.8.0", "wheel": "0.45.1", "triton": "3.2.0",
    "Pillow": "11.1.0", "onnx": "1.17.0", "protobuf": "5.29.4",
    "onnxruntime": "1.21.0", "coloredlogs": "15.0.1", "humanfriendly": "10.0",
    "flatbuffers": "25.2.10", "packaging": "24.2", "opencv-python-headless": "4.11.0.86",
    "tensorrt-cu12": "10.8.0.43", "tensorrt-cu12-bindings": "10.8.0.43",
}
PLATFORMS = ["manylinux_2_28_x86_64", "manylinux_2_27_x86_64", "manylinux2014_x86_64",
             "manylinux_2_17_x86_64", "manylinux_2_5_x86_64", "manylinux1_x86_64", "linux_x86_64"]
TAGS = set(cpython_tags((3, 12), ["cp312"], PLATFORMS)) | set(compatible_tags((3, 12), "cp312", PLATFORMS))


def fetch(item, output):
    name, version = item
    with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=30) as response:
        metadata = json.load(response)
    candidates = []
    for entry in metadata["urls"]:
        if entry["filename"].endswith(".whl") and parse_wheel_filename(entry["filename"])[3] & TAGS:
            candidates.append(entry)
    if not candidates and name == "tensorrt-cu12":
        candidates = [e for e in metadata["urls"] if e["packagetype"] == "sdist"]
    if not candidates:
        raise ValueError(f"no Linux CPython 3.12 distribution for {name}=={version}")
    entry = sorted(candidates, key=lambda e: e["filename"])[0]
    target = output / entry["filename"]
    if not target.exists():
        partial = target.with_suffix(target.suffix + ".partial")
        with urllib.request.urlopen(entry["url"], timeout=60) as response, partial.open("wb") as stream:
            while block := response.read(4 * 1024 * 1024):
                stream.write(block)
        partial.replace(target)
    with target.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != entry["digests"]["sha256"]:
        raise ValueError(f"wheel hash mismatch: {target}")
    print(f"cached {target.name}", flush=True)
    return {"name": name, "version": version, "file": target.name,
            "sha256": actual, "url": entry["url"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(lambda item: fetch(item, args.output), VERSIONS.items()))
    (args.output / "wheelhouse.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
