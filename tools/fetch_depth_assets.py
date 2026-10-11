#!/usr/bin/env python3
"""Download pinned public V1 assets; large binaries stay outside Git."""
import argparse
import concurrent.futures
import hashlib
import json
import urllib.request
from pathlib import Path

MODEL_COMMIT = "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf"
WEIGHT_REVISION = "03876f8651c73a60fe4c2c48294e09fcb6838fcf"
DATASET_REVISION = "528f83c26f02a6a7a11c5a036dfc9ee19907ebd9"
ASSETS = {
    "weights": (f"https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/{WEIGHT_REVISION}/depth_anything_v2_vits.pth",
                "depth_anything_v2_vits.pth", "715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378"),
    "dataset": (f"https://huggingface.co/datasets/depth-anything/DA-2K/resolve/{DATASET_REVISION}/DA-2K.zip", "DA-2K.zip", "ff0e48e7cc53273efd1312610e51f1ec87bea0b8a22daf37125fd81246592b81"),
    "source": (f"https://codeload.github.com/DepthAnything/Depth-Anything-V2/zip/{MODEL_COMMIT}", "depth-anything-v2-source.zip", "0022639d5b23499329ca4fd29a92993e86b52fdfe9aaf21df683383136e91b65"),
    "ort": ("https://github.com/microsoft/onnxruntime/releases/download/v1.21.0/onnxruntime-linux-x64-1.21.0.tgz", "onnxruntime-linux-x64-1.21.0.tgz", "7485c7e7aac6501b27e353dcbe068e45c61ab51fbaf598d13970dfae669d20bf"),
    "trt-headers": ("https://codeload.github.com/NVIDIA/TensorRT/tar.gz/64e56ab6cbe95c8220d80bc7ff10ccf61935759a", "tensorrt-headers.tar.gz", "2119292c6b3de7b5d47173c7633a48520e1fa60457dad2307954305637d7be7c"),
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(name, directory):
    url, filename, expected = ASSETS[name]
    target = directory / filename
    if not target.exists():
        partial = target.with_suffix(target.suffix + ".partial")
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as stream:
            while block := response.read(4 * 1024 * 1024):
                stream.write(block)
        if expected and digest(partial) != expected:
            raise ValueError(f"checksum mismatch: {filename}")
        partial.replace(target)
    actual = digest(target)
    if expected and actual != expected:
        raise ValueError(f"existing file checksum mismatch: {filename}")
    record = {"asset": name, "url": url, "file": filename, "bytes": target.stat().st_size, "sha256": actual}
    print(json.dumps(record), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", nargs="+", choices=ASSETS, default=list(ASSETS))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(lambda name: fetch(name, args.output), args.assets))
    (args.output / "downloads.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
