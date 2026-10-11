#!/usr/bin/env bash
# Optional V1 toolchain. No root access and no changes to the CPU-only venv.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
depth_env="${MINIINFER_DEPTH_ENV:-$HOME/.venvs/miniinfer-depth}"
depth_cache="${MINIINFER_DEPTH_CACHE:-$HOME/.cache/miniinfer-depth}"
trt_commit=64e56ab6cbe95c8220d80bc7ff10ccf61935759a
mkdir -p "$depth_cache"
python3 -m venv "$depth_env"
"$depth_env/bin/python" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
NVIDIA_TENSORRT_DISABLE_INTERNAL_PIP=1 "$depth_env/bin/python" -m pip install \
    --extra-index-url https://pypi.nvidia.com -r "$project_dir/tools/depth-requirements.txt"
assets="$project_dir/data/generated/depth/assets"
"$depth_env/bin/python" "$project_dir/tools/fetch_depth_assets.py" --output "$assets"
"$depth_env/bin/python" -m zipfile -e "$assets/depth-anything-v2-source.zip" "$depth_cache"
cp -- "$assets/depth_anything_v2_vits.pth" "$depth_cache/depth_anything_v2_vits.pth"
cp -- "$assets/onnxruntime-linux-x64-1.21.0.tgz" "$depth_cache/onnxruntime-linux-x64-1.21.0.tgz"
tar -xzf "$assets/tensorrt-headers.tar.gz" -C "$depth_cache" "TensorRT-$trt_commit/include"
tar -xzf "$depth_cache/onnxruntime-linux-x64-1.21.0.tgz" -C "$depth_cache"
printf 'Depth Python: %s\nModel/cache: %s\n' "$depth_env/bin/python" "$depth_cache"
