#!/usr/bin/env bash
# Configure a GPU-enabled build without changing the CPU-only presets.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
depth_env="${MINIINFER_DEPTH_ENV:-$HOME/.venvs/miniinfer-depth}"
depth_cache="${MINIINFER_DEPTH_CACHE:-$HOME/.cache/miniinfer-depth}"
build_dir="${1:-$HOME/builds/MiniInfer/depth-release}"
build_type="${2:-Release}"
site_dir="$("$depth_env/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
cmake -S "$project_dir" -B "$build_dir" -G Ninja -DCMAKE_BUILD_TYPE="$build_type" \
  -DMINIINFER_ENABLE_ORT=ON -DMINIINFER_ENABLE_TRT=ON \
  -DMINIINFER_DEPTH_PYTHON="$depth_env/bin/python" \
  -DMINIINFER_ORT_ROOT="$depth_cache/onnxruntime-linux-x64-1.21.0" \
  -DMINIINFER_TRT_INCLUDE="$depth_cache/TensorRT-64e56ab6cbe95c8220d80bc7ff10ccf61935759a/include" \
  -DMINIINFER_TRT_LIB="$site_dir/tensorrt_libs" \
  -DMINIINFER_CUDA_RUNTIME="$site_dir/nvidia/cuda_runtime" \
  -DMINIINFER_CUDA_NVCC_INCLUDE="$site_dir/nvidia/cuda_nvcc/include"
