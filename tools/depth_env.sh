#!/usr/bin/env bash
# Source this before running installed GPU CLI programs with pip-provided libs.
depth_env_path="${MINIINFER_DEPTH_ENV:-$HOME/.venvs/miniinfer-depth}"
depth_site_path="$("$depth_env_path/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
export LD_LIBRARY_PATH="$depth_site_path/tensorrt_libs:$depth_site_path/nvidia/cuda_runtime/lib:${MINIINFER_DEPTH_CACHE:-$HOME/.cache/miniinfer-depth}/onnxruntime-linux-x64-1.21.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset depth_env_path depth_site_path
