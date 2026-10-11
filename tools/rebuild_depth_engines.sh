#!/usr/bin/env bash
# Rebuild on this host, keep frozen thresholds, validate, and package new engines.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: bash tools/rebuild_depth_engines.sh NEW_OUTPUT_DIRECTORY [BUILD_DIRECTORY]" >&2
  exit 2
fi
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="$(realpath -m -- "$1")"
build_dir="$(realpath -m -- "${2:-$HOME/builds/MiniInfer/depth-release}")"
depth_python="${MINIINFER_DEPTH_ENV:-$HOME/.venvs/miniinfer-depth}/bin/python"
mkdir -- "$output_dir" # Never overwrite previous engines or evidence.
cd "$project_dir"
source tools/depth_env.sh
onnx=data/generated/depth/export-v1/depth_anything_v2_vits_518.onnx
reference=data/generated/depth/reference-v1
for precision in fp32 fp16; do
  "$build_dir/bin/depth_engine_builder" --onnx "$onnx" \
    --output "$output_dir/$precision.engine" --precision "$precision" \
    > "$output_dir/$precision-build.log" 2>&1
  "$depth_python" tools/depth/validate_backend.py \
    --cli "$build_dir/bin/depth_trt_cli" --model "$output_dir/$precision.engine" \
    --reference "$reference" --samples data/depth/samples.json --split final \
    --thresholds "data/depth/thresholds_trt_$precision.json" --output "$output_dir/$precision-validation" \
    > "$output_dir/$precision-validation.log" 2>&1
  "$depth_python" tools/package_depth.py create --backend trt --precision "$precision" \
    --model "$output_dir/$precision.engine" --onnx "$onnx" --reference "$reference" \
    --samples data/depth/samples.json --thresholds "data/depth/thresholds_trt_$precision.json" \
    --report "$output_dir/$precision-validation/report.json" --output "$output_dir/$precision-package"
  "$depth_python" tools/verify_depth_sdk.py --cli "$build_dir/bin/depth_sdk_cli" \
    --package "$output_dir/$precision-package" --output "$output_dir/$precision-sdk"
done
