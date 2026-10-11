#!/usr/bin/env python3
"""Record the actual V1 environment and execute a tiny TensorRT CUDA engine."""
import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path


def probe():
    import numpy as np
    import torch
    import tensorrt as trt

    result = {
        "schema_version": 1,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "torchvision", "numpy", "onnx", "onnxruntime",
                      "tensorrt-cu12", "opencv-python-headless", "Pillow")},
        "gpu_query": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
             "--format=csv,noheader"], text=True).strip(),
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "tensorrt": trt.__version__,
    }
    if not result["cuda_available"]:
        raise RuntimeError("CUDA is not available; V1.1 GPU gate did not pass")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    source = network.add_input("x", trt.float32, (1, 4))
    output = network.add_elementwise(source, source, trt.ElementWiseOperation.SUM).get_output(0)
    output.name = "y"
    network.mark_output(output)
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)
    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT probe engine build failed")
    result["probe_build_seconds"] = time.perf_counter() - started
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError("TensorRT probe engine load failed")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("TensorRT probe context creation failed")
    tensor_in = torch.tensor([[1.0, -2.0, 3.5, 0.0]], device="cuda")
    tensor_out = torch.empty_like(tensor_in)
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    if not context.set_tensor_address("x", tensor_in.data_ptr()):
        raise RuntimeError("TensorRT probe input binding failed")
    if not context.set_tensor_address("y", tensor_out.data_ptr()):
        raise RuntimeError("TensorRT probe output binding failed")
    if not context.execute_async_v3(stream.cuda_stream):
        raise RuntimeError("TensorRT probe enqueue failed")
    stream.synchronize()
    actual = tensor_out.cpu().numpy()
    np.testing.assert_array_equal(actual, np.array([[2.0, -4.0, 7.0, 0.0]], dtype=np.float32))
    result["probe"] = {"operation": "y=x+x", "output": actual.tolist(),
                       "device_completion_waited": True, "passed": True}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new evidence file")
    result = probe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"passed": True, "evidence": str(args.output), "gpu": result["gpu_query"]}))


if __name__ == "__main__":
    main()
