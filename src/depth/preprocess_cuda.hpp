#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace miniinfer::depth {

// Independent of the public CPU Geometry class; all dimensions are pixels.
// Source storage is contiguous interleaved RGB8. Destination is FP32 [1,3,S,S].
struct CudaPreprocessGeometry {
    std::uint32_t original_width;
    std::uint32_t original_height;
    std::uint32_t resized_width;
    std::uint32_t resized_height;
    std::uint32_t pad_left;
    std::uint32_t pad_top;
    std::uint32_t model_size;
};

enum class CudaPreprocessKernel {
    Scalar,     // One thread per output element; repeats coordinates per channel.
    FusedRgb    // One thread per output pixel; reuses coordinates across RGB.
};

// No allocation, ownership transfer, copies, or synchronization. Caller supplies
// non-overlapping device buffers and keeps both alive until stream completion.
// A successful return means launch accepted; asynchronous errors are reported by
// the caller's usual stream/event synchronization. Invalid pointers (null only),
// dimensions, centered geometry, or kernel choice return cudaErrorInvalidValue.
// Supported model sizes are 518 and 392; source limits match the CPU contract
// (sides <=4096, pixels <=2097152). Build CUDA with --fmad=false (no fast math).
cudaError_t enqueue_preprocess_rgb8(const std::uint8_t* device_rgb,
    float* device_nchw, CudaPreprocessGeometry transform,
    CudaPreprocessKernel kernel, cudaStream_t stream) noexcept;

// Path B: input is already resized, zero-padded FP32 HWC in RGB/255 units.
// Same ownership and stream rules as above, with 3*S*S input/output elements.
cudaError_t enqueue_normalize_hwc(const float* device_padded_hwc,
    float* device_nchw, std::uint32_t model_size, cudaStream_t stream) noexcept;

}  // namespace miniinfer::depth
