#include "preprocess_cuda.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <type_traits>

namespace miniinfer::depth {
namespace {
static_assert(std::is_trivial<CudaPreprocessGeometry>::value
    && std::is_standard_layout<CudaPreprocessGeometry>::value, "CUDA geometry must be POD");
constexpr unsigned block_size = 256;

bool valid_size(std::uint32_t size) noexcept { return size == 518 || size == 392; }
bool valid_geometry(CudaPreprocessGeometry g) noexcept {
    if (!valid_size(g.model_size) || !g.original_width || !g.original_height
        || g.original_width > 4096 || g.original_height > 4096
        || static_cast<std::uint64_t>(g.original_width) * g.original_height > 2097152)
        return false;
    const auto longest = std::max(g.original_width, g.original_height);
    const auto width = std::max(1U, (g.original_width * g.model_size + longest / 2) / longest);
    const auto height = std::max(1U, (g.original_height * g.model_size + longest / 2) / longest);
    return g.resized_width == width && g.resized_height == height
        && g.pad_left == (g.model_size - width) / 2
        && g.pad_top == (g.model_size - height) / 2;
}

struct Coordinate { unsigned lower, upper; float weight; };
__device__ Coordinate coordinate(unsigned index, unsigned source, float scale) {
    const float center = __fadd_rn(static_cast<float>(index), 0.5F);
    const float scaled = __fmul_rn(center, scale);
    const float shifted = __fsub_rn(scaled, 0.5F);
    const float clamped = fminf(fmaxf(shifted, 0.0F), static_cast<float>(source - 1));
    const auto lower = static_cast<unsigned>(clamped);
    return {lower, min(lower + 1, source - 1), __fsub_rn(clamped, static_cast<float>(lower))};
}

__device__ float normalize(float value, unsigned channel) {
    const float mean = channel == 0 ? 0.485F : (channel == 1 ? 0.456F : 0.406F);
    const float deviation = channel == 0 ? 0.229F : (channel == 1 ? 0.224F : 0.225F);
    return __fdiv_rn(__fsub_rn(value, mean), deviation);
}

__device__ float sample(const std::uint8_t* rgb, unsigned width,
    Coordinate x, Coordinate y, unsigned channel) {
    // Convert each source value before interpolation, exactly as the CPU path.
    const float a = __fdiv_rn(static_cast<float>(rgb[(y.lower * width + x.lower) * 3 + channel]), 255.0F);
    const float b = __fdiv_rn(static_cast<float>(rgb[(y.lower * width + x.upper) * 3 + channel]), 255.0F);
    const float d = __fdiv_rn(static_cast<float>(rgb[(y.upper * width + x.lower) * 3 + channel]), 255.0F);
    const float e = __fdiv_rn(static_cast<float>(rgb[(y.upper * width + x.upper) * 3 + channel]), 255.0F);
    const float upper = __fadd_rn(a, __fmul_rn(__fsub_rn(b, a), x.weight));
    const float lower = __fadd_rn(d, __fmul_rn(__fsub_rn(e, d), x.weight));
    return __fadd_rn(upper, __fmul_rn(__fsub_rn(lower, upper), y.weight));
}

__global__ void preprocess_scalar(const std::uint8_t* rgb, float* output,
    CudaPreprocessGeometry g, float scale_x, float scale_y) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned plane = g.model_size * g.model_size;
    if (index >= 3 * plane) return;
    const unsigned channel = index / plane;
    const unsigned pixel = index % plane;
    const unsigned x = pixel % g.model_size, y = pixel / g.model_size;
    float value = 0.0F;
    if (x >= g.pad_left && x - g.pad_left < g.resized_width
        && y >= g.pad_top && y - g.pad_top < g.resized_height) {
        value = sample(rgb, g.original_width,
            coordinate(x - g.pad_left, g.original_width, scale_x),
            coordinate(y - g.pad_top, g.original_height, scale_y), channel);
    }
    output[index] = normalize(value, channel);
}

__global__ void preprocess_fused_rgb(const std::uint8_t* rgb, float* output,
    CudaPreprocessGeometry g, float scale_x, float scale_y) {
    const unsigned pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned plane = g.model_size * g.model_size;
    if (pixel >= plane) return;
    const unsigned x = pixel % g.model_size, y = pixel / g.model_size;
    const bool inside = x >= g.pad_left && x - g.pad_left < g.resized_width
        && y >= g.pad_top && y - g.pad_top < g.resized_height;
    Coordinate sx{}, sy{};
    if (inside) {
        sx = coordinate(x - g.pad_left, g.original_width, scale_x);
        sy = coordinate(y - g.pad_top, g.original_height, scale_y);
    }
    // Shares coordinate/clamp/weights across channels. Each warp still writes
    // contiguous values in each NCHW plane; performance is measured separately.
#pragma unroll
    for (unsigned channel = 0; channel < 3; ++channel) {
        const float value = inside ? sample(rgb, g.original_width, sx, sy, channel) : 0.0F;
        output[channel * plane + pixel] = normalize(value, channel);
    }
}

__global__ void normalize_hwc(const float* input, float* output, unsigned plane) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= 3 * plane) return;
    const unsigned channel = index / plane;
    output[index] = normalize(input[(index % plane) * 3 + channel], channel);
}
}  // namespace

cudaError_t enqueue_preprocess_rgb8(const std::uint8_t* device_rgb,
    float* device_nchw, CudaPreprocessGeometry g,
    CudaPreprocessKernel kernel, cudaStream_t stream) noexcept {
    if (!device_rgb || !device_nchw || !valid_geometry(g)
        || (kernel != CudaPreprocessKernel::Scalar && kernel != CudaPreprocessKernel::FusedRgb))
        return cudaErrorInvalidValue;
    // CPU axis() computes double division then rounds once to float32.
    const float scale_x = static_cast<float>(static_cast<double>(g.original_width) / g.resized_width);
    const float scale_y = static_cast<float>(static_cast<double>(g.original_height) / g.resized_height);
    const unsigned plane = g.model_size * g.model_size;
    if (kernel == CudaPreprocessKernel::Scalar)
        preprocess_scalar<<<(3 * plane + block_size - 1) / block_size, block_size, 0, stream>>>(
            device_rgb, device_nchw, g, scale_x, scale_y);
    else
        preprocess_fused_rgb<<<(plane + block_size - 1) / block_size, block_size, 0, stream>>>(
            device_rgb, device_nchw, g, scale_x, scale_y);
    return cudaGetLastError();
}

cudaError_t enqueue_normalize_hwc(const float* device_padded_hwc,
    float* device_nchw, std::uint32_t size, cudaStream_t stream) noexcept {
    if (!device_padded_hwc || !device_nchw || device_padded_hwc == device_nchw || !valid_size(size))
        return cudaErrorInvalidValue;
    const unsigned plane = size * size;
    normalize_hwc<<<(3 * plane + block_size - 1) / block_size, block_size, 0, stream>>>(
        device_padded_hwc, device_nchw, plane);
    return cudaGetLastError();
}
}  // namespace miniinfer::depth
