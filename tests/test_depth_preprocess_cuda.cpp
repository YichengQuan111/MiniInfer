#include "miniinfer/depth/preprocess.hpp"
#include "preprocess_cuda.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace miniinfer::depth;
// Frozen before implementation; do not loosen to hide arithmetic changes.
constexpr float absolute_tolerance = 2.0e-6F;
constexpr float relative_tolerance = 1.0e-6F;
constexpr std::size_t guards = 17;
constexpr float sentinel = -913.25F;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
template<class T> struct DeviceBuffer {
    T* pointer = nullptr;
    explicit DeviceBuffer(std::size_t elements) {
        check(cudaMalloc(reinterpret_cast<void**>(&pointer), elements * sizeof(T)), "cudaMalloc");
    }
    ~DeviceBuffer() { if (pointer) cudaFree(pointer); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};
struct Stream {
    cudaStream_t stream = nullptr;
    Stream() { check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate"); }
    ~Stream() { if (stream) cudaStreamDestroy(stream); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};

enum class Pattern { Random, Black, White, Alternating };
std::vector<std::uint8_t> pixels(std::size_t width, std::size_t height, Pattern pattern) {
    std::vector<std::uint8_t> result(width * height * 3);
    std::uint32_t state = 0x5eeda11U;
    for (std::size_t index = 0; index < result.size(); ++index) {
        state = state * 1664525U + 1013904223U;
        switch (pattern) {
        case Pattern::Random: result[index] = static_cast<std::uint8_t>(state >> 24); break;
        case Pattern::Black: result[index] = 0; break;
        case Pattern::White: result[index] = 255; break;
        case Pattern::Alternating:
            // Small (0/1) and extreme (0/255) values expose conversion order,
            // channel ordering, boundary clamps, and fractional interpolation.
            result[index] = static_cast<std::uint8_t>((index / 3) % 2 ? (index % 3 ? 255 : 1) : 0);
            break;
        }
    }
    return result;
}

std::vector<float> padded_hwc(const RgbImage& image, const Geometry& g, std::size_t size) {
    std::vector<float> source(image.pixels().size());
    for (std::size_t i = 0; i < source.size(); ++i)
        source[i] = static_cast<float>(image.pixels()[i]) / 255.0F;
    const auto resized = bilinear_half_pixel(source, image.width(), image.height(), 3,
        g.resized_width, g.resized_height);
    std::vector<float> result(size * size * 3, 0.0F);
    for (std::size_t y = 0; y < g.resized_height; ++y)
        std::copy_n(resized.begin() + static_cast<std::ptrdiff_t>(y * g.resized_width * 3),
            g.resized_width * 3,
            result.begin() + static_cast<std::ptrdiff_t>(((y + g.pad_top) * size + g.pad_left) * 3));
    return result;
}

float maximum_absolute_error = 0.0F;
void compare(const std::vector<float>& expected, const std::vector<float>& guarded,
    const std::string& label) {
    require(guarded.size() == expected.size() + 2 * guards, "test output length");
    for (std::size_t i = 0; i < guards; ++i)
        require(guarded[i] == sentinel && guarded[guards + expected.size() + i] == sentinel,
            label + " output boundary overwritten");
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const float actual = guarded[guards + i];
        const float difference = std::abs(actual - expected[i]);
        maximum_absolute_error = std::max(maximum_absolute_error, difference);
        if (!std::isfinite(actual) || difference > absolute_tolerance + relative_tolerance * std::abs(expected[i]))
            throw std::runtime_error(label + " mismatch at " + std::to_string(i)
                + ": CPU=" + std::to_string(expected[i]) + ", GPU=" + std::to_string(actual)
                + ", abs=" + std::to_string(difference));
    }
}

void run_case(std::size_t width, std::size_t height, std::size_t size, Pattern pattern) {
    const RgbImage image(width, height, pixels(width, height, pattern));
    const auto expected = preprocess(image, size);
    const auto g = expected.transform();
    const CudaPreprocessGeometry cuda_g{static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height),
        static_cast<std::uint32_t>(g.resized_width), static_cast<std::uint32_t>(g.resized_height),
        static_cast<std::uint32_t>(g.pad_left), static_cast<std::uint32_t>(g.pad_top),
        static_cast<std::uint32_t>(size)};
    const std::string label = std::to_string(width) + "x" + std::to_string(height)
        + "->" + std::to_string(size) + " pattern=" + std::to_string(static_cast<int>(pattern));
    Stream stream;
    DeviceBuffer<std::uint8_t> source(image.pixels().size());
    DeviceBuffer<float> output(expected.values().size() + 2 * guards);
    std::vector<float> guarded(expected.values().size() + 2 * guards, sentinel);
    check(cudaMemcpyAsync(source.pointer, image.pixels().data(), image.pixels().size(),
        cudaMemcpyHostToDevice, stream.stream), "upload RGB8");
    for (const auto kernel : {CudaPreprocessKernel::Scalar, CudaPreprocessKernel::FusedRgb}) {
        std::fill(guarded.begin(), guarded.end(), sentinel);
        check(cudaMemcpyAsync(output.pointer, guarded.data(), guarded.size() * sizeof(float),
            cudaMemcpyHostToDevice, stream.stream), "initialize guarded output");
        check(enqueue_preprocess_rgb8(source.pointer, output.pointer + guards, cuda_g, kernel, stream.stream),
            "enqueue_preprocess_rgb8");
        check(cudaMemcpyAsync(guarded.data(), output.pointer, guarded.size() * sizeof(float),
            cudaMemcpyDeviceToHost, stream.stream), "download RGB8 result");
        check(cudaStreamSynchronize(stream.stream), "synchronize RGB8 result");
        compare(expected.values(), guarded, label + (kernel == CudaPreprocessKernel::Scalar ? " scalar" : " fused"));
    }
    const auto padded = padded_hwc(image, g, size);
    DeviceBuffer<float> hwc(padded.size());
    std::fill(guarded.begin(), guarded.end(), sentinel);
    check(cudaMemcpyAsync(output.pointer, guarded.data(), guarded.size() * sizeof(float),
        cudaMemcpyHostToDevice, stream.stream), "initialize B guarded output");
    check(cudaMemcpyAsync(hwc.pointer, padded.data(), padded.size() * sizeof(float),
        cudaMemcpyHostToDevice, stream.stream), "upload padded HWC");
    check(enqueue_normalize_hwc(hwc.pointer, output.pointer + guards,
        static_cast<std::uint32_t>(size), stream.stream), "enqueue_normalize_hwc");
    check(cudaMemcpyAsync(guarded.data(), output.pointer, guarded.size() * sizeof(float),
        cudaMemcpyDeviceToHost, stream.stream), "download HWC result");
    check(cudaStreamSynchronize(stream.stream), "synchronize HWC result");
    compare(expected.values(), guarded, label + " normalize HWC");
}

void invalid_arguments() {
    DeviceBuffer<std::uint8_t> source(3);
    DeviceBuffer<float> output(3 * 518 * 518);
    DeviceBuffer<float> hwc(3 * 518 * 518);
    const CudaPreprocessGeometry valid{1, 1, 518, 518, 0, 0, 518};
    auto invalid = [&](CudaPreprocessGeometry g) {
        require(enqueue_preprocess_rgb8(source.pointer, output.pointer, g,
            CudaPreprocessKernel::FusedRgb, nullptr) == cudaErrorInvalidValue,
            "malformed geometry accepted");
    };
    auto wrong = valid; wrong.original_width = 0; invalid(wrong);
    wrong = valid; wrong.original_height = 0; invalid(wrong);
    wrong = valid; wrong.original_width = 4097; invalid(wrong);
    wrong = valid; wrong.original_width = 2049; wrong.original_height = 1024; invalid(wrong);
    wrong = valid; wrong.original_width = UINT32_MAX; invalid(wrong);
    wrong = valid; wrong.resized_width = 517; invalid(wrong);
    wrong = valid; wrong.resized_height = 0; invalid(wrong);
    wrong = valid; wrong.pad_left = 1; invalid(wrong);
    wrong = valid; wrong.pad_top = UINT32_MAX; invalid(wrong);
    wrong = valid; wrong.model_size = 400; invalid(wrong);
    require(enqueue_preprocess_rgb8(nullptr, output.pointer, valid, CudaPreprocessKernel::Scalar, nullptr)
        == cudaErrorInvalidValue, "null RGB8 accepted");
    require(enqueue_preprocess_rgb8(source.pointer, nullptr, valid, CudaPreprocessKernel::Scalar, nullptr)
        == cudaErrorInvalidValue, "null output accepted");
    require(enqueue_preprocess_rgb8(source.pointer, output.pointer, valid,
        static_cast<CudaPreprocessKernel>(99), nullptr) == cudaErrorInvalidValue, "unknown kernel accepted");
    require(enqueue_normalize_hwc(nullptr, output.pointer, 518, nullptr) == cudaErrorInvalidValue,
        "null HWC accepted");
    require(enqueue_normalize_hwc(hwc.pointer, nullptr, 518, nullptr) == cudaErrorInvalidValue,
        "null B output accepted");
    require(enqueue_normalize_hwc(hwc.pointer, output.pointer, 400, nullptr) == cudaErrorInvalidValue,
        "invalid B size accepted");
    require(enqueue_normalize_hwc(output.pointer, output.pointer, 518, nullptr) == cudaErrorInvalidValue,
        "overlapping B input/output accepted");
    check(cudaDeviceSynchronize(), "invalid calls must not launch kernels");
}
}  // namespace

int main() {
    try {
        int devices = 0;
        const auto status = cudaGetDeviceCount(&devices);
        if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver
            || (status == cudaSuccess && devices == 0)) {
            std::cout << "SKIP: CUDA device/driver unavailable\n";
            return 77;
        }
        check(status, "cudaGetDeviceCount");
        invalid_arguments();
        std::size_t cases = 0;
        for (const std::size_t size : {518U, 392U}) {
            // Includes both singleton axes, source clamp boundaries, non-block
            // output tails, both odd-padding axes, up/downsample and no resize.
            for (const auto& shape : std::vector<std::pair<std::size_t, std::size_t>>{
                {1, 1}, {1, 4096}, {4096, 1}, {4, 3}, {3, 4}, {13, 7}, {7, 13},
                {size, size}, {2 * size, size}, {521, 389}, {2048, 1024}}) {
                run_case(shape.first, shape.second, size, Pattern::Random);
                ++cases;
            }
            for (const auto pattern : {Pattern::Black, Pattern::White, Pattern::Alternating}) {
                run_case(2 * size, size, size, pattern);
                ++cases;
            }
        }
        std::cout << "CUDA preprocess passed: " << cases << " cases x 3 kernels; atol=" << absolute_tolerance
            << " rtol=" << relative_tolerance << " max_abs=" << maximum_absolute_error << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "CUDA preprocess failed: " << error.what() << '\n';
        return 1;
    }
}
