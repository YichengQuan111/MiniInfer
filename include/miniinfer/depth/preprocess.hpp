#pragma once

#include "miniinfer/depth/image.hpp"
#include <memory>

namespace miniinfer::depth {

struct Geometry {
    std::size_t original_width, original_height;
    std::size_t resized_width, resized_height;
    std::size_t pad_left, pad_top, pad_right, pad_bottom;
};
Geometry geometry(std::size_t width, std::size_t height, std::size_t size = model_size);
std::size_t geometry_size(const Geometry& transform);
void validate_geometry(const Geometry& transform);

class PreparedImage final {
public:
    PreparedImage(std::vector<float> nchw, Geometry transform);
    const std::vector<float>& values() const & noexcept { return values_; }
    const std::vector<float>& values() const && = delete;
    Geometry transform() const noexcept { return transform_; }
private:
    friend class CpuWorkspace;
    std::vector<float> values_;
    Geometry transform_;
};

// Interleaved HW/HWC FP32 resize. Explicit float32 half-pixel/clamp lerps.
// Scalar operations must be built with -ffp-contract=off -fno-fast-math.
std::vector<float> bilinear_half_pixel(const std::vector<float>& source,
    std::size_t source_width, std::size_t source_height, std::size_t channels,
    std::size_t target_width, std::size_t target_height);
PreparedImage preprocess(const RgbImage& image, std::size_t size = model_size);
DepthMap postprocess(const RawDepth& raw, const Geometry& transform);

// One worker owns one workspace. Scratch is preallocated for admitted bounds;
// prepare() is valid until the next prepare/destruction. restore() returns an
// independent owning result. No caller may use this workspace concurrently.
class CpuWorkspace final {
public:
    explicit CpuWorkspace(std::size_t size = model_size);
    ~CpuWorkspace();
    CpuWorkspace(const CpuWorkspace&) = delete;
    CpuWorkspace& operator=(const CpuWorkspace&) = delete;
    const PreparedImage& prepare(const RgbImage& image);
    DepthMap restore(const RawDepth& raw, const Geometry& transform);
    std::size_t capacity_bytes() const noexcept;
    // Six preallocated vector storage blocks; excludes the small Impl object.
    static constexpr std::size_t allocation_count = 6;
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace miniinfer::depth
