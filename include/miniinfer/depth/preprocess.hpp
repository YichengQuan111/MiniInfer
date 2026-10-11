#pragma once

#include "miniinfer/depth/image.hpp"

namespace miniinfer::depth {

struct Geometry {
    std::size_t original_width, original_height;
    std::size_t resized_width, resized_height;
    std::size_t pad_left, pad_top, pad_right, pad_bottom;
};
Geometry geometry(std::size_t width, std::size_t height);
void validate_geometry(const Geometry& transform);

class PreparedImage final {
public:
    PreparedImage(std::vector<float> nchw, Geometry transform);
    const std::vector<float>& values() const & noexcept { return values_; }
    const std::vector<float>& values() const && = delete;
    Geometry transform() const noexcept { return transform_; }
private:
    std::vector<float> values_;
    Geometry transform_;
};

// Interleaved HW/HWC FP32 resize. Explicit float32 half-pixel/clamp lerps.
// Scalar operations must be built with -ffp-contract=off -fno-fast-math.
std::vector<float> bilinear_half_pixel(const std::vector<float>& source,
    std::size_t source_width, std::size_t source_height, std::size_t channels,
    std::size_t target_width, std::size_t target_height);
PreparedImage preprocess(const RgbImage& image);
DepthMap postprocess(const RawDepth& raw, const Geometry& transform);

}  // namespace miniinfer::depth
