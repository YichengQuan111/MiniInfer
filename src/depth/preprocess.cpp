#include "miniinfer/depth/preprocess.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace miniinfer::depth {
namespace {
struct Coordinate { std::size_t lower, upper; float weight; };
std::vector<Coordinate> axis(std::size_t source, std::size_t target) {
    const float scale = static_cast<float>(static_cast<double>(source) / static_cast<double>(target));
    std::vector<Coordinate> coordinates(target);
    for (std::size_t index = 0; index < target; ++index) {
        const float center = static_cast<float>(index) + 0.5F;
        const float scaled = center * scale;
        const float shifted = scaled - 0.5F;
        const float clamped = std::clamp(shifted, 0.0F, static_cast<float>(source - 1));
        const auto lower = static_cast<std::size_t>(std::floor(clamped));
        coordinates[index] = {lower, std::min(lower + 1, source - 1), clamped - static_cast<float>(lower)};
    }
    return coordinates;
}
}  // namespace

Geometry geometry(std::size_t width, std::size_t height) {
    validate_dimensions(width, height);
    const auto longest = std::max(width, height);
    const auto resized_width = std::max(std::size_t{1}, (width * model_size + longest / 2) / longest);
    const auto resized_height = std::max(std::size_t{1}, (height * model_size + longest / 2) / longest);
    const auto left = (model_size - resized_width) / 2;
    const auto top = (model_size - resized_height) / 2;
    return {width, height, resized_width, resized_height, left, top,
            model_size - resized_width - left, model_size - resized_height - top};
}

void validate_geometry(const Geometry& transform) {
    const auto expected = geometry(transform.original_width, transform.original_height);
    if (transform.resized_width != expected.resized_width || transform.resized_height != expected.resized_height
        || transform.pad_left != expected.pad_left || transform.pad_top != expected.pad_top
        || transform.pad_right != expected.pad_right || transform.pad_bottom != expected.pad_bottom)
        throw std::invalid_argument("geometry does not match frozen depth-v1 transform");
}

PreparedImage::PreparedImage(std::vector<float> nchw, Geometry transform)
    : values_(std::move(nchw)), transform_(transform) {
    validate_geometry(transform_);
    if (values_.size() != input_elements) throw std::invalid_argument("input must contain 1*3*518*518 float32 values");
    validate_finite(values_);
}

std::vector<float> bilinear_half_pixel(const std::vector<float>& source,
    std::size_t source_width, std::size_t source_height, std::size_t channels,
    std::size_t target_width, std::size_t target_height) {
    const auto source_pixels = validate_dimensions(source_width, source_height);
    const auto target_pixels = validate_dimensions(target_width, target_height);
    if (channels != 1 && channels != 3) throw std::invalid_argument("bilinear supports one or three channels");
    if (source.size() != source_pixels * channels) throw std::invalid_argument("bilinear source length mismatch");
    validate_finite(source);
    const auto xs = axis(source_width, target_width);
    const auto ys = axis(source_height, target_height);
    std::vector<float> result(target_pixels * channels);
    for (std::size_t y = 0; y < target_height; ++y) {
        for (std::size_t x = 0; x < target_width; ++x) {
            for (std::size_t c = 0; c < channels; ++c) {
                const float a = source[(ys[y].lower * source_width + xs[x].lower) * channels + c];
                const float b = source[(ys[y].lower * source_width + xs[x].upper) * channels + c];
                const float d = source[(ys[y].upper * source_width + xs[x].lower) * channels + c];
                const float e = source[(ys[y].upper * source_width + xs[x].upper) * channels + c];
                const float upper_difference = b - a;
                const float upper_weighted = upper_difference * xs[x].weight;
                const float upper = a + upper_weighted;
                const float lower_difference = e - d;
                const float lower_weighted = lower_difference * xs[x].weight;
                const float lower = d + lower_weighted;
                const float vertical_difference = lower - upper;
                const float vertical_weighted = vertical_difference * ys[y].weight;
                result[(y * target_width + x) * channels + c] = upper + vertical_weighted;
            }
        }
    }
    validate_finite(result);
    return result;
}

PreparedImage preprocess(const RgbImage& image) {
    const auto transform = geometry(image.width(), image.height());
    if (image.pixels().size() != image.width() * image.height() * 3)
        throw std::invalid_argument("RGB8 input storage length mismatch");
    std::vector<float> pixels(image.pixels().size());
    for (std::size_t index = 0; index < pixels.size(); ++index)
        pixels[index] = static_cast<float>(image.pixels()[index]) / 255.0F;
    const auto resized = bilinear_half_pixel(pixels, image.width(), image.height(), 3,
                                            transform.resized_width, transform.resized_height);
    constexpr std::array<float, 3> mean{0.485F, 0.456F, 0.406F};
    constexpr std::array<float, 3> deviation{0.229F, 0.224F, 0.225F};
    std::vector<float> nchw(input_elements);
    for (std::size_t c = 0; c < 3; ++c) {
        for (std::size_t y = 0; y < model_size; ++y) {
            for (std::size_t x = 0; x < model_size; ++x) {
                float value = 0.0F;
                if (x >= transform.pad_left && x < transform.pad_left + transform.resized_width
                    && y >= transform.pad_top && y < transform.pad_top + transform.resized_height)
                    value = resized[((y - transform.pad_top) * transform.resized_width + x - transform.pad_left) * 3 + c];
                const float centered = value - mean[c];
                nchw[c * raw_elements + y * model_size + x] = centered / deviation[c];
            }
        }
    }
    return PreparedImage(std::move(nchw), transform);
}

DepthMap postprocess(const RawDepth& raw, const Geometry& transform) {
    validate_geometry(transform);
    if (raw.values().size() != raw_elements) throw std::invalid_argument("raw depth storage length mismatch");
    validate_finite(raw.values());
    std::vector<float> cropped(transform.resized_width * transform.resized_height);
    for (std::size_t y = 0; y < transform.resized_height; ++y)
        std::copy_n(raw.values().begin() + static_cast<std::ptrdiff_t>((y + transform.pad_top) * model_size + transform.pad_left),
                    transform.resized_width, cropped.begin() + static_cast<std::ptrdiff_t>(y * transform.resized_width));
    return DepthMap(transform.original_width, transform.original_height,
                   bilinear_half_pixel(cropped, transform.resized_width, transform.resized_height, 1,
                                       transform.original_width, transform.original_height));
}
}  // namespace miniinfer::depth
