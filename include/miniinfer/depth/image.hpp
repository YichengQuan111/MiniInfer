#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string_view>
#include <vector>

namespace miniinfer::depth {

inline constexpr std::size_t model_size = 518;
inline constexpr std::size_t max_side = 4096;
inline constexpr std::size_t max_pixels = 2097152;
inline constexpr std::size_t raw_elements = model_size * model_size;
inline constexpr std::size_t input_elements = 3 * raw_elements;

// Throws invalid_argument for invalid dimensions/length/non-finite values.
std::size_t validate_dimensions(std::size_t width, std::size_t height);
void validate_finite(const std::vector<float>& values);
std::size_t validate_model_size(std::size_t size);
std::string_view model_contract(std::size_t size);
std::size_t contract_model_size(std::string_view contract);

class RgbImage final {
public:
    RgbImage(std::size_t width, std::size_t height, std::vector<std::uint8_t> pixels);
    std::size_t width() const noexcept { return width_; }
    std::size_t height() const noexcept { return height_; }
    const std::vector<std::uint8_t>& pixels() const & noexcept { return pixels_; }
    const std::vector<std::uint8_t>& pixels() const && = delete;
private:
    std::size_t width_, height_;
    std::vector<std::uint8_t> pixels_;
};

// Independent owned FP32 HW output. Copies create independent storage.
class DepthMap final {
public:
    DepthMap(std::size_t width, std::size_t height, std::vector<float> values);
    std::size_t width() const noexcept { return width_; }
    std::size_t height() const noexcept { return height_; }
    const std::vector<float>& values() const & noexcept { return values_; }
    const std::vector<float>& values() const && = delete;
private:
    std::size_t width_, height_;
    std::vector<float> values_;
};

class RawDepth final {
public:
    explicit RawDepth(std::vector<float> values);
    std::size_t size() const noexcept { return values_.size() == 392 * 392 ? 392 : (values_.size() == raw_elements ? 518 : 0); }
    const std::vector<float>& values() const & noexcept { return values_; }
    const std::vector<float>& values() const && = delete;
private:
    std::vector<float> values_;
};

RgbImage read_ppm(const std::filesystem::path& path);
void write_ppm(const std::filesystem::path& path, const RgbImage& image);
std::vector<float> read_f32(const std::filesystem::path& path, std::size_t elements);
// All writers use exclusive creation and refuse existing destinations.
void write_f32(const std::filesystem::path& path, const std::vector<float>& values);
void write_text_exclusive(const std::filesystem::path& path, std::string_view text);

}  // namespace miniinfer::depth
