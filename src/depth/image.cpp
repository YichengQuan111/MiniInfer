#include "miniinfer/depth/image.hpp"

#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>
#include <utility>

namespace miniinfer::depth {
namespace {
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "depth-v1 requires IEEE754 float32");

std::string token(std::istream& stream, char& separator) {
    std::string result;
    char character = 0;
    auto whitespace = [](char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\v' || c == '\f'; };
    while (stream.get(character)) {
        if (character == '#') {
            std::size_t count = 0;
            while (stream.get(character) && character != '\n') {
                if (++count > 4096) throw std::invalid_argument("PPM comment too long");
            }
        } else if (!whitespace(character)) {
            result.push_back(character);
            break;
        }
    }
    if (result.empty()) throw std::invalid_argument("truncated PPM header");
    while (stream.get(character)) {
        if (whitespace(character)) {
            separator = character;
            return result;
        }
        result.push_back(character);
        if (result.size() > 32) throw std::invalid_argument("PPM header token too long");
    }
    throw std::invalid_argument("truncated PPM header");
}

std::size_t dimension_token(const std::string& text) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("invalid PPM dimensions");
    std::size_t consumed = 0;
    const auto value = std::stoull(text, &consumed);
    if (consumed != text.size() || value > max_side) throw std::invalid_argument("invalid PPM dimensions");
    return static_cast<std::size_t>(value);
}

void write_bytes_exclusive(const std::filesystem::path& path, const void* data, std::size_t bytes) {
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (fd < 0) throw std::system_error(errno, std::generic_category(), "cannot exclusively create " + path.string());
    const auto* cursor = static_cast<const std::uint8_t*>(data);
    std::size_t position = 0;
    while (position < bytes) {
        const auto written = ::write(fd, cursor + position, bytes - position);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) {
            const int error = written < 0 ? errno : EIO;
            ::close(fd);
            throw std::system_error(error, std::generic_category(), "write failed: " + path.string());
        }
        position += static_cast<std::size_t>(written);
    }
    if (::close(fd) < 0) throw std::system_error(errno, std::generic_category(), "close failed: " + path.string());
}
}  // namespace

std::size_t validate_dimensions(std::size_t width, std::size_t height) {
    if (width == 0 || height == 0 || width > max_side || height > max_side || width > max_pixels / height)
        throw std::invalid_argument("image dimensions must be positive, max_side4096 and max_pixels2097152");
    return width * height;
}

void validate_finite(const std::vector<float>& values) {
    for (float value : values) if (!std::isfinite(value)) throw std::invalid_argument("tensor contains NaN or infinity");
}

std::size_t validate_model_size(std::size_t size) {
    if (size != 518 && size != 392) throw std::invalid_argument("static model size must be 518 or 392");
    return size;
}
std::string_view model_contract(std::size_t size) {
    return validate_model_size(size) == 518 ? "depth-v1" : "depth-v2-392";
}
std::size_t contract_model_size(std::string_view contract) {
    if (contract == "depth-v1") return 518;
    if (contract == "depth-v2-392") return 392;
    throw std::invalid_argument("unknown depth model contract");
}

RgbImage::RgbImage(std::size_t width, std::size_t height, std::vector<std::uint8_t> pixels)
    : width_(width), height_(height), pixels_(std::move(pixels)) {
    if (pixels_.size() != 3 * validate_dimensions(width, height))
        throw std::invalid_argument("RGB8 payload byte count must equal width*height*3");
}
DepthMap::DepthMap(std::size_t width, std::size_t height, std::vector<float> values)
    : width_(width), height_(height), values_(std::move(values)) {
    if (values_.size() != validate_dimensions(width, height)) throw std::invalid_argument("depth payload length mismatch");
    validate_finite(values_);
}
RawDepth::RawDepth(std::vector<float> values) : values_(std::move(values)) {
    if (values_.size() != raw_elements && values_.size() != 392 * 392)
        throw std::invalid_argument("raw depth must contain S*S float32 values for S=518 or 392");
    validate_finite(values_);
}

RgbImage read_ppm(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open PPM: " + path.string());
    char separator = 0;
    if (token(stream, separator) != "P6") throw std::invalid_argument("PPM must be P6");
    const auto width = dimension_token(token(stream, separator));
    const auto height = dimension_token(token(stream, separator));
    const auto count = validate_dimensions(width, height);
    if (token(stream, separator) != "255") throw std::invalid_argument("PPM maxval must be 255");
    if (separator == '\r' && stream.peek() == '\n') stream.get();
    std::vector<std::uint8_t> pixels(count * 3);
    stream.read(reinterpret_cast<char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
    if (stream.gcount() != static_cast<std::streamsize>(pixels.size()) || stream.peek() != std::char_traits<char>::eof() || stream.bad())
        throw std::invalid_argument("PPM raster must have exactly width*height*3 bytes");
    return RgbImage(width, height, std::move(pixels));
}

void write_ppm(const std::filesystem::path& path, const RgbImage& image) {
    const std::string header = "P6\n" + std::to_string(image.width()) + " " + std::to_string(image.height()) + "\n255\n";
    std::vector<std::uint8_t> bytes(header.begin(), header.end());
    bytes.insert(bytes.end(), image.pixels().begin(), image.pixels().end());
    write_bytes_exclusive(path, bytes.data(), bytes.size());
}

std::vector<float> read_f32(const std::filesystem::path& path, std::size_t elements) {
    if (elements > 3 * max_pixels) throw std::invalid_argument("float32 artifact exceeds supported element budget");
    const std::size_t bytes = elements * 4;
    if (std::filesystem::file_size(path) != bytes) throw std::invalid_argument("float32 file has incorrect byte count: " + path.string());
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open float32 file: " + path.string());
    std::vector<std::uint8_t> encoded(bytes);
    stream.read(reinterpret_cast<char*>(encoded.data()), static_cast<std::streamsize>(bytes));
    if (stream.gcount() != static_cast<std::streamsize>(bytes) || stream.peek() != std::char_traits<char>::eof() || stream.bad())
        throw std::runtime_error("float32 file changed or could not be read completely: " + path.string());
    std::vector<float> values(elements);
    for (std::size_t index = 0; index < elements; ++index) {
        const auto* p = encoded.data() + index * 4;
        const std::uint32_t bits = static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8)
            | (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
        std::memcpy(&values[index], &bits, sizeof(float));
    }
    validate_finite(values);
    return values;
}

void write_f32(const std::filesystem::path& path, const std::vector<float>& values) {
    if (values.size() > 3 * max_pixels) throw std::invalid_argument("float32 artifact exceeds supported element budget");
    validate_finite(values);
    std::vector<std::uint8_t> bytes(values.size() * 4);
    for (std::size_t index = 0; index < values.size(); ++index) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &values[index], sizeof(float));
        for (unsigned byte = 0; byte < 4; ++byte) bytes[index * 4 + byte] = static_cast<std::uint8_t>(bits >> (byte * 8));
    }
    write_bytes_exclusive(path, bytes.data(), bytes.size());
}

void write_text_exclusive(const std::filesystem::path& path, std::string_view text) {
    write_bytes_exclusive(path, text.data(), text.size());
}
}  // namespace miniinfer::depth
