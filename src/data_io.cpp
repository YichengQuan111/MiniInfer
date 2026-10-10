#include "miniinfer/data_io.hpp"

#include "miniinfer/checked_size.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace miniinfer {
namespace {

[[noreturn]] void fail_weight(const std::filesystem::path& path, const std::string& reason) {
    throw std::runtime_error("FP32 file '" + path.string() + "': " + reason);
}

void validate_platform() {
    const std::uint32_t marker = 1;
    unsigned char first_byte = 0;
    std::memcpy(&first_byte, &marker, sizeof(first_byte));
    if (sizeof(float) != 4 || !std::numeric_limits<float>::is_iec559 || first_byte != 1) {
        throw std::runtime_error("FP32 files require a little-endian IEEE 754 FP32 host");
    }
}

}  // namespace

Buffer load_f32_file(const std::filesystem::path& path, std::size_t elements) {
    validate_platform();
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error)) {
        fail_weight(path, "expected a regular file" +
                              (error ? ": " + error.message() : std::string{}));
    }

    const auto bytes = checked_multiply(elements, sizeof(float));
    if (bytes > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()) ||
        bytes > static_cast<std::size_t>(std::numeric_limits<std::streamoff>::max())) {
        fail_weight(path, "byte size cannot be represented by the file stream");
    }

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        fail_weight(path, "cannot open or seek the file");
    }
    const auto length = file.tellg();
    if (length < std::streampos{0}) {
        fail_weight(path, "cannot determine byte size");
    }
    if (length != std::streampos{static_cast<std::streamoff>(bytes)}) {
        fail_weight(path, "expected exactly " + std::to_string(bytes) +
                              " bytes, got " + std::to_string(static_cast<std::streamoff>(length)));
    }
    file.seekg(0, std::ios::beg);
    if (!file) {
        fail_weight(path, "cannot seek to the start");
    }

    Buffer values(elements);
    const auto count = static_cast<std::streamsize>(bytes);
    file.read(reinterpret_cast<char*>(values.data()), count);
    if (file.gcount() != count || !file) {
        fail_weight(path, "failed to read the complete weight data");
    }
    // Also reject a file that grew after the initial size check.
    char trailing_byte = 0;
    if (file.get(trailing_byte)) {
        fail_weight(path, "unexpected trailing bytes after the weight data");
    }
    if (file.bad() || !file.eof()) {
        fail_weight(path, "I/O failure while checking end of file");
    }
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (!std::isfinite(values.data()[index])) {
            fail_weight(path, "non-finite FP32 value at element " + std::to_string(index));
        }
    }
    return values;
}

}  // namespace miniinfer
