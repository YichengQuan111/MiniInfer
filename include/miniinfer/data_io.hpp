#pragma once

#include "miniinfer/buffer.hpp"
#include <cstddef>
#include <filesystem>

namespace miniinfer {

// Exact-length, finite little-endian FP32 data, loaded into owned CPU storage.
Buffer load_f32_file(const std::filesystem::path& path, std::size_t elements);

}  // namespace miniinfer
