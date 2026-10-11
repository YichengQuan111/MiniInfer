#pragma once
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <vector>
#include <cstdint>

namespace miniinfer::depth {
using Manifest = std::map<std::string, std::string>;
// SHA-256 identity checks detect corruption; manifests are not signatures.
std::string sha256(std::string_view bytes);
std::string file_sha256(const std::filesystem::path& path);
std::vector<std::uint8_t> read_artifact(const std::filesystem::path& path,
                                     std::size_t max_bytes = 1024ULL * 1024 * 1024);
Manifest read_manifest(const std::filesystem::path& path);
std::string manifest_text(const Manifest& manifest);
const std::string& require_field(const Manifest& manifest, const std::string& key);
void require_equal(const Manifest& manifest, const std::string& key, const std::string& expected);
} // namespace miniinfer::depth
