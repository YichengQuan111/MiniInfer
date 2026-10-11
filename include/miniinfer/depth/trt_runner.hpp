#pragma once
#include "miniinfer/depth/artifact.hpp"
#include "miniinfer/depth/preprocess.hpp"
#include <filesystem>
#include <memory>

namespace miniinfer::depth {
struct TrtBuildOptions {
    bool fp16 = false;
    int device = 0;
    std::size_t workspace_bytes = 1024ULL * 1024 * 1024;
};
// Offline only. Writes engine, .manifest, .layers.json exclusively.
Manifest build_trt_engine(const std::filesystem::path& onnx,
    const std::filesystem::path& output, const TrtBuildOptions& options = {});
Manifest trt_environment(int device = 0);
struct TrtTiming { double h2d_ms=0, compute_ms=0, d2h_ms=0, host_ms=0; };

// One context/stream/buffer slot. Concurrent calls serialize. Completion is
// synchronized before owned output is returned. Caller maintains object lifetime.
class TrtRunner final {
public:
    explicit TrtRunner(const std::filesystem::path& engine, int device = 0);
    ~TrtRunner();
    TrtRunner(const TrtRunner&) = delete;
    TrtRunner& operator=(const TrtRunner&) = delete;
    RawDepth run_raw(const PreparedImage& input, TrtTiming* timing = nullptr);
    DepthMap run(const RgbImage& image);
    const Manifest& manifest() const noexcept;
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace miniinfer::depth
