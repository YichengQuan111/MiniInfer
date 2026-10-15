#pragma once

#include "miniinfer/depth/preprocess.hpp"
#include <filesystem>
#include <memory>

namespace miniinfer::depth {

// Static image[1,3,S,S] -> depth[1,S,S], S=518 or 392, CPU FP32 only.
// run/run_raw may overlap on a live runner: Session.Run is serialized per runner
// for ORT telemetry safety; every call owns input/output/temporary storage.
// The SDK creates a separate runner/Session per worker. Caller must keep the
// runner alive. No queue or global memory admission limit is provided here.
class OrtRunner final {
public:
    explicit OrtRunner(const std::filesystem::path& model_path, int threads = 4);
    ~OrtRunner();
    OrtRunner(const OrtRunner&) = delete;
    OrtRunner& operator=(const OrtRunner&) = delete;
    OrtRunner(OrtRunner&&) noexcept;
    OrtRunner& operator=(OrtRunner&&) noexcept;
    RawDepth run_raw(const PreparedImage& input) const;
    DepthMap run(const RgbImage& image) const;
    int threads() const;
    std::size_t input_size() const;
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace miniinfer::depth
