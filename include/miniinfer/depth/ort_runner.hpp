#pragma once

#include "miniinfer/depth/preprocess.hpp"
#include <filesystem>
#include <memory>

namespace miniinfer::depth {

// Static image[1,3,518,518] -> depth[1,518,518], CPU FP32 only.
// run/run_raw may overlap on a live runner: Session is shared; every call owns
// input/output/temporary storage. Caller must keep the runner alive. No queue
// or global memory admission limit is provided at this backend layer.
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
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace miniinfer::depth
