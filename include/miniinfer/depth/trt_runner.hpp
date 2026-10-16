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
struct TrtTiming {
    double h2d_ms=0, compute_ms=0, d2h_ms=0, host_ms=0;
    // Input staging copy only; owning output copy remains part of host_ms.
    double pinned_copy_ms=0, gpu_preprocess_ms=0, model_ms=0, synchronize_ms=0;
    std::size_t h2d_bytes=0;
};
struct TrtResources {
    std::size_t io_device_bytes=0, io_pinned_bytes=0;
    std::size_t peak_io_device_bytes=0, peak_io_pinned_bytes=0;
    std::size_t device_allocation_calls=0, pinned_allocation_calls=0;
    std::size_t engine_serialized_bytes=0; // Serialized plan size per instance, not actual GPU weight bytes.
    std::size_t context_required_bytes=0; // TensorRT requirement, excludes weights/libraries.
};

// One context/stream/buffer slot. Concurrent calls serialize. Completion is
// synchronized before owned output is returned. Caller maintains object lifetime.
class TrtRunner final {
public:
    explicit TrtRunner(const std::filesystem::path& engine, int device = 0, bool reuse_buffers = true,
                       PreprocessMode mode = PreprocessMode::Cpu);
    ~TrtRunner();
    TrtRunner(const TrtRunner&) = delete;
    TrtRunner& operator=(const TrtRunner&) = delete;
    RawDepth run_raw(const PreparedImage& input, TrtTiming* timing = nullptr);
    // captured_nchw optionally reads actual device input before inference; validation only.
    RawDepth run_image_raw(const RgbImage& image, TrtTiming* timing = nullptr,
                          std::vector<float>* captured_nchw = nullptr);
    // Allows SDK CPU workers to prepare path B before acquiring a GPU slot.
    RawDepth run_hwc_raw(const PaddedImage& input, TrtTiming* timing = nullptr,
                        std::vector<float>* captured_nchw = nullptr);
    DepthMap run(const RgbImage& image);
    const Manifest& manifest() const noexcept;
    std::size_t input_size() const noexcept;
    TrtResources resource_snapshot() const noexcept;
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace miniinfer::depth
