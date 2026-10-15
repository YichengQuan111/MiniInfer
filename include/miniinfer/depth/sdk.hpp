#pragma once

#include "miniinfer/depth/image.hpp"
#include <chrono>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

namespace miniinfer::depth {
namespace detail { struct SdkTestAccess; }

enum class SdkErrorCode { InvalidInput, Busy, Closed, ModelIncompatible, BackendFailure };
class SdkError final : public std::runtime_error {
public:
    SdkError(SdkErrorCode code, const std::string& message) : std::runtime_error(message), code_(code) {}
    SdkErrorCode code() const noexcept { return code_; }
private:
    SdkErrorCode code_;
};

struct SdkConfig {
    std::filesystem::path package_dir;
    std::size_t queue_capacity = 4;  // Waiting tasks; reservations <=capacity+worker_count.
    std::size_t byte_budget = 128ULL * 1024 * 1024;
    int ort_threads = 4;  // Intra-op threads per worker's independent ORT Session.
    int device = 0;
    std::size_t worker_count = 4;
};

struct SdkSnapshot {
    bool accepting = true;
    bool shutdown_complete = false;
    std::size_t inflight = 0, reserved_bytes = 0, queued = 0, executing = 0;
    // succeeded/failed count backend terminals; delivered counts accepted run
    // calls finalized with either a result or a backend error (rejects excluded).
    std::size_t accepted = 0, succeeded = 0, failed = 0, delivered = 0;
    std::size_t rejected_busy = 0, rejected_closed = 0, rejected_invalid = 0;
    std::string backend, precision;
};

// Optional per-call host timings. Read only after run returns/throws. Rejections
// have accepted=false; zeros for stages that never ran are not GPU durations.
struct DepthTiming {
    bool accepted = false;
    double admission_ms = 0, queue_ms = 0, preprocess_ms = 0;
    double backend_ms = 0, postprocess_ms = 0;
    // Actual successful baseline H2D transfer; zero for CPU or a failed backend.
    std::size_t h2d_bytes = 0;
};

// Synchronous bounded admission, four backend workers by default. Concurrent run/shutdown
// are supported while the caller keeps this object and each input alive.
// Once admission succeeds, run owns an independent RGB copy. shutdown rejects
// new requests, drains accepted work and waits for their run calls to deliver.
// Returned DepthMaps remain valid after later calls or shutdown.
// Per-request reservation is 19*pixels+16MiB through delivery. byte_budget
// excludes caller storage and fixed model weights/ORT arena/GPU context; it is
// an admission budget, not an upper bound on total process RSS or GPU memory.
class DepthSdk final {
public:
    explicit DepthSdk(SdkConfig config);
    ~DepthSdk() noexcept;
    DepthSdk(const DepthSdk&) = delete;
    DepthSdk& operator=(const DepthSdk&) = delete;
    DepthSdk(DepthSdk&&) = delete;
    DepthSdk& operator=(DepthSdk&&) = delete;
    DepthMap run(const RgbImage& image, DepthTiming* timing = nullptr);
    const SdkConfig& config() const & noexcept;
    const SdkConfig& config() const && = delete;
    SdkSnapshot snapshot() const;
    void shutdown();
    static std::size_t reservation_bytes(std::size_t width, std::size_t height);
private:
    friend struct detail::SdkTestAccess;
    DepthSdk(SdkConfig config, std::function<DepthMap(const RgbImage&)> backend,
             std::function<void()> before_delivery);
    DepthSdk(SdkConfig config,
             std::function<std::function<DepthMap(const RgbImage&)>(std::size_t)> factory,
             bool worker_local, std::function<void()> before_delivery);
    void wait_for_test(const std::function<bool(const SdkSnapshot&)>& condition,
                       std::chrono::milliseconds timeout);
    class Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace miniinfer::depth
