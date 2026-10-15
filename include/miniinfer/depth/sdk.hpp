#pragma once

#include "miniinfer/depth/image.hpp"
#include <chrono>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

namespace miniinfer::depth {
namespace detail { struct SdkTestAccess; struct DepthRequestState; }

enum class SdkErrorCode { InvalidInput, Busy, Closed, ModelIncompatible, BackendFailure };
class SdkError final : public std::runtime_error {
public:
    SdkError(SdkErrorCode code, const std::string& message, std::uint64_t request_id = 0)
        : std::runtime_error(message), code_(code), request_id_(request_id) {}
    SdkErrorCode code() const noexcept { return code_; }
    std::uint64_t request_id() const noexcept { return request_id_; }
private:
    SdkErrorCode code_;
    std::uint64_t request_id_;
};

struct SdkConfig {
    std::filesystem::path package_dir;
    std::size_t queue_capacity = 4;  // Waiting tasks; total reservations <=capacity+worker_count.
    std::size_t byte_budget = 128ULL * 1024 * 1024;
    int ort_threads = 4;
    int device = 0;
    std::size_t worker_count = 4; // Independent CPU workers, each owning an ORT Session/workspace.
    std::size_t gpu_slots = 1; // Independent TensorRT contexts, 1 or 2; unrelated to worker_count.
    bool reuse_host_workspace = true, reuse_gpu_buffers = true;
};

struct SdkSnapshot {
    bool accepting = true;
    bool shutdown_complete = false;
    std::size_t inflight = 0, reserved_bytes = 0, queued = 0, executing = 0;
    std::size_t ready = 0, abandoned = 0, abandoned_completed = 0;
    bool fatal_backend = false;
    // Fixed host scratch only; excludes owning input/output and allocating fallback.
    std::size_t gpu_slots = 1, host_workspace_bytes = 0, host_workspace_allocation_calls = 0;
    std::size_t io_device_bytes = 0, io_pinned_bytes = 0;
    // Sum of slot peaks, which need not occur at the same instant.
    std::size_t peak_io_device_bytes = 0, peak_io_pinned_bytes = 0;
    std::size_t device_allocation_calls = 0, pinned_allocation_calls = 0;
    std::size_t context_required_bytes = 0; // Static TRT requirement, excludes model weights/libs.
    // inflight = queued+executing+ready; accepted = delivered+
    // abandoned_completed+inflight; abandoned may include pending execution.
    // succeeded/failed count backend terminals; delivered counts accepted run
    // get() calls finalized with either result/error (rejects excluded).
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
    std::size_t h2d_bytes = 0; // Successful actual FP32 prepared input transfer; zero for CPU.
    // Actual terminal publication after synchronized execution and restoration;
    // read after get returns/throws, never replace with a late observer timestamp.
    std::chrono::steady_clock::time_point ready_at{};
};

// Owning move-only request. A single handle is not safe for simultaneous method
// calls; separate handles are independent. Wait timeout does not cancel work.
// get consumes once (also on error); destruction/abandon gives up delivery.
// Handle and returned DepthMap may outlive the SDK. Abandoned pending work is
// still drained; its input and quota remain until safe execution completion.
class DepthRequest final {
public:
    DepthRequest() noexcept = default;
    ~DepthRequest() noexcept;
    DepthRequest(DepthRequest&&) noexcept;
    DepthRequest& operator=(DepthRequest&&) noexcept;
    DepthRequest(const DepthRequest&) = delete;
    DepthRequest& operator=(const DepthRequest&) = delete;
    std::uint64_t id() const;
    bool valid() const noexcept;
    bool ready() const;
    void wait() const;
    bool wait_for(std::chrono::milliseconds timeout) const;
    DepthMap get(DepthTiming* timing = nullptr);
    void abandon() noexcept;
private:
    friend class DepthSdk;
    explicit DepthRequest(std::shared_ptr<detail::DepthRequestState> state) noexcept;
    std::shared_ptr<detail::DepthRequestState> state_;
};

// Concurrent submit/run/shutdown are supported while callers keep SDK alive.
// submit synchronously copies input under bounded admission. shutdown rejects
// new requests, drains accepted execution and destroys backend fixed resources;
// it does not wait for callers to consume ready handles. Per-slot TRT engines
// are independent (including duplicate model storage), never writable-shared.
// Per-request reservation is 19*pixels+16MiB through get/terminal abandonment. byte_budget
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
    DepthRequest submit(const RgbImage& image);
    const SdkConfig& config() const & noexcept;
    const SdkConfig& config() const && = delete;
    SdkSnapshot snapshot() const;
    void shutdown();
    static std::size_t reservation_bytes(std::size_t width, std::size_t height);
private:
    friend struct detail::SdkTestAccess;
    DepthSdk(SdkConfig config, std::function<DepthMap(const RgbImage&)> backend,
             std::function<void()> before_delivery,
             std::function<bool(const std::exception_ptr&)> fatal_error = {});
    DepthSdk(SdkConfig config,
             std::function<std::function<DepthMap(const RgbImage&)>(std::size_t)> factory,
             bool worker_local, std::function<void()> before_delivery);
    void wait_for_test(const std::function<bool(const SdkSnapshot&)>& condition,
                       std::chrono::milliseconds timeout);
    class Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace miniinfer::depth
