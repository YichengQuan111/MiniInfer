#include "miniinfer/depth/sdk.hpp"
#include "miniinfer/depth/artifact.hpp"
#include "miniinfer/depth/profile.hpp"
#include "miniinfer/thread_pool.hpp"
#if defined(MINIINFER_SDK_HAS_ORT)
#include "miniinfer/depth/ort_runner.hpp"
#endif
#if defined(MINIINFER_SDK_HAS_TRT)
#include "miniinfer/depth/trt_runner.hpp"
#endif

#include <condition_variable>
#include <future>
#include <limits>
#include <mutex>
#include <set>
#include <utility>
#include <vector>

namespace miniinfer::depth {
namespace {
using BasicBackend = std::function<DepthMap(const RgbImage&)>;
using BasicBackendFactory = std::function<BasicBackend(std::size_t)>;
using Backend = std::function<DepthMap(const RgbImage&, DepthTiming*)>;
using BackendFactory = std::function<Backend(std::size_t)>;
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
#if defined(MINIINFER_SDK_HAS_ORT) || defined(MINIINFER_SDK_HAS_TRT)
template<class Runner> DepthMap measured_run(Runner& runner, const RgbImage& image, DepthTiming* timing, bool gpu = false) {
    if (!timing) return runner.run(image);
    auto begin = Clock::now();
    const auto prepared = preprocess(image);
    if (timing) timing->preprocess_ms = elapsed(begin);
    begin = Clock::now();
    auto raw = runner.run_raw(prepared);
    if (timing) {
        timing->backend_ms = elapsed(begin);
        timing->h2d_bytes = gpu ? prepared.values().size() * sizeof(float) : 0;
    }
    begin = Clock::now();
    auto output = postprocess(raw, prepared.transform());
    if (timing) timing->postprocess_ms = elapsed(begin);
    return output;
}
#endif
Backend instrument_backend(BasicBackend backend) {
    if (!backend) return {};
    return [backend = std::move(backend)](const RgbImage& image, DepthTiming* timing) {
        const auto begin = Clock::now();
        auto result = backend(image);
        if (timing) timing->backend_ms = elapsed(begin);
        return result;
    };
}
std::vector<Backend> create_backends(std::size_t count, const BackendFactory& factory) {
    if (!factory) throw SdkError(SdkErrorCode::InvalidInput, "test backend factory cannot be empty");
    std::vector<Backend> backends;
    backends.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        auto backend = factory(index);
        if (!backend) throw SdkError(SdkErrorCode::InvalidInput, "backend factory returned an empty backend");
        backends.push_back(std::move(backend));
    }
    return backends;  // Partial construction releases every previously built backend on failure.
}
void validate_config(const SdkConfig& config) {
    if (config.worker_count == 0 || config.queue_capacity == 0
        || config.queue_capacity > std::numeric_limits<std::size_t>::max() - config.worker_count)
        throw SdkError(SdkErrorCode::InvalidInput, "worker_count and queue_capacity must be positive with a representable sum");
    if (config.byte_budget == 0 || config.ort_threads < 1 || config.device < 0)
        throw SdkError(SdkErrorCode::InvalidInput, "byte_budget/ort_threads must be positive and device nonnegative");
}
struct Loaded { std::vector<Backend> backends; std::string name, precision; };
Loaded load_backend(const SdkConfig& config) {
    validate_config(config);
    std::filesystem::path model;
    std::string backend, precision;
    try {
        if (config.package_dir.empty()) throw std::invalid_argument("package_dir is required");
        const auto root = std::filesystem::canonical(config.package_dir);
        const auto manifest = read_manifest(root / "manifest.txt");
        const std::set<std::string> expected{"schema", "contract", "backend", "precision", "model", "model_sha256"};
        if (manifest.size() != expected.size()) throw std::invalid_argument("model package must have exactly six manifest fields");
        for (const auto& pair : manifest)
            if (expected.count(pair.first) == 0) throw std::invalid_argument("unknown package field: " + pair.first);
        require_equal(manifest, "schema", "miniinfer-model-1");
        require_equal(manifest, "contract", "depth-v1");
        backend = require_field(manifest, "backend");
        precision = require_field(manifest, "precision");
        if ((backend != "ort" && backend != "trt") || (precision != "fp32" && precision != "fp16"))
            throw std::invalid_argument("unsupported backend/precision");
        if (backend == "ort" && precision != "fp32") throw std::invalid_argument("ORT package must use fp32");
        const auto filename = require_field(manifest, "model");
        const std::filesystem::path relative(filename);
        if (filename.empty() || filename == "." || filename == ".." || relative.is_absolute()
            || relative.filename() != relative || filename.find_first_of("/\\:") != std::string::npos)
            throw std::invalid_argument("model must be one relative filename without path traversal");
        model = std::filesystem::canonical(root / relative);
        if (model.parent_path() != root || !std::filesystem::is_regular_file(model))
            throw std::invalid_argument("model resolves outside package or is not a regular file");
        const auto digest = require_field(manifest, "model_sha256");
        if (digest.size() != 64 || digest.find_first_not_of("0123456789abcdef") != std::string::npos
            || file_sha256(model) != digest) throw std::invalid_argument("model SHA256 mismatch or invalid digest");
        if (backend == "trt") require_equal(read_manifest(model.string() + ".manifest"), "precision", precision);
    } catch (const std::exception& error) {
        throw SdkError(SdkErrorCode::ModelIncompatible, std::string("model package incompatible: ") + error.what());
    }
    try {
        if (backend == "ort") {
#if defined(MINIINFER_SDK_HAS_ORT)
            auto backends = create_backends(config.worker_count, [&](std::size_t) -> Backend {
                auto runner = std::make_shared<OrtRunner>(model, config.ort_threads);
                return [runner](const RgbImage& image, DepthTiming* timing) { return measured_run(*runner, image, timing); };
            });
            return {std::move(backends), backend, precision};
#else
            throw SdkError(SdkErrorCode::ModelIncompatible, "ORT backend was not built");
#endif
        }
#if defined(MINIINFER_SDK_HAS_TRT)
        auto backends = create_backends(1, [&](std::size_t) -> Backend {
            auto runner = std::make_shared<TrtRunner>(model, config.device);
            return [runner](const RgbImage& image, DepthTiming* timing) { return measured_run(*runner, image, timing, true); };
        });
        return {std::move(backends), backend, precision};
#else
        throw SdkError(SdkErrorCode::ModelIncompatible, "TensorRT backend was not built");
#endif
    } catch (const SdkError&) {
        throw;
    } catch (const std::invalid_argument& error) {
        throw SdkError(SdkErrorCode::ModelIncompatible, std::string("model contract incompatible: ") + error.what());
    } catch (const std::exception& error) {
        throw SdkError(SdkErrorCode::BackendFailure, std::string("backend initialization failed: ") + error.what());
    }
}
}  // namespace

class DepthSdk::Impl final {
public:
    Impl(SdkConfig config, Loaded loaded, std::function<void()> delivery = {})
        : configuration(std::move(config)), backends(std::move(loaded.backends)),
          before_delivery(std::move(delivery)), pool(configuration.worker_count, configuration.queue_capacity) {
        state.backend = std::move(loaded.name);
        state.precision = std::move(loaded.precision);
    }
    DepthMap execute(const RgbImage& image, std::size_t worker_id, DepthTiming* timing) {
        ProfileRange profile("depth.execute");
        {
            std::lock_guard<std::mutex> lock(mutex);
            --state.queued;
            ++state.executing;
            changed.notify_all();
        }
        try {
            // ORT owns one session per stable pool worker; TRT keeps one shared runner.
            auto result = backends.at(backends.size() == 1 ? 0 : worker_id)(image, timing);
            if (result.width() != image.width() || result.height() != image.height()
                || result.values().size() != image.width() * image.height())
                throw std::runtime_error("backend returned incorrect restored dimensions/storage");
            validate_finite(result.values());
            {
                std::lock_guard<std::mutex> lock(mutex);
                --state.executing;
                ++state.succeeded;
                changed.notify_all();
            }
            return result;
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                --state.executing;
                ++state.failed;
                changed.notify_all();
            }
            try { throw; }
            catch (const std::exception& error) {
                throw SdkError(SdkErrorCode::BackendFailure, std::string("backend execution failed: ") + error.what());
            } catch (...) {
                throw SdkError(SdkErrorCode::BackendFailure, "backend execution failed with an unknown exception");
            }
        }
    }
    void delivered(std::size_t bytes) {
        std::lock_guard<std::mutex> lock(mutex);
        --state.inflight;
        state.reserved_bytes -= bytes;
        ++state.delivered;
        changed.notify_all();
    }
    DepthMap run(const RgbImage& image, DepthTiming* timing) {
        ProfileRange profile("depth.run");
        const auto admission_begin = Clock::now();
        if (timing) *timing = {};
        std::size_t bytes = 0;
        try {
            bytes = DepthSdk::reservation_bytes(image.width(), image.height());
            if (image.pixels().size() != image.width() * image.height() * 3)
                throw std::invalid_argument("RGB storage length mismatch");
        } catch (const std::exception& error) {
            std::lock_guard<std::mutex> lock(mutex);
            ++state.rejected_invalid;
            throw SdkError(SdkErrorCode::InvalidInput, error.what());
        }
        std::future<DepthMap> result;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!state.accepting) {
                ++state.rejected_closed;
                throw SdkError(SdkErrorCode::Closed, "depth SDK is closed");
            }
            if (state.inflight >= configuration.queue_capacity + configuration.worker_count
                || pool.pending() >= configuration.queue_capacity
                || bytes > configuration.byte_budget || state.reserved_bytes > configuration.byte_budget - bytes) {
                ++state.rejected_busy;
                throw SdkError(SdkErrorCode::Busy, "depth SDK count or byte admission budget is full");
            }
            ++state.inflight;
            state.reserved_bytes += bytes;
            bool queued = false;
            try {
                // Reservation exists before allocating/copying caller pixels.
                auto owned = std::make_shared<RgbImage>(image);
                ++state.queued;
                queued = true;
                const auto enqueued = Clock::now();
                result = pool.submit([this, owned = std::move(owned), timing, enqueued](std::size_t worker_id) mutable {
                    // Release copied RGB before packaged_task publishes its
                    // future terminal, on success and on exception alike.
                    auto local = std::move(owned);
                    if (timing) timing->queue_ms = elapsed(enqueued);
                    return execute(*local, worker_id, timing);
                });
                ++state.accepted;
                if (timing) { timing->accepted = true; timing->admission_ms = elapsed(admission_begin); }
                changed.notify_all();
            } catch (...) {
                if (queued) --state.queued;
                --state.inflight;
                state.reserved_bytes -= bytes;
                changed.notify_all();
                try { throw; }
                catch (const QueueFull&) { ++state.rejected_busy; throw SdkError(SdkErrorCode::Busy, "depth SDK queue is full"); }
                catch (const PoolClosed&) { ++state.rejected_closed; throw SdkError(SdkErrorCode::Closed, "depth SDK is closed"); }
                catch (const std::exception& error) { throw SdkError(SdkErrorCode::BackendFailure, std::string("request allocation/submission failed: ") + error.what()); }
                catch (...) { throw SdkError(SdkErrorCode::BackendFailure, "request allocation/submission failed"); }
            }
        }
        struct DeliveryReservation {
            Impl& owner;
            std::size_t bytes;
            ~DeliveryReservation() { owner.delivered(bytes); }
        } reservation{*this, bytes};
        try {
            auto depth = result.get();
            if (before_delivery) before_delivery();  // Only private test constructor sets this hook.
            return depth;
        } catch (const SdkError&) {
            throw;
        } catch (const std::exception& error) {
            throw SdkError(SdkErrorCode::BackendFailure, std::string("result delivery failed: ") + error.what());
        } catch (...) {
            throw SdkError(SdkErrorCode::BackendFailure, "result delivery failed");
        }
    }
    void shutdown() {
        std::lock_guard<std::mutex> shutdown_lock(shutdown_mutex);
        {
            std::lock_guard<std::mutex> lock(mutex);
            state.accepting = false;
            changed.notify_all();
        }
        pool.shutdown();
        std::unique_lock<std::mutex> lock(mutex);
        changed.wait(lock, [this] { return state.inflight == 0; });
        backends.clear();  // Device/context/model release after execution and delivery finish.
        state.shutdown_complete = true;
        changed.notify_all();
    }
    SdkConfig configuration;
    std::vector<Backend> backends;
    std::function<void()> before_delivery;
    mutable std::mutex mutex;
    std::condition_variable changed;
    std::mutex shutdown_mutex;
    SdkSnapshot state;
    ThreadPool pool;  // Starts after backend construction; destroyed before backends and state.
};

DepthSdk::DepthSdk(SdkConfig config) {
    auto loaded = load_backend(config);
    try { impl_ = std::make_unique<Impl>(std::move(config), std::move(loaded)); }
    catch (const std::exception& error) { throw SdkError(SdkErrorCode::BackendFailure, std::string("SDK worker initialization failed: ") + error.what()); }
}
DepthSdk::DepthSdk(SdkConfig config, BasicBackend backend, std::function<void()> before_delivery) {
    validate_config(config);
    if (!backend) throw SdkError(SdkErrorCode::InvalidInput, "test backend cannot be empty");
    auto backends = create_backends(1, [&](std::size_t) { return instrument_backend(std::move(backend)); });
    impl_ = std::make_unique<Impl>(std::move(config), Loaded{std::move(backends), "fake", "fp32"}, std::move(before_delivery));
}
DepthSdk::DepthSdk(SdkConfig config, BasicBackendFactory factory, bool worker_local,
                   std::function<void()> before_delivery) {
    validate_config(config);
    try {
        if (!factory) throw SdkError(SdkErrorCode::InvalidInput, "test backend factory cannot be empty");
        auto backends = create_backends(worker_local ? config.worker_count : 1, [&](std::size_t index) {
            return instrument_backend(factory(index));
        });
        impl_ = std::make_unique<Impl>(std::move(config), Loaded{std::move(backends), "fake", "fp32"}, std::move(before_delivery));
    } catch (const SdkError&) {
        throw;
    } catch (const std::exception& error) {
        throw SdkError(SdkErrorCode::BackendFailure, std::string("backend initialization failed: ") + error.what());
    }
}
DepthSdk::~DepthSdk() noexcept {
    try { if (impl_) impl_->shutdown(); } catch (...) { std::terminate(); }
}
DepthMap DepthSdk::run(const RgbImage& image, DepthTiming* timing) { return impl_->run(image, timing); }
const SdkConfig& DepthSdk::config() const & noexcept { return impl_->configuration; }
SdkSnapshot DepthSdk::snapshot() const { std::lock_guard<std::mutex> lock(impl_->mutex); return impl_->state; }
void DepthSdk::shutdown() { impl_->shutdown(); }
std::size_t DepthSdk::reservation_bytes(std::size_t width, std::size_t height) {
    return 19 * validate_dimensions(width, height) + 16ULL * 1024 * 1024;
}
void DepthSdk::wait_for_test(const std::function<bool(const SdkSnapshot&)>& condition,
                            std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(impl_->mutex);
    if (!impl_->changed.wait_for(lock, timeout, [this, &condition] { return condition(impl_->state); }))
        throw std::runtime_error("SDK test condition timed out");
}
}  // namespace miniinfer::depth
