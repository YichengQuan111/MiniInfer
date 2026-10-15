#include "miniinfer/depth/sdk.hpp"
#include "miniinfer/depth/artifact.hpp"
#include "miniinfer/depth/profile.hpp"
#include "miniinfer/depth/preprocess.hpp"
#include "miniinfer/thread_pool.hpp"
#if defined(MINIINFER_SDK_HAS_ORT)
#include "miniinfer/depth/ort_runner.hpp"
#endif
#if defined(MINIINFER_SDK_HAS_TRT)
#include "miniinfer/depth/trt_runner.hpp"
#endif

#include <algorithm>
#include <condition_variable>
#include <future>
#include <limits>
#include <mutex>
#include <optional>
#include <set>
#include <utility>
#include <vector>

namespace miniinfer::depth {
namespace {
using Backend = std::function<DepthMap(std::size_t, const RgbImage&, DepthTiming*)>;
using Clock = std::chrono::steady_clock;
struct FatalBackend final : std::runtime_error { using std::runtime_error::runtime_error; };
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
using BasicBackend = std::function<DepthMap(const RgbImage&)>;
using BasicBackendFactory = std::function<BasicBackend(std::size_t)>;
#if defined(MINIINFER_SDK_HAS_ORT) || defined(MINIINFER_SDK_HAS_TRT)
template<class Run> DepthMap measured_run(std::size_t size, CpuWorkspace* workspace,
        const RgbImage& image, DepthTiming* timing, Run run, bool gpu) {
    auto begin = Clock::now();
    std::optional<PreparedImage> temporary;
    const PreparedImage* prepared = nullptr;
    if (workspace) prepared = &workspace->prepare(image);
    else { temporary.emplace(preprocess(image, size)); prepared = &*temporary; }
    if (timing) timing->preprocess_ms = elapsed(begin);
    begin = Clock::now();
    // CPU preparation finishes before requesting a GPU lease. Workers can
    // prepare different inputs while one or two independent GPU slots execute.
    auto raw = run(*prepared);
    if (timing) {
        timing->backend_ms = elapsed(begin);
        timing->h2d_bytes = gpu ? prepared->values().size() * sizeof(float) : 0;
    }
    begin = Clock::now();
    auto output = workspace ? workspace->restore(raw, prepared->transform()) : postprocess(raw, prepared->transform());
    if (timing) timing->postprocess_ms = elapsed(begin);
    return output;
}
std::vector<std::shared_ptr<CpuWorkspace>> create_workspaces(const SdkConfig& config, std::size_t size) {
    std::vector<std::shared_ptr<CpuWorkspace>> result(config.worker_count);
    if (config.reuse_host_workspace)
        for (auto& workspace : result) workspace = std::make_shared<CpuWorkspace>(size);
    return result;
}
void workspace_resources(const std::vector<std::shared_ptr<CpuWorkspace>>& workspaces, SdkSnapshot& state) {
    state.host_workspace_bytes = 0;
    state.host_workspace_allocation_calls = 0;
    for (const auto& workspace : workspaces) if (workspace) {
        state.host_workspace_bytes += workspace->capacity_bytes();
        state.host_workspace_allocation_calls += CpuWorkspace::allocation_count;
    }
}
#endif
#if defined(MINIINFER_SDK_HAS_TRT)
class GpuSlots final {
public:
    std::vector<std::shared_ptr<TrtRunner>> runners;
    std::vector<bool> busy;
    std::mutex mutex;
    std::condition_variable changed;
    bool failed = false;
    struct Lease {
        GpuSlots& owner;
        std::size_t index;
        ~Lease() {
            std::lock_guard<std::mutex> lock(owner.mutex);
            owner.busy[index] = false;
            owner.changed.notify_all();
        }
    };
    RawDepth run(const PreparedImage& prepared) {
        std::size_t index;
        {
            std::unique_lock<std::mutex> lock(mutex);
            changed.wait(lock, [&] {
                return failed || std::find(busy.begin(), busy.end(), false) != busy.end();
            });
            if (failed) throw FatalBackend("GPU slots stopped after an execution error");
            index = static_cast<std::size_t>(std::find(busy.begin(), busy.end(), false) - busy.begin());
            busy[index] = true;
        }
        Lease lease{*this, index};
        ProfileRange profile(index == 0 ? "depth.gpu_slot0" : "depth.gpu_slot1");
        try { return runners[index]->run_raw(prepared); }
        catch (...) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                failed = true;
                changed.notify_all();
            }
            try { throw; }
            catch (const std::exception& error) { throw FatalBackend(error.what()); }
            catch (...) { throw FatalBackend("unknown TensorRT execution error"); }
        }
    }
};
#endif
void validate_config(const SdkConfig& config) {
    if (config.gpu_slots < 1 || config.gpu_slots > 2)
        throw SdkError(SdkErrorCode::InvalidInput, "gpu_slots must be 1 or 2");
    if (config.worker_count == 0 || config.queue_capacity == 0
        || config.queue_capacity > std::numeric_limits<std::size_t>::max()-config.worker_count)
        throw SdkError(SdkErrorCode::InvalidInput, "worker_count and queue_capacity must be positive with a representable sum");
    if (config.byte_budget == 0 || config.ort_threads < 1 || config.device < 0)
        throw SdkError(SdkErrorCode::InvalidInput, "byte_budget/ort_threads must be positive and device nonnegative");
}
using Resources = std::function<void(SdkSnapshot&)>;
struct Loaded { Backend backend; std::string name, precision; Resources resources; };
Loaded load_backend(const SdkConfig& config) {
    validate_config(config);
    std::filesystem::path model;
    std::string backend, precision;
    [[maybe_unused]] std::size_t expected_size = model_size;
    try {
        if (config.package_dir.empty()) throw std::invalid_argument("package_dir is required");
        const auto root = std::filesystem::canonical(config.package_dir);
        const auto manifest = read_manifest(root / "manifest.txt");
        const std::set<std::string> expected{"schema", "contract", "backend", "precision", "model", "model_sha256"};
        if (manifest.size() != expected.size()) throw std::invalid_argument("model package must have exactly six manifest fields");
        for (const auto& pair : manifest)
            if (expected.count(pair.first) == 0) throw std::invalid_argument("unknown package field: " + pair.first);
        require_equal(manifest, "schema", "miniinfer-model-1");
        expected_size = contract_model_size(require_field(manifest, "contract"));
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
            std::vector<std::shared_ptr<OrtRunner>> runners;
            for (std::size_t worker = 0; worker < config.worker_count; ++worker) {
                auto runner = std::make_shared<OrtRunner>(model, config.ort_threads);
                if (runner->input_size() != expected_size) throw std::invalid_argument("package/model size mismatch");
                runners.push_back(std::move(runner));
            }
            auto workspaces = create_workspaces(config, expected_size);
            return {[runners, workspaces, expected_size](std::size_t worker, const RgbImage& image, DepthTiming* timing) {
                return measured_run(expected_size, workspaces.at(worker).get(), image, timing,
                    [&](const PreparedImage& prepared) { return runners.at(worker)->run_raw(prepared); }, false);
            }, backend, precision, [workspaces](SdkSnapshot& state) { workspace_resources(workspaces, state); }};
#else
            throw SdkError(SdkErrorCode::ModelIncompatible, "ORT backend was not built");
#endif
        }
#if defined(MINIINFER_SDK_HAS_TRT)
        auto slots = std::make_shared<GpuSlots>();
        for (std::size_t slot = 0; slot < config.gpu_slots; ++slot) {
            auto runner = std::make_shared<TrtRunner>(model, config.device, config.reuse_gpu_buffers);
            if (runner->input_size() != expected_size) throw std::invalid_argument("package/model size mismatch");
            slots->runners.push_back(std::move(runner));
            slots->busy.push_back(false);
        }
        auto workspaces = create_workspaces(config, expected_size);
        return {[slots, workspaces, expected_size](std::size_t worker, const RgbImage& image, DepthTiming* timing) {
            return measured_run(expected_size, workspaces.at(worker).get(), image, timing,
                [&](const PreparedImage& prepared) { return slots->run(prepared); }, true);
        }, backend, precision, [slots, workspaces](SdkSnapshot& state) {
            state.io_device_bytes = 0; state.io_pinned_bytes = 0;
            state.peak_io_device_bytes = 0; state.peak_io_pinned_bytes = 0;
            state.device_allocation_calls = 0; state.pinned_allocation_calls = 0;
            state.context_required_bytes = 0;
            workspace_resources(workspaces, state);
            for (const auto& runner : slots->runners) {
                // resource_snapshot reads atomic counters, never runner locks.
                const auto item = runner->resource_snapshot();
                state.io_device_bytes += item.io_device_bytes;
                state.io_pinned_bytes += item.io_pinned_bytes;
                state.peak_io_device_bytes += item.peak_io_device_bytes;
                state.peak_io_pinned_bytes += item.peak_io_pinned_bytes;
                state.device_allocation_calls += item.device_allocation_calls;
                state.pinned_allocation_calls += item.pinned_allocation_calls;
                state.context_required_bytes += item.context_required_bytes;
            }
        }};
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

namespace detail {
struct SdkLedger {
    mutable std::mutex mutex;
    std::condition_variable changed;
    SdkSnapshot snapshot;
    std::uint64_t next_id = 1;
};
struct DepthRequestState {
    std::shared_ptr<SdkLedger> ledger;
    std::uint64_t id = 0;
    std::size_t bytes = 0;
    bool done = false, abandoned = false, released = false;
    std::optional<DepthMap> output;
    std::exception_ptr failure;
    DepthTiming timing;
    std::function<void()> before_delivery;
};
void release_locked(DepthRequestState& request, bool delivered) {
    if (request.released) return;
    auto& snapshot = request.ledger->snapshot;
    --snapshot.inflight;
    --snapshot.ready;
    snapshot.reserved_bytes -= request.bytes;
    if (delivered) ++snapshot.delivered;
    else ++snapshot.abandoned_completed;
    request.released = true;
    request.ledger->changed.notify_all();
}
}

DepthRequest::DepthRequest(std::shared_ptr<detail::DepthRequestState> state) noexcept : state_(std::move(state)) {}
DepthRequest::~DepthRequest() noexcept { abandon(); }
DepthRequest::DepthRequest(DepthRequest&& other) noexcept : state_(std::move(other.state_)) {}
DepthRequest& DepthRequest::operator=(DepthRequest&& other) noexcept {
    if (this != &other) { abandon(); state_ = std::move(other.state_); }
    return *this;
}
bool DepthRequest::valid() const noexcept { return static_cast<bool>(state_); }
std::uint64_t DepthRequest::id() const {
    if (!state_) throw std::logic_error("invalid depth request");
    return state_->id;
}
bool DepthRequest::ready() const {
    if (!state_) throw std::logic_error("invalid depth request");
    std::lock_guard<std::mutex> lock(state_->ledger->mutex);
    return state_->done;
}
void DepthRequest::wait() const {
    if (!state_) throw std::logic_error("invalid depth request");
    std::unique_lock<std::mutex> lock(state_->ledger->mutex);
    state_->ledger->changed.wait(lock, [&] { return state_->done; });
}
bool DepthRequest::wait_for(std::chrono::milliseconds timeout) const {
    if (!state_) throw std::logic_error("invalid depth request");
    if (timeout.count() < 0) throw std::invalid_argument("negative request wait timeout");
    std::unique_lock<std::mutex> lock(state_->ledger->mutex);
    return state_->ledger->changed.wait_for(lock, timeout, [&] { return state_->done; });
}
DepthMap DepthRequest::get(DepthTiming* timing) {
    if (timing) *timing = {};
    if (!state_) throw std::logic_error("invalid or already consumed depth request");
    auto request = std::move(state_); // One consumption, including error paths.
    auto ledger = request->ledger;
    {
        std::unique_lock<std::mutex> lock(ledger->mutex);
        ledger->changed.wait(lock, [&] { return request->done; });
        if (timing) *timing = request->timing;
    }
    // Test hook uses no SDK pointer, and cannot hold shutdown hostage.
    std::exception_ptr delivery_error;
    try { if (request->before_delivery) request->before_delivery(); }
    catch (...) { delivery_error = std::current_exception(); }
    std::unique_lock<std::mutex> lock(ledger->mutex);
    auto failure = request->failure;
    auto result = std::move(request->output);
    request->output.reset();
    detail::release_locked(*request, true);
    lock.unlock();
    if (failure) std::rethrow_exception(failure);
    if (delivery_error) {
        try { std::rethrow_exception(delivery_error); }
        catch (const std::exception& error) {
            throw SdkError(SdkErrorCode::BackendFailure, std::string("result delivery failed: ")+error.what(), request->id);
        } catch (...) {
            throw SdkError(SdkErrorCode::BackendFailure, "result delivery failed", request->id);
        }
    }
    if (!result) throw SdkError(SdkErrorCode::BackendFailure, "missing terminal output", request->id);
    return std::move(*result);
}
void DepthRequest::abandon() noexcept {
    if (!state_) return;
    auto request = std::move(state_);
    std::lock_guard<std::mutex> lock(request->ledger->mutex);
    request->abandoned = true;
    ++request->ledger->snapshot.abandoned;
    if (request->done) {
        request->output.reset();
        detail::release_locked(*request, false);
    }
    request->ledger->changed.notify_all();
}

class DepthSdk::Impl final {
public:
    Impl(SdkConfig config, Loaded loaded, std::function<void()> delivery = {},
         std::function<bool(const std::exception_ptr&)> fatal = {})
        : configuration(std::move(config)), backend(std::move(loaded.backend)),
          resources(std::move(loaded.resources)),
          before_delivery(std::move(delivery)), is_fatal(std::move(fatal)),
          ledger(std::make_shared<detail::SdkLedger>()),
          pool(configuration.worker_count, configuration.queue_capacity) {
        ledger->snapshot.backend = std::move(loaded.name);
        ledger->snapshot.precision = std::move(loaded.precision);
        ledger->snapshot.gpu_slots = configuration.gpu_slots;
        if (resources) resources(ledger->snapshot);
    }
    void execute(std::size_t worker_id, std::shared_ptr<RgbImage> input,
                 const std::shared_ptr<detail::DepthRequestState>& request, Clock::time_point enqueued) {
        ProfileRange profile("depth.execute");
        DepthTiming timing;
        bool fatal_already = false;
        {
            std::lock_guard<std::mutex> lock(ledger->mutex);
            auto& state = ledger->snapshot;
            --state.queued; ++state.executing;
            timing = request->timing;
            timing.queue_ms = elapsed(enqueued);
            fatal_already = state.fatal_backend;
            ledger->changed.notify_all();
        }
        std::optional<DepthMap> output;
        std::exception_ptr failure;
        bool fatal = false;
        try {
            if (fatal_already) throw SdkError(SdkErrorCode::BackendFailure, "SDK backend stopped after fatal execution error");
            output.emplace(backend(worker_id, *input, &timing));
            if (output->width() != input->width() || output->height() != input->height()
                || output->values().size() != input->width()*input->height())
                throw std::runtime_error("backend returned incorrect restored dimensions/storage");
            validate_finite(output->values());
        } catch (...) {
            const auto original = std::current_exception();
            try { std::rethrow_exception(original); }
            catch (const FatalBackend&) { fatal = true; }
            catch (...) {}
            if (is_fatal) {
                try { fatal = is_fatal(original) || fatal; }
                catch (...) { fatal = true; } // A broken fault classifier cannot keep admission open.
            }
            try {
                try { std::rethrow_exception(original); }
                catch (const std::exception& error) {
                    throw SdkError(SdkErrorCode::BackendFailure,
                        std::string("backend execution failed: ")+error.what(), request->id);
                } catch (...) {
                    throw SdkError(SdkErrorCode::BackendFailure, "backend execution failed with an unknown exception", request->id);
                }
            } catch (...) { failure = std::current_exception(); }
            output.reset();
        }
        input.reset(); // Copied input is released before terminal publication.
        {
            std::lock_guard<std::mutex> lock(ledger->mutex);
            auto& state = ledger->snapshot;
            --state.executing;
            if (failure) ++state.failed; else ++state.succeeded;
            if (fatal) { state.fatal_backend = true; state.accepting = false; }
            request->failure = std::move(failure);
            request->output = std::move(output);
            request->timing = timing;
            request->timing.ready_at = Clock::now();
            request->done = true;
            ++state.ready;
            if (request->abandoned) {
                request->output.reset();
                detail::release_locked(*request, false);
            }
            ledger->changed.notify_all();
        }
    }
    std::shared_ptr<detail::DepthRequestState> submit(const RgbImage& image) {
        ProfileRange profile("depth.submit");
        const auto admission_begin = Clock::now();
        std::size_t bytes = 0;
        try {
            bytes = DepthSdk::reservation_bytes(image.width(), image.height());
            if (image.pixels().size() != image.width()*image.height()*3)
                throw std::invalid_argument("RGB storage length mismatch");
        } catch (const std::exception& error) {
            std::lock_guard<std::mutex> lock(ledger->mutex);
            ++ledger->snapshot.rejected_invalid;
            throw SdkError(SdkErrorCode::InvalidInput, error.what());
        }
        std::lock_guard<std::mutex> lock(ledger->mutex);
        auto& state = ledger->snapshot;
        if (!state.accepting) {
            ++state.rejected_closed;
            throw SdkError(SdkErrorCode::Closed, "depth SDK is closed or its backend has failed");
        }
        if (state.inflight >= configuration.queue_capacity+configuration.worker_count
            || pool.pending() >= configuration.queue_capacity || bytes > configuration.byte_budget
            || state.reserved_bytes > configuration.byte_budget-bytes) {
            ++state.rejected_busy;
            throw SdkError(SdkErrorCode::Busy, "depth SDK count or byte admission budget is full");
        }
        if (ledger->next_id == std::numeric_limits<std::uint64_t>::max())
            throw SdkError(SdkErrorCode::BackendFailure, "request ID range exhausted");
        ++state.inflight; state.reserved_bytes += bytes;
        bool queued = false;
        try {
            auto request = std::make_shared<detail::DepthRequestState>();
            request->ledger = ledger; request->bytes = bytes; request->id = ledger->next_id++;
            request->before_delivery = before_delivery;
            auto owned = std::make_shared<RgbImage>(image);
            ++state.queued; queued = true;
            const auto enqueued = Clock::now();
            pool.submit([this, owned = std::move(owned), request, enqueued](std::size_t worker_id) mutable {
                execute(worker_id, std::move(owned), request, enqueued);
            });
            ++state.accepted;
            request->timing.accepted = true;
            request->timing.admission_ms = elapsed(admission_begin);
            ledger->changed.notify_all();
            return request;
        } catch (...) {
            if (queued) --state.queued;
            --state.inflight; state.reserved_bytes -= bytes;
            ledger->changed.notify_all();
            try { throw; }
            catch (const QueueFull&) { ++state.rejected_busy; throw SdkError(SdkErrorCode::Busy, "depth SDK queue is full"); }
            catch (const PoolClosed&) { ++state.rejected_closed; throw SdkError(SdkErrorCode::Closed, "depth SDK is closed"); }
            catch (const std::exception& error) {
                throw SdkError(SdkErrorCode::BackendFailure, std::string("request allocation/submission failed: ")+error.what());
            } catch (...) { throw SdkError(SdkErrorCode::BackendFailure, "request allocation/submission failed"); }
        }
    }
    void shutdown() {
        std::lock_guard<std::mutex> shutdown_lock(shutdown_mutex);
        {
            std::lock_guard<std::mutex> lock(ledger->mutex);
            ledger->snapshot.accepting = false;
            ledger->changed.notify_all();
        }
        pool.shutdown(); // Drain execution only; handles own their separate ledger/results.
        {
            std::lock_guard<std::mutex> lock(ledger->mutex);
            if (resources) resources(ledger->snapshot);
            backend = {};
            resources = {}; // Releases the second set of shared slot references.
            before_delivery = {};
            is_fatal = {};
            ledger->snapshot.io_device_bytes = 0;
            ledger->snapshot.io_pinned_bytes = 0;
            ledger->snapshot.host_workspace_bytes = 0;
            ledger->snapshot.shutdown_complete = true;
            ledger->changed.notify_all();
        }
    }
    SdkSnapshot snapshot_locked() const {
        auto state = ledger->snapshot;
        // A pool worker may have dequeued a task while waiting for the ledger
        // mutex. Count that assigned task as executing, not in the FIFO queue.
        state.queued = pool.pending();
        state.executing = state.inflight-state.ready-state.queued;
        if (resources) resources(state);
        return state;
    }
    SdkConfig configuration;
    Backend backend;
    Resources resources;
    std::function<void()> before_delivery;
    std::function<bool(const std::exception_ptr&)> is_fatal;
    std::shared_ptr<detail::SdkLedger> ledger;
    std::mutex shutdown_mutex;
    ThreadPool pool; // Joins before backend destruction, even during constructor unwind.
};

DepthSdk::DepthSdk(SdkConfig config) {
    auto loaded = load_backend(config);
    try { impl_ = std::make_unique<Impl>(std::move(config), std::move(loaded)); }
    catch (const std::exception& error) {
        throw SdkError(SdkErrorCode::BackendFailure, std::string("SDK worker initialization failed: ")+error.what());
    }
}
DepthSdk::DepthSdk(SdkConfig config, std::function<DepthMap(const RgbImage&)> backend,
                  std::function<void()> before_delivery,
                  std::function<bool(const std::exception_ptr&)> fatal_error) {
    validate_config(config);
    if (!backend) throw SdkError(SdkErrorCode::InvalidInput, "test backend cannot be empty");
    Backend measured = [backend = std::move(backend)](std::size_t, const RgbImage& image, DepthTiming* timing) {
        const auto begin = Clock::now();
        auto result = backend(image);
        if (timing) timing->backend_ms = elapsed(begin);
        return result;
    };
    impl_ = std::make_unique<Impl>(std::move(config), Loaded{std::move(measured), "fake", "fp32", {}},
                                   std::move(before_delivery), std::move(fatal_error));
}
DepthSdk::DepthSdk(SdkConfig config, BasicBackendFactory factory, bool worker_local,
                   std::function<void()> before_delivery) {
    validate_config(config);
    try {
        if (!factory) throw SdkError(SdkErrorCode::InvalidInput, "test backend factory cannot be empty");
        std::vector<BasicBackend> backends;
        const auto count = worker_local ? config.worker_count : 1;
        for (std::size_t index = 0; index < count; ++index) {
            auto item = factory(index);
            if (!item) throw SdkError(SdkErrorCode::InvalidInput, "backend factory returned an empty backend");
            backends.push_back(std::move(item));
        }
        Backend measured = [backends = std::move(backends)](std::size_t worker, const RgbImage& image, DepthTiming* timing) {
            const auto begin = Clock::now();
            auto result = backends.at(backends.size() == 1 ? 0 : worker)(image);
            if (timing) timing->backend_ms = elapsed(begin);
            return result;
        };
        impl_ = std::make_unique<Impl>(std::move(config), Loaded{std::move(measured), "fake", "fp32", {}},
                                      std::move(before_delivery));
    } catch (const SdkError&) { throw; }
    catch (const std::exception& error) {
        throw SdkError(SdkErrorCode::BackendFailure, std::string("backend initialization failed: ") + error.what());
    }
}
DepthSdk::~DepthSdk() noexcept {
    try { if (impl_) impl_->shutdown(); } catch (...) { std::terminate(); }
}
DepthRequest DepthSdk::submit(const RgbImage& image) { return DepthRequest(impl_->submit(image)); }
DepthMap DepthSdk::run(const RgbImage& image, DepthTiming* timing) {
    ProfileRange profile("depth.run");
    if (timing) *timing = {};
    return submit(image).get(timing);
}
const SdkConfig& DepthSdk::config() const & noexcept { return impl_->configuration; }
SdkSnapshot DepthSdk::snapshot() const {
    std::lock_guard<std::mutex> lock(impl_->ledger->mutex);
    return impl_->snapshot_locked();
}
void DepthSdk::shutdown() { impl_->shutdown(); }
std::size_t DepthSdk::reservation_bytes(std::size_t width, std::size_t height) {
    return 19*validate_dimensions(width,height)+16ULL*1024*1024;
}
void DepthSdk::wait_for_test(const std::function<bool(const SdkSnapshot&)>& condition,
                            std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(impl_->ledger->mutex);
    if (!impl_->ledger->changed.wait_for(lock, timeout, [this, &condition] {
        return condition(impl_->snapshot_locked());
    })) throw std::runtime_error("SDK test condition timed out");
}
} // namespace miniinfer::depth
