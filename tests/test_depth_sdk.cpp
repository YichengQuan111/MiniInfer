#include "depth_sdk_access.hpp"
#include "miniinfer/depth/artifact.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <iostream>
#include <limits>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace miniinfer::depth;
using miniinfer::depth::detail::SdkTestAccess;
void expect(bool value, const char* description) { if (!value) throw std::runtime_error(description); }
template<class Callable> void error(SdkErrorCode code, Callable callable, const char* description) {
    try { callable(); } catch (const SdkError& failure) {
        expect(failure.code() == code, description);
        return;
    }
    throw std::runtime_error(description);
}
struct Gate {
    std::mutex mutex;
    std::condition_variable changed;
    std::size_t entered = 0;
    bool open = false;
    void block() {
        std::unique_lock<std::mutex> lock(mutex);
        ++entered;
        changed.notify_all();
        if (!changed.wait_for(lock, std::chrono::seconds(5), [this] { return open; }))
            throw std::runtime_error("backend/delivery test gate timed out");
    }
    void wait(std::size_t count = 1) {
        std::unique_lock<std::mutex> lock(mutex);
        if (!changed.wait_for(lock, std::chrono::seconds(5), [this, count] { return entered >= count; }))
            throw std::runtime_error("test gate entry timed out");
    }
    void release() { std::lock_guard<std::mutex> lock(mutex); open = true; changed.notify_all(); }
};
RgbImage image(std::uint8_t value = 7, std::size_t width = 1, std::size_t height = 1) {
    return RgbImage(width, height, std::vector<std::uint8_t>(width * height * 3, value));
}
DepthMap output(const RgbImage& input) {
    return DepthMap(input.width(), input.height(), std::vector<float>(input.width() * input.height(), input.pixels()[0]));
}
struct BackendLifetime {
    explicit BackendLifetime(std::atomic<std::size_t>& count) : destroyed(count) {}
    ~BackendLifetime() { ++destroyed; }
    std::atomic<std::size_t>& destroyed;
};
void worker_backend_ownership() {
    constexpr std::size_t worker_count = 4;
    Gate first_wave, second_wave, delivery;
    std::atomic<std::size_t> destroyed{0}, deliveries{0};
    std::vector<std::weak_ptr<BackendLifetime>> lifetimes;
    std::vector<std::thread::id> owners(worker_count);
    std::vector<std::size_t> calls(worker_count, 0), active(worker_count, 0);
    std::mutex observations;
    bool fixed_owner = true, exclusive = true;
    std::size_t factory_calls = 0;
    SdkConfig config;
    config.worker_count = worker_count;
    auto sdk = SdkTestAccess::create_factory(config, [&](std::size_t index) {
        expect(index == factory_calls++, "backend factories must follow stable worker indices");
        auto lifetime = std::make_shared<BackendLifetime>(destroyed);
        lifetimes.push_back(lifetime);
        return [&, index, lifetime](const RgbImage& input) {
            std::size_t call;
            {
                std::lock_guard<std::mutex> lock(observations);
                const auto thread = std::this_thread::get_id();
                call = ++calls[index];
                if (call == 1) owners[index] = thread;
                else fixed_owner = fixed_owner && owners[index] == thread;
                exclusive = exclusive && ++active[index] == 1;
            }
            if (call == 1) first_wave.block();
            else second_wave.block();
            {
                std::lock_guard<std::mutex> lock(observations);
                --active[index];
            }
            return output(input);
        };
    }, true, [&] { if (++deliveries > worker_count) delivery.block(); });
    expect(factory_calls == worker_count && lifetimes.size() == worker_count,
           "worker-local SDK must create exactly one backend per configured worker before use");
    std::vector<std::future<DepthMap>> running;
    for (std::size_t index = 0; index < worker_count; ++index) {
        running.push_back(std::async(std::launch::async, [&, index] {
            return sdk->run(image(static_cast<std::uint8_t>(index + 1)));
        }));
        first_wave.wait(index + 1);
    }
    {
        std::lock_guard<std::mutex> lock(observations);
        expect(std::all_of(active.begin(), active.end(), [](std::size_t count) { return count == 1; }),
               "independent worker backends must overlap while each backend is occupied once");
        expect(std::set<std::thread::id>(owners.begin(), owners.end()).size() == worker_count,
               "each backend must belong to a distinct SDK worker");
    }
    first_wave.release();
    std::vector<DepthMap> results;
    for (auto& future : running) results.push_back(future.get());
    running.clear();
    for (std::size_t index = 0; index < worker_count; ++index) {
        running.push_back(std::async(std::launch::async, [&, index] {
            return sdk->run(image(static_cast<std::uint8_t>(index + 10)));
        }));
        second_wave.wait(index + 1);
    }
    {
        std::lock_guard<std::mutex> lock(observations);
        expect(fixed_owner && exclusive
               && std::all_of(calls.begin(), calls.end(), [](std::size_t count) { return count == 2; }),
               "backend must stay exclusive to the same worker across repeated requests");
    }
    auto shutdown = std::async(std::launch::async, [&] { sdk->shutdown(); });
    SdkTestAccess::wait(*sdk, [](const SdkSnapshot& state) { return !state.accepting; });
    expect(destroyed == 0, "shutdown released a worker backend while execution was blocked");
    second_wave.release();
    delivery.wait(worker_count);
    expect(!sdk->snapshot().shutdown_complete && destroyed == 0
           && std::all_of(lifetimes.begin(), lifetimes.end(), [](const auto& lifetime) { return !lifetime.expired(); }),
           "all worker backends must remain alive until accepted results are delivered");
    delivery.release();
    for (auto& future : running) results.push_back(future.get());
    shutdown.get();
    expect(destroyed == worker_count
           && std::all_of(lifetimes.begin(), lifetimes.end(), [](const auto& lifetime) { return lifetime.expired(); }),
           "shutdown must release every worker backend after execution and delivery");
    sdk.reset();
    for (std::size_t index = 0; index < worker_count; ++index) {
        expect(results[index].values()[0] == index + 1 && results[index + worker_count].values()[0] == index + 10,
               "results must remain independently owned after worker backends and SDK are destroyed");
    }
}
void partial_backend_initialization() {
    std::atomic<std::size_t> destroyed{0}, executions{0};
    std::vector<std::weak_ptr<BackendLifetime>> lifetimes;
    std::size_t factory_calls = 0;
    error(SdkErrorCode::BackendFailure, [&] {
        SdkTestAccess::create_factory({}, [&](std::size_t index) -> std::function<DepthMap(const RgbImage&)> {
            expect(index == factory_calls++, "backend construction order changed");
            if (index == 2) throw std::runtime_error("intentional third backend construction failure");
            auto lifetime = std::make_shared<BackendLifetime>(destroyed);
            lifetimes.push_back(lifetime);
            return [&, lifetime](const RgbImage& input) { ++executions; return output(input); };
        });
    }, "partial backend initialization must report BackendFailure");
    expect(factory_calls == 3 && destroyed == 2 && executions == 0
           && std::all_of(lifetimes.begin(), lifetimes.end(), [](const auto& lifetime) { return lifetime.expired(); }),
           "failed construction must release all earlier backends before starting SDK work");
    error(SdkErrorCode::InvalidInput, [&] { SdkTestAccess::create_factory({}, {}); },
          "empty injected backend factory accepted");
    error(SdkErrorCode::InvalidInput, [&] {
        SdkTestAccess::create_factory({}, [](std::size_t) { return std::function<DepthMap(const RgbImage&)>{}; });
    }, "empty worker backend accepted");
}
void ownership() {
    Gate gate;
    std::atomic<int> calls{0};
    auto sdk = SdkTestAccess::create({}, [&](const RgbImage& input) {
        if (++calls == 1) gate.block();
        return output(input);
    });
    auto original = image(7, 2, 2);
    auto first = std::async(std::launch::async, [&] { return sdk->run(original); });
    gate.wait();  // Backend has received an independently copied image.
    original = image(99, 2, 2);
    gate.release();
    const auto result = first.get();
    expect(result.values() == std::vector<float>(4, 7), "SDK copied caller input before execution");
    const auto later = sdk->run(original);
    expect(later.values()[0] == 99 && result.values()[0] == 7, "later run changed previous result");
    sdk->shutdown();
    sdk.reset();
    expect(result.values()[0] == 7, "SDK destruction invalidated result");
}
void count_and_parallelism(std::size_t worker_count = 4) {
    Gate gate;
    std::atomic<int> calls{0};
    std::mutex workers_mutex;
    std::set<std::thread::id> workers;
    SdkConfig config;
    config.queue_capacity = 1;
    if (worker_count != 4) config.worker_count = worker_count;
    auto sdk = SdkTestAccess::create(config, [&](const RgbImage& input) {
        {
            std::lock_guard<std::mutex> lock(workers_mutex);
            workers.insert(std::this_thread::get_id());
        }
        if (static_cast<std::size_t>(++calls) <= worker_count) gate.block();
        return output(input);
    });
    const auto input = image();
    std::vector<std::future<DepthMap>> running;
    for (std::uint8_t value = 1; value <= worker_count; ++value) {
        running.push_back(std::async(std::launch::async, [&, value] { return sdk->run(image(value)); }));
        gate.wait(value);  // Occupy each worker before submitting into the one-slot queue.
    }
    auto queued = std::async(std::launch::async, [&] { return sdk->run(input); });
    SdkTestAccess::wait(*sdk, [](const SdkSnapshot& state) { return state.queued == 1; });
    const auto full = sdk->snapshot();
    expect(sdk->config().worker_count == worker_count && full.inflight == worker_count + 1
           && full.executing == worker_count && full.queued == 1,
           "count must include configured executions and one waiting request");
    expect(static_cast<std::size_t>(calls.load()) == worker_count,
           "waiting request entered backend while all workers were occupied");
    {
        std::lock_guard<std::mutex> lock(workers_mutex);
        expect(workers.size() == worker_count, "pool must execute on the configured distinct worker threads");
    }
    error(SdkErrorCode::Busy, [&] { sdk->run(input); }, "count exhaustion must be Busy");
    gate.release();
    for (std::size_t index = 0; index < running.size(); ++index) {
        const auto result = running[index].get();
        expect(result.values()[0] == index + 1, "parallel result was mixed with another request");
    }
    const auto queued_result = queued.get();
    expect(queued_result.values()[0] == 7, "waiting request did not complete");
    const auto state = sdk->snapshot();
    expect(state.accepted == worker_count + 1 && state.succeeded == worker_count + 1
           && state.delivered == worker_count + 1 && state.inflight == 0
           && state.reserved_bytes == 0 && state.rejected_busy == 1, "count terminal accounting");
}
void byte_admission() {
    Gate gate;
    SdkConfig config;
    config.byte_budget = 4 * DepthSdk::reservation_bytes(1, 1);
    auto sdk = SdkTestAccess::create(config, [&](const RgbImage& input) { gate.block(); return output(input); });
    const auto input = image();
    std::vector<std::future<DepthMap>> running;
    for (int index = 0; index < 4; ++index) {
        running.push_back(std::async(std::launch::async, [&] { return sdk->run(input); }));
        gate.wait(static_cast<std::size_t>(index + 1));
    }
    expect(sdk->snapshot().reserved_bytes == config.byte_budget, "exact byte budget reservation");
    error(SdkErrorCode::Busy, [&] { sdk->run(input); }, "byte exhaustion must be Busy");
    gate.release();
    for (auto& result : running) result.get();
    expect(sdk->snapshot().reserved_bytes == 0, "delivery did not release byte reservation");
    config.byte_budget = DepthSdk::reservation_bytes(1, 1) - 1;
    auto too_small = SdkTestAccess::create(config, output);
    error(SdkErrorCode::Busy, [&] { too_small->run(input); }, "single request larger than budget admitted");
    expect(too_small->snapshot().inflight == 0 && too_small->snapshot().accepted == 0, "rejected request leaked reservation");
}
void ready_result_shutdown() {
    Gate delivery;
    SdkConfig config;
    config.byte_budget = DepthSdk::reservation_bytes(1, 1);
    auto sdk = SdkTestAccess::create(config, output, [&] { delivery.block(); });
    const auto input = image();
    auto run = std::async(std::launch::async, [&] { return sdk->run(input); });
    delivery.wait();
    const auto pending = sdk->snapshot();
    expect(pending.succeeded == 1 && pending.executing == 0 && pending.inflight == 1
           && pending.reserved_bytes == config.byte_budget, "ready result lost admission reservation before delivery");
    error(SdkErrorCode::Busy, [&] { sdk->run(input); }, "ready result must occupy byte budget");
    auto shutdown_one = std::async(std::launch::async, [&] { sdk->shutdown(); });
    SdkTestAccess::wait(*sdk, [](const SdkSnapshot& state) { return !state.accepting; });
    auto shutdown_two = std::async(std::launch::async, [&] { sdk->shutdown(); });
    error(SdkErrorCode::Closed, [&] { sdk->run(input); }, "run during shutdown must be Closed");
    expect(!sdk->snapshot().shutdown_complete, "shutdown finished while result awaited delivery");
    delivery.release();
    const auto result = run.get();
    shutdown_one.get();
    shutdown_two.get();
    sdk->shutdown();
    const auto state = sdk->snapshot();
    expect(state.shutdown_complete && !state.accepting && state.inflight == 0 && state.reserved_bytes == 0,
           "repeated/concurrent shutdown state");
    expect(result.values()[0] == 7, "accepted result not preserved through shutdown");
    error(SdkErrorCode::Closed, [&] { sdk->run(input); }, "run after shutdown must be Closed");
}
void drain_waiting_on_shutdown() {
    Gate gate;
    std::atomic<int> calls{0};
    SdkConfig config;
    config.queue_capacity = 1;
    auto sdk = SdkTestAccess::create(config, [&](const RgbImage& input) {
        if (++calls <= 4) gate.block();
        return output(input);
    });
    const auto input = image();
    std::vector<std::future<DepthMap>> running;
    for (int index = 0; index < 4; ++index) {
        running.push_back(std::async(std::launch::async, [&] { return sdk->run(input); }));
        gate.wait(static_cast<std::size_t>(index + 1));
    }
    auto fifth = std::async(std::launch::async, [&] { return sdk->run(input); });
    SdkTestAccess::wait(*sdk, [](const SdkSnapshot& state) { return state.queued == 1; });
    auto shutdown = std::async(std::launch::async, [&] { sdk->shutdown(); });
    SdkTestAccess::wait(*sdk, [](const SdkSnapshot& state) { return !state.accepting; });
    error(SdkErrorCode::Closed, [&] { sdk->run(input); }, "shutdown admitted new work");
    expect(!sdk->snapshot().shutdown_complete, "shutdown finished while four backends were blocked");
    gate.release();
    for (auto& future : running) {
        const auto result = future.get();
        expect(result.values().size() == 1, "executing work lost on shutdown");
    }
    const auto queued_result = fifth.get();
    expect(queued_result.values().size() == 1, "waiting work lost on shutdown");
    shutdown.get();
    const auto state = sdk->snapshot();
    expect(calls == 5 && state.delivered == 5 && state.shutdown_complete && state.inflight == 0
           && state.reserved_bytes == 0, "drain did not complete all five requests");
}
void errors_and_reuse() {
    Gate initial, recovered;
    auto sdk = SdkTestAccess::create({}, [&](const RgbImage& input) {
        const auto value = input.pixels()[0];
        if (value <= 4) initial.block();
        else recovered.block();
        switch (value) {
            case 1: throw std::runtime_error("intentional fake backend failure");
            case 2: throw 42;
            case 3: return DepthMap(2, 2, std::vector<float>(4));
            default: return output(input);
        }
    });
    auto moved_from = image();
    const auto valid = std::move(moved_from);
    error(SdkErrorCode::InvalidInput, [&] { sdk->run(moved_from); }, "moved-from input must be InvalidInput");
    expect(sdk->snapshot().reserved_bytes == 0 && sdk->snapshot().accepted == 0, "invalid input reserved resources");
    std::vector<std::future<DepthMap>> running;
    for (std::uint8_t value = 1; value <= 4; ++value)
        running.push_back(std::async(std::launch::async, [&, value] { return sdk->run(image(value)); }));
    initial.wait(4);
    initial.release();
    for (int index = 0; index < 3; ++index)
        error(SdkErrorCode::BackendFailure, [&] { running[index].get(); }, "backend failure class incorrect");
    const auto result = running[3].get();
    const auto failed = sdk->snapshot();
    expect(result.values()[0] == 4 && failed.failed == 3 && failed.succeeded == 1 && failed.delivered == 4
           && failed.inflight == 0 && failed.reserved_bytes == 0, "parallel failure leaked reservation or damaged successful peer");
    running.clear();
    for (std::uint8_t value = 10; value < 14; ++value)
        running.push_back(std::async(std::launch::async, [&, value] { return sdk->run(image(value)); }));
    recovered.wait(4);  // Every worker remains available after mixed backend failures.
    recovered.release();
    for (std::size_t index = 0; index < running.size(); ++index) {
        const auto recovered_result = running[index].get();
        expect(recovered_result.values()[0] == index + 10, "worker not reusable after failure");
    }
    const auto state = sdk->snapshot();
    expect(state.accepted == 8 && state.failed == 3 && state.succeeded == 5 && state.delivered == 8
           && state.inflight == 0 && state.reserved_bytes == 0, "recovery terminal accounting");
    sdk->shutdown();
    sdk.reset();
    expect(result.values()[0] == 4, "parallel result did not outlive recovered pool");
}
struct Temporary {
    std::filesystem::path path;
    Temporary() {
        std::string pattern = (std::filesystem::temp_directory_path() / "miniinfer-depth-sdk-XXXXXX").string();
        std::vector<char> buffer(pattern.begin(), pattern.end());
        buffer.push_back('\0');
        const char* created = ::mkdtemp(buffer.data());
        if (!created) throw std::runtime_error("cannot create test package directory");
        path = created;
    }
    ~Temporary() { std::error_code failure; std::filesystem::remove_all(path, failure); }
};
void initialization() {
    SdkConfig config;
    config.queue_capacity = 0;
    error(SdkErrorCode::InvalidInput, [&] { DepthSdk sdk(config); }, "zero queue capacity initialization");
    config.queue_capacity = 4;
    config.byte_budget = 0;
    error(SdkErrorCode::InvalidInput, [&] { DepthSdk sdk(config); }, "zero byte budget initialization");
    error(SdkErrorCode::InvalidInput, [&] { SdkTestAccess::create({}, {}); }, "empty injected backend accepted");
    config = {};
    config.worker_count = 0;
    error(SdkErrorCode::InvalidInput, [&] { SdkTestAccess::create(config, output); }, "zero worker count initialization");
    config.worker_count = std::numeric_limits<std::size_t>::max();
    config.queue_capacity = 1;
    error(SdkErrorCode::InvalidInput, [&] { SdkTestAccess::create(config, output); }, "worker plus queue capacity overflow accepted");
    Temporary temporary;
    const auto package = temporary.path / "package";
    std::filesystem::create_directory(package);
    config = {};
    config.package_dir = package;
    error(SdkErrorCode::ModelIncompatible, [&] { DepthSdk sdk(config); }, "missing manifest category");
    const auto model = package / "model.bin";
    write_text_exclusive(model, "not a real model");
    const Manifest base{{"schema", "miniinfer-model-1"}, {"contract", "depth-v1"}, {"backend", "ort"},
                        {"precision", "fp32"}, {"model", "model.bin"}, {"model_sha256", file_sha256(model)}};
    const auto manifest = package / "manifest.txt";
    auto set_manifest = [&](const Manifest& value) {
        std::filesystem::remove(manifest);
        write_text_exclusive(manifest, manifest_text(value));
    };
    for (const auto& pair : std::vector<std::pair<std::string, std::string>>{
             {"schema", "wrong"}, {"contract", "wrong"}, {"backend", "unknown"}, {"precision", "fp16"},
             {"model", "../model.bin"}, {"model", "/model.bin"}, {"model", "folder\\model.bin"},
             {"model_sha256", std::string(64, '0')}, {"extra", "field"}}) {
        auto invalid = base;
        invalid[pair.first] = pair.second;
        set_manifest(invalid);
        error(SdkErrorCode::ModelIncompatible, [&] { DepthSdk sdk(config); }, "invalid model package accepted");
    }
    std::filesystem::remove(manifest);
    write_text_exclusive(manifest, manifest_text(base) + "contract=depth-v1\n");
    error(SdkErrorCode::ModelIncompatible, [&] { DepthSdk sdk(config); }, "duplicate package field accepted");
    const auto outside = temporary.path / "outside.bin";
    write_text_exclusive(outside, "outside");
    std::filesystem::create_symlink(outside, package / "escape.bin");
    auto escaped = base;
    escaped["model"] = "escape.bin";
    escaped["model_sha256"] = file_sha256(outside);
    set_manifest(escaped);
    error(SdkErrorCode::ModelIncompatible, [&] { DepthSdk sdk(config); }, "symlink path escape accepted");
    auto trt = base;
    trt["backend"] = "trt";
    trt["precision"] = "fp32";
    write_text_exclusive(model.string() + ".manifest", "precision=fp16\n");
    set_manifest(trt);
    error(SdkErrorCode::ModelIncompatible, [&] { DepthSdk sdk(config); }, "TRT package versus engine precision mismatch accepted");
    set_manifest(base);
#if defined(MINIINFER_SDK_HAS_ORT)
    error(SdkErrorCode::BackendFailure, [&] { DepthSdk sdk(config); }, "broken backend initialization category");
#else
    error(SdkErrorCode::ModelIncompatible, [&] { DepthSdk sdk(config); }, "unavailable backend initialization category");
#endif
}
}  // namespace

int main() {
    try {
        ownership();
        worker_backend_ownership();
        partial_backend_initialization();
        count_and_parallelism();
        count_and_parallelism(1);
        byte_admission();
        ready_result_shutdown();
        drain_waiting_on_shutdown();
        errors_and_reuse();
        initialization();
        std::cout << "Depth SDK contract passed\n";
        return 0;
    } catch (const std::exception& failure) {
        std::cerr << "Depth SDK contract failed: " << failure.what() << '\n';
        return 1;
    }
}
