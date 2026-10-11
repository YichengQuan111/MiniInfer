#include "depth_sdk_access.hpp"
#include "miniinfer/depth/artifact.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
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
    bool entered = false, open = false;
    void block() {
        std::unique_lock<std::mutex> lock(mutex);
        entered = true;
        changed.notify_all();
        if (!changed.wait_for(lock, std::chrono::seconds(5), [this] { return open; }))
            throw std::runtime_error("backend/delivery test gate timed out");
    }
    void wait() {
        std::unique_lock<std::mutex> lock(mutex);
        if (!changed.wait_for(lock, std::chrono::seconds(5), [this] { return entered; }))
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
void count_and_serialization() {
    Gate gate;
    std::atomic<int> calls{0}, active{0}, maximum{0};
    SdkConfig config;
    config.queue_capacity = 1;
    auto sdk = SdkTestAccess::create(config, [&](const RgbImage& input) {
        const int current = ++active;
        maximum.store(std::max(maximum.load(), current));
        if (++calls == 1) gate.block();
        auto depth = output(input);
        --active;
        return depth;
    });
    const auto input = image();
    auto first = std::async(std::launch::async, [&] { return sdk->run(input); });
    gate.wait();
    auto second = std::async(std::launch::async, [&] { return sdk->run(input); });
    SdkTestAccess::wait(*sdk, [](const SdkSnapshot& state) { return state.queued == 1; });
    expect(sdk->snapshot().inflight == 2, "count includes one execution and one waiting request");
    error(SdkErrorCode::Busy, [&] { sdk->run(input); }, "count exhaustion must be Busy");
    gate.release();
    expect(first.get().width() == 1 && second.get().width() == 1, "accepted work completes");
    expect(maximum == 1, "more than one backend call overlapped");
    const auto state = sdk->snapshot();
    expect(state.accepted == 2 && state.succeeded == 2 && state.delivered == 2 && state.inflight == 0
           && state.reserved_bytes == 0 && state.rejected_busy == 1, "count terminal accounting");
}
void byte_admission() {
    Gate gate;
    SdkConfig config;
    config.byte_budget = DepthSdk::reservation_bytes(1, 1);
    auto sdk = SdkTestAccess::create(config, [&](const RgbImage& input) { gate.block(); return output(input); });
    const auto input = image();
    auto first = std::async(std::launch::async, [&] { return sdk->run(input); });
    gate.wait();
    expect(sdk->snapshot().reserved_bytes == config.byte_budget, "exact byte budget reservation");
    error(SdkErrorCode::Busy, [&] { sdk->run(input); }, "byte exhaustion must be Busy");
    gate.release();
    first.get();
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
        if (++calls == 1) gate.block();
        return output(input);
    });
    const auto input = image();
    auto first = std::async(std::launch::async, [&] { return sdk->run(input); });
    gate.wait();
    auto second = std::async(std::launch::async, [&] { return sdk->run(input); });
    SdkTestAccess::wait(*sdk, [](const SdkSnapshot& state) { return state.queued == 1; });
    auto shutdown = std::async(std::launch::async, [&] { sdk->shutdown(); });
    SdkTestAccess::wait(*sdk, [](const SdkSnapshot& state) { return !state.accepting; });
    error(SdkErrorCode::Closed, [&] { sdk->run(input); }, "shutdown admitted new work");
    gate.release();
    const auto first_result = first.get();
    const auto second_result = second.get();
    expect(first_result.values().size() == 1, "executing work lost on shutdown");
    expect(second_result.values().size() == 1, "waiting work lost on shutdown");
    shutdown.get();
    expect(calls == 2 && sdk->snapshot().delivered == 2, "drain did not complete both requests");
}
void errors_and_reuse() {
    std::atomic<int> calls{0};
    auto sdk = SdkTestAccess::create({}, [&](const RgbImage& input) {
        switch (++calls) {
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
    for (int index = 0; index < 3; ++index) {
        error(SdkErrorCode::BackendFailure, [&] { sdk->run(valid); }, "backend failure class incorrect");
        expect(sdk->snapshot().inflight == 0 && sdk->snapshot().reserved_bytes == 0, "failed request leaked reservation");
    }
    const auto result = sdk->run(valid);
    expect(result.values()[0] == 7 && sdk->snapshot().failed == 3 && sdk->snapshot().succeeded == 1, "backend not reusable after failure");
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
        count_and_serialization();
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
