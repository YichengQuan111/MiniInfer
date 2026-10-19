#include "miniinfer/depth/sdk.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// Real TensorRT lifecycle/association checks, without backend injection or GPU
// timing assumptions. Run under the CTest process timeout as the final deadlock
// guard; condition-variable deadlines below only guard test coordination.
namespace {
using namespace miniinfer::depth;
constexpr std::size_t submitter_count = 4;
constexpr std::size_t shutdown_count = 2;
constexpr auto coordination_timeout = std::chrono::seconds(120);

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void conservation(const SdkSnapshot& state) {
    require(state.inflight == state.queued + state.executing + state.ready, "inflight state conservation failed");
    require(state.accepted == state.delivered + state.abandoned_completed + state.inflight,
            "accepted/delivery/abandonment conservation failed");
    require(state.succeeded + state.failed == state.delivered + state.abandoned_completed + state.ready,
            "terminal backend/publication conservation failed");
    require(state.abandoned_completed <= state.abandoned, "completed abandonment exceeds abandoned handles");
    require(state.failed == 0 && !state.fatal_backend, "unexpected real backend failure");
    require((state.inflight == 0) == (state.reserved_bytes == 0), "byte quota and inflight lifetime disagree");
}
void drained(const SdkSnapshot& state) {
    conservation(state);
    require(state.shutdown_complete && !state.accepting && state.queued == 0 && state.executing == 0,
            "shutdown did not close admission and drain execution");
    require(state.succeeded == state.accepted && state.failed == 0, "shutdown lost an accepted backend terminal");
    require(state.io_device_bytes == 0 && state.io_pinned_bytes == 0 && state.host_workspace_bytes == 0,
            "shutdown retained fixed backend buffers");
}
void unique_id(std::set<std::uint64_t>& ids, const DepthRequest& request) {
    require(request.valid() && request.id() != 0 && ids.insert(request.id()).second,
            "accepted request ID is invalid or duplicated within an SDK");
}
void rejected(DepthSdk& sdk, const RgbImage& image, SdkErrorCode expected) {
    try {
        auto unexpected = sdk.submit(image);
        unexpected.abandon();
    } catch (const SdkError& error) {
        require(error.code() == expected && error.request_id() == 0, "admission rejection has wrong class or accepted ID");
        return;
    }
    throw std::runtime_error("submission was accepted when rejection was required");
}
DepthMap consume(DepthRequest& request, const DepthMap& expected, std::size_t h2d_bytes) {
    const auto id = request.id();
    require(request.wait_for(std::chrono::duration_cast<std::chrono::milliseconds>(coordination_timeout)),
            "real GPU result did not become ready within the test deadline");
    DepthTiming timing;
    auto result = request.get(&timing);
    require(!request.valid() && id != 0, "get did not consume its accepted handle");
    require(result.width() == expected.width() && result.height() == expected.height()
            && result.values() == expected.values(), "full-map result differs from its two-image serial baseline");
    validate_finite(result.values());
    require(timing.accepted && timing.ready_at != std::chrono::steady_clock::time_point{}
            && timing.ready_at <= std::chrono::steady_clock::now() && timing.h2d_bytes == h2d_bytes,
            "accepted result lost its terminal/transfer timing association");
    return result;
}

struct Observation {
    const char* scenario;
    std::size_t slots;
    SdkSnapshot before_close, last_live;
    std::size_t unready_observed_before_abandon = 0;
    std::size_t deliveries_after_destruction = 0;
};
void print_snapshot(const SdkSnapshot& state) {
    std::cout << "{\"queued\":" << state.queued << ",\"executing\":" << state.executing
        << ",\"ready\":" << state.ready << ",\"inflight\":" << state.inflight
        << ",\"reserved_bytes\":" << state.reserved_bytes << ",\"accepted\":" << state.accepted
        << ",\"succeeded\":" << state.succeeded << ",\"failed\":" << state.failed
        << ",\"delivered\":" << state.delivered << ",\"abandoned\":" << state.abandoned
        << ",\"abandoned_completed\":" << state.abandoned_completed
        << ",\"rejected_busy\":" << state.rejected_busy << ",\"rejected_closed\":" << state.rejected_closed << '}';
}

Observation pending_and_outliving(SdkConfig config, const std::array<RgbImage, 2>& images,
    const std::vector<DepthMap>& expected, const std::array<std::size_t, 2>& h2d) {
    Observation observation{"abandon-and-outlive", config.gpu_slots, {}, {}};
    DepthRequest retained_b;
    std::vector<DepthMap> delivered;
    std::set<std::uint64_t> ids;
    const auto bytes_a = DepthSdk::reservation_bytes(images[0].width(), images[0].height());
    const auto bytes_b = DepthSdk::reservation_bytes(images[1].width(), images[1].height());
    {
        DepthSdk sdk(config);
        std::vector<DepthRequest> requests;
        // Retain one of each image; abandon the other two immediately without
        // waiting. A fast GPU may already have completed: report the observation
        // rather than assert that a request must still be pending at this instant.
        for (std::size_t index = 0; index < 4; ++index) {
            const std::size_t image = index / 2;
            requests.push_back(sdk.submit(images[image]));
            unique_id(ids, requests.back());
            if (index % 2) {
                observation.unready_observed_before_abandon += !requests.back().ready();
                requests.back().abandon();
                requests.back().abandon();
                require(!requests.back().valid(), "abandon did not invalidate its handle");
            }
        }
        observation.before_close = sdk.snapshot();
        conservation(observation.before_close);
        require(observation.before_close.abandoned == 2, "immediate abandonment was not counted once");
        sdk.shutdown();
        const auto closed = sdk.snapshot();
        drained(closed);
        require(closed.accepted == 4 && closed.abandoned_completed == 2 && closed.ready == 2
                && closed.inflight == 2 && closed.reserved_bytes == bytes_a + bytes_b && closed.delivered == 0,
                "shutdown did not preserve ready handles while draining abandoned work");
        rejected(sdk, images[0], SdkErrorCode::Closed);
        delivered.push_back(consume(requests[0], expected[0], h2d[0]));
        retained_b = std::move(requests[2]);
        observation.last_live = sdk.snapshot();
        drained(observation.last_live);
        require(observation.last_live.delivered == 1 && observation.last_live.inflight == 1
                && observation.last_live.ready == 1 && observation.last_live.reserved_bytes == bytes_b,
                "get after shutdown did not release just its own reservation");
    }
    // No SDK pointer is used here. The last-live ledger above deliberately still
    // has one ready request; do not fabricate a post-destruction SDK snapshot.
    delivered.push_back(consume(retained_b, expected[1], h2d[1]));
    require(delivered[0].values() == expected[0].values() && delivered[1].values() == expected[1].values(),
            "SDK destruction invalidated an owned result or retained delivery");
    observation.deliveries_after_destruction = 1;
    return observation;
}

Observation ready_budget(SdkConfig config, const std::array<RgbImage, 2>& images,
    const std::vector<DepthMap>& expected, const std::array<std::size_t, 2>& h2d) {
    const auto bytes_a = DepthSdk::reservation_bytes(images[0].width(), images[0].height());
    const auto bytes_b = DepthSdk::reservation_bytes(images[1].width(), images[1].height());
    config.byte_budget = bytes_a + bytes_b;
    DepthSdk sdk(config);
    Observation observation{"ready-budget", config.gpu_slots, {}, {}};
    std::set<std::uint64_t> ids;
    auto first = sdk.submit(images[0]); unique_id(ids, first); first.wait();
    auto second = sdk.submit(images[1]); unique_id(ids, second); second.wait();
    auto state = sdk.snapshot();
    conservation(state);
    require(state.ready == 2 && state.inflight == 2 && state.queued == 0 && state.executing == 0
            && state.reserved_bytes == config.byte_budget, "unconsumed ready results did not fill the exact byte budget");
    rejected(sdk, images[0], SdkErrorCode::Busy);
    consume(first, expected[0], h2d[0]);
    state = sdk.snapshot(); conservation(state);
    require(state.ready == 1 && state.reserved_bytes == bytes_b && state.delivered == 1,
            "consuming ready result did not return its byte quota");
    auto replacement_a = sdk.submit(images[0]); unique_id(ids, replacement_a); replacement_a.wait();
    rejected(sdk, images[1], SdkErrorCode::Busy);
    second.abandon(); second.abandon();
    state = sdk.snapshot(); conservation(state);
    require(state.ready == 1 && state.reserved_bytes == bytes_a && state.abandoned_completed == 1,
            "abandoning ready result did not return its byte quota exactly once");
    auto replacement_b = sdk.submit(images[1]); unique_id(ids, replacement_b); replacement_b.wait();
    observation.before_close = sdk.snapshot(); conservation(observation.before_close);
    require(observation.before_close.ready == 2 && observation.before_close.reserved_bytes == config.byte_budget,
            "returned ready quota was not reusable");
    sdk.shutdown();
    // Closed takes precedence over Busy even though ready handles still occupy
    // the whole byte budget after backend teardown.
    rejected(sdk, images[0], SdkErrorCode::Closed);
    consume(replacement_b, expected[1], h2d[1]);
    consume(replacement_a, expected[0], h2d[0]);
    observation.last_live = sdk.snapshot(); drained(observation.last_live);
    require(observation.last_live.accepted == 4 && observation.last_live.delivered == 3
            && observation.last_live.abandoned_completed == 1 && observation.last_live.inflight == 0
            && observation.last_live.rejected_busy == 2 && observation.last_live.rejected_closed == 1,
            "ready admission/delivery/abandonment counters disagree");
    return observation;
}

struct Race {
    std::mutex mutex;
    std::condition_variable changed;
    std::size_t arrived = 0, warmup_done = 0, closers_ready = 0, shutdown_done = 0;
    bool start = false, race_open = false, aborted = false;
    std::exception_ptr failure;
    template<class Predicate> void wait(std::unique_lock<std::mutex>& lock, Predicate predicate) {
        require(changed.wait_for(lock, coordination_timeout, [&] { return aborted || predicate(); }),
                "thread coordination timed out");
        if (aborted) throw std::runtime_error("thread coordination aborted by another failure");
    }
    void begin() {
        std::unique_lock<std::mutex> lock(mutex);
        ++arrived; changed.notify_all();
        wait(lock, [&] { return start; });
    }
    void fail(std::exception_ptr error) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!failure) failure = std::move(error);
        aborted = true; changed.notify_all();
    }
};
struct Counts {
    std::mutex mutex;
    std::set<std::uint64_t> ids;
    std::size_t accepted = 0, delivered = 0, abandoned = 0, busy = 0, closed = 0, unready = 0;
};
struct OwnedRequest { DepthRequest request; std::size_t image; };

Observation concurrent_close(SdkConfig config, const std::array<RgbImage, 2>& images,
    const std::vector<DepthMap>& expected, const std::array<std::size_t, 2>& h2d) {
    DepthSdk sdk(config);
    Observation observation{"concurrent-submit-close", config.gpu_slots, {}, {}};
    Counts counts;
    Race race;
    auto submit = [&](std::size_t image, bool abandon, bool closed_required, std::vector<OwnedRequest>& retained) {
        try {
            auto request = sdk.submit(images[image]);
            if (closed_required) {
                request.abandon();
                throw std::runtime_error("post-shutdown submission was accepted");
            }
            {
                std::lock_guard<std::mutex> lock(counts.mutex);
                unique_id(counts.ids, request); ++counts.accepted;
            }
            if (abandon) {
                const bool unready = !request.ready();
                request.abandon();
                std::lock_guard<std::mutex> lock(counts.mutex);
                ++counts.abandoned; counts.unready += unready;
            } else retained.push_back({std::move(request), image});
        } catch (const SdkError& error) {
            require(error.request_id() == 0, "admission rejection incorrectly owns a request ID");
            if (closed_required) require(error.code() == SdkErrorCode::Closed, "post-shutdown rejection must be Closed");
            if (error.code() == SdkErrorCode::Closed) {
                require(!sdk.snapshot().accepting, "Closed reported while SDK still accepts work");
                std::lock_guard<std::mutex> lock(counts.mutex); ++counts.closed;
            } else if (error.code() == SdkErrorCode::Busy) {
                std::lock_guard<std::mutex> lock(counts.mutex); ++counts.busy;
            } else throw;
        }
        conservation(sdk.snapshot());
    };
    std::vector<OwnedRequest> seeds;
    submit(0, false, false, seeds);
    submit(1, false, false, seeds);
    submit(0, true, false, seeds);
    require(seeds.size() == 2 && counts.accepted == 3 && counts.abandoned == 1,
            "real-GPU race seeds were not admitted before concurrency");
    std::vector<std::thread> threads;
    threads.reserve(submitter_count + shutdown_count);
    try {
        for (std::size_t worker = 0; worker < submitter_count; ++worker) {
            threads.emplace_back([&, worker] {
                try {
                    race.begin();
                    std::vector<OwnedRequest> retained;
                    for (std::size_t attempt = 0; attempt < 2; ++attempt)
                        submit((worker + attempt) % 2, attempt == 0, false, retained);
                    {
                        std::unique_lock<std::mutex> lock(race.mutex);
                        ++race.warmup_done; race.changed.notify_all();
                        race.wait(lock, [&] { return race.race_open; });
                    }
                    for (std::size_t attempt = 0; attempt < 6; ++attempt)
                        submit((worker + attempt) % 2, attempt % 3 == 0, false, retained);
                    {
                        std::unique_lock<std::mutex> lock(race.mutex);
                        race.wait(lock, [&] { return race.shutdown_done == shutdown_count; });
                    }
                    submit(worker % 2, false, true, retained);
                    for (auto& item : retained) {
                        consume(item.request, expected[item.image], h2d[item.image]);
                        std::lock_guard<std::mutex> lock(counts.mutex); ++counts.delivered;
                    }
                } catch (...) { race.fail(std::current_exception()); }
            });
        }
        for (std::size_t closer = 0; closer < shutdown_count; ++closer) {
            threads.emplace_back([&] {
                try {
                    race.begin();
                    {
                        std::unique_lock<std::mutex> lock(race.mutex);
                        race.wait(lock, [&] { return race.warmup_done == submitter_count; });
                        ++race.closers_ready;
                        if (race.closers_ready == shutdown_count) {
                            observation.before_close = sdk.snapshot();
                            conservation(observation.before_close);
                            race.race_open = true; race.changed.notify_all();
                        } else race.wait(lock, [&] { return race.race_open; });
                    }
                    // Both closers and all submitters leave the same CV gate.
                    // Accepted/Busy/Closed is schedule-dependent and checked by
                    // ledger counts, not by assuming a particular GPU duration.
                    sdk.shutdown();
                    std::lock_guard<std::mutex> lock(race.mutex);
                    ++race.shutdown_done; race.changed.notify_all();
                } catch (...) { race.fail(std::current_exception()); }
            });
        }
        {
            std::unique_lock<std::mutex> lock(race.mutex);
            race.wait(lock, [&] { return race.arrived == submitter_count + shutdown_count; });
            race.start = true; race.changed.notify_all();
        }
    } catch (...) { race.fail(std::current_exception()); }
    for (auto& thread : threads) if (thread.joinable()) thread.join();
    if (race.failure) std::rethrow_exception(race.failure);
    require(race.shutdown_done == shutdown_count, "both shutdown callers did not return");
    for (auto& item : seeds) {
        consume(item.request, expected[item.image], h2d[item.image]);
        ++counts.delivered;
    }
    sdk.shutdown(); // An additional idempotent call must preserve the ledger.
    observation.last_live = sdk.snapshot(); drained(observation.last_live);
    observation.unready_observed_before_abandon = counts.unready;
    const auto& state = observation.last_live;
    require(state.inflight == 0 && state.reserved_bytes == 0 && state.ready == 0,
            "concurrent shutdown/delivery leaked a reservation");
    require(state.accepted == counts.accepted && counts.ids.size() == counts.accepted
            && state.delivered == counts.delivered && state.abandoned == counts.abandoned
            && state.abandoned_completed == counts.abandoned
            && state.rejected_busy == counts.busy && state.rejected_closed == counts.closed
            && counts.closed >= submitter_count && state.accepted == counts.delivered + counts.abandoned,
            "observed concurrent calls disagree with SDK accepted/terminal/rejection counters");
    return observation;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 5)
            throw std::invalid_argument("usage: test_depth_async_stress_gpu PACKAGE IMAGE_A IMAGE_B MODE");
        const std::array<RgbImage, 2> images{read_ppm(argv[2]), read_ppm(argv[3])};
        require(images[0].width() != images[1].width() || images[0].height() != images[1].height()
                || images[0].pixels() != images[1].pixels(), "mixed-image test requires distinct inputs");
        SdkConfig config;
        config.package_dir = argv[1];
        config.preprocess_mode = parse_preprocess_mode(argv[4]);
        config.worker_count = 4; config.queue_capacity = 8;
        config.reuse_host_workspace = true; config.reuse_gpu_buffers = true;
        config.byte_budget = (config.worker_count + config.queue_capacity) * std::max(
            DepthSdk::reservation_bytes(images[0].width(), images[0].height()),
            DepthSdk::reservation_bytes(images[1].width(), images[1].height()));
        std::vector<DepthMap> expected;
        std::array<std::size_t, 2> h2d{};
        {
            DepthSdk serial(config);
            require(serial.snapshot().backend == "trt", "stress test requires a real TensorRT SDK package");
            for (std::size_t image = 0; image < images.size(); ++image) {
                DepthTiming timing;
                expected.push_back(serial.run(images[image], &timing));
                h2d[image] = timing.h2d_bytes;
                require(timing.accepted && timing.h2d_bytes > 0, "serial baseline did not execute real GPU transfers");
            }
            serial.shutdown(); drained(serial.snapshot());
        }
        require(expected[0].width() != expected[1].width() || expected[0].height() != expected[1].height()
                || expected[0].values() != expected[1].values(), "serial images cannot distinguish result association");
        const bool full = config.preprocess_mode == PreprocessMode::CudaFullBasic
                       || config.preprocess_mode == PreprocessMode::CudaFullFused;
        if (full) for (std::size_t image = 0; image < images.size(); ++image)
            require(h2d[image] == images[image].pixels().size(), "full CUDA mode did not upload RGB8 source bytes");
        std::vector<Observation> observations;
        for (std::size_t slots : {1U, 2U}) {
            config.gpu_slots = slots;
            observations.push_back(pending_and_outliving(config, images, expected, h2d));
            observations.push_back(ready_budget(config, images, expected, h2d));
            observations.push_back(concurrent_close(config, images, expected, h2d));
        }
        std::cout << "{\"passed\":true,\"mode\":\"" << preprocess_mode_name(config.preprocess_mode)
            << "\",\"serial_baselines\":2,\"cases\":[";
        for (std::size_t index = 0; index < observations.size(); ++index) {
            if (index) std::cout << ',';
            const auto& observation = observations[index];
            std::cout << "{\"scenario\":\"" << observation.scenario << "\",\"gpu_slots\":" << observation.slots
                << ",\"before_close\":"; print_snapshot(observation.before_close);
            std::cout << ",\"last_live_snapshot\":"; print_snapshot(observation.last_live);
            std::cout << ",\"unready_observed_before_abandon\":" << observation.unready_observed_before_abandon
                << ",\"deliveries_after_destruction\":" << observation.deliveries_after_destruction << '}';
        }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Real GPU async stress failed: " << error.what() << '\n';
        return 1;
    }
}
