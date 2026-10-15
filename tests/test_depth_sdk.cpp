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
#include <type_traits>

namespace {
using namespace miniinfer::depth;
using miniinfer::depth::detail::SdkTestAccess;
static_assert(!std::is_copy_constructible_v<DepthRequest> && !std::is_copy_assignable_v<DepthRequest>);
static_assert(std::is_nothrow_move_constructible_v<DepthRequest> && std::is_nothrow_move_assignable_v<DepthRequest>);
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
    shutdown.get();
    expect(sdk->snapshot().shutdown_complete && destroyed == worker_count
           && std::all_of(lifetimes.begin(), lifetimes.end(), [](const auto& lifetime) { return lifetime.expired(); }),
           "backend resources must be released after execution while get delivery remains blocked");
    delivery.release();
    for (auto& future : running) results.push_back(future.get());
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
    DepthTiming timing;
    const auto later = sdk->run(original, &timing);
    expect(timing.accepted && timing.admission_ms >= 0 && timing.queue_ms >= 0 && timing.backend_ms >= 0,
           "accepted request timing is available after return");
    expect(timing.h2d_bytes == 0, "fake CPU backend must not report GPU transfer bytes");
    expect(later.values()[0] == 99 && result.values()[0] == 7, "later run changed previous result");
    sdk->shutdown();
    error(SdkErrorCode::Closed, [&] { sdk->run(original, &timing); }, "closed timing rejection");
    expect(!timing.accepted, "rejected request must not be counted as accepted");
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
    shutdown_one.get();
    shutdown_two.get();
    expect(sdk->snapshot().shutdown_complete, "shutdown must finish without consuming ready handles");
    delivery.release();
    const auto result = run.get();
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
template<class Callable> void invalid_handle(Callable callable) {
    try { callable(); } catch (const std::logic_error&) { return; }
    throw std::runtime_error("invalid/moved/consumed request method succeeded");
}
void async_ownership_and_lifetime() {
    Gate gate;
    auto sdk = SdkTestAccess::create({}, [&](const RgbImage& input) { gate.block(); return output(input); });
    auto source = image(31,2,2);
    auto request = sdk->submit(source);
    const auto id = request.id();
    expect(id != 0 && request.valid(), "accepted request has no ID/valid handle");
    gate.wait();
    source = image(99,2,2);
    expect(!request.ready() && !request.wait_for(std::chrono::milliseconds(1)), "unfinished request became ready");
    try { request.wait_for(std::chrono::milliseconds(-1)); throw std::runtime_error("negative wait allowed"); }
    catch (const std::invalid_argument&) {}
    auto moved = std::move(request);
    expect(!request.valid() && moved.id() == id, "move changed request identity");
    invalid_handle([&] { request.id(); });
    invalid_handle([&] { request.ready(); });
    invalid_handle([&] { request.wait(); });
    invalid_handle([&] { request.get(); });
    gate.release(); moved.wait();
    expect(moved.ready(), "wait returned before ready");
    sdk.reset(); // Unconsumed ready handle remains valid after SDK destruction.
    DepthTiming timing;
    const auto result = moved.get(&timing);
    expect(result.values() == std::vector<float>(4,31), "async input was borrowed or result died with SDK");
    expect(timing.accepted && timing.ready_at != std::chrono::steady_clock::time_point{}, "actual ready timestamp missing");
    expect(timing.ready_at <= std::chrono::steady_clock::now(), "ready timestamp lies in future");
    invalid_handle([&] { moved.get(); });
    moved.abandon(); moved.abandon(); // Invalid abandonment is harmless.
}
void ready_quota_and_abandonment() {
    SdkConfig config; config.worker_count=1; config.queue_capacity=1;
    const auto bytes=DepthSdk::reservation_bytes(1,1);
    config.byte_budget=3*bytes; // Count bound reaches 2 first; bytes remain available.
    auto sdk=SdkTestAccess::create(config,output);
    auto first=sdk->submit(image(1)); first.wait();
    auto second=sdk->submit(image(2)); second.wait();
    auto state=sdk->snapshot();
    expect(state.executing==0 && state.queued==0 && state.ready==2 && state.inflight==2
           && state.reserved_bytes==2*bytes, "slow consumer escaped count/byte quota");
    error(SdkErrorCode::Busy,[&] { sdk->submit(image()); },"ready results must exhaust total quota");
    first.abandon(); first.abandon(); // Idempotent even before handle destruction.
    state=sdk->snapshot();
    expect(!first.valid() && state.abandoned==1 && state.abandoned_completed==1
           && state.ready==1 && state.inflight==1 && state.reserved_bytes==bytes,"ready abandon did not release quota");
    auto replacement=sdk->submit(image(3)); replacement.wait();
    const auto old_id=second.id();
    second=std::move(replacement); // Move assignment abandons its previous ready result.
    expect(second.id()!=old_id && !replacement.valid(),"move assignment did not transfer identity");
    expect(sdk->snapshot().abandoned_completed==2,"move assignment failed to abandon old handle");
    const auto replacement_result=second.get();
    expect(replacement_result.values()[0]==3,"replacement result association failed");
    state=sdk->snapshot();
    expect(state.accepted==3 && state.delivered==1 && state.abandoned_completed==2 && state.ready==0
           && state.inflight==0 && state.reserved_bytes==0,"ready-consumer ledger conservation");
}
void pending_abandonment_and_shutdown() {
    Gate gate; std::atomic<int> calls{0};
    SdkConfig config; config.worker_count=1; config.queue_capacity=1;
    auto sdk=SdkTestAccess::create(config,[&](const RgbImage& input) {
        if(++calls==1) gate.block();
        return output(input);
    });
    auto first=sdk->submit(image(10)); gate.wait();
    auto waiting=sdk->submit(image(20));
    const auto bytes=DepthSdk::reservation_bytes(1,1);
    first.abandon(); waiting.abandon(); first.abandon(); waiting.abandon();
    auto state=sdk->snapshot();
    expect(state.abandoned==2 && state.abandoned_completed==0 && state.executing==1
           && state.queued==1 && state.inflight==2 && state.reserved_bytes==2*bytes,
           "pending abandonment prematurely freed input/execution quota");
    error(SdkErrorCode::Busy,[&] { sdk->submit(image()); },"pending abandoned task lost quota");
    auto shutdown=std::async(std::launch::async,[&] { sdk->shutdown(); });
    SdkTestAccess::wait(*sdk,[](const SdkSnapshot& snapshot) {return !snapshot.accepting;});
    expect(!sdk->snapshot().shutdown_complete,"shutdown did not wait for executing abandoned task");
    error(SdkErrorCode::Closed,[&] { sdk->submit(image()); },"shutdown race accepted another task");
    gate.release(); shutdown.get();
    state=sdk->snapshot();
    expect(calls==2 && state.succeeded==2 && state.abandoned_completed==2 && state.ready==0
           && state.executing==0 && state.queued==0 && state.inflight==0 && state.reserved_bytes==0,
           "abandoned accepted tasks were not drained and released");
}
void two_slot_out_of_order() {
    Gate first_gate, second_gate;
    std::mutex order_mutex; std::vector<unsigned> started;
    SdkConfig config; config.worker_count=2; config.gpu_slots=2; config.queue_capacity=2;
    auto sdk=SdkTestAccess::create(config,[&](const RgbImage& input) {
        {std::lock_guard<std::mutex> lock(order_mutex); started.push_back(input.pixels()[0]);}
        if(input.pixels()[0]==11) first_gate.block();
        else if(input.pixels()[0]==22) second_gate.block();
        return output(input);
    });
    auto first=sdk->submit(image(11)); first_gate.wait();
    auto second=sdk->submit(image(22)); second_gate.wait();
    auto third=sdk->submit(image(33));
    auto fourth=sdk->submit(image(44));
    expect(first.id()<second.id() && second.id()<third.id() && third.id()<fourth.id(),"IDs are not unique monotonic associations");
    expect(sdk->snapshot().executing==2 && sdk->snapshot().queued==2,"two workers did not own separate active slots");
    error(SdkErrorCode::Busy,[&] { sdk->submit(image()); },"two-slot queue+slot reservation bound exceeded");
    second_gate.release();
    second.wait(); third.wait(); fourth.wait();
    expect(!first.ready(),"first request completed through a different slot");
    const auto second_result=second.get(), third_result=third.get(), fourth_result=fourth.get();
    expect(second_result.values()[0]==22 && third_result.values()[0]==33 && fourth_result.values()[0]==44,
           "out-of-order result/request association changed");
    {std::lock_guard<std::mutex> lock(order_mutex);
        expect(started==std::vector<unsigned>{11,22,33,44},"waiting requests did not dequeue FIFO");}
    first_gate.release();
    const auto first_result=first.get();
    expect(first_result.values()[0]==11,"slow first result corrupted by later slot work");
    auto state=sdk->snapshot();
    expect(state.accepted==4 && state.succeeded==4 && state.delivered==4 && state.inflight==0,
           "two-slot terminal accounting mismatch");
}
void simultaneous_shutdown_submit() {
    for(int iteration=0;iteration<12;++iteration) {
        Gate execution, race;
        SdkConfig config; config.worker_count=1; config.queue_capacity=1;
        std::atomic<int> calls{0};
        auto sdk=SdkTestAccess::create(config,[&](const RgbImage& input) {
            if(++calls==1) execution.block();
            return output(input);
        });
        auto first=sdk->submit(image(1)); execution.wait();
        auto submit=std::async(std::launch::async,[&] {
            race.block();
            try {return sdk->submit(image(2));}
            catch(const SdkError& failure) {
                expect(failure.code()==SdkErrorCode::Closed,"submit/shutdown race gave wrong error");
                return DepthRequest{};
            }
        });
        auto shutdown=std::async(std::launch::async,[&] {race.block(); sdk->shutdown();});
        auto shutdown_two=std::async(std::launch::async,[&] {race.block(); sdk->shutdown();});
        race.wait(3); race.release();
        auto second=submit.get();
        SdkTestAccess::wait(*sdk,[](const SdkSnapshot& state) {return !state.accepting;});
        execution.release(); shutdown.get(); shutdown_two.get();
        const bool second_accepted=second.valid();
        const auto first_result=first.get();
        expect(first_result.values()[0]==1,"shutdown race lost first accepted output");
        if(second_accepted) {
            const auto second_result=second.get();
            expect(second_result.values()[0]==2,"shutdown race lost newly accepted output");
        }
        const auto state=sdk->snapshot();
        expect(state.shutdown_complete && state.accepted==(second_accepted?2U:1U)
               && state.delivered==state.accepted && state.inflight==0 && state.reserved_bytes==0,
               "submit/shutdown race ledger did not conserve accepted terminals");
    }
}
void async_errors_and_fatal_stop() {
    auto sdk=SdkTestAccess::create({},[](const RgbImage&) -> DepthMap {throw std::runtime_error("ordinary");});
    auto request=sdk->submit(image()); const auto id=request.id();
    DepthTiming timing;
    try { request.get(&timing); throw std::runtime_error("missing correlated backend error"); }
    catch(const SdkError& failure) {
        expect(failure.code()==SdkErrorCode::BackendFailure && failure.request_id()==id,"failure lost request association");
    }
    expect(!request.valid() && timing.accepted && sdk->snapshot().inflight==0 && sdk->snapshot().accepting,
           "ordinary failure consumed quota or stopped whole SDK");
    invalid_handle([&] {request.get();});
    Gate fatal_gate;
    std::atomic<int> calls{0};
    SdkConfig config; config.worker_count=1; config.queue_capacity=2;
    auto fatal_sdk=SdkTestAccess::create(config,[&](const RgbImage&) -> DepthMap {
        ++calls; fatal_gate.block(); throw std::runtime_error("fatal test device error");
    },{},[](const std::exception_ptr&) {return true;});
    auto first=fatal_sdk->submit(image()); fatal_gate.wait();
    auto second=fatal_sdk->submit(image());
    fatal_gate.release(); first.wait(); second.wait();
    auto state=fatal_sdk->snapshot();
    expect(state.fatal_backend && !state.accepting && state.failed==2 && calls==1,
           "fatal error did not stop acceptance and explicitly fail queued tasks");
    for(auto* pending : {&first,&second}) {
        const auto pending_id=pending->id();
        try {pending->get();throw std::runtime_error("fatal request unexpectedly succeeded");}
        catch(const SdkError& failure) {expect(failure.request_id()==pending_id,"fatal terminal ID mismatch");}
    }
    error(SdkErrorCode::Closed,[&] {fatal_sdk->submit(image());},"fatal SDK resumed admission");
    expect(fatal_sdk->snapshot().inflight==0 && fatal_sdk->snapshot().reserved_bytes==0,
           "fatal task terminal leaked quota");
    auto delivery_sdk=SdkTestAccess::create({},output,[] {throw std::runtime_error("delivery test error");});
    auto delivery=delivery_sdk->submit(image()); const auto delivery_id=delivery.id();
    try {delivery.get();throw std::runtime_error("missing delivery exception");}
    catch(const SdkError& failure) {expect(failure.request_id()==delivery_id,"delivery error lost request ID");}
    state=delivery_sdk->snapshot();
    expect(!delivery.valid() && state.succeeded==1 && state.failed==0 && state.delivered==1
           && state.inflight==0 && state.reserved_bytes==0,"delivery exception leaked quota or changed execution terminal");
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
    config = {}; config.gpu_slots=3;
    error(SdkErrorCode::InvalidInput,[&] {SdkTestAccess::create(config,output);},"unsupported slot count accepted");
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
        async_ownership_and_lifetime();
        ready_quota_and_abandonment();
        pending_abandonment_and_shutdown();
        two_slot_out_of_order();
        simultaneous_shutdown_submit();
        async_errors_and_fatal_stop();
        initialization();
        std::cout << "Depth SDK contract passed\n";
        return 0;
    } catch (const std::exception& failure) {
        std::cerr << "Depth SDK contract failed: " << failure.what() << '\n';
        return 1;
    }
}
