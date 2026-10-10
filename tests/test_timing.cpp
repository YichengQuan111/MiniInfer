#include "miniinfer/thread_pool.hpp"
#include <iostream>
#include <stdexcept>

namespace {
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void check(const miniinfer::TaskTiming& timing) {
    expect(timing.enqueued != miniinfer::TaskTiming::Clock::time_point{}, "missing enqueue stamp");
    expect(timing.enqueued <= timing.started && timing.started <= timing.finished,
           "timing stamps must follow acceptance/execution order");
    expect(timing.finished <= miniinfer::TaskTiming::Clock::now(), "finish stamp lies in the future");
}
}

int main() {
    try {
        miniinfer::BoundedQueue<int> queue(1);
        bool accepted = false;
        queue.push(1, [&]() noexcept { accepted = true; });
        expect(accepted, "acceptance hook must run");
        accepted = false;
        try { queue.push(2, [&]() noexcept { accepted = true; }); }
        catch (const miniinfer::QueueFull&) {}
        expect(!accepted && queue.size() == 1, "rejected task must not invoke acceptance hook");
        queue.close();
        try { queue.push(2, [&]() noexcept { accepted = true; }); }
        catch (const miniinfer::PoolClosed&) {}
        expect(!accepted, "closed queue must not invoke acceptance hook");

        miniinfer::ThreadPool pool(1, 4);
        auto success = std::make_shared<miniinfer::TaskTiming>();
        auto result = pool.submit([](std::size_t) { return 42; }, success);
        expect(result.get() == 42, "timed task result changed");
        check(*success);
        auto failure = std::make_shared<miniinfer::TaskTiming>();
        auto failed = pool.submit([](std::size_t) { throw std::runtime_error("injected"); }, failure);
        bool delivered = false;
        try { failed.get(); } catch (const std::runtime_error&) { delivered = true; }
        expect(delivered, "timed task exception missing");
        check(*failure);
        auto recovered = pool.submit([](std::size_t) {});
        recovered.get();
        pool.shutdown();
        std::cout << "timing: accepted/rejected hooks; success/void exception stamps; recovery passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
