#include "miniinfer/bounded_queue.hpp"
#include "miniinfer/thread_pool.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <exception>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>

namespace {

using miniinfer::BoundedQueue;
using miniinfer::PoolClosed;
using miniinfer::QueueFull;
using miniinfer::ThreadPool;

static_assert(!std::is_copy_constructible_v<ThreadPool>,
              "ThreadPool must not be copy constructible");
static_assert(!std::is_copy_assignable_v<ThreadPool>, "ThreadPool must not be copy assignable");
static_assert(!std::is_move_constructible_v<ThreadPool>, "ThreadPool must not be movable");
static_assert(!std::is_move_assignable_v<ThreadPool>, "ThreadPool must not be move assignable");

void expect(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

template <typename Exception, typename Callable>
void expect_throws(Callable&& callable, std::string_view message) {
    try {
        std::forward<Callable>(callable)();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error(std::string(message));
}

class Gate final {
public:
    Gate() : released_future_(released_.get_future().share()) {}
    ~Gate() { release(); }

    Gate(const Gate&) = delete;
    Gate& operator=(const Gate&) = delete;

    void wait() const { released_future_.wait(); }

    void release() noexcept {
        bool expected = false;
        if (released_flag_.compare_exchange_strong(expected, true)) {
            try {
                released_.set_value();
            } catch (...) {
            }
        }
    }

private:
    std::promise<void> released_;
    std::shared_future<void> released_future_;
    std::atomic<bool> released_flag_{false};
};

class ScopeRelease final {
public:
    explicit ScopeRelease(Gate& gate) noexcept : gate_(gate) {}
    ~ScopeRelease() { gate_.release(); }

    ScopeRelease(const ScopeRelease&) = delete;
    ScopeRelease& operator=(const ScopeRelease&) = delete;

private:
    Gate& gate_;
};

void test_bounded_queue_capacity_fifo_and_close() {
    expect_throws<std::invalid_argument>([] { BoundedQueue<int> invalid(0); },
                                         "zero queue capacity must throw invalid_argument");

    BoundedQueue<int> queue(2);
    expect(queue.capacity() == 2 && queue.size() == 0 && queue.accepting(),
           "queue getters must report initial state");
    queue.push(10);
    queue.push(20);
    expect(queue.size() == 2, "size must count queued values");
    expect_throws<QueueFull>([&queue] { queue.push(30); },
                             "a full queue must reject a push with QueueFull");

    expect(queue.pop() == 10, "queue must pop in FIFO order");
    queue.push(30);
    expect(queue.pop() == 20 && queue.pop() == 30,
           "the ring must preserve FIFO order after wrapping");

    queue.push(40);
    queue.push(50);
    queue.close();
    queue.close();
    expect(!queue.accepting(), "closed queue must stop accepting values");
    expect_throws<PoolClosed>([&queue] { queue.push(60); },
                             "closed state must take priority over full state");
    expect(queue.pop() == 40 && queue.pop() == 50,
           "closing a queue must leave queued values available to drain");
    expect(!queue.pop().has_value(), "closed and drained queue must return nullopt");
}

void test_thread_pool_validation_and_bounded_submission() {
    expect_throws<std::invalid_argument>([] { ThreadPool invalid(0); },
                                         "zero workers must throw invalid_argument");
    expect_throws<std::invalid_argument>([] { ThreadPool invalid(1, 0); },
                                         "zero queue capacity must throw invalid_argument");

    Gate gate;
    std::promise<void> started;
    std::future<void> started_future = started.get_future();
    ThreadPool pool(1, 1);
    ScopeRelease release_gate(gate);

    std::future<std::size_t> active = pool.submit([&gate, &started](std::size_t worker_id) {
        started.set_value();
        gate.wait();
        return worker_id;
    });
    started_future.wait();

    std::future<std::size_t> queued = pool.submit([](std::size_t worker_id) {
        return worker_id + 10;
    });
    expect(pool.pending() == 1, "one waiting task must fill a capacity-one queue");
    expect_throws<QueueFull>([&pool] { (void)pool.submit([](std::size_t) { return 99; }); },
                             "submission into a full pool must throw QueueFull");

    gate.release();
    expect(active.get() == 0, "the active task must receive its stable worker id");
    expect(queued.get() == 10, "the accepted queued task must complete after the gate opens");
    expect(pool.worker_count() == 1 && pool.queue_capacity() == 1 && pool.pending() == 0,
           "pool getters must report configured and current state");
}

void test_future_exceptions_move_only_tasks_and_worker_ids() {
    ThreadPool pool(1, 4);
    expect(pool.submit([](std::size_t worker_id) { return worker_id; }).get() == 0,
           "the first worker must have id zero");
    expect(pool.submit([](std::size_t worker_id) { return worker_id; }).get() == 0,
           "the worker id must remain stable across tasks");

    std::future<int> failed = pool.submit([](std::size_t) -> int {
        throw std::runtime_error("task failure");
    });
    expect_throws<std::runtime_error>([&failed] { (void)failed.get(); },
                                     "task exception must be delivered through its future");
    expect(pool.submit([](std::size_t) { return 42; }).get() == 42,
           "a task exception must not stop the worker");

    auto owned_value = std::make_unique<int>(73);
    std::future<int> moved = pool.submit(
        [value = std::move(owned_value)](std::size_t worker_id) { return *value + static_cast<int>(worker_id); });
    expect(!owned_value && moved.get() == 73,
           "submission must support a move-only callable capture");

    std::future<int> self_shutdown = pool.submit([&pool](std::size_t) {
        pool.shutdown();
        return 1;
    });
    expect_throws<std::logic_error>([&self_shutdown] { (void)self_shutdown.get(); },
                                    "worker self-shutdown logic_error must reach its future");
    expect(pool.submit([](std::size_t) { return 5; }).get() == 5,
           "rejecting worker self-shutdown must leave the pool usable");

    pool.shutdown();
    pool.shutdown();
    expect_throws<PoolClosed>([&pool] { (void)pool.submit([](std::size_t) { return 0; }); },
                              "submitting after shutdown must throw PoolClosed");
}

void test_worker_ids_are_stable_with_two_workers() {
    Gate first_gate;
    Gate second_gate;
    std::promise<std::size_t> first_started;
    std::promise<std::size_t> second_started;
    std::future<std::size_t> first_started_future = first_started.get_future();
    std::future<std::size_t> second_started_future = second_started.get_future();
    ThreadPool pool(2, 2);
    ScopeRelease release_first_gate(first_gate);
    ScopeRelease release_second_gate(second_gate);

    std::future<std::size_t> first = pool.submit(
        [&first_gate, &first_started](std::size_t worker_id) {
            first_started.set_value(worker_id);
            first_gate.wait();
            return worker_id;
        });
    const std::size_t first_id = first_started_future.get();

    std::future<std::size_t> second = pool.submit(
        [&second_gate, &second_started](std::size_t worker_id) {
            second_started.set_value(worker_id);
            second_gate.wait();
            return worker_id;
        });
    const std::size_t second_id = second_started_future.get();
    expect(first_id < 2 && second_id < 2 && first_id != second_id,
           "two blocked tasks must run on distinct workers with distinct ids");

    std::future<std::size_t> follow_up = pool.submit([](std::size_t worker_id) {
        return worker_id;
    });
    first_gate.release();
    expect(follow_up.get() == first_id,
           "the freed worker must keep the same id for its next queued task");
    expect(first.get() == first_id, "the first worker task must complete with its assigned id");
    second_gate.release();
    expect(second.get() == second_id, "the second worker task must complete with its assigned id");
}

void test_shutdown_drains_and_rejects() {
    Gate gate;
    std::promise<void> active_started;
    std::promise<void> active_finished;
    std::promise<void> queued_finished;
    std::future<void> active_started_future = active_started.get_future();
    std::future<void> active_finished_future = active_finished.get_future();
    std::future<void> queued_finished_future = queued_finished.get_future();
    ThreadPool pool(1, 2);
    ScopeRelease release_gate(gate);

    std::future<std::size_t> active = pool.submit(
        [&gate, &active_started, &active_finished](std::size_t worker_id) {
            active_started.set_value();
            gate.wait();
            active_finished.set_value();
            return worker_id;
        });
    active_started_future.wait();
    std::future<std::size_t> queued = pool.submit(
        [&queued_finished](std::size_t worker_id) {
            queued_finished.set_value();
            return worker_id;
        });

    gate.release();
    pool.shutdown();
    expect(active.get() == 0 && queued.get() == 0,
           "shutdown must drain and complete every accepted task");
    active_finished_future.wait();
    queued_finished_future.wait();
    expect_throws<PoolClosed>([&pool] { (void)pool.submit([](std::size_t) { return 1; }); },
                              "shutdown must reject later submissions");
}

void test_destruction_waits_for_accepted_tasks() {
    std::promise<void> task_started;
    std::promise<void> task_finished;
    std::future<void> task_started_future = task_started.get_future();
    std::future<void> task_finished_future = task_finished.get_future();
    std::future<std::size_t> accepted;
    Gate gate;
    {
        ThreadPool pool(1, 1);
        ScopeRelease release_gate(gate);
        accepted = pool.submit([&gate, &task_started, &task_finished](std::size_t worker_id) {
            task_started.set_value();
            gate.wait();
            task_finished.set_value();
            return worker_id;
        });
        task_started_future.wait();
        gate.release();
    }
    expect(accepted.get() == 0, "scope destruction must allow accepted work to finish");
    expect(task_finished_future.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
           "scope destruction must wait until its accepted task finishes");
}

struct TestCase {
    std::string_view name;
    void (*run)();
};

}  // namespace

int main() {
    constexpr TestCase tests[] = {
        {"bounded queue capacity, FIFO, and close", test_bounded_queue_capacity_fifo_and_close},
        {"pool validation and bounded submission", test_thread_pool_validation_and_bounded_submission},
        {"future exceptions, move-only task, and worker IDs",
         test_future_exceptions_move_only_tasks_and_worker_ids},
        {"stable IDs with two workers", test_worker_ids_are_stable_with_two_workers},
        {"shutdown drain and rejection", test_shutdown_drains_and_rejects},
        {"destruction waits for accepted tasks", test_destruction_waits_for_accepted_tasks},
    };

    std::size_t passed = 0;
    for (const TestCase& test : tests) {
        try {
            test.run();
            ++passed;
        } catch (const std::exception& error) {
            std::cerr << "test_thread_pool: " << test.name << ": " << error.what() << '\n';
            return 1;
        } catch (...) {
            std::cerr << "test_thread_pool: " << test.name << ": unknown failure\n";
            return 1;
        }
    }

    std::cout << "test_thread_pool: " << passed << " test groups passed\n";
    return 0;
}
