#include "miniinfer/mlp_input.hpp"
#include "miniinfer/mlp_model.hpp"
#include "miniinfer/mlp_result.hpp"
#include "miniinfer/runtime.hpp"
#include "miniinfer/thread_pool.hpp"

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using miniinfer::MlpInput;
using miniinfer::MlpModel;
using miniinfer::MlpResult;
using miniinfer::PoolClosed;
using miniinfer::QueueFull;
using miniinfer::Runtime;
using miniinfer::RuntimeConfig;
using miniinfer::ThreadPool;

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
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string(message) + ": wrong exception: " + error.what());
    }
    throw std::runtime_error(std::string(message) + ": expected exception was not thrown");
}

template <typename Exception, typename T>
void expect_future_throws(std::future<T>& future, std::string_view message) {
    try {
        (void)future.get();
    } catch (const Exception&) {
        return;
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string(message) + ": wrong future exception: " +
                                 error.what());
    }
    throw std::runtime_error(std::string(message) + ": future completed without an exception");
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

    void release() noexcept { gate_.release(); }

private:
    Gate& gate_;
};

class ThreadGroup final {
public:
    ThreadGroup() = default;
    ~ThreadGroup() { join_no_throw(); }

    ThreadGroup(const ThreadGroup&) = delete;
    ThreadGroup& operator=(const ThreadGroup&) = delete;

    template <typename Callable>
    void spawn(Callable&& callable) {
        threads_.emplace_back(
            [this, task = std::decay_t<Callable>(std::forward<Callable>(callable))]() mutable {
                try {
                    task();
                } catch (...) {
                    std::lock_guard<std::mutex> lock(failure_mutex_);
                    if (!failure_) {
                        failure_ = std::current_exception();
                    }
                }
            });
    }

    void join_and_rethrow() {
        join_all();
        std::exception_ptr failure;
        {
            std::lock_guard<std::mutex> lock(failure_mutex_);
            failure = failure_;
        }
        if (failure) {
            std::rethrow_exception(failure);
        }
    }

private:
    void join_all() {
        for (std::thread& thread : threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

    void join_no_throw() noexcept {
        for (std::thread& thread : threads_) {
            if (thread.joinable()) {
                try {
                    thread.join();
                } catch (...) {
                    try {
                        if (thread.joinable()) {
                            thread.detach();
                        }
                    } catch (...) {
                    }
                }
            }
        }
    }

    std::vector<std::thread> threads_;
    std::mutex failure_mutex_;
    std::exception_ptr failure_;
};

struct FactoryLifecycle {
    std::mutex mutex;
    std::condition_variable changed;
    std::size_t calls = 0;
    std::size_t started = 0;
    std::size_t active = 0;
    std::size_t finished = 0;
};

struct FactoryResource {
    explicit FactoryResource(std::atomic<std::size_t>& destroyed) : destroyed_(destroyed) {}
    ~FactoryResource() { destroyed_.fetch_add(1, std::memory_order_relaxed); }

    std::atomic<std::size_t>& destroyed_;
};

void test_thread_creation_failure_reclaims_started_workers(std::size_t fail_index) {
    auto lifecycle = std::make_shared<FactoryLifecycle>();
    std::atomic<std::size_t> resources_destroyed{0};
    auto resource = std::make_shared<FactoryResource>(resources_destroyed);
    std::weak_ptr<FactoryResource> resource_weak = resource;
    const std::error_code injected_code =
        std::make_error_code(std::errc::resource_unavailable_try_again);

    ThreadPool::ThreadFactory factory =
        [lifecycle, resource = std::move(resource), fail_index](std::function<void()> worker) {
            (void)resource;
            std::size_t call = 0;
            {
                std::lock_guard<std::mutex> lock(lifecycle->mutex);
                call = lifecycle->calls++;
            }
            if (call == fail_index) {
                throw std::system_error(
                    std::make_error_code(std::errc::resource_unavailable_try_again),
                    "injected thread creation failure");
            }

            std::thread started_worker(
                [lifecycle, task = std::move(worker)]() mutable {
                    {
                        std::lock_guard<std::mutex> lock(lifecycle->mutex);
                        ++lifecycle->started;
                        ++lifecycle->active;
                        lifecycle->changed.notify_all();
                    }
                    task();
                    {
                        std::lock_guard<std::mutex> lock(lifecycle->mutex);
                        --lifecycle->active;
                        ++lifecycle->finished;
                        lifecycle->changed.notify_all();
                    }
                });
            {
                std::unique_lock<std::mutex> lock(lifecycle->mutex);
                lifecycle->changed.wait(lock, [lifecycle, call] {
                    return lifecycle->started >= call + 1;
                });
            }
            return started_worker;
        };

    bool saw_injected_failure = false;
    try {
        ThreadPool pool(4, 1, std::move(factory));
        (void)pool;
    } catch (const std::system_error& error) {
        saw_injected_failure = error.code() == injected_code &&
                               std::string(error.what()).find("injected thread creation failure") !=
                                   std::string::npos;
    }

    expect(saw_injected_failure, "the injected system_error must escape ThreadPool construction");
    expect(resource_weak.expired() && resources_destroyed.load(std::memory_order_relaxed) == 1,
           "the failed constructor must release resources captured by its factory");
    {
        std::lock_guard<std::mutex> lock(lifecycle->mutex);
        expect(lifecycle->calls == fail_index + 1,
               "the factory must fail at the requested worker index");
        expect(lifecycle->started == fail_index && lifecycle->finished == fail_index &&
                   lifecycle->active == 0,
               "all workers started before the failure must finish before construction unwinds");
    }
}

void test_nonjoinable_factory_result_is_rejected() {
    std::size_t calls = 0;
    ThreadPool::ThreadFactory factory = [&calls](std::function<void()>) {
        ++calls;
        return std::thread{};
    };

    bool clear_error = false;
    try {
        ThreadPool pool(1, 1, std::move(factory));
        (void)pool;
    } catch (const std::system_error& error) {
        clear_error =
            error.code() == std::make_error_code(std::errc::resource_unavailable_try_again) &&
            std::string(error.what()).find("non-joinable thread") != std::string::npos;
    }
    expect(calls == 1 && clear_error,
           "a factory returning a non-joinable thread must raise a clear system_error");
}

void test_full_queue_close_priority_and_concurrent_shutdown(bool abort_before_closers = false) {
    Gate worker_gate;
    Gate start_gate;
    std::promise<void> worker_started;
    std::future<void> worker_started_future = worker_started.get_future();
    std::atomic<int> active_runs{0};
    std::atomic<int> failed_runs{0};
    std::atomic<int> recovery_runs{0};
    std::atomic<int> probe_runs{0};
    std::atomic<std::size_t> producer_closed{0};
    std::mutex accepted_mutex;
    std::vector<std::future<int>> accepted_probes;
    accepted_probes.reserve(8);
    // Every object referenced by a worker/caller must outlive release and join,
    // including when an assertion or thread construction throws.
    std::function<void()> probe_submit;
    ThreadPool pool(1, 2);
    ThreadGroup callers;
    struct CloseBeforeJoiningCallers {
        ThreadPool& pool;
        ~CloseBeforeJoiningCallers() { pool.shutdown(); }
    } close_on_unwind{pool};
    ScopeRelease release_worker(worker_gate);
    ScopeRelease release_start(start_gate);
    std::future<int> active = pool.submit([&](std::size_t worker_id) {
        active_runs.fetch_add(1, std::memory_order_relaxed);
        worker_started.set_value();
        worker_gate.wait();
        return static_cast<int>(worker_id);
    });
    worker_started_future.wait();
    std::future<int> failed = pool.submit([&](std::size_t) -> int {
        failed_runs.fetch_add(1, std::memory_order_relaxed);
        throw std::runtime_error("expected queued task failure");
    });
    std::future<int> recovery = pool.submit([&](std::size_t) {
        recovery_runs.fetch_add(1, std::memory_order_relaxed);
        return 42;
    });
    expect(pool.pending() == pool.queue_capacity(),
           "the queued tasks must saturate the waiting queue while the worker is gated");
    expect_throws<QueueFull>([&] { (void)pool.submit([](std::size_t) { return 8; }); },
                             "a full open queue must report QueueFull before shutdown starts");

    probe_submit = [&] {
        std::future<int> future = pool.submit([&probe_runs](std::size_t) {
            probe_runs.fetch_add(1, std::memory_order_relaxed);
            return 9;
        });
        std::lock_guard<std::mutex> lock(accepted_mutex);
        accepted_probes.push_back(std::move(future));
    };

    constexpr std::size_t producer_count = 4;
    constexpr std::size_t shutdown_count = 3;
    for (std::size_t producer = 0; producer < producer_count; ++producer) {
        callers.spawn([&, producer] {
            (void)producer;
            start_gate.wait();
            for (;;) {
                try {
                    probe_submit();
                } catch (const QueueFull&) {
                    std::this_thread::yield();
                } catch (const PoolClosed&) {
                    producer_closed.fetch_add(1, std::memory_order_relaxed);
                    return;
                }
            }
        });
    }
    if (abort_before_closers) {
        throw std::runtime_error("injected caller launch failure");
    }
    for (std::size_t closer = 0; closer < shutdown_count; ++closer) {
        callers.spawn([&] {
            start_gate.wait();
            pool.shutdown();
        });
    }

    start_gate.release();
    for (;;) {
        try {
            probe_submit();
        } catch (const QueueFull&) {
            std::this_thread::yield();
        } catch (const PoolClosed&) {
            break;
        }
    }
    expect(pool.pending() == pool.queue_capacity(),
           "PoolClosed must take priority over QueueFull while the waiting queue remains full");
    expect(active_runs.load(std::memory_order_relaxed) == 1 &&
               failed_runs.load(std::memory_order_relaxed) == 0 &&
               recovery_runs.load(std::memory_order_relaxed) == 0,
           "the held worker must keep all queued tasks pending until the gate opens");

    release_worker.release();
    callers.join_and_rethrow();
    expect(producer_closed.load(std::memory_order_relaxed) == producer_count,
           "every racing submitter must observe the closed pool");

    expect(active.get() == 0, "the gated accepted task must complete with its worker id");
    expect_future_throws<std::runtime_error>(failed,
                                             "the queued task exception must reach its future");
    expect(recovery.get() == 42,
           "the worker must execute a later accepted task after an earlier task throws");
    expect(active_runs.load(std::memory_order_relaxed) == 1 &&
               failed_runs.load(std::memory_order_relaxed) == 1 &&
               recovery_runs.load(std::memory_order_relaxed) == 1,
           "each accepted task, including a failed task, must execute exactly once");
    for (std::future<int>& accepted : accepted_probes) {
        expect(accepted.get() == 9, "any accepted probe must finish during shutdown drain");
    }
    expect(probe_runs.load(std::memory_order_relaxed) ==
               static_cast<int>(accepted_probes.size()),
           "each accepted probe must execute exactly once");
    expect(accepted_probes.empty(),
           "no racing submission may be accepted while the worker is gated and queue is full");

    pool.shutdown();
    pool.shutdown();
    expect_throws<PoolClosed>([&] { (void)pool.submit([](std::size_t) { return 0; }); },
                              "all later submissions must be rejected after repeated shutdown");
}

struct RuntimeFuture {
    std::size_t id = 0;
    bool expect_failure = false;
    std::size_t batch = 0;
    std::future<MlpResult> future;
};

std::vector<float> make_runtime_values(std::size_t batch, bool overflow) {
    const float value = overflow ? std::numeric_limits<float>::max() : 0.0F;
    return std::vector<float>(batch * MlpModel::input_features, value);
}

void verify_result(const MlpResult& result, std::size_t expected_batch) {
    expect(result.batch() == expected_batch, "Runtime must return the submitted batch size");
    const auto logits = result.logits();
    expect(logits.rows() == expected_batch && logits.cols() == MlpModel::output_features,
           "Runtime result must own logits with shape [batch,10]");
    expect(result.labels().size() == expected_batch,
           "Runtime result must own one label per input row");
    for (std::size_t row = 0; row < logits.rows(); ++row) {
        for (std::size_t column = 0; column < logits.cols(); ++column) {
            expect(std::isfinite(logits.at(row, column)),
                   "successful Runtime results must retain finite logits");
        }
    }
}

void test_runtime_submit_shutdown_and_future_lifetimes(const std::filesystem::path& directory) {
    std::vector<RuntimeFuture> accepted;
    accepted.reserve(40);
    std::mutex accepted_mutex;
    std::atomic<std::size_t> next_id{0};
    std::atomic<std::size_t> attempts{0};
    std::atomic<std::size_t> full{0};
    std::atomic<std::size_t> closed{0};

    {
        Gate start_gate;
        RuntimeConfig config;
        config.workers = 2;
        config.queue_capacity = 16;
        config.max_batch = 16;
        Runtime runtime(MlpModel::load(directory), config);
        ThreadGroup callers;
        ScopeRelease release_start(start_gate);

        MlpInput success_seed(1, make_runtime_values(1, false));
        std::future<MlpResult> success_future = runtime.submit(success_seed);
        const std::size_t success_id = next_id.fetch_add(1, std::memory_order_relaxed);
        accepted.push_back({success_id, false, 1, std::move(success_future)});
        success_seed = MlpInput(1, make_runtime_values(1, true));

        MlpInput failure_seed(1, make_runtime_values(1, true));
        std::future<MlpResult> failure_future = runtime.submit(failure_seed);
        const std::size_t failure_id = next_id.fetch_add(1, std::memory_order_relaxed);
        accepted.push_back({failure_id, true, 1, std::move(failure_future)});
        failure_seed = MlpInput(1, make_runtime_values(1, false));

        constexpr std::size_t producer_count = 4;
        constexpr std::size_t attempts_per_producer = 8;
        std::vector<std::shared_future<void>> ready;
        ready.reserve(producer_count + 1);
        for (std::size_t producer = 0; producer < producer_count; ++producer) {
            auto ready_promise = std::make_shared<std::promise<void>>();
            ready.push_back(ready_promise->get_future().share());
            callers.spawn([&, producer, ready_promise] {
                ready_promise->set_value();
                start_gate.wait();
                for (std::size_t attempt = 0; attempt < attempts_per_producer; ++attempt) {
                    attempts.fetch_add(1, std::memory_order_relaxed);
                    const std::size_t id = next_id.fetch_add(1, std::memory_order_relaxed);
                    const bool fails = (producer + attempt) % 2 != 0;
                    const std::size_t batch = fails ? 1 : config.max_batch;
                    MlpInput input(batch, make_runtime_values(batch, fails));
                    try {
                        std::future<MlpResult> future = runtime.submit(std::move(input));
                        std::lock_guard<std::mutex> lock(accepted_mutex);
                        accepted.push_back({id, fails, batch, std::move(future)});
                    } catch (const QueueFull&) {
                        full.fetch_add(1, std::memory_order_relaxed);
                    } catch (const PoolClosed&) {
                        closed.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            });
        }

        auto shutdown_ready = std::make_shared<std::promise<void>>();
        ready.push_back(shutdown_ready->get_future().share());
        callers.spawn([&runtime, &start_gate, shutdown_ready] {
            shutdown_ready->set_value();
            start_gate.wait();
            runtime.shutdown();
        });
        for (const std::shared_future<void>& signal : ready) {
            signal.wait();
        }
        start_gate.release();
        callers.join_and_rethrow();

        expect(attempts.load(std::memory_order_relaxed) ==
                   full.load(std::memory_order_relaxed) + closed.load(std::memory_order_relaxed) +
                       (accepted.size() - 2),
               "every concurrent Runtime submission must be accepted or rejected exactly once");
        expect_throws<PoolClosed>(
            [&] { (void)runtime.submit(MlpInput(1, make_runtime_values(1, false))); },
            "Runtime must reject submissions after concurrent shutdown completes");
    }

    std::size_t successes = 0;
    std::size_t failures = 0;
    std::vector<MlpResult> retained_results;
    retained_results.reserve(accepted.size());
    for (RuntimeFuture& item : accepted) {
        if (item.expect_failure) {
            expect_future_throws<std::overflow_error>(
                item.future, "accepted overflowing Runtime task must return its exception");
            ++failures;
        } else {
            MlpResult result = item.future.get();
            verify_result(result, item.batch);
            retained_results.push_back(std::move(result));
            ++successes;
        }
    }
    expect(successes != 0 && failures != 0,
           "accepted Runtime futures must include successful and failed work");
    expect(successes + failures == accepted.size(),
           "every accepted Runtime future must be collected exactly once after Runtime destruction");

    for (std::size_t left = 0; left < retained_results.size(); ++left) {
        verify_result(retained_results[left], retained_results[left].batch());
        for (std::size_t right = left + 1; right < retained_results.size(); ++right) {
            expect(retained_results[left].logits().data() != retained_results[right].logits().data(),
                   "separate Runtime futures must own separate result storage");
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "test_shutdown: expected exactly one reference data directory argument\n";
        return 2;
    }

    try {
        test_thread_creation_failure_reclaims_started_workers(0);
        test_thread_creation_failure_reclaims_started_workers(2);
        test_nonjoinable_factory_result_is_rejected();
        test_full_queue_close_priority_and_concurrent_shutdown();
        try {
            test_full_queue_close_priority_and_concurrent_shutdown(true);
            throw std::runtime_error("missing injected caller launch failure");
        } catch (const std::runtime_error& error) {
            expect(std::string(error.what()) == "injected caller launch failure",
                   "exception unwinding must close pool before joining racing producers");
        }
        test_runtime_submit_shutdown_and_future_lifetimes(std::filesystem::path(argv[1]));
        std::cout << "test_shutdown: thread creation rollback, shutdown races, and Runtime future lifetimes passed\n";
    } catch (const std::exception& error) {
        std::cerr << "test_shutdown: " << error.what() << '\n';
        return 1;
    } catch (...) {
        std::cerr << "test_shutdown: unknown failure\n";
        return 1;
    }
    return 0;
}
