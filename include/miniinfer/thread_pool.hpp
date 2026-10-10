#pragma once

#include "miniinfer/bounded_queue.hpp"
#include "miniinfer/task_timing.hpp"

#include <cstddef>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace miniinfer {

class ThreadPool final {
public:
    using ThreadFactory = std::function<std::thread(std::function<void()>)>;

    explicit ThreadPool(
        std::size_t workers = 1, std::size_t queue_capacity = 64,
        ThreadFactory thread_factory = [](std::function<void()> worker) {
            return std::thread(std::move(worker));
        });
    ~ThreadPool() noexcept;

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    template <class F>
    auto submit(F&& callable, std::shared_ptr<TaskTiming> timing = {})
        -> std::future<std::invoke_result_t<std::decay_t<F>&, std::size_t>> {
        using Callable = std::decay_t<F>;
        using Result = std::invoke_result_t<Callable&, std::size_t>;

        auto packaged = std::make_shared<std::packaged_task<Result(std::size_t)>>(
            [function = Callable(std::forward<F>(callable)), timing](std::size_t worker_id) mutable -> Result {
                if (timing) timing->started = TaskTiming::Clock::now();
                detail::FinishStamp finish(timing.get());
                return std::invoke(function, worker_id);
            });
        std::future<Result> result = packaged->get_future();
        std::packaged_task<void(std::size_t)> queued(
            [packaged](std::size_t worker_id) { (*packaged)(worker_id); });
        queue_.push(std::move(queued), [&timing]() noexcept {
            if (timing) timing->enqueued = TaskTiming::Clock::now();
        });
        return result;
    }

    void shutdown();

    std::size_t queue_capacity() const noexcept { return queue_.capacity(); }
    std::size_t pending() const { return queue_.size(); }
    std::size_t worker_count() const noexcept { return workers_.size(); }

private:
    void worker_loop(std::size_t worker_id) noexcept;

    BoundedQueue<std::packaged_task<void(std::size_t)>> queue_;
    std::vector<std::thread> workers_;
    std::mutex shutdown_mutex_;
    bool shutdown_started_ = false;
};

}  // namespace miniinfer
