#include "miniinfer/thread_pool.hpp"

#include <stdexcept>
#include <system_error>
#include <utility>

namespace miniinfer {
namespace {

thread_local const ThreadPool* current_worker_pool = nullptr;

class WorkerPoolScope final {
public:
    explicit WorkerPoolScope(const ThreadPool* pool) noexcept
        : previous_(current_worker_pool) {
        current_worker_pool = pool;
    }

    ~WorkerPoolScope() { current_worker_pool = previous_; }

    WorkerPoolScope(const WorkerPoolScope&) = delete;
    WorkerPoolScope& operator=(const WorkerPoolScope&) = delete;

private:
    const ThreadPool* previous_;
};

}  // namespace

ThreadPool::ThreadPool(std::size_t workers, std::size_t queue_capacity,
                       ThreadFactory thread_factory)
    : queue_(queue_capacity) {
    if (workers == 0) {
        throw std::invalid_argument("thread pool must have at least one worker");
    }
    if (!thread_factory) {
        throw std::invalid_argument("thread pool thread factory must not be empty");
    }

    workers_.reserve(workers);
    try {
        for (std::size_t worker_id = 0; worker_id < workers; ++worker_id) {
            std::thread worker = thread_factory(
                std::function<void()>([this, worker_id] { worker_loop(worker_id); }));
            if (!worker.joinable()) {
                throw std::system_error(
                    std::make_error_code(std::errc::resource_unavailable_try_again),
                    "ThreadPool factory returned a non-joinable thread");
            }
            workers_.emplace_back(std::move(worker));
        }
    } catch (...) {
        queue_.close();
        for (std::thread& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        throw;
    }
}

ThreadPool::~ThreadPool() noexcept { shutdown(); }

void ThreadPool::shutdown() {
    if (current_worker_pool == this) {
        throw std::logic_error("a thread pool worker cannot shut down its own pool");
    }

    std::lock_guard<std::mutex> lock(shutdown_mutex_);
    if (!shutdown_started_) {
        queue_.close();
        shutdown_started_ = true;
    }

    for (std::thread& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void ThreadPool::worker_loop(std::size_t worker_id) noexcept {
    WorkerPoolScope pool_scope(this);
    for (;;) {
        std::optional<std::packaged_task<void(std::size_t)>> task = queue_.pop();
        if (!task) {
            return;
        }
        try {
            (*task)(worker_id);
        } catch (...) {
            // packaged_task stores callable exceptions in its future. Keep the worker alive
            // if invoking a malformed task itself reports an exception.
        }
    }
}

}  // namespace miniinfer
