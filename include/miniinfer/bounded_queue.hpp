#pragma once

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace miniinfer {

class QueueFull : public std::runtime_error {
public:
    QueueFull() : std::runtime_error("bounded queue is full") {}
};

class PoolClosed : public std::runtime_error {
public:
    PoolClosed() : std::runtime_error("thread pool is closed") {}
};

template <class T>
class BoundedQueue final {
    static_assert(std::is_nothrow_move_constructible_v<T>,
                  "BoundedQueue values must be nothrow move constructible");
    static_assert(std::is_nothrow_destructible_v<T>,
                  "BoundedQueue values must be nothrow destructible");

public:
    explicit BoundedQueue(std::size_t capacity)
        : slots_(checked_capacity(capacity)), capacity_(capacity) {}

    BoundedQueue(const BoundedQueue&) = delete;
    BoundedQueue& operator=(const BoundedQueue&) = delete;
    BoundedQueue(BoundedQueue&&) = delete;
    BoundedQueue& operator=(BoundedQueue&&) = delete;

    void push(T task) {
        push(std::move(task), []() noexcept {});
    }

    // Runs under the queue lock after acceptance, before any worker can pop.
    // The hook must not throw or call back into this queue.
    template <class OnAccepted>
    void push(T task, OnAccepted&& on_accepted) {
        static_assert(std::is_nothrow_invocable_v<OnAccepted&>,
                      "queue acceptance hook must be noexcept");
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_) {
                throw PoolClosed();
            }
            if (size_ == capacity_) {
                throw QueueFull();
            }

            slots_[tail_].emplace(std::move(task));
            tail_ = next_index(tail_);
            ++size_;
            on_accepted();
        }
        available_.notify_one();
    }

    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        available_.wait(lock, [this] { return size_ != 0 || closed_; });
        if (size_ == 0) {
            return std::nullopt;
        }

        std::optional<T> task(std::move(slots_[head_]));
        slots_[head_].reset();
        head_ = next_index(head_);
        --size_;
        return task;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        available_.notify_all();
    }

    std::size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return size_;
    }

    std::size_t capacity() const noexcept { return capacity_; }

    bool accepting() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return !closed_;
    }

private:
    static std::size_t checked_capacity(std::size_t capacity) {
        if (capacity == 0) {
            throw std::invalid_argument("bounded queue capacity must be greater than zero");
        }
        return capacity;
    }

    std::size_t next_index(std::size_t index) const noexcept {
        return index + 1 == capacity_ ? 0 : index + 1;
    }

    std::vector<std::optional<T>> slots_;
    const std::size_t capacity_;
    mutable std::mutex mutex_;
    std::condition_variable available_;
    std::size_t head_ = 0;
    std::size_t tail_ = 0;
    std::size_t size_ = 0;
    bool closed_ = false;
};

}  // namespace miniinfer
