#pragma once

#include <chrono>

namespace miniinfer {

// Optional per-request instrumentation. Use a fresh record for each submission;
// only read it after future::get()/wait() has observed task completion.
struct TaskTiming {
    using Clock = std::chrono::steady_clock;
    Clock::time_point enqueued{};
    Clock::time_point started{};
    Clock::time_point finished{};
};

namespace detail {
class FinishStamp final {
public:
    explicit FinishStamp(TaskTiming* timing) noexcept : timing_(timing) {}
    ~FinishStamp() { if (timing_) timing_->finished = TaskTiming::Clock::now(); }
    FinishStamp(const FinishStamp&) = delete;
    FinishStamp& operator=(const FinishStamp&) = delete;
private:
    TaskTiming* timing_;
};
}  // namespace detail
}  // namespace miniinfer
