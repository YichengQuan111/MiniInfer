#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <functional>
#include <poll.h>
#include <sys/types.h>

namespace miniinfer::net {
using Clock = std::chrono::steady_clock;
using Deadline = Clock::time_point;
enum class IoStatus { Ok, Eof, Truncated, Timeout, SystemError, Stopped };
struct IoResult {
    IoStatus status;
    std::size_t transferred;
    int error_number = 0;
};
// Empty hooks use native syscalls/steady_clock. Hooks report syscall errors via
// errno; tests may advance now to model EINTR, short transfers and deadlines.
struct IoHooks {
    std::function<ssize_t(int, void*, std::size_t, int)> recv;
    std::function<ssize_t(int, const void*, std::size_t, int)> send;
    std::function<int(pollfd*, nfds_t, int)> poll;
    std::function<Deadline()> now;
};
IoResult recv_exact(int fd, void* data, std::size_t size, Deadline deadline,
                    const std::atomic<bool>* stop = nullptr, IoHooks* hooks = nullptr);
IoResult send_all(int fd, const void* data, std::size_t size, Deadline deadline,
                  const std::atomic<bool>* stop = nullptr, IoHooks* hooks = nullptr);
}  // namespace miniinfer::net
